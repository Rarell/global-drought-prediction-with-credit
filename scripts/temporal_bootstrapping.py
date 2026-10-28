"""
Script written by Claude

Temporal resampling utilities for autocorrelated, short-sample,
overlapping-forecast-window datasets of shape
 
    (time, number_of_features, forecast_length, latitude, longitude)
 
A naive i.i.d. bootstrap along the time axis is invalid here for three
compounding reasons (all three were named by the user driving this):
 
  (a) temporal autocorrelation       -- consecutive time steps are not
                                         independent, so resampling single
                                         time indices with replacement
                                         understates the true variance of
                                         any statistic computed across time.
  (b) a small temporal sample size   -- with few independent time steps to
                                         begin with, the problem in (a) is
                                         worse: there may not even be enough
                                         *effectively independent* samples
                                         to support a stable bootstrap
                                         distribution at all.
  (c) overlapping forecast windows   -- each sample already spans
                                         `forecast_length` steps into the
                                         future, so samples that are even
                                         just a few indices apart in `time`
                                         share most of their forecast
                                         window's valid period with their
                                         neighbors. This adds a *second*,
                                         structural source of dependence on
                                         top of (a), and it sets a hard
                                         floor on how short a bootstrap
                                         block may usefully be.
 
The standard fix for (a)/(b) is a **block bootstrap**: resample contiguous
*blocks* of time indices (not single indices) so that the short-range
dependence structure within each block is preserved. (c) is handled by
choosing the block length to be at least as long as the forecast window, so
no block boundary can fall inside a single sample's own forecast horizon,
and by estimating the data's own decorrelation timescale so the block length
is not just "long enough for the forecast window" but long enough to also
capture the data's autocorrelation.
 
Three bootstrap flavors are implemented (choose via `method`):
  - "moving"    : fixed-length blocks, overlapping start points (Kunsch 1989,
                  the classic moving block bootstrap). Simple, but truncates
                  at the series edges and can slightly under-represent the
                  tails of the series.
  - "circular"  : like "moving" but wraps around the end of the series
                  (Politis & Romano 1992), so every time index has an equal
                  chance of starting a block -- recommended default for
                  SHORT series, since it has no edge-truncation bias.
  - "stationary": block lengths are drawn from a Geometric distribution with
                  mean `block_length` (Politis & Romano 1994) instead of a
                  single fixed length, which smooths over the sensitivity of
                  the other two methods to one arbitrary block-length choice,
                  at the cost of a little extra variance from the random
                  block lengths themselves.
 
Design choice carried over from this conversation's other large-array work:
these functions return **index arrays**, not materialized data, so a
resample of a huge zarr-backed array never has to exist all at once in RAM.
`apply_block_bootstrap` is provided to materialize exactly one resample at a
time via fancy indexing, meant to be called inside your own per-resample
loop.
"""
 
from __future__ import annotations
 
from dataclasses import dataclass
from typing import Optional, Union
 
import numpy as np
 
 
# --------------------------------------------------------------------------- #
# Decorrelation timescale / effective sample size
# --------------------------------------------------------------------------- #
 
def _acf_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Sample autocorrelation function of a 1-D series, lags 0..max_lag,
    ignoring NaNs pairwise at each lag (so a few missing values don't
    poison the whole estimate). acf[0] == 1.0 by construction.
    """
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    max_lag = min(max_lag, T - 1)
    mu = np.nanmean(x)
    xc = x - mu
    var0 = np.nanmean(xc * xc)
    acf = np.empty(max_lag + 1, dtype=np.float64)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        a = xc[:-lag]
        b = xc[lag:]
        valid = ~(np.isnan(a) | np.isnan(b))
        if not np.any(valid) or var0 == 0:
            acf[lag] = 0.0
        else:
            acf[lag] = np.mean(a[valid] * b[valid]) / var0
    return acf
 
 
def estimate_decorrelation_timescale(
    x: np.ndarray,
    time_axis: int = 0,
    max_lag: int = 60,
    threshold: float = 1.0 / np.e,
    aggregate: str = "mean",
) -> float:
    """
    Estimate the e-folding decorrelation timescale of a (possibly
    multi-dimensional) time series: the smallest lag at which the
    autocorrelation function (ACF) drops below `threshold` (default 1/e,
    the standard e-folding convention).
 
    This is meant to directly inform block-length selection for the
    bootstrap below: a block shorter than the decorrelation timescale will
    still cut real short-range dependence, however long the forecast
    window is.
 
    Parameters
    ----------
    x : ndarray
        Any shape; the ACF is computed along `time_axis` independently for
        every other index, then aggregated (see `aggregate`). For a
        (time, features, forecast_length, lat, lon)-shaped array this means
        computing one ACF per (feature, forecast_length, lat, lon) point
        and summarizing across all of them -- which can be slow for a huge
        spatial grid, so consider subsampling space first (e.g. pass in
        `x[:, var_idx, 0, ::stride, ::stride]`) if you just need a quick,
        representative estimate rather than an exact one.
    time_axis : int, default 0
    max_lag : int, default 60
        Largest lag to search; raise this if your series is long and you
        suspect slower decorrelation than this allows for.
    threshold : float, default 1/e
        ACF threshold defining "decorrelated".
    aggregate : {"mean", "median", "max"}, default "mean"
        How to combine the per-series timescales found at every non-time
        index into a single number. "max" is the conservative choice (use
        the slowest-decorrelating series found, so no block is too short
        anywhere); "mean"/"median" are more typical single-number summaries.
 
    Returns
    -------
    float
        Estimated decorrelation timescale, in units of time steps. Returns
        max_lag (with a NaN-safe fallback of 1.0 if the series is too short
        or degenerate) if the ACF never drops below `threshold` within
        max_lag -- in that case, re-run with a larger max_lag.
    """
    x = np.moveaxis(np.asarray(x), time_axis, 0)
    T = x.shape[0]
    flat = x.reshape(T, -1)
 
    timescales = []
    # Subsample columns if there are a huge number of them -- a few hundred
    # representative series are enough to estimate a timescale; looping
    # over every single grid point is wasted work.
    n_cols = flat.shape[1]
    if n_cols > 500:
        col_idx = np.linspace(0, n_cols - 1, 500).astype(int)
    else:
        col_idx = np.arange(n_cols)
 
    for c in col_idx:
        series = flat[:, c]
        if np.all(np.isnan(series)):
            continue
        acf = _acf_1d(series, max_lag)
        below = np.where(acf < threshold)[0]
        if below.size > 0:
            timescales.append(float(below[0]))
        else:
            timescales.append(float(max_lag))
 
    if not timescales:
        return 1.0
 
    if aggregate == "mean":
        return float(np.mean(timescales))
    elif aggregate == "median":
        return float(np.median(timescales))
    elif aggregate == "max":
        return float(np.max(timescales))
    else:
        raise ValueError(f"Unknown aggregate: {aggregate!r}")
 
 
def effective_sample_size(T: int, decorrelation_timescale: float) -> float:
    """
    A simple, standard diagnostic for "how many effectively independent
    samples do I actually have": T / decorrelation_timescale. Use this to
    sanity-check whether your validation/sample size is as small as it
    looks once autocorrelation is accounted for -- directly addressing the
    "small temporal sample size" concern. An effective sample size well
    under ~20-30 is a sign that any bootstrap-based uncertainty estimate
    (block or otherwise) should be treated as approximate, and that
    widening the time window, if at all possible, would help more than any
    amount of resampling cleverness.
    """
    if decorrelation_timescale <= 0:
        return float(T)
    return float(T) / float(decorrelation_timescale)
 
 
# --------------------------------------------------------------------------- #
# Block bootstrap index generation
# --------------------------------------------------------------------------- #
 
@dataclass
class BlockBootstrapDiagnostics:
    block_length_used: float
    n_blocks_per_resample: int
    decorrelation_timescale: Optional[float] = None
    effective_sample_size: Optional[float] = None
    warnings: tuple = ()
 
 
def block_bootstrap_time_indices(
    T: int,
    n_resamples: int = 1000,
    block_length: Optional[int] = None,
    forecast_length: Optional[int] = None,
    method: str = "circular",
    rng: Optional[np.random.Generator] = None,
    auto_block_length_from: Optional[np.ndarray] = None,
    time_axis: int = 0,
) -> "tuple[np.ndarray, BlockBootstrapDiagnostics]":
    """
    Generate block-bootstrap resampled TIME INDICES for a series of length T
    (indices only -- see apply_block_bootstrap to materialize data).
 
    Parameters
    ----------
    T : int
        Length of the time axis being resampled.
    n_resamples : int, default 1000
        How many independent resamples to generate.
    block_length : int, optional
        Length of each contiguous block (for "stationary", this is the MEAN
        block length; actual block lengths are geometric-distributed around
        it). If not given, falls back to `forecast_length` if that is
        given, else to `auto_block_length_from`'s estimated decorrelation
        timescale (via estimate_decorrelation_timescale), else to
        max(1, T // 10) as a last-resort heuristic.
    forecast_length : int, optional
        Length, in time steps, of each sample's forecast window. If
        `block_length` ends up shorter than this, a block boundary could
        fall inside a single forecast window, double-counting or
        splitting it across two different bootstrap blocks -- a warning is
        attached to the returned diagnostics in that case, and block_length
        is NOT silently overridden (the caller decides whether to accept
        that trade-off, e.g. because the series is too short to afford a
        longer block).
    method : {"circular", "moving", "stationary"}, default "circular"
        See module docstring for the differences. "circular" is recommended
        for short series (no edge-truncation bias); "stationary" is
        recommended when you are unsure block_length is exactly right and
        want to smooth over that choice.
    rng : np.random.Generator, optional
    auto_block_length_from : ndarray, optional
        If `block_length` is not given and this is provided, block_length is
        set from this array's own estimated decorrelation timescale (passed
        through estimate_decorrelation_timescale with `time_axis`).
    time_axis : int, default 0
        Only used if `auto_block_length_from` is given.
 
    Returns
    -------
    indices : ndarray, shape (n_resamples, T)
        indices[r] is one resample's sequence of original time indices
        (length T, i.e. each resample is the same length as the original
        series, built by concatenating blocks until T indices are reached
        and then truncating to exactly T).
    diagnostics : BlockBootstrapDiagnostics
        Records the block length actually used and any warnings (e.g.
        block_length < forecast_length).
    """
    if rng is None:
        rng = np.random.default_rng()
 
    warnings = []
    decorrelation_timescale = None
 
    if block_length is None:
        if forecast_length is not None:
            block_length = forecast_length
        elif auto_block_length_from is not None:
            decorrelation_timescale = estimate_decorrelation_timescale(
                auto_block_length_from, time_axis=time_axis
            )
            block_length = max(1, int(round(decorrelation_timescale)))
        else:
            block_length = max(1, T // 10)
            warnings.append(
                "block_length not given and no forecast_length or "
                "auto_block_length_from provided; falling back to a "
                "heuristic of T // 10. Consider passing one of these for a "
                "more principled choice."
            )
 
    block_length = int(max(1, block_length))
 
    if forecast_length is not None and block_length < forecast_length:
        warnings.append(
            f"block_length ({block_length}) is shorter than forecast_length "
            f"({forecast_length}); a block boundary can fall inside a "
            f"single sample's forecast window, which reintroduces some of "
            f"the dependence this bootstrap is meant to control for. "
            f"Consider raising block_length to at least forecast_length."
        )
 
    if block_length >= T:
        warnings.append(
            f"block_length ({block_length}) >= series length T ({T}); every "
            f"resample will be (close to) the single block [0, T), so this "
            f"bootstrap will produce almost no variability across "
            f"resamples. The series may be too short for this block length."
        )
 
    n_blocks_per_resample = int(np.ceil(T / block_length))
 
    if method == "moving":
        max_start = T - block_length  # inclusive; blocks cannot wrap
        if max_start < 0:
            max_start = 0
        starts = rng.integers(0, max_start + 1, size=(n_resamples, n_blocks_per_resample))
        indices = _starts_to_indices(starts, block_length, T, wrap=False)
 
    elif method == "circular":
        starts = rng.integers(0, T, size=(n_resamples, n_blocks_per_resample))
        indices = _starts_to_indices(starts, block_length, T, wrap=True)
 
    elif method == "stationary":
        # Geometric-distributed block lengths with mean `block_length`
        # (Politis & Romano 1994 stationary bootstrap), blocks wrap like
        # the circular bootstrap to stay valid at the series edges.
        p = 1.0 / block_length  # success prob so that mean geometric length = block_length
        indices = np.empty((n_resamples, T), dtype=np.int64)
        for r in range(n_resamples):
            pos = 0
            cur = int(rng.integers(0, T))
            row = np.empty(T, dtype=np.int64)
            while pos < T:
                length = rng.geometric(p)
                length = min(length, T - pos)
                block = (cur + np.arange(length)) % T
                row[pos:pos + length] = block
                pos += length
                cur = int(rng.integers(0, T))
            indices[r] = row
 
    else:
        raise ValueError(f"Unknown method: {method!r}")
 
    diagnostics = BlockBootstrapDiagnostics(
        block_length_used=block_length,
        n_blocks_per_resample=n_blocks_per_resample,
        decorrelation_timescale=decorrelation_timescale,
        effective_sample_size=(
            effective_sample_size(T, decorrelation_timescale)
            if decorrelation_timescale is not None
            else None
        ),
        warnings=tuple(warnings),
    )
    return indices, diagnostics
 
 
def _starts_to_indices(starts: np.ndarray, block_length: int, T: int, wrap: bool) -> np.ndarray:
    """
    Expand an (n_resamples, n_blocks) array of block start positions into an
    (n_resamples, n_blocks * block_length) array of time indices, then
    truncate each row to exactly T indices.
    """
    n_resamples, n_blocks = starts.shape
    offsets = np.arange(block_length)
    # shape: (n_resamples, n_blocks, block_length)
    full = starts[:, :, None] + offsets[None, None, :]
    if wrap:
        full = full % T
    else:
        full = np.clip(full, 0, T - 1)
    full = full.reshape(n_resamples, n_blocks * block_length)
    return full[:, :T].astype(np.int64)
 
 
def apply_block_bootstrap(
    data,
    resample_indices: np.ndarray,
    time_axis: int = 0,
    resample_index: int = 0,
):
    """
    Materialize exactly ONE resample from `data` (a numpy array, a dask
    array, or a zarr array -- anything that supports fancy indexing along
    one axis) using the index sequence `resample_indices[resample_index]`
    produced by block_bootstrap_time_indices.
 
    This is intentionally a thin wrapper, not a batch-materializer: call it
    once per resample inside your own loop (e.g. your Monte Carlo /
    uncertainty loop) so that only one resample's worth of data is ever
    resident in memory at a time, regardless of how large `n_resamples` is
    or how large the underlying zarr-backed array is.
 
    Parameters
    ----------
    data : array-like, shape (..., T, ...) with T at `time_axis`
        For the use case this module was written for, shape is
        (time, number_of_features, forecast_length, latitude, longitude)
        with time_axis=0, but any axis position is supported.
    resample_indices : ndarray, shape (n_resamples, T)
        Output of block_bootstrap_time_indices.
    time_axis : int, default 0
    resample_index : int, default 0
        Which row of resample_indices to materialize.
 
    Returns
    -------
    array-like
        `data` indexed along `time_axis` by
        resample_indices[resample_index], same type as `data` where
        possible (numpy in, numpy out; a zarr/dask array will typically
        return a numpy array or a lazy dask array respectively, following
        that library's own fancy-indexing semantics).
    """
    idx = resample_indices[resample_index]
    return np.take(data, idx, axis=time_axis)