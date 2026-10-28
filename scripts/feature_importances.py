"""
Script written by Claude 

Feature-importance / sensitivity utilities for a spatiotemporal forecast model
whose predictions have shape (batch, number_of_features, forecast_length,
latitude, longitude).
 
Everything here is written against a *model-agnostic* contract so it works
whether `model` is a bare nn.Module, a `torch.compile`-wrapped module, or a
DDP/FSDP-wrapped module (see train.py lines ~255-291):
 
    y_pred = model(x)                      # x, y_pred: (B, C, L, H, W)
    metric = metric_fn(y_pred, y_true)     # scalar, lower is "better" by default
 
Two complementary methods are provided:
 
1. permutation_importance
   -----------------------
   The classic, model-agnostic approach: shuffle one input channel (variable)
   so its relationship to the target is destroyed while every other
   variable's values -- and that variable's own marginal distribution -- are
   left intact, then re-score the model. The drop in skill is the variable's
   importance. This requires one or more *extra forward passes* per variable,
   so it is the more expensive but more faithful of the two methods (it
   measures "how much does the model's actual skill degrade", not just local
   curvature).
 
2. gradient_sensitivity
   ---------------------
   A much cheaper complementary method: a single backward pass gives
   d(loss)/d(x) for every input simultaneously, so one forward+backward pass
   yields a sensitivity score for *all* variables at once (vs. one+ forward
   pass per variable for permutation importance). This is a *local*
   sensitivity measure, not a measure of actual skill degradation, so it can
   disagree with permutation importance for strongly nonlinear models --
   treat it as a fast screening tool / sanity check, not a replacement.

3. shapely_values
   -----------------------
   Variable-level Shapley values, estimated by permutation sampling (the same
    idea as shap.PermutationExplainer). The "players" are input variables
    (channels), or groups of channels you define (e.g. every pressure level of
    one physical variable). Per-pixel SHAP is not attempted: with 25 channels on
    a 180 x 360 grid there are millions of pixels, and the interesting question
    here is about variables.
    
    For one explained sample e and one background sample b (drawn from the pool
    of samples):
    1. start with the input of b (every variable "absent"),
    2. in a random order, switch each variable over to its value in e,
    3. run the model after every switch; a variable's Shapley estimate is the
        change in the scalar value function when it was switched on.
    Averaging over many random orders and backgrounds gives the (interventional)
    Shapley value. Replacing a variable by the value from another date is the
    same "break the link" move that permutation importance uses.
    
    The scalar value function is, by default, the model output averaged over
    all non-batch dimensions (mode="prediction"), or the per-sample squared
    error (mode="error"). Choose the output with output_channels and region
    (same arguments as in feature_importance.permutation_importance), or pass
    your own value_fn(y_pred, y_true) -> (batch,) tensor.
    
    How to read it
    --------------
    * shap_values[i, g] says how much variable g moved the value function for
    sample i, away from the average over backgrounds. Positive = raised it.
    For every sample, shap_values[i].sum() == f_x[i] - f_background_mean[i]
    (exactly, because every order telescopes).
    * mean_abs[g] (mean over samples of |SHAP|) is the usual global importance.
    It is not the same quantity as permutation importance: permutation
    importance measures loss of skill; SHAP measures contribution to the
    model's output (or, in mode="error", to its error).
    * Like permutation importance, this builds inputs that mix variables from
    different dates, which can be physically inconsistent (temperature and
    geopotential height are tied together). Grouping related channels via
    `groups` reduces this.
    
    Cost
    ----
    Forward passes = n_explain * n_permutations * (2 if antithetic) * (G + 1),
    G being the number of players. Memory: a coalition batch of (G + 1) samples
    sits on the CPU, and only micro_batch_size of them are on the GPU at once.
    Start small (n_explain=16, n_permutations=4) and look at mc_stderr.
 
Both functions are written to bound memory:
  - batches are processed one at a time and only scalar/1-D running
    statistics are accumulated (no per-sample predictions are kept),
  - permutation_importance never materializes more than
    `variables_per_pass` extra copies of a batch at once,
  - everything runs under torch.no_grad() (permutation_importance) or
    frees the graph immediately after use (gradient_sensitivity),
  - an optional `cache_batches=True` lets you pay the dataloader/zarr read
    cost once (recommended only for small validation subsets); the default
    `cache_batches=False` re-streams from the dataloader every repeat, which
    is slower but safe for large validation sets that do not fit in RAM.

Neither function cares how the batches were produced, so the same
dataloader built in train.py (`load_dataloader(conf, valid_dataset, ...)`)
can be passed straight through as `dataloader`.
"""
 
from __future__ import annotations

import os, warnings
import copy
import functools
import time
import math
import yaml
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from argparse import ArgumentParser
 
import numpy as np
 
import torch

from metric_calculations import calculate_acc, calculate_mae, calculate_mse, calculate_rmse
from feature_dataloaders import CachedXY, cache_samples
from temporal_bootstrapping import estimate_decorrelation_timescale, effective_sample_size, block_bootstrap_time_indices
from gpu_checker import select_free_gpu

from credit.distributed import distributed_model_wrapper, setup, get_rank_info

from credit.seed import seed_everything
from credit.parser import credit_main_parser, training_data_check
from credit.forecast import load_forecasts
from credit.models.checkpoint import load_model_state
from credit.postblock import GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer

from credit.datasets import setup_data_loading
from credit.models import load_model
 
# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #
 
def _unwrap_model(model):
    """
    Return the "real" module underneath torch.compile / DDP / FSDP wrapping,
    for the rare cases where importance code needs to inspect the raw module
    (not needed for a plain forward pass, which works identically on any of
    the wrapped forms -- this is here mainly for completeness/debugging).
    """
    m = model
    # torch.compile wraps the module and exposes the original at ._orig_mod
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    # DDP / FSDP expose the wrapped module at .module
    if hasattr(m, "module"):
        m = m.module
    return m
 
 
def _default_metric_fn(y_pred, y_true):
    """Mean squared error, reduced to a python float. Lower is better."""
    diff = y_pred - y_true
    return float(torch.mean(diff * diff).item())


def _bind_kwargs(fn, kwargs, device):
    """
    Return fn with `kwargs` bound as keyword arguments. Tensor values are
    moved to `device` once, here, so e.g. latitude weights built on the CPU
    work with GPU predictions. Other values pass through unchanged.
    """
    if not kwargs:
        return fn
    moved = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in kwargs.items()}
    return functools.partial(fn, **moved)


def _select_output_channels(fn, output_channels, output_names, axis, device):
    """
    Wrap fn so it only sees the chosen output channels of y_pred and y_true.
    `output_channels` may mix integer indices and names (names need
    `output_names`, the full list of output-channel names in channel order).
    """
    if output_channels is None:
        return fn
    idx = []
    for c in output_channels:
        if isinstance(c, str):
            if output_names is None:
                raise ValueError("output_names is required when output_channels contains names.")
            if c not in output_names:
                raise ValueError(f"Unknown output channel {c!r}.")
            idx.append(list(output_names).index(c))
        else:
            idx.append(int(c))
    index = torch.as_tensor(idx, dtype=torch.long, device=device)
 
    def selected(y_pred, y_true, **kw):
        return fn(y_pred.index_select(axis, index), y_true.index_select(axis, index), **kw)
 
    return selected

def region_from_bounds(lat, lon, lat_range, lon_range):
    """
    Turn coordinate bounds into the (lat_idx, lon_idx) index arrays that the
    `region` argument expects.
 
    lat, lon : 1-D coordinate arrays of the grid (any order; latitude may be
        descending). lon may be 0-360 or -180-180.
    lat_range : (south, north), inclusive.
    lon_range : (west, east), inclusive, in either convention. A range that
        crosses the seam (e.g. (340, 20) or (-20, 20)) is handled.
    Indices are returned in grid order.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.mod(np.asarray(lon, dtype=float), 360.0)
    lat_idx = np.where((lat >= min(lat_range)) & (lat <= max(lat_range)))[0]
    w, e = np.mod(lon_range[0], 360.0), np.mod(lon_range[1], 360.0)
    if w <= e:
        lon_idx = np.where((lon >= w) & (lon <= e))[0]
    else:  # crosses the 0/360 seam
        lon_idx = np.where((lon >= w) | (lon <= e))[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError("Region contains no grid points.")
    return lat_idx, lon_idx
 
 
def _select_region(fn, region, spatial_axes, region_kwargs, device):
    """
    Wrap fn so it only sees the grid points in `region` = (lat_idx, lon_idx).
    Keyword arguments named in `region_kwargs` (e.g. "lat_weights") are cut
    to the same region: a tensor axis whose length equals the full grid's
    lat (or lon) length is indexed, an axis of length 1 is left alone.
    """
    if region is None:
        return fn
    lat_axis, lon_axis = spatial_axes
    li = torch.as_tensor(np.asarray(region[0]), dtype=torch.long, device=device)
    lo = torch.as_tensor(np.asarray(region[1]), dtype=torch.long, device=device)
    region_kwargs = tuple(region_kwargs or ())
 
    def _cut(v, H, W):
        if not torch.is_tensor(v):
            return v
        if v.dim() == 1:
            return v.index_select(0, li) if v.shape[0] == H else v
        if v.shape[-2] == H:
            v = v.index_select(-2, li)
        if v.shape[-1] == W:
            v = v.index_select(-1, lo)
        return v
 
    def selected(y_pred, y_true, **kw):
        H, W = y_pred.shape[lat_axis], y_pred.shape[lon_axis]
        kw = {k: (_cut(v, H, W) if k in region_kwargs else v) for k, v in kw.items()}
        yp = y_pred.index_select(lat_axis, li).index_select(lon_axis, lo)
        yt = y_true.index_select(lat_axis, li).index_select(lon_axis, lo)
        return fn(yp, yt, **kw)
 
    return selected
 
 
def _iter_batches(dataloader, cached_batches):
    """Yield batches either from a pre-loaded cache or by re-streaming."""
    if cached_batches is not None:
        for b in cached_batches:
            yield b
    else:
        for b in dataloader:
            yield b
 
 
def _maybe_cache(dataloader, cache_batches, device, max_batches=None):
    """
    If cache_batches is True, pull every batch off the dataloader once, move
    it to `device`, and keep it in a list for reuse across variables/repeats.
    Only do this for validation sets you know are small -- this is the
    "trade RAM for speed" option, off by default.
    """
    if not cache_batches:
        return None
    cached = []
    for i, (x, y) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        cached.append((x.to(device), y.to(device)))
    return cached


def _setup_postblock(post_conf):
    """
    Setup a postblock for adding physics constraints to predictions
    """

    if post_conf["activate"]:
        flags = [False, False, False]
        postblock = []
        if post_conf["global_mass_fixer"]["activate"]:
            if post_conf["global_mass_fixer"]["activate_outside_model"]:
                flag_mass_conserve = True
                opt_mass = GlobalMassFixer(post_conf)
                flags[0] = flag_mass_conserve
                postblock.append(opt_mass)

        if post_conf["global_water_fixer"]["activate"]:
            if post_conf["global_water_fixer"]["activate_outside_model"]:
                flag_water_conserve = True
                opt_water = GlobalWaterFixer(post_conf)
                flags[1] = flag_water_conserve
                postblock.append(opt_water)

        if post_conf["global_energy_fixer"]["activate"]:
            if post_conf["global_energy_fixer"]["activate_outside_model"]:
                flag_energy_conserve = True
                opt_energy = GlobalEnergyFixer(post_conf)
                flags[2] = flag_energy_conserve
                postblock.append(opt_energy)
    
    else:
        flags = None
        postblock = None

    return flags, postblock


def _apply_postblock(y_pred, x, flags, postblock):
    """
    Apply postblock physics constraints to predictions
    """

    # No postblock; return y_pred unchanged
    if postblock is None:
        return y_pred

    flag_mass_conserve = flags[0]; flag_water_conserve = flags[1]; flag_energy_conserve = flags[2]
    if flag_mass_conserve:
        opt_mass = postblock[0]
        x_init = x.clone()
        input_dict = {"y_pred": y_pred, "x": x_init}
        input_dict = opt_mass(input_dict)
        y_pred = input_dict["y_pred"]

    if flag_water_conserve:
        opt_water = postblock[1]
        input_dict = {"y_pred": y_pred, "x": x}
        input_dict = opt_water(input_dict)
        y_pred = input_dict["y_pred"]

    if flag_energy_conserve:
        opt_energy = postblock[2]
        input_dict = {"y_pred": y_pred, "x": x}
        input_dict = opt_energy(input_dict)
        y_pred = input_dict["y_pred"]

    return y_pred


def _write_object(obj, obj_type, outputs = None, region = None, save_path = "./"):
    """
    Write the attrubutes of an object to a .csv file

    Note, it is assumed obj has attribute variable_names
    """

    # Use Pandas to convert the object into a DataFrame that can be easily written
    import pandas as pds 

    # Construct the filename
    outputs_str = "all" if outputs is None else "_".join(map(str, outputs))
    savename = f"{obj_type}_{region}_{outputs_str}"

    # Construct the dictionary
    if isinstance(obj, ShapResult):
        # Note ShapResults uses group_names instead of variable_names and requires netcdf for matrices
        from netCDF4 import Dataset

        # Write the matrix and matrix related variables
        N_samples, N_features = obj.shap_values.shape
        with Dataset(f"{save_path}/{savename}.nc", "w") as nc:
            nc.createDimension("n_samples", size = N_samples)
            nc.createDimension("n_features", size = N_features)

            nc.createVariable("shap_values", obj.shap_values.dtype, ("n_samples", "n_features"))
            nc.createVariable("sampling_error", obj.mc_stderr.dtype, ("n_samples", "n_features"))
            nc.createVariable("original_feature", obj.original_feature.dtype, ("n_samples", "n_features"))
            nc.createVariable("sample_index", obj.sample_index.dtype, ("n_samples", ))
            nc.createVariable("value_function", obj.f_x.dtype, ("n_samples", ))
            nc.createVariable("value_function_bg", obj.f_background_mean.dtype, ("n_samples", ))

            nc.variables["shap_values"][:] = obj.shap_values[:]
            nc.variables["sampling_error"][:] = obj.mc_stderr[:]
            nc.variables["original_feature"][:] = obj.original_feature[:]
            nc.variables["sample_index"][:] = obj.sample_index[:]
            nc.variables["value_function"][:] = obj.f_x[:]
            nc.variables["value_function_bg"][:] = obj.f_background_mean[:]

        obj_dict = {
            "importances": [item[0] for key, item in obj.mean_abs.items()],
            "importances_signed": [item for key, item in obj.mean_signed.items()],
            "importances_low_ci": [item[1] for key, item in obj.mean_abs.items()],
            "importances_high_ci": [item[2] for key, item in obj.mean_abs.items()],
        }
    
        # Construct the data frame
        obj_df = pds.DataFrame(obj_dict, index = obj.group_names)
    else:
        obj_dict = {
            key: item for key, item in obj.__dict__.items() if key != "variable_names"
        }

        # Construct the data frame
        obj_df = pds.DataFrame(obj_dict, index = obj.variable_names)

    # Save the dataframe to a csv
    obj_df.to_csv(f"{save_path}/{savename}.csv", sep = ",")

    print(savename)
    print(obj_df)
    
 
# --------------------------------------------------------------------------- #
# 1. Permutation importance
# --------------------------------------------------------------------------- #
 
@dataclass
class PermutationImportanceResult:
    variable_names: List[str]
    baseline_metric: float
    importance: Dict[str, float] = field(default_factory=dict)       # mean skill drop
    importance_std: Dict[str, float] = field(default_factory=dict)   # across n_repeats
    raw_metric: Dict[str, float] = field(default_factory=dict)       # mean permuted metric
    effective_sample_size: float = 0.0
    elapsed_seconds: float = 0.0
 
 
@torch.no_grad() if torch is not None else (lambda f: f)
def _score_dataloader(model, dataloader, metric_fn, device, autocast=False):
    """
    One full pass over `dataloader`, returning the (weighted-by-batch-size)
    mean metric. Only ever holds one batch's tensors in memory at a time.
    """
    total_metric = 0.0
    total_count = 0
    amp_ctx = (
        torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu")
        if autocast
        else _nullcontext()
    )
    with amp_ctx:
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_pred = model(x)
            m = metric_fn(y_pred, y)
            n = x.shape[0]
            total_metric += m * n
            total_count += n
    return total_metric / max(total_count, 1)
 
 
class _nullcontext:
    def __enter__(self):
        return None
 
    def __exit__(self, *exc):
        return False
 
 
def permutation_importance(
    model,
    dataloader,
    variable_names: Sequence[str],
    metric_fn: Optional[Callable] = None,
    variable_axis: int = 1,
    n_repeats: int = 3,
    variables_per_pass: int = 1,
    cache_batches: bool = False,
    max_cache_batches: Optional[int] = None,
    device: Optional["torch.device"] = None,
    autocast: bool = False,
    higher_is_better: bool = False,
    seed: Optional[int] = None,
    verbose: bool = False,
    micro_batch_size: Optional[int] = None,
    pooled: bool = False,
    n_samples: Optional[int] = None,
    output_channels: Optional[Sequence[Union[int, str]]] = None,
    output_names: Optional[Sequence[str]] = None,
    output_axis: int = 1,
    flags: Optional[list] = None,
    postblock: Optional[list] = None, 
    region: Optional[Tuple[Sequence[int], Sequence[int]]] = None,
    region_kwargs: Sequence[str] = (),
    spatial_axes: Tuple[int, int] = (-2, -1),
    metric_kwargs: Optional[dict] = None,
) -> PermutationImportanceResult:
    """
    Permutation feature importance for a model whose inputs/outputs are
    (batch, number_of_features, forecast_length, latitude, longitude)-shaped
    tensors (or any tensor with the variable/channel axis at `variable_axis`).
 
    Parameters
    ----------
    model : nn.Module (plain, torch.compile-wrapped, or DDP/FSDP-wrapped)
        Already moved to `device` and in eval mode (this function calls
        model.eval() defensively, but does not move the model for you).
    dataloader : iterable of (x, y)
        x : input tensor,  (B, C_in, L, H, W)
        y : target tensor, (B, C_out, L, H, W)  (or whatever shape metric_fn expects)
        Reused once for the baseline pass, and once per (variable, repeat)
        combination unless cache_batches=True.
    variable_names : sequence of str
        Names for each index along `variable_axis` of the *input* tensor.
        len(variable_names) must equal x.shape[variable_axis].
    metric_fn : callable(y_pred, y_true) -> float, optional
        Defaults to mean squared error. Should return a scalar where LOWER
        is better unless higher_is_better=True.
    variable_axis : int, default 1
        Axis of the input tensor indexing input variables/channels.
    n_repeats : int, default 3
        Number of independent random shuffles per variable (reduces the
        variance of the importance estimate from a single unlucky shuffle;
        the raw per-repeat metrics are also returned via importance_std).
    variables_per_pass : int, default 1
        How many *different* variables to permute (independently from each
        other) within the same forward pass. This does NOT reduce the number
        of forward passes needed to cover all variables with clean,
        single-variable attributions -- each variable still needs its own
        pass for a correct estimate -- but it is exposed for callers who
        want a cheaper, coarser "permute this whole group together" estimate
        of a *group's* joint importance (e.g. all pressure-level channels of
        one physical variable at once). Leave at 1 for standard per-variable
        importance.
    cache_batches : bool, default False
        If True, pulls the entire dataloader into memory (on `device`) once
        and reuses it for every variable/repeat, trading RAM for speed. Only
        safe for validation sets that comfortably fit in memory/GPU memory.
        Default False re-streams from the dataloader each time, which is
        slower (re-pays dataloader/zarr I/O cost) but bounded in RAM
        regardless of validation set size.
    max_cache_batches : int, optional
        If cache_batches=True, cap how many batches are cached (e.g. to use
        only a representative subset of a huge validation set).
    device : torch.device, optional
        Defaults to the device of the model's first parameter.
    autocast : bool, default False
        Run forward passes under torch.autocast for speed on GPUs that
        support it.
    higher_is_better : bool, default False
        Set True if metric_fn returns a skill score where higher is better
        (e.g. correlation). Importance is always reported as
        "how much does performance get WORSE when this variable is
        permuted", so the sign is flipped internally when higher_is_better.
    seed : int, optional
        Seed for the permutation RNG, for reproducibility.
    verbose : bool, default False
        Print progress per variable.
    micro_batch_size : int, optional
        Run the forward pass in chunks of this many samples. The shuffle
        still happens across the WHOLE batch (so a large dataloader batch
        keeps the permutation meaningful), but GPU activation memory only
        scales with micro_batch_size. Metric is combined as a sample-
        weighted mean, exact for mean-type metrics such as MSE.
 
    pooled : bool, default False
        If True, ignore the dataloader's batch boundaries: every sample is
        cached ON THE CPU once, and each repeat (a) draws a fresh random
        subset of `n_samples` samples from the whole pool, (b) scores that
        subset unpermuted to get a baseline, and (c) shuffles each variable
        across the ENTIRE subset (so values can come from any date in the
        pool, not just the same batch). The baseline and every variable in
        a repeat use the same subset, so degradation is a paired
        comparison. Only micro_batch_size samples are on the GPU at once.
        cache_batches / max_cache_batches (cap on batches read) apply.
    n_samples : int, optional
        Subset size per repeat when pooled=True. None uses the whole pool
        (a new permutation each repeat, but no subsampling).
 
    Returns
    -------
    PermutationImportanceResult
        .importance[name]      -> mean skill degradation (positive = important)
        .importance_std[name]  -> std of skill degradation across n_repeats
        .raw_metric[name]      -> mean permuted metric value itself
        .baseline_metric       -> metric with no permutation
 
    Notes on cost
    -------------
    Total forward passes ~= len(dataloader) * (1 + n_repeats * n_variables
    / variables_per_pass). For large n_variables this dominates runtime;
    consider gradient_sensitivity() first as a cheap screen, then run
    permutation_importance only on the top-K candidates it flags, with a
    larger n_repeats for those to get a trustworthy ranking.
    """
    if torch is None:
        raise ImportError("permutation_importance requires PyTorch to be installed.")
 
    if metric_fn is None:
        metric_fn = _default_metric_fn
 
    if device is None:
        device = next(model.parameters()).device

    # Setup the metric function
    metric_fn = _select_region(metric_fn, region, spatial_axes, region_kwargs, device)
    metric_fn = _select_output_channels(metric_fn, output_channels, output_names, output_axis, device)
    metric_fn = _bind_kwargs(metric_fn, metric_kwargs, device)
 
    model.eval()
    rng = np.random.default_rng(seed)

    print(torch.cuda.max_memory_allocated()/1e9)

    if pooled:
        return _permutation_importance_pooled(
            model, dataloader, variable_names, metric_fn, variable_axis,
            n_repeats, variables_per_pass, n_samples, max_cache_batches,
            device, autocast, higher_is_better, rng, micro_batch_size, flags, 
            postblock, verbose,
        )
 
    cached_batches = _maybe_cache(dataloader, cache_batches, device, max_cache_batches)
    data_source = cached_batches if cached_batches is not None else dataloader
 
    t0 = time.time()
 
    # --- baseline pass -----------------------------------------------------
    # baseline_metric = _score_dataloader(model, data_source, metric_fn, device, autocast)
    baseline_metric = _score_dataloader(
        model, data_source, metric_fn, device, autocast, micro_batch_size
    )
    if verbose:
        print(f"[permutation_importance] baseline metric = {baseline_metric:.6g}")
 
    n_vars = len(variable_names)
    importance: Dict[str, List[float]] = {name: [] for name in variable_names}
    raw_metric: Dict[str, List[float]] = {name: [] for name in variable_names}
 
    for v_start in range(0, n_vars, variables_per_pass):
        # Variable grouping information
        v_end = min(v_start + variables_per_pass, n_vars)
        group_names = list(variable_names[v_start:v_end])
        group_idx = list(range(v_start, v_end))

        sample_eff = []
        for rep in range(n_repeats):
            print(rep)
            total_metric = 0.0
            total_count = 0
            total_eff = 0
            rep_seed = None if seed is None else seed + 1000 * v_start + rep

            for x, y in _iter_batches(dataloader, cached_batches):
                # Collect inputs and outputs
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                # Obtain decorrelation timescale
                decorr_time = estimate_decorrelation_timescale(x.derach().float().cpu().numpy(), time_axis = 0)
 
                x_perm = x.clone()  # one extra copy of this batch only

                # Collect the INPUT batch size and shuffle an index along that axis
                batch_size = x.shape[0]
                if batch_size > 1:
                    perm_idx = rng.permutation(batch_size)
                else:
                    # cannot break batch-dim association with a single sample;
                    # caller should use a larger batch size for this method
                    perm_idx = np.array([0])
 
                idx_t = torch.as_tensor(perm_idx, device=x.device)

                # Find the variable(s) axis(es) in the inputs, and shuffle them along the batch size axis
                for vi in group_idx:
                    slicer = [slice(None)] * x.dim()
                    slicer[variable_axis] = vi
                    slicer_tuple = tuple(slicer)
                    src_slicer = [slice(None)] * x.dim()
                    src_slicer[variable_axis] = vi
                    src_slicer_tuple = tuple(src_slicer)

                    # shuffle this input variable's values across the batch axis
                    x_perm[slicer_tuple] = x[src_slicer_tuple][idx_t]
 
                mb = micro_batch_size or batch_size
                with torch.no_grad():
                    for i in range(0, batch_size, mb):
                        xc, yc = x_perm[i:i + mb], y[i:i + mb]
                        yc_pred = model(xc)
                        yc_pred = _apply_postblock(yc_pred, xc, flags, postblock)
                        m = metric_fn(yc_pred, yc)
                        total_metric += m * xc.shape[0]
                        total_count += xc.shape[0]

                # Evaluate the model with the shuffled inputs
                with torch.no_grad():
                    y_pred = model(x_perm)
                    y_pred = _apply_postblock(y_pred, x_perm, flags, postblock)
                    m = metric_fn(y_pred, y)
                del x_perm

                time_eff = effective_sample_size(batch_size, decorr_time)
                total_metric += m * batch_size
                total_count += batch_size
                total_eff += time_eff
                sample_eff.append(time_eff)

 
            mean_metric = total_metric / max(total_count, 1)
            degradation = (
                (mean_metric - baseline_metric)
                if not higher_is_better
                else (baseline_metric - mean_metric)
            )
            for name in group_names:
                importance[name].append(degradation)
                raw_metric[name].append(mean_metric)
 
            if verbose:
                print(
                    f"[permutation_importance] {group_names} "
                    f"repeat {rep + 1}/{n_repeats}: metric={mean_metric:.6g} "
                    f"degradation={degradation:.6g}"
                )
 
    result = PermutationImportanceResult(
        variable_names=list(variable_names),
        baseline_metric=baseline_metric,
        importance={k: float(np.mean(v)) for k, v in importance.items()},
        importance_std={k: float(np.std(v)) for k, v in importance.items()},
        raw_metric={k: float(np.mean(v)) for k, v in raw_metric.items()},
        effective_sample_size=np.mean(sample_eff), # Note this is an estimate as it is calculated for each sample, so it may vary in n_repeat
        elapsed_seconds=time.time() - t0,
    )
    return result

def _permutation_importance_pooled(
    model, dataloader, variable_names, metric_fn, variable_axis, n_repeats,
    variables_per_pass, n_samples, max_cache_batches, device, autocast,
    higher_is_better, rng, micro_batch_size, flags, postblock, verbose,
) -> PermutationImportanceResult:
    """Pooled variant: random subset per repeat, shuffle across the subset."""
    from tqdm import tqdm
    t0 = time.time()
 
    # Cache every sample on the CPU once (GPU only ever sees micro-batches).
    xs, ys = [], []
    for i, (x, y) in tqdm(enumerate(dataloader)):
        if max_cache_batches is not None and i >= max_cache_batches:
            break
        xs.append(x.detach().cpu())
        ys.append(y.detach().cpu())
    sizes = np.array([x.shape[0] for x in xs])
    cum = np.concatenate([[0], np.cumsum(sizes)])
    total = int(cum[-1])
    k = total if n_samples is None else min(int(n_samples), total)
    if k < 2:
        raise ValueError("pooled permutation importance needs at least 2 samples.")
    mb = micro_batch_size or k
 
    def gather(chunks, idx):
        out = []
        for g in idx:
            b = int(np.searchsorted(cum, g, side="right") - 1)
            w = int(g - cum[b])
            out.append(chunks[b][w:w + 1])
        return torch.cat(out, 0)
 
    def score(x_sub, y_sub, group_idx=None, perm=None):
        total_m = 0.0
        amp_ctx = (
            torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu")
            if autocast else _nullcontext()
        )
        eff_k = []
        for i in range(0, k, mb):
            xc = x_sub[i:i + mb].to(device)
            if group_idx:
                xc = xc.clone()  # never write into the cached subset
                perm_chunk = torch.as_tensor(perm[i:i + mb])
                for vi in group_idx:
                    sl = tuple([slice(None)] * variable_axis + [vi])
                    xc[sl] = x_sub[sl][perm_chunk].to(device)
            yc = y_sub[i:i + mb].to(device)
            with torch.no_grad(), amp_ctx:
                yc_pred = model(xc)
                yc_pred = _apply_postblock(yc_pred, xc, flags, postblock)
                m = metric_fn(yc_pred, yc)
            total_m += m * xc.shape[0]
            eff_k.append(estimate_decorrelation_timescale(xc.detach().float().cpu().numpy(), time_axis = 0)) # Estimate decorrelation timescale
            del xc, yc
            k_eff = effective_sample_size(k, np.max(eff_k)) # Take the minimum timescale for the effective sample size
        return total_m / k_eff, k_eff # total_m / k
 
    n_vars = len(variable_names)
    importance = {n: [] for n in variable_names}
    raw_metric = {n: [] for n in variable_names}
    baselines = []
 
    k_eff = []
    for rep in tqdm(range(n_repeats)):
        idx = rng.choice(total, size=k, replace=False)   # fresh random subset
        x_sub, y_sub = gather(xs, idx), gather(ys, idx)
        base, k_effective = score(x_sub, y_sub)
        base = base.cpu()
        baselines.append(base)
        k_eff.append(k_effective)
 
        for v_start in range(0, n_vars, variables_per_pass):
            group_idx = list(range(v_start, min(v_start + variables_per_pass, n_vars)))
            perm = rng.permutation(k)                    # across the whole subset
            m = score(x_sub, y_sub, group_idx, perm)[0].cpu()
            deg = (m - base) if not higher_is_better else (base - m)
            for vi in group_idx:
                importance[variable_names[vi]].append(deg)
                raw_metric[variable_names[vi]].append(m)
 
        if verbose:
            print(f"[permutation_importance:pooled] repeat {rep + 1}/{n_repeats} "
                  f"({k}/{total} samples) baseline={base:.6g}")
 
    return PermutationImportanceResult(
        variable_names=list(variable_names),
        baseline_metric=float(np.mean(baselines)),
        importance={k_: float(np.mean(v)) for k_, v in importance.items()},
        importance_std={k_: float(np.std(v)) for k_, v in importance.items()},
        raw_metric={k_: float(np.mean(v)) for k_, v in raw_metric.items()},
        effective_sample_size=np.mean(k_eff), # Note this is an estimate; k_eff is calculated for each sample, so it may vary from sample to sample in n_repeats
        elapsed_seconds=time.time() - t0,
    )
 
 
# --------------------------------------------------------------------------- #
# 2. Gradient-based sensitivity (cheap complementary screen)
# --------------------------------------------------------------------------- #
 
def gradient_sensitivity(
    model,
    dataloader,
    variable_names: Sequence[str],
    loss_fn: Optional[Callable] = None,
    variable_axis: int = 1,
    reduction: str = "rms",
    n_batches: Optional[int] = None,
    device: Optional["torch.device"] = None,
    normalize: bool = True,
    output_channels: Optional[Sequence[Union[int, str]]] = None,
    output_names: Optional[Sequence[str]] = None,
    output_axis: int = 1,
    flags: Optional[list] = None,
    postblock: Optional[list] = None, 
    region: Optional[Tuple[Sequence[int], Sequence[int]]] = None,
    region_kwargs: Sequence[str] = (),
    spatial_axes: Tuple[int, int] = (-2, -1),
    loss_kwargs: Optional[dict] = None,
) -> Dict[str, float]:
    """
    Fast local-sensitivity screen: d(loss)/d(x) in a single backward pass per
    batch gives a sensitivity score for every input variable simultaneously,
    so the whole dataset needs only one forward+backward pass per batch,
    versus one+ forward pass PER VARIABLE for permutation_importance. Use
    this first to narrow down which variables are worth a slower, more
    faithful permutation_importance run.
 
    This measures local curvature of the loss around the actual inputs, NOT
    "how much does the model's skill degrade if this variable is corrupted".
    The two can disagree, especially for strongly nonlinear models or
    variables the model uses in a saturating/thresholded way (small
    gradient almost everywhere, but the model is still highly sensitive to
    large perturbations of that variable) -- treat this as a screen, not a
    replacement for permutation_importance.
 
    Parameters
    ----------
    model : nn.Module
        Any forward-differentiable wrapping (plain / compiled / DDP / FSDP).
    dataloader : iterable of (x, y)
    variable_names : sequence of str
        Names along `variable_axis` of the input tensor.
    loss_fn : callable(y_pred, y_true) -> scalar tensor, optional
        Defaults to mean squared error. Must return a tensor (not a float)
        since it needs to be backpropagated.
    variable_axis : int, default 1
    reduction : {"rms", "mean_abs", "max_abs"}, default "rms"
        How to collapse the per-element gradient magnitude down to one
        number per variable, per batch (then averaged across batches).
    n_batches : int, optional
        Cap the number of batches used (for a quick look on a large
        validation set); default uses the whole dataloader.
    device : torch.device, optional
    normalize : bool, default True
        If True, rescale the returned scores so they sum to 1 (useful for
        comparing relative importance; set False to see raw gradient-magnitude
        units).
 
    Returns
    -------
    dict[str, float]
        Sensitivity score per variable (higher = model output is more
        sensitive to that variable, locally, around the validation inputs).
    """

    if torch is None:
        raise ImportError("gradient_sensitivity requires PyTorch to be installed.")

    # torch.autograd.grad(loss, x) never computes parameter gradients, so the
    # backward hooks that FSDP (resharding) and DDP (reducer bookkeeping) rely
    # on never fire. The first batch appears to work; the second forward then
    # finds the wrapper in a corrupt state (e.g. a conv weight expected as a
    # flat 1-D shard but received at full shape). Use an unwrapped model.
    wrapper_names = {c.__name__ for c in type(model).__mro__}
    if wrapper_names & {"FullyShardedDataParallel", "DistributedDataParallel"}:
        raise RuntimeError(
            "gradient_sensitivity() cannot be used with an FSDP- or DDP-wrapped "
            "model. Load the model unwrapped (predict mode 'none', i.e. "
            "load_model(conf, load_weights=True).to(device)) on a single GPU."
        )
 
    if loss_fn is None:
        def loss_fn(y_pred, y_true):
            diff = y_pred - y_true
            return torch.mean(diff * diff)

    # Setup the loss function
    loss_fn = _select_region(loss_fn, region, spatial_axes, region_kwargs, device)
    loss_fn = _select_output_channels(loss_fn, output_channels, output_names, output_axis, device)
    loss_fn = _bind_kwargs(loss_fn, loss_kwargs, device)
 
    if device is None:
        device = next(model.parameters()).device
 
    model.eval()
 
    n_vars = len(variable_names)
    running_sum = np.zeros(n_vars, dtype=np.float64)
    running_count = 0
 
    for i, (x, y) in enumerate(dataloader):
        if n_batches is not None and i >= n_batches:
            break
 
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x = x.clone().requires_grad_(True)
        print(i, x.size())
 
        y_pred = model(x.float())

        y_pred = _apply_postblock(y_pred, x, flags, postblock)
        loss = loss_fn(y_pred, y)
 
        torch.use_deterministic_algorithms(False)
        grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
        # print(grad.size())
        # grad has the same shape as x: (B, C, L, H, W); collapse every axis
        # except variable_axis.
        reduce_dims = tuple(d for d in range(grad.dim()) if d != variable_axis)
 
        if reduction == "rms":
            per_var = torch.sqrt(torch.mean(grad * grad, dim=reduce_dims))
        elif reduction == "mean_abs":
            per_var = torch.mean(torch.abs(grad), dim=reduce_dims)
        elif reduction == "max_abs":
            per_var = torch.amax(torch.abs(grad), dim=reduce_dims)
        else:
            raise ValueError(f"Unknown reduction: {reduction!r}")
 
        running_sum += per_var.detach().cpu().numpy().astype(np.float64)
        running_count += 1
 
        # free the graph/tensors for this batch explicitly
        del grad, loss, y_pred, x
 
    scores = running_sum / max(running_count, 1)
    if normalize:
        total = scores.sum()
        if total > 0:
            scores = scores / total
 
    return {name: float(s) for name, s in zip(variable_names, scores)}


# --------------------------------------------------------------------------- #
# 3. Shapely values
# --------------------------------------------------------------------------- #

@dataclass
class ShapResult:
    group_names: List[str]
    shap_values: np.ndarray            # (n_explain, G)
    mc_stderr: np.ndarray              # (n_explain, G) sampling error of each value
    original_feature: np.ndarray       # (n_explain, G) original feature
    sample_index: np.ndarray           # (n_explain,) index of each sample in the pool
    f_x: np.ndarray                    # (n_explain,) value function at the sample
    f_background_mean: np.ndarray      # (n_explain,) mean value function over backgrounds
    mean_abs: Dict[str, float] = field(default_factory=dict)
    mean_signed: Dict[str, float] = field(default_factory=dict)
    decorrelation_gap: Optional[int] = None # in pool steps, if a gap was used
    elapsed_seconds: float = 0.0


class _CpuPool:
    """All samples from a dataloader, cached once on the CPU."""
 
    def __init__(self, dataloader, max_cache_batches=None):
        self.xs, self.ys = [], []
        for i, (x, y) in enumerate(dataloader):
            if max_cache_batches is not None and i >= max_cache_batches:
                break
            self.xs.append(x.detach().cpu())
            self.ys.append(y.detach().cpu())
        sizes = np.array([x.shape[0] for x in self.xs])
        self.cum = np.concatenate([[0], np.cumsum(sizes)])
        self.total = int(self.cum[-1])
 
    def get(self, g: int):
        b = int(np.searchsorted(self.cum, g, side="right") - 1)
        w = int(g - self.cum[b])
        return self.xs[b][w:w + 1], self.ys[b][w:w + 1]


def _auto_gap(pool, variable_axis, aggregate="median") -> int:
    """
    Decorrelation time, in pool steps, of the inputs: the e-folding lag of 
    each channel's spatial-mean teimeseries, summarized over channels (median by
    default, because a single slowly varying channel such as soil moisture 
    or SST would otherwise set the gap for everything).
    """

    rows = []
    for x in pool.xs:
        dims = tuple(d for d in range(x.dim()) if d not in (0, variable_axis))
        rows.append(x.float().mean(dim=dims).numpy())
    series = np.concatenate(rows, axis=0) # (T, C)
    max_lag = max(2, min(60, series.shape[0] // 3))
    tau = estimate_decorrelation_timescale(
        series, time_axis=0, max_lag=max_lag, aggregate=aggregate
    )
    return max(1, int(math.ceil(tau)))


def _spaced_choice(rng, n_total, n_pick, spacing) -> np.ndarray:
    """Random indices at least 'spacing' apart (greedy, random order)."""
    blocked = np.zeros(n_total, dtype=bool)
    chosen = []
    for i in rng.permutation(n_total):
        if blocked[i]:
            continue
        chosen.append(i)
        blocked[max(0, i - spacing + 1): i + spacing] = True
        if len(chosen) == n_pick:
            break
    if len(chosen) < n_pick:
        warnings.warn(
            f"Only {len(chosen)} samples fit with explain_spacing={spacing} in a "
            f"pool of {n_total}: explaining those.",
            stacklevel=3,
        )
    return np.array(sorted(chosen))
 
 
def _mean_value(y_pred, y_true):
    return y_pred.reshape(y_pred.shape[0], -1).mean(dim=1)
 
 
def _error_value(y_pred, y_true):
    d = y_pred - y_true
    return (d * d).reshape(d.shape[0], -1).mean(dim=1)


def shapley_values(
    model,
    dataloader,
    variable_names: Sequence[str],
    groups: Optional[Dict[str, Sequence[Union[int, str]]]] = None,
    variable_axis: int = 1,
    n_explain: int = 32,
    n_permutations: int = 8,
    antithetic: bool = True,
    mode: str = "prediction",
    value_fn=None,
    value_kwargs: Optional[dict] = None,
    output_channels: Optional[Sequence[Union[int, str]]] = None,
    output_names: Optional[Sequence[str]] = None,
    output_axis: int = 1,
    region: Optional[Tuple[Sequence[int], Sequence[int]]] = None,
    region_kwargs: Sequence[str] = (),
    spatial_axes: Tuple[int, int] = (-2, -1),
    max_cache_batches: Optional[int] = None,
    micro_batch_size: Optional[int] = None,
    min_gap: Union[int, str, None] = None,
    max_gap: Optional[int] = None,
    explain_spacing: Union[int, str, None] = None,
    flags: Optional[list] = None,
    postblock: Optional[list] = None, 
    device=None,
    autocast: bool = False,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> ShapResult:
    """
    Estimate variable-level Shapley values. See the module docstring.
 
    Parameters
    ----------
    model, dataloader, variable_names, variable_axis
        As in feature_importance.permutation_importance. The dataloader may
        be the CreditXYBatches / CachedXY objects used there; all samples are
        cached on the CPU once and used both as explained samples and as
        backgrounds.
    groups : dict, optional
        {player name: channel indices or channel names}. Each group is
        switched on together. Channels in no group are held at the explained
        sample's values throughout. Default: every channel is its own player.
    n_explain : int
        Number of samples to explain, drawn at random from the pool.
    n_permutations : int
        Random orders per explained sample (each gets its own background).
        With antithetic=True each order is also run reversed, which cuts
        the sampling noise for little extra thought but doubles the cost.
    mode : {"prediction", "error"}
        Value function when value_fn is None: model output averaged over
        everything but the batch axis, or per-sample mean squared error.
        Averaging over channels with different units is rarely meaningful,
        so select one with output_channels (and optionally region).
    value_fn : callable(y_pred, y_true, **value_kwargs) -> (batch,) tensor, optional
        Your own scalar-per-sample value function (e.g. latitude-weighted).
    value_kwargs, output_channels, output_names, output_axis, region,
    region_kwargs, spatial_axes
        Same meaning as in permutation_importance (value_kwargs plays the
        role of metric_kwargs).
    max_cache_batches, micro_batch_size, device, autocast, seed, verbose
        As in permutation_importance.
    min_gap: int or "auto", optional
        Backgrounds must be MORE than this many pool steps away from the 
        explained sample. A background from the day before is almost the 
        same state, so switching variables on changes nothing and the SHAP 
        values shrink toward zero. "auto" uses the inputs' decorrelation 
        time (see _auto_gap). Default None only excludes the sample itself.
    max_gap: int, optional
        Backgrounds must also be within this many steps, so they come from 
        a similar time of year (a July background for a January sample mixes 
        seasons). Together with min_gap this selects an annulus in time.
    explain_spacing: int or "auto", optional
        Choose the explained samples at least this many steps apart, so they 
        are roughly independent and mean_abs/mc_stderr are not overconfident.
        "auto" as for min_gap.

    Pool order is taken as time order with a fixed step: samples from one 
    cache directory built from contiguous forecast slices, or the loader's 
    own order, satisfy this. If your pool mixes several inits per day or is 
    shuffled, gaps in pool steps do not match gaps in time.
    """
    from tqdm import tqdm

    if variable_axis == 0:
        raise ValueError("variable_axis must not be the batch axis (0).")
    if mode not in ("prediction", "error"):
        raise ValueError("mode must be 'prediction' or 'error'.")
    if device is None:
        device = next(model.parameters()).device
    if mode == "prediction" and value_fn is None and output_channels is None:
        warnings.warn(
            "mode='prediction' averages over ALL output channels; pass "
            "output_channels to explain one output variable.",
            stacklevel=2,
        )
 
    model.eval()
    rng = np.random.default_rng(seed)
    t0 = time.time()
 
    # ---- players -----------------------------------------------------------
    names = list(variable_names)
    if groups is None:
        groups = {n: [i] for i, n in enumerate(names)}
    group_names = list(groups.keys())
    idx_lists = []
    for gname in group_names:
        ids = [names.index(c) if isinstance(c, str) else int(c) for c in groups[gname]]
        idx_lists.append(ids)
    idx_tensors = [torch.as_tensor(np.asarray(ids), dtype=torch.long) for ids in idx_lists]
    G = len(group_names)
 
    # ---- value function ----------------------------------------------------
    fn = value_fn if value_fn is not None else (
        _mean_value if mode == "prediction" else _error_value
    )
    fn = _select_region(fn, region, spatial_axes, region_kwargs, device)
    fn = _select_output_channels(fn, output_channels, output_names, output_axis, device)
    fn = _bind_kwargs(fn, value_kwargs, device)
 
    # ---- data --------------------------------------------------------------
    pool = _CpuPool(dataloader, max_cache_batches)
    if pool.total < 2:
        raise ValueError("Shapley values need at least 2 samples (explained + background).")
    gap = None
    if "auto" in (min_gap, explain_spacing):
        gap = _auto_gap(pool, variable_axis)
        if verbose:
            print(f"[shapley_values] decorrelation gap = {gap} steps")
    min_gap_v = gap if min_gap == "auto" else (0 if min_gap is None else int(min_gap))
    spacing_v = gap if explain_spacing == "auto" else (
        None if explain_spacing is None else int(explain_spacing)
    )
    used = [v for v in (min_gap_v, spacing_v) if v]
    gap_used = gap if gap is not None else (max(used) if used else None)
    if spacing_v and spacing_v > 1:
        explain_idx = _spaced_choice(rng, pool.total, n_explain, spacing_v)
    else:
        explain_idx = rng.choice(pool.total, size=min(n_explain, pool.total), replace=False)
    all_idx = np.arange(pool.total)
 
    n_channels = pool.xs[0].shape[variable_axis]
    grouped = set(i for ids in idx_lists for i in ids)
    ungrouped = [c for c in range(n_channels) if c not in grouped]
    ungrouped_t = torch.as_tensor(np.asarray(ungrouped), dtype=torch.long) if ungrouped else None
    mb_default = G + 1
 
    def values_for(Z, y_e):
        """Value function for every row of Z (CPU tensor); returns np array."""
        out = []
        mb = micro_batch_size or mb_default
        amp = (
            torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu")
            if autocast else _nullcontext()
        )
        for i in range(0, Z.shape[0], mb):
            zc = Z[i:i + mb].to(device)
            yc = torch.cat([y_e] * zc.shape[0], 0).to(device)
            with torch.no_grad(), amp:
                yc_pred = model(zc)
                yc_pred = _apply_postblock(yc_pred, zc, flags, postblock)
                v = fn(yc_pred, yc)

                # v = fn(model(zc), yc)
                
            out.append(v.detach().float().cpu().numpy().reshape(-1))
            del zc, yc
        return np.concatenate(out)
 
    def sl(axis_index, rows=None):
        s = [slice(None)] * pool_dim
        s[variable_axis] = axis_index
        if rows is not None:
            s[0] = rows
        return tuple(s)
 
    pool_dim = pool.xs[0].dim()

    all_phi = np.zeros((len(explain_idx), G))
    all_se = np.zeros((len(explain_idx), G))
    f_x = np.zeros(len(explain_idx))
    f_bg = np.zeros(len(explain_idx))
    x_in = np.zeros((len(explain_idx), G))
 
    for n_e, e in enumerate(explain_idx):
        x_e, y_e = pool.get(int(e))
        runs, v0s, vfulls = [], [], []
        dist = np.abs(all_idx - e)
        keep = dist > min_gap_v        # min_gap_v = 0 excludes only e

        if max_gap is not None:
            keep &= dist <= max_gap
        candidates = all_idx[keep]
        if candidates.size == 0:
            raise ValueError(
                f"No background samples for sample {e} with min_gap={min_gap_v}, "
                f"max_gap={max_gap} in a pool of {pool.total}. Shrink the gap or "
                f"use more samples."
            )
        if candidates.size < n_permutations:
            warnings.warn(
                f"Only {candidates.size} candidate backgrounds for sample {e}; "
                "they will repeat across permutations.",
                stacklevel=2,
            )

        for _ in tqdm(range(n_permutations)):
        # for _ in range(n_permutations):
            b = int(rng.choice(candidates))
            # b = int(rng.integers(0, pool.total - 1))
            b = b + 1 if b >= e else b           # any sample except e
            x_b, _ = pool.get(b)
            orders = [rng.permutation(G)]
            if antithetic:
                orders.append(orders[0][::-1].copy())
 
            for perm in orders:
                Z = torch.cat([x_b] * (G + 1), 0)
                if ungrouped_t is not None:      # context channels stay at e
                    Z[sl(ungrouped_t)] = x_e[sl(ungrouped_t)]
                for k, gi in enumerate(perm, start=1):
                    Z[sl(idx_tensors[gi], rows=slice(k, None))] = x_e[sl(idx_tensors[gi])]
                v = values_for(Z, y_e)           # (G + 1,)
                
                phi_run = np.zeros(G)
                phi_run[perm] = v[1:] - v[:-1]
                runs.append(phi_run)
                v0s.append(v[0])
                vfulls.append(v[-1])

        runs = np.array(runs)
        all_phi[n_e] = runs.mean(axis=0)
        all_se[n_e] = runs.std(axis=0, ddof=1) / np.sqrt(len(runs)) if len(runs) > 1 else np.nan
        f_x[n_e] = float(np.mean(vfulls))
        f_bg[n_e] = float(np.mean(v0s))
        x_in[n_e] = x_e.detach().float().cpu().numpy().reshape(x_e.shape[0], x_e.shape[1], -1).mean(axis = (-1, 0))
        if verbose:
            print(f"[shapley_values] explained {n_e + 1}/{len(explain_idx)}")
 
    return ShapResult(
        group_names=group_names,
        shap_values=all_phi,
        mc_stderr=all_se,
        original_feature = x_in,
        sample_index=np.asarray(explain_idx),
        f_x=f_x,
        f_background_mean=f_bg,
        mean_abs={g: float(np.mean(np.abs(all_phi[:, i]))) for i, g in enumerate(group_names)},
        mean_signed={g: float(np.mean(all_phi[:, i])) for i, g in enumerate(group_names)},
        decorrelation_gap=gap_used,
        elapsed_seconds=time.time() - t0,
    )


def shap_global_ci(
        result: ShapResult,
        gap: Optional[int] = None,
        n_resamples: int = 1000,
        ci: float = 0.95,
        seed: Optional[int] = None,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Confidence intervals for mean |SHAP| per variable that respect temporal 
    autocorrelation among the explained samples, using the block bootstrap 
    from temporal_block_bootstrap.py (circular blocks).
    
    The explained samples are put in time order. If consecutive explained 
    samples are closer than the decorrelation gap, they carry overlapping 
    information, so whole blocks of consecutive samples are resampled rather 
    than single ones. Block length (in explained samples) = 
    ceil(gap / median spacing between explained samples), at least 1.
    
    gap : decorrelation gap in pool steps. Default: result.decorrelation_gap.
        If neither is available, samples are treated as independent (block 
        length 1) with a warning.

    Returns {variable: (mean_abs, lower, upper)}.
    """

    order = np.argsort(result.sample_index)
    idx = np.asarray(result.sample_index)[order]
    abs_phi = np.abs(result.shap_values[order])
    n = abs_phi.shape[0]

    gap = gap if gap is not None else result.decorrelation_gap
    if gap is None:
        warnings.warn(
            "No decorrelation gap known: treating explained samples as independent.",
            stacklevel=2,
        )
        block = 1
    else:
        spacing = float(np.median(np.diff(idx))) if n > 1 else 1.0
        block = max(1, int(math.ceil(gap / max(spacing, 1.0))))

    rng = np.random.default_rng(seed)
    resamples, _ = block_bootstrap_time_indices(
        n, n_resamples=n_resamples, block_length=block, method="circular", rng=rng
    )
    boot = abs_phi[resamples].mean(axis=1) # (R, G)
    lo, hi = np.percentile(boot, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2], axis = 0)
    return {
        g:(float(abs_phi[:, i].mean()), float(lo[i]), float(hi[i]))
        for i, g in enumerate(result.group_names)
    }

# --------------------------------------------------------------------------- #
# 4. CREDIT Model loader
# --------------------------------------------------------------------------- #

def load_credit_model(config, backend, dataset_type = 'training', nprocesses = 1, build_cache = False):
    """
    Load a PyTorch model based on the code work in CREDIT's train.py,
    using a CREDIT config file
    """

    data_config = setup_data_loading(config)

    # Config settings
    seed = config["seed"]
    seed_everything(seed)

    # clamp to remove outliers
    if config["data"]["data_clamp"] is None:
        flag_clamp = False
        clamp_min = None
        clamp_max = None
    else:
        flag_clamp = True
        clamp_min = float(config["data"]["data_clamp"][0])
        clamp_max = float(config["data"]["data_clamp"][1])

    local_rank, world_rank, world_size = get_rank_info(config["trainer"]["mode"])
    rank = world_rank

    if config["trainer"]["mode"] in ["fsdp", "ddp"]:
        setup(rank, world_size, config["trainer"]["mode"], backend)

    # infer device id from rank
    device = (
        torch.device(f"cuda:{rank % torch.cuda.device_count()}") if torch.cuda.is_available() else torch.device("cpu")
    )
    # torch.cuda.set_device(rank % torch.cuda.device_count())

    # Check GPU availability before using
    device = select_free_gpu(preferred = device)

    # length of forecast steps
    lead_time_periods = config["data"]["lead_time_periods"]

    # batch and ensemble size
    batch_size = config["predict"].get("batch_size", 1)
    ensemble_size = config["predict"].get("ensemble_size", 1)

    # Load the forecasts we wish to evaluate
    forecasts = load_forecasts(config)
    if len(forecasts) % world_size != 0:
        raise ValueError(
            f'Number of forecast inits ({len(forecasts)}) given by conf["predict"]["duration"] x len(conf["predict"]["start_hours"]) should be divisible by number of processes/GPUs ({world_size})'
        )

    # Build samples
    if build_cache == True:
        # Run in parallel
        if nprocesses > 1:
            import multiprocessing as mp

            # Construct the arguments list
            args = [(
                conf, 
                data_config, 
                forecasts, 
                batch_size, 
                rank,
                world_size, 
                device, 
                nprocesses, 
                subset, 
                lead_time_periods, 
                ensemble_size, 
                flag_clamp, clamp_min, clamp_max,
            ) for subset in np.arange(nprocesses)]

            with mp.Pool(nprocesses) as pool:
                pool.starmap(cache_samples, args)
        else:
            cache_samples(
                conf, 
                data_config,
                forecasts, 
                batch_size, 
                rank, 
                world_size, 
                device, 
                0, 
                0,
                lead_time_periods, 
                ensemble_size, 
                flag_clamp, clamp_min, clamp_max,
            )


    # Load cached batches
    loader = CachedXY(f'{conf["save_loc"]}/cache', batch_size = conf["predict"]["batch_size"])

    # Load the model
    if conf["trainer"]["mode"] == "none":
        m = load_model(conf, load_weights=True).to(device)
    else:
        m = load_model(config)

        # Have to send the module to the correct device first 
        m.to(device)

    # move out of eager-mode
    if config["trainer"].get("compile", False):
        m = torch.compile(m)

    # Wrap in DDP or FSDP module, or none
    if config["trainer"]["mode"] in ["ddp", "fsdp"]:
        model = distributed_model_wrapper(config, m, device)
        # Load model weights (if any), an optimizer, scheduler, and gradient scaler
        model = load_model_state(config, model, device)
    else:
        model = m

    return model, loader, device

if __name__ == "__main__":
    from config import INPUT_VARIABLE_NAMES as variable_names

    # Make parsar
    description = "Perform feature importance test of a CREDIT numerical weather prediction AI model using a specified dataset and configuration."
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        "--config",
        dest="model_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        help="Backend for distribted training.",
        default="nccl",
        choices=["nccl", "gloo", "mpi"],
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        help="Type of dataset to load and perform analysis on",
        default="validation",
        choices=["training", "validation", "test"]
    )
    parser.add_argument(
        "--analysis-type",
        type=str,
        help="Type of feature importance analysis",
        default="gradient_sensitivity",
        choices=["gradient_sensitivity", "permutation", "shapley"]
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        help="Number of repetitions to perform in permutation importance",
        default=100,
    )
    parser.add_argument(
        "--out-channels",
        type=int,
        nargs = "+",
        help="Indices of output channels the feature importance will be determined for",
        default=None,
    )
    parser.add_argument(
        "--region",
        type=str,
        help="Spatial region to focus on (results give feature importances of forecast over the region)",
        default="none",
    )
    parser.add_argument(
        "--n-processors", "-np",
        type=int,
        help="Number of multiprocess processors to use to construct the cached sample",
        default=1,
    )

    # Parse arguments and get the config files
    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("model_config")
    backend = args_dict.pop("backend")

    # Load the configuration and get the relevant variables
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    # handling config args
    conf = credit_main_parser(conf, parse_training=True, parse_predict=False, print_summary=False)
    training_data_check(conf, print_summary=False)

    if os.path.exists(f'{conf["save_loc"]}/cache'):
        # Determine if the sample cache needs to be built
        build_cache = False
    else:
        build_cache = True

    model, dataloader, device = load_credit_model(
        conf, 
        backend, 
        dataset_type = args.dataset_type, 
        nprocesses = args.n_processors, 
        build_cache = build_cache,
    )

    metric_kwargs = None

    # Load latitude weights if needed
    if conf["loss"]["use_latitude_weights"]:
        from netCDF4 import Dataset

        with Dataset(conf["loss"]["latitude_weights"], 'r') as nc:
            w_lat = nc.variables["coslat"][:]
            w_lat = w_lat / np.nanmean(w_lat)

            # Subset latitude weights if needed
            if args.region != "none":
                from utils import subset_data

                lat = nc.variables["latitude"][:]
                lon = nc.variables["longitude"][:]

                w_lat, _, _ = subset_data(w_lat, lat, lon, args.region)

            w_lat = torch.from_numpy(w_lat)[:, None]

        metric_kwargs = {"latitude_weights": w_lat}

    if args.analysis_type == "shapley":
        # For shapely values, metrics are summed/averaged over the all axes except the batch (first) axis
        if isinstance(metric_kwargs, dict):
            metric_kwargs.update({"axis": (-4, -3, -2, -1)})
        else:
            metric_kwargs = {"axis": (-4, -3, -2, -1)}

    # Setup region indices if needed
    if args.region != "none":
        from utils import subsets

        if args.region in subsets:
            lat_min, lat_max, lon_min, lon_max = subsets[args.region]
        else:
            valid_regions = [val_reg for val_reg in subsets.keys()]
            raise(f"region must be one of {valid_regions}, but got {args.region} instead.")

        # Setup region indices
        region = region_from_bounds(lat, lon, (lat_min, lat_max), (lon_min, lon_max))

    else:
        region = None

    flags, postblock = _setup_postblock(conf["model"]["post_conf"])

    # Determine loss/metric function
    if "explainable_loss" not in conf["predict"]:
        # Set the default loss to mse
        conf["predict"]["explainable_loss"] = "mse"

    if conf["predict"]["explainable_loss"] == "acc":
        metric_fn = calculate_acc
        higher_is_better = True
    elif conf["predict"]["explainable_loss"] == "mae":
        metric_fn = calculate_mae
        higher_is_better = False
    elif conf["predict"]["explainable_loss"] == "mse":
        metric_fn = calculate_mse
        higher_is_better = False
    elif conf["predict"]["explainable_loss"] == "rmse":
        metric_fn = calculate_rmse
        higher_is_better = False

    if args.analysis_type == "gradient_sensitivity":
        importances = gradient_sensitivity(
            model,
            dataloader,
            variable_names,
            reduction = "rms",
            device = device,
            normalize = False,
            output_channels = args.out_channels,
            flags = flags,
            postblock = postblock, 
            region = region,
            loss_fn = metric_fn,
            loss_kwargs = metric_kwargs,
        )
        for key, item in importances.keys():
            print(key, item)

    elif args.analysis_type == "permutation":
        
        importances = permutation_importance(
            model,
            dataloader,
            variable_names,
            n_repeats = args.n_repeats,
            variables_per_pass = 1,
            cache_batches = False,
            max_cache_batches = None,
            device = device,
            autocast = False,
            seed = conf["seed"],
            pooled = True,
            n_samples = conf["predict"]["batch_size"],
            higher_is_better = higher_is_better,
            output_channels = args.out_channels,
            flags = flags,
            postblock = postblock, 
            region = region,
            metric_fn = metric_fn,
            metric_kwargs = metric_kwargs,
            verbose = True,
        )

        # Write out the results
        _write_object(
            importances,
            "feature_importances",
            outputs = args.out_channels,
            region = args.region,
            save_path = conf["save_loc"],
        )

    elif args.analysis_type == 'shapley':
        
        importances = shapley_values(
            model,
            dataloader,
            variable_names,
            n_explain = args.n_repeats, # conf["predict"]["batch_size"],
            n_permutations = 10, # int(np.sqrt(args.n_repeats)),
            mode = "prediction",
            value_fn = metric_fn,
            value_kwargs = metric_kwargs,
            output_channels = args.out_channels,
            region = region,
            max_cache_batches = None,
            flags = flags,
            postblock = postblock, 
            min_gap = 10, # Min gap of 10 days to match permutations
            max_gap = 60, # Max spacing of 2 months to avoid seasonal overlap (note only forecast_len = 1 is explained)
            explain_spacing = "auto",
            device = device,
            autocast = False,
            seed = conf["seed"],
            verbose = True,
        )

        importances.mean_abs = shap_global_ci(
            importances,
            n_resamples = 5000,
            ci = 0.95,
            seed = conf["seed"],
        )

        print(importances)

        # Write out the results
        _write_object(
            importances,
            "shapely_values",
            outputs = args.out_channels,
            region = args.region,
            save_path = conf["save_loc"],
        )


        # outputs_str = "all" if args.out_channels is None else "_".join(map(str, args.out_channels))
        # savename = f"feature_importances_{args.region}_{outputs_str}.csv"
        # print(savename)

        # importances_dict = {
        #     key: item for key, item in importances.__dict__.items() if key != "variable_names"
        # }

        # importances_df = pds.DataFrame(importances_dict, index = importances.variable_names)
        # importances_df.to_csv(f"{conf['save_loc']}/{savename}", sep = ",")

        # print(importances_df)


    torch.distributed.destroy_process_group()

    # Create directories if they do not exist and copy yml file
    # save_loc = os.path.expandvars(conf["save_loc"])
    # os.makedirs(save_loc, exist_ok=True)

    
 
 
# --------------------------------------------------------------------------- #
# Notes on supplementary tooling
# --------------------------------------------------------------------------- #
#
# If you want attribution methods beyond these two, the `captum` library
# (https://captum.ai/) implements a number of additional options that plug
# into the same model-agnostic forward-pass contract used above, e.g.:
#   - captum.attr.FeatureAblation      : a more general ablation framework
#                                         than simple permutation (can zero,
#                                         mean-fill, or use a custom baseline
#                                         instead of shuffling).
#   - captum.attr.IntegratedGradients  : path-integrated gradients from a
#                                         baseline input to the real input;
#                                         addresses some of the "zero
#                                         gradient but still sensitive"
#                                         blind spots of plain gradient
#                                         sensitivity, at the cost of several
#                                         extra forward+backward passes
#                                         (controlled by its `n_steps`).
# Both are drop-in compatible with a model that takes a single tensor input
# like the one used here, so they are not reimplemented in this file, but
# gradient_sensitivity() above gives you the cheapest version of the same
# idea without the extra dependency.