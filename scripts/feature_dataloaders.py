from __future__ import annotations
 
import glob
import os
import time
from typing import Optional, Tuple, Iterable
 
import numpy as np

import torch

from credit.data import concat_and_reshape, reshape_only
from credit.transforms import load_transforms
from credit.forecast import load_forecasts

from credit.datasets.load_dataset_and_dataloader import BatchForecastLenDataLoader
from credit.datasets.era5_multistep_batcher import Predict_Dataset_Batcher

def _atomic_save(path: str, arr: np.ndarray) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)  # a sample file is either complete or absent

def iter_credit_xy_batches(
    data_loader,
    device,
    ensemble_size: int = 1,
    flag_clamp: bool = False,
    clamp_min: Optional[float] = None,
    clamp_max: Optional[float] = None,
    only_first_step: bool = True,
) -> Iterator[Tuple["torch.Tensor", "torch.Tensor"]]:
    """
    Generator yielding (x, y) tensor pairs from a BatchForecastLenDataLoader,
    using the identical construction logic as predict() in rollout_metrics.py
    (lines 256-291), minus the model call, the autoregressive next-step
    reassembly, the postblocks, and the metrics/CSV bookkeeping -- just the
    part that turns a raw `batch` dict into (x, y).
 
    Parameters
    ----------
    data_loader : BatchForecastLenDataLoader
        Exactly what you already build with
        `BatchForecastLenDataLoader(Predict_Dataset_Batcher(...))`.
    device : torch.device
    ensemble_size : int, default 1
        Same meaning as in rollout_metrics.py; set >1 only if you want
        importance/sensitivity computed on the repeated-ensemble x.
    flag_clamp, clamp_min, clamp_max : same meaning as in rollout_metrics.py.
    only_first_step : bool, default True
        If True (recommended -- see module docstring), only yields batches
        where `batch["forecast_step"].item() == 1`, i.e. x built entirely
        from real data. Other batches from later steps of the same
        forecasts are skipped, but iteration continues so later forecasts'
        step-1 batches are still reached.
 
    Notes
    -----
    This does NOT handle "y_diag" variables being length-mismatched with x,
    "x_surf"-less configs, or anything else your script's `batch` dict
    doesn't already provide -- it assumes `batch` has the same keys your
    rollout script already relies on (datetime, forecast_step, x, and
    optionally x_surf, x_forcing_static, y, y_surf, y_diag).
    """
    for batch in data_loader:
        forecast_step = batch["forecast_step"].item()
 
        if only_first_step and forecast_step != 1:
            continue
 
        if "x_surf" in batch:
            x = concat_and_reshape(batch["x"], batch["x_surf"]).to(device).float()
        else:
            x = reshape_only(batch["x"]).to(device).float()
 
        if ensemble_size > 1:
            x = torch.repeat_interleave(x, ensemble_size, 0)
 
        if "x_forcing_static" in batch:
            x_forcing_batch = (
                batch["x_forcing_static"].to(device).permute(0, 2, 1, 3, 4).float()
            )
            if ensemble_size > 1:
                x_forcing_batch = torch.repeat_interleave(
                    x_forcing_batch, ensemble_size, 0
                )
            x = torch.cat((x, x_forcing_batch), dim=1)
 
        if "y_surf" in batch:
            y = concat_and_reshape(batch["y"], batch["y_surf"]).to(device).float()
        else:
            y = reshape_only(batch["y"]).to(device).float()
 
        if "y_diag" in batch:
            y_diag_batch = batch["y_diag"].to(device).permute(0, 2, 1, 3, 4)
            y = torch.cat((y, y_diag_batch), dim=1).to(device).float()
 
        if flag_clamp:
            x = torch.clamp(x, min=clamp_min, max=clamp_max)
 
        yield x, y

def build_sample_cache(
    xy_batches,
    cache_dir: str,
    tag: str = "",
    max_samples: Optional[int] = None,
    dtype: str = "float32",
    log_every: int = 10,
    verbose: bool = True,
) -> int:
    """
    Stream (x, y) batches from `xy_batches` and write each sample to
    cache_dir as {tag}{index:06d}_x.npy / _y.npy. Only one batch is ever in
    memory. Returns the number of samples written.
 
    Running several processes in parallel (e.g. with your script's
    subset / no_subset options so each handles a slice of the forecast
    list) is safe as long as each uses a different `tag`, e.g. "s00_",
    "s01_"; CachedXY reads them all back in sorted order.
    """
    os.makedirs(cache_dir, exist_ok=True)
    n = 0
    t0 = time.time()
    for x, y in xy_batches:
        xn = x.detach().cpu().numpy().astype(dtype, copy=False)
        yn = y.detach().cpu().numpy().astype(dtype, copy=False)
        for j in range(xn.shape[0]):
            stem = os.path.join(cache_dir, f"{tag}{n:06d}")
            _atomic_save(stem + "_x.npy", xn[j])
            _atomic_save(stem + "_y.npy", yn[j])
            n += 1
            if verbose and n % log_every == 0:
                rate = (time.time() - t0) / n
                print(f"[build_sample_cache] {n} samples, {rate:.1f} s/sample")
            if max_samples is not None and n >= max_samples:
                return n
    if verbose:
        print(f"[build_sample_cache] done: {n} samples in {time.time() - t0:.0f} s")
    return n


def cache_samples(
        conf, 
        data_config, 
        forecasts, 
        batch_size, 
        rank, 
        world_size, 
        device, 
        number_of_subsets, 
        subset, 
        lead_time_periods, 
        ensemble_size, 
        flag_clamp, clamp_min, clamp_max
):
    """
    Build the cached samples
    """

    # Determine the forecasts based on the subset
    if number_of_subsets > 0 and subset >= 0:
        subsets = np.array_split(forecasts, number_of_subsets)
        forecasts = subsets[subset - 1]  # Select the subset based on subset_size
        # conf["predict"]["forecasts"] = forecasts
        # forecasts = load_forecasts(conf)
    print(conf["predict"]["forecasts"])
    # print(forecasts)

    # Load the dataset using the provided dataset_type
    # if dataset_type == 'training':
    #     dataset = load_dataset(config, rank=rank, world_size=world_size, is_train=True)
    # elif dataset_type == 'validation':
    #     dataset = load_dataset(config, rank=rank, world_size=world_size, is_train=False)
    # elif dataset_type == 'test':
    #     dataset = load_dataset(config, rank=rank, world_size=world_size, is_train=False, is_test = True)
    # Build the dataset
    dataset = Predict_Dataset_Batcher(
        varname_upper_air=data_config["varname_upper_air"],
        varname_surface=data_config["varname_surface"],
        varname_dyn_forcing=data_config["varname_dyn_forcing"],
        varname_forcing=data_config["varname_forcing"],
        varname_static=data_config["varname_static"],
        varname_diagnostic=data_config["varname_diagnostic"],
        filenames=data_config["all_ERA_files"],
        filename_surface=data_config["surface_files"],
        filename_dyn_forcing=data_config["dyn_forcing_files"],
        filename_forcing=data_config["forcing_files"],
        filename_static=data_config["static_files"],
        filename_diagnostic=data_config["diagnostic_files"],
        fcst_datetime=forecasts,
        lead_time_periods=lead_time_periods,
        history_len=data_config["history_len"],
        skip_periods=data_config["skip_periods"],
        transform=load_transforms(conf),
        sst_forcing=data_config["sst_forcing"],
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
    )

    # Load the dataloader
    # loader = load_dataloader(config, dataset, rank=rank, world_size=world_size, is_train=False)
    # Use a custom DataLoader so we get the len correct
    loader = BatchForecastLenDataLoader(dataset)

    # Construct an iterable loader that builds the input and output tensors
    loader = CreditXYBatches(
            loader, 
            device,
            ensemble_size = ensemble_size,
            flag_clamp = flag_clamp, clamp_min = clamp_min, clamp_max = clamp_max,
        )

    # Build the data cache to speed up future permutation importances
    build_sample_cache(
        loader,
        f'{conf["save_loc"]}/cache',
        f"{subset:02d}_",
        dtype = "float32",
        log_every = 10,
        verbose = True,
    )

 
class CreditXYBatches:
    """
    Re-iterable wrapper around iter_credit_xy_batches so it can be passed
    directly as the `dataloader` argument to permutation_importance() /
    gradient_sensitivity() -- both of those call `for x, y in dataloader`
    MORE THAN ONCE (once for the baseline pass, then again per variable per
    repeat), so a plain generator object would be exhausted after the
    first pass. This class's __iter__ calls iter_credit_xy_batches() fresh
    every time, exactly like iterating a normal torch DataLoader twice
    re-walks its Dataset from the start (which is what
    BatchForecastLenDataLoader itself already supports, since it's built
    the same way you already use it in your rollout loop).
 
    Example
    -------
        data_loader = BatchForecastLenDataLoader(dataset)   # as you already build it
        xy = CreditXYBatches(
            data_loader, device,
            ensemble_size=ensemble_size,
            flag_clamp=flag_clamp, clamp_min=clamp_min, clamp_max=clamp_max,
        )
 
        result = permutation_importance(
            model, xy, variable_names,
            n_repeats=3, cache_batches=True,   # see caution below
            device=device,
        )
 
    A note on cache_batches=True
    -----------------------------
    With only_first_step=True (the default), `xy` only yields ONE batch per
    forecast initialization (not one per rollout step), so the full set is
    usually small enough to cache in memory -- this is exactly the case
    cache_batches=True in permutation_importance was designed for: it saves
    you from re-running the whole zarr/dataset read pipeline once per
    variable per repeat, which otherwise dominates runtime far more than
    the model's own forward pass does. If you instead set
    only_first_step=False (full rollout, for the propagation-style
    experiment described above), reconsider cache_batches -- that set could
    be much larger.
    """
 
    def __init__(
        self,
        data_loader,
        device,
        ensemble_size: int = 1,
        flag_clamp: bool = False,
        clamp_min: Optional[float] = None,
        clamp_max: Optional[float] = None,
        only_first_step: bool = True,
    ):
        self.data_loader = data_loader
        self.device = device
        self.ensemble_size = ensemble_size
        self.flag_clamp = flag_clamp
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.only_first_step = only_first_step
 
    def __iter__(self):
        return iter_credit_xy_batches(
            self.data_loader,
            self.device,
            ensemble_size=self.ensemble_size,
            flag_clamp=self.flag_clamp,
            clamp_min=self.clamp_min,
            clamp_max=self.clamp_max,
            only_first_step=self.only_first_step,
        )

 
class CachedXY:
    """
    Re-iterable drop-in for a dataloader that yields (x, y) torch tensors
    of shape (batch, ...) from a cache written by build_sample_cache.
    """
 
    def __init__(self, cache_dir: str, batch_size: int = 1, pattern: str = "*"):
        self.x_files = sorted(glob.glob(os.path.join(cache_dir, pattern + "_x.npy")))
        self.y_files = [f[: -len("_x.npy")] + "_y.npy" for f in self.x_files]
        if not self.x_files:
            raise FileNotFoundError(f"No cached samples found in {cache_dir!r}")
        self.batch_size = batch_size
 
    @property
    def num_samples(self) -> int:
        return len(self.x_files)
 
    def __len__(self) -> int:
        return -(-len(self.x_files) // self.batch_size)
 
    def __iter__(self):
 
        bs = self.batch_size
        for i in range(0, len(self.x_files), bs):
            xs = np.stack([np.load(f) for f in self.x_files[i:i + bs]])
            ys = np.stack([np.load(f) for f in self.y_files[i:i + bs]])
            yield torch.from_numpy(xs), torch.from_numpy(ys)
 