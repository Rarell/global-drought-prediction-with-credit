"""
Shapley (SHAP) values for the input variables of a gridded forecast model.
 
What is computed
----------------
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
"""
 
from __future__ import annotations
 
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union
 
import numpy as np
import torch
 
from feature_importance import (
    _bind_kwargs,
    _nullcontext,
    _select_output_channels,
    _select_region,
)
 
 
@dataclass
class ShapResult:
    group_names: List[str]
    shap_values: np.ndarray            # (n_explain, G)
    mc_stderr: np.ndarray              # (n_explain, G) sampling error of each value
    sample_index: np.ndarray           # (n_explain,) index of each sample in the pool
    f_x: np.ndarray                    # (n_explain,) value function at the sample
    f_background_mean: np.ndarray      # (n_explain,) mean value function over backgrounds
    mean_abs: Dict[str, float] = field(default_factory=dict)
    mean_signed: Dict[str, float] = field(default_factory=dict)
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
    """
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
    explain_idx = rng.choice(pool.total, size=min(n_explain, pool.total), replace=False)
 
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
                v = fn(model(zc), yc)
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
 
    for n_e, e in enumerate(explain_idx):
        x_e, y_e = pool.get(int(e))
        runs, v0s, vfulls = [], [], []
 
        for _ in range(n_permutations):
            b = int(rng.integers(0, pool.total - 1))
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
        if verbose:
            print(f"[shapley_values] explained {n_e + 1}/{len(explain_idx)}")
 
    return ShapResult(
        group_names=group_names,
        shap_values=all_phi,
        mc_stderr=all_se,
        sample_index=np.asarray(explain_idx),
        f_x=f_x,
        f_background_mean=f_bg,
        mean_abs={g: float(np.mean(np.abs(all_phi[:, i]))) for i, g in enumerate(group_names)},
        mean_signed={g: float(np.mean(all_phi[:, i])) for i, g in enumerate(group_names)},
        elapsed_seconds=time.time() - t0,
    )