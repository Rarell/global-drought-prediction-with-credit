 
 
from __future__ import annotations
 
import contextlib
import os
import subprocess
import tempfile
from typing import List, Optional
 
import torch
 
 
class NoFreeGPUError(RuntimeError):
    """Raised when every visible GPU is already in use."""
 
 
def _visible_physical_ids(n_logical: int) -> Optional[List[int]]:
    """Map torch's logical GPU indices to physical NVML indices."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None or cvd.strip() == "":
        return list(range(n_logical))
    parts = [p.strip() for p in cvd.split(",") if p.strip()]
    if all(p.isdigit() for p in parts):
        return [int(p) for p in parts][:n_logical]
    return None  # UUIDs or MIG names: cannot map to an index
 
 
def _query_gpus(n_logical: int) -> List[dict]:
    """One dict per logical GPU: used_mb, total_mb, n_procs (None if unknown), util (None if unknown)."""
    phys = _visible_physical_ids(n_logical)
    if phys is None:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES uses UUIDs/MIG names, which select_free_gpu "
            "cannot map to GPU indices. Use numeric indices."
        )
    try:
        import pynvml
 
        pynvml.nvmlInit()
        try:
            gpus = []
            for pid in phys:
                h = pynvml.nvmlDeviceGetHandleByIndex(pid)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                procs = [
                    p for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h)
                    if p.pid != os.getpid()  # do not count ourselves
                ]
                gpus.append(dict(
                    used_mb=mem.used / 2**20, total_mb=mem.total / 2**20,
                    n_procs=len(procs),
                    util=pynvml.nvmlDeviceGetUtilizationRates(h).gpu,
                ))
            return gpus
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass  # pynvml missing or failing: fall back to nvidia-smi
 
    res = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    rows = {}
    for line in res.stdout.strip().splitlines():
        idx, used, total, util = [v.strip() for v in line.split(",")]
        rows[int(idx)] = dict(used_mb=float(used), total_mb=float(total),
                              n_procs=None, util=float(util))
    return [rows[p] for p in phys]
 
 
def _is_free(g: dict, max_used_mb, min_free_gb, max_util) -> bool:
    if g["n_procs"]:                                   # None or 0 passes
        return False
    if g["used_mb"] > max_used_mb:
        return False
    if min_free_gb is not None and (g["total_mb"] - g["used_mb"]) / 1024 < min_free_gb:
        return False
    if max_util is not None and g["util"] is not None and g["util"] > max_util:
        return False
    return True
 
 
def _describe(i: int, g: dict) -> str:
    procs = "unknown" if g["n_procs"] is None else str(g["n_procs"])
    util = "?" if g["util"] is None else f"{g['util']:.0f}%"
    return (f"  GPU {i}: {g['used_mb'] / 1024:.1f}/{g['total_mb'] / 1024:.1f} GB used, "
            f"{procs} other process(es), utilization {util}")
 
 
@contextlib.contextmanager
def _lock():
    try:
        import fcntl
    except ImportError:  # not on Linux/macOS: no locking
        yield
        return
    path = os.path.join(tempfile.gettempdir(), "select_free_gpu.lock")
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
 
 
def select_free_gpu(
    preferred: Optional[int] = None,
    max_used_mb: float = 1000.0,
    min_free_gb: Optional[float] = None,
    max_util: Optional[float] = None,
    allow_cpu: bool = False,
    verbose: bool = True,
) -> "torch.device":
    """
    Return a torch.device for a free GPU and make it the current device.
 
    Parameters
    ----------
    preferred : int, optional
        Logical GPU index to try first. Default: torch's current device.
    max_used_mb : float
        A GPU with more memory than this in use counts as busy. Default
        1000 MB leaves room for a bare CUDA context.
    min_free_gb : float, optional
        Also require at least this much free memory (e.g. 20).
    max_util : float, optional
        Also treat utilization above this percentage as busy.
    allow_cpu : bool
        If there is no CUDA at all, return the CPU instead of raising.
    verbose : bool
        Print which GPU was chosen and why.
 
    Raises
    ------
    NoFreeGPUError if every visible GPU is busy (or none exists and
    allow_cpu is False).
    """
    if not torch.cuda.is_available():
        if allow_cpu:
            return torch.device("cpu")
        raise NoFreeGPUError("No CUDA device is available to this process.")
 
    n = torch.cuda.device_count()
    first = torch.cuda.current_device() if preferred is None else preferred
    order = [first] + [i for i in range(n) if i != first]
 
    with _lock():
        gpus = _query_gpus(n)
        for i in order:
            if _is_free(gpus[i.index], max_used_mb, min_free_gb, max_util):
                torch.cuda.set_device(i)
                device = torch.device(f"cuda:{i.index}")
                torch.zeros(1, device=device)  # claim: creates our CUDA context
                if verbose:
                    note = "" if i == first else f" (GPU {first} was in use)"
                    print(f"[select_free_gpu] using cuda:{i.index}{note}")
                return device
 
    raise NoFreeGPUError(
        f"All {n} visible GPU(s) are in use; stopping.\n"
        + "\n".join(_describe(i, g) for i, g in enumerate(gpus))
        + "\nWait for one to free up, or loosen max_used_mb / max_util."
    )
 