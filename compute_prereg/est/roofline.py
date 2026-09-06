"""Roofline: step time = max(FLOPs / (peak x MFU), bytes / bandwidth), x steps, + overhead."""

from __future__ import annotations

from collections.abc import Callable

from ..models import Band, Money, PerGpuEstimate, PriceSource
from .gpus import GPUS

LAUNCH_OVERHEAD_HOURS = 5.0 / 60.0  # image pull, code sync, first-step warmup


def per_gpu_estimate(
    gpu_name: str,
    *,
    flops_per_step: float,
    bytes_per_step: float,
    total_steps: int,
    dtype: str,
    required_mem_gb: float,
    mfu_band: Band,
    mfu_source: str,
    launch_overhead_hours: float = LAUNCH_OVERHEAD_HOURS,
) -> PerGpuEstimate:
    spec = GPUS[gpu_name]
    notes: list[str] = []
    try:
        peak = spec.peak_flops(dtype)
    except KeyError:
        peak = spec.peak_flops("fp32")
        notes.append(f"{gpu_name} has no {dtype} tensor cores; used fp32 peak")
    if dtype in ("bf16",) and "bf16" not in spec.peak_tflops:
        notes.append(f"{gpu_name} lacks native bf16; fp16 peak used, expect emulation slowdown")

    mem_time = bytes_per_step / (spec.hbm_gbps * 1e9) if spec.hbm_gbps else 0.0

    def step_time(mfu: float) -> tuple[float, str]:
        compute = flops_per_step / (peak * mfu)
        return (compute, "compute") if compute >= mem_time else (mem_time, "memory")

    fast, bound_fast = step_time(mfu_band.hi)
    slow, _ = step_time(mfu_band.lo)
    mid, bound = step_time(mfu_band.mid)
    if bound == "memory":
        notes.append("memory-bandwidth bound at the prior MFU; MFU band is not the limiter")
    hours = Band(lo=fast * total_steps / 3600 + launch_overhead_hours,
                 hi=slow * total_steps / 3600 + launch_overhead_hours)
    price = spec.ref_price_per_hour
    usd = Band(lo=hours.lo * price.usd, hi=hours.hi * price.usd)
    fits = required_mem_gb <= spec.vram_gb
    if not fits:
        notes.append(f"needs {required_mem_gb:.1f} GB > {spec.vram_gb:.0f} GB VRAM; single-GPU does not fit")
    return PerGpuEstimate(
        gpu=gpu_name,
        fits_memory=fits,
        mfu_band=mfu_band,
        mfu_source=mfu_source,
        step_seconds=Band(lo=fast, hi=slow),
        hours=hours,
        usd_at_ref_price=usd,
        ref_price=price,
        bound=bound,
        notes=notes,
    )


def training_bytes_per_step(param_bytes: float, optimizer_state_bytes: float, activation_bytes: float) -> float:
    """HBM traffic per step: read weights (fwd), read weights + write grads (bwd), read+write
    optimizer state and weights (update), activations written then read."""
    return param_bytes * 4 + optimizer_state_bytes * 2 + activation_bytes * 2


MfuLookup = Callable[[str], tuple[Band, str]]


def derived_money(usd: float, note: str | None = None) -> Money:
    return Money(usd=round(usd, 4), source=PriceSource.derived, note=note)
