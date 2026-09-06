"""Prior MFU bands, used before any measurement exists for a (gpu, model class) key.

Numbers are deliberately wide: dense transformers on H100 land around 35-55% MFU in
published training runs; memory-bound or small-batch workloads well below that.
"""

from __future__ import annotations

from ..models import Band

# (gpu_class, model_class) -> (p10, p90). "*" is the fallback in either position.
_PRIORS: dict[tuple[str, str], tuple[float, float]] = {
    ("H100", "decoder-transformer"): (0.35, 0.55),
    ("H100", "encoder-transformer"): (0.35, 0.55),
    ("H200", "decoder-transformer"): (0.35, 0.55),
    ("H200", "encoder-transformer"): (0.35, 0.55),
    ("B200", "decoder-transformer"): (0.28, 0.48),
    ("A100", "decoder-transformer"): (0.40, 0.60),
    ("A100", "encoder-transformer"): (0.40, 0.60),
    ("L40S", "decoder-transformer"): (0.28, 0.48),
    ("L4", "decoder-transformer"): (0.22, 0.40),
    ("A10G", "decoder-transformer"): (0.25, 0.45),
    ("RTX4090", "decoder-transformer"): (0.25, 0.45),
    ("RTX6000Ada", "decoder-transformer"): (0.25, 0.45),
    ("A6000", "decoder-transformer"): (0.25, 0.45),
    ("V100", "decoder-transformer"): (0.30, 0.50),
    ("T4", "decoder-transformer"): (0.20, 0.35),
    ("MI300X", "decoder-transformer"): (0.22, 0.42),
    ("*", "decoder-transformer"): (0.30, 0.50),
    ("*", "encoder-transformer"): (0.30, 0.50),
    ("*", "cnn"): (0.20, 0.40),
    ("*", "mlp"): (0.15, 0.35),
    ("*", "diffusion"): (0.20, 0.40),
    ("*", "*"): (0.20, 0.45),
}

SMALL_BATCH_TOKENS = 512  # below this, kernels do not saturate the GPU
SMALL_BATCH_SCALE = 0.7


def prior_band(gpu_class: str, model_class: str, batch_tokens: int | None = None) -> Band:
    for key in [(gpu_class, model_class), ("*", model_class), (gpu_class, "*"), ("*", "*")]:
        if key in _PRIORS:
            lo, hi = _PRIORS[key]
            break
    else:  # pragma: no cover
        lo, hi = _PRIORS[("*", "*")]
    if batch_tokens is not None and 0 < batch_tokens < SMALL_BATCH_TOKENS:
        lo, hi = lo * SMALL_BATCH_SCALE, hi * SMALL_BATCH_SCALE
    return Band(lo=lo, hi=hi)


def batch_tokens_bucket(batch_tokens: int) -> str:
    """Power-of-two bucket label so calibration keys group similar batch shapes."""
    if batch_tokens <= 0:
        return "unknown"
    p = 1
    while p * 2 <= batch_tokens:
        p *= 2
    return f"{p}-{p * 2 - 1}"
