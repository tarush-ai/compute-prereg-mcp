"""Peak-memory model: params + grads + optimizer state (exact) + activations (estimated,
scaled by a margin because FakeTensor cannot see allocator behaviour) + a fixed reserve."""

from __future__ import annotations

from ..models import MemoryEstimate
from .analytic import dtype_bytes, optimizer_bytes_per_param

GB = 1024**3

# FakeTensor/MemTracker miss allocator fragmentation, workspace buffers and transient
# intermediates. The arXiv:2602.17817 study reports a systematic underestimate; 1.3x on
# activations plus a fixed reserve is the default correction.
ACTIVATION_MARGIN_FACTOR = 1.3
DEFAULT_RESERVE_GB = 4.0


def estimate_memory(
    params: int,
    dtype: str,
    optimizer: str,
    mixed_precision: bool,
    activation_bytes: float,
    *,
    method: str,
    reserve_gb: float = DEFAULT_RESERVE_GB,
    activation_margin: float = ACTIVATION_MARGIN_FACTOR,
    tracked: dict[str, float] | None = None,
) -> MemoryEstimate:
    """``tracked`` may carry MemTracker categories in bytes ("param", "grad", "opt",
    "act", "temp", "other"); when present they override the analytic terms."""
    bp = dtype_bytes(dtype)
    param_b = params * bp
    grad_b = params * bp
    opt_b = params * optimizer_bytes_per_param(optimizer, mixed_precision, dtype)
    act_b = activation_bytes
    temp_b = 0.0
    if tracked:
        param_b = tracked.get("param", param_b) or param_b
        grad_b = tracked.get("grad", grad_b) or grad_b
        # MemTracker only sees optimizer state after a step; keep the analytic figure
        # when it reports nothing.
        opt_b = tracked.get("opt") or opt_b
        act_b = max(act_b, tracked.get("act", 0.0) + tracked.get("other", 0.0))
        temp_b = tracked.get("temp", 0.0)
    act_with_margin = act_b * activation_margin
    point = (param_b + grad_b + opt_b + act_with_margin + temp_b) / GB
    breakdown = {
        "params": param_b / GB,
        "grads": grad_b / GB,
        "optimizer": opt_b / GB,
        "activations": act_b / GB,
        "activation_margin": (act_with_margin - act_b) / GB,
        "temp": temp_b / GB,
        "reserve": reserve_gb,
    }
    return MemoryEstimate(
        point_gb=round(point, 3),
        margin_gb=reserve_gb,
        method=f"{method}+margin",
        breakdown_gb={k: round(v, 3) for k, v in breakdown.items()},
    )
