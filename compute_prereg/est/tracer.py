"""FakeTensor + FlopCounterMode tracing of a user's model.

Entrypoint contract: the Python file exposes

    def prereg_build(config: dict) -> dict:
        return {
            "model": model,                # torch.nn.Module
            "inputs": x | (args,) | {...}, # example batch (shape matters, values do not)
            "loss": lambda out: ...,       # optional; enables backward FLOPs
            # optional metadata
            "model_class": "decoder-transformer", "total_steps": 20000,
            "batch_tokens": 1024, "optimizer": "adamw", "dtype": "bf16",
            "seq_len": 1024, "hidden_size": 1024, "num_layers": 24, "heads": 16,
        }

The module is imported with __name__ != "__main__", so a training script guarded by the
usual ``if __name__ == "__main__":`` idiom does not start training on import.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import EstimationError, UnsupportedTraceError


@dataclass
class TraceResult:
    flops_forward: float
    flops_backward: float
    per_module: dict[str, float]
    params: int
    param_bytes: float
    dtype: str
    tracked_memory: dict[str, float]  # bytes by MemTracker category
    activation_bytes: float
    model_class: str
    batch_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    has_backward: bool = False


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def sha256_json(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(s.encode()).hexdigest()


def load_entrypoint(path: str | Path):
    p = Path(path).resolve()
    if not p.exists():
        raise EstimationError(f"entrypoint not found: {p}")
    mod_name = f"_prereg_entry_{hashlib.sha1(str(p).encode()).hexdigest()[:10]}"
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise EstimationError(f"cannot import {p}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    if str(p.parent) not in sys.path:
        sys.path.insert(0, str(p.parent))
    spec.loader.exec_module(module)
    if not hasattr(module, "prereg_build"):
        raise EstimationError(
            f"{p.name} does not define prereg_build(config) -> dict; see est/tracer.py"
        )
    return module


def _infer_model_class(model) -> str:
    import torch.nn as nn

    names = [type(m).__name__.lower() for m in model.modules()]
    joined = " ".join(names)
    has_attn = any(
        isinstance(m, nn.MultiheadAttention) for m in model.modules()
    ) or "attention" in joined or "attn" in joined
    has_conv = any(isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) for m in model.modules())
    if has_attn:
        if "encoder" in joined and "decoder" not in joined and "causal" not in joined:
            return "encoder-transformer"
        return "decoder-transformer"
    if has_conv:
        return "cnn"
    return "mlp"


def _call(model, inputs):
    if isinstance(inputs, dict):
        return model(**inputs)
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)


def _count_tokens(inputs) -> int:
    import torch

    def first_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, dict):
            for v in x.values():
                t = first_tensor(v)
                if t is not None:
                    return t
        if isinstance(x, (tuple, list)):
            for v in x:
                t = first_tensor(v)
                if t is not None:
                    return t
        return None

    t = first_tensor(inputs)
    if t is None:
        return 0
    if t.dim() >= 2:
        return int(t.shape[0] * t.shape[1])
    return int(t.shape[0])


def trace_build(build: dict[str, Any], config: dict[str, Any] | None = None) -> TraceResult:
    """Run one forward (+backward when a loss is given) under FakeTensorMode and count."""
    try:
        import torch
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError as e:  # pragma: no cover
        raise EstimationError("torch is required for tracing; install compute-prereg-mcp[torch]") from e

    model = build["model"]
    inputs = build["inputs"]
    loss_fn = build.get("loss")
    warns: list[str] = []

    params = sum(p.numel() for p in model.parameters())
    p_dtype = next((p.dtype for p in model.parameters()), torch.float32)
    dtype = build.get("dtype") or {
        torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"
    }.get(p_dtype, "fp32")
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    tracked: dict[str, float] = {}
    mem_tracker = None
    try:
        from torch.distributed._tools.mem_tracker import MemTracker

        mem_tracker = MemTracker()
    except Exception as e:  # pragma: no cover
        warns.append(f"MemTracker unavailable ({type(e).__name__}); activation memory is analytic")

    fc = FlopCounterMode(display=False)
    has_backward = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if mem_tracker is not None:
                externals = [model] + ([inputs] if isinstance(inputs, torch.Tensor) else [])
                mem_tracker.track_external(*externals)
            ctx = mem_tracker if mem_tracker is not None else _null()
            with ctx, fc:
                out = _call(model, inputs)
                if loss_fn is not None:
                    loss = loss_fn(out) if _arity(loss_fn) == 1 else loss_fn(out, build.get("targets"))
                    loss.backward()
                    has_backward = True
            if mem_tracker is not None:
                snap = mem_tracker.get_tracker_snapshot("peak")
                for _dev, cats in snap.items():
                    for k, v in cats.items():
                        name = str(getattr(k, "value", k)).lower()
                        key = {"parameter": "param", "gradient": "grad", "activation": "act",
                               "optstate": "opt", "temp": "temp", "other": "other",
                               "buffer": "buffer"}.get(name)
                        if key:
                            tracked[key] = tracked.get(key, 0.0) + float(v)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if any(s in msg for s in ("Unsupported", "not implemented", "NotImplemented", "data-dependent",
                                   "GuardOnDataDependent", "unbacked", "DynamicOutputShape")):
            raise UnsupportedTraceError(msg) from e
        raise UnsupportedTraceError(msg) from e

    counts = fc.get_flop_counts()
    total = float(fc.get_total_flops())
    per_module = {name: float(sum(v.values())) for name, v in counts.items() if name != "Global"}
    # FlopCounterMode does not separate forward from backward. Without backward, all
    # counted FLOPs are forward. With backward, a dense net has backward = 2x forward
    # except for the first layer's input-gradient, so we solve fwd from the ratio the
    # counter observed on this model: run forward alone under a second counter.
    if has_backward:
        fc_f = FlopCounterMode(display=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad(), fc_f:
                _call(model, inputs)
        fwd = float(fc_f.get_total_flops())
        bwd = max(0.0, total - fwd)
    else:
        fwd, bwd = total, 0.0
        warns.append("no loss provided; backward FLOPs assumed 2x forward")
        bwd = 2.0 * fwd

    batch_tokens = int(build.get("batch_tokens") or _count_tokens(inputs))
    model_class = build.get("model_class") or _infer_model_class(model)
    act_bytes = float(tracked.get("act", 0.0) + tracked.get("other", 0.0))
    return TraceResult(
        flops_forward=fwd,
        flops_backward=bwd,
        per_module=per_module,
        params=int(params),
        param_bytes=float(param_bytes),
        dtype=dtype,
        tracked_memory=tracked,
        activation_bytes=act_bytes,
        model_class=model_class,
        batch_tokens=batch_tokens,
        metadata={k: v for k, v in build.items() if k not in {"model", "inputs", "loss", "targets"}},
        warnings=warns,
        has_backward=has_backward,
    )


def trace_entrypoint(path: str | Path, config: dict[str, Any] | None = None) -> TraceResult:
    """Import the entrypoint, build the model under FakeTensorMode, trace it."""
    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
    except ImportError as e:  # pragma: no cover
        raise EstimationError("torch is required for tracing; install compute-prereg-mcp[torch]") from e
    module = load_entrypoint(path)
    config = config or {}
    build = None
    errors: list[str] = []
    # Attempt 1: build inside FakeTensorMode (no real memory allocated).
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with FakeTensorMode():
                build = module.prereg_build(config)
                return trace_build(build, config)
    except UnsupportedTraceError as e:
        errors.append(f"fake build: {e}")
    except Exception as e:  # constructor did something FakeTensor cannot do
        errors.append(f"fake build: {type(e).__name__}: {e}")
    # Attempt 2: real CPU build, fake forward with non-fake inputs allowed.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            build = module.prereg_build(config)
            with FakeTensorMode(allow_non_fake_inputs=True):
                res = trace_build(build, config)
        res.warnings.append("model was built on CPU with real tensors; forward traced under FakeTensorMode")
        return res
    except Exception as e:
        errors.append(f"real build: {type(e).__name__}: {e}")
    raise UnsupportedTraceError("; ".join(errors))


def _arity(fn) -> int:
    import inspect

    try:
        return len([p for p in inspect.signature(fn).parameters.values()
                    if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
    except (TypeError, ValueError):
        return 1


def _is_fake(model) -> bool:
    try:
        from torch._subclasses.fake_tensor import FakeTensor

        return any(isinstance(p, FakeTensor) for p in model.parameters())
    except Exception:  # pragma: no cover
        return False


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
