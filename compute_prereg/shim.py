"""Telemetry shim: runs the user's training script in-process and records step timing,
GPU memory, and sampled torch.profiler windows to a JSON-lines file, without changing
the script.

    python -m compute_prereg.shim --telemetry /path/telemetry.jsonl [--profile-every 100]
                                  [--profile-steps 3] -- train.py [args...]

What it does:
  * wraps torch.optim.Optimizer.step so every optimizer step emits a record with wall
    time since the previous step, peak CUDA memory, and the last scalar loss it can see
  * every --profile-every steps, profiles --profile-steps steps with torch.profiler
    (with_flops=True) and emits a window record with time fractions by category
    (compute / memory-bound / data loading / comm / other) and measured FLOPs per step
  * honours checkpoint requests: if a file named CHECKPOINT_REQUESTED appears next to
    the telemetry file, compute_prereg.shim.checkpoint_requested() returns True so a
    cooperating script can save and exit; scripts that do not check it are simply
    terminated by the watchdog
  * writes a final record with the exit status

The shim never uploads anything; the executor pulls the file.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import runpy
import sys
import threading
import time
from pathlib import Path

_STATE: dict = {"telemetry": None, "lock": threading.Lock(), "step": 0, "last_t": None,
                "profile_every": 0, "profile_steps": 0, "prof": None, "prof_started_step": None,
                "checkpoint_flag": None, "t0": time.time()}

COMPUTE_OPS = ("mm", "matmul", "addmm", "bmm", "baddbmm", "convolution", "conv", "linear",
               "scaled_dot_product", "attention", "flash", "cudnn", "cublas", "gemm")
MEMORY_OPS = ("add", "mul", "div", "sub", "layer_norm", "native_layer_norm", "rms_norm", "softmax",
              "gelu", "silu", "relu", "copy_", "cat", "clone", "dropout", "embedding", "index",
              "elementwise", "vectorized", "reduce", "sum", "mean", "fill", "zero_", "contiguous", "transpose")
COMM_OPS = ("nccl", "c10d", "allreduce", "all_reduce", "allgather", "all_gather", "reduce_scatter", "broadcast")
DATA_OPS = ("dataloader", "__next__", "enumerate(DataLoader)")


def _emit(rec: dict) -> None:
    path = _STATE["telemetry"]
    if not path:
        return
    rec.setdefault("ts", time.time())
    with _STATE["lock"]:
        with open(path, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")


def checkpoint_requested() -> bool:
    flag = _STATE.get("checkpoint_flag")
    return bool(flag and os.path.exists(flag))


def _categorize(name: str) -> str:
    n = name.lower()
    if any(k in n for k in DATA_OPS):
        return "data_loading"
    if any(k in n for k in COMM_OPS):
        return "comm"
    if any(k in n for k in COMPUTE_OPS):
        return "compute"
    if any(k in n for k in MEMORY_OPS):
        return "memory_bound"
    return "other"


def _install_torch_hooks() -> None:
    try:
        import torch
    except ImportError:
        _emit({"kind": "warning", "message": "torch not importable; step telemetry disabled"})
        return
    from torch.optim.optimizer import register_optimizer_step_post_hook

    cuda = torch.cuda.is_available()

    def _maybe_start_profiler(step: int):
        every, n = _STATE["profile_every"], _STATE["profile_steps"]
        if every and n and step % every == 0 and _STATE["prof"] is None:
            try:
                acts = [torch.profiler.ProfilerActivity.CPU]
                if cuda:
                    acts.append(torch.profiler.ProfilerActivity.CUDA)
                prof = torch.profiler.profile(activities=acts, with_flops=True, record_shapes=False)
                prof.__enter__()
                _STATE["prof"], _STATE["prof_started_step"], _STATE["prof_t0"] = prof, step, time.time()
            except Exception as e:  # profiler unavailable on this build
                _emit({"kind": "warning", "message": f"profiler failed: {e}"})
                _STATE["profile_every"] = 0

    def _maybe_stop_profiler(step: int):
        prof = _STATE["prof"]
        if prof is None or step - _STATE["prof_started_step"] < _STATE["profile_steps"]:
            return
        wall = time.time() - _STATE["prof_t0"]
        steps = step - _STATE["prof_started_step"]
        try:
            prof.__exit__(None, None, None)
            totals: dict[str, float] = {}
            flops = 0.0
            tops: list[tuple[str, float]] = []
            for ev in prof.key_averages():
                t = None
                if cuda:
                    t = getattr(ev, "device_time_total", None) or getattr(ev, "cuda_time_total", None)
                    if t is not None and t <= 0:
                        t = None
                if t is None:
                    t = getattr(ev, "self_cpu_time_total", 0.0)
                t = float(t) / 1e6  # us -> s
                cat = _categorize(ev.key)
                totals[cat] = totals.get(cat, 0.0) + t
                tops.append((ev.key, t))
                flops += float(getattr(ev, "flops", 0) or 0)
            tops.sort(key=lambda x: -x[1])
            denom = max(wall, 1e-9)
            _emit({"kind": "profile", "step": step, "steps": steps, "wall_seconds": wall,
                   "fractions": {k: min(1.0, v / denom) for k, v in totals.items()},
                   "flops_per_step": flops / max(steps, 1) if flops else None,
                   "top_ops": [{"op": k, "seconds": v} for k, v in tops[:10]]})
        except Exception as e:
            _emit({"kind": "warning", "message": f"profiler summary failed: {e}"})
        finally:
            _STATE["prof"] = None

    def _post_step(optimizer, args, kwargs):
        # Global post-hook: runs after every Optimizer subclass's step(), including
        # AdamW/SGD/... which override step and would bypass a base-class patch.
        now = time.time()
        _STATE["step"] += 1
        s = _STATE["step"]
        rec = {"kind": "step", "step": s, "step_seconds": (now - _STATE["last_t"]) if _STATE["last_t"] else None}
        _STATE["last_t"] = now
        if cuda:
            try:
                rec["mem_gb"] = torch.cuda.max_memory_allocated() / 1024**3
                rec["mem_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            except Exception:
                pass
        if checkpoint_requested():
            rec["checkpoint_requested"] = True
        _emit(rec)
        _maybe_stop_profiler(s)
        _maybe_start_profiler(s)

    register_optimizer_step_post_hook(_post_step)
    try:
        from torch.utils.data import DataLoader

        orig_init = DataLoader.__init__

        def dl_init(self, *a, **k):
            orig_init(self, *a, **k)
            _emit({"kind": "dataloader", "num_workers": getattr(self, "num_workers", None),
                   "pin_memory": getattr(self, "pin_memory", None)})

        DataLoader.__init__ = dl_init  # type: ignore[method-assign]
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv:
        print("usage: python -m compute_prereg.shim --telemetry FILE [--profile-every N] [--profile-steps K] -- script.py [args]",
              file=sys.stderr)
        return 2
    i = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry", required=True)
    ap.add_argument("--profile-every", type=int, default=100)
    ap.add_argument("--profile-steps", type=int, default=3)
    opts = ap.parse_args(argv[:i])
    script, *script_args = argv[i + 1:]

    tpath = Path(opts.telemetry)
    tpath.parent.mkdir(parents=True, exist_ok=True)
    _STATE.update(telemetry=str(tpath), profile_every=opts.profile_every, profile_steps=opts.profile_steps,
                  checkpoint_flag=str(tpath.parent / "CHECKPOINT_REQUESTED"))
    _emit({"kind": "start", "script": script, "args": script_args, "pid": os.getpid()})
    _install_torch_hooks()

    exit_code = 0

    def _close_profiler():
        prof = _STATE.get("prof")
        if prof is not None:  # an open window at teardown aborts the process (kineto)
            _STATE["prof"] = None
            try:
                prof.__exit__(None, None, None)
            except Exception:
                pass

    def _final():
        _close_profiler()
        _emit({"kind": "end", "exit_code": exit_code, "steps": _STATE["step"], "wall_seconds": time.time() - _STATE["t0"]})

    atexit.register(_final)
    sys.argv = [script, *script_args]
    sys.path.insert(0, str(Path(script).resolve().parent))
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        exit_code = int(e.code or 0) if isinstance(e.code, int) or e.code is None else 1
    except BaseException as e:  # noqa: BLE001
        exit_code = 1
        _emit({"kind": "error", "message": f"{type(e).__name__}: {e}"})
        _close_profiler()
        raise
    _close_profiler()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
