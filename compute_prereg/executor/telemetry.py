"""Turn shim / dmon records into a TelemetrySummary."""

from __future__ import annotations

import statistics
from datetime import UTC, datetime
from typing import Any

from ..models import ProfilerSummary, TelemetrySummary


def _ts(v: float | None) -> datetime | None:
    return datetime.fromtimestamp(v, tz=UTC) if v else None


def summarize(records: list[dict[str, Any]], previous: TelemetrySummary | None = None) -> TelemetrySummary:
    steps = [r for r in records if r.get("kind") == "step"]
    profiles = [r for r in records if r.get("kind") == "profile"]
    gpu = [r for r in records if r.get("kind") == "gpu"]
    loaders = [r for r in records if r.get("kind") == "dataloader"]
    step_secs = [float(r["step_seconds"]) for r in steps if r.get("step_seconds")]
    mem = [float(r["mem_gb"]) for r in steps if r.get("mem_gb") is not None]
    mem += [float(r["mem_gb"]) for r in gpu if r.get("mem_gb") is not None]
    utils = [float(r["util"]) for r in gpu if r.get("util") is not None]
    losses = [r["loss"] for r in steps if r.get("loss") is not None]

    prof = None
    if profiles:
        keys = ("compute", "memory_bound", "data_loading", "comm", "other")
        acc = {k: 0.0 for k in keys}
        n_windows, n_steps = 0, 0
        flops = []
        top: dict[str, float] = {}
        for p in profiles:
            fr = p.get("fractions") or {}
            for k in keys:
                acc[k] += float(fr.get(k, 0.0))
            n_windows += 1
            n_steps += int(p.get("steps") or 0)
            if p.get("flops_per_step"):
                flops.append(float(p["flops_per_step"]))
            for op in p.get("top_ops") or []:
                top[op["op"]] = top.get(op["op"], 0.0) + float(op["seconds"])
        tops = sorted(top.items(), key=lambda kv: -kv[1])[:8]
        prof = ProfilerSummary(
            windows=n_windows, steps_profiled=n_steps,
            frac_compute=acc["compute"] / n_windows, frac_memory_bound=acc["memory_bound"] / n_windows,
            frac_data_loading=acc["data_loading"] / n_windows, frac_comm=acc["comm"] / n_windows,
            frac_other=acc["other"] / n_windows,
            measured_flops_per_step=statistics.fmean(flops) if flops else None,
            top_ops=[{"op": k, "seconds": round(v, 4)} for k, v in tops])

    last_ts = max((r.get("ts") for r in records if r.get("ts")), default=None)
    summary = TelemetrySummary(
        samples=len(records),
        steps_completed=max((int(r.get("step", 0)) for r in steps), default=0),
        mean_gpu_util=statistics.fmean(utils) if utils else None,
        peak_mem_gb=max(mem) if mem else None,
        mean_step_seconds=statistics.fmean(step_secs) if step_secs else None,
        p50_step_seconds=statistics.median(step_secs) if step_secs else None,
        first_step_at=_ts(steps[0].get("ts")) if steps else None,
        last_step_at=_ts(steps[-1].get("ts")) if steps else None,
        last_sample_at=_ts(last_ts),
        gaps=list(previous.gaps) if previous else [],
        num_workers=next((int(r["num_workers"]) for r in loaders if r.get("num_workers") is not None), None),
        profiler=prof,
        last_loss=float(losses[-1]) if losses else None,
    )
    return summary


def parse_dmon(text: str) -> list[dict[str, Any]]:
    """Parse `nvidia-smi dmon -s um -d 1` output into gpu records (util %, mem GB)."""
    out = []
    now = datetime.now(UTC).timestamp()
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            # columns: gpu sm mem enc dec ... fb bar1 (order per -s flags: u then m)
            util = float(parts[1])
            fb_mb = float(parts[-2]) if len(parts) >= 4 else None
        except (ValueError, IndexError):
            continue
        out.append({"kind": "gpu", "ts": now, "util": util, "mem_gb": (fb_mb / 1024) if fb_mb else None})
    return out
