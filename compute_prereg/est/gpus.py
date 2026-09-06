"""Static roofline table for common training GPUs (dense peak TFLOPS, no sparsity).

Reference prices are a static placeholder index (source=index). The broker uses live
provider quotes for the actual dollar figure; these are only for the "$ at reference
price" column of an estimate and for the fairness comparison in reconcile.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from ..models import GpuSpec, Money, PriceSource

_ASOF = datetime(2026, 9, 1, tzinfo=UTC)


def _ref(usd: float) -> Money:
    return Money(usd=usd, source=PriceSource.index, asof=_ASOF, note="static reference index")


GPUS: dict[str, GpuSpec] = {
    g.name: g
    for g in [
        GpuSpec(
            name="H100-SXM-80GB", gpu_class="H100", vram_gb=80, hbm_gbps=3350,
            peak_tflops={"fp8": 1979, "bf16": 989, "fp16": 989, "tf32": 495, "fp32": 67},
            ref_price_per_hour=_ref(2.49), interconnect="nvlink",
            aliases=["H100", "H100-80GB", "H100 SXM", "H100-SXM", "H100 80GB HBM3", "H100_SXM"],
        ),
        GpuSpec(
            name="H100-PCIe-80GB", gpu_class="H100", vram_gb=80, hbm_gbps=2000,
            peak_tflops={"fp8": 1513, "bf16": 756, "fp16": 756, "tf32": 378, "fp32": 51},
            ref_price_per_hour=_ref(1.99), interconnect="pcie",
            aliases=["H100 PCIe", "H100-PCIE", "H100_PCIE", "H100 PCIE 80GB"],
        ),
        GpuSpec(
            name="H200-141GB", gpu_class="H200", vram_gb=141, hbm_gbps=4800,
            peak_tflops={"fp8": 1979, "bf16": 989, "fp16": 989, "tf32": 495, "fp32": 67},
            ref_price_per_hour=_ref(3.49), interconnect="nvlink",
            aliases=["H200", "H200 SXM", "H200-SXM"],
        ),
        GpuSpec(
            name="B200-180GB", gpu_class="B200", vram_gb=180, hbm_gbps=8000,
            peak_tflops={"fp4": 9000, "fp8": 4500, "bf16": 2250, "fp16": 2250, "tf32": 1100, "fp32": 75},
            ref_price_per_hour=_ref(5.49), interconnect="nvlink",
            aliases=["B200", "B200 SXM"],
        ),
        GpuSpec(
            name="A100-SXM-80GB", gpu_class="A100", vram_gb=80, hbm_gbps=2039,
            peak_tflops={"bf16": 312, "fp16": 312, "tf32": 156, "fp32": 19.5},
            ref_price_per_hour=_ref(1.49), interconnect="nvlink",
            aliases=["A100", "A100-80GB", "A100 80GB", "A100 SXM4 80GB", "A100-SXM4-80GB", "A100_SXM"],
        ),
        GpuSpec(
            name="A100-PCIe-40GB", gpu_class="A100", vram_gb=40, hbm_gbps=1555,
            peak_tflops={"bf16": 312, "fp16": 312, "tf32": 156, "fp32": 19.5},
            ref_price_per_hour=_ref(1.09), interconnect="pcie",
            aliases=["A100-40GB", "A100 40GB", "A100 PCIe", "A100-PCIE-40GB"],
        ),
        GpuSpec(
            name="L40S-48GB", gpu_class="L40S", vram_gb=48, hbm_gbps=864,
            peak_tflops={"fp8": 733, "bf16": 362, "fp16": 362, "tf32": 183, "fp32": 91.6},
            ref_price_per_hour=_ref(0.99), interconnect="pcie",
            aliases=["L40S", "L40S 48GB"],
        ),
        GpuSpec(
            name="L4-24GB", gpu_class="L4", vram_gb=24, hbm_gbps=300,
            peak_tflops={"fp8": 242, "bf16": 121, "fp16": 121, "tf32": 60, "fp32": 30.3},
            ref_price_per_hour=_ref(0.49), interconnect="pcie",
            aliases=["L4", "L4 24GB"],
        ),
        GpuSpec(
            name="A10G-24GB", gpu_class="A10G", vram_gb=24, hbm_gbps=600,
            peak_tflops={"bf16": 70, "fp16": 70, "tf32": 35, "fp32": 31.2},
            ref_price_per_hour=_ref(0.59), interconnect="pcie",
            aliases=["A10G", "A10", "A10 24GB"],
        ),
        GpuSpec(
            name="RTX-4090-24GB", gpu_class="RTX4090", vram_gb=24, hbm_gbps=1008,
            peak_tflops={"fp8": 330, "bf16": 165, "fp16": 165, "tf32": 82.6, "fp32": 82.6},
            ref_price_per_hour=_ref(0.39), interconnect="pcie",
            aliases=["RTX 4090", "4090", "RTX4090", "GeForce RTX 4090"],
        ),
        GpuSpec(
            name="RTX-6000-Ada-48GB", gpu_class="RTX6000Ada", vram_gb=48, hbm_gbps=960,
            peak_tflops={"fp8": 365, "bf16": 182.5, "fp16": 182.5, "tf32": 91, "fp32": 91.1},
            ref_price_per_hour=_ref(0.89), interconnect="pcie",
            aliases=["RTX 6000 Ada", "RTX6000 Ada", "RTX 6000 Ada Generation", "RTX6000-Ada"],
        ),
        GpuSpec(
            name="RTX-A6000-48GB", gpu_class="A6000", vram_gb=48, hbm_gbps=768,
            peak_tflops={"bf16": 155, "fp16": 155, "tf32": 77, "fp32": 38.7},
            ref_price_per_hour=_ref(0.49), interconnect="pcie",
            aliases=["RTX A6000", "A6000", "RTXA6000"],
        ),
        GpuSpec(
            name="V100-32GB", gpu_class="V100", vram_gb=32, hbm_gbps=900,
            peak_tflops={"fp16": 125, "fp32": 15.7},
            ref_price_per_hour=_ref(0.79), interconnect="nvlink",
            aliases=["V100", "V100 32GB", "Tesla V100", "V100-SXM2-32GB"],
        ),
        GpuSpec(
            name="T4-16GB", gpu_class="T4", vram_gb=16, hbm_gbps=320,
            peak_tflops={"fp16": 65, "fp32": 8.1},
            ref_price_per_hour=_ref(0.29), interconnect="pcie",
            aliases=["T4", "Tesla T4"],
        ),
        GpuSpec(
            name="MI300X-192GB", gpu_class="MI300X", vram_gb=192, hbm_gbps=5300,
            peak_tflops={"fp8": 2615, "bf16": 1307, "fp16": 1307, "tf32": 653, "fp32": 163},
            ref_price_per_hour=_ref(2.49), interconnect="infinity-fabric",
            aliases=["MI300X", "AMD MI300X", "MI300"],
        ),
    ]
}

DEFAULT_CANDIDATES = ["H100-SXM-80GB", "H200-141GB", "A100-SXM-80GB", "L40S-48GB", "RTX-4090-24GB"]

_ALIAS_INDEX: dict[str, str] = {}
for _g in GPUS.values():
    _ALIAS_INDEX[_norm := _g.name.lower()] = _g.name
    for _a in _g.aliases:
        _ALIAS_INDEX[_a.lower()] = _g.name


def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


_SQUASHED = {_squash(k): v for k, v in _ALIAS_INDEX.items()}


def normalize_gpu_name(raw: str) -> str | None:
    """Map a provider's GPU string ("NVIDIA H100 80GB HBM3", "1x A100 SXM4 80GB") onto a
    canonical roofline name. Returns None when unknown."""
    if not raw:
        return None
    s = raw.strip()
    if s in GPUS:
        return s
    low = s.lower()
    if low in _ALIAS_INDEX:
        return _ALIAS_INDEX[low]
    sq = _squash(re.sub(r"^\d+\s*x\s*", "", low))
    sq = sq.replace("nvidia", "").replace("tesla", "").replace("geforce", "")
    if sq in _SQUASHED:
        return _SQUASHED[sq]
    # Heuristic families.
    if "h200" in sq:
        return "H200-141GB"
    if "b200" in sq:
        return "B200-180GB"
    if "h100" in sq:
        return "H100-PCIe-80GB" if "pcie" in sq else "H100-SXM-80GB"
    if "a100" in sq:
        return "A100-PCIe-40GB" if "40" in sq else "A100-SXM-80GB"
    if "l40s" in sq:
        return "L40S-48GB"
    if "l4" in sq and "l40" not in sq:
        return "L4-24GB"
    if "a10" in sq:
        return "A10G-24GB"
    if "4090" in sq:
        return "RTX-4090-24GB"
    if "6000ada" in sq or ("6000" in sq and "ada" in sq):
        return "RTX-6000-Ada-48GB"
    if "a6000" in sq:
        return "RTX-A6000-48GB"
    if "v100" in sq:
        return "V100-32GB"
    if "t4" in sq:
        return "T4-16GB"
    if "mi300" in sq:
        return "MI300X-192GB"
    return None


def gpu_matches_class(gpu_name: str, allowed: list[str]) -> bool:
    """True if the canonical gpu_name satisfies any entry of an envelope's gpu_classes.
    Entries may be canonical names ("H100-SXM-80GB"), classes ("H100"), aliases, or
    fnmatch patterns ("H100*")."""
    from fnmatch import fnmatch

    spec = GPUS.get(gpu_name)
    if spec is None:
        return False
    for entry in allowed:
        e = entry.strip()
        if not e:
            continue
        if e == gpu_name or e.lower() == spec.gpu_class.lower():
            return True
        if normalize_gpu_name(e) == gpu_name:
            return True
        if any(ch in e for ch in "*?[") and (fnmatch(gpu_name, e) or fnmatch(spec.gpu_class, e)):
            return True
        if e.lower().endswith("-class") and e[:-6].lower() == spec.gpu_class.lower():
            return True
    return False


def resolve_candidates(names: list[str] | None) -> list[str]:
    if not names:
        return list(DEFAULT_CANDIDATES)
    out: list[str] = []
    for n in names:
        c = normalize_gpu_name(n)
        if c is None:
            raise KeyError(f"unknown GPU '{n}'. Known: {sorted(GPUS)}")
        if c not in out:
            out.append(c)
    return out
