"""Modal adapter: serverless GPU functions, billed per second. No instances and no SSH,
so an Instance here is kind="function"; the executor's ModalLauncher runs the entrypoint
inside a Modal function with the job directory mounted.

Credentials come from Modal's own config (`modal token set`) or MODAL_TOKEN_ID /
MODAL_TOKEN_SECRET in the environment. Prices are Modal's published list (source=index).

Status: adapter logic is complete; the launcher path requires the `modal` package and an
account, and is not exercised by the test suite.
"""

from __future__ import annotations

import importlib.util
import os
from datetime import UTC, datetime, timedelta

from ... import ids
from ...errors import AdapterUnavailable
from ...est.gpus import GPUS, gpu_matches_class
from ...models import Envelope, Instance, Money, Offer, PriceSource, utcnow
from ..base import ProviderAdapter

_ASOF = datetime(2026, 9, 1, tzinfo=UTC)

# canonical GPU -> (modal gpu string, $/h list price)
MODAL_GPUS: dict[str, tuple[str, float]] = {
    "T4-16GB": ("T4", 0.59), "L4-24GB": ("L4", 0.80), "A10G-24GB": ("A10G", 1.10),
    "L40S-48GB": ("L40S", 1.95), "A100-PCIe-40GB": ("A100-40GB", 2.10), "A100-SXM-80GB": ("A100-80GB", 2.50),
    "H100-SXM-80GB": ("H100", 3.95), "H200-141GB": ("H200", 4.54), "B200-180GB": ("B200", 6.25),
}


class ModalAdapter(ProviderAdapter):
    name = "modal"

    def __init__(self, *, require_package: bool = True):
        has_pkg = importlib.util.find_spec("modal") is not None
        has_cred = bool(os.environ.get("MODAL_TOKEN_ID")) or os.path.exists(os.path.expanduser("~/.modal.toml"))
        if require_package and not has_pkg:
            raise AdapterUnavailable("modal package not installed (pip install compute-prereg-mcp[modal])")
        if require_package and not has_cred:
            raise AdapterUnavailable("no Modal credentials (run `modal token set` or set MODAL_TOKEN_ID/SECRET)")

    async def list_offers(self, envelope: Envelope) -> list[Offer]:
        out = []
        for canon, (modal_name, usd) in MODAL_GPUS.items():
            if not gpu_matches_class(canon, envelope.gpu_classes):
                continue
            for count in range(max(1, envelope.min_gpu_count), min(8, envelope.max_gpu_count) + 1):
                out.append(Offer(
                    offer_id=ids.offer_id(), provider=self.name, gpu_type=canon, count=count,
                    vram_gb=GPUS[canon].vram_gb, interconnect=GPUS[canon].interconnect if count > 1 else None,
                    tier="on-demand", availability="now", region="modal-us",
                    price_per_hour=Money(usd=usd * count, source=PriceSource.index, asof=_ASOF,
                                         note="Modal list price, billed per second"),
                    reliability_score=0.85,
                    provider_ref={"gpu": modal_name if count == 1 else f"{modal_name}:{count}"},
                    expires_at=utcnow() + timedelta(hours=1)))
        return out

    async def provision(self, offer: Offer, ssh_public_key: str, job_id: str) -> Instance:
        # Nothing to provision up front: the function is spawned by the launcher. The SSH
        # key is irrelevant for a function backend and is deliberately not sent anywhere.
        return Instance(instance_id=ids.instance_id(), provider=self.name, offer_id=offer.offer_id, kind="function",
                        provider_ref={"gpu": offer.provider_ref["gpu"], "job_id": job_id, "app_name": f"prereg-{job_id}"})

    async def terminate(self, instance: Instance) -> None:
        call_id = instance.provider_ref.get("function_call_id")
        if not call_id:
            return
        import modal  # lazy

        try:
            modal.FunctionCall.from_id(call_id).cancel()
        except Exception:
            pass

    async def usage(self, instance: Instance) -> Money | None:
        return None  # Modal has no per-call billing API; reconcile reports billed: unavailable

    async def instance_state(self, instance: Instance) -> str:
        return "running" if instance.provider_ref.get("function_call_id") else "unknown"
