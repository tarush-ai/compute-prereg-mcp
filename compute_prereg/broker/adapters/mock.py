"""Deterministic in-process adapter for tests and `--demo`. Instances are local
(the executor runs the job in a local subprocess). Inventory mirrors user story 1."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from ... import ids
from ...errors import ProvisionError
from ...models import Envelope, Instance, Money, Offer, PriceSource, utcnow
from ..base import ProviderAdapter


def _quoted(usd: float) -> Money:
    return Money(usd=usd, source=PriceSource.quoted)


def default_inventory() -> list[dict[str, Any]]:
    return [
        dict(underlying_provider="lambda", gpu_type="H100-SXM-80GB", vram_gb=80, tier="on-demand", price=2.49,
             availability="now", region="us-west-1", reliability_score=0.9),
        dict(underlying_provider="verda", gpu_type="H100-SXM-80GB", vram_gb=80, tier="on-demand", price=2.10,
             availability="queue", queue_eta_minutes=12, region="eu-fi-1", reliability_score=0.85),
        dict(underlying_provider="runpod", gpu_type="H100-SXM-80GB", vram_gb=80, tier="spot", price=1.65,
             availability="now", region="us-east-2", reliability_score=0.7),
        dict(underlying_provider="runpod", gpu_type="A100-SXM-80GB", vram_gb=80, tier="on-demand", price=1.39,
             availability="now", region="us-east-1", reliability_score=0.8),
        dict(underlying_provider="vast", gpu_type="RTX-4090-24GB", vram_gb=24, tier="spot", price=0.35,
             availability="now", region="us-ca-1", reliability_score=0.5),
        dict(underlying_provider="lambda", gpu_type="H100-SXM-80GB", vram_gb=80, count=8, interconnect="nvlink",
             tier="on-demand", price=19.9, availability="now", region="us-west-1", reliability_score=0.9),
    ]


class MockAdapter(ProviderAdapter):
    name = "mock"
    is_aggregator = True

    def __init__(self, inventory: list[dict[str, Any]] | None = None, *, fail_provision: bool = False,
                 leftover_on_fail: bool = False, billing_available: bool = True, name: str = "mock"):
        self.name = name
        self.inventory = inventory if inventory is not None else default_inventory()
        self.fail_provision = fail_provision
        self.leftover_on_fail = leftover_on_fail
        self.billing_available = billing_available
        self.provisioned: list[Instance] = []
        self.terminated: list[str] = []
        self._leftovers: dict[str, list[Instance]] = {}
        self._prices: dict[str, float] = {}
        self._starts: dict[str, datetime] = {}

    async def list_offers(self, envelope: Envelope) -> list[Offer]:
        out = []
        for i, item in enumerate(self.inventory):
            item = dict(item)
            price = item.pop("price")
            out.append(Offer(
                offer_id=ids.offer_id(), provider=self.name, price_per_hour=_quoted(price),
                provider_ref={"sku": i}, expires_at=utcnow() + timedelta(minutes=10), **item))
        return out

    async def provision(self, offer: Offer, ssh_public_key: str, job_id: str) -> Instance:
        if not ssh_public_key.startswith("ssh-ed25519 "):
            raise ProvisionError("mock adapter requires an ed25519 public key")
        inst = Instance(instance_id=ids.instance_id(), provider=self.name, offer_id=offer.offer_id, kind="local",
                        provider_ref={"job_id": job_id, "sku": offer.provider_ref.get("sku"),
                                      "authorized_key": ssh_public_key})
        if self.fail_provision:
            if self.leftover_on_fail:
                self._leftovers.setdefault(job_id, []).append(inst)
                self._starts[inst.instance_id] = datetime.now(UTC)
                self._prices[inst.instance_id] = offer.price_per_hour.usd
            raise ProvisionError("mock: capacity unavailable")
        self.provisioned.append(inst)
        self._starts[inst.instance_id] = datetime.now(UTC)
        self._prices[inst.instance_id] = offer.price_per_hour.usd
        return inst

    async def terminate(self, instance: Instance) -> None:
        self.terminated.append(instance.instance_id)
        job_id = instance.provider_ref.get("job_id")
        if job_id in self._leftovers:
            self._leftovers[job_id] = [i for i in self._leftovers[job_id] if i.instance_id != instance.instance_id]

    async def usage(self, instance: Instance) -> Money | None:
        if not self.billing_available or instance.instance_id not in self._starts:
            return None
        hours = (datetime.now(UTC) - self._starts[instance.instance_id]).total_seconds() / 3600
        return Money(usd=round(hours * self._prices[instance.instance_id], 6), source=PriceSource.quoted,
                     note="mock provider usage API")

    async def find_leftovers(self, job_id: str) -> list[Instance]:
        return list(self._leftovers.get(job_id, []))

    async def instance_state(self, instance: Instance) -> str:
        return "stopped" if instance.instance_id in self.terminated else "running"
