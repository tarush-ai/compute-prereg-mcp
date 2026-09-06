"""SkyPilot-as-a-library adapter: one integration covers 20+ clouds and Kubernetes.

Offers come from the SkyPilot catalog (list_accelerators). Provisioning launches a cluster
named after the job and appends the per-job public key to authorized_keys in the setup
step, so our executor can reach it over SSH with the ephemeral key. Terminate = sky.down.

Credentials are whatever SkyPilot's own cloud checks find (`sky check`).

Status: written against SkyPilot's Python API (0.8-0.10 series, which returns request ids
from launch/status and needs sky.get()); not exercised by the test suite.
"""

from __future__ import annotations

import asyncio
import importlib.util
from datetime import timedelta
from typing import Any

from ... import ids
from ...errors import AdapterUnavailable, ProvisionError
from ...est.gpus import GPUS, gpu_matches_class, normalize_gpu_name
from ...models import Envelope, Instance, Money, Offer, PriceSource, utcnow
from ...security import redact
from ..base import ProviderAdapter


def _get(sky, result):
    """SkyPilot >= 0.8 returns request ids from most API calls; resolve them."""
    if isinstance(result, str) and hasattr(sky, "get"):
        return sky.get(result)
    return result


class SkyPilotAdapter(ProviderAdapter):
    name = "skypilot"
    is_aggregator = True

    def __init__(self, clouds: list[str] | None = None):
        if importlib.util.find_spec("sky") is None:
            raise AdapterUnavailable("skypilot not installed (pip install compute-prereg-mcp[skypilot])")
        self.clouds = clouds

    async def list_offers(self, envelope: Envelope) -> list[Offer]:
        import sky

        def _catalog() -> dict[str, list[Any]]:
            fn = getattr(sky, "list_accelerators", None) or sky.core.list_accelerators
            return _get(sky, fn(gpus_only=True, clouds=self.clouds, all_regions=False, require_price=True))

        catalog = await asyncio.get_running_loop().run_in_executor(None, _catalog)
        out: list[Offer] = []
        for acc_name, infos in catalog.items():
            canon = normalize_gpu_name(acc_name)
            if canon is None or not gpu_matches_class(canon, envelope.gpu_classes):
                continue
            for info in infos:
                count = int(getattr(info, "accelerator_count", 1) or 1)
                if not (envelope.min_gpu_count <= count <= envelope.max_gpu_count):
                    continue
                cloud = str(getattr(info, "cloud", "unknown")).lower()
                region = str(getattr(info, "region", None) or "unknown")
                mem = getattr(info, "device_memory", None)
                vram = float(mem) if mem else GPUS[canon].vram_gb
                common = dict(provider=self.name, underlying_provider=cloud, gpu_type=canon, count=count,
                              vram_gb=vram, region=region, interconnect=GPUS[canon].interconnect,
                              expires_at=utcnow() + timedelta(minutes=30))
                ref = {"cloud": cloud, "instance_type": getattr(info, "instance_type", None),
                       "accelerator": f"{acc_name}:{count}", "region": region}
                price = getattr(info, "price", None)
                if price:
                    out.append(Offer(offer_id=ids.offer_id(), tier="on-demand", reliability_score=0.8,
                                     price_per_hour=Money(usd=float(price), source=PriceSource.index,
                                                          note="skypilot catalog"),
                                     provider_ref={**ref, "use_spot": False}, **common))
                spot = getattr(info, "spot_price", None)
                if spot and envelope.spot_ok:
                    out.append(Offer(offer_id=ids.offer_id(), tier="spot", reliability_score=0.6,
                                     price_per_hour=Money(usd=float(spot), source=PriceSource.index,
                                                          note="skypilot catalog"),
                                     provider_ref={**ref, "use_spot": True}, **common))
        return out

    async def provision(self, offer: Offer, ssh_public_key: str, job_id: str) -> Instance:
        import sky

        ref = offer.provider_ref
        cluster = f"prereg-{job_id.lower().replace('_', '-')}"
        setup = ("mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
                 f"echo '{ssh_public_key}' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys")

        def _launch():
            task = sky.Task(name=cluster, setup=setup, run="true")
            task.set_resources(sky.Resources(
                cloud=sky.clouds.CLOUD_REGISTRY.from_str(ref["cloud"]) if ref.get("cloud") else None,
                instance_type=ref.get("instance_type"), accelerators=ref.get("accelerator"),
                use_spot=bool(ref.get("use_spot")), region=None if ref.get("region") in (None, "unknown") else ref["region"]))
            _get(sky, sky.launch(task, cluster_name=cluster, detach_run=True))
            statuses = _get(sky, sky.status(cluster_names=[cluster]))
            if not statuses:
                raise ProvisionError("skypilot: cluster not found after launch")
            handle = statuses[0].get("handle")
            head_ip = getattr(handle, "head_ip", None)
            user = getattr(handle, "ssh_user", None) or getattr(getattr(handle, "docker_user", None), "user", None)
            return head_ip, user or "ubuntu"

        try:
            head_ip, user = await asyncio.get_running_loop().run_in_executor(None, _launch)
        except Exception as e:
            # A failed launch can leave a billing cluster; tear it down before reporting.
            try:
                await asyncio.get_running_loop().run_in_executor(None, lambda: _get(sky, sky.down(cluster)))
            except Exception:
                pass
            raise ProvisionError(redact(f"skypilot launch failed: {e}")) from e
        if not head_ip:
            raise ProvisionError("skypilot: cluster has no head IP")
        return Instance(instance_id=cluster, provider=self.name, offer_id=offer.offer_id, kind="ssh",
                        host=head_ip, port=22, user=user, provider_ref={"cluster": cluster, "job_id": job_id})

    async def terminate(self, instance: Instance) -> None:
        import sky

        cluster = instance.provider_ref.get("cluster", instance.instance_id)
        await asyncio.get_running_loop().run_in_executor(None, lambda: _get(sky, sky.down(cluster)))

    async def usage(self, instance: Instance) -> Money | None:
        # SkyPilot has cost reports for managed jobs but no per-cluster invoice; leave None.
        return None

    async def find_leftovers(self, job_id: str) -> list[Instance]:
        import sky

        cluster = f"prereg-{job_id.lower().replace('_', '-')}"
        statuses = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _get(sky, sky.status(cluster_names=[cluster])))
        return [Instance(instance_id=cluster, provider=self.name, offer_id="", provider_ref={"cluster": cluster})
                for _ in (statuses or [])]

    async def instance_state(self, instance: Instance) -> str:
        import sky

        cluster = instance.provider_ref.get("cluster", instance.instance_id)
        statuses = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _get(sky, sky.status(cluster_names=[cluster])))
        if not statuses:
            return "stopped"
        st = str(statuses[0].get("status", "")).upper()
        return "running" if "UP" in st else "stopped" if "STOPPED" in st else "unknown"
