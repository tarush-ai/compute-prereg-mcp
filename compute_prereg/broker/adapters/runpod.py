"""RunPod adapter (GraphQL API). Credential: RUNPOD_API_KEY from the environment only.

Secure-cloud pods are offered on-demand; community-cloud pods as spot (interruptible bid).
The per-job SSH public key is injected as the PUBLIC_KEY env var, which RunPod's official
images add to authorized_keys — nothing else is registered anywhere.

Status: written against the public GraphQL schema; not exercised against a live account
in this repository's test suite. Contract tests belong in CI against a sandbox account.
"""

from __future__ import annotations

import asyncio
import os
from datetime import timedelta
from typing import Any

import httpx

from ... import ids
from ...errors import AdapterUnavailable, ProvisionError
from ...est.gpus import GPUS, gpu_matches_class, normalize_gpu_name
from ...models import Envelope, Instance, Money, Offer, PriceSource, utcnow
from ...security import redact
from ..base import ProviderAdapter

GRAPHQL = "https://api.runpod.io/graphql"
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

_Q_GPUS = """
query { gpuTypes { id displayName memoryInGb secureCloud communityCloud
  lowestPrice(input: {gpuCount: 1}) { minimumBidPrice uninterruptablePrice stockStatus } } }
"""
_M_ONDEMAND = """
mutation($input: PodFindAndDeployOnDemandInput) {
  podFindAndDeployOnDemand(input: $input) { id machineId costPerHr } }
"""
_M_SPOT = """
mutation($input: PodRentInterruptableInput) {
  podRentInterruptable(input: $input) { id machineId costPerHr } }
"""
_Q_POD = """
query($id: String!) { pod(input: {podId: $id}) { id name desiredStatus costPerHr
  runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } } }
"""
_M_TERMINATE = "mutation($id: String!) { podTerminate(input: {podId: $id}) }"
_Q_MY_PODS = "query { myself { pods { id name desiredStatus costPerHr runtime { uptimeInSeconds } } } }"


class RunPodAdapter(ProviderAdapter):
    name = "runpod"

    def __init__(self, api_key: str | None = None, *, image: str = DEFAULT_IMAGE, volume_gb: int = 40,
                 container_disk_gb: int = 40, timeout: float = 30.0):
        key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not key:
            raise AdapterUnavailable("RUNPOD_API_KEY is not set")
        self._headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.image, self.volume_gb, self.container_disk_gb, self.timeout = image, volume_gb, container_disk_gb, timeout

    async def _gql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(GRAPHQL, json={"query": query, "variables": variables or {}}, headers=self._headers)
        try:
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            raise ProvisionError(redact(f"runpod API error: {e}")) from e
        if data.get("errors"):
            raise ProvisionError(redact(f"runpod GraphQL error: {data['errors']}"))
        return data["data"]

    async def list_offers(self, envelope: Envelope) -> list[Offer]:
        data = await self._gql(_Q_GPUS)
        out: list[Offer] = []
        for g in data.get("gpuTypes", []):
            canon = normalize_gpu_name(g.get("displayName") or g.get("id") or "")
            if canon is None or not gpu_matches_class(canon, envelope.gpu_classes):
                continue
            lp = g.get("lowestPrice") or {}
            vram = float(g.get("memoryInGb") or GPUS[canon].vram_gb)
            if lp.get("stockStatus") in (None, "None", "Unavailable"):
                pass
            common = dict(provider=self.name, gpu_type=canon, vram_gb=vram, region="runpod-global",
                          interconnect=GPUS[canon].interconnect, expires_at=utcnow() + timedelta(minutes=5))
            if g.get("secureCloud") and lp.get("uninterruptablePrice"):
                out.append(Offer(offer_id=ids.offer_id(), tier="on-demand", reliability_score=0.8,
                                 price_per_hour=Money(usd=float(lp["uninterruptablePrice"]), source=PriceSource.quoted),
                                 availability="now" if lp.get("stockStatus") not in ("None", None) else "queue",
                                 provider_ref={"gpuTypeId": g["id"], "cloudType": "SECURE"}, **common))
            if g.get("communityCloud") and lp.get("minimumBidPrice") and envelope.spot_ok:
                out.append(Offer(offer_id=ids.offer_id(), tier="spot", reliability_score=0.55,
                                 price_per_hour=Money(usd=float(lp["minimumBidPrice"]), source=PriceSource.quoted),
                                 availability="now",
                                 provider_ref={"gpuTypeId": g["id"], "cloudType": "COMMUNITY",
                                               "bidPerGpu": float(lp["minimumBidPrice"])}, **common))
        return out

    async def provision(self, offer: Offer, ssh_public_key: str, job_id: str) -> Instance:
        ref = offer.provider_ref
        inp: dict[str, Any] = {
            "cloudType": ref["cloudType"], "gpuCount": offer.count, "gpuTypeId": ref["gpuTypeId"],
            "name": f"prereg-{job_id}", "imageName": self.image, "ports": "22/tcp",
            "volumeInGb": self.volume_gb, "containerDiskInGb": self.container_disk_gb,
            "minVcpuCount": 2, "minMemoryInGb": 15, "volumeMountPath": "/workspace",
            "env": [{"key": "PUBLIC_KEY", "value": ssh_public_key}],
        }
        if ref["cloudType"] == "COMMUNITY":
            inp["bidPerGpu"] = ref["bidPerGpu"]
            data = await self._gql(_M_SPOT, {"input": inp})
            pod = data["podRentInterruptable"]
        else:
            data = await self._gql(_M_ONDEMAND, {"input": inp})
            pod = data["podFindAndDeployOnDemand"]
        if not pod or not pod.get("id"):
            raise ProvisionError("runpod: no pod returned (out of stock?)")
        pod_id = pod["id"]
        # Wait for SSH port mapping.
        for _ in range(60):
            await asyncio.sleep(5)
            st = (await self._gql(_Q_POD, {"id": pod_id}))["pod"]
            rt = (st or {}).get("runtime") or {}
            for p in rt.get("ports") or []:
                if p.get("privatePort") == 22 and p.get("isIpPublic"):
                    return Instance(instance_id=pod_id, provider=self.name, offer_id=offer.offer_id, kind="ssh",
                                    host=p["ip"], port=int(p["publicPort"]), user="root",
                                    provider_ref={"podId": pod_id, "job_id": job_id, "costPerHr": pod.get("costPerHr")})
        # Did not come up: terminate so it does not bill, then report.
        try:
            await self._gql(_M_TERMINATE, {"id": pod_id})
        except Exception:
            pass
        raise ProvisionError(f"runpod: pod {pod_id} never exposed SSH; terminated it")

    async def terminate(self, instance: Instance) -> None:
        await self._gql(_M_TERMINATE, {"id": instance.provider_ref.get("podId", instance.instance_id)})

    async def usage(self, instance: Instance) -> Money | None:
        try:
            st = (await self._gql(_Q_POD, {"id": instance.provider_ref.get("podId", instance.instance_id)}))["pod"]
        except Exception:
            return None
        if not st:
            return None
        rt = st.get("runtime") or {}
        up = rt.get("uptimeInSeconds")
        rate = st.get("costPerHr")
        if up is None or rate is None:
            return None
        # RunPod exposes rate x uptime, not an invoice line; label it as quoted-rate derived.
        return Money(usd=round(float(rate) * float(up) / 3600, 6), source=PriceSource.derived,
                     note="runpod costPerHr x uptimeInSeconds")

    async def find_leftovers(self, job_id: str) -> list[Instance]:
        data = await self._gql(_Q_MY_PODS)
        out = []
        for p in (data.get("myself") or {}).get("pods") or []:
            if p.get("name") == f"prereg-{job_id}":
                out.append(Instance(instance_id=p["id"], provider=self.name, offer_id="", kind="ssh",
                                    provider_ref={"podId": p["id"], "job_id": job_id}))
        return out

    async def instance_state(self, instance: Instance) -> str:
        try:
            st = (await self._gql(_Q_POD, {"id": instance.provider_ref.get("podId", instance.instance_id)}))["pod"]
        except Exception:
            return "unknown"
        if not st:
            return "stopped"
        return {"RUNNING": "running", "EXITED": "stopped", "TERMINATED": "stopped"}.get(st.get("desiredStatus"), "unknown")
