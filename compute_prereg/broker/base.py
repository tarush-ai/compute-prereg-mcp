"""Provider adapter interface. Every provider — direct or aggregator — implements this."""

from __future__ import annotations

import abc

from ..models import Envelope, Instance, Money, Offer

# Rough grid-carbon score per provider/region family (0 clean .. 1 dirty). Used only by
# the lowest-carbon ranking policy; unknown providers get 0.5.
CARBON_SCORES: dict[str, float] = {
    "crusoe": 0.1, "nebius": 0.3, "verda": 0.2, "datacrunch": 0.2, "scaleway": 0.25,
    "lambda": 0.45, "runpod": 0.5, "vast": 0.6, "modal": 0.45, "aws": 0.4, "gcp": 0.35, "azure": 0.4,
}


class ProviderAdapter(abc.ABC):
    """One integration point per provider. Adapters must never log or return credentials."""

    name: str = "abstract"
    is_aggregator: bool = False

    @abc.abstractmethod
    async def list_offers(self, envelope: Envelope) -> list[Offer]:
        """Return offers (normalized) that could plausibly satisfy ``envelope``. The broker
        applies the envelope fit itself; adapters may pre-filter for efficiency."""

    @abc.abstractmethod
    async def provision(self, offer: Offer, ssh_public_key: str, job_id: str) -> Instance:
        """Start an instance registered with exactly this public key. Must raise
        ProvisionError on failure after cleaning up anything billable, or list what could
        not be cleaned via find_leftovers()."""

    @abc.abstractmethod
    async def terminate(self, instance: Instance) -> None:
        """Terminate the instance. This is the key-revocation event."""

    async def usage(self, instance: Instance) -> Money | None:
        """Billed cost from the provider's usage API. None when unavailable — never derive."""
        return None

    async def find_leftovers(self, job_id: str) -> list[Instance]:
        """Instances tagged with ``job_id`` that still exist (after a failed provision)."""
        return []

    async def instance_state(self, instance: Instance) -> str:
        """"running" | "stopped" | "preempted" | "unknown"."""
        return "running"

    def carbon_score(self, underlying: str | None = None) -> float:
        return CARBON_SCORES.get((underlying or self.name).lower(), 0.5)
