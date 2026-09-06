"""Broker: fan out to adapters, normalize, fit against the envelope, rank."""

from __future__ import annotations

import asyncio
import logging

from ..errors import AdapterUnavailable, NoOfferFits
from ..models import OfferFit, Policy, Preregistration
from ..security import redact
from .base import ProviderAdapter
from .fit import fit_offer
from .ranking import rank

log = logging.getLogger(__name__)


class Broker:
    def __init__(self, adapters: list[ProviderAdapter]):
        self.adapters: dict[str, ProviderAdapter] = {a.name: a for a in adapters}
        self.last_errors: dict[str, str] = {}

    def adapter_for(self, provider: str) -> ProviderAdapter:
        try:
            return self.adapters[provider]
        except KeyError as e:
            raise AdapterUnavailable(f"no adapter named {provider!r} is configured") from e

    async def list_offers(self, prereg: Preregistration, policy: Policy = "cheapest",
                          providers: list[str] | None = None) -> list[OfferFit]:
        env = prereg.envelope
        names = providers or list(self.adapters)
        # Provider allow/deny on the envelope prunes the fan-out itself.
        if env.provider_allow:
            names = [n for n in names if n in env.provider_allow or self.adapters[n].is_aggregator]
        names = [n for n in names if n not in env.provider_deny]

        async def one(name: str):
            try:
                return name, await self.adapters[name].list_offers(env)
            except Exception as e:  # never let one adapter kill the hunt
                msg = redact(f"{type(e).__name__}: {e}")
                self.last_errors[name] = msg
                log.warning("adapter %s failed: %s", name, msg)
                return name, []

        results = await asyncio.gather(*(one(n) for n in names))
        fits: list[OfferFit] = []
        for name, offers in results:
            for o in offers:
                if o.provider != name:
                    o.provider = name
                if o.carbon_score == 0.5:
                    o.carbon_score = self.adapters[name].carbon_score(o.underlying_provider)
                fits.append(fit_offer(o, env, prereg.estimate))
        return rank(fits, policy)

    @staticmethod
    def best(fits: list[OfferFit]) -> OfferFit:
        ok = [f for f in fits if f.fits]
        if not ok:
            misses = "; ".join(f"{f.offer.provider}/{f.offer.gpu_type}: {','.join(f.failures)}" for f in fits[:5])
            raise NoOfferFits(f"no offer satisfies the envelope. Nearest misses: {misses or 'none seen'}")
        return ok[0]
