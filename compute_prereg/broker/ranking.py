"""Rank fitting offers by user policy. Never picks alone — the caller decides per mode."""

from __future__ import annotations

from ..models import OfferFit, Policy


def _key(policy: Policy):
    def cheapest(f: OfferFit):
        return (f.projected_usd.hi if f.projected_usd else f.offer.price_per_hour.usd, _start(f), -f.offer.reliability_score)

    def fastest(f: OfferFit):
        return (_start(f), f.offer.price_per_hour.usd, -f.offer.reliability_score)

    def reliable(f: OfferFit):
        return (-f.offer.reliability_score, 0 if f.offer.tier != "spot" else 1, f.offer.price_per_hour.usd)

    def carbon(f: OfferFit):
        return (f.offer.carbon_score, f.offer.price_per_hour.usd)

    return {"cheapest": cheapest, "fastest-to-start": fastest, "most-reliable": reliable,
            "lowest-carbon": carbon}[policy]


def _start(f: OfferFit) -> float:
    return 0.0 if f.offer.availability == "now" else (f.offer.queue_eta_minutes or 1e6)


def rank(fits: list[OfferFit], policy: Policy = "cheapest") -> list[OfferFit]:
    """Fitting offers first (ranked by policy), then nearest misses (fewest failing dimensions)."""
    ok = sorted([f for f in fits if f.fits], key=_key(policy))
    miss = sorted([f for f in fits if not f.fits], key=lambda f: (len(f.failures), f.offer.price_per_hour.usd))
    return ok + miss
