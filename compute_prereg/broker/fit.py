"""Envelope fit: which dimensions of an offer violate the preregistered envelope."""

from __future__ import annotations

from fnmatch import fnmatch

from ..est.gpus import gpu_matches_class
from ..models import Band, Envelope, Estimate, Offer, OfferFit


def fit_offer(offer: Offer, envelope: Envelope, estimate: Estimate | None = None) -> OfferFit:
    failures: list[str] = []
    details: dict[str, str] = {}

    if not gpu_matches_class(offer.gpu_type, envelope.gpu_classes):
        failures.append("gpu_class")
        details["gpu_class"] = f"{offer.gpu_type} not in {envelope.gpu_classes}"
    if not (envelope.min_gpu_count <= offer.count <= envelope.max_gpu_count):
        failures.append("gpu_count")
        details["gpu_count"] = f"{offer.count} outside [{envelope.min_gpu_count}, {envelope.max_gpu_count}]"
    if offer.vram_gb < envelope.min_vram_gb:
        failures.append("min_vram_gb")
        details["min_vram_gb"] = f"{offer.vram_gb:g} GB < {envelope.min_vram_gb:g} GB"
    if offer.tier == "spot" and not envelope.spot_ok:
        failures.append("spot_ok")
        details["spot_ok"] = "spot offer but envelope has spot_ok=false"
    if envelope.region_allow and not any(fnmatch(offer.region, p) for p in envelope.region_allow):
        failures.append("region_allow")
        details["region_allow"] = f"{offer.region} not in {envelope.region_allow}"
    providers = {offer.provider.lower()}
    if offer.underlying_provider:
        providers.add(offer.underlying_provider.lower())
    if envelope.provider_allow and not providers & {p.lower() for p in envelope.provider_allow}:
        failures.append("provider_allow")
        details["provider_allow"] = f"{sorted(providers)} not in {envelope.provider_allow}"
    if envelope.provider_deny and providers & {p.lower() for p in envelope.provider_deny}:
        failures.append("provider_deny")
        details["provider_deny"] = f"{sorted(providers)} denied"
    if envelope.interconnect and (offer.interconnect or "").lower() != envelope.interconnect.lower():
        failures.append("interconnect")
        details["interconnect"] = f"{offer.interconnect} != {envelope.interconnect}"

    # Projected cost. The contract number is the envelope's hours_max: if the job runs to
    # the tolerated maximum at this price, does it stay under the cap?
    est_hours = None
    if estimate and offer.gpu_type in estimate.per_gpu:
        est_hours = estimate.per_gpu[offer.gpu_type].hours
    hours = est_hours or Band(lo=envelope.hours_point, hi=envelope.hours_max)
    hours = Band(lo=hours.lo, hi=max(hours.hi, envelope.hours_max))
    usd = Band(lo=hours.lo * offer.price_per_hour.usd, hi=hours.hi * offer.price_per_hour.usd)
    cap_cost = envelope.hours_max * offer.price_per_hour.usd
    if cap_cost > envelope.dollar_cap:
        failures.append("dollar_cap")
        details["dollar_cap"] = (f"{envelope.hours_max:.2f} h x ${offer.price_per_hour.usd:.2f}/h = "
                                 f"${cap_cost:.2f} > cap ${envelope.dollar_cap:.2f}")
    return OfferFit(offer=offer, fits=not failures, failures=failures, details=details,
                    projected_hours=hours, projected_usd=usd)
