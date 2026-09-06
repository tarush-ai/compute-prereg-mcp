import pytest

from compute_prereg.broker import Broker, ProviderAdapter, fit_offer, rank
from compute_prereg.broker.adapters.mock import MockAdapter
from compute_prereg.errors import NoOfferFits
from compute_prereg.models import Envelope, Money, Offer, PriceSource


def offer(**kw):
    base = dict(offer_id="o", provider="mock", gpu_type="H100-SXM-80GB", vram_gb=80, tier="on-demand",
                price_per_hour=Money(usd=2.0, source=PriceSource.quoted), region="us-east-1")
    base.update(kw)
    return Offer(**base)


ENV = Envelope(gpu_classes=["H100"], min_vram_gb=40, hours_point=2.0, hours_tolerance=0.5, dollar_cap=10.0,
               region_allow=["us-*"], provider_deny=["vast"])


def test_fit_dimensions():
    assert fit_offer(offer(), ENV).fits
    f = fit_offer(offer(gpu_type="A100-SXM-80GB", tier="spot", region="eu-1", count=2, vram_gb=24,
                        underlying_provider="vast", price_per_hour=Money(usd=5, source=PriceSource.quoted)), ENV)
    assert set(f.failures) == {"gpu_class", "gpu_count", "min_vram_gb", "spot_ok", "region_allow", "provider_deny", "dollar_cap"}
    assert "3.00 h x $5.00/h" in f.details["dollar_cap"]
    # hours_max x price is the binding number: 3 h x $3.5 = 10.5 > 10
    assert fit_offer(offer(price_per_hour=Money(usd=3.5, source=PriceSource.quoted)), ENV).failures == ["dollar_cap"]
    assert fit_offer(offer(price_per_hour=Money(usd=3.3, source=PriceSource.quoted)), ENV).fits


def test_ranking_policies():
    fits = [fit_offer(o, ENV) for o in (
        offer(offer_id="cheap-queue", price_per_hour=Money(usd=1.0, source=PriceSource.quoted), availability="queue",
              queue_eta_minutes=30, reliability_score=0.6, carbon_score=0.9),
        offer(offer_id="now", price_per_hour=Money(usd=2.0, source=PriceSource.quoted), reliability_score=0.9, carbon_score=0.5),
        offer(offer_id="green", price_per_hour=Money(usd=2.5, source=PriceSource.quoted), reliability_score=0.7, carbon_score=0.1),
        offer(offer_id="miss", gpu_type="A100-SXM-80GB"),
    )]
    assert [f.offer.offer_id for f in rank(fits, "cheapest")] == ["cheap-queue", "now", "green", "miss"]
    assert rank(fits, "fastest-to-start")[0].offer.offer_id == "now"
    assert rank(fits, "most-reliable")[0].offer.offer_id == "now"
    assert rank(fits, "lowest-carbon")[0].offer.offer_id == "green"
    assert rank(fits)[-1].offer.offer_id == "miss"


class Broken(ProviderAdapter):
    name = "broken"

    async def list_offers(self, envelope):
        raise RuntimeError("api key rp_ABCDEFGHIJKLMNOPQRSTUVWXYZ rejected")

    async def provision(self, offer, ssh_public_key, job_id):
        raise NotImplementedError

    async def terminate(self, instance):
        raise NotImplementedError


async def test_broker_survives_failing_adapter(engine, example):
    from tests.conftest import make_prereg

    _, p = make_prereg(engine, example)
    b = Broker([MockAdapter(), Broken()])
    fits = await b.list_offers(p)
    assert any(f.fits for f in fits)
    assert "broken" in b.last_errors and "rp_ABCDEFGHIJKLMNOPQRSTUVWXYZ" not in b.last_errors["broken"]
    assert "[REDACTED]" in b.last_errors["broken"]
    with pytest.raises(NoOfferFits, match="Nearest misses"):
        Broker.best([f for f in fits if not f.fits])
