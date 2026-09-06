from datetime import UTC, datetime, timedelta

from compute_prereg import ids
from compute_prereg.calib import CalibrationStore
from compute_prereg.est.gpus import GPUS
from compute_prereg.models import (
    Job,
    JobStatus,
    Money,
    Offer,
    PriceSource,
    ProfilerSummary,
    TelemetrySummary,
)
from compute_prereg.recon import Reconciler
from tests.conftest import make_prereg


def _job(prereg, *, steps, run_seconds, mfu_target, data_frac=0.22, billed=None, workers=4):
    est = prereg.estimate
    gpu = "H100-SXM-80GB"
    peak = GPUS[gpu].peak_flops(est.job.dtype)
    # Choose the step count so the measured MFU lands where the test wants it.
    run_seconds = est.flops_per_step * (steps - 1) / (mfu_target * peak)
    t0 = datetime(2026, 9, 6, 12, 0, tzinfo=UTC)
    offer = Offer(offer_id=ids.offer_id(), provider="mock", underlying_provider="lambda", gpu_type=gpu, vram_gb=80,
                  price_per_hour=Money(usd=2.49, source=PriceSource.quoted), region="us-west-1")
    tel = TelemetrySummary(steps_completed=steps, first_step_at=t0 + timedelta(minutes=3),
                           last_step_at=t0 + timedelta(minutes=3, seconds=run_seconds), num_workers=workers,
                           peak_mem_gb=12.0,
                           profiler=ProfilerSummary(windows=2, steps_profiled=6, frac_compute=0.5, frac_memory_bound=0.09,
                                                    frac_data_loading=data_frac, frac_other=0.19))
    return Job(job_id=ids.job_id(), preregistration_id=prereg.preregistration_id, offer=offer, status=JobStatus.completed,
               provision_started_at=t0, started_at=t0 + timedelta(minutes=2, seconds=30),
               ended_at=t0 + timedelta(minutes=4, seconds=run_seconds), telemetry=tel, billed=billed)


def test_reconcile_attributes_and_calibrates(engine, example):
    _, p = make_prereg(engine, example, total_steps=1000)
    calib = CalibrationStore(None)
    r = Reconciler(calib).reconcile(p, _job(p, steps=1000, run_seconds=0, mfu_target=0.33,
                                            billed=Money(usd=7.72, source=PriceSource.quoted)))
    assert r.measured_mfu is not None and abs(r.measured_mfu - 0.33) < 0.01
    assert r.inside_mfu_band is False  # prior band on H100 is 35-55%
    assert r.measured_usd.usd == 7.72 and r.delta_usd_pct is not None
    causes = {a.cause: a for a in r.attribution}
    assert causes["data_loading"].fraction == 0.22 and "4 DataLoader workers" in causes["data_loading"].detail
    assert "memory_bound_ops" in causes and "launch_overhead" in causes
    assert any("num_workers=12" in rec for rec in r.recommendations)
    assert any("batch_tokens=512 is small" in rec for rec in r.recommendations)
    assert any("<40%" in rec for rec in r.recommendations)
    assert r.calibration_written
    e = calib.lookup(provider="lambda", gpu_type="H100-SXM-80GB", model_class="decoder-transformer",
                     framework=p.job.framework, dtype=p.job.dtype, batch_tokens=p.job.batch_tokens)
    assert e.n == 1 and e.source == "blended"
    assert "Billed $7.72 (quoted)" in r.narrative and "Attribution:" in r.narrative and "Calibration updated" in r.narrative


def test_reconcile_never_fabricates_billing(engine, example):
    _, p = make_prereg(engine, example, total_steps=100)
    r = Reconciler(CalibrationStore(None)).reconcile(p, _job(p, steps=100, run_seconds=0, mfu_target=0.5, data_frac=0.0))
    assert r.measured_usd is None and r.delta_usd_pct is None
    assert "Billed: unavailable" in r.narrative and "accrued at quoted price" in r.narrative
    assert r.accrued_usd.source == "derived"
    assert r.inside_mfu_band is True


def test_reconcile_skips_calibration_for_implausible_mfu(engine, example):
    _, p = make_prereg(engine, example, total_steps=100)
    calib = CalibrationStore(None)
    r = Reconciler(calib).reconcile(p, _job(p, steps=100, run_seconds=0, mfu_target=0.001))
    assert not r.calibration_written and calib.entries() == []
