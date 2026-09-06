import asyncio

import pytest

from compute_prereg.errors import EnvelopeViolation, PolicyError
from compute_prereg.executor.keys import generate_job_keypair
from compute_prereg.models import JobStatus, PreregStatus
from tests.conftest import make_prereg


async def _run_to_end(engine, prereg_id, offer_id=None, timeout=120):
    job = await engine.provision(prereg_id, offer_id)
    job = await engine.executor.wait(job.job_id, timeout)
    return engine.job_status(job.job_id)


async def test_full_run_completes_and_reconciles(engine, mock_adapter, example):
    _, p = make_prereg(engine, example, total_steps=60)
    fits = await engine.list_offers(p.preregistration_id)
    best = next(f for f in fits if f.fits)
    assert best.offer.underlying_provider == "runpod" and best.offer.tier == "spot"  # cheapest, spot_ok
    job = await _run_to_end(engine, p.preregistration_id, best.offer.offer_id)
    assert job.status == JobStatus.completed and job.exit_code == 0
    assert job.telemetry.steps_completed == 60
    assert job.telemetry.profiler and job.telemetry.profiler.windows >= 1
    assert job.telemetry.profiler.measured_flops_per_step == pytest.approx(p.estimate.flops_per_step, rel=0.02)
    assert job.ssh_public_key.startswith("ssh-ed25519 ")
    assert mock_adapter.provisioned[0].provider_ref["authorized_key"] == job.ssh_public_key
    assert mock_adapter.terminated == [job.instance.instance_id]  # termination == revocation
    assert not (engine.executor.key_dir / f"{job.job_id}.key").exists()
    assert job.billed is not None and job.billed.source == "quoted"
    assert job.accrued.source == "derived"
    assert engine.ledger.status(p.preregistration_id) == PreregStatus.completed
    rec = engine.get_reconciliation(job.job_id)
    assert rec.steps_completed == 60 and rec.steps_planned == 60
    assert rec.measured_usd is not None
    assert "measured" in rec.narrative
    assert not rec.calibration_written  # a CPU run cannot calibrate an H100 band


async def test_provision_failure_cleans_leftovers(engine, mock_adapter, example):
    mock_adapter.fail_provision = True
    mock_adapter.leftover_on_fail = True
    _, p = make_prereg(engine, example)
    fits = await engine.list_offers(p.preregistration_id)
    job = await _run_to_end(engine, p.preregistration_id, next(f for f in fits if f.fits).offer.offer_id)
    assert job.status == JobStatus.failed and "capacity unavailable" in job.error
    assert any("terminated leftover" in e for e in job.provider_events)
    assert len(mock_adapter.terminated) == 1 and await mock_adapter.find_leftovers(job.job_id) == []
    assert engine.ledger.status(p.preregistration_id) == PreregStatus.failed
    assert not list(engine.executor.key_dir.glob("*.key"))


async def test_code_change_after_prereg_is_refused(engine, tmp_path, example):
    copy = tmp_path / "train.py"
    copy.write_text(example.read_text())
    _, p = make_prereg(engine, copy)
    fits = await engine.list_offers(p.preregistration_id)
    copy.write_text(example.read_text() + "\n# drift\n")
    job = await _run_to_end(engine, p.preregistration_id, next(f for f in fits if f.fits).offer.offer_id)
    assert job.status == JobStatus.failed and "changed since preregistration" in job.error
    assert job.instance is None  # never provisioned


async def test_locked_mode_breach_terminates(engine, mock_adapter, example):
    _, p = make_prereg(engine, example, total_steps=10_000_000, mode="locked", hours_point=0.0002, hours_tolerance=0.0)
    fits = await engine.list_offers(p.preregistration_id)
    job = await _run_to_end(engine, p.preregistration_id, next(f for f in fits if f.fits).offer.offer_id, timeout=60)
    assert job.status == JobStatus.breached and job.breach_reason.startswith("hours:")
    assert mock_adapter.terminated == [job.instance.instance_id]
    assert engine.ledger.status(p.preregistration_id) == PreregStatus.breached
    rec = engine.get_reconciliation(job.job_id)
    assert any("breached" in r for r in rec.recommendations)


async def test_terminate_running_job(engine, mock_adapter, example):
    _, p = make_prereg(engine, example, total_steps=10_000_000)
    fits = await engine.list_offers(p.preregistration_id)
    job = await engine.provision(p.preregistration_id, next(f for f in fits if f.fits).offer.offer_id)
    for _ in range(100):
        await asyncio.sleep(0.2)
        if engine.job_status(job.job_id).status == JobStatus.running:
            break
    q = engine.terminate_quote(job.job_id)
    assert q["destroys"] and q["provider"] == "runpod"
    await engine.terminate(job.job_id)
    job = await engine.executor.wait(job.job_id, 60)
    assert job.status == JobStatus.terminated
    assert mock_adapter.terminated == [job.instance.instance_id]
    with pytest.raises(PolicyError):
        engine.terminate_quote(job.job_id)


async def test_provision_refuses_nonconforming_and_foreign_offers(engine, example):
    _, p = make_prereg(engine, example, provider_deny=["lambda"])
    fits = await engine.list_offers(p.preregistration_id)
    miss = next(f for f in fits if "gpu_count" in f.failures)
    with pytest.raises(EnvelopeViolation) as ei:
        engine.provision_binding(p.preregistration_id, miss.offer.offer_id)
    assert "gpu_count" in ei.value.failures and "dollar_cap" in ei.value.failures
    _, other = make_prereg(engine, example)
    with pytest.raises(EnvelopeViolation, match="was listed for"):
        engine.provision_binding(other.preregistration_id, fits[0].offer.offer_id)
    with pytest.raises(PolicyError, match="mode=ask"):
        engine.provision_binding(p.preregistration_id, None)


async def test_bounded_mode_autoselects_cheapest(engine, example):
    _, p = make_prereg(engine, example, mode="bounded")
    await engine.list_offers(p.preregistration_id)
    bound, summary = engine.provision_binding(p.preregistration_id, None)
    assert summary["provider"] == "runpod" and summary["tier"] == "spot"
    assert bound["envelope_hash"] == p.record_hash


def test_job_keys_are_unique_and_private(tmp_path):
    kp = generate_job_keypair("job_a", tmp_path / "keys")
    assert kp.private_key_path.stat().st_mode & 0o777 == 0o600
    assert kp.public_key.startswith("ssh-ed25519 ") and kp.public_key.endswith("prereg-job_a")
    with pytest.raises(FileExistsError):
        generate_job_keypair("job_a", tmp_path / "keys")
