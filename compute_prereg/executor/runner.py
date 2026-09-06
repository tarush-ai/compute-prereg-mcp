"""Executor: per-job key -> provision -> sync -> launch -> watchdog -> terminate -> reconcile.

Termination is the revocation event for the job key. The watchdog enforces the envelope
(hours, dollars) in every mode; what differs by mode is what happens next (§6):
  ask / bounded : stop, report, wait for the human
  locked        : checkpoint, terminate, report; cannot extend
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from ..errors import ProvisionError
from ..est.tracer import sha256_file
from ..models import (
    Instance,
    Job,
    JobStatus,
    Mode,
    Money,
    Offer,
    Preregistration,
    PreregStatus,
    PriceSource,
    utcnow,
)
from ..security import redact
from .keys import destroy_job_key, generate_job_keypair
from .launchers import Launcher, LocalLauncher, ModalLauncher, SSHLauncher
from .telemetry import summarize

log = logging.getLogger(__name__)

CHECKPOINT_GRACE_SECONDS = 30.0


class Executor:
    def __init__(self, ledger, broker, reconciler, *, data_dir: str | Path, poll_interval: float = 2.0,
                 profile_every: int = 100, profile_steps: int = 3, checkpoint_grace: float = CHECKPOINT_GRACE_SECONDS,
                 launcher_factory: Callable[[Instance, Path], Launcher] | None = None,
                 clock: Callable[[], datetime] = utcnow):
        self.ledger, self.broker, self.reconciler = ledger, broker, reconciler
        self.data_dir = Path(data_dir)
        self.key_dir = self.data_dir / "keys"
        self.work_dir = self.data_dir / "work"
        self.poll_interval = poll_interval
        self.profile_every, self.profile_steps = profile_every, profile_steps
        self.checkpoint_grace = checkpoint_grace
        self._launcher_factory = launcher_factory
        self.clock = clock
        self._stop_requested: set[str] = set()
        self._tasks: dict[str, asyncio.Task] = {}

    # ------------------------------------------------------------------ launchers
    def _launcher(self, instance: Instance, key_path: Path) -> Launcher:
        if self._launcher_factory:
            return self._launcher_factory(instance, key_path)
        if instance.kind == "local":
            return LocalLauncher(self.work_dir)
        if instance.kind == "ssh":
            return SSHLauncher(key_path, self.data_dir / "known_hosts")
        if instance.kind == "function":
            return ModalLauncher(self.work_dir)
        raise ProvisionError(f"no launcher for instance kind {instance.kind}")

    # ------------------------------------------------------------------ public
    def start(self, prereg: Preregistration, offer: Offer, job: Job) -> asyncio.Task:
        task = asyncio.create_task(self.run_job(prereg, offer, job), name=f"job:{job.job_id}")
        self._tasks[job.job_id] = task
        return task

    def request_stop(self, job_id: str) -> None:
        self._stop_requested.add(job_id)

    def is_active(self, job_id: str) -> bool:
        t = self._tasks.get(job_id)
        return t is not None and not t.done()

    async def wait(self, job_id: str, timeout: float | None = None) -> Job:
        t = self._tasks[job_id]
        return await asyncio.wait_for(t, timeout)

    # ------------------------------------------------------------------ main loop
    async def run_job(self, prereg: Preregistration, offer: Offer, job: Job) -> Job:
        adapter = self.broker.adapter_for(offer.provider)
        env = prereg.envelope
        entry = Path(prereg.job.entrypoint) if prereg.job.entrypoint else None
        launcher: Launcher | None = None

        def save(status: JobStatus | None = None):
            if status is not None:
                job.status = status
            self.ledger.put_job(job)

        # 0. The code must be what was preregistered.
        if entry is None or not entry.exists():
            job.error = "preregistration has no runnable entrypoint (spec-only estimate); nothing to execute"
            save(JobStatus.failed)
            self.ledger.set_status(prereg.preregistration_id, PreregStatus.failed, job.error)
            return job
        if prereg.job.code_hash and sha256_file(entry) != prereg.job.code_hash:
            job.error = f"{entry.name} changed since preregistration; re-estimate and re-preregister"
            save(JobStatus.failed)
            self.ledger.set_status(prereg.preregistration_id, PreregStatus.failed, job.error)
            return job

        # 1. Per-job key.
        kp = generate_job_keypair(job.job_id, self.key_dir)
        job.ssh_public_key = kp.public_key
        job.provision_started_at = self.clock()
        save(JobStatus.provisioning)
        self.ledger.set_status(prereg.preregistration_id, PreregStatus.provisioning, job.job_id)

        try:
            # 2. Provision. On failure: hunt for leftovers, terminate anything billable, report.
            try:
                instance = await adapter.provision(offer, kp.public_key, job.job_id)
            except Exception as e:
                job.error = redact(f"provision failed: {type(e).__name__}: {e}")
                for left in await _safe(adapter.find_leftovers(job.job_id), []):
                    try:
                        await adapter.terminate(left)
                        job.provider_events.append(f"terminated leftover instance {left.instance_id}")
                    except Exception as te:  # noqa: BLE001
                        job.provider_events.append(redact(f"COULD NOT terminate leftover {left.instance_id}: {te} — check the provider console"))
                save(JobStatus.failed)
                self.ledger.set_status(prereg.preregistration_id, PreregStatus.failed, job.error)
                return job
            job.instance = instance
            job.provider_events.append(f"provisioned {instance.instance_id} on {offer.provider}")
            launcher = self._launcher(instance, kp.private_key_path)

            # 3. Sync + launch.
            save(JobStatus.syncing)
            try:
                await launcher.sync(instance, entry.parent)
                run_env = {"PREREG_JOB_ID": job.job_id, "PREREG_CHECKPOINT_EVERY": str(env.checkpoint_interval_steps),
                           "PREREG_TOTAL_STEPS": str(prereg.job.total_steps)}
                run_env.update({k: str(v) for k, v in prereg.job.extra.items() if k.isupper()})
                await launcher.start(instance, entry.name, [], run_env, self.profile_every, self.profile_steps)
            except Exception as e:
                job.error = redact(f"launch failed: {type(e).__name__}: {e}")
                save(JobStatus.failed)
                raise _Finish() from None
            job.started_at = self.clock()
            save(JobStatus.running)
            self.ledger.set_status(prereg.preregistration_id, PreregStatus.running, job.job_id)

            # 4. Watchdog.
            gap_start: datetime | None = None
            while True:
                await asyncio.sleep(self.poll_interval)
                state = await launcher.poll(instance)
                try:
                    records = await launcher.read_telemetry(instance)
                    if gap_start is not None:
                        job.telemetry.gaps.append([gap_start, self.clock()])
                        gap_start = None
                    job.telemetry = summarize(records, job.telemetry)
                except Exception as e:  # provider/API error mid-job: job continues, gap logged
                    if gap_start is None:
                        gap_start = self.clock()
                        job.provider_events.append(redact(f"telemetry gap started: {e}"))
                job.accrued = Money(usd=round(job.elapsed_hours * offer.price_per_hour.usd, 4),
                                    source=PriceSource.derived, note="elapsed x quoted price; not a bill")

                breach = self._breach(job, env)
                if breach:
                    job.breach_reason = breach
                    await self._checkpoint_and_stop(launcher, instance, job)
                    job.exit_code = None
                    save(JobStatus.breached)
                    raise _Finish()
                if job.job_id in self._stop_requested:
                    await self._checkpoint_and_stop(launcher, instance, job)
                    save(JobStatus.terminated)
                    raise _Finish()
                if not state.running:
                    job.exit_code = state.exit_code
                    save(JobStatus.completed if state.exit_code == 0 else JobStatus.failed)
                    if state.exit_code != 0:
                        job.error = f"script exited with {state.exit_code}" + (f": {state.detail}" if state.detail else "")
                    raise _Finish()
                pstate = await _safe(adapter.instance_state(instance), "running")
                if pstate == "preempted":
                    job.preemptions += 1
                    job.provider_events.append("instance preempted")
                    save(JobStatus.preempted)
                    raise _Finish()
                save()
        except _Finish:
            pass
        except Exception as e:  # noqa: BLE001
            job.error = redact(f"executor error: {type(e).__name__}: {e}")
            save(JobStatus.failed)
        finally:
            job.ended_at = job.ended_at or self.clock()
            inst = job.instance
            if inst is not None:
                if launcher is not None:
                    try:
                        records = await launcher.read_telemetry(inst)
                        job.telemetry = summarize(records, job.telemetry)
                    except Exception:
                        pass
                    await _safe(launcher.cleanup(inst), None)
                job.billed = await _safe(adapter.usage(inst), None)
                if job.billed is None:
                    job.provider_events.append("billing: unavailable from provider usage API")
                # Termination is key revocation.
                try:
                    await adapter.terminate(inst)
                    job.provider_events.append(f"terminated {inst.instance_id} (job key revoked)")
                except Exception as e:  # noqa: BLE001
                    job.provider_events.append(redact(f"TERMINATE FAILED for {inst.instance_id}: {e} — instance may still bill"))
            destroy_job_key(job.job_id, self.key_dir)
            self._stop_requested.discard(job.job_id)
            if job.status in (JobStatus.running, JobStatus.syncing, JobStatus.provisioning):
                job.status = JobStatus.failed
            save()
            self.ledger.set_status(prereg.preregistration_id, _prereg_status(job.status), job.breach_reason or job.error)
            try:
                rec = self.reconciler.reconcile(prereg, job)
                self.ledger.put_reconciliation(rec)
            except Exception as e:  # noqa: BLE001
                log.exception("reconcile failed")
                job.provider_events.append(redact(f"reconcile failed: {e}"))
                save()
        return job

    # ------------------------------------------------------------------ helpers
    def _breach(self, job: Job, env) -> str | None:
        hours = job.elapsed_hours
        if hours > env.hours_max:
            return f"hours: {hours:.2f} h elapsed > {env.hours_max:.2f} h allowed ({env.hours_point} h +{env.hours_tolerance:.0%})"
        if job.accrued and job.accrued.usd > env.dollar_cap:
            return f"dollars: accrued ${job.accrued.usd:.2f} > cap ${env.dollar_cap:.2f}"
        return None

    async def _checkpoint_and_stop(self, launcher: Launcher, instance: Instance, job: Job) -> None:
        job.status = JobStatus.checkpointing
        self.ledger.put_job(job)
        await _safe(launcher.request_checkpoint(instance), None)
        deadline = asyncio.get_running_loop().time() + self.checkpoint_grace
        while asyncio.get_running_loop().time() < deadline:
            st = await launcher.poll(instance)
            if not st.running:
                job.exit_code = st.exit_code
                return
            await asyncio.sleep(min(1.0, self.poll_interval))
        await _safe(launcher.stop(instance), None)


class _Finish(Exception):
    pass


async def _safe(coro, default):
    try:
        return await coro
    except Exception:  # noqa: BLE001
        return default


def _prereg_status(js: JobStatus) -> PreregStatus:
    return {
        JobStatus.completed: PreregStatus.completed, JobStatus.failed: PreregStatus.failed,
        JobStatus.terminated: PreregStatus.terminated, JobStatus.breached: PreregStatus.breached,
        JobStatus.preempted: PreregStatus.failed,
    }.get(js, PreregStatus.failed)


__all__ = ["Executor", "Mode", "UTC"]
