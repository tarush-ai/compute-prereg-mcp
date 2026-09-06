"""Engine: wires estimator, ledger, calibration, broker, confirmer, executor and reconciler.
The MCP server and the CLI are thin layers over this."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .broker import Broker, ProviderAdapter, fit_offer
from .broker.adapters import autodetect, build_adapter
from .calib import CalibrationStore
from .confirm import Confirmer
from .errors import (
    AdapterUnavailable,
    EnvelopeViolation,
    NoOfferFits,
    NotFound,
    PolicyError,
    PreregError,
)
from .est import Estimator
from .executor import Executor
from .ledger import Ledger, Signer
from .models import (
    Confidence,
    Envelope,
    Estimate,
    Job,
    JobStatus,
    Mode,
    OfferFit,
    Policy,
    Preregistration,
    PreregStatus,
    utcnow,
)
from .recon import Reconciler
from .security import redact

DEFAULT_DATA_DIR = Path(os.environ.get("COMPUTE_PREREG_DATA_DIR", "~/.compute-prereg")).expanduser()
PROVISIONABLE = {PreregStatus.open, PreregStatus.failed, PreregStatus.breached, PreregStatus.terminated}


@dataclass
class EngineConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    adapters: list[str] | None = None  # None -> autodetect from environment
    demo: bool = False  # add the mock adapter (local execution)
    poll_interval: float = 2.0
    confirmation_ttl: int = 600
    reserve_gb: float = 4.0
    profile_every: int = 100
    profile_steps: int = 3
    extra: dict[str, Any] = field(default_factory=dict)


class Engine:
    def __init__(self, config: EngineConfig | None = None, *, adapters: list[ProviderAdapter] | None = None,
                 in_memory: bool = False, launcher_factory=None):
        self.config = cfg = config or EngineConfig()
        cfg.data_dir = Path(cfg.data_dir)
        if not in_memory:
            cfg.data_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(None if in_memory else cfg.data_dir / "ledger.sqlite",
                             Signer(None if in_memory else cfg.data_dir / "ledger_signing.key"))
        self.calibration = CalibrationStore(None if in_memory else cfg.data_dir / "calibration.sqlite")
        self.estimator = Estimator(self.calibration, reserve_gb=cfg.reserve_gb)
        self.adapter_errors: dict[str, str] = {}
        self.broker = Broker(adapters if adapters is not None else self._build_adapters())
        self.confirmer = Confirmer(self.ledger, ttl_seconds=cfg.confirmation_ttl)
        self.reconciler = Reconciler(self.calibration)
        self.executor = Executor(self.ledger, self.broker, self.reconciler, data_dir=cfg.data_dir,
                                 poll_interval=cfg.poll_interval, profile_every=cfg.profile_every,
                                 profile_steps=cfg.profile_steps, launcher_factory=launcher_factory)

    def _build_adapters(self) -> list[ProviderAdapter]:
        names = list(self.config.adapters) if self.config.adapters is not None else autodetect()
        if self.config.demo and "mock" not in names:
            names.append("mock")
        out = []
        for n in names:
            try:
                out.append(build_adapter(n))
            except AdapterUnavailable as e:
                self.adapter_errors[n] = redact(str(e))
            except Exception as e:  # noqa: BLE001
                self.adapter_errors[n] = redact(f"{type(e).__name__}: {e}")
        return out

    # ================================================================== read tier
    def estimate_job(self, *, entrypoint: str | None = None, config: dict | None = None, model_spec: dict | None = None,
                     gpu_candidates: list[str] | None = None, total_steps: int | None = None) -> Estimate:
        est = self.estimator.estimate(entrypoint=entrypoint, config=config, model_spec=model_spec,
                                      gpu_candidates=gpu_candidates, total_steps=total_steps)
        self.ledger.put_estimate(est)
        return est

    @staticmethod
    def suggest_envelope(est: Estimate, gpu: str | None = None, *, tolerance: float = 0.25, cap_factor: float = 1.5) -> Envelope:
        """A starting envelope from an estimate: the recommended GPU's class, its memory need,
        its hours midpoint, and a cap at cap_factor x the high end of the $ band."""
        gpu = gpu or est.recommended_gpu or next(iter(est.per_gpu))
        pg = est.per_gpu[gpu]
        from .est.gpus import GPUS

        return Envelope(
            gpu_classes=[GPUS[gpu].gpu_class], min_vram_gb=float(math.ceil(est.peak_memory.required_gb)),
            hours_point=round(pg.hours.mid, 2), hours_tolerance=tolerance,
            dollar_cap=round(max(1.0, pg.usd_at_ref_price.hi * cap_factor), 2), spot_ok=False)

    def get_preregistration(self, prereg_id: str) -> Preregistration:
        return self.ledger.get_preregistration(prereg_id)

    def list_preregistrations(self, status: str | None = None) -> list[Preregistration]:
        return self.ledger.list_preregistrations(PreregStatus(status) if status else None)

    async def list_offers(self, prereg_id: str, policy: Policy = "cheapest", providers: list[str] | None = None) -> list[OfferFit]:
        prereg = self.ledger.get_preregistration(prereg_id)
        if prereg.status == PreregStatus.superseded:
            raise PolicyError(f"{prereg_id} is superseded; use its successor")
        if not self.broker.adapters:
            raise AdapterUnavailable(
                "no provider adapters configured. Set RUNPOD_API_KEY, install skypilot/modal, or set "
                "COMPUTE_PREREG_ADAPTERS; COMPUTE_PREREG_DEMO=1 enables the local mock adapter. "
                f"Adapter errors: {self.adapter_errors or 'none'}")
        fits = await self.broker.list_offers(prereg, policy, providers)
        self.ledger.put_offers(prereg_id, [f.offer for f in fits])
        return fits

    def get_reconciliation(self, job_id: str):
        return self.ledger.get_reconciliation(job_id)

    def get_calibration(self, provider: str | None = None, gpu: str | None = None, model_class: str | None = None):
        entries = self.calibration.entries(provider=provider, gpu_type=gpu, model_class=model_class)
        priors = []
        if gpu:
            priors.append(self.calibration.priors(gpu, model_class or "decoder-transformer"))
        return {"measured": entries, "priors": priors}

    def job_status(self, job_id: str) -> Job:
        return self.ledger.get_job(job_id)

    # ================================================================== write tier
    def preregister_quote(self, estimate_id: str, envelope: dict, mode: str) -> dict[str, Any]:
        est = self.ledger.get_estimate(estimate_id)
        env = Envelope.model_validate(envelope)
        m = Mode(mode)
        fitting = [g for g, p in est.per_gpu.items() if p.fits_memory]
        notes = []
        if est.confidence == Confidence.low:
            notes.append("estimate confidence is LOW: provisioning in bounded/locked mode will require an explicit offer")
        if est.peak_memory.required_gb > env.min_vram_gb:
            notes.append(f"min_vram_gb={env.min_vram_gb:g} is below the estimated need of {est.peak_memory.required_gb:.1f} GB")
        return {"estimate_id": estimate_id, "mode": m, "envelope": env.model_dump(), "hours_max": round(env.hours_max, 3),
                "dollar_cap": env.dollar_cap, "estimate_confidence": est.confidence, "gpus_that_fit": fitting, "notes": notes}

    def preregister(self, estimate_id: str, envelope: dict, mode: str) -> Preregistration:
        est = self.ledger.get_estimate(estimate_id)
        return self.ledger.create_preregistration(est, Envelope.model_validate(envelope), Mode(mode))

    def amend_quote(self, prereg_id: str, changes: dict, actor: str) -> dict[str, Any]:
        old = self.ledger.get_preregistration(prereg_id)
        # Dry-run the policy so the quote already tells the agent what it may not do.
        try:
            self._amend_dry_run(old, changes, actor)
        except PolicyError as e:
            return {"preregistration_id": prereg_id, "changes": changes, "allowed": False, "reason": str(e)}
        return {"preregistration_id": prereg_id, "changes": changes, "allowed": True,
                "current_envelope": old.envelope.model_dump(), "actor": actor}

    def amend(self, prereg_id: str, changes: dict, actor: str, note: str | None = None) -> Preregistration:
        return self.ledger.amend(prereg_id, changes, actor=actor, note=note)

    def _amend_dry_run(self, old: Preregistration, changes: dict, actor: str) -> None:
        env = old.envelope.model_dump()
        for k, v in changes.items():
            if k != "mode":
                env[k] = v
        new_env = Envelope.model_validate(env)
        if actor != "human":
            widened = [n for n, cond in (("dollar_cap", new_env.dollar_cap > old.envelope.dollar_cap),
                                         ("hours", new_env.hours_max > old.envelope.hours_max),
                                         ("max_gpu_count", new_env.max_gpu_count > old.envelope.max_gpu_count)) if cond]
            if widened:
                raise PolicyError(f"agent may not raise {', '.join(widened)}; a human must do this via the CLI")

    # ================================================================== money tier
    def provision_binding(self, prereg_id: str, offer_id: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Resolve (bound params, summary) for provision. The token is bound to
        hash(offer ∪ envelope): a re-priced offer or an amended envelope voids it."""
        prereg = self.ledger.get_preregistration(prereg_id)
        if prereg.status not in PROVISIONABLE:
            raise PolicyError(f"{prereg_id} is {prereg.status}; cannot provision")
        if offer_id is None:
            if prereg.mode == Mode.ask:
                raise PolicyError("mode=ask: the human picks the offer; pass offer_id from list_offers")
            if prereg.estimate.confidence == Confidence.low:
                raise PolicyError("estimate confidence is low; auto-selection is disabled — pass an explicit offer_id")
            fits = [fit_offer(o, prereg.envelope, prereg.estimate) for o in self.ledger.offers_for(prereg_id)
                    if o.expires_at is None or o.expires_at > utcnow()]
            from .broker.ranking import rank

            ranked = rank(fits)
            if not ranked or not ranked[0].fits:
                raise NoOfferFits("no fresh conforming offer cached; call list_offers first")
            offer = ranked[0].offer
        else:
            offer, owner = self.ledger.get_offer(offer_id)
            if owner != prereg_id:
                raise EnvelopeViolation(f"offer {offer_id} was listed for {owner}, not {prereg_id}", ["offer_owner"])
            if offer.expires_at and offer.expires_at < utcnow():
                raise EnvelopeViolation(f"offer {offer_id} expired at {offer.expires_at.isoformat()}; call list_offers again", ["expired"])
        fit = fit_offer(offer, prereg.envelope, prereg.estimate)
        if not fit.fits:
            raise EnvelopeViolation(
                "offer exceeds the envelope in: " + ", ".join(f"{k} ({fit.details.get(k, '')})" for k in fit.failures)
                + ". It cannot be confirmed; amend the preregistration explicitly (new record) or pick another offer.",
                fit.failures)
        bound = {"envelope_id": prereg_id, "offer_id": offer.offer_id, "offer": offer.binding_view(),
                 "envelope_hash": prereg.record_hash, "dollar_cap": prereg.envelope.dollar_cap}
        summary = {
            "envelope_id": prereg_id, "offer_id": offer.offer_id, "mode": prereg.mode,
            "provider": offer.underlying_provider or offer.provider, "via": offer.provider if offer.underlying_provider else None,
            "gpu": f"{offer.count}x {offer.gpu_type}", "tier": offer.tier, "region": offer.region,
            "price_per_hour": offer.price_per_hour.model_dump(), "availability": offer.availability,
            "queue_eta_minutes": offer.queue_eta_minutes,
            "projected_hours": fit.projected_hours.as_list() if fit.projected_hours else None,
            "projected_usd": fit.projected_usd.as_list() if fit.projected_usd else None,
            "hours_max": round(prereg.envelope.hours_max, 3), "dollar_cap": prereg.envelope.dollar_cap,
            "cost_at_hours_max": round(prereg.envelope.hours_max * offer.price_per_hour.usd, 2),
            "estimate_confidence": prereg.estimate.confidence,
            "on_breach": {"ask": "stop, report, ask", "bounded": "stop, report, ask",
                          "locked": "checkpoint, terminate, report; cannot extend"}[prereg.mode],
            "ssh": "a fresh ed25519 keypair is generated for this job; only its public key is registered, on this instance only",
        }
        if prereg.estimate.confidence == Confidence.low:
            summary["warning"] = "estimate confidence is LOW; the hours/$ bands may be off by more than 2x"
        return bound, summary

    async def provision(self, prereg_id: str, offer_id: str | None) -> Job:
        bound, _ = self.provision_binding(prereg_id, offer_id)
        prereg = self.ledger.get_preregistration(prereg_id)
        offer, _ = self.ledger.get_offer(bound["offer_id"])
        active = [j for j in self.ledger.jobs_for(prereg_id) if self.executor.is_active(j.job_id)]
        if active:
            raise PolicyError(f"job {active[0].job_id} is already running under {prereg_id}")
        from . import ids

        job = Job(job_id=ids.job_id(), preregistration_id=prereg_id, offer=offer, mode=prereg.mode)
        self.ledger.put_job(job)
        self.executor.start(prereg, offer, job)
        return job

    def terminate_quote(self, job_id: str) -> dict[str, Any]:
        job = self.ledger.get_job(job_id)
        if not self.executor.is_active(job_id):
            raise PolicyError(f"job {job_id} is not running (status {job.status})")
        return {
            "job_id": job_id, "status": job.status, "instance": job.instance.instance_id if job.instance else None,
            "provider": job.offer.underlying_provider or job.offer.provider, "gpu": job.offer.gpu_type,
            "elapsed_hours": round(job.elapsed_hours, 3), "steps_completed": job.telemetry.steps_completed,
            "accrued_usd": job.accrued.model_dump() if job.accrued else None,
            "price_per_hour": job.offer.price_per_hour.usd,
            "destroys": "the instance and everything on it not checkpointed; the job SSH key is revoked with it",
            "checkpoint": f"a checkpoint request is sent first; the script has {self.executor.checkpoint_grace:.0f}s to save",
        }

    async def terminate(self, job_id: str) -> Job:
        if not self.executor.is_active(job_id):
            raise PolicyError(f"job {job_id} is not running")
        self.executor.request_stop(job_id)
        return self.ledger.get_job(job_id)

    # ================================================================== misc
    def describe(self) -> dict[str, Any]:
        return {"data_dir": str(self.config.data_dir), "adapters": sorted(self.broker.adapters),
                "adapter_errors": self.adapter_errors, "ledger_public_key": self.ledger.signer.public_key_hex,
                "active_jobs": [j.job_id for j in self.ledger.list_jobs(active_only=True) if self.executor.is_active(j.job_id)]}

    def close(self) -> None:
        self.ledger.close()
        self.calibration.close()


__all__ = ["Engine", "EngineConfig", "PreregError", "NotFound", "JobStatus"]
