"""Reconciler: predicted vs measured (FLOPs, hours, $, MFU), gap attribution, and the
MFU write-back that makes the next estimate better."""

from __future__ import annotations

from .. import ids
from ..est.gpus import GPUS
from ..est.priors import prior_band
from ..est.roofline import per_gpu_estimate, training_bytes_per_step
from ..models import Attribution, Band, Job, Money, Preregistration, PriceSource, Reconciliation

MIN_STEPS_FOR_CALIBRATION = 10
MIN_PLAUSIBLE_MFU = 0.005


class Reconciler:
    def __init__(self, calibration):
        self.calibration = calibration

    def reconcile(self, prereg: Preregistration, job: Job) -> Reconciliation:
        est, env, offer = prereg.estimate, prereg.envelope, job.offer
        gpu = offer.gpu_type
        spec = GPUS.get(gpu)
        pg = est.per_gpu.get(gpu)
        if pg is None and spec is not None:
            pg = per_gpu_estimate(gpu, flops_per_step=est.flops_per_step,
                                  bytes_per_step=training_bytes_per_step(est.param_bytes, 0.0, 0.0),
                                  total_steps=est.job.total_steps, dtype=est.job.dtype,
                                  required_mem_gb=est.peak_memory.required_gb,
                                  mfu_band=prior_band(spec.gpu_class, est.job.model_class, est.job.batch_tokens),
                                  mfu_source="prior")
        pred_hours = pg.hours if pg else Band(lo=env.hours_point, hi=env.hours_max)
        pred_mfu = pg.mfu_band if pg else Band(lo=0.2, hi=0.45)
        price = offer.price_per_hour.usd
        pred_usd = Band(lo=pred_hours.lo * price, hi=pred_hours.hi * price)

        tel = job.telemetry
        measured_hours = job.elapsed_hours
        launch_overhead_h = 0.0
        if job.started_at and job.provision_started_at:
            launch_overhead_h = max(0.0, (job.started_at - job.provision_started_at).total_seconds() / 3600)
        steps = tel.steps_completed
        flops_per_step = est.flops_per_step
        measured_flops = None
        if tel.profiler and tel.profiler.measured_flops_per_step:
            flops_per_step = tel.profiler.measured_flops_per_step
        if steps:
            measured_flops = flops_per_step * steps

        measured_mfu = None
        if steps >= 2 and tel.first_step_at and tel.last_step_at and spec is not None:
            run_s = (tel.last_step_at - tel.first_step_at).total_seconds()
            if run_s > 0:
                achieved = flops_per_step * (steps - 1) / run_s
                measured_mfu = min(1.0, achieved / spec.peak_flops(est.job.dtype))

        accrued = job.accrued or (Money(usd=round(measured_hours * price, 4), source=PriceSource.derived,
                                        note="elapsed x quoted price") if measured_hours else None)
        billed = job.billed  # None == unavailable; never fabricated

        def pct(measured, predicted_mid):
            return None if measured is None or not predicted_mid else round((measured - predicted_mid) / predicted_mid * 100, 1)

        d_hours = pct(measured_hours, pred_hours.mid)
        d_usd = pct(billed.usd if billed else None, pred_usd.mid)
        d_mfu = pct(measured_mfu, pred_mfu.mid)

        attribution = self._attribute(job, measured_hours, launch_overhead_h)
        recs = self._recommend(prereg, job, measured_mfu, attribution, spec)
        rec = Reconciliation(
            reconciliation_id=ids.reconciliation_id(), job_id=job.job_id, preregistration_id=prereg.preregistration_id,
            gpu=gpu, provider=offer.underlying_provider or offer.provider,
            predicted_hours=pred_hours, predicted_usd=pred_usd, predicted_mfu=pred_mfu, predicted_flops=est.flops_total,
            measured_hours=round(measured_hours, 4), measured_usd=billed, accrued_usd=accrued,
            measured_mfu=round(measured_mfu, 4) if measured_mfu is not None else None,
            measured_flops=measured_flops, steps_completed=steps, steps_planned=est.job.total_steps,
            delta_hours_pct=d_hours, delta_usd_pct=d_usd, delta_mfu_pct=d_mfu,
            inside_hours_band=(pred_hours.lo <= measured_hours <= pred_hours.hi) if steps else None,
            inside_mfu_band=(pred_mfu.lo <= measured_mfu <= pred_mfu.hi) if measured_mfu is not None else None,
            attribution=attribution, recommendations=recs, telemetry_gaps=len(tel.gaps),
        )

        # Write-back: the asset. Guard against implausible values (a CPU-only run measured
        # against a GPU peak lands near 0 and must not poison the store).
        if measured_mfu is not None and measured_mfu >= MIN_PLAUSIBLE_MFU and steps >= MIN_STEPS_FOR_CALIBRATION \
                and self.calibration is not None and job.status.value in ("completed", "terminated", "breached"):
            key = self.calibration.make_key(provider=offer.underlying_provider or offer.provider, gpu_type=gpu,
                                            model_class=est.job.model_class, framework=est.job.framework,
                                            dtype=est.job.dtype, batch_tokens=est.job.batch_tokens)
            self.calibration.record(key, measured_mfu, job.job_id)
            rec.calibration_written = True
        rec.narrative = self._narrative(rec, job)
        return rec

    # ------------------------------------------------------------------ attribution
    @staticmethod
    def _attribute(job: Job, measured_hours: float, launch_overhead_h: float) -> list[Attribution]:
        out: list[Attribution] = []
        total = max(measured_hours, 1e-9)
        if launch_overhead_h > 0:
            out.append(Attribution(cause="launch_overhead", fraction=round(min(1.0, launch_overhead_h / total), 3),
                                   detail=f"{launch_overhead_h * 60:.1f} min from provision to first launch"))
        p = job.telemetry.profiler
        if p and p.windows:
            if p.frac_data_loading > 0.02:
                nw = f"; {job.telemetry.num_workers} DataLoader workers" if job.telemetry.num_workers is not None else ""
                out.append(Attribution(cause="data_loading", fraction=round(p.frac_data_loading, 3),
                                       detail=f"{p.frac_data_loading:.0%} of step time in DataLoader __next__ (CPU-bound{nw})"))
            if p.frac_memory_bound > 0.05:
                top = ", ".join(o["op"] for o in p.top_ops[:3]) if p.top_ops else "elementwise/norm ops"
                out.append(Attribution(cause="memory_bound_ops", fraction=round(p.frac_memory_bound, 3),
                                       detail=f"{p.frac_memory_bound:.0%} in memory-bound kernels ({top})"))
            if p.frac_comm > 0.02:
                out.append(Attribution(cause="comm", fraction=round(p.frac_comm, 3),
                                       detail=f"{p.frac_comm:.0%} in collective communication"))
        if job.preemptions:
            out.append(Attribution(cause="preemption", fraction=0.0, detail=f"{job.preemptions} preemption(s)"))
        if job.telemetry.gaps:
            out.append(Attribution(cause="telemetry_gap", fraction=0.0,
                                   detail=f"{len(job.telemetry.gaps)} telemetry gap(s); affected windows unattributed"))
        return out

    @staticmethod
    def _recommend(prereg, job, mfu, attribution, spec) -> list[str]:
        recs: list[str] = []
        tel = job.telemetry
        causes = {a.cause: a for a in attribution}
        if "data_loading" in causes and causes["data_loading"].fraction > 0.10:
            nw = tel.num_workers
            target = max(8, (nw or 4) * 3)
            recs.append(f"DataLoader: num_workers={target} (currently {nw if nw is not None else 'unknown'}), pin_memory=True, prefetch_factor=4")
        if "memory_bound_ops" in causes and causes["memory_bound_ops"].fraction > 0.20:
            recs.append("torch.compile(model) to fuse LayerNorm/residual/activation chains; use fused optimizer (fused=True)")
        if "comm" in causes and causes["comm"].fraction > 0.15:
            recs.append("increase gradient-accumulation steps or DDP bucket_cap_mb to overlap communication")
        pg = prereg.estimate.per_gpu.get(job.offer.gpu_type)
        below_band = mfu is not None and pg is not None and mfu < pg.mfu_band.lo
        if below_band and prereg.job.batch_tokens < 2048:
            recs.append(f"batch_tokens={prereg.job.batch_tokens} is small for {spec.gpu_class if spec else 'this GPU'}; raise batch or sequence length")
        if tel.peak_mem_gb and spec and tel.peak_mem_gb < 0.4 * spec.vram_gb:
            recs.append(f"peak memory {tel.peak_mem_gb:.1f} GB uses <40% of {spec.vram_gb:.0f} GB; a larger batch would raise MFU")
        if "launch_overhead" in causes and causes["launch_overhead"].fraction > 0.15:
            recs.append("launch overhead dominates: batch several short jobs per instance or prefer providers with 'now' availability")
        if job.status.value == "breached":
            recs.append("the envelope was breached: re-preregister with the measured hours before rerunning; do not widen blindly")
        return recs

    @staticmethod
    def _narrative(r: Reconciliation, job: Job) -> str:
        parts = [f"Predicted {r.predicted_hours.lo:.2f}-{r.predicted_hours.hi:.2f} h at "
                 f"{r.predicted_mfu.lo:.0%}-{r.predicted_mfu.hi:.0%} MFU (${r.predicted_usd.lo:.2f}-${r.predicted_usd.hi:.2f} on "
                 f"{r.provider}/{r.gpu}); measured {r.measured_hours:.2f} h"]
        if r.measured_mfu is not None:
            parts[-1] += f" at {r.measured_mfu:.0%} MFU"
        parts[-1] += f", {r.steps_completed}/{r.steps_planned} steps, job {job.status.value}."
        if r.measured_usd is not None:
            parts.append(f"Billed ${r.measured_usd.usd:.2f} ({r.measured_usd.source}).")
        else:
            acc = f" (accrued at quoted price: ${r.accrued_usd.usd:.2f})" if r.accrued_usd else ""
            parts.append(f"Billed: unavailable{acc}.")
        if r.attribution:
            parts.append("Attribution: " + "; ".join(a.detail for a in r.attribution) + ".")
        if r.recommendations:
            parts.append("Recommendation: " + " | ".join(r.recommendations))
        if r.calibration_written:
            parts.append("Calibration updated.")
        return " ".join(parts)
