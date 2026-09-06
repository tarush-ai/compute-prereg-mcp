"""Estimator: code or spec -> Estimate record with per-GPU time/$ bands and a confidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import ids
from ..errors import EstimationError, UnsupportedTraceError
from ..models import Band, Confidence, Estimate, JobSpec
from . import analytic
from .gpus import GPUS, resolve_candidates
from .memory import DEFAULT_RESERVE_GB, estimate_memory
from .priors import prior_band
from .roofline import per_gpu_estimate, training_bytes_per_step
from .tracer import sha256_file, sha256_json, trace_entrypoint


class Estimator:
    def __init__(self, calibration=None, *, reserve_gb: float = DEFAULT_RESERVE_GB):
        self.calibration = calibration  # calib.store.CalibrationStore or None
        self.reserve_gb = reserve_gb

    # ------------------------------------------------------------------ public
    def estimate(
        self,
        *,
        entrypoint: str | None = None,
        config: dict[str, Any] | None = None,
        model_spec: dict[str, Any] | None = None,
        gpu_candidates: list[str] | None = None,
        total_steps: int | None = None,
        provider_hint: str | None = None,
    ) -> Estimate:
        if not entrypoint and not model_spec:
            raise EstimationError("pass an entrypoint (Python file with prereg_build) or a model_spec")
        candidates = resolve_candidates(gpu_candidates)
        config = config or {}
        warnings: list[str] = []

        if entrypoint:
            return self._from_entrypoint(entrypoint, config, model_spec, candidates, total_steps, provider_hint, warnings)
        return self._from_spec(model_spec or {}, candidates, total_steps, provider_hint, warnings, method="analytic",
                               confidence_cap=Confidence.prior)

    # ------------------------------------------------------------------ paths
    def _from_entrypoint(self, entrypoint, config, model_spec, candidates, total_steps, provider_hint, warnings):
        path = Path(entrypoint)
        try:
            tr = trace_entrypoint(path, config)
        except UnsupportedTraceError as e:
            warnings.append(f"FakeTensor tracing failed: {e}")
            if model_spec:
                warnings.append("fell back to analytic estimate from model_spec; confidence=low. "
                                "A 60 s profiling run on the cheapest conforming GPU can replace this "
                                "(billed to the user as its own line item).")
                est = self._from_spec(model_spec, candidates, total_steps, provider_hint, warnings,
                                      method="analytic-fallback", confidence_cap=Confidence.low)
                est.job.entrypoint = str(path)
                est.job.code_hash = sha256_file(path)
                est.job.config_hash = sha256_json(config)
                return est
            raise EstimationError(
                f"tracing failed and no model_spec to fall back on: {e}. Provide model_spec "
                "(params, model_class, batch_tokens, seq_len...) or fix prereg_build."
            ) from e

        warnings.extend(tr.warnings)
        meta = tr.metadata
        steps = int(total_steps or meta.get("total_steps") or config.get("total_steps") or 0)
        if steps <= 0:
            warnings.append("total_steps unknown; defaulted to 1000 — pass total_steps")
            steps = 1000
        optimizer = str(meta.get("optimizer") or config.get("optimizer") or "adamw")
        mixed = bool(meta.get("mixed_precision", tr.dtype != "fp32"))
        job = JobSpec(
            entrypoint=str(path),
            code_hash=sha256_file(path),
            config_hash=sha256_json(config),
            framework=_torch_version(),
            model_class=str(meta.get("model_class") or tr.model_class),
            params=tr.params,
            dtype=tr.dtype,
            batch_tokens=tr.batch_tokens,
            total_steps=steps,
            optimizer=optimizer,
            mixed_precision=mixed,
            extra={k: v for k, v in meta.items() if isinstance(v, (int, float, str, bool))},
        )
        act_bytes = tr.activation_bytes
        if act_bytes <= 0 and meta.get("seq_len") and meta.get("hidden_size") and meta.get("num_layers"):
            act_bytes = analytic.transformer_activation_bytes(
                tr.batch_tokens, int(meta["seq_len"]), int(meta["hidden_size"]), int(meta["num_layers"]),
                int(meta.get("heads", 1)), analytic.dtype_bytes(tr.dtype))
        mem = estimate_memory(tr.params, tr.dtype, optimizer, mixed, act_bytes,
                              method="faketensor+memtracker" if tr.tracked_memory else "faketensor+analytic",
                              reserve_gb=self.reserve_gb, tracked=tr.tracked_memory or None)
        return self._finish(job, tr.flops_forward, tr.flops_backward, tr.param_bytes, mem, tr.per_module,
                            candidates, provider_hint, warnings, method="faketensor+flopcounter",
                            confidence_cap=Confidence.exact)

    def _from_spec(self, spec, candidates, total_steps, provider_hint, warnings, *, method, confidence_cap):
        hf = spec.get("hf_config")
        params = int(spec.get("params") or 0)
        hidden = spec.get("hidden_size")
        layers = spec.get("num_layers")
        heads = spec.get("heads") or spec.get("num_attention_heads")
        if hf:
            params = params or analytic.params_from_hf_config(hf)
            hidden = hidden or hf.get("hidden_size") or hf.get("n_embd")
            layers = layers or hf.get("num_hidden_layers") or hf.get("n_layer")
            heads = heads or hf.get("num_attention_heads") or hf.get("n_head")
        if params <= 0:
            raise EstimationError("model_spec needs params (or an hf_config to derive it)")
        seq_len = int(spec.get("seq_len") or 0)
        batch_size = int(spec.get("batch_size") or 0)
        batch_tokens = int(spec.get("batch_tokens") or (seq_len * batch_size) or 0)
        if batch_tokens <= 0:
            raise EstimationError("model_spec needs batch_tokens or seq_len x batch_size")
        steps = int(total_steps or spec.get("total_steps") or 0)
        if steps <= 0:
            warnings.append("total_steps unknown; defaulted to 1000 — pass total_steps")
            steps = 1000
        model_class = str(spec.get("model_class") or ("decoder-transformer" if hf else "unknown"))
        dtype = str(spec.get("dtype") or "bf16")
        optimizer = str(spec.get("optimizer") or "adamw")
        mixed = bool(spec.get("mixed_precision", dtype != "fp32"))
        if "transformer" in model_class:
            fwd, bwd = analytic.transformer_flops(params, batch_tokens, seq_len or None,
                                                  int(hidden) if hidden else None, int(layers) if layers else None)
        else:
            fwd, bwd = analytic.generic_flops(params, batch_tokens)
        act_bytes = analytic.transformer_activation_bytes(
            batch_tokens, seq_len, int(hidden or 0), int(layers or 0), int(heads or 1),
            analytic.dtype_bytes(dtype), flash_attention=bool(spec.get("flash_attention", True)))
        if act_bytes <= 0:
            # Generic rule of thumb: ~ 16 bytes per parameter-token / 1e3 ... fall back to 20% of weights.
            act_bytes = params * analytic.dtype_bytes(dtype) * 0.2 * max(1.0, batch_tokens / 1024)
            warnings.append("activation memory is a rule-of-thumb (no seq_len/hidden/layers given)")
        param_bytes = params * analytic.dtype_bytes(dtype)
        mem = estimate_memory(params, dtype, optimizer, mixed, act_bytes, method="analytic", reserve_gb=self.reserve_gb)
        job = JobSpec(
            entrypoint=spec.get("entrypoint"), framework=str(spec.get("framework") or "torch"),
            model_class=model_class, params=params, dtype=dtype, batch_tokens=batch_tokens,
            total_steps=steps, optimizer=optimizer, mixed_precision=mixed,
            config_hash=sha256_json(spec),
            extra={k: v for k, v in spec.items() if isinstance(v, (int, float, str, bool))},
        )
        return self._finish(job, fwd, bwd, param_bytes, mem, {}, candidates, provider_hint, warnings,
                            method=method, confidence_cap=confidence_cap)

    # ------------------------------------------------------------------ shared tail
    def _finish(self, job, fwd, bwd, param_bytes, mem, per_module, candidates, provider_hint, warnings,
                *, method, confidence_cap):
        flops_step = fwd + bwd
        opt_bytes = job.params * analytic.optimizer_bytes_per_param(job.optimizer, job.mixed_precision, job.dtype)
        act_bytes = mem.breakdown_gb.get("activations", 0.0) * (1024**3)
        bytes_step = training_bytes_per_step(param_bytes, opt_bytes, act_bytes)
        per_gpu = {}
        best_conf = Confidence.prior
        for g in candidates:
            band, source, conf = self._mfu(g, job, provider_hint)
            if _conf_rank(conf) > _conf_rank(best_conf):
                best_conf = conf
            per_gpu[g] = per_gpu_estimate(
                g, flops_per_step=flops_step, bytes_per_step=bytes_step, total_steps=job.total_steps,
                dtype=job.dtype, required_mem_gb=mem.required_gb, mfu_band=band, mfu_source=source)
        confidence = min(best_conf, confidence_cap, key=_conf_rank)
        fitting = [p for p in per_gpu.values() if p.fits_memory]
        recommended = min(fitting, key=lambda p: p.usd_at_ref_price.mid).gpu if fitting else None
        if not fitting:
            warnings.append("no candidate GPU fits the estimated peak memory on a single device")
        return Estimate(
            estimate_id=ids.estimate_id(), job=job, flops_forward=fwd, flops_backward=bwd,
            flops_total=flops_step * job.total_steps, param_bytes=param_bytes, peak_memory=mem,
            confidence=confidence, method=method, per_module_flops=per_module, per_gpu=per_gpu,
            warnings=warnings, recommended_gpu=recommended,
        )

    def _mfu(self, gpu: str, job: JobSpec, provider_hint: str | None) -> tuple[Band, str, Confidence]:
        spec = GPUS[gpu]
        if self.calibration is not None:
            entry = self.calibration.lookup(
                provider=provider_hint or "*", gpu_type=gpu, model_class=job.model_class,
                framework=job.framework, dtype=job.dtype, batch_tokens=job.batch_tokens)
            if entry is not None and entry.source != "prior":
                conf = Confidence.exact if (entry.source == "measured" and entry.n >= 10) else Confidence.calibrated
                return entry.band, f"{entry.source}(n={entry.n})", conf
        return prior_band(spec.gpu_class, job.model_class, job.batch_tokens), "prior", Confidence.prior


_CONF_RANK = {Confidence.low: 0, Confidence.prior: 1, Confidence.calibrated: 2, Confidence.exact: 3}


def _conf_rank(c: Confidence) -> int:
    return _CONF_RANK[c]


def _torch_version() -> str:
    try:
        import torch

        return "torch-" + torch.__version__.split("+")[0]
    except Exception:  # pragma: no cover
        return "torch"
