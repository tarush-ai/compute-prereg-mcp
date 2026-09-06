import pytest

from compute_prereg.calib import CalibrationStore
from compute_prereg.errors import EstimationError
from compute_prereg.est import Estimator, gpu_matches_class, normalize_gpu_name
from compute_prereg.est.analytic import params_from_hf_config, transformer_flops
from compute_prereg.models import Confidence


def test_trace_tiny_transformer(example):
    est = Estimator(CalibrationStore(None)).estimate(entrypoint=str(example), gpu_candidates=["H100", "A100-80GB"],
                                                    total_steps=1000)
    assert est.method == "faketensor+flopcounter"
    assert est.confidence == Confidence.prior
    assert est.job.model_class == "decoder-transformer"
    assert est.job.params == 470_528 and est.job.batch_tokens == 512
    assert est.flops_backward == pytest.approx(2 * est.flops_forward)
    # Analytic 2*N*T (+attention) overcounts embedding lookups; must be within 30%.
    fwd_analytic, _ = transformer_flops(est.job.params, est.job.batch_tokens, 64, 128, 2)
    assert abs(est.flops_forward - fwd_analytic) / fwd_analytic < 0.3
    assert est.flops_total == pytest.approx(est.flops_per_step * 1000)
    assert est.job.code_hash and est.job.code_hash.startswith("sha256:")
    assert set(est.peak_memory.breakdown_gb) >= {"params", "grads", "optimizer", "activations", "reserve"}
    assert est.peak_memory.method.startswith("faketensor+memtracker")
    assert est.peak_memory.required_gb > est.peak_memory.point_gb  # margin applied
    for g in ("H100-SXM-80GB", "A100-SXM-80GB"):
        p = est.per_gpu[g]
        assert p.fits_memory and p.hours.lo <= p.hours.hi and p.usd_at_ref_price.lo <= p.usd_at_ref_price.hi
        assert p.mfu_source == "prior"
    assert est.recommended_gpu in est.per_gpu
    assert "TinyDecoder.blocks.0.attn" in est.per_module_flops


def test_spec_path_hf_config():
    cfg = {"hidden_size": 1024, "num_hidden_layers": 24, "vocab_size": 50257, "num_attention_heads": 16,
           "intermediate_size": 4096, "hidden_act": "gelu"}
    n = params_from_hf_config(cfg)
    assert 300e6 < n < 400e6  # GPT-2 medium class
    est = Estimator(None).estimate(model_spec={"hf_config": cfg, "seq_len": 1024, "batch_size": 1, "total_steps": 20000},
                                   gpu_candidates=["H100"])
    assert est.method == "analytic" and est.confidence == Confidence.prior
    assert est.job.params == n and est.job.batch_tokens == 1024
    assert est.flops_forward > 2 * n * 1024  # attention term adds on top
    assert est.flops_total == pytest.approx(3 * est.flops_forward * 20000)
    p = est.per_gpu["H100-SXM-80GB"]
    # 4.3e16 FLOP at 35-55% of 989 TFLOPS is ~2-3.5 min of compute plus the 5 min launch overhead.
    assert 0.10 < p.hours.lo < p.hours.hi < 0.20 and p.bound == "compute"
    assert est.peak_memory.required_gb > 16 * n / 1024**3  # weights+grads+adam+master >= 16 B/param


def test_spec_requires_params():
    with pytest.raises(EstimationError):
        Estimator(None).estimate(model_spec={"batch_tokens": 10})
    with pytest.raises(EstimationError):
        Estimator(None).estimate()


def test_tracing_failure_falls_back_low(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("import torch\n"
                   "def prereg_build(config):\n"
                   "    raise RuntimeError('Unsupported operator: aten::nonzero data-dependent')\n")
    est = Estimator(None).estimate(entrypoint=str(bad), model_spec={"params": 1_000_000, "batch_tokens": 512,
                                                                     "model_class": "mlp"}, gpu_candidates=["H100"])
    assert est.confidence == Confidence.low
    assert est.method == "analytic-fallback"
    assert any("tracing failed" in w for w in est.warnings)
    with pytest.raises(EstimationError):
        Estimator(None).estimate(entrypoint=str(bad))


def test_calibration_changes_confidence(example):
    calib = CalibrationStore(None)
    e1 = Estimator(calib).estimate(entrypoint=str(example), gpu_candidates=["H100"], total_steps=10)
    key = calib.make_key(provider="lambda", gpu_type="H100-SXM-80GB", model_class="decoder-transformer",
                         framework=e1.job.framework, dtype=e1.job.dtype, batch_tokens=e1.job.batch_tokens)
    for m in (0.48, 0.50, 0.52):
        calib.record(key, m, "job")
    e2 = Estimator(calib).estimate(entrypoint=str(example), gpu_candidates=["H100"], total_steps=10)
    p = e2.per_gpu["H100-SXM-80GB"]
    assert p.mfu_source == "measured(n=3)"
    assert e2.confidence == Confidence.calibrated
    assert p.mfu_band.lo < 0.50 < p.mfu_band.hi
    assert p.hours.hi < e1.per_gpu["H100-SXM-80GB"].hours.hi  # tighter band


@pytest.mark.parametrize("raw,expected", [
    ("NVIDIA H100 80GB HBM3", "H100-SXM-80GB"), ("H100 PCIe", "H100-PCIe-80GB"), ("1x A100 SXM4 80GB", "A100-SXM-80GB"),
    ("A100-40GB", "A100-PCIe-40GB"), ("RTX 4090", "RTX-4090-24GB"), ("L4", "L4-24GB"), ("L40S", "L40S-48GB"),
    ("Tesla T4", "T4-16GB"), ("nonsense-9000", None),
])
def test_normalize(raw, expected):
    assert normalize_gpu_name(raw) == expected


def test_gpu_matches_class():
    assert gpu_matches_class("H100-SXM-80GB", ["H100"])
    assert gpu_matches_class("H100-PCIe-80GB", ["H100-class"])
    assert gpu_matches_class("H200-141GB", ["H100", "H200"])
    assert gpu_matches_class("H100-SXM-80GB", ["H100*"])
    assert not gpu_matches_class("A100-SXM-80GB", ["H100", "H200"])
