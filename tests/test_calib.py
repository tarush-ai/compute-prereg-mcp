from compute_prereg.calib import CalibrationStore
from compute_prereg.est.priors import batch_tokens_bucket, prior_band


def test_prior_then_blend_then_measured():
    c = CalibrationStore(None)
    kw = dict(provider="lambda", gpu_type="H100-SXM-80GB", model_class="decoder-transformer", framework="torch-2.8.1",
              dtype="bf16", batch_tokens=1024)
    e = c.lookup(**kw)
    assert e.source == "prior" and e.n == 0 and (e.p10, e.p90) == (0.35, 0.55)
    key = c.make_key(**kw)
    assert key.framework == "torch-2.8" and key.batch_tokens_bucket == "1024-2047"
    c.record(key, 0.30)
    e = c.lookup(**kw)
    assert e.source == "blended" and e.n == 1 and e.p10 < e.mean < e.p90 and e.mean < 0.45
    c.record(key, 0.32)
    c.record(key, 0.28)
    e = c.lookup(**kw)
    assert e.source == "measured" and e.n == 3 and 0.27 <= e.p10 <= 0.29 and 0.31 <= e.p90 <= 0.33
    # Fallback hierarchy: another provider sees the measurements via the wildcard.
    other = c.lookup(**{**kw, "provider": "runpod"})
    assert other.n == 3
    assert c.entries(gpu_type="H100-SXM-80GB")[0].n == 3


def test_priors_small_batch_and_buckets():
    assert prior_band("H100", "decoder-transformer", 128).hi < prior_band("H100", "decoder-transformer", 4096).hi
    assert prior_band("ZZZ", "cnn").lo == 0.20
    assert batch_tokens_bucket(0) == "unknown" and batch_tokens_bucket(600) == "512-1023"
