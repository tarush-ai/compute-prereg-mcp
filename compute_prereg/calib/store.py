"""Calibration store: achieved-MFU samples keyed by
(provider, gpu_type, model_class, framework, dtype, batch_tokens bucket).

Bootstrapped from published priors; measurements replace priors as runs accumulate.
Lookup falls back along a hierarchy (exact -> any provider -> any framework -> any bucket)
and blends few-sample keys with the prior so one lucky run cannot collapse the band.
"""

from __future__ import annotations

import sqlite3
import statistics
from datetime import UTC, datetime
from pathlib import Path

from ..est.gpus import GPUS
from ..est.priors import batch_tokens_bucket, prior_band
from ..models import Band, CalibrationEntry, CalibrationKey

MIN_SAMPLES_FOR_MEASURED = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mfu_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    gpu_type TEXT NOT NULL,
    model_class TEXT NOT NULL,
    framework TEXT NOT NULL,
    dtype TEXT NOT NULL,
    batch_bucket TEXT NOT NULL,
    mfu REAL NOT NULL,
    job_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_mfu_key ON mfu_samples(provider, gpu_type, model_class, framework, dtype, batch_bucket);
"""


class CalibrationStore:
    def __init__(self, path: str | Path | None):
        self.path = ":memory:" if path is None else str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------ writes
    def record(self, key: CalibrationKey, mfu: float, job_id: str | None = None) -> None:
        if not (0.0 < mfu <= 1.0):
            raise ValueError(f"MFU must be in (0, 1], got {mfu}")
        self._conn.execute(
            "INSERT INTO mfu_samples(provider,gpu_type,model_class,framework,dtype,batch_bucket,mfu,job_id,created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (key.provider, key.gpu_type, key.model_class, key.framework, key.dtype,
             key.batch_tokens_bucket, float(mfu), job_id, datetime.now(UTC).isoformat()),
        )
        self._conn.commit()

    # ------------------------------------------------------------------ reads
    def make_key(self, *, provider: str, gpu_type: str, model_class: str, framework: str, dtype: str,
                 batch_tokens: int) -> CalibrationKey:
        return CalibrationKey(provider=provider, gpu_type=gpu_type, model_class=model_class,
                              framework=_framework_major(framework), dtype=dtype,
                              batch_tokens_bucket=batch_tokens_bucket(batch_tokens))

    def samples(self, key: CalibrationKey, *, wildcard_provider=False, wildcard_framework=False,
                wildcard_bucket=False) -> list[float]:
        q = "SELECT mfu FROM mfu_samples WHERE gpu_type=? AND model_class=? AND dtype=?"
        args: list = [key.gpu_type, key.model_class, key.dtype]
        if not wildcard_provider:
            q += " AND provider=?"
            args.append(key.provider)
        if not wildcard_framework:
            q += " AND framework=?"
            args.append(key.framework)
        if not wildcard_bucket:
            q += " AND batch_bucket=?"
            args.append(key.batch_tokens_bucket)
        return [r[0] for r in self._conn.execute(q, args).fetchall()]

    def lookup(self, *, provider: str, gpu_type: str, model_class: str, framework: str, dtype: str,
               batch_tokens: int) -> CalibrationEntry | None:
        if gpu_type not in GPUS:
            return None
        key = self.make_key(provider=provider, gpu_type=gpu_type, model_class=model_class,
                            framework=framework, dtype=dtype, batch_tokens=batch_tokens)
        prior = prior_band(GPUS[gpu_type].gpu_class, model_class, batch_tokens)
        for wp, wf, wb in [(provider == "*", False, False), (True, False, False), (True, True, False), (True, True, True)]:
            xs = self.samples(key, wildcard_provider=wp, wildcard_framework=wf, wildcard_bucket=wb)
            if xs:
                return self._entry(key, xs, prior)
        return CalibrationEntry(key=key, n=0, mean=prior.mid, p10=prior.lo, p90=prior.hi, source="prior")

    def entries(self, *, provider: str | None = None, gpu_type: str | None = None,
                model_class: str | None = None) -> list[CalibrationEntry]:
        q = ("SELECT provider,gpu_type,model_class,framework,dtype,batch_bucket FROM mfu_samples WHERE 1=1")
        args: list = []
        for col, val in (("provider", provider), ("gpu_type", gpu_type), ("model_class", model_class)):
            if val:
                q += f" AND {col}=?"
                args.append(val)
        keys = {tuple(r) for r in self._conn.execute(q, args).fetchall()}
        out = []
        for k in sorted(keys):
            key = CalibrationKey(provider=k[0], gpu_type=k[1], model_class=k[2], framework=k[3], dtype=k[4],
                                 batch_tokens_bucket=k[5])
            prior = prior_band(GPUS[key.gpu_type].gpu_class if key.gpu_type in GPUS else "*", key.model_class)
            out.append(self._entry(key, self.samples(key), prior))
        return out

    def priors(self, gpu_type: str, model_class: str, batch_tokens: int = 0) -> CalibrationEntry:
        cls = GPUS[gpu_type].gpu_class if gpu_type in GPUS else "*"
        b = prior_band(cls, model_class, batch_tokens or None)
        key = CalibrationKey(provider="*", gpu_type=gpu_type, model_class=model_class, framework="*", dtype="*",
                             batch_tokens_bucket=batch_tokens_bucket(batch_tokens))
        return CalibrationEntry(key=key, n=0, mean=b.mid, p10=b.lo, p90=b.hi, source="prior")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _entry(key: CalibrationKey, xs: list[float], prior: Band) -> CalibrationEntry:
        n = len(xs)
        mean = statistics.fmean(xs)
        if n >= MIN_SAMPLES_FOR_MEASURED:
            p10, p90 = _quantile(xs, 0.10), _quantile(xs, 0.90)
            # Never report a zero-width band: keep at least +-5% around the mean.
            p10, p90 = min(p10, mean * 0.95), max(p90, mean * 1.05)
            return CalibrationEntry(key=key, n=n, mean=mean, p10=p10, p90=p90, source="measured")
        # Blend: weight measured mean by n/(n+2) against the prior midpoint; keep prior width shrunk.
        w = n / (n + 2)
        blended_mean = w * mean + (1 - w) * prior.mid
        half = (prior.hi - prior.lo) / 2 * (1 - 0.5 * w)
        return CalibrationEntry(key=key, n=n, mean=blended_mean, p10=max(0.01, blended_mean - half),
                                p90=min(1.0, blended_mean + half), source="blended")


def _quantile(xs: list[float], q: float) -> float:
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    pos = (len(ys) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


def _framework_major(framework: str) -> str:
    # "torch-2.8.0" -> "torch-2.8"; keeps calibration keys stable across patch releases.
    if "-" in framework:
        name, _, ver = framework.partition("-")
        parts = ver.split(".")
        return f"{name}-{'.'.join(parts[:2])}"
    return framework
