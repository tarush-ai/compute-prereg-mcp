"""Append-only ledger (SQLite). Preregistrations are immutable, hashed and signed; edits
create a new record that references the old one. Status changes are append-only events.
Jobs, offers, reconciliations and confirmation tokens live alongside."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .. import ids
from ..errors import LedgerIntegrityError, NotFound, PolicyError
from ..models import (
    Confirmation,
    Envelope,
    Estimate,
    Job,
    Mode,
    Offer,
    Preregistration,
    PreregStatus,
    Reconciliation,
    utcnow,
)
from .signing import Signer

_SCHEMA = """
CREATE TABLE IF NOT EXISTS estimates (
    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS preregistrations (
    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, supersedes TEXT,
    record_hash TEXT NOT NULL, signature TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS prereg_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, prereg_id TEXT NOT NULL, status TEXT NOT NULL,
    reason TEXT, at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS ix_events ON prereg_events(prereg_id, id);
CREATE TABLE IF NOT EXISTS offers (
    id TEXT PRIMARY KEY, prereg_id TEXT NOT NULL, fetched_at TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, prereg_id TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    status TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reconciliations (
    id TEXT PRIMARY KEY, job_id TEXT NOT NULL, prereg_id TEXT NOT NULL, created_at TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS confirmations (
    token TEXT PRIMARY KEY, tool TEXT NOT NULL, tier TEXT NOT NULL, binding_hash TEXT NOT NULL,
    issued_at TEXT NOT NULL, expires_at TEXT NOT NULL, summary TEXT NOT NULL, consumed_at TEXT);
"""

# Envelope fields an amendment may touch. dollar_cap raises are human-only (see amend()).
AMENDABLE = {"gpu_classes", "min_gpu_count", "max_gpu_count", "min_vram_gb", "hours_point", "hours_tolerance",
             "dollar_cap", "spot_ok", "max_preemptions", "region_allow", "provider_allow", "provider_deny",
             "checkpoint_interval_steps", "interconnect", "mode"}
AMENDABLE_STATUSES = {PreregStatus.open, PreregStatus.breached, PreregStatus.failed, PreregStatus.terminated,
                      PreregStatus.completed}


class Ledger:
    def __init__(self, path: str | Path | None, signer: Signer | None = None):
        self.path = ":memory:" if path is None else str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL") if self.path != ":memory:" else None
        self._conn.executescript(_SCHEMA)
        self.signer = signer or Signer(None)

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------ estimates
    def put_estimate(self, est: Estimate) -> None:
        self._conn.execute("INSERT OR REPLACE INTO estimates VALUES (?,?,?)",
                           (est.estimate_id, est.created_at.isoformat(), est.model_dump_json()))

    def get_estimate(self, estimate_id: str) -> Estimate:
        row = self._conn.execute("SELECT body FROM estimates WHERE id=?", (estimate_id,)).fetchone()
        if not row:
            raise NotFound(f"estimate {estimate_id} not found")
        return Estimate.model_validate_json(row[0])

    # ------------------------------------------------------------------ preregistrations
    def create_preregistration(self, estimate: Estimate, envelope: Envelope, mode: Mode,
                               *, supersedes: str | None = None, amended_by: str | None = None,
                               note: str | None = None) -> Preregistration:
        rec = Preregistration(preregistration_id=ids.preregistration_id(), supersedes=supersedes,
                              job=estimate.job, estimate=estimate, envelope=envelope, mode=mode,
                              amended_by=amended_by, amendment_note=note)
        h, sig = self.signer.sign(rec.signable_body())
        rec.record_hash, rec.signature = h, sig
        self._conn.execute("INSERT INTO preregistrations VALUES (?,?,?,?,?,?)",
                           (rec.preregistration_id, rec.created_at.isoformat(), supersedes, h, sig,
                            rec.model_dump_json()))
        self._event(rec.preregistration_id, PreregStatus.open, "created")
        if supersedes:
            self._event(supersedes, PreregStatus.superseded, f"superseded by {rec.preregistration_id}")
        return rec

    def get_preregistration(self, prereg_id: str) -> Preregistration:
        row = self._conn.execute("SELECT body, record_hash, signature FROM preregistrations WHERE id=?",
                                 (prereg_id,)).fetchone()
        if not row:
            raise NotFound(f"preregistration {prereg_id} not found")
        rec = Preregistration.model_validate_json(row[0])
        if not self.signer.verify(rec.signable_body(), row[1], row[2]):
            raise LedgerIntegrityError(f"{prereg_id}: stored record does not verify against its signature")
        rec.status = self.status(prereg_id)
        return rec

    def list_preregistrations(self, status: PreregStatus | None = None) -> list[Preregistration]:
        rows = self._conn.execute("SELECT id FROM preregistrations ORDER BY created_at DESC").fetchall()
        out = [self.get_preregistration(r[0]) for r in rows]
        return [p for p in out if status is None or p.status == status]

    def status(self, prereg_id: str) -> PreregStatus:
        row = self._conn.execute("SELECT status FROM prereg_events WHERE prereg_id=? ORDER BY id DESC LIMIT 1",
                                 (prereg_id,)).fetchone()
        return PreregStatus(row[0]) if row else PreregStatus.open

    def set_status(self, prereg_id: str, status: PreregStatus, reason: str | None = None) -> None:
        current = self.status(prereg_id)
        if current == PreregStatus.superseded:
            raise PolicyError(f"{prereg_id} is superseded; act on its successor")
        self._event(prereg_id, status, reason)

    def events(self, prereg_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT status, reason, at FROM prereg_events WHERE prereg_id=? ORDER BY id",
                                  (prereg_id,)).fetchall()
        return [{"status": r[0], "reason": r[1], "at": r[2]} for r in rows]

    def _event(self, prereg_id: str, status: PreregStatus, reason: str | None) -> None:
        self._conn.execute("INSERT INTO prereg_events(prereg_id,status,reason,at) VALUES (?,?,?,?)",
                           (prereg_id, status.value, reason, utcnow().isoformat()))

    def amend(self, prereg_id: str, changes: dict[str, Any], *, actor: str, note: str | None = None
              ) -> Preregistration:
        """Create a new record superseding ``prereg_id``. ``actor`` is "human" or "agent".
        No agent may raise dollar_cap (or widen hours/gpu count); only decreases are agent-amendable."""
        old = self.get_preregistration(prereg_id)
        if old.status not in AMENDABLE_STATUSES:
            raise PolicyError(f"{prereg_id} is {old.status}; terminate the job before amending")
        unknown = set(changes) - AMENDABLE
        if unknown:
            raise PolicyError(f"not amendable: {sorted(unknown)}")
        env = old.envelope.model_dump()
        mode = old.mode
        for k, v in changes.items():
            if k == "mode":
                mode = Mode(v)
            else:
                env[k] = v
        new_env = Envelope.model_validate(env)
        if actor != "human":
            widened = []
            if new_env.dollar_cap > old.envelope.dollar_cap:
                widened.append("dollar_cap")
            if new_env.hours_max > old.envelope.hours_max:
                widened.append("hours")
            if new_env.max_gpu_count > old.envelope.max_gpu_count:
                widened.append("max_gpu_count")
            if widened:
                raise PolicyError(
                    f"agent may not raise {', '.join(widened)}; a human must do this via the CLI "
                    f"(compute-prereg amend {prereg_id} ...)"
                )
        return self.create_preregistration(old.estimate, new_env, mode, supersedes=prereg_id,
                                           amended_by=actor, note=note)

    def verify_all(self) -> list[tuple[str, bool]]:
        rows = self._conn.execute("SELECT id, body, record_hash, signature FROM preregistrations").fetchall()
        out = []
        for pid, body, h, sig in rows:
            rec = Preregistration.model_validate_json(body)
            out.append((pid, self.signer.verify(rec.signable_body(), h, sig)))
        return out

    # ------------------------------------------------------------------ offers
    def put_offers(self, prereg_id: str, offers: list[Offer]) -> None:
        for o in offers:
            self._conn.execute("INSERT OR REPLACE INTO offers VALUES (?,?,?,?)",
                               (o.offer_id, prereg_id, o.fetched_at.isoformat(), o.model_dump_json()))

    def get_offer(self, offer_id: str) -> tuple[Offer, str]:
        row = self._conn.execute("SELECT body, prereg_id FROM offers WHERE id=?", (offer_id,)).fetchone()
        if not row:
            raise NotFound(f"offer {offer_id} not found; call list_offers first")
        return Offer.model_validate_json(row[0]), row[1]

    def offers_for(self, prereg_id: str) -> list[Offer]:
        rows = self._conn.execute("SELECT body FROM offers WHERE prereg_id=? ORDER BY fetched_at DESC",
                                  (prereg_id,)).fetchall()
        return [Offer.model_validate_json(r[0]) for r in rows]

    # ------------------------------------------------------------------ jobs
    def put_job(self, job: Job) -> None:
        now = utcnow().isoformat()
        self._conn.execute(
            "INSERT INTO jobs VALUES (?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at,"
            " status=excluded.status, body=excluded.body",
            (job.job_id, job.preregistration_id, job.created_at.isoformat(), now, job.status.value,
             job.model_dump_json()))

    def get_job(self, job_id: str) -> Job:
        row = self._conn.execute("SELECT body FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise NotFound(f"job {job_id} not found")
        return Job.model_validate_json(row[0])

    def jobs_for(self, prereg_id: str) -> list[Job]:
        rows = self._conn.execute("SELECT body FROM jobs WHERE prereg_id=? ORDER BY created_at", (prereg_id,)).fetchall()
        return [Job.model_validate_json(r[0]) for r in rows]

    def list_jobs(self, active_only: bool = False) -> list[Job]:
        rows = self._conn.execute("SELECT body FROM jobs ORDER BY created_at DESC").fetchall()
        jobs = [Job.model_validate_json(r[0]) for r in rows]
        if active_only:
            jobs = [j for j in jobs if j.status.value in {"pending", "provisioning", "syncing", "running", "checkpointing"}]
        return jobs

    # ------------------------------------------------------------------ reconciliations
    def put_reconciliation(self, rec: Reconciliation) -> None:
        self._conn.execute("INSERT OR REPLACE INTO reconciliations VALUES (?,?,?,?,?)",
                           (rec.reconciliation_id, rec.job_id, rec.preregistration_id,
                            rec.created_at.isoformat(), rec.model_dump_json()))

    def get_reconciliation(self, job_id: str) -> Reconciliation:
        row = self._conn.execute("SELECT body FROM reconciliations WHERE job_id=? ORDER BY created_at DESC LIMIT 1",
                                 (job_id,)).fetchone()
        if not row:
            raise NotFound(f"no reconciliation for job {job_id} (job still running?)")
        return Reconciliation.model_validate_json(row[0])

    # ------------------------------------------------------------------ confirmations
    def put_confirmation(self, c: Confirmation) -> None:
        self._conn.execute("INSERT INTO confirmations VALUES (?,?,?,?,?,?,?,?)",
                           (c.token, c.tool, c.tier.value, c.binding_hash, c.issued_at.isoformat(),
                            c.expires_at.isoformat(), json.dumps(c.summary, default=str), None))

    def get_confirmation(self, token: str) -> Confirmation | None:
        row = self._conn.execute("SELECT token,tool,tier,binding_hash,issued_at,expires_at,summary,consumed_at"
                                 " FROM confirmations WHERE token=?", (token,)).fetchone()
        if not row:
            return None
        return Confirmation(token=row[0], tool=row[1], tier=row[2], binding_hash=row[3],
                            issued_at=datetime.fromisoformat(row[4]), expires_at=datetime.fromisoformat(row[5]),
                            summary=json.loads(row[6]),
                            consumed_at=datetime.fromisoformat(row[7]) if row[7] else None)

    def consume_confirmation(self, token: str) -> bool:
        """Atomically mark a token consumed. Returns False if it was already consumed/missing."""
        cur = self._conn.execute("UPDATE confirmations SET consumed_at=? WHERE token=? AND consumed_at IS NULL",
                                 (datetime.now(UTC).isoformat(), token))
        return cur.rowcount == 1
