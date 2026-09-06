import pytest

from compute_prereg.errors import LedgerIntegrityError, PolicyError
from compute_prereg.models import PreregStatus
from tests.conftest import make_prereg


def test_sign_verify_and_tamper(engine, example):
    _, p = make_prereg(engine, example)
    got = engine.ledger.get_preregistration(p.preregistration_id)
    assert got.signature.startswith("ed25519:") and got.record_hash.startswith("sha256:")
    assert got.status == PreregStatus.open
    assert all(ok for _, ok in engine.ledger.verify_all())
    # Tamper with the stored body: the cap is raised behind the ledger's back.
    conn = engine.ledger._conn
    body = conn.execute("SELECT body FROM preregistrations WHERE id=?", (p.preregistration_id,)).fetchone()[0]
    tampered = body.replace(f'"dollar_cap":{p.envelope.dollar_cap}', '"dollar_cap":999999.0')
    assert tampered != body
    conn.execute("UPDATE preregistrations SET body=? WHERE id=?", (tampered, p.preregistration_id))
    with pytest.raises(LedgerIntegrityError):
        engine.ledger.get_preregistration(p.preregistration_id)


def test_agent_cannot_raise_cap_but_human_can(engine, example):
    _, p = make_prereg(engine, example)
    cap = p.envelope.dollar_cap
    with pytest.raises(PolicyError, match="dollar_cap"):
        engine.ledger.amend(p.preregistration_id, {"dollar_cap": cap * 2}, actor="agent")
    with pytest.raises(PolicyError, match="hours"):
        engine.ledger.amend(p.preregistration_id, {"hours_tolerance": 5.0}, actor="agent")
    lowered = engine.ledger.amend(p.preregistration_id, {"dollar_cap": cap / 2}, actor="agent")
    assert lowered.supersedes == p.preregistration_id and lowered.envelope.dollar_cap == cap / 2
    assert engine.ledger.status(p.preregistration_id) == PreregStatus.superseded
    raised = engine.ledger.amend(lowered.preregistration_id, {"dollar_cap": cap * 3}, actor="human", note="approved")
    assert raised.envelope.dollar_cap == cap * 3 and raised.amended_by == "human"
    assert raised.signature != lowered.signature
    with pytest.raises(PolicyError):  # superseded records are frozen
        engine.ledger.amend(p.preregistration_id, {"dollar_cap": 1.0}, actor="human")
    with pytest.raises(PolicyError):
        engine.ledger.amend(raised.preregistration_id, {"nonsense": 1}, actor="human")
    assert [e["status"] for e in engine.ledger.events(p.preregistration_id)] == ["open", "superseded"]


def test_amend_blocked_while_running(engine, example):
    _, p = make_prereg(engine, example)
    engine.ledger.set_status(p.preregistration_id, PreregStatus.running, "job")
    with pytest.raises(PolicyError, match="terminate"):
        engine.ledger.amend(p.preregistration_id, {"dollar_cap": 0.5}, actor="human")


def test_estimate_roundtrip(engine, example):
    est, _ = make_prereg(engine, example)
    assert engine.ledger.get_estimate(est.estimate_id).flops_forward == est.flops_forward
