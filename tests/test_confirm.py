import pytest

from compute_prereg.confirm import Confirmer, binding_hash
from compute_prereg.errors import ConfirmationError
from compute_prereg.models import Tier


def test_token_lifecycle(engine):
    c = Confirmer(engine.ledger, ttl_seconds=60)
    params = {"envelope_id": "p1", "offer": {"price": 2.0}}
    tok = c.issue("provision", Tier.money, params, {"x": 1})
    assert tok.binding_hash == binding_hash("provision", params)
    with pytest.raises(ConfirmationError, match="differ"):
        c.redeem(tok.token, "provision", {"envelope_id": "p1", "offer": {"price": 2.1}})
    with pytest.raises(ConfirmationError, match="issued for"):
        c.redeem(tok.token, "terminate", params)
    got = c.redeem(tok.token, "provision", params)
    assert got.consumed_at is not None
    with pytest.raises(ConfirmationError, match="already used"):
        c.redeem(tok.token, "provision", params)
    with pytest.raises(ConfirmationError, match="unknown"):
        c.redeem("tok_nope", "provision", params)


def test_token_expiry(engine):
    c = Confirmer(engine.ledger, ttl_seconds=-1)
    tok = c.issue("preregister", Tier.write, {"a": 1}, {})
    with pytest.raises(ConfirmationError, match="expired"):
        c.redeem(tok.token, "preregister", {"a": 1})
