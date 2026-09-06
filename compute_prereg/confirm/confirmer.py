"""Two-call confirmation handshake.

Call 1 (quote): the caller presents (tool, params) -> summary + single-use token bound to
hash(tool, canonical params). Call 2 (execute): the caller returns the token with the
*identical* params. Any change in the bound params invalidates the token; the caller must
re-quote. Tokens expire, are consumed atomically, and are enforced in the tier wrapper —
not in tool descriptions.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import timedelta
from typing import Any

from ..errors import ConfirmationError
from ..ledger.signing import canonical_json
from ..models import Confirmation, Tier, utcnow

DEFAULT_TTL_SECONDS = 600


def binding_hash(tool: str, params: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json({"tool": tool, "params": params})).hexdigest()


class Confirmer:
    def __init__(self, ledger, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self.ledger = ledger
        self.ttl = ttl_seconds

    def issue(self, tool: str, tier: Tier, params: dict[str, Any], summary: dict[str, Any]) -> Confirmation:
        token = "tok_" + secrets.token_urlsafe(24)
        c = Confirmation(token=token, tool=tool, tier=tier, binding_hash=binding_hash(tool, params),
                         expires_at=utcnow() + timedelta(seconds=self.ttl), summary=summary)
        self.ledger.put_confirmation(c)
        return c

    def redeem(self, token: str, tool: str, params: dict[str, Any]) -> Confirmation:
        c = self.ledger.get_confirmation(token)
        if c is None:
            raise ConfirmationError("unknown confirmation token; call the tool without a token to get a quote")
        if c.tool != tool:
            raise ConfirmationError(f"token was issued for {c.tool}, not {tool}")
        if c.consumed_at is not None:
            raise ConfirmationError("confirmation token already used; tokens are single-use — request a new quote")
        if utcnow() > c.expires_at:
            raise ConfirmationError("confirmation token expired; request a new quote")
        if c.binding_hash != binding_hash(tool, params):
            raise ConfirmationError("parameters differ from the quoted ones; the token is void — request a new quote")
        if not self.ledger.consume_confirmation(token):
            raise ConfirmationError("confirmation token already used")
        c.consumed_at = utcnow()
        return c
