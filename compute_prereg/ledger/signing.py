"""Ed25519 signing of ledger records over canonical JSON."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False).encode()


def record_hash(body: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(body)).hexdigest()


class Signer:
    """Holds the ledger's signing key. Generated on first use, stored 0600."""

    def __init__(self, key_path: str | Path | None):
        self.key_path = Path(key_path) if key_path else None
        self._key = self._load_or_create()

    def _load_or_create(self) -> Ed25519PrivateKey:
        if self.key_path and self.key_path.exists():
            return serialization.load_pem_private_key(self.key_path.read_bytes(), password=None)  # type: ignore[return-value]
        key = Ed25519PrivateKey.generate()
        if self.key_path:
            self.key_path.parent.mkdir(parents=True, exist_ok=True)
            pem = key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
            )
            fd = os.open(self.key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(pem)
        return key

    @property
    def public_key_hex(self) -> str:
        raw = self._key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        return raw.hex()

    def sign(self, body: dict[str, Any]) -> tuple[str, str]:
        h = record_hash(body)
        sig = self._key.sign(h.encode())
        return h, "ed25519:" + base64.b64encode(sig).decode()

    def verify(self, body: dict[str, Any], h: str, signature: str) -> bool:
        if record_hash(body) != h:
            return False
        return verify_with_public(self._key.public_key(), h, signature)


def verify_with_public(pub: Ed25519PublicKey, h: str, signature: str) -> bool:
    try:
        algo, _, b64 = signature.partition(":")
        if algo != "ed25519":
            return False
        pub.verify(base64.b64decode(b64), h.encode())
        return True
    except Exception:
        return False
