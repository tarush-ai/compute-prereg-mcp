"""Prefixed, time-sortable identifiers (ULID-style: 48-bit ms timestamp + 80 bits random)."""

from __future__ import annotations

import os
import time

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32


def _b32(value: int, width: int) -> str:
    out = []
    for _ in range(width):
        out.append(_ALPHABET[value & 31])
        value >>= 5
    return "".join(reversed(out))


def ulid() -> str:
    ts = int(time.time() * 1000)
    rand = int.from_bytes(os.urandom(10), "big")
    return _b32(ts, 10) + _b32(rand, 16)


def new_id(prefix: str) -> str:
    return f"{prefix}_{ulid()}"


def estimate_id() -> str:
    return new_id("est")


def preregistration_id() -> str:
    return new_id("prereg")


def offer_id() -> str:
    return new_id("off")


def job_id() -> str:
    return new_id("job")


def token_id() -> str:
    return new_id("tok")


def reconciliation_id() -> str:
    return new_id("recon")


def instance_id() -> str:
    return new_id("inst")
