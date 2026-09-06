"""Secret handling: provider keys come from the environment only and are redacted from
every string that can leave the process (errors, logs, tool results)."""

from __future__ import annotations

import os
import re

# Environment variable names whose values are treated as secrets.
SECRET_ENV_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_TOKEN_ID", "_TOKEN_SECRET", "_PASSWORD")

# Provider key shapes we recognise even when they did not come from our own environment.
_PATTERNS = [
    re.compile(r"\brp_[A-Za-z0-9]{20,}\b"),  # RunPod
    re.compile(r"\bak-[A-Za-z0-9]{16,}\b"),  # Modal token id
    re.compile(r"\bas-[A-Za-z0-9]{16,}\b"),  # Modal token secret
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bsecret_[A-Za-z0-9]{16,}\b"),  # Lambda
    re.compile(r"(?i)(api[_-]?key|token|secret|password)(\s*[=:]\s*)([\"']?)([A-Za-z0-9_\-./+]{12,})"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
]

REDACTED = "[REDACTED]"


def secret_env_values() -> list[str]:
    vals = []
    for k, v in os.environ.items():
        if v and len(v) >= 8 and (k.endswith(SECRET_ENV_SUFFIXES) or k in {"MODAL_TOKEN_ID"}):
            vals.append(v)
    # Longest first so partial overlaps do not leak suffixes.
    return sorted(set(vals), key=len, reverse=True)


def redact(text: str) -> str:
    """Return ``text`` with every known secret replaced by ``[REDACTED]``."""
    if not text:
        return text
    out = text
    for v in secret_env_values():
        out = out.replace(v, REDACTED)
    for pat in _PATTERNS:
        if pat.groups >= 4:
            out = pat.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{REDACTED}", out)
        else:
            out = pat.sub(REDACTED, out)
    return out


def require_env(name: str) -> str:
    """Read a secret from the environment. Never accept secrets as CLI args or tool params."""
    v = os.environ.get(name)
    if not v:
        raise KeyError(f"environment variable {name} is not set")
    return v
