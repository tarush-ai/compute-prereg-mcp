"""Per-job ed25519 SSH keypairs. The private key lives only under the executor's key
directory (0600) for the life of the job and is deleted at termination. The public key
is registered only on the provisioned instance. Never a user's personal key."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


@dataclass(frozen=True)
class JobKeypair:
    job_id: str
    private_key_path: Path
    public_key: str  # OpenSSH single-line form


def generate_job_keypair(job_id: str, key_dir: str | Path) -> JobKeypair:
    key_dir = Path(key_dir)
    key_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(key_dir, 0o700)
    except OSError:
        pass
    priv = Ed25519PrivateKey.generate()
    pem = priv.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.OpenSSH,
                             serialization.NoEncryption())
    pub = priv.public_key().public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH).decode()
    path = key_dir / f"{job_id}.key"
    if path.exists():
        raise FileExistsError(f"key for {job_id} already exists; job keys are never reused")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(pem)
    return JobKeypair(job_id=job_id, private_key_path=path, public_key=f"{pub} prereg-{job_id}")


def destroy_job_key(job_id: str, key_dir: str | Path) -> bool:
    path = Path(key_dir) / f"{job_id}.key"
    if path.exists():
        try:  # overwrite before unlink so the bytes do not linger on disk
            with open(path, "r+b") as f:
                f.write(b"\0" * path.stat().st_size)
        except OSError:
            pass
        path.unlink()
        return True
    return False
