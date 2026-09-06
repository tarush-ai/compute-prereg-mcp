"""Launchers: how the executor talks to a provisioned instance.

  LocalLauncher  - kind="local"    (mock adapter, tests, --demo): subprocess on this machine
  SSHLauncher    - kind="ssh"      : ssh/rsync over the per-job key
  ModalLauncher  - kind="function" : a Modal function wrapping the shim
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..errors import ProvisionError
from ..models import Instance
from ..security import redact
from .telemetry import parse_dmon

PACKAGE_DIR = Path(__file__).resolve().parent.parent  # compute_prereg/


@dataclass
class RunState:
    running: bool
    exit_code: int | None = None
    detail: str | None = None


class Launcher(Protocol):
    async def sync(self, instance: Instance, code_dir: Path) -> None: ...
    async def start(self, instance: Instance, entrypoint: str, args: list[str], env: dict[str, str],
                    profile_every: int, profile_steps: int) -> None: ...
    async def poll(self, instance: Instance) -> RunState: ...
    async def read_telemetry(self, instance: Instance) -> list[dict[str, Any]]: ...
    async def request_checkpoint(self, instance: Instance) -> None: ...
    async def stop(self, instance: Instance) -> None: ...
    async def cleanup(self, instance: Instance) -> None: ...


def _parse_jsonl(text: str) -> list[dict[str, Any]]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# ---------------------------------------------------------------------------- local


class LocalLauncher:
    def __init__(self, work_root: str | Path, python: str | None = None):
        self.work_root = Path(work_root)
        self.python = python or sys.executable
        self._procs: dict[str, asyncio.subprocess.Process] = {}

    def _dir(self, instance: Instance) -> Path:
        return self.work_root / instance.instance_id

    async def sync(self, instance: Instance, code_dir: Path) -> None:
        d = self._dir(instance)
        if (d / "job").exists():
            shutil.rmtree(d / "job")
        shutil.copytree(code_dir, d / "job", ignore=shutil.ignore_patterns(".git", "__pycache__", ".venv", "*.pt"))

    async def start(self, instance, entrypoint, args, env, profile_every, profile_steps) -> None:
        d = self._dir(instance)
        cmd = [self.python, "-m", "compute_prereg.shim", "--telemetry", str(d / "telemetry.jsonl"),
               "--profile-every", str(profile_every), "--profile-steps", str(profile_steps), "--",
               str(d / "job" / entrypoint), *args]
        full_env = {**os.environ, **env, "PYTHONPATH": str(PACKAGE_DIR.parent) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        log = open(d / "run.log", "ab")
        self._procs[instance.instance_id] = await asyncio.create_subprocess_exec(
            *cmd, cwd=d / "job", env=full_env, stdout=log, stderr=asyncio.subprocess.STDOUT)

    async def poll(self, instance: Instance) -> RunState:
        p = self._procs.get(instance.instance_id)
        if p is None:
            return RunState(running=False, exit_code=None, detail="not started")
        if p.returncode is None:
            return RunState(running=True)
        return RunState(running=False, exit_code=p.returncode)

    async def read_telemetry(self, instance: Instance) -> list[dict[str, Any]]:
        f = self._dir(instance) / "telemetry.jsonl"
        return _parse_jsonl(f.read_text()) if f.exists() else []

    async def request_checkpoint(self, instance: Instance) -> None:
        (self._dir(instance) / "CHECKPOINT_REQUESTED").touch()

    async def stop(self, instance: Instance) -> None:
        p = self._procs.get(instance.instance_id)
        if p and p.returncode is None:
            p.terminate()
            try:
                await asyncio.wait_for(p.wait(), 10)
            except asyncio.TimeoutError:
                p.kill()

    async def cleanup(self, instance: Instance) -> None:
        self._procs.pop(instance.instance_id, None)

    def log_path(self, instance: Instance) -> Path:
        return self._dir(instance) / "run.log"


# ---------------------------------------------------------------------------- ssh


class SSHLauncher:
    """ssh/rsync subprocesses using the per-job private key. Host keys are pinned per
    job in a dedicated known_hosts file (accept-new), never the user's ~/.ssh."""

    def __init__(self, key_path: Path, known_hosts: Path, remote_root: str = "~/prereg", remote_python: str = "python3"):
        self.key_path = Path(key_path)
        self.known_hosts = Path(known_hosts)
        self.remote_root = remote_root
        self.remote_python = remote_python

    def _ssh_opts(self, instance: Instance) -> list[str]:
        return ["-i", str(self.key_path), "-p", str(instance.port), "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"UserKnownHostsFile={self.known_hosts}", "-o", "IdentitiesOnly=yes",
                "-o", "ConnectTimeout=20", "-o", "BatchMode=yes"]

    def _target(self, instance: Instance) -> str:
        return f"{instance.user}@{instance.host}"

    def _rdir(self, instance: Instance) -> str:
        return f"{self.remote_root}/{instance.instance_id}"

    async def _run(self, argv: list[str], check: bool = True) -> tuple[int, str]:
        p = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        out, _ = await p.communicate()
        text = out.decode(errors="replace")
        if check and p.returncode != 0:
            raise ProvisionError(redact(f"{argv[0]} failed ({p.returncode}): {text[-2000:]}"))
        return p.returncode or 0, text

    async def ssh(self, instance: Instance, command: str, check: bool = True) -> tuple[int, str]:
        return await self._run(["ssh", *self._ssh_opts(instance), self._target(instance), command], check=check)

    async def sync(self, instance: Instance, code_dir: Path) -> None:
        # Wait for sshd.
        for _ in range(30):
            rc, _ = await self.ssh(instance, "true", check=False)
            if rc == 0:
                break
            await asyncio.sleep(5)
        else:
            raise ProvisionError("ssh never became reachable")
        rdir = self._rdir(instance)
        await self.ssh(instance, f"mkdir -p {rdir}/job {rdir}/pkg")
        ssh_cmd = "ssh " + " ".join(shlex.quote(o) for o in self._ssh_opts(instance))
        excl = ["--exclude", ".git", "--exclude", "__pycache__", "--exclude", ".venv"]
        await self._run(["rsync", "-az", *excl, "-e", ssh_cmd, f"{code_dir}/", f"{self._target(instance)}:{rdir}/job/"])
        await self._run(["rsync", "-az", *excl, "-e", ssh_cmd, f"{PACKAGE_DIR}/", f"{self._target(instance)}:{rdir}/pkg/compute_prereg/"])

    async def start(self, instance, entrypoint, args, env, profile_every, profile_steps) -> None:
        rdir = self._rdir(instance)
        envs = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        argstr = " ".join(shlex.quote(a) for a in args)
        cmd = (f"cd {rdir} && (nvidia-smi dmon -s um -d 1 > dmon.log 2>/dev/null &) ; "
               f"cd {rdir}/job && (PYTHONPATH={rdir}/pkg {envs} nohup {self.remote_python} -m compute_prereg.shim "
               f"--telemetry {rdir}/telemetry.jsonl --profile-every {profile_every} --profile-steps {profile_steps} "
               f"-- {shlex.quote(entrypoint)} {argstr} > {rdir}/run.log 2>&1; echo $? > {rdir}/exit.code) "
               f"> /dev/null 2>&1 & echo $! > {rdir}/run.pid")
        await self.ssh(instance, cmd)

    async def poll(self, instance: Instance) -> RunState:
        rdir = self._rdir(instance)
        rc, out = await self.ssh(instance, f"cat {rdir}/exit.code 2>/dev/null || echo RUNNING", check=False)
        if rc != 0:
            return RunState(running=True, detail="ssh unreachable")
        out = out.strip()
        if out == "RUNNING":
            return RunState(running=True)
        try:
            return RunState(running=False, exit_code=int(out))
        except ValueError:
            return RunState(running=False, exit_code=None, detail=out[-200:])

    async def read_telemetry(self, instance: Instance) -> list[dict[str, Any]]:
        rdir = self._rdir(instance)
        rc, out = await self.ssh(instance, f"cat {rdir}/telemetry.jsonl 2>/dev/null; echo '---DMON---'; tail -n 30 {rdir}/dmon.log 2>/dev/null", check=False)
        if rc != 0:
            raise ProvisionError("telemetry read failed: ssh unreachable")
        jsonl, _, dmon = out.partition("---DMON---")
        return _parse_jsonl(jsonl) + parse_dmon(dmon)

    async def request_checkpoint(self, instance: Instance) -> None:
        await self.ssh(instance, f"touch {self._rdir(instance)}/CHECKPOINT_REQUESTED", check=False)

    async def stop(self, instance: Instance) -> None:
        rdir = self._rdir(instance)
        await self.ssh(instance, f"pkill -TERM -P $(cat {rdir}/run.pid) 2>/dev/null; kill $(cat {rdir}/run.pid) 2>/dev/null; true", check=False)

    async def cleanup(self, instance: Instance) -> None:
        return None


# ---------------------------------------------------------------------------- modal


class ModalLauncher:
    """Runs the shim + entrypoint inside a Modal function. Telemetry is returned when the
    function finishes (Modal has no live file access), so the watchdog sees only elapsed
    time until then."""

    def __init__(self, work_root: str | Path, image_packages: tuple[str, ...] = ("torch",)):
        self.work_root = Path(work_root)
        self.image_packages = image_packages
        self._calls: dict[str, Any] = {}
        self._results: dict[str, dict[str, Any]] = {}
        self._code: dict[str, Path] = {}

    async def sync(self, instance: Instance, code_dir: Path) -> None:
        self._code[instance.instance_id] = code_dir

    async def start(self, instance, entrypoint, args, env, profile_every, profile_steps) -> None:
        import modal  # lazy

        code_dir = self._code[instance.instance_id]
        image = (modal.Image.debian_slim(python_version="3.12").pip_install(*self.image_packages)
                 .add_local_dir(str(code_dir), remote_path="/job")
                 .add_local_dir(str(PACKAGE_DIR), remote_path="/pkg/compute_prereg"))
        app = modal.App(instance.provider_ref.get("app_name", f"prereg-{instance.instance_id}"))

        @app.function(image=image, gpu=instance.provider_ref["gpu"], timeout=24 * 3600)
        def run_job(entry: str, argv: list[str], envs: dict[str, str], pe: int, ps: int) -> dict[str, Any]:
            import os as _os
            import subprocess
            import sys as _sys

            _os.environ.update(envs)
            cmd = [_sys.executable, "-m", "compute_prereg.shim", "--telemetry", "/tmp/telemetry.jsonl",
                   "--profile-every", str(pe), "--profile-steps", str(ps), "--", f"/job/{entry}", *argv]
            p = subprocess.run(cmd, cwd="/job", env={**_os.environ, "PYTHONPATH": "/pkg"}, capture_output=True, text=True)
            tel = open("/tmp/telemetry.jsonl").read() if _os.path.exists("/tmp/telemetry.jsonl") else ""
            return {"exit_code": p.returncode, "telemetry": tel, "log": (p.stdout + p.stderr)[-20000:]}

        self._app_ctx = app.run()
        self._app_ctx.__enter__()
        call = run_job.spawn(entrypoint, args, env, profile_every, profile_steps)
        self._calls[instance.instance_id] = call
        instance.provider_ref["function_call_id"] = call.object_id

    async def poll(self, instance: Instance) -> RunState:
        call = self._calls.get(instance.instance_id)
        if call is None:
            return RunState(running=False, detail="not started")
        try:
            res = call.get(timeout=0)
        except TimeoutError:
            return RunState(running=True)
        except Exception as e:
            return RunState(running=False, exit_code=1, detail=redact(str(e)))
        self._results[instance.instance_id] = res
        return RunState(running=False, exit_code=int(res.get("exit_code", 1)))

    async def read_telemetry(self, instance: Instance) -> list[dict[str, Any]]:
        res = self._results.get(instance.instance_id)
        return _parse_jsonl(res["telemetry"]) if res else []

    async def request_checkpoint(self, instance: Instance) -> None:
        return None  # no live filesystem; scripts on Modal cannot be signalled

    async def stop(self, instance: Instance) -> None:
        call = self._calls.get(instance.instance_id)
        if call is not None:
            try:
                call.cancel()
            except Exception:
                pass

    async def cleanup(self, instance: Instance) -> None:
        ctx = getattr(self, "_app_ctx", None)
        if ctx is not None:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
