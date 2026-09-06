"""MCP surface (CONTEXT.md §5.2). Tools are tiered like GPU.ai:

  read  : free
  write : two-call confirm (quote -> token -> execute); consequence stated in the quote
  money : quote -> single-use token bound to hash(offer ∪ envelope) -> execute

Tiering is enforced by the `gate` wrapper below, not by tool descriptions. No tool raises
a dollar cap: that is a human action in the CLI (`compute-prereg amend`).

Run: `compute-prereg-mcp` (stdio). Env: COMPUTE_PREREG_DATA_DIR, COMPUTE_PREREG_ADAPTERS
(comma list; default autodetect), COMPUTE_PREREG_DEMO=1 (adds the local mock adapter),
provider keys (RUNPOD_API_KEY, MODAL_TOKEN_ID/SECRET, SkyPilot's own cloud credentials).

Loudly: running an MCP client with permission prompts disabled turns the confirmation
handshake into a formality. Set an org-level spend limit at each provider as the ceiling
no client mode can talk past.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .engine import Engine, EngineConfig
from .errors import PreregError
from .models import Tier
from .security import redact

INSTRUCTIONS = """Compute preregistration engine. The loop is:
estimate_job -> preregister (locks a signed envelope BEFORE any provider is contacted) ->
list_offers (brokered, filtered to the envelope) -> provision (confirmed against the envelope, not
the provider's quote) -> job_status -> get_reconciliation (predicted vs measured; feeds calibration).

Tiers: read tools return immediately. Write tools (preregister, amend_preregistration) and money
tools (provision, terminate) return {"status": "confirmation_required", "confirm_token": ...} on the
first call; show the summary to the human, then call again with identical parameters plus
confirm_token. Tokens are single-use, expire, and are void if any parameter changes.

No tool can raise a dollar cap or widen hours; only a human can, outside this server, via the CLI.
If an offer exceeds the envelope it cannot be confirmed — amend_preregistration (a new record) or
choose another offer. Never silently widen. When estimate confidence is 'low', tell the human before
preregistering."""


def _dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, list):
        return [_dump(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _dump(v) for k, v in obj.items()}
    return obj


def build_server(engine: Engine) -> MCPServer:
    server = MCPServer("compute-prereg", instructions=INSTRUCTIONS, version="0.1.0")

    async def gate(*, tier: Tier, tool: str, consequence: str, bind: Callable[[], tuple[dict, dict]],
                   execute: Callable[[], Awaitable[Any] | Any], token: str | None) -> dict[str, Any]:
        """Two-call handshake. Call 1 (no token): resolve bound params + summary, issue a token.
        Call 2 (token): redeem against freshly re-resolved bound params, then execute."""
        try:
            bound, summary = bind()
            if token is None:
                conf = engine.confirmer.issue(tool, tier, bound, summary)
                return {"status": "confirmation_required", "tier": tier, "tool": tool, "consequence": consequence,
                        "summary": _dump(summary), "confirm_token": conf.token,
                        "expires_at": conf.expires_at.isoformat(),
                        "next": "show the summary to the human; if approved, call again with the same parameters and confirm_token"}
            engine.confirmer.redeem(token, tool, bound)
            result = execute()
            if hasattr(result, "__await__"):
                result = await result
            return {"status": "executed", "tier": tier, "tool": tool, "result": _dump(result)}
        except PreregError as e:
            raise ToolError(redact(f"{type(e).__name__}: {e}")) from None
        except (ValueError, KeyError) as e:
            raise ToolError(redact(f"{type(e).__name__}: {e}")) from None

    def guard(fn):
        async def wrapper(*a, **k):
            try:
                r = fn(*a, **k)
                if hasattr(r, "__await__"):
                    r = await r
                return _dump(r)
            except PreregError as e:
                raise ToolError(redact(f"{type(e).__name__}: {e}")) from None
            except (ValueError, KeyError, FileNotFoundError) as e:
                raise ToolError(redact(f"{type(e).__name__}: {e}")) from None
        return wrapper

    # ------------------------------------------------------------------ read tier
    @server.tool(description="[read] Estimate FLOPs, peak memory, and per-GPU hours/$ bands for a job. Pass "
                             "`entrypoint` (a .py exposing prereg_build(config)) and/or `model_spec` "
                             "({params, model_class, batch_tokens|seq_len+batch_size, dtype, total_steps, "
                             "hidden_size, num_layers, hf_config...}). Returns an Estimate with a confidence "
                             "(exact/calibrated/prior/low) and a suggested envelope to start from.")
    async def estimate_job(entrypoint: str | None = None, config: dict[str, Any] | None = None,
                           model_spec: dict[str, Any] | None = None, gpu_candidates: list[str] | None = None,
                           total_steps: int | None = None) -> dict[str, Any]:
        @guard
        def _run():
            est = engine.estimate_job(entrypoint=entrypoint, config=config, model_spec=model_spec,
                                      gpu_candidates=gpu_candidates, total_steps=total_steps)
            suggested = Engine.suggest_envelope(est) if est.per_gpu else None
            return {"estimate": est, "suggested_envelope": suggested,
                    "note": "confidence=low means tracing failed; a human must see this before preregistering"}
        return await _run()

    @server.tool(description="[read] Brokered offers for a preregistered envelope, across configured providers, "
                             "each annotated with whether it fits and which envelope dimensions fail. "
                             "policy: cheapest | fastest-to-start | most-reliable | lowest-carbon.")
    async def list_offers(envelope_id: str, policy: str = "cheapest", providers: list[str] | None = None) -> dict[str, Any]:
        @guard
        async def _run():
            fits = await engine.list_offers(envelope_id, policy, providers)  # type: ignore[arg-type]
            return {"envelope_id": envelope_id, "policy": policy,
                    "fitting": [f for f in fits if f.fits], "nearest_misses": [f for f in fits if not f.fits][:10],
                    "adapter_errors": engine.broker.last_errors, "configured_adapters": sorted(engine.broker.adapters)}
        return await _run()

    @server.tool(description="[read] Fetch a preregistration record (signature verified) with its status history.")
    async def get_preregistration(preregistration_id: str) -> dict[str, Any]:
        @guard
        def _run():
            p = engine.get_preregistration(preregistration_id)
            return {"preregistration": p, "events": engine.ledger.events(preregistration_id),
                    "jobs": [j.job_id for j in engine.ledger.jobs_for(preregistration_id)]}
        return await _run()

    @server.tool(description="[read] List preregistrations, optionally filtered by status.")
    async def list_preregistrations(status: str | None = None) -> dict[str, Any]:
        @guard
        def _run():
            return {"preregistrations": [
                {"preregistration_id": p.preregistration_id, "created_at": p.created_at, "status": p.status, "mode": p.mode,
                 "supersedes": p.supersedes, "dollar_cap": p.envelope.dollar_cap, "hours_max": round(p.envelope.hours_max, 2),
                 "gpu_classes": p.envelope.gpu_classes, "entrypoint": p.job.entrypoint}
                for p in engine.list_preregistrations(status)]}
        return await _run()

    @server.tool(description="[read] Predicted vs measured report for a finished job: hours, $, MFU, gap attribution, recommendations.")
    async def get_reconciliation(job_id: str) -> dict[str, Any]:
        @guard
        def _run():
            return {"reconciliation": engine.get_reconciliation(job_id)}
        return await _run()

    @server.tool(description="[read] Calibration store: measured MFU distributions per (provider, gpu, model class, ...) and priors.")
    async def get_calibration(provider: str | None = None, gpu: str | None = None, model_class: str | None = None) -> dict[str, Any]:
        @guard
        def _run():
            return engine.get_calibration(provider, gpu, model_class)
        return await _run()

    @server.tool(description="[read] Live status and telemetry summary of a job (steps, step time, memory, accrued $).")
    async def job_status(job_id: str) -> dict[str, Any]:
        @guard
        def _run():
            j = engine.job_status(job_id)
            return {"job": j, "active": engine.executor.is_active(job_id), "elapsed_hours": round(j.elapsed_hours, 4)}
        return await _run()

    @server.tool(description="[read] Known GPU roofline rows (peak TFLOPS by dtype, HBM bandwidth, VRAM, reference $/h) and server info.")
    async def list_gpus() -> dict[str, Any]:
        from .est.gpus import GPUS

        return _dump({"gpus": list(GPUS.values()), "server": engine.describe()})

    # ------------------------------------------------------------------ write tier
    @server.tool(description="[write, confirm] Lock an estimate into a signed, immutable envelope. envelope: "
                             "{gpu_classes, min_vram_gb, hours_point, hours_tolerance, dollar_cap, spot_ok, "
                             "max_preemptions, region_allow, provider_allow, provider_deny, min/max_gpu_count, "
                             "checkpoint_interval_steps}. mode: ask | bounded | locked. First call returns a "
                             "confirm_token; second call with the same params executes.")
    async def preregister(estimate_id: str, envelope: dict[str, Any], mode: str = "ask",
                          confirm_token: str | None = None) -> dict[str, Any]:
        params = {"estimate_id": estimate_id, "envelope": envelope, "mode": mode}
        return await gate(
            tier=Tier.write, tool="preregister",
            consequence="This locks the budget: the envelope becomes a signed contract that every later offer is checked against. "
                        "Only a human can raise its dollar cap afterwards.",
            bind=lambda: (params, engine.preregister_quote(estimate_id, envelope, mode)),
            execute=lambda: engine.preregister(estimate_id, envelope, mode), token=confirm_token)

    @server.tool(description="[write, confirm] Amend an envelope: creates a NEW record superseding the old one. Agents may "
                             "only tighten (lower cap/hours, fewer GPUs); raising is human-only via the CLI.")
    async def amend_preregistration(preregistration_id: str, changes: dict[str, Any], note: str | None = None,
                                    confirm_token: str | None = None) -> dict[str, Any]:
        params = {"preregistration_id": preregistration_id, "changes": changes}

        def bind():
            q = engine.amend_quote(preregistration_id, changes, actor="agent")
            if not q.get("allowed", True):
                raise PreregError(q["reason"])
            return params, q

        return await gate(tier=Tier.write, tool="amend_preregistration",
                          consequence="Supersedes the current envelope with a new signed record; history is kept.",
                          bind=bind, execute=lambda: engine.amend(preregistration_id, changes, actor="agent", note=note),
                          token=confirm_token)

    # ------------------------------------------------------------------ money tier
    @server.tool(description="[money, quote->confirm] Provision an offer for a preregistered envelope and start the job. "
                             "The token is bound to hash(offer ∪ envelope): a re-priced offer or an amended envelope voids it. "
                             "offer_id may be omitted only in bounded/locked mode (best conforming cached offer). Returns job_id.")
    async def provision(envelope_id: str, offer_id: str | None = None, confirm_token: str | None = None) -> dict[str, Any]:
        return await gate(tier=Tier.money, tool="provision",
                          consequence="Spends money: starts billing on the provider until the job ends, the envelope is breached, or terminate is confirmed.",
                          bind=lambda: engine.provision_binding(envelope_id, offer_id),
                          execute=lambda: engine.provision(envelope_id, offer_id), token=confirm_token)

    @server.tool(description="[money, quote->confirm] Terminate a running job: quotes what is destroyed and what it has cost so far, then on confirm "
                             "requests a checkpoint, stops the script, terminates the instance (revoking the job key) and reconciles.")
    async def terminate(job_id: str, confirm_token: str | None = None) -> dict[str, Any]:
        return await gate(tier=Tier.money, tool="terminate",
                          consequence="Destroys the instance and un-checkpointed state; ends billing.",
                          bind=lambda: ({"job_id": job_id}, engine.terminate_quote(job_id)),
                          execute=lambda: engine.terminate(job_id), token=confirm_token)

    return server


def engine_from_env() -> Engine:
    adapters_env = os.environ.get("COMPUTE_PREREG_ADAPTERS")
    cfg = EngineConfig(
        adapters=[a.strip() for a in adapters_env.split(",") if a.strip()] if adapters_env else None,
        demo=os.environ.get("COMPUTE_PREREG_DEMO", "") in ("1", "true", "yes"),
    )
    return Engine(cfg)


def main() -> None:
    build_server(engine_from_env()).run("stdio")


if __name__ == "__main__":
    main()
