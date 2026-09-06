import asyncio
import json

from mcp.client import Client

from compute_prereg.models import JobStatus
from compute_prereg.server import build_server

EXPECTED_TOOLS = {"estimate_job", "list_offers", "get_preregistration", "list_preregistrations", "get_reconciliation",
                  "get_calibration", "job_status", "list_gpus", "preregister", "amend_preregistration", "provision", "terminate"}


def _data(res):
    if res.structured_content:
        return res.structured_content
    return json.loads(res.content[0].text)


def _err(res) -> str:
    assert res.is_error, "expected an error result"
    return res.content[0].text


async def test_tool_surface_and_tiering(engine):
    async with Client(build_server(engine)) as client:
        tools = await client.list_tools()
        names = {t.name for t in tools.tools}
        assert names == EXPECTED_TOOLS
        by_name = {t.name: t for t in tools.tools}
        assert by_name["provision"].description.startswith("[money")
        assert by_name["preregister"].description.startswith("[write")
        assert by_name["estimate_job"].description.startswith("[read")


async def test_full_loop_over_mcp(engine, mock_adapter, example):
    async with Client(build_server(engine)) as client:
        est = _data(await client.call_tool("estimate_job", {"entrypoint": str(example), "gpu_candidates": ["H100"],
                                                           "total_steps": 50}))
        estimate_id = est["estimate"]["estimate_id"]
        env = est["suggested_envelope"]
        env["spot_ok"] = True

        # Write tier: first call quotes, second executes; changed params void the token.
        q = _data(await client.call_tool("preregister", {"estimate_id": estimate_id, "envelope": env, "mode": "ask"}))
        assert q["status"] == "confirmation_required" and q["tier"] == "write" and "locks the budget" in q["consequence"]
        tok = q["confirm_token"]
        bad = _err(await client.call_tool("preregister", {"estimate_id": estimate_id, "envelope": {**env, "dollar_cap": env["dollar_cap"] * 10},
                                                          "mode": "ask", "confirm_token": tok}))
        assert "differ" in bad
        done = _data(await client.call_tool("preregister", {"estimate_id": estimate_id, "envelope": env, "mode": "ask",
                                                            "confirm_token": tok}))
        assert done["status"] == "executed"
        prereg_id = done["result"]["preregistration_id"]
        assert "already used" in _err(await client.call_tool("preregister", {"estimate_id": estimate_id, "envelope": env,
                                                                             "mode": "ask", "confirm_token": tok}))

        # Read tier: free.
        got = _data(await client.call_tool("get_preregistration", {"preregistration_id": prereg_id}))
        assert got["preregistration"]["signature"].startswith("ed25519:")
        listing = _data(await client.call_tool("list_preregistrations", {}))
        assert listing["preregistrations"][0]["preregistration_id"] == prereg_id

        offers = _data(await client.call_tool("list_offers", {"envelope_id": prereg_id, "policy": "cheapest"}))
        assert offers["fitting"] and offers["nearest_misses"]
        offer_id = offers["fitting"][0]["offer"]["offer_id"]

        # Nothing raises a cap over MCP.
        assert "may not raise" in _err(await client.call_tool(
            "amend_preregistration", {"preregistration_id": prereg_id, "changes": {"dollar_cap": env["dollar_cap"] * 5}}))
        # Non-conforming offers cannot be confirmed.
        miss_id = offers["nearest_misses"][0]["offer"]["offer_id"]
        assert "exceeds the envelope" in _err(await client.call_tool("provision", {"envelope_id": prereg_id, "offer_id": miss_id}))

        # Money tier: quote -> token bound to hash(offer ∪ envelope) -> execute.
        q = _data(await client.call_tool("provision", {"envelope_id": prereg_id, "offer_id": offer_id}))
        assert q["status"] == "confirmation_required" and q["tier"] == "money"
        assert q["summary"]["cost_at_hours_max"] <= q["summary"]["dollar_cap"]
        assert "ed25519" in q["summary"]["ssh"]
        assert "unknown confirmation token" in _err(await client.call_tool(
            "provision", {"envelope_id": prereg_id, "offer_id": offer_id, "confirm_token": "tok_forged"}))
        started = _data(await client.call_tool("provision", {"envelope_id": prereg_id, "offer_id": offer_id,
                                                             "confirm_token": q["confirm_token"]}))
        job_id = started["result"]["job_id"]

        for _ in range(300):
            await asyncio.sleep(0.2)
            st = _data(await client.call_tool("job_status", {"job_id": job_id}))
            if not st["active"]:
                break
        assert st["job"]["status"] == JobStatus.completed
        assert st["job"]["telemetry"]["steps_completed"] == 50
        assert st["job"]["ssh_public_key"] is not None
        rec = _data(await client.call_tool("get_reconciliation", {"job_id": job_id}))["reconciliation"]
        assert rec["job_id"] == job_id and rec["steps_completed"] == 50
        cal = _data(await client.call_tool("get_calibration", {"gpu": "H100-SXM-80GB"}))
        assert cal["priors"][0]["source"] == "prior"
        gpus = _data(await client.call_tool("list_gpus", {}))
        assert gpus["server"]["adapters"] == ["mock"] and len(gpus["gpus"]) >= 12
        assert mock_adapter.terminated  # instance gone, key revoked


async def test_terminate_over_mcp(engine, mock_adapter, example):
    async with Client(build_server(engine)) as client:
        est = _data(await client.call_tool("estimate_job", {"entrypoint": str(example), "gpu_candidates": ["H100"],
                                                           "total_steps": 10_000_000}))
        env = {**est["suggested_envelope"], "spot_ok": True, "hours_point": 5.0, "dollar_cap": 100.0}
        q = _data(await client.call_tool("preregister", {"estimate_id": est["estimate"]["estimate_id"], "envelope": env, "mode": "bounded"}))
        prereg_id = _data(await client.call_tool("preregister", {"estimate_id": est["estimate"]["estimate_id"], "envelope": env,
                                                                 "mode": "bounded", "confirm_token": q["confirm_token"]}))["result"]["preregistration_id"]
        await client.call_tool("list_offers", {"envelope_id": prereg_id})
        q = _data(await client.call_tool("provision", {"envelope_id": prereg_id}))  # bounded: auto-pick
        job_id = _data(await client.call_tool("provision", {"envelope_id": prereg_id, "confirm_token": q["confirm_token"]}))["result"]["job_id"]
        for _ in range(100):
            await asyncio.sleep(0.2)
            if _data(await client.call_tool("job_status", {"job_id": job_id}))["job"]["status"] == "running":
                break
        q = _data(await client.call_tool("terminate", {"job_id": job_id}))
        assert q["tier"] == "money" and "destroys" in q["summary"]
        _data(await client.call_tool("terminate", {"job_id": job_id, "confirm_token": q["confirm_token"]}))
        for _ in range(100):
            await asyncio.sleep(0.2)
            st = _data(await client.call_tool("job_status", {"job_id": job_id}))
            if not st["active"]:
                break
        assert st["job"]["status"] == "terminated"
        assert mock_adapter.terminated == [st["job"]["instance"]["instance_id"]]
