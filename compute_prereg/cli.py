"""CLI. This is where the *human* acts: raising a dollar cap happens here, never via MCP.

  compute-prereg estimate examples/tiny_transformer.py [--steps N] [--gpus H100 A100]
  compute-prereg ledger list | show ID | events ID | verify
  compute-prereg amend ID --set dollar_cap=40 --set hours_tolerance=0.5
  compute-prereg jobs | job JOB_ID | recon JOB_ID | calib [--gpu G] | gpus
  compute-prereg demo [--entrypoint P] [--steps N] [--mode ask|bounded|locked]
  compute-prereg serve
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from .engine import Engine, EngineConfig
from .errors import PreregError
from .security import redact


def _dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, list):
        return [_dump(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _dump(v) for k, v in obj.items()}
    return obj


def _print(obj: Any) -> None:
    print(json.dumps(_dump(obj), indent=2, default=str))


def _parse_value(v: str) -> Any:
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def _engine(args) -> Engine:
    cfg = EngineConfig(demo=bool(getattr(args, "demo", False)))
    if getattr(args, "data_dir", None):
        cfg.data_dir = Path(args.data_dir)
    if getattr(args, "adapters", None):
        cfg.adapters = args.adapters.split(",")
    return Engine(cfg)


def cmd_estimate(args) -> int:
    eng = _engine(args)
    est = eng.estimate_job(entrypoint=args.entrypoint, config=json.loads(args.config) if args.config else None,
                           model_spec=json.loads(args.spec) if args.spec else None, gpu_candidates=args.gpus,
                           total_steps=args.steps)
    print(f"estimate {est.estimate_id}  confidence={est.confidence}  method={est.method}")
    print(f"  model_class={est.job.model_class} params={est.job.params:,} dtype={est.job.dtype} "
          f"batch_tokens={est.job.batch_tokens} steps={est.job.total_steps}")
    print(f"  forward {est.flops_forward / 1e9:.2f} GFLOP/step, backward {est.flops_backward / 1e9:.2f}, "
          f"total {est.flops_total / 1e15:.3f} PFLOP")
    pm = est.peak_memory
    print(f"  peak memory {pm.point_gb:.1f} GB + {pm.margin_gb:.0f} GB reserve = {pm.required_gb:.1f} GB ({pm.method})")
    for g, p in est.per_gpu.items():
        fit = "fits" if p.fits_memory else "NO FIT"
        print(f"  {g:18s} {fit:6s} MFU {p.mfu_band.lo:.0%}-{p.mfu_band.hi:.0%} ({p.mfu_source}) "
              f"{p.hours.lo:.2f}-{p.hours.hi:.2f} h  ${p.usd_at_ref_price.lo:.2f}-${p.usd_at_ref_price.hi:.2f} [{p.bound}]")
    for w in est.warnings:
        print(f"  warning: {w}")
    if est.per_gpu:
        print("  suggested envelope:", json.dumps(Engine.suggest_envelope(est).model_dump()))
    if args.json:
        _print(est)
    return 0


def cmd_ledger(args) -> int:
    eng = _engine(args)
    if args.sub == "list":
        for p in eng.list_preregistrations(args.status):
            print(f"{p.preregistration_id}  {p.status:11s} {p.mode:8s} cap=${p.envelope.dollar_cap:<8.2f} "
                  f"hours<={p.envelope.hours_max:<6.2f} {p.envelope.gpu_classes} {p.job.entrypoint or '(spec)'}"
                  + (f"  supersedes {p.supersedes}" if p.supersedes else ""))
    elif args.sub == "show":
        _print(eng.get_preregistration(args.id))
    elif args.sub == "events":
        _print(eng.ledger.events(args.id))
    elif args.sub == "verify":
        bad = 0
        for pid, ok in eng.ledger.verify_all():
            print(f"{pid}  {'ok' if ok else 'SIGNATURE MISMATCH'}")
            bad += not ok
        print(f"ledger public key: {eng.ledger.signer.public_key_hex}")
        return 1 if bad else 0
    return 0


def cmd_amend(args) -> int:
    eng = _engine(args)
    changes = {}
    for s in args.set:
        k, _, v = s.partition("=")
        changes[k] = _parse_value(v)
    new = eng.amend(args.id, changes, actor="human", note=args.note)
    print(f"amended: {args.id} -> {new.preregistration_id}")
    _print(new.envelope)
    return 0


def cmd_jobs(args) -> int:
    eng = _engine(args)
    for j in eng.ledger.list_jobs():
        print(f"{j.job_id}  {j.status:12s} {j.preregistration_id}  {j.offer.provider}/{j.offer.gpu_type} "
              f"steps={j.telemetry.steps_completed} elapsed={j.elapsed_hours:.3f}h")
    return 0


def cmd_job(args) -> int:
    _print(_engine(args).job_status(args.id))
    return 0


def cmd_recon(args) -> int:
    r = _engine(args).get_reconciliation(args.id)
    print(r.narrative)
    if args.json:
        _print(r)
    return 0


def cmd_calib(args) -> int:
    _print(_engine(args).get_calibration(args.provider, args.gpu, args.model_class))
    return 0


def cmd_gpus(args) -> int:
    from .est.gpus import GPUS

    for g in GPUS.values():
        print(f"{g.name:20s} {g.gpu_class:10s} {g.vram_gb:5.0f} GB  {g.hbm_gbps:5.0f} GB/s  "
              f"bf16 {g.peak_tflops.get('bf16', g.peak_tflops.get('fp16', 0)):6.0f} TFLOPS  ${g.ref_price_per_hour.usd:.2f}/h (index)")
    return 0


def cmd_serve(args) -> int:
    from .server import build_server

    build_server(_engine(args)).run("stdio")
    return 0


async def _demo(args) -> int:
    """The whole loop locally: estimate -> preregister -> list offers -> provision (mock) -> run -> reconcile."""
    args.demo = True
    eng = _engine(args)
    eng.executor.profile_every, eng.executor.profile_steps = 10, 3
    entry = args.entrypoint
    print(f"== estimate  {entry}")
    est = eng.estimate_job(entrypoint=entry, gpu_candidates=["H100", "A100-80GB", "RTX 4090"], total_steps=args.steps)
    for g, p in est.per_gpu.items():
        print(f"   {g:18s} MFU {p.mfu_band.lo:.0%}-{p.mfu_band.hi:.0%} ({p.mfu_source})  {p.hours.lo:.3f}-{p.hours.hi:.3f} h")
    env = Engine.suggest_envelope(est, "H100-SXM-80GB")
    env.spot_ok = True
    print(f"== preregister  mode={args.mode}  cap=${env.dollar_cap} hours<={env.hours_max:.2f} classes={env.gpu_classes}")
    prereg = eng.preregister(est.estimate_id, env.model_dump(), args.mode)
    print(f"   {prereg.preregistration_id}  signature {prereg.signature[:24]}...")
    print("== broker")
    fits = await eng.list_offers(prereg.preregistration_id, "cheapest")
    for f in fits:
        tag = "FITS" if f.fits else "miss:" + ",".join(f.failures)
        print(f"   {f.offer.offer_id}  {f.offer.underlying_provider or f.offer.provider:8s} {f.offer.count}x{f.offer.gpu_type:16s} "
              f"{f.offer.tier:9s} ${f.offer.price_per_hour.usd:5.2f}/h  {f.offer.availability:5s}  {tag}")
    best = next(f for f in fits if f.fits)
    bound, summary = eng.provision_binding(prereg.preregistration_id, best.offer.offer_id)
    print(f"== confirm  {summary['gpu']} on {summary['provider']} ${summary['price_per_hour']['usd']}/h; "
          f"cost at hours_max ${summary['cost_at_hours_max']} <= cap ${summary['dollar_cap']}")
    conf = eng.confirmer.issue("provision", "money", bound, summary)
    eng.confirmer.redeem(conf.token, "provision", bound)
    print(f"   token {conf.token[:16]}... redeemed (single use)")
    print("== execute (local subprocess via mock adapter)")
    job = await eng.provision(prereg.preregistration_id, best.offer.offer_id)
    while eng.executor.is_active(job.job_id):
        await asyncio.sleep(1.0)
        j = eng.job_status(job.job_id)
        print(f"   {j.status:12s} steps={j.telemetry.steps_completed} accrued=${j.accrued.usd if j.accrued else 0:.4f}")
    job = eng.job_status(job.job_id)
    print(f"   final: {job.status} exit={job.exit_code} events={job.provider_events}")
    print("== reconcile")
    rec = eng.get_reconciliation(job.job_id)
    print("   " + rec.narrative)
    print(f"   calibration: {eng.get_calibration(gpu='H100-SXM-80GB')['measured']}")
    eng.close()
    return 0 if job.status.value == "completed" else 1


def cmd_demo(args) -> int:
    return asyncio.run(_demo(args))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="compute-prereg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", help="ledger/calibration/key directory (default ~/.compute-prereg)")
    ap.add_argument("--adapters", help="comma-separated adapter names (default: autodetect)")
    ap.add_argument("--demo", action="store_true", help="add the local mock adapter")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("estimate")
    p.add_argument("entrypoint", nargs="?")
    p.add_argument("--config")
    p.add_argument("--spec")
    p.add_argument("--gpus", nargs="*")
    p.add_argument("--steps", type=int)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_estimate)

    p = sub.add_parser("ledger")
    ps = p.add_subparsers(dest="sub", required=True)
    ps.add_parser("list").add_argument("--status")
    ps.add_parser("show").add_argument("id")
    ps.add_parser("events").add_argument("id")
    ps.add_parser("verify")
    p.set_defaults(fn=cmd_ledger)

    p = sub.add_parser("amend", help="human-only: amend an envelope (may raise the cap)")
    p.add_argument("id")
    p.add_argument("--set", action="append", default=[], help="key=value (JSON values ok)")
    p.add_argument("--note")
    p.set_defaults(fn=cmd_amend)

    sub.add_parser("jobs").set_defaults(fn=cmd_jobs)
    p = sub.add_parser("job")
    p.add_argument("id")
    p.set_defaults(fn=cmd_job)
    p = sub.add_parser("recon")
    p.add_argument("id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_recon)
    p = sub.add_parser("calib")
    p.add_argument("--provider")
    p.add_argument("--gpu")
    p.add_argument("--model-class")
    p.set_defaults(fn=cmd_calib)
    sub.add_parser("gpus").set_defaults(fn=cmd_gpus)
    sub.add_parser("serve").set_defaults(fn=cmd_serve)
    p = sub.add_parser("demo")
    p.add_argument("--entrypoint", default=str(Path(__file__).resolve().parent.parent / "examples" / "tiny_transformer.py"))
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--mode", default="ask")
    p.set_defaults(fn=cmd_demo)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except PreregError as e:
        print(redact(f"error: {type(e).__name__}: {e}"), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
