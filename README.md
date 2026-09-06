# compute-prereg-mcp

A compute preregistration engine for ML jobs, exposed as a tiered MCP server.

Derive a job's compute requirement from its code, lock it as a signed envelope *before* any
provider is contacted, broker offers that fit the envelope, confirm against the envelope
(not against a provider's quote), execute with a per-job SSH key, and reconcile predicted vs
measured so every run makes the next estimate better.

```
ESTIMATE -> PREREGISTER -> BROKER -> CONFIRM -> EXECUTE -> RECONCILE
   ^                                                          |
   +---------------- calibration store (MFU) <----------------+
```

The product is the loop. The asset is the reconciliation data. `CONTEXT.md` is the spec
this implements; this README is the operator's view.

## Status

MVP milestone (CONTEXT.md §11), implemented and tested locally:

| Piece | State |
|---|---|
| Estimator: FakeTensor + FlopCounterMode + MemTracker, analytic fallback, 15-GPU roofline table, prior MFU bands | done; validated on `examples/tiny_transformer.py` (traced FLOPs/step match the profiler-measured FLOPs/step within 0.03%) |
| Ledger: SQLite, ed25519-signed immutable records, append-only status events, amend-by-supersede | done |
| Calibration store with write-back and prior fallback | done |
| Broker: adapter interface, envelope fit with per-dimension failures, four ranking policies | done |
| Confirmer: two-call handshake, single-use tokens bound to hash(offer ∪ envelope) | done |
| Executor: per-job ed25519 key, provision, rsync/ssh launch, telemetry shim, watchdog, termination = key revocation | done; exercised end-to-end through the local mock adapter |
| Reconciler: predicted vs measured hours/$/MFU, gap attribution, recommendations, calibration write-back | done |
| MCP server with read / write / money tiers enforced in the wrapper | done |
| Adapters: `mock` (local) | done, tested |
| Adapters: `runpod`, `modal`, `skypilot` | written against the public APIs, **not yet exercised against live accounts** |
| MFU on a real H100 landing inside the prior band | **pending**: needs a GPU run |

What "not exercised" means: the adapter code paths compile, are importable, and follow each
provider's documented API, but no test in this repo talks to a provider. Contract tests
against sandbox accounts are the next step (CONTEXT.md §13.2).

## Install

Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # core + torch (dev) — tracing needs torch
uv sync --extra all     # + skypilot + modal
```

Optional extras: `torch` (FakeTensor tracing; without it only `model_spec` estimates work),
`skypilot`, `modal`.

## Try the whole loop locally

```bash
uv run compute-prereg --data-dir /tmp/prereg-demo demo --steps 200
```

This estimates the example transformer, preregisters an envelope, brokers offers from the
mock inventory (a Lambda H100 on demand, a Verda H100 with a queue, a RunPod H100 spot, an
A100, a 4090, an 8xH100 node), fits them against the envelope, issues and redeems a
confirmation token, runs the job in a local subprocess through the telemetry shim,
terminates, and prints the reconciliation paragraph. It runs on CPU, so the measured MFU is
meaningless against an H100 peak and is deliberately *not* written to calibration.

## Run the MCP server

```bash
uv run compute-prereg-mcp          # stdio
```

Claude Desktop / Claude Code config:

```json
{
  "mcpServers": {
    "compute-prereg": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/compute-prereg-mcp", "compute-prereg-mcp"],
      "env": {
        "COMPUTE_PREREG_DATA_DIR": "/Users/you/.compute-prereg",
        "RUNPOD_API_KEY": "...",
        "COMPUTE_PREREG_ADAPTERS": "runpod,skypilot"
      }
    }
  }
}
```

Environment:

| Variable | Meaning |
|---|---|
| `COMPUTE_PREREG_DATA_DIR` | ledger, calibration store, signing key, per-job SSH keys (default `~/.compute-prereg`) |
| `COMPUTE_PREREG_ADAPTERS` | comma list of adapters; default autodetects from credentials/packages |
| `COMPUTE_PREREG_DEMO=1` | adds the local `mock` adapter |
| `RUNPOD_API_KEY` | RunPod adapter |
| `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` (or `~/.modal.toml`) | Modal adapter |
| SkyPilot's own cloud credentials (`sky check`) | SkyPilot adapter |

Secrets are read from the environment only, never accepted as tool or CLI arguments, and a
redaction pass runs on every error string that leaves the server.

### Tools

| Tier | Tool | What it does |
|---|---|---|
| read | `estimate_job(entrypoint?, config?, model_spec?, gpu_candidates?, total_steps?)` | `Estimate` with per-GPU hours/$ bands, confidence, and a suggested envelope |
| read | `list_offers(envelope_id, policy?)` | brokered offers, each marked fits / which dimensions fail |
| read | `get_preregistration`, `list_preregistrations`, `get_reconciliation`, `get_calibration`, `job_status`, `list_gpus` | ledger and telemetry reads |
| write | `preregister(estimate_id, envelope, mode)` | locks the envelope (two-call confirm) |
| write | `amend_preregistration(id, changes)` | new record superseding the old; agents may only tighten |
| money | `provision(envelope_id, offer_id?)` | quote -> token bound to hash(offer ∪ envelope) -> execute |
| money | `terminate(job_id)` | quotes what is destroyed and what it has cost, then checkpoint + terminate |

Write and money tools return `{"status": "confirmation_required", "confirm_token": ...,
"summary": ..., "consequence": ...}` on the first call. Show the summary to the human, then
call again with identical parameters plus `confirm_token`. Tokens are single-use, expire
after 10 minutes, and are void if any bound parameter changes (a re-priced offer, an amended
envelope). A forged or pre-existing token unlocks nothing.

**No tool raises a dollar cap or widens hours.** That is a human action:

```bash
compute-prereg amend prereg_01J... --set dollar_cap=40 --set hours_tolerance=0.5 --note "approved"
```

### Loudly: do not disable permission prompts

If you run an MCP client with tool approvals turned off, the confirmation handshake becomes
a formality: the agent will quote and confirm to itself. Set an organisation-level spend
limit at each provider as the ceiling no client mode can talk past. The dollar cap in the
envelope is enforced server-side (at quote, at confirm, and by the watchdog while running),
but it is only as strong as the human on the other end of the confirm.

## Modes

| Mode | Who picks the offer | On envelope breach |
|---|---|---|
| `ask` (default) | human, from the ranked list (`offer_id` required) | stop, report, ask |
| `bounded` | agent, best conforming cached offer (`offer_id` optional) | stop, report, ask |
| `locked` | agent, inside envelope | checkpoint, terminate, report; cannot extend |

An estimate with `confidence: low` disables auto-selection in every mode.

## Entrypoint contract

The estimator traces your model without running it. Your training script exposes:

```python
def prereg_build(config: dict) -> dict:
    model = build_model(config)
    x = torch.randint(0, vocab, (batch, seq_len))
    return {
        "model": model,
        "inputs": x,                                  # tensor | tuple | dict
        "loss": lambda out: F.cross_entropy(out.view(-1, vocab), x.view(-1)),  # optional -> backward FLOPs
        "model_class": "decoder-transformer",         # optional metadata
        "total_steps": 20000, "batch_tokens": batch * seq_len,
        "seq_len": seq_len, "hidden_size": d, "num_layers": L, "heads": H,
        "optimizer": "adamw", "dtype": "bf16",
    }
```

The module is imported with `__name__ != "__main__"`, so the usual `if __name__ ==
"__main__": main()` guard keeps training from starting. When tracing hits an unsupported op
the estimator falls back to `model_spec` (if given) and marks the estimate `low`.

`model_spec` alone also works: `{"params": ..., "batch_tokens": ..., "model_class": ...}` or
`{"hf_config": {...}, "seq_len": 1024, "batch_size": 8, "total_steps": 20000}`.

### What runs on the instance

The executor rsyncs your entrypoint's directory and this package to the instance and starts

```
python -m compute_prereg.shim --telemetry telemetry.jsonl -- train.py
```

The shim registers a global optimizer post-hook (step timing, CUDA peak memory), samples
`torch.profiler` windows (`with_flops=True`) every N steps and classifies time into
compute / memory-bound / data-loading / comm, and honours checkpoint requests: if
`compute_prereg.shim.checkpoint_requested()` is true, save and exit. Scripts that ignore it
are terminated after the grace period. Set `PREREG_TOTAL_STEPS` from the environment if you
want the preregistered step count to drive the run.

## CLI

```
compute-prereg estimate train.py [--steps N] [--gpus H100 A100-80GB] [--spec JSON] [--json]
compute-prereg ledger list | show ID | events ID | verify
compute-prereg amend ID --set key=value ...      # human-only; may raise the cap
compute-prereg jobs | job JOB_ID | recon JOB_ID
compute-prereg calib [--gpu G] [--provider P] | gpus
compute-prereg demo [--steps N] [--mode ask|bounded|locked]
compute-prereg serve
```

## Layout

```
compute_prereg/
  models.py       data model (§7): Estimate, Preregistration, Envelope, Offer, Confirmation, Job, Reconciliation, CalibrationEntry
  est/            gpus.py (roofline table), priors.py, analytic.py, tracer.py (FakeTensor), memory.py, roofline.py, estimator.py
  ledger/         signing.py (ed25519 over canonical JSON), store.py (append-only SQLite)
  calib/          calibration store with prior fallback and write-back
  broker/         base.py (adapter interface), fit.py, ranking.py, broker.py, adapters/{mock,runpod,modal,skypilot}.py
  confirm/        two-call token handshake
  executor/       keys.py (per-job ed25519), launchers.py (local/ssh/modal), telemetry.py, runner.py (watchdog)
  recon/          reconciler (attribution, recommendations, calibration write-back)
  shim.py         in-process telemetry for the user's script
  engine.py       wiring; server.py (MCP, tiered); cli.py (human actions)
examples/tiny_transformer.py   a small decoder used by the tests and the demo
tests/                         40 tests: estimator, ledger, calibration, broker, confirmer, executor, reconciler, MCP end-to-end
```

## Failure handling (§10)

| Failure | Behaviour |
|---|---|
| tracing unsupported | analytic fallback from `model_spec`, `confidence: low`; auto-selection disabled |
| no offer fits | nearest misses returned with the failing dimensions; never auto-widened |
| provision fails | leftovers looked up and terminated; job `failed`; no silent retry |
| code changed after preregistration | refused before provisioning (hash mismatch) |
| envelope breach (hours or $) | checkpoint request, stop, terminate; status `breached`; reconcile |
| telemetry read fails mid-job | job continues; gap logged; reconcile marks it |
| billing unavailable | `billed: unavailable`, accrued cost shown as `derived`; never fabricated |
| spot preemption | detected and reported; resume-from-checkpoint is v1 |

## Tests

```bash
uv run pytest -q
```

## Non-goals

No GPU ownership or resale, no provider account management, no training framework, no
point predictions: the promise is a calibrated range with a stated confidence and a
reconciliation that explains the gap.
