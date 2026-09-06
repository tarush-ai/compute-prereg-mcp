# Compute Preregistration Engine — Spec v0.1

**Working name:** Envelope (placeholder)
**Author:** Tarush Gupta · **Date:** 2026-09-06 · **Status:** draft for MVP scoping

---

## 1. One-line thesis

Derive a job's compute requirement from its code, lock it as a preregistered envelope *before* any provider is contacted, broker offers across providers that fit the envelope, confirm against the envelope (not against a provider's quote), execute with ephemeral credentials, and reconcile predicted vs. measured so every run makes the next estimate better.

The product is the loop, not any one step. The moat is the reconciliation data.

---

## 2. Problem

Today an ML engineer picks a GPU by folklore, guesses hours, provisions on whichever provider they already have a key for, and finds out the real cost when the invoice lands. The pieces to do this properly exist but are disconnected:

| Layer | What exists | What's missing |
|---|---|---|
| **Estimate** (code → FLOPs, bytes, memory, time) | `torch.utils.flop_counter.FlopCounterMode` + `FakeTensorMode` (exact op-level FLOPs, no GPU); `vram-check` (static memory); Vidur / Calculon / LLM-Viewer (research simulators) | No tool an agent can call. Nothing chains FLOPs → roofline → time → $. |
| **Commit** (lock a budget before spending) | GPU.ai's quote → confirm → execute with a parameter-bound single-use token | Single-provider. The commitment is to *a provider's quote*, not to *the job's requirement*. |
| **Broker** (find capacity across providers) | Shadeform (20–30 providers, unified API, cheapest-routing), SkyPilot (open-source, 24-provider catalog, failover, managed jobs), Prime Intellect, SF Compute, Vast.ai | Search is driven by a human-typed GPU type and count, not by an estimate. No envelope constrains the search. |
| **Price** (what does an hour cost) | flopsindex-mcp (cross-provider reference index), provider pricing APIs | Fine as is. Consume, don't rebuild. |
| **Execute** | Every provider's API/SDK; SkyPilot as a launcher | Fine as is. Compose. |
| **Reconcile** (predicted vs. measured) | `torch.profiler`, DCGM, provider usage APIs, Horace He's MFU gist | Nobody closes the loop. No system learns achieved MFU per (provider, GPU, model class). |

The gap is the **Estimate → Commit → Reconcile** spine. Broker, price and execute are integration work against existing systems.

---

## 3. Core concept: the preregistration loop

```
   ┌──────────┐    ┌──────────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌───────────┐
   │ ESTIMATE │ →  │ PREREGISTER  │ →  │ BROKER  │ →  │ CONFIRM │ →  │ EXECUTE │ →  │ RECONCILE │
   │ from code│    │ lock envelope│    │ hunt    │    │ vs env. │    │ ephem.  │    │ pred/meas │
   └──────────┘    └──────────────┘    └─────────┘    └─────────┘    └─────────┘    └─────┬─────┘
         ▲                                                                                │
         └────────────────────── calibration store (MFU priors) ◄─────────────────────────┘
```

**Estimate.** Run the user's model under `FakeTensorMode` + `FlopCounterMode`. Output: exact forward/backward FLOPs per module, parameter bytes, activation-memory estimate (with a stated safety margin — FakeTensor systematically underestimates peak memory because it can't see allocator reserve, intermediates, or optimizer state), and a roofline time range per candidate GPU using peak FLOPS/bandwidth and a calibrated MFU band.

**Preregister.** Freeze the estimate into an immutable, signed record: model hash, config hash, FLOPs, bytes, minimum VRAM, GPU class(es) acceptable, hours (with tolerance), dollar cap, interconnect needs, spot tolerance. This is the *contract*. It is written before any provider is queried, so hunting cannot drift it.

**Broker.** Query providers (directly and via aggregators), normalize offers to a common schema, filter to those that satisfy the envelope, rank by user policy (cheapest / fastest-to-start / most-reliable / lowest-carbon).

**Confirm.** Present offers. Confirmation is a parameter-bound single-use token, GPU.ai-style, but bound to *the envelope*, not to the provider's quote. An offer that would exceed the envelope in any dimension cannot be confirmed — it triggers a re-preregistration (a new, explicit contract), never a silent widen.

**Execute.** Generate a per-job SSH keypair. Register only the public key, only on the provisioned instance. Launch. Stream telemetry. Terminate on completion or on envelope breach (mode-dependent). Termination *is* key revocation — deleting a key on a provider does not revoke access on instances already provisioned with it.

**Reconcile.** Measured FLOPs (profiler), wall time, achieved MFU, billed dollars (provider usage API). Diff against the preregistration. Attribute the gap (launch overhead, data loading, memory-bound ops, comm, preemption). Write the achieved MFU back to the calibration store keyed by (provider, GPU, model class, framework, batch shape).

**Learn.** Next estimate for a similar job uses the calibrated MFU band, not a guess. Over time the system's error bars shrink per provider — that's the asset.

---

## 4. User stories

1. **"Run this training script for 3 epochs."** Agent estimates 2.1 PFLOP forward+backward, 38 GB peak memory, recommends 1×H100 80GB, 2.4–3.6 h at 40–60% MFU, $8–15. User approves envelope. Agent brokers: Lambda H100 $2.49/h available now; Shadeform→Verda $2.10/h, 12-min queue; RunPod spot $1.65/h. User picks. Confirmation token issued against the envelope. Job runs. Reconcile: 3.1 h, 51% MFU, $7.72. Calibration updated.

2. **Locked mode, unattended.** User sets `mode: locked`, envelope $20 / 4 h / ≤1×H100-class. Agent may pick any conforming offer without asking. If the job hits 4 h, it is checkpointed and terminated; the agent returns to the user with the reconcile report. It cannot extend.

3. **The estimate is wrong.** Model uses an op FakeTensor doesn't support. Estimator falls back to a 60-second real profiling run on the cheapest available GPU, extrapolates, and marks the envelope `confidence: low`. User sees the confidence before approving.

4. **The user wants to know why it cost more.** Reconcile report: "Predicted 2.9 h at 50% MFU; measured 4.4 h at 33% MFU. Attribution: 22% of step time in data loading (CPU-bound; 4 workers), 9% in unfused LayerNorm+residual. Recommendation: `num_workers=12`, `torch.compile`." That paragraph is the product.

---

## 5. Architecture

### 5.1 Components

**Estimator (`est/`)**
- Input: a Python entrypoint + config, or a serialized model spec (HF config, params, seq_len, batch, dtype, optimizer).
- FakeTensor tracing → `FlopCounterMode` → per-module FLOP table (forward; backward if a loss is provided).
- Memory: params + grads + optimizer state (exact) + activations (estimated, ± margin) + allocator reserve (fixed margin, default +4 GB, configurable).
- Roofline: per-GPU table of peak FLOPS by dtype, HBM bandwidth, VRAM. Time = max(FLOPs / (peak × MFU), bytes / bandwidth) per step × steps. MFU comes from the calibration store, falling back to a prior band (e.g. 35–55% for dense transformers on H100, lower for small batch).
- Output: `Estimate` record with confidence (`exact` / `calibrated` / `prior` / `low`).

**Ledger (`ledger/`)**
- Append-only store of `Preregistration` records (schema in §7). Each record is hashed and signed. Edits create a new record referencing the old one — history is never lost.

**Broker (`broker/`)**
- Provider adapters implementing one interface: `list_offers(envelope) -> [Offer]`, `provision(offer, ssh_pubkey) -> Instance`, `terminate(instance)`, `usage(instance) -> BilledCost`.
- Aggregator adapters first (Shadeform, SkyPilot-as-library) because they cover the long tail; direct adapters for providers with the best APIs or unique models (Modal serverless, RunPod, Lambda, Vast.ai).
- Normalizes to `Offer {provider, gpu_type, count, vram, interconnect, tier (on-demand/spot/reserved), price_per_hour, availability (now / queue ETA), region, reliability_score, underlying_provider (for aggregators)}`.
- Ranks by user policy. Never picks alone unless mode permits.

**Confirmer (`confirm/`)**
- Two-call handshake. Call 1: agent presents `(offer, envelope_id)` → server returns summary + single-use token bound to `hash(offer ∪ envelope)`. Call 2: agent returns token + identical params → execute. Any change re-quotes. Pre-authorization before a token exists unlocks nothing.
- Enforced in the tier wrapper, not in tool descriptions.

**Executor (`exec/`)**
- Per-job SSH keypair (ed25519). Public key registered to the provider for this instance only. Private key never leaves the user's machine / agent sandbox.
- Launch via provider adapter. Sync code + data (rsync over the ephemeral key, or the provider's object store).
- Telemetry: `nvidia-smi dmon` / DCGM at 1 Hz; `torch.profiler` traces on a sampled subset of steps; step timing; loss curve. Streamed to the reconciler.
- Watchdog: enforces envelope in `locked` mode (hours, dollars); checkpoint-then-terminate on breach.

**Reconciler (`recon/`)**
- Inputs: preregistration, telemetry, provider billing.
- Outputs: `Reconciliation` record — predicted vs measured for FLOPs, time, $, MFU; gap attribution; recommendations.
- Writes achieved MFU to the calibration store.

**Calibration store (`calib/`)**
- Key: (provider, gpu_type, model_class, framework, dtype, batch_tokens bucket). Value: MFU distribution (n, mean, p10, p90). Bootstrapped with published priors; replaced by measurements as runs accumulate.

### 5.2 MCP surface

Tools, tiered exactly like GPU.ai (reads free; writes confirm; money quote-and-confirm):

| Tier | Tool | Description |
|---|---|---|
| Read | `estimate_job(entrypoint \| model_spec, gpu_candidates?)` | Returns `Estimate` with per-GPU time/$ ranges and confidence. |
| Read | `list_offers(envelope_id, policy?)` | Brokered offers satisfying the envelope. |
| Read | `get_preregistration(id)`, `list_preregistrations()` | Ledger reads. |
| Read | `get_reconciliation(job_id)` | Predicted vs measured report. |
| Read | `get_calibration(provider?, gpu?, model_class?)` | MFU priors. |
| Read | `job_status(job_id)` | Live telemetry summary. |
| Write | `preregister(estimate_id, tolerances, dollar_cap, mode)` | Creates the envelope. One confirmation (consequence: "this locks the budget"). |
| Write | `amend_preregistration(id, changes)` | New record, references old. One confirmation. |
| Money | `provision(envelope_id, offer_id)` | Quote → token → execute. Bound to envelope. |
| Money | `terminate(job_id)` | Quote what's destroyed and what it currently costs → confirm. |

No tool raises a dollar cap. Raising happens outside the agent, by the human, in the ledger UI or CLI. (Copied deliberately from GPU.ai's "decreases only" rule.)

---

## 6. Configurability: the three modes

| Mode | Who picks the offer | On envelope breach | Use when |
|---|---|---|---|
| **Ask** (default) | Human, from ranked list | Stop, report, ask | Interactive work |
| **Bounded** | Agent, any offer inside the envelope, no re-confirm | Stop, report, ask | Agent loops, overnight sweeps |
| **Locked** | Agent, inside envelope | Checkpoint, terminate, report. Cannot extend. | Unattended, hard budget |

Tolerance knobs on the envelope: `hours_tolerance` (e.g. +25%), `dollar_cap` (hard), `gpu_classes` (allowed set), `spot_ok` (bool + max preemptions), `min_vram_gb`, `region_allow`, `provider_allow/deny`, `checkpoint_interval`.

A run in any mode that fails to provision after N offers, or whose estimate confidence is `low`, returns to the human regardless.

---

## 7. Data model

```json
{
  "preregistration_id": "prereg_01J...",
  "created_at": "2026-09-06T21:40:00Z",
  "supersedes": null,
  "signature": "ed25519:...",
  "job": {
    "entrypoint": "train.py",
    "code_hash": "sha256:...",
    "config_hash": "sha256:...",
    "framework": "torch-2.8",
    "model_class": "decoder-transformer",
    "params": 353_000_000,
    "dtype": "bf16",
    "batch_tokens": 1024,
    "total_steps": 20000
  },
  "estimate": {
    "estimate_id": "est_01J...",
    "flops_forward": 9.30e10,
    "flops_backward": 1.86e11,
    "flops_total": 5.58e15,
    "param_bytes": 7.06e8,
    "peak_memory_gb": { "point": 31.2, "margin_gb": 4.0, "method": "faketensor+margin" },
    "confidence": "calibrated",
    "per_gpu": {
      "H100-80GB": { "mfu_band": [0.42, 0.58], "hours": [2.4, 3.3], "usd_at_ref_price": [6.0, 8.3] },
      "A100-80GB": { "mfu_band": [0.45, 0.60], "hours": [7.1, 9.5], "usd_at_ref_price": [10.7, 14.3] }
    }
  },
  "envelope": {
    "gpu_classes": ["H100-80GB", "H200"],
    "min_gpu_count": 1, "max_gpu_count": 1,
    "min_vram_gb": 40,
    "hours_point": 3.0, "hours_tolerance": 0.25,
    "dollar_cap": 15.00,
    "spot_ok": true, "max_preemptions": 2,
    "region_allow": ["us-*"],
    "provider_deny": [],
    "checkpoint_interval_steps": 500
  },
  "mode": "bounded",
  "status": "open"
}
```

`Offer`, `Confirmation`, `Job`, `Reconciliation`, and `CalibrationEntry` follow the same shape. Every dollar figure carries its source (`quoted` / `index` / `derived`) and a timestamp.

---

## 8. Provider integration plan

Integrate aggregators first: two adapters cover most of the market. Then direct adapters for providers with distinct billing models or the best developer APIs. Hyperscalers last — expensive and API-heavy, but needed for enterprise users.

### Tier 1 — build first (MVP)

| Provider | Type | Why first | Notes |
|---|---|---|---|
| **SkyPilot** (as a library) | Open-source multi-cloud broker | One integration → 20+ clouds + Kubernetes, spot failover, managed jobs, live catalog | Not a provider; a launcher. Embed it as the default execute path. |
| **Shadeform** | Aggregator (20–30 providers) | Unified API, shows underlying provider, cheapest routing; where new neocloud inventory appears first | Marketplace margin; verify compliance info surfaced per provider |
| **Modal** | Serverless GPU, per-second | Distinct model (functions, not instances); ideal for short jobs, evals; excellent Python SDK; $30/mo free credit | No SSH/instances — executor needs a "function" backend; expensive at sustained utilization |
| **RunPod** | Neocloud + serverless | Pods (secure & community), spot pods, templates, GraphQL API; cheap; widely used | Community cloud reliability varies |
| **Lambda** | Neocloud | Simple REST API, SSH, popular, competitive H100/B200 pricing | On-demand only, capacity constraints |
| **Vast.ai** | Peer marketplace | Cheapest floor; interruptible bidding is a natural fit for `spot_ok`; DLPerf score per host | Reliability highly variable; treat hosts as individually scored |

### Tier 2 — v1

| Provider | Type | Notes |
|---|---|---|
| **GPU.ai** | Neocloud with MCP | Already has quote-confirm-execute; can be consumed *as* an MCP peer rather than an adapter |
| **Prime Intellect** | Marketplace aggregator | Multi-provider compute + decentralized training; overlaps Shadeform |
| **SF Compute** | H100 cluster market | Order-book, hour-granular contracts; best fit for `reserved`/burst clusters |
| **Nebius** | Neocloud | Strong pricing, managed K8s, EU presence |
| **Crusoe** | Neocloud | Clean-energy positioning → `lowest-carbon` policy |
| **CoreWeave** | Neocloud (enterprise) | Kubernetes-native; reserved-heavy; needed for large users |
| **Together AI** | Clusters + inference | Dedicated clusters; also an inference fallback |
| **Hyperbolic** | Marketplace + inference | Cheap on-demand; consumer-to-datacenter range |
| **FluidStack**, **Voltage Park** | Bare-metal clusters | Multi-node interconnect jobs |
| **Lightning AI** | Studios | Notebook-first users |
| **Paperspace/DigitalOcean**, **Cudo**, **Hyperstack**, **Verda** (ex-DataCrunch), **TensorDock**, **Massed Compute**, **Scaleway**, **Vultr**, **Latitude** | Neoclouds / marketplaces | Mostly reachable via Shadeform/SkyPilot; direct adapters only if a user asks |

### Tier 3 — v2

| Provider | Type | Notes |
|---|---|---|
| **AWS** (p5/p4d, EC2 Spot, Capacity Blocks) | Hyperscaler | Via SkyPilot first; direct adapter for Capacity Blocks |
| **GCP** (A3, spot, Dynamic Workload Scheduler) | Hyperscaler | Via SkyPilot; DWS calendar mode maps well to preregistration |
| **Azure** (ND-series) | Hyperscaler | Via SkyPilot |
| **OCI** | Hyperscaler | Via SkyPilot; competitive bare-metal pricing |
| **io.net**, **Akash**, **Spheron**, **Salad** | DePIN / consumer | Only with `spot_ok` and a high `max_preemptions`; reliability scoring mandatory |

**Price feed:** consume flopsindex (reference index across spot/on-demand/DePIN) as the "is this offer fair" baseline; use provider live prices for the actual quote.

---

## 9. Security

- **API keys:** per provider, environment block only, never CLI args, never echoed; redaction pass on every error string.
- **Dedicated agent keys:** read-only where possible; provisioning keys separate from the user's own CLI keys for revocability and attribution.
- **SSH:** ed25519 keypair generated per job; public key registered only for that instance; private key held only by the executor; instance termination is the revocation event. Never reuse a job key; never register a user's personal key.
- **Spend:** dollar cap enforced server-side in the money-tier wrapper, and as a second layer via the provider's own org spending limit where available. Agent cannot raise either.
- **No auto-approve:** document loudly that running the client with permission prompts disabled turns confirmation into a formality; recommend an org-level cap as the ceiling no client mode can talk past.
- **Data:** code and data go instance-to-instance over the ephemeral key or the provider's object store; the broker never proxies user data.

---

## 10. Failure handling

| Failure | Behavior |
|---|---|
| Estimator unsupported ops / dynamic control flow | Fall back to a 60 s real profiling run on the cheapest conforming GPU; extrapolate; mark `confidence: low`; always return to human |
| No offer fits envelope | Report nearest misses with the dimension that fails; suggest amended envelope; never auto-widen |
| Provision fails | Check for leftovers (a failed provision can leave a billing instance); terminate anything billable; report; never silently retry |
| Spot preemption | Resume from last checkpoint on next conforming offer, up to `max_preemptions`; then return to human |
| Envelope breach (hours or $) | `ask`/`bounded`: stop and ask. `locked`: checkpoint, terminate, report |
| Provider API error mid-job | Job continues; telemetry gap logged; reconcile marks affected windows |
| Reconcile can't read billing | Report with `billed: unavailable`; never fabricate a cost |

---

## 11. Milestones

**MVP (1–2 weekends).** Estimator (FakeTensor + FlopCounterMode + static roofline table for ~12 GPUs + prior MFU bands) · Ledger (SQLite, signed JSON) · MCP server with `estimate_job`, `preregister`, `list_offers`, `provision`, `terminate` · One aggregator adapter (SkyPilot-as-library) + Modal + RunPod · Reconcile from `torch.profiler` + wall clock + provider usage API · Calibration store with write-back. Validate on your own transformer: predicted 92.96 GFLOP/forward must match measured; MFU on one H100 run must land inside the prior band or the band gets corrected.

**v1.** Shadeform, Lambda, Vast.ai adapters · three modes · spot/resume · gap attribution (data loading, unfused ops, comm) with recommendations · CLI + minimal web ledger view · flopsindex as fairness baseline.

**v2.** Hyperscaler direct adapters · multi-node (interconnect-aware envelopes, sequence/tensor-parallel FLOP/comm modeling) · inference jobs (tokens/s envelopes, KV-cache memory modeling) · shared calibration (opt-in anonymized MFU data across users — the network effect).

---

## 12. Non-goals

- Owning GPUs or reselling capacity. Pure broker + ledger.
- Replacing the provider's dashboard for raising limits, deposits, or account management.
- Being a training framework. Executes whatever the user's script does.
- Exact time prediction. The promise is a *calibrated range with a stated confidence* and a *reconciliation that explains the gap*, not a point estimate.

---

## 13. Risks and open questions

1. **Estimation accuracy.** Analytic roofline is within ~2× cold. The value proposition depends on calibration converging fast. Mitigation: bootstrap priors from published MFU numbers per (GPU, model class); make the first N runs per user explicitly "calibration runs" in the UI.
2. **Provider API churn.** Adapters will break. Mitigation: aggregators absorb most churn; contract tests per adapter in CI against live sandboxes.
3. **Spot availability and preemption behavior** differ wildly by provider; the `spot_ok` envelope needs per-provider preemption priors from the calibration store.
4. **Automated provisioning and provider ToS.** Some providers restrict programmatic resale or account sharing. Broker only ever acts on the user's own accounts with the user's own keys.
5. **Aggregator margin vs. direct.** Shadeform/SkyPilot may not surface a provider's best direct price. Reconcile step should record `price_via_aggregator` vs `price_direct` where both are observable.
6. **Multi-node comm modeling** is a research problem (Calculon-class). Out of scope for v1; envelope for multi-node jobs carries `confidence: prior` until v2.
7. **Who pays for the profiling fallback run?** Default: the user, disclosed in the quote as its own line item.

---

## 14. Sources consulted

- GPU.ai MCP docs (confirmation contract, tiering, async provisioning, key hygiene): https://gpu.ai/docs/mcp
- flopsindex-mcp (public price indices, citation payload): https://pypi.org/project/flopsindex-mcp/
- PyTorch `flop_counter.py` (counting semantics; not a hardware measurement): https://github.com/pytorch/pytorch/blob/main/torch/utils/flop_counter.py
- FakeTensor memory-estimation accuracy study (systematic underestimate; ~1 s runtime; unsupported-op failures): https://arxiv.org/pdf/2602.17817
- vram-check (static VRAM estimation CLI): https://pypi.org/project/vram-check/
- SkyPilot GPU catalog (24 providers): https://gpus.skypilot.co/
- Shadeform provider coverage: https://www.clustermax.ai/cloudreview/shadeform , https://aimultiple.com/gpu-marketplace
- GPU cloud landscape / aggregator taxonomy: https://saturncloud.io/blog/gpu-clouds-aggregators-and-the-new-economics-of-ai-compute/
- Vidur (Microsoft, inference simulator): https://github.com/microsoft/vidur