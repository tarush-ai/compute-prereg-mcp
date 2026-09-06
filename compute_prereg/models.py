"""Data model (CONTEXT.md §7). Estimate, Preregistration, Offer, Confirmation, Job,
Reconciliation and CalibrationEntry share one shape convention: every dollar figure is a
``Money`` carrying its source (quoted / index / derived) and a timestamp."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utcnow() -> datetime:
    return datetime.now(UTC)


class Confidence(StrEnum):
    exact = "exact"  # traced FLOPs, calibrated MFU with many samples
    calibrated = "calibrated"  # traced FLOPs, MFU from measurements
    prior = "prior"  # traced or analytic FLOPs, MFU from published priors
    low = "low"  # tracing failed / fell back; must return to the human


class Mode(StrEnum):
    ask = "ask"
    bounded = "bounded"
    locked = "locked"


class Tier(StrEnum):
    read = "read"
    write = "write"
    money = "money"


class PriceSource(StrEnum):
    quoted = "quoted"  # a provider quoted this number for this offer
    index = "index"  # a reference index (flopsindex / static table)
    derived = "derived"  # computed by us (hours x price)


class Money(BaseModel):
    model_config = ConfigDict(frozen=True)
    usd: float
    source: PriceSource
    asof: datetime = Field(default_factory=utcnow)
    note: str | None = None


class Band(BaseModel):
    """A [low, high] range. Always ordered."""

    model_config = ConfigDict(frozen=True)
    lo: float
    hi: float

    @model_validator(mode="after")
    def _order(self) -> Band:
        if self.lo > self.hi:
            object.__setattr__(self, "lo", self.hi)
            object.__setattr__(self, "hi", self.lo)
        return self

    @property
    def mid(self) -> float:
        return (self.lo + self.hi) / 2

    def as_list(self) -> list[float]:
        return [self.lo, self.hi]


# --------------------------------------------------------------------------- estimate


class GpuSpec(BaseModel):
    """Static roofline row."""

    name: str  # canonical, e.g. "H100-SXM-80GB"
    gpu_class: str  # e.g. "H100"
    vram_gb: float
    peak_tflops: dict[str, float]  # dtype -> dense TFLOPS
    hbm_gbps: float  # GB/s
    ref_price_per_hour: Money
    aliases: list[str] = Field(default_factory=list)
    interconnect: str | None = None

    def peak_flops(self, dtype: str) -> float:
        """Peak FLOP/s for a dtype, falling back along fp8->bf16->fp16->tf32->fp32."""
        order = [dtype, "bf16", "fp16", "tf32", "fp32"]
        for d in order:
            if d in self.peak_tflops:
                return self.peak_tflops[d] * 1e12
        raise KeyError(f"{self.name} has no peak for {dtype}")


class MemoryEstimate(BaseModel):
    point_gb: float
    margin_gb: float
    method: str
    breakdown_gb: dict[str, float] = Field(default_factory=dict)

    @property
    def required_gb(self) -> float:
        return self.point_gb + self.margin_gb


class PerGpuEstimate(BaseModel):
    gpu: str
    fits_memory: bool
    mfu_band: Band
    mfu_source: str  # "prior" | "calibrated(n=..)" | "measured(n=..)"
    step_seconds: Band
    hours: Band
    usd_at_ref_price: Band
    ref_price: Money
    bound: Literal["compute", "memory"]
    notes: list[str] = Field(default_factory=list)


class JobSpec(BaseModel):
    entrypoint: str | None = None
    code_hash: str | None = None
    config_hash: str | None = None
    framework: str = "torch"
    model_class: str = "unknown"
    params: int = 0
    dtype: str = "bf16"
    batch_tokens: int = 0
    total_steps: int = 0
    optimizer: str = "adamw"
    mixed_precision: bool = True
    extra: dict[str, Any] = Field(default_factory=dict)


class Estimate(BaseModel):
    estimate_id: str
    created_at: datetime = Field(default_factory=utcnow)
    job: JobSpec
    flops_forward: float  # per step
    flops_backward: float  # per step
    flops_total: float  # all steps
    param_bytes: float
    peak_memory: MemoryEstimate
    confidence: Confidence
    method: str
    per_module_flops: dict[str, float] = Field(default_factory=dict)
    per_gpu: dict[str, PerGpuEstimate] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    recommended_gpu: str | None = None

    @property
    def flops_per_step(self) -> float:
        return self.flops_forward + self.flops_backward


# --------------------------------------------------------------------------- envelope


class Envelope(BaseModel):
    gpu_classes: list[str]
    min_gpu_count: int = 1
    max_gpu_count: int = 1
    min_vram_gb: float
    hours_point: float
    hours_tolerance: float = 0.25
    dollar_cap: float
    spot_ok: bool = False
    max_preemptions: int = 0
    region_allow: list[str] = Field(default_factory=lambda: ["*"])
    provider_allow: list[str] = Field(default_factory=list)
    provider_deny: list[str] = Field(default_factory=list)
    checkpoint_interval_steps: int = 500
    interconnect: str | None = None

    @field_validator("dollar_cap")
    @classmethod
    def _cap_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("dollar_cap must be positive")
        return v

    @property
    def hours_max(self) -> float:
        return self.hours_point * (1.0 + self.hours_tolerance)


class PreregStatus(StrEnum):
    open = "open"
    provisioning = "provisioning"
    running = "running"
    completed = "completed"
    terminated = "terminated"
    breached = "breached"
    failed = "failed"
    superseded = "superseded"


class Preregistration(BaseModel):
    preregistration_id: str
    created_at: datetime = Field(default_factory=utcnow)
    supersedes: str | None = None
    signature: str | None = None
    record_hash: str | None = None
    job: JobSpec
    estimate: Estimate
    envelope: Envelope
    mode: Mode = Mode.ask
    status: PreregStatus = PreregStatus.open
    amended_by: str | None = None  # "human" | "agent" for amendments
    amendment_note: str | None = None

    def signable_body(self) -> dict[str, Any]:
        """The bytes that get hashed and signed: everything except signature/hash/status."""
        d = self.model_dump(mode="json")
        d.pop("signature", None)
        d.pop("record_hash", None)
        d.pop("status", None)
        return d


# --------------------------------------------------------------------------- broker


class Offer(BaseModel):
    offer_id: str
    provider: str  # adapter name, e.g. "runpod", "skypilot", "shadeform"
    underlying_provider: str | None = None  # for aggregators
    gpu_type: str  # canonical name from est.gpus
    count: int = 1
    vram_gb: float
    interconnect: str | None = None
    tier: Literal["on-demand", "spot", "reserved"] = "on-demand"
    price_per_hour: Money
    availability: Literal["now", "queue"] = "now"
    queue_eta_minutes: float | None = None
    region: str = "unknown"
    reliability_score: float = 0.5  # 0..1
    carbon_score: float = 0.5  # 0 (clean) .. 1 (dirty); used by lowest-carbon policy
    provider_ref: dict[str, Any] = Field(default_factory=dict)  # opaque, needed to provision
    fetched_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = None

    def binding_view(self) -> dict[str, Any]:
        """Fields that a confirmation token is bound to. Anything here changing re-quotes."""
        return {
            "offer_id": self.offer_id,
            "provider": self.provider,
            "underlying_provider": self.underlying_provider,
            "gpu_type": self.gpu_type,
            "count": self.count,
            "tier": self.tier,
            "price_per_hour_usd": self.price_per_hour.usd,
            "region": self.region,
        }


class OfferFit(BaseModel):
    offer: Offer
    fits: bool
    failures: list[str] = Field(default_factory=list)  # dimension names, e.g. "dollar_cap"
    details: dict[str, str] = Field(default_factory=dict)
    projected_hours: Band | None = None
    projected_usd: Band | None = None


Policy = Literal["cheapest", "fastest-to-start", "most-reliable", "lowest-carbon"]


# --------------------------------------------------------------------------- confirm


class Confirmation(BaseModel):
    token: str
    tool: str
    tier: Tier
    binding_hash: str
    issued_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime
    summary: dict[str, Any] = Field(default_factory=dict)
    consumed_at: datetime | None = None


# --------------------------------------------------------------------------- execute


class Instance(BaseModel):
    instance_id: str
    provider: str
    offer_id: str
    kind: Literal["ssh", "function", "local"] = "ssh"
    host: str | None = None
    port: int = 22
    user: str = "root"
    provider_ref: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class ProfilerSummary(BaseModel):
    """Fractions of wall time inside profiled windows, by cause."""

    windows: int = 0
    steps_profiled: int = 0
    frac_compute: float = 0.0  # matmul / conv / attention kernels
    frac_memory_bound: float = 0.0  # elementwise, norms, softmax, copies
    frac_data_loading: float = 0.0  # DataLoader __next__
    frac_comm: float = 0.0  # nccl / c10d
    frac_other: float = 0.0
    measured_flops_per_step: float | None = None  # from torch.profiler with_flops
    top_ops: list[dict[str, Any]] = Field(default_factory=list)


class TelemetrySummary(BaseModel):
    samples: int = 0
    steps_completed: int = 0
    mean_gpu_util: float | None = None
    peak_mem_gb: float | None = None
    mean_step_seconds: float | None = None
    p50_step_seconds: float | None = None
    first_step_at: datetime | None = None
    last_step_at: datetime | None = None
    last_sample_at: datetime | None = None
    gaps: list[list[datetime]] = Field(default_factory=list)  # [start, end] telemetry gaps
    num_workers: int | None = None
    profiler: ProfilerSummary | None = None
    last_loss: float | None = None


class JobStatus(StrEnum):
    pending = "pending"
    provisioning = "provisioning"
    syncing = "syncing"
    running = "running"
    checkpointing = "checkpointing"
    completed = "completed"
    failed = "failed"
    terminated = "terminated"
    breached = "breached"
    preempted = "preempted"


class Job(BaseModel):
    job_id: str
    preregistration_id: str
    offer: Offer
    instance: Instance | None = None
    status: JobStatus = JobStatus.pending
    mode: Mode = Mode.ask
    created_at: datetime = Field(default_factory=utcnow)
    provision_started_at: datetime | None = None
    started_at: datetime | None = None  # run command launched
    ended_at: datetime | None = None
    exit_code: int | None = None
    preemptions: int = 0
    telemetry: TelemetrySummary = Field(default_factory=TelemetrySummary)
    billed: Money | None = None  # from the provider usage API; None == unavailable
    accrued: Money | None = None  # derived: elapsed x price; never presented as billed
    breach_reason: str | None = None
    error: str | None = None
    ssh_public_key: str | None = None
    provider_events: list[str] = Field(default_factory=list)

    @property
    def elapsed_hours(self) -> float:
        start = self.started_at or self.provision_started_at
        if not start:
            return 0.0
        end = self.ended_at or utcnow()
        return max(0.0, (end - start).total_seconds() / 3600)


# --------------------------------------------------------------------------- reconcile


class Attribution(BaseModel):
    cause: str  # "data_loading" | "memory_bound_ops" | "launch_overhead" | "comm" | "preemption" | "unexplained"
    fraction: float  # of measured wall time
    detail: str


class Reconciliation(BaseModel):
    reconciliation_id: str
    job_id: str
    preregistration_id: str
    created_at: datetime = Field(default_factory=utcnow)
    gpu: str
    provider: str
    predicted_hours: Band
    predicted_usd: Band
    predicted_mfu: Band
    predicted_flops: float
    measured_hours: float
    measured_usd: Money | None  # None == billed: unavailable
    accrued_usd: Money | None
    measured_mfu: float | None
    measured_flops: float | None
    steps_completed: int
    steps_planned: int
    delta_hours_pct: float | None
    delta_usd_pct: float | None
    delta_mfu_pct: float | None
    inside_hours_band: bool | None
    inside_mfu_band: bool | None
    attribution: list[Attribution] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    narrative: str = ""
    calibration_written: bool = False
    telemetry_gaps: int = 0


# --------------------------------------------------------------------------- calibration


class CalibrationKey(BaseModel):
    model_config = ConfigDict(frozen=True)
    provider: str
    gpu_type: str
    model_class: str
    framework: str
    dtype: str
    batch_tokens_bucket: str


class CalibrationEntry(BaseModel):
    key: CalibrationKey
    n: int
    mean: float
    p10: float
    p90: float
    source: Literal["prior", "blended", "measured"]
    updated_at: datetime = Field(default_factory=utcnow)

    @property
    def band(self) -> Band:
        return Band(lo=self.p10, hi=self.p90)
