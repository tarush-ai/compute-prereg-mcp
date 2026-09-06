"""Analytic FLOP / parameter estimates from a model spec (no torch needed).

Used when the caller passes a serialized spec instead of code, and as the fallback when
FakeTensor tracing fails. Standard dense-transformer accounting:
  forward  = 2 * params * tokens  +  attention term (4 * layers * tokens * seq * hidden)
  backward = 2 * forward
"""

from __future__ import annotations

from typing import Any

DTYPE_BYTES = {"fp32": 4, "float32": 4, "tf32": 4, "bf16": 2, "bfloat16": 2, "fp16": 2,
               "float16": 2, "fp8": 1, "int8": 1, "fp4": 0.5}


def dtype_bytes(dtype: str) -> float:
    return DTYPE_BYTES.get(dtype.lower(), 2)


def params_from_hf_config(cfg: dict[str, Any]) -> int:
    """Approximate parameter count from a HuggingFace-style config dict."""
    d = int(cfg.get("hidden_size") or cfg.get("n_embd") or cfg.get("d_model") or 0)
    L = int(cfg.get("num_hidden_layers") or cfg.get("n_layer") or cfg.get("num_layers") or 0)
    V = int(cfg.get("vocab_size") or 0)
    ffn = int(cfg.get("intermediate_size") or cfg.get("n_inner") or 4 * d)
    heads = int(cfg.get("num_attention_heads") or cfg.get("n_head") or 1)
    kv_heads = int(cfg.get("num_key_value_heads") or heads)
    gated = str(cfg.get("hidden_act", "")).lower() in {"silu", "swiglu", "geglu"} or bool(
        cfg.get("gated_mlp", False)
    )
    if not (d and L):
        raise ValueError("hf_config needs hidden_size and num_hidden_layers")
    head_dim = d // max(heads, 1)
    attn = d * d * 2 + 2 * d * (kv_heads * head_dim)  # q,o full; k,v possibly GQA
    mlp = (3 if gated else 2) * d * ffn
    norms = 2 * d * 2
    per_layer = attn + mlp + norms
    tied = bool(cfg.get("tie_word_embeddings", True))
    embed = V * d * (1 if tied else 2)
    return int(L * per_layer + embed + d)


def transformer_flops(
    params: int,
    tokens: int,
    seq_len: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
) -> tuple[float, float]:
    """(forward, backward) FLOPs for one step over ``tokens`` tokens."""
    fwd = 2.0 * params * tokens
    if seq_len and hidden_size and num_layers:
        fwd += 4.0 * num_layers * tokens * seq_len * hidden_size
    return fwd, 2.0 * fwd


def generic_flops(params: int, tokens: int) -> tuple[float, float]:
    fwd = 2.0 * params * tokens
    return fwd, 2.0 * fwd


def transformer_activation_bytes(
    tokens: int, seq_len: int, hidden_size: int, num_layers: int, heads: int, bytes_per: float,
    flash_attention: bool = True,
) -> float:
    """Megatron-style per-layer activation memory without recompute, summed over layers.
    With flash attention the s^2 attention-score term disappears."""
    if not (tokens and seq_len and hidden_size and num_layers):
        return 0.0
    b = tokens / seq_len
    per_layer = 34.0 * seq_len * b * hidden_size * (bytes_per / 2.0)
    if not flash_attention:
        per_layer += 5.0 * heads * seq_len * seq_len * b * (bytes_per / 2.0)
    return per_layer * num_layers


def optimizer_bytes_per_param(optimizer: str, mixed_precision: bool, dtype: str) -> float:
    """Optimizer state + master weights per parameter."""
    opt = optimizer.lower()
    if opt in {"adamw", "adam", "lamb", "lion"}:
        state = 8.0 if opt != "lion" else 4.0
    elif opt in {"sgd_momentum", "sgdm"}:
        state = 4.0
    elif opt in {"adafactor"}:
        state = 1.0
    elif opt in {"sgd"}:
        state = 0.0
    else:
        state = 8.0
    master = 4.0 if (mixed_precision and dtype_bytes(dtype) < 4) else 0.0
    return state + master
