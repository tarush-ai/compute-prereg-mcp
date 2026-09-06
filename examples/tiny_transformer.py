"""A small decoder-only transformer used to validate the estimator and to exercise the
execute -> reconcile path locally.

Two entry points:
  prereg_build(config)  -> used by the estimator (FakeTensor tracing; nothing runs)
  main()                -> a real (tiny) training loop; what the executor launches
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyDecoder(nn.Module):
    def __init__(self, vocab: int, d: int, layers: int, heads: int, seq_len: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq_len, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.seq_len = seq_len

    def forward(self, idx):
        b, t = idx.shape
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=idx.device), 1)
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln(x))


DEFAULTS = dict(vocab=256, d=128, layers=2, heads=4, seq_len=64, batch_size=8, total_steps=30)


def _cfg(config: dict) -> dict:
    c = dict(DEFAULTS)
    c.update({k: v for k, v in (config or {}).items() if k in c})
    return c


def prereg_build(config: dict) -> dict:
    c = _cfg(config)
    model = TinyDecoder(c["vocab"], c["d"], c["layers"], c["heads"], c["seq_len"])
    idx = torch.randint(0, c["vocab"], (c["batch_size"], c["seq_len"]))

    def loss(out):
        return F.cross_entropy(out.view(-1, c["vocab"]), idx.view(-1))

    return {
        "model": model,
        "inputs": idx,
        "loss": loss,
        "model_class": "decoder-transformer",
        "total_steps": c["total_steps"],
        "batch_tokens": c["batch_size"] * c["seq_len"],
        "seq_len": c["seq_len"],
        "hidden_size": c["d"],
        "num_layers": c["layers"],
        "heads": c["heads"],
        "optimizer": "adamw",
        "dtype": "fp32",
        "mixed_precision": False,
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    c = _cfg({k: int(v) for k, v in (a.split("=") for a in argv if "=" in a)})
    steps = int(os.environ.get("PREREG_TOTAL_STEPS") or os.environ.get("TOTAL_STEPS") or c["total_steps"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = TinyDecoder(c["vocab"], c["d"], c["layers"], c["heads"], c["seq_len"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, c["vocab"], (c["batch_size"], c["seq_len"]), device=device)
        out = model(idx)
        loss = F.cross_entropy(out.view(-1, c["vocab"]), idx.view(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0:
            print(f"step {step} loss {loss.item():.3f}", flush=True)
    print(f"done {steps} steps in {time.time() - t0:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
