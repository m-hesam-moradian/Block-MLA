"""
worker/model/nanogpt.py

NanoGPT model — 60M parameters (Colab T4 compatible).

Custom implementation following Karpathy's nanoGPT design. The "tiny" variant
(~1M params) is used for Phase -1 CPU sanity check speed. The "small" variant
(~60M) is used for all real experiments.

Ref: implementation_plan.md § Technology Stack (Model row)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 256      # sequence length
    vocab_size:  int = 50257   # GPT-2 BPE vocab
    n_layer:     int = 6
    n_head:      int = 6
    n_embd:      int = 384
    dropout:     float = 0.0
    bias:        bool = False  # no bias for cleaner cosine similarity


PRESETS: dict[str, GPTConfig] = {
    # ~1M params — CPU Phase -1 sanity check (fast)
    "tiny": GPTConfig(block_size=64, n_layer=2, n_head=2, n_embd=64),
    # ~60M params — full experiment, fits T4 VRAM
    "small": GPTConfig(block_size=256, n_layer=6, n_head=6, n_embd=384),
    # ~124M params — A100-only (optional, see Open Questions in plan)
    "medium": GPTConfig(block_size=512, n_layer=12, n_head=12, n_embd=768),
}


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head  = config.n_head
        self.n_embd  = config.n_embd
        self.dropout = config.dropout

        # Key projection matrices — these are the ELIGIBLE params for Block-MLA
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd,     bias=config.bias)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        # Key projection matrices — ELIGIBLE params for Block-MLA
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.drop   = nn.Dropout(config.dropout)
        self.act    = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.c_proj(self.act(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# GPT
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight  # type: ignore[assignment]

        # Init weights
        self.apply(self._init_weights)
        # Scale residual projections (GPT-2 style)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | "GPTOutput":
        device  = idx.device
        B, T    = idx.size()
        assert T <= self.config.block_size, \
            f"Sequence length {T} > block_size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            return _GPTOutput(loss=loss, logits=logits)

        # inference only: last token
        logits = self.lm_head(x[:, [-1], :])
        return _GPTOutput(loss=None, logits=logits)


class _GPTOutput:
    """Simple output container matching the plan's model(x, targets=...).loss usage."""
    __slots__ = ("loss", "logits")

    def __init__(self, loss: torch.Tensor | None, logits: torch.Tensor) -> None:
        self.loss   = loss
        self.logits = logits


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_nanogpt(size: str = "small") -> GPT:
    """
    Build a NanoGPT model.

    Parameters
    ----------
    size : str
        "tiny"   — ~1M params, CPU-only, for Phase -1 sanity check
        "small"  — ~60M params, for all real experiments (T4 compatible)
        "medium" — ~124M params, for A100-only workers (optional)
    """
    if size not in PRESETS:
        raise ValueError(f"Unknown model size '{size}'. Choose from: {list(PRESETS)}")
    return GPT(PRESETS[size])
