"""
server/optimizer/parameter_filter.py

Separates model parameters into eligible (large 2D weight matrices) and bypass
(biases, LayerNorm, embedding table) groups.

Following the Muon optimizer heuristic: cosine similarity is only meaningful on
large 2D projection matrices. Sparse embedding rows and 1D parameters produce
corrupted cosine scores under non-IID data, so they are bypassed to plain
Nesterov momentum.

Ref: implementation_plan.md § 3.2
"""
from __future__ import annotations

import torch.nn as nn

# Patterns that identify large 2D attention/MLP projection matrices.
ELIGIBLE_PATTERNS: tuple[str, ...] = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


def build_param_groups(
    model: nn.Module,
) -> tuple[list[tuple[str, nn.Parameter]], list[tuple[str, nn.Parameter]]]:
    """
    Split model parameters into two groups:

    Returns
    -------
    eligible : list of (name, param)
        Large 2D weight matrices. Block-MLA cosine-gated step is applied here.
    bypass : list of (name, param)
        1D biases, LayerNorm weights, embedding table.
        Simple Nesterov bypass step is applied here.
    """
    eligible: list[tuple[str, nn.Parameter]] = []
    bypass: list[tuple[str, nn.Parameter]] = []

    for name, param in model.named_parameters():
        if any(pat in name for pat in ELIGIBLE_PATTERNS) and param.dim() == 2:
            eligible.append((name, param))
        else:
            bypass.append((name, param))

    return eligible, bypass
