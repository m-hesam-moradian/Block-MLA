"""
server/optimizer/block_mla_optimizer.py

Block-MLA Outer Optimizer — server-side.

Applied once per gradient upload from any worker (async, no barrier).
The sign correctness depends entirely on the worker computing:
    Delta = theta_global - theta_local      (implementation_plan.md §3.1)

Key algorithm:
  1. Compute cosine similarity c_b between current pseudo-grad block and
     momentum buffer block.
  2. If c_b < 0 (anti-momentum), amplify the momentum decay coefficient
     gamma_tb = min(beta * (1 + |c_b|), 1.0) to damp the conflicting update.
  3. Update momentum buffer and apply the look-ahead step.

This is the algorithmic contribution: staleness correction via dynamic
per-block gamma_tb, NOT via a hard staleness gate (which was removed in v2).

Ref: implementation_plan.md § 3.3
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Bypass helper (for 1D / embedding / LayerNorm params)
# ---------------------------------------------------------------------------

def nesterov_bypass_step(
    param: nn.Parameter,
    pseudo_grad: torch.Tensor,
    state: dict[str, Any],
    lr: float = 0.7,
    beta: float = 0.9,
) -> None:
    """
    Simple Nesterov momentum update for bypass parameters (no cosine gate).

    Used for biases, LayerNorm weights, and embedding tables where cosine
    similarity would be corrupted by sparsity under non-IID data.

    theta_{t+1} = theta_t - lr * (beta * m_t + pseudo_grad)
    """
    if "momentum_buffer" not in state:
        state["momentum_buffer"] = torch.zeros_like(param)

    m = state["momentum_buffer"]
    m.mul_(beta).add_(pseudo_grad, alpha=(1.0 - beta))
    # Nesterov: use look-ahead estimate
    param.add_(m, alpha=-(lr * beta))
    param.add_(pseudo_grad, alpha=-lr)


# ---------------------------------------------------------------------------
# Block-MLA Optimizer
# ---------------------------------------------------------------------------

class BlockMLAOptimizer(torch.optim.Optimizer):
    """
    Block-MLA outer optimizer, applied on the parameter server.

    The sign convention: Delta = theta_global - theta_local.
    When the server applies  theta -= lr * Delta  it moves theta toward
    theta_local (where local training improved things). See §3.1 for the
    full derivation of why this sign is correct and the v1 sign was wrong.

    Parameters
    ----------
    params : iterable
        Only eligible 2D projection weight tensors (from build_param_groups).
    lr : float
        Outer learning rate (default 0.7 as per plan §3.3).
    momentum : float
        Beta for momentum buffer (default 0.9).
    eps : float
        Numerical stability for cosine norm (default 1e-8).
    """

    def __init__(
        self,
        params,
        lr: float = 0.7,
        momentum: float = 0.9,
        eps: float = 1e-8,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step_block(
        self,
        pseudo_grad_block: torch.Tensor,
        param: nn.Parameter,
        state: dict[str, Any],
        group: dict[str, Any],
    ) -> dict[str, float]:
        """
        Apply one Block-MLA outer step to a single parameter block.

        Parameters
        ----------
        pseudo_grad_block : Tensor
            Delta = theta_global - theta_local for this block.
        param : Parameter
            The global model parameter to update in-place.
        state : dict
            Per-parameter optimizer state (momentum buffer, step count).
        group : dict
            Optimizer hyperparameters (lr, momentum, eps).

        Returns
        -------
        metrics : dict
            cosine_score, gamma_tb, pseudo_grad_norm — logged to W&B / Prometheus.
        """
        lr   = group["lr"]
        beta = group["momentum"]
        eps  = group["eps"]

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(param)
            state["step_count"] = 0

        m_b     = state["momentum_buffer"]
        delta_b = pseudo_grad_block  # Delta = theta_global - theta_local

        # --- Cosine similarity between current grad and momentum ---
        u_hat = delta_b / (delta_b.norm() + eps)
        v_hat = m_b     / (m_b.norm()     + eps)
        c_b   = torch.sum(u_hat * v_hat).item()   # scalar in [-1, 1]

        # --- Dynamic coefficient: amplify penalty when anti-momentum ---
        # If c_b < 0: update conflicts with momentum history → raise decay
        # If c_b >= 0: aligned → use standard beta
        gamma_tb = min(beta * (1.0 + abs(c_b)), 1.0) if c_b < 0 else beta

        # --- Momentum update ---
        # m_{t+1,b} = gamma * m_{t,b} + (1 - gamma) * Delta_b
        m_b.mul_(gamma_tb).add_(delta_b, alpha=(1.0 - gamma_tb))

        # --- Parameter update (look-ahead Nesterov-style) ---
        # theta_{t+1,b} = theta_{t,b} - lr * (gamma * m_{t+1,b} + Delta_b)
        # Moves theta toward theta_local because Delta = theta_global - theta_local
        param.add_(m_b,     alpha=-(lr * gamma_tb))
        param.add_(delta_b, alpha=-lr)

        state["step_count"] += 1

        return {
            "cosine_score":      c_b,
            "gamma_tb":          gamma_tb,
            "pseudo_grad_norm":  delta_b.norm().item(),
        }

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        """
        Standard optimizer.step() interface.
        Not used directly — call step_block() per-block on the server.
        Kept for API compatibility with torch.optim.Optimizer.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        return loss
