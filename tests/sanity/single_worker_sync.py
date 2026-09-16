"""
tests/sanity/single_worker_sync.py

Phase -1: Algorithm Math Validation
====================================
Single worker, synchronous (no staleness), CPU only.

Exit criteria: validation loss must DECREASE over 50 outer steps.
If loss goes UP → the sign convention in compute_pseudo_gradient() is wrong.

Run with:
    python -m tests.sanity.single_worker_sync

Expected output:
    Step   0: val_loss = 4.xxxx
    Step  49: val_loss = ~3.8 (rough, model is tiny + random data)
    PASS: loss decreased from X.XXXX to Y.YYYY

This must PASS before any other code is written or infrastructure is provisioned.

Ref: implementation_plan.md § Phase -1, §-1.2
"""
from __future__ import annotations

import sys
import logging

import torch

from worker.model.nanogpt import build_nanogpt
from server.optimizer.block_mla_optimizer import BlockMLAOptimizer, nesterov_bypass_step
from server.optimizer.parameter_filter import build_param_groups

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def run_sanity_check(n_outer_steps: int = 50, n_inner_steps: int = 80) -> None:
    """
    Run the Phase -1 sanity check.

    Parameters
    ----------
    n_outer_steps : int
        Number of outer (server) optimization steps.
    n_inner_steps : int
        Number of inner (worker local) AdamW steps per outer step.
    """
    logger.info("=" * 60)
    logger.info("Phase -1: Algorithm Math Validation")
    logger.info("  Model:       tiny NanoGPT (~1M params)")
    logger.info("  Device:      CPU")
    logger.info(f"  Outer steps: {n_outer_steps}")
    logger.info(f"  Inner steps: {n_inner_steps}")
    logger.info("=" * 60)

    torch.manual_seed(42)  # reproducible

    # --- Build model and optimizer ---
    model = build_nanogpt("tiny")  # ~1M param, fast on CPU
    eligible, bypass = build_param_groups(model)

    logger.info(f"Eligible params (Block-MLA): {len(eligible)}")
    logger.info(f"Bypass params (Nesterov):    {len(bypass)}")

    # Server-side outer optimizer — only eligible params
    optimizer = BlockMLAOptimizer(
        [{"params": [p for _, p in eligible]}],
        lr=0.7,
        momentum=0.9,
    )

    # Worker-side inner optimizer — all params
    inner_opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    # Build fast lookup: name → param for eligible set
    eligible_names = {name for name, _ in eligible}

    # --- Optimizer state dict (server holds this, keyed by param name) ---
    server_states: dict[str, dict] = {}

    losses: list[float] = []

    for outer_step in range(n_outer_steps):
        # ----------------------------------------------------------------
        # Step 1 — Worker: snapshot current global params
        # ----------------------------------------------------------------
        theta_global: dict[str, torch.Tensor] = {
            name: param.data.clone().detach()
            for name, param in model.named_parameters()
        }

        # ----------------------------------------------------------------
        # Step 2 — Worker: inner loop (fake random batches for speed)
        # ----------------------------------------------------------------
        model.train()
        cfg = model.config
        for _ in range(n_inner_steps):
            x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
            targets = x  # next-token prediction (target = input shifted)
            out = model(x, targets=targets)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            inner_opt.step()
            inner_opt.zero_grad()

        # ----------------------------------------------------------------
        # Step 3 — Worker: compute pseudo-gradient (CORRECT SIGN — v2 fix)
        #
        #   Delta = theta_global - theta_local
        #
        # This points AWAY from where local training went.
        # When server subtracts:
        #   theta_new = theta_global - eta * Delta
        #             = theta_global - eta*(theta_global - theta_local)
        #             = (1-eta)*theta_global + eta*theta_local
        # ...it moves the global model TOWARD theta_local. Correct.
        #
        # v1 had theta_local - theta_global, which with the server's subtraction
        # would push the model in the WRONG direction → silent loss increase.
        # ----------------------------------------------------------------
        pseudo_grad: dict[str, torch.Tensor] = {
            name: theta_global[name] - param.data.clone()
            for name, param in model.named_parameters()
        }

        # ----------------------------------------------------------------
        # Step 4 — Server: apply Block-MLA outer step
        # ----------------------------------------------------------------
        with torch.no_grad():
            for name, param in model.named_parameters():
                grad = pseudo_grad[name]
                state = server_states.setdefault(name, {})

                if name in eligible_names:
                    # Block-MLA cosine-gated update
                    metrics = optimizer.step_block(
                        pseudo_grad_block=grad,
                        param=param,
                        state=state,
                        group=optimizer.defaults,
                    )
                    if outer_step % 10 == 0 and "attn.c_attn.weight" in name:
                        logger.debug(
                            f"  [{name}] cos={metrics['cosine_score']:.3f} "
                            f"gamma={metrics['gamma_tb']:.3f} "
                            f"grad_norm={metrics['pseudo_grad_norm']:.4f}"
                        )
                else:
                    # Nesterov bypass for biases / LN / embedding
                    nesterov_bypass_step(param, grad, state)

        # ----------------------------------------------------------------
        # Step 5 — Evaluate validation loss
        # ----------------------------------------------------------------
        model.eval()
        with torch.no_grad():
            x_val = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
            out_val = model(x_val, targets=x_val)
            val_loss = out_val.loss.item()

        losses.append(val_loss)
        if outer_step % 5 == 0 or outer_step == n_outer_steps - 1:
            logger.info(f"Step {outer_step:3d}: val_loss = {val_loss:.4f}")

    # ----------------------------------------------------------------
    # Exit criteria (implementation_plan.md §-1.2)
    # Second half average must be lower than first half average.
    # ----------------------------------------------------------------
    half = n_outer_steps // 2
    first_half_mean  = sum(losses[:half]) / half
    second_half_mean = sum(losses[half:]) / len(losses[half:])

    logger.info("")
    logger.info(f"First  half avg loss: {first_half_mean:.4f}")
    logger.info(f"Second half avg loss: {second_half_mean:.4f}")

    if second_half_mean < first_half_mean:
        logger.info(
            f"✓ PASS: loss decreased from {first_half_mean:.4f} to {second_half_mean:.4f}"
        )
        logger.info("Phase -1 PASSED. Proceed to Phase 0.")
    else:
        logger.error(
            f"✗ FAIL: loss did not decrease. "
            f"first_half={first_half_mean:.4f}, second_half={second_half_mean:.4f}. "
            f"CHECK SIGN CONVENTION in pseudo_grad computation FIRST."
        )
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase -1: Algorithm sanity check")
    parser.add_argument("--outer-steps", type=int, default=50)
    parser.add_argument("--inner-steps", type=int, default=80)
    args = parser.parse_args()

    run_sanity_check(
        n_outer_steps=args.outer_steps,
        n_inner_steps=args.inner_steps,
    )
