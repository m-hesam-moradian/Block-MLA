# Block-MLA

> **Dynamic Tensor-Aware Momentum Look-Ahead — 20-Node Async Distributed Training**

[![Phase -1](https://img.shields.io/badge/Phase%20--1-✓%20PASSED%206.12→0.26-brightgreen)](tests/sanity/single_worker_sync.py)
[![Phase 0](https://img.shields.io/badge/Phase%200-in%20progress-yellow)](tests/integration/)
[![Phase 1](https://img.shields.io/badge/Phase%201-pending-lightgrey)](server/)
[![Phase 2](https://img.shields.io/badge/Phase%202-pending-lightgrey)](docker-compose.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A production-grade distributed ML training system demonstrating:
- A novel outer optimizer (`BlockMLAOptimizer`) with dynamic per-block cosine-gated momentum
- Real 20-node async coordination over the open internet (Colab + Kaggle + RunPod)
- Full MLOps stack: W&B, Prometheus, Grafana, Redis, PostgreSQL

---

## Execution Gate Sequence (Mandatory)

```
Phase -1 → Phase 0 → Phase 1 → Phase 2
```

**Phase N cannot begin until Phase N-1 passes its exit criteria.**

| Phase | Description | Exit Criteria |
|-------|-------------|---------------|
| **-1** | Algorithm math validation (CPU, single worker) | Loss decreases over 50 outer steps |
| **0** | Local 5-worker simulation | Block-MLA loss < Async-Nesterov loss at step 10 |
| **1** | Network infra (Oracle VM + Cloudflare) | 2 real Colab nodes connect & upload |
| **2** | Scale to 20 nodes + full MLOps | All 20 workers live, Grafana streaming |

---

## Quick Start

```bash
# 1. Install
pip install -e .

# 2. Run Phase -1 (MUST pass first)
python -m tests.sanity.single_worker_sync
# Expected: PASS: loss decreased from X.XXXX to Y.YYYY

# 3. Run Phase 0 integration test (only after Phase -1 passes)
pytest tests/integration/test_server_worker_loop.py -v -m slow
```

---

## Project Structure

```
block-mla/
├── server/
│   ├── optimizer/
│   │   ├── block_mla_optimizer.py   # Block-MLA outer optimizer
│   │   └── parameter_filter.py     # Eligible/bypass param split
│   └── main.py                     # FastAPI server (Phase 1+)
├── worker/
│   ├── model/
│   │   └── nanogpt.py              # NanoGPT 60M (tiny/small/medium)
│   ├── compression/
│   │   └── gradient_codec.py       # BF16 encode/decode
│   └── worker.py                   # Worker loop (Phase 1+)
├── tests/
│   ├── sanity/
│   │   └── single_worker_sync.py   # Phase -1 ← START HERE
│   └── integration/
│       └── test_server_worker_loop.py  # Phase 0
├── monitoring/                     # Prometheus + Grafana configs
├── .github/workflows/ci.yml        # CI/CD
├── docker-compose.yml              # Full server stack
└── pyproject.toml
```

---

## The Algorithm

Block-MLA is an **asynchronous outer optimizer** that corrects for stale gradients via:

1. **Per-block cosine similarity** between the incoming pseudo-gradient and the momentum buffer
2. **Dynamic `gamma_tb`**: If a block is anti-momentum (`c_b < 0`), the momentum decay coefficient is amplified to damp the conflicting update — instead of discarding it
3. **No hard staleness gate**: Worker 19 (pace=30, 91% ArXiv data) sends extremely stale gradients. The `gamma_tb` coefficient handles this, not a discard threshold

```
Delta = theta_global - theta_local    ← correct DiLoCo sign convention
gamma_tb = min(beta*(1+|c_b|), 1.0) if c_b < 0 else beta
m_{t+1} = gamma_tb * m_t + (1 - gamma_tb) * Delta
theta_{t+1} = theta_t - lr * (gamma_tb * m_{t+1} + Delta)
```

---

## Implementation Plan

See [`implementation_plan.md`](implementation_plan.md) — this is the **strict source of truth** for all architectural and algorithmic decisions.
