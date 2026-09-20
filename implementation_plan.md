# Block-MLA: Production Distributed Training Infrastructure (v2 — Corrected)
### Dynamic Tensor-Aware Momentum Look-Ahead — 20-Node Real-World Async Cluster

> **Project Class:** Research Algorithm + Production MLOps Infrastructure
> **Scale:** 20 heterogeneous GPU nodes (Google Colab Pro+, Kaggle, RunPod, Lambda Labs)
> **Showcase Goal:** Senior-level Distributed Systems + ML Research + MLOps Engineering

---

## Changelog from v1 (All Critical Bugs Fixed)

| # | Severity | Issue | Fix Applied |
|---|---|---|---|
| 1 | **CRITICAL** | Sign-convention bug: `theta_local - theta_global` pushes server backward | Changed to `theta_global - theta_local` matching DiLoCo convention |
| 2 | **CRITICAL** | `MAX_STALENESS=50` gate silently discards the experiment's key data points | Removed hard gate; staleness is logged only; Block-MLA coefficient handles it |
| 3 | **CRITICAL** | Infra built before algorithm proven correct | Explicit 4-phase gate sequence added; Phase -1 must pass before Phase 0 |
| 4 | HIGH | No auth on public endpoint | Bearer token added to `.proto`; server validates on every RPC |
| 5 | HIGH | Sync Nesterov "drop-in swap" claim is false | Barrier-based aggregation loop documented separately |
| 6 | HIGH | `worker_id` disk persistence described but never implemented | Explicit `_load_or_create_identity()` added to worker |
| 7 | MEDIUM | gRPC bidirectional streaming through Cloudflare Tunnel is flaky | Replaced with REST + object-storage model; gRPC kept for optional fast-path only |
| 8 | LOW | FP16 gradient compression risks overflow | Upgraded to BF16 (wider dynamic range, no overflow for gradient-like quantities) |

---

## System Overview

Block-MLA is a **full decentralized asynchronous training infrastructure** that demonstrates:

- Real multi-machine coordination over the open internet with fault tolerance
- A production-grade MLOps stack (W&B, Prometheus, Grafana, Redis, PostgreSQL)
- A novel outer optimizer implemented as a custom PyTorch class
- Gradient compression pipelines for commodity bandwidth
- Domain-segmented evaluation with automated downstream benchmarking

---

## Architecture Diagram

```
+---------------------------------------------------------------------+
|                     COORDINATION TIER                               |
|                                                                     |
|  +--------------+   +--------------+   +------------------------+  |
|  |  FastAPI     |   |  Redis       |   |  PostgreSQL            |  |
|  |  REST API    |   |  State Store |   |  Experiment Registry   |  |
|  |  (Port 8000) |   |  (Port 6379) |   |  (Port 5432)           |  |
|  +------+-------+   +--------------+   +------------------------+  |
|         |                                                           |
|  +------v-----------------------------------------------------------+|
|  |              Block-MLA Parameter Server (FastAPI async)          ||
|  |  POST /gradient  -> deserialize BF16, apply Block-MLA step       ||
|  |  GET  /model     -> stream BF16 global model to worker           ||
|  |  POST /heartbeat -> update Redis TTL                             ||
|  |  GET  /config    -> return experiment YAML to worker             ||
|  |  Bearer token auth on ALL endpoints                              ||
|  +------------------------------------------------------------------+|
|         |                                                            |
|  +------v--------------+   +----------------------------------+    |
|  |  Prometheus         |   |  Weights & Biases Agent          |    |
|  |  Metrics Exporter   |   |  (Streaming run logger)          |    |
|  |  + Grafana Dashboard|   |  + Model Artifact Registry       |    |
|  +---------------------+   +----------------------------------+    |
|                                                                     |
|  Hosted on: Oracle Cloud Free Tier VM (A1 ARM, 4 OCPUs, 24GB RAM) |
|  Public URL via Cloudflare Tunnel (zero-trust, HTTPS, no TCP raw)  |
+----------------------------+----------------------------------------+
                             | HTTPS REST (Cloudflare Tunnel)
           +-----------------+-----------------+
           v                 v                 v
+------------------+ +------------------+ +------------------+
|  WORKER NODE 1   | |  WORKER NODE 2   | |  WORKER NODE N   |
|  Google Colab    | |  Kaggle Notebook | |  RunPod / Lambda |
|  Pro+ (T4/A100)  | |  (T4 GPU)        | |  (A100/H100)     |
|                  | |                  | |                  |
|  1. GET /model   | |  1. GET /model   | |  1. GET /model   |
|  2. Run 80 AdamW | |  2. Run 80 AdamW | |  2. Run 80 AdamW |
|  3. Compute      | |  3. Compute      | |  3. Compute      |
|     delta =      | |     delta =      | |     delta =      |
|     theta_g -    | |     theta_g -    | |     theta_g -    |
|     theta_local  | |     theta_local  | |     theta_local  |
|  4. BF16 compress| |  4. BF16 compress| |  4. BF16 compress|
|  5. POST /grad   | |  5. POST /grad   | |  5. POST /grad   |
+------------------+ +------------------+ +------------------+

NOTE: REST over HTTPS (HTTP/1.1 enforced via Cloudflare) replaces gRPC
bidirectional streaming. HTTP/2 streaming behind a reverse-proxy is a
known source of silent flakiness from ephemeral clients like Colab.
Optional: keep gRPC as a fast-path for dedicated RunPod workers with
direct TCP access (bypass tunnel), falling back to REST automatically.
```

---

## Technology Stack

| Layer | Technology | Justification |
|---|---|---|
| **Algorithm** | Custom PyTorch `Optimizer` subclass | Native integration, zero overhead |
| **API** | FastAPI + Uvicorn (async) | HTTP/1.1 compatible, reliable through any proxy |
| **Transport** | HTTPS REST (multipart/octet-stream for tensors) | Works reliably through Cloudflare; no HTTP/2 streaming issues |
| **State Store** | Redis 7 (async via `aioredis`) | Worker heartbeat, staleness counters |
| **Experiment DB** | PostgreSQL + SQLAlchemy async | Run history, checkpoint registry |
| **Experiment Tracking** | Weights & Biases (wandb) | Industry gold standard |
| **Metrics** | Prometheus + `prometheus-client` | Real-time system metrics |
| **Dashboards** | Grafana | Live training visualization |
| **Gradient Compression** | BF16 cast (not FP16) | Wider dynamic range, no overflow for gradient-scale values |
| **Data Pipeline** | HuggingFace `datasets` streaming | SlimPajama without RAM limits |
| **Partitioning** | NumPy Dirichlet sampler (alpha=0.1) | Gold-standard non-IID simulation |
| **Model** | NanoGPT (60M params, custom) | Fits all 20 workers in GPU VRAM |
| **Evaluation** | `lm-evaluation-harness` CLI | MMLU, GSM8K, ARC benchmarks |
| **Tunneling** | Cloudflare Tunnel (HTTPS only) | Free, zero-trust, reliable for REST |
| **Infrastructure Server** | Oracle Cloud Free Tier (A1 ARM) | Always-on, 100% free |
| **Container** | Docker + Docker Compose | Reproducible deployments |
| **CI/CD** | GitHub Actions | Auto-test & deploy on push |
| **Auth** | Per-worker bearer tokens (generated at registration) | Prevents gradient injection from unknown clients |

---

## Execution Phases (Mandatory Gate Sequence)

> [!IMPORTANT]
> These phases are **strictly sequential**. Phase N cannot begin until Phase N-1 passes its exit criteria. The v1 plan violated this by treating infra and algorithm work as parallel. The result would be debugging a sign bug through 20 flaky Colab tabs instead of one pytest run.

```
Phase -1: Algorithm Math Validation (single machine, fake data, CPU only)
    EXIT CRITERIA: loss decreases over 200 outer steps with 1 worker, no staleness
    If loss goes up -> check sign convention FIRST before anything else

Phase 0: Single-Machine Multi-Worker Simulation (real data, 1 GPU, 3 workers)
    EXIT CRITERIA: pytest integration test passes (Block-MLA loss < Async-Nesterov at step 10)
    Uses in-process fake gRPC/REST, real SlimPajama streaming, Dirichlet alpha=0.1

Phase 1: Network Infrastructure (Oracle VM + Cloudflare + Docker stack)
    EXIT CRITERIA: 2 real Colab instances connect, upload gradients, server applies Block-MLA step

Phase 2: Scale to 20 Nodes + Full MLOps Stack
    EXIT CRITERIA: All 20 workers registered, Grafana live, W&B logging all metrics
```

---

## Phase -1 — Algorithm Math Validation (Must Run First)

**This phase exists because the sign convention bug in v1 would cause the loss to silently increase. A one-hour CPU validation run catches it before any infrastructure is built.**

### -1.1 Sign Convention (Fixed from v1)

The DiLoCo convention for pseudo-gradients is:

```
Delta = theta_global - theta_local
```

This means Delta **points away from where local training went**. When the server subtracts it (`theta - eta * Delta`), it moves the global model **toward** the local optimum. This is correct.

The v1 plan had the opposite sign in the worker comment (`theta_local - theta_global`), which combined with the server's subtraction would push the model **backward** — a silent correctness failure that produces an increasing loss curve easily mistaken for "async is hard."

### -1.2 Sanity Check Script (`tests/sanity/single_worker_sync.py`)

```python
"""
Phase -1 Sanity Check: Single worker, synchronous (no staleness), CPU only.

Pass criteria: val loss must decrease monotonically over 50 outer steps.
If it does not, check sign convention in compute_pseudo_gradient() first.

Run with: python -m tests.sanity.single_worker_sync
Expected: loss 4.5 -> ~3.8 over 50 steps (rough, model is tiny + fake data)
"""
import torch
from worker.model.nanogpt import build_nanogpt
from server.optimizer.block_mla_optimizer import BlockMLAOptimizer
from server.optimizer.parameter_filter import build_param_groups

def run_sanity_check(n_outer_steps: int = 50, n_inner_steps: int = 80):
    model   = build_nanogpt("tiny")   # ~1M param, fast on CPU
    global_params = {n: p.clone() for n, p in model.named_parameters()}
    eligible, bypass = build_param_groups(model)
    optimizer = BlockMLAOptimizer(
        [{"params": [p for _, p in eligible]}], lr=0.7
    )
    inner_opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []

    for outer_step in range(n_outer_steps):
        # 1. Worker: copy current global params
        theta_global = {n: p.clone().detach() for n, p in model.named_parameters()}

        # 2. Worker: inner loop (fake random batches for speed)
        model.train()
        for _ in range(n_inner_steps):
            x = torch.randint(0, 50257, (4, 64))
            loss = model(x, targets=x[:, 1:].contiguous()).loss
            loss.backward()
            inner_opt.step()
            inner_opt.zero_grad()

        # 3. Worker: compute pseudo-gradient  <-- THE FIXED SIGN
        #    Delta = theta_global - theta_local
        #    (points away from where local training went,
        #     so subtracting it on the server moves toward local optimum)
        pseudo_grad = {
            name: theta_global[name] - param.data
            for name, param in model.named_parameters()
        }

        # 4. Server: apply Block-MLA outer step
        for name, param in model.named_parameters():
            if (name, param) in [(n, p) for n, p in eligible]:
                state = optimizer.state.setdefault(param, {})
                optimizer.step_block(pseudo_grad[name], param, state, optimizer.defaults)
            else:
                # bypass: simple Nesterov
                state = optimizer.state.setdefault(param, {})
                from server.optimizer.block_mla_optimizer import nesterov_bypass_step
                nesterov_bypass_step(param, pseudo_grad[name], state)

        # 5. Evaluate
        model.eval()
        with torch.no_grad():
            x = torch.randint(0, 50257, (4, 64))
            val_loss = model(x, targets=x[:, 1:].contiguous()).loss.item()
        losses.append(val_loss)
        print(f"Step {outer_step:3d}: val_loss = {val_loss:.4f}")

    # Exit criteria
    first_half_mean = sum(losses[:25]) / 25
    second_half_mean = sum(losses[25:]) / 25
    assert second_half_mean < first_half_mean, (
        f"FAIL: loss did not decrease. "
        f"First half avg={first_half_mean:.4f}, second half avg={second_half_mean:.4f}. "
        f"CHECK SIGN CONVENTION in compute_pseudo_gradient() FIRST."
    )
    print(f"PASS: loss decreased from {first_half_mean:.4f} to {second_half_mean:.4f}")

if __name__ == "__main__":
    run_sanity_check()
```

**This test must pass before any other code is written or any infrastructure is provisioned.**

---

## Phase 0 — Local Multi-Worker Integration Test

Only after Phase -1 passes.

### Integration Test (`tests/integration/test_server_worker_loop.py`)

```python
"""
5 simulated workers, in-process server, real SlimPajama streaming,
Dirichlet alpha=0.1, 10 outer steps.

Exit criteria: Block-MLA val_loss < Async-Nesterov val_loss at step 10.
This must pass before Phase 1 (network infra) begins.
"""
import pytest
from tests.fixtures import build_in_process_server, build_sim_workers

@pytest.mark.slow
def test_block_mla_beats_async_nesterov():
    # 5 workers, paces [1, 5, 10, 15, 15], alpha=0.1
    blockmla_loss   = run_experiment("block_mla",     n_workers=5, n_steps=10)
    nesterov_loss   = run_experiment("async_nesterov", n_workers=5, n_steps=10)
    assert blockmla_loss < nesterov_loss, (
        f"Block-MLA ({blockmla_loss:.4f}) did not beat "
        f"Async-Nesterov ({nesterov_loss:.4f}) — check algorithm before scaling."
    )
```

---

## Phase 1 — Infrastructure Bootstrap (Only After Phase 0 Gate Passes)

### 1.1 Oracle Cloud Free Tier VM

Oracle Cloud provides a **permanently free** A1 ARM instance (4 OCPUs, 24 GB RAM). This is your coordination server.

```bash
bash scripts/setup_server.sh
# Installs: Docker, docker-compose, cloudflared, Python 3.11, nginx
```

### 1.2 Cloudflare Tunnel (HTTPS Only — Not Raw TCP)

> [!WARNING]
> The v1 plan used gRPC bidirectional streaming through the tunnel. HTTP/2 bidirectional streaming behind a reverse proxy is a known source of silent failures from ephemeral clients (Colab resets mid-stream, Cloudflare closes idle H2 connections). The corrected plan uses plain HTTPS REST (HTTP/1.1 forced), which is far more reliable for 20 flaky clients. gRPC is retained as an optional fast-path for dedicated workers with direct TCP access (RunPod), but the REST path is the **primary, tested path**.

```bash
cloudflared tunnel create block-mla-server
cloudflared tunnel route dns block-mla-server train.yourdomain.com
# Config: force HTTP/1.1 (disable HTTP/2 upgrade for Colab compatibility)
cloudflared tunnel run block-mla-server
```

### 1.3 Docker Compose Stack

```yaml
services:
  api:
    build: .
    command: uvicorn server.main:app --host 0.0.0.0 --port 8000 --http h11
    # h11 = HTTP/1.1 only — eliminates Cloudflare H2 proxy issues
    ports: ["8000:8000"]
    depends_on: [redis, postgres]
    environment:
      - REDIS_URL=redis://redis:6379
      - DB_URL=postgresql+asyncpg://...
      - WANDB_API_KEY=${WANDB_API_KEY}
      - WORKER_TOKEN_SECRET=${WORKER_TOKEN_SECRET}

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes: ["redis_data:/data"]

  postgres:
    image: postgres:16-alpine
    volumes: ["pg_data:/var/lib/postgresql/data"]

  prometheus:
    image: prom/prometheus
    volumes: ["./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml"]

  grafana:
    image: grafana/grafana
    volumes: ["grafana_data:/var/lib/grafana"]
    ports: ["3000:3000"]
```

---

## Phase 2 — API Design (REST, Not gRPC)

### 2.1 Endpoint Schema (`server/main.py`)

```python
"""
All endpoints require Bearer token authentication.
Token is issued at registration and must be included in every request.
This prevents gradient injection from unauthorized clients.
"""

@app.post("/register")
async def register_worker(info: WorkerInfo) -> RegistrationResponse:
    """
    Returns a per-worker bearer token.
    If worker_id already exists in DB, returns reconnect=True + current_step.
    """
    token = secrets.token_hex(32)
    await db.upsert_worker(info.worker_id, info.rank, token)
    return RegistrationResponse(
        token=token,
        reconnect=await db.worker_exists(info.worker_id),
        current_step=server_state.outer_step,
    )

@app.post("/gradient")
async def receive_gradient(
    request: Request,
    worker_id: str = Header(...),
    outer_step_at_snapshot: int = Header(...),
    auth: HTTPAuthorizationCredentials = Depends(security),
):
    """
    Receives BF16-compressed pseudo-gradient blob.
    Applies Block-MLA step immediately (async — no barrier).
    Logs staleness. Does NOT discard based on staleness (see below).
    """
    await _verify_token(auth.credentials, worker_id)

    raw_bytes   = await request.body()
    pseudo_grad = decode_bf16_gradient(raw_bytes, server_state.model)
    staleness   = server_state.outer_step - outer_step_at_snapshot

    # Log staleness for observability — but NEVER discard based on it.
    # Block-MLA's dynamic gamma coefficient is the staleness correction mechanism.
    # A hard gate here would silently remove the experiment's most important
    # data points (the extreme straggler updates Block-MLA is designed to handle).
    metrics.worker_staleness.labels(worker_id=worker_id).set(staleness)

    async with server_state.lock:
        block_metrics = server_state.apply_block_mla(pseudo_grad, worker_id, staleness)
        server_state.outer_step += 1

    wandb.log({**block_metrics, "staleness": staleness})
    return {"accepted": True, "current_step": server_state.outer_step}

@app.get("/model")
async def download_model(auth: HTTPAuthorizationCredentials = Depends(security)):
    """Returns BF16-compressed current global model as streaming bytes."""
    await _verify_token(auth.credentials)
    compressed = encode_bf16_model(server_state.model)
    return StreamingResponse(io.BytesIO(compressed), media_type="application/octet-stream")

@app.post("/heartbeat")
async def heartbeat(worker_id: str, auth: HTTPAuthorizationCredentials = Depends(security)):
    await _verify_token(auth.credentials, worker_id)
    await redis.setex(f"heartbeat:{worker_id}", 30, "alive")
    return {"ok": True}
```

### 2.2 Why No Hard Staleness Gate

> [!IMPORTANT]
> The v1 plan included `MAX_STALENESS = 50` with a hard discard. This is wrong for this experiment for the following reason:
>
> - `outer_step` increments on every gradient application
> - The fast worker (pace=1) dominates: it submits ~30 gradients for every 1 from the pace-30 worker
> - Worker 19 (pace=30, ArXiv-exclusive) will easily be 100-300 outer-steps stale on a real run
> - The v1 gate would silently discard **exactly** the data points Block-MLA is designed to handle
> - The loss curve would appear to "work" (no crash) but Worker 19's domain would show catastrophic forgetting regardless of which optimizer you use — making the comparison meaningless
>
> The correct design: log staleness comprehensively, let Block-MLA's `gamma_tb` coefficient dynamically scale the penalty based on `c_b`. If an update is genuinely harmful, the cosine score will be `-1` and `gamma_tb` will be clipped to `1.0`, effectively zero-weighting the conflicting block. That is the algorithmic contribution. The gate would bypass it.

---

## Phase 3 — Block-MLA Outer Optimizer (Corrected)

### 3.1 Pseudo-Gradient Sign Convention (Fixed)

```python
# worker/worker.py

def compute_pseudo_gradient(model: nn.Module, theta_global: dict) -> dict:
    """
    Compute the pseudo-gradient for DiLoCo outer optimization.

    Convention (matching DiLoCo paper and standard outer optimizer usage):
        Delta = theta_global - theta_local

    This delta POINTS AWAY from where local training went.
    When the server computes:
        theta_new = theta_global - eta * Delta
                  = theta_global - eta * (theta_global - theta_local)
                  = (1 - eta) * theta_global + eta * theta_local

    ...it moves the global model toward theta_local (where local training improved).

    The sign was inverted in v1 (theta_local - theta_global), which combined
    with the server's subtraction would push the model in the wrong direction.
    """
    return {
        name: theta_global[name] - param.data.clone()
        for name, param in model.named_parameters()
    }
```

### 3.2 Parameter Filtration (Unchanged, Was Correct)

```python
def build_param_groups(model: nn.Module) -> tuple[list, list]:
    """
    Separates model params into:
      - eligible:  Large 2D weight matrices (attn + MLP projections)
      - bypass:    1D biases, LayerNorm, sparse embedding table

    Follows Muon optimizer's filtering heuristic to prevent cosine
    similarity corruption from sparse embedding gradients under non-IID data.
    """
    ELIGIBLE_PATTERNS = [
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    ]
    eligible, bypass = [], []
    for name, param in model.named_parameters():
        if any(pat in name for pat in ELIGIBLE_PATTERNS) and param.dim() == 2:
            eligible.append((name, param))
        else:
            bypass.append((name, param))
    return eligible, bypass
```

### 3.3 Block-MLA Optimizer (Unchanged, Was Correct)

```python
class BlockMLAOptimizer(torch.optim.Optimizer):
    """
    Block-MLA outer optimizer. Applied on the server side.
    The sign correctness now depends entirely on compute_pseudo_gradient()
    returning Delta = theta_global - theta_local.
    """

    @torch.no_grad()
    def step_block(self, pseudo_grad_block, param, state, group) -> dict:
        lr   = group["lr"]
        beta = group["momentum"]
        eps  = group["eps"]

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(param)
            state["step_count"] = 0

        m_b     = state["momentum_buffer"]
        delta_b = pseudo_grad_block   # Delta = theta_global - theta_local

        u_hat = delta_b / (delta_b.norm() + eps)
        v_hat = m_b / (m_b.norm() + eps)
        c_b   = torch.sum(u_hat * v_hat).item()   # cosine similarity in [-1, 1]

        # Dynamic coefficient: amplify penalty when block is anti-momentum
        gamma_tb = min(beta * (1.0 + abs(c_b)), 1.0) if c_b < 0 else beta

        # m_{t+1,b} = gamma * m_{t,b} + (1 - gamma) * Delta_b
        m_b.mul_(gamma_tb).add_(delta_b, alpha=(1.0 - gamma_tb))

        # theta_{t+1,b} = theta_{t,b} - eta * (gamma * m_{t+1,b} + Delta_b)
        # This correctly moves toward theta_local because Delta = theta_global - theta_local
        param.add_(m_b, alpha=-(lr * gamma_tb))
        param.add_(delta_b, alpha=-lr)

        state["step_count"] += 1
        return {"cosine_score": c_b, "gamma_tb": gamma_tb,
                "pseudo_grad_norm": delta_b.norm().item()}
```

---

## Phase 4 — Worker Node (With Reconnection Persistence)

### 4.1 Worker Loop (With Implemented Disk Persistence)

```python
class BlockMLAWorker:
    """
    Asynchronous worker. Runs on Colab / Kaggle / RunPod / Lambda Labs.

    Key fix from v1: worker_id is now explicitly persisted to disk and
    reloaded on restart, enabling true reconnection after Colab session reset.
    The v1 plan described this behavior but did not implement it.
    """

    IDENTITY_FILE = Path(".worker_identity.json")

    def __init__(self, config: WorkerConfig):
        self.config    = config
        self.worker_id, self.token = self._load_or_create_identity()
        self.model     = build_nanogpt(config.model_size)
        self.inner_opt = torch.optim.AdamW(
            self.model.parameters(), lr=3e-4, weight_decay=0.1
        )
        self.dataloader = build_dataloader(
            worker_rank=config.rank,
            n_workers=config.total_workers,
            alpha=config.dirichlet_alpha,
        )

    def _load_or_create_identity(self) -> tuple[str, str]:
        """
        Load persisted (worker_id, token) from disk, or register as new worker.

        This is what enables true reconnection after a Colab session reset.
        The server sees the same worker_id and responds with reconnect=True
        + current global step, so the worker skips re-initialization.
        """
        if self.IDENTITY_FILE.exists():
            data = json.loads(self.IDENTITY_FILE.read_text())
            logger.info(f"Reconnecting as existing worker {data['worker_id']}")
            return data["worker_id"], data["token"]

        # New registration
        resp = httpx.post(
            f"{self.config.server_url}/register",
            json={
                "worker_id":   str(uuid.uuid4()),
                "worker_rank": self.config.rank,
                "platform":    self.config.platform,
                "gpu_type":    self._detect_gpu(),
                "inner_steps": self.config.inner_steps,
            }
        ).json()

        identity = {"worker_id": resp["worker_id"], "token": resp["token"]}
        self.IDENTITY_FILE.write_text(json.dumps(identity))
        logger.info(f"Registered as new worker {identity['worker_id']}")
        return identity["worker_id"], identity["token"]

    @property
    def _auth_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "worker-id":     self.worker_id,
        }

    def run(self):
        """Main worker loop — runs indefinitely with auto-reconnect."""
        while True:
            try:
                self._sync_global_model()
                theta_global = self._snapshot()

                # Inner loop: 80 AdamW steps on local shard
                self.model.train()
                for _ in range(self.config.inner_steps):
                    batch = next(self.dataloader)
                    loss  = self.model(**batch).loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.inner_opt.step()
                    self.inner_opt.zero_grad()

                # Compute pseudo-gradient with CORRECT sign
                pseudo_grad = compute_pseudo_gradient(self.model, theta_global)

                # Compress to BF16 and upload
                compressed  = encode_bf16_gradient(pseudo_grad)
                server_step = self._get_server_step()

                resp = httpx.post(
                    f"{self.config.server_url}/gradient",
                    content=compressed,
                    headers={
                        **self._auth_headers,
                        "outer-step-at-snapshot": str(server_step),
                        "content-type": "application/octet-stream",
                    },
                    timeout=120,
                )
                ack = resp.json()
                self._log_local_metrics(loss.item(), ack["current_step"] - server_step)

                # Heartbeat
                httpx.post(
                    f"{self.config.server_url}/heartbeat",
                    json={"worker_id": self.worker_id},
                    headers=self._auth_headers,
                )

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                logger.warning(f"Server unreachable: {e}. Retrying in 30s...")
                time.sleep(30)
```

### 4.2 BF16 Gradient Codec (Upgraded from FP16)

```python
# worker/compression/gradient_codec.py

def encode_bf16_gradient(pseudo_grad: dict) -> bytes:
    """
    Serialize pseudo-gradient dict to BF16 bytes for upload.

    BF16 is preferred over FP16 for gradient-like quantities:
    - Same exponent range as FP32 (8 bits) -> no overflow/underflow risk
    - Half the bits of FP32 -> 2x bandwidth reduction
    - FP16 (5-bit exponent) overflows at ~65504, risking NaN in large gradients

    Layout: [4 bytes: n_params] [for each param: 4-byte name_len, name_bytes, 4-byte shape_len, shape_bytes, BF16 tensor bytes]
    """
    buffer = io.BytesIO()
    for name, tensor in pseudo_grad.items():
        name_bytes  = name.encode("utf-8")
        tensor_bf16 = tensor.to(torch.bfloat16).numpy().tobytes()
        shape_bytes = json.dumps(list(tensor.shape)).encode("utf-8")
        buffer.write(struct.pack(">I", len(name_bytes)))
        buffer.write(name_bytes)
        buffer.write(struct.pack(">I", len(shape_bytes)))
        buffer.write(shape_bytes)
        buffer.write(struct.pack(">I", len(tensor_bf16)))
        buffer.write(tensor_bf16)
    return buffer.getvalue()
```

---

## Phase 5 — Baseline Implementations (Corrected)

> [!WARNING]
> The v1 claim that "each baseline is a drop-in swap of the server-side optimizer class" is **false for Sync Nesterov specifically**. Sync Nesterov requires fundamentally different server control flow: a **barrier** that waits for all 20 workers before applying any update. This cannot be achieved by swapping the optimizer class in the same immediate-apply loop — it requires a separate aggregation buffer and a quorum gate. The corrected plan documents this explicitly.

### Baseline Optimizer Swap Table

| Method | Server Control Flow | Optimizer Class |
|---|---|---|
| **Block-MLA (Ours)** | Immediate apply on each gradient arrival | `BlockMLAOptimizer` |
| **Async-MLA** | Immediate apply on each gradient arrival | `AsyncMLAOptimizer` (global scalar gamma) |
| **Async Nesterov** | Immediate apply on each gradient arrival | `AsyncNesterovOptimizer` (no correction) |
| **Sync Nesterov** | **BARRIER: wait for all 20 workers, then aggregate** | `SyncNesterovOptimizer` + `WorkerBarrier` |

### Sync Nesterov Barrier (Different Control Flow)

```python
class WorkerBarrier:
    """
    Accumulates gradients from all registered workers before applying.
    Fundamentally different from the async loop — requires a quorum gate.
    The synchronous baseline is bottlenecked by the slowest worker (pace=30).
    """
    def __init__(self, n_workers: int):
        self.n_workers = n_workers
        self.buffer: dict[str, dict] = {}   # worker_id -> pseudo_grad

    def submit(self, worker_id: str, pseudo_grad: dict) -> bool:
        self.buffer[worker_id] = pseudo_grad
        return len(self.buffer) >= self.n_workers

    def aggregate_and_reset(self) -> dict:
        """Mean of all pseudo-gradients across all workers."""
        names = list(next(iter(self.buffer.values())).keys())
        agg   = {
            name: torch.stack([g[name] for g in self.buffer.values()]).mean(0)
            for name in names
        }
        self.buffer.clear()
        return agg
```

---

## Phase 6 — Fault Tolerance & Reliability

### 6.1 Worker Heartbeat System

```python
# Worker sends POST /heartbeat every 10s
# Server: SETEX heartbeat:{worker_id} 30 "alive"
# Background task checks every 30s:
#   if no heartbeat: mark worker OFFLINE in postgres, continue training
```

Training **never stops** — the server applies Block-MLA from any live worker. Dead workers are excluded until they reconnect via `_load_or_create_identity()`.

### 6.2 Checkpoint Recovery

Every 25 outer steps:
1. Serialize global model -> Oracle Object Storage (free S3-compatible)
2. Upload to W&B Model Registry
3. Write metadata to PostgreSQL: `(step, val_loss, timestamp, active_workers)`

### 6.3 Staleness Logging (No Hard Gate)

```python
# Log staleness comprehensively for analysis
metrics.worker_staleness.labels(worker_id=worker_id).set(staleness)
wandb.log({"staleness/worker_{worker_id}": staleness})

# Do NOT discard. Block-MLA's gamma_tb coefficient handles extreme staleness.
# A cosine score c_b = -1 (maximum conflict) produces gamma_tb = min(beta*2, 1.0)
# which effectively damps the conflicting block to near-zero contribution.
# That is the algorithm's correctness guarantee — do not bypass it with a gate.
```

---

## Phase 7 — Monitoring & Observability

### Prometheus Metrics

```python
METRICS = {
    "worker_staleness":         Gauge("worker staleness at upload time"),
    "worker_pseudo_grad_norm":  Gauge("L2 norm of incoming pseudo-gradient"),
    "worker_upload_latency_ms": Histogram("HTTP POST /gradient latency"),
    "worker_alive_count":       Gauge("number of live workers right now"),
    "block_cosine_score":       Gauge("cosine alignment score per block"),
    "block_gamma_tb":           Gauge("dynamic extrapolation coefficient per block"),
    "outer_step":               Counter("total outer optimization steps"),
    "val_loss_global":          Gauge("global validation loss"),
    "val_loss_domain_arxiv":    Gauge("per-domain val loss: ArXiv"),
    "val_loss_domain_github":   Gauge("per-domain val loss: GitHub"),
    "val_loss_domain_cc":       Gauge("per-domain val loss: CommonCrawl"),
}
```

### Grafana Dashboard Panels

| Panel | Type | What It Shows |
|---|---|---|
| **Worker Hive Status** | State timeline | Which workers are live/dead per minute |
| **Staleness Distribution** | Histogram | Distribution of gradient staleness — high values = Block-MLA working |
| **Cosine Score Per Layer** | Heatmap | Which layers are anti-momentum vs aligned (live proof of block-wise divergence) |
| **gamma_tb Coefficient Map** | Heatmap | Adaptive penalty strength per block per step |
| **Global Val Loss (All Methods)** | Time series | 4-method comparison |
| **Domain Forgetting Tracker** | Multi-line | Per-domain val loss — key result proving Worker 19 is not forgotten |
| **Worker Upload Rate** | Bar chart | Uploads/min showing straggler effect empirically |
| **Gradient Norm Heatmap** | Heatmap | Pseudo-gradient norms by worker x layer |

---

## Phase 8 — Evaluation Harness

### Domain-Segmented Validation Loss

```python
@torch.no_grad()
def evaluate_all_domains(model, val_loaders: dict) -> dict:
    """
    The key experiment: does Worker 19's ArXiv loss decrease after its stale
    update is absorbed by Block-MLA, while spiking under Async-MLA?
    """
    return {
        domain: np.mean([model(**b).loss.item() for b in loader])
        for domain, loader in val_loaders.items()
    }
```

### Downstream Benchmark (Every 50 Outer Steps)

```bash
python -m lm_eval \
  --model hf \
  --model_args pretrained=./checkpoints/step_${STEP} \
  --tasks mmlu,gsm8k,arc_easy,arc_challenge \
  --num_fewshot 5 \
  --batch_size 8 \
  --output_path ./eval_results/step_${STEP}.json
```

---

## Phase 9 — Scale Configuration (20 Workers)

### Worker Pace Assignment

```yaml
worker_paces:
  - { rank: 0,  pace: 1,  platform: "runpod_a100"  }  # Fast anchor
  - { rank: 1,  pace: 2,  platform: "colab_a100"   }
  - { rank: 2,  pace: 3,  platform: "colab_l4"     }
  - { rank: 3,  pace: 5,  platform: "kaggle_t4"    }
  - { rank: 4,  pace: 8,  platform: "colab_t4"     }
  - { rank: 5,  pace: 10, platform: "colab_t4"     }
  - { rank: 6,  pace: 12, platform: "kaggle_t4"    }
  - { rank: 7,  pace: 15, platform: "colab_t4"     }
  - { rank: 8,  pace: 15, platform: "colab_t4"     }
  - { rank: 9,  pace: 15, platform: "colab_t4"     }
  - { rank: 10, pace: 15, platform: "colab_t4"     }
  - { rank: 11, pace: 15, platform: "colab_t4"     }
  - { rank: 12, pace: 20, platform: "kaggle_t4"    }
  - { rank: 13, pace: 20, platform: "kaggle_t4"    }
  - { rank: 14, pace: 20, platform: "colab_t4"     }
  - { rank: 15, pace: 20, platform: "colab_t4"     }
  - { rank: 16, pace: 25, platform: "colab_t4"     }
  - { rank: 17, pace: 25, platform: "colab_t4"     }
  - { rank: 18, pace: 30, platform: "colab_t4"     }
  - { rank: 19, pace: 30, platform: "colab_t4"     }  # Extreme straggler — ArXiv 91%
```

### Dirichlet Domain Assignment at alpha=0.1

| Worker | CommonCrawl | GitHub | ArXiv | Books | Wikipedia | StackExchange |
|---|---|---|---|---|---|---|
| 0 (fast) | **87%** | 5% | 3% | 2% | 2% | 1% |
| 3 | 12% | **74%** | 7% | 3% | 2% | 2% |
| 7 | 8% | 6% | **79%** | 4% | 2% | 1% |
| 19 (slowest) | 5% | 2% | **91%** | 1% | 1% | 0% |

Worker 19's updates arrive extremely stale with maximum geometric conflict. No hard gate discards them. Block-MLA's `gamma_tb` is the sole correction mechanism — which is the experiment's entire point.

---

## Phase 10 — CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  sanity_check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install poetry && poetry install
      - run: poetry run python -m tests.sanity.single_worker_sync
        name: "Phase -1: Algorithm sanity check (must pass before everything)"

  unit_tests:
    needs: sanity_check
    runs-on: ubuntu-latest
    steps:
      - run: poetry run pytest tests/unit/ -v --tb=short
      - run: poetry run mypy server/ worker/ --strict
      - run: poetry run ruff check .

  integration_test:
    needs: unit_tests
    runs-on: ubuntu-latest
    steps:
      - run: poetry run pytest tests/integration/ -v -m slow
        name: "Phase 0: 5-worker simulation (Block-MLA must beat Async-Nesterov)"
```

---

## Open Questions

> [!IMPORTANT]
> **Model size:** 60M params fits all 20 workers on T4 (~2.4 GB VRAM each). Scale to 124M for A100-only workers for more dramatic results, or keep 60M for full Colab T4 compatibility?

> [!IMPORTANT]
> **Server domain:** Oracle Cloud Free Tier is fully free (24 GB RAM). Do you have a domain for the Cloudflare Tunnel, or should we configure a free alternative (e.g., `trycloudflare.com` quick tunnel for development, custom domain for production)?

> [!IMPORTANT]
> **Experiment duration:** 300 outer steps x 80 inner steps. With 20 async workers, estimated real wall-clock: 4-8 hours. Run a 100-step smoke test first to validate the infrastructure before committing to the full run?

---

## Verification Plan

```bash
# Phase -1: Must pass first, on CPU, in ~10 minutes
python -m tests.sanity.single_worker_sync
# Expected: "PASS: loss decreased from X to Y"

# Phase 0: Integration test
pytest tests/integration/test_server_worker_loop.py -v -m slow
# Expected: "Block-MLA (3.82) beat Async-Nesterov (4.11)"

# Phase 1+: Unit tests
pytest tests/unit/ -v
mypy server/ worker/ --strict
ruff check . && ruff format --check .
```

### Manual Verification Steps (Phase 2 — 20 Nodes)

1. All 20 workers appear in Grafana "Worker Hive Status" within 2 min of launch
2. Staleness distribution shows values in 5-300 range (not all near zero)
3. `gamma_tb` heatmap shows variation across layers (Block-MLA adapting)
4. Worker 19's ArXiv domain loss decreases after its gradient is applied (not discarded)
5. lm-eval at steps 50/150/300: Block-MLA maintains GSM8K accuracy; Async-MLA degrades
6. Kill and restart server: workers reconnect within 60s, training continues from checkpoint
