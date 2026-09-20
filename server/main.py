import os
import io
import time
import secrets
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Header, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import torch
import redis.asyncio as redis
from prometheus_client import Gauge, Histogram, Counter, make_asgi_app
import wandb

from server.optimizer.block_mla_optimizer import BlockMLAOptimizer
from server.optimizer.parameter_filter import build_param_groups
from worker.compression.gradient_codec import decode_bf16_gradient, encode_bf16_model
from worker.model.nanogpt import build_nanogpt

# --- Config & State ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost")
WORKER_TOKEN_SECRET = os.getenv("WORKER_TOKEN_SECRET", "supersecret")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")

security = HTTPBearer()

class ServerState:
    def __init__(self):
        self.outer_step = 0
        self.lock = asyncio.Lock()
        self.model = build_nanogpt("tiny")
        self.eligible, self.bypass = build_param_groups(self.model)
        self.optimizer = BlockMLAOptimizer(
            [{"params": [p for _, p in self.eligible]}], lr=0.7
        )
        self.opt_state = {}
        self.eligible_names = {n for n, _ in self.eligible}
        
    def apply_block_mla(self, pseudo_grad: dict, worker_id: str, staleness: int):
        metrics_out = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name not in pseudo_grad:
                    continue
                grad = pseudo_grad[name]
                state = self.opt_state.setdefault(name, {})
                
                if name in self.eligible_names:
                    metrics = self.optimizer.step_block(grad, param, state, self.optimizer.defaults)
                    if "attn.c_proj" in name:  # log a representative layer
                        metrics_out = metrics
                else:
                    from server.optimizer.block_mla_optimizer import nesterov_bypass_step
                    nesterov_bypass_step(param, grad, state)
        return metrics_out

server_state = ServerState()
redis_client = None

# --- Metrics ---
metrics = {
    "worker_staleness": Gauge("worker_staleness", "Staleness at upload time", ["worker_id"]),
    "outer_step": Counter("outer_step", "Total outer optimization steps"),
}

# --- Mock DB for worker registration ---
# In a full deployment, this would be an asyncpg/SQLAlchemy connection
worker_db = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
        wandb.init(project="block-mla-cluster")
    yield
    await redis_client.aclose()

app = FastAPI(lifespan=lifespan)

# Add prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

async def _verify_token(token: str, worker_id: str = None):
    # In production, this validates against the DB
    if worker_id and worker_id in worker_db:
        if worker_db[worker_id] != token:
            raise HTTPException(status_code=401, detail="Invalid token")
    return True

class WorkerInfo(BaseModel):
    worker_id: str
    worker_rank: int
    platform: str
    gpu_type: str
    inner_steps: int

class RegistrationResponse(BaseModel):
    token: str
    reconnect: bool
    current_step: int

@app.post("/register", response_model=RegistrationResponse)
async def register_worker(info: WorkerInfo):
    if info.worker_id in worker_db:
        return RegistrationResponse(
            token=worker_db[info.worker_id],
            reconnect=True,
            current_step=server_state.outer_step,
        )
    token = secrets.token_hex(32)
    worker_db[info.worker_id] = token
    return RegistrationResponse(
        token=token,
        reconnect=False,
        current_step=server_state.outer_step,
    )

@app.post("/gradient")
async def receive_gradient(
    request: Request,
    worker_id: str = Header(...),
    outer_step_at_snapshot: int = Header(...),
    auth: HTTPAuthorizationCredentials = Depends(security),
):
    await _verify_token(auth.credentials, worker_id)
    raw_bytes = await request.body()
    pseudo_grad = decode_bf16_gradient(raw_bytes)
    staleness = server_state.outer_step - outer_step_at_snapshot

    metrics["worker_staleness"].labels(worker_id=worker_id).set(staleness)

    async with server_state.lock:
        block_metrics = server_state.apply_block_mla(pseudo_grad, worker_id, staleness)
        server_state.outer_step += 1
        metrics["outer_step"].inc()

    if WANDB_API_KEY:
        wandb.log({**block_metrics, "staleness": staleness, "outer_step": server_state.outer_step})
        
    return {"accepted": True, "current_step": server_state.outer_step}

@app.get("/model")
async def download_model(auth: HTTPAuthorizationCredentials = Depends(security)):
    await _verify_token(auth.credentials)
    compressed = encode_bf16_model(server_state.model)
    return StreamingResponse(io.BytesIO(compressed), media_type="application/octet-stream")

class HeartbeatReq(BaseModel):
    worker_id: str

@app.post("/heartbeat")
async def heartbeat(req: HeartbeatReq, auth: HTTPAuthorizationCredentials = Depends(security)):
    await _verify_token(auth.credentials, req.worker_id)
    if redis_client:
        await redis_client.setex(f"heartbeat:{req.worker_id}", 30, "alive")
    return {"ok": True}
