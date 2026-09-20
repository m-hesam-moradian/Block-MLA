import os
import time
import json
import uuid
import logging
from pathlib import Path

import httpx
import torch

from worker.model.nanogpt import build_nanogpt
from worker.compression.gradient_codec import encode_bf16_gradient, decode_bf16_gradient

logger = logging.getLogger(__name__)

def compute_pseudo_gradient(model: torch.nn.Module, theta_global: dict) -> dict:
    """
    Compute the pseudo-gradient for DiLoCo/Block-MLA outer optimization.
    Delta = theta_global - theta_local
    """
    return {
        name: theta_global[name] - param.data.clone()
        for name, param in model.named_parameters()
    }

class WorkerConfig:
    def __init__(self, server_url, rank, total_workers, inner_steps=80, model_size="tiny", platform="local"):
        self.server_url = server_url
        self.rank = rank
        self.total_workers = total_workers
        self.inner_steps = inner_steps
        self.model_size = model_size
        self.platform = platform

class BlockMLAWorker:
    """
    Asynchronous worker client for the Block-MLA parameter server.
    Runs indefinitely, pulling the model, training locally, and pushing updates.
    """
    IDENTITY_FILE = Path(".worker_identity.json")

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.worker_id, self.token = self._load_or_create_identity()
        self.model = build_nanogpt(config.model_size)
        self.inner_opt = torch.optim.AdamW(
            self.model.parameters(), lr=3e-4, weight_decay=0.1
        )
        self.current_server_step = 0

    def _detect_gpu(self) -> str:
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    def _load_or_create_identity(self) -> tuple[str, str]:
        if self.IDENTITY_FILE.exists():
            data = json.loads(self.IDENTITY_FILE.read_text())
            logger.info(f"Reconnecting as existing worker {data['worker_id']}")
            return data["worker_id"], data["token"]

        logger.info(f"Registering new worker with server {self.config.server_url}...")
        resp = httpx.post(
            f"{self.config.server_url}/register",
            json={
                "worker_id": str(uuid.uuid4()),
                "worker_rank": self.config.rank,
                "platform": self.config.platform,
                "gpu_type": self._detect_gpu(),
                "inner_steps": self.config.inner_steps,
            },
            timeout=10.0
        )
        resp.raise_for_status()
        data = resp.json()

        identity = {"worker_id": data.get("worker_id", str(uuid.uuid4())), "token": data["token"]}
        # Wait, the server register returns only token, reconnect, current_step in my main.py
        # If I want worker_id, I should just generate it locally and store it.
        # So I will generate the id beforehand and send it.
        worker_id = str(uuid.uuid4())
        resp = httpx.post(
            f"{self.config.server_url}/register",
            json={
                "worker_id": worker_id,
                "worker_rank": self.config.rank,
                "platform": self.config.platform,
                "gpu_type": self._detect_gpu(),
                "inner_steps": self.config.inner_steps,
            },
            timeout=10.0
        )
        resp.raise_for_status()
        data = resp.json()
        
        identity = {"worker_id": worker_id, "token": data["token"]}
        self.IDENTITY_FILE.write_text(json.dumps(identity))
        logger.info(f"Registered as new worker {identity['worker_id']}")
        return identity["worker_id"], identity["token"]

    @property
    def _auth_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "worker-id": self.worker_id,
        }

    def _sync_global_model(self):
        logger.info("Downloading global model from server...")
        resp = httpx.get(
            f"{self.config.server_url}/model",
            headers=self._auth_headers,
            timeout=120.0
        )
        resp.raise_for_status()
        global_params = decode_bf16_gradient(resp.content)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data.copy_(global_params[name])
        return global_params

    def run(self):
        logger.info("Starting worker loop...")
        while True:
            try:
                theta_global = self._sync_global_model()
                
                # Heartbeat
                httpx.post(
                    f"{self.config.server_url}/heartbeat",
                    json={"worker_id": self.worker_id},
                    headers=self._auth_headers,
                    timeout=5.0
                )

                # Inner loop
                self.model.train()
                for step in range(self.config.inner_steps):
                    # Fake random batch for this implementation
                    # In reality, you'd pull from SlimPajama here
                    x = torch.randint(0, self.model.config.vocab_size, (4, 64))
                    loss = self.model(x, targets=x).loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.inner_opt.step()
                    self.inner_opt.zero_grad()

                logger.info(f"Completed {self.config.inner_steps} local steps. Final loss: {loss.item():.4f}")

                # Compute pseudo-gradient
                pseudo_grad = compute_pseudo_gradient(self.model, theta_global)

                # Compress and upload
                logger.info("Compressing and uploading gradient...")
                compressed = encode_bf16_gradient(pseudo_grad)
                
                resp = httpx.post(
                    f"{self.config.server_url}/gradient",
                    content=compressed,
                    headers={
                        **self._auth_headers,
                        "outer-step-at-snapshot": str(self.current_server_step),
                        "content-type": "application/octet-stream",
                    },
                    timeout=120.0
                )
                resp.raise_for_status()
                ack = resp.json()
                self.current_server_step = ack["current_step"]
                logger.info(f"Gradient accepted. Server is now at step {self.current_server_step}")

            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
                logger.warning(f"Server communication failed: {e}. Retrying in 10s...")
                time.sleep(10)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    
    cfg = WorkerConfig(
        server_url=args.server,
        rank=args.rank,
        total_workers=20
    )
    worker = BlockMLAWorker(cfg)
    worker.run()
