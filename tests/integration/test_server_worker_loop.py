import torch
import pytest
from tests.fixtures import build_in_process_server, build_sim_workers

def generate_non_iid_batch(worker_idx, vocab_size, batch_size, block_size):
    """
    Simulates Dirichlet alpha=0.1 (strong non-IID) by completely segmenting 
    the vocabulary distribution per worker. This ensures that stale gradients 
    from a slow worker (like pace=15) are highly conflicting with the fast workers,
    which is exactly what Block-MLA's cosine gate is designed to catch and damp.
    """
    # Each worker gets a distinct window of the vocabulary
    window = max(100, vocab_size // 5)
    shift = (worker_idx * window) % vocab_size
    x = torch.randint(0, window, (batch_size, block_size))
    x = (x + shift) % vocab_size
    return x

def run_experiment(method, n_workers, n_steps):
    torch.manual_seed(42)
    server_model, apply_gradient = build_in_process_server(method)
    paces = [1, 5, 10, 15, 15]
    workers = build_sim_workers(n_workers, server_model, paces)
    
    vocab_size = server_model.config.vocab_size
    batch_size = 4
    block_size = 32
    inner_steps = 50

    outer_steps = 0
    tick = 0

    # All workers initially snapshot the global model
    for w in workers:
        w["snapshot"] = {n: p.data.clone().detach() for n, p in server_model.named_parameters()}
        w["ticks_working"] = 0

    while outer_steps < n_steps:
        tick += 1
        for i, w in enumerate(workers):
            w["ticks_working"] += 1
            
            # When the worker finishes its computation (based on pace)
            if w["ticks_working"] >= w["pace"]:
                # 1. Local inner steps (done on the snapshot)
                w["model"].train()
                # Initialize worker model with the snapshot
                with torch.no_grad():
                    for n, p in w["model"].named_parameters():
                        p.data.copy_(w["snapshot"][n])
                
                for _ in range(inner_steps):
                    x = generate_non_iid_batch(i, vocab_size, batch_size, block_size)
                    out = w["model"](x, targets=x)
                    out.loss.backward()
                    torch.nn.utils.clip_grad_norm_(w["model"].parameters(), 1.0)
                    w["opt"].step()
                    w["opt"].zero_grad()
                
                # 2. Compute Pseudo Gradient against the snapshot
                pseudo_grad = {
                    name: w["snapshot"][name] - param.data.clone()
                    for name, param in w["model"].named_parameters()
                }
                
                # 3. Server applies gradient (it's now stale because server has advanced!)
                apply_gradient(pseudo_grad)
                outer_steps += 1
                
                # 4. Worker downloads new snapshot and resets
                w["snapshot"] = {n: p.data.clone().detach() for n, p in server_model.named_parameters()}
                w["ticks_working"] = 0
                
                if outer_steps >= n_steps:
                    break
        if outer_steps >= n_steps:
            break

    # Evaluate validation loss
    server_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i in range(n_workers):
            x_val = generate_non_iid_batch(i, vocab_size, batch_size, block_size)
            out_val = server_model(x_val, targets=x_val)
            val_loss += out_val.loss.item()
            
    return val_loss / n_workers

@pytest.mark.slow
def test_block_mla_beats_async_nesterov():
    # We use a larger n_steps so the momentum and staleness divergence has time to accumulate
    # and the Block-MLA correction can prove its value.
    blockmla_loss   = run_experiment("block_mla",     n_workers=5, n_steps=60)
    nesterov_loss   = run_experiment("async_nesterov", n_workers=5, n_steps=60)
    
    print(f"Block-MLA loss: {blockmla_loss:.4f}")
    print(f"Async-Nesterov loss: {nesterov_loss:.4f}")
    
    assert blockmla_loss < nesterov_loss, (
        f"Block-MLA ({blockmla_loss:.4f}) did not beat "
        f"Async-Nesterov ({nesterov_loss:.4f}) — check algorithm before scaling."
    )
