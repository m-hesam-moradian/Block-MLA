import torch
from worker.model.nanogpt import build_nanogpt
from server.optimizer.block_mla_optimizer import BlockMLAOptimizer, nesterov_bypass_step
from server.optimizer.parameter_filter import build_param_groups

def build_in_process_server(method="block_mla"):
    model = build_nanogpt("tiny")
    eligible, bypass = build_param_groups(model)
    
    optimizer = BlockMLAOptimizer(
        [{"params": [p for _, p in eligible]}],
        lr=0.7,
        momentum=0.9,
    )
    
    server_states = {}
    eligible_names = {name for name, _ in eligible}
    
    def apply_gradient(worker_pseudo_grad):
        with torch.no_grad():
            for name, param in model.named_parameters():
                grad = worker_pseudo_grad[name]
                state = server_states.setdefault(name, {})
                
                if method == "block_mla" and name in eligible_names:
                    optimizer.step_block(grad, param, state, optimizer.defaults)
                else:
                    nesterov_bypass_step(param, grad, state)

    return model, apply_gradient

def build_sim_workers(n_workers, server_model, paces):
    workers = []
    for i in range(n_workers):
        worker_model = build_nanogpt("tiny")
        inner_opt = torch.optim.AdamW(worker_model.parameters(), lr=3e-4)
        workers.append({
            "id": f"worker_{i}",
            "model": worker_model,
            "opt": inner_opt,
            "pace": paces[i],
            "step_counter": 0
        })
    return workers
