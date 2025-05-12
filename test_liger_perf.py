import time
import torch
import wandb
from liger_kernel.ops.swiglu import swiglu_forward, swiglu_backward

# ─── Microbenchmark function ───────────────────────────────────────────────────
def benchmark_swiglu(batch: int, dim: int, runs: int, warmups: int):
    # Initialize a new W&B run for this configuration
    run = wandb.init(
        project="liger-swiglu-microbench",
        name=f"batch{batch}_dim{dim}",
        config={"batch": batch, "dim": dim, "runs": runs, "warmups": warmups},
        reinit=True
    )

    device = "cuda"
    a = torch.randn(batch, dim, device=device)
    b = torch.randn(batch, dim, device=device)
    grad_out = torch.randn_like(a)

    # Warm‑up
    for _ in range(warmups):
        out, _, _ = swiglu_forward(a, b)
        swiglu_backward(a, b, grad_out)
        torch.cuda.synchronize()

    # Create CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    # Timed runs
    torch.cuda.synchronize()
    start.record()
    for _ in range(runs):
        out, _, _ = swiglu_forward(a, b)
        swiglu_backward(a, b, grad_out)
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_ms   = total_ms / runs

    # Log to W&B
    wandb.log({
        "total_time_ms": total_ms,
        "avg_time_per_run_ms": avg_ms
    })

    run.finish()

# ─── Main: iterate over configurations ─────────────────────────────────────────
if __name__ == "__main__":
    configs = [
        (32,  4096),
        (64,  4096),
        (32,  8192),
        (128, 8192),
    ]
    runs = 200
    warmups = 20

    for batch, dim in configs:
        benchmark_swiglu(batch=batch, dim=dim, runs=runs, warmups=warmups)


