import torch
from liger_kernel.ops.swiglu import swiglu_forward, swiglu_backward

def benchmark_swiglu(batch=32, dim=4096, runs=100, warmups=10):
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
    print(f"Liger SwiGLU (batch={batch}, dim={dim}):")
    print(f"  Total over {runs} runs: {total_ms:.2f} ms")
    print(f"  Average per run:      {avg_ms:.4f} ms")

if __name__ == "__main__":
    benchmark_swiglu(batch=32,  dim=4096, runs=200, warmups=20)
    benchmark_swiglu(batch=64,  dim=4096, runs=200, warmups=20)
    benchmark_swiglu(batch=32,  dim=8192, runs=200, warmups=20)
    benchmark_swiglu(batch=128, dim=8192, runs=200, warmups=20)
