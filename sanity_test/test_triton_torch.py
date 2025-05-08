# benchmark_vec_add_fixed.py

import torch
import triton
import triton.language as tl

@triton.jit
def vec_add_triton(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask    = offsets < N
    x       = tl.load(x_ptr + offsets, mask=mask)
    y       = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def benchmark(N=10**7, BLOCK=256, runs=10):
    # Prepare inputs
    x   = torch.randn(N, device='cuda', dtype=torch.float32)
    y   = torch.randn(N, device='cuda', dtype=torch.float32)
    out = torch.empty_like(x)

    def time_gpu(fn):
        # Warm‑up
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        # Timing
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    # Triton launch wrapper
    grid = lambda META: ((N + META['BLOCK'] - 1) // META['BLOCK'],)
    triton_fn = lambda: vec_add_triton[grid](x, y, out, N, BLOCK=BLOCK)

    # PyTorch add wrapper (fixed!)
    torch_fn  = lambda: torch.add(x, y, out=out)

    # Run benchmarks
    t_triton = sum(time_gpu(triton_fn) for _ in range(runs)) / runs
    t_torch  = sum(time_gpu(torch_fn ) for _ in range(runs)) / runs

    print(f"Average over {runs} runs:")
    print(f"  Triton vec_add ({N:,} elements): {t_triton:7.3f} ms")
    print(f"  PyTorch   vec_add ({N:,} elements): {t_torch:7.3f} ms")

if __name__ == "__main__":
    benchmark()

