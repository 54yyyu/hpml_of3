# bench_swiglu_vs_relu.py

import torch
import time
import triton
import triton.language as tl

# Triton‐fused SwiGLU kernel: one GEMM for both branches + gate in a single launch
@triton.jit
def swiglu_fused_kernel(
    X_ptr, W_ptr, Z_ptr,
    B, D,
    stride_xb, stride_xd,
    stride_wd, stride_w2d,
    stride_zb, stride_zd,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # load X tile
    x_tile = tl.load(
        X_ptr + offs_m[:, None] * stride_xb + offs_n[None, :] * stride_xd,
        mask=(offs_m[:, None] < B) & (offs_n[None, :] < D),
        other=0.0
    )

    # load fused weight tiles for branch A and branch B
    w_a = tl.load(
        W_ptr + offs_n[:, None] * stride_wd + tl.arange(0, BLOCK_N)[None, :] * stride_w2d,
        mask=(offs_n[:, None] < D) & (tl.arange(0, BLOCK_N)[None, :] < 2*D),
        other=0.0
    )
    w_b = tl.load(
        W_ptr + offs_n[:, None] * stride_wd + (tl.arange(0, BLOCK_N)[None, :] + D) * stride_w2d,
        mask=(offs_n[:, None] < D) & ((tl.arange(0, BLOCK_N)[None, :] + D) < 2*D),
        other=0.0
    )

    # two GEMMs: X·W_a and X·W_b
    a = tl.dot(x_tile, w_a)
    b = tl.dot(x_tile, w_b)

    # SwiGLU gate
    z = a * tl.sigmoid(b)

    # write back z tile
    tl.store(
        Z_ptr + offs_m[:, None] * stride_zb + offs_n[None, :] * stride_zd,
        z,
        mask=(offs_m[:, None] < B) & (offs_n[None, :] < D)
    )


def bench_pytorch(B, D, iters=200):
    W1 = torch.randn(D, D, device='cuda')
    b1 = torch.randn(D,    device='cuda')
    W2 = torch.randn(D, D, device='cuda')
    x  = torch.randn(B, D, device='cuda')

    # Warm-up
    for _ in range(5):
        y = torch.relu(x @ W1 + b1)
        _ = y @ W2
    torch.cuda.synchronize()

    # Timed loop
    t0 = time.perf_counter()
    for _ in range(iters):
        y = torch.relu(x @ W1 + b1)
        out = y @ W2
        torch.cuda.synchronize()
        _ = out.sum().item()
    print(f"PyTorch ReLU        B={B:<3d} D={D:<4d} → {(time.perf_counter()-t0)/iters*1000:6.2f} ms")


def bench_triton(B, D, iters=200, block=64):
    x   = torch.randn(B, D,   device='cuda')
    Wf  = torch.randn(D, 2*D, device='cuda')  # fused weight for both branches
    Z   = torch.empty(B, D,   device='cuda')
    out = torch.empty(B, D,   device='cuda')

    # compile & warm-up
    grid = (triton.cdiv(B, block), triton.cdiv(D, block))
    for _ in range(5):
        swiglu_fused_kernel[grid](
            x, Wf, Z,
            B, D,
            x.stride(0), x.stride(1),
            Wf.stride(0), Wf.stride(1),
            Z.stride(0), Z.stride(1),
            BLOCK_M=block, BLOCK_N=block
        )
        _ = Z @ Wf[:, :D]  # dummy to warm-up second GEMM
    torch.cuda.synchronize()

    # Timed loop
    t0 = time.perf_counter()
    for _ in range(iters):
        # fused branch GEMM + gate
        swiglu_fused_kernel[grid](
            x, Wf, Z,
            B, D,
            x.stride(0), x.stride(1),
            Wf.stride(0), Wf.stride(1),
            Z.stride(0), Z.stride(1),
            BLOCK_M=block, BLOCK_N=block
        )
        # final projection (second GEMM)
        out = Z @ Wf[:, :D]
        torch.cuda.synchronize()
        _ = out.sum().item()
    print(f"Triton-fused SwiGLU B={B:<3d} D={D:<4d} → {(time.perf_counter()-t0)/iters*1000:6.2f} ms")


if __name__ == "__main__":
    print("\nBenchmarking on your L4 (forward-only, two GEMMs + activation):\n")
    for B, D in [(64, 4096), (128, 4096), (256, 4096)]:
        bench_pytorch(B, D)
        bench_triton(B, D)
