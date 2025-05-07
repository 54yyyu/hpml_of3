# bench_compare_swiglu_impls.py

import torch
import time
import triton
import triton.language as tl
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

# ——— Custom fused‐branch Triton SwiGLU kernel ———
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

    x_tile = tl.load(
        X_ptr + offs_m[:, None] * stride_xb + offs_n[None, :] * stride_xd,
        mask=(offs_m[:, None] < B) & (offs_n[None, :] < D),
        other=0.0
    )
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
    a = tl.dot(x_tile, w_a)
    b = tl.dot(x_tile, w_b)
    z = a * tl.sigmoid(b)
    tl.store(
        Z_ptr + offs_m[:, None] * stride_zb + offs_n[None, :] * stride_zd,
        z,
        mask=(offs_m[:, None] < B) & (offs_n[None, :] < D)
    )

def bench_fused_triton(B, D, iters=200, block=64):
    x   = torch.randn(B, D,   device='cuda')
    Wf  = torch.randn(D, 2*D, device='cuda')
    Z   = torch.empty(B, D,   device='cuda')
    out = torch.empty(B, D,   device='cuda')

    # compile & warm-up
    grid = (triton.cdiv(B, block), triton.cdiv(D, block))
    for _ in range(5):
        swiglu_fused_kernel[grid](
            x, Wf, Z, B, D,
            x.stride(0), x.stride(1),
            Wf.stride(0), Wf.stride(1),
            Z.stride(0), Z.stride(1),
            BLOCK_M=block, BLOCK_N=block
        )
        _ = Z @ Wf[:, :D]
    torch.cuda.synchronize()

    # timed
    t0 = time.perf_counter()
    for _ in range(iters):
        swiglu_fused_kernel[grid](
            x, Wf, Z, B, D,
            x.stride(0), x.stride(1),
            Wf.stride(0), Wf.stride(1),
            Z.stride(0), Z.stride(1),
            BLOCK_M=block, BLOCK_N=block
        )
        out = Z @ Wf[:, :D]
        torch.cuda.synchronize()
        _ = out.sum().item()
    print(f"Fused-Triton SwiGLU B={B} D={D} → {(time.perf_counter()-t0)/iters*1000:6.2f} ms")

# ——— Liger-kernel SwiGLU implementation ———
def bench_liger_kernel(B, D, iters=200):
    W1a = torch.randn(D, D, device='cuda')
    W1b = torch.randn(D, D, device='cuda')
    W2  = torch.randn(D, D, device='cuda')
    x   = torch.randn(B, D, device='cuda')

    # warm-up
    for _ in range(5):
        a = x @ W1a
        b = x @ W1b
        z = LigerSiLUMulFunction.apply(a, b)
        _ = z @ W2
    torch.cuda.synchronize()

    # timed
    t0 = time.perf_counter()
    for _ in range(iters):
        a = x @ W1a
        b = x @ W1b
        z = LigerSiLUMulFunction.apply(a, b)
        out = z @ W2
        torch.cuda.synchronize()
        _ = out.sum().item()
    print(f"Liger-kernel SwiGLU B={B} D={D} → {(time.perf_counter()-t0)/iters*1000:6.2f} ms")


if __name__ == "__main__":
    print("\nCompare fused-Triton vs Liger-kernel SwiGLU on L4:\n")
    for B, D in [(64,4096), (128,4096), (256,4096)]:
        bench_fused_triton(B, D)
        bench_liger_kernel(B, D)
