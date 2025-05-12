import triton
import triton.language as tl

@triton.jit
def swiglu_fused_backward_kernel(
    X_ptr, W_ptr, dZ_ptr, dX_ptr,      # pointers
    B, D,                              # batch size, hidden dim
    stride_xb, stride_xd,              # x strides
    stride_wd, stride_w2d,             # Wf strides
    stride_dzb, stride_dzd,            # dZ strides
    stride_dxb, stride_dxd,            # dX strides
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # 1) tile indices
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # row offsets
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # col offsets

    # 2) load X tile [BLOCK_M×BLOCK_N]
    x_tile = tl.load(
        X_ptr + offs_m[:, None]*stride_xb + offs_n[None, :]*stride_xd,
        mask=(offs_m[:, None]<B)&(offs_n[None,:]<D),
        other=0.0
    )

    # 3) load fused‐weight half A (W₁ₐ) [BLOCK_N×BLOCK_N]
    w_a = tl.load(
        W_ptr + offs_n[:, None]*stride_wd + tl.arange(0, BLOCK_N)[None, :]*stride_w2d,
        mask=(offs_n[:, None]<D)&(tl.arange(0, BLOCK_N)[None, :]<2*D),
        other=0.0
    )
    
    # 4) load fused‐weight half B (W₁ᵦ) [BLOCK_N×BLOCK_N]
    w_b = tl.load(
        W_ptr + offs_n[:, None]*stride_wd + (tl.arange(0, BLOCK_N)[None, :]+D)*stride_w2d,
        mask=(offs_n[:, None]<D)&((tl.arange(0, BLOCK_N)[None, :]+D)<2*D),
        other=0.0
    )

    # 5) load dZ tile [BLOCK_M×BLOCK_N]
    dz_tile = tl.load(
        dZ_ptr + offs_m[:, None]*stride_dzb + offs_n[None, :]*stride_dzd,
        mask=(offs_m[:, None]<B)&(offs_n[None,:]<D),
        other=0.0
    )

    # 6) recompute forward branch activations
    a = tl.dot(x_tile, w_a)                    # = X·W₁ₐ
    b = tl.dot(x_tile, w_b)                    # = X·W₁ᵦ
    s = tl.sigmoid(b)                          # gate σ(b)

    # 7) compute branch gradients
    dA = dz_tile * (b * s)                                 # ∂L/∂A
    dB = dz_tile * (a * s + a * b * s * (1 - s))           # ∂L/∂B

    # 8) compute dX = dA·W₁ₐᵀ + dB·W₁ᵦᵀ
    dXa = tl.dot(dA, tl.trans(w_a))
    dXb = tl.dot(dB, tl.trans(w_b))
    dX_tile = dXa + dXb

    # 9) store dX back
    tl.store(
        dX_ptr + offs_m[:, None]*stride_dxb + offs_n[None, :]*stride_dxd,
        dX_tile,
        mask=(offs_m[:, None]<B)&(offs_n[None,:]<D)
    )
