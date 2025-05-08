import triton, triton.language as tl

@triton.jit
def swiglu_fused_forward_kernel(
    X_ptr, W_ptr, Z_ptr,
    B, D,
    stride_xb, stride_xd,
    stride_wd, stride_w2d,
    stride_zb, stride_zd,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x_tile = tl.load(
        X_ptr + offs_m[:, None]*stride_xb + offs_n[None, :]*stride_xd,
        mask=(offs_m[:, None]<B)&(offs_n[None,:]<D), other=0.0
    )
    # two half-tiles of the fused weight
    w_a = tl.load(
        W_ptr + offs_n[:,None]*stride_wd + tl.arange(0,BLOCK_N)[None,:]*stride_w2d,
        mask=(offs_n[:,None]<D)&(tl.arange(0,BLOCK_N)[None,:]<2*D), other=0.0
    )
    w_b = tl.load(
        W_ptr + offs_n[:,None]*stride_wd + (tl.arange(0,BLOCK_N)[None,:]+D)*stride_w2d,
        mask=(offs_n[:,None]<D)&((tl.arange(0,BLOCK_N)[None,:]+D)<2*D), other=0.0
    )
    a = tl.dot(x_tile, w_a)
    b = tl.dot(x_tile, w_b)
    z = a * b * tl.sigmoid(b)
    
    tl.store(
        Z_ptr + offs_m[:,None]*stride_zb + offs_n[None,:]*stride_zd,
        z, mask=(offs_m[:,None]<B)&(offs_n[None,:]<D)
    )
