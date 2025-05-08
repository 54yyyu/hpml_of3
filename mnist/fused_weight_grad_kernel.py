import triton, triton.language as tl

@triton.jit
def swiglu_fused_weight_grad_kernel(
    X_ptr, dZ_ptr, Wg_ptr,
    B, IFeats, D,
    stride_xb, stride_xf,
    stride_dzb, stride_dzd,
    stride_wgb, stride_wgd,
    BLOCK_B: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_m, pid_i, pid_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_b = pid_m*BLOCK_B + tl.arange(0, BLOCK_B)
    offs_i = pid_i*BLOCK_I + tl.arange(0, BLOCK_I)
    offs_d = pid_n*BLOCK_D + tl.arange(0, BLOCK_D)

    # load X tile
    x_tile = tl.load(
        X_ptr + offs_b[:,None]*stride_xb + offs_i[None,:]*stride_xf,
        mask=(offs_b[:,None]<B)&(offs_i[None,:]<IFeats), other=0.0
    )
    # load dZ tiles for both branches
    dZA = tl.load(
        dZ_ptr + offs_b[:,None]*stride_dzb + offs_d[None,:]*stride_dzd,
        mask=(offs_b[:,None]<B)&(offs_d[None,:]<D), other=0.0
    )
    dZB = tl.load(
        dZ_ptr + offs_b[:,None]*stride_dzb + (offs_d[None,:]+D)*stride_dzd,
        mask=(offs_b[:,None]<B)&((offs_d[None,:]+D)<2*D), other=0.0
    )

    # compute partial weight‐grads
    dW1a = tl.dot(tl.trans(x_tile), dZA)   # [BLOCK_I×BLOCK_D]
    dW1b = tl.dot(tl.trans(x_tile), dZB)

    # write into Wg first half
    tl.store(
      Wg_ptr + offs_i[:,None]*stride_wgb + offs_d[None,:]*stride_wgd,
      dW1a,
      mask=(offs_i[:,None]<IFeats)&(offs_d[None,:]<D)
    )
    # then second half
    tl.store(
      Wg_ptr + offs_i[:,None]*stride_wgb + (offs_d[None,:]+D)*stride_wgd,
      dW1b,
      mask=(offs_i[:,None]<IFeats)&((offs_d[None,:]+D)<2*D)
    )
