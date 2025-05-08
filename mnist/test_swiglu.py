# test_swiglu.py
import torch
import triton
from fused_forward_kernel import swiglu_fused_forward_kernel
from fused_backward_kernel import swiglu_fused_backward_kernel

# Pure-PyTorch reference
def ref_swiglu(x, Wf):
    D = Wf.shape[1] // 2
    W1a, W1b = Wf[:, :D], Wf[:, D:]
    a = x @ W1a
    b = x @ W1b
    return a * b * torch.sigmoid(b)

def main():
    torch.manual_seed(1)
    device = "cuda"
    B, D = 128, 512
    BLOCK = 64

    # Scale random values to reduce extreme values
    Wf = torch.randn(D, 2*D, device=device, requires_grad=True) * 0.1  # Scale down
    x  = torch.randn(B, D,   device=device, requires_grad=True) * 0.1  # Scale down
    z_ref = ref_swiglu(x, Wf)

    # --- Forward test ---
    Z = torch.empty_like(x)
    grid = (triton.cdiv(B, BLOCK), triton.cdiv(D, BLOCK))
    swiglu_fused_forward_kernel[grid](
        x, Wf, Z, B, D,
        x.stride(0), x.stride(1),
        Wf.stride(0), Wf.stride(1),
        Z.stride(0), Z.stride(1),
        BLOCK_M=BLOCK, BLOCK_N=BLOCK
    )
    torch.cuda.synchronize()
    fwd_err = (Z - z_ref).abs().max().item()
    print(f"Forward max-abs error: {fwd_err:.2e}")

    # --- Backward test ---
    dZ = torch.randn_like(Z) * 0.1  # Scale down gradients too
    dX_ref, _ = torch.autograd.grad(z_ref, (x, Wf), grad_outputs=dZ)

    dX = torch.empty_like(x)
    swiglu_fused_backward_kernel[grid](
        x, Wf, dZ, dX, B, D,
        x.stride(0), x.stride(1),
        Wf.stride(0), Wf.stride(1),
        dZ.stride(0), dZ.stride(1),
        dX.stride(0), dX.stride(1),
        BLOCK_M=BLOCK, BLOCK_N=BLOCK
    )
    torch.cuda.synchronize()
    bwd_err = (dX - dX_ref).abs().max().item()
    print(f"Backward-dX max-abs error: {bwd_err:.2e}")

    if fwd_err < 1e-4 and bwd_err < 1e-4:
        print("Fused SwiGLU satisfy tolerance")
    else:
        print("Errors, larger than tolerance")

if __name__ == "__main__":
    main()