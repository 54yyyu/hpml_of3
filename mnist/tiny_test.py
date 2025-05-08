# tiny_test.py
import torch
import math
import triton
from fused_forward_kernel import swiglu_fused_forward_kernel
from fused_backward_kernel import swiglu_fused_backward_kernel

def ref_swiglu(x, Wf):
    D = Wf.shape[1] // 2
    W1a, W1b = Wf[:, :D], Wf[:, D:]
    a = x @ W1a
    b = x @ W1b
    return a * b * torch.sigmoid(b)

def main():
    torch.manual_seed(0)
    device = "cuda"
    
    # Small test case with dimensions at least 16
    B, D = 16, 16
    BLOCK = 16
    grid = (1, 1)  # single 16×16 tile
    
    # Create test data with manageable values
    x = torch.ones((B, D), device=device, requires_grad=True)
    
    # Create weight matrix without in-place operations
    Wf_first_half = torch.ones((D, D), device=device)
    Wf_second_half = 2 * torch.ones((D, D), device=device)
    Wf = torch.cat([Wf_first_half, Wf_second_half], dim=1).requires_grad_()
    
    dZ = torch.ones((B, D), device=device)  # Gradient all ones for simplicity
    
    # Reference calculation
    z_ref = ref_swiglu(x, Wf)
    dX_ref, _ = torch.autograd.grad(z_ref, (x, Wf), grad_outputs=dZ)
    
    # Your kernels
    Z_fused = torch.empty_like(x)
    swiglu_fused_forward_kernel[grid](
        x, Wf, Z_fused, B, D,
        x.stride(0), x.stride(1),
        Wf.stride(0), Wf.stride(1),
        Z_fused.stride(0), Z_fused.stride(1),
        BLOCK_M=BLOCK, BLOCK_N=BLOCK,
    )
    
    dX_fused = torch.empty_like(x)
    swiglu_fused_backward_kernel[grid](
        x, Wf, dZ, dX_fused, B, D,
        x.stride(0), x.stride(1),
        Wf.stride(0), Wf.stride(1),
        dZ.stride(0), dZ.stride(1),
        dX_fused.stride(0), dX_fused.stride(1),
        BLOCK_M=BLOCK, BLOCK_N=BLOCK,
    )
    
    # Print only first few values for readability
    print("First few values of Reference Z:")
    print(z_ref[0:2, 0:2].detach().cpu().numpy())
    print("\nFirst few values of Fused Z:")
    print(Z_fused[0:2, 0:2].cpu().numpy())
    print("\nFirst few values of Reference dX:")
    print(dX_ref[0:2, 0:2].detach().cpu().numpy())
    print("\nFirst few values of Fused dX:")
    print(dX_fused[0:2, 0:2].cpu().numpy())
    
    # Print errors
    fwd_err = (Z_fused - z_ref).abs().max().item()
    bwd_err = (dX_fused - dX_ref).abs().max().item()
    print(f"\nForward max-abs error: {fwd_err:.2e}")
    print(f"Backward-dX max-abs error: {bwd_err:.2e}")
    
    # Calculate expected values manually for verification
    print("\nManual calculation for first element:")
    # For x = 1s and Wf = [1s, 2s], we expect:
    # W1a = 1s, W1b = 2s
    # a = x @ W1a = 16 (dot product of 16 ones with 16 ones)
    # b = x @ W1b = 32 (dot product of 16 ones with 16 twos)
    # sigmoid(32) ≈ 1.0
    # z = a * b * sigmoid(b) = 16 * 32 * 1.0 = 512
    print(f"Expected first Z value: {16 * 32 * (1.0 / (1.0 + math.exp(-32))):.4f}")

if __name__ == "__main__":
    main()