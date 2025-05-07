import torch, time, triton, torch.optim as optim
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from fused_kernel import swiglu_fused_kernel

def train_liger(B, D, iters=50):
    W1a = torch.nn.Parameter(torch.randn(D, D, device='cuda'))
    W1b = torch.nn.Parameter(torch.randn(D, D, device='cuda'))
    W2  = torch.nn.Parameter(torch.randn(D, D, device='cuda'))
    opt = optim.SGD([W1a, W1b, W2], lr=0.01)
    x = torch.randn(B, D, device='cuda')
    for _ in range(5):  # warmup
        opt.zero_grad()
        z = LigerSiLUMulFunction.apply(x @ W1a, x @ W1b)
        loss = (z @ W2).sum()
        loss.backward(); opt.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad()
        z = LigerSiLUMulFunction.apply(x @ W1a, x @ W1b)
        loss = (z @ W2).sum()
        loss.backward(); opt.step()
    torch.cuda.synchronize()
    print(f"Liger-train B={B} D={D} → {(time.perf_counter()-t0)/iters*1000:6.2f} ms")

def train_fused(B, D, iters=50, block=64):
    Wf  = torch.nn.Parameter(torch.randn(D, 2*D, device='cuda'))
    W2  = torch.nn.Parameter(torch.randn(D,   D, device='cuda'))
    opt = optim.SGD([Wf, W2], lr=0.01)
    x   = torch.randn(B, D, device='cuda')
    Z   = torch.empty(B, D, device='cuda')
    grid = (triton.cdiv(B,block), triton.cdiv(D,block))
    for _ in range(5):  # warmup
        opt.zero_grad()
        swiglu_fused_kernel[grid](x, Wf, Z, B, D,
            x.stride(0), x.stride(1),
            Wf.stride(0), Wf.stride(1),
            Z.stride(0), Z.stride(1),
            BLOCK_M=block, BLOCK_N=block)
        loss = (Z @ Wf[:, :D]).sum()
        loss.backward(); opt.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad()
        swiglu_fused_kernel[grid](x, Wf, Z, B, D,
            x.stride(0), x.stride(1),
            Wf.stride(0), Wf.stride(1),
            Z.stride(0), Z.stride(1),
            BLOCK_M=block, BLOCK_N=block)
        loss = (Z @ Wf[:, :D]).sum()
        loss.backward(); opt.step()
    torch.cuda.synchronize()
    print(f"Fused-train B={B} D={D} → {(time.perf_counter()-t0)/iters*1000:6.2f} ms")

if __name__ == "__main__":
    for B in [256, 512]:
        for D in [2048, 4096]:
            train_liger(B, D)
            train_fused(B, D)
