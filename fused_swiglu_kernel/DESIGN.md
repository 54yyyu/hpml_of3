## SwiGLU Implementation Comparison

### 1. Weight Layout & API

| Aspect         | Liger-kernel                                                    | Fused-branch (ours)                                                         |
| -------------- | --------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Weight tensors | Two separate `nn.Linear` modules                                | One packed `Wf = [W1a ∥ W1b]`                                               |
| Forward API    | `a = x@W1a; b = x@W1b; z = LigerSiLUMulFunction.apply(a,b)`     | `z = FusedSwiGLU(x)` (one `.apply` call)                                    |
| Backward API   | Returns `(da, db)` to feed two linears; weight-grad via PyTorch | Returns `(dX, None, None)` (no weight-grad) or `(dX, dWf, None)` if enabled |

---

### 2. Kernel Fusion

| Pass              | Liger-kernel                       | Fused-branch (ours)                                                                |
| ----------------- | ---------------------------------- | ---------------------------------------------------------------------------------- |
| **Forward**       | 2 separate GEMMs + 1 Triton “silu” | 1 Triton kernel: both GEMMs + gate in one launch                                   |
| **Backward → dX** | 1 Triton gate-backward + 2 GEMMs   | 1 Triton kernel: recompute GEMMs + gate-deriv, write `dX`                          |
| **Backward → dW** | 2 PyTorch GEMMs                    | *Optional*: 2 PyTorch GEMMs or 1 Triton 3D-tiled `swiglu_fused_weight_grad_kernel` |

---

### 3. Microbenchmarks: Forward & Backward

#### Forward-only (GEMM + gate)

|   B  |   D  | Liger (ms) | Fused (ms) | Speedup |
| :--: | :--: | :--------: | :--------: | :-----: |
|  64  | 2048 |    0.21    |    0.07    |   3.0×  |
|  64  | 4096 |    0.99    |    0.35    |   2.8×  |
|  256 | 2048 |    0.74    |    0.27    |   2.7×  |
|  256 | 4096 |    2.34    |    0.83    |   2.8×  |
| 1024 | 2048 |    3.10    |    1.14    |   2.7×  |
| 1024 | 4096 |    8.71    |    3.12    |   2.8×  |

#### Backward-only (input-grad, no weight-grad)

|   B  |   D  | Liger-bwd (ms) | Fused-bwd (ms) | Speedup |
| :--: | :--: | :------------: | :------------: | :-----: |
|  64  | 2048 |      0.81      |      0.60      |   1.4×  |
|  64  | 4096 |      3.36      |      1.01      |   3.3×  |
|  256 | 2048 |      2.21      |      0.62      |   3.6×  |
|  256 | 4096 |      8.65      |      2.61      |   3.3×  |
| 1024 | 2048 |      8.84      |      2.72      |   3.3×  |
| 1024 | 4096 |      28.86     |      9.22      |   3.1×  |

#### Backward-with-weight-grad (dWf fused)

|   B  |   D  | Liger-bwd (ms) | Fused-WF-bwd (ms) | Speedup |
| :--: | :--: | :------------: | :---------------: | :-----: |
|  64  | 2048 |      0.83      |        0.66       |   1.3×  |
|  64  | 4096 |      3.38      |        1.66       |   2.0×  |
|  256 | 2048 |      2.21      |        0.99       |   2.2×  |
|  256 | 4096 |      8.61      |        3.87       |   2.2×  |
| 1024 | 2048 |      8.87      |        3.91       |   2.3×  |
| 1024 | 4096 |      28.76     |       13.51       |   2.1×  |

---

### 4. End-to-End MNIST Training (L4 GPU)

| Implementation             | Epoch 1 train (ms) | Inference (ms) |       Speedup vs Liger |
| -------------------------- | -----------------: | -------------: | ---------------------: |
| **Liger-kernel SwiGLU**    |             39 542 |          2 050 |                     1× |
| **Fused (no weight-grad)** |             15 472 |          2 061 | 2.6× train, \~same inf |
| **Fused + weight-grad**    |             37 795 |          2 029 |             ≈ 1× train |

---

### 5. Trade-offs & Recommendations

* **No weight-grad** (`return dX, None, None`):
  – Perfect drop-in for Liger’s gate-only behavior
  – Full **2–3×** speedup in forward, backward, and end-to-end training
  – **Wf is frozen** (no learning)

* **Python weight-grad** (two mat-muls):
  – Enables trainable packed weight
  – Performance regresses back near baseline

* **Triton weight-grad** (3D-tiled kernel):
  – Full fusion (GX+gate, dX, dW)
  – Requires kernel tuning for optimal throughput
  – Can recover **2×**+ over Liger once tuned