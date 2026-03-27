---
name: kernel_agent
description: Generate high-quality Triton kernels from PyTorch code using a multi-stage pipeline (test generation → kernel synthesis → verification → iterative refinement). Based on Meta's KernelAgent.
metadata:
  openclaw:
    emoji: "🚀"
    homepage: https://github.com/pytorch-labs/KernelAgent
    skillKey: kernel-agent
    requires: { "bins": [], "env": [], "config": [] }
---

# KernelAgent - Triton Kernel Generation

Use this skill when you need to convert PyTorch operations into optimized Triton kernels. This follows Meta's KernelAgent methodology with a focus on correctness, performance, and fusion.

## Core Philosophy

**Three-Stage Pipeline:**
1. **Test Generation** - Create a validation harness that defines what "correct" looks like
2. **Kernel Synthesis** - Generate Triton implementations that pass the test
3. **Iterative Refinement** - Fix errors, optimize, and verify until tests pass

**Fusion-First Approach:** Always prefer a single fused kernel over multiple kernels. Fuse compatible operations (matmul + bias + activation, QK^T + softmax + PV, etc.).

## Two Workflow Modes

### Mode A: Simple Kernels (Single Operation)
For straightforward operations (element-wise, simple GEMM, single activation):
- Analyze the operation
- Write test harness
- Generate single Triton kernel
- Verify and iterate

### Mode B: Complex Models (Subgraph Fusion Pipeline)
For complex models with multiple operations:
1. **Extract Subgraphs** - Identify fusable operation chains
2. **Generate Per-Subgraph Kernels** - Treat each subgraph independently
3. **Compose End-to-End** - Stitch verified kernels into complete program

---

## Step 1: Analyze the PyTorch Code

Before generating anything, understand what operations need to be kernelized:

```python
# What operations are present?
# - Linear/matmul operations
# - Activations (ReLU, GELU, sigmoid, etc.)
# - Normalization (LayerNorm, BatchNorm)
# - Attention patterns (QK^T, softmax, PV)
# - Reductions (sum, mean, max)
# - Element-wise ops

# What are the shapes?
# - Input tensor shapes (batch, channels, height, width)
# - Weight shapes (output_channels, input_channels, kernel_size)
# - Output shape

# What dtype and device?
# - torch.float32, torch.float16, torch.bfloat16, torch.float8
# - 'cuda', 'cpu', 'xpu'

# What can be fused together?
# - matmul + bias + activation → single kernel
# - QK^T + softmax + PV → fused attention
# - conv + batchnorm (+ activation) → fused conv-bn
```

## Step 2: Generate Test Harness

The test is your source of truth. It must:

1. Import the kernel function: `from kernel import kernel_function`
2. Create input tensors with known shapes/dtypes
3. Compute reference output using PyTorch
4. Call the Triton kernel
5. Compare results (tolerance-based for floating point)
6. Print "PASS" and exit(0) on success

**Test Template:**
```python
import torch
import numpy as np

def get_inputs():
    """Return input tensors as a list."""
    a = torch.randn(1024, device='cuda', dtype=torch.float32)
    b = torch.randn(1024, device='cuda', dtype=torch.float32)
    return [a, b]

def reference_forward(a, b):
    """PyTorch reference implementation."""
    return a + b  # or whatever the operation is

def test_kernel():
    """Validate Triton kernel against reference."""
    inputs = get_inputs()
    ref_output = reference_forward(*inputs)

    # Call Triton kernel (returns output as PyTorch tensor)
    triton_output = kernel_function(*inputs)

    # Compare with tolerance
    assert torch.allclose(triton_output, ref_output, rtol=1e-3, atol=1e-3), \
        f"Max diff: {(triton_output - ref_output).abs().max()}"

    print("PASS")
    return True

if __name__ == "__main__":
    test_kernel()
```

## Step 3: Generate Triton Kernel

### Critical Structure Requirements

**Wrapper Function (`kernel_function`):**
- Takes normal Python/PyTorch arguments
- Performs shape/dtype validation
- Allocates output tensors
- Calculates grid configuration
- Launches the Triton kernel
- Returns results as PyTorch tensors

**Actual Kernel (`@triton.jit` decorated function):**
- All computation happens here using Triton operations
- NO PyTorch compute operations allowed
- Uses `tl.load`, `tl.store`, `tl.dot`, `tl.reduce`, etc.

### Forbidden PyTorch Operations

**NEVER use these in the kernel:**
```python
# torch.nn modules
import torch.nn
from torch import nn
nn.Linear(...), nn.ReLU(), etc.

# torch.nn.functional
import torch.nn.functional as F
F.relu(x), F.gelu(x), F.sigmoid(x), F.conv2d(...), etc.

# Direct torch operations for compute
torch.add(x, y), torch.matmul(x, y), torch.sum(x)
x.add(y), x.matmul(y), x.sum()

# Low-level aten ops
torch.ops.aten.conv2d, torch.ops.aten.addmm
torch.ops.aten.layer_norm, torch.ops.aten.mean

# PyTorch activations/pooling
torch.relu, torch.sigmoid, torch.tanh, torch.gelu
torch.max_pool2d, torch.avg_pool2d
```

### Allowed PyTorch Usage

**Only in wrapper for setup:**
```python
# Allocation (output tensors)
output = torch.empty_like(input_a)
output = torch.empty((M, N), device='cuda', dtype=torch.float32)

# Shape/dtype checks (assertions)
assert input_a.shape == (M, K)
assert input_a.dtype == torch.float32

# Grid calculation helpers
grid = (triton.cdiv(N, BLOCK_SIZE),)
```

### Kernel Implementation Patterns

#### Element-wise Operations
```python
@triton.jit
def _kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    # Load using Triton
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute using Triton ops
    result = x + y

    # Store using Triton
    tl.store(out_ptr + offsets, result, mask=mask)

def kernel_function(x, y):
    """Element-wise addition kernel."""
    output = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _kernel[grid](x, y, output, n, BLOCK_SIZE)
    return output
```

#### Matrix Multiplication (GEMM)
```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
                  M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_SIZE_M: tl.constexpr,
                  BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr):
    # Program and block IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Masks for boundary
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am +
                         (k + offs_k[None, :]) * stride_ak)
        b_ptrs = b_ptr + ((k + offs_k[:, None]) * stride_bk +
                         offs_n[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=(mask_m[:, None] &
                  ((k + offs_k)[None, :] < K)), other=0.0)
        b = tl.load(b_ptrs, mask=((k + offs_k)[:, None] < K) &
                  mask_n[None, :], other=0.0)

        # Tensor core matmul
        acc = tl.dot(a, b, acc)

    # Store output
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(mask_m[:, None] & mask_n[None, :]))

def matmul(a, b):
    """GEMM wrapper."""
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
    matmul_kernel[grid](a, b, c, M, N, K,
                       a.stride(0), a.stride(1),
                       b.stride(0), b.stride(1),
                       c.stride(0), c.stride(1),
                       BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32)
    return c
```

#### Fused Conv + BatchNorm + Activation
```python
@triton.jit
def fused_conv_bn_act_kernel(x_ptr, weight_ptr, bias_ptr, mean_ptr, var_ptr,
                             gamma_ptr, beta_ptr, out_ptr,
                             N, C, H, W, OC,
                             stride_h, stride_w,
                             kernel_h, kernel_w,
                             eps: tl.constexpr,
                             BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    oc = pid

    # Load BN parameters for this output channel
    mean = tl.load(mean_ptr + oc).to(tl.float32)
    var = tl.load(var_ptr + oc).to(tl.float32)
    gamma = tl.load(gamma_ptr + oc).to(tl.float32)
    beta = tl.load(beta_ptr + oc).to(tl.float32)

    # Conv accumulator
    acc = tl.zeros((kernel_h, kernel_w), dtype=tl.float32)

    # Loop over input channels
    for ci in range(C):
        for hi in range(kernel_h):
            for wi in range(kernel_w):
                x_ptrs = x_ptr + (oc * stride_h + ci * H * W +
                                 (hi + 0) * W + (wi + 0))
                w = tl.load(weight_ptr + (oc * C * kernel_h * kernel_w +
                                          ci * kernel_h * kernel_w + hi * kernel_w + wi))
                acc = tl.sum(acc + w * x_ptrs, axis=None)

    # BN + activation (ReLU)
    x_norm = (acc - mean) / tl.sqrt(var + eps)
    x_hat = x_norm * gamma + beta
    out = tl.where(x_hat > 0, x_hat, 0)  # ReLU

    # Store
    tl.store(out_ptr + oc * H * W, out)

def fused_conv_bn_act(x, weight, bias, mean, var, gamma, beta):
    """Fused Conv + BatchNorm + ReLU."""
    N, C, H, W = x.shape
    OC, _, KH, KW = weight.shape
    out = torch.empty((N, OC, H - KH + 1, W - KW + 1), device=x.device, dtype=x.dtype)
    grid = (OC,)
    fused_conv_bn_act_kernel[grid](x, weight, bias, mean, var, gamma, beta, out,
                                   N, C, H, W, OC,
                                   x.stride(0), x.stride(2),
                                   KH, KW, 1e-5, 128)
    return out
```

#### Attention Pattern (QK^T + softmax + PV)
```python
@triton.jit
def attention_kernel(q_ptr, k_ptr, v_ptr, out_ptr,
                     B, H, N_CTX, HEAD_DIM,
                     scale: tl.constexpr,
                     BLOCK_M: tl.constexpr,
                     BLOCK_N: tl.constexpr):
    # ... implementation of fused attention ...
    # Key steps:
    # 1. q * k^T (tl.dot)
    # 2. softmax over N dimension (online algorithm)
    # 3. (softmax) * v (tl.dot)
    pass
```

## Step 4: Verification and Refinement

### Code Validation Checklist

Before submitting, verify:

- [ ] **No PyTorch compute in kernel**: Search for forbidden patterns
- [ ] **All computation in Triton ops**: `tl.load`, `tl.store`, `tl.dot`, `tl.reduce`
- [ ] **Proper masking**: All loads/stores have boundary masks
- [ ] **Correct grid calculation**: Use `triton.cdiv()` for ceiling division
- [ ] **Float32 accumulation**: Accumulate in `tl.float32`, cast on store
- [ ] **Valid dtype handling**: Handle `float16`, `bfloat16` correctly

### Common Error Patterns

**"AttributeError: module 'triton.language' has no attribute 'X'"**
- Check Triton version compatibility
- Use alternative operations if needed

**"Masked operations have mismatched shapes"**
- Ensure all offsets and masks have compatible shapes
- Use broadcasting correctly: `mask_m[:, None] & mask_n[None, :]`

**"Out of bounds memory access"**
- Verify all masks: `offsets < dimension_size`
- Check stride calculations

**Numerical accuracy issues**
- Accumulate in `float32`, cast at the end
- Use appropriate tolerance in tests

### Refinement Prompt Template

When fixing errors:
```
TEST CODE:
[test code]

CURRENT KERNEL:
[broken kernel code]

ERROR:
[stderr/stdout from test run]

FIX: Analyze the error and provide a corrected kernel implementation.
```

## Fusion Patterns Reference

### Always Fuse
- **GEMM + bias + activation**: `y = matmul(x, w) + bias; y = relu(y)`
- **QK^T + softmax + PV**: Attention patterns
- **Conv + BN + activation**: Common in vision models

### Fuse When Possible
- Multiple element-wise ops: `y = relu(x * w + b) + skip`
- Normalization with reshape: LayerNorm after GEMM

### Keep Separate
- Operations with different tile requirements
- Ops that would exceed register/shared memory limits

## Performance Optimization Guidelines

### Block Size Selection
- **Element-wise**: 1024, 2048
- **GEMM**: 64x64, 128x128, 128x64 + 32 K
- **Attention**: BLOCK_M=64/128, BLOCK_N=32/64/128

### Autotune for Complex Kernels
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=3, num_warps=8),
    ],
    key=['N_CTX', 'HEAD_DIM']
)
@triton.jit
def attention_kernel(...):
    ...
```

### Memory Access Optimization
- **Coalesced access**: Access contiguous memory along innermost stride
- **Use `tl.max_contiguous` + `tl.multiple_of`**: Help compiler optimization

## Quick Reference

| Operation | Triton Op | Notes |
|-----------|-----------|-------|
| Load | `tl.load(ptr + offset, mask=...)` | Always use mask |
| Store | `tl.store(ptr + offset, value, mask=...)` | Always use mask |
| Element-wise | `tl.add`, `tl.mul`, `tl.sub`, `tl.div` | Element-wise ops |
| Matmul | `tl.dot(a, b, acc)` | Accumulate in float32 |
| Reduce | `tl.reduce(data, axis, combine_fn)` | `axis=0` for row |
| Max | `tl.max(x, axis)` | Or `tl.reduce` with max |
| Softmax | Online algorithm with `exp` + `sum` | Numerically stable |
| Broadcast | `x[:, None]` + `y[None, :]` | Manual broadcasting |

## Example: Complete Workflow

**Input PyTorch:**
```python
class Model(torch.nn.Module):
    def forward(self, x, weight):
        return torch.nn.functional.relu(torch.nn.functional.linear(x, weight))
```

**Test:**
```python
def get_inputs():
    x = torch.randn(128, 512, device='cuda', dtype=torch.float32)
    weight = torch.randn(256, 512, device='cuda', dtype=torch.float32)
    return [x, weight]

def reference_forward(x, weight):
    return torch.nn.functional.linear(x, weight)  # raw matmul
    # User's model applies relu separately
    # return torch.nn.functional.relu(torch.nn.functional.linear(x, weight))

def test_kernel():
    inputs = get_inputs()
    x, weight = inputs
    ref = torch.nn.functional.relu(torch.nn.functional.linear(x, weight))
    out = kernel_function(x, weight)
    assert torch.allclose(out, ref, rtol=1e-3)
    print("PASS")
```

**Generated Kernel:**
```python
import triton
import triton.language as tl
import torch

@triton.jit
def _linear_relu_kernel(x_ptr, weight_ptr, bias_ptr, out_ptr,
                         M, N, K,
                         stride_xm, stride_xk,
                         stride_wk, stride_wn,
                         stride_om, stride_on,
                         BLOCK_SIZE_M: tl.constexpr,
                         BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    mask_m = offs_m < M
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_SIZE_M, N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm +
                         (k + offs_k)[None, :] * stride_xk)
        w_ptrs = weight_ptr + ((k + offs_k)[:, None] * stride_wk +
                               tl.arange(0, BLOCK_SIZE_M)[None, :] * stride_wn)

        x = tl.load(x_ptrs, mask=(mask_m[:, None] &
                  ((k + offs_k)[None, :] < K)), other=0.0)
        w = tl.load(w_ptrs, mask=((k + offs_k)[:, None] < K) &
                  tl.arange(0, BLOCK_SIZE_M)[None, :] < N, other=0.0)
        acc = tl.dot(x, w, acc)

    # Apply bias and ReLU
    for n in tl.arange(0, N):
        bias_val = tl.load(bias_ptr + n)
        val = acc[n] + bias_val
        val = tl.where(val > 0, val, 0)  # ReLU
        tl.store(out_ptr + offs_m * stride_om + n, val,
                mask=mask_m & (n < N))

def kernel_function(x, weight, bias=None):
    """Fused Linear + ReLU kernel."""
    M, K = x.shape
    K, N = weight.shape

    if bias is None:
        bias = torch.zeros(N, device=x.device, dtype=x.dtype)

    output = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(M, 128),)
    _linear_relu_kernel[grid](x, weight, bias, output,
                              M, N, K,
                              x.stride(0), x.stride(1),
                              weight.stride(0), weight.stride(1),
                              output.stride(0), output.stride(1),
                              BLOCK_SIZE_M=128, BLOCK_SIZE_K=32)
    return output
```

## Final Checklist Before Output

When presenting the kernel to the user:

1. ✅ Test code validates correctness
2. ✅ No PyTorch compute in kernel
3. ✅ All Triton ops for computation
4. ✅ Proper masks on all loads/stores
5. ✅ Float32 accumulation with dtype cast
6. ✅ Fusion documented (if applicable)
7. ✅ Performance considerations mentioned
8. ✅ Grid calculation uses `triton.cdiv()`

**Remember:** The goal is correct, performant Triton code that passes tests while being readable and maintainable.

---

## Advanced: Complex Model Pipeline (Subgraph Extraction & Composition)

For complex models with multiple stages, follow this pipeline:

### Phase 1: Subgraph Extraction

Analyze the model and identify fusable operation chains:

```python
# Example: ResNet block
class Bottleneck(nn.Module):
    def forward(self, x):
        shortcut = x
        
        # Subgraph 1: Conv + BN + ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Subgraph 2: Conv + BN (no activation)
        x = self.conv2(x)
        x = self.bn2(x)
        
        # Residual connection
        x = x + shortcut
        x = F.relu(x)
        return x
```

**Subgraph Analysis Template:**

| Subgraph ID | Operations | Input Shape | Output Shape | Weights Fused |
|-------------|-----------|-------------|--------------|---------------|
| sg_1 | conv2d → batch_norm → relu | [B, 64, 56, 56] | [B, 64, 56, 56] | conv.weight, bn.weight, bn.bias |
| sg_2 | conv2d → batch_norm | [B, 64, 56, 56] | [B, 64, 56, 56] | conv.weight, bn.weight, bn.bias |

**Key Questions:**
- What operations can be fused together?
- What are the exact input/output tensor shapes?
- Which weights are fused into the kernel?
- What's the data layout (NCHW, NHWC, etc.)?

### Phase 2: Generate Per-Subgraph Kernels

For each subgraph, generate an independent Triton kernel:

```python
# Subgraph 1: Fused Conv + BN + ReLU
def kernel_function_conv_bn_relu(x, conv_weight, bn_mean, bn_var, bn_gamma, bn_beta):
    """
    Fused Convolution + BatchNorm + ReLU.
    
    Args:
        x: Input tensor [B, C_in, H, W]
        conv_weight: Convolution weights [C_out, C_in, KH, KW]
        bn_mean: BatchNorm running mean [C_out]
        bn_var: BatchNorm running var [C_out]
        bn_gamma: BatchNorm scale [C_out]
        bn_beta: BatchNorm bias [C_out]
    
    Returns:
        Output tensor [B, C_out, H_out, W_out]
    """
    # Implementation using tl.load, tl.dot, tl.store
    pass
```

**Subgraph Generation Rules:**
1. Each kernel has its own `kernel_function`
2. Accept all weights/biases as parameters
3. Validate shapes against the subgraph spec
4. Return the computed output tensor
5. Include self-test that matches reference PyTorch

### Phase 3: Compose End-to-End

Combine verified subgraph kernels into the complete program:

```python
# Complete model using verified subgraph kernels
import triton
import triton.language as tl

# === Subgraph 1: Conv + BN + ReLU ===
@triton.jit
def conv_bn_relu_kernel(...):
    ...

def subgraph_conv_bn_relu(x, conv_w, bn_params):
    """Wrapper for subgraph 1."""
    ...

# === Subgraph 2: Conv + BN ===
@triton.jit
def conv_bn_kernel(...):
    ...

def subgraph_conv_bn(x, conv_w, bn_params):
    """Wrapper for subgraph 2."""
    ...

# === End-to-End Model ===
def kernel_function(x, conv1_weight, bn1_params, conv2_weight, bn2_params):
    """
    Complete ResNet bottleneck block.
    
    This composes sg_1 (conv1 + bn1 + relu) and sg_2 (conv2 + bn2),
    then adds the residual connection and final ReLU.
    """
    # Stage 1: conv1 + bn1 + relu
    shortcut = x
    x = subgraph_conv_bn_relu(x, conv1_weight, bn1_params)
    
    # Stage 2: conv2 + bn2
    x = subgraph_conv_bn(x, conv2_weight, bn2_params)
    
    # Residual add
    assert x.shape == shortcut.shape
    # Note: Residual add should be fused into one of the kernels
    # For simplicity, implement as separate operation here
    
    # Final ReLU
    x = torch.relu(x + shortcut)
    return x
```

### Subgraph Composition Strategies

**Strategy 1: Sequential Composition**
- Simple chain of subgraphs
- Output of one feeds into next
- Example: conv → bn → relu → conv → bn → add → relu

**Strategy 2: Parallel Branches with Merge**
- Multiple subgraphs compute in parallel
- Results merged (add, concat, etc.)
- Example: ResNet residual blocks

**Strategy 3: In-Place Operations**
- Modify tensors without allocation
- Reduce memory pressure
- Requires careful dependency tracking

**Strategy 4: Fused Residual**
- Combine residual computation into adjacent kernel
- Avoid separate add operation
- Better memory bandwidth utilization

### Common Subgraph Patterns

**Vision Models (ResNet, etc.):**
```
sg_conv_bn_relu:  conv2d → batch_norm → relu
sg_conv_bn:       conv2d → batch_norm
sg_avg_pool:      adaptive_avg_pool2d
sg_add:           elementwise add
```

**Transformers (BERT, etc.):**
```
sg_qkv_proj:      linear (Q) + linear (K) + linear (V)
sg_attention:     qk^T → softmax → pv
sg_mlp:           linear → gelu → linear
sg_layer_norm:    layernorm computation
```

**Detection Models (YOLO, etc.):**
```
sg_conv_relu:     conv2d → relu
sg_upsample:      interpolate + concat
sg_detection_head: multiple convs + sigmoid
```

### Shape Signature for Deduplication

When extracting subgraphs, use shape signatures to avoid duplicate work:

```python
def compute_shape_signature(subgraph):
    """Generate unique signature based on shapes."""
    return json.dumps({
        "input_shape": subgraph.input_shape,
        "output_shape": subgraph.output_shape,
        "weight_shapes": {k: v.shape for k, v in subgraph.weights.items()},
        "dtype": str(subgraph.dtype),
        "layout": subgraph.data_layout,
    }, sort_keys=True)

# Same shape = same kernel
# Different shapes = generate separate kernel
```

### Error Handling in Composition

When composing fails:

1. **Shape Mismatch**: Check subgraph input/output contracts
2. **Memory Allocation**: Verify intermediate tensor sizes
3. **Device Consistency**: Ensure all tensors on same device
4. **Dtype Compatibility**: Check precision requirements

### Performance Considerations

1. **Kernel Count**: More subgraphs = more kernel launches
2. **Memory Bandwidth**: Fused kernels reuse data better
3. **Register Pressure**: Larger fused kernels may hit limits
4. **Autotune**: Let Triton find optimal block sizes per subgraph

---

## Complete Complex Example: Vision Transformer Block

**Input Model:**
```python
class TransformerBlock(nn.Module):
    def forward(self, x, attn_qkv, attn_proj, mlp_fc1, mlp_fc2):
        # Self-attention
        qkv = F.linear(x, attn_qkv)
        q, k, v = qkv.split(attn_qkv.shape[0] // 3, dim=-1)
        attn = F.scaled_dot_product_attention(q, k, v)
        x = F.linear(attn, attn_proj)
        
        # MLP
        x = x + F.gelu(F.linear(x, mlp_fc1))
        x = x + F.linear(x, mlp_fc2)
        return x
```

**Subgraph Extraction:**
```
sg_attention:  linear (qkv) → split → scaled_dot_product_attention → linear (proj)
sg_mlp:        linear → gelu → linear
sg_add:        residual adds
```

**Generated Kernels:**
1. `kernel_attention(x, qkv_weight, proj_weight)` → fused QK attention
2. `kernel_mlp(x, fc1_weight, fc2_weight)` → fused MLP
3. `kernel_add(x, residual)` → elementwise add

**Composed Kernel:**
```python
def kernel_function(x, attn_qkv, attn_proj, mlp_fc1, mlp_fc2):
    """Complete ViT block."""
    # Attention subgraph
    attn_out = kernel_attention(x, attn_qkv, attn_proj)
    x = x + attn_out
    
    # MLP subgraph
    x = x + kernel_mlp(x, mlp_fc1, mlp_fc2)
    return x
```

---

## Summary: When to Use Each Mode

| Scenario | Mode | Rationale |
|----------|------|-----------|
| Element-wise ops | A | Simple, single kernel |
| Single GEMM | A | Well-understood pattern |
| Conv + BN + Act | A | Standard fusion pattern |
| Complex model | B | Multiple stages need decomposition |
| Attention block | A or B | Can be single fused kernel |
| Residual networks | B | Parallel branches, merge points |
| UNet encoder/decoder | B | Skip connections, multiple paths |

**Key Principle:** Start with the simplest approach that works. Only decompose into subgraphs when necessary for correctness or when fusion becomes too complex.
