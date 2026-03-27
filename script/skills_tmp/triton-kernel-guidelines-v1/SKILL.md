---
name: triton_kernel_guidelines
description: Triton kernel programming guidelines, patterns, and real-world examples for generating high-quality Triton kernels.
metadata:
  {
    "openclaw":
      {
        "emoji": "🧮",
        "homepage": "https://github.com/meta-pytorch/KernelAgent",
        "primaryEnv": "dev",
        "skillKey": "triton-kernel-guidelines",
        "requires": { "bins": [], "env": [], "config": [] }
      }
  }
---

# Triton Kernel Programming Guidelines

Use this skill whenever the user wants you to:

- Write or modify a Triton kernel (`@triton.jit`)
- Design a Triton kernel wrapper (Python-side allocation + validation + launch)
- Optimize an existing Triton kernel (memory access, block sizes, autotune, etc.)
- Migrate / fuse PyTorch operators into Triton kernels

Your goal is to **strictly follow these guidelines and produce practical, well-structured, and reasonably optimized Triton kernels and wrappers**.

---

## 1. Kernel Structure

When designing or generating a Triton kernel:

- Kernel functions must use the `@triton.jit` decorator.
- Name the function according to the task (e.g. `kernel_function`, `matmul_kernel`, `fused_attention_kernel`) and keep naming consistent within a given task.
- Use `tl.constexpr` for compile-time constants, e.g.:
  - `BLOCK_SIZE`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`
  - `GROUP_SIZE_M`, `NUM_WARPS`, `NUM_STAGES`, etc.
- Add launch metadata where appropriate (e.g. `launch_metadata=_matmul_launch_metadata`) to help debugging and profiling.
- The Python wrapper is responsible for:
  - Shape / dtype / device / contiguity validation
  - Grid computation (usually with a lambda + `triton.cdiv`)
  - Allocating output tensors
  - Launching the kernel via `kernel[grid](...)`
- **Do not** perform mathematical compute in the wrapper. All compute must live inside Triton kernels (see “Runtime Constraints” below).

---

## 2. Memory Access Patterns

When generating or refactoring kernels, prioritize efficient and correct memory access:

- Use `tl.load` / `tl.store` with proper `mask`:
  - Always avoid out-of-bounds accesses, especially on tail tiles.
- Aim for coalesced memory access:
  - For row-major tensors, access contiguous elements along the innermost stride (typically the column dimension).
  - Use `tl.max_contiguous` and `tl.multiple_of` to help Triton infer alignment and contiguity.
- Advanced memory operations:
  - When needed, use TensorDescriptor / Tensor Memory Accelerator (TMA) descriptors for complex layouts and advanced access patterns.
  - When using TensorDescriptor, set `shape` / `strides` / `block_shape` correctly.
- Boundary handling:
  - Use masks to clamp `offs` to valid ranges.
  - For K loops, use separate `offs_k_for_mask` and `tl.load(..., mask=..., other=0.0)` to handle the last partial tile safely.

---

## 3. Indexing and Grid Design

When designing indexing and grid layout for a new kernel:

- Use `tl.program_id(axis)` to obtain block/tile indices:
  - 1D kernels: `pid = tl.program_id(axis=0)`
  - 2D/3D kernels: `pid_m = tl.program_id(0)`, `pid_n = tl.program_id(1)`, etc.
- Basic offset calculations:
  - Elementwise: `offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`
  - Tile-based: `start_m = pid_m * BLOCK_SIZE_M`, etc., plus `tl.arange`.
- Ceiling division:
  - Use `tl.cdiv` to compute tile counts, e.g. `num_tiles = tl.cdiv(M, BLOCK_SIZE_M)`.
- **Always use masks**:
  - Whenever `offs` can be out-of-bounds, construct a mask and pass it into loads/stores.
  - Example: `mask = offs < N`, or for 2D: `(offs_m[:, None] < M) & (offs_n[None, :] < N)`.
- Persistent kernels / multi-tile assignment:
  - You can iterate tiles in a loop using `tl.range(start_pid, num_tiles, NUM_SMS, flatten=True)` to implement persistent mapping.
  - Use helper functions (e.g. `_compute_pid`) to map a linear tile id to `(pid_m, pid_n)` consistently.

---

## 4. Optimization Techniques

When the user cares about performance, actively propose and implement reasonable optimizations:

- Use `@triton.autotune`:
  - Provide a set of configs (`BLOCK_M/N/K`, `num_warps`, `num_stages`, `GROUP_SIZE_M`, etc.).
  - Use a `key=[...]` list that includes the main problem dimensions (e.g. `"M", "N", "K"` or `"N_CTX", "HEAD_DIM"`).
- Block sizes:
  - Prefer powers of two: `64, 128, 256, 512, 1024`.
  - For GEMM/attention, choose sizes that fit the head dimension / tile shapes well.
- Tensor Cores:
  - Use `tl.dot` for matrix multiplications and attention `qk`/`pv` inner products.
  - Accumulate in `tl.float32` where possible.
- Warp specialization:
  - Expose `warp_specialize: tl.constexpr` and use it with `tl.range(..., warp_specialize=warp_specialize)` when appropriate.
- Pipeline staging:
  - Use `num_stages` or an explicit `STAGE` flag to define multi-stage pipelines (e.g. off-band vs on-band regions).
- Epilogue subtiling:
  - Sub-tile the epilogue/write-back stage to reduce shared memory usage for large tiles or multi-stage pipelines.
- **Aggressive fusion**:
  - Aggressively fuse compatible operator stages into a single kernel to reduce:
    - Global memory traffic
    - Kernel launch overhead
  - Only keep stages separate when fusion is clearly infeasible (e.g. due to resource limits, very different launch patterns, or required reuse).
  - For fused kernels, clearly document which ops/stages are fused (e.g. “conv + bn + relu” or “qk + softmax + pv”).

---

## 5. Common Patterns and Templates

When implementing functionality, prefer the following patterns and provide code templates as needed.

### 5.1 Elementwise: Load → Compute → Store

- Indexing: one- or two-dimensional offsets.
- Typical flow:
  1. `x = tl.load(x_ptr + offsets, mask=mask, other=0)`
  2. `y = f(x)`
  3. `tl.store(y_ptr + offsets, y, mask=mask)`

### 5.2 Reductions

- Use `tl.sum`, `tl.max`, or `tl.reduce` over the appropriate axis.
- Organize blocks so that each block covers the reduction dimension correctly (e.g. each block handles one row and reduces across columns).

### 5.3 Matrix Multiplication (GEMM)

- Use a tile-based GEMM:
  - `BLOCK_M`, `BLOCK_N`, `BLOCK_K`
  - 2D offsets `offs_am` / `offs_bn` plus a K-loop over tiles.
- Inner loop:
  - Use `tl.dot(a, b, accumulator)` with masking for the last K tile.
- Follow the persistent matmul example pattern:
  - Safe offsets with `tl.where` and masks
  - `tl.max_contiguous` / `tl.multiple_of` for coalesced loads
  - Final write-back with per-tile masking and dtype conversion.

### 5.4 Softmax (Numerically Stable)

- Use online normalization:
  1. Compute local `qk` and subtract `m_ij = max(qk, axis=1)` (or scaled variant).
  2. Use `exp2` + `log2` (`tl.math.exp2`, `tl.math.log2`) for softmax and normalization.
  3. Track `m_i` and `l_i` across tiles:
     - `alpha = exp2(m_prev - m_new)`
     - Scale accumulators by `alpha`.
- Refer to the fused attention forward example’s `_attn_fwd_inner` structure when implementing.

### 5.5 Fused Operations

When the user wants to fuse multiple ops into a single Triton kernel (e.g. Conv+BN+Activation):

- Implement the fused math directly in the kernel.
- For batch norm inside a fused conv, follow patterns like:

  ```python
  bn_mean_f32 = tl.load(mean_ptr + oc).to(tl.float32)
  bn_var_f32 = tl.load(var_ptr + oc).to(tl.float32)
  gamma_f32 = tl.load(gamma_ptr + oc).to(tl.float32)
  beta_f32  = tl.load(beta_ptr + oc).to(tl.float32)
  acc_f32   = acc.to(tl.float32)  # conv accumulator
  x_norm    = (acc_f32 - bn_mean_f32) / tl.sqrt(bn_var_f32 + eps)
  out_f32   = x_norm * gamma_f32 + beta_f32
  tl.store(out_ptr + out_idx, out_f32.to(out_ptr.dtype.element_ty), mask=mask)
  ```

- For per-row LayerNorm:

  ```python
  mean_f32 = tl.zeros((), dtype=tl.float32)
  var_f32  = tl.zeros((), dtype=tl.float32)
  # accumulate in fp32; use tl.sqrt(var_f32 + eps); cast on store
  ```

- Always clearly document in comments which stages are fused in this kernel.

---

## 6. Advanced Features

When complexity and performance targets justify it, you may consider:

- **Persistent kernels**:
  - Have each block process multiple tiles to improve SM utilization.
  - Follow patterns like the persistent matmul example:
    - Use `NUM_SMS`
    - Loop over tile ids via `tl.range`
    - Map tiles to `(pid_m, pid_n)` using a helper like `_compute_pid`.
- **Tensor Memory Accelerator (TMA) descriptors**:
  - Use for complex layouts or when you need advanced memory scheduling via TensorDescriptor/TMA.
- **Multi-stage pipelines**:
  - Use `num_stages` or explicit `STAGE` flags to define multiple passes (e.g. off-band vs on-band).
- **Warp specialization**:
  - Control with `warp_specialize: tl.constexpr` and use in `tl.range`.
  - Very useful in fused attention and other bandwidth-limited kernels.

---

## 7. Runtime Constraints (Python Wrapper Side)

You must strictly enforce the following constraints:

- Wrappers may only:
  - Validate inputs (shape, dtype, device, contiguity, layout)
  - Allocate outputs and any necessary buffers (`torch.empty`, `torch.zeros`, etc.)
  - Compute grid / launch parameters
  - Call Triton kernels via `kernel[grid](...)`
- Wrappers **must not** perform real compute:
  - No `torch.nn` or `torch.nn.functional` (e.g. `F.*`).
  - No `torch.matmul` / `mm` / `bmm` / `einsum` / or any general tensor–tensor math.
  - No non-trivial elementwise math in Python; such logic must live inside Triton kernels.
- PyTorch is allowed **only** for:
  - Allocation
  - Dtype/device checks
  - Packaging results (e.g. simple reshapes/views, not numerical transformations).

---

## 8. Real-World Examples (for Style and Structure)

When the user asks to “follow high-quality Triton kernel examples” or wants more complex patterns, you can adapt and simplify the patterns below (don’t blindly copy large blocks unless explicitly requested).

### 8.1 Persistent Matrix Multiplication (Pattern Reference)

Key ideas:

- Use a helper like `_compute_pid` to map a tile id to `(pid_m, pid_n)`.
- Use `@triton.autotune` with configs based on `M, N, K`.
- Inside the kernel:
  - Use `tl.cdiv` for tile counts
  - Use `tl.range(..., flatten=True)` for persistent tiling
  - Use `tl.max_contiguous` / `tl.multiple_of` for coalesced loads
  - Use `tl.dot(a, b, accumulator)` for GEMM
- In the epilogue:
  - Convert the accumulator to the desired dtype (`float8e4nv` or `float16`) and store with a mask.

When generating similar kernels, ensure that:

- All offsets and masks correctly match the target tensor shapes.
- Dtype branches are correct and numerically safe.

### 8.2 Fused Attention Forward (Pattern Reference)

Key ideas:

- Split attention into:
  - `_attn_fwd_inner`: core loop over K/V tiles, including qk, softmax, and pv.
  - `_attn_fwd`: descriptor construction, grid, and STAGE control.
  - `_attention` autograd Function: wrapper that builds descriptors, allocates workspace (e.g. `M`), and launches `_attn_fwd`.
- Use TensorDescriptor for q/k/v/o:
  - Correct `shape`, `strides`, and `block_shape` for each descriptor.
- Stage control:
  - Use `STAGE & 1` / `STAGE & 2` to enable off-band and on-band passes.
- Online softmax:
  - Scale with `qk_scale`, use `exp2`/`log2` (`tl.math.exp2`, `tl.math.log2`) for numerical stability.
  - Track and update `m_i`, `l_i`, and apply final normalization in the epilogue.

When generating fused attention or variants, follow this structure but adapt shapes, dtypes, and causal flags to the user’s requirements and clearly document the stages.

---

## 9. How to Use This Skill (For the Agent)

Whenever the user asks for Triton kernels:

1. Clarify input/output tensor shapes, dtypes, and layouts (e.g. NCHW vs NHWC).
2. Choose an appropriate pattern (elementwise, reduction, GEMM, attention, fused).
3. Design grid, tiling, `BLOCK_*` sizes, and masks.
4. Implement:
   - A Triton kernel with `@triton.jit` and optional `@triton.autotune`.
   - A Python wrapper that handles validation, allocation, and launch only.
5. In your answer, briefly explain:
   - Indexing and grid layout
   - Memory access pattern and masking
   - Boundary handling
   - Key optimizations (autotune, fusion, warp specialization, etc., if applicable)

Always ensure:

- All compute is inside Triton kernels.
- PyTorch is only used for allocation, checks, and packaging.
- Memory access and masks are correct and safe for all shapes and edge cases.
