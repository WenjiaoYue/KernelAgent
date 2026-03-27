---
name: triton_kernel_guidelines
description: Triton kernel programming guidelines and best practices for generating high-quality Triton kernels and wrappers.
metadata:
  {
    "openclaw":
      {
        "emoji": "🧮",
        "homepage": "https://github.com/meta-pytorch/KernelAgent",
        "skillKey": "triton-kernel-guidelines",
        "requires": { "bins": [], "env": [], "config": [] }
      }
  }
---

# Triton Kernel Guidelines

Use this skill whenever the user wants Triton kernels or Triton-based implementations instead of plain PyTorch/TorchScript.

Your goal is to **produce correct, performant, and reviewable Triton kernels and Python wrappers** that follow these rules.

---

## 1. Kernel Structure

- Kernel functions **must** use `@triton.jit`.
- Name kernels based on their function:
  - Examples: `kernel_function`, `matmul_kernel`, `fused_attention_kernel`.
- Use `tl.constexpr` for compile-time constants:
  - `BLOCK_SIZE`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, `GROUP_SIZE_M`, `NUM_WARPS`, `NUM_STAGES`, etc.
- Add launch metadata when helpful (e.g. for debugging / profiling).
- For each kernel, provide a Python wrapper that:
  - Validates shape, dtype, device, layout (e.g. contiguous, expected rank).
  - Allocates outputs / workspaces.
  - Computes a `grid` lambda using `triton.cdiv`.
  - Launches the kernel via `kernel[grid](...)`.

Do **not** implement math or algorithmic compute in wrappers; keep all compute inside Triton kernels.

---

## 2. Memory Access Patterns

- All loads and stores use `tl.load` / `tl.store` with proper masking.
- Aim for **coalesced** memory access:
  - Access contiguous elements along the innermost stride of row-major tensors.
  - Use `tl.max_contiguous` and `tl.multiple_of` to expose alignment and contiguity to the compiler.
- Always handle boundary conditions with masks, especially for:
  - Tail tiles in M/N dimensions.
  - Final partial tile along K.
- When using advanced memory operations (TensorDescriptor / TMA):
  - Configure `shape`, `strides`, and `block_shape` correctly.
  - Document how the descriptor maps logical indices to memory.

---

## 3. Indexing and Grid Design

- Use `tl.program_id(axis)` to index tiles/blocks:
  - 1D: `pid = tl.program_id(0)`
  - 2D: `pid_m = tl.program_id(0)`, `pid_n = tl.program_id(1)`
  - 3D if needed (e.g. batch/head dimension).
- Typical offset patterns:
  - Elementwise: `offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`
  - Tile-based:
    - `start_m = pid_m * BLOCK_SIZE_M`
    - `start_n = pid_n * BLOCK_SIZE_N`
    - `offs_m = start_m + tl.arange(0, BLOCK_SIZE_M)`
    - `offs_n = start_n + tl.arange(0, BLOCK_SIZE_N)`
- Use `tl.cdiv` for tile counts:
  - `num_tiles_m = tl.cdiv(M, BLOCK_SIZE_M)`
  - `num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)`
- **Masks are mandatory** when offsets may go out of bounds:
  - 1D: `mask = offsets < N`
  - 2D: `mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)`
- For persistent kernels:
  - Use `tl.range(start_pid, num_tiles, NUM_SMS, flatten=True)` to iterate tiles.
  - Map linear tile id to `(pid_m, pid_n)` using a helper function.

---

## 4. Optimization Techniques

- Use `@triton.autotune` for non-trivial kernels:
  - Provide several configs varying `BLOCK_*/GROUP_SIZE_M/num_warps/num_stages`.
  - Specify selector keys that match problem dimensions (e.g. `"M", "N", "K"`, `"N_CTX", "HEAD_DIM"`).
- Choose block sizes as powers of two:
  - Typical choices: `64, 128, 256, 512, 1024`.
- Use `tl.dot` for matrix multiplications and attention inner products:
  - Accumulate in `tl.float32` where possible.
  - Convert to output dtype in the epilogue.
- Use warp specialization when beneficial:
  - Parameterize as `warp_specialize: tl.constexpr`.
  - Use `tl.range(..., warp_specialize=warp_specialize)` inside loops.
- Use multi-stage pipelines (`num_stages` / `STAGE` flags) when:
  - There is clear reuse or split between off-band/on-band work.
  - Pipelining improves latency or bandwidth usage.
- Apply **epilogue subtiling** to:
  - Reduce shared memory pressure.
  - Allow larger tiles while keeping resource usage manageable.

---

## 5. Fusion Strategy

- Aggressively fuse compatible operator stages into a single kernel when:
  - They share the same shape / layout / tiling scheme.
  - They are bandwidth-bound or latency-limited by kernel launches.
- Common fusion targets:
  - Pointwise ops after GEMM (e.g. bias, activation).
  - Conv + BatchNorm (+ Activation).
  - Attention: `qk` + softmax + `pv`.
- Keep stages separate only when:
  - Fusion would exceed resource limits (registers / shared mem).
  - Launch patterns or shapes are fundamentally incompatible.
- For each fused kernel, clearly document:
  - Which ops are fused.
  - How numerical stability is preserved (if relevant, e.g. softmax, LayerNorm, BatchNorm).

---

## 6. Common Patterns (High-Level)

When planning kernels, map the user’s request to one of these templates:

1. **Elementwise**: `Load → Compute → Store`
2. **Reduction**: reduce along one or more axes using tiles, `tl.sum`, `tl.max`, or `tl.reduce`.
3. **GEMM-like**: tile-based multiplication using `BLOCK_M/N/K` and `tl.dot`.
4. **Softmax / Normalization**: use online normalization for numeric stability.
5. **Fused Blocks**:
   - Conv + BatchNorm (+ Activation)
   - Attention (QK, softmax, PV)
   - MLP blocks (GEMM + activation + dropout, etc.)

Reference implementations and patterns live in `{baseDir}/REFERENCES.md`.

---

## 7. Advanced Features

- **Persistent kernels**:
  - Keep SMs busy by assigning multiple tiles per block in a loop.
  - Carefully manage indices and masks to avoid double-processing tiles.
- **TensorDescriptor / TMA**:
  - Use descriptors for complex layouts, FP8 storage, and advanced streaming patterns.
- **Multi-stage pipelines**:
  - Split into STAGE 1 (off-band), STAGE 2 (on-band), etc., especially in attention-like workloads.
- **Warp specialization**:
  - Consider for latency-sensitive and bandwidth-heavy loops (e.g. attention).

---

## 8. Runtime Constraints (Python Wrappers)

Wrappers must obey the following constraints:

- Wrappers **are allowed to**:
  - Validate arguments (shape, dtype, device, layout).
  - Allocate outputs and any necessary temporary tensors.
  - Compute launch parameters (`grid`, meta-parameters).
  - Invoke Triton kernels.
- Wrappers **must NOT**:
  - Use `torch.nn` or `torch.nn.functional` (`F.*`) for compute.
  - Call `torch.matmul`, `mm`, `bmm`, `einsum`, or any heavy tensor–tensor compute.
  - Implement non-trivial elementwise math in Python.
- PyTorch is limited to:
  - Determining shapes, dtypes, and devices.
  - Allocating tensors.
  - Simple reshapes / views for packaging results.

If the user requests operations that would normally use PyTorch for compute, move that compute into a Triton kernel instead.

---

## 9. How to Use These Guidelines

When asked to implement or optimize a Triton kernel:

1. Clarify shapes, dtypes, and layout.
2. Select an appropriate pattern (elementwise, reduction, GEMM, attention, fusion).
3. Design indexing and grid using the rules above.
4. Choose reasonable block sizes and, when appropriate, provide autotune configs.
5. Implement:
   - A `@triton.jit` kernel (possibly with `@triton.autotune`).
   - A Python wrapper that only validates, allocates, and launches.
6. Explain briefly:
   - Indexing and masks.
   - Memory access patterns.
   - Any fusion and optimizations applied.

For concrete code patterns, see `{baseDir}/REFERENCES.md`.
