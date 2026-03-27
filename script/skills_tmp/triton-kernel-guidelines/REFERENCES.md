# Triton Kernel References and Patterns

This file collects **concrete code patterns** and **realistic examples** you can adapt when implementing kernels under `triton_kernel_guidelines`.

Do not blindly copy these into user answers unless requested. Instead:
- Adapt the patterns to the user’s shapes, dtypes, and constraints.
- Keep comments and structure, but simplify when possible.

---

## 1. Persistent GEMM Pattern (Conceptual Sketch)

Key concepts:
- `_compute_pid(tile_id, ...)` to map tile ids to `(pid_m, pid_n)`.
- Persistent loop over tiles with `tl.range(start_pid, num_tiles, NUM_SMS, flatten=True)`.
- Separate `a` and `b` tile loads, each with masks to handle K-tail.
- Epilogue that writes the accumulator to `c_ptr` with proper dtype conversion (`float8e4nv` vs `float16`) and masks.

Pseudocode (heavily simplified):

```python
@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n

@triton.autotune(configs=..., key=["M", "N", "K"])
@triton.jit
def matmul_kernel_persistent(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        # compute offs_am, offs_bn, apply tl.max_contiguous / tl.multiple_of
        # ...
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            # compute offs_k, a_ptrs, b_ptrs
            # a = tl.load(a_ptrs, mask=..., other=0.0)
            # b = tl.load(b_ptrs, mask=..., other=0.0)
            # accumulator = tl.dot(a, b, accumulator)
            ...
        # compute final pid for writeback (if needed), offs_cm, offs_cn
        # c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        # c = accumulator.to(output_dtype)
        # tl.store(c_ptrs, c, mask=c_mask)
```

Wrapper sketch:

```python
def matmul_persistent(a, b):
    assert a.shape[1] == b.shape[0]
    assert a.dtype == b.dtype
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (
        min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )
    matmul_kernel_persistent[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        NUM_SMS=NUM_SMS,
    )
    return c
```

---

## 2. Fused Attention Forward Pattern (Conceptual Sketch)

Pattern:
- `_attn_fwd_inner`: QK, online softmax, PV per range of keys/values.
- `_attn_fwd`: descriptor setup, grid, STAGE dispatch.
- `_attention` (autograd Function): PyTorch wrapper that:
  - Allocates output and workspace (`M`).
  - Builds TensorDescriptors for `q`, `k`, `v`, `o`.
  - Launches `_attn_fwd` via `@triton.autotune`.

Key ideas:

```python
@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    desc_k, desc_v,
    offset_y, dtype: tl.constexpr, start_m, qk_scale,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
    N_CTX: tl.constexpr, warp_specialize: tl.constexpr,
):
    # determine lo/hi range for this stage
    # initialize offsetk_y, offsetv_y based on STAGE and dtype
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        # k = desc_k.load([...])
        # qk = tl.dot(q, k)
        # update m_ij, qk normalization based on STAGE
        # p = tl.math.exp2(qk)
        # alpha = tl.math.exp2(m_i - m_ij)
        # l_ij = tl.sum(p, 1)
        # v = desc_v.load([...])
        # p = p.to(dtype)
        # acc = acc * alpha[:, None]
        # acc = tl.dot(p, v, acc)
        # l_i = l_i * alpha + l_ij
        # m_i = m_ij
        ...
    return acc, l_i, m_i
```

Descriptor and wrapper pattern:

```python
@triton.autotune(configs=..., key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"])
@triton.jit
def _attn_fwd(
    sm_scale, M,
    Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    STAGE: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    # build descriptors from desc_q/k/v/o if needed
    # compute offsets, initialize m_i, l_i, acc, qk_scale
    # load q tile
    # call _attn_fwd_inner for stage 1 and/or stage 2
    # finalize softmax: m_i += tl.math.log2(l_i); acc /= l_i[:, None]
    # write m_i, and acc converted to dtype
    ...
```

Higher-level wrapper:

```python
class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, warp_specialize=True):
        HEAD_DIM_Q = q.shape[-1]
        assert HEAD_DIM_Q == k.shape[-1] == v.shape[-1]
        assert HEAD_DIM_Q in {16, 32, 64, 128, 256}

        o = torch.empty_like(q)
        stage = 3 if causal else 1
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        # build TensorDescriptors for q, k, v, o
        # define grid() based on BLOCK_M and sequence length

        _attn_fwd[grid](
            sm_scale, M,
            q.shape[0], q.shape[1],
            desc_q, desc_k, desc_v, desc_o,
            N_CTX=q.shape[2],
            HEAD_DIM=HEAD_DIM_Q,
            FP8_OUTPUT=(q.dtype == torch.float8_e5m2),
            STAGE=stage,
            warp_specialize=warp_specialize,
        )

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_Q
        ctx.causal = causal
        return o

attention = _attention.apply
```

User-facing API pattern:

```python
def fused_attention(q, k, v, causal=True, sm_scale=None):
    if sm_scale is None:
        sm_scale = 1.0 / (q.shape[-1] ** 0.5)
    return attention(q, k, v, causal, sm_scale)
```

---

## 3. BN-in-Conv and LayerNorm Fusion Snippets

BatchNorm inside fused conv (per-output-channel):

```python
bn_mean_f32 = tl.load(mean_ptr + oc).to(tl.float32)
bn_var_f32 = tl.load(var_ptr + oc).to(tl.float32)
gamma_f32  = tl.load(gamma_ptr + oc).to(tl.float32)
beta_f32   = tl.load(beta_ptr + oc).to(tl.float32)
acc_f32    = acc.to(tl.float32)
x_norm     = (acc_f32 - bn_mean_f32) / tl.sqrt(bn_var_f32 + eps)
out_f32    = x_norm * gamma_f32 + beta_f32
tl.store(out_ptr + out_idx, out_f32.to(out_ptr.dtype.element_ty), mask=mask)
```

Per-row LayerNorm (pattern):

```python
mean_f32 = tl.zeros((), dtype=tl.float32)
var_f32  = tl.zeros((), dtype=tl.float32)
# accumulate in fp32 across the row
# use tl.sqrt(var_f32 + eps)
# cast back to output dtype on store
```

Use these snippets inside larger kernels whenever fusing BN/LayerNorm.

---

## 4. Checklist for New Kernels

When implementing a new kernel using these references:

1. Map user operation → one of the patterns above (GEMM/attention/elementwise/fused).
2. Decide on tiling and BLOCK sizes.
3. Design indexing and grid.
4. Add masks for all potential OOB accesses.
5. Verify dtype conversions and accumulation precision.
6. Implement a thin wrapper (no PyTorch compute; only allocation + launch).
7. Document:
   - Fused ops
   - Important numerical stability tricks
   - Any constraints on shapes/dtypes/layouts.
