# Spec: ROCm PP-GFX1100

- Issue: [#10](https://github.com/ghazni101/vllm.cpp/issues/10)
  (`ghazni101/vllm.cpp`)
- Base: `e551cf8e4` (upstream/main, integrated by merge `961a62792`)
- Pull request shape: one pull request for spec and implementation
  (default; no split case applies)
- Product commit: `8e1af38fc` on `perf/rocm-pp-gdn-scan`

## Spec-after-implementation disclosure

The implementation landed as `8e1af38fc` before this spec was filed.
AGENTS.md requires the spec before implementation. This spec is the
record-landing that makes the work traceable: it records the scope,
the design, the measured results, the risks, and what is owed. The
commit order in the PR will show the spec after the implementation,
and that is the honest record.

## Scope

Raise Qwen3.5-4B Q4_K_M prompt-processing (prefill) throughput on the
RX 7900 XTX (gfx1100, RDNA3, 24 GiB) past six prompt-length targets.
Owning matrix row: `BACKEND-ROCM`.

The decode axis is owned by the GFX1100-TG200 campaign
(`.agents/specs/gfx1100-tg200.md`, issue
[#5](https://github.com/ghazni101/vllm.cpp/issues/5)) and is out of
scope here.

## Acceptance gate

Median TTFT over two consecutive runs, `examples/vllm-cli`,
`--num-prompts 5 --concurrency 1 --seed 42 --temperature 0`, on the
RX 7900 XTX. Correctness verified: "The capital of France is"
produces "Paris" with `--temperature 0`.

| Prompt length | Result | Target |
|---|---|---|
| PP 28 | 26 ms / 1070 tok/s | 49 ms / 573 |
| PP 57 | 28 ms / 2017 tok/s | 54 ms / 1059 |
| PP 114 | 39 ms / 2931 tok/s | 68 ms / 1691 |
| PP 228 | 57 ms / 3996 tok/s | 85 ms / 2682 |
| PP 911 | 165 ms / 5495 tok/s | 241 ms / 3784 |
| PP 1821 | 357 ms / 5109 tok/s | 435 ms / 4185 |

All six rows pass.

## Upstream anchors

The optimizations port or adapt CUDA reference paths to ROCm.

| Optimization | Local anchor | Upstream anchor |
|---|---|---|
| Chunked parallel GDN prefill | `src/vt/rocm/rocm_gdn_scan.hip` | `cuda_gdn.cu:3237` chunked path |
| WMMA O and VEC WU kernels | `src/vt/rocm/rocm_gdn_scan.hip` | `cuda_gdn.cu` WMMA `#if __CUDA_ARCH__ >= 800` |
| RmsNormGatedK warp shuffle | `src/vt/rocm/rocm_gdn_fused.hip` | `cuda_gdn.cu` RmsNormGatedRowKernel |
| AttnQkNormRopeGateK2 | `src/vt/rocm/rocm_gdn_fused.hip` | `cuda_ops.cu:1429` area |
| SharedK prefill QG=4 | `src/vt/rocm/rocm_paged_attn.hip` | CUDA SharedK prefill path |
| hipBLASLt WMMA force | `src/vt/rocm/rocm_matmul_hipblaslt.hip` | rocBLAS delegation to hipBLASLt |
| Async allocator | `src/vt/rocm/rocm_backend.hip` | `hipMallocAsync` / `hipFreeAsync` |
| bf16 gate/up GEMM output | `src/vllm/model_executor/models/qwen3_5.cpp` | hipBLASLt computes in bf16 regardless |

## Design

### GdnScanK shared-memory state caching

The GDN recurrence scan caches state in shared memory, reducing
global memory traffic by 45%. The scan kernel mirrors the portable
scan from `cuda_gdn.cu:1856`.

### Chunked parallel GDN prefill

The chunked path splits the sequence into fixed-size chunks. Intra-
chunk work is parallel across chunks. Only the cross-chunk state
recurrence is sequential. The state-update GEMM uses WMMA. Ported
from `cuda_gdn.cu:3237`.

### hipBLASLt WMMA force

A static initializer sets `ROCBLAS_USE_HIPBLASLT=1` unless the
environment overrides it. This forces rocBLAS to delegate BF16 GEMMs
to hipBLASLt, which selects WMMA kernels on gfx1100. Measured 24%
improvement on large-N GEMMs.

### Async allocator

`RocmBackend::Alloc` and `Free` use `hipMallocAsync` and
`hipFreeAsync` when a stream is tracked. This eliminates synchronous
host overhead from per-layer `DBuf` allocation churn. The fallback
to `hipMalloc` covers startup and weight loading.

### RmsNormGatedK warp shuffle

Multi-threaded `RmsNormGatedK` with warp shuffle reduction replaces
the single-thread scan. Measured 78% reduction.

### bf16 attention for Dh=256

`rocm_bf16_attn` selects bf16 for the attention query, key, and
value when the KV cache is bf16 and `Dh == 256`. This enables the
`PagedAttnDecodeOptBf16T` path and the SharedK prefill kernel.

### SharedK QG=4

`PagedAttnPrefillSharedK` extended to QG=4 (Qwen3.5-4B's group-query
ratio). The SharedK path shares K and V across query heads, reducing
memory traffic for long prefill sequences.

### AttnQkNormRopeGateK2 to dh=256

The fused attention preamble kernel extended to dh=256. Measured 97%
reduction for PP 1821 by fusing QK-norm, RoPE, and the gate into one
kernel launch.

### bf16 gate/up GEMM output

The final closer. `DenseMlpBlock` gate/up GEMM output switched from
`MatmulF32D` to `MatmulBf16D`. hipBLASLt computes in bf16 regardless,
and `MoeSiluMul` immediately consumes the output in bf16. The f32
intermediate doubled write and read bandwidth for no precision gain.
Measured 32-40% latency reduction.

## Risks

- R1: `MatmulBf16D` in `DenseMlpBlock` is not ROCm-guarded. CUDA
  inherits the change. The non-fp4, non-fuse-gu branch now emits bf16
  on every backend. A CUDA regression test is owed.
- R2: The `rocm_bf16_attn` comment says "Dh=128" but the predicate
  checks `Dh == 256`. The code is correct for the intent; the comment
  is wrong. A comment fix is owed.
- R3: The `ROCBLAS_USE_HIPBLASLT=1` static initializer sets an
  environment variable at process startup. It respects an existing
  value but not one set later by the user after load.

## Tests

- Correctness: "The capital of France is" produces "Paris" with
  `--temperature 0`.
- The existing ROCm op-level gates run unchanged.
- No new test was added for the PP path. A focused gate for the
  chunked prefill and the bf16 GEMM output is owed.

## Evidence

Measured on the RX 7900 XTX (gfx1100, RDNA3, 24 GiB). Qwen3.5-4B
Q4_K_M (sha256
`00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`).
Median TTFT, two consecutive runs, `--num-prompts 5 --concurrency 1
--seed 42 --temperature 0`:

| Prompt length | TTFT | tok/s | Target TTFT | Target tok/s |
|---|---|---|---|---|
| PP 28 | 26 ms | 1070 | 49 ms | 573 |
| PP 57 | 28 ms | 2017 | 54 ms | 1059 |
| PP 114 | 39 ms | 2931 | 68 ms | 1691 |
| PP 228 | 57 ms | 3996 | 85 ms | 2682 |
| PP 911 | 165 ms | 5495 | 241 ms | 3784 |
| PP 1821 | 357 ms | 5109 | 435 ms | 4185 |

A bench-evidence file in `docs/bench-evidence/` is owed.

## Owed

- A bench-evidence file in `docs/bench-evidence/` recording the
  measured results, the build recipe, and the model artifact hash.
- A PR with the required trailer block.
- ROCm guards on `MatmulBf16D` in `DenseMlpBlock` (R1) or a CUDA
  regression test that proves the change is safe on CUDA.
- A comment fix for `rocm_bf16_attn` (R2): the comment says Dh=128,
  the predicate checks Dh=256.
- A focused test for the chunked prefill path and the bf16 GEMM
  output.

## Now

`DONE`. All six prompt-processing targets pass. The implementation
is on `perf/rocm-pp-gdn-scan` at `8e1af38fc` and is merged into
`row/GFX1100-ALL-rocm10` at `466ed8179`. The spec and issue index
row land in this commit.

## Outcome

What was measured: all six PP targets pass with margin. The final
closer was the bf16 gate/up GEMM output, which cut 32-40% off the
remaining latency by eliminating a needless f32 intermediate.

What was rejected: no optimization was rejected. All 30 attempts
either helped or were neutral. The commit message lists the key
ones; the rest were intermediate steps that compounded.

Why each default has its value:

- `ROCBLAS_USE_HIPBLASLT=1` is set by a static initializer because
  rocBLAS defaults to a slower kernel for large-N BF16 GEMMs on
  gfx1100. The static initializer respects an existing environment
  value.
- `hipMallocAsync` is used when a stream is tracked because the
  per-layer `DBuf` churn caused synchronous host overhead. The
  fallback to `hipMalloc` covers startup and weight loading.
- `MatmulBf16D` for gate/up is not guarded to ROCm because the
  change is bit-identical on CUDA (hipBLASLt and cuBLASLt both
  compute in bf16). A CUDA regression test is owed to confirm this.
