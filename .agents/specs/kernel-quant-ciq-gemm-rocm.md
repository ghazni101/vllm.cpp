# KERNEL-QUANT-CIQ-GEMM-ROCM — keep-quant GEMM providers on kROCM

- Issue: [#1587](https://github.com/mudler/vllm.cpp/issues/1587)
- Base: `e2a9e035d` (upstream/main)
- State at commit: `SPIKE` accepted; W1 implementation rides this pull request
- Pull request shape: one pull request for spec and implementation
  (developer decision 2026-08-21)

## Scope

The ROCm backend registers roughly 44 ops and has no quantized-weight GEMM
provider. A search for `MatmulBTQuant`, `kMatmulBTQuant`, and `vec_dot` over
`src/vt/rocm/` and `include/vt/` returns nothing. Every GGUF k-quant weight
on an AMD card therefore computes off device today.

Two waves, one row:

1. **W1 (this change).** `kROCM` providers for `OpId::kMatmulBTQuant` and
   `OpId::kMatmulBTQuantGrouped`: the GGUF Q8_K-family keep-quant GEMM,
   mirroring the CUDA sibling's contract. Registering the provider flips
   `GgufQuantComputeAvailable()` true on the platform, so every GGUF
   k-quant model reaches it with zero model-code edits.
2. **W2 (owed, see `## Owed`).** The upstream `csrc/rocm` W4A16 GPTQ/AWQ
   family (`wvSplitK_int4_g`, `gptq_gemm_rdna3`,
   `gptq_gemm_rdna3_wmma`, `moe_gptq_gemm_rdna3`). This project cannot
   reach those kernels yet: GPTQ and AWQ checkpoints have only a host
   dequant path (`awq_gptq_dequant.cpp`) and no W4A16 consumer. Porting
   them before a consumer exists would land dead code.

Out of scope: FP8 on gfx1100 (upstream refuses it on this arch;
`supports_fp8()` is gfx9 or gfx12x only), Triton-on-ROCm families, and any
loader work for AWQ/GPTQ checkpoints.

## Upstream anchors

Pinned vLLM `555967922`:

- `csrc/rocm/torch_bindings.cpp` names the whole HIP quant-GEMM surface:
  `LLMM1`, `wvSplitK`, `wvSplitKrc`, `wvSplitK_int4_g`, `wvSplitKQ`, and
  the `VLLM_ROCM_GFX1100`-gated `gptq_gemm_rdna3`,
  `gptq_gemm_rdna3_wmma`, `moe_gptq_gemm_rdna3`.
- `csrc/rocm/q_gemm_rdna3.cu:1-40` (header) records the RDNA3 hardware
  facts W2 inherits: wave32 geometry, no native packed fp16/bf16 atomic
  add (emulated with `global_atomic_cmpswap_b64`), `v_dot2_f32_f16` for
  fp16, fp32-widened accumulate for bf16, and the WMMA forward at
  `M >= 16`.
- `vllm/platforms/rocm.py` `supports_fp8` excludes gfx1100.

Classification per `.agents/porting.md`: W1 has **no upstream
counterpart** — vLLM has no GGUF keep-quant device path anywhere. It is
derived from our own CUDA sibling plus the ggml CPU reference semantics,
and it is recorded as such in `porting-inventory.md` section 9. W2 is a
1:1 port of the pinned files.

## Local anchors

- `include/vt/ops.h:176` `kMatmulBTQuant`; `:184` `kMatmulBTQuantGrouped`;
  `:1629` the `MatmulBTQuant` entry signature.
- `src/vt/ops.cpp:186-211` validation and dispatch through
  `GetOp(OpId::kMatmulBTQuant, q.device.type)`.
- `src/vt/cpu/cpu_quant_gemm.cpp:302-310` the CPU registrar — the exact
  oracle.
- `src/vt/cuda/cuda_quant_dot.cu:1-18` the oracle chain (kernel wiring,
  per-block dot, activation quant); `:1814` the provider; `:1990-1993`
  the registrar whose registration flips the loader default.
- `src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:75-78`
  `GgufQuantComputeAvailable()` — the reachability flip.
- `src/vt/rocm/rocm_ops.hip:101` the kROCM registration pattern;
  `src/vt/rocm/rocm_backend.hip:328` the unified-memory bool that decides
  CPU-reference fallthrough on APUs versus discrete cards.

## Design

W1 adds `src/vt/rocm/rocm_quant_dot.hip`, structured like
`cuda_quant_dot.cu`:

- Quantize each activation row to `Q8_K` on the device, then run an
  integer dot against the compressed weight blocks per output element.
  Integer arithmetic is exact, so the provider gates **bit-exact**
  against the CPU provider — the same bar the CUDA sibling meets.
- First-wave weight types: `Q4_K`, `Q5_K`, `Q6_K`, `Q2_K`, `Q3_K`. The
  IQ codebook types (`IQ2_XXS`, `IQ3_XXS`) join when their tables port
  cleanly; `Q8_0`/`Q4_0` activations fall back to the CPU provider over a
  drained queue, exactly as `cuda_quant_dot.cu:1834` does.
- Geometry sized for wave32 on gfx1100; the CUDA warp-per-output shape
  carries over with wave-size adjustments. Geometry is a performance
  concern only; correctness comes from the exact integer core.
- Memory: gfx1100 is a discrete card, so weight blocks must be
  device-resident. The unified-memory assumptions in the model paths do
  not hold here. The provider requires device pointers and relies on the
  existing weight-staging path; the implementer verifies
  `needs_weight_staging()` reports true for `kROCM` so loaders stage
  blocks once. If staging needs model-path edits beyond the platform
  seam, that is a stop condition below.
- Registration follows `rocm_ops.hip:101`. From registration onward,
  `GgufQuantComputeAvailable()` is true on `kROCM` and the GGUF loader
  routes keep-quant towers to the device. A rollback env kill switch
  mirrors whichever flag the CUDA side exposes.

## Risks

- R1: weight residency on a discrete card. The DeepSeek-V4 and Qwen3.5
  paths stage weights through `ResidentWeight` gated on
  `needs_weight_staging()`; if that predicate is CUDA-only, W1 grows the
  platform-seam fix and says so.
- R2: wave32 geometry differences make the first build slower than the
  CPU tier on some shapes. That is a recorded measurement, not a
  correctness failure; the provider stays default-on only if it wins or
  ties, else ships behind the kill switch with the numbers in the spec.
- R3: hipcc `-O0` device code starts a hostcall listener that can deadlock
  at exit (#132). Builds set a `CMAKE_BUILD_TYPE`; the container baseline
  uses Release.
- R4: IQ codebook tables grow `.rodata`; deferring them keeps W1 small.

## Tests

Red-first, in the same change:

1. Extend the quant-dot operator tests with `kROCM` arms: per-type
   bit-exact equality against the CPU provider on random and boundary
   inputs, the `K % 256` refusal, the grouped variant's expert-index
   contract, and the unsupported-dtype CPU-fallback arm. Capture the red
   before the provider exists (`OpRegistered(kMatmulBTQuant, kROCM)` is
   false and keep-quant stays off).
2. Focused gate: `ctest -R 'rocm|cross_device|quant'` inside the
   `rocm-dev:7.14.0` container under the host GPU mutex.
3. Model-level smoke: one small GGUF checkpoint (Qwen3.5-0.8B Q4_K_M,
   fetched under the recorded authority) decodes end to end on gfx1100
   with keep-quant routed to the device, token-identical to the same
   build forced onto the CPU provider.

## Gates

Correctness gate: bit-exact versus the CPU `kMatmulBTQuant` provider on
the declared types, plus the model smoke above. The pinned-vLLM ROCm
oracle does not cover GGUF keep-quant (`BACKEND-GATE-ROCM-VLLM` stays
`INVENTORIED`), so vLLM parity for this wave is out of reach by
construction and said so. Performance axes are measured and recorded; no
throughput floor is claimed in W1.

## Evidence

- Container baseline on 7.14: build 586 of 586 targets green; focused
  gate 4 of 5 with the `MoeSiluMul` bf16 exactness failure recorded on
  [#1586](https://github.com/mudler/vllm.cpp/issues/1586).
- This row appends its measurements to `## Outcome` when it reaches DONE.

## Stop conditions

- `NEEDS_DECISION`: weight staging requires edits to model forward paths
  rather than the platform seam.
- Stop and report if bit-exactness cannot be reached; the integer-dot
  premise would be violated, which means the port is wrong somewhere.

## Owed

- W2: the upstream `csrc/rocm` W4A16 family port together with the loader
  consumer that makes it reachable. Stays owed unless it lands in this
  pull request.
- `porting-inventory.md` section 9 entry for the W1 derivation.
