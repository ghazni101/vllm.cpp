ID: ISSUE-LOCAL-01M29ECVDVH5A89TBKA8K72YRD
Title: IQ4_NL has no native quantized compute on ROCm or CUDA
Row: QUANT-GGUF-IQ4_NL
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-11
Updated: 2026-09-12
Closed: -

## Problem

IQ4_NL (ggml id 20) reaches the GGUF reader, the CPU dequantizer and the CPU keep-quant dot, but has no native quantized compute on any device and no ROCm gather arm.

Measured on the tree at b7fb4e51f:

- ROCm: src/vt/rocm/rocm_quant_dot.hip declares ten WType entries (IQ2_XXS, IQ3_XXS, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_S, IQ1_S, IQ1_XXXS). IQ4_NL is absent. src/vt/rocm/rocm_grouped_gemm.hip:1783 and :1855 throw naming it. The backend is discrete, so an unsupported dtype cannot fall back to the CPU kernel and the throw is terminal.
- CUDA: IsCudaKeepQuantSupported has no IQ4_NL arm. The blocker is structural rather than per-format: IQ4_NL is a 32-element block paired with Q8_0 activation, not a 256-element Q8_K super-block, and the CUDA file has no Q8_0-activation GEMM variant. DotMXFP4 already sits in cuda_quant_dot.cu marked [[maybe_unused]] with a comment stating it awaits exactly that variant. Q5_0, Q4_0 and MXFP4 are queued behind the same gap.
- Gather: OpId::kEmbeddingQuant is registered for kCPU (cpu_ops.cpp:4282) and kCUDA (cuda_ops.cu:4135) only. DeviceQuantGatherSupported is therefore false for kROCM, and qwen4_exp_weights.cpp:665 refuses the load by name before any tensor I/O.

Why this blocks a shipped artifact. Every published unsloth/Qwen3.8-Flash-Next-GGUF quant stores 91,465,564,160 elements in IQ4_NL, read from the three shard headers over HTTP range requests: the 20M-entry n-gram embedding table and the 48 ffn_down_exps. The table needs the gather and the experts need the dot, so both halves are required and neither is optional for this checkpoint. UD-IQ1_S needs IQ4_NL alone; UD-IQ1_M additionally needs IQ1_M; UD-Q2_K_XL additionally needs IQ2_XS.

Record defect in the same area. .agents/quantization-matrix.md carries QUANT-GGUF-IQ4_NL as INVENTORIED with R, M, C, E and P all unset, but the reader arm (gguf_reader.cpp case 20), the dequantizer (DequantIQ4_NL) and the CPU keep-quant dot (VecDotIQ4_NLQ8_0, cpu_quant_dot.cpp:140) all landed under W6a (#1989). The row understates what the tree does.

## Resolution

-

## Reconciliation 2026-09-12: the gather half landed while this issue was open

**The problem statement above is kept verbatim because it was accurate when
written, and one of its three bullets is now false.**

[#3097](https://github.com/mudler/vllm.cpp/pull/3097)
(`feat(BACKEND-ROCM-QUANT-GATHER): gather packed embeddings on ROCm`,
`82de418e8`) landed on `main` and:

- registers `OpId::kEmbeddingQuant` for `kROCM` at `src/vt/rocm/rocm_ops.hip:191`;
- carries IQ4_NL in the ROCm gather codec list,
  `X(kIQ4_NL, DqIQ4_NL)` at `src/vt/rocm/rocm_embedding_quant.hip:88`;
- adds `src/vt/rocm/rocm_quant_iq_tables.h`, which carries
  `d_kvalues_iq4nl[16]` at `:1198` from stock `ggml-common.h:1120`.

`DeviceQuantGatherSupported` is exactly
`vt::OpRegistered(vt::OpId::kEmbeddingQuant, dev)`
(`gguf_keep_quant.cpp:220`), so it is **now true for ROCm** and the
`qwen4_exp_weights.cpp:665` refusal no longer fires on that device.

**What is still open, and is what this issue now means:** the ROCm IQ4_NL
keep-quant DOT, and the CUDA IQ4_NL keep-quant GEMM. `WType` in
`rocm_quant_dot.hip` still declares ten entries without IQ4_NL, and
`rocm_grouped_gemm.hip:1783` still throws naming it. The gather can read the
n-gram table and the expert GEMM still cannot multiply `ffn_down_exps`.

This also removed the hard dependency on
[#3029](https://github.com/mudler/vllm.cpp/pull/3029) that the spec first
recorded: the ROCm codebook header IQ4_NL needs is now on `main` by another
route. #3029 remains a conflict surface over eight files, and its two
`sanitize-cpu` reds are a repository-wide pre-existing failure in `dots3`
tests it does not touch.
