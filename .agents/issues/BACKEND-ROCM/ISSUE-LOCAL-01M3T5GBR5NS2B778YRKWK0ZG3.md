ID: ISSUE-LOCAL-01M3T5GBR5NS2B778YRKWK0ZG3
Title: EXL3 prefill on ROCm: recon arm's hipblasGemmEx picks a scalar kernel (HSS) — wire hipBLASLt
Row: BACKEND-ROCM
State: OPEN
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-01
Updated: 2026-10-01
Closed: -

## Problem

Prefill (m>32) routes to Exl3ReconstructGemmKernelRocm, whose hipblasGemmEx HIPBLAS_COMPUTE_32F resolves to Cijk...HSS at 16.4ms/call on gfx1101 (~2 TFLOP/s, 6% of gfx1101 fp16 peak). rocprofv3 on vllm-cpp:rocm-gfx1101-exl3 serving Qwen3.5-9B-EXL3-4.00bpw, 2x2022-token prompts: recon GEMM 34.0%, GdnScanK 32.3%, PagedAttnOnline 22.4%.

## Resolution

-

## Findings (2026-10-01)

hipBLASLt fixed it: 10.8x on the GEMM, 1.46x end-to-end on a 2022-token prompt.

- `hipblasLtMatmulAlgoGetHeuristic(..., 16, ...)` returns 10 candidates for m=2022 n=4096 k=4096 (f16->f32, OP_N/OP_N col-major). Rank-0 measured 15.7 ms; index 4 measured 1.5 ms (44.6 TFLOP/s). The heuristic ordering is wrong on gfx1101 for this problem family, so the arm times candidates on the caller's buffers once per (m,n,k,ldc,dtype) key and caches the winner.
- Steady-state serving, same prompt x3: baseline image 11.05 s elapsed; this build with Lt sweep 7.56 s.
- Correctness: `tests/vt/test_exl3_rocm_gemv` (60 asserts) and `test_exl3_rocm` (33 asserts) pass on gfx1101. Output parity on 3 greedy prompts: byte-identical to the baseline image on 2/3, one synonym divergence ("Count the Repetitions" vs "Count the Sentences") at token ~550 on the third — accumulation-order drift, and `VT_EXL3_RECON_NO_LT=1` on the same binary restores byte-parity on all three.

## Resolution

Implemented on branch `pp-exl3-recon-lt` in worktree vllm.cpp-pp (HgemmLtRowMajor + timed-algo selection inside Exl3ReconstructGemmKernelRocm, env escape `VT_EXL3_RECON_NO_LT=1`). Remaining prefill shares after this fix: GdnScanK ~32%, PagedAttnOnline ~24% — separate levers, not part of this issue.
