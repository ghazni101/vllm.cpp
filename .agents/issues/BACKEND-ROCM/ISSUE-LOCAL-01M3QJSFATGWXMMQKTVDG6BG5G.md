ID: ISSUE-LOCAL-01M3QJSFATGWXMMQKTVDG6BG5G
Title: ROCm EXL3 decode is 97% Exl3GemmK scalar-transcription kernel — port GEMV arm
Row: BACKEND-ROCM
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-30
Updated: 2026-09-30
Closed: -

## Problem

On gfx1101 serving Qwen3.5-9B-EXL3-4.00bpw, rocprofv3 attributes 11392ms of ~11700ms kernel time to vt::rocm::Exl3GemmK (~712ms/token, ~1.7 tok/s). Byte-exact CPU transcription: 16/256 live threads at m=1, double barriers per k-tile, no prefetch. Upstream's m<=8 exl3_gemv arm (exl3_gemv_kernel.cuh; ROCm-proven in exllamav3-rocm) unregistered for kROCM; kExl3ReconstructGemm CUDA-only.

## Resolution

-
