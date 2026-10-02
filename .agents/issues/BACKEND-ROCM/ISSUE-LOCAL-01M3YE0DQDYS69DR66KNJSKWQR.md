ID: ISSUE-LOCAL-01M3YE0DQDYS69DR66KNJSKWQR
Title: fused EXL3 GEMV staging: __float2half contracts to single-rounding v_fma_mixlo_f16, diverges from HadK double rounding
Row: BACKEND-ROCM
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-02
Updated: 2026-10-02
Closed: -

## Problem

HadBlock128ToH2 (src/vt/rocm/rocm_exl3_gemv.hip) wrote its staged a_had elements with __float2half(x*kInvSqrt128), which hipcc contracts to v_fma_mixlo_f16 — a single-rounding f32*f32->f16. Reference HadK<true,true> computes res=sh*r_scale in f32 then DF32ToF16 (double rounding). On ~1/5K boundary inputs the staged bits differ by 1 ulp; that error is linearly combined into all 64 output columns of the block's group, producing fused-vs-unfused C differences (maxrel ~0.002) and serving token divergence. Fix: write DF32ToF16(x*kGemvInvSqrt128) through uint16_t* so the f32 product materializes and the bit pattern is stored, matching HadK's double rounding. Verified: staging-bits and fused C now bit-identical on k=5120 n=1024/4096 and k=17408 n=1024.

## Resolution

-
