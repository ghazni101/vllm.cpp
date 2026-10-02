ID: ISSUE-LOCAL-01M3Y3BJDWBFKM6YPDDD3ZE3MR
Title: gfx1100: f32q GQA decode arm rejects Qwen3.8-27B head geometry (hq=24,kv=4,d=256) -> PagedAttnOnline 183us/call
Row: BACKEND-ROCM
State: OPEN
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-02
Updated: 2026-10-02
Closed: -

## Problem

Qwen3.8-27B-EXL3 runs 16 full-attn layers at hq=24 kv=4 d=256 with f32 query/out. The VT_ATTN_DECODE_GQA4 f32-query arm requires hq==16, so every full-attn decode falls to PagedAttnOnline at 183us/call = 2.9ms/token. The kernel is generic over QG; the guard is artificial.

## Resolution

-
