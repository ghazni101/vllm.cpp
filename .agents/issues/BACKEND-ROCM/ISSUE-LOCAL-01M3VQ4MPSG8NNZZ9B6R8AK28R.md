ID: ISSUE-LOCAL-01M3VQ4MPSG8NNZZ9B6R8AK28R
Title: gfx1101 decode latency: f32-query attention misses fast arms, GDN scan reads state twice
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

Qwen3.5-EXL3 decode on gfx1101 ran 33.7ms/token with the GPU idle ~60%: (1) prefill/decode attention for the model's f32-query/bf16-KV dtype mix missed every fast arm and fell to PagedAttnOnline at 2.7ms/call; (2) GdnScanCoopK read the state row twice per token (349us/call); (3) the dense decode graph and SharedK WMMA prefill stay gated to gfx1100 while gfx1101 carries identical hardware.

## Resolution

-
