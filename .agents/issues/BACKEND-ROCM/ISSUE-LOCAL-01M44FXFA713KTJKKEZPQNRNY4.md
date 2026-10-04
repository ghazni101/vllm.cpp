ID: ISSUE-LOCAL-01M44FXFA713KTJKKEZPQNRNY4
Title: EXL3 decode degenerates into token loops after a few hundred tokens on gfx1100
Row: BACKEND-ROCM
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-04
Updated: 2026-10-04
Closed: -

## Problem

Served Qwen3.8-27B-EXL3-3.5bpw on HEAD ae0c9c874 (rocm-gfx11-exl3-perf) degenerates into exact n-gram repetition loops (e.g. 'tcp_slow_start_after_after?' x99 at default sampling T=1 top_k=20 top_p=0.95) after a few hundred generated tokens; short generations and greedy runs observed clean. All 10 branch unit gates pass on GPU, so the defect sits above unit coverage (long-context decode, accumulated state, or arm selection). Repro: vllmcpp:git-ae0c9c874-rocm10.0.0 container, --max-num-seqs 1 --max-model-len 8192, prompt prompts/p2_explain.txt, 900 tokens, artifacts under ~/agent-artifacts/rocm-exl3-perf-review/.

## Resolution

-
