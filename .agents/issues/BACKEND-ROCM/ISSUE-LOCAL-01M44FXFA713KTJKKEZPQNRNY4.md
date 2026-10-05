ID: ISSUE-LOCAL-01M44FXFA713KTJKKEZPQNRNY4
Title: EXL3 decode degenerates into token loops after a few hundred tokens on gfx1100
Row: BACKEND-ROCM
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-04
Updated: 2026-10-05
Closed: -

## Problem

Served Qwen3.8-27B-EXL3-3.5bpw on HEAD ae0c9c874 (rocm-gfx11-exl3-perf) degenerates into exact n-gram repetition loops (e.g. 'tcp_slow_start_after_after?' x99 at default sampling T=1 top_k=20 top_p=0.95) after a few hundred generated tokens; short generations and greedy runs observed clean. All 10 branch unit gates pass on GPU, so the defect sits above unit coverage (long-context decode, accumulated state, or arm selection). Repro: vllmcpp:git-ae0c9c874-rocm10.0.0 container, --max-num-seqs 1 --max-model-len 8192, prompt prompts/p2_explain.txt, 900 tokens, artifacts under ~/agent-artifacts/rocm-exl3-perf-review/.

## Resolution

2026-10-05 — narrowed to a checkpoint sampling attractor, not a branch
kernel defect. (a) exllamav3 oracle logits at the first decode position of
a looping prompt show ' The' top-1 at 10.92 vs '1' at 10.88 — a 0.046-logit
near-tie; vllm.cpp's greedy argmax matches the oracle's. (b) The 9-knob
isolation ladder (GQA4, GEMV, dot, GDN scan coop/fused z-splits, postconv
row, normgated coop, preamble coop, prefill f32q, static graph) leaves the
looped output byte-identical — no kernel arm owns it. (c) The EXL3
bf16/f16 transcription pipeline is byte-exact at every tested m on every
real projection shape incl. the 6bpw lm_head. (d) The pre-branch baseline
build echoes byte-identically. What remains: whether vllm.cpp's top-20
distribution deviates from the oracle's enough to sample the attractor
more often (observed ~1-in-5 legs at default sampling vs oracle's clean
legs); the loop rate difference is the only unexplained delta. Still open.
