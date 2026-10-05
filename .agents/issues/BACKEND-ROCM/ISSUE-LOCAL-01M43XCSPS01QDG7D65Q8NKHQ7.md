ID: ISSUE-LOCAL-01M43XCSPS01QDG7D65Q8NKHQ7
Title: Served greedy decode corrupt on continuation prompts at rocm-gfx11-exl3-perf head (oracle-clean comparison)
Row: BACKEND-ROCM
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-04
Updated: 2026-10-05
Closed: 2026-10-05

## Problem

Reproduced 2026-10-04 on gfx1100, head ae0c9c874, default knobs, model Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw: the 128-token greedy continuation of the campaign's standard history-of-computing prompt (BASE x3, ~45 tokens) degenerates into verbatim prompt copying in a loop. The pinned secondary oracle exllamav3 (container exl3-pr:rocm, greedy, same prompt bytes) produces a clean diverse continuation for the same prompt, so this is a vllm.cpp defect, not checkpoint behavior. The cont2048 (BASE x48) loop is checkpoint-inherent: the oracle loops with the same token cycle. Knob A/B at head: loop byte-identical with VT_ATTN_DECODE_GQA4=0, VT_EXL3_GEMV_FOLD_OUT=0, VT_EXL3_M1K3_WNT=2 (those arms exonerated). Isolation ladder (VT_GDN_SCAN_ZSPLIT=0, VT_GDN_SCAN_ZSPLIT_PREFILL=0, VT_EXL3_GEMV=0, VLLM_CPP_ROCM_STATIC_GRAPH=0, VT_ROCM_GDN_POSTCONV_ROW=0, VT_GDN_NORMGATED_COOP=0) in flight. Evidence: ~/agent-artifacts/exl3-perf-gfx1100/review/loopaudit/. Note: every unit gate stays green; the campaign's cross-build md5 checks compared builds to each other and could not see an output that was wrong in all of them.

## Resolution

2026-10-05 — falsified as a vllm.cpp defect. The pinned exllamav3 oracle
(git-584dd44f-rocm10.0.0-pp-overlay) returns logits at the first decode
position of the same BASE x3 prompt: top-1 is ' The' (561) at 10.92 with
'1' (16) at 10.88, a 0.046-logit near-tie. vllm.cpp's greedy argmax on the
same prompt is also ' The' (prompt_logprobs: ' The' -0.96 vs '\n' -1.50),
so the two implementations' greedy paths AGREE — the echo IS the oracle's
greedy output. The clean oracle completions on file were SAMPLED (the
serve_openai default honors the checkpoint's T=1/top_k=20/top_p=0.95),
where rank-2 '1' wins the dice roll at a near-tie; a fine sweep across
44..64-token non-repetitive prompts shows the same attractor at scattered
lengths (48/57/63 echo, neighbours coherent), consistent with near-tie
sensitivity rather than corruption. Complementary evidence: the full knob
isolation ladder (every ROCm arm off, incl. VT_EXL3_GEMV=0 transcription)
leaves the output byte-identical, and the EXL3 bf16/f16 pipeline is
byte-exact on every real projection shape at m in {1,33,44,48,64,85,106,127}
(scratch_lenprobe). Remaining open question — whether vllm.cpp's logit
distribution materially differs below rank 1 (oracle '1' sits at rank 2,
vllm.cpp's rank-2 is '\n'); both put ' The' first, so the greedy behaviour
is the same. The sampled-decode divergence tracked in ISSUE-LOCAL-01M44FXFA
713KTJKKEZPQNRNY4 is the live part.
