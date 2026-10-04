ID: ISSUE-LOCAL-01M43XCSPS01QDG7D65Q8NKHQ7
Title: Served greedy decode corrupt on continuation prompts at rocm-gfx11-exl3-perf head (oracle-clean comparison)
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

Reproduced 2026-10-04 on gfx1100, head ae0c9c874, default knobs, model Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw: the 128-token greedy continuation of the campaign's standard history-of-computing prompt (BASE x3, ~45 tokens) degenerates into verbatim prompt copying in a loop. The pinned secondary oracle exllamav3 (container exl3-pr:rocm, greedy, same prompt bytes) produces a clean diverse continuation for the same prompt, so this is a vllm.cpp defect, not checkpoint behavior. The cont2048 (BASE x48) loop is checkpoint-inherent: the oracle loops with the same token cycle. Knob A/B at head: loop byte-identical with VT_ATTN_DECODE_GQA4=0, VT_EXL3_GEMV_FOLD_OUT=0, VT_EXL3_M1K3_WNT=2 (those arms exonerated). Isolation ladder (VT_GDN_SCAN_ZSPLIT=0, VT_GDN_SCAN_ZSPLIT_PREFILL=0, VT_EXL3_GEMV=0, VLLM_CPP_ROCM_STATIC_GRAPH=0, VT_ROCM_GDN_POSTCONV_ROW=0, VT_GDN_NORMGATED_COOP=0) in flight. Evidence: ~/agent-artifacts/exl3-perf-gfx1100/review/loopaudit/. Note: every unit gate stays green; the campaign's cross-build md5 checks compared builds to each other and could not see an output that was wrong in all of them.

## Resolution

-
