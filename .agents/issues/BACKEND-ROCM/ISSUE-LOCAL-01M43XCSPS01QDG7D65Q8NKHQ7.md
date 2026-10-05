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

2026-10-04/05 investigation (artifacts under ~/agent-artifacts/exl3-perf-gfx1100/review/loopaudit/): (1) NOT a branch regression: pre-campaign main (fce36733b) loops at ~33 tokens (prompt-copy md5 87db8a063a) where head is clean at 43; the campaign moved the threshold UP, the defect predates it in shared engine code. (2) No sharp threshold: token sweep N=40..70 (thresh_head.json) is intermittent above ~40 - loops at 41,42,45,47,48,54-60,62-64,66-69, clean at 40,43,44,46,49-53,61,70; loop rows start at varying mid-prompt offsets = attention reading prompt KV at a wrong offset. (3) Oracle conviction: exllamav3 at the SAME 64-token prompt is clean where head loops (oracle_probe.json); at 2048 the oracle loops with the same token cycle as head (561,3712,314,...) = checkpoint-inherent. Per-prompt N=45/47/69 oracle checks remain owed (VRAM contention with a concurrent session). (4) EXL3 kernels exonerated at served shapes: bf16 vs f16 pipeline byte-identical AND both match the f64 chain at (3,2)/(4,2)/(6,2) x k=1024/5120/17408/248320 x m=1/64 (scratch_probe_v5.log); an earlier ~100% out_diff reading was a probe bug (f32 words decoded as f16), corrected. (5) Fifteen knob arms leave the loop byte-identical (md5 9cb38029d3 on BASE x3): GQA4 off, FOLD_OUT=0, WNT=2, GDN z-split/coop/donor, EXL3_GEMV=0 (full transcription), static-graph off, postconv row off, normgated off, preamble off, prefill f32q off; pre-dot-rove binary reproduces the same bytes. The break is in something shared by all arms. (6) First generated token is already the copy => prefill logits corrupt, not decode; length-gated => prefill KV write / pager / position accounting is the prime suspect. (7) All 13 GPU unit gates green at head while serving corrupts (the e1982b8aa blind-spot again). NEXT: parallel session's prefix-logits dump localizes the first divergent layer; a committed served-greedy gate is owed per ISSUE-LOCAL-01M43XDG3FEXNKWBQPRKNSV3CF.
