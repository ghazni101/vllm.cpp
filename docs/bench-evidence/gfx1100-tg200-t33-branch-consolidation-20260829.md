# GFX1100-TG200 T33 — branch consolidation three-point audit (2026-08-29)

Question (user): the branch integrated changes from other branches — are they
helping? Evaluate `row/GFX1100-TG200` properly and consolidate.

## Method

Three branch commits rebuilt under ONE toolchain (`rocm-dev:10.0.0` docker
image, HIP 7.15.26333, AMD clang 23, gfx1100) and A/B'd on the acceptance
workload inside the container (examples/vllm-cli, batch 1, greedy, 256
tokens, all adopted levers, 1 discarded warm-up + 5 measured reps, token
coherence via `cmp`, gpu-ctl lock held):

| commit | branch point | median tok/s (5 reps) |
|---|---|---|
| `6836c11cc` | T22-era (08-26) — the "~103 tok/s" record | 89.25 (89.08–89.36) |
| `b058bb752` | pre-T31/T32 (08-28), after both upstream integrations | 90.80 (90.65–90.88) |
| `7beb76e27` | HEAD (08-29) | 91.24 (91.06–91.32) |

Coherence: every point produced 5/5 byte-identical outputs. HEAD and
premerge outputs are byte-identical to the campaign reference
(`body.off1.txt`, md5 `9d3beddb521dd3d1b7b58d588de66ecc`); the T22-era body
differs legitimately (it predates later adopted levers).

## Findings

1. **The merged changes HELP.** Upstream integrations (`e1ea27c82`,
   `62f37025e` = upstream `e551cf8e4`) plus the fp8-KV, keep-quant, and
   sample commits are worth **+1.7%** net (premerge 90.80 vs T22-era
   89.25). T31 (device-mirror port, wash) + T32 (silu-mul+Q8_K fusion)
   add +0.5% on top (head 91.24). No merged change regressed the
   acceptance workload.
2. **The perceived 103 → 91 regression was the toolchain, not the branch.**
   The ~103 record was measured under rocm-dev:7.14 (native `/opt/rocm`,
   removed from the host 2026-08-29). Re-measured under one current
   toolchain, the branch monotonically improved +2.2% across the window
   (89.25 → 90.80 → 91.24).
3. **Merge hygiene is sound.** All 17 adopted env levers plus T31/T32 are
   present and wired post-merge. Conflict resolution kept the TG200 kernel
   files; only 4 benign files differ from the pre-merge branch on hot paths
   (platform device-fit probe #1934, reorder-threshold mirror #2129 —
   value 1 at batch 1, an fwrite guard, a HIP probe helper). The fp8
   decode-attn extras the merge brought were dropped to
   `row/fp8-kv-decode-attn` (`3a345b5ae`) and are parked there.
4. **Correctness gates pass in-container for HEAD**:
   `test_rocm_quant_dot` (132,094 assertions), `test_rocm_backend`,
   `test_rocm_arch`, `test_rocm_skinny_f32` — 4/4 green.

## Going forward

Acceptance measurements run inside `rocm-dev:10.0.0` containers
(`--device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993`), the
gpu-ctl lock requirement unchanged. Pre-swap numbers (~103 @ 7.14) are
retired as a baseline; the current baseline is **91.2 tok/s** in-container.

Raw logs/outputs: `/home/ghazni/agent-artifacts/tg200-t33-eval/` (run scripts
`tg200-eval-ab.sh`, `tg200-eval-ab-inner.sh`; per-rep `.log`/`.txt` for all
15 measured reps).
