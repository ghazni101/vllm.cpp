# GFX1100-TG200 — T10+T11: warp postconv and row-split scan ADOPTED (corrected record)

Date: 2026-08-26 (valid windows 03:20Z and 04:14–04:20Z plus full-config
verification 05:2xZ). Host: local RX 7900 XTX (gfx1100), native `build-hip`,
branch `row/GFX1100-TG200`. Checkpoint sha256
`00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`.

## CORRECTION HISTORY — read before citing

An earlier revision of this file claimed +4.3%/+3.9% from a window whose
outputs were later found DEGENERATE (token loops). Root cause: T10's
GdnPostConvWarpK computed the conv row stride `key_dim+value_dim` instead
of the donor's `2*key_dim+value_dim` ([q|k|v] layout) — decode rows masked
it, prefill rows read wrong memory. The stride is fixed; the claims below
come from post-fix windows whose bodies were coherence-checked. The failed
windows and the process rules they forced (body-content check per arm,
engagement witness per window, all-targets relink) are retained in the
git history of this file.

## T10 — GdnPostConvWarpK (`VT_GDN_POSTCONV_COOP=1`, default OFF)

Warp-per-item remap of the chunked donor (which hands each decode item to
ONE thread walking dk=128 serially twice): lane-strided walks, shfl sumsq
trees. Sumsq association changes → opt-in flag, adjudication owed before
any default flip.
Kernel time (rocpd): **27.9 → 2.76 µs** (10×).
Clean-window A/B x5 interleaved pairs, only the flag varied:
OFF median **82.42 tok/s**, ON median **86.31 tok/s** — ON wins all five
pairs, **+4.7%**. Bodies coherent analytic prose both arms; divergence at
expected tie-flip points.

## T11 — GdnScanCoopSplitK (`VT_GDN_SCAN_SPLIT=1`, requires SCAN_COOP)

Row-split blocks (RS=4: 32→128 blocks at decode) plus register-cached row
segments between the dot and update passes. State rows are independent, so
per-row arithmetic is UNCHANGED: engine outputs are BIT-IDENTICAL — all
five stacked pairs byte-identical across 256 greedy tokens through 24
layers.
Kernel time (rocpd): CoopK **30.4 → 9.57 µs** (3.2×).
A/B x5 interleaved pairs (on the T10-OFF base): OFF median **82.29**,
ON median **84.95** — ON wins all five pairs, **+3.2%**.

## Gate

Focused suite **15/15 cases, 826 assertions** including the T10
COOP-vs-donor NMSE + inertness case. Post-retraction hardening: the arm's
env toggle reads PER CALL (the once-per-process static let the unit test's
ON arm silently reuse the donor — mutation-verified fix, nmse 1.30 RED
with the stride bug reintroduced).

## Full-stack position

All adopted levers on (`MMVQ SKINNY GQA4 SCAN_COOP PREAMBLE_COOP
NORM_QUANT_FUSED RMSNORM_ROW_COOP NORMGATED_COOP POSTCONV_COOP
SCAN_SPLIT`):
- Short prompt (~45 tok): warmup 89.5, steady **99.9 tok/s ×2**.
- Canonical 70-token prompt: warmup 84.3, steady **92.9/92.7 tok/s**,
  coherent.

Prompt-length caveat: tonight's paired A/Bs used the ~45-token prompt;
older windows used longer prompts, so absolute numbers are not
cross-era comparable — the PAIRED DELTAS are the verified quantities. A
formal acceptance-gate rerun (canonical long prompt, idle host, 6-rep
median) on this config remains owed for the campaign's absolute position
record.

## Session ledger context

Adopted across sessions: T5a (+23%), T5b (+13.5%), T6a (+4.6%), T6b
(+4.6%), T8 (+3.2%), T9 (+2.6%), T10 (+4.7%), T11 (+3.2%) — all paired,
all coherence-checked. Closed negative/not-adopted: T5c, T7, T12.
Failed-attempt ledger: 3 of 10.

## Full-config verification (2026-08-26 late, clean GPU)

With the sibling training finished (full VRAM), the complete eleven-flag
config was verified end-to-end:
- Graph replay ENGAGES with all new arms captured: "[DenseDecodeGraph]
  captured ... S=1", "126 total replays" over 128 tokens — capture-safety
  of every arm added this session is empirically confirmed.
- Short prompt (~45 tok): warmup 89.5, steady **99.9/101.1 tok/s**.
- Canonical 70-token prompt: warmup 84.3, steady **92.9/92.7 tok/s**,
  coherent analytic output.

Fresh rocpd budget at this config (8.89 ms/tok kernel busy): the three
streaming families hold 6.33 ms/tok at their audited near-peak rates;
every latency-class kernel added or remapped this session sits at
0.02–0.75 ms/tok. Remaining non-kernel time ~1.9 ms/step decomposes into
the ~290 us sampling round trip plus per-op launch gaps — T13 scope,
requiring the async-serving engine path (the blocking CLI cannot engage
AsyncScheduler), which is the next session's scoped item.
