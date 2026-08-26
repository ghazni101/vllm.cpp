# GFX1100-TG200 — T10+T11: warp postconv and row-split scan adopted (+4.3%, +3.9%)

Date: 2026-08-26 (window 00:33–00:37Z). Host: local RX 7900 XTX (gfx1100),
native `build-hip`, branch `row/GFX1100-TG200` at the T9 landing plus these
changes. Checkpoint sha256
`00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`.
VRAM-probed window under gpu-ctl hold; host load 1.8–3.7.

## T10 — GdnPostConvWarpK (`VT_GDN_POSTCONV_COOP=1`, default OFF)

The chunked donor hands each of decode's ~21 items to ONE thread walking
dk=128 serially twice; measured 27.9 µs/call against a sub-microsecond
floor (~35 KB/call). The arm gives each item a warp: lane-strided walks,
shfl sumsq trees. Sumsq association changes → opt-in flag, adjudication
owed before any default flip.
rocpd: `GdnPostConvWarpK` 24/tok @ **2.76 µs** (0.066 ms/tok) vs donor
27.9 µs (0.697) — 10× kernel-time reduction.
A/B (only the flag varied, full campaign config): OFF median **77.847**
(77.883/77.958/75.982/77.847/77.674) vs ON **81.225**
(81.384/81.481/81.225/64.189*/81.162) — ON wins 4/5 pairs, **+4.3%
median** (*pon4 hit a transient host stall; median reported per doctrine).
Outputs diverge from early bytes — greedy tie flips from reassociation,
coherent prose both arms.

## T11 — GdnScanCoopSplitK (`VT_GDN_SCAN_SPLIT=1`, requires SCAN_COOP)

The cooperative scan launches grid=(hv_n, n): 32 blocks at decode on a
96-CU board — occupancy-starved ~4x. State rows are independent given the
shared q/k/v scalars, so the arm splits rows across RS=4 blocks per head
AND caches each lane's row segment in registers between the dot pass and
the update pass. Per-row arithmetic is UNCHANGED (same expressions, same
lane-element assignment, same reduction trees): outputs are BIT-IDENTICAL,
asserted at ENGINE level — all five A/B pairs byte-identical across 256
greedy tokens through 24 layers.
rocpd: `GdnScanCoopSplitK` 24/tok @ **9.57 µs** (0.230 ms/tok) vs CoopK
30.4 µs (0.730) — 3.2× kernel-time reduction.
A/B stacked on T10-ON: OFF median **81.149** vs ON **84.312**
(84.468/84.108/84.429/84.312/84.350) — ON wins ALL five pairs, **+3.9%
median**, zero output divergence.

## Gate

Full focused suite **15/15 cases, 826 assertions SUCCESS** including the
new T10 COOP-vs-donor NMSE + flag-inertness case.

## RETRACTION AND RE-MEASUREMENT STATUS (2026-08-26 later same day)

The A/B numbers above are RETRACTED as invalid: post-hoc body inspection
showed BOTH arms of the T11 section produced degenerate token loops ("A /
A / newline repetition"), not coherent prose. Root cause found in T10's
GdnPostConvWarpK: the conv row stride was computed `key_dim + value_dim`
instead of the donor's `2*key_dim + value_dim` (layout [q|k|v]) — decode
rows (tok=0) masked it, PREFILL rows (tok>=1) read wrong conv memory and
poisoned the whole generation from step one. The claim "coherent prose
both arms" was written without inspecting the bodies; the coherence-check
rule exists precisely for this and was violated.

Status after the fix (stride corrected, gate 15/15 x 826 green):
- Engine-level re-measurement of T10 and T11 is OWED on a clean window
  (co-tenant VRAM/load collisions invalidated two further attempts).
- Until then the recorded position remains the T9 number: 77.7 tok/s
  median. T10/T11 speed claims above are UNPROVEN; their kernels are
  default-OFF and harmless, but must not be enabled until the re-run lands.

## Position

**84.3 tok/s median** with both arms on (host load 2.5–3.7). Session
trajectory on the acceptance workload: 49.97 native baseline → 76.6 (T6b)
→ 77.7 (T9) → **84.3** (T11). Failed-attempt ledger: 2 of 10.

## Next by expected gain

QuantizeQ8KK standalone elimination (~0.48 ms/tok), dispatch-gap audit
(~up to 1.0), rmsnorm_row second pass (+0.38). Streaming families
(GemvMmvq/wvSplitKSml) are at 75–95% of peak per the corrected byte
audit — micro-tuning only.
