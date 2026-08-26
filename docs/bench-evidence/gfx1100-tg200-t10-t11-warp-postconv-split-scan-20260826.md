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

## Dispatch-gap audit (T13 target definition, same-day)

Argmax-delimited decode-step analysis over the campaign-config capture
(`cap-t1011`): 503 steady-state steps average **span 10.84 ms**, kernel
busy 9.00 ms ⇒ **~1.83 ms/step non-kernel time** under trace. Largest
single stalls cluster around the per-token sampling round trip:
`__amd_rocclr_copyBuffer` pairs flanking `EmbeddingErr` carry idles of
302/80/62 µs — the argmax-result D2H copy serializing each step against
the host before the next embedding fill. Candidate lever: on-device
sampling/token feedback (the LAGUNA path already has
`VT_LAGUNA_ONDEV_SAMPLE`; the main GDN path does not). Second-order gaps
of 10–16 µs repeat after the GemvMmvq→SiluMul boundary (~24/tok ≈ 0.3 ms
aggregate). Numbers are trace-inflated; an untraced paired measurement
owes before any lever claim.

## Re-measurement decision rules (mechanical, applied to batch-final output)

1. Every arm's body must pass the letter-density coherence probe AND read
   as analytic prose on manual spot-check; an empty or degenerate body
   voids the window.
2. T11: all five pairs must be BYTE-IDENTICAL (bit-exactness is the
   lever's design contract). Any divergence kills the split arm.
3. T10: divergences are expected tie flips — record first-divergence byte
   and confirm both streams coherent; teacher-forced ceremony stays owed.
4. Adopt iff ON median beats OFF median with ON winning >= 4 of 5 pairs;
   then enable both flags in the campaign config, restore this file's
   numbers, and update the spec position. Otherwise revert both arms
   byte-restored and close per the T5c/T7 precedent.

## Dispatch-gap refinement (memory-copy table + per-step sequence)

The campaign-config capture contains ZERO D2H copies inside decode
windows — the sampled-id handoff rides `__amd_rocclr_copyBuffer` KERNEL
entries. Per-step anatomy at an Argmax boundary:

| op | dur | gap before |
|---|---|---|
| ArgmaxK (greedy, [1, vocab] f32) | **153.96 us** | — |
| rocclr_copyBuffer (sampled id D2H) | 3.2 | 8 |
| — **stall** — | — | **289.4** |
| rocclr_copyBuffer #2 (next-step setup) | 3.2 | 40 |
| EmbeddingKernel | 2.9 | 14.6 |
| ~dozens of state-update copyBuffers | 2-6 ea | 3-5 ea |

Three concrete sub-targets, ranked:
1. **Sampling round trip ~290 us/step**: host wakes on the D2H, processes
   one token, issues the next step. On-device token feedback or a
   one-step-deferred sync removes it.
2. **ArgmaxK 154 us for a 993 KB row**: launch geometry walks the row
   serially at batch 1 — same single-block class as T8/T9/T11 won on.
   Expected floor ~5 us ⇒ ~0.15 ms/tok.
3. **Small-copy storm**: dozens of 2-6 us copies behind 3-5 us host gaps
   ≈ 0.3-0.5 ms/tok aggregate — fold into the decode graph or batch the
   host API calls.

## ISA verification: the dp4a core already uses RDNA3 hardware dot (2026-08-26)

Question raised by the objective ("use the architecture fully"): does the
repo's scalar `Dp4a` fallback (four int8 multiplies + adds,
`rocm_grouped_gemm.hip:63`) actually lower to the hardware dot instruction
on gfx1100, or is every quant kernel emulating it?

Answer, by disassembling an `-O3 --offload-arch=gfx1100` compile of both
forms: the scalar body AUTO-FORMS `v_dot4_i32_iu8` with
`neg_lo:[1,1,0]` signedness handling — 12 instructions total for the whole
test kernel. The explicit `__builtin_amdgcn_sdot4` intrinsic, by contrast,
fails to compile unless the `dot1-insts` target feature is forced. So the
idiom-recognition path is not just sufficient but the ONLY practical
spelling, and every K-quant kernel (GemvMmvq family, grouped paths) already
executes the hardware dot instruction per element group.

Consequence for the ladder: the integer-dot core of the quant families has
no instruction-selection headroom on this silicon. Combined with the
78-88%-of-peak streaming audit, this closes the last speculative lever on
the GemvMmvq/wvSplitK families at the ISA level — their remaining costs are
memory-system physics, matching the T5c/T7 measurements. Future levers stay
in the latency/fusion/dispatch classes named above.
