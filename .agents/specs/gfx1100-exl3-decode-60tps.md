# gfx1100 EXL3 decode to 60 tok/s for Qwen3.8-27B-3.5bpw (BACKEND-ROCM)

Row: `BACKEND-ROCM`. Issue:
`.agents/issues/BACKEND-ROCM/ISSUE-LOCAL-01M3WVXG0PV5F0MDB70VK9T9YX.md`.
Branch: `rocm-gfx11-exl3-perf` (developer directive: all work lands there).

## Problem

`Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` (sha256-pinned in
`docs/benchmarks/qwen38-27b-exl3-gb10.md`) decodes on the host RX 7900 XTX
(gfx1100, 24 GiB, ~960 GB/s peak DRAM) at **19.5 tok/s** at branch head
`ecd81113c` (256-token greedy decode leg, `--max-num-seqs 1`, no speculative
decoding). The developer target is **60 tok/s decode** without speculative
decoding, no re-quantization, KV cache no narrower than 8 bit.

rocprofv3 kernel census of one decode token (busy 46.3 ms, span 56.3 ms,
2580 launches; decode graph `S=1` captured and replaying):

| Kernel | ms/token | calls | Share |
|---|---|---|---|
| `Exl3GemvM1K4<2>` (4 bpw body weights) | 14.35 | 262 | 31% |
| `Exl3GemvM1K<3,2>` (3 bpw layers) | 12.40 | 137 | 27% |
| `Exl3DotKImpl<6,8>` (6 bpw lm_head) | 3.81 | 1 | 8% |
| `GdnScanK` (48 GDN layers, default arm) | 2.71 | 48 | 6% |
| `PagedAttnOnline<f32,bf16,f32>` (16 full-attn layers) | 2.45 | 16 | 5% |
| `HadK` in/out (unfused Hadamard) | 2.88 | 802 | 6% |
| `AttnQkNormRopeGateK` + norms + postconv | ~5.6 | ~400 | 12% |
| casts + misc | ~1.6 | ~1300 | 4% |

Roofline: ~11.7 GB of EXL3 trellis traffic per token through the two GEMV
arms ⇒ 27 ms/token at ~437 GB/s effective. 60 tok/s (16.7 ms/token) needs
~700 GB/s sustained on that traffic plus the fixed ~19 ms of everything else
cut to ~4 ms. Both halves of the work are therefore required:

- **Kernel side**: the m=1 GEMV arms and the lm_head dot arm must run nearer
  the DRAM roofline on gfx1100.
- **Structural side**: `GdnScanK`'s default walk, the f32-query
  `PagedAttnOnline` fallback, unfused Hadamard stages and per-launch graph
  overhead are 20+ ms/token of recoverable work.

## Constraints (developer-specified)

- No speculative decoding of any kind (no draft model, no MTP arm, no
  `--speculative-config`).
- No re-quantization of the model weights.
- KV cache stays at bf16/fp8 — never below 8 bit.
- Honest measurement: numbers are decode-only tok/s (`1000/mean tpot`) and
  whole-run tok/s, both reported, on the real server behind the OpenAI
  streaming client in `~/agent-artifacts/qwen38-27b-exl3-bench/client.py`.
- Correctness checks integrated at critical places: existing
  `test_exl3_rocm*` unit gates per kernel change, plus greedy-continuation
  checks on the served model after every landing.

## Design / candidate levers (measured, in order)

1. **Opt-in arms that already exist, flipped where verified**: GDN coop
   scan (`GdnScanCoopFusedK`, bit-identical reduction order per
   gfx1101 spec), f32-query GQA decode attention (`VT_ATTN_DECODE_GQA4`),
   measured KSPLIT for the m=1 arms.
2. **GEMV kernel rework** (`rocm_exl3_gemv.hip`): replace the per-lane
   scalar boundary-word load in `Exl3GemvM1K4` with a `__shfl` (one load
   stream instead of two), lower VGPRs below the occupancy cliff (168 regs
   measured), deepen/widen the prefetch stream, and evaluate a
   `uint4`-loads-per-lane variant where each lane owns codewords from its
   own 16B window (the M1K4 pattern) extended to 3 bpw.
3. **lm_head dot arm** (`rocm_exl3_dot.hip`): 3.8 ms for a 1.4 GB read is
   ~370 GB/s; same roofline work as the GEMV arms.
4. **Hadamard fusion**: 802 `HadK` launches/token at 2.9 ms. Where the
   GEMV arm is single-block-scope the pre/post Hadamard can ride the same
   kernel boundary-free; otherwise fold the output Had into the C-store
   epilogue (f32 → per-column scale) where numerics allow.
5. **Launch-count reduction**: 2580 launches/token inside a captured graph
   still costs ~4 µs/launch of dead time (~10 ms/token). Fewer, wider
   kernels (fused norms, batched per-layer GEMV grouping where shapes
   allow) cut this directly.

## Rejected levers

- Speculative decoding, draft models, MTP: excluded by the developer.
- KV cache below 8 bit: excluded by the developer.
- Weight re-quantization: excluded by the developer.

## Gates

- `test_exl3_rocm`, `test_exl3_rocm_gemv`, `test_ops_gdn`,
  `test_ops_paged_attn*`, `test_paged_attn_route` on gfx1100 after each
  kernel change.
- Served-model greedy probe after each landing: `POST /v1/completions`
  "The capital of France is" T=0 must return the documented continuation;
  plus a 64-token greedy decode A/B vs the pre-change build (token-exact
  or documented drift).
- Performance: decode-only and whole-run tok/s on the fixed 12-prompt
  HumanEval leg; report both.

## Stop conditions

- ≥60 tok/s decode-only at `--max-num-seqs 1`, correctness gates green.
- Or a measured roofline argument that the checkpoint's byte traffic makes
  60 tok/s unreachable on this card — reported with the per-component
  numbers, not asserted.

## Evidence

- Baseline census: `~/agent-artifacts/exl3-perf-gfx1100/trace/run1/`
  (rocprofv3 kernel trace + stats CSV, 2026-10-01).
- Baseline decode: 19.5 tok/s (256-token leg), 17.7 tok/s under rocprofv3.

## Owed

- The ~10 ms/token of launch overhead inside the captured graph — the
  graph replays but per-node launch cost remains; grouped/batched kernel
  structure is the fix.
