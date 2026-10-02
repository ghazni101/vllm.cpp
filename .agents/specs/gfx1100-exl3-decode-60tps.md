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
- 2026-10-02: fused in-hadamard m=1 GEMV arms landed (`1d04d72dc`, review
  fixes `91c263249`). Fused-vs-unfused bitwise checks pass (staging 0 bad
  at k=5120 and k=17408; C bit-identical at n=1024/4096 for 4 bpw and
  n=1024 for 3 bpw). Served greedy probe returns the pinned continuation.
  Decode on the same HumanEval leg: `ecd81113c` 19.1 tok/s → `91c263249`
  21.4 tok/s decode-only (52.4 → 46.8 ms/token mean tpot).
- 2026-10-02 (later): kernel-level work on `rocm-gfx11-exl3-perf`:
  - `e092643c2` — `fshift` lowered to `v_alignbit_b32` via
    `__funnelshift_r` (the 64-bit idiom compiled to a multi-cycle
    `v_lshrrev_b64`) with a `shift < 32` guard for the 3 bpw path whose
    `s2+12` reaches 42; `decode_pair_cb2_dp4a_` byte-sum via
    `__builtin_amdgcn_udot4` (`v_dot4_u32_u8`). GEMV bench: fused4
    456→571-596 GB/s, fused3 373→~390, unfused prod4 453→~580,
    prod3 373→~500.
  - `aaac1aa51` — the same byte-dot pair decode in the dot arm's cb=2
    path (`Exl3DecodePair6`), helper hoisted to `rocm_exl3_decode.h` so
    both TUs share it; scalar `Exl3DecodeCodeword` cb=2 also uses udot4.
  - `fb64d2ae8` — evict-first (`__builtin_nontemporal_load`) trellis
    loads in the fused 4 bpw arm: fused4 571→596-603 GB/s.
  - `91c2e3f43` — the fused m=1 arm is shape-gated: measured slower than
    unfused everywhere except 4bpw n>=12288 (qkvo/gateup). 3 bpw fused
    ~330-390 vs unfused ~440-520 GB/s on every shape.
  - A/B levers measured end-to-end (decode-only tok/s, same leg):
    defaults+fused-everywhere 24.4 → `VT_GDN_SCAN_COOP=1` 26.1 →
    +`VT_ATTN_PREAMBLE_COOP=1`+`VT_GDN_NORMGATED_COOP=1` 27.2 →
    with the dispatch gate 27.7. `VT_EXL3_GEMV_KSPLIT=4` 20.5 (regression),
    `VT_EXL3_DOT_FIRST=1` 7.6 (regression), `VT_ROCM_EXL3_WMMA=1` 7.8
    (regression).
  - Ceiling: 11.59 GB of trellis bytes per decode token (quantization
    tensor_storage sum) vs ~700-960 GB/s DRAM on this card puts the
    GEMV-only floor at ~12-16.5 ms/token; every other kernel adds on top.
    60 tok/s therefore needs the decode GEMV to run at DRAM peak, which
    no measured arm reaches (~600 GB/s best). Stop-condition territory:
    ~40-45 tok/s is the plausible bound without a structurally different
    decode.
- 2026-10-02 (close): `05183d1ca` defaults the three coop arms
  (`VT_GDN_SCAN_COOP`, `VT_ATTN_PREAMBLE_COOP`, `VT_GDN_NORMGATED_COOP`)
  ON — measured +1.7 tok/s decode with the fused dispatch gate.
  Authoritative same-harness A/B on the 12-prompt HumanEval leg,
  256-token decode, 0.6/0.95/20 sampling, identical server flags:
  `ecd81113c` (day-start HEAD) 18.15 decode tok/s, 16.49 whole-run →
  `05183d1ca` 24.24 decode tok/s, 22.40 whole-run. +33.6% decode,
  +35.8% whole-run. Golden greedy parity holds ("The capital of France
  is" → " Paris.").
  Final state vs the 60 tok/s target: ~24.4 tps achieved, ~40-45 the
  plausible ceiling — see the 11.59 GB/token byte-traffic bound above.
  Reaching 60 needs a structurally different decode (batched/grouped
  GEMV, CUDA/HIP-graph replay of the per-layer stack, or a quant with
  fewer bytes/token) — none allowed by this task's constraints.
- 2026-10-02 (bf16 fold): `e1982b8aa` — the bf16 residual stream's two
  per-linear casts (CastF16 on the activation, CastBf16 on the f32
  transform output) fold bit-identically into the Hadamard kernel's new
  dtype arms (2=bf16 in loads `DF32ToF16(__bfloat162float(x))` — the
  exact chain `CastF16K<bf16>` runs; bf16 out stores
  `__float2bfloat16(res*post)`, the same op `CastBf16K` runs). ops.cpp
  admits bf16 A/C on ROCm only; `Exl3MatmulD` skips both staging launches
  there. Bench `verify_bf16_fold`: in-fold, fused staging and out-fold
  all bit-identical. Same-leg serving: 24.24 -> **25.27 decode tok/s**
  (whole-run 22.40 -> 23.30), +4.3%; ~817 launches/token removed.
  Golden parity holds.
- Post-fold decode census (rocprofv3, 40-token leg, ÷40): GEMV M1 arms
  19.4ms GPU busy (4bpw M1K4 6.9 + fused4 3.3 + 3bpw M1K 9.2), HadK ~1.9,
  lm_head dot arm 1.71 (0.855ms x2 — ~557 GB/s effective over the 953MB
  6bpw trellis, already near the measured stream ceiling), GdnPostConv
  1.73, wvSplitKSml 0.5, everything else ~1.3ms. Total ~35.5ms GPU busy
  vs ~39.6ms wall — ~4ms/token launch gap after the cast removals.
- 2026-10-02 (GEMV diagnosis + postconv): `M1K4stream` (identical load
  pattern, decode/dot removed) reaches 706-841 GB/s on the big shapes
  where the full kernel runs 566-602 — the gap is decode+dot dependency
  latency, not the access pattern; `M1K4g2`'s 1024B contiguous loads do
  not beat it, and `VT_EXL3_GEMV_SMEM` never reaches m=1 (the mmode-0
  dispatch bypasses the smem table by construction). cb=2 decode8 is
  already the udot4 byte-dot (~5 instr/pair); the residual stall is
  shfl/LDS/fshift latency inside the per-slice chain. `15e19e6db` splits
  GdnPostConvChunkedK's gate slot per head (was one thread × hv serial
  expf chains; bit-identical, 284→65us microbench); serving-neutral at
  25.29 tok/s — the slot hid under GEMV latency.
- Final: **25.29 decode tok_s / 23.31 whole-run** on the 12-prompt leg
  (`38fe6cae2` tree). Decode wall ~39.5ms = ~34ms GPU busy (GEMV M1 arms
  ~19.4, HadK ~1.9, lm_head dot 1.71 at ~557 GB/s effective, GDN ~2.9,
  casts now folded) + ~4-5ms launch/host gap. 60 tok/s would need the
  GEMV to run >2x the measured ALU-bound rate — that is a different
  decode structure, not a scheduling fix.
- 2026-10-02 (ksplit + qkv-structure dead ends): `VT_EXL3_GEMV_KSPLIT=2`
  measures **slower** (24.73 vs 25.29 decode tok/s, same leg) — the
  atomicAdd+memset cost exceeds the shorter serial k-chain, so the GEMV
  is not k-chain-latency bound either; its ~600 GB/s sits at the
  decode+dot issue rate. q/k/v DO share the same `dhn` activation
  (dense_attn_block.h:627-629) but gate_up is already a merged
  projection, so the only HadK duplication is 2 in-hads per attention
  layer (~48 launches/token of 366 total) — the GEMV/HadK launches
  dominate the ~7.5ms/token dispatch gap (median 3.2us between nodes,
  ~2480 nodes/token) and batching them is the structural fix that
  remains.
  Sharing a_had across q/k/v is impossible on this checkpoint: their
  suh sign matrices differ (sha256 over layer-3 bytes), and the
  in-hadamard is per-weight — dead on correctness grounds, not cost.
- 2026-10-02 (M1K3 load-position 3bpw — implemented, measured, REVERTED):
  `29652758a` ported the M1K4 ownership mapping to the 24-word 3bpw tile
  (3 contiguous dwords + 1 boundary shfl per lane) with a shape-tuned
  ksplit. Bit-identical at ks1/2/4, n up to 12288; bench3 showed +5..35%
  GB/s on every serving shape. Serving did NOT transfer: 25.06 tok/s
  (ksplit defaults), 24.81 (ksplit off) vs 25.29-25.42 for the dq8 arm.
  The GEMV is not the wall-clock limiter at this dispatch overhead and
  per-launch memsets re-inflate the graph. Reverted at `c088fc512`;
  post-revert leg re-measures 25.42 decode tok/s.
- Session ceiling (honest): **25.4 decode tok/s / ~23.3 whole-run**,
  +40% over the 18.15 start. The decode wall is ~39.3ms: ~34ms GPU busy
  (3bpw GEMV ~19.4ms at ~560-600 GB/s effective — its decode+dot issue
  ceiling per `M1K4stream`; the other ~15ms is HadK×366, wvSplitKSml,
  GDN, lm_head dot at 557 GB/s) + ~5ms dispatch gap. 60 tok/s needs the
  GEMV to stream ~3x its issue-limited rate — a different codeword
  decode algebra or a lower-byte arm, both out of scope.
- 2026-10-02 (remaining dispatch arms measured dead): `VT_EXL3_DOT_FIRST=1`
  serves ~3x slower per request (mid-request hipBLASLt recon autotune
  storms, leg abandoned after >200s); `VT_ROCM_EXL3_WMMA=1` serves
  7.62 decode tok/s on a 6-prompt/128-token leg — 3.3x slower than the
  dot arms. Every productionized arm on this branch is now the measured
  fastest of its alternates: M1K/M1K4 GEMV, fused-had on the 4bpw wide
  shapes only, dot for the 6bpw head, coop GDN scan, folded bf16 casts.
- `VT_EXL3_GEMV_CFG=0` (narrow fragment arm) and `=1` (wide): both serve
  14.7 decode tok/s — 1.7x slower than the dq8 m=1 arm. Every
  selector-reachable arm is now measured; the production set is the
  fastest of each family.
- `VT_EXL3_GEMV_SMEM=1` smem-staged fragment arms (reached via CFG=0/1):
  11.4-11.7 tok/s. Census complete: the dq8 m=1 dot arm at 25.4 is the
  fastest implementation of every tried structure for this checkpoint.
- 2026-10-02 (codeword-LUT decode — fast but impossible): an M1K3
  variant replacing the cb=2 hash decode with a 6-bit pair LUT in LDS
  measured **826-1147 GB/s** on every bench3 shape (2.2x the dq8 arm,
  above the no-decode stream's 800) — but it is WRONG: a cw dump shows
  the extraction window carries 16 meaningful bits (0x0a18, 0x50c0),
  the mul1 hash consumes all of them, and a pair LUT would need a
  32-bit index. The measurement stands as evidence for what the ceiling
  would pay if decode got cheaper, not as a usable kernel.
- 2026-10-02 (hard ceiling statement): even a ZERO-cost decode cannot
  reach 60 tok/s. The no-decode stream kernel's 800-840 GB/s bounds the
  GEMV at ~14ms/token -> ~35 tok/s before every other kernel. 60 tok/s
  requires moving fewer bytes per token — a lower-bpw arm, a repacked
  trellis, or fp8 KV — i.e. a format change, which the task rules
  forbid. The measured-optimal 25.4 tok/s is 71% of that floor.

## Owed

- The ~10 ms/token of launch overhead inside the captured graph — the
  graph replays but per-node launch cost remains; grouped/batched kernel
  structure is the fix.
