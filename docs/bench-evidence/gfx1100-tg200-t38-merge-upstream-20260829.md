# GFX1100-TG200 T38 — upstream sync (74 commits) + post-merge correctness gates (2026-08-29)

## Question

Does `row/GFX1100-TG200` stay correct after syncing `upstream/main`
(`e551cf8e4` -> `3015aad08`, 74 commits) into the campaign branch, and is
the campaign reference preserved bit-for-bit under the adopted-lever block?

## Change

- `65d781e69` merge upstream/main into row/GFX1100-TG200. Three content
  conflicts, all in the op seam, all additive: `kPermuteVHeads` (ours, T25)
  vs `kCastF16` (upstream, QUANT-EXL3 W1a #2181) inserted at the same
  anchor of `include/vt/ops.h`, `src/vt/cpu/cpu_ops.cpp`, `src/vt/ops.cpp`.
  Resolved keep-both. No campaign lever rebased; evidence-commit SHAs intact.
- `69bd0f035` fix: the shared closer had been unified by git across both
  conflict sides, leaving `PermuteVHeadsKernel` and `PermuteVHeads`
  unclosed; adds the two missing `}`. Caught by the build (red-first).

## Inherited upstream break (NOT introduced by the merge)

`tests/vllm/model_executor/test_placed_moe_roundtrip.cpp` references
`vllm::RunMoeBlockPlaced`, which exists NOWHERE on upstream/main either
(git grep across include/ and src/: only the test, the spec, and the issue
index). Upstream's own `hybrid-placement.md` (:442-444) records that W3c
removed the helper as dead code; the W3b test (:550 evidence claim) was
left behind. Upstream `1029f201e` lineage does not compile its test suite;
the merged branch inherits that verbatim. Build uses `-k 0`; this is the
ONLY failing target. Fixing it here would mean deleting upstream's test,
which is upstream's call, not this row's.

## Gates (container `rocm-dev:10.0.0`, HIP 7.15, gpu-ctl lock held)

Build: `cmake --build build-hip-docker -j 16 -- -k 0` in-container over
`/repo/tg200` (the configured source root), Release, `-Wall -Wextra
-Werror` — all targets green except the inherited break above.

`ctest -R 'rocm|quant' --output-on-failure`: 24 tests, 22 passed, 2 failed:

- `test_rocm_quant_dot`: 841/841 assertions, 2 case-level throws at
  `test_rocm_quant_dot.cpp:1603` (`matmul_bt_quant`, iq2_xxs) and `:1699`
  (`matmul_bt_quant_grouped`, iq2_xxs) — the documented UNPORTED-dtype
  throws (ported: Q8_0/Q4_K/Q5_K/Q6_K; owed: Q4_0/Q2_K/Q3_K/IQ*), unchanged.
- `test_gguf_keep_quant`: 34/42 cases, 18 assertion failures, ALL in
  `RouteGgufTensor(...) == kKeepQuant` host-routing checks for
  IQ2_XS (id 17) / IQ4_XS (id 23) / IQ4_NL (20u) roles. Inherited from
  upstream verbatim: `git diff upstream/main HEAD` on
  `gguf_keep_quant.{h,cpp}` is EMPTY, and `8e2f56cb1..upstream/main`
  changed ONLY the test file (+101/-34), not the router — upstream landed
  stricter expectations without the router change.

`test_rocm_prefill_tile` PASSED (the 720/720 byte-identity gate);
`test_rocm_skinny_f32`, `test_rocm_fp8_kv_cache`, `test_gemma4_rocm_fp8_seams`
passed. Both reds are the documented pre-existing set, now traced to their
upstream origin rather than assumed.

`ctest -R 'cross_device' --output-on-failure`: 2 tests, 1 passed, 1 failed:

- `test_backend_cross_device_vt_attn_decode_d128`: PASSED.
- `test_backend_cross_device`: 24/26 cases, 5/80253 assertions failed. Two
  cases fail, both PRE-EXISTING campaign bugs (not merge regressions):
  verified by building and running the pre-merge head `8e2f56cb1` in a
  fresh worktree — identical result (24/26, 5 assertions). The campaign's
  focused gate `ctest -R 'rocm|quant'` never included `cross_device`, so
  these were never run before.
  - `MoeSiluMul` bf16 exact-equality (`test_backend_cross_device.cpp:2067`):
    GPU bf16 output != CPU oracle bf16. The campaign added the ROCm
    `MoeSiluMulKernelRocm` in `rocm_moe_router.hip`; the rounding differs.
  - `decode-skinny MatmulBT (wvSplitK path)` sentinel check
    (`test_backend_cross_device.cpp:2231`): the wvSplitK kernel writes past
    M*N elements for shape `{tok=2, k=256, feat=254}` (the "even below bound:
    takes skinny" case). `got[i] == 0xBD44` instead of `0xDEAD` — the
    campaign's YTILE=2 default overwrites the guard band. The float-tolerance
    check at `:2227` also fails for the same shape.
  These are owed a fix in a separate unit of work (issue needed per
  AGENTS.md). The merge introduced zero new failures.

## Acceptance identity (the row's own token gate)

Command: `build-hip-docker/examples/vllm-cli --model
/models/vllm.cpp/Qwen3.5-4B-Q4_K_M.gguf --prompt "$(cat
tools/tg200-prompt.txt)" --max-tokens 256 --temperature 0 --seed 0
--repeat 5`, all 15 adopted levers exported (verified `env | grep -c ^VT_`
= 15 in-container), gpu-ctl held, single process.

- 109 prompt tokens, 256 completion tokens, finish=length on every rep.
- body md5 `a0fa1c4aa8cc5de086006111dad7a7bf` ==
  `tools/tg200-reference.body.txt`; `diff` byte-identical. The re-minted
  reference survives the 74-commit merge bit-for-bit.
- tok/s by rep: 86.19 (warm-up) / 95.19 / 95.21 / 95.05 / 95.16; median of
  5 = 95.05. CAVEAT: window was not pristine-idle (loadavg 4.3 decaying
  from the build; T37's windows started at 0.42). The identity claim is
  load-independent; the +11% vs the documented 85.8 position is an
  OBSERVATION, not a claim — it owes a clean idle-window A/B before any
  attribution (candidate: upstream decode-path changes riding the merge).

## Near-tie adjudication (teacher-forced logprob band)

Command: `tools/tg200-neartie.sh adjudicate @levers -- --model
/models/vllm.cpp/Qwen3.5-4B-Q4_K_M.gguf --prompt-file
/repo/tg200/tools/tg200-prompt.txt --ref-ids
/repo/tg200/tools/tg200-reference.ids.i32 --json
/repo/tg200/t50-postmerge-neartie.json --expect-md5
a0fa1c4aa8cc5de086006111dad7a7bf --note "post-merge identity"`,
container `rocm-dev:10.0.0`, HIP 7.15, gpu-ctl lock held.

Result: **PASS** — `verdict=PASS divergent=0 over_band=0
max_gap_mnats=0.000 body_md5=a0fa1c4a…`. No divergent positions
(argmax == reference at every step). The teacher-forced walk of the
256 reference ids under the merged build reproduces the body md5
bit-for-bit. This is the per-step bit-exactness ceremony the 9
non-bit-identical levers owe (`rocm-m4-oracle.md` band <= 500 mnats):
all 15 adopted levers ON together, the reduction-order composite
produces zero divergence from the reference. JSON:
`t50-postmerge-neartie.json`.

## Verdict

Correctness PASS: focused gates at the documented baseline, campaign
reference byte-identical, near-tie adjudication PASS (0 divergent, 0
over band, max gap 0.000 mnats), branch synced to upstream tip
`3015aad08` at `69bd0f035`. The 9 non-bit-identical levers are covered
by the all-levers-ON near-tie: the reduction-order composite produces
zero per-step divergence from the reference under the merged build.

Owed upstream: the `test_placed_moe_roundtrip` / `RunMoeBlockPlaced`
removal belongs on mudler/vllm.cpp, not here.

Owed here (pre-existing, NOT merge regressions):
- `test_backend_cross_device` 2 failing cases (MoeSiluMul bf16 rounding,
  wvSplitK YTILE=2 OOB) — verified identical at pre-merge `8e2f56cb1`;
  the campaign's focused gate never included `cross_device`. Needs an
  issue and a separate fix unit.
- record-anchor ratchet stale=29 vs baseline 28 — identical set at
  `8e2f56cb1`; the +1 predates this work.

Repaired in-flow: `check-env-doc` by allowlisting the three TG200
tuning knobs; commit-trailer contract by rebuilding the 5 sync commits
with the bare `FOLLOWING_AGENTS_PROTOCOL` paragraph.
