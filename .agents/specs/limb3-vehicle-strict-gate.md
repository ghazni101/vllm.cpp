# Limb 3 has a vehicle now: pin it, fetch it, and run the strict gate

Row: `QUANT-QWEN38-27B-GGUF-ARM`
Issue: [#2884](https://github.com/mudler/vllm.cpp/issues/2884)
Refs: [#2864](https://github.com/mudler/vllm.cpp/issues/2864),
[#2854](https://github.com/mudler/vllm.cpp/issues/2854),
[#2740](https://github.com/mudler/vllm.cpp/issues/2740),
[#2497](https://github.com/mudler/vllm.cpp/issues/2497),
[#2534](https://github.com/mudler/vllm.cpp/issues/2534)

## 0. Now

Filled in by §12 when the gate has run. Until then this spec is the design and
the pin, and no row changes lifecycle state. `STRIX_ARM_SPEED_RATIFIED_BY`
stays unset and
[`../scripts/rocm-strix-ourarm-staged.sh`](../scripts/rocm-strix-ourarm-staged.sh)
stays refusing whatever this gate returns, because limb 2's denominator
question is [#2534](https://github.com/mudler/vllm.cpp/issues/2534)'s and not
this spec's.

## 1. Scope

[`q4km-limb3-kquant-vehicle.md`](q4km-limb3-kquant-vehicle.md) (#2864, landed
`465cd27e3`) pre-registered six conditions, read all 77 GGUFs this fleet holds
header-first, and found the intersection **empty**. Its §7 named three ways a
vehicle could come to exist and closed option 2 with:

> in practice this means a second dense `qwen35` checkpoint that is not
> Qwen3.8-27B. Whether one exists **was not established here, because fetching
> it needs recorded authority and none was given.**

The authority is now recorded in
`.agents/developer-preferences.md` (developer,
2026-09-04), scoped to ONE vehicle meeting exactly those six conditions. This
spec spends it.

**In scope:** identify and pin a candidate, verify it against #2864's own
predicate before fetching it, stage it, and score one STRICT free-running
token-exact gate of our ROCm k-quant path against the pinned vLLM on
`strix:gpu0`.

**Out of scope, and measured to be out of scope.** No throughput, latency or
memory figure for either engine, and no cross-engine ratio. `AGENTS.md` §Gates
admits no performance result from an arm whose declared token gate has not
passed, and #2497 already carries one retraction for exactly that. This spec
rescores nothing in #2497, #2534, #2546, #2740, #2809, #2854 or #2864, and it
changes no file under `src/`, `include/` or `tests/`.

## 2. The candidate, and why it is the only one

The six conditions, unchanged from #2864 §2, are: k-quant tensors; an
architecture in `kGgufArchArms`; a family the **pinned** vLLM registers; dense;
fits the board's ~62.8 GB free carve; and not the arm's own model.

`kGgufArchArms` dispatches eight architectures. The pinned vLLM registers five
of them, and four of those five are MoE in every checkpoint the ecosystem
publishes, so condition 4 leaves `qwen35`. `vllm-gguf-plugin`'s
`_ADAPTER_REGISTRY` is `Gemma3`, `Gemma4`, `OLMoE`, `Qwen35`, `Qwen35Mtp`, and
this tree has no Gemma or OLMoE GGUF arm, which narrows it the same way from
the other side. The search space is therefore **dense `qwen35` checkpoints that
are not Qwen3.8-27B**, and the family publishes exactly two: `Qwen3.5-27B` and
`Qwen3.6-27B`.

`Qwen3.6-27B` is chosen over `Qwen3.5-27B` because its `text_config` key set is
**identical** to the arm's, `output_gate_type`, `partial_rotary_factor` and
`tie_word_embeddings` included, while `Qwen3.5-27B` carries none of the three.
The closer the config surface, the more of the same forward code both sides
execute, which is the whole point of limb 3.

**The pin is a revision, not a repo id.** #2497 refused the UD family because
`Qwen3.8-27B-UD-Q4_K_XL`'s published bytes moved in place under an unchanged
name. The plain `Q4_K_M` arm is taken rather than a `UD-*` one for the same
reason and because it is the quant tier the arm itself runs.

## 3. Method

### 3.1 Before the download

[`../../docs/bench-evidence/limb3-vehicle-pin-20260904/`](../../docs/bench-evidence/limb3-vehicle-pin-20260904/).
`vehicle_pin_check.sh` re-runs the whole determination.
`remote_gguf_header.py` reads the candidate's own header **by HTTP range
request**, and it drives #2864's committed `gguf_header.py` rather than
re-implementing the parse, so the two searches cannot disagree about what a
header says.

The four-surface oracle check of #2864 §5 is re-run for the candidate. There
the expected answer was negative and the control was positive; here the
expected answer is positive, so the control is the **negative** one:
`muse-glimmer` is re-probed on the identical four surfaces and must still read
0. A grep that matched everything would light both rows and be visible.

### 3.2 The gate

[`../../docs/bench-evidence/limb3-strict-gate-20260904/`](../../docs/bench-evidence/limb3-strict-gate-20260904/).
`gate.sh` is the `rc` job, `gen_vehicle.py` is the oracle's side and
`score_strict.py` is the verdict. One lease on `strix:gpu0`, nothing by `ssh`,
`HSA_OVERRIDE_GFX_VERSION` set nowhere, and the job refuses to start if it
inherited any `HSA_*`, `ROCR_*`, `PYTORCH_*`, `HIP_*` or `VT_*` variable.

**Both sides are REUSED, and each is asserted to be the object it claims.** The
`strix:gpu0` worker has not rebooted since 2026-09-01 (`boot_id`
`a5bc8128-f6ad-4767-8614-6923f88032e1`), so `/tmp` still carries the
token-gate-v2 build and the #2740 vLLM venv. `gate.sh` asserts our three build
products against the sha256 values committed in
[`../../docs/bench-evidence/qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md`](../../docs/bench-evidence/qwen38-27b-q4km-rocm-gfx1151-token-gate-v2-20260902.md)
and asserts the venv's `vllm.__version__`, its device, its `RocmPlatform`
resolution and its plugin extension before it generates anything. A rebuild
would fail those assertions rather than pass silently.

### 3.3 What "strict" means, declared before the run

Free-running greedy decode, `ignore_eos`, batch 1, MTP off, **48 tokens on each
of the six pre-registered prompts**. Every token of every prompt, 288 steps.
Not teacher-forced, and no near-tie band: a rank-2 token is a divergence here.

**The prompts are pre-registered in the strongest sense available.** They are
not chosen by this spec. They are the six the declared Q4_K_M gate has scored
since 2026-08-23, `prompts_sha256`
`c8a5080046c3206a1186b42a320a21e887691eff43d2116364192f58e5776c7e`, a value
committed in three evidence documents that predate this run. `gate.sh` hashes
the file and refuses on a mismatch before it scores anything, and
`score_strict.py` refuses again independently.

The prompt **token ids** are produced by our engine from the vehicle's own GGUF
vocab and FED to vLLM, and vLLM's own tokenizer re-derives them and prints
whether it agrees. #2740 established the pattern: a generation comparison must
not be contaminated by a tokenizer difference.

### 3.4 The oracle's determinism is a PRECONDITION, not a footnote

#2740 measured, on the arm's own artifact, that the pinned vLLM's eager and
compiled configurations are each self-reproducible and **disagree with each
other on 2 of 6 prompts**. A vehicle inherits nothing from that measurement, so
it is re-measured here, and three conditions must all hold:

1. `EAGER1_EQ_EAGER2`
2. `COMPILED1_EQ_COMPILED2`
3. `EAGER1_EQ_COMPILED1`

plus a fourth that no earlier run in this campaign has taken: the oracle's
one-pass **prefill argmax** over (prompt + its own generated tokens) must
reproduce the tokens its incremental decode emitted.

If (3) fails, the answer is `STRICT_LIMB3=NO` with the reason "the oracle is not
deterministic on this vehicle". **Picking whichever configuration agrees with
us is forbidden**, and it is forbidden in the scoring script rather than in
prose: `score_strict.py` evaluates the oracle's self-consistency first and
short-circuits the verdict on it.

## 4. Risks

**The reused binaries could be the wrong object.** Mitigated by asserting three
sha256 values that were committed by a different run, on a branch this job
cannot edit.

**A dead instrument reads as a result.** Every leg is counted into one of three
outcomes -- OK, board fault, harness error -- and a run with fewer than two
clean legs returns `NOT_MEASURED` rather than a verdict. Our legs additionally
assert zero `[vt reference-tier]` hits and at least one `device=5` selection, so
a leg that silently ran the CPU tier under a ROCm label fails instead of
scoring.

**The board faulted 17 times in 18 legs on this workload before `27da7787e`.**
The `LEGS` budget is a scheduling parameter and not a verdict: an all-fault run
is `NOT_MEASURED`, which is a complete answer.

**The vehicle is not BIGGER than the arm.** The ratification's wording is "a
BIGGER dense model"; `Qwen3.6-27B` is 64 blocks against the arm's 65 (the arm's
65th is its `nextn` drafter) at identical width. No dense `qwen35` checkpoint
larger than 27B exists, so the strongest available vehicle is a same-class
sibling. **This is recorded as a shortfall rather than argued away**: a pass
here satisfies the six conditions #2864 pre-registered and does not by itself
satisfy the word "bigger", and whether that is enough is a ratification
decision this spec does not make.

## 5. Tests

None. No product code changes, so there is nothing a test could reach. The
re-runnable artifacts are `vehicle_pin_check.sh` and `gate.sh`, whose outputs
are committed verbatim beside them.

## 6. Gates

Run by name on this head, each exit code read from the process:

- `scripts/check-agent-record.py`
- `scripts/check-commit-style.py --range origin/main..HEAD`
- `scripts/check-commit-trailers.py --range origin/main..HEAD`
- `scripts/check-pr-size.py --base origin/main --head HEAD`, which
  `agent-preflight.sh` skips because it supplies no `--base`/`--head`

## 7. Stop conditions

- No candidate passes §2 before the download: stop and report. Do not fetch
  something that fails a condition in order to have fetched something.
- The oracle is not self-consistent on the vehicle: report that, and do not
  score against the half that agrees.
- Fewer than two clean legs on either side: `NOT_MEASURED`.
- A divergence is a complete answer. **`STRICT_LIMB3=NO` is a result, not a
  failure of this spec**, and it is more useful than a stretched pass.

## 8. Owed

Nothing yet.
