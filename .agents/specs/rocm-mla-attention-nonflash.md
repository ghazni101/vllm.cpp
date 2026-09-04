# ROCM-MLA-ATTENTION — the two attention ops GLM-5.3 non-flash refuses on `gfx1151`

Row: `BACKEND-ROCM`
Issue: [#2926](https://github.com/mudler/vllm.cpp/issues/2926)
Base SHA: `d023e3357b907927fb6d459f83d21b4729b78d84`

## Now

`ACTIVE`.

## Scope

Register native `DeviceType::kROCM` kernels for `vt::MlaPrefillAttention` and
`vt::MlaDecodeAttention`, so GLM-5.3 non-flash generates text again on
`strix:gpu0`.

Out of scope, and each recorded under `## Owed`: `kDsaIndexerLogits`,
`kDsaTopkSelect`, `kMergeAttnStates` and `kFusedChain`. §Reached set says why
none of them blocks generation on this workload.

## The reached set, and how it was established

**Generation is all-or-nothing here.** With the reference tier withdrawn, any
ONE unregistered op the forward reaches refuses the step. So the set that
blocks generation is what has to be priced, not the set that is unregistered.

Four ops have **zero** registrations under `src/vt/rocm/` at the base SHA:
`kMlaPrefillAttention`, `kMlaDecodeAttention`, `kDsaIndexerLogits`,
`kDsaTopkSelect`. Only the first two are reached.

**The primary evidence is a measurement, not a reading.** `9f3e6e223`'s GLM-5.3
run on this board, recorded in `.agents/specs/rocm-fused-norm-rope.md`
§Evidence, names the FIVE ops it observed on the then-eligible reference tier:
`ConcatAndCacheMla`, `ConcatMlaNopeRope`, `MlaPrefillAttention`,
`BatchedMatmul`, `MlaDecodeAttention`. That is the model's own forward,
enumerated by `VT_OP_PROVIDER_STATS=1` on the production entry point, on this
part, on this prompt. Three of the five landed native at `928d89a9b` (#2715
W1). **The two that remain are `kMlaPrefillAttention` and
`kMlaDecodeAttention`.**

The code agrees, and says why the others are absent:

| Op | Reached? | Why |
|---|---|---|
| `kMlaPrefillAttention` | **YES, first** | `mla_attention.cpp` §5a `ForwardMlaPrefillMha`, on every prefill step |
| `kMlaDecodeAttention` | **YES, second** | `src/vllm/v1/attention/backend.cpp:343`, from §5b, on every decode step |
| `kDsaIndexerLogits` / `kDsaTopkSelect` | no | guarded by `run_indexer = dims.has_indexer() && sparse_step`, and `sparse_step = !meta.indexer_cu_seqlens_q.empty()` (`mla_attention.cpp:493-494`). That mirrors upstream's `use_dense_mha = prefill_max_seq_len <= self.topk_tokens` (`sparse_mla_attention.py:296-299`): below the threshold the top-k selects every causal candidate and DENSE attention is upstream's own answer. A short prompt is a dense step. |
| `kGatherMlaCache` | no (and already native) | chunked context only (`has_context = !meta.chunks.empty()`); a fresh prompt has none. Landed at `928d89a9b`. |
| `kMergeAttnStates` | no | same chunked-context branch |
| `kFusedChain` | no | Tier-1 interpreter, reached only at `VT_FUSED_TIER=1` (`src/vt/ops.cpp:1287-1298`, `include/vt/fused_recipe.h:176-179`), which is a rejected performance arm on every backend that measured it |

**ORDER.** `kMlaPrefillAttention` refuses first: the first forward of a fresh
prompt is prefill-only (`decode_toks == 0`), so §5b never runs. Landing prefill
alone therefore moves the failure from step 1 to step 2, which is visible
progress and is what the gate ladder below measures.

## Why it refuses rather than running slowly

`6b97a6800` (#2511) narrowed managed allocation to `PageableMemoryAccess == 1`
and made the unified claim FOLLOW the allocator
(`include/vt/rocm/rocm_arch.h:180-195`). `gfx1151` reports 0.
`docs/ROCM.md:83-85` states that such a part gets plain `hipMalloc` and no
reference tier. Measured by the probe case landed with #2843:

```text
ROCm reference tier: UnifiedMemory=0 DeviceMemoryIsHostAddressable=0
                     ReferenceTierEligible=0
VERDICT: the tier is WITHDRAWN here — the missing MLA ops are a REFUSAL,
         so W2/W3 block GENERATION and not only measurement
```

`9f3e6e223`'s ` Paris, which is` is not falsified — it is true of a tree that
does not contain `6b97a6800`. `git merge-base --is-ancestor 6b97a6800
9f3e6e223` answers no, and `6b97a6800` IS an ancestor of `main`.

## Pricing — and the correction it makes to the campaign spec

`.agents/specs/rocm-mla-dsa-ops.md` prices `kMlaDecodeAttention` at ~600 HIP
lines, **high risk**, from a 560-line CUDA donor. That price is for porting the
CUDA arm, which is the two-stage split-KV flash decode
(`cuda_mla_attn.cu:140-750`), and the four risks it names are all properties of
the SPLIT: a >48 KiB dynamic shared-memory request with no HIP opt-in, a
`multiprocessor_count` probe `DeviceCaps` does not carry, a grow-only
capture-safe `[B, Hq, splits, Dv+1]` workspace, and a sink that stage 2 must
count once per row rather than once per split.

**None of those is a property of the OP.** vLLM's behaviour is the MQA decode
over the compressed latent (`triton_mla.py:189-260 forward_mqa`), and this tree
already carries a second, independent port of it that has none of the four
risks: `src/vt/cpu/cpu_mla_attn.cpp`, itself a port of upstream's OWN CPU
kernel `vllm/csrc/cpu/mla_decode.cpp` (`mla_decode_kvcache_cpu_impl`). A ROCm
arm written to that shape — one block per `(request, head)`, one streaming
online-softmax pass over the keys — is a native kernel that computes the same
function, and the split disappears along with every one of its four risks.

| Op | Shape | HIP est. | Test est. | Risk |
|---|---|---|---|---|
| `kMlaPrefillAttention` | one block per `(query row, head)`, streaming online softmax over the visible keys; causal bottom-right bound + optional sliding window | ~150 | ~120 | low |
| `kMlaDecodeAttention` | one block per `(request, head)`, streaming online softmax over the block-table walk; window + DSA selection + attention-sink arms | ~180 | ~150 | low-medium — three optional arms, each its own gate |
| **Total** | | **~330** | **~270** | |

The two kernels are the SAME kernel with two different key enumerations, which
is why they share one translation unit.

**What this costs.** The split-KV schedule exists to fill a GPU when
`batch * heads` is small; a single-pass kernel leaves the tail of a small decode
under-occupied. That is a SPEED property, and no speed number is admissible
from this board for this model anyway (`docs/ROCM.md:60-61` — and see §Owed:
the DSA pair is still unregistered, so a sparse run would still refuse). The
split stays owed as a performance wave, named rather than silently skipped.

**This spec does not retract the campaign spec's estimate.** It measured the
CUDA port, and it is right about the CUDA port. What it did not price is the
CPU port, which is the cheaper of the two mirrors and reaches the same
behaviour.

## Upstream anchors

vLLM at the pinned oracle `5559679229` (`.agents/upstream-sync.md`).
`GlmMoeDsaForCausalLM` is `vllm/model_executor/models/deepseek_v2.py:1930`
(**not** `deepseek_v32.py`; at this pin that file carries no registration).

| Op | Upstream behaviour | Structural donor in this tree |
|---|---|---|
| `kMlaPrefillAttention` | `vllm/v1/attention/backends/mla/prefill/flash_attn.py:153-248` — `_flash_attn_varlen_diff_headdims`, `run_prefill_new_tokens` (causal) and `run_prefill_context_chunk` (non-causal). The V zero-pad to the QK width and the output slice-back (`:164-168`, `:196-197`) are a launcher detail; computing at the true widths is the same number. | `src/vt/cpu/cpu_mla_prefill.cpp` (the numerics), `src/vt/rocm/rocm_mla_fused_norm_rope.hip:98-116` (the shared-memory reduction tree) |
| `kMlaDecodeAttention` | `vllm/v1/attention/backends/mla/triton_mla.py:189-260` `TritonMLAImpl.forward_mqa`, which passes ONE buffer as both K and V with `is_mla=True` (`:236-244`); NUMERICS from `vllm/csrc/cpu/mla_decode.cpp` `mla_decode_kvcache_cpu_impl` | `src/vt/cpu/cpu_mla_attn.cpp` (the numerics), `src/vt/cuda/cuda_mla_attn.cu:255-266,296-320,398-407` (the three optional arms' exact rules) |

**There is no vendored FlashAttention to mirror on ROCm**, and that is the one
place this port is not a transcription. `cuda_mla_prefill.cu`'s
`kMlaPrefillAttention` arm is inside `#ifdef VLLM_CPP_FLASH_ATTN` and is the
FA-2 launcher, so the ROCm arm is written against the CPU reference — exactly as
`.agents/specs/rocm-mla-dsa-ops.md` §Owed already records it must be.

## Design

One new translation unit, `src/vt/rocm/rocm_mla_attn.hip`, two kernels, two
entry points, two `RegisterOp` lines in `rocm_ops.hip`, two `CMakeLists.txt`
entries (the source list and the `HIP_ARCHITECTURES` property list).

Both kernels share one body shape:

- **Grid** is one block per output row. Prefill: `(total_q, num_heads)`.
  Decode: `(batch, num_heads)`. **Block** is 256 threads — four whole
  wavefronts, the width `rocm_rmsnorm.hip:41` fixes and for the reason stated
  there.
- **Dynamic shared memory** holds `q[head_dim]`, `acc[v_head_dim]` and a
  `float[256]` reduction scratch. At GLM-5.3 geometry that is
  `(576 + 512 + 256) * 4 = 5,376 B`, an order of magnitude under the 48 KiB
  every architecture guarantees. This is the whole reason the CUDA arm's
  `cudaFuncSetAttribute` risk does not appear: the split kernel tiles
  `(kBlockH + kNTile)` whole rows into shared memory and this one holds one.
- **The dot product** is the `__syncthreads()` shared-memory tree reduce
  `rocm_rmsnorm.hip` and `rocm_mla_fused_norm_rope.hip` already use. It uses no
  warp-level primitive, so it is wavefront-width agnostic — the property that
  makes it correct on a 64-lane wavefront without a `warpSize` assumption.
- **The softmax is streaming online**, key by key, in the CPU reference's own
  order: `m_new = max(m, qk)`, `rescale = isinf(m) ? 0 : exp(m - m_new)`,
  `acc = acc * rescale + p * v`, `l = l * rescale + p`. Since the key loop is
  sequential within the block, the reduction order over KEYS is identical to
  the CPU oracle's, and only the order over the head dimension differs. That is
  what makes an NMSE bound the right gate rather than a hope.

Three arms carry over verbatim, each cited to the line that defines it:

1. **Sliding window.** Prefill: `first = clamp(iq + causal_shift - win_left)`
   against the same bottom-right position the causal bound uses
   (`cpu_mla_prefill.cpp`, the `first` computation). Decode:
   `j_start = max(0, seq_len - 1 - win_left)` (`cuda_mla_attn.cu:196`).
   `win_left < 0` is the absent state and restores the byte-identical full
   range.
2. **DSA selection** (decode only). `sel`/`sel_cnt` present maps visit index
   `i` to token position `sel[b * sel_s0 + i]`, count clamped
   `min(topk, max(0, sel_cnt[b]))` — the kernel contains what `ops.cpp` could
   not validate on device memory (`ops.cpp`'s own note). A `-1` or
   out-of-range position scores nothing. **This mirrors the CUDA arm, not the
   CPU one**, and the divergence is upstream of this change: on an
   out-of-range position `cpu_mla_attn.cpp` REFUSES by name where
   `cuda_mla_attn.cu:281-289` returns a number. A device kernel cannot throw,
   so the CUDA rule is the only one this arm can take, and the divergence stays
   recorded in `.agents/specs/dots3-note.md` `## Owed` where it already lives.
3. **Attention sink** (decode only). Seeded ONCE per row: `m = sink[h]`,
   `l = 1`, `acc = 0` (`cpu_mla_attn.cpp`, and `cuda_mla_attn.cu:398-407` for
   why the split arm has to put it in stage 2 instead). A single-pass kernel has
   no per-split double-count to avoid, which is the second of the four W3 risks
   dissolving rather than being solved.

`MlaDecodeAttentionArgs::num_kv_splits` is **ignored** by this arm and that is
stated rather than implied: it is a scheduling hint for a split kernel, and
`ops.cpp` validates only `>= 0`. The CUDA arm's own heuristic overrides it too
when it is 0.

Dtypes are `f32` and `bf16`, the two `rocm_mla_fused_norm_rope.hip` carries and
the two a GLM-5.3 GGUF arm produces. `f16` refuses BY NAME rather than
silently, so a caller meets a message and not a wrong answer.

## Reachability

The production entry points are
`src/vllm/model_executor/layers/attention/mla_attention.cpp` §5a
(`ForwardMlaPrefillMha` -> `vt::MlaPrefillAttention`) and
`src/vllm/v1/attention/backend.cpp:343` (`vt::MlaDecodeAttention`), both reached
from `ModelRegistry::Forward` through `vllm_engine_load` and `vllm-cli`.
Nothing is added to the model layer: this change alters which implementation
`GetOp` returns for a call the model already makes on every MLA step.

The mutation that proves it is deleting each `RegisterOp` line, REBUILDING, and
rerunning the e2e leg: it must refuse again, by name, exactly as `main` does.

## Tests and gates

New arms in `tests/vt/test_backend_cross_device.cpp`, in the file's existing
style. Each asserts THREE things, and the middle one is load-bearing:

1. The device result matches the CPU oracle at NMSE <= 5e-4.
2. `vt::OpRegistered(op, device)` is **true**. `OpRegistered` is a native-only
   probe (`src/vt/op_provider.cpp:788-806`), so it is the only one of the three
   that can tell a native kernel from the reference tier.
3. `vt::GetReferenceTierHits()` does not increase across the call — the same
   quantity `docs/ROCM.md:60-61` disqualifies a performance result on.

**The #2715 wave's own trap is why assertion 2 exists.** Four of its RED cases
PASSED on their `REQUIRE` guards alone, because the device loop bodies never
ran and the oracle-equality assertion is green with no kernel at all. On this
board the tier is withdrawn, so the shape is different — but the RED must still
be read as BOTH counts, and a non-zero assertion count is what proves the cases
asserted rather than threw.

Arms:

- Prefill, causal, multi-request varlen with UNEQUAL query and key lengths, so
  the bottom-right `causal_shift` is non-zero and a top-left implementation
  fails.
- Prefill, non-causal (the context-chunk call, `flash_attn.py:246`).
- Prefill, sliding window.
- Prefill LSE against the oracle's `[num_heads, total_q]` layout.
- Decode, dense, block table with SHUFFLED blocks so the walk is exercised.
- Decode, sliding window.
- Decode, DSA selection, including the FULL-selection identity case: a
  selection naming every causal key must reproduce the dense answer.
- Decode, attention sink.
- Decode LSE.

Gate ladder on `strix:gpu0`, in this order, because each step moves the failure
further along:

1. Focused unit target, case AND assertion counts both read.
2. GLM-5.3 through `vllm-cli` on the production entry point at the tree with
   ONLY the prefill kernel: the refusal must move from `MlaPrefillAttention` to
   `MlaDecodeAttention`.
3. The same with both kernels: **generated text printed verbatim**, with
   `GetReferenceTierHits()` beside it.
4. Mutation ladder, each REBUILT and each restored by sha256.

## Risks

- **A long prefill is slow.** One block per `(query row, head)` with a
  whole-block reduction per key is `O(len_k)` block reductions per output row.
  It is correct and it is not the split schedule. No speed number is admissible
  here, and the split stays owed.
- **`MoeSiluMul` bf16 is a standing red on this board** (±1 ULP vs the CPU
  oracle, #1954/#1513). It is pre-existing, this change touches no MoE
  arithmetic, and it is REPORTED rather than tolerated into green.
- **`-Werror` reaches HIP translation units** since `6f6caa725`, so a
  diagnostic in the new file is a build failure. A diagnostic in code this
  change did not write is `main` red under its own flag and is reported as
  such.

## Stop conditions

- A numeric arm that needs a widened tolerance to pass stops the wave. The
  bound is the one every other GEMM/attention arm in this file uses.
- The board faults or resets in a way that is a property of the board rather
  than of this change (#2546 measured 12/12 GPU resets for a gate-sized native
  run). Report it as such.
- A THIRD op turns out to block generation. Return the op's name and stop;
  §Reached set would then be wrong and has to be corrected before more code.

## Owed

- `kDsaIndexerLogits` + `kDsaTopkSelect` on ROCm — #2715's W2, still owed. Not
  generation-blocking on a dense step; a SPARSE step (a prompt longer than
  `index_topk`) still refuses. Owner `BACKEND-ROCM`, issue
  [#2926](https://github.com/mudler/vllm.cpp/issues/2926).
- `kMergeAttnStates` on ROCm — chunked-context prefill only. Owner
  `BACKEND-ROCM`, issue [#2926](https://github.com/mudler/vllm.cpp/issues/2926).
- `kFusedChain` on ROCm — non-gating (`VT_FUSED_TIER=1` only). Owner
  `BACKEND-ROCM`, issue [#2926](https://github.com/mudler/vllm.cpp/issues/2926).
- The **split-KV decode schedule** on ROCm, as a performance wave. Owner
  `BACKEND-ROCM`, issue [#2926](https://github.com/mudler/vllm.cpp/issues/2926).
- The ROCm GLM-5.3 speed axis stays **VOID**.
