# NVFP4 KV cache (`cache_dtype=nvfp4`) — the spike, and why W1 is a refusal

Row: `KV-NVFP4-TURBO` ([`.agents/engine-matrix.md`](../engine-matrix.md), KV
cache and memory). Issue: [#2620](https://github.com/mudler/vllm.cpp/issues/2620).

Pinned oracle vLLM `5559679229bc961848b121ccdeaa8fa5d79bec98` (the parity pin in
[`upstream-sync.md`](../upstream-sync.md)) at the checkout `.env` `VLLM_SOURCE`
names. Every `file:line` below is in that tree and was read there, not
transcribed from a summary.

## Why the row exists

`--kv-cache-dtype nvfp4` is refused by name.
[`include/vllm/v1/kv_cache_dtype.h`](../../include/vllm/v1/kv_cache_dtype.h)
implements `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3` and `fp8_e5m2`, and
every other `CacheDType` member falls through to a `VT_CHECK(false, ...)`.

The gap is not academic. The published recipe for the #2495 benchmark target —
`Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw` — serves the pair with `-cq nvfp4`,
and [`docs/benchmarks/qwen38-27b-exl3-gb10.md`](../../docs/benchmarks/qwen38-27b-exl3-gb10.md)
carries the KV dtype as one of the two remaining unmatched axes of that
comparison.

## Scope

- **In (this change, W0):** the spike that measures what serving `nvfp4` costs;
  the 1:1 port of upstream's `KVQuantMode` / `get_kv_quant_mode`
  (`vllm/v1/kv_cache_interface.py:33-84`) and of the packed-page dimension
  `nvfp4_kv_cache_full_dim` (`vllm/utils/torch_utils.py:414-416`); and a refusal
  that names, per `CacheDType` member, the row that owes it and what is missing,
  instead of one fallthrough string shared by six unrelated gaps.
- **Out, and owed below:** the packed page, the quantizing store, the
  dequantizing paged-attention read, and the runner wiring. W1/W2/W3 in the port
  map own them. This spec does **not** schedule them, because two of the three
  findings below have to be settled by a decision or by hardware first.

## Upstream chain, measured

The `CacheDType` Literal carries `nvfp4` (`vllm/config/cache.py:19-35`), and
three separate pieces implement it:

1. **The mode map.** `KVQuantMode` (`vllm/v1/kv_cache_interface.py:33-60`) and
   `get_kv_quant_mode` (`:62-75`) turn the string into an enum so backends
   dispatch without string matching; `nvfp4` is `KVQuantMode.NVFP4 = 5`.
   `is_quantized_kv_cache` and `kv_cache_uses_per_token_head_scales` are then
   derived from it (`:77-85`). **Upstream carries a second, older copy of those
   two predicates as string tests** (`vllm/utils/torch_utils.py:75-85`), and it
   is the copy `vllm/config/cache.py:12-15` imports. The two agree on the whole
   current Literal; this tree already mirrors the string copy
   (`src/vllm/v1/attention/backend.cpp:106`) and now mirrors the enum copy
   beside it, rather than picking one and calling the other a divergence.

   **The enum was already here and the map was not, which is the interesting
   half.** `KVQuantMode` landed with `AttentionSpec::kv_quant_mode` "for field
   fidelity" and sat in `include/vllm/v1/kv_cache_interface.h`. Nothing ever
   produced a value for it: no `get_kv_quant_mode` existed, so every spec in the
   tree carried the `kNone` default and the field could not disagree with
   anything. A mirrored field with no mirrored producer reads as ported and
   measures nothing. This change moves the definition down into
   `kv_cache_dtype.h`, beside the string it is resolved from, and gives it the
   producer.
2. **The page geometry.** `nvfp4_kv_cache_full_dim(head_size)` is
   `head_size // 2 + head_size // 16` (`vllm/utils/torch_utils.py:414-416`):
   `head_size/2` bytes of packed fp4 data (two E2M1 values per byte) plus
   `head_size/16` bytes of E4M3 block scales, one scale per 16 elements.
   `AttentionSpec.real_page_size_bytes` branches on `is_nvfp4` to use it
   (`vllm/v1/kv_cache_interface.py:205-207`), and so do both MLA specs
   (`:329-335`, `:550-553`). The allocation shape is
   `(num_blocks, 2 * num_kv_heads, block_size, full_dim)` in uint8
   (`vllm/v1/attention/backends/flashinfer.py:403-406`) — note `2 * num_kv_heads`
   rather than the `2 * head_size` content packing every other dtype uses.
3. **The store.** `reshape_and_cache_nvfp4_kernel`
   (`csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu:49-176`), dispatched from
   `cache_kernels.cu:775-784` when `kv_cache_dtype == "nvfp4"`. The page is
   `[K_data | K_scale | V_data | V_scale]`, with the scale region starting
   `num_heads * block_size * head_size/2` bytes into each side
   (`nvfp4_kv_cache_kernels.cu:222-226`). The quantization global scale is the
   reciprocal of the checkpoint scale (`:97`), and `tests/kernels/attention/
   test_cache.py:261-263` sets that scale to `amax / 448.0` — i.e. the E4M3 max,
   not the `448 * 6` product `tests/kernels/quantization/nvfp4_utils.py:141`
   uses for weights.

The **read** is the piece with no portable implementation: it is FlashInfer's
`trtllm_batch_context_with_kv_cache` / `trtllm_batch_decode_with_kv_cache`
(`flashinfer.py:2023,2214`), fed the data and block-scale regions as two split
views (`:1838-1849`, via `nvfp4_split_data_scale`,
`vllm/utils/torch_utils.py:419-460`).

## Three findings, and what each one blocks

### F1. Pinned vLLM refuses `nvfp4` on our only GPU

`FlashInferBackend.supports_kv_cache_dtype` admits `nvfp4` only when
`current_platform.is_device_capability_family(100)` **and** trtllm attention is
available for both prefill and decode (`flashinfer.py:875-882`), and
`FlashInferMetadataBuilder` raises `"--kv-cache-dtype nvfp4 requires the SM100
trtllm-gen FlashInfer path."` otherwise (`:711-722`). No other backend lists
`nvfp4` in `supported_kv_cache_dtypes`.

`GATE_DEVICE` here is a GB10, `DEVICE_ARCH=sm_121a` — capability family 121, not
100. **The pinned oracle therefore cannot run `--kv-cache-dtype nvfp4` on the
box every gate number on this row would be measured on.** There is no oracle run
to gate a port against, and AGENTS.md's rule is to run the pinned vLLM on the
identical workload, not to infer the answer from its source.

This does not on its own forbid the port — our attention kernels are ours, and a
backend capability is not a format property. It does mean the correctness gate
would be against a reference implementation only, never against a live oracle,
and the spec that schedules W1 has to say so in its `## Gates`.

### F2. The page layout is not expressible on the current attention seam

`vt::PagedAttention` takes exactly two cache tensors, each
`[num_blocks, block_size, num_kv_heads, head_size]`
(`include/vt/ops.h:5343-5350`). The fp8 arm fits that seam because an fp8 page is
the same shape in `kI8` storage — one byte per element instead of two — which is
why `KV-FP8` W1 needed no signature change.

An NVFP4 page does not fit it. Per side the data region is `head_size/2` bytes
per head-token and the block-scale region is `head_size/16` bytes per head-token,
based at a **different page offset** — two strided views over one page, which is
exactly what `nvfp4_split_data_scale` constructs upstream. Serving it needs four
cache views where the seam has two, on every provider: the CPU kernel, the CUDA
paged/flash/WMMA launchers, ROCm, and the runner's `KvSlice` view construction
(`src/vllm/v1/worker/gpu/runner.cpp:1540-1560`), plus `CheckKvCacheShape`, which
validates the 5-dim NHD geometry the engine allocates.

That is the "kernel work beyond a dtype resolution" this row's dispatch named as
a stop condition. It is a decision about the shape of `vt::PagedAttention`, not
an implementation detail to be settled inside a KV row.

### F3. Upstream's own K-side scale layout is self-contradictory, and nothing here can settle it

The store kernel writes **K** block scales linearly and **V** block scales
4x4-swizzled for the SM100 trtllm-gen MHA kernel
(`nvfp4_kv_cache_kernels.cu:157-172`, swizzle at `:26-39`).

Upstream's own reference dequant `dequant_nvfp4_kv_cache`
(`tests/kernels/quantization/nvfp4_utils.py:90-146`) un-swizzles **both** sides,
and `tests/kernels/attention/test_cache.py:357-372` feeds K through it. The
swizzle is `stored_t = (t/4)*4 + s/(S/4)`, `stored_s = (s%(S/4))*4 + t%4`, whose
inverse is precisely the permutation that reference applies; it is the identity
only when `S == 4` **and** `s == t%4`, so for any `head_size` other than 64 the
store and the reference disagree about where a K block scale lives.

`test_cache.py:205-207` skips the whole case below `has_device_capability(100)`,
so that disagreement has never been executed on any machine we can reach, and
neither reading is confirmed. Porting either one is inventing a layout, which is
what "mirror vLLM" exists to prevent.

**This is filed as [#2925](https://github.com/mudler/vllm.cpp/issues/2925) and is
the first thing the W1 spec must resolve** — by running upstream on an SM100 box,
or by reading the trtllm-gen kernel's own scale addressing, not by choosing.

## What this change lands

One brick, reachable and gated:

- `include/vllm/v1/kv_cache_dtype.h` — `GetKvQuantMode`,
  `KvQuantModeIsPerTokenHead`, `KvQuantModeIsNvfp4` (1:1 with
  `kv_cache_interface.py:51-75`), `Nvfp4KvCacheFullDim` (1:1 with
  `torch_utils.py:414-416`), `IsCacheDTypeName` (the `CacheDType` Literal
  membership, `config/cache.py:19-35`), the `quant_mode` field on
  `ResolvedCacheDType`, and a per-member refusal.
- `include/vllm/v1/kv_cache_interface.h` — the `KVQuantMode` definition moves out
  of it and into the header above, which it already includes. The underlying
  `uint8_t` is kept, because the enum is a struct field.
- `tests/vt/test_ops_fp8_kv_cache.cpp` — the red-before cases.

The refusal is the behaviour change an operator sees. Before, `--kv-cache-dtype
nvfp4` said it "is not implemented in KV-FP8 W1" and listed five other names
beside it; a reader learned which row does **not** own it. After, it names
`KV-NVFP4-TURBO`, #2620, this spec, and the three missing bricks, and it tells
the operator that `--kv-cache-dtype fp8` is the served quantized arm. The other
five members each name their own owner: `fp8_inc` and `fp8_ds_mla` name
`QUANT-KV-FP8-VENDOR` and `KV-DSV4-MULTICACHE`, and the turboquant and
per-token-head members name this row.

A name outside the Literal entirely (`--kv-cache-dtype not_a_dtype`) now refuses
as "not a vLLM CacheDType" and lists the members, which is upstream's pydantic
Literal validation (`config/cache.py:19-35`) rather than a backend-support
answer. The two were one message before, and they are different facts.

## Reachability

`ParseCacheDType` is on the production load path:
`LoadedEngine::ApplyResolvedCacheDType` (`src/vllm/entrypoints/model_loader.cpp:1795`)
calls it for every engine, from the probe KV config onward, so the refusal fires
before a byte is allocated. It is reachable from three shipped entry points:
`vllm-bench --kv-cache-dtype`, `vllm-server --kv-cache-dtype`, and a checkpoint
that declares `kv_cache_quant_algo: NVFP4`, which
`MapModeloptKvAlgo` (`src/vllm/config/cache.cpp:35-39`) deliberately resolves to
the string `nvfp4` so that a declared-and-unserved format refuses instead of
silently serving bf16.

`tests/examples/test_bench_kv_cache_dtype.cpp` already execs the built
`vllm-bench` with `--kv-cache-dtype nvfp4` and asserts the refusal reaches the
operator through `argv`, which is the production entry point for a harness.

## Tests

Ported in this change:

- The `CacheDType` Literal membership and the `KVQuantMode` map, over **every**
  one of the 16 members of `config/cache.py:19-35` — the enum value for each,
  `is_per_token_head` for each, `is_nvfp4` for each. Upstream has no unit test
  over `get_kv_quant_mode`; the parameter set is the Literal itself, so the test
  enumerates it and fails when a member is added upstream and not here.
- `nvfp4_kv_cache_full_dim` against the values `torch_utils.py:414-416` yields
  for the head sizes the shipped registries use.
- The refusal identity: each unserved member refuses **and names its own owner
  row**, which is what makes the per-member split observable rather than
  cosmetic.

Owed to W1 when it is scheduled: the port of
`tests/kernels/attention/test_cache.py::test_reshape_and_cache_flash[nvfp4]`
with its parameters (`head_size % 16 == 0`, `(head_size/16) % 4 == 0`,
`block_size % 4 == 0`, fp16/bf16 input only, per-tensor scales only) and its
tolerances (`atol=1.5, rtol=0.5`), against a port of `dequant_nvfp4_kv_cache`.

## Gates

- `ctest` over `test_ops_fp8_kv_cache`, red before the header change and green
  after, with the mutation table in `## Outcome`.
- `tests/examples/test_bench_kv_cache_dtype.cpp` stays green: the refusal it
  asserts is still a refusal, still reaches `argv`, and still names `nvfp4` and
  `cache_dtype`.
- No device gate. Nothing here allocates a page or writes a byte, and F1 means
  no oracle run exists for the arm this row will eventually serve.

## Risks

- **A per-member message list is a second copy of the `CacheDType` Literal.**
  It is mitigated by `IsCacheDTypeName` being the one list, and by the test
  enumerating that list rather than restating it. It is not eliminated: when
  upstream adds a member, this file and that list both need the row. The
  upstream-sync procedure owns that, as it does for every mirrored Literal.
- **A refusal message is weak to gate.** The test therefore asserts the OWNER
  identity per member, not the prose, so a message reworded for clarity stays
  green and a message that loses the owner reds.

## Owed

- **The nvfp4 KV arm itself** — [#2620](https://github.com/mudler/vllm.cpp/issues/2620).
  Missing: (a) the packed page — `AttentionSpec::real_page_size_bytes` has no
  nvfp4 branch and the allocation shape is `2 * head_size` content-packed rather
  than `2 * num_kv_heads` side-packed; (b) the quantizing store, CPU and CUDA;
  (c) the dequantizing paged-attention read, which needs the four-view seam of
  F2; (d) the runner wiring and the `head_size % 16`, `(head_size/16) % 4`,
  `block_size % 4` admission checks. Blocked on the F2 decision and on F3.
- **The K-side block-scale layout contradiction** —
  [#2925](https://github.com/mudler/vllm.cpp/issues/2925). Blocked on SM100
  hardware or on reading the trtllm-gen kernel's scale addressing.
- **`fp8_per_token_head`, `int8_per_token_head`, `int4_per_token_head` and the
  four `turboquant_*` members.** This row owns them, none is ported, and none has
  a spike. They are refused by name and each names this row.

## Now

`KV-NVFP4-TURBO` stays `INVENTORIED`. This change gives it its committed spec
and a measured spike; it does not make it `READY`, because F2 is a decision this
row cannot take alone and F3 needs hardware the fleet does not have.

## Outcome

Recorded when the row leaves `INVENTORIED`. What this spike measured is in
`## Three findings` above, and the reason W1 is a refusal rather than an
implementation is F1 plus F2 plus F3, in that order of weight: no oracle run on
the gate device, a seam that cannot express the page, and an upstream layout
that contradicts its own reference.
