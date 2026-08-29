// GLM-5.3-Flash registry TU — the ADDITIVE self-registration seam (W1 of
// MODEL-MM-GLM53-FLASH, #2067). Follows the qwen4_exp_registry.cpp /
// glm4_moe_lite_registry.cpp seam exactly: a NEW translation unit with ONE
// REGISTER_VLLM_MODEL line and ZERO edit to any shared array.
//
// UPSTREAM. `Glm5NextForConditionalGeneration` is registered by NO vLLM
// revision. `git grep "Glm5\|glm5_next"` returns zero hits at our parity pin
// `555967922` AND at vLLM `origin/main`; `vllm-project/vllm#53906` would
// register it and is OPEN and unmerged, and an unmerged pull request is not a
// revision. SGLang, vllm-omni and llama.cpp implement nothing either. That is
// ABSENCE from vLLM `main` rather than staleness in our pin, so this TU
// deliberately carries no pinned upstream module/class anchor — the convention
// `MODEL-MM-qwen4-exp-*` follows for a beyond-pin arm — and no pin was
// advanced. The ALGORITHM source is transformers **v5.16.1**; see
// `.agents/specs/glm5-next-flash.md` `## Oracles`.
//
// The MTP head is deliberately NOT registered as a second architecture. The
// checkpoint carries a 46th layer directory that is a DeepSeek-V3-style MTP
// block, and the transformers reference DISCARDS it
// (`_keys_to_ignore_on_load_unexpected = [r"layers\.45\.", ...]`), so there is
// no second architecture string to register. That is why this row moves the
// architecture count by ONE.
//
// SCOPE HONESTY, and W5b-2b moved it. Registering this arch made it RESOLVE
// (W1); W5c made it LOAD; this TU's `forward` hook now RUNS, on the CPU device,
// through `glm5_next::Glm5NextHostForward`. What has NOT changed is the reason
// the earlier refusal existed: no oracle for this model runs on any hardware
// this project owns (the smallest published artifact is 181.32 GiB against
// ~119.63 GiB on GB10), so there is no downstream token gate that would catch a
// forward returning plausible garbage, and O1 still says so. What replaces the
// blanket refusal is a set of NARROW ones, each naming what is owed rather than
// the whole capability: a non-CPU queue, a multi-request step, and the
// safetensors load are each refused by name in place.
#include "vllm/model_executor/models/model_registry.h"

#include "vt/dtype.h"  // VT_CHECK

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/glm5_next.h"
#include "vllm/model_executor/models/glm5_next_loader.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits complete type
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"  // v1::ResolveKvCacheDType
#include "vllm/v1/kv_cache_interface.h"

namespace vllm {
namespace {

// Text generation, multimodal (image AND video: the wrapper carries all six
// placeholder ids and a `vision_config`), and HYBRID — 34 of 45 layers are KDA
// linear attention carrying recurrent state, so this belongs with the hybrids
// and not with the pure-attention arms.
inline constexpr ModelInfo kGlm5NextInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = true,
    // FALSE by the house convention the blanket assertion in
    // test_model_registry.cpp enforces: our ModelInfo is a consumed subset
    // whose only reader short-circuits on is_hybrid, so every hybrid wrapper
    // here leaves this false even though upstream's class carries inner state.
    .has_inner_state = false,
    .supports_multimodal = true,
    .score_type = "bi-encoder",
};

std::unique_ptr<LoadedModel> LoadGlm5NextForConditionalGeneration(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind == ModelSource::Kind::kGguf) {
    // W5c ([#2242](https://github.com/mudler/vllm.cpp/issues/2242)) LOADS it.
    // The GGUF k-quant arm is OWED, not optional (AGENTS.md,
    // porting-a-model.md), and for this row it is the ONLY arm that fits a
    // host we own: `unsloth/GLM-5.3-Flash-GGUF` UD-Q2_K_XL is 101.2535 GiB on
    // disk against ~119.63 GiB usable on GB10, where every safetensors artifact
    // (FP8 305.78 GiB, BF16 598.53 GiB, NVFP4 181.32 GiB) does not.
    //
    // THE ARTIFACT EXISTS, and the refusal this replaced said it did not. That
    // sentence — "NO `.gguf` of this model exists anywhere" — was true when W1
    // wrote it and stopped being true when `unsloth/GLM-5.3-Flash-GGUF`
    // revision `d425e572fb9686125831f476129e51cea34bc5b4` was published and
    // staged: 1412 tensors, four shards, `general.architecture = glm5next`,
    // read out of the file's own header. A record correction that leaves the
    // lie in the product is not a correction, so it is removed here and not
    // only in the spec. O7 is W7b's
    // ([#2225](https://github.com/mudler/vllm.cpp/issues/2225)) to discharge;
    // this change does not discharge it and does not contradict it — what W7b
    // still owes is the sha256, the conversion recipe and the peak RSS of a
    // real load, none of which this wave measured.
    //
    // A null `gguf` reaches here from a caller that set the KIND without the
    // FILE. Refused by name rather than dereferenced: the alternative is a
    // segmentation fault inside a loader the reader is entitled to read as
    // "GGUF is not supported here".
    if (source.gguf == nullptr) {
      throw std::runtime_error(
          "Glm5NextForConditionalGeneration: the model source says GGUF but "
          "carries no file. See .agents/specs/glm5-next-flash.md and issue "
          "#2242.");
    }
    return std::make_unique<Glm5NextLoadedModel>(
        registration, LoadGlm5NextFromGguf(*source.gguf, config));
  }
  (void)registration;
  (void)config;
  // The safetensors arm stays refused, and NOT because it is the harder one.
  // Every published safetensors artifact of this model is larger than every
  // device this project owns, so an arm that read them would be code nothing
  // could ever run. The spec's `## Owed` records it with that reason rather
  // than as an unqualified to-do.
  throw std::runtime_error(
      "Glm5NextForConditionalGeneration: the safetensors weight loader is not "
      "ported (every published safetensors artifact -- FP8 305.78 GiB, BF16 "
      "598.53 GiB and NVFP4 181.32 GiB -- exceeds every device this project "
      "owns at ~119.63 GiB on GB10, so the GGUF arm is the supported one). "
      "See .agents/specs/glm5-next-flash.md and issue #1998.");
}

void PrepareGlm5NextForConditionalGeneration(LoadedModel& model,
                                             const HfConfig& config,
                                             vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardGlm5NextForConditionalGeneration(
    LoadedModel& model, const ModelForwardInput& input) {
  (void)model;
  (void)input;
  // THE REFUSAL COMES FIRST, AND THERE IS NO DOWNCAST ABOVE IT. The house shape
  // opens the type-erased handle with `ModelAs<...>` before anything else,
  // because a bare `static_cast` down the hierarchy is undefined behaviour on
  // an object that is not really that type (#775, #730).
  //
  // W5c CHANGED THE PREMISE HALF-WAY AND THE ORDER STILL STANDS. The earlier
  // version of this comment argued that nothing could PRODUCE a loaded
  // GLM-5.3-Flash while `load_weights` refused unconditionally, so the only
  // handle a caller could present was a foreign one. That is no longer true:
  // the GGUF arm above returns a real `Glm5NextLoadedModel`. What has not
  // changed is that there is no forward to open it FOR, so a downcast placed
  // first would report a type mismatch on a foreign handle and then fall
  // through to this same refusal on our own -- two messages for one missing
  // capability, and the refusal reachable only on the path where it says
  // least. W5b ([#2241](https://github.com/mudler/vllm.cpp/issues/2241))
  // restores `ModelAs` in the same change that gives it something to read.
  //
  // `VT_CHECK(false, ...)` IN THE HOOK BODY, not a bare throw behind a
  // `Class::ForwardDevice` delegate: `check-runner-routing-consistency.py`
  // recognises a refuse-by-name stub by exactly this token and classifies the
  // hook body itself, and a model it cannot classify lands in the silently
  // exempt NONE bucket. And `[[noreturn]]` on a non-void return type is MSVC
  // C4646, promoted to C2220 under /W4 /WX.
  //
  // WHAT THIS REFUSAL BUYS, exactly: it prevents a plausible-but-wrong forward,
  // not a wrong number. There is no partial numeric path here to fall back to.
  // Every primitive named below is unimplemented for THIS model, and two of
  // them look implemented and are not -- our KDA is Kimi-Linear's softplus
  // forget gate where this model needs the sigmoid branch, and our
  // `HcHeadCollapse` is DeepSeek-V4's weighted collapse where this model needs
  // an unweighted mean. Reusing either would generate fluent, wrong text that
  // no gate on this fleet could detect.
  VT_CHECK(false,
           "Glm5NextForConditionalGeneration: the forward is not ported yet. W2 "
           "owes the KDA forget gate's SIGMOID branch (`gate_lower_bound` "
           "-5.0; our kimi_kda.cpp implements the softplus branch and is NOT a "
           "substitute), the strict-fp32 gated RMSNorm and `l2norm`; W3 the "
           "NoPE MLA block -- `MlaBlockDims::Validate` still refuses "
           "`qk_rope_head_dim == 0` -- and the DSA k-pool indexer; W4 the "
           "UNWEIGHTED mHC head collapse (`deepseek_v4_mhc.cpp`'s "
           "`HcHeadCollapse` is the weighted DeepSeek-V4 one and is NOT a "
           "substitute); W5b the decoder layer, the DSA attention block and the "
           "assembled text forward; W6 the vision tower, processor and "
           "placeholder expansion. The WEIGHT TOWER is ported and this model "
           "LOADS -- W5c (#2242) -- so a handle reaching here is real and the "
           "missing part is the forward, not the load. "
           "See .agents/specs/glm5-next-flash.md and issue #1998.");
  return ForwardLogits{};  // unreachable; VT_CHECK always throws here
}

// ─── The heterogeneous KV-cache spec (W5, #2223) ─────────────────────────────
//
// THREE published groups, and the shape of them is the decision this function
// exists to record:
//
//   0. the 11 DSA layers' MLA latent        `MLAAttentionSpec`, head 512
//   1. the 34 KDA layers' recurrent state   `MambaSpec`, 2 states
//   2. the 11 DSA layers' indexer cache     `MLAAttentionSpec`, head 257
//
// Following `kimi_linear_registry.cpp:135-166`, which publishes the same MLA +
// KDA pair; the third group is this model's and Kimi-Linear has no analogue.
//
// GROUP 0 IS AN MLA LATENT, NOT A K+V PAIR. `Glm5NextTextAttention` caches the
// compressed `kv_a_proj_with_mqa` output and reconstructs K and V from it
// through `kv_b_proj` (`modeling_glm5_next.py:1136-1153`, `expand_kv`), so one
// row per token of `kv_lora_rank + qk_rope_head_dim` elements, `num_kv_heads`
// 1, and NO separate V. `qk_rope_head_dim` is ZERO on this model -- upstream's
// `validate_architecture` requires it ("Expecting NoPE for the DSA attention
// layers") -- so the row is 512 wide where every DeepSeek variant's and
// Kimi-Linear's is 576. A port that reuses the 576 over-allocates by 12.5% and
// nothing downstream reads the difference.
//
// GROUP 1 IS ONE UNIFORM RECURRENT GROUP AND ITS CONV STATE IS `conv_kernel_dim`
// WIDE, NOT `conv_kernel_dim - 1`. Both halves are upstream's.
//
//   * Uniform, because that is all upstream can express:
//     `get_mamba_state_shape_from_config` is a CLASSMETHOD over the config with
//     no `layer_idx` (`vllm/model_executor/models/interfaces.py:809-812` at the
//     parity pin `5559679229`) and `get_mamba_groups` asserts every `MambaSpec`
//     in the model equal (`vllm/v1/worker/mamba_utils.py:441`). Every one of
//     this model's 34 KDA layers carries the same two states anyway, so
//     uniformity costs nothing here.
//   * `conv_kernel_dim` wide, because the reference ALLOCATES it that wide:
//     `LinearAttentionLayer.lazy_initialization` builds
//     `torch.zeros((*shape[:-1], conv_kernel_size))`
//     (`transformers` v5.16.1 `cache_utils.py:1015-1024`) and
//     `Glm5NextTextLinearAttention.forward` passes
//     `conv_kernel_size=self.conv_kernel_size` (`modeling_glm5_next.py:669-671`),
//     so the state is `[B, conv_dim, 4]` and not the `[B, conv_dim, 3]` the
//     convolution arithmetic alone would need. `causal_conv1d_update` then reads
//     `state_len = conv_state.shape[-1]` (`:382`) and writes back that many
//     columns, so the slack column is part of the contract rather than padding.
//     `glm5_next_kda.h` records the same width for the host reference's
//     `Glm5NextKdaCache::conv_state`, and publishing `K - 1` here -- which is
//     what `kimi_linear_registry.cpp:156` publishes for ITS model -- would give
//     the runner a cache one column short of what the layer reads.
//
// ONE CONV STATE, NOT THREE. The checkpoint stores `self_attn.{q,k,v}_conv1d`
// separately and the reference declares ONE grouped depthwise conv over the
// concatenated `[q; k; v]` channel axis (`modeling_glm5_next.py:620-628`), so
// the CACHE is one `3 * num_heads * head_dim` channel state. An earlier
// revision of this function's refusal said "three separate conv states", which
// would have tripled this group; `glm5_next_kda.h` "THREE LAYOUT FACTS" settles
// it and the case below pins the single width.
//
// GROUP 2 IS AN `MLAAttentionSpec` AND THAT IS LOAD-BEARING, not an MLA claim.
// `MLAAttentionSpec` is the key-only page budget -- one vector per stored state
// instead of a K+V pair. A `FullAttentionSpec` in that position is absorbed by
// the runner's leftover scan as the single `fa_draft` draft-KV slot
// (`src/vllm/v1/worker/gpu/runner.cpp`, the `draft_slot_taken` arm, which
// `continue`s); the leftover count then stays 0, `multi_cache_topology` stays
// false, and the side cache is published and never allocated with nothing
// reported. `MODEL-MM-QWEN4-EXP` W5c-1 (#2206) measured that on its own third
// group and the same arm is live here.
//
// ITS ROW IS 257 WIDE AND `compress_ratio` IS 1. `PackIndexerStates`
// (`glm5_next_dsa.h`) stores `concat[k(head_dim), gate_scores(head_dim),
// valid(1)]` PER TOKEN (`modeling_glm5_next.py:798-801`), so 2 * 128 + 1 = 257
// elements and one row per token. The k-pool stage compresses at READ time
// inside `GetPooledStates`, not at store time, so nothing here divides by
// `index_kpool` -- which is the opposite of `MODEL-MM-QWEN4-EXP`'s QSA side
// cache, where the compression IS in the store and `compress_ratio` is 4. Our
// DeepSeek-V4 parent stores 128 (the key alone); reading that number across
// would under-allocate this cache by half.
//
// REAL PER-LAYER NAMES, NEVER PLACEHOLDERS. `ResolveKVCacheGroupLayerNames`
// (`src/vllm/v1/kv_cache_interface.cpp`) rewrites a placeholder group set into
// per-layer names, but its fallback can name only a TARGET attention group and
// one `fa_draft` slot: a third attention group gets `layer_names.clear()` and
// the runner then refuses the unnamed group. Publishing the real names also
// makes the rewrite a no-op by its own idempotence guard, so what the runner
// allocates is what this function said. #1963/#1966 are the standing reason a
// KV arithmetic here is re-derived against the runner rather than trusted.
v1::KVCacheConfig MakeGlm5NextKVCache(const HfConfig& config, int block_size,
                                      int num_blocks) {
  (void)config;
  (void)block_size;
  (void)num_blocks;
  // Unreachable while the loader refuses, and refusing by name anyway rather
  // than returning an empty config. This model needs THREE distinct cache
  // shapes in one spec -- a KDA recurrent state plus three separate conv states
  // on 34 layers, a 512-wide MLA latent on 11, and a DSA indexer side cache
  // that is 257 floats per token per layer rather than the DeepSeek-V4 parent's
  // 128 because of the k-pool stage -- and a spec that silently omitted any of
  // them would allocate a wrong-sized cache that nothing downstream checks.
  // #1963/#1966 are the standing reason a KV arithmetic here is re-derived
  // against the runner rather than trusted.
  throw std::runtime_error(
      "Glm5NextForConditionalGeneration: the KV-cache spec is not ported yet "
      "(W3 owes the NoPE MLA latent group and the k-pool indexer side cache, "
      "W5b the KDA recurrent and three-conv state group). See "
      ".agents/specs/glm5-next-flash.md and issue #1998.");
}

const ModelFactory kGlm5NextFactory{
    .parse_config = &ParseGlm5NextConfig,
    .load_weights = &LoadGlm5NextForConditionalGeneration,
    .prepare = &PrepareGlm5NextForConditionalGeneration,
    .forward = &ForwardGlm5NextForConditionalGeneration,
    .make_kv_cache = &MakeGlm5NextKVCache,
    .is_dense_model = false,
    // W5b-2c (#2348): this forward READS the cache set keyed by published layer
    // name, so `ModelRegistry::Forward` stops refusing the step above it. The
    // consuming code is `glm5_next_kv.cpp` and the refusals it can raise are
    // all by name.
    .consumes_multi_kv = true,
};

}  // namespace

REGISTER_VLLM_MODEL(glm5_next, "Glm5NextForConditionalGeneration",
                    kGlm5NextFactory, kGlm5NextInfo)

}  // namespace vllm
