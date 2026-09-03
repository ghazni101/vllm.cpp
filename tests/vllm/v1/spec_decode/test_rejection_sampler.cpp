// Ported from (test-porting protocol .agents/test-porting.md):
//   * tests/v1/sample/test_rejection_sampler.py @ e24d1b24 —
//       test_perfect_match          :133
//       test_early_mismatch         :154
//       test_multiple_sequences     :179
//       test_single_token_sequence  :204
//       test_empty_sequence         :225
//       test_multiple_mismatches    :246
//       test_parametrized_cases     :288
//     (the legacy-sampler suite; its ASSERTIONS are realized here against our
//     MRV2-shaped RejectionSampler — test-porting rule 3.)
//   * tests/v1/spec_decode/test_rejection_sampler_utils.py @ e24d1b24 —
//       test_greedy_rejection_sample        :183 (k in {1, 3})
//       test_placeholder_draft_token_rejected :285
//     (the stochastic :141, synthetic :215 and block-verification :325/:372
//      cases are SKIPPED until M-mtp-3 — see the SKIPPED note at the bottom.)
//
// The upstream cases construct logits whose argmax at expanded row j is
// output_tokens[r][j], schedule spec_tokens[r] as the drafts, and assert the
// emitted token stream. Upstream's fixed-width [num_reqs, k+1] output pads
// rejected positions with PLACEHOLDER_TOKEN_ID (-1,
// vllm/v1/sample/rejection_sampler.py:30); our RejectionSamplerOutput returns
// ragged per-request vectors of exactly num_sampled tokens, so a padded
// upstream row [a, b, -1, -1] becomes {a, b} plus num_sampled == 2. Both forms
// are asserted.
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "vllm/v1/spec_decode/rejection_sampler.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

using vllm::v1::RejectionSampler;
using vllm::v1::RejectionSamplerOutput;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Queue Q() { return Queue{Cpu(), nullptr}; }

constexpr int kVocab = 16;

// One verify-step fixture, built exactly the way upstream's
// `create_logits_tensor` + `create_spec_decode_metadata` do:
//   * `target_tokens[r]` are the tokens the TARGET argmaxes at request r's
//     expanded rows (length 1 + k_r);
//   * `spec_tokens[r]` are the k_r draft tokens the scheduler proposed.
// The draft_sampled array mirrors `input_ids[logits_indices]`: row cu[r] holds
// the previous step's token (never compared — upstream reads draft_sampled at
// logit_idx + 1, rejection_sampler_utils.py:534) and rows cu[r]+1.. hold the
// drafts.
struct VerifyStep {
  std::vector<float> logits;
  std::vector<int32_t> draft_sampled;
  std::vector<int32_t> cu_num_logits;
  int64_t num_logits = 0;
};

VerifyStep MakeStep(const std::vector<std::vector<int32_t>>& spec_tokens,
                    const std::vector<std::vector<int32_t>>& target_tokens) {
  REQUIRE(spec_tokens.size() == target_tokens.size());
  VerifyStep s;
  s.cu_num_logits.push_back(0);
  int32_t total = 0;
  for (size_t r = 0; r < spec_tokens.size(); ++r) {
    REQUIRE(target_tokens[r].size() == spec_tokens[r].size() + 1);
    total += static_cast<int32_t>(target_tokens[r].size());
    s.cu_num_logits.push_back(total);
  }
  s.num_logits = total;
  s.logits.assign(static_cast<size_t>(total) * kVocab, 0.0f);
  s.draft_sampled.assign(static_cast<size_t>(total), 0);
  int32_t row = 0;
  for (size_t r = 0; r < spec_tokens.size(); ++r) {
    // Row cu[r] input id: an arbitrary previously-sampled token (never read).
    s.draft_sampled[static_cast<size_t>(row)] = 0;
    for (size_t j = 0; j < target_tokens[r].size(); ++j) {
      // Make target_tokens[r][j] the strict argmax of expanded row cu[r]+j.
      s.logits[static_cast<size_t>(row + static_cast<int32_t>(j)) * kVocab +
               static_cast<size_t>(target_tokens[r][j])] = 10.0f;
      if (j < spec_tokens[r].size()) {
        s.draft_sampled[static_cast<size_t>(row) + j + 1] = spec_tokens[r][j];
      }
    }
    row += static_cast<int32_t>(target_tokens[r].size());
  }
  return s;
}

RejectionSamplerOutput Run(const std::vector<std::vector<int32_t>>& spec_tokens,
                           const std::vector<std::vector<int32_t>>& target_tokens,
                           int num_speculative_steps) {
  VerifyStep s = MakeStep(spec_tokens, target_tokens);
  Queue q = Q();
  Tensor logits = Tensor::Contiguous(s.logits.data(), DType::kF32, Cpu(),
                                     {s.num_logits, static_cast<int64_t>(kVocab)});
  RejectionSampler sampler(num_speculative_steps);
  return sampler.forward(q, logits, s.draft_sampled, s.cu_num_logits);
}

// The upstream fixed-width form: pad each request's emitted tokens to
// max_len with PLACEHOLDER_TOKEN_ID (-1).
std::vector<std::vector<int32_t>> Padded(const RejectionSamplerOutput& out, size_t width) {
  std::vector<std::vector<int32_t>> rows;
  for (const auto& toks : out.sampled_token_ids) {
    std::vector<int32_t> row = toks;
    row.resize(width, -1);
    rows.push_back(row);
  }
  return rows;
}

}  // namespace

// ---------------------------------------------------------------------------
// test_perfect_match (test_rejection_sampler.py:133): every draft matches, so
// the emitted stream is the drafts plus the bonus token.
TEST_CASE("rejection_sampler: perfect_match emits every draft plus the bonus token") {
  const RejectionSamplerOutput out = Run({{1, 2, 3}}, {{1, 2, 3, 4}}, /*k=*/3);
  CHECK(out.sampled_token_ids.size() == 1);
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{1, 2, 3, 4});
  CHECK(out.num_sampled[0] == 4);
  CHECK(out.num_rejected[0] == 0);
  CHECK(Padded(out, 4)[0] == std::vector<int32_t>{1, 2, 3, 4});
}

// test_early_mismatch (:154): mismatch at position 1 -> emit the target argmax
// there and STOP; nothing after it is accepted.
TEST_CASE("rejection_sampler: early_mismatch emits the target argmax and stops") {
  const RejectionSamplerOutput out = Run({{1, 2, 3}}, {{1, 5, 3, 4}}, /*k=*/3);
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{1, 5});
  CHECK(out.num_sampled[0] == 2);
  // 3 drafts scheduled, 1 accepted -> 2 rejected (the scheduler rolls back by 2).
  CHECK(out.num_rejected[0] == 2);
  CHECK(Padded(out, 4)[0] == std::vector<int32_t>{1, 5, -1, -1});
}

// test_multiple_sequences (:179): two requests with DIFFERENT k_i (2 and 1).
TEST_CASE("rejection_sampler: multiple_sequences with different per-request k") {
  const RejectionSamplerOutput out = Run({{1, 2}, {3}}, {{1, 2, 5}, {3, 4}}, /*k=*/2);
  CHECK(out.sampled_token_ids.size() == 2);
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{1, 2, 5});
  CHECK(out.sampled_token_ids[1] == std::vector<int32_t>{3, 4});
  CHECK(out.num_sampled[0] == 3);
  CHECK(out.num_sampled[1] == 2);
  CHECK(out.num_rejected[0] == 0);
  CHECK(out.num_rejected[1] == 0);
  CHECK(Padded(out, 3)[0] == std::vector<int32_t>{1, 2, 5});
  CHECK(Padded(out, 3)[1] == std::vector<int32_t>{3, 4, -1});
}

// test_single_token_sequence (:204): k == 1, accepted.
TEST_CASE("rejection_sampler: single_token_sequence (k=1) accepts and emits the bonus") {
  const RejectionSamplerOutput out = Run({{1}}, {{1, 2}}, /*k=*/1);
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{1, 2});
  CHECK(out.num_sampled[0] == 2);
  CHECK(out.num_rejected[0] == 0);
}

// test_empty_sequence (:225): NO drafts — the k == 0 reduction. This is the
// byte-identity anchor: the emitted token is exactly the plain greedy argmax.
TEST_CASE("rejection_sampler: empty_sequence (k=0) reduces to the plain greedy argmax") {
  const RejectionSamplerOutput out = Run({{}}, {{5}}, /*k=*/1);
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{5});
  CHECK(out.num_sampled[0] == 1);
  CHECK(out.num_rejected[0] == 0);
}

// test_multiple_mismatches (:246): both requests reject, at different positions.
TEST_CASE("rejection_sampler: multiple_mismatches reject independently per request") {
  const RejectionSamplerOutput out =
      Run({{1, 2, 3}, {4, 5, 6}}, {{1, 2, 7, 6}, {4, 8, 6, 9}}, /*k=*/3);
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{1, 2, 7});
  CHECK(out.sampled_token_ids[1] == std::vector<int32_t>{4, 8});
  CHECK(out.num_sampled[0] == 3);
  CHECK(out.num_sampled[1] == 2);
  CHECK(out.num_rejected[0] == 1);
  CHECK(out.num_rejected[1] == 2);
  CHECK(Padded(out, 4)[0] == std::vector<int32_t>{1, 2, 7, -1});
  CHECK(Padded(out, 4)[1] == std::vector<int32_t>{4, 8, -1, -1});
}

// test_parametrized_cases (:288): the three upstream parametrizations.
TEST_CASE("rejection_sampler: parametrized cases (perfect / first-mismatch / mixed)") {
  SUBCASE("perfect match with bonus") {
    const RejectionSamplerOutput out = Run({{1, 2}}, {{1, 2, 3}}, /*k=*/2);
    CHECK(Padded(out, 3)[0] == std::vector<int32_t>{1, 2, 3});
  }
  SUBCASE("first mismatch") {
    const RejectionSamplerOutput out = Run({{1}}, {{2, 3}}, /*k=*/1);
    CHECK(Padded(out, 2)[0] == std::vector<int32_t>{2, -1});
    CHECK(out.num_sampled[0] == 1);
    CHECK(out.num_rejected[0] == 1);
  }
  SUBCASE("mixed matches") {
    const RejectionSamplerOutput out = Run({{1, 2}, {3, 4}}, {{1, 5, 6}, {3, 4, 7}}, /*k=*/2);
    CHECK(Padded(out, 3)[0] == std::vector<int32_t>{1, 5, -1});
    CHECK(Padded(out, 3)[1] == std::vector<int32_t>{3, 4, 7});
  }
}

// ---------------------------------------------------------------------------
// test_greedy_rejection_sample (test_rejection_sampler_utils.py:183, k in {1,3}):
// "Verify that greedy (temperature=0) always outputs the target argmax at every
// accepted position." Upstream drives one shared target distribution across many
// trials; we drive the same invariant over an exhaustive draft-pattern sweep,
// which additionally pins WHERE the run stops.
TEST_CASE("rejection_sampler: greedy_rejection_sample — every emitted token is the target argmax") {
  for (int k : {1, 3}) {
    // A fixed target argmax sequence; sweep every subset of matching drafts.
    const std::vector<int32_t> target_seq = {3, 7, 2, 9};  // length 4 >= k+1
    std::vector<int32_t> target(target_seq.begin(), target_seq.begin() + k + 1);
    const int num_patterns = 1 << k;
    for (int pattern = 0; pattern < num_patterns; ++pattern) {
      std::vector<int32_t> drafts(static_cast<size_t>(k));
      for (int i = 0; i < k; ++i) {
        // bit set => the draft matches the target argmax at position i.
        drafts[static_cast<size_t>(i)] =
            (pattern >> i) & 1 ? target[static_cast<size_t>(i)]
                               : (target[static_cast<size_t>(i)] + 1) % kVocab;
      }
      const RejectionSamplerOutput out = Run({drafts}, {target}, k);
      // Expected accepted length = the number of leading matching drafts.
      int expect_len = 0;
      while (expect_len < k && drafts[static_cast<size_t>(expect_len)] ==
                                   target[static_cast<size_t>(expect_len)]) {
        ++expect_len;
      }
      CAPTURE(k);
      CAPTURE(pattern);
      CHECK(out.num_sampled[0] == expect_len + 1);
      CHECK(out.num_rejected[0] == k - expect_len);
      REQUIRE(out.sampled_token_ids[0].size() == static_cast<size_t>(expect_len) + 1);
      // THE INVARIANT: every emitted token equals the target argmax at its row.
      for (int j = 0; j <= expect_len; ++j) {
        CHECK(out.sampled_token_ids[0][static_cast<size_t>(j)] ==
              target[static_cast<size_t>(j)]);
      }
    }
  }
}

// test_placeholder_draft_token_rejected (:285): a -1 placeholder draft id must
// be rejected without any out-of-bounds logits read.
TEST_CASE("rejection_sampler: placeholder draft token (-1) is rejected") {
  const RejectionSamplerOutput out = Run({{-1}}, {{5, 6}}, /*k=*/1);
  CHECK(out.num_sampled[0] == 1);
  CHECK(out.sampled_token_ids[0] == std::vector<int32_t>{5});
  CHECK(out.num_rejected[0] == 1);
}

// A chunked-prefilling row samples and rejects nothing
// (_get_num_sampled_and_rejected_kernel, gpu/input_batch.py:421-433).
TEST_CASE("rejection_sampler: a chunked-prefilling row reports 0 sampled and 0 rejected") {
  VerifyStep s = MakeStep({{1, 2}, {3}}, {{1, 9, 5}, {3, 4}});
  Queue q = Q();
  Tensor logits = Tensor::Contiguous(s.logits.data(), DType::kF32, Cpu(),
                                     {s.num_logits, static_cast<int64_t>(kVocab)});
  RejectionSampler sampler(2);
  const RejectionSamplerOutput out =
      sampler.forward(q, logits, s.draft_sampled, s.cu_num_logits, {1, 0});
  CHECK(out.num_sampled[0] == 0);
  CHECK(out.num_rejected[0] == 0);
  CHECK(out.num_sampled[1] == 2);
  CHECK(out.num_rejected[1] == 0);
}

// ─── SPEC-DFLASH2 A2-2 (#2802): THE DEVICE-RESIDENT SPLIT ───────────────────
//
// `forward` used to be one call that ran the accept walk, copied both outputs to
// the host and drained the queue. A2-2 splits it so the walk's outputs can stay
// on the device and the caller decides where the wait goes — a copy-queue event
// instead of a main-queue drain, which is one of the two compute-stream drains a
// speculative step pays. Upstream never pays the first one: its
// `RejectionSampler.__call__` returns DEVICE tensors
// (rejection_sampler.py:262-272 @ pin 5559679229) and the D2H is issued later by
// `AsyncOutput` on the copy stream (model_runner.py:1492-1499).
//
// WHAT THE CPU TIER CAN GATE HERE IS TOKEN IDENTITY, AND ONLY THAT. On this
// backend `Copy` is a memcpy and every event is a null-handle no-op, so nothing
// here can observe an overlap, and nothing here claims one — that is G3/G4 at
// A2-5 and needs a GPU. What it CAN observe is that the split did not move a
// token: every id, every `num_sampled` and every `num_rejected` that comes out
// of `verify` + `CopyToHost` + `finalize` must equal what `forward` produces on
// the identical step.
//
// RED-first: before the change none of `RejectionSamplerDeviceOutput`,
// `verify`, `CopyToHost` or `finalize` existed and this binary did not compile.
// After the change, mutating `finalize`'s `num_rejected` to `row_logits - 1`, or
// its emitted-token loop bound from `ns` to `ns - 1`, reds these cases while
// every pre-existing case in this file stays green — because those all go
// through `forward`, which is the same two halves in one call.
namespace {

// Drive the SPLIT halves by hand, exactly as the runner's copy-queue route does:
// issue the walk, copy the two outputs, wait, reduce on the host.
RejectionSamplerOutput RunSplit(const VerifyStep& s, int num_speculative_steps,
                                const std::vector<char>& is_chunked_prefilling) {
  Queue q = Q();
  Tensor logits = Tensor::Contiguous(const_cast<float*>(s.logits.data()), DType::kF32,
                                     Cpu(), {s.num_logits, static_cast<int64_t>(kVocab)});
  RejectionSampler sampler(num_speculative_steps);
  vllm::v1::RejectionSamplerDeviceOutput dev =
      sampler.verify(q, logits, s.draft_sampled, s.cu_num_logits);
  const int64_t rows = dev.num_reqs();
  const int64_t width = dev.width();
  std::vector<int32_t> host_sampled(static_cast<size_t>(rows * width));
  std::vector<int32_t> host_num_sampled(static_cast<size_t>(rows));
  dev.CopyToHost(q, host_sampled.data(), host_num_sampled.data());
  vt::GetBackend(dev.device().type).Synchronize(q);
  return RejectionSampler::finalize(host_sampled, width, host_num_sampled,
                                    s.cu_num_logits, is_chunked_prefilling);
}

// `forward` on the identical step, for the comparison.
RejectionSamplerOutput RunWhole(const VerifyStep& s, int num_speculative_steps,
                                const std::vector<char>& is_chunked_prefilling) {
  Queue q = Q();
  Tensor logits = Tensor::Contiguous(const_cast<float*>(s.logits.data()), DType::kF32,
                                     Cpu(), {s.num_logits, static_cast<int64_t>(kVocab)});
  RejectionSampler sampler(num_speculative_steps);
  return sampler.forward(q, logits, s.draft_sampled, s.cu_num_logits,
                         is_chunked_prefilling);
}

// Assert the two results id for id, not vector for vector, so a failure names
// the request and the position rather than saying "these differ".
void CheckSameTokens(const RejectionSamplerOutput& split,
                     const RejectionSamplerOutput& whole) {
  REQUIRE(split.num_sampled.size() == whole.num_sampled.size());
  REQUIRE(split.sampled_token_ids.size() == whole.sampled_token_ids.size());
  for (size_t r = 0; r < whole.num_sampled.size(); ++r) {
    INFO("request ", r);
    CHECK(split.num_sampled[r] == whole.num_sampled[r]);
    CHECK(split.num_rejected[r] == whole.num_rejected[r]);
    REQUIRE(split.sampled_token_ids[r].size() == whole.sampled_token_ids[r].size());
    for (size_t j = 0; j < whole.sampled_token_ids[r].size(); ++j) {
      INFO("token ", j);
      CHECK(split.sampled_token_ids[r][j] == whole.sampled_token_ids[r][j]);
    }
  }
}

}  // namespace

// A2-2.1 — the single-request shapes this file already covers, across the split.
TEST_CASE("A2-2: the split emits the same tokens as forward, one request") {
  SUBCASE("perfect match") {
    const VerifyStep s = MakeStep({{1, 2, 3}}, {{1, 2, 3, 4}});
    const RejectionSamplerOutput split = RunSplit(s, 3, {});
    CheckSameTokens(split, RunWhole(s, 3, {}));
    // Non-vacuous: the step really did emit four tokens, so the comparison above
    // was over something. Two empty results would compare equal.
    REQUIRE(split.sampled_token_ids[0] == std::vector<int32_t>{1, 2, 3, 4});
  }
  SUBCASE("early mismatch") {
    const VerifyStep s = MakeStep({{1, 2, 3}}, {{1, 5, 3, 4}});
    const RejectionSamplerOutput split = RunSplit(s, 3, {});
    CheckSameTokens(split, RunWhole(s, 3, {}));
    REQUIRE(split.sampled_token_ids[0] == std::vector<int32_t>{1, 5});
    REQUIRE(split.num_rejected[0] == 2);
  }
  SUBCASE("placeholder draft id") {
    const VerifyStep s = MakeStep({{-1}}, {{5, 6}});
    const RejectionSamplerOutput split = RunSplit(s, 1, {});
    CheckSameTokens(split, RunWhole(s, 1, {}));
    REQUIRE(split.sampled_token_ids[0] == std::vector<int32_t>{5});
  }
}

// A2-2.2 — THE MIXED STEP, and it is the case that matters.
//
// `num_reqs == 1` cannot separate a per-step reading of anything from a
// per-request one, which is the failure this repository has already paid for
// (#2710, and the note at the top of tests/vllm/v1/worker/
// test_combine_row_predicate.cpp). So the split is exercised on a batch whose
// rows carry DIFFERENT k: one row drafts nothing at all (k = 0, one expanded row
// — the shape the plain sampler would have handled), one accepts every draft,
// one rejects at its first. The expanded tensor is 1 + 3 + 4 = 8 rows for 3
// requests, so a reduction that used `num_reqs` where it needed `cu_num_logits`
// reads the wrong rows and this case says which request and which position.
TEST_CASE("A2-2: the split is token-identical on a MIXED step, num_reqs > 1") {
  const VerifyStep s = MakeStep({{}, {1, 2}, {7, 8, 9}},
                                {{4}, {1, 2, 6}, {7, 3, 9, 5}});
  const RejectionSamplerOutput split = RunSplit(s, 3, {});
  const RejectionSamplerOutput whole = RunWhole(s, 3, {});
  CheckSameTokens(split, whole);

  // Non-vacuous, and the three rows are genuinely different shapes:
  //   row 0: k = 0            -> one token, the plain greedy argmax
  //   row 1: k = 2, all accepted -> the two drafts plus the bonus
  //   row 2: k = 3, reject at 1  -> the first draft plus the target's argmax
  REQUIRE(split.sampled_token_ids[0] == std::vector<int32_t>{4});
  REQUIRE(split.num_rejected[0] == 0);
  REQUIRE(split.sampled_token_ids[1] == std::vector<int32_t>{1, 2, 6});
  REQUIRE(split.num_rejected[1] == 0);
  REQUIRE(split.sampled_token_ids[2] == std::vector<int32_t>{7, 3});
  REQUIRE(split.num_rejected[2] == 2);
  // The rows the sampler was handed are NOT one per request, which is the whole
  // reason the route predicate is a per-step one.
  REQUIRE(s.num_logits == 8);
}

// A2-2.3 — the chunked-prefill zeroing lives in `finalize`, the HOST half, and
// must survive the split. It never reaches the kernel, so this is the one place
// the split could have dropped a rule silently.
TEST_CASE("A2-2: the split keeps the chunked-prefill zeroing, on a mixed step") {
  const VerifyStep s = MakeStep({{1, 2}, {3}}, {{1, 9, 5}, {3, 4}});
  const std::vector<char> prefilling{1, 0};
  const RejectionSamplerOutput split = RunSplit(s, 2, prefilling);
  CheckSameTokens(split, RunWhole(s, 2, prefilling));
  REQUIRE(split.num_sampled[0] == 0);
  REQUIRE(split.num_rejected[0] == 0);
  REQUIRE(split.num_sampled[1] == 2);
}

// A2-2.4 — `verify` leaves its outputs ON THE DEVICE and waits for nothing. The
// CPU backend cannot show the overlap that buys, but it CAN show the shape the
// runner's copy-queue route depends on: the walk is issued, the result is a
// live object carrying [num_reqs, width], and the bytes only appear on the host
// when the caller asks for them.
TEST_CASE("A2-2: verify returns device-resident outputs, shaped num_reqs x width") {
  const VerifyStep s = MakeStep({{}, {1, 2}}, {{4}, {1, 2, 6}});
  Queue q = Q();
  Tensor logits = Tensor::Contiguous(const_cast<float*>(s.logits.data()), DType::kF32,
                                     Cpu(), {s.num_logits, static_cast<int64_t>(kVocab)});
  RejectionSampler sampler(2);
  vllm::v1::RejectionSamplerDeviceOutput dev =
      sampler.verify(q, logits, s.draft_sampled, s.cu_num_logits);
  CHECK(dev.num_reqs() == 2);
  // Upstream's row stride: num_speculative_steps + 1
  // (rejection_sampler_utils.py:1026-1028).
  CHECK(dev.width() == 3);
  CHECK(dev.device().type == DeviceType::kCPU);

  // The download is a separate act, and it can happen more than once from the
  // same device result — which is what lets the runner put the wait on a queue
  // the walk did not run on.
  std::vector<int32_t> a(6), b(6), na(2), nb(2);
  dev.CopyToHost(q, a.data(), na.data());
  dev.CopyToHost(q, b.data(), nb.data());
  vt::GetBackend(dev.device().type).Synchronize(q);
  CHECK(a == b);
  CHECK(na == nb);
  CHECK(na[0] == 1);  // k = 0 row: exactly the one greedy argmax
  CHECK(na[1] == 3);  // k = 2, all accepted: two drafts plus the bonus
}

// A2-2.5 — the device result is MOVE-ONLY and owns what the kernel touches.
// Moving it must not double-free and must not lose the buffers, because the
// runner's copy-queue route holds it across the fork/copy/wait window.
TEST_CASE("A2-2: the device result moves without losing or double-freeing its buffers") {
  const VerifyStep s = MakeStep({{1, 2}}, {{1, 2, 6}});
  Queue q = Q();
  Tensor logits = Tensor::Contiguous(const_cast<float*>(s.logits.data()), DType::kF32,
                                     Cpu(), {s.num_logits, static_cast<int64_t>(kVocab)});
  RejectionSampler sampler(2);
  vllm::v1::RejectionSamplerDeviceOutput src =
      sampler.verify(q, logits, s.draft_sampled, s.cu_num_logits);
  vllm::v1::RejectionSamplerDeviceOutput dst = std::move(src);
  CHECK(src.num_reqs() == 0);  // moved-from is empty and safe to destroy
  CHECK(dst.num_reqs() == 1);
  CHECK(dst.width() == 3);
  std::vector<int32_t> out(3), ns(1);
  dst.CopyToHost(q, out.data(), ns.data());
  vt::GetBackend(dst.device().type).Synchronize(q);
  CHECK(ns[0] == 3);
  CHECK(out == std::vector<int32_t>{1, 2, 6});
}

// SKIPPED (test-porting rule 6), tracked to M-mtp-3 (spec §5):
//   * test_stochastic_rejection_sample (test_rejection_sampler_utils.py:141) and
//     test_synthetic_rejection_sample (:215) — the Gumbel / probability-ratio
//     path is out of scope for I3 (greedy only).
//   * test_block_verification_rejection_sample (:325) and
//     test_block_verification_matches_standard (:372) — block verification.
//   * tests/v1/worker/test_gpu_rejection_sampler_i64.py:109 (>2^31 logits-buffer
//     indexing) — needs the block-verification buffer layout.
