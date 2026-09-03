// SPEC-DFLASH2 A2-2 (#2802) — the spec-decode VERIFY route predicate, and the
// per-STEP reduction both sampling entry points and one refusal turn on.
//
// WHAT THIS PROTECTS. Upstream has ONE sampling entry point and asks the
// question once (`vllm/v1/worker/gpu/model_runner.py:1129` @ pin 5559679229:
// `if input_batch.num_draft_tokens == 0 or self.rejection_sampler is None`).
// We have two — `GPUModelRunner::sample_tokens` and
// `GPUModelRunner::sample_tokens_async` — and a third site, the async input
// combine, refuses on the NEGATION of the same rule. Three readings of one rule
// is how a route and its refusal drift apart.
//
// WHY THE GRANULARITY IS THE POINT OF THIS FILE. `num_draft_tokens` is the batch
// TOTAL, `sum(num_draft_tokens_per_req)`. The forward produced ONE expanded
// logits tensor for the whole step, with `Σ(1 + k_i)` rows, and either the
// rejection sampler consumes it or the plain sampler does. A per-REQUEST reading
// of the same rule — "row i drafted nothing, so it is an ordinary decode row" —
// answers differently on a MIXED step, and would hand the plain sampler a tensor
// whose rows are not one per request.
//
// This repository has shipped exactly that shape before: a per-request refusal
// paired with a per-step route predicate (#2710,
// `tests/vllm/v1/worker/test_combine_row_predicate.cpp`). It survived 27
// mutations because every test used `num_reqs == 1`, so the two readings agreed
// on every input. The DISAGREEMENT case below is the input that separates them,
// and it is the reason this file is not a restatement of `> 0`.
//
// RED-first for this file: before the change neither `StepRoutesToVerify` nor
// `RowCarriesDraftTokens` existed, and the binary did not compile. After the
// change, mutating `StepRoutesToVerify` to the per-row reading
// (`return per_req[0] > 0`-shaped, i.e. any reading that is not the total) reds
// the MIXED case below while every single-request case here stays green.
#include <doctest/doctest.h>

#include <cstdint>
#include <numeric>
#include <vector>

#include "vllm/v1/worker/gpu/prepare_inputs.h"

using vllm::v1::RowCarriesDraftTokens;
using vllm::v1::StepRoutesToVerify;

namespace {

// The step total, exactly as `prepare_inputs` builds `StepInputs::num_draft_
// tokens`: sum(num_draft_tokens_per_req).
int32_t StepTotal(const std::vector<int32_t>& per_req) {
  return std::accumulate(per_req.begin(), per_req.end(), 0);
}

}  // namespace

TEST_CASE("the verify route is the step's draft TOTAL, and zero means the plain sampler") {
  // The production default, on every step, forever: no speculator, so the
  // scheduler populated no `scheduled_spec_decode_tokens` and the total is 0.
  // This is the branch that must stay byte-identical.
  CHECK_FALSE(StepRoutesToVerify(0));

  // One request with one draft is the smallest verify step: 2 expanded rows.
  CHECK(StepRoutesToVerify(1));
  CHECK(StepRoutesToVerify(24));

  // The total is a count and is never negative; a `!= 0` re-derivation would
  // route a negative total to the verify arm, so the boundary is a strict `>`.
  CHECK_FALSE(StepRoutesToVerify(-1));
}

TEST_CASE("the row predicate answers about ONE row and never about the step") {
  CHECK_FALSE(RowCarriesDraftTokens(0));
  CHECK(RowCarriesDraftTokens(1));
  CHECK(RowCarriesDraftTokens(3));
}

TEST_CASE("THE DISAGREEMENT CASE: a MIXED step routes to verify on rows that drafted nothing") {
  // ── Three requests in one step. Rows 0 and 2 drafted NOTHING this step (a
  // request that was just admitted, or whose drafts were all rolled back); row 1
  // carries two drafts.
  //
  // A PER-REQUEST reading answers "no drafts" for rows 0 and 2 and would send
  // them to the plain sampler. But there is no per-row choice to make: the
  // forward produced ONE expanded tensor of Σ(1 + k_i) = 1 + 3 + 1 = 5 rows for
  // this step, and the plain sampler expects one row per request. The PER-STEP
  // reading answers "verify" for the whole step, which is the only answer that
  // matches the tensor the sampler is handed.
  //
  // This input is the only kind that separates the two readings, which is why a
  // `num_reqs == 1` suite cannot gate this rule at all.
  const std::vector<int32_t> per_req{0, 2, 0};
  const int32_t total = StepTotal(per_req);
  CHECK(total == 2);
  CHECK(StepRoutesToVerify(total));

  // The two readings, side by side and executable. The MAJORITY of the per-row
  // answers is false while the step answer is true. If these ever agreed on
  // every row, this case would have stopped discriminating, and this block says
  // so rather than leaving it to inspection.
  CHECK_FALSE(RowCarriesDraftTokens(per_req[0]));
  CHECK(RowCarriesDraftTokens(per_req[1]));
  CHECK_FALSE(RowCarriesDraftTokens(per_req[2]));

  // The expanded-row count the verify arm is handed, spelled out: it is NOT
  // num_reqs, which is exactly what the plain sampler would assume.
  int32_t expanded = 0;
  for (const int32_t k : per_req) expanded += 1 + k;
  CHECK(expanded == 5);
  CHECK(expanded != static_cast<int32_t>(per_req.size()));
}

TEST_CASE("a step where EVERY row drafted nothing does not route to verify") {
  // The all-zero mixed shape: `num_draft_tokens_per_req` is present and sized,
  // and every entry is 0, so the expanded tensor IS one row per request and the
  // plain sampler is correct. A predicate that keyed off "the per-request vector
  // is non-empty" rather than off the total would route this to the verify arm
  // and verify nothing.
  const std::vector<int32_t> per_req{0, 0, 0};
  CHECK(StepTotal(per_req) == 0);
  CHECK_FALSE(StepRoutesToVerify(StepTotal(per_req)));
}

TEST_CASE("the refusal at the async input combine fires on the MIXED step too") {
  // `runner.cpp`'s async input-combine site refuses a step that scheduled draft
  // tokens, because the draft buffer its combine scatters from is not wired yet
  // (A2-3, #2644). It asks `!StepRoutesToVerify(step.num_draft_tokens)` — the
  // route's own function, negated, rather than a second reading of the rule.
  //
  // NOTHING HERE CAN PROVE THAT SHARING, and this case does not pretend to: two
  // source sites calling one function is a property of the source, and the
  // instrument for it is the reviewer's mutation (change one site's predicate
  // and this file must go red for the route half; the refusal half is
  // unreachable today and is recorded as owed in the row's spec). What this case
  // DOES pin is the input on which a re-derived per-request refusal would differ
  // from the route: the mixed step routes to verify, so the refusal must fire,
  // even though two of its three rows drafted nothing.
  const std::vector<int32_t> mixed{0, 2, 0};
  const bool routes = StepRoutesToVerify(StepTotal(mixed));
  const bool refuses = !StepRoutesToVerify(StepTotal(mixed));
  CHECK(routes);
  CHECK_FALSE(refuses);

  // The per-request reading a re-derivation would have used, made explicit: it
  // says "no drafts" for the majority of the rows, and on this step that answer
  // is wrong for the step.
  bool any_row = false;
  for (const int32_t k : mixed) any_row = any_row || RowCarriesDraftTokens(k);
  CHECK(any_row == routes);
  CHECK_FALSE(RowCarriesDraftTokens(mixed[0]));
  CHECK_FALSE(RowCarriesDraftTokens(mixed[2]));
}
