# ROCm split sampling scratch ownership

Row: `BACKEND-ROCM`.
Issue: [#3062](https://github.com/mudler/vllm.cpp/issues/3062).
Contribution: [#3010](https://github.com/mudler/vllm.cpp/pull/3010), closing
[#3009](https://github.com/mudler/vllm.cpp/issues/3009).
Base: `066d2fd80f324c647709a56601bb382d526ebfe4`.

## Scope

Repair the split sampler's scratch ownership in `src/vt/rocm/rocm_sample.hip`.
Add dispatch tests and device tests through `vt::RandomSample`.
Do not change sampling arithmetic, defaults, thresholds, or backend lifetime.

## Source and gap

`RandomSampleKernelRocm:522-536` shares `part_score`, `part_idx`, and
`part_rows` across all queues and devices. Two streams can overwrite each
other's partials. Growing a batch discards allocation pointers.
`git log -S part_rows -- src/vt/rocm/rocm_sample.hip` identifies `be3af091b`
as the introducing commit.

The upstream sampling operation stays unchanged. The pinned vLLM source is
specified in `../upstream-sync.md`. The local mirror calls
`GumbelScore` and `ArgReduce` from `include/vt/sample_common.h`.
This repair changes allocator ownership, without porting new arithmetic.
The established local lifetime reference is `src/vt/grow_only_stream_scratch.h`,
which retains allocations that captured graphs can reference.

## Design

Use `GrowOnlyStreamScratch` with a process-unique queue identity.
`Queue::id` distinguishes different devices and reused native stream handles.
Allocate one slab containing 64 rows of 128 scores and indices per queue.
The split dispatch already rejects batches above 64 rows.
The slab uses 98,304 bytes. Its index slice stays aligned to `int64_t`.
Return pointers by value and publish the slab under the existing pool lock.
Keep the slab resident under the existing graph lifetime contract.
Do not free scratch when a queue disappears because a graph can still refer
to it. Queue churn still retains one slab per identity.

## Tests and gates

Commit the regression tests before the implementation.
The host dispatch harness compiles the production launcher with fake HIP
allocation and launch functions. It checks queue separation, device separation,
handle reuse, fixed allocation capacity, and production launch reachability.
It tests host ownership only and does not claim device execution.

The ROCm case calls `vt::RandomSample` from two queues with disjoint one-hot
distributions. Concurrent graph replay must preserve every requested token.
Capture at a small batch, run a larger batch, and replay the original graph
to verify pointer lifetime. Run the existing sampler cases unchanged.

Focused gates: the host dispatch test and `test_ops_sample` on ROCm with
`VT_FAST_RANDOM_SAMPLE=1 VT_SAMPLE_SPLIT=1`.
Repeat device cases with `VT_SAMPLE_SPLIT=0` as the reference path.
The operator runs the HIP build and full CTest suite under its device lease.
Run `scripts/agent-preflight.sh` and inspect its failed-gate count.
The fresh reviewer mutates ownership, allocation capacity, and the production
call site and requires the applicable focused tests to fail.

## Risks and stop conditions

Stop if the repair requires different sampling behavior or backend lifetime.
Device execution stays PENDING until the operator supplies lease evidence.
The pool does not serialize two host callers interleaving phases on one queue.
Callers continue to own submission serialization for a shared queue.

## Now

ACTIVE: spec committed before regression and implementation.
No correctness or performance gate result is claimed yet.
