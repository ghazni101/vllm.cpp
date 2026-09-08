#!/usr/bin/env python3
"""Compile the production ROCm sampling dispatcher with a fake HIP runtime.

Only launch syntax is adapted. These tests check host allocation ownership,
not kernel arithmetic or GPU execution. Device coverage lives in test_ops_sample.
"""

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]

PRELUDE = r"""
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>
#include "vt/grow_only_stream_scratch.h"
using hipStream_t = void*;
struct Device { int index; };
struct Queue { Device device; void* handle; uint64_t id; };
struct Tensor {
  int64_t shape[2];
  template <typename T> T* Ptr() const { return nullptr; }
};
constexpr int kSampleSplitBlocks = 128, kVocabBlock = 1024;
bool FastRandomSampleEnabled() { return true; }
bool SampleSplitEnabled() { return true; }
hipStream_t AsStream(const Queue& q) { return q.handle; }
void Check(int result, const char*) { assert(result == 0); }
int hipGetLastError() { return 0; }
std::vector<void*> allocations;
std::vector<size_t> allocation_sizes;
int hipMallocAsync(void** ptr, size_t bytes, hipStream_t) {
  *ptr = std::malloc(bytes);
  assert(*ptr);
  allocations.push_back(*ptr);
  allocation_sizes.push_back(bytes);
  return 0;
}
struct Launch { float* score; int64_t* index; };
std::vector<Launch> phase_a, phase_b;
void RandomSampleSplitAK(float* score, int64_t* index, float*, int64_t*, int64_t, int) {
  phase_a.push_back({score, index});
}
void RandomSampleSplitBK(int64_t*, float* score, int64_t* index, int) {
  phase_b.push_back({score, index});
}
void RandomSampleKernelSlow(int64_t*, float*, int64_t*, int64_t) { assert(false); }
void RandomSampleK(int64_t*, float*, int64_t*, int64_t) { assert(false); }
"""

MAIN = r"""
int main(int argc, char** argv) {
  assert(argc == 2);
  const std::string mode = argv[1];
  Tensor out{}, seeds{};
  auto run = [&](Queue& q, int64_t rows) {
    Tensor probs{{rows, 8192}};
    const size_t before = phase_a.size();
    RandomSampleKernelRocm(q, out, probs, seeds);
    assert(phase_a.size() == before + 1);
    assert(phase_b.size() == phase_a.size());
    assert(phase_a.back().score == phase_b.back().score);
    assert(phase_a.back().index == phase_b.back().index);
    assert(reinterpret_cast<uintptr_t>(phase_a.back().index) % alignof(int64_t) == 0);
    return phase_a.back();
  };
  Queue a{{0}, reinterpret_cast<void*>(1), 101};
  Queue b{{0}, reinterpret_cast<void*>(2), 102};
  if (mode == "devices") {
    b.device.index = 1;
    b.handle = a.handle; // Null/default or reused native handles must not alias.
  }
  if (mode == "reuse") b.handle = a.handle;
  if (mode == "growth") {
    const auto first = run(a, 1);
    const auto count = allocations.size();
    for (int64_t rows = 2; rows <= 64; ++rows) {
      const auto next = run(a, rows);
      assert(first.score == next.score);
      assert(first.index == next.index);
    }
    assert(allocations.size() == count);
    size_t bytes = 0;
    for (size_t size : allocation_sizes) bytes += size;
    assert(bytes == 64 * 128 * (sizeof(float) + sizeof(int64_t)));
  } else {
    const auto first = run(a, 64);
    const auto second = run(b, 64);
    assert(first.score != second.score);
    assert(first.index != second.index);
    const auto again = run(a, 1);
    assert(first.score == again.score);
    assert(first.index == again.index);
  }
  for (void* allocation : allocations) std::free(allocation);
}
"""


class RocmSampleScratchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="rocm-sample-dispatch-")
        cls.addClassCleanup(cls.temp.cleanup)
        source = (ROOT / "src/vt/rocm/rocm_sample.hip").read_text()
        begin = source.index("void RandomSampleKernelRocm(")
        end = source.index("void ApplyPenaltiesKernelRocm(", begin)
        dispatch = re.sub(r"<<<.*?>>>", "", source[begin:end], flags=re.S)
        path = Path(cls.temp.name) / "dispatch.cpp"
        path.write_text(PRELUDE + dispatch + MAIN)
        cls.binary = Path(cls.temp.name) / "dispatch"
        subprocess.run(
            [os.environ.get("CXX", "c++"), "-std=c++17", "-pthread", "-I",
             str(ROOT / "src"), str(path), "-o", str(cls.binary)], check=True,
        )

    def run_case(self, mode):
        result = subprocess.run([str(self.binary), mode], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_queues_have_independent_partials(self):
        self.run_case("queues")

    def test_devices_have_independent_partials(self):
        self.run_case("devices")

    def test_reused_native_handles_do_not_reuse_scratch(self):
        self.run_case("reuse")

    def test_batch_growth_preserves_captured_pointers_without_more_allocations(self):
        self.run_case("growth")


if __name__ == "__main__":
    unittest.main()
