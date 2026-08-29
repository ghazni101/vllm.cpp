// T37 (GFX1100-TG200) microbench: decode GEMV shapes through the REAL
// kMatmulBTQuant dispatch (m == 1), sweeping the VT_GEMV_WARPS (bit-identical
// launch geometry) and VT_GEMV_SPLITK (split-K, reduction-order change)
// knobs. Executable-only (NOT add_test) per the paged-attn-wmma precedent:
// this is a measurement tool for the T37 lever round, not a gate.
//
// Shapes are the trace-attributed engine launches (T34 budget):
//   ssm_out     Q5_K n=2560  K=4096 (24 launches/token, 54% BW)
//   attn_gate   Q4_K n=4096  K=2560 (24 launches/token, 44% BW)
//   ffn_down_q4 Q4_K n=2560  K=9216 (16 launches/token)
//   ffn_down_q6 Q6_K n=2560  K=9216 (16 launches/token)
//   gate_up     Q4_K n=18432 K=2560 (32 launches/token, 57% BW, control)
//
// Each config: 10 warm-up + 50 timed launches, mean us reported.
// Split-K configs also byte-compare their f32 output against the baseline's
// and report the mismatching-element count (ULP-level differences expected:
// the cross-split combine reorders the float sum; integer core is exact).
#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/quant.h"
#include "vt/rocm/rocm_runtime.h"
#include "vt/tensor.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

Device GpuDev() { return Device{DeviceType::kROCM, 0}; }

struct Shape {
  const char* name;
  DType wdt;
  int64_t n;
  int64_t k;
};

const Shape kShapes[] = {
    {"ssm_out_q5", DType::kQ5_K, 2560, 4096},
    {"attn_gate_q4", DType::kQ4_K, 4096, 2560},
    {"ffn_down_q4", DType::kQ4_K, 2560, 9216},
    {"ffn_down_q6", DType::kQ6_K, 2560, 9216},
    {"gate_up_q4", DType::kQ4_K, 18432, 2560},
};

int64_t T37BlockBytes(DType dt) {
  switch (dt) {
    case DType::kQ4_K: return 144;
    case DType::kQ5_K: return 176;
    case DType::kQ6_K: return 210;
    default: return 0;
  }
}

std::vector<uint8_t> RandomWeights(DType dt, int64_t bytes, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint8_t> b(static_cast<size_t>(bytes));
  for (auto& x : b) x = static_cast<uint8_t>(rng() & 0xFF);
  // Keep the f16 scale words small-but-normal so outputs stay bounded.
  const int step = dt == DType::kQ6_K ? 210 : (dt == DType::kQ5_K ? 176 : 144);
  for (size_t off = 0; off + 3 < b.size(); off += static_cast<size_t>(step)) {
    const uint16_t h = vt::F32ToF16(0.0125f);
    std::memcpy(b.data() + off, &h, 2);
    if (dt != DType::kQ6_K) {
      const uint16_t m = vt::F32ToF16(0.0075f);
      std::memcpy(b.data() + off + 2, &m, 2);
    }
  }
  return b;
}

Tensor DevTensor(void* p, DType dt, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = p;
  t.dtype = dt;
  t.device = GpuDev();
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

void SetEnv(const char* k, const char* v) {
  if (v == nullptr) ::unsetenv(k);
  else ::setenv(k, v, 1);
}

double TimeUs(Backend& gpu, Queue& q, const std::function<void()>& launch,
              int warm, int iters) {
  for (int i = 0; i < warm; ++i) launch();
  gpu.Synchronize(q);
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) launch();
  gpu.Synchronize(q);
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

}  // namespace

TEST_CASE("t37 gemv microbench (timing only)") {
  if (!vt::rocm::DeviceAvailable()) {
    MESSAGE("no AMD GPU; skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue q = gpu.CreateQueue();

  // The campaign's decode routing (m == 1 GEMVs).
  SetEnv("VT_GEMV_MMVQ", "1");
  SetEnv("VT_QUANT_Q8K_WARP", "1");
  SetEnv("VT_PREFILL_TILE", nullptr);
  SetEnv("VT_GEMV_WARPS", nullptr);
  SetEnv("VT_GEMV_SPLITK", nullptr);

  for (const Shape& sh : kShapes) {
    const int64_t nsb = sh.k / 256;
    const int64_t wb = nsb * T37BlockBytes(sh.wdt);
    auto w = RandomWeights(sh.wdt, sh.n * wb, 0x5EEDU);
    // The activation tensor is [1, K] bf16 — allocate its declared bytes.
    std::vector<uint8_t> act(static_cast<size_t>(sh.k) * 2, 0x11);

    void* d_w = gpu.Alloc(w.size());
    gpu.Copy(q, d_w, w.data(), w.size());
    void* d_a = gpu.Alloc(act.size());
    gpu.Copy(q, d_a, act.data(), act.size());
    const size_t out_bytes = static_cast<size_t>(sh.n) * 4;
    void* d_o = gpu.Alloc(out_bytes);
    std::vector<float> base_out(static_cast<size_t>(sh.n));

    Tensor at = DevTensor(d_a, DType::kBF16, {1, sh.k});
    Tensor bt = DevTensor(d_w, sh.wdt, {sh.n, sh.k});
    Tensor ot = DevTensor(d_o, DType::kF32, {1, sh.n});

    auto launch = [&]() { vt::MatmulBTQuant(q, ot, at, bt); };

    // Baseline (also the reference output for byte-comparisons).
    SetEnv("VT_GEMV_WARPS", nullptr);
    SetEnv("VT_GEMV_SPLITK", nullptr);
    const double us = TimeUs(gpu, q, launch, 10, 50);
    gpu.Copy(q, base_out.data(), d_o, out_bytes);
    gpu.Synchronize(q);
    std::printf("T37BENCH %-14s %-18s %8.2f us\n", sh.name, "baseline", us);

    struct Knob {
      const char* w;
      const char* s;
      const char* label;
    };
    const Knob knobs[] = {
        {"2", nullptr, "warps2"},
        {"4", nullptr, "warps4"},
        {nullptr, "2", "splitk2"},
        {nullptr, "4", "splitk4"},
        {nullptr, "8", "splitk8"},
    };
    for (const Knob& kn : knobs) {
      SetEnv("VT_GEMV_WARPS", kn.w);
      SetEnv("VT_GEMV_SPLITK", kn.s);
      const double u2 = TimeUs(gpu, q, launch, 10, 50);
      gpu.Synchronize(q);
      std::vector<float> got(static_cast<size_t>(sh.n));
      gpu.Copy(q, got.data(), d_o, out_bytes);
      gpu.Synchronize(q);
      int64_t mismatches = 0;
      double max_abs = 0.0;
      for (int64_t i = 0; i < sh.n; ++i) {
        if (std::memcmp(&got[static_cast<size_t>(i)],
                        &base_out[static_cast<size_t>(i)], 4) != 0) {
          ++mismatches;
          max_abs =
              std::max(max_abs, static_cast<double>(std::abs(
                                    got[static_cast<size_t>(i)] -
                                    base_out[static_cast<size_t>(i)])));
        }
      }
      std::printf("T37BENCH %-14s %-18s %8.2f us  (%.2fx)  mism=%lld maxdiff=%g\n",
                  sh.name, kn.label, u2, us / u2,
                  static_cast<long long>(mismatches), max_abs);
    }
    SetEnv("VT_GEMV_WARPS", nullptr);
    SetEnv("VT_GEMV_SPLITK", nullptr);
    gpu.Free(d_w);
    gpu.Free(d_a);
    gpu.Free(d_o);
  }
  gpu.Synchronize(q);
  gpu.DestroyQueue(q);
}
