// ROCm keep-quant GEMM gate (GFX1100-TG200). The campaign spec names
// `tests/vt/test_rocm_quant_dot.cpp` as the quant-path lever gate; until T4a
// that file DID NOT EXIST — the GPU-parity cases lived in
// tests/vt/test_cuda_quant_dot.cpp behind HasCuda() and so SKIPPED on this
// ROCm-only box (the exact T3a blind spot: op-level green while the engine
// produced garbage). This file is the fix: a focused gate for the ROCm
// kMatmulBTQuant provider (src/vt/rocm/rocm_grouped_gemm.hip) guarded on ROCM
// availability, never on CUDA.
//
// RED-first contract: before the dispatch arm exists VT_GEMV_MMVQ=1 is inert,
// so ON==OFF trivially; the dispatch-gate cases below fail if the flag never
// engages the arm.
//
// T4a REPAIR ROUND numerics contract: the arm must be BYTE-IDENTICAL TO THE
// DEFAULT (warp-reduction) KERNEL — the engine-safety property the FIRST
// round lacked. Round 1 was bit-exact vs the CPU ORACLE while the ENGINE
// degraded: oracle association != baseline tree association, and greedy
// near-ties flipped (extended ON-vs-OFF sweep red at N=2304..248320,
// isolated first-diverging rows). This gate therefore asserts ON==OFF raw
// byte identity on every case below, PLUS the standard 1e-6 NMSE band vs
// the CPU oracle for the ON arm (the same band the default arm is held to).
//
// Skips cleanly (returns) when the build has HIP but the box has no AMD GPU,
// so the CPU CI leg stays green.
#include <doctest/doctest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
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

namespace vt::rocm {
void MmvqQuantScratchForTesting(Queue& q, void* dst, const Tensor& a,
                                bool fused_semantics);

// T4a REPAIR-ROUND-2 routing witness (review findings F1/F2): the HOST-side
// dispatch counters exposed by rocm_grouped_gemm.hip. ON and OFF arms are
// BIT-EQUAL on outputs by design, so no output comparison can witness which
// dispatch branch a call took -- these integer counters can.
struct MmvqRouteCounts {
  long long baseline;    // KQuantGemmK warp-reduction dispatches
  long long gemv_mmvq;   // non-fused MMVQ GEMV dispatches (standalone quant)
  long long gemv_fused;  // fused-fold sub-branch dispatches
};
MmvqRouteCounts MmvqRouteCountsForTesting();
void MmvqResetRouteCountsForTesting();
}  // namespace vt::rocm
namespace {

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Device GpuDev() { return Device{DeviceType::kROCM, 0}; }

// test-backend-ops.cpp:4277 via test_cuda_quant_dot.cpp:78 — the NMSE band the
// DEFAULT (warp-reduction) arm is held to vs the CPU oracle. Only the
// VT_GEMV_MMVQ=1 arm claims bit-exactness.


constexpr double kMaxNmseVsCpu = 1e-6;

struct WeightCase {
  DType dtype;
  int64_t block_elems;
  int64_t block_bytes;
  int d_off;
  int dmin_off;
  const char* name;
};

// Same table discipline as test_cuda_quant_dot.cpp:113 (offsets restated from
// ggml-common.h): the three K-quants the ROCm provider serves natively.
const WeightCase kKQuantCases[] = {
    {DType::kQ4_K, 256, 144, 0, 2, "q4_K"},
    {DType::kQ5_K, 256, 176, 0, 2, "q5_K"},
    {DType::kQ6_K, 256, 210, 208, -1, "q6_K"},
};

std::vector<uint8_t> RandomBlocks(const WeightCase& c, int64_t nblocks,
                                  uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint8_t> bytes(static_cast<size_t>(nblocks * c.block_bytes));
  for (uint8_t& b : bytes) b = static_cast<uint8_t>(rng() & 0xFF);
  for (int64_t i = 0; i < nblocks; ++i) {
    uint8_t* blk = bytes.data() + i * c.block_bytes;
    auto put_f16 = [&](int off, float v) {
      const uint16_t h = vt::F32ToF16(v);
      std::memcpy(blk + off, &h, sizeof(h));
    };
    const float jitter = 1.0F + 0.05F * static_cast<float>(i % 7);
    put_f16(c.d_off, 0.0125F * jitter);
    if (c.dmin_off >= 0) put_f16(c.dmin_off, 0.0075F * jitter);
  }
  return bytes;
}

void GenerateData(float offset, size_t n, float* dst) {
  for (size_t i = 0; i < n; i++)
    dst[i] = 0.1F + 2 * std::cos(static_cast<float>(i) + offset);
}

double Nmse(const std::vector<float>& got, const std::vector<float>& ref) {
  double num = 0, den = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
    num += d * d;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  return num / den;
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

struct EnvGuard {
  explicit EnvGuard(bool on) { ::setenv("VT_GEMV_MMVQ", on ? "1" : "0", 1); }
  ~EnvGuard() { ::unsetenv("VT_GEMV_MMVQ"); }
};

}  // namespace

TEST_CASE("ROCm K-quant decode arm (VT_GEMV_MMVQ=1) is BYTE-EXACT vs the default arm and within the oracle NMSE band") {
  if (!vt::rocm::DeviceAvailable()) {
    MESSAGE("no AMD GPU on this host; ROCm keep-quant gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};

  // m=1 (the decode shape the arm serves), Q4_K/Q5_K/Q6_K, nsb edges
  // (nsb=1 -> one partial pass; nsb=3 -> ragged tail pass) and odd-but-valid
  // N (warp-guard edge).
  for (const WeightCase& c : kKQuantCases) {
    for (int64_t nsb : {int64_t{1}, int64_t{3}, int64_t{10}}) {
      const int64_t k = nsb * c.block_elems;
      for (int64_t n : {int64_t{1}, int64_t{7}, int64_t{129}}) {
        for (uint32_t seed : {0x5EEDU, 0xA11CEU}) {
          CAPTURE(c.name);
          CAPTURE(k);
          CAPTURE(n);
          CAPTURE(seed);

          std::vector<uint8_t> wq = RandomBlocks(c, n * nsb, seed);
          // Engine-realistic dtypes too: the model runs these projections with
          // bf16 activations and bf16 outputs; f32-only tests were the blind
          // spot that let the first fused build pass ops while the engine
          // degraded. Activation storage is generated in `adt`.
          for (DType adt : {DType::kF32, DType::kBF16, DType::kF16}) {
          for (DType odt : {DType::kF32, DType::kBF16}) {
          CAPTURE(adt);
          CAPTURE(odt);
          std::vector<float> af(static_cast<size_t>(k));
          GenerateData(static_cast<float>(seed) + 0.5F * static_cast<float>(int(adt)),
                       af.size(), af.data());
          std::vector<uint8_t> abuf(af.size() *
                                    (adt == DType::kF32 ? 4 : 2));
          for (size_t i2 = 0; i2 < af.size(); ++i2) {
            if (adt == DType::kF32)
              std::memcpy(abuf.data() + 4 * i2, &af[i2], 4);
            else if (adt == DType::kBF16) {
              const uint16_t h = vt::F32ToBF16(af[i2]);
              std::memcpy(abuf.data() + 2 * i2, &h, 2);
            } else {
              const uint16_t h = vt::F32ToF16(af[i2]);
              std::memcpy(abuf.data() + 2 * i2, &h, 2);
            }
          }

          // --- CPU oracle (host tensors, generic nrc==1 tier at m==1) -------
          std::vector<float> cpu_out(static_cast<size_t>(n), 0.0F);
          {
            Tensor at = Tensor::Contiguous(abuf.data(), adt, Cpu(), {1, k});
            Tensor bt =
                Tensor::Contiguous(wq.data(), DType::kF32, Cpu(), {n, k});
            bt.dtype = c.dtype;
            Tensor ot =
                Tensor::Contiguous(cpu_out.data(), DType::kF32, Cpu(), {1, n});
            vt::MatmulBTQuant(cq, ot, at, bt);
          }

          // --- ROCm path: BOTH arms at this shape; ON must equal OFF
          // byte-for-byte (raw output buffer), and ON stays within the
          // 1e-6 NMSE band vs the CPU oracle (the default arm's band) ----
          const size_t oesz = odt == DType::kF32 ? 4 : 2;
          void* d_a = gpu.Alloc(abuf.size());
          void* d_w = gpu.Alloc(wq.size());
          gpu.Copy(gq, d_a, abuf.data(), abuf.size());
          gpu.Copy(gq, d_w, wq.data(), wq.size());
          std::vector<std::vector<float>> arm_out(2);
          std::vector<std::vector<unsigned char>> arm_raw(2);
          for (int arm = 0; arm < 2; ++arm) {
            void* d_o = gpu.Alloc(oesz * static_cast<size_t>(n));
            {
              EnvGuard on(arm == 1);
              Tensor at = DevTensor(d_a, adt, {1, k});
              Tensor bt = DevTensor(d_w, c.dtype, {n, k});
              Tensor ot = DevTensor(d_o, odt, {1, n});
              vt::MatmulBTQuant(gq, ot, at, bt);
              arm_raw[arm].resize(oesz * static_cast<size_t>(n));
              gpu.Copy(gq, arm_raw[arm].data(), d_o, arm_raw[arm].size());
              arm_out[arm].resize(static_cast<size_t>(n), 0.0F);
              for (size_t i2 = 0; i2 < arm_out[arm].size(); ++i2)
                arm_out[arm][i2] =
                    odt == DType::kF32
                        ? reinterpret_cast<float*>(arm_raw[arm].data())[i2]
                        : vt::BF16ToF32(
                              reinterpret_cast<uint16_t*>(arm_raw[arm].data())[i2]);
              gpu.Synchronize(gq);
            }
            gpu.Free(d_o);
          }
          gpu.Free(d_a);
          gpu.Free(d_w);

          // ON arm must be BYTE-IDENTICAL to the default kernel
          CHECK(std::memcmp(arm_raw[0].data(), arm_raw[1].data(),
                            arm_raw[0].size()) == 0);
          // CPU side mirrors the output dtype conversion exactly
          std::vector<float> cpu_ref(cpu_out.size());
          for (size_t i2 = 0; i2 < cpu_out.size(); ++i2)
            cpu_ref[i2] = odt == DType::kF32
                              ? cpu_out[i2]
                              : vt::BF16ToF32(vt::F32ToBF16(cpu_out[i2]));
          const double nmse_on = Nmse(arm_out[1], cpu_ref);
          CAPTURE(nmse_on);
          CHECK(nmse_on <= kMaxNmseVsCpu);
          }  // odt
          }  // adt
        }
      }
    }
  }
  gpu.DestroyQueue(gq);
}

TEST_CASE("T4a repair: MULTI-M calls stay byte-exact ON-vs-OFF (the m-gate red)") {
  if (!vt::rocm::DeviceAvailable()) {
    MESSAGE("no AMD GPU on this host; ROCm keep-quant gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  // RED-first for the TRUE defect-1: the arm's non-fused branch originally
  // gated ONLY the fused fold on m==1, so ENGINE PREFILL calls (observed
  // m=39) took the GEMV kernel, which writes row 0 only -- rows 1..m-1 of
  // the output were left UNWRITTEN while every op-level test (m==1) stayed
  // green. This case runs m>1 batches and asserts the FULL m x n output is
  // byte-identical between the arms.
  struct MCase { DType wt; int64_t m, n, k; };
  const std::vector<MCase> cases = {
      {DType::kQ4_K, 3, 7, 2560},
      {DType::kQ4_K, 39, 18432, 2560},   // the engine's observed prefill shape
      {DType::kQ6_K, 5, 129, 9216},
      {DType::kQ6_K, 2, 248320, 2560},   // lm_head-class with m=2
  };
  for (const MCase& mc : cases) {
    const WeightCase* c = nullptr;
    for (const WeightCase& wc : kKQuantCases)
      if (wc.dtype == mc.wt) c = &wc;
    const int64_t nsb = mc.k / c->block_elems;
    CAPTURE(mc.m);
    CAPTURE(mc.n);
    CAPTURE(mc.k);
    std::vector<uint8_t> wq = RandomBlocks(*c, mc.n * nsb, 0x5EEDU);
    const size_t aesz = 2;  // bf16 activations, engine-realistic
    std::vector<uint16_t> abuf(static_cast<size_t>(mc.m * mc.k));
    for (size_t i = 0; i < abuf.size(); ++i)
      abuf[i] = static_cast<uint16_t>((i * 2654435761u) >> 11);
    void* d_w = gpu.Alloc(wq.size());
    void* d_a = gpu.Alloc(abuf.size() * aesz);
    gpu.Copy(gq, d_w, wq.data(), wq.size());
    gpu.Copy(gq, d_a, abuf.data(), abuf.size() * aesz);
    constexpr size_t kOesz = 2;
    std::vector<std::vector<unsigned char>> outs(2);
    for (int arm = 0; arm < 2; ++arm) {
      void* d_o = gpu.Alloc(kOesz * static_cast<size_t>(mc.m * mc.n));
      {
        // Canary-fill so any UNWRITTEN row is detected rather than
        // coincidentally matching stale allocation contents.
        std::vector<unsigned char> canary(kOesz * static_cast<size_t>(mc.m * mc.n),
                                          arm == 1 ? 0xAB : 0xCD);
        gpu.Copy(gq, d_o, canary.data(), canary.size());
        EnvGuard guard(arm == 1);
        Tensor at = DevTensor(d_a, DType::kBF16, {mc.m, mc.k});
        Tensor bt = DevTensor(d_w, c->dtype, {mc.n, mc.k});
        Tensor ot = DevTensor(d_o, DType::kBF16, {mc.m, mc.n});
        vt::MatmulBTQuant(gq, ot, at, bt);
        outs[arm].resize(kOesz * static_cast<size_t>(mc.m * mc.n));
        gpu.Copy(gq, outs[arm].data(), d_o, outs[arm].size());
        gpu.Synchronize(gq);
      }
      gpu.Free(d_o);
    }
    gpu.Free(d_w);
    gpu.Free(d_a);
    size_t first_bad = outs[0].size();
    for (size_t i = 0; i < outs[0].size(); ++i)
      if (outs[0][i] != outs[1][i]) { first_bad = i; break; }
    CAPTURE(first_bad);
    CHECK(outs[0] == outs[1]);
  }
  gpu.DestroyQueue(gq);
}

TEST_CASE("T4a repair: ON-vs-OFF BYTE identity over the ENGINE shape set (incl. lm_head-sized N)") {
  if (!vt::rocm::DeviceAvailable()) {
    MESSAGE("no AMD GPU on this host; ROCm keep-quant gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();

  // The REAL (dtype, N, K) set the engine serves, from the GGUF tensor
  // manifest of the acceptance checkpoint (Qwen3.5-4B-Q4_K_M: ne0=K,
  // ne1=N) plus the operator's ON-capture grids (grid = ceil(N/4) at
  // kGemvWarps=4 -> grid 80/576/256/7760) and the contract-named
  // lm_head-class probes. Defect-1 hypothesis under test: a 32-bit
  // offset/index overflow once N*w_row_bytes grows past 2^31 (lm_head
  // N x nsb x 210B ~= 0.5 GB at these shapes -- near the int32 edge).
  struct ShapeCase {
    DType wt;
    const char* name;
    int64_t n, k;
    bool all_act_dtypes;  // giants run bf16-only (engine-realistic) to bound suite time
  };
  const std::vector<ShapeCase> shapes = {
      {DType::kQ6_K, "q6_K grid=80", 320, 2560, true},
      {DType::kQ4_K, "q4_K grid=80", 320, 2560, true},
      {DType::kQ4_K, "q4_K grid=576", 2304, 2560, true},
      {DType::kQ6_K, "q6_K blk.out", 1024, 2560, true},
      {DType::kQ4_K, "q4_K ffn-out", 2560, 4096, true},
      {DType::kQ5_K, "q5_K ffn-out", 2560, 4096, true},
      {DType::kQ5_K, "q5_K gate_up", 8192, 2560, true},
      {DType::kQ4_K, "q4_K gate_up", 8192, 2560, true},
      {DType::kQ4_K, "q4_K down", 2560, 9216, true},
      {DType::kQ6_K, "q6_K down", 2560, 9216, true},
      {DType::kQ6_K, "q6_K grid=7760 (operator lm_head-class)", 31040, 4096, false},
      {DType::kQ6_K, "q6_K lm_head-class N=151936 (contract-named)", 151936, 4096, false},
      {DType::kQ6_K, "q6_K lm_head REAL N=248320", 248320, 2560, false},
      // Exact tuples observed from the ENGINE dispatch trace (bf16 x bf16):
      {DType::kQ4_K, "ENGINE q4_K n=18432 k=2560", 18432, 2560, false},
      {DType::kQ4_K, "ENGINE q4_K n=1024 k=2560", 1024, 2560, true},
      {DType::kQ4_K, "ENGINE q4_K n=2560 k=4096", 2560, 4096, true},
      {DType::kQ4_K, "ENGINE q4_K n=8192 k=2560", 8192, 2560, true},
  };

  for (const ShapeCase& sc : shapes) {
    const WeightCase* c = nullptr;
    for (const WeightCase& wc : kKQuantCases)
      if (wc.dtype == sc.wt) c = &wc;
    const int64_t nsb = sc.k / c->block_elems;
    CHECK(sc.k % c->block_elems == 0);
    CAPTURE(std::string(sc.name));
    CAPTURE(sc.n);
    CAPTURE(sc.k);

    std::vector<uint8_t> wq = RandomBlocks(*c, sc.n * nsb, 0x5EEDU);
    const size_t wbytes = wq.size();
    void* d_w = gpu.Alloc(wbytes);
    gpu.Copy(gq, d_w, wq.data(), wbytes);

    std::vector<DType> adts{DType::kBF16};
    if (sc.all_act_dtypes) adts = {DType::kF32, DType::kBF16, DType::kF16};
    for (DType adt : adts) {
      CAPTURE(adt);
      const size_t aesz = adt == DType::kF32 ? 4 : 2;
      // One fixed activation row, magnitudes the engine actually sees.
      std::vector<float> af(static_cast<size_t>(sc.k));
      GenerateData(3.0F, af.size(), af.data());
      std::vector<uint8_t> abuf(af.size() * aesz);
      for (size_t i = 0; i < af.size(); ++i) {
        if (adt == DType::kF32)
          std::memcpy(abuf.data() + 4 * i, &af[i], 4);
        else if (adt == DType::kBF16) {
          const uint16_t h = vt::F32ToBF16(af[i]);
          std::memcpy(abuf.data() + 2 * i, &h, 2);
        } else {
          const uint16_t h = vt::F32ToF16(af[i]);
          std::memcpy(abuf.data() + 2 * i, &h, 2);
        }
      }
      void* d_a = gpu.Alloc(abuf.size());
      gpu.Copy(gq, d_a, abuf.data(), abuf.size());

      // Run BOTH arms at the SAME output dtype (bf16, engine-realistic)
      // and compare RAW output bytes.
      constexpr size_t kOesz = 2;  // bf16
      std::vector<std::vector<unsigned char>> outs(2);
      for (int arm = 0; arm < 2; ++arm) {
        void* d_o = gpu.Alloc(kOesz * static_cast<size_t>(sc.n));
        {
          EnvGuard guard(arm == 1);
          Tensor at = DevTensor(d_a, adt, {1, sc.k});
          Tensor bt = DevTensor(d_w, c->dtype, {sc.n, sc.k});
          Tensor ot = DevTensor(d_o, DType::kBF16, {1, sc.n});
          vt::MatmulBTQuant(gq, ot, at, bt);
          outs[arm].resize(kOesz * static_cast<size_t>(sc.n));
          gpu.Copy(gq, outs[arm].data(), d_o, outs[arm].size());
          gpu.Synchronize(gq);
        }
        gpu.Free(d_o);
      }
      // Byte identity: locate and report the FIRST divergence for triage.
      size_t first_bad = outs[0].size();
      for (size_t i = 0; i < outs[0].size(); ++i)
        if (outs[0][i] != outs[1][i]) { first_bad = i; break; }
      CAPTURE(first_bad);
      CHECK(outs[0] == outs[1]);
      gpu.Free(d_a);
    }
    gpu.Free(d_w);
  }
  gpu.DestroyQueue(gq);
}

TEST_CASE("ROCm K-quant DEFAULT arm (env unset) stays within 1e-6 NMSE vs CPU") {
  if (!vt::rocm::DeviceAvailable()) {
    MESSAGE("no AMD GPU on this host; ROCm keep-quant gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};

  // Default-OFF inertness probe: with no VT_GEMV_MMVQ in the environment the
  // baseline warp-reduction kernel must be untouched by the T4a change. The
  // baseline's shfl tree reassociates the float sum, so this holds it to the
  // SAME 1e-6 NMSE-vs-CPU band as the CUDA sibling gate — not bit-exactness.
  const WeightCase& c = kKQuantCases[0];  // q4_K
  const int64_t nsb = 10, k = nsb * c.block_elems, n = 7;
  std::vector<uint8_t> wq = RandomBlocks(c, n * nsb, 0x5EEDU);
  std::vector<float> a(static_cast<size_t>(k));
  GenerateData(1.0F, a.size(), a.data());

  std::vector<float> cpu_out(static_cast<size_t>(n), 0.0F);
  {
    Tensor at = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {1, k});
    Tensor bt = Tensor::Contiguous(wq.data(), DType::kF32, Cpu(), {n, k});
    bt.dtype = c.dtype;
    Tensor ot = Tensor::Contiguous(cpu_out.data(), DType::kF32, Cpu(), {1, n});
    vt::MatmulBTQuant(cq, ot, at, bt);
  }

  void* d_a = gpu.Alloc(a.size() * sizeof(float));
  void* d_w = gpu.Alloc(wq.size());
  void* d_o = gpu.Alloc(sizeof(float) * static_cast<size_t>(n));
  gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
  gpu.Copy(gq, d_w, wq.data(), wq.size());
  std::vector<float> rocm_out(static_cast<size_t>(n), 0.0F);
  {
    EnvGuard off(false);  // explicitly "0": the arm must NOT engage
    Tensor at = DevTensor(d_a, DType::kF32, {1, k});
    Tensor bt = DevTensor(d_w, c.dtype, {n, k});
    Tensor ot = DevTensor(d_o, DType::kF32, {1, n});
    vt::MatmulBTQuant(gq, ot, at, bt);
    gpu.Copy(gq, rocm_out.data(), d_o, rocm_out.size() * sizeof(float));
    gpu.Synchronize(gq);
  }
  gpu.Free(d_a);
  gpu.Free(d_w);
  gpu.Free(d_o);

  const double nmse = Nmse(rocm_out, cpu_out);
  CAPTURE(nmse);
  CHECK(nmse <= kMaxNmseVsCpu);
  gpu.DestroyQueue(gq);
}

TEST_CASE("Fused-prologue Q8_K quantization is BYTE-IDENTICAL to the standalone quantizer") {
  if (!vt::rocm::DeviceAvailable()) {
    MESSAGE("no AMD GPU on this host; ROCm keep-quant gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  // nsb=10 covers this model's decode K; inputs: pseudo-random rows plus an
  // ADVERSARIAL tied-amax row (+max first, equal-magnitude negative later, so
  // the amax FIRST-occurrence tie-break is what decides mx's sign) and an
  // all-zero row.
  const int64_t k = 10 * 256;
  std::mt19937 rng(0xB00B5U);
  std::vector<std::vector<float>> rows;
  for (int r = 0; r < 4; ++r) {
    std::vector<float> a(static_cast<size_t>(k));
    for (float& v : a) v = static_cast<float>(static_cast<int>(rng() % 2001) - 1000) / 500.0F;
    rows.push_back(std::move(a));
  }
  {
    std::vector<float> a(static_cast<size_t>(k), 0.0F);
    a[0] = 3.5F;
    a[17] = -3.5F;  // exact fabs tie; FIRST occurrence (index 0) must win
    a[291] = -3.5F; // another tie, still after index 0
    rows.push_back(std::move(a));
  }
  rows.push_back(std::vector<float>(static_cast<size_t>(k), 0.0F));

  for (size_t r = 0; r < rows.size(); ++r) {
    CAPTURE(r);
    const std::vector<float>& a = rows[r];
    void* d_a = gpu.Alloc(a.size() * sizeof(float));
    void* d_sa = gpu.Alloc(10 * 292);   // sizeof(BlockQ8_K), pinned by static_assert
    void* d_sb = gpu.Alloc(10 * 292);
    gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
    Tensor at = DevTensor(d_a, DType::kF32, {1, k});
    vt::rocm::MmvqQuantScratchForTesting(gq, d_sa, at, false);
    vt::rocm::MmvqQuantScratchForTesting(gq, d_sb, at, true);
    std::vector<unsigned char> sa(10 * 292), sb(10 * 292);
    gpu.Copy(gq, sa.data(), d_sa, sa.size());
    gpu.Copy(gq, sb.data(), d_sb, sb.size());
    gpu.Synchronize(gq);
    gpu.Free(d_a); gpu.Free(d_sa); gpu.Free(d_sb);
    CHECK(std::memcmp(sa.data(), sb.data(), sa.size()) == 0);
  }
  gpu.DestroyQueue(gq);
}

TEST_CASE("T4a repair: per-grid OFF-vs-ON timing at the operator's captured grids") {
  if (!vt::rocm::DeviceAvailable()) {
    MESSAGE("no AMD GPU on this host; ROCm keep-quant gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  // Evidence-only case (no correctness assert): median us/call per arm at
  // the grids the operator's rocprofv3 parse captured (grid = ceil(N/4)),
  // plus the real lm_head shape. bf16 act/out, engine-realistic.
  struct BenchShape { DType wt; const char* name; int64_t n, k; int reps; };
  const std::vector<BenchShape> shapes = {
      {DType::kQ6_K, "grid=80  Li2 (320x2560)", 320, 2560, 30},
      {DType::kQ4_K, "grid=80  Li0 (320x2560)", 320, 2560, 30},
      {DType::kQ4_K, "grid=576 Li0 (2304x2560)", 2304, 2560, 30},
      {DType::kQ6_K, "grid=7760 Li2 (31040x4096)", 31040, 4096, 12},
      {DType::kQ6_K, "lm_head real (248320x2560)", 248320, 2560, 8},
  };
  for (const BenchShape& sc : shapes) {
    const WeightCase* c = nullptr;
    for (const WeightCase& wc : kKQuantCases)
      if (wc.dtype == sc.wt) c = &wc;
    const int64_t nsb = sc.k / c->block_elems;
    std::vector<uint8_t> wq = RandomBlocks(*c, sc.n * nsb, 0x5EEDU);
    std::vector<float> af(static_cast<size_t>(sc.k));
    GenerateData(3.0F, af.size(), af.data());
    std::vector<uint16_t> abuf(af.size());
    for (size_t i = 0; i < af.size(); ++i)
      abuf[i] = vt::F32ToBF16(af[i]);
    void* d_w = gpu.Alloc(wq.size());
    void* d_a = gpu.Alloc(abuf.size() * 2);
    void* d_o = gpu.Alloc(2 * static_cast<size_t>(sc.n));
    gpu.Copy(gq, d_w, wq.data(), wq.size());
    gpu.Copy(gq, d_a, abuf.data(), abuf.size() * 2);
    MESSAGE(sc.name);
    double med[2] = {0, 0};
    for (int arm = 0; arm < 2; ++arm) {
      EnvGuard guard(arm == 1);
      Tensor at = DevTensor(d_a, DType::kBF16, {1, sc.k});
      Tensor bt = DevTensor(d_w, c->dtype, {sc.n, sc.k});
      Tensor ot = DevTensor(d_o, DType::kBF16, {1, sc.n});
      for (int w = 0; w < 3; ++w) {  // warmup
        vt::MatmulBTQuant(gq, ot, at, bt);
        gpu.Synchronize(gq);
      }
      std::vector<double> t;
      for (int r = 0; r < sc.reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        vt::MatmulBTQuant(gq, ot, at, bt);
        gpu.Synchronize(gq);
        const auto t1 = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
      }
      std::sort(t.begin(), t.end());
      med[arm] = t[t.size() / 2];
    }
    char buf[128];
    std::snprintf(buf, sizeof(buf),
                  "  OFF %9.1f us/call | ON %9.1f us/call | ratio ON/OFF %.2fx",
                  med[0], med[1], med[1] / med[0]);
    MESSAGE(buf);
    gpu.Free(d_w);
    gpu.Free(d_a);
    gpu.Free(d_o);
  }
  gpu.DestroyQueue(gq);
}

// ---------------------------------------------------------------------------
// T4a REPAIR ROUND 2 (reviewer findings F1/F2). The round-1 gate could not
// witness ROUTING: EnvGuard(false) writes "0" (never a true unset), and since
// ON==OFF are bit-equal by design, every output comparison is blind to which
// dispatch branch ran. These two cases pin routing itself via the host-side
// dispatch counters.

// F1: with VT_GEMV_MMVQ TRULY ABSENT (unsetenv, not "0") the call must take
// the BASELINE branch; with VT_GEMV_MMVQ=1 it must NOT. Catches an inverted
// getenv default (mutation M3) that outputs cannot see.
TEST_CASE("T4a repair-2 F1: ROUTING WITNESS -- env truly unset routes to BASELINE; ON routes to the GEMV arm") {
  if (!vt::rocm::DeviceAvailable()) {
    MESSAGE("no AMD GPU on this host; ROCm keep-quant gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  const WeightCase& c = kKQuantCases[0];  // q4_K
  const int64_t nsb = 10, k = nsb * c.block_elems, n = 7;
  std::vector<uint8_t> wq = RandomBlocks(c, n * nsb, 0x5EEDU);
  std::vector<float> a(static_cast<size_t>(k));
  GenerateData(1.5F, a.size(), a.data());

  void* d_a = gpu.Alloc(a.size() * sizeof(float));
  void* d_w = gpu.Alloc(wq.size());
  void* d_o = gpu.Alloc(sizeof(float) * static_cast<size_t>(n));
  gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
  gpu.Copy(gq, d_w, wq.data(), wq.size());

  auto run_once = [&] {
    Tensor at = DevTensor(d_a, DType::kF32, {1, k});
    Tensor bt = DevTensor(d_w, c.dtype, {n, k});
    Tensor ot = DevTensor(d_o, DType::kF32, {1, n});
    vt::MatmulBTQuant(gq, ot, at, bt);
    gpu.Synchronize(gq);
  };

  // TRUE unset: the flag string must be absent from the environment -- NOT
  // EnvGuard(false), which sets "0". Default-OFF inertness means the
  // BASELINE counter advances and no GEMV counter moves.
  ::unsetenv("VT_GEMV_MMVQ");
  vt::rocm::MmvqResetRouteCountsForTesting();
  run_once();
  const auto off_counts = vt::rocm::MmvqRouteCountsForTesting();
  CHECK(off_counts.baseline == 1);
  CHECK(off_counts.gemv_mmvq == 0);
  CHECK(off_counts.gemv_fused == 0);

  // Paired ON case: exactly the reverse. n=7 <= kMmvqFoldMaxRows, so the
  // arm engages via its FUSED sub-branch; either way the baseline counter
  // must not move.
  {
    EnvGuard on(true);
    vt::rocm::MmvqResetRouteCountsForTesting();
    run_once();
    const auto on_counts = vt::rocm::MmvqRouteCountsForTesting();
    CHECK(on_counts.baseline == 0);
    CHECK(on_counts.gemv_fused == 1);
    CHECK(on_counts.gemv_mmvq == 0);
  }
  ::unsetenv("VT_GEMV_MMVQ");
  gpu.Free(d_a);
  gpu.Free(d_w);
  gpu.Free(d_o);
  gpu.DestroyQueue(gq);
}

// F2: fold-crossover WITNESS. With the arm ON, n=256 (<= kMmvqFoldMaxRows)
// must dispatch through the FUSED sub-branch and n=2304 (> 512, within the
// reviewer's mutated range (512,4096]) must dispatch through the NON-FUSED
// GEMV branch. Catches a kMmvqFoldMaxRows drift (mutation M4: 512 -> 4096)
// that flips measured per-call ratios while staying output-green.
TEST_CASE("T4a repair-2 F2: FOLD-CROSSOVER WITNESS -- fused sub-branch only at n <= kMmvqFoldMaxRows") {
  if (!vt::rocm::DeviceAvailable()) {
    MESSAGE("no AMD GPU on this host; ROCm keep-quant gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  const WeightCase& c = kKQuantCases[0];  // q4_K
  const int64_t nsb = 10, k = nsb * c.block_elems;

  struct FoldShape { const char* name; int64_t n; long long want_fused, want_gemv, want_baseline; };
  const FoldShape shapes[] = {
      {"n=256  (fold expected)", 256, 1, 0, 0},
      {"n=2304 (fold NOT expected)", 2304, 0, 1, 0},
  };
  for (const FoldShape& sc : shapes) {
    CAPTURE(sc.name);
    std::vector<uint8_t> wq = RandomBlocks(c, sc.n * nsb, 0x5EEDU);
    std::vector<float> a(static_cast<size_t>(k));
    GenerateData(2.5F, a.size(), a.data());
    void* d_a = gpu.Alloc(a.size() * sizeof(float));
    void* d_w = gpu.Alloc(wq.size());
    void* d_o = gpu.Alloc(sizeof(float) * static_cast<size_t>(sc.n));
    gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
    gpu.Copy(gq, d_w, wq.data(), wq.size());
    {
      EnvGuard on(true);
      vt::rocm::MmvqResetRouteCountsForTesting();
      Tensor at = DevTensor(d_a, DType::kF32, {1, k});
      Tensor bt = DevTensor(d_w, c.dtype, {sc.n, k});
      Tensor ot = DevTensor(d_o, DType::kF32, {1, sc.n});
      vt::MatmulBTQuant(gq, ot, at, bt);
      gpu.Synchronize(gq);
      const auto counts = vt::rocm::MmvqRouteCountsForTesting();
      CHECK(counts.gemv_fused == sc.want_fused);
      CHECK(counts.gemv_mmvq == sc.want_gemv);
      CHECK(counts.baseline == sc.want_baseline);
    }
    ::unsetenv("VT_GEMV_MMVQ");
    gpu.Free(d_a);
    gpu.Free(d_w);
    gpu.Free(d_o);
  }
  gpu.DestroyQueue(gq);
}
