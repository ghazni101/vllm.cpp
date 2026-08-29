// T36 (GFX1100-TG200) prefill M-tiled K-quant GEMM gate.
//
// The campaign's standing quant gate (tests/vt/test_rocm_quant_dot.cpp) stays
// UNCHANGED per the campaign constraint; this file adds the op-level gate for
// the VT_PREFILL_TILE arm (KQuantGemmMTiledK in
// src/vt/rocm/rocm_grouped_gemm.hip), which that file cannot express: its
// ON/OFF arms share one env lever, so with VT_PREFILL_TILE=1 both arms route
// to the tiled kernel and the memcmp is a tautology.
//
// Contract identical to the T4a gate: the tiled arm is BIT-IDENTICAL to the
// warp-per-(i,j) baseline at every (m, n, nsb, dtype) -- asserted here by
// raw-byte memcmp of OFF vs ON outputs at PREFILL shapes (m > 1, the regime
// the arm owns) -- and the ON arm stays within the 1e-6 NMSE band vs the CPU
// oracle (the same band the default arm is held to).
//
// Skips cleanly when the box has no AMD GPU (CPU CI leg stays green).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Device GpuDev() { return Device{DeviceType::kROCM, 0}; }

constexpr double kMaxNmseVsCpu = 1e-6;

struct WeightCase {
  DType dtype;
  int64_t block_elems;
  int64_t block_bytes;
  int d_off;
  int dmin_off;
  const char* name;
};

// Same three K-quants the ROCm provider serves natively (offsets restated
// from ggml-common.h, mirroring test_rocm_quant_dot.cpp's table).
const WeightCase kCases[] = {
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
    if (c.d_off >= 0) put_f16(c.d_off, 0.0125F * jitter);
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

// Scoped VT_PREFILL_TILE writer: "0" forces the baseline kernel, "1" the
// tiled arm (the dispatch reads the flag PER CALL, so no re-init is needed).
struct TileGuard {
  explicit TileGuard(bool on) { ::setenv("VT_PREFILL_TILE", on ? "1" : "0", 1); }
  ~TileGuard() { ::unsetenv("VT_PREFILL_TILE"); }
};

}  // namespace

TEST_CASE("T36 prefill M-tiled arm is BYTE-EXACT vs the baseline and within the oracle NMSE band") {
  if (!vt::rocm::DeviceAvailable()) {
    MESSAGE("no AMD GPU on this host; ROCm keep-quant gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};

  // m sweep covers a partial tile (3 % 8), exact tiles (8), engine-chunk-like
  // (39) and a long-prompt chunk (512); n covers tiny, prime, and engine-like
  // widths. Every (dtype, nsb) crosses the lane-strided superblock map.
  for (const WeightCase& c : kCases) {
    for (int64_t nsb : {int64_t{1}, int64_t{3}, int64_t{10}}) {
      const int64_t k = nsb * c.block_elems;
      for (int64_t m : {int64_t{2}, int64_t{3}, int64_t{8}, int64_t{39}, int64_t{512}}) {
        for (int64_t n : {int64_t{1}, int64_t{7}, int64_t{129}, int64_t{257}}) {
          for (uint32_t seed : {0x5EEDU, 0xA11CEU}) {
            CAPTURE(c.name);
            CAPTURE(nsb);
            CAPTURE(m);
            CAPTURE(n);
            CAPTURE(seed);

            const int64_t nwb = n * nsb;
            std::vector<uint8_t> wq = RandomBlocks(c, nwb, seed);
            const size_t asz = static_cast<size_t>(m) * static_cast<size_t>(k);
            std::vector<float> af(asz);
            GenerateData(static_cast<float>(seed), asz, af.data());
            std::vector<uint16_t> abf(asz);
            for (size_t i2 = 0; i2 < af.size(); ++i2)
              abf[i2] = vt::F32ToBF16(af[i2]);

            // --- CPU oracle (host tensors; the kMatmulBTQuant CIQ GEMM) ----
            std::vector<float> cpu_out(static_cast<size_t>(m) *
                                           static_cast<size_t>(n),
                                       0.0F);
            {
              Tensor at = Tensor::Contiguous(abf.data(), DType::kBF16, Cpu(), {m, k});
              Tensor bt = Tensor::Contiguous(wq.data(), DType::kF32, Cpu(), {n, k});
              bt.dtype = c.dtype;
              Tensor ot = Tensor::Contiguous(cpu_out.data(), DType::kF32, Cpu(), {m, n});
              vt::MatmulBTQuant(cq, ot, at, bt);
            }

            // --- ROCm: baseline (TILE=0) vs tiled (TILE=1) at this shape ---
            const size_t oesz = 4;  // f32 out keeps the oracle diff simple
            const size_t oesz_tot = oesz * static_cast<size_t>(m) *
                                    static_cast<size_t>(n);
            void* d_w = gpu.Alloc(wq.size());
            gpu.Copy(gq, d_w, wq.data(), wq.size());
            void* d_ab = gpu.Alloc(asz * 2);
            gpu.Copy(gq, d_ab, abf.data(), asz * 2);
            std::vector<std::vector<unsigned char>> arm_raw(2);
            for (int arm = 0; arm < 2; ++arm) {
              void* d_o = gpu.Alloc(oesz_tot);
              {
                TileGuard on(arm == 1);
                Tensor at = DevTensor(d_ab, DType::kBF16, {m, k});
                Tensor bt = DevTensor(d_w, c.dtype, {n, k});
                Tensor ot = DevTensor(d_o, DType::kF32, {m, n});
                vt::MatmulBTQuant(gq, ot, at, bt);
                arm_raw[arm].resize(oesz_tot);
                gpu.Copy(gq, arm_raw[arm].data(), d_o, arm_raw[arm].size());
                gpu.Synchronize(gq);
              }
              gpu.Free(d_o);
            }
            gpu.Free(d_ab);
            gpu.Free(d_w);

            // Tiled arm must be BYTE-IDENTICAL to the baseline kernel.
            CHECK(std::memcmp(arm_raw[0].data(), arm_raw[1].data(),
                              arm_raw[0].size()) == 0);
            // And within the oracle NMSE band.
            std::vector<float> on_out(arm_raw[1].size() / 4);
            for (size_t i2 = 0; i2 < on_out.size(); ++i2)
              on_out[i2] = reinterpret_cast<const float*>(arm_raw[1].data())[i2];
            const double nmse_on = Nmse(on_out, cpu_out);
            CAPTURE(nmse_on);
            CHECK(nmse_on <= kMaxNmseVsCpu);
          }
        }
      }
    }
  }
  gpu.DestroyQueue(gq);
}
