// tg200-neartie — the GFX1100-TG200 teacher-forced logprob-band adjudicator.
//
// The campaign's correctness policy (.agents/specs/gfx1100-tg200.md ## Correctness
// policy) owes "near-tie adjudication with teacher-forced logprob gaps per the
// ratified band doctrine" (.agents/specs/rocm-m4-oracle.md) for every lever that
// changes floating-point reduction order — the ceremony whose absence closed T35
// red (docs/bench-evidence/gfx1100-tg200-t35-gdn-ba-merge-20260829.md). This tool
// IS that ceremony, wired through the public C ABI exactly like vllm-cli
// (include/vllm.h only, links vllm::shared): it needs no engine change because
// ABI v8's vllm_logits_processor already hands a host callback a MUTABLE view of
// the request's logits row once per decode step, before sampling.
//
// Band doctrine (rocm-m4-oracle.md): a token divergence is a NEAR-TIE when the
// teacher-forced logprob gap at that position is <= 500 milli-nats. At decode
// step n the engine, conditioned on prompt + ref[0..n), scores ref[n]:
//
//     gap_nats(n) = max_j logits[j] - logits[ref[n]]   (>= 0; the softmax
//     normalizer cancels, so this is logprob(argmax) - logprob(ref) exactly)
//
// which is the m4 oracle lane's neartie gap. ref[n] is then FORCED by masking
// every other entry to -inf (the row's ref entry keeps its computed value), so
// the greedy walk follows the reference body regardless of the arm's own
// preferences — teacher forcing, per position, in one pass.
//
// Modes:
//   capture    one untouched greedy gate run (body text + md5, the
//              `vllm-cli ... > body.txt` convention), then the same run under
//              an OBSERVER processor that records argmax of each pre-sampling
//              logits row — the ids the greedy sampler emits (greedy token k
//              is prefix-determined). The engine's own processor token view is
//              empty on this engine's async path, so the ids are read from the
//              logits, where nothing can lag.
//   adjudicate teacher-force the reference ids under THIS build (the ARM):
//              per-position gaps, divergent/over-band counts, PASS/FAIL against
//              the band, JSON + human summary. Integrity is the forced walk's
//              body md5 (--expect-md5), proven end-to-end.
//
// Exit codes: 0 PASS · 1 FAIL (a gap exceeds the band) · 2 usage · 3 runtime ·
// 4 integrity (prefix violation / callback mismatch / capture mismatch — a
// broken ceremony is never a band verdict).
//
// Reference config: the campaign's adopted-lever env (T33/T34 block) and the
// gate workload (tools/tg200-prompt.txt, 256 tokens, greedy) go through
// tools/tg200-neartie.sh, which owns the gpu-ctl lock and the container.
#include "vllm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// ── MD5 (RFC 1321; the compact public-domain shape) ──────────────────────────
// The campaign names reference bodies by `md5sum` of the captured stdout bytes
// (vllm-cli prints the body text followed by one '\n'). body_md5 below is
// md5(text || "\n"), byte-identical to `./vllm-cli ... > body.txt; md5sum body.txt`.
struct Md5 {
  uint32_t a = 0x67452301, b = 0xefcdab89, c = 0x98badcfe, d = 0x10325476;
  uint64_t len = 0;
  uint8_t buf[64] = {0};
  size_t fill = 0;

  void Blocks(const uint8_t* p, size_t n) {
    static const uint32_t K[64] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a,
        0xa8304613, 0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
        0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340,
        0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8,
        0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
        0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92,
        0xffeff47d, 0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
        0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};
    static const int R[64] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
                              5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
                              4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
                              6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};
    while (n >= 64) {
      uint32_t m[16];
      std::memcpy(m, p, 64);
      uint32_t A = a, B = b, C = c, D = d;
      for (int i = 0; i < 64; ++i) {
        uint32_t f;
        int g;
        if (i < 16) {
          f = (B & C) | (~B & D);
          g = i;
        } else if (i < 32) {
          f = (D & B) | (~D & C);
          g = (5 * i + 1) % 16;
        } else if (i < 48) {
          f = B ^ C ^ D;
          g = (3 * i + 5) % 16;
        } else {
          f = C ^ (B | ~D);
          g = (7 * i) % 16;
        }
        const uint32_t sum = A + f + K[i] + m[g];
        const uint32_t rotated = (sum << (R[i] % 32)) | (sum >> ((32 - R[i]) % 32));
        const uint32_t tmp = D;
        D = C;
        C = B;
        B = B + rotated;
        A = tmp;
      }
      a += A;
      b += B;
      c += C;
      d += D;
      p += 64;
      n -= 64;
    }
  }

  void Update(const uint8_t* p, size_t n) {
    len += n;
    if (fill > 0) {
      const size_t take = std::min(n, 64 - fill);
      std::memcpy(buf + fill, p, take);
      fill += take;
      p += take;
      n -= take;
      if (fill == 64) {
        Blocks(buf, 64);
        fill = 0;
      }
    }
    if (n >= 64) {
      const size_t whole = n & ~size_t{63};
      Blocks(p, whole);
      p += whole;
      n -= whole;
    }
    if (n > 0) {
      std::memcpy(buf + fill, p, n);
      fill = n;
    }
  }

  std::string Digest() const {
    static const char* hex = "0123456789abcdef";
    const uint64_t bits = len * 8;
    const size_t pad = (fill < 56) ? 56 - fill : 120 - fill;
    uint8_t block[128];
    std::memcpy(block, buf, fill);
    block[fill] = 0x80;
    std::memset(block + fill + 1, 0, pad - 1);
    for (int i = 0; i < 8; ++i)
      block[fill + pad + i] = static_cast<uint8_t>(bits >> (8 * i));
    Md5 t = *this;
    t.Blocks(block, fill + pad + 8);
    const uint32_t out[4] = {t.a, t.b, t.c, t.d};
    std::string s;
    s.reserve(32);
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) {
        s.push_back(hex[(out[i] >> (8 * j + 4)) & 0xf]);
        s.push_back(hex[(out[i] >> (8 * j)) & 0xf]);
      }
    return s;
  }
};

std::string Md5Hex(const std::string& s) {
  Md5 m;
  m.Update(reinterpret_cast<const uint8_t*>(s.data()), s.size());
  return m.Digest();
}

// md5s are compared case-insensitively (operators type them lower-case).
bool Md5Matches(const std::string& have, const std::string& want) {
  if (have.size() != want.size()) return false;
  for (size_t i = 0; i < have.size(); ++i) {
    char h = have[i], w = want[i];
    if (h >= 'A' && h <= 'F') h = static_cast<char>(h - 'A' + 'a');
    if (w >= 'A' && w <= 'F') w = static_cast<char>(w - 'A' + 'a');
    if (h != w) return false;
  }
  return true;
}

// ── argv ─────────────────────────────────────────────────────────────────────
struct Args {
  std::string mode;                // "capture" | "adjudicate"
  std::string model;
  std::string prompt_file;
  std::string ref_ids;             // adjudicate: input ids
  std::string out_prefix;          // capture: PREFIX.{body.txt,ids.i32}
  std::string json_out;            // adjudicate: JSON report path
  std::string body_out;            // adjudicate: optional body text out
  std::string expect_md5;          // capture: hard-bind the body md5
  std::string note;                // echoed into the JSON report
  int max_tokens = 256;            // capture only (adjudicate uses the ref length)
  double band_mnats = 500.0;       // rocm-m4-oracle: the ratified near-tie band
  int top_k = 16;                  // human-table rows (sorted by gap desc)
};

void Usage(const char* argv0, std::FILE* out) {
  std::fprintf(
      out,
      "tg200-neartie — teacher-forced logprob-band adjudicator (GFX1100-TG200)\n\n"
      "  %s capture --model <gguf> --prompt-file <txt> --out <prefix>\n"
      "            [--max-tokens N] [--expect-md5 <md5>] [--note <s>]\n"
      "      Greedy gate run; writes <prefix>.body.txt, <prefix>.ids.i32 and\n"
      "      prints the body md5 (md5 of the body bytes + trailing newline,\n"
      "      the campaign's `md5sum body.txt` convention).\n\n"
      "  %s adjudicate --model <gguf> --prompt-file <txt> --ref-ids <ids.i32>\n"
      "               --json <out.json> [--band-mnats 500] [--top-k 16]\n"
      "               [--expect-md5 <md5>] [--body-out <txt>] [--note <s>]\n"
      "      Teacher-force the reference ids under THIS build; per-position\n"
      "      logprob gaps, counts, PASS/FAIL vs the band (rocm-m4-oracle).\n"
      "      --expect-md5 hard-binds the walk: the forced body must reproduce\n"
      "      it or the tool exits 4 instead of reporting a verdict.\n\n"
      "Exit codes: 0 PASS | 1 FAIL (gap over band) | 2 usage | 3 runtime |\n"
      "            4 integrity (prefix violation / callback mismatch).\n",
      argv0, argv0);
}

const char* NextArg(int argc, char** argv, int& i) {
  if (i + 1 >= argc) {
    std::fprintf(stderr, "tg200-neartie: missing value after %s\n", argv[i]);
    std::exit(2);
  }
  return argv[++i];
}

bool ParseArgs(int argc, char** argv, Args& a) {
  if (argc >= 2) a.mode = argv[1];
  if (a.mode != "capture" && a.mode != "adjudicate") return false;
  for (int i = 2; i < argc; ++i) {
    const std::string f = argv[i];
    if (f == "--model") a.model = NextArg(argc, argv, i);
    else if (f == "--prompt-file") a.prompt_file = NextArg(argc, argv, i);
    else if (f == "--ref-ids") a.ref_ids = NextArg(argc, argv, i);
    else if (f == "--out") a.out_prefix = NextArg(argc, argv, i);
    else if (f == "--json") a.json_out = NextArg(argc, argv, i);
    else if (f == "--body-out") a.body_out = NextArg(argc, argv, i);
    else if (f == "--expect-md5") a.expect_md5 = NextArg(argc, argv, i);
    else if (f == "--note") a.note = NextArg(argc, argv, i);
    else if (f == "--max-tokens") a.max_tokens = std::atoi(NextArg(argc, argv, i));
    else if (f == "--band-mnats") a.band_mnats = std::atof(NextArg(argc, argv, i));
    else if (f == "--top-k") a.top_k = std::atoi(NextArg(argc, argv, i));
    else { std::fprintf(stderr, "tg200-neartie: unknown argument %s\n", argv[i]); return false; }
  }
  if (a.model.empty() || a.prompt_file.empty()) {
    std::fprintf(stderr, "tg200-neartie: --model and --prompt-file are required\n");
    return false;
  }
  if (a.mode == "capture" && a.out_prefix.empty()) {
    std::fprintf(stderr, "tg200-neartie: capture needs --out <prefix>\n");
    return false;
  }
  if (a.mode == "adjudicate" && (a.ref_ids.empty() || a.json_out.empty())) {
    std::fprintf(stderr, "tg200-neartie: adjudicate needs --ref-ids and --json\n");
    return false;
  }
  return true;
}

// $(cat file) semantics: the gate passes `--prompt "$(cat tools/tg200-prompt.txt)"`,
// and command substitution strips trailing newlines. Byte-identical prompt text
// or the walk is not the gate's walk.
std::string ReadPromptStripped(const char* path) {
  std::FILE* f = std::fopen(path, "rb");
  if (f == nullptr) {
    std::fprintf(stderr, "tg200-neartie: cannot open prompt file %s\n", path);
    std::exit(3);
  }
  std::string s;
  char buf[4096];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
  std::fclose(f);
  while (!s.empty() && s.back() == '\n') s.pop_back();
  if (s.empty()) {
    std::fprintf(stderr, "tg200-neartie: prompt file %s is empty\n", path);
    std::exit(3);
  }
  return s;
}

std::vector<int32_t> ReadIds(const char* path) {
  std::FILE* f = std::fopen(path, "rb");
  if (f == nullptr) {
    std::fprintf(stderr, "tg200-neartie: cannot open ref ids %s\n", path);
    std::exit(3);
  }
  std::vector<int32_t> ids;
  int32_t v;
  while (std::fread(&v, sizeof(v), 1, f) == 1) ids.push_back(v);
  std::fclose(f);
  if (ids.empty()) {
    std::fprintf(stderr, "tg200-neartie: ref ids %s is empty\n", path);
    std::exit(3);
  }
  return ids;
}

void WriteBytes(const char* path, const void* p, size_t n) {
  std::FILE* f = std::fopen(path, "wb");
  if (f == nullptr) {
    std::fprintf(stderr, "tg200-neartie: cannot write %s\n", path);
    std::exit(3);
  }
  if (n > 0) std::fwrite(p, 1, n, f);
  std::fclose(f);
}

std::string JsonEscape(const std::string& s) {
  std::string o;
  o.reserve(s.size() + 8);
  for (char ch : s) {
    switch (ch) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20) {
          char b[8];
          std::snprintf(b, sizeof(b), "\\u%04x", static_cast<int>(ch) & 0xff);
          o += b;
        } else {
          o.push_back(ch);
        }
    }
  }
  return o;
}

// ── shared engine plumbing ───────────────────────────────────────────────────
vllm_engine* LoadEngine(const std::string& model) {
  vllm_model_params mp = vllm_model_params_default();
  mp.model_path = model.c_str();
  vllm_engine* engine = nullptr;
  const vllm_status st = vllm_engine_load(&mp, &engine);
  if (st != VLLM_OK || engine == nullptr) {
    std::fprintf(stderr, "tg200-neartie: model load failed (status %d): %s\n",
                 static_cast<int>(st), vllm_last_error());
    std::exit(3);
  }
  return engine;
}

// The gate's sampling shape, verbatim: --temperature 0 --seed 0 (examples/cli
// defaults top_p 1.0 / top_k 0; <= 0 temperature is greedy, so the seed is
// inert but recorded the same way the gate records it).
vllm_sampling_params GateSampling(int max_tokens) {
  vllm_sampling_params sp = vllm_sampling_params_default();
  sp.temperature = 0.0f;
  sp.top_p = 1.0f;
  sp.top_k = 0;
  sp.max_tokens = max_tokens;
  sp.has_seed = 1;
  sp.seed = 0;
  return sp;
}

// vllm-cli prints text then '\n'; the campaign's reference md5s are md5sums of
// exactly those redirected stdout bytes.
std::string BodyBytes(const vllm_completion& c) {
  std::string body = (c.text != nullptr) ? c.text : "";
  body.push_back('\n');
  return body;
}

}  // namespace

// ── capture ──────────────────────────────────────────────────────────────────
namespace {

// CAPTURE records the greedy ids from the raw logits row itself. The engine's
// own processor token view (token_ids/n_token_ids) is EMPTY every step on this
// engine's async path (observed 2026-08-29: five callbacks, every n_token_ids
// == 0 on the GGUF qwen3.5 rocm-dev:10.0.0 gate config), so the reference ids
// are read where they cannot lag: argmax of the pre-mutation logits row IS the
// token the greedy sampler is about to emit. Greedy token k depends only on
// the prefix, so recording argmax at every step yields exactly the generated
// id sequence. Exact-float ties (the m4 lane's 0.0000-nat France/Italy class)
// are the one caveat: the adjudicate self-test binds the captured ids to the
// campaign body md5 end-to-end, so a tie-break divergence cannot pass silently.
struct CaptureState {
  std::vector<int32_t> ids;  // per-step argmax of the raw logits row
  int calls = 0;
};

void CaptureCb(const int32_t* /*token_ids*/, int32_t n_token_ids, float* logits,
               int32_t vocab_size, void* user_data) {
  auto* st = static_cast<CaptureState*>(user_data);
  ++st->calls;
  static const bool debug = [] {
    const char* e = std::getenv("TG200_NEARTIE_DEBUG");
    return e != nullptr && e[0] == '1';
  }();
  if (debug) {
    std::fprintf(stderr, "tg200-neartie: capture cb #%d n=%d\n", st->calls,
                 static_cast<int>(n_token_ids));
  }
  int32_t top = 0;
  float lmax = logits[0];
  for (int32_t j = 1; j < vocab_size; ++j) {
    if (logits[j] > lmax) {
      lmax = logits[j];
      top = j;
    }
  }
  st->ids.push_back(top);
}
int Capture(const Args& a) {
  vllm_engine* engine = LoadEngine(a.model);
  const std::string prompt = ReadPromptStripped(a.prompt_file.c_str());

  // Run A: the gate workload verbatim, NO processor registered — the body
  // bytes and their md5 come from an untouched run so the reference stays
  // byte-bound to the campaign's `vllm-cli ... > body.txt` convention.
  vllm_completion ra{};
  const vllm_sampling_params sp_a = GateSampling(a.max_tokens);
  vllm_status st = vllm_complete(engine, prompt.c_str(), &sp_a, &ra);
  if (st != VLLM_OK) {
    std::fprintf(stderr, "tg200-neartie: capture run A failed (status %d): %s\n",
                 static_cast<int>(st), vllm_last_error());
    return 3;
  }
  const std::string ta = BodyBytes(ra);
  const std::string body_md5 = Md5Hex(ta);
  vllm_completion_free(&ra);
  if (!a.expect_md5.empty() && !Md5Matches(body_md5, a.expect_md5)) {
    std::fprintf(stderr,
                 "tg200-neartie: capture integrity: body md5 %s does not match "
                 "the expected reference %s — refusing to mint reference ids "
                 "from a foreign body\n",
                 body_md5.c_str(), a.expect_md5.c_str());
    vllm_engine_free(engine);
    return 4;
  }

  // Run B: same gate run under an OBSERVER processor. The callback mutates
  // nothing; it records argmax of each pre-sampling logits row — exactly the
  // ids the greedy sampler emits (greedy token k is prefix-determined).
  CaptureState cs;
  vllm_sampling_params sp = GateSampling(a.max_tokens);
  sp.logits_processor = &CaptureCb;
  sp.logits_processor_user_data = &cs;
  vllm_completion rb{};
  st = vllm_complete(engine, prompt.c_str(), &sp, &rb);
  if (st != VLLM_OK) {
    std::fprintf(stderr, "tg200-neartie: capture run B failed (status %d): %s\n",
                 static_cast<int>(st), vllm_last_error());
    vllm_engine_free(engine);
    return 3;
  }
  const std::string tb = BodyBytes(rb);
  vllm_completion_free(&rb);
  vllm_engine_free(engine);
  if (static_cast<int>(cs.ids.size()) != a.max_tokens || cs.calls != a.max_tokens) {
    std::fprintf(stderr,
                 "tg200-neartie: capture integrity: observer recorded %d ids "
                 "over %d calls, expected %d — the sampler never handed the "
                 "processor one row per decode step\n",
                 static_cast<int>(cs.ids.size()), cs.calls, a.max_tokens);
    return 4;
  }
  // The observer run must be numerics-neutral: same body bytes as run A.
  if (tb != ta) {
    std::fprintf(stderr,
                 "tg200-neartie: capture integrity: observer run body differs "
                 "from the gate run body (md5 %s vs %s) — registering the "
                 "processor changed the run, which must never happen\n",
                 Md5Hex(tb).c_str(), body_md5.c_str());
    return 4;
  }
  const std::string body_path = a.out_prefix + ".body.txt";
  WriteBytes(body_path.c_str(), ta.data(), ta.size());
  const std::string ids_path = a.out_prefix + ".ids.i32";
  WriteBytes(ids_path.c_str(), cs.ids.data(), cs.ids.size() * sizeof(int32_t));
  std::printf("tg200-neartie: capture body_md5=%s\n", body_md5.c_str());
  std::printf("tg200-neartie: capture body=%s\n", body_path.c_str());
  std::printf("tg200-neartie: capture ids=%s (%d tokens)\n", ids_path.c_str(),
              static_cast<int>(cs.ids.size()));
  return 0;
}

}  // namespace

// ── adjudicate ───────────────────────────────────────────────────────────────
namespace {

struct GapRow {
  int n = 0;
  int32_t ref = 0;
  int32_t argmax = 0;
  double gap_mnats = 0.0;
  double ref_logprob_mnats = 0.0;
  bool divergent = false;
};

struct AdjudicateState {
  const int32_t* ref = nullptr;
  int n_ref = 0;
  int calls = 0;
  std::vector<GapRow> rows;
};

// NOTE: the engine's per-step token view (token_ids/n_token_ids) is empty on
// this engine's async path (see the capture comment above), so prefix
// integrity is NOT taken from the engine's word — it is proven end-to-end by
// the forced walk's completion: forcing every step pins the sequence to
// ref[0..n), and the run's body md5 must equal the reference body md5 the
// caller passes (--expect-md5). A walk that slipped a position cannot
// reproduce that md5 and exits 4 instead of reporting a band verdict.
void AdjudicateCb(const int32_t* /*token_ids*/, int32_t /*n_token_ids*/,
                  float* logits, int32_t vocab_size, void* user_data) {
  auto* st = static_cast<AdjudicateState*>(user_data);
  const int n = st->calls;
  if (n >= st->n_ref) return;  // extra steps must not exist; checked after run.
  const int32_t ref_tok = st->ref[n];

  const float ref_logit = logits[ref_tok];
  // One pass for the argmax, one for the log-sum-exp around it.
  int32_t top = 0;
  float lmax = logits[0];
  for (int32_t j = 1; j < vocab_size; ++j) {
    if (logits[j] > lmax) {
      lmax = logits[j];
      top = j;
    }
  }
  double lse = 0.0;
  for (int32_t j = 0; j < vocab_size; ++j) {
    lse += std::exp(static_cast<double>(logits[j]) - static_cast<double>(lmax));
  }
  lse = static_cast<double>(lmax) + std::log(lse);

  GapRow& row = st->rows[n];
  row.n = n;
  row.ref = ref_tok;
  row.argmax = top;
  row.gap_mnats = (static_cast<double>(lmax) - static_cast<double>(ref_logit)) * 1000.0;
  row.ref_logprob_mnats = (static_cast<double>(ref_logit) - lse) * 1000.0;
  row.divergent = top != ref_tok;

  // Force ref[n]: mask every other entry to -inf; the ref entry keeps its
  // computed value, so the greedy argmax appends exactly the reference token.
  for (int32_t j = 0; j < vocab_size; ++j) logits[j] = -INFINITY;
  logits[ref_tok] = ref_logit;
  ++st->calls;
}

int Adjudicate(const Args& a) {
  const std::vector<int32_t> ref = ReadIds(a.ref_ids.c_str());
  const std::string prompt = ReadPromptStripped(a.prompt_file.c_str());
  const int n_ref = static_cast<int>(ref.size());

  vllm_engine* engine = LoadEngine(a.model);
  AdjudicateState st;
  st.ref = ref.data();
  st.n_ref = n_ref;
  st.rows.resize(n_ref);

  vllm_sampling_params sp = GateSampling(n_ref);
  sp.logits_processor = &AdjudicateCb;
  sp.logits_processor_user_data = &st;
  vllm_completion out{};
  const vllm_status status = vllm_complete(engine, prompt.c_str(), &sp, &out);
  if (status != VLLM_OK) {
    std::fprintf(stderr, "tg200-neartie: adjudication run failed (status %d): %s\n",
                 static_cast<int>(status), vllm_last_error());
    return 3;
  }
  const std::string body = BodyBytes(out);
  const std::string body_md5 = Md5Hex(body);
  vllm_completion_free(&out);
  vllm_engine_free(engine);

  if (st.calls != n_ref) {
    std::fprintf(stderr,
                 "tg200-neartie: integrity: %d sampler callbacks for %d "
                 "reference positions — the engine did not give the processor "
                 "one row per decode step; no verdict is possible\n",
                 st.calls, n_ref);
    return 4;
  }
  if (!a.expect_md5.empty() && !Md5Matches(body_md5, a.expect_md5)) {
    std::fprintf(stderr,
                 "tg200-neartie: integrity: forced-walk body md5 %s does not "
                 "match the reference %s — the teacher-forced walk did not "
                 "reproduce the reference body; the report would be fiction\n",
                 body_md5.c_str(), a.expect_md5.c_str());
    return 4;
  }

  // Verdict per the ratified band: a position fails only when the gap EXCEEDS
  // the band; in-band divergences are the near-ties the band exists to admit.
  int divergent = 0, over_band = 0;
  double max_gap = 0.0;
  for (const GapRow& r : st.rows) {
    if (r.divergent) ++divergent;
    if (r.gap_mnats > a.band_mnats) ++over_band;
    max_gap = std::max(max_gap, r.gap_mnats);
  }
  const bool pass = over_band == 0;

  // Body evidence: write the forced run's body (text + '\n', the md5sum shape).
  const std::string body_path =
      a.body_out.empty() ? a.json_out + ".body.txt" : a.body_out;
  WriteBytes(body_path.c_str(), body.data(), body.size());

  // JSON report (machine-readable).
  std::string j;
  char fbuf[64];
  j += "{\n";
  j += "  \"schema\": \"tg200-neartie/v1\",\n";
  std::snprintf(fbuf, sizeof(fbuf), "%.3f", a.band_mnats);
  j += "  \"band_mnats\": " + std::string(fbuf) + ",\n";
  j += std::string("  \"verdict\": \"") + (pass ? "PASS" : "FAIL") + "\",\n";
  j += "  \"model\": \"" + JsonEscape(a.model) + "\",\n";
  j += "  \"prompt_file\": \"" + JsonEscape(a.prompt_file) + "\",\n";
  j += "  \"ref_ids\": \"" + JsonEscape(a.ref_ids) + "\",\n";
  if (!a.expect_md5.empty()) {
    j += "  \"reference_body_md5\": \"" + JsonEscape(a.expect_md5) + "\",\n";
  }
  if (!a.note.empty()) j += "  \"note\": \"" + JsonEscape(a.note) + "\",\n";
  j += "  \"n_positions\": " + std::to_string(n_ref) + ",\n";
  j += "  \"divergent_positions\": " + std::to_string(divergent) + ",\n";
  j += "  \"positions_over_band\": " + std::to_string(over_band) + ",\n";
  std::snprintf(fbuf, sizeof(fbuf), "%.3f", max_gap);
  j += "  \"max_gap_mnats\": " + std::string(fbuf) + ",\n";
  j += "  \"forced_body_md5\": \"" + body_md5 + "\",\n";
  j += "  \"positions\": [\n";
  for (int i = 0; i < n_ref; ++i) {
    const GapRow& r = st.rows[i];
    std::snprintf(fbuf, sizeof(fbuf), "%.3f", r.gap_mnats);
    const std::string gap = fbuf;
    std::snprintf(fbuf, sizeof(fbuf), "%.3f", r.ref_logprob_mnats);
    j += "    {\"n\": " + std::to_string(r.n) +
         ", \"ref\": " + std::to_string(r.ref) +
         ", \"argmax\": " + std::to_string(r.argmax) +
         ", \"gap_mnats\": " + gap +
         ", \"ref_logprob_mnats\": " + std::string(fbuf) + "}" +
         (i + 1 < n_ref ? "," : "") + "\n";
  }
  j += "  ]\n}\n";
  WriteBytes(a.json_out.c_str(), j.data(), j.size());

  // Human summary: the worst positions first (a bit-identical arm has none).
  std::vector<const GapRow*> sorted;
  for (const GapRow& r : st.rows) sorted.push_back(&r);
  std::sort(sorted.begin(), sorted.end(),
            [](const GapRow* x, const GapRow* y) { return x->gap_mnats > y->gap_mnats; });
  std::printf("tg200-neartie: band=%.3f mnats positions=%d\n", a.band_mnats, n_ref);
  std::printf("%6s %8s %10s %8s %14s %18s\n", "rank", "pos", "ref", "argmax",
              "gap_mnats", "ref_logprob_mnats");
  int shown = 0;
  for (int i = 0; i < static_cast<int>(sorted.size()); ++i) {
    if (a.top_k >= 0 && shown >= a.top_k) break;
    if (sorted[i]->gap_mnats <= 0.0) break;
    const GapRow* r = sorted[i];
    std::snprintf(fbuf, sizeof(fbuf), "%.3f", r->gap_mnats);
    const std::string gap = fbuf;
    std::snprintf(fbuf, sizeof(fbuf), "%.3f", r->ref_logprob_mnats);
    std::printf("%6d %8d %10d %8d %14s %18s\n", i + 1, r->n, r->ref, r->argmax,
                gap.c_str(), fbuf);
    ++shown;
  }
  if (shown == 0) {
    std::printf("tg200-neartie: no divergent positions (argmax == reference at every step)\n");
  }
  std::printf("tg200-neartie: verdict=%s divergent=%d over_band=%d max_gap_mnats=%.3f "
              "body_md5=%s json=%s\n",
              pass ? "PASS" : "FAIL", divergent, over_band, max_gap,
              body_md5.c_str(), a.json_out.c_str());
  return pass ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  if (!ParseArgs(argc, argv, a)) {
    Usage(argv[0], stderr);
    return 2;
  }
  return a.mode == "capture" ? Capture(a) : Adjudicate(a);
}
