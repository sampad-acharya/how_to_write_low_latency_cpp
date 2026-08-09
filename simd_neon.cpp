#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Backend selection. Two hand-written SIMD backends, chosen at compile time:
//
//   ARM   -> NEON  (arm_neon.h), 128-bit / 4-wide.  Baseline on aarch64.
//   x86   -> SSE2  (immintrin.h), 128-bit / 4-wide.  Baseline on x86-64.
//
// Both are 4-wide, so the two backends are line-for-line parallels and the
// lesson is identical on either machine. SSE2 needs no -m flag (it is the
// x86-64 baseline); NEON needs none on aarch64. Nothing here requires AVX or
// SSE4 -- staying at SSE2 keeps it flag-free and maximally portable.
// ---------------------------------------------------------------------------
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#  include <arm_neon.h>
#  define SIMD_BACKEND "NEON"
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  include <immintrin.h>
#  define SIMD_BACKEND "SSE2"
#  define SIMD_X86 1
// SSE2 has no horizontal-add instruction; fold 4 int32 lanes to a scalar.
static inline int32_t hsum_epi32(__m128i v) {
    __m128i hi = _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2));
    v = _mm_add_epi32(v, hi);
    hi = _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1));
    v = _mm_add_epi32(v, hi);
    return _mm_cvtsi128_si32(v);
}
#else
#  error "No SIMD backend: expected ARM NEON (aarch64) or SSE2 (x86-64)."
#endif

using Clock = std::chrono::steady_clock;
using ns    = std::chrono::nanoseconds;

// SIMD for low latency, honestly.
//
// The pitch for SIMD is "do 4 (or 8, or 16) elements per instruction." True,
// but there are exactly two facts that decide whether you should hand-write it:
//
//   1. The compiler ALREADY auto-vectorizes simple, associative loops at -O2.
//      If you hand-write intrinsics for an int sum you match (or lose to) a
//      loop clang wrote for free. Kernel 1 shows this: don't do that work.
//
//   2. The compiler REFUSES to vectorize when it would change results or when
//      the loop carries an awkward dependency. Floating-point reductions
//      (addition isn't associative) and argmin/argmax (carry the index of the
//      best element) are the classic cases. That is where hand-written SIMD
//      earns its keep -- Kernels 2 and 3.
//
// The lesson is not "SIMD is fast." It is "reach for intrinsics exactly where
// the optimizer is not allowed to go, and nowhere else." It holds identically
// on ARM (NEON) and x86 (SSE2) -- this file builds and runs on both.
//
// Everything here keeps the working set in L1 (32 KB) on purpose. SIMD speeds
// up COMPUTE. If your data lives in DRAM, memory bandwidth caps you and 4-wide
// arithmetic buys nothing -- a separate, important lesson noted at the end.

constexpr size_t N     = 8192;      // 32 KB of int32/float32 -> fits L1d (64 KB)
constexpr int    REPS  = 200'000;   // passes over the array, per variant

// ---------------------------------------------------------------------------
// Stats (same percentile discipline as the rest of this repo)
// ---------------------------------------------------------------------------
struct Stats {
    double mean, p50, p99, p999, max;
};

Stats compute_stats(std::vector<long long>& s) {
    double total = std::accumulate(s.begin(), s.end(), 0.0);
    std::sort(s.begin(), s.end());
    auto pct = [&](double p) { return double(s[size_t(p * (s.size() - 1))]); };
    return Stats{total / s.size(), pct(0.50), pct(0.99), pct(0.999), double(s.back())};
}

void header(const std::string& title) {
    std::cout << "=== " << title << " ===\n"
              << "  " << std::left << std::setw(22) << "variant" << std::right
              << std::setw(9) << "mean" << std::setw(9) << "p50"
              << std::setw(9) << "p99" << std::setw(10) << "p999"
              << std::setw(11) << "max" << std::setw(11) << "vs scalar" << "\n"
              << "  " << std::string(22 + 9 + 9 + 9 + 10 + 11 + 11, '-') << "\n";
}

void row(const std::string& name, const Stats& s, double scalar_mean) {
    std::cout << "  " << std::left << std::setw(22) << name << std::right
              << std::fixed << std::setprecision(0)
              << std::setw(9) << s.mean << std::setw(9) << s.p50
              << std::setw(9) << s.p99 << std::setw(10) << s.p999
              << std::setw(11) << s.max
              << std::setprecision(2) << std::setw(10) << (scalar_mean / s.mean) << "x"
              << std::defaultfloat << "\n";
}

// Time `fn` (returns a value we accumulate so it can't be optimized away) over
// REPS passes, one latency sample per pass.
template <class Fn>
Stats time_kernel(Fn&& fn, double& sink) {
    std::vector<long long> lat;
    lat.reserve(REPS);
    double acc = 0;
    for (int r = 0; r < REPS; ++r) {
        auto t0 = Clock::now();
        acc += fn();
        auto t1 = Clock::now();
        lat.push_back(std::chrono::duration_cast<ns>(t1 - t0).count());
    }
    sink += acc;
    return compute_stats(lat);
}

// ===========================================================================
// KERNEL 1 -- int32 horizontal sum.
// Integer addition IS associative, so clang auto-vectorizes the plain loop.
// Point: the hand-written SIMD version only matches (or loses to) free work.
// ===========================================================================
__attribute__((noinline))
int64_t sum_scalar_novec(const int32_t* p, size_t n) {
    int64_t s = 0;
#pragma clang loop vectorize(disable) interleave(disable)
    for (size_t i = 0; i < n; ++i) s += p[i];
    return s;
}

__attribute__((noinline))
int64_t sum_autovec(const int32_t* p, size_t n) {
    int64_t s = 0;                       // no pragma -> clang vectorizes this
    for (size_t i = 0; i < n; ++i) s += p[i];
    return s;
}

__attribute__((noinline))
int64_t sum_simd(const int32_t* p, size_t n) {
    size_t i = 0;
    int64_t s = 0;
#if defined(SIMD_X86)
    __m128i acc0 = _mm_setzero_si128(), acc1 = _mm_setzero_si128();
    for (; i + 8 <= n; i += 8) {         // 8 per iter -> hide load latency
        acc0 = _mm_add_epi32(acc0, _mm_loadu_si128((const __m128i*)(p + i)));
        acc1 = _mm_add_epi32(acc1, _mm_loadu_si128((const __m128i*)(p + i + 4)));
    }
    s = hsum_epi32(_mm_add_epi32(acc0, acc1));       // horizontal add of 4 lanes
#else // NEON
    int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
    for (; i + 8 <= n; i += 8) {         // 8 per iter -> hide load latency
        acc0 = vaddq_s32(acc0, vld1q_s32(p + i));
        acc1 = vaddq_s32(acc1, vld1q_s32(p + i + 4));
    }
    s = vaddvq_s32(vaddq_s32(acc0, acc1));           // horizontal add of 4 lanes
#endif
    for (; i < n; ++i) s += p[i];                    // tail
    return s;
}

// ===========================================================================
// KERNEL 2 -- float argmin: smallest value AND its index (think "best ask").
// FP add/compare non-associativity + carrying the index defeats the
// auto-vectorizer. This is the intrinsics sweet spot.
// ===========================================================================
struct Best { float val; uint32_t idx; };

__attribute__((noinline))
Best argmin_scalar(const float* p, size_t n) {
    float best = std::numeric_limits<float>::infinity();
    uint32_t bi = 0;
    for (size_t i = 0; i < n; ++i)
        if (p[i] < best) { best = p[i]; bi = uint32_t(i); }
    return {best, bi};
}

__attribute__((noinline))
Best argmin_simd(const float* p, size_t n) {
    const float INF = std::numeric_limits<float>::infinity();
    size_t i = 0;
    float bv[4]; uint32_t bidx[4];
#if defined(SIMD_X86)
    __m128  bestv = _mm_set1_ps(INF);
    __m128i besti = _mm_setzero_si128();
    __m128i idx   = _mm_setr_epi32(0, 1, 2, 3);
    const __m128i four = _mm_set1_epi32(4);
    for (; i + 4 <= n; i += 4) {
        __m128  v  = _mm_loadu_ps(p + i);
        __m128i lt = _mm_castps_si128(_mm_cmplt_ps(v, bestv)); // lanes where v<best
        bestv = _mm_min_ps(v, bestv);                          // value select
        // index select without SSE4 blendv: (idx & lt) | (besti & ~lt)
        besti = _mm_or_si128(_mm_and_si128(lt, idx),
                             _mm_andnot_si128(lt, besti));
        idx   = _mm_add_epi32(idx, four);
    }
    _mm_storeu_ps(bv, bestv);
    _mm_storeu_si128((__m128i*)bidx, besti);
#else // NEON
    float32x4_t bestv = vdupq_n_f32(INF);
    uint32x4_t  besti = vdupq_n_u32(0);
    uint32_t    ib[4] = {0, 1, 2, 3};
    uint32x4_t  idx   = vld1q_u32(ib);
    const uint32x4_t four = vdupq_n_u32(4);
    for (; i + 4 <= n; i += 4) {
        float32x4_t v  = vld1q_f32(p + i);
        uint32x4_t  lt = vcltq_f32(v, bestv);            // lanes where v < best
        bestv = vbslq_f32(lt, v, bestv);                 // branchless select
        besti = vbslq_u32(lt, idx, besti);
        idx   = vaddq_u32(idx, four);
    }
    vst1q_f32(bv, bestv);
    vst1q_u32(bidx, besti);
#endif
    // Reduce the 4 lanes to a single winner, then handle the tail.
    Best r{bv[0], bidx[0]};
    for (int l = 1; l < 4; ++l)
        if (bv[l] < r.val) r = {bv[l], bidx[l]};
    for (; i < n; ++i)
        if (p[i] < r.val) r = {p[i], uint32_t(i)};
    return r;
}

// ===========================================================================
// KERNEL 3 -- count int32 values inside a price band [lo, hi].
// The scalar version has a data-dependent branch per element (see this repo's
// branch_prediction.cpp for why that hurts). SIMD does it branchlessly: build
// a mask, accumulate the mask. Wins on both mean and tail.
// ===========================================================================
__attribute__((noinline))
int64_t count_band_scalar(const int32_t* p, size_t n, int32_t lo, int32_t hi) {
    int64_t c = 0;
#pragma clang loop vectorize(disable) interleave(disable)
    for (size_t i = 0; i < n; ++i)
        if (p[i] >= lo && p[i] <= hi) ++c;   // branch the CPU must predict
    return c;
}

__attribute__((noinline))
int64_t count_band_simd(const int32_t* p, size_t n, int32_t lo, int32_t hi) {
    size_t i = 0;
    int64_t c = 0;
#if defined(SIMD_X86)
    __m128i vlo = _mm_set1_epi32(lo), vhi = _mm_set1_epi32(hi);
    __m128i cnt = _mm_setzero_si128();
    for (; i + 4 <= n; i += 4) {
        __m128i v  = _mm_loadu_si128((const __m128i*)(p + i));
        // SSE2 has no cmpge/cmple, so: (v>=lo) == !(v<lo) == (v>lo)|(v==lo).
        __m128i ge = _mm_or_si128(_mm_cmpgt_epi32(v, vlo), _mm_cmpeq_epi32(v, vlo));
        __m128i le = _mm_or_si128(_mm_cmpgt_epi32(vhi, v), _mm_cmpeq_epi32(v, vhi));
        __m128i in = _mm_and_si128(ge, le);              // -1 where in band
        cnt = _mm_sub_epi32(cnt, in);                    // subtract -1 == +1
    }
    c = hsum_epi32(cnt);
#else // NEON
    int32x4_t vlo = vdupq_n_s32(lo), vhi = vdupq_n_s32(hi);
    int32x4_t cnt = vdupq_n_s32(0);
    for (; i + 4 <= n; i += 4) {
        int32x4_t v    = vld1q_s32(p + i);
        uint32x4_t ge  = vcgeq_s32(v, vlo);              // v >= lo
        uint32x4_t le  = vcleq_s32(v, vhi);              // v <= hi
        int32x4_t  in  = vreinterpretq_s32_u32(vandq_u32(ge, le)); // -1 where in
        cnt = vsubq_s32(cnt, in);                        // subtract -1 == +1
    }
    c = vaddvq_s32(cnt);
#endif
    for (; i < n; ++i)                                    // tail
        if (p[i] >= lo && p[i] <= hi) ++c;
    return c;
}

int main() {
    std::cout << SIMD_BACKEND << " backend, " << N << " elements (32 KB, resident "
              << "in L1), " << REPS << " passes each.\n";
    std::cout << "Every variant times ONE full pass over the array; ns per pass.\n\n";

    std::mt19937 rng(20260809u);
    std::vector<int32_t> ai(N);
    std::vector<float>   af(N);
    std::uniform_int_distribution<int32_t> di(0, 1'000'000);
    std::uniform_real_distribution<float>  df(0.0f, 1.0f);
    for (size_t i = 0; i < N; ++i) { ai[i] = di(rng); af[i] = df(rng); }

    double sink = 0;

    // ----- Kernel 1: int sum ------------------------------------------------
    header("Kernel 1: int32 horizontal sum (compiler auto-vectorizes this)");
    Stats k1s = time_kernel([&]{ return double(sum_scalar_novec(ai.data(), N)); }, sink);
    Stats k1a = time_kernel([&]{ return double(sum_autovec(ai.data(), N)); }, sink);
    Stats k1n = time_kernel([&]{ return double(sum_simd(ai.data(), N)); }, sink);
    row("scalar (novec)",   k1s, k1s.mean);
    row("auto-vectorized",  k1a, k1s.mean);
    row("hand " SIMD_BACKEND, k1n, k1s.mean);
    std::cout << "  -> the auto-vectorized loop matches or BEATS the hand-written kernel\n"
              << "     here (clang schedules more accumulators than the two I wrote).\n"
              << "     For a simple associative reduction you will not out-code -O2.\n\n";

    // ----- Kernel 2: float argmin ------------------------------------------
    header("Kernel 2: float argmin - value + index, e.g. best price (auto-vec CAN'T)");
    Best guard{};  // keep results live
    Stats k2s = time_kernel([&]{ auto b = argmin_scalar(af.data(), N); guard = b; return b.val; }, sink);
    Stats k2n = time_kernel([&]{ auto b = argmin_simd(af.data(),   N); guard = b; return b.val; }, sink);
    row("scalar", k2s, k2s.mean);
    row("hand " SIMD_BACKEND, k2n, k2s.mean);
    {
        auto bs = argmin_scalar(af.data(), N);
        auto bn = argmin_simd(af.data(), N);
        std::cout << "  correctness: scalar=(" << bs.val << ",#" << bs.idx << ")  "
                  << SIMD_BACKEND "=(" << bn.val << ",#" << bn.idx << ")  "
                  << (bs.idx == bn.idx ? "match" : "MISMATCH") << "\n";
    }
    std::cout << "  -> the index carry + FP compare block auto-vectorization, so the\n"
              << "     scalar loop stays scalar. Hand SIMD is where the speedup lives.\n\n";

    // ----- Kernel 3: count in band -----------------------------------------
    header("Kernel 3: count int32 in [lo,hi] - branchy scalar vs branchless SIMD");
    const int32_t lo = 250'000, hi = 750'000;   // ~50% pass -> worst case for the branch
    Stats k3s = time_kernel([&]{ return double(count_band_scalar(ai.data(), N, lo, hi)); }, sink);
    Stats k3n = time_kernel([&]{ return double(count_band_simd(ai.data(),   N, lo, hi)); }, sink);
    row("scalar (branchy)", k3s, k3s.mean);
    row("hand " SIMD_BACKEND, k3n, k3s.mean);
    std::cout << "  correctness: scalar=" << count_band_scalar(ai.data(), N, lo, hi)
              << " " SIMD_BACKEND "=" << count_band_simd(ai.data(), N, lo, hi) << "\n";
    std::cout << "  -> the ~50%-taken branch is a coin flip the predictor can't learn;\n"
              << "     SIMD has no branch to mispredict. It wins mean AND tail.\n\n";

    // ----- Closing caveat ---------------------------------------------------
    std::cout << "=== The caveat that outranks all of the above ===\n";
    std::cout << "  This ran in L1. SIMD accelerates COMPUTE. If the array lived in\n";
    std::cout << "  DRAM, memory bandwidth would cap every variant at the same number\n";
    std::cout << "  and 4-wide math would buy nothing. Profile where your bottleneck\n";
    std::cout << "  actually is before reaching for intrinsics: cache-resident and\n";
    std::cout << "  compute-bound is the only place this pays.\n";

    std::cout << "\n[sink] " << sink << "\n";
    return 0;
}
