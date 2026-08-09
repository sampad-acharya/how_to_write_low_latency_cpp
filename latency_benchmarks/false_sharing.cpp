#include "bench_util.h"
#include <atomic>

// False sharing, and why the fix is architecture-dependent.
//
// Two threads writing two DIFFERENT variables should never contend. They do,
// if those variables share a cache line: each write invalidates the other
// core's copy and the line ping-pongs across the coherence fabric.
//
//   x86-64        : 64-byte lines.  alignas(64) fixes it.
//   Apple Silicon : 128-byte lines. alignas(64) is NOT enough -- two
//                   64B-aligned cells still share one 128B line.
//
// Same "correct" padding, different result per machine. Run on both.

using namespace bench;

struct alignas(64)  Cell64  { std::atomic<uint64_t> v{0}; };
struct alignas(128) Cell128 { std::atomic<uint64_t> v{0}; };

constexpr uint64_t INCREMENTS = 1'500'000;
constexpr int      REPS       = 15;

template <class Access>
long long run_once(int T, Access access) {
    std::vector<std::thread> ths;
    ths.reserve(T);
    std::atomic<int> go{0};
    for (int i = 0; i < T; ++i)
        ths.emplace_back([&, i] {
            while (go.load(std::memory_order_acquire) == 0) {}
            for (uint64_t k = 0; k < INCREMENTS; ++k)
                access(i).fetch_add(1, std::memory_order_relaxed);
        });
    auto t0 = Clock::now();
    go.store(1, std::memory_order_release);
    for (auto& t : ths) t.join();
    return std::chrono::duration_cast<ns>(Clock::now() - t0).count();
}

static void report(const std::string& name, std::vector<long long>& s, int T) {
    Stats st = compute_stats(s);
    double mops = double(T) * INCREMENTS / (st.p50 / 1e9) / 1e6;
    std::cout << "  " << std::left << std::setw(26) << name << std::right
              << std::fixed << std::setprecision(2)
              << std::setw(9) << (st.p50 / 1e6) << " ms"
              << std::setw(11) << mops << " Mops/s\n" << std::defaultfloat;
}

int main() {
    int T = std::min<int>(std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 8, 8);
    banner("False sharing");
    std::cout << T << " threads x " << INCREMENTS << " independent increments, "
              << REPS << " runs (median shown).\n\n";

    std::vector<std::atomic<uint64_t>> packed(T);
    std::vector<Cell64>  c64(T);
    std::vector<Cell128> c128(T);
    auto ap = [&](int i) -> std::atomic<uint64_t>& { return packed[i]; };
    auto a64  = [&](int i) -> std::atomic<uint64_t>& { return c64[i].v; };
    auto a128 = [&](int i) -> std::atomic<uint64_t>& { return c128[i].v; };

    std::vector<long long> sp, s64, s128;
    for (int r = 0; r < 3; ++r) { run_once(T, a128); run_once(T, ap); }  // warmup
    for (int r = 0; r < REPS; ++r) {
        s128.push_back(run_once(T, a128));
        s64.push_back (run_once(T, a64));
        sp.push_back  (run_once(T, ap));
    }
    report("128B padded (own line)",  s128, T);
    report("64B padded (alignas 64)", s64,  T);
    report("packed (shared line)",    sp,   T);

    std::cout << "\n";
    if (HW_LINE >= 128)
        std::cout << "Apple Silicon (128B line): 64B padding stays well behind 128B --\n"
                  << "the x86 'alignas(64)' fix is a latency bug here.\n";
    else
        std::cout << "x86-64 (64B line): 64B padding already ties 128B. The SAME binary\n"
                  << "regresses on Apple Silicon -- that is the cross-architecture point.\n";
    return 0;
}
