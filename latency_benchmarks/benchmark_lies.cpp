#include "bench_util.h"
#include <cmath>

// Why microbenchmarks lie: the four ways your numbers are already wrong.
//
//   1. DEAD-CODE ELIMINATION. If nothing observes the result, the optimizer
//      deletes the whole loop and you "measure" 0 ns. (This is not theoretical
//      -- the store loop in memory_ordering.cpp reported 0 until we forced it.)
//   2. HOISTING. A loop-invariant computation is run once, not N times, so you
//      divide one result by N and report a fraction of the truth.
//   3. TIMER OVERHEAD. Clock::now() itself costs tens of ns. Time an op faster
//      than that and you are mostly measuring the clock.
//   4. NO WARMUP. The first pass pays page faults, cold caches, and CPU
//      frequency ramp -- a cold mean is not a steady-state mean.
//
// Each section shows the lie and the fix side by side. This is the opener for
// the observability part of the talk: trust a number only after you know which
// of these it survived.

using namespace bench;

constexpr int N = 20'000'000;

// Do N units of a real computation and return elapsed ns.
template <class Body>
long long timed(Body&& b) {
    auto t0 = Clock::now();
    b();
    return std::chrono::duration_cast<ns>(Clock::now() - t0).count();
}

int main() {
    banner("Why microbenchmarks lie");

    // ---- Lie 1: dead-code elimination -------------------------------------
    long long dce_lie = timed([]{
        double s = 0;
        for (int i = 0; i < N; ++i) s += std::sqrt(double(i));
        // s is never used -> the optimizer may delete the entire loop.
    });
    double sink = 0;
    long long dce_fix = timed([&]{
        double s = 0;
        for (int i = 0; i < N; ++i) s += std::sqrt(double(i));
        sink = s; do_not_optimize(sink);   // now the work must happen
    });
    std::cout << "1) Dead-code elimination:\n"
              << "   unobserved result : " << (double(dce_lie) / N) << " ns/op   <- the lie\n"
              << "   do_not_optimize   : " << (double(dce_fix) / N) << " ns/op   <- the truth\n\n";

    // ---- Lie 2: loop-invariant hoisting -----------------------------------
    volatile int seed = 7;
    long long hoist_lie = timed([&]{
        double s = 0; double c = std::sin(1.234);   // invariant, hoisted out
        for (int i = 0; i < N; ++i) s += c;
        sink = s; do_not_optimize(sink);
    });
    long long hoist_fix = timed([&]{
        double s = 0;
        for (int i = 0; i < N; ++i) s += std::sin(1.234 + i * 1e-9 * seed); // depends on i
        sink = s; do_not_optimize(sink);
    });
    std::cout << "2) Loop-invariant hoisting:\n"
              << "   invariant in loop : " << (double(hoist_lie) / N) << " ns/op   <- computed once\n"
              << "   made i-dependent  : " << (double(hoist_fix) / N) << " ns/op   <- computed N times\n\n";

    // ---- Lie 3: timer overhead --------------------------------------------
    std::vector<long long> clk;
    for (int i = 0; i < 200000; ++i) {
        auto a = Clock::now(); auto b = Clock::now();
        clk.push_back(std::chrono::duration_cast<ns>(b - a).count());
    }
    Stats c = compute_stats(clk);
    std::cout << "3) Timer overhead (cost of two Clock::now() calls):\n"
              << "   mean " << c.mean << " ns, p50 " << c.p50 << " ns.\n"
              << "   Any per-op time near this is mostly measuring the clock, not the op.\n\n";

    // ---- Lie 4: no warmup -------------------------------------------------
    std::vector<long long> passes;
    for (int r = 0; r < 12; ++r) {
        passes.push_back(timed([&]{
            double s = 0; for (int i = 0; i < N/4; ++i) s += std::sqrt(double(i));
            sink = s; do_not_optimize(sink);
        }));
    }
    std::cout << "4) No warmup (same work, 12 consecutive passes, ms):\n   ";
    for (size_t i = 0; i < passes.size(); ++i)
        std::cout << std::fixed << std::setprecision(1) << (passes[i] / 1e6)
                  << (i + 1 < passes.size() ? " " : "\n");
    std::cout << std::defaultfloat
              << "   The first pass is the outlier (cold cache, freq ramp). Discard warmup\n"
              << "   or you fold startup cost into your steady-state mean.\n\n";

    std::cout << "Every clean benchmark in this suite already applies all four fixes.\n";
    return 0;
}
