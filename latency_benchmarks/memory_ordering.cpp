#include "bench_util.h"
#include <atomic>

// The cost of a memory ordering is set by the hardware model, not the keyword.
//
//   x86-64 (TSO): acquire/release loads & stores are FREE (plain mov); only a
//                 seq_cst STORE costs extra; a lock-prefixed RMW is already a
//                 full barrier, so relaxed == seq_cst for RMW.
//   ARM  (weak):  relaxed is a bare ldr/str; acquire/release are ldar/stlr;
//                 seq_cst RMW uses ldaxr/stlxr. Every step up costs cycles --
//                 though Apple's cores make ldar/stlr nearly free, so on this
//                 chip the visible tax is mostly on the RMW.
//
// "seq_cst is the safe default" is free on x86, a real tax on ARM. Run on both.
// Single-threaded on purpose: no contention, just fence/instruction cost.

using namespace bench;

constexpr uint64_t ITERS = 40'000'000;
constexpr int      REPS  = 9;

template <class Body>
double bench_op(Body&& body) {
    std::vector<double> per_op;
    for (int r = 0; r < REPS; ++r) {
        auto t0 = Clock::now();
        for (uint64_t k = 0; k < ITERS; ++k) body(k);
        auto t1 = Clock::now();
        per_op.push_back(double(std::chrono::duration_cast<ns>(t1 - t0).count()) / ITERS);
    }
    std::sort(per_op.begin(), per_op.end());
    return per_op[per_op.size() / 2];
}

static void group(const std::string& title, const char* n[3], double v[3]) {
    std::cout << title << ":\n";
    for (int i = 0; i < 3; ++i)
        std::cout << "    " << std::left << std::setw(9) << n[i] << std::right
                  << std::fixed << std::setprecision(3) << std::setw(8) << v[i] << " ns/op"
                  << std::setw(8) << std::setprecision(2) << (v[i] / v[0]) << "x vs relaxed"
                  << (i == 0 ? "  (baseline)" : "") << "\n" << std::defaultfloat;
    std::cout << "\n";
}

int main() {
    banner("Memory-ordering cost (single thread)");
    std::cout << ITERS << " ops x " << REPS << " reps (median).\n\n";

    // `as` is volatile: else the compiler deletes the store loop as dead and
    // reports 0 ns/op (that trap is its own benchmark -- see benchmark_lies).
    alignas(128) volatile std::atomic<uint64_t> as{0};
    alignas(128) std::atomic<uint64_t> al{0};
    alignas(128) std::atomic<uint64_t> ar{0};
    volatile uint64_t sink = 0;

    double st[3] = {
        bench_op([&](uint64_t k){ as.store(k, std::memory_order_relaxed); }),
        bench_op([&](uint64_t k){ as.store(k, std::memory_order_release); }),
        bench_op([&](uint64_t k){ as.store(k, std::memory_order_seq_cst); })};
    const char* sn[3] = {"relaxed", "release", "seq_cst"};
    group("STORE  (x86: only seq_cst should jump)", sn, st);

    double ld[3] = {
        bench_op([&](uint64_t){ sink += al.load(std::memory_order_relaxed); }),
        bench_op([&](uint64_t){ sink += al.load(std::memory_order_acquire); }),
        bench_op([&](uint64_t){ sink += al.load(std::memory_order_seq_cst); })};
    const char* ln[3] = {"relaxed", "acquire", "seq_cst"};
    group("LOAD   (x86: all three tie)", ln, ld);

    double rmw[3] = {
        bench_op([&](uint64_t){ ar.fetch_add(1, std::memory_order_relaxed); }),
        bench_op([&](uint64_t){ ar.fetch_add(1, std::memory_order_acq_rel); }),
        bench_op([&](uint64_t){ ar.fetch_add(1, std::memory_order_seq_cst); })};
    const char* rn[3] = {"relaxed", "acq_rel", "seq_cst"};
    group("RMW    (x86: lock is already full-barrier -> all tie)", rn, rmw);

    std::cout << "The ordering tax is a property of the CPU, not the source line.\n";
    std::cout << "[sink] " << sink << " [rmw] " << ar.load(std::memory_order_relaxed) << "\n";
    return 0;
}
