#include "bench_util.h"
#include <atomic>
#include <mutex>

// When do atomics LOSE to a mutex? Under contention.
//
// At low thread counts a lock-free atomic increment wins easily: no syscall,
// no blocking. But every contended atomic RMW still bounces the cache line
// between cores, and a spinlock burns cores retrying. Past the physical core
// count the picture inverts: the spinlock/atomic tail explodes (starvation,
// coherence storms) while a real mutex parks waiters and degrades gracefully.
//
// We sweep thread count and report the per-op latency PERCENTILES for three
// primitives. The crossover point -- and it differs on x86 vs ARM -- is the
// slide. Sampled 1-in-64 ops so the timer does not dominate the measurement.

using namespace bench;

constexpr uint64_t OPS_PER_THREAD = 400'000;
constexpr int      SAMPLE_EVERY   = 64;

struct SpinLock {
    std::atomic<bool> f{false};
    void lock()   { while (f.exchange(true, std::memory_order_acquire)) { while (f.load(std::memory_order_relaxed)) cpu_relax(); } }
    void unlock() { f.store(false, std::memory_order_release); }
};

// Each thread increments a shared counter OPS_PER_THREAD times; every 64th op
// is individually timed to build a latency distribution. `mode`: 0 atomic,
// 1 spinlock, 2 mutex.
template <int MODE>
Stats run(int T, std::atomic<uint64_t>& counter, SpinLock& spin, std::mutex& mtx,
          double& throughput) {
    std::vector<std::vector<long long>> per_thread(T);
    std::atomic<int> go{0};
    std::vector<std::thread> ths;
    auto body = [&](int id) {
        auto& lat = per_thread[id];
        lat.reserve(OPS_PER_THREAD / SAMPLE_EVERY + 1);
        while (go.load(std::memory_order_acquire) == 0) {}
        for (uint64_t k = 0; k < OPS_PER_THREAD; ++k) {
            bool sample = (k % SAMPLE_EVERY) == 0;
            Clock::time_point t0;
            if (sample) t0 = Clock::now();
            if constexpr (MODE == 0) {
                counter.fetch_add(1, std::memory_order_relaxed);
            } else if constexpr (MODE == 1) {
                spin.lock(); volatile uint64_t x = counter.load(std::memory_order_relaxed) + 1;
                counter.store(x, std::memory_order_relaxed); spin.unlock();
            } else {
                std::lock_guard<std::mutex> g(mtx);
                counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
            }
            if (sample) lat.push_back(std::chrono::duration_cast<ns>(Clock::now() - t0).count());
        }
    };
    for (int i = 0; i < T; ++i) ths.emplace_back(body, i);
    auto t0 = Clock::now();
    go.store(1, std::memory_order_release);
    for (auto& t : ths) t.join();
    double secs = std::chrono::duration_cast<ns>(Clock::now() - t0).count() / 1e9;
    throughput = double(T) * OPS_PER_THREAD / secs / 1e6;

    std::vector<long long> all;
    for (auto& v : per_thread) all.insert(all.end(), v.begin(), v.end());
    return compute_stats(all);
}

int main() {
    banner("Atomics vs mutex under contention");
    unsigned hw = std::thread::hardware_concurrency();
    std::cout << "hardware_concurrency = " << hw << ".  Per-op latency (ns), "
              << "1-in-" << SAMPLE_EVERY << " sampled.\n";
    std::cout << "Watch the tail (p99/p999) cross over as threads exceed cores.\n\n";

    std::vector<int> thread_counts = {1, 2, 4, int(hw ? hw : 8), int(2 * (hw ? hw : 8))};

    for (int T : thread_counts) {
        std::cout << "---- " << T << " threads ----\n";
        print_header();
        std::atomic<uint64_t> counter{0}; SpinLock spin; std::mutex mtx; double tp;
        Stats a = run<0>(T, counter, spin, mtx, tp);
        std::cout << std::setw(0); print_row("atomic  (" + std::to_string(int(tp)) + " Mops/s)", a);
        counter = 0; Stats s = run<1>(T, counter, spin, mtx, tp);
        print_row("spinlock(" + std::to_string(int(tp)) + " Mops/s)", s);
        counter = 0; Stats m = run<2>(T, counter, spin, mtx, tp);
        print_row("mutex   (" + std::to_string(int(tp)) + " Mops/s)", m);
        std::cout << "\n";
    }
    std::cout << "Low contention: atomic wins. High contention: watch the spinlock's\n"
              << "p999 blow up while the mutex stays bounded. The crossover moves with\n"
              << "core count and architecture -- benchmark it on the CPU you deploy on.\n";
    return 0;
}
