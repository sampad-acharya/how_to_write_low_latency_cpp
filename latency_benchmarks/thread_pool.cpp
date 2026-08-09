#include "bench_util.h"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>

// Fixed thread pool vs a thread per task.
//
// std::thread creation is a syscall (clone/pthread_create): a fresh kernel
// stack, a scheduler enqueue -- microseconds, with a fat tail when the machine
// is busy. A pool pays that ONCE at startup; afterwards a task is a queue push
// and a condvar wake.
//
// The fair way to measure the DISPATCH primitive is one task at a time: hand a
// tiny unit of work to another thread and wait for it to finish. Serialized, so
// no queue backlog muddies the number -- we are comparing "spawn+run+join" to
// "submit+run+signal". (Flooding a small pool with thousands of tasks measures
// your queue depth, not dispatch cost; that is a different graph.)
//
// Scheduler-bound, so the gap is set by the OS more than the ISA. Run on both.

using namespace bench;

constexpr int DISPATCHES = 20'000;

class Pool {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> q;
    std::mutex m; std::condition_variable cv;
    bool stop = false;
public:
    explicit Pool(int n) {
        for (int i = 0; i < n; ++i) workers.emplace_back([this] {
            for (;;) {
                std::function<void()> job;
                { std::unique_lock<std::mutex> lk(m);
                  cv.wait(lk, [this]{ return stop || !q.empty(); });
                  if (stop && q.empty()) return;
                  job = std::move(q.front()); q.pop(); }
                job();
            }
        });
    }
    void submit(std::function<void()> f) {
        { std::lock_guard<std::mutex> lk(m); q.push(std::move(f)); }
        cv.notify_one();
    }
    ~Pool() {
        { std::lock_guard<std::mutex> lk(m); stop = true; } cv.notify_all();
        for (auto& w : workers) w.join();
    }
};

static inline uint64_t tiny_work(uint64_t x) { return x * 2654435761u ^ (x >> 15); }

int main() {
    banner("Fixed thread pool vs thread-per-task");
    int n = std::min<int>(std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 4, 8);
    std::cout << DISPATCHES << " dispatches, ONE outstanding at a time (pure dispatch cost).\n";
    std::cout << "Per-dispatch latency: hand off a tiny task and wait for it, ns.\n\n";

    std::atomic<uint64_t> sink{0};

    // --- thread-per-task: spawn a thread, it runs, join waits for it ---
    std::vector<long long> spawn_lat; spawn_lat.reserve(DISPATCHES);
    for (int i = 0; i < DISPATCHES; ++i) {
        auto t0 = Clock::now();
        std::thread th([&, i] { sink.fetch_add(tiny_work(i), std::memory_order_relaxed); });
        th.join();
        spawn_lat.push_back(std::chrono::duration_cast<ns>(Clock::now() - t0).count());
    }

    // --- fixed pool: submit one task, wait for its completion signal ---
    std::vector<long long> pool_lat; pool_lat.reserve(DISPATCHES);
    {
        Pool pool(n);
        for (int i = 0; i < DISPATCHES; ++i) {
            std::atomic<int> done{0};
            auto t0 = Clock::now();
            pool.submit([&, i] {
                sink.fetch_add(tiny_work(i), std::memory_order_relaxed);
                done.store(1, std::memory_order_release);
                done.notify_one();
            });
            done.wait(0, std::memory_order_acquire);   // C++20 atomic wait
            pool_lat.push_back(std::chrono::duration_cast<ns>(Clock::now() - t0).count());
        }
    }

    Stats sp = compute_stats(spawn_lat);
    Stats pl = compute_stats(pool_lat);
    print_header();
    print_row("thread-per-task", sp);
    print_row("fixed pool", pl);
    std::cout << "\nper-dispatch median: spawn " << sp.p50 << " ns  vs  pool " << pl.p50
              << " ns  (" << std::fixed << std::setprecision(1)
              << (double(sp.p50) / pl.p50) << "x)\n" << std::defaultfloat;
    std::cout << "Thread creation dominates the spawn path and drags its tail; the warm\n"
              << "pool just wakes a sleeping worker. In a hot loop, never spawn per task.\n";
    do_not_optimize(sink.load());
    return 0;
}
