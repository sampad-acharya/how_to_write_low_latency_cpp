#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
using ns    = std::chrono::nanoseconds;

struct Stats {
    double mean;
    double p50;
    double p90;
    double p99;
    double p999;
};

Stats compute_stats(std::vector<long long>& samples) {
    std::sort(samples.begin(), samples.end());
    auto get_pct = [&](double p) {
        size_t idx = static_cast<size_t>(p * (samples.size() - 1));
        return static_cast<double>(samples[idx]);
    };
    double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    return Stats{
        mean,
        get_pct(0.50),
        get_pct(0.90),
        get_pct(0.99),
        get_pct(0.999)
    };
}

// CPU-bound work
void do_work(int iterations) {
    volatile double x = 0.0;
    for (int i = 0; i < iterations; ++i) {
        x += std::sin(i * 0.001);
    }
}

int main() {
    const int N = 100000;

    // predictable always does this much work
    const int predictable_iters = 120;

    // jittery usually does this much work
    const int jitter_base_iters = 100;

    // rare spike adds this much work
    const int jitter_extra_iters = 5000;

    std::vector<long long> predictable_lat;
    std::vector<long long> jittery_lat;
    predictable_lat.reserve(N);
    jittery_lat.reserve(N);

    // -------------------------------
    // Pre-generate jitter pattern
    // -------------------------------
    std::vector<bool> spike(N);

    std::mt19937_64 rng(12345);  // deterministic seed
    std::bernoulli_distribution spike_dist(0.01); // 1% chance

    for (int i = 0; i < N; ++i) {
        spike[i] = spike_dist(rng);
    }

    // Warmup
    for (int i = 0; i < 1000; ++i) {
        do_work(predictable_iters);
        do_work(jitter_base_iters + (spike[i % N] ? jitter_extra_iters : 0));
    }

    // Measure predictable system
    for (int i = 0; i < N; ++i) {
        auto start = Clock::now();
        do_work(predictable_iters);
        auto end = Clock::now();
        predictable_lat.push_back(std::chrono::duration_cast<ns>(end - start).count());
    }

    // Measure jittery system
    for (int i = 0; i < N; ++i) {
        int iters = jitter_base_iters;
        if (spike[i]) {
            iters += jitter_extra_iters;
        }

        auto start = Clock::now();
        do_work(iters);
        auto end = Clock::now();
        jittery_lat.push_back(std::chrono::duration_cast<ns>(end - start).count());
    }

    Stats s_pred = compute_stats(predictable_lat);
    Stats s_jit  = compute_stats(jittery_lat);

    auto print_stats = [](const std::string& name, const Stats& s) {
        std::cout << name << " (ns):\n"
                  << "  mean : " << s.mean  << "\n"
                  << "  p50  : " << s.p50   << "\n"
                  << "  p90  : " << s.p90   << "\n"
                  << "  p99  : " << s.p99   << "\n"
                  << "  p999 : " << s.p999  << "\n\n";
    };

    print_stats("Predictable system", s_pred);
    print_stats("Jittery system", s_jit);

    std::cout << "Note: Expect jittery to have LOWER mean but MUCH HIGHER p99/p999.\n";
    std::cout << "This shows: a ~10% faster average can still lose in tail latency.\n";

    return 0;
}
