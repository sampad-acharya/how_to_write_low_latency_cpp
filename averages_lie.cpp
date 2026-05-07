#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
using ns    = std::chrono::nanoseconds;

// "Averages lie" demonstration.
//
// Two systems built so their MEAN per-op latency matches within a couple of
// percent, but their tail distributions differ by an order of magnitude:
//
//   Smooth: every op does the same amount of work.
//   Bursty: 95% of ops do LESS work than Smooth, 5% do much more.
//
// The work levels are tuned so:
//   mean(Bursty) = 0.95 * t(base) + 0.05 * t(spike)
//                ≈ t(smooth_iters)
//
// Reading both means side-by-side, the systems look equivalent.
// Reading p99 / p999, Bursty is ~10x worse — those are the operations a
// real user actually feels.

struct Stats {
    double mean;
    double p50;
    double p90;
    double p99;
    double p999;
    double p9999;
    double max;
};

Stats compute_stats(std::vector<long long>& s) {
    std::sort(s.begin(), s.end());
    auto pct = [&](double p) {
        size_t idx = static_cast<size_t>(p * (s.size() - 1));
        return static_cast<double>(s[idx]);
    };
    double mean = std::accumulate(s.begin(), s.end(), 0.0) / s.size();
    return Stats{
        mean,
        pct(0.50),
        pct(0.90),
        pct(0.99),
        pct(0.999),
        pct(0.9999),
        static_cast<double>(s.back())
    };
}

// Returning x prevents the optimizer from deleting the loop.
double do_work(int iterations) {
    double x = 0.0;
    for (int i = 0; i < iterations; ++i) {
        x = x + std::sin(i * 0.001);
    }
    return x;
}

int main() {
    const int N = 1000000;

    // Tuned so mean(bursty) ≈ mean(smooth):
    //   0.95 * 50 + 0.05 * 1050 = 47.5 + 52.5 = 100  → matches smooth_iters.
    const int    smooth_iters = 100;
    const int    bursty_base  = 50;
    const int    bursty_spike = 1050;
    const double spike_p      = 0.05;

    // Build the spike pattern at runtime so the compiler can't fold the
    // branch. (mt19937 evaluation is opaque at -O2 — no need for a file.)
    std::mt19937_64 rng(99999);
    std::bernoulli_distribution sd(spike_p);
    std::vector<uint8_t> spike(N);
    size_t spikes = 0;
    for (int i = 0; i < N; ++i) {
        spike[i] = sd(rng) ? 1 : 0;
        spikes += spike[i];
    }
    std::cout << "Spike rate: " << (100.0 * spikes / N) << "% ("
              << spikes << " of " << N << ")\n";
    std::cout << "Smooth: do_work(" << smooth_iters << ") every op\n";
    std::cout << "Bursty: do_work(" << bursty_base  << ") on " << (1.0 - spike_p) * 100
              << "% of ops, do_work(" << bursty_spike << ") on the rest\n\n";

    std::vector<long long> smooth_lat;
    std::vector<long long> bursty_lat;
    smooth_lat.reserve(N);
    bursty_lat.reserve(N);

    double sink_smooth = 0.0;
    double sink_bursty = 0.0;

    // Bursty runs first so its loop absorbs residual warmup (CPU frequency
    // ramp, vDSO icache fill). Smooth then runs in steady state.
    for (int i = 0; i < N; ++i) {
        int iters = spike[i] ? bursty_spike : bursty_base;
        auto start = Clock::now();
        double r = do_work(iters);
        auto end = Clock::now();
        sink_bursty += r;
        bursty_lat.push_back(std::chrono::duration_cast<ns>(end - start).count());
    }

    for (int i = 0; i < N; ++i) {
        auto start = Clock::now();
        double r = do_work(smooth_iters);
        auto end = Clock::now();
        sink_smooth += r;
        smooth_lat.push_back(std::chrono::duration_cast<ns>(end - start).count());
    }

    Stats sm = compute_stats(smooth_lat);
    Stats bu = compute_stats(bursty_lat);

    auto print_stats = [](const std::string& name, const Stats& s) {
        std::cout << name << " (ns):\n"
                  << "  mean  : " << s.mean   << "\n"
                  << "  p50   : " << s.p50    << "\n"
                  << "  p90   : " << s.p90    << "\n"
                  << "  p99   : " << s.p99    << "\n"
                  << "  p999  : " << s.p999   << "\n"
                  << "  p9999 : " << s.p9999  << "\n"
                  << "  max   : " << s.max    << "\n\n";
    };

    print_stats("Smooth", sm);
    print_stats("Bursty", bu);

    std::cout << "=== Verdict ===\n";
    std::cout << "By mean : bursty/smooth = " << (bu.mean / sm.mean)
              << "x  ← looks identical\n";
    std::cout << "By p99  : bursty/smooth = " << (bu.p99  / sm.p99)
              << "x  ← bursty is much worse\n";
    std::cout << "By p999 : bursty/smooth = " << (bu.p999 / sm.p999)
              << "x  ← same story\n";
    std::cout << "By p9999: bursty/smooth = " << (bu.p9999 / sm.p9999)
              << "x\n\n";
    std::cout << "Provisioning to the mean: 5% of requests blow your SLA.\n";
    std::cout << "Provisioning to p99: you actually serve users.\n";

    std::cout << "\n[sinks] smooth=" << sink_smooth
              << " bursty=" << sink_bursty << "\n";
    return 0;
}
