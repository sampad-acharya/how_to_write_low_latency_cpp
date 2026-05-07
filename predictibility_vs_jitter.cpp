#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <numeric>
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
double do_work(int iterations) {
    double x = 0.0;
    for (int i = 0; i < iterations; ++i) {
        x += std::sin(i * 0.001);
    }
    return x;
}

int main() {
    const int N = 1000000;

    // predictable always does this much work
    const int predictable_iters = 120;

    // jittery usually does this much work
    const int jitter_base_iters = 100;

    // rare spike adds this much work
    const int jitter_extra_iters = 1000;

    std::vector<long long> predictable_lat;
    std::vector<long long> jittery_lat;
    predictable_lat.reserve(N);
    jittery_lat.reserve(N);

    // -------------------------------
    // Load jitter pattern from file
    // -------------------------------
    // Loading at runtime hides the pattern from the compiler so it cannot
    // hoist, predict, or constant-fold the spike branch.
    std::vector<uint8_t> spike;
    {
        const char* path = "spike_vector.bin";
        FILE* f = std::fopen(path, "rb");
        if (!f) {
            std::cerr << "Failed to open " << path
                      << ". Run ./generate_spike_vector first.\n";
            return 1;
        }
        size_t count = 0;
        if (std::fread(&count, sizeof(count), 1, f) != 1) {
            std::cerr << "Failed to read header from " << path << "\n";
            std::fclose(f);
            return 1;
        }
        spike.resize(count);
        if (std::fread(spike.data(), 1, count, f) != count) {
            std::cerr << "Failed to read spike data from " << path << "\n";
            std::fclose(f);
            return 1;
        }
        std::fclose(f);

        if (spike.size() != static_cast<size_t>(N)) {
            std::cerr << "spike_vector.bin has " << spike.size()
                      << " entries; expected " << N
                      << ". Regenerate with: ./generate_spike_vector " << N << "\n";
            return 1;
        }
    }

    // Sinks: accumulate do_work's return value so the compiler must actually
    // run the loops. Printed at the end.
    double sink_warmup = 0.0;
    double sink_pred   = 0.0;
    double sink_jit    = 0.0;

    // Warmup

    // Measure jittery system first so its loop absorbs the residual warmup
    // (CPU frequency ramp, branch predictor settle, vDSO icache fill).

    // Predictable then runs in steady state for a cleaner per-op comparison.
    for (int i = 0; i < N; ++i) {
        int iters = jitter_base_iters;
        if (spike[i]) {
            iters += jitter_extra_iters;
        }

        auto start = Clock::now();
        double r = do_work(iters);
        auto end = Clock::now();
        sink_jit += r;
        jittery_lat.push_back
        (std::chrono::duration_cast<ns>(end - start).count());
    }

    // Measure predictable system
    for (int i = 0; i < N; ++i) {
        auto start = Clock::now();
        double r = do_work(predictable_iters);
        auto end = Clock::now();
        sink_pred += r;
        predictable_lat.push_back
        (std::chrono::duration_cast<ns>(end - start).count());
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


    // Print sinks so the compiler cannot discard do_work.
    std::cout << " pred=" << sink_pred
              << " jit="  << sink_jit << "\n";

    return 0;
}
