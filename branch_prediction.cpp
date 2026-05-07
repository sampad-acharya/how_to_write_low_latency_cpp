#include <algorithm>
#include <chrono>
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
    return Stats{mean, get_pct(0.50), get_pct(0.90), get_pct(0.99), get_pct(0.999)};
}

// The kernel under test. noinline so the call site can't specialize on the
// data. The asm volatile barrier inside the if-body has an unspecified side
// effect on `sum`, which prevents the compiler from:
//   (a) auto-vectorizing the loop into a branchless SIMD masked-add, or
//   (b) if-converting the branch into a scalar csel.
// Without this, clang at -O2 turns the loop into branchless code and the
// branch predictor never gets exercised.
__attribute__((noinline))
uint64_t conditional_sum(const uint32_t* data, size_t n) {
    uint64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        if (data[i] >= 128) {
            sum += data[i];
            asm volatile("" : "+r"(sum));
        }
    }
    return sum;
}

static std::vector<uint32_t> load_data(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::cerr << "Failed to open " << path
                  << ". Run ./generate_branch_data first.\n";
        std::exit(1);
    }
    size_t count = 0;
    if (std::fread(&count, sizeof(count), 1, f) != 1) {
        std::cerr << "Failed to read header from " << path << "\n";
        std::exit(1);
    }
    std::vector<uint32_t> v(count);
    if (std::fread(v.data(), sizeof(uint32_t), count, f) != count) {
        std::cerr << "Failed to read data from " << path << "\n";
        std::exit(1);
    }
    std::fclose(f);
    return v;
}

int main() {
    const int TRIALS = 2000;

    auto unsorted = load_data("branch_data.bin");
    auto sorted   = unsorted;
    std::sort(sorted.begin(), sorted.end());

    const size_t N = unsorted.size();
    std::cout << "Array: " << N << " uint32_t values, "
              << TRIALS << " trials per case\n";
    std::cout << "Each trial: one full pass summing values where v >= 128\n\n";

    std::vector<long long> unsorted_lat;
    std::vector<long long> sorted_lat;
    unsorted_lat.reserve(TRIALS);
    sorted_lat.reserve(TRIALS);

    uint64_t sink_unsorted = 0;
    uint64_t sink_sorted   = 0;

    // Run unsorted first so it absorbs residual warmup (CPU frequency ramp,
    // vDSO icache fill, predictor's first-touch state). Sorted then runs in
    // a fully steady state, so the gap we measure is real branch-prediction
    // cost, not warmup contamination.
    for (int t = 0; t < TRIALS; ++t) {
        auto start = Clock::now();
        uint64_t s = conditional_sum(unsorted.data(), N);
        auto end = Clock::now();
        sink_unsorted += s;
        unsorted_lat.push_back(std::chrono::duration_cast<ns>(end - start).count());
    }

    for (int t = 0; t < TRIALS; ++t) {
        auto start = Clock::now();
        uint64_t s = conditional_sum(sorted.data(), N);
        auto end = Clock::now();
        sink_sorted += s;
        sorted_lat.push_back(std::chrono::duration_cast<ns>(end - start).count());
    }

    Stats s_uns = compute_stats(unsorted_lat);
    Stats s_srt = compute_stats(sorted_lat);

    auto print_stats = [](const std::string& name, const Stats& s) {
        std::cout << name << " (ns per pass):\n"
                  << "  mean : " << s.mean  << "\n"
                  << "  p50  : " << s.p50   << "\n"
                  << "  p90  : " << s.p90   << "\n"
                  << "  p99  : " << s.p99   << "\n"
                  << "  p999 : " << s.p999  << "\n\n";
    };

    print_stats("Unsorted (unpredictable branch)", s_uns);
    print_stats("Sorted   (predictable branch)",   s_srt);

    if (s_srt.p50 > 0) {
        std::cout << "Sorted is " << (s_uns.p50 / s_srt.p50)
                  << "x faster at p50 — the only difference is branch predictability.\n";
    }

    std::cout << "[sinks] unsorted=" << sink_unsorted
              << " sorted=" << sink_sorted
              << " (must be equal — same values, just reordered)\n";
    return 0;
}
