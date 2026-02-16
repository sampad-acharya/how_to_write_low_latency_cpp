#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using Clock = std::chrono::steady_clock;
using ns    = std::chrono::nanoseconds;

struct BigObject {
    // 1 KB object to amplify reallocation cost
    char data[1024];
};

struct Stats {
    double mean, p50, p90, p99, p999;
};

Stats compute_stats(std::vector<long long>& v) {
    std::sort(v.begin(), v.end());
    auto pct = [&](double p) {
        size_t idx = static_cast<size_t>(p * (v.size() - 1));
        return static_cast<double>(v[idx]);
    };
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    return {mean, pct(0.50), pct(0.90), pct(0.99), pct(0.999)};
}

void print_stats(const std::string& name, const Stats& s) {
    std::cout << name << " (ns):\n"
              << "  mean : " << s.mean  << "\n"
              << "  p50  : " << s.p50   << "\n"
              << "  p90  : " << s.p90   << "\n"
              << "  p99  : " << s.p99   << "\n"
              << "  p999 : " << s.p999  << "\n\n";
}

int main() {
    const int N = 20000;

    std::vector<BigObject> predictable;
    predictable.reserve(N);

    std::vector<BigObject> jittery; // no reserve()

    std::vector<long long> predictable_lat;
    std::vector<long long> jittery_lat;
    predictable_lat.reserve(N);
    jittery_lat.reserve(N);

    // Warmup
    for (int i = 0; i < 2000; ++i) {
        predictable.push_back(BigObject{});
        jittery.push_back(BigObject{});
    }
    predictable.clear();
    jittery.clear();

    // predictable path
    for (int i = 0; i < N; ++i) {
        auto start = Clock::now();
        predictable.push_back(BigObject{});
        auto end = Clock::now();
        predictable_lat.push_back(std::chrono::duration_cast<ns>(end - start).count());
    }

    // jittery path
    for (int i = 0; i < N; ++i) {
        auto start = Clock::now();
        jittery.push_back(BigObject{});
        auto end = Clock::now();
        jittery_lat.push_back(std::chrono::duration_cast<ns>(end - start).count());
    }

    Stats s_pred = compute_stats(predictable_lat);
    Stats s_jit  = compute_stats(jittery_lat);

    print_stats("Predictable (reserved)", s_pred);
    print_stats("Jittery (dynamic reallocation)", s_jit);

    std::cout << "Dynamic reallocation introduces rare but massive latency spikes.\n";
}
