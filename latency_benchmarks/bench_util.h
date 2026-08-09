#pragma once
// Shared plumbing for the low-latency benchmark suite.
//
// Every benchmark in this folder times work per-operation, sorts the samples,
// and reports percentiles -- because the whole point of the talk is that the
// mean hides the tail that users actually feel. This header holds the pieces
// that would otherwise be copy-pasted nine times: a percentile Stats struct,
// a portable CPU-relax hint, the platform cache-line size, and an arch label.
//
// Portable across x86-64 and AArch64. No intrinsics headers required.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace bench {

using Clock = std::chrono::steady_clock;
using ns    = std::chrono::nanoseconds;

// ---- platform facts --------------------------------------------------------
#if defined(__cpp_lib_hardware_interference_size)
#  include <new>
inline constexpr std::size_t HW_LINE = std::hardware_destructive_interference_size;
#elif defined(__APPLE__) && defined(__aarch64__)
inline constexpr std::size_t HW_LINE = 128;   // Apple M-series: 128-byte lines
#else
inline constexpr std::size_t HW_LINE = 64;    // x86-64 and most ARM
#endif

inline const char* arch_name() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    return "x86 (TSO / strong memory model)";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "ARM64 (weak memory model)";
#elif defined(__arm__) || defined(_M_ARM)
    return "ARM32 (weak memory model)";
#else
    return "unknown architecture";
#endif
}

// A hint to the core that we are in a busy-wait: lets it save power and yield
// pipeline resources to a sibling hyperthread. x86 PAUSE, ARM YIELD.
inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

// ---- percentile statistics -------------------------------------------------
struct Stats {
    double mean, p50, p90, p99, p999, max;
    std::size_t n;
};

inline Stats compute_stats(std::vector<long long>& s) {
    if (s.empty()) return {0, 0, 0, 0, 0, 0, 0};
    double total = std::accumulate(s.begin(), s.end(), 0.0);
    std::sort(s.begin(), s.end());
    auto pct = [&](double p) {
        return double(s[std::size_t(p * (s.size() - 1))]);
    };
    return {total / s.size(), pct(0.50), pct(0.90), pct(0.99), pct(0.999),
            double(s.back()), s.size()};
}

inline void print_header(const std::string& unit = "ns") {
    std::cout << "  " << std::left << std::setw(22) << "variant" << std::right
              << std::setw(13) << ("mean(" + unit + ")")
              << std::setw(13) << "p50"
              << std::setw(13) << "p90"
              << std::setw(13) << "p99"
              << std::setw(13) << "p999"
              << std::setw(13) << "max" << "\n"
              << "  " << std::string(22 + 13 * 6, '-') << "\n";
}

inline void print_row(const std::string& name, const Stats& s) {
    std::cout << "  " << std::left << std::setw(22) << name << std::right
              << std::fixed << std::setprecision(0)
              << std::setw(13) << s.mean
              << std::setw(13) << s.p50
              << std::setw(13) << s.p90
              << std::setw(13) << s.p99
              << std::setw(13) << s.p999
              << std::setw(13) << s.max
              << std::defaultfloat << "\n";
}

// Force the compiler to treat `v` as observed, so it cannot delete the work
// that produced it. The workhorse behind benchmark_lies.cpp; used everywhere.
template <class T>
inline void do_not_optimize(T const& v) {
    asm volatile("" : : "r,m"(v) : "memory");
}

inline void banner(const std::string& title) {
    std::cout << "=== " << title << " ===\n"
              << "arch: " << arch_name()
              << "   (assumed cache line: " << HW_LINE << " B)\n\n";
}

} // namespace bench
