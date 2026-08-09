#include "bench_util.h"
#include <cstddef>

// Struct layout, padding, and alignment as a latency contract.
//
// The compiler inserts padding to satisfy each member's alignment. Order your
// members carelessly and a struct that "should" be 24 bytes becomes 40 -- you
// just made every cache line hold fewer objects and every traversal touch more
// memory. Reorder members big-to-small and the padding collapses.
//
// This prints the sizeofs (the alignment lesson is visible without running a
// clock) and then measures a hot sum over an array of each layout, so the
// footprint difference shows up as real traversal latency.

using namespace bench;

// Bad: alternating sizes force padding after each small member.
struct Bad {
    char     a;      // 1  (+7 pad)
    double   x;      // 8
    char     b;      // 1  (+3 pad)
    int      i;      // 4
    char     c;      // 1  (+7 pad)
    double   y;      // 8
};

// Good: same fields, ordered large-to-small. Padding nearly vanishes.
struct Good {
    double   x;      // 8
    double   y;      // 8
    int      i;      // 4
    char     a;      // 1
    char     b;      // 1
    char     c;      // 1  (+1 pad)
};

constexpr int N    = 4'000'000;
constexpr int REPS = 30;

template <class T>
Stats sweep(std::vector<T>& v, double (*get)(const T&)) {
    std::vector<long long> lat; lat.reserve(REPS);
    volatile double sink = 0;
    for (int r = 0; r < REPS; ++r) {
        auto t0 = Clock::now();
        double s = 0;
        for (const auto& e : v) s += get(e);
        auto t1 = Clock::now();
        sink = s;
        lat.push_back(std::chrono::duration_cast<ns>(t1 - t0).count());
    }
    do_not_optimize(sink);
    return compute_stats(lat);
}

int main() {
    banner("Struct layout / padding / alignment");
    std::cout << "sizeof(Bad)  = " << sizeof(Bad)  << " (alignof " << alignof(Bad)  << ")\n";
    std::cout << "sizeof(Good) = " << sizeof(Good) << " (alignof " << alignof(Good) << ")\n";
    std::cout << "same 6 fields, reordered large->small: "
              << (100.0 * (sizeof(Bad) - sizeof(Good)) / sizeof(Bad))
              << "% smaller, so more objects per cache line.\n";
    std::cout << "offsetof(Bad.y)=" << offsetof(Bad, y)
              << "  offsetof(Good.y)=" << offsetof(Good, y) << "\n\n";

    std::vector<Bad>  bad(N);
    std::vector<Good> good(N);
    for (int i = 0; i < N; ++i) {
        bad[i]  = {char(i), double(i), char(i), i, char(i), double(i * 2)};
        good[i] = {double(i), double(i * 2), i, char(i), char(i), char(i)};
    }

    std::cout << "Hot sum of x+y over " << N << " objects, " << REPS << " reps (ns/pass):\n";
    print_header();
    Stats sg = sweep<Good>(good, [](const Good& e){ return e.x + e.y; });
    Stats sb = sweep<Bad> (bad,  [](const Bad&  e){ return e.x + e.y; });
    print_row("Good layout", sg);
    print_row("Bad layout",  sb);
    std::cout << "\nThe Bad array is larger for identical data, so the same traversal\n"
              << "streams more bytes and misses cache more often. Padding is not free.\n"
              << "Note: exact sizeof/alignof can differ on ARM64 vs x86-64 -- run both.\n";
    return 0;
}
