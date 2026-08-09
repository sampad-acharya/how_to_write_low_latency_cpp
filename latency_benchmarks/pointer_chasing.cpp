#include "bench_util.h"
#include <numeric>
#include <random>

// Pointer chasing: why std::list is a latency trap.
//
// A std::vector is contiguous: the hardware prefetcher sees the linear stream
// and has the next line waiting before you ask. A std::list (or any node graph)
// scatters nodes across the heap; each `->next` is a dependent load the CPU
// cannot predict, so it stalls the full memory latency on every hop. Same N,
// same sum, but one is bandwidth-bound and the other is latency-bound.
//
// This is the mechanism under "AoS vs SoA" and "the wrong container": the cost
// is not big-O, it is the cache-miss-per-element that pointer indirection buys.
// We compare a vector walk against a linked walk whose nodes are deliberately
// shuffled in memory (the realistic post-allocation state).

using namespace bench;

constexpr int N    = 4'000'000;
constexpr int REPS = 30;

struct Node { uint64_t val; Node* next; };

int main() {
    banner("Pointer chasing: contiguous vs linked");
    std::cout << N << " elements, sum traversal, " << REPS << " reps (ns/pass).\n\n";

    // Contiguous array.
    std::vector<uint64_t> vec(N);
    std::iota(vec.begin(), vec.end(), 1);

    // Linked list over a shuffled index order, so next-pointers jump around the
    // heap the way real long-lived lists do.
    std::vector<Node> nodes(N);
    std::vector<int>  order(N);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), std::mt19937_64(99));
    for (int i = 0; i < N; ++i) {
        nodes[order[i]].val  = uint64_t(order[i] + 1);
        nodes[order[i]].next = (i + 1 < N) ? &nodes[order[i + 1]] : nullptr;
    }
    Node* head = &nodes[order[0]];

    volatile uint64_t sink = 0;
    std::vector<long long> vlat, llat;

    for (int r = 0; r < REPS; ++r) {
        auto t0 = Clock::now();
        uint64_t s = 0; for (uint64_t x : vec) s += x;
        auto t1 = Clock::now();
        sink = s; vlat.push_back(std::chrono::duration_cast<ns>(t1 - t0).count());
    }
    for (int r = 0; r < REPS; ++r) {
        auto t0 = Clock::now();
        uint64_t s = 0; for (Node* p = head; p; p = p->next) s += p->val;
        auto t1 = Clock::now();
        sink = s; llat.push_back(std::chrono::duration_cast<ns>(t1 - t0).count());
    }
    do_not_optimize(sink);

    Stats v = compute_stats(vlat);
    Stats l = compute_stats(llat);
    print_header();
    print_row("vector (contiguous)", v);
    print_row("list (pointer chase)", l);
    std::cout << "\nper-element: vector " << (v.p50 / N) << " ns   list " << (l.p50 / N)
              << " ns  (" << (double(l.p50) / v.p50) << "x)\n";
    std::cout << "Identical work and complexity. The list pays one unpredictable,\n"
              << "dependent cache miss per node; the vector streams and prefetches.\n"
              << "This is the real cost behind 'pick the right container'.\n";
    return 0;
}
