#include "bench_util.h"
#include <memory_resource>

// Custom allocators as a latency contract.
//
// Every `new` is a call into a general-purpose allocator: it takes a lock,
// walks free lists, and occasionally asks the OS for more memory -- and that
// last case is a rare, unschedulable spike in your tail. An arena (bump)
// allocator does none of it: allocation is a pointer add, and you free the
// whole region at once. You trade generality for a flat, predictable tail.
//
// We allocate + touch many small nodes two ways and compare per-op latency
// percentiles. std::pmr::monotonic_buffer_resource is the arena; the default
// resource is plain new/delete. Same interface, very different p999.

using namespace bench;

constexpr int NODES = 200'000;
constexpr int REPS  = 20;

struct Node {                       // 64 bytes: a plausible small graph/list node
    uint64_t key;
    uint64_t payload[6];
    Node*    next;
};

// Allocate NODES nodes from `mr`, linking + touching each so nothing is elided.
// Returns per-allocation latency samples.
static std::vector<long long> alloc_run(std::pmr::memory_resource* mr) {
    std::pmr::polymorphic_allocator<Node> alloc(mr);
    std::vector<long long> lat;
    lat.reserve(NODES);
    Node* head = nullptr;
    for (int i = 0; i < NODES; ++i) {
        auto t0 = Clock::now();
        Node* n = alloc.allocate(1);
        auto t1 = Clock::now();
        n->key = i; n->payload[0] = uint64_t(i); n->next = head; head = n;
        lat.push_back(std::chrono::duration_cast<ns>(t1 - t0).count());
    }
    do_not_optimize(head);
    // Free (arena frees in bulk at scope exit; new/delete must walk the list).
    while (head) { Node* nx = head->next; alloc.deallocate(head, 1); head = nx; }
    return lat;
}

static Stats aggregate(std::pmr::memory_resource* mr, bool arena) {
    std::vector<long long> all;
    std::vector<std::byte> buffer;
    for (int r = 0; r < REPS; ++r) {
        if (arena) {
            buffer.assign(std::size_t(NODES) * sizeof(Node) + 4096, std::byte{});
            std::pmr::monotonic_buffer_resource pool(buffer.data(), buffer.size());
            auto v = alloc_run(&pool);
            all.insert(all.end(), v.begin(), v.end());
        } else {
            auto v = alloc_run(mr);
            all.insert(all.end(), v.begin(), v.end());
        }
    }
    return compute_stats(all);
}

int main() {
    banner("Arena allocator vs new/delete");
    std::cout << NODES << " small (" << sizeof(Node) << "-byte) nodes x " << REPS
              << " reps. Per-allocation latency.\n\n";
    print_header();
    Stats heap  = aggregate(std::pmr::new_delete_resource(), false);
    Stats arena = aggregate(nullptr, true);
    print_row("new/delete", heap);
    print_row("arena (monotonic)", arena);
    std::cout << "\nThe mean gap is real, but the story is the TAIL: new/delete's p999\n"
              << "carries the occasional free-list walk / OS mmap. The arena's p999 is\n"
              << "flat because allocation is just a pointer bump. Predictable by design.\n";
    return 0;
}
