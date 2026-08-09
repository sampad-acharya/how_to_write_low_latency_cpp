#include "bench_util.h"
#include <numeric>
#include <random>

// TLB pressure: the miss that is not a cache miss.
//
// Virtual addresses are translated to physical by walking page tables, cached
// in the TLB (a few hundred to a few thousand entries). The trick to isolate
// the TLB from the data cache: touch exactly ONE cache line per page, and
// pointer-chase between pages in random order.
//
//   - Data-cache footprint = (#pages) x 64 B  -> stays small, fits L2 easily.
//   - TLB footprint         = (#pages) entries -> grows without bound.
//
// So when per-hop latency jumps as the page count grows, it is the page-table
// walk, NOT a data cache miss. We hold one line per 4 KB slot and grow the
// number of slots. Cross-arch reveal: x86 has 4 KB pages, so every slot is a
// new page and the TLB cliff comes early; Apple Silicon has 16 KB pages, so
// four slots share a page and the TLB reaches ~4x further before it thrashes.

using namespace bench;

constexpr size_t SLOT   = 4096;         // one touched line per 4 KB slot
constexpr size_t HOPS   = 3'000'000;    // dependent hops per measurement
constexpr int    REPS   = 7;

// Lay out a random permutation cycle across `slots` slots; each slot stores the
// byte offset of the next slot to visit. Pointer-chasing it defeats prefetch
// and serializes one translation per hop.
static Stats chase(std::vector<uint8_t>& mem, size_t slots) {
    std::vector<size_t> order(slots);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), std::mt19937_64(slots * 2654435761u));
    for (size_t i = 0; i < slots; ++i) {
        size_t here = order[i] * SLOT;
        size_t next = order[(i + 1) % slots] * SLOT;
        *reinterpret_cast<size_t*>(&mem[here]) = next;   // first-touch this line
    }

    std::vector<long long> lat;
    lat.reserve(REPS);
    volatile size_t sink = 0;
    for (int r = 0; r < REPS; ++r) {
        size_t cur = 0;
        auto t0 = Clock::now();
        for (size_t k = 0; k < HOPS; ++k)
            cur = *reinterpret_cast<size_t*>(&mem[cur]);   // dependent load
        auto t1 = Clock::now();
        sink = cur;
        lat.push_back(std::chrono::duration_cast<ns>(t1 - t0).count() / (long long)HOPS);
    }
    do_not_optimize(sink);
    return compute_stats(lat);
}

int main() {
    banner("TLB pressure (page-walk cost)");
    std::cout << "One cache line per " << SLOT << "-byte slot, random pointer-chase, "
              << "per-hop ns.\n";
    std::cout << "Data footprint stays tiny (one line/slot); only the PAGE count grows,\n"
              << "so the jump is the TLB, not the data cache.\n\n";

    // Footprints in slots. Max 65536 slots x 4 KB = 256 MB of address space.
    size_t footprints[] = {64, 256, 1024, 4096, 16384, 65536};
    size_t max_slots = footprints[sizeof(footprints)/sizeof(footprints[0]) - 1];
    std::vector<uint8_t> mem(max_slots * SLOT);

    print_header();
    for (size_t P : footprints) {
        Stats s = chase(mem, P);
        double mb = double(P) * SLOT / (1024.0 * 1024.0);
        std::string label = std::to_string(P) + " slots (" +
                            std::to_string(int(mb)) + " MB span)";
        print_row(label, s);
    }
    std::cout << "\nPer-hop latency climbs in steps as the working set outgrows first the\n"
              << "L1 TLB, then the L2 TLB. On x86 (4 KB pages) the cliff arrives at fewer\n"
              << "slots than on Apple Silicon (16 KB pages, 4 slots/page -> 4x TLB reach).\n"
              << "Same code; the cliff location is a hardware property. Huge/2 MB pages\n"
              << "move it further still -- one reason low-latency systems reserve them.\n";
    return 0;
}
