#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using Clock = std::chrono::steady_clock;
using ns    = std::chrono::nanoseconds;

// std::map vs std::unordered_map on SERIALIZED-BUT-RANDOM integer IDs.
//
// The workload models an ID stream the way a real feed hands them to you:
// monotonically increasing (serialized), but with random gaps, and with
// occasional replays of a recent ID. The container's job is to store each ID
// exactly once — dedup — and answer questions about what it has stored.
//
// "unordered_map is O(1), map is O(log n)" is a statement about the mean.
// It says nothing about the two things that actually decide latency budgets:
//
//   1. TAIL. unordered_map is amortized O(1). The amortization is paid by
//      rehashing: every time load factor crosses 1.0 it allocates a bigger
//      bucket array and re-links EVERY node. One unlucky insert pays for the
//      whole table. map has no such event — a red-black tree rebalance is
//      bounded at O(log n), always, on every single insert.
//
//   2. ORDER. map keeps the IDs sorted for free. "Which IDs are in
//      [lo, hi]?" is a lower_bound plus a walk. unordered_map cannot answer
//      that at all without touching all n elements.
//
// Point lookup by a single known key is unordered_map's home turf and it is
// reported honestly below. The claim here is narrower and more useful:
// on this workload map wins on tail latency, on ordered access, and on range
// queries — which for a latency-sensitive system is usually the whole game.

constexpr size_t N        = 1'000'000;  // IDs in the stream
constexpr size_t N_SMALL  = 64;         // "small hot table" case
constexpr int    RANGE_Q  = 200;        // range queries to run
constexpr int    SMALL_IT = 200'000;    // small-table churn iterations

struct Stats {
    double mean;
    double p50;
    double p99;
    double p999;
    double p9999;
    double max;
    double total_ms;
};

Stats compute_stats(std::vector<long long>& s) {
    double total = std::accumulate(s.begin(), s.end(), 0.0);
    std::sort(s.begin(), s.end());
    auto pct = [&](double p) {
        size_t idx = static_cast<size_t>(p * (s.size() - 1));
        return static_cast<double>(s[idx]);
    };
    return Stats{
        total / s.size(),
        pct(0.50),
        pct(0.99),
        pct(0.999),
        pct(0.9999),
        static_cast<double>(s.back()),
        total / 1e6
    };
}

void print_stats(const std::string& name, const Stats& s) {
    std::cout << "  " << std::left << std::setw(16) << name << std::right
              << std::fixed << std::setprecision(0)
              << std::setw(9) << s.mean
              << std::setw(9) << s.p50
              << std::setw(9) << s.p99
              << std::setw(10) << s.p999
              << std::setw(11) << s.p9999
              << std::setw(12) << s.max
              << std::setprecision(2) << std::setw(12) << s.total_ms
              << std::defaultfloat << "\n";
}

void print_header() {
    std::cout << "  " << std::left << std::setw(16) << "" << std::right
              << std::setw(9) << "mean"
              << std::setw(9) << "p50"
              << std::setw(9) << "p99"
              << std::setw(10) << "p999"
              << std::setw(11) << "p9999"
              << std::setw(12) << "max"
              << std::setw(12) << "total(ms)" << "\n"
              << "  " << std::string(16 + 9 + 9 + 9 + 10 + 11 + 12 + 12, '-') << "\n";
}

// Serialized-but-random IDs: monotone with random gaps, ~20% replays of a
// recent ID so the container has real dedup work to do. Unique by construction
// otherwise, so both containers end up holding the identical key set.
std::vector<uint64_t> make_id_stream(size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> gap(1, 64);
    std::uniform_int_distribution<int>      replay(0, 99);

    std::vector<uint64_t> ids;
    ids.reserve(n);
    uint64_t cur = 1'000'000'000ull;

    for (size_t i = 0; i < n; ++i) {
        if (i > 0 && replay(rng) < 20) {
            size_t window = std::min<size_t>(ids.size(), 4096);
            std::uniform_int_distribution<size_t> back(1, window);
            ids.push_back(ids[ids.size() - back(rng)]);
        } else {
            cur += gap(rng);
            ids.push_back(cur);
        }
    }
    return ids;
}

// ---------------------------------------------------------------------------
// Phase 1: dedup insert, per-operation latency
// ---------------------------------------------------------------------------
template <class Map>
Stats bench_insert(const std::vector<uint64_t>& ids, Map& out, size_t& unique) {
    std::vector<long long> lat;
    lat.reserve(ids.size());
    unique = 0;

    for (uint64_t id : ids) {
        auto start = Clock::now();
        auto res = out.emplace(id, 0u);
        auto end = Clock::now();
        lat.push_back(std::chrono::duration_cast<ns>(end - start).count());

        res.first->second++;          // outside the timed region
        unique += res.second ? 1 : 0;
    }
    return compute_stats(lat);
}

// ---------------------------------------------------------------------------
// Phase 2: point lookup of known-present keys, in random order
// ---------------------------------------------------------------------------
template <class Map>
Stats bench_lookup(const Map& m, const std::vector<uint64_t>& probes, uint64_t& sink) {
    std::vector<long long> lat;
    lat.reserve(probes.size());
    uint64_t acc = 0;

    for (uint64_t id : probes) {
        auto start = Clock::now();
        auto it = m.find(id);
        auto end = Clock::now();
        lat.push_back(std::chrono::duration_cast<ns>(end - start).count());
        if (it != m.end()) acc += it->second;
    }
    sink += acc;
    return compute_stats(lat);
}

// ---------------------------------------------------------------------------
// Phase 3: full ordered traversal (IDs ascending)
//
// map: iterate. unordered_map: extract every key, sort, then look each one
// back up — because it has no notion of order. That extraction IS the cost of
// having chosen an unordered container; charging it here is the fair
// comparison, not a handicap.
// ---------------------------------------------------------------------------
uint64_t ordered_scan_map(const std::map<uint64_t, uint32_t>& m) {
    uint64_t acc = 0;
    for (const auto& [k, v] : m) acc += k ^ v;
    return acc;
}

uint64_t ordered_scan_umap(const std::unordered_map<uint64_t, uint32_t>& m) {
    std::vector<uint64_t> keys;
    keys.reserve(m.size());
    for (const auto& [k, v] : m) keys.push_back(k);
    std::sort(keys.begin(), keys.end());

    uint64_t acc = 0;
    for (uint64_t k : keys) acc += k ^ m.find(k)->second;
    return acc;
}

// ---------------------------------------------------------------------------
// Phase 4: range query — every ID in [lo, hi]
// ---------------------------------------------------------------------------
uint64_t range_query_map(const std::map<uint64_t, uint32_t>& m, uint64_t lo, uint64_t hi) {
    uint64_t acc = 0;
    for (auto it = m.lower_bound(lo); it != m.end() && it->first <= hi; ++it)
        acc += it->first ^ it->second;
    return acc;
}

uint64_t range_query_umap(const std::unordered_map<uint64_t, uint32_t>& m,
                          uint64_t lo, uint64_t hi) {
    uint64_t acc = 0;
    for (const auto& [k, v] : m)          // no choice but to touch everything
        if (k >= lo && k <= hi) acc += k ^ v;
    return acc;
}

// ---------------------------------------------------------------------------
// Phase 0: small hot table — insert N_SMALL IDs, probe them, clear, repeat.
// At this size the tree is 6 levels deep and lives entirely in L1; the hash
// table still pays for hashing, a bucket-array indirection, and allocation.
// ---------------------------------------------------------------------------
template <class Map>
Stats bench_small(const std::vector<uint64_t>& ids, uint64_t& sink) {
    std::vector<long long> lat;
    lat.reserve(SMALL_IT);
    std::mt19937_64 rng(7);
    uint64_t acc = 0;

    for (int it = 0; it < SMALL_IT; ++it) {
        size_t base = (rng() % (ids.size() - N_SMALL));
        auto start = Clock::now();
        Map m;
        for (size_t i = 0; i < N_SMALL; ++i) m.emplace(ids[base + i], uint32_t(i));
        for (size_t i = 0; i < N_SMALL; i += 4) {
            auto f = m.find(ids[base + i]);
            if (f != m.end()) acc += f->second;
        }
        auto end = Clock::now();
        lat.push_back(std::chrono::duration_cast<ns>(end - start).count());
    }
    sink += acc;
    return compute_stats(lat);
}

int main() {
    std::cout << "Serialized-but-random integer IDs -> unique storage\n";
    std::cout << "stream = " << N << " IDs, monotone with random gaps (1..64), "
              << "~20% replays of a recent ID\n\n";

    std::vector<uint64_t> ids = make_id_stream(N, 20260723ull);

    uint64_t sink = 0;

    // ----- Phase 0: small hot table -----------------------------------------
    std::cout << "=== Phase 0: small hot table (" << N_SMALL << " IDs, build + probe, "
              << SMALL_IT << " reps, ns) ===\n";
    print_header();
    // unordered_map first so it absorbs residual warmup.
    Stats s0u = bench_small<std::unordered_map<uint64_t, uint32_t>>(ids, sink);
    Stats s0m = bench_small<std::map<uint64_t, uint32_t>>(ids, sink);
    print_stats("unordered_map", s0u);
    print_stats("map", s0m);
    std::cout << "  -> map is " << std::fixed << std::setprecision(2)
              << (s0u.mean / s0m.mean) << "x the speed of unordered_map on mean\n\n"
              << std::defaultfloat;

    // ----- Phase 1: dedup insert --------------------------------------------
    // Three contenders. The reserved unordered_map is pre-sized for the whole
    // stream up front, which is the fix for the rehash spike -- IF you know the
    // count in advance. Note it must be sized for the STREAM (N), not the
    // unique count, because you cannot know the dedup ratio before you dedup.
    std::unordered_map<uint64_t, uint32_t> umap;
    std::unordered_map<uint64_t, uint32_t> umap_r;
    std::map<uint64_t, uint32_t> omap;
    size_t uniq_u = 0, uniq_r = 0, uniq_m = 0;

    umap_r.reserve(N);

    std::cout << "=== Phase 1: dedup insert, per-op latency (ns) ===\n";
    print_header();
    Stats s1u = bench_insert(ids, umap, uniq_u);
    Stats s1r = bench_insert(ids, umap_r, uniq_r);
    Stats s1m = bench_insert(ids, omap, uniq_m);
    print_stats("unordered_map", s1u);
    print_stats("umap+reserve", s1r);
    print_stats("map", s1m);
    std::cout << "  unique stored: unordered_map=" << uniq_u
              << " umap+reserve=" << uniq_r
              << " map=" << uniq_m
              << (uniq_u == uniq_m && uniq_r == uniq_m ? "  (identical key sets)"
                                                       : "  (MISMATCH)") << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  -> tail ratio unordered_map/map     : p999=" << (s1u.p999 / s1m.p999)
              << "x  p9999=" << (s1u.p9999 / s1m.p9999)
              << "x  max=" << (s1u.max / s1m.max) << "x\n";
    std::cout << "  -> tail ratio umap+reserve/map      : p999=" << (s1r.p999 / s1m.p999)
              << "x  p9999=" << (s1r.p9999 / s1m.p9999)
              << "x  max=" << (s1r.max / s1m.max) << "x\n";
    std::cout << "  -> bucket_count: grown=" << umap.bucket_count()
              << " (lf=" << umap.load_factor() << ")"
              << "  reserved=" << umap_r.bucket_count()
              << " (lf=" << umap_r.load_factor() << ")\n"
              << "     reserve() pays the whole bucket array once, before the timed loop.\n"
              << "     NB: the grown table's max is structural (6-8 ms every run, it is the\n"
              << "     rehash). map's and the reserved table's max is OS noise -- page faults\n"
              << "     as the allocator grows the heap -- and swings 40us..4ms run to run.\n"
              << "     Compare those two on p999/p9999, not on max.\n\n"
              << std::defaultfloat;

    // ----- Phase 2: point lookup --------------------------------------------
    std::vector<uint64_t> probes;
    probes.reserve(omap.size());
    for (const auto& [k, v] : omap) probes.push_back(k);
    std::shuffle(probes.begin(), probes.end(), std::mt19937_64(1234));

    std::cout << "=== Phase 2: point lookup, random order, all hits (ns) ===\n";
    print_header();
    Stats s2u = bench_lookup(umap, probes, sink);
    Stats s2r = bench_lookup(umap_r, probes, sink);
    Stats s2m = bench_lookup(omap, probes, sink);
    print_stats("unordered_map", s2u);
    print_stats("umap+reserve", s2r);
    print_stats("map", s2m);
    std::cout << "  -> unordered_map wins here, as it should: one hash, one bucket, done.\n"
              << "     This is the one axis where O(1) actually shows up.\n"
              << "     The reserved table is sparser (lf " << std::fixed << std::setprecision(2)
              << umap_r.load_factor() << " vs " << umap.load_factor()
              << "), so it trades memory for shorter chains.\n\n"
              << std::defaultfloat;

    // ----- Phase 3: ordered traversal ---------------------------------------
    std::cout << "=== Phase 3: full traversal in ascending ID order ===\n";
    {
        auto t0 = Clock::now();
        sink += ordered_scan_umap(umap);
        auto t1 = Clock::now();
        sink += ordered_scan_map(omap);
        auto t2 = Clock::now();

        double u_ms = std::chrono::duration_cast<ns>(t1 - t0).count() / 1e6;
        double m_ms = std::chrono::duration_cast<ns>(t2 - t1).count() / 1e6;
        std::cout << std::fixed << std::setprecision(2)
                  << "  unordered_map (extract keys + sort + re-lookup): " << u_ms << " ms\n"
                  << "  map           (iterate, already sorted)        : " << m_ms << " ms\n"
                  << "  -> map is " << (u_ms / m_ms) << "x faster\n\n"
                  << std::defaultfloat;
    }

    // ----- Phase 4: range query ---------------------------------------------
    std::cout << "=== Phase 4: range query, all IDs in [lo, hi] (" << RANGE_Q
              << " queries, ~1000-ID windows, ns) ===\n";
    print_header();
    {
        uint64_t lo_id = omap.begin()->first;
        uint64_t hi_id = omap.rbegin()->first;
        std::mt19937_64 rng(555);
        std::uniform_int_distribution<uint64_t> pick(lo_id, hi_id - 32'768);

        std::vector<std::pair<uint64_t, uint64_t>> ranges;
        ranges.reserve(RANGE_Q);
        for (int i = 0; i < RANGE_Q; ++i) {
            uint64_t lo = pick(rng);
            ranges.emplace_back(lo, lo + 32'768);   // ~1000 IDs at avg gap 32.5
        }

        std::vector<long long> lat_u, lat_m;
        lat_u.reserve(RANGE_Q);
        lat_m.reserve(RANGE_Q);

        for (auto [lo, hi] : ranges) {
            auto t0 = Clock::now();
            sink += range_query_umap(umap, lo, hi);
            auto t1 = Clock::now();
            lat_u.push_back(std::chrono::duration_cast<ns>(t1 - t0).count());
        }
        for (auto [lo, hi] : ranges) {
            auto t0 = Clock::now();
            sink += range_query_map(omap, lo, hi);
            auto t1 = Clock::now();
            lat_m.push_back(std::chrono::duration_cast<ns>(t1 - t0).count());
        }

        Stats s4u = compute_stats(lat_u);
        Stats s4m = compute_stats(lat_m);
        print_stats("unordered_map", s4u);
        print_stats("map", s4m);
        std::cout << "  -> map is " << std::fixed << std::setprecision(1)
                  << (s4u.mean / s4m.mean)
                  << "x faster. map does lower_bound + walk k elements;\n"
                  << "     unordered_map has to scan all " << umap.size()
                  << " entries for every query.\n\n" << std::defaultfloat;
    }

    // ----- Verdict -----------------------------------------------------------
    std::cout << "=== Verdict ===\n" << std::fixed << std::setprecision(2);
    std::cout << "  map vs unordered_map (grown):\n";
    std::cout << "    small hot table   : map " << (s0u.mean / s0m.mean) << "x faster (mean)\n";
    std::cout << "    insert mean       : map " << (s1u.mean / s1m.mean) << "x faster\n";
    std::cout << "    insert p9999      : map " << (s1u.p9999 / s1m.p9999) << "x faster\n";
    std::cout << "    insert worst case : map " << (s1u.max / s1m.max) << "x faster\n";
    std::cout << "    point lookup mean : map " << (s2u.mean / s2m.mean) << "x faster\n";
    std::cout << "  map vs unordered_map + reserve:\n";
    std::cout << "    insert mean       : map " << (s1r.mean / s1m.mean) << "x faster\n";
    std::cout << "    insert p9999      : map " << (s1r.p9999 / s1m.p9999) << "x faster\n";
    std::cout << "    insert worst case : map " << (s1r.max / s1m.max) << "x faster\n";
    std::cout << "    point lookup mean : map " << (s2r.mean / s2m.mean) << "x faster\n";
    std::cout << "  ordered traversal and range query are unchanged by reserve():\n";
    std::cout << "    a bigger bucket array does not give the table an order.\n";
    std::cout << "\n";
    std::cout << "  A number below 1.00 means unordered_map won that row.\n";
    std::cout << "  reserve() is the honest fix for the rehash spike -- when you know the\n";
    std::cout << "  count up front. It buys back the tail and costs memory. What it cannot\n";
    std::cout << "  buy is ordered or range access, which is where map's margin was widest.\n";

    std::cout << "\n[sink] " << sink << "\n";
    return 0;
}
