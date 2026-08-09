# Low-Latency C++ Benchmark Suite

Nine self-contained benchmarks for the talk, each demonstrating one place where
latency (especially tail latency) is won or lost. Every benchmark:

- reports **percentiles** (p50/p90/p99/p999/max), not just the mean — the tail
  is the point;
- is **portable across x86-64 and ARM64** with no code changes;
- guards against the usual measurement lies (dead-code elimination, hoisting,
  cold-start) via `bench_util.h`.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# run one:
./build/false_sharing
# or build + run all in sequence:
cmake --build build --target run_all
```

Requires CMake ≥ 3.16 and a C++20 compiler (clang or gcc).

### Running on x86 as well as ARM

Several demos are **microarchitectural** — their *timing* is a property of the
physical CPU. On an Apple-Silicon Mac you can compile an x86 binary
(`--target=x86_64-apple-darwin`) and run it under Rosetta, but Rosetta executes
on the M-series silicon, so the **x86 timing is not real** for the hardware-effect
demos below. To get true x86 numbers, build and run on an actual x86 box (a
small Linux VM is enough). Correctness/logic demos are fine under Rosetta.

| Benchmark          | x86 numbers need real x86 HW? |
|--------------------|-------------------------------|
| false_sharing      | **Yes** (cache-line size is physical) |
| memory_ordering    | **Yes** (TSO vs weak is physical) |
| tlb_pressure       | **Yes** (page size / TLB is physical) |
| atomic_vs_mutex    | Mostly (contention crossover shifts by HW) |
| the other five     | No — logic/allocator/scheduler, Rosetta ok |

## The nine

| # | File | Section of the talk | The one-line reveal |
|---|------|---------------------|---------------------|
| 1 | `false_sharing.cpp`   | Memory layout & cache | `alignas(64)` fixes false sharing on x86 (64B lines) but **not** on Apple Silicon (128B lines). |
| 2 | `memory_ordering.cpp` | Atomics vs mutexes    | The cost of `seq_cst` is ~free on x86 (TSO) and a real tax on ARM (weak) — same source. |
| 3 | `atomic_vs_mutex.cpp` | Atomics vs mutexes    | Atomics win at low contention; past core count the spinlock's p999 explodes and the mutex wins. |
| 4 | `arena_allocator.cpp` | STL containers        | A bump allocator flattens the allocation tail that `new`/`delete` leaves behind. |
| 5 | `thread_pool.cpp`     | Threading & scheduling| Thread-per-task pays a syscall with a fat tail; a warm pool turns dispatch into a queue push. |
| 6 | `struct_layout.cpp`   | Memory layout & cache | Reordering members large→small shrinks the struct 40%, so the same traversal streams less memory. |
| 7 | `tlb_pressure.cpp`    | Memory layout & cache | Page-walk cost is distinct from cache misses; x86's 4KB pages hit the cliff before Apple's 16KB. |
| 8 | `benchmark_lies.cpp`  | Benchmarking          | The four ways your microbenchmark is already wrong: DCE, hoisting, timer overhead, no warmup. |
| 9 | `pointer_chasing.cpp` | Memory layout & cache | Same O(n) work, but a linked list pays one dependent cache miss per node (~100x a vector walk). |

## Reading the output

Each program prints the detected architecture and assumed cache-line size in its
banner, then a percentile table. When presenting on two machines, put the two
outputs side by side — the *shape* of the table (which rows tie, which tail
blows up) is the story, and it changes between x86 and ARM.

`bench_util.h` holds the shared plumbing (percentile `Stats`, a portable
`cpu_relax()`, `do_not_optimize()`, cache-line size, arch label) so each
benchmark file stays focused on its one idea.
