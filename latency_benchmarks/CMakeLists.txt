cmake_minimum_required(VERSION 3.16)
project(latency_benchmarks CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)   # for the inline asm in do_not_optimize / cpu_relax

# Default to an optimized build -- these numbers are meaningless at -O0.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
if(NOT CMAKE_CXX_FLAGS_RELEASE MATCHES "-O")
  set(CMAKE_CXX_FLAGS_RELEASE "-O2")
endif()

find_package(Threads REQUIRED)

# Every .cpp here is a standalone benchmark sharing only bench_util.h.
set(BENCHMARKS
  false_sharing        # cache line size differs: x86 64B vs Apple 128B
  memory_ordering      # ordering tax: free on x86 TSO, real on ARM
  atomic_vs_mutex      # contention crossover: when atomics lose
  arena_allocator      # custom allocator flattens the allocation tail
  thread_pool          # fixed pool vs thread-per-task dispatch latency
  struct_layout        # padding/alignment -> footprint -> traversal cost
  tlb_pressure         # page-walk cost; x86 4KB vs Apple 16KB pages
  benchmark_lies       # the four ways microbenchmarks deceive you
  pointer_chasing      # contiguous vs linked: the cache-miss-per-node cost
)

foreach(bench ${BENCHMARKS})
  add_executable(${bench} ${bench}.cpp)
  target_link_libraries(${bench} PRIVATE Threads::Threads)
endforeach()

# `cmake --build . --target run_all` builds everything and runs each in turn.
add_custom_target(run_all
  COMMAND ${CMAKE_COMMAND} -E echo "Running all benchmarks..."
)
foreach(bench ${BENCHMARKS})
  add_custom_command(TARGET run_all POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "==================== ${bench} ===================="
    COMMAND ${bench}
  )
  add_dependencies(run_all ${bench})
endforeach()
