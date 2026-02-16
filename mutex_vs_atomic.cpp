#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <mutex>
#include <condition_variable>

// ------------------------------------------------------------
// Original Mutex Queue
// ------------------------------------------------------------
class MutexQueue {
    std::vector<int> queue{};
    int current_idx{};
    int queue_size{};
    int capacity{};
    std::mutex mtx{};
    std::condition_variable cv{};

public:
    explicit MutexQueue(int cap) : capacity(cap) {
        queue.resize(capacity);
    }

    void enqueue(int element) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this](){ return queue_size < capacity; });

        queue[++current_idx % capacity] = element;
        ++queue_size;
        cv.notify_one();
    }

    int dequeue() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this](){ return queue_size > 0; });

        int end_idx = capacity + current_idx - queue_size + 1;
        int result = queue[end_idx % capacity];
        --queue_size;
        cv.notify_one();
        return result;
    }
};

// ------------------------------------------------------------
// Atomic Spin Queue
// ------------------------------------------------------------
class AtomicQueue {
    std::vector<int> queue{};
    std::atomic<int> current_idx{0};
    std::atomic<int> queue_size{0};
    int capacity{};

public:
    explicit AtomicQueue(int cap) : capacity(cap) {
        queue.resize(capacity);
    }

    void enqueue(int element) {
        while (queue_size.load(std::memory_order_acquire) >= capacity) {
            // spin
        }

        int idx = current_idx.fetch_add(1, std::memory_order_relaxed) + 1;
        queue[idx % capacity] = element;

        queue_size.fetch_add(1, std::memory_order_release);
    }

    int dequeue() {
        while (queue_size.load(std::memory_order_acquire) <= 0) {
            // spin
        }

        int cur = current_idx.load(std::memory_order_relaxed);
        int qs  = queue_size.load(std::memory_order_relaxed);

        int end_idx = capacity + cur - qs + 1;
        int result = queue[end_idx % capacity];

        queue_size.fetch_sub(1, std::memory_order_release);
        return result;
    }
};

// ------------------------------------------------------------
// Benchmark Harness
// ------------------------------------------------------------
template <typename Queue>
long long benchmark_queue(const std::string& name, int capacity, int ops_per_thread, int producers, int consumers) {
    Queue q(capacity);

    auto start = std::chrono::steady_clock::now();

    std::vector<std::thread> threads;

    // Producers
    for (int p = 0; p < producers; ++p) {
        threads.emplace_back([&]() {
            for (int i = 0; i < ops_per_thread; ++i) {
                q.enqueue(i);
            }
        });
    }

    // Consumers
    for (int c = 0; c < consumers; ++c) {
        threads.emplace_back([&]() {
            for (int i = 0; i < ops_per_thread; ++i) {
                q.dequeue();
            }
        });
    }

    for (auto& t : threads) t.join();

    auto end = std::chrono::steady_clock::now();
    long long ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    std::cout << name << " took " << ns / 1e6 << " ms\n";
    return ns;
}

// ------------------------------------------------------------
// main
// ------------------------------------------------------------
int main() {
    const int capacity = 1024;
    const int ops_per_thread = 20000000;
    const int producers = 2;
    const int consumers = 2;
    const int small_ops_per_thread = 10000; 
    const int small_producers = 1;
    const int small_consumers = 1;

    std::cout << "Small bench mark with " << small_ops_per_thread << " operations per thread" << std::endl;
    long long t_samll_1 = benchmark_queue<MutexQueue>("Warmup Mutex", capacity, small_ops_per_thread, small_producers, small_consumers);
    long long t_samll_2 = benchmark_queue<AtomicQueue>("Warmup Atomic", capacity, small_ops_per_thread, small_producers, small_consumers);
    std::cout << "\nSpeedup: " << (double)t_samll_1/ t_samll_2 << "x faster\n";

    std::cout << "\nRunning real benchmark " << ops_per_thread << " operations per thread" << std::endl;

    long long t1 = benchmark_queue<MutexQueue>("Mutex Queue", capacity, ops_per_thread, producers, consumers);
    long long t2 = benchmark_queue<AtomicQueue>("Atomic Queue", capacity, ops_per_thread, producers, consumers);

    std::cout << "\nSpeedup: " << (double)t1 / t2 << "x faster\n";
}