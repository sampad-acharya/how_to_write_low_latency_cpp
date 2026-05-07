#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

// Writes a spike vector to a binary file.
// File layout: [size_t count][count bytes, each 0 or 1].
//
// Usage:  ./generate_spike_vector [N] [path] [probability]
// Defaults: N=100000  path=spike_vector.bin  probability=0.01

int main(int argc, char** argv) {
    const size_t N         = (argc > 1) ? std::stoull(argv[1]) : 1000000ull;
    const char*  path      = (argc > 2) ? argv[2]              : "spike_vector.bin";
    const double spike_p   = (argc > 3) ? std::stod(argv[3])   : 0.01;

    std::mt19937_64 rng(12345);
    std::bernoulli_distribution spike_dist(spike_p);

    std::vector<uint8_t> spike(N);
    size_t spike_count = 0;
    for (size_t i = 0; i < N; ++i) {
        spike[i] = spike_dist(rng) ? 1u : 0u;
        spike_count += spike[i];
    }

    FILE* f = std::fopen(path, "wb");
    if (!f) {
        std::perror("fopen");
        return 1;
    }
    std::fwrite(&N, sizeof(N), 1, f);
    std::fwrite(spike.data(), 1, N, f);
    std::fclose(f);

    std::printf("Wrote %zu entries (%zu spikes, %.3f%%) to %s\n",
                N, spike_count, 100.0 * spike_count / N, path);
    return 0;
}
