#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

// Writes a random integer array to a binary file for the branch-prediction
// benchmark. Loading at runtime hides the data from the compiler so it cannot
// constant-fold the comparison or vectorize on known values.
//
// File layout: [size_t count][count uint32_t values, each in 0..255]
//
// Usage:  ./generate_branch_data [N] [path]
// Defaults: N=32768  path=branch_data.bin

int main(int argc, char** argv) {
    const size_t N    = (argc > 1) ? std::stoull(argv[1]) : 32768ull;
    const char*  path = (argc > 2) ? argv[2]              : "branch_data.bin";

    std::mt19937_64 rng(67890);
    std::uniform_int_distribution<uint32_t> dist(0, 255);

    std::vector<uint32_t> data(N);
    for (size_t i = 0; i < N; ++i) data[i] = dist(rng);

    FILE* f = std::fopen(path, "wb");
    if (!f) { std::perror("fopen"); return 1; }
    std::fwrite(&N, sizeof(N), 1, f);
    std::fwrite(data.data(), sizeof(uint32_t), N, f);
    std::fclose(f);

    std::printf("Wrote %zu uint32_t values (range 0..255) to %s\n", N, path);
    return 0;
}
