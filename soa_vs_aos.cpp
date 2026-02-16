#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

using namespace std;

struct DateOfBirth {
    int year;
    int month;
    int day;
};

struct Person {
    string name;
    int age;
    DateOfBirth dob;
    //char address[128];
};

// ---------------- SoA Layout ----------------
struct PeopleSoA {
    vector<string> names;
    vector<int> ages;
    vector<DateOfBirth> dobs;

    PeopleSoA(size_t n) {
        names.reserve(n);
        ages.reserve(n);
        dobs.reserve(n);
    }
};

// ---------------- Utility: Timer ----------------
inline uint64_t ns() {
    return chrono::duration_cast<chrono::nanoseconds>(
        chrono::steady_clock::now().time_since_epoch()
    ).count();
}

// ---------------- Benchmark Functions ----------------
double average_age_aos(const vector<Person>& people) {
    long long sum = 0;
    for (const auto& p : people)
        sum += p.age;
    return double(sum) / people.size();
}

double average_age_soa(const PeopleSoA& p) {
    long long sum = 0;
    for (int age : p.ages)
        sum += age;
    return double(sum) / p.ages.size();
}

// ---------------- Main Benchmark ----------------
int main() {
    constexpr size_t N = 2'000'000;
    constexpr int ITER = 10;

    // ----------- Generate Data -----------
    vector<Person> aos;
    aos.reserve(N);

    PeopleSoA soa(N);

    for (size_t i = 0; i < N; i++) {
        string name = "Person_" + to_string(i);
        int age = 20 + int(i % 50);

        int year  = 1970 + int(i % 40);
        int month = 1 + int(i % 12);
        int day   = 1 + int(i % 28);

        DateOfBirth dob{year, month, day};

        aos.push_back({name, age, dob});

        soa.names.push_back(name);
        soa.ages.push_back(age);
        soa.dobs.push_back(dob);
    }

    // Warmup
    average_age_aos(aos);
    average_age_soa(soa);

    // ----------- Benchmark AoS -----------
    uint64_t aos_total = 0;
    for (int i = 0; i < ITER; i++) {
        uint64_t t0 = ns();
        volatile double avg = average_age_aos(aos);
        uint64_t t1 = ns();
        aos_total += (t1 - t0);
    }

    // ----------- Benchmark SoA -----------
    uint64_t soa_total = 0;
    for (int i = 0; i < ITER; i++) {
        uint64_t t0 = ns();
        volatile double avg = average_age_soa(soa);
        uint64_t t1 = ns();
        soa_total += (t1 - t0);
    }

    cout << fixed << setprecision(2);
    cout << "AoS avg time: " << (aos_total / double(ITER)) / 1e6 << " ms\n";
    cout << "SoA avg time: " << (soa_total / double(ITER)) / 1e6 << " ms\n";

    if (aos_total < soa_total)
        cout << "Winner: AoS (faster)\n";
    else
        cout << "Winner: SoA (faster)\n";

    return 0;
}
