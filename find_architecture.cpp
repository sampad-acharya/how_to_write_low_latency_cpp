#include <iostream>

int main() {
#if defined(__x86_64__) || defined(_M_X64)
    std::cout << "Architecture: x86_64\n";
#elif defined(__i386__) || defined(_M_IX86)
    std::cout << "Architecture: x86 (32-bit)\n";
#elif defined(__aarch64__) || defined(_M_ARM64)
    std::cout << "Architecture: ARM64 (AArch64)\n";
#elif defined(__arm__) || defined(_M_ARM)
    std::cout << "Architecture: ARM (32-bit)\n";
#else
    std::cout << "Architecture: Unknown\n";
#endif
}
