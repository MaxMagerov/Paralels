#include <iostream>
#include <cmath>
#include <vector>

#ifdef FLOAT
typedef float TYPE;
#else
typedef double TYPE;
#endif

int main() {
    std::vector<TYPE> arr(10'000'000);
    TYPE sum{};
    for (int i = 0; i < 10'000'000; ++i)
    {
#ifdef FLOAT
        arr[i] = sinf(2 * M_PI * i / 10'000'000);
#else
        arr[i] = sin(2 * M_PI * i / 10'000'000);
#endif
        sum += arr[i];
    }
    std::cout << sum << std::endl;
    return 0;
}
