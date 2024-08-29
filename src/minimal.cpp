#include "optimized.h"
#include <iostream>
#include <random>

std::vector<float> inp;

std::vector<float> getBatchInputs(size_t batchSize) {

    std::random_device rndDevice;
    std::mt19937 mersenneEngine{ rndDevice() };  // Generates random integers

    std::uniform_real_distribution<float> dist{ -4.0, 4.0 };

    auto gen = [&dist, &mersenneEngine]() { return dist(mersenneEngine); };

    std::vector<float> ret(24 * batchSize);

    std::generate(ret.begin(), ret.end(), gen);

    return ret;
}

int main() {
    inp = getBatchInputs(4096);

    for (size_t i = 0; i < 100000; ++i) {
        auto out = optimized::multithresholdLinearPerTensor(inp);
        std::cout << out[0] << "\n";
    }

    return 0;
}