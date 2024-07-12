#include <iostream>
#include <vector>
#include "naive.h"
#include <ios>
#include "join.hpp"

int main() {
    std::vector<float> inputs = { 0.5527185,0.39846906, -0.11766014,  0.19299345, -0.38549745,  0.08441927
,   0.26880047,  0.42681944, -0.10539523, -0.02164167,  0.41527015, -0.09802981
,  -0.07409753, -0.41598308,  0.09711669, -0.11594991, -0.4557323,   0.27337435
,  -0.11517189,  0.37859723, -0.15901394,  0.29185423, -0.344608,    0.08293352 };


    std::vector<int8_t> expectedResults = { 23,  17,  -5,   8, -16,   4,  11,  18,  -4,  -1,  17,  -4,  -3, -17,
    4,  -5, -19,  11,  -5,  16,  -7,  12, -14,   3 };


    auto ret = referenceOuter(inputs);

    std::cout << thresholds.size() << "\n";

    auto ret2 = multithreshold(inputs);

    std::cout << std::boolalpha << "Reference equal to expected:" << (ret == expectedResults) << "\n";
    std::cout << std::boolalpha << "Naive equal to expected:    " << (expectedResults == ret2) << "\n";


    std::vector<int> out(ret.begin(), ret.end());
    std::cout << "Reference Out:" << join(out, ",") << "\n";

    std::vector<int> out2(ret2.begin(), ret2.end());
    std::cout << "Naive Out:    " << join(out2, ",") << "\n";
}