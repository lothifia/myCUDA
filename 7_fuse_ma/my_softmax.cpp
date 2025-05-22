#include <iostream>
#include <vector>
#include <cmath>

std::vector<float> naive_softmax(const std::vector<float>& input) {
    float sum = 0;
    for (float i : input) {
        sum += std::exp(i);
    }
    std::vector<float> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = exp(input[i]) / sum;
    }
    return output;
}
std::vector<float> safe_softmax(const std::vector<float>& input) {
    float max_val = -0x3f3f3f3f;
    float sum = 0;
    std::vector<float> output(input.size());
    for (float i : input) {
        max_val = std::max(max_val, i);
    }
    for (float i : input) {
        sum += std::exp(i - max_val);
    }
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = exp(input[i] - max_val) / sum;
    }
    return output;
}

#pragma unroll
std::vector<float> online_softmax(const std::vector<float>& input) {
    float curr_max_val = -0x3f3f3f3f;
    float prev_max_val = -0x3f3f3f3f;
    float sum = 0;
    std::vector<float> output(input.size());
    for(size_t i = 0; i < input.size(); ++i) {
        prev_max_val = curr_max_val;
        curr_max_val = std::max(curr_max_val, input[i]);
        sum = sum * (exp(prev_max_val - curr_max_val)) + exp(input[i] - curr_max_val);
    }
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = exp(input[i] - curr_max_val) / sum;
    }
    return output;
}

float online_softmaxAndDotproduct(const std::vector<float>& value, const std::vector<float>& src) {
    float curr_max_val = -0x3f3f3f3f;
    float prev_max_val = -0x3f3f3f3f;
    float sum = 0;
    float res = 0.0;
    std::vector<float> output(src.size());
    for(size_t i = 0; i < src.size(); ++i) {
        prev_max_val = curr_max_val;
        curr_max_val = std::max(curr_max_val, src[i]);
        sum = sum * (exp(prev_max_val - curr_max_val)) + exp(src[i] - curr_max_val);
        std::cout << sum << std:: endl;  
    }
    for (size_t i = 0; i < src.size(); ++i) {
        res += exp(src[i] - curr_max_val) / sum * value[i];
    }
    return res;
}
// 目标是得到最终的P * V的值， 而不是每个位置的softmax， 所以只需要一个循环。
float online_softmaxAndDotproduct_1loop(const std::vector<float>& value, const std::vector<float>& src) {
    float curr_max_val = -1.0/0.0;
    float prev_max_val = -1.0/0.0;
    float pre_sum = 0.f;
    float curr_sum = 0.f;
    float res = 0.0;
    for(size_t i = 0; i < src.size(); ++i) {
        prev_max_val = curr_max_val;
        curr_max_val = std::max(curr_max_val, src[i]);
        pre_sum = curr_sum;
        curr_sum = pre_sum * exp(prev_max_val - curr_max_val) + exp(src[i] - curr_max_val);
        res = res * exp(prev_max_val - curr_max_val) * pre_sum / curr_sum + exp(src[i] - curr_max_val) * value[i] /curr_sum; 
    }
    return res;
}


int main() {
    std::vector<float> test {1, 2, 3, 4, 6, 9};
    std::vector<float> value {2, 2, 2, 2, 2, 2};
    std::vector<float> naive_res{naive_softmax(test)};
    std::vector<float> safe_res{safe_softmax(test)};
    std::vector<float> online_res{online_softmax(test)};
    for(int i = 0; i < test.size(); ++i) {
        std::cout << "naive softmax: " << naive_res[i] << "safe softmax: " << safe_res[i] << "online softmax: " << online_res[i] << std::endl;
    }
    std::cout << "online softmax and dot product: " << online_softmaxAndDotproduct(value, test) << std::endl;
    std::cout << "better" << online_softmaxAndDotproduct_1loop(value, test) << std::endl;
    return 0;
}