#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <chrono>

struct Net : torch::nn::Module {
    Net(int64_t N, int64_t M) {
        W = register_parameter("W", torch::randn({N, M}));
        b = register_parameter("b", torch::randn({M}));
    }
    torch::Tensor forward(torch::Tensor input) {
        return torch::addmm(b, input, W);
    }
    torch::Tensor W, b;
};

int main() {
    Net model(3, 5);
    torch::Tensor input = torch::randn({2, 3});
    torch::Tensor output = model.forward(input);
    std::cout << output << std::endl;
    return 0;
}