#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <chrono>

// Define a CNN model
struct CNNImpl : torch::nn::Module {
    CNNImpl()
        : conv1(torch::nn::Conv2dOptions(1, 32, 3).stride(1).padding(1)),
          conv2(torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1)),
          fc1(64 * 7 * 7, 128),
          fc2(128, 10) {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1(x));
        x = torch::max_pool2d(x, 2);
        x = torch::relu(conv2(x));
        x = torch::max_pool2d(x, 2);
        x = x.view({x.size(0), -1});
        x = torch::relu(fc1(x));
        x = fc2(x);
        return torch::log_softmax(x, 1);
    }

    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
};
TORCH_MODULE(CNN);

int main() {
    // Check for CUDA support
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Training on: " << (device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

    // Create model
    CNN model;
    model->to(device);

    // Load MNIST dataset
    // auto train_dataset = torch::data::datasets::MNIST("./mnist")
    auto train_dataset = torch::data::datasets::MNIST("../mnist")

        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());

    auto data_loader = torch::data::make_data_loader(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(64).workers(2));

    // Define loss function and optimizer
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01).momentum(0.9));

    const int epochs = 5;
    model->train();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double running_loss = 0.0;
        int batch_count = 0;

        auto start_time = std::chrono::high_resolution_clock::now();

        for (auto& batch : *data_loader) {
            optimizer.zero_grad();

            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            // Forward pass
            auto output = model->forward(data);
            auto loss = torch::nll_loss(output, target);

            // Backward pass
            loss.backward();
            optimizer.step();

            running_loss += loss.item<double>();
            batch_count++;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        std::cout << "Epoch [" << (epoch + 1) << "/" << epochs << "] - Loss: " 
                  << (running_loss / batch_count) 
                  << " - Time: " << elapsed.count() << "s" << std::endl;
    }

    // Save model
    torch::save(model, "cnn_model.pt");
    std::cout << "Model saved to cnn_model.pt" << std::endl;

    return 0;
}
