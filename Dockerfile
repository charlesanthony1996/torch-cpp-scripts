# Use a PyTorch base image that supports C++ and CUDA
FROM pytorch/pytorch:latest

# Install dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Download LibTorch for CPU
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip && \
    unzip libtorch-shared-with-deps-latest.zip && \
    rm libtorch-shared-with-deps-latest.zip

# Download MNIST dataset
# Download MNIST dataset from an alternative source
# Download MNIST dataset from an alternative source
RUN mkdir -p /app/mnist && \
    wget -q https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz -P /app/mnist && \
    wget -q https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz -P /app/mnist && \
    wget -q https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz -P /app/mnist && \
    wget -q https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz -P /app/mnist && \
    gunzip /app/mnist/*.gz



# Copy project files
COPY . .

# Build project
RUN mkdir build && cd build && \
    cmake -DCMAKE_PREFIX_PATH=/app/libtorch .. && \
    cmake --build .

# Run the compiled binary
CMD ["./build/TorchExample"]
