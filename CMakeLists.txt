cmake_minimum_required(VERSION 3.10)
project(torch_cnn)

set(CMAKE_CXX_STANDARD 17)

# Set path to your local libtorch
set(Torch_DIR "/Users/charles/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)

add_executable(example1 example1.cpp)
target_link_libraries(example1 "${TORCH_LIBRARIES}")
set_property(TARGET example1 PROPERTY CXX_STANDARD 17)
