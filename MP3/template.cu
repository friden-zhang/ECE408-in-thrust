#include <cuda_runtime.h>
#include <cxxopts.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

struct MatrixBuffer {
  int row, col;
  thrust::host_vector<float> data;
};

MatrixBuffer load_matrix(const std::string &file_path) {
  MatrixBuffer matrix;
  std::ifstream file(file_path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("failed to open " + file_path);
  }

  file >> matrix.row >> matrix.col;
  auto size = matrix.row * matrix.col;
  matrix.data.resize(size);
  for (int i = 0; i < size; i++) {
    file >> matrix.data[i];
  }

  return matrix;
}

constexpr int kTileDim = 16;

__global__ void MatMulTiledKernel(const float *__restrict__ A,
                                  const float *__restrict__ B,
                                  float *__restrict__ C, int M, int N, int P) {
  __shared__ float shared_a[kTileDim][kTileDim];
  __shared__ float shared_b[kTileDim][kTileDim];

  const int row = blockIdx.y * kTileDim + threadIdx.y;
  const int col = blockIdx.x * kTileDim + threadIdx.x;
  const int tiles = (N + kTileDim - 1) / kTileDim;

  float sum = 0.0f;

  for (int tile = 0; tile < tiles; ++tile) {
    const int k_base = tile * kTileDim;

    if (row < M && (k_base + threadIdx.x) < N) {
      shared_a[threadIdx.y][threadIdx.x] = A[row * N + k_base + threadIdx.x];
    } else {
      shared_a[threadIdx.y][threadIdx.x] = 0.0f;
    }

    if (col < P && (k_base + threadIdx.y) < N) {
      shared_b[threadIdx.y][threadIdx.x] =
          B[(k_base + threadIdx.y) * P + col];
    } else {
      shared_b[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

#pragma unroll
    for (int k = 0; k < kTileDim; ++k) {
      sum += shared_a[threadIdx.y][k] * shared_b[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < P) {
    C[row * P + col] = sum;
  }
}

inline void check_cuda(cudaError_t result, const char *msg) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + ": " +
                             cudaGetErrorString(result));
  }
}

int main(int argc, char **argv) {

  cxxopts::Options options("MP1 Template",
                           "CUDA Programming Assignment Template Code");
  // clang-format off
  options.add_options()
  ("w,work_dir", "Working directory", cxxopts::value<std::string>()->default_value("./data/"))
  ("h,help", "Print usage");
  // clang-format on

  auto result = options.parse(argc, argv);
  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  // get work directory
  std::string work_dir = result["work_dir"].as<std::string>();

  // list all sub dir in work dir
  std::filesystem::path work_path(work_dir);
  if (!std::filesystem::exists(work_path) ||
      !std::filesystem::is_directory(work_path)) {
    std::cerr << "Error: Work directory " << work_dir << " does not exist."
              << std::endl;
    return 1;
  }

  // list all dir in work dir
  for (const auto &entry : std::filesystem::directory_iterator(work_path)) {
    if (entry.is_directory()) {
      std::cout << "Found sub-directory: " << entry.path() << std::endl;
    }

    std::string input_file1 = entry.path().string() + "/input0.raw";
    std::string input_file2 = entry.path().string() + "/input1.raw";
    std::string output_file = entry.path().string() + "/output.raw";

    // load input matrices
    MatrixBuffer input_a = load_matrix(input_file1);
    MatrixBuffer input_b = load_matrix(input_file2);
    MatrixBuffer expected_output = load_matrix(output_file);

    if (input_a.col != input_b.row || input_a.row != expected_output.row ||
        input_b.col != expected_output.col) {
      std::cerr
          << "Error: Matrix dimensions do not match for multiplication, path: "
          << entry.path() << std::endl;
      continue;
    }

    int M = input_a.row;
    int N = input_a.col;
    int P = input_b.col;

    std::cout << "Processing matrices of size " << M << "x" << N << " and " << N
              << "x" << P << ", dir: " << entry.path() << std::endl;

    // allocate device memory
    thrust::device_vector<float> d_a = input_a.data;
    thrust::device_vector<float> d_b = input_b.data;
    thrust::device_vector<float> d_c(M * P);

    // perform matrix multiplication on device using shared-memory tiling
    dim3 block_dim(kTileDim, kTileDim);
    dim3 grid_dim((P + kTileDim - 1) / kTileDim,
                  (M + kTileDim - 1) / kTileDim);
    MatMulTiledKernel<<<grid_dim, block_dim>>>(
        thrust::raw_pointer_cast(d_a.data()),
        thrust::raw_pointer_cast(d_b.data()),
        thrust::raw_pointer_cast(d_c.data()), M, N, P);
    check_cuda(cudaGetLastError(), "MatMulTiledKernel launch failed");
    check_cuda(cudaDeviceSynchronize(), "MatMulTiledKernel execution failed");
    // copy result back to host
    thrust::host_vector<float> output_host_vector = d_c;
    // compare results
    for (int i = 0; i < M * P; i++) {
      if (std::abs(output_host_vector[i] - expected_output.data[i]) > 1e-2) {
        std::cerr << "Mismatch at index " << i << ": " << output_host_vector[i]
                  << " vs " << expected_output.data[i] << std::endl;
        return 1;
      }
    }
    std::cout << "Matrix multiplication result matches expected output."
              << std::endl;
  }

  return 0;
}
