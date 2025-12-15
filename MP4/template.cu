#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cxxopts.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>


// Helper function to read raw binary file
thrust::host_vector<float> read_raw_file(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file " << filename << std::endl;
    exit(1);
  }

  int count;
  file >> count;

  std::vector<float> temp_data;
  temp_data.reserve(count);

  float value;
  for (int i = 0; i < count; i++) {
    file >> value;
    temp_data.push_back(value);
  }

  file.close();

  thrust::host_vector<float> data(temp_data.begin(), temp_data.end());
  return data;
}

//@@ Define any useful program-wide constants here
constexpr int kKernelRadius = 1;
constexpr int kKernelWidth = 2 * kKernelRadius + 1;
constexpr int kKernelSize = kKernelWidth * kKernelWidth * kKernelWidth;

constexpr int kTileDim = 8;

//@@ Define constant memory for device kernel here
__constant__ float kKernel[kKernelSize];

inline void check_cuda(cudaError_t result, const char *msg) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA error: " << msg << ": " << cudaGetErrorString(result)
              << std::endl;
    std::exit(1);
  }
}

__global__ void conv3d(const float *__restrict__ input,
                       float *__restrict__ output, int dim_z, int dim_y,
                       int dim_x) {
  constexpr int kSharedDim = kTileDim + 2 * kKernelRadius;
  constexpr int kSharedSize = kSharedDim * kSharedDim * kSharedDim;
  __shared__ float shmem[kSharedSize];

  const int block_origin_x = static_cast<int>(blockIdx.x) * kTileDim;
  const int block_origin_y = static_cast<int>(blockIdx.y) * kTileDim;
  const int block_origin_z = static_cast<int>(blockIdx.z) * kTileDim;

  const int thread_linear =
      (static_cast<int>(threadIdx.z) * blockDim.y + static_cast<int>(threadIdx.y)) *
          blockDim.x +
      static_cast<int>(threadIdx.x);
  const int block_threads = static_cast<int>(blockDim.x * blockDim.y * blockDim.z);

  for (int i = thread_linear; i < kSharedSize; i += block_threads) {
    const int sx = i % kSharedDim;
    const int sy = (i / kSharedDim) % kSharedDim;
    const int sz = i / (kSharedDim * kSharedDim);

    const int gx = block_origin_x + sx - kKernelRadius;
    const int gy = block_origin_y + sy - kKernelRadius;
    const int gz = block_origin_z + sz - kKernelRadius;

    if (gx >= 0 && gx < dim_x && gy >= 0 && gy < dim_y && gz >= 0 &&
        gz < dim_z) {
      const int global_idx = (gz * dim_y + gy) * dim_x + gx;
      shmem[i] = input[global_idx];
    } else {
      shmem[i] = 0.0f;
    }
  }

  __syncthreads();

  const int x = block_origin_x + static_cast<int>(threadIdx.x);
  const int y = block_origin_y + static_cast<int>(threadIdx.y);
  const int z = block_origin_z + static_cast<int>(threadIdx.z);
  if (x >= dim_x || y >= dim_y || z >= dim_z) {
    return;
  }

  float sum = 0.0f;
  const int base_sx = static_cast<int>(threadIdx.x) + kKernelRadius;
  const int base_sy = static_cast<int>(threadIdx.y) + kKernelRadius;
  const int base_sz = static_cast<int>(threadIdx.z) + kKernelRadius;

  for (int dz = -kKernelRadius; dz <= kKernelRadius; dz++) {
    for (int dy = -kKernelRadius; dy <= kKernelRadius; dy++) {
      for (int dx = -kKernelRadius; dx <= kKernelRadius; dx++) {
        const int sx = base_sx + dx;
        const int sy = base_sy + dy;
        const int sz = base_sz + dz;
        const int s_idx = (sz * kSharedDim + sy) * kSharedDim + sx;

        const int k_idx = ((dz + kKernelRadius) * kKernelWidth +
                           (dy + kKernelRadius)) *
                              kKernelWidth +
                          (dx + kKernelRadius);
        sum += shmem[s_idx] * kKernel[k_idx];
      }
    }
  }

  const int out_idx = (z * dim_y + y) * dim_x + x;
  output[out_idx] = sum;
}

int main(int argc, char *argv[]) {
  cxxopts::Options options("MP4 Template",
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

    std::string input_file = entry.path().string() + "/input.dat";
    std::string mask_file = entry.path().string() + "/kernel.dat";
    std::string output_file = entry.path().string() + "/output.dat";

    // Read input files
    auto input_host_vector = read_raw_file(input_file);
    auto mask_host_vector = read_raw_file(mask_file);
    auto expected_output_host_vector = read_raw_file(output_file);

    assert(input_host_vector.size() == expected_output_host_vector.size());
    assert(mask_host_vector.size() == kKernelSize);

    const int dim_z = static_cast<int>(std::lround(input_host_vector[0]));
    const int dim_y = static_cast<int>(std::lround(input_host_vector[1]));
    const int dim_x = static_cast<int>(std::lround(input_host_vector[2]));
    const int volume_size = dim_z * dim_y * dim_x;

    assert(static_cast<int>(input_host_vector.size()) == volume_size + 3);

    std::cout << "Processing volume " << dim_z << "x" << dim_y << "x" << dim_x
              << " (voxels=" << volume_size << "), dir: " << entry.path()
              << std::endl;

    //@@ Copy kernel to constant memory
    check_cuda(cudaMemcpyToSymbol(kKernel, mask_host_vector.data(),
                                  kKernelSize * sizeof(float)),
               "cudaMemcpyToSymbol(kKernel) failed");

    //@@ Allocate device memory and copy input data to device
    thrust::host_vector<float> input_volume(input_host_vector.begin() + 3,
                                            input_host_vector.end());
    thrust::device_vector<float> d_input = input_volume;
    thrust::device_vector<float> d_output(volume_size);

    //@@ Launch kernel
    dim3 block_dim(kTileDim, kTileDim, kTileDim);
    dim3 grid_dim((dim_x + kTileDim - 1) / kTileDim,
                  (dim_y + kTileDim - 1) / kTileDim,
                  (dim_z + kTileDim - 1) / kTileDim);
    conv3d<<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(d_input.data()),
                                    thrust::raw_pointer_cast(d_output.data()),
                                    dim_z, dim_y, dim_x);
    check_cuda(cudaGetLastError(), "conv3d launch failed");
    check_cuda(cudaDeviceSynchronize(), "conv3d execution failed");

    // Copy result back to host
    thrust::host_vector<float> output_volume = d_output;
    thrust::host_vector<float> output_host_vector(input_host_vector.size());
    output_host_vector[0] = input_host_vector[0];
    output_host_vector[1] = input_host_vector[1];
    output_host_vector[2] = input_host_vector[2];
    for (int i = 0; i < volume_size; i++) {
      output_host_vector[i + 3] = output_volume[i];
    }

    // compare results
    for (size_t i = 0; i < output_host_vector.size(); i++) {
      if (std::abs(output_host_vector[i] - expected_output_host_vector[i]) >
          1e-2) {
        std::cerr << "Mismatch at index " << i << ": " << output_host_vector[i]
                  << " (computed) vs " << expected_output_host_vector[i]
                  << " (expected)" << std::endl;
        return 1;
      }
    }
    std::cout << "Output matches expected results." << std::endl;
  }

  return 0;
}
