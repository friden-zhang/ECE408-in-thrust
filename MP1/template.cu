#include <assert.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <cxxopts.hpp>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <nvtx3/nvToolsExt.h>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <utility>
#include <vector>


// Helper function to read raw binary file
thrust::host_vector<float> read_raw_file(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file " << filename << std::endl;
    exit(1);
  }

  // read by lanes
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

    // Read input files
    auto input1_host_vector = read_raw_file(input_file1);
    auto input2_host_vector = read_raw_file(input_file2);
    auto expected_output_host_vector = read_raw_file(output_file);

    assert(input1_host_vector.size() == input2_host_vector.size() &&
           input1_host_vector.size() == expected_output_host_vector.size());

    auto data_length = input1_host_vector.size();

    // Allocate device memory and copy data
    nvtxRangePushA("Data Transfer to Device");
    thrust::device_vector<float> input1_device_vector = input1_host_vector;
    thrust::device_vector<float> input2_device_vector = input2_host_vector;
    thrust::device_vector<float> output_device_vector(data_length);
    nvtxRangePop();

    nvtxRangePushA("Thrust Transform");
    // run thrust algorithm
    thrust::transform(input1_device_vector.begin(), input1_device_vector.end(),
                      input2_device_vector.begin(),
                      output_device_vector.begin(), thrust::plus<float>());

    nvtxRangePop();

    // Copy result back to host
    thrust::host_vector<float> output_host_vector = output_device_vector;

    // compare results
    for (size_t i = 0; i < data_length; i++) {
      if (std::abs(output_host_vector[i] - expected_output_host_vector[i]) >
          1e-2) {
        std::cerr << "Mismatch at index " << i << ": " << output_host_vector[i]
                  << " vs " << expected_output_host_vector[i] << std::endl;
        return 1;
      }
    }

    std::cout << "Results match!" << std::endl;
  }

  return 0;
}
