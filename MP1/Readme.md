## Important Note

```bash
sudo apt install libcxxopts-dev
```

Make sure your CUDA driver version is compatible with the CUDA runtime version you're using. You can check your driver version with:

```bash
nvidia-smi
```

If you have multiple CUDA versions installed, you can switch between them using:

```bash
sudo update-alternatives --config cuda
```

## Objective

The purpose of this lab is to become familiar with using the Thrust library by implementing a simple vector addition operation. This implementation uses Thrust's high-level abstractions to perform GPU-accelerated operations.

## Prerequisites

Before starting this lab, make sure that:

* You have completed all week 1 lectures or videos
* You have a CUDA-capable GPU and appropriate drivers installed
* You are familiar with C++ and basic CUDA concepts
* Chapter 2 of the text book would also be helpful

## Building the Project

This project uses CMake as the build system. To build:

```bash
cd /home/friden/Code/ECE408-in-thrust
mkdir -p build
cd build
cmake ..
make
```

## Running the Program

After building, you can run the vector addition program:

```bash
./build/MP1/template -w /home/friden/Code/ECE408-in-thrust/MP1/data
```

### Command Line Options

* `-w, --work_dir`: Specify the working directory containing test datasets (default: `./data/`)
* `-h, --help`: Print usage information

Example:
```bash
./build/MP1/template --work_dir /path/to/data
```

## Input Data Format

The program reads `.raw` files with the following text format:
* First line: integer count of elements
* Following lines: float values (one per line)

Each test case directory should contain:
* `input0.raw`: First input vector
* `input1.raw`: Second input vector
* `output.raw`: Expected output (sum of input vectors)

## Expected Output

If your solution is correct, you should see output like:

```
Found sub-directory: "/home/friden/Code/ECE408-in-thrust/MP1/data/0"
Results match!
Found sub-directory: "/home/friden/Code/ECE408-in-thrust/MP1/data/1"
Results match!
...
```

## Implementation Details

The current implementation uses:
* `thrust::host_vector<float>` for host-side data storage
* `thrust::device_vector<float>` for device-side data storage
* `thrust::transform` with `thrust::plus<float>()` for vector addition
* NVTX markers for performance profiling

## Troubleshooting

### CUDA Driver/Runtime Version Mismatch

If you see an error like:
```
cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version
```

Try:
1. Check your driver version: `nvidia-smi`
2. Switch to a compatible CUDA version
3. Rebuild the project after switching CUDA versions

### Library Not Found Error

If you see:
```
error while loading shared libraries: libcudart.so.XX
```

Set the library path:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Or add it permanently to your `~/.zshrc` or `~/.bashrc`.

### Floating Point Precision Issues

The comparison tolerance is set to `1e-3` to account for floating-point precision differences between CPU and GPU computations.