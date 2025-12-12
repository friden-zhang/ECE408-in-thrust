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

This module mirrors the MP1 build instructions but the kernel now performs dense
matrix multiplication. Thrust is only used to manage device buffers, while the
heavy lifting happens inside a tiled CUDA kernel that stages matrix blocks in
shared memory to maximize data reuse. The goal is to understand how to express a
more complex operation—`C = A * B`—with classic shared-memory tiling.

## Prerequisites

Before starting this lab, make sure that:

* You have completed all week 1 lectures or videos
* You have a CUDA-capable GPU and appropriate drivers installed
* You are familiar with C++ and basic CUDA concepts
* Chapter 2 of the text book would also be helpful

## Building the Project

This project uses CMake as the build system. To build:

```bash
cd /path/ECE408-in-thrust
mkdir -p build
cd build
cmake ..
make
```

## Running the Program

After building (make sure the top-level CMakeLists adds `add_subdirectory(MP3)`),
launch the MP3 executable and point it at the dataset folder:

```bash
./build/MP3/mp3 -w /path/ECE408-in-thrust/MP3/data
```

### Command Line Options

* `-w, --work_dir`: Specify the working directory containing test datasets (default: `./data/`)
* `-h, --help`: Print usage information

Example:
```bash
./build/MP3/mp3 --work_dir /path/to/data
```

## Input Data Format

Each subdirectory in `MP3/data` contains three `.raw` files that describe
matrices in row-major order:

* First line: integer number of rows
* Second line: integer number of columns
* Remaining lines: `rows * cols` floating point values

Files inside a dataset folder:

* `input0.raw`: matrix `A` of shape `M x N`
* `input1.raw`: matrix `B` of shape `N x P`
* `output.raw`: reference matrix `C_ref` of shape `M x P`

`input0.raw` columns must equal `input1.raw` rows, otherwise the program
reports a dimension mismatch and skips the directory.

## Expected Output

When the computation succeeds the program prints the directory currently being
processed, the matrix dimensions, and a success message for each dataset:

```
Found sub-directory: "/path/ECE408-in-thrust/MP3/data/0"
Processing matrices of size 128x64 and 64x32, dir: "/path/.../0"
Matrix multiplication result matches expected output.
```

## Implementation Details

The current implementation uses:
* `MatrixBuffer` helper to read the `(rows, cols, values...)` files
* `thrust::device_vector<float>` to stage matrices on the GPU
* A hand-written CUDA kernel (`MatMulTiledKernel`) that:
  * decomposes the problem into `16x16` tiles (`kTileDim`)
  * cooperatively loads a tile of `A` and `B` into shared memory per block
  * performs the inner-product using the cached tiles before writing back `C`
* CUDA runtime error checks plus a host-side comparison (tolerance `1e-2`)

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

The comparison tolerance is set to `1e-2` to account for floating-point
precision differences between CPU and GPU computations.
