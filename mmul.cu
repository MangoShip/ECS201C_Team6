// This program computes a simple version of matrix multiplication
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <chrono>

using std::cout;
using std::generate;
using std::vector;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void matrixMul(const int *a, const int *b, int *c, size_t N) {
  // Compute each thread's global row and column index
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  // Iterate over row, and down column
  c[row * N + col] = 0;
  for (size_t k = 0; k < N; k++) {
    // Accumulate results for a single element
    c[row * N + col] += a[row * N + k] * b[k * N + col];
  }
}

// Check result on the CPU
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c, int N) {
  // For every row...
  for (int i = 0; i < N; i++) {
    // For every column...
    for (int j = 0; j < N; j++) {
      // For every element in the row-column pair
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        // Accumulate the partial results
        tmp += a[i * N + k] * b[k * N + j];
      }

      // Check against the CPU result
      assert(tmp == c[i * N + j]);
    }
  }
}

int main() {
  // Matrix size of 1024 x 1024;
  size_t N = 1 << 16;

  // Size (in bytes) of matrix
  size_t bytes = N * N * sizeof(int);

  // Host vectors
  vector<int> h_a(N * N);
  vector<int> h_b(N * N);
  vector<int> h_c(N * N);

  // Initialize matrices
  generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
  generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

  // Allocate device memory
  int *d_a, *d_b, *d_c;
  gpuErrchk(cudaMalloc(&d_a, bytes));
  gpuErrchk(cudaMalloc(&d_b, bytes));
  gpuErrchk(cudaMalloc(&d_c, bytes));

  // Copy data to the device
  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

  // Threads per CTA dimension
  int THREADS = 32;

  // Blocks per grid dimension (assumes THREADS divides N evenly)
  int BLOCKS = N / THREADS;

  // Use dim3 structs for block  and grid dimensions
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  std::cout << "Kernel starting:\n";
  auto start = std::chrono::steady_clock::now();

  // Launch kernel
  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N);

  // Copy back to the host
  gpuErrchk(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
  
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> time = end - start;
  cout << "Kernel Duration: " << time.count() << "\n";

  // Check result
  //verify_result(h_a, h_b, h_c, N);

  cout << "COMPLETED SUCCESSFULLY\n";

  // Free memory on device
  gpuErrchk(cudaFree(d_a));
  gpuErrchk(cudaFree(d_b));
  gpuErrchk(cudaFree(d_c));

  return 0;
}