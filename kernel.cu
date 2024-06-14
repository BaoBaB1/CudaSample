#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <iostream>
#include <numeric>
#include <cassert>
#include <chrono>

#define MEASURE_EXEC_TIME(funccall) \
  auto start = std::chrono::high_resolution_clock::now(); \
  funccall; \
  auto stop = std::chrono::high_resolution_clock::now(); \
  std::cout << std::chrono::duration<double, std::milli>(stop - start).count() << '\n';

template<typename T>
__global__ void add1D(T* a, T* b, T* c, int N)
{
  // 1-dim grid blocks:
  // (0, 0), (1, 0), (2, 0) ...
  // or 
  // (0, 0)
  // (0, 1)
  // (0, 2) 
  // ...

  // 2-dim grid blocks:
  // (0, 0), (1, 0), (2, 0) ...
  // (0, 1), (1, 1), (2, 1) ...
  // (0, 2), (1, 2), (2, 2) ...
  // ...

  int i = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) +
    threadIdx.y * blockDim.x +
    threadIdx.x;
  if (i < N) {
    c[i] = a[i] + b[i];
  }
}

template<typename T>
__global__ void add2D(T* a, T* b, T* c, int rows, int cols)
{
  int i = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) +
    threadIdx.y * blockDim.x +
    threadIdx.x;
  if (i < rows * cols) {
    c[i] = a[i] + b[i];
  }
}

template<typename T>
void add1DOnCpu(T* a, T* b, T* c, int N)
{
  for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}

template<typename T>
void add2DOnCpu(T* a, T* b, T* c, int width, int height)
{
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      c[i * width + j] = a[i * width + j] + b[i * width + j];
    }
  }
}

template<typename T, int N>
void testPerformance(const dim3 gridSize, const dim3 threadsPerBlock, const int arrayDim)
{
  T cpuRes = 0;
  T cudaRes = 0;

  std::cout << "Array dimension == " << arrayDim << ", N elems == " << (arrayDim == 1 ? N : N * N) << '\n';

  if (arrayDim == 1)
  {

    // on cpu
    {
      std::vector<T> a(N), b(N), c(N);
      for (int i = 0; i < N; i++) {
        a[i] = i + 1;
        b[i] = (i + 1) * 2;
      }

      std::cout << "Time taken on cpu: ";
      MEASURE_EXEC_TIME(add1DOnCpu(a.data(), b.data(), c.data(), N));
      cpuRes = std::accumulate(c.begin(), c.end(), 0);
    }

    // on cuda
    {
      std::vector<T> a(N), b(N), c(N);
      const size_t bytes = sizeof(T) * N;
      for (int i = 0; i < N; i++) {
        a[i] = i + 1;
        b[i] = (i + 1) * 2;
      }

      T* da, * db, * dc;
      cudaMalloc(&da, bytes);
      cudaMalloc(&db, bytes);
      cudaMalloc(&dc, bytes);

      cudaMemcpy(da, a.data(), bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);
      cudaMemcpy(db, b.data(), bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);

      std::cout << "Time taken on gpu: ";
      MEASURE_EXEC_TIME((add1D<<<gridSize, threadsPerBlock>>>(da, db, dc, N)));
      std::cout << '\n';

      cudaMemcpy(c.data(), dc, bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost);
      cudaRes = std::accumulate(c.begin(), c.end(), 0);

      cudaFree(da);
      cudaFree(db);
      cudaFree(dc);
    }
  }
  else if (arrayDim == 2)
  {

    // on cpu
    {
      std::vector<T> a(N * N);
      std::vector<T> b(N * N);
      std::vector<T> c(N * N);
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          a[i * N + j] = i * N + j; 
          b[i * N + j] = std::pow(i * N + j, 2);
        }
      }

      std::cout << "Time taken on cpu: ";
      MEASURE_EXEC_TIME(add2DOnCpu(a.data(), b.data(), c.data(), N, N));
      cpuRes = std::accumulate(c.begin(), c.end(), 0);
    }

    // on cuda
    {
      std::vector<T> a(N * N), b(N * N), c(N * N);
      const size_t bytes = sizeof(T) * N * N;
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          a[i * N + j] = i * N + j;
          b[i * N + j] = std::pow(i * N + j, 2);
        }
      }

      T* da, * db, * dc;
      cudaMalloc(&da, bytes);
      cudaMalloc(&db, bytes);
      cudaMalloc(&dc, bytes);

      cudaMemcpy(da, a.data(), bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);
      cudaMemcpy(db, b.data(), bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);

      std::cout << "Time taken on gpu: ";
      MEASURE_EXEC_TIME((add2D<<<gridSize, threadsPerBlock>>>(da, db, dc, N, N)));
      std::cout << '\n';

      cudaMemcpy(c.data(), dc, bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost);
      cudaRes = std::accumulate(c.begin(), c.end(), 0);

      cudaFree(da);
      cudaFree(db);
      cudaFree(dc);
    }
  }
  else
  {
    throw std::runtime_error("Not implemented\n");
  }

  if (cpuRes != cudaRes) {
    std::cout << "Array dim = " << arrayDim << ", N = " << N << ", cpuRes = " 
      << cpuRes << ", cudaRes = " << cudaRes << '\n';
  }
  assert(cpuRes == cudaRes);
}

int main()
{

  {
    // 1 row, 2 cols, 1D array
    dim3 gridSize(2, 1);
    dim3 threadsPerBlock(16, 16);
    testPerformance<int, 512>(gridSize, threadsPerBlock, 1);
  }

  {
    // 4 rows, 1 col, 1D array
    dim3 gridSize(1, 4);
    dim3 threadsPerBlock(16, 16);
    testPerformance<int, 1024>(gridSize, threadsPerBlock, 1);
  }

  {
    // 14 rows, 15 cols, 1D array
    dim3 gridSize(15, 14);
    dim3 threadsPerBlock(16, 16);
    testPerformance<size_t, 50000>(gridSize, threadsPerBlock, 1);
  }

  {
    // 14 rows, 15 cols, 1D array
    dim3 gridSize(30, 30);
    dim3 threadsPerBlock(24, 24);
    testPerformance<size_t, 500000>(gridSize, threadsPerBlock, 1);
  }

  {
    // 1 row, 3 cols, 2D array
    dim3 gridSize(3, 1);
    dim3 threadsPerBlock(12, 12);
    testPerformance<size_t, 16>(gridSize, threadsPerBlock, 2);
  }

  {
    // 22 rows, 22 cols, 2D array
    dim3 gridSize(22, 22);
    dim3 threadsPerBlock(24, 24);
    testPerformance<size_t, 512>(gridSize, threadsPerBlock, 2);
  }
}
