#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <cublas_v2.h>
#include "myhead.h"
template<const int Br, const int Bc, const int TM, const int TN>
__global__ void my_FA_kernel(
    float* Q,
    float* K,
    float* V,
    float* D,
    int M,
    int N,
    int d,
    float alpha,
    float beta) {
        __shared__ float Qi[Br * d];
        __shared__ float Kj[Bc * d];
        
        
  // Kernel implementation here
  // This is a placeholder for the actual kernel code
  // You would typically perform matrix multiplication or other operations here
  // using the input matrices A, B, and C.
}