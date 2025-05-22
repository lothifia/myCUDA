#define baseline 0
#define V0 1
#include <cstdio>
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"

#include "cute/tensor.hpp"

using namespace cute;
__global__ void cuda_reduce(float* A, float* B, float* C, int N) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    float sum = 0.0f;
    for(int i = 0; i < N; i++) {
        sum += A[i];
    }
    *C = sum;
}

template<int block_size>
__global__ void cuda_reduce_v0(float* A, float* B, float* C, int N) {
    int tidx = threadIdx.x;
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    if(gidx < N) {
        // 使用smem 进行加速 smem 需要静态申请
        __shared__ float smem[block_size];
        smem[tidx] = A[gidx];
        __syncthreads();
        /* epoch 1: idx = 1(2): 0 ——> 0, 0 + 1; 1 x ; 2 -> 2, 2 + 1; 3x
           epoch 2：idx = 2(4)： 0 ——> 0, 0 + 2; 1 x ; 2x ; 3x; 4 -> 4, 
           epoch 3：idx = 4(8)： 
           epoch n： i = i *2
        */
        for(int idx = 1; idx < blockDim.x; idx *= 2) {
            if(tidx % (idx * 2) == 0) {
                smem[tidx] += smem[tidx + idx];
            }// warp divergence : warp中一部分是if 另一部分是另一部分 就出现了warp divergence
            // int i = 2 * idx * tidx;
            // if( i < blockDim.x) {
            //     smem[i] += smem[i + idx];
            // }
            __syncthreads();
        }
        if(tidx == 0) {
            C[blockIdx.x] = smem[0];
        }
    }
}
template<int block_size>
__global__ void cuda_reduce_v1(float* A, float* B, float* C, int N) {
    int tidx = threadIdx.x;
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    if(gidx < N) {
        // 使用smem 进行加速 smem 需要静态申请
        __shared__ float smem[block_size];
        smem[tidx] = A[gidx];
        __syncthreads();
        /* epoch 1: idx = 1(2): 0 ——> 0, 0 + 1; 1 x ; 2 -> 2, 2 + 1; 3x
           epoch 2：idx = 2(4)： 0 ——> 0, 0 + 2; 1 x ; 2x ; 3x; 4 -> 4, 
           epoch 3：idx = 4(8)： 
           epoch n： i = i *2
        */
        for(int idx = 1; idx < blockDim.x; idx *= 2) {
            // if(tidx % (idx * 2) == 0) {
            //     smem[tidx] += smem[tidx + idx];
            // }// warp divergence : warp中一部分是if 另一部分是另一部分 就出现了warp divergence
            // reduce 1 no div and mod
            int i = 2 * idx * tidx;
            if( i < blockDim.x) {
                smem[i] += smem[i + idx];
            }
            __syncthreads();
        }
        if(tidx == 0) {
            C[blockIdx.x] = smem[0];
        }
    }
}


template<int block_size>
__global__ void cuda_reduce_v2(float* A, float* B, float* C, int N) {
    int tidx = threadIdx.x;
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    if(gidx < N) {
        // 使用smem 进行加速 smem 需要静态申请
        __shared__ float smem[block_size];
        smem[tidx] = A[gidx];
        __syncthreads();
        /* epoch 1: idx = 1(2): 0 ——> 0, 0 + 1; 1 x ; 2 -> 2, 2 + 1; 3x
           epoch 2：idx = 2(4)： 0 ——> 0, 0 + 2; 1 x ; 2x ; 3x; 4 -> 4, 
           epoch 3：idx = 4(8)： 
           epoch n： i = i *2
        */
        for(int idx = blockDim.x / 2; idx > 0; idx = idx >> 2) {
            if(tidx < idx) {
                smem[tidx] += smem[tidx + idx];
            }
            __syncthreads();
        }
        if(tidx == 0) {
            C[blockIdx.x] = smem[0];
        }
    }
}

int main() {
    const int N = (1 << 10) + 1;
    const int nbytes = N * sizeof(float);
    const int grid_x = 1;
    const int block_x = 128;
    const int block_size = block_x;
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    cudaEvent_t start, stop;
    float time;


    h_A = (float *)malloc(nbytes);
    h_B = (float *)malloc(nbytes);
    h_C = (float *)malloc(grid_x * block_x * sizeof(float));
    cudaMalloc((void**)&d_A, nbytes);
    cudaMalloc((void**)&d_B, nbytes);
    cudaMalloc((void**)&d_C, grid_x* block_x * sizeof(float));

    #if baseline
    dim3 block(1, 1, 1);
    dim3 grid(1, 1, 1);
    #endif

    #if V0
    dim3 block(block_size, 1, 1);
    dim3 grid(ceil_div(N, block_size), 1, 1);
    #endif

    for(int i = 0; i < N; i++) {
        h_A[i] = float(10);
        h_B[i] = float(10);
    }

    for(int i = 0; i < grid_x * block_x; i++) {
        h_C[i] = 0.0f;
    }

    cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nbytes, cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // cuda_reduce<<<grid, block>>>(d_A, d_B, d_C, N);
    // cuda_reduce_v0<block_size><<<grid, block>>>(d_A, d_B, d_C, N);
    cuda_reduce_v2<block_size><<<grid, block>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaMemcpy(h_C, d_C, grid_x * block_x * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time: %f ms\n", time);
    for(int i = 0; i < grid.x; i++ ) {
        printf("h_C[%d] = %f", i, h_C[i]);
        printf("\n");
    }
}