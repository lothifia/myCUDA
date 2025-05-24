#define baseline 0
#define V0 1
#include <cstdio>
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"

#include "cute/tensor.hpp"

using namespace cute;

__device__ void WarpReduce(volatile float* smem, int tidx) {
    // warp reduce
    float x = smem[tidx];
    if(blockDim.x >= 64) {
        x += smem[tidx + 32]; __syncwarp(); //通过warp 同步保证前后顺序 
        smem[tidx] = x; __syncwarp();
    }
    x += smem[tidx + 16]; __syncwarp();
    smem[tidx] = x; __syncwarp();
    x += smem[tidx + 8]; __syncwarp();
    smem[tidx] = x; __syncwarp();
    x += smem[tidx + 4]; __syncwarp();
    smem[tidx] = x; __syncwarp();
    x += smem[tidx + 2]; __syncwarp();
    smem[tidx] = x; __syncwarp();
    x += smem[tidx + 1]; __syncwarp();
    smem[tidx] = x; __syncwarp();
    // print("hi");

}

template<int block_size>
__device__ void BlockReduce(volatile float* smem, int tidx) {
    // warp reduce
    if(block_size >= 1024) {
        if(threadIdx.x < 512) {
            smem[tidx] += smem[tidx + 512];
        }
        __syncthreads();
    }
    if(block_size >= 512) {
        if(threadIdx.x < 256) {
            smem[tidx] += smem[tidx + 256];
        }
        __syncthreads();
    }
    if(block_size >= 256) {
        if(threadIdx.x < 128) {
            smem[tidx] += smem[tidx + 128];
        }
        __syncthreads();
    }
    if(block_size >= 128) {
        if(threadIdx.x < 64) {
            smem[tidx] += smem[tidx + 64];
        }
        __syncthreads();
    }

    // 看32线程
    float x = smem[tidx];
    if(threadIdx.x < 32) {
        if(block_size >= 64) {
            x += smem[tidx + 32]; __syncwarp(); //通过warp 同步保证前后顺序 
            smem[tidx] = x; __syncwarp();
        }
        x += smem[tidx + 16]; __syncwarp();
        smem[tidx] = x; __syncwarp();
        x += smem[tidx + 8]; __syncwarp();
        smem[tidx] = x; __syncwarp();
        x += smem[tidx + 4]; __syncwarp();
        smem[tidx] = x; __syncwarp();
        x += smem[tidx + 2]; __syncwarp();
        smem[tidx] = x; __syncwarp();
        x += smem[tidx + 1]; __syncwarp();
        smem[tidx] = x; __syncwarp();
    }
}


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
        for(int idx = blockDim.x / 2; idx > 0; idx = idx >> 1) {
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
template<int block_size>
__global__ void cuda_reduce_v3(float* A, float* B, float* C, int N) {
    int tidx = threadIdx.x;
    int gidx = blockDim.x * blockIdx.x * 2+ threadIdx.x;
    if(gidx < N) {
        // 使用smem 进行加速 smem 需要静态申请
        __shared__ float smem[block_size];
        if(gidx + blockDim.x < N) {
            smem[tidx] = A[gidx] + A[gidx + blockDim.x];
        } else {
            smem[tidx] = A[gidx];
        }
        __syncthreads();
        /* epoch 1: idx = 1(2): 0 ——> 0, 0 + 1; 1 x ; 2 -> 2, 2 + 1; 3x
           epoch 2：idx = 2(4)： 0 ——> 0, 0 + 2; 1 x ; 2x ; 3x; 4 -> 4, 
           epoch 3：idx = 4(8)： 
           epoch n： i = i *2
        */
        for(int idx = blockDim.x / 2; idx > 0; idx = idx >> 1) {
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
template<int block_size>
__global__ void cuda_reduce_v4(float* A, float* B, float* C, int N) {
    int tidx = threadIdx.x;
    int gidx = blockDim.x * blockIdx.x * 2+ threadIdx.x;
    if(gidx < N) {
        // 使用smem 进行加速 smem 需要静态申请
        __shared__ float smem[block_size];
        if(gidx + blockDim.x < N) {
            smem[tidx] = A[gidx] + A[gidx + blockDim.x];
        } else {
            smem[tidx] = A[gidx];
        }
        __syncthreads();
        /* epoch 1: idx = 1(2): 0 ——> 0, 0 + 1; 1 x ; 2 -> 2, 2 + 1; 3x
           epoch 2：idx = 2(4)： 0 ——> 0, 0 + 2; 1 x ; 2x ; 3x; 4 -> 4, 
           epoch 3：idx = 4(8)： 
           epoch n： i = i *2
        */
        for(int idx = blockDim.x / 2; idx > 32; idx = idx >> 1) {
            if(tidx < idx) {
                smem[tidx] += smem[tidx + idx];
            }
            __syncthreads();
        }

        if(tidx < 32) {
            WarpReduce(smem, tidx);
        }

        if(tidx == 0) {
            C[blockIdx.x] = smem[0];
        }
    }
}

template<int block_size>
__global__ void cuda_reduce_v5(float* A, float* B, float* C, int N) {
    int tidx = threadIdx.x;
    int gidx = blockDim.x * blockIdx.x * 2+ threadIdx.x;
    if(gidx < N) {
        // 使用smem 进行加速 smem 需要静态申请
        __shared__ float smem[block_size];
        if(gidx + blockDim.x < N) {
            smem[tidx] = A[gidx] + A[gidx + blockDim.x];
        } else {
            smem[tidx] = A[gidx];
        }
        __syncthreads();
        /* epoch 1: idx = 1(2):0 ——> 0, 0 + 1; 1 x ; 2 -> 2, 2 + 1; 3x
           epoch 2：idx = 2(4)：0 ——> 0, 0 + 2; 1 x ; 2x ; 3x; 4 -> 4, 
           epoch 3：idx = 4(8)： 
           epoch n： i = i *2
        */
        BlockReduce<block_size>(smem, tidx);

        if(tidx == 0) {
            C[blockIdx.x] = smem[0];
            // atomicAdd(C[0], smem[0]);
        }
    }
}

// 线程数远小于N
template<int block_size>
__global__ void cuda_reduce_v6(float* A, float* B, float* C, int N) {
    int tidx = threadIdx.x;
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
        // 使用smem 进行加速 smem 需要静态申请
        float temp = 0.0f;
        for(int i = gidx; i < N; i += blockDim.x * gridDim.x) {
            temp += A[i];
        }
        __shared__ float smem[block_size];
        smem[tidx] = temp;
        
        __syncthreads();
        /* epoch 1: idx = 1(2):0 ——> 0, 0 + 1; 1 x ; 2 -> 2, 2 + 1; 3x
           epoch 2：idx = 2(4)：0 ——> 0, 0 + 2; 1 x ; 2x ; 3x; 4 -> 4, 
           epoch 3：idx = 4(8)： 
           epoch n： i = i *2
        */
        BlockReduce<block_size>(smem, tidx);

        if(tidx == 0) {
            // atomicAdd(C, smem[0]);
            C[blockIdx.x] = smem[0];
        }
}

int main() {
    const int N = (1 << 19) + 1;
    printf("N = %d\n", N);
    printf("res is %f\n", float(N) * 10.0f);
    const int nbytes = N * sizeof(float);
    const int block_x = 128;
    const int grid_x = ceil_div(N, block_x);
    const int block_size = block_x;
    float *h_A, *h_B, *h_C, *res_C;
    float *d_A, *d_B, *d_C, *out_C;
    cudaEvent_t start, stop;
    float time;
    printf("gird_x * block_x = %d\n", grid_x * block_x);

    h_A = (float *)malloc(nbytes);
    h_B = (float *)malloc(nbytes);
    h_C = (float *)malloc(grid_x * block_x * sizeof(float));
    res_C = (float*)malloc(1 * sizeof(float));
    cudaMalloc((void**)&d_A, nbytes);
    cudaMalloc((void**)&d_B, nbytes);
    cudaMalloc((void**)&d_C, grid_x* block_x * sizeof(float));
    cudaMalloc((void**)&out_C, 1 * sizeof(float));

    #if baseline
    dim3 block(1, 1, 1);
    dim3 grid(1, 1, 1);
    #endif

    #if V0
    dim3 block(block_size, 1, 1);
    dim3 grid(grid_x, 1, 1);
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
    // cuda_reduce_v2<block_size><<<grid, block>>>(d_A, d_B, d_C, N);
    // cuda_reduce_v2<block_size><<<grid, block>>>(d_A, d_B, d_C, N);
    cuda_reduce_v6<block_size><<<grid, block>>>(d_A, d_B, d_C, N);
    cuda_reduce_v6<block_size><<<1, block>>>(d_C, d_B, out_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaMemcpy(h_C, d_C, grid_x * block_x * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(res_C, out_C, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time: %f ms\n", time);
    for(int i = 0; i < grid.x; i++ ) {
        printf("h_C[%d] = %f", i, h_C[i]);
        printf("\n");
    }
    printf("res_C = %f\n", res_C[0]);
}