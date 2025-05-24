#include <cstdio>
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"

#include "cute/tensor.hpp"

#define baseline 0
#define V0 1
#define warp_size 32

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
__device__ float WarpShuffle(float sum) {
    // warp reduce    
    // 1. 返回前面的thread向后面的thread要的数据，比如__shfl_down_sync(0xffffffff, sum, 16)那就是返回16号线程，17号线程的数据
    // 2. 使用warp shuffle指令的数据交换不会出现warp在shared memory上交换数据时的不一致现象，这一点是由GPU driver完成，故无需任何sync, 比如syncwarp
    // 3. 原先15-19行有5个if判断block size的大小，目前已经被移除，确认了一下__shfl_down_sync等warp shuffle指令可以handle一个block或一个warp的线程数量<32，不足32会自动填充0
        sum += __shfl_down_sync(0xffffffff, sum, 16); 
        sum += __shfl_down_sync(0xffffffff, sum, 8); 
        sum += __shfl_down_sync(0xffffffff, sum, 4); 
        sum += __shfl_down_sync(0xffffffff, sum, 2); 
    // // if(blockDim.x >= 2) {
    //     sum += __shfl_down_sync(0xFFFFFFFF, sum, 1); 
    // }
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
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
// 一共640 T， 控制1025个数据， 
template<int block_size>
__global__ void cuda_reduce_v6(float* A, float* B, float* C, int N) {
    int tidx = threadIdx.x;
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
        // 使用smem 进行加速 smem 需要静态申请
        float temp = 0.0f;
        // for 循环讲N数据读入到每个block内
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
       // block内进行展开求和
        BlockReduce<block_size>(smem, tidx);

        if(tidx == 0) {
            // atomicAdd(C, smem[0]);
            C[blockIdx.x] = smem[0];
        }
}

template<int block_size>
__global__ void cuda_reduce_v7_warpshf(float* A, float* B, float* C, int N) {
    //     float sum = 0;//当前线程的私有寄存器，即每个线程都会拥有一个sum寄存器

    // unsigned int tid = threadIdx.x;
    // unsigned int gtid = blockIdx.x * block_size + threadIdx.x;

    // // 分配的线程总数
    // unsigned int total_thread_num = block_size * gridDim.x;
    // // 基于v5的改进：不用显式指定一个线程处理2个元素，而是通过L30的for循环来自动确定每个线程处理的元素个数
    // for (int i = gtid; i < N; i += total_thread_num)
    // {
    //     sum += A[i];
    // }
    
    // // 用于存储partial sums for each warp of a block
    // __shared__ float WarpSums[block_size / warp_size]; 
    // // 当前线程在其所在warp内的ID
    // const int laneId = tid % warp_size;
    // // 当前线程所在warp在所有warp范围内的ID
    // const int warpId = tid / warp_size; 
    // // 对当前线程所在warp作warpshuffle操作，直接交换warp内线程间的寄存器数据
    // sum = WarpShuffle(sum);
    // if(laneId == 0) {
    //     WarpSums[warpId] = sum;
    // }
    // __syncthreads();
    // //至此，得到了每个warp的reduce sum结果
    // //接下来，再使用第一个warp(laneId=0-31)对每个warp的reduce sum结果求和
    // //首先，把warpsums存入前blockDim.x / WarpSize个线程的sum寄存器中
    // //接着，继续warpshuffle
    // sum = (tid < block_size / warp_size) ? WarpSums[laneId] : 0;
    // // Final reduce using first warp
    // if (warpId == 0) {
    //     sum = WarpShuffle(sum); 
    // }
    // // store: 哪里来回哪里去，把reduce结果写回显存
    // if (tid == 0) {
    //     C[blockIdx.x] = sum;
    // }
    int tidx = threadIdx.x;
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    // 使用smem 进行加速 smem 需要静态申请
    float warp_sum = 0.0f;
    // for 循环讲N数据读入到每个block内
    for(int i = gidx; i < N; i += blockDim.x * gridDim.x) {
        warp_sum += A[i];
    }

    // 把block 内的 每个 warp 展开
    __shared__ float smem[block_size / warp_size];
    // 计算每个warp的和
    /* epoch 1: idx = 1(2):0 ——> 0, 0 + 1; 1 x ; 2 -> 2, 2 + 1; 3x
        epoch 2：idx = 2(4)：0 ——> 0, 0 + 2; 1 x ; 2x ; 3x; 4 -> 4, 
        epoch 3：idx = 4(8)： 
        epoch n： i = i *2
    */
    warp_sum =  WarpShuffle(warp_sum);

    const int widx = tidx / warp_size;
    const int wtidx = tidx % warp_size;
    if(wtidx == 0) {
        smem[widx] = warp_sum;
        printf("warpId = %d, sum = %f, blockIdx.x = %d\n", widx, warp_sum, blockIdx.x);
    }
    __syncthreads();

    // 每个block 内部 求出了每个warpSize 的值， 并存在widx的smem处
    // 再做一次 warp shuffle
    // 计算每个warp的

    if(tidx < (block_size / warp_size)) {
        warp_sum = smem[wtidx];
    }else {
        warp_sum = 0.0f;
    }
    //  前 block_size / warpSize的 warp sum 才有值 xxxxxxxx! 错了 前32个线程做展开
    // warp sum 进行一次warp shuffle 这里只有前  block_size / warp_size 个 thread有值
    // 其他的线程都是0
    if(tidx < 33) {
        warp_sum = WarpShuffle(warp_sum);
    }
    // block内部求和结束 存回HBm
    if(tidx == 0) {
        C[blockIdx.x] = warp_sum;
    }
}

int main() {
    const int N = (1 << 10) + 1;
    printf("N = %d\n", N);
    printf("res is %f\n", float(N) * 10.0f);
    const int nbytes = N * sizeof(float);
    const int block_x = 128;
    // const int grid_x = ceil_div(N, block_x);
    const int grid_x = 5;
    const int block_size = block_x;
    float *h_A, *h_B, *h_C, *res_C;
    float *d_A, *d_B, *d_C, *out_C;
    cudaEvent_t start, stop;
    float time;

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
    // 求和：每个block的结果存入了D_C的第blockIdx.x个位置， 所以对这些位置进行一个reduce 即可。
    cuda_reduce_v7_warpshf<block_size><<<grid, block>>>(d_A, d_B, d_C, N);
    cuda_reduce_v7_warpshf<block_size><<<1, block>>>(d_C, d_B, out_C, grid.x);
    // cuda_reduce_v6<block_size><<<grid, block>>>(d_A, d_B, d_C, N);
    // cuda_reduce_v6<block_size><<<1, block>>>(d_C, d_B, out_C, grid.x);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    printf("gird_x * block_x = %d\n", grid_x * block_x);
    printf("grid.x = %d\n", grid.x);
    printf("block.x = %d\n", block.x);

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