#include<cstdio>

#include <cooperative_groups.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>

namespace cg = cooperative_groups;

struct Reduce_add
{
    __device__ static int apply(int a, int b) {
        return a + b;
    }
};

template<typename Op, typename T>
__device__ T warp_reduce(T val) {
    for(int i = 32/2; i > 0; i >>= 1) {
        int add = 0xffffffff;
        val = Op::apply(val, __shfl_xor_sync(add, val, i));
    }
    return val;
}
template<typename T>
__device__ T warp_pre(T val, int lane_id) {
    for(int i = 1; i < 32; i <<= 1) {
        int temval = __shfl_up_sync(0xffffffff, val, i);
        if(lane_id >= i) {
            val += temval;
        }
    }
    return val;
}

__device__ void part_sum(float* smem) {
    if(threadIdx.x == 0) {
        float tmp = smem[threadIdx.x];
        for(int i = 1; i < blockDim.x; i++) {
            smem[i] += tmp;
            tmp = smem[threadIdx.x];
        }
    }
    __syncthreads();
}

__global__ void warp_reduce(float* A, float* B, int size) {
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int lane_id = tx % 32;
    int warp_id = tx / 32;
    int gid = bx * blockDim.x + tx;
    float val = gid < size ? A[gid] : 0;
        val = warp_reduce<Reduce_add, float>(val);
    
        if(lane_id == 0) {
            B[bx * blockDim.x + warp_id] = val;
        }
    __syncthreads();

        val = tx < blockDim.x ? B[bx * blockDim.x + tx] : 0;
        val = warp_reduce<Reduce_add, float>(val);
        if(tx == 0) {
            B[0] = val;
        }
    __syncthreads();
}
// template<typename T>
// __device__ T warp_exclusive_scan(T val) {
//     // 获取当前线程在warp内的ID (0-31)
//     int laneId = threadIdx.x % warpSize;

//     // 步骤1: 使用分治法计算包含型前缀和 (Inclusive Scan)
//     // 循环的步长按2的幂次增加: 1, 2, 4, 8, 16
//     #pragma unroll
//     for (int offset = 1; offset < warpSize; offset *= 2) {
//         // 从左边 offset 距离的线程获取值
//         T received_val = __shfl_up_sync(0xffffffff, val, offset);
        
//         // 只有右边的线程需要累加 (避免越界读取)
//         if (laneId >= offset) {
//             val += received_val;
//         }
//     }
//     // 循环结束后，val 中存储的是 Inclusive Scan 的结果

//     // 步骤2: 将 Inclusive Scan 转换为 Exclusive Scan
//     // 每个线程从其左边邻居获取最终结果，实现数据向右平移一位
//     val = __shfl_up_sync(0xffffffff, val, 1);

//     // 第一个线程没有左邻居，其前缀和为0 (单位元)
//     if (laneId == 0) {
//         val = 0;
//     }

//     return val;
// }

__global__ void warp_reduce_2(float* A, float* B, int size) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int lane_id = tx % 32;
    int warp_id = tx / 32;
    int gid = bx * blockDim.x + tx;
    __shared__ float smem[32];
    float val = gid < size ? A[gid] : 0;
    val = warp_reduce<Reduce_add, float>(val);
    if(lane_id == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();
    val = tx < blockDim.x / 32 ? smem[tx] : 0;
    __syncthreads();
    val = warp_reduce<Reduce_add, float>(val);
    // printf("val = %f\n", val);
    if(tx == 0) {
        B[bx] = val;
    }
}
template<int block_size>
__global__ void scan_selfLock(float* A, float* B, float* scan_value, int size, int* block_idx, int* flagBlock) {
    int tx = threadIdx.x;
    __shared__ int bx;
    if(tx == 0) {
        bx = atomicAdd(block_idx, 1);
    }
    // int bx = blockIdx.x;
    __syncthreads();
    int lane_id = tx % 32;
    int warp_id = tx / 32;
    int gid = bx * blockDim.x + tx;
    __shared__ float smem[32];
    __shared__ float block_sum[block_size];
    float val = gid < size ? A[gid] : 0;
    val = warp_pre<float>(val, lane_id);
    if(lane_id == 31) {
        smem[warp_id] = val;
    }
    __syncthreads();

    for(int warp_now = warp_id - 1; warp_now >= 0; warp_now--) {
        val += smem[warp_now];
    }
    __syncthreads();
    // block sync
    // if(tx == blockDim.x - 1) {
    //     scan_value[bx] = val;
    // }
    // __syncthreads();
    __shared__ float presum;
    if(tx == blockDim.x - 1) {
        while(atomicAdd(&flagBlock[bx], 0) == 0) {;
        }
        presum = scan_value[bx];
        scan_value[bx + 1] = presum + val;
        __threadfence();
        atomicAdd(&flagBlock[bx + 1], 1);
    }
    __syncthreads();
    if(gid < size) {
        B[gid] = val + presum;
    }
}
template<int block_size>
__global__ void scan_multikernel(float* A, float* B, int size, float* scan_value) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int gid = bx * blockDim.x + tx;
    int lane_id = tx % 32;
    int warp_id = tx / 32;
    __shared__ float smem[32];
    float val = gid < size ? A[gid] : 0;
    val = warp_pre<float>(val, lane_id);
    if(lane_id == 31) {
        smem[warp_id] = val;
    }
    __syncthreads();
    if(warp_id == 0) {
        float val_warp = smem[lane_id];
        val_warp = warp_pre<float>(val_warp, lane_id);
        smem[lane_id] = val_warp;
    }
    __syncthreads();
    val += warp_id == 0 ? 0: smem[warp_id - 1];
    if(tx == blockDim.x - 1) {
        scan_value[bx + 1] = val;
    }
    if(gid < size) {
        B[gid] = val;
    }
}
template<int block_size>
__global__ void scan_add(float* A, float* B, int size, float* scan_value) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int gid = bx * blockDim.x + tx;
    int lane_id = tx % 32;
    int warp_id = tx / 32;
    __shared__ float presum;
    if(tx == 0) {
        presum = scan_value[bx];
    }
    __syncthreads();
    float val = gid < size ? A[gid] : 0;
    val += presum;
    if(gid < size) {
        B[gid] = val;
    }
}

template<int block_size>
void scan_gpu(float* d_A, float* d_B, int size) {
    dim3 blockDim(block_size);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
    // float* scan_value = (float*)malloc((grid.x + 1) * sizeof(float));
    // float* d_A, *d_B, *d_scan_value;
    float* d_scan_value;
    // cudaMalloc(&d_A, size * sizeof(float));
    // cudaMalloc(&d_B, size * sizeof(float));
    cudaMalloc(&d_scan_value, (gridDim.x + 1) * sizeof(float));
    // cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_B, B, size * sizeof(float), cudaMemcpyHostToDevice);
    scan_multikernel<block_size><<<gridDim, blockDim>>>(d_A, d_B, size, d_scan_value);
    // printf("gridDim.x = %d\n", gridDim.x);
    if(gridDim.x > 1) {
        scan_gpu<block_size>(d_scan_value, d_scan_value, gridDim.x);
        scan_add<block_size><<<gridDim, blockDim>>>(d_B, d_B, size, d_scan_value);
    }

    // cudaMemcpy(B, d_B, size * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(scan_value, d_scan_value, (grid.x + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < grid.x + 1; i++) {
    //     printf("scan_value[%d] = %f\n", i, scan_value[i]);
    // }
    // Free device memory
}

int main() {
    // unsigned int test = 0xffffffff;
    // printf("%x\n", test>>1);
    const int size = 1024 * 1024 ;
    const int block_size = 1024;
    const int grid_size = size / block_size;
    int block_idx = 0;
    int* flagBlock = (int*)malloc(size * sizeof(int));
    float* A = (float*)malloc(size * sizeof(float));
    float* B = (float*)malloc(size * sizeof(float));
    float* scan_value = (float*)malloc((grid_size + 1) * sizeof(float));
    for(int i = 0; i < size; i++) {
        A[i] = 1;
    }
    flagBlock[0] = 1;
    float* d_A, *d_B, *d_scan_value;
    int* d_block_idx, *d_flagBlock;
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMalloc(&d_flagBlock, size * sizeof(int));
    cudaMalloc(&d_scan_value, size * sizeof(float));
    cudaMalloc(&d_block_idx, sizeof(int));
    cudaMemcpy(d_block_idx, &block_idx, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flagBlock, flagBlock, size * sizeof(int), cudaMemcpyHostToDevice);
    dim3 block(block_size);
    dim3 grid((size + block.x - 1) / block.x);
    // scan_selfLock<block_size><<<grid, block>>>(d_A, d_B, d_scan_value, size, d_block_idx, d_flagBlock);
    
    // Call the scan_gpu function
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    scan_gpu<block_size>(d_A, d_B, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("time = %f\n", time);
    cudaMemcpy(B, d_B, size * sizeof(float), cudaMemcpyDeviceToHost);

    // thrust
    float* b_ref = (float*)malloc(size * sizeof(float));
    thrust::device_vector<float> d_A_thrust(A, A + size);
    thrust::device_vector<float> d_B_thrust(size);
    thrust::inclusive_scan(d_A_thrust.begin(), d_A_thrust.end(), d_B_thrust.begin());
    thrust::copy(d_B_thrust.begin(), d_B_thrust.end(), b_ref); 

    // check
    int correct = 1;
    for(int i = 0; i < size; i++) {
        // printf("B[%d] = %f, b_ref[%d] = %f\n", i, B[i], i, b_ref[i]);
        if(B[i] != b_ref[i]) {
            printf("B[%d] = %f, b_ref[%d] = %f\n", i, B[i], i, b_ref[i]);
            correct = 0;
            break;
        }
    }
    if(correct) {
        printf("correct\n");
    } else {
        printf("incorrect\n");
    }
    cudaFree(d_A);
    cudaFree(d_B);
    free(A);
    free(B);
    return 0;
}