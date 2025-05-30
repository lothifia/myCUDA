#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

template<typename T, int size>
struct alignas(sizeof(T) * size) Alignment {
    T vec[size];
};

template<typename T>
struct AddOp {
    __device__ AddOp() {}
    __device__ T operator() (const T &a, const T &b) {
        return a + b;
    }
};

template<typename T>
struct MaxOp {
    __device__ MaxOp() {}
    __device__ T operator() (const T &a, const T &b) {
        return max(a, b);
    }
};
template<typename T>
constexpr __inline__ __device__ T Inf() 
{
    /* data */
    return 0x3f3f3f3f;
};

template<typename T>
__inline__ __device__ T Exp(const T& val);

template<>
__inline__ __device__ float Exp<float>(const float& val) {
    return exp(val);
}

template<typename T>
__inline__ __device__ T Div(const T& val, const T& val2);

__inline__ __device__ float Div(const float& val, const float& val2) {
    return val / val2;
}


template<template<typename> class OpFunc, typename T, int warp_size = 32>
__inline__ __device__ T ReduceWarp(T var) {
    for(int mask = warp_size / 2; mask > 0; mask /= 2) {
        var = OpFunc<T>()(var,  __shfl_xor_sync(uint(-1), var, mask));
    }
    return var;
}
__global__ void SoftmaxKernel() {
    printf("%d \n", threadIdx.x);
}
template< typename T, int vec_size>
__device__ void load(T* dst, T* src) {
    using arrt =  Alignment<T, vec_size>;

            // if(blockIdx.x == 0 && threadIdx.x == 0)
            //     printf("%f ", src[0]);
    
    *reinterpret_cast<arrt*>(dst) = *reinterpret_cast<arrt*>(src);
//             if(blockIdx.x == 0 && threadIdx.x == 31)
//                 printf("%f ", dst[7]);
}
//warp level
/*
超大 M N 矩阵
 一个warp处理一行矩阵：32T 处理 N个数据， 每行需要 N / 32 epoch。 
一个block 有 blocksize / warp 个w 行， 共gridDim.x个block， 共有gx* w 行， M行数据， 每个warp需要重复处理M /（gx*w）次， 每次stride 为: gx * w

*/
template<typename T, int vec_size, int block_size, int warp_size = 32, int data_per_t, int row_per_t>
__global__ void SoftmaxKernel(T* A, T* B, T* C, int n, int M, int N) {
    int iy = threadIdx.x % warp_size;
    int ix = threadIdx.x / warp_size;

    int pack_size = N / (warp_size * vec_size);
    int gird_row_stride = (block_size / warp_size) * gridDim.x;
    T tReg[data_per_t];
    for(int row =  (blockDim.x / warp_size) * blockIdx.x + ix; row < M; row += gird_row_stride) {
        T row_max = -Inf<T>();
        for(int pack_idx = 0; pack_idx < pack_size; ++pack_idx) {
            int col = vec_size * warp_size * pack_idx + vec_size * iy;
            T * t_offest = reinterpret_cast<T*>( tReg + pack_idx * vec_size);
            T * A_offest = reinterpret_cast<T*>( A + row * N + col);
                    // printf("%d %d ", col, pack_idx);
            if(col < N) {
                load<T, vec_size> (t_offest, A_offest);
                for(int i = 0; i < vec_size; i++) {
                    row_max = max(row_max, t_offest[i]);
                }
            }
            else {
                for(int i = 0; i < vec_size; i++) {
                    t_offest[i] = -Inf<T>();
                }
            }
        }
        // for(int i = 0; i < data_per_t; i++) {
        //             if(blockIdx.x == 0 && threadIdx.x == 0) {
        //                 printf(" load %f\n", tReg[i]);
        //             }

        // }
        const T warp_max = ReduceWarp<MaxOp, T, warp_size>(row_max);
                
        T thread_sum = 0;
        for(int i = 0; i < data_per_t; i++) {
            tReg[i] = Exp(tReg[i] - warp_max);
            thread_sum += tReg[i];
        }
        const T warp_sum = ReduceWarp<AddOp, T, warp_size>(thread_sum);
        
        for(int i = 0; i < data_per_t; i++) {
            tReg[i] = Div(tReg[i], warp_sum);
        }
        for(int pack_idx = 0; pack_idx < pack_size; ++pack_idx) {
            int col = vec_size * warp_size * pack_idx + vec_size * iy;
            T * t_offest = reinterpret_cast<T*>( tReg + pack_idx * vec_size);
            T * B_offest = reinterpret_cast<T*>( B + row * N + col);
            
                    if(blockIdx.x == 8 && threadIdx.x == 31) {
                    // printf("%d %d ", col, pack_idx);
                    printf(" %f , of %d, %d", t_offest[1], pack_idx * vec_size, col);
                    }
            if(col < N) {
                load<T, vec_size>(B_offest, t_offest);
                    if(blockIdx.x == 4 && threadIdx.x == 31) {
                        printf(" warp_sum %f\n", B_offest[1]);
                    }

            }
        }
    }
}

int main() {
    // int deviceCount = 0;
    // cudaGetDeviceCount(&deviceCount);
    // if (deviceCount == 0) {
    //     printf("No CUDA devices found!\n");
    // }
    using element_type = float;
    element_type* h_a, *h_b, *h_c;
    element_type* d_a, *d_b, *d_c;

    constexpr int block_size = 256;
    constexpr int warp_size = 32;
    constexpr int ver_size = 8;
    constexpr int M = 1024, N = 1024;
    constexpr int data_per_t = N / warp_size;
    constexpr int grid_size = M * N / ver_size /block_size; 
    int n = M * N;

    dim3 block(block_size);
    dim3 grid(n / (ver_size * block_size));
    constexpr int row_per_t = (M + grid_size* warp_size -1) / (grid_size * warp_size);


    h_a = (element_type*)malloc(n * sizeof(element_type));
    h_b = (element_type*)malloc(n * sizeof(element_type));
    h_c = (element_type*)malloc(n * sizeof(element_type));
    
    cudaMalloc((void**)&d_a, n * sizeof(element_type));
    cudaMalloc((void**)&d_b, n * sizeof(element_type));
    cudaMalloc((void**)&d_c, n * sizeof(element_type));

    for(int i = 0; i < n; i ++) {
        h_a[i] = static_cast<element_type>(i);
        // printf("%f, ", h_a[i]);
        h_c[i] = static_cast<element_type>(0);
    }

    cudaMemcpy(d_a, h_a, n * sizeof(element_type), cudaMemcpyHostToDevice);
    printf("grid  block %d, %d, %d \n", grid.x, block.x, block.y);

    SoftmaxKernel<element_type, ver_size, block_size, warp_size, data_per_t, row_per_t><<<grid, block>>>(d_a, d_b, d_c, n, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();  // 等待所有 GPU 操作完成
    cudaMemcpy(h_b, d_b,  n * sizeof(element_type), cudaMemcpyDeviceToHost);
    for(int i = 0; i < M; i++) {

        float test = 0;
        for(int j = 0; j < N ; j++) {
            test += h_b[i * N + j];
        }
        printf("%f \n", test);
    }

    return 0;
}





