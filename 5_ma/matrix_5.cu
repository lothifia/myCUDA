#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include "myhead.h"
// #define BLOCKSIZE 32
template <const int BM, const int BK, const int BN, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1) mul_shared_mem_2d(float* A, float* B, float* C, int M, int K, int N) {

    int block_row = blockIdx.x; // block row
    int block_col = blockIdx.y; // block col    
    int thread_Row = threadIdx.x / (BN / TN); // thread
    int thread_Col = threadIdx.x % (BN / TN /* BN / TN -> thread"block"*/);
    __shared__ float s_A[BM * BK];
    __shared__ float s_B[BK * BN];
    float thread_TM[TM * TN]{0.0};

    const int totalResPerBlock = BM * BN;
    const int numThreadBlocktile = totalResPerBlock / (TM * TN);
    // printf("totalResPerBlock %d numThreadBlocktile %d \n", totalResPerBlock, numThreadBlocktile);
    assert(numThreadBlocktile == blockDim.x);
    /*per block is a (16 * 16) size block with threads, in fact is a 256 vector, we need to use this 256 thread deal 128 * 8 size data,
    means per thread execute 128 * 8 / 256 = 4 data. the numThreadBlocktile means 256 thread .
    nTBt/ BK = 256 / 8 = 32 means each threads the gap between data per thread exec in A.
    same on B. 
    example in A: 
    |1 the data map to block  ********
    |2 the data map to block  ********
    | .......
    |32 the data map to block ********
    |the data cope in loop    --------
    |the data cope in loop    --------
    |the data cope in loop    --------
    |the data cope in loop    --------
    */
    const int strideA = numThreadBlocktile / BK;
    const int innerAx = threadIdx.x / BK;
    const int innerAy = threadIdx.x % BK;
    const int strideB = numThreadBlocktile / BN;
    const int innerBx = threadIdx.x / BN;
    const int innerBy = threadIdx.x % BN;

    // printf("strideA %d innerAx %d innerAy %d strideB %d innerBx %d innerBy %d \n", strideA, innerAx, innerAy, strideB, innerBx, innerBy);
    float regM[TM]{0.0};
    float regN[TN]{0.0};
    A += block_row * BM * K;
    B += block_col * BN;
    C += block_row * BM * N + block_col * BN;
    for(int block_idx = 0; block_idx < K; block_idx += BK) {
        for(int row_offest = 0; row_offest < BM; row_offest += strideA) {
            s_A[innerAy + (innerAx + row_offest) * BK] = A[innerAy + (innerAx + row_offest) * K];
        }
        for(int row_offest = 0; row_offest < BK; row_offest += strideB) {
            s_B[innerBy + (innerBx + row_offest) * BN] = B[innerBy + (innerBx + row_offest) * N];
        }
        __syncthreads();
        A += BK;
        B += BK * N;
        /*each thread should exec 8 * 8 = 64 data. so there is a regM and regN , per out loop deal 64 times data single fma*/
        for(int idx_BK = 0; idx_BK < BK; idx_BK++) {
            for(int j = 0; j < TM; ++j) {
                regM[j] = s_A[idx_BK +  (thread_Row * TM + j) * BK];
            }
            for(int j = 0; j < TN; ++j) {
                regN[j] = s_B[idx_BK * BN + TN * thread_Col + j];       
            }
            for(int i = 0; i < TM; ++i) {
                for(int j = 0; j < TN; ++j) {
                    thread_TM[i * TN + j] += regM[i] * regN[j];
                }
            }
        }
        __syncthreads();
    }

    for(int i = 0; i < TM; ++i) {
        for(int j = 0; j < TN; ++j) {
            C[(thread_Row * TM + i) * N + thread_Col * TN + j] = thread_TM[i * TN + j];
        }
    }
}

int main() {
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    printf("Using device %d: %s %d\n",dev,deviceProp.name, deviceProp.warpSize);
    printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    CHECK(cudaSetDevice(dev));

    size_t nx_A = 1 << 12;
    // size_t nx_B = 1 << 13;
    size_t ny_A = 1 << 12;
    // size_t ny_B = 1 << 13;
    size_t nxy_A = nx_A * ny_A; 
    size_t nByte_A = nx_A * ny_A * sizeof(float);
    // size_t totByte_B = nx_B * ny_B * sizeof(float);
    const int TM = 8;
    const int TN = 8;
    const int BK = 8;
    const int BM = 128;
    const int BN = 128;
    dim3 gridDim(CEIL_DIV(nx_A, BM), CEIL_DIV(ny_A, BN));
    dim3 blockDim((BM * BN) / (TM * TN));
    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    float* h_A = (float* )malloc(nByte_A);
    float* h_B = (float* )malloc(nByte_A);
    float* h_C = (float* )malloc(nByte_A);
    float* d_A = NULL;
    float* d_B = NULL;
    float* d_C = NULL;
    CHECK(cudaMalloc((void**)&d_A, nByte_A));
    CHECK(cudaMalloc((void**)&d_B, nByte_A));
    CHECK(cudaMalloc((void**)&d_C, nByte_A));
    

    for(int i = 0; i < nxy_A; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    double sTime = cpuSecond();
    CHECK(cudaMemcpy(d_A, h_A, nByte_A, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nByte_A, cudaMemcpyHostToDevice));

    // cudaFuncAttribute(mul_shared_mem<32>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    cudaEventRecord(beg);
    mul_shared_mem_2d<BM, BK, BN, TM, TN><<<gridDim, blockDim>>> (d_A, d_B, d_C, nx_A, nx_A, ny_A);
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end); 
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1.0e3f;
    long flops = 2 * nx_A * ny_A * nx_A;
    printf("elapsed time %f s, TFLOPS %f\n", elapsed_time, flops / elapsed_time / 1.0e12);

    CHECK(cudaMemcpy(h_C, d_C, nByte_A, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());
    printf("over \n");
    // for(int i = 0; i < nxy_A / nx_A; i++) {
    //     printf("%f ", h_C[i]);
    //     if(i % nx_A == nx_A - 1) {
    //         printf("\n");
    //     }
    // }
    // for(int i = 0; i < nxy_A; i++) {
    //     printf("%f ", h_A[i]);
    //     if(i % nx_A == nx_A - 1) {
    //         printf("\n");
    //     }
    // }
    // for(int i = 0; i < nxy_A; i++) {
    //     printf("%f ", h_B[i]);
    //     if(i % nx_A == nx_A - 1) {
    //         printf("\n");
    //     }
    // }
    return 0;
}