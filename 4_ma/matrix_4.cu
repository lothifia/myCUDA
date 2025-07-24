#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include "myhead.h"
// #define BLOCKSIZE 32
template <const int AM, const int AK, const int BN, const int tmp>
__global__ void mul_shared_mem_1d(float* d_A, float* d_B, float* d_C, size_t nx, size_t nk, size_t ny) {
    size_t cCol = blockIdx.y; // block.y
    size_t cRow = blockIdx.x; // block.x
    size_t tCol = threadIdx.x % BN; // 1. row first ! 2. "mapping a one-dimensional index to a two-dimensional coordinate"
    size_t tRow = threadIdx.x / BN; // same on
    printf("tCOl is%d \n", tCol);

    d_A += cRow * AK * nk; // the block first row
    d_B += cCol * BN; // the block first col
    d_C += cRow * BN * ny + cCol * BN; // the target block fisrt thread

    __shared__ float s_A[AM * AK];
    __shared__ float s_B[AK * BN];

    assert(AM * AK == blockDim.x);
    assert(AK * BN == blockDim.x);
    const int innerAx = threadIdx.x / AK;
    const int innerAy = threadIdx.x % AK;
    const int innerBx = threadIdx.x / BN;
    const int innerBy = threadIdx.x % BN;
    float threadTmp[tmp]{0.0};
    for(int block_i = 0; block_i < nk; block_i += AK) {       
        s_A[innerAx * AK + innerAy] = d_A[innerAx * nk + innerAy];
        s_B[innerBx * BN + innerBy] = d_B[innerBx * ny + innerBy];
        // 512 one time
        __syncthreads();
        d_A += AK; // next block in A
        d_B += AK * ny; // next block in B
        for(int i = 0; i < AK; ++ i) {
            float tmpB = s_B[i * BN + innerBy];
            for (int j = 0; j < tmp; ++ j) {
                threadTmp[j] += s_A[(tRow * tmp + j ) * AK + i] * tmpB;
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < tmp; ++ i) {
        d_C[(tRow * tmp + i) * ny + tCol] = threadTmp[i];
    }
}
template <const int AM, const int AK, const int AN, const int TM>
__global__ void mul_shared_mem_1d_re(float* d_A, float* d_B, float* d_C, int M, int K, int N) {
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int thread_Row = threadIdx.x / AN;
    int thread_Col = threadIdx.x % AN;
    __shared__ float s_A[AM * AK];
    __shared__ float s_B[AK * AN];
    float thread_TM[TM]{0.0};
    d_A += block_x * AM * K;
    d_B += block_y * AN;
    d_C += block_x * AM * N + block_y * AN;
    int innerAx = thread_Row; 
    int innerAy = thread_Col;
    // int innerBx =  //not use
    int innerBy = thread_Col;
    for(int iBlock = 0; iBlock < K; iBlock += AK) {
        // load A but every thread load one element
        s_A[thread_Row * AK + thread_Col] = d_A[thread_Row * K + thread_Col];  
        s_B[thread_Row * AN + thread_Col] = d_B[thread_Row * N + thread_Col];
        __syncthreads();
        d_A += AK;
        d_B += AK * N;

        for(int sB_row = 0; sB_row < AK; ++ sB_row) {
            float tmpB = s_B[sB_row * AN + innerBy];
            for(int sA_Col = 0; sA_Col < TM; ++ sA_Col) {
                thread_TM[sA_Col] += s_A[(thread_Row * TM + sA_Col) * AK + sB_row] * tmpB;
            }
        }
        __syncthreads();
    }
    for(int i = 0; i < TM; ++ i) {
        d_C[(thread_Row  * TM + i) * N + thread_Col] = thread_TM[i];
    }
}
template <const int BLOCKSIZE>
__global__ void mul_shared_mem(float* d_A, float* d_B, float* d_C, size_t nx, size_t nk, size_t ny) {
    size_t cCol = blockIdx.y; // block.y
    size_t cRow = blockIdx.x; // block.x
    size_t tCol = threadIdx.x % BLOCKSIZE; // 1. row first ! 2. "mapping a one-dimensional index to a two-dimensional coordinate"
    size_t tRow = threadIdx.x / BLOCKSIZE; // same on

    d_A += cRow * BLOCKSIZE * nk; // the block first row
    d_B += cCol * BLOCKSIZE; // the block first col
    d_C += cRow * BLOCKSIZE * ny + cCol * BLOCKSIZE; // the target block fisrt thread

    __shared__ float s_A[BLOCKSIZE * BLOCKSIZE];
    __shared__ float s_B[BLOCKSIZE * BLOCKSIZE];


    float tmp = 0.0;
    for(int block_i = 0; block_i < nk; block_i += BLOCKSIZE) {       

        s_A[tRow * BLOCKSIZE + tCol] = d_A[tRow * nk + tCol];
        s_B[tRow * BLOCKSIZE + tCol] = d_B[tRow * ny + tCol];
        __syncthreads();
        d_A += BLOCKSIZE; // next block in A
        d_B += BLOCKSIZE * ny; // next block in B
        
        
        for(int i = 0; i < BLOCKSIZE; ++ i) {
            tmp += s_A[tRow * BLOCKSIZE + i] * s_B[i * BLOCKSIZE + tCol];
        }
        __syncthreads();
    }
    d_C[tRow * ny + tCol] = tmp; 
    /*
         d_C pointer at the start of block.
         * If in (tRow = 0, tCol = 0) d_C = tmp;
         * if (1, 1) then d_C have cross the WHOLE line to get the next row and plus col to get the thread.
    */
}


int main() {
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    printf("Using device %d: %s %d\n",dev,deviceProp.name, deviceProp.warpSize);
    printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    CHECK(cudaSetDevice(dev));

    size_t nx_A = 1 << 13;
    // size_t nx_B = 1 << 13;
    size_t ny_A = 1 << 13;
    // size_t ny_B = 1 << 13;
    size_t nxy_A = nx_A * ny_A; 
    size_t nByte_A = nx_A * ny_A * sizeof(float);
    // size_t totByte_B = nx_B * ny_B * sizeof(float);
    const int AM = 64;
    const int AK = 8;
    const int BN = 64;
    const int tmp = 8;
    
    dim3 gridDim(CEIL_DIV(nx_A, AM), CEIL_DIV(ny_A, BN));
    dim3 blockDim((AM * BN) / tmp);

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
    printf("mem copy time %f \n", cpuSecond() - sTime);

    // cudaFuncAttribute(mul_shared_mem<32>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    mul_shared_mem<32><<<gridDim, blockDim>>> (d_A, d_B, d_C, nx_A, nx_A, ny_A);
    // mul_shared_mem_1d<AM, AK, BN, tmp><<<gridDim, blockDim>>> (d_A, d_B, d_C, nx_A, nx_A, ny_A);
    // mul_shared_mem_1d_re<AM, AK, BN, tmp><<<gridDim, blockDim>>> (d_A, d_B, d_C, nx_A, nx_A, ny_A);

    CHECK(cudaMemcpy(h_C, d_C, nByte_A, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());
    printf("total dev time %f \n", cpuSecond() - sTime);
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