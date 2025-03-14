#include <cuda_runtime.h>
#include <stdio.h>
#include "myhead.h"

__global__ void mulMatrix(float* d_A, float* d_B, float* d_C, size_t nx, size_t nk, size_t ny) {
    size_t ix = blockDim.x * blockIdx.x + threadIdx.x;
    size_t iy = blockDim.y * blockIdx.y + threadIdx.y;
    float tem = 0;
    // printf("sbzjq");
    if(ix < nx && iy < ny) {
                int cxy = (ix * nx + iy);
                int int_ix = ix;
        //    printf("blockx%d blocky%d threadx%d thready%d, %d ix is %d !\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, cxy, int_ix);
        for(int i = 0; i < nk; i++) {
            tem += d_A[ix * nk + i] * d_B[iy + i * ny];
        }
        d_C[ix * nx + iy] = tem;
    }
}

int main() {
    double cTime = cpuSecond();
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

    dim3 gridDim(CEIL_DIV(nx_A, 32), CEIL_DIV(ny_A, 32), 1);
    dim3 blockDim(32, 32, 1);

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

    mulMatrix<<<gridDim, blockDim>>> (d_A, d_B, d_C, nx_A, nx_A, ny_A);
    CHECK(cudaMemcpy(h_C, d_C, nByte_A, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());
    printf("total dev time %f \n", cpuSecond() - sTime);
    printf("tot time %f \n", cpuSecond() - cTime);
    printf("over \n");
    // for(int i = 0; i < nxy_A; i++) {
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