#include <cuda_runtime.h>
#include <stdio.h>
#include "myhead.h"

__global__ void mulMatrix(float* da, float* db, float* dc) {

}

int main() {
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    printf("Using device %d: %s %d\n",dev,deviceProp.name, deviceProp.warpSize);
    printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    CHECK(cudaSetDevice(dev));

    size_t nx = 1 << 13;
    size_t ny = 1 << 13;

    dim3 gridDim(CEIL_DIV(nx, 32), CEIL_DIV(ny, 32), 1);
    dim3 blockDim(32, 32, 1);

    float* h_A;
    float* h_B;
    float* h_C;

    mulMatrix<<<gridDim, blockDim>>> ()
    return 0;
}