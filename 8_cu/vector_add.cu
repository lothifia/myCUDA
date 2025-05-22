#include <cstdlib>
#include <cstdio>
#include <string>
#include <cassert>
#define dbg_print(x) print(#x": "); print(x); print("\n");
#define dbg_print_layout(x) print(#x": "); print_layout(x); print("\n");

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>
#include <cublas_v2.h>

using namespace cute;
__global__ void cuda_add(float* A, float* B, float* C, int N) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
        C[ix] = A[ix] + B[ix];
}

__global__ void vec_cuda_add(float* A, float* B, float* C, int N, size_t nbyte) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = ix; i < N / 4; i += blockDim.x * gridDim.x) { // 分配线程时出现线程数 小于 数据数量， 前几个线程可能处理两个数据
    float4 a4 = reinterpret_cast<float4*>(A)[i]; // 先转换，在进行index
    float4 b4 = reinterpret_cast<float4*>(B)[i];
    float4 c4;
        c4.x = a4.x + b4.x;
        c4.y = a4.y + b4.y;
        c4.z = a4.z + b4.z;
        c4.w = a4.w + b4.w;
    reinterpret_cast<float4*>(C)[i] = c4;
    }
}

int main(int argc, char* argv[]) {
    for(int i = 1; i < argc; i++) {
        std::string arg = argv[i];
    }

    using namespace cute;
    using element = __half;
    //cuda
    int num_size = 1 << 11;
    int nbytes = num_size * sizeof(float);
    dim3 block(128, 1, 1);
    dim3 grid((num_size + block.x - 1) / block.x, 1, 1);
    print(grid);
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    
    h_A = (float *)malloc(nbytes);
    h_B = (float *)malloc(nbytes);
    h_C = (float *)malloc(nbytes);

    cudaMalloc((void**)&d_A, nbytes);
    cudaMalloc((void**)&d_B, nbytes);
    cudaMalloc((void**)&d_C, nbytes);

    for (int i = 0; i < num_size; i++) {
        h_A[i] = float(i);
        h_B[i] = float(i);
        h_C[i] = 0.0f;
    }   
    cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nbytes, cudaMemcpyHostToDevice);
    // cuda_add<<<grid, block>>>(d_A, d_B, d_C, num_size);
    vec_cuda_add<<<grid, block>>>(d_A, d_B, d_C, num_size, nbytes);
    cudaMemcpy(h_C, d_C, nbytes, cudaMemcpyDeviceToHost);

    //cute
    // thrust::host_vector<element> h_A2(num_size);
    // thrust::host_vector<element> h_B2(num_size);
    // thrust::host_vector<element> h_C2(num_size);
    
    // for(size_t i = 0; i < num_size; i++) {
    //     h_A2[i] = static_cast<element>(i);
    //     h_B2[i] = static_cast<element>(i);
    //     h_C2[i] = static_cast<element>(0.0f);
    // }

    // thrust::device_vector<element> d_A2 = h_A2;
    // thrust::device_vector<element> d_B2 = h_B2;
    // thrust::device_vector<element> d_C2 = h_C2;


    // using MMA_ARCH = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
    // using TiledMma = TiledMMA<
    //     MMA_ARCH,
    //     Layout<Shape<_1, _1, _1>>,
    //     Tile<Int<16>, _8, _16>
    // >;
    // TiledMMA MmaC = TiledMma{};
    // dim3 dimGrid(size(ceil_div(num_size, 16)));
    // dim3 dimBlock(size(MmaC));

    for (int i = 0; i < num_size; i++) {
        printf("%f ", h_C[i]);
        if (i % 32 == 31) {
            printf("\n");
        }
    }
    return 0;

}

