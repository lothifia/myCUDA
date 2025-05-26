#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T, int size>
struct alignas(size * sizeof(T)) AlignedVector {
    T data[size];
    __device__ __host__ inline T& operator[](int index) {
        return data[index];
    }
    __device__ __host__ inline const T& operator[](int index) const {
        return data[index];
    }
};  
template <typename T>
struct fused_mask_bias_add_scale{
    __device__ fused_mask_bias_add_scale(const T* mask, const T*add, const float scale, T* a, T* b) 
        : mask(mask), add(add), scale(scale) {}
    __device__ T operator()(const T x, int i) const {
        return (x * static_cast<T>(mask[i] * (scale)) + add[i]);
    }
    const T* mask;
    const T* add;
    const float scale;
};

// half 特化， 使用intrinsic 进行双倍运算
template <>
struct  fused_mask_bias_add_scale<__half> {
    __device__ fused_mask_bias_add_scale(const __half* mask, const __half* add, const float scale) 
        : mask(mask), add(add), scale(scale) {}
    __device__ __half operator()(__half x, int i) const {
        __half res = __hadd(__hmul(x, __hmul(mask[i], __float2half(scale))), add[i]);
        return res;
    }
    __device__ void apply2(const __half* a, __half* b, int i) const {
        const __half2 a2 = *reinterpret_cast<const __half2*>(a + i);
        const __half2 m2 = *reinterpret_cast<const __half2*>(mask + i);
        const __half2 add2 = *reinterpret_cast<const __half2*>(add + i);

        __half2 b2 = __hadd2(__hmul2(a2, __hmul2(m2, __float2half2_rn(scale))), add2);
        // printf("apply2: a2 = %f, %f, m2 = %f, %f, add2 = %f, %f, b2 = %f, %f, scale %f\n", 
        //        __half2float(a2.x), __half2float(a2.y), 
        //        __half2float(m2.x), __half2float(m2.y), 
        //        __half2float(add2.x), __half2float(add2.y),
        //        __half2float(b2.x), __half2float(b2.y),
        //         scale);
        *reinterpret_cast<__half2*>(b) = b2;
    };

    const __half* mask;
    const __half* add;
    const float scale;
};
//向量化
template <int vec_size>
__global__ void fused_mask_bias_add_scale_kernel() {};
template <int vec_size>
__global__ void fused_mask_bias_add_scale_kernel(const __half* a, __half* b, 
                                                 const __half* mask, const __half* add, 
                                                 const float *scale, int n) {
    // printf("%f \n", *scale);
    int gidx = (blockIdx.x * blockDim.x + threadIdx.x) * vec_size;
    int stried = blockDim.x * gridDim.x * vec_size;
    fused_mask_bias_add_scale<__half> fused(mask, add, *scale);

    using Arrt = AlignedVector<__half, vec_size>;
    for (int idx = gidx; idx < n; idx += stried) {
        __half b_reg[vec_size];
        const __half * m_reg = static_cast<const __half*>(mask + idx);
        const __half * a_reg = static_cast<const __half*>(a + idx);
        const __half * add_reg = static_cast<const __half*>(add + idx);
        
        for(int i = 0; i < vec_size; i += 2) {
            fused.apply2(a_reg, &b_reg[i], i);
        }
        *reinterpret_cast<Arrt*>(b + idx) = *reinterpret_cast<Arrt*>(b_reg);
    }
    
                                                 }
void checkFuse(const __half* a, __half* b, 
                const __half* mask, const __half* add, 
                const float *scale, int n) {
    for(int i = 0; i < n; i ++) {
        b[i] = __half2float(a[i]) * __half2float(mask[i]) * (*scale) + __half2float(add[i]);
    }
}

int main() {
    cudaDeviceProp prop;
    cudaSetDevice(0);
    cudaGetDeviceProperties(&prop, 0);
    printf("Device name: %s\n", prop.name);
    printf("Total global memory: %zu bytes\n", prop.totalGlobalMem);
    printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Registers per block: %d\n", prop.regsPerBlock);
    printf("Grid size: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    
    // bias mask scale add

    int n = 800;
    const int block_size = 256;
    const int vec_size = 8; // 8 elements per vector
    int pad_n = n % vec_size == 0 ? n : n + (vec_size - n % vec_size);

    __half *d_a, *d_b, *d_c;
    __half *mask;
    __half *bias;
    float *scale;
    __half *add;
    __half *h_mask = new __half[pad_n];
    __half *h_bias = new __half[pad_n];
    float h_scale = 0.5f;
    __half *h_add = new __half[pad_n];
    __half *h_a = new __half[pad_n];
    __half *h_b = new __half[pad_n];
    __half *h_c = new __half[pad_n];

    cudaMalloc((void**)&d_a, pad_n * sizeof(__half));
    cudaMalloc((void**)&d_b, pad_n * sizeof(__half));
    cudaMalloc((void**)&d_c, pad_n * sizeof(__half));
    cudaMalloc((void**)&mask, pad_n * sizeof(__half));
    cudaMalloc((void**)&bias, pad_n * sizeof(__half));
    cudaMalloc((void**)&add, pad_n * sizeof(__half));
    cudaMalloc((void**)&scale, 1 * sizeof(float));

    // Initialize host data
    for (int i = 0; i < pad_n; ++i) {
        h_a[i] = __float2half(static_cast<float>(i));
        h_mask[i] = __float2half(1.0f);
        h_bias[i] = __float2half(0.1f);
        h_add[i] = __float2half(0.5f);
    }
    // Copy data to device
    cudaMemcpy(d_a, h_a, pad_n * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(mask, h_mask, pad_n * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(bias, h_bias, pad_n * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(add, h_add, pad_n * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(scale, &h_scale, sizeof(float), cudaMemcpyHostToDevice);

    checkFuse(h_a, h_c, h_mask, h_add, &h_scale, pad_n);
    for(int i = 0; i < pad_n; ++i) {
        printf("h_c[%d] = %f\n", i, __half2float(h_c[i]));
    }

    auto checkAlign = [](const void *p, int alignment) {
        return reinterpret_cast<uintptr_t>(p) % alignment == 0;
    };
    constexpr auto alignSize = alignof(AlignedVector<__half, vec_size>);
    if (!checkAlign(d_a, alignSize) || !checkAlign(d_b, alignSize) || 
        !checkAlign(d_c, alignSize) || !checkAlign(mask, alignSize) || 
        !checkAlign(bias, alignSize) || 
        !checkAlign(add, alignSize)) {
        printf("Memory not aligned to %lu bytes\n", alignSize);
        return -1;
    }else {
        dim3 block(block_size, 1, 1);
        dim3 grid( (pad_n / vec_size + block.x - 1) / block.x, 1, 1);

        printf("Block size: %d, Grid size: %d\n", block.x, grid.x);
        // Launch the kernel
        fused_mask_bias_add_scale_kernel<vec_size><<<grid, block>>>(d_a, d_b, mask, add, scale, pad_n);
        // fused_mask_bias_add_scale_kernel<vec_size><<<grid, block>>>();

        cudaMemcpy(h_b, d_b, pad_n * sizeof(__half), cudaMemcpyDeviceToHost);
        
    }

    for(int i = 0; i < pad_n; ++i) {
        printf("h_b[%d] = %f\n", i, __half2float(h_b[i]));
    }

    free(h_a);
    free(h_b);
    free(h_mask);
    free(h_bias);
    free(h_add);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(mask);
    cudaFree(bias);
    cudaFree(add);
    cudaFree(scale);
    cudaDeviceReset();
    printf("All done!\n");


    return 0;
}