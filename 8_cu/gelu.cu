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

__device__ float tanh_approx(float x) {
    // float r;
    // asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    // return r;
    return tanhf(x);
}

//gelu公式：x / 2 * (1 + tan(0.7978845608028654 * (x + 0.044714998453855515 * x^3)))
template <typename T>
struct gelufunc {
    static constexpr T alpha = static_cast<T>(0.7978845608028654);
    static constexpr T beta = static_cast<T>(0.044714998453855515);

    __device__ gelufunc() {};

    __device__ T operator()(T x) const {
        const T half = static_cast<T>(0.5);
        const T one = static_cast<T>(1.0);
        const T tanh_in = alpha * (x + beta * x * x * x);
        return half * x * (one + tanh_approx(tanh_in));
    }
};
// 特化给__half类型
template <>
struct gelufunc<__half> {
    static constexpr float alpha = gelufunc<float>::alpha;
    static constexpr float beta = gelufunc<float>::beta;
    gelufunc<float> geluFunc;

    __device__ gelufunc() {};

    __device__ __half operator()(__half x) const {
        // 使用float的gelu函数
        const float tanh_in = 
            __half2float(__float2half_rn(alpha) * (x + __float2half_rn(beta) * x * x * x));
        const float tanh_out = tanh_approx(tanh_in);
        return __float2half_rn(0.5f) * x * (__float2half_rn(1.0f) + __float2half_rn(tanh_out));
    }

    // use half2 intrinsic
    __device__ void apply2(const __half* x, __half* y) const {
        const __half2 x2 =  *(reinterpret_cast<const __half2*>(x));
        const float2 tanh_in = 
            __half22float2(__hmul2(__float2half2_rn(alpha), __hadd2(x2, __hmul2(__hmul2(__hmul2(__float2half2_rn(beta), x2), x2), x2))));
        const float2 tanh_out {
            tanh_approx(tanh_in.x),
            tanh_approx(tanh_in.y)
        };
        const __half2 result = __hmul2(__float2half2_rn(0.5f), __hmul2(x2, __hadd2(__float2half2_rn(1.0f), __float22half2_rn(tanh_out))));
        *(reinterpret_cast<__half2*>(y)) = result;
    }
};

template <int VecSize>
__global__ void FP16GeluKernel(__half* x, __half* y, int n) {
    int offset = (blockIdx.x * blockDim.x + threadIdx.x) * VecSize; // block :4, grid :4 ;(2, 3) = (2 * 4 + 3) * 8 = 88; 前87个都被处理完了
    int stride = blockDim.x * gridDim.x * VecSize;
    gelufunc<__half> gelufwd;
    // 每个线程要处理 loop1 个vector 保证数据被处理完
    __half y_reg[VecSize];
    __half x_reg[VecSize];
    using ArrT = AlignedVector<__half, VecSize>; 
    for(int idx = offset; idx < n; idx += stride) {
        const __half* x_in = x + idx;
        const __half* x_in8 = x + idx; // 每个线程处理8个元素 
        *reinterpret_cast<ArrT*>(x_reg) = *reinterpret_cast<const ArrT*>(x_in8);

        if(VecSize == 1) {
            y_reg[0] = gelufwd(x_in[0]);
        } else {
            for(int i = 0; i < VecSize; i += 2) {
                // gelufwd.apply2(&x_in[i], &y_reg[i]);
                gelufwd.apply2(&x_reg[i], &y_reg[i]);
            }
        }
        *reinterpret_cast<ArrT*>(y + idx) = *reinterpret_cast<ArrT*>(y_reg);
    }
}

void gelu_check(float *x, int n) {
    for(int i = 0; i < n; i ++) {
        x[i] = x[i] / 2 * (1 + tanhf(0.7978845608028654 * (x[i] + 0.044714998453855515 * x[i] * x[i] * x[i])));
    }
}
// baseline
int main() {
    // 对齐
    int pad_n = 1001;
    const int block_size = 512;
    const int vec_size = 8;
    int n = pad_n % vec_size == 0? pad_n: pad_n + (vec_size - pad_n%vec_size);

    float milliseconds = 0;
    
    // __half *x = new __half[n];
    // __half *y = new __half[n];
    __half *x = (__half*)malloc(n * sizeof(__half));
    __half *y = (__half*)malloc(n * sizeof(__half));
    float *x_check = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++)
    {
      x[i] = (__half)(float(i) / 100.0);
    //   printf("%d: %f\n", i, __half2float(x[i]));
      x_check[i] = static_cast<float>(i) / 100.0;
    }

    gelu_check(x_check, n);
    // // print check
    // for (int i = 0; i < n; i++) {
    //     printf("%d: %f\n", i, x_check[i]);
    // }
    __half * d_x, *d_y;
    cudaMalloc((void **)&d_x, n * sizeof(__half));
    cudaMalloc((void **)&d_y, n * sizeof(__half));
    cudaMemcpy(d_x, x, sizeof(__half) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, sizeof(__half) * n, cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    auto is_aligned = [](const void* p, int alignment) {
        return reinterpret_cast<uintptr_t>(p) % alignment == 0;
    };
    //编译时即可计算                                                                  
    constexpr auto kAlignment = alignof(AlignedVector<__half, 8>); 
    // Note: when you have ampere GPU, you can enable the 134-136 line to get performance improvement by half2 intrinsic.
    if (n % 8 == 0 && is_aligned(x, kAlignment) && is_aligned(y, kAlignment)) {                                          

        int thread = std::min<int>(512, deviceProp.maxThreadsPerBlock); 
        //int block = (n / 8 + thread - 1) / thread;                  
        //block = std::min<int>(block, deviceProp.maxGridSize[0]);                                  
        //FP16GeluCUDAKernel<8><<<block, thread>>>(d_x, d_y, n);  
        int block = (n /8 + thread - 1) / thread;                  
        block = std::min<int>(block, deviceProp.maxGridSize[0]);                                  
        printf("thread%d, block %d, \n", thread, block);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        FP16GeluKernel<8><<<block, thread>>>(d_x, d_y, n);                      
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        //   FP16GeluCUDAKernel<8><<<block, thread>>>(d_x, d_y, n);
        cudaMemcpy(y, d_y, sizeof(__half) * pad_n, cudaMemcpyDeviceToHost);                                                          
    }   
    //print
    for (int i = 0; i < pad_n; i++) {
        printf("%d: %f\n", i, __half2float(y[i]));
    }
    printf("pass");
    printf("Time: %f ms\n", milliseconds);
    cudaFree(d_x);
    cudaFree(d_y);
}