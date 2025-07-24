#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <cublas_v2.h>
// tk = 32 
// grid (tiledm ,tiledn)
// block (tk * tk)
template <int TK>
__global__ void gemm_tiled(float* A, float* B, float* C, int M, int N, int K, int tiled_K) {
    int tid = threadIdx.x;
    int thread_row = tid / TK;
    int thread_col = tid % TK;
    int row_M = blockIdx.x * TK  + thread_row;
    int col_N = blockIdx.y * TK  + thread_col;

    __shared__ float a_tiled[TK * TK];
    __shared__ float b_tiled[TK * TK];

    float c = 0.0f;
    for(int k_loop = 0; k_loop < tiled_K; k_loop++) {
        int a_idx = row_M * K + k_loop * TK  + thread_col;
        int b_idx = col_N + (k_loop * TK  + thread_row) * N; 
        a_tiled[tid] = A[a_idx];
        b_tiled[tid] = B[b_idx];
        __syncthreads();
        for(int i = 0; i < TK; i++) {
            c += a_tiled[thread_row * TK + i] * b_tiled[i * TK + thread_col];
        }
        __syncthreads(); 
    }
    C[row_M * N + col_N] = c;
}
template <int TK>
__global__ void gemm_tiled_11(float* A, float* B, float* C, int M, int N, int K, int tiled_K) {
    int tid = threadIdx.x;
    int thread_row = tid / TK;
    int thread_col = tid % TK;
    int row_M = blockIdx.x * TK  + thread_row;
    int col_N = blockIdx.y * TK  + thread_col;

    __shared__ float a_tiled[TK * TK];
    __shared__ float b_tiled[TK * TK];

    float c = 0.0f;
    for(int k_loop = 0; k_loop < tiled_K; k_loop++) {
        int a_idx = row_M * K + k_loop * TK  + thread_col;
        int b_idx = col_N + (k_loop * TK  + thread_row) * N; 
        a_tiled[tid] = A[a_idx];
        b_tiled[thread_col * TK + thread_row] = B[b_idx];
        __syncthreads();
        for(int i = 0; i < TK; i++) {
            
        }
    }
    C[row_M * N + col_N] = c;
}
// tiled_size = 128 * 8
// per thread = 8 * 8 data (tm * tn)
// -> 16 * 16 thread
// 
template <typename T, int vec_size>
struct vec_t
{
    T data[vec_size]   /* data */;
};


//
template <int tm, int tn, int TA, int TB, int TK , int vec_size>
__global__ void gemm_tiled_2(float* A, float* B, float* C, int M, int N, int K, bool outdot, bool is_A_transposed, bool is_B_transposed) {
    int tid = threadIdx.x;
    int bx = blockIdx.x; // block_x -> number of tiles in x direction
    int by = blockIdx.y; // block_y -> number of tiles in y direction
    int thread_row = tid / (TB / tn);
    int thread_col = tid % (TB / tn);
    __shared__ float a_tiled[TA * TK];
    __shared__ float b_tiled[TK * TB];
    const int total_threads = TA * TB / (tm * tn);

    // copy used
    //stride_b is always in row major
    const int copy_T_col_A = TK / vec_size;
    const int copy_T_row_A = total_threads / copy_T_col_A;
    const int total_data_once_block_A = copy_T_row_A * TK;
    const int stride_a = copy_T_row_A;

    const int copy_T_col_B = TB / vec_size;
    const int copy_T_row_B = total_threads / copy_T_col_B;
    const int total_data_once_block_B = copy_T_row_B * TK;
    const int stride_b = copy_T_row_B;
    
    // compute used
    const int threads_in_m = TA / tm;
    const int threads_in_n = TB / tn;
    const int t_col = tid % threads_in_n;
    const int t_row = tid / threads_in_n;


    // float c[TA * TB / (tm * tn)]; x : too many regs
    float c[tm * tn];
    float reg_tm[tm];
    float reg_tn[tn];
    float* a_ptr = A + bx * TA * K; 
    float* b_ptr = B + by * TB; 
    float* c_ptr = C + bx * TA * N + by * TB;

    int tiled_K = (K + TK - 1) / TK;
    for(int k_loop = 0; k_loop < tiled_K; k_loop++) {
        static_assert(vec_size == 4, "vec_size must be 4");
        static_assert(total_threads / (TK / vec_size) == 128, "128 row to process mem copy");
        int row_a = tid / copy_T_col_A;
        int row_b = tid / copy_T_col_B;
        int col_a = tid % copy_T_col_A;
        int col_b = tid % copy_T_col_B;

        for(int a_offset = row_a; a_offset < TA; a_offset += stride_a){
            float* a_in_gmem = a_ptr + a_offset * K + k_loop * TK + col_a * vec_size;
            float* a_in_smem = a_tiled + a_offset * TK + col_a * vec_size;
            reinterpret_cast<float4*>(a_in_smem)[0] = reinterpret_cast<float4*>(a_in_gmem)[0];
        }
        // if(tid * vec_size < TB * TK){
        for(int b_offset = row_b; b_offset < TK; b_offset += stride_b){
            float* b_in_gmem = b_ptr + k_loop * TK * N + b_offset * N + col_b * vec_size;
            float* b_in_smem = b_tiled + b_offset * TB + col_b * vec_size;
            reinterpret_cast<float4*>(b_in_smem)[0] = reinterpret_cast<float4*>(b_in_gmem)[0];
        }
        // }
        __syncthreads();
        // inner product
        if(!outdot){
            for(int m = 0; m < tm; m++) {
                for(int n = 0; n < tn; n++) {
                    for(int p = 0; p < TK; p++) {
                        c[m * tn + n] += a_tiled[(t_row + m * threads_in_m) * TK + p] * b_tiled[(t_col + n * threads_in_n) + p * TB];//stop
                        // b_tiled[(t_col + n * threads_in_n) + p * TB];
                    }
                }
            }
        }
        // outter product
        else{
            for(int p = 0; p < TK; p++) {
                for(int n = 0; n < tn; n++) {
                    reg_tn[n] = b_tiled[t_col + n * threads_in_n + p * TB];
                }
                for(int m = 0; m < tm; m++) {
                    reg_tm[m] = a_tiled[(t_row + m * threads_in_m) * TK + p];
                }
                for(int m = 0; m < tm; m++) {
                    for(int n = 0; n < tn; n++) {
                        c[m * tn + n] += reg_tm[m] * reg_tn[n];
                    }
                }
            }
        }
        __syncthreads();
    }

    for(int m = 0; m < tm; m++) {
        for(int n = 0; n < tn; n++){
            c_ptr[(t_row + m * threads_in_m) * N + n * threads_in_n + t_col] = c[m * tn + n];
        }
    }
}


template <int tm, int tn, int TA, int TB, int TK , int vec_size>
__global__ void gemm_tiled_2_vecIO(float* A, float* B, float* C, int M, int N, int K, bool outdot, bool is_A_transposed, bool is_B_transposed) {
    int tid = threadIdx.x;
    int bx = blockIdx.x; // block_x -> number of tiles in x direction
    int by = blockIdx.y; // block_y -> number of tiles in y direction
    int thread_row = tid / (TB / tn);
    int thread_col = tid % (TB / tn);
    __shared__ float a_tiled[TA * TK];
    __shared__ float b_tiled[TK * TB];
    const int total_threads = TA * TB / (tm * tn);

    // copy used
    //stride_b is always in row major
    const int copy_T_col_A = TK / vec_size;
    const int copy_T_row_A = total_threads / copy_T_col_A;
    const int total_data_once_block_A = copy_T_row_A * TK;
    const int stride_a = copy_T_row_A;

    const int copy_T_col_B = TB / vec_size;
    const int copy_T_row_B = total_threads / copy_T_col_B;
    const int total_data_once_block_B = copy_T_row_B * TK;
    const int stride_b = copy_T_row_B;
    
    // compute used
    const int threads_in_m = TA / tm; // 128 / 8 = 16
    const int threads_in_n = TB / tn;
    const int t_col = tid % threads_in_n;
    const int t_row = tid / threads_in_n;


    // float c[TA * TB / (tm * tn)]; x : too many regs
    float c[tm * tn];
    float reg_tm[tm];
    float reg_tn[tn];
    float* a_ptr = A + bx * TA * K; 
    float* b_ptr = B + by * TB; 
    float* c_ptr = C + bx * TA * N + by * TB;

    int tiled_K = (K + TK - 1) / TK;
    for(int k_loop = 0; k_loop < tiled_K; k_loop++) {
        static_assert(vec_size == 4, "vec_size must be 4");
        static_assert(total_threads / (TK / vec_size) == 128, "128 row to process mem copy");
        int row_a = tid / copy_T_col_A;
        int row_b = tid / copy_T_col_B;
        int col_a = tid % copy_T_col_A;
        int col_b = tid % copy_T_col_B;

        for(int a_offset = row_a; a_offset < TA; a_offset += stride_a){
            float* a_in_gmem = a_ptr + a_offset * K + k_loop * TK + col_a * vec_size;
            float* a_in_smem = a_tiled + a_offset * TK + col_a * vec_size;
            reinterpret_cast<float4*>(a_in_smem)[0] = reinterpret_cast<float4*>(a_in_gmem)[0];
        }
        // if(tid * vec_size < TB * TK){
        for(int b_offset = row_b; b_offset < TK; b_offset += stride_b){
            float* b_in_gmem = b_ptr + k_loop * TK * N + b_offset * N + col_b * vec_size;
            float* b_in_smem = b_tiled + b_offset * TB + col_b * vec_size;
            reinterpret_cast<float4*>(b_in_smem)[0] = reinterpret_cast<float4*>(b_in_gmem)[0];
        }
        // }
        __syncthreads();
        // inner product
        if(!outdot){
            for(int m = 0; m < tm; m++) {
                for(int n = 0; n < tn; n++) {
                    for(int p = 0; p < TK; p++) {
                        c[m * tn + n] += a_tiled[(t_row + m * threads_in_m) * TK + p] * b_tiled[(t_col + n * threads_in_n) + p * TB];//stop
                        // b_tiled[(t_col + n * threads_in_n) + p * TB];
                    }
                }
            }
        }
        // outter product
        else{
            for(int p = 0; p < TK; p++) {
                for(int n = 0; n < tn; n++) {
                    reg_tn[n] = b_tiled[t_col + n * threads_in_n + p * TB];
                }
                for(int m = 0; m < tm; m++) {
                    reg_tm[m] = a_tiled[(t_row + m * threads_in_m) * TK + p];
                }
                for(int m = 0; m < tm; m++) {
                    for(int n = 0; n < tn; n++) {
                        c[m * tn + n] += reg_tm[m] * reg_tn[n];
                    }
                }
            }
        }
        __syncthreads();
    }
    // write back (reg -> smem -> gmem)
    // 3
    int t_to_sync = TA * TK / (total_threads);
    int smem_col = TA * TK / (threads_in_m); // 128 * 8 / 16 = 64
    // vec_write use
    const int row_in_smem = t_row;
    const int col_in_smem = t_col * vec_size;
    // vec_data prepare
    int cnt = t_to_sync;
    for(int m = 0; m < tm; m++) {
        for(int n = 0; n < tn; n++) {

            a_tiled[t_row * smem_col + t_col + (n % t_to_sync)* threads_in_n] = c[m * tn + n];
            if(bx == 0 && by == 0) {
                float test = 707981312.000000;
                float test2 = 708371712.000000 ;
                // printf("Writing to row_in_gmem=%d, col_in_gmem=%d (offset=%d)\n",
                // row_in_gmem,
                // col_in_gmem,
                // row_in_gmem * N + col_in_gmem);
                if(c[m * tn + n] - test ==  0.000000) {
                printf("expected %f \n", test);
                    printf( " diff = %f \n", c[m * tn + n] - test);
                    printf(" m %d n %d \n", m, n);
                    printf(" row_in_smem %d col_in_smem %d \n", row_in_smem, col_in_smem);
                    printf(" c[m * tn + n] %f \n", c[m * tn + n]);
                }
                if(c[m * tn + n] - test2 ==  0.000000) {
                printf("got %f \n", test2);
                    printf( " diff = %f \n", c[m * tn + n] - test2);
                    printf(" m %d n %d \n", m, n);
                    printf(" row_in_smem %d col_in_smem %d \n", row_in_smem, col_in_smem);
                    printf(" c[m * tn + n] %f \n", c[m * tn + n]);
                }
            }
            --cnt;
            if(cnt == 0) {
                __syncthreads();
                // printf(" m %d n %d \n", m, n);
                // printf(" row_in_smem %d col_in_smem %d \n", row_in_smem, col_in_smem);
                cnt = t_to_sync;
                const int row_in_gmem = (t_row + m * threads_in_m);
                const int col_in_gmem = ((n * threads_in_n) / smem_col * smem_col + t_col * vec_size);
                // could loop
                // wrong !: reinterpret_cast<float4*>(c_ptr + row_in_gmem * TB + col_in_gmem)[0] = reinterpret_cast<float4*>(a_tiled + t_row * smem_col + col_in_smem)[0];  
                reinterpret_cast<float4*>(c_ptr + row_in_gmem * N + col_in_gmem)[0] = reinterpret_cast<float4*>(a_tiled + t_row * smem_col + t_col * vec_size)[0];  
                if(bx == 0 && by == 0) {
                    if(row_in_gmem * N + col_in_gmem == 64) {
                        printf(" data in smem %f \n", a_tiled[t_row * smem_col + t_col + n * threads_in_n]);
                        printf("t_row %d t_col %d \n", t_row, t_col);
                    }
                }

                __syncthreads();
            }

        }
    }


    // for(int m = 0; m < tm; m++) {
    //     for(int n = 0; n < tn; n++){
    //         c_ptr[(t_row + m * threads_in_m) * N + n * threads_in_n + t_col] = c[m * tn + n];
    //     }
    // }
}


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS Error at %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


int main() {
    int M = 1024;
    int N = 1024;
    int K = 128;
    float alpha = 1.0f;
    float beta = 0.0f;

    // 1. Host memory allocation and initialization
    float* A = (float*)malloc(M * K * sizeof(float));
    float* B = (float*)malloc(K * N * sizeof(float));
    float* C = (float*)malloc(M * N * sizeof(float)); // This will hold the result
    float* C_ref = (float*)malloc(M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) { A[i] = float(i); }
    for (int i = 0; i < K * N; i++) { B[i] = float(i); }
    // for (int i = 0; i < M * K; i++) { A[i] = 1.0f; }
    // for (int i = 0; i < K * N; i++) { B[i] = 2.0f; }
    printf("Matrices initialized.\n");

    // 2. Device memory allocation with error checking
    float *d_A, *d_B, *d_C, *d_C_ref;
    printf("Allocating device memory...\n");
    CUDA_CHECK(cudaMalloc((void**)&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C_ref, M * N * sizeof(float)));
    printf("Device memory allocated.\n");

    // 3. Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // 4. Perform GEMM using cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    printf("Performing GEMM...\n");
    
    // Note: The result will be stored in d_C
CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                         N, M, K,              // 注意：M和N交换了位置
                         &alpha, 
                         d_B, N,               // 注意：B在前，A在后
                         d_A, K, 
                         &beta, 
                         d_C_ref, N));              // 注意：ldc现在是N
    
    CUDA_CHECK(cudaMemcpy(C_ref, d_C_ref, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUBLAS_CHECK(cublasDestroy(handle));
    printf("cuBLAS GEMM finished.\n");

    // // 1.0
    // const int TK = 32;
    // int tiled_M = (M + TK - 1) / TK;
    // int tiled_K = (K + TK - 1) / TK;
    // int tield_N = (N + TK - 1) / TK;

    // dim3 block_tiled(TK * TK);
    // dim3 grid_tiled(tiled_M, tield_N);
    // gemm_tiled<TK><<<grid_tiled, block_tiled>>>(d_A, d_B, d_C, M, N, K, tiled_K);

    // 2.0
    const int TK = 8;
    const int tm = 8;
    const int tn = 8;
    const int TA = 128;
    const int TB = 128;
    const int tiled_K = (K + TK - 1) / TK;
    const int vec_size = 4;
    dim3 block_tiled(TA * TB / (tm * tn));
    dim3 gird_tiled(M / TA, N / TB);

    bool outdot = true;
    // gemm_tiled_2<tm, tn, TA, TB, TK, vec_size><<<gird_tiled, block_tiled>>>(d_A, d_B, d_C, M, N, K, outdot, false, false);
    gemm_tiled_2_vecIO<tm, tn, TA, TB, TK, vec_size><<<gird_tiled, block_tiled>>>(d_A, d_B, d_C, M, N, K, outdot, false, false);

    // 5. Copy result from device to host
    // 【修正】从 d_C 拷回数据到 C
    CUDA_CHECK(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    // for(int i = 0; i < M * N; i++) {
    //     printf(" %f " , C[i]);
    // }
    printf("Result copied back to host.\n");
    // for(int i = 0; i < M * N; i++) {
    //     printf(" %f " , C[i]);
    // }
    int debug_cnt = 0;
    // for(int i = 0; i < M * N; i++) {
        // printf(" %f ", C[i]);
        // if(C[i] - 512.0 >  1e-5) {
            // debug_cnt++;
        // }
    // }

    bool correct = true;
    for(int i = 0; i < M * N; i++) {
        if(C[i] - C_ref[i] >  1e-5) {
            printf("Verification FAILED at index %d! Got %f, expected %f\n", i, C[i], C_ref[i]);
            correct = false;
        }
    }
    if(correct) {
        printf("✅ GEMM result is correct!\n");
    } 
    free(A);
    free(B);
    free(C);
    free(C_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C_ref));
    printf("Memory freed.\n");



    return 0;
}