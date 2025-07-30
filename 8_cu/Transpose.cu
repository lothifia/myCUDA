#include <cstdio>
//element wise
__global__ void transpose_naive(float* A, float* B, int M, int N){
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int gid = bx * blockDim.x + tx;
    int row_a = gid / N;
    int col_a = gid % N;
    if(row_a < M) {
        B[col_a * M + row_a] = A[row_a * N + col_a];
    }
}
__global__ void transpose_naive_coalsedwrited(float* A, float* B, int M, int N){
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int gid = bx * blockDim.x + tx;
    int row_b = gid / M;
    int col_b = gid % M;
    if(row_b < M) {
        B[row_b * M + col_b] = A[col_b * N + row_b];
    }
}


__global__ void transpose_naive_1(float* A, float* B, int M, int N){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = bx * blockDim.x + tx;
    int gy = by * blockDim.y + ty;
    if(gx < M && gy < N) {
        // if(bx == 0 && by == 0) {
        //     printf("B idx %d: \n", gy * M + gx);
        // }
        B[gy * M + gx] = A[gx * N + gy];
    }
}

// __global__ void transpose_1(float* A, float* B, int M, int N){
//     int bx = blockIdx.x;
//     int tx = threadIdx.x;
//     int gid = bx * blockDim.x + tx;
//     int row_a = gid / N;
//     int col_a = gid % N;
//     __shared__ float smem[];
//     if(row_a < M) {
//         B[col_a * M + row_a] = A[row_a * N + col_a];
//     }
// }


int main() {
    const int M = 8192;
    const int N = 4096;
    float* A = (float*)malloc(M * N * sizeof(float));
    float* B = (float*)malloc(N * M * sizeof(float));
    float* B_ref = (float*)malloc(N * M * sizeof(float));
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            A[i * N + j] = i * N + j;
        }
    }
    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            B_ref[i * M + j] = A[j * N + i];
        }
    }
    float* d_A;
    float* d_B;
    cudaMalloc((void**)&d_A, M * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * M * sizeof(float));
    cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * M * sizeof(float), cudaMemcpyHostToDevice);

    // 1
    dim3 block_1(1024);
    transpose_naive<<<((M * N + block_1.x - 1) / block_1.x), block_1>>>(d_A, d_B, M, N);
    cudaMemcpy(B, d_B, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    // 1.5 
    dim3 block_15(32, 24);
    dim3 grid_15((M + block_15.x - 1) / block_15.x, (N + block_15.y - 1) / block_15.y);
    transpose_naive_1<<<grid_15, block_15>>>(d_A, d_B, M, N);
    cudaMemcpy(B, d_B, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    // 1 write coalesced
    dim3 block_1_coalesced(768);
    transpose_naive_coalsedwrited<<<((M * N + block_1_coalesced.x - 1) / block_1_coalesced.x), block_1_coalesced>>>(d_A, d_B, M, N);
    cudaMemcpy(B, d_B, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    bool is_correct = true;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            if(B[i * M + j] != B_ref[i * M + j]) {
                printf("B[%d][%d] = %f, B_ref[%d][%d] = %f\n", i, j, B[i * M + j], i, j, B_ref[i * M + j]);
                is_correct = false;
                break;
            }   
            if(!is_correct) {
                break;
            }
        }
    }
    if(is_correct) {
        printf("transpose_naive is correct\n");
    }
    else {
        printf("transpose_naive is incorrect\n");
    }




    return 0;
}