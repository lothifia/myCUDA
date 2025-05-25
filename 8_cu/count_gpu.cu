#include <cstdio>
#include <unordered_map>

__global__ void cuda_count(int *A, int *B, int N) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    // 使用原子加进行
    // B[A[ix]]++;
    atomicAdd(&B[A[ix]], 1);
}

__global__ void cuda_count_smem(int *A, int *B, int N) {
    __shared__ int smem[256];
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    smem[threadIdx.x] = 0;
    __syncthreads();
    for(int i = ix; i < N; i += blockDim.x * gridDim.x) {
        int val = A[i];
        atomicAdd(&smem[val], 1);
    }
    __syncthreads();
    atomicAdd(&B[threadIdx.x], smem[threadIdx.x]);
}

__host__ bool check_out(int *A, int *B, int N) {
    std::unordered_map<int, int> dic;
    for(int i = 0; i < N; i++) {
        int val = A[i];
        if(dic.find(val) == dic.end()) {
            dic[val] = 1;
        } else {
            dic[val]++;
        }
    }

    for(int i = 0; i < N; i++) {
        int val = A[i];
        if(dic[val] != B[val]) {
            printf("Error: %d %d\n", dic[val], B[val]);
            return false;
        }
    }
    return true;

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

    const int N = 25600000;
    int *d_A, *d_B;
    int *h_A = (int *)malloc(N * sizeof(int));
    int *h_B = (int *)malloc(N * sizeof(int));
    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() % 256;
        h_B[i] = 0;
    }
    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cuda_count<<<numBlocks, blockSize>>>(d_A, d_B, N);
    // cuda_count_smem<<<numBlocks, blockSize>>>(d_A, d_B, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken: %f ms\n", milliseconds);

    cudaMemcpy(h_B, d_B, N * sizeof(int), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < N; i++) {
    //     printf("%d ", h_A[i]);
    // }
    // printf("\n");
    // for (int i = 0; i < N; i++) {
    //     printf("%d ", h_B[i]);
    // }
    // printf("\n");
    bool result = check_out(h_A, h_B, N);
    if(result) {
        printf("Result is correct!\n");
    } else {
        printf("Result is incorrect!\n");
    }
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);
    return 0;
}