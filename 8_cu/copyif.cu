#include <cstdio>

// __global__ void copyif(int *A, int *B, int N, int *recn) {
//     int ix = blockDim.x * blockIdx.x + threadIdx.x;
//     if (ix < N && A[ix] > 0) {
//         B[atomicAdd(recn, 1)] = A[ix];
//         printf("recn %d \n", *recn);
//     }
// }
// 数据量为256000000时，latency=14.37ms
// naive kernel
__global__ void copyif(int *dst, int *nres, int *src, int n) {
 int i = threadIdx.x + blockIdx.x * blockDim.x;
 // 输入数据大于0的，计数器+1，并把该数写到输出显存以计数器值为索引的地址
 if(i < n && src[i] > 0)
   dst[atomicAdd(nres, 1)] = src[i];
}
int main() {
    int N = 256000;
    int *src, *dst;
    int *d_src, *d_dst;
    int *recn ;
    int *d_recn;

    // Allocate memory on the host
    src = (int *)malloc(N * sizeof(int));
    dst = (int *)malloc(N * sizeof(int));
    recn = (int *)malloc(sizeof(int));
    *recn = 0; // Initialize count of positive numbers to 0

    // Initialize A with some values
   for(int i = 0; i < N; i++){
        src[i] = 1;
    }

    int groundtruth = 0;
    for(int j = 0; j < N; j++){
        if (src[j] > 0) {
            groundtruth += 1;
        }
    }



    // Allocate memory on the device
    cudaMalloc((void **)&d_src, N * sizeof(int));
    cudaMalloc((void **)&d_dst, N * sizeof(int));
    cudaMalloc((void **)&d_recn, 1 * sizeof(int));

    // Copy A to device
    cudaMemcpy(d_src, src, N * sizeof(int), cudaMemcpyHostToDevice);
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    copyif<<<numBlocks, blockSize>>>(d_dst, d_recn, d_src, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken: %f ms\n", milliseconds);

    // Copy result back to host
    cudaMemcpy(recn, d_recn, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(dst, d_dst, *recn * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Number of positive elements: %d\n", *recn);
    // Print results
    // for (int i = 0; i < *recn; i++) {
    //     printf("%d ", dst[i]);
    // }
    if(groundtruth == *recn){
        printf("groundtruth is equal to recn\n");
    }else{
        printf("groundtruth is not equal to recn\n");
    }
    // Free device memory
    cudaFree(d_src);
    cudaFree(d_dst);
    
    // Free host memory
    free(src);
    free(dst);

    return 0;
}