#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    cudaDeviceProp prop;
    int device;
    
    // 获取当前设备号
    cudaGetDevice(&device);
    
    // 获取当前设备的属性
    cudaGetDeviceProperties(&prop, device);

    printf("设备名称: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    printf("每个线程块最大线程数: %d\n", prop.maxThreadsPerBlock);
    printf("线程块x方向最大线程数: %d\n", prop.maxThreadsDim[0]);
    printf("线程块y方向最大线程数: %d\n", prop.maxThreadsDim[1]);
    printf("线程块z方向最大线程数: %d\n", prop.maxThreadsDim[2]);

    return 0;
}
