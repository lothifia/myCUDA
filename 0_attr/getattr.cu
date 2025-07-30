#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
void CalTheoreticalBandWidth()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);               // 获取设备上的GPU个数

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);    // 获取当前GPU的相关属性


        std::cout << "GPU: " << i << std::endl;
        std::cout << "Name: " << deviceProp.name << std::endl;
        std::cout << "Bit width: " << deviceProp.memoryBusWidth << " bit" << std::endl;
        std::cout << "Memory clock rate: " << deviceProp.memoryClockRate << " kHz" << std::endl;

        int bw = static_cast<size_t>(deviceProp.memoryClockRate) * 1000 * deviceProp.memoryBusWidth / 8 * 2 / 1000000000;
        
        std::cout << "Theoretical band width = " << bw << " GB/s" << std::endl;
    }
}

int main() {
    CalTheoreticalBandWidth();
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
    printf("reg per thread: %d\n", prop.regsPerBlock / prop.maxThreadsPerBlock);
    printf("reg per Block: %d\n", prop.regsPerBlock);
    printf("Smem per Block: %zu\n", prop.sharedMemPerBlock);
    printf("threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("shared mem per sm: %zu B\n", prop.sharedMemPerMultiprocessor);
    // printf("", prop.reg)



    cudaDeviceProp props = prop;
    printf("Device ID: %d\n\
        Name: %s\n\
        Compute Capability: %d.%d\n\
        memoryBusWidth: %d\n\
        maxThreadsPerBlock: %d\n\
        maxThreadsPerMultiProcessor: %d\n\
        maxRegsPerBlock: %d\n\
        maxRegsPerMultiProcessor: %d\n\
        totalGlobalMem: %zuMB\n\
        sharedMemPerBlock: %zuKB\n\
        sharedMemPerMultiprocessor: %zuKB\n\
        totalConstMem: %zuKB\n\
        multiProcessorCount: %d\n\
        Warp Size: %d\n",
             0, props.name, props.major, props.minor, props.memoryBusWidth,
             props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
             props.regsPerBlock, props.regsPerMultiprocessor,
             props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
             props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
             props.multiProcessorCount, props.warpSize);
    return 0;

}
