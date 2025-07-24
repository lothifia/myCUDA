add_executable(ctest ctest.cu)
add_executable(vecotr_add vector_add.cu)
add_executable(reduce_kernel reduce_kernel.cu)
add_executable(count_gpu count_gpu.cu)
add_executable(v3 v3.cu)
add_executable(copyif copyif.cu)
add_executable(gelu gelu.cu)
add_executable(fused fused.cu)
add_executable(softmax softmax.cu)
add_executable(gemm gemm.cu)  # new
# 链接 cuBLAS
find_package(CUDAToolkit REQUIRED)
target_link_libraries(ctest PRIVATE CUDA::cublas)
target_link_libraries(gemm PRIVATE CUDA::cublas)