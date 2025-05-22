add_executable(ctest ctest.cu)
add_executable(vecotr_add vector_add.cu)
add_executable(reduce_kernel reduce_kernel.cu)
# 链接 cuBLAS
find_package(CUDAToolkit REQUIRED)
target_link_libraries(ctest PRIVATE CUDA::cublas)