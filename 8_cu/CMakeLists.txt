add_executable(ctest ctest.cu)
add_executable(add_cut add_cut.cu)
add_executable(vecotr_add vector_add.cu)
# 链接 cuBLAS
find_package(CUDAToolkit REQUIRED)
target_link_libraries(ctest PRIVATE CUDA::cublas)