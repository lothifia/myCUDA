find_package(CUDA REQUIRED)

add_executable(6_matrix matrix_6.cu)
target_link_libraries(6_matrix cublas)