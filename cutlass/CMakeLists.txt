add_executable(cut cut.cu)
# add_executable(mcut mcut.cu)

# set_target_properties(cut PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_ARCHITECTURES 89  # 你已经在顶层设置了，这里可以省略
# )
# set_target_properties(mcut PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_ARCHITECTURES 89  # 你已经在顶层设置了，这里可以省略
# )