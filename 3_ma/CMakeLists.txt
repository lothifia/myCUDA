add_executable(3_matriex matrix_3.cu)
target_compile_options(3_matriex PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)