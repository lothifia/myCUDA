/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*
  This example demonstrates how to call a CUTLASS GEMM kernel and provides a naive reference
  matrix multiply kernel to verify its correctness.

  The CUTLASS Gemm template is instantiated in the function CutlassSgemmNN. This is kernel computes
  the general matrix product (GEMM) using single-precision floating-point arithmetic and assumes
  all matrices have column-major layout.

  The threadblock tile size is chosen as 128x128x8 which offers good performance for large matrices.
  See the CUTLASS Parallel for All blog post for more exposition on the tunable parameters available
  in CUTLASS.

  https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/

  Aside from defining and launching the SGEMM kernel, this example does not use any other components
  or utilities within CUTLASS. Such utilities are demonstrated elsewhere in other examples and are
  prevalent in the CUTLASS unit tests.

  This example has delibrately been kept similar to the basic_gemm example from cutlass-1.3 to
  highlight the minimum amount of differences needed to transition to cutlass-2.0.

  Cutlass-1.3 sgemm: https://github.com/NVIDIA/cutlass/blob/master/examples/00_basic_gemm/basic_gemm.cu
*/

// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>

// Helper methods to check for errors
#include "helper.h"

//
// CUTLASS includes needed for single-precision GEMM kernel
//

// Defines cutlass::gemm::device::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/cutlass.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////
/// Define a CUTLASS GEMM template and launch a GEMM kernel.
using ElementInputA = int8_t;
using ElementInputB = int8_t;
using ElementInputC = int32_t;
using ElementOutput = int32_t;
using ElementAccumulator = int32_t;
using ElementComputeEpilogue = int32_t;


using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutInputC = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;
using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm75;
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 64>;
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>;
using ShapeMMAInstruction = cutlass::gemm::GemmShape<8, 8, 16>;

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementComputeEpilogue
    >;
constexpr int NumStages = 2;


using Gemm = cutlass::gemm::device::Gemm<
    ElementInputA,
    LayoutInputA,
    ElementInputB,
    LayoutInputB,
    ElementOutput,
    LayoutOutput,
    ElementAccumulator
    ,MMAOp
    ,SmArch
    ,ShapeMMAThreadBlock
    ,ShapeMMAWarp,
    ShapeMMAInstruction,
    EpilogueOp,
    SwizzleThreadBlock,
    NumStages
    >;


void run() {
    const int M = 16;
    const int N = 16;
    const int K = 16;

    cutlass::gemm::GemmCoord problem_size(M, N, K);
    cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
    problem_size.mk()); 
    cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
    problem_size.kn());
    cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
    problem_size.mn());

    cutlass::reference::host::TensorFillRandomUniform(
        tensor_a.host_view(), 1, ElementInputA(1.0), ElementInputA(1.0), 0);
    for(auto i = tensor_a.host_data(); i != tensor_a.host_data() + tensor_a.capacity(); ++i) {
        std::cout << *i << " ";
    }
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_b.host_view(), 1, ElementInputB(2.0), ElementInputB(2.0), 0);
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_c.host_view(), 1, ElementOutput(1.0), ElementOutput(1.0), 0);
    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_c.sync_device();

    ElementComputeEpilogue alpha = ElementComputeEpilogue(1.0);
    ElementComputeEpilogue beta = ElementComputeEpilogue(1.0);

    int splitK = 1;

    typename Gemm::Arguments Arguments{
        problem_size,
        tensor_a.device_ref(), tensor_b.device_ref(), tensor_c.device_ref(), tensor_c.device_ref(),
        {alpha, beta},
        {splitK}
    };
    
    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(Arguments);
    CUTLASS_CHECK(status);

    status = gemm_op.initialize(Arguments);
    CUTLASS_CHECK(status);
    status = gemm_op();
    CUTLASS_CHECK(status);

    cudaDeviceSynchronize();

    tensor_c.sync_host();

    for(auto i = tensor_c.host_data(); i != tensor_c.host_data() + tensor_c.capacity(); ++i) {
        std::cout << *i << " ";
    }
    return ;
}

int main() {
    int device_id = 0;
    cudaSetDevice(device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);

    std::cout << "Running on device " << device_id << ": " << device_prop.name << std::endl;

    run();

    return 0;
}