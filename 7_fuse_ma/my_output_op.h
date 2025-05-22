#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"

/// 自定义融合算子：Scale + Bias + ReLU
template <
  typename ElementOutput_,       // 输出类型，如 float
  typename ElementAccumulator_,  // 累加器类型，如 float
  typename ElementCompute_       // 计算类型，一般也是 float
>
struct MyScaleBiasRelu {

  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  // 构造参数：alpha/beta 控制缩放（可选）
  ElementCompute alpha;
  ElementCompute beta;

  CUTLASS_HOST_DEVICE
  MyScaleBiasRelu(ElementCompute alpha_ = ElementCompute(1),
                  ElementCompute beta_  = ElementCompute(1))
      : alpha(alpha_), beta(beta_) {}

  // 每个元素的融合函数：accum × scale + bias → ReLU → output
  CUTLASS_HOST_DEVICE
  ElementOutput operator()(
    ElementAccumulator const& accum,
    ElementCompute const& scale,
    ElementCompute const& bias
  ) const {
    ElementCompute tmp = ElementCompute(accum) * scale + bias;
    tmp = cutlass::maximum<ElementCompute>()(tmp, ElementCompute(0)); // ReLU
    return ElementOutput(tmp);
  }
};
