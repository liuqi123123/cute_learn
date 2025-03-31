
#pragma once
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "include/cuda_helper.h"
#include "cutlass/fast_math.h"
#include "cute/numeric/integral_constant.hpp"
#include "cute/pointer.hpp"
#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/array_subbyte.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include <cmath>
#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <iostream>
#include "cutlass/half.h"
#include "/home/liuqi/project/cutlass/include/cutlass/conv/threadblock/conv2d_params.h"

#define DEVICE_INLINE __device__ __forceinline__
using namespace cute;


struct ActivateParams{
  cutlass::FastDivmod pq_divmod;
  cutlass::FastDivmod q_divmod;
  int PQ;
  int inc_next[3]; // {next S, next R, next C}
  int filter_c_delta;

  CUTLASS_HOST_DEVICE
  ActivateParams() { }

  CUTLASS_HOST_DEVICE
  ActivateParams(cutlass::conv::Conv2dProblemSize const &problem_size,
                 int element_bytes, int blockshape_kK)
      :
        PQ(problem_size.P * problem_size.Q),
        pq_divmod(problem_size.P * problem_size.Q)
        // ,pq_divmod(problem_size.P * problem_size.Q)
        ,q_divmod(problem_size.Q)

        {


    // return;
    int conv_sign = (problem_size.mode == cutlass::conv::Mode::kConvolution ? -1 : 1);

    auto activate_layout = make_layout(
            make_shape(problem_size.N, problem_size.H, problem_size.W,
                       problem_size.C),
            make_stride(problem_size.H * problem_size.W * problem_size.C,
                        problem_size.W * problem_size.C, problem_size.C, 1));
    // next S
    inc_next[0] = conv_sign * (
      (stride<2>(activate_layout)) * problem_size.dilation_w
    ) * element_bytes;

    // next R
    inc_next[1] = conv_sign * (
        (stride<1>(activate_layout)) * problem_size.dilation_h
        - (problem_size.S - 1) * stride<2>(activate_layout) * problem_size.dilation_w
      ) * element_bytes;

    // next C
    inc_next[2] = (
        blockshape_kK * problem_size.split_k_slices
        - conv_sign * int32_t(problem_size.R - 1) * stride<1>(activate_layout) * problem_size.dilation_h
        - conv_sign * int32_t(problem_size.S - 1) * stride<2>(activate_layout) * problem_size.dilation_w
      ) * element_bytes;

    // logical offset added to internal channel counter - units are elements, not bytes
    filter_c_delta = blockshape_kK * problem_size.split_k_slices;
    // printf("problem_size.split_k_slices:%d\n", problem_size.split_k_slices);
  }
};




struct FilterParams{
  int RS;

  int32_t inc_next_k;         // offset in units of bytes to next K position
  int32_t inc_next_rs;        // offset in units of bytes to next RS position
  int32_t inc_next_c;         // offset in units of bytes to next C position
  CUTLASS_HOST_DEVICE
  FilterParams() { }

  CUTLASS_HOST_DEVICE
  FilterParams(cutlass::conv::Conv2dProblemSize const &problem_size,
                                                      //BlockThreadArrangement stride, //warp stride iteration num
               int element_bytes, int blockshape_kK, int threadmap_delta_stride, int threadmap_iterations_stride)

  {
    auto filter_layout = make_layout(make_shape(problem_size.K, problem_size.R,
                                             problem_size.S, problem_size.C),
                                  make_stride(problem_size.R * problem_size.S *
                                                  problem_size.C,
                                              problem_size.S * problem_size.C,
                                              problem_size.C, 1));
    RS = problem_size.R * problem_size.S;
    inc_next_k = int32_t(stride<0>(filter_layout)) * threadmap_delta_stride * element_bytes;

    inc_next_rs =
      ( int32_t(stride<2>(filter_layout))
        - int32_t(stride<0>(filter_layout)) * (threadmap_iterations_stride - 1) * threadmap_delta_stride
      ) * element_bytes;

    inc_next_c =
      (
        blockshape_kK * problem_size.split_k_slices
        - int32_t(RS - 1) * stride<2>(filter_layout)
        - int32_t(threadmap_iterations_stride - 1) * threadmap_delta_stride * stride<0>(filter_layout)
      ) * element_bytes;
      // printf("blockshape_kK: %d\n", blockshape_kK);
      // printf("threadmap_iterations_stride: %d\n", threadmap_iterations_stride);
      // printf("threadmap_delta_stride: %d\n", threadmap_delta_stride);
      // printf("inc_next_k: %d\n", inc_next_k);


  }
};

DEVICE_INLINE auto at(const cutlass::conv::Conv2dProblemSize& problem, int n, int p, int q, int r, int s, int filter_c)  {
  if (problem.mode == cutlass::conv::Mode::kConvolution) {
    r = problem.R - 1 - r;
    s = problem.S - 1 - s;
  }
  // r = problem.R - 1 - r;
  // s = problem.S - 1 - s;
  int h = p * problem.stride_h - problem.pad_h + r * problem.dilation_h;
  int w = q * problem.stride_w - problem.pad_w + s * problem.dilation_w;
  return make_coord(n, h, w, filter_c);
}


  /// Clears the predicates
template <int kStrided, typename T>
DEVICE_INLINE void clear_mask(bool clear, T (&mask)[kStrided][2]) {
  CUTLASS_PRAGMA_UNROLL
  for (int s = 0; s < kStrided; ++s) {

    mask[s][0] = clear ? 0 : mask[s][0];
    mask[s][1] = clear ? 0 : mask[s][1];
  }
}




DEVICE_INLINE  void clear_filter_mask(bool clear, uint32_t& predicates_) {
    predicates_ = clear ? 0u : predicates_;
}


///< D = relu(scale * (A @ B) + bias)
///< {N, P, Q, K} = {K} * (NPQK) + {K}
///< int32_t = float * int32_t + float
template<typename ElementA, typename ElementB, typename ElementC, typename ElementScaleBias,typename ElementOutput>
void host_kernel(const ElementScaleBias* scale, const ElementA* input, const ElementB* weight,
                 const ElementScaleBias* bias, ElementC* output,
                 int n, int h, int w, int c, int k, int r, int s, int p, int q, int padding_h, int padding_w,
                 int stride_h, int stride_w, int dilation_h, int dilation_w) {
  for (int n_i = 0; n_i < n; n_i++) {
    for (int p_i = 0; p_i < p; p_i++) {
      for (int q_i = 0; q_i < q; q_i++) {
        for (int k_i = 0; k_i < k; k_i++) {
          int acc = 0;
          for (int r_i = 0; r_i < r; r_i++) {
            for (int s_i = 0; s_i < s; s_i++) {
              for (int c_i = 0; c_i < c; c_i++) {

                int h_i = p_i * stride_h - padding_h + r_i * dilation_h;
                int w_i = q_i * stride_w - padding_w + s_i * dilation_w;

                if (h_i >= 0 && h_i < h && w_i >= 0 && w_i < w) {
                  int a = static_cast<int>(input[n_i * h*w*c + h_i * w*c + w_i * c + c_i]);
                  int b = static_cast<int>(weight[k_i * r*s*c + r_i * s*c + s_i * c + c_i]);

                  acc += a * b;
                }
              }
            }
          }
          // std::cout << acc << std::endl;
          // output[n_i * p*q*k + p_i * q*k + q_i * k + k_i] = \
          //   (scale[k_i] * float(acc) + bias[k_i] > 0.0f ? int(scale[k_i] * float(acc) + bias[k_i]) : 0);
          // output[n_i * p*q*k + p_i * q*k + q_i * k + k_i] = int(scale[k_i] * float(acc) + bias[k_i]);
          if constexpr (std::is_same_v<ElementOutput, int8_t>) {
            int res = int(scale[k_i] * float(acc) + bias[k_i]) > 127 ?
                          127 :
                          (int(scale[k_i] * float(acc) + bias[k_i]) < -128 ?
                               -128 :
                               int(scale[k_i] * float(acc) + bias[k_i]));
            output[n_i * p * q * k + p_i * q * k + q_i * k + k_i] = static_cast<int8_t>(res);
          } else {
            ElementScaleBias res =  ElementScaleBias(scale[k_i] * float(acc) + bias[k_i]);
            output[n_i * p * q * k + p_i * q * k + q_i * k + k_i] = static_cast<ElementC>(res);
          }
        }
      }
    }
  }
}