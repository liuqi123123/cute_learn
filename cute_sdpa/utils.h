#pragma once
#include "cute/numeric/integral_constant.hpp"
#include "cute/pointer.hpp"
#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/array_subbyte.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "include/cuda_helper.h"
#include "sdpa_problem.h"
#include <cmath>
#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <iostream>
#include "cutlass/half.h"

#define DEVICE_INLINE __device__ __forceinline__
using namespace cute;


// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
// if using m16n8k16, or to (4, MMA_M, MMA_N) if using m16n8k8.
template <typename MMA_traits, typename Layout>
DEVICE_INLINE auto convert_layout_acc_Aregs(Layout acc_layout) {
  using X = Underscore;
  static_assert(decltype(size<0>(acc_layout))::value == 4);
  static_assert(decltype(rank(acc_layout))::value == 3);
  constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
  static_assert(mma_shape_K == 8 || mma_shape_K == 16);
  if constexpr (mma_shape_K == 8) {
    return acc_layout;
  }
  auto l = logical_divide(acc_layout,
                          Shape<X, X, _2>{}); // (4, MMA_M, (2, MMA_N / 2)))
  return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l),
                     get<2, 1>(l));
};


template<typename T, int Size>
DEVICE_INLINE void Initial_1D_Regs(T (&R)[Size], T value) {
  #pragma unroll
  for (int s = 0; s < Size; ++s) {
    R[s] = value;
  }
}


template<int S0, int S1, typename Func>
DEVICE_INLINE void Operator_2D_Regs(const Func& f){
  #pragma unroll
  for (int i = 0 ; i < S0; ++i) {
  #pragma unroll
    for (int j = 0; j < S1; ++j) {
        f(i, j);
    }
  }
}

template<int S0, int S1, int S2, typename Func>
DEVICE_INLINE void Operator_3D_Regs(const Func& f){
      #pragma unroll
    for (int i = 0; i < S0; ++i) {
        #pragma unroll
        for (int j = 0; j < S1; ++j) {
            #pragma unroll
            for (int m = 0; m < S2; ++m) {
                f(i, j, m);
            }
        }
    }
}

template <typename Element, typename ElementAccumulator, typename TCrC, typename T,
          int Size>
DEVICE_INLINE void Operator_Softmax(TCrC &thr_tCrC,
                                    T (&exp_oldm_sub_newm)[Size],
                                    T (&d)[Size],
                                    T (&m_max)[Size],
                                    int itile
                                    ) {
    ElementAccumulator new_tCrC_max[2];
    ElementAccumulator new_m_max[2];
    ElementAccumulator new_d[2];
    ElementAccumulator new_exp_tcrc_sum[2];
    Initial_1D_Regs<ElementAccumulator, 2>(new_tCrC_max, -INFINITY);

    Initial_1D_Regs<ElementAccumulator, 2>(new_m_max, -INFINITY);
    Initial_1D_Regs<ElementAccumulator, 2>(new_d, 0.0f);
    Initial_1D_Regs<ElementAccumulator, 2>(new_exp_tcrc_sum, 0.0f);

Operator_3D_Regs<size<2>(TCrC{}), 2, 2>(
    [&](int i, int j, int m) {
          new_tCrC_max[j] =
              max(new_tCrC_max[j], (ElementAccumulator)(thr_tCrC(make_coord(m, j), 0, i)));
    });

#pragma unroll
    for (int i = 1; i <= 2; i <<= 1) {

      new_tCrC_max[0] =
          max(__shfl_xor_sync(0xffffffff, new_tCrC_max[0], i), new_tCrC_max[0]);
      new_tCrC_max[1] =
          max(__shfl_xor_sync(0xffffffff, new_tCrC_max[1], i), new_tCrC_max[1]);
    }

    new_m_max[0] = max(m_max[0], new_tCrC_max[0]);
    new_m_max[1] = max(m_max[1], new_tCrC_max[1]);

    exp_oldm_sub_newm[0] = exp(m_max[0] - new_m_max[0]);
    exp_oldm_sub_newm[1] = exp(m_max[1] - new_m_max[1]);
    //when k == 0, don't want O to scale, so make exp_oldm_sub_newm == 1;
    if (itile==0) {
      exp_oldm_sub_newm[0] = (ElementAccumulator)1;
      exp_oldm_sub_newm[1] = (ElementAccumulator)1;
    }

Operator_3D_Regs<size<2>(TCrC{}), 2, 2>(
    [&](int i, int j, int m) {
        thr_tCrC(make_coord(m, j), 0, i) = static_cast<Element>(cutlass::fast_exp((
            thr_tCrC(make_coord(m, j), 0, i) - (Element)new_m_max[j])));
        new_exp_tcrc_sum[j] += (ElementAccumulator)(thr_tCrC(make_coord(m, j), 0, i));
    });

    if (thread0()) {
      // PRINT(layout(thr_tCrC));
      // printf("new_exp_tcrc_sum[0]:%f\n", new_exp_tcrc_sum[0]);
      // printf("new_exp_tcrc_sum[1]:%f\n", new_exp_tcrc_sum[1]);
    }

#pragma unroll
    for (int i = 1; i <= 2; i <<= 1) {
      new_exp_tcrc_sum[0] +=
          __shfl_xor_sync(0xffffffff, new_exp_tcrc_sum[0], i);
      new_exp_tcrc_sum[1] +=
          __shfl_xor_sync(0xffffffff, new_exp_tcrc_sum[1], i);
    }

    new_d[0] = d[0] * exp_oldm_sub_newm[0] + new_exp_tcrc_sum[0];
    new_d[1] = d[1] * exp_oldm_sub_newm[1] + new_exp_tcrc_sum[1];

    d[0] = new_d[0];
    d[1] = new_d[1];
    m_max[0] = new_m_max[0];
    m_max[1] = new_m_max[1];
}
