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

using namespace cute;
using Element = cutlass::half_t;
using ElementAccumulator = float;
static const int kE = 32;



// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
// if using m16n8k16, or to (4, MMA_M, MMA_N) if using m16n8k8.
template <typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
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

template <typename Element_, typename ElementAccumulator_, int Align_, int kE_,
          int Split_kF_, int Stages_>
struct KernelTraits {
  static const int kE = kE_;
  static const int Split_kF = Split_kF_;
  static const int Stages = Stages_;
  static const int Align = Align_;

  using Element = Element_;
  using ElementAccumulator = ElementAccumulator_;
  using ThreadblockShape0 = cutlass::gemm::GemmShape<64, 64, kE>;
  using WarpShape0 = cutlass::gemm::GemmShape<16, ThreadblockShape0::kN, kE>;
  // ThreadblockShape0::kK can be adapt?
  using ThreadblockShape1 =
      cutlass::gemm::GemmShape<ThreadblockShape0::kM, Split_kF,
                               ThreadblockShape0::kN>;
  using WarpShape1 =
      cutlass::gemm::GemmShape<WarpShape0::kM, Split_kF, ThreadblockShape0::kN>;
  static const int warp_num = ThreadblockShape0::kM / WarpShape0::kM;

  struct SharedStorage {
    cutlass::AlignedArray<Element, ThreadblockShape0::kM * kE> q_smem;
    cutlass::AlignedArray<Element, ThreadblockShape0::kN *
                                       ThreadblockShape0::kK * Stages>
        k_smem;
    cutlass::AlignedArray<Element, ThreadblockShape1::kK *
                                       ThreadblockShape1::kN * Stages>
        v_smem;
  };

  static const int q_continuous_thread = kE / 8;
  static const int v_continuous_thread = ThreadblockShape0::kN / 8;

  using T_Q = cutlass::AlignedArray<Element, Align>;
  using q_g2r_copy_op = UniversalCopy<T_Q>;
  using q_g2r_traits = Copy_Traits<q_g2r_copy_op>;
  using q_g2r_copy_atom = Copy_Atom<q_g2r_traits, T_Q>;
  using q_g2r_copy_tile = decltype(make_tiled_copy(
      q_g2r_copy_atom{},
      Layout<Shape<Int<32 * warp_num / q_continuous_thread>,
                   Int<q_continuous_thread>>,
             Stride<Int<q_continuous_thread>, _1>>{},
      Layout<Shape<_1, Int<Align>>>{}));

  using test_T_Q = cutlass::AlignedArray<Element, 1>;
  using test_q_g2r_copy_op = UniversalCopy<test_T_Q>;
  using test_q_g2r_traits = Copy_Traits<test_q_g2r_copy_op>;
  using test_q_g2r_copy_atom = Copy_Atom<test_q_g2r_traits, T_Q>;


  using T_K = cutlass::Array<Element, Align>;
  using k_g2r_copy_op = UniversalCopy<T_K>;
  using k_g2r_traits = Copy_Traits<k_g2r_copy_op>;
  using k_g2r_copy_atom = Copy_Atom<k_g2r_traits, T_K>;
  using k_g2r_copy_tile = decltype(make_tiled_copy(
      k_g2r_copy_atom{},
      Layout<Shape< Int<32 * warp_num / q_continuous_thread>,Int<q_continuous_thread>>,
             Stride<Int<q_continuous_thread>, _1>>{},
      Layout<Shape< _1, Int<Align>>>{}));
  // using k_g2r_copy_tile = decltype(make_tiled_copy(
  //     k_g2r_copy_atom{},
  //     Layout<Shape<Int<q_continuous_thread>,
  //                  Int<32 * warp_num / q_continuous_thread>>,
  //            Stride<_1, Int<q_continuous_thread>>>{},
  //     Layout<Shape<Int<Align>, _1>>{}));

  using T_V = cutlass::AlignedArray<Element, Align>;
  using v_g2r_copy_op = UniversalCopy<T_V>;
  using v_g2r_traits = Copy_Traits<v_g2r_copy_op>;
  using v_g2r_copy_atom = Copy_Atom<v_g2r_traits, T_V>;
  using v_g2r_copy_tile = decltype(make_tiled_copy(
      v_g2r_copy_atom{},
      Layout<Shape<Int<32 * warp_num / v_continuous_thread>,
                   Int<v_continuous_thread>>,
             Stride<Int<v_continuous_thread>, _1>>{},
      Layout<Shape<_1, Int<Align>>>{}));
  // using v_g2r_copy_tile = decltype(make_tiled_copy(
  //     v_g2r_copy_atom{},
  //     Layout<Shape<Int<32 * warp_num / q_continuous_thread>,
  //                  Int<q_continuous_thread>>,
  //            Stride<Int<q_continuous_thread>, _1>>{},
  //     Layout<Shape<_1, Int<Align>>>{}));


  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  using MMA = decltype(make_tiled_mma(
      mma_atom{},
      make_layout(Shape<Int<ThreadblockShape0::kM / WarpShape0::kM>, _1, _1>{}), // 描述了warp的排布状态，此时是1*1的排布
      make_layout(
          Shape<
              _1, Int<ThreadblockShape0::kN / 8>,
              _1>{}))); // 描述了每个线程持有的mma寄存器的重复状态，并没有发生寄存器重用，只是扩展，与tile上的重复状态定义重合，所以在3.4后删除？
};
template <typename KernelTraits>
__global__ void mha(MultiHeadAttentionProblemSize params, Element *q,
                    Element *k, Element *v, Element *out) {
  static const int kE = KernelTraits::kE;
  static const int Split_kF = KernelTraits::Split_kF;
  static const int Stages = KernelTraits::Stages;
  using SharedStorage = typename KernelTraits::SharedStorage;
  extern __shared__ int s[];

  SharedStorage &smem = *(reinterpret_cast<SharedStorage *>(s));
  auto global_q =
      make_tensor(make_gmem_ptr(q), make_shape(params.H, params.S, Int<kE>{}),
                  make_stride(params.S * Int<kE>{}, Int<kE>{}, _1{}));
  auto global_k =
      make_tensor(make_gmem_ptr(k), make_shape(params.H, params.T, Int<kE>{}),
                  make_stride(params.T * Int<kE>{}, Int<kE>{}, _1{}));
  // auto global_v =
  //     make_tensor(make_gmem_ptr(v), make_shape(params.H, params.F, params.T),
  //                 make_stride(params.T * params.F, _1{}, params.F));
  auto global_v =
      make_tensor(make_gmem_ptr(v), make_shape(params.H, params.F, params.T),
                  make_stride(params.T * params.F, params.T, _1{}));

  auto global_out =
      make_tensor(make_gmem_ptr(out), make_shape(params.H, params.S, params.F),
                  make_stride(params.S * params.F, params.F, _1{}));

  auto block_q = coalesce(local_tile(
      global_q,
      make_shape(Int<1>{}, Int<KernelTraits::ThreadblockShape0::kM>{},
                 Int<kE>{}),
      make_coord(blockIdx.z, blockIdx.x, 0)));
  auto block_k =
      coalesce(local_tile(global_k, make_shape(Int<1>{}, params.T, Int<kE>{}),
                          make_coord(blockIdx.z, 0, 0)));
  auto block_v = coalesce(
      local_tile(global_v, make_shape(Int<1>{}, Int<Split_kF>{}, params.T),
                 make_coord(blockIdx.z, blockIdx.y, 0)));
  auto block_out = coalesce(local_tile(
      global_out,
      make_shape(Int<1>{}, Int<KernelTraits::ThreadblockShape0::kM>{},
                 Int<Split_kF>{}),
      make_coord(blockIdx.z, blockIdx.x, blockIdx.y)));

  auto tiled_block_k = local_tile(
      block_k,
      make_shape(Int<KernelTraits::ThreadblockShape0::kN>{}, Int<kE>{}),
      make_coord(_, 0));
  auto tiled_block_v = local_tile(
      block_v,
      make_shape(Int<Split_kF>{}, Int<KernelTraits::ThreadblockShape1::kK>{}),
      make_coord(0, _));

  auto smem_q = make_tensor(
      make_smem_ptr((Element *)&smem.q_smem),
      make_shape(Int<KernelTraits::ThreadblockShape0::kM>{}, Int<kE>{}),
      make_stride(Int<kE>{}, _1{}));
  auto smem_k = make_tensor(
      make_smem_ptr((Element *)&smem.k_smem),
      make_shape(Int<KernelTraits::ThreadblockShape0::kN>{}, Int<kE>{}),
      make_stride(Int<kE>{}, _1{}));
  auto smem_v = make_tensor(
      make_smem_ptr((Element *)&smem.v_smem),
      make_shape(Int<Split_kF>{}, Int<KernelTraits::ThreadblockShape1::kK>{}),
      make_stride(Int<KernelTraits::ThreadblockShape1::kK>{}, _1{}));

  typename KernelTraits::q_g2r_copy_tile q_tiled_copy;
  auto q_g2s_thr_copy = q_tiled_copy.get_slice(threadIdx.x);
  auto tQgQ = q_g2s_thr_copy.partition_S(block_q);
  auto tQsQ = q_g2s_thr_copy.partition_D(smem_q);
  auto tQrQ = make_fragment_like(tQgQ(_, 0, 0));

#pragma unroll
  for (int i = 0; i < size<1>(tQgQ); ++i) {
#pragma unroll
    for (int j = 0; j < size<2>(tQgQ); ++j) {
      // cute::copy(tQgQ(_, i, j), tQsQ(_, i, j));
      cute::copy(tQgQ(_, i, j), tQrQ);
      cute::copy(tQrQ, tQsQ(_, i, j));
    }
  }
  // __syncthreads();

  auto tiled_smem_q = local_tile(
      smem_q, make_shape(Int<KernelTraits::ThreadblockShape0::kM>{}, _16{}),
      make_coord(0, _));

  typename KernelTraits::k_g2r_copy_tile k_tiled_copy;

  auto k_g2s_thr_copy = k_tiled_copy.get_slice(threadIdx.x);

  auto tKgK = k_g2s_thr_copy.partition_S(tiled_block_k(_, _, 0));
  auto tKsK = k_g2s_thr_copy.partition_D(smem_k);
  auto tKrK = make_fragment_like(tKgK(_, 0, 0));

#pragma unroll
  for (int i = 0; i < size<1>(tKgK); ++i) {
#pragma unroll
    for (int j = 0; j < size<2>(tKgK); ++j) {
      // cute::copy(tKgK(_, i, j), tKsK(_, i, j));
      cute::copy(tKgK(_, i, j),  tKrK);
      cute::copy(tKrK,  tKsK(_, i, j));
    }
  }
  // __syncthreads();

  typename KernelTraits::v_g2r_copy_tile v_tiled_copy;

  auto v_g2s_thr_copy = v_tiled_copy.get_slice(threadIdx.x);

  auto tVgV = v_g2s_thr_copy.partition_S(tiled_block_v(_, _, 0));
  auto tVsV = v_g2s_thr_copy.partition_D(smem_v);
  auto tVrV = make_fragment_like(tVgV(_, 0, 0));

#pragma unroll
  for (int i = 0; i < size<1>(tVgV); ++i) {
#pragma unroll
    for (int j = 0; j < size<2>(tVgV); ++j) {
      // cute::copy(tVgV(_, i, j), tVsV(_, i, j));
      cute::copy(tVgV(_, i, j),  tVrV);
      cute::copy(tVrV,  tVsV(_, i, j));
    }
  }
  __syncthreads();
  // cp_async_fence();


  typename KernelTraits::MMA tiled_mma;
  // auto s2r_tiled_copy_q = make_tiled_copy_A(typename KernelTraits::test_q_g2r_copy_atom{}, tiled_mma);
  // auto s2r_thr_copy_q = s2r_tiled_copy_q.get_slice(threadIdx.x);
  //   auto thr_QsQ =
  //     s2r_thr_copy_q.partition_S(smem_q(_, _));


  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  auto thr_QsQ =
      thr_mma.partition_A(smem_q(_, _)); // (MMA, MMA_M, MMA_K, num_tile_k)
  auto thr_KsK =
      thr_mma.partition_B(smem_k(_, _)); // (MMA, MMA_N, MMA_K, num_tile_k)
  // auto thr_CsC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N)

  auto thr_tQrQ = thr_mma.partition_fragment_A(smem_q(_, _)); // (MMA, MMA_M, MMA_K)
  // auto thr_tQrQ = thr_mma.partition_fragment_A(block_q(_, _)); // (MMA, MMA_M, MMA_K)
  // auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N,
  // MMA_K)
  auto thr_tKrK =
      thr_mma.partition_fragment_B(smem_k(_, _)); // (MMA, MMA_N, MMA_K)
  auto thr_tCrC = partition_fragment_C(
      tiled_mma,
      Shape<Int<KernelTraits::ThreadblockShape0::kM>,
            Int<KernelTraits::ThreadblockShape0::kN>>{}); // (MMA, MMA_M, MMA_N)

  if (thread0()) {
    // PRINT(layout((tiled_block_v)));
    // PRINT(layout((tiled_block_k)));
    // PRINT(layout(block_out));

    // PRINT(layout(smem_v));
    // PRINT(layout(smem_k));
    // PRINT(layout(tKgK));
    // PRINT(layout(tKsK));
    // PRINT(layout(thr_QsQ));
    // PRINT(layout(thr_KsK));
    // PRINT(layout(thr_tQrQ));
    // PRINT(layout(thr_tKrK));
    // PRINT(layout(thr_tCrC));
  }
// #pragma unroll
//   for (int i = 0; i < size<1>(thr_QsQ); ++i) {
// #pragma unroll
//     for (int j = 0; j < size<2>(thr_QsQ); ++j) {
//       for (int m = 0; m < 2; ++m) {
//         for (int n = 0; n < 2; ++n) {
//           for (int k = 0; k < 2; ++k) {
//       cute::copy(thr_QsQ(k + n *2 + m *4, i, j), thr_tQrQ((k + n *2 + m *4), i, j));

//           }
//         }
//       }

//   }

  // cp_async_wait<0>();

  // __syncthreads();

  cute::copy(thr_QsQ, thr_tQrQ);
  // cute::copy(s2r_tiled_copy_q, thr_QsQ, thr_tQrQ);

  cute::copy(thr_KsK, thr_tKrK);
  // __syncthreads();

  //   if (thread0()) {
  //     PRINT(layout(thr_QsQ));
  //     PRINT((thr_tQrQ));
  //     //         for (int i = 0; i < 8; ++i) {
  //     //   printf("thr_QsQ(i): %f\n", i,
  //     //          (float)(thr_QsQ(i)));
  //     // }
  //             for (int i = 0; i < 2; ++i) {
  //       printf("*(thr_QsQ.data() + %d): %f\n", i,
  //              (float)*(thr_QsQ.data() + i));
  //     }
  //     printf("cnm\n");

  //             for (int i = 0; i < 4; ++i) {
  //       printf("thr_KsK(i): %f\n", i,
  //              (float)(thr_KsK(i)));
  //     }
  //             for (int i = 0; i < 2; ++i) {
  //       printf("*(thr_KsK.data() + %d): %f\n", i,
  //              (float)*(thr_KsK.data() + i));
  //     }
  //     printf("cnm\n");
  //     //      for (int i = 0; i < 8; ++i) {
  //     //   printf("thr_tQrQ(i): %f\n", i,
  //     //          (float)(thr_tQrQ(i)));
  //     // }
  //     //            for (int i = 0; i < 4; ++i) {
  //     //   printf("*(thr_tKrK.data() + %d) : %f\n", i,
  //     //          (float)*(thr_tKrK.data() + i));
  //     // }
  //         for (int i = 0; i < 8; ++i) {
  //       printf("*(thr_tQrQ.data() + %d): %f\n", i,
  //              (float)*(thr_tQrQ.data() + i));
  //     }
  //           for (int i = 0; i < 4; ++i) {
  //       printf("*(thr_tKrK.data() + %d) : %f\n", i,
  //              (float)*(thr_tKrK.data() + i));
  //     }
  // }

    //   if (thread0()) {
    //   for (int i = 0; i < 4; ++i) {
    //     printf("thr_tKrK[%d]:%f\n", i, (float)*((Element*)thr_tKrK.data() + i));
    //     printf("thr_KsK[%d]:%f\n", i, (float)(thr_KsK[i]));
    //   }
    //   PRINT(layout(thr_KsK));
    //   PRINT(layout(thr_tKrK));
    // }

  clear(thr_tCrC);
  cute::gemm(tiled_mma, thr_tCrC, thr_tQrQ, thr_tKrK, thr_tCrC);


  // if (thread0()) {
  //         for (int i = 0; i < 8; ++i) {
  //       printf("threadIdx.x: %d, val: %f\n", threadIdx.x,
  //              (float)*(thr_tQrQ.data() + i));
  //     }
  //           for (int i = 0; i < 4; ++i) {
  //       printf("threadIdx.x: %d, val: %f\n", threadIdx.x,
  //              (float)*(thr_tKrK.data() + i));
  //     }
  //     for (int i = 0; i < 4; ++i) {
  //       printf("threadIdx.x: %d, val: %f\n", threadIdx.x,
  //              (float)*(thr_tCrC.data() + i));
  //     }
  // }
#if 1
  using ElementAccumulator = typename KernelTraits::ElementAccumulator;
  ElementAccumulator tCrC_max[2];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    tCrC_max[i] = -INFINITY;
  }

#pragma unroll
  for (int i = 0; i < size<2>(thr_tCrC); ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const int offset = m + 2 * j + i * 4;
        tCrC_max[j] = max(tCrC_max[j], (ElementAccumulator)*(thr_tCrC.data() + offset));
      }
    }
  }

#pragma unroll
  for (int i = 1; i <= 2; i <<= 1) {
    tCrC_max[0] = max(__shfl_xor_sync(0xffffffff, tCrC_max[0], i), tCrC_max[0]);
    tCrC_max[1] = max(__shfl_xor_sync(0xffffffff, tCrC_max[1], i), tCrC_max[1]);
  }

  ElementAccumulator d[2];
  d[0] = 0;
  d[1] = 0;
  ElementAccumulator exp_tcrc_sum[2];
  exp_tcrc_sum[0] = 0;
  exp_tcrc_sum[1] = 0;

  auto exp_tcrc = make_tensor<ElementAccumulator>(shape(thr_tCrC));

#pragma unroll
  for (int i = 0; i < size<2>(thr_tCrC); ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
      for (int m = 0; m < 2; ++m) {

        // exp_tcrc((m, j), 0, i) =
        //     exp((ElementAccumulator)thr_tCrC((m, j), 0, i) - tCrC_max[j]);
        // exp_tcrc_sum[j] += exp_tcrc((m, j), 0, i);
        const int offset = m + j * 2 + i * 4;
        *(exp_tcrc.data() + offset) = exp((ElementAccumulator)(*(thr_tCrC.data() + offset)) - tCrC_max[j]);
        exp_tcrc_sum[j] += (ElementAccumulator)(*(exp_tcrc.data() + offset));

        if (thread0()) {
          // printf("thr_tCrC((%d, %d), 0, %d): %f\n", m, j, i,(float)(thr_tCrC((m, j), 0, i)));
          // printf("exp_tcrc((%d, %d), 0, %d): %f\n", m, j, i, (float)(exp_tcrc((m, j), 0, i)));
        }
      }
    }
  }

  if (thread0()) {

    // printf("exp_tcrc_sum[%d]:%f\n", 0, (float)exp_tcrc_sum[0]);
    // printf("exp_tcrc_sum[%d]:%f\n", 1, (float)exp_tcrc_sum[1]);

  }
#pragma unroll
  for (int i = 1; i <= 2; i <<= 1) {
    exp_tcrc_sum[0] += __shfl_xor_sync(0xffffffff, exp_tcrc_sum[0], i);
    exp_tcrc_sum[1] += __shfl_xor_sync(0xffffffff, exp_tcrc_sum[1], i);
  }

  d[0] = exp_tcrc_sum[0];
  d[1] = exp_tcrc_sum[1];

  if (thread0()) {
    // printf("d[%d]:%f\n", 0, (float)d[0]);
    // printf("d[%d]:%f\n", 1, (float)d[1]);
    // printf("exp_tcrc_sum[%d]:%f\n", 0, (float)exp_tcrc_sum[0]);
    // printf("exp_tcrc_sum[%d]:%f\n", 1, (float)exp_tcrc_sum[1]);
    // printf("tCrC_max[%d]:%f\n", 0, (float)tCrC_max[0]);
    // printf("tCrC_max[%d]:%f\n", 1, (float)tCrC_max[1]);
    // PRINT(layout(thr_tCrC));
  }

#pragma unroll
  for (int i = 0; i < size<2>(thr_tCrC); ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        // thr_tCrC((m, j), 0, i) = (Element)(exp_tcrc((m, j), 0, i) / d[j]);
        const int offset = m + j * 2 + i * 4;
        *(thr_tCrC.data() + offset) = (Element)(*(exp_tcrc.data() + offset) / d[j]);

        // if (thread0()) {
        //   printf("thr_tCrC((%d, %d), 0, %d): %f\n", m, j, i,
        //   (float)(thr_tCrC((m, j), 0, i))); printf("exp_tcrc((%d, %d), 0,
        //   %d): %f\n", m, j, i, (float)(exp_tcrc((m, j), 0, i)));
        //   printf("d[%d]: %f\n", m, (float)d[j]);
        // }
      }
    }
  }

  typename KernelTraits::MMA tiled_mma1;

  auto thr_mma1 = tiled_mma1.get_slice(threadIdx.x);
  auto mma1_tVsV = thr_mma1.partition_B(smem_v);
  auto mma1_tOgO = thr_mma1.partition_C(block_out);

  auto mma1_tVrV = thr_mma1.partition_fragment_B(smem_v);
  auto mma1_tOrO = thr_mma1.partition_fragment_C(block_out(_, _));

  clear(mma1_tOrO);

  cute::copy(mma1_tVsV, mma1_tVrV);
    //   if (thread0()) {
    //   for (int i = 0; i < 4; ++i) {
    //     printf("mma1_tVsV[%d]:%f\n", i, (float)mma1_tVsV[i]);
    //     printf("mma1_tVrV[%d]:%f\n", i, (float)mma1_tVrV[i]);
    //   }

    // }
  Tensor tOrrC = make_tensor(
      thr_tCrC.data(),
      convert_layout_acc_Aregs<KernelTraits::mma_traits>(thr_tCrC.layout()));

  cute::gemm(tiled_mma1, mma1_tOrO, tOrrC, mma1_tVrV, mma1_tOrO);

  //   if (thread0()) {
  //     printf("cnm?!!!\n");
  //         for (int i = 0; i < 8; ++i) {
  //       printf("threadIdx.x: %d, val: %f\n", threadIdx.x,
  //              (float)*(tOrrC.data() + i));
  //     }
  //           for (int i = 0; i < 4; ++i) {
  //       printf("threadIdx.x: %d, val: %f\n", threadIdx.x,
  //              (float)*(mma1_tVrV.data() + i));
  //     }
  //     for (int i = 0; i < 4; ++i) {
  //       printf("threadIdx.x: %d, val: %f\n", threadIdx.x,
  //              (float)*(mma1_tOrO.data() + i));
  //     }
  // }
  // if (thread0()) {
  //   PRINT(layout((mma1_tVsV)));
  //   PRINT(layout((mma1_tVsV(_,_,_,0))));
  //   PRINT(layout(mma1_tVrV));
  //   PRINT(layout(tOrrC));
  //   PRINT(layout(mma1_tOrO));
  //   PRINT(layout(mma1_tOgO));
  // }


  #pragma unroll
    for(int i = 0; i < size<2>(mma1_tOrO); ++i) {
  #pragma unroll
      for (int j = 0; j < 2; ++j) {
  #pragma unroll
        for (int m = 0; m < 2; ++m) {
          if (thread0()) {
          // printf("phase0 mma1_tOrO((%d, %d), 0, %d): %f\n", m, j, i,
          //        (float)(*(mma1_tOrO.data() + m + j * 2 + i * 4)));
        }
        }
      }
    }

  int K = (params.T + KernelTraits::ThreadblockShape0::kN - 1) /
          KernelTraits::ThreadblockShape0::kN;
  if (thread0()) {
    // printf("params.T:%d\n", params.T);
    // printf("KernelTraits::ThreadblockShape0::kN:%d\n", KernelTraits::ThreadblockShape0::kN);
    // printf("K:%d\n", K);
  }
  for (int k = 1; k < K; ++k) {
    tKgK = k_g2s_thr_copy.partition_S(tiled_block_k(_, _, k));

#pragma unroll
    for (int i = 0; i < size<1>(tKgK); ++i) {
#pragma unroll
      for (int j = 0; j < size<2>(tKgK); ++j) {
        // cute::copy(tKgK(_, i, j), tKsK(_, i, j));
        cute::copy(tKgK(_, i, j),  tKrK);
        cute::copy(tKrK,  tKsK(_, i, j));
      }
    }

    tVgV = v_g2s_thr_copy.partition_S(tiled_block_v(_, _, k));
#pragma unroll
    for (int i = 0; i < size<1>(tVgV); ++i) {
#pragma unroll
      for (int j = 0; j < size<2>(tVgV); ++j) {
        // cute::copy(tVgV(_, i, j), tVsV(_, i, j));
        cute::copy(tVgV(_, i, j),  tVrV);
        cute::copy(tVrV,  tVsV(_, i, j));

      }
    }

    __syncthreads();

    cute::copy(thr_KsK, thr_tKrK);
    if (thread0()) {
      for (int i = 0; i < 4; ++i) {
        // printf("thr_tKrK[%d]:%f\n", i, (float)thr_tKrK[i]);
        // printf("thr_KsK[%d]:%f\n", i, (float)thr_KsK[i]);
      }

    }
    clear(thr_tCrC);
    //Q的顺序为内部->col->row, 为何不是内部->row->col?
    cute::gemm(tiled_mma, thr_tCrC, thr_tQrQ, thr_tKrK, thr_tCrC);
    if (thread0()) {
      // PRINT(layout(thr_tQrQ));
      // PRINT(layout(thr_tKrK));
      // PRINT(layout(thr_tCrC));
      // PRINT((thr_tQrQ));
      // PRINT((thr_tKrK));
      // PRINT((thr_tCrC));
    }



    ElementAccumulator new_tCrC_max[2];
    ElementAccumulator exp_oldm_sub_newm[2];
    ElementAccumulator new_m_max[2];

    for (int i = 0; i < 2; ++i) {
      new_tCrC_max[i] = -INFINITY;
      exp_oldm_sub_newm[i] = -INFINITY;
      new_m_max[i] = -INFINITY;
    }


    ElementAccumulator new_d[2]{0};
    ElementAccumulator new_exp_tcrc_sum[2]{0};
    ElementAccumulator new_o_scale[2]{0.f};

#pragma unroll
    for (int i = 0; i < size<2>(thr_tCrC); ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          const int offset = m + 2 * j + i * 4;
          new_tCrC_max[j] =
              max(new_tCrC_max[j], (ElementAccumulator)*(thr_tCrC.data() + offset));
        }
      }
    }
#pragma unroll
    for (int i = 1; i <= 2; i <<= 1) {

      new_tCrC_max[0] =
          max(__shfl_xor_sync(0xffffffff, new_tCrC_max[0], i), new_tCrC_max[0]);
      new_tCrC_max[1] =
          max(__shfl_xor_sync(0xffffffff, new_tCrC_max[1], i), new_tCrC_max[1]);
    }

    new_m_max[0] = max(tCrC_max[0], new_tCrC_max[0]);
    new_m_max[1] = max(tCrC_max[1], new_tCrC_max[1]);
    exp_oldm_sub_newm[0] = exp(tCrC_max[0] - new_m_max[0]);
    exp_oldm_sub_newm[1] = exp(tCrC_max[1] - new_m_max[1]);



    auto new_exp_tcrc = make_tensor<ElementAccumulator>(shape(thr_tCrC));

#pragma unroll
    for (int i = 0; i < size<2>(thr_tCrC); ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          // new_exp_tcrc((m, j), 0, i) =
          //     exp((ElementAccumulator)thr_tCrC((m, j), 0, i) - new_tCrC_max[j]);
          // new_exp_tcrc_sum[j] += new_exp_tcrc((m, j), 0, i);

        const int offset = m + j * 2 + i * 4;
        *(new_exp_tcrc.data() + offset) = exp((ElementAccumulator)(*(thr_tCrC.data() + offset)) - new_m_max[j]);
        new_exp_tcrc_sum[j] += (ElementAccumulator)(*(new_exp_tcrc.data() + offset));

          //   if (thread0()) {
          //   printf("thr_tCrC((%d, %d), 0, %d): %f\n", m, j, i,
          //   (float)(thr_tCrC((m, j), 0, i))); printf("exp_tcrc((%d, %d), 0,
          //   %d): %f\n", m, j, i, (float)(exp_tcrc((m, j), 0, i)));
          // }
        }
      }
    }
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

    new_o_scale[0] = d[0] * exp_oldm_sub_newm[0] / new_d[0];
    new_o_scale[1] = d[1] * exp_oldm_sub_newm[1] / new_d[1];
    // if (thread0()) {
    //   printf("new_o_scale[0]:%f\n", new_o_scale[0]);
    //   printf("new_o_scale[1]:%f\n", new_o_scale[1]);
    //   printf("new_d[0]:%f\n", new_d[0]);
    //   printf("new_d[1]:%f\n", new_d[1]);
    //   printf("d[0]:%f\n", d[0]);
    //   printf("d[1]:%f\n", d[1]);
    //   printf("exp_oldm_sub_newm[0]:%f\n", exp_oldm_sub_newm[0]);
    //   printf("exp_oldm_sub_newm[1]:%f\n", exp_oldm_sub_newm[1]);
    //   printf("new_m_max[0]:%f\n", new_m_max[0]);
    //   printf("new_m_max[1]:%f\n", new_m_max[1]);
    //   printf("new_exp_tcrc_sum[0]:%f\n", new_exp_tcrc_sum[0]);
    //   printf("new_exp_tcrc_sum[1]:%f\n", new_exp_tcrc_sum[1]);
    // }

#pragma unroll
    for (int i = 0; i < size<2>(mma1_tOrO); ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          // mma1_tOrO((m, j), 0, i) *= (Element)new_o_scale[j];

          const int offset = m + 2 * j + i * 4;
          *(mma1_tOrO.data() + offset) =  *(mma1_tOrO.data() + offset) * (Element)new_o_scale[j];
        }
      }
    }

#pragma unroll
    for (int i = 0; i < size<2>(thr_tCrC); ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          // thr_tCrC((m, j), 0, i) =
          //     (Element)exp(
          //         ((ElementAccumulator)thr_tCrC((m, j), 0, i) - new_m_max[j])) /
          //     new_d[j];

        const int offset = m + 2 * j + i * 4;
                  *(thr_tCrC.data() + offset) =
              exp(
                  ((ElementAccumulator)(*(thr_tCrC.data() + offset)) - new_m_max[j])) /
              new_d[j];
        // if (thread0()) {
        //   printf("thr_tCrC((%d, %d), 0, %d): %f\n", m, j, i,
        //          (float)(*(thr_tCrC.data() + offset)));
        // }



        }
      }
    }

    cute::copy(mma1_tVsV, mma1_tVrV);
    if (thread0()) {
      for (int i = 0; i < 4; ++i) {
        // printf("phase0 mma1_tVsV[%d]:%f\n", i, (float)mma1_tVsV[i]);
        // printf("phase0 mma1_tVrV[%d]:%f\n", i, (float)mma1_tVrV[i]);
      }

    }
    cute::gemm(tiled_mma1, mma1_tOrO, tOrrC, mma1_tVrV, mma1_tOrO);

    d[0] = new_d[0];
    d[1] = new_d[1];
    tCrC_max[0] = new_m_max[0];
    tCrC_max[1] = new_m_max[1];
  }

  cute::copy(mma1_tOrO, mma1_tOgO);
  if (thread0()) {
    // PRINT(layout((mma1_tOrO)));
    // PRINT(layout((mma1_tOgO)));
  }
  auto mma1_tOrO_data = mma1_tOrO.data();

#pragma unroll
  for (int i = 0; i < size<2>(mma1_tOrO); ++i) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        if (thread0()) {
          // printf("mma1_tOrO((%d, %d), 0, %d): %f\n", m, j, i,
          //        (float)(*(mma1_tOrO_data + m + j * 2 + i * 4)));
        }
      }
    }
  }


#endif
}

/**
 *LAYOUT:
 *Q: N H L E
 *K: N H S E
 *V: N H Ev S(S 连续)
 *OUT: N H L Ev
*/

int main() {
  // int N = 1, S = 16, T = 64, F = 32, H = 1;
  int N = 1, S = 1024, T = 1024, F = 128, H = 1;
  // int N = 1, S = 64, T = 128, F = 32, H = 1;
  // int N = 1, S = 16, T = 32, F = 32, H = 1;
  MultiHeadAttentionProblemSize problem(N, S, T, kE, F, H);
  using KernelT = KernelTraits<cutlass::half_t, float, 8, 32, 32, 1>;

  MemHelper<KernelT::Element> helper;
  auto Q = helper.GetCpuGpuBuffer(problem.QuerySize());
  // auto Q = helper.GetCpuGpuBuffer(problem.QuerySize(), InitialType::Increment);
  auto K = helper.GetCpuGpuBuffer(problem.KeySize());
  auto V = helper.GetCpuGpuBuffer(problem.ValueSize());
  // auto Q = helper.GetCpuGpuBuffer(problem.QuerySize(), InitialType::AllOne);
  // auto K = helper.GetCpuGpuBuffer(problem.KeySize(), InitialType::AllOne);
  // auto V = helper.GetCpuGpuBuffer(problem.ValueSize(), InitialType::AllOne);
  auto Out = helper.GetCpuGpuBuffer(problem.OutputSize());

  dim3 block = dim3{32 * KernelT::warp_num};
  dim3 grid = dim3(
      (S + KernelT::ThreadblockShape0::kM - 1) / KernelT::ThreadblockShape0::kM,
      (F + KernelT::ThreadblockShape1::kN - 1) / KernelT::ThreadblockShape1::kN,
      H);
  int smem_s = sizeof(KernelT::SharedStorage);

  // auto s = KernelT::SharedStorage{};
  // printf("smem bytes size:%d\n", smem_s);
  // printf("q smem bytes size:%d\n", s.q_smem.max_size() * 2);
  // printf("k smem bytes size:%d\n", s.k_smem.max_size() * 2);
  // printf("v smem bytes size:%d\n", s.v_smem.max_size() * 2);

  if (smem_s >= (48 << 10)) {
    cudaError_t err = cudaFuncSetAttribute(
        mha<KernelT>,
        cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, smem_s);
    if (err != cudaError_t::cudaSuccess) {
      std::cout << "initialize error: " << cudaGetErrorString(err) << "{\n"
                << "    shared memory size: " << smem_s << "Bytes\n"
                << "}" << std::endl;
      exit(-1);
    }
  }

  mha<KernelT><<<grid, block, smem_s>>>(
      problem, (Element *)(Q.second), (Element *)(K.second),
      (Element *)(V.second), (Element *)(Out.second));

  helper.SyncGpuToCpu(Out);

  cudaError_t err = cudaGetLastError();

  if (err != cudaError_t::cudaSuccess) {
    std::cout << "final error: " << cudaGetErrorString(err) << std::endl;
    exit(-1);
  }
  Element *C_gt = (Element *)helper.GetCpuBuffer(S * T);
  Element *m_gt = (Element *)helper.GetCpuBuffer(S);
  Element *res_gt = (Element *)helper.GetCpuBuffer(S * F);


  Element* cpu_a = (Element*)Q.first;
  Element* cpu_b = (Element*)K.first;
  Element* cpu_d = (Element*)V.first;

  int m = S;
  int n = T;
  int k = kE;
  for (int i = 0 ; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      ElementAccumulator sum = (ElementAccumulator)0.f;
      for (int l = 0; l < k; ++l) {
        sum += cpu_a[i * k + l] * cpu_b[l + j * k];
      }
      // printf("sum:%f\n", (float)sum);
      C_gt[i * n + j] = (Element)sum;
    }
  }

  for (int i = 0; i < m; ++i) {
    ElementAccumulator m_max = -INFINITY;
    for (int j = 0; j < n; ++j) {
      m_max= max(m_max, (ElementAccumulator)C_gt[i * n + j]);
    }
    m_gt[i] = m_max;
  }

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      C_gt[i * n + j] = (Element)exp(C_gt[i * n + j] - m_gt[i]);
    }
  }

  for (int i = 0; i < m; ++i) {
    ElementAccumulator sum = (ElementAccumulator)0.f;
    for (int j = 0; j < n; ++j) {
      sum += C_gt[i * n + j];
    }
    for (int j = 0; j < n; ++j) {
      C_gt[i * n + j] = C_gt[i * n + j] / (Element)sum;
    }
  }

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < F; ++j) {
      ElementAccumulator sum = (ElementAccumulator)0;
      for (int p = 0; p < n; ++p) {
        // sum += C_gt[i * n + p] * cpu_d[j + p * F];
        sum += C_gt[i * n + p] * cpu_d[j * n + p];
      }
      res_gt[i * F + j] = (Element)sum;
    }
  }


  helper.Regression(res_gt, (Element*)Out.first);


}