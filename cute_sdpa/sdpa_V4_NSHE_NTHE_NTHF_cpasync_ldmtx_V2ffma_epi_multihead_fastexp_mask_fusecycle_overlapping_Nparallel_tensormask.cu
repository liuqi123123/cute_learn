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
// #include <numeric>

#define CPU_CHECK
using namespace cute;
using Element = cutlass::half_t;
using ElementAccumulator = float;

static const int kE = 32;
static const bool UseMask = true;

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
          int Split_kF_, bool Mask_, int Align_Mask_>
struct KernelTraits {
  static const int kE = kE_;
  static const int Split_kF = Split_kF_;
  static const int Align = Align_;
  static const bool Mask = Mask_;
  static const int Align_Mask = Align_Mask_;

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

  struct SharedStorageNoMasked {
    cutlass::AlignedArray<Element, ThreadblockShape0::kM * kE> q_smem;
    cutlass::AlignedArray<Element, ThreadblockShape0::kN *
                                       ThreadblockShape0::kK>
        k_smem;
    cutlass::AlignedArray<Element, ThreadblockShape1::kK *
                                       ThreadblockShape1::kN>
        v_smem;
  };
  struct SharedStorageMasked {
    cutlass::AlignedArray<Element, ThreadblockShape0::kM * kE> q_smem;
    cutlass::AlignedArray<Element, ThreadblockShape0::kN *
                                       ThreadblockShape0::kK>
        k_smem;
    cutlass::AlignedArray<Element, ThreadblockShape0::kM * ThreadblockShape0::kN> m_smem;

    cutlass::AlignedArray<Element, ThreadblockShape1::kK *
                                       ThreadblockShape1::kN>
        v_smem;
  };
  using SharedStorage = typename std::conditional<Mask, SharedStorageMasked, SharedStorageNoMasked>::type;

  struct  EpilogueSharedStorage{
    cutlass::Array<Element, ThreadblockShape0::kM * Split_kF> e_smem;
  };

  static_assert(sizeof(SharedStorage) >= sizeof(EpilogueSharedStorage), "SharedStorage should be lager then epilogue shared smem");

  static const int q_continuous_thread = kE / 8;
  static const int v_continuous_thread = ThreadblockShape1::kN / 8;
  static const int mask_continuous_thread = ThreadblockShape0::kN / Align_Mask;

  static_assert((32 * warp_num) % mask_continuous_thread == 0, "mask_continuous_thread must be divisible by (32 * warp_num)");

  static const int epilogue_continuous_thread = Split_kF / 8;


  using q_g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using q_g2s_copy_traits = Copy_Traits<q_g2s_copy_op>;
  using q_g2s_copy_atom = Copy_Atom<q_g2s_copy_traits, Element>;



  using q_G2SCopy=
      decltype(make_tiled_copy(q_g2s_copy_atom{},
                               make_layout(make_shape(Int<32 * warp_num / q_continuous_thread>{}, Int<q_continuous_thread>{}),
                                           make_stride(Int<q_continuous_thread>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<Align>{}))));
  using k_G2SCopy=
      decltype(make_tiled_copy(q_g2s_copy_atom{},
                               make_layout(make_shape(Int<32 * warp_num / q_continuous_thread>{}, Int<q_continuous_thread>{}),
                                           make_stride(Int<q_continuous_thread>{}, Int<1>{})),
                               make_layout(make_shape(Int<1>{}, Int<Align>{}))));

  using T_M = cutlass::AlignedArray<Element, Align_Mask>;
  using mask_g2s_copy_op = UniversalCopy<T_M>;
  using mask_g2s_copy_traits = Copy_Traits<mask_g2s_copy_op>;
  using mask_g2s_copy_atom = Copy_Atom<mask_g2s_copy_traits, Element>;
  using mask_g2s_copy_tile = decltype(make_tiled_copy(
      mask_g2s_copy_atom{},
      Layout<Shape<Int<32 * warp_num / mask_continuous_thread>,
                   Int<mask_continuous_thread>>,
             Stride<Int<mask_continuous_thread>, _1>>{},
      Layout<Shape<_1, Int<Align_Mask>>>{}));

  // using v_G2SCopy=
  //     decltype(make_tiled_copy(q_g2s_copy_atom{},
  //                              make_layout(make_shape(Int<32 * warp_num / v_continuous_thread>{}, Int<v_continuous_thread>{}),
  //                                          make_stride(Int<v_continuous_thread>{}, Int<1>{})),
  //                              make_layout(make_shape(Int<1>{}, Int<Align>{}))));
  using v_G2SCopy=
      decltype(make_tiled_copy(q_g2s_copy_atom{},
                               make_layout(make_shape(Int<v_continuous_thread>{}, Int<32 * warp_num / v_continuous_thread>{}),
                                           make_stride(Int<1>{}, Int<v_continuous_thread>{})),
                               make_layout(make_shape(Int<Align>{}, Int<1>{}))));
  using SmemLayoutAtomQ = decltype(composition(
      // B M S
      Swizzle<3, 3, 3>{}, make_layout(Shape<Int<ThreadblockShape0::kM>, Int<kE>>{},
                               Stride<Int<kE>, _1>{})));
  using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<ThreadblockShape0::kM>, Int<kE>>{}));
  using SmemLayoutAtomK = decltype(composition(
    // B M S
    Swizzle<3, 3, 3>{}, make_layout(Shape<Int<ThreadblockShape0::kN>, Int<kE>>{},
                              Stride<Int<kE>, _1>{})));
  using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<Int<ThreadblockShape0::kN>, Int<kE>>{}));

  using SmemLayoutAtomMask = decltype(composition(
    // B M S
    Swizzle<3, 3, 3>{}, make_layout(Shape<Int<ThreadblockShape0::kM>, Int<ThreadblockShape0::kN>>{},
                              Stride<Int<ThreadblockShape0::kN>, _1>{})));

  using SmemLayoutMask = decltype(tile_to_shape(SmemLayoutAtomMask{}, Shape<Int<ThreadblockShape0::kM>, Int<ThreadblockShape0::kN>>{}));


  static const int V_swizzle_S = Split_kF == 128 ? 4 : 3;
  using SmemLayoutAtomV = decltype(composition(
    // B M S
    Swizzle<3, 3, V_swizzle_S>{}, make_layout(Shape<Int<Split_kF>, Int<ThreadblockShape1::kK>>{},
    // Swizzle<0, 0, 0>{}, make_layout(Shape<Int<Split_kF>, Int<ThreadblockShape1::kK>>{},
                              Stride<_1, Int<Split_kF>>{})));
  using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{},Shape<Int<Split_kF>, Int<ThreadblockShape1::kK>>{}));
  using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutV{}));

  static const int Epi_swizzle_S = Split_kF == 128 ? 4 : 3;

  using SmemLayoutEpilogue = decltype(composition(
    // B M S
    Swizzle<3, 3, Epi_swizzle_S>{}, make_layout(Shape<Int<ThreadblockShape0::kM>, Int<Split_kF>>{},
    // Swizzle<0, 0, 0>{}, make_layout(Shape<Int<ThreadblockShape0::kM>, Int<Split_kF>>{},
                              Stride<Int<Split_kF>, _1>{})));

  using T_E = cutlass::AlignedArray<Element, Align>;
  using e_s2g_copy_op = UniversalCopy<T_E>;
  using e_s2g_traits = Copy_Traits<e_s2g_copy_op>;
  using e_s2g_copy_atom = Copy_Atom<e_s2g_traits, Element>;
  using e_s2g_copy_tile = decltype(make_tiled_copy(
      e_s2g_copy_atom{},
      Layout<Shape<Int<32 * warp_num / epilogue_continuous_thread>,
                   Int<epilogue_continuous_thread>>,
             Stride<Int<epilogue_continuous_thread>, _1>>{},
      Layout<Shape<_1, Int<Align>>>{}));


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

  using q_s2r_copy_op = SM75_U32x4_LDSM_N;
  using q_s2r_copy_traits = Copy_Traits<q_s2r_copy_op>;
  using q_s2r_copy_atom = Copy_Atom<q_s2r_copy_traits, Element>;

  using k_s2r_copy_op = SM75_U32x4_LDSM_N;
  using k_s2r_copy_traits = Copy_Traits<k_s2r_copy_op>;
  using k_s2r_copy_atom = Copy_Atom<k_s2r_copy_traits, Element>;

  // using v_s2r_copy_op = SM75_U16x8_LDSM_T;
  using v_s2r_copy_op = SM75_U16x4_LDSM_T;

  using v_s2r_copy_traits = Copy_Traits<v_s2r_copy_op>;
  using v_s2r_copy_atom = Copy_Atom<v_s2r_copy_traits, Element>;


  using c_identity_copy_op = SM75_U32x2_LDSM_N;
  using c_identity_copy_traits = Copy_Traits<c_identity_copy_op>;
  using c_identity_copy_atom = Copy_Atom<c_identity_copy_traits, Element>;

  using mask_s2r_copy_op = SM75_U32x2_LDSM_N;
  using mask_s2r_copy_traits = Copy_Traits<mask_s2r_copy_op>;
  using mask_s2r_copy_atom = Copy_Atom<mask_s2r_copy_traits, Element>;

  using S2RCopyAtomQ = q_s2r_copy_atom;
  using S2RCopyAtomK = k_s2r_copy_atom;
  using S2RCopyAtomV = v_s2r_copy_atom;
  using S2RCopyAtomIdentityC = c_identity_copy_atom;
  using S2RCopyAtomMask = mask_s2r_copy_atom;

  using T_K = cutlass::Array<Element, Align>;
  using k_g2r_copy_op = UniversalCopy<T_K>;
  using k_g2r_traits = Copy_Traits<k_g2r_copy_op>;
  using k_g2r_copy_atom = Copy_Atom<k_g2r_traits, T_K>;
  using k_g2r_copy_tile = decltype(make_tiled_copy(
      k_g2r_copy_atom{},
      Layout<Shape< Int<32 * warp_num / q_continuous_thread>,Int<q_continuous_thread>>,
             Stride<Int<q_continuous_thread>, _1>>{},
      Layout<Shape< _1, Int<Align>>>{}));


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

  using mma_atom_shape = mma_traits::Shape_MNK;
  static constexpr int kMmaEURepeatM = ThreadblockShape0::kM / WarpShape0::kM;
  static constexpr int kMmaEURepeatN = 1;
  static constexpr int kMmaEURepeatK = 1;

  static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
  static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});//这里的2是通过增加寄存器的方法，使得Atom在N方向扩展两倍
  static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});
  // using MMA = decltype(make_tiled_mma(
  //     mma_atom{},
  //     make_layout(Shape<Int<ThreadblockShape0::kM / WarpShape0::kM>, _1, _1>{}), // 描述了warp的排布状态，此时是4*1的排布
  //     make_layout(
  //         Shape<
  //             _1, Int<ThreadblockShape0::kN / 8>,
  //             _1>{}))); // 描述了每个线程持有的mma寄存器的重复状态，并没有发生寄存器重用，只是扩展，与tile上的重复状态定义重合，所以在3.4后删除？
  using MMA = decltype(make_tiled_mma(
      mma_atom{},
      make_layout(Shape<Int<kMmaEURepeatM>, Int<kMmaEURepeatN>, Int<kMmaEURepeatK>>{}) // 描述了warp的排布状态，此时是4*1的排布
      ,Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>{}
      // ,make_layout(
      //     Shape<
      //         _1, Int<ThreadblockShape0::kN / 8>,
      //         _1>{})
      ));

  static constexpr int kMmaPM1 = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
  static constexpr int kMmaPN1 = 1 * kMmaEURepeatN * get<1>(mma_atom_shape{});
  static constexpr int kMmaPK1 = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});
  using MMA1 = decltype(make_tiled_mma(
      mma_atom{},
      make_layout(Shape<Int<kMmaEURepeatM>, Int<kMmaEURepeatN>, Int<kMmaEURepeatK>>{}) // 描述了warp的排布状态，此时是4*1的排布
      ,Tile<Int<kMmaPM1>, Int<kMmaPN1>, Int<kMmaPK1>>{}
      ));

};
template <typename KernelTraits>
__global__ void mha(MultiHeadAttentionProblemSize params, Element *q,
                    Element *k, Element *v, Element* mask, Element *out) {
  static const int kE = KernelTraits::kE;
  static const int Split_kF = KernelTraits::Split_kF;
  static const bool TensorMask = KernelTraits::Mask;
  using SharedStorage = typename KernelTraits::SharedStorage;
  extern __shared__ int s[];

  SharedStorage &smem = *(reinterpret_cast<SharedStorage *>(s));
  auto global_q =
      make_tensor(make_gmem_ptr(q), make_shape(params.N, params.S,params.H, Int<kE>{}),
                  make_stride(params.S * params.H * Int<kE>{}, params.H * Int<kE>{}, Int<kE>{}, _1{}));
  auto global_k =
      make_tensor(make_gmem_ptr(k), make_shape(params.N, params.T,params.H, Int<kE>{}),
                  make_stride(params.T * params.H * Int<kE>{}, params.H * Int<kE>{}, Int<kE>{}, _1{}));
  // auto global_v =
  //     make_tensor(make_gmem_ptr(v), make_shape(params.H, params.F, params.T),
  //                 make_stride(params.T * params.F, _1{}, params.F));
  auto global_v =
      make_tensor(make_gmem_ptr(v), make_shape(params.N, params.H, params.F, params.T),
                  make_stride(params.H * params.F * params.T, params.F, _1{}, params.H * params.F));

  auto global_out =
      make_tensor(make_gmem_ptr(out), make_shape(params.N, params.S, params.H, params.F),
                  make_stride(params.S * params.H * params.F, params.H * params.F, params.F, _1{}));

  auto block_q = coalesce(local_tile(
      global_q,
      make_shape(Int<1>{}, Int<KernelTraits::ThreadblockShape0::kM>{}, Int<1>{},
                 Int<kE>{}),
      make_coord(blockIdx.z / params.H, blockIdx.x, blockIdx.z % params.H, 0)));




   auto block_k =
      coalesce(local_tile(global_k, make_shape(Int<1>{}, params.T, Int<1>{},  Int<kE>{}),
                          make_coord(blockIdx.z / params.H, 0, blockIdx.z % params.H, 0)));
  auto block_v = coalesce(
      local_tile(global_v, make_shape(Int<1>{}, Int<1>{}, Int<Split_kF>{}, params.T),
                 make_coord(blockIdx.z / params.H, blockIdx.z % params.H, blockIdx.y, 0)));
  // auto block_v = coalesce(
  //     local_tile(global_v, make_shape(params.T, Int<1>{}, Int<Split_kF>{}),
  //                make_coord(0, blockIdx.z, blockIdx.y)));
  auto block_out = coalesce(local_tile(
      global_out,
      make_shape(Int<1>{},Int<KernelTraits::ThreadblockShape0::kM>{},Int<1>{},
                 Int<Split_kF>{}),
      make_coord(blockIdx.z / params.H, blockIdx.x, blockIdx.z % params.H, blockIdx.y)));



  auto tiled_block_k = local_tile(
      block_k,
      make_shape(Int<KernelTraits::ThreadblockShape0::kN>{}, Int<kE>{}),
      make_coord(_, 0));
  auto tiled_block_v = local_tile(
      block_v,
      make_shape(Int<Split_kF>{}, Int<KernelTraits::ThreadblockShape1::kK>{}),
      make_coord(0, _));
// if (thread0()) {
//   PRINT(block_v);
//   PRINT(tiled_block_v);
// }
#if 1
  auto smem_q = make_tensor(make_smem_ptr((Element *)&smem.q_smem),
                            typename KernelTraits::SmemLayoutQ{});
  auto smem_k = make_tensor(make_smem_ptr((Element *)&smem.k_smem),
                            typename KernelTraits::SmemLayoutK{});
  auto smem_v = make_tensor(make_smem_ptr((Element *)&smem.v_smem),
                            typename KernelTraits::SmemLayoutV{});
  auto smem_v_NoSwizzle = make_tensor(make_smem_ptr((Element *)&smem.v_smem),
                            typename KernelTraits::SmemLayoutVtransposedNoSwizzle{});


  typename KernelTraits::q_G2SCopy q_tiled_copy;
  // typename KernelTraits::q_g2r_copy_tile q_tiled_copy;
  auto q_g2s_thr_copy = q_tiled_copy.get_slice(threadIdx.x);
  auto tQgQ = q_g2s_thr_copy.partition_S(block_q);
  auto tQsQ = q_g2s_thr_copy.partition_D(smem_q);


  copy(q_tiled_copy, tQgQ, tQsQ);
  cp_async_fence();
  // cp_async_wait<0>();
  // __syncthreads();

  typename KernelTraits::k_G2SCopy k_tiled_copy;

  auto k_g2s_thr_copy = k_tiled_copy.get_slice(threadIdx.x);

  auto tKgK = k_g2s_thr_copy.partition_S(tiled_block_k(_, _, 0));
  auto tKsK = k_g2s_thr_copy.partition_D(smem_k);
  auto tKrK = make_fragment_like(tKgK(_, 0, 0));

  copy(k_tiled_copy, tKgK, tKsK);
  cp_async_fence();
  // cp_async_wait<0>();
  // __syncthreads();
  typename KernelTraits::v_G2SCopy v_tiled_copy;

  auto v_g2s_thr_copy = v_tiled_copy.get_slice(threadIdx.x);

  auto tVgV = v_g2s_thr_copy.partition_S(tiled_block_v(_, _, 0));
  auto tVsV = v_g2s_thr_copy.partition_D(smem_v);
  if(thread0()) {
    // PRINT(tiled_block_v);
    // PRINT(tVgV);
    // PRINT(tVsV);
    // PRINT(v_tiled_copy);
  }
#if 1

  // copy(v_tiled_copy, tVgV, tVsV);




  typename KernelTraits::MMA tiled_mma;
    if (thread0()) {
    // printf("\n");
    // PRINT(tiled_mma.get_layoutA_TV());
    // PRINT(tiled_mma);
    // PRINT(make_shape(tile_size<0>(tiled_mma),tile_size<2>(tiled_mma)));

    // PRINT((tiled_mma.get_layoutA_TV()));
    // PRINT(size<1>(tiled_mma.get_layoutA_TV()));

    // PRINT((typename KernelTraits::S2RCopyAtomQ::ValLayoutRef{}));
    // PRINT(size<1>(typename KernelTraits::S2RCopyAtomQ::ValLayoutRef{}));

  }

  auto s2r_tiled_copy_q = make_tiled_copy_A(typename KernelTraits::S2RCopyAtomQ{}, tiled_mma);
  auto s2r_thr_copy_q = s2r_tiled_copy_q.get_slice(threadIdx.x);
  auto thr_QsQ = s2r_thr_copy_q.partition_S(smem_q);  // ? (CPY, CPY_M, CPY_K, kStage)

    if (thread0()) {
    // printf("\n");
    // PRINT(tiled_mma.get_layoutB_TV());
    // PRINT(make_shape(tile_size<1>(tiled_mma),tile_size<2>(tiled_mma)));
    // PRINT(layout(thr_QsQ));

    // PRINT((tiled_mma.get_layoutB_TV()));
    // PRINT(size<1>(tiled_mma.get_layoutB_TV()));

    // PRINT((typename KernelTraits::S2RCopyAtomK::ValLayoutRef{}));
    // PRINT(size<1>(typename KernelTraits::S2RCopyAtomK::ValLayoutRef{}));

  }
  // __syncthreads();

#if 1
  auto s2r_tiled_copy_k = make_tiled_copy_B(typename KernelTraits::S2RCopyAtomK{}, tiled_mma);
  auto s2r_thr_copy_k = s2r_tiled_copy_k.get_slice(threadIdx.x);
  auto thr_KsK = s2r_thr_copy_k.partition_S(smem_k);  // ? (CPY, CPY_M, CPY_K, kStage)


  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  // auto thr_QsQ =
  //     thr_mma.partition_A(smem_q(_, _)); // (MMA, MMA_M, MMA_K, num_tile_k)
  // auto thr_KsK = thr_mma.partition_B(smem_k(_, _)); // (MMA, MMA_N, MMA_K, num_tile_k)
  // auto thr_CsC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N)

  auto thr_tQrQ =
      thr_mma.partition_fragment_A(smem_q(_, _)); // (MMA, MMA_M, MMA_K)
  // auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N,
  // MMA_K)
  auto thr_tKrK =
      ( thr_mma.partition_fragment_B(smem_k(_, _))); // (MMA, MMA_N, MMA_K)
  auto thr_tCrC = partition_fragment_C(
      tiled_mma,
      Shape<Int<KernelTraits::ThreadblockShape0::kM>,
            Int<KernelTraits::ThreadblockShape0::kN>>{}); // (MMA, MMA_M, MMA_N)

  auto thr_tKrK_view = s2r_thr_copy_k.retile_D(thr_tKrK);
  if (thread0()) {
    // PRINT(layout((tiled_block_v)));
    // PRINT(layout((tiled_block_k)));
    // PRINT(layout(block_out));

    // PRINT(layout(smem_v));
    // PRINT((thr_mma));
    // PRINT(layout(smem_k));
    // PRINT(layout(tKgK));
    // PRINT(layout(tKsK));
    // PRINT(layout(thr_QsQ));
    // PRINT(layout(thr_KsK));
    // PRINT(layout(thr_tQrQ));
    // PRINT(layout(thr_tKrK));
    // PRINT(layout(thr_tKrK_view));
    // PRINT(layout(thr_tCrC));
    //如果这里print,后面必须同步
  }

  // if constexpr(TensorMask) {
  //   auto smem_m = make_tensor(make_smem_ptr((Element*)&smem.m_smem), typename KernelTraits::SmemLayoutMask{});
  //   auto tensor_mask = make_tensor(make_gmem_ptr(mask), make_shape(params.S, params.T), make_stride(params.T, Int<1>{}));
  //   auto tiled_tensor_mask = local_tile(tensor_mask, make_shape(params.S, Int<KernelTraits::ThreadblockShape0::kN>{}), make_coord(0, _));

  //   typename KernelTraits::mask_g2s_copy_tile mask_g2s_copy_tile;
  //   auto mask_g2s_copy_thr = mask_g2s_copy_tile.get_slice(threadIdx.x);
  //   // auto g2s_tMgM = mask_g2s_copy_thr.partition_S(tiled_tensor_mask(_,_, 0));
  //   auto g2s_tMsM = mask_g2s_copy_thr.partition_D(smem_m);
  //   decltype(thr_tCrC) s2r_tMrM;
  // }
  // __syncthreads();
// #if 0
  cp_async_wait<1>();
  __syncthreads();
  cute::copy(s2r_tiled_copy_q, thr_QsQ, thr_tQrQ);
  if (thread0()) {
    // PRINT(thr_QsQ);
    // PRINT(thr_tQrQ);


  }


  using ElementAccumulator = typename KernelTraits::ElementAccumulator;

  typename KernelTraits::MMA1 tiled_mma1;

  auto thr_mma1 = tiled_mma1.get_slice(threadIdx.x);
  // auto mma1_tVsV = thr_mma1.partition_B(smem_v);
  auto mma1_tOgO = thr_mma1.partition_C(block_out);

  auto mma1_tVrV = thr_mma1.partition_fragment_B(smem_v_NoSwizzle);
  auto mma1_tOrO = thr_mma1.partition_fragment_C(block_out(_, _));

  auto s2r_tiled_copy_v = make_tiled_copy_B(typename KernelTraits::S2RCopyAtomV{}, tiled_mma1);
  auto s2r_thr_copy_v = s2r_tiled_copy_v.get_slice(threadIdx.x);
  auto mma1_tVsV = s2r_thr_copy_v.partition_S(smem_v);  // ? (CPY, CPY_M, CPY_K, kStage)


  Tensor tOrrC = make_tensor(
      thr_tCrC.data(),
      convert_layout_acc_Aregs<KernelTraits::mma_traits>(thr_tCrC.layout()));


  ElementAccumulator m_max[2];
  ElementAccumulator d[2];

  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    m_max[i] = -INFINITY;
    d[i] = (ElementAccumulator)0.0f;
  }


  int ntile = (params.T + KernelTraits::ThreadblockShape0::kN - 1) /
          KernelTraits::ThreadblockShape0::kN;


  ///< main loop
  for (int itile = 0; itile < ntile; ++itile) {
    cp_async_wait<0>();
    __syncthreads();

    tVgV = v_g2s_thr_copy.partition_S(tiled_block_v(_, _, itile));


    copy(v_tiled_copy, tVgV, tVsV);
    cp_async_fence();




  //   cp_async_wait<0>();
  // __syncthreads();
    // cute::copy(thr_KsK, thr_tKrK);
    // smem -> reg
    cute::copy(s2r_tiled_copy_k, thr_KsK, thr_tKrK_view);

    clear(thr_tCrC);
    //Q的顺序为内部->col->row, 为何不是内部->row->col?
    cute::gemm(tiled_mma, thr_tCrC, thr_tQrQ, thr_tKrK, thr_tCrC);

    if constexpr(TensorMask) {
      auto smem_m = make_tensor(make_smem_ptr((Element*)&smem.m_smem), typename KernelTraits::SmemLayoutMask{});
      auto tensor_mask = make_tensor(make_gmem_ptr(mask), make_shape(params.S, params.T), make_stride(params.T, Int<1>{}));
      auto tiled_tensor_mask = (local_tile(tensor_mask, make_shape(Int<KernelTraits::ThreadblockShape0::kM>{}, Int<KernelTraits::ThreadblockShape0::kN>{}), make_coord(blockIdx.x, _)));
      // auto tiled_tensor_mask = local_tile(block_mask, make_shape(Int<KernelTraits::ThreadblockShape0::kM>{},), make_coord(0, _));
      if (thread0()) {
        // PRINT(tensor_mask);
        // PRINT(block_mask);
        // PRINT(tiled_tensor_mask);
      }
      typename KernelTraits::mask_g2s_copy_tile mask_g2s_copy_tile;
      auto mask_g2s_copy_thr = mask_g2s_copy_tile.get_slice(threadIdx.x);
      // auto g2s_tMgM = mask_g2s_copy_thr.partition_S(tiled_tensor_mask(_,_, 0));
      auto g2s_tMgM = mask_g2s_copy_thr.partition_S(tiled_tensor_mask(_,_, itile));
      auto g2s_tMsM = mask_g2s_copy_thr.partition_D(smem_m);

      auto tMpM = make_tensor<bool>(make_shape(size<1>(g2s_tMgM),size<2>(g2s_tMgM)));
      auto mask_identity_tensor = make_identity_tensor(shape(smem_m));
      auto tMiM = mask_g2s_copy_thr.partition_D(mask_identity_tensor);
            if (thread0()) {
        // PRINT(g2s_tMgM);
        // PRINT(g2s_tMsM);
        // print_tensor(tMiM);
      }
      #pragma unroll
      for (int i = 0; i < size<1>(g2s_tMgM); ++i) {
        for (int j = 0; j < size<2>(g2s_tMgM); ++j) {
        tMpM(i,j) = get<0>(tMiM(make_coord(0, 0), i, j)) < params.S - blockIdx.x * KernelTraits::ThreadblockShape0::kM &&
                    get<1>(tMiM(make_coord(0, 0), i, j)) < params.T - itile * KernelTraits::ThreadblockShape0::kN;
      }
      }

      // cute::copy(mask_g2s_copy_tile, g2s_tMgM, g2s_tMsM);
      cute::copy_if(mask_g2s_copy_tile, tMpM, g2s_tMgM, g2s_tMsM);
    }

  //for tiled_mma gemm and g2r mask copy
  __syncthreads();

    if constexpr(TensorMask) {
      auto smem_m = make_tensor(make_smem_ptr((Element*)&smem.m_smem), typename KernelTraits::SmemLayoutMask{});
      auto s2r_mask_tile_copy = make_tiled_copy_C(typename KernelTraits::S2RCopyAtomMask{}, tiled_mma1);
      auto s2r_mask_thr_copy = s2r_mask_tile_copy.get_slice(threadIdx.x);
      auto s2r_tMsM = s2r_mask_thr_copy.partition_S(smem_m);
      decltype(thr_tCrC) s2r_tMrM;

      // auto s2r_tMsM = partition_fragment_C(tiled_mma, smem_m);
      // cute::copy(s2r_tMsM, s2r_tMrM);
      clear(s2r_tMrM);
      cute::copy(s2r_mask_tile_copy, s2r_tMsM, s2r_tMrM);
      if (thread0()) {
        // PRINT(s2r_tMsM);
        // PRINT(s2r_tMrM);
      }
      #pragma unroll
      for (int i = 0; i < size<0>(thr_tCrC); ++i) {
      #pragma unroll
        for (int j = 0; j < size<1>(thr_tCrC); ++j) {
              #pragma unroll
           for (int m = 0; m < size<2>(thr_tCrC); ++m) {
            thr_tCrC(make_coord(i%2, i/2), j, m) += s2r_tMrM(make_coord(i%2, i/2), j, m);
        }
      }
    }
  }


  // if (itile + 1 < ntile) {
    int next_itile = (itile + 1) % ntile;
    tKgK = k_g2s_thr_copy.partition_S(tiled_block_k(_, _, next_itile));

    copy(k_tiled_copy, tKgK, tKsK);
  // }
  cp_async_fence();

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
    ElementAccumulator new_d[2];
    ElementAccumulator new_exp_tcrc_sum[2];
    for (int i = 0; i < 2; ++i) {
      new_tCrC_max[i] = -INFINITY;
      exp_oldm_sub_newm[i] = -INFINITY;
      new_m_max[i] = -INFINITY;
      new_d[i] = 0.0f;
      new_exp_tcrc_sum[i] = 0.0f;
    }

  // deal with residual
  if (itile == ntile - 1) {

    // ElementAccumulator new_o_scale[2]{0.f};
    auto C_identity = make_identity_tensor(Shape<Int<KernelTraits::ThreadblockShape0::kM>,
            Int<KernelTraits::ThreadblockShape0::kN>>{});
    // auto s2r_c_identity_copy = make_tiled_copy_C(typename KernelTraits::S2RCopyAtomIdentityC{}, tiled_mma);
    // auto s2r_c_identity_thr_copy = s2r_c_identity_copy.get_slice(threadIdx.x);
    // auto tCiC = s2r_c_identity_thr_copy.partition_S(C_identity);
    auto tCiC = thr_mma.partition_C(C_identity);
    auto tCpC = make_tensor<bool>(make_shape(_2{}, size<1>(tCiC), size<2>(tCiC)));
    if (thread0()) {
      // print_tensor(tCiC);
      // PRINT(get<1>(tCiC(4 * 1 + 1)));
      // PRINT((tCiC(2)));
      // PRINT(get<1>(tCiC((make_coord(1,0), 0, 7))));
      // PRINT(get<0>(tCiC((make_coord(0,1), 0, 7))));
      // PRINT(get<1>(tCiC((make_coord(0,1), 0, 7))));
    }
#pragma unroll
    for (int i = 0; i < size<2>(tCpC); ++i) {
      {
#pragma unroll
        for (int j = 0; j < size<0>(tCpC); ++j) {
#pragma unroll
          for (int m = 0; m < size<1>(tCpC); ++m) {
            // tCpC(j, m, i) = threadIdx.x % 32 % 4 * 2 + i * 8 + j >= params.T - itile * KernelTraits::ThreadblockShape0::kN;
            //why tCiC don't use (make_coord(0,0), 0, i)) to get value ?
            tCpC(j, m, i) = get<1>(tCiC(4 * i + j)) >= params.T - itile * KernelTraits::ThreadblockShape0::kN;
          if (threadIdx.x < 32 && tCpC(j, m, i)) {
        // print("threadidx.x:%d,i:%d, threadIdx.x % 32 % 4 * 2 + i * 8 + j: %d, params.T - k * KernelTraits::ThreadblockShape0::kN: %d\n", threadIdx.x,i, threadIdx.x % 32 % 4 * 2 + i * 8 + j, params.T - k * KernelTraits::ThreadblockShape0::kN);
        // printf("k:%d,get<1>(tCiC((0, 0, %d)): %d\n", k, i,(int)get<1>(tCiC((make_coord(0,0), 0, i))));
        // printf("k:%d,get<0>(tCiC((0, 0, %d)): %d\n", k, i,(int)get<1>(tCiC((make_coord(0,1), 0, i))));
        // printf("k:%d,get<0>(tCiC((0, 0, %d)): %d\n", k, i,(int)get<1>(tCiC((make_coord(1,0), 0, i))));
        // printf("k:%d,get<0>(tCiC((0, 0, %d)): %d\n", k, i,(int)get<1>(tCiC((make_coord(1,1), 0, i))));
        // printf("k:%d,get<0>(tCiC((0, 0, %d)): %d\n", k, i,get<0>(tCiC((make_coord(0,0), 0, i))));
        // PRINT(tCiC((0, 0, i)));

      }
          }
        }
      }
    }
if (thread0()) {
  // print(size<0>(shape((thr_tCrC(make_coord(_,_), 0, 0)))));
  // print_tensor(get<0, 1>(thr_tCrC));
  // print(get<1>(thr_tCrC));
  // print(get<2>(thr_tCrC));
}
#pragma unroll
      for (int i = 0; i < size<0>(thr_tCrC); ++i) {
      #pragma unroll
        for (int j = 0; j < size<1>(thr_tCrC); ++j) {
              #pragma unroll
           for (int m = 0; m < size<2>(thr_tCrC); ++m) {
          if (tCpC(i % 2, j, m)) {
            thr_tCrC(make_coord(i%2, i/2), j, m) = -platform::numeric_limits<Element>::infinity();//half -inf
            // printf("threadIdx:%d,i % 2:%d, m:%d, thr_tCrC(make_coord(%d, %d), j, m): %f\n", threadIdx.x,i%2, m,i%2, i/2, float(thr_tCrC(make_coord(i%2, i/2), j, m)));
          }
        }
      }
    }
  }

    if (thread0()) {
      // PRINT(tCiC);
      // PRINT(tCpC);
      // PRINT(thr_tCrC);
    }
  //   if (itile + 1 < ntile) {
  //   tKgK = k_g2s_thr_copy.partition_S(tiled_block_k(_, _, itile + 1));

  //   copy(k_tiled_copy, tKgK, tKsK);
  //   //   cp_async_wait<0>();
  //   // __syncthreads();

  //   tVgV = v_g2s_thr_copy.partition_S(tiled_block_v(_, _, itile + 1));

  //   copy(v_tiled_copy, tVgV, tVsV);
  //   cp_async_fence();
  //       cp_async_wait<0>();
  // __syncthreads();
  //   }

#pragma unroll
    for (int i = 0; i < size<2>(thr_tCrC); ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          // const int offset = m + 2 * j + i * 4;
          // new_tCrC_max[j] =
          //     max(new_tCrC_max[j], (ElementAccumulator)*(thr_tCrC.data() + offset));
          new_tCrC_max[j] =
              max(new_tCrC_max[j], (ElementAccumulator)(thr_tCrC(make_coord(m, j), 0, i)));
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

    new_m_max[0] = max(m_max[0], new_tCrC_max[0]);
    new_m_max[1] = max(m_max[1], new_tCrC_max[1]);

    exp_oldm_sub_newm[0] = exp(m_max[0] - new_m_max[0]);
    exp_oldm_sub_newm[1] = exp(m_max[1] - new_m_max[1]);
    //when k == 0, don't want O to scale, so make exp_oldm_sub_newm == 1;
    if (itile==0) {
      exp_oldm_sub_newm[0] = (ElementAccumulator)1;
      exp_oldm_sub_newm[1] = (ElementAccumulator)1;
    }


#pragma unroll
    for (int i = 0; i < size<2>(thr_tCrC); ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          // new_exp_tcrc((m, j), 0, i) =
          //     exp((ElementAccumulator)thr_tCrC((m, j), 0, i) - new_tCrC_max[j]);
          // new_exp_tcrc_sum[j] += new_exp_tcrc((m, j), 0, i);
        thr_tCrC(make_coord(m, j), 0, i) = static_cast<Element>(cutlass::fast_exp((
            thr_tCrC(make_coord(m, j), 0, i) - (Element)new_m_max[j])));

        // const int offset = m + j * 2 + i * 4;
        //         *(thr_tCrC.data() + offset) = static_cast<Element>(cutlass::fast_exp((
        //     (*(thr_tCrC.data() + offset)) - (Element)new_m_max[j])));

        new_exp_tcrc_sum[j] += (ElementAccumulator)(thr_tCrC(make_coord(m, j), 0, i));

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


    if (thread0()) {
      // printf("new_o_scale[0]:%f\n", new_o_scale[0]);
      // printf("new_o_scale[1]:%f\n", new_o_scale[1]);
      // printf("new_d[0]:%f\n", new_d[0]);
      // printf("new_d[1]:%f\n", new_d[1]);

    }

#pragma unroll
    for (int i = 0; i < size<2>(mma1_tOrO); ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          // mma1_tOrO((m, j), 0, i) *= (Element)new_o_scale[j];

          // const int offset = m + 2 * j + i * 4;
          // *(mma1_tOrO.data() + offset) =  *(mma1_tOrO.data() + offset) * static_cast<Element>(exp_oldm_sub_newm[j]);
          mma1_tOrO(make_coord(m, j), 0, i) = mma1_tOrO(make_coord(m, j), 0, i) * static_cast<Element>(exp_oldm_sub_newm[j]);
        }
      }
    }
    cp_async_wait<1>();
  __syncthreads();

    cute::copy(s2r_tiled_copy_v, mma1_tVsV, mma1_tVrV);
    if (thread0()) {
      for (int i = 0; i < 4; ++i) {
        // printf("phase0 mma1_tVsV[%d]:%f\n", i, (float)mma1_tVsV[i]);
        // printf("phase0 mma1_tVrV[%d]:%f\n", i, (float)mma1_tVrV[i]);
      }

    }
    if (thread0()) {
      // PRINT(tOrrC);
      // for (int i = 0; i < size(tOrrC); ++i) {
      //   printf("tOrrC(%d):%f\n", i, float(tOrrC(i)));
      // }
      // PRINT(thr_tCrC);
      // for (int i = 0; i < size(thr_tCrC); ++i) {
      //   printf("thr_tCrC(%d):%f\n", i, float(thr_tCrC(i)));
      // }
    }
    cute::gemm(tiled_mma1, mma1_tOrO, tOrrC, mma1_tVrV, mma1_tOrO);
  // __syncthreads();

  // __syncthreads();

    d[0] = new_d[0];
    d[1] = new_d[1];
    m_max[0] = new_m_max[0];
    m_max[1] = new_m_max[1];
  }

#pragma unroll
    for (int i = 0; i < size<2>(mma1_tOrO); ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          // mma1_tOrO((m, j), 0, i) *= (Element)new_o_scale[j];
          // const int offset = m + 2 * j + i * 4;
          // *(mma1_tOrO.data() + offset) =  *(mma1_tOrO.data() + offset) / static_cast<Element>(d[j]);
          mma1_tOrO(make_coord(m, j), 0, i) = mma1_tOrO(make_coord(m, j), 0, i) / static_cast<Element>(d[j]);

        }
      }
    }

  using EpilogueSharedStorage = typename KernelTraits::EpilogueSharedStorage;
  EpilogueSharedStorage &epi_smem_storage = *(reinterpret_cast<EpilogueSharedStorage *>(s));

  Tensor e_smem = make_tensor(make_smem_ptr((Element*)&epi_smem_storage.e_smem), typename KernelTraits::SmemLayoutEpilogue{});

  auto mma1_tOsO = thr_mma1.partition_C(e_smem);

  cute::copy(mma1_tOrO, mma1_tOsO);
  __syncthreads();
  typename KernelTraits::e_s2g_copy_tile epi_smem_copy_tile;

  auto thr_epi_smem_copy = epi_smem_copy_tile.get_slice(threadIdx.x);
  auto tEsE = thr_epi_smem_copy.partition_S(e_smem);
  auto tEgE = thr_epi_smem_copy.partition_D(block_out);
  auto block_out_identity = make_identity_tensor(shape(block_out));
  auto tEiE = thr_epi_smem_copy.partition_S(block_out_identity);
  auto tEpE_S = make_tensor<bool>(make_shape(size<1>(tEgE)));
  auto tEpE_F = make_tensor<bool>(make_shape(size<2>(tEgE)));
#pragma unroll
  for (int i = 0; i < size<0>(tEpE_S); ++i) {
    tEpE_S(i) = get<0>(tEiE(make_coord(0,0), i, 0)) + blockIdx.x * KernelTraits::ThreadblockShape1::kM < params.S;
  }

#pragma unroll
  for (int i = 0 ; i < size<0>(tEpE_F); ++i) {
    tEpE_F(i) = get<1>(tEiE(make_coord(0,0), 0, i)) + blockIdx.y * KernelTraits::ThreadblockShape1::kN < params.F;

  }


  if (thread0()) {
    // PRINT(block_out);
    // PRINT(e_smem);
    // PRINT(tEiE);
    // PRINT(tEgE);
    // PRINT(tEpE_S);
    // PRINT(tEpE_F);
  }
#pragma unroll
for (int s = 0; s < size<1>(tEsE); ++s) {
  #pragma unroll
  for (int f = 0; f < size<2>(tEsE); ++f) {
    if (tEpE_S(s) && tEpE_F(f)) {
      // cute::copy(epi_smem_copy_tile, tEsE(_, s, f), tEgE(_, s, f));
      cute::copy(tEsE(_, s, f), tEgE(_, s, f));
    }
  }
}
  // cute::copy(epi_smem_copy_tile, tEsE, tEgE);

  // cute::copy(mma1_tOrO, mma1_tOgO);
  if (thread0()) {
    // PRINT(tEsE);
    // PRINT(tEgE);
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
          // printf("mma1_tO1rO((%d, %d), 0, %d): %f\n", m, j, i,
          //        (float)(*(mma_tOrO_data + m + j * 2 + i * 4)));
        }
      }
    }
  }
  #endif
  #endif
  #endif
}

/**
 *LAYOUT:
 *Q: N L H  E
 *K: N S H  E
 *V: N H Ev S(S 连续)
 *OUT: N L H Ev
*/

int main() {
  // int N = 1, S = 12, T = 120, F = 128, H = 1;
  // int N = 1, S = 33, T = 111, F = 128, H = 1;
  // int N = 1, S = 1024, T = 1024, F = 128, H = 8;
  int N = 3, S = 1111, T = 1200, F = 104, H = 3;
  // int N = 5, S = 100, T = 1, F = 32, H = 4;
  // int N = 1, S = 16, T = 32, F = 32, H = 1;
  MultiHeadAttentionProblemSize problem(N, S, T, kE, F, H);
  using KernelT = KernelTraits<cutlass::half_t, float, 8, kE, 128, UseMask, 8>;

  MemHelper<KernelT::Element> helper;
  auto Q = helper.GetCpuGpuBuffer(problem.QuerySize());
  auto K = helper.GetCpuGpuBuffer(problem.KeySize());
  auto V = helper.GetCpuGpuBuffer(problem.ValueSize());
  // auto Q = helper.GetCpuGpuBuffer(problem.QuerySize(), InitialType::AllOne);
  // auto K = helper.GetCpuGpuBuffer(problem.KeySize(), InitialType::AllOne);
  // auto V = helper.GetCpuGpuBuffer(problem.ValueSize(), InitialType::AllOne);
  auto Mask = helper.GetCpuGpuBuffer(problem.PSize());
  auto Out = helper.GetCpuGpuBuffer(problem.OutputSize());

  dim3 block = dim3{32 * KernelT::warp_num};
  dim3 grid = dim3(
      (S + KernelT::ThreadblockShape0::kM - 1) / KernelT::ThreadblockShape0::kM,
      (F + KernelT::ThreadblockShape1::kN - 1) / KernelT::ThreadblockShape1::kN,
      H * N);
  int smem_s = sizeof(KernelT::SharedStorage);
  printf("size of smem in KB:%d\n", smem_s / 1024);

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
      (Element *)(V.second),(Element*)(Mask.second), (Element *)(Out.second));

  helper.SyncGpuToCpu(Out);

  cudaError_t err = cudaGetLastError();

  if (err != cudaError_t::cudaSuccess) {
    std::cout << "final error: " << cudaGetErrorString(err) << std::endl;
    exit(-1);
  }
  Element *C_gt = (Element *)helper.GetCpuBuffer(N * H *S * T);
  Element *m_gt = (Element *)helper.GetCpuBuffer(N * H *S);
  Element *res_gt = (Element *)helper.GetCpuBuffer(N * H * S * F);



#ifdef CPU_CHECK
  Element* cpu_a = (Element*)Q.first;
  Element* cpu_b = (Element*)K.first;
  Element* cpu_d = (Element*)V.first;
  Element* mask = (Element*)Mask.first;

  int m = S;
  int n = T;
  int k = kE;
for (int in = 0; in< N; ++in) {
  Element* each_N_cpu_a = cpu_a + in * kE * S * H;
  Element* each_N_cpu_b = cpu_b + in * kE * T * H;
  Element* each_N_cpu_C_gt = C_gt + in * S * T * H;
  Element* each_N_cpu_m_gt = m_gt + in * S * H;
  Element* each_N_cpu_cpu_d = cpu_d + in * F * H * T;
  Element* each_N_cpu_res_gt = res_gt + in * F * H * S;
for (int h = 0; h < H; ++h) {
  Element* each_head_cpu_a = each_N_cpu_a + h * kE;
  Element* each_head_cpu_b = each_N_cpu_b + h * kE;
  Element* each_head_cpu_C_gt = each_N_cpu_C_gt + h * S * T;
  Element* each_head_cpu_m_gt = each_N_cpu_m_gt + h * S;
  Element* each_head_cpu_cpu_d = each_N_cpu_cpu_d + h * F;
  Element* each_head_cpu_res_gt = each_N_cpu_res_gt + h * F;
  for (int i = 0 ; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      ElementAccumulator sum = (ElementAccumulator)0.f;
      for (int l = 0; l < k; ++l) {
        sum += each_head_cpu_a[i * H * k + l] * each_head_cpu_b[l + j * H * k];
      }
      // printf("sum:%f\n", (float)sum);
      each_head_cpu_C_gt[i * n + j] = (Element)sum;
    }
  }

  for (int i = 0; i < m; ++i) {
    ElementAccumulator m_max = -INFINITY;
    for (int j = 0; j < n; ++j) {
      m_max= max(m_max, (ElementAccumulator)each_head_cpu_C_gt[i * n + j]);
    }
    each_head_cpu_m_gt[i] = m_max;
  }
  if constexpr (UseMask) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        each_head_cpu_C_gt[i * n + j] +=  mask[i * n + j];
      }
    }
  }
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      each_head_cpu_C_gt[i * n + j] = (Element)exp(each_head_cpu_C_gt[i * n + j] - each_head_cpu_m_gt[i]);
    }
  }

  for (int i = 0; i < m; ++i) {
    ElementAccumulator sum = (ElementAccumulator)0.f;
    for (int j = 0; j < n; ++j) {
      sum += each_head_cpu_C_gt[i * n + j];
    }
    for (int j = 0; j < n; ++j) {
      each_head_cpu_C_gt[i * n + j] = each_head_cpu_C_gt[i * n + j] / (Element)sum;
    }
  }

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < F; ++j) {
      ElementAccumulator sum = (ElementAccumulator)0;
      for (int p = 0; p < n; ++p) {
        sum += each_head_cpu_C_gt[i * n + p] * each_head_cpu_cpu_d[p * H *  F + j];
      }
      each_head_cpu_res_gt[i * H * F + j] = (Element)sum;
    }
  }
}
}
  helper.Regression(res_gt, (Element*)Out.first, false, false);
#endif

}