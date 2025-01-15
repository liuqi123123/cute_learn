
#include "utils.h"

// #define CPU_CHECK
using Element = cutlass::half_t;
using ElementAccumulator = float;

static const int kE = 32;
static const bool UseMask = true;



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

  using v_G2SCopy=
      decltype(make_tiled_copy(q_g2s_copy_atom{},
                               make_layout(make_shape(Int<v_continuous_thread>{}, Int<32 * warp_num / v_continuous_thread>{}),
                                           make_stride(Int<1>{}, Int<v_continuous_thread>{})),
                               make_layout(make_shape(Int<Align>{}, Int<1>{}))));
  static const int Q_swizzle_S = kE == 128 ? 4 : 3;
  using SmemLayoutAtomQ = decltype(composition(
      // B M S
      Swizzle<3, 3, Q_swizzle_S>{}, make_layout(Shape<Int<ThreadblockShape0::kM>, Int<kE>>{},
      // Swizzle<0, 0, 0>{}, make_layout(Shape<Int<ThreadblockShape0::kM>, Int<kE>>{},
                               Stride<Int<kE>, _1>{})));
  using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, Shape<Int<ThreadblockShape0::kM>, Int<kE>>{}));
  static const int K_swizzle_S = kE == 128 ? 4 : 3;

  using SmemLayoutAtomK = decltype(composition(
    // B M S
    Swizzle<3, 3, K_swizzle_S>{}, make_layout(Shape<Int<ThreadblockShape0::kN>, Int<kE>>{},
                              Stride<Int<kE>, _1>{})));
  using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<Int<ThreadblockShape0::kN>, Int<kE>>{}));
  static const int Mask_swizzle_S = ThreadblockShape0::kN == 128 ? 4 : 3;

  using SmemLayoutAtomMask = decltype(composition(
    // B M S
    Swizzle<3, 3, Mask_swizzle_S>{}, make_layout(Shape<Int<ThreadblockShape0::kM>, Int<ThreadblockShape0::kN>>{},
                              Stride<Int<ThreadblockShape0::kN>, _1>{})));

  using SmemLayoutMask = decltype(tile_to_shape(SmemLayoutAtomMask{}, Shape<Int<ThreadblockShape0::kM>, Int<ThreadblockShape0::kN>>{}));


  static const int V_swizzle_S = Split_kF == 128 ? 4 : 3;
  using SmemLayoutAtomV = decltype(composition(
    // B M S
    Swizzle<3, 3, V_swizzle_S>{}, make_layout(Shape<Int<Split_kF>, Int<ThreadblockShape1::kK>>{},
                              Stride<_1, Int<Split_kF>>{})));
  using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{},Shape<Int<Split_kF>, Int<ThreadblockShape1::kK>>{}));
  // using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutV{}));
  using SmemLayoutVtransposedNoSwizzle = decltype(make_layout(Shape<Int<Split_kF>, Int<ThreadblockShape1::kK>>{},
                              Stride<_1, Int<Split_kF>>{}));

  static const int Epi_swizzle_S = Split_kF == 128 ? 4 : 3;

  using SmemLayoutEpilogue = decltype(composition(
    // B M S
    Swizzle<3, 3, Epi_swizzle_S>{}, make_layout(Shape<Int<ThreadblockShape0::kM>, Int<Split_kF>>{},
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


  using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  using mma_atom_shape = mma_traits::Shape_MNK;
  static constexpr int kMmaEURepeatM = ThreadblockShape0::kM / WarpShape0::kM;
  static constexpr int kMmaEURepeatN = 1;
  static constexpr int kMmaEURepeatK = 1;

  static constexpr int kMmaPM = 1;
  static constexpr int kMmaPN = 2;
  static constexpr int kMmaPK = 1;
  using MMA = decltype(make_tiled_mma(
      mma_atom{},
      make_layout(Shape<Int<kMmaEURepeatM>, Int<kMmaEURepeatN>, Int<kMmaEURepeatK>>{})
      ,Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>{}
      ));

  static constexpr int kMmaPM1 = 1;
  static constexpr int kMmaPN1 = 1;
  static constexpr int kMmaPK1 = 1;
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


  auto smem_q = make_tensor(make_smem_ptr((Element *)&smem.q_smem),
                            typename KernelTraits::SmemLayoutQ{});
  auto smem_k = make_tensor(make_smem_ptr((Element *)&smem.k_smem),
                            typename KernelTraits::SmemLayoutK{});
  auto smem_v = make_tensor(make_smem_ptr((Element *)&smem.v_smem),
                            typename KernelTraits::SmemLayoutV{});
  auto smem_v_NoSwizzle = make_tensor(make_smem_ptr((Element *)&smem.v_smem),
                            typename KernelTraits::SmemLayoutVtransposedNoSwizzle{});


  typename KernelTraits::q_G2SCopy q_tiled_copy;

  auto q_g2s_thr_copy = q_tiled_copy.get_slice(threadIdx.x);
  auto tQgQ = q_g2s_thr_copy.partition_S(block_q);
  auto tQsQ = q_g2s_thr_copy.partition_D(smem_q);
  clear(tQsQ);

  ///< only last blockIdx.x need mask
  if ((blockIdx.x + 1) * KernelTraits::ThreadblockShape0::kM > params.S) {

    auto Q_identity_tensor = make_identity_tensor(shape(smem_q));
    auto tQiQ = q_g2s_thr_copy.partition_S(Q_identity_tensor);
    auto tQpQ = make_tensor<bool>(make_shape(size<1>(tQiQ)));
#pragma unroll
    for (int i = 0; i < size<1>(tQiQ); ++i) {
      tQpQ(i) = (int)get<0>(tQiQ(make_coord(0, 0), i, 0)) <
                params.S - blockIdx.x * KernelTraits::ThreadblockShape0::kM;
    }

    Operator_2D_Regs<size<1>(tQiQ), size<2>(tQiQ)>([&](int i, int j) {
      if (tQpQ(i)) {
        copy(tQgQ(_, i, j), tQsQ(_, i, j));
      }
    });
  } else {
    copy(q_tiled_copy, tQgQ, tQsQ);
  }

  // #pragma unroll
  //   for (int i = 0 ; i < size<1>(tQiQ); ++i) {
  //   #pragma unroll
  //     for (int j = 0; j < size<2>(tQiQ); ++j) {
  //       if (tQpQ(i)) {
  //         copy(tQgQ, tQsQ);
  //       }
  //     }
  //   }



  cp_async_fence();


  typename KernelTraits::k_G2SCopy k_tiled_copy;

  auto k_g2s_thr_copy = k_tiled_copy.get_slice(threadIdx.x);

  auto tKgK = k_g2s_thr_copy.partition_S(tiled_block_k(_, _, 0));
  auto tKsK = k_g2s_thr_copy.partition_D(smem_k);
  auto tKrK = make_fragment_like(tKgK(_, 0, 0));
  clear(tKsK);

  auto K_indentity_tensor = make_identity_tensor(shape(smem_k));
  auto tKiK = k_g2s_thr_copy.partition_S(K_indentity_tensor);
  auto tKpK = make_tensor<bool>(make_shape(size<1>(tKiK)));

  #pragma unroll
  for (int i = 0; i < size<1>(tKiK); ++i) {
    tKpK(i) =  (int)get<0>(tKiK(make_coord(0, 0), i, 0)) < params.T - KernelTraits::ThreadblockShape0::kN * 0;
  }
  // if (thread0()) {
  //   PRINT(tKpK);
  // }

  Operator_2D_Regs<size<1>(tKiK), size<2>(tKiK)>([&](int i, int j) {
    if (tKpK(i)) {
      copy(tKgK(_, i, j), tKsK(_, i, j));
    }
  });

  // copy(k_tiled_copy, tKgK, tKsK);
  cp_async_fence();

  typename KernelTraits::v_G2SCopy v_tiled_copy;

  auto v_g2s_thr_copy = v_tiled_copy.get_slice(threadIdx.x);

  auto tVgV = v_g2s_thr_copy.partition_S(tiled_block_v(_, _, 0));
  auto tVsV = v_g2s_thr_copy.partition_D(smem_v);
  auto V_identity_tensor = make_identity_tensor(shape(smem_v));
  auto tViV = v_g2s_thr_copy.partition_S(V_identity_tensor);
  auto tVpV = make_tensor<bool>(make_shape(size<1>(tViV), size<2>(tViV)));
  if (thread0()) {
    // PRINT(shape(smem_v));
    // PRINT(layout(tViV));
    // PRINT(tViV);
    // PRINT(shape(tVpV));
    // PRINT(shape(tVsV));
    // PRINT(shape(tVgV));
  }


  typename KernelTraits::MMA tiled_mma;

  auto s2r_tiled_copy_q = make_tiled_copy_A(typename KernelTraits::S2RCopyAtomQ{}, tiled_mma);
  auto s2r_thr_copy_q = s2r_tiled_copy_q.get_slice(threadIdx.x);
  auto thr_QsQ = s2r_thr_copy_q.partition_S(smem_q);  // ? (CPY, CPY_M, CPY_K, kStage)

  auto s2r_tiled_copy_k = make_tiled_copy_B(typename KernelTraits::S2RCopyAtomK{}, tiled_mma);
  auto s2r_thr_copy_k = s2r_tiled_copy_k.get_slice(threadIdx.x);
  auto thr_KsK = s2r_thr_copy_k.partition_S(smem_k);  // ? (CPY, CPY_M, CPY_K, kStage)


  auto thr_mma = tiled_mma.get_slice(threadIdx.x);


  auto thr_tQrQ =
      thr_mma.partition_fragment_A(smem_q(_, _)); // (MMA, MMA_M, MMA_K)

  auto thr_tKrK =
      ( thr_mma.partition_fragment_B(smem_k(_, _))); // (MMA, MMA_N, MMA_K)
  auto thr_tCrC = partition_fragment_C(
      tiled_mma,
      Shape<Int<KernelTraits::ThreadblockShape0::kM>,
            Int<KernelTraits::ThreadblockShape0::kN>>{}); // (MMA, MMA_M, MMA_N)

  auto thr_tKrK_view = s2r_thr_copy_k.retile_D(thr_tKrK);



  cp_async_wait<1>();
  __syncthreads();
  cute::copy(s2r_tiled_copy_q, thr_QsQ, thr_tQrQ);


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
  Initial_1D_Regs<ElementAccumulator, 2>(m_max, -INFINITY);
  Initial_1D_Regs<ElementAccumulator, 2>(d, 0.0f);



  int ntile = (params.T + KernelTraits::ThreadblockShape0::kN - 1) /
          KernelTraits::ThreadblockShape0::kN;

  ///< main loop
  for (int itile = 0; itile < ntile; ++itile) {
    cp_async_wait<0>();
    __syncthreads();

    tVgV = v_g2s_thr_copy.partition_S(tiled_block_v(_, _, itile));

    clear(tVsV);
    ///< only last split_F or tile need predicate
    if ((blockIdx.y + 1) * Split_kF > params.F ||
        (itile + 1) * KernelTraits::ThreadblockShape1::kK > params.T) {

      ///< V g2s before  gemm Q@K
      Operator_2D_Regs<size<1>(tViV), size<2>(tViV)>([&](int i, int j) {
        tVpV(i, j) =
            (int)get<1>(tViV(make_coord(0, 0), i, j)) <
                params.T - itile * KernelTraits::ThreadblockShape1::kK &&
            (int)get<0>(tViV(make_coord(0, 0), i, j)) <
                params.F - blockIdx.y * Split_kF;
      });

      ///< V g2s before  gemm Q@K
      cute::copy_if(v_tiled_copy, tVpV, tVgV, tVsV);
    } else {
      ///< V g2s before  gemm Q@K
      cute::copy(v_tiled_copy, tVgV, tVsV);
    }



    cp_async_fence();
    // smem_k -> reg_k
    cute::copy(s2r_tiled_copy_k, thr_KsK, thr_tKrK_view);

    clear(thr_tCrC);
    //gemm Q@K
    cute::gemm(tiled_mma, thr_tCrC, thr_tQrQ, thr_tKrK, thr_tCrC);

    ///< scale
    Operator_3D_Regs<size<0>(thr_tCrC), size<1>(thr_tCrC), size<2>(thr_tCrC)>(
      [&](int i, int j, int m){thr_tCrC(make_coord(i%2, i/2), j, m) *= (Element)params.Scale;}
    );
    ///< Mask mode
    if constexpr (TensorMask) {
      auto smem_m = make_tensor(make_smem_ptr((Element *)&smem.m_smem),
                                typename KernelTraits::SmemLayoutMask{});
      auto tensor_mask =
          make_tensor(make_gmem_ptr(mask), make_shape(params.S, params.T),
                      make_stride(params.T, Int<1>{}));
      auto tiled_tensor_mask =
          (local_tile(tensor_mask,
                      make_shape(Int<KernelTraits::ThreadblockShape0::kM>{},
                                 Int<KernelTraits::ThreadblockShape0::kN>{}),
                      make_coord(blockIdx.x, _)));

      typename KernelTraits::mask_g2s_copy_tile mask_g2s_copy_tile;
      auto mask_g2s_copy_thr = mask_g2s_copy_tile.get_slice(threadIdx.x);
      auto g2s_tMgM =
            mask_g2s_copy_thr.partition_S(tiled_tensor_mask(_, _, itile));
      auto g2s_tMsM = mask_g2s_copy_thr.partition_D(smem_m);
      clear(g2s_tMsM);

      //only last blockidx.x or last tile need predicate
      if ((blockIdx.x + 1) * KernelTraits::ThreadblockShape0::kM > params.S ||
          (itile + 1) * KernelTraits::ThreadblockShape0::kN > params.T) {
        auto tMpM =
            make_tensor<bool>(make_shape(size<1>(g2s_tMgM), size<2>(g2s_tMgM)));
        auto mask_identity_tensor = make_identity_tensor(shape(smem_m));
        auto tMiM = mask_g2s_copy_thr.partition_D(mask_identity_tensor);

        Operator_2D_Regs<size<1>(g2s_tMgM), size<2>(g2s_tMgM)>([&](int i,
                                                                   int j) {
          tMpM(i, j) =
              (int)get<0>(tMiM(make_coord(0, 0), i, j)) <
                  params.S - blockIdx.x * KernelTraits::ThreadblockShape0::kM &&
              (int)get<1>(tMiM(make_coord(0, 0), i, j)) <
                  params.T - itile * KernelTraits::ThreadblockShape0::kN;
        });
        cute::copy_if(mask_g2s_copy_tile, tMpM, g2s_tMgM, g2s_tMsM);
      } else {
        cute::copy(mask_g2s_copy_tile, g2s_tMgM, g2s_tMsM);
      }

      // Mask mode, for tiled_mma gemm and g2s mask copy
      __syncthreads();

      auto s2r_mask_tile_copy = make_tiled_copy_C(
          typename KernelTraits::S2RCopyAtomMask{}, tiled_mma1);
      auto s2r_mask_thr_copy = s2r_mask_tile_copy.get_slice(threadIdx.x);
      auto s2r_tMsM = s2r_mask_thr_copy.partition_S(smem_m);
      decltype(thr_tCrC) s2r_tMrM;

      cute::copy(s2r_mask_tile_copy, s2r_tMsM, s2r_tMrM);

      Operator_3D_Regs<size<0>(thr_tCrC), size<1>(thr_tCrC), size<2>(thr_tCrC)>(
          [&](int i, int j, int m) {
            thr_tCrC(make_coord(i % 2, i / 2), j, m) +=
                s2r_tMrM(make_coord(i % 2, i / 2), j, m);
          });

    }
    else {
      // Nomask mode, for tiled_mma gemm
      __syncthreads();
    }

    int next_itile = (itile + 1) % ntile;
    tKgK = k_g2s_thr_copy.partition_S(tiled_block_k(_, _, next_itile));
    clear(tKsK);
    ///< next_k g2s copy after gemm Q@K, only last tile need mask
  if (itile + 1 == ntile - 1) {
    #pragma unroll
    for (int i = 0; i < size<1>(tKiK); ++i) {
      ///< don't forget cast to int when compare, because type(get....) is not int
      tKpK(i) =  (int)get<0>(tKiK(make_coord(0, 0), i, 0)) < params.T - KernelTraits::ThreadblockShape0::kN * (itile + 1);
    }

    Operator_2D_Regs<size<1>(tKiK), size<2>(tKiK)>([&](int i, int j) {
      if (tKpK(i)) {
        copy(tKgK(_, i, j), tKsK(_, i, j));
      }
    });
  } else {
    ///< next_k g2s copy after gemm Q@K, tile other than last don't need mask
    copy(k_tiled_copy, tKgK, tKsK);
  }


    cp_async_fence();


    // deal with C residual
    if (itile == ntile - 1) {

  auto C_identity =
      make_identity_tensor(Shape<Int<KernelTraits::ThreadblockShape0::kM>,
                                 Int<KernelTraits::ThreadblockShape0::kN>>{});

  auto tCiC = thr_mma.partition_C(C_identity);
  auto tCpC = make_tensor<bool>(make_shape(_2{}, size<1>(tCiC), size<2>(tCiC)));

  Operator_3D_Regs<size<2>(tCpC), size<0>(tCpC), size<1>(tCpC)>(
      [&](int i, int j, int m) {
        tCpC(j, m, i) = (int)get<1>(tCiC(4 * i + j)) >=
                        params.T - itile * KernelTraits::ThreadblockShape0::kN;
      });

  Operator_3D_Regs<size<0>(thr_tCrC), size<1>(thr_tCrC), size<2>(thr_tCrC)>(
      [&](int i, int j, int m) {
        if (tCpC(i % 2, j, m)) {
          thr_tCrC(make_coord(i % 2, i / 2), j, m) =
              -platform::numeric_limits<Element>::infinity(); // half -inf
        }
      });
    }

    ElementAccumulator exp_oldm_sub_newm[2];
    Initial_1D_Regs<ElementAccumulator, 2>(exp_oldm_sub_newm, -INFINITY);

    Operator_Softmax<Element, ElementAccumulator>(thr_tCrC, exp_oldm_sub_newm, d, m_max, itile);

    ///< scale for O
Operator_3D_Regs<size<2>(mma1_tOrO), 2, 2>(
    [&](int i, int j, int m) {
        mma1_tOrO(make_coord(m, j), 0, i) = mma1_tOrO(make_coord(m, j), 0, i) * static_cast<Element>(exp_oldm_sub_newm[j]);
    });

    cp_async_wait<1>();
  __syncthreads();

    cute::copy(s2r_tiled_copy_v, mma1_tVsV, mma1_tVrV);
    ///< gemm  QK @ V
    cute::gemm(tiled_mma1, mma1_tOrO, tOrrC, mma1_tVrV, mma1_tOrO);

  }


///< last scale for O in softmax
Operator_3D_Regs<size<2>(mma1_tOrO), 2, 2>(
    [&](int i, int j, int m) {
        mma1_tOrO(make_coord(m, j), 0, i) = mma1_tOrO(make_coord(m, j), 0, i) / static_cast<Element>(d[j]);
    });

  ///< epilogue
  using EpilogueSharedStorage = typename KernelTraits::EpilogueSharedStorage;
  EpilogueSharedStorage &epi_smem_storage = *(reinterpret_cast<EpilogueSharedStorage *>(s));

  Tensor e_smem = make_tensor(make_smem_ptr((Element*)&epi_smem_storage.e_smem), typename KernelTraits::SmemLayoutEpilogue{});

  auto mma1_tOsO = thr_mma1.partition_C(e_smem);

  ///< epilogue reg -> smem
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
    tEpE_S(i) = (int)get<0>(tEiE(make_coord(0,0), i, 0)) + blockIdx.x * KernelTraits::ThreadblockShape1::kM < params.S;
  }

#pragma unroll
  for (int i = 0 ; i < size<0>(tEpE_F); ++i) {
    tEpE_F(i) = (int)get<1>(tEiE(make_coord(0,0), 0, i)) + blockIdx.y * KernelTraits::ThreadblockShape1::kN < params.F;

  }
  ///< epilogue smem -> global
  Operator_2D_Regs<size<1>(tEsE), size<2>(tEsE)>([&](int s, int f) {
    if (tEpE_S(s) && tEpE_F(f)) {
      cute::copy(tEsE(_, s, f), tEgE(_, s, f));
    }
  });
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
  // int N = 2, S = 111, T = 112, F = 128, H = 2;
  // int N = 2, S = 128, T = 128, F = 128, H = 2;
  // int N = 1, S = 128, T = 128, F = 128, H = 1;
  // int N = 1, S = 128, T = 61, F = 128, H = 1;
  // int N = 2, S = 1111, T = 1111, F = 104, H = 2;
  int N = 1, S = 1020, T = 1015, F = 128, H = 8;
  // int N = 1, S = 597, T = 491, F = 32, H = 1;
  // int N = 5, S = 100, T = 1, F = 32, H = 4;
  // int N = 1, S = 16, T = 32, F = 32, H = 1;
  MultiHeadAttentionProblemSize problem(N, S, T, kE, F, H);

  using KernelT = KernelTraits<cutlass::half_t, float, 8, kE, 128, UseMask, 1>;

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
  int smem_s = sizeof(typename KernelT::SharedStorage);
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
  //scale
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      each_head_cpu_C_gt[i * n + j] *= (Element)problem.Scale;
    }
  }
    if constexpr (UseMask) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        each_head_cpu_C_gt[i * n + j] +=  mask[i * n + j];
      }
    }
  }
  for (int i = 0; i < m; ++i) {
    ElementAccumulator m_max = -INFINITY;
    for (int j = 0; j < n; ++j) {
      m_max= max(m_max, (ElementAccumulator)each_head_cpu_C_gt[i * n + j]);
    }
    each_head_cpu_m_gt[i] = m_max;
  }




  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      auto temp = (Element)exp(each_head_cpu_C_gt[i * n + j] - each_head_cpu_m_gt[i]);
      // each_head_cpu_C_gt[i * n + j] = (Element)exp(each_head_cpu_C_gt[i * n + j] - each_head_cpu_m_gt[i]);
      // if (std::isinf(temp)) {
      //   printf("each_head_cpu_C_gt[%d * n + %d] :%f\n", i, j, (float)each_head_cpu_C_gt[i * n + j]);
      //   printf("each_head_cpu_m_gt[%d] :%f\n", i, (float)each_head_cpu_m_gt[i]);
      // }
      each_head_cpu_C_gt[i * n + j] = temp;
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
      // if (std::isnan((float)sum)) {
      //   printf("out row:%d col:%d is nan\n", i , j);
      // }
    }
  }
}
}
  helper.Regression(res_gt, (Element*)Out.first, false, false);
#endif

}