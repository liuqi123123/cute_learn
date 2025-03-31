
#include "conv_utils.h"
// #define CPU_CHECK
#define CLOCK_PRINT
using namespace cute;
// using BlockShape = cutlass::gemm::GemmShape<96, 128, 64>;
// using BlockShape = cutlass::gemm::GemmShape<64, 64, 64>;
// using WarpShape = cutlass::gemm::GemmShape<32, 16, 64>;

// SM80_16x8x32_S32S8S8S32_TN
using BlockShape = cutlass::gemm::GemmShape<96, 128, 64>;
using WarpShape = cutlass::gemm::GemmShape<48, 64, 64>;
using Element = int8_t;
using ElementC = half_t;
using ElementAccumulator = int32_t;
using ElementCompute = float;
static const int kStage= 3;




template <typename ElementA_, typename ElementB_, typename ElementC_,
          typename ElementCompute_, typename ElementAccumulator_,
          typename ElementOutput_, typename BlockShape_, typename WarpShape_, int kStage_, int Alignment_, int AlignC_>
struct KernelTraits {
  static const int kSmemLayoutCBatch = 1;

  using ElementA = ElementA_;
  using Element = ElementA;
  using ElementC = ElementC_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using BlockShape = BlockShape_;
  using WarpShape = WarpShape_;
  static const int kStage = kStage_;
  static const int warp_m = BlockShape::kM / WarpShape::kM;
  static const int warp_n = BlockShape::kN / WarpShape::kN;
  static const int warp_num = warp_m * warp_n;
  static const int Align = Alignment_;
  static const int AlignC = AlignC_;
  static const int Implit_A_G2S_Thr_kContigous = BlockShape::kK / Alignment_;
  static const int Implit_A_G2S_Thr_kStrided = warp_num * 32 / Implit_A_G2S_Thr_kContigous;
  using Implit_A_Thr_Coord_Layout = Layout<Shape<Int<Implit_A_G2S_Thr_kStrided>, Int<Implit_A_G2S_Thr_kContigous>>, Stride<_1, Int<Align>>>;
  static_assert(BlockShape::kM % Implit_A_G2S_Thr_kStrided == 0);
  static const int ActivationkStrided = (BlockShape::kM / Implit_A_G2S_Thr_kStrided);

  static const int Implit_B_G2S_Thr_kContigous = Implit_A_G2S_Thr_kContigous;
  static const int Implit_B_G2S_Thr_kStrided = Implit_A_G2S_Thr_kStrided;
  using Implit_B_Thr_Coord_Layout = Layout<Shape<Int<Implit_B_G2S_Thr_kContigous>, Int<Implit_B_G2S_Thr_kStrided>>, Stride<Int<Align>, _1>>;
  static const int FilterkStrided = (BlockShape::kN / Implit_B_G2S_Thr_kStrided);
  static_assert(BlockShape::kN % FilterkStrided == 0);



  using activate_g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using activate_g2s_copy_traits = Copy_Traits<activate_g2s_copy_op>;
  using activate_g2s_copy_atom = Copy_Atom<activate_g2s_copy_traits, Element>;
  using Activate_g2s_Tile_Copy = decltype(make_tiled_copy(
      activate_g2s_copy_atom{},
      make_layout(Shape<Int<Implit_A_G2S_Thr_kStrided>,
                        Int<Implit_A_G2S_Thr_kContigous>>{},
                  Stride<Int<Implit_A_G2S_Thr_kContigous>, _1>{}),
      Layout<Shape<_1, Int<Align>>>{}));
  using Filter_g2s_Tile_Copy = decltype(make_tiled_copy(
      activate_g2s_copy_atom{},
      make_layout(
          Shape<Int<Implit_B_G2S_Thr_kStrided>, Int<Implit_B_G2S_Thr_kContigous>>{},
          Stride<Int<Implit_B_G2S_Thr_kContigous>, _1>{}), make_layout(Shape<_1, Int<Align>>{})));

  struct SharedStorage {
    cutlass::AlignedArray<Element, BlockShape::kM*BlockShape::kK*kStage, 16> activate_smem;
    cutlass::AlignedArray<Element, BlockShape::kN*BlockShape::kK*kStage, 16> filter_smem;
  };

  using SmemLayoutAtomA = decltype(composition(
    // B M S
    Swizzle<3, 4, 3>{}, make_layout(Shape<Int<BlockShape::kM>, Int<BlockShape::kK>>{},
                              Stride<Int<BlockShape::kK>, _1>{})));
  using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{}, Shape<Int<BlockShape::kM>, Int<BlockShape::kK>, Int<kStage>>{}));

  using SmemLayoutAtomF = decltype(composition(
    // B M S
    Swizzle<3, 4, 3>{}, make_layout(Shape<Int<BlockShape::kN>, Int<BlockShape::kK>>{},
                              Stride<Int<BlockShape::kK>, _1>{})));
  using SmemLayoutF = decltype(tile_to_shape(SmemLayoutAtomF{}, Shape<Int<BlockShape::kN>, Int<BlockShape::kK>, Int<kStage>>{}));



  static constexpr int kMmaEURepeatM = warp_m;
  static constexpr int kMmaEURepeatN = warp_n;
  static constexpr int kMmaEURepeatK = 1;

  using mma_op = SM80_16x8x32_S32S8S8S32_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;
  using MMA = decltype(make_tiled_mma(
      mma_atom{},
      make_layout(
          Shape<Int<kMmaEURepeatM>, Int<kMmaEURepeatN>, Int<kMmaEURepeatK>>{}),
      Tile<Int<1>, Int<1>, Int<1>>{}));
  static const int kWarpGemmIterations =  WarpShape::kK / get<2>(typename mma_traits::Shape_MNK{});


    /// Number of cp.async instructions to load on group of operand A
  static int const kAccessesPerGroupA =
        (ActivationkStrided + kWarpGemmIterations - 1) / kWarpGemmIterations;

    /// Number of cp.async instructions to load on group of operand B
  static int const kAccessesPerGroupB =
        (FilterkStrided + kWarpGemmIterations - 1) / kWarpGemmIterations;

  using activate_s2r_copy_op = SM75_U32x4_LDSM_N;
  using activate_s2r_copy_traits = Copy_Traits<activate_s2r_copy_op>;
  using activate_s2r_copy_atom = Copy_Atom<activate_s2r_copy_traits, Element>;

  using filter_s2r_copy_op = SM75_U32x2_LDSM_N;
  using filter_s2r_copy_traits = Copy_Traits<filter_s2r_copy_op>;
  using filter_s2r_copy_atom = Copy_Atom<filter_s2r_copy_traits, Element>;

  using SmemLayoutAtomEpilogue = decltype(composition(
      // B M S
      Swizzle<2, 3, 2>{},
      // Swizzle<0, 0, 0>{},
      make_layout(Shape<Int<BlockShape::kM>, Int<BlockShape::kN>>{},
                  Stride<Int<BlockShape::kN>, _1>{})));
  using SmemLayoutEpilogue = decltype(tile_to_shape(SmemLayoutAtomEpilogue{}, Shape<Int<BlockShape::kM>, Int<BlockShape::kN>>{}));

  using mma_atom_shape = mma_traits::Shape_MNK;
  static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{}); // 1 * 2 * 16 = 32
  static constexpr int kMmaPN = 1 * kMmaEURepeatN * get<1>(mma_atom_shape{}); // 1 * 2 * 8 = 16
  static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{}); // 1 * 1 * 32 = 32

  using SmemLayoutAtomC = decltype(composition(
      Swizzle<1, 2, 3>{}, make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}), // 32 * 16
      // Swizzle<1, 3, 2>{}, make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}), // 32 * 16
                                      make_stride(Int<kMmaPN>{}, Int<1>{})))); // 32 , 1

  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomC{},
      make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{}))); // 32 16 1
  static const int AlignAccumulate =  16 / sizeof(ElementAccumulator);
  static const int e_s2g_copy_thr_contiguous = Int<kMmaPN>{} / AlignC;
  static_assert(e_s2g_copy_thr_contiguous <= warp_num * 32);

  static const int EpiJoinThreadNum = size(SmemLayoutC{}) / AlignC;
  // static_assert(EpiJoinThreadNum == 64);
  static_assert(e_s2g_copy_thr_contiguous == 2);
  using e_s2g_copy_op = UniversalCopy<cutlass::AlignedArray<ElementAccumulator, AlignC>>;
  using e_s2g_copy_traits = Copy_Traits<e_s2g_copy_op>;
  using e_s2g_copy_atom = Copy_Atom<e_s2g_copy_traits, ElementAccumulator>;

  using e_s2g_copy_tile = decltype(make_tiled_copy(
      // e_s2g_copy_atom{}, make_layout(Shape<Int<warp_num * 32 / e_s2g_copy_thr_contiguous>,
      e_s2g_copy_atom{}, make_layout(Shape<Int<EpiJoinThreadNum / e_s2g_copy_thr_contiguous>,
                                      Int<e_s2g_copy_thr_contiguous>>{}, Stride<Int<e_s2g_copy_thr_contiguous>, _1>{}),
      Layout<Shape<_1, Int<AlignC>>>{}));

  using e_c_s2g_copy_op = UniversalCopy<cutlass::AlignedArray<ElementC, AlignC>>;
  using e_c_s2g_copy_traits = Copy_Traits<e_c_s2g_copy_op>;
  using e_c_s2g_copy_atom = Copy_Atom<e_c_s2g_copy_traits, ElementC>;
  using e_c_s2g_copy_tile = decltype(make_tiled_copy(
      // e_c_s2g_copy_atom{}, make_layout(Shape<Int<warp_num * 32 / e_s2g_copy_thr_contiguous>,
      e_c_s2g_copy_atom{}, make_layout(Shape<Int<EpiJoinThreadNum / e_s2g_copy_thr_contiguous>,
                                      Int<e_s2g_copy_thr_contiguous>>{}, Stride<Int<e_s2g_copy_thr_contiguous>, _1>{}),
      Layout<Shape<_1, Int<AlignC>>>{}));


  using e_scale_s2g_copy_op = UniversalCopy<cutlass::AlignedArray<ElementCompute, AlignC>>;
  using e_scale_s2g_copy_traits = Copy_Traits<e_scale_s2g_copy_op>;
  using e_scale_s2g_copy_atom = Copy_Atom<e_scale_s2g_copy_traits, ElementCompute>;
  using e_scale_s2g_copy_tile = decltype(make_tiled_copy(
      // e_scale_s2g_copy_atom{}, make_layout(Shape<Int<warp_num * 32 / e_s2g_copy_thr_contiguous>,
      e_scale_s2g_copy_atom{}, make_layout(Shape<Int<EpiJoinThreadNum / e_s2g_copy_thr_contiguous>,
                                      Int<e_s2g_copy_thr_contiguous>>{}, Stride<Int<e_s2g_copy_thr_contiguous>, _1>{}),
      Layout<Shape<_1, Int<AlignC>>>{}));


  using R2SCopyAtomC = Copy_Atom<UniversalCopy<cute::uint64_t>, ElementAccumulator>;
  using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, ElementAccumulator>;
  // using S2GCopyC = decltype(make_tiled_copy(
  //     e_s2g_copy_atom{},
  //     make_layout(Shape<Int<warp_num * 32 / e_s2g_copy_thr_contiguous>,
  //                       Int<e_s2g_copy_thr_contiguous>>{},
  //                 Stride<Int<e_s2g_copy_thr_contiguous>, _1>{}),
  //     Layout<Shape<_1, Int<AlignC>>>{}));


  // struct EpilogueSharedStorage {
  //   cutlass::AlignedArray<ElementAccumulator, BlockShape::kM*BlockShape::kN, 16> epilogue_smem;
  // };

  struct EpilogueSharedStorage {
    cutlass::AlignedArray<ElementAccumulator, size(SmemLayoutC{}), 16> epilogue_smem;
  };


};


struct ParamsPack {
  cutlass::conv::Conv2dProblemSize p;
  const Element *a;
  const Element *b;
  ElementC *c;
  ActivateParams const ap;
   FilterParams const fp;
  const ElementCompute *device_scale;
  const ElementCompute *device_bias;
  uint32_t *device_time_info;
};





// template<typename KernelTraits>
// __global__ void Conv(cutlass::conv::Conv2dProblemSize p, const typename KernelTraits::Element *a, const typename KernelTraits::Element *b,
//                                 typename KernelTraits::ElementC *c, ActivateParams const ap, FilterParams const fp, const typename KernelTraits::ElementCompute * device_scale,  const typename KernelTraits::ElementCompute * device_bias, uint32_t* device_time_info) {
template<typename KernelTraits>
__global__ void Conv(ParamsPack params_pack) {
  cutlass::conv::Conv2dProblemSize& p = params_pack.p;
  const typename KernelTraits::Element *a = params_pack.a;
  const typename KernelTraits::Element *b = params_pack.b;
  typename KernelTraits::ElementC *c = params_pack.c;
  ActivateParams const& ap = params_pack.ap;
   FilterParams const& fp = params_pack.fp;
  const typename KernelTraits::ElementCompute *device_scale = params_pack.device_scale;
  const typename KernelTraits::ElementCompute *device_bias = params_pack.device_bias;
  uint32_t *device_time_info = params_pack.device_time_info;



  unsigned int start, end;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(start) :: "memory");
  extern __shared__ int s0[];
  static const int kStage = KernelTraits::kStage;
  if (thread0()) {
    // printf("KernelTraits::ActivationkStrided:%d\n", KernelTraits::ActivationkStrided);
    // printf("KernelTraits::FilterkStrided:%d\n", KernelTraits::FilterkStrided);
  }
  static const int Align = KernelTraits::Align;
  using BlockShape = typename KernelTraits::BlockShape;
  typename KernelTraits::SharedStorage& s = *(reinterpret_cast<typename KernelTraits::SharedStorage*>(s0));
  using Element = typename KernelTraits::ElementA;
  // int implit_m = p.N*p.P*p.Q;
  // int implit_k = p.R*p.S*p.C;
  // int implit_n = p.K;
  // auto implit_gemm_A = make_layout(Shape<p.N*p.O*p.Q, p.R*p.S*p.C>, Stride<>)
  auto activation_layout = make_layout(make_shape(p.N, p.H, p.W, p.C), make_stride(p.H*p.W*p.C, p.W*p.C, p.C, 1));
  auto activation = make_tensor(make_gmem_ptr(a), activation_layout);

  auto filter_layout = make_layout(make_shape(p.K, p.R, p.S, p.C), make_stride(p.R*p.S*p.C, p.S*p.C, p.C, 1));
  auto filter = make_tensor(make_gmem_ptr(b), filter_layout);
  auto output = make_tensor(make_gmem_ptr(c), make_shape(p.N*p.P*p.Q, p.K), make_stride(p.K, 1));
  auto scale_tensor = make_tensor(make_gmem_ptr(device_scale), make_shape(_1{}, p.K), make_stride(_0{}, _1{}));
  auto bias_tensor = make_tensor(make_gmem_ptr(device_bias), make_shape(_1{}, p.K), make_stride(_0{}, _1{}));
  auto block_scale_tensor = local_tile(scale_tensor, Shape<_1, Int<BlockShape::kN>>{}, make_coord(0, blockIdx.y));
  auto block_bias_tensor = local_tile(bias_tensor, Shape<_1, Int<BlockShape::kN>>{}, make_coord(0, blockIdx.y));

  auto block_output = local_tile(output, Shape<Int<BlockShape::kM>, Int<BlockShape::kN>>{}, make_coord(blockIdx.x, blockIdx.y));
  auto smem_activate =
      make_tensor(make_smem_ptr((Element *)(&s.activate_smem)), typename KernelTraits::SmemLayoutA{});
  auto smem_filter =
      make_tensor(make_smem_ptr((Element *)(&s.filter_smem)), typename KernelTraits::SmemLayoutF{});

  char const *activate_pointer[KernelTraits::ActivationkStrided];

  int filter_r_ = 0;
  int filter_s_ = 0;
  int filter_c_ = 0;
  int filter_rs_ = 0;
  int32_t masks_[KernelTraits::ActivationkStrided][2];

  typename KernelTraits::Implit_A_Thr_Coord_Layout implit_A_thr_coord_layout;

  // filter_c_ = threadIdx.x % KernelTraits::Implit_A_G2S_Thr_kContigous * KernelTraits::Align;
  filter_c_ = implit_A_thr_coord_layout(0, threadIdx.x % KernelTraits::Implit_A_G2S_Thr_kContigous);
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x < 128) {
    // printf("threadIddx:%d, filter_c_:%d\n", threadIdx.x, filter_c_);
  }
  int offset_n[KernelTraits::ActivationkStrided];
  int offset_p[KernelTraits::ActivationkStrided];
  int offset_q[KernelTraits::ActivationkStrided];
  CUTLASS_PRAGMA_UNROLL
  for (int s = 0; s < KernelTraits::ActivationkStrided; ++s) {
    activate_pointer[s] = reinterpret_cast<char const *>(a);
    int offset_npq = blockIdx.x * KernelTraits::BlockShape::kM + s * KernelTraits::Implit_A_G2S_Thr_kStrided + implit_A_thr_coord_layout(threadIdx.x / KernelTraits::Implit_A_G2S_Thr_kContigous, 0);
  //   if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x < 128 && s == 2) {
  //   printf("threadIddx:%d, s:%d, offset_npq:%d\n", threadIdx.x,s, offset_npq);
  // }
    int residual;
    ap.pq_divmod(offset_n[s], residual, offset_npq);
    ap.q_divmod(offset_p[s], offset_q[s], residual);

    auto coord = at(p, offset_n[s], offset_p[s], offset_q[s], 0, 0, filter_c_);
    activate_pointer[s] += activation_layout(coord) * sizeof(Element);
  }
  clear_mask<KernelTraits::ActivationkStrided>(true, masks_);

    CUTLASS_PRAGMA_NO_UNROLL
    for (int r = 0; r < p.R; ++r) {
      CUTLASS_PRAGMA_UNROLL
      for (int s_idx = 0; s_idx < KernelTraits::ActivationkStrided; ++s_idx) {

        int r_ = r;
        if (p.mode == cutlass::conv::Mode::kConvolution) {
          r_ = p.R - 1 - r;
        }

        int h = offset_p[s_idx] * p.stride_h - p.pad_h + r_ * p.dilation_h;

        bool pred = (offset_n[s_idx] < p.N && h >= 0 && h < p.H);
          masks_[s_idx][0] |= (pred << r);
      }
    }

    CUTLASS_PRAGMA_NO_UNROLL
    for (int s = 0; s < p.S; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int s_idx = 0; s_idx < KernelTraits::ActivationkStrided; ++s_idx) {

        int s_ = s;
        if (p.mode == cutlass::conv::Mode::kConvolution) {
          s_ = p.S - 1 - s;
        }
        int w = offset_q[s_idx] * p.stride_w - p.pad_w + s_ * p.dilation_w;

        bool pred = (w >= 0 && w < p.W);
          masks_[s_idx][1] |= (pred << s);
      }
    }

    clear_mask(filter_c_  >= p.C, masks_);
      // if (threadIdx.x >= 124 && blockIdx.x == 0 && blockIdx.y == 0) {
      //     printf("threadIdx.x:%d, masks_[%d][%d]:%d\n", threadIdx.x, 0, 0, masks_[0][0]);
      //     printf("threadIdx.x:%d, masks_[%d][%d]:%d\n", threadIdx.x, 0, 1, masks_[0][1]);
      //   }
    // set_iteration_index(0);

  typename KernelTraits::Implit_B_Thr_Coord_Layout implit_B_thr_coord_layout;


    const char* filter_point = reinterpret_cast<const char*>(b);
    uint32_t filter_predicates(0);
    int32_t column = blockIdx.y * KernelTraits::BlockShape::kN + implit_B_thr_coord_layout(_0{}, threadIdx.x / KernelTraits::Implit_B_G2S_Thr_kContigous);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < KernelTraits::FilterkStrided; ++s) {
      uint32_t pred = ((column + s * KernelTraits::Implit_B_G2S_Thr_kStrided < p.K) ? 1u : 0);
      if (thread0()) {
        // printf("column:%d, s * KernelTraits::Implit_B_G2S_Thr_kStrided:%d, pred:%d\n", column,s * KernelTraits::Implit_B_G2S_Thr_kStrided, pred);
      }
      filter_predicates |= (pred << s);
    }

    // printf("threadIdx.x:%d, filter_c_:%d\n", threadIdx.x, filter_c_);
    clear_filter_mask(filter_c_ >= p.C, filter_predicates);

    filter_point += filter_layout(column, 0, 0, filter_c_);
    // set_iteration_index(0);
    typename KernelTraits::Activate_g2s_Tile_Copy activate_g2s_tile_copy;
    auto activate_g2s_thr_copy = activate_g2s_tile_copy.get_slice(threadIdx.x);
    auto activate_g2s_thr_copy_D = activate_g2s_thr_copy.partition_D(smem_activate);
    if (thread0()) {
      // PRINT(layout(activate_g2s_thr_copy_D));
    }

    typename KernelTraits::Filter_g2s_Tile_Copy filter_g2s_tile_copy;
    auto filter_g2s_thr_copy = filter_g2s_tile_copy.get_slice(threadIdx.x);
    auto filter_g2s_thr_copy_D = filter_g2s_thr_copy.partition_D(smem_filter);
    if (thread0()) {
      // PRINT(layout(activate_g2s_thr_copy_D));
      // PRINT(layout(filter_g2s_thr_copy_D));
    }
    typename KernelTraits::MMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto thr_mma_tArA = thr_mma.partition_fragment_A(smem_activate(_, _, 0));
    auto thr_mma_tFrF = thr_mma.partition_fragment_B(smem_filter(_, _, 0));

    auto activation_s2r_tiled_copy = make_tiled_copy_A(typename KernelTraits::activate_s2r_copy_atom{}, tiled_mma);
    auto activation_s2r_thr_copy = activation_s2r_tiled_copy.get_slice(threadIdx.x);
    auto activation_s2r_tASA = activation_s2r_thr_copy.partition_S(smem_activate);
    auto thr_mma_tArA_view = activation_s2r_thr_copy.retile_D(thr_mma_tArA);
    auto thr_mma_tArA_view_shape = thr_mma_tArA_view.shape();
    auto thr_mma_tArA_view_stride = thr_mma_tArA_view.stride();
    // to get ((_16,_1),_3,_2):((_1,_0),_16,_48), how to simplify?
    auto thr_mma_tArA_overlap = make_tensor_like<KernelTraits::ElementA>(
        make_layout(thr_mma_tArA_view_shape,
                    make_stride(get<0>(thr_mma_tArA_view_stride),
                                get<1>(thr_mma_tArA_view_stride),
                                size<0>(thr_mma_tArA_view_shape) *
                                    size<1>(thr_mma_tArA_view_shape))));

    //  ((_16,_1),_3,_2):((_1,_0),_16,_48)
    // auto thr_mma_tArA_overlap = make_tensor_like<KernelTraits::ElementA>(
    //     make_layout(Shape<Shape<_16, _1>, _3, _2>{}, Stride<Stride<_1, _0>, _16, Int<48>>{})
    //     );


    if (thread0()) {
      // PRINT(layout(thr_mma_tArA));
      // PRINT(layout(thr_mma_tArA_view));
      // PRINT(layout(thr_mma_tArA_overlap));
    }

    auto filter_s2r_tiled_copy = make_tiled_copy_B(typename KernelTraits::filter_s2r_copy_atom{}, tiled_mma);
    auto filter_s2r_thr_copy = filter_s2r_tiled_copy.get_slice(threadIdx.x);
    auto filter_s2r_tFSF = filter_s2r_thr_copy.partition_S(smem_filter);
    auto thr_mma_tFrF_view = filter_s2r_thr_copy.retile_D(thr_mma_tFrF);

    auto thr_mma_tFrF_view_shape = thr_mma_tFrF_view.shape();
    auto thr_mma_tFrF_view_stride = thr_mma_tFrF_view.stride();

  // to get layout((_8,_1),_8,_2):((_1,_0),_8,_64), how to simply?

    auto thr_mma_tFrF_overlap = make_tensor_like<KernelTraits::ElementA>(
        make_layout(thr_mma_tFrF_view_shape,
                    make_stride(get<0>(thr_mma_tFrF_view_stride),
                                get<1>(thr_mma_tFrF_view_stride),
                                size<0>(thr_mma_tFrF_view_shape) *
                                    size<1>(thr_mma_tFrF_view_shape)
                                    )));
// auto thr_mma_tFrF_overlap = make_tensor_like<KernelTraits::ElementA>(
//   make_layout(Shape<Shape<_8, _1>, _8, _2>{}, Stride<Stride<_1, _0>, _8, Int<64>>{})
// );
    if (thread0()) {
      // PRINT(layout(thr_mma_tFrF_view));
      // PRINT((thr_mma_tFrF_view_stride));
      // PRINT(layout(thr_mma_tFrF_overlap));
    }
    // return;

        auto thr_mma_tCrC = partition_fragment_C(
            tiled_mma, Shape<Int<KernelTraits::BlockShape::kM>,
                             Int<KernelTraits::BlockShape::kN>>{});

    auto activate_thr_valid = [&](int iteration_strided){
      return
            (masks_[iteration_strided][0] & ((1) << filter_r_)) &&
          (masks_[iteration_strided][1] & ((1) << filter_s_));
    };

    auto filter_thr_valid = [&](int iteration_strided){
      return (filter_predicates & (1u << iteration_strided));

    };
    auto activate_add_byte_offset = [&](int byte_offset) {
      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < KernelTraits::ActivationkStrided; ++s) {
        activate_pointer[s] += byte_offset;
      }
    };
    auto activate_advance = [&]() {
      int next_idx = 0;

      // moves to the next tile
      ++filter_s_;
      if (filter_s_ == p.S) {
        filter_s_ = 0;
        ++filter_r_;

        if (filter_r_ < p.R) {
          next_idx = 1;
        } else {
          filter_r_ = 0;
          next_idx = 2;
        }
      }

      activate_add_byte_offset(ap.inc_next[next_idx]);
      if (thread0()) {
        // printf("next_idx:%d, ap.inc_next[next_idx]:%d\n",next_idx, ap.inc_next[next_idx]);
      }

      if (next_idx == 2) {
        filter_c_ += ap.filter_c_delta;
      }
      clear_mask(filter_c_ >= p.C, masks_);
    };

    auto filter_advance = [&]() {
      int next = fp.inc_next_rs;

      // moves to the next tile
      ++filter_rs_;
      if (filter_rs_ == fp.RS) {

        filter_rs_ = 0;
        next = fp.inc_next_c;
        if (thread0()) {
          // printf("@@@@@@hit next = fp.inc_next_c\n");
        }
        // filter_c_ += ap.filter_c_delta; // already move in activation
      }
      clear_filter_mask(filter_c_ >= p.C, filter_predicates);
      filter_point += next;

    };

    auto ActivationG2S = [&](int ismem_write) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < KernelTraits::ActivationkStrided; ++i) {
          cutlass::arch::cp_async_zfill<Align * sizeof(Element)>(
              activate_g2s_thr_copy_D(_, i, 0, ismem_write).data().get(),
              (const void *)activate_pointer[i], activate_thr_valid(i));
      }
    };

    auto FilterG2S = [&](int ismem_write){
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < KernelTraits::FilterkStrided; ++i) {

          cutlass::arch::cp_async_zfill<Align * sizeof(Element)>(
              filter_g2s_thr_copy_D(_, i, 0, ismem_write).data().get(),
              (const void *)filter_point, filter_thr_valid(i));
        if (i <= KernelTraits::FilterkStrided - 2) {
          filter_point += fp.inc_next_k;
        }
      }
    };

    auto copy_tiles_and_advance = [&](int ismem_write, int group_start_A = 0,
                                      int group_start_B = 0) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < KernelTraits::kAccessesPerGroupA; ++j) {
        if (group_start_A + j < KernelTraits::ActivationkStrided) {
          if (thread0()) {
              // printf("group_start_A + j:%d\n", group_start_A + j);
          }

          cutlass::arch::cp_async_zfill<Align * sizeof(Element)>(
              activate_g2s_thr_copy_D(_, group_start_A + j , 0, ismem_write).data().get(),
              (const void *)activate_pointer[group_start_A + j], activate_thr_valid(group_start_A + j));
          // if (group_start_A + j == KernelTraits::ActivationkStrided - 1) {
          //   activate_advance();
          //   if (thread0()) {
          //     printf("ismem_write:%d, has activate_advance\n", ismem_write);
          //   }
          // }
        }
      }


      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < KernelTraits::kAccessesPerGroupB; ++j) {
        if (group_start_B + j < KernelTraits::FilterkStrided) {
          if (thread0()) {
              // printf("group_start_B + j:%d\n", group_start_B + j);
          }
                    cutlass::arch::cp_async_zfill<Align * sizeof(Element)>(
              filter_g2s_thr_copy_D(_, group_start_B + j, 0, ismem_write).data().get(),
              (const void *)filter_point, filter_thr_valid(group_start_B + j));
        if (group_start_B + j <= KernelTraits::FilterkStrided - 2) {
          filter_point += fp.inc_next_k;
        }
        // else if (group_start_B + j == KernelTraits::FilterkStrided - 1) {
        //   filter_advance();
        //     if (thread0()) {
        //       printf("ismem_write:%d, has filter_advance\n", ismem_write);
        //     }
        // }
      }

      }
};



    int itile_to_read = 0;
    int ismem_read = 0;
    // int ismem_write = 0;

    int gemm_k_iterations =
        p.R * p.S * cute::ceil_div(p.C, Int<KernelTraits::BlockShape::kK>{});
    if (thread0()) {
    // printf("initial gemm_k_iterations: %d\n",  gemm_k_iterations);

    }


#if 1
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < kStage - 1;
         ++stage, --gemm_k_iterations) {
      ActivationG2S(stage);
      FilterG2S(stage);

      activate_advance();
      filter_advance();

      cp_async_fence();

      ++itile_to_read;
      // ++ismem_write;

    }

   cp_async_wait<kStage - 2>();
  __syncthreads();

  // int ik = 0;
  // smem -> reg
      cute::copy(activation_s2r_tiled_copy, activation_s2r_tASA(_, _, 0, ismem_read),
                 thr_mma_tArA_overlap(_, _, 0));
      cute::copy(filter_s2r_tiled_copy, filter_s2r_tFSF(_, _, 0, ismem_read),
                 thr_mma_tFrF_overlap(_, _, 0));
      if (thread0()) {
        // PRINT(layout(activation_s2r_tASA(_, _, 0, ismem_read)));
        // PRINT(layout(thr_mma_tArA_overlap(_, _, 0)));
      }

    // int smem_write_stage_idx = kStage - 1;
    int ismem_write = kStage - 1;
    // int smem_read_stage_idx = 0;
    copy_tiles_and_advance(ismem_write);

    clear(thr_mma_tCrC);

    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > (-kStage + 1);) {
      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < KernelTraits::kWarpGemmIterations;
           ++warp_mma_k) {
          int warp_mma_k_next = (warp_mma_k + 1) % KernelTraits::kWarpGemmIterations;
          if (thread0()) {
            // printf("ismem_read: %d\n", ismem_read);
            // printf("ismem_write: %d\n", ismem_write);
            // printf("gemm_k_iterations: %d\n", gemm_k_iterations);
            // printf("warp_mma_k_next: %d\n", warp_mma_k_next);
          }

          cute::copy(activation_s2r_tiled_copy,
                     activation_s2r_tASA(_, _, warp_mma_k_next, ismem_read),
                     thr_mma_tArA_overlap(_, _, (warp_mma_k + 1) % 2));
          cute::copy(filter_s2r_tiled_copy,
                     filter_s2r_tFSF(_, _, warp_mma_k_next, ismem_read),
                     thr_mma_tFrF_overlap(_, _, (warp_mma_k + 1) % 2));
          int group_start_iteration_A, group_start_iteration_B;

          if (warp_mma_k + 1 == KernelTraits::kWarpGemmIterations) {
          group_start_iteration_A = 0;
          group_start_iteration_B = 0;
          } else {
          group_start_iteration_A =
              (warp_mma_k + 1) * KernelTraits::kAccessesPerGroupA;
          group_start_iteration_B =
              (warp_mma_k + 1) * KernelTraits::kAccessesPerGroupB;
          }

          copy_tiles_and_advance(ismem_write, group_start_iteration_A,
                                 group_start_iteration_B);
          if (thread0()) {
            // PRINT_TENSOR(thr_mma_tArA_overlap(_, _, warp_mma_k % 2));
            // PRINT_TENSOR(thr_mma_tFrF_overlap(_, _, warp_mma_k % 2));
          }
          cute::gemm(tiled_mma, thr_mma_tCrC,
                     thr_mma_tArA_overlap(_, _, warp_mma_k % 2),
                     thr_mma_tFrF_overlap(_, _, warp_mma_k % 2), thr_mma_tCrC);
          if (warp_mma_k + 2 == KernelTraits::kWarpGemmIterations) {
          // Inserts a fence to group cp.async instructions into stages.
          cutlass::arch::cp_async_fence();

          // Waits until kStages-2 stages of cp.async have committed
          cp_async_wait<kStage - 2>();
          __syncthreads();

          activate_advance();
          filter_advance();

          if (ismem_write == (kStage - 1)) {
            ismem_write = 0;
          } else {
            ++ismem_write;
          }
          if (ismem_read == (kStage - 1)) {
            ismem_read = 0;
          }else {
            ++ismem_read;
          }
          // ismem_read = (ismem_read + 1) % kStage;
          --gemm_k_iterations;
          }
      }
    }
#endif
    // clear(thr_mma_tCrC);
  if (thread0()) {
    // printf("thr_mma_tArA_overlap: %d\n",thr_mma_tArA_overlap(make_coord(_0{}, _0{}), _0{}, _0{}));
    // PRINT_TENSOR(thr_mma_tCrC);
    // PRINT(layout(thr_mma_tCrC));
  }
//     device_time_info[2] = ((uint32_t)size<0>(thr_mma_tCrC));
#ifdef CLOCK_PRINT

asm volatile("mov.u32 %0, %%clock;" : "=r"(end) :: "memory");
  // uint32_t time_us = (end - start) *  1000.0f/ clockRate;
  if (thread0()) {
    device_time_info[0] = end - start;
    // PRINT(layout(thr_mma_tCrC));
    // printf("%d\n", thr_mma_tCrC(make_coord(0, 0), 0, 0));
    // printf("%d\n", int(thr_mma_tCrC.data()));
    // PRINT_TENSOR((thr_mma_tCrC(make_coord(0, 0), 0, 0)));
    // device_time_info[2] = thr_mma_tCrC(make_coord(0, 0), 0, 0);
  }
  // if (thread0()) {
  //   printf("Kernel multistage clock: %d,\n",end - start);
  // }

  unsigned int start1, end1;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(start1) :: "memory");
#endif

#if 1

    using ElementAccumulator = typename KernelTraits::ElementAccumulator;
    ///> epilgue
    typename KernelTraits::EpilogueSharedStorage& e_s = *(reinterpret_cast<typename KernelTraits::EpilogueSharedStorage*>(s0));
    // auto e_smem = make_tensor(make_smem_ptr((ElementAccumulator*)(&e_s.epilogue_smem)), typename KernelTraits::SmemLayoutEpilogue{});
    auto e_smem = make_tensor(make_smem_ptr((ElementAccumulator*)(&e_s.epilogue_smem)), typename KernelTraits::SmemLayoutC{});
    auto r2s_tiled_copy_c = make_tiled_copy_C(typename KernelTraits::R2SCopyAtomC{}, tiled_mma);
    auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(threadIdx.x);
    auto tCrC_r2s = r2s_thr_copy_c.retile_S(thr_mma_tCrC);   // (CPY, CPY_M, CPY_N)
    auto tCsC_r2s = r2s_thr_copy_c.partition_D(e_smem);  // (CPY, _1, _1, pipe)
    if (thread0()) {
      // PRINT(layout(thr_mma_tCrC));
      PRINT(layout(tCrC_r2s));
      PRINT(layout(tCsC_r2s));
    }


    typename KernelTraits::e_s2g_copy_tile s2g_tiled_copy_c;
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_slice(threadIdx.x);
    auto tCsC_s2g = s2g_thr_copy_c.partition_S(e_smem);// (CPY, _1, _1, pipe)
    // auto tCrC_s2g = make_tensor_like<ElementAccumulator>(tCsC_s2g);
    // auto tCgC_s2g = s2g_thr_copy_c.partition_D(block_output);// (CPY, CPY_M, CPY_N)


    typename KernelTraits::e_c_s2g_copy_tile s2g_c_tiled_copy_c;
    auto s2g_c_thr_copy_c = s2g_c_tiled_copy_c.get_slice(threadIdx.x);
    auto tCgC_s2g = s2g_c_thr_copy_c.partition_D(block_output);// (CPY, CPY_M, CPY_N)

    typename KernelTraits::e_scale_s2g_copy_tile s2g_scale_tiled_copy_c;
    auto s2g_scale_thr_copy_c = s2g_scale_tiled_copy_c.get_slice(threadIdx.x);
    auto tCgScale = s2g_scale_thr_copy_c.partition_D(block_scale_tensor);
    auto tCgBias = s2g_scale_thr_copy_c.partition_D(block_bias_tensor);
    auto tCrScale = make_tensor_like<ElementCompute>(tCgScale);
    auto tCrBias = make_tensor_like<ElementCompute>(tCgBias);
    // auto tCrScale = make_tensor_like<ElementCompute>(tCgScale(_, 0, 0));
    // auto tCrBias = make_tensor_like<ElementCompute>(tCgBias(_, 0, 0));
    // cute::copy(s2g_scale_tiled_copy_c, tCgScale, tCrScale);
    // cute::copy(s2g_scale_tiled_copy_c, tCgBias, tCrBias);

    // if (thread0()){
    //   PRINT(layout(tCgC_s2g));
    //   PRINT(layout(tCgScale));
    //   PRINT(layout(tCgBias));
    // }

   auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);  // (CPY_, CPY_MN)
    auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);  // (CPY_, CPY_MN)


    bool need_pred = ((blockIdx.x + 1) * BlockShape::kM > p.N * p. P * p.Q) || ((blockIdx.y + 1) * BlockShape::kN > p.K);

    // auto tOrM = make_tensor<bool>(make_shape(size<1>(epi_s2g_tEGE)));
    // auto tOrN = make_tensor<bool>(make_shape(size<2>(epi_s2g_tEGE)));
  if (thread0()) {
    // printf("=====================\n");

    // PRINT(layout(typename KernelTraits::SmemLayoutC{}));
    // PRINT(layout(thr_mma_tCrC));
    // PRINT(layout(tCrC_r2s));
    // PRINT(layout(tCsC_r2s));
    // printf("|||||||||||||||||||\n");
    PRINT(layout(tCsC_s2g));
    // PRINT(layout(tCgC_s2g));
    // PRINT(layout(tCgC_s2gx));
    // PRINT(layout(tCrC_r2sx));
    // printf("=====================\n");
    // printf("BlockShape::kM:%d BlockShape::kN:%d size typename "
    //        "KernelTraits::SmemLayoutC{}:%d\n",
    //        BlockShape::kM, BlockShape::kN,
    //        int(size(typename KernelTraits::SmemLayoutC{})));
  }
  // return;
  int step = size<3>(tCsC_r2s); // pipe


#pragma unroll
  for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
    if (i % size<1>(tCgC_s2g) == 0) {
        cute::copy(s2g_scale_tiled_copy_c, tCgScale(_, 0, i / size<1>(tCgC_s2g)), tCrScale(_, 0, i / size<1>(tCgC_s2g)));
        cute::copy(s2g_scale_tiled_copy_c, tCgBias(_, 0, i / size<1>(tCgC_s2g)), tCrBias(_, 0, i / size<1>(tCgC_s2g)));
    }
    __syncthreads();
    // reg -> shm
#pragma unroll
    for (int j = 0; j < step; ++j) {
      cute::copy(r2s_tiled_copy_c, tCrC_r2sx(_, i + j), tCsC_r2s(_, 0, 0, j));

    }
    __syncthreads();
    // cute::copy(s2g_tiled_copy_c, tCsC_s2g, tCrC_s2g);
    // cute::copy(tCsC_s2g, tCrC_s2g);
    // shm -> global
  if (threadIdx.x < KernelTraits::EpiJoinThreadNum) {
  for (int j = 0; j < step; ++j) {
      auto t = make_tensor_like<ElementC>(tCsC_s2g(_, 0, 0, j));

      if (need_pred) {
        // printf("blockIdx.x:%d, blockIdx.y:%d\n", blockIdx.x,  blockIdx.y);
        auto block_out_identity = make_identity_tensor(shape(block_output));
        auto tOrEi = s2g_c_thr_copy_c.partition_D(block_out_identity);
        auto tOrEi_x = group_modes<1, 3>(tOrEi);

        if ((int)get<0>(tOrEi_x(make_coord(0, 0), i + j)) +
                    blockIdx.x * BlockShape::kM <
                p.N * p.P * p.Q &&
            (int)get<1>(tOrEi_x(make_coord(0, 0), i + j)) +
                    blockIdx.y * BlockShape::kN <
                p.K) {

#pragma unroll
          for (int idx = 0; idx < size(t); ++idx) {
            t(idx) = (ElementC)((ElementCompute)tCsC_s2g(make_coord(idx, 0), 0,
                                                         0, j) *
                                    tCrScale(make_coord(idx, 0), 0,
                                             (i + j) / size<1>(tCgC_s2g)) +
                                tCrBias(make_coord(idx, 0), 0,
                                        (i + j) / size<1>(tCgC_s2g)));
          }



        // cutlass::arch::global_store<decltype(t), size(t) * sizeof(ElementC)>(t, tCgC_s2gx(_, i + j).data().get(), threadIdx.x < KernelTraits::EpiJoinThreadNum);
        cutlass::arch::global_store<decltype(t), size(t) * sizeof(ElementC)>(t, tCgC_s2gx(_, i + j).data().get(), true);
        }
      } else {
#pragma unroll

        for (int idx = 0; idx < size(t); ++idx) {
          t(idx) =
              (ElementC)((ElementCompute)tCsC_s2g(make_coord(idx, 0), 0, 0, j) *
                             tCrScale(make_coord(idx, 0), 0,
                                      (i + j) / size<1>(tCgC_s2g)) +
                         tCrBias(make_coord(idx, 0), 0,
                                 (i + j) / size<1>(tCgC_s2g)));
        }
        cutlass::arch::global_store<decltype(t), size(t) * sizeof(ElementC)>(t, tCgC_s2gx(_, i + j).data().get(), true);

      }
    }
  }
  }
#pragma unroll

#ifdef CLOCK_PRINT
  asm volatile("mov.u32 %0, %%clock;" : "=r"(end1) :: "memory");
  // uint32_t time_us1 = (end1 - start1) *  1000.0f/ clockRate;
  if (thread0()) {
  device_time_info[1] = end1 - start1;

  }
  // if (thread0()) {
  //   printf("Kernel epi clock: %d\n",end1 - start1);
  // }
#endif

#endif

}

int main() {


  // int N = 1;
  // int c = 64;
  // int H = 32;
  // int W = 32;
  // int K = 128;
  int N = 3;
  int c = 256;
  int H = 144;
  int W = 240;
  int K = 256;
  int R = 3;
  int S = 3;
  int pad_h = 1, pad_w = 1;
  int stride_h = 1, stride_w = 1;
  int dilation_h = 1, dilation_w = 1;

  int P = (H + pad_h * 2 - (R - 1) * dilation_h - 1) / stride_h + 1;
  int Q = (W + pad_w * 2 - (S - 1) * dilation_w - 1) / stride_w + 1;
  printf("p:%d, q:%d\n", P, Q);
  using KT = KernelTraits<Element, Element, ElementC, ElementCompute, ElementAccumulator, ElementC, BlockShape, WarpShape, kStage, 16, 8>;
  //N, H, W, C, P, Q, K, R, S;
  cutlass::conv::Conv2dProblemSize p{N, H, W, c, K, R, S, P, Q, pad_h, pad_w,
                                     stride_h, stride_w, dilation_h, dilation_w,
                                     cutlass::conv::Mode::kCrossCorrelation};
  // printf("p:%d, q:%d, nt(sizeof(Element)):%d, int(KT::BlockShape::kK):%d, \n", p.P, p.Q,int(sizeof(Element)), int(KT::BlockShape::kK));
  // cutlass::conv::threadblock::Conv2dFpropActivationIteratorOptimizedParams<cutlass::layout::TensorNHWC> test(p, {1,1,1,1}, 1, {1,1}, 1,1,{1,1},{1,1});
  ActivateParams ap(p, int(sizeof(Element)), int(KT::BlockShape::kK));
// return 0;
  FilterParams fp(p, int(sizeof(Element)), int(KT::BlockShape::kK),
                  //与cutlass实现不同，cute的移动是以block为一整体的！ 而cutlass是每个warp去分裂移动
                  KT::Implit_B_G2S_Thr_kStrided, KT::FilterkStrided);
                  // 1482, KT::FilterkStrided);

  MemHelper<Element> helper;
  MemHelper<ElementC> out_helper;
  MemHelper<ElementCompute> compute_helper;
  MemHelper<uint32_t> time_helper;

  // auto A = helper.GetCpuGpuBuffer(p.activation_size(),InitialType::Increment);
  // auto B = helper.GetCpuGpuBuffer(p.filter_size(),InitialType::Increment);
  auto A = helper.GetCpuGpuBuffer(p.activation_size());
  auto B = helper.GetCpuGpuBuffer(p.filter_size());

  auto C = out_helper.GetCpuGpuBuffer(p.output_size());
  // auto hscale = compute_helper.GetCpuGpuBuffer(p.K, InitialType::Increment);
  // auto hbias = compute_helper.GetCpuGpuBuffer(p.K, InitialType::Increment);
  auto hscale = compute_helper.GetCpuGpuBuffer(p.K);
  auto hbias = compute_helper.GetCpuGpuBuffer(p.K);
  // printf("n:%d, p:%d, q:%d, k:%d, output_size:%d\n", p.N, p.P, p.Q,p.K, int(p.output_size()));

  auto gt = out_helper.GetCpuBuffer(p.output_size());

  auto time_info = time_helper.GetCpuGpuBuffer(4);


  dim3 block{32* KT::warp_num, 1, 1};
  dim3 grid{(uint32_t(p.N * p.P * p.Q) + BlockShape::kM - 1) / BlockShape::kM, (uint32_t(p.K) + BlockShape::kN - 1) / BlockShape::kN, 1};
  int smem_s = max(sizeof(typename KT::SharedStorage), sizeof(typename KT::EpilogueSharedStorage));
  // cudaDeviceProp prop;
  // cudaGetDeviceProperties(&prop, 0);
  // int clockRate = prop.clockRate;
  ParamsPack params_pack{p,
                         (const Element *)A.second,
                         (const Element *)B.second,
                         (ElementC *)C.second,
                         ap,
                         fp,
                         (const ElementCompute *)hscale.second,
                         (const ElementCompute *)hbias.second,
                         (uint32_t *)time_info.second};
  // Conv<KT><<<grid, block, smem_s>>>(p, (const Element *)A.second, (const Element *)B.second,
  //                               (ElementC *)C.second, ap, fp, (const ElementCompute *)hscale.second,  (const ElementCompute *)hbias.second, (uint32_t*)time_info.second);
Conv<KT><<<grid, block, smem_s>>>(params_pack);
  out_helper.SyncGpuToCpu(C);
  time_helper.SyncGpuToCpu(time_info);
#ifdef CLOCK_PRINT

  std::cout << "multi_time(clock):" << ((uint32_t*)time_info.first)[0] << std::endl;
  std::cout << "epi_time(clock):" << ((uint32_t*)time_info.first)[1] << std::endl;
#endif
  // std::cout << "kernel over!!!!" <<std::endl;

#ifdef CPU_CHECK
  std::cout << "host_kernel  begin!!!!" <<std::endl;
  host_kernel<Element,Element, ElementC, ElementCompute, ElementC>((ElementCompute*)hscale.first, (Element*)A.first, (Element*)B.first, (ElementCompute*)hbias.first, (ElementC*)gt,
              p.N, p.H, p.W, p.C, p.K, p.R, p.S, p.P, p.Q,
              p.pad_h, p.pad_w, p.stride_h, p.stride_w, p.dilation_h, p.dilation_w);
  // std::cout << "host_kernel over!!!!" <<std::endl;
  // std::cout << "Regression begin!!!!" <<std::endl;
  out_helper.Regression((ElementC*)gt, (ElementC*)C.first, true, true);
  // out_helper.Regression((ElementC*)gt, (ElementC*)C.first, false, true);
  // std::cout << "Regression over!!!!" <<std::endl;

#endif

}

