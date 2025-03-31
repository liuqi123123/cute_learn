
#include "conv_utils.h"
#define CPU_CHECK
using namespace cute;
using BlockShape = cutlass::gemm::GemmShape<96, 128, 64>;
using WarpShape = cutlass::gemm::GemmShape<48, 64, 64>;

using Element = int8_t;
using ElementC = half_t;
using ElementAccumulator = int32_t;
using ElementCompute = float;
static const int Stages= 1;




template <typename ElementA_, typename ElementB_, typename ElementC_,
          typename ElementCompute_, typename ElementAccumulator_,
          typename ElementOutput_, typename BlockShape_, typename WarpShape_, int Stages_, int Alignment_, int AlignC_>
struct KernelTraits {
  using ElementA = ElementA_;
  using Element = ElementA;
  using ElementC = ElementC_;
  using ElementAccumulator = ElementAccumulator_;

  using BlockShape = BlockShape_;
  using WarpShape = WarpShape_;
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
    cutlass::AlignedArray<Element, BlockShape::kM*BlockShape::kK, 16> activate_smem;
    cutlass::AlignedArray<Element, BlockShape::kN*BlockShape::kK, 16> filter_smem;
  };

  using SmemLayoutAtomA = decltype(composition(
    // B M S
    Swizzle<3, 4, 3>{}, make_layout(Shape<Int<BlockShape::kM>, Int<BlockShape::kK>>{},
                              Stride<Int<BlockShape::kK>, _1>{})));
  using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{}, Shape<Int<BlockShape::kM>, Int<BlockShape::kK>>{}));

  using SmemLayoutAtomF = decltype(composition(
    // B M S
    Swizzle<3, 4, 3>{}, make_layout(Shape<Int<BlockShape::kN>, Int<BlockShape::kK>>{},
                              Stride<Int<BlockShape::kK>, _1>{})));
  using SmemLayoutF = decltype(tile_to_shape(SmemLayoutAtomF{}, Shape<Int<BlockShape::kN>, Int<BlockShape::kK>>{}));

  struct EpilogueSharedStorage {
    cutlass::AlignedArray<ElementAccumulator, BlockShape::kM*BlockShape::kN, 16> epilogue_smem;
  };

  using mma_op = SM80_16x8x32_S32S8S8S32_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;
  using MMA = decltype(make_tiled_mma(
      mma_atom{},
      make_layout(
          Shape<Int<warp_m>, Int<warp_n>, Int<1>>{}),
      Tile<Int<1>, Int<1>, Int<1>>{}));

  using activate_s2r_copy_op = SM75_U32x4_LDSM_N;
  using activate_s2r_copy_traits = Copy_Traits<activate_s2r_copy_op>;
  using activate_s2r_copy_atom = Copy_Atom<activate_s2r_copy_traits, Element>;

  using filter_s2r_copy_op = SM75_U32x2_LDSM_N;
  using filter_s2r_copy_traits = Copy_Traits<filter_s2r_copy_op>;
  using filter_s2r_copy_atom = Copy_Atom<filter_s2r_copy_traits, Element>;

  using SmemLayoutAtomEpilogue = decltype(composition(
      // B M S
      Swizzle<2, 2, 8>{},
      make_layout(Shape<Int<BlockShape::kM>, Int<BlockShape::kN>>{},
                  Stride<Int<BlockShape::kN>, _1>{})));
  using SmemLayoutEpilogue = decltype(tile_to_shape(SmemLayoutAtomEpilogue{}, Shape<Int<BlockShape::kM>, Int<BlockShape::kN>>{}));

  static const int e_s2g_copy_thr_contiguous = BlockShape::kN / AlignC;
  static_assert(e_s2g_copy_thr_contiguous <= warp_num * 32);
  using e_s2g_copy_op = UniversalCopy<cutlass::AlignedArray<ElementAccumulator, AlignC>>;
  using e_s2g_copy_traits = Copy_Traits<e_s2g_copy_op>;
  using e_s2g_copy_atom = Copy_Atom<e_s2g_copy_traits, ElementAccumulator>;
  using e_s2g_copy_tile = decltype(make_tiled_copy(
      e_s2g_copy_atom{}, make_layout(Shape<Int<warp_num * 32 / e_s2g_copy_thr_contiguous>,
                                      Int<e_s2g_copy_thr_contiguous>>{}, Stride<Int<e_s2g_copy_thr_contiguous>, _1>{}),
      Layout<Shape<_1, Int<AlignC>>>{}));
};








template<typename KernelTraits>
__global__ void Conv(cutlass::conv::Conv2dProblemSize p, const typename KernelTraits::Element *a, const typename KernelTraits::Element *b,
                                typename KernelTraits::ElementC *c, ActivateParams ap, FilterParams fp, const typename KernelTraits::Element *null) {
#if 1
  extern __shared__ int s0[];
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
    // if (thread0()) {
    //   PRINT(activate_g2s_thr_copy_D);
    // }
    #if 1

    typename KernelTraits::Filter_g2s_Tile_Copy filter_g2s_tile_copy;
    auto filter_g2s_thr_copy = filter_g2s_tile_copy.get_slice(threadIdx.x);
    auto filter_g2s_thr_copy_D = filter_g2s_thr_copy.partition_D(smem_filter);

    typename KernelTraits::MMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto thr_mma_tArA = thr_mma.partition_fragment_A(smem_activate);
    auto thr_mma_tFrF = thr_mma.partition_fragment_B(smem_filter);

    auto activation_s2r_tiled_copy = make_tiled_copy_A(typename KernelTraits::activate_s2r_copy_atom{}, tiled_mma);
    auto activation_s2r_thr_copy = activation_s2r_tiled_copy.get_slice(threadIdx.x);
    auto activation_s2r_tASA = activation_s2r_thr_copy.partition_S(smem_activate);
    auto thr_mma_tArA_view = activation_s2r_thr_copy.retile_D(thr_mma_tArA);

    auto filter_s2r_tiled_copy = make_tiled_copy_B(typename KernelTraits::filter_s2r_copy_atom{}, tiled_mma);
    auto filter_s2r_thr_copy = filter_s2r_tiled_copy.get_slice(threadIdx.x);
    auto filter_s2r_tFSF = filter_s2r_thr_copy.partition_S(smem_filter);
    auto thr_mma_tFrF_view = filter_s2r_thr_copy.retile_D(thr_mma_tFrF);

    auto thr_mma_tCrC = partition_fragment_C(
        tiled_mma,
        Shape<Int<KernelTraits::BlockShape::kM>, Int<KernelTraits::BlockShape::kN>>{});

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


    clear(thr_mma_tCrC);
    int nk = p.R*p.S * cute::ceil_div(p.C, Int<KernelTraits::BlockShape::kK>{});
    ///< main loop
    for (int ik = 0; ik < nk; ++ik) {
      // if (ik > 0) break;
      ///< activate g2s
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < KernelTraits::ActivationkStrided; ++i) {
          cutlass::arch::cp_async_zfill<Align * sizeof(Element)>(
              activate_g2s_thr_copy_D(_, i, 0).data().get(),
              (const void *)activate_pointer[i], activate_thr_valid(i));
      }


    ///< filter g2s

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < KernelTraits::FilterkStrided; ++i) {
          cutlass::arch::cp_async_zfill<Align * sizeof(Element)>(
              filter_g2s_thr_copy_D(_, i, 0).data().get(),
              (const void *)filter_point, filter_thr_valid(i));
        if (i <= KernelTraits::FilterkStrided - 2) {
          filter_point += fp.inc_next_k;

        }
      }
      cp_async_fence();
      cp_async_wait<0>();
      __syncthreads();

      activate_advance();
      filter_advance();

      cute::copy(activation_s2r_tiled_copy, activation_s2r_tASA, thr_mma_tArA_view);
      cute::copy(filter_s2r_tiled_copy, filter_s2r_tFSF, thr_mma_tFrF_view);

      cute::gemm(tiled_mma, thr_mma_tCrC, thr_mma_tArA, thr_mma_tFrF, thr_mma_tCrC);
      if (thread0()) {
        // print_tensor(smem_activate);
        // print_tensor(smem_filter);
        // PRINT_TENSOR(thr_mma_tArA);
        // PRINT_TENSOR(thr_mma_tFrF);
        // PRINT_TENSOR(thr_mma_tCrC);
      }
      __syncthreads();
    }

    using ElementAccumulator = typename KernelTraits::ElementAccumulator;
    ///> epilgue
    typename KernelTraits::EpilogueSharedStorage& e_s = *(reinterpret_cast<typename KernelTraits::EpilogueSharedStorage*>(s0));
    auto e_smem = make_tensor(make_smem_ptr((ElementAccumulator*)(&e_s.epilogue_smem)), typename KernelTraits::SmemLayoutEpilogue{});
    auto tESE = thr_mma.partition_C(e_smem);
    cute::copy(thr_mma_tCrC, tESE);
    __syncthreads();

  typename KernelTraits::e_s2g_copy_tile epi_s2g_copy_tile;
  auto epi_s2g_thr_copy = epi_s2g_copy_tile.get_slice(threadIdx.x);
  auto epi_s2g_tESE = epi_s2g_thr_copy.partition_S(e_smem);
  auto epi_s2g_tEGE = epi_s2g_thr_copy.partition_D(block_output);

  auto convert_epi_s2g_tESE = make_fragment_like<ElementC>(epi_s2g_tESE);
  axpby(1, epi_s2g_tESE, _0{}, convert_epi_s2g_tESE);

  if ((blockIdx.x + 1) * BlockShape::kM > p.N * p. P * p.Q || (blockIdx.y + 1) * BlockShape::kN > p.K) {
    auto block_out_identity = make_identity_tensor(shape(block_output));
    auto tOrEi = epi_s2g_thr_copy.partition_D(block_out_identity);
    auto tOrM = make_tensor<bool>(make_shape(size<1>(epi_s2g_tEGE)));
    auto tOrN = make_tensor<bool>(make_shape(size<2>(epi_s2g_tEGE)));
#pragma unroll
  for (int i = 0; i < size<0>(tOrM); ++i) {
    tOrM(i) = (int)get<0>(tOrEi(make_coord(0, 0), i, 0)) +
                    blockIdx.x * BlockShape::kM <
                p.N * p.P * p.Q;
  }

#pragma unroll
  for (int i = 0; i < size<0>(tOrN); ++i) {
    tOrN(i) = (int)get<1>(tOrEi(make_coord(0, 0), 0, i)) +
                    blockIdx.y * BlockShape::kN <
                p.K;
  }
  CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<1>(epi_s2g_tESE); ++i) {
      for (int j = 0; j < size<2>(epi_s2g_tESE); ++j) {
        if (tOrM(i) && tOrN(j)) {
        cute::copy(convert_epi_s2g_tESE(_, i, j), epi_s2g_tEGE(_, i, j));

        }
      }
    }

  } else {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<1>(epi_s2g_tESE); ++i) {
      for (int j = 0; j < size<2>(epi_s2g_tESE); ++j) {
        // if (thread0()) {
        //   print_tensor(convert_epi_s2g_tESE(_, i, j));
        // }
        cute::copy(convert_epi_s2g_tESE(_, i, j), epi_s2g_tEGE(_, i, j));
      }
    }
  }
  #endif

#endif
}

int main() {
  int N = 1;
  int c = 64;
  int H = 32;
  int W = 32;
  int K = 128;
  // int N = 3;
  // int c = 256;
  // int H = 144;
  // int W = 240;
  // int K = 256;
  int R = 3;
  int S = 3;
  int pad_h = 1, pad_w = 1;
  int stride_h = 1, stride_w = 1;
  int dilation_h = 1, dilation_w = 1;

  int P = (H + pad_h * 2 - (R - 1) * dilation_h - 1) / stride_h + 1;
  int Q = (W + pad_w * 2 - (S - 1) * dilation_w - 1) / stride_w + 1;
  using KT = KernelTraits<Element, Element, ElementC, Element, ElementAccumulator, ElementC, BlockShape, WarpShape, Stages, 16, 4>;
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
  // auto A = helper.GetCpuGpuBuffer(p.activation_size(),InitialType::AllOne);
  // auto B = helper.GetCpuGpuBuffer(p.filter_size(),InitialType::AllOne);
  auto A = helper.GetCpuGpuBuffer(p.activation_size());
  auto B = helper.GetCpuGpuBuffer(p.filter_size());

  auto C = out_helper.GetCpuGpuBuffer(p.output_size());
  auto hscale = compute_helper.GetCpuGpuBuffer(p.K, InitialType::AllOne);
  auto hbias = compute_helper.GetCpuGpuBuffer(p.K, InitialType::AllZero);
  // printf("n:%d, p:%d, q:%d, k:%d, output_size:%d\n", p.N, p.P, p.Q,p.K, int(p.output_size()));
  auto test_null = helper.GetCpuGpuBuffer(KT::Align, InitialType::AllZero);

  auto gt = out_helper.GetCpuBuffer(p.output_size());


  dim3 block{32* KT::warp_num, 1, 1};
  dim3 grid{(uint32_t(p.N * p.P * p.Q) + BlockShape::kM - 1) / BlockShape::kM, (uint32_t(p.K) + BlockShape::kN - 1) / BlockShape::kN, 1};
  int smem_s = max(sizeof(typename KT::SharedStorage), sizeof(typename KT::EpilogueSharedStorage));
  Conv<KT><<<grid, block, smem_s>>>(p, (const Element *)A.second, (const Element *)B.second,
                                (ElementC *)C.second, ap, fp, (const Element *)test_null.second);
  out_helper.SyncGpuToCpu(C);
  // std::cout << "kernel over!!!!" <<std::endl;

#ifdef CPU_CHECK
  // std::cout << "host_kernel  begin!!!!" <<std::endl;
  // host_kernel<Element,Element, ElementC, ElementCompute, ElementC>((ElementCompute*)hscale.first, (Element*)A.first, (Element*)B.first, (ElementCompute*)hbias.first, (ElementC*)gt,
  //             p.N, p.H, p.W, p.C, p.K, p.R, p.S, p.P, p.Q,
  //             p.pad_h, p.pad_w, p.stride_h, p.stride_w, p.dilation_h, p.dilation_w);
  // std::cout << "host_kernel over!!!!" <<std::endl;
  // std::cout << "Regression begin!!!!" <<std::endl;
  // out_helper.Regression((ElementC*)gt, (ElementC*)C.first);
  // out_helper.Regression((ElementC*)gt, (ElementC*)C.first, false, true);
  // std::cout << "Regression over!!!!" <<std::endl;

#endif

}

