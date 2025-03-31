#include <iostream>

#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/array_subbyte.h"
#include "cutlass/cutlass.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_conversion.h"

#include <random>

#include <cute/tensor.hpp>
#include "cute/numeric/integral_constant.hpp"
#include <bitset>
#include "include/cuda_helper.h"
using namespace cute;

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

using Element = cutlass::half_t;
using ElementAccumulator = cutlass::half_t;
using ElementCompute = float;

template<typename Element_,typename ElementAccumulator_, typename ElementCompute_, int Align_, int WarpNum_, int WarpStrideIterations_, int BlockThreadArrangementkContiguous_>
struct KernelTraits{
  static const int WarpNum = WarpNum_;
  static const int ThreadNum = WarpNum * 32;
  static const int WarpStrideIterations = WarpStrideIterations_;
  static const int BlockThreadArrangementkContiguous = BlockThreadArrangementkContiguous_;
  static const int BlockThreadArrangementStride = ThreadNum / BlockThreadArrangementkContiguous;
  static const int Align = Align_;
  using TileShapeA = Shape<Int<BlockThreadArrangementStride * WarpStrideIterations>, Int<Align * BlockThreadArrangementkContiguous>>;
  using TileShapeB = Shape<_1, Int<Align * BlockThreadArrangementkContiguous>>;
  using TileShapeC = Shape<Int<BlockThreadArrangementStride * WarpStrideIterations>, _1>;
  using Element = Element_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;

  static_assert(ThreadNum >= BlockThreadArrangementkContiguous && ThreadNum % BlockThreadArrangementkContiguous == 0);

  using TiledCopyA = decltype(make_tiled_copy(
      Copy_Atom<UniversalCopy<cutlass::AlignedArray<Element, Align>>,
                Element>{},
      Layout<Shape<Int<BlockThreadArrangementStride>,
                   Int<BlockThreadArrangementkContiguous>>,
             Stride<Int<BlockThreadArrangementkContiguous>, _1>>{},
      Layout<Shape<_1, Int<Align>>>{}));
  using TiledCopyB = TiledCopyA;

  using TiledCopyC = decltype(make_tiled_copy(Copy_Atom<UniversalCopy<Element>, Element>{},
    Layout<Shape<Int<BlockThreadArrangementStride>,
                   Int<BlockThreadArrangementkContiguous>>,Stride<Int<BlockThreadArrangementkContiguous>, _1>>{},
      Layout<Shape<_1, _1>>{}));
};



struct Params {
  const Element* A;
  const Element* B;
  Element* C;
  int M;
  int N;
  int K;

  __host__ __device__
  Params(){}

  __host__ __device__
  Params(Element* A_, Element* B_, Element* C_, int M_, int N_, int K_): A(A_), B(B_), C(C_), M(M_),N(N_),K(K_){}
};

template<typename KernelTraits>
__global__ void gemv(Params p) {

  __shared__ typename KernelTraits::ElementAccumulator  smem[KernelTraits::WarpNum * KernelTraits::WarpStrideIterations];

  Tensor A = make_tensor(make_gmem_ptr(p.A), make_layout(make_shape(p.M, p.K), make_stride(p.K, _1{})));
  Tensor B = make_tensor(make_gmem_ptr(p.B), make_layout(make_shape(_1{}, p.K), make_stride(_0{}, _1{})));
  Tensor C = make_tensor(make_gmem_ptr(p.C), make_layout(make_shape(p.M, _1{}), make_stride(_1{}, _0{})));


  Tensor tiled_A = local_tile(A, typename KernelTraits::TileShapeA{}, make_coord(blockIdx.x, _));
  Tensor tiled_B = local_tile(B, typename KernelTraits::TileShapeB{}, make_coord(_0{}, _));
  Tensor tiled_C = local_tile(C, typename KernelTraits::TileShapeC{}, make_coord(blockIdx.x, _0{}));

  typename KernelTraits::TiledCopyA tiled_copy_a;
  auto thr_copy_a = tiled_copy_a.get_slice(threadIdx.x);
  auto tCgA = thr_copy_a.partition_S(tiled_A);
  auto tCrA = make_fragment_like(tCgA(_, _, 0, 0));

  typename KernelTraits::TiledCopyB tiled_copy_b;
  auto thr_copy_b = tiled_copy_b.get_slice(threadIdx.x);
  auto tCgB = thr_copy_b.partition_S(tiled_B);
  auto tCrB = make_fragment_like(tCgB(_, _, 0, 0));

  typename KernelTraits::TiledCopyC tiled_copy_c;
  auto thr_copy_c = tiled_copy_c.get_slice(threadIdx.x);
  auto tCgC = thr_copy_c.partition_D(tiled_C);


  if (thread0()) {
    // PRINT(tCgA);
    // PRINT(tCrA);
    // PRINT(tiled_A);
    // PRINT(A);

    // PRINT(tCgB);
    // PRINT(tCrB);
    // PRINT(tiled_C);
    // PRINT(tCgC);

  }

  auto accumulator = make_tensor<typename KernelTraits::ElementAccumulator>(
      make_shape(Int<KernelTraits::WarpStrideIterations>{}));

  clear(accumulator);

  int NK = cute::ceil_div(p.K, KernelTraits::Align * KernelTraits::BlockThreadArrangementkContiguous);

  ///< main loop

  for (int ik = 0; ik < NK; ++ik) {
    cute::copy(tiled_copy_a, tCgA(_, _, 0, ik), tCrA);
    cute::copy(tiled_copy_b, tCgB(_, _, 0, ik), tCrB);
    #pragma unroll
    for (int s = 0 ; s < KernelTraits::WarpStrideIterations; ++s) {
    #pragma unroll
      for (int i = 0; i < KernelTraits::Align; ++i) {
        accumulator(s) += (ElementAccumulator)tCrA(make_coord(i, 0), s) * (ElementAccumulator)tCrB(make_coord(i, 0), 0);
        // float a = (ElementAccumulator)tCgA(make_coord(i, 0), s, 0, ik);
        // float b =  (ElementAccumulator)tCgB(make_coord(i, 0), 0, 0, ik);
        if (thread0()) {
          // PRINT_TENSOR(accumulator);
          // printf("a:%f, b:%f\n", a, b);
        }
      }
    }
  }
  // if (thread0()) {
  //   PRINT_TENSOR(accumulator);

  // }

  // Tensor compute_accumulator = convert_type<typename KernelTraits::ElementCompute>(accumulator);
Tensor compute_accumulator = make_fragment_like<typename KernelTraits::ElementCompute>(accumulator);
#pragma unroll
for (int i = 0;i < KernelTraits::WarpStrideIterations; ++i) {
  compute_accumulator(i) = accumulator(i);
}

  // if (thread0()) {
  //   PRINT_TENSOR(accumulator);
  //   PRINT_TENSOR(compute_accumulator);
  // }
  int warp_idx = threadIdx.x / 32;
  int lane_idx = threadIdx.x % 32;
  static const int contigous_warp_num = KernelTraits::BlockThreadArrangementkContiguous / 32;
  static const int stride_warp_num = KernelTraits::WarpNum / contigous_warp_num;
  #pragma unroll
for (int s = 0; s < KernelTraits::WarpStrideIterations; ++s) {
    #pragma unroll
  for (int i = 32 >> 1; i > 0; i >>= 1) {
      compute_accumulator(s) += __shfl_down_sync(0xffffffff, compute_accumulator(s), i);
  }

  if (lane_idx == 0) {
    smem[s * KernelTraits::WarpNum + warp_idx] = compute_accumulator(s);
    // if (thread0()) {
      // printf("[s * KernelTraits::WarpNum + warp_idx]:%d\n", s * KernelTraits::WarpNum + warp_idx);
      // printf("[s :%d\n", s );
      // printf("[KernelTraits::WarpNum %d\n",  KernelTraits::WarpNum );
      // printf("[warp_idx]:%d\n", warp_idx);
    // }
  }
}
__syncthreads();
// return;
if (threadIdx.x < stride_warp_num * KernelTraits::WarpStrideIterations) {
  ElementAccumulator contigous_warp_sum = (ElementAccumulator)0.0f;
  #pragma unroll
  for (int i = 0; i < contigous_warp_num; ++i) {
    contigous_warp_sum += smem[threadIdx.x * contigous_warp_num + i];
  }
  // if (thread0()) {
  //   printf("contigous_warp_sum:%f\n", contigous_warp_sum);
  // }
      //   if (threadIdx.x == 0 && blockIdx.x == 1) {
      //     printf("[contigous_warp_sum]:%f\n", contigous_warp_sum);
      //     printf("blockIdx.x * KernelTraits::BlockThreadArrangementStride * KernelTraits::WarpNum + threadIdx.x:%d\n", blockIdx.x * KernelTraits::BlockThreadArrangementStride * KernelTraits::WarpNum + threadIdx.x);
      //     printf("blockIdx.x:%d\n", blockIdx.x );
      //     printf("KernelTraits::BlockThreadArrangementStride:%d\n", KernelTraits::BlockThreadArrangementStride);
      //     printf(" KernelTraits::WarpNum:%d\n",  KernelTraits::WarpNum);
      //     printf("threadIdx.x:%d\n", threadIdx.x );
      // }
  C[blockIdx.x * KernelTraits::BlockThreadArrangementStride * KernelTraits::WarpStrideIterations + threadIdx.x] = (Element)contigous_warp_sum;
}



  // Tensor output = convert_type<typename KernelTraits::Element>(compute_accumulator);
//   Tensor output = make_fragment_like<typename KernelTraits::Element>(compute_accumulator);
// #pragma unroll
// for (int i = 0;i < KernelTraits::WarpStrideIterations; ++i) {
//   output(i) = compute_accumulator(i);
// }


//   // if (thread0())  {
//   //   PRINT_TENSOR(output);
//   // }
//   if (threadIdx.x % KernelTraits::BlockThreadArrangementkContiguous == 0) {
//     // cute::copy(tiled_copy_c, output, tCgC);
//     cute::copy(output, tCgC(make_coord(0,0),_,0));
//     // PRINT(tCgC);
//   }
}

int main() {
  MemHelper<Element> mem_helper;
  int M = 1024*18;
  int N = 1;
  int K = 1024*4;
  // int M = 512;
  // int N = 1;
  // int K = 512;
  // auto A = mem_helper.GetCpuGpuBuffer(M * K, InitialType::AllOne);
  // auto B = mem_helper.GetCpuGpuBuffer(K, InitialType::AllOne);
  auto A = mem_helper.GetCpuGpuBuffer(M * K);
  auto B = mem_helper.GetCpuGpuBuffer(K);
  auto C = mem_helper.GetCpuGpuBuffer(M);
  auto gt = mem_helper.GetCpuBuffer(M);

// template<typename Element_,typename ElementAccumulator_, typename ElementCompute_, int Align_, int WarpNum_, int WarpStrideIterations_, int BlockThreadArrangementkContiguous_>
  using KT = KernelTraits<Element, ElementAccumulator,ElementCompute, 8, 8,2,128>;
  Params p{(Element*)A.second, (Element*)B.second, (Element*)C.second, M, N, K};

  unsigned int threadnum = KT::WarpNum * 32;
  unsigned int blocknum = cute::ceil_div(M,  KT::BlockThreadArrangementStride * KT::WarpStrideIterations);

  dim3 grid{blocknum};
  dim3 block{threadnum};
for (int i = 0; i < 50; ++i) {
  gemv<KT><<<grid, block>>>(p);
}

  mem_helper.SyncGpuToCpu(C);

  Element* cpu_A = reinterpret_cast<Element*>(A.first);
  Element* cpu_B = reinterpret_cast<Element*>(B.first);

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      ElementAccumulator sum = (ElementAccumulator)0;
      for (int k = 0; k < K; ++k) {
        sum += cpu_A[i * K + k] * cpu_B[k + j * K];
      }
      ((Element*)gt)[i] = sum;
    }
  }
  mem_helper.Regression((Element*)gt, (Element*)C.first);




}



