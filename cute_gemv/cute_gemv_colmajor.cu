#include <iostream>

#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/array_subbyte.h"
#include "cutlass/cutlass.h"
#include "cutlass/matrix_shape.h"
// #include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"
// #include "venom/host/data_initializer.h"
#include <random>
// #include "cutlass/gemm/threadblock/default_mma_core.h"
// #include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
// #include "cutlass/gemm/threadblock/mma_multistage.h"
// #include "cutlass/gemm/threadblock/threadblock_swizzle.h"
// #include "cutlass/transform/pitch_linear_thread_map.h"
#include <cute/tensor.hpp>
#include "cute/numeric/integral_constant.hpp"
using Element = float;
using namespace cute;


#define  PRINT(STR) \
        do { \
          printf("%s:\n", #STR); \
          print(STR); \
          printf("\n"); \
        } while (0)


template <typename Element, int Alignment, typename ThreadblockShape, typename WarpShape,typename WarpArrangment, typename CopyTile, typename CopyTileB>
__global__ void f(void* device_A,
                  void* device_B,
                  void* device_C,
                  int batch,
                  int m,
                  int k,
                  int a,
                  int b,
                  int batch_stride_A,
                  int batch_stride_B,
                  int batch_stride_C) {

  extern __shared__ Element smem[];

  auto ga = make_tensor(make_gmem_ptr((Element*)(device_A)), make_layout(make_shape(m, k), make_stride(_1{}, m)));
  auto gb = make_tensor(make_gmem_ptr((Element*)(device_B)), make_layout(make_shape(_1{}, k)));
  auto gb_identity = make_identity_tensor(make_shape(size<0>(gb), size<1>(gb)));
  auto gb_identity_lp = local_partition(gb_identity, Shape<_1, Int<32 * WarpArrangment::kCount>>{}, threadIdx.x);
  // if (thread0()) {
  //   PRINT(gb_identity_lp(1));
  //   PRINT(layout(gb_identity_lp));
  // }
  auto tBpB = make_tensor<bool>(make_shape(size<0>(gb_identity_lp)));



  auto a_tile = make_shape(Int<Alignment>{}, k);
  auto local_ga = local_tile(ga, a_tile, make_coord(blockIdx.x, _0{}));

  // auto gc = make_tensor(make_gmem_ptr((Element*)(device_C)), make_layout(_1{}, m));
  // auto target_c = local_tile(gc, Shape<_1, _4>{}, make_coord(0, blockIdx.x));

  CopyTile copy_tile;
  auto g2r_thr_copy = copy_tile.get_slice(threadIdx.x);
  auto thr_g_a = g2r_thr_copy.partition_S(local_ga);


  CopyTileB copy_tile_b;
  auto g2r_thr_copy_b = copy_tile_b.get_slice(threadIdx.x);
  auto thr_g_b = g2r_thr_copy_b.partition_S(gb);

  auto thr_r_a = make_fragment_like(thr_g_a(_, _, 0));
  auto thr_r_b = make_fragment_like(thr_g_b(_, _, 0));
  // decltype(thr_r_a) accum;
  // auto accum_element = recast<Element>(accum);
  auto accum_element = make_tensor<Element>(Shape< Int<1>,Int<Alignment>>{});
  if (thread0()) {
    // print(thr_g_a);
    // printf("\n");
    // print(thr_g_b);
    // printf("\n");

    // print(size<0>(thr_r_a));
    // printf("\n");
    // print(size<0>(accum_element));

    // printf("\n");
  }


  for (int i = 0; i < size<2>(thr_g_a); ++i) {
    clear(thr_r_a);
    clear(thr_r_b);

    CUTE_UNROLL
    for (int p = 0; p < size<0>(tBpB); ++p) {
      tBpB(p) = get<1>(gb_identity_lp(0, i)) < k;
    }
    cute::copy_if(copy_tile, tBpB,  thr_g_a(_, _, i), thr_r_a);
    cute::copy_if(copy_tile_b, tBpB,thr_g_b(_, _, i), thr_r_b);


    CUTE_UNROLL
    for (int j = 0; j < Alignment; ++j) {
      accum_element[j] += thr_r_a(j) * thr_r_b(0);
    }
    //   if (thread0()) {
    // print(accum_element);
    // printf("\n");
    //   }
  }
// #if 0
  // shuffle
  CUTE_UNROLL
  for (int i = 0; i < Alignment; ++i) {
    for (int offset = 16; offset > 0; offset /= 2) {
      accum_element[i] += __shfl_down_sync(0xffffffff, accum_element[i], offset);
    }
  }

  int warp_idx = threadIdx.x / 32;
  int lane_idx = threadIdx.x % 32;

  auto smem_accum = make_tensor(make_smem_ptr((Element*)(smem)), make_layout(Shape<_1, Int<WarpArrangment::kCount * Alignment>>{}));

  auto local_smem = local_tile(smem_accum, Shape<_1, Int<Alignment>>{}, make_coord(_0{}, warp_idx));

  if (lane_idx == 0) {
    cute::copy(accum_element, local_smem);
    // if (thread0()) {
    //   printf("%f\n", accum[0]);
    // }
    __syncthreads();
  }


  Element ele = (Element)0.0f;

  if (threadIdx.x < WarpArrangment::kCount * Alignment) {
    ele = ((Element*)smem)[threadIdx.x];
  }
  //warp_idx must be 0, so WarpArrangment::kCount * Alignment must <= 32
  if (warp_idx == 0) {
    for (int offset = WarpArrangment::kCount * Alignment / 2; offset >= Alignment; offset /= 2) {
      ele += __shfl_down_sync(0xffffffff, ele, offset);
    }
  }
    if (threadIdx.x < Alignment) {
      ((Element*)device_C)[blockIdx.x * Alignment + threadIdx.x] = ele;
    }
// #endif


}



template <typename Element, int Alignment>
void func(void* device_A,
          void* device_B,
          void* device_C,
          int batch,
          int m,
          int k,
          int a,
          int b,
          int batch_stride_A,
          int batch_stride_B,
          int batch_stride_C) {
  static const int kAlign = Alignment;
  ///< *
  using ThreadblockShape = cutlass::MatrixShape<kAlign, 128>;

  ///< *
  using WarpShape = cutlass::MatrixShape<kAlign, 32>;

  ///< *
  using WarpArrangment = cutlass::PitchLinearShape<ThreadblockShape::kRow / WarpShape::kRow,
                                              ThreadblockShape::kColumn / WarpShape::kColumn>;

  static const int kTheadCount = WarpArrangment::kCount * 32;



  int griddim_x = (m + ThreadblockShape::kRow - 1) / ThreadblockShape::kRow;

  dim3 grid(griddim_x, batch, 1);
  dim3 block(kTheadCount, 1, 1);

  using T = cutlass::AlignedArray<Element, kAlign>;
  using g2r_copy_op = UniversalCopy<T>;
  using g2r_copy_traits = Copy_Traits<g2r_copy_op>;
  using g2r_copy_atom = Copy_Atom<g2r_copy_traits, T>;

  // using g2r_copy_tile = decltype(make_tiled_copy(g2r_copy_atom{}, Layout<Shape<_1, Int<kTheadCount>>, Stride<_1>>{}, Layout<Shape<_1, Int<Alignment>>>{}));
  using g2r_copy_tile = decltype(make_tiled_copy(g2r_copy_atom{}, Layout<Shape<_1, Int<kTheadCount>>>{}, Layout<Shape<Int<kAlign>, _1>>{}));

  // using Tb = cutlass::AlignedArray<Element, 1>;
  using g2r_copy_op_b = UniversalCopy<Element>;
  using g2r_copy_traits_b = Copy_Traits<g2r_copy_op_b>;
  using g2r_copy_atom_b = Copy_Atom<g2r_copy_traits_b, Element>;

  using g2r_copy_tile_b = decltype(make_tiled_copy(g2r_copy_atom_b{}, Layout<Shape<_1, Int<kTheadCount>>>{}, Layout<Shape<_1, _1>>{}));

  // print(g2r_copy_tile{});
  int smem = sizeof(Element) * WarpArrangment::kCount * kAlign;
  f<Element, kAlign, ThreadblockShape, WarpShape, WarpArrangment,  g2r_copy_tile, g2r_copy_tile_b><<<grid, block, smem>>>(
      device_A, device_B, device_C, batch, m, k, a, b, batch_stride_A, batch_stride_B, batch_stride_C);
}

int main() {
  // int batch = 1, m = 1000, k = 512, a = 1, b = 0;
  // int batch = 13, m = 1721344, k = 4, a = 1, b = 0;
  // int batch = 1, m = 2048, k = 512, a = 1, b = 0;
  int batch = 1, m = 1026, k = 1023, a = 1, b = 0;
  // int batch = 1, m = 1, k = 512, a = 1, b = 0;
  // int batch = 1, m = 5, k = 5, a = 1, b = 0;
  int batch_stride_A = m * k, batch_stride_B = k, batch_stride_C = m;
  Element* A = new Element[batch * k * m];
  Element* B = new Element[batch * k];
  Element* C = new Element[batch * m];
  Element* Grand_True = new Element[batch * m]{0};



  std::random_device rd;
  std::default_random_engine eng(rd());
  std::uniform_real_distribution<float> distr(0.0f, 1.0f);

  for (int i = 0; i < batch * k * m; ++i) {
    A[i] = distr(eng);
    // A[i] = i;
    // A[i] = 2;
  }
  for (int i = 0; i < batch * k; ++i) {
    B[i] = distr(eng);
    // B[i] = i;
    // B[i] = 2;
  }

  Element *device_A, *device_B, *device_C;
  cudaMalloc(&device_A, sizeof(Element) * batch * k * m);
  cudaMemcpy(device_A, A, sizeof(Element) * batch * k * m, cudaMemcpyHostToDevice);
  cudaMalloc(&device_B, sizeof(Element) * batch * k);
  cudaMemcpy(device_B, B, sizeof(Element) * batch * k, cudaMemcpyHostToDevice);
  cudaMalloc(&device_C, sizeof(Element) * batch * m);
  constexpr int Alignment = 2;

  func<Element, Alignment>(device_A,
                           device_B,
                           device_C,
                           batch,
                           m,
                           k,
                           a,
                           b,
                           batch_stride_A,
                           batch_stride_B,
                           batch_stride_C);

  cudaMemcpy(C, device_C, sizeof(Element) * m * batch, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < m; ++j) {
      for (int p = 0; p < k; ++p) {
        Grand_True[i  * m + j] += a * A[i * k * m + j + p * m] * B[i * k + p];
      }
    }
  }

  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < m; ++j) {
      float diff = fabs(float(C[i * m + j]) - float(Grand_True[i * m + j]));
      if (diff > float(0.01)) {
        std::cout << "batch: " << i << ", m: " << j << " Gpu: " << float(C[i * m + j])
                  << " cpu: " << float(Grand_True[i * m + j]) << " diff:" << diff << std::endl;
        break;
      }
    }
  }


  delete[] A;
  delete[] B;
  delete[] C;
  delete[] Grand_True;
  return 0;
}