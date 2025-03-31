#include <iostream>

#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/array_subbyte.h"
#include "cutlass/cutlass.h"
#include "cutlass/matrix_shape.h"
#include <random>
#include <cute/tensor.hpp>
#include "cute/numeric/integral_constant.hpp"
#include <bitset>
using Element = float;
using namespace cute;


template <typename Element, int Alignment, typename ThreadblockShape, typename WarpShape,typename WarpArrangment, typename CopyTile>
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

// CopyTile copytile;
// auto thr_copy = copytile.get_slice(threadIdx.x);


  extern __shared__ Element smem[];
  auto shape_a = make_shape(m, k);
  auto stride_a = make_stride(k, _1{});
  auto tensor_a = make_tensor(make_gmem_ptr((Element*)(device_A)), make_layout(shape_a, stride_a));
  auto tensor_a_block = local_tile(tensor_a, make_shape(_1{}, k), make_coord(blockIdx.x, 0));

  auto shape_b = make_shape(_1{}, k);
  auto stride_b = make_stride(_0{}, _1{});
  auto tensor_b = make_tensor(make_gmem_ptr((Element*)(device_B)), make_layout(shape_b, stride_b));

  auto tensor_a_block_identity = make_identity_tensor(make_shape(size<0>(tensor_a_block), size<1>(tensor_a_block)));
  // if (thread0()) {
  //   print(layout(tensor_a_block_identity));
  // }
  auto tile = Shape<Int<WarpShape::kRow>, Int<WarpShape::kColumn * Alignment>>{};
  auto lp = local_partition(tensor_a_block_identity, tile, threadIdx.x * Alignment);
  auto tApA = make_tensor<bool>(make_shape(size<0>(lp)), Stride<_0>{});

  CopyTile copy_tile;
  auto g2r_thr_copy = copy_tile.get_slice(threadIdx.x);
  auto thr_g_a = g2r_thr_copy.partition_S(tensor_a_block);

  auto thr_g_b = g2r_thr_copy.partition_S(tensor_b);

    // if (thread0()) {
    // if (threadIdx.x == 127) {
        // print(tBpB);
        // printf("\n");
        // printf("%d\n", size(tApA));
        // print(size<0>(tBpB));
        // printf("\n");
        // print(size<1>(tBpB));
        // printf("\n");
        // print(size<2>(tBpB));
        // printf("\n");
        // print(size<3>(tBpB));
        // printf("\n");
        // print(thr_g_a);
        // printf("\n");
        // print(thr_g_a.size());
        // printf("\n");
        // print(size<0>(thr_g_a));
        // printf("\n");
        // print(size<1>(thr_g_a));
        // printf("\n");
        // print(size<2>(thr_g_a));
        // printf("\n");
        // print(rank(thr_g_a));
        // printf("\n");
        // print(depth(thr_g_a));
        // printf("\n");
        // print(shape(thr_g_a));

        // printf("\n");
        // print(thr_g_a(_, _, 0));
        // printf("\n");
        // print(thr_g_a(_, 0, 0));
        // printf("\n");
        // print(thr_g_a(_, _, 0).size());
        // printf("\n");
        // print(size<0>(thr_g_a(_, _, 0)));
        // printf("\n");
        // print(size<1>(thr_g_a(_, _, 0)));
        // printf("\n");
        // print(size<2>(thr_g_a(_, _, 0)));
        // printf("\n");
        // print(rank(thr_g_a(_, _, 0)));
        // printf("\n");
        // print(depth(thr_g_a(_, _, 0)));
        // printf("\n");
        // print(shape(thr_g_a(_, _, 0)));
        // printf("\n");
        // print(thr_g_b);
        // printf("\n");
        // printf("%d\n ",thr_g_a.size());
        // printf("\n");
        // printf("%d\n ",size(thr_g_b));
        // printf("\n");
    // }
  /**
   * thr_g_a(_, _, 0) 和thr_g_a(_, 0, 0)虽然在内存上都只有4个数，但是秩却不一样，
   * thr_g_a(_, _, 0) 的秩为2， thr_g_a(_, 0, 0)的秩为1，copyif针对秩为1的情况会直接忽略掉pred,
   * 所以此处要为make_fragment_like(thr_g_a(_, _, 0))
   *
   *
  */
  auto thr_r_a = make_fragment_like(thr_g_a(_, _, 0));
  auto thr_r_b = make_fragment_like(thr_g_b(_, _, 0));


    Element accum = 0.0f;
    for (int i = 0; i < size<2>(thr_g_a); ++i) {
      clear(thr_r_a);
      clear(thr_r_b);

      CUTE_UNROLL
      for (int j = 0; j < size<0>(tApA); ++j) {
        tApA(j) = get<1>(lp(0, i)) < k;
      }

      cute::copy_if(copy_tile, tApA, thr_g_a(_, _, i), thr_r_a);
      cute::copy_if(copy_tile, tApA, thr_g_b(_, _, i), thr_r_b);

      Element temp_accum = (Element)0;
      CUTE_UNROLL
      for (int k = 0; k < Alignment; ++k) {
        temp_accum += thr_r_a(k) * thr_r_b(k);
      }
      accum += temp_accum;
    }

    //shuffle
    for (int offset = 16; offset > 0; offset /= 2) {
      accum += __shfl_down_sync(0xffffffff, accum, offset);
    }

    int warp_idx = threadIdx.x / 32;
    int lane_idx = threadIdx.x % 32;
    if (lane_idx == 0) {
      ((Element*)smem)[warp_idx] = accum;
    }
    __syncthreads();

    if (threadIdx.x < WarpArrangment::kCount) {
      accum =  smem[threadIdx.x];
    }
    if (warp_idx == 0) {
      for (int offset = WarpArrangment::kCount / 2; offset > 0; offset /= 2) {
        accum += __shfl_down_sync(0xffffffff, accum, offset);
      }

    }

    if (threadIdx.x == 0 ) {
      ((Element*)device_C)[blockIdx.x] = accum;
    }


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
  using ThreadblockShape = cutlass::MatrixShape<1, 512>;

  ///< *
  using WarpShape = cutlass::MatrixShape<1, 128>;

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
  using g2r_copy_atom = Copy_Atom<g2r_copy_traits, Element>;

  // using g2r_copy_tile = decltype(make_tiled_copy(g2r_copy_atom{}, Layout<Shape<_1, Int<kTheadCount>>, Stride<_1>>{}, Layout<Shape<_1, Int<Alignment>>>{}));
  using g2r_copy_tile = decltype(make_tiled_copy(g2r_copy_atom{}, Layout<Shape<_1, Int<kTheadCount>>>{}, Layout<Shape<_1, Int<kAlign>>>{}));
  // print(g2r_copy_tile{});
  int smem = sizeof(Element) * WarpArrangment::kCount;
  f<Element, kAlign, ThreadblockShape, WarpShape, WarpArrangment,  g2r_copy_tile><<<grid, block, smem>>>(
      device_A, device_B, device_C, b, m, k, a, b, batch_stride_A, batch_stride_B, batch_stride_C);
}

int main() {
  // int batch = 1, m = 1000, k = 512, a = 1, b = 0;
  // int batch = 13, m = 1721344, k = 4, a = 1, b = 0;
  // int batch = 1, m = 2048, k = 512, a = 1, b = 0;
  int batch = 1, m = 1, k = 1020, a = 1, b = 0;
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
  }
  for (int i = 0; i < batch * k; ++i) {
    B[i] = distr(eng);
    // B[i] = i;
  }

  Element *device_A, *device_B, *device_C;
  cudaMalloc(&device_A, sizeof(Element) * batch * k * m);
  cudaMemcpy(device_A, A, sizeof(Element) * batch * k * m, cudaMemcpyHostToDevice);
  cudaMalloc(&device_B, sizeof(Element) * batch * k);
  cudaMemcpy(device_B, B, sizeof(Element) * batch * k, cudaMemcpyHostToDevice);
  cudaMalloc(&device_C, sizeof(Element) * batch * m);
  constexpr int Alignment = 4;

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
        Grand_True[i * m + j] += a * A[i * k * m + k * j + p] * B[i * k + p];
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