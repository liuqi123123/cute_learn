#include <iostream>

#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/array_subbyte.h"
#include "cutlass/cutlass.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"
#include <random>

#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/mma_multistage.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include <cute/tensor.hpp>
#include "cute/numeric/integral_constant.hpp"
using Element = float;

#define THREAD 2
#define P(VAR, type)                                                \
  if (threadIdx.y == 8 && threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.x == 1) { \
    printf("" #VAR ": " #type " \n", VAR);                          \
  };

template<typename Element, int Alignment, typename ThreadblockShape, typename G2SCopyA, typename S2GCopyA>
__global__ void device_kernel(int m, int n, void* a, void* b) {
  using namespace cute;
   __shared__  Element  seme[ThreadblockShape::kCount];
  Element* device_a = (Element*)a;
  Element* device_b = (Element*)b;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // int idx = tx + ty * blockDim.x;
  int idx = threadIdx.x;
  int tx = idx % (ThreadblockShape::kColumn / Alignment);
  int ty = idx / (ThreadblockShape::kColumn / Alignment);

  // int device_a_x = bx * ThreadblockShape::kColumn + tx * Alignment;
  // int device_a_y = by * ThreadblockShape::kRow + ty;

  auto in_shape = make_shape(m, n);

  auto in_stride = make_stride(n, _1{});

  // device_a += (device_a_x + device_a_y * n);

  auto tensor_in = make_tensor(make_gmem_ptr((Element*)device_a), make_layout(in_shape, in_stride));

  auto out_shape = make_shape(m, n);
  auto out_stride = make_stride(_1{}, m);
  auto tensor_out = make_tensor(make_gmem_ptr((Element*)device_b), make_layout(out_shape, out_stride));


  auto smem_shape = make_shape(Int<ThreadblockShape::kRow>{}, Int<ThreadblockShape::kColumn>{});
  auto smem_stride = make_stride(Int<ThreadblockShape::kColumn>{}, _1{});

  auto tensor_smem = make_tensor(make_smem_ptr(seme), make_layout(smem_shape, smem_stride));
  // auto tensor_smem = make_tensor(make_smem_ptr(seme), make_layout(Shape<_16, _32>{}, Stride<_32, _1>{}));

  // auto tensor_in_s = local_tile(tensor_smem, make_shape(Int<1>{}, Int<Alignment>{}), make_coord(ty, tx));

  auto tensor_in_tile = local_tile(tensor_in, make_shape((Int<ThreadblockShape::kRow>{}), (Int<ThreadblockShape::kColumn>{})), make_coord((by), (bx)));
  auto tensor_out_tile = local_tile(tensor_out, make_shape((Int<ThreadblockShape::kRow>{}), (Int<ThreadblockShape::kColumn>{})), make_coord((by), (bx)));
  if (idx == 0) {
        // print_tensor(tensor_out);
        // printf("\n");

        // print_tensor(tensor_out_tile);
        // printf("\n");

        // print(thr_g);
        // print(thr_s);
        // printf("\n");
        //  printf("Thread ID: %d hold A value %.1f, %.1f\n",
        // idx, float(thr_g(0)), float(thr_s(0)));
  }

// tensor_in_tile.data();
  auto tensor_in_g = local_tile(tensor_in_tile, make_shape(_1{}, Int<Alignment>()), make_coord(ty, tx));
  // auto tensor_out_g = local_tile(tensor_out_tile, make_shape(Int<Alignment>(), _1{}), make_coord(tx % 4, ty *2 + tx /4 ));
  static const int out_thread_stride = (ThreadblockShape::kRow / Alignment);
  auto tensor_out_g = local_tile(tensor_out_tile, make_shape(Int<Alignment>(), _1{}), make_coord(idx % out_thread_stride, idx/out_thread_stride ));
  // auto tensor_out_g = local_tile(tensor_out_tile, make_shape(Int<Alignment>(), _1{}), make_coord(tx, ty));


  // auto tensor_in_r = make_tensor_like(tensor_in_g);


  //   auto gA = make_tensor(make_gmem_ptr((Element*)device_a),
  //       make_layout(make_shape(Shape<_16>{}, Shape<_4, _8>{}),
  //       make_stride(Stride<_32>{}, Stride<_1, _4>{})));


  // auto sA = make_tensor(make_smem_ptr((Element*)seme),
  //       make_layout(make_shape(Shape<_16>{}, Shape<_4, _8>{}),
  //       make_stride(Stride<_32>{}, Stride<_1, _4>{})));

  // printf(size(tensor_in_g));
  // cute::copy(tensor_in_g, tensor_in_r);
  G2SCopyA g2s_tiled_copy;
  auto g2s_thr_tiled_copy = g2s_tiled_copy.get_thread_slice(idx);
  auto thr_g = g2s_thr_tiled_copy.partition_S(tensor_in_tile);
  // auto thr_r = make_fragment_like(thr_g);
  auto thr_s = g2s_thr_tiled_copy.partition_D(tensor_smem);
  // if (idx == 1) {
  //   print(thr_g);
  //       printf("\n");

  //   print(thr_s);
  //       printf("\n");

  //   print(thr_g(_, 0, 0));
  //       printf("\n");

  //   print(thr_s(_, 0, 0));
  //       printf("\n");
  // }
  cute::copy(g2s_tiled_copy, thr_g, thr_s);
  cp_async_wait<0>();
  __syncthreads();

  S2GCopyA s2g_tiled_copy;

  auto s2g_thr_copy = s2g_tiled_copy.get_thread_slice(idx);
  auto thr_s2g_s = s2g_thr_copy.partition_S(tensor_smem);

  cute::copy(thr_s2g_s, tensor_out_g);



  // cute::copy(tensor_in_g, tensor_in_r);
  // cute::copy(tensor_in_r, tensor_in_s);

  // if (thread0()) {
  if (idx == 0) {
        // print_tensor(thr_s);
        // printf("\n");
        // print_tensor(thr_s2g_s);
        // printf("\n");
        // print_tensor(tensor_out_g);
        // printf("\n");
        // print_tensor(tensor_out_tile);
        // printf("\n");
        // print(tensor_out_tile);
        // printf("\n");
        // print(tensor_out_g);
        // printf("\n");
        // print_tensor(tensor_out_g);
        // printf("\n");
        // print_tensor(tensor_out_tile);
        // printf("\n");

        // print(thr_g);
        // print(thr_s);
        // printf("\n");
        //  printf("Thread ID: %d hold A value %.1f, %.1f\n",
        // idx, float(thr_g(0)), float(thr_s(0)));
  }
}

template<typename Element, int Alignment>
void host_func(int m, int n, Element* host_a, Element* host_b, void* device_a, void* device_b){
  static const int align = Alignment;

  using ThreadblockShape = cutlass::MatrixShape<16, 32>;
  // using WarpShape = cutlass::MatrixShape<1, 32>;
  // using WarpArragement = cutlass::MatrixShape<ThreadblockShape::kRow / WarpShape::kRow,
  //                                             ThreadblockShape::kColumn / WarpShape::kColumn>;
  int griddim_x = (n + ThreadblockShape::kColumn - 1) / ThreadblockShape::kColumn;
  int griddim_y = (m + ThreadblockShape::kRow - 1) / ThreadblockShape::kRow;
  dim3 grid(griddim_x, griddim_y, 1);
  // dim3 block(32/ Alignment, 32 , 1);
  static const int thread_num = ThreadblockShape::kColumn/ Alignment * ThreadblockShape::kRow;
  dim3 block(thread_num, 1 , 1);
  // int smem = sizeof(Element) * 16 * 32;
  using namespace cute;
  using T = Element;
  static const int tiled_kcolumn = ThreadblockShape::kColumn / align;
  printf("tiled_kcolumn: %d\n", tiled_kcolumn);
  printf("align: %d\n", align);

  using g2s_copy_op = SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, Element>;

  using G2SCopyA = decltype(make_tiled_copy(g2s_copy_atom{},
                      make_layout(make_shape(Int<ThreadblockShape::kRow>{}, Int<tiled_kcolumn>{}),
                                  make_stride(Int<tiled_kcolumn>{}, Int<1>{})),
                      make_layout(make_shape(Int<1>{}, Int<align>{}))));  // Copy Tile: (16, 32)
  // using AccessType = cutlass::AlignedArray<Element, 4>;
  // using Atom = Copy_Atom<UniversalCopy<AccessType>, Element>;
  // using G2SCopyA = decltype(make_tiled_copy(Atom{},
  //                     make_layout(make_shape(Int<ThreadblockShape::kRow>{}, Int<tiled_kcolumn>{}),
  //                                 make_stride(Int<tiled_kcolumn>{}, Int<1>{})),
  //                     make_layout(make_shape(Int<1>{}, Int<align>{}))));  // Copy Tile: (16, 32)

  static const int s2g_copy_row = Int<ThreadblockShape::kRow>{} / align;
  static const int s2g_copy_col = thread_num / s2g_copy_row;

  printf("s2g_copy_row: %d\n", s2g_copy_row);
  printf("s2g_copy_col: %d\n", s2g_copy_col);

  using s2g_copy_op = UniversalCopy<Element>;
  using s2g_copy_traits = Copy_Traits<s2g_copy_op>;
  using s2g_copy_atom = Copy_Atom<s2g_copy_traits, Element>;
  using S2GCopyA = decltype(make_tiled_copy(
      s2g_copy_atom{},
      make_layout(
          make_shape(Int<s2g_copy_row>{}, Int<s2g_copy_col>{}),
          make_stride(_1{}, Int<s2g_copy_row>{})),
      make_layout(make_shape(Int<align>{}, Int<1>{}))));

  device_kernel<Element, align, ThreadblockShape, G2SCopyA, S2GCopyA><<<grid, block, 0, 0>>>(m, n, device_a, device_b);
}

int main() {
  // int m = 32, n = 576*960;
  int m = 1024, n = 1024;
  // int m = 32, n = 32;
  // int m = 16, n = 32;
  Element* host_a =  new Element[m * n];
  Element* host_b =  new Element[n * m];
  Element* gt =  new Element[n * m];

  std::random_device rd;
  std::default_random_engine eng(rd());
  std::uniform_real_distribution<float> distr(0.0f, 1.0f);

  for (int i = 0; i < m * n; ++i) {
    // host_a[i] = (Element)distr(eng);
    host_a[i] = i;
    host_b[i] = i;
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      gt[i * m + j] = host_a[j * n + i];
    }
  }

  void* device_a, * device_b;
  cudaMalloc(&device_a, sizeof(Element) * m * n);
  cudaMalloc(&device_b, sizeof(Element) * m * n);

  cudaMemcpy(device_a, host_a, sizeof(Element) * m * n, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, host_b, sizeof(Element) * m * n, cudaMemcpyHostToDevice);

  static const int align = 4;

  host_func<Element, align>(m, n, host_a, host_b, device_a, device_b);

  cudaMemcpy(host_b, device_b, sizeof(Element) * m * n, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      Element dif = abs((float)gt[i * m + j] - (float)host_b[i * m + j]);
      if ((float)dif > 0.01) {
      // if (1) {
        std::cout << "Error : (" << i << "," << j << "): gpu:(" << host_b[i * m + j] << ") cpu:("
                  << gt[i * m + j] << "), dif:" << dif << std::endl;
        return 0;
      }
    }
  }
  delete[] host_a;
  delete[] host_b;
  cudaFree(device_a);
  cudaFree(device_b);

}