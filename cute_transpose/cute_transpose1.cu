#include <iostream>

#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/array_subbyte.h"
#include "cutlass/cutlass.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"

#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/mma_multistage.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include <cute/tensor.hpp>
using Element = float;

#define THREAD 2
#define P(VAR, type)                                                \
  if (threadIdx.y == 8 && threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.x == 1) { \
    printf("" #VAR ": " #type " \n", VAR);                          \
  };

template<typename Element, int Alignment, typename ThreadblockShape>
__global__ void device_kernel(int m, int n, void* a, void* b) {
  using namespace cute;
  __shared__ Element seme[32][32];
  Element* device_a = (Element*)a;
  Element* device_b = (Element*)b;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int device_a_x = bx * ThreadblockShape::kColumn + tx * Alignment;
  int device_a_y = by * ThreadblockShape::kRow + ty;

  auto in_shape = make_shape(m, n);

  auto in_stride = make_stride(n, Int<1>{});

  // device_a += (device_a_x + device_a_y * n);

  auto tensor_in = make_tensor(make_gmem_ptr(device_a), make_layout(in_shape, in_stride));

  auto tensor_in_tile = local_tile(tensor_in, make_shape((Int<ThreadblockShape::kRow>{}), (Int<ThreadblockShape::kColumn>{})), make_coord((by), (bx)));

  auto tensor_in_g = local_tile(tensor_in_tile, make_shape(Int<1>(), Int<Alignment>()), make_coord(ty, tx));

  auto tensor_in_r = make_tensor_like(tensor_in_g);
  // printf(size(tensor_in_g));
  cute::copy(tensor_in_g, tensor_in_r);


  auto out_shape = make_shape(n, m);
  auto out_stride = make_stride(m, 1);
  // device_b += (device_a_y * m + device_a_x);
  auto tensor_out = make_tensor(make_gmem_ptr(device_b), make_layout(out_shape, out_stride));
  auto tensor_out_tile = local_tile(tensor_out, make_shape(Int<ThreadblockShape::kColumn>{}, Int<ThreadblockShape::kRow>{}), make_coord(bx, by));
  auto tensor_out_g = local_tile(tensor_out_tile, make_shape(Int<Alignment>(), Int<1>()), make_coord(tx,ty));

  // auto tensor_out_r = make_tensor_like(tensor_out_g);

  cute::copy(tensor_in_r, tensor_out_g);



  // using Vector = cutlass::Array<Element, Alignment>;
  // cutlass::arch::cp_async<sizeof(Vector)>(&seme[ty][(tx ^ (ty /Alignment)) * Alignment], &device_a[device_a_y * n + device_a_x], true);
  // // cutlass::arch::cp_async<sizeof(Vector)>(&seme[ty][tx* Alignment], &device_a[device_a_y * n + device_a_x], true);
  // cutlass::arch::cp_async_wait<0>();
  // __syncthreads();

  // Element* tans_b = &device_b[bx * ThreadblockShape::kColumn * m + by * ThreadblockShape::kRow];

  // Vector out;

  // CUTLASS_PRAGMA_UNROLL
  // for (int i = 0; i < Alignment; ++i) {
  //   out[i] = seme[tx * Alignment + i][((ty / Alignment) ^ tx) * Alignment + ty % Alignment];
  //   // tans_b[ty * m + tx * Alignment + i] = seme[tx * Alignment + i][((ty / Alignment) ^ tx) * Alignment + ty % Alignment];
  // }
  // cutlass::arch::global_store<Vector, sizeof(Vector)>(out, &tans_b[ty * m + tx * Alignment], true);



}

template<typename Element, int Alignment>
void host_func(int m, int n, Element* host_a, Element* host_b, void* device_a, void* device_b){
  static const int align = Alignment;

  using ThreadblockShape = cutlass::MatrixShape<32, 32>;
  // using WarpShape = cutlass::MatrixShape<1, 32>;
  // using WarpArragement = cutlass::MatrixShape<ThreadblockShape::kRow / WarpShape::kRow,
  //                                             ThreadblockShape::kColumn / WarpShape::kColumn>;
  int griddim_x = (n + ThreadblockShape::kColumn - 1) / ThreadblockShape::kColumn;
  int griddim_y = (m + ThreadblockShape::kRow - 1) / ThreadblockShape::kRow;
  dim3 grid(griddim_x, griddim_y, 1);
  // dim3 block(32/ Alignment, 32 , 1);
  dim3 block(ThreadblockShape::kColumn/ Alignment, ThreadblockShape::kRow , 1);
  int smem = sizeof(Element) * 32 * 32;
  device_kernel<Element, align, ThreadblockShape><<<grid, block, smem, 0>>>(m, n, device_a, device_b);
}

int main() {
  // int m = 32, n = 576*960;
  // int m = 32, n = 32;
  int m = 1024, n = 1024;
  Element* host_a =  new Element[m * n];
  Element* host_b =  new Element[n * m];
  Element* gt =  new Element[n * m];

  std::random_device rd;
  std::default_random_engine eng(rd());
  std::uniform_real_distribution<Element> distr(0.0f, 1.0f);

  for (int i = 0; i < m * n; ++i) {
    host_a[i] = distr(eng);
    // host_a[i] = 1;
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

  static const int align = 1;

  host_func<Element, align>(m, n, host_a, host_b, device_a, device_b);

  cudaMemcpy(host_b, device_b, sizeof(Element) * m * n, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      Element dif = abs(gt[i * m + j] - host_b[i * m + j]);
      if (dif > 0.01) {
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