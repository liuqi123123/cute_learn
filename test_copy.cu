
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "include/cuda_helper.h"
#include "cutlass/fast_math.h"
#include "cute/numeric/integral_constant.hpp"
#include "cute/pointer.hpp"
#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/array_subbyte.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include <cmath>
#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <iostream>
#include "cutlass/half.h"

using namespace cute;

using ElementCompute = float;
  static const int c =  32 / 4;
  using e_scale_s2g_copy_op = UniversalCopy<cutlass::AlignedArray<int8_t, 4>>;
  using e_scale_s2g_copy_traits = Copy_Traits<e_scale_s2g_copy_op>;
  using e_scale_s2g_copy_atom = Copy_Atom<e_scale_s2g_copy_traits, ElementCompute>;
  using e_scale_s2g_copy_tile = decltype(make_tiled_copy(
      e_scale_s2g_copy_atom{}, make_layout(Shape<Int<64 / c>,
                                      Int<c>>{}, Stride<Int<c>, _1>{}),
      Layout<Shape<_1, Int<4>>>{}));
template<typename TensorA, typename TensorB>
__global__ void test(TensorA tensor_a, TensorB tensor_b) {
  // auto tensor_a = make_tensor(make_gmem_ptr(A), make_shape(_32{}, _32{}), make_stride(_32{}, _1{}));
  // auto tensor_b = make_tensor(make_gmem_ptr(B), make_shape(_32{}, _32{}), make_stride(_32{}, _1{}));

  e_scale_s2g_copy_tile copy_tile;
  auto thr_copy = copy_tile.get_slice(threadIdx.x);
  auto s = thr_copy.partition_S(tensor_a);
  auto d = thr_copy.partition_D(tensor_b);

  if (threadIdx.x < 64)
  cute::copy(copy_tile, s, d);
  // if (thread0()) {
  //   PRINT(s);
  //   PRINT(d);
  //   //事实证明 拷贝的两个float4的地址是连续的！！！！
  //   printf("addr0:%p\n", s(make_coord(_, 0), 0, 0).data().get());
  //   printf("addr1:%p\n", s(make_coord(_, 1), 0, 0).data().get());
  // }
  if (threadIdx.x == 0) {
    PRINT(s);
    PRINT(d);
    //事实证明 拷贝的两个float4的地址是连续的！！！！
    // printf("addr0:%p\n", s(make_coord(_, 0), 0, 0).data().get());
    // printf("addr1:%p\n", s(make_coord(_, 1), 0, 0).data().get());
  }
  // if (threadIdx.x == 64) {
  //   PRINT(s);
  //   PRINT(d);
  //   //事实证明 拷贝的两个float4的地址是连续的！！！！
  //   // printf("addr0:%p\n", s(make_coord(_, 0), 0, 0).data().get());
  //   // printf("addr1:%p\n", s(make_coord(_, 1), 0, 0).data().get());
  // }

}

int main() {
  MemHelper<float> helper;
  auto A = helper.GetCpuGpuBuffer(32*32, InitialType::Increment);
  auto B = helper.GetCpuGpuBuffer(32*32);
  auto tensor_a =  make_tensor(make_gmem_ptr((float*)A.second), make_shape(_32{}, _32{}), make_stride(_32{}, _1{}));
  auto tensor_b = make_tensor(make_gmem_ptr((float*)B.second), make_shape(_32{}, _32{}), make_stride(_32{}, _1{}));

  test<<<1, 128>>>(tensor_a, tensor_b);




}