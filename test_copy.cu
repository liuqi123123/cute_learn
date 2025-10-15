
// #include "cutlass/conv/conv2d_problem_size.h"
// #include "cutlass/conv/device/implicit_gemm_convolution.h"
// #include "include/cuda_helper.h"
// #include "cutlass/fast_math.h"
// #include "cute/numeric/integral_constant.hpp"
// #include "cute/pointer.hpp"
// #include "cutlass/aligned_buffer.h"
// #include "cutlass/arch/memory.h"
// #include "cutlass/array.h"
// #include "cutlass/array_subbyte.h"
// #include "cutlass/cutlass.h"
// #include "cutlass/gemm/gemm.h"
// #include "cutlass/matrix_shape.h"
// #include <cmath>
#include <cute/tensor.hpp>
// #include <cutlass/numeric_conversion.h>
// #include <cutlass/numeric_types.h>
// #include <iostream>
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
  // MemHelper<cutlass::half_t> helper;
  // auto A = helper.GetCpuGpuBuffer(32*32, InitialType::Increment);
  // auto B = helper.GetCpuGpuBuffer(64*64);
  // auto tensor_a =  make_tensor(make_gmem_ptr((float*)A.second), make_shape(_32{}, _32{}), make_stride(_32{}, _1{}));
  // auto tensor_b = make_tensor(make_gmem_ptr((float*)B.second), make_shape(_32{}, _32{}), make_stride(_32{}, _1{}));

  // // test<<<1, 128>>>(tensor_a, tensor_b);

  // cutlass::half_t* t = new  cutlass::half_t[64*64];
  // using mma_op = SM80_16x8x16_F16F16F16F16_TN;
  // using mma_traits = MMA_Traits<mma_op>;
  // using mma_atom = MMA_Atom<mma_traits>;
  // using MMA1 = decltype(make_tiled_mma(
  //     mma_atom{},
  //     Layout<Shape<Int<4>, Int<1>, Int<1>>>{}
  //     ));
  // MMA1 mma1;
  // auto thr_mma =  mma1.get_slice(_16{});
  // using noswizzle_layout = decltype(composition(Swizzle<0, 0, 0>{}, Layout<Shape<_64, _64>, Stride<_64, _1>>{}));
  // using swizzle_layout = decltype(composition(Swizzle<3, 3, 3>{}, Layout<Shape<_64, _64>, Stride<_64, _1>>{}));
  // auto noswizzle_tensor_c = make_tensor(make_gmem_ptr((cutlass::half_t*)t), noswizzle_layout{});
  // auto swizzle_tensor_c = make_tensor(make_gmem_ptr((cutlass::half_t*)t), swizzle_layout{});
  // auto swizzle_reg = thr_mma.partition_C(swizzle_tensor_c);
  // auto noswizzle_reg = thr_mma.partition_C(noswizzle_tensor_c);
  // print(noswizzle_reg);
  // print("\n");
  // print(swizzle_reg);

//   auto layout_a = make_layout(make_shape (Int< 9>{}, make_shape (Int< 4>{}, Int<8>{})),
//                             make_stride(Int<59>{}, make_stride(Int<13>{}, Int<1>{})));
// // B: shape is (3,8)
// auto tiler = make_tile(Layout<_3,_3>{},           // Apply     3:3     to mode-0
//                        Layout<Shape <_2,_4>,      // Apply (2,4):(1,8) to mode-1
//                               Stride<_1,_8>>{});

// // ((TileM,RestM), (TileN,RestN)) with shape ((3,3), (8,4))
// auto ld = logical_divide(layout_a, tiler);
// // ((TileM,TileN), (RestM,RestN)) with shape ((3,8), (3,4))
// auto zd = zipped_divide(layout_a, tiler);
// auto td = tiled_divide(layout_a, tiler);
// auto fd = flat_divide(layout_a, tiler);
using T = cute::tuple<int, int>;
T t = cute::make_tuple(2, 3);
print(t);
get<0>(t) = 3;
print(t);
// auto l0 = Layout<Shape<_3, _6>, Stride<_16, _9>>{};
// auto l1 = Layout<_16, _9>{};
// auto l2 = composition(l0, l1);
// print(l2);
// auto l0 = Layout<Shape<_3, _6, _2, _8>, Stride<_16, _9, _5, _3>>{};
// auto l1 = Layout<Shape<_16>, Stride<_9>>{};
// auto l2 = composition(l0, l1);
// print(l2);
// print(ld);
// printf("\n");
// print(zd);
// printf("\n");
// print(td);
// printf("\n");
// print(fd);



}