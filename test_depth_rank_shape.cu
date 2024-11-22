#include <iostream>
#include "cutlass/array.h"
#include <cute/tensor.hpp>
#include "include/cuda_helper.h"

using namespace cute;

void test() {
  auto thr_g_a = Layout<Shape<_8, _6, _7>, Stride<Int<42>, _7, _1>>{};
  auto t = make_tensor_like<float>(thr_g_a);
  // PRINT(layout(t(0, 0, 0)));

  auto test = make_tensor<bool>(shape(_2{}));
  test(0) = 1;
  test(1) = 0;
  std::cout << test(0) << std::endl;
  std::cout << test(1) << std::endl;
  printf("test(0):%d\n", int(test(0)));
  printf("test(1):%d\n", int(test(1)));

  // auto b = group_modes<1,3>(make_tensor_like<float>(thr_g_a));
  // printf("b:\n");
  // print_layout(b);
  // print(b);
  // printf("\n");

  // auto thr_g_a = Layout<Shape<_8, _6, Shape<_7, _2>>>{};
  // auto thr_g_a = Layout<Shape<Shape<_8, _6, _4>, Shape<_7, _2, _3>, Shape<_7, _2, _3>>>{}; // rank表示

  /**
   * rank表示维度数量:3
   * depth表示最大的嵌套深度(即1(最外层的shape) + “Shape”最多的那一维度的shape数量):4
   * size<idx>表示第idx维度的总大小，size<0> = 7， size<1> = 40
  */
  // auto thr_g_a =
  //     Layout<Shape<_7, Shape<Shape<_8, Shape<_5>>>, Shape<_7>>>{};

  /**
   * 没有指定stride，则默认为LayoutLeft，无论depth为多少，都为colmajor
  */
  // auto thr_g_a = Layout<Shape<Shape<_4, _2>, Shape<_3, _5>>>{};
  // auto thr_g_a = Layout<Shape<_4, _2>>{};
  // auto thr_g_a = Layout<Shape<_4, Shape<_8, _4>>, Stride<_8, Stride<_1, _32>>>{};
  // auto thr_g_a = Layout<Shape<_8, _16, Int<768>>, Stride<Int<16 * 768>, Int<768>, _1>>{};

  /**
   * LayoutRight默认layout为列优先，但是局限于depth为1的layout,如果是depth>1，则会很奇怪：https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout.md
  */

  // auto thr_g_a = make_layout(Shape<Shape<_4>, Shape<_3>>{},  LayoutRight{});

  // auto thr_g_a = make_layout(Shape<Shape<_4, _2>, Shape<_3, _5>>{},  LayoutRight{});

  // auto thr_g_a = make_layout(Shape<_1, Int<1000>>{});
  // auto b = make_layout(Shape<_1, Int<8>>{});

  // Layout idAB = logical_divide(thr_g_a, b);

  // print(idAB);
  printf("thr_g_a:\n");
  // print_layout(thr_g_a);
  print(thr_g_a);
  printf("\n");

  printf("size<0>:\n");
  print(size<0>(thr_g_a));
  printf("\n");

  printf("size<1>:\n");
  print(size<1>(thr_g_a));
  printf("\n");
  // printf("size<2>:\n");
  // print(size<2>(thr_g_a));
  // printf("\n");

  /**
   *rank = 3
   *
  */
  printf("rank:\n");
  print(rank(thr_g_a));
  printf("\n");
  /**
   * depth = 1
  */
  printf("depth:\n");
  print(depth(thr_g_a));
  printf("\n");

  printf("shape:\n");
  print(shape(thr_g_a));
  printf("\n");

}

int main() {

  test();
  return 0;

}