#include <iostream>
#include "cutlass/array.h"
#include <cute/tensor.hpp>
#include "cute/numeric/integral_constant.hpp"

using namespace cute;

void test() {
  // auto thr_g_a = Layout<Shape<_8, _6, _7>, Stride<Int<42>, _7, _1>>{};

  /**
   * 没有指定stride，则默认为LayoutLeft，无论depth为多少，都为行排列
  */
  // auto thr_g_a = Layout<Shape<Shape<_4, _2>, Shape<_3, _5>>>{};

  /**
   * LayoutRight默认layout为列优先，但是局限于depth为1的layout,如果是depth>1，则会很奇怪：https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout.md
  */

  // auto thr_g_a = make_layout(Shape<Shape<_4>, Shape<_3>>{},  LayoutRight{});

  // auto thr_g_a = make_layout(Shape<Shape<_4, _2>, Shape<_3, _5>>{},  LayoutRight{});

  auto thr_g_a = make_layout(Shape<_1, Int<1000>>{});
  auto b = make_layout(Shape<_1, Int<8>>{})

  Layout idAB = logical_divide(thr_g_a, b);

  print(idAB);
  // printf("thr_g_a:\n");
  // print(thr_g_a);
  // printf("\n");

  // printf("size<0>:\n");
  // print(size<0>(thr_g_a));
  // printf("\n");

  // printf("size<1>:\n");
  // print(size<1>(thr_g_a));
  // printf("\n");
  // print(size<2>(thr_g_a));
  // printf("\n");

  /**
   *rank = 3
   *
  */
  // printf("rank:\n");
  // print(rank(thr_g_a));
  // printf("\n");
  /**
   * depth = 1
  */
  // printf("depth:\n");
  // print(depth(thr_g_a));
  // printf("\n");

  // printf("shape:\n");
  // print(shape(thr_g_a));
  // printf("\n");

}

int main() {

  test();
  return 0;

}