#include <iostream>
#include "cutlass/array.h"
#include <cute/tensor.hpp>
#include "cute/numeric/integral_constant.hpp"
#include "include/cuda_helper.h"
using namespace cute;

void test() {

  auto t = make_tensor<float>(Layout<Shape<_8, _16, Int<765>>, Stride<Int<16 * 765>, Int<765>, _1>>{});
  auto l = local_tile(t, make_shape(_1{}, _16{}, _16{}), make_coord(_5{}, _0{}, Int<47>{}));
  auto l1 = coalesce(l);
  auto l2 = group_modes<0, 2>(l);
  // PRINT(t);
  PRINT(l);
  PRINT(l1);
  PRINT(l2);
  // printf("l:\n");
  // print_layout(thr_g_a);
  // print(thr_g_a);
  // printf("\n");

  // printf("size<0>:\n");
  // print(size<0>(thr_g_a));
  // printf("\n");

  // printf("size<1>:\n");
  // print(size<1>(thr_g_a));
  // printf("\n");



}

int main() {

  test();
  return 0;

}