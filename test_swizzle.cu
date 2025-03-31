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


int main() {
  // auto a = Layout<Shape<_64, _32>, Stride<_32, _1>>{};
  // auto a = Layout<Shape<_64, _8>, Stride<_8, _1>>{};
  // auto a = Layout<Shape<_64, _64>, Stride<_1, _64>>{};
  auto a = Layout<Shape<_16, _64>, Stride<_64, _1>>{};
  // print_layout(a);

  // auto b = composition(Swizzle<0, 0, 0>{}, a);
  // auto b = composition(Swizzle<1, 0, 6>{}, a);
  auto c = composition(Swizzle<3, 1, 4>{}, a);
  // print_layout(b);
  print_latex(c);
}