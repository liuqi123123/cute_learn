#include <iostream>
#include "cutlass/array.h"
#include <cute/tensor.hpp>
#include "cute/numeric/integral_constant.hpp"

using namespace cute;

int main() {

  auto l = Layout<Shape<_8, _6>, Stride<_6, _1>>{};
  // auto tile = Layout<Shape<_2, _3>, Stride<_3, _1>>{};
  auto tile = Layout<Shape<_2, _3>>{};
  // auto p = local_partition(l, tile, Coord<_0, _0>{});
  auto ll = make_identity_tensor(Shape<_8, _6>{});
  auto p = local_partition(ll, tile, 0);
  printf("p:\n");
  for (int i = 0; i < size<0>(p); ++i) {
    for (int j = 0; j < size<1>(p); ++j) {
      printf("coord(%d, %d) -> val(%d, %d)\n", i, j, get<0>(p(i, j)), get<1>(p(i, j)));
    }
  }

  printf("pp:\n");

  auto pp = local_partition(ll, tile, 1);

  for (int i = 0; i < size<0>(pp); ++i) {
    for (int j = 0; j < size<1>(pp); ++j) {
      printf("coord(%d, %d) -> val(%d, %d)\n", i, j, get<0>(pp(i, j)), get<1>(pp(i, j)));
    }
  }

  printf("ppp:\n");

  auto ppp = local_partition(ll, tile, 2);

  for (int i = 0; i < size<0>(ppp); ++i) {
    for (int j = 0; j < size<1>(ppp); ++j) {
      printf("coord(%d, %d) -> val(%d, %d)\n", i, j, get<0>(ppp(i, j)), get<1>(ppp(i, j)));
    }
  }

  printf("pppp:\n");
  auto pppp = local_partition(ll, tile, (1,1));

  for (int i = 0; i < size<0>(pppp); ++i) {
    for (int j = 0; j < size<1>(pppp); ++j) {
      printf("coord(%d, %d) -> val(%d, %d)\n", i, j, get<0>(pppp(i, j)), get<1>(pppp(i, j)));
    }
  }

  // print(layout(p));
  // printf("\n");
  // print(size(p));
  // printf("\n");
  // print(rank(l));
  // printf("\n");
  // print(stride(l));
  // printf("\n");
  // print(shape(l));
  // printf("\n");
  // print(size<0>(l));
  // printf("\n");
  // print(size<1>(l));
  // printf("\n");
  // print(size(l));
  // printf("\n");
  // print(cosize(l));
  // printf("\n");
  // print(get<0>(l));
  // printf("\n");
  // print(get<1>(l));
  // printf("\n");
  // print(get<2>(l));
  // printf("\n");

}