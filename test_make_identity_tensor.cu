#include <iostream>
#include "cutlass/array.h"
#include <cute/tensor.hpp>
#include "cute/numeric/integral_constant.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/pointer.hpp"
#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/array_subbyte.h"
#include "cutlass/cutlass.h"
#include "cutlass/matrix_shape.h"
#include "include/cuda_helper.h"
#include <cmath>
#include <cute/tensor.hpp>

using namespace cute;

void test_complete_shape() {
  auto l = Layout<Shape<_8, _6>, Stride<_6, _1>>{};


  auto tile = Layout<Shape<_2, _3>>{};

  auto ll = make_identity_tensor(Shape<_8, _6>{});


  /**
   * 规律：
   *      tile的shape是(2, 3), idx = 0时，p 取每个tile里的coord(0, 0)
   *                        , idx = 1时，p 取每个tile里的coord(1, 0)
   *                        , idx = 2时，p 取每个tile里的coord(0, 1)
   *                        , idx = 3时，p 取每个tile里的coord(1, 1)
   *                        , idx = 4时，p 取每个tile里的coord(0, 2)
   *                        , idx = 5时，p 取每个tile里的coord(1, 2)
   * question:idx递增时，扫描tile时，为什么是列优先？
   *
  */
  printf("p:\n");
  int idx = 0;
  auto p = local_partition(ll, tile, idx);
  for (int i = 0; i < size<0>(p); ++i) {
    for (int j = 0; j < size<1>(p); ++j) {
      printf("coord(%d, %d) -> val(%d, %d)\n", i, j, get<0>(p(i, j)), get<1>(p(i, j)));
    }
  }
  // print(shape(p));
  // printf("\n");

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
  auto pppp = local_partition(ll, tile, 3);

  for (int i = 0; i < size<0>(pppp); ++i) {
    for (int j = 0; j < size<1>(pppp); ++j) {
      printf("coord(%d, %d) -> val(%d, %d)\n", i, j, get<0>(pppp(i, j)), get<1>(pppp(i, j)));
    }
  }

    printf("ppppp:\n");
  auto ppppp = local_partition(ll, tile, 4);

  for (int i = 0; i < size<0>(ppppp); ++i) {
    for (int j = 0; j < size<1>(ppppp); ++j) {
      printf("coord(%d, %d) -> val(%d, %d)\n", i, j, get<0>(ppppp(i, j)), get<1>(ppppp(i, j)));
    }
  }
      printf("pppppp:\n");
  auto pppppp = local_partition(ll, tile, 5);

  for (int i = 0; i < size<0>(pppppp); ++i) {
    for (int j = 0; j < size<1>(pppppp); ++j) {
      printf("coord(%d, %d) -> val(%d, %d)\n", i, j, get<0>(pppppp(i, j)), get<1>(pppppp(i, j)));
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

void test_incomplete_shape() {
  auto l = Layout<Shape<_8, _6>, Stride<_6, _1>>{};

  auto tile = Layout<Shape<_2, _3>>{};

  auto ll = make_identity_tensor(Shape<_8, _5>{});

  /**
   * 规律：local_partition会把make_identity_tensor生成的tensor填成可以被tile整除的模样
  */
  printf("p:\n");
  int idx = 0;
  auto p = local_partition(ll, tile, idx);
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
  auto pppp = local_partition(ll, tile, 3);

  for (int i = 0; i < size<0>(pppp); ++i) {
    for (int j = 0; j < size<1>(pppp); ++j) {
      printf("coord(%d, %d) -> val(%d, %d)\n", i, j, get<0>(pppp(i, j)), get<1>(pppp(i, j)));
    }
  }

    printf("ppppp:\n");
  auto ppppp = local_partition(ll, tile, 4);

  for (int i = 0; i < size<0>(ppppp); ++i) {
    for (int j = 0; j < size<1>(ppppp); ++j) {
      printf("coord(%d, %d) -> val(%d, %d)\n", i, j, get<0>(ppppp(i, j)), get<1>(ppppp(i, j)));
    }
  }
      printf("pppppp:\n");
  auto pppppp = local_partition(ll, tile, 5);

  for (int i = 0; i < size<0>(pppppp); ++i) {
    for (int j = 0; j < size<1>(pppppp); ++j) {
      printf("coord(%d, %d) -> val(%d, %d)\n", i, j, get<0>(pppppp(i, j)), get<1>(pppppp(i, j)));
    }
  }
}

__global__ void fuck() {
  if (thread0()) {
    auto l = make_identity_tensor(make_shape(_32{}, 64));

    using T_V = cutlass::AlignedArray<cutlass::half_t, 8>;
    using v_g2r_copy_op = UniversalCopy<T_V>;
    using v_g2r_traits = Copy_Traits<v_g2r_copy_op>;
    using v_g2r_copy_atom = Copy_Atom<v_g2r_traits, T_V>;
    // using v_g2r_copy_tile = decltype(make_tiled_copy(
    //     v_g2r_copy_atom{},
    //     Layout<
    //         Shape<_4,
    //               Shape<_8, _4>>,
    //         Stride<_8, Stride<_1, _32>>>{},
    //     Layout<Shape<_1,_8>>{}));

    using v_g2r_copy_tile = decltype(make_tiled_copy(
        v_g2r_copy_atom{}, Layout<Shape<_16, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1, _8>>{}));

    v_g2r_copy_tile v_copy_tile;
    auto v_g2r_thr_copy = v_copy_tile.get_thread_slice(10);

    auto thr_v_c = v_g2r_thr_copy.partition_S(l);
    printf("get<0>(thr_v_c(0, 0, i)) : %d\n", int(get<0>(thr_v_c(0, 0, 0))));
    printf("get<1>(thr_v_c(0, 0, i)): %d\n", int(get<1>(thr_v_c(0, 0, 0))));
    bool can = get<0>(thr_v_c(0, 0, 0)) < 32 && get<1>(thr_v_c(0, 0, 0)) < 64;
    printf("can: %d\n", can);
    PRINT(layout(thr_v_c));
  }
}

void test_patitionS() {
  fuck<<<1,1>>>();
  cudaDeviceSynchronize();

  auto l = make_identity_tensor(make_shape(_32{}, 64));

  using T_V = cutlass::AlignedArray<cutlass::half_t, 8>;
  using v_g2r_copy_op = UniversalCopy<T_V>;
  using v_g2r_traits = Copy_Traits<v_g2r_copy_op>;
  using v_g2r_copy_atom = Copy_Atom<v_g2r_traits, T_V>;
  // using v_g2r_copy_tile = decltype(make_tiled_copy(
  //     v_g2r_copy_atom{},
  //     Layout<
  //         Shape<_4,
  //               Shape<_8, _4>>,
  //         Stride<_8, Stride<_1, _32>>>{},
  //     Layout<Shape<_1,_8>>{}));

  using v_g2r_copy_tile = decltype(make_tiled_copy(
      v_g2r_copy_atom{}, Layout<Shape<_16, _8>, Stride<_8, _1>>{},
      Layout<Shape<_1, _8>>{}));

  v_g2r_copy_tile v_copy_tile;
  auto v_g2r_thr_copy = v_copy_tile.get_thread_slice(10);

  auto thr_v_c = v_g2r_thr_copy.partition_S(l);
  printf("get<0>(thr_v_c(0, 0, 0)) : %d\n", int(get<0>(thr_v_c(0, 0, 0))));
  printf("get<1>(thr_v_c(0, 0, 0)): %d\n", int(get<1>(thr_v_c(0, 0, 0))));
  bool can = get<0>(thr_v_c(0, 0, 0)) < 32 && get<1>(thr_v_c(0, 0, 0)) < 64;
  printf("can: %d\n", can);
  PRINT(layout(thr_v_c));

  //   auto l = make_identity_tensor(make_shape(_32{}, 64));

  //   using T_V = cutlass::AlignedArray<cutlass::half_t, 8>;
  //   using v_g2r_copy_op = UniversalCopy<T_V>;
  //   using v_g2r_traits = Copy_Traits<v_g2r_copy_op>;
  //   using v_g2r_copy_atom = Copy_Atom<v_g2r_traits, T_V>;
  //   using v_g2r_copy_tile = decltype(make_tiled_copy(
  //       v_g2r_copy_atom{},
  //       Layout<
  //           Shape<_4,
  //                 Shape<_8, _4>>,
  //           Stride<_8, Stride<_1, _32>>>{},
  //       Layout<Shape<_1,_8>>{}));

  // v_g2r_copy_tile v_copy_tile;
  // auto v_g2r_thr_copy = v_copy_tile.get_thread_slice(10);

  //   auto thr_v_c = v_g2r_thr_copy.partition_S(l);
  // printf("get<0>(thr_v_c(0, 0, i)) : %d\n",int(get<0>(thr_v_c(0, 0, 0))));
  //  printf("get<1>(thr_v_c(0, 0, i)): %d\n", int(get<1>(thr_v_c(0, 0, 0))));
  //  bool can = get<0>(thr_v_c(0, 0, 0)) < 32 && get<1>(thr_v_c(0, 0, 0)) < 32;
  //  printf("can: %d\n", can);
  //  PRINT(layout(thr_v_c));
}

int main() {
  // test_complete_shape();
  // test_incomplete_shape();
    // auto l = Layout<Shape<_8, Shape<_6,_2>, _9>>{};
    auto l = make_layout(Shape<_8, Shape<_6,_2>, _9>{}, LayoutRight{});
    printf("%3d  ", l(0, make_coord(0, 1), 0));
  PRINT(layout(l));
  PRINT((get<1>(l)));
  PRINT((get<1, 1, 0>(l)));
  PRINT(stride<_0{}>(l));
  PRINT(shape<_1{}>(l));
  auto s = make_shape(3, 4);
  auto v = make_shape(4, 4);
  s = v;
  PRINT(s);
  Layout<Shape<>, Stride<>> l0;
  Layout l1 = Layout<Shape<_1,_1>, Stride<_1, _2>>{};
  l0 = l1;
  PRINT(l0);


  // PRINT(stride(l)(0));

auto layout = Layout<Shape <_2,Shape <_1,_6>>,
                     Stride<_1,Stride<_6,_2>>>{};
auto result = coalesce(layout);
  print(layout);
  print(result);
  // test_patitionS();
  return 0;

}