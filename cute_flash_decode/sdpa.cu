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
#include <iostream>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#define CPU_TEST 1
#define KERNEL_CUR 1

using Element = cutlass::half_t;
using ElementCompute = float;
using namespace cute;

struct Params {
  uint32_t N;
  uint32_t head_num;
  uint32_t E;
  uint32_t L;
  uint32_t S;
  uint32_t Ev;
  uint32_t phase1_block_num;
  float scale;
  void *device_q;
  void *device_k;
  void *device_v;
  void *device_masked;
  void *device_local_block_max;
  void *device_local_exp_sum;
  void *device_temp_to_reduce;
  void *output;

  CUTLASS_HOST_DEVICE
  Params(uint32_t _E, uint32_t _S, uint32_t _Ev, uint32_t _head_num,
         uint32_t _N = 1, uint32_t _L = 1)
      : N(_N), head_num(_head_num), E(_E), L(_L), S(_S), Ev(_Ev) {
    scale = 1 / sqrtf(E);
    // scale = 1;
  }
};

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <int thread_num, int Align, int Align_Ev, typename K_WarpShape, typename K_BlockShape,
          typename q_g2r_copy_tile, typename k_g2r_copy_tile,
          typename v_g2r_copy_tile, typename SharedMem>
__global__ void scale_dot(Params p) {

  extern __shared__ int s[];
  SharedMem& smem = *reinterpret_cast<SharedMem *>(s);

  Element *block_device_q = (Element *)p.device_q + blockIdx.y * p.L * p.E;

  Element *block_device_k = (Element *)p.device_k + blockIdx.y * p.S * p.E;


  Element *block_device_v = (Element *)p.device_v + blockIdx.y * p.S * p.Ev;
  bool* mask = (bool*)p.device_masked + blockIdx.x * K_BlockShape::kColumn;
  ElementCompute *local_block_max =
      static_cast<ElementCompute *>(p.device_local_block_max) +
      blockIdx.y * gridDim.x + blockIdx.x;
  ElementCompute *local_exp_sum =
      static_cast<ElementCompute *>(p.device_local_exp_sum) +
      blockIdx.y * gridDim.x + blockIdx.x;

  auto Q =
      make_tensor(make_gmem_ptr(block_device_q),
                  make_layout(make_shape(_1{}, p.E), make_stride(_0{}, _1{})));
  auto K =
      make_tensor(make_gmem_ptr((Element *)block_device_k),
                  make_layout(make_shape(p.S, p.E), make_stride(p.E, _1{})));

  auto block_K = local_tile(K, make_shape(Int<K_BlockShape::kColumn>{}, p.E),
                            make_coord(blockIdx.x, _0{}));

  constexpr int continuous_thread_num_in_warp = K_WarpShape::kRow / Align;


  q_g2r_copy_tile q_copy_tile;
  ///< thread in same stride_id load same q
  auto q_g2r_thr_copy =
      q_copy_tile.get_slice(threadIdx.x % continuous_thread_num_in_warp);
  auto thr_q_g = q_g2r_thr_copy.partition_S(Q);

  k_g2r_copy_tile k_copy_tile;
  auto k_g2r_thr_copy = k_copy_tile.get_slice(threadIdx.x);
  auto thr_k_g = k_g2r_thr_copy.partition_S(block_K);

  auto thr_r_q = make_fragment_like(thr_q_g(_, _, 0));
  auto thr_r_k = make_fragment_like((thr_k_g(_, _, 0)));
  // auto thr_r_q = make_tensor<Element>(shape(thr_q_g(_, _, 0)));
  // auto thr_r_k = make_tensor<Element>(shape(thr_k_g(_, _, 0)));

  if (thread0()) {
    // PRINT(layout(thr_k_g));
    // PRINT(size<0>(thr_k_g));
    // PRINT(size<1>(thr_k_g));
    // PRINT(size<2>(thr_k_g));
    // PRINT(shape(block_K));
    // PRINT(shape(thr_r_q));
    // PRINT(shape(thr_r_k));
  }
  auto block_K_identity =
      make_identity_tensor(make_shape(size<0>(block_K), size<1>(block_K)));
  auto thr_k_c = k_g2r_thr_copy.partition_S(block_K_identity);


  auto thr_k_c_p = make_tensor<bool>(make_shape(size<1>(thr_k_c)));
  auto thr_q_c_p = make_tensor<bool>(make_shape(size<1>(thr_r_q)));



  // ElementCompute accum = (ElementCompute)0.0f;
  Tensor accum = make_tensor<ElementCompute>(make_shape(size<1>(thr_k_c)));
  clear(accum);
  if (thread0()) {
    // PRINT(shape(thr_q_c_p));
    // PRINT(shape(thr_k_c_p));
    // PRINT(size(thr_k_c_p));
    // PRINT(shape(thr_k_c));
    // PRINT(shape(thr_r_q));
    // PRINT(shape(thr_q_g(_, _, 0)));
    // PRINT(shape(thr_k_g(_, _, 0)));
    // PRINT(rank(thr_q_g(_, _, 0)));
    // PRINT(rank(thr_k_g(_, _, 0)));
    // PRINT(shape(accum));
    // PRINT(shape(thr_r_k));
    // PRINT(shape(thr_k_g));
  }
  for (int i = 0; i < size<2>(thr_k_g); ++i) {
#pragma unroll
    for (int j = 0; j < size(thr_q_c_p); ++j) {
      // thr_q_c_p(j) = 1;
      thr_q_c_p(j) = get<1>(thr_k_c(0, 0, i)) < (int)p.E;
      // thr_q_c_p(j) = (0 < 1);

// if (threadIdx.x == 1) {
//     printf("get<1>(thr_k_c(0, 0, i)):%d\n", get<1>(thr_k_c(0, 0, i)));
//     printf("p.E:%d\n", p.E);
//     printf("thr_q_c_p(0):%d\n", int(thr_q_c_p(0)));
//   }
}

#pragma unroll
    for (int k = 0; k < size<0>(thr_k_c_p); ++k) {

      ///< thr_k_c(0, 0, i)
      ///< 第一个0代表连续取的第8个数中的第一个,第二个0代表一个tile中的第一个8个数
      // thr_k_c_p(k) = true;
      // thr_k_c_p(k) = false;
      thr_k_c_p(k) = (get<1>(thr_k_c(0, 0, i)) < (int)p.E) &&
                     (get<0>(thr_k_c(0, k, i)) <
                      ((int)p.S - blockIdx.x * K_BlockShape::kColumn));
      // if (!thr_k_c_p(k)) {
        // printf("threadIdx.x:%d, blockIdx:%d, k:%d\n", threadIdx.x, blockIdx.x, k);
      // }
      // thr_k_c_p(k) = (get<1>(thr_k_c(0, 0, i)) < (int)p.E) ;
      // thr_k_c_p(k) =
      //                (get<0>(thr_k_c(0, 0, i)) <
      //                 ((int)p.S - blockIdx.x * K_BlockShape::kColumn));
    //   if (thread0()) {
    // printf("thr_k_c_p(%d):%d\n", k, int(thr_k_c_p(0)));
    //   }
    //   if (thread0()) {
    // printf("thr_q_g(0, 0, 0):%f\n", float(thr_q_g(0, 0, 0)));
    // printf("thr_k_g(0, 0, 0):%f\n", float(thr_k_g(0, 0, 0)));
    // if (thr_q_c_p(0)) {
    //   printf("thr_q_c_p(0) true\n");
    // }
    // if (thr_k_c_p(0)) {
    //   printf("thr_k_c_p(0) true\n");
    // }
    // printf("thr_r_q(0):%f\n", float(thr_r_q(0)));
    // printf("thr_r_k(0):%f\n", float(thr_r_k(0)));
    //   }

      //       if (threadIdx.x == 67) {
      //   printf("threadIdx.x:%d, get<1>(thr_k_c(0, 0, %d)):%d,
      //   get<0>(thr_k_c(0, %d, %d)):%d, blockIdx.x * K_BlockShape::kColumn :
      //   %d, p.S:%d, p.E:%d\n",
      //          threadIdx.x,i,get<1>(thr_k_c(0, 0, i)),k, i,get<0>(thr_k_c(0,
      //          k, i)),blockIdx.x * K_BlockShape::kColumn, p.S, p.E);
      // }
      // }

    }


    clear(thr_r_q);
    clear(thr_r_k);

#pragma unroll
    for (int j = 0; j < size(thr_q_c_p); ++j) {
      if (thr_q_c_p(j)) {
        cute::copy(thr_q_g(_, j, i), thr_r_q(_, j));
      }
    }
    ///< cute 3.4.0 is ok, but 3.0.0 not, is it a bug ?
    // cute::copy_if(q_copy_tile, thr_q_c_p, thr_q_g(_, _, i), thr_r_q);

    #pragma unroll
    for (int k = 0; k < size(thr_k_c_p); ++k) {
      if (thr_k_c_p(k)) {
        cute::copy(thr_k_g(_, k, i), thr_r_k(_, k));
      }
    }
    ///< why can't write like this?
    // cute::copy_if(k_copy_tile, thr_k_c_p, thr_k_g(_, _, i), thr_r_k);

    // for (int r_k = 0; r_k < 2; ++r_k) {
    //     cute::copy(thr_k_g(_, r_k, i), thr_r_k(_, r_k));
    // }
    if (thread0()) {
      // PRINT(shape(thr_k_g(_, _, 0)));
      // PRINT(shape(thr_r_k));
      // PRINT(flatten(thr_r_k));
      // PRINT((thr_r_k));
      // auto s = coalesce(thr_r_k);
      // auto s = group_modes<0, 2>(flatten(thr_r_k));
      // PRINT(s);
    }

#pragma unroll
    for (int k = 0; k < size<0>(thr_k_c_p); ++k) {
#pragma unroll
      for (int idx = 0; idx < Align; ++idx) {
        // printf("idx: %d, k:%d\n", idx, k);
          // Element temp = thr_r_q(idx) * thr_r_k(idx, k);
          // accum(k) += (ElementCompute)temp;

        accum(k) +=
            (ElementCompute)(thr_r_q(idx)) * (ElementCompute)((thr_r_k(idx, k)));
        // accum(k) +=
        //     (ElementCompute)(thr_r_q(idx)) * (ElementCompute)(thr_r_k(0* Align + idx));
      }
    }


  }


  const int contiguous_thread_num = K_WarpShape::kRow / Align;

  const int warp_num = thread_num / 32;
  //   if (thread0()){
  //   printf("smem.AccumSmem.data()[0]: %f\n", smem.AccumSmem.data()[0]);
  // }
#pragma unroll
  for (int k = 0; k < size(thr_k_c_p); ++k) {
#pragma unroll
    for (int i = 1; i < contiguous_thread_num; i <<= 1) {
      accum(k) += (ElementCompute)__shfl_xor_sync(0xffffffff, accum(k), i);

    }
  }

  if (threadIdx.x % contiguous_thread_num == 0) {
    int smem_idx = threadIdx.x / contiguous_thread_num;
#pragma unroll
    for (int k = 0; k < size(thr_k_c_p); ++k) {

      ///< maybe out of seme bounds
      if (thr_k_c_p(k)) {
        int offset = smem_idx + k * warp_num * K_WarpShape::kColumn;
        if (!mask[offset]) {
          *(smem.AccumSmem.data() + offset) = accum(k) * p.scale;
        } else {
          *(smem.AccumSmem.data() + offset) = (ElementCompute)-INFINITY;
          // printf("*(smem.AccumSmem.data() + offset):%f\n", *(smem.AccumSmem.data() + offset));
        }
    }

    }
  }
  __syncthreads();




  ///< only one warp to reduce
  if (threadIdx.x < 32) {
    const int iterate_num = (K_BlockShape::kColumn + 31) / 32;
    ElementCompute temp = -INFINITY;

#pragma unroll
    for (int i = 0; i < iterate_num; ++i) {
      int smem_offset =  i * 32 + threadIdx.x;
      if (smem_offset < K_BlockShape::kColumn && smem_offset < p.S - blockIdx.x * K_BlockShape::kColumn) {
        // temp = max(temp, smem.AccumSmem[i * 32 + threadIdx.x]);
        temp = max(temp, *(smem.AccumSmem.data() +smem_offset));

      }
    }



    #pragma unroll
    for (int i = 1; i < 32; i <<= 1) {
      temp = max(temp, __shfl_xor_sync(0xffffffff, temp, i));
    }

    ///< one warp don't need sync?
    if (threadIdx.x == 0) {

    ///< avoid when temp == -inf, exp(-inf - (-inf)) == nan
      smem.block_k_max = temp == -INFINITY ? 0.0f : temp;
      *local_block_max = temp;
      // printf("local_block_max:%f\n" ,*local_block_max);
    }

  // __syncthreads(); ///< why can't __syncthreads？

  __syncwarp();




#pragma unroll
    for (int i = 0; i < iterate_num; ++i) {
      int smem_offset =  i * 32 + threadIdx.x;
      if (smem_offset < K_BlockShape::kColumn && smem_offset <  p.S - blockIdx.x * K_BlockShape::kColumn) {
        ElementCompute pre = *(smem.AccumSmem.data() + smem_offset);
        ElementCompute now = pre - smem.block_k_max;
        *(smem.AccumSmem.data() + smem_offset) = expf(now);
      }
    }
//         if (thread0()) {
//   printf("smem.AccumSmem.data()[0]:%f\n", smem.AccumSmem.data()[0]);
// }
// return;

    ElementCompute expf_sum_temp = (ElementCompute)0;
#pragma unroll
    for (int i = 0; i < iterate_num; ++i) {
      int smem_offset = i * 32 + threadIdx.x;
      if (smem_offset < K_BlockShape::kColumn && smem_offset <  p.S - blockIdx.x * K_BlockShape::kColumn) {
        expf_sum_temp += *(smem.AccumSmem.data() + threadIdx.x + i * 32);
    // if (threadIdx.x <  4) {
    //   printf("iterate_num:%d, threadIdx.xthreadIdx.x:%d, expf_sum_temp:%f\n",iterate_num,threadIdx.x , expf_sum_temp);
    // }
      }
    }

#pragma unroll
    for (int i = 1; i < 32; i <<= 1) {
      expf_sum_temp += __shfl_xor_sync(0xffffffff, expf_sum_temp, i);
    }
    if (threadIdx.x == 0) {
      *local_exp_sum = expf_sum_temp;
    }
  }


  __syncthreads();

  auto V =
      make_tensor(make_gmem_ptr(block_device_v),
                  make_layout(make_shape(p.S, p.Ev), make_stride(p.Ev, _1{})));
  auto block_v = local_tile(V, make_shape(Int<K_BlockShape::kColumn>{}, p.Ev),
                            make_coord(blockIdx.x, _0{}));
  v_g2r_copy_tile v_copy_tile;
  auto v_g2r_thr_copy = v_copy_tile.get_thread_slice(threadIdx.x);
  auto thr_v_g = v_g2r_thr_copy.partition_S(block_v);


  auto thr_r_v = make_fragment_like(thr_v_g(_, 0, 0));

  auto block_v_identity =
      make_identity_tensor(make_shape(size<0>(block_v), size<1>(block_v)));

  auto thr_v_c = v_g2r_thr_copy.partition_S(block_v_identity);

  auto thr_v_c_p = make_tensor<bool>(make_shape(_1{}));

  /// 一个线程迭代去smem拿A
  auto thr_smem_a = make_tensor<ElementCompute>(shape(thr_v_g(0, _, 0)));
  if (thread0()) {
    // PRINT(layout(block_v_identity));
    // PRINT(layout(thr_v_g));
    // PRINT(layout(thr_v_c_p));
    // PRINT(layout(thr_smem_a));
    // PRINT(layout(thr_r_v));

    // PRINT(size<0>(thr_k_g));
    // PRINT(size<1>(thr_k_g));
    // PRINT(size<2>(thr_k_g));
    // PRINT(shape(block_v));
    // PRINT(shape(thr_r_k));
  }

///< 去smem_A中拿K_BlockShape::kColumn / (32 /
///< contiguous_thread_num)个数，准备计算
  // __syncthreads();

clear(thr_smem_a);
#pragma unroll
  for (int i = 0; i < size<1>(thr_v_g); ++i) {
    ElementCompute smem_data = *((ElementCompute*)smem.AccumSmem.data() + threadIdx.x % 32 / contiguous_thread_num +
                      i * (32 / contiguous_thread_num));
    thr_smem_a(i) = smem_data;
    if (threadIdx.x == 64) {
      // printf("smem.AccumSmem.data():%p,smem_data: %f\n",
      //        smem.AccumSmem.data() + threadIdx.x % 32 / 8 +
      //            i * (32 / contiguous_thread_num),
      //        smem_data);
    }
  }



  // auto o_accum = make_fragment_like(thr_v_g(_, 0, 0));
  Tensor o_accum = make_tensor<ElementCompute>(make_shape(size<0>(thr_v_g)));

  Element *block_temp_to_reduce = (Element *)p.device_temp_to_reduce +
                                  p.Ev * (gridDim.x * blockIdx.y + blockIdx.x);

  int lane_idx = threadIdx.x % 32;
  int warp_idx = threadIdx.x / 32;

  for (int i = 0; i < size<2>(thr_v_g); ++i) {
    clear(o_accum);

#pragma unroll
    for (int q = 0; q < size<1>(thr_v_g); ++q) {

      thr_v_c_p(0) =
          get<1>(thr_v_c(0, 0, i)) < p.Ev &&
          (get<0>(thr_v_c(0, q, i)) < p.S - blockIdx.x * K_BlockShape::kColumn);
    //remember  clear!
    clear(thr_r_v);
      if (thr_v_c_p(0)) {

        // bool can_copy = get<1>(thr_v_c(0, 0, i)) < p.Ev &&
        //   (get<0>(thr_v_c(0, q, i)) < p.S - blockIdx.x * K_BlockShape::kColumn);
      // if (can_copy) {
        // cute::copy(v_copy_tile, thr_v_g(_, q, i), thr_r_v);
        cute::copy(thr_v_g(_, q, i), thr_r_v);

      // }
      // thr_v_g(_, q, i) rank == 1,不会走if逻辑
      // cute::copy_if(v_copy_tile, thr_v_c_p, thr_v_g(_, q, i), thr_r_v);
      // if (thread0()) {
      //   PRINT(layout(thr_v_g(_, q, i)));
      //   PRINT(rank(thr_v_g(_, q, i)));
      //   auto src = group_modes<1,decltype(thr_v_g(_, q,
      //   i))::layout_type::rank>(thr_v_g(_, q, i)); PRINT(layout(src));
      }
#pragma unroll
      for (int m = 0; m < Align_Ev; ++m) {
        o_accum(m) += thr_r_v(m) * thr_smem_a(q);
      }
    }

#pragma unroll
    for (int n = contiguous_thread_num; n < 32; n <<= 1) {

#pragma unroll
      for (int s = 0; s < Align_Ev; ++s) {
        o_accum(s) += __shfl_down_sync(uint32_t(-1), o_accum(s), n);
      }
    }
    int Ev_offset =
        (i * contiguous_thread_num * warp_num +
         warp_idx * contiguous_thread_num + lane_idx % contiguous_thread_num) *
        Align_Ev;
    // Tensor element_o_accum = recast<Element>(o_accum);
    Tensor element_o_accum = make_tensor<Element>(make_shape(size<0>(thr_v_g)));
    // Tensor element_o_accum = convert_type<Element>(o_accum);
    // axpby(1, o_accum, Int<0>{}, element_o_accum);

#pragma unroll
    for (int l = 0; l < size<0>(thr_v_g); ++l) {
      element_o_accum(l) = (Element)(o_accum(l));
    }

    cutlass::arch::global_store<decltype(element_o_accum),
                                sizeof(Element) * size<0>(thr_v_g)>(
        element_o_accum, block_temp_to_reduce + Ev_offset,
        (Ev_offset < p.Ev) && (lane_idx < contiguous_thread_num));

  }
}

template <int p2_Align_Ev, typename Smem, typename WarpShape, typename BlockShape,
          typename temp_g2r_copy_tile>
__global__ void reduce(Params p) {
  extern __shared__ ElementCompute exp_mi_sub_mn[];
  ElementCompute *Mn = exp_mi_sub_mn + p.phase1_block_num;
  ElementCompute *Dn = Mn + 1;
  // Smem smem = *(reinterpret_cast<Smem*>(s));
  int lane_idx = threadIdx.x % 32;
  int warp_idx = threadIdx.x / 32;
  ElementCompute *local_block_max =
      static_cast<ElementCompute *>(p.device_local_block_max) +
      blockIdx.y * p.phase1_block_num;
  ElementCompute *block_local_exp_sum =
      static_cast<ElementCompute *>(p.device_local_exp_sum) +
      blockIdx.y * p.phase1_block_num;
  Element *block_target_output =
      static_cast<Element *>(p.output) + blockIdx.y * p.Ev;
  Tensor t = make_tensor(
      make_gmem_ptr((Element *)p.device_temp_to_reduce),
      make_layout(make_shape(gridDim.y, p.phase1_block_num, p.Ev),
                  make_stride(p.phase1_block_num * p.Ev, p.Ev, _1{})));
  Tensor block_t = local_tile(
      t, make_shape(_1{}, p.phase1_block_num, Int<BlockShape::kColumn>{}),
      make_coord(blockIdx.y, _0{}, blockIdx.x));
  // Tensor block_t_co = coalesce(block_t);
  Tensor block_t_co = take<1, 3>(block_t);

  temp_g2r_copy_tile g2r_copy_tile;
  auto thr_g2r_copy = g2r_copy_tile.get_slice(threadIdx.x);
  auto thr_temp_g = thr_g2r_copy.partition_S(block_t_co);
  Tensor temp_r = make_fragment_like(thr_temp_g(_, 0, _));
  // if (thread0()) {
    // PRINT(temp_r);
    // PRINT(layout(block_t_co));
  // }
  ///< 1 warp to reduce, good or not?
  if (threadIdx.x < 32) {
    ///< get Mn
    ElementCompute thr_Mn = -INFINITY;
    for (int i = 0; i < (p.phase1_block_num + 31 / 32); ++i) {
      int idx = i * 32 + threadIdx.x;
      if (idx < p.phase1_block_num) {
        thr_Mn = max(thr_Mn, local_block_max[idx]);
      }
    }
  // if (thread0()) {
  //   printf("thr_Mn: %f\n", float(thr_Mn));
  //   printf("local_block_max[0]: %f\n", float(local_block_max[0]));
  //   printf("local_block_max[0] addr: %p\n", (local_block_max));
  // }
#pragma unroll
    for (int j = 1; j < 32; j <<= 1) {
      thr_Mn = max(__shfl_xor_sync(0xffffffff, thr_Mn, j), thr_Mn);
    }
      /*
          may need thr_Mn = thr_Mn == -INFINITY ? 0.0f : thr_Mn;
          when Mn = -inf, exp_mi_sub_mn[idx] will be nan!!!, but this won't happen when maskedfill all -inf
      */
    if (threadIdx.x == 0) {
      *Mn = thr_Mn;
    }
  }
  __syncthreads();

  ///< get exp_mi_sub_mn
  int block_cur_num = (p.phase1_block_num + blockDim.x - 1) / blockDim.x;
  for (int i = 0; i < block_cur_num; ++i) {
    int idx = threadIdx.x + i * blockDim.x;
    if (idx < p.phase1_block_num) {

      exp_mi_sub_mn[idx] = expf(local_block_max[idx] - *Mn);
    }
  }

  __syncthreads();

  ///< get Dn
  if (threadIdx.x < 32) {
    ElementCompute thr_Dn = (ElementCompute)0.0f;
    for (int i = 0; i < (p.phase1_block_num + 31 / 32); ++i) {
      int idx = i * 32 + threadIdx.x;
      if (idx < p.phase1_block_num) {
        thr_Dn += block_local_exp_sum[idx] * exp_mi_sub_mn[idx];
      }

    }
#pragma unroll
    for (int j = 1; j < 32; j <<= 1) {
      thr_Dn += __shfl_xor_sync(0xffffffff, thr_Dn, j);
    }
    if (threadIdx.x == 0) {
      *Dn = thr_Dn;
    }
  }

  __syncthreads();


  Tensor block_O_identity = make_identity_tensor(shape(block_t_co));
  auto thr_o_c = thr_g2r_copy.partition_S(block_O_identity);

  auto thr_o_p = make_tensor<bool>(size<2>(thr_o_c));

  auto thr_accum = make_tensor<ElementCompute>(shape(temp_r));
  clear(thr_accum);

  for (int i = 0; i < size<1>(thr_o_c); ++i) {
    clear(temp_r);
#pragma unroll
    for (int j = 0; j < size<2>(thr_o_c); ++j) {
      thr_o_p(j) = (get<1>(thr_o_c(0, i, j)) < p.Ev - blockIdx.x * BlockShape::kColumn) &&
                   get<0>(thr_o_c(0, i, j)) < p.phase1_block_num;
      if (thr_o_p(j)) {
      cute::copy(thr_temp_g(_, i, _), temp_r(_, j));
      }

    }

    // cute::copy_if(g2r_copy_tile, thr_o_p, thr_temp_g(_, i, _), temp_r);

#pragma unroll
    for (int m = 0; m < size<1>(temp_r); ++m) {
#pragma unroll
      for (int k = 0; k < p2_Align_Ev; ++k) {
        int exp_mi_sub_mn_idx = i * 32 + lane_idx;
        if (exp_mi_sub_mn_idx < p.phase1_block_num) {
          thr_accum(k, m) +=
              (ElementCompute)temp_r(k, m) * exp_mi_sub_mn[exp_mi_sub_mn_idx];
        }
      }
    }
  }

#pragma unroll
  for (int m = 0; m < size<1>(temp_r); ++m) {
#pragma unroll
    for (int k = 0; k < p2_Align_Ev; ++k) {
#pragma unroll
      for (int i = 1; i < 32; i <<= 1) {
        thr_accum(k, m) += __shfl_xor_sync(0xffffffff, thr_accum(k, m), i);
      }
    }
    ///< bug : when p2_Align_Ev == 4, Element == half, sizeof(thr_output) = 16 ???
    Tensor thr_output = make_tensor<Element>(size<0>(thr_accum));

      /*
         a potential bug, when maskedfill all -inf， *Dn will be zero
      */
     axpby(1 / *Dn, thr_accum(_,m), Int<0>{}, thr_output);
// #pragma unroll
//     for (int i = 0; i < p2_Align_Ev; ++i) {

      // thr_output(i) = (Element)thr_accum(i, m) / (*Dn);
      // thr_output(i) = (Element)thr_accum(i, m) ;
    // }
      // thr_output /=  (*Dn);

    const int warp_num = BlockShape::kColumn / p2_Align_Ev;

    int offset = blockIdx.x * BlockShape::kColumn +
                 m * warp_num * WarpShape::kColumn +
                 warp_idx * WarpShape::kColumn;

    ///< bug in 3.0.0, when p2_Align_Ev == 4, Element == half, sizeof(thr_output) = 16
    // cutlass::arch::global_store<decltype(thr_output), sizeof(thr_output)>(
    cutlass::arch::global_store<decltype(thr_output), sizeof(Element) * p2_Align_Ev>(
        thr_output, block_target_output + offset,
        lane_idx == 0 && offset < p.Ev);
  }
}

const uint32_t Align = 16 / sizeof(Element);
const uint32_t Align_Ev = 16 / sizeof(Element);
const uint32_t p2_Align_Ev = 8;

// const uint32_t A = 4;
// const uint32_t Align = 8;
// const uint32_t Align_Ev = 2;
// const uint32_t p2_Align_Ev = 4;


const uint32_t thread_num = 128;
const uint32_t phase2_thread_num = 128;


const uint32_t continous_thread_in_warp_k = 8;
///< K must be colmajor
using K_WarpShape = cutlass::MatrixShape<continous_thread_in_warp_k * Align,
                                         32 / continous_thread_in_warp_k>;

const uint32_t block_iterate_num = 8;
using K_BlockShape =
    cutlass::MatrixShape<K_WarpShape::kRow, K_WarpShape::kColumn * thread_num /
                                                32 * block_iterate_num>;

const int smem_size = K_BlockShape::kColumn;
struct SharedMem {
  cutlass::AlignedBuffer<ElementCompute, smem_size> AccumSmem;
  // cutlass::AlignedArray<ElementCompute, 32>  AccumSmem;
  ElementCompute block_k_max;
};
struct Phase2_Smem {};

int main() {
  // Params p{128, 2048, 128, 8};
  // Params p{128, 2048, 128, 32};
  Params p{64, 273, 64, 8};
  // Params p{128, 4096, 128, 8};
  // Params p{768, 256, 768, 8};
  // Params p{273, 64, 273, 8};
  // Params p{266, 61, 266, 3};
  // Params p{1, 5, 1, 1};
  // Params p{231, 167, 231, 5};
  // Params p{768, 256, 768, 1};
  // Params p{64,16, 64, 1};



  MemHelper<Element> mem_helper;
  auto test = mem_helper.GetCpuGpuBuffer(p.N * p.head_num * (p.E + p.E * p.S + p.Ev * p.S + p.Ev));
  auto test1 = mem_helper.GetCpuGpuBuffer(p.N * p.head_num * (p.E + p.E * p.S + p.Ev * p.S + p.Ev));
  cudaMemcpy(test.second, test1.second, sizeof(Element) * p.N * p.head_num * (p.E + p.E * p.S + p.Ev * p.S + p.Ev), cudaMemcpyDeviceToDevice);
  std::pair<void *, void *> Q_pair =
      mem_helper.GetCpuGpuBuffer(p.N * p.head_num * p.L * p.E, true);

  std::pair<void *, void *> K_pair =
      mem_helper.GetCpuGpuBuffer(p.N * p.head_num * p.S * p.E, true);
  std::pair<void *, void *> V_pair =
      mem_helper.GetCpuGpuBuffer(p.N * p.head_num * p.S * p.Ev, true);

  MemHelper<bool> bool_mem_helper;

  std::pair<void *, void *> Mask_pair = bool_mem_helper.GetCpuGpuBuffer(
      p.S, true); // broadcast to [N,Head_num,,L,S]

  uint32_t block_num =
      (p.S + K_BlockShape::kColumn - 1) / K_BlockShape::kColumn;

  std::pair<void *, void *> temp_to_reduce =
      mem_helper.GetCpuGpuBuffer(p.N * p.head_num * p.Ev * block_num, false);

  std::pair<void *, void *> output =
      mem_helper.GetCpuGpuBuffer(p.N * p.head_num * p.Ev, false);

  MemHelper<ElementCompute> compute_mem_helper;
  std::pair<void *, void *> local_max_pair =
      compute_mem_helper.GetCpuGpuBuffer(p.N * p.head_num * block_num, false);
  std::pair<void *, void *> local_exp_sum =
      compute_mem_helper.GetCpuGpuBuffer(p.N * p.head_num * block_num, false);

  p.device_q = Q_pair.second;
  p.device_k = K_pair.second;
  p.device_v = V_pair.second;
  p.device_masked = Mask_pair.second;
  p.device_local_block_max = local_max_pair.second;
  p.device_local_exp_sum = local_exp_sum.second;
  p.device_temp_to_reduce = temp_to_reduce.second;
  p.phase1_block_num = block_num;
  p.output = output.second;

  dim3 block{thread_num};
  dim3 grid{block_num, p.head_num, p.N};

  using T_Q = cutlass::AlignedArray<Element, Align>;
  using q_g2r_copy_op = UniversalCopy<T_Q>;
  using q_g2r_traits = Copy_Traits<q_g2r_copy_op>;
  using q_g2r_copy_atom = Copy_Atom<q_g2r_traits, T_Q>;
  using q_g2r_copy_tile = decltype(make_tiled_copy(
      q_g2r_copy_atom{},
      Layout<Shape<Int<1>, Int<continous_thread_in_warp_k>>, Stride<_0, _1>>{},
      Layout<Shape<_1, Int<Align>>>{}));

  using T_K = cutlass::Array<Element, Align>;
  using k_g2r_copy_op = UniversalCopy<T_K>;
  using k_g2r_traits = Copy_Traits<k_g2r_copy_op>;
  using k_g2r_copy_atom = Copy_Atom<k_g2r_traits, T_K>;
  using k_g2r_copy_tile = decltype(make_tiled_copy(
      k_g2r_copy_atom{},
      Layout<Shape<Int<thread_num / continous_thread_in_warp_k>,
                   Int<continous_thread_in_warp_k>>,
             Stride<Int<continous_thread_in_warp_k>, _1>>{},
      Layout<Shape<_1, Int<Align>>>{}));

  using T_V = cutlass::AlignedArray<Element, Align_Ev>;
  using v_g2r_copy_op = UniversalCopy<T_V>;
  using v_g2r_traits = Copy_Traits<v_g2r_copy_op>;
  using v_g2r_copy_atom = Copy_Atom<v_g2r_traits, T_V>;
  using v_g2r_copy_tile = decltype(make_tiled_copy(
      v_g2r_copy_atom{},
      Layout<
          Shape<Int<32 / continous_thread_in_warp_k>,
                Shape<Int<continous_thread_in_warp_k>, Int<thread_num / 32>>>,
          Stride<Int<continous_thread_in_warp_k>, Stride<_1, _32>>>{},
      Layout<Shape<_1, Int<Align_Ev>>>{}));
for (int i = 0; i < KERNEL_CUR; ++i) {
  scale_dot<thread_num, Align, Align_Ev, K_WarpShape, K_BlockShape, q_g2r_copy_tile,
  k_g2r_copy_tile, v_g2r_copy_tile, SharedMem><<<grid,
  block,sizeof(SharedMem), 0>>>(p);
}

  mem_helper.SyncGpuToCpu(temp_to_reduce);
  compute_mem_helper.SyncGpuToCpu(local_max_pair);
  compute_mem_helper.SyncGpuToCpu(local_exp_sum);



  // Element* cpu_temp_reduce = (Element*)temp_to_reduce.first;
  // for (int h = 0; h < p.head_num; ++h) {
  //   for (int b = 0; b < p.phase1_block_num; ++b) {
  //     for (int ev = 0; ev < p.Ev; ++ev) {
  //       Element value = cpu_temp_reduce[h * p.phase1_block_num * p.Ev + b * p.Ev + ev];
  //       printf("h:%d, b:%d, ev:%d, value: %f\n", h, b, ev, float(value));
  //     }
  //   printf("h:%d, b:%d, local block max:%f\n",h, b,((float*)(local_max_pair.first))[h * p.phase1_block_num +  b]);
  //   printf("h:%d, b:%d, device_local_exp_sum:%f\n",h,b,((float*)(local_exp_sum.first))[h * p.phase1_block_num + b]);
  //   }
  // }



  const uint32_t phase2_warp_num = phase2_thread_num / 32;
  ///< a warp reduce one column
  uint32_t phase2_block_num =
      (p.Ev + phase2_warp_num * p2_Align_Ev - 1) / (phase2_warp_num * p2_Align_Ev);

  dim3 phase2_block{phase2_thread_num};
  dim3 phase2_grid{phase2_block_num, p.head_num, p.N};

  using p2_WarpShape = cutlass::MatrixShape<32, p2_Align_Ev>;
  using p2_BlockShape = cutlass::MatrixShape<32, p2_Align_Ev * phase2_warp_num>;

  using T_temp = cutlass::AlignedArray<Element, p2_Align_Ev>;
  using temp_g2r_copy_op = UniversalCopy<T_temp>;
  using temp_g2r_traits = Copy_Traits<temp_g2r_copy_op>;
  using temp_g2r_copy_atom = Copy_Atom<temp_g2r_traits, T_temp>;
  using temp_g2r_copy_tile = decltype(make_tiled_copy(
      temp_g2r_copy_atom{},
      Layout<Shape<_32, Int<phase2_warp_num>>, Stride<_1, _32>>{},
      Layout<Shape<_1, Int<p2_Align_Ev>>>{}));

  ///< shared: 1 for Mn, 1 for Dn
for (int i = 0; i < KERNEL_CUR; ++i) {
  reduce<p2_Align_Ev, Phase2_Smem, p2_WarpShape, p2_BlockShape, temp_g2r_copy_tile>
      <<<phase2_grid, phase2_block,
         sizeof(ElementCompute) * (block_num + 1 + 1), 0>>>(p);
}

  mem_helper.SyncGpuToCpu(output);
    cudaDeviceSynchronize();


  ///< get cpu gt

  Element *q = (Element *)Q_pair.first;
  Element *k = (Element *)K_pair.first;
  Element *v = (Element *)V_pair.first;

  Element *cpu_temp_qk =
      (Element *)mem_helper.GetCpuBuffer(p.N * p.head_num * p.S);
  Element *cpu_gt = (Element *)mem_helper.GetCpuBuffer(p.N * p.head_num * p.Ev);

  if (CPU_TEST) {
    // p.N = 1
    for (int h = 0; h < p.head_num; ++h) {
      Element *target_qk = cpu_temp_qk + h * p.S;
      Element *target_q = q + h * p.E;
      Element *target_k = k + h * p.S * p.E;
      bool* mask = (bool*)Mask_pair.first;
      // k @ v
      for (int s = 0; s < p.S; ++s) {
        Element *kv = target_qk + s;
        ElementCompute sum = 0.0f;
        for (int e = 0; e < p.E; ++e) {
          sum += (ElementCompute)target_q[e] *
                 (ElementCompute)target_k[s * p.E + e];
        // std::cout << target_q[e] << std::endl;
        // std::cout << target_k[s * p.E + e] << std::endl;
        }
        // to do scale
        // std::cout << "sum"<< sum << "p.scale"<<p.scale << std::endl;

        *kv = (Element)(sum * p.scale);
        // *kv = (Element)(sum);
        // std::cout << *kv << std::endl;

      }
      // to do  maskedfill
      for (int m = 0; m < p.S; ++m) {
        Element *kv = target_qk + m;
        if (mask[m]) {
          *kv = (Element)(-INFINITY);
        }
      }
      // softmax
      ElementCompute max_kv = -INFINITY;
      std::vector<ElementCompute> temp_qk(p.S);
      for (int s = 0; s < p.S; ++s) {
        max_kv = std::max(max_kv, (ElementCompute)target_qk[s]);
        temp_qk[s] = (ElementCompute)target_qk[s];
        // std::cout << "temp_qk["<<s<<"]:"<<temp_qk[s] << std::endl;
      }

      for (int s = 0; s < p.S; ++s) {
        temp_qk[s] = temp_qk[s] - max_kv;
        // std::cout << "-max, temp_qk["<<s<<"]:"<<temp_qk[s] << std::endl;
      }
      ElementCompute exp_sum = 0.0f;
      for (int s = 0; s < p.S; ++s) {
        exp_sum += expf(temp_qk[s]);
      }
        // std::cout << "exp_sum@@@@:"<<exp_sum << std::endl;

      for (int s = 0; s < p.S; ++s) {
        target_qk[s] = (Element)(expf(temp_qk[s]) / exp_sum);
        // std::cout << " target_qk["<<s<<"]:"<<target_qk[s] << std::endl;

      }

      //(qk) * v
      Element *target_v = v + h * p.S * p.Ev;
      Element *cpu_target_gt = cpu_gt + h * p.Ev;

      for (int ev = 0; ev < p.Ev; ++ev) {
        Element *target_out = cpu_target_gt + ev;
        ElementCompute sum = 0.0f;
        for (int s = 0; s < p.S; ++s) {
          sum += (ElementCompute)target_qk[s] *
                 (ElementCompute)target_v[ev + p.Ev * s];
        }
        *target_out = (Element)sum;
      }
    }

    mem_helper.Regression(cpu_gt, (Element *)output.first);
  }

  return 0;
}
