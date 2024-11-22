#pragma once
#include "cuda_runtime.h"
#include "cutlass/half.h"
#include "cute/util/print.hpp"
#include <iostream>
#include <random>
#include <vector>

#define PRINT(STR)                                                             \
  do {                                                                         \
    printf("%s:\n", #STR);                                                     \
    print(STR);                                                                \
    printf("\n\n");                                                              \
  } while (0)




template <typename T> class MemHelper {
  struct pair_hash {
    template <typename T1, typename T2>
    size_t operator()(const std::pair<T1, T2> &p) const noexcept {
      auto hash1 = std::hash<T1>{}(p.first);
      auto hash2 = std::hash<T2>{}(p.second);
      return hash1 ^ (hash2 << 1);
    }
  };

private:
  // std::pair<cpu_ptr, gpu_ptr>
  using CG_PTR = std::pair<void *, void *>;
  std::vector<void *> cpu_ptr_set_;
  std::vector<CG_PTR> cpu_gpu_ptr_set_;
  std::unordered_map<CG_PTR, size_t, pair_hash> cg_size_;
  std::unordered_map<void *, size_t> c_size_;

  std::random_device rd_;
  std::default_random_engine eng_;
  std::uniform_real_distribution<float> distr_;

public:
  MemHelper() : rd_(), eng_(rd_()), distr_(-1.0f, 1.0f) {}
  CG_PTR GetCpuGpuBuffer(size_t size, bool need_initial = true) {
    T *cpu_ptr = new T[size]{static_cast<T>(0)};

    if (need_initial) {
      if constexpr (std::is_same_v<bool, T>) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution d(0.5);
        for (size_t i = 0; i < size; ++i) {
          cpu_ptr[i] = d(gen);
          // cpu_ptr[i] = 1;
        }
      } else {
        // if (size == 5) {
        //       for (size_t i = 0; i < size ; ++i) {
        //       cpu_ptr[i] = (T)(int)i;

        //     }
        // } else {

        for (size_t i = 0; i < size ; ++i) {
          cpu_ptr[i] = (T)distr_(eng_);
        }
        }
      // }
    }

    T *gpu_ptr;
    cudaMalloc((void **)&gpu_ptr, size * sizeof(T));
    cudaMemset((void *)gpu_ptr, 0, size * sizeof(T));
    cudaMemcpy(gpu_ptr, cpu_ptr, size * sizeof(T), cudaMemcpyHostToDevice);
    cpu_gpu_ptr_set_.emplace_back(
        std::make_pair((void *)cpu_ptr, (void *)gpu_ptr));

    cg_size_[{cpu_ptr, gpu_ptr}] = size;
    c_size_[cpu_ptr] = size;
    return {cpu_ptr, gpu_ptr};
  }

  T *GetCpuBuffer(size_t size) {
    T *cpu_ptr = new T[size]{static_cast<T>(0)};
    cpu_ptr_set_.push_back(cpu_ptr);
    c_size_[(void *)cpu_ptr] = size;
    return cpu_ptr;
  }

  void SyncGpuToCpu(const CG_PTR &cg) {
    assert(cg_size_.count(cg));
    cudaMemcpy(cg.first, cg.second, cg_size_[cg] * sizeof(T),
               cudaMemcpyDeviceToHost);
  }

  void Regression(T *gt, T *result, bool first_wrong_break = true) {
    size_t size = c_size_[(void *)gt];
    using Ts = float;

    for (int i = 0; i < size; ++i) {
      Ts trans_gt = Ts(((T *)gt)[i]);
      Ts trans_res = Ts(((T *)result)[i]);


      float dif = std::abs(((T *)gt)[i] - ((T *)result)[i]);
      float absolute_error = 0.1;

      if constexpr (std::is_same_v<T, int8_t>) {
        absolute_error = 1;
      } else if constexpr (std::is_same_v<T, cutlass::half_t>) {
        absolute_error = 0.01;
      } else {
        absolute_error = 0.1;
      }

      if (std::isnan(trans_gt)) {
        std::cout << "gt idx: " << i << " is nan!" << std::endl;
        break;
      }else if (std::isinf(trans_gt))  {
        std::cout << "gt idx: " << i << " is inf!" << std::endl;
        break;
      }

      if (std::isnan(trans_res)) {
        std::cout << "res idx: " << i << " is nan!" << std::endl;
        break;
      }else if (std::isinf(trans_res))  {
        std::cout << "res idx: " << i << " is inf!" << std::endl;
        break;
      }

      if (dif > (Ts)absolute_error) {
        std::cout << "idx: " << i << " get wrong: "
                  << "gt: " << (Ts)((T *)gt)[i] << " vs "
                  << (Ts)((T *)result)[i] << std::endl;
        if (first_wrong_break) {
          break;
        }
      }
    }
  }

  ~MemHelper() {
    for (auto &p : cpu_gpu_ptr_set_) {
      delete[] (T *)p.first;
      cudaFree(p.second);
    }
    for (auto c : cpu_ptr_set_) {
      delete[] (T *)c;
    }
  }

  //   template<typename T>
  // using Input = const std::vector<T*>&;

  // template<typename T>
  // using Output = const std::vector<T*>&;

  // template<typename T>
  // using Shape_Info = const std::array<T , 3>&;

  // std::function<void(Input<Element>, Output<Element>, Shape_Info<Element>)> =
  // [](Input in, Output out, Shape_Info shape){
  //   int m = shape[0];
  //   int n = shape[1];
  //   int k = shape[2];

  // };
};
