#pragma once
#include "cuda_runtime.h"
#include "cute/util/print.hpp"
#include "cutlass/half.h"
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

#define PRINT(STR)                                                             \
  do {                                                                         \
    printf("%s:\n", #STR);                                                     \
    print(STR);                                                                \
    printf("\n\n");                                                            \
  } while (0)

#define PRINT_LAYOUT(STR)                                                      \
  do {                                                                         \
    printf("%s:\n", #STR);                                                     \
    print_layout(STR);                                                         \
    printf("\n\n");                                                            \
  } while (0)

#define PRINT_TENSOR(STR)                                                      \
  do {                                                                         \
    printf("%s:\n", #STR);                                                     \
    print_tensor(STR);                                                         \
    printf("\n\n");                                                            \
  } while (0)

enum class InitialType {
  Random,
  AllOne,
  AllZero,
  Identity,
  Increment,
  PaddingDim0,

};

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
  CG_PTR GetCpuGpuBuffer(size_t size, InitialType type = InitialType::Random,
                         int padding_start_length = 0,
                         int padding_end_length = 0) {
    T *cpu_ptr = new T[size]{static_cast<T>(0)};

    if (type == InitialType::Random) {
      if constexpr (std::is_same_v<bool, T>) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution d(0.5);
        for (size_t i = 0; i < size; ++i) {
          cpu_ptr[i] = d(gen);
        }
      } else if (std::is_same_v<int8_t, T>) {
        std::random_device rd;
        std::mt19937 gen(rd());
        // 使用 std::numeric_limits 来获得 int8_t 的最小值和最大值
        std::uniform_int_distribution<int> d(
            std::numeric_limits<int8_t>::min(),
            std::numeric_limits<int8_t>::max());
        for (size_t i = 0; i < size; ++i) {
          cpu_ptr[i] = static_cast<int8_t>(d(gen));
        }
      } else {
        for (size_t i = 0; i < size; ++i) {
          cpu_ptr[i] = (T)distr_(eng_);
        }
      }
    } else if (type == InitialType::AllOne) {
      for (size_t i = 0; i < size; ++i) {
        cpu_ptr[i] = (T)1;
      }
    } else if (type == InitialType::AllZero) {
      for (size_t i = 0; i < size; ++i) {
        cpu_ptr[i] = (T)0;
      }
    }

    else if (type == InitialType::Identity) {
      double sqrtNum = sqrt(size);
      assert(sqrtNum == floor(sqrtNum) &&
             "size must be the square of an integer");
      int H = sqrtNum;
      for (int i = 0; i < H; ++i) {
        for (int j = 0; j < H; ++j) {
          if (i == j) {
            cpu_ptr[i * H + j] = (T)1;
          } else {
            cpu_ptr[i * H + j] = (T)0;
          }
        }
      }
    } else if (type == InitialType::Increment) {
      for (size_t i = 0; i < size; ++i) {
        // cpu_ptr[i] = (float)(i % 2048) * 0.001;
        cpu_ptr[i] = (float)i * 0.001;
      }
    } else if (type == InitialType::PaddingDim0) {
      if constexpr (std::is_same_v<bool, T>) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution d(0.5);
        for (size_t i = 0; i < size; ++i) {
          cpu_ptr[i] = d(gen);
        }
      } else {
        for (size_t i = 0; i < size; ++i) {
          cpu_ptr[i] = (T)distr_(eng_);
        }
      }
      for (size_t i = 0; i < size; ++i) {
        int row = i / padding_end_length;
        int col = i % padding_end_length;
        if (col >= padding_start_length) {
          cpu_ptr[col + padding_end_length * row] = (T)0;
        }
      }
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

  void Regression(T *gt, T *result, bool first_wrong_break = true,
                  bool always_print_ans = false) {
    size_t size = c_size_[(void *)gt];
    using Ts = float;
    bool first_wrong_found = false;

    for (int i = 0; i < size; ++i) {
      Ts trans_gt = Ts(((T *)gt)[i]);
      Ts trans_res = Ts(((T *)result)[i]);

      float dif = std::abs(((T *)gt)[i] - ((T *)result)[i]);
      float absolute_error = 0.1;

      if constexpr (std::is_same_v<T, int8_t>) {
        absolute_error = 1;
      } else if constexpr (std::is_same_v<T, cutlass::half_t>) {
        absolute_error = 0.05;
      } else {
        absolute_error = 0.001;
      }

      if (std::isnan(trans_gt)) {
        std::cout << "gt idx: " << i << " is nan!" << std::endl;
        break;
      } else if (std::isinf(trans_gt)) {
        std::cout << "gt idx: " << i << " is inf!" << std::endl;
        break;
      }

      if (std::isnan(trans_res)) {
        std::cout << "res idx: " << i << " is nan!" << std::endl;
        break;
      } else if (std::isinf(trans_res)) {
        std::cout << "res idx: " << i << " is inf!" << std::endl;
        break;
      }

      if (always_print_ans || first_wrong_found) {
        std::cout << "idx: " << i << " : "
                  << "gt: " << (Ts)((T *)gt)[i]
                  << " vs res:" << (Ts)((T *)result)[i] << std::endl;
      }
      if (dif > (Ts)absolute_error) {
        first_wrong_found = true;
        std::cout << "idx: " << i << " get wrong: "
                  << "gt: " << (Ts)((T *)gt)[i]
                  << " vs res:" << (Ts)((T *)result)[i] << std::endl;
        // if (first_wrong_break && false == always_print_ans) {
        //   break;
        // }
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
