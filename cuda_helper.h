#pragma once

#include <vector>
#include "cuda_runtime.h"
#include <random>
#include "cutlass/half.h"

#define  PRINT(STR) \
        do { \
          printf("%s:\n", #STR); \
          print(STR); \
          printf("\n"); \
        } while (0)


template<typename T>
class MemHelper{
 struct pair_hash{
  template<typename T1, typename T2>
  size_t operator()(const std::pair<T1, T2>& p) const noexcept{
    auto hash1 = std::hash<T1>{}(p.first);
    auto hash2 = std::hash<T2>{}(p.second);
    return hash1 ^ (hash2 << 1);
  }
 };


 private:
  //std::pair<cpu_ptr, gpu_ptr>
  using CG_PTR = std::pair<void*, void*>;
  std::vector<void*> cpu_ptr_set_;
  std::vector<CG_PTR> cpu_gpu_ptr_set_;
  std::unordered_map<CG_PTR, size_t, pair_hash> cg_size_;
  std::unordered_map<void*, size_t> c_size_;

  std::random_device rd_;
  std::default_random_engine eng_;
  std::uniform_real_distribution<float> distr_;

 public:
  MemHelper() : rd_(), eng_(rd_()), distr_(-1.0f, 1.0f) {

  }
  CG_PTR GetCpuGpuBuffer(size_t size, bool need_initial = true) {
    T* cpu_ptr = new T[size]{static_cast<T>(0)};

    if (need_initial) {
      for (size_t i = 0; i < size; ++i) {
        cpu_ptr[i] = (T)distr_(eng_);
      }
    }

    T* gpu_ptr;
    cudaMalloc((void**)&gpu_ptr, size * sizeof(T));
    cudaMemset((void**)&gpu_ptr, 0, size * sizeof(T));
    cudaMemcpy(gpu_ptr, cpu_ptr, size * sizeof(T), cudaMemcpyHostToDevice);
    cpu_gpu_ptr_set_.emplace_back(std::make_pair((void*)cpu_ptr, (void*)gpu_ptr));

    cg_size_[{cpu_ptr, gpu_ptr}] = size;
    return {cpu_ptr, gpu_ptr};
  }

  T* GetCpuBuffer(size_t size) {
    T* cpu_ptr = new T[size]{static_cast<T>(0)};
    cpu_ptr_set_.push_back(cpu_ptr);
    c_size_[(void*)cpu_ptr] = size;
    return cpu_ptr;
  }

  void SyncGpuToCpu(const CG_PTR& cg) {
    cudaMemcpy(cg.first, cg.second, cg_size_[cg] * sizeof(T), cudaMemcpyDeviceToHost);
  }

  void Regression(T* gt, T* result, bool first_wrong_break = true) {
    size_t size = c_size_[(void*)gt];
    for (int i = 0; i < size; ++i) {
      float dif =std::abs(((T*)gt)[i] - ((T*)result)[i]);

      using Ts = float;
      float absolute_error = 0.1;

      if constexpr (std::is_same_v<T, int8_t>) {
        absolute_error = 1;
      } else if constexpr(std::is_same_v<T, cute::half_t>) {
        absolute_error = 0.5;
      } else {
        absolute_error = 0.1;
      }
      if (dif > (Ts)absolute_error) {
        std::cout << "idx: " << i << " get wrong: "
                  << "gt: " << (Ts)((T*)gt)[i] << " vs " << (Ts)((T*)result)[i]
                  << std::endl;
        if (first_wrong_break) {
          break;
        }
      }
    }


  }


  ~MemHelper() {
    for(auto& p : cpu_gpu_ptr_set_) {
      delete[] (T*)p.first;
      cudaFree(p.second);
    }
    for (auto c : cpu_ptr_set_) {
      delete[] (T*)c;
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
