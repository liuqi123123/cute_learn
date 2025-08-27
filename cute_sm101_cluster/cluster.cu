#include "../include/cuda_helper.h"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"

#include "cutlass/cluster_launch.hpp"
#include "cutlass/trace.h"
#include "cutlass/arch/arch.h"
#include "cutlass/array.h"
#include <cooperative_groups.h>

using Element = float;
using namespace cute;

  static const int cluster_num = 4;
  static const int M = 8;
  static const int all_M = M*cluster_num;
  static const int N = 8;
class Kernel {
 public:
  static const int MaxThreadsPerBlock = 128;
  static const int MinBlocksPerMultiprocessor = 1;
  struct Params {
    Element* A;
    Element* B;
  };
  struct SharedStorage{
    cutlass::AlignedArray<Element, M*N> mem;
  };
  __device__ void operator() (const Params & parmas, char* s){
    SharedStorage& s0 = *reinterpret_cast<SharedStorage*>(s);
    Element* smem = reinterpret_cast<Element*>(&s0.mem);
    namespace cg = cooperative_groups;
    // extern __shared__ Element smem[];
    int tid = cg::this_grid().thread_rank();
  // Cluster initialization, size and calculating local bin offsets.
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int clusterBlockRank = cluster.block_rank();
    int cluster_size = cluster.dim_blocks().x;

    if (threadIdx.x == 0 && blockIdx.x == 1) {
     // printf("clusterBlockRank:%d\n", clusterBlockRank);
      // printf("cluster_size:%d\n", cluster_size);
    }

    for (int i = 0;i  < M; ++i) {
      for (int j = 0; j < N ; ++j) {
        smem[N * i + j] = blockIdx.x;
      }
    }

    float *dst_smem = cluster.map_shared_rank(smem, 2);
    for (int i = 0;i  < M; ++i) {
      for (int j = 0; j < N ; ++j) {
        // printf("blockidx:%d, %f ", blockIdx.x,dst_smem[i * N + j]);
        dst_smem[M * i + j] = blockIdx.x;
      }
      // printf("\n");
    }
    __syncthreads();

    Element* target_output = parmas.B + blockIdx.x * M * N;

    for (int i = 0;i  < M; ++i) {
      for (int j = 0; j < N ; ++j) {
        target_output[N * i + j] = smem[N * i + j];
      }
    }
    cluster.sync();
  }

};



int main() {
  MemHelper<Element> helper;
  auto A = helper.GetCpuGpuBuffer(all_M*N);
  auto B = helper.GetCpuGpuBuffer(all_M*N);
  dim3 grid{cluster_num, 1, 1};
  dim3 block{1, 1, 1};
  dim3 cluster{cluster_num, 1, 1};
  using Params = typename Kernel::Params;
  Params params{(Element*)A.second, (Element*)B.second};
  void* kernel_params[] = {&params};
  void const* kernel = (void const*) cutlass::device_kernel<Kernel>;
  uint32_t smem_s = M*N*sizeof(Element);
  // uint32_t smem_s = 224000;
    if (smem_s >= (48 << 10)) {
    cudaError_t err = cudaFuncSetAttribute(
        cutlass::device_kernel<Kernel>,
        cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, smem_s);
    if (err != cudaError_t::cudaSuccess) {
      std::cout << "initialize error: " << cudaGetErrorString(err) << "{\n"
                << "    shared memory size: " << smem_s << "Bytes\n"
                << "}" << std::endl;
      exit(-1);
    }
  }
  cudaStream_t stream = nullptr;
  cutlass::Status launch_result;
  launch_result = cutlass::ClusterLauncher::launch(grid, cluster, block, smem_s, stream, kernel, kernel_params);
  cudaDeviceSynchronize();
  cudaError_t result = cudaGetLastError();

  helper.SyncGpuToCpu(B);
  Element* cpu_b = static_cast<Element*>(B.first);
   for (int i = 0 ; i < all_M; ++i) {
     for (int j = 0; j < N; ++j) {
       printf("%f ", cpu_b[i * N + j]);
     }
     printf("\n");
   }
  if (cudaSuccess == result && cutlass::Status::kSuccess == launch_result) {
      return 0;
  }
}
