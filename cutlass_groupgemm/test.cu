#include "../include/cuda_helper.h"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"

#include "cutlass/cluster_launch.hpp"
#include "cutlass/trace.h"
#include "cutlass/arch/arch.h"
#include "cutlass/array.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <unordered_map>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"

int main() {
  using Element = cutlass::half_t;
  int problem_count = 2;

  // int m0 = 128, n0 = 128, k0 = 128;
  // int m1 = 256, n1 = 256, k1 = 256;
  int m0 = 500, n0 = 1280, k0 = 1280;
  int m1 = 500, n1 = 1280, k1 = 1280;
  MemHelper<Element> helper;

  auto gemm0_A = helper.GetCpuGpuBuffer(m0 * k0);
  auto gemm0_B = helper.GetCpuGpuBuffer(n0 * k0);
  auto gemm0_C = helper.GetCpuGpuBuffer(n0 * m0);
  auto gemm1_A = helper.GetCpuGpuBuffer(m1 * k1);
  auto gemm1_B = helper.GetCpuGpuBuffer(n1 * k1);
  auto gemm1_C = helper.GetCpuGpuBuffer(n1 * m1);

  auto gemm_group_A = helper.GetCpuGpuBuffer(m0 * k0 + m1 * k1);
  auto gemm_group_B = helper.GetCpuGpuBuffer(n0 * k0 + n1 * k1);
  auto gemm_group_C = helper.GetCpuGpuBuffer(n0 * m0 + n1 * m1);


  cudaMemcpy(gemm_group_A.second, gemm0_A.second, sizeof(Element) * m0 * k0, cudaMemcpyDeviceToDevice);
  cudaMemcpy(((Element*)gemm_group_A.second + m0 * k0), gemm1_A.second, sizeof(Element) * m1 * k1 ,cudaMemcpyDeviceToDevice);

  cudaMemcpy(gemm_group_B.second, gemm0_B.second, sizeof(Element) * n0 * k0, cudaMemcpyDeviceToDevice);
  cudaMemcpy(((Element*)gemm_group_B.second + n0 * k0), gemm1_B.second, sizeof(Element) * n1 * k1, cudaMemcpyDeviceToDevice);


  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;


  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementA,
    LayoutA,
    cutlass::ComplexTransform::kNone,
    8,
    ElementB,
    LayoutB,
    cutlass::ComplexTransform::kNone,
    8,
    ElementOutput, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        // ElementOutput, 1,
        ElementAccumulator, ElementAccumulator, cutlass::epilogue::thread::ScaleType::Kind::Nothing>,
    // NOTE: Threadblock swizzling is currently not supported by CUTLASS's grouped kernels.
    // This parameter is passed in at present to match the APIs of other kernels. The parameter
    // is unused within the kernel.
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly
    >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

  std::vector<cutlass::gemm::GemmCoord> problem_sizes;

  problem_sizes.reserve(problem_count);
  cutlass::gemm::GemmCoord problem0(m0, n0, k0);
  cutlass::gemm::GemmCoord problem1(m1, n1, k1);

  problem_sizes.push_back(problem0);
  problem_sizes.push_back(problem1);

  int threadblock_count = Gemm::sufficient(problem_sizes.data(), problem_count);

  // Early exit
  if (!threadblock_count) {
    std::cout << "Active CUDA device lacks hardware resources to run CUTLASS Grouped GEMM kernel." << std::endl;
    return -1;
  }

  typename Gemm::EpilogueOutputOp::Params epilogue_op(1, 0);  // alpha  beta

cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problem_sizes_device;
  problem_sizes_device.reset(problem_count);
  problem_sizes_device.copy_from_host(problem_sizes.data());


  std::vector<int64_t> offset_A;
  std::vector<int64_t> offset_B;
  std::vector<int64_t> offset_C;
  std::vector<int64_t> offset_D;

  std::vector<int64_t> lda_host;
  std::vector<int64_t> ldb_host;
  std::vector<int64_t> ldc_host;
  std::vector<int64_t> ldd_host;

  cutlass::DeviceAllocation<int64_t> lda;
  cutlass::DeviceAllocation<int64_t> ldb;
  cutlass::DeviceAllocation<int64_t> ldc;
  cutlass::DeviceAllocation<int64_t> ldd;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementC> block_D;

  cutlass::DeviceAllocation<ElementA *> ptr_A;
  cutlass::DeviceAllocation<ElementB *> ptr_B;
  cutlass::DeviceAllocation<ElementC *> ptr_C;
  cutlass::DeviceAllocation<ElementC *> ptr_D;

    int64_t total_elements_A = 0;
    int64_t total_elements_B = 0;
    int64_t total_elements_C = 0;
    int64_t total_elements_D = 0;

    lda_host.resize(problem_count);
    ldb_host.resize(problem_count);
    ldc_host.resize(problem_count);
    ldd_host.resize(problem_count);

    for (int32_t i = 0; i < problem_count; ++i) {

      auto problem = problem_sizes.at(i);

      lda_host.at(i) = LayoutA::packed({problem.m(), problem.k()}).stride(0);
      ldb_host.at(i) = LayoutB::packed({problem.k(), problem.n()}).stride(0);
      ldc_host.at(i) = LayoutC::packed({problem.m(), problem.n()}).stride(0);
      ldd_host.at(i) = LayoutC::packed({problem.m(), problem.n()}).stride(0);

      offset_A.push_back(total_elements_A);
      offset_B.push_back(total_elements_B);
      offset_C.push_back(total_elements_C);
      offset_D.push_back(total_elements_D);

      int64_t elements_A = problem.m() * problem.k();
      int64_t elements_B = problem.k() * problem.n();
      int64_t elements_C = problem.m() * problem.n();
      int64_t elements_D = problem.m() * problem.n();

      total_elements_A += elements_A;
      total_elements_B += elements_B;
      total_elements_C += elements_C;
      total_elements_D += elements_D;
    }

    lda.reset(problem_count);
    ldb.reset(problem_count);
    ldc.reset(problem_count);
    ldd.reset(problem_count);

    block_A.reset((ElementA*)gemm_group_A.second, m0 * k0 + m1 * k1);
    block_B.reset((ElementB*)gemm_group_B.second, n0 * k0 + n1 * k1);
    // block_C.reset(total_elements_C);
    block_D.reset((ElementC*)gemm_group_C.second, m0 * n0 + m1 * n1);


    lda.copy_from_host(lda_host.data());
    ldb.copy_from_host(ldb_host.data());
    ldc.copy_from_host(ldc_host.data());
    ldd.copy_from_host(ldd_host.data());

    std::vector<ElementA *> ptr_A_host(problem_count);
    std::vector<ElementB *> ptr_B_host(problem_count);
    std::vector<ElementC *> ptr_C_host(problem_count);
    std::vector<ElementC *> ptr_D_host(problem_count);

    for (int32_t i = 0; i < problem_count; ++i) {
      ptr_A_host.at(i) = block_A.get() + offset_A.at(i);
      ptr_B_host.at(i) = block_B.get() + offset_B.at(i);
      ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
      ptr_D_host.at(i) = block_D.get() + offset_D.at(i);
    }

    ptr_A.reset(problem_count);
    ptr_A.copy_from_host(ptr_A_host.data());

    ptr_B.reset(problem_count);
    ptr_B.copy_from_host(ptr_B_host.data());

    ptr_C.reset(problem_count);
    ptr_C.copy_from_host(ptr_C_host.data());

    ptr_D.reset(problem_count);
    ptr_D.copy_from_host(ptr_D_host.data());



    // Configure GEMM arguments
    typename Gemm::Arguments args(
      problem_sizes_device.get(),
      problem_count,
      threadblock_count,
      epilogue_op,
      ptr_A.get(),
      ptr_B.get(),
      ptr_C.get(),
      ptr_D.get(),
      lda.get(),
      ldb.get(),
      ldc.get(),
      ldd.get(),
      problem_sizes.data()
    );

    // Initialize the GEMM object
    Gemm gemm;

    size_t workspace_size = gemm.get_workspace_size(args);
    cutlass::DeviceAllocation<uint8_t> workspace(workspace_size);
    cutlass::Status status ;
    status = gemm.initialize(args, workspace.get());

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize CUTLASS Grouped GEMM kernel." << std::endl;
      return -1;
    }
    // Run the grouped GEMM object
    status = gemm.run();

    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to run CUTLASS Grouped GEMM kernel." << std::endl;
      return -1;
    }

    // Wait for completion
    cudaError_t error = cudaDeviceSynchronize();

    if ( error != cudaSuccess)  {
      std::cerr << "Kernel execution error: " << cudaGetErrorString(error);
      return -1;
    }


  //get cpu gemm0
  for (int i = 0; i < m0; ++i) {
    for (int j = 0; j < n0; ++ j) {
      float sum = 0.0f;
      for (int k = 0; k < k0; ++k) {
        sum += *((Element*)gemm0_A.first + k + i * k0)  * (*((Element*)gemm0_B.first + k + j * k0));
      }
      *((Element*)gemm0_C.first + j + i * n0) = (Element)sum;
    }
  }

  //get cpu gemm1
  for (int i = 0; i < m1; ++i) {
    for (int j = 0; j < n1; ++ j) {
      float sum = 0.0f;
      for (int k = 0; k < k1; ++k) {
        sum += *((Element*)gemm1_A.first + k + i * k1)  * (*((Element*)gemm1_B.first + k + j * k1));
      }
      *((Element*)gemm1_C.first + j + i * n1) = (Element)sum;
    }
  }
  helper.SyncGpuToCpu(gemm_group_C);

  helper.Regression((Element *)gemm0_C.first, (Element*)gemm_group_C.first);
  helper.Regression((Element *)gemm1_C.first, (Element*)gemm_group_C.first + m0 * n0);


}