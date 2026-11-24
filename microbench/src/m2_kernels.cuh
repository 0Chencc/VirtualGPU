#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// 1. MMA throughput vs warps per SM (mma.sync.aligned.m16n8k16)
// ---------------------------------------------------------------------------

template <bool IS_BF16>
__global__ void mma_m16n8k16_kernel(float* __restrict__ sink, int iters) {
  uint32_t a0 = 0x3c003c00u + (threadIdx.x & 7u);
  uint32_t a1 = 0x3c003c00u;
  uint32_t a2 = 0x3c003c00u;
  uint32_t a3 = 0x3c003c00u;
  uint32_t b0 = 0x3c003c00u;
  uint32_t b1 = 0x3c003c00u;

  // 4 independent accumulators per warp to hide MMA pipeline latency.
  float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f;
  float c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f;
  float c20 = 0.f, c21 = 0.f, c22 = 0.f, c23 = 0.f;
  float c30 = 0.f, c31 = 0.f, c32 = 0.f, c33 = 0.f;

  for (int i = 0; i < iters; ++i) {
    if constexpr (!IS_BF16) {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c00), "+f"(c01), "+f"(c02), "+f"(c03)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c10), "+f"(c11), "+f"(c12), "+f"(c13)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c20), "+f"(c21), "+f"(c22), "+f"(c23)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c30), "+f"(c31), "+f"(c32), "+f"(c33)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    } else {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c00), "+f"(c01), "+f"(c02), "+f"(c03)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c10), "+f"(c11), "+f"(c12), "+f"(c13)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c20), "+f"(c21), "+f"(c22), "+f"(c23)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c30), "+f"(c31), "+f"(c32), "+f"(c33)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  }

  float sum = c00 + c01 + c02 + c03 + c10 + c11 + c12 + c13 +
              c20 + c21 + c22 + c23 + c30 + c31 + c32 + c33;
  if (sum == -999999.0f) {
    sink[blockIdx.x * blockDim.x + threadIdx.x] = sum;
  }
}

__global__ void fp32_fma_kernel(float* __restrict__ sink, int iters) {
  float a = 1.0001f + 1e-6f * threadIdx.x;
  float b = 0.9999f;
  float r0 = 0.1f, r1 = 0.2f, r2 = 0.3f, r3 = 0.4f;
  float r4 = 0.5f, r5 = 0.6f, r6 = 0.7f, r7 = 0.8f;
  for (int i = 0; i < iters; ++i) {
    r0 = fmaf(r0, a, b);
    r1 = fmaf(r1, a, b);
    r2 = fmaf(r2, a, b);
    r3 = fmaf(r3, a, b);
    r4 = fmaf(r4, a, b);
    r5 = fmaf(r5, a, b);
    r6 = fmaf(r6, a, b);
    r7 = fmaf(r7, a, b);
  }
  float s = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;
  if (s == -999999.0f) {
    sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
  }
}

static int run_mma(int device, int samples, int warmup) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  const int sm_count = prop.multiProcessorCount;
  const int max_warps_per_block = prop.maxThreadsPerBlock / prop.warpSize;
  const int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / prop.warpSize;

  std::vector<int> warp_counts;
  for (int w : {1, 2, 4, 8, 16, 32}) {
    if (w <= max_warps_per_block && w <= max_warps_per_sm) {
      warp_counts.push_back(w);
    }
  }

  float* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(sm_count) * prop.maxThreadsPerBlock * sizeof(float)));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  auto measure_mma_variant = [&](const char* key, bool is_bf16) {
    std::printf("    \"%s\": [", key);
    for (size_t wi = 0; wi < warp_counts.size(); ++wi) {
      int warps = warp_counts[wi];
      int threads = warps * 32;
      // Scale iterations so each launch takes ~1-3 ms regardless of warp count.
      int iters = std::max(4000, 32000 / std::min(warps, 8));
      double flops_per_sm = static_cast<double>(warps) * iters * 4.0 * 4096.0;

      for (int w = 0; w < warmup; ++w) {
        if (!is_bf16) {
          mma_m16n8k16_kernel<false><<<sm_count, threads>>>(d_sink, iters);
        } else {
          mma_m16n8k16_kernel<true><<<sm_count, threads>>>(d_sink, iters);
        }
      }
      CUDA_CHECK(cudaDeviceSynchronize());

      std::vector<double> tflops_samples;
      tflops_samples.reserve(samples);
      for (int s = 0; s < samples; ++s) {
        CUDA_CHECK(cudaEventRecord(ev_start));
        if (!is_bf16) {
          mma_m16n8k16_kernel<false><<<sm_count, threads>>>(d_sink, iters);
        } else {
          mma_m16n8k16_kernel<true><<<sm_count, threads>>>(d_sink, iters);
        }
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        double tflops_per_sm = (flops_per_sm / 1e12) / (static_cast<double>(ms) / 1e3);
        tflops_samples.push_back(tflops_per_sm);
      }

      if (wi > 0) std::printf(",");
      std::printf("\n      {\"mma_warps_per_sm\": %d, \"tflops_per_sm_samples\": [", warps);
      for (size_t k = 0; k < tflops_samples.size(); ++k) {
        if (k > 0) std::printf(", ");
        std::printf("%.4f", tflops_samples[k]);
      }
      std::printf("]}");
    }
    std::printf("\n    ]");
  };

  // Measure FP32 FMA total TFLOP/s (16 warps/SM, 8 FMAs = 16 FLOPs per thread per iter).
  int fma_warps = std::min(16, max_warps_per_block);
  int fma_threads = fma_warps * 32;
  int fma_iters = 40000;
  double fma_total_flops = static_cast<double>(sm_count) * fma_threads * fma_iters * 16.0;
  for (int w = 0; w < warmup; ++w) {
    fp32_fma_kernel<<<sm_count, fma_threads>>>(d_sink, fma_iters);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> fp32_samples;
  fp32_samples.reserve(samples);
  for (int s = 0; s < samples; ++s) {
    CUDA_CHECK(cudaEventRecord(ev_start));
    fp32_fma_kernel<<<sm_count, fma_threads>>>(d_sink, fma_iters);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    fp32_samples.push_back((fma_total_flops / 1e12) / (static_cast<double>(ms) / 1e3));
  }

  std::printf("{\n  \"sm_count\": %d,\n  \"samples\": %d,\n  \"mma\": {\n", sm_count, samples);
  measure_mma_variant("mma.sync:f16:f16:f32:16x8x16", false);
  std::printf(",\n");
  measure_mma_variant("mma.sync:bf16:bf16:f32:16x8x16", true);
  std::printf("\n  },\n  \"fp32_fma_tflops_samples\": [");
  for (size_t k = 0; k < fp32_samples.size(); ++k) {
    if (k > 0) std::printf(", ");
    std::printf("%.4f", fp32_samples[k]);
  }
  std::printf("]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}

// ---------------------------------------------------------------------------
// 2. L2 bandwidth vs active SMs & L2 effective capacity working-set sweep
// ---------------------------------------------------------------------------

static __device__ __forceinline__ uint4 ld_cg_u4(const uint4* ptr) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr));
  return v;
}

static __device__ __forceinline__ void st_cg_u4(uint4* ptr, uint4 v) {
  asm volatile("st.global.cg.v4.u32 [%0], {%1,%2,%3,%4};"
               :
               : "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}

__global__ void l2_read_cg_kernel(const uint4* __restrict__ src,
                                  uint4* __restrict__ sink,
                                  size_t n_vec4,
                                  int passes) {
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  uint32_t acc = 0;
  for (int p = 0; p < passes; ++p) {
    for (size_t i = tid; i < n_vec4; i += stride) {
      uint4 v = ld_cg_u4(src + i);
      acc ^= (v.x + v.y + v.z + v.w);
    }
  }
  if (acc == 0xdeadbeefu) {
    sink[tid & 1023u] = make_uint4(acc, acc, acc, acc);
  }
}

__global__ void l2_write_cg_kernel(uint4* __restrict__ dst,
                                   size_t n_vec4,
                                   int passes,
                                   uint32_t seed) {
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  uint4 v = make_uint4(seed, seed + 1u, seed + 2u, seed + 3u);
  for (int p = 0; p < passes; ++p) {
    for (size_t i = tid; i < n_vec4; i += stride) {
      st_cg_u4(dst + i, v);
    }
  }
}

static int run_l2(int device, int samples, int warmup) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;
  const size_t l2_bytes = std::max<size_t>(prop.l2CacheSize, 4ULL * 1024ULL * 1024ULL);

  // Working set for pure L2 BW: ~25% of L2, clamped to [2 MB, 12 MB].
  size_t l2_ws_bytes = std::max<size_t>(2ULL * 1024ULL * 1024ULL,
                                        std::min<size_t>(12ULL * 1024ULL * 1024ULL, l2_bytes / 4));
  l2_ws_bytes = (l2_ws_bytes / (1024ULL * 1024ULL)) * (1024ULL * 1024ULL);

  // Sweep sizes from ~0.1x to ~2.5x L2 to locate the effective L2 capacity cliff.
  std::vector<size_t> sweep_bytes;
  for (double frac : {0.10, 0.25, 0.50, 0.65, 0.75, 0.85, 0.95, 1.05, 1.20, 1.50, 2.00, 2.50}) {
    size_t sz = static_cast<size_t>((l2_bytes * frac) / (1024ULL * 1024ULL)) * (1024ULL * 1024ULL);
    sz = std::max<size_t>(1ULL * 1024ULL * 1024ULL, sz);
    if (sweep_bytes.empty() || sz > sweep_bytes.back()) {
      sweep_bytes.push_back(sz);
    }
  }
  size_t max_bytes = std::max(l2_ws_bytes, sweep_bytes.back());

  uint4* d_buf = nullptr;
  uint4* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_buf, max_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, 1024 * sizeof(uint4)));
  CUDA_CHECK(cudaMemset(d_buf, 3, max_bytes));

  std::vector<int> targets;
  auto add_t = [&](int s) {
    s = std::max(1, std::min(sm_count, s));
    if (std::find(targets.begin(), targets.end(), s) == targets.end()) targets.push_back(s);
  };
  add_t(1);
  add_t(2);
  add_t(4);
  add_t(sm_count / 4);
  add_t(sm_count / 2);
  add_t(sm_count);
  std::sort(targets.begin(), targets.end());

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  const int threads = 512;
  const size_t l2_vec4 = l2_ws_bytes / sizeof(uint4);

  std::printf("{\n  \"l2_spec_bytes\": %zu,\n  \"l2_bw_ws_bytes\": %zu,\n  \"samples\": %d,\n",
              l2_bytes, l2_ws_bytes, samples);

  // 1) L2 read BW vs active SMs.
  std::printf("  \"read_curve\": [");
  for (size_t ti = 0; ti < targets.size(); ++ti) {
    int target_sms = targets[ti];
    int grid = (target_sms == sm_count) ? (sm_count * 4) : target_sms;
    int passes = (target_sms <= 4) ? 16 : 64;
    double total_bytes = static_cast<double>(l2_ws_bytes) * passes;

    for (int w = 0; w < warmup; ++w) {
      l2_read_cg_kernel<<<grid, threads>>>(d_buf, d_sink, l2_vec4, 4);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> gbps;
    gbps.reserve(samples);
    for (int s = 0; s < samples; ++s) {
      CUDA_CHECK(cudaEventRecord(ev_start));
      l2_read_cg_kernel<<<grid, threads>>>(d_buf, d_sink, l2_vec4, passes);
      CUDA_CHECK(cudaEventRecord(ev_stop));
      CUDA_CHECK(cudaEventSynchronize(ev_stop));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
      gbps.push_back((total_bytes / 1e9) / (static_cast<double>(ms) / 1e3));
    }

    if (ti > 0) std::printf(",");
    std::printf("\n    {\"target_sms\": %d, \"gbps_samples\": [", target_sms);
    for (size_t k = 0; k < gbps.size(); ++k) {
      if (k > 0) std::printf(", ");
      std::printf("%.3f", gbps[k]);
    }
    std::printf("]}");
  }
  std::printf("\n  ],\n");

  // 2) Full-GPU L2 write BW.
  {
    int grid = sm_count * 4;
    int passes = 64;
    double total_bytes = static_cast<double>(l2_ws_bytes) * passes;
    for (int w = 0; w < warmup; ++w) {
      l2_write_cg_kernel<<<grid, threads>>>(d_buf, l2_vec4, 4, static_cast<uint32_t>(w));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("  \"write_full_gbps_samples\": [");
    for (int s = 0; s < samples; ++s) {
      CUDA_CHECK(cudaEventRecord(ev_start));
      l2_write_cg_kernel<<<grid, threads>>>(d_buf, l2_vec4, passes, static_cast<uint32_t>(s + 10));
      CUDA_CHECK(cudaEventRecord(ev_stop));
      CUDA_CHECK(cudaEventSynchronize(ev_stop));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
      if (s > 0) std::printf(", ");
      std::printf("%.3f", (total_bytes / 1e9) / (static_cast<double>(ms) / 1e3));
    }
    std::printf("],\n");
  }

  // 3) Working-set sweep for effective L2 capacity.
  std::printf("  \"capacity_sweep\": [");
  const int sweep_grid = sm_count * 4;
  for (size_t si = 0; si < sweep_bytes.size(); ++si) {
    size_t ws = sweep_bytes[si];
    size_t n_vec4 = ws / sizeof(uint4);
    int passes = std::max(4, static_cast<int>((512ULL * 1024ULL * 1024ULL) / ws));
    double total_bytes = static_cast<double>(ws) * passes;

    for (int w = 0; w < 2; ++w) {
      l2_read_cg_kernel<<<sweep_grid, threads>>>(d_buf, d_sink, n_vec4, 2);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> gbps;
    gbps.reserve(5);
    for (int s = 0; s < 5; ++s) {
      CUDA_CHECK(cudaEventRecord(ev_start));
      l2_read_cg_kernel<<<sweep_grid, threads>>>(d_buf, d_sink, n_vec4, passes);
      CUDA_CHECK(cudaEventRecord(ev_stop));
      CUDA_CHECK(cudaEventSynchronize(ev_stop));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
      gbps.push_back((total_bytes / 1e9) / (static_cast<double>(ms) / 1e3));
    }
    std::sort(gbps.begin(), gbps.end());
    double med = gbps[gbps.size() / 2];
    if (si > 0) std::printf(",");
    std::printf("\n    {\"working_set_bytes\": %zu, \"median_gbps\": %.3f}", ws, med);
  }
  std::printf("\n  ]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_buf));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}

// ---------------------------------------------------------------------------
// 3. cp.async bandwidth per SM & shared memory bandwidth per SM
// ---------------------------------------------------------------------------

__global__ void cp_async_bw_kernel(const uint4* __restrict__ src,
                                   uint4* __restrict__ sink,
                                   size_t n_vec4_mask,
                                   int iters) {
  __shared__ uint4 smem[4 * 256];  // 4 stages x 4 KB = 16 KB
  uint32_t tid = threadIdx.x;
  size_t base = (static_cast<size_t>(blockIdx.x) * 256ULL + tid) & n_vec4_mask;
  size_t step = static_cast<size_t>(gridDim.x) * 256ULL;

  for (int i = 0; i < iters; ++i) {
    uint32_t stage = static_cast<uint32_t>(i & 3);
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem[stage * 256 + tid]));
    const void* gptr = src + ((base + static_cast<size_t>(i) * step) & n_vec4_mask);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gptr));
    asm volatile("cp.async.commit_group;\n" ::);
    if (i >= 3) {
      asm volatile("cp.async.wait_group 2;\n" ::);
    }
  }
  asm volatile("cp.async.wait_all;\n" ::);
  __syncthreads();
  uint4 v = smem[tid];
  if (v.x == 0xdeadbeefu && v.y == 0xdeadbeefu) {
    sink[tid] = v;
  }
}

__global__ void smem_bw_kernel(uint4* __restrict__ sink, int iters, uint32_t seed) {
  // 512 threads (16 warps/SM) x 4 independent 128-bit loads per iter, 32 KB smem, read-only loop.
  // Rotate the byte offset across 4 distinct slots per iteration so ptxas cannot hoist loads out of the loop.
  __shared__ uint4 smem[512 * 4];
  uint32_t tid = threadIdx.x;
  smem[tid] = make_uint4(tid + seed, tid + 1u, tid + 2u, tid + 3u);
  smem[tid + 512] = make_uint4(tid + 4u + seed, tid + 5u, tid + 6u, tid + 7u);
  smem[tid + 1024] = make_uint4(tid + 8u + seed, tid + 9u, tid + 10u, tid + 11u);
  smem[tid + 1536] = make_uint4(tid + 12u + seed, tid + 13u, tid + 14u, tid + 15u);
  __syncthreads();

  uint32_t smem_base = static_cast<uint32_t>(__cvta_generic_to_shared(&smem[0]));
  uint32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

  for (int i = 0; i < iters; ++i) {
    uint32_t idx0 = (tid + static_cast<uint32_t>(i)) & 511u;
    uint32_t a0 = smem_base + (idx0 << 4);
    uint32_t a1 = a0 + 8192u;
    uint32_t a2 = a0 + 16384u;
    uint32_t a3 = a0 + 24576u;

    uint4 v0, v1, v2, v3;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v0.x), "=r"(v0.y), "=r"(v0.z), "=r"(v0.w)
                 : "r"(a0)
                 : "memory");
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v1.x), "=r"(v1.y), "=r"(v1.z), "=r"(v1.w)
                 : "r"(a1)
                 : "memory");
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v2.x), "=r"(v2.y), "=r"(v2.z), "=r"(v2.w)
                 : "r"(a2)
                 : "memory");
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v3.x), "=r"(v3.y), "=r"(v3.z), "=r"(v3.w)
                 : "r"(a3)
                 : "memory");
    acc0 += v0.x + v1.y + v2.z + v3.w;
    acc1 += v0.y + v1.z + v2.w + v3.x;
    acc2 += v0.z + v1.w + v2.x + v3.y;
    acc3 += v0.w + v1.x + v2.y + v3.z;
  }
  if ((acc0 + acc1 + acc2 + acc3) == 0xdeadbeefu) {
    sink[blockIdx.x * blockDim.x + tid] = make_uint4(acc0, acc1, acc2, acc3);
  }
}

static int run_cp_async(int device, int samples, int warmup) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;

  // Power-of-two L2-resident source buffer (8 MB = 524288 uint4).
  const size_t n_vec4 = 524288ULL;
  const size_t n_vec4_mask = n_vec4 - 1ULL;
  uint4* d_src = nullptr;
  uint4* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_src, n_vec4 * sizeof(uint4)));
  CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(sm_count) * 512 * sizeof(uint4)));
  CUDA_CHECK(cudaMemset(d_src, 5, n_vec4 * sizeof(uint4)));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  // 1) cp.async per-SM BW (1 CTA/SM, 256 threads, 16 B/thread/iter).
  const int cp_iters = 16384;
  const double cp_bytes_per_sm = static_cast<double>(256) * 16.0 * cp_iters;
  for (int w = 0; w < warmup; ++w) {
    cp_async_bw_kernel<<<sm_count, 256>>>(d_src, d_sink, n_vec4_mask, 1024);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> cp_samples;
  cp_samples.reserve(samples);
  for (int s = 0; s < samples; ++s) {
    CUDA_CHECK(cudaEventRecord(ev_start));
    cp_async_bw_kernel<<<sm_count, 256>>>(d_src, d_sink, n_vec4_mask, cp_iters);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    cp_samples.push_back((cp_bytes_per_sm / 1e9) / (static_cast<double>(ms) / 1e3));
  }

  // 2) Shared memory per-SM BW (512 threads = 16 warps/SM, 4 x 16B = 64 B/thread/iter, read-only).
  const int sm_iters = 16384;
  const double sm_bytes_per_sm = static_cast<double>(512) * 64.0 * sm_iters;
  for (int w = 0; w < warmup; ++w) {
    smem_bw_kernel<<<sm_count, 512>>>(d_sink, 1024, static_cast<uint32_t>(w));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> smem_samples;
  smem_samples.reserve(samples);
  for (int s = 0; s < samples; ++s) {
    CUDA_CHECK(cudaEventRecord(ev_start));
    smem_bw_kernel<<<sm_count, 512>>>(d_sink, sm_iters, static_cast<uint32_t>(s + 1));
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    smem_samples.push_back((sm_bytes_per_sm / 1e9) / (static_cast<double>(ms) / 1e3));
  }

  std::printf("{\n  \"sm_count\": %d,\n  \"cp_async_bw_per_sm_gbps_samples\": [", sm_count);
  for (size_t k = 0; k < cp_samples.size(); ++k) {
    if (k > 0) std::printf(", ");
    std::printf("%.3f", cp_samples[k]);
  }
  std::printf("],\n  \"smem_bw_per_sm_gbps_samples\": [");
  for (size_t k = 0; k < smem_samples.size(); ++k) {
    if (k > 0) std::printf(", ");
    std::printf("%.3f", smem_samples[k]);
  }
  std::printf("]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_src));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}

// ---------------------------------------------------------------------------
// 4. Load latency: smem, L2 (unloaded & loaded), DRAM (unloaded & loaded)
// ---------------------------------------------------------------------------

static __device__ __forceinline__ uint64_t ld_cg_u64(const uint64_t* ptr) {
  uint64_t v;
  asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(ptr));
  return v;
}

__global__ void smem_latency_kernel(double* __restrict__ out_ns, int steps) {
  __shared__ uint32_t chain[1024];
  if (threadIdx.x == 0) {
    for (uint32_t i = 0; i < 1024; ++i) {
      chain[i] = (i + 37u) & 1023u;
    }
    uint32_t idx = 0;
    for (int w = 0; w < 256; ++w) idx = chain[idx];
    uint64_t t0 = read_globaltimer();
    for (int i = 0; i < steps; ++i) {
      idx = chain[idx];
    }
    uint64_t t1 = read_globaltimer();
    *out_ns = static_cast<double>(t1 - t0) / static_cast<double>(steps);
    if (idx == 999999u) chain[0] = idx;
  }
}

__global__ void pchase_latency_kernel(const uint64_t* __restrict__ chain,
                                      uint64_t* __restrict__ sink,
                                      double* __restrict__ out_ns,
                                      int steps,
                                      const uint4* __restrict__ bg_buf,
                                      size_t bg_vec4,
                                      volatile int* __restrict__ done_flag) {
  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) {
      uint64_t idx = 0;
      uint64_t t0 = read_globaltimer();
      for (int i = 0; i < steps; ++i) {
        idx = ld_cg_u64(chain + idx);
      }
      uint64_t t1 = read_globaltimer();
      *out_ns = static_cast<double>(t1 - t0) / static_cast<double>(steps);
      if (done_flag) {
        atomicExch(const_cast<int*>(done_flag), 1);
      }
      *sink = idx;
    }
  } else if (bg_buf != nullptr && done_flag != nullptr) {
    size_t tid = static_cast<size_t>(blockIdx.x - 1) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x - 1) * blockDim.x;
    uint32_t acc = 0;
    size_t i = tid % bg_vec4;
    while (*done_flag == 0) {
      for (int k = 0; k < 16; ++k) {
        uint4 v = ld_cg_u4(bg_buf + i);
        acc ^= v.x;
        i += stride;
        if (i >= bg_vec4) i -= bg_vec4;
      }
    }
    if (acc == 0xdeadbeefu) *sink = acc;
  }
}

static std::vector<uint64_t> build_random_chain(size_t bytes, size_t stride_bytes, uint32_t seed) {
  size_t n_u64 = bytes / sizeof(uint64_t);
  size_t step_u64 = stride_bytes / sizeof(uint64_t);
  size_t n_nodes = n_u64 / step_u64;

  std::vector<size_t> perm(n_nodes);
  std::iota(perm.begin(), perm.end(), 0);
  std::mt19937 rng(seed);
  std::shuffle(perm.begin(), perm.end(), rng);

  std::vector<uint64_t> buf(n_u64, 0);
  for (size_t i = 0; i < n_nodes; ++i) {
    size_t cur = perm[i] * step_u64;
    size_t nxt = perm[(i + 1) % n_nodes] * step_u64;
    buf[cur] = static_cast<uint64_t>(nxt);
  }
  return buf;
}

static int run_latency(int device, int samples) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;
  const size_t l2_bytes = std::max<size_t>(prop.l2CacheSize, 4ULL * 1024ULL * 1024ULL);

  const size_t l2_chain_bytes = 4ULL * 1024ULL * 1024ULL;
  const size_t dram_chain_bytes = std::max<size_t>(128ULL * 1024ULL * 1024ULL, l2_bytes * 3ULL);
  const size_t bg_l2_bytes = 8ULL * 1024ULL * 1024ULL;
  const size_t bg_dram_bytes = std::max<size_t>(128ULL * 1024ULL * 1024ULL, l2_bytes * 3ULL);

  auto h_l2_chain = build_random_chain(l2_chain_bytes, 256, 101);
  auto h_dram_chain = build_random_chain(dram_chain_bytes, 256, 202);

  uint64_t* d_l2_chain = nullptr;
  uint64_t* d_dram_chain = nullptr;
  uint4* d_bg_dram = nullptr;
  uint64_t* d_sink = nullptr;
  double* d_out_ns = nullptr;
  int* d_done = nullptr;

  CUDA_CHECK(cudaMalloc(&d_l2_chain, l2_chain_bytes));
  CUDA_CHECK(cudaMalloc(&d_dram_chain, dram_chain_bytes));
  CUDA_CHECK(cudaMalloc(&d_bg_dram, bg_dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_out_ns, sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_done, sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_l2_chain, h_l2_chain.data(), l2_chain_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dram_chain, h_dram_chain.data(), dram_chain_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_bg_dram, 7, bg_dram_bytes));

  auto print_arr = [](const char* name, const std::vector<double>& v, bool comma) {
    std::printf("  \"%s\": [", name);
    for (size_t i = 0; i < v.size(); ++i) {
      if (i > 0) std::printf(", ");
      std::printf("%.2f", v[i]);
    }
    std::printf("]%s\n", comma ? "," : "");
  };

  // 1) Shared memory latency.
  std::vector<double> smem_ns;
  smem_ns.reserve(samples);
  for (int s = 0; s < samples; ++s) {
    smem_latency_kernel<<<1, 32>>>(d_out_ns, 50000);
    CUDA_CHECK(cudaDeviceSynchronize());
    double v = 0.0;
    CUDA_CHECK(cudaMemcpy(&v, d_out_ns, sizeof(double), cudaMemcpyDeviceToHost));
    smem_ns.push_back(v);
  }

  // 2) Unloaded L2 latency (warm L2).
  pchase_latency_kernel<<<1, 32>>>(d_l2_chain, d_sink, d_out_ns, 16384, nullptr, 0, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> l2_ns;
  l2_ns.reserve(samples);
  for (int s = 0; s < samples; ++s) {
    pchase_latency_kernel<<<1, 32>>>(d_l2_chain, d_sink, d_out_ns, 16384, nullptr, 0, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    double v = 0.0;
    CUDA_CHECK(cudaMemcpy(&v, d_out_ns, sizeof(double), cudaMemcpyDeviceToHost));
    l2_ns.push_back(v);
  }

  // 3) Loaded L2 latency (background CTAs hammer 8 MB L2).
  std::vector<double> l2_loaded_ns;
  l2_loaded_ns.reserve(samples);
  for (int s = 0; s < samples; ++s) {
    CUDA_CHECK(cudaMemset(d_done, 0, sizeof(int)));
    pchase_latency_kernel<<<sm_count, 256>>>(d_l2_chain, d_sink, d_out_ns, 8192,
                                             d_bg_dram, bg_l2_bytes / sizeof(uint4), d_done);
    CUDA_CHECK(cudaDeviceSynchronize());
    double v = 0.0;
    CUDA_CHECK(cudaMemcpy(&v, d_out_ns, sizeof(double), cudaMemcpyDeviceToHost));
    l2_loaded_ns.push_back(v);
  }

  // 4) Unloaded DRAM latency (flush L2 before each sample).
  std::vector<double> dram_ns;
  dram_ns.reserve(samples);
  for (int s = 0; s < samples; ++s) {
    l2_flush_kernel<<<sm_count * 4, 256>>>(d_bg_dram, bg_dram_bytes / sizeof(uint4), static_cast<uint32_t>(s));
    pchase_latency_kernel<<<1, 32>>>(d_dram_chain, d_sink, d_out_ns, 8192, nullptr, 0, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    double v = 0.0;
    CUDA_CHECK(cudaMemcpy(&v, d_out_ns, sizeof(double), cudaMemcpyDeviceToHost));
    dram_ns.push_back(v);
  }

  // 5) Loaded DRAM latency (background CTAs hammer >=3x L2 DRAM).
  std::vector<double> dram_loaded_ns;
  dram_loaded_ns.reserve(samples);
  for (int s = 0; s < samples; ++s) {
    l2_flush_kernel<<<sm_count * 4, 256>>>(d_bg_dram, bg_dram_bytes / sizeof(uint4), static_cast<uint32_t>(s + 50));
    CUDA_CHECK(cudaMemset(d_done, 0, sizeof(int)));
    pchase_latency_kernel<<<sm_count, 256>>>(d_dram_chain, d_sink, d_out_ns, 4096,
                                             d_bg_dram, bg_dram_bytes / sizeof(uint4), d_done);
    CUDA_CHECK(cudaDeviceSynchronize());
    double v = 0.0;
    CUDA_CHECK(cudaMemcpy(&v, d_out_ns, sizeof(double), cudaMemcpyDeviceToHost));
    dram_loaded_ns.push_back(v);
  }

  std::printf("{\n");
  print_arr("smem_latency_ns_samples", smem_ns, true);
  print_arr("l2_latency_ns_samples", l2_ns, true);
  print_arr("l2_latency_loaded_ns_samples", l2_loaded_ns, true);
  print_arr("dram_latency_ns_samples", dram_ns, true);
  print_arr("dram_latency_loaded_ns_samples", dram_loaded_ns, false);
  std::printf("}\n");

  CUDA_CHECK(cudaFree(d_l2_chain));
  CUDA_CHECK(cudaFree(d_dram_chain));
  CUDA_CHECK(cudaFree(d_bg_dram));
  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_out_ns));
  CUDA_CHECK(cudaFree(d_done));
  return 0;
}

// ---------------------------------------------------------------------------
// 5. Launch overhead & per-CTA launch latency
// ---------------------------------------------------------------------------

__global__ void empty_kernel() {}

static int run_launch(int device, int samples, int warmup) {
  (void)device;
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  const int batch = 500;
  const int large_grid = 4096;

  // Capture two CUDA Graphs (1-CTA and large-grid) to measure device-side launch overhead
  // matching Agent 5's graph + event timing convention.
  cudaGraph_t graph_small = nullptr, graph_large = nullptr;
  cudaGraphExec_t exec_small = nullptr, exec_large = nullptr;

  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < batch; ++i) empty_kernel<<<1, 1, 0, stream>>>();
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph_small));
  CUDA_CHECK(cudaGraphInstantiate(&exec_small, graph_small, nullptr, nullptr, 0));

  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < batch; ++i) empty_kernel<<<large_grid, 1, 0, stream>>>();
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph_large));
  CUDA_CHECK(cudaGraphInstantiate(&exec_large, graph_large, nullptr, nullptr, 0));

  for (int w = 0; w < warmup; ++w) {
    CUDA_CHECK(cudaGraphLaunch(exec_small, stream));
    CUDA_CHECK(cudaGraphLaunch(exec_large, stream));
    for (int i = 0; i < 50; ++i) empty_kernel<<<1, 1, 0, stream>>>();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<double> device_launch_us;
  std::vector<double> host_launch_us;
  std::vector<double> per_cta_ns;
  device_launch_us.reserve(samples);
  host_launch_us.reserve(samples);
  per_cta_ns.reserve(samples);

  for (int s = 0; s < samples; ++s) {
    // 1) Device-side CUDA Graph launch overhead (1 CTA).
    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    CUDA_CHECK(cudaGraphLaunch(exec_small, stream));
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms_dev_small = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_dev_small, ev_start, ev_stop));
    device_launch_us.push_back((static_cast<double>(ms_dev_small) * 1e3) / batch);

    // 2) Device-side CUDA Graph per-CTA launch slope (4096 CTAs vs 1 CTA).
    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    CUDA_CHECK(cudaGraphLaunch(exec_large, stream));
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms_dev_large = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_dev_large, ev_start, ev_stop));
    double diff_ns = ((static_cast<double>(ms_dev_large) - static_cast<double>(ms_dev_small)) * 1e6) / batch;
    per_cta_ns.push_back(std::max(0.05, diff_ns / static_cast<double>(large_grid - 1)));

    // 3) Host-submitted stream batch launch overhead (includes WSL/driver submission).
    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    for (int i = 0; i < batch; ++i) empty_kernel<<<1, 1, 0, stream>>>();
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms_host = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_host, ev_start, ev_stop));
    host_launch_us.push_back((static_cast<double>(ms_host) * 1e3) / batch);
  }

  auto print_vec = [](const char* key, const std::vector<double>& v, bool comma) {
    std::printf("  \"%s\": [", key);
    for (size_t i = 0; i < v.size(); ++i) {
      if (i > 0) std::printf(", ");
      std::printf("%.3f", v[i]);
    }
    std::printf("]%s\n", comma ? "," : "");
  };

  std::printf("{\n  \"batch\": %d,\n  \"large_grid\": %d,\n", batch, large_grid);
  print_vec("device_launch_overhead_us_samples", device_launch_us, true);
  print_vec("host_launch_overhead_us_samples", host_launch_us, true);
  print_vec("per_cta_launch_latency_ns_samples", per_cta_ns, false);
  std::printf("}\n");

  CUDA_CHECK(cudaGraphExecDestroy(exec_small));
  CUDA_CHECK(cudaGraphExecDestroy(exec_large));
  CUDA_CHECK(cudaGraphDestroy(graph_small));
  CUDA_CHECK(cudaGraphDestroy(graph_large));
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
