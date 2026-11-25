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

// ---------------------------------------------------------------------------
// 6. Latency vs achieved bandwidth curves (DRAM & L2) for Little's Law model
// ---------------------------------------------------------------------------

static __device__ __forceinline__ int ld_acq_s32(const volatile int* ptr) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(ptr) : "memory");
  return v;
}

struct BgCtaStats {
  uint64_t bytes;
  uint64_t t_start_ns;
  uint64_t t_end_ns;
};

__global__ void pchase_bw_curve_kernel(const uint64_t* __restrict__ chain,
                                       uint64_t* __restrict__ sink,
                                       double* __restrict__ out_latency_ns,
                                       BgCtaStats* __restrict__ out_bg_stats,
                                       int steps,
                                       const uint4* __restrict__ bg_buf,
                                       size_t bg_vec4,
                                       int bg_active_threads,
                                       volatile int* __restrict__ start_flag,
                                       volatile int* __restrict__ done_flag,
                                       unsigned long long* __restrict__ chunk_counter) {
  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) {
      uint64_t idx = 0;
      for (int w = 0; w < 256; ++w) {
        idx = ld_cg_u64(chain + idx);
      }
      if (start_flag) {
        atomicExch(const_cast<int*>(start_flag), 1);
      }
      uint64_t t0 = read_globaltimer();
      for (int i = 0; i < steps; ++i) {
        idx = ld_cg_u64(chain + idx);
      }
      uint64_t t1 = read_globaltimer();
      if (done_flag) {
        atomicExch(const_cast<int*>(done_flag), 1);
      }
      *out_latency_ns = static_cast<double>(t1 - t0) / static_cast<double>(steps);
      *sink = idx;
    }
  } else if (bg_buf != nullptr && done_flag != nullptr) {
    if (static_cast<int>(threadIdx.x) >= bg_active_threads) return;
    __shared__ size_t s_chunk_base;
    // Each chunk is 32 * bg_active_threads uint4s; CTAs claim disjoint chunks so
    // no CTA reads data recently fetched into L2 by another CTA.
    const size_t chunk_vec4 = 32ULL * static_cast<size_t>(bg_active_threads);
    const size_t raw_chunks = bg_vec4 / chunk_vec4;
    const size_t n_chunks = (raw_chunks > 0ULL) ? raw_chunks : 1ULL;
    uint32_t acc = 0;

    auto claim_and_read = [&]() {
      if (threadIdx.x == 0) {
        unsigned long long cid = atomicAdd(chunk_counter, 1ULL);
        s_chunk_base = (static_cast<size_t>(cid) % n_chunks) * chunk_vec4;
      }
      __syncthreads();
      size_t base = s_chunk_base + threadIdx.x;
      for (int k = 0; k < 32; ++k) {
        uint4 v = ld_cg_u4(bg_buf + base + static_cast<size_t>(k) * static_cast<size_t>(bg_active_threads));
        acc ^= v.x;
      }
    };

    while (ld_acq_s32(start_flag) == 0 && ld_acq_s32(done_flag) == 0) {
      claim_and_read();
    }

    __syncthreads();
    uint64_t bg_t0 = (threadIdx.x == 0) ? read_globaltimer() : 0;
    unsigned long long loops = 0;
    while (ld_acq_s32(done_flag) == 0) {
      claim_and_read();
      ++loops;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      uint64_t bg_t1 = read_globaltimer();
      uint64_t cta_bytes = static_cast<uint64_t>(loops) * chunk_vec4 * 16ULL;
      out_bg_stats[blockIdx.x - 1] = BgCtaStats{cta_bytes, bg_t0, bg_t1};
    }
    if (acc == 0xdeadbeefu) *sink = acc;
  }
}

static int run_latency_curve(int device, int samples) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;
  const size_t l2_bytes = std::max<size_t>(prop.l2CacheSize, 4ULL * 1024ULL * 1024ULL);

  const size_t l2_chain_bytes = 4ULL * 1024ULL * 1024ULL;
  const size_t dram_chain_bytes = std::max<size_t>(128ULL * 1024ULL * 1024ULL, l2_bytes * 3ULL);
  const size_t bg_l2_bytes = 8ULL * 1024ULL * 1024ULL;
  // Use >= 256 MiB (> 7x L2 on RTX 4070) so background DRAM traffic has zero L2 reuse.
  const size_t bg_dram_bytes = std::max<size_t>(256ULL * 1024ULL * 1024ULL, l2_bytes * 6ULL);

  auto h_l2_chain = build_random_chain(l2_chain_bytes, 256, 303);
  auto h_dram_chain = build_random_chain(dram_chain_bytes, 256, 404);

  uint64_t* d_l2_chain = nullptr;
  uint64_t* d_dram_chain = nullptr;
  uint4* d_bg_buf = nullptr;
  uint4* d_sink_vec = nullptr;
  uint64_t* d_sink = nullptr;
  double* d_lat_ns = nullptr;
  BgCtaStats* d_bg_stats = nullptr;
  int* d_start = nullptr;
  int* d_done = nullptr;
  unsigned long long* d_chunk_counter = nullptr;

  int max_bg = std::max(1, sm_count - 1);
  CUDA_CHECK(cudaMalloc(&d_l2_chain, l2_chain_bytes));
  CUDA_CHECK(cudaMalloc(&d_dram_chain, dram_chain_bytes));
  CUDA_CHECK(cudaMalloc(&d_bg_buf, bg_dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink_vec, 1024 * sizeof(uint4)));
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_lat_ns, sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_bg_stats, static_cast<size_t>(max_bg) * sizeof(BgCtaStats)));
  CUDA_CHECK(cudaMalloc(&d_start, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_done, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_chunk_counter, sizeof(unsigned long long)));

  CUDA_CHECK(cudaMemcpy(d_l2_chain, h_l2_chain.data(), l2_chain_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dram_chain, h_dram_chain.data(), dram_chain_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_bg_buf, 9, bg_dram_bytes));

  struct LoadLevel {
    int bg_blocks;
    int bg_threads;
  };
  std::vector<LoadLevel> levels = {
      {0, 0},
      {1, 64},
      {1, 256},
      {std::min(2, max_bg), 256},
      {std::min(4, max_bg), 256},
      {std::min(8, max_bg), 256},
      {std::min(16, max_bg), 256},
      {max_bg, 256},
  };
  std::vector<BgCtaStats> h_bg_stats(static_cast<size_t>(max_bg));

  auto measure_levels = [&](const char* key, bool is_dram, bool comma) {
    const uint64_t* chain = is_dram ? d_dram_chain : d_l2_chain;
    const size_t bg_vec4 = (is_dram ? bg_dram_bytes : bg_l2_bytes) / sizeof(uint4);
    const int steps = is_dram ? 4096 : 8192;

    if (!is_dram) {
      CUDA_CHECK(cudaMemset(d_start, 0, sizeof(int)));
      CUDA_CHECK(cudaMemset(d_done, 0, sizeof(int)));
      CUDA_CHECK(cudaMemset(d_chunk_counter, 0, sizeof(unsigned long long)));
      pchase_bw_curve_kernel<<<1, 32>>>(chain, d_sink, d_lat_ns, d_bg_stats,
                                        4096, nullptr, 0, 0, d_start, d_done, d_chunk_counter);
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::printf("  \"%s\": [", key);
    for (size_t li = 0; li < levels.size(); ++li) {
      int bg_b = levels[li].bg_blocks;
      int bg_t = levels[li].bg_threads;
      int grid = 1 + bg_b;
      int threads = (bg_b == 0) ? 32 : bg_t;

      std::vector<double> bw_samples;
      std::vector<double> lat_samples;
      bw_samples.reserve(samples);
      lat_samples.reserve(samples);

      for (int s = 0; s < samples; ++s) {
        if (is_dram) {
          // Clean read-only L2 flush avoids dirty write-backs during measurement.
          l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_bg_buf, d_sink_vec, bg_dram_bytes / sizeof(uint4), 1);
        }
        CUDA_CHECK(cudaMemset(d_start, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_done, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_chunk_counter, 0, sizeof(unsigned long long)));
        if (bg_b > 0) {
          CUDA_CHECK(cudaMemset(d_bg_stats, 0, static_cast<size_t>(bg_b) * sizeof(BgCtaStats)));
        }

        pchase_bw_curve_kernel<<<grid, threads>>>(
            chain, d_sink, d_lat_ns, d_bg_stats, steps,
            (bg_b > 0) ? d_bg_buf : nullptr, bg_vec4, bg_t, d_start, d_done, d_chunk_counter);
        CUDA_CHECK(cudaDeviceSynchronize());

        double lat_ns = 0.0;
        CUDA_CHECK(cudaMemcpy(&lat_ns, d_lat_ns, sizeof(double), cudaMemcpyDeviceToHost));
        double bw_gbps = 0.0;
        if (bg_b > 0) {
          CUDA_CHECK(cudaMemcpy(h_bg_stats.data(), d_bg_stats,
                                static_cast<size_t>(bg_b) * sizeof(BgCtaStats),
                                cudaMemcpyDeviceToHost));
          // Total background bytes divided by global wall span [min(t_start), max(t_end)].
          uint64_t total_bytes = 0;
          uint64_t min_t0 = UINT64_MAX;
          uint64_t max_t1 = 0;
          for (int b = 0; b < bg_b; ++b) {
            total_bytes += h_bg_stats[b].bytes;
            if (h_bg_stats[b].t_start_ns > 0 && h_bg_stats[b].t_start_ns < min_t0) {
              min_t0 = h_bg_stats[b].t_start_ns;
            }
            if (h_bg_stats[b].t_end_ns > max_t1) {
              max_t1 = h_bg_stats[b].t_end_ns;
            }
          }
          if (max_t1 > min_t0) {
            bw_gbps = static_cast<double>(total_bytes) / static_cast<double>(max_t1 - min_t0);
          }
        }
        bw_samples.push_back(bw_gbps);
        lat_samples.push_back(lat_ns);
      }

      if (li > 0) std::printf(",");
      std::printf("\n    {\"bg_blocks\": %d, \"bg_threads\": %d, \"achieved_bw_gbps_samples\": [",
                  bg_b, bg_t);
      for (size_t k = 0; k < bw_samples.size(); ++k) {
        if (k > 0) std::printf(", ");
        std::printf("%.3f", bw_samples[k]);
      }
      std::printf("], \"latency_ns_samples\": [");
      for (size_t k = 0; k < lat_samples.size(); ++k) {
        if (k > 0) std::printf(", ");
        std::printf("%.2f", lat_samples[k]);
      }
      std::printf("]}");
    }
    std::printf("\n  ]%s\n", comma ? "," : "");
  };

  std::printf("{\n  \"samples\": %d,\n", samples);
  measure_levels("dram_latency_by_bw", true, true);
  measure_levels("l2_latency_by_bw", false, false);
  std::printf("}\n");

  CUDA_CHECK(cudaFree(d_l2_chain));
  CUDA_CHECK(cudaFree(d_dram_chain));
  CUDA_CHECK(cudaFree(d_bg_buf));
  CUDA_CHECK(cudaFree(d_sink_vec));
  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_lat_ns));
  CUDA_CHECK(cudaFree(d_bg_stats));
  CUDA_CHECK(cudaFree(d_start));
  CUDA_CHECK(cudaFree(d_done));
  CUDA_CHECK(cudaFree(d_chunk_counter));
  return 0;
}

// ---------------------------------------------------------------------------
// 7. L2 rescan hit rate vs working-set size (0.25x to 4.0x L2)
// ---------------------------------------------------------------------------

static int run_l2_rescan(int device, int samples) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;
  const size_t l2_bytes = std::max<size_t>(prop.l2CacheSize, 4ULL * 1024ULL * 1024ULL);

  // Dense sampling around the L2 thrashing cliff (0.96x - 1.25x L2).
  std::vector<double> ratios = {
      0.25, 0.50, 0.75, 0.90, 0.96, 1.00, 1.04, 1.08, 1.12, 1.16, 1.25, 1.50, 2.00, 3.00, 4.00};
  std::vector<size_t> ws_list;
  for (double r : ratios) {
    size_t sz = static_cast<size_t>((l2_bytes * r) / (1024ULL * 1024ULL)) * (1024ULL * 1024ULL);
    sz = std::max<size_t>(1ULL * 1024ULL * 1024ULL, sz);
    if (ws_list.empty() || sz > ws_list.back()) {
      ws_list.push_back(sz);
    }
  }
  const size_t max_ws = ws_list.back();
  const size_t flush_bytes = std::max<size_t>(l2_bytes * 3ULL, 128ULL * 1024ULL * 1024ULL);

  uint4* d_buf = nullptr;
  uint4* d_flush = nullptr;
  uint4* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_buf, max_ws));
  CUDA_CHECK(cudaMalloc(&d_flush, flush_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, 1024 * sizeof(uint4)));
  CUDA_CHECK(cudaMemset(d_buf, 11, max_ws));
  CUDA_CHECK(cudaMemset(d_flush, 12, flush_bytes));

  cudaEvent_t ev0, ev1, ev2;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));
  CUDA_CHECK(cudaEventCreate(&ev2));

  const int grid = sm_count * 4;
  const int threads = 512;
  const size_t flush_vec4 = flush_bytes / sizeof(uint4);

  std::printf("{\n  \"l2_spec_bytes\": %zu,\n  \"samples\": %d,\n  \"points\": [",
              l2_bytes, samples);

  for (size_t wi = 0; wi < ws_list.size(); ++wi) {
    size_t ws = ws_list[wi];
    size_t ws_vec4 = ws / sizeof(uint4);

    // Warmup (read-only flush leaves L2 clean with disjoint data, avoiding dirty write-backs).
    l2_read_cg_kernel<<<grid, threads>>>(d_flush, d_sink, flush_vec4, 1);
    l2_read_cg_kernel<<<grid, threads>>>(d_buf, d_sink, ws_vec4, 1);
    l2_read_cg_kernel<<<grid, threads>>>(d_buf, d_sink, ws_vec4, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> p1_gbps, p2_gbps;
    p1_gbps.reserve(samples);
    p2_gbps.reserve(samples);

    for (int s = 0; s < samples; ++s) {
      l2_read_cg_kernel<<<grid, threads>>>(d_flush, d_sink, flush_vec4, 1);
      CUDA_CHECK(cudaEventRecord(ev0));
      l2_read_cg_kernel<<<grid, threads>>>(d_buf, d_sink, ws_vec4, 1);
      CUDA_CHECK(cudaEventRecord(ev1));
      l2_read_cg_kernel<<<grid, threads>>>(d_buf, d_sink, ws_vec4, 1);
      CUDA_CHECK(cudaEventRecord(ev2));
      CUDA_CHECK(cudaEventSynchronize(ev2));

      float ms1 = 0.0f, ms2 = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms1, ev0, ev1));
      CUDA_CHECK(cudaEventElapsedTime(&ms2, ev1, ev2));
      p1_gbps.push_back((static_cast<double>(ws) / 1e9) / (static_cast<double>(ms1) / 1e3));
      p2_gbps.push_back((static_cast<double>(ws) / 1e9) / (static_cast<double>(ms2) / 1e3));
    }

    if (wi > 0) std::printf(",");
    std::printf("\n    {\"working_set_bytes\": %zu, \"ratio_to_l2\": %.4f, \"pass1_bw_gbps_samples\": [",
                ws, static_cast<double>(ws) / static_cast<double>(l2_bytes));
    for (size_t k = 0; k < p1_gbps.size(); ++k) {
      if (k > 0) std::printf(", ");
      std::printf("%.3f", p1_gbps[k]);
    }
    std::printf("], \"pass2_bw_gbps_samples\": [");
    for (size_t k = 0; k < p2_gbps.size(); ++k) {
      if (k > 0) std::printf(", ");
      std::printf("%.3f", p2_gbps[k]);
    }
    std::printf("]}");
  }
  std::printf("\n  ]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev0));
  CUDA_CHECK(cudaEventDestroy(ev1));
  CUDA_CHECK(cudaEventDestroy(ev2));
  CUDA_CHECK(cudaFree(d_buf));
  CUDA_CHECK(cudaFree(d_flush));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}

// ---------------------------------------------------------------------------
// 8. MLP curves: delivered BW vs bytes in flight (cp.async & ld.global.v4)
// ---------------------------------------------------------------------------

template <int STAGES, int SUB_PER_STAGE>
__global__ void mlp_cp_async_kernel(const uint4* __restrict__ src,
                                    uint4* __restrict__ sink,
                                    size_t n_vec4_mask,
                                    int steps) {
  // Disjoint shared memory slot per (stage, sub) so every cp.async writes a unique address.
  extern __shared__ uint4 smem[];
  uint32_t tid = threadIdx.x;
  size_t idx = (static_cast<size_t>(blockIdx.x) * 256ULL + tid) & n_vec4_mask;
  size_t stride = static_cast<size_t>(gridDim.x) * 256ULL;

  auto issue_group = [&](int slot) {
    #pragma unroll
    for (int sub = 0; sub < SUB_PER_STAGE; ++sub) {
      size_t smem_off = (static_cast<size_t>(slot) * SUB_PER_STAGE + sub) * 256ULL + tid;
      uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem[smem_off]));
      const void* gptr = src + idx;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gptr) : "memory");
      idx = (idx + stride) & n_vec4_mask;
    }
    asm volatile("cp.async.commit_group;\n" :: : "memory");
  };

  #pragma unroll
  for (int s = 0; s < STAGES; ++s) {
    issue_group(s);
  }

  for (int step = STAGES; step < steps; ++step) {
    if constexpr (STAGES == 1) {
      asm volatile("cp.async.wait_group 0;\n" ::);
    } else if constexpr (STAGES == 2) {
      asm volatile("cp.async.wait_group 1;\n" ::);
    } else if constexpr (STAGES == 4) {
      asm volatile("cp.async.wait_group 3;\n" ::);
    }
    issue_group(step % STAGES);
  }
  asm volatile("cp.async.wait_all;\n" ::);
  __syncthreads();
  uint4 v = smem[tid];
  if (v.x == 0xdeadbeefu && v.y == 0xdeadbeefu) {
    sink[tid] = v;
  }
}

template <int UNROLL>
__global__ void mlp_ld_v4_kernel(const uint4* __restrict__ src,
                                 uint4* __restrict__ sink,
                                 size_t n_vec4_mask,
                                 int outer_iters) {
  uint32_t tid = threadIdx.x;
  size_t idx = (static_cast<size_t>(blockIdx.x) * 256ULL + tid) & n_vec4_mask;
  size_t stride = static_cast<size_t>(gridDim.x) * 256ULL;
  uint32_t acc = 0;

  for (int it = 0; it < outer_iters; ++it) {
    uint32_t round_xor = 0;
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      uint4 v = ld_cg_u4(src + ((idx + static_cast<size_t>(u) * stride) & n_vec4_mask));
      round_xor ^= (v.x ^ v.y ^ v.z ^ v.w);
    }
    acc ^= round_xor;
    // Data dependency forces all UNROLL loads to return before issuing the next batch,
    // holding exactly UNROLL * 256 * 16 B in flight per CTA.
    size_t dep = (round_xor == 0xdeadbeefu) ? 1ULL : 0ULL;
    idx = (idx + static_cast<size_t>(UNROLL) * stride + dep) & n_vec4_mask;
  }
  if (acc == 0xdeadbeefu) {
    sink[tid] = make_uint4(acc, acc, acc, acc);
  }
}

static int run_mlp(int device, int samples, int warmup) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;

  // Power-of-two buffers for cheap bitmask wrap:
  // DRAM = 256 MiB (> 7x L2 on RTX 4070), L2 = 8 MiB (< 0.25x L2).
  const size_t dram_bytes = 256ULL * 1024ULL * 1024ULL;
  const size_t l2_bytes = 8ULL * 1024ULL * 1024ULL;
  const size_t dram_mask = (dram_bytes / sizeof(uint4)) - 1ULL;
  const size_t l2_mask = (l2_bytes / sizeof(uint4)) - 1ULL;

  uint4* d_dram = nullptr;
  uint4* d_flush = nullptr;
  uint4* d_l2 = nullptr;
  uint4* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dram, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_flush, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_l2, l2_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, 1024 * sizeof(uint4)));
  CUDA_CHECK(cudaMemset(d_dram, 13, dram_bytes));
  CUDA_CHECK(cudaMemset(d_flush, 14, dram_bytes));
  CUDA_CHECK(cudaMemset(d_l2, 15, l2_bytes));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  struct MlpPoint {
    int ctas;
    int mult;  // Total 4 KB units in flight per CTA (1, 2, 4, 8, 12, 16)
  };
  // Dense log-spaced coverage across 32-256 KB (>= 4 points per octave) and extended to >= 6 MB
  // (up to 8.83 MB on 46-SM RTX 4070 with 138 concurrent CTAs = 3 CTAs/SM).
  std::vector<MlpPoint> raw_pts = {
      {1, 1},   // 4 KB
      {2, 1},   // 8 KB
      {4, 1},   // 16 KB
      {8, 1},   // 32 KB
      {10, 1},  // 40 KB
      {12, 1},  // 48 KB
      {14, 1},  // 56 KB
      {16, 1},  // 64 KB
      {19, 1},  // 76 KB
      {22, 1},  // 88 KB
      {26, 1},  // 104 KB
      {std::min(32, sm_count), 1},  // 128 KB
      {std::min(38, sm_count), 1},  // 152 KB
      {sm_count, 1},                // 184 KB
      {std::min(28, sm_count), 2},  // 224 KB
      {std::min(32, sm_count), 2},  // 256 KB
      {sm_count, 2},                // 368 KB
      {std::min(32, sm_count), 4},  // 512 KB
      {sm_count, 4},                // 736 KB
      {sm_count * 2, 4},            // 1472 KB
      {sm_count * 3, 4},            // 2208 KB
      {sm_count * 3, 8},            // 4416 KB (32 KB/CTA x 3 CTAs/SM = 96 KB/SM)
      {sm_count * 4, 8},            // 5888 KB
      {sm_count * 5, 8},            // 7360 KB (> 7 MiB)
  };
  // Deduplicate and sort strictly by bytes_in_flight = ctas * mult * 4096.
  std::sort(raw_pts.begin(), raw_pts.end(), [](const MlpPoint& a, const MlpPoint& b) {
    return (a.ctas * a.mult) < (b.ctas * b.mult);
  });
  std::vector<MlpPoint> pts;
  for (const auto& p : raw_pts) {
    if (pts.empty() || (p.ctas * p.mult) > (pts.back().ctas * pts.back().mult)) {
      pts.push_back(p);
    }
  }

  auto run_suite = [&](const char* key, bool is_dram, bool comma) {
    const uint4* buf = is_dram ? d_dram : d_l2;
    const size_t mask = is_dram ? dram_mask : l2_mask;
    const size_t flush_vec4 = dram_bytes / sizeof(uint4);

    std::printf("  \"%s\": [", key);
    bool first_out = true;
    for (const char* mode_name : {"cp_async", "ld_global_v4"}) {
      bool is_cp = (std::strcmp(mode_name, "cp_async") == 0);
      for (const auto& pt : pts) {
        int ctas = pt.ctas;
        int mult = pt.mult;
        size_t bytes_per_cta = static_cast<size_t>(mult) * 4096ULL;
        size_t bif = static_cast<size_t>(ctas) * bytes_per_cta;

        // Pick total bytes transferred so each launch takes ~0.4-2.0 ms.
        size_t target_bytes = is_dram
                                  ? std::min(dram_bytes, std::max<size_t>(16ULL * 1024ULL * 1024ULL, bif * 128ULL))
                                  : std::max<size_t>(64ULL * 1024ULL * 1024ULL, bif * 256ULL);
        int cp_sub = (mult <= 4) ? 1 : (mult / 4);
        int cp_stages = (mult <= 4) ? mult : 4;
        int cp_steps = std::max(cp_stages * 4, static_cast<int>(target_bytes / (static_cast<size_t>(ctas) * 4096ULL * cp_sub)));
        int ld_iters = std::max(4, static_cast<int>(target_bytes / (static_cast<size_t>(ctas) * 4096ULL * mult)));
        double actual_bytes = is_cp
                                  ? (static_cast<double>(ctas) * 4096.0 * cp_sub * cp_steps)
                                  : (static_cast<double>(ctas) * 4096.0 * mult * ld_iters);

        const size_t cp_dyn_smem = static_cast<size_t>(cp_stages) * static_cast<size_t>(cp_sub) * 4096ULL;
        auto launch_once = [&]() {
          if (is_cp) {
            if (mult == 1) mlp_cp_async_kernel<1, 1><<<ctas, 256, cp_dyn_smem>>>(buf, d_sink, mask, cp_steps);
            else if (mult == 2) mlp_cp_async_kernel<2, 1><<<ctas, 256, cp_dyn_smem>>>(buf, d_sink, mask, cp_steps);
            else if (mult == 4) mlp_cp_async_kernel<4, 1><<<ctas, 256, cp_dyn_smem>>>(buf, d_sink, mask, cp_steps);
            else mlp_cp_async_kernel<4, 2><<<ctas, 256, cp_dyn_smem>>>(buf, d_sink, mask, cp_steps);
          } else {
            if (mult == 1) mlp_ld_v4_kernel<1><<<ctas, 256>>>(buf, d_sink, mask, ld_iters);
            else if (mult == 2) mlp_ld_v4_kernel<2><<<ctas, 256>>>(buf, d_sink, mask, ld_iters);
            else if (mult == 4) mlp_ld_v4_kernel<4><<<ctas, 256>>>(buf, d_sink, mask, ld_iters);
            else mlp_ld_v4_kernel<8><<<ctas, 256>>>(buf, d_sink, mask, ld_iters);
          }
        };

        for (int w = 0; w < warmup; ++w) {
          if (is_dram) {
            l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink, flush_vec4, 1);
          }
          launch_once();
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> gbps;
        gbps.reserve(samples);
        for (int s = 0; s < samples; ++s) {
          if (is_dram) {
            l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink, flush_vec4, 1);
          }
          CUDA_CHECK(cudaEventRecord(ev_start));
          launch_once();
          CUDA_CHECK(cudaEventRecord(ev_stop));
          CUDA_CHECK(cudaEventSynchronize(ev_stop));
          float ms = 0.0f;
          CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
          gbps.push_back((actual_bytes / 1e9) / (static_cast<double>(ms) / 1e3));
        }

        if (!first_out) std::printf(",");
        first_out = false;
        std::printf("\n    {\"issue_mode\": \"%s\", \"bytes_in_flight\": %zu, \"active_ctas\": %d, \"bytes_per_cta\": %zu, \"gbps_samples\": [",
                    mode_name, bif, ctas, bytes_per_cta);
        for (size_t k = 0; k < gbps.size(); ++k) {
          if (k > 0) std::printf(", ");
          std::printf("%.3f", gbps[k]);
        }
        std::printf("]}");
      }
    }
    std::printf("\n  ]%s\n", comma ? "," : "");
  };

  std::printf("{\n  \"samples\": %d,\n", samples);
  run_suite("dram_bw_by_bytes_in_flight", true, true);
  run_suite("l2_bw_by_bytes_in_flight", false, false);
  std::printf("}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_dram));
  CUDA_CHECK(cudaFree(d_flush));
  CUDA_CHECK(cudaFree(d_l2));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}

// ---------------------------------------------------------------------------
// 9. Closed-loop MLP benchmark: D tokens circulating between DRAM and compute
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void spin_cycles(uint64_t cycles) {
  if (cycles == 0) return;
  uint64_t t0 = static_cast<uint64_t>(clock64());
  while (static_cast<uint64_t>(clock64()) - t0 < cycles) {
  }
}

__global__ void calibrate_clock_kernel(uint64_t cycles, uint64_t* __restrict__ sink) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    uint64_t t0 = static_cast<uint64_t>(clock64());
    while (static_cast<uint64_t>(clock64()) - t0 < cycles) {
    }
    *sink = static_cast<uint64_t>(clock64()) - t0;
  }
}

static double measure_cycles_per_ns(cudaEvent_t ev_start, cudaEvent_t ev_stop) {
  uint64_t* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint64_t)));
  const uint64_t cal_cycles = 5000000ULL;  // ~1.7-2.0 ms
  calibrate_clock_kernel<<<1, 32>>>(cal_cycles, d_sink);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(ev_start));
  calibrate_clock_kernel<<<1, 32>>>(cal_cycles, d_sink);
  CUDA_CHECK(cudaEventRecord(ev_stop));
  CUDA_CHECK(cudaEventSynchronize(ev_stop));
  float ms = 1.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
  CUDA_CHECK(cudaFree(d_sink));
  return static_cast<double>(cal_cycles) / (static_cast<double>(ms) * 1e6);
}

template <int TOKENS, int VECS_PER_THREAD>
__global__ void closed_loop_cp_async_kernel(const uint4* __restrict__ src,
                                            uint4* __restrict__ sink,
                                            size_t n_vec4_mask,
                                            int steps,
                                            uint64_t compute_cycles) {
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const size_t stage_vec4 = static_cast<size_t>(256) * static_cast<size_t>(VECS_PER_THREAD);
  const size_t grid_stage_vec4 = static_cast<size_t>(gridDim.x) * stage_vec4;
  size_t base = (static_cast<size_t>(blockIdx.x) * stage_vec4 + tid) & n_vec4_mask;

  auto issue_stage = [&](int slot, size_t cur_base) {
    uint4* slot_ptr = smem + static_cast<size_t>(slot) * stage_vec4;
    #pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; ++v) {
      uint32_t smem_addr = static_cast<uint32_t>(
          __cvta_generic_to_shared(slot_ptr + static_cast<size_t>(v) * 256ULL + tid));
      const void* gptr = src + ((cur_base + static_cast<size_t>(v) * 256ULL) & n_vec4_mask);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gptr));
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };

  // Prefill TOKENS stages into the ring buffer.
  int issued = 0;
  #pragma unroll
  for (int s = 0; s < TOKENS; ++s) {
    if (s < steps) {
      issue_stage(s, base);
      base = (base + grid_stage_vec4) & n_vec4_mask;
      ++issued;
    }
  }

  for (int step = 0; step < steps; ++step) {
    // Wait until the oldest stage arrives, leaving at most (TOKENS - 1) stages in flight.
    if constexpr (TOKENS == 1) {
      asm volatile("cp.async.wait_group 0;\n" ::);
    } else if constexpr (TOKENS == 2) {
      asm volatile("cp.async.wait_group 1;\n" ::);
    } else if constexpr (TOKENS == 3) {
      asm volatile("cp.async.wait_group 2;\n" ::);
    } else if constexpr (TOKENS == 4) {
      asm volatile("cp.async.wait_group 3;\n" ::);
    }
    // Hold the arrived stage buffer during compute before releasing and refilling it.
    spin_cycles(compute_cycles);
    if (issued < steps) {
      issue_stage(step % TOKENS, base);
      base = (base + grid_stage_vec4) & n_vec4_mask;
      ++issued;
    }
  }

  asm volatile("cp.async.wait_all;\n" ::);
  __syncthreads();
  uint4 val = smem[tid];
  if (val.x == 0xdeadbeefu && val.y == 0xdeadbeefu) {
    sink[tid] = val;
  }
}

template <int VECS_PER_THREAD>
__global__ void closed_loop_ld_v4_kernel(const uint4* __restrict__ src,
                                         uint4* __restrict__ sink,
                                         size_t n_vec4_mask,
                                         int steps,
                                         uint64_t compute_cycles) {
  const uint32_t tid = threadIdx.x;
  const size_t stage_vec4 = static_cast<size_t>(256) * static_cast<size_t>(VECS_PER_THREAD);
  const size_t grid_stage_vec4 = static_cast<size_t>(gridDim.x) * stage_vec4;
  size_t base = (static_cast<size_t>(blockIdx.x) * stage_vec4 + tid) & n_vec4_mask;
  uint32_t acc = 0;

  for (int step = 0; step < steps; ++step) {
    uint32_t round_xor = 0;
    #pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; ++v) {
      uint4 val = ld_cg_u4(src + ((base + static_cast<size_t>(v) * 256ULL) & n_vec4_mask));
      round_xor ^= (val.x ^ val.y ^ val.z ^ val.w);
    }
    acc ^= round_xor;
    // Force all loads in this stage to complete before compute_cycles begins.
    uint64_t dep_cyc = (round_xor == 0xdeadbeefu) ? 1ULL : 0ULL;
    spin_cycles(compute_cycles + dep_cyc);
    base = (base + grid_stage_vec4 + static_cast<size_t>(dep_cyc)) & n_vec4_mask;
  }
  if (acc == 0xdeadbeefu) {
    sink[tid] = make_uint4(acc, acc, acc, acc);
  }
}

static int run_closed_loop(int device, int samples, int warmup) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;

  const size_t dram_bytes = 256ULL * 1024ULL * 1024ULL;
  const size_t l2_bytes = 8ULL * 1024ULL * 1024ULL;
  const size_t dram_mask = (dram_bytes / sizeof(uint4)) - 1ULL;
  const size_t l2_mask = (l2_bytes / sizeof(uint4)) - 1ULL;
  const size_t flush_vec4 = dram_bytes / sizeof(uint4);

  // Opt-in to 64 KB dynamic shared memory for tokens=4, stage_bytes=16384.
  CUDA_CHECK(cudaFuncSetAttribute(
      closed_loop_cp_async_kernel<4, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));

  uint4* d_dram = nullptr;
  uint4* d_flush = nullptr;
  uint4* d_l2 = nullptr;
  uint4* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dram, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_flush, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_l2, l2_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, 1024 * sizeof(uint4)));
  CUDA_CHECK(cudaMemset(d_dram, 17, dram_bytes));
  CUDA_CHECK(cudaMemset(d_flush, 18, dram_bytes));
  CUDA_CHECK(cudaMemset(d_l2, 19, l2_bytes));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));
  const double cycles_per_ns = measure_cycles_per_ns(ev_start, ev_stop);

  const std::vector<int> token_list = {0, 1, 2, 3, 4};
  const std::vector<int> cta_list = {16, 32, sm_count, sm_count * 2, 128};
  const std::vector<int> stage_bytes_list = {4096, 8192, 16384};
  const std::vector<uint64_t> compute_ns_list = {0ULL, 250ULL, 500ULL, 1000ULL, 2000ULL};

  auto max_resident_ctas = [&](int tokens, int vecs, size_t dyn_smem) -> int {
    int max_blk = 1;
    if (tokens == 0) {
      if (vecs == 1) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_ld_v4_kernel<1>, 256, 0);
      else if (vecs == 2) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_ld_v4_kernel<2>, 256, 0);
      else cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_ld_v4_kernel<4>, 256, 0);
    } else if (tokens == 1) {
      if (vecs == 1) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_cp_async_kernel<1, 1>, 256, dyn_smem);
      else if (vecs == 2) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_cp_async_kernel<1, 2>, 256, dyn_smem);
      else cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_cp_async_kernel<1, 4>, 256, dyn_smem);
    } else if (tokens == 2) {
      if (vecs == 1) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_cp_async_kernel<2, 1>, 256, dyn_smem);
      else if (vecs == 2) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_cp_async_kernel<2, 2>, 256, dyn_smem);
      else cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_cp_async_kernel<2, 4>, 256, dyn_smem);
    } else if (tokens == 3) {
      if (vecs == 1) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_cp_async_kernel<3, 1>, 256, dyn_smem);
      else if (vecs == 2) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_cp_async_kernel<3, 2>, 256, dyn_smem);
      else cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_cp_async_kernel<3, 4>, 256, dyn_smem);
    } else {
      if (vecs == 1) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_cp_async_kernel<4, 1>, 256, dyn_smem);
      else if (vecs == 2) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_cp_async_kernel<4, 2>, 256, dyn_smem);
      else cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blk, closed_loop_cp_async_kernel<4, 4>, 256, dyn_smem);
    }
    return sm_count * max_blk;
  };

  auto run_level = [&](const char* key, bool is_dram, bool comma) {
    const uint4* buf = is_dram ? d_dram : d_l2;
    const size_t mask = is_dram ? dram_mask : l2_mask;
    std::printf("  \"%s\": [", key);
    bool first = true;

    for (int tokens : token_list) {
      const char* mode_name = (tokens == 0) ? "ld_global_v4" : "cp_async";
      for (int ctas : cta_list) {
        for (int stage_bytes : stage_bytes_list) {
          const int vecs = stage_bytes / 4096;
          const size_t dyn_smem = static_cast<size_t>(tokens) * static_cast<size_t>(stage_bytes);
          // Skip configs that exceed single-wave residency so every point is truly concurrent.
          if (ctas > max_resident_ctas(tokens, vecs, dyn_smem)) continue;

          for (uint64_t tc_ns : compute_ns_list) {
            const uint64_t tc_cyc = static_cast<uint64_t>(static_cast<double>(tc_ns) * cycles_per_ns);
            const double bw_ref = is_dram ? 450.0 : 1400.0;
            double est_bw_step_ns = (static_cast<double>(ctas) * stage_bytes) / bw_ref;
            double est_step_ns = std::max({static_cast<double>(tc_ns), est_bw_step_ns, is_dram ? 250.0 : 120.0});
            int steps = std::max(256, std::min(1024, static_cast<int>(500000.0 / est_step_ns)));
            double total_bytes = static_cast<double>(ctas) * static_cast<double>(stage_bytes) * steps;

            auto launch_once = [&]() {
              if (tokens == 0) {
                if (vecs == 1) closed_loop_ld_v4_kernel<1><<<ctas, 256>>>(buf, d_sink, mask, steps, tc_cyc);
                else if (vecs == 2) closed_loop_ld_v4_kernel<2><<<ctas, 256>>>(buf, d_sink, mask, steps, tc_cyc);
                else closed_loop_ld_v4_kernel<4><<<ctas, 256>>>(buf, d_sink, mask, steps, tc_cyc);
              } else if (tokens == 1) {
                if (vecs == 1) closed_loop_cp_async_kernel<1, 1><<<ctas, 256, dyn_smem>>>(buf, d_sink, mask, steps, tc_cyc);
                else if (vecs == 2) closed_loop_cp_async_kernel<1, 2><<<ctas, 256, dyn_smem>>>(buf, d_sink, mask, steps, tc_cyc);
                else closed_loop_cp_async_kernel<1, 4><<<ctas, 256, dyn_smem>>>(buf, d_sink, mask, steps, tc_cyc);
              } else if (tokens == 2) {
                if (vecs == 1) closed_loop_cp_async_kernel<2, 1><<<ctas, 256, dyn_smem>>>(buf, d_sink, mask, steps, tc_cyc);
                else if (vecs == 2) closed_loop_cp_async_kernel<2, 2><<<ctas, 256, dyn_smem>>>(buf, d_sink, mask, steps, tc_cyc);
                else closed_loop_cp_async_kernel<2, 4><<<ctas, 256, dyn_smem>>>(buf, d_sink, mask, steps, tc_cyc);
              } else if (tokens == 3) {
                if (vecs == 1) closed_loop_cp_async_kernel<3, 1><<<ctas, 256, dyn_smem>>>(buf, d_sink, mask, steps, tc_cyc);
                else if (vecs == 2) closed_loop_cp_async_kernel<3, 2><<<ctas, 256, dyn_smem>>>(buf, d_sink, mask, steps, tc_cyc);
                else closed_loop_cp_async_kernel<3, 4><<<ctas, 256, dyn_smem>>>(buf, d_sink, mask, steps, tc_cyc);
              } else {
                if (vecs == 1) closed_loop_cp_async_kernel<4, 1><<<ctas, 256, dyn_smem>>>(buf, d_sink, mask, steps, tc_cyc);
                else if (vecs == 2) closed_loop_cp_async_kernel<4, 2><<<ctas, 256, dyn_smem>>>(buf, d_sink, mask, steps, tc_cyc);
                else closed_loop_cp_async_kernel<4, 4><<<ctas, 256, dyn_smem>>>(buf, d_sink, mask, steps, tc_cyc);
              }
            };

            for (int w = 0; w < warmup; ++w) {
              if (is_dram) l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink, flush_vec4, 1);
              launch_once();
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<double> bw_samples;
            std::vector<double> turn_samples;
            bw_samples.reserve(samples);
            turn_samples.reserve(samples);

            for (int s = 0; s < samples; ++s) {
              if (is_dram) l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink, flush_vec4, 1);
              CUDA_CHECK(cudaEventRecord(ev_start));
              launch_once();
              CUDA_CHECK(cudaEventRecord(ev_stop));
              CUDA_CHECK(cudaEventSynchronize(ev_stop));
              float ms = 0.0f;
              CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
              bw_samples.push_back((total_bytes / 1e9) / (static_cast<double>(ms) / 1e3));
              turn_samples.push_back((static_cast<double>(ms) * 1e6) / static_cast<double>(steps));
            }

            if (!first) std::printf(",");
            first = false;
            std::printf(
                "\n    {\"issue_mode\": \"%s\", \"tokens\": %d, \"active_ctas\": %d, "
                "\"stage_bytes\": %d, \"compute_ns\": %llu, \"bw_gbps_samples\": [",
                mode_name, tokens, ctas, stage_bytes, static_cast<unsigned long long>(tc_ns));
            for (size_t k = 0; k < bw_samples.size(); ++k) {
              if (k > 0) std::printf(", ");
              std::printf("%.3f", bw_samples[k]);
            }
            std::printf("], \"turnaround_ns_samples\": [");
            for (size_t k = 0; k < turn_samples.size(); ++k) {
              if (k > 0) std::printf(", ");
              std::printf("%.2f", turn_samples[k]);
            }
            std::printf("]}");
          }
        }
      }
    }
    std::printf("\n  ]%s\n", comma ? "," : "");
  };

  std::printf("{\n  \"samples\": %d,\n  \"cycles_per_ns\": %.4f,\n", samples, cycles_per_ns);
  run_level("dram_closed_loop", true, true);
  run_level("l2_closed_loop", false, false);
  std::printf("}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_dram));
  CUDA_CHECK(cudaFree(d_flush));
  CUDA_CHECK(cudaFree(d_l2));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}

// ---------------------------------------------------------------------------
// 10. L2 miss-merging benchmark: sharing degree R with stagger=0 vs stagger=16
// ---------------------------------------------------------------------------

template <int VECS_PER_THREAD>
__global__ void l2_merge_cp_async_kernel(const uint4* __restrict__ src,
                                         uint4* __restrict__ sink,
                                         double* __restrict__ out_cta_turn_cycles,
                                         size_t n_vec4_mask,
                                         int steps,
                                         int sharing,
                                         uint64_t stagger_delay_cycles) {
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const int group_id = blockIdx.x / sharing;
  const int rank = blockIdx.x % sharing;
  const int n_groups = gridDim.x / sharing;
  const size_t stage_vec4 = static_cast<size_t>(256) * static_cast<size_t>(VECS_PER_THREAD);
  const size_t group_stride_vec4 = static_cast<size_t>(n_groups) * stage_vec4;
  size_t base = (static_cast<size_t>(group_id) * stage_vec4 + tid) & n_vec4_mask;

  // When staggered, followers wait briefly so rank 0 fetches data into L2 first (resident hit).
  if (stagger_delay_cycles > 0 && rank > 0) {
    spin_cycles(stagger_delay_cycles);
  }
  __syncthreads();
  uint64_t t0 = (tid == 0) ? static_cast<uint64_t>(clock64()) : 0;

  for (int step = 0; step < steps; ++step) {
    #pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; ++v) {
      uint32_t smem_addr = static_cast<uint32_t>(
          __cvta_generic_to_shared(smem + static_cast<size_t>(v) * 256ULL + tid));
      const void* gptr = src + ((base + static_cast<size_t>(v) * 256ULL) & n_vec4_mask);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gptr));
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
    base = (base + group_stride_vec4) & n_vec4_mask;
  }

  __syncthreads();
  if (tid == 0) {
    uint64_t t1 = static_cast<uint64_t>(clock64());
    out_cta_turn_cycles[blockIdx.x] = static_cast<double>(t1 - t0) / static_cast<double>(steps);
  }
  uint4 val = smem[tid];
  if (val.x == 0xdeadbeefu && val.y == 0xdeadbeefu) {
    sink[tid] = val;
  }
}

static int run_l2_merge(int device, int samples, int warmup) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;

  const size_t dram_bytes = 256ULL * 1024ULL * 1024ULL;
  const size_t dram_mask = (dram_bytes / sizeof(uint4)) - 1ULL;
  const size_t flush_vec4 = dram_bytes / sizeof(uint4);

  uint4* d_dram = nullptr;
  uint4* d_flush = nullptr;
  uint4* d_sink = nullptr;
  double* d_cta_turn = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dram, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_flush, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, 1024 * sizeof(uint4)));
  CUDA_CHECK(cudaMalloc(&d_cta_turn, 256 * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_dram, 21, dram_bytes));
  CUDA_CHECK(cudaMemset(d_flush, 22, dram_bytes));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));
  const double cycles_per_ns = measure_cycles_per_ns(ev_start, ev_stop);

  const std::vector<int> n_list = {16, 23, sm_count, sm_count * 2};
  const std::vector<int> r_list = {1, 2, 4, 8, 16, 23};
  const std::vector<int> k_list = {4096, 8192, 16384};
  const int steps = 256;

  std::vector<double> h_cta_turn(256, 0.0);
  std::printf("{\n  \"samples\": %d,\n  \"cycles_per_ns\": %.4f,\n  \"l2_merge_by_sharing\": [",
              samples, cycles_per_ns);
  bool first = true;

  for (int n : n_list) {
    for (int r : r_list) {
      if (n % r != 0) continue;
      for (int k : k_list) {
        const int vecs = k / 4096;
        const size_t dyn_smem = static_cast<size_t>(k);
        std::vector<int> staggers = (r == 1) ? std::vector<int>{0} : std::vector<int>{0, 16};
        for (int stag : staggers) {
          // ~300 ns per staggered stage so rank 0 populates L2 ahead of followers.
          uint64_t stag_cyc = static_cast<uint64_t>(static_cast<double>(stag) * 300.0 * cycles_per_ns);
          double total_l2_bytes = static_cast<double>(n) * static_cast<double>(k) * steps;

          auto launch_once = [&]() {
            if (vecs == 1) l2_merge_cp_async_kernel<1><<<n, 256, dyn_smem>>>(d_dram, d_sink, d_cta_turn, dram_mask, steps, r, stag_cyc);
            else if (vecs == 2) l2_merge_cp_async_kernel<2><<<n, 256, dyn_smem>>>(d_dram, d_sink, d_cta_turn, dram_mask, steps, r, stag_cyc);
            else l2_merge_cp_async_kernel<4><<<n, 256, dyn_smem>>>(d_dram, d_sink, d_cta_turn, dram_mask, steps, r, stag_cyc);
          };

          for (int w = 0; w < warmup; ++w) {
            l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink, flush_vec4, 1);
            launch_once();
          }
          CUDA_CHECK(cudaDeviceSynchronize());

          std::vector<double> turn_ns_s, dram_bw_s, l2_bw_s;
          turn_ns_s.reserve(samples);
          dram_bw_s.reserve(samples);
          l2_bw_s.reserve(samples);

          for (int s = 0; s < samples; ++s) {
            l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink, flush_vec4, 1);
            CUDA_CHECK(cudaEventRecord(ev_start));
            launch_once();
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            CUDA_CHECK(cudaMemcpy(h_cta_turn.data(), d_cta_turn,
                                  static_cast<size_t>(n) * sizeof(double), cudaMemcpyDeviceToHost));
            double avg_cyc = 0.0;
            for (int c = 0; c < n; ++c) avg_cyc += h_cta_turn[c];
            avg_cyc /= static_cast<double>(n);
            double l2_bw = (total_l2_bytes / 1e9) / (static_cast<double>(ms) / 1e3);
            turn_ns_s.push_back(avg_cyc / cycles_per_ns);
            l2_bw_s.push_back(l2_bw);
            dram_bw_s.push_back(l2_bw / static_cast<double>(r));
          }

          if (!first) std::printf(",");
          first = false;
          std::printf(
              "\n    {\"active_ctas\": %d, \"sharing\": %d, \"bytes_per_cta\": %d, \"stagger_stages\": %d, "
              "\"turnaround_ns_samples\": [",
              n, r, k, stag);
          for (size_t i = 0; i < turn_ns_s.size(); ++i) {
            if (i > 0) std::printf(", ");
            std::printf("%.2f", turn_ns_s[i]);
          }
          std::printf("], \"dram_unique_bw_gbps_samples\": [");
          for (size_t i = 0; i < dram_bw_s.size(); ++i) {
            if (i > 0) std::printf(", ");
            std::printf("%.3f", dram_bw_s[i]);
          }
          std::printf("], \"l2_to_sm_bw_gbps_samples\": [");
          for (size_t i = 0; i < l2_bw_s.size(); ++i) {
            if (i > 0) std::printf(", ");
            std::printf("%.3f", l2_bw_s[i]);
          }
          std::printf("]}");
        }
      }
    }
  }
  std::printf("\n  ]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_dram));
  CUDA_CHECK(cudaFree(d_flush));
  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_cta_turn));
  return 0;
}

// ---------------------------------------------------------------------------
// 11. Per-stage fixed sync/pipeline overhead (hot cache, minimal data)
// ---------------------------------------------------------------------------

template <int TOKENS>
__global__ void stage_overhead_cp_async_kernel(const uint4* __restrict__ hot_src,
                                               uint4* __restrict__ sink,
                                               double* __restrict__ out_cycles_per_stage,
                                               int steps) {
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const uint32_t nthreads = blockDim.x;
  // Each thread reads its own 16B slot in a tiny hot L2-resident buffer (<= 4 KB/CTA, zero bank contention).
  const void* gptr = hot_src + static_cast<size_t>(blockIdx.x) * nthreads + tid;

  auto issue_tiny = [&](int slot) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem[slot * nthreads + tid]));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gptr) : "memory");
    asm volatile("cp.async.commit_group;\n" :: : "memory");
  };

  // Warmup and prefill TOKENS stages.
  for (int w = 0; w < 16; ++w) {
    issue_tiny(w % TOKENS);
    asm volatile("cp.async.wait_all;\n" ::);
    __syncthreads();
  }
  #pragma unroll
  for (int s = 0; s < TOKENS; ++s) {
    issue_tiny(s);
  }
  __syncthreads();
  uint64_t t0 = (tid == 0) ? static_cast<uint64_t>(clock64()) : 0;

  for (int step = 0; step < steps; ++step) {
    if constexpr (TOKENS == 1) {
      asm volatile("cp.async.wait_group 0;\n" ::);
    } else if constexpr (TOKENS == 2) {
      asm volatile("cp.async.wait_group 1;\n" ::);
    } else if constexpr (TOKENS == 3) {
      asm volatile("cp.async.wait_group 2;\n" ::);
    } else if constexpr (TOKENS == 4) {
      asm volatile("cp.async.wait_group 3;\n" ::);
    }
    __syncthreads();
    issue_tiny(step % TOKENS);
  }

  asm volatile("cp.async.wait_all;\n" ::);
  __syncthreads();
  if (tid == 0) {
    uint64_t t1 = static_cast<uint64_t>(clock64());
    out_cycles_per_stage[blockIdx.x] = static_cast<double>(t1 - t0) / static_cast<double>(steps);
  }
  uint4 val = smem[tid];
  if (val.x == 0xdeadbeefu && val.y == 0xdeadbeefu) {
    sink[tid] = val;
  }
}

__global__ void stage_overhead_ld_v4_kernel(const uint4* __restrict__ hot_src,
                                            uint4* __restrict__ sink,
                                            double* __restrict__ out_cycles_per_stage,
                                            int steps) {
  const uint32_t tid = threadIdx.x;
  const uint4* gptr = hot_src + (tid & 31u);
  uint32_t acc = 0;

  for (int w = 0; w < 16; ++w) {
    uint4 v;
    asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(gptr));
    acc ^= v.x;
    __syncthreads();
  }

  uint64_t t0 = (tid == 0) ? static_cast<uint64_t>(clock64()) : 0;
  for (int step = 0; step < steps; ++step) {
    uint4 v;
    asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(gptr + ((acc == 0xdeadbeefu) ? 1u : 0u)));
    acc ^= (v.x ^ v.y ^ v.z ^ v.w);
    __syncthreads();
  }
  if (tid == 0) {
    uint64_t t1 = static_cast<uint64_t>(clock64());
    out_cycles_per_stage[blockIdx.x] = static_cast<double>(t1 - t0) / static_cast<double>(steps);
  }
  if (acc == 0xdeadbeefu) {
    sink[tid] = make_uint4(acc, acc, acc, acc);
  }
}

template <int TOKENS>
__global__ void fixed_sync_empty_kernel(double* __restrict__ out_cycles_per_stage, int steps) {
  #pragma unroll
  for (int s = 0; s < TOKENS; ++s) {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
  __syncthreads();
  uint64_t t0 = (threadIdx.x == 0) ? static_cast<uint64_t>(clock64()) : 0;
  for (int step = 0; step < steps; ++step) {
    if constexpr (TOKENS == 1) asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    else if constexpr (TOKENS == 2) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    else if constexpr (TOKENS == 3) asm volatile("cp.async.wait_group 2;\n" ::: "memory");
    else if constexpr (TOKENS == 4) asm volatile("cp.async.wait_group 3;\n" ::: "memory");
    __syncthreads();
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
  asm volatile("cp.async.wait_all;\n" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    out_cycles_per_stage[0] = static_cast<double>(static_cast<uint64_t>(clock64()) - t0) / static_cast<double>(steps);
  }
}

__global__ void min_async_roundtrip_kernel(const uint4* __restrict__ hot_src,
                                           uint4* __restrict__ sink,
                                           double* __restrict__ out_cycles_per_stage,
                                           int steps) {
  __shared__ uint4 smem[32];
  uint32_t tid = threadIdx.x;
  uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(&smem[0]));
  for (int w = 0; w < 16; ++w) {
    if (tid == 0) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(hot_src) : "memory");
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();
  }
  uint64_t t0 = (tid == 0) ? static_cast<uint64_t>(clock64()) : 0;
  for (int step = 0; step < steps; ++step) {
    if (tid == 0) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(hot_src) : "memory");
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
  }
  if (tid == 0) {
    out_cycles_per_stage[0] = static_cast<double>(static_cast<uint64_t>(clock64()) - t0) / static_cast<double>(steps);
    if (smem[0].x == 0xdeadbeefu) sink[0] = smem[0];
  }
}

static int run_stage_overhead(int device, int samples, int warmup) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  // Run on 1 CTA so per-stage sync/pipeline overhead is isolated from cross-SM L2 contention.
  const int grid = 1;
  uint4* d_hot = nullptr;
  uint4* d_sink = nullptr;
  double* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_hot, 512 * sizeof(uint4)));
  CUDA_CHECK(cudaMalloc(&d_sink, 512 * sizeof(uint4)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(double)));
  CUDA_CHECK(cudaMemset(d_hot, 3, 512 * sizeof(uint4)));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));
  const double cycles_per_ns = measure_cycles_per_ns(ev_start, ev_stop);

  const std::vector<int> warp_list = {4, 8};
  const std::vector<int> token_list = {0, 1, 2, 3, 4};
  const int steps = 4096;

  std::printf("{\n  \"samples\": %d,\n  \"cycles_per_ns\": %.4f,\n  \"stage_sync_overhead_ns\": [",
              samples, cycles_per_ns);
  bool first = true;

  for (int warps : warp_list) {
    int threads = warps * 32;
    for (int tokens : token_list) {
      const char* mode_name = (tokens == 0) ? "ld_global_v4" : "cp_async";
      size_t dyn_smem = static_cast<size_t>(std::max(1, tokens)) * static_cast<size_t>(threads) * sizeof(uint4);

      auto launch_once = [&]() {
        if (tokens == 0) {
          stage_overhead_ld_v4_kernel<<<grid, threads>>>(d_hot, d_sink, d_out, steps);
        } else if (tokens == 1) {
          stage_overhead_cp_async_kernel<1><<<grid, threads, dyn_smem>>>(d_hot, d_sink, d_out, steps);
        } else if (tokens == 2) {
          stage_overhead_cp_async_kernel<2><<<grid, threads, dyn_smem>>>(d_hot, d_sink, d_out, steps);
        } else if (tokens == 3) {
          stage_overhead_cp_async_kernel<3><<<grid, threads, dyn_smem>>>(d_hot, d_sink, d_out, steps);
        } else {
          stage_overhead_cp_async_kernel<4><<<grid, threads, dyn_smem>>>(d_hot, d_sink, d_out, steps);
        }
      };

      for (int w = 0; w < warmup; ++w) launch_once();
      CUDA_CHECK(cudaDeviceSynchronize());

      std::vector<double> ns_samples;
      ns_samples.reserve(samples);
      for (int s = 0; s < samples; ++s) {
        launch_once();
        CUDA_CHECK(cudaDeviceSynchronize());
        double cyc = 0.0;
        CUDA_CHECK(cudaMemcpy(&cyc, d_out, sizeof(double), cudaMemcpyDeviceToHost));
        ns_samples.push_back(cyc / cycles_per_ns);
      }

      if (!first) std::printf(",");
      first = false;
      std::printf("\n    {\"issue_mode\": \"%s\", \"num_warps\": %d, \"tokens\": %d, \"overhead_ns_samples\": [",
                  mode_name, warps, tokens);
      for (size_t i = 0; i < ns_samples.size(); ++i) {
        if (i > 0) std::printf(", ");
        std::printf("%.2f", ns_samples[i]);
      }
      std::printf("]}");
    }
  }
  std::printf("\n  ],\n  \"fixed_sync_ns\": [");
  bool first_fs = true;
  for (int warps : warp_list) {
    int threads = warps * 32;
    for (int tokens : {1, 2, 3, 4}) {
      auto launch_fs = [&]() {
        if (tokens == 1) fixed_sync_empty_kernel<1><<<grid, threads>>>(d_out, steps);
        else if (tokens == 2) fixed_sync_empty_kernel<2><<<grid, threads>>>(d_out, steps);
        else if (tokens == 3) fixed_sync_empty_kernel<3><<<grid, threads>>>(d_out, steps);
        else fixed_sync_empty_kernel<4><<<grid, threads>>>(d_out, steps);
      };
      for (int w = 0; w < warmup; ++w) launch_fs();
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<double> fs_samples;
      fs_samples.reserve(samples);
      for (int s = 0; s < samples; ++s) {
        launch_fs();
        CUDA_CHECK(cudaDeviceSynchronize());
        double cyc = 0.0;
        CUDA_CHECK(cudaMemcpy(&cyc, d_out, sizeof(double), cudaMemcpyDeviceToHost));
        fs_samples.push_back(cyc / cycles_per_ns);
      }
      if (!first_fs) std::printf(",");
      first_fs = false;
      std::printf("\n    {\"num_warps\": %d, \"tokens\": %d, \"overhead_ns_samples\": [", warps, tokens);
      for (size_t i = 0; i < fs_samples.size(); ++i) {
        if (i > 0) std::printf(", ");
        std::printf("%.2f", fs_samples[i]);
      }
      std::printf("]}");
    }
  }

  for (int w = 0; w < warmup; ++w) min_async_roundtrip_kernel<<<1, 128>>>(d_hot, d_sink, d_out, steps);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::printf("\n  ],\n  \"min_async_roundtrip_ns_samples\": [");
  for (int s = 0; s < samples; ++s) {
    min_async_roundtrip_kernel<<<1, 128>>>(d_hot, d_sink, d_out, steps);
    CUDA_CHECK(cudaDeviceSynchronize());
    double cyc = 0.0;
    CUDA_CHECK(cudaMemcpy(&cyc, d_out, sizeof(double), cudaMemcpyDeviceToHost));
    if (s > 0) std::printf(", ");
    std::printf("%.2f", cyc / cycles_per_ns);
  }
  std::printf("]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_hot));
  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}

// ---------------------------------------------------------------------------
// 12. Closed-loop DRAM MLP with real MMA compute contention (1/2/3 CTAs/SM)
// ---------------------------------------------------------------------------

template <int TOKENS>
__global__ void closed_loop_mma_kernel(const uint4* __restrict__ src,
                                       float* __restrict__ sink,
                                       size_t n_vec4_mask,
                                       int steps,
                                       int mma_iters) {
  // 128 threads (4 warps/CTA), 8192 B per stage (4 x 16B per thread).
  // Max dynamic smem = 4 * 8192 = 32 KB/CTA -> 3 CTAs/SM = 96 KB/SM (single wave on RTX 4070).
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const size_t stage_vec4 = 512ULL;  // 8192 B / 16 B
  const size_t grid_stage_vec4 = static_cast<size_t>(gridDim.x) * stage_vec4;
  size_t base = (static_cast<size_t>(blockIdx.x) * stage_vec4 + tid) & n_vec4_mask;

  uint32_t a0 = 0x3c003c00u + (tid & 7u), a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u;
  uint32_t b0 = 0x3c003c00u, b1 = 0x3c003c00u;
  float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f;
  float c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f;
  float c20 = 0.f, c21 = 0.f, c22 = 0.f, c23 = 0.f;
  float c30 = 0.f, c31 = 0.f, c32 = 0.f, c33 = 0.f;

  auto issue_stage = [&](int slot, size_t cur_base) {
    uint4* slot_ptr = smem + static_cast<size_t>(slot) * stage_vec4;
    #pragma unroll
    for (int v = 0; v < 4; ++v) {
      uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(slot_ptr + v * 128 + tid));
      const void* gptr = src + ((cur_base + static_cast<size_t>(v) * 128ULL) & n_vec4_mask);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(gptr) : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  int issued = 0;
  #pragma unroll
  for (int s = 0; s < TOKENS; ++s) {
    if (s < steps) {
      issue_stage(s, base);
      base = (base + grid_stage_vec4) & n_vec4_mask;
      ++issued;
    }
  }

  for (int step = 0; step < steps; ++step) {
    if constexpr (TOKENS == 1) asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    else if constexpr (TOKENS == 2) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    else if constexpr (TOKENS == 3) asm volatile("cp.async.wait_group 2;\n" ::: "memory");
    else if constexpr (TOKENS == 4) asm volatile("cp.async.wait_group 3;\n" ::: "memory");

    for (int m = 0; m < mma_iters; ++m) {
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
    }

    if (issued < steps) {
      issue_stage(step % TOKENS, base);
      base = (base + grid_stage_vec4) & n_vec4_mask;
      ++issued;
    }
  }

  asm volatile("cp.async.wait_all;\n" ::: "memory");
  __syncthreads();
  float sum = c00 + c01 + c02 + c03 + c10 + c11 + c12 + c13 +
              c20 + c21 + c22 + c23 + c30 + c31 + c32 + c33 + static_cast<float>(smem[tid].x);
  if (sum == -999999.0f) {
    sink[blockIdx.x * blockDim.x + tid] = sum;
  }
}

template <int TOKENS, int LD_PASSES>
__global__ void closed_loop_mma_8w_kernel(const uint4* __restrict__ src,
                                          float* __restrict__ sink,
                                          size_t n_vec4_mask,
                                          int steps,
                                          int mma_iters) {
  // 256 threads (8 warps/CTA), 8192 B per stage (2 x 16B per thread).
  // Max dynamic smem = 4 * 8192 = 32 KB/CTA -> 2 CTAs/SM = 64 KB/SM (single wave on RTX 4070).
  // LD_PASSES in {0, 4, 8} => 0 / 16384 / 32768 B of ldmatrix.x4 reads per stage per CTA.
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid >> 5;
  const uint32_t lane = tid & 31u;
  const size_t stage_vec4 = 512ULL;  // 8192 B / 16 B
  const size_t grid_stage_vec4 = static_cast<size_t>(gridDim.x) * stage_vec4;
  size_t base = (static_cast<size_t>(blockIdx.x) * stage_vec4 + tid) & n_vec4_mask;

  uint32_t a0 = 0x3c003c00u + (tid & 7u), a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u;
  uint32_t b0 = 0x3c003c00u, b1 = 0x3c003c00u;
  uint32_t live_acc = 0u;
  float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f;
  float c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f;
  float c20 = 0.f, c21 = 0.f, c22 = 0.f, c23 = 0.f;
  float c30 = 0.f, c31 = 0.f, c32 = 0.f, c33 = 0.f;

  auto issue_stage = [&](int slot, size_t cur_base) {
    uint4* slot_ptr = smem + static_cast<size_t>(slot) * stage_vec4;
    #pragma unroll
    for (int v = 0; v < 2; ++v) {
      uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(slot_ptr + v * 256 + tid));
      const void* gptr = src + ((cur_base + static_cast<size_t>(v) * 256ULL) & n_vec4_mask);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(gptr) : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  int issued = 0;
  #pragma unroll
  for (int s = 0; s < TOKENS; ++s) {
    if (s < steps) {
      issue_stage(s, base);
      base = (base + grid_stage_vec4) & n_vec4_mask;
      ++issued;
    }
  }

  for (int step = 0; step < steps; ++step) {
    if constexpr (TOKENS == 1) asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    else if constexpr (TOKENS == 2) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    else if constexpr (TOKENS == 4) asm volatile("cp.async.wait_group 3;\n" ::: "memory");

    if constexpr (LD_PASSES > 0) {
      asm volatile("bar.sync 0;\n" ::: "memory");
      uint4* cur_slot = smem + static_cast<size_t>(step % TOKENS) * stage_vec4;
      #pragma unroll
      for (int p = 0; p < LD_PASSES; ++p) {
        size_t off_u4 = (static_cast<size_t>(p & 1) * 8ULL + warp_id) * 32ULL + lane;
        uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(cur_slot + off_u4));
        uint32_t r0, r1, r2, r3;
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(saddr) : "memory");
        live_acc ^= (r0 ^ r1 ^ r2 ^ r3);
        if (p == LD_PASSES - 2) {
          b0 = r0;
          b1 = r1;
        } else if (p == LD_PASSES - 1) {
          a0 = r0;
          a1 = r1;
          a2 = r2;
          a3 = r3;
        }
      }
    }

    for (int m = 0; m < mma_iters; ++m) {
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
    }

    if constexpr (LD_PASSES > 0) {
      asm volatile("bar.sync 0;\n" ::: "memory");
    }

    if (issued < steps) {
      issue_stage(step % TOKENS, base);
      base = (base + grid_stage_vec4) & n_vec4_mask;
      ++issued;
    }
  }

  asm volatile("cp.async.wait_all;\n" ::: "memory");
  __syncthreads();
  float sum = c00 + c01 + c02 + c03 + c10 + c11 + c12 + c13 +
              c20 + c21 + c22 + c23 + c30 + c31 + c32 + c33 +
              static_cast<float>(smem[tid].x ^ live_acc);
  if (sum == -999999.0f) {
    sink[blockIdx.x * blockDim.x + tid] = sum;
  }
}

template <int LD_PASSES>
static void dispatch_closed_loop_mma_8w(int tokens, int active_ctas, size_t dyn_smem,
                                        const uint4* d_dram, float* d_sink_f,
                                        size_t dram_mask, int steps, int mma_iters) {
  if (tokens == 1) closed_loop_mma_8w_kernel<1, LD_PASSES><<<active_ctas, 256, dyn_smem>>>(d_dram, d_sink_f, dram_mask, steps, mma_iters);
  else if (tokens == 2) closed_loop_mma_8w_kernel<2, LD_PASSES><<<active_ctas, 256, dyn_smem>>>(d_dram, d_sink_f, dram_mask, steps, mma_iters);
  else closed_loop_mma_8w_kernel<4, LD_PASSES><<<active_ctas, 256, dyn_smem>>>(d_dram, d_sink_f, dram_mask, steps, mma_iters);
}

static int run_closed_loop_mma(int device, int samples, int warmup) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;

  const size_t dram_bytes = 256ULL * 1024ULL * 1024ULL;
  const size_t dram_mask = (dram_bytes / sizeof(uint4)) - 1ULL;
  const size_t flush_vec4 = dram_bytes / sizeof(uint4);

  uint4* d_dram = nullptr;
  uint4* d_flush = nullptr;
  uint4* d_sink_u4 = nullptr;
  float* d_sink_f = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dram, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_flush, dram_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink_u4, 1024 * sizeof(uint4)));
  CUDA_CHECK(cudaMalloc(&d_sink_f, static_cast<size_t>(sm_count) * 4 * 256 * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_dram, 0x3c, dram_bytes));
  CUDA_CHECK(cudaMemset(d_flush, 24, dram_bytes));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  // Calibrate solo 4-warp MMA time per iteration (1 CTA/SM across all SMs, no memory traffic).
  const int cal_iters = 20000;
  for (int w = 0; w < 2; ++w) {
    mma_m16n8k16_kernel<false><<<sm_count, 128>>>(d_sink_f, cal_iters);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(ev_start));
  mma_m16n8k16_kernel<false><<<sm_count, 128>>>(d_sink_f, cal_iters);
  CUDA_CHECK(cudaEventRecord(ev_stop));
  CUDA_CHECK(cudaEventSynchronize(ev_stop));
  float cal_ms = 1.0f;
  CUDA_CHECK(cudaEventElapsedTime(&cal_ms, ev_start, ev_stop));
  const double ns_per_mma_iter = (static_cast<double>(cal_ms) * 1e6) / static_cast<double>(cal_iters);
  // 4 warps * 4 unrolled m16n8k16 * 4096 FLOPs = 65536 FLOPs per CTA per iter.
  const double flops_per_cta_iter = 4.0 * 4.0 * 4096.0;
  const double peak_tflops_per_sm = (flops_per_cta_iter / 1e12) / (ns_per_mma_iter / 1e9);

  const int stage_bytes = 8192;
  const int steps = 256;
  const std::vector<int> ctas_per_sm_list = {1, 2, 3};
  const std::vector<int> token_list = {1, 2, 3, 4};
  const std::vector<double> target_tc_ns_list = {0.0, 250.0, 500.0, 1000.0, 2000.0};

  std::printf("{\n  \"samples\": %d,\n  \"ns_per_mma_iter\": %.3f,\n  \"peak_tflops_per_sm\": %.4f,\n  \"dram_closed_loop_mma\": [",
              samples, ns_per_mma_iter, peak_tflops_per_sm);
  bool first = true;

  for (int cps : ctas_per_sm_list) {
    int active_ctas = sm_count * cps;
    for (int tokens : token_list) {
      size_t dyn_smem = static_cast<size_t>(tokens) * static_cast<size_t>(stage_bytes);
      for (double target_tc : target_tc_ns_list) {
        int mma_iters = (target_tc <= 0.0) ? 0 : std::max(1, static_cast<int>(target_tc / ns_per_mma_iter + 0.5));
        double solo_tc_ns = static_cast<double>(mma_iters) * ns_per_mma_iter;
        double total_bytes = static_cast<double>(active_ctas) * static_cast<double>(stage_bytes) * steps;
        double total_flops_per_sm = static_cast<double>(cps) * flops_per_cta_iter * mma_iters * steps;

        auto launch_once = [&]() {
          if (tokens == 1) closed_loop_mma_kernel<1><<<active_ctas, 128, dyn_smem>>>(d_dram, d_sink_f, dram_mask, steps, mma_iters);
          else if (tokens == 2) closed_loop_mma_kernel<2><<<active_ctas, 128, dyn_smem>>>(d_dram, d_sink_f, dram_mask, steps, mma_iters);
          else if (tokens == 3) closed_loop_mma_kernel<3><<<active_ctas, 128, dyn_smem>>>(d_dram, d_sink_f, dram_mask, steps, mma_iters);
          else closed_loop_mma_kernel<4><<<active_ctas, 128, dyn_smem>>>(d_dram, d_sink_f, dram_mask, steps, mma_iters);
        };

        for (int w = 0; w < warmup; ++w) {
          l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink_u4, flush_vec4, 1);
          launch_once();
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> bw_s, turn_s, tf_s, util_s;
        bw_s.reserve(samples);
        turn_s.reserve(samples);
        tf_s.reserve(samples);
        util_s.reserve(samples);

        for (int s = 0; s < samples; ++s) {
          l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink_u4, flush_vec4, 1);
          CUDA_CHECK(cudaEventRecord(ev_start));
          launch_once();
          CUDA_CHECK(cudaEventRecord(ev_stop));
          CUDA_CHECK(cudaEventSynchronize(ev_stop));
          float ms = 0.0f;
          CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
          double sec = static_cast<double>(ms) / 1e3;
          double tf_per_sm = (total_flops_per_sm / 1e12) / sec;
          bw_s.push_back((total_bytes / 1e9) / sec);
          turn_s.push_back((static_cast<double>(ms) * 1e6) / static_cast<double>(steps));
          tf_s.push_back(tf_per_sm);
          util_s.push_back((peak_tflops_per_sm > 0.0) ? (tf_per_sm / peak_tflops_per_sm) : 0.0);
        }

        if (!first) std::printf(",");
        first = false;
        std::printf(
            "\n    {\"ctas_per_sm\": %d, \"active_ctas\": %d, \"tokens\": %d, \"num_warps\": 4, "
            "\"stage_bytes\": %d, \"ldmatrix_bytes_per_stage\": 0, \"mma_iters_per_stage\": %d, "
            "\"solo_compute_ns\": %.2f, \"bw_gbps_samples\": [",
            cps, active_ctas, tokens, stage_bytes, mma_iters, solo_tc_ns);
        for (size_t i = 0; i < bw_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.3f", bw_s[i]); }
        std::printf("], \"turnaround_ns_samples\": [");
        for (size_t i = 0; i < turn_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.2f", turn_s[i]); }
        std::printf("], \"mma_tflops_per_sm_samples\": [");
        for (size_t i = 0; i < tf_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.4f", tf_s[i]); }
        std::printf("], \"tensor_util_samples\": [");
        for (size_t i = 0; i < util_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.4f", util_s[i]); }
        std::printf("]}");
      }
    }
  }

  // 8-warp contention sweep: ctas_per_sm in {1, 2}, tokens in {1, 2, 4},
  // ldmatrix_bytes_per_stage in {0, 16384, 32768}.
  const int cal_iters_8w = 10000;
  for (int w = 0; w < 2; ++w) {
    mma_m16n8k16_kernel<false><<<sm_count, 256>>>(d_sink_f, cal_iters_8w);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(ev_start));
  mma_m16n8k16_kernel<false><<<sm_count, 256>>>(d_sink_f, cal_iters_8w);
  CUDA_CHECK(cudaEventRecord(ev_stop));
  CUDA_CHECK(cudaEventSynchronize(ev_stop));
  float cal_ms_8w = 1.0f;
  CUDA_CHECK(cudaEventElapsedTime(&cal_ms_8w, ev_start, ev_stop));
  const double ns_per_mma_iter_8w = (static_cast<double>(cal_ms_8w) * 1e6) / static_cast<double>(cal_iters_8w);
  const double flops_per_cta_iter_8w = 8.0 * 4.0 * 4096.0;
  const double peak_tflops_per_sm_8w = (flops_per_cta_iter_8w / 1e12) / (ns_per_mma_iter_8w / 1e9);

  for (int ld_passes : {0, 4, 8}) {
    const int ldm_bytes = ld_passes * 4096;
    for (int cps : {1, 2}) {
      int active_ctas = sm_count * cps;
      for (int tokens : {1, 2, 4}) {
        size_t dyn_smem = static_cast<size_t>(tokens) * static_cast<size_t>(stage_bytes);
        for (double target_tc : target_tc_ns_list) {
          int mma_iters = (target_tc <= 0.0) ? 0 : std::max(1, static_cast<int>(target_tc / ns_per_mma_iter_8w + 0.5));
          double solo_tc_ns = static_cast<double>(mma_iters) * ns_per_mma_iter_8w;
          double total_bytes = static_cast<double>(active_ctas) * static_cast<double>(stage_bytes) * steps;
          double total_flops_per_sm = static_cast<double>(cps) * flops_per_cta_iter_8w * mma_iters * steps;

          auto launch_8w = [&]() {
            if (ld_passes == 0) dispatch_closed_loop_mma_8w<0>(tokens, active_ctas, dyn_smem, d_dram, d_sink_f, dram_mask, steps, mma_iters);
            else if (ld_passes == 4) dispatch_closed_loop_mma_8w<4>(tokens, active_ctas, dyn_smem, d_dram, d_sink_f, dram_mask, steps, mma_iters);
            else dispatch_closed_loop_mma_8w<8>(tokens, active_ctas, dyn_smem, d_dram, d_sink_f, dram_mask, steps, mma_iters);
          };

          for (int w = 0; w < warmup; ++w) {
            l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink_u4, flush_vec4, 1);
            launch_8w();
          }
          CUDA_CHECK(cudaDeviceSynchronize());

          std::vector<double> bw_s, turn_s, tf_s, util_s;
          bw_s.reserve(samples);
          turn_s.reserve(samples);
          tf_s.reserve(samples);
          util_s.reserve(samples);

          for (int s = 0; s < samples; ++s) {
            l2_read_cg_kernel<<<sm_count * 4, 512>>>(d_flush, d_sink_u4, flush_vec4, 1);
            CUDA_CHECK(cudaEventRecord(ev_start));
            launch_8w();
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            double sec = static_cast<double>(ms) / 1e3;
            double tf_per_sm = (total_flops_per_sm / 1e12) / sec;
            bw_s.push_back((total_bytes / 1e9) / sec);
            turn_s.push_back((static_cast<double>(ms) * 1e6) / static_cast<double>(steps));
            tf_s.push_back(tf_per_sm);
            util_s.push_back((peak_tflops_per_sm_8w > 0.0) ? (tf_per_sm / peak_tflops_per_sm_8w) : 0.0);
          }

          if (!first) std::printf(",");
          first = false;
          std::printf(
              "\n    {\"ctas_per_sm\": %d, \"active_ctas\": %d, \"tokens\": %d, \"num_warps\": 8, "
              "\"stage_bytes\": %d, \"ldmatrix_bytes_per_stage\": %d, \"mma_iters_per_stage\": %d, "
              "\"solo_compute_ns\": %.2f, \"bw_gbps_samples\": [",
              cps, active_ctas, tokens, stage_bytes, ldm_bytes, mma_iters, solo_tc_ns);
          for (size_t i = 0; i < bw_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.3f", bw_s[i]); }
          std::printf("], \"turnaround_ns_samples\": [");
          for (size_t i = 0; i < turn_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.2f", turn_s[i]); }
          std::printf("], \"mma_tflops_per_sm_samples\": [");
          for (size_t i = 0; i < tf_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.4f", tf_s[i]); }
          std::printf("], \"tensor_util_samples\": [");
          for (size_t i = 0; i < util_s.size(); ++i) { if (i) std::printf(", "); std::printf("%.4f", util_s[i]); }
          std::printf("]}");
        }
      }
    }
  }
  std::printf("\n  ]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_dram));
  CUDA_CHECK(cudaFree(d_flush));
  CUDA_CHECK(cudaFree(d_sink_u4));
  CUDA_CHECK(cudaFree(d_sink_f));
  return 0;
}

// ---------------------------------------------------------------------------
// 13. MMA chain length & ldmatrix serial stage overhead (1 CTA/SM, no GMEM)
// ---------------------------------------------------------------------------

template <int M_CHAIN, int LD_PER_WARP, bool DEPENDENT>
__global__ void mma_chain_stage_kernel(const uint4* __restrict__ init_src,
                                       float* __restrict__ sink,
                                       int steps) {
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid >> 5;
  const uint32_t lane = tid & 31u;
  const uint32_t num_warps = blockDim.x >> 5;
  const size_t tile_u4 = static_cast<size_t>(LD_PER_WARP) * static_cast<size_t>(num_warps) * 32ULL;

  // Initialize shared memory from runtime buffer (filled with f16 1.0) so ptxas cannot constant-fold.
  if constexpr (LD_PER_WARP > 0) {
    for (size_t i = tid; i < 2ULL * tile_u4; i += blockDim.x) {
      smem[i] = init_src[i];
    }
    __syncthreads();
  }

  constexpr int REGS_N = (LD_PER_WARP > 0) ? LD_PER_WARP : 1;
  uint32_t reg[REGS_N][4];
  #pragma unroll
  for (int r = 0; r < REGS_N; ++r) {
    reg[r][0] = 0x3c003c00u;
    reg[r][1] = 0x3c003c00u;
    reg[r][2] = 0x3c003c00u;
    reg[r][3] = 0x3c003c00u;
  }
  uint32_t live_acc = 0u;
  float c00 = 0.f, c01 = 0.f, c02 = 0.f, c03 = 0.f;
  float c10 = 0.f, c11 = 0.f, c12 = 0.f, c13 = 0.f;
  float c20 = 0.f, c21 = 0.f, c22 = 0.f, c23 = 0.f;
  float c30 = 0.f, c31 = 0.f, c32 = 0.f, c33 = 0.f;

  for (int step = 0; step < steps; ++step) {
    if constexpr (LD_PER_WARP > 0) {
      const size_t slot_base = static_cast<size_t>(step & 1) * tile_u4;
      #pragma unroll
      for (int r = 0; r < LD_PER_WARP; ++r) {
        size_t off_u4 = slot_base + (static_cast<size_t>(r) * num_warps + warp_id) * 32ULL + lane;
        uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off_u4));
        asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(reg[r][0]), "=r"(reg[r][1]), "=r"(reg[r][2]), "=r"(reg[r][3])
                     : "r"(saddr) : "memory");
      }
    }

    if constexpr (DEPENDENT) {
      #pragma unroll
      for (int i = 0; i < M_CHAIN; ++i) {
        const int ra = (LD_PER_WARP > 0) ? (i % REGS_N) : 0;
        const int rb = (LD_PER_WARP > 0) ? ((i + 1) % REGS_N) : 0;
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c00), "+f"(c01), "+f"(c02), "+f"(c03)
            : "r"(reg[ra][0]), "r"(reg[ra][1]), "r"(reg[ra][2]), "r"(reg[ra][3]),
              "r"(reg[rb][0]), "r"(reg[rb][1]));
      }
    } else {
      constexpr int FULL_GROUPS = M_CHAIN / 4;
      constexpr int REM = M_CHAIN % 4;
      #pragma unroll
      for (int g = 0; g < FULL_GROUPS; ++g) {
        const int i0 = g * 4;
        const int ra0 = (LD_PER_WARP > 0) ? ((i0 + 0) % REGS_N) : 0;
        const int rb0 = (LD_PER_WARP > 0) ? ((i0 + 1) % REGS_N) : 0;
        const int ra1 = (LD_PER_WARP > 0) ? ((i0 + 1) % REGS_N) : 0;
        const int rb1 = (LD_PER_WARP > 0) ? ((i0 + 2) % REGS_N) : 0;
        const int ra2 = (LD_PER_WARP > 0) ? ((i0 + 2) % REGS_N) : 0;
        const int rb2 = (LD_PER_WARP > 0) ? ((i0 + 3) % REGS_N) : 0;
        const int ra3 = (LD_PER_WARP > 0) ? ((i0 + 3) % REGS_N) : 0;
        const int rb3 = (LD_PER_WARP > 0) ? ((i0 + 4) % REGS_N) : 0;
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c00), "+f"(c01), "+f"(c02), "+f"(c03)
            : "r"(reg[ra0][0]), "r"(reg[ra0][1]), "r"(reg[ra0][2]), "r"(reg[ra0][3]),
              "r"(reg[rb0][0]), "r"(reg[rb0][1]));
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c10), "+f"(c11), "+f"(c12), "+f"(c13)
            : "r"(reg[ra1][0]), "r"(reg[ra1][1]), "r"(reg[ra1][2]), "r"(reg[ra1][3]),
              "r"(reg[rb1][0]), "r"(reg[rb1][1]));
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c20), "+f"(c21), "+f"(c22), "+f"(c23)
            : "r"(reg[ra2][0]), "r"(reg[ra2][1]), "r"(reg[ra2][2]), "r"(reg[ra2][3]),
              "r"(reg[rb2][0]), "r"(reg[rb2][1]));
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c30), "+f"(c31), "+f"(c32), "+f"(c33)
            : "r"(reg[ra3][0]), "r"(reg[ra3][1]), "r"(reg[ra3][2]), "r"(reg[ra3][3]),
              "r"(reg[rb3][0]), "r"(reg[rb3][1]));
      }
      if constexpr (REM >= 1) {
        const int rb0 = (LD_PER_WARP > 0) ? (1 % REGS_N) : 0;
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c00), "+f"(c01), "+f"(c02), "+f"(c03)
            : "r"(reg[0][0]), "r"(reg[0][1]), "r"(reg[0][2]), "r"(reg[0][3]),
              "r"(reg[rb0][0]), "r"(reg[rb0][1]));
      }
      if constexpr (REM >= 2) {
        const int ra1 = (LD_PER_WARP > 0) ? (1 % REGS_N) : 0;
        const int rb1 = (LD_PER_WARP > 0) ? (2 % REGS_N) : 0;
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c10), "+f"(c11), "+f"(c12), "+f"(c13)
            : "r"(reg[ra1][0]), "r"(reg[ra1][1]), "r"(reg[ra1][2]), "r"(reg[ra1][3]),
              "r"(reg[rb1][0]), "r"(reg[rb1][1]));
      }
      if constexpr (REM >= 3) {
        const int ra2 = (LD_PER_WARP > 0) ? (2 % REGS_N) : 0;
        const int rb2 = (LD_PER_WARP > 0) ? (3 % REGS_N) : 0;
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c20), "+f"(c21), "+f"(c22), "+f"(c23)
            : "r"(reg[ra2][0]), "r"(reg[ra2][1]), "r"(reg[ra2][2]), "r"(reg[ra2][3]),
              "r"(reg[rb2][0]), "r"(reg[rb2][1]));
      }
    }

    // Keep any ldmatrix outputs live when M_CHAIN < LD_PER_WARP so ptxas cannot elide them.
    if constexpr (LD_PER_WARP > M_CHAIN) {
      #pragma unroll
      for (int r = M_CHAIN; r < LD_PER_WARP; ++r) {
        live_acc ^= (reg[r][0] ^ reg[r][1] ^ reg[r][2] ^ reg[r][3]);
      }
    }

    asm volatile("bar.sync 0;\n" ::: "memory");
  }

  float sum = c00 + c01 + c02 + c03 + c10 + c11 + c12 + c13 +
              c20 + c21 + c22 + c23 + c30 + c31 + c32 + c33 + static_cast<float>(live_acc & 1u);
  if (sum == -999999.0f) {
    sink[blockIdx.x * blockDim.x + tid] = sum;
  }
}

template <int LD_PER_WARP, bool DEPENDENT>
static void dispatch_mma_chain_len(int m_chain, int grid, int threads, size_t dyn_smem,
                                   const uint4* d_init, float* d_sink, int steps) {
  switch (m_chain) {
    case 1: mma_chain_stage_kernel<1, LD_PER_WARP, DEPENDENT><<<grid, threads, dyn_smem>>>(d_init, d_sink, steps); break;
    case 2: mma_chain_stage_kernel<2, LD_PER_WARP, DEPENDENT><<<grid, threads, dyn_smem>>>(d_init, d_sink, steps); break;
    case 4: mma_chain_stage_kernel<4, LD_PER_WARP, DEPENDENT><<<grid, threads, dyn_smem>>>(d_init, d_sink, steps); break;
    case 8: mma_chain_stage_kernel<8, LD_PER_WARP, DEPENDENT><<<grid, threads, dyn_smem>>>(d_init, d_sink, steps); break;
    case 16: mma_chain_stage_kernel<16, LD_PER_WARP, DEPENDENT><<<grid, threads, dyn_smem>>>(d_init, d_sink, steps); break;
    case 32: mma_chain_stage_kernel<32, LD_PER_WARP, DEPENDENT><<<grid, threads, dyn_smem>>>(d_init, d_sink, steps); break;
    case 64: mma_chain_stage_kernel<64, LD_PER_WARP, DEPENDENT><<<grid, threads, dyn_smem>>>(d_init, d_sink, steps); break;
    case 128: mma_chain_stage_kernel<128, LD_PER_WARP, DEPENDENT><<<grid, threads, dyn_smem>>>(d_init, d_sink, steps); break;
    default: break;
  }
}

template <bool DEPENDENT>
static void dispatch_mma_chain(int ld_per_warp, int m_chain, int grid, int threads, size_t dyn_smem,
                               const uint4* d_init, float* d_sink, int steps) {
  switch (ld_per_warp) {
    case 0: dispatch_mma_chain_len<0, DEPENDENT>(m_chain, grid, threads, dyn_smem, d_init, d_sink, steps); break;
    case 2: dispatch_mma_chain_len<2, DEPENDENT>(m_chain, grid, threads, dyn_smem, d_init, d_sink, steps); break;
    case 3: dispatch_mma_chain_len<3, DEPENDENT>(m_chain, grid, threads, dyn_smem, d_init, d_sink, steps); break;
    case 4: dispatch_mma_chain_len<4, DEPENDENT>(m_chain, grid, threads, dyn_smem, d_init, d_sink, steps); break;
    case 6: dispatch_mma_chain_len<6, DEPENDENT>(m_chain, grid, threads, dyn_smem, d_init, d_sink, steps); break;
    case 8: dispatch_mma_chain_len<8, DEPENDENT>(m_chain, grid, threads, dyn_smem, d_init, d_sink, steps); break;
    default: break;
  }
}

static int run_mma_chain(int device, int samples, int warmup) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;

  const size_t max_init_u4 = (2ULL * 16384ULL) / sizeof(uint4);
  std::vector<uint4> h_init(max_init_u4, make_uint4(0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u));
  uint4* d_init = nullptr;
  float* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_init, max_init_u4 * sizeof(uint4)));
  CUDA_CHECK(cudaMemcpy(d_init, h_init.data(), max_init_u4 * sizeof(uint4), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(sm_count) * 256 * sizeof(float)));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  struct ChainConfig {
    const char* variant;
    int tile_bytes;
    bool dependent;
  };
  const std::vector<ChainConfig> configs = {
      {"mma_only_independent", 0, false},
      {"mma_only_dependent", 0, true},
      {"ldmatrix_independent", 8192, false},
      {"ldmatrix_independent", 12288, false},
      {"ldmatrix_independent", 16384, false},
      {"ldmatrix_dependent", 8192, true},
      {"ldmatrix_dependent", 12288, true},
      {"ldmatrix_dependent", 16384, true},
  };
  const std::vector<int> warp_list = {4, 8};
  const std::vector<int> m_list = {1, 2, 4, 8, 16, 32, 64, 128};

  std::printf("{\n  \"samples\": %d,\n  \"mma_chain_latency\": [", samples);
  bool first = true;

  for (const auto& cfg : configs) {
    for (int warps : warp_list) {
      int threads = warps * 32;
      int ld_per_warp = cfg.tile_bytes / (warps * 512);
      size_t dyn_smem = static_cast<size_t>(cfg.tile_bytes) * 2ULL;

      for (int m : m_list) {
        // Keep each kernel launch ~0.35 ms to avoid GPU Boost thermal/power droop at large m.
        int est_ns = std::max(20, (warps / 4) * 12 * m + 30);
        int steps = std::max(128, std::min(16384, 350000 / est_ns));
        auto launch_once = [&]() {
          if (cfg.dependent) {
            dispatch_mma_chain<true>(ld_per_warp, m, sm_count, threads, dyn_smem, d_init, d_sink, steps);
          } else {
            dispatch_mma_chain<false>(ld_per_warp, m, sm_count, threads, dyn_smem, d_init, d_sink, steps);
          }
        };

        for (int w = 0; w < warmup; ++w) launch_once();
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> ns_samples;
        ns_samples.reserve(samples);
        for (int s = 0; s < samples; ++s) {
          CUDA_CHECK(cudaEventRecord(ev_start));
          launch_once();
          CUDA_CHECK(cudaEventRecord(ev_stop));
          CUDA_CHECK(cudaEventSynchronize(ev_stop));
          float ms = 0.0f;
          CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
          ns_samples.push_back((static_cast<double>(ms) * 1e6) / static_cast<double>(steps));
        }

        if (!first) std::printf(",");
        first = false;
        std::printf(
            "\n    {\"variant\": \"%s\", \"num_warps\": %d, \"tile_bytes\": %d, "
            "\"ldmatrix_x4_per_warp\": %d, \"mma_chain_len\": %d, \"stage_time_ns_samples\": [",
            cfg.variant, warps, cfg.tile_bytes, ld_per_warp, m);
        for (size_t i = 0; i < ns_samples.size(); ++i) {
          if (i) std::printf(", ");
          std::printf("%.3f", ns_samples[i]);
        }
        std::printf("]}");
      }
    }
  }
  std::printf("\n  ]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_init));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}

// ---------------------------------------------------------------------------
// 14. Serial stage fill & ldmatrix read (1 CTA/SM, hot L2, S=1, no MMA)
// ---------------------------------------------------------------------------

template <int STAGE_BYTES, int NUM_WARPS, bool USE_CP_ASYNC, bool WITH_LDMATRIX>
__global__ void serial_stage_fill_kernel(const uint4* __restrict__ src,
                                         uint32_t* __restrict__ sink,
                                         size_t l2_mask,
                                         int steps) {
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const uint32_t warp_id = tid >> 5;
  const uint32_t lane = tid & 31u;
  constexpr size_t STAGE_U4 = static_cast<size_t>(STAGE_BYTES) / sizeof(uint4);
  const size_t grid_stage_u4 = static_cast<size_t>(gridDim.x) * STAGE_U4;
  size_t base = (static_cast<size_t>(blockIdx.x) * STAGE_U4) & l2_mask;
  uint32_t acc = 0u;

  for (int step = 0; step < steps; ++step) {
    if constexpr (USE_CP_ASYNC) {
      if constexpr (STAGE_BYTES == 2048 && NUM_WARPS == 8) {
        if (tid < 128) {
          uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + tid));
          const void* gptr = src + ((base + tid) & l2_mask);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(gptr) : "memory");
        }
      } else {
        constexpr int VECS = STAGE_BYTES / (NUM_WARPS * 32 * 16);
        #pragma unroll
        for (int v = 0; v < VECS; ++v) {
          size_t off = static_cast<size_t>(v) * (NUM_WARPS * 32) + tid;
          uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off));
          const void* gptr = src + ((base + off) & l2_mask);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr), "l"(gptr) : "memory");
        }
      }
      asm volatile("cp.async.commit_group;\n" ::: "memory");
      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    } else {
      if constexpr (STAGE_BYTES == 2048 && NUM_WARPS == 8) {
        if (tid < 128) {
          uint4 val = ld_cg_u4(src + ((base + tid) & l2_mask));
          uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + tid));
          asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n"
                       :: "r"(saddr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w) : "memory");
        }
      } else {
        constexpr int VECS = STAGE_BYTES / (NUM_WARPS * 32 * 16);
        uint4 vals[VECS];
        #pragma unroll
        for (int v = 0; v < VECS; ++v) {
          size_t off = static_cast<size_t>(v) * (NUM_WARPS * 32) + tid;
          vals[v] = ld_cg_u4(src + ((base + off) & l2_mask));
        }
        #pragma unroll
        for (int v = 0; v < VECS; ++v) {
          size_t off = static_cast<size_t>(v) * (NUM_WARPS * 32) + tid;
          uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off));
          asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n"
                       :: "r"(saddr), "r"(vals[v].x), "r"(vals[v].y), "r"(vals[v].z), "r"(vals[v].w) : "memory");
        }
      }
    }

    asm volatile("bar.sync 0;\n" ::: "memory");

    if constexpr (WITH_LDMATRIX) {
      if constexpr (STAGE_BYTES == 2048 && NUM_WARPS == 8) {
        if (warp_id < 4) {
          size_t off_u4 = static_cast<size_t>(warp_id) * 32ULL + lane;
          uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off_u4));
          uint32_t r0, r1, r2, r3;
          asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                       : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(saddr) : "memory");
          acc ^= (r0 ^ r1 ^ r2 ^ r3);
        }
      } else {
        constexpr int LD_N = STAGE_BYTES / (NUM_WARPS * 512);
        #pragma unroll
        for (int r = 0; r < LD_N; ++r) {
          size_t off_u4 = (static_cast<size_t>(r) * NUM_WARPS + warp_id) * 32ULL + lane;
          uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem + off_u4));
          uint32_t r0, r1, r2, r3;
          asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                       : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(saddr) : "memory");
          acc ^= (r0 ^ r1 ^ r2 ^ r3);
        }
      }
      asm volatile("bar.sync 0;\n" ::: "memory");
    }

    base = (base + grid_stage_u4) & l2_mask;
  }

  uint4 tail = smem[tid & (STAGE_U4 - 1ULL)];
  acc ^= (tail.x ^ tail.y ^ tail.z ^ tail.w);
  if (acc == 0xdeadbeefu) {
    sink[blockIdx.x * blockDim.x + tid] = acc;
  }
}

template <int NUM_WARPS, bool USE_CP_ASYNC, bool WITH_LDMATRIX>
static void dispatch_serial_fill_bytes(int stage_bytes, int grid, const uint4* d_l2,
                                       uint32_t* d_sink, size_t l2_mask, int steps) {
  const int threads = NUM_WARPS * 32;
  const size_t dyn_smem = static_cast<size_t>(stage_bytes);
  switch (stage_bytes) {
    case 2048:
      serial_stage_fill_kernel<2048, NUM_WARPS, USE_CP_ASYNC, WITH_LDMATRIX>
          <<<grid, threads, dyn_smem>>>(d_l2, d_sink, l2_mask, steps);
      break;
    case 4096:
      serial_stage_fill_kernel<4096, NUM_WARPS, USE_CP_ASYNC, WITH_LDMATRIX>
          <<<grid, threads, dyn_smem>>>(d_l2, d_sink, l2_mask, steps);
      break;
    case 8192:
      serial_stage_fill_kernel<8192, NUM_WARPS, USE_CP_ASYNC, WITH_LDMATRIX>
          <<<grid, threads, dyn_smem>>>(d_l2, d_sink, l2_mask, steps);
      break;
    case 16384:
      serial_stage_fill_kernel<16384, NUM_WARPS, USE_CP_ASYNC, WITH_LDMATRIX>
          <<<grid, threads, dyn_smem>>>(d_l2, d_sink, l2_mask, steps);
      break;
    default:
      break;
  }
}

static int run_serial_fill(int device, int samples, int warmup) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sm_count = prop.multiProcessorCount;

  // 4 MiB working set (<< 36 MiB L2 on RTX 4070), pre-warmed in L2 before timing.
  const size_t l2_bytes = 4ULL * 1024ULL * 1024ULL;
  const size_t l2_mask = (l2_bytes / sizeof(uint4)) - 1ULL;
  uint4* d_l2 = nullptr;
  uint32_t* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_l2, l2_bytes));
  CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(sm_count) * 256 * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_l2, 0x3c, l2_bytes));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  const std::vector<int> grid_list = {1, sm_count};
  const std::vector<int> warp_list = {4, 8};
  const std::vector<int> bytes_list = {2048, 4096, 8192, 16384};
  const int steps = 4096;

  std::printf("{\n  \"samples\": %d,\n  \"serial_stage_fill\": [", samples);
  bool first = true;

  for (int grid : grid_list) {
    for (bool use_cp : {true, false}) {
      const char* t_mode = use_cp ? "cp_async" : "ld_st";
      for (bool with_ldm : {false, true}) {
        const char* r_mode = with_ldm ? "fill_and_ldmatrix" : "fill_only";
        for (int warps : warp_list) {
          for (int sb : bytes_list) {
            auto launch_once = [&]() {
              if (warps == 4) {
                if (use_cp && !with_ldm) dispatch_serial_fill_bytes<4, true, false>(sb, grid, d_l2, d_sink, l2_mask, steps);
                else if (use_cp && with_ldm) dispatch_serial_fill_bytes<4, true, true>(sb, grid, d_l2, d_sink, l2_mask, steps);
                else if (!use_cp && !with_ldm) dispatch_serial_fill_bytes<4, false, false>(sb, grid, d_l2, d_sink, l2_mask, steps);
                else dispatch_serial_fill_bytes<4, false, true>(sb, grid, d_l2, d_sink, l2_mask, steps);
              } else {
                if (use_cp && !with_ldm) dispatch_serial_fill_bytes<8, true, false>(sb, grid, d_l2, d_sink, l2_mask, steps);
                else if (use_cp && with_ldm) dispatch_serial_fill_bytes<8, true, true>(sb, grid, d_l2, d_sink, l2_mask, steps);
                else if (!use_cp && !with_ldm) dispatch_serial_fill_bytes<8, false, false>(sb, grid, d_l2, d_sink, l2_mask, steps);
                else dispatch_serial_fill_bytes<8, false, true>(sb, grid, d_l2, d_sink, l2_mask, steps);
              }
            };

            for (int w = 0; w < warmup; ++w) launch_once();
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<double> ns_samples;
            std::vector<double> bpns_samples;
            ns_samples.reserve(samples);
            bpns_samples.reserve(samples);
            for (int s = 0; s < samples; ++s) {
              CUDA_CHECK(cudaEventRecord(ev_start));
              launch_once();
              CUDA_CHECK(cudaEventRecord(ev_stop));
              CUDA_CHECK(cudaEventSynchronize(ev_stop));
              float ms = 0.0f;
              CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
              double stage_ns = (static_cast<double>(ms) * 1e6) / static_cast<double>(steps);
              ns_samples.push_back(stage_ns);
              bpns_samples.push_back(static_cast<double>(sb) / stage_ns);
            }

            if (!first) std::printf(",");
            first = false;
            std::printf(
                "\n    {\"transfer_mode\": \"%s\", \"read_mode\": \"%s\", \"num_warps\": %d, "
                "\"stage_bytes\": %d, \"active_ctas\": %d, \"stage_time_ns_samples\": [",
                t_mode, r_mode, warps, sb, grid);
            for (size_t i = 0; i < ns_samples.size(); ++i) {
              if (i) std::printf(", ");
              std::printf("%.3f", ns_samples[i]);
            }
            std::printf("], \"effective_bytes_per_ns_samples\": [");
            for (size_t i = 0; i < bpns_samples.size(); ++i) {
              if (i) std::printf(", ");
              std::printf("%.3f", bpns_samples[i]);
            }
            std::printf("]}");
          }
        }
      }
    }
  }
  std::printf("\n  ]\n}\n");

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaFree(d_l2));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}
