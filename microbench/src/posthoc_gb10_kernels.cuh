// Post-hoc microbenchmarks for the v0.2 GB10 GEMM hypotheses H3 and H4
// (docs/results/analysis/v0.3-gb10-h3-h4-posthoc.md).
//
// wave-rescan:  GEMM B-panel streams read again by later waves (cross-wave L2 reuse).
// b-stream:     one wave of B-panel streams at DRAM, the M = 48 GEMM shape.
// staged-mma:   closed loop with 16 or 32 KiB stages, MMA and ldmatrix, L2 resident.
//
// Every mode prints, before and after each block of points, a same-session DRAM
// read ceiling with its samples and host UTC time (ms since the epoch).

#include <chrono>

// ---------------------------------------------------------------------------
// B-panel stream kernel
// ---------------------------------------------------------------------------
//
// CTA b reads panel (b % panels) of a row-major [k_rows, panels * row_bytes]
// matrix: each stage is (STAGE / row_bytes) rows of row_bytes bytes, one
// row_stride apart, like a Triton GEMM B tile. TOKENS stage buffers, so the
// prefetch distance is TOKENS - 1. Each stage then runs mma_iters x 4
// m16n8k16 MMAs per warp. A CTA reads its panel from its last stage to its
// first when reverse_mode is 2, or when it is 1 and its wave
// (b / wave_slots) is odd.

template <int TOKENS, int VPT>
__global__ void __launch_bounds__(128) b_stream_kernel(const uint8_t* __restrict__ src,
                                                       uint4* __restrict__ sink,
                                                       uint32_t panels,
                                                       uint32_t row_bytes,
                                                       uint64_t row_stride,
                                                       uint32_t k_steps,
                                                       uint32_t mma_iters,
                                                       uint32_t wave_slots,
                                                       uint32_t reverse_mode) {
  constexpr uint32_t kStage = 128u * 16u * VPT;
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x;
  const uint32_t panel = blockIdx.x % panels;
  const bool reverse = reverse_mode == 2u || (reverse_mode == 1u && ((blockIdx.x / wave_slots) & 1u));
  const uint32_t rows_per_stage = kStage / row_bytes;
  const uint8_t* panel_base = src + static_cast<uint64_t>(panel) * row_bytes;

  auto issue = [&](uint32_t slot, uint32_t step) {
    const uint32_t k = reverse ? k_steps - 1u - step : step;
    uint4* slot_ptr = smem + static_cast<size_t>(slot) * (kStage / 16u);
    #pragma unroll
    for (int v = 0; v < VPT; ++v) {
      const uint32_t byte = (static_cast<uint32_t>(v) * 128u + tid) * 16u;
      const uint32_t row = byte / row_bytes;
      const uint32_t col = byte % row_bytes;
      const uint8_t* g = panel_base +
                         (static_cast<uint64_t>(k) * rows_per_stage + row) * row_stride + col;
      const uint32_t s = static_cast<uint32_t>(
          __cvta_generic_to_shared(slot_ptr + static_cast<size_t>(v) * 128u + tid));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(g) : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  uint32_t issued = 0;
  #pragma unroll
  for (int s = 0; s < TOKENS; ++s) {
    if (issued < k_steps) issue(issued % TOKENS, issued), ++issued;
  }
  uint32_t acc = 0;
  float c[4][4] = {};
  for (uint32_t step = 0; step < k_steps; ++step) {
    if constexpr (TOKENS == 1) asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    else if constexpr (TOKENS == 2) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    else if constexpr (TOKENS == 3) asm volatile("cp.async.wait_group 2;\n" ::: "memory");
    else asm volatile("cp.async.wait_group 3;\n" ::: "memory");
    __syncthreads();
    acc ^= smem[static_cast<size_t>(step % TOKENS) * (kStage / 16u) + tid].x;
    const uint32_t a = 0x3c003c00u + (acc & 7u);
    for (uint32_t m = 0; m < mma_iters; ++m) {
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
            : "r"(a), "r"(a), "r"(a), "r"(a), "r"(a), "r"(a));
      }
    }
    __syncthreads();
    if (issued < k_steps) issue(step % TOKENS, issued), ++issued;
  }
  asm volatile("cp.async.wait_all;\n" ::: "memory");
  float sum = 0.0f;
  #pragma unroll
  for (int j = 0; j < 4; ++j) sum += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  if (acc == 0xdeadbeefu || sum == -999999.0f) sink[tid] = make_uint4(acc, acc, acc, acc);
}

struct BStreamPoint {
  int tokens;
  int vpt;
  int ctas_per_sm;
  uint32_t grid;
  uint32_t panels;
  uint32_t row_bytes;
  uint64_t row_stride;
  uint32_t k_steps;
  uint32_t mma_iters;
  uint32_t reverse_mode;
  uint32_t wave_slots;
};

template <int TOKENS, int VPT>
static int b_stream_launch(const BStreamPoint& p, const uint8_t* src, uint4* sink, size_t smem,
                           bool query_only) {
  auto kernel = b_stream_kernel<TOKENS, VPT>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(smem)));
  if (query_only) {
    int blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, 128, smem));
    return blocks;
  }
  kernel<<<p.grid, 128, smem>>>(src, sink, p.panels, p.row_bytes, p.row_stride, p.k_steps,
                                p.mma_iters, p.wave_slots, p.reverse_mode);
  return 0;
}

template <int TOKENS>
static int b_stream_dispatch_vpt(const BStreamPoint& p, const uint8_t* src, uint4* sink,
                                 size_t smem, bool query_only) {
  switch (p.vpt) {
    case 2: return b_stream_launch<TOKENS, 2>(p, src, sink, smem, query_only);
    case 4: return b_stream_launch<TOKENS, 4>(p, src, sink, smem, query_only);
    case 8: return b_stream_launch<TOKENS, 8>(p, src, sink, smem, query_only);
    default: return b_stream_launch<TOKENS, 16>(p, src, sink, smem, query_only);
  }
}

// Returns the occupancy when query_only, else launches.
static int b_stream_run(const BStreamPoint& p, const uint8_t* src, uint4* sink, bool query_only) {
  // Pad dynamic shared memory so exactly ctas_per_sm CTAs fit on an SM.
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  int per_sm = 0, reserved = 0, per_block = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  const size_t need = static_cast<size_t>(p.tokens) * 128u * 16u * p.vpt;
  size_t smem = static_cast<size_t>(per_sm / p.ctas_per_sm - reserved) / 128u * 128u;
  smem = std::min<size_t>(smem, static_cast<size_t>(per_block));
  if (smem < need) return query_only ? 0 : -1;
  switch (p.tokens) {
    case 1: return b_stream_dispatch_vpt<1>(p, src, sink, smem, query_only);
    case 2: return b_stream_dispatch_vpt<2>(p, src, sink, smem, query_only);
    case 3: return b_stream_dispatch_vpt<3>(p, src, sink, smem, query_only);
    default: return b_stream_dispatch_vpt<4>(p, src, sink, smem, query_only);
  }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

struct PosthocBuffers {
  uint8_t* data = nullptr;
  uint4* flush = nullptr;
  uint4* sink = nullptr;
  size_t data_bytes = 0;
  size_t flush_bytes = 0;
};

static PosthocBuffers posthoc_alloc(size_t data_bytes, size_t l2_bytes) {
  PosthocBuffers b;
  b.data_bytes = data_bytes;
  b.flush_bytes = std::max<size_t>(3 * l2_bytes, 128ULL << 20);
  CUDA_CHECK(cudaMalloc(&b.data, data_bytes));
  CUDA_CHECK(cudaMalloc(&b.flush, b.flush_bytes));
  CUDA_CHECK(cudaMalloc(&b.sink, 1024 * sizeof(uint4)));
  CUDA_CHECK(cudaMemset(b.data, 0x3c, data_bytes));
  CUDA_CHECK(cudaMemset(b.flush, 0x11, b.flush_bytes));
  CUDA_CHECK(cudaDeviceSynchronize());
  return b;
}

static void posthoc_free(PosthocBuffers& b) {
  CUDA_CHECK(cudaFree(b.data));
  CUDA_CHECK(cudaFree(b.flush));
  CUDA_CHECK(cudaFree(b.sink));
}

// Read-only flush: leaves L2 clean and holding unrelated lines.
static void posthoc_flush(const PosthocBuffers& b, int sm_count) {
  l2_read_cg_kernel<<<sm_count * 4, 512>>>(b.flush, b.sink, b.flush_bytes / sizeof(uint4), 1);
}

static float posthoc_time_ms(cudaEvent_t e0, cudaEvent_t e1) {
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  return ms;
}

static void print_samples(const char* key, const std::vector<double>& v);

// Same-session DRAM read ceiling: ld.global.cg stream over 256 MiB after a clean
// flush. Prints "key": {"gbps_samples": [...], "unix_ms": ...}.
static void print_ceiling(const char* key, const PosthocBuffers& b, int sm_count, cudaEvent_t e0,
                          cudaEvent_t e1, int samples) {
  const size_t bytes = std::min<size_t>(b.data_bytes, 256ULL << 20);
  std::vector<double> v;
  for (int s = 0; s < samples; ++s) {
    posthoc_flush(b, sm_count);
    CUDA_CHECK(cudaEventRecord(e0));
    l2_read_cg_kernel<<<sm_count * 4, 512>>>(reinterpret_cast<const uint4*>(b.data), b.sink,
                                             bytes / sizeof(uint4), 1);
    CUDA_CHECK(cudaEventRecord(e1));
    v.push_back(static_cast<double>(bytes) / (posthoc_time_ms(e0, e1) * 1e6));
  }
  const long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
  std::printf("\"%s\": {", key);
  print_samples("gbps_samples", v);
  std::printf(", \"unix_ms\": %lld}", ms);
}

static void print_samples(const char* key, const std::vector<double>& v) {
  std::printf("\"%s\": [", key);
  for (size_t i = 0; i < v.size(); ++i) std::printf(i ? ", %.3f" : "%.3f", v[i]);
  std::printf("]");
}

// Times one b-stream launch after a clean flush, in ns.
static double b_stream_time_ns(const BStreamPoint& p, const PosthocBuffers& b, int sm_count,
                               cudaEvent_t e0, cudaEvent_t e1) {
  posthoc_flush(b, sm_count);
  CUDA_CHECK(cudaEventRecord(e0));
  if (b_stream_run(p, b.data, b.sink, false) < 0) return -1.0;
  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaGetLastError());
  return posthoc_time_ms(e0, e1) * 1e6;
}

// ---------------------------------------------------------------------------
// wave-rescan (H3)
// ---------------------------------------------------------------------------
//
// One wave holds S = sm_count * ctas_per_sm CTAs and covers rows_per_wave
// rows of P = S / rows_per_wave panels, so each wave reads every panel once
// and later waves read them again, as a GROUP_M = 1 GEMM reads B. The
// footprint of one wave is footprint_ratio * L2. One MMA iteration per stage
// (the core models only kernels with MMA work). Each point is launched with
// 1 and with 4 waves; the difference over 3 is the time of a re-reading wave.

static int run_wave_rescan(int device, int samples, int warmup) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sms = prop.multiProcessorCount;
  const size_t l2 = prop.l2CacheSize;
  const std::vector<double> ratios = {0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0};
  const uint32_t row_bytes = 256;  // BLOCK_N = 128 of f16
  const int vpt = 4;               // 8 KiB stages, BLOCK_K = 32
  const uint32_t stage = 128u * 16u * vpt;
  const uint32_t rows_per_stage = stage / row_bytes;
  PosthocBuffers b = posthoc_alloc(static_cast<size_t>(3.2 * l2) + (64ULL << 20), l2);
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));

  std::printf("{\n  \"mode\": \"wave-rescan\", \"sm_count\": %d, \"l2_bytes\": %zu, \"samples\": %d,\n",
              sms, l2, samples);
  std::printf("  \"row_bytes\": %u, \"stage_bytes\": %u, \"tokens\": 3, \"num_warps\": 4,\n",
              row_bytes, stage);
  std::printf("  \"blocks\": [");
  for (int cps : {1, 2, 4}) {
    std::printf("%s\n    {\"ctas_per_sm\": %d, ", cps == 1 ? "" : ",", cps);
    print_ceiling("ceiling_before", b, sms, e0, e1, samples);
    std::printf(", \"points\": [");
    bool first = true;
    for (int rows_per_wave : {1, 2}) {
      for (double ratio : ratios) {
        const uint32_t slots = static_cast<uint32_t>(sms * cps);
        const uint32_t panels = slots / rows_per_wave;
        const uint64_t row_stride = static_cast<uint64_t>(panels) * row_bytes;
        uint32_t k_steps = static_cast<uint32_t>(ratio * l2 / (row_stride * rows_per_stage) + 0.5);
        k_steps = std::max<uint32_t>(k_steps, 4);
        const uint64_t footprint = row_stride * rows_per_stage * k_steps;
        BStreamPoint p{3, vpt, cps, slots, panels, row_bytes, row_stride, k_steps, 1, 0, slots};
        const int occ = b_stream_run(p, b.data, b.sink, true);
        // Forward: every wave walks K up. Reversed: odd waves walk K down; its
        // one-wave baseline walks K down, so W=4 and W=1 share a direction mix.
        for (uint32_t rev : {0u, 1u}) {
          std::vector<double> t1, t4;
          for (int w = 0; w < warmup; ++w) {
            p.grid = 4 * slots;
            p.reverse_mode = rev;
            b_stream_time_ns(p, b, sms, e0, e1);
          }
          for (int s = 0; s < samples; ++s) {
            p.grid = slots;
            p.reverse_mode = rev ? 2u : 0u;
            t1.push_back(b_stream_time_ns(p, b, sms, e0, e1));
            p.grid = 4 * slots;
            p.reverse_mode = rev;
            t4.push_back(b_stream_time_ns(p, b, sms, e0, e1));
          }
          std::printf("%s\n      {\"rows_per_wave\": %d, \"occupancy\": %d, \"panels\": %u, "
                      "\"k_steps\": %u, \"footprint_bytes\": %llu, \"footprint_ratio\": %.4f, "
                      "\"reversed\": %u, ",
                      first ? "" : ",", rows_per_wave, occ, panels, k_steps,
                      static_cast<unsigned long long>(footprint),
                      static_cast<double>(footprint) / l2, rev);
          first = false;
          print_samples(rev ? "one_wave_down_ns_samples" : "one_wave_up_ns_samples", t1);
          std::printf(", ");
          print_samples("four_wave_ns_samples", t4);
          std::printf("}");
        }
      }
    }
    std::printf("\n    ], ");
    print_ceiling("ceiling_after", b, sms, e0, e1, samples);
    std::printf("}");
  }
  std::printf("\n  ]\n}\n");
  posthoc_free(b);
  return 0;
}

// ---------------------------------------------------------------------------
// b-stream (H4)
// ---------------------------------------------------------------------------
//
// One wave (grid = sm_count * ctas_per_sm) of B-panel streams from DRAM:
// row-major B of N = grid * BLOCK_N, rows of 128 or 256 bytes, or one
// contiguous row per stage ("linear"). Each panel is k_steps stages long,
// sized so a launch moves about 256 MiB.

static int run_b_stream(int device, int samples, int warmup) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sms = prop.multiProcessorCount;
  const size_t l2 = prop.l2CacheSize;
  const size_t target = 256ULL << 20;
  PosthocBuffers b = posthoc_alloc(target + (64ULL << 20), l2);
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));

  std::printf("{\n  \"mode\": \"b-stream\", \"sm_count\": %d, \"l2_bytes\": %zu, \"samples\": %d,\n"
              "  \"blocks\": [",
              sms, l2, samples);
  bool first_block = true;
  for (int tokens : {1, 2, 3}) {
    std::printf("%s\n    {\"tokens\": %d, ", first_block ? "" : ",", tokens);
    print_ceiling("ceiling_before", b, sms, e0, e1, samples);
    std::printf(", \"points\": [");
    first_block = false;
    bool first = true;
    for (int cps : {1, 2, 3, 4, 6, 9}) {
      for (int vpt : {2, 4, 8, 16}) {
        for (uint32_t row_bytes : {0u, 256u, 128u}) {
          for (uint32_t mma_iters : {1u, 8u}) {
            const uint32_t stage = 128u * 16u * vpt;
            const uint32_t rb = row_bytes ? row_bytes : stage;
            const uint32_t grid = static_cast<uint32_t>(sms * cps);
            const uint64_t row_stride = static_cast<uint64_t>(grid) * rb;
            const uint32_t k_steps = static_cast<uint32_t>(target / (static_cast<uint64_t>(grid) * stage));
            BStreamPoint p{tokens, vpt, cps, grid, grid, rb, row_stride, k_steps, mma_iters, 0, grid};
            const int occ = b_stream_run(p, b.data, b.sink, true);
            if (occ < cps) continue;
            for (int w = 0; w < warmup; ++w) b_stream_time_ns(p, b, sms, e0, e1);
            std::vector<double> t;
            for (int s = 0; s < samples; ++s) t.push_back(b_stream_time_ns(p, b, sms, e0, e1));
            std::printf("%s\n      {\"ctas_per_sm\": %d, \"occupancy\": %d, \"stage_bytes\": %u, "
                        "\"row_bytes\": %u, \"pattern\": \"%s\", \"mma_iters_per_stage\": %u, "
                        "\"k_steps\": %u, \"bytes\": %llu, ",
                        first ? "" : ",", cps, occ, stage, rb, row_bytes ? "rows" : "linear",
                        mma_iters, k_steps,
                        static_cast<unsigned long long>(static_cast<uint64_t>(grid) * stage * k_steps));
            first = false;
            print_samples("time_ns_samples", t);
            std::printf("}");
          }
        }
      }
    }
    std::printf("\n    ], ");
    print_ceiling("ceiling_after", b, sms, e0, e1, samples);
    std::printf("}");
  }
  std::printf("\n  ]\n}\n");
  posthoc_free(b);
  return 0;
}

// ---------------------------------------------------------------------------
// staged-mma (H3 negative control)
// ---------------------------------------------------------------------------
//
// 8 warps, TOKENS stage buffers of VPT * 4 KiB, 32 KiB of ldmatrix and
// mma_iters x 4 m16n8k16 MMAs per warp per stage. Every stage of a CTA reads
// the same L2-resident bytes, so the whole grid touches at most 3 MiB.

template <int TOKENS, int VPT>
__global__ void __launch_bounds__(256) staged_mma_kernel(const uint4* __restrict__ src,
                                                         float* __restrict__ sink,
                                                         int steps,
                                                         int mma_iters) {
  constexpr size_t kStageVec4 = 256u * VPT;
  extern __shared__ uint4 smem[];
  const uint32_t tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
  const size_t base = static_cast<size_t>(blockIdx.x) * kStageVec4;
  uint32_t a0 = 0x3c003c00u + (tid & 7u), a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u;
  uint32_t b0 = 0x3c003c00u, b1 = 0x3c003c00u, live = 0u;
  float c[4][4] = {};

  auto issue = [&](int slot) {
    uint4* slot_ptr = smem + static_cast<size_t>(slot) * kStageVec4;
    #pragma unroll
    for (int v = 0; v < VPT; ++v) {
      const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(slot_ptr + v * 256 + tid));
      const void* g = src + base + static_cast<size_t>(v) * 256u + tid;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(g) : "memory");
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  };

  int issued = 0;
  #pragma unroll
  for (int s = 0; s < TOKENS; ++s) {
    if (issued < steps) issue(issued % TOKENS), ++issued;
  }
  for (int step = 0; step < steps; ++step) {
    if constexpr (TOKENS == 1) asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    else if constexpr (TOKENS == 2) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
    else asm volatile("cp.async.wait_group 2;\n" ::: "memory");
    asm volatile("bar.sync 0;\n" ::: "memory");
    uint4* cur = smem + static_cast<size_t>(step % TOKENS) * kStageVec4;
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
      const size_t off = (static_cast<size_t>(p) * 8u + warp) * 32u % kStageVec4 + lane;
      const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(cur + off));
      uint32_t r0, r1, r2, r3;
      asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s) : "memory");
      live ^= r0 ^ r1 ^ r2 ^ r3;
      if (p == 6) b0 = r0, b1 = r1;
      if (p == 7) a0 = r0, a1 = r1, a2 = r2, a3 = r3;
    }
    for (int m = 0; m < mma_iters; ++m) {
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      }
    }
    asm volatile("bar.sync 0;\n" ::: "memory");
    if (issued < steps) issue(step % TOKENS), ++issued;
  }
  asm volatile("cp.async.wait_all;\n" ::: "memory");
  float sum = static_cast<float>(live);
  #pragma unroll
  for (int j = 0; j < 4; ++j) sum += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  if (sum == -999999.0f) sink[blockIdx.x * blockDim.x + tid] = sum;
}

template <int TOKENS, int VPT>
static int staged_mma_launch(int grid, size_t smem, const uint4* src, float* sink, int steps,
                             int mma_iters, bool query_only) {
  auto kernel = staged_mma_kernel<TOKENS, VPT>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(smem)));
  if (query_only) {
    int blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, 256, smem));
    return blocks;
  }
  kernel<<<grid, 256, smem>>>(src, sink, steps, mma_iters);
  return 0;
}

static int staged_mma_run(int tokens, int vpt, int grid, size_t smem, const uint4* src,
                          float* sink, int steps, int mma_iters, bool query_only) {
  if (tokens == 1 && vpt == 4) return staged_mma_launch<1, 4>(grid, smem, src, sink, steps, mma_iters, query_only);
  if (tokens == 2 && vpt == 4) return staged_mma_launch<2, 4>(grid, smem, src, sink, steps, mma_iters, query_only);
  if (tokens == 3 && vpt == 4) return staged_mma_launch<3, 4>(grid, smem, src, sink, steps, mma_iters, query_only);
  if (tokens == 1 && vpt == 8) return staged_mma_launch<1, 8>(grid, smem, src, sink, steps, mma_iters, query_only);
  if (tokens == 2 && vpt == 8) return staged_mma_launch<2, 8>(grid, smem, src, sink, steps, mma_iters, query_only);
  return staged_mma_launch<3, 8>(grid, smem, src, sink, steps, mma_iters, query_only);
}

static int run_staged_mma(int device, int samples, int warmup) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sms = prop.multiProcessorCount;
  PosthocBuffers b = posthoc_alloc(256ULL << 20, prop.l2CacheSize);
  float* sink = nullptr;
  CUDA_CHECK(cudaMalloc(&sink, static_cast<size_t>(sms) * 4 * 256 * sizeof(float)));
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));

  // Solo MMA time per iteration of 4 MMAs: 1 CTA per SM, no loads.
  mma_m16n8k16_kernel<false><<<sms, 256>>>(sink, 20000);
  CUDA_CHECK(cudaEventRecord(e0));
  mma_m16n8k16_kernel<false><<<sms, 256>>>(sink, 20000);
  CUDA_CHECK(cudaEventRecord(e1));
  const double ns_per_iter = posthoc_time_ms(e0, e1) * 1e6 / 20000.0;

  int per_sm = 0, reserved = 0, per_block = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  const uint4* src = reinterpret_cast<const uint4*>(b.data);

  std::printf("{\n  \"mode\": \"staged-mma\", \"sm_count\": %d, \"samples\": %d, "
              "\"ns_per_mma_iter_8w\": %.3f,\n  \"points\": [",
              sms, samples, ns_per_iter);
  bool first = true;
  for (int cps : {1, 2}) {
    for (int tokens : {1, 2, 3}) {
      for (int vpt : {4, 8}) {
        for (int mma_iters : {1, 8, 16, 32}) {
          size_t smem = static_cast<size_t>(per_sm / cps - reserved) / 128u * 128u;
          smem = std::min<size_t>(smem, static_cast<size_t>(per_block));
          if (smem < static_cast<size_t>(tokens) * 4096u * vpt) continue;
          const int grid = sms * cps;
          const int occ = staged_mma_run(tokens, vpt, grid, smem, src, sink, 1, 0, true);
          if (occ < cps) continue;
          const int steps = 1024;
          auto once = [&]() {
            CUDA_CHECK(cudaEventRecord(e0));
            staged_mma_run(tokens, vpt, grid, smem, src, sink, steps, mma_iters, false);
            CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaGetLastError());
            return posthoc_time_ms(e0, e1) * 1e6 / steps;
          };
          for (int w = 0; w < std::max(warmup, 1); ++w) once();
          std::vector<double> t;
          for (int s = 0; s < samples; ++s) t.push_back(once());
          std::printf("%s\n    {\"ctas_per_sm\": %d, \"occupancy\": %d, \"tokens\": %d, "
                      "\"stage_bytes\": %d, \"ldmatrix_bytes_per_stage\": 32768, "
                      "\"mma_iters_per_stage\": %d, \"solo_compute_ns\": %.1f, \"steps\": %d, ",
                      first ? "" : ",", cps, occ, tokens, 4096 * vpt, mma_iters,
                      mma_iters * ns_per_iter, steps);
          first = false;
          print_samples("stage_turnaround_ns_samples", t);
          std::printf("}");
        }
      }
    }
  }
  std::printf("\n  ]\n}\n");
  CUDA_CHECK(cudaFree(sink));
  posthoc_free(b);
  return 0;
}
