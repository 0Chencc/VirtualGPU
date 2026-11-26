// DRAM read probe for GB10 (docs/results/analysis/v0.3-gb10-dram-probe-plan.md).
//
// dram-probe-sweep: saturation sweep around the candidates, one cold 256 MiB pass each.
// dram-probe:       the fixed candidates P1, P2, P3a, P3b, order rotated every sample.
// mlp-plateau:      the DRAM MLP points at 512 KiB in flight or more, as the workload.
//
// Every timed launch follows a read-only flush outside the timed interval and
// reads each byte of the 256 MiB region exactly once.

constexpr size_t kProbeBytes = 256ULL << 20;

struct ProbeGeometry {
  const char* name;
  bool cp_async;
  int ctas_per_sm;  // cp.async: CTAs per SM (128 threads); ld.v4: CTAs per SM (256 threads)
  int tokens;       // cp.async stage buffers
  int vpt;          // cp.async stage = 2 KiB * vpt
  int unroll;       // ld.v4 loads in flight per thread
};

static long long unix_ms_now() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

// Returns GB/s of one cold pass, or a negative value if the geometry does not fit.
static double probe_once(const ProbeGeometry& g, const PosthocBuffers& b, int sms, cudaEvent_t e0,
                         cudaEvent_t e1, int* occupancy) {
  if (g.cp_async) {
    const uint32_t stage = 128u * 16u * g.vpt;
    const uint32_t grid = static_cast<uint32_t>(sms * g.ctas_per_sm);
    const uint32_t k_steps = static_cast<uint32_t>(kProbeBytes / (static_cast<uint64_t>(grid) * stage));
    BStreamPoint p{g.tokens, g.vpt, g.ctas_per_sm, grid, grid, stage,
                   static_cast<uint64_t>(grid) * stage, k_steps, 0, 0, grid};
    const int occ = b_stream_run(p, b.data, b.sink, true);
    if (occupancy) *occupancy = occ;
    if (occ < g.ctas_per_sm) return -1.0;
    const double bytes = static_cast<double>(grid) * stage * k_steps;
    return bytes / b_stream_time_ns(p, b, sms, e0, e1);
  }
  const int grid = sms * g.ctas_per_sm;
  const size_t mask = kProbeBytes / sizeof(uint4) - 1;
  const int iters = static_cast<int>(kProbeBytes / (static_cast<size_t>(grid) * 256u * 16u * g.unroll));
  int occ = 0;
  auto launch = [&](bool query) {
    const uint4* src = reinterpret_cast<const uint4*>(b.data);
    switch (g.unroll) {
      case 1: if (query) CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, mlp_ld_v4_kernel<1>, 256, 0)); else mlp_ld_v4_kernel<1><<<grid, 256>>>(src, b.sink, mask, iters); break;
      case 2: if (query) CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, mlp_ld_v4_kernel<2>, 256, 0)); else mlp_ld_v4_kernel<2><<<grid, 256>>>(src, b.sink, mask, iters); break;
      case 4: if (query) CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, mlp_ld_v4_kernel<4>, 256, 0)); else mlp_ld_v4_kernel<4><<<grid, 256>>>(src, b.sink, mask, iters); break;
      case 8: if (query) CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, mlp_ld_v4_kernel<8>, 256, 0)); else mlp_ld_v4_kernel<8><<<grid, 256>>>(src, b.sink, mask, iters); break;
      default: if (query) CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, mlp_ld_v4_kernel<16>, 256, 0)); else mlp_ld_v4_kernel<16><<<grid, 256>>>(src, b.sink, mask, iters); break;
    }
  };
  launch(true);
  if (occupancy) *occupancy = occ;
  if (occ < g.ctas_per_sm) return -1.0;
  posthoc_flush(b, sms);
  CUDA_CHECK(cudaEventRecord(e0));
  launch(false);
  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaGetLastError());
  const double bytes = static_cast<double>(grid) * 256.0 * 16.0 * g.unroll * iters;
  return bytes / (posthoc_time_ms(e0, e1) * 1e6);
}

static void print_geometry(const ProbeGeometry& g) {
  std::printf("\"name\": \"%s\", \"issue\": \"%s\", \"ctas_per_sm\": %d, ", g.name,
              g.cp_async ? "cp_async" : "ld_global_v4", g.ctas_per_sm);
  if (g.cp_async) {
    std::printf("\"tokens\": %d, \"stage_bytes\": %d, \"bytes_in_flight_per_cta\": %d, ", g.tokens,
                2048 * g.vpt, 2048 * g.vpt * g.tokens);
  } else {
    std::printf("\"threads_per_cta\": 256, \"loads_in_flight_per_thread\": %d, "
                "\"bytes_in_flight_per_cta\": %d, ",
                g.unroll, 4096 * g.unroll);
  }
}

static const std::vector<ProbeGeometry>& probe_candidates() {
  static const std::vector<ProbeGeometry> c = {
      {"P1", true, 2, 3, 8, 0},
      {"P2", true, 2, 4, 2, 0},
      {"P3a", false, 4, 0, 0, 4},
      {"P3b", false, 4, 0, 0, 8},
  };
  return c;
}

static int run_dram_probe(int device, int samples, int warmup) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sms = prop.multiProcessorCount;
  PosthocBuffers b = posthoc_alloc(kProbeBytes, prop.l2CacheSize);
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  const auto& c = probe_candidates();
  std::vector<std::vector<double>> gbps(c.size());
  std::vector<int> occ(c.size(), 0);
  for (int w = 0; w < warmup; ++w)
    for (size_t i = 0; i < c.size(); ++i) probe_once(c[i], b, sms, e0, e1, &occ[i]);
  const long long start = unix_ms_now();
  for (int s = 0; s < samples; ++s)
    for (size_t j = 0; j < c.size(); ++j) {
      const size_t i = (j + static_cast<size_t>(s)) % c.size();
      gbps[i].push_back(probe_once(c[i], b, sms, e0, e1, &occ[i]));
    }
  std::printf("{\n  \"mode\": \"dram-probe\", \"sm_count\": %d, \"samples\": %d, \"bytes\": %zu, "
              "\"unix_ms_start\": %lld, \"unix_ms_end\": %lld,\n  \"candidates\": [",
              sms, samples, kProbeBytes, start, unix_ms_now());
  for (size_t i = 0; i < c.size(); ++i) {
    std::printf("%s\n    {", i ? "," : "");
    print_geometry(c[i]);
    std::printf("\"occupancy\": %d, ", occ[i]);
    print_samples("gbps_samples", gbps[i]);
    std::printf("}");
  }
  std::printf("\n  ]\n}\n");
  posthoc_free(b);
  return 0;
}

static int run_dram_probe_sweep(int device, int samples, int warmup) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sms = prop.multiProcessorCount;
  PosthocBuffers b = posthoc_alloc(kProbeBytes, prop.l2CacheSize);
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  std::vector<ProbeGeometry> pts;
  for (int cps : {1, 2, 3, 4})
    for (int tokens : {2, 3, 4})
      for (int vpt : {2, 4, 8, 16}) pts.push_back({"cp", true, cps, tokens, vpt, 0});
  for (int cps : {1, 2, 4, 6})
    for (int unroll : {1, 2, 4, 8, 16}) pts.push_back({"ld", false, cps, 0, 0, unroll});
  std::printf("{\n  \"mode\": \"dram-probe-sweep\", \"sm_count\": %d, \"samples\": %d, \"bytes\": %zu,\n"
              "  \"points\": [",
              sms, samples, kProbeBytes);
  bool first = true;
  for (const auto& g : pts) {
    int occ = 0;
    if (probe_once(g, b, sms, e0, e1, &occ) < 0) continue;
    for (int w = 1; w < warmup; ++w) probe_once(g, b, sms, e0, e1, nullptr);
    std::vector<double> v;
    for (int s = 0; s < samples; ++s) v.push_back(probe_once(g, b, sms, e0, e1, nullptr));
    std::printf("%s\n    {", first ? "" : ",");
    first = false;
    print_geometry(g);
    std::printf("\"occupancy\": %d, \"unix_ms\": %lld, ", occ, unix_ms_now());
    print_samples("gbps_samples", v);
    std::printf("}");
  }
  std::printf("\n  ]\n}\n");
  posthoc_free(b);
  return 0;
}

// The DRAM MLP points of run_mlp at 512 KiB or more in flight, both issue modes.
static int run_mlp_plateau(int device, int samples, int warmup) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  const int sms = prop.multiProcessorCount;
  PosthocBuffers b = posthoc_alloc(kProbeBytes, prop.l2CacheSize);
  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  const size_t mask = kProbeBytes / sizeof(uint4) - 1;
  const uint4* buf = reinterpret_cast<const uint4*>(b.data);
  struct Pt { int ctas, mult; };
  const std::vector<Pt> pts = {{std::min(32, sms), 4}, {sms, 4}, {sms * 2, 4}, {sms * 3, 4},
                               {sms * 3, 8}, {sms * 4, 8}, {sms * 5, 8}};
  std::printf("{\n  \"mode\": \"mlp-plateau\", \"sm_count\": %d, \"samples\": %d, "
              "\"unix_ms_start\": %lld,\n  \"points\": [",
              sms, samples, unix_ms_now());
  bool first = true;
  for (const char* mode : {"cp_async", "ld_global_v4"}) {
    const bool cp = std::strcmp(mode, "cp_async") == 0;
    for (const auto& pt : pts) {
      const size_t bif = static_cast<size_t>(pt.ctas) * pt.mult * 4096u;
      const size_t target = std::min(kProbeBytes, std::max<size_t>(16ULL << 20, bif * 128u));
      const int sub = pt.mult <= 4 ? 1 : pt.mult / 4;
      const int stages = pt.mult <= 4 ? pt.mult : 4;
      const int cp_steps = std::max(stages * 4, static_cast<int>(target / (static_cast<size_t>(pt.ctas) * 4096u * sub)));
      const int ld_iters = std::max(4, static_cast<int>(target / (static_cast<size_t>(pt.ctas) * 4096u * pt.mult)));
      const double bytes = cp ? 4096.0 * pt.ctas * sub * cp_steps : 4096.0 * pt.ctas * pt.mult * ld_iters;
      const size_t smem = static_cast<size_t>(stages) * sub * 4096u;
      auto once = [&]() {
        posthoc_flush(b, sms);
        CUDA_CHECK(cudaEventRecord(e0));
        if (cp) {
          if (pt.mult == 4) mlp_cp_async_kernel<4, 1><<<pt.ctas, 256, smem>>>(buf, b.sink, mask, cp_steps);
          else mlp_cp_async_kernel<4, 2><<<pt.ctas, 256, smem>>>(buf, b.sink, mask, cp_steps);
        } else {
          if (pt.mult == 4) mlp_ld_v4_kernel<4><<<pt.ctas, 256>>>(buf, b.sink, mask, ld_iters);
          else mlp_ld_v4_kernel<8><<<pt.ctas, 256>>>(buf, b.sink, mask, ld_iters);
        }
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaGetLastError());
        return bytes / (posthoc_time_ms(e0, e1) * 1e6);
      };
      for (int w = 0; w < warmup; ++w) once();
      std::vector<double> v;
      for (int s = 0; s < samples; ++s) v.push_back(once());
      std::printf("%s\n    {\"issue_mode\": \"%s\", \"bytes_in_flight\": %zu, \"active_ctas\": %d, "
                  "\"bytes_per_cta\": %d, ",
                  first ? "" : ",", mode, bif, pt.ctas, pt.mult * 4096);
      first = false;
      print_samples("gbps_samples", v);
      std::printf("}");
    }
  }
  std::printf("\n  ],\n  \"unix_ms_end\": %lld\n}\n", unix_ms_now());
  posthoc_free(b);
  return 0;
}
