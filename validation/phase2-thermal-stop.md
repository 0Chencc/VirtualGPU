# GB10 transfer collection stopped at the thermal gate

The secondary `prefill-1024` case, session 1, reached the published
20-minute timeout at 2026-11-25T11:20:52.896969+00:00. It never met the
preregistered temperature range of at most 2 C over the preceding 30
seconds after at least 60 seconds of untimed warm-up. The timeout occurred
before timed graph construction. This attempt produced no timing labels.

The 1 Hz temperature trace contains 1,201 samples. The range
across the attempt was 67 to 82 C.
The narrowest window after the 60-second minimum was 3 C, ending at
elapsed second 397. The final sampled window spanned 75 to
80 C. These windows use 30 consecutive 1 Hz samples for
reporting. The harness applies the gate to its original telemetry stream.

The timeout trace was copied from Spark 1 to Bifrost. Its SHA-256 is
`e1588d488542b58ece843bbc970665ac479a3f01425097988d06e1294c50cfe4`. The two earlier `prefill-1024` timeouts
are recorded in [the timeout amendment](phase2-thermal-audit.json). The
in-flight `prefill-2048` session 1 also reached the original five-minute
limit before timing, after further dispatch was paused.

## Partial seal receipts

Ten of 57 planned case-sessions completed AOT/JIT checks and were sealed
with mode 000. Collection stopped with 47 case-sessions pending. The label
reports remain on Spark 1. This partial receipt records the files without
opening them. Formal scoring is pending.

| Cohort | Report | SHA-256 | Measured UTC recorded |
| --- | --- | --- | --- |
| primary | `decode-1-session-1.json` | `c4a65f1dad43665e467477ccc972cbb9ffe5aa05c8e176f3409f16eb4e8a4dcb` | 2026-11-25T10:34:10.079768+00:00 |
| primary | `decode-16-session-1.json` | `7eaa796f26f58b2f23b10bab83464c171095e727ee9ce7d89feef670f8885bbb` | 2026-11-25T10:35:18.279671+00:00 |
| primary | `decode-32-session-1.json` | `c9cc274c5b3f82bc2b2d534b0e06302d472bacefad90e8d7ab5cf6693322efc6` | 2026-11-25T10:36:26.707050+00:00 |
| primary | `m1-cube-session-1.json` | `8e734eca10a01f71059477cad50277a996854d263a19060f85151e375cdb3830` | 2026-11-25T10:32:57.406697+00:00 |
| primary | `mask-m-session-1.json` | `d9edec88850f259eb909d4071abbee0f3fdc8f63b648f53c5550844a61144871` | 2026-11-25T10:37:43.959917+00:00 |
| primary | `sm-round-above-session-1.json` | `80d84d4a2b2a90ced57df2e083f7e3b1cdcab1257173d1ec6c002175eb7c8171` | 2026-11-25T10:42:10.482091+00:00 |
| primary | `sm-round-below-session-1.json` | `ad7b90da97162508f04d1c2da02dfc6539f67b8c3dce925633ea98e1dcfd6240` | 2026-11-25T10:39:54.494846+00:00 |
| primary | `sm-round-exact-session-1.json` | `661d3fcb71f49b4b44608cc4776a27cb84fd70fa6ee72721d52f9a24a9fbf2cf` | 2026-11-25T10:41:02.452215+00:00 |
| primary | `swizzle-320-session-1.json` | `6c804506378c7019d2e304c60d94a937dd69c48960b097ca69cd67e559616f47` | 2026-11-25T10:38:46.504613+00:00 |
| secondary | `prefill-256-session-1.json` | `97960e27b3c72906147b209637e2e84873dba5275851fe5ae70de010721b52f0` | 2026-11-25T10:43:22.631780+00:00 |
