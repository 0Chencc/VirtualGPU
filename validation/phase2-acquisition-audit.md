# GB10 acquisition audit

## Spark 1 compute contention on November 25, 2026

The secondary `l2-working-set-below` case exited its second session during
the GPU availability check, before any timed trial. At 12:11:10 UTC,
`nvidia-smi` showed a separate compute process, PID 3205635, using
37,060 MiB of GPU memory. Device utilization was 96%.

At 12:11:31 UTC, a fresh compute-process query under the GPU lock returned
an empty list. The one-minute host load was 1.25. Collection resumed with
the same `fd5c947` harness, case, and session number. The rerun passed
AOT/JIT validation and its sealed report was recorded at
12:13:24.078772 UTC. See the
[completed seal receipt](phase2-gb10-seal.json) for
`secondary/sealed/l2-working-set-below-session-2.json`.
