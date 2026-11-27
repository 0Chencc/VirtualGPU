# v0.3 holdout candidate replacement rule

Apply this rule to the four candidate cohorts in `d8eefdb` before
releasing their manifests. Use published workload specifications only.
Add the opened Ascend diagnostic and holdout geometries and the
documented development exclusions to the NVIDIA history. Ascend's
evaluation stays outside the v0.3 freeze. Its historical shapes
participate in the novelty audit.

Keep any candidate whose dimensions clear the updated history. For a
conflicting GEMM dimension, search by increasing absolute distance,
trying the smaller integer first on ties. Preserve the original
remainder modulo the least common multiple of the tile sizes on that
axis. Keep other axes fixed. Treat dimensions shared by the three wave
cases as a unit. If a triplet's varying N conflicts, shift all three N
values by the same offset on the same search grid, keeping each case's
registered reference CTA count. Reject a choice that reuses a historical
axis value or an axis value reserved by the other device. Preserve the
original case mix, wave ordering and candidate configuration IDs.

For a conflicting attention length, use the same nearest-first search
with the original remainder modulo the least common multiple of the
query tiles. Reserve all other lengths on both devices. Keep batch,
head counts, head dimension, causal flag and configuration filter.
Use only positive lengths and keep the configuration IDs identical.
Where multiple repairs are needed, visit the devices and cases in the
original manifest order and the GEMM axes in M, N, K order.

Compile the resulting matrix offline. If the first geometrically valid
replacement fails the reviewed resource, wave or case-mix checks,
stop for review. Neither timing nor prediction results enter selection.
