# Replay, artifact lineage, and regression gates

The systems CI treats benchmark evidence as a reproducible artifact rather than
a screenshot or a manually copied headline number.

## Deterministic control-plane event log

`ControlPlaneEventLog` records semantic scheduler decisions with a contiguous
sequence number and no wall-clock fields, so identical semantic traces have the
same SHA-256 digest. The replay validator checks request lifecycle integrity,
terminal states, and the invariant that a single request is never routed to the
same worker twice.

The chaos benchmark emits a JSONL event stream and CI replays it into a compact
summary. This makes failure-routing behavior inspectable after the run.

## Regression gate

`configs/benchmark_gates.json` defines machine-readable acceptance rules.
Rules may be absolute (`ge`, `le`) or relative to another metric in the same
report (`ge_metric`, `le_metric`). CI currently gates deterministic
properties of failure recovery, SLO hedging, and workload admission.

Timing gates are intentionally relative inside a synthetic benchmark rather
than tied to a single GitHub runner's absolute speed. GPU deployments should
add hardware-scoped baselines for TTFT/TBT, tokens/s, memory, and utilization.

## Artifact lineage

Every retained benchmark/replay artifact is hashed into a manifest containing
its path, byte size, and SHA-256 digest. CI immediately verifies the manifest.
The manifest can therefore travel with benchmark outputs and detect accidental
or post-run mutation.

These mechanisms do not substitute for GPU-scale measurements. They make those
measurements auditable, regression-gated, and reproducible once collected.
