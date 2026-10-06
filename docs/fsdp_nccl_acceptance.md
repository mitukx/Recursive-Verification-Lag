# FSDP/NCCL acceptance

The repository already contained a CUDA/NCCL FSDP learner entry point. The
acceptance path now treats resumability and measured scaling as first-class
evidence rather than only checking that a sharded optimizer step runs.

## Acceptance sequence

On a machine with two CUDA GPUs:

1. run the FSDP token-level RL update with one rank;
2. run the same update with two ranks;
3. compute throughput speedup and scaling efficiency;
4. save a distributed checkpoint;
5. launch another two-rank job from that checkpoint;
6. require the resumed job to complete and emit `resumed_from_checkpoint=true`.

The FSDP mixed-precision policy selects BF16 when CUDA reports BF16 support and
otherwise uses FP16 parameters/buffers with FP32 reductions. The selected dtype
is written into the distributed report.

The checkpoint path uses PyTorch Distributed Checkpoint and the canonical
state-dict APIs so sharded state remains compatible with FSDP resharding.

## Run

```bash
bash scripts/run_fsdp_scaling_gpu.sh
```

The script expects at least two visible CUDA GPUs and retains raw per-run
reports, GPU telemetry, the scaling summary and checkpoint files. No scaling
number is claimed until this script is executed on real hardware.
