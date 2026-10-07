# Corrected RLVR pilot v2

This is the second small model-training iteration after the retained negative
[v1 pilot and numerical diagnosis](free_colab_pilot_evidence.md). The original
runner, source lock and raw evidence remain unchanged. Execution outcomes are
pending until a complete retained artifact is available.

## Pre-execution decisions

The [v2 lock](../configs/colab_rlvr_pilot_v2.json) pins corrected systems source
`c7e646b043cb56e5ea3c2623bb8a61e065451f72`, the same Qwen2.5-0.5B and GSM8K
revisions, FP32, learning rate 5e-7, seeds 17/29/43, two reward arms and four
updates per condition. Training uses the same first 16 official train rows.
The 32 official test rows 256..287 are selected before their outcomes are seen,
outside the first pilot and the retained alignment/learned-verifier bridge
protocols. These are held out from this optimization, not guaranteed absent
from model pretraining. No evaluation-driven learning-rate or task selection,
promotion, success stopping or replacement of failed conditions is allowed.

The new neutral generation controls can also change the baseline relative to
v1. The changed evaluation split means v1/v2 accuracy is not a matched measure
of the software fix. Report v2 terminal changes relative to its own baseline.

## Gate every update

Before each optimizer step, retain all actions, behavior scores and independently
rescored learner token log probabilities. Require maximum absolute log-ratio
at most **0.002** and clipping fraction exactly **zero**. Failure retains the
sample and stops before that optimizer step. The tolerance is fixed before
corrected 256-token learning; it accommodates the previously observed 0.00110242
historical long-sequence replay residual while bounding ratio mismatch far below
0.2 clipping. It is explicitly different from the 1e-4 gate of the separate
64-token diagnostic. Do not relax it after seeing v2 scores or accuracy.

An event ledger associates each successful optimizer step with its preceding
fresh-policy gate. Offline aggregation recomputes token-score differences,
reward labels, supplied shuffled labels and correctness from retained responses.
It checks the update ordering rather than trusting a success flag. Real tiny
model tests confirm that mismatched behavior scores fail without changing model
parameters or populating optimizer state. Synthetic fixtures test integrity
contracts only; they are never published as GPU measurements.

## Parameter and systems evidence

Hash every named parameter, shape, dtype and byte before and after each condition.
All conditions must start with the same whole-model hash as the baseline. A hash
change documents parameter motion, not learning benefit. This replaces v1's
limited first-parameter probe for this new experiment.

Retain wall time for generation, fresh scoring, reward computation, learning,
terminal evaluation and parameter hashing, plus allocated GPU memory. These are
single-GPU stage measurements; no paired speedup, kernel throughput, utilization,
serving recovery or scaling result follows from them.

## Run and audit

Use an interactive free CUDA runtime and the exact locked dependency versions.
Do not install a serving/distributed stack or purchase compute for this pilot.

```bash
python scripts/run_colab_rlvr_pilot_v2.py --output /tmp/rvl-v2-new-attempt
python -m scripts.summarize_colab_rlvr_pilot_v2 /tmp/rvl-v2-new-attempt \
  --output /tmp/rvl-v2-scorecard.json
```

The output directory must be new. Partial results and errors are retained, and
an incomplete run yields no capability summary. The source protocol and runner
bytes are saved alongside manifests to avoid confusing original source hashes
with reformatted JSON copies. Artifact consistency is not remote GPU attestation.

Report every seed/arm, paired terminal-minus-baseline outcomes and the
trusted-minus-shuffled contrast. Task-bootstrap intervals are descriptive and
conditional on three retained seeds; they do not measure seed uncertainty.

## What this iteration can establish

A completed run can establish whether the corrected sampler/learner stays within
its declared numerical bounds through 24 real optimizer steps and what happened
to capability on a separate small math evaluation split. It cannot establish
held-out software-engineering agent learning, long-horizon success, preference
optimization superiority, production serving or multi-GPU scaling. Those require
the shared [research campaign](research_execution_plan.md) and real workloads.
