# Systems repository split plan

The research and systems tracks should ultimately be separate public repositories.
The [research execution plan](research_execution_plan.md) keeps one shared
campaign until standalone real-workload evidence exists. The first executable
MLSys operator task is development evidence and does not satisfy the exit
criteria below.

## Recommended split

Keep **Recursive-Verification-Lag** as the paper/research repository:
- theory, assumptions, proofs, negative results;
- controlled experiments and benchmark evidence;
- links to the systems implementation used for large-model validation.

Create a separate systems repository, tentatively **rvl-rl-systems**:
- rollout/inference backends;
- verifier execution;
- GRPO/RLVR training adapters;
- scheduler, checkpoint/recovery, telemetry;
- throughput and failure-recovery benchmarks;
- model-scale experiment recipes.

## Migration boundary

The package under `src/rvl_systems/` is intentionally self-contained. It should be moved with its tests, benchmark scripts, and CI into the future systems repository with minimal path changes. The research repository should then retain only a thin integration adapter plus a link to the systems repository.

## Exit criteria before splitting

Split once the systems package has:
1. remote vLLM/SGLang rollout integration validated on a real model;
2. checkpoint/resume and failure recovery;
3. a PyTorch model-training adapter;
4. reproducible throughput and latency benchmark output;
5. one real RLVR benchmark (e.g. MBPP+/GSM8K) with pinned model/revisions.

At that point the systems repository will stand on its own rather than looking like an extracted scaffold.
