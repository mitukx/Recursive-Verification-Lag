# Executable MLSys research-automation entry point

Issue [#84](https://github.com/mitukx/Recursive-Verification-Lag/issues/84)
requires actual AI-engineering work and correctness-gated performance rewards.
The first development task now uses an **exact source extraction** of the real
tokenwise GRPO surrogate called by this repository's causal-LM trainer. It is an
operator task, not yet a complete repository repair or a held-out task bank.

## Prepare and evaluate

```bash
docker build -f docker/Dockerfile.mlsys -t rvl-mlsys:local .
docker image inspect rvl-mlsys:local --format '{{.Id}}'
python -m src.mlsys_env.evaluator prepare --output artifacts/mlsys-workspace
# Edit only artifacts/mlsys-workspace/solution.py.
python -m src.mlsys_env.evaluator evaluate \
  --candidate artifacts/mlsys-workspace/solution.py \
  --image sha256:REPLACE_WITH_THE_IMAGE_ID \
  --output artifacts/mlsys-evaluation
```

Candidate execution has no network, a read-only filesystem, one CPU, 2 GiB RAM,
bounded frames, bounded retained logs, and a per-request timeout. Only the
candidate source and numeric worker are mounted. The evaluator, oracle, private
spec and expected outputs remain outside the container. Exact source and
function hashes are locked in `tasks/mlsys/grpo-surrogate-v1/task.json`.

For **already reviewed, trusted** code on a Mac without Docker:

```bash
python -m src.mlsys_env.evaluator evaluate \
  --candidate tasks/mlsys/grpo-surrogate-v1/baseline.py \
  --trusted-local --output artifacts/mlsys-local-smoke
```

This explicitly records unisolated execution and grants no performance reward.
Never use that option for model-generated or otherwise untrusted candidates.

## Reward and measurements

The evaluator compares all four outputs and the surrogate gradient against its
hash-locked oracle, including dtype, shape, strided input, clipping boundaries,
positive/negative/zero advantages and invalid-argument behavior. Public
development probes are generated numerical inputs, not logged production data.
No performance credit is given if any correctness check fails.

The evaluator measures elapsed wall time around numeric RPC. Candidate timings
are never accepted. Timings include serialization and transport, exclude import
startup, and are explicitly **not** GPU, kernel-only or end-to-end training
measurements. Baseline/candidate order alternates, both see the same fresh probe,
and all seven raw pairs are retained. A performance reward requires correctness,
Docker isolation and at least 1.05 speedup in every pair. This conservative
development gate is not a statistical generalization certificate.

Reports retain exact sources, source commit, candidate/oracle/evaluation hashes,
CPU budget, machine identity, dependencies, stderr, failure status and raw pairs.
Output directories are never overwritten. The GitHub workflow retains failures
and records the exact Docker image identity.

## Existing agent and RL integration

```python
from pathlib import Path
from src.mlsys_env.agent import MLSysPublicGrader, coding_task
from src.rvl_systems.lab.tool_agent import CodingToolAgent

public = MLSysPublicGrader(image="sha256:YOUR_IMAGE_ID", artifact_root=Path("artifacts/public-grades"))
agent = CodingToolAgent(public, max_steps=128, deadline_s=7200)
task = coding_task()
# Existing episode-pinned serving lease:
# episode = await agent.run(task, lease, "episode-0001", journal="artifacts/episode-0001.json")
```

Public feedback contains correctness reward only. The full report lives in
evaluator storage, outside model history. Terminal evaluation is called by the
trusted outer loop **after** the candidate/episode is fixed. A private terminal
evaluation spec may replace the public probes via `--evaluation-spec`; it is
hashed, never mounted into the candidate container, and not copied to the public
report directory. Reusing private probes for iterative selection would invalidate
their interpretation as terminal evidence. Different numeric seeds on this same
operator do not establish cross-task transfer.

The existing `CodingToolAgent.training_turns` can attach evaluator-owned terminal
return to sampled JSON actions. The added adapter makes that path executable; it
does not claim a trained-agent result or implement a new credit-assignment method.

## Acceptance remaining

1. Freeze multiple task families in actual repository snapshots: inference,
   training numerics/memory, distributed recovery and serving. Extracted operators
   are development exercises; complete issue repair needs repository tools.
2. Split by task and source provenance before optimization, not numeric seed.
3. Run baseline and improved agents at identical tokens, tool calls and compute.
4. Freeze the selected agent, then open independent terminal evaluation once.
5. Publish positive, null and negative task-level results and full trajectories.
6. Validate an accepted change inside real training/serving and on its target
   hardware before claiming end-to-end or GPU performance impact.

Keep #84 and #85 open until these outcomes exist.
