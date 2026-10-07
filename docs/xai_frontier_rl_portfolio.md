# xAI-aligned frontier RL / AI4AI portfolio

This repository should be read as one platform with six evidence tracks, not as
six unrelated demos.

The [MTS execution plan](xai_mts_execution_plan.md) connects these tracks to
verified public role requirements and sequences one shared evidence campaign.
The [MLSys task runbook](mlsys_research_automation.md) provides the first real-code
operator development task and the existing-agent adapter; its development-only
boundary is explicit.

## North-star question

**Can an agent improve real AI engineering work over long horizons, learn from
executable feedback with RL, and remain trustworthy when the policy, reward
model, evaluator and infrastructure evolve at different rates?**

That question unifies Recursive Verification Lag with post-training, coding RL,
long-horizon agents, model-development automation and RL infrastructure.

## Common architecture

```text
real AI-engineering task
        |
        v
long-horizon tool agent ---- durable trajectory / checkpoint / resume
        |
        v
public executable verifier + learned reward model
        |
        +---- trusted hidden evaluation
        |
        v
GRPO / preference optimization / curriculum
        |
        v
candidate model or harness
        |
        v
champion/challenger promotion ---- rollback
        |
        v
model-development task generator / failure analysis
        |
        '----------------------------- recursive next generation
```

The infrastructure layer underneath this loop owns token-exact rollout
provenance, asynchronous actor/verifier/learner scheduling, version fencing,
distributed training, serving, telemetry and failure recovery.

## Track 1 — RSI / research automation for MLSys and MLE

The strongest differentiator should be AI improving the process of building AI,
not generic self-modification. Tasks should be concrete engineering work:
profiling an inference path, improving a kernel or vectorized operator, fixing a
training-performance regression, reducing memory while preserving numerical
parity, repairing distributed-training failure, or optimizing serving latency.

Each task needs a frozen repository snapshot, public diagnostics, hidden
correctness tests, a benchmark protocol, resource limits and an evaluator-owned
reward. Promotion must require correctness before performance. The RSI
controller should propose bounded changes, run experiments, reject regressions
and retain failed hypotheses.

**Acceptance:** a real held-out bank of model-development tasks where the
post-trained/search-improved agent beats its baseline under the same compute
budget, with at least one measured production-style performance improvement.

## Track 2 — Long-horizon RL

A two-hour timeout is not long-horizon evidence. Record real trajectories that
persist through many tool calls, compiles/tests/benchmarks and failures.
Experiments should compare terminal-only return with at least one credit
assignment intervention, and measure success versus trajectory length.

**Acceptance:** completed long tasks, crash/resume evidence, no reward leakage,
and a held-out success improvement after training.

## Track 3 — Coding RL

Replace affine-function tasks as the headline evidence with real repositories.
Use repository-level issue repair, test repair, performance work and
construction tasks. Keep public tests distinct from hidden evaluators and retain
the complete agent trajectory.

**Acceptance:** before/after performance on a frozen held-out coding suite plus
behavioral analysis (repository exploration, testing, self-verification,
recovery from failed edits).

## Track 4 — Grok-style RL infrastructure

The codebase already has substantial control-plane depth. Stop adding isolated
features unless a measurement requires them. The next signal is real hardware:
throughput, tail latency, utilization, policy-weight synchronization, FSDP/NCCL
scaling, failure recovery and verification-aware async RL quality/throughput
tradeoffs.

**Acceptance:** raw GPU evidence and at least one externally useful performance
contribution or reproducible patch against a real RL/serving framework.

## Track 5 — End-to-end model building

Demonstrate model-building judgment rather than only one algorithm. One campaign
should connect curated task data, supervised/preference data where useful,
post-training, evaluation, failure analysis and a second iteration. The final
report must explain why the next data/training intervention was chosen.

**Acceptance:** reproducible data -> training -> evaluation -> diagnosis ->
improved second run, with independent evaluation and raw artifacts.

## Track 6 — Post-Training and RL

Keep the capability-first campaign: oracle RLVR, learned reward model,
fresh/stale/shuffled verifier arms and independent held-out evaluation. Add a
preference-optimization baseline only after the RL path is stable.

**Acceptance:** real capability numbers, reward-model calibration under policy
shift, and a clean comparison of GRPO/RLVR versus preference optimization.

## What not to optimize for

Do not maximize file count, feature count or README claims. Do not call
synthetic queue benchmarks frontier-scale results. Do not use the sealed set
during iterative selection. Do not replace failed GPU runs. Do not add a new
scheduler, verifier or RSI primitive unless it closes a named evidence gap.

The portfolio should eventually have three headline numbers:
1. held-out AI-engineering/coding task improvement;
2. long-horizon success and recovery improvement;
3. measured RL systems throughput/latency improvement.

Everything else should explain those numbers.
