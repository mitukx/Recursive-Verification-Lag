# Bounded Recursive Self-Improvement Controller

> **This is a bounded experimental self-improvement system. It is not evidence of unrestricted or generally recursive intelligence improvement.**

`src/rsi_controller/` adds an auditable champion/challenger loop on top of the repository's RVL, replay, verifier, training and serving infrastructure. The research question is whether independent capability can improve across recursive generations while the policy, verifier, evaluator and improvement strategy evolve at different rates—and when verifier lag turns apparent progress into reward hacking or overfitting.

## Architecture and lifecycle

The controller separates `theta` (policy/model), `F` (optimization), `V` (verifier/reward stack), and `H` (harness/prompts/tools/memory/curriculum). Each generation executes:

```text
champion -> development evidence -> failure clusters -> hypothesis proposal
         -> allowlisted candidate -> bounded sandbox
         -> development evaluation -> anti-gaming/RVL checks
         -> independent promotion evaluation -> promote/reject
         -> [after the last generation only] sealed terminal audit
```

Every proposal records a concrete hypothesis, target component, deterministic change, expected effects, risks, evaluation plan, rollback plan, seed, dependency metadata and resource limits. Rejected candidates remain in history.

## Security boundary

Candidate agents receive declarative mutation data, not arbitrary shell or Python execution. Harness RSI only mutates allowlisted `H` and selected safe `V` configuration. Code patches are disabled by default and, if explicitly enabled, are restricted to a configured path allowlist. Promotion rules, evaluation implementation, sealed task contents, audit logs, research history, sandbox entrypoint allowlist and resource ceilings live outside the mutation surface.

The reference sandbox uses a trusted-entrypoint allowlist, child-process isolation, wall-clock timeout, CPU/memory/file-output limits where supported, reduced inherited environment, stdout/stderr capture, crash recovery and a kill switch. It rejects network-enabled runs. OS `resource` limits are not a substitute for container/network namespaces; future candidate-controlled code must use the repository's disposable Docker or an equivalent stronger sandbox.

The controller does **not** implement credential acquisition, unrestricted external networking, autonomous deployment, self-propagation, silent evaluator modification, persistence outside configured experiment storage, or mechanisms to weaken its own resource/safety boundaries.

## Evaluation separation

`EvaluationStack` distinguishes four result families and enforces an access boundary rather than treating them as interchangeable:

- **evolution**: repeatedly exposed optimization/evolution tasks;
- **development**: observable diagnostics used during candidate iteration;
- **promotion**: independent gate used to compare champion/challenger;
- **sealed**: hidden tasks held by `SealedEvaluationVault`; normal candidate evaluation returns no sealed metric. The suite is opened once after the final promotion decision, and that terminal result cannot be followed by more generations in the same experiment directory.

Real integrations should replace the synthetic task model with repository benchmark/grader adapters while preserving this split contract. A task-generating model must not unilaterally provide the authoritative answer.

## Failure analysis and research memory

`FailureAnalyzer` clusters reasoning, tool selection, verifier false positives/negatives, reward-hacking candidates, timeouts, context failures, curriculum mismatch, policy-verifier distribution shift and infrastructure failure. Each diagnosis carries measured evidence; model-generated explanations are not treated as ground truth.

`ResearchMemory` is SQLite-backed. Proposals, candidates, evaluations, decisions, champions, lessons and events are append-only and protected by `BEFORE UPDATE/DELETE` triggers. Only the current-champion pointer is mutable. Rollback switches that pointer and appends an audit event rather than rewriting prior results.

## Champion/challenger promotion

A higher proxy reward is insufficient. `PromotionGate` checks development trusted performance, independent promotion-set gain, trusted gain, verifier agreement, failure-rate regression, p95 latency, uncertainty margin, verification gap, reward-hacking indicators and RVL trust. It never consumes sealed metrics. Thresholds are configuration-driven.

False-progress diagnostics expose:

- `apparent_gain`: development reward delta;
- `trusted_gain`: independent promotion trusted-score delta;
- `generalization_gain`: independent promotion trusted-score delta;
- `verification_gap`: largest proxy/trusted or apparent/trusted divergence.

A run is not called self-improvement unless improvement survives an appropriate independent evaluation.

## Recursive Verification Lag control

`RecursiveVerificationLagMonitor` tracks approximations to policy improvement rate, verifier refresh rate, verifier disagreement, trusted-label disagreement, reward/ground-truth divergence, distribution shift and stale verifier age.

- **high trust**: continue normal bounded improvement;
- **uncertain trust**: increase independent verification, refresh the verifier and re-evaluate before promotion;
- **low trust**: freeze promotion pending trusted labels/recalibration/refresh.

Every intervention is appended to research memory. These thresholds are experimental controls, not statistical safety certificates.

## Modes

**Mode 1 — Harness RSI:** frozen weights; bounded changes to prompt/harness state, retry/context/memory/curriculum/sampling and selected verifier composition. This is the CPU-demonstrated reference mode.

**Mode 2 — Adapter RSI:** `AdapterRSIBackend` exposes a PEFT/LoRA configuration only when PEFT is explicitly installed. It does not auto-install dependencies or download a model. End-to-end adapter promotion is not yet validated.

**Mode 3 — RL RSI:** `RLRsiBackend` bridges to the repository's existing `AsyncHFLab`, token-level GRPO, durable token replay, trusted RVL audits, verifier refits and checkpoint publication. The bridge is implemented; multi-generation real-model RSI promotion is not yet demonstrated by this controller.

## Bounded curriculum

The default ladder is simple code completion → bug fixing → tests → multi-step debugging → optimization → multi-file modification → tool-use tasks → agent-harness improvement. Selection targets tasks just beyond reliable competence. Generated tasks require external validation, remain separate from promotion/sealed tasks, and cannot self-certify their own answers.

## Reproducible baseline

```bash
python -m unittest tests.test_rsi_controller -v
python -m src.rsi_controller.run \
  --config configs/rsi/harness_baseline.yaml \
  --generations 4
```

Outputs include `research_memory.sqlite`, JSONL/CSV generation metrics, `summary.json`, and six SVG plots. Adaptive generation telemetry contains development/promotion metrics but no sealed score. `summary.json` contains a single `final_sealed_audit` comparing the baseline and terminal champion after all promotion decisions. Rejected generations are never filtered from the artifacts.

The acceptance workflow exercises both promotion and rejection. Historical implementation-session outputs produced under the older repeatedly-opened sealed contract were removed rather than presented as evidence for the stronger terminal-only protocol.

## Known limitations

The baseline validates controller semantics with a synthetic capability model; it is not evidence of learned-model capability gain. Process limits are weaker than a production container sandbox. Sealed evaluation is an application boundary rather than cryptographic remote attestation. Adapter RSI, multi-generation real-model GRPO RSI, multi-GPU RSI and multi-hour agent trajectories remain explicit validation gaps.
