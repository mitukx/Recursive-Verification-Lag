# RSI controller engineering report

Date: 2026-10-06

## Repository architecture analysis
The repository already had durable async replay, versioned policy/checkpoint machinery, token-level GRPO, trusted and ensemble verifiers, RVL audit/refit control, Docker executable grading, vLLM/SGLang adapters, scheduling/telemetry and DDP/FSDP paths. The new work therefore adds an orchestration/governance layer rather than duplicating rollout/training infrastructure.

## Implemented
- typed proposals, candidates, failure clusters, split metrics, RVL assessments, promotion decisions and champion records;
- evidence-based failure analysis;
- allowlisted H/F/V/theta mutation policy and code-patch path gate;
- bounded trusted-entrypoint sandbox with timeout, crash recovery, captured output, resource limits and kill switch;
- evolution/development/promotion split evaluation plus terminal-only sealed audit;
- champion/challenger promotion gates and explicit reward-hacking/false-progress detection;
- configuration-driven Recursive Verification Lag interventions;
- append-only SQLite research memory and rollback;
- bounded open-ended curriculum;
- Harness/Adapter/RL mode interfaces, with RL reusing the existing AsyncHFLab/GRPO/RVL stack;
- JSONL/CSV metrics and six SVG plots retaining rejected generations;
- CLI, tests, docs, CI workflow and evidence-matrix update.

## Executed evidence policy
The original implementation-session Harness baseline repeatedly exposed sealed aggregate scores during iterative selection. That is weaker evaluation hygiene than the current contract, so its committed machine-readable outputs were removed rather than grandfathered as evidence. The current acceptance workflow regenerates a bounded CPU run under the stronger rule: sealed metrics are unavailable to planner/promotion logic, opened exactly once after the final decision, and the experiment directory becomes terminal afterward.

## Tests passed locally
Proposal schema; mutation isolation; traversal-safe evaluator/code-patch protection; Harness-mode F/theta immutability; sealed-suite digest immutability and terminal-only access; successful promotion; reward-hacking rejection; deterministic paired evaluation; stale-verifier detection; append-only memory; rollback; timeout and post-timeout recovery; kill switch; curriculum advancement; multi-generation E2E smoke with both promotion and rejection plus plot/audit artifacts.

## Experimental evidence
The controller is acceptance-tested for multi-generation Harness RSI mechanics, terminal sealed-evaluation hygiene, negative-candidate retention, anti-gaming rejection, verifier-lag routing, append-only history, rollback and deterministic metrics. The repository separately contains stronger pre-existing evidence for real model gradients, distributed training and executable grading; that does not imply the new controller has demonstrated real-model recursive capability gain.

## Not yet validated
Real-model Harness RSI capability improvement; end-to-end LoRA RSI; multi-generation GRPO/RL RSI with sealed gains; multi-GPU/NCCL/FSDP RSI; multi-hour agent trajectories; cryptographic sealed-eval robustness; calibrated safety guarantees for RVL thresholds; unrestricted or generally recursive intelligence improvement.
