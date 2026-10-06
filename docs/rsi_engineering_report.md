# RSI controller engineering report

Date: 2026-10-06

## Repository architecture analysis
The repository already had durable async replay, versioned policy/checkpoint machinery, token-level GRPO, trusted and ensemble verifiers, RVL audit/refit control, Docker executable grading, vLLM/SGLang adapters, scheduling/telemetry and DDP/FSDP paths. The new work therefore adds an orchestration/governance layer rather than duplicating rollout/training infrastructure.

## Implemented
- typed proposals, candidates, failure clusters, split metrics, RVL assessments, promotion decisions and champion records;
- evidence-based failure analysis;
- allowlisted H/F/V/theta mutation policy and code-patch path gate;
- bounded trusted-entrypoint sandbox with timeout, crash recovery, captured output, resource limits and kill switch;
- evolution/development/promotion/sealed split evaluation;
- champion/challenger promotion gates and explicit reward-hacking/false-progress detection;
- configuration-driven Recursive Verification Lag interventions;
- append-only SQLite research memory and rollback;
- bounded open-ended curriculum;
- Harness/Adapter/RL mode interfaces, with RL reusing the existing AsyncHFLab/GRPO/RVL stack;
- JSONL/CSV metrics and six SVG plots retaining rejected generations;
- CLI, tests, docs, CI workflow and evidence-matrix update.

## Actually executed
The RSI-specific local suite passed **13/13 tests** after two defects found by the first E2E run were corrected: generation-2 planner priority could bypass the intended anti-gaming probe, and timeout pipe cleanup could make recovery unreliable.

A four-candidate CPU Harness-RSI baseline was then run. Results:
- generation 0 baseline: promotion 0.4337, sealed 0.4150;
- generation 1: **PROMOTED**, promotion 0.4412, sealed 0.4225, gap 0;
- generation 2: **REJECTED**, promotion 0.4272, sealed 0.4085, development proxy reward 0.5624 vs trusted 0.4274, verification gap 0.135;
- generation 3: **PROMOTED**, promotion 0.4487, sealed 0.4300, with verifier-refresh intervention;
- generation 4: **PROMOTED**, promotion 0.4562, sealed 0.4375.

SQLite `PRAGMA integrity_check` returned `ok`.

## Tests passed locally
Proposal schema; mutation isolation; evaluator/code-patch protection; Harness-mode F/theta immutability; sealed-suite digest immutability; successful promotion; reward-hacking rejection; deterministic seeds; stale-verifier detection; append-only memory; rollback; timeout and post-timeout recovery; kill switch; curriculum advancement; multi-generation E2E smoke with both promotion and rejection plus plot/audit artifacts.

## Experimental evidence
The new controller is **CPU-demonstrated** for multi-generation Harness RSI mechanics, evaluation separation, negative-candidate retention, anti-gaming rejection, verifier-lag routing, append-only history, rollback and deterministic metrics. The repository separately contains stronger pre-existing evidence for real model gradients, distributed training and executable grading; that does not imply the new controller has demonstrated real-model recursive capability gain.

## Not yet validated
Real-model Harness RSI capability improvement; end-to-end LoRA RSI; multi-generation GRPO/RL RSI with sealed gains; multi-GPU/NCCL/FSDP RSI; multi-hour agent trajectories; cryptographic sealed-eval robustness; calibrated safety guarantees for RVL thresholds; unrestricted or generally recursive intelligence improvement.
