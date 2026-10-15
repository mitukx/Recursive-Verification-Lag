# Documentation index

This directory contains the project’s current explanations, experiment protocols, and systems runbooks. Start with the short route that matches your goal; the [progress archive](../notes/progress/README.md) and [master research record](../notes/master_research_record.md) preserve the longer chronology.

## Recommended reading paths

| Goal | Reading path |
|---|---|
| Understand the research question | [Research overview](research_overview.md) → [theory ledger](theory.md) → [experiment map](experiments.md) → [corrections and negative results](corrections_and_negative_results.md) |
| Judge the strength of the evidence | [Evidence matrix](evidence_matrix.md) → [evidence units and costs](evidence_and_cost_units.md) → [claim and decision record](oral_research_decision.md) |
| Run the CPU reference lab | [Async RL lab guide](mini_frontier_lab.md) → [replay and regression](replay_and_regression.md) |
| Inspect coding and research automation | [Executable operator task](mlsys_research_automation.md) → [task contract](../tasks/mlsys/grpo-surrogate-v1/README.md) |
| Inspect model-training and verification paths | [Async verification design](verification_aware_async_rl.md) → [Qwen RLVR protocol](qwen_rlvr_experiment.md) → [post-training campaign](posttraining_capability_campaign.md) |
| Plan or run hardware experiments | [Free-GPU runbook](free_gpu_runbook.md) → [GPU evidence bundle](gpu_evidence_bundle.md) → [FSDP/NCCL acceptance](fsdp_nccl_acceptance.md) |

## Research question, theory, and evidence

| Document | Use it for |
|---|---|
| [Research overview](research_overview.md) | A compact statement of the problem, mechanism, and scope |
| [Theory ledger](theory.md) | Assumptions and status of proved, conditional, and open results |
| [Experiments](experiments.md) | Controlled simulations and the specific conclusions they support |
| [Corrections and negative results](corrections_and_negative_results.md) | Rejected conjectures, scope corrections, and failed interventions |
| [Evidence matrix](evidence_matrix.md) | Implementation, experiment, and hardware evidence by capability |
| [Evidence units and cost model](evidence_and_cost_units.md) | What each audit budget counts and how independent samples are defined |
| [Observable refresh frontier](observable_refresh_frontier.md) | Conditional bounds, counterexamples, and limits of observable refresh rules |
| [Linear hitting verifier](linear_hitting_verifier.md) | A bounded verifier for a stated linear error class |
| [Research upgrade](frontier_research_upgrade.md) | Bottleneck diagnosis and criteria for a useful next experiment |
| [Research status snapshot (October 7, 2026)](frontier_research_status.md) | Dated status of the controlled studies and planned real-model gates |
| [Research execution plan](research_execution_plan.md) | Work order, resource constraints, acceptance criteria, and open claims |
| [Research roadmap](roadmap.md) | Historical MBPP+ analysis plan and its decision rules |
| [Claim and decision record](oral_research_decision.md) | Supported claims, competing explanations, and stop conditions |
| [Systems extraction criteria](systems_split_plan.md) | Conditions for separating the systems package from the research repo |

Detailed derivations and audits are indexed in [theory notes](../notes/theory/README.md). The [progress archive](../notes/progress/README.md) links each dated result, including superseded plans and negative outcomes.

## Model and task experiments

| Document | Use it for |
|---|---|
| [Free-T4 pilot evidence](free_colab_pilot_evidence.md) | Locked Qwen/GSM8K outcome, sampler/learner mismatch, failed diagnostic gate, and raw evidence |
| [Pilot recovery record](pilot_recovery_record.md) | Execution attempts, recovery decisions, and retained partial results |
| [Pretrained pilot runbook](pretrained_pilot_runbook.md) | Fixed candidate-bank generation and scoring procedure |
| [MBPP+ real-program protocol](mbppplus_real_program_protocol.md) | Isolated execution, frozen tasks, and scoring decisions |
| [MBPP+ recursive-loop lock](mbppplus_recursive_loop_lock.md) | Pre-scoring hypotheses and analysis boundaries |
| [MBPP+ transfer v1](transfer_v1_protocol.md) | First disjoint transfer protocol |
| [MBPP+ transfer v2](transfer_v2_protocol.md) | Revised task split and frozen analysis |
| [MBPP+ pilot claim gate](standard_code_validation_gate.md) | What the first scored standard-program pilot establishes and what it does not |
| [Qwen RLVR experiment](qwen_rlvr_experiment.md) | Held-out task selection, training protocol, and optional transactional promotion |
| [Post-training capability campaign](posttraining_capability_campaign.md) | Oracle and learned-verifier arms and their scorecard |
| [Oral-grade experiment protocol](oral_grade_experiment_protocol.md) | Prospective study design and falsification criteria |
| [OpenAI Math / RVL bridge](openai_math_rvl.md) | Exact trusted-verification task bridge |
| [Mathematical RSI control layers](oai_math_rsi_leap.md) | Bounded controllers motivated by the OpenAI Math benchmark |
| [Real-LLM mathematical RSI](real_llm_mathematical_rsi.md) | Model-backed protocol and evidence boundary |
| [Recovered-Qwen mathematical RSI](recovered_qwen_mathematical_rsi.md) | Replay on retained Qwen evidence |
| [Refresh timing identification](refresh_timing_identification.md) | What equal-stream timing comparisons can identify |

## Agent, evaluator, and promotion systems

| Document | Use it for |
|---|---|
| [Async RL reference lab](mini_frontier_lab.md) | CPU reference, causal-LM mechanics, coding-agent path, replay, and recovery |
| [Research-track architecture](frontier_rl_architecture.md) | How research automation, long episodes, coding tasks, infrastructure, and post-training relate |
| [MLSys operator task](mlsys_research_automation.md) | Correctness-gated GRPO surrogate task and evaluator boundary |
| [Bounded improvement controller](rsi_controller.md) | Hypothesis lifecycle, sandbox, independent evaluation, promotion, and rollback |
| [Controller engineering report](rsi_engineering_report.md) | Implemented controller components and their evidence status |
| [Verification-aware asynchronous RL](verification_aware_async_rl.md) | Durable replay and policy/verifier lag handling |
| [RLVR system overview](rl_systems.md) | Training, rollout, checkpoint, and serving entry points |
| [RL system architecture](rl_system_architecture.md) | Component boundaries and runtime flow |
| [Replay and regression](replay_and_regression.md) | Deterministic events, artifact lineage, and regression gates |
| [Policy deployment recovery](policy_deployment_recovery.md) | Crash-consistent activation and recovery invariants |

## Serving, evaluation, and performance operations

| Document | Use it for |
|---|---|
| [Distributed verifier service](distributed_verifier_service.md) | Versioned verifier RPC and worker behavior |
| [Verifier deployment](verifier_deployment.md) | Prepare, activate, and recover verifier versions |
| [Learned verifier head](learned_verifier_head.md) | Deployable neural verifier component and evidence boundary |
| [Admission control](admission_control.md) | Work units, overload handling, and admission invariants |
| [SLO scheduler](slo_scheduler.md) | Deadline-aware routing, hedging, and recovery |
| [Verifier tail latency](verifier_tail_latency.md) | Completion-time routing and latency evidence |
| [Serving cost model](serving_cost_model.md) | Workload costs and serving measurements |
| [Numerical parity](numerical_parity.md) | Precision and behavior-parity gates |
| [FSDP/NCCL acceptance](fsdp_nccl_acceptance.md) | Required evidence before claiming distributed GPU results |
| [Triton GRPO](triton_grpo.md) | Fused surrogate contract, parity, and benchmark limits |
| [Token-exact vLLM bridge](vllm_token_exact_bridge.md) | Server-owned token IDs and learner likelihood contract |
| [GPU evidence bundle](gpu_evidence_bundle.md) | One-shot run contents, hashes, and acceptance semantics |
| [Free-GPU runbook](free_gpu_runbook.md) | Serving and failure-recovery procedures on an available GPU |
| [Verification-aware GPU execution](verification_aware_gpu_execution.md) | GPU experiment workflow and evidence requirements |
| [Systems split plan](systems_split_plan.md) | Hardware and standalone-workload gates for a future package boundary |

## Reproduction assets

- [Task data notes](../data/README.md) describe the curated numerical outputs.
- [Figure inventory](../figures/README.md) identifies the archived plots.
- [Versioned GRPO surrogate task](../tasks/mlsys/grpo-surrogate-v1/README.md) records its pinned source, public diagnostics, evaluator, and reward limits.
- `configs/` contains the machine-readable protocol locks; matching runners and validators are under `scripts/` and `src/`.
- `results/` retains run-level files. Read the run manifest and the linked protocol before interpreting aggregates.

The index describes navigation, not evidence strength. Each result document is authoritative for its own protocol, raw data, and limitations; newer summaries do not silently replace historical outcomes.
