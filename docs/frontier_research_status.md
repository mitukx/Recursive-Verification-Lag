# Frontier-lab research evidence: October 7, 2026

This is a research-readiness assessment, not a prediction of employment.
The user's target is xAI and comparable frontier research groups.

The [current xAI Post-Training and RL role](https://job-boards.greenhouse.io/xai/jobs/5114737007)
emphasizes reward modeling, preference optimization, reasoning and real-world
capabilities. This repository is relevant to that scope. An original, falsifiable
claim and independent measurements are still needed to make it a research
centerpiece. The [systems evidence matrix](https://github.com/mitukx/Recursive-Verification-Lag/blob/main/docs/xai_evidence_matrix.md) on the main
branch separately tracks implementation and real GPU measurement gaps.

## What the latest work established

1. Completed development results are preserved with full data and lineage.
   The locked fresh-task refresh primary is negative and the public-test control
   is stronger. The analysis retains zero-success tasks and cost distinctions.
2. A subsequent exploratory diagnostic removes a secondary apparent benefit
   by matching the exponential-policy path. Public-only fitting alters a
   positive score slope; it does not improve candidate ranking. Richer cheap
   features do not beat the free public control in these comparisons.
3. The theory explains exactly what this diagnostic can identify and why
   unpaid-task, distribution-free safety requires additional information or
   valid structural assumptions.
4. A real-model T4 learning pilot is independently locked and launched.
   Its outcome is pending until complete raw evidence is retrieved.

## Prior-art implications

The 2026 primary sources below reinforce why generic verifier exploitation,
stronger verifiers and auditing alone are insufficient novelty claims:

- [Mahmoud et al., Reward Hacking in Rubric-Based Reinforcement Learning](https://arxiv.org/abs/2605.12474)
  separates verifier failure from rubric limitations and compares training
  feedback to independent judging. Its domains and evaluator design differ
  from executable-code source auditing here.
- [Helff et al., LLMs Gaming Verifiers](https://arxiv.org/abs/2604.15149)
  studies logical rule induction and isomorphic verification. It is not a
  theorem about adaptive paid-label cost or refresh timing.
- The earlier [timing-identification note](refresh_timing_identification.md)
  documents reward-model overoptimization, HackProbe and adversarial reward
  auditing overlap. The present affine and box-bound algebra is elementary.

This is a focused primary-source comparison, not an exhaustive novelty review.
Do not claim the field has not studied verifier failure, or count the above
algebra as a major new theorem.

## Evidence gates for the next research claims

| Proposed claim | Evidence required | Current decision |
|---|---|---|
| Refresh improves new-task verification | Independent task/sample split, strong public baseline, matched optimization strength, paid-label curve | Original development primary negative; scalar secondary explained |
| An observable controller predicts stale-verifier harm | Development-only fitting, pre-update features, five-task heldout identification gate, report false interventions and missed harm | Five development onset tasks; heldout unopened |
| Learned capability improves | Actual parameter updates, independent terminal evaluation, no-update and shuffled-label controls, multiple seeds, raw predictions | T4 pilot launched; pending |
| Practical verification efficiency | All physical scoring, distinct labels, wall time, compute and rejected proposals reported together | Development costs archived; full tradeoff not established |
| Scalable RL systems work | Real GPU serving, training, synchronization, profiler and multi-GPU throughput | Main has harnesses; queued self-hosted jobs are not evidence |

A negative GPU learning pilot should narrow the project, not be replaced by a
success-only demo. A small positive pilot would justify a larger separately
locked replication; it would not alone establish frontier-lab readiness.
