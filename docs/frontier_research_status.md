# Frontier-lab research evidence: October 7, 2026

This is a research-readiness assessment, not a prediction of employment.
The user's target is xAI and comparable frontier research groups.

The [current xAI Post-Training and RL role](https://job-boards.greenhouse.io/xai/jobs/5114737007)
emphasizes reward modeling, preference optimization, reasoning, truthfulness and
real-world capabilities. The [RL Training Framework role](https://job-boards.greenhouse.io/xai/jobs/5186992007)
also emphasizes end-to-end RL systems, small-scale ablations through production
training, profiling, scalability, observability and RL numerics. This repository
now has a credible bridge across those two surfaces: a falsifiable verifier/RL
research mechanism plus a separate systems implementation track. Real-model
external validity and real GPU evidence remain the main unresolved gates.

## What the latest work established

1. Completed development results are preserved with full data and lineage.
   The locked fresh-task refresh primary is negative and the public-test control
   is stronger. The analysis retains zero-success tasks and cost distinctions.
2. The matched-KL archive audit corrected an earlier prose error: the all-feature
   fit has 20 strict pairwise reversals across three tasks. More importantly,
   12/16 tasks change policy at matched KL without any ranking reversal, so
   within-ranking score geometry matters beyond ranking drift alone.
3. A prospective controlled mechanism test was locked before execution. Across
   512 independent finite-support tasks, rotating verifier-error alignment while
   holding error norm and policy KL fixed flips the sign of true progress while
   proxy progress remains positive. The declared KL=0.005 phase prediction has
   accuracy 1.0; harmful-cell false-progress rate is 1.0 versus 0.0 in benign
   cells. The mean-sign phase boundary remains correct through KL 0.3.
4. The local theory explains the phase: d E_q[y] / d beta at beta=0 equals
   Cov_p(y,v), and the intervention makes its sign equal sign(1+sigma*rho).
   This is a direct mechanism showing why policy KL alone cannot determine safe
   verifier reuse.
5. A real-model T4 learning pilot is independently locked with true versus
   shuffled reward, fixed train/test identities, no-update reproducibility and
   terminal evaluation that never chooses updates. Its GPU outcome is still a
   separate evidence gate.

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
| Verifier-error geometry can cause false progress at fixed policy KL | Pre-registered intervention varying alignment while fixing KL/error norm; complete raw grid | Established in synthetic mechanism test; 512 tasks, primary phase accuracy 1.0 |
| Refresh improves new-task verification | Independent task/sample split, strong public baseline, matched optimization strength, paid-label curve | Original development primary negative; scalar secondary explained |
| Geometry transfers to a learned LLM verifier | Real sampled rollouts, learned/proxy verifier, matched realized KL, untouched terminal evaluation, multiple seeds | Not established; highest-priority external-validity gap |
| An observable controller predicts stale-verifier harm | Development-only fitting, pre-update features, five-task heldout identification gate, report false interventions and missed harm | Five development onset tasks; heldout unopened |
| Learned capability improves | Actual parameter updates, independent terminal evaluation, no-update and shuffled-label controls, multiple seeds, raw predictions | T4 pilot locked; GPU result pending |
| Practical verification efficiency | All physical scoring, distinct labels, wall time, compute and rejected proposals reported together | Development costs archived; full tradeoff not established |
| Scalable RL systems work | Real GPU serving, training, synchronization, profiler and multi-GPU throughput | Main has harnesses; real hardware measurements remain the gate |

The transactional systems runner repeatedly evaluates its promotion set to
choose accepted updates. That set is development feedback for model selection,
even if disjoint from gradient-training examples; its terminal score cannot be
presented as an untouched confirmatory result. The new pilot measures terminal
accuracy without using evaluation to select updates. A future deployment study
must keep promotion and sealed final evaluation tasks separate.

A negative GPU learning pilot should narrow the project, not be replaced by a
success-only demo. A small positive pilot would justify a larger separately
locked replication; it would not alone establish frontier-lab readiness.

## Current assessment

The repository is now meaningfully stronger than a broad "mini frontier lab"
demo because it contains a falsifiable mechanism, a pre-outcome protocol, a
completed intervention, negative results that were preserved, and systems code
that maps to current frontier-RL work. It should still not be described as
"xAI-level" in the sense of proven production-scale model training. The two
highest-value missing pieces are:

1. reproduce the verifier-error-geometry effect with an actual learned/proxy
   verifier on real LLM rollouts while matching realized policy KL;
2. execute and retain the real-GPU RL systems evidence (serving, RL update,
   synchronization, profiling and multi-GPU scaling) already scaffolded on main.

Those two results would be substantially more valuable for hiring evidence than
adding additional orchestration abstractions or README claims.
