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
4. The theory now extends beyond the local derivative. Along the exact
   exponential path, true progress is the path integral of Cov_q(y,v), while
   proxy progress integrates Var_q(v) and KL integrates beta*Var_q(v). This
   makes endpoint-KL insufficiency exact at finite update size.
5. A second prospective experiment turns the mechanism into a bounded trusted-
   label controller. Across 1,024 tasks, a Hoeffding gate on Cov_p(y,v) reaches
   harmful/benign decisive rates 0.8408/0.8213 at 16 labels and
   0.9775/0.9775 at 32 labels, with zero observed wrong-sign decisive decisions.
   The statistical contract explicitly rejects adaptive-priority samples as
   i.i.d. evidence.
6. A real-model Qwen/GSM8K matched-drift bridge is locked with paired harmful
   and benign proxy interventions, calibration-only LR selection, exact
   model/optimizer/RNG reset, post-update token-k3 drift measurement, and
   evaluation labels withheld until both updates are fixed. Its immutable-source
   self-hosted GPU workflow has been triggered; the outcome is still a separate
   evidence gate.

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
| Trusted labels can detect harmful local alignment | Pre-registered iid-policy covariance audit with confidence bound and no KL input | Established synthetically; 1,024 tasks, 97.75% harmful/benign decisive rate by 32 labels, zero observed wrong-sign decisions |
| Refresh improves new-task verification | Independent task/sample split, strong public baseline, matched optimization strength, paid-label curve | Original development primary negative; scalar secondary explained |
| Geometry transfers to a neural-policy update | Real sampled rollouts, proxy intervention, matched realized drift, untouched terminal evaluation, multiple seeds | Locked Qwen bridge and immutable GPU workflow exist; result pending |
| Geometry transfers to a learned verifier rather than artificial proxy intervention | Learned reward/verifier model, prospective alignment measurement/intervention, sealed evaluation | Protocol/runner/CI implemented prospectively for Qwen hidden-state neural heads (oracle/fresh/stale/shuffled); GPU result not executed and no scientific claim yet |
| An observable controller predicts stale-verifier harm | Development-only fitting, pre-update features, five-task heldout identification gate, report false interventions and missed harm | Five development onset tasks; heldout unopened |
| Learned capability improves | Actual parameter updates, independent terminal evaluation, no-update and shuffled-label controls, multiple seeds, raw predictions | T4 pilot locked; GPU result pending |
| Practical verification efficiency | All physical scoring, distinct labels, wall time, compute and rejected proposals reported together | Development costs archived; full tradeoff not established |
| Real candidate-bank trusted-label efficiency | Same retained LLM candidates, passive vs propensity/HT vs minimax audit, fixed budgets, terminal harm reported separately from covariance-sign correctness | Protocol/adapter/tests implemented prospectively; real replay blocked on learned-verifier GPU artifact |
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

1. execute the locked Qwen matched-drift bridge and retain the result even if
   negative or underpowered. Its evidence validator now independently recomputes
   token-level drift, evaluation margins, manifest hashes and source lineage;
2. after Issue #38 resolves, execute the already locked learned-neural-verifier
   bridge (oracle/fresh/stale/shuffled, common current-policy snapshot and
   matched realized k3), retaining null/negative/underpowered outcomes unchanged;
3. feed that immutable learned-verifier artifact into the already implemented
   passive vs propensity/HT vs fixed-budget minimax offline audit replay before
   wiring any allow/block/refresh action into the online runtime;
4. execute and retain the real-GPU RL systems evidence (serving, RL update,
   synchronization, profiling and multi-GPU scaling) already scaffolded on main.

Those measured results would be substantially more valuable for hiring evidence than
adding additional orchestration abstractions or README claims.