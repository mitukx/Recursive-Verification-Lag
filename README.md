# Recursive Verification Lag

## Frontier RL / AI4AI research

The repository is now organized around one north-star problem: **train long-horizon
agents to improve real AI engineering work, while keeping optimization grounded
by executable/trusted evaluation and reliable RL infrastructure.**

Six evidence tracks share the same platform: **RSI/research automation for
MLSys/MLE, long-horizon RL, coding RL, RL infrastructure, end-to-end model
building, and post-training/RL**. Their current evidence gaps and fail-closed
claim requirements are locked in
[`configs/frontier_tracks_v1.json`](configs/frontier_tracks_v1.json).
See [the research architecture](docs/frontier_rl_architecture.md).

The [research execution plan](docs/research_execution_plan.md) defines the shared experimental
campaign and the evidence required for each research claim.
The first [executable MLSys development task](docs/mlsys_research_automation.md)
extracts the exact GRPO operator used by the causal-LM trainer: evaluator-owned
forward/gradient parity, Docker candidate isolation, paired external timing and
correctness-gated performance reward. It connects to the existing durable coding
agent. One public operator task does not establish held-out agent capability,
long-horizon learning or GPU performance.

The next headline results should be real measurements, not additional feature
count: (1) held-out AI-engineering/coding improvement, (2) successful
long-horizon trajectories with recovery and an RL gain, and (3) measured
GPU/serving performance or upstream systems impact.

**Measured free-T4 evidence (2026-10-08):** [six-condition RL pilot and numerical diagnosis](docs/free_colab_pilot_evidence.md).
The locked Qwen/GSM8K pilot completed 24 updates and 96 rollouts. Both reward
arms fell from 31.25% baseline accuracy to 18.75% mean terminal accuracy; no
capability gain was observed. A sampler/learner probability mismatch was found
and repaired. Six short corrected T4 samples passed the declared 1e-4 log-ratio
gate (maximum 7.46e-5), but historical-score reconstruction failed its separate
gate; the diagnostic is retained as an overall failure. Raw evidence and a
CI-recomputed scorecard include both failed diagnostic attempts.

## Post-training capability track

The repository now treats **real post-training capability improvement** as the primary acceptance target. The central empirical question is whether verified RL can improve an independent held-out capability metric, and whether a learned reward/verifier remains useful as policy optimization shifts the model distribution.

The current campaign combines two existing real-model paths:

- **Oracle RLVR:** Qwen/GSM8K rollout -> deterministic trusted reward -> token-level GRPO -> held-out before/after evaluation.
- **Learned reward model:** matched-update `oracle / fresh / stale / shuffled` verifier arms with trusted terminal evaluation withheld from optimization.

A machine-readable scorecard combines these results without hiding null, negative, failed, or underpowered runs. See [Post-training capability campaign v1](docs/posttraining_capability_campaign.md) and `configs/posttraining_capability_campaign_v1.json`.

**The retained small T4 pilot does not establish a capability gain. Independently evaluated positive learning evidence remains required.** The distributed serving, replay, verifier deployment, failure recovery, and profiling stack below exists to make those experiments reproducible and auditable.


**Latest evidence (2026-10-07):** [completed development and fresh-task transfer](notes/progress/36_completed_development_and_transfer.md).
The locked 16/16 development transfer has a negative primary refresh effect
(-0.0002595 expected pass probability; descriptive interval crosses zero).
The public-score control is stronger. All 32 development tasks, the scored
candidate bank, paid transcripts and full results are archived. Five tasks
show later-onset failures in the separate timing study; independent heldout
confirmation and real generator improvement remain open.

**Calibration diagnosis:** [equal-KL controls and theory](notes/progress/37_kl_matched_transfer_diagnostic.md)
show that the secondary public-only refresh benefit vanishes after removing
score-scale changes. The public-only fit preserves rankings exactly; the
all-feature fit has 20 strict pairwise reversals across 3/16 tasks. More
importantly, matched-KL policy differences occur on 14/16 tasks, including
12 tasks with no reversal, implicating within-ranking score geometry rather
than rank change alone. This is an explicitly exploratory development analysis.

**Prospective mechanism test:** [equal-KL verifier-error alignment](notes/progress/39_equal_kl_alignment_phase.md)
was locked before execution and run on 512 independent finite-support tasks.
At matched policy KL, rotating only verifier-error alignment produces opposite
true-progress signs while proxy reward still improves. The declared small-KL
phase prediction achieved 1.0 mean-sign accuracy: harmful cells have false-
progress rate 1.0 and benign cells 0.0 at KL 0.005. The boundary remains correct
in mean sign through KL 0.3. This is synthetic mechanism identification, not
heldout LLM evidence.

**Geometry-aware audit controller:** [trusted-label covariance gate](notes/progress/40_geometry_aware_trusted_label_gate.md)
was also locked before execution and run on 1,024 independent tasks. Using only
trusted audit labels plus free proxy scores, the fail-closed gate blocks harmful
alignment and allows benign alignment with 84.1% / 82.1% decisive rates at 16
labels and 97.75% / 97.75% at 32 labels; no wrong-sign decisive decision was
observed at any reported budget. The bound is valid only for explicit i.i.d.
single-policy audits; adaptive priority sampling is not treated as certified.

**When must verification catch up with a self-improving policy?**

This repository contains an independent research project on the statistical limits of recursively reusing imperfect verifiers during policy optimization.

**Evidence checkpoint (2026-09-23):** The first isolated, scored
pretrained-code-model MBPP+ pilot has a largely **negative** timing result:
early versus uniform verification differs by just one baseline failure among
480 matched settings, and safe projection never activates. See the
[scored pilot](notes/progress/33_mbppplus_first_scored_pilot.md) and the
[claim/decision record](docs/oral_research_decision.md) before extrapolating
from the controlled phase diagrams below. Eight benchmark tasks and a frozen
candidate bank cannot establish learned self-improvement or an Oral result.

The central question is:

> How much genuinely fresh trusted verification is needed to sustain recursive policy improvement when the policy adaptively optimizes an imperfect verifier?

The working hypothesis is more nuanced than “verify every round”: fresh verification is needed when optimization creates **statistically new or poorly covered policy comparisons faster than the verifier can generalize or refresh**. The scored standard-program pilot has not confirmed a general failure boundary.

## Executable asynchronous RVL lab

The new [mini-lab runbook](docs/mini_frontier_lab.md) connects persistent replay,
asynchronous actors/learner, real policy gradients, trusted verification,
learned residual critics, adversarial task search and reward re-evaluation.
It includes a dependency-free CPU/Mac reference, a real causal-LM path,
model-driven Python tool agents with external Docker grading, and a torchrun
token-replay learner. Features and GPU evidence boundaries are listed explicitly.

~~~bash
python -m src.run_mini_lab --output artifacts/async-lab --episodes 256 --actors 4
python -m src.benchmark_mini_lab --output artifacts/ablation --seeds 17,29,43 --episodes 256 --learning-rates 0.08,0.2 --cadences 2,8
~~~

Candidate updates are now transactional: the active policy and candidate are compared on the same held-out suite, promotion fails closed on configured regression criteria, rejected learner state is rolled back, and every decision is recorded in a tamper-evident hash chain.

CI runs Linux/macOS replay, recovery, numerical and integration tests; real
Transformers gradients; two-rank DDP RL; executable Docker rewards; and paired
adaptive/fixed/no-refresh comparisons. No 7B/30B capability or GPU scaling result
is claimed. The code supports configurable long-episode budgets; multi-hour
successful trajectories remain an unvalidated acceptance criterion.

## Verified RL systems engineering track

Alongside the RVL research, this repository contains a tested post-training
systems stack for verifiable-reward RL. The engineering path is deliberately
separated into lightweight CI checks, real-model checks, and GPU-scale
experiments that still require external hardware.

| Capability | Evidence |
|---|---|
| Real causal-LM RL path | Hugging Face model rollout -> verifier -> token-level clipped GRPO update in CI, plus held-out Qwen/GSM8K runner with optional per-step transactional promotion and optimizer/RNG rollback |
| Distributed training primitives | real CPU `torchrun` DDP CI plus GPU-ready NCCL/FSDP sharded checkpoint-resume and 1→2 GPU scaling acceptance |
| Rollout serving | bounded async rollout, EWMA + prefill/decode/KV-aware routing, weighted workload admission, fair multi-workload scheduling, overload shedding, end-to-end deadlines, optional hedged requests |
| Failure handling | cross-worker failover, circuit-breaker quarantine, single half-open recovery probes, loser cancellation, stale policy-version rejection |
| Remote workers | actual asyncio TCP RPC server/client with request IDs, ping, errors, and policy-version checks |
| Trainer/worker coordination | immutable weight manifests, SHA-256 integrity checks, monotonically increasing versions, worker acknowledgements, plus real-model NCCL policy-broadcast latency/bandwidth benchmark |
| Numerics | fp32 log-softmax, ratio clipping, non-finite guards, gradient checks, FP16/BF16 parity gates, real-GPU FP32/BF16/FP16 GRPO comparison, and INT8 weight-only numerical contracts |
| Kernel optimization | optional fused Triton tokenwise GRPO surrogate with custom backward, PyTorch forward/gradient parity benchmark, GPU speed harness |
| Observability | counters, tokens/s, p50/p95/max latency, Chrome traces, deterministic control-plane replay, machine-readable benchmark reports, and named real-GPU GRPO stage profiling |
| GPU serving integration | OpenAI-compatible vLLM/SGLang adapter, streaming TTFT/TBT benchmark, Prometheus metrics capture, GPU telemetry, real process-failure harness, and one-shot two-GPU evidence workflow |

Start with [the systems architecture](docs/rl_system_architecture.md), the
[engineering runbook](docs/rl_systems.md), the
[SLO-aware scheduler design](docs/slo_scheduler.md),
[workload admission design](docs/admission_control.md),
[replay/regression design](docs/replay_and_regression.md),
[free-GPU runbook](docs/free_gpu_runbook.md),
[Triton GRPO kernel note](docs/triton_grpo.md),
[Qwen RLVR experiment](docs/qwen_rlvr_experiment.md),
[FSDP/NCCL acceptance](docs/fsdp_nccl_acceptance.md),
[one-shot GPU evidence bundle](docs/gpu_evidence_bundle.md), and the
[evidence/gap matrix](docs/evidence_matrix.md).

A local control-plane smoke run requires no GPU:

```bash
python -m unittest tests.test_rlvr_systems -v
python -m src.run_rlvr_systems_demo --rounds 12 --samples 32
python -m src.benchmark_rollout_engine --requests 16 --samples 8 --latency-ms 5 --concurrency 8
python -m src.benchmark_scheduler_chaos --requests 32 --recovery-requests 12
python -m src.benchmark_scheduler_slo --trials 12 --primary-ms 40 --backup-ms 2 --hedge-after-ms 5 --deadline-ms 20
```

The repository does **not** yet claim production-scale GPU performance. Real
vLLM/SGLang GPU measurements, held-out Qwen RLVR results, and NCCL/FSDP
multi-GPU validation are explicit open acceptance criteria in issues
[#10](https://github.com/mitukx/Recursive-Verification-Lag/issues/10),
[#11](https://github.com/mitukx/Recursive-Verification-Lag/issues/11), and
[#12](https://github.com/mitukx/Recursive-Verification-Lag/issues/12).

**Evidence checkpoint (2026-09-23):** The first isolated, scored
pretrained-code-model MBPP+ pilot has a largely **negative** timing result:
early versus uniform verification differs by just one baseline failure among
480 matched settings, and safe projection never activates. See the
[scored pilot](notes/progress/33_mbppplus_first_scored_pilot.md) and the
[claim/decision record](docs/oral_research_decision.md) before extrapolating
from the controlled phase diagrams below. Eight benchmark tasks and a frozen
candidate bank cannot establish learned self-improvement or an Oral result.

The central question is:

> How much genuinely fresh trusted verification is needed to sustain recursive policy improvement when the policy adaptively optimizes an imperfect verifier?

The working hypothesis is more nuanced than “verify every round”: fresh verification is needed when optimization creates **statistically new or poorly covered policy comparisons faster than the verifier can generalize or refresh**. The scored standard-program pilot has not confirmed a general failure boundary.

## Main findings

### 1. Adaptive moving-tail verification can be expensive

In a worst-case adaptive family, current-policy auditing pays a factor proportional to candidate amplification. For equal amplification parameter `M` over `T` rounds, the fixed-budget high-probability complexity is

\[
B_{\mathrm{current}}^\star
=\Theta\!\left(TM\log\frac{T}{\delta}\right),
\]

while candidate-aware auditing reduces this to

\[
B_{\mathrm{candidate}}^\star
=\Theta\!\left(T\log\frac{T}{\delta}\right).
\]

This is a worst-case result, not a universal law. With structured verifier/reward classes, raw density-ratio amplification is replaced by a restricted coverage geometry.

### 2. Optimizing and certifying with the same learned verifier can be structurally blind

In a canonical linear plug-in model,

\[
\widehat\Delta
=\eta\|\widehat\theta\|_\Sigma^2\ge 0,
\]

so a naive same-verifier self-certificate has zero rejection power. This result is architecture-specific; it is not claimed for arbitrary verifiers.

### 3. Misspecification creates a stable optimization threshold, but recursive refresh changes the story

Across three controlled settings—non-Gaussian reward, finite program semantics, and a learned autoregressive DSL generator—the location of the semantic sign-reversal threshold is nearly invariant to a large change in trusted-data budget, while the transition width shrinks approximately as `n^(-1/2)`.

However, in a recursive loop, refitting the verifier on the current policy can self-correct. Failure then depends strongly on **verifier staleness**.

### 4. The right recursive unit is a composed policy comparison

If a verifier is frozen for `L` exponential updates,

\[
p_{s+L}(y)
\propto
p_s(y)\exp\!\left(\Lambda_s v_s(y)\right),
\qquad
\Lambda_s=\sum_{k=1}^L\eta_{s,k}.
\]

The entire stale block can therefore be viewed as one composed endpoint comparison `p_s -> q_{s:L}`.

For a structured reward/error class `F`, the block-end trusted-label complexity is

\[
m_s
=O\!\left(
\frac{\mathcal V_{\mathcal F_s}(p_s,q_{s:L};\mu_s)}{\Gamma_s^2}
\log\frac1{\delta_s}
\right),
\]

conditional on the endpoint candidate being fixed before the current certification batch is observed.

This leads to the working interpretation:

> **Verification lag is policy shift relative to verifier-error geometry, not elapsed time.**

### 5. Finite-KL progress is a path integral of verifier/truth alignment

For the exponential path (q_\beta \propto p e^{\beta v}),

\[
\mathbb E_{q_\beta}[y]-\mathbb E_p[y]
=\int_0^\beta \operatorname{Cov}_{q_t}(y,v)\,dt,
\]

while proxy progress is (int_0^\beta \operatorname{Var}_{q_t}(v)dt\ge0)
and endpoint KL is (int_0^\beta t\operatorname{Var}_{q_t}(v)dt).
Thus equal endpoint KL does not determine true progress unless extra assumptions
couple verifier/truth covariance to proxy variance.

## Representative empirical results

| Experiment | Main result |
|---|---|
| Finite non-Gaussian reward | threshold `eta* ~= 3.50`; width `~ n^-0.515` |
| Finite program synthesis | threshold `eta* ~= 24.61`; width `~ n^-0.525` |
| Learned autoregressive DSL generator | threshold `eta* ~= 22.54`; width `~ n^-0.531` |
| Verifier-representation ablation | richer representation moves or removes threshold |
| Best-of-N control | performance peaks around `N=6`, becomes harmful around `N=34` |
| Recursive refresh cadence | fast refresh self-corrects; stale reuse can collapse |
| 2D recursive phase | failure approximately collapses under stale exposure within a fixed score calibration |
| Invariance test | max log-density ratio transfers well across optimizer families, but no shift-only scalar works across verifier classes |
| Prospective equal-KL alignment stress | same KL + same error norm can flip true progress by rotating verifier-error alignment; primary phase accuracy 1.0 |
| Geometry-aware covariance audit | 1,024 tasks; harmful/benign decisive rates 84.1%/82.1% at 16 trusted labels and 97.75%/97.75% at 32; zero observed wrong-sign decisions |

## Key figures

### Recursive verification-lag phase

![Recursive verification-lag phase](figures/recursive_eta_refresh_population_heatmap.png)

### Verifier representation changes the failure boundary

![Verifier representation](figures/learned_generator_representation_eta_star.png)

### Composed-block verification hardness

![Composed verification hardness](figures/composed_block_verification_hardness.png)

## Repository structure

```text
recursive-verification-lag/
├── README.md
├── requirements.txt
├── PUBLISH_TO_GITHUB.md
├── docs/
│   ├── research_overview.md
│   ├── theory.md
│   ├── experiments.md
│   ├── corrections_and_negative_results.md
│   └── roadmap.md
├── notes/
│   ├── master_research_record.md
│   └── progress/
├── src/
│   ├── reproduce_key_figures.py
│   ├── summarize_results.py
│   └── composed_block_demo.py
├── data/
└── figures/
```

## Reproducing the archived plots

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python src/reproduce_key_figures.py
python src/summarize_results.py
```

The archived CSV files are the numerical outputs of the experiments run during this project. The plotting scripts reproduce selected figures from those outputs.

`src/composed_block_demo.py` is a standalone mathematical sanity-check for stale exponential-policy composition and the associated chi-square verification geometry.

## What is *not* claimed

This is a research project in progress. In particular:

- The archived experiments use a small learned GRU/DSL generator, **not a pretrained frontier code LLM**. See the separately labeled pretrained expression pilot below.
- Numerical thresholds such as `eta* ~= 22.54` are environment-specific.
- There is no universal `eta * L`, KL, or density-ratio safety threshold.
- More trusted data do not inherently make a policy worse; in the misspecified self-certification experiments, more data make the fitted wrong model more statistically stable.
- A previously explored conjecture that two structurally distinct audit streams are necessary was **refuted**; a balanced single stream can serve both current certification and future training roles.
- Some supporting results are standard or standard-adjacent (restricted chi-square, reusable holdout machinery, confidence ellipsoids). The intended contribution is their recursive interaction with optimizer-induced coverage shift and verifier reuse.

## Research status

The project currently has a theorem/experiment bridge but is not presented here as a finished peer-reviewed result. The full research ledger records proof status, corrections, falsified conjectures, and open problems in detail:

- [`notes/master_research_record.md`](notes/master_research_record.md)

## Current next step

The central mechanism is now prospective in a controlled environment. The highest-value external-validity step is to move the same geometry into a real-model post-training experiment:

1. use actual sampled LLM rollouts and independent trusted reward,
2. measure or intervene on verifier-error alignment before the update,
3. match realized policy KL across alignment conditions,
4. keep a no-update and shuffled/misaligned-reward control,
5. evaluate only on an untouched terminal split across multiple seeds,
6. retain raw generations, verifier outputs, parameter-update diagnostics, compute and failures.

A separately pinned Qwen/GSM8K matched-drift bridge now implements the real-update contract: paired harmful/benign proxy geometry, calibration-only learning-rate selection, realized token-KL matching, and evaluation labels withheld until both updates are fixed. Its immutable-source self-hosted GPU workflow has been triggered; real GPU outcomes remain a separate evidence gate.

---

**Status:** active independent research, October 2026.


## Pretrained candidate-bank pipeline

The continuation of merged PR #1 adds generation, finite-domain evaluation,
Best-of-N and soft selection, paid-audit budgets, preventive heuristic refresh,
and exploratory held-group analysis. This is a finite-bank selection experiment;
it does not update the code LM weights. See [the runbook](docs/pretrained_pilot_runbook.md),
[conditional observable bounds](docs/observable_refresh_frontier.md), and
[the progress report](notes/progress/23_pretrained_pilot.md) for evidence and limits.


The prospective two-bank transfer study preserves both the favorable 32-draw result
and a reversed six-source safety comparison. All 512 transfer candidates are
imperfect. See [transfer evidence and limitations](notes/progress/26_prospective_transfer.md).

## Verification-aware asynchronous RL

The causal-LM systems path now separates **generation, verification, and learning** into independently scheduled stages. Behavior tokens/logprobs are inserted into durable replay before reward computation; a separate verifier worker leases pending groups, attaches a versioned verifier result, and only then makes them learner-eligible. Learner admission enforces independent policy-lag and verifier-lag bounds. Learned-verifier refreshes requeue stale ready groups for re-verification without regenerating behavior trajectories, and repeated verifier failure is quarantined rather than converted into a training reward.
A bounded **verification-debt controller** now applies rollout backpressure using pending/verifying work, stale verifier rewards, policy/verifier lag, and unverified age. A deterministic sync-vs-naive-async-vs-verification-aware queue benchmark is retained as control-plane evidence before the real-GPU phase diagram in Issue #66.

This is currently contract-level CPU/tiny-model evidence, not a GPU throughput claim. See [the verification-aware async RL design](docs/verification_aware_async_rl.md).

## Bounded recursive self-improvement controller

`src/rsi_controller/` adds an explicitly bounded champion/challenger research loop around the repository's RVL and post-training infrastructure. Each generation is hypothesis-attributed, runs through an allowlisted mutation surface and bounded sandbox, and is evaluated on evolution, development, and independent promotion sets. The sealed suite is not returned to the planner or promotion gate: it is opened once, only after the final promotion decision, for terminal generalization auditing. Higher proxy reward alone cannot promote a candidate.

The CPU reference mode keeps model weights frozen and can be run without a GPU:

```bash
python -m unittest tests.test_rsi_controller -v
python -m src.rsi_controller.run --config configs/rsi/harness_baseline.yaml --generations 4
```

The acceptance workflow runs a bounded CPU Harness-RSI baseline and requires both promotion and rejection behavior while retaining all generations. Earlier implementation-session artifacts that consulted sealed aggregates during iterative selection were removed as superseded; the current contract reserves sealed evaluation for a terminal-only audit. Adapter RSI and the bridge to the existing GRPO/RVL learner remain implemented interfaces, not demonstrated learned-model capability gains. See [the RSI controller design](docs/rsi_controller.md), [engineering report](docs/rsi_engineering_report.md), and [evidence matrix](docs/evidence_matrix.md).

> This is a bounded experimental self-improvement system. It is not evidence of unrestricted or generally recursive intelligence improvement.
