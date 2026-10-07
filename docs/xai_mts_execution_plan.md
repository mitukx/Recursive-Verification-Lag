# xAI MTS evidence plan

Reviewed against official public roles on 2026-10-08. This is an independent
project and a plan for evidence; it does not establish employment, affiliation,
frontier-model capability, or involvement in building Grok.

## One centerpiece

**Train an agent to improve AI engineering work, verify that the improvement is
real, and explain how the RL and serving system made it affordable/reliable.**

The distinguishing research question is whether verifier freshness and trusted
audit allocation preserve actual progress during recursive optimization. The
operator development task is the first executable workload entry, while existing
Qwen RLVR and learned-verifier experiments supply the model-training path.

The MTS story should have three independently supported outcomes: held-out
engineering-task improvement, successful long trajectories with recovery, and
measured RL systems performance. Do not multiply six tracks into six unfinished
flagship projects.

## Public role requirements and project targets

The user's RSI, research-automation, long-horizon, coding-RL and Grok-infra names
are used as capability targets. Those exact names were not confirmed as current
job titles on the reviewed public board. The closest verified public roles are:

| Public role | Requirement relevant to this project | Evidence to produce |
|---|---|---|
| [Post-Training and RL](https://job-boards.greenhouse.io/xai/jobs/5114737007) | reward modeling, RLHF/DPO, RL improving real capabilities | pinned baseline vs RLVR vs learned-verifier arms; preference baseline on the same split; independent capability measurements |
| [RL Training Framework](https://job-boards.greenhouse.io/xai/jobs/5186992007) | end-to-end performance profiling, debugging, scalability, observability, RL numerics | real training trace, bottleneck diagnosis, paired optimization, checkpoint/recovery and 1-to-2 GPU experiment |
| [RL Inference](https://job-boards.greenhouse.io/xai/jobs/5180223007) | distributed systems, LLM inference, low precision, quantization, vLLM/SGLang | real serving throughput/tail latency, correctness and low-precision checks, measured worker recovery |
| [Model Training](https://job-boards.greenhouse.io/xai/jobs/5086324007) | modeling judgment and useful model building | one reproducible data -> training -> evaluation -> failure diagnosis -> second iteration |

The public applications request an exceptional-work description in 100 words
or fewer. Write that statement from measured outcomes after the campaign. A
project implementation can be described today; capability or scaling gains
cannot be supplied by aspirational wording.

## Work order

The user's compute constraint is **free first; Colab Pro only if necessary**.
Use standard public-repository CPU CI for isolation/contracts and an interactive
free Colab session for the small single-GPU pilot. Reuse the already locked
`scripts/run_colab_rlvr_pilot.py` and `configs/colab_rlvr_pilot_v1.json`; do not
start with full GPU-serving/distributed dependency installation. Do not purchase
compute or start a paid session from this plan. Colab GPU allocation and runtime
availability are variable even on paid plans ([official FAQ](https://research.google.com/colaboratory/faq.html)).
One Colab GPU cannot validate 1-to-2 GPU scaling. Keep that outcome pending
until suitable hardware exists.

The historical pilot notebook is [`notebooks/free_colab_mts_pilot.ipynb`](../notebooks/free_colab_mts_pilot.ipynb).
It pins the existing research commit and original protocol, checks GPU presence,
retains the Colab CUDA PyTorch, installs only the locked model/data dependencies,
streams logs and exports successes, failures and partial results. It does not
execute model-generated code. The original locked pilot has now completed on free T4, with no capability gain.
The notebook deliberately preserves its affected source pin. A separate
zero-update GPU diagnostic confirms bounded corrected probability parity, while
historical-score reconstruction fails its own gate. See
[the raw evidence and limits](free_colab_pilot_evidence.md). A corrected training
replication needs a new source/protocol lock and separate outputs.

### 1. Make the real-workload evaluation trustworthy

Use `docs/mlsys_research_automation.md` as the initial operator development
entry. Extend it to frozen complete repository tasks. Start with a small number
of high-value task families, not a large generated inventory. Keep task identity,
source commit, dependencies, public feedback, private evaluator, resource budget,
failure outcomes and patch lineage immutable. Correctness always gates speed.

Development operators should not enter the private task bank. Split by task and
source family; numeric-seed variants are not independent tasks. Diagnose public
test exploitation, output forgery, gradient mistakes and benchmark shortcuts
before awarding model-training rewards.

### 2. Demonstrate learning and delayed-credit behavior

Reuse the token-exact rollout/GRPO stack and durable coding journals. Select an
open model and dataset revisions, record the untrained baseline, fix training
and evaluation budgets, and retain at least three independent training seeds
when resources permit. Compare an unchanged agent, a search/harness intervention,
and weight-updated RL. Keep these three effects separately attributable.

For post-training, reuse the oracle/fresh/stale/shuffled verifier campaign and
add a preference baseline on the same split. Select hyperparameters on dev only.
Measure independent task success, proxy reward, trusted reward and verifier
calibration under policy shift. Confidence intervals should reflect task and
run variation, not only repeated samples on one operator.

For long horizons, require genuinely completed tens-of-minutes trajectories
under a declared multi-hour budget. Record inspect/edit/test/benchmark actions,
failed hypotheses, actual wall time and injected interruption/recovery. Compare
terminal return with one defined credit-assignment intervention at equal budget.
A 7200-second timeout or short resumable episode is supporting mechanics only.

### 3. Measure systems impact on actual hardware

Run the existing locked GPU bundles after a compute environment and cost limit
are specified. Record model, library and image revisions and raw failures. Measure
one-GPU/multi-GPU throughput, latency percentiles, utilization, weight-sync cost,
FSDP/NCCL resume and worker recovery. Confirm operator improvements in actual
training or serving; IPC timing is insufficient for kernel-speed claims.

Profile first, change the measured bottleneck, rerun matched workloads, and retain
the numerical and quality regression checks. Pursue an upstream verl/SGLang/vLLM
patch only when a real bottleneck and reproducible reproducer support it.

## Repository decision

Keep the implementation connected to RVL while independently evaluated agent
learning and real AI-engineering workload outcomes are missing. The small T4
numerical diagnostic alone does not meet those extraction criteria. Immediately extracting the existing stack would duplicate
experiments and leave two repositories with the same missing evidence.

After a reproducible real-model/workload result, extract an independent flagship
repository such as `verified-ai-engineering-rl`: agent environments, rollout,
verifier, learner, promotion, recovery, performance experiments and CI. Keep
Recursive-Verification-Lag for theory, controlled experiments, negative results
and paper artifacts. Preserve commit history and source attribution, pin the
integration version, and run both repositories' checks before replacing code
with an adapter. See `systems_split_plan.md` for the existing exit criteria.

## Current claim boundary

The reviewed main branch already has substantial RL systems implementation,
real-model/CPU-DDP CI, and a retained free-T4 negative pilot with bounded
probability-consistency validation. The new MLSys entry proves only that an exact real-code
operator can be evaluated with external correctness/gradient checks and a
correctness-gated performance protocol. It is one public development task.

Missing outcomes remain: independently evaluated agent learning, a held-out
repository-task bank, completed long trajectories, real GPU scaling/serving
measurements and a coherent second model-building iteration. Null and negative
outcomes should remain visible. State what was learned and what decision it led
to; do not convert missing outcomes into positive application claims.
