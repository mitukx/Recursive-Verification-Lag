# Mini frontier lab: executable systems research

This lab reproduces post-training failure modes at small scale. It is not a
frontier model, a hiring guarantee, or evidence of 7B/30B capability gains.
The research question is whether adaptive trusted verification improves learning
relative to fixed cadence at comparable actual label cost.

## Three executable paths

1. **CPU/Mac RVL lab:** multi-step restricted-DSL coding agents, independent
   asynchronous actor tasks and learner, clipped per-decision off-policy
   gradients, leased SQLite replay, trusted-test audits, learned residual critic,
   curriculum, failure-prioritized generation, bandit red-team search and
   trusted-success imitation. No third-party dependencies.
2. **Real causal LM lab:** separate HF inference and learner model copies,
   immutable behavior token IDs/logprobs, complete prompt groups in durable
   replay, concurrent inference/backward, atomic model/optimizer checkpoint
   pointers, trusted-label residual fitting and stale reward re-evaluation.
   The tiny-model parity smoke is a mechanics test, not a capability benchmark.
3. **Model-driven Python coding RL:** JSON inspect/edit/public_test/finish agents,
   episode-pinned weights, per-tool journals, Docker executable public/hidden
   tests, episode-level terminal credit, grouped candidate returns and the
   same asynchronous LM learner. Useful training requires a model that can
   produce valid tool calls. No coding-capability gain has yet been established.

The existing scheduling/RPC/dynamic-batching stack remains available. The new
blue/green serving fleet uses its load balancing, admission, deadlines and
failover, while retaining old policy pools for in-flight episode leases.

## Run on CPU or an Apple Silicon Mac

Python 3.11+ on Linux/macOS:

~~~bash
python -m unittest discover -s tests -p 'test_mini_lab*.py' -v
python -m src.run_mini_lab --output artifacts/async-lab --episodes 256 --actors 4
python -m src.run_mini_lab --output artifacts/repro --episodes 256 --actors 1 --deterministic --tool-latency-ms 0
python -m src.benchmark_mini_lab --output artifacts/ablation --seeds 17,29,43 --episodes 256 --learning-rates 0.08,0.2 --cadences 2,8
~~~

Reuse a run directory with a larger episode target to resume. Other configuration
changes are rejected. Benchmarks require a fresh directory. Reports include
actual trusted calls, exploit counts, policy/reward versions, replay states,
p50/p95/p99 latency, tool actions/s and committed semantic hashes. CPU tool
actions are explicitly not labeled LM tokens.

Deterministic mode fixes episode/batch scheduling to one actor. Concurrent
execution preserves version and replay invariants, but OS scheduling can change
which version an episode uses. Floating-point values are rounded to ten decimal
places only in the semantic report hash; full-precision checkpoint checksums
remain exact. This is not a cross-device bitwise floating-point guarantee.

~~~bash
pip install -r requirements-systems.txt
python -m unittest discover -s tests -p 'test_mini_lab_torch.py' -v
python -m src.run_async_lm_lab --model sshleifer/tiny-gpt2 --groups 3 --samples 4
torchrun --standalone --nproc-per-node=2 -m src.run_distributed_mini_lab
~~~

The local LM reference holds two model copies. It is suitable for mechanics
tests and small-model experiments, not a memory-efficient 30B deployment.

## Real Python coding agents

Build the existing isolated runtime, inspect its immutable image ID, and pass
that exact ID:

~~~bash
docker build -f docker/Dockerfile.mbppplus -t rvl-agent-runtime .
docker image inspect rvl-agent-runtime --format '{{.Id}}'
python -m src.run_coding_agent_lab --model YOUR_TOOL_CAPABLE_MODEL --image sha256:YOUR_IMAGE_ID --platform linux/arm64 --device mps --samples 4 --max-steps 32
~~~

On Linux/CUDA use the corresponding platform/device. The model is not downloaded
until this command is explicitly run. Longer experiments can configure up to
128+ tool calls and a 7200-second deadline. **Multi-hour completed trajectories
have not been validated.** The deadline is a supported budget, not a measured
long-horizon success result.

Expected outputs stay in the evaluator process. Candidate Python runs in a
disposable non-root container without host mounts or networking, with CPU,
memory, PID, output and wall-clock limits. The external evaluator compares JSON
values, including their types. Forged reward objects, assertion replacement and
output flooding are tested. This does not establish robustness against all
container escapes or adaptive benchmark contamination.

## RVL interventions

The CPU reference estimates endpoint KL from the last verified policy anchor.
The LM path uses an accumulated sampled behavior-KL diagnostic; it is not an
exact endpoint KL. Both combine policy movement with verifier disagreement,
posterior uncertainty, audited-cell coverage, observed proxy/trusted residuals,
policy lag and reward-version lag.

Audit acquisition ranks these quantities subject to a shared trusted-label
budget. Critic fitting occurs when enough new labels arrive. Reward re-evaluation
keeps the original behavior tokens/logprobs and policy version unchanged.
Search-generated attacks do not pretend to be stochastic behavior-policy samples.
Only trusted successful search candidates enter imitation training.

The controller is **heuristic**, not a theorem-backed safety certificate.
Feature cells can remain misspecified. A critic trained on audit labels is not
an independent stronger LLM. Remote JSON LLM judges can be added via JSONJudge
and MultiVerifier; their capabilities and resistance to prompt injection must
be measured separately. Required grader failures abstain and reject rather than
silently granting reward.

The bandit attacker changes its choice among constant-output, sign-reversal and
off-by-one programs based on observed exploitation and current verifier scores.
The verifier refits from those attacks. This demonstrates narrow co-evolution;
it is not general autonomous research self-improvement.

## Replay and recovery

- One local control-plane driver owns an OS file lock. Independent processes
  may use the SQLite replay interface on the same host.
- Actor reservations persist before rollout. Each tool action has a resumable
  journal and pinned policy provenance.
- Replay insertion is idempotent; incompatible reuse of an ID raises an error.
- Expiring leases and fencing tokens prevent late workers acknowledging a
  reclaimed batch. Future-version samples are not admitted to a learner.
- Over-age trajectories are archived as stale rather than silently trained.
- CPU learner state, critic/controller state and replay acknowledgement commit
  in one SQLite transaction. On restart, abandoned local leases are reclaimed.
- LM weights/optimizer are written first; their checkpoint pointer and replay
  consumption commit together. Orphan checkpoint files are ignored.
- The CPU held-out online gate can withhold new actor-serving weights. It does
  not turn a reused gate suite into an independent final test set.
- Capacity bounds ready/in-flight replay, not all archived disk usage. A
  production deployment needs retention, distributed storage and monitoring.
- Trusted-call budgets count logical audits in these reference experiments.
  Exactly-once billing across crashes of external judge services needs a
  provider-side idempotency ledger; it is not claimed here.

## Distributed learners and serving

The token-replay trainer performs real GRPO-style gradients under torchrun.
Global reward grouping is computed before rank sharding. Equal rank-local
sample counts avoid mismatched backward collectives. Backward microbatches
release each response graph; DDP/FSDP synchronize at the final microbatch.
Reported loss/reward statistics reduce across ranks. DDP parameters are checked
for agreement, and FSDP uses distributed model/optimizer checkpoints.

~~~bash
# Two or more CUDA GPUs; explicitly run only on configured hardware.
torchrun --standalone --nproc-per-node=2 -m src.run_distributed_mini_lab --mode fsdp

# Multi-node launcher contract, each node with its own NODE_RANK.
torchrun --nnodes=2 --nproc-per-node=4 --node-rank="$NODE_RANK" --rdzv-id=rvl-lab --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:29500" -m src.run_distributed_mini_lab --mode fsdp --model /shared/model --replay /shared/token-experiences.jsonl --output /shared/checkpoints
~~~

TokenServingBackend requests server-owned prompt/response token IDs and validates
model identity and one logprob per response token. Missing IDs are rejected;
retokenizing server text would silently change the behavior likelihood. It can
be used as a worker backend in VersionedServingFleet. This adapter has schema
contract tests; real GPU vLLM/SGLang compatibility still needs acceptance evidence.

Each JSONL record is a VerifiedGeneration, including prompt_token_ids,
response_token_ids and response_token_logprobs. This entry point is a
**distributed replay learner**, not yet a fully integrated multi-node online
actor/learner service. No NCCL/FSDP GPU run is claimed unless a linked acceptance
run passes. The manual GPU workflow requires an already configured two-GPU
self-hosted runner and installed compatible dependencies.

VersionedServingFleet prepares health-probed immutable worker pools, activates a
new pool for new episodes, and drains the old pool only after all episode leases
end. It inherits the existing scheduler's balancing and backpressure. Production
vLLM/SGLang endpoints must serve immutable model versions with verified artifacts.
This is blue/green routing requiring spare capacity; in-place zero-downtime weight
reload and multi-node vLLM training synchronization remain unverified.

GPUProfiler records actual nvidia-smi utilization, memory and power samples when
available. MFU remains null unless explicit FLOPs/token and aggregate hardware
peak inputs are supplied to mfu_estimate. No CPU throughput is extrapolated to
GPU tokens/s or 7B/30B speed.

Primary API references:
[PyTorch FSDP](https://docs.pytorch.org/docs/stable/fsdp.html),
[distributed checkpoints](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html),
[vLLM serving](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).

## Acceptance boundaries

| Requirement | What runs | What remains |
|---|---|---|
| End-to-end training | CPU tool policy and real causal-LM gradients | held-out pretrained coding gains |
| Fully async actors/learner | 4 CPU actors; separate LM inference/learner | distributed online tensor learner service |
| Long horizon | bounded/resumable model tool episodes | actual multi-hour completion/recovery measurements |
| Experience management | WAL replay, leases, idempotency, version/lag filters | network replay service, retention, fault domains |
| Weight publication | immutable CPU artifacts; LM checkpoint publication; episode-pinned pools | real GPU fleet reload and multi-node delivery |
| Serving | existing vLLM/SGLang HTTP adapters; blue/green scheduler | actual GPU serving throughput/uptime measurements |
| Multi-verifier | Docker I/O tests, Bayesian critics, async remote judge adapter | calibrated stronger-model judge benchmark |
| Exploitation | public-test overfitting, forged grading, assertion replacement, flooding | broader adaptive red teaming |
| Self-improvement | task curriculum, failure replay, bandit attacks, verifier fitting, imitation | open-ended research-task generation/self-play |
| Online evaluation | frozen gate suite, regression blocking | independent final suite and statistical deployment gates |
| Distributed learner | two-rank CPU DDP clipped token RL | CUDA/NCCL/FSDP and multi-node 7B/30B validation |
| Profiling | actual CPU/LM timings; GPU sampler; explicit-input MFU | sustained GPU/MFU scaling curves |

A researcher can use this code to study real control-plane errors and narrow
learning dynamics. Evaluation by xAI/OpenAI/Anthropic depends on demonstrated
results, debugging ability and explaining these limitations; the project does
not claim their internal architecture or hiring approval.
