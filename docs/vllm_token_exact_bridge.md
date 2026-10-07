# Real vLLM -> token-exact GRPO bridge

This evidence path closes a specific systems gap: a rollout produced by an
OpenAI-compatible vLLM service must be consumable by the GRPO learner without
client-side retokenization or reconstructed behavior likelihoods.

The bridge uses the existing fail-closed `TokenServingBackend`, which requests
server-owned prompt/response token IDs and one sampled log-probability per
response token. The replay is verified on pinned GSM8K tasks, frozen to disk,
and hashed. The vLLM server is then stopped and the exact same immutable model
snapshot is loaded into the HF GRPO learner.

Before any optimizer step, the learner recomputes token log-probabilities on the
server-provided IDs and records the maximum absolute behavior log-ratio. The run
fails closed if that pre-update serving/learner mismatch exceeds the declared
threshold. Only after this parity check does the learner execute a real GRPO
step. The report retains gradient metrics and a deterministic sampled-parameter
delta probe.

The GPU launcher pins the Hugging Face model revision to a local snapshot before
either serving or learning, records separate serving/training GPU telemetry,
retains raw server logs and dependency versions, validates replay identity
across both phases, and builds a SHA-256 evidence manifest.

This is a serving/training compatibility result, not a benchmark-quality or
capability-improvement claim. A zero parameter delta is retained if the sampled
reward groups contain no useful within-group variance.
