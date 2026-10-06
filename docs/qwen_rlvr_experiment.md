# Held-out Qwen RLVR experiment

This experiment is the capability-gain acceptance path for the systems stack.
It is separate from the small CI smoke test.

## Dataset

The default task source is the current Hugging Face `openai/gsm8k` dataset,
configuration `main`. Training examples are sampled from the official train
split and evaluation examples from the official test split. The runner extracts
the GSM8K final numeric answer and asks the model to end its response with
`#### <number>`.

A custom JSONL source is also supported with fields:

```json
{"id":"task-1","prompt":"...","answer":"42","split":"train"}
```

Use `split=eval` or `split=test` for held-out examples.

## Protocol

1. Load one Qwen/Hugging Face causal LM.
2. Measure greedy held-out accuracy before training.
3. Generate multiple sampled responses for each training prompt.
4. Score each response with deterministic numeric verification.
5. Normalize rewards per prompt and apply the repository's clipped token-level
   GRPO/PPO-style update.
6. Repeat for the configured number of steps.
7. Measure greedy held-out accuracy again on the same frozen evaluation set.
8. Retain per-example before/after predictions, training history, telemetry,
   environment metadata and checksums.

The model used for rollout and learning is the same model instance. Behavior
token IDs and token log-probabilities are captured before each update and are
recomputed differentiably during training.

## Run

```bash
pip install -r requirements-systems.txt
pip install -r requirements-gpu.txt
bash scripts/run_qwen_rlvr_gpu.sh
```

For a first free-GPU run, keep the default Qwen 0.5B model and small limits.
Increase train/eval examples only after the full pipeline is stable.

A positive accuracy delta is not hard-coded as a test condition. RL results are
empirical, and a negative or zero result must remain visible rather than being
hidden by a benchmark script. The hiring evidence is the raw before/after
result, training diagnostics, exact config and the engineering explanation of
why the result moved or did not move.

## Transactional promotion mode

Add `--transactional-promotion` to treat every optimizer step as a candidate. Before the step, the trainer snapshots model parameters, AdamW optimizer state and RNG state. The incumbent and candidate are then greedily evaluated on the same frozen held-out task identities. A deterministic regression gate checks mean accuracy delta and uncompensated new failures; rejected updates restore the complete training state instead of becoming the parent of the next step. Every decision is appended to `promotion-ledger.jsonl` as a hash-chained audit record. This is a deployment/regression gate, not a significance test.
