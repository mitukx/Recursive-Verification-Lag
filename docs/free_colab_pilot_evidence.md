# Free T4 pilot: negative learning outcome and a probability mismatch

Measured on 2026-10-08 (Tokyo; attempt started 2026-10-07 UTC), using free
interactive Colab and one Tesla T4. No paid compute was started. The model was
Qwen2.5-0.5B-Instruct at revision `7ae557604adf67be50417f59c2c2f167def9a775`;
GSM8K was pinned to `740312add88f781978c0658806c59bc2815b9866`.
The runtime used PyTorch 2.11.0+cu130, Transformers 4.57.1 and FP32.

## What actually ran

The original [locked protocol](../configs/colab_rlvr_pilot_v1.json) was executed
without changing model, tasks, seeds, learning rate or training budget. It used
16 training tasks, 16 separate evaluation tasks, seeds 17/29/43, trusted versus
within-prompt shuffled reward, four updates per condition and four completions
per update. All six conditions completed: 24 optimizer steps, 96 training
rollouts and 24,077 response tokens. Only seven updates had nonzero RL gradients;
all four updates of seed 29 had constant zero rewards in each arm.

The unchanged greedy baseline and its zero-update repeat were identical on all
16 tasks. The baseline was 5/16 correct (31.25%).

| Seed | Trusted terminal | Shuffled terminal | Each arm's change from baseline |
|---|---:|---:|---:|
| 17 | 1/16 (6.25%) | 1/16 (6.25%) | -25 percentage points |
| 29 | 5/16 (31.25%) | 5/16 (31.25%) | 0 |
| 43 | 3/16 (18.75%) | 3/16 (18.75%) | -12.5 percentage points |

Both arms averaged 18.75% terminal accuracy: -12.5 percentage points relative
to the baseline. Trusted minus shuffled averaged zero; the descriptive paired
task-bootstrap interval is [-6.25, +6.25] percentage points. This interval is
conditional on these three retained seeds and does not quantify seed uncertainty.
Equal counts do not imply identical predictions. This tiny, subsequently
inspected evaluation split is not an independent confirmatory result.

Peak allocated GPU memory was 10,506,167,808 bytes (9.785 GiB). Recorded training
forward/backward/optimizer time totals 42.74 seconds; that excludes generation
and evaluation and is **not** end-to-end runtime or a speedup measurement.
The parameter-motion probe covers only the first 4,096 elements of the first
parameter; it is not a full-model weight-change measurement.

## Numerical bug and bounded GPU validation

Before the first update, the first trusted group already had a maximum absolute
log-probability ratio of 2.22088 and a mean clipping fraction of 39.43%.
A same-weight fresh rollout should not have this discrepancy. The pinned
[model generation configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/generation_config.json)
sets `repetition_penalty=1.1`. Generation inherited that transformation while
the learner scored raw model logits. This is a demonstrated distribution
mismatch; the remaining replay residual below prevents claiming that this
single processor explains every numerical difference.

The corrected backend supplies neutral generation controls explicitly, retains
model special-token configuration, and records sampling temperature. The
learner scores the same temperature-scaled distribution; greedy scores are
rejected for training. EOS tokens sharing the padding ID remain scored actions.
Transformers is pinned to the tested 4.57.1 API. Real-model regression tests also
cover inherited beam/suppression settings and restoration of training mode.

A separate [exploratory diagnostic lock](../configs/hf_behavior_parity_diagnostic_v1.json)
was fixed after observing the original pilot. It performs **zero optimization
steps**. Both numerical tolerances were set to 1e-4 before running it.

| GPU diagnostic | Observed maximum | Declared gate |
|---|---:|---|
| Old recorded scores versus raw initial-weight scores, 4 retained actions | 2.22087669 | descriptive diagnosis |
| Old scores reconstructed with repetition penalty 1.1 | 0.00110242 | **FAIL**, greater than 1e-4 |
| Corrected sampling versus learner, 6 new actions at temperatures 0.7/1.0/1.3 | 0.0000746250 | **PASS**, at most 1e-4 |

All six corrected samples had zero clipping fraction. They use at most 64
new tokens; the old pilot used at most 256, so the two measurements are not a
matched long-sequence comparison. The diagnostic **process exited 1 overall**:
its historical-score reconstruction failed, even though corrected parity passed.
The tolerance was not relaxed. The residual remains unresolved; attributing it
to CUDA/cache numerics would require another controlled experiment.
The first diagnostic attempt also failed before loading the model because the
verifier compared the original lock's byte hash with a reformatted JSON copy.
That verifier bug was repaired by checking the source hash and semantic identity
separately. Both failed attempts and their logs remain visible.

Global Colab `pip check` reported pre-existing package conflicts. Those conflicts
were retained; pinned imports and a CUDA matrix smoke check passed. This does
not certify the whole Colab image as dependency-clean.

## Audit and reproduction

The [evidence directory](../results/colab_rlvr_pilot_20261008/20261007T162508345719Z)
contains all 52 indexed raw files, diagnostic source snapshots and a derived
scorecard. The original pilot and first diagnostic artifacts are unchanged in
the second export. Checksum verification proves artifact consistency, not remote
hardware attestation. Stored predictions and rewards are independently rescored;
legacy score errors are recomputed from retained arrays. Corrected learner
statistics are captured measurements whose aggregates are checked; CPU CI does
not independently regenerate their GPU logits.

```bash
pip install numpy==2.2.6
python -m scripts.verify_colab_evidence_bundle
python -m unittest tests.test_colab_rlvr_pilot_summary tests.test_colab_evidence_bundle -v
```

A successful artifact check explicitly prints `diagnostic_process_status:
failed`; it verifies faithful reporting rather than promoting a failed experiment.
The historical [Colab notebook](../notebooks/free_colab_mts_pilot.ipynb) still pins
the original affected implementation for reproduction. It must not be mistaken
for a corrected training run. The isolated corrected numerical diagnostic can
be run on another CUDA runtime with the retained pilot evidence:

```bash
pip install -r requirements-systems.txt
python -m scripts.run_hf_behavior_parity_diagnostic \
  --pilot-evidence results/colab_rlvr_pilot_20261008/20261007T162508345719Z/raw/pilot-results \
  --output /tmp/rvl-new-behavior-diagnostic
```

The output directory must be new. A corrected *learning* experiment requires a
new source/protocol lock, a fresh parity gate before updates, and separate raw
outputs. Do not overwrite this historical pilot or reuse its inspected test
split as independent confirmation.

## MTS evidence this supports

This supports a concrete engineering account: execute a pinned real-model RL
pilot on free hardware; identify an unexpected fresh-policy ratio error; repair
sampler/learner distribution semantics; validate bounded parity on T4; retain
negative learning results, failed checks and machine-verifiable raw evidence.
It does **not** demonstrate capability improvement, long-horizon learning,
held-out AI-engineering agent gains, kernel speedups, multi-GPU scaling,
production Grok infrastructure or involvement in building Grok. The original
mismatched training path also prevents a causal claim about correct GRPO learning.
