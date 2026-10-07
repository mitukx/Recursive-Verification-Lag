# Progress 38 — Colab T4 real-model pilot locked and launched

Recorded 2026-10-07 (Asia/Tokyo). The user authorized choosing an available
compute environment, including Colab. A T4 runtime was connected in a new
[research notebook](https://colab.research.google.com/drive/1Hp9O2jWuK2unpjAQVvXna_DtspzDYxI2).
No subscription was purchased and no Google Drive mount or GitHub credential
is sent to the remote runtime.

The pre-execution protocol is `configs/colab_rlvr_pilot_v1.json` (commit
`7277510`). The executed runner includes a pre-execution dataclass-contract
correction at source `7493603`. The systems implementation is separately pinned
to `1a63903d22e1233594d1f1f0c269f7f55baf99b7`; Qwen model and GSM8K dataset
revisions, task-selection rule, dependencies, optimizer, generation budget,
precision, seeds, arms and report criteria are fixed in the protocol.

The pilot uses full-parameter FP32 GRPO, three seeds (17,29,43), four updates,
one prompt per update, four samples per prompt and 16 official train / 16
independent official test tasks. True-reward and within-prompt shuffled-reward
arms share the same split and initial generation seeds. An initial greedy
baseline is repeated without training to check reproducibility. Only terminal
evaluation measures each learned policy; evaluation does not promote, reject or
choose updates. Generation temperature is one, matching the trainer's base
policy log-probabilities. All raw rollout token records and per-update gradients,
clipping, ratios, parameter probes, timings and terminal predictions are saved.

**Status at this commit: running / pending evidence.** Runtime connection and
script launch are not successful training, improvement, or a GPU benchmark
result. Failed attempts and partial outputs will be retained. This small pilot
cannot establish general RVL, frontier-scale learning.

Reproduction on a CUDA environment:

```sh
pip install transformers==4.57.1 datasets==3.6.0 accelerate==1.10.1
python scripts/run_colab_rlvr_pilot.py
```

The runner writes `results/colab_rlvr_pilot_v1/` and refuses to overwrite an
existing attempt. Download its raw output after execution before releasing the
Colab runtime. Any follow-up design is a separately labeled lock; do not
substitute tasks, model or precision to erase a failed outcome.
