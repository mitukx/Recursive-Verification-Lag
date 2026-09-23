# MBPP+ disjoint transfer v2: pre-generation protocol

**Status:** fixed after inspecting the eight-task scored pilot, **before**
generating or scoring any candidate on the next 64 tasks. This is one
outcome-informed redesign; it does not turn the same public benchmark into
a secret benchmark. `configs/mbppplus_task_split_v2.json` records the exact
task IDs and test hashes from the pinned parquet once its outcome-blind
selection job completes. Never replace a task because its model completions
are invalid or its trusted reward is uninformative.

## Task and model locks

- Official `evalplus/mbppplus`, commit
  `b2d74c91837c3f2a20c1299ae98133cbe7cfa077`, parquet SHA256
  `dc20030b3788fccf617444edcb34138ef13d7e4fafd17bfcb8c1279dbb12399b`.
- Preserve the original pilot's salt and eligibility: exactly one top-level
  reference function named in at least three public assertions; prompt
  shorter than 600 characters. Sort by SHA256(salt + task_id). The already
  scored ranks 0–7 are excluded; ranks 8–39 are **development** and 40–71
  are **heldout**. No eligibility depends on candidate or trusted outcomes.
- Qwen/Qwen2.5-Coder-1.5B-Instruct at immutable model revision
  `2e1fd397ee46e1388853d2af2c993145b0f1098a`; 16 samples per task,
  256 generated tokens, temperature .8, top-p .95. Use development seed
  `20260928`, heldout seed `20260929`, four samples per batch. The prompt
  template is unchanged from the first bank. Emit the unscored JSONL and
  bank hash before any Docker scoring. Use the same declared device for
  both splits; device-specific random-number implementations need not
  produce bit-identical generations. No model-parameter training is in
  this experiment.
- The official public assertions are the cheap verifier; the released
  additional tests are the evaluation-only trusted reward. They may occur
  in model pretraining. Execute model code only in disposable networkless
  Docker with a fixed resource limit, and validate **every reference**
  before any candidate scores. Pin dataset, image, scorer and bank hashes.

## Primary analysis and interpretation

1. On *every task* run the 32-paid-draw early/uniform/late fixed timing
   designs with twelve rounds, both soft strengths .25/1/4 and Best-of-N
   N=2/4/16, both public/all feature fits and seeds 0–4. All generated
   occurrences, including duplicates, invalid outputs and timeouts, stay.
   The optimizer updates only a distribution over this bank.
2. The timing-identification arm replays the exact same ordered stream of
   six distinct source texts, two paid labels per refresh, under early,
   uniform and late schedules. Source identity and final six labels must
   match per setting. Tasks with fewer than six distinct source texts
   cannot satisfy this intervention: report their IDs and counts and mark
   the source-timing contrast **unidentified** for them, while retaining
   them in the all-task 32-draw arm. Do not substitute another task.
3. Primary descriptive differences (early minus uniform): probability of
   any *new* below-initial crossing after round one among initially safe
   matched settings, and final true reward change on **all** settings.
   Report the initial failure rate and every task separately. Crossings
   before the intervention are never attributed to refresh timing. Unit
   for descriptive intervals is task, not the correlated seed/config run.
   If fewer than five distinct heldout task IDs show any later-onset
   crossing under a fixed schedule, call the failure boundary
   underidentified. A null timing contrast is retained.
4. Secondary: exact source-cost adaptive/box-certified controllers and
   full-safe-set projection, which may abstain and may concentrate on
   audited candidates. Report actual paid distinct labels, rejection,
   source concentration, and gain. An equal *maximum* budget is not a
   matched actual cost. Label-acquisition rules that change source ID
   answer a different question from timing.

## Predictor transfer and scope

Analyze candidate-proposed movement **before** seeing the trusted outcome
of that update. Fit any signs, thresholds or combinations of age×strength,
KL, max log-density ratio and verification-relevant geometry using only
development task outcomes. Write a frozen predictor file, versioned source
and a primary cost-sensitive warning target before heldout scoring. If the
metric requires a representation-specific residual envelope, its coverage
must be calibrated on independent development sources/tasks and checked on
heldout outcomes; the cheap-feature Lipschitz assumption was already
falsified on part of the earlier banks. Report a failed calibration event,
an all-safe stratum and an unidentified onset set instead of optimizing a
replacement on heldout scores.

Neither 32 development nor 32 heldout tasks implies learned capability
creation: this is a frozen-bank verifier-selection study. A parameter-update
or truly fresh proposal-generation loop must be designed as a distinct
experiment if these signals transfer.

## Reproducible commands and strict order

From the repository root, install `requirements-mbppplus.txt` and run the
generator only after the frozen task manifest is present:

```sh
python -m src.generate_mbppplus_bank --task-offset 8 --task-count 32 --samples 16 --seed 20260928 --frozen-split configs/mbppplus_task_split_v2.json --split development_unscored --output data/mbppplus_qwen15b_development_v2.jsonl
python -m src.generate_mbppplus_bank --task-offset 40 --task-count 32 --samples 16 --seed 20260929 --frozen-split configs/mbppplus_task_split_v2.json --split heldout_unscored --output data/mbppplus_qwen15b_heldout_v2.jsonl
```

The generator checks IDs, prompt/public/additional-test hashes, the model
revision, output count and seed **before loading model weights**. On a
Docker-enabled M1 Pro, build the image and record its SHA256 as in
`docs/mbppplus_real_program_protocol.md`; score development only at first:

```sh
python -m src.score_mbppplus_docker data/mbppplus_qwen15b_development_v2.jsonl --image "$RVL_IMAGE_ID" --frozen-split configs/mbppplus_task_split_v2.json --split development_unscored --output data/mbppplus_qwen15b_development_v2_scored.jsonl
```

Freeze development-derived warning rules and code hashes before running
the analogous heldout scorer with `--split heldout_unscored`. The scorer
checks bank/source hashes, tasks, samples and frozen split before executing
any candidate. Existing eight-task scored data remain a separate pilot.
