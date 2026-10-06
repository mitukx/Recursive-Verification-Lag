# Progress 36 — Completed development evidence, negative fresh-task endpoint

Recorded 2026-10-07 (Asia/Tokyo). Development run
[37389940505](https://github.com/mitukx/Recursive-Verification-Lag/actions/runs/37389940505)
and transfer run
[37393792712](https://github.com/mitukx/Recursive-Verification-Lag/actions/runs/37393792712)
both completed successfully. The scored bank and complete analysis outputs are
now archived in Git, rather than relying on expiring Actions artifacts.

## Primary result

The pre-outcome development lock uses 16 training and 16 new evaluation tasks.
The frozen fit consumes 32 training-source labels; the refreshed fit consumes
64. Both arms acquire the same final source set and total ledger cost, but
**their decision-time information differs**. This is not a pure timing
comparison with equal information at decision time.

After four soft updates (eta=1, all cheap features), frozen expected pass
probability is 0.7781233367 and refreshed is 0.7778638339. The primary
refreshed-minus-frozen difference is **-0.0002595027** (-0.02595 percentage
points); its descriptive task-bootstrap interval is
[-0.0014615791, 0.0008056867]. Improvement is not demonstrated.
The zero-label public-score control reaches 0.7935806532; uniform occurrence
selection reaches 0.55078125. All 16 tasks, including two zero-success tasks,
remain in the denominator. These are probabilities on a frozen bank, not
freshly generated pass@1 measurements or learned generator improvements.

Public-only soft refresh has a small positive secondary contrast. Its
best-of-N policies are unchanged by refresh. A positive scalar slope on one
public feature changes exponential-policy temperature without changing
ranking. Therefore the secondary result cannot yet be called better error
correction. The next diagnostic will compare policies at equal KL from their
common baseline and record ranking changes. It will be labeled exploratory:
this decision was made after inspecting the locked primary and secondary
summaries. No heldout outcome has been opened.

## Development timing study

The 32-task extension retains 512 generated occurrences, 32 reference
validations and all invalid/duplicate outputs. Five tasks show some later-onset
failure in the prescribed settings (142, 264, 410, 476, 778). This meets a
**development triage** gate only, not the heldout identification criterion.

The identical-source-stream comparison covers 30 tasks (127 and 139 have
fewer than six distinct source texts). Early-minus-uniform final gain is
0.00566705, descriptive task-bootstrap interval [-0.00100962, 0.01449130].
Late-minus-uniform is -0.01591276, interval [-0.02444678, -0.00810972].
The later-onset-rate intervals cross zero. Different exposures to intermediate
scores can explain endpoint changes; see the existing exposure identity in
`docs/refresh_timing_identification.md`.

## Reproduction and provenance

- Bank: `data/mbppplus_qwen15b_development_v2_scored.jsonl`.
- Bank SHA-256: `1aba49310cf5fd44bedabb9dd0ee25b70e5c0ef2e98b4a01611c450f35755d82`.
- Original transfer outputs: `results/fresh_task_verifier_transfer_v1/`.
- Original development outputs: `results/mbppplus_development_v2_archive/`.
  Large JSONL files are deterministic gzip archives. `archive_manifest.json`
  records both stored and decompressed hashes. No rows were filtered.
- Transfer code and all seven output hashes were checked against the original
  manifest. Physical scoring, reference validation and controller costs remain
  separate. Evaluation-task decision-time trusted queries: zero.

```sh
python -m src.fresh_task_transfer data/mbppplus_qwen15b_development_v2_scored.jsonl --output results/fresh_task_transfer_reproduction
```

## Frontier-lab goal

The useful next research increment is a calibration/ranking diagnosis followed
by a separately locked independent test if a specific claim survives.
The existing GPU workflow is queued without an available self-hosted runner;
queued jobs provide no GPU evidence. Frontier-lab readiness and unrestricted
recursive improvement remain unestablished.
