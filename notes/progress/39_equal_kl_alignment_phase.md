# Progress 39 — Prospective equal-KL verifier-error alignment phase

Recorded 2026-10-07 (Asia/Tokyo). The protocol was committed before executing
the outcome grid. This is a controlled finite-support mechanism experiment, not
heldout LLM evidence.

## Locked prediction

For an exponential update q_beta proportional to p exp(beta v),

d/d beta E_q[y] at beta=0 = Cov_p(y,v),

while proxy progress has derivative Var_p(v) > 0 for a nonconstant score.
The locked construction writes the verifier error as

e = sigma * (rho * y_std + sqrt(1-rho^2) * u),

with u p-centered, p-unit and p-orthogonal to the standardized true reward.
Therefore the local true-progress sign is exactly sign(1 + sigma*rho). The
experiment holds the baseline, true reward, error norm sigma and target policy
KL fixed while rotating only rho.

Primary endpoint: requested KL 0.005, excluding exact first-order boundary cells.
The pre-registered prediction required at least 90% agreement between the sign
of mean true progress and sign(1+sigma*rho), with nonnegative proxy progress.

## Executed result

GitHub Actions run 37490793700 completed successfully and committed the
deterministic evidence to this branch. The full grid contains 73,728 task/config
rows from 512 independent finite-support tasks. 71,655 rows attain the requested
KL exactly; 2,073 unattainable cases are retained rather than silently replaced.

The primary prediction passed with phase accuracy 1.0. At KL 0.005, all four
predicted harmful cells have false-progress rate 1.0 and all thirty predicted
benign cells have false-progress rate 0.0. Mean true progress is -0.03021 in
harmful cells versus +0.03676 in benign cells, while every cell still has
positive mean proxy progress.

The same mean-sign phase boundary remains correct at all four requested KL
budgets:

| requested KL | phase accuracy | harmful false-progress rate | benign false-progress rate |
|---:|---:|---:|---:|
| 0.005 | 1.000 | 1.0000 | 0.0000 |
| 0.020 | 1.000 | 1.0000 | 0.0000 |
| 0.100 | 1.000 | 0.9956 | 0.0000 |
| 0.300 | 1.000 | 0.9668 | 0.00117 |

The exact first-order boundary sigma=2, rho=-0.5 was excluded from the primary
sign test as declared. Its finite-KL false-progress rate stays near one half
(0.488 at KL 0.005 and 0.496 at KL 0.3), consistent with first-order ambiguity.

## Interpretation

This is a direct counterexample to using policy KL alone as a safety coordinate.
At the same KL and the same verifier-error norm, rotating error alignment can
flip the sign of true progress while the optimized proxy still improves.

The result strengthens the RVL mechanism claim:

> verification difficulty is controlled by policy movement relative to
> verifier-error geometry, not by elapsed rounds or policy KL alone.

It does not establish that a learned LLM verifier exhibits this exact synthetic
phase boundary. The next external-validity step is a real-model experiment that
measures or intervenes on verifier-error alignment while preserving comparable
policy displacement.

## Evidence and reproduction

- Protocol: configs/equal_kl_alignment_stress_v1.json
- Theory: notes/theory/equal_kl_false_progress_geometry.md
- Runner: src/equal_kl_alignment_stress.py
- Tests: tests/test_equal_kl_alignment_stress.py
- Results: results/equal_kl_alignment_stress_v1/
- Successful workflow: GitHub Actions run 37490793700

The committed manifest records hashes for cells.csv, summary.csv and the
compressed 73,728-row raw table rows.csv.gz. The earlier workflow run
37490435755 also completed the scientific computation successfully but its final
git push was rejected because the research branch advanced concurrently; the
workflow was made fetch/rebase-safe and rerun without changing the locked
protocol or analysis code.
