# Progress 40 — Geometry-aware trusted-label gate

Recorded 2026-10-07 (Asia/Tokyo). This experiment was locked before execution
and is separate from the earlier matched-KL phase experiment.

## Question

Can a small amount of trusted supervision distinguish a harmful verifier update
from a benign one when policy-shift coordinates alone are identical or
uninformative?

The local exponential-update derivative is

[
left.rac{d}{deta}mathbb E_{q_eta}[y]ight|_{eta=0}
=operatorname{Cov}_p(y,v).
]

The audit therefore estimates verifier/truth covariance directly from paid
labels instead of using policy KL as a safety proxy.

## Locked controller

For each finite-support task:

- proxy scores are free and their baseline mean is known exactly;
- trusted binary reward is observed only on sampled audit items;
- estimate `Cov_p(y,v)` with
  `mean y_i * (v_i - E_p[v])`;
- form a two-sided Hoeffding interval using the exact support range;
- `UCB < 0` => block/refresh;
- `LCB > 0` => allow only the local direction;
- otherwise remain inconclusive.

The protocol fixes 1,024 tasks, support size 32, error norm sigma=2, harmful
rho=-1, benign rho=+1, confidence delta=0.05 and trusted-label budgets
4, 8, 16, 32, 64 and 128.

## Result

GitHub Actions executed and archived all 12,288 task/arm/budget rows. The
pre-registered n=128 primary passed:

- harmful block rate: 1.000
- benign allow rate: 1.000
- harmful wrong-sign decisive rate: 0
- benign wrong-sign decisive rate: 0

The useful result is that the controller becomes decisive much earlier:

| trusted labels | harmful block | benign allow | inconclusive harmful | inconclusive benign |
|---:|---:|---:|---:|---:|
| 4 | 0.3477 | 0.3242 | 0.6523 | 0.6758 |
| 8 | 0.6035 | 0.6260 | 0.3965 | 0.3740 |
| 16 | 0.8408 | 0.8213 | 0.1592 | 0.1787 |
| 32 | 0.9775 | 0.9775 | 0.0225 | 0.0225 |
| 64 | 1.0000 | 0.9971 | 0.0000 | 0.0029 |
| 128 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

No wrong-sign decisive decision was observed at any reported budget. Empirical
coverage of the nominal 95% interval was approximately 99% or higher in every
arm/budget cell; this conservatism is expected from the bounded Hoeffding
construction.

## Interpretation

This is the first step in the project that turns the verifier-error-geometry
mechanism into an explicit bounded controller. A policy-only gate cannot
distinguish the paired worlds at baseline because both have KL=0 and identical
policy state. The trusted-label covariance gate can distinguish them.

The result does not show that 16 or 32 labels are sufficient in a real LLM
setting. The synthetic arms are deliberately strong alignment interventions.
Its value is to establish the controller contract and sample-complexity logic
before attempting the same measurement on neural-policy rollouts.

## Repository evidence

- protocol: `configs/alignment_covariance_audit_v1.json`
- implementation: `src/alignment_covariance_audit.py`
- tests: `tests/test_alignment_covariance_audit.py`
- theory: `notes/theory/alignment_covariance_audit_bound.md`
- results: `results/alignment_covariance_audit_v1/`
- full raw table: `rows.csv.gz`
- successful workflow: `rvl-alignment-covariance-audit`

The real-model bridge remains the next external-validity gate. Its self-hosted
GPU workflow has been triggered on
`gpu-evidence/qwen-alignment-bridge-v1`; the scientific protocol must not be
changed in response to its eventual outcome.
