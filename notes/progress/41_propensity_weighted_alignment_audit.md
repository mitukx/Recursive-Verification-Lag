# Progress 41 — Propensity-weighted alignment auditing

Recorded 2026-10-07 (Asia/Tokyo). This follow-up was designed after seeing the
fixed-budget covariance-gate result, so it is a prospective efficiency study,
not an independent confirmation of the original mechanism.

## Question

Can randomized nonuniform trusted auditing improve verifier/truth covariance
estimation without introducing the selection bias of deterministic top-k
auditing?

Each candidate is independently audited with known inclusion probability
(pi_i>0). The probabilities depend only on free proxy-side information.
Trusted rewards never enter propensity allocation.

For target mass (p_i), proxy score (v_i), trusted reward (y_i), and
(mu_v=E_p[v]), the estimator is

[
widehat C
=
sum_i
rac{I_i p_i y_i(v_i-mu_v)}{pi_i}.
]

It is Horvitz--Thompson unbiased for
(C=operatorname{Cov}_p(y,v)). A Hoeffding radius is

[
r
=
sqrt{
rac{log(2/delta)}{2}
sum_i
left(
rac{p_i|v_i-mu_v|}{pi_i}
ight)^2
}.
]

At fixed expected audit budget, minimizing this bound yields the proxy-only
allocation

[
pi_ipropto
left(p_i|v_i-mu_v|ight)^{2/3}
]

before clipping to the declared minimum propensity and one.

## Locked experiment

- 1,024 independent tasks
- support size 64
- verifier-error norm (sigma=2)
- harmful alignment (ho=-0.75)
- benign alignment (ho=+0.75)
- expected trusted-label budgets 4, 8, 16, 32
- uniform Bernoulli audit versus proxy-leverage Bernoulli audit
- confidence (delta=0.05)
- every selected index and every propensity retained

The primary budget was 16 expected labels. The locked prediction required
proxy-leverage to have lower mean radius in both arms, pooled decisive rate no
worse than uniform, and wrong-sign decisive rate at most 0.08.

## Result

The primary passed.

At expected 16 labels:

| arm | method | mean radius | decisive rate | wrong-sign rate |
|---|---|---:|---:|---:|
| benign | uniform | 2.3101 | 0.0322 | 0 |
| benign | proxy leverage | 1.4936 | 0.2227 | 0 |
| harmful | uniform | 1.1406 | 0.0000 | 0 |
| harmful | proxy leverage | 0.6470 | 0.00293 | 0 |

Pooled decisive rate rises from 0.0161 to 0.1128, about a seven-fold increase,
at essentially the same expected trusted-label count. The confidence radius is
smaller under proxy leverage in both arms.

At expected 32 labels:

| arm | method | decisive rate |
|---|---|---:|
| benign | uniform | 0.5742 |
| benign | proxy leverage | 0.9990 |
| harmful | uniform | 0.0176 |
| harmful | proxy leverage | 0.0410 |

No wrong-sign decisive decision was observed in any arm, method or budget.

## Important asymmetry

The efficiency gain is not a claim that harmful alignment becomes easy to
detect. In this geometry, (sigma=2,ho=-0.75) gives a much smaller absolute
trusted-progress derivative than the benign arm. The harmful arm is therefore
nearer the sign boundary and remains mostly inconclusive even after 32 expected
labels.

This is useful rather than inconvenient: a controller should not turn weak
evidence into a safety claim. The correct action near the boundary is to remain
inconclusive, spend more trusted budget, reduce update size, or refresh the
verifier.

## Interpretation

Randomized known propensities provide a principled middle ground between:

1. uniform random audits, which are statistically clean but inefficient; and
2. deterministic priority audits, which can be efficient operationally but do
   not identify the full-support covariance without additional assumptions.

The proxy-leverage allocation improves the **bound before observing trusted
reward**, while Horvitz--Thompson weighting corrects the resulting selection
bias.

This does not yet solve population generalization for a learned LLM policy. It
certifies the declared finite support or fixed rollout batch. A production
version must also preserve policy-version provenance and distinguish finite-batch
alignment from population alignment.

## Evidence

- protocol: `configs/propensity_alignment_audit_v1.json`
- implementation: `src/propensity_alignment_audit.py`
- tests: `tests/test_propensity_alignment_audit.py`
- theory: `notes/theory/propensity_weighted_alignment_audit.md`
- results: `results/propensity_alignment_audit_v1/`
- complete selections/propensities: `selection_records.jsonl.gz`
- complete task rows: `rows.csv.gz`
- successful workflow: `rvl-propensity-alignment-audit`

The result motivates a systems follow-up: replace deterministic priority-only
statistical claims with explicit randomized audit propensities and persist those
propensities in the rollout ledger.
