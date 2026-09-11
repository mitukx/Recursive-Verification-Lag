# Progress Update XVI — Finite-domain program-synthesis threshold experiment

**Date:** 2026-09-11  
**Status:** Completed. The §FD budget-independent threshold signature survives in a finite program-semantics environment with explicit public-test overfitting and an actually fitted misspecified verifier.

## 1. Question

The previous experiment showed a stable optimization-strength threshold in a non-Gaussian synthetic reward model. This update moves one empirical level closer to the intended application.

The test is whether a verifier trained from trusted semantic observations still exhibits a **budget-insensitive failure threshold** when candidate programs can overfit a public test suite.

The falsifier is direct:

> If increasing trusted-data budget substantially moves the critical optimization strength, the §FD interpretation is not robust. If the threshold center stays fixed while only the transition width shrinks, the misspecification-ceiling mechanism survives.

## 2. Program-synthesis environment

Finite input domain:
```text
x ∈ {0,...,255}
```

True semantic target:
```python
def target(x):
    return (x*x + 3*x + 7) % 11
```

Fallback rule:
```python
def fallback(x):
    return (x*x + 2*x + 7) % 11
```

A fixed public suite contains 32 inputs.

The candidate pool contains 243 deterministic programs:
- 180 **regular approximate programs**: correct on a deterministic modular subset of inputs, fallback elsewhere;
- 60 **exploit programs**: explicitly return the correct output on all public tests, but use a weak branch plus fallback elsewhere;
- 3 rare **exact programs**.

Pool diagnostics:

| program_type | count | base_policy_mass | mean_visible_pass_rate | mean_exhaustive_accuracy |
| --- | --- | --- | --- | --- |
| regular | 180 | 0.9368 | 0.6905 | 0.6829 |
| exploit | 60 | 0.0625 | 1.0000 | 0.3638 |
| exact | 3 | 0.0008 | 1.0000 | 1.0000 |

Thus the exploit programs are not assigned an artificial negative label. Their reward is computed from their actual exhaustive behavior over all 256 inputs.

## 3. Trusted verifier

Each trusted observation:
1. samples a program from the current/base policy;
2. samples a uniformly random semantic input;
3. observes exact correctness on that input.

The verifier class is deliberately misspecified:
```text
v_hat(program) = beta0_hat + beta1_hat * public_test_pass_rate(program).
```

It is fitted by OLS. The confidence calculation uses an HC1 heteroskedasticity-robust slope standard error.

The population projection is
```text
beta_pop = (0.287406, 0.529324)
```
and the base-policy exhaustive semantic accuracy is
```text
0.663239.
```

## 4. Candidate optimization and self-certificate

For optimization strength `eta`,
```text
q_eta(program) ∝ p(program) exp(eta * v_hat(program)).
```

For evaluation only, the true gain is computed exhaustively:
```text
Delta_true = E_q[semantic accuracy] - E_p[semantic accuracy].
```

The self-certificate freezes the realized candidate and evaluates its proxy gain with the **same fitted verifier**. A run is counted as *confident-and-wrong* when the one-sided 95% lower bound on proxy gain is positive while the exact semantic gain is negative.

In this one-feature model the self-certificate has a useful exact simplification. For every `eta>0`,
```text
certificate fires  <=>  |beta1_hat| > 1.645 * SE(beta1_hat).
```
The confidence decision is therefore driven by confidence in the fitted proxy relation, whereas the true sign can still reverse after strong optimization because the verifier cannot represent the overfit-program distinction.

## 5. Population threshold

With infinite trusted data:
```text
maximum beneficial true gain = 0.046261
eta at maximum gain          = 10.244
population sign threshold    = 24.608329
```

The candidate initially improves semantic reward, but beyond `eta_star` further proxy optimization shifts too much mass toward public-test overfit programs.

## 6. Main budget sweep

| n | replications | beta1_mean | beta1_sd | self_certificate_fire_prob | eta_10pct_wrong | eta_50pct_wrong | eta_90pct_wrong | transition_width_10_90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200.0000 | 2200.0000 | 0.5286 | 0.1922 | 0.8518 | 16.7500 | 24.7019 | — | — |
| 1000.0000 | 1600.0000 | 0.5300 | 0.0857 | 1.0000 | 20.3816 | 24.5312 | 31.1667 | 10.7851 |
| 5000.0000 | 800.0000 | 0.5311 | 0.0380 | 1.0000 | 22.4625 | 24.5179 | 27.0909 | 4.6284 |
| 20000.0000 | 400.0000 | 0.5282 | 0.0191 | 1.0000 | 23.5417 | 24.7135 | 25.7812 | 2.2396 |

Across a 100x budget increase (`n=200` to `n=20000`), the 50% confident-wrong threshold changes by only
```text
0.1957.
```

The 10%-90% transition width scales empirically as
```text
width ∝ n^-0.525.
```

Thus the finite-sample signature is:

> **More trusted data sharpen the transition at approximately square-root rate, but do not materially move its center.**

![Confident-wrong curves](program_synthesis_confident_wrong_curves.png)

![Threshold versus budget](program_synthesis_threshold_vs_budget.png)

## 7. Selected threshold points

Confident-and-wrong probability:

| n | eta=20 | eta=24 | eta=26 | eta=30 | eta=40 |
| --- | --- | --- | --- | --- | --- |
| 200.000 | 0.251 | 0.463 | 0.560 | 0.684 | 0.845 |
| 1000.000 | 0.082 | 0.452 | 0.636 | 0.873 | 0.993 |
| 5000.000 | 0.003 | 0.386 | 0.782 | 0.996 | 1.000 |
| 20000.000 | 0.000 | 0.247 | 0.935 | 1.000 | 1.000 |

Below the limiting threshold, larger trusted datasets remove finite-sample harmful proposals.

Above the threshold, the reverse occurs: the fitted verifier stabilizes and the self-certificate becomes **more consistently confident in a semantically harmful selected distribution**.

This is the central empirical distinction from an ordinary variance-limited failure.

## 8. Robustness to the public test suite

The entire environment was regenerated with four different public-test suites, keeping the semantic target, candidate generator family, verifier class, and base-policy weighting fixed.

| public_suite_seed | population_eta_star | eta_50pct_wrong_at_n5000 | absolute_gap | population_max_true_gain |
| --- | --- | --- | --- | --- |
| 20260911.0000 | 24.6083 | 24.4722 | 0.1361 | 0.0463 |
| 20260912.0000 | 26.0772 | 26.2625 | 0.1853 | 0.0491 |
| 17.0000 | 23.1910 | 23.3750 | 0.1840 | 0.0477 |
| 99.0000 | 28.4530 | 28.5556 | 0.1025 | 0.0516 |

At `n=5000`, the empirical 50% wrong threshold follows each environment's own population semantic sign threshold.

![Public-suite robustness](program_synthesis_public_suite_robustness.png)

The numerical threshold is therefore environment-specific, as it should be. What is stable is its **insensitivity to audit budget once the environment/verifier class is fixed**.

## 9. Fresh post-selection semantic audit

As a control, after `q_eta` is fixed we independently sample fresh semantic evaluations from `q_eta` and `p` and estimate the true gain directly.

| eta | true_gain | m_per_distribution | empirical_sign_error |
| --- | --- | --- | --- |
| 25.0000 | -0.0018 | 361663.0000 | 0.0568 |
| 30.0000 | -0.0251 | 1948.0000 | 0.0536 |
| 40.0000 | -0.0676 | 275.0000 | 0.0558 |

The direct semantic audit is expensive close to the zero-gain point and rapidly becomes cheaper as the harmful effect grows.

Therefore this experiment again rejects the overly broad statement “past the threshold verification is impossible.” The supported statement is narrower and stronger:

> **the self-evaluation certificate can become confidently wrong because its verifier class cannot represent the selected failure, while fresh post-selection semantic verification can still reject the update.**

## 10. What is supported

1. The budget-independent threshold signature survives outside the Gaussian/quadratic model.
2. It also survives when misspecification arises from explicit public-test-overfit programs with exhaustive hidden semantics.
3. Trusted budget primarily reduces transition width, approximately as `n^(-1/2)`.
4. Above the limiting threshold, more same-model trusted data can increase the probability of a confident wrong certificate.
5. Fresh candidate evaluation remains effective away from the zero-gain point.

## 11. What is not yet established

- Candidate programs are generated procedurally rather than by an LLM.
- The verifier uses only public-test pass rate as its learned feature.
- Exploit programs are deliberately present in the base-policy support.
- The experiment demonstrates a mechanism and a phase signature, not its prevalence in real code agents.
- A richer verifier class that represents the exploit mechanism can move or eliminate the threshold.

## 12. Research decision

This is a second positive empirical step. The same qualitative signature now appears in:
1. a finite non-Gaussian misspecified reward model;
2. a finite program-semantics environment with explicit test-suite overfitting.

The next escalation should be a **small actual code-generation model**, not another theorem or synthetic family.

Recommended design:
- generate multiple code candidates per task;
- public tests provide the cheap proxy signal;
- stronger hidden tests provide trusted semantic reward;
- vary Best-of-N / soft selection strength independently from trusted-verifier data budget;
- plot the semantic-failure threshold against budget.

The key question remains unchanged: **does more trusted data move the optimization-strength failure threshold, or mainly sharpen the transition around a stable location?**
