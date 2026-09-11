# Progress Update XV — §FD experiment beyond the Gaussian simulation

**Date:** 2026-09-11  
**Status:** Controlled experiment completed. The central §FD signature survives in a finite, non-Gaussian, bounded-label setting with an actually fitted misspecified verifier.

## 1. Question

The v15 note identified one distinctive falsifiable prediction worth testing before proving anything else:

> As optimization strength increases, there is a true-gain sign threshold that is set by verifier misspecification rather than audit budget. Increasing the audit budget should sharpen confidence around the threshold, not move the threshold itself; above it, the self-evaluation certificate becomes confidently wrong.

This update tests that claim outside the original Gaussian quadratic simulation.

## 2. Experimental setup

Outcome space is finite:
- 401 outcomes, `y in [-4,4]`.
- Current/base policy is an **asymmetric discrete-Laplace** distribution
  `p(y) ∝ exp(-|y| + 0.15 y)`, hence not Gaussian.

The main true reward is bounded and nonlinear:
```text
r_A(y) = tanh(y - 0.30 y^2 - 0.01 y^4).
```
It is not representable by the verifier class.

Trusted labels are stochastic and bounded:
```text
Y ∈ {-1,+1},
P(Y=+1 | y) = (1+r_A(y))/2.
```

The verifier is actually fitted from `n` current-policy trusted samples by OLS:
```text
v_hat(y) = beta0_hat + beta1_hat y.
```
HC1 robust standard errors are used, so the certificate is not relying on homoscedastic Gaussian noise.

A candidate is produced by the same exponential/KL tilt used throughout the note:
```text
q_eta(y) ∝ p(y) exp(eta * v_hat(y)).
```

For evaluation only, the exact synthetic true gain
`Delta = E_q[r_A] - E_p[r_A]`
is computed by summing over all 401 outcomes. The verifier never receives this quantity.

The sealed self-certificate freezes the realized candidate and forms a one-sided 95% lower bound on the **proxy** gain using the fitted linear model. A run is counted as *confident-and-wrong* when this lower bound is positive while the exact true gain is negative.

Monte Carlo replications:
- n=50: 3000
- n=200: 3000
- n=1000: 2000
- n=5000: 1000
- n=20000: 500

Random seed: 20260911.

## 3. Population threshold

The population linear projection of the nonlinear reward under `p` is

```text
beta_pop = (-0.124865, 0.292476).
```

Using this limiting fitted verifier, the exact true gain changes sign at

```text
eta_star = 3.497506.
```

This is the budget-independent reference threshold.

## 4. Main result — the threshold center does not move

| n | replications | slope_mean | slope_sd | eta_10pct_wrong | eta_50pct_wrong | eta_90pct_wrong | transition_width_10_90 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50.0000 | 3000.0000 | 0.3040 | 0.1237 | 2.2228 | 3.4458 | — | — |
| 200.0000 | 3000.0000 | 0.2955 | 0.0587 | 2.7543 | 3.4565 | 4.6500 | 1.8957 |
| 1000.0000 | 2000.0000 | 0.2930 | 0.0257 | 3.1280 | 3.5029 | 3.9103 | 0.7823 |
| 5000.0000 | 1000.0000 | 0.2931 | 0.0115 | 3.3192 | 3.4901 | 3.6750 | 0.3558 |
| 20000.0000 | 500.0000 | 0.2925 | 0.0058 | 3.4125 | 3.4968 | 3.5867 | 0.1742 |

For all `n >= 200`, the empirical 50% confident-wrong crossing remains near `eta ≈ 3.498`. Its total range across `n=200,...,20000` is only
`0.0464`,
about `1.33%` of the threshold.

What changes with budget is the **width**, not the center. Regressing the 10%-90% transition width against `n` gives

```text
transition width ∝ n^-0.515.
```

That is essentially the expected `n^(-1/2)` finite-sample sharpening.

![Confident-wrong curves](fd_confident_wrong_curves.png)

![Threshold vs budget](fd_threshold_vs_budget.png)

## 5. Budget sweep

Probability of a confident wrong certificate:

| n | eta=3 | eta=3.5 | eta=4 | eta=5 | eta=6 |
| --- | --- | --- | --- | --- | --- |
| 50.000 | 0.368 | 0.513 | 0.628 | 0.762 | 0.806 |
| 200.000 | 0.224 | 0.524 | 0.737 | 0.941 | 0.988 |
| 1000.000 | 0.035 | 0.496 | 0.936 | 1.000 | 1.000 |
| 5000.000 | 0.000 | 0.529 | 1.000 | 1.000 | 1.000 |
| 20000.000 | 0.000 | 0.518 | 1.000 | 1.000 | 1.000 |

Two opposite effects occur as `n` increases:

- Below the population threshold (`eta=3.0`), finite-sample harmful proposals disappear and the wrong-certificate probability goes to zero.
- Above the threshold (`eta>=4`), the fitted verifier stabilizes and the certificate becomes **more reliably wrong**, approaching probability one.

Thus more verification improves estimation of the misspecified verifier but does not repair the objective misspecification.

## 6. Robustness to a qualitatively different misspecification

To avoid making the result depend on a polynomial/quadratic-shaped reward, the experiment was repeated with a hidden nonlinear cliff:

```text
r_B(y) = tanh(0.8 y - 0.8 max(y-1,0)^2).
```

The population fitted slope is `0.264812` and the exact sign threshold is

```text
eta_star_B = 3.422735.
```

| n | eta_10pct_wrong | eta_50pct_wrong | eta_90pct_wrong | transition_width_10_90 |
| --- | --- | --- | --- | --- |
| 200.0000 | 2.6080 | 3.4181 | 4.7286 | 2.1206 |
| 1000.0000 | 3.0269 | 3.4064 | 3.9250 | 0.8981 |
| 5000.0000 | 3.2286 | 3.4177 | 3.6260 | 0.3975 |
| 20000.0000 | 3.3205 | 3.4216 | 3.5262 | 0.2058 |

The 50% crossing stays within a range of only
`0.0152`
across a 100x audit-budget increase. The transition-width exponent is

```text
transition width ∝ n^-0.507.
```

Again the center is stable and the transition sharpens at approximately square-root rate.

![Hidden-cliff robustness](fd_hidden_cliff_robustness.png)

## 7. Fresh post-selection audit control

A separate model-free audit draws new labels from both `q_eta` and `p` after the candidate is fixed. Using the usual variance-based sample size for a one-sided 95% sign test gives:

| eta | true_gain | m_per_distribution | sign_error_rate |
| --- | --- | --- | --- |
| 4.0000 | -0.0755 | 941.0000 | 0.0507 |
| 5.0000 | -0.2410 | 89.0000 | 0.0497 |
| 6.0000 | -0.3971 | 31.0000 | 0.0687 |

The required fresh sample count falls rapidly after moving away from the zero-gain point. This reproduces the corrected v13/v15 interpretation: **past the threshold, harmful updates are not intrinsically hard to verify; they are hard only for the self-evaluation certificate that cannot represent the failure.**

## 8. Interpretation

### Supported by this experiment

1. **The §FD qualitative signature survives beyond the Gaussian simulation.**
   A learned misspecified verifier on a finite non-Gaussian policy distribution exhibits a stable optimization-strength threshold.

2. **Audit budget controls sharpness, not threshold location.**
   The 50% threshold converges to the population value, while its finite-sample width shrinks approximately as `n^(-1/2)`.

3. **More same-model verification can increase confidence in the wrong answer.**
   Above the threshold, larger `n` makes the fitted verifier more stable and drives the confident-wrong probability toward one.

4. **Fresh candidate verification behaves differently.**
   Away from the zero-gain point, direct post-selection sign testing becomes cheaper as the harmful effect grows.

### Not established

- This is still a controlled synthetic finite-domain experiment, not an LLM or program-synthesis result.
- It does not prove that every misspecified learned verifier has a single sharp threshold.
- “Perfectly sharp” should be read as an **asymptotic** statement. At finite `n`, the transition has nonzero width, empirically about `n^(-1/2)` here.
- The threshold is budget-independent only after fixing the learning procedure, verifier class, base distribution, and optimization rule. Changing those objects can move it.

## 9. Consequence for the paper

The experiment is a positive result and justifies moving to the next empirical level. The strongest plot is not “reward hacking increases with optimization.” It is:

> **Across audit budgets spanning orders of magnitude, the failure curves cross at essentially the same optimization strength; added audit data only make the transition sharper.**

That is substantially more discriminating than a generic overoptimization curve.

## 10. Next action

Do **not** return to theorem expansion yet.

The next experiment should instantiate the same two-axis sweep in a finite-domain program-synthesis task:
- x-axis: optimization/selection strength;
- curves: trusted verification budget;
- hidden true reward: exhaustive semantic correctness;
- fitted proxy verifier: limited visible tests or learned pass predictor.

The target falsifier is the same: determine whether the failure threshold moves with trusted-data budget or merely sharpens around a stable location.
