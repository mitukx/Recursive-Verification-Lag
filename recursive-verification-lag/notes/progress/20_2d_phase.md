# Progress Update XX — 2D recursive verification-lag phase law

**Date:** 2026-09-11  
**Status:** Completed. A two-dimensional phase diagram over optimization strength and verifier refresh interval reveals an approximately one-dimensional stale-exposure law. A simple adaptive KL refresh heuristic was also tested and partially falsified.

## 1. Research question

Progress Update XIX showed that a misspecified verifier can be safe when refreshed every round yet destructive when reused for many recursive policy updates.

The next question is quantitative:

> How does the recursive failure boundary depend jointly on per-step optimization strength `eta` and verifier refresh interval `L`?

A natural first hypothesis is that the relevant quantity is not `eta` or `L` separately, but their accumulated stale exposure.

## 2. Exact stale-block identity

Suppose a verifier score `v(y)` is frozen for `L` consecutive exponential/KL policy updates:

```text
p_(k+1)(y) ∝ p_k(y) exp(eta v(y)).
```

Then induction gives the exact identity

\[
p_L(y)
=
\frac{p_0(y)\exp(L\eta v(y))}
     {\mathbb E_{p_0}[\exp(L\eta v)]}.
\]

Therefore, **within a stale-verifier block**, `L` updates of strength `eta` are exactly equivalent to one update of effective strength

\[
\boxed{\eta_{\rm eff}=L\eta.}
\]

This identity is elementary and should not be claimed as a novel theorem. Its value is that it gives a principled reason to test `eta × L` as the first collapse variable.

## 3. Population 2D phase diagram

The learned autoregressive DSL generator from Updates XVII–XIX is used again.

Grid:
```text
eta ∈ {0.50, 0.75, ..., 6.00}
L   ∈ {1, 2, ..., 16}
horizon T = 24
```

At every refresh:
1. fit the exact population linear projection of hidden semantic reward onto public-test pass rate under the current policy;
2. freeze that verifier for `L` policy updates;
3. refresh again.

Failure is defined as the trajectory **ever falling below the initial semantic reward**.

![Population phase diagram](recursive_eta_refresh_population_heatmap.png)

For every `eta` at which failure occurs within the tested `L` range, the smallest failing refresh interval is:

| eta_step | critical_refresh_interval | critical_etaL |
| --- | --- | --- |
| 1.50 | 11.00 | 16.50 |
| 1.75 | 10.00 | 17.50 |
| 2.00 | 8.00 | 16.00 |
| 2.25 | 8.00 | 18.00 |
| 2.50 | 7.00 | 17.50 |
| 2.75 | 6.00 | 16.50 |
| 3.00 | 6.00 | 18.00 |
| 3.25 | 5.00 | 16.25 |
| 3.50 | 5.00 | 17.50 |
| 3.75 | 5.00 | 18.75 |
| 4.00 | 4.00 | 16.00 |
| 4.25 | 4.00 | 17.00 |
| 4.50 | 4.00 | 18.00 |
| 4.75 | 4.00 | 19.00 |
| 5.00 | 4.00 | 20.00 |
| 5.25 | 4.00 | 21.00 |
| 5.50 | 3.00 | 16.50 |
| 5.75 | 3.00 | 17.25 |
| 6.00 | 3.00 | 18.00 |

The critical products have:

```text
median critical eta*L = 17.50
IQR                    = [16.50, 18.00]
```

Thus, over a 4x range of per-step optimization strength, the first baseline-collapse boundary is concentrated around

\[
\boxed{\eta L \approx 17\text{--}18}
\]

in this environment.

A single threshold on `eta × L` classifies the entire population grid with accuracy

```text
98.64%
```

using an optimal cutoff near

```text
eta × L = 15.875.
```

![Population etaL collapse](recursive_etaL_population_collapse.png)

## 4. Is `eta L` the best shift variable?

Three run-level scalar summaries were compared:

| scalar | best_threshold | classification_accuracy |
| --- | --- | --- |
| eta_times_L | 15.8750 | 0.9864 |
| max_score_span_exposure | 10.1211 | 0.9918 |
| max_block_KL | 1.9061 | 1.0000 |

`max_block_KL` perfectly classifies the population runs in this grid, while `eta × L` already reaches 98.6%.

However, the perfect KL result is **post-hoc**: it uses the maximum KL attained anywhere in the realized trajectory, including after the system may already have entered the bad regime. It therefore should not be presented as a predictive theorem.

To remove this leakage, block-level records were analyzed separately.

## 5. Block-level predictive diagnostics

Each interval between verifier refreshes is treated as one stale block.

For every block we record quantities knowable from the policy path:
- `eta × block length`;
- score-span exposure under the frozen verifier;
- KL divergence from the policy at the last verifier refresh to the block endpoint.

The target is whether the block:
1. contains any harmful update;
2. causes the trajectory to cross below the initial baseline.

| target | scalar | best_threshold | accuracy |
| --- | --- | --- | --- |
| contains_harmful_update | etaL_block | 8.6250 | 0.9069 |
| contains_harmful_update | planned_score_span_exposure | 5.9636 | 0.8066 |
| contains_harmful_update | endpoint_KL_from_refresh_policy | 0.5691 | 0.8302 |
| causes_below_initial | etaL_block | 16.3750 | 0.9366 |
| causes_below_initial | planned_score_span_exposure | 13.9721 | 0.9473 |
| causes_below_initial | endpoint_KL_from_refresh_policy | 2.0713 | 0.9453 |

For baseline-crossing blocks:
- `eta L` gives about **93.7%** classification accuracy;
- score-span exposure gives about **94.7%**;
- endpoint KL gives about **94.5%**.

Therefore the more portable policy-shift quantities slightly improve on raw `eta L`, but there is **not** yet one perfect pre-failure scalar law at block level.

The useful conclusion is narrower:

> `eta L` is an excellent first-order stale-exposure coordinate because of the exact frozen-update identity; actual policy-shift quantities such as KL may refine the boundary when score scales or verifier classes change.

## 6. Finite-sample 2D phase

The same two-dimensional sweep was repeated with:

```text
trusted labels per refresh n = 1000
150 Monte Carlo runs per cell
eta ∈ {2,3,4,5,6}
L ∈ {2,3,4,5,6,8,10,12}
```

The verifier is refit from finite trusted semantic observations. If its slope is no longer significantly nonzero, the recursive optimizer halts.

![Finite-sample phase diagram](recursive_eta_refresh_finite_heatmap.png)

The first `L` at which at least 50% of runs fall below baseline is:

| eta_step | first_L_with_failure_prob_ge_0.5 | etaL_at_boundary | failure_probability |
| --- | --- | --- | --- |
| 2.000 | 10.000 | 20.000 | 0.967 |
| 3.000 | 6.000 | 18.000 | 0.973 |
| 4.000 | 4.000 | 16.000 | 0.893 |
| 5.000 | 3.000 | 15.000 | 0.660 |
| 6.000 | 3.000 | 18.000 | 0.993 |

The boundary products are approximately

```text
20, 18, 16, 15, 18
```

for `eta = 2,3,4,5,6`.

The median remains near

\[
\boxed{\eta L \approx 18}.
\]

Across all finite-sample cells:
- rank correlation between `eta L` and failure probability is **0.914**;
- a single threshold on `eta L` classifies whether failure probability exceeds 50% with **95.0%** accuracy.

![Finite etaL collapse](recursive_etaL_finite_collapse.png)

This is strong evidence that the population phase is not only an infinite-data artifact.

## 7. Important distinction: harmful update vs catastrophic drift

A lower stale exposure can already produce an occasional locally harmful update without pushing the whole trajectory below its starting reward.

For example, in the finite experiment many cells with small `L` have:

```text
P(ever harmful update) > 0
```

while

```text
P(ever below initial reward) = 0.
```

The next verifier refresh often repairs the local overshoot.

Therefore recursive safety has at least two natural criteria:
1. **strict monotonicity:** never accept a negative-gain update;
2. **trajectory safety:** never fall below a deployment baseline.

The phase transition is substantially sharper for the second criterion in this environment.

The paper should state explicitly which criterion is being evaluated.

## 8. Adaptive KL-triggered refresh experiment

A natural implementation idea is to refresh the verifier when policy drift from the last refresh exceeds a KL cap rather than on a fixed round schedule.

At population level this looked promising.

Across `eta ∈ [1,6]`:
- KL cap `0.75` preserved baseline safety in all tested settings;
- it required only **3.36 verifier fits on average over 24 rounds**;
- fixed `L=2` also preserved all settings but requires 12 scheduled fits if no early stopping is allowed.

This suggested a potentially large adaptive savings.

However, finite-sample evaluation changed the conclusion.

With `n=1000`, significance-based halting already causes fixed-cadence methods to stop refitting once useful signal disappears.

The finite-sample summary is:

| policy_label | mean_failure_fraction | mean_harmful_fraction | mean_refresh_count | mean_trusted_labels | mean_final_reward | mean_safe_probability |
| --- | --- | --- | --- | --- | --- | --- |
| KL cap 0.50 | 0.0000 | 0.7455 | 4.3018 | 4301.8182 | 0.6493 | 1.0000 |
| KL cap 0.75 | 0.0655 | 0.4236 | 3.2064 | 3206.3636 | 0.6506 | 0.9345 |
| KL cap 1.00 | 0.1255 | 0.7264 | 3.9100 | 3910.0000 | 0.6453 | 0.8745 |
| fixed_L1 | 0.0000 | 0.0527 | 4.4582 | 4458.1818 | 0.6526 | 1.0000 |
| fixed_L2 | 0.0000 | 0.3300 | 3.4036 | 3403.6364 | 0.6526 | 1.0000 |
| fixed_L3 | 0.3227 | 0.6691 | 4.9591 | 4959.0909 | 0.6250 | 0.6773 |

The safest KL rule (`cap=0.50`) and fixed `L=1`/`L=2` all achieve zero baseline-collapse probability over the tested `eta` values, but:

```text
KL cap 0.50 : ~4302 trusted labels
fixed L=2   : ~3404 trusted labels
fixed L=1   : ~4458 trusted labels
```

Thus **the population efficiency advantage of KL-triggered refresh does not survive this finite-sample protocol**. Fixed `L=2` is actually better here.

![Finite-sample KL-trigger Pareto](recursive_kl_trigger_finite_sample_pareto_v2.png)

This is a useful negative result. KL-triggered refresh should not be promoted as a contribution on the basis of the current experiment.

## 9. Main scientific result of this update

The strongest result is not the KL heuristic. It is the two-dimensional phase collapse:

\[
\boxed{
\text{recursive failure is largely controlled by stale optimization exposure}
}
\]

with the simplest coordinate

\[
\boxed{\eta_{\rm stale}=\eta L.}
\]

In this learned-generator environment, the baseline-collapse boundary lies around

\[
\boxed{\eta L \simeq 15\text{--}20}
\]

both at population level and with finite trusted data.

This directly operationalizes the phrase **recursive verification lag**:

> A verifier need not be refreshed every round. It must be refreshed before the optimizer accumulates too much movement under a stale proxy.

## 10. Relation to the earlier one-step threshold

The earlier learned-generator experiment found a one-step harmful-gain threshold near

```text
eta_star ≈ 22.54
```

from the initial policy.

The recursive collapse boundary appears at a somewhat smaller stale exposure (`eta L ≈ 17–18`).

There is no contradiction.

After one or more refreshes:
- the current policy has changed;
- the population projection of the misspecified verifier has changed;
- the local semantic margin has changed.

Therefore the safe one-step threshold is state-dependent. Repeated moderate stale blocks can reach a globally bad trajectory before the initial-policy one-step threshold is ever used directly.

This is exactly why the recursive problem is not reducible to a single static `eta_max`.

## 11. What is now supported

The empirical evidence supports:

1. **Fast refresh can prevent recursive collapse even with the same misspecified verifier class.**
2. **Slow refresh can make the same class destructive.**
3. **The boundary depends jointly on optimization strength and refresh cadence.**
4. **Within frozen-verifier blocks, the natural accumulation law is exactly `eta L`.**
5. **Across the tested recursive system, baseline collapse approximately follows an `eta L` phase boundary.**
6. **Finite trusted-data noise broadens the phase but does not erase it.**

## 12. What is not yet supported

- The numerical critical value `~17–18` is environment-specific.
- `eta L` is not invariant to arbitrary rescaling of verifier scores; policy-shift quantities such as KL are better candidates for cross-system comparison.
- Block-level KL does not perfectly predict failure.
- The KL-triggered adaptive algorithm did not show a robust efficiency advantage under finite-sample significance stopping.
- No pretrained LLM experiment has yet been run.
- No general theorem currently proves a universal stale-exposure threshold.

## 13. Research direction after this result

This is the first experiment in the sequence that produces a genuinely **recursive quantitative law**, rather than only a one-step misspecification effect.

The next high-value question is therefore not another generic theorem. It is invariance:

> Does the phase boundary collapse better under an actual policy-shift quantity than under raw `eta L` when the verifier score scale, representation quality, or optimizer family changes?

The next experiment should vary at least one of:
- verifier representation quality;
- score rescaling/calibration;
- Best-of-N instead of exponential tilt.

Then compare candidate coordinates:

```text
eta L
cumulative KL from last refresh
max log density ratio from last refresh
candidate-salient mass shift
```

If one shift coordinate collapses the phase across these changes, that quantity becomes the natural candidate for the paper's recursive verification-lag law.

If none does, the result should remain an environment-specific phase phenomenon rather than be elevated to a general law.
