# Progress Update XIX — Recursive verification lag: stale-verifier cadence experiment

**Date:** 2026-09-11  
**Status:** Completed. This update materially changes the interpretation of the preceding one-step threshold experiments.

## 1. Motivation

The previous experiments established a robust **one-step** phenomenon:

- for a fixed misspecified verifier class, the optimization-strength location at which a selected candidate becomes harmful is approximately insensitive to trusted-data budget;
- increasing the budget mainly sharpens the finite-sample transition;
- increasing verifier expressiveness can move or remove that threshold.

However, the project is about **recursive** self-improvement. A one-step threshold does not by itself imply that a recursively refreshed verifier will ever cross it.

This update therefore asks:

> If the verifier is refit as the policy changes, does recursive self-improvement still degrade, and how does the answer depend on verifier refresh cadence?

The learned autoregressive DSL code-generator distribution from Progress Updates XVII–XVIII is reused. No new theory mechanism is introduced.

## 2. Recursive setup

Initial program-policy distribution: learned autoregressive DSL generator.

Initial exhaustive semantic reward:
```text
0.571182
```

At round `t`:
1. fit a proxy verifier from trusted semantic observations;
2. form an exponential/KL update
   ```text
   q_t(program) ∝ p_t(program) exp(eta_step * v_hat_t(program))
   ```
3. use `eta_step = 3`;
4. if the certificate accepts, set `p_(t+1) = q_t`;
5. repeat.

The proxy verifier uses only public-test pass rate, so it cannot directly distinguish public-test memorization from genuine semantics.

The experiment compares:
- **sealed self-certificate:** fit once and reuse forever;
- **current-policy refresh:** refit on the current policy each round;
- **fresh guard:** generate with the proxy verifier, then use an independent post-selection semantic audit to accept/reject;
- **oracle guard:** accept iff the exact hidden semantic gain is positive.

The horizon is 14 recursive updates for the first comparison.

## 3. Population-level surprise

Before Monte Carlo simulation, the recursion was solved using the exact population projection of the proxy verifier.

### Sealed population verifier

If the verifier is fitted once on the initial policy and never refreshed, repeated updates are equivalent to increasingly strong optimization of the same misspecified proxy.

The true semantic reward eventually rises and then degrades.

### Current-policy population refresh

If the verifier is instead refit on the **current** policy every round, the story changes.

As exploit-program mass increases, the fitted slope between public-test score and hidden semantic accuracy shrinks:

```text
round 1 slope ≈ 0.760
round 4 slope ≈ 0.242
round 8 slope ≈ 0.025
round 12 slope ≈ 0.002
```

The update therefore becomes progressively weaker and the population recursion approaches a plateau near

```text
true reward ≈ 0.6543
```

without a harmful population update.

This is a direct correction to an overly broad reading of §FD:

> **A one-step misspecification threshold does not imply inevitable recursive collapse when the verifier is refreshed on the distribution the policy actually reaches.**

![Population recursion](recursive_population_comparison.png)

## 4. Strategy comparison at finite sample size

Trusted verifier budget: `n = 1000` per fit.  
Fresh guard budget: `m = 1000` semantic samples from each of `p_t` and `q_t`.  
Monte Carlo replications: 500.

| strategy | n_train | m_fresh | replications | final_mean_true_reward | final_below_baseline_fraction | ever_harmful_accepted_fraction | mean_num_updates | median_stop_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sealed_self | 1000 | 0 | 500 | 0.4797 | 1.0000 | 1.0000 | 14.0000 | 15.0000 |
| current_self | 1000 | 0 | 500 | 0.6523 | 0.0000 | 0.0340 | 3.9200 | 5.0000 |
| fresh_guard | 1000 | 1000 | 500 | 0.6100 | 0.0000 | 0.0000 | 0.9700 | 2.0000 |
| oracle | 1000 | 0 | 500 | 0.6540 | 0.0000 | 0.0000 | 5.8240 | 7.0000 |

### Sealed verifier

The sealed self-certificate is disastrous in this environment:

```text
initial true reward       = 0.571
final mean true reward    = 0.480
ever harmful accepted     = 100%
below initial at the end  = 100%
```

Because the same statistically significant proxy slope is reused, the system accepts all 14 updates and eventually optimizes past the proxy's semantic validity.

### Current-policy refresh

Refitting the same misspecified verifier every round produces a very different result:

```text
final mean true reward    ≈ 0.652
ever harmful accepted     ≈ 3.4%
below initial at the end  = 0%
```

The verifier is still misspecified, but the current-policy data expose enough of the exploit region that its proxy slope collapses before recursive semantic collapse.

### Fresh post-selection guard

The fresh guard accepted fewer updates and was conservative:

```text
mean accepted updates     ≈ 0.97
ever harmful accepted     = 0%
final mean true reward    ≈ 0.610
```

This is safe but under-utilizes available improvement at the chosen fixed fresh-audit budget.

The oracle reaches approximately `0.654`, showing that current-policy refresh nearly attains the best accessible plateau in this instance.

![Recursive strategy comparison](recursive_strategy_comparison.png)

## 5. Current-policy refresh versus trusted budget

The current-policy refresh strategy was repeated for three trusted sample budgets.

| n_train | replications | final_mean_true_reward | final_below_baseline_fraction | ever_harmful_accepted_fraction | mean_num_updates | median_stop_round |
| --- | --- | --- | --- | --- | --- | --- |
| 200.0000 | 600.0000 | 0.6419 | 0.0000 | 0.0267 | 2.3200 | 3.0000 |
| 1000.0000 | 500.0000 | 0.6524 | 0.0000 | 0.0200 | 3.9320 | 5.0000 |
| 5000.0000 | 250.0000 | 0.6539 | 0.0000 | 0.0280 | 5.4160 | 6.0000 |

Increasing `n` mainly allows the loop to take more small beneficial updates before statistical significance vanishes:

```text
mean updates:
n=200   -> 2.32
n=1000  -> 3.93
n=5000  -> 5.42
```

The final reward correspondingly approaches the population plateau.

Importantly, **none of the tested budgets produced final reward below the initial baseline**.

![Current-refresh budget comparison](recursive_current_refresh_budget.png)

## 6. The actual verification-lag experiment

The key experiment now varies the number of policy updates for which a fitted verifier is kept stale.

`refresh_interval = L` means:
- fit the verifier;
- reuse it for `L` recursive policy updates;
- then refit on the reached current policy.

### Population result

| refresh_interval | final_true_reward | minimum_true_reward | below_initial_at_end | harmful_update_count |
| --- | --- | --- | --- | --- |
| 1 | 0.6543 | 0.5712 | False | 0 |
| 2 | 0.6543 | 0.5712 | False | 1 |
| 3 | 0.6543 | 0.5712 | False | 7 |
| 4 | 0.6537 | 0.5712 | False | 11 |
| 6 | 0.5383 | 0.4576 | True | 13 |
| 8 | 0.5656 | 0.2568 | True | 11 |
| 12 | 0.2064 | 0.2064 | True | 18 |
| 24 | 0.4479 | 0.4479 | True | 21 |

The qualitative transition is sharp in this instance:

- `L = 1`: no harmful population update;
- `L = 2–4`: some local overshoot can occur, but the next refresh repairs it before baseline collapse;
- `L >= 6`: the loop can fall below its initial semantic reward;
- very long stale periods reproduce the sealed-verifier failure.

The endpoint is not monotone in `L` because a later refresh can partially recover from an earlier overshoot. The scientifically relevant variable is therefore both:
- **minimum reward reached during the run**, and
- **whether a harmful update is ever accepted**,
not only final reward.

![Population refresh-cadence phase](recursive_refresh_interval_phase.png)

## 7. Finite-sample refresh-cadence phase

With `n=1000` trusted labels per refresh and 400 Monte Carlo replications:

| refresh_interval | n | replications | mean_final_reward | mean_minimum_reward | final_below_initial_fraction | ever_below_initial_fraction | ever_harmful_update_fraction | mean_num_updates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | 1000.0000 | 400.0000 | 0.6522 | 0.5712 | 0.0000 | 0.0000 | 0.0375 | 3.9200 |
| 2.0000 | 1000.0000 | 400.0000 | 0.6530 | 0.5712 | 0.0000 | 0.0000 | 0.1725 | 4.2200 |
| 3.0000 | 1000.0000 | 400.0000 | 0.6533 | 0.5712 | 0.0000 | 0.0000 | 0.2475 | 4.4400 |
| 4.0000 | 1000.0000 | 400.0000 | 0.6496 | 0.5711 | 0.0000 | 0.0050 | 0.9175 | 13.2800 |
| 6.0000 | 1000.0000 | 400.0000 | 0.5428 | 0.4344 | 0.5500 | 0.9650 | 1.0000 | 23.2800 |
| 8.0000 | 1000.0000 | 400.0000 | 0.5185 | 0.2697 | 0.5075 | 1.0000 | 1.0000 | 23.6600 |
| 12.0000 | 1000.0000 | 400.0000 | 0.2076 | 0.2076 | 1.0000 | 1.0000 | 1.0000 | 24.0000 |
| 24.0000 | 1000.0000 | 400.0000 | 0.4486 | 0.4486 | 1.0000 | 1.0000 | 1.0000 | 24.0000 |

The empirical phase is especially clear:

```text
refresh every 1–3 rounds:
    0% of runs ever fall below the initial baseline

refresh every 4 rounds:
    0.5% ever fall below baseline

refresh every 6 rounds:
    96.5% ever fall below baseline

refresh every 8+ rounds:
    approximately 100% ever fall below baseline
```

Thus, in this controlled learned-generator system, **verification lag itself has a cadence threshold**.

![Finite-sample lag phase](recursive_refresh_interval_mc_phase.png)

## 8. Main scientific correction

This update changes the strongest empirical interpretation of the project.

The previous one-step statement

```text
more verification does not move the misspecification threshold
```

is correct only while the verifier class and the distribution on which it is fitted are held fixed.

In a recursive loop, **refreshing on the new current distribution changes the population projection itself**. That can reduce the proxy's optimism and prevent the loop from ever reaching the one-step failure threshold.

Therefore the stronger statement

> “a recursively self-improving system with a misspecified verifier inevitably crosses a budget-independent failure threshold”

is **not supported** by this experiment.

The surviving and better statement is:

> **Recursive failure depends jointly on optimizer strength and verifier staleness. A misspecified verifier can be safe when refreshed quickly enough, but the same verifier can become destructive when reused for too many optimization steps before refresh.**

This is much closer to the original concept of **recursive verification lag**.

## 9. New empirical law suggested by the results

The experiment suggests that the relevant control variable is not trusted-label count alone, but a form of **stale optimization exposure**:

```text
verification lag
≈ optimization pressure accumulated between verifier refreshes.
```

For a fixed per-step optimization strength, longer refresh intervals let the policy move farther under a verifier whose error geometry was measured on an increasingly obsolete distribution.

A future theorem should not be started yet, but if the empirical effect survives additional environments, a natural mathematical target would be a condition of the form

```text
(stale optimizer movement before refresh)
    ×
(local verifier misspecification / coverage error)
< safety margin.
```

The existing density-ratio and candidate-salient information machinery in the note may already be sufficient to formalize this; no new bespoke metric should be introduced prematurely.

## 10. Relation to the original moving-tail thesis

This result is favorable to the **recursive verification lag** framing but unfavorable to a simplistic “current-policy refresh is insufficient” headline.

In this learned DSL environment:
- current-policy refresh every round is sufficient;
- a permanently sealed verifier is not;
- intermediate refresh cadences exhibit a clear degradation transition.

Therefore the empirical question for real models should be reframed from

> “Does current-policy verification work?”

to

> **“How much policy optimization can occur before the verifier must be refreshed?”**

That is a sharper operational question and naturally produces a verification budget **per unit of policy movement**, not merely per training round.

## 11. Research decision

This result is important enough to change the next step.

Do not immediately pursue a new theorem, and do not claim the one-step threshold as the whole recursive story.

The next empirical experiment should vary **two independent controls**:
1. per-update optimization strength;
2. verifier refresh interval.

The primary phase diagram should plot whether semantic reward ever falls below baseline.

If the boundary approximately collapses under a product such as

```text
optimization strength × refresh interval
```

or, better, under an empirically measured policy-shift quantity such as cumulative log density ratio / candidate-salient shift, then the project regains a genuinely recursive quantitative law.

This is now a higher-value empirical target than another one-step budget sweep.

## 12. Current assessment after this correction

The result is scientifically positive even though it falsifies an overly broad interpretation.

It shows that:
- the one-step §FD mechanism is real;
- fresh verification is not always required every round;
- current-policy refresh can self-correct;
- **staleness of the verifier is the variable that turns misspecification into recursive damage**.

That is a more nuanced and more defensible version of the project's original thesis.
