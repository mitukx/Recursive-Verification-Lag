# Progress 31 — Safe-region projection tests the directional bottleneck

## Hypothesis and controlled comparison

The previous boundary result only blocked interpolation along the optimizer's
given direction. Hypothesis: with the *same* paid source labels, the nearest
fully safe distribution to that proposal may still move and gain reward.
After the source budget is exhausted and the candidate fails the sharp box
certificate, minimize L1 distance to it over **all** distributions satisfying
the baseline certificate. Keep the original two observed 1.5B expression
banks, tasks, 160 matched optimizer/representation/seed settings each, 12
rounds, identical initial two-source streams, six-source cap, and greedy
source acquisition. The fixed-proposal interpolation comparison was already
inspected, so this is explicitly exploratory.

## Results

| Bank | Mean gain: abstain | Mean gain: projection | Paired delta | Descriptive task bootstrap 95% | Mean paid sources | Projected states / accepted | Failure states |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| v1 (8 tasks) | .188034 | .385993 | +.197959 | [.097183,.289238] | 4.7875 | 460 / 451 | 0 / 1920 |
| v2 (8 tasks) | .101377 | .225567 | +.124190 | [.063553,.188249] | 4.7500 | 831 / 776 | 0 / 1920 |

These are paired over settings but only eight task clusters per bank; the
intervals describe task resampling within these *already inspected* suites.
Results are on finite expression banks, not on an isolated standard program
benchmark. Projection leaves the average paid-source count equal to abstention
in these runs, but later acquisition choices can change with the new policy.
Numerically the smallest computed certificate is within 3.5e-16 below zero,
and no true gain falls below the initial policy. The LP uses 1e-10 primal/dual
feasibility tolerances and checks its output; the guarantee is exact
mathematics conditional on its assumptions, while the experiment is subject
to numerical precision. Projected policies can still lose reward relative to
the **previous** accepted policy.

At the final round, v1 mean mass on audited source classes is .485 under
abstention versus .670 under projection (initial mass on the eventually
audited classes .341); the analogous v2 values are .404 versus .500 (initial
.321). Mean effective source supports \(1/\sum_g p_g^2\) fall from 10.46 to
5.20 (v1) and 12.67 to 8.27 (v2). Thus some of the finite-bank gain comes
with concentration on paid source identities, a serious transfer limitation.

## Interpretation, falsifiability and decision

The interpolation obstruction was largely **directional** on these suites:
full feasible-set movement recovers substantial finite-bank reward at an
unchanged label budget. This reframes the actionable question as whether
verification geometry allows any useful redistribution of policy mass, not
merely whether the raw optimizer trajectory crosses a scalar threshold.
The exact LP theorem in `notes/theory/safe_policy_projection.md` is an
elementary identification/optimization result and does not establish a new
Oral-grade general law. The projection may favor already audited programs,
which could fail completely when the bank changes or new programs are drawn.

Run the locked MBPP+ standard program test after isolated scoring with both
the frozen proposal and the projected controller; compare budget-matched
fixed cadence and concentration on audited source identities. Separately,
formulate and validate a structural residual model **only if** evidence
shows finite-bank source redistributions transfer across independently drawn
candidate banks. A negative transfer would narrow the paper's claim rather
than be hidden.

Reproduce: `python -m src.certified_refresh DATA --output OUT --fallback
project` for each bank; then `python -m src.analyze_certified_projection
--output results/certified_projection_comparison.csv` using the documented
paths. All 41 unit tests pass.
