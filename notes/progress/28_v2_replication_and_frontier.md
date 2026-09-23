# Progress XXVIII — Stronger-generator transfer and an identified safety frontier

## Decision first

On an eight-task suite frozen before generation, the 1.5B model again supplies
correct expressions (seven tasks), but the direction of the early-refresh
failure contrast **does not replicate** the earlier 1.5B bank. A sharp
bounded-reward identification interval supports a finite-bank zero-failure
controller, yet it gives up a large fraction of the exploratory fixed-policy
gain. The result is a *safety/progress tradeoff*, not a general adaptive
refresh law. No standard-program benchmark, weight update, or new statistical
theorem is claimed. A cross-task causal mechanism is not identified.

## Locked hypothesis and generation

The task file `src/transfer_code_tasks_v2.py` and
`docs/transfer_v2_protocol.md` were placed in the draft PR before generation.
The exact model revision, 8×16 samples, seed 20260926, temperature .8, top-p
.95, 64 token cap, public/hidden partition, designs, and primary contrasts
were fixed. The completed bank has 128 rows and SHA256
`71a4cecd9d9414deb2adc28b2afd5d54d5ba6dbfd036f1ca50b1364a6147c935`;
the manifest hash matches and independent deterministic rescoring passed.
There are 21 fully correct occurrences across seven of eight tasks; the
asymmetric_hinge task has zero. Of 128 completions, 94 parse as valid
expressions, 30 pass every public test yet fail hidden tests, and none hit the
64-token cap. The preregistered 4/8 support feasibility threshold passes.

These manually designed tasks share the restricted grammar with previous
experiments and were authored after earlier findings. They are not an external
benchmark or a clean-room holdout. The different task specification is a
within-family prospective replication, not an independent model family.

## Hypothesis: early refresh decreases failure at fixed paid cost

The 32-draw experiment completed 4,320 configurations; the six-source experiment
completed 4,800. All failures from the initial step and all eight tasks remain.
Task-cluster 95% intervals below are descriptive, based on eight task clusters
without multiplicity adjustment. Differences are early minus uniform.

| Setting | Failure difference | Gain difference | Prior observed 1.5B bank failure difference |
| --- | ---: | ---: | ---: |
| 32 paid draws | +0.417 pp [−0.625, +1.875] | +0.00747 [−0.00455, +0.02239] | −0.833 pp |
| Six identical streamed source labels | +1.875 pp [−2.292, +6.464] | +0.02052 [−0.00331, +0.04823] | −1.042 pp |

The frozen desired-sign criterion fails in both arms, even though gain point
estimates remain positive. These intervals cannot establish that early refresh
is worse in the underlying task population. At 32 draws, most of the positive
failure difference comes from two_landmark_distance (+5 pp within task),
partially offset by asymmetric_hinge (−1.667 pp); six tasks have zero difference.
Under identical streamed labels, task differences range from −6.667 to +15 pp.
The result directly weakens a uniform early-is-safer narrative. Policy-driven
source acquisition is a secondary comparison with a changed information set.

## Hypothesis: a frozen scalar predicts failure across suites

Previously locked development thresholds were evaluated without refitting on
the v2 bank. Mean row-level balanced accuracy across optimizer/representation
strata: KL .556, max log-density ratio .605, restricted feature geometry .609,
geometry over squared proxy margin .525. Eta-times-age is soft-only (.515).
These summaries do not establish that geometry is best: the top difference is
.004 and only eight tasks were authored. The prior 1.5B bank had just two
identified strata for most metrics and a different ranking. Balanced accuracy
is not a prospective lead-time guarantee; no stable scalar boundary emerges.

## Exact identification and a certified intervention

`notes/theory/identified_refresh_frontier.md` proves a sharp deterministic
identification interval for policy gain from exact audited-source labels and
independent [0,1] reward bounds on unaudited sources. If the interval straddles
zero, opposite true-gain signs are compatible with the same observations.
This proposition is elementary robust identification, not an Oral-level new
theorem. It exposes a source-relevant uncertainty width rather than a scalar
shift law. `src/certified_refresh.py` audits high-impact unaudited sources
before a proposed update and abstains at the six-source cap unless nonnegative
gain versus the initial policy is certified. Its zero-failure property follows
from the proposition, **conditional on** frozen finite support, exact shared
source rewards, and [0,1] bounds; it does not rely on measured zero failures.

Exploratory diagnostics on v2 (160 matched optimizer/representation/audit-seed
settings, 12 rounds) found zero failures, mean final gain .1014, mean 4.75
distinct labels, and 49.1% accepted updates. The same six-source stream
schedules in those settings had mean gains .2988 (early), .2784 (uniform),
.2438 (late), with failure rates 15.625%, 12.5%, and 10.625%. Paired gain
differences certified minus early = −.1974 [−.2937, −.1022], minus uniform
= −.1771 [−.2708, −.0932] (task bootstrap). The v1 1.5B bank shows the
same pattern: certified zero failures, gain .1880 versus uniform .3656, mean
4.79 versus 6 labels. The controller was developed using that observed bank
while v2 generation was underway; its v2 evaluation is a secondary exploratory
check, **not** part of the original fixed primary v2 protocol. Audited source
identities after the shared initial two differ, so this comparison is not an
isolated refresh-timing effect. Safety can be achieved by abstaining; the
large lost gain is a substantive negative result.

The retrospective identification diagnostic is separate from the controller:
after each paid audit and accepted update, early schedules are certified
nonnegative in 69.4% of v2 trajectory states, uniform in 54.7%, late in 30.1%;
these fractions are influenced by the number and timing of acquired labels.
It is **not** a prospective warning score or a causal estimate. All actual
gains fell within the computed sharp intervals; the result follows by design
and was checked for implementation integrity.

## Next decision

Do not claim a universal cadence or a solved safety/progress frontier.
The highest-value follow-up is a stronger useful-progress certificate or a
source acquisition policy with a declared performance objective, evaluated on
a genuinely external code task collection in an isolated execution environment.
Before seeking further positive results, preserve the failed early-replication
contrast, the zero-support task, and the certified controller's gain loss.
On the user's M1 Pro/32GB, 1.5B CPU inference is feasible for these 128-sample
banks (this workspace completed v2 generation in 394 s); standard Python
evaluation needs container isolation and pinned tests as defined in
`docs/standard_code_validation_gate.md`.
