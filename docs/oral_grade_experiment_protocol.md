# Oral-grade validation protocol

This document fixes the next experiment before looking at real-model outcomes. The goal is not to maximize a benchmark score; it is to test whether recursive verification lag survives a substantially more realistic generator and whether an observable refresh law predicts failure across optimizer families.

## Primary scientific question

Does semantic degradation occur when optimization moves a code-model policy into verifier-error directions faster than trusted verification is refreshed, and can one observable verification-relevant movement coordinate predict that degradation across optimization rules?

## Candidate generation

Use at least two pretrained code-model families and at least two model sizes when compute permits. Generation is separated from recursive selection: for each task, sample a fixed candidate bank once and store model log-probabilities plus source text. This prevents later sweeps from silently changing the support.

Minimum target:

- 100 finite-domain Python tasks;
- >=128 candidates per task;
- temperature/top-p fixed and recorded;
- exact prompt, model revision, seed and decoding parameters archived;
- candidate bank frozen before any RVL sweep.

A cheap first pass may use one small code model. Cross-family claims must wait for the full design.

## Verification hierarchy

Each candidate receives two kinds of evaluation.

**Cheap/proxy information (allowed to drive optimization)**

- public-test pass rate;
- syntactic validity;
- execution success;
- code length / runtime;
- optionally a learned verifier trained only on trusted samples exposed at refresh times.

**Trusted information (not available to the optimizer between refreshes)**

- hidden tests disjoint from public tests;
- exhaustive enumeration when the task domain is finite;
- deterministic semantic score whenever possible.

The public and hidden suites must be generated independently enough that memorizing public tests is a real failure mode rather than a trivial duplicate of the trusted target.

## Recursive loop

At refresh time `s`:

1. sample a trusted audit from the current policy;
2. fit the verifier using only the audit transcript available at `s`;
3. freeze the verifier for the stale block;
4. optimize the candidate distribution for `L` updates;
5. record proxy reward, trusted reward, policy movement and verifier diagnostics after each update;
6. refresh according to either a fixed schedule or an observable adaptive controller.

The trusted score may be computed offline for analysis, but it must never enter an adaptive refresh decision except in oracle controls explicitly labeled as such.

## Independent experimental factors

Sweep independently:

- per-step optimization pressure;
- verifier refresh interval;
- trusted audit budget per refresh;
- verifier representation richness;
- optimizer family.

Required optimizer families:

1. soft/exponential reweighting;
2. Best-of-N / hard selection.

A policy-gradient or GRPO-like update is an optional third family if the setup supports it without changing the task semantics.

## Primary outcome

For each run, define failure as trusted semantic reward falling below its initial-policy value at any time.

Primary phase diagram:

`optimization pressure x refresh interval -> P(ever below initial)`.

Secondary outcomes:

- minimum trusted reward;
- first failure round;
- proxy/trusted reward gap;
- trusted labels consumed;
- recovery after refresh.

## Candidate collapse coordinates

Evaluate, without retroactively redefining them:

- cumulative optimizer strength (`eta * L` where meaningful);
- cumulative KL from last refresh;
- maximum log density ratio from last refresh;
- candidate-salient mass shift;
- restricted verification geometry estimated from verifier features.

A candidate scalar law is considered supported only if it both:

1. separates safe and failing runs in held-out tasks/models; and
2. yields a stable boundary across per-step optimization strengths and at least two optimizer families.

Raw `eta * L` is not considered invariant because score rescaling changes it.

## Adaptive-refresh intervention

The central intervention compares:

- fixed refresh every `k` rounds;
- scalar movement-triggered refresh;
- verification-geometry / margin-triggered refresh;
- oracle refresh using trusted gain (upper-bound control only).

Success criterion: at matched trusted-label cost, an observable adaptive controller materially reduces failure probability relative to the best fixed cadence selected on training tasks, and transfers to held-out tasks/model family.

This is more important than merely re-observing a phase transition: it converts RVL from a descriptive phenomenon into a control rule.

## Falsification criteria

The central RVL interpretation should be weakened if any of the following persist after adequate power checks:

- no degradation appears for any stale block despite strong proxy optimization;
- degradation is explained by elapsed rounds but not policy movement;
- a proposed movement coordinate only works for one score scale or one optimizer;
- richer verifier representation does not alter the boundary while sample count does;
- adaptive movement-triggered refresh provides no advantage over fixed cadence at matched audit cost.

If no scalar coordinate transfers across verifier classes, retain the weaker conclusion that verification-relevant geometry is inherently class-dependent rather than forcing a universal scalar law.

## Statistical protocol

- freeze candidate banks before sweeps;
- predefine task split into development and held-out sets;
- report every seed and every task, not only aggregate means;
- bootstrap confidence intervals over tasks;
- report phase-boundary uncertainty;
- select controller thresholds only on development tasks;
- evaluate final transfer exactly once on held-out tasks;
- archive negative results and failed collapse coordinates.

## Deliverables

The experiment is complete only when the repository contains:

1. immutable candidate-bank metadata;
2. generation script/configuration;
3. public/hidden test specifications;
4. recursive sweep outputs;
5. held-out phase diagrams;
6. collapse-coordinate comparison;
7. adaptive-refresh cost/failure frontier;
8. a short falsification report stating which claims survived.
