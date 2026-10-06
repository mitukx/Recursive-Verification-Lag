# Mathematical RSI leap from the OpenAI Math 2026 results

This module uses several OpenAI Math results as **algorithm-design principles**
inside the bounded Recursive Self-Improvement controller. It does not claim that
those theorems directly establish safety or convergence for LLM self-improvement.

The implementation lives in `src/rsi_controller/math_rsi.py`. The locked
experiment configuration is `configs/rsi/oai_math_leap_v1.json`.

## 1. Family 136 -> coded verification

The PCP-for-PPAD result shows a robust local-constraint representation in which
an endpoint can still be decoded even when an adversary chooses a fixed
fraction of constraints to violate.

RVL uses this as an architectural pattern:

```
candidate improvement
  -> critical global invariants
  -> redundant local obligations
  -> bounded corruption budget
  -> promotion certificate
```

Critical obligations (independent trusted non-regression, anti-reward-hacking,
and no low-trust RVL state) may never fail. Noncritical obligations are
redundant; the configured `max_corrupt_fraction` controls how many may fail.
This is an engineering error-correcting gate, not a PCP-for-PPAD reduction.

## 2. Family 116 -> bounded fixed-probe verification

The universal hitting-set results motivate a stronger notion than periodic
refresh: define a bounded mutation/error class *before optimization*, attach a
fixed set of probes to every allowed mutation coordinate, and fail closed when
the optimizer leaves that class.

The current fixed probe registry covers the allowlisted Harness, Training,
Verifier, and Policy mutation coordinates. Candidate-controlled code patches
are outside the class. So is a mutation whose declared complexity exceeds the
locked bound.

This does **not** prove that the probes are a universal hitting set for model
failures. The research target is to replace this engineering coverage contract
with a theorem-backed error class.

## 3. Family 140 -> trusted-information budget

The noiseless-Gaussian-regression result gives a memory/sample lower bound with
the characteristic scaling

[
dlog(1/epsilon).
]

The RSI controller uses this only as an explicit research proxy:

[
N_{mathrm{trusted, required}}
=
leftlceil c,d_{mathrm{eff}}log(1/epsilon)ightceil.
]

Here `d_eff` is the mutation/probe complexity exposed by the bounded candidate
and `epsilon` is the target verification error. Promotion is blocked if the
available independent trusted promotion samples are below the required proxy
budget.

This is **not** claimed as a transferred lower bound for LLM verification. A
formal reduction from a verifier-tracking problem to the assumptions of family
140 remains an open theorem target.

## 4. Family 229 -> recursive signal criticality

For the exact three-state reconstruction model the sharp condition is

[
dlambda^2>1,
]

with non-reconstruction at equality. The module computes the corresponding
criticality diagnostic using a declared branching factor and, under a binary
symmetric-channel abstraction, (hatlambda=2a-1) from verifier agreement
(a).

The current champion/challenger loop evaluates a single challenger and does not
establish the independent homogeneous tree-channel assumptions. Therefore
family 229 is **diagnostic-only by default**. It becomes a hard gate only when a
future branching experiment explicitly declares those assumptions and enables
`reconstruction_assumptions_met`.

`src/benchmark_mathematical_rsi.py` locks the basic phase behavior:
(d=4,a=0.75) gives (dlambda^2=1) and is not supercritical, while
(d=4,a=0.8) is above threshold.

## Promotion semantics

The default controller keeps the mathematical certificate purely additive.
The locked `oai_math_leap_v1` experiment goes one step further: it delegates
four designated noncritical checks (verification gap, failure-rate delta,
latency, and verifier agreement) to the coded-verification layer. This allows a
bounded fraction of those local obligations to fail, while the following remain
hard invariants: independent trusted non-regression, anti-reward-hacking, no
low-trust RVL state, configured capability gain, probe coverage, trusted
information budget, and the statistical uncertainty gate.

```
hard invariants
AND coded local obligations with <= delta corruption
AND bounded fixed-probe coverage
AND trusted-information budget
-> promote
```

Thus the OAI experiment changes promotion semantics in a narrow, explicit,
reversible way rather than silently weakening all gates.

Every assessment is appended to research memory as
`mathematical_rsi_assessment`. A verifier refresh recomputes the certificate
before a second promotion decision.

## Immediate research program

The highest-value next experiment is to replace the synthetic local obligations
with real-model evidence: code/tool-use or Lean proof trajectories, multiple
independently scored local obligations, sparse trusted audits, and adversarial
verifier corruption. Then compare:

1. ordinary scalar promotion;
2. coded verification only;
3. coded verification + fixed probe coverage;
4. coded verification + adaptive trusted-information budget;
5. a genuine multi-branch recursive experiment where the family-229
   assumptions are deliberately instantiated and the (dlambda^2) boundary
   can be tested prospectively.

A successful result would be a real technical step beyond "refresh the verifier
when stale": the system would make verification **redundant, structurally
covered, and budgeted by information complexity**.
