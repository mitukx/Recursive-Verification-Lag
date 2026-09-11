# Research Overview

## Research question

The project studies recursive policy-verifier loops in which a policy is repeatedly improved by optimizing an imperfect verifier, while trusted external verification is limited.

The central question is:

> When does recursive self-improvement genuinely require new trusted verification rather than reuse of old trusted evidence?

The current thesis is:

> **Freshness is required by statistical novelty and coverage loss, not simply by elapsed rounds.**

A verifier can remain useful indefinitely when future policy comparisons stay inside a well-covered, low-complexity error class. Conversely, fresh trusted information is unavoidable when optimization repeatedly reaches new, poorly covered reward/error directions.

## Three layers of the theory

### Layer 1 — policy shift

An optimizer changes the policy from `p` to `q`. In unstructured settings, current-policy verification can become expensive when `q/p` is large on candidate-salient regions.

### Layer 2 — verifier-error geometry

Raw distribution shift is not enough. If the possible reward/verifier errors belong to a structured class `F`, the relevant quantity is

\[
\mathcal V_{\mathcal F}(p,q;\mu)
=
\sup_{f\in\mathcal F,f\ne0}
\frac{(\mathbb E_q f-\mathbb E_p f)^2}{\mathbb E_\mu f^2}.
\]

For a linear feature class this becomes a Mahalanobis/restricted-chi-square geometry.

### Layer 3 — recursive reuse and staleness

When a verifier is reused over several updates, the relevant verification object is the composed endpoint comparison between verifier refreshes. In a frozen exponential block,

\[
q_{s:L}(y)
\propto
p_s(y)e^{\Lambda_s v_s(y)}.
\]

Thus recursive verification can be organized around **refresh blocks**, not necessarily individual optimization rounds.

## Main mechanism: same-verifier self-evaluation blindness

In a canonical linear plug-in model, a learned verifier `theta_hat` produces the candidate direction

\[
d=\eta\Sigma\widehat\theta.
\]

The same verifier then reports

\[
\widehat\Delta=d^T\widehat\theta
=\eta\|\widehat\theta\|_\Sigma^2\ge0.
\]

This creates zero rejection power for the naive same-verifier accept-if-positive rule.

With misspecification, the true gain can become negative while the plug-in certificate remains positive and increasingly confident.

## Main empirical mechanism

Across multiple controlled experiments, larger trusted-data budgets mostly sharpen finite-sample uncertainty around a fixed population misspecification boundary. Enriching the verifier representation, by contrast, can move or remove the boundary.

In recursive experiments, current-policy refresh can self-correct. Failure emerges when the optimizer is allowed to move too far under a stale verifier.

This motivates the current conceptual definition:

> **Verification lag = optimizer-induced policy shift relative to verifier-error geometry.**

## Why this matters for recursive self-improvement

An external verifier that is perfectly hidden can still become informationally obsolete if future policy changes enter reward directions not identified by the original trusted data. Conversely, a fixed trusted dataset can remain valid for many updates when the relevant verification class has saturated.

The important resource is therefore not only verifier secrecy or benchmark size, but **how much statistically relevant new information the improvement loop generates over time**.
