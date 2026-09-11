# Corrections and Negative Results

A core goal of this repository is to preserve failed ideas and corrections rather than retroactively presenting a straight-line success story.

## Reject-all loophole

The original safety-only lower-bound objective admitted a zero-query algorithm that rejected every proposal. The formal objective was corrected to require power/completeness or familywise correct update classification.

## Fixed-budget versus expected sequential complexity

The `M log(1/delta)` rare-tail law is a fixed-budget/high-probability latency statement. Expected sequential discovery is only `Theta(M)` in the exact-label rare-event setting.

## Raw amplification is not universal

Large `q/p` does not universally imply large verification cost. Structured error/reward classes can extrapolate into shifted regions, replacing raw amplification by restricted feature geometry.

## Generic adaptive-data-analysis machinery is not the novelty

Reusable holdouts, max-information, privacy/stability transfer, and winner's-curse corrections are established areas. They are supporting machinery here, not standalone contribution claims.

## Mean optimism bias is not a universal lower bound

The plug-in winner's-curse calculation is architecture-specific. Mean bias alone cannot establish an information-theoretic impossibility for arbitrary bias-corrected estimators.

## `eta_max` is not a verification ceiling

The misspecification `eta_max` is the true-gain zero crossing of a fixed update family. Fresh post-selection semantic evidence can still reject harmful candidates beyond the crossing.

## “More oversight makes things worse” was too broad

More data alone do not worsen the true policy. A negative effect requires either operational coupling between confidence and optimization aggression or a misspecified self-certificate becoming more certain about the wrong model.

## Two structurally distinct audit streams are not necessary

A conjecture that selection control and misspecification control fundamentally require separate audit streams was refuted. A balanced mixture can serve both current certification and future training roles within constant factors in the studied model.

## A learned generator did not initially collapse

The first GRU generator placed enough genuinely good programs on the public-test frontier that stronger optimization remained beneficial. This showed that exploit presence alone is insufficient; the optimizer-selected proxy frontier must itself be semantically poor.

## One-step threshold does not imply inevitable recursive collapse

Current-policy verifier refresh can change the population projection and self-dampen the optimizer. The recursive variable is verifier staleness relative to policy movement, not just the existence of one-step misspecification.

## Raw `eta * L` is not universal

Rescaling verifier scores rescales the apparent `eta * L` threshold. Policy-level quantities are more portable, but even KL or max density ratio alone fail across verifier representations.

## KL-triggered adaptive refresh did not win

A KL-triggered refresh rule looked efficient at population level but lost its advantage once finite-sample significance stopping was included. It is retained as a negative result, not promoted as a contribution.
