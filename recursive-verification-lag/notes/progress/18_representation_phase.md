# Progress Update XVIII — Verification-budget × verifier-representation phase experiment

**Date:** 2026-09-11  
**Status:** Completed locally. Attempted escalation to a pretrained code LM was blocked by the execution environment, so the research was advanced by a stricter mechanism-separation experiment on the already-trained autoregressive DSL generator.

## 1. Why this update

Progress Update XVII established the §FD signature with a learned autoregressive code generator:

- under a misspecified verifier, true semantic gain first rises and then crosses below baseline at a stable optimization strength;
- increasing trusted-data budget sharpens the failure transition but barely moves its center;
- adding an exact exploit indicator to the verifier removes the harmful crossing.

The next question is more precise:

> Is the failure threshold mainly controlled by **how much trusted data the verifier receives**, or by **whether the verifier class can represent the optimizer-selected failure mode**?

This update varies those two axes independently.

## 2. Attempted pretrained-model escalation

The intended next step was a small real pretrained code LM. The concrete candidate identified was **BigCode TinyStarCoderPy**, a 164M-parameter GPT-BigCode model pretrained on Python code.

That run was **not performed** in this environment:

- `torch` is available on CPU;
- `transformers` is not installed;
- outbound package installation is blocked;
- direct model-weight download is also unavailable through the execution runtime.

Therefore no result below should be described as a pretrained-LLM result.

The empirical work instead reuses the previously trained autoregressive DSL generator so that the candidate distribution is still a learned generative policy rather than an enumerated uniform pool.

## 3. Two independent resources

Let the learned generator induce base program probabilities `p_j`.

Every program has:
- `visible_j`: public-test pass rate;
- `memflag_j`: whether it uses the public-test memorization/exploit mechanism;
- `true_j`: exhaustive hidden semantic accuracy.

The trusted verifier is fitted from semantic observations sampled from the learned generator.

Two resources are varied:

### Resource A — trusted sample budget

```text
n ∈ {500, 2000, 10000}
```

### Resource B — representation quality

The base verifier sees only
```text
[1, public_test_pass_rate].
```

A richer verifier additionally receives a static code-analysis signature
```text
h = memflag + sigma * epsilon_program,
```
where `epsilon_program` is one fixed program-level nuisance feature. Smaller `sigma` means the feature more accurately reveals the exploit mechanism.

The tested classes are:

```text
proxy_only            : no exploit-signature feature
weak_signature        : sigma = 4.0
medium_signature      : sigma = 1.0
strong_signature      : sigma = 0.35
near_exact_signature  : sigma = 0.10
exact_exploit_flag    : sigma = 0
```

This is intentionally a representation experiment, not a claim that real verifiers observe `memflag`.

## 4. Population phase diagram

For each verifier class, fit its infinite-data weighted least-squares projection and optimize it by exponential tilt.

| verifier_feature | signature_noise_sd | population_eta_star | max_true_gain | eta_at_max_gain | true_gain_eta250 | proxy_gain_eta250 |
| --- | --- | --- | --- | --- | --- | --- |
| proxy_only | — | 22.5409 | 0.0831 | 9.1000 | -0.1320 | 0.2799 |
| weak_signature | 4.0000 | 22.6224 | 0.0832 | 9.1000 | -0.1295 | 0.2799 |
| medium_signature | 1.0000 | 26.1460 | 0.0878 | 9.8000 | -0.0567 | 0.2889 |
| strong_signature | 0.3500 | >250 (no crossing) | 0.3024 | 163.4000 | 0.3000 | 0.3110 |
| near_exact_signature | 0.1000 | >250 (no crossing) | 0.2912 | 102.5000 | 0.2792 | 0.3626 |
| exact_exploit_flag | 0.0000 | >250 (no crossing) | 0.4283 | 250.0000 | 0.4283 | 0.3587 |

The result is qualitatively sharp:

- **proxy only:** harmful crossing at `eta* = 22.54`;
- **weak signature:** essentially unchanged (`22.62`);
- **medium signature:** threshold moves to `26.15`;
- **strong signature:** no harmful crossing anywhere in `eta <= 250`;
- **near-exact / exact exploit representation:** also no crossing.

Thus the population threshold is **not universal**. It is a property of the pair

```text
(optimizer, verifier function class)
```

and can disappear when the verifier can represent the selected failure.

![Representation threshold](learned_generator_representation_eta_star.png)

![Population gain curves](learned_generator_representation_population.png)

## 5. Finite-sample 2D sweep

The same verifier classes were then fitted from finite trusted samples and tested using the same-model confidence certificate.

| verifier_feature | n | replications | eta_10pct_wrong | eta_50pct_wrong | eta_90pct_wrong | transition_width_10_90 | wrong_prob_eta60 | harmful_prob_eta60 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| proxy_only | 500 | 180 | 18.6429 | 22.5000 | 27.6667 | 9.0238 | 1.0000 | 1.0000 |
| proxy_only | 2000 | 120 | 20.2857 | 22.3333 | 24.8889 | 4.6032 | 1.0000 | 1.0000 |
| proxy_only | 10000 | 70 | 21.2143 | 22.6304 | 23.6875 | 2.4732 | 1.0000 | 1.0000 |
| medium_signature | 500 | 180 | 17.8750 | 26.3750 | — | — | 0.8056 | 0.8056 |
| medium_signature | 2000 | 120 | 21.5000 | 27.8750 | 47.5000 | 26.0000 | 0.9083 | 0.9083 |
| medium_signature | 10000 | 70 | 23.0000 | 25.6667 | 29.2500 | 6.2500 | 1.0000 | 1.0000 |
| strong_signature | 500 | 180 | 24.0000 | — | — | — | 0.2278 | 0.2278 |
| strong_signature | 2000 | 120 | — | — | — | — | 0.0500 | 0.0500 |
| strong_signature | 10000 | 70 | — | — | — | — | 0.0000 | 0.0000 |

### Proxy-only class

Within the fixed misspecified proxy class, the 50% wrong threshold is

```text
n=500    : eta ≈ 22.50
n=2000   : eta ≈ 22.33
n=10000  : eta ≈ 22.63
```

Its total range is only

```text
0.297
```

despite a 20x increase in trusted data.

The transition width shrinks from roughly
```text
9.02 -> 4.60 -> 2.47,
```
again consistent with ordinary square-root concentration around an approximately fixed misspecification threshold.

### Medium representation

The population threshold has already moved upward to `26.15`. Finite-sample transitions are broader because the noisy exploit feature itself must be estimated, but the `n=10000` threshold (`25.67`) is close to its population value.

### Strong representation

There is no population harmful crossing up to `eta=250`. At `eta=60`, the harmful-candidate probability decreases from about

```text
0.228  (n=500)
0.050  (n=2000)
0.000  (n=10000).
```

Here more data does help, because the verifier class **contains enough information to model the exploit** and the remaining failure is estimation error rather than irreducible approximation error.

![2D budget/representation summary](learned_generator_representation_budget_2d.png)

![Same-budget representation curves](learned_generator_representation_curves_n2000.png)

## 6. Main scientific conclusion

The experiments now separate two regimes:

### Estimation-limited regime

If the verifier class can represent the optimizer-selected failure,

```text
more trusted data -> fewer harmful candidates.
```

The failure can disappear with budget.

### Misspecification-limited regime

If the verifier class cannot represent the selected failure,

```text
more trusted data -> sharper convergence to the wrong population model.
```

The optimization-strength threshold converges to a nonzero class-dependent value instead of moving indefinitely with budget.

The strongest empirical statement supported by the current sequence of experiments is therefore:

> **Trusted-label quantity controls uncertainty around a verifier's population projection; verifier representation controls where optimization of that projection becomes semantically harmful.**

Equivalently:

```text
budget changes width;
representation changes center.
```

This is a cleaner mechanism statement than “more verification does not help.”

## 7. Relation to Theorem U / §FD

The result supports the architecture-specific reading already forced by the v13–v15 corrections.

The problematic object is not verification in general. It is **same-model self-certification after optimizing the same misspecified verifier**.

Once the verifier converges inside a misspecified class:
1. its statistical uncertainty vanishes;
2. its optimization target remains wrong in selected regions;
3. the self-certificate becomes more stable;
4. the true sign can still reverse.

A fresh model-free post-selection semantic audit remains a distinct escape route.

## 8. What should and should not be claimed

### Supported

- The threshold is approximately budget-independent **within a fixed misspecified verifier class**.
- Its finite-sample width shrinks with more trusted data.
- Improving the verifier's representation can move the threshold substantially or eliminate it.
- Therefore the failure is not explained by sample scarcity alone.

### Not supported

- There is no universal numerical `eta_max`.
- There is no theorem here saying every misspecified verifier has one scalar threshold.
- `memflag` is a controlled feature, not a realistic static analyzer.
- This is not yet a pretrained-code-LLM result.
- The representation-vs-budget distinction is conceptually related to ordinary approximation-vs-estimation error, so that distinction alone should not be sold as novel.

The paper-level novelty must remain the recursive/self-evaluation mechanism and its interaction with optimizer-induced selection.

## 9. Updated empirical story

The current evidence chain is now:

1. **Finite non-Gaussian reward experiment**  
   stable center, `width ~ n^-1/2`.

2. **Finite program-semantics experiment**  
   public-test overfitting creates the same signature.

3. **Learned autoregressive DSL generator**  
   the signature survives when the candidate distribution comes from a learned generative policy; Best-of-N also shows rise-then-fall.

4. **Representation ablation (this update)**  
   budget barely moves the threshold inside a misspecified class, while a feature that reveals the exploit moves or removes the threshold.

The fourth experiment is the strongest causal/mechanistic control so far.

## 10. Next experiment

No new theorem is justified yet.

The next external-compute experiment should use an actual pretrained code LM. A minimal protocol is:

1. model: TinyStarCoderPy or Qwen2.5-Coder-0.5B;
2. 50–200 finite-domain coding tasks with exhaustive/large hidden tests;
3. generate a fixed candidate bank per task;
4. cheap proxy: public-test pass rate plus a learned verifier trained with trusted-label budgets `n`;
5. optimization: Best-of-N and/or exponential reweighting of candidate logits;
6. sweep `n` independently from optimization strength;
7. repeat with a richer verifier feature set;
8. primary statistics:
   - semantic gain,
   - confident-wrong rate,
   - 50% failure threshold,
   - transition width,
   - threshold movement under larger `n`,
   - threshold movement under richer representation.

The decisive plot is a two-axis comparison:

> **Does increasing data move the failure boundary less than increasing verifier expressiveness?**

If yes on a real pretrained code model, the empirical section becomes substantially stronger. If no, the current threshold phenomenon should be presented as a controlled-model mechanism rather than a broad law.
