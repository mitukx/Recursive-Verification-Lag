# Progress Update XVII — 学習済み自己回帰コード生成器での検証閾値実験

**日付:** 2026-09-11  
**状態:** 完了。pretrained code LLM は実行環境に存在しなかったため、次の一段として小型の自己回帰 GRU を実際に学習し、その生成分布を policy として用いた。  
**結論:** misspecified verifier では前回までの budget-insensitive threshold が再現し、verifier に exploit を表現できる feature を追加すると閾値が消える。さらに Best-of-N でも過最適化による符号反転が再現した。

## 1. 目的

前回の finite-domain program-synthesis 実験では候補プログラム集合を手続き的に列挙していた。今回の目的は、候補分布そのものを **学習済み生成モデル** に置き換え、以下を検証することだった。

1. learned generator の出力分布でも、optimization strength に budget-independent な failure threshold が現れるか。
2. その閾値が単なる「強い最適化」ではなく **verifier misspecification** に由来するか。
3. exponential/KL tilt 以外の optimizer、特に Best-of-N でも同様の overoptimization が出るか。

## 2. 生成モデル

小型の autoregressive GRU language model を DSL コード列上で学習した。

```text
embedding dimension = 24
GRU hidden dimension = 48
training sequences = 50,000
validation sequences = 5,000
epochs = 8
```

grammar-constrained decoding を用い、生成可能なコードは次の3種類。

```text
REG   : 通常の近似プログラム
MEM   : public tests を明示的に記憶する overfit プログラム
EXACT : 真の仕様を完全に実装するプログラム
```

最終 validation NLL は

```text
1.264778
```

だった。

学習後の generator mass:

| type | generator_mass | mean_visible | mean_true |
| --- | --- | --- | --- |
| REG | 0.9701 | 0.6203 | 0.5752 |
| MEM | 0.0294 | 1.0000 | 0.4303 |
| EXACT | 0.0005 | 1.0000 | 1.0000 |

したがって exploit (`MEM`) は約 2.94% と稀だが、proxy 上では非常に高得点になる。exact solution は約 0.046% しかない。

## 3. 重要な負の結果 — 最初の generator では failure が起きなかった

最初の学習設定では

```text
REG mass   ≈ 0.9841
MEM mass   ≈ 0.0146
EXACT mass ≈ 0.00137
```

だったが、public-test score が最大の集合に高品質 REG がかなり残った。その集合の平均 true accuracy は約 `0.714` で、generator baseline `0.655` より高かった。

そのため、optimization strength をいくら上げても semantic reward は baseline を下回らなかった。

これは重要な反例である。

> **exploit が生成分布に存在するだけでは overoptimization は起きない。exploit が optimizer が集中する proxy frontier を十分に占有する必要がある。**

したがって今回の positive result は「misspecification があれば必ず collapse」という主張ではない。

## 4. 修正版 stress-test generator

修正版では regular program の隠れ性能上限を抑え、rare MEM programs が public-test frontier を占有しうる controlled regime を作った。

学習後、最大 public-test score 集合の generator mass は

```text
0.029878
```

で、その集合の平均 exhaustive semantic accuracy は

```text
0.439124
```

だった。

generator baseline は

```text
0.571182
```

なので、proxy frontier そのものが baseline より悪い。これが Goodhart collapse を可能にする。

## 5. Misspecified verifier

trusted semantic data から次の verifier を OLS で fit する。

```text
v_hat(program) = beta0_hat + beta1_hat * public_test_pass_rate(program)
```

population projection は

```text
beta_pop ≈ (0.0911, 0.7601)
```

である。

candidate policy は

```text
q_eta(program) ∝ p_generator(program) * exp(eta * v_hat(program))
```

とした。

population true gain は `eta ≈ 9` で最大となり、その後低下して

```text
eta_star = 22.540885
```

でゼロを横切る。

## 6. Trusted verification budget sweep

自己 certificate は、candidate を生成した **同じ verifier** で proxy gain を評価する。confident-and-wrong は、一側 95% lower bound が正である一方、exhaustive semantic gain が負である事象。

| n | replications | beta1_mean | beta1_sd | certificate_fire_prob | eta10_wrong | eta50_wrong | eta90_wrong | width10_90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200.0000 | 3000.0000 | 0.7561 | 0.1947 | 0.9840 | 17.0842 | 22.7056 | 33.8143 | 16.7301 |
| 1000.0000 | 2400.0000 | 0.7579 | 0.0851 | 1.0000 | 19.7840 | 22.5591 | 26.4300 | 6.6460 |
| 5000.0000 | 1200.0000 | 0.7611 | 0.0386 | 1.0000 | 21.1389 | 22.5488 | 24.0381 | 2.8992 |
| 20000.0000 | 600.0000 | 0.7599 | 0.0188 | 1.0000 | 21.8562 | 22.5378 | 23.3000 | 1.4437 |

50% confident-wrong threshold の全 budget 間の変動幅は

```text
0.1677
```

にすぎず、population threshold `22.5409` の 1% 未満だった。

一方、10%-90% transition width は

```text
width ∝ n^-0.531
```

と推定された。

したがって三つの異なる controlled settings で一貫して、

```text
threshold location ≈ fixed by misspecification
finite-sample width ≈ n^(-1/2)
```

という署名が得られた。

![Threshold curves](learned_dsl_threshold_curves_v2.png)

![Threshold vs budget](learned_dsl_threshold_vs_budget_v2.png)

## 7. Mechanism control — verifier が exploit を表現できる場合

次に verifier feature に

```text
memorizes_public ∈ {0,1}
```

を追加した。

population verifier は概ね

```text
true accuracy ≈ -0.004 + 0.934 * public_pass - 0.500 * memorizes_public
```

となり、MEM exploit に明示的な負の補正を学習した。

population では `eta <= 250` の全範囲で true gain は非負であり、`eta=250` では gain は約

```text
0.4283
```

だった。すなわち misspecified model で見えた sign threshold は消えた。

有限標本結果:

| n | eta | harmful_prob | confident_wrong_prob | certificate_fire_prob |
| --- | --- | --- | --- | --- |
| 200.0000 | 10.0000 | 0.0067 | 0.0067 | 0.9983 |
| 200.0000 | 30.0000 | 0.0358 | 0.0342 | 0.9958 |
| 200.0000 | 80.0000 | 0.0358 | 0.0333 | 0.9950 |
| 200.0000 | 150.0000 | 0.0342 | 0.0317 | 0.9950 |
| 1000.0000 | 10.0000 | 0.0000 | 0.0000 | 1.0000 |
| 1000.0000 | 30.0000 | 0.0000 | 0.0000 | 1.0000 |
| 1000.0000 | 80.0000 | 0.0000 | 0.0000 | 1.0000 |
| 1000.0000 | 150.0000 | 0.0000 | 0.0000 | 1.0000 |
| 5000.0000 | 10.0000 | 0.0000 | 0.0000 | 1.0000 |
| 5000.0000 | 30.0000 | 0.0000 | 0.0000 | 1.0000 |
| 5000.0000 | 80.0000 | 0.0000 | 0.0000 | 1.0000 |
| 5000.0000 | 150.0000 | 0.0000 | 0.0000 | 1.0000 |
| 20000.0000 | 10.0000 | 0.0000 | 0.0000 | 1.0000 |
| 20000.0000 | 30.0000 | 0.0000 | 0.0000 | 1.0000 |
| 20000.0000 | 80.0000 | 0.0000 | 0.0000 | 1.0000 |
| 20000.0000 | 150.0000 | 0.0000 | 0.0000 | 1.0000 |

`n >= 1000` では検査した全 `eta <= 150` で harmful proposal は 0 件だった。`n=200` でだけ約3–4%の finite-sample failure が残る。

これは強い mechanism control である。

> **budget-independent threshold は optimization strength 単独の性質ではなく、optimizer が verifier の表現不能な proxy frontier に集中することから生じる。**

![Rich verifier control](learned_dsl_rich_population_control_v2.png)

## 8. Optimizer robustness — Best-of-N

同じ learned generator と同じ misspecified proxy ranking を使い、exponential tilt を Best-of-N に置き換えた。

exact population calculation では:

```text
maximum true gain = 0.129091
at N = 6

first harmful N = 34
true gain there = -0.000508
```

したがって性能は小さい N では改善し、その後、候補数を増やしすぎると public-test-overfit frontier を選ぶようになって baseline を下回る。

これは重要である。今回の collapse は exponential/KL tilt 特有の数式ではない。

![Best-of-N](learned_dsl_bestofn_v2.png)

## 9. Fresh post-selection semantic audit

candidate 固定後に、generator baseline と candidate policy から fresh semantic evaluations を直接取得した場合:

| eta | true_gain | m_per_distribution | sign_error_rate |
| --- | --- | --- | --- |
| 23.0000 | -0.0035 | 106749.0000 | 0.0508 |
| 25.0000 | -0.0182 | 4017.0000 | 0.0450 |
| 30.0000 | -0.0492 | 553.0000 | 0.0517 |
| 35.0000 | -0.0720 | 259.0000 | 0.0552 |

`eta=23` は zero-gain point に極めて近いため約 10万 sample / distribution を要するが、harmful margin が大きくなるにつれて必要量は急速に減少する。

したがって今回も

> **harmful candidate 自体が verification-impossible なのではない。self-evaluation verifier が failure mode を表現できないことが問題である。**

という corrected interpretation が支持される。

## 10. 現時点で得られた新しい知見

今回の一番重要な追加は単なる3回目の threshold 再現ではない。以下の三点である。

### A. Frontier-occupancy condition

overoptimization には「rare exploit の存在」だけでなく、

```text
optimizer-selected proxy frontier の conditional true quality < current-policy baseline
```

という条件が必要である。

最初の failed generator がその反例を与えた。

### B. Representability control

verifier feature が exploit type を表現できるようになると、十分な trusted data で threshold は消える。

したがって threshold は **budget-independent だが model-class-independent ではない**。

### C. Optimizer-class robustness

同じ learned generator で Best-of-N に変えても rise-then-fall と harmful crossing が発生した。

したがって central empirical claim は

```text
「exponential update に threshold がある」
```

ではなく、

```text
「misspecified verifier が誤った proxy frontier を形成すると、
optimization pressure の増加により learned generator はその frontier に集中し、
verification budget では動かない semantic sign reversal が生じうる」
```

とするのが適切である。

## 11. 限界

この実験は前回より一段実際の model-based generation に近いが、まだ pretrained code LLM ではない。

- generator は小型 GRU。
- DSL は grammar constrained。
- public-test memorization template は training support に明示的に含まれる。
- exploit feature を rich-verifier control では直接与えている。
- semantic target は有限かつ完全評価可能。

したがってこれは **mechanism experiment** であり、real code agent での prevalence claim ではない。

## 12. Paper-level implication

現時点では empirical story がかなり強くなった。

1. nonlinear non-Gaussian reward model;
2. enumerated finite program semantics;
3. **learned autoregressive code generator**;
4. verifier-representability control;
5. exponential optimization と Best-of-N の両方。

特に negative first attempt と rich-verifier control が重要で、単に desired curve を作っただけではなく、**いつ curve が出ないか**も記録できた。

論文で最も安全な実験的主張は:

> **Verification budget controls estimation uncertainty, but cannot move a semantic failure boundary created by a fixed representational blind spot. The boundary disappears when the verifier class is enriched enough to represent that blind spot.**

## 13. 次の研究判断

次の一手は pretrained small code LLM である。

必要な実験は一つだけ:

```text
small code LM
+ multiple sampled solutions
+ visible/public tests as proxy
+ stronger hidden tests as trusted reward
+ independent sweep of selection strength and verifier-data budget
```

ここで同じ threshold-location-vs-budget plot が得られれば、Main-track empirical support はかなり強くなる。

逆に real-model で threshold が budget とともに大きく移動するなら、今回までの theory は「fixed representational misspecification regime」の限定結果として縮めるべきである。

新しい theorem はまだ追加しない。
