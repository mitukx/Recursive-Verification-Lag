# Master Research Record — Recursive Verification Lag / Fresh Verification Budgets

**統合日:** 2026-09-11  
**対象:** 2026-09-08〜2026-09-11 に進めた理論・自己監査・反例・数値検証・synthetic / program-synthesis / learned-generator 実験の統合記録  
**主要ソース:** `recursive_verification_research_note_20260910_v15.md` および Progress Update XV–XX  
**目的:** 論文本文ではなく、現時点で「何が証明済みか・何が経験的に支持されたか・何が撤回されたか・何が未解決か」を失わず残す研究台帳

---

## 0. 現時点の結論

研究の最初の問いは次だった。

> **How much genuinely fresh verification is fundamentally necessary to sustain \(T\) rounds of policy improvement when the policy adaptively optimizes an imperfect verifier?**

現在までの研究で、この問いに対する答えは単純な「毎 round fresh verification が必要」ではないことが分かった。

現時点の最も正確な整理は次である。

\[
\boxed{
\text{必要な fresh verification は recursive depth }T\text{ そのものではなく、}
\text{新しい verification directions と audit coverage の成長に依存する。}
}
\]

さらに、同じ verifier を最適化と自己評価の両方に使う endogenous loop では、別の構造的問題がある。

\[
\boxed{
\widehat\Delta
=
\eta\|\theta+e\|_\Sigma^2
\ge 0
}
\]

となり、**最適化に使った推定器でその更新を評価すると、測定された improvement が構造的に非負になる**。したがって、その self-certificate は harmful proposal を reject する能力を持たない。これが Theorem U の中核である。

一方、misspecification があるだけで recursive collapse が必然になるわけでもない。2026-09-11 の recursive experiments では、current-policy 上で verifier を十分頻繁に refresh すると、同じ misspecified verifier class でも自己修正して collapse を回避した。

最新の empirical synthesis は、

\[
\boxed{
\text{recursive damage is governed by verification lag:
optimizer movement accumulated while the verifier is stale.}
}
\]

である。

固定 verifier を \(L\) step 使い、各 update が exponential tilt strength \(\eta\) なら、stale block 内では厳密に

\[
p_{t+L}(y)
\propto
p_t(y)\exp(L\eta v_t(y)),
\]

なので

\[
\boxed{\eta_{\rm stale}=\eta L}
\]

が自然な第一近似の stale-exposure coordinate になる。learned DSL generator の実験では baseline-collapse boundary が概ね

\[
\boxed{\eta L \approx 15\text{--}20}
\]

に集約した。ただしこの数値自体は環境依存であり、一般定理ではない。

---

# Part I — 研究課題と基本モデル

## 1. 基本設定

round \(t\) において、

- \(p_t\): current / pre-update policy distribution
- \(q_t\): verifier を最適化して得た provisional candidate
- \(r\): unknown true utility / trusted reward
- \(v_t\): proxy verifier
- \(\mu_t\): trusted audit distribution
- \(m_t\): trusted labels
- \(\Delta_t=\mathbb E_{q_t}r-\mathbb E_{p_t}r\): true gain

を考える。

代表的 update は exponential / KL tilt:

\[
q_t(y)
=
\frac{p_t(y)\exp(\eta_t v_t(y))}
{\mathbb E_{p_t}[\exp(\eta_t v_t)]}.
\]

主要な recursive objective は、単純 safety-only ではなく、lower bound を非自明にするため **safety + power / familywise correct classification** とする。

\[
\Pr(
\text{all \(T\) update decisions are correct}
)
\ge 1-\delta.
\]

これは研究途中で行った重要な修正である。元の「harmful update を accept しない」だけなら、何も調べず全て reject する zero-query algorithm が成立するため、verification lower bound は意味を失う。

---

## 2. 最初の exact decomposition

verifier error を

\[
e=v-r
\]

とすると、

\[
G=\mathbb E_q[v]-\mathbb E_p[v]
\]

に対して exact に

\[
\boxed{
\Delta
=
G-
\left(
\mathbb E_q[e]-\mathbb E_p[e]
\right).
}
\]

したがって reward hacking / proxy exploitation の本体は、**optimizer が verifier error への exposure をどれだけ増やしたか**である。

平均 error が小さいだけでは安全性を保証できない。

---

## 3. \(L_2\)-coverage / concentrability upper bound

audit distribution \(\mu\) 上で

\[
\|e\|_{L_2(\mu)}\le\varepsilon
\]

なら、

\[
\boxed{
\Delta
\ge
G-\varepsilon\left(C(q\|\mu)+C(p\|\mu)\right),
}
\]

where

\[
C(q\|\mu)
=
\left(
\mathbb E_\mu
\left[
\left(\frac{dq}{d\mu}\right)^2
\right]
\right)^{1/2}
=
\sqrt{1+\chi^2(q\|\mu)}.
\]

current-policy audit \(\mu=p\) では candidate shift が error を増幅する。

exponential optimization で verifier span \(B\) を持つ場合、

\[
C(q\|p)\le e^{\eta B/2},
\]

なので worst-case sufficient condition は exponential に悪化し得る。

これは後に「raw \(M\) は unstructured worst case であり universal ではない」と修正・一般化された。

---

# Part II — Moving-tail verification lower bounds

## 4. Rare-tail one-step separation

candidate-salient set \(A\) に対して

\[
p(A)=\frac1M,
\qquad
q(A)=\frac12
\]

とする。

\(A\) 上の true reward の符号だけが unknown の二世界を作ると、current-policy sampling では \(A\) を一度も見ない確率が

\[
\left(1-\frac1M\right)^m.
\]

fixed-budget / high-probability decision では

\[
\boxed{
m_{\rm current}
=
\Omega\left(M\log\frac1\delta\right),
}
\]

一方 candidate audit では \(q(A)=1/2\) なので

\[
\boxed{
m_{\rm candidate}
=
O\left(\log\frac1\delta\right).
}
\]

ここで重要なのは cost model の区別である。

- fixed query cap / high-probability latency: \(\Theta(M\log(1/\delta))\)
- sequential sampling until discovery: expected \(\Theta(M)\)

\(\log\) factor を expected sequential complexity として主張してはいけない。

---

## 5. Adaptive moving-tail construction — Theorem A′

### 状態

**PROVED in the adaptive binary-tree hard family.**

binary tree の各 node に独立 hidden sign

\[
\theta_x\in\{-1,+1\}
\]

を置く。

現在 node \(x_t\) では exploit action \(b\) の current mass が

\[
p_t(b\mid x_t)=1/M_t,
\]

verifier は \(b\) を好み、KL/exponential update により

\[
q_t(b\mid x_t)=1/2.
\]

正しい accept/reject decision が hidden sign \(\theta_{x_t}\) と一致するよう reward を設定する。

前 round の decision が次の node を選ぶため、verification target 自体が adaptive に移動する。

### Theorem A′

per-round deterministic query cap \(m_t\) に対して、

\[
\boxed{
\Pr(\text{all \(T\) decisions correct})
\le
\prod_{t=1}^T
\left[
1-\frac12
\left(1-\frac1{M_t}\right)^{m_t}
\right].
}
\]

各 visited node は過去 transcript から独立な fresh sign を持つので、previous repair は次の tail について情報を持たない。

---

## 6. Heterogeneous minimax budget

\[
h_t
=
\frac1{-\log(1-1/M_t)},
\qquad
H=\sum_{t=1}^T h_t.
\]

hard family において、

\[
\boxed{
B_{\rm current}^\star
=
\Theta
\left(
\sum_t
h_t
\log
\frac{H}{\delta h_t}
\right).
}
\]

equal \(M_t=M\) なら

\[
\boxed{
B_{\rm current}^\star
=
\Theta
\left(
TM\log\frac{T}{\delta}
\right).
}
\]

candidate-policy audit では

\[
\boxed{
B_{\rm candidate}^\star
=
\Theta
\left(
T\log\frac{T}{\delta}
\right),
}
\]

したがって

\[
\boxed{
\frac{B_{\rm current}^\star}
{B_{\rm candidate}^\star}
=
\Theta(M).
}
\]

これは worst-case adaptive family における「verification must chase the policy」の最も clean な query-complexity statement である。

---

## 7. Mixed audit phase transition

\[
\mu_{\lambda,t}
=
(1-\lambda_t)p_t+\lambda_t q_t.
\]

rare-tail hard family では informative audit mass が

\[
\mu_{\lambda,t}(A_t)
=
\frac{1-\lambda_t}{M_t}
+
\frac{\lambda_t}{2}.
\]

equal-\(M,\lambda\) で

\[
\boxed{
B_\lambda^\star
=
\Theta
\left[
\frac{T}{M^{-1}+\lambda}
\log\frac{T}{\delta}
\right].
}
\]

crossover:

\[
\boxed{
\lambda_c=\Theta(1/M).
}
\]

つまり candidate-policy mass を audit に大量に入れる必要はなく、**order \(1/M\)** の混合だけで scaling regime が変わる。

---

# Part III — Noisy verification and information law

## 8. Local Bhattacharyya information law — Theorem B

candidate-salient region \(A\) の audit mass を

\[
a=\mu(A)
\]

とする。

\(A\) 上で二つの label worlds の Bhattacharyya coefficient が \(\rho\) なら、一 query の information rate を

\[
\boxed{
I(a,\rho)
=
-\log[(1-a)+a\rho]
}
\]

と定義できる。

equal-prior binary test の minimax Bayes error は

\[
\boxed{
\frac14e^{-2mI(a,\rho)}
\le
P_e^\star(m)
\le
\frac12e^{-mI(a,\rho)}.
}
\]

したがって

\[
\boxed{
m_\varepsilon^\star
=
\Theta
\left(
\frac{\log(1/\varepsilon)}
{I(a,\rho)}
\right).
}
\]

これは新しい bespoke metric ではなく、通常の testing information を verification setting に埋め込んだもの。

---

## 9. Bernoulli noisy labels

\(A\) 上で

\[
\Pr_\theta(Y=+1\mid A)
=
\frac{1+\theta\gamma}{2}
\]

なら、

\[
\rho_\gamma=\sqrt{1-\gamma^2},
\]

and

\[
I(a,\gamma)
=
-\log
\left[
1-a+a\sqrt{1-\gamma^2}
\right].
\]

small / moderate regime で

\[
\boxed{
I(a,\gamma)=\Theta(a\gamma^2)
}
\]

なので

\[
\boxed{
m^\star
=
\Theta
\left(
\frac1{a\gamma^2}
\log\frac1\varepsilon
\right).
}
\]

結論:

\[
\boxed{
\text{rarity cost and label-noise cost multiply.}
}
\]

\[
a^{-1}\gamma^{-2},
\]

であり \(\max\{a^{-1},\gamma^{-2}\}\) ではない。

---

## 10. Noisy current-vs-candidate law

rare-tail specialization では、

current audit:

\[
\boxed{
m_{\rm current}
=
\Theta
\left(
\frac{M}{\Gamma^2}
\log\frac1\varepsilon
\right),
}
\]

candidate audit:

\[
\boxed{
m_{\rm candidate}
=
\Theta
\left(
\frac1{\Gamma^2}
\log\frac1\varepsilon
\right),
}
\]

so

\[
\boxed{
m_{\rm current}/m_{\rm candidate}
=
\Theta(M).
}
\]

noise を入れても amplification separation は残る。

---

## 11. Multi-round noisy theorem — Theorem C

round \(t\) の local information rate を \(I_t\) とすると、

\[
\boxed{
\Pr(\text{all \(T\) decisions correct})
\le
\prod_{t=1}^T
\left[
1-\frac14e^{-2m_t I_t}
\right].
}
\]

familywise reliability のためには

\[
\sum_t e^{-2m_tI_t}
\]

を control する必要があり、weighted allocation による minimax budget characterization が得られる。

equal-parameter noisy hard familyでは概ね

\[
\boxed{
B^\star
=
\Theta
\left[
\frac{T}
{\gamma^2(M^{-1}+\lambda)}
\log\frac{T}{\delta}
\right].
}
\]

---

# Part IV — Structured coverage: raw \(M\) から function-class geometry へ

## 12. Structured verification functional

possible reward/verifier errors を closed linear subspace

\[
\mathcal F\subseteq L_2(\mu)
\]

で制限し、policy-improvement functional

\[
L_{p,q}(f)
=
\mathbb E_q f-\mathbb E_p f
\]

を考える。

この functional の Riesz representer \(g_{\mathcal F}\) の squared norm

\[
\mathcal V_{\mathcal F}(p,q;\mu)
=
\|g_{\mathcal F}\|_{L_2(\mu)}^2
\]

が one-step verification difficulty を与える。

---

## 13. Theorem D — structured fresh-verification complexity

local boundedness condition の下で

\[
\boxed{
m^\star
=
\Theta
\left(
\frac{
\mathcal V_{\mathcal F}(p,q;\mu)
}{
\Gamma^2
}
\log\frac1\delta
\right).
}
\]

upper bound は Riesz representer を用いた unbiased estimator + median-of-means。

lower bound は

\[
f_\pm
=
\pm
\frac{\Gamma}{V}g
\]

という least-favorable directions と KL / Bretagnolle–Huber により得る。

これは one-round theory の中では最も一般的な形式の一つ。

---

## 14. Linear class の exact Mahalanobis form

\[
\mathcal F
=
\{f_\theta(y)=\theta^\top\phi(y)\}.
\]

\[
\Sigma_\mu
=
\mathbb E_\mu[\phi\phi^\top],
\qquad
d_{p,q}
=
\mathbb E_q[\phi]-\mathbb E_p[\phi].
\]

\(d_{p,q}\in\operatorname{Range}(\Sigma_\mu)\) なら

\[
\boxed{
\mathcal V_{\mathcal F}
=
d_{p,q}^\top
\Sigma_\mu^\dagger
d_{p,q}.
}
\]

range 外なら

\[
\boxed{
\mathcal V_{\mathcal F}=+\infty.
}
\]

つまり identifiability と sample complexity が feature covariance geometry で記述できる。

---

## 15. Raw \(M\)-law の recovery

unrestricted error class では

\[
\mathcal V_{\rm full}
=
\chi^2(q\|p),
\]

rare-tail hard family で

\[
\chi^2(q\|p)=\Theta(M).
\]

したがって earlier \(M\)-tax は structured theorem の unstructured special case である。

---

## 16. Smooth-class example — Theorem E

\(p=\mathrm{Unif}[0,1]\)、candidate が width \(1/M\) の tail を mass \(1/2\) へ amplify する。

unrestricted class なら

\[
\mathcal V_{\rm full}=\Theta(M).
\]

一方 verifier error class を degree-\(k\) polynomial に制限すると、\(k^2=o(M)\) で

\[
\boxed{
\mathcal V_{\mathcal F_k}
=
\frac{k(k+2)}4
\left[
1+O(k^2/M)
\right].
}
\]

よって

\[
\boxed{
m^\star_{\mathcal F_k}
=
\Theta
\left(
\frac{k^2}{\Gamma^2}
\log\frac1\delta
\right),
}
\]

while raw density ratio remains order \(M\).

重要な修正:

\[
\boxed{
\text{large policy amplification does NOT universally imply a large verification tax.}
}
\]

構造的 extrapolation が可能なら raw \(M\) dependence はほぼ消える。

---

# Part V — Reusable verification, adaptivity, and information replenishment

## 17. Finite reachable-class reusable verification

reachable comparisons の deterministic envelope を

\[
\mathcal H_T,
\qquad
|\mathcal H_T|=K_T
\]

とする。

各 comparison の variance difficulty が \(V_h\le V\) なら、一つの trusted dataset に対して uniform event を張ることで

\[
\boxed{
n
=
O
\left(
\frac{V}{\Gamma^2}
\log\frac{K_T}{\delta}
\right)
}
\]

で **同じデータから adaptively selected された全 update** を判定できる。

結論:

\[
\boxed{
\text{There is no universal requirement for \(\Theta(T)\) fresh batches.}
}
\]

reachable verification class が飽和し、coverage が十分なら old trusted data は再利用可能。

---

## 18. Fresh post-selection data の役割

candidate \(\widehat h=A(S)\) を old data \(S\) から選んだ後、independent fresh set \(S'\) を取れば、conditional on \(\widehat h\) で通常の pointwise test に戻る。

\[
\boxed{
m
=
O
\left(
\frac{V_{\widehat h}}{\Gamma^2}
\log\frac1\delta
\right).
}
\]

fresh data が消すのは **post-selection certification penalty** であり、candidate discovery / search の統計コストそのものではない。

この discovery vs certification の区別は研究途中で重要な修正として確立した。

---

## 19. Coverage × selection upper law

finite reachable class では

\[
\boxed{
n
=
O
\left[
\frac{V_T}{\Gamma^2}
\left(
\log K_T+\log\frac1\delta
\right)
\right].
}
\]

ただしこの product が一つの endogenous recursive family で minimax-tight であるという central theorem は未達。

generic adaptive-data-analysis / max-information / DP の machinery 自体は既存研究であり、本研究の独自 contribution としては扱わない。

---

# Part VI — External verification entropy and audit lifetime

## 20. Theorem J — External Verification Entropy Lower Bound

environment index

\[
\Theta\sim\mathrm{Unif}\{1,\dots,K\}
\]

を correct recursive operation から decode できる必要があるとする。

trusted observation \(O_i\) 一つ当たり

\[
I(\Theta;O_i\mid \Pi_{i-1})\le C_i
\]

なら Fano により

\[
\boxed{
\sum_i C_i
\ge
\log K
-
h(\alpha)
-
\alpha\log(K-1).
}
\]

uniform per-query cap \(C_i\le C\) なら

\[
\boxed{
N
\ge
\frac{
\log K-h(\alpha)-\alpha\log(K-1)
}{C}.
}
\]

意味:

\[
\boxed{
\text{secrecy can prevent gaming, but secrecy cannot manufacture missing reward information.}
}
\]

perfectly sealed audit でも、new independent reward directions が生まれれば exogenous information supply が必要。

---

## 21. Orthogonal innovation phase law

horizon \(T\) までに independent hidden reward directions が \(r_T\) 個だけなら、

current-policy rare-gate audit:

\[
N_T
=
\widetilde\Theta
\left(
\frac{M r_T}{\gamma^2}
\right),
\]

candidate-aware audit:

\[
N_T
=
\widetilde\Theta
\left(
\frac{r_T}{\gamma^2}
\right).
\]

したがって

\[
r_T=O(1)
\Rightarrow
N_T=\widetilde O(1),
\]

\[
r_T=o(T)
\Rightarrow
N_T=o(T),
\]

\[
r_T=\Theta(T)
\Rightarrow
N_T=\widetilde\Theta(T).
\]

重要な conceptual result:

\[
\boxed{
T\text{ itself is not fundamental; the growth of genuinely new verification directions is.}
}
\]

ただし rank-growth / task-eluder-like complexity 自体は既存 lifelong-learning theory と近く、それだけでは novelty として弱い。

---

# Part VII — Sealed linear audits and continuous geometry

## 22. Uniform confidence ellipsoid / sealed audit

linear-Gaussian model で sealed design \(A\) と OLS estimate \(\widehat\theta\) を考える。

confidence ellipsoid event 上で simultaneously for all \(d\),

\[
\boxed{
|d^\top(\widehat\theta-\theta)|
\le
\sigma\sqrt{c_{d,\delta}}
\sqrt{d^\top A^{-1}d}.
}
\]

したがって reachable directions 全体が低 leverage なら、adaptive choice of \(d_t\) 自体は追加 union bound を要求しない。

### 現在の位置づけ

この uniformity mechanism は有効だが、後の literature audit で **Theorem M は contribution ではなく標準的 confidence-ellipsoid lemma として demote** された。

---

## 23. Sealed audit lifetime

reachable direction set \(\mathcal D_T\) に対して

\[
L_T(A)
=
\sup_{d\in\mathcal D_T}
d^\top A^{-1}d.
\]

\[
L_T(A)
<
\frac{\Gamma^2}{\sigma^2c_{d,\delta}}
\]

なら sealed audit は horizon \(T\) まで有効。

directional impossibility bound も Bretagnolle–Huber で得られ、

\[
d^\top A^{-1}d
\]

が大きすぎる方向では、その sealed audit だけから sign classification は不可能。

したがって sealed audit は「feedback bits を何個見たか」だけで expire するのではなく、**optimizer が audit geometry の high-leverage region へ到達したか**で expire する。

---

## 24. Smooth Gaussian policy-shift law

\(p=N(0,\Sigma)\)、\(q=N(d,\Sigma)\) とし、

\[
M=d^\top\Sigma^{-1}d.
\]

audit mixture の second moment

\[
G_\lambda=\Sigma+\lambda dd^\top.
\]

directional verification difficulty:

\[
\boxed{
V_\lambda
=
d^\top G_\lambda^{-1}d
=
\frac{M}{1+\lambda M}.
}
\]

これは rare-tail の

\[
\lambda_c=\Theta(1/M)
\]

phase transition を smooth full-support Gaussian model で再現する。

### 注意

この result は exogenous / fixed candidate direction としては正しいが、**audit が verifier を fit し、その verifier が candidate を生成する endogenous loop では追加 selection effect がある**。後の Theorem P / U–W がこの不足を修正した。

---

# Part VIII — Endogenous self-evaluation

## 25. Honest plug-in endogenous model

sealed design \(A\),

\[
\widehat\theta=\theta+e,
\qquad
e\sim N(0,\sigma^2A^{-1}),
\]

candidate direction

\[
d=\eta\Sigma\widehat\theta.
\]

realized true linear gain:

\[
\Delta=d^\top\theta,
\]

same verifier による plug-in measured gain:

\[
\widehat\Delta=d^\top\widehat\theta.
\]

exact identities:

\[
\boxed{
\Delta
=
\eta
\left(
\theta^\top\Sigma\theta
+
e^\top\Sigma\theta
\right),
}
\]

\[
\boxed{
\widehat\Delta
=
\eta\|\theta+e\|_\Sigma^2.
}
\]

optimism bias:

\[
\boxed{
\mathbb E[\widehat\Delta-\Delta]
=
\eta\sigma^2
\operatorname{tr}(\Sigma A^{-1}).
}
\]

---

## 26. Theorem U — same-verifier self-evaluation blindness

### 状態

**PROVED in the canonical linear-Gaussian / exponential-tilt plug-in architecture.  
Current central mechanism.**

\[
\boxed{
\widehat\Delta
=
\eta\|\theta+e\|_\Sigma^2
\ge0
\quad
\text{for every realization.}
}
\]

accept iff measured gain \(>0\) という naive rule は全 candidate を accept する。

重要なのは「verifier が noisy だから」ではない。

\[
\boxed{
\text{the loop optimized the same estimate it then uses to certify itself.}
}
\]

そのため measured improvement が norm になる。

この result を一般の全 self-evaluation architecture に universalize してはいけない。scope は plug-in linear architecture。

---

## 27. Theorem V — winner's-curse floor

\(A=nG\) とすると mean optimism bias は

\[
\frac{
\eta\sigma^2
\operatorname{tr}(\Sigma G^{-1})
}{n}.
\]

plug-in certification rule について bias を one margin \(\Gamma\) 未満にするには

\[
\boxed{
n
\gtrsim
\frac{
\eta\sigma^2
\operatorname{tr}(\Sigma G^{-1})
}{\Gamma}.
}
\]

特徴は

\[
\boxed{\Gamma^{-1}}
\]

であり通常の variance-limited \(\Gamma^{-2}\) とは異なる。

### 現在の慎重な解釈

これは **plug-in rule の architecture-specific optimism floor** として扱う。

mean bias だけから arbitrary estimator に対する information-theoretic lower bound を主張してはいけない。

bias correction 後も selection variance term が同じ \(\Gamma^{-1}\) exponent を再現するという exact Gaussian calculationがある。

---

## 28. Bias-corrected endogenous budget law

exact-realizable Gaussian model では概ね

\[
\boxed{
n^\star
\asymp
\max
\left\{
\frac{
\eta^2\sigma^2c_\delta
\theta^\top\Sigma G^{-1}\Sigma\theta
}{
\Gamma^2
},
\;
\frac{
\eta\sigma^2
\sqrt{
c_\delta
\operatorname{tr}[(\Sigma G^{-1})^2]
}
}{
\Gamma
}
\right\}.
}
\]

第一項は signal / \(c\)-optimality。

第二項は endogenous selection / optimism geometry。

ただし well-specified matched design \(G=\Sigma\) では両者の ratio が asymptotically constant order になり、実験で clean に分離できないことも後に確認された。

---

## 29. Theorem W — rank-one candidate mixture saturation

rank-one mixture

\[
G_\lambda
=
\Sigma+\lambda dd^\top
\]

に対して exact spectrum identity:

\[
\boxed{
\operatorname{tr}(\Sigma G_\lambda^{-1})
=
(p-1)+\frac1{1+\lambda M},
}
\]

\[
\boxed{
\operatorname{tr}
[(\Sigma G_\lambda^{-1})^2]
=
(p-1)+\frac1{(1+\lambda M)^2}.
}
\]

candidate mixing は endogenous optimism directions のうち **一方向だけ**を改善し、残り \(p-1\) directions を残す。

したがって最大 improvement factor は

\[
\frac{p}{p-1}
\]

程度。

この result は後に two-audit necessity conjecture を反証するのにも使われた。

---

# Part IX — Misspecification

## 30. Quadratic misspecification model

true reward:

\[
r(y)=\theta^\top y+g(y),
\qquad
g(y)=\frac12y^\top Qy.
\]

linear verifier は \(g\) を表現できない。

candidate direction は依然として

\[
d=\eta\Sigma\widehat\theta.
\]

same plug-in estimate は変わらない:

\[
\widehat\Delta
=
\eta\|\theta+e\|_\Sigma^2
\ge0.
\]

したがって **Theorem U survives misspecification**。

---

## 31. Theorem Y — adverse-curvature zero crossing

nominal \(d_0=\eta\Sigma\theta\) に対して true gain は

\[
\Delta
=
\eta\theta^\top\Sigma\theta
+
\frac12
\eta^2
\theta^\top\Sigma Q\Sigma\theta.
\]

adverse curvature

\[
\theta^\top\Sigma Q\Sigma\theta<0
\]

なら

\[
\boxed{
\eta_{\max}
=
\frac{
2\theta^\top\Sigma\theta
}{
|\theta^\top\Sigma Q\Sigma\theta|
}.
}
\]

そして exact に

\[
\boxed{
\Delta
=
\Gamma
\left(
1-\frac{\eta}{\eta_{\max}}
\right),
\qquad
\Gamma=\eta\theta^\top\Sigma\theta.
}
\]

### 重要な後期修正

\(\eta_{\max}\) は **universal verification ceiling ではない**。

これはその fixed update family で

\[
\boxed{\Delta=0}
\]

になる optimization strength。

past \(\eta_{\max}\) では proposal が harmful なので correct action は reject。

---

## 32. Theorem Z — sealed linear certificate becomes confidently wrong

adverse curvature で \(\eta>\eta_{\max}\) なら

\[
\Delta<0
\]

だが Theorem U により

\[
\widehat\Delta\ge0
\]

always。

audit size \(n\to\infty\) で confidence width は 0 に shrink するため、

\[
\boxed{
\lim_{n\to\infty}
\Pr[
\text{certificate accepts and true gain is negative}
]
=
1.
}
\]

これは「少ない verification では失敗する」という現象ではない。

**同じ misspecified model をより正確に fit すると、wrong certificate がより confident になる**。

---

## 33. Corrected Theorem AG′ — verifiability versus useful optimization

研究途中で「past \(\eta_{\max}\) は any budget でも verify 不可能」という一般 claim を立てたが、これは **false** であった。

現在の correct dichotomy:

### Sealed linear certificate

一定の symmetry / orthogonality 条件下で \(\varrho=\eta/\eta_{\max}>1\) なら

\[
\boxed{
\lim_{n\to\infty}
\Pr[
\text{accept}\wedge\Delta<0
]
=
1.
}
\]

architecture-specific impossibility。

### Fresh model-free post-selection audit

candidate \(q_t\) と baseline \(p_t\) から fresh trusted rewards を取れば

\[
\widehat\Delta^{MC}
=
\bar r_q-\bar r_p
\]

は realized candidate conditional に unbiased。

概ね

\[
\boxed{
m_t
=
O
\left(
\frac{
(V_{q_t}+V_{p_t})
\log(1/\delta)
}{
\Delta_t^2
}
\right)
}
\]

で sign classification できる。

quadratic Gaussian example では far beyond zero crossing:

\[
V_q=O(\eta^2),
\qquad
\Delta^2=\Theta(\eta^4),
\]

so

\[
m=O(\eta^{-2}\log(1/\delta)),
\]

integer floor により \(O(1)\)。

結論:

\[
\boxed{
\eta_{\max}\text{ bounds useful optimization, not verifiability.}
}
\]

harmful proposal が大きくなるほど direct fresh audit はむしろ簡単になる。

### proof caveat

Gaussian-quadratic rewards は unbounded / sub-exponential なので、variance-only finite-sample bound は厳密には robust mean / sub-exponential constants を明示すべき。bounded-label experiments ではこの問題を回避した。

---

# Part X — Capability coupling and auxiliary endogenous-design theory

## 34. Theorem X — aggression coupled to audit budget

仮に system が accumulated audit budget \(n\) に応じて optimization aggression を

\[
\eta(n)=\eta_0n^\beta
\]

と増やす **operational coupling** を置くと、plug-in floor との fixed-point から三 regime:

- \(\beta<1\): sufficiently large budget で stable
- \(\beta=1\): coefficient threshold
- \(\beta>1\): stable budget set が bounded above

が得られる。

### 現在の解釈

これは「more data itself makes things worse」という general claim ではない。

\[
\boxed{
\text{more audit can hurt only because the system is assumed to convert trust into stronger optimization.}
}
\]

この coupling を明示しない「more oversight makes things worse」という headline は撤回済み。

---

## 35. Endogenous audit design O/P/Q/R/S/T — 成果と現在の位置づけ

この系統では audit design \(A\) が verifier error geometry を決め、verifier error が candidate direction を決める endogenous fixed-point を解析した。

主な成果:

### O — signal vs optimism geometry

surrogate stability condition は概念的に

- \(c\)-optimality: nominal contrast / signal を解像する
- \(E\)-optimality: optimizer の confidence / optimism geometry 全体を抑える

の max / joint condition に分かれる。

### 注意

初期 SDP 表現には reciprocal parameterization の注意があり、最終論文では LMI を再監査する必要がある。

### P — candidate-aware saturation floor

candidate mixing が nominal candidate direction を厚くしても transverse directions が残るので endogenous selection term は完全には消えない。

これは後の exact Theorem W により、より明瞭な rank-one spectrum identity に置き換えられた。

### Q — Lazy Candidate-Aware Refresh

elliptical / log-det potential により、fixed-capability linear model では expensive refresh rounds を ambient / effective dimension で bound できる。

### R — matching dimension lower bound

orthogonal directions により \(\Omega(p)\) fresh directional coverage が必要な family を構成。

### S — bounded / sublinear / linear phase

reachable innovation rank \(r(T)\) が

- \(O(1)\)
- \(o(T)\)
- \(\Theta(T)\)

のどれかで fresh-budget growth がそれぞれ bounded / sublinear / linear になる。

### T — myopic versus strategic audit allocation

audit choice が future candidate generation を変える endogenous innovation model では、myopic information value と strategic pre-coverage が分離し得る。

ただしこの whole branch は assumptions が強く、後の novelty audit で paper centrality を下げた。

---

## 36. Stackelberg / complexity branch — AA–AD

secondary theoretical exploration として、

- optimism-driven follower
- auditor as leader
- audit design が follower objective の confidence ellipsoid を変える

Stackelberg formulation を解析した。

### Proposition AA

steering gap と malice price の間に cubic deterrence threshold:

\[
\beta_c
\propto
\left(1-\frac{\chi}{\Gamma}\right)^3.
\]

### Theorem AB

mildly tempted optimizer を deter する extra cost は一定 factor 内に収まり、threshold を越えると \(\Theta(T)\) regime に移る model を得た。

### Proposition AC

follower response の conditional entropy が大きいほど leader commitment advantage が失われる Fano-type bound。

### Theorem AD

indivisible many-to-many audit instruments では minimum-cost steering-set selection が Set Cover に reduce され NP-hard。

coverage-greedy は \(H(K)\) approximation、value-greedy は悪化し得る。

### 現在の位置づけ

数学的には成果だが、central recursive verification paper には peripheral。

Set Cover / Stackelberg branch を main contribution として押すと story が散るので appendix / separate work 候補。

---

# Part XI — Two-audit allocation and its refutation

## 37. Historical two-audit allocation — Theorem AI

historical surrogate model で、

sealed audit size \(n\) が proposal quality を改善し、fresh audit cost

\[
m(n)
=
\frac{\kappa}{
(u-a/\sqrt n)^2
}
\]

とする。

\[
B(n)=n+Tm(n)
\]

を minimize すると

\[
\boxed{
n^\star
=
\left(
\frac{a+A}{u}
\right)^2,
}
\]

\[
\boxed{
m^\star
=
\frac{\kappa}{u^2}
\left(
\frac{a+A}{A}
\right)^2,
}
\]

\[
\boxed{
B^\star
=
\frac{(a+A)^3}{au^2},
\qquad
A=(T\kappa a)^{1/3}.
}
\]

large \(T\) の surrogate arithmetic では

\[
n^\star\sim T^{2/3},
\qquad
n^\star/B^\star\sim T^{-1/3}.
\]

### 現在の位置づけ

これは central theorem ではない。two-stage allocation arithmetic。

さらに historical Theorem AJ の「split ratio が \(\varrho\) independent」は、\(\kappa\) 自体が \(\eta\) / \(\varrho\) に依存し得るため **fixed-\(\kappa\) conditional statement にしかならない**。

historical Corollary AK も whole-run cost は一般には

\[
\sum_t \kappa_t u_t^{-2}
\]

であり fixed \(\kappa\) の schedule law を universalize してはいけない。

---

## 38. Two-audit necessity conjecture — REFUTED

一時、

> sealed / broad audit と fresh candidate audit は統計的に異なる二 resource で、単一 audit distribution では両方を制御できない

という uncertainty-principle 型 conjecture を central theorem 候補にした。

### 結果

**False. Direction closed.**

balanced mixture

\[
\boxed{
\mu_{1/2}
=
\frac12(p+q)
}
\]

が両方を同時にかなり良く制御する counterexample になった。

Theorem W を reverse に読むと candidate mass を足す selection-side cost は \(p\) directions のうち高々一方向分。

two-sample \(p/q\) allocation class では balanced allocation の variance cost は dedicated optimal allocation に対して **一般に高々 factor 2**。

v15 にある約 1.7 は tested variance ratios での数値であり universal constant としては 2 を使う方が安全。

unconstrained importance-audit optimum に対する universal constant-factor theorem は現在ない。Gaussian numerical checks のみ。

### 結論

\[
\boxed{
\text{two structurally separate audit streams are not statistically necessary.}
}
\]

残るのは temporal ordering:

candidate を作る情報は \(q_t\) より前に必要、candidate を post-selectively check する fresh data は \(q_t\) 固定後に取る。

---

## 39. SOAR-1 / single-stream interpretation

two-audit necessity の refutation 後、conceptual policy は

\[
\mu_{1/2}
=
\tfrac12(p_t+q_t)
\]

から round-wise trusted labels を取り、

1. current candidate の direct decision statistic に使う
2. 次 round verifier の accumulated design にも追加する

single-stream reuse に簡略化できる。

### 未解決 rigor

historical SOAR guarantee には次の問題が残る。

- sequential doubling stages だけでなく rounds \(t\) にも familywise \(\delta\) allocation が必要
- \(\Gamma_t=\eta_t\theta^\top\Sigma\theta\) は unknown \(\theta\) を含み一般には known ではない
- initial \(u_1\) lower bound の implementability
- adaptive design には sealed fixed-design uniformity ではなく self-normalized concentration が必要
- unbounded rewards なら robust concentration を明示する必要

したがって SOAR / SOAR-1 は **algorithmic implication / prototype** であり、現在の central proved guarantee ではない。

---

# Part XII — 重要な自己監査・撤回の履歴

## 40. Reject-all loophole

**修正済み。**

safety-only lower bound は trivial reject-all algorithm で破れる。

familywise correct classification または safety + power を objective に変更。

---

## 41. Fixed-budget と sequential expected cost の混同

**修正済み。**

\[
M\log(1/\delta)
\]

は fixed high-probability latency。

sequential expected discovery は

\[
\Theta(M).
\]

---

## 42. Raw \(M\) as universal law

**撤回 / 一般化済み。**

unstructured class では \(M\)-tax。

structured verifier class では restricted geometry

\[
d^\top\Sigma_\mu^\dagger d
\]

等に置き換わり、large density ratio でも cheap verification が可能。

---

## 43. Generic post-selection / DP / max-information novelty

**novelty claim から除外。**

adaptive data analysis の generic machinery は existing theory。

本研究の value は recursive optimizer-induced shift と verification budget を結びつけることに置く。

---

## 44. Theorem M novelty

**demoted.**

uniform confidence ellipsoid は有用だが standard lemma として引用する位置。

---

## 45. Winner's-curse lower-bound overclaim

mean optimism bias だけを information-theoretic impossibility として扱わない。

現在は plug-in architecture に scoped。

---

## 46. “Effective margin” terminology

**retracted.**

\[
\Gamma(1-\eta/\eta_{\max})
\]

は effective margin ではなく **true gain \(\Delta\) そのもの**。

---

## 47. \(\eta_{\max}\) as universal verification ceiling

**retracted.**

\(\eta_{\max}\) は beneficial-to-harmful zero crossing。

fresh post-selection audit は beyond crossing でも reject 可能。

---

## 48. “More oversight makes things worse”

**general claim retracted.**

成立するのは \(\eta(n)\) が audit coverage に応じて増える operational coupling を置いた場合のみ。

---

## 49. Two-audit structural necessity

**refuted.**

balanced mixture counterexample。

---

## 50. One-step threshold implies inevitable recursive collapse

**2026-09-11 experiments により修正。**

current-policy refresh が verifier projection 自体を変えるため、十分高速な refresh では collapse を回避可能。

現在の thesis は **staleness / lag dependent**。

---

# Part XIII — Empirical program

## 51. Initial finite-arm sanity checks

adaptive rare-tail family で equal \(M\) の current-policy query cap がほぼ linear in \(M\) になることを numerical check。

\(T=10,\delta=0.05\) では candidate audit の per-round cap がほぼ constant の一方、current-policy audit は \(M\) に比例。

mixed audit でも scaling variable \(\lambda M\) の crossover around order 1 を確認。

これは theorem sanity check であり non-toy evidence ではない。

---

# Part XIV — Progress Update XV: non-Gaussian misspecification experiment

## 52. Setup

Gaussian-quadratic analytic model から外すため、

- finite outcome space: 401 points
- asymmetric discrete-Laplace base policy
- bounded nonlinear reward
- stochastic trusted labels \(Y\in\{-1,+1\}\)
- actually fitted misspecified linear verifier
- HC1 robust standard error
- exponential candidate tilt

を使用。

main true reward:

\[
r_A(y)
=
\tanh(y-0.30y^2-0.01y^4).
\]

---

## 53. Main result

population true-gain sign threshold:

\[
\boxed{
\eta_\star
=
3.497506
}
\]

trusted budget を大きく変えても 50% confident-wrong threshold center はほぼ固定。

transition width regression:

\[
\boxed{
\text{width}\propto n^{-0.515}.
}
\]

つまり

\[
\boxed{
\text{more verification sharpens the transition but does not materially move its center.}
}
\]

### Main table

| n | replications | slope_mean | slope_sd | eta_10pct_wrong | eta_50pct_wrong | eta_90pct_wrong | transition_width_10_90 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50.0000 | 3000.0000 | 0.3040 | 0.1237 | 2.2228 | 3.4458 | — | — |
| 200.0000 | 3000.0000 | 0.2955 | 0.0587 | 2.7543 | 3.4565 | 4.6500 | 1.8957 |
| 1000.0000 | 2000.0000 | 0.2930 | 0.0257 | 3.1280 | 3.5029 | 3.9103 | 0.7823 |
| 5000.0000 | 1000.0000 | 0.2931 | 0.0115 | 3.3192 | 3.4901 | 3.6750 | 0.3558 |
| 20000.0000 | 500.0000 | 0.2925 | 0.0058 | 3.4125 | 3.4968 | 3.5867 | 0.1742 |

---

## 54. Hidden-cliff robustness

別 nonlinear misspecification:

\[
r_B(y)
=
\tanh
\left(
0.8y
-
0.8\max(y-1,0)^2
\right).
\]

population threshold:

\[
\eta_\star\approx3.422735.
\]

transition width:

\[
\text{width}\propto n^{-0.507}.
\]

同じ qualitative signature。

---

## 55. Fresh post-selection control

threshold のすぐ近くでは \(|\Delta|\) が小さいため direct audit は非常に expensive。

farther past crossing では harmful magnitude が増えるので required fresh samples は急減。

この結果は

\[
\boxed{
\text{harmful candidate is not intrinsically unverifiable;
the self-model is the problem.}
}
\]

という AG′ interpretation を支持。

---

# Part XV — Progress Update XVI: finite-domain program synthesis

## 56. Environment

input domain:

\[
x\in\{0,\dots,255\}.
\]

target:

```python
def target(x):
    return (x*x + 3*x + 7) % 11
```

32 public tests。

candidate pool:

- 180 regular approximate programs
- 60 public-test exploit / memorizing programs
- 3 exact programs

true reward = exhaustive semantic correctness on all 256 inputs。

verifier feature = public-test pass rate。

trusted labels = random program × random hidden semantic input の correctness。

---

## 57. Program-synthesis threshold

population sign threshold:

\[
\boxed{
\eta_\star
=
24.608329.
}
\]

trusted budget \(n=200\to20000\) の 100x change に対し、50% wrong threshold の range は約

\[
\boxed{0.196}
\]

のみ。

transition width exponent:

\[
\boxed{
-0.525.
}
\]

### Table

| n | replications | beta1_mean | beta1_sd | self_certificate_fire_prob | eta_10pct_wrong | eta_50pct_wrong | eta_90pct_wrong | transition_width_10_90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200.0000 | 2200.0000 | 0.5286 | 0.1922 | 0.8518 | 16.7500 | 24.7019 | — | — |
| 1000.0000 | 1600.0000 | 0.5300 | 0.0857 | 1.0000 | 20.3816 | 24.5312 | 31.1667 | 10.7851 |
| 5000.0000 | 800.0000 | 0.5311 | 0.0380 | 1.0000 | 22.4625 | 24.5179 | 27.0909 | 4.6284 |
| 20000.0000 | 400.0000 | 0.5282 | 0.0191 | 1.0000 | 23.5417 | 24.7135 | 25.7812 | 2.2396 |

これは misspecification が抽象 quadratic term ではなく **public-test overfitting versus exhaustive semantics** から生じても同じ現象が出ることを示した。

---

# Part XVI — Progress Update XVII: learned autoregressive code generator

## 58. Candidate distribution を learned generator へ

procedural candidate pool の uniform / hand-weighted policy から一段進め、小型 GRU autoregressive DSL code generator を実際に train し、その learned program distribution を \(p\) として使用。

これは pretrained LLM ではないが、candidate policy が learned generative model になった。

---

## 59. 重要な negative result: 最初の generator では failure が起きなかった

最初の learned-generator setting では public proxy frontier に high-quality regular programs が十分残っており、strong optimization でも true performance collapse が起きなかった。

この failure-to-fail から新しい condition が分かった。

\[
\boxed{
\text{exploit exists}
\not\Rightarrow
\text{overoptimization collapse}.
}
\]

必要なのは、optimizer が集中する **proxy frontier の conditional true quality が悪いこと**。

これをここでは frontier-occupancy condition と呼んでいるが、まだ formal theorem / metric ではない。

---

## 60. Stress-test generator

generator distribution を修正して exploit programs が proxy frontier を占める setting を作ると、misspecified verifier で

\[
\boxed{
\eta_\star
\approx
22.54.
}
\]

budget \(n=200,1000,5000,20000\) に対して 50% threshold はほぼ 22.5 前後。

transition width:

\[
\boxed{
\text{width}\propto n^{-0.531}
}
\]

程度。

---

## 61. Best-of-\(N\) robustness

optimizer を exponential tilt から Best-of-\(N\) に変更しても rise-then-fall が出た。

true performance は small \(N\) で improve し、その後 strong selection で baseline 以下に落ちる。

実験記録では peak around

\[
N\approx6
\]

harmful crossing around

\[
N\approx34\text{--}35
\]

を観測。

したがって threshold phenomenon は exponential / KL algebra だけの artifact ではない。

---

# Part XVII — Progress Update XVIII: representation × budget

## 62. 二つの resource を独立に変える

trusted sample budget:

\[
n\in\{500,2000,10000\}.
\]

verifier representation:

- proxy-only
- weak exploit signature
- medium signature
- strong signature
- near-exact signature
- exact exploit flag

signature は controlled static program feature であり real-world static analyzer claim ではない。

---

## 63. Population representation phase

| verifier_feature | signature_noise_sd | population_eta_star | max_true_gain | eta_at_max_gain | true_gain_eta250 | proxy_gain_eta250 |
| --- | --- | --- | --- | --- | --- | --- |
| proxy_only | — | 22.5409 | 0.0831 | 9.1000 | -0.1320 | 0.2799 |
| weak_signature | 4.0000 | 22.6224 | 0.0832 | 9.1000 | -0.1295 | 0.2799 |
| medium_signature | 1.0000 | 26.1460 | 0.0878 | 9.8000 | -0.0567 | 0.2889 |
| strong_signature | 0.3500 | ∞ | 0.3024 | 163.4000 | 0.3000 | 0.3110 |
| near_exact_signature | 0.1000 | ∞ | 0.2912 | 102.5000 | 0.2792 | 0.3626 |
| exact_exploit_flag | 0.0000 | ∞ | 0.4283 | 250.0000 | 0.4283 | 0.3587 |

主要結果:

- proxy only: \(\eta_\star\approx22.54\)
- weak signature:ほぼ同じ
- medium signature: \(\eta_\star\approx26.15\)
- strong signature: tested \(\eta\le250\) で harmful crossing 消失
- exact exploit representation: crossing 消失

したがって

\[
\boxed{
\text{budget changes width; representation changes center.}
}
\]

より正確には、

\[
\boxed{
\text{sample quantity controls uncertainty around a population projection;
function class controls what that projection can represent.}
}
\]

---

## 64. Finite-sample representation control

proxy-only class では \(n\) を増やしても threshold center はほぼ固定し transition が sharpen。

strong representation では \(n\) を増やすほど finite-sample harmful proposals が消える。

つまり

### estimation-limited

model class に exploit signal が含まれる:

\[
n\uparrow
\Rightarrow
\text{failure}\downarrow.
\]

### misspecification-limited

model class が exploit を表現できない:

\[
n\uparrow
\Rightarrow
\text{wrong population projection への confidence}\uparrow.
\]

---

# Part XVIII — Progress Update XIX: genuinely recursive experiment

## 65. 一歩 threshold から multi-round recursion へ

learned DSL generator 上で、

- fixed \(\eta_{\rm step}=3\)
- repeated policy updates
- verifier refresh cadence を変更

した。

比較:

- sealed verifier
- current-policy refresh
- fresh post-selection guard
- oracle guard

---

## 66. Sealed versus current refresh

initial semantic reward:

\[
0.571182.
\]

finite \(n=1000\) summary:

| strategy | n_train | m_fresh | replications | final_mean_true_reward | final_below_baseline_fraction | ever_harmful_accepted_fraction | mean_num_updates | median_stop_round |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sealed_self | 1000 | 0 | 500 | 0.4797 | 1.0000 | 1.0000 | 14.0000 | 15.0000 |
| current_self | 1000 | 0 | 500 | 0.6523 | 0.0000 | 0.0340 | 3.9200 | 5.0000 |
| fresh_guard | 1000 | 1000 | 500 | 0.6100 | 0.0000 | 0.0000 | 0.9700 | 2.0000 |
| oracle | 1000 | 0 | 500 | 0.6540 | 0.0000 | 0.0000 | 5.8240 | 7.0000 |

主要結果:

### sealed self-certificate

14 updates をほぼ全部受け入れ、

\[
0.571
\to
0.480
\]

程度まで degradation。

ever harmful accepted:

\[
100\%.
\]

final below baseline:

\[
100\%.
\]

### current-policy refresh

verifier を current policy で毎回 refit すると、exploit mass が増えるにつれて proxy slope 自体が shrink。

population recursion は around

\[
0.6543
\]

で plateau。

finite sample でも final below baseline は 0%。

### fresh guard

harmful acceptance 0% だが fixed fresh budget では conservative で、oracle より早く停止。

---

## 67. 重要な scientific correction

この experiment は次の broad claim を否定した。

> misspecified verifier があれば recursive system は eventual に one-step \(\eta_{\max}\) を越えて collapse する。

これは一般には false。

current-policy refresh は reached distribution 上で projection 自体を変えるので、optimizer aggression を self-dampen できる。

最新のより良い claim:

\[
\boxed{
\text{misspecification alone is insufficient;
stale optimization exposure is the key recursive variable.}
}
\]

---

## 68. Refresh cadence phase

\(n=1000\), 400 Monte Carlo runs。

| refresh_interval | n | replications | mean_final_reward | mean_minimum_reward | final_below_initial_fraction | ever_below_initial_fraction | ever_harmful_update_fraction | mean_harmful_update_count | mean_num_updates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | 1000.0000 | 400.0000 | 0.6522 | 0.5712 | 0.0000 | 0.0000 | 0.0375 | 0.0375 | 3.9200 |
| 2.0000 | 1000.0000 | 400.0000 | 0.6530 | 0.5712 | 0.0000 | 0.0000 | 0.1725 | 0.2750 | 4.2200 |
| 3.0000 | 1000.0000 | 400.0000 | 0.6533 | 0.5712 | 0.0000 | 0.0000 | 0.2475 | 0.7375 | 4.4400 |
| 4.0000 | 1000.0000 | 400.0000 | 0.6496 | 0.5711 | 0.0000 | 0.0050 | 0.9175 | 5.1100 | 13.2800 |
| 6.0000 | 1000.0000 | 400.0000 | 0.5428 | 0.4344 | 0.5500 | 0.9650 | 1.0000 | 11.5925 | 23.2800 |
| 8.0000 | 1000.0000 | 400.0000 | 0.5185 | 0.2697 | 0.5075 | 1.0000 | 1.0000 | 10.8650 | 23.6600 |
| 12.0000 | 1000.0000 | 400.0000 | 0.2076 | 0.2076 | 1.0000 | 1.0000 | 1.0000 | 18.2475 | 24.0000 |
| 24.0000 | 1000.0000 | 400.0000 | 0.4486 | 0.4486 | 1.0000 | 1.0000 | 1.0000 | 20.9175 | 24.0000 |

特に

- refresh every 1–3 rounds: baseline below 0%
- every 4 rounds: 0.5%
- every 6 rounds: 96.5%
- every 8+ rounds: ~100%

という sharp empirical phase。

この時点で project の中心 empirical question は

> **How much policy optimization can occur before the verifier must be refreshed?**

へ変化した。

---

# Part XIX — Progress Update XX: 2D recursive verification-lag law

## 69. Exact stale-block identity

frozen verifier \(v\) の下で

\[
p_{k+1}(y)
\propto
p_k(y)e^{\eta v(y)}
\]

を \(L\) 回繰り返すと

\[
\boxed{
p_L(y)
=
\frac{
p_0(y)e^{L\eta v(y)}
}{
\mathbb E_{p_0}[e^{L\eta v}]
}.
}
\]

よって stale block 内では

\[
\boxed{
L\text{ steps of strength }\eta
\equiv
1\text{ step of strength }L\eta.
}
\]

この identity は elementary で novelty claim ではないが、recursive verification-lag coordinate の根拠になる。

---

## 70. Population \(\eta\times L\) phase

grid:

\[
\eta\in\{0.5,0.75,\dots,6.0\},
\qquad
L\in\{1,\dots,16\},
\qquad
T=24.
\]

各 \(\eta\) で最初に baseline collapse が起きる \(L\) を見ると、critical product は概ね

\[
\boxed{
\eta L\approx17\text{--}18.
}
\]

median:

\[
17.5.
\]

IQR:

\[
16.5\text{--}18.0.
\]

single threshold on \(\eta L\) で population grid の collapse / non-collapse を

\[
\boxed{
98.6\%
}
\]

程度 classification。

### Boundary table

| eta_step | critical_refresh_interval | critical_etaL | critical_score_span_exposure | critical_block_KL |
| --- | --- | --- | --- | --- |
| 0.500 | — | — | — | — |
| 0.750 | — | — | — | — |
| 1.000 | — | — | — | — |
| 1.250 | — | — | — | — |
| 1.500 | 11.000 | 16.500 | 11.253 | 2.494 |
| 1.750 | 10.000 | 17.500 | 13.451 | 3.697 |
| 2.000 | 8.000 | 16.000 | 10.207 | 2.002 |
| 2.250 | 8.000 | 18.000 | 14.601 | 4.396 |
| 2.500 | 7.000 | 17.500 | 13.451 | 3.697 |
| 2.750 | 6.000 | 16.500 | 11.253 | 2.494 |
| 3.000 | 6.000 | 18.000 | 14.601 | 4.396 |
| 3.250 | 5.000 | 16.250 | 10.726 | 2.239 |
| 3.500 | 5.000 | 17.500 | 13.451 | 3.697 |
| 3.750 | 5.000 | 18.750 | 16.387 | 5.529 |
| 4.000 | 4.000 | 16.000 | 10.207 | 2.002 |
| 4.250 | 4.000 | 17.000 | 12.335 | 3.060 |
| 4.500 | 4.000 | 18.000 | 14.601 | 4.396 |
| 4.750 | 4.000 | 19.000 | 17.001 | 5.922 |
| 5.000 | 4.000 | 20.000 | 19.521 | 7.495 |
| 5.250 | 4.000 | 21.000 | 22.163 | 9.000 |
| 5.500 | 3.000 | 16.500 | 11.253 | 2.494 |
| 5.750 | 3.000 | 17.250 | 12.888 | 3.370 |
| 6.000 | 3.000 | 18.000 | 14.601 | 4.396 |

---

## 71. Alternative shift coordinates

run-level scalar comparison:

| scalar | best_threshold | classification_accuracy |
| --- | --- | --- |
| eta_times_L | 15.8750 | 0.9864 |
| max_score_span_exposure | 10.1211 | 0.9918 |
| max_block_KL | 1.9061 | 1.0000 |

`max_block_KL` はこの grid では perfect classification したが、trajectory 後半の事後情報を含むため predictive theorem とは言えない。

block-level pre/post refresh analysis では、

- raw \(\eta L\)
- score-span exposure
- endpoint KL

はいずれも baseline-crossing block classification around 94–95%。

したがって現状は

\[
\boxed{
\eta L
}
\]

が excellent first-order coordinate、KL / log density ratio が cross-system invariant candidate。

---

## 72. Finite-sample 2D phase

\(n=1000\), 150 runs/cell。

\[
\eta\in\{2,3,4,5,6\}
\]

で 50% failure を初めて超える \(L\):

| eta_step | first_L_with_failure_prob_ge_0.5 | etaL_at_boundary | failure_probability |
| --- | --- | --- | --- |
| 2.000 | 10.000 | 20.000 | 0.967 |
| 3.000 | 6.000 | 18.000 | 0.973 |
| 4.000 | 4.000 | 16.000 | 0.893 |
| 5.000 | 3.000 | 15.000 | 0.660 |
| 6.000 | 3.000 | 18.000 | 0.993 |

corresponding \(\eta L\):

\[
20,\ 18,\ 16,\ 15,\ 18.
\]

median near 18。

failure probability と \(\eta L\) rank correlation:

\[
\boxed{
0.914.
}
\]

single \(\eta L\) threshold による 50%-failure-cell classification:

\[
\boxed{
95\%.
}
\]

これは population artifact ではなく finite trusted-data noise 下でも phase collapse が残る evidence。

---

## 73. Strict monotonicity vs trajectory safety

small \(L\) でも occasional harmful update は起こる。

しかし次の refresh で repair され、trajectory 全体が initial baseline を割らない場合がある。

したがって safety criterion を区別する必要:

### Strict update monotonicity

\[
\Delta_t\ge0
\]

for every accepted update.

### Deployment-baseline / trajectory safety

\[
\min_t R(p_t)\ge R(p_0).
\]

今回 \(\eta L\) phase は第二 criterion でより sharp。

paper ではどちらを扱うか明示する必要。

---

## 74. KL-triggered adaptive refresh — useful negative result

policy drift from last refresh が KL cap を超えたら refresh する heuristic をテスト。

population では fixed cadence より少ない fits で safe に見えた。

しかし finite sample + significance stopping では advantage が消失。

| policy_label | mean_failure_fraction | mean_harmful_fraction | mean_refresh_count | mean_trusted_labels | mean_final_reward | mean_safe_probability |
| --- | --- | --- | --- | --- | --- | --- |
| KL cap 0.50 | 0.0000 | 0.7455 | 4.3018 | 4301.8182 | 0.6493 | 1.0000 |
| KL cap 0.75 | 0.0655 | 0.4236 | 3.2064 | 3206.3636 | 0.6506 | 0.9345 |
| KL cap 1.00 | 0.1255 | 0.7264 | 3.9100 | 3910.0000 | 0.6453 | 0.8745 |
| fixed_L1 | 0.0000 | 0.0527 | 4.4582 | 4458.1818 | 0.6526 | 1.0000 |
| fixed_L2 | 0.0000 | 0.3300 | 3.4036 | 3403.6364 | 0.6526 | 1.0000 |
| fixed_L3 | 0.3227 | 0.6691 | 4.9591 | 4959.0909 | 0.6250 | 0.6773 |

safe KL cap 0.50:

約 4302 trusted labels。

fixed \(L=2\):

約 3404 trusted labels で同じ tested safety。

したがって

\[
\boxed{
\text{KL-triggered refresh is not currently a contribution.}
}
\]

simple adaptive heuristic の population advantage は finite-sample protocol で再現しなかった。

---

# Part XX — 理論と実験の整合

## 75. Worst-case moving-tail theorem と current-refresh success は矛盾しない

Theorem A′ は **adaptive adversarial hard family**。

各 round に previous transcript と独立な fresh hidden reward sign を持つ candidate-salient region が現れる。

この family では current-policy refresh は新しい tail を十分な確率で見るために \(M\)-factor を払う。

一方 learned DSL experiment では reward structure が共有され、current policy が exploit region に移るほど verifier fit にその region の情報が入る。

したがって current refresh が自己修正する。

両者を統一すると:

\[
\boxed{
\text{fresh verification rate is determined by how fast the optimizer creates
new, poorly covered, statistically non-redundant verification directions.}
}
\]

これは研究初期に想定した「毎 round fresh」より一般的で、bounded/sublinear/linear phase results と consistent。

---

## 76. One-step misspecification threshold と recursive \(\eta L\) threshold の関係

initial learned-generator one-step threshold:

\[
\eta_\star\approx22.54.
\]

recursive baseline-collapse boundary:

\[
\eta L\approx17\text{--}18.
\]

contradiction ではない。

refresh 後には

- current policy
- local verifier projection
- local true margin
- exploit mass

が変わるため local one-step zero crossing も state-dependent。

recursive system を一つの static \(\eta_{\max}\) だけで記述できない。

---

# Part XXI — 現在の「成立していること」一覧

## 77. Strong / central candidates

### A. Adaptive moving-tail minimax separation

worst-case recursive family で current audit と candidate-aware audit の \(\Theta(M)\) separation。

### B. Structured verification geometry

raw density-ratio tax は verifier / reward class geometry により restricted Mahalanobis / restricted-\(\chi^2\) quantity に soften する。

### C. Theorem U

same learned verifier を optimize して same plug-in verifier で self-certify すると measured gain が norm になり zero rejection power。

### D. Theorem W

rank-one candidate mixing は endogenous optimism geometry の一方向しか除去しない exact spectrum identity。

### E. Misspecification architecture failure

sealed-linear plug-in certificate は adverse misspecification past true-gain zero crossing で confidently wrong。

### F. Fresh post-selection escape route

fresh model-free candidate/baseline samples は realized candidate の true sign を直接 estimate できる。

### G. Recursive verification-lag empirical law

learned generator で failure boundary が optimizer strength × verifier staleness に沿って collapse。

---

## 78. Strong but noncentral / standard-adjacent

- Bhattacharyya information law
- finite reachable-class uniform verification
- max-information / adaptive reuse bounds
- confidence-ellipsoid sealed reuse
- Fano external-information lower bound
- orthogonal innovation trichotomy
- log-det / elliptical potential refresh accounting
- optimal audit allocation arithmetic
- Set Cover / Stackelberg branch

これらは paper の technical support / appendix には使えるが、単独 novelty としては弱いものを含む。

---

## 79. Empirically supported, not yet theorem-level general laws

- threshold center fixed while width shrinks \(n^{-1/2}\) in several controlled settings
- verifier expressiveness moves / removes threshold
- frontier occupancy is needed for overoptimization collapse
- current-policy refresh can self-correct
- stale refresh cadence induces phase transition
- \(\eta L\) collapses learned-generator recursive phase over tested range

---

## 80. Explicitly not established

- real pretrained LLM / code-model replication
- universal \(\eta_{\max}\)
- universal \(\eta L\) critical constant
- universal KL trigger
- universal necessity of two audit streams
- any claim that more data alone worsens safety
- general theorem that all misspecified verifiers exhibit a sharp scalar threshold
- globally minimax optimal audit policy across all distributions / shared-data strategies
- current SOAR-1 familywise proof under all adaptive rounds and unknown parameters

---

# Part XXII — Novelty boundary

## 81. Claims not to sell as novel

literature audit により次を standalone novelty としては避ける。

- reward hacking exists
- fixed verifiers fail under enough optimization
- evaluate the optimized policy
- active evaluation allocation
- ordinary concentrability / restricted \(\chi^2\)
- generic adaptive-data-analysis / reusable holdout
- generic winner's curse after selection
- verification is a budgeted resource
- generic rank / eluder-dimension lifelong sample complexity
- generic min-cost set cover

---

## 82. Defensible novelty target

現時点で最も defensible なのは次の intersection。

\[
\boxed{
\text{optimizer-induced policy shift}
+
\text{recursive verifier reuse / staleness}
+
\text{fresh trusted-verification complexity}
}
\]

theoretical sideでは moving-tail minimax and endogenous self-evaluation mechanism。

empirical sideでは verifier staleness × optimization movement の phase law。

---

# Part XXIII — 現在の paper story

## 83. Recommended high-level framing

working title candidates:

- **Fresh Verification Budgets for Recursive Self-Improvement**
- **When Does Recursive Self-Improvement Need Fresh Verification?**
- **Verification Must Chase the Policy**
- **Recursive Verification Lag**
- **The Freshness Frontier: Statistical Limits of Reusing Verification in Self-Improving Systems**

現時点では後半の experiments を考えると、

> **Recursive Verification Lag: When Must Verification Catch Up with a Self-Improving Policy?**

のような framing も自然。

---

## 84. Current one-sentence thesis

> **Recursive self-improvement does not require fresh verification merely because time passes; it requires new trusted information when optimization moves into statistically new or poorly covered directions faster than the verifier can be refreshed or structurally generalize.**

補助 mechanism:

> **When the same misspecified verifier is both optimized and used to certify its own update, increased data can sharpen confidence without restoring rejection power.**

---

## 85. Suggested compact theorem package

paper を絞るなら:

1. **Adaptive moving-tail lower bound + minimax budget**
2. **Structured coverage version / smooth generalization**
3. **Theorem U + exact endogenous optimism identity**
4. **Misspecified self-certificate failure + fresh post-selection correction**

Theorem W は 3–4 の bridge。

AI/Stackelberg/Set-Cover 等は implications / appendix へ。

最新 empirical \(\eta L\) phase は theorem ではないため、main empirical finding として扱う。

---

# Part XXIV — 現時点の conference-level assessment

## 86. Scientific level

研究素材としては strong PhD / early-postdoc theory project level。

Main-track candidate として十分な中身はあるが、submission readiness は real-model experiment と proof audit に依存。

概念的 strength:

- clear question
- multiple matching lower/upper laws
- self-correction / retraction history
- falsifiable predictions
- controlled empirical mechanisms

弱点:

- theory branches が多く、最終 paper に圧縮が必要
- central endogenous results は architecture-specific
- real pretrained LLM evidence が未実施
- some auxiliary results have rigor / novelty caveats

---

## 87. Venue impression

現時点の目安:

| Venue level | Current assessment |
|---|---|
| AISTATS / UAI | 強い候補 |
| ICML / NeurIPS / ICLR Main | 十分射程 |
| Spotlight | real-model experiment またはさらに clean な unifying law が必要 |
| Oral | 現時点では不足 |
| COLT | theorem novelty / rigor をさらに厳しく整理する必要 |

保証ではなく research-positioning estimate。

---

# Part XXV — 次にやるべきこと

## 88. 最優先 empirical task

実 pretrained small code LM を external compute で使用。

候補:

- TinyStarCoderPy class
- Qwen2.5-Coder-0.5B class

本環境では `transformers` / external model weights の取得ができず未実施。

### Protocol

1. 50–200 finite-domain coding tasks
2. candidate bank を pretrained code LM から生成
3. cheap verifier = public tests / learned proxy
4. trusted reward = hidden / exhaustive stronger tests
5. vary optimization strength independently
6. vary verifier refresh interval independently
7. vary trusted-label budget
8. richer verifier representation control
9. Best-of-\(N\) と exponential / soft selection
10. compare collapse coordinates:
   - \(\eta L\)
   - cumulative KL since refresh
   - max log density ratio
   - candidate-salient mass shift

決定的問い:

\[
\boxed{
\text{Does one policy-shift coordinate predict when the verifier must be refreshed
across score rescaling, representation changes, and optimizer families?}
}
\]

---

## 89. Theory task — only after empirical invariance test

新 theorem を増やす前に、empirical phase が cross-setting で invariant か確認。

もし KL / density-ratio / restricted geometry のいずれかで collapse するなら、それを用いて recursive lag theorem を構築。

collapse しないなら \(\eta L\) は environment-specific empirical law として留める。

---

# Part XXVI — Reproducibility / artifact inventory

## 90. Core source notes

- `recursive_verification_research_note_20260910_v15.md`
- `recursive_verification_progress_20260911_fd_experiment.md`
- `recursive_verification_progress_20260911_program_synthesis.md`
- `recursive_verification_progress_20260911_learned_code_generator.md`
- `recursive_verification_progress_20260911_representation_phase.md`
- `recursive_verification_progress_20260911_recursive_lag.md`
- `recursive_verification_progress_20260911_2d_phase.md`

---

## 91. Key figures

### Misspecification threshold
- `fd_confident_wrong_curves.png`
- `fd_threshold_vs_budget.png`
- `fd_hidden_cliff_robustness.png`

### Program synthesis
- `program_synthesis_confident_wrong_curves.png`
- `program_synthesis_threshold_vs_budget.png`
- `program_synthesis_public_suite_robustness.png`

### Learned generator
- `learned_dsl_threshold_curves_v2.png`
- `learned_dsl_threshold_vs_budget_v2.png`
- `learned_dsl_bestofn_v2.png`

### Representation
- `learned_generator_representation_population.png`
- `learned_generator_representation_eta_star.png`
- `learned_generator_representation_budget_2d.png`
- `learned_generator_representation_curves_n2000.png`

### Recursive verification lag
- `recursive_population_comparison.png`
- `recursive_strategy_comparison.png`
- `recursive_current_refresh_budget.png`
- `recursive_refresh_interval_phase.png`
- `recursive_refresh_interval_mc_phase.png`

### 2D phase
- `recursive_eta_refresh_population_heatmap.png`
- `recursive_etaL_population_collapse.png`
- `recursive_eta_refresh_finite_heatmap.png`
- `recursive_etaL_finite_collapse.png`
- `recursive_kl_trigger_finite_sample_pareto_v2.png`

---

## 92. Key raw-data files

- `fd_main_threshold_summary.csv`
- `fd_budget_sweep.csv`
- `fd_fresh_audit_control.csv`
- `program_synthesis_threshold_summary.csv`
- `program_synthesis_budget_sweep.csv`
- `program_synthesis_fresh_audit_control.csv`
- `learned_dsl_misspecified_threshold_v2.csv`
- `learned_dsl_rich_control_v2.csv`
- `learned_dsl_bestofn_exact_1_200.csv`
- `learned_generator_representation_population.csv`
- `learned_generator_representation_budget_2d.csv`
- `recursive_strategy_summary.csv`
- `recursive_refresh_interval_mc_summary.csv`
- `recursive_eta_refresh_population_phase.csv`
- `recursive_eta_refresh_population_boundary.csv`
- `recursive_eta_refresh_finite_phase.csv`
- `recursive_eta_refresh_finite_boundary.csv`
- `recursive_stale_block_records.csv`
- `recursive_stale_block_classifier_metrics.csv`
- `recursive_kl_trigger_finite_sample_summary_v2.csv`

---

# Part XXVII — Final current synthesis

## 93. What the project has actually learned

The research started from a simple thesis:

> verification must chase the policy.

That thesis survived, but in a substantially more precise form.

### First refinement

Worst-case adaptive moving tails can make current-distribution verification pay a factor proportional to candidate amplification.

### Second refinement

That factor is not universal; structured verifier classes can extrapolate and replace raw density ratios by restricted feature geometry.

### Third refinement

Fresh data are not automatically required every round; reusable confidence is possible when the reachable verification class saturates.

### Fourth refinement

A self-improving loop that optimizes a learned verifier and evaluates itself with the same plug-in verifier has an endogenous selection problem: its measured improvement can be structurally non-negative.

### Fifth refinement

Misspecification can make that self-certificate confidently wrong even with enormous audit budgets, but the harmful update itself remains detectable with fresh model-free post-selection evidence.

### Sixth refinement

A separate sealed audit and fresh candidate audit are not fundamentally two different statistical resources; balanced / single-stream mixtures can serve both roles.

### Seventh and latest refinement

The one-step misspecification threshold does not imply inevitable recursive collapse. Current-policy refresh can modify the verifier projection and stop the optimizer before collapse.

The genuinely recursive quantity is therefore **lag**:

\[
\boxed{
\text{how much optimizer-induced policy movement accumulates before verification catches up.}
}
\]

In the learned generator experiment, that lag is well approximated by

\[
\boxed{\eta L}
\]

because exponential updates compose exactly under a stale verifier.

The latest evidence therefore points to a more mature research thesis:

\[
\boxed{
\begin{aligned}
&\textbf{Fresh verification is required not per round, but per genuinely new}\\
&\textbf{or insufficiently covered amount of optimizer movement.}
\end{aligned}
}
\]

This statement simultaneously accommodates:

- persistent moving-tail lower bounds;
- bounded reuse under shared structure;
- self-evaluation blindness;
- post-selection fresh-audit repair;
- current-policy self-correction;
- refresh-cadence phase transitions.

That is the strongest coherent synthesis of the project at this point.

---

# Appendix A — Post-v15 detailed progress records

The following sections preserve the detailed experiment records produced after v15. They are appended so that this master file remains a complete chronological research artifact rather than only a cleaned paper outline.



---

## Appendix source: `recursive_verification_progress_20260911_fd_experiment.md`

# Progress Update XV — §FD experiment beyond the Gaussian simulation

**Date:** 2026-09-11  
**Status:** Controlled experiment completed. The central §FD signature survives in a finite, non-Gaussian, bounded-label setting with an actually fitted misspecified verifier.

## 1. Question

The v15 note identified one distinctive falsifiable prediction worth testing before proving anything else:

> As optimization strength increases, there is a true-gain sign threshold that is set by verifier misspecification rather than audit budget. Increasing the audit budget should sharpen confidence around the threshold, not move the threshold itself; above it, the self-evaluation certificate becomes confidently wrong.

This update tests that claim outside the original Gaussian quadratic simulation.

## 2. Experimental setup

Outcome space is finite:
- 401 outcomes, `y in [-4,4]`.
- Current/base policy is an **asymmetric discrete-Laplace** distribution
  `p(y) ∝ exp(-|y| + 0.15 y)`, hence not Gaussian.

The main true reward is bounded and nonlinear:
```text
r_A(y) = tanh(y - 0.30 y^2 - 0.01 y^4).
```
It is not representable by the verifier class.

Trusted labels are stochastic and bounded:
```text
Y ∈ {-1,+1},
P(Y=+1 | y) = (1+r_A(y))/2.
```

The verifier is actually fitted from `n` current-policy trusted samples by OLS:
```text
v_hat(y) = beta0_hat + beta1_hat y.
```
HC1 robust standard errors are used, so the certificate is not relying on homoscedastic Gaussian noise.

A candidate is produced by the same exponential/KL tilt used throughout the note:
```text
q_eta(y) ∝ p(y) exp(eta * v_hat(y)).
```

For evaluation only, the exact synthetic true gain
`Delta = E_q[r_A] - E_p[r_A]`
is computed by summing over all 401 outcomes. The verifier never receives this quantity.

The sealed self-certificate freezes the realized candidate and forms a one-sided 95% lower bound on the **proxy** gain using the fitted linear model. A run is counted as *confident-and-wrong* when this lower bound is positive while the exact true gain is negative.

Monte Carlo replications:
- n=50: 3000
- n=200: 3000
- n=1000: 2000
- n=5000: 1000
- n=20000: 500

Random seed: 20260911.

## 3. Population threshold

The population linear projection of the nonlinear reward under `p` is

```text
beta_pop = (-0.124865, 0.292476).
```

Using this limiting fitted verifier, the exact true gain changes sign at

```text
eta_star = 3.497506.
```

This is the budget-independent reference threshold.

## 4. Main result — the threshold center does not move

| n | replications | slope_mean | slope_sd | eta_10pct_wrong | eta_50pct_wrong | eta_90pct_wrong | transition_width_10_90 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50.0000 | 3000.0000 | 0.3040 | 0.1237 | 2.2228 | 3.4458 | — | — |
| 200.0000 | 3000.0000 | 0.2955 | 0.0587 | 2.7543 | 3.4565 | 4.6500 | 1.8957 |
| 1000.0000 | 2000.0000 | 0.2930 | 0.0257 | 3.1280 | 3.5029 | 3.9103 | 0.7823 |
| 5000.0000 | 1000.0000 | 0.2931 | 0.0115 | 3.3192 | 3.4901 | 3.6750 | 0.3558 |
| 20000.0000 | 500.0000 | 0.2925 | 0.0058 | 3.4125 | 3.4968 | 3.5867 | 0.1742 |

For all `n >= 200`, the empirical 50% confident-wrong crossing remains near `eta ≈ 3.498`. Its total range across `n=200,...,20000` is only
`0.0464`,
about `1.33%` of the threshold.

What changes with budget is the **width**, not the center. Regressing the 10%-90% transition width against `n` gives

```text
transition width ∝ n^-0.515.
```

That is essentially the expected `n^(-1/2)` finite-sample sharpening.

![Confident-wrong curves](fd_confident_wrong_curves.png)

![Threshold vs budget](fd_threshold_vs_budget.png)

## 5. Budget sweep

Probability of a confident wrong certificate:

| n | eta=3 | eta=3.5 | eta=4 | eta=5 | eta=6 |
| --- | --- | --- | --- | --- | --- |
| 50.000 | 0.368 | 0.513 | 0.628 | 0.762 | 0.806 |
| 200.000 | 0.224 | 0.524 | 0.737 | 0.941 | 0.988 |
| 1000.000 | 0.035 | 0.496 | 0.936 | 1.000 | 1.000 |
| 5000.000 | 0.000 | 0.529 | 1.000 | 1.000 | 1.000 |
| 20000.000 | 0.000 | 0.518 | 1.000 | 1.000 | 1.000 |

Two opposite effects occur as `n` increases:

- Below the population threshold (`eta=3.0`), finite-sample harmful proposals disappear and the wrong-certificate probability goes to zero.
- Above the threshold (`eta>=4`), the fitted verifier stabilizes and the certificate becomes **more reliably wrong**, approaching probability one.

Thus more verification improves estimation of the misspecified verifier but does not repair the objective misspecification.

## 6. Robustness to a qualitatively different misspecification

To avoid making the result depend on a polynomial/quadratic-shaped reward, the experiment was repeated with a hidden nonlinear cliff:

```text
r_B(y) = tanh(0.8 y - 0.8 max(y-1,0)^2).
```

The population fitted slope is `0.264812` and the exact sign threshold is

```text
eta_star_B = 3.422735.
```

| n | eta_10pct_wrong | eta_50pct_wrong | eta_90pct_wrong | transition_width_10_90 |
| --- | --- | --- | --- | --- |
| 200.0000 | 2.6080 | 3.4181 | 4.7286 | 2.1206 |
| 1000.0000 | 3.0269 | 3.4064 | 3.9250 | 0.8981 |
| 5000.0000 | 3.2286 | 3.4177 | 3.6260 | 0.3975 |
| 20000.0000 | 3.3205 | 3.4216 | 3.5262 | 0.2058 |

The 50% crossing stays within a range of only
`0.0152`
across a 100x audit-budget increase. The transition-width exponent is

```text
transition width ∝ n^-0.507.
```

Again the center is stable and the transition sharpens at approximately square-root rate.

![Hidden-cliff robustness](fd_hidden_cliff_robustness.png)

## 7. Fresh post-selection audit control

A separate model-free audit draws new labels from both `q_eta` and `p` after the candidate is fixed. Using the usual variance-based sample size for a one-sided 95% sign test gives:

| eta | true_gain | m_per_distribution | sign_error_rate |
| --- | --- | --- | --- |
| 4.0000 | -0.0755 | 941.0000 | 0.0507 |
| 5.0000 | -0.2410 | 89.0000 | 0.0497 |
| 6.0000 | -0.3971 | 31.0000 | 0.0687 |

The required fresh sample count falls rapidly after moving away from the zero-gain point. This reproduces the corrected v13/v15 interpretation: **past the threshold, harmful updates are not intrinsically hard to verify; they are hard only for the self-evaluation certificate that cannot represent the failure.**

## 8. Interpretation

### Supported by this experiment

1. **The §FD qualitative signature survives beyond the Gaussian simulation.**
   A learned misspecified verifier on a finite non-Gaussian policy distribution exhibits a stable optimization-strength threshold.

2. **Audit budget controls sharpness, not threshold location.**
   The 50% threshold converges to the population value, while its finite-sample width shrinks approximately as `n^(-1/2)`.

3. **More same-model verification can increase confidence in the wrong answer.**
   Above the threshold, larger `n` makes the fitted verifier more stable and drives the confident-wrong probability toward one.

4. **Fresh candidate verification behaves differently.**
   Away from the zero-gain point, direct post-selection sign testing becomes cheaper as the harmful effect grows.

### Not established

- This is still a controlled synthetic finite-domain experiment, not an LLM or program-synthesis result.
- It does not prove that every misspecified learned verifier has a single sharp threshold.
- “Perfectly sharp” should be read as an **asymptotic** statement. At finite `n`, the transition has nonzero width, empirically about `n^(-1/2)` here.
- The threshold is budget-independent only after fixing the learning procedure, verifier class, base distribution, and optimization rule. Changing those objects can move it.

## 9. Consequence for the paper

The experiment is a positive result and justifies moving to the next empirical level. The strongest plot is not “reward hacking increases with optimization.” It is:

> **Across audit budgets spanning orders of magnitude, the failure curves cross at essentially the same optimization strength; added audit data only make the transition sharper.**

That is substantially more discriminating than a generic overoptimization curve.

## 10. Next action

Do **not** return to theorem expansion yet.

The next experiment should instantiate the same two-axis sweep in a finite-domain program-synthesis task:
- x-axis: optimization/selection strength;
- curves: trusted verification budget;
- hidden true reward: exhaustive semantic correctness;
- fitted proxy verifier: limited visible tests or learned pass predictor.

The target falsifier is the same: determine whether the failure threshold moves with trusted-data budget or merely sharpens around a stable location.


---

## Appendix source: `recursive_verification_progress_20260911_program_synthesis.md`

# Progress Update XVI — Finite-domain program-synthesis threshold experiment

**Date:** 2026-09-11  
**Status:** Completed. The §FD budget-independent threshold signature survives in a finite program-semantics environment with explicit public-test overfitting and an actually fitted misspecified verifier.

## 1. Question

The previous experiment showed a stable optimization-strength threshold in a non-Gaussian synthetic reward model. This update moves one empirical level closer to the intended application.

The test is whether a verifier trained from trusted semantic observations still exhibits a **budget-insensitive failure threshold** when candidate programs can overfit a public test suite.

The falsifier is direct:

> If increasing trusted-data budget substantially moves the critical optimization strength, the §FD interpretation is not robust. If the threshold center stays fixed while only the transition width shrinks, the misspecification-ceiling mechanism survives.

## 2. Program-synthesis environment

Finite input domain:
```text
x ∈ {0,...,255}
```

True semantic target:
```python
def target(x):
    return (x*x + 3*x + 7) % 11
```

Fallback rule:
```python
def fallback(x):
    return (x*x + 2*x + 7) % 11
```

A fixed public suite contains 32 inputs.

The candidate pool contains 243 deterministic programs:
- 180 **regular approximate programs**: correct on a deterministic modular subset of inputs, fallback elsewhere;
- 60 **exploit programs**: explicitly return the correct output on all public tests, but use a weak branch plus fallback elsewhere;
- 3 rare **exact programs**.

Pool diagnostics:

| program_type | count | base_policy_mass | mean_visible_pass_rate | mean_exhaustive_accuracy |
| --- | --- | --- | --- | --- |
| regular | 180 | 0.9368 | 0.6905 | 0.6829 |
| exploit | 60 | 0.0625 | 1.0000 | 0.3638 |
| exact | 3 | 0.0008 | 1.0000 | 1.0000 |

Thus the exploit programs are not assigned an artificial negative label. Their reward is computed from their actual exhaustive behavior over all 256 inputs.

## 3. Trusted verifier

Each trusted observation:
1. samples a program from the current/base policy;
2. samples a uniformly random semantic input;
3. observes exact correctness on that input.

The verifier class is deliberately misspecified:
```text
v_hat(program) = beta0_hat + beta1_hat * public_test_pass_rate(program).
```

It is fitted by OLS. The confidence calculation uses an HC1 heteroskedasticity-robust slope standard error.

The population projection is
```text
beta_pop = (0.287406, 0.529324)
```
and the base-policy exhaustive semantic accuracy is
```text
0.663239.
```

## 4. Candidate optimization and self-certificate

For optimization strength `eta`,
```text
q_eta(program) ∝ p(program) exp(eta * v_hat(program)).
```

For evaluation only, the true gain is computed exhaustively:
```text
Delta_true = E_q[semantic accuracy] - E_p[semantic accuracy].
```

The self-certificate freezes the realized candidate and evaluates its proxy gain with the **same fitted verifier**. A run is counted as *confident-and-wrong* when the one-sided 95% lower bound on proxy gain is positive while the exact semantic gain is negative.

In this one-feature model the self-certificate has a useful exact simplification. For every `eta>0`,
```text
certificate fires  <=>  |beta1_hat| > 1.645 * SE(beta1_hat).
```
The confidence decision is therefore driven by confidence in the fitted proxy relation, whereas the true sign can still reverse after strong optimization because the verifier cannot represent the overfit-program distinction.

## 5. Population threshold

With infinite trusted data:
```text
maximum beneficial true gain = 0.046261
eta at maximum gain          = 10.244
population sign threshold    = 24.608329
```

The candidate initially improves semantic reward, but beyond `eta_star` further proxy optimization shifts too much mass toward public-test overfit programs.

## 6. Main budget sweep

| n | replications | beta1_mean | beta1_sd | self_certificate_fire_prob | eta_10pct_wrong | eta_50pct_wrong | eta_90pct_wrong | transition_width_10_90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200.0000 | 2200.0000 | 0.5286 | 0.1922 | 0.8518 | 16.7500 | 24.7019 | — | — |
| 1000.0000 | 1600.0000 | 0.5300 | 0.0857 | 1.0000 | 20.3816 | 24.5312 | 31.1667 | 10.7851 |
| 5000.0000 | 800.0000 | 0.5311 | 0.0380 | 1.0000 | 22.4625 | 24.5179 | 27.0909 | 4.6284 |
| 20000.0000 | 400.0000 | 0.5282 | 0.0191 | 1.0000 | 23.5417 | 24.7135 | 25.7812 | 2.2396 |

Across a 100x budget increase (`n=200` to `n=20000`), the 50% confident-wrong threshold changes by only
```text
0.1957.
```

The 10%-90% transition width scales empirically as
```text
width ∝ n^-0.525.
```

Thus the finite-sample signature is:

> **More trusted data sharpen the transition at approximately square-root rate, but do not materially move its center.**

![Confident-wrong curves](program_synthesis_confident_wrong_curves.png)

![Threshold versus budget](program_synthesis_threshold_vs_budget.png)

## 7. Selected threshold points

Confident-and-wrong probability:

| n | eta=20 | eta=24 | eta=26 | eta=30 | eta=40 |
| --- | --- | --- | --- | --- | --- |
| 200.000 | 0.251 | 0.463 | 0.560 | 0.684 | 0.845 |
| 1000.000 | 0.082 | 0.452 | 0.636 | 0.873 | 0.993 |
| 5000.000 | 0.003 | 0.386 | 0.782 | 0.996 | 1.000 |
| 20000.000 | 0.000 | 0.247 | 0.935 | 1.000 | 1.000 |

Below the limiting threshold, larger trusted datasets remove finite-sample harmful proposals.

Above the threshold, the reverse occurs: the fitted verifier stabilizes and the self-certificate becomes **more consistently confident in a semantically harmful selected distribution**.

This is the central empirical distinction from an ordinary variance-limited failure.

## 8. Robustness to the public test suite

The entire environment was regenerated with four different public-test suites, keeping the semantic target, candidate generator family, verifier class, and base-policy weighting fixed.

| public_suite_seed | population_eta_star | eta_50pct_wrong_at_n5000 | absolute_gap | population_max_true_gain |
| --- | --- | --- | --- | --- |
| 20260911.0000 | 24.6083 | 24.4722 | 0.1361 | 0.0463 |
| 20260912.0000 | 26.0772 | 26.2625 | 0.1853 | 0.0491 |
| 17.0000 | 23.1910 | 23.3750 | 0.1840 | 0.0477 |
| 99.0000 | 28.4530 | 28.5556 | 0.1025 | 0.0516 |

At `n=5000`, the empirical 50% wrong threshold follows each environment's own population semantic sign threshold.

![Public-suite robustness](program_synthesis_public_suite_robustness.png)

The numerical threshold is therefore environment-specific, as it should be. What is stable is its **insensitivity to audit budget once the environment/verifier class is fixed**.

## 9. Fresh post-selection semantic audit

As a control, after `q_eta` is fixed we independently sample fresh semantic evaluations from `q_eta` and `p` and estimate the true gain directly.

| eta | true_gain | m_per_distribution | empirical_sign_error |
| --- | --- | --- | --- |
| 25.0000 | -0.0018 | 361663.0000 | 0.0568 |
| 30.0000 | -0.0251 | 1948.0000 | 0.0536 |
| 40.0000 | -0.0676 | 275.0000 | 0.0558 |

The direct semantic audit is expensive close to the zero-gain point and rapidly becomes cheaper as the harmful effect grows.

Therefore this experiment again rejects the overly broad statement “past the threshold verification is impossible.” The supported statement is narrower and stronger:

> **the self-evaluation certificate can become confidently wrong because its verifier class cannot represent the selected failure, while fresh post-selection semantic verification can still reject the update.**

## 10. What is supported

1. The budget-independent threshold signature survives outside the Gaussian/quadratic model.
2. It also survives when misspecification arises from explicit public-test-overfit programs with exhaustive hidden semantics.
3. Trusted budget primarily reduces transition width, approximately as `n^(-1/2)`.
4. Above the limiting threshold, more same-model trusted data can increase the probability of a confident wrong certificate.
5. Fresh candidate evaluation remains effective away from the zero-gain point.

## 11. What is not yet established

- Candidate programs are generated procedurally rather than by an LLM.
- The verifier uses only public-test pass rate as its learned feature.
- Exploit programs are deliberately present in the base-policy support.
- The experiment demonstrates a mechanism and a phase signature, not its prevalence in real code agents.
- A richer verifier class that represents the exploit mechanism can move or eliminate the threshold.

## 12. Research decision

This is a second positive empirical step. The same qualitative signature now appears in:
1. a finite non-Gaussian misspecified reward model;
2. a finite program-semantics environment with explicit test-suite overfitting.

The next escalation should be a **small actual code-generation model**, not another theorem or synthetic family.

Recommended design:
- generate multiple code candidates per task;
- public tests provide the cheap proxy signal;
- stronger hidden tests provide trusted semantic reward;
- vary Best-of-N / soft selection strength independently from trusted-verifier data budget;
- plot the semantic-failure threshold against budget.

The key question remains unchanged: **does more trusted data move the optimization-strength failure threshold, or mainly sharpen the transition around a stable location?**


---

## Appendix source: `recursive_verification_progress_20260911_learned_code_generator.md`

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


---

## Appendix source: `recursive_verification_progress_20260911_representation_phase.md`

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


---

## Appendix source: `recursive_verification_progress_20260911_recursive_lag.md`

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


---

## Appendix source: `recursive_verification_progress_20260911_2d_phase.md`

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


---

# Appendix B — Named-result ledger from v15

以下は v15 内で明示的に命名された theorem / proposition / corollary / lemma の見出し一覧。本体の統合評価が後期の修正を優先するため、ここに存在する名前がそのまま最終 contribution であることを意味しない。

- Proposition 1 — Exact proxy/true-gain decomposition [PROVED]
- Proposition 2 — \(L_2\)-concentrability stability bound [PROVED]
- Corollary 2.1 — Exponential optimization amplifies verifier error exponentially [PROVED]
- Proposition 3 — Sharp rare-tail failure under tiny on-policy error [PROVED]
- Proposition 4 — Pre-update verification has an \(M\)-factor information lower bound [PROVED]
- Proposition 5 — Optimal change-focused audit distribution [PROVED]
- Target Theorem A — Adaptive moving-tail lower bound [OPEN, HIGH PRIORITY]
- Theorem A' — Adaptive moving-tail verification lower bound
- Theorem 1 — One-step error amplification
- Theorem 2 — Adaptive-tree recursive verification lower bound
- Theorem 3 — Near-matching minimax budget characterization
- Theorem 4 — Audit-mixture phase transition
- Proposition — Expected sequential cost
- Theorem B — Local verification information law
- Theorem C — Recursive information-budget lower bound
- Theorem 1 — One-step proxy/true gain decomposition and shift amplification
- Theorem 2 — Adaptive moving-tail lower bound
- Theorem 3 — General local verification information law
- Theorem 4 — Noisy recursive minimax budget
- Theorem 5 — Candidate-audit phase transition
- Theorem 6 — Sequential cost distinction
- Theorem D — Structured fresh-verification complexity
- Theorem E — Smooth-class removal of rare-tail amplification
- AS. Proposition F — Finite reachable-class reusable verification [PROVED]
- Target Theorem G — Double-adaptivity minimax law
- AX. Concrete hard-family route for Target Theorem G
- Theorem F — Information-limited adaptive verification
- BP. Why Target Theorem G in its old form is retired
- Target Theorem H
- Theorem J — External Verification Entropy Lower Bound
- Target Theorem L — Sealed Verification Lifetime / Trust Allocation Frontier
- CM. Theorem M — Uniform sealed-audit certificate under arbitrary recursive adaptivity [PROVED]
- CR. Theorem N — Smooth Gaussian verification-lag phase transition [PROVED]
- CS. Why Theorem N is stronger than the old two-point construction
- Lemma 1 (antitonicity) [PROVED]
- Proposition 2 (existence of minimal stable designs) [PROVED]
- Theorem O — endogenous minimum-budget law [PROVED up to the factor 2 of \(\overline\Phi\)]
- DE. Theorem P — endogenous saturation floor, and a correction to Theorem N [PROVED]
- Theorem P (saturation floor)
- Corollary (Theorem N is unsafe in the loop) [PROVED, numerically exact]
- DF. Theorem Q — bounded refresh under arbitrary endogenous adaptivity [PROVED]
- Lemma 3 (exact refresh law) [PROVED]
- Theorem Q
- Theorem R — matching \(\Omega(p)\) lower bound [PROVED]
- DG. Theorem S — the bounded / sublinear / linear phase law [PROVED]
- Proposition U — greedy is exactly minimax-optimal on a *fixed* reachable set [PROVED]
- Lemma 4 (leverage contamination) [PROVED]
- DJ. Theorem T — the real separation: innovation generation [PROVED within the stated model]
- Theorem T (myopic–strategic separation under endogenous innovation)
- Corollary V — endogeneity turns audit allocation from submodular maximization into set cover [PROVED in \(\mathcal I\)]
- DN.1 Theorem M must be demoted from contribution to cited lemma
- EC. Theorem U — the naive endogenous audit has zero rejection power [PROVED]
- ED. Theorem V — the winner's-curse floor, two-sided [PROVED]
- EF. Theorem W — the exact saturation identity [PROVED; verified to 8 digits]
- Consequence — the strongest form of the correction to Theorem N
- EH. Theorem X — capability scaling and the maximum useful audit budget [PROVED]
- Theorem U survives [PROVED]
- FC. Theorem Y — the curvature ceiling [PROVED; verified to 6 digits]
- FD. Theorem Z — beyond \(\eta_{\max}\) the certificate is confident and wrong [PROVED; sharp in simulation]
- FE. Proposition — the well-specified endogenous law is not separately identifiable [PROVED]
- FG. Proposition AA — exact deterrence threshold [PROVED]
- FH. Theorem AB — minimax value, and the shape of the transition [PROVED UP TO CONSTANTS]
- FI. Proposition AC and Theorem AD — predictability, and complexity
- Proposition AC (Fano-type predictability bound) [PROVED UP TO CONSTANTS]
- Theorem AD — MCSS is NP-hard [PROVED]
- The reconciliation with Corollary V — the operational lesson [PROVED]
- GC. Theorem AE — the joint law is a second-order pole at the misspecification ceiling [PROVED UP TO CONSTANTS]
- GD. Corollary AF — verification budget buys proximity at a square-root rate [PROVED UP TO CONSTANTS]
- GE. Theorem AG — the dichotomy [PROVED, given (J1)–(J5)]
- GF. Proposition AH — the three growth laws [PROVED UP TO CONSTANTS]
- Theorem AE′ (restated, deflated) [PROVED UP TO CONSTANTS]
- Theorem AG′ (corrected dichotomy) [PROVED]
- IB. Theorem AI — optimal allocation [PROVED UP TO CONSTANTS; closed form verified exactly]
- Corollary AI.1 — limits recover the known laws [PROVED]
- Corollary AI.2 — scalings [PROVED UP TO CONSTANTS]
- IC. Theorem AJ — the \(\varrho\) phase law [PROVED UP TO CONSTANTS]
- ID. Corollary AK — total budget under the three aggression schedules [PROVED UP TO CONSTANTS; one prediction corrected by simulation]
- IE. Theorem AL — sealed once + adaptive fresh reserve [PROVED UP TO CONSTANTS]


---

# Appendix C — Provenance and caution

この master record は会話内で作成・更新された研究ノートと実験 artifact を統合したもの。v15 に含まれる literature-positioning claim は、そのノート時点の調査記録として保持しているが、この master 作成時に全参考文献を改めて外部再検証したものではない。投稿前には bibliography / priority / novelty を再検索する。

理論式についても、`PROVED` は当該研究ノート内で proof が与えられたという研究記録上の status を意味する。査読レベルの独立 proof audit は別途必要。特に auxiliary endogenous-design branch、SOAR familywise accounting、unbounded quadratic-reward concentration、SDP parameterization は再監査対象。


---

# Appendix D — Progress Update XXI

# Progress Update XXI — From stale-step count to coverage-relative shift

**Date:** 2026-09-11  
**Status:** Completed. The previous \(\eta L\) phase law survives as an optimizer-specific coordinate, but a stronger invariance test shows that no policy-shift scalar alone is a universal safety law. The best current synthesis is **optimizer-induced shift relative to verifier misspecification geometry**.

---

## 1. Motivation

Progress Update XX found an approximate recursive collapse boundary

\[
\eta L \approx 15\text{--}20
\]

for exponential/KL updates under a stale verifier.

That raised two questions:

1. Is \(\eta L\) a meaningful cross-system quantity, or only a consequence of the arbitrary numerical scale of the verifier score?
2. Can a policy-shift measure such as KL, \(\chi^2\), or maximum density-ratio amplification predict failure across:
   - score rescaling,
   - different optimizer families,
   - different verifier representations?

This update tests those invariances.

---

## 2. Result A — raw \(\eta L\) is not score-scale invariant

Take the same fitted verifier score \(v\), but replace it by

\[
v_c = c\,v.
\]

Under exponential update,

\[
q(y)\propto p(y)\exp(\eta v_c(y))
=
p(y)\exp(c\eta v(y)).
\]

Therefore changing the arbitrary score scale by \(c\) must change the raw \(\eta\)-threshold by \(1/c\).

This was tested in the recursive learned-generator environment.

| score_scale | min_failing_raw_etaL | min_failing_scale_adjusted_etaL |
| --- | --- | --- |
| 0.500 | 32.000 | 16.000 |
| 1.000 | 16.000 | 16.000 |
| 2.000 | 8.000 | 16.000 |
| 4.000 | 4.000 | 16.000 |

The first failing raw \(\eta L\) changes exactly as expected:

\[
32,\ 16,\ 8,\ 4
\]

for score scales

\[
c=0.5,\ 1,\ 2,\ 4.
\]

But the scale-adjusted product

\[
\boxed{c\eta L}
\]

is exactly

\[
\boxed{16}
\]

for all four settings in this grid.

### Consequence

\[
\boxed{
\eta L \text{ is not a universal physical quantity.}
}
\]

It is useful only after fixing verifier score calibration.

The invariant object must be a quantity induced by the **actual policy change**.

![Score-rescaling invariance](recursive_score_rescaling_invariance.png)

---

## 3. Result B — within a fixed verifier, max density-ratio amplification transfers best across optimizer families

The next test keeps the **same proxy-only verifier** but changes the optimizer:

- exponential tilt;
- exact Best-of-\(N\) selection.

For Best-of-\(N\), the selected distribution is computed exactly from the learned generator distribution, including score ties.

For each Best-of-\(N\) candidate distribution, an exponential-tilt candidate with the closest value of a chosen shift metric is found. The resulting true semantic gains are compared.

| matching_metric | mean_abs_true_gain_difference | median_abs_true_gain_difference | true_gain_correlation | sign_agreement |
| --- | --- | --- | --- | --- |
| max_log_ratio | 0.0043 | 0.0032 | 0.9985 | 0.9950 |
| mem_mass_shift | 0.0043 | 0.0032 | 0.9985 | 0.9950 |
| chi2 | 0.0053 | 0.0032 | 0.9990 | 0.9875 |
| proxy_gain | 0.0075 | 0.0033 | 0.9980 | 0.9749 |
| KL | 0.0085 | 0.0033 | 0.9955 | 0.9674 |
| TV | 0.0129 | 0.0034 | 0.9720 | 0.9524 |

The most transportable tested metric is

\[
\boxed{
\max_y \log\frac{q(y)}{p(y)}
=
\log \left\|\frac{dq}{dp}\right\|_\infty.
}
\]

It gives:

\[
\text{true-gain correlation}\approx 0.9985,
\]

\[
\text{mean absolute gain mismatch}\approx0.0043,
\]

\[
\text{sign agreement}\approx99.5\%.
\]

\(\chi^2\) is close. KL is useful but weaker.

### Interpretation

This result connects directly back to the original candidate-amplification parameter

\[
M=\left\|\frac{dq}{dp}\right\|_\infty.
\]

The original moving-tail theory used \(M\) because current-policy verification becomes expensive when the candidate puts much more mass than \(p\) on verifier-salient regions.

The new experiment suggests that this \(L_\infty\)-style amplification is not merely a hard-family artifact: **within a fixed verifier, it is also highly portable across two very different selection mechanisms.**

![Optimizer-family gain versus KL](recursive_optimizer_family_gain_vs_kl.png)

The plot above uses KL for readability; the quantitative matching table shows that max log-density ratio performs better.

---

## 4. Result C — even max density-ratio amplification is not enough across verifier classes

A stronger verifier changes not only the scale of the score but the **direction in program space that is being optimized**.

To isolate this effect, candidate distributions were matched to the same maximum log density ratio under three verifier representations:

- proxy only;
- medium exploit signature;
- strong exploit signature.

| target_max_log_ratio | representation | eta_required | actual_max_log_ratio | true_gain | proxy_gain | KL | chi2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | proxy_only | 4.0853 | 1.0000 | 0.0585 | 0.0686 | 0.1364 | 0.2922 |
| 1.0000 | medium_signature | 3.7382 | 1.0000 | 0.0556 | 0.0632 | 0.1152 | 0.2421 |
| 1.0000 | strong_signature | 3.5780 | 1.0000 | 0.0581 | 0.0616 | 0.1065 | 0.2177 |
| 2.0000 | proxy_only | 9.9490 | 2.0000 | 0.0825 | 0.1460 | 0.6675 | 2.0340 |
| 2.0000 | medium_signature | 8.7033 | 2.0000 | 0.0869 | 0.1305 | 0.5257 | 1.4313 |
| 2.0000 | strong_signature | 8.1627 | 2.0000 | 0.1041 | 0.1213 | 0.4484 | 1.0947 |
| 3.0000 | proxy_only | 22.1217 | 3.0000 | 0.0033 | 0.2356 | 2.0184 | 12.2589 |
| 3.0000 | medium_signature | 16.2642 | 3.0000 | 0.0642 | 0.1986 | 1.3516 | 5.9198 |
| 3.0000 | strong_signature | 14.3445 | 3.0000 | 0.1317 | 0.1736 | 1.0239 | 3.6035 |
| 4.0000 | medium_signature | 31.0520 | 4.0000 | -0.0258 | 0.2542 | 2.5803 | 19.4228 |
| 4.0000 | strong_signature | 23.1146 | 4.0000 | 0.1482 | 0.2170 | 1.8185 | 11.7376 |
| 5.0000 | medium_signature | 62.7767 | 5.0000 | -0.0879 | 0.2752 | 3.4504 | 39.3009 |
| 5.0000 | strong_signature | 36.6750 | 5.0000 | 0.1779 | 0.2546 | 2.9203 | 47.9186 |

At matched shift magnitude, the semantic outcome can differ substantially.

For example, around max log density ratio \(=4\):

- medium representation: true gain \(\approx -0.026\);
- strong representation: true gain \(\approx +0.148\).

Thus essentially the same worst-case policy amplification can be harmful under one verifier and strongly beneficial under another.

A matched-KL control shows the same point even more starkly.

| target_KL | representation | eta_required | true_gain | proxy_gain | cauchy_risk_ratio | actual_KL | chi2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | proxy_only | 12.8473 | 0.0725 | 0.1753 | 1.1873 | 1.0000 | 3.7323 |
| 1.0000 | medium_signature | 13.0893 | 0.0806 | 0.1745 | 1.1703 | 1.0000 | 3.6226 |
| 1.0000 | strong_signature | 14.0907 | 0.1310 | 0.1719 | 1.1196 | 1.0000 | 3.4616 |
| 2.0000 | proxy_only | 21.9257 | 0.0048 | 0.2347 | 1.5938 | 2.0000 | 12.0597 |
| 2.0000 | medium_signature | 22.8468 | 0.0207 | 0.2322 | 1.5919 | 2.0000 | 11.8736 |
| 2.0000 | strong_signature | 25.2399 | 0.1519 | 0.2245 | 1.7867 | 2.0000 | 15.0315 |
| 2.5000 | proxy_only | 28.1615 | -0.0388 | 0.2550 | 1.7925 | 2.5000 | 17.9940 |
| 2.5000 | medium_signature | 29.6758 | -0.0192 | 0.2515 | 1.8211 | 2.5000 | 18.2292 |
| 2.5000 | strong_signature | 31.3247 | 0.1644 | 0.2423 | 2.2939 | 2.5000 | 28.8534 |
| 3.0000 | proxy_only | 38.7456 | -0.0847 | 0.2703 | 1.9881 | 3.0000 | 24.8839 |
| 3.0000 | medium_signature | 40.7680 | -0.0597 | 0.2661 | 2.0858 | 3.0000 | 26.7588 |
| 3.0000 | strong_signature | 37.7138 | 0.1808 | 0.2568 | 2.9189 | 3.0000 | 52.4918 |

At KL \(\approx2.5\):

- proxy-only: true gain \(\approx-0.039\);
- medium representation: true gain \(\approx-0.019\);
- strong representation: true gain \(\approx+0.164\).

### Consequence

\[
\boxed{
\text{No policy-shift magnitude alone determines recursive safety.}
}
\]

A shift metric must be interpreted **relative to the verifier's error / misspecification geometry**.

![Matched max-log-ratio representation control](recursive_matched_maxlog_representation.png)

![Matched KL representation control](recursive_matched_kl_representation.png)

---

## 5. Result D — the generic Cauchy coverage certificate is valid but too conservative to explain the empirical phase

The exact decomposition is

\[
\Delta
=
G
-
\left(
\mathbb E_q e-\mathbb E_p e
\right),
\qquad
e=v-r.
\]

If the fitted verifier contains an intercept, then for its population least-squares projection under \(p\),

\[
\mathbb E_p[e]=0.
\]

Hence

\[
\mathbb E_q e
=
\mathbb E_p
\left[
\left(\frac{q}{p}-1\right)e
\right].
\]

By Cauchy-Schwarz,

\[
\left|
\mathbb E_qe-\mathbb E_pe
\right|
\le
\|e\|_{L_2(p)}
\sqrt{\chi^2(q\|p)}.
\]

Therefore every stale block satisfies the valid lower bound

\[
\boxed{
\Delta
\ge
G-
\|e\|_{L_2(p)}
\sqrt{\chi^2(q\|p)}.
}
\]

So a sufficient safety condition is

\[
\boxed{
G>
\|e\|_{L_2(p)}
\sqrt{\chi^2(q\|p)}.
}
\]

This is not a new theorem; it is the earlier coverage bound applied to the composed stale-block candidate.

The empirical test confirms validity:

| representation | optimizer | n_blocks | certified_safe_fraction | endpoint_harmful_fraction | false_safe_fraction | coverage_of_truly_safe_blocks |
| --- | --- | --- | --- | --- | --- | --- |
| medium_signature | bon | 180 | 0.1667 | 0.1167 | 0.0000 | 0.1887 |
| medium_signature | exp | 240 | 0.0833 | 0.0500 | 0.0000 | 0.0877 |
| proxy_only | bon | 180 | 0.0278 | 0.1611 | 0.0000 | 0.0331 |
| proxy_only | exp | 240 | 0.0792 | 0.0958 | 0.0000 | 0.0876 |
| strong_signature | bon | 180 | 0.2167 | 0.0111 | 0.0000 | 0.2191 |
| strong_signature | exp | 240 | 0.1000 | 0.0000 | 0.0000 | 0.1000 |

There were **zero false-safe blocks** in the tested population trajectories.

However, the certificate is highly conservative. Depending on representation and optimizer, it certifies only about 3–22% of truly safe blocks.

### Consequence

The generic \(L_2\times\chi^2\) bound is appropriate as a worst-case safety certificate, but it is **not tight enough to explain the observed phase boundary**.

This is useful: it tells us not to force the empirical \(\eta L\) transition into a loose global Cauchy bound.

---

## 6. The exact latent quantity

For any realized verifier \(v\), define

\[
G(p,q;v)
=
\mathbb E_qv-\mathbb E_pv,
\]

and the verifier-error shift

\[
E_{\rm shift}(p,q;v,r)
=
\mathbb E_q(v-r)-\mathbb E_p(v-r).
\]

Then exactly

\[
\boxed{
\Delta
=
G-E_{\rm shift}.
}
\]

When \(G>0\), define the normalized exploitation ratio

\[
\boxed{
\rho_{\rm exploit}
=
\frac{E_{\rm shift}}{G}.
}
\]

Then

\[
\boxed{
\Delta>0
\iff
\rho_{\rm exploit}<1,
}
\]

\[
\boxed{
\Delta<0
\iff
\rho_{\rm exploit}>1.
}
\]

This identity is tautological, not a contribution by itself.

Its importance is statistical:

> **the verification problem is precisely the problem of estimating the policy-induced shift of verifier error after the optimizer has selected \(q\).**

That is the quantity that old current-policy data may fail to identify when \(q/p\) amplifies poorly covered regions, and the quantity that fresh candidate-aware labels estimate directly.

---

## 7. New synthesis with the earlier structured-verification theorem

The earlier one-step structured theory defined the policy-improvement functional

\[
L_{p,q}(f)
=
\mathbb E_qf-\mathbb E_pf
\]

and its function-class verification difficulty

\[
\mathcal V_{\mathcal F}(p,q;\mu).
\]

Progress Updates XIX–XXI now suggest a recursive interpretation:

1. At a verifier refresh time \(s\), the audit data constrain the residual/error class around \(p_s\).
2. The optimizer then moves under a stale verifier for several updates.
3. These updates compose into a new candidate \(q_{s:L}\).
4. What must be certified is not the number of elapsed rounds, but the functional
   \[
   L_{p_s,q_{s:L}}(e).
   \]
5. Its difficulty is governed by how the **composed stale shift** intersects the error class.

Thus the natural recursive quantity is not merely

\[
L,\qquad \eta L,\qquad \mathrm{KL}(q\|p),\qquad M.
\]

It is the pair

\[
\boxed{
\text{stale policy shift}
\quad+\quad
\text{verifier-error geometry}.
}
\]

In the unstructured worst case this reduces to density-ratio amplification such as \(M\).

In a structured class it reduces to restricted feature geometry such as

\[
d^\top\Sigma_\mu^\dagger d.
\]

This reconnects the newest recursive experiments to the earliest minimax theory.

---

## 8. Updated interpretation of the \(\eta L\) phase

The empirical law from Update XX should now be stated more carefully.

Old wording:

> recursive failure is controlled by \(\eta L\).

Better wording:

> **For exponential updates with a fixed score calibration and fixed misspecified verifier class, \(\eta L\) is an exact parameterization of accumulated stale optimization pressure. Across optimizer families, density-ratio amplification is more portable. Across verifier classes, no shift-only scalar is sufficient.**

So the hierarchy is:

### Level 1 — optimizer-specific coordinate

\[
\eta L.
\]

Exact for repeated exponential tilts with a frozen verifier.

### Level 2 — policy-shift coordinate

\[
\log M
=
\log\left\|\frac{dq}{dp}\right\|_\infty,
\]

or related KL / \(\chi^2\).

More portable across optimizer families.

### Level 3 — verification-relevant coordinate

\[
\mathcal V_{\mathcal F}(p,q;\mu)
\]

or an equivalent residual-sensitive coverage quantity.

Needed when verifier representations differ.

This three-level hierarchy is currently the cleanest synthesis.

---

## 9. Research implication: the central recursive theorem should be a composed-candidate theorem, not an \(\eta L\) theorem

A future theorem should not attempt to prove a universal numerical threshold in \(\eta L\).

A better target is:

> Given a verifier refreshed at time \(s\), characterize the trusted information required to certify every update until the next refresh in terms of the **composed candidate distribution** \(q_{s:L}\) and the audit-visible error class.

For frozen exponential updates,

\[
q_{s:L}(y)
\propto
p_s(y)\exp(L\eta v_s(y)).
\]

Then the existing structured one-step theorem can be applied directly to the composed stale candidate.

A schematic target is

\[
\boxed{
m_s
\asymp
\frac{
\mathcal V_{\mathcal F_s}
(p_s,q_{s:L};\mu_s)
}{
\Gamma_s^2
}
\log\frac1\delta.
}
\]

The actual research challenge is to make this useful recursively:

- \(L\) is chosen adaptively;
- \(p_s\) changes after refresh;
- \(\mathcal F_s\) may itself change after new trusted data;
- familywise guarantees must hold over refresh blocks;
- candidate-aware fresh labels may be fed into later verifier fits.

This target does **not** invent a new metric. It asks whether the existing verification functional can be lifted from one-step candidates to recursively composed stale blocks.

---

## 10. Current empirical verdict

The invariance test yields four clean conclusions.

### Supported

1. Raw \(\eta L\) is score-scale dependent.
2. Max density-ratio amplification is highly transportable between exponential tilt and Best-of-\(N\) under the same verifier.
3. No shift-only metric is sufficient across verifier representations.
4. Generic \(L_2\times\chi^2\) safety bounds are valid but too conservative to explain the observed phase sharply.

### Rejected / downgraded

1. A universal \(\eta L\) law.
2. A universal KL-only recursive safety threshold.
3. The idea that policy movement magnitude by itself determines failure.

### Strengthened

The original broader thesis:

\[
\boxed{
\text{verification difficulty is about optimizer-induced movement into directions
that the current verification state does not control.}
}
\]

This survives all current corrections.

---

## 11. Best current paper-level message

The most defensible conceptual statement is now:

> **Recursive verification lag is not elapsed time. It is the accumulation of optimizer-induced policy shift in directions where the current verifier remains uncertain or misspecified.**

A shorter version:

\[
\boxed{
\text{verification lag}
=
\text{policy shift relative to verifier-error geometry}.
}
\]

This is more general than the empirical \(\eta L\) law and more faithful to the minimax theory.

---

## 12. Next research step

The next theoretical step is now justified, but it should be narrow.

Do **not** create another broad theorem family.

Attempt exactly one result:

### Target — Composed stale-candidate verification law

Take the existing structured one-step verification theorem and prove a blockwise recursive corollary for a verifier frozen over multiple policy updates.

Required output:

1. exact composition for exponential update;
2. blockwise verification complexity in terms of
   \[
   \mathcal V_{\mathcal F}(p_s,q_{s:L};\mu_s);
   \]
3. familywise accounting over adaptively chosen refresh blocks;
4. a lower-bound construction showing that if the composed candidate reaches a new poorly covered error direction, no amount of mere round-count bookkeeping can replace fresh trusted information;
5. explicit counterexample showing why shift magnitude alone is insufficient when verifier representation changes.

If this closes cleanly, it is the right bridge between the early lower-bound theory and the new recursive experiments.

If it does not close without strong artificial assumptions, stop theory expansion and present the \(\eta L\) / max-density-ratio results as controlled empirical mechanism evidence.

---

## 13. Artifact inventory for this update

Data:

- `recursive_score_rescaling_phase.csv`
- `recursive_score_rescaling_boundary.csv`
- `recursive_cross_setting_stale_blocks.csv`
- `recursive_cross_setting_runs.csv`
- `recursive_cross_setting_metric_accuracy.csv`
- `recursive_matched_kl_representation_control.csv`
- `recursive_matched_maxlog_representation_control.csv`
- `recursive_matched_kl_optimizer_family.csv`
- `recursive_optimizer_metric_matching_quality.csv`
- `recursive_cauchy_certificate_tightness.csv`

Figures:

- `recursive_score_rescaling_invariance.png`
- `recursive_optimizer_family_kl_alignment.png`
- `recursive_optimizer_family_gain_vs_kl.png`
- `recursive_matched_kl_representation.png`
- `recursive_matched_maxlog_representation.png`
- `recursive_cauchy_risk_alignment.png`



---

# Appendix E — Progress Update XXII

# Progress Update XXII — Composed stale-candidate verification theorem

**Date:** 2026-09-11  
**Status:** Core blockwise theorem proved as a conditional corollary of the existing structured one-step theorem. The result cleanly connects the early minimax theory to the later recursive-lag experiments, but **is not by itself a new central theorem**: the proof is one-step minimax verification + exact policy composition + conditional familywise accounting.

---

## 1. Why this update matters

The empirical sequence XVIII–XXI changed the project in an important way.

The naive recursive claim

> “verification becomes hard because many rounds have elapsed”

is false.

The better empirical claim was

> “verification becomes stale when the optimizer moves far under a verifier whose error geometry is no longer controlled.”

Update XXI then showed that even raw policy-shift magnitude is insufficient across verifier classes.

The natural mathematical question is therefore:

> **Can the recursive stale-verifier problem be reduced exactly to the already-proved one-step structured verification problem applied to the composed endpoint candidate?**

The answer is **yes**, under a clean temporal-independence condition.

This yields a precise result:

\[
\boxed{
\text{block verification cost}
\asymp
\frac{
\mathcal V_{\mathcal F}
(p_{\rm start},q_{\rm end};\mu)
}{
\Gamma^2
}
\log\frac1\delta,
}
\]

where \(q_{\rm end}\) is the policy reached after all stale updates in the block.

The number of stale steps \(L\) matters only through the **composed endpoint distribution** and the resulting true margin.

---

# 2. Base one-step theorem being reused

The v15 structured-verification theorem fixes a current policy \(p\), candidate \(q\), audit distribution \(\mu\), and a closed linear reward/error class

\[
\mathcal F\subseteq L_2(\mu).
\]

For

\[
L_{p,q}(f)
=
\mathbb E_q[f]-\mathbb E_p[f],
\]

let \(g_{\mathcal F}\) be the Riesz representer:

\[
L_{p,q}(f)
=
\langle g_{\mathcal F},f\rangle_{L_2(\mu)}.
\]

Define

\[
\boxed{
\mathcal V_{\mathcal F}(p,q;\mu)
=
\|g_{\mathcal F}\|_{L_2(\mu)}^2
=
\sup_{f\in\mathcal F,\ f\ne0}
\frac{
(\mathbb E_qf-\mathbb E_pf)^2
}{
\mathbb E_\mu[f^2]
}.
}
\]

For bounded trusted labels whose conditional mean is \(f\), the minimax one-step sign-classification cost is, under the local boundedness condition for the lower bound,

\[
\boxed{
m^\star
=
\Theta\!\left(
\frac{
\mathcal V_{\mathcal F}(p,q;\mu)
}{
\Gamma^2
}
\log\frac1\delta
\right).
}
\]

The upper bound uses the Riesz-weighted statistic and a median-of-means estimator.

The lower bound uses the least-favorable directions

\[
f_\pm
=
\pm
\frac{\Gamma}{V}g_{\mathcal F}
\]

and a Bretagnolle–Huber testing argument.

The present update does not re-prove that theorem from scratch; it lifts it to recursively generated stale blocks.

---

# 3. Exact composition of a stale exponential-update block

Let a verifier score \(v_s(y)\) be frozen at the beginning of a block.

Start from policy

\[
p_{s,0}=p_s.
\]

For \(k=0,\ldots,L-1\), perform

\[
p_{s,k+1}(dy)
=
\frac{
\exp(\eta_{s,k}v_s(y))
p_{s,k}(dy)
}{
\mathbb E_{p_{s,k}}
[\exp(\eta_{s,k}v_s)]
}.
\]

Define cumulative stale strength

\[
\Lambda_s
=
\sum_{k=0}^{L-1}\eta_{s,k}.
\]

## Lemma XXII.1 — Exact stale-block composition

\[
\boxed{
p_{s,L}(dy)
=
\frac{
\exp(\Lambda_s v_s(y))p_s(dy)
}{
\mathbb E_{p_s}
[\exp(\Lambda_s v_s)]
}.
}
\]

### Proof

Induction.

For one step the formula is the update definition.

Assume

\[
p_{s,k}(dy)
=
\frac{
e^{\Lambda_{s,k}v_s(y)}p_s(dy)
}{
Z(\Lambda_{s,k})
}
\]

with

\[
\Lambda_{s,k}
=
\sum_{\ell<k}\eta_{s,\ell}.
\]

Then

\[
p_{s,k+1}(dy)
\propto
p_{s,k}(dy)e^{\eta_{s,k}v_s(y)}
\propto
p_s(dy)
e^{(\Lambda_{s,k}+\eta_{s,k})v_s(y)}.
\]

Normalizing gives the claim. \(\square\)

### Immediate consequence

For a frozen verifier,

\[
\boxed{
\text{the endpoint depends on the stale update schedule only through }
\Lambda_s.
}
\]

In particular, for constant per-step strength \(\eta\),

\[
\Lambda_s=L\eta.
\]

This is why the empirical \(\eta L\) collapse in Update XX was so strong.

It is also why raw round count \(L\) cannot be fundamental.

---

# 4. Block-compression principle

Let

\[
q_s:=p_{s,L}
\]

be the endpoint of a stale block.

The true net block gain is

\[
\Delta_s
=
\mathbb E_{q_s}[r]
-
\mathbb E_{p_s}[r].
\]

For the purpose of deciding whether the block as a whole is beneficial, the entire internal path

\[
p_{s,1},\ldots,p_{s,L-1}
\]

can be discarded.

The statistical verification problem is exactly the one-step comparison

\[
p_s
\longrightarrow
q_s.
\]

## Proposition XXII.2 — Block compression

Assume:

1. true reward is static during the block;
2. the acceptance objective concerns the **net endpoint gain**
   \[
   \Delta_s
   =
   \mathbb E_{q_s}r-\mathbb E_{p_s}r;
   \]
3. audit observations depend on the reward and sampled outcome, not on the hidden path used to produce \(q_s\).

Then any two optimization paths with the same start \(p_s\) and the same endpoint \(q_s\) induce the same endpoint verification problem.

Therefore their minimax trusted-label complexities under a fixed \((\mathcal F_s,\mu_s,\Gamma_s,\delta_s)\) are identical.

### Important limitation

This proposition does **not** apply if the safety requirement is

\[
\Delta_{s,k}\ge0
\qquad
\text{for every intermediate update }k.
\]

If intermediate policies are deployed or irreversible, the path matters.

Thus there are two distinct operational goals:

- **block-end / rollback safety**: only the accepted endpoint must improve;
- **strict monotonicity**: every intermediate update must improve.

The composed theorem concerns the first.

---

# 5. Conditional recursive setup

Let

\[
\mathcal G_s
\]

be the complete transcript **after the block endpoint \(q_s\) has been generated but before the trusted certification batch for that block is observed**.

Conditional on \(\mathcal G_s\), assume the following are fixed / measurable:

- \(p_s\);
- \(q_s\);
- audit distribution \(\mu_s\);
- reward/error class \(\mathcal F_s\);
- required margin \(\Gamma_s\);
- error allocation \(\delta_s\).

Now acquire a fresh trusted batch

\[
S_s
=
\{(Y_{s,i},Z_{s,i})\}_{i=1}^{m_s},
\]

with

\[
Y_{s,i}\mid\mathcal G_s
\stackrel{\rm iid}{\sim}
\mu_s
\]

and bounded trusted labels satisfying

\[
\mathbb E[
Z_{s,i}
\mid
Y_{s,i}=y,\mathcal G_s
]
=
f(y),
\qquad
f\in\mathcal F_s.
\]

The crucial condition is:

\[
\boxed{
q_s\text{ is generated before seeing }S_s.
}
\]

The candidate may depend arbitrarily on **all previous** trusted batches.

It simply may not depend on the current certification batch before that batch is used to certify it.

---

# 6. Theorem XXII.3 — Conditional composed-block verification

Define

\[
V_s
=
\mathcal V_{\mathcal F_s}
(p_s,q_s;\mu_s).
\]

Suppose

\[
|L_{p_s,q_s}(f)|
\ge
\Gamma_s.
\]

Then there is a block-end sign test with conditional error probability at most \(\delta_s\) using

\[
\boxed{
m_s
=
O\!\left(
\frac{V_s}{\Gamma_s^2}
\log\frac1{\delta_s}
\right).
}
\]

### Proof

Condition on \(\mathcal G_s\).

After conditioning,

\[
p_s,q_s,\mu_s,\mathcal F_s
\]

are fixed and the new batch is iid from \(\mu_s\).

Therefore the setting is exactly the one-step structured verification theorem.

Let \(g_s\) be the conditional Riesz representer and define

\[
X_{s,i}
=
g_s(Y_{s,i})Z_{s,i}.
\]

Then

\[
\mathbb E[X_{s,i}\mid\mathcal G_s]
=
L_{p_s,q_s}(f),
\]

and

\[
\mathbb E[X_{s,i}^2\mid\mathcal G_s]
\le
V_s.
\]

Conditional median-of-means concentration gives

\[
|\widehat L_s-L_{p_s,q_s}(f)|
\le
C
\sqrt{
\frac{
V_s\log(1/\delta_s)
}{
m_s
}
}
\]

with conditional probability at least \(1-\delta_s\).

Choosing \(m_s\) so that the right-hand side is \(<\Gamma_s\) yields the correct sign. \(\square\)

---

# 7. Corollary XXII.4 — Familywise validity over adaptive refresh blocks

Let blocks \(s=1,2,\ldots\) be chosen adaptively.

Suppose each \(\delta_s\) is predictable before batch \(S_s\) is observed and

\[
\boxed{
\sum_s\delta_s
\le
\delta
\quad\text{almost surely}.
}
\]

Apply Theorem XXII.3 in every attempted block.

Then

\[
\boxed{
\Pr(
\text{any block-end verification error}
)
\le
\delta.
}
\]

### Proof

Let \(E_s\) be the event that block \(s\) is misclassified.

By the conditional theorem,

\[
\Pr(E_s\mid\mathcal G_s)\le\delta_s.
\]

Therefore

\[
\Pr\left(\bigcup_sE_s\right)
\le
\sum_s\Pr(E_s)
=
\sum_s
\mathbb E[
\Pr(E_s\mid\mathcal G_s)
]
\le
\mathbb E\sum_s\delta_s
\le
\delta.
\]

\(\square\)

### Known finite number of blocks

If at most \(J\) blocks are attempted, the simplest choice is

\[
\delta_s=\delta/J.
\]

Then

\[
\boxed{
m_s
=
O\!\left(
\frac{V_s}{\Gamma_s^2}
\log\frac{J}{\delta}
\right).
}
\]

Hence total fresh labels obey

\[
\boxed{
B
=
O\!\left(
\log\frac{J}{\delta}
\sum_{s=1}^J
\frac{V_s}{\Gamma_s^2}
\right).
}
\]

This is a valid online familywise statement.

### Weighted confidence allocation

If deterministic difficulty envelopes

\[
\bar h_s
\ge
V_s/\Gamma_s^2
\]

are known in advance, one may use

\[
\delta_s
=
\delta
\frac{\bar h_s}{\sum_j\bar h_j}
\]

to recover the weighted logarithmic form

\[
\sum_s
\bar h_s
\log
\frac{
\sum_j\bar h_j
}{
\delta\bar h_s
}.
\]

If future \(h_s\) are unknown, this oracle allocation should **not** be presented as directly implementable; equal or summable predictable \(\delta_s\) schedules remain valid.

---

# 8. Corollary XXII.5 — Forward reuse of one trusted stream

After block \(s\) has been certified using \(S_s\), the complete batch \(S_s\) may be fed into:

- verifier training;
- policy optimization;
- feature construction;
- candidate selection;
- future audit design.

That is, \(S_s\) may become part of the next transcript

\[
\mathcal G_{s+1}.
\]

This does not invalidate the block-\(s\) guarantee because \(q_s\) was fixed before \(S_s\) was observed.

Therefore:

\[
\boxed{
\text{one stream of trusted batches can be fresh for the current candidate
and reusable for all future candidates.}
}
\]

This formalizes the temporal idea behind the post-v15 single-stream interpretation.

It also explains why the earlier “sealed audit and fresh audit are two structurally necessary resources” conjecture was false.

The division is temporal, not necessarily physical:

\[
\boxed{
\text{fresh now}
\;\longrightarrow\;
\text{reusable later}.
}
\]

---

# 9. Conditional lower bound: when genuinely fresh information is necessary

A generic lower bound cannot simply sum one-step costs across blocks.

Why?

Because old trusted data may already determine the reward direction relevant to the new block.

A blockwise lower bound requires **residual statistical novelty**.

## Definition — conditionally fresh verification direction

At block \(s\), call a reward direction conditionally fresh if, given the old transcript \(\mathcal G_s\), there remain two reward worlds

\[
f_+,\ f_-\in\mathcal F_s
\]

such that:

1. the law of the old transcript is identical under the two worlds;
2. their correct block decisions are opposite:
   \[
   L_{p_s,q_s}(f_\pm)=\pm\Gamma_s.
   \]

This says the past contains no information that resolves the current sign.

## Theorem XXII.6 — Conditional fresh-information lower bound

Let \(g_s\) be the Riesz representer and

\[
V_s=\|g_s\|_{L_2(\mu_s)}^2.
\]

Assume \(g_s\) is essentially bounded and the local-margin condition

\[
\Gamma_s
\le
\frac{
V_s
}{
2\|g_s\|_\infty
}
\]

holds.

If the least-favorable pair

\[
f_\pm
=
\pm
\frac{\Gamma_s}{V_s}g_s
\]

is a conditionally fresh direction in the above sense, then any block-end test with worst-case conditional error at most \(\delta_s<1/4\) requires

\[
\boxed{
m_s
=
\Omega\!\left(
\frac{
V_s
}{
\Gamma_s^2
}
\log\frac1{\delta_s}
\right).
}
\]

### Proof

Condition on the common old transcript.

By assumption, all pre-existing information has the same law in the two reward worlds.

Only the current trusted batch can distinguish them.

The conditional one-query KL is the same as in the one-step lower bound:

\[
D_{\rm KL}(P_+\|P_-)
\le
4\frac{\Gamma_s^2}{V_s}.
\]

After \(m_s\) fresh observations,

\[
D_{\rm KL}
\le
4m_s\frac{\Gamma_s^2}{V_s}.
\]

Bretagnolle–Huber then gives the same lower bound. \(\square\)

### Interpretation

This is the correct place where “freshness” enters.

\[
\boxed{
\text{fresh labels are necessary when the composed endpoint probes a reward
direction that remains unresolved after conditioning on all old trusted information.}
}
\]

If old data already identify that direction, this lower bound does not apply.

That is exactly consistent with the shared-structure experiments where current-policy refresh self-corrects.

---

# 10. No generic additive lower bound without innovation

The upper bound can always spend a new batch in every block.

But a lower bound of the form

\[
\sum_s
\frac{V_s}{\Gamma_s^2}
\]

is **not universal** if reward structure is shared.

A single old observation may resolve several future blocks.

Additivity requires an independent/orthogonal innovation construction, such as the existing adaptive moving-tail family.

Thus:

\[
\boxed{
\text{blockwise hardness is generic;
blockwise additive hardness requires blockwise statistical innovation.}
}
\]

This distinction prevents the new composed theorem from reintroducing the old error “fresh verification every round.”

---

# 11. Full unstructured class: exact divergence formulas

Take the unrestricted class

\[
\mathcal F=L_2(\mu).
\]

When \(p,q\ll\mu\),

\[
g(y)
=
\frac{dq}{d\mu}(y)
-
\frac{dp}{d\mu}(y).
\]

Hence

\[
\boxed{
\mathcal V_{\rm full}(p,q;\mu)
=
\int
\frac{(dq-dp)^2}{d\mu}.
}
\]

Three audit choices are especially informative.

## Current-policy audit: \(\mu=p\)

\[
\boxed{
V_{\rm current}
=
\chi^2(q\|p).
}
\]

This is the original distribution-shift verification tax.

## Candidate audit: \(\mu=q\)

\[
\boxed{
V_{\rm candidate}
=
\chi^2(p\|q).
}
\]

This can be extremely large if the candidate suppresses regions important under \(p\).

## Balanced audit: \(\mu=(p+q)/2\)

\[
V_{\rm bal}
=
2
\int
\frac{(q-p)^2}{p+q}.
\]

Since

\[
(q-p)^2
\le
(p+q)^2,
\]

\[
\boxed{
V_{\rm bal}
\le4.
}
\]

Therefore, for bounded rewards and a fixed selected block endpoint,

\[
\boxed{
m_{\rm bal}
=
O\!\left(
\frac1{\Gamma^2}
\log\frac1\delta
\right)
}
\]

independently of how large the raw density ratio \(q/p\) becomes.

This is the clean mathematical reason a balanced \(p/q\) audit defeated the earlier two-audit necessity conjecture.

It is not a novel statistical fact; it is essentially the bounded-variance two-sample mean-difference principle expressed in the structured-verification notation.

---

# 12. Exponential stale block under current-policy auditing

For a frozen verifier \(v\), define

\[
q_\Lambda(dy)
=
\frac{
e^{\Lambda v(y)}p(dy)
}{
Z(\Lambda)
},
\qquad
Z(\Lambda)
=
\mathbb E_p[e^{\Lambda v}].
\]

Let

\[
\psi(\Lambda)
=
\log Z(\Lambda).
\]

For current-policy audit \(\mu=p\),

\[
V(\Lambda)
=
\chi^2(q_\Lambda\|p).
\]

Exactly,

\[
\boxed{
1+V(\Lambda)
=
\frac{
Z(2\Lambda)
}{
Z(\Lambda)^2
}
=
\exp[
\psi(2\Lambda)-2\psi(\Lambda)
].
}
\]

## Monotonicity

For \(\Lambda\ge0\),

\[
\frac{d}{d\Lambda}
\log(1+V(\Lambda))
=
2[
\psi'(2\Lambda)-\psi'(\Lambda)
]
\ge0,
\]

because \(\psi\) is convex.

Therefore

\[
\boxed{
\chi^2(q_\Lambda\|p)
\text{ is nondecreasing in cumulative stale strength }\Lambda.
}
\]

This formalizes the intuition that a stale current-policy audit sees progressively worse coverage as the optimizer repeatedly tilts away from it.

---

# 13. But verification hardness is \(V/\Gamma^2\), not \(V\)

Let the actual true block gain be

\[
\Delta_r(\Lambda)
=
\mathbb E_{q_\Lambda}[r]
-
\mathbb E_p[r].
\]

For sign verification at the realized margin, the relevant normalized hardness is

\[
\boxed{
h(\Lambda)
=
\frac{
V(\Lambda)
}{
\Delta_r(\Lambda)^2
}.
}
\]

This corrects another tempting but false simplification:

> larger policy shift does not necessarily mean larger sample complexity.

The margin changes too.

---

# 14. Local asymptotic law near the refresh point

Assume the relevant moments exist and

\[
\operatorname{Cov}_p(r,v)\ne0.
\]

For small \(\Lambda\),

\[
\psi(\Lambda)
=
\Lambda\mathbb E_pv
+
\frac{\Lambda^2}{2}
\operatorname{Var}_p(v)
+
O(\Lambda^3).
\]

Therefore

\[
\boxed{
V(\Lambda)
=
\Lambda^2
\operatorname{Var}_p(v)
+
O(\Lambda^3).
}
\]

Also,

\[
\frac{d}{d\Lambda}
\mathbb E_{q_\Lambda}[r]
\Big|_{\Lambda=0}
=
\operatorname{Cov}_p(r,v),
\]

so

\[
\boxed{
\Delta_r(\Lambda)
=
\Lambda
\operatorname{Cov}_p(r,v)
+
O(\Lambda^2).
}
\]

Hence

\[
\boxed{
\lim_{\Lambda\downarrow0}
h(\Lambda)
=
\frac{
\operatorname{Var}_p(v)
}{
\operatorname{Cov}_p(r,v)^2
}.
}
\]

### Interpretation

Near a freshly fitted policy, shift and useful signal both grow linearly in \(\Lambda\).

The shift variance \(V\) grows quadratically.

The squared true gain also grows quadratically.

Their ratio therefore approaches a finite constant.

So:

\[
\boxed{
\text{small stale movement is not automatically harder to verify merely because
more optimization steps have accumulated.}
}
\]

---

# 15. Quadratic blow-up at a true-gain zero crossing

Suppose there is a finite

\[
\Lambda_\star>0
\]

such that

\[
\Delta_r(\Lambda_\star)=0,
\]

with

\[
V(\Lambda_\star)>0
\]

and a transversal crossing

\[
\Delta_r'(\Lambda_\star)\ne0.
\]

Then

\[
\Delta_r(\Lambda)
=
\Delta_r'(\Lambda_\star)
(\Lambda-\Lambda_\star)
+
o(|\Lambda-\Lambda_\star|).
\]

Therefore

\[
\boxed{
h(\Lambda)
\sim
\frac{
V(\Lambda_\star)
}{
[\Delta_r'(\Lambda_\star)]^2
(\Lambda-\Lambda_\star)^2
}.
}
\]

Thus direct sign verification becomes hardest **near the point where the true gain is nearly zero**.

This is independent of the self-certificate pathology.

It is ordinary statistical indistinguishability of a near-zero effect.

Past the crossing, if \(|\Delta_r|\) grows again, the fresh-audit cost can fall.

This gives a theorem-level explanation for the earlier empirical observation:

> fresh candidate auditing is extremely expensive near the harmful/beneficial boundary but becomes easier again farther into the harmful regime.

---

# 16. Numerical sanity check on the learned code-generator environment

The learned autoregressive DSL generator from Updates XVII–XXI gives:

\[
\Lambda_\star
\approx
22.5409.
\]

The local current-audit hardness limit predicted by

\[
\frac{
\operatorname{Var}_p(v)
}{
\operatorname{Cov}_p(r,v)^2
}
\]

is

\[
\boxed{
55.661.
}
\]

At \(\Lambda=1\), the measured

\[
V_{\rm current}/\Delta^2
\approx59.6,
\]

already close to the asymptotic value.

Near the true-gain crossing, hardness blows up.

| lambda_stale | true_gain | V_current | V_balanced | V_candidate | h_current | h_balanced | h_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | 0.0172 | 0.0176 | 0.0177 | 0.0181 | 59.6257 | 59.7832 | 61.3177 |
| 5.0000 | 0.0669 | 0.4437 | 0.3772 | 0.5551 | 99.1337 | 84.2803 | 124.0341 |
| 10.0000 | 0.0824 | 2.0586 | 1.1129 | 4.3797 | 303.0134 | 163.8050 | 644.6611 |
| 15.0000 | 0.0593 | 5.3790 | 1.8129 | 33.0290 | 1528.0322 | 514.9907 | 9382.6247 |
| 20.0000 | 0.0203 | 10.1001 | 2.3655 | 321.3774 | 24628.3400 | 5768.0470 | 783658.3141 |
| 22.0000 | 0.0042 | 12.1353 | 2.5431 | 856.1309 | 682604.2766 | 143046.8242 | 48157059.0123 |
| 22.5000 | 0.0003 | 12.6421 | 2.5839 | 1099.5786 | 126275694.1334 | 25809127.9181 | 10983190427.1120 |
| 23.0000 | -0.0035 | 13.1459 | 2.6233 | 1415.0285 | 1057763.2557 | 211078.7629 | 113857571.9675 |
| 25.0000 | -0.0182 | 15.1150 | 2.7679 | 3951.7533 | 45600.9549 | 8350.7144 | 11922182.7708 |
| 30.0000 | -0.0492 | 19.5008 | 3.0507 | 57126.5184 | 8056.3539 | 1260.3173 | 23600628.0514 |
| 40.0000 | -0.0882 | 25.4400 | 3.3850 | 15889978.6211 | 3270.2237 | 435.1350 | 2042604976.3897 |

The balanced audit keeps its raw coverage term bounded; in this entire numerical sweep,

\[
V_{\rm balanced}<3.49<4.
\]

Nevertheless

\[
V_{\rm balanced}/\Delta^2
\]

still diverges at \(\Lambda_\star\), because the **margin** vanishes.

![True gain over stale strength](composed_block_true_gain.png)

![Verification hardness](composed_block_verification_hardness.png)

This is an important conceptual separation:

\[
\boxed{
\text{current audit suffers both shift and small-margin cost;}
}
\]

\[
\boxed{
\text{balanced fresh audit removes the shift blow-up but cannot remove
the fundamental }1/\Delta^2\text{ sign-testing cost.}
}
\]

---

# 17. Block batching can reduce verification frequency

The block-compression theorem suggests a practical architecture:

1. freeze a verifier;
2. perform several internal optimization steps;
3. do **not deploy them irreversibly**;
4. audit the block endpoint against the block start;
5. accept the entire block or roll back;
6. feed the fresh trusted batch into the next verifier refresh.

This allows one trusted audit to certify several internal optimization steps.

In the learned-generator population environment with per-step strength

\[
\eta=3,
\]

the first-block tradeoff is:

| stale_steps_L | cumulative_strength_etaL | block_true_gain | block_end_reward | minimum_reward_inside_block | Hoeffding_m_per_distribution_for_95pct_positive_certificate | total_fresh_labels_for_block |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | 3.0000 | 0.0462 | 0.6173 | 0.5712 | 4112.0000 | 8224.0000 |
| 2.0000 | 6.0000 | 0.0740 | 0.6452 | 0.5712 | 1600.0000 | 3200.0000 |
| 3.0000 | 9.0000 | 0.0831 | 0.6543 | 0.5712 | 1270.0000 | 2540.0000 |
| 4.0000 | 12.0000 | 0.0766 | 0.6477 | 0.5712 | 1496.0000 | 2992.0000 |
| 5.0000 | 15.0000 | 0.0593 | 0.6305 | 0.5712 | 2490.0000 | 4980.0000 |
| 6.0000 | 18.0000 | 0.0365 | 0.6077 | 0.5712 | 6577.0000 | 13154.0000 |
| 8.0000 | 24.0000 | -0.0110 | 0.5602 | 0.5602 | — | — |

The largest first-block gain occurs at

\[
L=3,
\]

where

\[
\Delta_{\rm block}
\approx0.0831.
\]

A simple Hoeffding two-sample certificate at 95% confidence would require about

\[
2540
\]

fresh semantic labels total for that block, versus about

\[
8224
\]

for the smaller \(L=1\) gain under the same conservative bound.

But \(L=8\) already has negative net block gain.

Thus there is an operational tradeoff:

\[
\boxed{
\text{too-frequent verification gives small hard-to-certify gains;}
}
\]

\[
\boxed{
\text{too-infrequent verification allows stale overoptimization;}
}
\]

with an intermediate block length potentially giving the best progress per trusted label.

### Caution

This is an illustrative oracle/population calculation, not yet an optimized algorithm.

The verifier fit cost, unknown margin, adaptive block-length choice, and familywise confidence accounting all matter in a deployable procedure.

---

# 18. What this theorem does and does not establish

## Established

1. **Exact stale-block composition** for frozen exponential verifiers.
2. **Exact reduction of block-end verification to one-step endpoint verification.**
3. A conditional blockwise minimax upper bound
   \[
   O(V_s\Gamma_s^{-2}\log(1/\delta_s)).
   \]
4. A matching conditional lower bound when the block probes a reward direction unresolved by the old transcript.
5. **Familywise correctness over adaptively generated blocks** under predictable summable error allocation.
6. **Forward reuse:** the same trusted batch can certify the current candidate and train all future verifiers.
7. In the full class:
   - current audit \(V=\chi^2(q\|p)\);
   - candidate audit \(V=\chi^2(p\|q)\);
   - balanced audit \(V\le4\).
8. Under exponential stale movement:
   \[
   1+\chi^2(q_\Lambda\|p)
   =
   e^{\psi(2\Lambda)-2\psi(\Lambda)}.
   \]
9. Local hardness is finite near \(\Lambda=0\) when covariance is nonzero.
10. Hardness diverges quadratically near a transversal true-gain zero crossing.

## Not established

1. A universal additive lower bound across arbitrary recursive blocks.
2. A universal critical \(\eta L\).
3. A universal KL or density-ratio threshold for harm.
4. A theorem saying every stale verifier must fail.
5. A theorem saying fresh candidate labels are necessary when old data already identify the relevant reward direction.
6. Strict safety of every intermediate policy inside a rollback block.
7. Optimal adaptive choice of block length.

---

# 19. Relationship to the adaptive moving-tail theorem

The new theorem and the old moving-tail lower bound are complementary.

## Shared-structure world

The conditional residual reward class shrinks as trusted data accumulate.

Eventually the current block direction may be determined by old data.

Then the lower-bound freshness assumption fails, correctly allowing reuse.

## Persistent moving-tail world

Every block reaches a new reward/error coordinate whose sign remains conditionally unresolved.

Then the conditional lower bound activates repeatedly.

In the orthogonal adaptive-tree family, these costs accumulate, recovering the earlier linear-in-innovation lower bounds.

Therefore the common principle is:

\[
\boxed{
\text{fresh verification is required exactly when the composed stale candidate
creates a verification functional that remains statistically unresolved by the old transcript.}
}
\]

This is more precise than either:

- “fresh labels are needed every round,” or
- “fresh labels are never needed if the audit is sealed.”

---

# 20. Relationship to Theorem U

The composed-block theorem concerns **external trusted verification of the endpoint**.

Theorem U concerns **same-model self-certification**.

They are different.

For the canonical linear plug-in architecture,

\[
\widehat\Delta
=
\eta\|\widehat\theta\|_\Sigma^2
\ge0,
\]

so the self-certificate cannot reject.

The blockwise external audit instead estimates

\[
\Delta
=
\mathbb E_qr-\mathbb E_pr
\]

from trusted labels.

Thus the two results fit together:

\[
\boxed{
\text{optimization can make internal evidence structurally optimistic,
while fresh endpoint evidence remains statistically valid.}
}
\]

The fresh evidence may be expensive near \(\Delta=0\), but that is an ordinary small-margin problem, not self-evaluation blindness.

---

# 21. Novelty assessment

The result is useful but should be described accurately.

The core proof consists of:

- an elementary exponential-tilt composition identity;
- the previously proved one-step restricted-\(\chi^2\) / Riesz verification theorem;
- conditional probability / union-bound accounting.

Therefore:

\[
\boxed{
\text{XXII.3 by itself is not a Spotlight-level new theorem.}
}
\]

Its value is as a **bridge theorem**.

It makes the paper story coherent:

\[
\text{one-step structured coverage}
\longrightarrow
\text{composed stale candidate}
\longrightarrow
\text{recursive refresh blocks}.
\]

The genuinely new/high-upside theorem would still need to characterize the **optimal adaptive refresh frontier** in an endogenous family, rather than merely apply a one-step theorem after a block has been chosen.

---

# 22. Next theoretical target after this result

The right next target is now very narrow.

Define the residual uncertainty after old transcript \(\mathcal H_s\), and a block hardness

\[
h_s(L)
=
\frac{
\mathcal V_{\mathcal F_s^{\rm residual}}
(p_s,q_{s:L};\mu_s)
}{
\Gamma_s(L)^2
}.
\]

Then ask:

> **Given a total trusted-label budget, when should the verifier be refreshed so that useful progress per label is maximized while familywise block safety is maintained?**

A satisfactory result would derive an adaptive stopping / refresh rule from \(h_s(L)\) or a valid upper confidence envelope for it.

This would be more than the current bridge theorem because it would optimize the endogenous choice of \(L\).

However, do not pursue this immediately unless the rule can be made observable without knowing the true margin \(\Gamma_s(L)\).

If it requires oracle access to the true reward gain, the direction should be stopped.

---

# 23. Research decision

This update successfully closes the specific task posed after Update XXI.

The early theory and latest experiments can now be joined without inventing a new metric:

\[
\boxed{
\text{recursive lag}
\;\to\;
\text{composed endpoint }q_{s:L}
\;\to\;
\mathcal V_{\mathcal F}(p_s,q_{s:L};\mu_s)
\;\to\;
\text{trusted-label complexity}.
}
\]

The key conceptual advance is not a new divergence.

It is the identification of the **correct unit of recursive verification**:

\[
\boxed{
\text{a statistically novel composed policy comparison between verifier refreshes.}
}
\]

This should replace “one verification payment per round” in the final paper framing.
