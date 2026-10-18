---
layout: default
title: "Statistical Learning Theory"
date: 2026-04-20
# excerpt: "Notes on 'SDEs & Diffusion Models' (SS 2026) — starting with a measure-theoretic outline of Brownian motion, Donsker's theorem, Markov properties, and a panoramic introduction to stochastic differential equations, Itô calculus, the Fokker–Planck equation, time-reversal of diffusions, and score-based generative modelling."
tags:
#   - stochastic-differential-equations
#   - brownian-motion
#   - diffusion-models
#   - markov-processes
#   - score-matching
#   - fokker-planck
#   - time-reversal
#   - ito-calculus
#   - stochastic-integration
#   - martingales
#   - girsanov
#   - ornstein-uhlenbeck
# math: true
---

<style>
  .accordion summary {
    font-weight: 600;
    color: var(--accent-strong, #2c3e94);
    background-color: var(--accent-soft, #f5f6ff);
    padding: 0.35rem 0.6rem;
    border-left: 3px solid var(--accent-strong, #2c3e94);
    border-radius: 0.25rem;
  }
</style>

**Table of Contents**
- TOC
{:toc}


<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(1.1(a) — Arithmetic–Geometric Mean Inequality)</span></p>

Let $m \in \mathbb{N}$ and $x_1,\ldots,x_m \geq 0$. Prove the **arithmetic–geometric mean inequality**:

$$\frac{1}{m}\sum_{i=1}^{m}x_i \geq \left(\prod_{i=1}^{m}x_i\right)^{1/m}$$

**Hint:** The natural logarithm $\ln:(0,\infty)\to\mathbb R$ is a concave function.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(1.1(b) — ERM with Average Distribution Risk)</span></p>

Let $\mathcal H$ be a finite hypothesis class of binary classifiers $h:\mathcal X\to\mathcal Y$, where $\mathcal Y=\lbrace 0,1\rbrace$.

Suppose we have a training sample

$$S:=((X_1,Y_1),\ldots,(X_m,Y_m))$$

of size $m\in\mathbb N$, where the random variables

$$X_i\sim D_i,\qquad i=1,\ldots,m,$$

are **independent, but not necessarily identically distributed**, and each $D_i$ is a probability distribution on $\mathcal X$.

Define the **average distribution**

$$\overline D_m:=\frac1m\sum_{i=1}^{m}D_i.$$

Assume that the sample is labelled by an unknown target function $f\in\mathcal H$, meaning

$$Y_i=f(X_i),\qquad i=1,\ldots,m.$$

**Proposition.** Let $\delta,\varepsilon\in(0,1)$. Suppose that

$$m\geq\frac1\varepsilon \ln\left(\frac{|\mathcal H|}{\delta}\right).$$

Show that, with probability less than $\delta$, there exists some $h\in\mathcal H$ such that

$$L_{(\overline D_m,f)}(h)>\varepsilon \qquad\text{and}\qquad L_{(S,f)}(h)=0.$$

Here, the two quantities denote the **true error** under the average distribution and the **empirical error**, respectively.

**Hints:**

1. First, formalise the claim of the proposition mathematically.
2. Use part (a), together with the elementary inequality

   $$1-x\leq e^{-x},\qquad x\in\mathbb R.$$

3. Conclude by establishing the requirement on the sample size $m$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

* Recall
  
  $$L_{(\overline D_m,f)}(h) = \frac{1}{m} \sum_{n=1}^m \mathbb{P}_{X_i \sim D_i}(h(X_i) \neq f(X_i)).$$

* Mathematical formulation of the proposition is 
  
  $$\mathbb{P}(\bigcup_{h\in H, L_{(\overline D_m,f)}(h) \gt \varepsilon} \lbrace L_{(S,f)}(h) = 0 \rbrace) \lt \delta.$$

* Expand LHS using Union Bound:

  $$\mathbb{P}(\bigcup_{h\in H, L_{(\overline D_m,f)}(h) \gt \varepsilon} \lbrace L_{(S,f)}(h) = 0 \rbrace) \leq \sum_{h \in H_\varepsilon} \mathbb{P}(L_{(S,f)}(h) = 0),$$

  where $H_\varepsilon := \lbrace h\in H \mid L_{(\overline D_m,f)}(h) \gt \varepsilon \rbrace$.

* The strategy to continue the proof is the following:
  1. We have the probability bound, which is $L_{(\overline D_m,f)}(h) = \frac{1}{m} \sum_{n=1}^\infty \mathbb{P}_{X_i \sim D_i}(h(X_i) \neq f(X_i)) \gt \varepsilon$, which is out the only bridge from abstract probabilities to numerical values.
  2. We start from the product relying on the independence of $X_1,\ldots,X_m$.
  3. Then move to sums via AM-GM inequality.
  4. Bound sums using the bound from the point 1. and the elementary inequality from the hint.
   
* The RHS $\mathbb{P}(L_{(S,f)}(h) = 0)$ expresses the probability that $h$ is ideal on the sample $S$, which could be also expresses via the product in ($1-$ "$h$ is somewhere wrong"):
   
   $$
   \begin{aligned}
   1 - (1 - \prod_{i=1}^m (1-\mathbb{P}_{X_i \sim D_i}(h(X_i) \neq f(X_i)))) \\
   &= \prod_{i=1}^m (1-\mathbb{P}_{X_i \sim D_i}(h(X_i) \neq f(X_i))) \\
   &\overset{\text{AM-GM}}{\leq} (\frac{1}{m} \sum_{i=1}^m (1-\mathbb{P}_{X_i \sim D_i}(h(X_i) \neq f(X_i))))^m \\
   &= (1 - \frac{1}{m} \sum_{i=1}^m \mathbb{P}_{X_i \sim D_i}(h(X_i) \neq f(X_i)))^m \\
   &\overset{\text{given}}{\lt} (1 - \varepsilon)^m \\
   &\overset{\text{elem.}}{\leq} \exp(-m\cdot \varepsilon).
   \end{aligned}
   $$

* Coming back to the RHS:

  $$\sum_{h \in H_\varepsilon} \mathbb{P}(L_{(S,f)}(h) = 0) \leq \sharp H_\varepsilon \cdot \exp(-m\cdot \varepsilon) \overset{\text{assum.}}{\lt} \delta.$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(ERM with average distribution risk)</span></p>

The underlying proof strategy is particularly transferable:

$$
\begin{gathered}
\text{Independence}\\
\downarrow\\
\text{Product of probabilities}\\
\downarrow\quad\text{AM--GM}\\
\text{Exponential bound}\\
\downarrow\quad\text{Union bound}\\
\text{Uniform guarantee}.
\end{gathered}
$$

This is useful throughout probability, empirical process theory, concentration inequalities, and statistical learning. Crucially, it is useful to **generalize a theorem by examining which hypotheses its proof actually needs**.

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bayes Optimal Predictor)</span></p>

Let $(X,Y)\sim D$, where $D$ is a probability distribution over $\mathcal X\times\mathcal Y$, with $\mathcal Y=\lbrace 0,1\rbrace$.

The **Bayes optimal predictor** $f_D:\mathcal X\to\mathcal Y$ is defined by

$$
f_D(x):=
\begin{cases}
1, & \mathbb P(Y=1\mid X=x)\geq \frac12,\\[4pt]
0, & \mathbb P(Y=1\mid X=x)<\frac12.
\end{cases}
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Bayes Optimal Predictor)</span></p>

For every probability distribution $D$ over $\mathcal X\times\mathcal Y$, the respective Bayes optimal predictor f_D satisfies

$$f_D\in\operatorname*{arg\,min}_{h:\mathcal X\to\mathcal Y}L_D(h).$$

Here, the minimisation is taken over all possible classifiers $h:\mathcal X\to\mathcal Y$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By definition, for $0-1$ loss for binary classification, the risk of misclassification probability is

$$L_D(h) = \mathbb{P}_D(h(X) \neq Y).$$

We can rewrite the risk in a more useful expectation form via the identity $P(X\in A) = \mathbb{E}[\mathbb{1}\_A X]$ for some RV $X$ and a measurable set $A$:

$$\mathbb{P}_D(h(X) \neq Y) = \mathbb{E}_D[\mathbb{1}\_E],$$

where $E := \lbrace (x,y)\in\mathcal{X}\times\mathcal{Y} \mid h(x)\neq y \rbrace$ is a set of error. Then with $0-1$ loss the value is non-zero only on $E$:

$$
\begin{aligned}
\mathbb{E}_D[\mathbb{1}_E] 
&= \int_E D(dx, dy)
&= \int_{\mathcal{X}\times\mathcal{Y}} l(h, (x,y)) D(dx, dy) \\
&= \int_\mathcal{X} [\int_\mathcal{Y} l(h, (x, y)) D(dy\mid x)]D_X(dx) \\
&= \int_\mathcal{X} \underbrace{[l(h, (x, 0))\mathbb{P}_D(Y=0\mid X=x) + l(h, (x, 1))\mathbb{P}_D(Y=1\mid X=x)]}_{=: r_x(h(x))}D_X(dx),
\end{aligned}
$$

assuming a suitable conditional distribution exists. If such exists, then we can **minimise the integrand pointwise**. Now we establish the decision rule for the classifier that minimizes the integrand and observe that it coincides with the Bayes Optimal Predictor. 

The **key observation** is that the conditional risk $r_x$ takes only two possible values:

$$r_x(0)=\eta(x),\qquad r_x(1)=1-\eta(x).$$

Consequently,

$$
\operatorname_{arg\,min}_{a\in\lbrace 0,1\rbrace}r_x(a)
=
\begin{cases}
\lbrace 1\rbrace,&\eta(x)>1/2,\\
\lbrace 0,1\rbrace,&\eta(x)=1/2,\\
\lbrace 0\rbrace,&\eta(x)<1/2.
\end{cases}
$$

This is precisely the decision rule defining $f_D$, with ties resolved in favour of $1$.

**Complete the argument by integrating**

Because $f_D$ minimises the conditional risk at every $x$ (up to $D_X$-null sets), for every measurable classifier $h$,

$$r_x(f_D(x))\le r_x(h(x)) \qquad D_X\text{-a.e.}$$

Integrating gives

$$L_D(f_D) =\int_{\mathcal X}r_x(f_D(x))\,D_X(dx) \le \int_{\mathcal X}r_x(h(x))\,D_X(dx) =L_D(h).$$

Since $h$ was arbitrary, this proves the proposition.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bayes Optimal Predictor)</span></p>

An elementary but exceptionally general idea: **minimizing a global expected loss can sometimes be reduced to minimizing the conditional loss pointwise**.

$$\inf_h\mathbb E[\ell(h(X),Y)]$$

becomes a family of conditional optimization problems indexed by $x$. This pattern appears in statistical decision theory, Bayesian estimation, regression, optimal prediction, and classification-calibrated losses. **The proof is short, but the principle behind it is extremely valuable.**

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Excess classification risk)</span></p>

The proof of **Bayes Optimal Predictor** actually establishes a stronger identity:

$$L_D(h)-L_D(f_D) = \mathbb E\left[ |2\eta(X)-1| \mathbf 1_{\{h(X)\ne f_D(X)\}} \right].$$

This exactly characterises the **excess classification risk**. In particular, equality holds if and only if $h=f_D$ almost surely on the set where $\eta(X)\ne 1/2$. On the tie set, either prediction is optimal.

</div>


<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Agnostic PAC Learnability Implies PAC Learnability)</span></p>

Let $\mathcal H:=\lbrace h:\mathcal X\to\mathcal Y\rbrace$ be a hypothesis class of binary classifiers on $\mathcal X$, where $\mathcal Y=\lbrace 0,1\rbrace$.

Verify that **agnostic PAC learnability** is a generalisation of **PAC learnability** in the following sense:

1. If the class $\mathcal H$ is agnostically PAC learnable, then it is PAC learnable as well.
2. Moreover, if a learning algorithm $A$ is a successful agnostic PAC learner for $\mathcal H$, then $A$ is also a successful PAC learner for $\mathcal H$.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(1.3 — No Free Lunch Theorem)</span></p>

Assume that $\lvert \mathcal X\rvert <\infty$.

Recall the **No Free Lunch Theorem (Theorem 1.5)**:

**Theorem.** Let \(A\) be an arbitrary learning algorithm for binary classification with respect to the \(0\)-\(1\) loss over \(\mathcal X\).

Consider a training set \(S\) of size \(m\in\mathbb N\) such that

$$m<\frac{|\mathcal X|}{2}.$$

Then there exists a probability distribution $D$ over \(\mathcal X\times\mathcal Y\) such that:

1. There exists a target function \(f:\mathcal X\to\mathcal Y\) satisfying

   $L_D(f)=0.$

2. With probability at least $1/7$, over the choice of an i.i.d. training sample $S\sim D^m$, it holds that

   $$\mathbb P_{S\sim D^m} \left(L_D(A(S))\geq\frac18\right) \geq\frac17.$$

Recapitulate and complete the proof of this theorem by working on the following tasks.

**(a) Proof of Theorem 1.5(i).**

Sketch the proof of Theorem 1.5(i); see the lecture script, pp. 7–8.

Summarise its core idea and describe the main argument in your own words.

**(b) Proof of Theorem 1.5(ii).**

Employ the bound

$$\mathbb E_D[L_D(A(S))]\geq\frac14,$$

arising from the former argument, to infer the second part of Theorem 1.5.

More concretely, show the implication

$$\mathbb E_D[L_D(A(S))]\geq\frac14 \quad\Longrightarrow\quad \mathbb P_D\left(L_D(A(S))\geq\frac18\right) \geq\frac17.$$

**Hint:** Start by verifying the following general result.

**Lemma.** Let \(L:\Omega\to[0,1]\) be a random variable with expectation \(\mu:=\mathbb E[L]<\infty\). Then, for every \(a\in(0,1)\),

$$\mathbb P(L>a)\geq\frac{\mu-a}{1-a}.$$

</div>
