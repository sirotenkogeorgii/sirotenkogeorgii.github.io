---
layout: default
title: Introduction to Learning Theory
date: 2024-11-01
# excerpt: Connecting differential equations, stability analysis, and attractor theory to the training dynamics of modern machine learning models.
# tags:
#   - dynamical-systems
#   - machine-learning
#   - theory
---

# The Classic Theory for Supervised Learning

## Classic Learning Theory: Setup and Main Questions

Statistical learning theory (SLT) tries to answer questions such as:

* Which **learning tasks** can be performed by computers (positive and negative results)?
* What kind of **assumptions** do we have to make such that machine learning can be successful?
* What are the key properties a **learning algorithm** needs to satisfy in order to be successful?
* Which **performance guarantees** can we give on the results of certain learning algorithms?

In the following we focus on the case of **binary classification**.

## The Framework

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Setup)</span></p>

**The data.** Data points $(X_i, Y_i)$ are an i.i.d. sample from some underlying (unknown) probability distribution $P$ on the space $\mathcal{X} \times \lbrace \pm 1 \rbrace$.

**Goal.** Learn a deterministic function $f : \mathcal{X} \to \lbrace \pm 1 \rbrace$ such that the expected loss (risk) according to some given loss function $\ell$ is as small as possible. In classification, the natural loss function is the **0-1 loss**.

</div>

**Assumptions we (do not) make:**

* We do not make any assumption on the **underlying distribution** $P$ that generates our data — it can be anything.
* True **labels** do not have to be a deterministic function of the input (consider the example of predicting male/female based on body height).
* Data points have been sampled **i.i.d.**
* Data does not change over time (the ordering of the training points does not matter, and the distribution $P$ does not change).
* The distribution $P$ is unknown at the time of learning.

### The Bayes Classifier

The Bayes classifier for a particular learning problem is the classifier that achieves the **minimal expected risk**. If we knew the underlying distribution $P$, then we would also know the Bayes classifier (just look at the regression function).

The challenge is that we do not know $P$. The goal is now to construct a classifier that is "as close to the Bayes classifier" as possible.

## Convergence and Consistency

Assume we have a set of $n$ points $(X_i, Y_i)$ drawn from $P$. Consider a given function class $\mathcal{F}$ from which we are allowed to pick our classifier. Denote:

* $f^\*$ the Bayes classifier corresponding to $P$.
* $f\_\mathcal{F}$ the best classifier in $\mathcal{F}$, that is $f\_\mathcal{F} = \arg\min\_{f \in \mathcal{F}} R(f)$.
* $f_n$ the classifier chosen from $\mathcal{F}$ by some training algorithm on the given sample of $n$ points.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Consistency)</span></p>

1. A learning algorithm is called **consistent with respect to $\mathcal{F}$ and $P$** if the risk $R(f_n)$ converges in probability to the risk $R(f\_\mathcal{F})$ of the best classifier in $\mathcal{F}$, that is for all $\varepsilon > 0$,

   $$P(R(f_n) - R(f_\mathcal{F}) > \varepsilon) \to 0 \quad \text{as } n \to \infty.$$

2. A learning algorithm is called **Bayes-consistent with respect to $P$** if the risk $R(f_n)$ converges to the risk $R(f^\*)$ of the Bayes classifier, that is for all $\varepsilon > 0$,

   $$P(R(f_n) - R(f^*) > \varepsilon) \to 0 \quad \text{as } n \to \infty.$$

3. A learning algorithm is called **universally consistent with respect to $\mathcal{F}$** (resp. **universally Bayes-consistent**) if it is consistent with respect to $\mathcal{F}$ (resp. Bayes-consistent) for all probability distributions $P$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Consistency and Errors)</span></p>

Consistency with respect to a fixed function class $\mathcal{F}$ only concerns the **estimation error**, not the approximation error:

* Here consistency means that our decisions are not affected systematically from the fact that we only get to see a finite sample, rather than the full space. In other words, all "finite sample effects" cancel out once we get to see enough data.
* If a learning algorithm is consistent, it means that it does **not overfit** when it gets to see enough data (low estimation error, low variance).
* Consistency with respect to $\mathcal{F}$ does **not** tell us anything about underfitting (approximation error; this depends on the choice of $\mathcal{F}$).

</div>

## Empirical Risk Minimization (ERM)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Empirical Risk Minimization)</span></p>

**True risk** of a function:

$$R(f) = E(\ell(X, f(X), Y)).$$

**Empirical risk:**

$$R_n(f) = \frac{1}{n} \sum_{i=1}^{n} \ell(X_i, f(X_i), Y_i).$$

**Empirical risk minimization:** given $n$ training points $(X_i, Y_i)\_{i=1,\dots,n}$ and a fixed function class $\mathcal{F}$, select the function $f_n$ that minimizes the training error on the data:

$$f_n = \arg\min_{f \in \mathcal{F}} R_n(f).$$

</div>

## Controlling the Estimation Error: Generalization Bounds

### Law of Large Numbers and Concentration

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Law of Large Numbers, simplest version)</span></p>

Let $(Z_i)\_{i \in \mathbb{N}}$ be a sequence of independent random variables that have been drawn according to some probability distribution $P$, denote its expectation as $E(Z)$. Then (under mild assumptions)

$$\frac{1}{n} \sum_{i=1}^{n} Z_i \to E(Z) \quad \text{(almost surely)}.$$

</div>

Note that **independence** is a crucial assumption for the LLN. For example, let $X_1$ be the toss of a fair coin and let $X_2, X_3, \dots$ be identical to $X_1$. Each of the rv's follows the same Bernoulli $\text{Ber}(0.5)$ distribution and they have expectation $E(Z_i) = 0.5$, but they are not independent. The empirical average is either 0 or 1, so it does not converge to 0.5.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Concentration Inequality, Chernoff 1952, Hoeffding 1963)</span></p>

Assume that the random variables $Z_1, \dots, Z_n$ are independent and take values in $[0, 1]$. Define their sum $S_n := (1/n) \cdot \sum_{i=1}^{n} Z_i$. Then for any $\varepsilon > 0$

$$P\!\left(\lvert S_n - E(S_n) \rvert \ge \varepsilon\right) \le 2 \exp(-2n\varepsilon^2).$$

</div>

Observe:
* The rv's don't need to have the same distribution, as long as we can control their range (or more generally, their variance).
* Independence is crucial for this theorem to hold.

### LLN/Hoeffding Applied to Classification Loss

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Risks Converge for Fixed Function)</span></p>

Fix a function $f_0 \in \mathcal{F}$. Then, for this **fixed function** $f_0$,

$$R_n(f_0) \to R(f_0) \quad \text{(almost surely)}.$$

</div>

*Proof.* Apply the Hoeffding bound to the variables $Z_i := \ell(f_0(X_i), Y_i)$. This leads to convergence in probability. For almost sure convergence, apply the Borel-Cantelli lemma (the key is that $\sum_{n=1}^{\infty} \exp(-2n\varepsilon^2)$ is finite).

**Key question:** Let $f_n$ be the function selected by empirical risk minimization based on the first $n$ sample points. Does the LLN imply that $R_n(f_n) - R(f_n) \to 0$?

**No!** The key property in the LLN or Hoeffding inequality is independence. However, the function $f_n$ depends on **all** data points $(X_i, Y_i)\_{i=1,\dots,n}$. So even if the data points $(X_i, Y_i)\_{i=1,\dots,n}$ are independent, the random variables $(Z_i)\_{i=1,\dots,n}$ with $Z_i := \ell(f_n(X_i), Y_i)$ are typically **not independent** any more.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Counter-Example: LLN Fails for ERM)</span></p>

* $\mathcal{X} = [0, 1]$ with uniform distribution; labels deterministic with $x < 0.5 \implies y = -1$ and $x \ge 0.5 \implies y = +1$.
* Draw $n$ training points.
* Define $f_n$ as follows: for all points in the training sample, predict the training label; for all other points predict $-1$.
* Here we have $R_n(f_n) = 0$ but $R(f_n) = 0.5$, for all $n$.
* Or more formally, in the Hoeffding notation: if we set $Z_i := \ell(f_n(X_i), Y_i)$, then we always have $S_n = \sum_i Z_i / n = 0$, but $E(S_n) = 0.5$.

</div>

## Uniform Convergence

We require that **for all functions** in $\mathcal{F}$, the empirical risk (as measured on the data) has to be close to the true risk. Intuitively, for any $\varepsilon > 0$, we want that with high probability,

$$\sup_{f \in \mathcal{F}} \lvert R_n(f) - R(f) \rvert < \varepsilon.$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Uniform Convergence)</span></p>

We say that the **law of large numbers holds uniformly** over a function class $\mathcal{F}$ if for all $\varepsilon > 0$,

$$P\!\left(\sup_{f \in \mathcal{F}} \lvert R(f) - R_n(f) \rvert \ge \varepsilon\right) \to 0 \quad \text{as } n \to \infty.$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Uniform Convergence is Sufficient for Consistency)</span></p>

Let $f_n$ be the function that minimizes the empirical risk in $\mathcal{F}$. Then:

$$P\!\left(\lvert R(f_n) - R(f_\mathcal{F}) \rvert \ge \varepsilon\right) \;\le\; P\!\left(\sup_{f \in \mathcal{F}} \lvert R(f) - R_n(f) \rvert \ge \varepsilon/2\right).$$

</div>

<details class="accordion" markdown="1">
<summary>Proof</summary>

$$\lvert R(f_n) - R(f_\mathcal{F}) \rvert$$

By definition of $f\_\mathcal{F}$ we know that $R(f_n) - R(f\_\mathcal{F}) \ge 0$, so

$$= R(f_n) - R(f_\mathcal{F})$$

$$= R(f_n) - R_n(f_n) + R_n(f_n) - R_n(f_\mathcal{F}) + R_n(f_\mathcal{F}) - R(f_\mathcal{F})$$

Note that $R_n(f_n) - R_n(f\_\mathcal{F}) \le 0$ by definition of $f_n$, so

$$\le R(f_n) - R_n(f_n) + R_n(f_\mathcal{F}) - R(f_\mathcal{F})$$

$$\le 2 \sup_{f \in \mathcal{F}} \lvert R(f) - R_n(f) \rvert.$$

</details>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Vapnik & Chervonenkis, 1971)</span></p>

Let $\mathcal{F}$ be any function class. Then empirical risk minimization is uniformly consistent with respect to $\mathcal{F}$ **if and only if** uniform convergence holds:

$$P\!\left(\sup_{f \in \mathcal{F}} \lvert R(f) - R_n(f) \rvert > \varepsilon\right) \to 0 \quad \text{as } n \to \infty.$$

</div>

The big question is now: **How do we know whether we have uniform convergence for some function class $\mathcal{F}$?**

## Capacity Measures for Function Classes

### Capacity Measures: Intuition

We have seen:

* If a function class is too large (as in the counter-example), then we don't have uniform convergence.
* If a function class is small (say, it only consists of a single function), then we have uniform convergence.

We now want to come up with ways to measure the size of a function class — in such a way that we can bound the term $P(\sup_{f \in \mathcal{F}} \lvert R(f) - R_n(f) \rvert > \varepsilon)$.

### Finite Classes

#### Generalization Bound for Finite Classes

Recall the Hoeffding bound for a fixed function $f_0$:

$$P\!\left(\lvert R(f_0) - R_n(f_0) \rvert \ge \varepsilon\right) \le 2\exp(-2n\varepsilon^2).$$

Now consider a function class with finitely many functions: $\mathcal{F} = \lbrace f_1, \dots, f_m \rbrace$. We get:

$$P\!\left(\sup_{f \in \mathcal{F}} \lvert R(f) - R_n(f) \rvert \ge \varepsilon\right) = P\!\left(\sup_{i=1,\dots,m} \lvert R(f_i) - R_n(f_i) \rvert \ge \varepsilon\right) \le \sum_{i=1}^{m} P\!\left(\lvert R(f_i) - R_n(f_i) \rvert \ge \varepsilon\right) \le 2m\exp(-2n\varepsilon^2).$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Generalization of Finite Classes)</span></p>

Assume $\mathcal{F}$ is finite and contains $m$ functions. Choose any $\varepsilon$, $0 < \varepsilon < 1$. Then, with probability at least $1 - 2m\exp(-2n\varepsilon^2)$, we have for all $f \in \mathcal{F}$ that

$$\lvert R(f) - R_n(f) \rvert < \varepsilon.$$

</div>

Note that this statement is somewhat inconvenient — it is "the wrong way round": we choose the size of the error, and get the probability that this error holds; but we would like to say that with a chosen confidence (error probability), the error is only as large as BLA.

So we reverse the statement: set the probability to some value $\delta$, and solve for $\varepsilon$:

$$\delta = 2m\exp(-2n\varepsilon^2) \implies \varepsilon = \sqrt{\frac{\log(2m) + \log(1/\delta)}{2n}}.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Generalization Bound for Finite Classes)</span></p>

Assume $\mathcal{F}$ is finite and contains $m$ functions. Choose some failure probability $0 < \delta < 1$. Then, with probability at least $1 - \delta$, for all $f \in \mathcal{F}$ we have

$$R(f) \le R_n(f) + \sqrt{\frac{\log(2m) + \log(1/\delta)}{2n}}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Interpreting the Generalization Bound)</span></p>

* It bounds the true risk by the empirical risk plus a "capacity term".
* If the function class gets larger ($m$ increases), then the bound gets worse.
* If $m$ is "small enough" compared to $n$ (in the sense that $\log m / n$ is small), then we get a tight bound.
* The whole bound only holds with probability $1 - \delta$. When we decrease $\delta$ (higher confidence), the bound gets worse.
* If $m$ is fixed, and the confidence value $\delta$ is fixed, and $n \to \infty$, then the empirical risk converges to the true risk. The speed of convergence is of the order $1/\sqrt{n}$.
* If you want to grow your function space with $n$ in order to be able to fit more accurately if you have more data, you need to make sure that $(\log m)/n \to 0$ if you want to get consistency.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Piecewise Constant Functions on a Grid)</span></p>

Consider $\mathcal{X} = [0, 1]$, split it into a grid of $k$ cells of the same size. As function class, consider all functions that are piecewise constant (0 or 1) on all cells. Denote this function class by $\mathcal{F}_k$.

**Case 1: $k$ is fixed.** ERM is uniformly consistent with respect to $\mathcal{F}_k$ since $\mathcal{F}_k$ is finite ($m = 2^k$). Is the classifier also Bayes consistent? Exercise: prove or give a counter-example.

**Case 2: $k$ grows with $n$.** Formally $k$ is a function of $n$, denoted by $k(n)$. How fast can $k(n)$ grow such that we still have consistency with respect to $\mathcal{F}\_{k(n)}$? What about the approximation error / Bayes consistency?

</div>

**Bottom line:** For finite function classes, we can measure the size of $\mathcal{F}$ by its number $m$ of functions. This leads to a generalization bound with plausible behavior. However, what should we do if $\mathcal{F}$ is infinite (say, space of all linear functions)? Then the approach above does not work.

### Shattering Coefficient

We now want to measure the capacity of an infinite class of functions.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Shattering Coefficient)</span></p>

For a given sample $X_1, \dots, X_n \in \mathcal{X}$ and a function class $\mathcal{F}$, define $\mathcal{F}\_{X_1,\dots,X_n}$ as the set of those functions that we get by restricting $\mathcal{F}$ to the sample:

$$\mathcal{F}_{X_1,\dots,X_n} := \lbrace f \vert_{X_1,\dots,X_n} ;\; f \in \mathcal{F} \rbrace.$$

The **shattering coefficient** $\mathcal{N}(\mathcal{F}, n)$ of a function class $\mathcal{F}$ is defined as the maximal number of functions in $\mathcal{F}\_{X_1,\dots,X_n}$:

$$\mathcal{N}(\mathcal{F}, n) := \max\lbrace \lvert \mathcal{F}_{X_1,\dots,X_n} \rvert \;;\; X_1, \dots, X_n \in \mathcal{X} \rbrace.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Right Half-Spaces on $\mathbb{R}$)</span></p>

$\mathcal{X} = \mathbb{R}$, $\mathcal{F} = \lbrace \mathbb{1}\_{[a, \infty[} \mid a \in \mathbb{R} \rbrace$ (all half-spaces with the "infinite end" at the right side). With $\mathcal{F}$ we can only realize 4 out of the 8 possible labelings of 3 points. Hence $\mathcal{N}(\mathcal{F}, 3) = 4$. (Because this argument holds for all possible sets of three points.)

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Half-Spaces in $\mathbb{R}^2$)</span></p>

$\mathcal{X} = \mathbb{R}^2$, $\mathcal{F}$ such that positive class $=$ space above a horizontal line. For different data configurations of 5 points: one arrangement gives 5 different ways to separate, so $\mathcal{N}(\mathcal{F}, 5) \ge 5$; another gives 6 ways, so $\mathcal{N}(\mathcal{F}, 5) \ge 6$; collinear points give only 2 ways.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Interior of Circles in $\mathbb{R}^2$)</span></p>

$\mathcal{X} = \mathbb{R}^2$, $\mathcal{F} =$ interior of circles. Challenge: can you come up with a bound on the shattering coefficient for a small $n$?

</div>

#### Generalization Bound with Shattering Coefficient

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Generalization Bound with Shattering Coefficient)</span></p>

Let $\mathcal{F}$ be any arbitrary function class. Then for all $0 < \varepsilon < 1$,

$$P\!\left(\sup_{f \in \mathcal{F}} \lvert R(f) - R_n(f) \rvert > \varepsilon\right) \le 2\mathcal{N}(\mathcal{F}, 2n) \exp(-n\varepsilon^2/4).$$

The other way round: with probability at least $1 - \delta$, all functions $f \in \mathcal{F}$ satisfy

$$R(f) \le R_n(f) + 2\sqrt{\frac{\log(\mathcal{N}(\mathcal{F}, 2n)) - \log(\delta)}{n}}.$$

</div>

#### Proof of Theorem 39 by Symmetrization

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Symmetrization Lemma)</span></p>

By $R_n$ we denote the risk on our given sample of $n$ points. By $R_n'$ we denote the risk that we get on a second, independent sample of $n$ points, called the "ghost sample". Then:

$$P\!\left(\sup_{f \in \mathcal{F}} \lvert R(f) - R_n(f) \rvert > \varepsilon\right) \;\le\; 2\,P\!\left(\sup_{f \in \mathcal{F}} \lvert R_n(f) - R_n'(f) \rvert > \varepsilon/2\right).$$

</div>

**Key idea of the proof:** The right hand side only depends on the values of the functions $f$ on the two samples. If two functions $f$ and $g$ coincide on all points of the original sample and the ghost sample (i.e., $f(x) = g(x)$ for all $x$ in the samples), then $R_n(f) = R_n(g)$ and $R_n'(f) = R_n'(g)$.

So the supremum over $f \in \mathcal{F}$ in fact only runs over **finitely many functions**: all possible binary functions on the two samples. The number of such functions is bounded by the shattering coefficient $\mathcal{N}(\mathcal{F}, 2n)$. Now Theorem 39 is a consequence of Theorem 38.

### Discussion of the Generalization Bound with Shattering Coefficient

* The bound is analogous to the one for finite function classes, just the number $m$ of functions has been replaced by the shattering coefficient.
* Intuitively, the shattering coefficient measures "how powerful" a function class is, how many different labelings of a data set it can possibly realize.
* **Overfitting** happens if a function class is very powerful and can in principle fit everything. Then we don't get consistency, the shattering coefficient is large. The smaller the shattering coefficient, the less prone we are to overfitting (in the extreme case of one function, we don't overfit).
* To prove consistency of a classifier, we need to establish that $\log \mathcal{N}(\mathcal{F}, 2n) / n \to 0$ as $n \to \infty$. Intuitively: the number of possibilities in which a data set can be labeled has to grow at most **polynomially** in $n$.
* Shattering coefficients are complicated to compute and to deal with. To prove consistency, we would need to know how fast the shattering coefficients grow with $n$ (exponentially or less). We now study a tool that can help us with this.

### VC Dimension

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(VC Dimension)</span></p>

We say that a function class $\mathcal{F}$ **shatters** a set of points $X_1, \dots, X_n$ if $\mathcal{F}$ can realize all possible labelings of the points, that is $\lvert \mathcal{F}\_{X_1,\dots,X_n} \rvert = 2^n$.

The **VC dimension** of $\mathcal{F}$ is defined as the largest number $n$ such that there exists a sample of size $n$ which is shattered by $\mathcal{F}$. Formally,

$$\text{VC}(\mathcal{F}) = \max\lbrace n \in \mathbb{N} \mid \exists\, X_1, \dots, X_n \in \mathcal{X} \;\text{s.t.}\; \lvert \mathcal{F}_{X_1,\dots,X_n} \rvert = 2^n \rbrace.$$

If the maximum does not exist, the VC dimension is defined to be infinity.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Closed Intervals on $\mathbb{R}$)</span></p>

$\mathcal{X} = \mathbb{R}$, $\mathcal{F} = \lbrace \mathbb{1}\_{[a,b]} \mid a, b \in \mathbb{R} \rbrace$ (closed intervals).

* Can find a set of **two** points that can be shattered.
* Can we find a set of **three** points that can be shattered? No! For any three points, we can never realize the labeling where the two outer points are positive and the middle point is negative.
* Hence $\text{VC}(\mathcal{F}) = 2$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Axis-Aligned Rectangles in $\mathbb{R}^2$)</span></p>

$\mathcal{X} = \mathbb{R}^2$, $\mathcal{F} =$ interior of axis-aligned rectangles.

* There exists a point configuration of 4 points that can be shattered, so $\text{VC} \ge 4$.
* Note: we cannot shatter **all** sets of 4 points (some configurations cannot be shattered).
* No set of 5 points can be shattered: there has to exist one point that is not an extreme point in both axis-parallel directions.
* Hence $\text{VC}(\mathcal{F}) = 4$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Convex Polygons in $\mathbb{R}^2$)</span></p>

$\mathcal{X} = \mathbb{R}^2$, $\mathcal{F}_d =$ convex polygons with $d$ corners. Then $\text{VC} = 2d + 1$.

**Lower bound:** $2d + 1$ points on a circle can be shattered. If we have fewer than $d$ red points, we can draw a convex polygon around the blue points. If we have more than $d$ red points, we can draw a convex polygon around the red points instead.

**Upper bound:** More technical, prove that max. number of shattered points is achieved for points on a circle.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Sine Waves)</span></p>

$\mathcal{X} = \mathbb{R}$, $\mathcal{F} = \lbrace \text{sgn}(\sin(tx)) \mid t \in \mathbb{R} \rbrace$. Then $\text{VC} = \infty$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Linear Hyperplanes and Margin Classifiers)</span></p>

Relevant for SVMs:

* $\mathcal{X} = \mathbb{R}^d$, $\mathcal{F} =$ linear hyperplanes. Then $\text{VC}(\mathcal{F}) = d + 1$.
* $\mathcal{X} = \mathbb{R}^d$, $\rho > 0$, $\mathcal{F}\_\rho :=$ linear hyperplanes with margin at least $\rho$. If the data points are restricted to a ball of radius $R$, then

$$\text{VC}(\mathcal{F}) = \min\!\left\lbrace d,\; \frac{2R^2}{\rho^2} \right\rbrace + 1.$$

</div>

#### Sauer-Shelah Lemma

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Sauer-Shelah Lemma, Vapnik-Chervonenkis-Sauer-Shelah)</span></p>

Let $\mathcal{F}$ be a function class with finite VC dimension $d$. Then

$$\mathcal{N}(\mathcal{F}, n) \le \sum_{i=0}^{d} \binom{n}{i}$$

for all $n \in \mathbb{N}$. In particular, for all $n \ge d$ we have

$$\mathcal{N}(\mathcal{F}, n) \le \left(\frac{en}{d}\right)^d.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sauer-Shelah Lemma Consequences)</span></p>

* If a function class has a **finite** VC dimension, then the shattering coefficient only grows **polynomially**.
* If a function class has **infinite** VC dimension, then the shattering coefficient grows **exponentially**.
* It is impossible that the growth rate of the function class is "slightly smaller" than $2^n$. Either it is $2^n$, or much smaller, polynomial.

</div>

#### Generalization Bound with VC Dimension

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Generalization Bound with VC Dimension)</span></p>

Let $\mathcal{F}$ be a function class with VC dimension $d$. Then with probability at least $1 - \delta$, all functions $f \in \mathcal{F}$ satisfy

$$R(f) \le R_n(f) + 2\sqrt{\frac{d \log(2en/d) - \log(\delta)}{n}}.$$

**Consequence:** VC-dim finite $\implies$ consistency.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(VC Dimension Characterizes ERM Consistency)</span></p>

Empirical risk minimization is consistent with respect to $\mathcal{F}$ **if and only if** $\text{VC}(\mathcal{F})$ is finite.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sample Complexity from VC Dimension)</span></p>

How many samples do we need to draw to achieve error at most $\varepsilon$? Set $\varepsilon := 2\sqrt{\frac{d\log(2en/d) - \log(\delta)}{n}}$, solve for $n$ and ignore all constants. Result: we need of the order $n = d / \varepsilon^2$ many sample points.

</div>

### Rademacher Complexity

The shattering coefficient is a purely combinatorial object — it does not take into account what the actual probability distribution is. This seems suboptimal.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Rademacher Complexity)</span></p>

Fix a number $n$ of points. Let $\sigma_1, \dots, \sigma_n$ be i.i.d. tosses of a fair coin (result is $-1$ or $1$ with probability 0.5 each). The **Rademacher complexity** of a function class $\mathcal{F}$ with respect to $n$ is defined as

$$\text{Rad}_n(\mathcal{F}) := E \sup_{f \in \mathcal{F}} \frac{1}{n} \sum_{i=1}^{n} \sigma_i f(X_i).$$

The expectation is both over the draw of the random points $X_i$ and the random labels $\sigma_i$. It measures how well a function class can fit random labels.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Rademacher Generalization Bound)</span></p>

With probability at least $1 - \delta$, for all $f \in \mathcal{F}$,

$$R(f) \le R_n(f) + 2\,\text{Rad}_n(\mathcal{F}) + \sqrt{\frac{\log(1/\delta)}{2n}}.$$

</div>

Computing Rademacher complexities for function classes is in many cases much simpler than computing shattering coefficients or VC dimensions.

### Generalization Bounds: Conclusions

* Generalization bounds are a tool to answer the question whether a learning algorithm is **consistent**.
* Consistency refers to the **estimation error**, not the approximation error.
* Typically, generalization bounds have the following form: with probability at least $1 - \delta$, for all $f \in \mathcal{F}$,

$$R(f) \le R_n(f) + \text{capacity term} + \text{confidence term}.$$

  The capacity term measures the size of the function class. The confidence term deals with how certain we are about our statement.
* There are many different ways to measure the capacity of function classes; we just scratched the surface.
* Generalizations are **worst case bounds**: worst case over all possible probability distributions, and worst case over all learning algorithms that pick a function from $\mathcal{F}$.

## Controlling the Approximation Error

### Nested Function Classes

So far, we always fixed a function class $\mathcal{F}$ and investigated whether the estimation error in this class vanishes as we get to see more data. However, we need to take into account the **approximation error** as well.

The idea is to consider function classes that slowly **grow with $n$**: $\mathcal{F}_1 \subset \mathcal{F}_2 \subset \dots$

* If we have few data, the class is supposed to be small to avoid overfitting (generalization bound!).
* Eventually, when we see enough data, we can afford a larger function class without overfitting. The larger the class, the smaller our approximation error.

There are two major approaches:

* **Structural risk minimization:** explicit approach
* **Regularization:** implicit approach

### Structural Risk Minimization (SRM)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Structural Risk Minimization)</span></p>

Consider a nested sequence of function spaces: $\mathcal{F}_1 \subset \mathcal{F}_2 \subset \dots$ We now select an appropriate function class and a good function in this class simultaneously:

$$f_n := \arg\min_{m \in \mathbb{N},\, f \in \mathcal{F}_m} R_n(f) + \text{capacity term}(\mathcal{F}_m).$$

The capacity term is the one that comes from a generalization bound. If the nested function classes approximate the space of "all" functions, one can prove that such an approach can lead to universal consistency.

</div>

### Regularization

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Regularized Risk Minimization)</span></p>

Regularized risk minimization:

$$\min_f \; R_n(f) + \lambda \cdot \Omega(f)$$

where $\Omega$ punishes "complex" functions. Regularization is an **implicit** way of performing structural risk minimization.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Proving Consistency for Regularization)</span></p>

Proving consistency for regularization is technical but very elegant:

* Make sure that your overall space of functions $\mathcal{F}$ is **dense** in the space of continuous functions. Example: linear combinations of a universal kernel.
* Consider a sequence of regularization constants $\lambda_n$ with $\lambda_n \to 0$ as $n \to \infty$.
* Define function class $\mathcal{F}\_n := \lbrace f \in \mathcal{F} \;;\; \lambda_n \cdot \Omega(f) \le \text{const} \rbrace$.
* Choose $\lambda_n \to 0$ so slow that $\log \mathcal{N}(\mathcal{F}\_n, n) / n \to 0$.
  * On the one hand, this ensures that in the limit, the **estimation error** goes to 0 (we won't overfit).
  * On the other hand, if $\lambda_n \to 0$, then $\mathcal{F}\_n \to \mathcal{F}$ because $\lambda \Omega(f) < c \implies \Omega(f) \le c / \lambda_n \to \infty$. Hence the **approximation error** goes to 0 (we won't underfit).

Reference: Steinwart, "Support Vector Machines Are Universally Consistent," Journal of Complexity, 2002.

</div>

### Brief History

* The first proof that there exists a learning algorithm that is universally Bayes consistent was the **Theorem of Stone (1977)**, about the kNN classifier.
* The combinatorial tools and generalization bounds have essentially been developed in the early 1970s already (Vapnik, Chervonenkis, 1971, 1972, etc.) and refined in the years around 2000.
* The statistics community also proved many results, in particular rates of convergence. There the focus is more on regression rather than classification.
* By and large, the theory is well understood by now, the focus of attention moved to different areas of machine learning theory (for example, online learning, unsupervised learning, etc).

## Getting Back to Occam's Razor

### Examples Revisited

Consider two candidate functions that both fit the training data: a simple linear function (Guess 1) and a complex wiggly curve (Guess 2). The question is which of the two functions should be preferred. The intuitive answer is: unless we have a strong belief that the right curve is correct, we should prefer the left one due to "simplicity".

This principle is often called **"Occam's razor"** or **"principle of parsimony"**:

> *When we choose from a set of otherwise equivalent models, the simpler model should be preferred.*

### Occam's Razor vs. Learning Theory

The main message of learning theory was that we need to control the **size of the function class** $\mathcal{F}$. We had not at all talked about "simplicity" of functions! Is this a contradiction? Is Occam's razor wrong?

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(We Don't Need "Simplicity")</span></p>

Consider a function class that just contains 10 functions, all of which are very "complicated" (not "simple"). For the estimation error, this would be great — we would soon be able to detect which of the functions minimizes the ERM, with high probability. If the function class also happens to be able to describe the underlying phenomenon (low approximation error), this would be perfect. In this case, we do **not** need simple functions!

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Spaces of Simple Functions Tend to Be Small)</span></p>

Example: Polynomials in one variable, with a discrete set of coefficients:

$$f(x) = \sum_{k=1}^{d} a_k x^k \quad \text{with } a_k \in \lbrace -1, -0.99, -0.98, \dots, 0.98, 0.99, 1 \rbrace.$$

There are about 200 polynomials of degree 1, $200^2$ polynomials of degree 2, $200^d$ polynomials of degree $d$. The spaces get larger the more "parameters" we have.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Data Compression Perspective)</span></p>

Both points of view come together if we talk about **data compression**:

* A space with few functions can be represented with few bits (say, by a small lookup table).
* A space with "simple" functions can be represented with few bits as well (encode all the parameters).
* A space of "complex" functions cannot be compressed.

Intuitive conclusion:
* Spaces of simple functions are small, spaces of complex functions tend to be large.
* Learning theory tells us that we should prefer **small function spaces**.
* This often leads to spaces of simple functions.

This intuition can be made rigorous and formal: sample compression bounds in statistical learning theory, and the whole branch of learning based on the "Minimum description length principle" (comprehensive book in this area by Peter Grünwald).

</div>

**Bottom line:**
* The quantity that is important is not so much the simplicity of the functions but rather the **size of the function space**.
* But spaces of simple functions tend to be small and are good candidates for learning.
* Occam's razor slightly misses the point, but is a good first proxy. It is not always correct, but often...

## The No-Free-Lunch Theorem

### Intuition

Intuitively the no free lunch theorem (NFL) says that there does not exist a single best classifier that outperforms any other classifier on **all** learning problems. There exist many different versions to state this formally; below we describe the easiest one.

### NFL: Simple Version

**Setup:**

* The space of all input points is a finite set $\mathcal{X} = \lbrace x_1, \dots, x_m \rbrace$. The marginal distribution over these points is the uniform one (each value $x_i$ is equally likely).
* We consider binary classification, $\mathcal{Y} = \lbrace \pm 1 \rbrace$, and the labels are deterministic functions of the input.
* There exists some function $f : \mathcal{X} \to \mathcal{Y}$ that does not make any error.

Now consider a table: rows correspond to all possible true functions (there are $2^m$ such functions), columns correspond to all possible estimated functions. The entries $r_{ij}$ give the true error of function $f_i$ when the true function is $f_j$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Risk in Each Row is the Same)</span></p>

In each row of the table, each risk value occurs the same number of times.

*Proof.*
* $r_{ij} = 0$ exactly once (if $f_i = f_j$).
* $r_{ij} = 1/m$ exactly $m$ times.
* $r_{ij} = 2/m$ exactly $\binom{m}{2}$ times.
* ...

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Simple NFL)</span></p>

In the model introduced above: on average over all true functions $f$, the performance of all classifiers $\hat{f}$ is the same.

*Proof.* Obvious consequence of the previous proposition.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Simple NFL with Training Data)</span></p>

In the model introduced above: assume we are given a training set $(X_i, Y_i)\_{i=1,\dots,n}$. Then, on average over all test distributions, all classifiers that are consistent with the training set perform the same.

*Proof.* In the table above, eliminate all columns that are not consistent with the training data. Among the remaining ones, all distributions over test labels are possible. Then the result follows by a similar argument as above.

</div>

Note that much more general theorems exist, for example for the standard machine learning scenario where we draw data from joint distribution $P$ on $\mathcal{X} \times \mathcal{Y}$ and $\mathcal{X}$ is any space you want.

### Discussion

* "The best possible classifier for all data sets" does not exist.
* **Should we give up? Is machine learning meaningless?** No: the key is that in practice we do not see "all possible data sets". As soon as we make **assumptions** on the data sets, the NFL breaks down ("making assumptions" means to delete some columns from the above matrix, and then the proof breaks down).
* This shows once more how important it is to incorporate these assumptions to the machine learning algorithm $\leadsto$ **inductive bias!**
* A very specialized algorithm works well on few problems but very badly otherwise. A general purpose algorithm has average performance across all learning problems. By NFL, the integral over all these curves is exactly the same.

## A Glimpse on Modern Results in Learning Theory

### What Is Wrong with Classic Learning Theory?

We have just scratched the surface and considered the simplest results in the "classic regime" of supervised learning. However, this type of learning theory can **NOT** explain why deep networks work.

#### Function Classes Seem Too Large

**Huge function classes:** The function classes used in modern ML are huge: the language model GPT-3 has of the order $10^{10}$ many parameters! One would expect that the capacity is huge.

**Empirical observation:** Deep neural networks can fit random labels. This means that their shattering coefficient is always maximal (Bengio et al, Understanding deep learning requires rethinking generalization, 2017 and 2021).

**Classic SLT breaks down:** Huge function classes with huge shattering coefficients leads to useless generalization bounds. This does not mean that the bounds are not correct; but the generalization bounds cannot explain why the test error becomes small (uniform convergence does not hold over the huge function class).

#### Common Practice of Overfitting Seems Wrong

**DNNs are trained to overfit!** DNNs are trained to highly overfit the data — but still they generalize! This is all the more surprising, given that the large function classes contain many, many functions that have close to 0 training error. Many of them would not generalize to new, unseen data.

#### Why Don't We Get Stuck in Local Optima?

**Computational surprise:** DNNs induce highly non-convex optimization problems in very high-dimensional spaces. Yet it is very surprising to see that the optimisation algorithms manage to find good local (global?) optima.

#### Need New Tools

All this does not mean that the standard learning theory is wrong. Its results simply do not shed any insights on deep learning, or we need to apply it more carefully. We need other tools to explain DNNs.

In the following we look at two sides of the story:

1. Why do DNNs work at all?
2. Why we really need DNNs (or related models) to solve today's machine learning problems.

### Why DNNs Might Work: Overfitting and the Double Descent Curve

**Literature:**

* Mikhail Belkin, Daniel Hsu, Siyuan Ma, Soumik Mandal: *Reconciling modern machine-learning practice and the classical bias-variance trade-off.* PNAS, 2019.
* Mikhail Belkin: *Fit without fear, remarkable mathematical phenomena of deep learning through the prism of interpolation.* Acta Numerica, 2021.
* Peter Bartlett, Andrea Montanari, Sasha Rakhlin: *Deep learning, a statistical viewpoint.* Arxiv, 2021.

#### Double Descent Curve

Radically new insight: need to distinguish between the **underparameterized regime** (less parameters than data points, cannot interpolate, "traditional" behavior) and the **overparameterized / interpolation regime**. Here the test risk sometimes decreases dramatically again and sometimes even gets lower than in the traditional regime.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Double Descent Curve)</span></p>

The classical U-shaped risk curve arises from the bias-variance trade-off: as model capacity increases, training risk decreases while test risk first decreases then increases.

The **double-descent risk curve** incorporates the U-shaped risk curve (the "classical" regime) together with the observed behavior from using high-capacity function classes (the "modern" interpolating regime), separated by the **interpolation threshold**. The predictors to the right of the interpolation threshold have zero training risk.

In the original paper by Belkin et al (2019), such behavior has been established for a range of algorithms including neural networks and random forests.

</div>

#### Explaining Double Descent: Implicit Regularization

Consider the overparameterized regime. There might exist many functions that interpolate the data perfectly. Some of them would generalize well, some of them would be awful. Why is it the case that in practice the functions generalize so well?

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Bartlett, Montanari, Rakhlin, informal)</span></p>

Consider a set of $n$ data points in $\mathbb{R}^d$ with real-valued labels. Assume the overparameterized regime: $d > n$. Choose the $L_2$ loss function and consider the linear regression problem. (Then there are many solutions of the interpolation problem, they fill a space of dim $d - n$.) Then the solution that is achieved by gradient descent converges to one specific solution in this space of all solutions, namely the **minimum norm solution**.

A similar result holds for classification with the logistic loss, where gradient descent converges to the maximum $\ell_2$-margin separator.

Similar results for specific, simple neural networks (e.g. fully connected, linear activation).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Implicit Regularization)</span></p>

**Implicit regularization** is the phenomenon where, even though we do not actively model the fact that we would like to find a "simple" (nice, smooth, sparse, ...) solution, this happens implicitly through the choice of the optimization algorithm.

Bartlett et al: *"In overparameterized problems that admit multiple minimizers of the empirical objective, the choice of the optimization method and the choice of the parameterization both play crucial roles in selecting a minimizer with certain properties."*

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Consistency of Minimum Norm Interpolators)</span></p>

One can prove that **under certain conditions, the minimum norm interpolators are consistent** (they converge to the optimal solution). For example, the following decomposition has been used:

We write the prediction rule in the form

$$f = f_0 + \Delta$$

where $f_0$ is benign and takes care of generalization, and $\Delta$ is a very spiky function that takes care of the interpolation, but otherwise "does not hurt".

This is particularly helpful in high-dimensional settings where the volume of the "spikes" is so low that it does not hurt generalization.

(Bartlett, Montanari, Rakhlin, 2021)

</div>

#### Why Is There No Contradiction to Classic SLT?

**Literature:** On Uniform Convergence and Low-Norm Interpolation Learning, Lijia Zhou, Danica J. Sutherland, Nathan Srebro, NeurIPS 2020.

The current story goes as follows:

* Function classes in deep learning are so large that classic generalization bounds are not informative.
* Fine, because generalization bounds themselves provide sufficient conditions for uniform convergence. So violating the sufficient conditions might not yet mean that convergence cannot take place.
* However, there was Vapnik's theorem: uniform convergence is necessary and sufficient for consistency.
* But the function classes in deep learning are still so large that uniform convergence seems difficult to obtain...

So, where is the catch?

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Catch)</span></p>

* Even though the function class that in principle can be used by a DNN is huge, in the end we select our solution from a tiny subset of this class. For example, we know that the selected functions have a small norm. And the set of small norm functions might again be small enough such that uniform convergence might hold.
* As it turns out, this still does not easily work out: Srebro et al (2020) prove that the set of functions of small norm is still too large for uniform convergence: *"uniformly bounding the difference between empirical and population errors cannot show any learning in the norm ball, and cannot show consistency for any set, even one depending on the exact algorithm and distribution."*
* However, one can get uniform convergence if one further restricts the function space to those functions that have a small norm **AND** are interpolating functions.

</div>

### Why We Might Really Need Large Models: Smooth Interpolation

**Literature:** A universal law of robustness via isoperimetry. Sebastien Bubeck, Mark Selke, NeurIPS 2021.

#### Smooth Interpolation: Intuition

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Robustness via Lipschitz Constant)</span></p>

To achieve a robust solution, we would like to have a classifier that is "not too wiggly". This is measured in terms of the **Lipschitz constant**:

$$L(f) := \max_{x,y} \frac{\lvert f(x) - f(y) \rvert}{\lVert x - y \rVert}.$$

The higher $L(f)$, the more wiggly is the function, the less robust. Ideally, we would like to achieve a constant Lipschitz constant (not depending on $n$ or $d$ for example).

</div>

**Interpolation:** To solve a system of $n$ equations, you typically need only $n$ unknowns. However, the solution can have very high Lipschitz constants, so it is not robust.

**Is there a way to achieve robust interpolation?** (An interpolating classifier that has low Lipschitz constant?) If we use a larger function class, we might have more solutions to our interpolation problem. Is there one that has a low Lipschitz constant?

**Empirically** it has been observed that larger networks help tremendously for robustness.

#### Result by Bubeck and Selke

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Bubeck and Selke, 2021, informal)</span></p>

Consider any function class $\mathcal{F}$ that is smoothly parameterized by $p$ parameters, and a $d$-dimensional data set that satisfies mild regularity assumptions (comes from a somewhat nice distribution). Then any (!) function in $\mathcal{F}$ that fits the training data below the noise model (e.g. it interpolates) must (!) have Lipschitz constant larger than $\sqrt{nd/p}$.

The other way round, if we want to achieve Lipschitz constant $L = 1$, then we need to choose the number of parameters $p$ at least as large as $nd$.

</div>

<details class="accordion" markdown="1">
<summary>Proof sketch (Bubeck and Selke)</summary>

**Intuition: constructing an example.**

* Consider the uniform distribution on the unit sphere in $\mathbb{R}^d$.
* Randomly sample $n$ points ($n$ moderate with respect to $d$).
* Then with probability at least $1 - \exp(\Omega(d))$, the distance between any two sample points is at least 1 (concentration argument due to high dimension).
* Now choose any labels for the points that you like.
* Then we can construct a function $f$ on the whole sphere that takes the values of the labels: let $g$ be something like a "bump" (some RBF function), and set $f(x) = \sum_{i=1}^{n} g(\lVert x - x_i \rVert) y_i$.
* This function interpolates, needs $d \cdot n$ many parameters (the centers of the $n$ bumps) and has Lipschitz constant 1.

**Proof strategy for the main theorem (finite $\mathcal{F}$):**

* Consider $\mathcal{F}$ with $N$ functions with Lipschitz constant $L$, fix some $f \in \mathcal{F}$.
* Draw a random data set of $n$ points from the underlying distribution (with random labels).
* Due to concentration (the "niceness assumption" on the high-dimensional distribution): the probability that the fixed $f$ can fit the labels is bounded by $\exp(-nd / L^2)$.
* By a union bound: the probability that there exists some $f \in \mathcal{F}$ that fits the sample is at most

$$N \exp(-nd/L^2) = \exp(\log N - nd/L^2).$$

  This bound only gets large if $L$ is on the order $\sqrt{nd / \log(N)}$. (The other way round: any function that interpolates the training data must have $L$ of the order $\sqrt{nd / \log(N)}$ or larger.)
* A standard argument involving $\varepsilon$-nets extends the argument to infinite function spaces. As it turns out, the $\log N$ is then replaced by $p$.

The results can also be extended towards generalization bounds, see the paper for details.

</details>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Application to ImageNet)</span></p>

Applied to ImageNet, the authors estimate that for a robust estimator, one would need of the order $10^{11}$ many parameters (current models use about $10^9$ parameters).

If this conjecture is correct, it is only a matter of time until we achieve robust classification on ImageNet. We would not need new tools, just larger models.

</div>

### Summary of "Modern" Learning Theory

To explain the generalization capability of DNNs, we use the following chain of arguments:

* Consider the highly overparameterized regime, high-dimensional setting.
* By the geometry of the loss landscape, we can find global optima (interpolating solutions) easily through SGD.
* Under certain conditions, through implicit regularization, this interpolating solution has nice properties (e.g. it is the minimum norm solution).
* For such nice solutions, one can prove for certain models that we achieve generalization (e.g. though the "simple-plus-spiky" decomposition).

This explains why DNNs might work at all. It does not yet explain whether we actually need DNNs (or other huge models) to solve modern ML problems. Could it be the case that we just haven't found a simpler approach?

Perhaps, the answer is no. The arguments on smooth interpolations show that if we want to achieve robust solutions, then we need large models (not necessarily DNNs, but models with many parameters).

Many of the results above (and many others as well) have only been established in special cases, under special assumptions, etc. But they start painting a picture. Much more work required to pin it down...

