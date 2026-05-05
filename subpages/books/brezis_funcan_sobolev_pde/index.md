---
layout: default
title: "Functional Analysis, Sobolev Spaces and PDEs — Brezis"
date: 2026-05-05
excerpt: Notes on Haim Brezis's textbook covering the Hahn–Banach theorems, weak topologies, Sobolev spaces, and elliptic boundary value problems.
tags:
  - functional-analysis
  - sobolev-spaces
  - partial-differential-equations
  - mathematics
---

# Functional Analysis, Sobolev Spaces and PDEs — Brezis

**Table of Contents**
- TOC
{:toc}

## Chapter 1: The Hahn–Banach Theorems. Introduction to the Theory of Conjugate Convex Functions

The Hahn–Banach theorem is one of the cornerstones of linear functional analysis. It exists in two complementary guises:

* an **analytic form**, asserting that linear functionals defined on a subspace can be extended to the whole space while remaining dominated by a sublinear functional, and
* **geometric forms**, asserting that disjoint convex sets can be separated by hyperplanes.

The two viewpoints are equivalent — separating hyperplanes are the level sets of extended linear functionals — but each is the natural tool for a different family of problems. The chapter culminates in a quick introduction to *conjugate convex functions*, which is the duality theory underlying convex optimization, and the **Fenchel–Moreau** and **Fenchel–Rockafellar** theorems.

### 1.1 The Analytic Form of the Hahn–Banach Theorem: Extension of Linear Functionals

Let $E$ be a vector space over $\mathbb{R}$. We recall that a *functional* is a function defined on $E$, or on some subspace of $E$, *with values in* $\mathbb{R}$. The main result of this section concerns the extension of a linear functional defined on a linear subspace of $E$ by a linear functional defined on all of $E$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1.1</span><span class="math-callout__name">(Helly, Hahn–Banach analytic form)</span></p>

Let $p : E \to \mathbb{R}$ be a function satisfying

$$
\begin{align}
p(\lambda x) &= \lambda p(x) && \forall x \in E,\ \forall \lambda > 0, \tag{1} \\
p(x + y) &\le p(x) + p(y) && \forall x, y \in E. \tag{2}
\end{align}
$$

Let $G \subset E$ be a linear subspace and let $g : G \to \mathbb{R}$ be a linear functional such that

$$
g(x) \le p(x) \quad \forall x \in G. \tag{3}
$$

Then there exists a linear functional $f$ defined on all of $E$ that **extends** $g$, i.e., $g(x) = f(x)\ \forall x \in G$, and such that

$$
f(x) \le p(x) \quad \forall x \in E. \tag{4}
$$

</div>

A function $p$ satisfying $(1)$ and $(2)$ is sometimes called a *Minkowski functional*. The proof depends on Zorn's lemma, a celebrated and very useful property of ordered sets. Before stating it we clarify some notions.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Order-Theoretic Vocabulary)</span></p>

Let $P$ be a set with a (partial) order relation $\le$.

* A subset $Q \subset P$ is **totally ordered** if for any pair $(a, b) \in Q$ either $a \le b$ or $b \le a$ (or both).
* For a subset $Q \subset P$, an element $c \in P$ is an **upper bound** for $Q$ if $a \le c$ for every $a \in Q$.
* An element $m \in P$ is a **maximal element** of $P$ if there is no element $x \in P$ such that $m \le x$, except for $x = m$. A maximal element of $P$ need *not* be an upper bound for $P$.
* $P$ is **inductive** if every totally ordered subset $Q \subset P$ has an upper bound.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 1.1</span><span class="math-callout__name">(Zorn)</span></p>

Every nonempty ordered set that is inductive has a maximal element.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(On Zorn's lemma)</span></p>

Zorn's lemma follows from the axiom of choice and has many important applications in analysis. It is a *basic tool* in proving some *seemingly innocent existence statements* such as

* "every vector space has a basis" (see Exercise 1.5);
* "on any vector space there are nontrivial linear functionals."

Most analysts do not know how to prove Zorn's lemma; but it is essential for an analyst to understand the *statement* of Zorn's lemma and to be able to use it properly.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 1.1</summary>

Consider the set

$$
P = \left\lbrace h : D(h) \subset E \to \mathbb{R} \;\middle|\;
\begin{aligned}
&D(h) \text{ is a linear subspace of } E, \\
&h \text{ is linear},\ G \subset D(h), \\
&h \text{ extends } g,\text{ and } h(x) \le p(x)\ \forall x \in D(h)
\end{aligned}
\right\rbrace.
$$

On $P$ define the order

$$
(h_1 \le h_2) \iff (D(h_1) \subset D(h_2) \text{ and } h_2 \text{ extends } h_1).
$$

It is clear that $P$ is nonempty, since $g \in P$. We claim that $P$ is inductive: let $Q \subset P$ be a totally ordered subset, write $Q = (h_i)_{i \in I}$, and set

$$
D(h) = \bigcup_{i \in I} D(h_i),\qquad h(x) = h_i(x) \text{ if } x \in D(h_i) \text{ for some } i.
$$

It is easy to see that the definition of $h$ makes sense, that $h \in P$, and that $h$ is an upper bound for $Q$. Apply Zorn's lemma to obtain a maximal element $f \in P$. We claim that $D(f) = E$, which completes the proof.

Suppose, by contradiction, that $D(f) \neq E$. Pick $x_0 \notin D(f)$, set $D(h) = D(f) + \mathbb{R}x_0$, and for every $x \in D(f)$ set $h(x + tx_0) = f(x) + t\alpha\ (t \in \mathbb{R})$, where $\alpha \in \mathbb{R}$ will be chosen so that $h \in P$. We must ensure that

$$
f(x) + t\alpha \le p(x + tx_0) \quad \forall x \in D(f),\ \forall t \in \mathbb{R}.
$$

In view of $(1)$ it suffices to check that

$$
\begin{cases}
f(x) + \alpha \le p(x + x_0) &\forall x \in D(f), \\
f(x) - \alpha \le p(x - x_0) &\forall x \in D(f).
\end{cases}
$$

In other words, we must find some $\alpha$ satisfying

$$
\sup_{y \in D(f)} \lbrace f(y) - p(y - x_0) \rbrace \le \alpha \le \inf_{x \in D(f)} \lbrace p(x + x_0) - f(x) \rbrace.
$$

Such an $\alpha$ exists, since for all $x, y \in D(f)$,

$$
f(y) - p(y - x_0) \le p(x + x_0) - f(x);
$$

indeed, this follows from $(2)$:

$$
f(x) + f(y) \le p(x + y) \le p(x + x_0) + p(y - x_0).
$$

We conclude that $f \le h$; but this is impossible, since $f$ is maximal and $h \neq f$. $\square$

</details>
</div>

We now describe simple applications of Theorem 1.1 to the case in which $E$ is a *normed vector space* (n.v.s.) with norm $\|\cdot\|$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Dual space, scalar product for the duality)</span></p>

We denote by $E^\star$ the **dual space** of $E$, that is, the space of all *continuous* linear functionals on $E$; the (dual) **norm** on $E^\star$ is defined by

$$
\|f\|_{E^\star} = \sup_{\substack{\|x\| \le 1 \\ x \in E}} |f(x)| = \sup_{\substack{\|x\| \le 1 \\ x \in E}} f(x). \tag{5}
$$

When there is no confusion we write $\|f\|$ instead of $\|f\|_{E^\star}$. Given $f \in E^\star$ and $x \in E$ we often write $\langle f, x \rangle$ instead of $f(x)$; we say $\langle\,,\,\rangle$ is the **scalar product for the duality** $E^\star, E$. It is well known that $E^\star$ is a Banach space, i.e., $E^\star$ is complete (even if $E$ is not); this follows from the fact that $\mathbb{R}$ is complete.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 1.2</span><span class="math-callout__name">(Norm-preserving extension)</span></p>

Let $G \subset E$ be a linear subspace. If $g : G \to \mathbb{R}$ is a continuous linear functional, then there exists $f \in E^\star$ that extends $g$ and such that

$$
\|f\|_{E^\star} = \sup_{\substack{x \in G \\ \|x\| \le 1}} |g(x)| = \|g\|_{G^\star}.
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Apply Theorem 1.1 with $p(x) = \|g\|_{G^\star} \|x\|$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 1.3</span><span class="math-callout__name">(Functional realizing the norm)</span></p>

For every $x_0 \in E$ there exists $f_0 \in E^\star$ such that

$$
\|f_0\| = \|x_0\| \quad \text{and} \quad \langle f_0, x_0 \rangle = \|x_0\|^2.
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Apply Corollary 1.2 with $G = \mathbb{R}x_0$ and $g(tx_0) = t\|x_0\|^2$, so that $\|g\|_{G^\star} = \|x_0\|$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Duality map)</span></p>

The element $f_0$ given by Corollary 1.3 is in general *not unique* (try to construct an example or see Exercise 1.2). However, if $E^\star$ is **strictly convex** — for instance if $E$ is a Hilbert space (Chapter 5) or if $E = L^p(\Omega)$ with $1 < p < \infty$ (Chapter 4) — then $f_0$ is unique. In general we set, for every $x_0 \in E$,

$$
F(x_0) = \lbrace f_0 \in E^\star\,;\ \|f_0\| = \|x_0\|\text{ and }\langle f_0, x_0 \rangle = \|x_0\|^2 \rbrace.
$$

The (multivalued) map $x_0 \mapsto F(x_0)$ is called the **duality map** from $E$ into $E^\star$. (Recall: a normed space is *strictly convex* if $\|tx + (1-t)y\| < 1$ for all $t \in (0,1)$ and all $x, y$ with $\|x\| = \|y\| = 1$ and $x \neq y$.)

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 1.4</span><span class="math-callout__name">(Dual representation of the norm)</span></p>

For every $x \in E$,

$$
\|x\| = \sup_{\substack{f \in E^\star \\ \|f\| \le 1}} |\langle f, x \rangle| = \max_{\substack{f \in E^\star \\ \|f\| \le 1}} |\langle f, x \rangle|. \tag{6}
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Assume $x \neq 0$. The inequality $\sup_{\|f\| \le 1} \lvert \langle f, x \rangle \rvert \le \|x\|$ is clear. Conversely, by Corollary 1.3 there exists $f_0 \in E^\star$ with $\|f_0\| = \|x\|$ and $\langle f_0, x \rangle = \|x\|^2$. Set $f_1 = f_0 / \|x\|$, so that $\|f_1\| = 1$ and $\langle f_1, x \rangle = \|x\|$. Hence the $\sup$ is achieved (which is why we may write $\max$). $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Definition vs. statement; reflexivity)</span></p>

Formula $(5)$ — which is a *definition* — should not be confused with formula $(6)$, which is a *statement*. In general, the "$\sup$" in $(5)$ is *not achieved*; see Exercise 1.3. However, the "$\sup$" in $(5)$ *is* achieved if $E$ is a reflexive Banach space (see Chapter 3); a deep result due to **R. C. James** asserts the converse: if $E$ is a Banach space such that for every $f \in E^\star$ the sup in $(5)$ is achieved, then $E$ is reflexive.

</div>

### 1.2 The Geometric Forms of the Hahn–Banach Theorem: Separation of Convex Sets

We turn to the second face of Hahn–Banach: separating disjoint convex sets by a hyperplane. In what follows $E$ denotes an n.v.s.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Affine hyperplane)</span></p>

An **affine hyperplane** is a subset $H$ of $E$ of the form

$$
H = \lbrace x \in E\,;\ f(x) = \alpha \rbrace,
$$

where $f$ is a linear functional that does not vanish identically and $\alpha \in \mathbb{R}$ is a given constant. We write $H = [f = \alpha]$ and say that $f = \alpha$ is the **equation** of $H$.

We do *not* assume that $f$ is continuous: in every infinite-dimensional normed space there exist discontinuous linear functionals (see Exercise 1.5).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1.5</span><span class="math-callout__name">(Closed hyperplanes ↔ continuous functionals)</span></p>

The hyperplane $H = [f = \alpha]$ is closed if and only if $f$ is continuous.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

If $f$ is continuous, $H$ is closed. Conversely, assume $H$ is closed. The complement $H^c$ is open and nonempty (since $f$ does not vanish identically). Pick $x_0 \in H^c$ with, say, $f(x_0) < \alpha$. Choose $r > 0$ such that $B(x_0, r) \subset H^c$, where $B(x_0, r) = \lbrace x \in E\,;\ \|x - x_0\| < r \rbrace$.

We claim that

$$
f(x) < \alpha \quad \forall x \in B(x_0, r). \tag{7}
$$

Otherwise, take $x_1 \in B(x_0, r)$ with $f(x_1) > \alpha$; the segment $\lbrace x_t = (1-t)x_0 + tx_1\,;\ t \in [0,1] \rbrace$ lies in $B(x_0, r)$, hence $f(x_t) \neq \alpha$ for all $t \in [0,1]$. But $f(x_t) = \alpha$ for $t = (\alpha - f(x_0))/(f(x_1) - f(x_0)) \in (0,1)$, contradiction. So $(7)$ holds.

From $(7)$, $f(x_0 + rz) < \alpha$ for all $z \in B(0,1)$, hence $\|f\| \le \tfrac{1}{r}(\alpha - f(x_0))$. $\square$

</details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Separation of sets, convex set)</span></p>

Let $A$ and $B$ be two subsets of $E$. The hyperplane $H = [f = \alpha]$ **separates** $A$ and $B$ if

$$
\boxed{\;f(x) \le \alpha\ \forall x \in A \quad \text{and} \quad f(x) \ge \alpha\ \forall x \in B.\;}
$$

It **strictly separates** them if there exists $\varepsilon > 0$ such that

$$
\boxed{\;f(x) \le \alpha - \varepsilon\ \forall x \in A \quad \text{and} \quad f(x) \ge \alpha + \varepsilon\ \forall x \in B.\;}
$$

A subset $A \subset E$ is **convex** if

$$
\boxed{\;tx + (1-t)y \in A \quad \forall x, y \in A,\ \forall t \in [0,1].\;}
$$

</div>

Geometrically, separation means $A$ lies in one of the half-spaces determined by $H$, and $B$ in the other.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1.6</span><span class="math-callout__name">(Hahn–Banach, first geometric form)</span></p>

Let $A \subset E$ and $B \subset E$ be two nonempty convex subsets such that $A \cap B = \emptyset$. Assume that one of them is **open**. Then there exists a closed hyperplane that separates $A$ and $B$.

</div>

The proof of Theorem 1.6 relies on two lemmas, both interesting in their own right.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 1.2</span><span class="math-callout__name">(Minkowski gauge of an open convex set)</span></p>

Let $C \subset E$ be an open convex set with $0 \in C$. For every $x \in E$ set

$$
p(x) = \inf\lbrace \alpha > 0\,;\ \alpha^{-1} x \in C \rbrace. \tag{8}
$$

($p$ is called the **gauge** of $C$, or the **Minkowski functional** of $C$.) Then $p$ satisfies $(1)$, $(2)$, and the following properties:

$$
\begin{align}
\text{there is a constant } M \text{ such that } 0 \le p(x) \le M\|x\|\ \forall x \in E, \tag{9} \\
C = \lbrace x \in E\,;\ p(x) < 1 \rbrace. \tag{10}
\end{align}
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Lemma 1.2</summary>

$(1)$ is obvious.

*Proof of $(9)$.* Let $r > 0$ with $B(0, r) \subset C$; then $p(x) \le \tfrac{1}{r}\|x\|$.

*Proof of $(10)$.* If $x \in C$, openness gives $(1+\varepsilon)x \in C$ for small $\varepsilon > 0$, hence $p(x) \le 1/(1+\varepsilon) < 1$. Conversely, if $p(x) < 1$, take $\alpha \in (0, 1)$ with $\alpha^{-1} x \in C$, and write $x = \alpha(\alpha^{-1}x) + (1-\alpha)\,0 \in C$ by convexity.

*Proof of $(2)$.* Let $x, y \in E$ and $\varepsilon > 0$. Using $(1)$ and $(10)$, $\frac{x}{p(x)+\varepsilon}, \frac{y}{p(y)+\varepsilon} \in C$. Therefore

$$
\frac{tx}{p(x)+\varepsilon} + \frac{(1-t)y}{p(y)+\varepsilon} \in C \quad \forall t \in [0,1].
$$

Choosing $t = \frac{p(x)+\varepsilon}{p(x)+p(y)+2\varepsilon}$ gives $\frac{x+y}{p(x)+p(y)+2\varepsilon} \in C$, hence $p(x+y) < p(x) + p(y) + 2\varepsilon$ for all $\varepsilon > 0$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 1.3</span><span class="math-callout__name">(Separating a point from an open convex set)</span></p>

Let $C \subset E$ be a nonempty open convex set and let $x_0 \in E$ with $x_0 \notin C$. Then there exists $f \in E^\star$ such that $f(x) < f(x_0)$ for every $x \in C$. In particular, the hyperplane $[f = f(x_0)]$ separates $\lbrace x_0 \rbrace$ and $C$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Lemma 1.3</summary>

After translation we may assume $0 \in C$. Introduce the gauge $p$ of $C$ (Lemma 1.2). Consider the linear subspace $G = \mathbb{R}x_0$ and the linear functional $g : G \to \mathbb{R}$ defined by

$$
g(tx_0) = t,\quad t \in \mathbb{R}.
$$

It is clear that $g(x) \le p(x)\ \forall x \in G$ (consider the cases $t > 0$ and $t \le 0$ separately, using $x_0 \notin C$, so $p(x_0) \ge 1$). By Theorem 1.1 there exists a linear functional $f$ on $E$ that extends $g$ and satisfies $f(x) \le p(x)\ \forall x \in E$. In particular $f(x_0) = 1$, and $f$ is continuous by $(9)$. From $(10)$, $f(x) < 1$ for every $x \in C$. $\square$

</details>
</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 1.6</summary>

Set $C = A - B$. Then $C$ is convex (check), $C$ is open (since $C = \bigcup_{y \in B}(A - y)$), and $0 \notin C$ (because $A \cap B = \emptyset$). By Lemma 1.3 there is $f \in E^\star$ with $f(z) < 0$ for every $z \in C$, i.e.,

$$
f(x) < f(y) \quad \forall x \in A,\ \forall y \in B.
$$

Fix $\alpha$ with $\sup_{x \in A} f(x) \le \alpha \le \inf_{y \in B} f(y)$. The hyperplane $[f = \alpha]$ separates $A$ and $B$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1.7</span><span class="math-callout__name">(Hahn–Banach, second geometric form)</span></p>

Let $A \subset E$ and $B \subset E$ be two nonempty convex subsets such that $A \cap B = \emptyset$. Assume that $A$ is **closed** and $B$ is **compact**. Then there exists a closed hyperplane that **strictly** separates $A$ and $B$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 1.7</summary>

Set $C = A - B$. Then $C$ is convex, closed (check!), and $0 \notin C$. Hence there is $r > 0$ with $B(0, r) \cap C = \emptyset$. By Theorem 1.6 applied to $B(0, r)$ and $C$, there is a closed hyperplane separating them, i.e., some $f \in E^\star,\ f \not\equiv 0$, such that

$$
f(x - y) \le f(rz) \quad \forall x \in A,\ \forall y \in B,\ \forall z \in B(0,1).
$$

Hence $f(x - y) \le -r\|f\|\ \forall x \in A, \forall y \in B$. With $\varepsilon = \tfrac{1}{2}r\|f\| > 0$,

$$
f(x) + \varepsilon \le f(y) - \varepsilon \quad \forall x \in A,\ \forall y \in B.
$$

Choose $\alpha$ with $\sup_{x \in A} f(x) + \varepsilon \le \alpha \le \inf_{y \in B} f(y) - \varepsilon$. Then $[f = \alpha]$ strictly separates $A$ and $B$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(How sharp are the hypotheses?)</span></p>

It is *not* in general possible to separate two disjoint convex sets without further hypotheses, even when both are closed. One can construct an example in which $A$ and $B$ are *both closed* and disjoint yet *cannot* be separated by a closed hyperplane (Exercise 1.14).

However, in *finite dimensions* one can always separate any two nonempty disjoint convex sets $A$ and $B$ — no openness, closedness, or compactness needed (Exercise 1.9). The pathology is genuinely an infinite-dimensional phenomenon.

</div>

We conclude this section with a very useful corollary, the workhorse for proving density of subspaces.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 1.8</span><span class="math-callout__name">(Functionals vanishing on a non-dense subspace)</span></p>

Let $F \subset E$ be a linear subspace such that $\overline{F} \neq E$. Then there exists some $f \in E^\star,\ f \not\equiv 0$, such that

$$
\langle f, x \rangle = 0 \quad \forall x \in F.
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $x_0 \in E$ with $x_0 \notin \overline{F}$. Apply Theorem 1.7 with $A = \overline{F}$ and $B = \lbrace x_0 \rbrace$ to find a closed hyperplane $[f = \alpha]$ strictly separating them: $\langle f, x \rangle < \alpha < \langle f, x_0 \rangle$ for all $x \in F$. Since $F$ is a linear space, $\langle f, x \rangle = 0$ for every $x \in F$ (otherwise $\lambda \langle f, x \rangle$ exceeds $\alpha$ for some $\lambda \in \mathbb{R}$). $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(How Corollary 1.8 is used in practice)</span></p>

Corollary 1.8 is used very often to prove that a linear subspace $F \subset E$ is **dense**. It suffices to show that *every continuous linear functional on $E$ that vanishes on $F$ must vanish everywhere on $E$.* Concretely:

* If you want to prove $\overline{F} = E$, take any $f \in E^\star$ with $\langle f, x \rangle = 0$ for all $x \in F$;
* show that necessarily $f \equiv 0$ on $E$;
* by contrapositive of Corollary 1.8, this forces $\overline{F} = E$.

This pattern recurs throughout PDE theory (e.g., density of smooth functions in Sobolev spaces) and harmonic analysis (e.g., completeness of Fourier systems).

</div>

### 1.3 The Bidual $E^{\star\star}$. Orthogonality Relations

Let $E$ be an n.v.s. and $E^\star$ its dual with norm $\|f\|_{E^\star} = \sup_{\|x\| \le 1,\, x \in E} \lvert \langle f, x\rangle \rvert$. The **bidual** $E^{\star\star}$ is the dual of $E^\star$ with norm

$$
\|\xi\|_{E^{\star\star}} = \sup_{\substack{f \in E^\star \\ \|f\| \le 1}} |\langle \xi, f \rangle| \quad (\xi \in E^{\star\star}).
$$

There is a **canonical injection** $J : E \to E^{\star\star}$ defined as follows: given $x \in E$, the map $f \mapsto \langle f, x \rangle$ is a continuous linear functional on $E^\star$, hence an element of $E^{\star\star}$, denoted $Jx$. We have

$$
\langle Jx, f \rangle_{E^{\star\star}, E^\star} = \langle f, x \rangle_{E^\star, E} \quad \forall x \in E,\ \forall f \in E^\star.
$$

(Caution: $J$ should not be confused with the duality map $F : E \to E^\star$ defined in §1.1.)

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Canonical injection is an isometry)</span></p>

The map $J$ is linear, and it is an **isometry**, i.e., $\|Jx\|_{E^{\star\star}} = \|x\|_E$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

$$
\|Jx\|_{E^{\star\star}} = \sup_{\substack{f \in E^\star \\ \|f\| \le 1}} |\langle Jx, f \rangle| = \sup_{\substack{f \in E^\star \\ \|f\| \le 1}} |\langle f, x \rangle| = \|x\|
$$

by Corollary 1.4. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reflexivity)</span></p>

It may happen that $J$ is *not surjective* from $E$ onto $E^{\star\star}$ (see Chapters 3 and 4 for $L^1, L^\infty$, $\ell^1, \ell^\infty$, $C(K)$, …). However, it is convenient to *identify* $E$ with a subspace of $E^{\star\star}$ via $J$. If $J$ turns out to be surjective, $E$ is called **reflexive**, and $E^{\star\star}$ is identified with $E$ (Chapter 3).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Orthogonal subspaces)</span></p>

* If $M \subset E$ is a linear subspace,
  $$\boxed{\;M^\perp = \lbrace f \in E^\star\,;\ \langle f, x \rangle = 0\ \forall x \in M \rbrace.\;}$$
* If $N \subset E^\star$ is a linear subspace,
  $$\boxed{\;N^\perp = \lbrace x \in E\,;\ \langle f, x \rangle = 0\ \forall f \in N \rbrace.\;}$$

By definition, $N^\perp$ is a subset of **$E$ rather than $E^{\star\star}$**. Both $M^\perp$ and $N^\perp$ are *closed* linear subspaces (of $E^\star$ and $E$ respectively). We say $M^\perp$ (resp. $N^\perp$) is **orthogonal** to $M$ (resp. $N$).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1.9</span><span class="math-callout__name">(Bipolar / orthogonality relations)</span></p>

Let $M \subset E$ be a linear subspace. Then

$$
\boxed{\;(M^\perp)^\perp = \overline{M}.\;}
$$

Let $N \subset E^\star$ be a linear subspace. Then

$$
(N^\perp)^\perp \supset \overline{N}.
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

It is clear that $M \subset (M^\perp)^\perp$, and since $(M^\perp)^\perp$ is closed, $\overline{M} \subset (M^\perp)^\perp$.

Conversely, suppose $x_0 \in (M^\perp)^\perp$ with $x_0 \notin \overline{M}$. By Theorem 1.7 there is a closed hyperplane that strictly separates $\lbrace x_0 \rbrace$ from $\overline{M}$: some $f \in E^\star$ and $\alpha \in \mathbb{R}$ with

$$
\langle f, x \rangle < \alpha < \langle f, x_0 \rangle \quad \forall x \in M.
$$

Since $M$ is a linear space, $\langle f, x \rangle = 0\ \forall x \in M$, so $f \in M^\perp$, hence $\langle f, x_0 \rangle = 0$ — contradicting $\langle f, x_0 \rangle > \alpha$.

For $N$, $N \subset (N^\perp)^\perp$ and the inclusion is closed, so $\overline{N} \subset (N^\perp)^\perp$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why $(N^\perp)^\perp = \overline{N}$ may fail)</span></p>

It may happen that $(N^\perp)^\perp$ is *strictly bigger* than $\overline{N}$ (Exercise 1.16). It is instructive to "try" the same proof and see why it fails. Suppose $f_0 \in E^\star$ with $f_0 \in (N^\perp)^\perp$ and $f_0 \notin \overline{N}$. Apply Hahn–Banach in $E^\star$: there is $\xi \in E^{\star\star}$ that strictly separates $\lbrace f_0 \rbrace$ and $\overline{N}$. *We cannot derive a contradiction* unless we happen to know $\xi \in E$ — more precisely, $\xi = Jx_0$ for some $x_0 \in E$. Hence:

* If $E$ is **reflexive**, $J$ is surjective and indeed $(N^\perp)^\perp = \overline{N}$.
* In general, one can show that $(N^\perp)^\perp$ coincides with the closure of $N$ in the **weak-$\star$ topology** $\sigma(E^\star, E)$ (Chapter 3).

</div>

### 1.4 A Quick Introduction to the Theory of Conjugate Convex Functions

In this section we consider functions $\varphi$ defined on a set $E$ with values in $(-\infty, +\infty]$, so that $\varphi$ may take the value $+\infty$ but $-\infty$ is excluded. The reason for the asymmetry is convenient bookkeeping in convex analysis: if both $\pm \infty$ were allowed, basic operations like $\varphi + \psi$ would produce undefined sums $\infty - \infty$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Domain, epigraph)</span></p>

* The **(effective) domain** of $\varphi$ is $D(\varphi) = \lbrace x \in E\,;\ \varphi(x) < +\infty \rbrace$.
* The **epigraph** of $\varphi$ is the set
  $$
  \mathrm{epi}\,\varphi = \lbrace [x, \lambda] \in E \times \mathbb{R}\,;\ \varphi(x) \le \lambda \rbrace.
  $$

We say $\varphi \not\equiv +\infty$ if $D(\varphi) \neq \emptyset$.

</div>

#### Lower semicontinuity

We assume now that $E$ is a **topological space**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lower semicontinuous function)</span></p>

A function $\varphi : E \to (-\infty, +\infty]$ is **lower semicontinuous** (l.s.c.) if for every $\lambda \in \mathbb{R}$ the sublevel set

$$
[\varphi \le \lambda] = \lbrace x \in E\,;\ \varphi(x) \le \lambda \rbrace
$$

is closed.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Elementary facts about l.s.c. functions)</span></p>

1. $\varphi$ is l.s.c. $\iff$ $\mathrm{epi}\,\varphi$ is closed in $E \times \mathbb{R}$.
2. If $\varphi$ is l.s.c., then for every $x \in E$ and every $\varepsilon > 0$ there is a neighborhood $V$ of $x$ such that $\varphi(y) \ge \varphi(x) - \varepsilon$ for all $y \in V$. Conversely, this characterizes l.s.c. In particular, in a *metric space*, $\varphi$ is l.s.c. iff $\liminf_{n \to \infty} \varphi(x_n) \ge \varphi(x)$ for every sequence $x_n \to x$ — i.e.,
   $$\boxed{\;\liminf_{n \to \infty} \varphi(x_n) \ge \varphi(x).\;}$$
3. If $\varphi_1, \varphi_2$ are l.s.c., then $\varphi_1 + \varphi_2$ is l.s.c.
4. If $(\varphi_i)_{i \in I}$ is a family of l.s.c. functions, then their **superior envelope** $\varphi(x) = \sup_{i \in I} \varphi_i(x)$ is l.s.c.
5. If $E$ is **compact** and $\varphi$ is l.s.c., then $\inf_E \varphi$ is achieved.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why l.s.c. is the right notion in optimization)</span></p>

Lower semicontinuity is the minimal regularity required to guarantee existence of minimizers via the **direct method of the calculus of variations**: take a minimizing sequence, extract a convergent subsequence (using compactness), and use l.s.c. to conclude the limit is a minimizer. The supremum-stability property (4) is what allows one to *manufacture* l.s.c. functions out of families of continuous ones — this is exactly how the conjugate $\varphi^\star$ below acquires its l.s.c. property.

</div>

#### Convex functions

We now assume $E$ is a *vector space*.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Convex function)</span></p>

A function $\varphi : E \to (-\infty, +\infty]$ is **convex** if

$$
\boxed{\;\varphi(tx + (1-t)y) \le t\varphi(x) + (1-t)\varphi(y) \quad \forall x, y \in E,\ \forall t \in (0, 1).\;}
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Elementary facts about convex functions)</span></p>

1. $\varphi$ is convex $\iff$ $\mathrm{epi}\,\varphi$ is a convex set in $E \times \mathbb{R}$.
2. If $\varphi$ is convex, then for every $\lambda \in \mathbb{R}$ the sublevel set $[\varphi \le \lambda]$ is convex; the converse is *not* true.
3. If $\varphi_1, \varphi_2$ are convex, then $\varphi_1 + \varphi_2$ is convex.
4. If $(\varphi_i)_{i \in I}$ is a family of convex functions, then $\sup_i \varphi_i$ is convex.

</div>

#### The conjugate function

We assume hereinafter that $E$ is an n.v.s.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Conjugate / Legendre–Fenchel transform)</span></p>

Let $\varphi : E \to (-\infty, +\infty]$ with $\varphi \not\equiv +\infty$ (so $D(\varphi) \neq \emptyset$). The **conjugate function** $\varphi^\star : E^\star \to (-\infty, +\infty]$ is

$$
\boxed{\;\varphi^\star(f) = \sup_{x \in E} \lbrace \langle f, x \rangle - \varphi(x) \rbrace \quad (f \in E^\star).\;}
$$

($\varphi^\star$ is also called the **Legendre transform** of $\varphi$.)

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">($\varphi^\star$ is convex and l.s.c.)</span></p>

The conjugate $\varphi^\star$ is **convex and l.s.c.** on $E^\star$. Indeed, for each fixed $x \in E$ the affine map $f \mapsto \langle f, x \rangle - \varphi(x)$ is convex and continuous (hence l.s.c.) on $E^\star$. Therefore $\varphi^\star$, the superior envelope of these affine functions as $x$ ranges over $E$, is convex and l.s.c. by Properties 4 of l.s.c. and convexity above.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Young's inequality)</span></p>

Directly from the definition of $\varphi^\star$,

$$
\boxed{\;\langle f, x \rangle \le \varphi(x) + \varphi^\star(f) \quad \forall x \in E,\ \forall f \in E^\star.\;} \tag{11}
$$

This is sometimes called **Young's inequality**. (Of course this fact is *obvious* from the definition.) The classical form of Young's inequality (proved in Theorem 4.6 of Chapter 4) asserts that

$$
ab \le \tfrac{1}{p} a^p + \tfrac{1}{p'} b^{p'} \quad \forall a, b \ge 0, \tag{12}
$$

with $1 < p < \infty$ and $\tfrac{1}{p} + \tfrac{1}{p'} = 1$. Inequality $(12)$ becomes a special case of $(11)$ with $E = E^\star = \mathbb{R}$ and $\varphi(t) = \tfrac{1}{p}\lvert t \rvert^p$, $\varphi^\star(s) = \tfrac{1}{p'}\lvert s \rvert^{p'}$ (Exercise 1.18).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1.10</span><span class="math-callout__name">(Convex l.s.c. functions are bounded below by an affine function)</span></p>

Assume $\varphi : E \to (-\infty, +\infty]$ is convex, l.s.c., and $\varphi \not\equiv +\infty$. Then $\varphi^\star \not\equiv +\infty$, and in particular $\varphi$ is bounded below by an affine continuous function.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $x_0 \in D(\varphi)$ and pick $\lambda_0 < \varphi(x_0)$. Apply Theorem 1.7 (Hahn–Banach, second geometric form) in the space $E \times \mathbb{R}$ with $A = \mathrm{epi}\,\varphi$ (closed because $\varphi$ is l.s.c.; convex because $\varphi$ is convex) and $B = \lbrace [x_0, \lambda_0] \rbrace$ (compact). There exists a closed hyperplane $H = [\Phi = \alpha]$ in $E \times \mathbb{R}$ that strictly separates $A$ and $B$. Note that $x \mapsto \Phi([x, 0])$ is a continuous linear functional on $E$, so $\Phi([x, 0]) = \langle f, x \rangle$ for some $f \in E^\star$. Letting $k = \Phi([0, 1])$,

$$
\Phi([x, \lambda]) = \langle f, x \rangle + k\lambda \quad \forall [x, \lambda] \in E \times \mathbb{R}.
$$

Writing $\Phi > \alpha$ on $A$ and $\Phi < \alpha$ on $B$,

$$
\langle f, x \rangle + k\lambda > \alpha\ \forall [x, \lambda] \in \mathrm{epi}\,\varphi, \tag{13}
$$

$$
\langle f, x_0 \rangle + k\lambda_0 < \alpha.
$$

In particular $\langle f, x \rangle + k\varphi(x) > \alpha\ \forall x \in D(\varphi)$, and $\langle f, x_0 \rangle + k\varphi(x_0) > \alpha > \langle f, x_0 \rangle + k\lambda_0$, so $k > 0$. By $(13)$,

$$
\Big\langle -\tfrac{1}{k} f,\, x \Big\rangle - \varphi(x) < -\tfrac{\alpha}{k} \quad \forall x \in D(\varphi),
$$

hence $\varphi^\star(-\tfrac{1}{k} f) < +\infty$. $\square$

</details>
</div>

#### The biconjugate and the Fenchel–Moreau theorem

If we iterate the operation $\star$, we obtain a function $\varphi^{\star\star}$ defined a priori on $E^{\star\star}$. Instead, we choose to *restrict* $\varphi^{\star\star}$ to $E$ via the canonical injection $J$:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Biconjugate)</span></p>

$$
\boxed{\;\varphi^{\star\star}(x) = \sup_{f \in E^\star} \lbrace \langle f, x \rangle - \varphi^\star(f) \rbrace \quad (x \in E).\;}
$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1.11</span><span class="math-callout__name">(Fenchel–Moreau)</span></p>

Assume $\varphi : E \to (-\infty, +\infty]$ is **convex, l.s.c., and $\varphi \not\equiv +\infty$**. Then

$$
\varphi^{\star\star} = \varphi.
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 1.11</summary>

We proceed in two steps.

**Step 1.** Assume in addition that $\varphi \ge 0$. We claim $\varphi^{\star\star} = \varphi$.

It is obvious that $\varphi^{\star\star} \le \varphi$ (since $\langle f, x \rangle - \varphi^\star(f) \le \varphi(x)\ \forall x, f$). Suppose, by contradiction, $\varphi^{\star\star}(x_0) < \varphi(x_0)$ for some $x_0 \in E$. We may possibly have $\varphi(x_0) = +\infty$, but $\varphi^{\star\star}(x_0)$ is always finite. Apply Theorem 1.7 in $E \times \mathbb{R}$ with $A = \mathrm{epi}\,\varphi$ and $B = \lbrace [x_0, \varphi^{\star\star}(x_0)] \rbrace$. There exist $f \in E^\star,\ k \in \mathbb{R},\ \alpha \in \mathbb{R}$ with

$$
\langle f, x \rangle + k\lambda > \alpha \quad \forall [x, \lambda] \in \mathrm{epi}\,\varphi, \tag{14}
$$

$$
\langle f, x_0 \rangle + k\varphi^{\star\star}(x_0) < \alpha. \tag{15}
$$

It follows from $(14)$ that $k \ge 0$ (fix some $x \in D(\varphi)$ and let $\lambda \to +\infty$). Note: unlike the proof of Proposition 1.10, we cannot assert $k > 0$; we may have $k = 0$, which corresponds to a "vertical" separating hyperplane.

Let $\varepsilon > 0$. Since $\varphi \ge 0$, $(14)$ gives

$$
\langle f, x \rangle + (k + \varepsilon)\varphi(x) \ge \alpha \quad \forall x \in D(\varphi).
$$

Therefore $\varphi^\star\!\left(-\frac{f}{k+\varepsilon}\right) \le -\frac{\alpha}{k+\varepsilon}$. From the definition of $\varphi^{\star\star}(x_0)$,

$$
\varphi^{\star\star}(x_0) \ge \Big\langle -\tfrac{f}{k+\varepsilon}, x_0 \Big\rangle - \varphi^\star\!\left(-\tfrac{f}{k+\varepsilon}\right) \ge \Big\langle -\tfrac{f}{k+\varepsilon}, x_0 \Big\rangle + \tfrac{\alpha}{k+\varepsilon}.
$$

Hence $\langle f, x_0 \rangle + (k + \varepsilon)\varphi^{\star\star}(x_0) \ge \alpha\ \forall \varepsilon > 0$, which contradicts $(15)$.

**Step 2.** *General case.* Fix $f_0 \in D(\varphi^\star)$ ($D(\varphi^\star) \neq \emptyset$ by Proposition 1.10) and define

$$
\overline{\varphi}(x) = \varphi(x) - \langle f_0, x \rangle + \varphi^\star(f_0),
$$

so that $\overline{\varphi}$ is convex, l.s.c., $\overline{\varphi} \not\equiv +\infty$, and $\overline{\varphi} \ge 0$ (the latter by Young's inequality). Step 1 gives $(\overline{\varphi})^{\star\star} = \overline{\varphi}$. Direct computation:

$$
(\overline{\varphi})^\star(f) = \varphi^\star(f + f_0) - \varphi^\star(f_0),
$$

$$
(\overline{\varphi})^{\star\star}(x) = \varphi^{\star\star}(x) - \langle f_0, x \rangle + \varphi^\star(f_0).
$$

Equating $(\overline{\varphi})^{\star\star} = \overline{\varphi}$ yields $\varphi^{\star\star} = \varphi$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric meaning of Fenchel–Moreau)</span></p>

Theorem 1.11 says: a *convex l.s.c.* function is the **pointwise supremum of all the affine functions that minorize it**. The conjugate $\varphi^\star$ is just the bookkeeping device that records the affine minorants: each $f \in D(\varphi^\star)$ encodes the affine function $x \mapsto \langle f, x \rangle - \varphi^\star(f)$, which is $\le \varphi$ everywhere.

The convexity, l.s.c., and "$\not\equiv +\infty$" hypotheses are *necessary*: without them, the supremum of affine minorants gives the **convex l.s.c. envelope** of $\varphi$, not $\varphi$ itself.

</div>

#### Examples

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1</span><span class="math-callout__name">($\varphi(x) = \|x\|$)</span></p>

For $\varphi(x) = \|x\|$, one checks easily

$$
\varphi^\star(f) = \begin{cases} 0 & \text{if } \|f\| \le 1, \\ +\infty & \text{if } \|f\| > 1. \end{cases}
$$

Therefore

$$
\varphi^{\star\star}(x) = \sup_{\substack{f \in E^\star \\ \|f\| \le 1}} \langle f, x \rangle.
$$

Writing the equality $\varphi^{\star\star} = \varphi$, we obtain again part of Corollary 1.4: $\|x\| = \sup_{\|f\| \le 1} \langle f, x \rangle$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2</span><span class="math-callout__name">(Indicator function and supporting function of a set)</span></p>

Given a nonempty set $K \subset E$, define

$$
\boxed{\;I_K(x) = \begin{cases} 0 & \text{if } x \in K, \\ +\infty & \text{if } x \notin K. \end{cases}\;}
$$

The function $I_K$ is the **indicator function** of $K$ — *not* to be confused with the characteristic function $\chi_K$ (which is $1$ on $K$ and $0$ outside). Note that $I_K$ is convex iff $K$ is convex; $I_K$ is l.s.c. iff $K$ is closed.

The conjugate $(I_K)^\star$ is called the **supporting function** of $K$:

$$
(I_K)^\star(f) = \sup_{x \in K} \langle f, x \rangle.
$$

**Special case.** If $K = M$ is a *linear subspace*, then

$$
(I_M)^\star = I_{M^\perp},\qquad (I_M)^{\star\star} = I_{(M^\perp)^\perp}.
$$

Assuming $M$ is closed, write $(I_M)^{\star\star} = I_M$; then $(M^\perp)^\perp = M$ — so in some sense **Theorem 1.11 is a counterpart of Proposition 1.9.**

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Indicator functions as a unifying device)</span></p>

The reason convex analysts work with $(-\infty, +\infty]$-valued functions instead of plain $\mathbb{R}$-valued ones is **precisely** to accommodate $I_K$: minimizing $f(x)$ subject to $x \in K$ is equivalent to minimizing the unconstrained sum $f + I_K$ on the whole space. Constraints become *additional terms* in the objective. This trick makes Lagrangian duality, the KKT conditions, etc., flow effortlessly from the conjugate calculus.

</div>

#### The Fenchel–Rockafellar duality theorem

We conclude this chapter with another useful property of conjugate functions.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1.12</span><span class="math-callout__name">(Fenchel–Rockafellar)</span></p>

Let $\varphi, \psi : E \to (-\infty, +\infty]$ be two convex functions. Assume there is some $x_0 \in D(\varphi) \cap D(\psi)$ such that $\varphi$ is **continuous at $x_0$**. Then

$$
\inf_{x \in E} \lbrace \varphi(x) + \psi(x) \rbrace = \sup_{f \in E^\star} \lbrace -\varphi^\star(-f) - \psi^\star(f) \rbrace = \max_{f \in E^\star} \lbrace -\varphi^\star(-f) - \psi^\star(f) \rbrace = -\min_{f \in E^\star} \lbrace \varphi^\star(-f) + \psi^\star(f) \rbrace.
$$

</div>

The proof relies on:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 1.4</span><span class="math-callout__name">(Interior of a convex set)</span></p>

Let $C \subset E$ be a convex set. Then $\mathrm{Int}\,C$ is convex. If, in addition, $\mathrm{Int}\,C \neq \emptyset$, then

$$
\overline{C} = \overline{\mathrm{Int}\,C}.
$$

</div>

(For a proof of Lemma 1.4, see Exercise 1.7.)

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 1.12</summary>

Set $a = \inf_x \lbrace \varphi(x) + \psi(x) \rbrace$ and $b = \sup_f \lbrace -\varphi^\star(-f) - \psi^\star(f) \rbrace$. Young's inequality gives $b \le a$.

If $a = -\infty$ the conclusion is obvious. Assume $a \in \mathbb{R}$. Set $C = \mathrm{epi}\,\varphi$, so $\mathrm{Int}\,C \neq \emptyset$ (since $\varphi$ is continuous at $x_0$). Apply Theorem 1.6 (first geometric form) with $A = \mathrm{Int}\,C$ and

$$
B = \lbrace [x, \lambda] \in E \times \mathbb{R}\,;\ \lambda \le a - \psi(x) \rbrace.
$$

Both $A, B$ are nonempty convex, and $A \cap B = \emptyset$: if $[x, \lambda] \in A$ then $\lambda > \varphi(x)$, while $\psi(x) \ge a - \varphi(x) \ge a - \lambda$ (by definition of $a$), so $[x,\lambda] \notin B$. Thus there is a closed hyperplane $H$ separating $A$ and $B$. By Lemma 1.4, $H$ also separates $\overline{A} = \overline{C}$ and $B$. So $f \in E^\star,\ k \in \mathbb{R},\ \alpha \in \mathbb{R}$ satisfy

$$
\langle f, x \rangle + k\lambda \ge \alpha\quad \forall [x, \lambda] \in C, \tag{16}
$$

$$
\langle f, x \rangle + k\lambda \le \alpha\quad \forall [x, \lambda] \in B. \tag{17}
$$

Choose $x = x_0$ and let $\lambda \to +\infty$ in $(16)$: $k \ge 0$. We claim $k > 0$. Suppose $k = 0$; then $\|f\| \neq 0$ (since $\Phi \not\equiv 0$). From $(16)$, $\langle f, x \rangle \ge \alpha$ for all $x \in D(\varphi)$. From $(17)$, $\langle f, x \rangle \le \alpha$ for all $x \in D(\psi)$. But $B(x_0, \varepsilon_0) \subset D(\varphi)$ for small $\varepsilon_0 > 0$, so $\langle f, x_0 + \varepsilon_0 z \rangle \ge \alpha\ \forall z \in B(0,1)$, i.e., $\langle f, x_0 \rangle \ge \alpha + \varepsilon_0 \|f\|$. On the other hand $\langle f, x_0 \rangle \le \alpha$ since $x_0 \in D(\psi)$, contradiction. Hence $k > 0$.

From $(16)$ and $(17)$,

$$
\varphi^\star\!\left(-\tfrac{f}{k}\right) \le -\tfrac{\alpha}{k}, \qquad \psi^\star\!\left(\tfrac{f}{k}\right) \le \tfrac{\alpha}{k} - a,
$$

so $-\varphi^\star(-f/k) - \psi^\star(f/k) \ge a$. Combined with $-\varphi^\star(-f/k) - \psi^\star(f/k) \le b \le a$, we get $a = b = -\varphi^\star(-f/k) - \psi^\star(f/k)$. $\square$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3</span><span class="math-callout__name">(Distance to a convex set)</span></p>

Let $K \subset E$ be a nonempty convex set. For every $x_0 \in E$,

$$
\mathrm{dist}(x_0, K) = \inf_{x \in K} \|x - x_0\| = \max_{\substack{f \in E^\star \\ \|f\| \le 1}} \lbrace \langle f, x_0 \rangle - (I_K)^\star(f) \rbrace. \tag{19}
$$

Indeed, $\inf_{x \in K} \|x - x_0\| = \inf_{x \in E} \lbrace \varphi(x) + \psi(x) \rbrace$ with $\varphi(x) = \|x - x_0\|$ and $\psi(x) = I_K(x)$. Apply Theorem 1.12. In the special case $K = M$ a linear subspace,

$$
\mathrm{dist}(x_0, M) = \inf_{x \in M} \|x - x_0\| = \max_{\substack{f \in M^\perp \\ \|f\| \le 1}} \langle f, x_0 \rangle.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Primal vs. dual problems)</span></p>

Relation $(19)$ is informative even when $\inf_{x \in K} \|x - x_0\|$ is *not achieved* (Exercise 1.17). In that case the **primal problem** $\inf\lbrace \varphi(x) + \psi(x) \rbrace$ has no solution, while the **dual problem** $\max\lbrace -\varphi^\star(-f) - \psi^\star(f) \rbrace$ does. This asymmetry is one of the chief reasons to study duality at all: even when the primal is ill-posed, the dual gives sharp upper / lower bounds and exposes the *value* of the problem. The systematic theory of minimal surfaces, optimal transport, and a host of variational inequalities is built on this asymmetry; see I. Ekeland–R. Temam.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4</span><span class="math-callout__name">(Annihilator of a subspace)</span></p>

Let $\varphi : E \to \mathbb{R}$ be convex and continuous and let $M \subset E$ be a linear subspace. Then

$$
\inf_{x \in M} \varphi(x) = -\min_{f \in M^\perp} \varphi^\star(f).
$$

Apply Theorem 1.12 with $\psi = I_M$.

</div>

### Comments on Chapter 1

#### Generalizations and variants of the Hahn–Banach theorems

The first geometric form of Hahn–Banach (Theorem 1.6) remains valid in *general topological vector spaces*. The second geometric form (Theorem 1.7) holds in **locally convex spaces** — such spaces play an important role, for instance, in the theory of distributions (see L. Schwartz, F. Treves, etc.).

#### Applications of the Hahn–Banach theorems

The Hahn–Banach theorems have a *wide* and *diversified* range of applications. Two emblematic examples:

* **(a) The Krein–Milman theorem.** The second geometric form of Hahn–Banach is a basic ingredient in its proof. Recall: the **convex hull** of $A \subset E$, denoted $\mathrm{conv}\,A$, is the smallest convex set containing $A$; equivalently, $\mathrm{conv}\,A$ consists of all *finite* convex combinations of elements of $A$:
  $$
  \mathrm{conv}\,A = \Big\lbrace \sum_{i \in I} t_i a_i\,;\ I \text{ finite},\ a_i \in A,\ t_i \ge 0,\ \sum_i t_i = 1 \Big\rbrace.
  $$
  The **closed convex hull** of $A$, denoted $\overline{\mathrm{conv}}\,A$, is its closure. A point $x$ in a convex set $K$ is **extremal** if it cannot be written as $x = (1-t)x_0 + tx_1$ with $t \in (0,1)$ and $x_0, x_1 \in K$, $x_0 \neq x_1$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1.13</span><span class="math-callout__name">(Krein–Milman)</span></p>

Let $K \subset E$ be a *compact* convex set. Then $K$ coincides with the closed convex hull of its extremal points.

</div>

Krein–Milman has numerous extensions (Choquet's integral representation theorem, Bochner's theorem, Bernstein's theorem, …). For its proof, see Problem 1.

* **(b) Theory of partial differential equations.** The existence of a *fundamental solution* for a general differential operator $P(D)$ with constant coefficients (the **Malgrange–Ehrenpreis theorem**) relies on the analytic form of Hahn–Banach. In the same spirit, the proof of the existence of the Green's function for the Laplacian via the method of P. Lax (Lax §9.5 and P. Garabedian) uses Hahn–Banach. The proof of the existence of $u \in L^\infty(\Omega)$ for $\mathrm{div}\,u = f$ in $\Omega \subset \mathbb{R}^N$, given $f \in L^N(\Omega)$, also rests on Hahn–Banach (see Bourgain–Brezis). Surprisingly, the resulting $u$ depends *nonlinearly* on $f$; in fact, there exists no bounded linear operator $L^N \to L^\infty$ giving $u$ as a function of $f$. **The use of Zorn's lemma — and of the underlying axiom of choice — in the proof of Hahn–Banach can be delicate and may destroy the linear character of the problem.** Sometimes there is no way to circumvent this obstruction.

#### Convex functions

*Convex analysis* and *duality principles* have considerably expanded in recent years; see Moreau, Rockafellar, Ekeland–Temam, Ekeland–Turnbull, Clarke, Aubin–Ekeland, Hiriart-Urruty–Lemaréchal, etc. Among the applications:

* (a) Game theory, economics, optimization, convex programming.
* (b) Mechanics (Moreau, Germain, Duvaut–J. L. Lions, Temam–Strang); also nonconvex duality (Toland), problems in plasma physics (Damlamian), rotating chains (Auchmuty).
* (c) Theory of monotone operators and nonlinear semigroups (Brezis, Browder, Barbu, Phelps).
* (d) Variational problems involving *periodic solutions* of Hamiltonian systems and *nonlinear vibrating strings* (Clarke, Ekeland, Lasry, Brezis, Coron, Nirenberg).
* (e) Theory of *large deviations* in probability (Azencott, Stroock).
* (f) Theory of partial differential equations and complex analysis (Hörmander).

#### Extensions of bounded linear operators

Let $E, F$ be two Banach spaces and let $G \subset E$ be a closed subspace. Let $S : G \to F$ be a bounded linear operator. One may ask whether it is possible to extend $S$ to a bounded linear $T : E \to F$. Note that Corollary 1.2 settles this question only when $F = \mathbb{R}$. In general, the answer is *negative* — even if $E$ and $F$ are both reflexive (Exercise 1.27) — except in special cases:

* **(a)** If $\dim F < \infty$. Choose a basis in $F$ and apply Corollary 1.2 to each component of $S$.
* **(b)** If $G$ admits a *topological complement* (Section 2.4). True in particular if $\dim G < \infty$ or $\mathrm{codim}\,G < \infty$, or if $E$ is a **Hilbert space**.

One may also ask whether there is an extension $T$ *with the same norm*, i.e., $\|T\|_{\mathcal{L}(E,F)} = \|S\|_{\mathcal{L}(G,F)}$. The answer is yes only in some *exceptional cases*; see Nachbin, Kelley, Exercise 5.15.

## Chapter 2: The Uniform Boundedness Principle and the Closed Graph Theorem

While Chapter 1 was about *extending* and *separating* — qualitative existence results that hold in any normed space — Chapter 2 is about *quantitative* control. The leitmotif: in a *complete* space, certain pointwise estimates upgrade automatically to uniform ones. The mechanism is the **Baire category theorem**, and the three classical consequences are:

* the **uniform boundedness principle** (Banach–Steinhaus): pointwise bounded $\Longrightarrow$ uniformly bounded,
* the **open mapping theorem** (and its corollary, the closed graph theorem): a continuous linear surjection between Banach spaces is open,
* a structural theory of **complementary subspaces** and **right/left inverses** of bounded linear operators.

These let us study **unbounded operators** rigorously — including their adjoints, kernels, ranges, and the duality between closed range and a priori estimates, which is the foundation for solvability theory in PDE.

### 2.1 The Baire Category Theorem

The following classical result plays an essential role in the proofs of Chapter 2.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.1</span><span class="math-callout__name">(Baire)</span></p>

Let $X$ be a *complete metric space* and let $(X_n)_{n \ge 1}$ be a sequence of *closed* subsets of $X$. Assume that

$$
\mathrm{Int}\,X_n = \emptyset \quad \text{for every } n \ge 1.
$$

Then

$$
\mathrm{Int}\!\left(\bigcup_{n=1}^\infty X_n\right) = \emptyset.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Equivalent contrapositive form)</span></p>

The Baire category theorem is often used in the following form. Let $X$ be a nonempty complete metric space. Let $(X_n)_{n \ge 1}$ be a sequence of closed subsets such that

$$
\bigcup_{n=1}^\infty X_n = X.
$$

Then there exists some $n_0$ such that $\mathrm{Int}\,X_{n_0} \neq \emptyset$.

In other words: a complete metric space is *not* a countable union of nowhere-dense closed sets — at least one of them must contain an open ball.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 2.1</summary>

Set $O_n = X_n^c$, so that each $O_n$ is open and dense in $X$ (since $\mathrm{Int}\,X_n = \emptyset$). Our aim is to prove that $G = \bigcap_{n=1}^\infty O_n$ is dense in $X$. Let $\omega$ be a nonempty open set in $X$; we shall prove $\omega \cap G \neq \emptyset$.

As usual, set $B(x, r) = \lbrace y \in X\,;\ d(y, x) < r \rbrace$.

Pick any $x_0 \in \omega$ and $r_0 > 0$ with $\overline{B(x_0, r_0)} \subset \omega$. Then choose $x_1 \in B(x_0, r_0) \cap O_1$ and $r_1 > 0$ such that

$$
\begin{cases} \overline{B(x_1, r_1)} \subset B(x_0, r_0) \cap O_1, \\ 0 < r_1 < r_0/2, \end{cases}
$$

which is always possible since $O_1$ is open and dense. By induction one constructs sequences $(x_n)$ and $(r_n)$ with

$$
\begin{cases} \overline{B(x_{n+1}, r_{n+1})} \subset B(x_n, r_n) \cap O_{n+1}, \quad \forall n \ge 0, \\ 0 < r_{n+1} < r_n/2. \end{cases}
$$

It follows that $(x_n)$ is Cauchy; let $x_n \to \ell$. Since $x_{n+p} \in B(x_n, r_n)$ for all $n, p \ge 0$, we obtain $\ell \in \overline{B(x_n, r_n)}$ for every $n \ge 0$. In particular, $\ell \in \omega \cap G$. $\square$

</details>
</div>

### 2.2 The Uniform Boundedness Principle

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Bounded linear operators)</span></p>

Let $E$ and $F$ be two n.v.s. We denote by $\mathcal{L}(E, F)$ the space of *continuous* (i.e., bounded) linear operators from $E$ into $F$, equipped with the norm

$$
\|T\|_{\mathcal{L}(E, F)} = \sup_{\substack{x \in E \\ \|x\| \le 1}} \|Tx\|.
$$

As usual, we write $\mathcal{L}(E)$ instead of $\mathcal{L}(E, E)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.2</span><span class="math-callout__name">(Banach–Steinhaus, uniform boundedness principle)</span></p>

Let $E$ and $F$ be two Banach spaces and let $(T_i)_{i \in I}$ be a family (not necessarily countable) of continuous linear operators from $E$ into $F$. Assume

$$
\sup_{i \in I} \|T_i x\| < \infty \quad \forall x \in E. \tag{1}
$$

Then

$$
\sup_{i \in I} \|T_i\|_{\mathcal{L}(E, F)} < \infty. \tag{2}
$$

In other words, there exists a constant $c$ such that

$$
\|T_i x\| \le c \|x\| \quad \forall x \in E,\ \forall i \in I.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Pointwise → uniform)</span></p>

The conclusion of Theorem 2.2 is *quite remarkable and surprising*. From **pointwise estimates** one derives a **global (uniform) estimate**. The completeness of $E$ is essential; in non-complete spaces the conclusion can fail.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 2.2</summary>

For every $n \ge 1$, let

$$
X_n = \lbrace x \in E\,;\ \forall i \in I,\ \|T_i x\| \le n \rbrace,
$$

so that $X_n$ is closed (intersection of closed sets), and by $(1)$,

$$
\bigcup_{n=1}^\infty X_n = E.
$$

By Baire (Remark above), $\mathrm{Int}\,X_{n_0} \neq \emptyset$ for some $n_0 \ge 1$. Pick $x_0 \in E$ and $r > 0$ with $B(x_0, r) \subset X_{n_0}$. Then

$$
\|T_i(x_0 + rz)\| \le n_0 \quad \forall i \in I,\ \forall z \in B(0, 1),
$$

leading to $r \|T_i\|_{\mathcal{L}(E, F)} \le n_0 + \|T_i x_0\|$, which implies $(2)$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Linearity is essential)</span></p>

Recall that, in general, a *pointwise limit* of continuous maps need *not* be continuous. The linearity assumption plays an essential role in Theorem 2.2. Note, however, that in the setting of Theorem 2.2 it does **not** follow that $\|T_n - T\|_{\mathcal{L}(E, F)} \to 0$: pointwise convergence of operators does not imply norm convergence.

</div>

Here are a few direct consequences of the uniform boundedness principle.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2.3</span><span class="math-callout__name">(Pointwise limit of bounded operators)</span></p>

Let $E$ and $F$ be two Banach spaces. Let $(T_n)$ be a sequence of continuous linear operators from $E$ into $F$ such that for every $x \in E$, $T_n x$ converges (as $n \to \infty$) to a limit denoted by $Tx$. Then:

1. $\sup_n \|T_n\|\_{\mathcal{L}(E, F)} < \infty$,
2. $T \in \mathcal{L}(E, F)$,
3. $\|T\|\_{\mathcal{L}(E, F)} \le \liminf_{n \to \infty} \|T_n\|\_{\mathcal{L}(E, F)}$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

(a) follows directly from Theorem 2.2: there exists $c$ with $\|T_n x\| \le c \|x\|\ \forall n,\ \forall x$. At the limit, $\|Tx\| \le c\|x\|$. $T$ is clearly linear, hence (b). Finally, $\|T_n x\| \le \|T_n\|_{\mathcal{L}(E,F)} \|x\|$, and (c) follows by taking $\liminf$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2.4</span><span class="math-callout__name">(Weakly bounded ⇔ strongly bounded)</span></p>

Let $G$ be a Banach space and let $B \subset G$. Assume that

$$
\text{for every } f \in G^\star \text{ the set } f(B) = \lbrace \langle f, x \rangle\,;\ x \in B \rbrace \text{ is bounded (in } \mathbb{R}\text{)}. \tag{3}
$$

Then

$$
B \text{ is bounded.} \tag{4}
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Apply Theorem 2.2 with $E = G^\star$, $F = \mathbb{R}$, and $I = B$. For every $b \in B$ set $T_b(f) = \langle f, b \rangle$, $f \in E = G^\star$. By $(3)$, $\sup_{b \in B} \lvert T_b(f) \rvert < \infty\ \forall f \in E$. It follows from Theorem 2.2 that there is a constant $c$ with

$$
\lvert \langle f, b \rangle \rvert \le c\|f\| \quad \forall f \in G^\star,\ \forall b \in B.
$$

By Corollary 1.4, $\|b\| \le c\ \forall b \in B$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Looking at $B$ through linear functionals)</span></p>

Corollary 2.4 says: in order to prove that a set $B$ is bounded it suffices to "look" at $B$ through the bounded linear functionals. This is a familiar procedure in *finite-dimensional* spaces, where the linear functionals are the components with respect to some basis. In some sense, Corollary 2.4 *replaces*, in infinite-dimensional spaces, the use of components.

Sometimes one expresses the conclusion of Corollary 2.4 by saying that **"weakly bounded" $\iff$ "strongly bounded"** (see Chapter 3 for the formal weak-topology framework).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2.5</span><span class="math-callout__name">(Dual statement)</span></p>

Let $G$ be a Banach space and let $B^\star \subset G^\star$. Assume that

$$
\text{for every } x \in G \text{ the set } \langle B^\star, x \rangle = \lbrace \langle f, x \rangle\,;\ f \in B^\star \rbrace \text{ is bounded (in } \mathbb{R}\text{)}. \tag{5}
$$

Then $B^\star$ is bounded.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Apply Theorem 2.2 with $E = G$, $F = \mathbb{R}$, $I = B^\star$, and $T_b(x) = \langle b, x \rangle$ for $b \in B^\star,\ x \in G$. We obtain $c$ such that $\lvert \langle b, x \rangle \rvert \le c\|x\|\ \forall b \in B^\star, \forall x \in G$. By the definition of the dual norm, $\|b\| \le c$ for every $b \in B^\star$. $\square$

</details>
</div>

### 2.3 The Open Mapping Theorem and the Closed Graph Theorem

Two basic results due to Banach.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.6</span><span class="math-callout__name">(Open mapping theorem)</span></p>

Let $E$ and $F$ be two Banach spaces and let $T \in \mathcal{L}(E, F)$ be **surjective** ($=$ onto). Then there exists a constant $c > 0$ such that

$$
T(B_E(0, 1)) \supset B_F(0, c). \tag{7}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why "open mapping"?)</span></p>

Property $(7)$ implies that **the image under $T$ of any open set in $E$ is an open set in $F$** — which justifies the name. Indeed, suppose $U$ is open in $E$; let $y_0 \in T(U)$, $y_0 = Tx_0$ for some $x_0 \in U$. Pick $r > 0$ with $B(x_0, r) \subset U$, i.e., $x_0 + B(0, r) \subset U$. Then

$$
y_0 + T(B(0, r)) \subset T(U).
$$

Using $(7)$ (and homogeneity), $T(B(0, r)) \supset B(0, rc)$, hence $B(y_0, rc) \subset T(U)$.

</div>

Some important consequences.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2.7</span><span class="math-callout__name">(Continuous bijection ⇒ continuous inverse)</span></p>

Let $E$ and $F$ be two Banach spaces and let $T \in \mathcal{L}(E, F)$ be **bijective** (injective and surjective). Then $T^{-1}$ is also continuous (from $F$ into $E$).

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Corollary 2.7</summary>

By $(7)$ and injectivity: if $x \in E$ is chosen so that $\|Tx\| < c$, then $\|x\| < 1$. By homogeneity,

$$
\|x\| \le \tfrac{1}{c} \|Tx\| \quad \forall x \in E,
$$

and therefore $T^{-1}$ is continuous. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2.8</span><span class="math-callout__name">(Equivalence of comparable Banach norms)</span></p>

Let $E$ be a vector space provided with two norms, $\|\cdot\|_1$ and $\|\cdot\|_2$. Assume that $E$ is a Banach space for **both** norms and that there is $C \ge 0$ with

$$
\|x\|_2 \le C \|x\|_1 \quad \forall x \in E.
$$

Then the two norms are **equivalent**: there exists $c > 0$ such that

$$
\|x\|_1 \le c \|x\|_2 \quad \forall x \in E.
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Corollary 2.8</summary>

Apply Corollary 2.7 with $E = (E, \|\cdot\|_1)$, $F = (E, \|\cdot\|_2)$, and $T = I$ (identity). $\square$

</details>
</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 2.6 (sketch in two steps)</summary>

**Step 1.** Let $T$ be a linear surjective operator from $E$ onto $F$. Then there is $c > 0$ such that

$$
\overline{T(B(0, 1))} \supset B(0, 2c). \tag{8}
$$

Set $X_n = n \overline{T(B(0, 1))}$. Surjectivity gives $\bigcup_{n=1}^\infty X_n = F$. Baire yields some $n_0$ with $\mathrm{Int}(X_{n_0}) \neq \emptyset$, hence $\mathrm{Int}\,\overline{T(B(0,1))} \neq \emptyset$. Pick $c > 0$ and $y_0 \in F$ with

$$
B(y_0, 4c) \subset \overline{T(B(0, 1))}. \tag{9}
$$

In particular $y_0 \in \overline{T(B(0,1))}$ and by symmetry $-y_0 \in \overline{T(B(0,1))}$. Adding the two,

$$
B(0, 4c) \subset \overline{T(B(0, 1))} + \overline{T(B(0, 1))}.
$$

Convexity of $\overline{T(B(0,1))}$ gives $\overline{T(B(0,1))} + \overline{T(B(0,1))} = 2\overline{T(B(0,1))}$, and $(8)$ follows.

**Step 2.** Assume $T$ is a continuous linear operator from $E$ into $F$ satisfying $(8)$. Then $T(B(0, 1)) \supset B(0, c)$.

Choose any $y \in F$ with $\|y\| < c$. We aim to find $x \in E$ with $\|x\| < 1$ and $Tx = y$. By $(8)$,

$$
\forall \varepsilon > 0\ \exists z \in E \text{ with } \|z\| < \tfrac{1}{2} \text{ and } \|y - Tz\| < \varepsilon. \tag{12}
$$

Take $\varepsilon = c/2$ to get $z_1$ with $\|z_1\| < 1/2$ and $\|y - Tz_1\| < c/2$. Apply the same construction to $y - Tz_1$ with $\varepsilon = c/4$ to find $z_2$ with $\|z_2\| < 1/4$ and $\|(y - Tz_1) - Tz_2\| < c/4$. Inductively obtain $(z_n)$ with $\|z_n\| < 1/2^n$ and $\|y - T(z_1 + \cdots + z_n)\| < c/2^n$. The partial sums $x_n = z_1 + \cdots + z_n$ form a Cauchy sequence; let $x_n \to x$ with $\|x\| < 1$ and $y = Tx$ (by continuity of $T$). $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.9</span><span class="math-callout__name">(Closed graph theorem)</span></p>

Let $E$ and $F$ be two Banach spaces. Let $T$ be a *linear* operator from $E$ into $F$. Assume that the graph of $T$,

$$
G(T) = \lbrace [x, Tx]\,;\ x \in E \rbrace,
$$

is **closed** in $E \times F$. Then $T$ is continuous.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Converse is trivial)</span></p>

The converse is obviously true, since the graph of any continuous map (linear or not) is closed. Theorem 2.9 is the *non-trivial* direction: linearity + closed graph in Banach spaces is enough for continuity.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 2.9</summary>

Consider, on $E$, the two norms

$$
\|x\|_1 = \|x\|_E + \|Tx\|_F \quad \text{and} \quad \|x\|_2 = \|x\|_E
$$

(the norm $\|\cdot\|_1$ is called the **graph norm**). It is easy to check, using the assumption that $G(T)$ is closed, that $E$ is a Banach space for the norm $\|\cdot\|_1$. Trivially, $E$ is also a Banach space for $\|\cdot\|_2$, and $\|\cdot\|_2 \le \|\cdot\|_1$. By Corollary 2.8 the two norms are equivalent: there is $c > 0$ with $\|x\|_1 \le c\|x\|_2$, i.e., $\|Tx\|_F \le c\|x\|_E$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why the closed graph theorem matters)</span></p>

To prove $T : E \to F$ is continuous, it suffices to verify the *weaker-looking* property: whenever $x_n \to x$ in $E$ **and** $Tx_n \to y$ in $F$, then $y = Tx$. We do *not* have to argue that $Tx_n$ converges — only that *if* it converges, the limit is $Tx$. The benefit: the hypothesis bundles together *two* convergences, both of which are usually easier to track than continuity directly. This makes the closed graph theorem a workhorse of operator theory and PDE.

</div>

### 2.4 Complementary Subspaces. Right and Left Invertibility of Linear Operators

We start with geometric properties of closed subspaces in a Banach space that follow from the open mapping theorem.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.10</span><span class="math-callout__name">(Decomposition with norm control)</span></p>

Let $E$ be a Banach space. Assume that $G$ and $L$ are two closed linear subspaces such that $G + L$ is closed. Then there is a constant $C \ge 0$ such that

$$
\begin{cases} \text{every } z \in G + L \text{ admits a decomposition } \\ z = x + y \text{ with } x \in G,\ y \in L,\ \|x\| \le C\|z\|,\ \|y\| \le C\|z\|. \end{cases} \tag{13}
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Consider the product $G \times L$ with norm $\|[x, y]\| = \|x\| + \|y\|$ and the space $G + L$ with the induced norm of $E$. The map $T : G \times L \to G + L$, $T[x, y] = x + y$, is continuous, linear, surjective. By the open mapping theorem there is $c > 0$ such that every $z \in G + L$ with $\|z\| < c$ can be written $z = x + y$ with $x \in G$, $y \in L$, $\|x\| + \|y\| < 1$. By homogeneity, every $z \in G + L$ admits $z = x + y$ with $\|x\| + \|y\| \le \tfrac{1}{c}\|z\|$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2.11</span><span class="math-callout__name">(Distance to intersection)</span></p>

Under the same assumptions as Theorem 2.10, there is a constant $C$ with

$$
\mathrm{dist}(x, G \cap L) \le C \lbrace \mathrm{dist}(x, G) + \mathrm{dist}(x, L) \rbrace \quad \forall x \in E. \tag{14}
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Given $x \in E$ and $\varepsilon > 0$, pick $a \in G$ and $b \in L$ with $\|x - a\| \le \mathrm{dist}(x, G) + \varepsilon$ and $\|x - b\| \le \mathrm{dist}(x, L) + \varepsilon$. Apply $(13)$ to $z = a - b \in G + L$: there exist $a' \in G,\ b' \in L$ with $a - b = a' + b'$, $\|a'\| \le C\|a - b\|$, $\|b'\| \le C\|a - b\|$. Then $a - a' \in G \cap L$ (since $a - a' = b + b'$) and

$$
\mathrm{dist}(x, G \cap L) \le \|x - (a - a')\| \le \|x - a\| + \|a'\| \le (1 + C)(\mathrm{dist}(x, G) + \mathrm{dist}(x, L)) + (1 + 2C)\varepsilon.
$$

Let $\varepsilon \to 0$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Converse of Corollary 2.11)</span></p>

The converse is also true: if $G$ and $L$ are two closed linear subspaces such that $(14)$ holds, then $G + L$ is closed (Exercise 2.16). So the property "$G + L$ closed" is *equivalent* to a quantitative distance-control statement.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Topological complement)</span></p>

Let $G \subset E$ be a closed subspace of a Banach space $E$. A subspace $L \subset E$ is said to be a **topological complement** (or simply *complement*) of $G$ if

1. $L$ is closed,
2. $G \cap L = \lbrace 0 \rbrace$ and $G + L = E$.

We then say $G$ and $L$ are **complementary subspaces** of $E$. If this holds, every $z \in E$ may be uniquely written $z = x + y$ with $x \in G,\ y \in L$. By Theorem 2.10, the projections $z \mapsto x$ and $z \mapsto y$ are continuous linear operators. (This property could serve as an alternative definition.)

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Examples</span><span class="math-callout__name">(Subspaces that admit a complement)</span></p>

1. **Finite-dimensional subspaces.** Every finite-dimensional subspace $G$ admits a complement. Let $e_1, \ldots, e_n$ be a basis of $G$; every $x \in G$ writes $x = \sum x_i e_i$. Set $\varphi_i(x) = x_i$ on $G$. By Hahn–Banach (Corollary 1.2), each $\varphi_i$ extends to $\widetilde{\varphi}\_i \in E^\star$. Then $L = \bigcap_{i=1}^n \widetilde{\varphi}\_i^{-1}(0)$ is a closed complement of $G$.

2. **Closed subspaces of finite codimension.** Every closed subspace $G$ of finite codimension admits a complement: choose any finite-dimensional $L$ with $G \cap L = \lbrace 0 \rbrace$ and $G + L = E$ (closed since finite-dimensional).

   *Typical example:* let $N \subset E^\star$ be a $p$-dimensional subspace; then

   $$
   G = \lbrace x \in E\,;\ \langle f, x \rangle = 0\ \forall f \in N \rbrace = N^\perp
   $$

   is closed and of codimension $p$. Indeed, take a basis $f_1, \ldots, f_p$ of $N$; one finds $e_1, \ldots, e_p \in E$ with $\langle f_i, e_j \rangle = \delta_{ij}$ (look at the surjection $\Phi : E \to \mathbb{R}^p$, $\Phi(x) = (\langle f_1, x\rangle, \ldots, \langle f_p, x\rangle)$; non-surjectivity would yield, via Hahn–Banach (second geometric form), a nonzero $\alpha = (\alpha_1, \ldots, \alpha_p)$ with $\alpha \cdot \Phi(x) = \langle \sum \alpha_i f_i, x\rangle = 0$ for all $x$ — absurd). The span of the $e_i$'s is a complement of $G$.

3. **Hilbert spaces.** In a Hilbert space, every closed subspace admits a complement — namely its orthogonal complement (Section 5.2).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Lindenstrauss–Tzafriri)</span></p>

It is important to know that some closed subspaces (even in *reflexive* Banach spaces) have *no* complement. In fact, a remarkable result of **J. Lindenstrauss and L. Tzafriri** asserts: in *every* Banach space that is not isomorphic to a Hilbert space, there exist closed subspaces *without* any complement. So the existence of complements characterizes Hilbert spaces (up to isomorphism).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Right and left inverses)</span></p>

Let $T \in \mathcal{L}(E, F)$.

* A **right inverse** of $T$ is an operator $S \in \mathcal{L}(F, E)$ with $T \circ S = I_F$.
* A **left inverse** of $T$ is an operator $S \in \mathcal{L}(F, E)$ with $S \circ T = I_E$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.12</span><span class="math-callout__name">(Right invertibility)</span></p>

Assume $T \in \mathcal{L}(E, F)$ is **surjective**. The following are equivalent:

(i) $T$ admits a right inverse.

(ii) $N(T) = T^{-1}(0)$ admits a complement in $E$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

(i) $\Rightarrow$ (ii). If $S$ is a right inverse of $T$, then $R(S) = S(F)$ is a complement of $N(T)$.

(ii) $\Rightarrow$ (i). Let $L$ be a complement of $N(T)$. Let $P$ be the (continuous) projection from $E$ onto $L$. Given $f \in F$, denote by $x$ any solution of $Tx = f$, and note that $S f = P x$ does not depend on the choice of $x$. Then $S \in \mathcal{L}(F, E)$ and $T \circ S = I_F$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Surjective operators without right inverse)</span></p>

In view of Lindenstrauss–Tzafriri and Theorem 2.12, it is easy to construct surjective operators $T$ without a right inverse: pick a closed subspace $G \subset E$ without a complement, set $F = E/G$, and let $T$ be the canonical projection $E \to F$ (Section 11.2).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.13</span><span class="math-callout__name">(Left invertibility)</span></p>

Assume $T \in \mathcal{L}(E, F)$ is **injective**. The following are equivalent:

(i) $T$ admits a left inverse.

(ii) $R(T) = T(E)$ is closed and admits a complement in $F$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

(i) $\Rightarrow$ (ii). $R(T)$ is closed; $N(S)$ is a complement of $R(T)$ (write $f = T S f + (f - T S f)$).

(ii) $\Rightarrow$ (i). Let $P$ be a continuous projection from $F$ onto $R(T)$. For $f \in F$, $Pf \in R(T)$, so there is a unique $x \in E$ with $Tx = Pf$. Set $S f = x$. Then $S \circ T = I_E$, and $S$ is continuous by Corollary 2.7. $\square$

</details>
</div>

### 2.5 Orthogonality Revisited

We give some simple formulas for the orthogonal complement of a sum or an intersection.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.14</span><span class="math-callout__name">(Orthogonal of a sum / intersection)</span></p>

Let $G$ and $L$ be two closed subspaces in $E$. Then

$$
\boxed{\;G \cap L = (G^\perp + L^\perp)^\perp,\;} \tag{16}
$$

$$
\boxed{\;G^\perp \cap L^\perp = (G + L)^\perp.\;} \tag{17}
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

*Proof of $(16)$.* Clearly $G \cap L \subset (G^\perp + L^\perp)^\perp$: if $x \in G \cap L$ and $f = f_1 + f_2 \in G^\perp + L^\perp$, then $\langle f, x\rangle = 0$. Conversely, $G^\perp \subset G^\perp + L^\perp$, so $(G^\perp + L^\perp)^\perp \subset (G^\perp)^\perp = G$ (since $G$ is closed, by Proposition 1.9); similarly $(G^\perp + L^\perp)^\perp \subset L$. Hence $(G^\perp + L^\perp)^\perp \subset G \cap L$.

*Proof of $(17)$.* Same argument. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2.15</span><span class="math-callout__name">(Closure of a sum of orthogonals)</span></p>

Let $G$ and $L$ be two closed subspaces in $E$. Then

$$
(G \cap L)^\perp \supset \overline{G^\perp + L^\perp}, \tag{18}
$$

$$
(G^\perp \cap L^\perp)^\perp = \overline{G + L}. \tag{19}
$$

</div>

(Use Propositions 1.9 and 2.14.)

Here is a deeper result.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.16</span><span class="math-callout__name">(Closedness ↔ sum of orthogonals)</span></p>

Let $G$ and $L$ be two closed subspaces in a Banach space $E$. The following properties are equivalent:

(a) $G + L$ is closed in $E$,

(b) $G^\perp + L^\perp$ is closed in $E^\star$,

(c) $G + L = (G^\perp \cap L^\perp)^\perp$,

(d) $G^\perp + L^\perp = (G \cap L)^\perp$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

(a) $\iff$ (c) follows from $(19)$. (d) $\Rightarrow$ (b) is obvious. We prove (a) $\Rightarrow$ (d) and (b) $\Rightarrow$ (a).

**(a) ⇒ (d).** By $(18)$ it suffices to prove $(G \cap L)^\perp \subset G^\perp + L^\perp$. Given $f \in (G \cap L)^\perp$, define $\varphi : G + L \to \mathbb{R}$ by $\varphi(x) = \langle f, a \rangle$ for any decomposition $x = a + b$ with $a \in G, b \in L$. Independence of decomposition is clear (any two decompositions differ by an element of $G \cap L$, on which $f$ vanishes). By Theorem 2.10 we may pick $a$ with $\|a\| \le C\|x\|$, so $\lvert \varphi(x) \rvert \le C\|x\|$. Extend $\varphi$ to $\widetilde{\varphi} \in E^\star$ (Corollary 1.2), and write $f = (f - \widetilde{\varphi}) + \widetilde{\varphi}$, with $f - \widetilde{\varphi} \in G^\perp$ and $\widetilde{\varphi} \in L^\perp$.

**(b) ⇒ (a).** By Corollary 2.11 there is $C$ with

$$
\mathrm{dist}(f, G^\perp \cap L^\perp) \le C \lbrace \mathrm{dist}(f, G^\perp) + \mathrm{dist}(f, L^\perp) \rbrace \quad \forall f \in E^\star. \tag{20}
$$

Using Theorem 1.12 (Fenchel–Rockafellar) with $\varphi(x) = I_{B_E}(x) - \langle f, x\rangle$ and $\psi(x) = I_G(x)$ (where $B_E = \lbrace x \in E\,;\ \|x\| \le 1 \rbrace$), one shows

$$
\mathrm{dist}(f, G^\perp) = \sup_{\substack{x \in G \\ \|x\| \le 1}} \langle f, x \rangle, \tag{21}
$$

and similarly for $\mathrm{dist}(f, L^\perp)$, $\mathrm{dist}(f, G^\perp \cap L^\perp) = \mathrm{dist}(f, (G + L)^\perp)$. Combining,

$$
\sup_{\substack{x \in \overline{G + L} \\ \|x\| \le 1}} \langle f, x \rangle \le C \Big\lbrace \sup_{\substack{x \in G \\ \|x\| \le 1}} \langle f, x\rangle + \sup_{\substack{x \in L \\ \|x\| \le 1}} \langle f, x \rangle \Big\rbrace \quad \forall f \in E^\star. \tag{24}
$$

This implies $\overline{B_G + B_L} \supset \tfrac{1}{C} B_{\overline{G+L}}$ — otherwise some $x_0 \in \overline{G+L}$ with $\|x_0\| \le 1/C$ and $x_0 \notin \overline{B_G + B_L}$ could be strictly separated, contradicting $(24)$. Now apply Step 2 of the open mapping theorem to $T : G \times L \to \overline{G+L}$, $T[x,y] = x + y$: surjectivity onto $\overline{G+L}$ follows. Hence $G + L = \overline{G + L}$, i.e., $G + L$ is closed. $\square$

</details>
</div>

### 2.6 An Introduction to Unbounded Linear Operators. Definition of the Adjoint

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Unbounded linear operator)</span></p>

Let $E$ and $F$ be two Banach spaces. An **unbounded linear operator** from $E$ into $F$ is a linear map $A : D(A) \subset E \to F$ defined on a linear subspace $D(A) \subset E$ with values in $F$. The set $D(A)$ is the **domain** of $A$.

We say $A$ is **bounded** (or *continuous*) if $D(A) = E$ and there is $c \ge 0$ such that

$$
\|Au\| \le c\|u\| \quad \forall u \in E.
$$

The norm of a bounded operator is

$$
\boxed{\;\|A\|_{\mathcal{L}(E, F)} = \sup_{u \neq 0} \frac{\|Au\|}{\|u\|}.\;}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Terminology)</span></p>

It may happen that an unbounded operator turns out to be bounded — the terminology is slightly inconsistent ("unbounded" $\subset$ "possibly unbounded") but is standard and does not lead to confusion in practice. Concretely, "unbounded" means *we allow $D(A) \neq E$ and we do not require boundedness*; if both happen to hold, we call it bounded.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Graph, range, kernel)</span></p>

$$
\boxed{\;G(A) = \lbrace [u, Au]\,;\ u \in D(A) \rbrace \subset E \times F\;} \quad \text{(graph)},
$$

$$
\boxed{\;R(A) = \lbrace Au\,;\ u \in D(A) \rbrace \subset F\;} \quad \text{(range)},
$$

$$
\boxed{\;N(A) = \lbrace u \in D(A)\,;\ Au = 0 \rbrace \subset E\;} \quad \text{(kernel)}.
$$

A map $A$ is **closed** if $G(A)$ is closed in $E \times F$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(How to verify $A$ is closed)</span></p>

To prove an operator $A$ is closed, one proceeds in general as follows. Take a sequence $(u_n)$ in $D(A)$ with $u_n \to u$ in $E$ and $Au_n \to f$ in $F$. Then check **two** facts:

1. $u \in D(A)$,
2. $f = Au$.

Note: it does *not* suffice to consider sequences with $u_n \to 0$ in $E$ and $Au_n \to f$ in $F$ (and to prove $f = 0$) — that's a strictly weaker assertion.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Closedness, kernel, range)</span></p>

If $A$ is closed, then $N(A)$ is closed; however, $R(A)$ need *not* be closed. In practice, *most* unbounded operators that arise in PDE are *closed* and *densely defined* — i.e., $\overline{D(A)} = E$.

</div>

#### The adjoint $A^\star$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Adjoint of a densely defined operator)</span></p>

Let $A : D(A) \subset E \to F$ be a densely defined unbounded linear operator. The adjoint $A^\star : D(A^\star) \subset F^\star \to E^\star$ is defined as follows.

**Domain.**
$$
D(A^\star) = \lbrace v \in F^\star\,;\ \exists c \ge 0 \text{ such that } \lvert \langle v, Au \rangle \rvert \le c\|u\| \ \forall u \in D(A) \rbrace.
$$

**Action.** For $v \in D(A^\star)$, the linear map $g : D(A) \to \mathbb{R}$, $g(u) = \langle v, Au \rangle$, satisfies $\lvert g(u) \rvert \le c\|u\|$ on $D(A)$. By Hahn–Banach (or by extension by continuity, since $D(A)$ is dense), $g$ extends *uniquely* to $f \in E^\star$ with $\lvert f(u) \rvert \le c\|u\|$. Set

$$
\boxed{\;A^\star v = f.\;}
$$

The fundamental relation between $A$ and $A^\star$ is

$$
\boxed{\;\langle v, Au \rangle_{F^\star, F} = \langle A^\star v, u \rangle_{E^\star, E} \quad \forall u \in D(A),\ \forall v \in D(A^\star).\;}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(No need for Hahn–Banach)</span></p>

It is *not* necessary to invoke Hahn–Banach to extend $g$. It suffices to use the classical *extension by continuity*, which applies since $D(A)$ is dense, $g$ is uniformly continuous on $D(A)$, and $\mathbb{R}$ is complete.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Density of $D(A^\star)$)</span></p>

It may happen that $D(A^\star)$ is *not* dense in $F^\star$ (even if $A$ is closed); but this is a rather pathological situation (Exercise 2.22). It is always true that **if $A$ is closed, then $D(A^\star)$ is dense in $F^\star$ for the weak-$\star$ topology** $\sigma(F^\star, F)$ (Chapter 3, Problem 9). In particular, if $F$ is reflexive, $D(A^\star)$ is dense in $F^\star$ for the usual norm topology (Theorem 3.24).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bounded $A$ ⇒ $\|A^\star\| = \|A\|$)</span></p>

If $A$ is a bounded operator then $A^\star$ is also a bounded operator (from $F^\star$ into $E^\star$) and

$$
\boxed{\;\|A^\star\|_{\mathcal{L}(F^\star, E^\star)} = \|A\|_{\mathcal{L}(E, F)}.\;}
$$

Indeed, $D(A^\star) = F^\star$, and from $\langle A^\star v, u \rangle = \langle v, Au\rangle$ we get $\lvert \langle A^\star v, u \rangle \rvert \le \|A\|\|u\|\|v\|$, so $\|A^\star v\| \le \|A\|\|v\|$, i.e., $\|A^\star\| \le \|A\|$. Conversely, $\lvert \langle v, Au \rangle \rvert \le \|A^\star\|\|u\|\|v\|$ implies (Corollary 1.4) $\|Au\| \le \|A^\star\|\|u\|$, so $\|A\| \le \|A^\star\|$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.17</span><span class="math-callout__name">($A^\star$ is closed)</span></p>

Let $A : D(A) \subset E \to F$ be a densely defined unbounded linear operator. Then $A^\star$ is **closed**, i.e., $G(A^\star)$ is closed in $F^\star \times E^\star$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $v_n \in D(A^\star)$ with $v_n \to v$ in $F^\star$ and $A^\star v_n \to f$ in $E^\star$. Need: (a) $v \in D(A^\star)$, (b) $A^\star v = f$. We have $\langle v_n, Au \rangle = \langle A^\star v_n, u \rangle\ \forall u \in D(A)$. Limit: $\langle v, Au \rangle = \langle f, u \rangle\ \forall u \in D(A)$. Hence $\lvert \langle v, Au\rangle \rvert \le \|f\|\|u\|$, so $v \in D(A^\star)$ and $A^\star v = f$. $\square$

</details>
</div>

#### A geometric link: graphs and orthogonals

The graphs of $A$ and $A^\star$ are related by a very simple **orthogonality relation**. Consider the isomorphism $I : F^\star \times E^\star \to E^\star \times F^\star$ defined by $I([v, f]) = [-f, v]$. Then for a densely defined operator $A$,

$$
\boxed{\;I[G(A^\star)] = G(A)^\perp.\;}
$$

Indeed, $[v, f] \in G(A^\star) \iff \langle f, u \rangle = \langle v, Au \rangle\ \forall u \in D(A) \iff -\langle f, u \rangle + \langle v, Au\rangle = 0 \iff [-f, v] \in G(A)^\perp$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2.18</span><span class="math-callout__name">(Orthogonality between range and kernel)</span></p>

Let $A : D(A) \subset E \to F$ be densely defined and closed. Then

$$
\begin{aligned}
&\text{(i)} && N(A) = R(A^\star)^\perp, \\
&\text{(ii)} && N(A^\star) = R(A)^\perp, \\
&\text{(iii)} && N(A)^\perp \supset \overline{R(A^\star)}, \\
&\text{(iv)} && N(A^\star)^\perp = \overline{R(A)}.
\end{aligned}
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

(iii) and (iv) follow from (i), (ii), and Proposition 1.9. For (i) and (ii), one can give a direct proof (Exercise 2.18). It is instructive, however, to relate them to Proposition 2.14 via the device: in $X = E \times F$ (so $X^\star = E^\star \times F^\star$), let $G = G(A)$ and $L = E \times \lbrace 0 \rbrace$. One checks

$$
N(A) \times \lbrace 0 \rbrace = G \cap L, \qquad E \times R(A) = G + L,
$$

$$
\lbrace 0 \rbrace \times N(A^\star) = G^\perp \cap L^\perp, \qquad R(A^\star) \times F^\star = G^\perp + L^\perp.
$$

Then (i) follows from $(G^\perp + L^\perp)^\perp = G \cap L$ (Proposition 2.14): $R(A^\star)^\perp \times \lbrace 0\rbrace = N(A) \times \lbrace 0 \rbrace$. Similarly (ii). $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">($N(A)^\perp$ vs $\overline{R(A^\star)}$)</span></p>

It may happen, even if $A$ is bounded, that $N(A)^\perp \neq \overline{R(A^\star)}$ (Exercise 2.23). However, it is always true that $N(A)^\perp$ is the closure of $R(A^\star)$ for the weak-$\star$ topology $\sigma(E^\star, E)$. In particular, if $E$ is **reflexive**, $N(A)^\perp = \overline{R(A^\star)}$.

</div>

### 2.7 A Characterization of Operators with Closed Range. A Characterization of Surjective Operators

The main result on closed range:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.19</span><span class="math-callout__name">(Closed range theorem)</span></p>

Let $A : D(A) \subset E \to F$ be densely defined and closed. The following are equivalent:

(i) $R(A)$ is closed,

(ii) $R(A^\star)$ is closed,

(iii) $R(A) = N(A^\star)^\perp$,

(iv) $R(A^\star) = N(A)^\perp$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

With the notation from the proof of Corollary 2.18,

* (i) $\iff$ $G + L$ closed in $X$,
* (ii) $\iff$ $G^\perp + L^\perp$ closed in $X^\star$,
* (iii) $\iff$ $G + L = (G^\perp \cap L^\perp)^\perp$,
* (iv) $\iff$ $(G \cap L)^\perp = G^\perp + L^\perp$.

The conclusion follows from Theorem 2.16. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Closed range ↔ a priori distance estimate)</span></p>

For closed unbounded $A$, $R(A)$ is closed iff there is a constant $C$ such that

$$
\mathrm{dist}(u, N(A)) \le C\|Au\| \quad \forall u \in D(A);
$$

see Exercise 2.14. This is the *a priori* estimate that one verifies in practice for elliptic PDE.

</div>

The next result is a useful characterization of *surjective* operators.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.20</span><span class="math-callout__name">(Surjectivity ↔ a priori estimate on $A^\star$)</span></p>

Let $A : D(A) \subset E \to F$ be densely defined and closed. The following are equivalent:

(a) $A$ is surjective, i.e., $R(A) = F$,

(b) there is a constant $C$ such that
   $$ \|v\| \le C\|A^\star v\| \quad \forall v \in D(A^\star), $$

(c) $N(A^\star) = \lbrace 0 \rbrace$ and $R(A^\star)$ is closed.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Method of a priori estimates)</span></p>

The implication (b) $\Rightarrow$ (a) is sometimes useful in practice to establish that an operator $A$ is surjective. One proceeds as follows. Assume $v$ satisfies $A^\star v = f$. One *tries* to prove that

$$
\|v\| \le C\|f\|
$$

(with $C$ independent of $f$). This is the **method of a priori estimates**. *We are not concerned with whether the equation $A^\star v = f$ admits a solution*; we *assume* $v$ is a priori given and try to estimate its norm. Once this estimate holds for *all* $v \in D(A^\star)$, surjectivity of $A$ follows automatically.

This method is the heart of regularity theory for elliptic and parabolic PDE: existence is reduced to bounding solutions in advance.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

**(a) ⇒ (b).** Set $B^\star = \lbrace v \in D(A^\star)\,;\ \|A^\star v\| \le 1 \rbrace$. By homogeneity, it suffices to prove $B^\star$ is bounded. By Corollary 2.5 (uniform boundedness), it suffices to show that for every $f_0 \in F$ the set $\langle B^\star, f_0\rangle$ is bounded in $\mathbb{R}$. Since $A$ is surjective, $f_0 = Au_0$ for some $u_0 \in D(A)$. For $v \in B^\star$,

$$
\langle v, f_0\rangle = \langle v, Au_0 \rangle = \langle A^\star v, u_0 \rangle,
$$

so $\lvert \langle v, f_0 \rangle \rvert \le \|u_0\|$.

**(b) ⇒ (c).** $N(A^\star) = \lbrace 0 \rbrace$ is immediate from (b). For closedness of $R(A^\star)$: if $f_n = A^\star v_n \to f$, then by (b), $\|v_n - v_m\| \le C\|A^\star v_n - A^\star v_m\| = C\|f_n - f_m\|$, so $(v_n)$ is Cauchy; let $v_n \to v$. Since $A^\star$ is closed (Proposition 2.17), $A^\star v = f$.

**(c) ⇒ (a).** Closed range theorem (Theorem 2.19): $R(A) = N(A^\star)^\perp = \lbrace 0\rbrace^\perp = F$. $\square$

</details>
</div>

There is a *dual* statement.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.21</span><span class="math-callout__name">(Surjectivity of $A^\star$ ↔ a priori estimate on $A$)</span></p>

Let $A : D(A) \subset E \to F$ be densely defined and closed. The following are equivalent:

(a) $A^\star$ is surjective, i.e., $R(A^\star) = E^\star$,

(b) there is a constant $C$ such that
   $$ \|u\| \le C\|Au\| \quad \forall u \in D(A), $$

(c) $N(A) = \lbrace 0 \rbrace$ and $R(A)$ is closed.

</div>

(The proof is analogous to Theorem 2.20; left as an exercise.)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Finite-dimensional case)</span></p>

If *either* $\dim E < \infty$ *or* $\dim F < \infty$, then

$$
A \text{ surjective} \iff A^\star \text{ injective}, \qquad A^\star \text{ surjective} \iff A \text{ injective},
$$

a classical fact for linear operators in finite-dimensional spaces. The reason: $R(A)$ and $R(A^\star)$ are finite-dimensional, hence closed.

In the *general* case one only has the implications

$$
A \text{ surjective} \Longrightarrow A^\star \text{ injective}, \qquad A^\star \text{ surjective} \Longrightarrow A \text{ injective}.
$$

The converses *fail*. **Counterexample.** Let $E = F = \ell^2$; for $x = (x_n)\_{n \ge 1} \in \ell^2$, $Ax = (\tfrac{1}{n}x_n)\_{n \ge 1}$. Then $A$ is bounded with $A^\star = A$. Both $A$ and $A^\star$ are injective but *not* surjective: $R(A)$ is dense in $\ell^2$ but not closed.

</div>

### Comments on Chapter 2

1. **Subspaces without complement — explicit examples.** One may write down explicitly some simple closed subspaces *without complement*. For example, $c_0$ is a closed subspace of $\ell^\infty$ without complement (cf. C. DeVito; the notation $c_0$ and $\ell^\infty$ is explained in Section 11.3). Other examples in W. Rudin (subspaces of $L^1$), G. Köthe, B. Beauzamy (subspaces of $\ell^p$, $p \neq 2$).

2. **Beyond Banach spaces: Fréchet spaces.** Most of the results of Chapter 2 extend to *Fréchet spaces* (locally convex spaces that are metrizable and complete). There are many possible extensions (Schaefer, Horváth, Edwards, Treves, Köthe), motivated by the **theory of distributions** (L. Schwartz), in which many important spaces are *not* Banach spaces. For applications to PDE see Hörmander, Treves.

3. **Further extensions.** There are various extensions of the results of Section 2.5 in T. Kato.

## Chapter 3: Weak Topologies. Reflexive Spaces. Separable Spaces. Uniform Convexity

So far we have worked exclusively with the *strong* (norm) topology. Chapter 3 enlarges the toolbox by introducing two coarser topologies on a Banach space and its dual:

* the **weak topology** $\sigma(E, E^\star)$ on $E$ — the coarsest topology making every functional $f \in E^\star$ continuous;
* the **weak-$\star$ topology** $\sigma(E^\star, E)$ on $E^\star$ — the coarsest topology making every evaluation $f \mapsto \langle f, x\rangle$ (for fixed $x \in E$) continuous.

Why bother? **Compactness.** In an infinite-dimensional Banach space the closed unit ball is *never* strongly compact (Theorem 6.5), so the direct method of the calculus of variations cannot rely on the strong topology. A coarser topology has *more* compact sets, and in particular:

* (Banach–Alaoglu–Bourbaki) the closed unit ball $B_{E^\star}$ is *always* weak-$\star$ compact;
* (Kakutani) the closed unit ball $B_E$ is weakly compact iff $E$ is **reflexive**.

These compactness facts, combined with weak-l.s.c. of convex strongly continuous functionals, are the engine behind existence proofs in PDE, optimization, and the calculus of variations.

The chapter also introduces two further properties of Banach spaces — **separability** and **uniform convexity** — which control the metrizability of weak topologies on bounded sets and which, in the case of uniform convexity (Milman–Pettis), automatically imply reflexivity.

### 3.1 The Coarsest Topology for Which a Collection of Maps Becomes Continuous

Let us recall a well-known concept from general topology. Suppose $X$ is a set (with no structure) and $(Y_i)_{i \in I}$ is a collection of topological spaces. We are given a collection of maps $(\varphi_i)_{i \in I}$ with $\varphi_i : X \to Y_i$. Two natural questions:

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem 1</span><span class="math-callout__name">(Cheapest topology making all $\varphi_i$ continuous)</span></p>

Construct a topology on $X$ that makes all the maps $(\varphi_i)_{i \in I}$ continuous. If possible, find a topology $\mathcal{T}$ that is the **most economical** in the sense that it has the *fewest open sets*.

</div>

The discrete topology trivially works but is far from cheapest — it is the most expensive. As we shall see, there is always a (unique) cheapest topology, called the **coarsest** or **weakest** (or *initial*) topology associated to the collection $(\varphi_i)\_{i \in I}$.

If $\omega_i \subset Y_i$ is any open set, then $\varphi_i^{-1}(\omega_i)$ *must* be open in $\mathcal{T}$. As $\omega_i$ runs through the family of open sets of $Y_i$ and $i$ runs through $I$, we obtain a family of subsets of $X$, each of which must be open. Denote this family by $(U_\lambda)_{\lambda \in \Lambda}$. This family need not be a topology, leading to:

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Problem 2</span><span class="math-callout__name">(Cheapest topology containing $(U_\lambda)$)</span></p>

Given a set $X$ and a family $(U_\lambda)\_{\lambda \in \Lambda}$ of subsets, construct the cheapest topology $\mathcal{T}$ on $X$ in which $U_\lambda$ is open for all $\lambda$.

</div>

In other words, find the cheapest family $\mathcal{F}$ of subsets of $X$ that is *stable* under $\bigcap_{\text{finite}}$ and $\bigcup_{\text{arbitrary}}$ and contains $(U_\lambda)$. Construction:

1. First take all *finite* intersections $\bigcap_{\lambda \in \Gamma} U_\lambda$ ($\Gamma \subset \Lambda$ finite). Call the resulting family $\Phi$.
2. Then take *arbitrary* unions of elements from $\Phi$. Call the resulting family $\mathcal{F}$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 3.1</span><span class="math-callout__name">($\mathcal{F}$ is stable under finite intersections)</span></p>

The family $\mathcal{F}$ is stable under $\bigcap_{\text{finite}}$ (and trivially under $\bigcup_{\text{arbitrary}}$).

</div>

(The proof is a delightful exercise in set theory; see Folland.)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Order of operations matters)</span></p>

One **cannot reverse the order of operations**. If we first took $\bigcup_{\text{arbitrary}}$ and then $\bigcap_{\text{finite}}$, the outcome would be a family stable under finite intersection but *not* under arbitrary unions; one would have to consider arbitrary unions once more, and the process stabilizes only after this iteration.

</div>

To summarize: the open sets of $\mathcal{T}$ are obtained by first taking $\bigcap_{\text{finite}}$ of sets of the form $\varphi_i^{-1}(\omega_i)$, then $\bigcup_{\text{arbitrary}}$. For every $x \in X$, a **basis of neighborhoods** of $x$ for $\mathcal{T}$ is given by sets

$$
\bigcap_{i \in J,\ \text{finite}} \varphi_i^{-1}(V_i),\qquad V_i \text{ a neighborhood of } \varphi_i(x) \text{ in } Y_i.
$$

In what follows we equip $X$ with this weakest topology $\mathcal{T}$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.1</span><span class="math-callout__name">(Sequential convergence in $\mathcal{T}$)</span></p>

Let $(x_n)$ be a sequence in $X$. Then $x_n \to x$ in $\mathcal{T}$ if and only if $\varphi_i(x_n) \to \varphi_i(x)$ for every $i \in I$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.2</span><span class="math-callout__name">(Continuity into $X$)</span></p>

Let $Z$ be a topological space and $\psi : Z \to X$. Then $\psi$ is continuous iff $\varphi_i \circ \psi : Z \to Y_i$ is continuous for every $i \in I$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Proposition 3.1</summary>

If $x_n \to x$, continuity of each $\varphi_i$ gives $\varphi_i(x_n) \to \varphi_i(x)$. Conversely, let $U$ be a neighborhood of $x$. We may assume $U = \bigcap_{i \in J} \varphi_i^{-1}(V_i)$ with $J \subset I$ finite. For each $i \in J$ pick $N_i$ such that $\varphi_i(x_n) \in V_i$ for $n \ge N_i$. For $n \ge N = \max_{i \in J} N_i$, $x_n \in U$. $\square$

</details>
</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Proposition 3.2</summary>

If $\psi$ is continuous, so is $\varphi_i \circ \psi$. Conversely, every open $U \subset X$ has the form $U = \bigcup_{\text{arb}} \bigcap_{\text{fin}} \varphi_i^{-1}(\omega_i)$, and

$$
\psi^{-1}(U) = \bigcup_{\text{arb}} \bigcap_{\text{fin}} \psi^{-1}[\varphi_i^{-1}(\omega_i)] = \bigcup_{\text{arb}} \bigcap_{\text{fin}} (\varphi_i \circ \psi)^{-1}(\omega_i),
$$

open in $Z$. $\square$

</details>
</div>

### 3.2 Definition and Elementary Properties of the Weak Topology $\sigma(E, E^\star)$

Let $E$ be a Banach space and let $f \in E^\star$. Denote by $\varphi_f : E \to \mathbb{R}$ the linear functional $\varphi_f(x) = \langle f, x\rangle$. As $f$ runs through $E^\star$ we obtain a collection $(\varphi_f)_{f \in E^\star}$ of maps from $E$ into $\mathbb{R}$. We now ignore the usual norm topology on $E$ and define a new topology:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Weak topology)</span></p>

The **weak topology** $\sigma(E, E^\star)$ on $E$ is the coarsest topology associated to the collection $(\varphi_f)_{f \in E^\star}$ (in the sense of §3.1, with $X = E$, $Y_i = \mathbb{R}$, $I = E^\star$).

</div>

Since every $\varphi_f$ is continuous for the strong topology, **the weak topology is weaker (has fewer open sets) than the strong topology**.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.3</span><span class="math-callout__name">($\sigma(E, E^\star)$ is Hausdorff)</span></p>

The weak topology $\sigma(E, E^\star)$ is Hausdorff.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Given $x_1 \neq x_2$ in $E$, by Hahn–Banach (second geometric form) there is a closed hyperplane strictly separating $\lbrace x_1\rbrace$ and $\lbrace x_2\rbrace$: $f \in E^\star,\ \alpha \in \mathbb{R}$ with $\langle f, x_1\rangle < \alpha < \langle f, x_2\rangle$. Set

$$
O_1 = \varphi_f^{-1}((-\infty, \alpha)),\qquad O_2 = \varphi_f^{-1}((\alpha, +\infty)).
$$

Then $O_1, O_2$ are disjoint weakly open neighborhoods of $x_1, x_2$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.4</span><span class="math-callout__name">(Basis of weak neighborhoods)</span></p>

Let $x_0 \in E$. Given $\varepsilon > 0$ and a *finite* set $\lbrace f_1, \ldots, f_k \rbrace \subset E^\star$, the set

$$
V(f_1, \ldots, f_k;\,\varepsilon) = \lbrace x \in E\,;\ \lvert \langle f_i, x - x_0\rangle \rvert < \varepsilon\ \forall i\rbrace
$$

is a $\sigma(E, E^\star)$-neighborhood of $x_0$, and these form a **basis of neighborhoods** of $x_0$ as $\varepsilon, k$, and the $f_i$'s vary.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Weak convergence)</span></p>

If $(x_n)$ converges to $x$ in the weak topology $\sigma(E, E^\star)$ we write

$$
\boxed{\;x_n \rightharpoonup x.\;}
$$

To avoid confusion we say "$x_n \rightharpoonup x$ weakly in $\sigma(E, E^\star)$"; for emphasis we say "$x_n \to x$ strongly," meaning $\|x_n - x\| \to 0$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.5</span><span class="math-callout__name">(Properties of weak convergence)</span></p>

Let $(x_n)$ be a sequence in $E$.

1. $x_n \rightharpoonup x$ in $\sigma(E, E^\star)$ $\iff$ $\langle f, x_n\rangle \to \langle f, x\rangle$ for every $f \in E^\star$.
2. If $x_n \to x$ strongly, then $x_n \rightharpoonup x$.
3. If $x_n \rightharpoonup x$ weakly, then $(\|x_n\|)$ is **bounded** and $\|x\| \le \liminf \|x_n\|$.
4. If $x_n \rightharpoonup x$ in $E$ and $f_n \to f$ strongly in $E^\star$, then $\langle f_n, x_n\rangle \to \langle f, x\rangle$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

(i) is Proposition 3.1 applied to the family $(\varphi_f)$. (ii) follows from (i): $\lvert \langle f, x_n\rangle - \langle f, x\rangle\rvert \le \|f\|\|x_n - x\| \to 0$. (iii) uses uniform boundedness (Corollary 2.4): for every $f \in E^\star$, $(\langle f, x_n\rangle)_n$ is bounded, hence $(\|x_n\|)$ is bounded; passing to the limit in $\lvert \langle f, x_n\rangle\rvert \le \|f\|\|x_n\|$ gives $\lvert \langle f, x\rangle\rvert \le \|f\|\liminf \|x_n\|$, so by Corollary 1.4, $\|x\| \le \liminf \|x_n\|$. (iv) follows from $\lvert \langle f_n, x_n\rangle - \langle f, x\rangle\rvert \le \|f_n - f\|\|x_n\| + \lvert \langle f, x_n - x\rangle\rvert$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.6</span><span class="math-callout__name">(Finite dimension: weak = strong)</span></p>

When $E$ is **finite-dimensional**, the weak topology $\sigma(E, E^\star)$ and the usual topology are the *same*. In particular, $(x_n)$ converges weakly iff it converges strongly.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Since the weak topology has fewer open sets, it suffices to show every strongly open set is weakly open. Let $x_0 \in E$ and $U$ a strong neighborhood. Pick $r > 0$ with $B(x_0, r) \subset U$, and a basis $e_1, \ldots, e_k$ of $E$ with $\|e_i\| = 1$. Each coordinate map $x \mapsto x_i$ is in $E^\star$; call it $f_i$. For $V = \lbrace x\,;\ \lvert \langle f_i, x - x_0\rangle\rvert < \varepsilon\ \forall i\rbrace$ and $\varepsilon = r/k$,

$$
\|x - x_0\| \le \sum_{i=1}^k \lvert \langle f_i, x - x_0\rangle\rvert < k\varepsilon = r,
$$

so $V \subset U$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Strict coarseness in infinite dimension)</span></p>

In *any* infinite-dimensional space the weak topology is **strictly coarser** than the strong topology. Two illustrative examples:

* **Example 1.** The unit *sphere* $S = \lbrace x\,;\ \|x\| = 1\rbrace$ is *never* weakly closed in infinite dimensions:

  $$\overline{S}^{\sigma(E, E^\star)} = B_E \quad (\text{closed unit ball}).$$

  Indeed, every $x_0 \in E$ with $\|x_0\| < 1$ is in the weak closure of $S$. (Geometric reason: every weak neighborhood $V$ of $x_0$ contains an *infinite-dimensional affine subspace* through $x_0$; since $E$ is infinite-dimensional, the kernel $\bigcap_i f_i^{-1}(0)$ of finitely many functionals is non-trivial, so $V$ contains a line — in fact a "huge" affine subspace — through $x_0$, which must hit $S$.) Conversely $B_E = \bigcap_{\|f\| \le 1} \lbrace x\,;\ \lvert \langle f, x\rangle\rvert \le 1\rbrace$ is an intersection of weakly closed sets.

* **Example 2.** The open unit ball $U = \lbrace x\,;\ \|x\| < 1\rbrace$ is *never* weakly open (otherwise $S = B_E \cap U^c$ would be weakly closed, contradicting Example 1).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Weak topology is not metrizable)</span></p>

In infinite-dimensional spaces the weak topology is **never metrizable** — there is no metric (a fortiori no norm) on $E$ that induces $\sigma(E, E^\star)$ on all of $E$ (Exercise 3.8). However, as we shall see (Theorem 3.29), if $E^\star$ is *separable* one can define a metric on $E$ that induces on **bounded sets** the topology $\sigma(E, E^\star)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Schur-type spaces)</span></p>

Usually, in infinite-dimensional spaces, there exist sequences that converge weakly but not strongly. For example, if $E^\star$ is separable or $E$ is reflexive, one can construct $(x_n)$ with $\|x_n\| = 1$ and $x_n \rightharpoonup 0$ (Exercise 3.22). However, there exist **infinite-dimensional spaces in which every weakly convergent sequence is strongly convergent** — e.g., $\ell^1$ has this *Schur property* (Problem 8). Such spaces are quite "rare" and somewhat *pathological*. This does **not** contradict the previous remark: weak and strong topologies are always distinct in infinite dimension; what coincides is only the set of *convergent sequences*.

Keep in mind: two **metric** spaces with the same convergent sequences have identical topologies; two **topological** spaces with the same convergent sequences need *not* have identical topologies.

</div>

### 3.3 Weak Topology, Convex Sets, and Linear Operators

Every weakly closed set is strongly closed, and the converse fails. However, for *convex* sets the two notions coincide:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.7</span><span class="math-callout__name">(Mazur: convex weakly closed = convex strongly closed)</span></p>

Let $C \subset E$ be convex. Then $C$ is closed in $\sigma(E, E^\star)$ iff $C$ is closed in the strong topology.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Assume $C$ is strongly closed. We show $C^c$ is weakly open. Let $x_0 \notin C$. By Hahn–Banach (second geometric form), there is a closed hyperplane strictly separating $\lbrace x_0\rbrace$ and $C$: $f \in E^\star,\ \alpha \in \mathbb{R}$ with $\langle f, x_0\rangle < \alpha < \langle f, y\rangle\ \forall y \in C$. Set $V = \lbrace x\,;\ \langle f, x\rangle < \alpha\rbrace$. Then $x_0 \in V$, $V \cap C = \emptyset$, and $V$ is weakly open. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 3.8</span><span class="math-callout__name">(Mazur: strong limits of convex combinations)</span></p>

Assume $x_n \rightharpoonup x$ weakly. Then there exists a sequence $(y_n)$ made of *convex combinations* of the $x_n$'s that converges **strongly** to $x$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $C = \mathrm{conv}(\bigcup_p \lbrace x_p\rbrace)$. Since $x$ is in the weak closure of $\bigcup_p\lbrace x_p\rbrace$, hence of $C$, Theorem 3.7 gives $x \in \overline{C}$ (strong closure). $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Closed convex = intersection of half-spaces)</span></p>

The proof shows that every closed convex set $C$ coincides with the *intersection of all closed half-spaces containing $C$*. Variants of Corollary 3.8 are in Exercises 3.4 and 5.24.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 3.9</span><span class="math-callout__name">(Convex strongly l.s.c. ⇒ weakly l.s.c.)</span></p>

If $\varphi : E \to (-\infty, +\infty]$ is convex and l.s.c. in the strong topology, then $\varphi$ is l.s.c. in the weak topology $\sigma(E, E^\star)$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

For every $\lambda \in \mathbb{R}$, $A = \lbrace \varphi \le \lambda \rbrace$ is convex and strongly closed, hence weakly closed by Theorem 3.7. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(In practice)</span></p>

It can be hard to verify weak l.s.c. directly. Corollary 3.9 is often used as

$$
\boxed{\;\varphi \text{ convex and strongly continuous } \Longrightarrow \varphi \text{ weakly l.s.c.}\;}
$$

For example, $\varphi(x) = \|x\|$ is convex and strongly continuous, hence weakly l.s.c. — i.e., $\|x\| \le \liminf \|x_n\|$ when $x_n \rightharpoonup x$ (compare Proposition 3.5(iii)).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.10</span><span class="math-callout__name">(Continuity in strong ⇔ continuity in weak)</span></p>

Let $E, F$ be Banach spaces and $T : E \to F$ a *linear* operator. Then $T$ is continuous in the strong topologies iff $T$ is continuous from $\sigma(E, E^\star)$ to $\sigma(F, F^\star)$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

By Proposition 3.2, $T$ is weak-to-weak continuous iff $f \circ T : E \to \mathbb{R}$ is continuous in $\sigma(E, E^\star)$ for every $f \in F^\star$. If $T$ is strongly continuous, then $f \circ T \in E^\star$, hence weakly continuous.

Conversely, if $T$ is weak-to-weak continuous, then $G(T)$ is closed in $E \times F$ in the product weak topology, hence in the product strong topology (any weakly closed set is strongly closed). The closed graph theorem (Theorem 2.9) gives $T$ strong-to-strong continuous. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Strong-to-weak ⇒ strong-to-strong, but nonlinear maps differ)</span></p>

The argument above shows more: if a *linear* $T$ is continuous from strong-$E$ into weak-$F$, then it is continuous from strong-$E$ into strong-$F$. So for linear operators the four continuity properties

$$
S \to S, \quad W \to W, \quad S \to W, \quad W \to S
$$

(with $S$ = strong, $W$ = weak) are nearly all equivalent (the last is rare: holds iff $T$ is continuous $S \to S$ *and* $\dim R(T) < \infty$).

For *nonlinear* maps, however, strong-strong continuous does **not** imply weak-weak continuous (Exercise 4.20). This is a *major source of difficulty in nonlinear problems* — the natural compactness arguments yield only weak limits, and one must work hard to pass to the limit in nonlinear terms.

</div>

### 3.4 The Weak-$\star$ Topology $\sigma(E^\star, E)$

So far we have two topologies on $E^\star$:

* (a) the usual (strong) topology associated to the dual norm,
* (b) the weak topology $\sigma(E^\star, E^{\star\star})$ — applying §3.3 to $E^\star$.

We define a *third* topology on $E^\star$, the **weak-$\star$ topology**, denoted $\sigma(E^\star, E)$. The "$\star$" in the name reminds us that this topology is defined only on dual spaces.

For every $x \in E$ consider the linear functional $\varphi_x : E^\star \to \mathbb{R}$, $\varphi_x(f) = \langle f, x\rangle$. As $x$ runs through $E$ we obtain a collection $(\varphi_x)_{x \in E}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Weak-$\star$ topology)</span></p>

The **weak-$\star$ topology** $\sigma(E^\star, E)$ is the coarsest topology on $E^\star$ associated to the collection $(\varphi_x)_{x \in E}$.

</div>

Since $E \subset E^{\star\star}$ (via the canonical injection $J$), we have

$$
\boxed{\;\sigma(E^\star, E) \subset \sigma(E^\star, E^{\star\star}) \subset \text{strong topology on } E^\star.\;}
$$

So the weak-$\star$ topology has the fewest open sets of the three.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why bother with another topology?)</span></p>

The reader probably wonders why there is such enthusiasm over weak topologies. The reason: **a coarser topology has more compact sets**. The closed unit ball $B_{E^\star} \subset E^\star$ is *never* compact in the strong topology (unless $\dim E < \infty$; Theorem 6.5), but it is *always* compact in the weak-$\star$ topology — Banach–Alaoglu–Bourbaki below. Knowing the basic role of compact sets (e.g., in existence/minimization), one understands the importance of the weak-$\star$ topology.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.11</span><span class="math-callout__name">($\sigma(E^\star, E)$ is Hausdorff)</span></p>

The weak-$\star$ topology is Hausdorff.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Given $f_1 \neq f_2$ in $E^\star$, pick $x \in E$ with $\langle f_1, x\rangle \neq \langle f_2, x\rangle$ (this does *not* use Hahn–Banach — just $f_1 \neq f_2$). Choose $\alpha$ between, set $O_j = \varphi_x^{-1}(\text{appropriate side})$, and they are disjoint weak-$\star$ neighborhoods. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.12</span><span class="math-callout__name">(Basis of weak-$\star$ neighborhoods)</span></p>

Let $f_0 \in E^\star$. Given a *finite* set $\lbrace x_1, \ldots, x_k\rbrace \subset E$ and $\varepsilon > 0$, the set

$$
V(x_1, \ldots, x_k;\,\varepsilon) = \lbrace f \in E^\star\,;\ \lvert \langle f - f_0, x_i\rangle\rvert < \varepsilon\ \forall i\rbrace
$$

is a $\sigma(E^\star, E)$-neighborhood of $f_0$, and these form a **basis of neighborhoods** of $f_0$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Weak-$\star$ convergence)</span></p>

If a sequence $(f_n)$ in $E^\star$ converges to $f$ in the weak-$\star$ topology we write

$$
\boxed{\;f_n \overset{\star}{\rightharpoonup} f.\;}
$$

To avoid confusion: "$f_n \overset{\star}{\rightharpoonup} f$ in $\sigma(E^\star, E)$," "$f_n \rightharpoonup f$ in $\sigma(E^\star, E^{\star\star})$," "$f_n \to f$ strongly."

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.13</span><span class="math-callout__name">(Properties of weak-$\star$ convergence)</span></p>

Let $(f_n)$ be a sequence in $E^\star$.

1. $f_n \overset{\star}{\rightharpoonup} f$ in $\sigma(E^\star, E)$ $\iff$ $\langle f_n, x\rangle \to \langle f, x\rangle\ \forall x \in E$.
2. $f_n \to f$ strongly $\Rightarrow$ $f_n \rightharpoonup f$ in $\sigma(E^\star, E^{\star\star})$ $\Rightarrow$ $f_n \overset{\star}{\rightharpoonup} f$.
3. If $f_n \overset{\star}{\rightharpoonup} f$, then $(\|f_n\|)$ is bounded and $\|f\| \le \liminf \|f_n\|$.
4. If $f_n \overset{\star}{\rightharpoonup} f$ and $x_n \to x$ strongly in $E$, then $\langle f_n, x_n\rangle \to \langle f, x\rangle$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Mixed convergence pitfall)</span></p>

If $f_n \overset{\star}{\rightharpoonup} f$ in $\sigma(E^\star, E)$ (or even $f_n \rightharpoonup f$ in $\sigma(E^\star, E^{\star\star})$) and $x_n \rightharpoonup x$ in $\sigma(E, E^\star)$, one *cannot* conclude in general that $\langle f_n, x_n\rangle \to \langle f, x\rangle$. Easy counterexamples in Hilbert spaces (orthonormal sequences).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Finite dimension)</span></p>

If $E$ is finite-dimensional, the three topologies on $E^\star$ (strong, weak, weak-$\star$) coincide: $J : E \to E^{\star\star}$ is surjective, hence $\sigma(E^\star, E) = \sigma(E^\star, E^{\star\star})$, and Proposition 3.6 gives them equal to the strong topology.

</div>

#### Continuous functionals on $E^\star$ for the weak-$\star$ topology

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.14</span><span class="math-callout__name">(Continuous w-$\star$ linear functionals come from $E$)</span></p>

Let $\varphi : E^\star \to \mathbb{R}$ be a linear functional that is continuous in the weak-$\star$ topology. Then there exists $x_0 \in E$ such that

$$
\varphi(f) = \langle f, x_0\rangle \quad \forall f \in E^\star.
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 3.2</span><span class="math-callout__name">(Linear-algebra lemma)</span></p>

Let $X$ be a vector space and $\varphi, \varphi_1, \ldots, \varphi_k$ be $(k+1)$ linear functionals on $X$ such that

$$
[\varphi_i(v) = 0\ \forall i] \Longrightarrow [\varphi(v) = 0]. \tag{2}
$$

Then there exist $\lambda_1, \ldots, \lambda_k \in \mathbb{R}$ such that $\varphi = \sum_{i=1}^k \lambda_i \varphi_i$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Lemma 3.2</summary>

Define $F : X \to \mathbb{R}^{k+1}$ by $F(u) = [\varphi(u), \varphi_1(u), \ldots, \varphi_k(u)]$. By assumption, $a = [1, 0, \ldots, 0] \notin R(F)$. Strictly separate $\lbrace a\rbrace$ from $R(F)$ by a hyperplane in $\mathbb{R}^{k+1}$: $\lambda, \lambda_1, \ldots, \lambda_k$ and $\alpha$ with

$$
\lambda < \alpha < \lambda \varphi(u) + \sum_{i=1}^k \lambda_i \varphi_i(u) \quad \forall u \in X.
$$

Since the right-hand side is a linear function of $u$ taking the value $0$ at $u = 0$, we must have $\lambda \varphi + \sum \lambda_i \varphi_i = 0$ identically and $\lambda < 0$. Solve for $\varphi$. $\square$

</details>
</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Proposition 3.14</summary>

Since $\varphi$ is weak-$\star$ continuous, there is a weak-$\star$ neighborhood $V$ of $0$ with $\lvert \varphi(f)\rvert < 1$ on $V$. We may take $V = \lbrace f\,;\ \lvert \langle f, x_i\rangle\rvert < \varepsilon\ \forall i = 1, \ldots, k\rbrace$ for some $x_i \in E$. In particular,

$$
[\langle f, x_i\rangle = 0\ \forall i] \Longrightarrow [\varphi(f) = 0].
$$

Apply Lemma 3.2 with $\varphi_i(f) = \langle f, x_i\rangle$ to get $\varphi(f) = \sum \lambda_i \langle f, x_i\rangle = \langle f, \sum \lambda_i x_i\rangle$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 3.15</span><span class="math-callout__name">(Weak-$\star$ closed hyperplanes)</span></p>

Assume $H$ is a hyperplane in $E^\star$ that is closed in $\sigma(E^\star, E)$. Then $H$ has the form

$$
H = \lbrace f \in E^\star\,;\ \langle f, x_0\rangle = \alpha \rbrace
$$

for some $x_0 \in E$, $x_0 \neq 0$, and some $\alpha \in \mathbb{R}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Weak-closed but not weak-$\star$ closed)</span></p>

Assume $J : E \to E^{\star\star}$ is *not* surjective (i.e., $E$ is not reflexive). Then $\sigma(E^\star, E)$ is **strictly coarser** than $\sigma(E^\star, E^{\star\star})$. Take $\xi \in E^{\star\star}$ with $\xi \notin J(E)$; then

$$
H = \lbrace f \in E^\star\,;\ \langle \xi, f\rangle = 0 \rbrace
$$

is closed in $\sigma(E^\star, E^{\star\star})$ but, by Corollary 3.15, *not* closed in $\sigma(E^\star, E)$. We learn from this example that *convex sets that are closed in the strong topology need not be closed in the weak-$\star$ topology*. There are two types of closed convex sets in $E^\star$:

* (a) those closed in the strong topology (= closed in $\sigma(E^\star, E^{\star\star})$ by Theorem 3.7),
* (b) those closed in $\sigma(E^\star, E)$ — a *strictly smaller* class in general.

</div>

#### Banach–Alaoglu–Bourbaki

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.16</span><span class="math-callout__name">(Banach–Alaoglu–Bourbaki)</span></p>

The closed unit ball

$$
B_{E^\star} = \lbrace f \in E^\star\,;\ \|f\| \le 1\rbrace
$$

is **compact** in the weak-$\star$ topology $\sigma(E^\star, E)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The most essential property)</span></p>

The compactness of $B_{E^\star}$ is *the* most essential property of the weak-$\star$ topology and the deepest reason for its importance.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 3.16</summary>

Consider the Cartesian product $Y = \mathbb{R}^E$ (all maps $E \to \mathbb{R}$), with the product topology. Recall: in this product topology a basis at $\omega \in Y$ is given by $\bigcap_{x \in S\text{ finite}} \lbrace \omega'\,;\ \lvert \omega'_x - \omega_x\rvert < \varepsilon\rbrace$, i.e., the topology of *pointwise convergence*. (Tychonoff: arbitrary products of compact spaces are compact.)

Equip $E^\star$ with $\sigma(E^\star, E)$. Define $\Phi : E^\star \to Y$, $\Phi(f) = (\langle f, x\rangle)_{x \in E}$. Then $\Phi$ is continuous (each coordinate $f \mapsto \langle f, x\rangle$ is, by definition of $\sigma(E^\star, E)$). Its inverse $\Phi^{-1}$ is continuous from $\Phi(E^\star)$ (with subspace product topology) into $E^\star$ (use Proposition 3.2; check each $\omega \mapsto \langle \Phi^{-1}(\omega), x\rangle = \omega_x$ is continuous on $\Phi(E^\star)$). Hence $\Phi$ is a homeomorphism onto its image. Moreover $\Phi(B_{E^\star}) = K$, where

$$
K = \lbrace \omega \in Y\,;\ \lvert \omega_x\rvert \le \|x\|,\ \omega_{x+y} = \omega_x + \omega_y,\ \omega_{\lambda x} = \lambda \omega_x \ \forall \lambda \in \mathbb{R},\ \forall x, y\rbrace.
$$

Write $K = K_1 \cap K_2$ with $K_1 = \prod_{x \in E} [-\|x\|, +\|x\|]$ (compact by Tychonoff) and $K_2 = $ intersection of closed sets $A_{x,y} = \lbrace \omega\,;\ \omega_{x+y} - \omega_x - \omega_y = 0\rbrace$ and $B_{\lambda, x} = \lbrace \omega\,;\ \omega_{\lambda x} - \lambda \omega_x = 0\rbrace$. Hence $K$ is compact, and $B_{E^\star} = \Phi^{-1}(K)$ is compact in $\sigma(E^\star, E)$. $\square$

</details>
</div>

### 3.5 Reflexive Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Reflexive Banach space)</span></p>

Let $E$ be a Banach space and $J : E \to E^{\star\star}$ the canonical injection. $E$ is **reflexive** if $J$ is *surjective*, i.e., $J(E) = E^{\star\star}$.

When $E$ is reflexive, $E^{\star\star}$ is usually identified with $E$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Examples)</span></p>

Many important spaces are reflexive:

* **Finite-dimensional** spaces (since $\dim E = \dim E^\star = \dim E^{\star\star}$);
* $L^p$ and $\ell^p$ for $1 < p < \infty$ (Chapter 4 / 11);
* **Hilbert** spaces (Chapter 5).

Equally important are spaces that are *not* reflexive:

* $L^1$ and $L^\infty$ (and $\ell^1, \ell^\infty$);
* $C(K)$, the continuous functions on an infinite compact metric space $K$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why $J$ matters)</span></p>

It is *essential* to use $J$ in the definition. **R. C. James** has constructed a striking example of a non-reflexive space for which there exists a surjective isometry $E \to E^{\star\star}$ — but this isometry is not $J$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.17</span><span class="math-callout__name">(Kakutani)</span></p>

A Banach space $E$ is reflexive if and only if

$$
B_E = \lbrace x \in E\,;\ \|x\| \le 1\rbrace
$$

is **compact in the weak topology** $\sigma(E, E^\star)$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof (sketch)</summary>

**(⇒).** $J(B_E) = B_{E^{\star\star}}$. By Banach–Alaoglu, $B_{E^{\star\star}}$ is compact in $\sigma(E^{\star\star}, E^\star)$. Then $J^{-1}$ is continuous from $E^{\star\star}$ (with $\sigma(E^{\star\star}, E^\star)$) onto $E$ (with $\sigma(E, E^\star)$), via Proposition 3.2 (check that for every $f \in E^\star$, $\xi \mapsto \langle f, J^{-1}\xi\rangle = \langle \xi, f\rangle$ is continuous on $E^{\star\star}$ for $\sigma(E^{\star\star}, E^\star)$).

**(⇐).** More delicate; rely on:

* **Lemma 3.3 (Helly).** Given $f_1, \ldots, f_k \in E^\star$, $\gamma_1, \ldots, \gamma_k \in \mathbb{R}$:

$$
[\forall \varepsilon > 0\ \exists x_\varepsilon \in B_E,\ \lvert \langle f_i, x_\varepsilon\rangle - \gamma_i\rvert < \varepsilon] \iff \Big\lvert\sum \beta_i \gamma_i \Big\rvert \le \Big\| \sum \beta_i f_i\Big\| \ \forall \beta_i.
$$

* **Lemma 3.4 (Goldstine).** $J(B_E)$ is dense in $B_{E^{\star\star}}$ for $\sigma(E^{\star\star}, E^\star)$, and $J(E)$ is dense in $E^{\star\star}$ for $\sigma(E^{\star\star}, E^\star)$.

Assuming $B_E$ is weakly compact: $J$ is continuous from $\sigma(E, E^\star)$ into $\sigma(E^{\star\star}, E^\star)$, so $J(B_E)$ is compact in $E^{\star\star}$ for $\sigma(E^{\star\star}, E^\star)$, hence closed. Combined with Goldstine ($J(B_E)$ is dense in $B_{E^{\star\star}}$), we get $J(B_E) = B_{E^{\star\star}}$, and $J(E) = E^{\star\star}$. $\square$

</details>
</div>

#### Compactness consequences

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.18</span><span class="math-callout__name">(Bounded sequences in reflexive spaces)</span></p>

Assume $E$ is reflexive and let $(x_n)$ be a bounded sequence in $E$. Then there exists a subsequence $(x_{n_k})$ that converges in the weak topology $\sigma(E, E^\star)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.19</span><span class="math-callout__name">(Eberlein–Šmulian)</span></p>

If $E$ is a Banach space such that *every bounded sequence* admits a weakly convergent subsequence, then $E$ is reflexive.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Compactness vs. sequential compactness)</span></p>

To clarify the connection between Theorems 3.17–3.19:

1. In a *metric* space, compact $\iff$ every sequence has a convergent subsequence.
2. There exist *compact topological* spaces in which some sequences have *no* convergent subsequence — typical example: $X = B_{E^\star}$ in $\sigma(E^\star, E)$ when $E = \ell^\infty$ (Exercise 3.18).
3. There exist *topological* spaces with the property "every sequence admits a convergent subsequence" that are *not* compact.

So Theorems 3.17 and 3.18/3.19 are non-trivial *both* directions in the infinite-dimensional non-metric setting.

</div>

#### Further properties of reflexive spaces

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.20</span><span class="math-callout__name">(Closed subspaces inherit reflexivity)</span></p>

If $E$ is reflexive and $M \subset E$ is a closed linear subspace (with the induced norm), then $M$ is reflexive.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

$M$ has, a priori, two weak topologies: that induced by $\sigma(E, E^\star)$, and its own $\sigma(M, M^\star)$. By Hahn–Banach, every continuous linear functional on $M$ extends to one on $E$, so the two topologies agree. By Kakutani, it suffices to show $B_M$ is compact for $\sigma(M, M^\star)$, equivalently for the topology induced by $\sigma(E, E^\star)$. But $B_E$ is $\sigma(E, E^\star)$-compact (Theorem 3.17) and $M$ is $\sigma(E, E^\star)$-closed (Theorem 3.7, since $M$ is convex strongly closed). So $B_M = B_E \cap M$ is compact. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 3.21</span><span class="math-callout__name">(Reflexivity passes to and from the dual)</span></p>

A Banach space $E$ is reflexive $\iff$ $E^\star$ is reflexive.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

**($E$ reflexive ⇒ $E^\star$ reflexive).** If $E = E^{\star\star}$ via $J$, then $E^\star = E^{\star\star\star}$. Concretely: given $\varphi \in E^{\star\star\star}$, the map $x \mapsto \langle \varphi, Jx\rangle$ is in $E^\star$, call it $f$. One checks $\langle \varphi, \xi\rangle = \langle \xi, f\rangle\ \forall \xi \in E^{\star\star}$, i.e., the canonical injection $E^\star \to E^{\star\star\star}$ is surjective.

**($E^\star$ reflexive ⇒ $E$ reflexive).** From the previous step, $E^{\star\star}$ is reflexive. $J(E)$ is a closed subspace of $E^{\star\star}$, hence reflexive (Proposition 3.20). Therefore $E$ is reflexive. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 3.22</span><span class="math-callout__name">(Bounded closed convex ⇒ weakly compact)</span></p>

If $E$ is reflexive and $K \subset E$ is bounded, closed, and convex, then $K$ is compact in $\sigma(E, E^\star)$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

$K$ is weakly closed (Theorem 3.7); $K \subset m B_E$ for some $m$, and $m B_E$ is weakly compact (Theorem 3.17). $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 3.23</span><span class="math-callout__name">(Existence of minimizers)</span></p>

Let $E$ be reflexive, $A \subset E$ nonempty closed convex, and $\varphi : A \to (-\infty, +\infty]$ convex l.s.c. with $\varphi \not\equiv +\infty$ and

$$
\lim_{\substack{x \in A \\ \|x\| \to \infty}} \varphi(x) = +\infty \quad \text{(no assumption if $A$ is bounded).} \tag{5}
$$

Then $\varphi$ achieves its minimum on $A$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Pick $a \in A$ with $\varphi(a) < +\infty$, and consider $\widetilde{A} = \lbrace x \in A\,;\ \varphi(x) \le \varphi(a)\rbrace$. By $(5)$, $\widetilde{A}$ is bounded; it is also closed and convex. By Corollary 3.22, $\widetilde{A}$ is weakly compact. Now $\varphi$ is also weakly l.s.c. (Corollary 3.9), and a weakly l.s.c. function on a weakly compact set attains its infimum (Property 5 of l.s.c. functions). $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reflexivity + convexity ⇒ existence)</span></p>

Corollary 3.23 is the *main reason* why **reflexive spaces** and **convex functions** are so important in the calculus of variations and in optimization. The classical recipe — *minimize a coercive convex l.s.c. functional on a closed convex set in a reflexive Banach space* — captures most existence proofs in modern PDE.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.24</span><span class="math-callout__name">(Biadjoint in reflexive spaces)</span></p>

Let $E, F$ be reflexive Banach spaces and $A : D(A) \subset E \to F$ densely defined and closed. Then $D(A^\star)$ is dense in $F^\star$ (so $A^{\star\star}$ is well defined as $D(A^{\star\star}) \subset E^{\star\star} \to F^{\star\star}$, viewed as $E \to F$), and

$$
\boxed{\;A^{\star\star} = A.\;}
$$

</div>

### 3.6 Separable Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Separable space)</span></p>

A metric space $E$ is **separable** if there exists a countable dense subset $D \subset E$.

</div>

Many important spaces are separable: finite-dimensional, $L^p$ and $\ell^p$ for $1 \le p < \infty$, $C(K)$ for $K$ a compact metric space. However, $L^\infty$ and $\ell^\infty$ are *not* separable.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.25</span><span class="math-callout__name">(Subsets of separable spaces are separable)</span></p>

If $E$ is separable and $F \subset E$, then $F$ is also separable.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $(u_n)$ be countable dense in $E$ and $(r_m)$ a sequence of positive numbers with $r_m \to 0$. Pick any $a_{m,n} \in B(u_n, r_m) \cap F$ whenever this set is nonempty. The countable family $(a_{m,n})$ is dense in $F$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.26</span><span class="math-callout__name">($E^\star$ separable ⇒ $E$ separable)</span></p>

Let $E$ be a Banach space such that $E^\star$ is separable. Then $E$ is separable.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Converse fails)</span></p>

The converse is *not* true. As we shall see in Chapter 4, $E = L^1$ is separable but $E^\star = L^\infty$ is *not* separable.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 3.26</summary>

Let $(f_n)_{n \ge 1}$ be countable and dense in $E^\star$. By definition $\|f_n\| = \sup_{\|x\| \le 1} \langle f_n, x\rangle$, so we can find $x_n \in E$ with $\|x_n\| = 1$ and $\langle f_n, x_n\rangle \ge \tfrac{1}{2}\|f_n\|$.

Let $L_0$ = $\mathbb{Q}$-vector space generated by $(x_n)$ (countable). Let $L$ = $\mathbb{R}$-vector space generated by $(x_n)$ (so $L_0 \subset L$ is dense). We claim $L$ is dense in $E$ — then $L_0$ is countable dense, proving separability.

By Corollary 1.8, it suffices to show: every $f \in E^\star$ vanishing on $L$ vanishes everywhere. Given $\varepsilon > 0$, pick $N$ with $\|f - f_N\| < \varepsilon$. Then

$$
\tfrac{1}{2}\|f_N\| \le \langle f_N, x_N\rangle = \langle f_N - f, x_N\rangle < \varepsilon
$$

(since $\langle f, x_N\rangle = 0$). Hence $\|f\| \le \|f - f_N\| + \|f_N\| < 3\varepsilon$, so $f = 0$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 3.27</span><span class="math-callout__name">(Reflexive separable: $E$ vs $E^\star$)</span></p>

For $E$ a Banach space:

$$
[E \text{ reflexive and separable}] \iff [E^\star \text{ reflexive and separable}].
$$

</div>

#### Metrizability of weak topologies on bounded sets

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.28</span><span class="math-callout__name">(Metrizability of $B_{E^\star}$ in $\sigma(E^\star, E)$)</span></p>

Let $E$ be a separable Banach space. Then $B_{E^\star}$ is metrizable in the weak-$\star$ topology $\sigma(E^\star, E)$.

Conversely, if $B_{E^\star}$ is metrizable in $\sigma(E^\star, E)$, then $E$ is separable.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.29</span><span class="math-callout__name">(Dual statement: $B_E$ in $\sigma(E, E^\star)$)</span></p>

Let $E$ be a Banach space such that $E^\star$ is separable. Then $B_E$ is metrizable in $\sigma(E, E^\star)$.

Conversely, if $B_E$ is metrizable in $\sigma(E, E^\star)$, then $E^\star$ is separable.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 3.28</summary>

Let $(x_n)_{n \ge 1}$ be a countable dense subset of $B_E$. For $f \in E^\star$ set

$$
[f] = \sum_{n=1}^\infty \frac{1}{2^n} \lvert \langle f, x_n\rangle\rvert.
$$

Then $[\cdot]$ is a norm on $E^\star$ with $[f] \le \|f\|$; let $d(f, g) = [f - g]$. We show that on $B_{E^\star}$, the topology induced by $d$ equals $\sigma(E^\star, E)$.

(a) Let $f_0 \in B_{E^\star}$, $V = \lbrace f \in B_{E^\star}\,;\ \lvert \langle f - f_0, y_i\rangle\rvert < \varepsilon\ \forall i \le k\rbrace$ a $\sigma(E^\star, E)$-neighborhood, with $\|y_i\| \le 1$. For each $i$ pick $n_i$ with $\|y_i - x_{n_i}\| < \varepsilon/4$. Choose $r > 0$ with $2^{n_i} r < \varepsilon/2$. Then $d(f, f_0) < r$ implies $\tfrac{1}{2^{n_i}}\lvert \langle f - f_0, x_{n_i}\rangle\rvert < r$, hence

$$
\lvert \langle f - f_0, y_i\rangle\rvert \le \lvert \langle f - f_0, y_i - x_{n_i}\rangle\rvert + \lvert \langle f - f_0, x_{n_i}\rangle\rvert < \varepsilon/2 + \varepsilon/2 = \varepsilon.
$$

(b) Let $f_0 \in B_{E^\star}$, $r > 0$. Pick $\varepsilon = r/2$, $k$ large enough that $1/2^{k-1} < r/2$, and $V = \lbrace f \in B_{E^\star}\,;\ \lvert \langle f - f_0, x_i\rangle\rvert < \varepsilon\ \forall i \le k\rbrace$. For $f \in V$,

$$
d(f, f_0) \le \sum_{n=1}^k \frac{\varepsilon}{2^n} + 2 \sum_{n=k+1}^\infty \frac{1}{2^n} < \varepsilon + \frac{1}{2^{k-1}} < r.
$$

The converse direction (Exercise 3.24): if $B_{E^\star}$ is metrizable, set $U_n = \lbrace f \in B_{E^\star}\,;\ d(f, 0) < 1/n\rbrace$ and pick weak-$\star$ neighborhoods $V_n \subset U_n$ of the form $\lbrace f\,;\ \lvert \langle f, x\rangle\rvert < \varepsilon_n\ \forall x \in \Phi_n\rbrace$ with $\Phi_n$ finite. Then $D = \bigcup_n \Phi_n$ is countable, and the span of $D$ is dense in $E$ (Corollary 1.8 applied: any $f \in E^\star$ vanishing on $D$ lies in every $V_n$, hence every $U_n$, so $f = 0$). $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(But not on all of $E$)</span></p>

In infinite-dimensional spaces the weak topology $\sigma(E, E^\star)$ (resp. weak-$\star$ topology $\sigma(E^\star, E)$) on *all* of $E$ (resp. $E^\star$) is *not* metrizable, even when $E$ (or $E^\star$) is separable. The norm $[\cdot]$ above induces $\sigma(E^\star, E)$ only on bounded sets.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 3.30</span><span class="math-callout__name">(Bounded sequences in $E^\star$ when $E$ is separable)</span></p>

Let $E$ be a separable Banach space and $(f_n)$ a bounded sequence in $E^\star$. Then there exists a subsequence $(f_{n_k})$ that converges in $\sigma(E^\star, E)$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

We may assume $\|f_n\| \le 1$. By Theorems 3.16 and 3.28, $B_{E^\star}$ is compact and metrizable in $\sigma(E^\star, E)$. The conclusion follows. $\square$

</details>
</div>

### 3.7 Uniformly Convex Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Uniformly convex Banach space)</span></p>

A Banach space $E$ is **uniformly convex** if

$$
\boxed{\;\forall \varepsilon > 0\ \exists \delta > 0:\ \big[x, y \in E,\ \|x\| \le 1,\ \|y\| \le 1,\ \|x - y\| > \varepsilon\big] \Longrightarrow \Big\| \frac{x + y}{2}\Big\| < 1 - \delta.\;}
$$

</div>

This is a *geometric* property of the unit ball: if we slide a stick of length $\varepsilon$ inside the unit ball, its midpoint stays within a ball of radius $(1 - \delta)$. In particular the unit *sphere* must be "round" — it cannot contain any line segment.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1</span><span class="math-callout__name">(Norms on $\mathbb{R}^2$)</span></p>

On $E = \mathbb{R}^2$:

* $\|x\|_2 = (\lvert x_1\rvert^2 + \lvert x_2\rvert^2)^{1/2}$ is uniformly convex (the unit ball is a *round* disk);
* $\|x\|_1 = \lvert x_1\rvert + \lvert x_2\rvert$ is *not* uniformly convex (the unit ball is a square — its sides contain line segments);
* $\|x\|_\infty = \max(\lvert x_1\rvert, \lvert x_2\rvert)$ is *not* uniformly convex (also a square).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2</span><span class="math-callout__name">($L^p$ and Hilbert spaces)</span></p>

$L^p$ spaces are uniformly convex for $1 < p < \infty$ (Chapters 4, 5). Hilbert spaces are uniformly convex.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.31</span><span class="math-callout__name">(Milman–Pettis)</span></p>

Every uniformly convex Banach space is **reflexive**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Geometric ⇒ topological)</span></p>

Uniform convexity is a *geometric* property of the *norm* — an equivalent norm need *not* be uniformly convex. On the other hand, reflexivity is a *topological* property: a reflexive space remains reflexive under any equivalent norm. It is striking that a *geometric* property of the norm forces a *topological* property of the space.

Uniform convexity is often the easiest tool to verify reflexivity in concrete examples; but it is *not* the ultimate tool — there exist reflexive spaces that admit *no* equivalent uniformly convex norm.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 3.31</summary>

Let $\xi \in E^{\star\star}$ with $\|\xi\| = 1$. We show $\xi \in J(B_E)$. Since $J(B_E)$ is closed in $E^{\star\star}$ for the strong topology (as $J$ is an isometry), it suffices to prove

$$
\forall \varepsilon > 0\ \exists x \in B_E,\ \|\xi - J(x)\| \le \varepsilon. \tag{7}
$$

Fix $\varepsilon > 0$, $\delta = $ modulus of uniform convexity. Choose $f \in E^\star$ with $\|f\| = 1$ and

$$
\langle \xi, f\rangle > 1 - \delta/2. \tag{8}
$$

Set $V = \lbrace \eta \in E^{\star\star}\,;\ \lvert \langle \eta - \xi, f\rangle\rvert < \delta/2\rbrace$, a neighborhood of $\xi$ in $\sigma(E^{\star\star}, E^\star)$. By Goldstine, $J(B_E)$ is dense in $B_{E^{\star\star}}$ for $\sigma(E^{\star\star}, E^\star)$, so $V \cap J(B_E) \neq \emptyset$: pick $x \in B_E$ with $J(x) \in V$. Claim: this $x$ works.

Suppose not: $\|\xi - J(x)\| > \varepsilon$, i.e., $\xi \in (J(x) + \varepsilon B_{E^{\star\star}})^c =: W$, also a neighborhood of $\xi$ in $\sigma(E^{\star\star}, E^\star)$ (since $B_{E^{\star\star}}$ is closed in $\sigma(E^{\star\star}, E^\star)$). Apply Goldstine again: $V \cap W \cap J(B_E) \neq \emptyset$, so there is $y \in B_E$ with $J(x), J(y) \in V$. Then

$$
\lvert \langle f, x\rangle - \langle \xi, f\rangle\rvert < \delta/2,\qquad \lvert \langle f, y\rangle - \langle \xi, f\rangle\rvert < \delta/2.
$$

Adding: $2\langle \xi, f\rangle < \langle f, x + y\rangle + \delta \le \|x + y\| + \delta$. Combined with $(8)$,

$$
\Big\| \frac{x + y}{2}\Big\| > 1 - \delta.
$$

By uniform convexity, $\|x - y\| \le \varepsilon$ — but $J(y) \in W$ means $\|J(x) - J(y)\| > \varepsilon$, i.e., $\|x - y\| > \varepsilon$. Contradiction. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.32</span><span class="math-callout__name">(Weak + norm convergence ⇒ strong, in uniformly convex)</span></p>

Assume $E$ is uniformly convex. Let $(x_n)$ be a sequence with

$$
x_n \rightharpoonup x \text{ weakly} \quad \text{and} \quad \limsup \|x_n\| \le \|x\|.
$$

Then $x_n \to x$ **strongly**.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Assume $x \neq 0$ (else trivial). Set $\lambda_n = \max(\|x_n\|, \|x\|)$, $y_n = \lambda_n^{-1} x_n$, $y = \|x\|^{-1} x$. Then $\lambda_n \to \|x\|$, $y_n \rightharpoonup y$, $\|y\| = 1$, $\|y_n\| \le 1$. By weak l.s.c. of the norm (Proposition 3.5), $\|(y_n + y)/2\| \to 1$. Then *uniform convexity* (in contrapositive) gives $\|y_n - y\| \to 0$; hence $x_n \to x$ strongly. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why this matters)</span></p>

Proposition 3.32 is the standard *upgrade lemma*: if you've already established weak convergence of minimizers and matched the norms, you get strong convergence for free. This pattern is ubiquitous in PDE — particularly in $L^p$ spaces with $1 < p < \infty$, all of which are uniformly convex.

</div>

### Comments on Chapter 3

1. **Locally convex topologies.** The topologies $\sigma(E, E^\star)$, $\sigma(E^\star, E)$, etc., are *locally convex* topologies. As such, they enjoy all the properties of locally convex spaces — for example, Hahn–Banach (geometric form), Krein–Milman, etc., still hold. See Bourbaki, Knapp, Problem 9.

2. **A further compactness theorem.** Another remarkable property of the weak-$\star$ topology:

   <div class="math-callout math-callout--theorem" markdown="1">
     <p class="math-callout__title"><span class="math-callout__label">Theorem 3.33</span><span class="math-callout__name">(Banach–Dieudonné–Krein–Šmulian)</span></p>

   Let $E$ be a Banach space and $C \subset E^\star$ convex. Assume that for every $n$ the set $C \cap (n B_{E^\star})$ is closed in $\sigma(E^\star, E)$. Then $C$ is closed in $\sigma(E^\star, E)$.

   </div>

   See Bourbaki, Larsen, Holmes, Dunford–Schwartz, Schaefer, Problem 11. These references contain much material on the Eberlein–Šmulian theorem (Theorem 3.19).

3. **Vector spaces in duality.** The theory of *vector spaces in duality* — extending the duality $\langle E, E^\star\rangle$ — was very popular in the late 1940s/early 1950s, especially in connection with the theory of distributions. Two vector spaces $X, Y$ are *in duality* if there is a bilinear form $\langle\cdot, \cdot\rangle : X \times Y \to \mathbb{R}$ that *separates points*. Many topologies may be defined on $X$ (or $Y$): the weak topology $\sigma(X, Y)$, **Mackey's topology** $\tau(X, Y)$, the *strong topology* $\beta(X, Y)$. These are of interest in spaces *not* of Banach type. References: Bourbaki, Schaefer, Köthe, Treves, Kelley–Namioka, Edwards, Horváth.

4. **Geometry of Banach spaces.** Separability, reflexivity, and uniform convexity are closely related to *differentiability* properties of $x \mapsto \|x\|$ (Diestel, Beauzamy, Problem 13). The existence of equivalent norms with nice geometric properties has been extensively studied — for example, when does a Banach space admit an equivalent *uniformly convex* norm? (such spaces are called **superreflexive**; Diestel, Beauzamy.) The geometry of Banach spaces has flourished since the early sixties as an active field associated with: Dvoretzky, Grothendieck, R. C. James, Lindenstrauss, Milman, Tzafriri, Pełczyński, Enflo, Schwartz, Pisier, Maurey, Beauzamy, Johnson, Rosenthal, Bourgain, Preiss, Talagrand, Tomczak-Gowers, and many others. Standard references: Beauzamy, Diestel, Lindenstrauss–Tzafriri, Schwartz, Deville–Godefroy–Zizler, Benyamini–Lindenstrauss, Albiac–Kalton, Pietsch.

## Chapter 4: $L^p$ Spaces

After three chapters of abstract Banach-space theory, we turn to the most ubiquitous family of concrete Banach spaces in analysis: the **Lebesgue spaces** $L^p(\Omega)$. The chapter has four threads:

* **Hölder, Minkowski, completeness** — making $L^p$ a Banach space and giving us the basic algebra of $p$-norms;
* **Reflexivity, separability, duals** — running the abstract machinery of Chapter 3 on $L^p$ and tabulating the answers $1 < p < \infty$ vs. $p = 1$ vs. $p = \infty$;
* **Convolution and regularization** — the *mollifier* trick, which approximates $L^p$ functions by $C^\infty_c$ functions and is the unsung workhorse of distribution theory and PDE;
* **Compactness criteria** — Kolmogorov–M. Riesz–Fréchet, the $L^p$ analogue of Ascoli–Arzelà.

Throughout, $(\Omega, \mathcal{M}, \mu)$ denotes a measure space:

* $\mathcal{M}$ is a $\sigma$-algebra on $\Omega$ (closed under complement and countable union, contains $\emptyset$);
* $\mu : \mathcal{M} \to [0, \infty]$ is a measure ($\mu(\emptyset) = 0$, countably additive on disjoint families);
* $\Omega$ is **$\sigma$-finite**: $\Omega = \bigcup_n \Omega_n$ with $\mu(\Omega_n) < \infty$.

We write $\lvert A\rvert$ for $\mu(A)$ and identify two functions that coincide a.e. The space $L^1 = L^1(\Omega, \mu)$ consists of integrable real-valued functions; $\int f$ stands for $\int_\Omega f\,d\mu$.

### 4.1 Some Results about Integration That Everyone Must Know

We collect the basic convergence theorems and Fubini–Tonelli for reference.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.1</span><span class="math-callout__name">(Monotone convergence, Beppo Levi)</span></p>

Let $(f_n)$ be a sequence in $L^1$ with

(a) $f_1 \le f_2 \le \cdots \le f_n \le f_{n+1} \le \cdots$ a.e. on $\Omega$,

(b) $\sup_n \int f_n < \infty$.

Then $f_n(x)$ converges a.e. to a finite limit $f(x)$, $f \in L^1$, and $\|f_n - f\|_1 \to 0$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.2</span><span class="math-callout__name">(Dominated convergence, Lebesgue)</span></p>

Let $(f_n)$ be a sequence in $L^1$ with

(a) $f_n(x) \to f(x)$ a.e. on $\Omega$,

(b) there exists $g \in L^1$ such that $\lvert f_n(x)\rvert \le g(x)$ a.e. on $\Omega$ for all $n$.

Then $f \in L^1$ and $\|f_n - f\|_1 \to 0$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 4.1</span><span class="math-callout__name">(Fatou)</span></p>

Let $(f_n)$ be a sequence in $L^1$ with

(a) $f_n \ge 0$ a.e. for every $n$,

(b) $\sup_n \int f_n < \infty$.

For a.e. $x \in \Omega$ set $f(x) = \liminf_n f_n(x)$. Then $f \in L^1$ and

$$
\int f \le \liminf_{n \to \infty} \int f_n.
$$

</div>

A basic example is $\Omega = \mathbb{R}^N$ with Lebesgue $\mathcal{M}$ and Lebesgue measure $\mu$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">($C_c(\mathbb{R}^N)$)</span></p>

$C_c(\mathbb{R}^N)$ is the space of continuous functions on $\mathbb{R}^N$ with **compact support**:

$$
C_c(\mathbb{R}^N) = \lbrace f \in C(\mathbb{R}^N)\,;\ f(x) = 0\ \forall x \in \mathbb{R}^N \setminus K \text{ for some compact } K\rbrace.
$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.3</span><span class="math-callout__name">(Density of $C_c$ in $L^1$)</span></p>

The space $C_c(\mathbb{R}^N)$ is dense in $L^1(\mathbb{R}^N)$: for every $f \in L^1(\mathbb{R}^N)$ and $\varepsilon > 0$ there is $f_1 \in C_c(\mathbb{R}^N)$ with $\|f - f_1\|_1 \le \varepsilon$.

</div>

Let $(\Omega_1, \mathcal{M}_1, \mu_1)$ and $(\Omega_2, \mathcal{M}_2, \mu_2)$ be $\sigma$-finite. The product measure space $(\Omega, \mathcal{M}, \mu)$ on $\Omega = \Omega_1 \times \Omega_2$ is the standard one.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.4</span><span class="math-callout__name">(Tonelli)</span></p>

Let $F : \Omega_1 \times \Omega_2 \to \mathbb{R}$ be measurable and assume

(a) $\int_{\Omega_2} \lvert F(x, y)\rvert\, d\mu_2 < \infty$ for a.e. $x \in \Omega_1$,

(b) $\int_{\Omega_1} d\mu_1 \int_{\Omega_2} \lvert F(x, y)\rvert\, d\mu_2 < \infty$.

Then $F \in L^1(\Omega_1 \times \Omega_2)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.5</span><span class="math-callout__name">(Fubini)</span></p>

Assume $F \in L^1(\Omega_1 \times \Omega_2)$. Then for a.e. $x \in \Omega_1$, $F(x, \cdot) \in L^1(\Omega_2)$ and $\int_{\Omega_2} F(x, y)\, d\mu_2 \in L^1(\Omega_1)$; symmetrically in $y$. Moreover

$$
\int_{\Omega_1} d\mu_1 \int_{\Omega_2} F(x, y)\,d\mu_2 = \int_{\Omega_2} d\mu_2 \int_{\Omega_1} F(x, y)\,d\mu_1 = \iint_{\Omega_1 \times \Omega_2} F(x, y)\,d\mu_1\,d\mu_2.
$$

</div>

### 4.2 Definition and Elementary Properties of $L^p$ Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($L^p$, $1 \le p \le \infty$)</span></p>

For $1 \le p < \infty$:

$$
L^p(\Omega) = \lbrace f : \Omega \to \mathbb{R}\,;\ f \text{ measurable and } \lvert f\rvert^p \in L^1(\Omega)\rbrace,
$$

$$
\|f\|_{L^p} = \|f\|_p = \Big[\int_\Omega \lvert f(x)\rvert^p\,d\mu\Big]^{1/p}.
$$

For $p = \infty$:

$$
L^\infty(\Omega) = \lbrace f : \Omega \to \mathbb{R}\,;\ f \text{ measurable and } \exists C \ge 0,\ \lvert f(x)\rvert \le C \text{ a.e.}\rbrace,
$$

$$
\|f\|_{L^\infty} = \|f\|_\infty = \inf\lbrace C\,;\ \lvert f(x)\rvert \le C \text{ a.e.}\rbrace.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Pointwise control by $\|f\|_\infty$)</span></p>

If $f \in L^\infty$, then $\lvert f(x)\rvert \le \|f\|_\infty$ a.e. (the inf in the definition of $\|f\|_\infty$ *is* attained — pick a sequence $C_n \to \|f\|_\infty$ with $\lvert f\rvert \le C_n$ outside a null set $E_n$, and put $E = \bigcup_n E_n$).

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Conjugate exponent)</span></p>

For $1 \le p \le \infty$ the **conjugate exponent** $p'$ is defined by

$$
\boxed{\;\frac{1}{p} + \frac{1}{p'} = 1.\;}
$$

(So $p = 1, p' = \infty$ and $p = \infty, p' = 1$.)

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.6</span><span class="math-callout__name">(Hölder's inequality)</span></p>

Assume $f \in L^p$ and $g \in L^{p'}$ with $1 \le p \le \infty$. Then $fg \in L^1$ and

$$
\boxed{\;\int \lvert fg\rvert \le \|f\|_p \|g\|_{p'}.\;} \tag{1}
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

The cases $p = 1, \infty$ are obvious. For $1 < p < \infty$ recall **Young's inequality**

$$
ab \le \frac{1}{p} a^p + \frac{1}{p'} b^{p'}\quad \forall a, b \ge 0, \tag{2}
$$

a consequence of concavity of $\log$. Applied pointwise to $\lvert f(x)\rvert$ and $\lvert g(x)\rvert$,

$$
\lvert f(x) g(x)\rvert \le \frac{1}{p} \lvert f(x)\rvert^p + \frac{1}{p'} \lvert g(x)\rvert^{p'}\quad \text{a.e.}
$$

Integrate to get $\int \lvert fg\rvert \le \tfrac{1}{p}\|f\|_p^p + \tfrac{1}{p'}\|g\|_{p'}^{p'}$. Replace $f$ by $\lambda f$ ($\lambda > 0$):

$$
\int \lvert fg\rvert \le \frac{\lambda^{p-1}}{p}\|f\|_p^p + \frac{1}{\lambda p'}\|g\|_{p'}^{p'}.
$$

Optimize over $\lambda$: the choice $\lambda = \|f\|_p^{-1}\|g\|_{p'}^{p'/p}$ minimizes the RHS, giving $(1)$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Generalized Hölder & interpolation)</span></p>

A useful extension: if $f_i \in L^{p_i}$ ($1 \le i \le k$) with $\tfrac{1}{p} = \tfrac{1}{p_1} + \cdots + \tfrac{1}{p_k} \le 1$, then $f_1 \cdots f_k \in L^p$ and

$$
\|f_1 \cdots f_k\|_p \le \|f_1\|_{p_1} \cdots \|f_k\|_{p_k}.
$$

In particular, if $f \in L^p \cap L^q$ with $1 \le p \le q \le \infty$, then $f \in L^r$ for all $r \in [p, q]$, and we have the **interpolation inequality**

$$
\boxed{\;\|f\|_r \le \|f\|_p^\alpha \|f\|_q^{1-\alpha},\quad \frac{1}{r} = \frac{\alpha}{p} + \frac{1-\alpha}{q},\ 0 \le \alpha \le 1.\;}
$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.7</span><span class="math-callout__name">($L^p$ is a normed space)</span></p>

$L^p$ is a vector space and $\|\cdot\|_p$ is a norm for any $1 \le p \le \infty$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof (Minkowski for $1 < p < \infty$)</summary>

For $f, g \in L^p$, $\lvert f + g\rvert^p \le 2^p(\lvert f\rvert^p + \lvert g\rvert^p)$ shows $f + g \in L^p$. Then

$$
\|f + g\|_p^p = \int \lvert f + g\rvert^p \le \int \lvert f + g\rvert^{p-1}\lvert f\rvert + \int \lvert f + g\rvert^{p-1}\lvert g\rvert.
$$

Since $\lvert f + g\rvert^{p-1} \in L^{p'}$ (because $(p-1)p' = p$), Hölder gives

$$
\|f + g\|_p^p \le \|f + g\|_p^{p-1} (\|f\|_p + \|g\|_p),
$$

i.e., $\|f + g\|_p \le \|f\|_p + \|g\|_p$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.8</span><span class="math-callout__name">(Fischer–Riesz: completeness)</span></p>

$L^p$ is a Banach space for every $1 \le p \le \infty$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

**Case $p = \infty$.** A Cauchy sequence $(f_n)$ in $L^\infty$: choose $N_k$ with $\|f_m - f_n\|_\infty \le 1/k$ for $m, n \ge N_k$, and a null set $E_k$ on whose complement the inequality holds pointwise. On $\Omega \setminus \bigcup_k E_k$, $(f_n(x))$ is Cauchy in $\mathbb{R}$, hence converges to $f(x)$. Pass to the limit in the pointwise inequality to get $\|f - f_n\|_\infty \le 1/k$ for $n \ge N_k$.

**Case $1 \le p < \infty$.** Cauchy in $L^p$: extract a subsequence $f_{n_k}$ (write $f_k$) with $\|f_{k+1} - f_k\|_p \le 2^{-k}$. Set $g_n(x) = \sum_{k=1}^n \lvert f_{k+1}(x) - f_k(x)\rvert$, so $\|g_n\|_p \le 1$. By monotone convergence $g_n \to g$ a.e. with $g \in L^p$. The series $f_k$ is Cauchy in $\mathbb{R}$ a.e., converging to $f$, with $\lvert f - f_n\rvert \le g - g_{n-1} \to 0$ a.e.; dominated convergence gives $\|f - f_n\|_p \to 0$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.9</span><span class="math-callout__name">(Convergence in $L^p$ ⇒ subsequence converges a.e.)</span></p>

If $f_n \to f$ in $L^p$ ($1 \le p \le \infty$), then there exist a subsequence $(f_{n_k})$ and $h \in L^p$ such that

(a) $f_{n_k}(x) \to f(x)$ a.e.,

(b) $\lvert f_{n_k}(x)\rvert \le h(x)$ a.e. for every $k$.

</div>

### 4.3 Reflexivity. Separability. Dual of $L^p$

We treat three regimes separately: (A) $1 < p < \infty$, (B) $p = 1$, (C) $p = \infty$.

#### A. Study of $L^p$ for $1 < p < \infty$

This is the most "favorable" case: $L^p$ is reflexive, separable, and $(L^p)^\star = L^{p'}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.10</span><span class="math-callout__name">($L^p$ is reflexive for $1 < p < \infty$)</span></p>

$L^p$ is reflexive for every $1 < p < \infty$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof (sketch in three steps)</summary>

**Step 1. Clarkson's first inequality.** For $2 \le p < \infty$,

$$
\Big\|\frac{f + g}{2}\Big\|_p^p + \Big\|\frac{f - g}{2}\Big\|_p^p \le \tfrac{1}{2}(\|f\|_p^p + \|g\|_p^p)\quad \forall f, g \in L^p. \tag{8}
$$

It suffices to verify the pointwise inequality $\lvert\tfrac{a+b}{2}\rvert^p + \lvert\tfrac{a-b}{2}\rvert^p \le \tfrac{1}{2}(\lvert a\rvert^p + \lvert b\rvert^p)$, which follows from convexity of $x \mapsto \lvert x\rvert^{p/2}$ when $p \ge 2$ together with $\alpha^p + \beta^p \le (\alpha^2 + \beta^2)^{p/2}$.

**Step 2. $L^p$ is uniformly convex for $p \ge 2$, hence reflexive (Milman–Pettis, Theorem 3.31).** From $(8)$, if $\|f\|_p, \|g\|_p \le 1$ and $\|f - g\|_p > \varepsilon$, then $\|(f+g)/2\|_p < (1 - (\varepsilon/2)^p)^{1/p} = 1 - \delta$.

**Step 3. $L^p$ is reflexive for $1 < p \le 2$.** Define $T : L^p \to (L^{p'})^\star$ by $\langle Tu, f\rangle = \int uf$. Hölder gives $\|Tu\|_{(L^{p'})^\star} \le \|u\|_p$. Conversely, plug in $f_0 = \lvert u\rvert^{p-2} u$ to get $\|Tu\|_{(L^{p'})^\star} \ge \|u\|_p$. So $T$ is an isometry. Since $L^{p'}$ is reflexive (by Step 2, since $p' \ge 2$), $(L^{p'})^\star$ is reflexive (Corollary 3.21), hence its closed subspace $T(L^p)$ is reflexive (Proposition 3.20), hence $L^p$ is reflexive. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Clarkson's second inequality)</span></p>

In fact, $L^p$ is also uniformly convex for $1 < p \le 2$. This is a consequence of **Clarkson's second inequality** (for $1 < p \le 2$):

$$
\Big\|\frac{f + g}{2}\Big\|_p^{p'} + \Big\|\frac{f - g}{2}\Big\|_p^{p'} \le \Big(\tfrac{1}{2}\|f\|_p^p + \tfrac{1}{2}\|g\|_p^p\Big)^{1/(p-1)}\quad \forall f, g \in L^p.
$$

This is trickier to prove than the first; see Hewitt–Stromberg, Morawetz, or Diestel.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.11</span><span class="math-callout__name">(Riesz representation: $(L^p)^\star = L^{p'}$ for $1 < p < \infty$)</span></p>

Let $1 < p < \infty$ and $\phi \in (L^p)^\star$. There exists a unique $u \in L^{p'}$ such that

$$
\langle \phi, f\rangle = \int u f\quad \forall f \in L^p,
$$

with $\|u\|_{p'} = \|\phi\|_{(L^p)^\star}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Importance and identification)</span></p>

Theorem 4.11 says every continuous linear functional on $L^p$ ($1 < p < \infty$) is concretely an integral. We systematically identify

$$
\boxed{\;(L^p)^\star = L^{p'}.\;}
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

The map $T : L^{p'} \to (L^p)^\star$, $\langle Tu, f\rangle = \int uf$, is an isometry (Step 3 of Theorem 4.10), so $E = T(L^{p'})$ is closed in $(L^p)^\star$. To show $E = (L^p)^\star$ it suffices to show $E$ is *dense*: any $h \in (L^p)^{\star\star}$ vanishing on $E$ satisfies $\langle h, Tu\rangle = 0\ \forall u \in L^{p'}$. By reflexivity of $L^p$, $h \in L^p$, so $\int uh = 0\ \forall u \in L^{p'}$. Choosing $u = \lvert h\rvert^{p-2} h$ gives $\int \lvert h\rvert^p = 0$, hence $h = 0$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.12</span><span class="math-callout__name">($C_c$ dense in $L^p$, $1 \le p < \infty$)</span></p>

$C_c(\mathbb{R}^N)$ is dense in $L^p(\mathbb{R}^N)$ for every $1 \le p < \infty$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Two reductions. **(a)** Given $f \in L^p(\mathbb{R}^N)$ and $\varepsilon > 0$, find $g \in L^\infty \cap L^p$ with compact support and $\|f - g\|_p < \varepsilon$: take $f_n = \chi_{B(0,n)} T_n f$ where $T_n$ is the truncation $T_n(r) = r$ if $\lvert r\rvert \le n$, else $T_n(r) = nr/\lvert r\rvert$. Then $f_n \to f$ in $L^p$ by dominated convergence.

**(b)** Approximate such $g$ by $C_c$. By Theorem 4.3 there is $g_1 \in C_c(\mathbb{R}^N)$ with $\|g - g_1\|_1 < \delta$. We may assume $\|g_1\|_\infty \le \|g\|_\infty$ (otherwise truncate). Then by interpolation,

$$
\|g - g_1\|_p \le \|g - g_1\|_1^{1/p} \|g - g_1\|_\infty^{1 - 1/p} \le \delta^{1/p} (2\|g\|_\infty)^{1 - 1/p}.
$$

Choose $\delta$ small enough. $\square$

</details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Separable measure space)</span></p>

The measure space $\Omega$ is **separable** if there is a countable family $(E_n) \subset \mathcal{M}$ such that the $\sigma$-algebra generated by $(E_n)$ coincides with $\mathcal{M}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($\mathbb{R}^N$ is separable)</span></p>

$\mathbb{R}^N$ with Lebesgue measure is separable: choose $(E_n)$ to be a countable basis of open sets. More generally, any separable metric space with the Borel $\sigma$-algebra is a separable measure space.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.13</span><span class="math-callout__name">($L^p$ is separable for $1 \le p < \infty$, on separable spaces)</span></p>

If $\Omega$ is a separable measure space, then $L^p(\Omega)$ is separable for every $1 \le p < \infty$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof (case $\Omega = \mathbb{R}^N$)</summary>

Let $\mathcal{R}$ be the countable family of rectangles $R = \prod_{k=1}^N (a_k, b_k)$ with $a_k, b_k \in \mathbb{Q}$. Let $\mathcal{E}$ be the $\mathbb{Q}$-vector space generated by $\lbrace \chi_R\,;\ R \in \mathcal{R}\rbrace$ (countable). Given $f \in L^p$ and $\varepsilon > 0$, by Theorem 4.12 pick $f_1 \in C_c(\mathbb{R}^N)$ with $\|f - f_1\|_p < \varepsilon$. Let $R \in \mathcal{R}$ be a large cube containing $\mathrm{supp}\,f_1$. By uniform continuity of $f_1$, on a fine partition of $R$ into small rational rectangles the oscillation of $f_1$ is $< \delta$; the corresponding step function $f_2 \in \mathcal{E}$ satisfies $\|f_1 - f_2\|_\infty < \delta$ and $f_2 = 0$ outside $R$, so $\|f_1 - f_2\|_p < \delta\lvert R\rvert^{1/p}$. Choose $\delta$ small enough. $\square$

</details>
</div>

#### B. Study of $L^1$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.14</span><span class="math-callout__name">(Riesz representation: $(L^1)^\star = L^\infty$)</span></p>

Let $\phi \in (L^1)^\star$. There exists a unique $u \in L^\infty$ such that

$$
\langle \phi, f\rangle = \int u f\quad \forall f \in L^1,
$$

with $\|u\|_\infty = \|\phi\|_{(L^1)^\star}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Identification)</span></p>

Every continuous linear functional on $L^1$ is an integral against a bounded measurable function. We identify

$$
\boxed{\;(L^1)^\star = L^\infty.\;}
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof (sketch via $L^2$)</summary>

Let $\Omega = \bigcup_n \Omega_n$ with $\lvert \Omega_n\rvert < \infty$. Construct $\theta \in L^2(\Omega)$ with $\theta(x) \ge \varepsilon_n > 0$ on $\Omega_n$ (define $\theta = \alpha_n$ on $\Omega_n \setminus \Omega_{n-1}$ and adjust $\alpha_n \in (0, \infty)$ so that $\theta \in L^2$).

The map $L^2(\Omega) \to \mathbb{R}$, $f \mapsto \langle \phi, \theta f\rangle$, is a continuous linear functional on $L^2$. By Theorem 4.11 (with $p = 2$) there is $v \in L^2$ with $\langle \phi, \theta f\rangle = \int v f\ \forall f \in L^2$. Set $u = v/\theta$; then $u\chi_n \in L^2$, $\langle \phi, \chi_n g\rangle = \int u \chi_n g\ \forall g \in L^\infty$, and a standard check on $A = \lbrace \lvert u\rvert > C\rbrace$ for $C > \|\phi\|$ shows $u \in L^\infty$ with $\|u\|_\infty \le \|\phi\|$. Pass to the limit to extend to all of $L^1$ via truncation. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">($L^1$ is *not* reflexive)</span></p>

$L^1(\Omega)$ is *never* reflexive — except in the trivial case where $\Omega$ consists of a finite number of atoms (so $L^1$ is finite-dimensional). Two case sketches:

* **(i) Diffuse part exists.** Choose nested $\omega_n$ with $0 < \mu(\omega_n) \to 0$. The unit-norm sequence $u_n = \chi_{\omega_n}/\|\chi_{\omega_n}\|_1$ has, by Theorem 3.18, a weakly convergent subsequence $u_n \rightharpoonup u$ in $\sigma(L^1, L^\infty)$. Test against $\chi_{\omega_j}$ ($j$ fixed, $n > j$): $\int u_n \chi_{\omega_j} = 1$, so $\int u\chi_{\omega_j} = 1\ \forall j$; but $\chi_{\omega_j} \to 0$ in $L^\infty$ pointwise/dominated, so $\int u\chi_{\omega_j} \to 0$ — contradiction.

* **(ii) Purely atomic, infinite atoms.** Then $L^1 \cong \ell^1$. Test the canonical basis $e_n = (0, \ldots, 1, 0, \ldots)$: if $\ell^1$ were reflexive, some subsequence $e_{n_k} \rightharpoonup x \in \ell^1$. Test against $\varphi_j = (0, \ldots, 0, 1, 1, \ldots)$ (zeros in positions $< j$): $\langle \varphi_j, e_{n_k}\rangle = 1$ for $n_k \ge j$, hence $\langle \varphi_j, x\rangle = 1\ \forall j$; but $\langle \varphi_j, x\rangle \to 0$ since $x \in \ell^1$ — contradiction.

</div>

#### C. Study of $L^\infty$

We already know $L^\infty = (L^1)^\star$ (Theorem 4.14). Being a dual space, $L^\infty$ enjoys some nice properties:

* The closed unit ball $B_{L^\infty}$ is **compact** in the weak-$\star$ topology $\sigma(L^\infty, L^1)$ (Banach–Alaoglu, Theorem 3.16).
* If $\Omega \subset \mathbb{R}^N$ is measurable, $L^1(\Omega)$ is separable, hence by Corollary 3.30 + Theorem 4.13, every bounded sequence in $L^\infty(\Omega)$ has a weak-$\star$ convergent subsequence.

However $L^\infty$ is **not reflexive** (Corollary 3.21 + Remark above on $L^1$): $(L^\infty)^\star$ contains $L^1$ strictly. Concretely, the functional $\phi(f) = f(0)$ on $C_c(\mathbb{R}^N)$ extends by Hahn–Banach to $\phi \in L^\infty(\mathbb{R}^N)^\star$, but no $u \in L^1$ represents it (see the table below).

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What does $(L^\infty)^\star$ look like?)</span></p>

The dual of $L^\infty$ does *not* coincide with $L^1$. To describe it concretely, view $L^\infty(\Omega; \mathbb{C})$ as a commutative $C^\star$-algebra. By **Gelfand's theorem**, $L^\infty(\Omega; \mathbb{C}) \cong C(K; \mathbb{C})$ for some compact Hausdorff $K$ (the *spectrum* of the algebra). Then $(L^\infty(\Omega; \mathbb{C}))^\star$ may be identified with the space of complex-valued **Radon measures** on $K$. See Comment 3, W. Rudin, K. Yosida.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">($L^\infty$ is not separable)</span></p>

$L^\infty(\Omega)$ is **not separable** (except when $\Omega$ has finitely many atoms). Tool:

* **Lemma 4.2.** Let $E$ be a Banach space. If there exists an uncountable family $(O_i)_{i \in I}$ of pairwise disjoint nonempty open sets, then $E$ is *not* separable. (A countable dense set must hit each $O_i$, contradiction.)

In $L^\infty(\Omega)$ apply this to $O_i = \lbrace f\,;\ \|f - \chi_{\omega_i}\|_\infty < 1/2\rbrace$ for an uncountable family $(\omega_i)$ with $\mu(\omega_i \,\Delta\, \omega_j) > 0$ for $i \neq j$. Existence of $(\omega_i)$ comes from either an open ball decomposition (in $\mathbb{R}^N$) or, on atomic $\Omega = \bigcup_n \lbrace a_n\rbrace$, from $\omega_A = \bigcup_{n \in A} a_n$ as $A$ ranges over the uncountable family of subsets of $\mathbb{N}$.

</div>

#### Summary table

The following table summarizes the main properties of $L^p(\Omega)$ for $\Omega \subset \mathbb{R}^N$ measurable:

| | Reflexive | Separable | Dual space |
| --- | --- | --- | --- |
| $L^p,\ 1 < p < \infty$ | YES | YES | $L^{p'}$ |
| $L^1$ | NO | YES | $L^\infty$ |
| $L^\infty$ | NO | NO | strictly bigger than $L^1$ |

### 4.4 Convolution and Regularization

#### Convolution as a bounded bilinear map

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.15</span><span class="math-callout__name">(Young: convolution $L^1 \star L^p \to L^p$)</span></p>

Let $f \in L^1(\mathbb{R}^N)$ and $g \in L^p(\mathbb{R}^N)$ with $1 \le p \le \infty$. Then for a.e. $x \in \mathbb{R}^N$ the function $y \mapsto f(x - y) g(y)$ is integrable, and we set

$$
\boxed{\;(f \star g)(x) = \int_{\mathbb{R}^N} f(x - y) g(y)\,dy.\;}
$$

In addition $f \star g \in L^p(\mathbb{R}^N)$ and

$$
\boxed{\;\|f \star g\|_p \le \|f\|_1 \|g\|_p.\;}
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

The case $p = \infty$ is obvious. For $p = 1$: $F(x, y) = f(x-y)g(y)$. By translation invariance, $\int_{\mathbb{R}^N} \lvert F(x, y)\rvert\,dx = \lvert g(y)\rvert\|f\|_1$, hence $\int\int \lvert F\rvert < \infty$. Tonelli–Fubini give the conclusion.

For $1 < p < \infty$: from $p = 1$ case, $y \mapsto \lvert f(x-y)\rvert\lvert g(y)\rvert^p$ is integrable for a.e. $x$, i.e., $\lvert f(x-y)\rvert^{1/p}\lvert g(y)\rvert \in L^p_y$. Since $\lvert f(x-y)\rvert^{1/p'} \in L^{p'}_y$ (with $\|\cdot\|_{p'} = \|f\|_1^{1/p'}$), Hölder gives

$$
\int \lvert f(x-y)g(y)\rvert\,dy \le \|f\|_1^{1/p'}\Big(\int \lvert f(x-y)\rvert\lvert g(y)\rvert^p\,dy\Big)^{1/p}.
$$

Hence $\lvert(f\star g)(x)\rvert^p \le \|f\|_1^{p/p'}(\lvert f\rvert \star \lvert g\rvert^p)(x)$. Integrate using $p = 1$ case:

$$
\|f\star g\|_p^p \le \|f\|_1^{p/p'}\|f\|_1\|g\|_p^p = \|f\|_1^p\|g\|_p^p.\quad \square
$$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.16</span><span class="math-callout__name">(Adjointness via reflection)</span></p>

Given $f$ on $\mathbb{R}^N$, set $\check{f}(x) = f(-x)$. For $f \in L^1, g \in L^p, h \in L^{p'}$,

$$
\int_{\mathbb{R}^N} (f \star g) h = \int_{\mathbb{R}^N} g\, (\check{f} \star h).
$$

(Direct from Fubini.)

</div>

#### Support of a measurable function

The notion of support of a function $f$ — the complement of the largest open set on which $f$ vanishes — is *not* well-suited to $L^p$, where functions are equivalence classes a.e. We need an *intrinsic* definition.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.17</span><span class="math-callout__name">(Definition of $\mathrm{supp}\,f$)</span></p>

Let $f : \mathbb{R}^N \to \mathbb{R}$. Consider the family $(\omega_i)_{i \in I}$ of all open sets in $\mathbb{R}^N$ such that $f = 0$ a.e. on $\omega_i$. Set $\omega = \bigcup_i \omega_i$. Then $f = 0$ a.e. on $\omega$.

By definition, **$\mathrm{supp}\,f$ is the complement of $\omega$** in $\mathbb{R}^N$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Compatibility)</span></p>

(a) If $f_1 = f_2$ a.e., then $\mathrm{supp}\,f_1 = \mathrm{supp}\,f_2$ — so it makes sense to talk about $\mathrm{supp}\,f$ for $f \in L^p$.

(b) For continuous $f$ the new definition agrees with the usual one.

The proof of Proposition 4.17 reduces an arbitrary union of open sets to a *countable* one, using that any open cover of an open subset of $\mathbb{R}^N$ admits a countable subcover (Lindelöf).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.18</span><span class="math-callout__name">(Support of a convolution)</span></p>

For $f \in L^1, g \in L^p$,

$$
\boxed{\;\mathrm{supp}(f \star g) \subset \overline{\mathrm{supp}\,f + \mathrm{supp}\,g}.\;}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(When does $f\star g$ have compact support?)</span></p>

If *both* $f$ and $g$ have compact support, then $f \star g$ also has compact support. If only one of them does, $f\star g$ generally has *non-compact* support.

</div>

#### Convolution and differentiation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($L^p_{\mathrm{loc}}$)</span></p>

For $\Omega \subset \mathbb{R}^N$ open, $f : \Omega \to \mathbb{R}$ belongs to $L^p_{\mathrm{loc}}(\Omega)$ if $f\chi_K \in L^p(\Omega)$ for every compact $K \subset \Omega$. (Note: $f \in L^p_{\mathrm{loc}}$ implies $f \in L^1_{\mathrm{loc}}$.)

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.19</span><span class="math-callout__name">(Convolution with $C_c$ is continuous)</span></p>

If $f \in C_c(\mathbb{R}^N)$ and $g \in L^1_{\mathrm{loc}}(\mathbb{R}^N)$, then $(f \star g)(x)$ is well defined for *every* $x$, and $f \star g \in C(\mathbb{R}^N)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">($C^k$ classes)</span></p>

For $\Omega$ open:

* $C(\Omega)$ — continuous functions;
* $C^k(\Omega)$ — $k$-times continuously differentiable functions ($k \ge 1$);
* $C^\infty(\Omega) = \bigcap_k C^k(\Omega)$;
* $C_c(\Omega)$ — compactly supported continuous functions;
* $C^k_c(\Omega) = C^k(\Omega) \cap C_c(\Omega)$, $C^\infty_c(\Omega) = C^\infty(\Omega) \cap C_c(\Omega)$ (sometimes written $\mathcal{D}(\Omega)$ or $C^\infty_0(\Omega)$).

For $f \in C^1(\Omega)$, $\nabla f = (\partial f/\partial x_1, \ldots, \partial f/\partial x_N)$. For $\alpha = (\alpha_1, \ldots, \alpha_N)$ a multi-index of length $\lvert \alpha\rvert = \alpha_1 + \cdots + \alpha_N \le k$,

$$
D^\alpha f = \frac{\partial^{\alpha_1}}{\partial x_1^{\alpha_1}}\cdots\frac{\partial^{\alpha_N}}{\partial x_N^{\alpha_N}} f.
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.20</span><span class="math-callout__name">(Convolution inherits regularity)</span></p>

If $f \in C^k_c(\mathbb{R}^N)$ ($k \ge 1$) and $g \in L^1_{\mathrm{loc}}(\mathbb{R}^N)$, then $f \star g \in C^k(\mathbb{R}^N)$ and

$$
\boxed{\;D^\alpha(f \star g) = (D^\alpha f) \star g\quad \forall \lvert \alpha\rvert \le k.\;}
$$

In particular, $f \in C^\infty_c \Rightarrow f \star g \in C^\infty$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Smoothing principle)</span></p>

This is the key point: convolving against a smooth compactly supported $f$ *transfers all derivatives* of $f$ to $f \star g$, no matter how singular $g$ is. So convolving against a $C^\infty_c$ function automatically smooths $g$ to $C^\infty$. This is the foundation of regularization.

</div>

#### Mollifiers

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Mollifier sequence)</span></p>

A sequence of **mollifiers** $(\rho_n)_{n \ge 1}$ is any sequence of functions on $\mathbb{R}^N$ with

$$
\rho_n \in C^\infty_c(\mathbb{R}^N),\quad \mathrm{supp}\,\rho_n \subset \overline{B(0, 1/n)},\quad \int \rho_n = 1,\quad \rho_n \ge 0.
$$

</div>

A canonical construction starts from a single $\rho \in C^\infty_c$, e.g.,

$$
\rho(x) = \begin{cases} e^{1/(\lvert x\rvert^2 - 1)} & \text{if } \lvert x\rvert < 1, \\ 0 & \text{if } \lvert x\rvert \ge 1, \end{cases}
$$

and rescales $\rho_n(x) = C n^N \rho(nx)$ with $C = 1/\int\rho$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.21</span><span class="math-callout__name">(Uniform approximation on compacts)</span></p>

If $f \in C(\mathbb{R}^N)$, then $\rho_n \star f \to f$ uniformly on compact sets of $\mathbb{R}^N$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.22</span><span class="math-callout__name">($L^p$ approximation, $1 \le p < \infty$)</span></p>

If $f \in L^p(\mathbb{R}^N)$ with $1 \le p < \infty$, then $\rho_n \star f \to f$ in $L^p(\mathbb{R}^N)$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Given $\varepsilon > 0$, by Theorem 4.12 fix $f_1 \in C_c$ with $\|f - f_1\|_p < \varepsilon$. By Proposition 4.21, $\rho_n \star f_1 \to f_1$ uniformly on compacts; combined with $\mathrm{supp}(\rho_n \star f_1) \subset \overline{B(0, 1/n)} + \mathrm{supp}\,f_1$ (a fixed compact for $n$ large), $\|\rho_n \star f_1 - f_1\|_p \to 0$. Now

$$
\rho_n \star f - f = \rho_n \star (f - f_1) + (\rho_n \star f_1 - f_1) + (f_1 - f),
$$

and Theorem 4.15 gives $\|\rho_n \star (f - f_1)\|_p \le \|f - f_1\|_p < \varepsilon$. So $\limsup_n \|\rho_n \star f - f\|_p \le 2\varepsilon$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 4.23</span><span class="math-callout__name">($C^\infty_c$ dense in $L^p$, $1 \le p < \infty$)</span></p>

Let $\Omega \subset \mathbb{R}^N$ open. Then $C^\infty_c(\Omega)$ is dense in $L^p(\Omega)$ for $1 \le p < \infty$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Given $f \in L^p(\Omega)$, extend by $0$ to $\bar f \in L^p(\mathbb{R}^N)$. Choose compact $K_n \subset \Omega$ with $K_n \uparrow \Omega$ and $\mathrm{dist}(K_n, \Omega^c) \ge 2/n$. Set $g_n = \chi_{K_n}\bar f$, $f_n = \rho_n \star g_n$. Then $\mathrm{supp}\,f_n \subset \overline{B(0, 1/n)} + K_n \subset \Omega$, so $f_n \in C^\infty_c(\Omega)$. And $\|f_n - f\|_{L^p(\Omega)} \le \|g_n - \bar f\|_p + \|\rho_n \star \bar f - \bar f\|_p \to 0$ by dominated convergence and Theorem 4.22. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 4.24</span><span class="math-callout__name">(Du Bois-Reymond / fundamental lemma)</span></p>

Let $\Omega \subset \mathbb{R}^N$ open and $u \in L^1_{\mathrm{loc}}(\Omega)$. If $\int u f = 0$ for every $f \in C^\infty_c(\Omega)$, then $u = 0$ a.e. on $\Omega$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

For $g \in L^\infty(\mathbb{R}^N)$ with compact support in $\Omega$, set $g_n = \rho_n \star g \in C^\infty_c(\Omega)$ for $n$ large. Hypothesis gives $\int u g_n = 0$. Since $g_n \to g$ in $L^1$, a subsequence converges a.e. with uniform $L^\infty$ bound; dominated convergence gives $\int u g = 0$. Take $g = \chi_K \mathrm{sign}\,u$ for any compact $K \subset \Omega$ to get $\int_K \lvert u\rvert = 0$. $\square$

</details>
</div>

### 4.5 Criterion for Strong Compactness in $L^p$

When does a family of functions in $L^p(\Omega)$ have *compact closure* in $L^p$ (strong topology)? In $C(K)$ the answer is given by Ascoli–Arzelà; we now state and prove the $L^p$ analogue.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.25</span><span class="math-callout__name">(Ascoli–Arzelà)</span></p>

Let $K$ be a compact metric space and $\mathcal{H} \subset C(K)$ a bounded **uniformly equicontinuous** subset:

$$
\forall \varepsilon > 0\ \exists \delta > 0\ \text{ such that }\ d(x_1, x_2) < \delta \Rightarrow \lvert f(x_1) - f(x_2)\rvert < \varepsilon\ \forall f \in \mathcal{H}.
$$

Then the closure of $\mathcal{H}$ in $C(K)$ is compact.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Translation operator)</span></p>

For $h \in \mathbb{R}^N$, the **shift** $\tau_h$ is defined by $(\tau_h f)(x) = f(x + h)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.26</span><span class="math-callout__name">(Kolmogorov–M. Riesz–Fréchet)</span></p>

Let $\mathcal{F}$ be a bounded set in $L^p(\mathbb{R}^N)$ with $1 \le p < \infty$. Assume the **integral equicontinuity**

$$
\boxed{\;\lim_{\lvert h\rvert \to 0} \|\tau_h f - f\|_p = 0\quad \text{uniformly in } f \in \mathcal{F},\;} \tag{22}
$$

i.e., $\forall \varepsilon > 0\ \exists \delta > 0$ such that $\|\tau_h f - f\|_p < \varepsilon\ \forall f \in \mathcal{F}, \forall h$ with $\lvert h\rvert < \delta$.

Then for any measurable $\Omega \subset \mathbb{R}^N$ with **finite measure**, the set $\mathcal{F}\rvert_\Omega$ has compact closure in $L^p(\Omega)$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof (sketch)</summary>

**Step 1.** $\|\rho_n \star f - f\|_p \le \varepsilon$ for $1/n < \delta$, uniformly in $f \in \mathcal{F}$. By Hölder,

$$
\lvert(\rho_n \star f)(x) - f(x)\rvert^p \le \int \lvert f(x-y) - f(x)\rvert^p \rho_n(y)\,dy.
$$

Integrate over $x$ and Fubini.

**Step 2.** $\|\rho_n \star f\|_\infty \le C_n \|f\|_p$ and $\lvert(\rho_n \star f)(x_1) - (\rho_n \star f)(x_2)\rvert \le C_n \|f\|_p \lvert x_1 - x_2\rvert$. (Hölder + $\nabla(\rho_n \star f) = (\nabla \rho_n) \star f$.)

**Step 3.** Truncation: there is bounded measurable $\omega \subset \Omega$ with $\|f\|_{L^p(\Omega \setminus \omega)} < \varepsilon\ \forall f \in \mathcal{F}$. (Use Step 1 and $\lvert \Omega \setminus \omega\rvert$ small.)

**Step 4.** On $\omega$, the family $(\rho_n \star \mathcal{F})\rvert_{\overline\omega}$ is bounded uniformly equicontinuous in $C(\overline\omega)$ (by Step 2), hence has compact closure in $C(\overline\omega)$ by Ascoli–Arzelà, hence in $L^p(\omega)$. Cover by finitely many balls of radius $\varepsilon$ in $L^p(\omega)$. Combine with Step 1 and Step 3 to cover $\mathcal{F}\rvert_\Omega$ by balls of radius $3\varepsilon$ in $L^p(\Omega)$. Total boundedness in the complete space $L^p(\Omega)$ gives compact closure. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why finite measure?)</span></p>

In Theorem 4.26 one cannot conclude in general that $\mathcal{F}$ itself has compact closure in $L^p(\mathbb{R}^N)$ (Exercise 4.33). Without finite measure, an extra "tightness at infinity" hypothesis is needed:

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 4.27</span><span class="math-callout__name">(Compactness on all of $\mathbb{R}^N$)</span></p>

Under the assumptions of Theorem 4.26, if in addition

$$
\forall \varepsilon > 0\ \exists \Omega \subset \mathbb{R}^N\ \text{bounded measurable such that}\ \|f\|_{L^p(\mathbb{R}^N \setminus \Omega)} < \varepsilon\ \forall f \in \mathcal{F}, \tag{27}
$$

then $\mathcal{F}$ has compact closure in $L^p(\mathbb{R}^N)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 4.28</span><span class="math-callout__name">(Compactness via convolution)</span></p>

Let $G \in L^1(\mathbb{R}^N)$ be fixed and $\mathcal{B}$ bounded in $L^p(\mathbb{R}^N)$ ($1 \le p < \infty$). Set

$$
\mathcal{F} = G \star \mathcal{B} = \lbrace G \star u\,;\ u \in \mathcal{B}\rbrace.
$$

Then $\mathcal{F}\rvert_\Omega$ has compact closure in $L^p(\Omega)$ for every measurable $\Omega$ with finite measure.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Lemma 4.3: for $G \in L^q(\mathbb{R}^N)$ with $1 \le q < \infty$, $\lim_{h \to 0}\|\tau_h G - G\|_q = 0$. (Approximate $G$ in $L^q$ by $C_c$, on which translation is continuous.) Then for $f = G \star u$,

$$
\|\tau_h f - f\|_p = \|(\tau_h G - G) \star u\|_p \le \|\tau_h G - G\|_1 \|u\|_p,
$$

uniformly small in $u \in \mathcal{B}$. Apply Theorem 4.26. $\square$

</details>
</div>

### Comments on Chapter 4

1. **Egorov's theorem.** In a finite-measure space, a.e. convergence is *almost* uniform convergence:

   <div class="math-callout math-callout--theorem" markdown="1">
     <p class="math-callout__title"><span class="math-callout__label">Theorem 4.29</span><span class="math-callout__name">(Egorov)</span></p>

   Let $\Omega$ be a measure space with $\lvert \Omega\rvert < \infty$ and $(f_n)$ a sequence of measurable functions with $f_n(x) \to f(x)$ a.e. and $\lvert f(x)\rvert < \infty$ a.e. Then for every $\varepsilon > 0$ there exists $A \subset \Omega$ measurable with $\lvert \Omega \setminus A\rvert < \varepsilon$ and $f_n \to f$ uniformly on $A$.

   </div>

   See Exercise 4.14, Halmos, Folland, Hewitt–Stromberg, etc.

2. **Weakly compact sets in $L^1$.** Since $L^1$ is not reflexive, bounded sets do not play a strong role in $\sigma(L^1, L^\infty)$. The right notion is *equi-integrability*:

   <div class="math-callout math-callout--theorem" markdown="1">
     <p class="math-callout__title"><span class="math-callout__label">Theorem 4.30</span><span class="math-callout__name">(Dunford–Pettis)</span></p>

   Let $\mathcal{F}$ be bounded in $L^1(\Omega)$. Then $\mathcal{F}$ has compact closure in $\sigma(L^1, L^\infty)$ iff $\mathcal{F}$ is **equi-integrable**:

   * $\forall \varepsilon > 0\ \exists \delta > 0$ such that $\int_A \lvert f\rvert < \varepsilon$ for all $A$ with $\lvert A\rvert < \delta$ and all $f \in \mathcal{F}$;
   * $\forall \varepsilon > 0\ \exists \omega \subset \Omega$ with $\lvert \omega\rvert < \infty$ such that $\int_{\Omega \setminus \omega} \lvert f\rvert < \varepsilon$ for all $f \in \mathcal{F}$.

   </div>

   See Problem 23, Dunford–Schwartz, Beauzamy, Diestel, Fonseca–Leoni, Neveu, Dellacherie–Meyer.

3. **Radon measures.** Bounded sets in $L^1$ have no compactness; an effective remedy is to embed $L^1$ into the larger space $\mathcal{M}(\overline\Omega)$ of **Radon measures** (the dual of $C(\overline\Omega)$). The natural embedding $T : L^1(\Omega) \to \mathcal{M}(\overline\Omega)$, $\langle Tf, u\rangle = \int f u$, is an isometry. Then bounded sequences in $L^1$ have weak-$\star$ convergent subsequences in $\mathcal{M}(\overline\Omega)$ (e.g., a sequence in $L^1$ may converge to a *Dirac measure* in this sense).

   <div class="math-callout math-callout--theorem" markdown="1">
     <p class="math-callout__title"><span class="math-callout__label">Theorem 4.31</span><span class="math-callout__name">(Riesz: Radon ↔ signed Borel measures)</span></p>

   Let $\mu$ be a Radon measure on $\overline\Omega$. There exists a unique signed Borel measure $\nu$ on $\overline\Omega$ such that

   $$
   \langle \mu, u\rangle = \int_{\overline\Omega} u\,d\nu\quad \forall u \in C(\overline\Omega).
   $$

   </div>

   See Royden, Rudin, Folland, Knapp, Malliavin, Halmos, Fonseca–Leoni.

4. **Bochner integral of vector-valued functions.** For $E$ a Banach space, $L^p(\Omega; E)$ consists of measurable $f : \Omega \to E$ with $\int \|f\|^p\,d\mu < \infty$. Most properties of §4.2–4.3 carry over with mild assumptions on $E$: e.g., if $E$ is reflexive and $1 < p < \infty$, then $L^p(\Omega; E)$ is reflexive with dual $L^{p'}(\Omega; E^\star)$. See Yosida, Cohn, Hille, Beauzamy, Schwartz. The space $L^p(\Omega; E)$ is the natural setting for evolution equations when $\Omega$ is a time interval (Chapter 10).

5. **Interpolation theory.**

   <div class="math-callout math-callout--theorem" markdown="1">
     <p class="math-callout__title"><span class="math-callout__label">Theorem 4.32</span><span class="math-callout__name">(Schur, M. Riesz, Thorin)</span></p>

   Let $\Omega$ be a measure space with $\lvert \Omega\rvert < \infty$, and $T : L^1 \to L^1$, $T : L^\infty \to L^\infty$ a bounded linear operator, with norms $M_1 = \|T\|\_{\mathcal{L}(L^1, L^1)}$, $M_\infty = \|T\|\_{\mathcal{L}(L^\infty, L^\infty)}$. Then $T : L^p \to L^p$ is bounded for every $1 < p < \infty$, with

   $$
   M_p \le M_1^{1/p}\,M_\infty^{1/p'}.
   $$

   </div>

   See Schur, Riesz, Thorin, Marcinkiewicz, Zygmund; followed by Lions, Peetre, Calderon, Stein, Gagliardo. References: Folland, Dunford–Schwartz Vol. 1, Bergh–Löfström, Stein–Weiss, Lions–Magenes, Reed–Simon Vol. 2.

6. **Young's inequality (full form).**

   <div class="math-callout math-callout--theorem" markdown="1">
     <p class="math-callout__title"><span class="math-callout__label">Theorem 4.33</span><span class="math-callout__name">(Young, generalized)</span></p>

   Let $f \in L^p(\mathbb{R}^N)$, $g \in L^q(\mathbb{R}^N)$, with $1 \le p, q \le \infty$ and $\tfrac{1}{r} = \tfrac{1}{p} + \tfrac{1}{q} - 1 \ge 0$. Then $f \star g \in L^r(\mathbb{R}^N)$ and $\|f \star g\|_r \le \|f\|_p \|g\|_q$.

   </div>

   See Exercise 4.30.

7. **Convolution and PDEs.** The notion of convolution — extended to *distributions* (L. Schwartz, Knapp) — is fundamental in PDE theory. For example, $P(D) u = f$ in $\mathbb{R}^N$ (with $P(D)$ a constant-coefficient differential operator) admits the solution $u = E \star f$, where $E$ is a **fundamental solution** of $P(D)$ (Malgrange–Ehrenpreis; cf. Comment 2(b) of Chapter 1). In particular, $\Delta u = f$ in $\mathbb{R}^3$ has a solution $u = E \star f$ with $E(x) = -(4\pi\lvert x\rvert)^{-1}$.

## Chapter 5: Hilbert Spaces

Among Banach spaces, the **Hilbert spaces** form a remarkably tractable class — they enjoy *all* the structural properties one might wish for:

* a *scalar product* induces the norm, so the geometry is Euclidean (parallelogram law, projections, orthogonality);
* every closed subspace has a topological complement (the orthogonal complement), in stark contrast to general Banach spaces;
* the dual is canonically *isomorphic* to the space itself (Riesz–Fréchet);
* on a closed convex set, **continuous coercive bilinear forms** always realize their min via Stampacchia (and Lax–Milgram in the unconstrained symmetric case) — the engine behind the variational formulation of elliptic PDE;
* every separable Hilbert space admits an *orthonormal basis* and is isometrically isomorphic to $\ell^2$.

This chapter develops these foundational tools. The material is short but pivotal: virtually every elliptic existence proof in Chapters 8–9 ultimately reduces to a Lax–Milgram or Stampacchia application in some Sobolev Hilbert space.

### 5.1 Definitions and Elementary Properties. Projection onto a Closed Convex Set

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Scalar product, pre-Hilbert space)</span></p>

Let $H$ be a vector space. A **scalar product** $(u, v)$ is a bilinear form $H \times H \to \mathbb{R}$ that is

* **symmetric:** $(u, v) = (v, u)$,
* **positive:** $(u, u) \ge 0\ \forall u \in H$,
* **definite:** $(u, u) \neq 0\ \forall u \neq 0$.

</div>

A scalar product satisfies the **Cauchy–Schwarz inequality**

$$
\boxed{\;\lvert (u, v)\rvert \le (u, u)^{1/2} (v, v)^{1/2}\quad \forall u, v \in H.\;}
$$

(The proof of Cauchy–Schwarz does not actually require the *definite* assumption.) Hence the quantity

$$
\boxed{\;\lvert u\rvert = (u, u)^{1/2}\;}
$$

is a norm — we often denote norms arising from scalar products by $\lvert\cdot\rvert$ instead of $\|\cdot\|$. Indeed,

$$
\lvert u + v\rvert^2 = \lvert u\rvert^2 + 2(u, v) + \lvert v\rvert^2 \le \lvert u\rvert^2 + 2\lvert u\rvert\lvert v\rvert + \lvert v\rvert^2 = (\lvert u\rvert + \lvert v\rvert)^2.
$$

A norm coming from a scalar product satisfies the classical **parallelogram law**:

$$
\boxed{\;\Big\lvert \frac{a + b}{2}\Big\rvert^2 + \Big\lvert \frac{a - b}{2}\Big\rvert^2 = \tfrac{1}{2}(\lvert a\rvert^2 + \lvert b\rvert^2)\quad \forall a, b \in H.\;} \tag{1}
$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hilbert space)</span></p>

A **Hilbert space** is a vector space $H$ equipped with a scalar product such that $H$ is *complete* for the norm $\lvert\cdot\rvert$.

In what follows, $H$ will always denote a Hilbert space.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Basic examples)</span></p>

$L^2(\Omega)$ equipped with $(u, v) = \int_\Omega u(x) v(x)\,d\mu$ is a Hilbert space; in particular $\ell^2$ is a Hilbert space. The Sobolev space $H^1$ studied in Chapters 8 and 9 is "modeled" on $L^2(\Omega)$ and is another fundamental example.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5.1</span><span class="math-callout__name">(Hilbert ⇒ uniformly convex ⇒ reflexive)</span></p>

$H$ is uniformly convex, and thus reflexive.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Given $\varepsilon > 0$ and $u, v \in H$ with $\lvert u\rvert \le 1, \lvert v\rvert \le 1, \lvert u - v\rvert > \varepsilon$. By the parallelogram law,

$$
\Big\lvert \frac{u + v}{2}\Big\rvert^2 = \tfrac{1}{2}(\lvert u\rvert^2 + \lvert v\rvert^2) - \Big\lvert \frac{u - v}{2}\Big\rvert^2 < 1 - \tfrac{\varepsilon^2}{4}.
$$

Hence $\lvert (u+v)/2\rvert < 1 - \delta$ with $\delta = 1 - (1 - \varepsilon^2/4)^{1/2} > 0$. Reflexivity follows from Milman–Pettis (Theorem 3.31). $\square$

</details>
</div>

#### Projection onto a closed convex set

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.2</span><span class="math-callout__name">(Projection onto a closed convex set)</span></p>

Let $K \subset H$ be a nonempty closed convex set. For every $f \in H$ there exists a unique $u \in K$ such that

$$
\lvert f - u\rvert = \min_{v \in K} \lvert f - v\rvert = \mathrm{dist}(f, K). \tag{2}
$$

Moreover, $u$ is **characterized** by

$$
\boxed{\;u \in K \quad \text{and}\quad (f - u, v - u) \le 0\quad \forall v \in K.\;} \tag{3}
$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Projection)</span></p>

The element $u$ is called the **projection** of $f$ onto $K$ and is denoted

$$
\boxed{\;u = P_K f.\;}
$$

</div>

Geometrically, $(3)$ says that the angle between the vector $\overrightarrow{uf} = f - u$ and any vector $\overrightarrow{uv} = v - u$ ($v \in K$) is $\ge \pi/2$ — the line from $u$ to $f$ "leaves" the convex set $K$.

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 5.2</summary>

**(a) Existence — two arguments.**

*Argument 1.* The function $\varphi(v) = \lvert f - v\rvert$ is convex, continuous, and $\varphi(v) \to +\infty$ as $\lvert v\rvert \to \infty$. By Corollary 3.23 (existence of minimizers in reflexive spaces), $\varphi$ achieves its minimum on the closed convex $K$.

*Argument 2 (direct, no reflexivity).* Let $(v_n) \subset K$ be a minimizing sequence: $d_n = \lvert f - v_n\rvert \to d = \inf_K \lvert f - v\rvert$. Apply the parallelogram law to $a = f - v_n, b = f - v_m$:

$$
\Big\lvert f - \frac{v_n + v_m}{2}\Big\rvert^2 + \Big\lvert \frac{v_n - v_m}{2}\Big\rvert^2 = \tfrac{1}{2}(d_n^2 + d_m^2).
$$

Since $(v_n + v_m)/2 \in K$, the first term is $\ge d^2$, so $\lvert (v_n - v_m)/2\rvert^2 \le \tfrac{1}{2}(d_n^2 + d_m^2) - d^2 \to 0$. Hence $(v_n)$ is Cauchy, $v_n \to u \in K$ (closed), and $d = \lvert f - u\rvert$.

**(b) Equivalence of $(2)$ and $(3)$.** Assume $u$ satisfies $(2)$. For $w \in K$ and $t \in [0, 1]$, $v = (1-t)u + tw \in K$, so

$$
\lvert f - u\rvert^2 \le \lvert f - v\rvert^2 = \lvert f - u\rvert^2 - 2t(f - u, w - u) + t^2\lvert w - u\rvert^2.
$$

Hence $2(f - u, w - u) \le t\lvert w - u\rvert^2\ \forall t \in (0, 1]$; let $t \to 0$ to get $(3)$.

Conversely, $(3)$ ⇒ $(2)$ via $\lvert u - f\rvert^2 - \lvert v - f\rvert^2 = 2(f - u, v - u) - \lvert u - v\rvert^2 \le 0$.

**(c) Uniqueness.** If $u_1, u_2$ both satisfy $(3)$:
$(f - u_1, v - u_1) \le 0\ \forall v \in K$ and $(f - u_2, v - u_2) \le 0\ \forall v \in K$.
Choose $v = u_2$ in the first and $v = u_1$ in the second; adding gives $\lvert u_1 - u_2\rvert^2 \le 0$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Minimization → variational inequality)</span></p>

It is not surprising that a *minimization problem* leads to a *system of inequalities*. Recall a basic example: if $F : \mathbb{R} \to \mathbb{R}$ is differentiable and achieves its minimum at $u \in [0, 1]$, then either

* $u \in (0, 1)$ and $F'(u) = 0$, or
* $u = 0$ and $F'(u) \ge 0$, or
* $u = 1$ and $F'(u) \le 0$.

These three cases summarize as $u \in [0, 1]$ and $F'(u)(v - u) \le 0\ \forall v \in [0, 1]$. The same logic applied at level of the gradient gives $(3)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Beyond Hilbert: uniformly convex Banach)</span></p>

Theorem 5.2 extends to nonempty closed convex subsets of any *uniformly convex* Banach space — see Exercise 3.32.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5.3</span><span class="math-callout__name">($P_K$ is a contraction)</span></p>

$P_K$ does not increase distance:

$$
\lvert P_K f_1 - P_K f_2\rvert \le \lvert f_1 - f_2\rvert\quad \forall f_1, f_2 \in H.
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Set $u_i = P_K f_i$. From $(3)$: $(f_1 - u_1, v - u_1) \le 0$ and $(f_2 - u_2, v - u_2) \le 0$ for all $v \in K$. Choose $v = u_2$ in the first, $v = u_1$ in the second, and add:

$$
\lvert u_1 - u_2\rvert^2 \le (f_1 - f_2, u_1 - u_2) \le \lvert f_1 - f_2\rvert\lvert u_1 - u_2\rvert.\quad \square
$$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 5.4</span><span class="math-callout__name">(Orthogonal projection onto a subspace)</span></p>

Assume $M \subset H$ is a *closed linear subspace*. Then $u = P_M f$ is characterized by

$$
\boxed{\;u \in M \quad \text{and}\quad (f - u, v) = 0\quad \forall v \in M.\;} \tag{8}
$$

Moreover, $P_M$ is a *linear* operator, called the **orthogonal projection**.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

By $(3)$, $(f - u, v - u) \le 0\ \forall v \in M$. Since $M$ is a subspace, replace $v - u$ by $tv$ for any $t \in \mathbb{R}$, $v \in M$: $(f - u, tv) \le 0\ \forall t$, hence $(f - u, v) = 0$. Linearity of $P_M$ is immediate from $(8)$. $\square$

</details>
</div>

### 5.2 The Dual Space of a Hilbert Space

It is very easy to construct continuous linear functionals on $H$: for any $f \in H$ the map $u \mapsto (f, u)$ is continuous and linear. The remarkable fact is that *all* continuous linear functionals arise this way.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.5</span><span class="math-callout__name">(Riesz–Fréchet representation)</span></p>

Given any $\varphi \in H^\star$, there exists a unique $f \in H$ such that

$$
\boxed{\;\langle \varphi, u\rangle = (f, u)\quad \forall u \in H.\;}
$$

Moreover, $\lvert f\rvert = \|\varphi\|_{H^\star}$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof — two approaches</summary>

**Approach 1 (via reflexivity, mirrors Theorem 4.11).** Define $T : H \to H^\star$ by $\langle Tf, u\rangle = (f, u)$. By Cauchy–Schwarz $\|Tf\|_{H^\star} \le \lvert f\rvert$; choosing $u = f$ gives $\|Tf\| \ge \lvert f\rvert$. So $T$ is a linear isometry; $T(H)$ is closed in $H^\star$. To show $T(H)$ is dense, let $h \in H^{\star\star}$ vanish on $T(H)$. By reflexivity (Proposition 5.1), $h \in H$ and $\langle Tf, h\rangle = (f, h) = 0\ \forall f \in H$, so $h = 0$.

**Approach 2 (direct, no reflexivity).** Let $M = \varphi^{-1}(\lbrace 0\rbrace)$, a closed subspace. Assume $M \neq H$ (else take $f = 0$). Pick $g_0 \notin M$ and let $g_1 = P_M g_0$, $g = (g_0 - g_1)/\lvert g_0 - g_1\rvert$, so $\lvert g\rvert = 1$ and $(g, v) = 0\ \forall v \in M$. Given $u \in H$, set $v = u - \lambda g$ with $\lambda = \langle\varphi, u\rangle/\langle\varphi, g\rangle$. Then $v \in M$, so $(g, v) = 0$, i.e., $(g, u) = \lambda$. Hence $\langle \varphi, u\rangle = \langle\varphi, g\rangle (g, u)$, which gives the conclusion with $f = \langle \varphi, g\rangle g$. $\square$

</details>
</div>

#### To identify or not to identify? The triplet $V \subset H \subset V^\star$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Pivot space / Gelfand triple)</span></p>

Theorem 5.5 says there is a canonical isometry $H \cong H^\star$. It is therefore *legitimate* to identify $H$ and $H^\star$. We shall **often do so but not always.** Here is a typical situation in which one should be cautious.

Assume $H$ is a Hilbert space with scalar product $(\cdot, \cdot)$ and norm $\lvert\cdot\rvert$. Let $V \subset H$ be a linear subspace dense in $H$, equipped with its *own* norm $\|\cdot\|$ making $V$ a Banach space, and assume the *injection* $V \hookrightarrow H$ is continuous: $\lvert v\rvert \le C\|v\|\ \forall v \in V$.

(Example: $H = L^2(0, 1)$ and $V = L^p(0, 1)$ with $p > 2$, or $V = C([0, 1])$.)

There is a canonical map $T : H^\star \to V^\star$, namely the *restriction* to $V$ of $\varphi \in H^\star$:

$$
\langle T\varphi, v\rangle_{V^\star, V} = \langle \varphi, v\rangle_{H^\star, H}.
$$

Properties of $T$: (i) $\|T\varphi\|\_{V^\star} \le C\|\varphi\|\_{H^\star}$; (ii) $T$ is injective; (iii) $R(T)$ is dense in $V^\star$ if $V$ is reflexive.

Identifying $H^\star$ with $H$ and using $T$ to embed $H \hookrightarrow V^\star$, one writes

$$
\boxed{\;V \subset H \simeq H^\star \subset V^\star,\;}
$$

with all injections continuous and dense (provided $V$ is reflexive). $H$ is called the **pivot space** of the *Gelfand triple*. The pairing $\langle\cdot, \cdot\rangle_{V^\star, V}$ extends $(\cdot, \cdot)_H$:

$$
\langle f, v\rangle_{V^\star, V} = (f, v)_H\quad \forall f \in H,\ \forall v \in V.
$$

**Caution.** If $V$ is itself a Hilbert space with its own scalar product $((\cdot, \cdot))$, one *cannot* identify simultaneously $V$ with $V^\star$ *and* $H$ with $H^\star$ — the identifications would conflict. The standard convention: identify $H \simeq H^\star$ and *not* $V$ with $V^\star$.

**Concrete example.** $H = \ell^2$ with $(u, v) = \sum u_n v_n$, and

$$
V = \Big\lbrace u = (u_n)\,;\ \sum n^2 u_n^2 < \infty \Big\rbrace
$$

with $((u, v)) = \sum n^2 u_n v_n$. Identifying $H^\star \simeq H$, we find $V^\star = \lbrace f = (f_n)\,;\ \sum f_n^2/n^2 < \infty\rbrace$ — strictly bigger than $H$. The Riesz–Fréchet isomorphism $V \to V^\star$ is $u = (u_n) \mapsto Tu = (n^2 u_n)$, while the canonical embedding $V \hookrightarrow H \hookrightarrow V^\star$ is just $u \mapsto u$ — clearly different maps.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reflexivity, redux)</span></p>

It is easy to prove Hilbert spaces are reflexive without invoking uniform convexity: apply Riesz–Fréchet to $H$ and again to $H^\star$ (also Hilbert) — the canonical injection $H \to H^{\star\star}$ is the composition.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Orthogonal complement and complement)</span></p>

If $H \simeq H^\star$ and $M \subset H$, the previously defined $M^\perp \subset H^\star$ (Section 1.3) becomes a subspace of $H$:

$$
M^\perp = \lbrace u \in H\,;\ (u, v) = 0\ \forall v \in M\rbrace.
$$

Clearly $M \cap M^\perp = \lbrace 0\rbrace$. If $M$ is closed, then $M + M^\perp = H$: every $f \in H$ writes $f = P_M f + (f - P_M f)$ with $f - P_M f = P_{M^\perp} f \in M^\perp$. So **every closed subspace of a Hilbert space has a (topological) complement** — a special property not shared by general Banach spaces.

</div>

### 5.3 The Theorems of Stampacchia and Lax–Milgram

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Continuous & coercive bilinear form)</span></p>

A bilinear form $a : H \times H \to \mathbb{R}$ is

* **continuous** if there is $C$ such that $\lvert a(u, v)\rvert \le C\lvert u\rvert\lvert v\rvert\ \forall u, v$;
* **coercive** if there is $\alpha > 0$ such that $a(v, v) \ge \alpha \lvert v\rvert^2\ \forall v$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.6</span><span class="math-callout__name">(Stampacchia)</span></p>

Assume $a(u, v)$ is a continuous coercive bilinear form on $H$. Let $K \subset H$ be a nonempty closed convex set. Then for every $\varphi \in H^\star$ there exists a *unique* $u \in K$ such that

$$
\boxed{\;a(u, v - u) \ge \langle \varphi, v - u\rangle\quad \forall v \in K.\;} \tag{10}
$$

Moreover, if $a$ is *symmetric*, $u$ is characterized by

$$
\boxed{\;u \in K\quad \text{and}\quad \tfrac{1}{2} a(u, u) - \langle \varphi, u\rangle = \min_{v \in K}\Big\lbrace \tfrac{1}{2} a(v, v) - \langle\varphi, v\rangle\Big\rbrace.\;} \tag{11}
$$

</div>

The proof relies on the following classical result:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.7</span><span class="math-callout__name">(Banach fixed-point theorem / contraction mapping principle)</span></p>

Let $X$ be a nonempty complete metric space and $S : X \to X$ a strict contraction:

$$
d(Sv_1, Sv_2) \le k\, d(v_1, v_2)\quad \forall v_1, v_2 \in X,\ \text{with } k < 1.
$$

Then $S$ has a unique fixed point $u = Su$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 5.6</summary>

By Riesz–Fréchet, write $\langle\varphi, v\rangle = (f, v)$ for some $f \in H$. For fixed $u \in H$, the map $v \mapsto a(u, v)$ is in $H^\star$, so there is a unique $Au \in H$ with $a(u, v) = (Au, v)\ \forall v$. The operator $A : H \to H$ is linear and satisfies

$$
\lvert Au\rvert \le C\lvert u\rvert,\qquad (Au, u) \ge \alpha \lvert u\rvert^2\quad \forall u \in H.
$$

Problem $(10)$ becomes: find $u \in K$ with $(Au, v - u) \ge (f, v - u)\ \forall v \in K$, equivalently

$$
(\rho f - \rho A u + u - u, v - u) \le 0\quad \forall v \in K,
$$

i.e., $u = P_K(\rho f - \rho A u + u)$ for any constant $\rho > 0$.

Define $S : K \to K$, $Sv = P_K(\rho f - \rho A v + v)$. Using $P_K$-non-expansiveness (Proposition 5.3) and the bilinear estimates,

$$
\lvert Sv_1 - Sv_2\rvert^2 \le \lvert v_1 - v_2\rvert^2 (1 - 2\rho \alpha + \rho^2 C^2).
$$

Choose $0 < \rho < 2\alpha/C^2$ so $k^2 = 1 - 2\rho\alpha + \rho^2 C^2 < 1$; then $S$ is a strict contraction, and Banach's fixed point theorem gives a unique $u = Su$.

**Symmetric case.** If $a$ is symmetric, $a(u, v)$ is itself a scalar product on $H$ (positivity from coercivity, definiteness from coercivity, symmetry by hypothesis); the corresponding norm $a(u, u)^{1/2}$ is equivalent to $\lvert\cdot\rvert$, so $H$ is a Hilbert space for $a$. Riesz–Fréchet in this new structure represents $\varphi$ by some $g$: $\langle\varphi, v\rangle = a(g, v)$. Problem $(10)$ becomes

$$
a(g - u, v - u) \le 0\quad \forall v \in K,
$$

i.e., $u$ is the projection of $g$ onto $K$ for the new scalar product $a$. By Theorem 5.2, $u$ minimizes $a(g - v, g - v)^{1/2}$ on $K$, equivalently $a(v, v) - 2 a(g, v) + a(g, g)$, equivalently $\tfrac{1}{2} a(v, v) - \langle\varphi, v\rangle$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Convexity of the energy)</span></p>

If $a(v, v) \ge 0\ \forall v$, then $v \mapsto a(v, v)$ is convex (an immediate check), and so is the *energy* $J(v) = \tfrac{1}{2}a(v, v) - \langle\varphi, v\rangle$. This is what makes Stampacchia's variational characterization $(11)$ a *minimization* over $K$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 5.8</span><span class="math-callout__name">(Lax–Milgram)</span></p>

Assume $a(u, v)$ is a continuous coercive bilinear form on $H$. For every $\varphi \in H^\star$ there exists a *unique* $u \in H$ with

$$
\boxed{\;a(u, v) = \langle \varphi, v\rangle\quad \forall v \in H.\;} \tag{17}
$$

If $a$ is symmetric, $u$ is characterized by

$$
\boxed{\;u \in H\quad \text{and}\quad \tfrac{1}{2} a(u, u) - \langle\varphi, u\rangle = \min_{v \in H}\Big\lbrace \tfrac{1}{2} a(v, v) - \langle\varphi, v\rangle\Big\rbrace.\;} \tag{18}
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Apply Theorem 5.6 with $K = H$. Since $K$ is the whole space, the inequality $a(u, v - u) \ge \langle\varphi, v - u\rangle$ holds for $v$ replaced by $v + u$, hence becomes the equality $a(u, v) = \langle\varphi, v\rangle\ \forall v \in H$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Lax–Milgram in PDE — Euler equations and energy)</span></p>

Lax–Milgram is a *simple and efficient* tool for solving linear elliptic PDE (Chapters 8–9). The connection between $(17)$ and the minimization $(18)$ is the bridge to the *calculus of variations*: in mechanics/physics, $(18)$ is often the natural starting point — least action, minimization of energy — and $(17)$ is the **Euler equation** associated to it. Roughly, $(17)$ is "$F'(u) = 0$" for $F(v) = \tfrac{1}{2} a(v, v) - \langle\varphi, v\rangle$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Direct elementary proof of Lax–Milgram)</span></p>

There is a direct elementary argument for $(17)$: prove $A : H \to H$ is bijective. Three facts:

(a) $A$ is *injective* — coercivity gives $\alpha\lvert u\rvert \le \lvert Au\rvert$ from $\alpha\lvert u\rvert^2 \le (Au, u) \le \lvert Au\rvert\lvert u\rvert$.

(b) $R(A)$ is *closed* — same coercivity estimate makes $A$ proper-on-bounded sets.

(c) $R(A)$ is *dense* — if $v$ satisfies $(Au, v) = 0\ \forall u \in H$, take $u = v$ and use coercivity to get $v = 0$.

Combine: $A$ is bijective. $\square$

</div>

### 5.4 Hilbert Sums. Orthonormal Bases

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hilbert sum)</span></p>

Let $(E_n)_{n \ge 1}$ be a sequence of *closed subspaces* of $H$. We say $H$ is the **Hilbert sum** of the $E_n$'s, written $H = \bigoplus_n E_n$, if

(a) the $E_n$'s are mutually orthogonal: $(u, v) = 0\ \forall u \in E_n, \forall v \in E_m,\ m \neq n$;

(b) the algebraic linear span of $\bigcup_n E_n$ is dense in $H$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.9</span><span class="math-callout__name">(Bessel–Parseval)</span></p>

Suppose $H = \bigoplus_n E_n$. For $u \in H$ set $u_n = P_{E_n} u$ and $S_n = \sum_{k=1}^n u_k$. Then

$$
\lim_{n \to \infty} S_n = u, \tag{19}
$$

$$
\boxed{\;\sum_{k=1}^\infty \lvert u_k\rvert^2 = \lvert u\rvert^2.\;} \tag{20}
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.1</span><span class="math-callout__name">(Convergence of orthogonal series)</span></p>

Assume $(v_n) \subset H$ satisfies

(i) $(v_m, v_n) = 0\ \forall m \neq n$,

(ii) $\sum_k \lvert v_k\rvert^2 < \infty$.

Set $S_n = \sum_{k=1}^n v_k$. Then $S = \lim_n S_n$ exists, and

$$
\lvert S\rvert^2 = \sum_{k=1}^\infty \lvert v_k\rvert^2.
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Lemma 5.1</summary>

For $m > n$, orthogonality gives $\lvert S_m - S_n\rvert^2 = \sum_{k=n+1}^m \lvert v_k\rvert^2 \to 0$, so $(S_n)$ is Cauchy. Pass to the limit in $\lvert S_n\rvert^2 = \sum_{k=1}^n \lvert v_k\rvert^2$. $\square$

</details>
</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 5.9</summary>

By Corollary 5.4, $(u - u_n, v) = 0\ \forall v \in E_n$, in particular $(u, u_n) = \lvert u_n\rvert^2$. Adding over $k = 1, \ldots, n$, $(u, S_n) = \sum_{k=1}^n \lvert u_k\rvert^2 = \lvert S_n\rvert^2$ (by orthogonality of distinct $u_k$'s). Hence $\lvert S_n\rvert \le \lvert u\rvert$, so $\sum \lvert u_k\rvert^2 \le \lvert u\rvert^2$.

By Lemma 5.1, $S = \lim_n S_n$ exists. We claim $S = P_{\overline{F}} u$ where $F$ is the algebraic span of $\bigcup_n E_n$: $(u - S_n, v) = 0\ \forall v \in E_m,\ m \le n$, and as $n \to \infty$, $(u - S, v) = 0\ \forall v \in E_m,\ \forall m$, hence $(u - S, v) = 0\ \forall v \in F$, hence $\forall v \in \overline{F}$. By assumption (b), $\overline{F} = H$, so $S = u$. $(20)$ follows from $\lvert S_n\rvert^2 = \sum_{k=1}^n \lvert u_k\rvert^2$. $\square$

</details>
</div>

#### Orthonormal bases

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Orthonormal / Hilbert basis)</span></p>

A sequence $(e_n)_{n \ge 1}$ in $H$ is an **orthonormal basis** (or **Hilbert basis**, or simply *basis*) if

(i) $\lvert e_n\rvert = 1\ \forall n$ and $(e_m, e_n) = 0\ \forall m \neq n$,

(ii) the linear span of the $e_n$'s is dense in $H$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Hilbert basis vs. Hamel/algebraic basis)</span></p>

Not to be confused with an *algebraic* (Hamel) basis $(e_i)_{i \in I}$, in which every $u \in H$ is a *finite* linear combination of the $e_i$'s (Exercise 1.5). For Hilbert bases the expansions are *infinite series*, requiring the topological structure.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 5.10</span><span class="math-callout__name">(Fourier expansion in a Hilbert basis)</span></p>

Let $(e_n)$ be an orthonormal basis. For every $u \in H$,

$$
\boxed{\;u = \sum_{k=1}^\infty (u, e_k) e_k,\qquad \lvert u\rvert^2 = \sum_{k=1}^\infty \lvert(u, e_k)\rvert^2.\;}
$$

Conversely, given $(\alpha_n) \in \ell^2$, the series $\sum \alpha_k e_k$ converges to some $u \in H$ with $(u, e_k) = \alpha_k\ \forall k$ and $\lvert u\rvert^2 = \sum \alpha_k^2$.

</div>

(Apply Theorem 5.9 to $E_n = \mathbb{R} e_n$, noting $P_{E_n} u = (u, e_n) e_n$, plus Lemma 5.1 for the converse.)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Convergence is not absolute)</span></p>

In general the series $\sum u_k$ in Theorem 5.9 and $\sum (u, e_k) e_k$ in Corollary 5.10 are *not absolutely* convergent — it can happen that $\sum \lvert u_k\rvert = \infty$ or $\sum \lvert(u, e_k)\rvert = \infty$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.11</span><span class="math-callout__name">(Existence of orthonormal bases — separable case)</span></p>

Every **separable** Hilbert space has an orthonormal basis.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof (Gram–Schmidt)</summary>

Let $(v_n)$ be a countable dense subset. Let $F_k$ = span of $\lbrace v_1, \ldots, v_k\rbrace$. The sequence $(F_k)$ is non-decreasing, finite-dimensional, with $\bigcup_k F_k$ dense in $H$. Pick a unit vector $e_1 \in F_1$. If $F_2 \neq F_1$, choose $e_2 \in F_2$ with $\lbrace e_1, e_2\rbrace$ orthonormal in $F_2$. Continue Gram–Schmidt to build an orthonormal basis. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why other Hilbert spaces still matter)</span></p>

Theorem 5.11 + Corollary 5.10 show all separable Hilbert spaces are *isomorphic and isometric* to $\ell^2$ via $u \mapsto ((u, e_k))_{k \ge 1}$. Despite this seemingly spectacular result, it remains *very important* to consider other Hilbert spaces — $L^2(\Omega)$, the Sobolev $H^1(\Omega)$, etc. — because many natural linear and (especially) nonlinear operators look unmanageable when expressed in coordinates.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Non-separable Hilbert spaces)</span></p>

If $H$ is *non-separable* (a rather unusual situation), one may still construct — using Zorn's lemma — an *uncountable* orthonormal basis $(e_i)_{i \in I}$. See Rudin, Taylor–Lay, Folland, Choquet.

</div>

### Comments on Chapter 5

#### 1. Characterization of Hilbert spaces

When is a given norm $\|\cdot\|$ on $E$ a **Hilbert norm** — i.e., $\|u\| = (u, u)^{1/2}$ for some scalar product? Various criteria:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.12</span><span class="math-callout__name">(Fréchet–von Neumann–Jordan)</span></p>

If $\|\cdot\|$ satisfies the **parallelogram law** $(1)$, then $\|\cdot\|$ is a Hilbert norm.

</div>

(See Yosida, Exercise 5.1.)

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.13</span><span class="math-callout__name">(Kakutani)</span></p>

Let $E$ be a normed space with $\dim E \ge 3$. Assume every subspace $F$ of dimension $2$ has a projection operator of *norm $1$*, i.e., $\exists P : E \to F$ bounded linear with $Pu = u\ \forall u \in F$ and $\|P\| \le 1$. Then $\|\cdot\|$ is a Hilbert norm.

</div>

(Every subspace of dimension $1$ already has a norm-$1$ projection, by Hahn–Banach.)

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.14</span><span class="math-callout__name">(de Figueiredo–Karlovitz)</span></p>

Let $\dim E \ge 3$. Consider the *radial projection on the unit ball*

$$
T u = \begin{cases} u & \text{if } \|u\| \le 1, \\ u/\|u\| & \text{if } \|u\| > 1. \end{cases}
$$

If $\|Tu - Tv\| \le \|u - v\|\ \forall u, v \in E$, then $\|\cdot\|$ is a Hilbert norm.

</div>

(In *any* normed space, $T$ satisfies $\|Tu - Tv\| \le 2\|u - v\|$, and the constant $2$ cannot be improved; Exercise 5.6.)

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.15</span><span class="math-callout__name">(Lindenstrauss–Tzafriri)</span></p>

If $E$ is a Banach space such that every closed subspace has a complement, then $E$ is *Hilbertizable*: there exists an equivalent Hilbert norm on $E$.

</div>

(Cf. Remark 8 of Chapter 2.)

#### 2. Variational inequalities

Stampacchia's theorem is the starting point of the theory of **variational inequalities** (Kinderlehrer–Stampacchia), with applications to:

* mechanics and physics (Duvaut–J. L. Lions);
* free boundary problems (Baiocchi–Capelo, Friedman);
* optimal control (J.-L. Lions, Barbu);
* stochastic control (Bensoussan–J.-L. Lions).

#### 3. Nonlinear extensions

Stampacchia and Lax–Milgram extend to nonlinear *monotone* operators:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.16</span><span class="math-callout__name">(Minty–Browder)</span></p>

Let $E$ be a reflexive Banach space and $A : E \to E^\star$ a continuous nonlinear map satisfying

$$
\langle A v_1 - A v_2, v_1 - v_2\rangle > 0\quad \forall v_1 \neq v_2 \in E,
$$

$$
\lim_{\|v\| \to \infty} \frac{\langle A v, v\rangle}{\|v\|} = +\infty.
$$

Then for every $f \in E^\star$ the equation $A u = f$ has a unique solution $u \in E$.

</div>

See Browder, J.-L. Lions, Problem 31.

#### 4. Special bases. Fourier series. Wavelets

In Chapter 6 we present a powerful technique for constructing orthonormal bases: take eigenvectors of a compact self-adjoint operator. In practice one often uses bases of $L^2(\Omega)$ consisting of *eigenfunctions of differential operators* (Sections 8.6, 9.8). The orthonormal basis on $L^2(0, \pi)$

$$
e_n(x) = \sqrt{2/\pi}\sin(nx),\ n \ge 1\quad \text{or}\quad e_n(x) = \sqrt{2/\pi}\cos(nx),\ n \ge 0,
$$

leads to **Fourier series** and harmonic analysis (Ash, Dym–McKean, Katznelson, Rees–Shah–Stanojevic).

A puzzle that occupied analysts for decades: given $u \in L^2(0, \pi)$, $S_n = \sum_{k=1}^n (u, e_k) e_k$. We know $S_n \to u$ in $L^2$ (Corollary 5.10), and a subsequence converges a.e. (Theorem 4.9). Does the *full sequence* converge a.e.?

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.17</span><span class="math-callout__name">(Carleson)</span></p>

If $u \in L^2(0, \pi)$, then $S_n(x) \to u(x)$ a.e.

</div>

Other classical bases of $L^2(0, 1)$ or $L^2(\mathbb{R})$ are due to *Bessel, Legendre, Hermite, Laguerre, Chebyshev, Jacobi*. See Courant–Hilbert, Dautray–Lions Vol. VIII; spectral properties of the Sturm–Liouville operator at end of Chapter 8. The **Haar** and **Walsh** bases (step-function bases of $L^2(0, 1)$) are detailed in Exercises 5.31–5.32, Alexits, Harmuth.

The theory of **wavelets** provides a powerful and beautiful new family of bases — useful for decomposing functions, signals, speech, images, etc. Standard references: Y. Meyer, Coifman–Meyer, Daubechies, David, Chui, Ruskai et al., Benedetto–Frazier, Kaiser, Kahane–Lemarié-Rieusset, Mallat, Bachman–Narici–Beckenstein, Chan–Shen, Wojtaszczyk, Hernandez–Weiss.

#### 5. Schauder bases in Banach spaces

A sequence $(e_n)$ in a Banach space $E$ is a **Schauder basis** if every $u \in E$ has a unique expansion $u = \sum \alpha_k e_k$ in the *norm topology*. All classical separable Banach spaces (in analysis) have a Schauder basis. P. Enflo (1973) gave a counterexample to the long-standing conjecture that every separable Banach space has a Schauder basis. One can even construct closed subspaces of $\ell^p$ ($1 < p < \infty, p \neq 2$) without a Schauder basis (Lindenstrauss–Tzafriri). Szankowski showed $\mathcal{L}(H)$ (with operator norm, $H$ infinite-dimensional separable Hilbert) has *no* Schauder basis. In Chapter 6 a related problem for compact operators — the *approximation property* — also has a negative answer.

## Chapter 6: Compact Operators. Spectral Decomposition of Self-Adjoint Compact Operators

Compact operators are the closest infinite-dimensional analogue of finite-rank matrices. The unit ball is no longer compact (Riesz, Theorem 6.5), so a *general* bounded operator can carry $B_E$ off into a non-compact set; **compact** operators are precisely those that "compress" $B_E$ back to a (relatively) compact image. This single property — the right ingredient to extract subsequences via Bolzano–Weierstrass — drives almost every spectral / regularity result in elliptic PDE.

The chapter has three big arcs:

* **Riesz–Fredholm theory**: for $T \in \mathcal{K}(E)$, the operator $I - T$ behaves like a finite-dimensional linear map — the **Fredholm alternative** $(N(I-T) = \lbrace 0\rbrace) \iff (R(I-T) = E)$ holds, $\dim N(I-T) = \dim N(I - T^\star)$, and $R(I-T)$ is closed with the explicit description $R(I-T) = N(I-T^\star)^\perp$;
* **Spectrum of compact operators**: $\sigma(T) \setminus \lbrace 0\rbrace$ consists of *isolated* eigenvalues with finite-dimensional eigenspaces, accumulating only at $0$ (Theorem 6.8);
* **Spectral decomposition of self-adjoint compact operators on Hilbert space**: $H = \bigoplus_{n \ge 0} N(T - \lambda_n I)$ with $\lambda_n \to 0$ — the classical infinite-dimensional generalization of the diagonalization of a symmetric matrix (Theorem 6.11).

This last result is the unsung workhorse behind Fourier-type expansions in PDE — the eigenfunctions of a self-adjoint compact resolvent give the orthonormal basis used for series solutions of elliptic problems (Sections 8.6 and 9.8).

### 6.1 Definitions. Elementary Properties. Adjoint

Throughout this chapter, unless otherwise specified, $E$ and $F$ denote two Banach spaces.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Compact operator)</span></p>

A bounded operator $T \in \mathcal{L}(E, F)$ is said to be **compact** if $T(B_E)$ has compact closure in $F$ (in the strong topology).

The set of all compact operators from $E$ into $F$ is denoted $\mathcal{K}(E, F)$. We write $\mathcal{K}(E)$ for $\mathcal{K}(E, E)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.1</span><span class="math-callout__name">($\mathcal{K}(E, F)$ is closed in $\mathcal{L}(E, F)$)</span></p>

The set $\mathcal{K}(E, F)$ is a closed linear subspace of $\mathcal{L}(E, F)$ in the operator-norm topology.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

The sum of two compact operators is compact (clear). Suppose $(T_n) \subset \mathcal{K}(E, F)$ and $\|T_n - T\|_{\mathcal{L}(E, F)} \to 0$ for some $T \in \mathcal{L}(E, F)$. We claim $T \in \mathcal{K}(E, F)$. Since $F$ is complete, it suffices to show $T(B_E)$ is *totally bounded*: for every $\varepsilon > 0$ there is a finite $\varepsilon$-cover. Fix $n$ with $\|T_n - T\| < \varepsilon/2$. Since $T_n(B_E)$ has compact closure, it admits a finite cover $\bigcup_{i \in I} B(f_i, \varepsilon/2)$. Then $T(B_E) \subset \bigcup_{i \in I} B(f_i, \varepsilon)$. $\square$

</details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Finite-rank operator)</span></p>

$T \in \mathcal{L}(E, F)$ has **finite rank** if $R(T)$ is finite-dimensional.

</div>

Any finite-rank operator is compact (its image of $B_E$ is bounded in a finite-dimensional space). Combined with Theorem 6.1:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 6.2</span><span class="math-callout__name">(Norm-limits of finite-rank are compact)</span></p>

Let $(T_n)$ be a sequence of finite-rank operators with $\|T_n - T\|_{\mathcal{L}(E, F)} \to 0$. Then $T \in \mathcal{K}(E, F)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The approximation problem)</span></p>

The celebrated **approximation problem** (Banach, Grothendieck) asks the converse: given a compact operator $T$, does there always exist a sequence $(T_n)$ of finite-rank operators with $\|T_n - T\|_{\mathcal{L}(E, F)} \to 0$? It was open for a long time until **P. Enflo (1972)** discovered a counterexample. Subsequently simpler examples were found, e.g., with $F$ a closed subspace of $\ell^p$ (any $1 < p < \infty,\ p \neq 2$); see Lindenstrauss–Tzafriri.

The answer is **positive in some special cases** — for example if $F$ is a *Hilbert space*. Indeed, set $K = \overline{T(B_E)}$, compact in $F$. Given $\varepsilon > 0$, cover $K \subset \bigcup_{i \in I} B(f_i, \varepsilon)$ with finite $I$. Let $G = \mathrm{span}(f_i)$ and $T_\varepsilon = P_G T$ (finite-rank). For $x \in B_E$, pick $i_0$ with $\|Tx - f_{i_0}\| < \varepsilon$. Then

$$
\|P_G T x - f_{i_0}\| < \varepsilon\quad \text{(since } P_G f_{i_0} = f_{i_0}\text{)},
$$

so $\|P_G T x - Tx\| < 2\varepsilon$, i.e., $\|T_\varepsilon - T\|_{\mathcal{L}(E, F)} < 2\varepsilon$.

More generally, the answer is positive whenever $F$ has a *Schauder basis*.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Approximating nonlinear maps by finite-rank ones)</span></p>

A useful technique in *nonlinear* analysis: approximate a continuous map $T : X \to F$ ($X$ topological, $F$ Banach) such that $T(X)$ has compact closure, by *nonlinear* maps of finite rank. For $\varepsilon > 0$, cover $K = \overline{T(X)} \subset \bigcup_{i \in I} B(f_i, \varepsilon/2)$ and define

$$
T_\varepsilon(x) = \frac{\sum_{i \in I} q_i(x) f_i}{\sum_{i \in I} q_i(x)},\quad q_i(x) = \max\lbrace \varepsilon - \|Tx - f_i\|,\ 0\rbrace.
$$

Then $\|T_\varepsilon(x) - T(x)\| < \varepsilon\ \forall x \in X$.

This kind of approximation is, e.g., the bridge from Brouwer's fixed-point theorem to **Schauder's fixed-point theorem** (Deimling, Granas–Dugundji, Franklin, Exercise 6.26). A similar technique was used by **Lomonosov** to prove the existence of nontrivial *invariant subspaces* for a large class of linear operators (Pearcy, Akhiezer–Glazman, Granas–Dugundji, Problem 42). Another linear application with a simple proof based on Schauder's fixed-point theorem is the **Krein–Rutman theorem** (Theorem 6.13, Problem 41).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.3</span><span class="math-callout__name">(Compactness is a two-sided ideal)</span></p>

Let $E, F, G$ be Banach spaces. If $T \in \mathcal{L}(E, F)$, $S \in \mathcal{K}(F, G)$, then $S \circ T \in \mathcal{K}(E, G)$. Symmetrically, if $T \in \mathcal{K}(E, F)$ and $S \in \mathcal{L}(F, G)$, then $S \circ T \in \mathcal{K}(E, G)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.4</span><span class="math-callout__name">(Schauder: $T$ compact ⇔ $T^\star$ compact)</span></p>

If $T \in \mathcal{K}(E, F)$, then $T^\star \in \mathcal{K}(F^\star, E^\star)$. And conversely.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

We show $T^\star(B_{F^\star})$ has compact closure in $E^\star$. Let $(v_n)$ be a sequence in $B_{F^\star}$. We claim $(T^\star v_n)$ has a convergent subsequence. Let $K = \overline{T(B_E)}$ (compact metric space). Consider $\mathcal{H} \subset C(K)$ defined by

$$
\mathcal{H} = \lbrace \varphi_n : x \in K \mapsto \langle v_n, x\rangle\,;\ n = 1, 2, \ldots\rbrace.
$$

The hypotheses of Ascoli–Arzelà (Theorem 4.25) are satisfied: $\mathcal{H}$ is bounded (uniform bound $\le 1$ since $\|v_n\| \le 1$ and $K$ bounded) and equicontinuous ($\lvert\varphi_n(x) - \varphi_n(y)\rvert \le \|x - y\|$). Pass to a subsequence $(\varphi_{n_k})$ converging uniformly on $K$ to some $\varphi \in C(K)$. Then

$$
\sup_{u \in B_E} \lvert\langle v_{n_k}, Tu\rangle - \varphi(Tu)\rvert \xrightarrow[k \to \infty]{} 0,
$$

so $(T^\star v_{n_k})$ is Cauchy in $E^\star$, hence converges.

Conversely, if $T^\star \in \mathcal{K}$, the first half gives $T^{\star\star} \in \mathcal{K}(E^{\star\star}, F^{\star\star})$. In particular, $T^{\star\star}(B_E)$ has compact closure in $F^{\star\star}$; but $T(B_E) = T^{\star\star}(B_E)$, and $F$ is closed in $F^{\star\star}$, so $T(B_E)$ has compact closure in $F$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Compact operators turn weak into strong)</span></p>

If $T \in \mathcal{K}(E, F)$ and $u_n \rightharpoonup u$ weakly in $E$, then $Tu_n \to Tu$ **strongly** in $F$. The converse is also true if $E$ is *reflexive* (Exercise 6.7). This *weak-to-strong* upgrade is what makes compact operators the right tool for converting weak compactness (free in reflexive spaces) into strong compactness.

</div>

### 6.2 The Riesz–Fredholm Theory

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 6.1</span><span class="math-callout__name">(Riesz's lemma)</span></p>

Let $E$ be an n.v.s. and $M \subset E$ a closed linear subspace with $M \neq E$. Then

$$
\forall \varepsilon > 0\ \exists u \in E\text{ with } \|u\| = 1\text{ and } \mathrm{dist}(u, M) \ge 1 - \varepsilon.
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $v \in E$, $v \notin M$. Since $M$ is closed, $d = \mathrm{dist}(v, M) > 0$. Pick $m_0 \in M$ with $d \le \|v - m_0\| \le d/(1 - \varepsilon)$. Set $u = (v - m_0)/\|v - m_0\|$. For every $m \in M$,

$$
\|u - m\| = \frac{\|v - m_0 - \|v - m_0\| m\|}{\|v - m_0\|} \ge \frac{d}{\|v - m_0\|} \ge 1 - \varepsilon
$$

(since $m_0 + \|v - m_0\|m \in M$). $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">($\varepsilon = 0$?)</span></p>

If $M$ is finite-dimensional (or, more generally, *reflexive*), one can choose $\varepsilon = 0$ in Riesz's lemma. But this is not true in general (Exercise 1.17).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.5</span><span class="math-callout__name">(Riesz: $B_E$ compact ⇔ $\dim E < \infty$)</span></p>

Let $E$ be an n.v.s. with $B_E$ compact. Then $E$ is finite-dimensional.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Suppose $E$ is infinite-dimensional. Build a sequence $E_n$ of finite-dimensional subspaces with $E_{n-1} \subsetneq E_n$. By Lemma 6.1 there is a sequence $(u_n) \subset E_n$ with $\|u_n\| = 1$ and $\mathrm{dist}(u_n, E_{n-1}) \ge 1/2$. Then $\|u_n - u_m\| \ge 1/2$ for $m < n$, so $(u_n) \subset B_E$ has no convergent subsequence — contradiction. $\square$

</details>
</div>

#### The Fredholm alternative

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.6</span><span class="math-callout__name">(Fredholm alternative)</span></p>

Let $T \in \mathcal{K}(E)$. Then:

(a) $N(I - T)$ is finite-dimensional,

(b) $R(I - T)$ is closed, and more precisely $R(I - T) = N(I - T^\star)^\perp$,

(c) $N(I - T) = \lbrace 0\rbrace \iff R(I - T) = E$,

(d) $\dim N(I - T) = \dim N(I - T^\star)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What "alternative" means)</span></p>

The Fredholm alternative concerns solvability of $u - Tu = f$:

* **Either** for every $f \in E$ the equation $u - Tu = f$ has a unique solution,
* **Or** the homogeneous equation $u - Tu = 0$ admits $n$ linearly independent solutions, and in this case the inhomogeneous equation is solvable **iff** $f$ satisfies $n$ orthogonality conditions, i.e., $f \in N(I - T^\star)^\perp$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why (c) is remarkable)</span></p>

Property (c) is familiar in finite-dimensional spaces: an endomorphism is injective iff it is surjective. However, *in infinite-dimensional spaces a bounded operator may be injective without being surjective and conversely* — e.g., the right shift (resp. left shift) in $\ell^2$. So (c) is a remarkable property of operators of the form $I - T$ with $T \in \mathcal{K}(E)$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 6.6</summary>

**(a)** Let $E_1 = N(I - T)$. Then $B_{E_1} \subset T(B_E)$ (since $u = Tu$ for $u \in E_1$), so $B_{E_1}$ has compact closure. By Theorem 6.5, $E_1$ is finite-dimensional.

**(b)** Closed range. Let $f_n = u_n - Tu_n \to f$. Set $d_n = \mathrm{dist}(u_n, N(I - T))$. Since $N(I - T)$ is finite-dimensional, there is $v_n \in N(I - T)$ with $d_n = \|u_n - v_n\|$. Then $f_n = (u_n - v_n) - T(u_n - v_n)$. We claim $\|u_n - v_n\|$ is bounded. If not, set $w_n = (u_n - v_n)/\|u_n - v_n\|$; then $w_n - Tw_n \to 0$. By compactness of $T$, a subsequence $Tw_{n_k} \to z$, so $w_{n_k} \to z$ and $z \in N(I - T)$. But $\mathrm{dist}(w_n, N(I - T)) = 1$ — contradiction. So $\|u_n - v_n\|$ is bounded; pass to a subsequence with $T(u_{n_k} - v_{n_k}) \to \ell$. Then $u_{n_k} - v_{n_k} \to f + \ell$, and setting $g = f + \ell$ gives $g - Tg = f$, i.e., $f \in R(I - T)$. So $R(I - T)$ is closed; by Theorem 2.19, $R(I - T) = N(I - T^\star)^\perp$ and $R(I - T^\star) = N(I - T)^\perp$.

**(c)** ($\Rightarrow$): suppose $E_1 = R(I - T) \neq E$. Then $T(E_1) \subset E_1$, so $T\rvert_{E_1} \in \mathcal{K}(E_1)$, and $E_2 = (I - T)(E_1)$ is closed in $E_1$, with $E_2 \neq E_1$ (since $I - T$ is injective). Iterate: $E_n = (I - T)^n(E)$ is a strictly decreasing sequence of closed subspaces. By Riesz's lemma, build $u_n \in E_n,\ \|u_n\| = 1, \mathrm{dist}(u_n, E_{n+1}) \ge 1/2$. For $n > m$,

$$
Tu_n - Tu_m = -(u_n - Tu_n) + (u_m - Tu_m) + (u_n - u_m) \in -E_{n+1} + E_{m+1} + E_n \subset E_{m+1},
$$

so $\|Tu_n - Tu_m\| \ge \mathrm{dist}(u_m, E_{m+1}) \ge 1/2$ — contradicting compactness of $T$.

($\Leftarrow$): if $R(I - T) = E$, then $N(I - T^\star) = R(I - T)^\perp = \lbrace 0\rbrace$, and applying the previous step to $T^\star \in \mathcal{K}(E^\star)$ gives $R(I - T^\star) = E^\star$, hence $N(I - T) = R(I - T^\star)^\perp = \lbrace 0\rbrace$.

**(d)** Set $d = \dim N(I - T)$, $d^\star = \dim N(I - T^\star)$. We show $d^\star \le d$. Suppose, for contradiction, $d < d^\star$. Since $N(I - T)$ is finite-dimensional, it admits a complement in $E$ (Section 2.4); let $P$ be a continuous projection onto $N(I - T)$. On the other hand, $R(I - T) = N(I - T^\star)^\perp$ has finite codimension $d^\star$, hence has a complement $F$ in $E$ of dimension $d^\star$. Since $d < d^\star$, there is a linear injective non-surjective map $\Lambda : N(I - T) \to F$. Set $S = T + \Lambda \circ P$. Then $S \in \mathcal{K}(E)$ ($\Lambda \circ P$ has finite rank). $N(I - S) = \lbrace 0\rbrace$: if $u = Su = Tu + \Lambda P u$, then $u - Tu = \Lambda Pu \in F \cap R(I - T) = \lbrace 0\rbrace$, so $u \in N(I - T)$ and $\Lambda u = 0$, hence $u = 0$. By (c), $R(I - S) = E$ — but for any $f \in F \setminus R(\Lambda)$, $u - Su = f$ has no solution. Contradiction.

Apply the same to $T^\star$: $\dim N(I - T^{\star\star}) \le \dim N(I - T^\star) \le \dim N(I - T)$. Since $N(I - T^{\star\star}) \supset N(I - T)$, we get $d = d^\star$. $\square$

</details>
</div>

### 6.3 The Spectrum of a Compact Operator

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Resolvent, spectrum, eigenvalue)</span></p>

Let $T \in \mathcal{L}(E)$.

* The **resolvent set** is

  $$\rho(T) = \lbrace \lambda \in \mathbb{R}\,;\ T - \lambda I \text{ is bijective from } E \text{ onto } E\rbrace.$$

* The **spectrum** is $\sigma(T) = \mathbb{R} \setminus \rho(T)$.

* $\lambda$ is an **eigenvalue** if $N(T - \lambda I) \neq \lbrace 0\rbrace$. The space $N(T - \lambda I)$ is the corresponding **eigenspace**. The set of eigenvalues is denoted $EV(T)$.

</div>

If $\lambda \in \rho(T)$, then $(T - \lambda I)^{-1} \in \mathcal{L}(E)$ (Corollary 2.7).

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">($EV(T) \subsetneq \sigma(T)$ in general)</span></p>

Clearly $EV(T) \subset \sigma(T)$. The inclusion can be **strict**: there may exist $\lambda$ with $N(T - \lambda I) = \lbrace 0\rbrace$ but $R(T - \lambda I) \neq E$ (so $\lambda \in \sigma(T)$ but is not an eigenvalue).

**Example.** $E = \ell^2$, right shift $T(u_1, u_2, \ldots) = (0, u_1, u_2, \ldots)$. Then $0 \in \sigma(T)$ (not surjective) but $0 \notin EV(T)$ ($T$ is injective). One can show $EV(T) = \emptyset$ and $\sigma(T) = [-1, +1]$ (Exercise 6.18).

In *finite or infinite-dimensional spaces over $\mathbb{C}$* the situation is *totally different*: the study of eigenvalues and spectra is much richer over $\mathbb{C}$. In finite dim over $\mathbb{C}$, $EV(T) = \sigma(T) \neq \emptyset$ (roots of characteristic polynomial). In infinite dim over $\mathbb{C}$ a non-trivial result asserts $\sigma(T)$ is *always nonempty* (Section 11.4). However, $EV(T)$ may still be empty (e.g., right shift in $\ell^2$).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.7</span><span class="math-callout__name">(Spectrum is compact and bounded by $\|T\|$)</span></p>

For $T \in \mathcal{L}(E)$, $\sigma(T)$ is compact and

$$
\sigma(T) \subset [-\|T\|, +\|T\|].
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

If $\lvert\lambda\rvert > \|T\|$: equation $Tu - \lambda u = f$ becomes $u = \lambda^{-1}(Tu - f)$, a strict contraction (Banach fixed-point, Theorem 5.7), hence has a unique solution. So $\lambda \in \rho(T)$.

$\rho(T)$ open: if $\lambda_0 \in \rho(T)$, $Tu - \lambda u = f$ rewrites as $u = (T - \lambda_0 I)^{-1}[f + (\lambda - \lambda_0)u]$, a contraction provided $\lvert\lambda - \lambda_0\rvert\|(T - \lambda_0 I)^{-1}\| < 1$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.8</span><span class="math-callout__name">(Spectrum of a compact operator)</span></p>

Let $T \in \mathcal{K}(E)$ with $\dim E = \infty$. Then

(a) $0 \in \sigma(T)$,

(b) $\sigma(T) \setminus \lbrace 0\rbrace = EV(T) \setminus \lbrace 0\rbrace$,

(c) one of the following holds:

  * $\sigma(T) = \lbrace 0\rbrace$,
  * $\sigma(T) \setminus \lbrace 0\rbrace$ is a finite set,
  * $\sigma(T) \setminus \lbrace 0\rbrace$ is a sequence converging to $0$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 6.2</span><span class="math-callout__name">(Eigenvalue accumulation only at $0$)</span></p>

Let $T \in \mathcal{K}(E)$ and let $(\lambda_n)$ be a sequence of *distinct* real numbers with $\lambda_n \to \lambda$ and $\lambda_n \in \sigma(T) \setminus \lbrace 0\rbrace\ \forall n$. Then $\lambda = 0$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 6.8</summary>

**(a)** Suppose $0 \notin \sigma(T)$. Then $T$ is bijective and $I = T \circ T^{-1}$ is compact (composition of compact and bounded). So $B_E$ is compact, $\dim E < \infty$ (Theorem 6.5) — contradiction.

**(b)** If $\lambda \in \sigma(T) \setminus \lbrace 0\rbrace$, suppose $\lambda$ is not an eigenvalue: $N(T - \lambda I) = \lbrace 0\rbrace$. Then $N(I - T/\lambda) = \lbrace 0\rbrace$ with $T/\lambda$ compact, so by Fredholm (Theorem 6.6(c)), $R(I - T/\lambda) = E$, hence $\lambda \in \rho(T)$ — contradiction.

**(c)** Use Lemma 6.2: any sequence of distinct $\lambda_n \in \sigma(T)\setminus\lbrace 0\rbrace$ converging to a limit must converge to $0$. So for every $n$, $\sigma(T) \cap \lbrace \lambda\,;\ \lvert\lambda\rvert \ge 1/n\rbrace$ is *empty or finite* (else infinitely many would have a subsequence converging to some $\lambda$ with $\lvert\lambda\rvert \ge 1/n$, contradicting Lemma 6.2). Hence $\sigma(T)\setminus\lbrace 0\rbrace$ is countable; if infinite it can be ordered as a sequence converging to $0$. $\square$

</details>
</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Lemma 6.2</summary>

Pick eigenvectors $e_n \neq 0$ with $(T - \lambda_n I)e_n = 0$. Let $E_n = \mathrm{span}\lbrace e_1, \ldots, e_n\rbrace$. Then $E_n \subsetneq E_{n+1}$ — otherwise $e_{n+1} = \sum_{i=1}^n \alpha_i e_i$ would give $T e_{n+1} = \sum \alpha_i \lambda_i e_i = \lambda_{n+1}\sum \alpha_i e_i$, so $\sum \alpha_i(\lambda_i - \lambda_{n+1}) e_i = 0$, hence all $\alpha_i = 0$ (eigenvectors with distinct eigenvalues are independent). $(T - \lambda_n I)E_n \subset E_{n-1}$.

By Riesz's lemma, choose $u_n \in E_n,\ \|u_n\| = 1,\ \mathrm{dist}(u_n, E_{n-1}) \ge 1/2$. For $2 \le m < n$,

$$
\Big\|\frac{Tu_n}{\lambda_n} - \frac{Tu_m}{\lambda_m}\Big\| = \Big\|\frac{(T - \lambda_n I)u_n}{\lambda_n} - \frac{(T - \lambda_m I)u_m}{\lambda_m} + u_n - u_m\Big\| \ge \mathrm{dist}(u_n, E_{n-1}) \ge 1/2,
$$

since the first three terms lie in $E_{n-1}$. If $\lambda_n \to \lambda \neq 0$, then $(Tu_n)$ would have a convergent subsequence by compactness, contradicting the lower bound on $\|Tu_n/\lambda_n - Tu_m/\lambda_m\|$. Hence $\lambda = 0$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Every $\sigma(T)\setminus\lbrace 0\rbrace$ is realizable)</span></p>

Given *any* sequence $(\alpha_n)$ converging to $0$, there is a compact operator $T$ with $\sigma(T) = (\alpha_n) \cup \lbrace 0\rbrace$: in $\ell^2$, the multiplication operator $T(u_1, u_2, \ldots) = (\alpha_1 u_1, \alpha_2 u_2, \ldots)$ works (compact as norm-limit of finite-rank truncations $T_n u = (\alpha_1 u_1, \ldots, \alpha_n u_n, 0, \ldots)$). Note that $0$ may or may not belong to $EV(T)$; if $0 \in EV(T)$, the eigenspace $N(T)$ may be finite- or infinite-dimensional.

</div>

### 6.4 Spectral Decomposition of Self-Adjoint Compact Operators

In what follows $E = H$ is a Hilbert space and $T \in \mathcal{L}(H)$. Identifying $H^\star \simeq H$, we view $T^\star$ as an operator from $H$ into itself.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Self-adjoint operator)</span></p>

$T \in \mathcal{L}(H)$ is **self-adjoint** if $T^\star = T$, i.e.,

$$
\boxed{\;(Tu, v) = (u, Tv)\quad \forall u, v \in H.\;}
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.9</span><span class="math-callout__name">(Spectrum lies in $[m, M]$, with extremes in $\sigma(T)$)</span></p>

Let $T \in \mathcal{L}(H)$ be self-adjoint. Set

$$
m = \inf_{\substack{u \in H \\ \lvert u\rvert = 1}} (Tu, u),\qquad M = \sup_{\substack{u \in H \\ \lvert u\rvert = 1}} (Tu, u).
$$

Then $\sigma(T) \subset [m, M]$, $m \in \sigma(T)$, $M \in \sigma(T)$, and $\|T\| = \max\lbrace \lvert m\rvert,\lvert M\rvert\rbrace$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $\lambda > M$. Then for all $u \in H$, $((\lambda I - T)u, u) \ge (\lambda - M)\lvert u\rvert^2 = \alpha\lvert u\rvert^2$ with $\alpha = \lambda - M > 0$. By Lax–Milgram (Corollary 5.8) applied to $a(u, v) = (\lambda u - Tu, v)$, $\lambda I - T$ is bijective, so $\lambda \in \rho(T)$. Similarly $\lambda < m$. Hence $\sigma(T) \subset [m, M]$.

To show $M \in \sigma(T)$: the bilinear form $a(u, v) = (Mu - Tu, v)$ is symmetric and $a(v, v) \ge 0$, so it satisfies Cauchy–Schwarz: $\lvert a(u, v)\rvert \le a(u, u)^{1/2} a(v, v)^{1/2}$. Hence

$$
\lvert Mu - Tu\rvert \le C (Mu - Tu, u)^{1/2}.
$$

Pick $(u_n)$ with $\lvert u_n\rvert = 1$ and $(Tu_n, u_n) \to M$. Then $\lvert Mu_n - Tu_n\rvert \to 0$. If $M \in \rho(T)$ we'd have $u_n = (MI - T)^{-1}(Mu_n - Tu_n) \to 0$ — contradicting $\lvert u_n\rvert = 1$.

For the norm: $4(Tu, v) = (T(u+v), u+v) - (T(u-v), u-v) \le M\lvert u+v\rvert^2 - m\lvert u-v\rvert^2$, so $4\lvert(Tu, v)\rvert \le \mu(\lvert u+v\rvert^2 + \lvert u-v\rvert^2) = 2\mu(\lvert u\rvert^2 + \lvert v\rvert^2)$, with $\mu = \max\lbrace\lvert m\rvert, \lvert M\rvert\rbrace$. Optimize over $v$ to get $\lvert(Tu, v)\rvert \le \mu\lvert u\rvert\lvert v\rvert$, so $\|T\| \le \mu$. Conversely $\lvert(Tu, u)\rvert \le \|T\|\lvert u\rvert^2$ gives $\mu \le \|T\|$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 6.10</span><span class="math-callout__name">($\sigma(T) = \lbrace 0\rbrace \Rightarrow T = 0$, self-adjoint case)</span></p>

If $T \in \mathcal{L}(H)$ is self-adjoint with $\sigma(T) = \lbrace 0\rbrace$, then $T = 0$.

</div>

#### Spectral theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.11</span><span class="math-callout__name">(Spectral decomposition of self-adjoint compact operators)</span></p>

Let $H$ be a *separable* Hilbert space and $T \in \mathcal{K}(H)$ self-adjoint. Then there exists a Hilbert basis of $H$ composed of *eigenvectors* of $T$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $(\lambda_n)_{n \ge 1}$ be the (distinct) nonzero eigenvalues of $T$. Set $\lambda_0 = 0$, $E_0 = N(T)$, $E_n = N(T - \lambda_n I)$. Recall:

* $0 \le \dim E_0 \le \infty$ (kernel can be finite- or infinite-dim);
* $0 < \dim E_n < \infty$ for $n \ge 1$ (Fredholm 6.6(a)).

We claim $H$ is the Hilbert sum of $(E_n)_{n \ge 0}$ (Section 5.4).

**(i) Mutual orthogonality.** For $u \in E_m,\ v \in E_n,\ m \neq n$: $Tu = \lambda_m u,\ Tv = \lambda_n v$, so $\lambda_m(u, v) = (Tu, v) = (u, Tv) = \lambda_n(u, v)$. Since $\lambda_m \neq \lambda_n$, $(u, v) = 0$.

**(ii) Density.** Let $F$ = span of $\bigcup_{n \ge 0} E_n$. We show $F$ is dense in $H$. Clearly $T(F) \subset F$, so $T(F^\perp) \subset F^\perp$ (for $u \in F^\perp$, $(Tu, v) = (u, Tv) = 0\ \forall v \in F$, so $Tu \in F^\perp$). Let $T_0 = T\rvert_{F^\perp}$, a self-adjoint compact operator on $F^\perp$. We claim $\sigma(T_0) = \lbrace 0\rbrace$. Otherwise some $\lambda \neq 0$ in $\sigma(T_0)$ is an eigenvalue of $T_0$ (by Theorem 6.8): there is $u \in F^\perp,\ u \neq 0$ with $T_0 u = \lambda u$, i.e., $\lambda = \lambda_n$ for some $n \ge 1$, $u \in E_n \subset F$. So $u \in F^\perp \cap F = \lbrace 0\rbrace$ — contradiction.

Hence $\sigma(T_0) = \lbrace 0\rbrace$, so by Corollary 6.10, $T_0 = 0$, i.e., $T$ vanishes on $F^\perp$. Therefore $F^\perp \subset N(T) = E_0 \subset F$, hence $F^\perp \subset F$, hence $F^\perp = \lbrace 0\rbrace$ and $F$ is dense.

Pick a Hilbert basis in each $E_n$ (existence via Theorem 5.11 for $E_0$ which is separable, since $H$ is; trivial for finite-dimensional $E_n,\ n \ge 1$). Their union is a Hilbert basis of $H$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Diagonalization and finite-rank approximation)</span></p>

For any $u \in H$, write $u = \sum_{n \ge 0} u_n$ with $u_n \in E_n$ (Theorem 5.9). Then $Tu = \sum_{n \ge 1} \lambda_n u_n$. The truncations $T_k u = \sum_{n=1}^k \lambda_n u_n$ are *finite-rank* and

$$
\|T_k - T\| \le \sup_{n \ge k+1} \lvert\lambda_n\rvert \to 0\quad \text{as } k \to \infty.
$$

So a self-adjoint compact operator is the norm limit of an explicit sequence of finite-rank operators — in fact (Remark 1) **every compact operator on a Hilbert space**, not necessarily self-adjoint, is the norm limit of finite-rank operators.

</div>

### Comments on Chapter 6

#### 1. Fredholm operators

Theorem 6.6 is the first step toward the theory of **Fredholm operators**. Given Banach spaces $E, F$, an operator $A \in \mathcal{L}(E, F)$ is a **Fredholm operator** (or *Noether operator*), written $A \in \Phi(E, F)$, if

* $N(A)$ is finite-dimensional,
* $R(A)$ is closed and has finite codimension.

The **index** of $A$ is

$$
\boxed{\;\mathrm{ind}\,A = \dim N(A) - \mathrm{codim}\,R(A).\;}
$$

For $A = I - T$ with $T \in \mathcal{K}(E)$, Theorem 6.6 gives $\mathrm{ind}\,A = 0$.

Main properties (Kato, Schechter, Lang, Taylor–Lay, Lax, Hörmander Vol. 3, Problem 38):

* (a) $\Phi(E, F)$ is *open* in $\mathcal{L}(E, F)$, and $A \mapsto \mathrm{ind}\,A$ is *continuous* (constant on connected components).
* (b) Every $A \in \Phi(E, F)$ is **invertible modulo finite-rank operators**: there is $B \in \mathcal{L}(F, E)$ with $A B - I_F$ and $B A - I_E$ of finite rank. Conversely, if $B \in \mathcal{L}(F, E)$ exists with $A B - I_F$ and $B A - I_E$ in $\mathcal{K}$, then $A \in \Phi(E, F)$.
* (c) **Stability under compact perturbations.** $A \in \Phi(E, F),\ T \in \mathcal{K}(E, F) \Rightarrow A + T \in \Phi(E, F)$ and $\mathrm{ind}(A + T) = \mathrm{ind}\,A$.
* (d) **Composition.** $A \in \Phi(E, F), B \in \Phi(F, G) \Rightarrow B \circ A \in \Phi(E, G)$ with $\mathrm{ind}(B \circ A) = \mathrm{ind}\,A + \mathrm{ind}\,B$.

#### 2. Hilbert–Schmidt operators

Let $H$ be a separable Hilbert space. $T \in \mathcal{L}(H)$ is **Hilbert–Schmidt** if there is a Hilbert basis $(e_n)$ with $\|T\|_{HS}^2 = \sum_n \lvert Te_n\rvert^2 < \infty$. (Independent of basis; $\|\cdot\|_{HS}$ is a norm.) Every Hilbert–Schmidt operator is compact. The fundamental example:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.12</span><span class="math-callout__name">($L^2$-kernels ↔ Hilbert–Schmidt)</span></p>

Let $H = L^2(\Omega)$ and $K(x, y) \in L^2(\Omega \times \Omega)$. Then

$$
u \mapsto (Ku)(x) = \int_\Omega K(x, y) u(y)\,dy
$$

is a Hilbert–Schmidt operator. Conversely, every Hilbert–Schmidt operator on $L^2(\Omega)$ is of this form for some unique $K \in L^2(\Omega \times \Omega)$.

</div>

References: Balakrishnan, Dunford–Schwartz Vol. 2, Problem 40.

#### 3. Multiplicity of eigenvalues

For $T \in \mathcal{K}(E)$ and $\lambda \in \sigma(T)\setminus\lbrace 0\rbrace$, the sequence $N((T - \lambda I)^k)$ ($k \ge 1$) is *strictly increasing* up to some $p$ then constant (Taylor–Lay, Kreyszig, Problem 36). $p$ is the **ascent** of $T - \lambda I$. Two notions of multiplicity:

* $\dim N(T - \lambda I)$ — **geometric multiplicity**;
* $\dim N((T - \lambda I)^p)$ — **algebraic multiplicity**.

They coincide if $E$ is a Hilbert space and $T$ is self-adjoint (Problem 36).

#### 4. Spectral analysis

For self-adjoint $T \in \mathcal{L}(H)$ (possibly non-compact), there is a construction — the **spectral family** of $T$ — that extends Theorem 6.11. It defines a *functional calculus*: for any continuous $f$, the operator $f(T) \in \mathcal{L}(H)$ is meaningful. The theory extends to *unbounded* and *non-self-adjoint* operators, requiring only that $T$ be **normal** ($TT^\star = T^\star T$). A vast subject, especially in Banach spaces over $\mathbb{C}$ (Section 11.4): Rudin, Kreyszig, Friedman, Yosida (elementary), Reed–Simon Vol. 1, Kato, Dautray–Lions Vols. VIII–IX, Dunford–Schwartz Vol. 2, Akhiezer–Glazman, Taylor–Lay, Weidmann, Conway, Schechter (more advanced).

#### 5. The min–max principle

The **min–max formulas** (Courant–Fischer) give a very useful way of computing eigenvalues. References: Courant–Hilbert, Lax, Problem 37; the monograph Weinberger contains numerous developments.

#### 6. The Krein–Rutman theorem

A useful tool in spectral properties of second-order elliptic operators (Chapter 9):

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.13</span><span class="math-callout__name">(Krein–Rutman)</span></p>

Let $E$ be a Banach space and $P \subset E$ a *convex cone* with vertex at $0$ (i.e., $\lambda x + \mu y \in P\ \forall \lambda, \mu \ge 0,\ \forall x, y \in P$). Assume $P$ is closed, $\mathrm{Int}\,P \neq \emptyset$, $P \neq E$. Let $T \in \mathcal{K}(E)$ such that $T(P \setminus \lbrace 0\rbrace) \subset \mathrm{Int}\,P$. Then there exist $x_0 \in \mathrm{Int}\,P$ and $\lambda_0 > 0$ such that $T x_0 = \lambda_0 x_0$. Moreover, $\lambda_0$ is the *unique* eigenvalue corresponding to an eigenvector of $T$ in $P$, i.e., $Tx = \lambda x$ with $x \in P$, $x \neq 0$, implies $\lambda = \lambda_0$ and $x = mx_0$ for some $m > 0$. Finally,

$$
\lambda_0 = \max\lbrace \lvert\lambda\rvert\,;\ \lambda \in \sigma(T)\rbrace,
$$

and the multiplicity (both geometric and algebraic) of $\lambda_0$ equals one.

</div>

The proof in Problem 41 is due to P. Rabinowitz. Variants in Schaefer, Nussbaum, Bonsall, Toland.

## Chapter 7: The Hille–Yosida Theorem

So far the operators we have studied (compact, self-adjoint, etc.) have all been *bounded*. PDE theory, however, is dominated by **unbounded** operators — the Laplacian, the heat operator, transport operators, etc. — and these typically arise as generators of *time evolutions*. The central question is:

$$
\text{When does the Cauchy problem}\quad \frac{du}{dt} + Au = 0,\ u(0) = u_0\quad \text{have a solution?}
$$

The answer for *linear* unbounded $A$ on a Hilbert space (and more generally a Banach space) is the **Hille–Yosida theorem**: solvability for all $u_0 \in D(A)$ is equivalent to $A$ being **maximal monotone** ($m$-accretive in the Banach setting). The strategy is *Yosida's regularization*: replace $A$ by its bounded approximation $A_\lambda = \tfrac{1}{\lambda}(I - J_\lambda)$ (with $J_\lambda = (I + \lambda A)^{-1}$, the *resolvent*), solve the easier problem with $A_\lambda$, and pass to the limit.

This chapter develops:

* **§7.1 — Maximal monotone operators.** The right algebraic object: $A$ monotone ($Av, v \ge 0$) and $I + A$ surjective. Properties of resolvent $J_\lambda$ and Yosida approximation $A_\lambda$.
* **§7.2 — The Hille–Yosida theorem.** Existence/uniqueness for $u' + Au = 0$ with initial data in $D(A)$, plus the *contraction semigroup* structure.
* **§7.3 — Higher regularity** for initial data in $D(A^k)$.
* **§7.4 — The self-adjoint case.** Smoothing effect: solutions become $C^\infty$ for $t > 0$ even when $u_0 \in H$ — the abstract analogue of parabolic regularity.

This is the abstract engine behind Chapter 10 (evolution PDEs).

### 7.1 Definition and Elementary Properties of Maximal Monotone Operators

Throughout this chapter $H$ denotes a Hilbert space.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Monotone, maximal monotone operator)</span></p>

An unbounded linear operator $A : D(A) \subset H \to H$ is **monotone** if

$$
\boxed{\;(Av, v) \ge 0\quad \forall v \in D(A).\;}
$$

It is **maximal monotone** if, in addition, $R(I + A) = H$, i.e.,

$$
\forall f \in H\ \exists u \in D(A)\text{ such that } u + Au = f.
$$

(Some authors say $A$ is *accretive* / $-A$ is *dissipative*.)

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7.1</span><span class="math-callout__name">(Basic properties of maximal monotone operators)</span></p>

Let $A$ be a maximal monotone operator. Then

(a) $D(A)$ is dense in $H$,

(b) $A$ is a closed operator,

(c) For every $\lambda > 0$, $(I + \lambda A)$ is bijective from $D(A)$ onto $H$, $(I + \lambda A)^{-1}$ is a bounded operator, and $\|(I + \lambda A)^{-1}\|_{\mathcal{L}(H)} \le 1$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

**(a) $D(A)$ is dense.** Let $f \in H$ with $(f, v) = 0\ \forall v \in D(A)$; we show $f = 0$. By maximal monotonicity, pick $v_0 \in D(A)$ with $v_0 + Av_0 = f$. Then

$$
0 = (f, v_0) = \lvert v_0\rvert^2 + (Av_0, v_0) \ge \lvert v_0\rvert^2,
$$

so $v_0 = 0$ and $f = 0$.

**(b) Closedness.** Given $f \in H$, the equation $u + Au = f$ has a *unique* solution: if $u_1, u_2$ both solve it, $(u_1 - u_2) + A(u_1 - u_2) = 0$, take the scalar product with $u_1 - u_2$ and use monotonicity to get $\lvert u_1 - u_2\rvert \le 0$. Moreover $\lvert u\rvert \le \lvert f\rvert$ from $\lvert u\rvert^2 + (Au, u) = (f, u) \le \lvert f\rvert\lvert u\rvert$. Hence $f \mapsto u =: (I + A)^{-1}f$ is a bounded linear operator with norm $\le 1$.

For closedness: if $u_n \in D(A)$, $u_n \to u$, $Au_n \to f$, then $u_n = (I + A)^{-1}(u_n + Au_n) \to (I + A)^{-1}(u + f)$, so $u = (I + A)^{-1}(u + f)$, i.e., $u \in D(A)$ and $u + Au = u + f$, hence $Au = f$.

**(c) Bijectivity for all $\lambda > 0$.** Suppose $R(I + \lambda_0 A) = H$ for some $\lambda_0 > 0$. We show $R(I + \lambda A) = H$ for all $\lambda > \lambda_0/2$. The equation $u + \lambda Au = f$ rewrites as

$$
u = (I + \lambda_0 A)^{-1}\Big[\frac{\lambda_0}{\lambda} f + \Big(1 - \frac{\lambda_0}{\lambda}\Big) u\Big].
$$

If $\lvert 1 - \lambda_0/\lambda\rvert < 1$, i.e., $\lambda > \lambda_0/2$, the contraction-mapping principle gives a solution. By induction starting from $\lambda_0 = 1$: $\lambda > 1/2 \Rightarrow \lambda > 1/4 \Rightarrow \cdots$ covers all $\lambda > 0$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Stability under sum?)</span></p>

If $A$ is maximal monotone, so is $\lambda A$ for every $\lambda > 0$. However, if $A$ and $B$ are maximal monotone, the sum $A + B$ defined on $D(A) \cap D(B)$ is **not** in general maximal monotone (subtle conditions are needed).

</div>

#### Resolvent and Yosida approximation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Resolvent and Yosida approximation)</span></p>

Let $A$ be a maximal monotone operator. For every $\lambda > 0$, set

$$
\boxed{\;J_\lambda = (I + \lambda A)^{-1},\qquad A_\lambda = \frac{1}{\lambda}(I - J_\lambda).\;}
$$

* $J_\lambda$ is the **resolvent** of $A$;
* $A_\lambda$ is the **Yosida approximation** (or *regularization*) of $A$.

Note $\|J_\lambda\|_{\mathcal{L}(H)} \le 1$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7.2</span><span class="math-callout__name">(Properties of $J_\lambda$ and $A_\lambda$)</span></p>

Let $A$ be maximal monotone. Then

* $(\text{a}_1)$ $A_\lambda v = A(J_\lambda v)\ \forall v \in H,\ \forall \lambda > 0$,
* $(\text{a}_2)$ $A_\lambda v = J_\lambda(A v)\ \forall v \in D(A),\ \forall \lambda > 0$,
* (b) $\lvert A_\lambda v\rvert \le \lvert Av\rvert\ \forall v \in D(A),\ \forall \lambda > 0$,
* (c) $\lim_{\lambda \to 0} J_\lambda v = v\ \forall v \in H$,
* (d) $\lim_{\lambda \to 0} A_\lambda v = Av\ \forall v \in D(A)$,
* (e) $(A_\lambda v, v) \ge 0\ \forall v \in H,\ \forall \lambda > 0$,
* (f) $\lvert A_\lambda v\rvert \le (1/\lambda)\lvert v\rvert\ \forall v \in H,\ \forall \lambda > 0$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

$(\text{a}_1)$ Direct: $v = J_\lambda v + \lambda A(J_\lambda v)$ rearranges as $A_\lambda v = A(J_\lambda v)$.

$(\text{a}_2)$ For $v \in D(A)$: $A_\lambda v + A(v - J_\lambda v) = Av$ (by $(\text{a}_1)$, replacing $v - J_\lambda v$ by $\lambda A(J_\lambda v) = \lambda A_\lambda v$), so $A_\lambda v + \lambda A(A_\lambda v) = Av$, i.e., $A_\lambda v = J_\lambda(Av)$.

(b) From $(\text{a}_2)$ and $\|J_\lambda\| \le 1$.

(c) For $v \in D(A)$: $\lvert v - J_\lambda v\rvert = \lambda\lvert A_\lambda v\rvert \le \lambda\lvert Av\rvert \to 0$. For general $v \in H$: pick $v_1 \in D(A)$ with $\lvert v - v_1\rvert \le \varepsilon$ (using density), then $\lvert J_\lambda v - v\rvert \le \lvert J_\lambda v - J_\lambda v_1\rvert + \lvert J_\lambda v_1 - v_1\rvert + \lvert v_1 - v\rvert \le 2\varepsilon + \lvert J_\lambda v_1 - v_1\rvert$; let $\lambda \to 0$.

(d) From $(\text{a}_2)$ and (c): $A_\lambda v = J_\lambda(Av) \to Av$.

(e) $(A_\lambda v, v) = (A_\lambda v, v - J_\lambda v) + (A_\lambda v, J_\lambda v) = \lambda\lvert A_\lambda v\rvert^2 + (A(J_\lambda v), J_\lambda v) \ge \lambda\lvert A_\lambda v\rvert^2 \ge 0$.

(f) From the inequality $(A_\lambda v, v) \ge \lambda\lvert A_\lambda v\rvert^2$ and Cauchy–Schwarz: $\lambda\lvert A_\lambda v\rvert^2 \le \lvert A_\lambda v\rvert\lvert v\rvert$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why Yosida approximation matters)</span></p>

$(A_\lambda)_{\lambda > 0}$ is a family of *bounded* operators that **approximates** the unbounded $A$ as $\lambda \to 0$. This is the bridge that lets us solve evolution equations with unbounded $A$: solve the easier problem $u'_\lambda + A_\lambda u_\lambda = 0$ (a linear ODE in $H$) and pass to the limit $\lambda \to 0$. Of course, in general $\|A_\lambda\|_{\mathcal{L}(H)} \le 1/\lambda$ "blows up" as $\lambda \to 0$ — what we use are *uniform* estimates, not norm convergence.

</div>

### 7.2 Solution of the Evolution Problem $\frac{du}{dt} + Au = 0$ on $[0, +\infty)$, $u(0) = u_0$

We start with the classical bounded result.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 7.3</span><span class="math-callout__name">(Cauchy, Lipschitz, Picard)</span></p>

Let $E$ be a Banach space and $F : E \to E$ Lipschitz: $\|Fu - Fv\| \le L\|u - v\|\ \forall u, v \in E$. For every $u_0 \in E$ there is a *unique* solution $u \in C^1([0, +\infty); E)$ of

$$
\frac{du}{dt}(t) = Fu(t) \text{ on } [0, +\infty),\qquad u(0) = u_0. \tag{4}
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Reformulate as the integral equation $u(t) = u_0 + \int_0^t F(u(s))\,ds$. For $k > 0$ set

$$
X = \Big\lbrace u \in C([0, +\infty); E)\,;\ \sup_{t \ge 0} e^{-kt}\|u(t)\| < \infty\Big\rbrace,\quad \|u\|_X = \sup_{t \ge 0} e^{-kt}\|u(t)\|.
$$

Then $X$ is a Banach space. Define $\Phi u(t) = u_0 + \int_0^t F(u(s))\,ds \in X$. Compute

$$
\|\Phi u - \Phi v\|_X \le \frac{L}{k}\|u - v\|_X.
$$

Choose $k > L$: $\Phi$ is a strict contraction on $X$, with a unique fixed point. Uniqueness on bounded intervals reduces to Gronwall: $\varphi(t) = \|u(t) - \overline{u}(t)\|$ satisfies $\varphi(t) \le L\int_0^t \varphi$, so $\varphi \equiv 0$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 7.4</span><span class="math-callout__name">(Hille–Yosida)</span></p>

Let $A$ be a *maximal monotone* operator on $H$. Given any $u_0 \in D(A)$, there exists a *unique* function

$$
u \in C^1([0, +\infty); H) \cap C([0, +\infty); D(A))
$$

satisfying

$$
\boxed{\;\frac{du}{dt} + Au = 0\text{ on } [0, +\infty),\qquad u(0) = u_0.\;} \tag{6}
$$

Moreover,

$$
\lvert u(t)\rvert \le \lvert u_0\rvert,\qquad \Big\lvert\frac{du}{dt}(t)\Big\rvert = \lvert Au(t)\rvert \le \lvert Au_0\rvert\quad \forall t \ge 0.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(From evolution to stationary)</span></p>

The main interest of Theorem 7.4: the study of the *evolution problem* $u' + Au = 0$ reduces to the study of the *stationary equation* $u + Au = f$ — assuming we already know $A$ is monotone, which is easy to check in practice.

</div>

The proof has six steps. The strategy: replace $A$ by its Yosida approximation $A_\lambda$, apply Theorem 7.3, and pass to the limit using *uniform-in-$\lambda$* estimates.

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 7.4</summary>

**Step 1: Uniqueness.** If $u, \overline{u}$ both solve (6), $(\frac{d}{dt}(u - \overline{u}), u - \overline{u}) = -(A(u - \overline{u}), u - \overline{u}) \le 0$. Hence $\tfrac{1}{2}\frac{d}{dt}\lvert u - \overline{u}\rvert^2 \le 0$, so $\lvert u - \overline{u}\rvert$ is nonincreasing. Combined with $\lvert u(0) - \overline{u}(0)\rvert = 0$, we get $u \equiv \overline{u}$.

**Step 2: Approximate problem and basic estimates.** Let $u_\lambda$ solve

$$
\frac{du_\lambda}{dt} + A_\lambda u_\lambda = 0\text{ on } [0, +\infty),\quad u_\lambda(0) = u_0 \in D(A) \tag{7}
$$

(Theorem 7.3 with the bounded $F = -A_\lambda$). The estimates

$$
\lvert u_\lambda(t)\rvert \le \lvert u_0\rvert,\quad \Big\lvert\frac{du_\lambda}{dt}(t)\Big\rvert = \lvert A_\lambda u_\lambda(t)\rvert \le \lvert A u_0\rvert \tag{8, 9}
$$

follow from a useful auxiliary lemma:

> **Lemma 7.1.** Let $w \in C^1([0, +\infty); H)$ satisfy $\frac{dw}{dt} + A_\lambda w = 0$. Then $t \mapsto \lvert w(t)\rvert$ and $t \mapsto \lvert\frac{dw}{dt}(t)\rvert = \lvert A_\lambda w(t)\rvert$ are nonincreasing on $[0, +\infty)$.
>
> *Proof.* $(\frac{dw}{dt}, w) + (A_\lambda w, w) = 0$, with $(A_\lambda w, w) \ge 0$ (Prop. 7.2(e)), gives $\tfrac{1}{2}\frac{d}{dt}\lvert w\rvert^2 \le 0$. Since $A_\lambda \in \mathcal{L}(H)$, by induction $w \in C^\infty$ with $\frac{d}{dt}\big(\frac{dw}{dt}\big) + A_\lambda\big(\frac{dw}{dt}\big) = 0$, so the same applies to $\frac{dw}{dt}$.

Apply Lemma 7.1 to $u_\lambda$ and use $\lvert A_\lambda u_0\rvert \le \lvert Au_0\rvert$ (Prop. 7.2(b)).

**Step 3: $u_\lambda(t)$ converges to some $u(t)$, uniformly on $[0, T]$.** For $\lambda, \mu > 0$,

$$
\frac{du_\lambda}{dt} - \frac{du_\mu}{dt} + A_\lambda u_\lambda - A_\mu u_\mu = 0,
$$

so $\tfrac{1}{2}\frac{d}{dt}\lvert u_\lambda - u_\mu\rvert^2 + (A_\lambda u_\lambda - A_\mu u_\mu, u_\lambda - u_\mu) = 0$. Decompose the cross term using $u_\lambda = J_\lambda u_\lambda + \lambda A_\lambda u_\lambda$:

$$
(A_\lambda u_\lambda - A_\mu u_\mu, u_\lambda - u_\mu) = (A(J_\lambda u_\lambda) - A(J_\mu u_\mu), J_\lambda u_\lambda - J_\mu u_\mu) + (A_\lambda u_\lambda - A_\mu u_\mu, \lambda A_\lambda u_\lambda - \mu A_\mu u_\mu).
$$

The first term is $\ge 0$ (monotonicity), so

$$
\tfrac{1}{2}\frac{d}{dt}\lvert u_\lambda - u_\mu\rvert^2 \le 2(\lambda + \mu)\lvert Au_0\rvert^2,
$$

which integrates to $\lvert u_\lambda(t) - u_\mu(t)\rvert \le 2\sqrt{(\lambda + \mu)t}\lvert Au_0\rvert$. So $(u_\lambda)$ is uniformly Cauchy on $[0, T]$; let $u(t) = \lim u_\lambda(t)$, with $u \in C([0, +\infty); H)$.

**Step 4 (assuming $u_0 \in D(A^2)$): $\frac{du_\lambda}{dt}$ converges uniformly on $[0, T]$.** Set $v_\lambda = \frac{du_\lambda}{dt}$, satisfying $\frac{dv_\lambda}{dt} + A_\lambda v_\lambda = 0$. Repeating Step 3 with extra care:

$$
\tfrac{1}{2}\frac{d}{dt}\lvert v_\lambda - v_\mu\rvert^2 \le 2(\lambda + \mu)\lvert A^2 u_0\rvert^2,
$$

using the bounds $\lvert A_\lambda v_\lambda(t)\rvert \le \lvert A_\lambda v_\lambda(0)\rvert = \lvert A_\lambda A u_0\rvert$ and $A_\lambda A u_0 = J_\lambda^2 A^2 u_0$ when $A u_0 \in D(A)$. Hence $v_\lambda$ converges uniformly on $[0, T]$.

**Step 5 (still $u_0 \in D(A^2)$): the limit $u$ solves (6).** From Steps 3, 4: $u \in C^1([0, +\infty); H)$ and $\frac{du_\lambda}{dt} \to \frac{du}{dt}$ uniformly. Rewrite (7) as $\frac{du_\lambda}{dt}(t) + A(J_\lambda u_\lambda(t)) = 0$. Since $J_\lambda u_\lambda(t) \to u(t)$ (by $\lvert J_\lambda u_\lambda - u\rvert \le \lvert J_\lambda u_\lambda - J_\lambda u\rvert + \lvert J_\lambda u - u\rvert \le \lvert u_\lambda - u\rvert + \lvert J_\lambda u - u\rvert \to 0$) and $A$ is closed, we get $u(t) \in D(A)$ and $\frac{du}{dt} + Au = 0$. $u \in C([0, +\infty); D(A))$ follows from continuity of $Au(t) = -\frac{du}{dt}(t)$.

**Step 6: $u_0 \in D(A)$ via density of $D(A^2)$.** A useful lemma:

> **Lemma 7.2.** $D(A^2) = \lbrace v \in D(A)\,;\ Av \in D(A)\rbrace$ is dense in $D(A)$ for the graph norm.
>
> *Proof.* Set $\overline{u}_0 = J_\lambda u_0$, so $\overline{u}_0 \in D(A)$ and $\overline{u}_0 + \lambda A\overline{u}_0 = u_0$ gives $A\overline{u}_0 \in D(A)$. By Prop. 7.2(c, d), $\overline{u}_0 \to u_0$ and $A\overline{u}_0 = J_\lambda Au_0 \to Au_0$ as $\lambda \to 0$.

Approximate $u_0 \in D(A)$ by $u_{0n} \in D(A^2)$ with $u_{0n} \to u_0$ and $Au_{0n} \to Au_0$. Solutions $u_n$ satisfy $\lvert u_n(t) - u_m(t)\rvert \le \lvert u_{0n} - u_{0m}\rvert \to 0$ and $\lvert\frac{du_n}{dt}(t) - \frac{du_m}{dt}(t)\rvert \le \lvert Au_{0n} - Au_{0m}\rvert \to 0$, both uniform in $t$. Pass to the limit; closedness of $A$ gives $u(t) \in D(A)$ and the equation. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Generalized vs. classical solutions)</span></p>

Let $u_\lambda$ solve (7).

(a) If $u_0 \in D(A)$: Step 3 already gives convergence $u_\lambda \to u$, with $u \in C^1([0, +\infty); H) \cap C([0, +\infty); D(A))$ a *classical* solution of (6).

(b) If we only assume $u_0 \in H$: the limit $u(t) = \lim_{\lambda \to 0} u_\lambda(t)$ still exists for every $t \ge 0$, but in general $u(t) \notin D(A)$ and $u$ may be nowhere differentiable on $[0, +\infty)$. So problem (6) has *no classical solution* — we view $u$ as a **generalized solution**. We shall see in §7.4 that if $A$ is **self-adjoint** this never happens: $u$ is a classical solution for *every* $u_0 \in H$, even when $u_0 \notin D(A)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Contraction semigroups)</span></p>

For each $t \ge 0$ the map $u_0 \in D(A) \mapsto u(t) \in D(A)$ is linear and $\lvert u(t)\rvert \le \lvert u_0\rvert$. Since $D(A)$ is dense in $H$, this map extends by continuity to all of $H$ as a bounded operator $S_A(t) \in \mathcal{L}(H)$. The family $\lbrace S_A(t)\rbrace_{t \ge 0}$ satisfies:

* **(a)** $\|S_A(t)\|_{\mathcal{L}(H)} \le 1$ for all $t \ge 0$;
* **(b)** $S_A(t_1 + t_2) = S_A(t_1) \circ S_A(t_2)\ \forall t_1, t_2 \ge 0$ and $S_A(0) = I$;
* **(c)** $\lim_{t \to 0^+} \lvert S_A(t) u_0 - u_0\rvert = 0\ \forall u_0 \in H$.

Such a family is called a **continuous semigroup of contractions**. A remarkable result of Hille and Yosida asserts the **converse**: every continuous semigroup of contractions $S(t)$ on $H$ arises as $S(t) = S_A(t)$ for a unique maximal monotone $A$. This is a *bijective correspondence* between maximal monotone operators and continuous semigroups of contractions.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Adding a constant)</span></p>

The shifted problem $\frac{du}{dt} + Au + \lambda u = 0$ reduces to (6) via $v(t) = e^{\lambda t} u(t)$, which satisfies $\frac{dv}{dt} + Av = 0$.

</div>

### 7.3 Regularity

For $u_0 \in D(A)$, the Hille–Yosida solution lies in $C^1([0, +\infty); H) \cap C([0, +\infty); D(A))$. With *more regular* initial data, the solution gains corresponding regularity.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Iterated domains $D(A^k)$)</span></p>

For $k \ge 2$ define inductively

$$
D(A^k) = \lbrace v \in D(A^{k-1})\,;\ Av \in D(A^{k-1})\rbrace.
$$

$D(A^k)$ is a Hilbert space for the scalar product

$$
(u, v)_{D(A^k)} = \sum_{j=0}^k (A^j u, A^j v),\quad \lvert u\rvert_{D(A^k)} = \Big(\sum_{j=0}^k \lvert A^j u\rvert^2\Big)^{1/2}.
$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 7.5</span><span class="math-callout__name">(Higher regularity)</span></p>

Assume $u_0 \in D(A^k)$ for some $k \ge 2$. The Hille–Yosida solution $u$ of (6) satisfies

$$
\boxed{\;u \in C^{k-j}([0, +\infty); D(A^j))\quad \forall j = 0, 1, \ldots, k.\;}
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

**Case $k = 2$.** Consider $H_1 = D(A)$ as Hilbert space (graph norm), and $A_1 : D(A_1) = D(A^2) \subset H_1 \to H_1$, $A_1 u = Au$. One checks $A_1$ is maximal monotone in $H_1$. Apply Theorem 7.4 in $H_1$: there exists $u \in C^1([0, +\infty); H_1) \cap C([0, +\infty); D(A_1))$ with $\frac{du}{dt} + A_1 u = 0$. By uniqueness this is the Hille–Yosida solution. Since $A \in \mathcal{L}(H_1, H)$, $\frac{d}{dt}(Au) = A\frac{du}{dt}$, so $u \in C^2([0, +\infty); H)$ and $\frac{d}{dt}\big(\frac{du}{dt}\big) + A\big(\frac{du}{dt}\big) = 0$.

**Inductive step $k \ge 3$.** Set $v = \frac{du}{dt}$; from the case $k = 2$, $v$ satisfies $v' + Av = 0$ with $v(0) = -Au_0 \in D(A^{k-1})$. By induction, $v \in C^{k-1-j}([0, +\infty); D(A^j))\ \forall j = 0, \ldots, k-1$, equivalently $u \in C^{k-j}([0, +\infty); D(A^j))\ \forall j = 0, \ldots, k-1$. Finally use $\frac{du}{dt} \in C([0, +\infty); D(A^{k-1}))$ and $\frac{du}{dt} + Au = 0$ to get $Au \in C([0, +\infty); D(A^{k-1}))$, i.e., $u \in C([0, +\infty); D(A^k))$. $\square$

</details>
</div>

### 7.4 The Self-Adjoint Case

Let $A : D(A) \subset H \to H$ be unbounded with $\overline{D(A)} = H$. Identifying $H^\star \simeq H$, view $A^\star : D(A^\star) \subset H \to H$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Symmetric vs. self-adjoint)</span></p>

* $A$ is **symmetric** if $(Au, v) = (u, Av)\ \forall u, v \in D(A)$.
* $A$ is **self-adjoint** if $D(A^\star) = D(A)$ and $A^\star = A$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Symmetric ≠ self-adjoint for unbounded operators)</span></p>

For *bounded* operators, symmetric and self-adjoint coincide. For *unbounded* operators there is a subtle difference: every self-adjoint operator is symmetric, but the converse is false. $A$ symmetric just means $A \subset A^\star$ (i.e., $D(A) \subset D(A^\star)$ and $A^\star = A$ on $D(A)$); it may happen that $D(A) \neq D(A^\star)$. **However, under maximal monotonicity, the two notions coincide:**

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7.6</span><span class="math-callout__name">(Maximal monotone + symmetric ⇒ self-adjoint)</span></p>

Let $A$ be maximal monotone and symmetric. Then $A$ is self-adjoint.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Let $J_1 = (I + A)^{-1}$. We first show $J_1$ is self-adjoint as a bounded operator. Set $u_1 = J_1 u, v_1 = J_1 v$, so $u_1 + Au_1 = u, v_1 + Av_1 = v$. By symmetry of $A$, $(u_1, Av_1) = (Au_1, v_1)$, hence $(u_1, v) = (u, v_1)$, i.e., $(J_1 u, v) = (u, J_1 v)$.

Now let $u \in D(A^\star)$ and set $f = u + A^\star u$. For any $v \in D(A)$, $(f, v) = (u, v + Av)$, i.e., $(f, J_1 w) = (u, w)\ \forall w \in H$. By self-adjointness of $J_1$, $(J_1 f, w) = (u, w)$, so $u = J_1 f \in D(A)$. Hence $D(A^\star) \subset D(A)$, and combined with the reverse inclusion (always true for symmetric), $D(A^\star) = D(A)$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Adjoint of monotone ≠ monotone)</span></p>

If $A$ is monotone (even symmetric monotone), $A^\star$ need not be monotone. One can show:

$$
A \text{ maximal monotone} \iff A^\star \text{ maximal monotone} \iff A \text{ closed},\ D(A) \text{ dense},\ A \text{ and } A^\star \text{ both monotone}.
$$

</div>

#### Smoothing effect: parabolic regularity

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 7.7</span><span class="math-callout__name">(Smoothing for self-adjoint maximal monotone)</span></p>

Let $A$ be a self-adjoint maximal monotone operator. For *every* $u_0 \in H$ there exists a *unique* function

$$
u \in C([0, +\infty); H) \cap C^1((0, +\infty); H) \cap C((0, +\infty); D(A))
$$

with $\frac{du}{dt} + Au = 0$ on $(0, +\infty)$, $u(0) = u_0$. Moreover

$$
\boxed{\;\lvert u(t)\rvert \le \lvert u_0\rvert,\qquad \Big\lvert\frac{du}{dt}(t)\Big\rvert = \lvert Au(t)\rvert \le \frac{1}{t}\lvert u_0\rvert\quad \forall t > 0,\;}
$$

and $u \in C^k((0, +\infty); D(A^\ell))$ for all integers $k, \ell$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Difference from Theorem 7.4)</span></p>

The contrast with the general Hille–Yosida theorem: here $u_0 \in H$ (not $D(A)$); in exchange, the solution is smooth *away from* $t = 0$ — but $\lvert\frac{du}{dt}(t)\rvert$ may "blow up" as $t \to 0^+$. This is the abstract analogue of *parabolic smoothing* (the heat equation produces $C^\infty$ solutions for $t > 0$ from rough initial data).

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof (sketch)</summary>

**Uniqueness.** Same as Theorem 7.4: $\varphi(t) = \lvert u(t) - \overline{u}(t)\rvert^2$ is nonincreasing on $(0, +\infty)$, continuous on $[0, +\infty)$, $\varphi(0) = 0$, hence $\varphi \equiv 0$.

**Step 1 (existence for $u_0 \in D(A^2)$).** Show

$$
\Big\lvert\frac{du}{dt}(t)\Big\rvert \le \frac{1}{t}\lvert u_0\rvert. \tag{27}
$$

Use the approximate problem $\frac{du_\lambda}{dt} + A_\lambda u_\lambda = 0$. Note $A_\lambda^\star = A_\lambda$ when $A^\star = A$. Take the scalar product of $\frac{du_\lambda}{dt} + A_\lambda u_\lambda = 0$ with $u_\lambda$ and integrate over $[0, T]$:

$$
\tfrac{1}{2}\lvert u_\lambda(T)\rvert^2 + \int_0^T (A_\lambda u_\lambda, u_\lambda)\,dt = \tfrac{1}{2}\lvert u_0\rvert^2. \tag{29}
$$

Take the scalar product with $t\frac{du_\lambda}{dt}$ and integrate, using self-adjointness ($A_\lambda^\star = A_\lambda$):

$$
\int_0^T \Big\lvert\frac{du_\lambda}{dt}(t)\Big\rvert^2 t\,dt + \tfrac{1}{2}(A_\lambda u_\lambda(T), u_\lambda(T)) T - \tfrac{1}{2}\int_0^T (A_\lambda u_\lambda, u_\lambda)\,dt = 0.
$$

Combining with $(29)$ and using monotonicity of $t \mapsto \lvert\frac{du_\lambda}{dt}(t)\rvert$ (Lemma 7.1):

$$
\tfrac{1}{2}\lvert u_\lambda(T)\rvert^2 + T(A_\lambda u_\lambda(T), u_\lambda(T)) + T^2 \Big\lvert\frac{du_\lambda}{dt}(T)\Big\rvert^2 \le \tfrac{1}{2}\lvert u_0\rvert^2.
$$

Hence $\lvert\frac{du_\lambda}{dt}(T)\rvert \le \frac{1}{T}\lvert u_0\rvert$. Pass to the limit $\lambda \to 0$.

**Step 2 (general $u_0 \in H$).** Approximate by $u_{0n} \in D(A^2)$ (which is dense in $H$ since $D(A^2)$ is dense in $D(A)$ which is dense in $H$). Theorem 7.4 + Step 1 give

$$
\lvert u_n(t) - u_m(t)\rvert \le \lvert u_{0n} - u_{0m}\rvert,\qquad \Big\lvert\frac{du_n}{dt}(t) - \frac{du_m}{dt}(t)\Big\rvert \le \frac{1}{t}\lvert u_{0n} - u_{0m}\rvert.
$$

So $u_n$ converges uniformly on $[0, +\infty)$ and $\frac{du_n}{dt}$ converges uniformly on every $[\delta, +\infty),\ \delta > 0$. The limit satisfies (6) on $(0, +\infty)$.

**Higher regularity.** By induction on $k$: in $\widetilde{H} = D(A^{k-1})$ with $\widetilde{A}u = Au$ for $u \in D(A^k)$, $\widetilde{A}$ is maximal monotone and symmetric in $\widetilde{H}$, hence self-adjoint. Applying the result on $\widetilde{H}$ to initial data $u(\varepsilon)$ (already in $D(A^{k-1}) = \widetilde{H}$), gives $u \in C((\varepsilon, +\infty); D(A^k))$. $\square$

</details>
</div>

### Comments on Chapter 7

#### 1. The Hille–Yosida theorem in Banach spaces

Hille–Yosida extends to Banach spaces. Let $E$ be a Banach space, $A : D(A) \subset E \to E$ unbounded. $A$ is **$m$-accretive** if $\overline{D(A)} = E$ and for every $\lambda > 0$, $I + \lambda A$ is bijective from $D(A)$ onto $E$ with $\|(I + \lambda A)^{-1}\|_{\mathcal{L}(E)} \le 1$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 7.8</span><span class="math-callout__name">(Hille–Yosida in Banach spaces)</span></p>

Let $A$ be $m$-accretive on $E$. Given $u_0 \in D(A)$ there exists a unique

$$
u \in C^1([0, +\infty); E) \cap C([0, +\infty); D(A))
$$

with $\frac{du}{dt} + Au = 0$ on $[0, +\infty)$ and $u(0) = u_0$. Moreover

$$
\|u(t)\| \le \|u_0\|,\qquad \Big\|\frac{du}{dt}(t)\Big\| = \|Au(t)\| \le \|Au_0\|\quad \forall t \ge 0.
$$

The map $u_0 \mapsto u(t)$, extended by continuity to all of $E$, is a continuous semigroup of contractions $S_A(t)$ on $E$, and conversely every such semigroup arises this way for a unique $m$-accretive $A$.

</div>

References: Lax, Pazy, Goldstein, Davies, Yosida, Reed–Simon Vol. 2, Tanabe, Dunford–Schwartz Vol. 1, Schechter, Friedman, Dautray–Lions Ch. XVII, Balakrishnan, Kato, Rudin.

#### 2. The exponential formula

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 7.9</span><span class="math-callout__name">(Exponential formula)</span></p>

Assume $A$ is $m$-accretive. For every $u_0 \in D(A)$ the solution $u$ of (38) is given by

$$
\boxed{\;u(t) = \lim_{n \to +\infty} \Big[\Big(I + \frac{t}{n} A\Big)^{-1}\Big]^n u_0.\;}
$$

</div>

(Yosida, Pazy.) This corresponds, in numerical analysis, to convergence of an *implicit time discretization*: divide $[0, t]$ into $n$ intervals of length $\Delta t = t/n$ and inductively solve $\frac{u_{j+1} - u_j}{\Delta t} + A u_{j+1} = 0$, giving $u_n = (I + \Delta t A)^{-n} u_0 \to u(t)$.

#### 3. Analytic semigroups

Theorem 7.7 is a first step toward the theory of **analytic semigroups**. Yosida, Kato, Reed–Simon Vol. 2, Friedman, Pazy, Tanabe.

#### 4. Inhomogeneous and nonlinear equations

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 7.10</span><span class="math-callout__name">(Inhomogeneous Hille–Yosida)</span></p>

Assume $A$ is $m$-accretive. For every $u_0 \in D(A)$ and $f \in C^1([0, T]; E)$ there exists a unique

$$
u \in C^1([0, T]; E) \cap C([0, T]; D(A))
$$

solving

$$
\frac{du}{dt}(t) + Au(t) = f(t)\text{ on } [0, T],\quad u(0) = u_0,
$$

given by the **Duhamel formula**

$$
\boxed{\;u(t) = S_A(t) u_0 + \int_0^t S_A(t - s) f(s)\,ds.\;}
$$

</div>

If $f \in L^1((0, T); E)$ only, the Duhamel formula still defines a *generalized* solution. References: Kato, Pazy, Martin, Tanabe.

In physical applications one often encounters **semilinear** equations $\frac{du}{dt} + Au = F(u)$ with $F$ nonlinear. References: Martin, Cazenave–Haraux. Some results extend to nonlinear $m$-accretive operators (Brezis, Barbu).

## Chapter 8: Sobolev Spaces and the Variational Formulation of Boundary Value Problems in One Dimension

The functional-analytic machinery of Chapters 1–7 was abstract. Chapter 8 cashes it in: it builds the **Sobolev space** $W^{1,p}(I)$ on a one-dimensional interval and uses Hilbert-space methods (Lax–Milgram, Stampacchia) to solve elliptic boundary value problems by the *variational method*. The chapter is a one-dimensional warm-up for Chapter 9, where the same scheme is run in $\mathbb{R}^N$.

The recurring pattern is the **four-step variational program** for solving an elliptic BVP:

* **Step A.** Define a notion of *weak solution* that requires less regularity than a classical $C^2$ solution. The natural setting is a Sobolev space.
* **Step B.** Prove existence and uniqueness of a weak solution by a variational method (Lax–Milgram or Stampacchia).
* **Step C.** *Regularity*: show the weak solution is more regular than required by its definition (e.g., $C^2$).
* **Step D.** Recover a *classical* solution from a regular weak solution.

Steps A, B are infrastructure; Step C is the deepest. We end with the **maximum principle** and the **spectral theorem** for Sturm–Liouville operators (the basis of "Fourier-type" expansions in PDE).

### 8.1 Motivation

Consider the model problem: given $f \in C([a, b])$, find $u$ satisfying

$$
\begin{cases} -u'' + u = f \text{ on } [a, b], \\ u(a) = u(b) = 0. \end{cases} \tag{1}
$$

A *classical* (*strong*) solution is $u \in C^2([a, b])$ satisfying (1) in the usual sense. Multiplying by $\varphi \in C^1([a, b])$ with $\varphi(a) = \varphi(b) = 0$ and integrating by parts:

$$
\int_a^b u'\varphi' + \int_a^b u\varphi = \int_a^b f\varphi\quad \forall \varphi \in C^1([a, b]),\ \varphi(a) = \varphi(b) = 0. \tag{2}
$$

(2) makes sense as soon as $u \in C^1([a, b])$ — only *one* derivative is needed instead of two. In fact (2) makes sense as soon as $u, u' \in L^1(a, b)$, where $u'$ has a meaning yet to be made precise. We provisionally call a $C^1$ function $u$ that satisfies (2) a **weak solution** of (1).

The variational program in Steps A–D above carries this idea through. Note: **Step D is easy** — if $u \in C^2$, $u(a) = u(b) = 0$, satisfies (2), then integrating by parts gives $\int_a^b(-u'' + u - f)\varphi = 0\ \forall \varphi \in C^1_c((a, b))$, hence $-u'' + u = f$ a.e. by Corollary 4.15 (du Bois-Reymond). The hard work is Steps A and C.

### 8.2 The Sobolev Space $W^{1, p}(I)$

Let $I = (a, b)$ be an open (possibly unbounded) interval and $1 \le p \le \infty$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The Sobolev space $W^{1, p}(I)$)</span></p>

$$
\boxed{\;W^{1, p}(I) = \Big\lbrace u \in L^p(I)\,;\ \exists g \in L^p(I)\text{ such that }\int_I u\varphi' = -\int_I g\varphi\ \forall \varphi \in C^1_c(I)\Big\rbrace.\;}
$$

We set $H^1(I) = W^{1, 2}(I)$. For $u \in W^{1, p}$ we denote $u' = g$ (well defined a.e. by Corollary 4.24).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Test functions and consistency with classical derivatives)</span></p>

* The functions $\varphi \in C^1_c(I)$ are called **test functions**. We could equally use $C^\infty_c(I)$ — for $\varphi \in C^1_c(I)$, $\rho_n \star \varphi \in C^\infty_c(I)$ for $n$ large and $\rho_n \star \varphi \to \varphi$ in $C^1$ (Section 4.4).
* If $u \in C^1(I) \cap L^p(I)$ and $u' \in L^p(I)$ (in the classical sense), then $u \in W^{1, p}(I)$ and the classical derivative coincides with $u'$ in the $W^{1, p}$ sense — the notation is consistent. In particular, on a bounded $I$, $C^1(\bar I) \subset W^{1, p}$ for all $1 \le p \le \infty$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Distributional viewpoint)</span></p>

To define $W^{1, p}$ one can use the language of *distributions* (Schwartz, Knapp). Every $u \in L^p(I)$ has a *distributional* derivative — an element of $\mathcal{D}'(I)$. We say $u \in W^{1, p}$ when this distributional derivative happens to lie in $L^p$, making $W^{1, p}$ a subspace of $\mathcal{D}'(I)$. When $I = \mathbb{R}$ and $p = 2$, $W^{1, 2}$ can also be defined via the Fourier transform.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Examples</span><span class="math-callout__name">($W^{1, p}$-membership)</span></p>

Let $I = (-1, 1)$.

(i) The function $u(x) = \lvert x\rvert$ belongs to $W^{1, p}(I)$ for every $1 \le p \le \infty$ with

$$
u'(x) = g(x) = \begin{cases} +1 & 0 < x < 1, \\ -1 & -1 < x < 0. \end{cases}
$$

More generally, a continuous piecewise $C^1$ function on $\bar I$ belongs to $W^{1, p}$ for all $p$.

(ii) The function $g$ above does *not* belong to $W^{1, p}(I)$ for any $p$ — its distributional derivative is the Dirac measure $2\delta_0$, not in $L^p$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">($W^{1, p}$ and $H^1$ norms)</span></p>

$$
\boxed{\;\|u\|_{W^{1, p}} = \|u\|_{L^p} + \|u'\|_{L^p},\;}
$$

or equivalently (for $1 < p < \infty$) $(\|u\|_{L^p}^p + \|u'\|_{L^p}^p)^{1/p}$. $H^1$ is equipped with the scalar product

$$
(u, v)_{H^1} = (u, v)_{L^2} + (u', v')_{L^2} = \int_a^b (uv + u'v').
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8.1</span><span class="math-callout__name">(Banach / reflexivity / separability of $W^{1, p}$)</span></p>

The space $W^{1, p}$ is a Banach space for $1 \le p \le \infty$, *reflexive* for $1 < p < \infty$, and *separable* for $1 \le p < \infty$. $H^1$ is a separable Hilbert space.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

**Banach.** If $(u_n)$ is Cauchy in $W^{1, p}$, $(u_n)$ and $(u'_n)$ are Cauchy in $L^p$; let $u_n \to u, u'_n \to g$ in $L^p$. Pass to the limit in $\int u_n \varphi' = -\int u'_n \varphi$ to get $u \in W^{1, p}$ with $u' = g$.

**Reflexive (for $1 < p < \infty$).** $E = L^p(I) \times L^p(I)$ is reflexive; the operator $T : W^{1, p} \to E$, $Tu = [u, u']$, is an isometry, so $T(W^{1, p})$ is a closed subspace of $E$, hence reflexive (Proposition 3.20). Therefore $W^{1, p}$ is reflexive.

**Separable.** Since $E$ is separable, so is $T(W^{1, p})$ (Proposition 3.25). $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why reflexivity matters here)</span></p>

Reflexivity of $W^{1, p}$ ($1 < p < \infty$) is a *considerable* advantage — in the calculus of variations one prefers $W^{1, p}$ over $C^1$, which is *not* reflexive. Existence of minimizers is then easily established (Corollary 3.23).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Useful convergence criterion)</span></p>

If $(u_n) \subset W^{1, p}$ with $u_n \to u$ and $u'_n \to g$ in $L^p$, then $u \in W^{1, p}$ and $\|u_n - u\|_{W^{1, p}} \to 0$. When $1 < p \le \infty$, it suffices that $u_n \to u$ in $L^p$ and $\|u'_n\|_{L^p}$ stay *bounded* (Exercise 8.2).

</div>

#### Functions in $W^{1, p}$ are primitives

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8.2</span><span class="math-callout__name">(Continuous representative; FTC for $W^{1, p}$)</span></p>

Let $u \in W^{1, p}(I)$, $1 \le p \le \infty$, $I$ bounded or unbounded. Then there exists $\tilde u \in C(\bar I)$ such that $u = \tilde u$ a.e. on $I$ and

$$
\boxed{\;\tilde u(x) - \tilde u(y) = \int_y^x u'(t)\,dt\quad \forall x, y \in \bar I.\;}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Replace $u$ by $\tilde u$)</span></p>

Theorem 8.2 says every $u \in W^{1, p}$ admits a *unique* continuous representative on $\bar I$. We henceforth identify $u$ with this $\tilde u$ — useful when we need $u(x)$ pointwise. Note: "$u$ has a continuous representative" is *not* the same as "$u$ is continuous a.e." If $u \in W^{1, p}$ and $u' \in C(\bar I)$, then $u \in C^1(\bar I)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 8.1</span><span class="math-callout__name">(Functions with vanishing derivative)</span></p>

Let $f \in L^1_{\mathrm{loc}}(I)$ with $\int_I f\varphi' = 0\ \forall \varphi \in C^1_c(I)$. Then $f = C$ a.e. on $I$ for some constant $C$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 8.2</span><span class="math-callout__name">(Primitives are in $W^{1, p}$)</span></p>

Let $g \in L^1_{\mathrm{loc}}(I)$ and $y_0 \in I$. Set $v(x) = \int_{y_0}^x g(t)\,dt$. Then $v \in C(I)$ and $\int_I v\varphi' = -\int_I g\varphi\ \forall \varphi \in C^1_c(I)$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 8.2</summary>

Fix $y_0 \in I$ and set $\bar u(x) = \int_{y_0}^x u'(t)\,dt$. By Lemma 8.2, $\int_I \bar u \varphi' = -\int_I u'\varphi\ \forall \varphi \in C^1_c$. Hence $\int_I (u - \bar u)\varphi' = 0\ \forall \varphi \in C^1_c$, so by Lemma 8.1, $u - \bar u = C$ a.e. The function $\tilde u(x) = \bar u(x) + C$ is the continuous representative. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8.3</span><span class="math-callout__name">($W^{1, p}$ via dual estimate, $1 < p \le \infty$)</span></p>

For $u \in L^p$ with $1 < p \le \infty$, the following are equivalent:

(i) $u \in W^{1, p}$;

(ii) there exists $C$ such that $\big\lvert \int_I u\varphi'\big\rvert \le C\|\varphi\|_{L^{p'}}\ \forall \varphi \in C^1_c$.

Furthermore $C = \|u'\|_{L^p}$ in (ii).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Absolutely continuous and $BV$ functions)</span></p>

For $p = 1$: (i) $\Rightarrow$ (ii) but not the converse. Functions satisfying (i) (i.e., $W^{1, 1}$) are the **absolutely continuous** functions on $\bar I$:

$$
(AC)\quad \forall \varepsilon > 0\ \exists \delta > 0\ \forall \text{ disjoint } (a_k, b_k) \subset I\text{ with }\sum (b_k - a_k) < \delta:\ \sum \lvert u(b_k) - u(a_k)\rvert < \varepsilon.
$$

Functions satisfying (ii) with $p = 1$ are the **functions of bounded variation** ($BV$):

* differences of two bounded nondecreasing functions on $I$;
* satisfying $(BV)\quad \exists C: \sum_{i=0}^{k-1} \lvert u(t_{i+1}) - u(t_i)\rvert \le C\ \forall t_0 < \cdots < t_k$;
* functions in $L^1$ with distributional derivative a *bounded measure*.

$BV$ functions need not have a continuous representative.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8.4</span><span class="math-callout__name">($W^{1, \infty}$ = Lipschitz)</span></p>

$u \in L^\infty(I)$ belongs to $W^{1, \infty}(I)$ iff $u$ has a Lipschitz continuous representative: $\lvert u(x) - u(y)\rvert \le C\lvert x - y\rvert$ for a.e. $x, y \in I$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8.5</span><span class="math-callout__name">($W^{1, p}$ via translation, $1 < p < \infty$)</span></p>

For $u \in L^p(\mathbb{R})$ with $1 < p < \infty$, the following are equivalent:

(i) $u \in W^{1, p}(\mathbb{R})$;

(ii) there exists $C$ such that $\|\tau_h u - u\|_{L^p(\mathbb{R})} \le C\lvert h\rvert\ \forall h$,

where $(\tau_h u)(x) = u(x + h)$. Moreover $C = \|u'\|_{L^p}$.

</div>

#### The extension theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8.6</span><span class="math-callout__name">(Extension operator)</span></p>

Let $1 \le p \le \infty$. There exists a bounded linear *extension operator* $P : W^{1, p}(I) \to W^{1, p}(\mathbb{R})$ with

(i) $Pu\rvert_I = u$ for every $u \in W^{1, p}(I)$;

(ii) $\|Pu\|\_{L^p(\mathbb{R})} \le C\|u\|\_{L^p(I)}$;

(iii) $\|Pu\|\_{W^{1, p}(\mathbb{R})} \le C\|u\|\_{W^{1, p}(I)}$,

where $C$ depends only on $\lvert I\rvert \le \infty$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof (sketch)</summary>

**Half-line $I = (0, \infty)$ — extension by reflection.** Define $u^\star(x) = u(\lvert x\rvert)$. Setting $v(x) = u'(x)$ for $x > 0$ and $v(x) = -u'(-x)$ for $x < 0$, one checks $v \in L^p(\mathbb{R})$ and $u^\star(x) - u^\star(0) = \int_0^x v$, so $u^\star \in W^{1, p}(\mathbb{R})$ with norm $\le 2\|u\|$.

**Bounded $I = (0, 1)$.** Use a smooth cutoff $\eta(x) = 1$ for $x < 1/4$, $0$ for $x > 3/4$. Write $u = \eta u + (1 - \eta)u$. Extend $\eta u$ to $(0, \infty)$ by zero (possible since $\eta = 0$ near $1$), then reflect about $0$. Symmetrically for $(1 - \eta)u$ about $1$. Sum the two extensions. $\square$

</details>
</div>

#### Density and the basic Sobolev embedding

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8.7</span><span class="math-callout__name">(Density)</span></p>

Let $u \in W^{1, p}(I)$ with $1 \le p < \infty$. There exists a sequence $(u_n) \subset C^\infty_c(\mathbb{R})$ with $u_n\rvert_I \to u$ in $W^{1, p}(I)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(No $C^\infty_c(I)$ density!)</span></p>

In general there is *no* sequence in $C^\infty_c(I)$ converging to $u$ in $W^{1, p}(I)$ — see Section 8.3 ($W^{1, p}_0$ is a *strictly* smaller subspace). Contrast: in $L^p$, $C^\infty_c(I)$ *is* dense (Corollary 4.23).

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 8.7</summary>

By Theorem 8.6 we may assume $I = \mathbb{R}$. Use convolution + cutoff.

**(a) Convolution lemma.** For $\rho \in L^1(\mathbb{R})$ and $v \in W^{1, p}(\mathbb{R})$, $\rho \star v \in W^{1, p}(\mathbb{R})$ and $(\rho \star v)' = \rho \star v'$. (Direct from Propositions 4.16 and 4.20.)

**(b) Cutoff.** Fix $\zeta \in C^\infty_c(\mathbb{R})$ with $\zeta = 1$ on $\lvert x\rvert < 1$, $0$ on $\lvert x\rvert \ge 2$, and set $\zeta_n(x) = \zeta(x/n)$. Then $\zeta_n f \to f$ in $L^p$ for $f \in L^p$.

**(c) Combination.** Set $u_n = \zeta_n(\rho_n \star u)$ with $\rho_n$ mollifiers. Then $\|u_n - u\|_p \to 0$ and $\|u'_n - u'\|_p \to 0$ (using $u'_n = \zeta'_n(\rho_n \star u) + \zeta_n(\rho_n \star u')$, with $\|\zeta'_n\|_\infty \le C/n \to 0$). $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8.8</span><span class="math-callout__name">(Sobolev embedding $W^{1, p} \hookrightarrow L^\infty$, dim 1)</span></p>

There exists a constant $C$ (depending only on $\lvert I\rvert \le \infty$) such that

$$
\boxed{\;\|u\|_{L^\infty(I)} \le C\|u\|_{W^{1, p}(I)}\quad \forall u \in W^{1, p}(I),\ \forall 1 \le p \le \infty.\;}
$$

In other words, $W^{1, p}(I) \subset L^\infty(I)$ with continuous injection. If $I$ is *bounded*:

* the injection $W^{1, p}(I) \subset C(\bar I)$ is **compact** for $1 < p \le \infty$,
* the injection $W^{1, 1}(I) \subset L^q(I)$ is **compact** for $1 \le q < \infty$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

**Step 1: $\|u\|_\infty \le C\|u\|_{W^{1, p}(\mathbb{R})}$ for $v \in C^1_c(\mathbb{R})$.** With $G(s) = \lvert s\rvert^{p-1}s$, $w = G(v) \in C^1_c$ and $w' = G'(v)v' = p\lvert v\rvert^{p-1}v'$. Hence $G(v(x)) = \int_{-\infty}^x p\lvert v\rvert^{p-1}v'\,dt$, so $\lvert v(x)\rvert^p \le p\|v\|_p^{p-1}\|v'\|_p$, giving $\|v\|_\infty \le C\|v\|_{W^{1, p}}$ with $C$ universal.

**Step 2: density.** Theorem 8.7 gives $u_n \in C^1_c(\mathbb{R})$ with $u_n \to u$ in $W^{1, p}$, so $(u_n)$ is Cauchy in $L^\infty$ and $u_n \to u$ in $L^\infty$.

**Step 3 (compactness, $I$ bounded, $1 < p \le \infty$).** The unit ball $\mathcal{H} \subset W^{1, p}$ satisfies $\lvert u(x) - u(y)\rvert = \big\lvert\int_y^x u'\big\rvert \le \|u'\|_p\lvert x - y\rvert^{1/p'} \le \lvert x - y\rvert^{1/p'}$ for $u \in \mathcal{H}$. By Ascoli–Arzelà (Theorem 4.25), $\mathcal{H}$ has compact closure in $C(\bar I)$.

**Step 4 (compactness, $W^{1, 1} \to L^q$).** Apply Kolmogorov–M. Riesz–Fréchet (Theorem 4.26) using $\|\tau_h f - f\|_{L^1(\mathbb{R})} \le \|f'\|_{L^1}\lvert h\rvert$ (Proposition 8.5) and $L^q$ interpolation. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Continuous but not compact)</span></p>

The injection $W^{1, p}(I) \subset C(\bar I)$ is continuous but *never compact* — even on bounded $I$ (Exercise 8.2). However, by **Helly's selection theorem** (Kolmogorov–Fomin), every bounded sequence in $W^{1, 1}$ has a *pointwise convergent* subsequence. For unbounded $I$ and $1 < p \le \infty$, $W^{1, p}(I) \subset L^\infty(I)$ is continuous but *never compact*; nevertheless, every bounded sequence has a subsequence converging in $L^\infty(J)$ on every bounded $J \subset I$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 8.9</span><span class="math-callout__name">(Decay at infinity)</span></p>

If $I$ is unbounded and $u \in W^{1, p}(I)$ with $1 \le p < \infty$, then

$$
\lim_{\substack{x \in I \\ \lvert x\rvert \to \infty}} u(x) = 0.
$$

</div>

#### Differentiation rules

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 8.10</span><span class="math-callout__name">(Product rule, integration by parts)</span></p>

If $u, v \in W^{1, p}(I)$, $1 \le p \le \infty$, then $uv \in W^{1, p}(I)$ and

$$
\boxed{\;(uv)' = u'v + uv',\;}
$$

with the integration-by-parts formula

$$
\int_y^x u'v = u(x)v(x) - u(y)v(y) - \int_y^x uv'\quad \forall x, y \in \bar I.
$$

</div>

(Note: in general $L^p$ is *not* an algebra under multiplication; $W^{1, p}(I)$ *is* — the Sobolev embedding $W^{1, p} \hookrightarrow L^\infty$ is what makes it work.)

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 8.11</span><span class="math-callout__name">(Chain rule)</span></p>

Let $G \in C^1(\mathbb{R})$ with $G(0) = 0$, and $u \in W^{1, p}(I)$, $1 \le p \le \infty$. Then

$$
G \circ u \in W^{1, p}(I)\quad \text{and}\quad (G \circ u)' = (G' \circ u) u'.
$$

(The condition $G(0) = 0$ is unnecessary if $I$ is bounded or $p = \infty$; it is essential if $I$ is unbounded and $1 \le p < \infty$.)

</div>

#### The spaces $W^{m, p}$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($W^{m, p}$)</span></p>

For an integer $m \ge 2$ and $1 \le p \le \infty$, define inductively

$$
W^{m, p}(I) = \lbrace u \in W^{m-1, p}(I)\,;\ u' \in W^{m-1, p}(I)\rbrace,\qquad H^m(I) = W^{m, 2}(I).
$$

Equivalently, $u \in W^{m, p}$ iff there exist $g_1, \ldots, g_m \in L^p$ with $\int_I u D^j\varphi = (-1)^j\int_I g_j \varphi\ \forall \varphi \in C^\infty_c$, $j = 1, \ldots, m$. We write $Du = g_1, D^2 u = g_2, \ldots$ The norm is

$$
\|u\|_{W^{m, p}} = \|u\|_p + \sum_{\alpha = 1}^m \|D^\alpha u\|_p,
$$

equivalent to $\|u\|_p + \|D^m u\|_p$.

</div>

If $I$ is bounded, $W^{m, p}(I) \subset C^{m-1}(\bar I)$ continuously, and compactly for $1 < p \le \infty$.

### 8.3 The Space $W^{1, p}_0$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($W^{1, p}_0$)</span></p>

For $1 \le p < \infty$, $W^{1, p}_0(I)$ is the **closure of $C^1_c(I)$** in $W^{1, p}(I)$. Set $H^1_0(I) = W^{1, 2}_0(I)$. The space $W^{1, p}_0$ is equipped with the norm of $W^{1, p}$, $H^1_0$ with the scalar product of $H^1$.

$W^{1, p}_0(I)$ is a separable Banach space, reflexive for $1 < p < \infty$. $H^1_0$ is a separable Hilbert space.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Whole line)</span></p>

When $I = \mathbb{R}$, $C^\infty_c(\mathbb{R})$ is dense in $W^{1, p}(\mathbb{R})$ (Theorem 8.7), so $W^{1, p}_0(\mathbb{R}) = W^{1, p}(\mathbb{R})$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8.12</span><span class="math-callout__name">($W^{1, p}_0$ ↔ vanishing on boundary)</span></p>

Let $u \in W^{1, p}(I)$. Then $u \in W^{1, p}_0(I)$ iff $u = 0$ on $\partial I$ (i.e., the continuous representative of $u$ vanishes at the endpoints of $I$).

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

**($\Rightarrow$)** Approximate $u$ by $u_n \in C^1_c(I)$ in $W^{1, p}$. Then $u_n \to u$ uniformly on $\bar I$ (Theorem 8.8), so $u = 0$ on $\partial I$.

**($\Leftarrow$)** Assume $u = 0$ on $\partial I$. Fix $G \in C^1(\mathbb{R})$ with $G(t) = 0$ for $\lvert t\rvert \le 1$ and $G(t) = t$ for $\lvert t\rvert \ge 2$, with $\lvert G(t)\rvert \le \lvert t\rvert$. Set $u_n = (1/n) G(nu)$. Then $u_n \in W^{1, p}$ (Corollary 8.11) and $\mathrm{supp}\,u_n \subset \lbrace x \in I\,;\ \lvert u(x)\rvert \ge 1/n\rbrace$, a compact subset of $I$ (since $u = 0$ on $\partial I$ and $u(x) \to 0$ at $\infty$). So $u_n \in W^{1, p}_0$, and $u_n \to u$ in $W^{1, p}$ by dominated convergence. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8.13</span><span class="math-callout__name">(Poincaré's inequality)</span></p>

Suppose $I$ is **bounded**. Then there exists $C$ (depending on $\lvert I\rvert$) such that

$$
\boxed{\;\|u\|_{W^{1, p}(I)} \le C\|u'\|_{L^p(I)}\quad \forall u \in W^{1, p}_0(I).\;}
$$

In other words, on $W^{1, p}_0$ the quantity $\|u'\|_{L^p}$ is a norm equivalent to the $W^{1, p}$ norm.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

For $u \in W^{1, p}_0(I)$ with $I = (a, b)$, $u(a) = 0$, so $u(x) = \int_a^x u'$, hence $\|u\|_{L^\infty} \le \|u'\|_{L^1}$. Apply Hölder. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">($W^{m, p}_0$ and the dual)</span></p>

For $m \ge 2$, $W^{m, p}_0(I)$ is the closure of $C^\infty_c(I)$ in $W^{m, p}$, characterized by $u = Du = \cdots = D^{m-1}u = 0$ on $\partial I$. Note the difference

$$
W^{2, p}_0(I) = \lbrace u \in W^{2, p}\,;\ u = Du = 0\text{ on }\partial I\rbrace \neq W^{2, p}(I) \cap W^{1, p}_0(I) = \lbrace u \in W^{2, p}\,;\ u = 0\text{ on }\partial I\rbrace.
$$

The dual of $W^{1, p}_0(I)$ is denoted $W^{-1, p'}(I)$, of $H^1_0$ denoted $H^{-1}$. We identify $L^2$ with its dual but *not* $H^1_0$ with its dual: $H^1_0 \subset L^2 \subset H^{-1}$ with continuous and dense injections.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8.14</span><span class="math-callout__name">(Representation of $W^{-1, p'}$)</span></p>

Let $F \in W^{-1, p'}(I)$. There exist $f_0, f_1 \in L^{p'}(I)$ with

$$
\langle F, u\rangle = \int_I f_0 u + \int_I f_1 u'\quad \forall u \in W^{1, p}_0,\qquad \|F\|_{W^{-1, p'}} = \max\lbrace \|f_0\|_{p'}, \|f_1\|_{p'}\rbrace.
$$

When $I$ is bounded, one can take $f_0 = 0$. The element $F$ is usually identified with the *distribution* $f_0 - f'_1$.

</div>

### 8.4 Some Examples of Boundary Value Problems

We now run the variational program on a series of boundary value problems on $I = (0, 1)$.

#### Homogeneous Dirichlet

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Classical and weak solution of Dirichlet)</span></p>

Consider

$$
\begin{cases} -u'' + u = f \text{ on } I = (0, 1), \\ u(0) = u(1) = 0. \end{cases} \tag{14}
$$

A **classical solution** is $u \in C^2(\bar I)$ satisfying (14). A **weak solution** is $u \in H^1_0(I)$ satisfying

$$
\int_I u'v' + \int_I uv = \int_I fv\quad \forall v \in H^1_0(I). \tag{15}
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8.15</span><span class="math-callout__name">(Dirichlet's principle)</span></p>

Given $f \in L^2(I)$, there exists a *unique* $u \in H^1_0$ solving (15). It is given by

$$
\boxed{\;\min_{v \in H^1_0}\Big\lbrace \tfrac{1}{2}\int_I (v'^2 + v^2) - \int_I fv\Big\rbrace.\;}
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Apply Lax–Milgram (Corollary 5.8) in $H = H^1_0(I)$ with $a(u, v) = \int u'v' + \int uv = (u, v)_{H^1}$ and $\varphi(v) = \int fv$. $a$ is continuous, coercive (it is the $H^1$-scalar product), and symmetric, so the variational characterization holds. $\square$

</details>
</div>

**Steps C and D — Regularity.** If $f \in L^2$ and $u \in H^1_0$ solves (15), then $\int u'v' = \int(f - u)v\ \forall v \in C^1_c$, so $u' \in H^1$ (i.e., $u \in H^2$) since $f - u \in L^2$. If $f \in C(\bar I)$, $u'' = u - f \in C(\bar I)$, so $u \in C^2(\bar I)$ — a classical solution. More generally, $f \in H^k \Rightarrow u \in H^{k+2}$.

#### More Examples

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1</span><span class="math-callout__name">(Inhomogeneous Dirichlet)</span></p>

$-u'' + u = f$ on $I$, $u(0) = \alpha, u(1) = \beta$. Two methods:

* **Method 1** — substitution. Let $u_0$ be a smooth function with $u_0(0) = \alpha, u_0(1) = \beta$ (e.g., affine), and set $\tilde u = u - u_0$; reduce to the homogeneous case.
* **Method 2** — Stampacchia. Let $K = \lbrace v \in H^1\,;\ v(0) = \alpha, v(1) = \beta\rbrace$ (closed convex). Apply Theorem 5.6 to obtain $u \in K$ with $\int u'(v - u)' + \int u(v - u) \ge \int f(v - u)\ \forall v \in K$. Setting $v = u \pm w$ with $w \in H^1_0$ recovers the equation; minimization characterization $\tfrac{1}{2}\int(v'^2 + v^2) - \int fv$ on $K$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2</span><span class="math-callout__name">(Sturm–Liouville)</span></p>

$-(pu')' + qu = f$ on $I$, $u(0) = u(1) = 0$, with $p \in C^1(\bar I), p \ge \alpha > 0$, $q \in C(\bar I)$. Bilinear form $a(u, v) = \int pu'v' + \int quv$ on $H^1_0$. Continuous and symmetric; coercive if $q \ge 0$ (by Poincaré). Lax–Milgram gives $u \in H^1_0$ unique; regularity gives $u \in H^2$, classical if $f \in C$.

For the more general $-(pu')' + ru' + qu = f$, the form is *not* symmetric. It is coercive under conditions like $q \ge 1$ and $r^2 < 4\alpha$, or $q \ge 1, r' \le 2$. There is a slick **trick** to make it symmetric: let $R$ be a primitive of $r/p$ and set $\zeta = e^{-R}$; multiplying by $\zeta$ and using $\zeta' p + \zeta r = 0$ gives $-(\zeta p u')' + \zeta q u = \zeta f$, with symmetric form $a(u, v) = \int \zeta p u'v' + \int \zeta q uv$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3</span><span class="math-callout__name">(Homogeneous Neumann)</span></p>

$-u'' + u = f$, $u'(0) = u'(1) = 0$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8.17</span><span class="math-callout__name">()</span></p>

Given $f \in L^2$, there is a unique $u \in H^2(I)$ solving the Neumann problem; $u$ is given by $\min_{v \in H^1(I)}\lbrace \tfrac{1}{2}\int(v'^2 + v^2) - \int fv\rbrace$. If $f \in C(\bar I)$, then $u \in C^2(\bar I)$.

</div>

The crucial observation: Neumann uses **$H^1$**, not $H^1_0$, since $u(0)$ and $u(1)$ are *unknown* (only the *fluxes* $u'$ are prescribed). Lax–Milgram on $H^1$ gives $u \in H^1$, $u \in H^2$ by regularity, and integration by parts $(23)$:

$$
\int_I (-u'' + u - f)v + u'(1)v(1) - u'(0)v(0) = 0\quad \forall v \in H^1.
$$

Choosing $v \in H^1_0$ first gives $-u'' + u = f$ a.e.; the remaining boundary terms $u'(1)v(1) - u'(0)v(0) = 0$ for all $v$ yield the **natural boundary condition** $u'(0) = u'(1) = 0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Examples 4–7</span><span class="math-callout__name">(Other boundary conditions)</span></p>

* **Inhomogeneous Neumann** $u'(0) = \alpha, u'(1) = \beta$: minimize $\tfrac{1}{2}\int(v'^2 + v^2) - \int fv - \alpha v(0) + \beta v(1)$ on $H^1$.
* **Mixed boundary** $u(0) = 0, u'(1) = 0$: work in $H = \lbrace v \in H^1\,;\ v(0) = 0\rbrace$.
* **Robin / "third type"** $u'(0) = ku(0), u(1) = 0$: form $a(u, v) = \int u'v' + \int uv + ku(0)v(0)$ on $\lbrace v \in H^1\,;\ v(1) = 0\rbrace$, coercive for $k \ge 0$ (and even some $k < 0$ small).
* **Periodic** $u(0) = u(1), u'(0) = u'(1)$: work in $H = \lbrace v \in H^1\,;\ v(0) = v(1)\rbrace$ — the second condition $u'(0) = u'(1)$ is the *natural* counterpart and need not be imposed.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8</span><span class="math-callout__name">(BVP on $\mathbb{R}$)</span></p>

$-u'' + u = f$ on $\mathbb{R}$, $u(x) \to 0$ as $\lvert x\rvert \to \infty$. Work in $H^1(\mathbb{R})$: any classical solution with the decay condition is also in $H^1(\mathbb{R})$ (multiply by a cutoff $\zeta_n u$ and integrate). Lax–Milgram in $H^1(\mathbb{R})$ with $a(u, v) = \int u'v' + \int uv = (u, v)_{H^1(\mathbb{R})}$ gives a unique weak solution $u \in H^2(\mathbb{R})$, classical if $f \in C \cap L^2$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(When the technique fails)</span></p>

The problem $-u'' = f$ on $\mathbb{R}$, $u(x) \to 0$ at $\infty$, **cannot** be attacked by this technique: the bilinear form $a(u, v) = \int u'v'$ is *not* coercive on $H^1(\mathbb{R})$. In fact this problem need not have a solution even for smooth compactly supported $f$ (consider $\int f \neq 0$ — integrate $-u'' = f$ to get a contradiction with $u' \to 0$).

The technique *does* apply to half-line problems with decay: $-u'' + u = f$ on $(0, +\infty)$ with $u(0) = 0$ and $u \to 0$ at $\infty$.

</div>

### 8.5 The Maximum Principle

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8.19</span><span class="math-callout__name">(Maximum principle for Dirichlet)</span></p>

Let $f \in L^2(I)$ with $I = (0, 1)$ and $u \in H^2(I)$ solve $-u'' + u = f$ on $I$, $u(0) = \alpha, u(1) = \beta$. Then for every $x \in \bar I$,

$$
\boxed{\;\min\lbrace \alpha, \beta, \mathrm{ess\,inf}_I f\rbrace \le u(x) \le \max\lbrace \alpha, \beta, \mathrm{ess\,sup}_I f\rbrace.\;}
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof (Stampacchia's truncation method)</summary>

The weak formulation $\int u'v' + \int uv = \int fv$ holds for all $v \in H^1_0$. Fix $G \in C^1(\mathbb{R})$ strictly increasing on $(0, +\infty)$ with $G(t) = 0$ for $t \le 0$. Let $K = \max\lbrace \alpha, \beta, \mathrm{ess\,sup}_I f\rbrace < \infty$. The function $v = G(u - K) \in H^1_0$ (since $u - K \le 0$ at $\partial I$), so $v \ge 0$ and:

$$
\int u'^2 G'(u - K) + \int (u - K)G(u - K) = \int (f - K)G(u - K).
$$

But $f - K \le 0$ a.e. and $G(u - K) \ge 0$, so the RHS $\le 0$, while $u'^2 G'(u - K) \ge 0$. Hence $\int (u - K)G(u - K) \le 0$. Since $tG(t) \ge 0$ for all $t$, this forces $(u - K)G(u - K) = 0$ a.e., hence $u \le K$ a.e., hence $u \le K$ on $\bar I$ by continuity. Apply the same to $-u$ for the lower bound. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Classical proof for smooth case)</span></p>

When $f \in C(\bar I)$ and $u \in C^2$, the maximum principle is also classical: at an interior maximum $x_0$, $u'(x_0) = 0, u''(x_0) \le 0$, so $u(x_0) = f(x_0) + u''(x_0) \le f(x_0) \le K$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 8.20</span><span class="math-callout__name">(Comparison and $L^\infty$ bounds)</span></p>

Let $u$ solve (34). Then:

* **(i)** If $u = 0$ on $\partial I$ and $f \ge 0$, then $u \ge 0$ on $I$;
* **(ii)** If $u = 0$ on $\partial I$ and $f \in L^\infty$, then $\|u\|\_{L^\infty(I)} \le \|f\|\_{L^\infty(I)}$;
* **(iii)** If $f = 0$ on $I$, then $\|u\|\_{L^\infty(I)} \le \|u\|\_{L^\infty(\partial I)}$.

</div>

A similar result for Neumann (Proposition 8.21): essential extrema of $f$ control $u$ pointwise.

### 8.6 Eigenfunctions and Spectral Decomposition

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8.22</span><span class="math-callout__name">(Spectral theorem for Sturm–Liouville)</span></p>

Let $p \in C^1(\bar I)$ with $p \ge \alpha > 0$, $q \in C(\bar I)$, $I = (0, 1)$. There exists a sequence $(\lambda_n)$ of real numbers and a Hilbert basis $(e_n)$ of $L^2(I)$ with $e_n \in C^2(\bar I)\ \forall n$ and

$$
\boxed{\;\begin{cases} -(p e'_n)' + q e_n = \lambda_n e_n \text{ on } I, \\ e_n(0) = e_n(1) = 0. \end{cases}\;}
$$

Furthermore, $\lambda_n \to +\infty$ as $n \to \infty$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

WLOG $q \ge 0$ (else shift by a constant). For every $f \in L^2$, Lax–Milgram gives a unique $u \in H^2 \cap H^1_0$ solving $-(pu')' + qu = f$, $u(0) = u(1) = 0$. Define $T : L^2 \to L^2$ by $f \mapsto u$. We claim:

* **$T$ is bounded** $L^2 \to H^1$. Multiplying by $u$ and integrating: $\alpha\|u'\|_{L^2}^2 \le \int p u'^2 + \int q u^2 = \int fu \le \|f\|_2\|u\|_2$, with $\|u\|_2 \le C\|u'\|_2$ (Poincaré), so $\|u\|_{H^1} \le C\|f\|_2$.

* **$T$ is compact** $L^2 \to L^2$. The injection $H^1(I) \hookrightarrow L^2(I)$ is compact (Theorem 8.8 with bounded $I$), so $T$ is the composition of bounded $L^2 \to H^1$ with compact $H^1 \to L^2$, hence compact.

* **$T$ is self-adjoint.** $\int (Tf) g = \int f (Tg)$ — set $u = Tf, v = Tg$, multiply $-(pu')' + qu = f$ by $v$ and $-(pv')' + qv = g$ by $u$, both give $\int pu'v' + \int quv$.

* **$N(T) = \lbrace 0\rbrace$**: $Tf = 0 \Rightarrow u = 0 \Rightarrow f = 0$; and $\int (Tf) f = \int p u'^2 + \int qu^2 \ge 0$.

Apply the spectral theorem (Theorem 6.11) to $T$: there is a Hilbert basis $(e_n)$ of $L^2$ with $T e_n = \mu_n e_n$, $\mu_n > 0$, $\mu_n \to 0$. Setting $\lambda_n = 1/\mu_n$ gives the equation, with $e_n \in H^2 \cap H^1_0$, hence $C^2(\bar I)$ since $\lambda_n e_n \in C(\bar I)$. $\square$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($p = 1, q = 0$: classical Fourier sine basis)</span></p>

For $p \equiv 1, q \equiv 0$:

$$
e_n(x) = \sqrt{2}\sin(n\pi x),\qquad \lambda_n = n^2 \pi^2,\quad n = 1, 2, \ldots
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Eigenvalues depend on boundary conditions)</span></p>

For the *same* differential operator the eigenvalues and eigenfunctions vary with the boundary conditions. As an exercise, determine the eigenvalues of $A u = -u''$ with the boundary conditions of Examples 3, 5, 6, 7.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Boundedness of $I$ is essential)</span></p>

The assumption that $I$ is bounded enters in the *compactness* of $T$. When $I = \mathbb{R}$, the conclusion of Theorem 8.22 is generally false; one encounters **continuous spectrum** instead (Reed–Simon). For $-u'' + u = f$ on $\mathbb{R}$ with the Example 8 boundary condition, $T : f \mapsto u$ is self-adjoint and bounded $L^2 \to L^2$ but *not* compact.

</div>

### Comments on Chapter 8

#### 1. Some further inequalities

* **(i) Poincaré–Wirtinger.** For $I$ bounded and $u \in W^{1, 1}(I)$, set $\bar u = (1/\lvert I\rvert)\int u$. Then

  $$
  \|u - \bar u\|_\infty \le \|u'\|_1 \quad \forall u \in W^{1, 1}(I).
  $$

* **(ii) Hardy.** For $u \in W^{1, p}_0((0, 1)),\ 1 < p < \infty$, $u(x)/(x(1-x)) \in L^p$ and

  $$
  \Big\|\frac{u}{x(1-x)}\Big\|_p \le C_p\|u'\|_p.
  $$

* **(iii) Gagliardo–Nirenberg interpolation.** Bounded $I$, $1 \le r \le \infty,\ 1 \le q \le p \le \infty$. There is $C$ such that

  $$
  \boxed{\;\|u\|_p \le C\|u\|_q^{1-a}\|u\|_{W^{1, r}}^a,\qquad a\Big(\frac{1}{q} - \frac{1}{r} + 1\Big) = \frac{1}{q} - \frac{1}{p}.\;}
  $$

  In particular, for $u \in W^{2, r}(I)$ with the harmonic mean $1/p = (1/q + 1/r)/2$,

  $$
  \|u'\|_p \le C\|u\|_{W^{2, r}}^{1/2}\|u\|_q^{1/2}.
  $$

#### 2. Hilbert–Schmidt operators

The solution operator $T : f \mapsto u$ for the Sturm–Liouville Dirichlet problem (assuming $p \ge \alpha > 0, q \ge 0$) is a **Hilbert–Schmidt operator** from $L^2(I)$ into itself (Exercise 8.37). This is a strengthening of compactness; it is consistent with Theorem 6.12 ($L^2$-kernels) once one observes that $T$ is the integral operator with the *Green's function* of the Sturm–Liouville problem as kernel.

#### 3. Spectral properties of Sturm–Liouville operators

Many fine properties of the operator $A u = -(pu')' + qu$ with Dirichlet on a bounded $I$ are known:

* **(i) Simple eigenvalues.** Each $\lambda_n$ has *multiplicity one*.
* **(ii) Sign and oscillation.** Arranging $(\lambda_n)$ in increasing order, $e_n$ has exactly $(n - 1)$ zeros on $I$. In particular, the *first eigenfunction* $e_1$ has constant sign; one usually takes $e_1 > 0$.
* **(iii) Asymptotic Weyl law.** $\lambda_n / n^2$ converges to a positive limit as $n \to \infty$.

References: Weinberger, Protter–Weinberger, Coddington–Levinson, Hartman, Agmon, Courant–Hilbert, Ince, Pinchover–Rubinstein, Zettl, Buttazzo–Giaquinta–Hildebrandt.

The celebrated **Gelfand–Levitan theory** addresses the inverse problem: *what information about $q(x)$ can be retrieved purely from the spectrum of $-u'' + q(x)u$?* See Levitan and Comment 13 of Chapter 9.

## Chapter 9: Sobolev Spaces and the Variational Formulation of Elliptic Boundary Value Problems in $N$ Dimensions

Chapter 8 ran the variational program in dimension 1, where the Sobolev embedding $W^{1, p}(I) \hookrightarrow L^\infty(I)$ holds for *every* $p \ge 1$ and provides immediate continuity of weak solutions. In dimension $N \ge 2$ the situation is fundamentally richer: $W^{1, p}(\Omega) \hookrightarrow L^\infty$ holds only for $p > N$; for $p \le N$ functions in $W^{1, p}$ may be unbounded, and the right embedding target is some $L^{p^\star}$ with $p^\star = Np/(N - p)$ — the **Sobolev exponent**.

This chapter develops the multidimensional infrastructure:

* **§9.1** — definitions and elementary properties of $W^{1, p}(\Omega)$, *Friedrichs density* (Theorem 9.2), distance / translation characterizations (Proposition 9.3), product / chain rule / change of variables.
* **§9.2** — *extension operators* $W^{1, p}(\Omega) \to W^{1, p}(\mathbb{R}^N)$, requiring $\Omega$ to be of class $C^1$ (rectifying via local charts + partition of unity), and the resulting density of $C^\infty_c(\mathbb{R}^N)$ restrictions in $W^{1, p}(\Omega)$.
* **§9.3** — the **Sobolev inequalities**: Sobolev–Gagliardo–Nirenberg ($1 \le p < N$), the borderline case $p = N$, and **Morrey** ($p > N$); the **Rellich–Kondrachov** compactness theorem.
* **§9.4** — the space $W^{1, p}_0(\Omega)$, with the same role (boundary-value setup) as in dimension 1.

These tools give us the same four-step program of Chapter 8 — weak formulation, Lax–Milgram / Stampacchia for existence, regularity, recovery of classical solutions — only now applied to $-\Delta u + u = f$ and friends in $\Omega \subset \mathbb{R}^N$.

### 9.1 Definition and Elementary Properties of the Sobolev Spaces $W^{1, p}(\Omega)$

Let $\Omega \subset \mathbb{R}^N$ be open and $1 \le p \le \infty$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($W^{1, p}(\Omega)$)</span></p>

$$
\boxed{\;W^{1, p}(\Omega) = \Big\lbrace u \in L^p(\Omega)\,;\ \exists g_1, \ldots, g_N \in L^p(\Omega)\text{ with }\int_\Omega u \frac{\partial \varphi}{\partial x_i} = -\int_\Omega g_i \varphi\ \forall \varphi \in C^\infty_c(\Omega),\ \forall i\Big\rbrace.\;}
$$

We set $H^1(\Omega) = W^{1, 2}(\Omega)$. For $u \in W^{1, p}$ we write $\partial u/\partial x_i = g_i$ and

$$
\nabla u = (\partial u/\partial x_1, \ldots, \partial u/\partial x_N).
$$

The norm is

$$
\|u\|_{W^{1, p}} = \|u\|_p + \sum_{i=1}^N \|\partial u/\partial x_i\|_p,
$$

equivalent (for $1 \le p < \infty$) to $(\|u\|_p^p + \sum \|\partial u/\partial x_i\|_p^p)^{1/p}$. $H^1$ is equipped with the scalar product

$$
(u, v)_{H^1} = (u, v)_{L^2} + \sum_{i=1}^N (\partial u/\partial x_i, \partial v/\partial x_i)_{L^2}.
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 9.1</span><span class="math-callout__name">(Banach / reflexive / separable)</span></p>

$W^{1, p}(\Omega)$ is a Banach space for every $1 \le p \le \infty$, *reflexive* for $1 < p < \infty$, *separable* for $1 \le p < \infty$. $H^1(\Omega)$ is a separable Hilbert space.

</div>

(Proof: adapt Proposition 8.1 using the embedding $T u = [u, \nabla u] : W^{1, p} \hookrightarrow L^p \times (L^p)^N$.)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Distributional viewpoint)</span></p>

For $u \in L^1_{\mathrm{loc}}$, distribution theory gives a meaning to $\partial u/\partial x_i \in \mathcal{D}'(\Omega)$. Then $W^{1, p}$ is the set of $u \in L^p$ for which all $\partial u/\partial x_i$ happen to lie in $L^p$. When $\Omega = \mathbb{R}^N$ and $p = 2$, $W^{1, 2}$ can also be defined via the Fourier transform.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Useful convergence criterion)</span></p>

If $(u_n) \subset W^{1, p}$ with $u_n \to u$ in $L^p$ and $\nabla u_n \to g$ in $(L^p)^N$, then $u \in W^{1, p}$ and $\|u_n - u\|_{W^{1, p}} \to 0$. When $1 < p \le \infty$, it suffices that $u_n \to u$ in $L^p$ and $\|\nabla u_n\|_p$ stay *bounded*.

</div>

#### Multiplication by a smooth cut-off

If $u \in W^{1, p}(\Omega)$ and $\alpha \in C^1_c(\mathbb{R}^N)$, set $\bar f(x) = f(x)$ on $\Omega$, $0$ outside. Then

$$
\overline{\alpha u} \in W^{1, p}(\mathbb{R}^N),\qquad \frac{\partial}{\partial x_i}(\overline{\alpha u}) = \overline{\alpha\frac{\partial u}{\partial x_i}} + \overline{\frac{\partial \alpha}{\partial x_i} u}.
$$

(In general $\bar u \notin W^{1, p}(\mathbb{R}^N)$ — that is one of the reasons we need extension operators in §9.2.)

#### Density / Friedrichs

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Strong inclusion)</span></p>

For open sets $\omega, \Omega \subset \mathbb{R}^N$, we say $\omega$ is **strongly included** in $\Omega$, written $\omega \subset\!\subset \Omega$, if $\bar\omega \subset \Omega$ and $\bar\omega$ is compact.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.2</span><span class="math-callout__name">(Friedrichs)</span></p>

Let $u \in W^{1, p}(\Omega)$ with $1 \le p < \infty$. Then there exists $(u_n) \subset C^\infty_c(\mathbb{R}^N)$ such that

$$
u_{n\rvert\Omega} \to u\text{ in } L^p(\Omega),\qquad \nabla u_{n\rvert\omega} \to \nabla u_{\rvert\omega}\text{ in } L^p(\omega)^N\ \forall \omega \subset\!\subset \Omega.
$$

If $\Omega = \mathbb{R}^N$, the convergence holds in all of $\mathbb{R}^N$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 9.1</span><span class="math-callout__name">(Convolution preserves $W^{1, p}$)</span></p>

For $\rho \in L^1(\mathbb{R}^N)$ and $v \in W^{1, p}(\mathbb{R}^N)$, $\rho \star v \in W^{1, p}(\mathbb{R}^N)$ and $\partial/\partial x_i(\rho \star v) = \rho \star \partial v/\partial x_i$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 9.2</summary>

Set $\bar u(x) = u(x)$ on $\Omega$, $0$ outside. Take mollifiers $\rho_n$ and define $v_n = \rho_n \star \bar u \in C^\infty(\mathbb{R}^N)$, with $v_n \to \bar u$ in $L^p(\mathbb{R}^N)$. We check $\nabla v_{n\rvert\omega} \to \nabla u_{\rvert\omega}$ for $\omega \subset\!\subset \Omega$. Pick $\alpha \in C^1_c(\Omega)$ with $\alpha = 1$ on a neighborhood of $\omega$.

For $n$ large, $\rho_n \star (\overline{\alpha u}) = \rho_n \star \bar u$ on $\omega$. By Lemma 9.1 and the multiplication formula,

$$
\frac{\partial}{\partial x_i}(\rho_n \star \overline{\alpha u}) = \rho_n \star \overline{\Big(\alpha \frac{\partial u}{\partial x_i} + \frac{\partial \alpha}{\partial x_i} u\Big)} \to \alpha \frac{\partial u}{\partial x_i} + \frac{\partial \alpha}{\partial x_i} u\text{ in } L^p(\mathbb{R}^N).
$$

Restricted to $\omega$: $\partial v_n/\partial x_i \to \partial u/\partial x_i$ in $L^p(\omega)$. Multiply by cutoffs $\zeta_n$ to get compactly supported $u_n = \zeta_n v_n$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Meyers–Serrin)</span></p>

It can be shown (**Meyers–Serrin theorem**) that for any open $\Omega$, $C^\infty(\Omega) \cap W^{1, p}(\Omega)$ is dense in $W^{1, p}(\Omega)$. This holds for arbitrary open sets — but the approximating functions need *not* extend to $C^1$ functions on $\mathbb{R}^N$. The stronger density of $C^\infty_c(\mathbb{R}^N)$-restrictions requires regularity of $\Omega$ (Corollary 9.8).

</div>

#### Translation characterization

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 9.3</span><span class="math-callout__name">(Characterizations of $W^{1, p}$, $1 < p \le \infty$)</span></p>

Let $u \in L^p(\Omega)$ with $1 < p \le \infty$. The following are equivalent:

(i) $u \in W^{1, p}(\Omega)$;

(ii) $\exists C: \big\lvert\int_\Omega u \partial \varphi/\partial x_i\big\rvert \le C\|\varphi\|_{L^{p'}}\ \forall \varphi \in C^\infty_c(\Omega),\ \forall i$;

(iii) $\exists C: \forall \omega \subset\!\subset \Omega, \forall h \in \mathbb{R}^N$ with $\lvert h\rvert < \mathrm{dist}(\omega, \partial\Omega)$, $\|\tau_h u - u\|_{L^p(\omega)} \le C\lvert h\rvert$.

Furthermore $C = \|\nabla u\|_{L^p(\Omega)}$. If $\Omega = \mathbb{R}^N$, $\|\tau_h u - u\|_{L^p(\mathbb{R}^N)} \le \lvert h\rvert\|\nabla u\|_{L^p(\mathbb{R}^N)}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">($p = 1$ and bounded variation)</span></p>

For $p = 1$: (i) $\Rightarrow$ (ii) $\iff$ (iii) but the converse fails. Functions satisfying (ii) (or (iii)) for $p = 1$ are the **functions of bounded variation** — $L^1$ functions whose distributional derivatives are bounded measures. They appear in *minimal surfaces* (Giusti, DeGiorgi, Miranda), *elasticity / plasticity* (Temam–Strang, Suquet), and *quasilinear conservation laws* (Volpert, Bressan). See Ambrosio–Fusco–Pallara.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">($W^{1, \infty}$ and Lipschitz functions)</span></p>

Proposition 9.3 with $p = \infty$ implies every $u \in W^{1, \infty}(\Omega)$ has a continuous representative. If $\Omega$ is connected, $\lvert u(x) - u(y)\rvert \le \|\nabla u\|_{L^\infty(\Omega)} \mathrm{dist}_\Omega(x, y)$, where $\mathrm{dist}_\Omega$ is *geodesic distance*. In particular, on convex $\Omega$, $\mathrm{dist}_\Omega = \lvert x - y\rvert$. Conversely, if $u \in W^{1, p}(\Omega)$ with $\nabla u = 0$ a.e., then $u$ is constant on each connected component of $\Omega$.

</div>

#### Differentiation rules

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 9.4</span><span class="math-callout__name">(Product rule)</span></p>

If $u, v \in W^{1, p}(\Omega) \cap L^\infty(\Omega)$, $1 \le p \le \infty$, then $uv \in W^{1, p} \cap L^\infty$ and

$$
\boxed{\;\frac{\partial}{\partial x_i}(uv) = \frac{\partial u}{\partial x_i} v + u \frac{\partial v}{\partial x_i}.\;}
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 9.5</span><span class="math-callout__name">(Chain rule)</span></p>

Let $G \in C^1(\mathbb{R})$ with $G(0) = 0$ and $\lvert G'(s)\rvert \le M\ \forall s$. For $u \in W^{1, p}(\Omega)$, $1 \le p \le \infty$, $G \circ u \in W^{1, p}$ and

$$
\frac{\partial}{\partial x_i}(G \circ u) = (G' \circ u)\frac{\partial u}{\partial x_i}.
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 9.6</span><span class="math-callout__name">(Change of variables)</span></p>

Let $\Omega, \Omega' \subset \mathbb{R}^N$ open and $H : \Omega' \to \Omega$ bijective with $H \in C^1(\bar{\Omega'}), H^{-1} \in C^1(\bar\Omega)$, both Jacobians bounded. For $u \in W^{1, p}(\Omega)$, $1 \le p \le \infty$, $u \circ H \in W^{1, p}(\Omega')$ and

$$
\boxed{\;\frac{\partial}{\partial y_j}u(H(y)) = \sum_i \frac{\partial u}{\partial x_i}(H(y))\frac{\partial H_i}{\partial y_j}(y).\;}
$$

</div>

#### Higher-order Sobolev spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($W^{m, p}(\Omega)$)</span></p>

For integer $m \ge 2$ and $1 \le p \le \infty$, define inductively

$$
W^{m, p}(\Omega) = \lbrace u \in W^{m-1, p}(\Omega)\,;\ \partial u/\partial x_i \in W^{m-1, p}(\Omega)\ \forall i\rbrace.
$$

Equivalently, $u \in W^{m, p}$ iff for every multi-index $\alpha = (\alpha_1, \ldots, \alpha_N),\ \lvert\alpha\rvert = \sum \alpha_i \le m$, there is $g_\alpha \in L^p$ with

$$
\int_\Omega u D^\alpha \varphi = (-1)^{\lvert\alpha\rvert}\int_\Omega g_\alpha \varphi\quad \forall \varphi \in C^\infty_c(\Omega),
$$

where $D^\alpha = \partial^{\lvert\alpha\rvert}/\partial x_1^{\alpha_1}\cdots \partial x_N^{\alpha_N}$. Set $D^\alpha u = g_\alpha$. Norm: $\|u\|_{W^{m, p}} = \sum_{\lvert\alpha\rvert \le m} \|D^\alpha u\|_p$. $H^m = W^{m, 2}$ is a Hilbert space.

</div>

If $\Omega$ is "smooth enough" (Section 9.2) with bounded $\Gamma = \partial\Omega$, the norm is equivalent to $\|u\|_p + \sum_{\lvert\alpha\rvert = m}\|D^\alpha u\|_p$ (Adams).

### 9.2 Extension Operators

It is convenient to deduce properties of $W^{1, p}(\Omega)$ by reducing to $\Omega = \mathbb{R}^N$ via extension. This is *not* always possible — but for sufficiently regular $\Omega$, an extension exists.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation & Definition</span><span class="math-callout__name">(Half-space, cube; class $C^1$)</span></p>

Write $x = (x', x_N)$ with $x' \in \mathbb{R}^{N-1}$, $\lvert x'\rvert = (\sum x_i^2)^{1/2}$. Set

$$
\mathbb{R}^N_+ = \lbrace x_N > 0\rbrace,\quad Q = \lbrace \lvert x'\rvert < 1, \lvert x_N\rvert < 1\rbrace,\quad Q_+ = Q \cap \mathbb{R}^N_+,\quad Q_0 = \lbrace x_N = 0\rbrace \cap Q.
$$

An open set $\Omega \subset \mathbb{R}^N$ is of **class $C^1$** if for every $x \in \partial\Omega = \Gamma$ there is a neighborhood $U$ and a *bijective local chart* $H : Q \to U$ with $H \in C^1(\bar Q)$, $H^{-1} \in C^1(\bar U)$, $H(Q_+) = U \cap \Omega$, $H(Q_0) = U \cap \Gamma$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.7</span><span class="math-callout__name">(Extension operator)</span></p>

Suppose $\Omega$ is of class $C^1$ with bounded $\Gamma$, or $\Omega = \mathbb{R}^N_+$. There is a bounded linear extension $P : W^{1, p}(\Omega) \to W^{1, p}(\mathbb{R}^N)$ for every $1 \le p \le \infty$, with

$$
Pu_{\rvert\Omega} = u,\quad \|Pu\|_{L^p(\mathbb{R}^N)} \le C\|u\|_{L^p(\Omega)},\quad \|Pu\|_{W^{1, p}(\mathbb{R}^N)} \le C\|u\|_{W^{1, p}(\Omega)},
$$

where $C$ depends only on $\Omega$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 9.2</span><span class="math-callout__name">(Reflection across $x_N = 0$)</span></p>

For $u \in W^{1, p}(Q_+)$, the *even reflection*

$$
u^\star(x', x_N) = \begin{cases} u(x', x_N) & x_N > 0, \\ u(x', -x_N) & x_N < 0 \end{cases}
$$

belongs to $W^{1, p}(Q)$ with $\|u^\star\|_{L^p(Q)} \le 2\|u\|_{L^p(Q_+)}$, $\|u^\star\|_{W^{1, p}(Q)} \le 2\|u\|_{W^{1, p}(Q_+)}$, and

$$
\frac{\partial u^\star}{\partial x_i} = \Big(\frac{\partial u}{\partial x_i}\Big)^\star\quad i \le N - 1,\qquad \frac{\partial u^\star}{\partial x_N} = \Big(\frac{\partial u}{\partial x_N}\Big)^\Box,
$$

where $f^\Box$ is the *odd* extension of $f$ (negation in $x_N < 0$).

</div>

(Proof uses test functions of the form $\eta_k(x_N)\psi(x', x_N) + \eta_k(x_N)\psi(x', -x_N)$ with $\eta_k$ approximating the indicator of $\lbrace x_N > 0\rbrace$, then takes $k \to \infty$.)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reflection works for the square too)</span></p>

Lemma 9.2 yields easy extensions for some $\Omega$ that are *not* $C^1$. E.g., for the square $\Omega = (0, 1)^2 \subset \mathbb{R}^2$, four successive reflections give $\tilde u \in W^{1, p}$ on $(-1, 3)^2$ extending $u$. Multiplying by a smooth cutoff gives the global $W^{1, p}(\mathbb{R}^2)$ extension.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 9.3</span><span class="math-callout__name">(Partition of unity)</span></p>

Let $\Gamma$ be compact in $\mathbb{R}^N$, covered by open sets $U_1, \ldots, U_k$. Then there exist $\theta_0, \theta_1, \ldots, \theta_k \in C^\infty(\mathbb{R}^N)$ with

* $0 \le \theta_i \le 1$ and $\sum \theta_i = 1$ on $\mathbb{R}^N$,
* $\mathrm{supp}\,\theta_i \subset U_i$ compact ($i \ge 1$), $\mathrm{supp}\,\theta_0 \subset \mathbb{R}^N \setminus \Gamma$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 9.7 (extension)</summary>

"Rectify" $\Gamma$ via local charts and a partition of unity. Cover $\Gamma$ by finitely many $U_i$ each diffeomorphic via $H_i : Q \to U_i$ to the half-cube. With partition of unity $\theta_0, \theta_1, \ldots, \theta_k$ from Lemma 9.3, write $u = \sum \theta_i u = \sum u_i$.

* **(a)** $u_0$ has compact support inside $\Omega$, so its zero extension to $\mathbb{R}^N$ lies in $W^{1, p}$.
* **(b)** For $i \ge 1$, "transfer" $u_i$ to $Q_+$ via $H_i$, reflect across $\lbrace x_N = 0\rbrace$ (Lemma 9.2), retransfer back via $H_i^{-1}$, and multiply by $\theta_i$. The result $\hat u_i \in W^{1, p}(\mathbb{R}^N)$ extends $u_i$.

Sum: $Pu = \tilde u_0 + \sum_{i=1}^k \hat u_i$ has the required properties. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 9.8</span><span class="math-callout__name">(Density of $C^\infty_c(\mathbb{R}^N)$-restrictions)</span></p>

If $\Omega$ is of class $C^1$ and $u \in W^{1, p}(\Omega)$ with $1 \le p < \infty$, then there is $(u_n) \subset C^\infty_c(\mathbb{R}^N)$ with $u_{n\rvert\Omega} \to u$ in $W^{1, p}(\Omega)$.

</div>

(For bounded $\Gamma$: take $\zeta_n(\rho_n \star Pu) \to Pu$ in $W^{1, p}(\mathbb{R}^N)$; otherwise localize first.)

### 9.3 Sobolev Inequalities

In dimension 1, $W^{1, p}(I) \hookrightarrow L^\infty(I)$ for *every* $p \ge 1$ (Theorem 8.8). In dimension $N \ge 2$ this fails for $p \le N$. Nevertheless, **Sobolev's embedding** asserts that for $1 \le p < N$, $W^{1, p}(\mathbb{R}^N) \hookrightarrow L^{p^\star}(\mathbb{R}^N)$ for a specific $p^\star = Np/(N-p)$.

#### A. The case $\Omega = \mathbb{R}^N$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.9</span><span class="math-callout__name">(Sobolev–Gagliardo–Nirenberg, $1 \le p < N$)</span></p>

Let $1 \le p < N$. Then $W^{1, p}(\mathbb{R}^N) \subset L^{p^\star}(\mathbb{R}^N)$, where

$$
\boxed{\;\frac{1}{p^\star} = \frac{1}{p} - \frac{1}{N},\;}
$$

and there is a constant $C = C(p, N)$ with

$$
\|u\|_{p^\star} \le C\|\nabla u\|_p\quad \forall u \in W^{1, p}(\mathbb{R}^N). \tag{17}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why this exponent?)</span></p>

The exponent $p^\star$ is *forced* by a scaling argument. If $\|u\|_q \le C\|\nabla u\|_p\ \forall u \in C^\infty_c(\mathbb{R}^N)$, plug in $u_\lambda(x) = u(\lambda x)$ to get $\|u\|_q \lambda^{-N/q} \le C\|\nabla u\|_p \lambda^{1 - N/p}$ for all $\lambda > 0$, forcing $1 + N/q - N/p = 0$, i.e., $q = p^\star$ (otherwise let $\lambda \to 0$ or $\lambda \to \infty$ to get a contradiction).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 9.4</span><span class="math-callout__name">(Loomis–Whitney via iterated Hölder)</span></p>

Let $N \ge 2$ and $f_i \in L^{N-1}(\mathbb{R}^{N-1})$, $i = 1, \ldots, N$. For $\tilde x_i = (x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_N)$, the function $f(x) = f_1(\tilde x_1) f_2(\tilde x_2) \cdots f_N(\tilde x_N)$ belongs to $L^1(\mathbb{R}^N)$ and

$$
\|f\|_{L^1(\mathbb{R}^N)} \le \prod_{i=1}^N \|f_i\|_{L^{N-1}(\mathbb{R}^{N-1})}.
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 9.9 (sketch)</summary>

**Step 1: $p = 1$, $u \in C^1_c(\mathbb{R}^N)$.** From $u(x_1, \ldots, x_N) = \int_{-\infty}^{x_i}(\partial u/\partial x_i)\,dt$ we get $\lvert u(x)\rvert \le \int_{-\infty}^{+\infty}\lvert\partial u/\partial x_i\rvert\,dt = f_i(\tilde x_i)$, so $\lvert u(x)\rvert^N \le \prod_i f_i(\tilde x_i)$. By Lemma 9.4,

$$
\int \lvert u\rvert^{N/(N-1)} \le \prod_i \|f_i\|_{L^1(\mathbb{R}^{N-1})}^{1/(N-1)} = \prod_i \|\partial u/\partial x_i\|_1^{1/(N-1)}.
$$

Hence $\|u\|_{L^{N/(N-1)}} \le \prod_i \|\partial u/\partial x_i\|_1^{1/N}$, giving (17) for $p = 1$ (with $p^\star = N/(N-1)$).

**Step 2: $1 < p < N$, $u \in C^1_c(\mathbb{R}^N)$.** Apply Step 1 to $\lvert u\rvert^{m-1}u$ for $m \ge 1$: $\|u\|_{mN/(N-1)}^m \le m\|u\|_{p'(m-1)}^{m-1}\prod\|\partial u/\partial x_i\|_p^{1/N}$. Choose $m = (N - 1)p^\star/N$ so that $mN/(N-1) = p^\star$ and $p'(m-1) = p^\star$. The result follows.

**Step 3: density.** Approximate $u \in W^{1, p}$ by $u_n \in C^1_c$ in $W^{1, p}$; pass to the limit using Fatou. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 9.10</span><span class="math-callout__name">(Embedding in $L^q$, $q \in [p, p^\star]$)</span></p>

For $1 \le p < N$, $W^{1, p}(\mathbb{R}^N) \subset L^q(\mathbb{R}^N)$ for every $q \in [p, p^\star]$ with continuous injection, by interpolation.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 9.11</span><span class="math-callout__name">(Limiting case $p = N$)</span></p>

$W^{1, N}(\mathbb{R}^N) \subset L^q(\mathbb{R}^N)$ for every $q \in [N, +\infty)$.

</div>

(The constant $C = C(q)$ blows up as $q \to \infty$ — and in fact $W^{1, N} \not\subset L^\infty$ in general, see Remark 16.)

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.12</span><span class="math-callout__name">(Morrey, $p > N$)</span></p>

Let $p > N$. Then

$$
W^{1, p}(\mathbb{R}^N) \subset L^\infty(\mathbb{R}^N)
$$

with continuous injection. Moreover, for every $u \in W^{1, p}(\mathbb{R}^N)$,

$$
\boxed{\;\lvert u(x) - u(y)\rvert \le C\lvert x - y\rvert^\alpha \|\nabla u\|_p\quad \text{a.e. } x, y,\;}
$$

with $\alpha = 1 - N/p$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Hölder-continuous representative)</span></p>

The Morrey inequality implies $u \in W^{1, p}(\mathbb{R}^N)$ ($p > N$) admits a *unique* continuous representative — in fact a $C^{0, \alpha}$ representative with Hölder exponent $\alpha = 1 - N/p$. We henceforth identify $u$ with this representative.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof (sketch)</summary>

For $u \in C^1_c(\mathbb{R}^N)$ and a cube $Q$ of side $r$ containing $0$,

$$
u(x) - u(0) = \int_0^1 \frac{d}{dt}u(tx)\,dt,\qquad \lvert u(x) - u(0)\rvert \le r\sum_i \int_0^1 \lvert \partial u/\partial x_i(tx)\rvert\,dt.
$$

Average over $Q$ and apply Hölder: $\lvert \bar u - u(0)\rvert \le \frac{r^{1 - N/p}}{1 - N/p}\|\nabla u\|_{L^p(Q)}$. Translate: $\lvert \bar u - u(x)\rvert \le \frac{r^{1 - N/p}}{1 - N/p}\|\nabla u\|_{L^p(Q)}\ \forall x \in Q$. Subtracting,

$$
\lvert u(x) - u(y)\rvert \le \frac{2r^{1 - N/p}}{1 - N/p}\|\nabla u\|_{L^p(Q)}\quad \forall x, y \in Q.
$$

Given $x \neq y$, take $r = 2\lvert x - y\rvert$ to get the Hölder estimate. The $L^\infty$ bound follows by combining Hölder with mean-value over a fixed unit cube. Density extends to all $W^{1, p}$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 9.13</span><span class="math-callout__name">(Higher-order Sobolev embeddings on $\mathbb{R}^N$)</span></p>

For integer $m \ge 1$ and $p \in [1, +\infty)$:

$$
\begin{cases} W^{m, p}(\mathbb{R}^N) \subset L^q(\mathbb{R}^N),\ \frac{1}{q} = \frac{1}{p} - \frac{m}{N} & \text{if } \tfrac{1}{p} - \tfrac{m}{N} > 0, \\ W^{m, p}(\mathbb{R}^N) \subset L^q(\mathbb{R}^N)\ \forall q \in [p, +\infty) & \text{if } \tfrac{1}{p} - \tfrac{m}{N} = 0, \\ W^{m, p}(\mathbb{R}^N) \subset L^\infty(\mathbb{R}^N) & \text{if } \tfrac{1}{p} - \tfrac{m}{N} < 0, \end{cases}
$$

continuous injections. Moreover, if $m - N/p > 0$ is *not* an integer, set $k = \lfloor m - N/p\rfloor$, $\theta = m - N/p - k \in (0, 1)$. Then for all $\lvert\alpha\rvert \le k$,

$$
\|D^\alpha u\|_{L^\infty} \le C\|u\|_{W^{m, p}},
$$

and for $\lvert\alpha\rvert = k$, $D^\alpha u$ has a Hölder-continuous representative with exponent $\theta$. In particular $W^{m, p}(\mathbb{R}^N) \subset C^k(\mathbb{R}^N)$.

</div>

#### B. The case of a domain $\Omega \subset \mathbb{R}^N$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 9.14</span><span class="math-callout__name">(Sobolev embeddings on $\Omega$ of class $C^1$)</span></p>

Suppose $\Omega$ is of class $C^1$ with bounded $\Gamma$, or $\Omega = \mathbb{R}^N_+$. Then for all $1 \le p \le \infty$:

$$
\boxed{\;\begin{cases} W^{1, p}(\Omega) \subset L^{p^\star}(\Omega),\ \frac{1}{p^\star} = \frac{1}{p} - \frac{1}{N} & p < N, \\ W^{1, p}(\Omega) \subset L^q(\Omega)\ \forall q \in [p, +\infty) & p = N, \\ W^{1, p}(\Omega) \subset L^\infty(\Omega) & p > N. \end{cases}\;}
$$

For $p > N$, $\lvert u(x) - u(y)\rvert \le C\|u\|_{W^{1, p}}\lvert x - y\rvert^\alpha$ a.e. on $\Omega$ with $\alpha = 1 - N/p$. In particular $W^{1, p}(\Omega) \subset C(\bar\Omega)$.

</div>

(Apply Theorem 9.7 to extend to $\mathbb{R}^N$ and use Theorem 9.9 / Corollary 9.11 / Theorem 9.12.)

#### Compact embedding: Rellich–Kondrachov

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.16</span><span class="math-callout__name">(Rellich–Kondrachov)</span></p>

Suppose $\Omega$ is **bounded** and of class $C^1$. Then the following injections are **compact**:

$$
\begin{cases} W^{1, p}(\Omega) \subset L^q(\Omega)\ \forall q \in [1, p^\star) & p < N, \\ W^{1, p}(\Omega) \subset L^q(\Omega)\ \forall q \in [p, +\infty) & p = N, \\ W^{1, p}(\Omega) \subset C(\bar\Omega) & p > N. \end{cases}
$$

In particular, $W^{1, p}(\Omega) \subset L^p(\Omega)$ with compact injection for *every* $p$ (and every $N$).

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof (sketch)</summary>

**Case $p > N$.** Combine Corollary 9.14 with Ascoli–Arzelà.

**Case $p = N$.** Reduces to the case $p < N$ by interpolation.

**Case $p < N$.** Let $\mathcal{H}$ be the unit ball in $W^{1, p}(\Omega)$, $P$ the extension operator (Theorem 9.7), $\mathcal{F} = P(\mathcal{H}\rvert_{\bar\Omega}})$. Apply Kolmogorov–M. Riesz–Fréchet (Theorem 4.26): the translation estimate

$$
\|\tau_h f - f\|_{L^p(\mathbb{R}^N)} \le \lvert h\rvert\|\nabla f\|_p\quad \forall f \in \mathcal{F}
$$

(Proposition 9.3) plus interpolation between $L^p$ and $L^{p^\star}$ gives $\|\tau_h f - f\|_{L^q} \le C\lvert h\rvert^\alpha$ uniformly in $\mathcal{F}$. Hence $\mathcal{H}\rvert_\Omega$ has compact closure in $L^q(\Omega)$ for $q \in [1, p^\star)$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sharpness)</span></p>

* **(i) Boundedness is essential.** The injection $W^{1, p}(\Omega) \subset L^p(\Omega)$ is *not* compact for unbounded $\Omega$ (in general).
* **(ii) The exponent $p^\star$ is critical.** $W^{1, p}(\Omega) \subset L^{p^\star}(\Omega)$ is *never* compact, even for bounded smooth $\Omega$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Borderline case $p = N$ — Trudinger)</span></p>

For bounded $\Omega$ of class $C^1$ and $u \in W^{1, N}(\Omega)$, in general $u \notin L^\infty$ — the function $u(x) = (\log(1/\lvert x\rvert))^\alpha$ with $0 < \alpha < 1 - 1/N$ on $\Omega = \lbrace \lvert x\rvert < 1/2\rbrace$ lies in $W^{1, N}(\Omega)$ but is unbounded near $0$. Nevertheless, **Trudinger's inequality** gives sharp exponential integrability:

$$
\int_\Omega e^{\lvert u\rvert^{N/(N-1)}} < \infty\quad \forall u \in W^{1, N}(\Omega).
$$

(Adams, Gilbarg–Trudinger.)

</div>

### 9.4 The Space $W^{1, p}_0(\Omega)$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($W^{1, p}_0(\Omega)$)</span></p>

For $1 \le p < \infty$, $W^{1, p}_0(\Omega)$ is the **closure of $C^1_c(\Omega)$** in $W^{1, p}(\Omega)$. Set $H^1_0 = W^{1, 2}_0$. $W^{1, p}_0$ inherits the $W^{1, p}$ norm; it is a separable Banach space, reflexive for $1 < p < \infty$. $H^1_0$ is a separable Hilbert space.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">($\Omega = \mathbb{R}^N$ vs. proper subsets)</span></p>

Since $C^\infty_c(\mathbb{R}^N)$ is dense in $W^{1, p}(\mathbb{R}^N)$ (Theorem 9.2), $W^{1, p}_0(\mathbb{R}^N) = W^{1, p}(\mathbb{R}^N)$.

In contrast, if $\Omega \subsetneq \mathbb{R}^N$, generally $W^{1, p}_0(\Omega) \neq W^{1, p}(\Omega)$. **However**, if $\mathbb{R}^N \setminus \Omega$ is "sufficiently thin" and $p < N$, then $W^{1, p}_0(\Omega) = W^{1, p}(\Omega)$. For instance, if $\Omega = \mathbb{R}^N \setminus \lbrace 0\rbrace$ and $N \ge 2$, $H^1_0(\Omega) = H^1(\Omega)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">($C^\infty_c$ density)</span></p>

A mollifier argument shows $C^\infty_c(\Omega)$ is dense in $W^{1, p}_0(\Omega)$. So $C^\infty_c(\Omega)$ could equally have been used in the definition.

</div>

The functions in $W^{1, p}_0$ are "roughly" those of $W^{1, p}$ that "vanish on $\Gamma = \partial\Omega$." Making this precise is delicate — $u \in W^{1, p}$ is defined only a.e., and $\Gamma$ has measure zero. The following lemma is a first step:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 9.5</span><span class="math-callout__name">(Compact-support functions are in $W^{1, p}_0$)</span></p>

Let $u \in W^{1, p}(\Omega)$ with $1 \le p < \infty$. If $\mathrm{supp}\,u$ is a compact subset of $\Omega$, then $u \in W^{1, p}_0(\Omega)$.

</div>

(Proof: pick $\alpha \in C^1_c(\omega)$ with $\alpha = 1$ on $\mathrm{supp}\,u$ and $\omega \subset\!\subset \Omega$, so $\alpha u = u$. Apply Theorem 9.2 to obtain $u_n \in C^\infty_c(\mathbb{R}^N)$ with $u_n \to u$ in $L^p$ and $\nabla u_n \to \nabla u$ in $L^p(\omega)$; the cutoff makes $u_n\rvert_\Omega \in C^\infty_c(\Omega)$ for $n$ large. The full characterization of $W^{1, p}_0$ via traces requires the *theory of traces* — see Comment 16.)

The cleanest characterization of $W^{1, p}_0$ functions, *for sufficiently regular $\Omega$*, is the one we expected:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.17</span><span class="math-callout__name">(Boundary-vanishing characterization of $W^{1, p}_0$)</span></p>

Suppose $\Omega$ is of class $C^1$ and let $u \in W^{1, p}(\Omega) \cap C(\bar\Omega)$ with $1 \le p < \infty$. Then the following are equivalent:

(i) $u = 0$ on $\Gamma$;

(ii) $u \in W^{1, p}_0(\Omega)$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 9.17</summary>

**(i) $\Rightarrow$ (ii).** First assume $\mathrm{supp}\,u$ is bounded. Fix $G \in C^1(\mathbb{R})$ with $\lvert G(t)\rvert \le \lvert t\rvert$, $G(t) = 0$ for $\lvert t\rvert \le 1$, $G(t) = t$ for $\lvert t\rvert \ge 2$. Set $u_n = (1/n) G(nu)$; by Proposition 9.5, $u_n \in W^{1, p}$ with $u_n \to u$ in $W^{1, p}$ (dominated convergence). Moreover

$$
\mathrm{supp}\,u_n \subset \lbrace x \in \Omega\,;\ \lvert u(x)\rvert \ge 1/n\rbrace,
$$

a *compact* subset of $\Omega$ (since $u \in C(\bar\Omega)$ with $u = 0$ on $\Gamma$). By Lemma 9.5, $u_n \in W^{1, p}_0$, hence $u \in W^{1, p}_0$. For unbounded $\mathrm{supp}\,u$, multiply by cutoffs $\zeta_n$ first.

**(ii) $\Rightarrow$ (i).** Using local charts the problem reduces to: $u \in W^{1, p}_0(Q_+) \cap C(\bar Q_+)$ implies $u = 0$ on $Q_0 = \lbrace x_N = 0\rbrace \cap Q$. Let $u_n \in C^1_c(Q_+)$ with $u_n \to u$ in $W^{1, p}(Q_+)$. For $(x', x_N) \in Q_+$,

$$
\lvert u_n(x', x_N)\rvert \le \int_0^{x_N}\Big\lvert \frac{\partial u_n}{\partial x_N}(x', t)\Big\rvert\,dt,
$$

so for $0 < \varepsilon < 1$,

$$
\frac{1}{\varepsilon}\int_{\lvert x'\rvert < 1}\int_0^\varepsilon \lvert u_n(x', x_N)\rvert\,dx'\,dx_N \le \int_{\lvert x'\rvert < 1}\int_0^\varepsilon \Big\lvert \frac{\partial u_n}{\partial x_N}(x', t)\Big\rvert\,dx'\,dt.
$$

Let $n \to \infty$, then $\varepsilon \to 0$: $\int_{\lvert x'\rvert < 1} \lvert u(x', 0)\rvert\,dx' = 0$, so $u = 0$ on $Q_0$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Smoothness needed for ⇐, not ⇒)</span></p>

The implication (i) $\Rightarrow$ (ii) does *not* use smoothness of $\Omega$. By contrast, (ii) $\Rightarrow$ (i) requires it: e.g., for $\Omega = \mathbb{R}^N \setminus \lbrace 0\rbrace$ with $N \ge 2$ and $p \le N$, $W^{1, p}_0(\Omega) = W^{1, p}(\Omega)$, so functions in $W^{1, p}_0$ need not vanish at $\lbrace 0\rbrace$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 9.18</span><span class="math-callout__name">(Two more characterizations of $W^{1, p}_0$)</span></p>

Suppose $\Omega$ is of class $C^1$ and let $u \in L^p(\Omega)$ with $1 < p < \infty$. The following are equivalent:

(i) $u \in W^{1, p}_0(\Omega)$;

(ii) $\exists C: \big\lvert\int_\Omega u \partial \varphi/\partial x_i\big\rvert \le C\|\varphi\|_{L^{p'}(\mathbb{R}^N)}\ \forall \varphi \in C^1_c(\mathbb{R}^N),\ \forall i$;

(iii) the zero-extension

$$
\bar u(x) = \begin{cases} u(x) & x \in \Omega, \\ 0 & x \in \mathbb{R}^N \setminus \Omega, \end{cases}
$$

belongs to $W^{1, p}(\mathbb{R}^N)$, and in this case $\partial \bar u/\partial x_i = \overline{\partial u/\partial x_i}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why this matters: extension by 0)</span></p>

The extension theorem (Theorem 9.7) and Sobolev embeddings on $\Omega$ require *smoothness* of $\Omega$, since the proof rectifies $\partial\Omega$ via local charts. **For $W^{1, p}_0(\Omega)$ no smoothness is needed**: the canonical extension by $0$ is always valid (Proposition 9.18 (i) ⇒ (iii)). Consequently:

* Sobolev embeddings (Corollary 9.14) hold on $W^{1, p}_0(\Omega)$ for *arbitrary* open $\Omega$;
* Rellich–Kondrachov (Theorem 9.16) holds on $W^{1, p}_0(\Omega)$ for *arbitrary* bounded open $\Omega$;
* The Sobolev inequality $\|u\|_{L^{p^\star}(\Omega)} \le C(p, N)\|\nabla u\|_{L^p(\Omega)}\ \forall u \in W^{1, p}_0(\Omega)$ holds for arbitrary $\Omega$ and $1 \le p < N$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 9.19</span><span class="math-callout__name">(Poincaré's inequality)</span></p>

Suppose $1 \le p < \infty$ and $\Omega$ is **bounded**. There exists $C$ (depending on $\Omega, p$) such that

$$
\boxed{\;\|u\|_{L^p(\Omega)} \le C\|\nabla u\|_{L^p(\Omega)}\quad \forall u \in W^{1, p}_0(\Omega).\;}
$$

In particular, $\|\nabla u\|\_{L^p}$ is a norm on $W^{1, p}_0$ equivalent to the $W^{1, p}$ norm. On $H^1_0(\Omega)$, $(u, v) = \sum_i \int_\Omega \partial u/\partial x_i \partial v/\partial x_i$ is a scalar product inducing the norm $\|\nabla u\|\_{L^2}$, equivalent to the $H^1$ norm.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Extensions of Poincaré)</span></p>

Poincaré's inequality remains true if $\Omega$ has *finite measure*, or if $\Omega$ has a bounded projection on some axis (a strip).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">($W^{m, p}_0$)</span></p>

For $m \ge 1$ and $1 \le p < \infty$, $W^{m, p}_0(\Omega)$ is the closure of $C^\infty_c(\Omega)$ in $W^{m, p}$. Roughly, $u \in W^{m, p}_0$ iff $u \in W^{m, p}$ and $D^\alpha u = 0$ on $\Gamma$ for all $\lvert\alpha\rvert \le m - 1$. **Distinction**: $W^{m, p}_0(\Omega) \subsetneq W^{m, p}(\Omega) \cap W^{1, p}_0(\Omega)$ for $m \ge 2$ (the former requires *all* derivatives up to order $m - 1$ to vanish on $\Gamma$; the latter only requires $u$ itself).

</div>

#### The dual space of $W^{1, p}_0$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">($W^{-1, p'}$ and $H^{-1}$)</span></p>

The dual of $W^{1, p}_0(\Omega)$ is denoted $W^{-1, p'}(\Omega)$, $1 \le p < \infty$. The dual of $H^1_0$ is $H^{-1}$. We identify $L^2$ with its dual but **not** $H^1_0$ with its dual — leading to the **Gelfand triple**

$$
\boxed{\;H^1_0(\Omega) \subset L^2(\Omega) \subset H^{-1}(\Omega)\;}
$$

with continuous dense injections. If $\Omega$ is bounded, $W^{1, p}_0(\Omega) \subset L^2(\Omega) \subset W^{-1, p'}(\Omega)$ for $2N/(N + 2) \le p < \infty$; if $\Omega$ is unbounded the same holds only for $2N/(N+2) \le p \le 2$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 9.20</span><span class="math-callout__name">(Representation of $W^{-1, p'}$)</span></p>

For every $F \in W^{-1, p'}(\Omega)$ there exist $f_0, f_1, \ldots, f_N \in L^{p'}(\Omega)$ with

$$
\langle F, v\rangle = \int_\Omega f_0 v + \sum_{i=1}^N \int_\Omega f_i \frac{\partial v}{\partial x_i}\quad \forall v \in W^{1, p}_0(\Omega),
$$

and $\|F\|_{W^{-1, p'}} = \max_{0 \le i \le N} \|f_i\|_{p'}$. If $\Omega$ is bounded, one can take $f_0 = 0$.

</div>

(Proof adapts Proposition 8.14: embed $W^{1, p}_0 \hookrightarrow L^p \times (L^p)^N$ via $u \mapsto [u, \nabla u]$, apply Hahn–Banach + Riesz.)

### 9.5 Variational Formulation of Some Boundary Value Problems

We now apply the multidimensional Sobolev machinery to elliptic PDEs of second order. The four-step program (Steps A–D from §8.1) carries over verbatim.

#### Example 1: Homogeneous Dirichlet for the Laplacian

Let $\Omega \subset \mathbb{R}^N$ be open bounded. Find $u : \bar\Omega \to \mathbb{R}$ with

$$
\begin{cases} -\Delta u + u = f & \text{in } \Omega, \\ u = 0 & \text{on } \Gamma = \partial\Omega, \end{cases}\qquad \Delta u = \sum_{i=1}^N \partial^2 u/\partial x_i^2. \tag{31}
$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Classical and weak solution)</span></p>

A **classical solution** is $u \in C^2(\bar\Omega)$ satisfying $(31)$. A **weak solution** is $u \in H^1_0(\Omega)$ satisfying

$$
\boxed{\;\int_\Omega \nabla u \cdot \nabla v + \int_\Omega uv = \int_\Omega fv\quad \forall v \in H^1_0(\Omega).\;} \tag{32}
$$

</div>

**Step A: classical ⇒ weak.** Multiply $(31)$ by $v \in C^1_c(\Omega)$, integrate by parts, and extend by density (Theorem 9.17 says $u \in H^1_0$ since $u = 0$ on $\Gamma$).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.21</span><span class="math-callout__name">(Dirichlet, Riemann, Poincaré, Hilbert)</span></p>

Given any $f \in L^2(\Omega)$, there exists a unique weak solution $u \in H^1_0(\Omega)$ of $(31)$. Furthermore, $u$ minimizes the energy:

$$
\boxed{\;u = \arg\min_{v \in H^1_0(\Omega)}\Big\lbrace \tfrac{1}{2}\int_\Omega(\lvert\nabla v\rvert^2 + v^2) - \int_\Omega fv\Big\rbrace.\;}
$$

This is **Dirichlet's principle**.

</div>

(Proof: apply Lax–Milgram in $H = H^1_0$ with $a(u, v) = \int(\nabla u \cdot \nabla v + uv)$ and $\varphi(v) = \int fv$.)

**Step C: regularity** is delicate — see §9.6.

**Step D: classical recovery.** If a weak solution $u \in C^2(\bar\Omega)$ and $\Omega$ is $C^1$, then $u = 0$ on $\Gamma$ (Theorem 9.17), and integration by parts gives $-\Delta u + u = f$ a.e. (Corollary 4.24).

#### Example 2: Inhomogeneous Dirichlet

$-\Delta u + u = f$ in $\Omega$, $u = g$ on $\Gamma$. Suppose there is $\tilde g \in H^1(\Omega) \cap C(\bar\Omega)$ with $\tilde g = g$ on $\Gamma$ — set

$$
K = \lbrace v \in H^1(\Omega)\,;\ v - \tilde g \in H^1_0(\Omega)\rbrace,
$$

a nonempty closed convex set in $H^1$ depending only on $g$. A **weak solution** is $u \in K$ satisfying $\int(\nabla u \cdot \nabla v + uv) = \int fv\ \forall v \in H^1_0$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 9.22</span><span class="math-callout__name">()</span></p>

Given $f \in L^2(\Omega)$, there exists a unique weak solution $u \in K$ of the inhomogeneous Dirichlet problem, given by

$$
u = \arg\min_{v \in K}\Big\lbrace \tfrac{1}{2}\int(\lvert\nabla v\rvert^2 + v^2) - \int fv\Big\rbrace.
$$

</div>

(Proof: $u$ is a weak solution iff $\int \nabla u \cdot \nabla(v - u) + \int u(v - u) \ge \int f(v - u)\ \forall v \in K$ — a *variational inequality*. Apply Stampacchia (Theorem 5.6).)

#### Example 3: General second-order elliptic equation

Given $a_{ij}(x) \in C^1(\bar\Omega)$ satisfying the **ellipticity condition**

$$
\boxed{\;\sum_{i, j = 1}^N a_{ij}(x)\xi_i\xi_j \ge \alpha\lvert\xi\rvert^2\quad \forall x \in \Omega,\ \forall \xi \in \mathbb{R}^N,\text{ for some }\alpha > 0,\;} \tag{36}
$$

and $a_0 \in C(\bar\Omega)$, find $u : \bar\Omega \to \mathbb{R}$ with

$$
\begin{cases} -\sum_{i, j} \partial/\partial x_j(a_{ij} \partial u/\partial x_i) + a_0 u = f & \text{in } \Omega, \\ u = 0 & \text{on } \Gamma. \end{cases} \tag{37}
$$

A weak solution is $u \in H^1_0$ satisfying

$$
\sum_{i, j}\int_\Omega a_{ij}\frac{\partial u}{\partial x_i}\frac{\partial v}{\partial x_j} + \int_\Omega a_0 u v = \int_\Omega fv\quad \forall v \in H^1_0.
$$

If $a_0 \ge 0$, the bilinear form $a(u, v) = \sum \int a_{ij}\partial u/\partial x_i \partial v/\partial x_j + \int a_0 uv$ is continuous and coercive (using ellipticity, $a_0 \ge 0$, and Poincaré). Lax–Milgram yields a unique weak solution. If $(a_{ij})$ is symmetric, $u$ minimizes $\tfrac{1}{2}\big(\sum \int a_{ij}\partial v/\partial x_i \partial v/\partial x_j + \int a_0 v^2\big) - \int fv$.

#### General elliptic with first-order term

$$
-\sum_{i,j} \partial/\partial x_j(a_{ij}\partial u/\partial x_i) + \sum_i a_i \partial u/\partial x_i + a_0 u = f, \quad u = 0 \text{ on } \Gamma. \tag{39}
$$

Bilinear form $a(u, v) = \sum \int a_{ij} \partial u/\partial x_i \partial v/\partial x_j + \sum \int a_i \partial u/\partial x_i v + \int a_0 uv$ — *not* symmetric in general, *not* always coercive. In coercive cases, Lax–Milgram applies directly. In the general (non-coercive) case:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.23</span><span class="math-callout__name">(Fredholm-type alternative for general elliptic)</span></p>

If $f = 0$, the set of weak solutions $u \in H^1_0$ of $(40)$ is a finite-dimensional vector space, say of dimension $d$. There is a subspace $F \subset L^2$ of dimension $d$ such that the equation $(40)$ has a solution iff

$$
\Big[\int_\Omega fv = 0\quad \forall v \in F\Big].
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Fix $\lambda > 0$ large enough that $a(u, v) + \lambda\int uv$ is coercive on $H^1_0$. For $f \in L^2$, this yields a unique $u \in H^1_0$ with $a(u, \varphi) + \lambda\int u\varphi = \int f\varphi\ \forall \varphi \in H^1_0$. Set $u = Tf$, so $T : L^2 \to L^2$ is *compact* (since $H^1_0 \subset L^2$ is compact for bounded $\Omega$ — Theorem 9.16 + Remark 20). Equation $(40)$ becomes $u = T(f + \lambda u)$, i.e., setting $v = f + \lambda u$,

$$
v - \lambda T v = f.
$$

Apply Fredholm's alternative (Theorem 6.6). $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Existence ⇔ uniqueness for elliptic problems)</span></p>

If the homogeneous equation $f = 0$ has $u = 0$ as its *unique* solution, then for *every* $f \in L^2$ there is a unique solution. (This is Fredholm's "either/or" specialized to elliptic operators.) When $a_0 \ge 0$ on $\Omega$, *no* assumption on $a_i$ is needed — uniqueness comes from a maximum-principle argument (Gilbarg–Trudinger).

</div>

#### Example 4: Homogeneous Neumann

$-\Delta u + u = f$ in $\Omega$, $\partial u/\partial n = 0$ on $\Gamma$, where $\partial u/\partial n = \nabla u \cdot \mathbf{n}$ is the outward normal derivative. A **weak solution** is $u \in H^1(\Omega)$ (note: $H^1$, not $H^1_0$, since $u\rvert_\Gamma$ is *not* prescribed) satisfying

$$
\int_\Omega \nabla u \cdot \nabla v + \int_\Omega uv = \int_\Omega fv\quad \forall v \in H^1(\Omega). \tag{45}
$$

**Step A** uses Green's formula

$$
\int_\Omega (\Delta u) v = \int_\Gamma \frac{\partial u}{\partial n} v\,d\sigma - \int_\Omega \nabla u \cdot \nabla v\quad \forall u \in C^2(\bar\Omega), \forall v \in C^1(\bar\Omega),
$$

and density (Corollary 9.8).

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 9.24</span><span class="math-callout__name">(Existence and uniqueness for Neumann)</span></p>

For every $f \in L^2(\Omega)$ there is a unique weak solution $u \in H^1(\Omega)$ of the Neumann problem, with the variational characterization

$$
u = \arg\min_{v \in H^1(\Omega)}\Big\lbrace \tfrac{1}{2}\int(\lvert\nabla v\rvert^2 + v^2) - \int fv\Big\rbrace.
$$

</div>

(Proof: Lax–Milgram in $H = H^1(\Omega)$.)

**Step D: classical recovery.** If $u \in C^2(\bar\Omega)$ is a weak solution, Green's formula gives

$$
\int_\Omega(-\Delta u + u)v + \int_\Gamma \frac{\partial u}{\partial n} v\,d\sigma = \int fv\quad \forall v \in C^1(\bar\Omega).
$$

Choose $v \in C^1_c(\Omega)$ first to get $-\Delta u + u = f$ in $\Omega$; then for general $v \in C^1(\bar\Omega)$, $\int_\Gamma \partial u/\partial n\,v\,d\sigma = 0$, hence $\partial u/\partial n = 0$ on $\Gamma$ — the **natural boundary condition**.

#### Example 5: Unbounded domains

For $\Omega$ unbounded, one imposes (in addition to the boundary conditions on $\Gamma$) a *boundary condition at infinity* — typically $u(x) \to 0$ as $\lvert x\rvert \to \infty$. At the level of weak formulation this translates to $u \in H^1$. Three illustrative cases:

* **(a) $\Omega = \mathbb{R}^N$.** $-\Delta u + u = f \in L^2(\mathbb{R}^N)$ has a unique weak solution $u \in H^1(\mathbb{R}^N)$ with $\int \nabla u \cdot \nabla v + \int uv = \int fv\ \forall v \in H^1(\mathbb{R}^N)$.

* **(b) $\Omega = \mathbb{R}^N_+$ with Dirichlet.** $-\Delta u + u = f$ in $\mathbb{R}^N_+$, $u(x', 0) = 0$ on $\partial\mathbb{R}^N_+$. Weak solution $u \in H^1_0(\Omega)$.

* **(c) $\Omega = \mathbb{R}^N_+$ with Neumann.** $-\Delta u + u = f$ in $\mathbb{R}^N_+$, $\partial u/\partial x_N(x', 0) = 0$. Weak solution $u \in H^1(\Omega)$.

### 9.6 Regularity of Weak Solutions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Class $C^m, C^\infty$)</span></p>

$\Omega$ is of **class $C^m$** ($m \ge 1$) if for every $x \in \Gamma$ there is a neighborhood $U$ and a bijective $H : Q \to U$ with $H \in C^m(\bar Q)$, $H^{-1} \in C^m(\bar U)$, $H(Q_+) = U \cap \Omega$, $H(Q_0) = U \cap \Gamma$. $\Omega$ is **$C^\infty$** if it is $C^m$ for all $m$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.25</span><span class="math-callout__name">(Regularity for the Dirichlet problem)</span></p>

Let $\Omega$ be of class $C^2$ with bounded $\Gamma$, or $\Omega = \mathbb{R}^N_+$. Let $f \in L^2(\Omega)$ and $u \in H^1_0(\Omega)$ satisfy $(48)$. Then

$$
\boxed{\;u \in H^2(\Omega)\quad \text{and}\quad \|u\|_{H^2} \le C\|f\|_{L^2},\;}
$$

with $C$ depending only on $\Omega$. Moreover, if $\Omega$ is of class $C^{m+2}$ and $f \in H^m(\Omega)$, then

$$
u \in H^{m+2}(\Omega),\qquad \|u\|_{H^{m+2}} \le C\|f\|_{H^m}.
$$

In particular, if $f \in H^m$ with $m > N/2$, $u \in C^2(\bar\Omega)$. Finally, if $\Omega$ is $C^\infty$ and $f \in C^\infty(\bar\Omega)$, then $u \in C^\infty(\bar\Omega)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.26</span><span class="math-callout__name">(Regularity for the Neumann problem)</span></p>

Same hypotheses and conclusions for $u \in H^1(\Omega)$ satisfying $(49)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(General elliptic operators)</span></p>

The same conclusions hold for general second-order elliptic operators (Example 3): if $f \in L^2$, $a_{ij} \in C^1(\bar\Omega)$, $a_i \in C(\bar\Omega)$, then $u \in H^2$. For higher regularity, $a_{ij} \in C^{m+1}$, $a_i \in C^m$, $f \in H^m \Rightarrow u \in H^{m+2}$.

</div>

#### Method of translations (Nirenberg)

We sketch the proof of Theorem 9.25; the proof of Theorem 9.26 is analogous. The strategy: split into

1. **Interior regularity** — $u$ is regular on every $\omega \subset\!\subset \Omega$. Same pattern as the case $\Omega = \mathbb{R}^N$.
2. **Boundary regularity** — $u$ is regular near $\Gamma$. Reduces (via local charts) to the half-space case $\Omega = \mathbb{R}^N_+$.

The essential ingredient is the **method of translations** (or *difference quotients*) due to Nirenberg.

#### A. The case $\Omega = \mathbb{R}^N$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Difference quotients $D_h u$)</span></p>

For $h \in \mathbb{R}^N \setminus \lbrace 0\rbrace$:

$$
D_h u(x) = \frac{u(x + h) - u(x)}{\lvert h\rvert} = \frac{1}{\lvert h\rvert}(\tau_h u - u).
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 9.25 — Case $\Omega = \mathbb{R}^N$, $f \in L^2 \Rightarrow u \in H^2$</summary>

In $(48)$ take $\varphi = D_{-h}(D_h u) \in H^1(\mathbb{R}^N)$:

$$
\int \lvert\nabla D_h u\rvert^2 + \int \lvert D_h u\rvert^2 = \int f\,D_{-h}(D_h u) \le \|f\|_2\|D_{-h}(D_h u)\|_2.
$$

Recall (Proposition 9.3) that $\|D_{-h}v\|_2 \le \|\nabla v\|_2\ \forall v \in H^1$. Apply this to $v = D_h u$:

$$
\|D_h u\|_{H^1}^2 \le \|f\|_2 \|\nabla(D_h u)\|_2 \le \|f\|_2 \|D_h u\|_{H^1},
$$

so $\|D_h u\|_{H^1} \le \|f\|_2$ uniformly in $h$. In particular $\|D_h \partial u/\partial x_i\|_2 \le \|f\|_2$. By Proposition 9.3, $\partial u/\partial x_i \in H^1$, hence $u \in H^2$.

For $f \in H^m \Rightarrow u \in H^{m+2}$: induct on $m$ by replacing $\varphi$ with $D\varphi$ in $(48)$ and verifying that the derivative $Du$ satisfies the same equation with right-hand side $Df$. $\square$

</details>
</div>

#### B. The case $\Omega = \mathbb{R}^N_+$

The key new tool: **translations parallel to the boundary**. Write $h \parallel \Gamma$ if $h \in \mathbb{R}^{N-1} \times \lbrace 0\rbrace$ (tangential direction). The crucial observation:

$$
\boxed{\;u \in H^1_0(\mathbb{R}^N_+) \Rightarrow \tau_h u \in H^1_0(\mathbb{R}^N_+)\quad \text{if } h \parallel \Gamma.\;}
$$

So $H^1_0$ is *invariant* under tangential translations (but not normal ones).

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 9.6</span><span class="math-callout__name">(Tangential difference-quotient bound)</span></p>

$\|D_h v\|_{L^2(\Omega)} \le \|\nabla v\|_{L^2(\Omega)}\ \forall v \in H^1(\Omega), \forall h \parallel \Gamma$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 9.25 — Case $\Omega = \mathbb{R}^N_+$ (sketch)</summary>

Tangential second derivatives $\partial^2 u/\partial x_j \partial x_k$ for $j \ne N$ or $k \ne N$ are obtained by applying difference quotients along tangential directions $h \parallel \Gamma$ — this works because of $H^1_0$-invariance under tangential translations:

$$
\Big\lvert \int u \frac{\partial^2 \varphi}{\partial x_j\partial x_k}\Big\rvert \le \|f\|_2 \|\varphi\|_2\quad \forall (j, k) \neq (N, N),\ \forall \varphi \in C^\infty_c(\Omega). \tag{56}
$$

For the **normal-normal** second derivative $\partial^2 u/\partial x_N^2$: return to the equation $-\Delta u + u = f$, which gives

$$
\frac{\partial^2 u}{\partial x_N^2} = -\sum_{i=1}^{N-1}\frac{\partial^2 u}{\partial x_i^2} + u - f.
$$

Combining with $(56)$ and the equation gives the missing estimate. Thus all $\partial^2 u/\partial x_j \partial x_k \in L^2$, i.e., $u \in H^2$. The induction $f \in H^m \Rightarrow u \in H^{m+2}$ proceeds via Lemma 9.7 below. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 9.7</span><span class="math-callout__name">(Tangential derivatives of $u$ stay in $H^1_0$)</span></p>

Let $u \in H^2(\Omega) \cap H^1_0(\Omega)$ satisfy $(48)$. For tangential derivatives $Du = \partial u/\partial x_j$, $1 \le j \le N - 1$, $Du \in H^1_0(\Omega)$ and

$$
\int \nabla(Du) \cdot \nabla\varphi + \int (Du)\varphi = \int (Df)\varphi\quad \forall \varphi \in H^1_0(\Omega). \tag{58}
$$

</div>

#### C. The general case

Localize via partition of unity $u = \sum_{i=0}^k \theta_i u$ as in the proof of Theorem 9.7 (extension). Two pieces:

* **C₁. Interior estimates.** $\theta_0 u \in C^\infty_c(\Omega)$, so it solves a perturbed equation in $\mathbb{R}^N$ with $L^2$ right-hand side; by Case A, $\theta_0 u \in H^2(\mathbb{R}^N)$ with $\|\theta_0 u\|_{H^2} \le C\|f\|_2$.

* **C₂. Boundary estimates.** Each $\theta_i u$ ($i \ge 1$) is supported in $U_i$; transfer to $Q_+$ via $H_i$ to obtain $w(y) = (\theta_i u)(H_i(y))$. By Lemma 9.8, $w \in H^1_0(Q_+)$ satisfies a *new* second-order elliptic equation in $Q_+$ (with new coefficients $a_{k\ell}$ involving the Jacobian of $H_i$, still satisfying ellipticity since the change of variable preserves ellipticity).

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 9.8</span><span class="math-callout__name">(Change of variables preserves ellipticity)</span></p>

With the notation above, $w \in H^1_0(Q_+)$ satisfies

$$
\sum_{k, \ell = 1}^N \int_{Q_+} a_{k\ell}\frac{\partial w}{\partial y_k}\frac{\partial \psi}{\partial y_\ell}\,dy = \int_{Q_+} \tilde g \psi\,dy\quad \forall \psi \in H^1_0(Q_+),
$$

with $\tilde g = (g \circ H)\lvert\det\mathrm{Jac}\,H\rvert \in L^2(Q_+)$ and $a_{k\ell} \in C^1(\bar Q_+)$ satisfying the ellipticity condition $(36)$.

</div>

Apply Case B (with the $a_{k\ell}$-weighted equation) to deduce $w \in H^2(Q_+)$, then return to $\Omega \cap U_i$ to get $\theta_i u \in H^2$. Sum: $u \in H^2(\Omega)$.

The induction on $m$ proceeds the same way using Lemma 9.7.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Local nature of regularity / hypoellipticity)</span></p>

The regularity results are *local*: if $u \in H^1(\Omega)$ satisfies $\int \nabla u \cdot \nabla \varphi = \int f\varphi\ \forall \varphi \in C^\infty_c(\Omega)$ (a *very weak* solution; no boundary condition prescribed), and $f \in H^m_{\mathrm{loc}}(\Omega)$, then $u \in H^{m+2}_{\mathrm{loc}}(\Omega)$. So if $f \in C^\infty(\omega)$ on some $\omega \subset\!\subset \Omega$, then $u \in C^\infty(\omega)$ — even if $f$ is very irregular outside $\omega$. This property is called **hypoellipticity**.

But in the absence of a prescribed boundary condition, *we cannot conclude* $u \in C(\bar\Omega)$ even if $\Omega$ and $f$ are very smooth.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(A surprising consequence)</span></p>

The regularity results are a little surprising: an assumption on $\Delta u = \sum \partial^2 u/\partial x_i^2$ (the *sum* of second derivatives) forces a conclusion of the same nature on *every individual* $\partial^2 u/\partial x_i \partial x_j$.

</div>

### 9.7 The Maximum Principle

The maximum principle is a very useful tool with many formulations.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.27</span><span class="math-callout__name">(Maximum principle for the Dirichlet problem)</span></p>

Let $\Omega \subset \mathbb{R}^N$ open, $f \in L^2(\Omega)$, $u \in H^1(\Omega) \cap C(\bar\Omega)$ satisfy

$$
\int_\Omega \nabla u \cdot \nabla \varphi + \int_\Omega u\varphi = \int_\Omega f\varphi\quad \forall \varphi \in H^1_0(\Omega). \tag{70}
$$

Then for every $x \in \bar\Omega$,

$$
\boxed{\;\min\Big\lbrace \mathrm{ess\,inf}_\Gamma u,\ \mathrm{ess\,inf}_\Omega f\Big\rbrace \le u(x) \le \max\Big\lbrace \mathrm{ess\,sup}_\Gamma u,\ \mathrm{ess\,sup}_\Omega f\Big\rbrace.\;}
$$

</div>

(The proof, using **Stampacchia's truncation method**, parallels the 1D proof of Theorem 8.19: take a $C^1$ function $G$ vanishing on $(-\infty, 0]$ and strictly increasing on $(0, +\infty)$; for $K = \max\lbrace \mathrm{ess\,sup}_\Gamma u, \mathrm{ess\,sup}_\Omega f\rbrace$ assumed finite, $v = G(u - K) \in H^1_0$ since $u - K \le 0$ on $\Gamma$. Plug into $(70)$ and use $G(t)t \ge 0$ together with $f - K \le 0$ to force $u \le K$ a.e.)

The argument splits into two cases depending on whether $\Omega$ has finite measure:

* **(a) $\lvert\Omega\rvert < \infty$.** Then $v = G(u - K) \in H^1$ via Proposition 9.5 (apply chain rule to $t \mapsto G(t - K) - G(-K)$ which is $C^1$ with bounded derivative; Proposition 9.5 yields $v \in H^1$, and $v = 0$ on $\Gamma$ since $u - K \le 0$ there, so $v \in H^1_0$ by Theorem 9.17).
* **(b) $\lvert\Omega\rvert = \infty$.** Pick $K' > K$ and use $v = G(u - K')$ instead; on the set $\lbrace u \ge K'\rbrace$ we have $K'\int_{\lbrace u \ge K'\rbrace}\lvert u\rvert \le \int_\Omega u^2 < \infty$, ensuring $G(u - K') \in L^1(\Omega)$. Plugging into $(70)$:

$$
\int_\Omega \lvert\nabla u\rvert^2 G'(u - K') + \int_\Omega u G(u - K') = \int_\Omega f G(u - K'), \tag{71}
$$

so $\int_\Omega (u - K') G(u - K') \le \int_\Omega (f - K') G(u - K') \le 0$. Since $tG(t) \ge 0$, $u \le K'$ a.e. Let $K' \to K$.

#### Corollaries and refinements

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 9.28</span><span class="math-callout__name">(Comparison and $L^\infty$ bounds)</span></p>

Let $f \in L^2(\Omega)$, $u \in H^1(\Omega) \cap C(\bar\Omega)$ satisfy $(70)$. Then:

* $[u \ge 0\text{ on }\Gamma\text{ and }f \ge 0\text{ in }\Omega] \Rightarrow [u \ge 0\text{ in }\Omega]$,
* $\|u\|_{L^\infty(\Omega)} \le \max\lbrace \|u\|_{L^\infty(\Gamma)}, \|f\|_{L^\infty(\Omega)}\rbrace$.

In particular:

$$
\boxed{\;[f = 0\text{ in }\Omega] \Rightarrow \|u\|_{L^\infty(\Omega)} \le \|u\|_{L^\infty(\Gamma)},\;}
$$

$$
\boxed{\;[u = 0\text{ on }\Gamma] \Rightarrow \|u\|_{L^\infty(\Omega)} \le \|f\|_{L^\infty(\Omega)}.\;}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Classical PDE proof for smooth $u$)</span></p>

If $\Omega$ is bounded and $u$ is a *classical* solution of $-\Delta u + u = f$, here is a more direct proof. Let $x_0 \in \bar\Omega$ achieve $\max u$. Either $x_0 \in \Gamma$, so $u(x_0) \le \sup_\Gamma u \le K$, or $x_0 \in \Omega$, where $\nabla u(x_0) = 0$ and $\partial^2 u/\partial x_i^2(x_0) \le 0$ for all $i$, so $\Delta u(x_0) \le 0$ and

$$
u(x_0) = f(x_0) + \Delta u(x_0) \le f(x_0) \le K.
$$

This argument has the advantage that **it generalizes directly to general elliptic operators**:

$$
-\sum_{i,j}\frac{\partial}{\partial x_j}\Big(a_{ij}\frac{\partial u}{\partial x_i}\Big) + \sum_i a_i \frac{\partial u}{\partial x_i} + u = f,
$$

since at a maximum $x_0$, $\sum a_{ij}(x_0) \partial^2 u/\partial x_i\partial x_j(x_0) \le 0$ (reduce to the diagonal case by a coordinate change). The same conclusion holds for *weak* solutions, but the proof is more delicate (Gilbarg–Trudinger).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 9.29</span><span class="math-callout__name">(Maximum principle for general elliptic operators)</span></p>

Suppose $a_{ij} \in L^\infty(\Omega)$ satisfies the ellipticity condition $(36)$, $a_i \in L^\infty(\Omega)$, $a_0 \in L^\infty(\Omega)$ with $a_0 \ge 0$ in $\Omega$. Let $f \in L^2(\Omega)$ and $u \in H^1(\Omega) \cap C(\bar\Omega)$ satisfy

$$
\sum_{i, j}\int_\Omega a_{ij}\frac{\partial u}{\partial x_i}\frac{\partial \varphi}{\partial x_j} + \sum_i\int_\Omega a_i \frac{\partial u}{\partial x_i}\varphi + \int_\Omega a_0 u\varphi = \int_\Omega f\varphi\quad \forall \varphi \in H^1_0(\Omega). \tag{78}
$$

Then

$$
[u \ge 0\text{ on }\Gamma\text{ and }f \ge 0\text{ in }\Omega] \Rightarrow [u \ge 0\text{ in }\Omega]. \tag{79}
$$

If, in addition, $a_0 \equiv 0$ and $\Omega$ is bounded, then

$$
[f \ge 0\text{ in }\Omega] \Rightarrow [u \ge \inf_\Gamma u\text{ in }\Omega], \tag{80}
$$

$$
[f = 0\text{ in }\Omega] \Rightarrow [\inf_\Gamma u \le u \le \sup_\Gamma u\text{ in }\Omega]. \tag{81}
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof (case $a_i \equiv 0$; the general case is more delicate)</summary>

To prove $(79)$ it suffices to show $[u \le 0\text{ on }\Gamma\text{ and }f \le 0\text{ in }\Omega] \Rightarrow [u \le 0\text{ in }\Omega]$. Choose $\varphi = G(u)$ in $(78)$ with $G$ as in Theorem 9.27:

$$
\sum_{i, j}\int a_{ij}\frac{\partial u}{\partial x_i}\frac{\partial u}{\partial x_j} G'(u) + \int a_0 u G(u) = \int f G(u) \le 0,
$$

so $\int \lvert\nabla u\rvert^2 G'(u) \le 0$ (using ellipticity and $a_0 u G(u) \ge 0$). Set $H(t) = \int_0^t [G'(s)]^{1/2}\,ds$; then $H(u) \in H^1_0$ and $\lvert\nabla H(u)\rvert^2 = \lvert\nabla u\rvert^2 G'(u) = 0$ a.e. Hence $H(u)$ is constant, in fact $H(u) = 0$, so $u \le 0$ in $\Omega$.

For $(80)$: set $K = \sup_\Gamma u$; $(u - K) \in H^1(\Omega)$ satisfies $(78)$ with $a_0 \equiv 0$ (so the equation is unchanged), and $u - K \le 0$ on $\Gamma$. Apply $(79')$ to get $u \le K$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 9.30</span><span class="math-callout__name">(Maximum principle for the Neumann problem)</span></p>

Let $f \in L^2(\Omega)$ and $u \in H^1(\Omega)$ satisfy $\int \nabla u \cdot \nabla \varphi + \int u\varphi = \int f\varphi\ \forall \varphi \in H^1(\Omega)$. Then

$$
\inf_\Omega f \le u(x) \le \sup_\Omega f\quad \text{a.e. } x \in \Omega.
$$

</div>

(Proof analogous to Theorem 9.27, choosing test functions in $H^1$ rather than $H^1_0$.)

### 9.8 Eigenfunctions and Spectral Decomposition

We now return to the multi-dimensional analogue of the Sturm–Liouville spectral theorem of §8.6. In this section $\Omega$ is bounded.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.31</span><span class="math-callout__name">(Spectral decomposition of the Dirichlet Laplacian)</span></p>

There exist a Hilbert basis $(e_n)_{n \ge 1}$ of $L^2(\Omega)$ and a sequence $(\lambda_n)_{n \ge 1}$ of real numbers with $\lambda_n > 0\ \forall n$ and $\lambda_n \to +\infty$ such that

$$
e_n \in H^1_0(\Omega) \cap C^\infty(\Omega), \qquad -\Delta e_n = \lambda_n e_n\text{ in }\Omega.
$$

The $\lambda_n$'s are called the **eigenvalues** of $-\Delta$ (with Dirichlet boundary condition); the $e_n$'s are the corresponding **eigenfunctions**.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

For $f \in L^2$, let $u = Tf$ be the unique weak solution in $H^1_0(\Omega)$ of $\int \nabla u \cdot \nabla \varphi = \int f\varphi\ \forall \varphi \in H^1_0$ (Lax–Milgram). View $T : L^2 \to L^2$:

* $T$ is **compact**: it is bounded $L^2 \to H^1_0$ (Lax–Milgram estimate $\|u\|_{H^1_0} \le C\|f\|_2$), and $H^1_0 \subset L^2$ is compact (Theorem 9.16 + Remark 20).
* $T$ is **self-adjoint**: $\int (Tf) g = \int f (Tg) = \int \nabla(Tf) \cdot \nabla(Tg)$ (symmetry of the Dirichlet form).
* $N(T) = \lbrace 0\rbrace$ and $(Tf, f)_{L^2} = \int \lvert\nabla u\rvert^2 \ge 0$.

By the spectral theorem for compact self-adjoint operators (Theorem 6.11), $L^2$ admits a Hilbert basis $(e_n)$ of eigenvectors of $T$, with eigenvalues $(\mu_n)$, $\mu_n > 0$, $\mu_n \to 0$. Set $\lambda_n = 1/\mu_n$. Then $e_n \in H^1_0$ and

$$
\int \nabla e_n \cdot \nabla \varphi = \frac{1}{\mu_n}\int e_n \varphi\quad \forall \varphi \in H^1_0,
$$

i.e., $-\Delta e_n = \lambda_n e_n$ weakly. By the regularity results of §9.6 (interior version, Remark 25), $e_n \in H^m_{\mathrm{loc}}(\Omega)$ for all $m$, hence $e_n \in C^\infty(\Omega)$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bases for $H^1_0$)</span></p>

The sequence $(e_n/\sqrt{\lambda_n})$ is a Hilbert basis of $H^1_0$ equipped with the scalar product $\int \nabla u \cdot \nabla v$, and $(e_n/\sqrt{\lambda_n + 1})$ is a Hilbert basis of $H^1_0$ equipped with $\int (\nabla u \cdot \nabla v + uv)$. Indeed, orthonormality follows from $-\Delta e_n = \lambda_n e_n$, and density from the fact that $(e_n)$ is dense in $L^2$ (use $f \in H^1_0$ orthogonal to all $e_n$ in $H^1_0$ and pass through the equation).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Smoothness of $e_n$ up to the boundary)</span></p>

For *general* bounded $\Omega$ one can show $e_n \in L^\infty(\Omega)$. If $\Omega$ is of class $C^\infty$, then $e_n \in C^\infty(\bar\Omega)$ — by Theorem 9.25 applied iteratively.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(General elliptic version)</span></p>

Let $a_{ij} \in L^\infty$ satisfy ellipticity and $a_0 \in L^\infty$. There exist a Hilbert basis $(e_n)$ of $L^2(\Omega)$ and $\lambda_n \to +\infty$ with $e_n \in H^1_0$ and

$$
\sum_{i, j}\int a_{ij}\frac{\partial e_n}{\partial x_i}\frac{\partial \varphi}{\partial x_j} + \int a_0 e_n \varphi = \lambda_n \int e_n \varphi\quad \forall \varphi \in H^1_0.
$$

</div>

### Comments on Chapter 9

#### 1. Domains less regular than $C^1$

We have often supposed $\Omega$ of class $C^1$, which excludes domains with *corners*. One can weaken this to: $\Omega$ piecewise of class $C^1$, $\Omega$ Lipschitz, $\Omega$ has the cone property, $\Omega$ has the segment property, etc. (Adams, Agmon).

#### 2. Extension for $W^{m, p}$

Theorem 9.7 (existence of an extension operator) adapts to $W^{m, p}(\Omega)$ for $\Omega$ of class $C^m$, with a generalization of the reflection technique.

#### 3. Useful Sobolev-norm inequalities

**A. Poincaré–Wirtinger.** Let $\Omega$ be a connected open set of class $C^1$, $1 \le p \le \infty$. Then there exists $C$ with

$$
\boxed{\;\|u - \bar u\|_p \le C\|\nabla u\|_p\quad \forall u \in W^{1, p}(\Omega),\quad \bar u = \frac{1}{\lvert\Omega\rvert}\int_\Omega u.\;}
$$

If additionally $p < N$, this combined with the Sobolev inequality gives $\|u - \bar u\|_{p^\star} \le C\|\nabla u\|_p$.

**B. Hardy.** Let $\Omega$ be a bounded domain of class $C^1$, $1 < p < \infty$, $d(x) = \mathrm{dist}(x, \Gamma)$. There exists $C$ such that

$$
\boxed{\;\Big\|\frac{u}{d}\Big\|_p \le C\|\nabla u\|_p\quad \forall u \in W^{1, p}_0(\Omega).\;}
$$

Conversely, $u \in W^{1, p}(\Omega)$ with $u/d \in L^p(\Omega) \Rightarrow u \in W^{1, p}_0(\Omega)$ (Lions–Magenes).

**C. Gagliardo–Nirenberg interpolation.** Let $\Omega$ be a regular bounded open set in $\mathbb{R}^N$.

* **Example 1.** For $u \in L^p(\Omega) \cap W^{2, r}(\Omega)$ with $1 \le p, r \le \infty$ and $q$ the *harmonic mean* of $p, r$ (i.e., $1/q = \tfrac{1}{2}(1/p + 1/r)$), $u \in W^{1, q}(\Omega)$ and

  $$
  \|Du\|_{L^q} \le C\|u\|_{W^{2, r}}^{1/2}\|u\|_{L^p}^{1/2}.
  $$

  In particular, for $p = \infty$:

  $$
  \|Du\|_{L^q} \le C\|u\|_{W^{2, r}}^{1/2}\|u\|_{L^\infty}^{1/2},
  $$

  showing that $W^{2, r} \cap L^\infty$ is an *algebra*.

  For $p = q = r$: $\|Du\|_{L^p} \le C\|u\|_{W^{2, p}}^{1/2}\|u\|_{L^p}^{1/2}$, hence $\|Du\|_{L^p} \le \varepsilon\|D^2 u\|_{L^p} + C_\varepsilon\|u\|_{L^p}$ for every $\varepsilon > 0$.

* **Example 2.** $1 \le q \le p < \infty$, $u \in W^{1, N}(\Omega)$:

  $$
  \|u\|_{L^p} \le C\|u\|_{L^q}^{1 - a}\|u\|_{W^{1, N}}^a,\quad a = 1 - q/p.
  $$

  Specialized to $N = 2, p = 4, q = 2, a = 1/2$:

  $$
  \|u\|_{L^4} \le C\|u\|_{L^2}^{1/2}\|u\|_{H^1}^{1/2}\quad \forall u \in H^1(\Omega).
  $$

* **Example 3.** $1 \le q \le p \le \infty$, $r > N$:

  $$
  \|u\|_{L^p} \le C\|u\|_{L^q}^{1 - a}\|u\|_{W^{1, r}}^a,\quad a = \tfrac{1/q - 1/p}{1/q + 1/N - 1/r}.
  $$

#### 4. A useful property

For $u \in W^{1, p}(\Omega)$ and any constant $k$, $\nabla u = 0$ a.e. on $\lbrace x \in \Omega\,;\ u(x) = k\rbrace$.

#### 5. A.e. differentiability for $p > N$

If $u \in W^{1, p}(\Omega)$ with $p > N$, then $u$ is differentiable in the *usual sense* a.e.: there is $A \subset \Omega$ of measure zero such that

$$
\lim_{h \to 0}\frac{u(x + h) - u(x) - h \cdot \nabla u(x)}{\lvert h\rvert} = 0\quad \forall x \in \Omega \setminus A.
$$

This fails for $p \le N$ when $N > 1$ (Stein, Ch. 8).

#### 6. Fractional Sobolev spaces

For $0 < s < 1$ and $1 \le p < \infty$, the **fractional Sobolev space** $W^{s, p}(\Omega)$ is

$$
\boxed{\;W^{s, p}(\Omega) = \Big\lbrace u \in L^p(\Omega)\,;\ \frac{\lvert u(x) - u(y)\rvert}{\lvert x - y\rvert^{s + N/p}} \in L^p(\Omega \times \Omega)\Big\rbrace,\;}
$$

with the natural norm. Set $H^s = W^{s, 2}$. For non-integer $s > 1$, write $s = m + \sigma$ with $m$ integer; then

$$
W^{s, p}(\Omega) = \lbrace u \in W^{m, p}(\Omega)\,;\ D^\alpha u \in W^{\sigma, p}(\Omega)\ \forall \lvert\alpha\rvert = m\rbrace.
$$

These spaces can also be defined as *interpolation spaces* between $W^{1, p}$ and $L^p$, or via the Fourier transform when $p = 2$. By local charts, $W^{s, p}(\Gamma)$ is defined for smooth manifolds $\Gamma$ — they play a key role in trace theory.

#### 7. Theory of traces

The theory of traces gives a precise meaning to "the values of $u$ on $\Gamma$" for $u \in W^{1, p}(\Omega)$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 9.9</span><span class="math-callout__name">(Trace estimate on the half-space)</span></p>

For $\Omega = \mathbb{R}^N_+$ there exists $C$ such that

$$
\Big(\int_{\mathbb{R}^{N-1}}\lvert u(x', 0)\rvert^p\,dx'\Big)^{1/p} \le C\|u\|_{W^{1, p}(\Omega)}\quad \forall u \in C^1_c(\mathbb{R}^N).
$$

</div>

(Proof: $G(u(x', 0)) = -\int_0^\infty G'(u(x', x_N)) \partial u/\partial x_N\,dx_N$ for $G(t) = \lvert t\rvert^{p-1}t$.)

By density, $u \mapsto u\rvert_\Gamma$ extends to a bounded linear operator $W^{1, p}(\Omega) \to L^p(\Gamma)$ — the **trace** of $u$, written $u\rvert_\Gamma$. This makes sense for $\Omega$ of class $C^1$ with bounded $\Gamma$. Three key properties:

* **(i) Better regularity.** $u\rvert_\Gamma \in W^{1 - 1/p, p}(\Gamma)$ with $\|u\rvert_\Gamma\|_{W^{1 - 1/p, p}(\Gamma)} \le C\|u\|_{W^{1, p}(\Omega)}$. The trace operator is *surjective* onto $W^{1 - 1/p, p}(\Gamma)$.

* **(ii) Kernel.** $W^{1, p}_0(\Omega) = \lbrace u \in W^{1, p}(\Omega)\,;\ u\rvert_\Gamma = 0\rbrace$.

* **(iii) Green's formula.** For $u, v \in H^1(\Omega)$ with $\Omega$ of class $C^1$,

  $$
  \int_\Omega \frac{\partial u}{\partial x_i} v = -\int_\Omega u \frac{\partial v}{\partial x_i} + \int_\Gamma uv\,(\mathbf{n} \cdot \mathbf{e}_i)\,d\sigma.
  $$

  Similarly, for $u \in W^{2, p}(\Omega)$ the **normal derivative** $\partial u/\partial n = (\nabla u)\rvert_\Gamma \cdot \mathbf{n}$ is well defined (in $L^p(\Gamma)$, in fact in $W^{1 - 1/p, p}(\Gamma)$), and Green's formula reads

  $$
  -\int_\Omega (\Delta u)v = \int_\Omega \nabla u \cdot \nabla v - \int_\Gamma \frac{\partial u}{\partial n} v\,d\sigma\quad \forall v \in H^2(\Omega).
  $$

* **(iv) Traces of $W^{2, p}$.** $u \mapsto \lbrace u\rvert_\Gamma, \partial u/\partial n\rbrace$ is bounded, linear, and *surjective* from $W^{2, p}(\Omega)$ onto $W^{2 - 1/p, p}(\Gamma) \times W^{1 - 1/p, p}(\Gamma)$ (Lions–Magenes).

**Warning:** functions in $L^p(\mathbb{R}^N_+)$ do *not* have a trace on $\Gamma$ — trace theory genuinely uses the existence of one weak derivative.

#### 8. Operators of order $2m$ and elliptic systems

Existence and regularity extend to elliptic operators of order $2m$ and to elliptic systems via Gårding's inequality (Agmon, Lions–Magenes, Agmon–Douglis–Nirenberg). The **biharmonic** $\Delta^2$ (theory of plates), the **elasticity system**, and the **Stokes** system (fluid mechanics) are leading examples.

#### 9. Regularity in $L^p$ and $C^{0, \alpha}$

The regularity theorems of Chapter 9 (proved for $p = 2$) extend to general $p$:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.32</span><span class="math-callout__name">(Agmon–Douglis–Nirenberg: $L^p$ regularity)</span></p>

Suppose $\Omega$ is of class $C^2$ with bounded $\Gamma$, $1 < p < \infty$. For every $f \in L^p(\Omega)$ there is a unique solution $u \in W^{2, p}(\Omega) \cap W^{1, p}_0(\Omega)$ of $-\Delta u + u = f$ in $\Omega$, with $\|u\|_{W^{2, p}} \le C\|f\|_{L^p}$. Moreover, if $\Omega \in C^{m+2}$ and $f \in W^{m, p}$, then $u \in W^{m+2, p}$.

</div>

The proof rests either on (a) an explicit fundamental-solution representation $u = G \star f$ on $\mathbb{R}^N$ (with $G(x) = c_N \lvert x\rvert^{2-N}e^{-\lvert x\rvert}$ in $\mathbb{R}^3$), but the second derivatives of $G$ have a critical singularity that fails to lie in $L^1$; or (b) the **theory of singular integrals** of Calderón–Zygmund (Stein, Bers–John–Schechter). Theorem 9.32 *fails* for $p = 1, \infty$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.33</span><span class="math-callout__name">(Schauder: Hölder regularity)</span></p>

Suppose $\Omega$ is bounded of class $C^{2, \alpha}$, $0 < \alpha < 1$. For every $f \in C^{0, \alpha}(\bar\Omega)$ there is a unique solution $u \in C^{2, \alpha}(\bar\Omega)$ of $-\Delta u + u = f$ in $\Omega$, $u = 0$ on $\Gamma$. If additionally $\Omega \in C^{m+2, \alpha}$ and $f \in C^{m, \alpha}(\bar\Omega)$, then $u \in C^{m+2, \alpha}(\bar\Omega)$, with $\|u\|_{C^{m+2, \alpha}} \le C\|f\|_{C^{m, \alpha}}$.

</div>

These are the *optimal regularity* results — they fail in the spaces $L^1, L^\infty, C(\bar\Omega)$, which is why one avoids these spaces in PDE.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.34</span><span class="math-callout__name">(De Giorgi–Nash–Stampacchia)</span></p>

Let $\Omega \subset \mathbb{R}^N$ ($N \ge 2$) be bounded regular, $a_{ij} \in L^\infty$ satisfying ellipticity $(36)$. Let $f \in L^p(\Omega)$ with $p > N/2$ and $u \in H^1_0(\Omega)$ satisfy

$$
\sum_{i, j}\int a_{ij}\frac{\partial u}{\partial x_i}\frac{\partial \varphi}{\partial x_j} = \int f\varphi\quad \forall \varphi \in H^1_0(\Omega).
$$

Then $u \in C^{0, \alpha}(\bar\Omega)$ for some $\alpha = \alpha(\Omega, a_{ij}, p) \in (0, 1)$.

</div>

This is the celebrated regularity result for elliptic equations with **discontinuous coefficients** — the resolution of Hilbert's 19th problem. References: Stampacchia, Gilbarg–Trudinger, Ladyzhenskaya–Uraltseva, Giusti.

#### 10. Drawbacks of the variational method

The variational method gives weak solutions easily — but is *not always applicable*. Two completion strategies:

**(a) Duality method.** For $f \in L^1(\Omega)$ — or even $f$ a Radon measure — the linear functional $\varphi \mapsto \int f\varphi$ is *not* defined on every $\varphi \in H^1_0$ when $N > 1$, so the variational method is ineffective. Instead, define $T : L^2 \to L^2$ via the variational solution. By Theorem 9.32, $T : L^p \to W^{2, p}$, and Sobolev gives $T : L^p \to C_0(\bar\Omega)$ for $p > N/2$. By duality, $T^\star : \mathcal{M}(\Omega) = C_0(\bar\Omega)^\star \to L^{p'}(\Omega)$ (with $p > N/2$). Since $T = T^\star$ in $L^2$, set $u = T^\star f$ as a generalized solution. For $f \in L^1$ this gives $u \in L^q$ for $q < N/(N-2)$, the **(very weak) solution** in the sense

$$
-\int_\Omega u\Delta\varphi + \int_\Omega u\varphi = \int_\Omega f\varphi\quad \forall \varphi \in C^2(\bar\Omega),\ \varphi = 0\text{ on }\Gamma.
$$

**(b) Density method.** For $g \in C(\Gamma)$ but $g$ not necessarily the trace of an $H^1$ function, find $u \in C(\bar\Omega)$ with $-\Delta u + u = 0$ in $\Omega$, $u = g$ on $\Gamma$:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.35</span><span class="math-callout__name">(Continuous Dirichlet data)</span></p>

For every $g \in C(\Gamma)$ there exists a unique $u \in C(\bar\Omega) \cap C^\infty(\Omega)$ solving $-\Delta u + u = 0$ in $\Omega$, $u = g$ on $\Gamma$.

</div>

(Proof: extend $g$ to $\tilde g \in C(\mathbb{R}^N)$ by Tietze–Urysohn, mollify $g_n = \tilde g_n\rvert_\Gamma$ to get smooth boundary data, solve $u_n \in C^2(\bar\Omega)$ for each, and use the maximum principle (Corollary 9.28) $\|u_m - u_n\|_\infty \le \|g_m - g_n\|_\infty$ to deduce convergence to $u \in C(\bar\Omega)$. Smoothness inside $\Omega$ comes from interior regularity (Remark 25).)

The classical alternative is the **Perron method** of potential theory: $u(x) = \sup\lbrace v(x)\,;\ -\Delta v + v \le 0,\ v \le g\text{ on }\Gamma\rbrace$ ("subsolutions").

#### 11. The strong maximum principle

The maximum principle (Proposition 9.29) admits a *stronger* form for *classical* solutions:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.36</span><span class="math-callout__name">(Hopf: strong maximum principle)</span></p>

Let $\Omega$ be a connected, bounded, regular open set. Suppose $a_{ij} \in C^1(\bar\Omega)$ satisfies ellipticity, $a_i, a_0 \in C(\bar\Omega)$ with $a_0 \ge 0$ on $\Omega$. Let $u \in C(\bar\Omega) \cap C^2(\Omega)$ satisfy

$$
-\sum_{i, j}\frac{\partial}{\partial x_j}\Big(a_{ij}\frac{\partial u}{\partial x_i}\Big) + \sum_i a_i \frac{\partial u}{\partial x_i} + a_0 u = f\text{ in }\Omega. \tag{91}
$$

Suppose $f \ge 0$ in $\Omega$. If there is $x_0 \in \Omega$ with $u(x_0) = \min_{\bar\Omega} u$ and $u(x_0) \le 0$, then **$u$ is constant on $\Omega$** (and furthermore $f \equiv 0$).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 9.37</span><span class="math-callout__name">(Strict positivity / dichotomy)</span></p>

Under the same hypotheses, with $f \ge 0$ in $\Omega$ and $u \ge 0$ on $\Gamma$:

$$
\boxed{\;\text{either}\quad u > 0\text{ in }\Omega\quad \text{or}\quad u \equiv 0\text{ in }\Omega.\;}
$$

</div>

(Bers–John–Schechter, Gilbarg–Trudinger, Protter–Weinberger, Pucci–Serrin.)

#### 12. Laplace–Beltrami operators

Elliptic operators on Riemannian manifolds (with or without boundary) — in particular the **Laplace–Beltrami operator** — play a key role in differential geometry and physics (Choquet–Dewitt–Dillard).

#### 13. Spectral properties. Inverse problems

For $\Omega$ a connected, bounded, regular open set, $a_{ij} \in C^1(\bar\Omega)$ satisfying ellipticity, $a_0 \in C(\bar\Omega)$, and $A u = -\sum \partial/\partial x_j(a_{ij}\partial u/\partial x_i) + a_0 u$ with Dirichlet conditions, let $\lambda_n$ denote the eigenvalues in increasing order. Then:

* **(i) Simple first eigenvalue.** $\lambda_1$ has multiplicity 1, and $e_1$ can be chosen $> 0$ in $\Omega$ — by **Krein–Rutman** (cf. comments on Chapter 6 + Problem 41). In dimension $N \ge 2$ the higher eigenvalues *can* have multiplicity > 1.
* **(ii) Weyl asymptotics.** $\lambda_n \sim c n^{2/N}$ as $n \to \infty$ for some $c > 0$ (Agmon).

The **spectral geometry** problem asks how much of $\Omega$ can be recovered purely from the spectrum $(\lambda_n)$:

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Famous Question</span><span class="math-callout__name">(Mark Kac, 1966: "Can one hear the shape of a drum?")</span></p>

Let $\Omega_1, \Omega_2 \subset \mathbb{R}^2$ be bounded with the same Dirichlet eigenvalues of $-\Delta$. Are $\Omega_1$ and $\Omega_2$ isometric?

The answer is *positive* if $\Omega_1$ is a disk. In **1991, Gordon–Webb–Wolpert** gave a **negative answer for domains with corners**. The problem is still open for *smooth* domains.

</div>

Other **inverse problems** seek to determine coefficients/parameters or shape from boundary measurements (Dirichlet-to-Neumann map) or scattering data — applications to medical imaging, seismology, etc. (Uhlmann, Croke).

#### 14. Degenerate elliptic problems

Problems with $\sum a_{ij}\xi_i\xi_j \ge 0$ instead of $\ge \alpha\lvert\xi\rvert^2$ — the matrix may degenerate at points of $\Omega$ (Kohn–Nirenberg, Baouendi–Goulaouic, Oleinik–Radkevitch).

#### 15. Nonlinear elliptic problems

A vast field motivated by geometry, mechanics, physics, optimal control, probability theory, etc. — with major development since Leray–Schauder (1930s).

* **(a) Semilinear problems.** $-\Delta u = f(x, u)$ in $\Omega$, $u = 0$ on $\Gamma$. Includes **bifurcation problems** $-\Delta u = f_\lambda(x, u)$ with parameter $\lambda$.
* **(b) Quasilinear problems.** $-\sum \partial/\partial x_j(a_{ij}(x, u, \nabla u)\partial u/\partial x_i) = f(x, u, \nabla u)$, possibly degenerate. Example: **minimal surfaces** with $a_{ij}(x, u, p) = \delta_{ij}(1 + \lvert p\rvert^2)^{-1/2}$. **Fully nonlinear:** $F(x, u, Du, D^2 u) = 0$ (e.g., **Monge–Ampère**).
* **(c) Free boundary problems.** Solve a linear elliptic equation in an unknown set $\Omega$ — compensated by *two* boundary conditions on $\Gamma$ (e.g., Dirichlet + Neumann).

**Techniques:**

* Monotonicity methods (Browder, Lions);
* Topological methods (Schauder fixed-point, Leray–Schauder degree theory) — Schwartz, Krasnoselskii, Nirenberg;
* Variational methods (critical point theory, min–max techniques, Morse theory) — Rabinowitz, Berger, Krasnoselskii, Nirenberg, Mawhin–Willem, Struwe.

#### 16. Geometric measure theory

At the interface of geometry and PDE, with major development since the 1960s (Federer, De Giorgi, Volpert, Almgren) — applications to calculus of variations, isoperimetric inequalities, phase transitions, fractures in mechanics, edge detection in image processing, line vortices, superconductors, superfluids. The space $BV$ plays a distinguished role. References: Ambrosio–Fusco–Pallara, Simon, Evans–Gariepy, Lin–Yang.

## Chapter 10: Evolution Problems — The Heat Equation and the Wave Equation

The previous chapters built the *static* picture: solving $-\Delta u + u = f$ in a domain, with various boundary conditions. We now add *time*. The two emblematic linear evolution PDEs of mathematical physics are:

* the **heat equation** $\partial u/\partial t - \Delta u = 0$ — a *parabolic* equation, modeling diffusion of heat / chemical species / probability mass;
* the **wave equation** $\partial^2 u/\partial t^2 - \Delta u = 0$ — a *hyperbolic* equation, modeling vibrations and propagation of signals.

The strategy is to view $u(x, t)$ as a *function of time alone* with values in a Hilbert space $H$ of functions in the spatial variable $x$ — say $H = L^2(\Omega)$ or $H = H^1_0(\Omega)$. Writing $u(t)$ for the function $x \mapsto u(x, t)$ converts both PDEs into abstract Cauchy problems

$$
\frac{du}{dt} + Au = 0 \quad \text{(heat)},\qquad \frac{d^2 u}{dt^2} + Au = 0 \quad \text{(wave)},
$$

with $A = -\Delta$ a *self-adjoint maximal monotone* operator on $H$ (with domain reflecting the boundary conditions). The Hille–Yosida theory of Chapter 7 then handles existence, uniqueness, and regularity automatically.

The **structural contrasts** between the two equations are striking:

| | **Heat** | **Wave** |
| --- | --- | --- |
| Order in $t$ | first | second |
| Dynamics | semigroup of contractions | group of isometries |
| Smoothing | $u_0 \in L^2 \Rightarrow u \in C^\infty$ for $t > 0$ | no smoothing — singularities propagate |
| Time direction | irreversible | reversible |
| Speed of propagation | infinite (Brownian) | finite (cone of dependence) |
| Conservation law | dissipation $\tfrac12\frac{d}{dt}\lvert u\rvert^2 = -\lvert\nabla u\rvert^2$ | $\lvert\partial u/\partial t\rvert^2 + \lvert\nabla u\rvert^2$ conserved |
| Maximum principle | yes | only in special cases |

Throughout this chapter $\Omega \subset \mathbb{R}^N$ is open, $\Gamma = \partial\Omega$, and we set

$$
Q = \Omega \times (0, +\infty),\qquad \Sigma = \Gamma \times (0, +\infty)
$$

(the *lateral boundary* of the cylinder $Q$). To simplify, we assume throughout that **$\Omega$ is of class $C^\infty$ with $\Gamma$ bounded** (the assumption can be considerably weakened if one only seeks weak solutions).

### 10.1 The Heat Equation: Existence, Uniqueness, and Regularity

We seek $u(x, t) : \bar\Omega \times [0, +\infty) \to \mathbb{R}$ satisfying

$$
\boxed{\;\begin{cases} \dfrac{\partial u}{\partial t} - \Delta u = 0 & \text{in } Q, \\ u = 0 & \text{on } \Sigma, \\ u(x, 0) = u_0(x) & \text{on } \Omega, \end{cases}\;} \tag{1, 2, 3}
$$

where $\Delta = \sum_{i=1}^N \partial^2/\partial x_i^2$ is the Laplacian in *space* variables, $t$ is time, and $u_0$ is the **initial (Cauchy) data**. Equation $(1)$ is the **heat equation**; $(2)$ is the (homogeneous) **Dirichlet boundary condition** ($\Gamma$ kept at zero temperature). One could replace $(2)$ by the Neumann condition $\partial u/\partial n = 0$ on $\Sigma$ (zero heat flux through $\Gamma$) or any of the boundary conditions of Chapters 8–9.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10.1</span><span class="math-callout__name">(Existence and uniqueness, $u_0 \in L^2$)</span></p>

Assume $u_0 \in L^2(\Omega)$. There exists a unique function $u(x, t)$ satisfying $(1), (2), (3)$ and

$$
u \in C([0, \infty); L^2(\Omega)) \cap C((0, \infty); H^2(\Omega) \cap H^1_0(\Omega)), \tag{4}
$$

$$
u \in C^1((0, \infty); L^2(\Omega)). \tag{5}
$$

Moreover

$$
\boxed{\;u \in C^\infty(\bar\Omega \times [\varepsilon, \infty))\quad \forall \varepsilon > 0,\;}
$$

and $u \in L^2(0, \infty; H^1_0(\Omega))$ with the **energy identity**

$$
\boxed{\;\tfrac{1}{2}\lvert u(T)\rvert_{L^2}^2 + \int_0^T \lvert\nabla u(t)\rvert_{L^2}^2\,dt = \tfrac{1}{2}\lvert u_0\rvert_{L^2}^2\quad \forall T > 0.\;} \tag{6}
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Apply Hille–Yosida (more precisely Theorem 7.7 — the *self-adjoint case with smoothing*) in $H = L^2(\Omega)$ to the unbounded operator

$$
A : D(A) \subset H \to H,\qquad D(A) = H^2(\Omega) \cap H^1_0(\Omega),\quad Au = -\Delta u.
$$

The boundary condition $(2)$ is *encoded* in the choice of $D(A)$ (functions in $H^1_0$ vanish on $\Gamma$ — see Theorem 9.17). We check $A$ is self-adjoint maximal monotone:

* **Monotone**: $(Au, u)_{L^2} = \int_\Omega(-\Delta u)u = \int_\Omega \lvert\nabla u\rvert^2 \ge 0$ (integration by parts in $H^1_0$).
* **Maximal monotone**: $R(I + A) = L^2$ — for every $f \in L^2$, the equation $u - \Delta u = f$ has a unique solution $u \in H^2 \cap H^1_0$ (Theorem 9.25).
* **Self-adjoint**: by Proposition 7.6 it suffices to check symmetry, $(Au, v)_{L^2} = \int \nabla u \cdot \nabla v = (u, Av)_{L^2}$.

Theorem 7.7 then yields a unique $u$ satisfying $(4), (5)$ with $\lvert u(t)\rvert \le \lvert u_0\rvert$ and $\lvert du/dt\rvert \le \tfrac{1}{t}\lvert u_0\rvert$.

By Theorem 9.25 (regularity), $D(A^\ell) = \lbrace u \in H^{2\ell}(\Omega)\,;\ u = \Delta u = \cdots = \Delta^{\ell - 1}u = 0\text{ on }\Gamma\rbrace$, with continuous injection. Theorem 7.7 gives $u \in C^k((0, \infty); D(A^\ell))$ for all $k, \ell$, and the Sobolev embedding $H^{2\ell} \hookrightarrow C^k(\bar\Omega)$ for large $\ell$ gives $u \in C^k((0, \infty); C^k(\bar\Omega))$ for every $k$, hence $u \in C^\infty(\bar\Omega \times [\varepsilon, \infty))$.

For the energy identity $(6)$: formally multiply $(1)$ by $u$ and integrate. Rigorously, $\varphi(t) = \tfrac{1}{2}\lvert u(t)\rvert^2$ is $C^1$ on $(0, \infty)$ with $\varphi'(t) = (du/dt, u) = (\Delta u, u) = -\int \lvert\nabla u\rvert^2$. Integrate over $[\varepsilon, T]$ and let $\varepsilon \to 0$. $\square$

</details>
</div>

#### Improved regularity at $t = 0$

If we add hypotheses on $u_0$, the solution becomes regular *up to* $t = 0$ (Theorem 10.1 only guarantees regularity for $t \ge \varepsilon > 0$).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10.2</span><span class="math-callout__name">(Regularity up to $t = 0$)</span></p>

Let $u$ be the solution of $(1), (2), (3)$ from Theorem 10.1.

* **(a)** If $u_0 \in H^1_0(\Omega)$, then

  $$u \in C([0, \infty); H^1_0(\Omega)) \cap L^2(0, \infty; H^2(\Omega)),\qquad \frac{du}{dt} \in L^2(0, \infty; L^2(\Omega)),$$

  and $\int_0^T \lvert du/dt(t)\rvert_{L^2}^2\,dt + \tfrac{1}{2}\lvert\nabla u(T)\rvert_{L^2}^2 = \tfrac{1}{2}\lvert\nabla u_0\rvert_{L^2}^2$.

* **(b)** If $u_0 \in H^2(\Omega) \cap H^1_0(\Omega)$, then $u \in C([0, \infty); H^2) \cap L^2(0, \infty; H^3)$ and $du/dt \in L^2(0, \infty; H^1_0)$.

* **(c)** If $u_0 \in H^k(\Omega)$ for every $k$ and satisfies the **compatibility conditions**

  $$
  \boxed{\;u_0 = \Delta u_0 = \cdots = \Delta^j u_0 = \cdots = 0\text{ on }\Gamma\;} \tag{8}
  $$

  for every $j$, then $u \in C^\infty(\bar\Omega \times [0, \infty))$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof (sketch)</summary>

* **(a)** Work in $H_1 = H^1_0(\Omega)$ with the scalar product $(u, v)_{H_1} = \int(\nabla u \cdot \nabla v + uv)$. Let $A_1 : D(A_1) \subset H_1 \to H_1$ be defined by $D(A_1) = \lbrace u \in H^3 \cap H^1_0\,;\ \Delta u \in H^1_0\rbrace$ and $A_1 u = -\Delta u$. One checks $A_1$ is self-adjoint maximal monotone in $H_1$ (using Theorem 9.25 for maximality + Theorem 9.17 for the kernel structure). Apply Theorem 7.7 in $H_1$ — uniqueness gives this is the same $u$ as in Theorem 10.1.
* **(b)** Same scheme with $H_2 = H^2 \cap H^1_0$ and $A_2$ defined accordingly.
* **(c)** Apply Theorem 7.5 (higher regularity in Hille–Yosida) — assumption $(8)$ says exactly $u_0 \in D(A^k)$ for every $k$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Smoothing effect and time irreversibility)</span></p>

Theorem 10.1 says the heat equation has a **strong smoothing effect**: $u(x, t)$ is $C^\infty$ in $x$ for every $t > 0$ even if $u_0$ is discontinuous! As a corollary, the heat equation is **time-irreversible**: in general one cannot solve the *backward* problem

$$
\frac{\partial u}{\partial t} - \Delta u = 0 \text{ in } \Omega \times (0, T),\quad u = 0 \text{ on } \Sigma,\quad u(x, T) = u_T(x).
$$

For solvability one would need at minimum $u_T \in C^\infty(\bar\Omega)$ with $\Delta^j u_T = 0$ on $\Gamma$ for every $j$ — and even then, in general, *no* such backward solution exists. (This is **not** the same as $-\partial u/\partial t - \Delta u = 0$ with final data, which always has a solution by the change $t \to T - t$.)

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Neumann variant)</span></p>

The above results are also true — with slight modifications — if the Dirichlet condition $u = 0$ on $\Sigma$ is replaced by the Neumann condition $\partial u/\partial n = 0$.

</div>

#### Spectral / Fourier method

When $\Omega$ is bounded, problem $(1), (2), (3)$ can be solved via *decomposition in a Hilbert basis* of $L^2(\Omega)$. Take $(e_i)$ to be eigenfunctions of $-\Delta$ with Dirichlet condition (Theorem 9.31): $-\Delta e_i = \lambda_i e_i$, $e_i = 0$ on $\Gamma$, $\lambda_i > 0$. Seek $u(x, t) = \sum_i a_i(t) e_i(x)$. Substituting into $(1)$ gives $a'_i(t) + \lambda_i a_i(t) = 0$, hence $a_i(t) = a_i(0) e^{-\lambda_i t}$, with $a_i(0) = (u_0, e_i)_{L^2} = \int_\Omega u_0 e_i$. Thus

$$
\boxed{\;u(x, t) = \sum_{i=1}^\infty a_i(0) e^{-\lambda_i t} e_i(x).\;} \tag{14}
$$

This is the **method of separation of variables** (or **Fourier method**) — the original setting in which Fourier discovered Fourier series, studying the heat equation in one space dimension. (For convergence and regularity of this series, see Weinberger.)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why the compatibility conditions look mysterious)</span></p>

The compatibility conditions $(8)$ are *necessary* for $u \in C^\infty(\bar\Omega \times [0, \infty))$. Indeed, if such $u$ exists, then $u = 0$ on $\Sigma$ extends by continuity to give $\partial^j u/\partial t^j = 0$ on $\Gamma \times [0, \infty)$ for all $j$. On the other hand, $\partial^2 u/\partial t^2 = \Delta(\partial u/\partial t) = \Delta^2 u$ in $Q$, and inductively $\partial^j u/\partial t^j = \Delta^j u$ in $Q$, hence on $\bar\Omega \times [0, \infty)$ by continuity. Comparing on $\Gamma \times \lbrace 0\rbrace$ gives $(8)$.

</div>

### 10.2 The Maximum Principle

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10.3</span><span class="math-callout__name">(Maximum principle for the heat equation)</span></p>

Assume $u_0 \in L^2(\Omega)$ and let $u$ be the solution of $(1), (2), (3)$. Then for all $(x, t) \in Q$,

$$
\boxed{\;\min\Big\lbrace 0, \inf_\Omega u_0\Big\rbrace \le u(x, t) \le \max\Big\lbrace 0, \sup_\Omega u_0\Big\rbrace.\;}
$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof (Stampacchia's truncation method)</summary>

Set $K = \max\lbrace 0, \sup_\Omega u_0\rbrace$, assume $K < \infty$. Take $G$ as in Theorem 9.27 ($G(s) = 0$ for $s \le 0$, strictly increasing for $s > 0$, $\lvert G'\rvert \le M$), and set $H(s) = \int_0^s G(\sigma)\,d\sigma$. Define

$$
\varphi(t) = \int_\Omega H(u(x, t) - K)\,dx.
$$

Then $\varphi \in C([0, \infty); \mathbb{R}) \cap C^1((0, \infty); \mathbb{R})$ with $\varphi(0) = 0$ and $\varphi \ge 0$. Compute

$$
\varphi'(t) = \int_\Omega G(u - K)\frac{\partial u}{\partial t} = \int_\Omega G(u - K)\Delta u = -\int_\Omega G'(u - K)\lvert\nabla u\rvert^2 \le 0,
$$

using integration by parts (valid since $G(u - K) \in H^1_0(\Omega)$ for every $t > 0$ — because $u(t) \in H^1_0$ and $u \le 0 \le K$ on $\Gamma$, so $u - K \le 0$ on $\Gamma$, and $G$ vanishes there). Hence $\varphi \equiv 0$, so $u \le K$ a.e. The lower bound applies to $-u$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 10.4</span><span class="math-callout__name">(Comparison + $L^\infty$ bound)</span></p>

Let $u_0 \in L^2(\Omega)$.

* **(i)** $u_0 \ge 0$ a.e. on $\Omega \Rightarrow u \ge 0$ in $Q$.
* **(ii)** $u_0 \in L^\infty(\Omega) \Rightarrow u \in L^\infty(Q)$ with

$$
\|u\|_{L^\infty(Q)} \le \|u_0\|_{L^\infty(\Omega)}. \tag{19}
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 10.5</span><span class="math-callout__name">(Continuity up to $\bar Q$)</span></p>

Let $u_0 \in C(\bar\Omega) \cap L^2(\Omega)$ with $u_0 = 0$ on $\Gamma$. Then the solution $u$ of $(1), (2), (3)$ belongs to $C(\bar Q)$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Corollary 10.5</summary>

Approximate $u_0$ by $u_{0n} \in C^\infty_c(\Omega)$ with $u_{0n} \to u_0$ in $L^\infty(\Omega)$ and in $L^2(\Omega)$. By Theorem 10.2(c), $u_n \in C^\infty(\bar Q)$. The maximum-principle estimate $(19)$ gives $\|u_n - u_m\|_{L^\infty(Q)} \le \|u_{0n} - u_{0m}\|_{L^\infty(\Omega)}$, so $(u_n)$ is Cauchy in $C(\bar Q)$, with limit $u \in C(\bar Q)$. $\square$

</details>
</div>

#### Classical maximum principle

There is also a classical proof — we sketch it for a more general parabolic inequality.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10.6</span><span class="math-callout__name">(Parabolic maximum principle, classical form)</span></p>

Suppose $\Omega$ is bounded and $u \in C(\bar\Omega \times [0, T])$ is $C^1$ in $t$, $C^2$ in $x$, with

$$
\frac{\partial u}{\partial t} - \Delta u \le 0\quad \text{in } \Omega \times (0, T). \tag{22}
$$

Then

$$
\boxed{\;\max_{\bar\Omega \times [0, T]} u = \max_P u,\;}
$$

where $P = (\bar\Omega \times \lbrace 0\rbrace) \cup (\Gamma \times [0, T])$ is the **parabolic boundary** of the cylinder.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Set $v(x, t) = u(x, t) + \varepsilon\lvert x\rvert^2$ with $\varepsilon > 0$. Then $\partial v/\partial t - \Delta v = (\partial u/\partial t - \Delta u) - 2\varepsilon N \le -2\varepsilon N < 0$. Suppose, for contradiction, $\max_{\bar\Omega \times [0, T]} v$ is attained at $(x_0, t_0) \notin P$, so $x_0 \in \Omega, 0 < t_0 \le T$. Then $\Delta v(x_0, t_0) \le 0$ (interior maximum in $x$) and $\partial v/\partial t(x_0, t_0) \ge 0$ (max in $t$, with $\ge 0$ since either $t_0 < T$ giving $= 0$ or $t_0 = T$ giving $\ge 0$). Hence $(\partial v/\partial t - \Delta v)(x_0, t_0) \ge 0$ — contradiction.

So $\max v = \max_P v \le \max_P u + \varepsilon C$ where $C = \sup_\Omega \lvert x\rvert^2$. Then $\max u \le \max v \le \max_P u + \varepsilon C$ for every $\varepsilon > 0$, giving the result. $\square$

</details>
</div>

### 10.3 The Wave Equation

We seek $u(x, t) : \bar\Omega \times [0, +\infty) \to \mathbb{R}$ satisfying

$$
\boxed{\;\begin{cases} \dfrac{\partial^2 u}{\partial t^2} - \Delta u = 0 & \text{in } Q, \\ u = 0 & \text{on } \Sigma, \\ u(x, 0) = u_0(x) & \text{on } \Omega, \\ \dfrac{\partial u}{\partial t}(x, 0) = v_0(x) & \text{on } \Omega. \end{cases}\;} \tag{27, 28, 29, 30}
$$

This is the **wave equation**. The operator $\partial^2/\partial t^2 - \Delta$ is often denoted $\Box$ and called the **d'Alembertian**. The wave equation is the prototype *hyperbolic* equation; it models small vibrations (a string for $N = 1$, a membrane for $N = 2$) and propagation of acoustic / electromagnetic waves in homogeneous elastic media. $(28)$ says the boundary is *fixed*; $(29), (30)$ are the **Cauchy data** — initial position and initial velocity.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10.7</span><span class="math-callout__name">(Existence and uniqueness for the wave equation)</span></p>

Assume $u_0 \in H^2(\Omega) \cap H^1_0(\Omega)$ and $v_0 \in H^1_0(\Omega)$. There exists a unique solution $u$ of $(27), (28), (29), (30)$ satisfying

$$
u \in C([0, \infty); H^2(\Omega) \cap H^1_0(\Omega)) \cap C^1([0, \infty); H^1_0(\Omega)) \cap C^2([0, \infty); L^2(\Omega)). \tag{31}
$$

Moreover, the **energy is conserved**:

$$
\boxed{\;\Big\lvert\frac{\partial u}{\partial t}(t)\Big\rvert_{L^2}^2 + \lvert\nabla u(t)\rvert_{L^2}^2 = \lvert v_0\rvert_{L^2}^2 + \lvert\nabla u_0\rvert_{L^2}^2\quad \forall t \ge 0.\;} \tag{32}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Conservation law)</span></p>

Equation $(32)$ is a **conservation law**: the total energy (kinetic $\tfrac{1}{2}\lvert\partial u/\partial t\rvert^2$ + potential $\tfrac{1}{2}\lvert\nabla u\rvert^2$) is *invariant* in time. Contrast this with the heat equation, where energy *dissipates*: $\tfrac{d}{dt}\tfrac{1}{2}\lvert u\rvert^2 = -\lvert\nabla u\rvert^2 \le 0$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 10.7</summary>

Convert to a first-order system by setting $v = \partial u/\partial t$. Then $(27)$ becomes

$$
\frac{\partial u}{\partial t} - v = 0,\qquad \frac{\partial v}{\partial t} - \Delta u = 0\quad \text{in } Q. \tag{33}
$$

Set $U = (u, v)^T$, so $(33)$ becomes $dU/dt + AU = 0$ with

$$
AU = \begin{pmatrix} 0 & -I \\ -\Delta & 0 \end{pmatrix}\begin{pmatrix} u \\ v\end{pmatrix} = \begin{pmatrix} -v \\ -\Delta u\end{pmatrix}. \tag{35}
$$

Apply Hille–Yosida in $H = H^1_0(\Omega) \times L^2(\Omega)$ with the scalar product $(U_1, U_2)_H = \int \nabla u_1 \cdot \nabla u_2 + \int u_1 u_2 + \int v_1 v_2$. Define $A : D(A) \subset H \to H$ by $(35)$ and $D(A) = (H^2 \cap H^1_0) \times H^1_0$.

We check $A + I$ is maximal monotone:

* **$A + I$ monotone**: $(AU, U)_H + \lvert U\rvert_H^2 = -\int \nabla v \cdot \nabla u - \int uv + \int(-\Delta u)v + \int u^2 + \int \lvert\nabla u\rvert^2 + \int v^2$. The first term equals $\int(-\Delta u)v$ on $H^1_0$ (integration by parts), so the cross-terms cancel: $= -\int uv + \int u^2 + \int v^2 + \int \lvert\nabla u\rvert^2 \ge 0$.

* **$A + I$ maximal monotone**: must show $A + 2I$ is surjective. Given $F = (f, g) \in H$, solve $-v + 2u = f$, $-\Delta u + 2v = g$. Eliminate: $-\Delta u + 4u = 2f + g$, which by Theorem 9.25 has a unique $u \in H^2 \cap H^1_0$. Then $v = 2u - f \in H^1_0$.

Apply Theorem 7.4 (Hille–Yosida): for $U_0 = (u_0, v_0) \in D(A)$ — hence $u_0 \in H^2 \cap H^1_0$, $v_0 \in H^1_0$ — there is a unique $U \in C^1([0, \infty); H) \cap C([0, \infty); D(A))$ satisfying $dU/dt + AU = 0$, $U(0) = U_0$. This gives $(31)$.

Energy conservation $(32)$: multiply $(27)$ by $\partial u/\partial t$ and integrate over $\Omega$:

$$
\frac{1}{2}\frac{d}{dt}\Big[\Big\lvert\frac{\partial u}{\partial t}\Big\rvert_{L^2}^2 + \lvert\nabla u\rvert_{L^2}^2\Big] = 0. \quad \square
$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Group of isometries)</span></p>

For bounded $\Omega$, an alternative is to work on $H^1_0$ with the *equivalent* scalar product $\int \nabla u_1 \cdot \nabla u_2$ (by Poincaré). Then on $H = H^1_0 \times L^2$, $(AU, U) = -\int \nabla v \cdot \nabla u + \int(-\Delta u)v = 0$ for $U = (u, v) \in D(A)$. So **both $A$ and $-A$ are maximal monotone**, and $A^\star = -A$ — i.e., $A$ is *skew-adjoint*. The semigroup $S_A(t)$ extends to a *group* on $\mathbb{R}$:

$$
\boxed{\;\frac{dU}{dt} - AU = 0\text{ on } [0, +\infty)\quad \Longleftrightarrow\quad \frac{dU}{dt} + AU = 0\text{ on } (-\infty, 0]\text{ (via }t \mapsto -t\text{).}\;}
$$

Equation $(32)$ becomes $\lvert U(t)\rvert_H = \lvert U_0\rvert_H\ \forall t \in \mathbb{R}$, so $\lbrace S_A(t)\rbrace_{t \in \mathbb{R}}$ is a **group of isometries** on $H$. **Time is reversible.**

</div>

#### Higher regularity

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10.8</span><span class="math-callout__name">(Smooth solutions for compatible smooth data)</span></p>

Assume $u_0, v_0 \in H^k(\Omega)$ for every $k$, and the **compatibility conditions**

$$
\Delta^j u_0 = 0\text{ on }\Gamma\quad \forall j \ge 0,\qquad \Delta^j v_0 = 0\text{ on }\Gamma\quad \forall j \ge 0.
$$

Then the solution $u$ of $(27), (28), (29), (30)$ belongs to $C^\infty(\bar\Omega \times [0, \infty))$.

</div>

(The proof identifies $D(A^k)$ explicitly and applies Theorem 7.5.)

#### No smoothing — propagation of singularities

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(No smoothing for the wave equation)</span></p>

The wave equation has **no smoothing effect** on initial data — sharply contrasting with the heat equation. To see this, consider $\Omega = \mathbb{R}$. The explicit **d'Alembert solution** of $(27)$ on $\mathbb{R}$:

$$
\boxed{\;u(x, t) = \tfrac{1}{2}(u_0(x + t) + u_0(x - t)) + \tfrac{1}{2}\int_{x-t}^{x+t} v_0(s)\,ds.\;} \tag{40}
$$

If $v_0 = 0$, $u(x, t) = \tfrac{1}{2}(u_0(x + t) + u_0(x - t))$. So $u$ is *no more regular than $u_0$*. More precisely, if $u_0 \in C^\infty(\mathbb{R} \setminus \lbrace x_0\rbrace)$ with a singularity at $x_0$, then $u$ is $C^\infty$ on $\mathbb{R} \times \mathbb{R}$ except on the two lines $x + t = x_0$ and $x - t = x_0$ — the **characteristics** through $(x_0, 0)$. Thus *singularities propagate along the characteristics*.

</div>

#### Spectral / Fourier method

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Series solution on bounded $\Omega$)</span></p>

For bounded $\Omega$, one can solve $(27)$–$(30)$ via decomposition in the Hilbert basis $(e_i)$ of $L^2(\Omega)$ from §9.8. Seek $u(x, t) = \sum_i a_i(t) e_i(x)$. Substituting into $(27)$ gives $a''_i(t) + \lambda_i a_i(t) = 0$, hence

$$
a_i(t) = a_i(0)\cos(\sqrt{\lambda_i}\,t) + \frac{a'_i(0)}{\sqrt{\lambda_i}}\sin(\sqrt{\lambda_i}\,t),
$$

with $a_i(0) = (u_0, e_i)$ and $a'_i(0) = (v_0, e_i)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Klein–Gordon variant)</span></p>

The methods of §10.3 also apply to the **Klein–Gordon equation**

$$
\frac{\partial^2 u}{\partial t^2} - \Delta u + m^2 u = 0\text{ in } Q,\quad m > 0. \tag{27'}
$$

Note that $(27')$ *cannot* be reduced to $(27)$ by a change of unknown like $v(x, t) = e^{\lambda t}u(x, t)$.

</div>

### Comments on Chapter 10

#### Comments on the heat equation

##### 1. The approach of J.-L. Lions

A general framework that proves existence + uniqueness of *weak solutions* for parabolic problems — viewed as a *parabolic counterpart of Lax–Milgram*. Let $H$ be a Hilbert space (identify $H^\star$ with $H$), $V \subset H$ a Hilbert space densely and continuously injected, so $V \subset H \subset V^\star$. For each $t \in [0, T]$ we are given a bilinear form $a(t; u, v) : V \times V \to \mathbb{R}$ satisfying:

* **(i)** $t \mapsto a(t; u, v)$ measurable for all $u, v \in V$;
* **(ii)** $\lvert a(t; u, v)\rvert \le M\|u\|\|v\|$ a.e.;
* **(iii)** $a(t; v, v) \ge \alpha\|v\|^2 - C\lvert v\rvert^2$ a.e., for some $\alpha > 0, C \ge 0$ (Gårding-type coercivity).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10.9</span><span class="math-callout__name">(J.-L. Lions)</span></p>

Given $f \in L^2(0, T; V^\star)$ and $u_0 \in H$, there exists a unique $u$ with

$$
u \in L^2(0, T; V) \cap C([0, T]; H),\qquad \frac{du}{dt} \in L^2(0, T; V^\star),
$$

satisfying $\langle du/dt(t), v\rangle + a(t; u(t), v) = \langle f(t), v\rangle$ for a.e. $t$, for all $v \in V$, and $u(0) = u_0$.

</div>

**Application.** $H = L^2(\Omega)$, $V = H^1_0(\Omega)$, $a(t; u, v) = \sum \int a_{ij}(x, t)\partial u/\partial x_i \partial v/\partial x_j + \sum \int a_i(x, t)\partial u/\partial x_i v + \int a_0(x, t)uv$ with $a_{ij}, a_i, a_0 \in L^\infty$ and uniform ellipticity yields a weak solution of the parabolic problem with general elliptic operator in space, Dirichlet boundary condition, and initial data $u_0$.

##### 2. $C^\infty$-regularity

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10.10</span><span class="math-callout__name">($C^\infty$ regularity)</span></p>

If $\Omega$ is bounded $C^\infty$ and the coefficients $a_{ij}, a_i, a_0 \in C^\infty(\bar\Omega \times [0, T])$ satisfy ellipticity, then for $u_0 \in L^2$ and $f \in C^\infty(\bar\Omega \times [0, T])$, the solution $u$ of $(43)$ belongs to $C^\infty(\bar\Omega \times [\varepsilon, T])$ for every $\varepsilon > 0$. If additionally $u_0 \in C^\infty(\bar\Omega)$ and $\lbrace f, u_0\rbrace$ satisfy appropriate compatibility conditions on $\Gamma \times \lbrace 0\rbrace$, then $u \in C^\infty(\bar\Omega \times [0, T])$.

</div>

There is also an *abstract* Hille–Yosida-type theory for $du/dt + A(t)u = f(t)$ with each $A(t)$ maximal monotone — Kato, Tanabe, Sobolevski, Friedman, Yosida.

##### 3. $L^p$ and $C^{0, \alpha}$ regularity

For the model heat equation $\partial u/\partial t - \Delta u = f$ in $\Omega \times (0, T)$, $u = 0$ on $\Sigma$, $u(\cdot, 0) = u_0$:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10.11</span><span class="math-callout__name">($L^2$-regularity)</span></p>

For $f \in L^2(\Omega \times (0, T))$ and $u_0 \in H^1_0(\Omega)$, there is a unique solution with

$$
u \in C([0, T]; H^1_0) \cap L^2(0, T; H^2 \cap H^1_0),\qquad \partial u/\partial t \in L^2(0, T; L^2).
$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10.12</span><span class="math-callout__name">($L^p$-regularity)</span></p>

For $1 < p < \infty$, $f \in L^p(\Omega \times (0, T))$ and $u_0 = 0$, there is a unique solution $u$ with $u, \partial u/\partial t, \partial u/\partial x_i, \partial^2 u/\partial x_i \partial x_j \in L^p(\Omega \times (0, T))$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10.13</span><span class="math-callout__name">(Hölder regularity)</span></p>

For $0 < \alpha < 1$, $f \in C^{\alpha, \alpha/2}(\bar\Omega \times [0, T])$ (Hölder $\alpha$ in $x$, $\alpha/2$ in $t$) and $u_0 \in C^{2+\alpha}(\bar\Omega)$ satisfying the natural compatibility conditions $u_0 = 0$ on $\Gamma$, $-\Delta u_0 = f(\cdot, 0)$ on $\Gamma$, there is a unique solution $u$ with $u, \partial u/\partial t, \partial u/\partial x_i, \partial^2 u/\partial x_i \partial x_j \in C^{\alpha, \alpha/2}(\bar\Omega \times [0, T])$.

</div>

The proofs (except $p = 2$) rely on (i) explicit fundamental-solution representations (the **heat kernel** $E(x, t) = (4\pi t)^{-N/2}e^{-\lvert x\rvert^2/4t}$ on $\mathbb{R}^N$) and (ii) singular-integral techniques. The general philosophy: if $u$ solves the heat equation with $u_0 = 0$, then $\partial u/\partial t$ and $\Delta u$ have the *same regularity as $f$*. **Nash–Moser** for irregular $a_{ij} \in L^\infty$: there is some $\alpha > 0$ with $u \in C^{\alpha, \alpha/2}(\bar\Omega \times [0, T])$.

##### 4. Some examples of parabolic equations

Linear and nonlinear parabolic equations occur in mechanics, physics, chemistry, biology, optimal control, probability, finance, image processing, etc.:

* **(i) Navier–Stokes**: $\partial u_i/\partial t - \Delta u_i + \sum_j u_j \partial u_i/\partial x_j = f_i + \partial p/\partial x_i$ with $\mathrm{div}\,u = 0$ (Temam).
* **(ii) Reaction–diffusion systems**: $\partial \mathbf{u}/\partial t - M\Delta\mathbf{u} = f(\mathbf{u})$, $M$ a diagonal matrix, $f : \mathbb{R}^m \to \mathbb{R}^m$ nonlinear. Models phenomena in chemistry, biology, neurophysiology, epidemiology, combustion, population genetics, ecology, geology — with rich behaviors including traveling waves and self-organized patterns.
* **(iii) Free boundary problems**: e.g., the **Stefan problem** (evolution of an ice/water mixture).
* **(iv) Diffusion** in probability — Brownian motion, Markov processes, stochastic differential equations.
* **(v)** Semilinear parabolic problems (Henry, Cazenave–Haraux).
* **(vi)** Atiyah–Singer index via the heat equation.
* **(vii) Image processing** uses sophisticated nonlinear diffusion (Perona–Malik). The **Perelman** proof of the Poincaré conjecture relies on Hamilton's study of **Ricci flow** — a kind of nonlinear heat equation.

##### 5. Maximum principle for parabolic equations

A *strong maximum principle* holds: if $u$ solves $(1), (2), (3)$ with $u_0 \ge 0$ and $u_0 \not\equiv 0$, then $u(x, t) > 0$ for *every* $x \in \Omega$ and $t > 0$ — strict positivity, with infinite speed of propagation.

#### Comments on the wave equation

##### 6. Weak solutions of the wave equation

A general abstract framework in the spirit of Lions (Comment 1):

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10.14</span><span class="math-callout__name">(J.-L. Lions, weak solutions of wave equation)</span></p>

Let $V \subset H \subset V^\star$. Given a $C^1$-in-$t$ symmetric continuous coercive bilinear form $a(t; u, v)$ on $V \times V$, $f \in L^2(0, T; H)$, $u_0 \in V$, $v_0 \in H$, there is a unique $u$ with $u \in C([0, T]; V)$, $du/dt \in C([0, T]; H)$, $d^2 u/dt^2 \in L^2(0, T; V^\star)$, satisfying

$$
\Big\langle\frac{d^2 u}{dt^2}, v\Big\rangle + a(t; u, v) = \langle f, v\rangle\quad \forall v \in V,\ a.e.\ t,
$$

with $u(0) = u_0$, $du/dt(0) = v_0$.

</div>

Note the initial-data assumptions ($u_0 \in H^1_0$, $v_0 \in L^2$) are *weaker* than in Theorem 10.7. Lions–Magenes contains the proof.

##### 7. The $L^p$-theory for wave equation

Delicate; **Strichartz estimates** are an important tool (Klainerman).

##### 8. Maximum principle for the wave equation

Special forms of maximum principle hold (Protter–Weinberger). For example, for $u$ solving $(27), (28), (29), (30)$:

* **(i)** $\Omega = \mathbb{R}$, $u_0 \ge 0$, $v_0 \ge 0$ $\Rightarrow$ $u \ge 0$ (from formula $(40)$).
* **(ii)** $\Omega = \mathbb{R}^N$, $u_0 \ge 0$, $v_0 = 0$ $\Rightarrow$ $u \ge 0$ (Mizohata, Folland, Weinberger, Courant–Hilbert, Mikhlin).
* **(iii)** $\Omega = (0, 1)$, $u_0 \ge 0$, $v_0 = 0$: in general one *cannot* conclude $u \ge 0$.
* **(iv)** $\Omega = \mathbb{R}^2$, $u_0 \ge 0$, $v_0 = 0$: in general one *cannot* conclude $u \ge 0$.

##### 9. Domain of dependence. Wave propagation. Huygens' principle

There is a **fundamental difference** between heat and wave:

* For the **heat equation**, a small perturbation of the initial data is *immediately felt everywhere*: $u_0 \ge 0$ and $u_0 \not\equiv 0$ implies $u(x, t) > 0$ for every $x \in \Omega, t > 0$. The heat propagates **at infinite speed**.
* For the **wave equation**, the situation is **completely different**. For $\Omega = \mathbb{R}$, the d'Alembert formula $(40)$ shows $u(\bar x, \bar t)$ depends *solely* on the values of $u_0, v_0$ in $[\bar x - \bar t, \bar x + \bar t]$ — the **domain of dependence**. In $\mathbb{R}^N$, the domain of dependence is the ball $\lbrace x\,;\ \lvert x - \bar x\rvert \le \bar t\rbrace$. **Waves propagate at speed at most $1$**: a signal localized in a domain $D$ at time $0$ is felt at $x$ only after time $\ge \mathrm{dist}(x, D)$.

For *odd* $N \ge 3$ (e.g., $N = 3$), there is an even more striking effect: $u(\bar x, \bar t)$ depends *only* on values of $u_0, v_0$ on the **sphere** $\lbrace x\,;\ \lvert x - \bar x\rvert = \bar t\rbrace$ — **Huygens' principle**. A signal localized in $D$ at time $0$ is observed at $x$ only during the time interval $[t_1, t_2]$ with $t_1 = \inf_{y \in D}\mathrm{dist}(x, y)$, $t_2 = \sup_{y \in D}\mathrm{dist}(x, y)$. After $t_2$ the signal has *passed*.

For *even* $N$ (e.g., $N = 2$), the signal *persists* at $x$ for all $t > t_1$.

**Application to music.** A listener in $\mathbb{R}^3$ at distance $d$ from a small instrument hears at time $t$ the note played at time $t - d$ and *nothing else*. (In $\mathbb{R}^2$ they would hear a weighted average of all notes played during $[0, t - d]$.)

References on Huygens' principle: Courant–Hilbert, Folland, Garabedian, Mikhlin.

## Chapter 11: Miscellaneous Complements

This chapter gathers technical complements that were left out of the main text to keep the presentation streamlined. The material is connected to Chapters 1–7 and clusters around four themes:

* **§11.1** — finite-dimensional and finite-codimensional subspaces, with the structural theorem on complementability of $G + L$ via finite-dimensional perturbations;
* **§11.2** — quotient spaces $E/M$ and the canonical isomorphism between $(E/M)^\star$ and $M^\perp$;
* **§11.3** — the classical sequence spaces $\ell^p, c, c_0$ — concrete realizations of the abstract reflexivity / separability / duality theory of Chapters 3 and 4;
* **§11.4** — the general theory of Banach spaces over $\mathbb{C}$, where most of Chapters 1–5 carries over verbatim but Chapter 6 (spectrum) acquires a substantially richer flavor (the spectrum is *always* nonempty over $\mathbb{C}$, normal operators have well-behaved spectral theory, etc.).

Several proofs are deliberately sketchy; the interested reader is invited to consult the references.

### 11.1 Finite-Dimensional and Finite-Codimensional Spaces

Recall: every finite-dimensional space $X$ of dimension $p$ is isomorphic to $\mathbb{R}^p$, complete, all norms on $X$ are equivalent, and $B_X$ is compact.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.1</span><span class="math-callout__name">(Finite-dimensional subspaces are closed)</span></p>

Let $E$ be a Banach space and $X \subset E$ a finite-dimensional subspace. Then $X$ is closed.

</div>

(Proof: a Cauchy sequence in $X$ converges in $E$, hence in the complete space $X$.)

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.2</span><span class="math-callout__name">(Operators from finite-dim are bounded)</span></p>

If $X$ is finite-dimensional and $F$ is a Banach space, every linear $T : X \to F$ is bounded.

</div>

In particular, every linear functional on a finite-dimensional space is continuous, $X^\star$ is finite-dimensional with $\dim X^\star = \dim X$, and a basis $(e_i)$ of $X$ induces a dual basis $(f_i)$ of $X^\star$ via $f_i(x) = x_i$ when $x = \sum x_j e_j$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.3</span><span class="math-callout__name">($X^\star$ finite-dim ⇒ $X$ finite-dim)</span></p>

Let $X$ be a Banach space (with $\dim X \le \infty$). If $X^\star$ is finite-dimensional, then $X$ is finite-dimensional and $\dim X = \dim X^\star$.

</div>

(Proof: Hahn–Banach (Corollary 1.4) gives an isometric embedding $J : X \to X^{\star\star}$. Since $\dim X^{\star\star} = \dim X^\star < \infty$, $X$ is finite-dimensional.)

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.4</span><span class="math-callout__name">(Adding a finite-dim subspace preserves closedness + complementability)</span></p>

Let $E$ be a Banach space, $M \subset E$ a closed subspace, and $X \subset E$ a finite-dimensional subspace. Then $M + X$ is closed. Moreover, $M + X$ admits a complement in $E$ if and only if $M$ does.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Warning</span><span class="math-callout__name">(In general $M_1 + M_2$ need not be closed)</span></p>

The sum of two *closed* subspaces of a Banach space need not be closed (Exercise 1.14). Proposition 11.4 says that adding a *finite-dimensional* perturbation does preserve closedness.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Proposition 11.4 (sketch)</summary>

**Closedness.** First reduce to the case $M \cap X = \lbrace 0\rbrace$ by replacing $X$ with a complement of $M \cap X$ in $X$ — a finite-dim manipulation. Then if $u_n = x_n + y_n \in M + X$ converges to $u$, with $x_n \in X$ and $y_n \in M$, we claim $(x_n)$ stays bounded. If not, $\|x_{n_k}\| \to \infty$ for a subsequence, and $x_{n_k}/\|x_{n_k}\|$ has a further subsequence converging to some $\xi \in X$ with $\|\xi\| = 1$. Then $y_{n_k}/\|x_{n_k}\| \to -\xi$, but $M$ is closed, so $\xi \in M \cap X = \lbrace 0\rbrace$ — contradiction. Bounded $(x_n)$ in finite-dim $X$ has a convergent subsequence; the corresponding $y_n$ also converges in $M$.

**Complementability.** If $M$ has a complement $N$ and $X$ has a complement $\widetilde N$ inside $N$ (which exists because $\dim P_N(X) \le \dim X < \infty$), then $\widetilde N$ is a complement of $M + X$. Conversely, given a complement $W$ of $M + X$ and a complement $\widetilde X$ of $M \cap X$ in $X$, $W + \widetilde X$ is a complement of $M$. $\square$

</details>
</div>

#### Codimension

Let $M$ be a subspace of $E$. We say $M$ has **finite codimension** if there is a finite-dimensional $X \subset E$ with $M + X = E$. We may always assume $M \cap X = \lbrace 0\rbrace$. The **codimension** $\mathrm{codim}\,M$ is $\dim X$ — independent of the choice of $X$, and equal to $\dim(E/M)$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Warning</span><span class="math-callout__name">(Finite codim ⇏ closed)</span></p>

A subspace of finite codimension need *not* be closed. For example, take a discontinuous linear functional $f$ on an infinite-dim $E$ (Exercise 1.5). Then $M = f^{-1}(\lbrace 0\rbrace)$ has codimension 1 but is *not closed* (Proposition 1.5); in fact $M$ is dense in $E$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.5</span><span class="math-callout__name">(Sandwich between closed finite-codim subspaces)</span></p>

Let $M$ be a closed subspace of finite codimension. Any subspace $\widetilde M$ of $E$ containing $M$ is closed.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.6</span><span class="math-callout__name">(Complement inside a dense subspace)</span></p>

If $M$ is closed of finite codimension and $D$ is a dense subspace of $E$, there exists a complement $X$ of $M$ with $X \subset D$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.7</span><span class="math-callout__name">(Complementability for $G + L$ closed)</span></p>

Let $G, L \subset E$ be closed subspaces. Assume there exist finite-dimensional $X_1, X_2 \subset E$ with

$$
G + L + X_1 = E\quad \text{and}\quad G \cap L \subset X_2.
$$

Then $G$ (resp. $L$) admits a complement.

</div>

(Proof: a two-step argument — first the case $X_2 = \lbrace 0\rbrace$, then localize to $G \cap L$ via a finite-dim splitting.)

### 11.2 Quotient Spaces

Let $E$ be a Banach space and $M \subset E$ a closed subspace. The equivalence relation $x \sim y \iff x - y \in M$ partitions $E$ into equivalence classes; the **quotient space** $E/M$ is the vector space of these classes. The canonical projection $\pi : E \to E/M$, $\pi(x) = [x]$, is surjective and linear. The **quotient norm** is

$$
\boxed{\;\|[x]\|_{E/M} = \inf_{m \in M}\|x - m\|.\;}
$$

This is a norm (using $M$ closed), and $\pi$ is bounded with $\|\pi\| \le 1$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.8</span><span class="math-callout__name">($E/M$ is Banach)</span></p>

Equipped with the quotient norm, $E/M$ is a Banach space.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Take a Cauchy sequence $(\pi(x_k))$ in $E/M$. Pass to a subsequence with $\|\pi(x_{k+1}) - \pi(x_k)\|_{E/M} < 1/2^k$. Choose $m_k \in M$ with $\|x_{k+1} - x_k - m_k\| < 1/2^k$ (using the inf in the quotient norm). Set $\bar x_k = x_k - \sum_{j < k} m_j$; then $\|\bar x_{k+1} - \bar x_k\| < 1/2^k$, so $(\bar x_k)$ is Cauchy in $E$, with limit $\ell \in E$. Then $\pi(x_k) = \pi(\bar x_k) \to \pi(\ell)$. $\square$

</details>
</div>

#### Duality: $(E/M)^\star \cong M^\perp$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.9</span><span class="math-callout__name">($\pi^\star$ is an isometry $(E/M)^\star \to M^\perp$)</span></p>

Let $M \subset E$ be a closed subspace and $\pi^\star : (E/M)^\star \to E^\star$ the adjoint of $\pi$. Then $R(\pi^\star) = M^\perp$, and $\pi^\star$ is an *isometric isomorphism* from $(E/M)^\star$ onto $M^\perp$:

$$
\boxed{\;(E/M)^\star \cong M^\perp\quad \text{(isomorphism + isometry).}\;}
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.10</span><span class="math-callout__name">($E^\star/M^\perp \cong M^\star$)</span></p>

Let $T : E^\star \to M^\star$ be restriction-to-$M$. Then $T$ factors as $T = \widetilde T \circ \pi$ with $\widetilde T : E^\star/M^\perp \to M^\star$, and $\widetilde T$ is a bijective *isometry*:

$$
\boxed{\;E^\star/M^\perp \cong M^\star\quad \text{(isomorphism + isometry).}\;}
$$

</div>

(Proof: Hahn–Banach (Corollary 1.2) ensures every $f \in M^\star$ extends to $\tilde f \in E^\star$ with $\|\tilde f\| = \|f\|$, and the equivalence class $[\tilde f] \in E^\star/M^\perp$ is independent of the extension.)

#### Inheritance of structural properties

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.11</span><span class="math-callout__name">(Reflexivity descends to quotients)</span></p>

If $E$ is reflexive and $M \subset E$ is a closed subspace, then $E/M$ is reflexive.

</div>

(Proof: $E^\star$ is reflexive (Corollary 3.21), so its closed subspace $M^\perp$ is reflexive (Proposition 3.20). By Proposition 11.9, $(E/M)^\star \cong M^\perp$ is reflexive, hence so is $E/M$ — Corollary 3.21 again.)

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.12</span><span class="math-callout__name">(Uniform convexity descends)</span></p>

If $E$ is uniformly convex and $M \subset E$ is a closed subspace, then $E/M$ is uniformly convex.

</div>

(Proof: given $\pi(x), \pi(y) \in E/M$ with norms $\le 1$ and $\|\pi(x) - \pi(y)\| > \varepsilon$, reflexivity of $E$ + Corollary 3.23 give $m_1, m_2 \in M$ achieving $\|x - m_1\|, \|y - m_2\| \le 1$. Apply uniform convexity in $E$ to $(x - m_1)$ and $(y - m_2)$.)

#### Dimension / codimension duality

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.13</span><span class="math-callout__name">(Dimension ↔ codimension via $\perp$)</span></p>

Let $M \subset E$ be a closed subspace.

* **(a)** $\dim M < \infty$ iff $\mathrm{codim}\,M^\perp < \infty$, in which case $\dim M = \mathrm{codim}\,M^\perp$.
* **(b)** $\mathrm{codim}\,M < \infty$ iff $\dim M^\perp < \infty$, in which case $\mathrm{codim}\,M = \dim M^\perp$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.14</span><span class="math-callout__name">(Dual statement, partially true)</span></p>

Let $N \subset E^\star$ be a closed subspace. Then $\dim N < \infty$ iff $\mathrm{codim}\,N^\perp < \infty$, and in that case $\dim N = \mathrm{codim}\,N^\perp$. It is *also* true that $\dim N^\perp \le \mathrm{codim}\,N$, but in general one may have $\dim N^\perp < \mathrm{codim}\,N < \infty$.

</div>

(The asymmetry comes from the failure of $\overline N = N^{\perp\perp}$ in general — see the remark after Proposition 1.9. Concrete example: $\xi \in E^{\star\star} \setminus J(E)$ and $N = \xi^{-1}(\lbrace 0\rbrace) \subset E^\star$ a hyperplane, but $N^\perp = \lbrace 0\rbrace$ in $E$.)

### 11.3 Some Classical Spaces of Sequences

For $x = (x_1, x_2, \ldots, x_k, \ldots)$ set

$$
\|x\|_p = \Big(\sum_{k=1}^\infty \lvert x_k\rvert^p\Big)^{1/p},\ 1 \le p < \infty,\qquad \|x\|_\infty = \sup_k \lvert x_k\rvert,
$$

and define

$$
\boxed{\;\ell^p = \lbrace x\,;\ \|x\|_p < \infty\rbrace,\quad \ell^\infty = \lbrace x\,;\ \|x\|_\infty < \infty\rbrace.\;}
$$

These are Banach spaces (either via Theorem 4.8 with $\Omega = \mathbb{N}$ + counting measure, or directly).

Two important subspaces of $\ell^\infty$:

$$
c = \lbrace x\,;\ \lim_k x_k\text{ exists}\rbrace,\qquad c_0 = \lbrace x\,;\ \lim_k x_k = 0\rbrace,
$$

both equipped with the $\ell^\infty$ norm; $c_0 \subset c \subset \ell^\infty$ with $c_0$ closed in $c$ and $c$ closed in $\ell^\infty$. Hölder's inequality reads

$$
\Big\lvert\sum x_k y_k\Big\rvert \le \|x\|_p \|y\|_{p'}\quad \forall x \in \ell^p, y \in \ell^{p'},\quad \tfrac{1}{p} + \tfrac{1}{p'} = 1.
$$

$\ell^2$ is a Hilbert space with $(x, y) = \sum x_k y_k$. The inclusions $\ell^p \subset \ell^q$ for $p \le q$ hold with $\|x\|_q \le \|x\|_p$; $\ell^p \subset c_0$ for every $p < \infty$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.15</span><span class="math-callout__name">($\ell^p$ is reflexive + uniformly convex, $1 < p < \infty$)</span></p>

$\ell^p$ is uniformly convex (and thus reflexive) for $1 < p < \infty$.

</div>

(Apply Theorem 4.10 + Exercise 4.12 with $\Omega = \mathbb{N}$.)

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.16</span><span class="math-callout__name">(Separability)</span></p>

$c, c_0$ and $\ell^p$ for $1 \le p < \infty$ are separable.

</div>

(Proof: $D = \lbrace x = (x_k)\,;\ x_k \in \mathbb{Q},\ x_k = 0\text{ for }k\text{ large}\rbrace$ is countable and dense in $\ell^p$ ($p < \infty$) and $c_0$; $D + \lambda(1, 1, 1, \ldots)$ with $\lambda \in \mathbb{Q}$ is countable dense in $c$.)

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.17</span><span class="math-callout__name">($\ell^\infty$ is *not* separable)</span></p>

$\ell^\infty$ is not separable.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

For any countable $A = (a^k) \subset \ell^\infty$, define $b = (b_k)$ by $b_k = a_k^k + 1$ if $\lvert a_k^k\rvert \le 1$, $b_k = 0$ otherwise. Then $\lvert b_k - a_k^k\rvert \ge 1\ \forall k$, so $\|b - a^k\|_\infty \ge 1$ — Cantor diagonalization in disguise. $\square$

</details>
</div>

#### Duals of sequence spaces

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.18</span><span class="math-callout__name">($(\ell^p)^\star = \ell^{p'}$, $1 \le p < \infty$)</span></p>

Given $\phi \in (\ell^p)^\star$, there exists a unique $u = (u_k) \in \ell^{p'}$ with

$$
\langle\phi, x\rangle = \sum_{k=1}^\infty u_k x_k\quad \forall x \in \ell^p,\qquad \|u\|_{p'} = \|\phi\|_{(\ell^p)^\star}.
$$

</div>

(Proof: take $u_k = \phi(e_k)$ with $e_k$ the canonical basis. For $1 < p < \infty$, choose $x_k = \lvert u_k\rvert^{p'-2} u_k$ truncated to get a lower bound on $\|u\|_{p'}$.)

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.19</span><span class="math-callout__name">($(c_0)^\star = \ell^1$)</span></p>

Given $\phi \in (c_0)^\star$, there is a unique $u \in \ell^1$ with $\langle\phi, x\rangle = \sum u_k x_k\ \forall x \in c_0$, and $\|u\|_1 = \|\phi\|_{(c_0)^\star}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.20</span><span class="math-callout__name">($c^\star = \ell^1 \times \mathbb{R}$)</span></p>

Given $\phi \in c^\star$, there is a unique pair $(u, \lambda) \in \ell^1 \times \mathbb{R}$ such that

$$
\langle\phi, x\rangle = \sum_{k=1}^\infty u_k x_k + \lambda\lim_{k \to \infty} x_k\quad \forall x \in c,
$$

with $\|u\|_1 + \lvert\lambda\rvert = \|\phi\|_{c^\star}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.21</span><span class="math-callout__name">($\ell^1, \ell^\infty, c, c_0$ are not reflexive)</span></p>

The spaces $\ell^1, \ell^\infty, c, c_0$ are *not* reflexive.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

From Proposition 11.18 ($p = 1$) and Proposition 11.19, $(c_0)^\star = \ell^1$ and $(\ell^1)^\star = \ell^\infty$. The canonical injection $J : c_0 \to (c_0)^{\star\star} = \ell^\infty$ is the identity inclusion, which is *not* surjective (since $c_0 \subsetneq \ell^\infty$). Hence $c_0$ is not reflexive. By Corollary 3.21, $\ell^1$ and $\ell^\infty$ are not reflexive. By Proposition 3.20, $c$ — having $c_0$ as a closed subspace — cannot be reflexive either. $\square$

</details>
</div>

#### Summary table

| | Reflexive | Separable | Dual |
| --- | --- | --- | --- |
| $\ell^p,\ 1 < p < \infty$ | YES | YES | $\ell^{p'}$ |
| $\ell^1$ | NO | YES | $\ell^\infty$ |
| $c_0$ | NO | YES | $\ell^1$ |
| $c$ | NO | YES | $\ell^1 \times \mathbb{R}$ |
| $\ell^\infty$ | NO | NO | strictly bigger than $\ell^1$ |

### 11.4 Banach Spaces over $\mathbb{C}$: What Is Similar and What Is Different?

We now sketch the changes when the scalar field is $\mathbb{C}$ instead of $\mathbb{R}$. **Most of Chapters 1–5 carries over verbatim with cosmetic adjustments**; the *major* change is in Chapter 6 (spectrum / eigenvalues), where the complex theory is *substantially richer*.

For an $\mathbb{R}$-vector space underlying a $\mathbb{C}$-vector space $E$, write $E_\mathbb{R}$ for $E$ regarded over $\mathbb{R}$. **Warning:** an $\mathbb{R}$-linear subspace of $E$ need not be $\mathbb{C}$-linear (e.g., a line in $\mathbb{R}^2 \cong \mathbb{C}$ rotated by $\pi/2$ is an $\mathbb{R}$-subspace but not a $\mathbb{C}$-subspace).

A norm on $E$ over $\mathbb{C}$ is a function $E \to [0, +\infty)$ with $\|\lambda x\| = \lvert\lambda\rvert\|x\|\ \forall \lambda \in \mathbb{C}$ (so it is also a norm on $E_\mathbb{R}$, but not conversely).

#### Real ↔ complex duals

A linear functional on a $\mathbb{C}$-space $E$ is a $\mathbb{C}$-linear $f : E \to \mathbb{C}$. Notation $\langle f, x\rangle = f(x) \in \mathbb{C}$, $\|f\|_{E^\star} = \sup_{\|x\| \le 1}\lvert f(x)\rvert$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.22</span><span class="math-callout__name">(Complex dual ↔ real dual via $\mathrm{Re}$)</span></p>

The map $I : f \in E^\star \mapsto \mathrm{Re}\,f \in E_\mathbb{R}^\star$ is a *bijective isometry* from $E^\star$ onto $E_\mathbb{R}^\star$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

**Inverse formula.** Given $\varphi \in E_\mathbb{R}^\star$, define $f : E \to \mathbb{C}$ by

$$
f(x) = \varphi(x) - i\,\varphi(ix)\quad \forall x \in E.
$$

Verify $f$ is $\mathbb{C}$-linear: $f(\lambda x) = \varphi(\lambda x) - i\varphi(i\lambda x)$ for $\lambda \in \mathbb{C}$. Splitting $\lambda = a + ib$ and using $\varphi$'s $\mathbb{R}$-linearity gives $f(\lambda x) = \lambda f(x)$. Then $\mathrm{Re}\,f = \varphi$, so $I$ is surjective.

**Isometry.** Let $f \in E^\star$. Clearly $\lvert\mathrm{Re}\,f(x)\rvert \le \lvert f(x)\rvert \le \|f\|\|x\|$, so $\|\mathrm{Re}\,f\|_{E_\mathbb{R}^\star} \le \|f\|_{E^\star}$. For the reverse, given $x \neq 0$ with $f(x) \neq 0$, set $\lambda = f(x)/\lvert f(x)\rvert$ ($\lvert\lambda\rvert = 1$). Then $\lvert f(x)\rvert = \tfrac{1}{\lambda}f(x) = f(x/\lambda)$. Since the result is real and nonneg, $\lvert f(x)\rvert = \mathrm{Re}\,f(x/\lambda) = \varphi(x/\lambda) \le \|\varphi\|\|x\|$. Hence $\|f\| \le \|\varphi\| = \|I(f)\|$. $\square$

</details>
</div>

This proposition lets one *transport* almost every result from the real to the complex setting, by working in $E_\mathbb{R}$ and using $\mathrm{Re}$.

#### Chapters 1–5: minor adjustments

* **Chapter 1.** Real Hahn–Banach extends a continuous $\mathbb{C}$-linear $g : G \to \mathbb{C}$ to $f \in E^\star$ with $\|f\| = \|g\|$ (Proposition 11.23). The geometric form (Proposition 11.24): nonempty disjoint convex $A, B$ with one open are separated by a *closed real* hyperplane $\lbrace \mathrm{Re}\,f = \alpha\rbrace$. Conjugates: $\varphi^\star(f) = \sup_x\lbrace \mathrm{Re}\langle f, x\rangle - \varphi(x)\rbrace$, and Theorem 1.11 (Fenchel–Moreau, Proposition 11.25) is unchanged.
* **Chapter 2.** All statements unchanged (with $\mathbb{R}$ replaced by $\mathbb{C}$).
* **Chapter 3.** All statements unchanged.
* **Chapter 4.** Totally unchanged.
* **Chapter 5.** A *complex Hilbert space* $H$ has a sesquilinear form $(u, v) : H \times H \to \mathbb{C}$ with $(u, v) = \overline{(v, u)}$, $v \mapsto (u, v)$ linear, and $(u, u) > 0\ \forall u \neq 0$. Then $\lvert u\rvert = (u, u)^{1/2}$ is a norm with $\lvert u + v\rvert^2 = \lvert u\rvert^2 + 2\mathrm{Re}(u, v) + \lvert v\rvert^2$ and Cauchy–Schwarz $\lvert(u, v)\rvert \le \lvert u\rvert\lvert v\rvert$. **Example.** $L^2(\Omega; \mathbb{C})$ with $(u, v) = \int u\overline v\,d\mu$.

If $H$ is a complex Hilbert space, $H_\mathbb{R}$ with $\mathrm{Re}(u, v)$ is a real Hilbert space, so all of Chapter 5 transfers. The complex versions of the projection theorem, Riesz–Fréchet, and Lax–Milgram read:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.26</span><span class="math-callout__name">(Projection — complex case)</span></p>

For $K \subset H$ nonempty closed convex, every $f \in H$ has a unique nearest point $u \in K$, characterized by

$$
u \in K\quad \text{and}\quad \mathrm{Re}(f - u, v - u) \le 0\quad \forall v \in K.
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.27</span><span class="math-callout__name">(Riesz–Fréchet — complex case)</span></p>

For every $\varphi \in H^\star$ there is a unique $f \in H$ with $\varphi(u) = (u, f)\ \forall u \in H$, and $\lvert f\rvert = \|\varphi\|$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.28</span><span class="math-callout__name">(Stampacchia — complex case)</span></p>

Let $a : H \times H \to \mathbb{C}$ be linear in the first argument, conjugate-linear in the second, continuous, and *coercive* in the sense $\mathrm{Re}\,a(u, u) \ge \alpha\lvert u\rvert^2$. Let $K \subset H$ be nonempty closed convex. For every $\varphi \in H^\star$ there is a unique $u \in K$ satisfying

$$
\mathrm{Re}\,a(u, v - u) \ge \mathrm{Re}\langle\varphi, v - u\rangle\quad \forall v \in K,
$$

and if $a$ is *Hermitian-symmetric* ($a(v, w) = \overline{a(w, v)}$), $u$ minimizes $\tfrac{1}{2}a(v, v) - \mathrm{Re}\langle\varphi, v\rangle$ on $K$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.29</span><span class="math-callout__name">(Lax–Milgram — complex case)</span></p>

If $T \in \mathcal{L}(H)$ satisfies $\lvert(Tu, u)\rvert \ge \alpha\lvert u\rvert^2\ \forall u$, then $T$ is bijective.

</div>

#### Chapter 6 over $\mathbb{C}$: the major change — the spectrum

**Sections 6.1 and 6.2 are totally unchanged** — the Riesz–Fredholm theory works equally well over $\mathbb{C}$.

The **major change** appears in Section 6.3 (spectrum). For $T \in \mathcal{L}(E)$ with $E$ over $\mathbb{C}$, define resolvent / spectrum / eigenvalues exactly as in §6.3 but with $\lambda \in \mathbb{C}$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.30</span><span class="math-callout__name">(Compactness + non-emptiness of $\sigma(T)$ over $\mathbb{C}$)</span></p>

For $T \in \mathcal{L}(E)$ with $E$ a Banach space over $\mathbb{C}$, $\sigma(T)$ is a *nonempty* compact subset of $\mathbb{C}$, contained in $\lbrace\lambda\,;\ \lvert\lambda\rvert \le \|T\|\rbrace$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why $\sigma(T) \neq \emptyset$ over $\mathbb{C}$ — Liouville)</span></p>

The proof relies on **complex analysis / Liouville's theorem**: if $\sigma(T) = \emptyset$, then $\lambda \mapsto (T - \lambda I)^{-1}$ would be entire and bounded on $\mathbb{C}$, hence constant — absurd. (References: Taylor–Lay, Rudin, Knapp.)

This nonemptiness *fails* over $\mathbb{R}$: a rotation by $\pi/2$ in $\mathbb{R}^2$ has empty real spectrum but eigenvalues $\pm i \in \mathbb{C}$.

</div>

#### Spectral radius

The **spectral radius** $r(T) = \lim_n \|T^n\|^{1/n}$ exists (Exercise 6.23), with $r(T) \le \|T\|$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.31</span><span class="math-callout__name">(Spectral radius formula)</span></p>

For $T \in \mathcal{L}(E)$ over $\mathbb{C}$,

$$
\boxed{\;r(T) = \max\lbrace\lvert\lambda\rvert\,;\ \lambda \in \sigma(T)\rbrace.\;}
$$

</div>

(References: Taylor–Lay, Rudin, Knapp. Over $\mathbb{R}$ one only has $\max\lvert\lambda\rvert \le r(T)$ with possible strict inequality even when $\sigma(T) \neq \emptyset$.)

#### Spectral mapping theorem

For polynomials $Q(t) = \sum a_k t^k$:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.32</span><span class="math-callout__name">(Polynomial spectral mapping over $\mathbb{C}$)</span></p>

For $T \in \mathcal{L}(E)$ over $\mathbb{C}$ and $Q \in \mathbb{C}[t]$,

$$
Q(\sigma(T)) = \sigma(Q(T)),\qquad Q(EV(T)) = EV(Q(T)).
$$

</div>

(Both inclusions $\subset$ are easy; the reverse uses factorization $Q(t) - \mu = \alpha\prod(t - t_i)$ over $\mathbb{C}$ — *not available* over $\mathbb{R}$, hence the equalities can fail in the real case, where one only has $\subset$.)

#### Numerical range

For $H$ complex Hilbert and $T \in \mathcal{L}(H)$, the **numerical range** is

$$
\boxed{\;W(T) = \lbrace(Tu, u)\,;\ u \in H,\ \lvert u\rvert = 1\rbrace \subset \mathbb{C}.\;}
$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.33</span><span class="math-callout__name">($\sigma(T) \subset \overline{W(T)}$ + convexity)</span></p>

$\sigma(T) \subset \overline{W(T)}$, and more precisely if $\lambda \notin \overline{W(T)}$ then $\lambda \in \rho(T)$ with

$$
\|(T - \lambda I)^{-1}\| \le 1/\mathrm{dist}(\lambda, \overline{W(T)}).
$$

Furthermore, $W(T)$ is **convex** (Toeplitz–Hausdorff).

</div>

(For $\sigma(T) \subset \overline{W(T)}$: dist condition + Lax–Milgram. The convexity of $W(T)$ is a counterintuitive theorem — see Halmos.)

#### Adjoints in complex Hilbert spaces

A small point of caution. For $T \in \mathcal{L}(H)$ over $\mathbb{C}$, the abstract adjoint via $H^\star$ satisfies $T^\star \in \mathcal{L}(H^\star)$ and $(\lambda T)^\star = \lambda T^\star$ (genuine, since the dual map $H^\star \to \mathbb{C}$ is $\mathbb{C}$-linear). After identifying $H^\star \cong H$ via Riesz–Fréchet, however, one defines $T^\star \in \mathcal{L}(H)$ by

$$
\boxed{\;(Tu, v) = (u, T^\star v)\quad \forall u, v \in H,\;}
$$

and then

$$
(\lambda T)^\star = \overline\lambda\,T^\star\quad \forall \lambda \in \mathbb{C}.
$$

This convention is standard. $T$ is **self-adjoint** (Hermitian) if $T^\star = T$, equivalently $(Tu, v) = (u, Tv)$. Then $(Tu, u) \in \mathbb{R}$, so $W(T) \subset \mathbb{R}$ and $\sigma(T) \subset \mathbb{R}$.

#### Spectral decomposition over $\mathbb{C}$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.34</span><span class="math-callout__name">(Spectral theorem for compact self-adjoint, complex case)</span></p>

Let $H$ be a separable complex Hilbert space and $T \in \mathcal{K}(H)$ self-adjoint. Then $H$ has a Hilbert basis of eigenvectors of $T$, with corresponding eigenvalues *real*.

</div>

(Identical to Theorem 6.11.) But the complex theory goes further:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Normal operator)</span></p>

$T \in \mathcal{L}(H)$ is **normal** if $T^\star \circ T = T \circ T^\star$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.35</span><span class="math-callout__name">(Spectral radius for normal operators)</span></p>

If $T \in \mathcal{L}(H)$ is normal on a complex Hilbert space, then

$$
\max\lbrace\lvert\lambda\rvert\,;\ \lambda \in \sigma(T)\rbrace = \|T\|.
$$

</div>

(Proof: normality $\Rightarrow \|T^p\| = \|T\|^p\ \forall p$, so $r(T) = \|T\|$, and Proposition 11.31 gives the conclusion.)

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.36</span><span class="math-callout__name">(Spectral theorem for compact normal, complex case)</span></p>

Let $H$ be a separable complex Hilbert space and $T \in \mathcal{K}(H)$ *normal*. Then $H$ has a Hilbert basis of eigenvectors of $T$ — but the eigenvalues may now be *complex*.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Numerical range of a normal compact)</span></p>

For compact normal $T$, $\overline{W(T)} = \mathrm{conv}\,\sigma(T)$ — the closed convex hull. (For *general* normal $T$, this still holds; see Halmos.)

</div>

#### Isometries and unitaries

$T \in \mathcal{L}(H)$ is an **isometry** if $\lvert Tu\rvert = \lvert u\rvert\ \forall u$ (equivalently $T^\star T = I$); a **unitary** is an isometry that is also surjective ($T^\star T = T T^\star = I$).

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11.37</span><span class="math-callout__name">(Spectrum of isometries / unitaries)</span></p>

* If $T$ is an isometry, $EV(T) \subset S^1 = \lbrace\lambda \in \mathbb{C}\,;\ \lvert\lambda\rvert = 1\rbrace$.
* If $T$ is unitary, $\sigma(T) \subset S^1$.
* If $T$ is an isometry but **not** unitary, $\sigma(T) = \overline{D} = \lbrace\lambda\,;\ \lvert\lambda\rvert \le 1\rbrace$ — the *closed* unit disk.

</div>

A **skew-adjoint** (or antisymmetric) operator is one with $T^\star = -T$. Equivalently, $iT$ is self-adjoint, so $EV(iT) \subset \mathbb{R}$, hence $EV(T) \subset i\mathbb{R}$ and $\sigma(T) \subset \overline{W(T)} \subset i\mathbb{R}$.

#### Chapter 7 over $\mathbb{C}$

Very little needs to change. The notion of monotone operator becomes: $\mathrm{Re}(Av, v) \ge 0\ \forall v \in D(A)$. Many computations in §§7.2–7.4 rely on the identity (for $\varphi \in C^1([0, +\infty); H)$):

$$
\frac{d}{dt}\lvert\varphi\rvert^2 = \frac{d}{dt}(\varphi, \varphi) = \Big(\frac{d\varphi}{dt}, \varphi\Big) + \Big(\varphi, \frac{d\varphi}{dt}\Big) = 2\,\mathrm{Re}\Big(\frac{d\varphi}{dt}, \varphi\Big).
$$

#### Chapters 8 and 9 over $\mathbb{C}$

Properties of the spectrum of *non-self-adjoint* second-order elliptic operators are richer over $\mathbb{C}$. See S. Agmon, Section 16.
