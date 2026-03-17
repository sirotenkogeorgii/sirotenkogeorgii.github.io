---
layout: default
title: "Course of Analysis for Computer Scientists — Semester 3"
date: 2026-03-16
excerpt: Notes from the third semester of Analysis (Pultr), covering advanced metric space theory, compactness, Baire category, and completions.
tags:
  - analysis
  - metric-spaces
  - topology
---

**Table of Contents**
- TOC
{:toc}

# XVII. More about Metric Spaces

## 1. Separability and Countable Bases

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Density)</span></p>

Recall the closure from XIII.3.6. A subset $M$ of a metric space $(X, d)$ is **dense** if $\overline{M} = X$. In other words, $M$ is dense if for each $x \in X$ and each $\varepsilon > 0$ there is an $m \in M$ such that $d(x, m) < \varepsilon$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Separable Space)</span></p>

A metric space $(X, d)$ is said to be **separable** if there exists a countable dense subset $M \subseteq X$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Basis of Open Sets)</span></p>

A subset $\mathcal{B}$ of the set $\operatorname{Open}(X, d)$ of all open sets in $(X, d)$ is said to be a **basis** (of open sets) if every open set is a union of sets from $\mathcal{B}$, that is, if

$$\forall U \in \operatorname{Open}(X) \;\; \exists \mathcal{B}_U \subseteq \mathcal{B} \;\;\text{such that}\;\; U = \bigcup \lbrace B \mid B \in \mathcal{B}_U \rbrace.$$

In other words,

$$\forall U \in \operatorname{Open}(X) \;\; U = \bigcup \lbrace B \mid B \in \mathcal{B}_U,\; B \subseteq U \rbrace.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span></p>

1. The set of all open intervals $(a, b)$, or already the set of all $(a, b)$ with rational $a, b$, is a basis of open sets of the real line $\mathbb{R}$.
2. In every metric space the set $\lbrace \Omega(x, \tfrac{1}{n}) \mid x \in X,\; n = 1, 2, \dots \rbrace$ is a basis (recall XIII.3.2).
3. The term "basis" here clashes with the homonymous term from linear algebra. There is no minimality or independence — rather, we have a generating set.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Cover, Subcover)</span></p>

A **cover** of a space $(X, d)$ is a subset $\mathcal{U} \subseteq \operatorname{Open}(X, d)$ such that $\bigcup \lbrace U \mid U \in \mathcal{U} \rbrace = X$. A **subcover** $\mathcal{V}$ of a cover $\mathcal{U}$ is a subset $\mathcal{V} \subseteq \mathcal{U}$ such that (still) $\bigcup \lbrace U \mid U \in \mathcal{V} \rbrace = X$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lindelöf Property)</span></p>

A space $X = (X, d)$ is said to be **Lindelöf** or to have the **Lindelöf property** if every cover of $X$ has a countable subcover.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1.6</span><span class="math-callout__name">(Separability, Countable Basis, Lindelöf)</span></p>

The following statements about a metric space $X$ are equivalent.

1. $X$ is separable.
2. $X$ has a countable basis.
3. $X$ has the Lindelöf property.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**(1)$\Rightarrow$(2):** Let $X$ be separable and let $M$ be a countable dense subset. Set

$$\mathcal{B} = \lbrace \Omega(m, r) \mid m \in M,\; r \text{ rational} \rbrace.$$

$\mathcal{B}$ is obviously countable; we show it is a basis. Let $U$ be open and let $x \in U$. Then there is an $\varepsilon > 0$ such that $\Omega(x, \varepsilon) \subseteq U$. Choose an $m_x \in M$ and a rational $r_x$ such that $d(x, m_x) < \tfrac{1}{3}\varepsilon$ and $\tfrac{1}{3}\varepsilon < r_x < \tfrac{2}{3}\varepsilon$. Then

$$x \in \Omega(m_x, r_x) \subseteq \Omega(x, \varepsilon) \subseteq U.$$

Indeed, $x \in \Omega(m_x, r_x)$ trivially and if $y \in \Omega(m_x, r_x)$ then $d(x, y) \le d(x, m_x) + d(m_x, y) < \tfrac{1}{3}\varepsilon + \tfrac{2}{3}\varepsilon = \varepsilon$. Thus $U = \bigcup \lbrace \Omega(m_x, r_x) \mid x \in U \rbrace$.

**(2)$\Rightarrow$(3):** Let $\mathcal{B}$ be a countable basis and let $\mathcal{U}$ be a cover of $X$. Since $U = \bigcup \lbrace B \mid B \in \mathcal{B},\; B \subseteq U \rbrace$ for each $U \in \mathcal{U}$ we have

$$X = \bigcup \lbrace B \in \mathcal{B} \mid \exists U_B \supseteq B,\; U_B \in U \rbrace.$$

The cover $\mathcal{A} = \lbrace B \in \mathcal{B} \mid \exists U_B \supseteq B,\; U_B \in U \rbrace$ is countable and hence so is also the cover $\mathcal{V} = \lbrace U_B \mid B \in \mathcal{A} \rbrace$.

**(3)$\Rightarrow$(1):** Let $X$ be Lindelöf. For covers $\mathcal{U}_n = \lbrace \Omega(x, \tfrac{1}{n}) \mid x \in X \rbrace$, choose countable subcovers

$$\Omega(x_{n1}, \tfrac{1}{n}),\; \Omega(x_{n2}, \tfrac{1}{n}),\; \dots,\; \Omega(x_{nk}, \tfrac{1}{n}),\; \dots$$

Then $\lbrace x_{nk} \mid n = 1, 2, \dots,\; k = 1, 2, \dots \rbrace$ is dense. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 1.7</span></p>

1. In more general *topological spaces*, the equivalence of Theorem 1.6 does not hold. The existence of a countable basis implies both separability and the Lindelöf property, but none of the other implications hold in general.
2. A countable basis is inherited by every subspace (recall XIII.3.4.3), so for metric spaces:
   - every subspace of a separable space is separable, and
   - every subspace of a Lindelöf space is Lindelöf.

   The latter statement is somewhat surprising — compare with compactness, which is inherited by closed subspaces only.

</div>

---

## 2. Totally Bounded Metric Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Totally Bounded)</span></p>

A metric space $(X, d)$ is **totally bounded** if

$$\forall \varepsilon > 0 \;\; \exists \text{ finite } M(\varepsilon) \;\;\text{such that}\;\; \forall\, x \in X,\;\; d(x, M(\varepsilon)) < \varepsilon.$$

Obviously every totally bounded space is bounded (recall XIII.7.4), but not every bounded space is totally bounded: take an infinite set $X$ with $d(x, y) = 1$ for $x \ne y$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation 2.1.1</span></p>

Total boundedness (and plain boundedness as well) is preserved when replacing a metric by a strongly equivalent one (recall XIII.4), but it is not a topological property. (For the second statement, consider the bounded open interval $(a, b)$ and the real line $\mathbb{R}$; recall XIII.6.8.)

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.2</span></p>

A subspace of a totally bounded $(X, d)$ is totally bounded.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $Y \subseteq X$. For $\varepsilon > 0$ take the $M(\tfrac{\varepsilon}{2}) \subseteq X$ from the definition and set

$$M_Y = \lbrace a \in M(\tfrac{\varepsilon}{2}) \mid \exists y \in Y,\; d(a, y) < \tfrac{\varepsilon}{2} \rbrace.$$

Now for each $a \in M_Y$ choose an $a_Y \in Y$ such that $d(a, a_Y) < \tfrac{\varepsilon}{2}$ and set $N(\varepsilon) = \lbrace a_Y \mid a \in M_Y \rbrace$. Then for every $y \in Y$ we have $d(y, N(\varepsilon)) < \varepsilon$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.3</span></p>

A product $X = \prod_{j=1}^{n} (X_j, d_j)$ of totally bounded spaces is totally bounded.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

For the product take the distance $d$ from XIII.5. Then if we take for $X_i$ the $M_i(\varepsilon)$ from the definition, the set $M(\varepsilon) = \prod M_i(\varepsilon)$ has the property needed for $X$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.4</span></p>

A subspace of $\mathbb{E}_n$ is totally bounded if and only if it is bounded.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By 2.2 and 2.3 it suffices to prove that the interval $\langle a, b \rangle$ is totally bounded. For $\varepsilon > 0$ take an $n$ such that $\tfrac{b - a}{n} < \varepsilon$ and set

$$M(\varepsilon) = \lbrace a + k\tfrac{b - a}{n} \mid k = 0, 1, 2, \dots \rbrace. \quad \square$$

</details>
</div>

### 2.5. A Characteristic of Total Boundedness Reminiscent of Compactness

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 2.5.1</span></p>

If $(X, d)$ is not totally bounded then there is a sequence that contains no Cauchy subsequence.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

If $(X, d)$ is not totally bounded then there is an $\varepsilon_0 > 0$ such that for every finite $M \subseteq X$ there is an $x_M \in X$ such that $d(x_M, M) \ge \varepsilon_0$. Choose $x_1$ arbitrarily and if $x_1, \dots, x_n$ are already chosen set $x_{n+1} = x_{\lbrace x_1, \dots, x_n \rbrace}$. Then any two elements of the resulting sequence have the distance at least $\varepsilon_0$ and hence there is no Cauchy subsequence. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.5.2</span></p>

A metric space $X$ is totally bounded if and only if every sequence in $X$ contains a Cauchy subsequence.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $(x_n)_n$ be a sequence in a totally bounded $(X, d)$. Consider the

$$M(\tfrac{1}{n}) = \lbrace y_{n1}, \dots, y_{nm_n} \rbrace$$

from the definition. If $A = \lbrace x_n \mid n = 1, 2, \dots \rbrace$ is finite then $(x_n)_n$ contains a constant subsequence. Thus, suppose $A$ is not finite. There is an $r_1$ such that $A_1 = A \cap \Omega(y_{1r_1}, 1)$ is infinite; choose $x_{k_1} \in A_1$. If we already have infinite

$$A_1 \supseteq A_2 \supseteq \cdots \supseteq A_s, \quad A_j \subseteq \Omega(y_{jr_j}, \tfrac{1}{j})$$

and $k_1 < \cdots < k_s$ such that $x_{k_j} \in A_j$, choose $r_{s+1}$ such that $A_{s+1} = A_s \cap \Omega(y_{s+1,r_{s+1}}, \tfrac{1}{s+1})$ is infinite and an $x_{k_{s+1}} \in A_{s+1}$ such that $k_{s+1} > k_s$. Then the subsequence $(x_{k_n})_n$ is Cauchy.

The converse is in 2.5.1. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.6</span></p>

A metric space is compact if and only if it is totally bounded and complete.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $X$ be compact. Then it is complete by XIII.7.7 and totally bounded by 2.5.1.

On the other hand let $X$ be totally bounded and let $(x_n)_n$ be a sequence in $X$. Then it contains a Cauchy subsequence and if it is, moreover, complete, this subsequence converges. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.6.1</span></p>

1. We already know the characterisation of the compact subspaces of $\mathbb{E}_n$ as the closed bounded ones (XIII.7.6). Realize that it is a special case of 2.6: a subset of $\mathbb{E}_n$ is complete iff it is closed (see XIII.6.6 and XIII.6.4), and it is totally bounded iff it is bounded (see 2.4).
2. Note that neither completeness nor total boundedness are topological properties, while their conjunction is.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.7</span></p>

Every totally bounded metric space is separable.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Take the sets $M(\varepsilon)$ from the definition again. The set $\bigcup_{n=1}^{\infty} M(\tfrac{1}{n})$ is countable and evidently dense. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2.7.1</span></p>

Every compact space is separable and hence Lindelöf.

</div>

---

## 3. Heine–Borel Theorem

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Accumulation Point)</span></p>

A point is an **accumulation point** of a set $A$ in a space $X$ if every neighbourhood of $x$ contains infinitely many points of $A$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.1</span></p>

A metric space $X$ is compact iff every infinite $M$ in $X$ has an accumulation point.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $X$ be compact and let $A$ be infinite. Choose an arbitrary sequence $x_1, x_2, \dots, x_n, \dots$ in $A$ such that $x_i \ne x_j$ for $i \ne j$. Then every neighbourhood of a limit $x$ of a subsequence $(x_{k_n})_n$ contains infinitely many $x_j$'s and hence $x$ is an accumulation point of $A$.

On the other hand let the second statement hold and let $(x_n)_n$ be a sequence in $X$. Then either $A = \lbrace x_n \mid n = 1, 2, \dots \rbrace$ is finite and then $(x_n)_n$ contains a constant subsequence, or $A$ has an accumulation point $x$. Then we can proceed as follows. Choose $x_{k_1}$ in $A \cap \Omega(x, 1)$ and if $x_{k_1}, \dots, x_{k_n}$ have been already chosen pick $x_{k_{n+1}}$ in $A \cap \Omega(x, \tfrac{1}{n+1})$ so that $k_{n+1} > k_n$ (this disqualifies only finitely many of infinite number of choices); then $\lim_n x_{k_n} = x$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.2</span><span class="math-callout__name">(Heine–Borel Theorem)</span></p>

A metric space is compact if and only if each cover of $X$ contains a finite subcover.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**I.** Let $X$ be compact but let there be a cover that has no finite subcover. By 2.7.1 $X$ is Lindelöf and hence there is a *countable* cover $U_1, U_2, \dots, U_n, \dots$ with no finite subcover. Define $V_1, V_2, \dots, V_n, \dots$ as follows:
- take for $V_1$ the first non-empty $U_k$, and
- if $V_1, V_2, \dots, V_n$ have been already chosen take for $V_{n+1}$ the first $U_k$ such that $U_k \not\subseteq \bigcup_{j=1}^{n} V_j$ (rejecting the $U_j$ that were redundant for covering the space, only without repetition).

Hence
1. $\lbrace V_n \mid n = 1, 2, \dots \rbrace$ is a subcover of $\lbrace U_n \mid n = 1, 2, \dots \rbrace$,
2. the procedure cannot stop, else we had a finite subcover, and
3. we can choose $x_n \in V_n \setminus \bigcup_{k=1}^{n-1} V_k$.

Now all the $x_n$ are distinct (if $k < n$ then $x_n \in V_n \setminus V_k$ while $x_k \in V_k$) and hence we have an infinite set $A = \lbrace x_1, x_2, \dots, x_n, \dots \rbrace$, and this set has to have an accumulation point $x$. Since $\lbrace V_n \mid n = 1, 2, \dots \rbrace$ is a cover, there is an $n$ such that $x \in V_n$. This is a contradiction since $V_n$ contains none of the $x_k$ with $k > n$ and hence $V_n \cap A$ is not infinite.

**II.** Let the statement about covers hold and let there be an infinite $A$ without an accumulation point. That is, no $x \in X$ is an accumulation point of $A$ and hence we have open $U_x \ni x$ such that $U_x \cap A$ is finite. Choose a finite subcover $U_{x_1}, U_{x_2}, \dots, U_{x_n}$ of the cover $\lbrace U_x \mid x \in X \rbrace$. Now we have

$$A = A \cap X = A \cap \bigcup_{k=1}^{n} U_{x_k} = \bigcup_{k=1}^{n} (A \cap U_{x_k})$$

which is a contradiction since the rightmost union is finite. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 3.3</span><span class="math-callout__name">(Finite Intersection Property)</span></p>

Let $\mathcal{A}$ be a system of closed subsets of a compact space. If $\bigcap \lbrace A \mid A \in \mathcal{A} \rbrace = \emptyset$ then there is a finite $\mathcal{A}_0 \subseteq \mathcal{A}$ such that $\bigcap \lbrace A \mid A \in \mathcal{A}_0 \rbrace = \emptyset$. Consequently, if

$$A_1 \supseteq A_2 \supseteq \cdots \supseteq A_n \supseteq \cdots$$

is a decreasing sequence of non-empty closed subsets of $X$ then $\bigcap_{n=1}^{\infty} A_n \ne \emptyset$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By De Morgan formula, $\lbrace X \setminus A \mid A \in \mathcal{A} \rbrace$ is a cover. $\square$

</details>
</div>

---

## 4. Baire Category Theorem

### 4.1. Diameter

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Diameter)</span></p>

Generalizing the diameter from XVI.1.3, we define in a general metric space $(X, d)$ for a subset $A \subseteq X$

$$\operatorname{diam}(A) = \sup \lbrace d(x, y) \mid x, y \in A \rbrace.$$

Note that $\operatorname{diam}(A)$ can be infinite: in fact $\operatorname{diam}(X)$ is finite only if the space is bounded.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observations 4.1.1</span></p>

From the triangle inequality we immediately obtain:
1. $\operatorname{diam}(\Omega(x, \varepsilon)) \le 2\varepsilon$, and
2. $\operatorname{diam}(\overline{A}) = \operatorname{diam}(A)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 4.2</span><span class="math-callout__name">(Cantor's Intersection Lemma)</span></p>

Let $(X, d)$ be a complete metric space. Let

$$A_1 \supseteq A_2 \supseteq \cdots \supseteq A_n \supseteq \cdots$$

be a decreasing sequence of non-empty closed subsets of $X$ with $\lim_n \operatorname{diam}(A_n) = 0$. Then

$$\bigcap_{n=1}^{\infty} A_n \ne \emptyset.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Choose $a_n \in A_n$. Then, by the assumption on diameters, $(a_n)_n$ is a Cauchy sequence and hence, by completeness, it has a limit $a$. Now the subsequence $a_n, a_{n+1}, a_{n+2}, \dots$ is in the *closed* $A_n$ and hence its limit $a$ is in $A_n$. As $n$ was arbitrary, $a \in \bigcap_{n=1}^{\infty} A_n$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notes 4.2.1</span></p>

1. The assumption on diminishing diameter is essential: take e.g. the closed $A_n = \langle n, +\infty)$ in the complete $\mathbb{R}$. It may look paradoxical that an intersection of small sets is non-void but an intersection of large ones is not necessarily so. But the principle is, hopefully, obvious.
2. In $\mathbb{R}$ or, more generally, in $\mathbb{E}_n$, the intersection in 4.2 consists necessarily of a single point (see 3.3). But this has to do with compactness, not with completeness — one can easily give an example with $\operatorname{diam}(A_n) = 1$, but not in $\mathbb{R}$ or $\mathbb{E}_n$.
3. Needless to say, the intersection in 4.2 consists necessarily of a single point.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 4.3</span></p>

If $0 < \varepsilon < \eta$ then $\overline{\Omega(x, \varepsilon)} \subseteq \Omega(x, \eta)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

This is an immediate consequence of the triangle inequality: if $d(y, \Omega(x, \varepsilon)) = 0$ choose a $z \in \Omega(x, \varepsilon)$ with $d(y, z) < \eta - \varepsilon$; then $d(x, y) \le d(x, z) + d(z, y) < \eta$. $\square$

</details>
</div>

### 4.4. Nowhere Dense Sets

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Nowhere Dense)</span></p>

A subset $A$ of a metric space $X$ is said to be **nowhere dense** if $X \setminus \overline{A}$ is dense, that is, if $\overline{X \setminus \overline{A}} = X$. Note that

$$A \text{ is nowhere dense iff } \overline{A} \text{ is nowhere dense.}$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Reformulation 4.4.1</span></p>

$A \subseteq X$ is nowhere dense iff for every non-empty open $U$ the intersection $U \cap (X \setminus \overline{A})$ is non-empty. (Indeed, this amounts to stating that for every $x$ and every $\varepsilon > 0$ the intersection $\Omega(x, \varepsilon) \cap (X \setminus \overline{A})$ is non-empty.)

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.4.2</span></p>

A union of finitely many nowhere dense sets is nowhere dense.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

It suffices to prove the statement for two sets. Let $A, B$ be nowhere dense and let $U$ be non-empty open. We have $U \cap (X \setminus (\overline{A \cup B})) = U \cap (X \setminus \overline{A}) \cap (X \setminus \overline{B})$. Now the open set $V = U \cap (X \setminus \overline{A})$ is non-empty, and hence $V \cap (X \setminus \overline{B})$ is non-empty as well. $\square$

</details>
</div>

### 4.5. Sets of First Category (Meagre Sets)

A countable union of nowhere dense sets can be already very far from being nowhere dense. Take for instance the one-point subsets $\lbrace x \rbrace$ of the space $X$ of all rational numbers: their union is the whole of $X$. But in complete spaces such countable unions can form only very small parts.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Set of First Category / Meagre Set)</span></p>

A subset of a metric space is said to be a **set of first category** (or a **meagre set**) if it is a countable union $\bigcup_{n=1}^{\infty} A_n$ of nowhere dense sets $A_n$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.5.1</span><span class="math-callout__name">(Baire Category Theorem)</span></p>

No complete metric space $X$ is of the first category in itself.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Suppose it is, that is, $X = \bigcup_{n=1}^{\infty} A_n$ with $X \setminus \overline{A_n}$ dense. We can assume all the $A_n$ closed; hence we have $X \setminus A_n$ dense open.

Choose $U_1 = \Omega(x, \varepsilon)$ such that $\Omega(x, 2\varepsilon) \subseteq X \setminus A_1$ and $2\varepsilon < 1$. Thus, by 4.1.1 and 4.3,

$$B_1 = \overline{U}_1 \subseteq X \setminus A_1 \quad \text{and} \quad \operatorname{diam}(B_1) < 1.$$

Let us have for $k \le n$ non-empty open $U_1, \dots, U_n$ with

$$U_{k-1} \supseteq B_k = \overline{U}_k \text{ for } k \le n, \quad B_k \subseteq X \setminus A_k, \quad \text{and } \operatorname{diam}(B_k) < \tfrac{1}{k}. \quad (*)$$

Since $U_n \cap (X \setminus A_{n+1})$ is non-empty open we can choose $U_{n+1} = \Omega(y, \eta)$ for some $y \in U_n \cap (X \setminus A_{n+1})$ and $\eta$ sufficiently small to have $\Omega(y, 2\eta) \subseteq U_n \cap (X \setminus A_{n+1})$ and $2\eta < \tfrac{1}{n+1}$. Then we have, by 4.1.1 and 4.3, the system $(*)$ extended from $n$ to $n + 1$ and we inductively obtain a sequence of non-empty closed sets $B_n$ such that

1. $B_1 \supseteq B_2 \supseteq \cdots \supseteq B_n \supseteq \cdots$,
2. $\operatorname{diam}(B_n) < \tfrac{1}{n}$, and
3. $B_n \subseteq X \setminus A_n$.

By (1),(2) and 4.2, $B = \bigcap_{n=1}^{\infty} B_n \ne \emptyset$, and by (3)

$$B \subseteq \bigcap_{n=1}^{\infty} (X \setminus A_n) = X \setminus \bigcup_{n=1}^{\infty} A_n = X \setminus X = \emptyset,$$

a contradiction. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note 4.5.2</span></p>

Realize how small part of a complete space $X$ a set of first category constitutes. A countable union of such sets is obviously still of first category, hence it not only smaller than $X$, but it is in effect so small that infinitely many disjoint copies cannot cover $X$.

</div>

---

## 5. Completion

### 5.1. Motivation

For various reasons, for applying metric spaces in analysis or geometry it is preferable to have the spaces complete. We have already seen the advantages of $\mathbb{R}$ as compared with $\mathbb{Q}$. Note that the extension of the rationals to reals is very satisfactory: we do not lose anything of the calculating power, in fact everything is in this respect only better, and $\mathbb{Q}$ is dense in $\mathbb{R}$ so that everything to be computed in $\mathbb{R}$ can be well approximated by computing with rationals.

In this section we will show that we can analogously extend every metric space. That is, for every $(X, d)$ we have a space $(\widetilde{X}, \widetilde{d})$ such that

- $(X, d)$ is dense in $(\widetilde{X}, \widetilde{d})$ (in our construction we will have an isometric embedding $\iota : (X, d) \to (\widetilde{X}, \widetilde{d})$ such that $\iota[X]$ is dense in $\widetilde{X}$), and
- $(\widetilde{X}, \widetilde{d})$ is complete.

### 5.2. The Construction

The idea is very natural. In the original space there can be Cauchy sequences without limits; thus, let us add the limits. This will be done by representing the limits by the so far limitless Cauchy sequences; only, we will have to identify the sequences that represent the same limit — see the equivalence $\sim$ below.

Denote by $\mathcal{C}(X, d)$, in short $\mathcal{C}(X)$, the set of all Cauchy sequences in $X$. For $(x_n)_n, (y_n)_n \in \mathcal{C}(X)$ define

$$d'((x_n)_n, (y_n)_n) = \lim_n d(x_n, y_n).$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.2.1</span></p>

The limit in the definition of $d'$ always exists and we have

1. $d'((x_n)_n, (x_n)_n) = 0$,
2. $d'((x_n)_n, (y_n)_n) = d'((y_n)_n, (x_n)_n)$, and
3. $d'((x_n)_n, (z_n)_n) \le d'((x_n)_n, (y_n)_n) + d'((y_n)_n, (z_n)_n)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The first statement is proved by showing that $(d(x_n, y_n))_n$ is Cauchy in $\mathbb{R}$. Indeed, $(x_n)_n$ and $(y_n)_n$ are Cauchy and hence for an $\varepsilon > 0$ we have an $n_0$ such that for $m, n > n_0$, $d(x_n, x_m) < \tfrac{\varepsilon}{2}$ and $d(y_n, y_m) < \tfrac{\varepsilon}{2}$. Then $d(x_n, y_n) \le d(x_n, x_m) + d(x_m, y_m) + d(y_m, y_n) < \varepsilon + d(x_m, y_m)$, hence $d(x_n, y_n) - d(x_m, y_m) < \varepsilon$ and by symmetry also $d(x_m, y_m) - d(x_n, y_n) < \varepsilon$, so $\lvert d(x_n, y_n) - d(x_m, y_m) \rvert < \varepsilon$.

(1) and (2) are trivial and (3) is very easy: choose $k$ such that

$$\lvert d'((x_n)_n, (z_n)_n) - d(x_k, z_k) \rvert < \varepsilon, \quad \lvert d'((x_n)_n, (y_n)_n) - d(x_k, y_k) \rvert < \varepsilon$$

and $\lvert d'((y_n)_n, (z_n)_n) - d(y_k, z_k) \rvert < \varepsilon$. Then we obtain from the triangle inequality of $d$ that $d'((x_n)_n, (z_n)_n) \le d'((x_n)_n, (y_n)_n) + d'((y_n)_n, (z_n)_n) + 3\varepsilon$, and since $\varepsilon$ was arbitrary, (3) follows. $\square$

</details>
</div>

Define an equivalence relation $\sim$ on $\mathcal{C}(X)$ by setting

$$(x_n)_n \sim (y_n)_n \quad \text{iff} \quad d'((x_n)_n, (y_n)_n) = 0$$

(from 5.2.1 it immediately follows that $\sim$ is an equivalence relation), denote

$$\widetilde{X} = \mathcal{C}(X) / {\sim},$$

and for classes $p = [(x_n)_n]$ and $q = [(y_n)_n]$ of this equivalence relation set

$$\widetilde{d}(p, q) = d'((x_n)_n, (y_n)_n).$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.2.3</span></p>

The value of $\widetilde{d}(p, q)$ does not depend on the choice of representatives of $p$ and $q$, and $(\widetilde{X}, \widetilde{d})$ is a metric space.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

If $(x_n)_n \sim (x_n')_n$ and $(y_n)_n \sim (y_n')_n$ we have

$$d'((x_n)_n, (y_n)_n) \le d'((x_n)_n, (x_n')_n) + d'((x_n')_n, (y_n')_n) + d'((y_n')_n, (y_n)_n) = 0 + d'((x_n')_n, (y_n')_n) + 0 = d'((x_n')_n, (y_n')_n)$$

and by symmetry also $d'((x_n')_n, (y_n')_n) \le d'((x_n)_n, (y_n)_n)$.

Now by 5.2.1, $\widetilde{d}$ satisfies the requirements XIII.2.1(2),(3) and the missing $\widetilde{d}(p, q) = 0 \Rightarrow p = q$ immediately follows from the definition of $\sim$: if $d(p, q) = d'((x_n)_n, (y_n)_n) = 0$ then $(x_n)_n \sim (y_n)_n$ and the sequences represent the same element of $\widetilde{X}$. $\square$

</details>
</div>

### 5.3. The Isometric Embedding and Completeness

Set $\widetilde{x} = (x, x, \dots, x, \dots)$ and define a mapping

$$\iota = \iota_{(X,d)} : (X, d) \to (\widetilde{X}, \widetilde{d})$$

by $\iota(x) = [\widetilde{x}]$. We have $d'(\widetilde{x}, \widetilde{y}) = d(x, y)$ and hence $\iota$ is an isometric embedding.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.3</span><span class="math-callout__name">(Completion)</span></p>

The image of the isometric embedding $\iota_{(X,d)}$ is dense in $(\widetilde{X}, \widetilde{d})$, and the space $(\widetilde{X}, \widetilde{d})$ is complete.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**Density.** Take a $p = [(x_n)_n] \in \widetilde{X}$ and an $\varepsilon > 0$. Since $(x_n)_n$ is Cauchy there is an $n_0$ such that for $m, k > n_0$, $d(x_m, x_k) \le \varepsilon$. But then $\widetilde{d}(\iota(x_{n_0}), p) = d'(\widetilde{x_{n_0}}, (x_k)_k) = \lim_k d(x_{n_0}, x_k) < \varepsilon$.

**Completeness.** Now let

$$p_1 = [(x_{1n})_n], \;\; p_2 = [(x_{2n})_n], \;\; \dots, \;\; p_k = [(x_{kn})_n], \;\; \dots \qquad (*)$$

be a Cauchy sequence in $(\widetilde{X}, \widetilde{d})$. For each $p_n$ choose, by the already proved density, an $x_n \in X$ such that $\widetilde{d}(p_n, \iota(x_n)) < \varepsilon$. For $\varepsilon > 0$ choose $n_0 > \tfrac{3}{\varepsilon}$ such that for $m, n \ge n_0$, $\widetilde{d}(p_m, p_n) < \tfrac{\varepsilon}{3}$. Then for $m, n \ge n_0$,

$$d(x_m, x_n) = \widetilde{d}(\iota(x_m), \iota(x_n)) \le \widetilde{d}(\iota(x_m), p_m) + \widetilde{d}(p_m, p_n) + \widetilde{d}(p_n, \iota(x_n)) < \tfrac{\varepsilon}{3} + \tfrac{\varepsilon}{3} + \tfrac{\varepsilon}{3} = \varepsilon$$

and we see that $(x_n)_n$ is Cauchy. We will prove that the sequence $(*)$ converges to $p = [(x_n)_n]$.

We know that $\widetilde{d}(p_n, \iota(x_n)) = \lim_k d(x_{nk}, x_n) < \tfrac{1}{n}$. Choose $n_0 > \tfrac{2}{\varepsilon}$ such that for $k, n \ge n_0$ we have $d(x_k, x_m) < \tfrac{\varepsilon}{2}$. Then

$$d(x_{nk}, x_k) \le d(x_{nk}, x_n) + d(x_n, x_k) < \tfrac{\varepsilon}{2} + \tfrac{\varepsilon}{2} = \varepsilon$$

and hence $\widetilde{d}(p_n, p) = \lim_k d(x_{nk}, x_k) \le \varepsilon$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.4</span></p>

The question naturally arises whether the completion extending the rational line $\mathbb{Q}$ to the real one, $\mathbb{R}$, can be constructed in the vein of the procedure just presented. The answer is a cautious YES; one has to keep in mind that we will have some troubles formulating precisely what we are doing. The construction above already works with metric spaces and the distances already have *real* values. But we can speak of Cauchy sequences, define equivalence $\sim$ of Cauchy sequences (but not by means of limits the existence of which is based on the properties of reals), and obtain the desired. But many readers would view the usually used method of Dedekind cuts as somewhat simpler.

</div>

---

# XVIII. Sequences and Series of Functions

## 1. Pointwise and Uniform Convergence

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Pointwise Convergence)</span></p>

Let $X = (X, d)$ and $Y = (Y, d')$ be metric spaces and let $f_n : X \to Y$ be a sequence of continuous mappings. If for each $x \in X$ there is a $\lim_n f(x) = f(x)$ (in $Y$) we say that the sequence $(f_n)_n$ **converges pointwise** to the mapping $f$ and usually write

$$f_n \to f.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.1.1</span></p>

Pointwise convergence does not preserve nice properties of the functions $f_n$, not even continuity, not to speak of possessing derivatives. Consider $X = Y = \langle 0, 1 \rangle$ and $f_n(x) = x^n$. Then $f(x) = \lim_n f_n(x)$ is $0$ for $x < 1$ while $f(1) = 1$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Uniform Convergence)</span></p>

A sequence $(f_n : (X, d) \to (Y, d'))_n$ converges **uniformly** to $f : X \to Y$ if

$$\forall \varepsilon > 0 \;\; \exists n_0 \;\;\text{such that}\;\; \forall x \in X \;\; (n \ge n_0 \;\Rightarrow\; d'(f_n(x), f(x)) < \varepsilon).$$

We speak of a **uniformly convergent sequence** of mappings and write $f_n \rightrightarrows f$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1.3</span><span class="math-callout__name">(Uniform Limit of Continuous Functions)</span></p>

Let $f_n : X \to Y$ be continuous mappings and let $f_n \rightrightarrows f$. Then $f$ is continuous.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Choose $x \in X$ and $\varepsilon > 0$. Fix an $n$ such that

$$\forall y \in X, \;\; d'(f_n(y), f(y)) < \tfrac{\varepsilon}{3}.$$

Since $f_n$ is continuous there is a $\delta > 0$ such that

$$d(x, z) < \delta \;\Rightarrow\; d'(f_n(x), f_n(z)) < \tfrac{\varepsilon}{3}.$$

Hence for $d(x, z) < \delta$,

$$d'(f(x), f(z)) \le d'(f(x), f_n(x)) + d'(f_n(x), f_n(z)) + d'(f_n(z), f(z)) < \tfrac{\varepsilon}{3} + \tfrac{\varepsilon}{3} + \tfrac{\varepsilon}{3} = \varepsilon. \quad \square$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notes 1.4</span></p>

1. The adjective "uniform" refers, similarly as in "uniform continuity", to the independence of the property in question on the location in the domain space. One might expect that, similarly as in the uniform continuity, we will obtain something for free in case of a compact domain. But it is not so: the sequence in Example 1.1.1 has a very simple compact domain and range and it is not uniformly convergent.
2. Theorem 1.3 holds for uniform continuity as well, that is: *if $f_n : X \to Y$ are uniformly continuous mappings and $f_n \rightrightarrows f$, then $f$ is uniformly continuous.* To prove this it suffices to adapt the proof of 1.3 by not fixing the $x$ at the start.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Locally Uniform Convergence)</span></p>

We say that a sequence $(f_n)_n$ converges to $f$ **locally uniformly** if for every $x \in X$ there exists a neighbourhood $U$ such that $f_n\vert_U \rightrightarrows f\vert_U$ for the restrictions on $U$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 1.5.1</span></p>

Let $f_n : X \to Y$ be continuous mappings and let the sequence $f_n$ converge to $f$ locally uniformly. Then $f$ is continuous.

</div>

---

## 2. More about Uniform Convergence: Derivatives, Riemann Integral

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.1</span></p>

Although uniform convergence preserves continuity, it does not preserve smoothness (existence of derivatives). Consider the functions

$$f_n : \langle -1, 1 \rangle \to \langle 0, 1 \rangle \quad \text{defined by} \quad f_n(x) = \sqrt{(1 - \tfrac{1}{n})x^2 + \tfrac{1}{n}}.$$

These smooth functions uniformly converge to $f(x) = \lvert x \rvert$ which has no derivative at $x = 0$: we have

$$\left\lvert \sqrt{(1 - \tfrac{1}{n})x^2 + \tfrac{1}{n}} - \lvert x \rvert \right\rvert = \frac{\tfrac{1}{n}(1 - x^2)}{\sqrt{(1 - \tfrac{1}{n})x^2 + \tfrac{1}{n}} + \lvert x \rvert} \le \sqrt{\tfrac{1}{n}}.$$

</div>

However, smoothness is preserved if the uniform convergence concerns the derivatives.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.2</span><span class="math-callout__name">(Uniform Convergence of Derivatives)</span></p>

Let $f_n$ be continuous real functions defined on an open interval $J$ and let them have continuous derivatives $f_n'$. Let $f_n \to f$ and $f_n' \rightrightarrows g$ on $J$. Then $f$ has a derivative on $J$ and $f' = g$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We have

$$A(h) = \left\lvert \frac{f(x+h) - f(x)}{h} - g(x) \right\rvert = \left\lvert \frac{f(x+h) - f_n(x+h)}{h} - \frac{f(x) - f_n(x)}{h} + \frac{f_n(x+h) - f_n(x)}{h} - g(x) \right\rvert$$

and since by the Lagrange theorem, $\tfrac{f_n(x+h) - f_n(x)}{h} = f_n'(x + \theta h)$ for some $\theta$ with $0 < \theta < 1$, we further obtain

$$A(h) \le \tfrac{1}{\lvert h \rvert} \lvert f(x+h) - f_n(x+h) \rvert + \tfrac{1}{\lvert h \rvert} \lvert f(x) - f_n(x) \rvert + \lvert f_n'(x + \theta h) - g(x + \theta h) \rvert + \lvert g(x + \theta h) - g(x) \rvert.$$

Since $f_n' \rightrightarrows g$, the function $g$ is continuous by 1.3. Choose $\delta > 0$ such that for $\lvert x - y \rvert < \delta$ we have $\lvert g(x) - g(y) \rvert < \varepsilon$; thus if $\lvert h \rvert < \delta$ the last summand is smaller than $\varepsilon$.

Now fix an $h$ with $\lvert h \rvert < \delta$ and choose an $n$ sufficiently large so that $\lvert f_n'(y) - g(y) \rvert < \varepsilon$, $\lvert f(x+h) - f_n(x+h) \rvert < \varepsilon \lvert h \rvert$, and $\lvert f(x) - f_n(x) \rvert < \varepsilon \lvert h \rvert$ (note that for the first we have to use the uniform convergence — we do not know precisely where $y = x + \theta h$ is; not so in the other two inequalities, where one uses just convergence in two fixed arguments $x$ and $x + h$). Then we obtain

$$A(h) = \left\lvert \frac{f(x+h) - f(x)}{h} - g(x) \right\rvert < 4\varepsilon$$

and the statement follows. $\square$

</details>
</div>

### 2.3. Integral of a Limit of Functions

For Riemann integral we do not generally have $\int_a^b \lim_n f_n = \lim_n \int_a^b f_n$ even if all the $\int_a^b f_n$ exist and all the functions $f_n$ are bounded by the same constant.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.3</span></p>

Order all the rational numbers between 0 and 1 in a sequence $r_1, r_2, \dots, r_n, \dots$ Set

$$f_n(x) = \begin{cases} 1 & \text{if } x = r_k \text{ with } k \le n, \\ 0 & \text{otherwise.} \end{cases}$$

Then obviously $\int_0^1 f_n = 0$ for every $n$. But the limit $f$ of the sequence $f_n$ is the well-known Dirichlet function for which (obviously again) the lower integral is 0 and the upper integral is 1.

</div>

For uniform convergence we have, however

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.3.1</span><span class="math-callout__name">(Uniform Convergence and Integration)</span></p>

Let $f_n \rightrightarrows f$ on $\langle a, b \rangle$ and let the Riemann integrals $\int_a^b f_n$ exist. Then also $\int_a^b f$ exists and we have

$$\int_a^b f = \lim_n \int_a^b f_n.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

For $\varepsilon > 0$ choose an $n_0$ such that for $n \ge n_0$,

$$\lvert f_n(x) - f(x) \rvert < \frac{\varepsilon}{b - a} \qquad (*)$$

for all $x \in \langle a, b \rangle$. Recall the notation from XI.2. For a partition $P : a = t_0 < t_1 < \cdots < t_{n-1} < t_n = b$ consider

$$m_j = \inf \lbrace f(x) \mid t_{j-1} \le x \le t_j \rbrace, \quad M_j = \sup \lbrace f(x) \mid t_{j-1} \le x \le t_j \rbrace$$

$$m_j^n = \inf \lbrace f_n(x) \mid t_{j-1} \le x \le t_j \rbrace, \quad M_j^n = \sup \lbrace f_n(x) \mid t_{j-1} \le x \le t_j \rbrace.$$

By $(*)$ we have for $n, k \ge n_0$

$$\lvert m_j - m_j^n \rvert,\; \lvert M_j - M_j^n \rvert \le \frac{\varepsilon}{b - a} \quad \text{and hence also} \quad \lvert M_j^k - M_j^n \rvert \le \frac{2\varepsilon}{b-a}$$

and we obtain for the lower sums

$$\lvert s(f, P) - s(f_n, P) \rvert = \left\lvert \sum (m_i - m_i^n)(t_i - t_{i-1}) \right\rvert \le \sum \lvert m_i - m_i^n \rvert (t_i - t_{i-1}) \le \varepsilon$$

and similarly for the upper sums $\lvert S(f, P) - S(f_n, P) \rvert \le \varepsilon$ and $\lvert S(f_k, P) - S(f_n, P) \rvert \le 2\varepsilon$.

Now, first take a $P$ such that $\lvert \int f_n - S(f_n, P) \rvert < \varepsilon$ and $\lvert \int f_k - S(f_k, P) \rvert < \varepsilon$; then we infer from the triangle inequality that $\lvert \int f_k - \int f_n \rvert < 4\varepsilon$ and see that $(\int f_n)_n$ is a Cauchy sequence. Hence there exists a limit $L = \lim_n \int f_n$. Choose $n \ge n_0$ sufficiently large to have $\lvert \int f_n - L \rvert < \varepsilon$.

Now if the partition $P$ is chosen so as to have $S(f_n, P) - \varepsilon < \int f_n < s(f_n, P) + \varepsilon$ we obtain

$$L - 3\varepsilon \le \int f_n - 2\varepsilon < s(f_n, P) - \varepsilon \le s(f, P) \le S(f, P) \le S(f_n, P) + \varepsilon \le \int f_n + 2\varepsilon \le L + 3\varepsilon$$

and since $\varepsilon > 0$ was arbitrary we conclude that $L = \underline{\int} f = \overline{\int} f$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note 2.3.2</span></p>

The example in 2.3 where the Riemann integrable functions pointwise converged to the Dirichlet function suggested that the trouble might be rather in the non-integrable limit function than in the value of the integral being different from the limit. This is only partly true. Indeed, if we take the more powerful Lebesgue integral (roughly speaking, based on the idea of sums of *countable* disjoint systems, while our Riemann integral is based on *finite* disjoint systems) the integral of the Dirichlet function is 0 (as the intuition suggests: the part of the interval in which the function is not 0 is infinitely smaller than the one with values 0).

But whatever the strength of the integral might be, the formula $\int_a^b \lim_n f_n = \lim_n \int_a^b f_n$ cannot hold generally.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.3.3</span></p>

Consider the functions $f_n, g_n : \langle -1, 1 \rangle \to \mathbb{R} \cup \lbrace +\infty \rbrace$ defined by

$$f_n(x) = \begin{cases} 0 & \text{for } x \le -\tfrac{1}{n} \text{ and } x \ge \tfrac{1}{n}, \\ n + n^2 x & \text{for } -\tfrac{1}{n} \le x \le 0, \\ n - n^2 x & \text{for } 0 \le x \le \tfrac{1}{n}, \end{cases} \qquad g_n(x) = \begin{cases} 0 & \text{for } x \ne 0, \\ n & \text{for } x = 0. \end{cases}$$

Then for each $n$, $\int_a^b f_n = 1$ and $\int_a^b g_n = 0$ while $\lim_n f_n = \lim_n g_n$.

In actual fact, for Lebesgue integral the formula $\int_a^b \lim_n f_n = \lim_n \int_a^b f_n$ holds for instance if the limit is monotone or if the functions are equally bounded by an integrable function.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 2.4</span></p>

Let $\lim_{n \to \infty} g(x_n) = A$ for each sequence $(x_n)_n$ such that $\lim_n x_n = a$. Then $\lim_{x \to a} g(x) = A$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Suppose $\lim_{x \to a} g(x)$ does not exist or is not equal to $A$. Then there is an $\varepsilon > 0$ such that for every $\delta > 0$ there is an $x(\delta)$, with $0 < \lvert a - x(\delta) \rvert < \delta$ and $\lvert A - g(x(\delta)) \rvert \ge \varepsilon$. Set $x_n = x(\tfrac{1}{n})$. Then $\lim_n x_n = a$ while $\lim_{n \to \infty} g(x_n)$ is not $A$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.4.1</span><span class="math-callout__name">(Continuity of Parameter Integral)</span></p>

Let $f : \langle a, b \rangle \times \langle c, d \rangle \to \mathbb{R}$ be a continuous function. Then

$$\lim_{y \to y_0} \int_a^b f(x, y)\,\mathrm{d}x = \int_a^b f(x, y_0)\,\mathrm{d}x.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Since $\langle a, b \rangle \times \langle c, d \rangle$ is compact, $f$ is uniformly continuous. Thus, for every $\varepsilon > 0$ there is a $\delta > 0$ such that $\max \lbrace \lvert x_1 - x_2 \rvert, \lvert y_1 - y_2 \rvert \rbrace < \delta$ implies $\lvert f(x_1, y_1) - f(x_2, y_2) \rvert < \varepsilon$.

Let $\lim_n y_n = y_0$. Set $g(x) = f(x, y_0)$ and $g_n(x) = f(x, y_n)$. If $\lvert y_n - y_0 \rvert < \delta$ we have $\lvert g_n(x) - g(x) \rvert < \varepsilon$ independently of $x$, hence $g_n \rightrightarrows g$ so that by 2.3.1, $\lim_n \int_a^b g_n(x)\,\mathrm{d}x = \int_a^b g(x)\,\mathrm{d}x$, that is, $\lim_n \int_a^b f(x, y_n)\,\mathrm{d}x = \int_a^b f(x, y_0)\,\mathrm{d}x$, and the statement follows from Lemma 2.4. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.4.2</span><span class="math-callout__name">(Differentiation under the Integral Sign)</span></p>

Let $f : \langle a, b \rangle \times \langle c, d \rangle \to \mathbb{R}$ be continuous and let it have a continuous partial derivative $\tfrac{\partial f(x,y)}{\partial y}$ in $\langle a, b \rangle \times (c, d)$. Then $F(y) = \int_a^b f(x, y)\,\mathrm{d}x$ has a derivative in $(c, d)$ and we have

$$\frac{\mathrm{d}}{\mathrm{d}y} \int_a^b f(x, y)\,\mathrm{d}x = \int_a^b \frac{\partial f(x, y)}{\partial y}\,\mathrm{d}x.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Fix $y \in (c, d)$ and choose an $\alpha > 0$ such that $c < y - \alpha < y + \alpha < d$. Set $F(y) = \int_a^b f(x, y)\,\mathrm{d}x$ and define

$$g(x, t) = \begin{cases} \tfrac{1}{t}(f(x, y+t) - f(x, y)) & \text{for } t \ne 0, \\ \tfrac{\partial f(x,y)}{\partial y} & \text{for } t = 0. \end{cases}$$

This function $g$ is continuous on the compact $\langle a, b \rangle \times \langle -\alpha, +\alpha \rangle$. This is obvious in the points $(x, t)$ with $t \ne 0$, and since by the Lagrange theorem

$$g(x, t) - g(x, 0) = \tfrac{1}{t}(f(x, y+t) - f(x, y)) - \tfrac{\partial f(x, y)}{\partial y} = \tfrac{\partial f(x, y + \theta t)}{\partial y} - \tfrac{\partial f(x, y)}{\partial y},$$

the continuity in $(x, 0)$ follows from the continuity of the partial derivative.

Hence we can apply 2.4.1 to obtain

$$\lim_{t \to 0} \int_a^b g(x, t)\,\mathrm{d}x = \int_a^b \frac{\partial f(x, y)}{\partial y}\,\mathrm{d}x.$$

And since for $t \ne 0$

$$\int_a^b g(x, t)\,\mathrm{d}x = \frac{1}{t}\left(\int_a^b f(x, y+t)\,\mathrm{d}x - \int_a^b f(x, y)\,\mathrm{d}x\right) = \frac{1}{t}(F(y+t) - F(y))$$

the statement follows. $\square$

</details>
</div>

---

## 3. The Space of Continuous Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($C(X)$)</span></p>

Let $X = (X, d)$ be a metric space. Denote by $C(X)$ the set of all bounded continuous real functions endowed with the metric

$$d(f, g) = \sup \lbrace \lvert f(x) - g(x) \rvert \mid x \in X \rbrace.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note 3.1.1</span></p>

There is no harm in allowing infinite distances; in effect, it has advantages. However, we have worked so far with finite distances and we will keep doing so. This is why we assume our functions bounded. But:
- most of what we will do in this section holds without the boundedness, and
- if $X$ is compact we have the functions bounded anyway.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.2</span></p>

A sequence $(f_n)_n$ converges to $f$ in $C(X)$ if and only if $f_n \rightrightarrows f$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We have $\lim_n f_n = f_n$ in $C(X)$ if for every $\varepsilon > 0$ there is an $n_0$ such that $d(f_n, f) = \sup \lbrace \lvert f_n(x) - f(x) \rvert \mid x \in X \rbrace \le \varepsilon$ for $n \ge n_0$. This is to say that for every $\varepsilon > 0$ there is an $n_0$ such that for all $n \ge n_0$ and for all $x \in X$ it holds that $\lvert f_n(x) - f(x) \rvert \le \varepsilon$, which is the definition of uniform convergence. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation 3.3</span></p>

Let $a$ be a real number. Then the function $g : \mathbb{R} \to \mathbb{R}$ defined by $g(x) = \lvert a - x \rvert$ is continuous. (Indeed, we have $\lvert a - y \rvert \le \lvert a - x \rvert + \lvert x - y \rvert$, hence $\lvert a - y \rvert - \lvert a - x \rvert \le \lvert x - y \rvert$ and by symmetry $\lvert \lvert a - y \rvert - \lvert a - x \rvert \rvert \le \lvert x - y \rvert$.)

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.3.1</span><span class="math-callout__name">(Completeness of $C(X)$)</span></p>

$C(X)$ is a complete metric space.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $(f_n)_n$ be a Cauchy sequence in $C(X)$. Thus, for every $\varepsilon > 0$ there is an $n_0$ such that

$$\forall m, n \ge n_0, \quad \forall x \in X \quad \lvert f_m(x) - f_n(x) \rvert < \varepsilon. \qquad (*)$$

Thus in particular each of the sequences $(f_n(x))_n$ is Cauchy in $\mathbb{R}$ and we have a limit $f(x) = \lim_n f_n(x)$.

Fix an $m \ge n_0$. Taking a limit in $(*)$ and using Observation 3.3 we obtain

$$\forall m \ge n_0, \quad \lvert f_m(x) - \lim_n f_n(x) \rvert = \lvert f_m(x) - f(x) \rvert \le \varepsilon,$$

independently on $x$. Thus $f_n \rightrightarrows f$ and hence
- by 1.3, $f$ is continuous; it is also bounded since if we fix an $m \ge n_0$ obviously $\lvert f(x) \rvert \le \lvert f_m(x) \rvert + \varepsilon$ (and $f_m$ is bounded) and hence $f \in C(X)$,
- and by 3.2 $\lim_n f_n = f$ in $C(X)$. $\square$

</details>
</div>

---

## 4. Series of Continuous Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Series of Functions)</span></p>

Series of continuous functions

$$\sum_{n=0}^{\infty} f_n(x) = f_0(x) + f_1(x) + \cdots + f_n(x) + \cdots$$

are treated as limits $\lim_n \sum_{k=0}^n f_k(x)$ of the partial finite sums. As with series of numbers, the really important ones are the **absolutely convergent series of functions**, namely those for which $\sum_{n=0}^{\infty} f_n(x)$ is absolutely convergent for each $x$ in the domain. In particular (recall III.2.4), *if $\sum_{n=0}^{\infty} f_n(x)$ is absolutely convergent then the sum does not depend on the order of the summands*.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Uniform and Locally Uniform Convergence of Series)</span></p>

A series $\sum_{n=0}^{\infty} f_n(x)$ is said to **converge uniformly** (resp. **converge locally uniformly**) if $(\sum_{k=0}^n f_k(x))_n$ is a uniformly convergent (resp. locally uniformly convergent) sequence of functions.

In the first case we will sometimes use the symbol

$$\sum_{n=0}^{\infty} f_n(x) \rightrightarrows f(x) \quad \text{or} \quad f_0(x) + f_1(x) + \cdots + f_n(x) + \cdots \rightrightarrows f(x).$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.3</span></p>

Let $\sum_{n=0}^{\infty} f_n(x)$ be a uniformly convergent series of functions. Then the sum is continuous.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.4</span><span class="math-callout__name">(Differentiation of Series)</span></p>

Let the series $\sum_{n=0}^{\infty} f_n(x)$ converge to $f(x)$, let the functions $f_n(x)$ have derivatives $f_n'(x)$ and let the series $\sum_{n=0}^{\infty} f_n'(x)$ uniformly converge. Then $f(x)$ has a derivative

$$\left(\sum_{n=0}^{\infty} f_n(x)\right)' = \sum_{n=0}^{\infty} f_n'(x).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.5</span><span class="math-callout__name">(Weierstrass M-test)</span></p>

Let $b_n \ge 0$ and let $\sum_{n=0}^{\infty} b_n$ converge. Let $f_n(x)$ be real functions on a domain $D$ such that $\lvert f_n(x) \rvert \le b_n$ for all $x \in D$. Then $\sum_{n=0}^{\infty} f_n(x)$ converges on $D$ absolutely and uniformly.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The absolute convergence is in the definition. Now let $\varepsilon > 0$. The sequence $(\sum_{k=0}^n b_k)_n$ is Cauchy and hence there is an $n_0$ such that for $m, n + 1 \ge n_0$, $\sum_{n+1}^{m} b_k < \varepsilon$. Then we have for $x \in D$

$$\left\lvert \sum_{n+1}^{m} f_k(x) \right\rvert \le \sum_{n+1}^{m} \lvert f_k(x) \rvert \le \sum_{n+1}^{m} b_k < \varepsilon$$

and hence in $C(D)$

$$d\!\left(\sum_{k=0}^{m} f_k,\; \sum_{k=0}^{n} f_k\right) = \sup \left\lbrace \left\lvert \sum_{n+1}^{m} f_k(x) \right\rvert \;\middle|\; x \in D \right\rbrace \le \varepsilon.$$

Thus, the sequence $(\sum_{k=0}^n f_k)_n$ is Cauchy in $C(D)$ and by 3.3.1 (and the definition 3.2) $\sum_{k=0}^{\infty} f_k(x)$ uniformly converges. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 4.5.1</span></p>

Let $f(x) = \sum_{n=0}^{\infty} f_n(x)$ converge and let $f_n(x)$ have derivatives. Let there be a convergent series $\sum_{n=0}^{\infty} b_n$ such that $\lvert f_n'(x) \rvert \le b_n$ for all $n$ and $x$. Then the derivative of $f$ exists and we have

$$\left(\sum_{n=0}^{\infty} f_n(x)\right)' = \sum_{n=0}^{\infty} f_n'(x).$$

</div>

---

# XIX. Power Series

## 1. Limes Superior

We will allow infinite limits of sequences of real numbers, that is,

$$\lim_n a_n = +\infty \quad \text{if} \quad \forall K \;\exists n_0 \;(n \ge n_0 \;\Rightarrow\; a_n \ge K),$$

$$\lim_n a_n = -\infty \quad \text{if} \quad \forall K \;\exists n_0 \;(n \ge n_0 \;\Rightarrow\; a_n \le K),$$

and infinite suprema for $M \subseteq \mathbb{R}$: $\sup M = +\infty$ if $M$ has no finite upper bound. We will set

$$(+\infty) \cdot a = a \cdot (+\infty) = +\infty \text{ for positive } a, \quad \text{and} \quad (+\infty) + a = a + (+\infty) = +\infty \text{ for finite } a.$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Limes Superior)</span></p>

For a sequence $(a_n)_n$ of real numbers define **limes superior** as the number

$$\limsup_n a_n = \lim_n \sup_{k \ge n} a_k = \inf_n \sup_{k \ge n} a_k.$$

The second equality is obvious: the sequence $(\sup_{k \ge n} a_k)_n$ is non-increasing. Limes superior is defined for an arbitrary sequence.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation 1.2.1</span></p>

If $\lim_n a_n$ exists then $\limsup_n a_n = \lim_n a_n$.

(If $\lim_n a_n = -\infty$ then $(\sup_{k \ge n} a_k)_n$ has no lower bound and if $\lim_n a_n = +\infty$ then $\sup_{k \ge n} a_k = +\infty$ for all $n$. Let $a = \lim_n a_n$ be finite and let $\varepsilon > 0$. Then $\lvert a_n - a \rvert < \varepsilon$ implies that $\lvert \sup_{k \ge n} a_k - a \rvert \le \varepsilon$.)

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1.3</span></p>

Suppose $a_n, b_n \ge 0$; set $a = \limsup_n a_n$. Let there exist a finite and positive $b = \lim_n b_n$. Then

$$\limsup_n a_n b_n = ab.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**I.** For an $\varepsilon > 0$ choose an $n_0$ such that

$$n \ge n_0 \;\Rightarrow\; b_n < b + \varepsilon \;\;\text{and}\;\; \sup_{k \ge n} a_k \le a + \varepsilon.$$

Then we have for $n \ge n_0$

$$\sup_{k \ge n} a_k b_k \le (\sup_{k \ge n} a_k)(b + \varepsilon) \le (a + \varepsilon)(b + \varepsilon) = ab + \varepsilon(a + b + \varepsilon)$$

and as $\varepsilon > 0$ was arbitrary, we see that $\limsup_n a_n b_n \le ab$ (this also includes the case of $a = +\infty$ where, of course, the estimate is trivial).

**II.** For $\varepsilon > 0$ sufficiently small to have $b - \varepsilon > 0$ choose an $n_0$ such that

$$n \ge n_0 \;\Rightarrow\; b_n > b - \varepsilon.$$

Since $\sup_{k \ge n} a_k \ge \inf_m \sup_{k \ge m} a_k = a$ for every $n$, there exist $k(n) \ge n$ such that $a_{k(n)} \ge a - \varepsilon$ if $a$ is finite, and $a_{k(n)} \ge n$ if $a = +\infty$. Then for $n \ge n_0$,

$$(a - \varepsilon)(b - \varepsilon) \le a_{k(n)} b_{k(n)} \le \sup_m a_m b_m \quad \text{resp.} \quad n(b - \varepsilon) \le \sup_m a_m b_m \text{ if } a = +\infty$$

so that $ab - \varepsilon(a + b - \varepsilon) \le \sup_m a_m b_m$ resp. $n(b - \varepsilon) \le \sup_m a_m b_m$ if $a = +\infty$, and since $\varepsilon > 0$ was arbitrary and since $n(b - \varepsilon)$ is arbitrarily large, we also have $ab \le \limsup_n a_n b_n$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note 1.4</span></p>

There is a counterpart of the limes superior called **limes inferior** defined for an arbitrary sequence $(a_n)_n$ of real numbers by setting

$$\liminf_n a_n = \lim_n \inf_{k \ge n} a_k = \sup_n \inf_{k \ge n} a_k.$$

Its properties are quite analogous.

</div>

---

## 2. Power Series and the Radius of Convergence

Until Chapter XXI we will not systematically treat complex functions of complex variable, but in this section it will be of advantage to consider the coefficients $a_n$, $c$ and the variable $x$ complex. This is not only because the proof of the theorem on the radius of convergence is literally the same; what is at the moment perhaps more important, it will explain the seemingly paradoxical behaviour of some *real* power series (see 2.4 below).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Power Series)</span></p>

Let $a_n$ and $c$ be complex numbers. A **power series** with coefficients $a_n$ and **center** $c$ is the series

$$\sum_{n=0}^{\infty} a_n (x - c)^n.$$

In this section it will be understood as a function of a complex variable $x$; the domain will be specified shortly.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Radius of Convergence)</span></p>

The **radius of convergence** of a power series $\sum_{n=0}^{\infty} a_n (x - c)^n$ is the number

$$\rho = \rho((a_n)_n) = \frac{1}{\limsup_n \sqrt[n]{\lvert a_n \rvert}}.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.3.1</span><span class="math-callout__name">(Radius of Convergence)</span></p>

Let $\rho = \rho((a_n)_n)$ be the radius of convergence of $\sum_{n=0}^{\infty} a_n (x - c)^n$ and let $r < \rho$. Then the series $\sum_{n=0}^{\infty} a_n (x - c)^n$ converges uniformly and absolutely in the set $\lbrace x \mid \lvert x - c \rvert \le r \rbrace$.

On the other hand, the series does not converge if $\lvert x - c \rvert > \rho$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**I.** For a fixed $r < \rho$ choose a $q$ such that

$$r \cdot \inf_n \sup_{k \ge n} \sqrt[k]{\lvert a_k \rvert} < q < 1.$$

Then there is an $n$ such that for all $k \ge n$,

$$r \cdot \sup_{k \ge n} \sqrt[k]{\lvert a_k \rvert} < q \quad \text{and hence} \quad r \cdot \sqrt[k]{\lvert a_k \rvert} < q.$$

For a sufficiently large $K \ge 1$ we have, moreover, $r^k \cdot \lvert a_k \rvert < Kq^k$ for all $k \le n$ so that

$$\text{if } \lvert x - c \rvert \le r \text{ then } \lvert a_k(x - c)^k \rvert \le Kq^k \text{ for all } k$$

and we see by XVIII.4.5 that $\sum_{n=0}^{\infty} a_n(x - c)^n$ converges uniformly and absolutely in $\lbrace x \mid \lvert x - c \rvert \le r \rbrace$.

**II.** If $\lvert x - c \rvert > \rho$ then $\lvert x - c \rvert \cdot \inf_n \sup_{k \ge n} \sqrt[k]{\lvert a_k \rvert} > 1$ and hence $\lvert x - c \rvert \cdot \sup_{k \ge n} \sqrt[k]{\lvert a_k \rvert} > 1$ for all $n$. Consequently, for each $n$ there is a $k(n) \ge n$ such that $\lvert x - c \rvert \cdot \sqrt[k(n)]{\lvert a_{k(n)} \rvert} > 1$ and hence $\lvert a_{k(n)}(x - c)^{k(n)} \rvert > 1$ so that the summands of the series do not even converge to zero. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2.3.2</span></p>

A power series $\sum_{n=0}^{\infty} a_n(x - c)^n$ converges on the open disc $D = \lbrace x \mid \lvert x - c \rvert < \rho((a_n)_n) \rbrace$ and converges in no $x$ with $\lvert x - c \rvert > \rho$. Consequently, the function $f(x) = \sum_{n=0}^{\infty} a_n(x - c)^n$ is continuous on $D$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notes 2.4</span></p>

1. Theorem 2.3.1 is in introductory texts of real analysis often interpreted as a statement about a real power series and its convergence on the interval $(c - \rho, c + \rho)$. The proofs in the real context and in the complex one (as we have interpreted it) are literally the same (although of course the triangle inequality for the absolute value of a complex number is a much deeper fact than in $\mathbb{R}$).
2. The domain $D$ of convergence of a power series is bounded by the open and closed discs

$$\lbrace x \mid \lvert x - c \rvert < \rho \rbrace \subseteq D \subseteq \lbrace x \mid \lvert x - c \rvert \le \rho \rbrace$$

in the complex plane and cannot expand beyond the closed one. This explains the seemingly paradoxical behaviour of the convergence on the real line. Take for instance the real function

$$f(x) = \frac{1}{1 + x^2}.$$

In the interval $(-1, 1)$ it can be written as the power series $1 - x^2 + x^4 - x^6 + x^8 - \cdots$ which abruptly stops converging after $+1$ (and for $x < -1$). There is no obvious reason if we think just in real terms: $f(x)$ gets just smaller after the bounds. But in the complex plane the discs $\lbrace x \mid \lvert x \rvert < r \rbrace$ as domains of $f(x)$ have to stop expanding after reaching $r = 1$: there are obstacles in the points $i$ and $-i$ although there is none on the real line.
3. Theorem 2.3.1 speaks about the convergence in the points of $\lbrace x \mid \lvert x \rvert < \rho \rbrace$ and the divergence for $\lvert x \rvert > \rho$. For the points of the circle $C = \lbrace x \mid \lvert x \rvert = \rho \rbrace$ there is no general rule.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.5</span></p>

The radius of convergence of the series $\sum_{n=1}^{\infty} na_n(x - c)^{n-1}$ is the same as the radius of convergence of the series $\sum_{n=0}^{\infty} a_n(x - c)^n$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

For $x \ne 0$ the series $\mathcal{S} = \sum_{n=1}^{\infty} na_n(x - c)^{n-1}$ obviously converges iff the series $\mathcal{S}_1 = \sum_{n=1}^{\infty} na_n(x - c)^n = x(\sum_{n=1}^{\infty} na_n(x-c)^{n-1})$ does. By 1.3 we have

$$\limsup_n \sqrt[n]{n \lvert a_n \rvert} = \limsup_n \sqrt[n]{n} \cdot \sqrt[n]{\lvert a_n \rvert} = \lim_n \sqrt[n]{n} \cdot \limsup_n \sqrt[n]{\lvert a_n \rvert} = \limsup_n \sqrt[n]{\lvert a_n \rvert}$$

since $\lim_n \sqrt[n]{n} = \lim_n \mathrm{e}^{\frac{1}{n}\lg n} = \mathrm{e}^0 = 1$. Consequently, the radius of convergence of $\mathcal{S}$, and hence of $\mathcal{S}_1$, is equal to $\rho((a_n)_n)$. $\square$

</details>
</div>

### 2.5.1. Differentiation and Integration of Power Series

By XVIII.4.5.1 we now obtain

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.5.1</span><span class="math-callout__name">(Term-by-term Differentiation and Integration)</span></p>

The series $f(x) = \sum_{n=0}^{\infty} a_n(x - c)^n$ has a derivative

$$f'(x) = \sum_{n=1}^{\infty} na_n(x - c)^{n-1}$$

and also a primitive function

$$\left(\int f\right)(x) = C + \sum_{n=0}^{\infty} \frac{a_n}{n+1}(x - c)^{n+1}$$

in the whole interval $J = (c - \rho, c + \rho)$ where $\rho = \rho((a_n)_n)$.

In other words, one can differentiate and integrate power series by individual summands.

</div>

---

## 3. Taylor Series

Recall VIII.7.3. Let a function $f$ have derivatives $f^{(n)}$ of all orders in an interval $J = (c - \Delta, c + \Delta)$. Then we have for each $n$ and $x \in J$,

$$f(x) = \sum_{k=0}^{n} \frac{f^{(k)}(c)}{k!}(x - c)^k + R_n(f, x)$$

with $R_n(f, x) = \frac{f^{(n+1)}(\xi)}{(n+1)!}(x - c)^{n+1}$ where $\xi$ is a number between $c$ and $x$.

### 3.1. Higher Derivatives of Power Series

Let $f(x) = \sum_{n=0}^{\infty} a_n(x - c)^n$ be a power series with the radius of convergence $\rho$. Then we have by 2.5.1

$$f^{(k)}(x) = \sum_{n=k}^{\infty} n(n-1)\cdots(n-k+1) a_n (x-c)^{n-k} = k! \, a_k + \sum_{n=k+1}^{\infty} n(n-1)\cdots(n-k+1) a_n (x-c)^{n-k}. \qquad (*)$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.3.1</span></p>

1. The coefficients of a power series $f(x) = \sum_{n=0}^{\infty} a_n(x - c)^n$ are uniquely determined by the function $f$.
2. A power series is its own Taylor series.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

1. By $(*)$ we have $a_k = \frac{f^{(k)}(c)}{k!}$.
2. If the series $f(x) = \sum_{n=0}^{\infty} a_n(x - c)^n$ converges we have

$$f(x) = \sum_{n=0}^{k} a_n(x-c)^n + \sum_{n=k+1}^{\infty} a_n(x-c)^n$$

and the remainder $R_k(f, x) = \sum_{n=k+1}^{\infty} a_n(x-c)^n$ converges to zero because of the convergence of the series $\sum_{n=0}^{\infty} a_n(x-c)^n$. Moreover, as we have already observed, we have $a_k = \frac{f^{(k)}(c)}{k!}$. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.1.1</span><span class="math-callout__name">(Taylor Series Representation)</span></p>

Let a function $f$ have derivatives $f^{(n)}$ of all orders in an interval $J = (c - \Delta, c + \Delta)$. Let us have for the remainder $R_n(f, x) = f(x) - \sum_{k=0}^{n} \frac{f^{(k)}(c)}{k!}(x - c)^k$

$$\lim_n R_n(f, x) = 0 \quad \text{for all } x \in J.$$

Then the function $f(x)$ can be expressed in $J$ as the power series

$$\sum_{n=0}^{\infty} \frac{f^{(n)}(c)}{n!}(x - c)^n.$$

This power series is called the **Taylor series** of $f$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We have

$$\lim_n \sum_{k=0}^{n} \frac{f^{(k)}(c)}{k!}(x - c)^k = \lim_n (f(x) - R_n(f, x)) = f(x) - \lim_n R_n(f, x) = f(x). \quad \square$$

</details>
</div>

### 3.2. Examples

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.2</span><span class="math-callout__name">(Standard Taylor Series)</span></p>

1. For an arbitrary large $K$ we have $\lim_n \frac{K^n}{n!} = 0$ (indeed, if we put $k_n = \frac{K^n}{n!}$ then for $n > 2K$, $k_{n+1} < \frac{k_n}{2}$ and hence $k_{n+m} < 2^{-m} k_n$). Consequently for any $x$ the remainder in the Taylor formula VIII.7.3 converges to zero for $e^x$, $\sin x$ and $\cos x$ and we have the Taylor series

$$e^x = 1 + \frac{x}{1!} + \frac{x^2}{2!} + \cdots + \frac{x^n}{n!} + \cdots,$$

$$\sin x = \frac{x}{1!} - \frac{x^3}{3!} + \frac{x^5}{5!} - \cdots \pm \frac{x^{2n+1}}{(2n+1)!} \mp \cdots, \quad \text{and}$$

$$\cos x = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \frac{x^6}{6!} + \cdots \pm \frac{x^{2n+2}}{(2n+2)!} \mp \cdots$$

all of them with the radius of convergence equal to $+\infty$.

2. Just the existence of derivatives of all orders does not suffice: the remainder does not automatically converge to zero. Consider the example from VIII.7.4,

$$f(x) = \begin{cases} e^{-1/x^2} & \text{for } x \ne 0, \\ 0 & \text{for } x = 0 \end{cases}$$

where $f^{(k)}(0) = 0$ for all $k$.

</div>

### 3.4. Determining Taylor Series via Differentiation and Integration

It is not always easy to obtain general formula for the coefficients $\frac{f^{(n)}(c)}{n!}$ of the Taylor series of a function $f$ by taking derivatives. Sometimes, however, we can determine the Taylor series very easily using Proposition 3.3.1 and Theorem 2.5.1.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.4.1</span><span class="math-callout__name">(Logarithm)</span></p>

We have $(\lg(1 - x))' = \frac{1}{x - 1}$. Since

$$\frac{1}{x - 1} = -1 - x - x^2 - x^3 - \cdots$$

we have by 2.5.1 (and 3.3.1)

$$\lg(1 - x) = C - x - \frac{1}{2}x^2 - \frac{1}{3}x^3 - \frac{1}{4}x^4 - \cdots$$

and since $\lg 1 = \lg(1 - 0) = 0$ we have $C = 0$ and obtain the well known formula $\lg(1 - x) = -\sum_{n=1}^{\infty} \frac{x^n}{n}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.4.2</span><span class="math-callout__name">(Arcus Tangens)</span></p>

We have $\arctan(x)' = \frac{1}{1 + x^2}$. Since

$$\frac{1}{1 + x^2} = 1 - x^2 + x^4 - x^6 + x^8 - \cdots$$

we obtain by taking the primitive function

$$\arctan(x) = x - \frac{1}{3}x^3 + \frac{1}{5}x^5 - \frac{1}{7}x^7 + \frac{1}{9}x^9 - \cdots \qquad (*)$$

The additive constant is 0, because $\arctan(0) = 0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.4.3</span><span class="math-callout__name">(A Formula for $\pi$)</span></p>

The formula $(*)$ suggests that

$$\frac{\pi}{4} = \arctan(1) = 1 - \frac{1}{3} + \frac{1}{5} - \frac{1}{7} + \frac{1}{9} - \cdots$$

This equation really holds true, but it is not quite immediate. Why: the radius of convergence of the power series $f(x) = x - \frac{1}{3}x^3 + \frac{1}{5}x^5 - \frac{1}{7}x^7 + \frac{1}{9}x^9 - \cdots$ is 1 so that the argument 1 is on the border of the disc of convergence $\lbrace x \mid \lvert x \rvert < 1 \rbrace$ about which the general propositions do not say anything (recall 2.4). The function $\arctan$ is continuous and for $\lvert x \rvert < 1$ we have $\arctan(x) = f(x)$. Hence we have to prove that

$$\lim_{x \to 1^-} f(x) = 1 - \frac{1}{3} + \frac{1}{5} - \frac{1}{7} + \frac{1}{9} - \cdots$$

Consider $\varepsilon > 0$. The series $1 - \frac{1}{3} + \frac{1}{5} - \frac{1}{7} + \frac{1}{9} - \cdots$ converges (albeit not absolutely) and hence there is an $n$ such that $\lvert P_n \rvert < \varepsilon$ for $P_n = \frac{1}{2n+1} - \frac{1}{2n+3} + \frac{1}{2n+5} - \cdots$ Now choose a $\delta > 0$ such that for $1 - \delta < x < 1$ and for $P_n(x) = \frac{1}{2n+1}x^{2n+1} - \frac{1}{2n+3}x^{2n+3} + \frac{1}{2n+5}x^{2n+5} - \cdots$ we have $\lvert P_n(x) \rvert < \varepsilon$ and

$$\lvert (x - \tfrac{1}{3}x^3 + \tfrac{1}{5}x^5 - \cdots \pm \tfrac{1}{2n-1}x^{2n-1}) - (1 - \tfrac{1}{3} + \tfrac{1}{5} - \cdots \pm \tfrac{1}{2n-1}) \rvert < \varepsilon.$$

Now we can estimate for $1 - \delta < x < 1$ the difference between $f(x)$ and the alternating sequence $1 - \frac{1}{3} + \frac{1}{5} - \frac{1}{7} + \frac{1}{9} - \cdots$:

$$\lvert f(x) - (1 - \tfrac{1}{3} + \tfrac{1}{5} - \tfrac{1}{7} + \tfrac{1}{9} - \cdots) \rvert \le \lvert (x - \tfrac{1}{3}x^3 + \cdots \pm \tfrac{1}{2n-1}x^{2n-1}) - (1 - \tfrac{1}{3} + \cdots \pm \tfrac{1}{2n-1}) \rvert + \lvert P_n(x) \rvert + \lvert P_n \rvert < 3\varepsilon.$$

Note that there is indeed a one-sided limit only: $f(x)$ does not make sense for $x > 1$.

</div>

---

# XX. Fourier Series

## 1. Periodic and Piecewise Smooth Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Piecewise Continuous and Smooth Functions)</span></p>

A real function $f : \langle a, b \rangle \to \mathbb{R}$ is **piecewise continuous** if there are

$$a = a_0 < a_1 < a_2 < \cdots < a_n = b$$

such that

- $f$ is continuous on each open interval $(a_j, a_{j+1})$ and
- there exist finite one-sided limits $\lim_{x \to a_j+} f(x)$, $j = 0, \dots, n-1$ and $\lim_{x \to a_j-} f(x)$, $j = 1, \dots, n$.

It is **piecewise smooth** if, moreover,

- $f$ has continuous derivatives on each open interval $(a_j, a_{j+1})$ and
- there exist finite one-sided limits $\lim_{x \to a_j+} f'(x)$, $j = 0, \dots, n-1$ and $\lim_{x \to a_j-} f'(x)$, $j = 1, \dots, n$.

For $y \in \langle a, b \rangle$ set

$$f(y+) = \lim_{x \to y+} f(x), \quad f(y-) = \lim_{x \to y-} f(x) \quad \text{and} \quad f(y\pm) = \frac{f(y+) + f(y-)}{2}.$$

We will speak of the $a_i$ as of the **exceptional points** of $f$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notes 1.1.1</span></p>

1. A piecewise continuous $f$ can be extended to a continuous function on each $\langle a_j, a_{j+1} \rangle$. Consequently it has a Riemann integral.
2. If $y \notin \lbrace a_0, a_1, \dots, a_n \rbrace$ then $f(y+) = f(y-) = f(y\pm) = f(y)$. If $y = a_i$ this may or may not hold. The division points $a_i$ in which $f(a_i+) = f(a_i-) = f(a_i)$ may be thought of as superfluous in the case of plain piecewise continuity, but not so in the case of piecewise smoothness: we consider also functions without derivatives of some of the points in which they are continuous.
3. One may ask whether the points in which $f(y+) = f(y-) \ne f(y)$ have some special status. Not really: we will be mostly interested in integrals of piecewise continuous functions, and values in isolated points will not play any role.
4. Recall VII.3.2.1. The last condition for piecewise smoothness is the same as requiring that $f$ has one-sided derivatives in the exceptional points.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Periodic Function)</span></p>

A real function $f : \mathbb{R} \to \mathbb{R}$ is said to be **periodic** with **period** $p$ if

$$\forall x \in \mathbb{R}, \quad f(x + p) = f(x).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Convention 1.2.1</span></p>

A periodic function will be called piecewise continuous resp. piecewise smooth if the restriction $f\vert\langle 0, p \rangle$ is piecewise continuous resp. piecewise smooth.

</div>

### 1.3. A Function on a Compact Interval Represented as a Periodic Function

In this chapter it will be of advantage to represent a real function $f : \langle a, b \rangle \to \mathbb{R}$ as the periodic function $\widetilde{f} : \mathbb{R} \to \mathbb{R}$ with period $p = b - a$ defined by

$$\widetilde{f}(x + kp) = f(x) \;\;\text{for}\;\; x \in (a, b) \;\;\text{and any integer } k,$$

$$\widetilde{f}(a + kp) = \tfrac{1}{2}(f(a) + f(b)).$$

If this replacement is obvious, we write simply $f$ instead of $\widetilde{f}$; typically when computing integrals, a possible change of values in $a$ and $b$ does not matter. Conversely, we do not lose any information when studying a periodic function with period $p$ restricted to some $\langle a, a + p \rangle$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1.4</span></p>

Let $f$ be a piecewise continuous periodic function with period $p$. Then

$$\int_0^p f(x)\,\mathrm{d}x = \int_a^{p+a} f(x)\,\mathrm{d}x \quad \text{for any } a \in \mathbb{R}.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Obviously $\int_b^c f = \int_{b+p}^{c+p} f$ and hence the equality holds for $a = kp$ with $k$ an integer. Now let $a$ be general. Choose an integer $k$ such that $a \le kp \le a + p$. Then

$$\int_a^{p+a} f = \int_a^{kp} f + \int_{kp}^{p+a} f = \int_{p+a}^{(k+1)p} f + \int_{kp}^{p+a} f = \int_{kp}^{(k+1)p} f = \int_0^p f. \quad \square$$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 1.4.1</span></p>

For an arbitrary real $C$ we have

$$\int_0^p f(x + C)\,\mathrm{d}x = \int_0^p f(x)\,\mathrm{d}x.$$

</div>

---

## 2. A Sort of Scalar Product

To be able to work with $\sin kx$ and $\cos kx$ without adjustment we will confine ourselves in the following, until 4.4.1, to periodic functions with the period $2\pi$.

The set of all piecewise smooth functions on $\langle -\pi, \pi \rangle$ constitutes a vector space, denoted $\operatorname{PSF}(\langle -\pi, \pi \rangle)$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Scalar Product on $\operatorname{PSF}$)</span></p>

For $f, g \in \operatorname{PSF}(\langle -\pi, \pi \rangle)$ define

$$[f, g] = \int_{-\pi}^{\pi} f(x) g(x)\,\mathrm{d}x.$$

This function $[-,-] : \operatorname{PSF}(\langle -\pi, \pi \rangle) \times \operatorname{PSF}(\langle -\pi, \pi \rangle) \to \mathbb{R}$ behaves almost like a scalar product.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.2.1</span></p>

We have

1. $[f, f] \ge 0$ and $[f, f] = 0$ iff $f(x) = 0$ in all the non-exceptional $x$,
2. $[f + g, h] = [f, h] + [g, h]$, and
3. $[\alpha f, g] = \alpha [f, g]$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Trivial; the only point that perhaps needs an explanation is the second part of (1). If $f(y) = a \ne 0$ in a non-exceptional point then for some $\delta > 0$, $f(x) > \frac{a}{2}$ for $y - \delta < x < y - \delta$ and we have

$$[f, f] = \int_{-\pi}^{\pi} f^2(y)\,\mathrm{d}x \ge \int_{y-\delta}^{y+\delta} f^2(x)\,\mathrm{d}x \ge \delta \frac{a^2}{2}. \quad \square$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note 2.2.2</span></p>

The only flaw is in $[f, f]$ not quite implying $f \equiv 0$. But this concerns only finitely many arguments and for our purposes it is inessential.

</div>

### 2.3. Trigonometric Product Formulas

From the standard formulas $\sin(\alpha + \beta) = \sin \alpha \cos \beta + \sin \beta \cos \alpha$ and $\cos(\alpha + \beta) = \cos \alpha \cos \beta - \sin \alpha \sin \beta$ one immediately obtains

$$\sin \alpha \cos \beta = \tfrac{1}{2}(\sin(\alpha + \beta) - \sin(\alpha - \beta)),$$

$$\sin \alpha \sin \beta = \tfrac{1}{2}(\cos(\alpha - \beta) - \cos(\alpha + \beta)),$$

$$\cos \alpha \cos \beta = \tfrac{1}{2}(\cos(\alpha + \beta) + \cos(\alpha - \beta)).$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.4</span><span class="math-callout__name">(Orthonormality of Trigonometric System)</span></p>

For any two $m, n \in \mathbb{N}$ we have $[\sin mx, \cos nx] = 0$. If $m \ne n$ then $[\sin mx, \sin nx] = 0$ and $[\cos mx, \cos nx] = 0$. Further, $[\cos 0x, \cos 0x] = [1, 1] = 2\pi$ and $[\cos nx, \cos nx] = [\sin nx, \sin nx] = \pi$ for all $n > 0$.

Thus, the system of functions

$$\frac{1}{2\pi},\; \frac{1}{\pi}\cos x,\; \frac{1}{\pi}\cos 2x,\; \frac{1}{\pi}\cos 3x,\; \dots,\; \frac{1}{\pi}\sin x,\; \frac{1}{\pi}\sin 2x,\; \frac{1}{\pi}\sin 3x,\; \dots$$

is **orthonormal** in $(\operatorname{PSF}(\langle -\pi, \pi \rangle), [-,-])$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By 2.3 we have $\sin mx \cos nx = \frac{1}{2}(\sin(m+n)x - \sin(m-n)x)$, $\sin mx \sin nx = \frac{1}{2}(\cos(m-n)x - \cos(m+n)x)$ and $\cos mx \cos nx = \frac{1}{2}(\cos(m+n)x + \cos(m-n)x)$. Primitive function of $\sin kx$ resp. $\cos kx$ is $-\frac{1}{k}\cos kx$ resp. $\frac{1}{k}\sin kx$ and we obtain the values easily from XI.4.3.1. $\square$

</details>
</div>

---

## 3. Two Useful Lemmas

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 3.1</span></p>

Let $g$ be a piecewise continuous function on $\langle a, b \rangle$. Then

$$\lim_{y \to +\infty} \int_a^b g(x) \sin(yx)\,\mathrm{d}x = 0.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

If $a_0, a_1, \dots, a_n$ are the exceptional points of $g$ we have $\int_a^b g = \sum_{i=0}^{n-1} \int_{a_i}^{a_{i+1}} g$ and hence it suffices to prove the statement for continuous (and hence uniformly continuous) $g$.

Since the primitive function of $\sin(yx)$ is $-\frac{1}{y}\cos(yx)$ we have for any bounds $u, v$,

$$\left\lvert \int_u^v \sin(yx)\,\mathrm{d}x \right\rvert = \left\lvert \left[-\tfrac{1}{y}\cos(yx)\right]_u^v \right\rvert \le \frac{2}{y}.$$

Choose an $\varepsilon > 0$. The function $g$ is uniformly continuous and hence there is a $\delta > 0$ such that for $\lvert x - z \rvert < \delta$, $\lvert g(x) - g(z) \rvert < \varepsilon$. Choose a partition $a = t_1 < t_2 < \cdots < t_n = b$ of $\langle a, b \rangle$ with mesh $< \delta$, that is such that $t_{i+1} - t_i < \delta$ for all $i$.

Now let $y > \frac{4}{\varepsilon} \sum_{i=1}^{n} \lvert g(t_i) \rvert$. Then we have

$$\left\lvert \int_a^b g(x) \sin(yx)\,\mathrm{d}x \right\rvert = \left\lvert \sum_{i=1}^{n} \left(\int_{t_{i-1}}^{t_i} (g(x) - g(t_i))\sin(yx)\,\mathrm{d}x + g(t_i)\int_{t_{i-1}}^{t_i} \sin(yx)\,\mathrm{d}x \right) \right\rvert$$

$$\le \sum_{i=1}^{n} \int_{t_{i-1}}^{t_i} \frac{\varepsilon}{2(b-a)}\,\mathrm{d}x + \sum_{i=1}^{n} \lvert g(t_i) \rvert \cdot \left\lvert \int_{t_{i-1}}^{t_i} \sin(yx)\,\mathrm{d}x \right\rvert \le \frac{\varepsilon}{2} + \sum \lvert g(t_i) \rvert \frac{2}{y} \le \varepsilon. \quad \square$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note 3.1.1</span></p>

Lemma 3.1 is in fact a very intuitive statement. Suppose we compute $\int_a^b C \sin(yx)\,\mathrm{d}x$ with a constant $C$. Then if $y$ is large we have approximately as much of the function under and over the $x$-axis. Moreover, if $y$ is much larger still, this happens already on short subintervals of $\langle a, b \rangle$ where $g$ behaves "almost like constant".

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 3.2</span><span class="math-callout__name">(Dirichlet Kernel Sum)</span></p>

Let $\sin \frac{\alpha}{2} \ne 0$. Then

$$\frac{1}{2} + \sum_{k=1}^{n} \cos k\alpha = \frac{\sin(2n+1)\frac{\alpha}{2}}{2\sin\frac{\alpha}{2}}.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By the first formula in 2.3 we have

$$2\sin\frac{\alpha}{2}\cos k\alpha = \sin\left(k\alpha + \frac{\alpha}{2}\right) - \sin\left((k-1)\alpha + \frac{\alpha}{2}\right).$$

Thus,

$$2\sin\frac{\alpha}{2}\left(\frac{1}{2} + \sum_{k=1}^{n}\cos k\alpha\right) = \sin\frac{\alpha}{2} + \sum_{k=1}^{n} 2\sin\frac{\alpha}{2}\cos k\alpha = \sin(2n+1)\frac{\alpha}{2}. \quad \square$$

</details>
</div>

---

## 4. Fourier Series

### 4.1. Motivation

Recall from linear algebra representing a general vector as a linear combination of an orthonormal basis. Let $\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_n$ be an orthonormal basis, that is, a basis such that $\mathbf{u}_i \mathbf{u}_j = \delta_{ij}$, of a vector space $V$ endowed with a scalar product $\mathbf{uv}$. Then a general vector $\mathbf{a}$ is expressed as

$$\mathbf{a} = \sum_{i=1}^{n} a_i \mathbf{u}_i \quad \text{where} \quad a_i = \mathbf{a}\mathbf{u}_i.$$

We will see that something similar happens with the orthonormal system from 2.4.

### 4.2. Fourier Coefficients

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fourier Coefficients)</span></p>

Let $f$ be a piecewise smooth periodic function with period $2\pi$. Set

$$a_k = [f, \tfrac{1}{\pi}\cos kx] = \frac{1}{\pi}\int_{-\pi}^{\pi} f(t)\cos kt\,\mathrm{d}t \quad \text{for } k \ge 0, \quad \text{and}$$

$$b_k = [f, \tfrac{1}{\pi}\sin kx] = \frac{1}{\pi}\int_{-\pi}^{\pi} f(t)\sin kt\,\mathrm{d}t \quad \text{for } k \ge 1.$$

We will aim at a proof that $f$ is almost equal to

$$\frac{a_0}{2} + \sum_{k=1}^{\infty}(a_k \cos kx + b_k \sin kx).$$

Thus, the orthonormal system from 2.3 behaves similarly like an orthonormal basis (as recalled in 4.1). There is, of course, the difference that we need **infinite sums** ("infinite linear combinations") to represent the $f \in \operatorname{PSF}(\langle -\pi, \pi \rangle)$ (which is essential) and that the $f$ will be represented up to finitely many values (which is inessential).

</div>

### 4.3. Partial Sums and the Dirichlet Integral

Set

$$s_n(x) = \frac{a_0}{2} + \sum_{k=1}^{n}(a_k \cos kx + b_k \sin kx).$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 4.3.1</span><span class="math-callout__name">(Dirichlet Integral Representation)</span></p>

For every $n$,

$$s_n(x) = \frac{1}{\pi}\int_0^{\pi} (f(x+t) + f(x-t)) \cdot \frac{\sin(n + \frac{1}{2})t}{2\sin\frac{1}{2}t}\,\mathrm{d}t.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Using the definitions of $a_n$ and $b_n$ and the standard formula for $\cos k(x - t) = \cos(kx - kt)$, and then using the equality from 3.2 we obtain

$$s_n(x) = \frac{1}{\pi}\int_{-\pi}^{\pi}\left(\frac{1}{2} + \sum_{k=1}^{n}\cos k(x-t)\right)f(t)\,\mathrm{d}t = \frac{1}{\pi}\int_{-\pi}^{\pi} f(t)\frac{\sin(n+\frac{1}{2})(x-t)}{2\sin\frac{x-t}{2}}\,\mathrm{d}t.$$

Now substitute $t = x + z$. Then $\mathrm{d}t = \mathrm{d}z$ and $z = t - x$, and since $\sin(-u) = -\sin u$ we proceed (using also 1.4)

$$\cdots = \frac{1}{\pi}\int_{-\pi}^{\pi}\left(f(x+z)\frac{\sin(n+\frac{1}{2})z}{2\sin\frac{1}{2}z}\right)\mathrm{d}z = \frac{1}{\pi}\left(\int_0^{\pi}\cdots + \int_{-\pi}^{0}\cdots\right).$$

Substituting $y = -z$ in the second summand we obtain

$$\cdots = \frac{1}{\pi}\int_0^{\pi}(f(x+t) + f(x-t))\frac{\sin(n+\frac{1}{2})t}{2\sin\frac{1}{2}t}\,\mathrm{d}t. \quad \square$$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 4.3.2</span></p>

For every $n$,

$$\frac{1}{\pi}\int_0^{\pi}\frac{\sin(n + \frac{1}{2})t}{\sin\frac{1}{2}t}\,\mathrm{d}t = 1.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Consider the constant function $f = (x \mapsto 1)$. Then $a_0 = 2$ and $a_k = b_k = 0$ for all $k \ge 1$. $\square$

</details>
</div>

### 4.4. The Main Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.4</span><span class="math-callout__name">(Fourier's Theorem)</span></p>

Let $f$ be a piecewise smooth periodic function with period $2\pi$. Then (as $f(x\pm) = \frac{1}{2}(f(x+) + f(x-))$) $\sum_{k=1}^{\infty}(a_k \cos kx + b_k \sin kx)$ converges in every $x \in \mathbb{R}$ and we have (recall 1.1)

$$f(x\pm) = \frac{a_0}{2} + \sum_{k=1}^{\infty}(a_k \cos kx + b_k \sin kx).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By 4.3.1 and 4.3.2 we obtain

$$s_n(x) = \frac{1}{\pi}\int_0^{\pi}(2f(x\pm) + f(x+t) - f(x+) + f(x-t) - f(x-))\frac{\sin(n+\frac{1}{2})t}{\sin\frac{1}{2}t}\,\mathrm{d}t$$

$$= f(x\pm) \cdot \frac{1}{\pi}\int_0^{\pi}\frac{\sin(n+\frac{1}{2})t}{\sin\frac{1}{2}t}\,\mathrm{d}t + \frac{1}{\pi}\int_0^{\pi}\left(\frac{f(x+t) - f(x+)}{t} + \frac{f(x-t) - f(x-)}{t}\right)\frac{\frac{1}{2}t}{\sin\frac{1}{2}t}\sin\left(n + \frac{1}{2}\right)t\,\mathrm{d}t.$$

Set

$$g(t) = \left(\frac{f(x+t) - f(x+)}{t} + \frac{f(x-t) - f(x-)}{t}\right)\frac{\frac{1}{2}t}{\sin\frac{1}{2}t}.$$

This function $g$ is piecewise continuous on $\langle 0, \pi \rangle$: this is obvious for $t > 0$ and in $t = 0$ we have a finite limit because of the left and right derivatives of $f$ in $x$ and the standard $\lim_{t \to 0}\frac{\frac{1}{2}t}{\sin\frac{1}{2}t} = 1$. Thus, we can apply Lemma 3.1 (and Corollary 4.3.2) to obtain

$$\lim_{n \to \infty} s_n(x) = f(x\pm). \quad \square$$

</details>
</div>

### 4.4.1. General Period

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.4.1</span><span class="math-callout__name">(Fourier Series for General Period)</span></p>

Theorem 4.4 can be easily transformed for piecewise smooth periodic function with a general period $p$. For such $f$ we obtain that

$$f(x\pm) = \frac{a_0}{2} + \sum_{k=1}^{\infty}\left(a_k \cos\frac{2\pi}{p}kx + b_k \sin\frac{2\pi}{p}kx\right)$$

where

$$a_k = \frac{2}{p}\int_0^p f(t)\cos\frac{2\pi}{p}kt\,\mathrm{d}t \quad \text{for } k \ge 0, \quad \text{and} \quad b_k = \frac{2}{p}\int_0^p f(t)\sin\frac{2\pi}{p}kt\,\mathrm{d}t \quad \text{for } k \ge 1.$$

Using the representation from 1.3 this can be applied for piecewise smooth functions on a compact interval $\langle a, b \rangle$, setting $p = b - a$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fourier Series)</span></p>

The series $\frac{a_0}{2} + \sum_{k=1}^{\infty}(a_k \cos kx + b_k \sin kx)$ resp. $\frac{a_0}{2} + \sum_{k=1}^{\infty}(a_k \cos\frac{2\pi}{p}kx + b_k \sin\frac{2\pi}{p}kx)$ is called the **Fourier series** of $f$. Note that the sum is equal to $f(x)$ in all the non-exceptional points.

</div>

---

## 5. Notes

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note 5.1</span></p>

The sums $s_n(x)$ are continuous while the resulting $f$ is not necessarily so. Thus, the convergence of the Fourier series in 4.4 is often not uniform (recall XIX.1.3).

If the sums $\sum \lvert a_n \rvert$ and $\sum \lvert b_n \rvert$ converge, then, of course, the Fourier series converges uniformly and absolutely, and if $\sum n\lvert a_n \rvert$ and $\sum n\lvert b_n \rvert$ converge then we can take derivative by the individual summands.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.2</span><span class="math-callout__name">(Differentiating by Summands May Fail)</span></p>

Differentiating by individual summands may be false even if the resulting sum has a derivative. Consider $f(x) = x$ on $(-\pi, \pi)$ extended to a periodic function with the period $2\pi$. Then we obtain

$$f(x\pm) = 2\left(\sin x - \frac{1}{2}\sin 2x + \frac{1}{3}\sin 3x - \frac{1}{4}\sin 4x + \cdots\right).$$

$f(x)$ has a derivative 1 in all the $x \ne (2k+1)\pi$. The formal differentiating by summands would yield

$$g(x) = 2(\cos x - \cos 2x + \cos 3x - \cos 4x + \cdots)$$

and if we write $g_n(x)$ for the partial sum up to the $n$-th summand we obtain $g_n(0) = 2(1 - 1 + 1 - \cdots + (-1)^{n+1})$, hence $g_n(0) = 0$ for $n$ even and $g_n(0) = 2$ for $n$ odd.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note 5.3</span></p>

Note that for $f$ with $f(-x) = f(x)$ all the $b_n$ are zero, and if $f(-x) = -f(x)$ then all the $a_n$ are zero.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note 5.4</span><span class="math-callout__name">(Acoustics Interpretation)</span></p>

Fourier series have an interesting interpretation in acoustics. A tone is described by a periodic function $f$. The pitch is determined by the period $p$ (more precisely, it is given by the *frequency* $\frac{1}{p}$). The function $f$ is seldom close to be sinusoidal. The concrete shape of $f$ determines the *quality* (timbre) making for the character of the sound of that or other musical instrument. In the Fourier interpretation, we see that with the first summand, a (sinusoidal) tone of the basic frequency defining the pitch, we have simultaneously sounding tones of double, triple, etc. frequency. Thus, e.g. when playing flute one gets from the first to the second octave by "blowing away the first basic tone" which results in a tone with twice the basic frequency.

</div>

---

# XXI. Curves and Line Integrals

## 1. Curves

In the applications in the following chapter we will need planar curves only. But for the material of the first two sections a restriction of dimension would not make anything simpler.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Parametrized Curve)</span></p>

A **parametrized curve** in $\mathbb{E}^n$ is a continuous mapping

$$\boldsymbol{\phi} = (\phi_1, \dots, \phi_n) : \langle a, b \rangle \to \mathbb{E}_n$$

(where the compact interval $\langle a, b \rangle$ will always be assumed non-trivial, i.e. with $a < b$).

</div>

### 1.2. Two Equivalences

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Weak Equivalence and Equivalence of Curves)</span></p>

Parametrized curves $\boldsymbol{\phi} = (\phi_1, \dots, \phi_n) : \langle a, b \rangle \to \mathbb{E}_n$ and $\boldsymbol{\psi} = (\psi_1, \dots, \psi_n) : \langle c, d \rangle \to \mathbb{E}_n$ are said to be **weakly equivalent** if there is a homeomorphism $\alpha : \langle a, b \rangle \to \langle c, d \rangle$ such that $\boldsymbol{\psi} \circ \alpha = \boldsymbol{\phi}$. We write $\boldsymbol{\phi} \sim \boldsymbol{\psi}$.

(This relation is obviously reflexive, symmetric and transitive.)

Curves $\boldsymbol{\phi}$ and $\boldsymbol{\psi}$ are said to be **equivalent** if there is an *increasing* homeomorphism $\alpha : \langle a, b \rangle \to \langle c, d \rangle$ such that $\boldsymbol{\psi} \circ \alpha = \boldsymbol{\phi}$. We write $\boldsymbol{\phi} \approx \boldsymbol{\psi}$.

</div>

We will work in particular with

- the curves represented by one-to-one $\boldsymbol{\phi}$, called **simple arcs**, and
- the curves represented by $\boldsymbol{\phi}$ one-to-one with the exception of $\boldsymbol{\phi}(a) = \boldsymbol{\phi}(b)$, called **simple closed curves**.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1.2.2</span></p>

The $\sim$-equivalence class of a simple arc or a simple closed curve is a disjoint union of precisely two $\approx$-equivalence classes.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Since $\boldsymbol{\phi} \approx \boldsymbol{\psi}$ implies $\boldsymbol{\phi} \sim \boldsymbol{\psi}$, a $\sim$-class is a (disjoint) union of $\approx$-classes. The homeomorphism $\alpha$ in $\boldsymbol{\psi} \circ \alpha = \boldsymbol{\phi}$ is (because of the assumption on $\boldsymbol{\phi}$) uniquely determined on $(a,b)$ and hence on the whole compact interval by IV.5.1 — there are sequences in $(a,b)$ converging to $a$ resp. $b$ — and hence for instance $\boldsymbol{\phi}$ and $\boldsymbol{\phi} \circ \iota$, where $\iota(t) = -t + b + a$, are $\sim$-equivalent but not $\approx$-equivalent. Now let $\boldsymbol{\phi} \sim \boldsymbol{\psi}$, with $\alpha$ such that $\boldsymbol{\psi} \circ \alpha = \boldsymbol{\phi}$. Then $\alpha$ by IV.3.4 either increases or decreases. In the first case, $\boldsymbol{\psi} \approx \boldsymbol{\phi}$, in the second one, $\boldsymbol{\psi} \circ \alpha \circ \iota = \boldsymbol{\phi} \circ \iota$ and $\alpha \circ \iota$ increases so that $\boldsymbol{\psi} \approx \boldsymbol{\phi} \circ \iota$. $\square$

</details>
</div>

### 1.3. Curves and Oriented Curves

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Curve, Oriented Curve, Parametrization)</span></p>

The $\sim$-equivalence class $L = [\boldsymbol{\phi}]_\sim$ is called a **curve**. The $\approx$-equivalence classes associated with this curve represent its orientations; we speak of **oriented curves** $L = [\boldsymbol{\phi}]_\approx$.

By 1.2.2, a simple arc, or a simple closed curve has two orientations.

A parametrized curve $\boldsymbol{\phi}$ such that $L = [\boldsymbol{\phi}]_\sim$ resp. $L = [\boldsymbol{\phi}]_\approx$ is called a **parametrization** of $L$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notes 1.3.1</span></p>

1. One may think of a parametrized curve as of a travel on a path with $\boldsymbol{\phi}(t)$ indicating where we are at the instant $t$. The $\sim$-equivalence gets rid of this extra information (now we have just the railroad and not an information of a concrete train moving on it). The orientation captures the direction of the path.
2. The reader may think of a simpler description of a curve as of the image $\boldsymbol{\phi}[\langle a, b \rangle]$, the "geometric shape" of $\boldsymbol{\phi}$. In effect, if $\boldsymbol{\phi}, \boldsymbol{\psi}$ parametrize a simple arc or a simple closed curve, one can easily prove that $\boldsymbol{\phi}[\langle a, b \rangle] = \boldsymbol{\psi}[\langle c, d \rangle]$ if and only if $\boldsymbol{\phi} \sim \boldsymbol{\psi}$. But using the equivalence classes has a lot of advantages (already orienting a curve is simpler).
3. Proposition 1.2.2 holds for simple arcs and simple closed curves only. Draw a picture with $\boldsymbol{\phi}(x) = \boldsymbol{\phi}(y)$ for some $x \ne a, b$ to see that there are more than two possible orientations.
4. The word "closed" in the expression "simple closed curve" has nothing to do with the closedness of a subset of a metric space. Of course every $\boldsymbol{\phi}[\langle a, b \rangle]$ is a compact and hence a closed subset.

</div>

### 1.4. Composing Oriented Curves

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Composition of Oriented Curves)</span></p>

Let $K, L$ be oriented curves represented by parametric ones $\boldsymbol{\phi} : \langle a, b \rangle \to \mathbb{E}_n$, $\boldsymbol{\psi} : \langle b, c \rangle \to \mathbb{E}_n$ (if the latter has not originally started in $b$ transform it as indicated in 1.3.1.2) such that $\boldsymbol{\phi}(b) = \boldsymbol{\psi}(b)$. Set

$$(\boldsymbol{\phi} * \boldsymbol{\psi})(t) = \begin{cases} \boldsymbol{\phi}(t) & \text{for } t \in \langle a, b \rangle \text{ and} \\ \boldsymbol{\psi}(t) & \text{for } t \in \langle b, c \rangle. \end{cases}$$

Obviously $\boldsymbol{\phi} * \boldsymbol{\psi}$ is a continuous mapping $\langle a, c \rangle \to \mathbb{E}_n$ and if $\boldsymbol{\phi} \approx \boldsymbol{\phi}_1 : \langle a_1, b_1 \rangle \to \mathbb{E}_n$ and $\boldsymbol{\psi} \approx \boldsymbol{\psi}_1 : \langle b_1, c_1 \rangle \to \mathbb{E}_n$ then $\boldsymbol{\phi} * \boldsymbol{\psi} \approx \boldsymbol{\phi}_1 * \boldsymbol{\psi}_1$ (note that it is essential that $K, L$ are *oriented* curves, not just curves). Thus, the oriented curve (determined by) $\boldsymbol{\phi} * \boldsymbol{\psi}$ depends on $K, L$ only; it will be denoted by

$$K + L.$$

(Note that the operation $K + L$ is associative.)

</div>

### 1.5. The Opposite Orientation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Opposite Orientation)</span></p>

For an oriented curve $L$ represented by $\boldsymbol{\phi} : \langle a, b \rangle \to \mathbb{E}_n$ define the **oriented curve with opposite orientation**

$$-L$$

as the $\approx$-class of $\boldsymbol{\phi} \circ \iota : \langle a, b \rangle \to \mathbb{E}_n$ with $\iota(t) = -t + b + a$ (recall the proof of 1.2.2). Obviously $-L$ is determined by $L$.

</div>

### 1.6. Piecewise Smooth Curves

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Piecewise Smooth Curve)</span></p>

Recall XX.1.1. A parametrized curve (oriented curve, or curve) $\boldsymbol{\phi} = (\phi_1, \dots, \phi_n) : \langle a, b \rangle \to \mathbb{E}_n$ is said to be **piecewise smooth** if each of the $\phi_j$ is piecewise smooth such that, moreover, the system of the exceptional points $a = a_0 < a_1 < a_2 < \cdots < a_n = b$ can be chosen so that

- for each of the open intervals $J = (a_i, a_{i+1})$, there is a $j$ such that $\phi_j'(t)$ is either positive or negative on the whole of $J$.

However, we will relax the definition of piecewise smoothness by allowing the one-sided limits $\lim_{t \to a_j+} \phi_j'(t)$ and $\lim_{t \to a_j-} \phi_j'(t)$ (in fact, the one-sided derivatives in the exceptional points — recall VII.3.2) infinite.

We will write $\boldsymbol{\phi}'$ for $(\phi_1', \dots, \phi_n')$ (thus in finitely many points $t \in \langle a, b \rangle$, the value $\boldsymbol{\phi}'(t)$ may be undefined; but the derivative will appear only under an integral so that it does not matter).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation 1.6.1</span></p>

Let curves $\boldsymbol{\phi} = (\phi_1, \dots, \phi_n) : \langle a, b \rangle \to \mathbb{E}_n$ and $\boldsymbol{\psi} = (\psi_1, \dots, \psi_n) : \langle c, d \rangle \to \mathbb{E}_n$ be piecewise smooth and let $\alpha$ be such that $\boldsymbol{\psi} = \boldsymbol{\phi} \circ \alpha$, providing either the $\sim$- or the $\approx$-equivalence of the two parametrizations. Then $\alpha$ is continuous and piecewise smooth.

(Indeed, between any two exceptional points, some of the $\phi_j$ is one-to-one. Then we have $\alpha = \phi_j^{-1} \circ \psi_j$ on the interval in question.)

</div>

---

## 2. Line Integrals

**Convention.** From now on, the curves will be always piecewise smooth.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Line Integral of the Second Kind)</span></p>

Let $\boldsymbol{\phi} = (\phi_1, \dots, \phi_n) : \langle a, b \rangle \to \mathbb{E}_n$ be a parametrization of an oriented curve $L$ and let $\mathbf{f} = (f_1, \dots, f_n) : U \to \mathbb{E}_n$ be a continuous vector function defined on a $U \supseteq \boldsymbol{\phi}[\langle a, b \rangle]$. The **line integral of the second kind** over the (oriented) curve $L$ is the number

$$\text{(II)}\!\int_L \mathbf{f} = \int_a^b \mathbf{f}(\boldsymbol{\phi}(t)) \cdot \boldsymbol{\phi}'(t)\,\mathrm{d}t = \sum_{j=1}^{n}\int_a^b f_j(\boldsymbol{\phi}(t))\phi_j'(t)\,\mathrm{d}t.$$

(Thus the dot in $\int_a^b \mathbf{f}(\boldsymbol{\phi}(t)) \cdot \boldsymbol{\phi}'(t)\,\mathrm{d}t$ indicates the standard scalar product of the $n$-tuples of reals.) If there is no danger of confusion, we write simply $\int_L$ instead of $\text{(II)}\!\int_L$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span></p>

The reader may encounter the line integral of the second kind of, say, vector functions $(P, Q)$ or $(P, Q, R)$, denoted by

$$\int_L P\,\mathrm{d}x + Q\,\mathrm{d}y \quad \text{or} \quad \int_L P\,\mathrm{d}x + Q\,\mathrm{d}y + R\,\mathrm{d}z.$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.2</span></p>

The value of the line integral $\int_L \mathbf{f}$ does not depend on the choice of parametrization of $L$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Suppose $\boldsymbol{\phi} = \boldsymbol{\psi} \circ \alpha$, with an increasing homeomorphism $\alpha : \langle a, b \rangle \to \langle c, d \rangle$. By 1.6.1, $\alpha$ is piecewise smooth. Then by XI.5.5

$$\sum_{j=1}^{n}\int_a^b f_j(\boldsymbol{\phi}(t))\phi_j'(t)\,\mathrm{d}t = \sum_{j=1}^{n}\int_a^b f_j(\boldsymbol{\psi}(\alpha(t)))\psi_j'(\alpha(t))\alpha'(t)\,\mathrm{d}t = \sum_{j=1}^{n}\int_c^d f_j(\boldsymbol{\psi}(t))\psi_j'(t)\,\mathrm{d}t. \quad \square$$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.3</span><span class="math-callout__name">(Properties of Line Integrals)</span></p>

For the operations from 1.5 and 1.4 we have

$$\text{(II)}\!\int_{-L} \mathbf{f} = -\text{(II)}\!\int_L \mathbf{f} \quad \text{and} \quad \text{(II)}\!\int_{L+K} \mathbf{f} = \text{(II)}\!\int_L \mathbf{f} + \text{(II)}\!\int_K \mathbf{f}.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

In the proof of 2.2 above we obtained $\int_c^d$ because $\alpha$ was increasing. For a decreasing $\alpha$ the substitution would yield $\int_d^c = -\int_c^d$, hence $\text{(II)}\!\int_{-L} \mathbf{f} = -\text{(II)}\!\int_L \mathbf{f}$. The other equation is obvious. $\square$

</details>
</div>

### 2.4. Line Integral of the First Kind (For Information)

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Line Integral of the First Kind)</span></p>

Sometimes also called *the line integral according to length*, it is defined for a non-oriented curve parametrized by $\boldsymbol{\phi} = (\phi_1, \dots, \phi_n) : \langle a, b \rangle \to \mathbb{E}_n$. Let $f : U \to \mathbb{R}$ be a continuous real function defined on a $U \supseteq \boldsymbol{\phi}[\langle a, b \rangle]$. The idea is in modifying Riemann integral by computing the sums along a (piecewise smooth) line instead of along an interval. The sums

$$\sum_{i=1}^{k} f(\boldsymbol{\phi}(t_i)) \lVert \boldsymbol{\phi}(t_i) - \boldsymbol{\phi}(t_{i-1}) \rVert$$

considered for partitions $a = t_0 < t_1 < \cdots < t_k = b$ converge with the mesh of the partitions converging to 0 to

$$\int_a^b f(\boldsymbol{\phi}(t)) \lVert \boldsymbol{\phi}'(t) \rVert\,\mathrm{d}t.$$

This integral is called the **line integral of the first kind** over $L$ and denoted by

$$\text{(I)}\!\int_L f \quad \text{or} \quad \text{(I)}\!\int_L f(\mathbf{x}) \lVert \mathrm{d}\mathbf{x} \rVert.$$

</div>

In particular, **the length of a curve** $L$ can be expressed as

$$\text{(I)}\!\int_L 1 = \int_a^b \lVert \boldsymbol{\phi}'(t) \rVert\,\mathrm{d}t.$$

It is easy to see that the line integral of the first kind can be represented as a line integral of the second kind: we have

$$\text{(I)}\!\int_L f = \text{(II)}\!\int_L \mathbf{f} \quad \text{where} \quad \mathbf{f}(\boldsymbol{\phi}(t)) = \frac{\boldsymbol{\phi}'(t)}{\lVert \boldsymbol{\phi}'(t) \rVert}.$$

### 2.5. Complex Line Integral

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Complex Functions of a Real Variable)</span></p>

We will identify the complex plane $\mathbb{C}$ with the Euclidean plane $\mathbb{E}_2$ (viewing $x + iy$ as $(x, y)$ and taking into account that the absolute value of the difference $\lvert z_1 - z_2 \rvert$ coincides with the Euclidean distance). We only must not forget that the structure of $\mathbb{C}$ is richer: in particular we have the multiplication in the *field* $\mathbb{C}$.

A complex function of one real variable will be decomposed into two real functions,

$$f(t) = f_1(t) + if_2(t)$$

and we will define (unsurprisingly) its derivative $f'(t)$ as $f_1'(t) + if_2(t)$ and its Riemann integral as

$$\int_a^b f(t)\,\mathrm{d}t = \int_a^b f_1(t)\,\mathrm{d}t + i\int_a^b f_2(t)\,\mathrm{d}t.$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Complex Line Integral)</span></p>

For an oriented piecewise smooth curve $\phi : \langle a, b \rangle \to \mathbb{C}$ define the **complex line integral** of a complex function of one complex variable by setting

$$\int_L f(z)\,\mathrm{d}z = \int_a^b f(\phi(t)) \cdot \phi'(t)\,\mathrm{d}t.$$

The multiplication indicated by $\cdot$ is now (unlike all the multiplications in previous pages) the **multiplication in the field $\mathbb{C}$**.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.5.3</span></p>

Think of a complex function of one complex variable $f(z) = f_1(z) + if_2(z)$ as of a vector function $\mathbf{f} = (f_1, f_2)$. Then the complex line integral over $L$ can be expressed as a line integral of second kind as follows:

$$\int_L f(z)\,\mathrm{d}z = \text{(II)}\!\int_L (f_1, -f_2) + i\,\text{(II)}\!\int_L (f_2, f_1).$$

Consequently,

- $\int_L f(z)\,\mathrm{d}z$ does not depend on the choice of parametrization, and
- we have $\int_{-L} f(z)\,\mathrm{d}z = -\int_L f(z)\,\mathrm{d}z$ and $\int_{L+K} f(z)\,\mathrm{d}z = \int_L f(z)\,\mathrm{d}z + \int_K f(z)\,\mathrm{d}z$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We have

$$\int_a^b f(\phi(t))\phi'(t)\,\mathrm{d}t = \int_a^b (f_1(\phi(t)) + if_2(\phi(t)))(\phi_1'(t) + i\phi_2'(t))\,\mathrm{d}t$$

$$= \int_a^b (f_1(\phi(t))\phi_1'(t) - f_2(\phi(t))\phi_2'(t))\,\mathrm{d}t + i\int_a^b (f_1(t)\phi_2'(t) + f_2(t)\phi_1'(t))\,\mathrm{d}t$$

$$= \text{(II)}\!\int_L (f_1, -f_2) + i\,\text{(II)}\!\int_L (f_2, f_1). \quad \square$$

</details>
</div>

---

## 3. Green's Theorem

### 3.1. The Jordan Theorem and Regions

A simple closed curve $L$ divides the plane into two connected regions (by "connected" one can understand that any two points can be connected by a curve, "divided" means that points from distinct regions cannot be so connected), one of them bounded, the other unbounded. This is the famous **Jordan theorem**, very easy to understand and visualize, but not very easy to prove.

The bounded region $U$ will be called the **region** of $L$. The curve $C$ is its boundary, and the closure $\overline{U}$ is equal to $U \cup C$ and (being closed and bounded) it is compact; we will speak of $\overline{C}$ as of the **closed region** of $C$.

The integral over a closed region $M$ can be understood as over an interval $J$ containing the region $M$, with the function extended by values zero on $J \setminus M$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.1.1</span><span class="math-callout__name">(Green's Theorem, Green's Formula)</span></p>

Let $L$ be a simple closed piecewise smooth curve oriented counterclockwise, and let $M$ be its closed region. Let $\mathbf{f} = (f_1, f_2)$ be such that both $f_j$ have continuous partial derivatives on the (open) region of $L$. Then

$$\text{(II)}\!\int_L \mathbf{f} = \int_M \left(\frac{\partial f_2}{\partial x_1} - \frac{\partial f_1}{\partial x_2}\right)\mathrm{d}x_1\,\mathrm{d}x_2.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 3.2</span></p>

Let $g : \langle a, b \rangle \to \mathbb{R}$ be a smooth function, let $f(x) \ge c$ for all $x$. Set

$$M = \lbrace (x, y) \mid a \le x \le b,\; c \le y \le g(x) \rbrace.$$

Let $L$ be the closed curve which is the perimeter of $M$. Then the Green formula holds true for $L$ and $M$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Write $L = L_1 + L_2 + L_3 + L_4$ where

$$-L_1 : \phi_1 : \langle a, b \rangle \to \mathbb{R}_2,\; \phi_1(t) = (t, g(t)),$$

$$-L_2 : \phi_2 : \langle c, g(a) \rangle \to \mathbb{R}_2,\; \phi_2(t) = (a, t),$$

$$L_3 : \phi_3 : \langle a, b \rangle \to \mathbb{R}_2,\; \phi_3(t) = (t, c),$$

$$L_4 : \phi_4 : \langle c, g(b) \rangle \to \mathbb{R}_2,\; \phi_4(t) = (b, t).$$

Hence $\phi_1'(t) = (1, g'(t))$, $\phi_2'(t) = \phi_a'(t) = (0, 1)$ and $\phi_3'(t) = (1, 0)$ and we have

$$\text{(II)}\!\int_{L_1} = -\int_a^b f_1(t, g(t))\,\mathrm{d}t - \int_a^b f_2(t, g(t))g'(t)\,\mathrm{d}t,$$

$$\text{(II)}\!\int_{L_2} = -\int_c^{g(a)} f_2(a, t)\,\mathrm{d}t, \quad \text{(II)}\!\int_{L_3} = \int_a^b f_1(t, c)\,\mathrm{d}t, \quad \text{(II)}\!\int_{L_4} = \int_c^{g(b)} f_2(b, t)\,\mathrm{d}t.$$

Substituting $\tau = g(t)$ in the second integral in the formula for $\text{(II)}\!\int_{L_1}$ and extending for the purpose of the integral in two variables the definition of $f_j$ to the interval $J = \langle a, b \rangle \times \langle c, g(a) \rangle$ by values 0 in $J \setminus M$ we obtain

$$f_2(b, x_2) - f_2(a, x_2) = \int_a^b \frac{\partial f_2(x_1, x_2)}{\partial x_1}\,\mathrm{d}x_1, \quad \text{and}$$

$$f_1(x_1, g(x_1)) - f_1(x_1, c) = \int_c^{g(a)} \frac{\partial f_1(x_1, x_2)}{\partial x_2}\,\mathrm{d}x_2$$

so that the formula above transforms to

$$\text{(II)}\!\int_L \mathbf{f} = \int_c^{g(a)}\left(\int_a^b \frac{\partial f_2(x_1, x_2)}{\partial x_1}\,\mathrm{d}x_1\right)\mathrm{d}x_2 - \int_a^b\left(\int_c^{g(a)} \frac{\partial f_1(x_1, x_2)}{\partial x_2}\,\mathrm{d}x_2\right)\mathrm{d}x_1$$

and the statement follows from Fubini's theorem (XVI.4.1). $\square$

</details>
</div>

### 3.3. Extending Green's Theorem

Now we have the Green formula in particular also for quadrangles and right-angled triangles with the hypotenuse possibly curved. Using the fact that $\text{(II)}\!\int_L = -\text{(II)}\!\int_{-L}$ we obtain the formula for any figure that can be cut into such figures.

**3.3.1.** *The Green formula holds for any triangle.*

**3.3.2.** *The Green formula holds for any disc.* (Note, however, that in this decomposition the parametrization from 3.2 would not work: the function $g$ would not have a requested derivative at one of the ends. One can use, for instance, $\phi(t) = (\cos t, \sin t)$. Or, of course, one can cut the disc into more than four pieces.)

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note 3.3.3</span></p>

In fact, any region of a piecewise smooth curve can be decomposed into subregions for which the formula follows from Lemma 3.2. This is easy to visualize. But we will need just simple figures for which the decompositions are obvious and a painstaking proof of the general statement is not necessary.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.4</span><span class="math-callout__name">(Green's Formula with a Singularity)</span></p>

Let $L$ be a circle with center $c$ and let $M$ be its closed region. Let $\mathbf{f}$ be bounded on $M$, let partial derivatives of $f_j$ exist and be continuous on $M \setminus \lbrace c \rbrace$, and let $\int_M \left(\frac{\partial f_2}{\partial x_1} - \frac{\partial f_1}{\partial x_2}\right)\mathrm{d}x_1\,\mathrm{d}x_2$ make sense. Then the Green formula holds.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Denote by $K^n$ the circle with center $c$ and diameter $\frac{1}{n}$ oriented clockwise, let $N(n)$ be its region. Let the $n$ be large enough so that $K^n$ (and hence also $N(n)$) is contained in $M$. Decompose the region between $L$ and $K^n$ into four "curved rectangles" $\widetilde{L}_k^n$ with regions $M_k(n)$. For these curves the Green formula obviously holds (suitable carving the shapes is easy) and we have

$$\text{(II)}\!\int_{\widetilde{L}_k^n} \mathbf{f} = \int_{M_k(n)} \left(\frac{\partial f_2}{\partial x_1} - \frac{\partial f_1}{\partial x_2}\right). \qquad (*)$$

By 2.3,

$$\text{(II)}\!\int_{\widetilde{L}_1^n} + \text{(II)}\!\int_{\widetilde{L}_2^n} + \text{(II)}\!\int_{\widetilde{L}_3^n} + \text{(II)}\!\int_{\widetilde{L}_4^n} = \text{(II)}\!\int_L + \text{(II)}\!\int_{K^n}. \qquad (**)$$

Set $V = V(x_1, x_2) = \frac{\partial f_2}{\partial x_1} - \frac{\partial f_1}{\partial x_2}$. Since we assume the Riemann integral $\int_M V(x_1, x_2)$ exists, $V$ is bounded, that is, we have $\lvert V(x_1, x_2) \rvert < A$ for some $A$. Since $N(n) \subseteq \langle c - \frac{1}{n}, c + \frac{1}{n} \rangle \times \langle c - \frac{1}{n}, c + \frac{1}{n} \rangle$, we have

$$\left\lvert \int_{N(n)} V \right\rvert < \varepsilon \quad \text{for sufficiently large } n.$$

$\mathbf{f}$ is bounded by assumption and hence we also have (we can parametrize $-K^n$, say, by $\phi(t) + \frac{1}{n}(\cos t, \sin t)$)

$$\left\lvert \text{(II)}\!\int_{K^n} \mathbf{f} \right\rvert < \varepsilon \quad \text{for sufficiently large } n.$$

Now we have by $(*)$ and $(**)$

$$\text{(II)}\!\int_L + \text{(II)}\!\int_{K^n} = \int_{M_1(k)} V + \int_{M_2(k)} V + \int_{M_3(k)} V + \int_{M_4(k)} V = \int_M V - \int_{N(k)} V$$

and hence

$$\left\lvert \text{(II)}\!\int_L \mathbf{f} - \int_M V \right\rvert \le \left\lvert \text{(II)}\!\int_{K^n} \mathbf{f} \right\rvert + \left\lvert \int_{N(k)} V \right\rvert$$

and since the right hand side is arbitrarily small the statement follows. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note 3.4.1</span></p>

1. Proposition 3.4 is only a very special case of a general fact. The same holds for a general piecewise smooth simple closed curve $L$ with region $M$ and an exceptional point $c \in M$.
2. The boundedness of $\mathbf{f}$ is essential as one can see for instance in XXII.4.1 below.

</div>

---

# XXII. Basics of Complex Analysis

## 1. Complex Derivative

In the field $\mathbb{C}$ of complex numbers we have not only all the arithmetic operations but also the metric structure allowing to speak about limits. Therefore, given a function $f$ defined in a neighbourhood $U \subseteq \mathbb{C}$ of a point $z$ we can ask whether there exists a limit

$$\lim_{h \to 0} \frac{f(z + h) - f(z)}{h}.$$

If it does we will speak of a **derivative** of $f$ at $z$, and denote the value by $f'(z)$, $\frac{\mathrm{d}f(z)}{\mathrm{d}z}$, $\frac{\mathrm{d}f}{\mathrm{d}z}z$, etc., similarly like in the real context. Thus for instance, like for the real power $x^n$ we have

$$(z^n)' = nz^{n-1}.$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1.1.2</span></p>

A function $f$ has a derivative $A$ at a $z \in \mathbb{C}$ if and only if there exists for a sufficiently small $\delta > 0$ a complex function $\mu : \lbrace h \mid \lvert h \rvert < \delta \rbrace \to \mathbb{C}$ such that

1. $\lim_{h \to 0} \mu(h) = 0$, and
2. for $0 < \lvert h \rvert < \delta$, $\quad f(z + h) - f(z) = Ah + \mu(h)h$.

($\lvert h \rvert$ is of course the absolute value in $\mathbb{C}$.)

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 1.1.3</span></p>

Let $f$ have a derivative at $z$. Then it is continuous at this point.

</div>

### 1.2. A Somewhat Surprising Example

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.2</span></p>

Proposition 1.1.2 seems to suggest that similarly like in the real case, the existence of a derivative can be interpreted as a "geometric tangent" and expresses a sort of smoothness. But it is a much more special property.

Consider $f(z) = \overline{z}$ (the complex conjugate) and compute the derivative. Writing $h = h_1 + ih_2$ we obtain

$$\frac{\overline{z + h} - \overline{z}}{h} = \frac{\overline{z} + \overline{h} - \overline{z}}{h} = \frac{\overline{h}}{h} = \begin{cases} 1 & \text{for } h_1 \ne 0 = h_2, \\ -1 & \text{for } h_1 = 0 \ne h_2. \end{cases}$$

Hence, there is no limit $\lim_{h \to 0} \frac{\overline{z+h} - \overline{z}}{h}$ and our $f$ does not have a derivative at any $z$ whatsoever, while there can be hardly any mapping $\mathbb{C} \to \mathbb{C}$ smoother than this $f$ which is just a mirroring along the real axis.

</div>

### 1.3. Complex Partial Derivatives

$$\frac{\partial f(x, \zeta)}{\partial z} \quad \text{resp.} \quad \frac{\partial f(x, \zeta)}{\partial \zeta}$$

are (similarly as in the real context) derivatives as above with $\zeta$ resp. $z$ fixed.

---

## 2. Cauchy–Riemann Conditions

Let us write a complex $z$ as $x + iy$ with real $x, y$ and express a complex function $f(z)$ of one complex variable as two real functions of two real variables

$$f(z) = P(x, y) + iQ(x, y).$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.1</span><span class="math-callout__name">(Cauchy–Riemann Equations, Necessity)</span></p>

Let $f$ have a derivative at $z = x + iy$. Then $P$ and $Q$ have partial derivatives at $(x, y)$ and satisfy the equations

$$\frac{\partial P}{\partial x}(x,y) = \frac{\partial Q}{\partial y}(x,y) \quad \text{and} \quad \frac{\partial P}{\partial y}(x,y) = -\frac{\partial Q}{\partial x}(x,y).$$

For the derivative $f'$ we then have the formula

$$f' = \frac{\partial P}{\partial x} + i\frac{\partial Q}{\partial x} = \frac{\partial Q}{\partial y} - i\frac{\partial P}{\partial y}.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We have

$$\frac{1}{h}(f(z+h) - f(z)) = \frac{1}{h_1 + ih_2}(P(x+h_1, y+h_2) - P(x,y)) + i\frac{1}{h_1 + ih_2}(Q(x+h_1, y+h_2) - Q(x,y)).$$

If there is a limit $L = \lim_{h \to 0}\frac{1}{h}(f(z+h) - f(z))$ then we have in particular the limits $L = \lim_{h_1 \to 0}\frac{1}{h_1}(f(z+h_1) - f(z))$ and $L = -i\lim_{h_2 \to 0}\frac{1}{ih_2}(f(z+ih_2) - f(z))$. That is,

$$L = \frac{\partial P}{\partial x}(x,y) + i\frac{\partial Q}{\partial x}(x,y)$$

and in the second case,

$$L = \frac{\partial Q}{\partial y}(x,y) - i\frac{\partial P}{\partial y}(x,y). \quad \square$$

</details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Cauchy–Riemann Equations)</span></p>

The (partial differential) equations

$$\frac{\partial P}{\partial x} = \frac{\partial Q}{\partial y} \quad \text{and} \quad \frac{\partial P}{\partial y} = -\frac{\partial Q}{\partial x}$$

are called the **Cauchy–Riemann equations** or the **Cauchy–Riemann conditions**. We have proved that they are necessary for the existence of a derivative. Now we will show that if we, in addition, assume continuity of the partial derivatives, these conditions suffice.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.2</span><span class="math-callout__name">(Cauchy–Riemann Equations, Sufficiency)</span></p>

Let a complex function $f(z) = P(x, y) + iQ(x, y)$ satisfy in an open set $U \subseteq \mathbb{C}$ the Cauchy–Riemann equations and let all the partial derivatives involved be continuous in $U$. Then $f$ has a derivative in $U$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By the Mean Value Theorem for real derivatives we have for suitable $0 < \alpha, \beta, \gamma, \delta < 1$,

$$\frac{1}{h}(f(z+h) - f(z)) = \frac{1}{h}\left(\frac{\partial P(x+h_1, y+\alpha h_2)}{\partial y}h_2 + \frac{\partial P(x+\beta h_1, y)}{\partial x}h_1 + i\frac{\partial Q(x+h_1, y+\gamma h_2)}{\partial y}h_2 + i\frac{\partial Q(x+\delta h_1, y)}{\partial x}h_1\right)$$

and using the Cauchy–Riemann equations we proceed

$$\cdots = \frac{\partial P(x+\beta h_1, y)}{\partial x} + F(h_1, h_2, \beta, \gamma)\frac{ih_2}{h} + i\frac{\partial Q(x+\delta h_1, y)}{\partial x} + G(h_1, h_2, \alpha, \delta)\frac{h_2}{h}$$

where

$$F(h_1, h_2, \beta, \gamma) = \frac{\partial P(x+h_1, y+\gamma h_2)}{\partial x} - \frac{\partial P(x+\beta h_1, y)}{\partial x}$$

$$G(h_1, h_2, \alpha, \delta) = \frac{\partial Q(x+h_1, y+\alpha h_2)}{\partial x} - \frac{\partial Q(x+\delta h_1, y)}{\partial x}.$$

Since $\lvert h_2 \rvert \le \lvert h \rvert$ and $F(\cdots)$ and $G(\cdots)$ converge to 0 for $h \to 0$ by continuity, the expression converges to $\frac{\partial P}{\partial x}(x,y) + i\frac{\partial Q}{\partial x}(x,y)$. $\square$

</details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Holomorphic Function)</span></p>

Complex functions $f : U \to \mathbb{C}$, $U \subseteq \mathbb{C}$, with continuous partial derivatives satisfying the Cauchy–Riemann conditions are said to be **holomorphic** (in $U$).

</div>

---

## 3. More about Complex Line Integral. Primitive Function.

Recall the complex line integral from XXI.2.5.2

$$\int_L f(z)\,\mathrm{d}z = \int_a^b f(\phi(t)) \cdot \phi'(t)\,\mathrm{d}t \qquad (*)$$

and its representation as a line integral of second kind (XXI.2.5.3)

$$\int_L f(z)\,\mathrm{d}z = \text{(II)}\!\int_L (f_1, -f_2) + i\,\text{(II)}\!\int_L (f_2, f_1).$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.1</span><span class="math-callout__name">(Differentiation under the Complex Line Integral)</span></p>

Let $f(z, \gamma)$ be a continuous complex function of two complex variables defined in $V \times U$, $U$ open, and let for each fixed $z \in V$ the function $f(z, -)$ be holomorphic in $U$. Let $L$ be a piecewise smooth oriented curve in $V$. Then for $\gamma \in U$,

$$\frac{\mathrm{d}}{\mathrm{d}\gamma}\int_L f(z, \gamma)\,\mathrm{d}z = \int_L \frac{\partial f(z, \gamma)}{\partial \gamma}\,\mathrm{d}z.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Write $z = x + iy$, $\gamma = \alpha + i\beta$ and $f(z, \gamma) = P(x,y,\alpha,\beta) + iQ(x,y,\alpha,\beta)$. By XXI.2.5.3 we have for $F(\gamma) = \int_L f(z, \gamma)\,\mathrm{d}z$ by the definition of complex line integral

$$F(\gamma) = \mathcal{P}(\alpha, \beta) + i\mathcal{Q}(\alpha, \beta)$$

where

$$\mathcal{P}(\alpha,\beta) = \text{(II)}\!\int_L (P(x,y,\alpha,\beta),\; -Q(x,y,\alpha,\beta)),$$

$$\mathcal{Q}(\alpha,\beta) = \text{(II)}\!\int_L (Q(x,y,\alpha,\beta),\; P(x,y,\alpha,\beta)).$$

Since $f$ is holomorphic at $\gamma$, it satisfies the equations $\frac{\partial P}{\partial \alpha} = \frac{\partial Q}{\partial \beta}$ and $\frac{\partial P}{\partial \beta} = -\frac{\partial Q}{\partial \alpha}$ and we obtain from the definitions of the complex line integral and its expression, and from the proposition on differentiating under the integral sign (Proposition XXI.2.4.2 adapted to line integrals) that

$$\frac{\partial \mathcal{P}}{\partial \alpha} = \text{(II)}\!\int_L\left(\frac{\partial P}{\partial \alpha},\; -\frac{\partial Q}{\partial \alpha}\right) = \text{(II)}\!\int_L\left(\frac{\partial Q}{\partial \beta},\; \frac{\partial P}{\partial \beta}\right) = \frac{\partial \mathcal{Q}}{\partial \beta}, \qquad (*)$$

$$\frac{\partial \mathcal{P}}{\partial \beta} = \text{(II)}\!\int_L\left(\frac{\partial P}{\partial \beta},\; -\frac{\partial Q}{\partial \beta}\right) = -\text{(II)}\!\int_L\left(\frac{\partial Q}{\partial \alpha},\; \frac{\partial P}{\partial \alpha}\right) = -\frac{\partial \mathcal{Q}}{\partial \alpha}$$

and hence the function $F(\gamma)$ is holomorphic in $U$. Using the formula for the derivative from 2.1 we can conclude that

$$\int_L \frac{\partial f(z, \gamma)}{\partial \gamma}\,\mathrm{d}z = \text{(II)}\!\int_L\left(\frac{\partial P}{\partial \alpha},\; -\frac{\partial Q}{\partial \alpha}\right) + i\,\text{(II)}\!\int_L\left(\frac{\partial Q}{\partial \alpha},\; \frac{\partial P}{\partial \alpha}\right) = \frac{\partial \mathcal{P}}{\partial \alpha} + i\frac{\partial \mathcal{Q}}{\partial \alpha} = \frac{\mathrm{d}F}{\mathrm{d}\gamma}. \quad \square$$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.2</span><span class="math-callout__name">(Uniform Convergence and Complex Line Integral)</span></p>

Let $L$ be an oriented curve parametrized by $\phi$ and let $f_n$ be continuous complex functions defined (at least) on $L$. If $f_n$ uniformly converge to $f$ then

$$\int_L f = \lim_n \int_L f_n.$$

In particular if $\sum_{n=1}^{\infty} g_n$ is a uniformly convergent series of continuous functions defined on $L$ then

$$\int_L\left(\sum_{n=1}^{\infty} g_n\right) = \sum_{n=1}^{\infty}\int_L g_n.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Since $\phi$ is piecewise smooth, $\phi'$ is bounded, say by $A$ on $L$. Consequently we have

$$\lvert f_n(\phi(t)) \cdot \phi'(t) - f(\phi(t)) \cdot \phi'(t) \rvert = \lvert (f_n(\phi(t)) - f(\phi(t))) \cdot \phi'(t) \rvert \le \lvert f_n(\phi(t)) - f(\phi(t)) \rvert \cdot A$$

and hence $f_n \rightrightarrows f$ implies that $(f_n \circ \phi) \cdot \phi' \rightrightarrows (f \circ \phi) \cdot \phi'$ and we can use XVIII.4.1 and the formula $(*)$.

For the second statement it now suffices to realize that $\int_L(f + g) = \int_L f + \int_L g$. $\square$

</details>
</div>

### 3.3. Cauchy's Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.3</span><span class="math-callout__name">(Cauchy's Theorem)</span></p>

1. Let $f$ have derivatives in an open set $U \subseteq \mathbb{C}$ and let $L$ be an oriented piecewise smooth simple closed curve such that its closed region is contained in $U$. Then

$$\int_L f(z)\,\mathrm{d}z = 0.$$

2. The formula also holds if $f$ is undefined at one of the points of its region provided $f$ is bounded.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By XXI.2.5.3 we have for $f(z) = P(x, y) + iQ(x, y)$,

$$\int_L f = \text{(II)}\!\int_L (P, -Q) + i\,\text{(II)}\!\int_L (Q, P)$$

and by the Green formula (whether we have in mind the situation from statement 1, or that from statement 2) we obtain

$$\int_L f = \int_M\left(-\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right) + i\int_M\left(\frac{\partial P}{\partial x} - \frac{\partial Q}{\partial y}\right) = 0$$

because by the Cauchy–Riemann equations the functions under the integrals $\int_M$ are zero. $\square$

</details>
</div>

### 3.4. Primitive Function

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Convex Set)</span></p>

Recall that a subset $U \subseteq \mathbb{C}$ is **convex** if for any two $a, b \in U$ the whole of the line segment $\lbrace z \mid z = a + t(b - a),\; 0 \le t \le 1 \rbrace$ is contained in $U$.

</div>

Let $f$ have a derivative in a convex open $U$. Choose an $a \in U$ and for an arbitrary $u \in U$ define $L(a, u)$ as the oriented curve parametrized by $\phi(t) = a + t(u - a)$. Set

$$F(u) = \int_{L(a,u)} f(z)\,\mathrm{d}z.$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.4.1</span><span class="math-callout__name">(Primitive Function)</span></p>

The function $F$ is a primitive function of $f$ in $U$. That is, for each $u \in U$ the (complex) derivative $F'(u)$ exists and is equal to $f(u)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $h$ be such that $u + h \in U$. We have the piecewise smooth closed simple curve $L(a, u) + L(u, u + h) - L(a, u + h)$ and hence by 3.3.1 and XXI.2.3,

$$F(u + h) - F(u) = \int_{L(a,u+h)} f - \int_{L(a,u)} f = \int_{L(u,u+h)} f.$$

Using the parametrization $\phi$ as above (and writing $f = P + iQ$) we obtain

$$\frac{1}{h}(F(u+h) - F(u)) = \frac{1}{h}\int_0^1 f(u + th)\,\mathrm{d}t = \frac{1}{h}\int_0^1 P(u + th)\,\mathrm{d}t + i\frac{1}{h}\int_0^1 Q(u + th)\,\mathrm{d}t = P(u + \theta_1 h) + iQ(u + \theta_2 h)$$

(for the last equality use the Integral Mean Value Theorem XI.3.3) and this converges to $f(u) = P(u) + iQ(u)$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note 3.4.2</span></p>

Working with a convex $U$ was just a matter of convenience. More generally, the same can be proved for simply connected open sets $U$ ("open sets without holes"). Instead of the $L(a, u)$ one can take oriented simple arcs $L$ starting with $a$ and ending in $u$; the integral over such an $L$ depends on $a$ and $u$ only (this is an immediate consequence of 3.3.1 if two such curves $L_1, L_2$ meet solely in $a$ and $u$ — use the simple closed curve $L_1 - L_2$ — but it can be proved for curves that intersect as well). For connected but not simply connected $U$ the situation is different, though.

</div>

---

## 4. Cauchy's Formula

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 4.1</span></p>

Let $K$ be a circle with center $z$ and an arbitrary radius $r$, oriented counterclockwise. Then

$$\int_K \frac{\mathrm{d}\zeta}{\zeta - z} = 2\pi i.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Parametrize $K$ by $\phi(t) = z + r(\cos t + i\sin t)$, $0 \le t \le 2\pi$. Then $\phi'(t) = r(-\sin t + i\cos t)$ and hence

$$\int_K \frac{\mathrm{d}\zeta}{\zeta - z} = \int_0^{2\pi} \frac{r(-\sin t + i\cos t)}{r(\cos t + i\sin t)}\,\mathrm{d}t = \int_0^{2\pi} i\,\mathrm{d}t = 2\pi i,$$

since $-\sin t + i\cos t = i(\cos t + i\sin t)$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note 4.1.1</span></p>

Compare this equality with the value 0 in 3.3.2. The function under the integral is holomorphic everywhere with the exception of just one point. But theorem 3.3.2 cannot be applied since $f$ is not bounded in the region of $K$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.2</span><span class="math-callout__name">(Cauchy's Formula)</span></p>

Let a complex function $f$ of one variable have a derivative in a set $U$ containing the closed region of a circle $K$ with center $z$, oriented counterclockwise. Then

$$\frac{1}{2\pi i}\int_K \frac{f(\zeta)}{\zeta - z}\,\mathrm{d}\zeta = f(z).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We have

$$\int_K \frac{f(\zeta)}{\zeta - z}\,\mathrm{d}\zeta = \int_K \frac{f(z)}{\zeta - z}\,\mathrm{d}\zeta + \int_K \frac{f(\zeta) - f(z)}{\zeta - z}\,\mathrm{d}\zeta = f(z)\int_K \frac{\mathrm{d}\zeta}{\zeta - z} + \int_K \frac{f(\zeta) - f(z)}{\zeta - z}\,\mathrm{d}\zeta = 2\pi i f(z) + \int_K \frac{f(\zeta) - f(z)}{\zeta - z}\,\mathrm{d}\zeta$$

by Lemma 4.1. Now the function $g(\zeta) = \frac{f(\zeta) - f(z)}{\zeta - z}$ is holomorphic for $\zeta \ne z$. In the point $z$ it has a limit, namely the derivative $f'(z)$. Thus it can be completed to a continuous function, hence it is bounded and we can apply 3.3.2 to see that the integral is 0. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note 4.2.1</span></p>

Cauchy formula plays in complex differential calculus a central role similar to that played by the Mean Value Theorem in real analysis. We will see some of it in the next chapter.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.3</span><span class="math-callout__name">(Derivatives of All Orders)</span></p>

If a complex function has a derivative in a neighbourhood of a point $z$ then it has derivatives of all orders in this neighbourhood. More concretely, we have

$$f^{(n)}(z) = \frac{n!}{2\pi i}\int_K \frac{f(\zeta)}{(\zeta - z)^{n+1}}\,\mathrm{d}\zeta.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

This is an immediate consequence of Cauchy's formula and Theorem 3.1: take repeatedly partial derivatives behind the integral sign. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note 4.3.1</span></p>

We have already observed that the existence of a derivative in the complex context differs from the differentiability in real analysis. Now we see how much stronger it is. In the next chapter we will see that in fact only power series have complex derivatives.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 4.4</span></p>

A function $f$ is holomorphic in an open set $U$ iff it has a derivative in $U$.

In other words $f$ has a derivative in $U$ iff it has continuous partial derivatives satisfying the Cauchy–Riemann equations.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

If $f$ has a derivative $f'$, it also has the second derivative $f''$ and hence $f'$ has to be continuous. The other implication is trivial. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note 4.4.1</span></p>

In other words, Theorem 2.2 can be reversed. The question naturally arises whether Theorem 2.1 can be reversed, that is, whether just the Cauchy–Riemann equations suffice (whether they automatically imply continuity). The answer is in the negative.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.5</span></p>

A complex function has a primitive function in a convex open set $U$ if and only if it has a derivative in $U$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

If it has a derivative, it has a primitive function by 3.4.1. On the other hand, if $F$ is a primitive function of $f$, it has by 4.3 the second derivative $F'' = f'$. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span></p>

This is another fact strongly contrasting with real analysis.

</div>

---

# XXIII. A Few More Facts of Complex Analysis

## 1. Taylor Formula

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1.1</span><span class="math-callout__name">(Complex Taylor Series Theorem)</span></p>

Let $f$ be holomorphic in a neighbourhood $V$ of a point $a$. Then in a sufficiently small neighbourhood $U$ of $a$ the function can be written as a power series

$$f(z) = f(a) + \frac{1}{1!}f'(a)(z-a) + \frac{1}{2!}f''(a)(z-a)^2 + \cdots + \frac{1}{n!}f^{(n)}(a)(z-a)^n + \cdots$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We have

$$\frac{1}{\zeta - z} = \frac{1}{\zeta - a} \cdot \frac{1}{1 - \frac{z-a}{\zeta - a}}. \qquad (*)$$

Take a circle $K$ with center $a$ and radius $r$ such that the associated disc (the region of $K$) is contained in $V$. Choose a $q$ with $0 < q < 1$ and a neighbourhood $U$ of $a$ sufficiently small such that for $z \in U$, $\lvert z - a \rvert < rq$. Then we have

$$\zeta \in K \;\Rightarrow\; \left\lvert \frac{z-a}{\zeta - a} \right\rvert < q < 1. \qquad (**)$$

Now we obtain for $x \in U$ from $(*)$

$$\frac{1}{\zeta - z} = \frac{1}{\zeta - a}\left(\sum_{n=0}^{\infty}\left(\frac{z-a}{\zeta-a}\right)^n\right)$$

and hence

$$\frac{f(\zeta)}{\zeta - z} = \sum_{n=0}^{\infty}\frac{f(\zeta)}{\zeta - a}\left(\frac{z-a}{\zeta-a}\right)^n.$$

The continuous function $f$ is bounded on the compact circle $K$ so that by $(**)$ for a suitable $A$,

$$\left\lvert \frac{f(\zeta)}{\zeta - a}\left(\frac{z-a}{\zeta-a}\right)^n \right\rvert < \frac{A}{r} \cdot q^n$$

and hence by XVIII.4.5 the series $\sum_{n=0}^{\infty}\frac{f(\zeta)}{\zeta - a}\left(\frac{z-a}{\zeta - a}\right)^n$ uniformly converges and we can use XXII.3.2 to obtain

$$\int_K \frac{f(\zeta)}{\zeta - z}\,\mathrm{d}\zeta = \sum_{n=0}^{\infty}\int_K \frac{f(\zeta)}{\zeta - a}\left(\frac{z-a}{\zeta-a}\right)^n\mathrm{d}\zeta = \sum_{n=0}^{\infty}(z-a)^n\int_K \frac{f(\zeta)}{(\zeta-a)^{n+1}}\,\mathrm{d}\zeta.$$

Using Cauchy's formula for the first integral and the formula from XXII.4.3 for the last one we conclude that

$$f(z) = \sum_{n=0}^{\infty}\frac{f^{(n)}(a)}{n!}(z-a)^n. \quad \square$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notes 1.1.1</span></p>

1. Thus, all complex functions with derivatives can be (locally) written as power series.
2. Compare the proof of 1.1 with its counterpart in real analysis. The complex variant is actually much simpler: we just write $\frac{1}{\zeta - z}$ as a suitable power series and take the integrals of the individual summands (we just have to know we are allowed to do that), and then we apply the Cauchy formula (and its derivatives). Of course, Cauchy's formula is a very strong tool, but this is not the only reason. In a way, in the real context we are proving a more general theorem: we have a lot of functions that have just a few derivatives for which the theorem applies.

</div>

### 1.2. The Exponential and Goniometric Functions

Using the techniques of complex analysis we can show that the goniometric functions the existence of which we have so far only assumed really exist. First *define* the exponential function for complex variable as the power series

$$e^z = \sum_{n=0}^{\infty}\frac{1}{n!}z^n.$$

We already have it in the real context. The (real) logarithm has been proved to exist (see XII.4), $e^x$ is its inverse and can be written as the (real) Taylor series as above.

We will need the addition formula $e^{u+v} = e^u e^v$ for general complex $u$ and $v$. It is easy:

$$e^u e^v = \left(\sum_{n=0}^{\infty}\frac{1}{n!}u^n\right)\left(\sum_{n=0}^{\infty}\frac{1}{n!}v^n\right) = \sum_{n=0}^{\infty}\left(\sum_{k+r=n}\frac{1}{k!}\frac{1}{r!}u^k v^r\right) = \sum_{n=0}^{\infty}\frac{1}{n!}\left(\sum_{k=0}^{n}\binom{n}{k}u^k v^{n-k}\right) = \sum_{n=0}^{\infty}\frac{1}{n!}(u+v)^n.$$

Now define (for general complex $z$)

$$\sin z = \frac{e^{iz} - e^{-iz}}{2i} = z - \frac{z^3}{3!} + \frac{z^5}{5!} - \frac{z^7}{7!} + \cdots, \quad \text{and}$$

$$\cos z = \frac{e^{iz} + e^{-iz}}{2} = 1 - \frac{z^2}{2!} + \frac{z^4}{4!} - \frac{z^6}{6!} + \cdots$$

We obviously have $\lim_{z \to 0}\frac{\sin z}{z} = 1$ and all the addition formulas are all we will need. We will prove, say, the formula for sinus:

$$\sin u \cos v + \sin v \cos u = \frac{1}{4i}((e^{iu} - e^{-iu})(e^{iv} + e^{-iv}) + (e^{iv} - e^{-iv})(e^{iu} + e^{-iu})) = \frac{1}{2i}(e^{i(u+v)} - e^{-i(u+v)}) = \sin(u+v).$$

---

## 2. Uniqueness Theorem

Recall that two polynomials of degree $n$ agreeing in $n + 1$ arguments coincide. Viewing power series as "polynomials of infinite degree" one may for a moment surmise that two series coinciding in infinite many arguments might coincide everywhere. This conjecture is of course immediately refused by such examples as $\sin nx$ and constant 0.

But in effect this conjecture is not all that wrong. The statement holds true if only the set of points of agreement has an accumulation point (recall XVII.3.1).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 2.2</span><span class="math-callout__name">(Local Uniqueness)</span></p>

Let $f$ and $g$ be holomorphic in an open set $U$ and let $c$ be in $U$. Let $c_n \ne c$, $c = \lim_n c_n$ and $f(c_n) = g(c_n)$ for all $n$. Then $f$ coincides with $g$ in a neighbourhood of $c$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

It suffices to prove that if $f(c_n) = 0$ for all $n$ then $f(z) = 0$ in a neighbourhood of $c$.

Since $c \in U$, the derivative of $f$ in $c$ exists and hence by 1.1 we have in a sufficiently small neighbourhood $V$ of $c$

$$f(z) = \sum_{k=0}^{\infty} a_k(z - c)^k.$$

If $f$ is not constant zero in $V$, some of the $a_k$ is not 0. Let $a_n$ be the first of them. Thus,

$$f(z) = (z - c)^n(a_n + a_{n+1}(z - c) + a_{n+2}(z - c)^2 + \cdots)$$

The series $g(z) = a_n + a_{n+1}(z - c) + a_{n+2}(z - c)^2 + \cdots$ is a continuous function and $g(0) = a_n \ne 0$ and hence $g(z) \ne 0$ in a neighbourhood $W$ of $c$, and $f(z) = (z - c)^n g(z)$ is in $W$ equal to 0 only at $c$. But for sufficiently large $n$, $c_n$ is in $W$, a contradiction. $\square$

</details>
</div>

### 2.3. Connectedness: Just a Few Facts

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Connected and Pathwise Connected)</span></p>

A non-empty metric space $X$ is said to be **disconnected** if there are disjoint non-empty open sets $U, V$ such that $X = U \cup V$. It is **connected** if it is not disconnected.

$X$ is said to be **pathwise connected** if for any two $x, y \in X$ there is a continuous mapping $\phi : \langle a, b \rangle \to X$ such that $\phi(a) = x$ and $\phi(b) = y$.

Of course, we speak of connected resp. pathwise connected *subset* of a metric space if the corresponding *subspace* is connected resp. pathwise connected.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notes 2.3.1</span></p>

1. For good reasons, void space is defined to be disconnected. But all our spaces will be non-void.
2. Since closed sets are precisely the complements of open sets, we see that $X$ is *disconnected* if there are disjoint non-empty closed sets $A, B$ such that $X = A \cup B$.
3. The pathwise connectedness means, of course, connecting of arbitrary pairs of points by curves if we generalize the concept of curve from $\mathbb{E}_n$ to an arbitrary metric space.
4. If we know that a space $X$ is connected we can prove a statement $\mathcal{V}(x)$ about elements $x \in X$ by showing that the set $\lbrace x \mid \mathcal{V}(x) \text{ holds} \rbrace$ is non-empty, open and closed.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Fact 2.3.2</span></p>

The compact interval $\langle a, b \rangle$ is connected.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Suppose that $\langle a, b \rangle = A \cup B$ with $A, B$ disjoint closed subsets, and let, say, $a \in A$. Set $s = \sup \lbrace x \mid \langle a, x \rangle \subseteq A \rbrace$. Since there are $x \in A$ arbitrary close to $s$, $s \in \overline{A} = A$. If $s < b$ there are $x \in B$ arbitrary close to $s$, making $s \in \overline{B} = B$ and contradicting the disjointness. Thus, $s = b$ and $B$ has to be empty. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Fact 2.3.3</span></p>

Each pathwise connected space is connected.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Suppose $X$ is pathwise connected but not connected. Then there are non-empty open disjoint $U, V$ such that $X = U \cup V$. Pick $x \in U$ and $y \in V$. There is a continuous $\phi : \langle a, b \rangle \to X$ such that $\phi(a) = x$ and $\phi(b) = y$. Then $U' = \phi^{-1}[U]$, $V' = \phi^{-1}[V]$ are non-empty disjoint open sets such that $U' \cup V' = \langle a, b \rangle$ contradicting 2.3.2. $\square$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Fact 2.3.4</span></p>

An open subset of $\mathbb{E}_n$ is connected if and only if it is pathwise connected.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $U \subseteq \mathbb{E}_n$ be non-empty open. For $x \in U$ define

$$U(x) = \lbrace y \in U \mid \exists \phi : \langle a, b \rangle \to U,\; \phi(a) = x,\; \phi(b) = y \rbrace.$$

Sets $U(x)$ and $U(y)$ are either disjoint or equal (if $z \in U(x) \cap U(y)$ choose oriented curves $L_1, L_2$ connecting $x$ with $z$ and $z$ with $y$; then $L_1 + L_2$ from XXI.1.4 proves that $y \in U(x)$ and using XXI.1.4 again we see that $U(y) \subseteq U(x)$).

Further, each $U(x)$ is open. Indeed let $y \in U(x)$ and let $L$ be an oriented curve connecting $x$ with $y$. Since $U$ is open there is an $\varepsilon > 0$ such that $\Omega(y, \varepsilon) \subseteq U$. Now for an arbitrary $z \in \Omega(y, \varepsilon)$ we have the oriented line segment $K$ parametrized by $\psi = (t \mapsto y + t(z - y)) : \langle 0, 1 \rangle \to \Omega(y, \varepsilon)$ and hence $L + K$ connecting $x$ with $z$. Thus, $\Omega(y, \varepsilon) \subseteq U(x)$.

Now if $U$ is not pathwise connected there are $x, y$ with $U(x) \cap U(y) = \emptyset$, the set $V = \bigcup \lbrace U(y) \mid y \in U,\; U(x) \cap U(y) = \emptyset \rbrace$ is non-empty open and $U(x) \cup V = U$ and $U$ is not connected. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.4</span><span class="math-callout__name">(Uniqueness Theorem)</span></p>

Let $f$ and $g$ be holomorphic in a connected open set $U$ and let there exist $c$ and $c_n \ne c$ in $U$ such that $c = \lim_n c_n$ and $f(c_n) = g(c_n)$ for all $n$. Then $f = g$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Set

$$V = \lbrace z \mid z \in U,\; f(u) = g(u) \text{ for all } u \text{ in a neighbourhood of } z \rbrace.$$

Then $V$ is by definition open and by 2.2 and the assumption on $c$ it is not empty. Now let $z_n \in V$ and $\lim_n z_n = z$. Then by 2.2, $z \in V$ so that $V$ is also closed, and hence $V = U$ by connectedness (recall 2.3.1.4). $\square$

</details>
</div>

---

## 3. Liouville's Theorem and Fundamental Theorem of Algebra

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 3.1</span></p>

Let $f$ be a complex function defined on a circle $K$ with radius $r$. If $\lvert f(z) \rvert \le A$ for all $z$ then

$$\left\lvert \int_L f(z)\,\mathrm{d}z \right\rvert \le 8A\pi r.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $L$ be parametrized by $\phi : \langle 0, 2\pi \rangle \to \mathbb{C}$ defined by $\phi(t) = c + r\cos t + ir\sin t$ so that $\phi'(t) = -r\sin t + ir\cos t$ and $\lvert \phi_1' \rvert, \lvert \phi_2' \rvert \le r$. Let $f = f_1 + if_2$. Then we have

$$\left\lvert \int_L f \right\rvert = \left\lvert \int_0^{2\pi} f(\phi(t))\phi'(t)\,\mathrm{d}t \right\rvert \le \left\lvert \int_0^{2\pi} f_1\phi_1' \right\rvert + \left\lvert \int_0^{2\pi} f_2\phi_2' \right\rvert + \left\lvert \int_0^{2\pi} f_1\phi_2' \right\rvert + \left\lvert \int_0^{2\pi} f_2\phi_1' \right\rvert \le 4\int_0^{2\pi} Ar\,\mathrm{d}t = 8A\pi r. \quad \square$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span></p>

This estimate is very rough, but it will do for our purposes.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.2</span><span class="math-callout__name">(Liouville's Theorem)</span></p>

If $f$ is bounded and holomorphic in the whole of $\mathbb{C}$ then it is constant.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By XXII.4.3 we have for an arbitrary circle $K$ with center $z$

$$f'(z) = \frac{2!}{2\pi i}\int_K \frac{f(\zeta)}{(\zeta - z)^2}\,\mathrm{d}\zeta.$$

Let $\lvert f(\zeta) \rvert < A$ for all $\zeta$. If we choose the circle $K$ with diameter $r$ we have $(\zeta - z)^2 = r^2$ for $\zeta$ on $K$, and hence

$$\left\lvert \frac{f(\zeta)}{(\zeta - z)^2} \right\rvert < \frac{A}{r^2}.$$

Hence by Lemma 3.1,

$$\lvert f'(z) \rvert < \frac{2!}{2\pi} \cdot 8\frac{A}{r^2}\pi r = \frac{8A}{r}.$$

Since $r$ can be chosen arbitrarily large we see that $f'(z)$ is constant zero, and hence $f$ is a constant. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.3</span><span class="math-callout__name">(Fundamental Theorem of Algebra)</span></p>

Each polynomial $p$ of $\deg(p) > 0$ with complex coefficients has a complex root.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let a polynomial $p(z) = z^n + a_{n-1}z^{n-1} + \cdots + a_1 z + a_0$ have no root. Then the holomorphic function

$$f(z) = \frac{1}{p(z)}$$

is defined on the whole of $\mathbb{C}$. Set $R = 2n\max\lbrace \lvert a_0 \rvert, \lvert a_1 \rvert, \dots, \lvert a_{n-1} \rvert, 1 \rbrace$. Then we have for $\lvert z \rvert \ge R$

$$\lvert p(z) \rvert \ge \lvert z \rvert^n - \lvert a_{n-1}z^{n-1} + \cdots + a_1 z + a_0 \rvert \ge \lvert z \rvert^n - \lvert z \rvert^{n-1}\frac{1}{2}R \ge \lvert z \rvert^{n-1}\frac{1}{2}R \ge \frac{1}{2}R^n.$$

Thus, $\lvert z \rvert \ge R \;\Rightarrow\; \lvert f(z) \rvert \le \frac{2}{R^n}$.

Finally, the set $\lbrace z \mid \lvert z \rvert \le R \rbrace$ is compact and hence the continuous function $f$ is bounded also for $\lvert z \rvert \le R$ and hence everywhere. Thus, by Liouville's Theorem, $f$ is constant and hence so is also $p$. $\square$

</details>
</div>

---

## 4. Notes on Conformal Maps

### 4.1. Angle Between Vectors

Recall from analytic geometry the formula for cosine of the angle $\alpha$ between two (non-zero) vectors $\mathbf{u}, \mathbf{v}$

$$\cos \alpha = \frac{\mathbf{u}\mathbf{v}}{\lVert \mathbf{u} \rVert \lVert \mathbf{v} \rVert}.$$

In view of this formula we will understand in this section under the expression "preserving the angle between $\mathbf{u}$ and $\mathbf{v}$" preserving the value $\frac{\mathbf{u}\mathbf{v}}{\lVert \mathbf{u} \rVert \lVert \mathbf{v} \rVert}$.

### 4.2. Regularity and the Jacobian

Let $U$ be a connected open subset of $\mathbb{C}$. We will be mostly interested in holomorphic functions $f$ and hence we will use (as before) the notation $f(z) = f(x + iy) = P(x,y) + iQ(x,y)$ for any $f : U \to \mathbb{C}$ with partial derivatives. In this notation we have

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Regular Mapping)</span></p>

Recall the Jacobian from XV.4 and also recall that a mapping $f : U \to \mathbb{C}$ with partial derivatives is said to be **regular** if

$$\frac{\mathrm{D}(f)}{\mathrm{D}(z)} = \frac{\mathrm{D}(P,Q)}{\mathrm{D}(x,y)} = \det\begin{pmatrix}\frac{\partial P}{\partial x} & \frac{\partial P}{\partial y} \\ \frac{\partial Q}{\partial x} & \frac{\partial Q}{\partial y}\end{pmatrix} = \frac{\partial P}{\partial x}\frac{\partial Q}{\partial y} - \frac{\partial Q}{\partial x}\frac{\partial P}{\partial y} \ne 0. \qquad (\mathrm{reg})$$

</div>

Let $f : U \to \mathbb{C}$ be a holomorphic function. Then by the Cauchy–Riemann equations the condition (reg) transforms to

$$\frac{\partial P}{\partial x}\frac{\partial Q}{\partial y} - \frac{\partial Q}{\partial x}\frac{\partial P}{\partial y} = \left(\frac{\partial P}{\partial x}\right)^2 + \left(\frac{\partial P}{\partial y}\right)^2 = \left(\frac{\partial Q}{\partial x}\right)^2 + \left(\frac{\partial Q}{\partial y}\right)^2$$

and we observe that *a holomorphic $f$ is regular on an open set $U$ iff for all $z \in U$, $f'(z) \ne 0$*.

### 4.3. Conformal Mappings

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Conformal Mapping)</span></p>

A mapping $f : U \to \mathbb{C}$ is said to be **conformal** if it is regular and if it preserves angles, by which we mean preserving the angles between tangent vectors of curves when transformed by $f$.

</div>

We will show that conformal regular mappings are closely connected with the holomorphic ones.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 4.4.1</span></p>

Let $f$ be a holomorphic mapping. Then for the scalar product $\mathbf{uv}$ of tangent vectors we have (the dot $\cdot$ designates the multiplication of real numbers)

$$\Phi'\Psi' = \frac{\mathrm{D}(f)}{\mathrm{D}(z)} \cdot \phi'\psi'.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Using the Cauchy–Riemann equations we obtain

$$\Phi_1'\Psi_1' + \Phi_2'\Psi_2' = \left(\frac{\partial P}{\partial x}\phi_1' + \frac{\partial P}{\partial y}\phi_2'\right)\left(\frac{\partial P}{\partial x}\psi_1' + \frac{\partial P}{\partial y}\psi_2'\right) + \left(-\frac{\partial P}{\partial y}\phi_1' + \frac{\partial P}{\partial x}\phi_2'\right)\left(-\frac{\partial P}{\partial y}\psi_1' + \frac{\partial P}{\partial x}\psi_2'\right) = (\phi_1'\psi_1' + \phi_2'\psi_2')\left(\left(\frac{\partial P}{\partial x}\right)^2 + \left(\frac{\partial P}{\partial y}\right)^2\right). \quad \square$$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.4.2</span><span class="math-callout__name">(Holomorphic with $f' \ne 0$ is Conformal)</span></p>

A holomorphic mapping $f : U \to \mathbb{C}$ such that $f'(z) \ne 0$ for all $z \in U$ is conformal.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

From Lemma 4.4.1 we also have for the norm that $\lVert \Phi' \rVert^2 = \Phi'\Phi' = \frac{\mathrm{D}(f)}{\mathrm{D}(z)} \lVert \phi' \rVert^2$ so that

$$\frac{\Phi'\Psi'}{\lVert \Phi' \rVert \lVert \Psi' \rVert} = \frac{\frac{\mathrm{D}(f)}{\mathrm{D}(z)}\phi'\psi'}{\sqrt{\frac{\mathrm{D}(f)}{\mathrm{D}(z)}} \lVert \phi' \rVert \sqrt{\frac{\mathrm{D}(f)}{\mathrm{D}(z)}} \lVert \psi' \rVert} = \frac{\phi'\psi'}{\lVert \phi' \rVert \lVert \psi' \rVert}.$$

Recall 4.1. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Note</span></p>

The condition of regularity, that is, $f'(z) \ne 0$, is essential. For instance the mapping $f(z) = z^2$ redoubles the angles at the point $z = 0$.

</div>

### 4.5. Is a Conformal Mapping Necessarily Holomorphic?

No, because for instance the mapping $\operatorname{conj} = (z \mapsto \overline{z}) : \mathbb{C} \to \mathbb{C}$ is conformal (even isometric) but not holomorphic (recall XXII.1.2). But if we would leave it at that it would be a rather cheap answer. In fact, nothing worse than an intervening of $\operatorname{conj}$ can happen. We have

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.5</span><span class="math-callout__name">(Characterization of Conformal Mappings)</span></p>

Let $U$ be an open subset of $\mathbb{C}$ and let $f : U \to \mathbb{C}$ be a regular mapping. Then the following statements are equivalent.

1. $f$ is conformal.
2. $f$ preserves orthogonality.
3. Either $f$ or $\operatorname{conj} \circ f$ is holomorphic.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**(1)$\Rightarrow$(2)** is trivial and **(3)$\Rightarrow$(1)** is in 4.4.2 (the modification by the mapping $\operatorname{conj}$ is obvious).

**(2)$\Rightarrow$(3):** Write $(u, v)$ for the tangent vector $\phi'(t)$ of a parametrization of a curve $\phi$. Transformed by $f$ it becomes

$$\left(\frac{\partial P}{\partial x}u + \frac{\partial P}{\partial y}v,\; \frac{\partial Q}{\partial x}u + \frac{\partial Q}{\partial y}v\right).$$

Now consider for $(u, v)$ two orthogonal vectors $(a, b)$ and $(-b, a)$. Then the scalar product of the transformed vectors

$$\left(\frac{\partial P}{\partial x}a + \frac{\partial P}{\partial y}b,\; \frac{\partial Q}{\partial x}a + \frac{\partial Q}{\partial y}b\right)\left(-\frac{\partial P}{\partial x}b + \frac{\partial P}{\partial y}a,\; -\frac{\partial Q}{\partial x}b + \frac{\partial Q}{\partial y}a\right)$$

should be zero. In particular for the vector $(a, b) = (1, 0)$ this yields

$$\frac{\partial P}{\partial x}\frac{\partial P}{\partial y} + \frac{\partial Q}{\partial x}\frac{\partial Q}{\partial y} = 0 \qquad (1)$$

and for $(a, b) = (1, 1)$ we obtain

$$\left(\frac{\partial P}{\partial y}\right)^2 + \left(\frac{\partial Q}{\partial y}\right)^2 - \left(\frac{\partial P}{\partial x}\right)^2 - \left(\frac{\partial Q}{\partial x}\right)^2 = 0. \qquad (2)$$

Now since $f$ is regular, some of the partial derivatives, say $\frac{\partial Q}{\partial x}(z)$, is not zero (if we concentrate to a particular argument). Set $\lambda = \frac{\partial P}{\partial x}\left(\frac{\partial Q}{\partial x}\right)^{-1}$ so that we have $\frac{\partial P}{\partial x} = \lambda\frac{\partial Q}{\partial x}$ and the equation (1) yields $\lambda\frac{\partial P}{\partial y} + \frac{\partial Q}{\partial y} = 0$, and substituting these two equalities into (2) we obtain

$$(1 + \lambda^2)\left(\frac{\partial P}{\partial y}\right)^2 = (1 + \lambda^2)\left(\frac{\partial Q}{\partial x}\right)^2$$

and since $\lambda$ is real, $1 + \lambda^2 \ne 0$ and we see that $\left(\frac{\partial P}{\partial y}\right)^2 = \left(\frac{\partial Q}{\partial x}\right)^2$.

Now either $\frac{\partial P}{\partial y} = -\frac{\partial Q}{\partial x}$ and then we obtain from (1) that $\frac{\partial P}{\partial x} = \frac{\partial Q}{\partial y}$, and $f$ satisfies the Cauchy–Riemann equations; since the partial derivatives are continuous, $f$ is holomorphic. Or $\frac{\partial P}{\partial y} = \frac{\partial Q}{\partial x}$ and then (1) yields that $\frac{\partial P}{\partial x} = -\frac{\partial Q}{\partial y}$. Then by the Chain Rule, $\operatorname{conj} \circ f$ satisfies the Cauchy–Riemann equations and hence it is holomorphic. $\square$

</details>
</div>
