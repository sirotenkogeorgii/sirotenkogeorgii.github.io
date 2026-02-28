---
layout: default
title: Introduction to Topology
date: 2024-11-01
# excerpt: Connecting differential equations, stability analysis, and attractor theory to the training dynamics of modern machine learning models.
# tags:
#   - dynamical-systems
#   - machine-learning
#   - theory
---

# Introduction to Topology

## 1. Theory of Sets

### 1.1 Introduction

There are two popular definitions of natural numbers

#### Peano axioms

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Peano axioms)</span></p>

We assume a set $\mathbb{N}$, a distinguished element $0$, and a function $S:\mathbb{N}\to\mathbb{N}$ called the **successor**.

1. **$0$ is a natural number.**
   * $0 \in \mathbb{N}$
2. **Successor is closed.**
   * For every $n \in \mathbb{N}$, $S(n) \in \mathbb{N}$.
3. **$0$ is not a successor.**
   * For every $n \in \mathbb{N}$, $S(n) \neq 0$.
4. **Successor is injective.**
   * If $S(m) = S(n)$, then $m = n$.
5. **Induction axiom.**
   * If $A \subseteq \mathbb{N}$ and
     * $0 \in A$, and
     * $n \in A \implies S(n) \in A$, 
       * then $A = \mathbb{N}$.

These axioms characterize the structure of the natural numbers (up to isomorphism). 

These are not the original axioms published by Peano, but are named in his honor. Some forms of the Peano axioms have $1$ in place of $0$. In ordinary arithmetic, the successor of $x$ is $x+1$.

</div>

#### 1.1.2 von Neumann naturals (Set-theoretic definition/construction)

This is a way to **build** the naturals inside set theory (ZF).

Define:
* $0 := \varnothing$
* $S(n) := n \cup \lbrace n\rbrace$
* So $n+1 := S(n)$

Then you get:
* $0=\lbrace\rbrace$
* $1 = 0 \cup \lbrace 0 \rbrace = \lbrace 0 \rbrace = \lbrace \lbrace\rbrace \rbrace$
* $2 = 1 \cup \lbrace 1 \rbrace = \lbrace 0,1 \rbrace = \bigl\lbrace \lbrace\rbrace, \lbrace \lbrace\rbrace \rbrace \bigr\rbrace$ 
* $3 = 2 \cup \lbrace 2 \rbrace = \lbrace 0,1,2 \rbrace = \bigl\lbrace \lbrace\rbrace, \lbrace \lbrace\rbrace \rbrace, \lbrace \lbrace\rbrace, \lbrace \lbrace\rbrace \rbrace \rbrace \bigr\rbrace$
* $n = n-1 \cup \lbrace n-1 \rbrace = \lbrace 0,1,\ldots,n-1 \rbrace = \bigl\lbrace \lbrace\rbrace, \lbrace \lbrace\rbrace \rbrace, \ldots, \lbrace \lbrace\rbrace, \lbrace \lbrace\rbrace \rbrace, \ldots \bigr\rbrace$

A key property here:
* $n = \lbrace 0,1,2,\dots,n-1\rbrace$.

So each natural number is literally the **set of all smaller naturals**.

The **von Neumann naturals are a concrete model of the Peano axioms**. Peano axioms define the naturals **abstractly** (any model is isomorphic). The set-theoretic approach provides a **specific standard model** you can point at.

### 1.2 DeMorgan's Laws

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(DeMorgan's Laws)</span></p>

Let $A\subset S$, $B\subset S$. Then

$$C(A \cup B) = C(A) \cap C(B)$$

$$C(A \cap B) = C(A) \cup C(B)$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(DeMorgan's Laws for indexed families)</span></p>

Let $\lbrace A_{\alpha}\rbrace_{\alpha \in I}$ be an indexed family of subsets of a set $S$. Then

$$C(\bigcup_{\alpha\in I} A_{\alpha}) = \bigcap_{\alpha\in I} C(A_{\alpha})$$

$$C(\bigcap_{\alpha\in I} A_{\alpha}) = \bigcup_{\alpha\in I} C(A_{\alpha})$$

</div>

## 2. Properties of Metric Spaces

At the end of the last chapter on metric spaces, we introduced two adjectives "open" and "closed". These are important because they'll grow up to be the *definition* for a general topological space, once we graduate from metric spaces.

To move forward, we provide a couple niceness adjectives that applies to *entire metric spaces*, rather than just a set relative to a parent space. They are "(totally) bounded" and "complete". These adjectives are specific to metric spaces, but will grow up to become the notion of *compactness*.

### 2.1 Boundedness

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bounded)</span></p>

A metric space $M$ is **bounded** if there is a constant $D$ such that $d(p, q) \le D$ for all $p, q \in M$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Boundedness with radii instead of diameters)</span></p>

A metric space $M$ is bounded if and only if for every point $p \in M$, there is a radius $R$ (possibly depending on $p$) such that $d(p, q) \le R$ for all $q \in M$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Examples of bounded spaces)</span></p>

* Finite intervals like $[0, 1]$ and $(a, b)$ are bounded.
* The unit square $[0, 1]^2$ is bounded.
* $\mathbb{R}^n$ is not bounded for any $n \ge 1$.
* A discrete space on an infinite set is bounded.
* $\mathbb{N}$ is not bounded, despite being homeomorphic to the discrete space!

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Totally Bounded)</span></p>

A metric space is **totally bounded** if for any $\varepsilon > 0$, we can cover $M$ with finitely many $\varepsilon$-neighborhoods.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Examples of totally bounded spaces)</span></p>

* A subset of $\mathbb{R}^n$ is bounded if and only if it is totally bounded.
* So for example $[0, 1]$ or $[0, 2] \times [0, 3]$ is totally bounded.
* In contrast, a discrete space on an infinite set is not totally bounded.

</div>

### 2.2 Completeness

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Cauchy Sequence)</span></p>

Let $x_1, x_2, \dots$ be a sequence which lives in a metric space $M = (M, d_M)$. We say the sequence is **Cauchy** if for any $\varepsilon > 0$, we have

$$d_M(x_m, x_n) < \varepsilon$$

for all sufficiently large $m$ and $n$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Complete)</span></p>

A metric space $M$ is **complete** if every Cauchy sequence converges.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Examples of complete spaces)</span></p>

* $\mathbb{R}$ is complete.
* The discrete space is complete, as the only Cauchy sequences are eventually constant.
* The closed interval $[0, 1]$ is complete.
* $\mathbb{R}^n$ is complete as well.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Non-examples of complete spaces)</span></p>

* The rationals $\mathbb{Q}$ are not complete.
* The open interval $(0, 1)$ is not complete, as the sequence $0.9, 0.99, 0.999, 0.9999, \dots$ is Cauchy but does not converge.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Completion)</span></p>

Every metric space can be "completed", i.e. made into a complete space by adding in some points.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($\mathbb{Q}$ completes to $\mathbb{R}$)</span></p>

The completion of $\mathbb{Q}$ is $\mathbb{R}$.

</div>

### 2.3 Let the Buyer Beware

There is something suspicious about both these notions: neither are preserved under homeomorphism!

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Something fishy is going on here)</span></p>

Let $M = (0, 1)$ and $N = \mathbb{R}$. As we saw much earlier $M$ and $N$ are homeomorphic. However:

* $(0, 1)$ is totally bounded, but not complete.
* $\mathbb{R}$ is complete, but not bounded.

</div>

This is the first hint of something going awry with the metric. As we progress further into our study of topology, we will see that in fact *open sets and closed sets* (which we motivated by using the metric) are the notion that will really shine later on.

### 2.4 Subspaces, and a Confusing Linguistic Point

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Subspace of a Metric Space)</span></p>

Every subset $S \subseteq M$ is a metric space in its own right, by reusing the distance function on $M$. We say that $S$ is a **subspace** of $M$.

</div>

It becomes important to distinguish between:

1. **"Absolute" adjectives** like "complete" or "bounded", which can be applied to both spaces, and hence even to subsets of spaces (by taking a subspace), and
2. **"Relative" adjectives** like "open (in $M$)" and "closed (in $M$)", which make sense only relative to a space.

## 3. Topological Spaces

### 3.1 Forgetting the Metric

A function $f\colon M \to N$ of metric spaces is continuous if and only if the pre-image of every open set in $N$ is open in $M$. This nicely doesn't refer to the metric at all, only the open sets. This motivates forgetting about the metric and starting with the open sets instead.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Topological Space)</span></p>

A **topological space** is a pair $(X, \mathcal{T})$, where $X$ is a set of points, and $\mathcal{T}$ is the **topology**, which consists of several subsets of $X$, called the **open sets** of $X$. The topology must obey the following axioms:

* $\varnothing$ and $X$ are both in $\mathcal{T}$.
* Finite intersections of open sets are also in $\mathcal{T}$.
* Arbitrary unions (possibly infinite) of open sets are also in $\mathcal{T}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Examples of topologies)</span></p>

* Given a metric space $M$, we can let $\mathcal{T}$ be the open sets in the metric sense. The point is that the axioms are satisfied.
* In particular, **discrete space** is a topological space in which every set is open.
* Given $X$, we can let $\mathcal{T} = \lbrace \varnothing, X \rbrace$, the opposite extreme of the discrete space.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Open Neighborhood)</span></p>

An **open neighborhood** of a point $x \in X$ is an open set $U$ which contains $x$.

</div>

### 3.2 Re-definitions

Now that we've defined a topological space, for nearly all of our metric notions we can write down as the definition the one that required only open sets.

#### 3.2.1 Continuity

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Continuous Function)</span></p>

We say function $f\colon X \to Y$ of topological spaces is **continuous** at a point $p \in X$ if the pre-image of any open neighborhood of $f(p)$ contains an open neighborhood of $p$. The function is continuous if it is continuous at every point.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Homeomorphism)</span></p>

A **homeomorphism** of topological spaces $(X, \tau_X)$ and $(Y, \tau_Y)$ is a bijection $f\colon X \to Y$ which induces a bijection from $\tau_X$ to $\tau_Y$: i.e. the bijection preserves open sets.

</div>

Any property defined only in terms of open sets is preserved by homeomorphism. Such a property is called a **topological property**.

#### 3.2.2 Closed Sets

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Closed Set, Closure)</span></p>

In a general topological space $X$, we say that $S \subseteq X$ is **closed** in $X$ if the complement $X \setminus S$ is open in $X$.

If $S \subseteq X$ is any set, the **closure** of $S$, denoted $\overline{S}$, is defined as the smallest closed set containing $S$.

</div>

#### 3.2.3 Properties that Don't Carry Over

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Complete and (totally) bounded are metric properties)</span></p>

The two metric properties "complete" and "(totally) bounded" are not topological properties. They rely on a metric, so as written we cannot apply them to topological spaces. Example 6.3.1 showing $(0,1) \cong \mathbb{R}$ tells us that it is hopeless to find an open-set-only definition for these.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sequences don't work well)</span></p>

You could also try to port over the notion of sequences and convergent sequences. However, this turns out to break a lot of desirable properties. Therefore if we are discussing sequences you should assume that we are working with a metric space.

</div>

### 3.3 Hausdorff Spaces

As you might have guessed, there exist topological spaces which cannot be realized as metric spaces (in other words, are not **metrizable**). One example is to take $X = \lbrace a, b, c \rbrace$ and the topology $\tau_X = \lbrace \varnothing, \lbrace a, b, c \rbrace \rbrace$. This topology can't tell apart any of the points!

To add a sanity condition, we use the **separation axioms** $T_n$. The most common is the $T_2$ axiom:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hausdorff)</span></p>

A topological space $X$ is **Hausdorff** if for any two distinct points $p$ and $q$ in $X$, there exists an open neighborhood $U$ of $p$ and an open neighborhood $V$ of $q$ such that

$$U \cap V = \varnothing.$$

</div>

All metric spaces are Hausdorff. In any case, basically any space we will encounter other than the Zariski topology is Hausdorff.

### 3.4 Subspaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Subspace Topology)</span></p>

Given a topological space $X$, and a subset $S \subseteq X$, we can make $S$ into a topological space by declaring that the open subsets of $S$ are $U \cap S$ for open $U \subseteq X$. This is called the **subspace topology**.

</div>

### 3.5 Connected Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Clopen)</span></p>

A subset $S$ of a topological space $X$ is **clopen** if it is both closed and open in $X$. (Equivalently, both $S$ and its complement are open.)

</div>

For example $\varnothing$ and the entire space are examples of clopen sets. The presence of a nontrivial clopen set other than these two leads to a so-called *disconnected* space.

We say $X$ is **disconnected** if there are nontrivial clopen sets, and **connected** otherwise.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Disconnected and connected spaces)</span></p>

* The metric space $\lbrace (x,y) \mid x^2 + y^2 \le 1 \rbrace \cup \lbrace (x,y) \mid (x-4)^2 + y^2 \le 1 \rbrace \subseteq \mathbb{R}^2$ is disconnected (it consists of two disks).
* The space $[0, 1] \cup [2, 3]$ is disconnected: it consists of two segments, each of which is a clopen set.
* A discrete space on more than one point is disconnected, since *every* set is clopen in the discrete space.
* The set $\lbrace x \in \mathbb{Q} \mid x^2 < 2014 \rbrace$ is a clopen subset of $\mathbb{Q}$. Hence $\mathbb{Q}$ is disconnected too — it has *gaps*.
* $[0, 1]$ is connected.

</div>

### 3.6 Path-Connected Spaces

A stronger and perhaps more intuitive notion of a connected space is a *path-connected* space. The short description: "walk around in the space".

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Path)</span></p>

A **path** in the space $X$ is a continuous function

$$\gamma\colon [0, 1] \to X.$$

Its **endpoints** are the two points $\gamma(0)$ and $\gamma(1)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Path-Connected)</span></p>

A space $X$ is **path-connected** if any two points in it are connected by some path.

</div>

Path-connected implies connected.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Examples of path-connected spaces)</span></p>

* $\mathbb{R}^2$ is path-connected, since we can "connect" any two points with a straight line.
* The unit circle $S^1$ is path-connected, since we can just draw the major or minor arc to connect two points.

</div>

### 3.7 Homotopy and Simply Connected Spaces

Consider the example of the complex plane $\mathbb{C}$ (which you can think of just as $\mathbb{R}^2$) with two points $p$ and $q$. There's a whole bunch of paths from $p$ to $q$ but somehow they're not very different from one another — you can stretch each one to any other one.

But in $\mathbb{C} \setminus \lbrace 0 \rbrace$, you can't move the red string to match the blue string: there's a meteor in the way. The paths are actually different.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Homotopy)</span></p>

Let $\alpha$ and $\beta$ be paths in $X$ whose endpoints coincide. A (path) **homotopy** from $\alpha$ to $\beta$ is a continuous function $F\colon [0, 1]^2 \to X$, which we'll write $F_s(t)$ for $s, t \in [0, 1]$, such that

$$F_0(t) = \alpha(t) \text{ and } F_1(t) = \beta(t) \text{ for all } t \in [0, 1]$$

and moreover

$$\alpha(0) = \beta(0) = F_s(0) \text{ and } \alpha(1) = \beta(1) = F_s(1) \text{ for all } s \in [0, 1].$$

If a path homotopy exists, we say $\alpha$ and $\beta$ are path **homotopic** and write $\alpha \simeq \beta$.

</div>

What this definition is doing is taking $\alpha$ and "continuously deforming" it to $\beta$, while keeping the endpoints fixed. The relation $\simeq$ is an equivalence relation.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Simply Connected)</span></p>

A space $X$ is **simply connected** if it's path-connected and for any points $p$ and $q$, all paths from $p$ to $q$ are homotopic.

</div>

That's why you don't ask questions when walking from $p$ to $q$ in $\mathbb{C}$: there's really only one way to walk. Hence the term "simply" connected.

### 3.8 Bases of Spaces

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Basis)</span></p>

A **basis** for a topological space $X$ is a subset $\mathcal{B}$ of the open sets such that every open set in $X$ is a union of some (possibly infinite) number of elements in $\mathcal{B}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Basis of $\mathbb{R}$)</span></p>

The open intervals form a basis of $\mathbb{R}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Basis of metric spaces)</span></p>

The $r$-neighborhoods form a basis of any metric space $M$.

</div>

*Proof.* Given an open set $U$, for every point $p$ inside $U$, draw an $r_p$-neighborhood $U_p$ contained entirely inside $U$. Then $\bigcup_p U_p$ is contained in $U$ and covers every point inside it. $\square$

## 4. Compactness

One of the most important notions of topological spaces is that of *compactness*. It generalizes the notion of "closed and bounded" in Euclidean space to any topological space.

For metric spaces, there are two equivalent ways of formulating compactness:
* A "natural" definition using *sequences*, called sequential compactness.
* A less natural definition using open covers.

### 4.1 Definition of Sequential Compactness

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Subsequence)</span></p>

A **subsequence** of an infinite sequence $x_1, x_2, \dots$ is exactly what it sounds like: a sequence $x_{i_1}, x_{i_2}, \dots$ where $i_1 < i_2 < \cdots$ are positive integers. Note that the sequence is required to be infinite.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Sequentially Compact)</span></p>

A metric space $M$ is **sequentially compact** if every sequence has a subsequence which converges.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Non-examples of compact metric spaces)</span></p>

* The space $\mathbb{R}$ is not compact: consider the sequence $1, 2, 3, 4, \dots$. Any subsequence explodes, hence $\mathbb{R}$ cannot possibly be compact.
* More generally, if a space is not bounded it cannot be compact.
* The open interval $(0, 1)$ is bounded but not compact: consider the sequence $\frac{1}{2}, \frac{1}{3}, \frac{1}{4}, \dots$. No subsequence can converge to a point in $(0, 1)$ because the sequence "converges to $0$".
* More generally, any space which is not complete cannot be compact.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Examples of compact spaces)</span></p>

* $[0, 1]$ is compact.
* The surface of a sphere, $S^2 = \lbrace (x, y, z) \mid x^2 + y^2 + z^2 = 1 \rbrace$ is compact.
* The unit ball $B^2 = \lbrace (x, y) \mid x^2 + y^2 \le 1 \rbrace$ is compact.
* The **Hawaiian earring** living in $\mathbb{R}^2$ is compact: it consists of mutually tangent circles of radius $\frac{1}{n}$ for each $n$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Closed subsets of compacts)</span></p>

Closed subsets of sequentially compact sets are compact.

</div>

### 4.2 Criteria for Compactness

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Tychonoff's theorem)</span></p>

If $X$ and $Y$ are compact spaces, then so is $X \times Y$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The interval is compact)</span></p>

$[0, 1]$ is compact.

</div>

*Proof sketch.* Split $[0, 1]$ into $[0, \frac{1}{2}] \cup [\frac{1}{2}, 1]$. By Pigeonhole, infinitely many terms of the sequence lie in the left half (say); let $x_1$ be the first one and then keep only the terms in the left half after $x_1$. Now split $[0, \frac{1}{2}]$ into $[0, \frac{1}{4}] \cup [\frac{1}{4}, \frac{1}{2}]$. Again, by Pigeonhole, infinitely many terms fall in some half; pick one of them, call it $x_2$. Rinse and repeat. In this way we generate a sequence $x_1, x_2, \dots$ which is Cauchy, implying that it converges since $[0, 1]$ is complete. $\square$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Bolzano-Weierstraß)</span></p>

A subset of $\mathbb{R}^n$ is compact if and only if it is closed and bounded.

</div>

*Proof.* Look at a closed and bounded $S \subseteq \mathbb{R}^n$. Since it's bounded, it lives inside some box $[a_1, b_1] \times [a_2, b_2] \times \cdots \times [a_n, b_n]$. By Tychonoff's theorem, since each $[a_i, b_i]$ is compact the entire box is. Since $S$ is a closed subset of this compact box, we're done. $\square$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Closed and bounded but not compact)</span></p>

Let $S = \lbrace s_1, s_2, \dots \rbrace$ be any infinite set equipped with the discrete metric. Then $S$ is closed (since all convergent sequences are constant sequences) and $S$ is bounded (all points are a distance $1$ from each other) but it's certainly not compact since the sequence $s_1, s_2, \dots$ doesn't converge.

</div>

One really has to work in $\mathbb{R}^n$ for Bolzano-Weierstraß to be true! In other spaces, this criterion can easily fail.

### 4.3 Compactness Using Open Covers

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Open Cover, Subcover)</span></p>

An **open cover** of a topological space $X$ is a collection of open sets $\lbrace U_\alpha \rbrace$ (possibly infinite or uncountable) which *cover* it: every point in $X$ lies in at least one of the $U_\alpha$, so that

$$X = \bigcup U_\alpha.$$

A **subcover** takes only some of the $U_\alpha$, while ensuring that $X$ remains covered.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Quasicompact, Compact)</span></p>

A topological space $X$ is **quasicompact** if *every* open cover has a finite subcover. It is **compact** if it is also Hausdorff.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(On the Hausdorff hypothesis)</span></p>

The "Hausdorff" hypothesis is a sanity condition which is not worth worrying about unless you're working on algebraic geometry chapters, since all the spaces you will deal with are Hausdorff. All metric spaces are Hausdorff and thus this condition can be safely ignored if you are working with metric spaces.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Example of a finite subcover)</span></p>

Suppose we cover the unit square $M = [0, 1]^2$ by putting an open disk of diameter $1$ centered at every point. This is clearly an open cover. But this is way overkill — we only need about four of these circles to cover the whole square. That's what is meant by a "finite subcover".

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Sequentially compact $\iff$ compact)</span></p>

A metric space $M$ is sequentially compact if and only if it is compact.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(An example of non-compactness)</span></p>

The space $X = [0, 1)$ is not compact in either sense. It is not sequentially compact, because it is not even complete (look at $x_n = 1 - \frac{1}{n}$). To see it is not compact under the covering definition, consider the sets

$$U_m = \left[0, 1 - \frac{1}{m+1}\right)$$

for $m = 1, 2, \dots$. Then $X = \bigcup U_i$; hence the $U_i$ are indeed a cover. But no finite collection of the $U_i$'s will cover $X$.

</div>

### 4.4 Applications of Compactness

Compactness lets us reduce *infinite* open covers to finite ones. Very often one takes an open cover consisting of an open neighborhood of $x \in X$ for every single point $x$ in the space; this is a huge number of open sets, and yet compactness lets us reduce to a finite set.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Compact $\implies$ totally bounded)</span></p>

Let $M$ be compact. Then $M$ is totally bounded.

</div>

*Proof using covers.* For every point $p \in M$, take an $\varepsilon$-neighborhood of $p$, say $U_p$. These cover $M$ for the horrendously stupid reason that each point $p$ is at the very least covered by its open neighborhood $U_p$. Compactness then lets us take a finite subcover. $\square$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Images of compacts are compact)</span></p>

Let $f\colon X \to Y$ be a continuous function, where $X$ is compact. Then the image $f^{\text{img}}(X) \subseteq Y$ is compact.

</div>

*Proof using covers.* Take any open cover $\lbrace V_\alpha \rbrace$ in $Y$ of $f^{\text{img}}(X)$. By continuity of $f$, it pulls back to an open cover $\lbrace U_\alpha \rbrace$ of $X$. Thus some finite subcover of this covers $X$. The corresponding $V$'s cover $f^{\text{img}}(X)$. $\square$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Extreme value theorem)</span></p>

Let $X$ be compact and consider a continuous function $f\colon X \to \mathbb{R}$. Then $f$ achieves a *maximum value* at some point, i.e. there is a point $p \in X$ such that $f(p) \ge f(q)$ for any other $q \in X$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Intermediate value theorem)</span></p>

Consider a continuous function $f\colon [0, 1] \to \mathbb{R}$. Then the image of $f$ is of the form $[a, b]$ for some real numbers $a \le b$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">($1/x$)</span></p>

The compactness hypothesis is really important here. Otherwise, consider the function

$$(0, 1) \to \mathbb{R} \quad \text{by} \quad x \mapsto \frac{1}{x}.$$

This function is not bounded; essentially, the issue is we can't extend it to a function on $[0, 1]$ because it explodes near $x = 0$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Uniformly Continuous)</span></p>

A function $f\colon M \to N$ of metric spaces is called **uniformly continuous** if for any $\varepsilon > 0$, there exists a $\delta > 0$ (depending only on $\varepsilon$) such that whenever $d_M(x, y) < \delta$ we also have $d_N(f(x), f(y)) < \varepsilon$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Uniform continuity)</span></p>

* The functions $\mathbb{R}$ to $\mathbb{R}$ of the form $x \mapsto ax + b$ are all uniformly continuous, since one can always take $\delta = \varepsilon / \lvert a \rvert$ (or $\delta = 1$ if $a = 0$).
* A differentiable function $\mathbb{R} \to \mathbb{R}$ with a bounded derivative is uniformly continuous.
* The function $f\colon \mathbb{R} \to \mathbb{R}$ by $x \mapsto x^2$ is *not* uniformly continuous, since for large $x$, tiny $\delta$ changes to $x$ lead to fairly large changes in $x^2$.
* However, when restricted to $(0, 1)$ or $[0, 1]$ the function $x \mapsto x^2$ becomes uniformly continuous.
* The function $(0, 1) \to \mathbb{R}$ by $x \mapsto 1/x$ is *not* uniformly continuous.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Continuous on compact $\implies$ uniformly continuous)</span></p>

If $M$ is compact and $f\colon M \to N$ is continuous, then $f$ is uniformly continuous.


</div>

*Proof using sequences.* Fix $\varepsilon > 0$, and assume for contradiction that for every $\delta = 1/k$ there exists points $x_k$ and $y_k$ within $\delta$ of each other but with images $\varepsilon$ apart. By compactness, take a convergent subsequence $x_{i_k} \to p$. Then $y_{i_k} \to p$ as well, since the $x_k$'s and $y_k$'s are close to each other. So both sequences $f(x_{i_k})$ and $f(y_{i_k})$ should converge to $f(p)$ by sequential continuity, but this can't be true since the two sequences are always $\varepsilon$ apart. $\square$

### 4.5 (Optional) Equivalence of Formulations of Compactness

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Heine-Borel for general metric spaces)</span></p>

For a metric space $M$, the following are equivalent:

1. Every sequence has a convergent subsequence,
2. The space $M$ is complete and totally bounded, and
3. Every open cover has a finite subcover.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Lebesgue number lemma)</span></p>

Let $M$ be a compact metric space and $\lbrace U_\alpha \rbrace$ an open cover. Then there exists a real number $\delta > 0$, called a **Lebesgue number** for that covering, such that the $\delta$-neighborhood of any point $p$ lies entirely in some $U_\alpha$.

</div>

*Proof.* Assume for contradiction that for every $\delta = 1/k$ there is a point $x_k \in M$ such that its $1/k$-neighborhood isn't contained in any $U_\alpha$. We construct a sequence $x_1, x_2, \dots$; thus we're allowed to take a subsequence which converges to some $x$. Then for every $\varepsilon > 0$ we can find an integer $n$ such that $d(x_n, x) + 1/n < \varepsilon$; thus the $\varepsilon$-neighborhood at $x$ isn't contained in any $U_\alpha$ for every $\varepsilon > 0$. This is impossible, because we assumed $x$ was covered by some open set. $\square$