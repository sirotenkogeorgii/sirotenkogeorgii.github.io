---
layout: default
title: "Mathematical Analysis (Klazar) — Semester 1"
date: 2026-03-16
excerpt: Notes from the first semester of Mathematical Analysis I (Klazar), covering sets, functions, real numbers, sequences, limits, and continuity.
tags:
  - analysis
  - mathematics
---

**Table of Contents**
- TOC
{:toc}

# Lecture 1 — Sets, Functions, Real Numbers

## What Does Mathematical Analysis Analyze?

Mathematical analysis studies **infinite processes and operations**. To illustrate the subtlety involved, consider the following paradox. The alternating harmonic-like series

$$S = 1 - 1 + \frac{1}{2} - \frac{1}{2} + \frac{1}{3} - \frac{1}{3} + \cdots + \frac{1}{n} - \frac{1}{n} + \cdots = 0,$$

since consecutive terms cancel. But after reordering the summands,

$$S = 1 + \frac{1}{2} - 1 + \frac{1}{3} + \frac{1}{4} - \frac{1}{2} + \cdots + \frac{1}{2n-1} + \frac{1}{2n} - \frac{1}{n} + \cdots > 0 \;?$$

because each grouped triple $\frac{1}{2n-1} + \frac{1}{2n} - \frac{1}{n} = \frac{1}{2n(2n-1)} > 0$. This shows that rearranging an infinite sum can change its value — a phenomenon that demands rigorous foundations.

## Review of Logical and Set-Theoretic Notation

**Logical connectives.** $\varphi \lor \psi$ (or), $\varphi \land \psi$ (and), $\varphi \Rightarrow \psi$ (implication), $\varphi \iff \psi$ (equivalence), $\lnot \varphi$ (negation). De Morgan's law:

$$\lnot(\varphi \lor \psi) \iff \lnot \varphi \land \lnot \psi.$$

Brackets and binding strength of each connective matter.

**Quantifiers.** $\forall x\colon \varphi(x)$ means "for every $x$ it holds that $\varphi(x)$"; $\exists x\colon \varphi(x)$ means "there is an $x$ such that $\varphi(x)$ holds." For example,

$$\lnot(\exists\, x\colon \varphi(x)) \iff \forall\, x\colon \lnot \varphi(x).$$

## Sets

We denote the **empty set** by $\emptyset$ and write $x \in A$ to mean that $x$ is an element of the set $A$. A set $M$ may be written by listing its elements, e.g.

$$M = \lbrace a,\, b,\, 2,\, \lbrace\emptyset,\, \lbrace\emptyset\rbrace\rbrace,\, \lbrace a\rbrace \rbrace$$

(this set has 5 elements), or by specifying a property. For example (with $\mathbb{N} := \lbrace 1, 2, 3, \dots \rbrace$),

$$M = \lbrace n \in \mathbb{N} \mid \exists\, m \in \mathbb{N}\colon n = 2m \rbrace$$

is the set of all even natural numbers.

**Relations between sets.**

$$A \subset B \;\stackrel{\text{def}}{\iff}\; \forall\, x\colon\; x \in A \Rightarrow x \in B$$

($A$ is a **subset** of $B$). Two sets $A$ and $B$ are **disjoint** if $\lnot \exists\, x\colon x \in A \land x \in B$. The **axiom of extensionality** determines equality of two sets:

$$A = B \iff (\forall\, x\colon\; x \in A \iff x \in B).$$

**Operations with sets.**

- **Union:** $A \cup B := \lbrace x \mid x \in A \lor x \in B \rbrace$
- **Intersection:** $A \cap B := \lbrace x \in A \mid x \in B \rbrace$
- **General union:** $\bigcup A := \lbrace x \mid \exists\, b \in A\colon x \in b \rbrace$
- **General intersection:** $\bigcap A := \lbrace x \mid \forall\, b \in A\colon x \in b \rbrace$
- **Set difference:** $A \setminus B := \lbrace x \in A \mid x \notin B \rbrace$
- **Power set:** $\mathcal{P}(A) := \lbrace X \mid X \subset A \rbrace$

## Ordered Pairs and Functions

For two sets $A$ and $B$, the set

$$(A, B) := \lbrace\lbrace B, A\rbrace,\, \lbrace A\rbrace\rbrace$$

is the **(ordered) pair** of $A$ and $B$. It always holds that

$$(A, B) = (A', B') \iff A = A' \land B = B'.$$

An **ordered triple** $(A, B, C) := (A, (B, C))$, and similarly for quadruples, etc. Alternatively, $(A, B, C) := \lbrace(1, A),\, (2, B),\, (3, C)\rbrace$.

The **Cartesian product** of sets $A$ and $B$ is

$$A \times B := \lbrace(a, b) \mid a \in A,\; b \in B\rbrace.$$

Any subset $C \subset A \times B$ is a **(binary) relation** between $A$ and $B$. Instead of $(a, b) \in C$ we write $a\,C\,b$ (e.g., $2 < 5$). If $A = B$, we speak of a relation on the set $A$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1</span><span class="math-callout__name">(Function)</span></p>

A **function** (or a **map**) $f$ from a set $A$ to a set $B$ is any ordered triple $(A, B, f)$ such that $f \subset A \times B$ and for every $a \in A$ there is exactly one $b \in B$ with $a\,f\,b$. We write $f\colon A \to B$ and $f(a) = b$.

The set $A$ is the **definition domain** (domain) and $B$ is the **range** (codomain) of $f$. The element $b$ is the **value** of $f$ on the **argument** $a$.

</div>

For $C \subset A$ and $C \subset B$ respectively, the **image** and **preimage** are

$$f[C] := \lbrace f(a) \mid a \in C \rbrace \subset B, \qquad f^{-1}[C] := \lbrace a \in A \mid f(a) \in C \rbrace \subset A.$$

## Families of Functions, Operations with Functions

A **sequence** (in a set $X$) is a function $a\colon \mathbb{N} \to X$. We write $(a_n) = (a_1, a_2, \dots) \subset X$ where $a_n := a(n)$, $n \in \mathbb{N}$ ($= \lbrace 1, 2, \dots \rbrace$). A **word** (over an alphabet $X$) is a function $u\colon [n] \to X$ for some $n \in \mathbb{N}_0 := \mathbb{N} \cup \lbrace 0 \rbrace$, where $[n] := \lbrace 1, 2, \dots, n \rbrace$ and $[0] := \emptyset$. For $n = 0$ the word is $u = \emptyset$. We write $u = a_1 a_2 \dots a_n$ where $a_i := u(i)$.

A **(binary) operation** on a set $X$ is a function $o\colon X \times X \to X$. Instead of $o((a, b)) = c$ we write $a\,o\,b = c$ (e.g. $1 + 1 = 2$).

A function $f\colon X \to Y$ is:
- **injective** (an injection) if $a \neq b \Rightarrow f(a) \neq f(b)$ for all $a, b \in X$,
- **surjective** (onto, a surjection) if $f[X] = Y$,
- **bijective** (one-to-one, a bijection) if it is both injective and surjective,
- **constant** if there is a $c \in Y$ such that $f(a) = c$ for every $a \in X$,
- an **identity function** if $f(a) = a$ for every $a \in X$.

If $f\colon X \to Y$ is an injection, its **inverse function** is $f^{-1}\colon f[X] \to X$ given by $f^{-1}(y) = x \iff f(x) = y$.

For two functions $g\colon X \to Y$ and $f\colon Y \to Z$, their **composition** (or composed function) is

$$f \circ g = f(g)\colon X \to Z, \quad (f \circ g)(a) := f(g(a)),\; a \in X.$$

## Linear Orders, Infima and Suprema

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2</span><span class="math-callout__name">(Linear Order)</span></p>

A **linear order** on a set $A$ is any relation $<$ on $A$ (i.e., $a, b, c \in A$) that is:

1. **Irreflexive:** $\forall\, a\colon\; a \not< a$.
2. **Transitive:** $\forall\, a, b, c\colon\; a < b \land b < c \Rightarrow a < c$.
3. **Trichotomous:** $\forall\, a, b\colon\; a < b \lor b < a \lor a = b$.

</div>

Note that properties 1 and 2 imply that in 3 always exactly one possibility occurs. The notation $a \le b$ means $a < b \lor a = b$, and $a > b$ means $b < a$. We write $(A, <)$ or $(A, <_A)$ to invoke a linear order on $A$.

Let $(A, <)$ be a linear order on $A$ and let $B \subset A$. We say that $B$ is **bounded from above** if there is an $a \in A$ such that $b \le a$ for every $b \in B$. Then $a$ is an **upper bound** of $B$. Boundedness from below and lower bounds are defined similarly. The set of all upper (resp. lower) bounds of $B$ is denoted by $U(B)$ (resp. $L(B)$).

The **maximum** (or the **largest element**) of $B$, which need not exist, is a $b \in B$ such that $\forall\, b' \in B\colon b' \le b$. The **minimum** (or the **least element**) of $B$ is defined similarly. These elements are denoted $\max(B)$ and $\min(B)$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3</span><span class="math-callout__name">(Supremum and Infimum)</span></p>

Suppose that $(A, <)$ is a linear order on $A$ and $B \subset A$.

If $U(B) \neq \emptyset$ and $\min(U(B))$ exists, we call it the **supremum** of $B$ and denote it by

$$\sup(B) := \min(U(B)).$$

If $L(B) \neq \emptyset$ and $\max(L(B))$ exists, we call it the **infimum** of $B$ and denote it by

$$\inf(B) := \max(L(B)).$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Suprema and Infima in $\mathbb{R}$)</span></p>

In the standard linear order of real numbers: $\min((0,1))$ does not exist, $\min([0,1)) = 0$, $\inf((0,1)) = \inf([0,1)) = 0$, and $\sup(\mathbb{N})$ does not exist because $U(\mathbb{N}) = \emptyset$.

</div>

## Ordered Fields

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4</span><span class="math-callout__name">(Ordered Field)</span></p>

An **ordered field** $F$ is an algebraic structure

$$F = (F,\; 0_F,\; 1_F,\; +_F,\; \cdot_F,\; <_F)$$

on a set $F$ that has two distinct distinguished elements $0_F$ and $1_F$ in $F$, two operations $+_F$ and $\cdot_F$ on $F$, and a linear order $<_F$ on $F$, such that the following axioms hold ($a, b, c \in F$):

1. $\forall\, a\colon\; a +_F 0_F = a$ and $a \cdot_F 1_F = a$ (the element $0_F$ is neutral in $+_F$, and $1_F$ is neutral in $\cdot_F$).
2. Both operations $+_F$ and $\cdot_F$ are **associative** and **commutative**.
3. $\forall\, a, b, c\colon\; a \cdot_F (b +_F c) = (a \cdot_F b) +_F (a \cdot_F c)$ (the **distributive law** holds).
4. $\forall\, a\;\exists\, b\colon\; a +_F b = 0_F$, and $\forall\, a \neq 0_F\;\exists\, b\colon\; a \cdot_F b = 1_F$ (**inverse elements** exist).
5. $\forall\, a, b, c\colon\; a <_F b \Rightarrow a +_F c <_F b +_F c$, and $\forall\, a, b\colon\; a, b >_F 0_F \Rightarrow a \cdot_F b >_F 0_F$ ($<_F$ **respects both operations**).

Axioms 1–4 are the axioms of a **field**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Ordered Fields)</span></p>

The **fractions** (or **rational numbers**) $\mathbb{Q} := \lbrace m/n \mid m, n \in \mathbb{Z},\; n \neq 0 \rbrace$, where $\mathbb{Z} := \lbrace \dots, -1, 0, 1, \dots \rbrace$ are the integers, form an ordered field. Another example is $\mathbb{Q}(\sqrt{2}) := \lbrace r + s\sqrt{2} \mid r, s \in \mathbb{Q} \rbrace$.

These two ordered fields differ: the equation $x^2 = 2$ is insoluble in $\mathbb{Q}$ but has a solution in $\mathbb{Q}(\sqrt{2})$.

</div>

## Incompleteness of the Ordered Field $\mathbb{Q}$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5</span><span class="math-callout__name">(Completeness)</span></p>

An ordered field is **complete** if every nonempty subset of it that is bounded from above has a supremum.

</div>

We show that the ordered field $\mathbb{Q}$ is not complete. For this we first recall the **principle of induction** — every nonempty set $X \subset \mathbb{N}$ has the least element — and prove the following theorem.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6</span><span class="math-callout__name">($\sqrt{2} \notin \mathbb{Q}$)</span></p>

In the field of rational numbers, the equation $x^2 = 2$ has no solution.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We assume the contrary that $(a/b)^2 = 2$ for some $a, b \in \mathbb{N}$. Thus $a^2 = 2b^2$, and by the principle of induction we may assume that $a$ is minimal. The number $a^2$ is even, therefore $a$ is even and $a = 2c$ for some $c \in \mathbb{N}$. But then

$$(2c)^2 = 2b^2 \;\leadsto\; 4c^2 = 2b^2 \;\leadsto\; b^2 = 2c^2.$$

Since $b < a$, we have obtained a solution of the equation with a number on the left-hand side that is smaller than $a$. This is a contradiction. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 7</span><span class="math-callout__name">(Incompleteness of $\mathbb{Q}$)</span></p>

The ordered field $\mathbb{Q} = (\mathbb{Q},\, 0,\, 1,\, +,\, \cdot,\, <)$ of fractions is not complete.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We show that the set of fractions

$$X := \lbrace r \in \mathbb{Q} \mid r^2 < 2 \rbrace$$

is nonempty and bounded from above but its supremum does not exist. The first two properties are clear: $\tfrac{4}{3} \in X$ and $x < 2$ for every $x \in X$.

For a contradiction, take the fraction $s := \sup(X)$.

- If $s^2 > 2$, there is a fraction $r > 0$ such that $s - r > 0$ and still $(s - r)^2 > 2$. But then $s - r > x$ for every $x \in X$, which contradicts the fact that $s$ is the least upper bound of $X$.
- If $s^2 < 2$, there is a fraction $r > 0$ such that $(s + r)^2 < 2$. Then $s + r \in X$, which contradicts the fact that $s$ is an upper bound of $X$.

By trichotomy it must be that $s^2 = 2$. But this is impossible by Theorem 6. $\square$

</details>
</div>

## The Complete Ordered Field $\mathbb{R}$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8</span><span class="math-callout__name">(Existence of $\mathbb{R}$)</span></p>

There exists a unique (see Theorem 9) complete ordered field

$$\mathbb{R} = (\mathbb{R},\; 0_\mathbb{R},\; 1_\mathbb{R},\; +_\mathbb{R},\; \cdot_\mathbb{R},\; <_\mathbb{R}).$$

We call it the **field of real numbers**.

</div>

Recall the axiom of completeness: if $X \subset \mathbb{R}$ is nonempty and there is a $y \in \mathbb{R}$ such that $x \le_\mathbb{R} y$ for every $x \in X$, then the set of such numbers $y$ has the least element. We shall omit the lower indices $_\mathbb{R}$ for the neutral elements, operations, and the linear order.

Every ordered field contains as its **prime field** (the smallest subfield) a copy of $\mathbb{Q}$. The completeness of an ordered field makes it unique in the following sense.

A bijection $f\colon F \to G$ between two ordered fields is their **isomorphism** if $f(0_F) = 0_G$, $f(1_F) = 1_G$ and for every $x, y \in F$:

$$f(x +_F y) = f(x) +_G f(y), \quad f(x \cdot_F y) = f(x) \cdot_G f(y), \quad x <_F y \iff f(x) <_G f(y).$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9</span><span class="math-callout__name">(Uniqueness of $\mathbb{R}$)</span></p>

Every two complete ordered fields are isomorphic.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 10</span><span class="math-callout__name">($\sqrt{2} \in \mathbb{R}$)</span></p>

In the field of real numbers, the equation $x^2 = 2$ has a solution.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We take a set similar to that in the proof of Corollary 7,

$$X := \lbrace a \in \mathbb{R} \mid a^2 < 2 \rbrace.$$

By Theorem 8 it has a supremum $s := \sup(X) \in \mathbb{R}$. The same arguments as in that proof show that neither $s^2 < 2$ nor $s^2 > 2$. Hence $s^2 = 2$. $\square$

</details>
</div>

Continuity of a function roughly means (a precise definition will come later) that a small change in the argument of a function results in a small change in the value.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11</span><span class="math-callout__name">(Bolzano–Cauchy Theorem)</span></p>

Let $a \le b$ be real numbers and $f\colon [a, b] \to \mathbb{R}$ be a continuous function such that $f(a)f(b) \le 0$. Then there is a number $c \in [a, b]$ such that $f(c) = 0$.

</div>

## Countable and Uncountable Sets

A set $X$ is **infinite** if there exists an injection $f\colon \mathbb{N} \to X$. If $X$ is not infinite, it is **finite**. One can show that for every finite set $X$ there is a surjection $f\colon \mathbb{N} \to X$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 12</span><span class="math-callout__name">((Un)countable Sets)</span></p>

We define the following kinds of sets:

1. $X$ is **countable** if there is a bijection $f\colon \mathbb{N} \to X$.
2. A set is **at most countable** if it is finite or countable.
3. A set is **uncountable** if it is not at most countable.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 13</span><span class="math-callout__name">($\mathbb{Q}$ is Countable)</span></p>

The set of fractions is countable.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

For a fraction $\tfrac{m}{n} \in \mathbb{Q}$ in lowest terms (meaning $n \in \mathbb{N}$ and the numerator $m \in \mathbb{Z}$ and denominator $n$ are coprime, i.e., the largest $k \in \mathbb{N}$ dividing both $m$ and $n$ is $k = 1$), we define the norm $\lVert \tfrac{m}{n} \rVert := |m| + n \in \mathbb{N}$ and the sets

$$Z_j := \lbrace z_{1,j} < z_{2,j} < \cdots < z_{k_j,j} \mid z_{i,j} \in \mathbb{Q},\; \lVert z_{i,j} \rVert = j \rbrace, \quad j \in \mathbb{N}.$$

For example, $Z_5 = \lbrace -\tfrac{4}{1} < -\tfrac{3}{2} < -\tfrac{2}{3} < -\tfrac{1}{4} < \tfrac{1}{4} < \tfrac{2}{3} < \tfrac{3}{2} < \tfrac{4}{1} \rbrace$ and $k_5 = 8$. Note that $\tfrac{0}{5} \notin Z_5$ because 0 and 5 are not coprime.

Clearly, $j \neq j' \Rightarrow Z_j$ and $Z_{j'}$ are disjoint, every $Z_j$ is finite (and $\neq \emptyset$), and $\bigcup_{j \in \mathbb{N}} Z_j = \mathbb{Q}$. The map $f\colon \mathbb{N} \to \mathbb{Q}$ is defined by

$$f(1) = z_{1,1},\; f(2) = z_{2,1},\; \dots,\; f(k_1) = z_{k_1,1},\; f(k_1 + 1) = z_{1,2},\; \dots$$

— the values of $f$ first run through the $k_1$ sorted fractions in $Z_1$, then through $Z_2$, and so on. It is easy to see that $f$ is a bijection. $\square$

</details>
</div>

We now prove the uncountability of real numbers as a consequence of the following fundamental set-theoretic result, which says that the power set $\mathcal{P}(X)$ is always a much larger set than $X$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 14</span><span class="math-callout__name">(Cantor's Theorem)</span></p>

For no set $X$ there exists a surjection $f\colon X \to \mathcal{P}(X)$ going from $X$ onto its power set.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We assume for the contrary that $X$ is a set and that $f\colon X \to \mathcal{P}(X)$ is a surjective map. We consider the subset

$$Y := \lbrace x \in X \mid x \notin f(x) \rbrace \subset X.$$

Since $f$ is onto, there exists a $y \in X$ such that $f(y) = Y$.
- If $y \in Y$, by the definition of $Y$ we have that $y \notin f(y) = Y$.
- If $y \notin Y = f(y)$, the element $y$ has the property defining $Y$ and therefore $y \in Y$.

In both cases we get a contradiction. $\square$

</details>
</div>

We denote by $\lbrace 0, 1 \rbrace^\mathbb{N}$ the set of all sequences $(a_n) \subset \lbrace 0, 1 \rbrace$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 15</span><span class="math-callout__name">(On 0-1 Sequences)</span></p>

There is no surjection $f\colon \mathbb{N} \to \lbrace 0, 1 \rbrace^\mathbb{N}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The map $g\colon \lbrace 0, 1 \rbrace^\mathbb{N} \to \mathcal{P}(\mathbb{N})$, $g((a_n)) := \lbrace n \in \mathbb{N} \mid a_n = 1 \rbrace$, is obviously a bijection. If the stated surjection $f$ existed, the composite map $g \circ f$ would go from $\mathbb{N}$ onto $\mathcal{P}(\mathbb{N})$, which would contradict Theorem 14. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 16</span><span class="math-callout__name">($\mathbb{R}$ is Uncountable)</span></p>

The set of real numbers is uncountable.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We prove more — there is no surjection $f\colon \mathbb{N} \to \mathbb{R}$. We think of the real numbers as infinite decimal expansions and take the set

$$X := \lbrace 0.a_1 a_2 \dots \mid a_n \in \lbrace 0, 1 \rbrace \rbrace \subset \mathbb{R}$$

of those with only zeros and ones after the decimal point. Clearly, we have a bijection $g\colon X \to \lbrace 0, 1 \rbrace^\mathbb{N}$. If the stated surjection $f$ existed, we could easily obtain from it a surjection $f_0\colon \mathbb{N} \to X$ (set $f_0(n) := f(n)$ if $f(n) \in X$, and $f_0(n) := 0.000\dots$ otherwise). But then the composite map $g \circ f_0$ would go from $\mathbb{N}$ onto $\lbrace 0, 1 \rbrace^\mathbb{N}$, which would contradict Corollary 15. $\square$

</details>
</div>

## Complex Numbers

We recall complex numbers and one fundamental property they possess. It is well known that

$$\mathbb{C} = \lbrace a + bi \mid a, b \in \mathbb{R} \rbrace, \quad i = \sqrt{-1},$$

and that $\mathbb{C}$ with the neutral elements $0_\mathbb{C} := 0 + 0i$ and $1_\mathbb{C} := 1 + 0i$ and the operations

$$(a + bi) +_\mathbb{C} (c + di) := (a + c) + (b + d)i$$

and

$$(a + bi) \cdot_\mathbb{C} (c + di) := (ac - bd) + (ad + bc)i$$

forms a **field**. It has the following important property.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 17</span><span class="math-callout__name">(Fundamental Theorem of Algebra)</span></p>

Every non-constant polynomial $p(z) \in \mathbb{C}[z]$ (with complex coefficients) has a root, a number $z_0 \in \mathbb{C}$ such that $p(z_0) = 0$.

</div>

# Lecture 2 — Existence Theorems for Limits of Sequences

## Computing with Infinities

Recall the real numbers $\mathbb{R}$ and the natural numbers $\mathbb{N} = \lbrace 1, 2, \dots \rbrace$. We denote the latter by letters $i, j, k, l, m, m_0, m_1, \dots, n, n_0, n_1, \dots$ and the letters $a, b, c, d, e, \delta, \varepsilon$ and $\theta$ (possibly with indices) denote real numbers. Always $\delta, \varepsilon, \theta > 0$ and we think of them as close to 0. Recall that $(a_n) \subset \mathbb{R}$ is a real sequence.

For the general notion of a limit we add to $\mathbb{R}$ the **infinities** $+\infty$ and $-\infty$. We get the **extended real axis**

$$\mathbb{R}^* := \mathbb{R} \cup \lbrace +\infty, -\infty \rbrace.$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Rules</span><span class="math-callout__name">(Arithmetic in $\mathbb{R}^*$)</span></p>

We always take only all upper or all lower signs:

- $A \in \mathbb{R} \cup \lbrace \pm\infty \rbrace \Rightarrow A + (\pm\infty) = \pm\infty + A := \pm\infty$,
- $A \in (0, +\infty) \cup \lbrace +\infty \rbrace \Rightarrow A \cdot (\pm\infty) = (\pm\infty) \cdot A := \pm\infty$,
- $A \in (-\infty, 0) \cup \lbrace -\infty \rbrace \Rightarrow A \cdot (\pm\infty) = (\pm\infty) \cdot A := \mp\infty$,
- $a \in \mathbb{R} \Rightarrow \dfrac{a}{\pm\infty} := 0$,
- $-(\pm\infty) := \mp\infty$, $\;-\infty < a < +\infty$.

Subtraction of $A \in \mathbb{R}^*$ reduces to adding $-A$ and division by $a \neq 0$ means multiplication by $1/a$. All remaining values of the operations, i.e.

$$\frac{A}{0},\; (\pm\infty) + (\mp\infty),\; 0 \cdot (\pm\infty),\; (\pm\infty) \cdot 0,\; \frac{\pm\infty}{\pm\infty}\; \text{and}\; \frac{\pm\infty}{\mp\infty},$$

are undefined — these are called **indeterminate expressions**. Elements of $\mathbb{R}^*$ are usually denoted by $A$, $B$, $K$ and $L$.

</div>

## Neighborhoods of Points and Infinities

We remind the notation for real intervals: $(a, b] = \lbrace x \in \mathbb{R} \mid a < x \le b \rbrace$, $(-\infty, a) = \lbrace x \in \mathbb{R} \mid x < a \rbrace$, etc.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1</span><span class="math-callout__name">(Neighborhoods)</span></p>

For any $\varepsilon > 0$, the **$\varepsilon$-neighborhood** of a point $b$ and the **deleted $\varepsilon$-neighborhood** of $b$ are defined, respectively, as

$$U(b, \varepsilon) := (b - \varepsilon,\; b + \varepsilon) \quad \text{and} \quad P(b, \varepsilon) := (b - \varepsilon,\; b) \cup (b,\; b + \varepsilon),$$

so that $P(b, \varepsilon) = U(b, \varepsilon) \setminus \lbrace b \rbrace$. An **$\varepsilon$-neighborhood of infinity** is

$$U(-\infty, \varepsilon) := (-\infty,\; -1/\varepsilon) \quad \text{and} \quad U(+\infty, \varepsilon) := (1/\varepsilon,\; +\infty).$$

We set $P(\pm\infty, \varepsilon) := U(\pm\infty, \varepsilon)$.

</div>

The **main property** of neighborhoods is that if $V, V' \in \lbrace U, P \rbrace$ then

$$A, B \in \mathbb{R}^*,\; A < B \;\Rightarrow\; \exists\, \varepsilon\colon\; V(A, \varepsilon) < V'(B, \varepsilon),$$

i.e., $a < b$ for every $a \in V(A, \varepsilon)$ and every $b \in V'(B, \varepsilon)$. In particular, $A \neq B \Rightarrow \exists\, \varepsilon\colon V(A, \varepsilon) \cap V'(B, \varepsilon) = \emptyset$.

## Limits of Sequences

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2</span><span class="math-callout__name">(Limit of a Sequence)</span></p>

Let $(a_n)$ be a real sequence and $L \in \mathbb{R}^*$. If

$$\forall\, \varepsilon\; \exists\, n_0\colon\; n \ge n_0 \;\Rightarrow\; a_n \in U(L, \varepsilon),$$

we write that $\lim a_n = L$ and say that the sequence $(a_n)$ **has the limit** $L$.

</div>

For $L \in \mathbb{R}$ we speak of a **finite** limit, and for $L = \pm\infty$ of an **infinite** limit. Sequences with finite limits **converge**, otherwise they **diverge**.

If $\lim a_n = a \in \mathbb{R}$ then for every real (and arbitrarily small) $\varepsilon > 0$ there is an index $n_0 \in \mathbb{N}$ such that for every index $n \in \mathbb{N}$ at least $n_0$ the distance between $a_n$ and $a$ is smaller than $\varepsilon$:

$$|a_n - a| < \varepsilon.$$

If $\lim a_n = -\infty$ then for every (negative) $c \in \mathbb{R}$ there is an index $n_0$ such that for every index $n$ at least $n_0$, $a_n < c$. Similarly, with the inequality reversed, for the limit $+\infty$. We also use the notation $\lim_{n \to \infty} a_n = L$ and $a_n \to L$.

The simplest convergent sequence is the **eventually constant** sequence $(a_n)$ with $a_n = a$ for every $n \ge n_0$; then of course $\lim a_n = a$. The popular image that "a sequence gets closer and closer to the limit but never reaches it (possibly only in infinity)" is poetic but incorrect.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3</span><span class="math-callout__name">(Uniqueness of Limits)</span></p>

Limits are unique: $\lim a_n = K$ and $\lim a_n = L \;\Rightarrow\; K = L$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $\lim a_n = K$, $\lim a_n = L$ and let an $\varepsilon$ be given. By Definition 2 there is an $n_0$ such that $n \ge n_0 \Rightarrow a_n \in U(K, \varepsilon)$ and $a_n \in U(L, \varepsilon)$. Thus $\forall\, \varepsilon\colon U(K, \varepsilon) \cap U(L, \varepsilon) \neq \emptyset$. By the main property of neighborhoods, $K = L$. $\square$

</details>
</div>

## Two Limits

We show that $\lim \frac{1}{n} = 0$. It is clear because for every $\varepsilon$ and every $n \ge n_0 := 1 + \lceil 1/\varepsilon \rceil$,

$$0 < \frac{1}{n} \le \frac{1}{1 + \lceil 1/\varepsilon \rceil} < \frac{1}{1/\varepsilon} = \varepsilon \;\leadsto\; 1/n \in U(0, \varepsilon).$$

Here $\lceil a \rceil \in \mathbb{Z}$ denotes the **upper integral part** (ceiling) of the number $a$, the least $v \in \mathbb{Z}$ such that $v \ge a$. Similarly, the **lower integral part** (floor) $\lfloor a \rfloor$ of $a$ is the largest $v \in \mathbb{Z}$ such that $v \le a$.

Our second example is that

$$\sqrt[3]{n} - \sqrt{n} \to -\infty.$$

Indeed, for any given $c < 0$ and every $n \ge n_0 > \max(4c^2, 2^6)$,

$$\sqrt[3]{n} - \sqrt{n} = n^{1/2} \cdot \underbrace{(n^{-1/6} - 1)}_{n > 2^6 \;\Rightarrow\; \cdots < -1/2} < -n^{1/2}/2 < -2|c|/2 = c.$$

It is not necessary to find an optimum $n_0$ in terms of $\varepsilon$ or $c$. It fully suffices to have some value $n_0$ such that for every $n \ge n_0$ the inequality in the definition of limit holds. But to achieve it one still needs some skill in manipulating inequalities and estimates.

## Subsequences

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4</span><span class="math-callout__name">(Subsequence)</span></p>

A sequence $(b_n)$ is a **subsequence** of a sequence $(a_n)$ if there is a sequence (of natural numbers) $m_1 < m_2 < \dots$ such that for every $n$,

$$b_n = a_{m_n}.$$

We will use the notation $(b_n) \preceq (a_n)$.

</div>

It is clear that the relation $\preceq$ on the set of sequences is reflexive and transitive. It is easy to find sequences $(a_n)$ and $(b_n)$ such that $(a_n) \preceq (b_n)$ and $(b_n) \preceq (a_n)$ but $(a_n) \neq (b_n)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5</span><span class="math-callout__name">($\preceq$ Preserves Limits)</span></p>

Let $(b_n) \preceq (a_n)$ and let $\lim a_n = L \in \mathbb{R}^*$. Then also $\lim b_n = L$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

It follows at once from Definitions 2 and 4 because the sequence $(m_n)$ in Definition 4 has the property that $m_n \ge n$ for every $n$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6</span><span class="math-callout__name">(On Subsequences)</span></p>

Let $(a_n)$ be a real sequence and let $A \in \mathbb{R}^*$. The following hold.

1. There is a sequence $(b_n)$ such that $(b_n) \preceq (a_n)$ and $(b_n)$ has a limit.
2. The sequence $(a_n)$ does not have a limit $\iff$ $(a_n)$ has two subsequences with different limits.
3. It is not true that $\lim a_n = A \iff$ there is a sequence $(b_n)$ such that $(b_n) \preceq (a_n)$ and $(b_n)$ has a limit different from $A$.

</div>

Therefore we can always refute that a sequence has a limit by exhibiting two subsequences of it that have different limits. For example,

$$(a_n) := ((-1)^n) = (-1, 1, -1, 1, -1, \dots)$$

does not have a limit because $(1, 1, \dots) \preceq (a_n)$ and $(-1, -1, \dots) \preceq (a_n)$.

## The Limit of the $n$-th Root of $n$

One should be able to recognize when the computation of a given limit is "trivial" and when it is "non-trivial". The former is the case when in the expression whose limit one computes no two growths fight each other, else the latter case occurs.

For instance, to compute the limits $\lim(2^n + 3^n)$ and $\lim \frac{4}{5n-3}$ is trivial, but to compute the limits $\lim(2^n - 3^n)$ and $\lim \frac{4n+7}{5n-3}$ is non-trivial. Often we compute a non-trivial limit by transforming the expression algebraically into a trivial form, like in the above example with $\sqrt[3]{n} - \sqrt{n}$.

The next limit of $n^{1/n}$ is non-trivial because $n \to +\infty$ but $1/n \to 0$ and $(+\infty)^0$ is another indeterminate expression. We will see that the exponent prevails and $n^{1/n} \to 1$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7</span><span class="math-callout__name">($n^{1/n} \to 1$)</span></p>

It holds that

$$\lim_{n \to \infty} n^{1/n} = \lim_{n \to \infty} \sqrt[n]{n} = 1.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Always $n^{1/n} \ge 1$. If $n^{1/n} \not\to 1$, there would be a number $c > 0$ and a sequence $2 \le n_1 < n_2 < \dots$ such that for every $i$ one has that $n_i^{1/n_i} > 1 + c$. By the Binomial Theorem we would have for every $i$ that

$$n_i > (1+c)^{n_i} = \sum_{j=0}^{n_i} \binom{n_i}{j} c^j = 1 + \binom{n_i}{1}c + \binom{n_i}{2}c^2 + \cdots + \binom{n_i}{n_i}c^{n_i} \ge \frac{n_i(n_i - 1)}{2} \cdot c^2$$

and so, for every $i$,

$$n_i > \frac{n_i(n_i - 1)}{2} \cdot c^2 \;\leadsto\; 1 + \frac{2}{c^2} > n_i.$$

This is a contradiction, the sequence $n_1 < n_2 < \dots$ cannot be upper-bounded. $\square$

</details>
</div>

## When a Sequence Has a Limit

We present four existence theorems (Theorems 9, 10, 13 and 15 of this lecture). The existence of the limit of a sequence and its value are not influenced by changing only finitely many terms in the sequence. Thus properties ensuring existence of limits should also be **robust** in this sense — they should be independent of changes of finitely many terms. For instance, boundedness of sequences, which we define next, is a robust property.

### Monotone Sequences

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 8</span><span class="math-callout__name">(Monotonicity)</span></p>

A sequence $(a_n)$ is

- **non-decreasing** if $a_n \le a_{n+1}$ for every $n$,
- **non-decreasing from $n_0$** if $a_n \le a_{n+1}$ for every $n \ge n_0$,
- **non-increasing** if $a_n \ge a_{n+1}$ for every $n$,
- **non-increasing from $n_0$** if $a_n \ge a_{n+1}$ for every $n \ge n_0$,
- **monotonous** if it is non-decreasing or non-increasing,
- **monotonous from $n_0$** if it is non-decreasing from $n_0$ or non-increasing from $n_0$.

The inequalities $a_n < a_{n+1}$, respectively $a_n > a_{n+1}$, yield a **(strictly) increasing**, respectively a **(strictly) decreasing**, sequence.

</div>

A sequence $(a_n)$ is **bounded from above** (BFA) if $\exists\, c\; \forall\, n\colon a_n < c$, else $(a_n)$ is **unbounded from above** (UFA). Taking the reverse inequality we get **boundedness**, resp. **unboundedness**, of $(a_n)$ **from below** (BFB and UFB). The sequence is **bounded** if it is bounded both from above and from below. Each of these five properties of sequences is robust.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9</span><span class="math-callout__name">(On Monotone Sequences)</span></p>

Any real sequence $(a_n)$ that is monotonous from $n_0$ has a limit. If $(a_n)$ is non-decreasing from $n_0$ then

$$\lim_{n \to \infty} a_n = \begin{cases} \sup(\lbrace a_n \mid n \ge n_0 \rbrace) & \dots\; (a_n) \text{ is BFA} \newline +\infty & \dots\; (a_n) \text{ is UFA.} \end{cases}$$

If $(a_n)$ is non-increasing from $n_0$ then

$$\lim_{n \to \infty} a_n = \begin{cases} \inf(\lbrace a_n \mid n \ge n_0 \rbrace) & \dots\; (a_n) \text{ is BFB} \newline -\infty & \dots\; (a_n) \text{ is UFB.} \end{cases}$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We consider only the first case of a sequence that is non-decreasing from $n_0$, the other case is similar.

If $(a_n)$ is unbounded from above then for any given $c$ there exists an $m$ such that $a_m > \max(c, a_1, a_2, \dots, a_{n_0})$. Thus $a_m > c$ and $m > n_0$. Therefore for every $n \ge m$,

$$a_n \ge a_{n-1} \ge \cdots \ge a_m > c \;\leadsto\; a_n > c$$

and $a_n \to +\infty$.

For $(a_n)$ bounded from above we set $s := \sup(\lbrace a_n \mid n \ge n_0 \rbrace)$. Suppose that an $\varepsilon > 0$ is given. By the definition of supremum there exists an $m \ge n_0$ such that $s - \varepsilon < a_m \le s$. Thus for every $n \ge m$,

$$s - \varepsilon < a_m \le \cdots \le a_{n-1} \le a_n \le s \;\leadsto\; s - \varepsilon < a_n \le s$$

and $a_n \to s$. $\square$

</details>
</div>

### Quasi-Monotonous Sequences

*(Not included in the exam.)* We say that a sequence $(a_n)$ is **quasi-monotone from $n_0$** if

$$n \ge n_0 \;\Rightarrow\; \text{every set } \lbrace m \mid a_m < a_n \rbrace \text{ is finite}$$

or

$$n \ge n_0 \;\Rightarrow\; \text{every set } \lbrace m \mid a_m > a_n \rbrace \text{ is finite.}$$

Clearly, any sequence monotonous from an $n_0$ is quasi-monotonous from the same $n_0$. It is not hard to devise a sequence that is not monotonous from $n_0$ for any $n_0$, but is quasi-monotonous from some $n_0$.

Quasi-monotonous sequences (with $n_0 = 1$) were introduced by the English mathematician *Godfrey H. Hardy (1877–1947)*.

In the next theorem we use the quantities $\limsup$ and $\liminf$ of a sequence. They are always defined, may attain values $\pm\infty$, and will be introduced in the next lecture.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10</span><span class="math-callout__name">(On Quasi-Monotonous Sequences)</span></p>

Every sequence $(a_n) \subset \mathbb{R}$ that is quasi-monotonous from $n_0$ has a limit. If $(a_n)$ satisfies the 1st, resp. the 2nd, condition in the definition, then

$$\lim a_n = \limsup a_n \in \mathbb{R}^*, \quad \text{resp.}\quad \lim a_n = \liminf a_n \in \mathbb{R}^*.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We consider only the case that $(a_n)$ satisfies the 1st condition for some $n_0$, the other case is similar. We suppose that $(a_n)$ is unbounded from above and that a $c$ is given. Hence there is an $m \ge n_0$ such that $a_m > c$. By the 1st condition there exists a $k$ such that $a_n \ge a_m > c$ for every $n \ge k$. Thus $a_n \to +\infty = \limsup a_n$.

Suppose that $(a_n)$ is bounded from above, that $s := \limsup a_n \in \mathbb{R}$ and that an $\varepsilon$ is given. By the definition of $\limsup a_n$, in

$$s - \varepsilon < a_m < s + \varepsilon$$

the first inequality holds for infinitely many $m$ and the second one for almost all $m$. By the 1st condition there exists a $k$ such that $s - \varepsilon < a_n < s + \varepsilon$ holds for every $n \ge k$. Thus $a_n \to s$. $\square$

</details>
</div>

### The Bolzano–Weierstrass Theorem

For its proof we need the next result that is of independent interest.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11</span><span class="math-callout__name">(Existence of Monotonous Subsequences)</span></p>

Any sequence of real numbers has a monotonous subsequence.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

For a given $(a_n)$ we consider the set

$$M := \lbrace n \mid \forall\, m\colon\; n \le m \;\Rightarrow\; a_n \ge a_m \rbrace.$$

If $M$ is infinite, $M = \lbrace m_1 < m_2 < \dots \rbrace$, we have the non-increasing subsequence $(a_{m_n})$.

If $M$ is finite, we take a number $m_1 > \max(M)$. Then certainly $m_1 \notin M$ and there is a number $m_2 > m_1$ such that $a_{m_1} < a_{m_2}$. As $m_2 \notin M$, there is an $m_3 > m_2$ such that $a_{m_2} < a_{m_3}$. And so on — we get a non-decreasing, even strictly increasing, subsequence $(a_{m_n})$. $\square$

</details>
</div>

The theorem on monotone sequences and the previous proposition have the following two immediate corollaries. The first one is part 1 of Proposition 6.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 12</span><span class="math-callout__name">(Subsequence with a Limit)</span></p>

Any real sequence has a subsequence that has a limit.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 13</span><span class="math-callout__name">(Bolzano–Weierstrass)</span></p>

Any bounded sequence of real numbers has a convergent subsequence.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $(a_n)$ be a bounded sequence and $(b_n) \preceq (a_n)$ be its monotonous subsequence guaranteed by Proposition 11. It is clear that $(b_n)$ is bounded and by Theorem 9 it has a finite limit. $\square$

</details>
</div>

*Karl Weierstrass (1815–1897)* was a German mathematician, "the father of the modern mathematical analysis". The priest, philosopher and mathematician *Bernard Bolzano (1781–1848)* had Italian, German and Czech roots. In Prague there is a street named after him (near Hlavní nádraží), in the Celetná street a plaque commemorates him and his grave is in Olšanské hřbitovy (cemetery).

### The Cauchy Condition

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 14</span><span class="math-callout__name">(Cauchy Sequences)</span></p>

A sequence $(a_n) \subset \mathbb{R}$ is **Cauchy** if

$$\forall\, \varepsilon\; \exists\, n_0\colon\; m, n \ge n_0 \;\Rightarrow\; |a_m - a_n| < \varepsilon,$$

i.e., $a_m \in U(a_n, \varepsilon)$.

</div>

The property that a sequence of real numbers is Cauchy is a robust one. It is clear that every Cauchy sequence is bounded.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 15</span><span class="math-callout__name">(Cauchy Condition)</span></p>

A sequence $(a_n) \subset \mathbb{R}$ converges if and only if $(a_n)$ is Cauchy.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**($\Rightarrow$)** Let $\lim a_n = a$ and let an $\varepsilon$ be given. Then there is an $n_0$ such that $n \ge n_0 \Rightarrow |a_n - a| < \varepsilon/2$. Thus

$$m, n \ge n_0 \;\Rightarrow\; |a_m - a_n| \le |a_m - a| + |a - a_n| < \varepsilon/2 + \varepsilon/2 = \varepsilon$$

and $(a_n)$ is a Cauchy sequence. (We have used that $a_m - a_n = (a_m - a) + (a - a_n)$ and the triangle inequality $|c + d| \le |c| + |d|$.)

**($\Leftarrow$)** Let $(a_n)$ be a Cauchy sequence. We know that $(a_n)$ is bounded, and therefore by the Bolzano–Weierstrass theorem it has a convergent subsequence $(a_{m_n})$ with a limit $a$. For a given $\varepsilon$ we have an $n_0$ such that $n \ge n_0 \Rightarrow |a_{m_n} - a| < \varepsilon/2$ and that $m, n \ge n_0 \Rightarrow |a_m - a_n| < \varepsilon/2$. Always $m_n \ge n$ and therefore

$$n \ge n_0 \;\Rightarrow\; |a_n - a| \le |a_n - a_{m_n}| + |a_{m_n} - a| < \varepsilon/2 + \varepsilon/2 = \varepsilon.$$

Thus $a_n \to a$. $\square$

</details>
</div>

The French mathematician *Augustin-Louis Cauchy (1789–1857)* also lived in Prague, in political exile in 1833–1838.

# Lecture 3 — Arithmetic of Limits. Limits and Order. Infinite Series

## Arithmetic of Limits

Last time we considered existence of limits of real sequences. Now we look at relations between limits and arithmetical operations, and between limits and ordering. Recall that $(a_n)$, $(b_n)$ and $(c_n)$ denote real sequences and that $\mathbb{R}^*$ is the extended real line. Recall how to compute with infinities.

The **variant form** of the triangle inequality $\|a + b\| \le \|a\| + \|b\|$ is

$$|a - b| \ge |a| - |b|.$$

The next theorem is useful for finding limits. In its proof we use a reformulation of existence of finite limits: if $(a_n) \subset \mathbb{R}$ and $a \in \mathbb{R}$ then

$$\lim a_n = a \iff a_n =: a + \underbrace{e_n}_{\text{error term}} \;\text{ where }\; e_n \to 0$$

(so $e_n = a_n - a$).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1</span><span class="math-callout__name">(Arithmetic of Limits)</span></p>

Let $\lim a_n = K \in \mathbb{R}^\ast$ and $\lim b_n = L \in \mathbb{R}^\ast$. Then

1. $\lim(a_n + b_n) = K + L$ whenever the right-hand side is defined.
2. $\lim a_n b_n = KL$ whenever the right-hand side is defined.
3. $\lim a_n / b_n = K/L$ whenever the right-hand side is defined. For $b_n = 0$ we set $a_n / b_n := 0$.

RS in 1 is not defined $\iff$ $K = -L = \pm\infty$. RS in 2 is not defined $\iff$ $K = \pm\infty$ or $K = \pm\infty$ and $L = 0$. RS in 3 is not defined $\iff$ $L = 0$ or $K = \pm\infty$ and $L = \pm\infty$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**Part 1.** Let $K, L \in \mathbb{R}$ and an $\varepsilon$ be given. There is an $n_0$ such that $n \ge n_0 \Rightarrow a_n =: K + c_n$ and $b_n =: L + d_n$ with $|c_n|, |d_n| < \varepsilon/2$. Thus $n \ge n_0 \Rightarrow a_n + b_n = K + L + \underbrace{c_n + d_n}_{e_n}$ with $|e_n| \le |c_n| + |d_n| < \varepsilon/2 + \varepsilon/2 = \varepsilon$. So $a_n + b_n \to K + L$.

Let $K \neq -\infty$, $L = +\infty$ and a $c$ be given. Then $a_n > d$ for every $n$ and some $d$, and $b_n > -d + c$ for every $n \ge n_0$. Thus $n \ge n_0 \Rightarrow a_n + b_n > d + (-d + c) = c$ and $a_n + b_n \to +\infty$. The case $K = -\infty$ and $L \neq +\infty$ is similar.

**Part 2.** Let $K, L \in \mathbb{R}$ and an $\varepsilon \in (0, 1)$ be given. There is an $n_0$ such that $n \ge n_0 \Rightarrow a_n =: K + c_n$ and $b_n =: L + d_n$ with $|c_n|, |d_n| < \varepsilon$. Thus $n \ge n_0 \Rightarrow a_n b_n = KL + \underbrace{c_n L + d_n K + c_n d_n}_{e_n}$ and $|e_n| \stackrel{\Delta\text{-ineq.}}{\le} \varepsilon(|K| + |L| + 1) \to 0$ as $\varepsilon \to 0$. So $a_n b_n \to KL$.

Let $K > 0$, $L = -\infty$ and a $c < 0$ be given. Then $a_n > d > 0$ for every $n$ and some $d > 0$, and $b_n < c/d$ for every $n \ge n_0$. Thus $n \ge n_0 \Rightarrow a_n b_n < d(c/d) = c$ and $a_n b_n \to -\infty$. The other cases with $K = \pm\infty$ or $L = \pm\infty$ are similar.

**Part 3.** Let $K, L \in \mathbb{R}$ with $L \neq 0$ and an $\varepsilon$ be given. There is an $n_0$ such that $n \ge n_0 \Rightarrow a_n =: K + c_n$ and $b_n =: L + d_n$ with $|c_n| < \varepsilon$ and $|d_n| < \min(\varepsilon, |L|/2)$. For every $n \ge n_0$ we then have that

$$\frac{a_n}{b_n} = \frac{K + c_n}{L + d_n} = \frac{K/L + c_n/L}{1 + d_n/L} = \frac{K}{L} - \underbrace{\frac{Kd_n/L^2}{1 + d_n/L} + \frac{c_n/L}{1 + d_n/L}}_{e_n}$$

and, due to $|1 + d_n/L| \ge 1 - |d_n|/|L| \ge 1 - 1/2 = 1/2$,

$$|e_n| \le \frac{|K|\varepsilon/L^2}{1/2} + \frac{\varepsilon/|L|}{1/2} = \varepsilon \cdot \left(\frac{2|K|}{L^2} + \frac{2}{|L|}\right) \to 0$$

as $\varepsilon \to 0$. Thus $a_n/b_n \to K/L$.

Let $K \in \mathbb{R}$, $L = -\infty$ and an $\varepsilon$ be given. Hence $(a_n)$ is bounded, $|a_n| < c$ for every $n$ and some $c > 0$, and there is an $n_0$ such that $n \ge n_0 \Rightarrow b_n < -c/\varepsilon$. Hence $n \ge n_0 \Rightarrow |a_n/b_n| < c/|b_n| < c/(c/\varepsilon) = \varepsilon$ and $a_n/b_n \to 0$. The other cases when $L \neq 0$ and either $K = \pm\infty$ or $L = \pm\infty$ are similar. $\square$

</details>
</div>

The theorem of course does not give complete characterization of arithmetic of limits. Even when its assumptions are not met, i.e., $K$ or $L$ does not exist or the right-hand side is not defined, the (unique) limit on the left-hand side may still exist. Below we list several such cases without proof.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2</span><span class="math-callout__name">(Supplement 1)</span></p>

Even when $K = \lim a_n$ does not exist, the following hold.

1. $(a_n)$ bounded and $L = \lim b_n = \pm\infty \;\Rightarrow\; \lim(a_n + b_n) = L$.
2. $(a_n)$ bounded and $L = \lim b_n = 0 \;\Rightarrow\; \lim a_n b_n = 0$.
3. $(a_n)$ satisfies $a_n > c > 0$ for $n \ge n_0$ and $L = \lim b_n = \pm\infty \;\Rightarrow\; \lim a_n b_n = L$.
4. $(a_n)$ bounded and $L = \lim b_n = \pm\infty \;\Rightarrow\; \lim a_n / b_n = 0$.
5. $(a_n)$ satisfies $a_n > c > 0$ for $n \ge n_0$, $b_n > 0$ for $n \ge n_0$ and $L = \lim b_n = 0 \;\Rightarrow\; \lim a_n / b_n = +\infty$.

</div>

But often it indeed happens that when the assumptions of the theorem are not satisfied, the limit on the left-hand side is not uniquely determined or does not exist.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3</span><span class="math-callout__name">(Supplement 2)</span></p>

For every $A \in \mathbb{R}^*$ there exist sequences $(a_n)$, $(b_n)$ such that

1. $\lim a_n = +\infty$, $\lim b_n = -\infty$ and $\lim(a_n + b_n) = A$.
2. $\lim a_n = 0$, $\lim b_n = \pm\infty$ and $\lim a_n b_n = A$ and $\lim a_n / b_n = A$.
3. $\lim a_n = \lim b_n = 0$ or $\lim a_n = \pm\infty$, $\lim b_n = \pm\infty$ and $\lim a_n / b_n = A$.

The limits $\lim(a_n + b_n)$, $\lim a_n b_n$ and $\lim a_n / b_n$ in 1–3 also need not exist.

</div>

## Sequences Given by Recurrences

We meet the first real limits of sequences, such as $\lim(n^{1/3} - n^{1/2})$, $\lim \frac{2n-3}{5n+4}$, etc. We saw earlier that these are in reality problems on limits of functions. We explain how to compute limits of recurrent sequences. We use in it the so called **AG inequality** (the inequality between arithmetic and geometric mean): for every two real numbers $a, b \ge 0$,

$$\frac{a + b}{2} \ge \sqrt{ab}.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4</span><span class="math-callout__name">(Recurrent Limit)</span></p>

Let $(a_n)$ be given by $a_1 = 1$ and, for $n \ge 2$,

$$a_n = \frac{a_{n-1}}{2} + \frac{1}{a_{n-1}}.$$

Then $\lim a_n = \sqrt{2}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Suppose that $L := \lim a_n \in \mathbb{R}$ exists and is finite. Since limits are preserved by subsequences, $\lim a_{n-1} = L$. By parts 3, 2 and 1 of the previous theorem we have that $\lim \frac{1}{a_{n-1}} = \frac{1}{L}$ for $L \neq 0$, always $\lim \frac{a_{n-1}}{2} = \frac{L}{2}$ and $\lim\left(\frac{a_{n-1}}{2} + \frac{1}{a_{n-1}}\right) = \frac{L}{2} + \frac{1}{L}$ for $L \neq 0$. Thus

$$L = \frac{L}{2} + \frac{1}{L} \;\leadsto\; L^2 - L^2/2 = 1 \;\leadsto\; L^2 = 2$$

and we have two solutions $L = \sqrt{2}$ and $L = -\sqrt{2}$. If we prove that $(a_n)$ converges, we get that $\lim a_n = \sqrt{2}$ because $a_n > 0$ for every $n$ and therefore $L \ge 0$ (as we see in the next part of the lecture).

However, to exclude that $L = 0$ we need an inequality stronger than $L \ge 0$. But next we show that $a_n \ge \sqrt{2}$ for every $n \ge 2$. Thus $L \ge \sqrt{2} > 0$, if $L$ exists, and certainly $L \neq 0$.

In order that we can use the theorem on monotone sequences from the last lecture, we show that $(a_n)$ is non-increasing from $n_0 = 2$. We need that for every $n \ge 2$,

$$a_n \ge a_{n+1} = \frac{a_n}{2} + \frac{1}{a_n} \iff \frac{a_n^2}{2} \ge 1 \iff a_n \ge \sqrt{2}.$$

But for $n \ge 2$ the AG inequality indeed shows that

$$a_n = \frac{a_{n-1}}{2} + \frac{1}{a_{n-1}} = \frac{a_{n-1} + 2a_{n-1}^{-1}}{2} \ge \sqrt{a_{n-1} \cdot 2a_{n-1}^{-1}} = \sqrt{2}.$$

Hence $(a_n)$ is non-increasing from $n_0 = 2$ and non-negative, so bounded from below. By the theorem on monotone sequences, $(a_n)$ has a non-negative finite limit. Thus $\lim a_n = \sqrt{2}$. $\square$

</details>
</div>

The initial computation, i.e., solving the equation obtained by replacing all $a_n, a_{n-1}, \dots$ in the recurrence with the putative limit $L$, is of any value only if we show that $(a_n)$ converges. For instance, the recurrence sequence $(a_n)$ defined by $a_1 = 1$ and $a_n = -a_{n-1}$ does *not* have the limit $\lim a_n = 0$ although the equation $L = -L$ has a unique solution $L = 0$, because $(a_n) = (1, -1, 1, -1, \dots)$ does not have a limit (as we noted earlier).

## Geometric Sequences

In the proof of the next proposition we use the simple observation that

$$\lim a_n = 0 \iff \lim |a_n| = 0.$$

Indeed, $a_n \to 0 \iff \forall\, \varepsilon\; \exists\, n_0\colon n \ge n_0 \Rightarrow \|a_n\| < \varepsilon \iff \forall\, \varepsilon\; \exists\, n_0\colon n \ge n_0 \Rightarrow |\,|a_n|\,| < \varepsilon \iff \|a_n\| \to 0$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5</span><span class="math-callout__name">(Geometric Sequences)</span></p>

For $q \in \mathbb{R}$ the limit

$$\lim_{n \to \infty} q^n \begin{cases} = 0 & \dots\; |q| < 1, \text{ i.e., } -1 < q < 1, \newline = 1 & \dots\; q = 1, \newline = +\infty & \dots\; q > 1, \newline \text{does not exist} & \dots\; q \le -1. \end{cases}$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**1.** Let $|q| < 1$. By the observation we may assume that $q \ge 0$. Then $(q^n)$ is non-increasing, bounded from below (since $q^n \ge 0$) and by the theorem on monotone sequences it has a non-negative finite limit $L$. From $q^n = q \cdot q^{n-1}$ we get the equation $L = q \cdot L \leadsto L = 0/(1 - q) = 0$.

**2.** For $q = 1$ we have the constant sequence $(1, 1, \dots)$ that has the limit 1.

**3.** Let $q > 1$. By part 1 of this proposition and by part 5 of Proposition 2,

$$\lim_{n \to \infty} q^n = \lim_{n \to \infty} \frac{1}{(1/q)^n} = \frac{1}{0^+} = +\infty.$$

**4.** Let $q \le -1$. For $q = -1$, $(q^n) = (-1, 1, -1, 1, \dots)$ does not have a limit because it has a subsequence with limit 1 and a subsequence with limit $-1$. For $q < -1$, $(q^n)$ does not have a limit because by part 3 of this proposition and by arithmetic of limits it has a subsequence with limit $+\infty$ and a subsequence with limit $-\infty$. $\square$

</details>
</div>

## Limits and Order

Relations between limits of real sequences and the linear order $(\mathbb{R}^*, <)$ are described in the next two theorems.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6</span><span class="math-callout__name">(Limits and Order)</span></p>

Suppose that $K, L \in \mathbb{R}^*$ and that $(a_n)$, $(b_n)$ are two real sequences with $\lim a_n = K$ and $\lim b_n = L$. The following hold.

1. If $K < L$ then there is an $n_0$ such that for every two (possibly distinct!) indices $m, n \ge n_0$ one has that $a_m < b_n$.
2. If for every $n_0$ there are indices $m$ and $n$ such that $m, n \ge n_0$ and $a_m \ge b_n$, then $K \ge L$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**1.** Let $K < L$. As we know from Lecture 2, there is an $\varepsilon$ such that $U(K, \varepsilon) < U(L, \varepsilon)$. By the definition of limit there is an $n_0$ such that $m, n \ge n_0 \Rightarrow a_m \in U(K, \varepsilon)$ and $b_n \in U(L, \varepsilon)$. So $m, n \ge n_0 \Rightarrow a_m < b_n$.

**2.** We get the proof for free by elementary logic: the implication $\varphi \Rightarrow \psi$ is equivalent with the variant $\lnot \psi \Rightarrow \lnot \varphi$. But the variant of the implication in part 1 is exactly part 2. $\square$

</details>
</div>

The previous theorem is often (in fact, almost always) presented in the weaker form that if $K < L$ then there is an $n_0$ such that $n \ge n_0 \Rightarrow a_n < b_n$. Similarly for part 2.

Strict inequality between terms of two sequences may turn in limit into equality of their limits: for $(a_n) := (1/n)$ and $(b_n) := (0, 0, \dots)$ we have that $a_m > b_n$ for every $m$ and $n$, but $\lim a_n = \lim b_n = 0$.

For $a, b \in \mathbb{R}$ we denote by $I(a, b)$ the interval with endpoints $a$ and $b$:

$$I(a, b) = [a, b] \text{ for } a \le b \quad \text{and} \quad I(a, b) = [b, a] \text{ for } a \ge b.$$

A set $M \subset \mathbb{R}$ is **convex** if $\forall\, a, b \in M\colon I(a, b) \subset M$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7</span><span class="math-callout__name">(On Intervals)</span></p>

Convex sets of real numbers are exactly: $\emptyset$, the singletons $\lbrace a \rbrace$ for $a \in \mathbb{R}$, the whole $\mathbb{R}$ and the intervals $(a, b)$, $(-\infty, a)$, $(a, +\infty)$, $(a, b]$, $[a, b)$, $[a, b]$, $(-\infty, a]$ and $[a, +\infty)$ for real numbers $a < b$.

</div>

Every neighborhood $U(A, \varepsilon)$ is convex. No deleted neighborhood $P(a, \varepsilon)$ is convex.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8</span><span class="math-callout__name">(Two Cops Theorem)</span></p>

Let $a \in \mathbb{R}$ and $(a_n)$, $(b_n)$ and $(c_n)$ be three real sequences such that

$$\lim a_n = \lim c_n = a \;\land\; \forall\, n \ge n_0\colon\; b_n \in I(a_n, c_n).$$

Then $\lim b_n = a$ too.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $a$, $(a_n)$, $(b_n)$ and $(c_n)$ be as stated and an $\varepsilon$ be given. By the definition of limit there is an $n_0$ such that $n \ge n_0 \Rightarrow a_n, c_n \in U(a, \varepsilon)$. Since $U(a, \varepsilon)$ is convex, $n \ge n_0 \Rightarrow I(a_n, c_n) \subset U(a, \varepsilon)$. Due to the assumption we have that $n \ge n_0 \Rightarrow b_n \in U(a, \varepsilon)$ and $b_n \to a$. $\square$

</details>
</div>

Two cops, the sequences $(a_n)$ and $(c_n)$, lead a suspect, the sequence $(b_n)$, to the common limit $a$. For infinite limit, one cop suffices: if $\lim a_n = -\infty$ and $b_n \le a_n$ for every $n \ge n_0$, then also $\lim b_n = -\infty$. Similarly for the limit $+\infty$. The two cops theorem is often presented in a weaker form, with inequalities $a_n \le b_n \le c_n$ in place of the membership $b_n \in I(a_n, c_n)$. Then the cops are firmly positioned to the left and right sides of the suspect, whereas in our version of the theorem they are allowed to exchange their places.

## Limes Inferior and Limes Superior of a Sequence

These are residues of Latin mathematical terminology which mean "the least limit" and "the largest limit", respectively.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 9</span><span class="math-callout__name">(Limit Point)</span></p>

Let $A \in \mathbb{R}^*$ and $(a_n) \subset \mathbb{R}$. We say that $A$ is a **limit point** of the sequence $(a_n)$ if $\lim a_{m_n} = A$ for a subsequence $(a_{m_n})$ of $(a_n)$. We set

$$H(a_n) := \lbrace A \in \mathbb{R}^* \mid A \text{ is a limit point of } (a_n) \rbrace \subset \mathbb{R}^*.$$

**Limes inferior** of a sequence $(a_n)$, denoted $\liminf a_n$, is defined as $\min(H(a_n))$ in the linear order $(\mathbb{R}^*, <)$. **Limes superior** of the sequence, denoted $\limsup a_n$, is the element $\max(H(a_n))$.

</div>

In the next theorem we show that these elements exist.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10</span><span class="math-callout__name">(Liminf and Limsup Exist)</span></p>

For every real sequence $(a_n)$, the set $H(a_n)$ is nonempty and it possesses in the linear order $(\mathbb{R}^*, <)$ both minimum and maximum element.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $(a_n)$ be a real sequence. In Lecture 2 we proved that $(a_n)$ has a subsequence with a limit, so that $H(a_n) \neq \emptyset$. We prove the existence of $\max(H(a_n))$; for the minimum element one proceeds similarly.

In the following four cases, which cover all possibilities, we define an element $A \in \mathbb{R}^*$.

**(i)** If $H(a_n) = \lbrace -\infty \rbrace$ then $A := -\infty$.
**(ii)** If $+\infty \in H(a_n)$ then $A := +\infty$.
**(iii)** If $H(a_n) \cap \mathbb{R} \neq \emptyset$ and this set is unbounded from above then $A := +\infty$.
**(iv)** Finally, if $+\infty \notin H(a_n)$ and the set $H(a_n) \cap \mathbb{R}$ is nonempty and bounded from above, then

$$A := \sup(H(a_n) \cap \mathbb{R}) \in \mathbb{R}.$$

We show that always $A = \max(H(a_n))$. In the cases (i) and (ii) it clearly holds. In the cases (iii) and (iv) it is clear that $A \ge h$ for every $h \in H(a_n)$ and it suffices to show that $A \in H(a_n)$. In the cases (iii) and (iv) it is also clear that there is a sequence

$$(b_n) \subset H(a_n) \cap \mathbb{R} \text{ such that } \lim b_n = A.$$

Since every number $b_n$ is the limit of a subsequence of $(a_n)$, we easily find a subsequence $(a_{m_n})$ such that

$$\forall\, n\colon\; a_{m_n} \in U(b_n,\, 1/n).$$

But then $\lim a_{m_n} = \lim b_n = A$ and $A \in H(a_n)$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 11</span><span class="math-callout__name">(Properties of Liminf and Limsup)</span></p>

For any real sequence $(a_n)$ the following hold.

1. If $\lim a_n$ exists then $H(a_n) = \lbrace \lim a_n \rbrace$.
2. Three exclusive cases occur and cover all possibilities:
   *(i)* $(a_n)$ is unbounded from above and $\limsup a_n = +\infty$,
   *(ii)* $\lim a_n = -\infty$ and $\limsup a_n = -\infty$,
   *(iii)* $\limsup a_n$ is finite and

   $$\limsup a_n = \lim_{n \to \infty} \bigl(\sup(\lbrace a_m \mid m \ge n \rbrace)\bigr) \in \mathbb{R}.$$

3. Three exclusive cases occur and cover all possibilities:
   *(i)* $(a_n)$ is unbounded from below and $\liminf a_n = -\infty$,
   *(ii)* $\lim a_n = +\infty$ and $\liminf a_n = +\infty$,
   *(iii)* $\liminf a_n$ is finite and

   $$\liminf a_n = \lim_{n \to \infty} \bigl(\inf(\lbrace a_m \mid m \ge n \rbrace)\bigr) \in \mathbb{R}.$$

4. Always $\liminf a_n \le \limsup a_n$ and equality holds if and only if $\lim a_n$ exists and then

   $$\liminf a_n = \limsup a_n = \lim a_n.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**1.** This is obvious: any subsequence of a sequence with a limit has the same limit.

**2.** The first two cases are more or less clear. Suppose neither of them occurs. For every $n$ we set $A_n := \lbrace a_m \mid m \ge n \rbrace$ and $b_n := \sup(A_n)$. Every set $A_n$ is bounded from above and nonempty, so that $(b_n)$ is a well defined real sequence that is obviously non-increasing. By the theorem on monotone sequences it has a limit $L := \lim b_n \in \mathbb{R} \cup \lbrace -\infty \rbrace$. Clearly $L \neq -\infty$ for else we would have that $\lim a_n = -\infty$. Hence $L \in \mathbb{R}$. By the definition of supremum,

$$\forall\, n\; \exists\, m\; (\ge n)\colon\; b_n - 1/n < a_m \le b_n.$$

It follows from this that $\lim b_n = L \in H(a_n)$. Suppose that $L$ is not the maximum of $H(a_n)$. Then there is a $\delta > 0$ such that for infinitely many $m$ one has that $a_m > L + \delta$. Then we can take an $n$ such that $b_n < L + \delta$. But then there would be an $m \ge n$ such that $a_m > L + \delta > b_n$, in contradiction with the definition of $b_n$. Thus $L = \max(H(a_n)) = \limsup a_n$.

**3.** Proof of this is very similar to the previous proof.

**4.** The first claim is clear. To prove the second one it suffices to prove that if $\liminf a_n = \limsup a_n =: L$ then $\lim a_n = L$. When $L = \pm\infty$, $\lim a_n = L$ by case (ii) in part 2 or part 3. Let $L \in \mathbb{R}$ and an $\varepsilon$ be given. By case (iii) in parts 2 and 3 we take an $n$ such that

$$L - \varepsilon < \inf(\lbrace a_m \mid m \ge n \rbrace) \le \sup(\lbrace a_m \mid m \ge n \rbrace) < L + \varepsilon.$$

Then $m \ge n \Rightarrow L - \varepsilon < a_m < L + \varepsilon$ so that $a_n \to L$. $\square$

</details>
</div>

## Infinite Series

We introduce basic notions of the theory of (infinite) series. More about series will follow in the next lecture.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 12</span><span class="math-callout__name">(Infinite Series)</span></p>

An **(infinite) series** is again a sequence $(a_n) \subset \mathbb{R}$. Its **sum** is the limit

$$\sum a_n = \sum_{n=1}^{\infty} a_n = a_1 + a_2 + \cdots := \lim(a_1 + a_2 + \cdots + a_n)$$

if it exists. The terms in the sequence $(a_1 + a_2 + \cdots + a_n)$ are called **partial sums** (of the series).

</div>

The symbols $\sum a_n$, $\sum_{n=1}^{\infty} a_n$ and $a_1 + a_2 + \dots$ are, however, often used to denote also the sequence $(a_n)$ itself. We met infinite series in the first paradox in Lecture 1. Is it true that

$$\sum_{n=1}^{\infty} (-1)^{n+1} = 1 - 1 + 1 - 1 + 1 - 1 + \cdots = 0 + 0 + 0 + \cdots = 0\;?$$

No, this is not true. The first equality holds, it is an equality between two sequences. The third equality holds as well, it says that the sum of all zeros series is zero. But the second equality does not hold: as an equality of two sequences it does not hold and neither it holds as an equality of sums of two series, because the series $1 - 1 + 1 - 1 + \dots$ does not have any sum — the sequence of partial sums $(1, 0, 1, 0, \dots)$ does not have a limit.

# Lecture 4 — More on Series. Limits of Functions. Elementary Functions

## Infinite Series (continued)

Recall that the symbols $\sum a_n = \sum_{n=1}^{\infty} a_n = a_1 + a_2 + \dots$ denote a **series**, i.e., the sequence $(a_n) \subset \mathbb{R}$ whose terms $a_n$ are called **summands**, and also the limit

$$\lim s_n = \lim_{n \to \infty}(a_1 + a_2 + \cdots + a_n) \in \mathbb{R}^*$$

of the sequence $(s_n) = (a_1 + a_2 + \cdots + a_n)$ of **partial sums** $s_n$, which is called the **sum** (of the series). If the sum is finite we say that the series **converges**, else it **diverges**. Convergence and divergence of any series do not depend on any change of only finitely many summands but, in contrast with limits of sequences, the sum may change after the change of a single summand.

We keep the indices in sequences $(a_n)$ to be $n \in \mathbb{N}$, but for series the summation index $n$ often runs through sets different from $\mathbb{N}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1</span><span class="math-callout__name">(Nonnegative Summands)</span></p>

Every series $\sum a_n$ whose summands $a_n \ge 0$ for every $n \ge n_0$, has a sum that differs from $-\infty$.

</div>

A similar proposition holds for series with almost all summands non-positive.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2</span><span class="math-callout__name">(Necessary Condition of Convergence)</span></p>

If the series $\sum a_n$ converges then $\lim a_n = 0$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

If $\sum a_n$ converges then $\lim s_n =: S \in \mathbb{R}$ (here $s_n = \sum_{j=1}^n a_j$). By limits of subsequences and by the arithmetic of limits,

$$\lim a_n = \lim(s_n - s_{n-1}) = \lim s_n - \lim s_{n-1} = S - S = 0. \quad \square$$

</details>
</div>

By this proposition both series $\sum_{n=1}^{\infty} 1 = 1 + 1 + \dots$ and $\sum_{n=0}^{\infty}(-1)^n = 1 - 1 + 1 - 1 + \dots$ diverge. The former has the sum $+\infty$ (see Proposition 1) and the latter (mentioned at the end of Lecture 3) does not have a sum.

### Harmonic Series

In the previous proposition the opposite implication does not hold. We consider the series with the summands

$$a_1 = \frac{1}{2},\; a_2 = a_3 = \frac{1}{4},\; a_4 = a_5 = a_6 = a_7 = \frac{1}{8},\; \dots,\; a_{2^k} = a_{2^k+1} = \cdots = a_{2^{k+1}-1} = \frac{1}{2^{k+1}},\; \dots$$

Clearly, $\lim a_n = 0$, but $s_1 < s_2 < \dots$ and

$$s_{2^{k+1}-1} = \frac{1}{2} + 2 \cdot \frac{1}{4} + 4 \cdot \frac{1}{8} + \cdots + 2^k \cdot \frac{1}{2^{k+1}} = \frac{k+1}{2},$$

so that $\sum a_n = \lim s_n = +\infty$ and the series diverges.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3</span><span class="math-callout__name">(Harmonic Series)</span></p>

The so called **harmonic series**

$$\sum_{n=1}^{\infty} \frac{1}{n} = 1 + \frac{1}{2} + \frac{1}{3} + \frac{1}{4} + \dots$$

diverges and has the sum $+\infty$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $(h_n)$ be the partial sums of the harmonic series and $(s_n)$ be the partial sums of the previous series $\sum a_n$. Then $1/n > a_n$ for every $n$, therefore also $h_n > s_n$ for every $n$. Since $\lim s_n = +\infty$, the one cop theorem gives that $\lim h_n = +\infty$ and the harmonic series has the sum $+\infty$. $\square$

</details>
</div>

Partial sums of the harmonic series are called **harmonic numbers**. We mention without proof two interesting results on them.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4</span><span class="math-callout__name">(On Harmonic Numbers)</span></p>

We consider the harmonic numbers $h_n = \sum_{j=1}^n 1/j$, $n \in \mathbb{N}$.

1. For every $n \in \mathbb{N}$,

   $$h_n = \log n + \gamma + \Delta_n,$$

   where $\gamma = 0.57721\dots$ is the so called **Euler's constant** and the numbers $\Delta_n \in \mathbb{R}$ satisfy $\|\Delta_n\| < c/n$ for a constant $c$ and every $n$.

2. $h_n \in \mathbb{N} \iff n = 1$.

</div>

The conjecture that $\gamma \notin \mathbb{Q}$ is still unproven.

### The Riemann Theorem

At the beginning of the 1st lecture we met in the paradox of infinite sums the series $1 - 1 + \frac{1}{2} - \frac{1}{2} + \frac{1}{3} - \frac{1}{3} + \cdots + \frac{1}{n} - \frac{1}{n} + \dots$ that has an "obvious" sum 0. By changing the order of summands we changed this sum to a positive one. The original sum 0 is correct, though, because the series has partial sums $1, 0, \frac{1}{2}, 0, \frac{1}{3}, 0, \dots$ going in limit to 0.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5</span><span class="math-callout__name">(Riemann's Theorem)</span></p>

Let $\sum_{n=1}^{\infty} a_n$ be a series of the same type, i.e.,

1. $\lim a_n = 0$,
2. $\sum a_{k_n} = +\infty$, where $a_{k_n}$ are positive summands of the series, and
3. $\sum a_{z_n} = -\infty$, where $a_{z_n}$ are negative summands of the series.

Then $\forall\, S \in \mathbb{R}^*$ there is a bijection $\pi\colon \mathbb{N} \to \mathbb{N}$ such that

$$\sum_{n=1}^{\infty} a_{\pi(n)} = S$$

— by changing the order of summands we can get any sum. There is of course also a bijection $\pi$ such that the series $\sum_{n=1}^{\infty} a_{\pi(n)}$ does not have a sum.

</div>

The theorem is named after the German mathematician *Bernhard Riemann (1826–1866)*. He also invented an integral of real functions which we will study in this course later.

### Absolutely Convergent Series

We introduce a class of series whose sums do not change under reordering of summands.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6</span><span class="math-callout__name">(AC Series)</span></p>

A series $\sum a_n$ is **absolutely convergent**, abbreviated **AC**, if the series $\sum \|a_n\|$ converges.

</div>

The class of AC series is the correct generalization of finite sums to infinitely many summands.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7</span><span class="math-callout__name">(On AC Series)</span></p>

Every AC series converges.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $\sum a_n$ be an AC series and $(s_n)$ be its partial sums. We show that $(s_n)$ is a Cauchy sequence. This suffices because by the theorem on Cauchy condition then $(s_n)$ converges. For every two indices $m \le n$ we have that

$$|s_n - s_m| = |a_{m+1} + a_{m+2} + \cdots + a_n| \stackrel{\Delta\text{-ineq.}}{\le} |a_{m+1}| + |a_{m+2}| + \cdots + |a_n| = t_n - t_m,$$

where $(t_n)$ are partial sums of the series $\sum \|a_n\|$. But the sequence $(t_n)$ is Cauchy (by the mentioned theorem) and therefore also $(s_n)$ is Cauchy. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8</span><span class="math-callout__name">(Commutativity of AC Series)</span></p>

If $\sum a_n$ is an AC series, then for every bijection $\pi\colon \mathbb{N} \to \mathbb{N}$ the series $\sum a_{\pi(n)}$ is AC. The sums of the original and reordered series are equal,

$$\sum_{n=1}^{\infty} a_n = \sum_{n=1}^{\infty} a_{\pi(n)}.$$

</div>

### Geometric Series

These are the series

$$\sum_{n=0}^{\infty} q^n = 1 + q + q^2 + \cdots + q^n + \dots$$

with the parameter $q \in \mathbb{R}$ called the **quotient**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9</span><span class="math-callout__name">(On Geometric Series)</span></p>

For $q \le -1$ the geometric series does not have a sum. For $-1 < q < 1$ the geometric series converges and has the sum

$$\sum_{n=0}^{\infty} q^n = \frac{1}{1 - q}.$$

For $q \ge 1$ the geometric series has the sum $+\infty$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

For every $q \in \mathbb{R} \setminus \lbrace 1 \rbrace$ and every $n \in \mathbb{N}$,

$$s_n := 1 + q + q^2 + \cdots + q^{n-1} = \frac{1 - q^n}{1 - q} = \frac{1}{1 - q} + \frac{q^n}{q - 1}.$$

So for $q < -1$ we have by the arithmetic of limits that $\lim s_{2n-1} = +\infty$, $\lim s_{2n} = -\infty$ and therefore $\lim s_n$ does not exist — the geometric series does not have a sum. For $q = -1$ we have similarly that $s_{2n-1} = 1$, $s_{2n} = 0$ and the geometric series again does not have a sum. For $-1 < q < 1$ one has that $\lim q^n = 0$ and by the arithmetic of limit the geometric series has the sum $\lim s_n = \frac{1}{1-q}$. For $q = 1$ one has that $s_n = n$ and the geometric series has the sum $\lim s_n = +\infty$. For $q > 1$, $\lim q^n = +\infty$ and by the arithmetic of limits the geometric series has the sum $\lim s_n = +\infty$. $\square$

</details>
</div>

A quick application of the formula for the sum of geometric series:

$$27.272727\dots = 27(1 + 10^{-2} + 10^{-4} + \dots) = 27 \cdot \frac{1}{1 - 10^{-2}} = \frac{27 \cdot 100}{99} = \frac{300}{11}.$$

It is easy to see that for $q \in (-1, 1)$ and $m \in \mathbb{Z}$ one has the more general formula

$$q^m + q^{m+1} + q^{m+2} + \cdots = \frac{q^m}{1 - q}.$$

It is also clear that every convergent geometric series is absolutely convergent.

### Zeta Function

The **zeta function** $\zeta(s)\colon \mathbb{C} \setminus \lbrace 1 \rbrace \to \mathbb{C}$ is defined by a series. Here we define it only for real $s > 1$. We use real powers $a^b$ for $a > 0$ which will be defined in the second half of this lecture. So for $s \in \mathbb{R}$ we take the series

$$\zeta(s) := \sum_{n=1}^{\infty} \frac{1}{n^s}.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10</span><span class="math-callout__name">(On Zeta Function)</span></p>

For $s \le 1$ the series $\zeta(s)$ has the sum $+\infty$. For $s > 1$ it (absolutely) converges.

</div>

The first claim follows from the divergence of harmonic series. L. Euler derived formulas for all values $\zeta(2n)$ for every $n$, for example $\zeta(2) = \pi^2/6$ and $\zeta(4) = \pi^4/90$. No formula is known for $\zeta(2n-1)$ for any $n \ge 2$. It is known that $\zeta(3) \notin \mathbb{Q}$.

## Limits of Functions

For any $A \in \mathbb{R}^*$ and any $\varepsilon > 0$, recall the $\varepsilon$-neighborhood $U(A, \varepsilon)$ of $A$ and the deleted $\varepsilon$-neighborhood $P(A, \varepsilon) = U(A, \varepsilon) \setminus \lbrace A \rbrace$ of $A$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 11</span><span class="math-callout__name">(Limit Points of a Set)</span></p>

We say that $L \in \mathbb{R}^*$ is a **limit point** of a set $M \subset \mathbb{R}$ if $\forall\, \varepsilon\colon P(L, \varepsilon) \cap M \neq \emptyset$.

</div>

In other words, $L \in \mathbb{R}^*$ is a limit point of a set $M \subset \mathbb{R}$ if and only if there is a sequence $(a_n) \subset M \setminus \lbrace L \rbrace$ with $\lim a_n = L$. Now we generalize the notion of limit from sequences to functions. Recall that for $f\colon A \to B$ and $C \subset A$, $f[C] = \lbrace f(x) \mid x \in C \rbrace \subset B$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 12</span><span class="math-callout__name">(Limits of Functions)</span></p>

Let $A, L \in \mathbb{R}^*$, $M \subset \mathbb{R}$, $A$ be a limit point of $M$ and let $f\colon M \to \mathbb{R}$ be a function. If

$$\forall\, \varepsilon\; \exists\, \delta\colon\; f[P(A, \delta) \cap M] \subset U(L, \varepsilon),$$

we write $\lim_{x \to A} f(x) = L$ and say that the function $f$ has **at $A$ the limit $L$**.

</div>

The limit does not depend on the value $f(A)$ and $f$ need not, and for $A = \pm\infty$ even cannot, be defined at $A$. For a sequence $(a_n) \subset \mathbb{R}$,

$$\lim a_n = \lim_{x \to +\infty} a(x),$$

where on the right-hand side we understand the sequence as a function $a\colon \mathbb{N} \to \mathbb{R}$. When $A$ is not a limit point of $M$ then for some $\delta$ one has that $M \cap P(A, \delta) = \emptyset$. Then $\emptyset = f[P(A, \delta) \cap M] \subset U(L, \varepsilon)$ for *every* $L \in \mathbb{R}^*$ and every $\varepsilon$, which is not good.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 13</span><span class="math-callout__name">(Uniqueness of Limits)</span></p>

Limits of functions are unique: if $M \subset \mathbb{R}$, $f\colon M \to \mathbb{R}$, $K, L, L' \in \mathbb{R}^*$ and $K$ is a limit point of the set $M$, then

$$\lim_{x \to K} f(x) = L \;\land\; \lim_{x \to K} f(x) = L' \;\Rightarrow\; L = L'.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We prove it directly, like for limits of sequences. For every $\varepsilon$ there is a $\delta$ such that the nonempty set $f[P(K, \delta) \cap M]$ is contained in both neighborhoods $U(L, \varepsilon)$ and $U(L', \varepsilon)$. In particular, $\forall\, \varepsilon\colon U(L, \varepsilon) \cap U(L', \varepsilon) \neq \emptyset$. Thus (by the main property of neighborhoods mentioned earlier) $L = L'$. $\square$

</details>
</div>

The next theorem shows how to reduce limits of functions to limits of sequences.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 14</span><span class="math-callout__name">(Heine's Definition)</span></p>

Let $M \subset \mathbb{R}$, $K, L$ be in $\mathbb{R}^*$, $K$ be a limit point of the set $M$ and let $f\colon M \to \mathbb{R}$. Then

$$\lim_{x \to K} f(x) = L \iff \forall\, (a_n) \subset M \setminus \lbrace K \rbrace\colon\; \lim a_n = K \;\Rightarrow\; \lim f(a_n) = L.$$

Thus $L$ is the limit of the function $f$ at $K$ iff for every sequence $(a_n)$ in $M$ that has the limit $K$ but never equals $K$, the values $(f(a_n))$ have the limit $L$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**Implication $\Rightarrow$.** We assume that $\lim_{x \to K} f(x) = L$, that $(a_n) \subset M \setminus \lbrace K \rbrace$ has the limit $K$ and that an $\varepsilon$ is given. Then there exists a $\delta$ such that for every $x \in M \cap P(K, \delta)$ one has that $f(x) \in U(L, \varepsilon)$. For this $\delta$ there is an $n_0$ such that $n \ge n_0 \Rightarrow a_n \in P(K, \delta) \cap M$. Hence $n \ge n_0 \Rightarrow f(a_n) \in U(L, \varepsilon)$ and $f(a_n) \to L$.

**Implication $\lnot \Rightarrow \lnot$.** We assume that $\lim_{x \to K} f(x) = L$ does not hold and deduce from this that the right-hand side of the equivalence does not hold. So there is an $\varepsilon > 0$ such that for every $\delta > 0$ there is a point $b = b(\delta) \in M \cap P(K, \delta)$ such that $f(b) \notin U(L, \varepsilon)$. We set $\delta = \frac{1}{n}$ for $n \in \mathbb{N}$ and for every $n$ *chose* a point $b_n := b(1/n) \in M \cap P(K, 1/n)$ such that $f(b_n) \notin U(L, \varepsilon)$. The sequence $(b_n)$ lies in $M \setminus \lbrace K \rbrace$ and has the limit $K$, but the sequence of values $(f(b_n))$ does not have the limit $L$. The right-hand side of the equivalence therefore does not hold. $\square$

In the proof of the implication $\Leftarrow$ we used the so called **axiom of choice** from set theory.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Limit of a Function)</span></p>

Due to the identity $a^2 - b^2 = (a - b)(a + b)$,

$$
\begin{aligned}
\lim_{x \to +\infty} \left(\sqrt{x + \sqrt{x}} - \sqrt{x}\right) \stackrel{\frac{(\cdots - \sqrt{x})(\cdots + \sqrt{x})}{\cdots + \sqrt{x}}}{=} \lim_{x \to +\infty} \frac{\sqrt{x}}{\sqrt{x + \sqrt{x}} + \sqrt{x}} \stackrel{\frac{\cdots/\sqrt{x}}{\cdots/\sqrt{x}}}{=} \lim_{x \to +\infty} \frac{1}{\sqrt{1 + 1/\sqrt{x}} + 1} = \frac{1}{1 + 1} = \frac{1}{2}
\end{aligned}
$$

</div>

## The Exponential Function

This is the most important elementary function.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 15</span><span class="math-callout__name">(The Exponential)</span></p>

For any $x \in \mathbb{R}$ we set

$$\mathrm{e}^x = \exp(x) := \sum_{n=0}^{\infty} \frac{x^n}{n!} = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \dots \;\colon\; \mathbb{R} \to \mathbb{R}.$$

</div>

This series is AC for every real (even complex) $x$, due to an estimate by geometric series: $\|x/n\| < 1$ whenever $n > \|x\|$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 16</span><span class="math-callout__name">(The Exponential Identity)</span></p>

For every $x, y \in \mathbb{R}$,

$$\exp(x + y) = \exp(x) \cdot \exp(y).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 17</span><span class="math-callout__name">(On the Exponential Function)</span></p>

It holds that

1. $\exp(0) = 1$.
2. $\forall\, x \in \mathbb{R}\colon \exp(x) > 0$ and $\exp(-x) = 1/\exp(x)$.
3. $\exp$ increases: $x < y \Rightarrow \exp(x) < \exp(y)$.
4. $\lim_{x \to -\infty} \exp(x) = 0$.
5. $\lim_{x \to +\infty} \exp(x) = +\infty$.
6. $\exp$ is a bijection from $\mathbb{R}$ to $(0, +\infty)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 18</span><span class="math-callout__name">(The Number $\mathrm{e}$)</span></p>

We define $\mathrm{e} := \exp(1) = 1 + \frac{1}{1!} + \frac{1}{2!} + \frac{1}{3!} + \cdots = 2.71828\dots$ It is called the **Euler number**.

</div>

It is not very hard to show that $\mathrm{e}$ is irrational, $\mathrm{e} \notin \mathbb{Q}$.

## The Logarithm

The **logarithm** $\log x$ is the inverse to the exponential function,

$$\log := \exp^{-1}\colon (0, +\infty) \to \mathbb{R}.$$

Its basic properties derive from those of the exponential function.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 19</span><span class="math-callout__name">(On Logarithm)</span></p>

It holds that

1. $\log(1) = 0$.
2. $\forall\, x, y \in (0, +\infty)\colon \log(xy) = \log x + \log y$.
3. $\log$ increases: $x < y \Rightarrow \log(x) < \log(y)$.
4. $\lim_{x \to 0} \log(x) = -\infty$.
5. $\lim_{x \to +\infty} \log(x) = +\infty$.
6. $\log$ is a bijection from $(0, +\infty)$ to $\mathbb{R}$.

</div>

## The Real Power

Here we introduce only the simplified version with nonnegative $a$. But everybody knows that, for example, $(-2)^3 = (-2) \cdot (-2) \cdot (-2) = -8$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 20</span><span class="math-callout__name">(Real Power)</span></p>

For $a, b \in \mathbb{R}$ with $a > 0$ we set

$$a^b := \exp(b \log a).$$

For every $b > 0$ we set $0^b := 0$.

</div>

For the number $\mathrm{e} = \exp(1)$ and every $x \in \mathbb{R}$ then indeed $\mathrm{e}^x = \exp(x \log(\exp(1))) = \exp(x \cdot 1) = \exp(x)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 21</span><span class="math-callout__name">(Three Power Identities)</span></p>

For any numbers $a, b, x, y \in \mathbb{R}$ with $a, b > 0$,

$$(a \cdot b)^x = a^x \cdot b^x, \quad a^x \cdot a^y = a^{x+y} \quad \text{and} \quad (a^x)^y = a^{x \cdot y}.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

1. $(ab)^x = \exp(x \log(ab)) = \exp(x \log a + x \log b) = \exp(x \log a)\exp(x \log b) = a^x b^x$.
2. $a^x a^y = \exp(x \log a)\exp(y \log a) = \exp(x \log a + y \log a) = \exp((x+y)\log a) = a^{x+y}$.
3. $(a^x)^y = \exp(y \log(\exp(x \log a))) = \exp(yx \log a) = a^{xy}$. $\square$

</details>
</div>

But note that $((-1)^2)^{1/2} = 1^{1/2} = 1 \neq -1 = (-1)^1 = (-1)^{2 \cdot 1/2}$.

The power $0^0$ is problematic because of the following reason.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 22</span><span class="math-callout__name">($0^0$ is Indeterminate)</span></p>

For every number $c \in [0, 1]$ there exist sequences $(a_n)$, $(b_n) \subset (0, +\infty)$ such that

$$\lim a_n = \lim b_n = 0 \quad \text{and} \quad \lim (a_n)^{b_n} = c.$$

Both sequences can be also selected so that $\lim (a_n)^{b_n}$ does not exist.

</div>

## Cosine and Sine

These functions can be defined by infinite series too but their origin lies in geometry.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 23</span><span class="math-callout__name">(Cosine and Sine)</span></p>

For every $t \in \mathbb{R}$ we define the functions

$$\cos t := \sum_{n=0}^{\infty} \frac{(-1)^n t^{2n}}{(2n)!} \quad \text{and} \quad \sin t := \sum_{n=0}^{\infty} \frac{(-1)^n t^{2n+1}}{(2n+1)!},$$

so that $\cos t = 1 - \frac{t^2}{2} + \frac{t^4}{24} - \dots$ and $\sin t = t - \frac{t^3}{6} + \frac{t^5}{120} - \dots$, going from $\mathbb{R}$ to $\mathbb{R}$.

</div>

Again by geometric series estimates we see that both series are AC for every $t \in \mathbb{R}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 24</span><span class="math-callout__name">(On a Runner)</span></p>

Let $t \in \mathbb{R}$ and $S := \lbrace(x, y) \in \mathbb{R}^2 \mid x^2 + y^2 = 1 \rbrace$ be the plane unit circle (i.e., with radius 1) with center in the origin. The runner that runs on the track $S$ with unit speed, starts at the point $(1, 0) \in S$ and runs counter-clockwise for $t > 0$ and clockwise for $t \le 0$, is in the time $\|t\|$ located in the point

$$(\cos t,\, \sin t) \in S.$$

</div>

Thus cosine and sine coincide with the geometrically defined functions bearing the same names.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 25</span><span class="math-callout__name">(The Number $\pi$)</span></p>

We can informally define $\pi = 3.14159\dots$ so that the circumference of $S$, i.e., the time when the runner again runs through the start, equals $2\pi$. The formal definition is that the smallest positive zero of the function $\cos t$ is $\pi/2$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 26</span><span class="math-callout__name">(On Sine and Cosine)</span></p>

It holds that

1. Cosine and sine are $2\pi$-periodic functions, $\cos(t + 2\pi) = \cos t$ and $\sin(t + 2\pi) = \sin t$ for every $t \in \mathbb{R}$.
2. Sine increases on $[0, \pi/2]$ from 0 to 1.
3. $\forall\, t \in [0, \pi]\colon \sin(t) = \sin(\pi - t)$ and $\forall\, t \in [0, 2\pi]\colon \sin(t) = -\sin(2\pi - t)$.
4. $\forall\, t \in [0, 2\pi]\colon \cos t = \sin(t + \pi/2)$.
5. $\forall\, t \in \mathbb{R}\colon \cos^2 t + \sin^2 t = 1$.
6. $\forall\, s, t \in \mathbb{R}\colon \sin(s \pm t) = \sin s \cdot \cos t \pm \cos s \cdot \sin t$ and $\cos(s \pm t) = \cos s \cdot \cos t \mp \sin s \cdot \sin t$.

</div>

Parts 2–4 imply that $\cos, \sin\colon \mathbb{R} \to [-1, 1]$. Part 4 says that the graph of cosine is just the shifted graph of sine.

Further trigonometric functions are the **tangent** $\tan t = \frac{\sin t}{\cos t}$ and the **cotangent** $\cot t = \frac{\cos t}{\sin t}$. The **arcsine** (*inverse sine*) and the **arccosine** (*inverse cosine*) are the inverses of the restriction of sine and cosine to the intervals $[-\pi/2, \pi/2]$ and $[0, \pi]$, respectively. They are the bijections

$$\arcsin\colon [-1, 1] \to [-\pi/2, \pi/2] \quad \text{and} \quad \arccos\colon [-1, 1] \to [0, \pi].$$

Similarly, the **arctangent** and the **arccotangent** are the inverses of the restriction of tangent and cotangent to the intervals $(-\pi/2, \pi/2)$ and $(0, \pi)$, respectively. They are the bijections

$$\arctan\colon \mathbb{R} \to (-\pi/2,\, \pi/2) \quad \text{and} \quad \text{arccot}\colon \mathbb{R} \to (0,\, \pi).$$

# Lecture 5 — Properties of Limits of Functions. Continuity at a Point

## One-Sided Limits of Functions

In contrast with $\mathbb{C}$ or with the spaces $\mathbb{R}^n$ of dimension $n \ge 2$, deletion of one point disconnects the real axis in two separated pieces. So in $\mathbb{R}$ there are exactly two directions to approach in limit the given point, and hence the left-sided and right-sided limits. They only concern finite points, not infinities.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1</span><span class="math-callout__name">(One-Sided Neighborhoods)</span></p>

For $\varepsilon, b \in \mathbb{R}$, the **left**, resp. **right**, $\varepsilon$-neighborhood of the point $b$ is

$$U^-(b, \varepsilon) := (b - \varepsilon,\, b], \quad \text{resp.}\quad U^+(b, \varepsilon) := [b,\, b + \varepsilon).$$

The **left**, resp. **right**, **deleted** $\varepsilon$-neighborhood of $b$ is

$$P^-(b, \varepsilon) := (b - \varepsilon,\, b), \quad \text{resp.}\quad P^+(b, \varepsilon) := (b,\, b + \varepsilon).$$

</div>

So again $P^-(b, \varepsilon) = U^-(b, \varepsilon) \setminus \lbrace b \rbrace$ and $P^+(b, \varepsilon) = U^+(b, \varepsilon) \setminus \lbrace b \rbrace$. By means of these neighborhoods we define one-sided limit points.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2</span><span class="math-callout__name">(One-Sided Limit Points)</span></p>

A point $b \in \mathbb{R}$ is a **left**, resp. **right**, **limit point** of $M \subset \mathbb{R}$ if

$$\forall\, \delta > 0\colon\; P^-(b, \delta) \cap M \neq \emptyset, \quad \text{resp.}\quad \forall\, \delta > 0\colon\; P^+(b, \delta) \cap M \neq \emptyset.$$

</div>

As before $b$ is a left (resp. right) limit point of $M$ iff there is a sequence $(a_n)$ in $(-\infty, b) \cap M$ (resp. in $(b, +\infty) \cap M$) such that $\lim a_n = b$. A left (resp. right) limit point of a set is its limit point. Any limit point of a set is its left or right limit point, but it need not be its left *and* right limit point.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3</span><span class="math-callout__name">(One-Sided Limits)</span></p>

Let $a \in \mathbb{R}$, $L \in \mathbb{R}^*$, $M \subset \mathbb{R}$, $a$ be a left (resp. right) limit point of $M$ and let $f\colon M \to \mathbb{R}$. We write $\lim_{x \to a^-} f(x) = L$, resp. $\lim_{x \to a^+} f(x) = L$, and say that the function $f$ has at the point $a$ the **left-sided**, resp. **right-sided**, **limit** $L$ if

$$\forall\, \varepsilon\; \exists\, \delta\colon\; f[P^-(a, \delta) \cap M] \subset U(L, \varepsilon), \quad \text{resp.}\quad f[P^+(a, \delta) \cap M] \subset U(L, \varepsilon).$$

</div>

It always holds that

$$\lim_{x \to a} f(x) = L \;\Rightarrow\; \lim_{x \to a^\pm} f(x) = L$$

or the one-sided limit of $f$ at $a$ is not defined because $a$ is not the respective left or right limit point of the definition domain. It always holds that

$$\lim_{x \to a^-} f(x) = L \;\land\; \lim_{x \to a^+} f(x) = L \;\Rightarrow\; \lim_{x \to a} f(x) = L.$$

But it may be that $\lim_{x \to a^-} f(x) = L \neq L' = \lim_{x \to a^+} f(x)$. Then $\lim_{x \to a} f(x)$ does not exist. For instance, the function **signum**

$$\operatorname{sgn}(x)\colon \mathbb{R} \to \lbrace -1, 0, 1 \rbrace,$$

defined as $\operatorname{sgn}(x) = -1$ for $x < 0$, $\operatorname{sgn}(0) = 0$ and $\operatorname{sgn}(x) = 1$ for $x > 0$, has at 0 different one-sided limits

$$\lim_{x \to 0^-} \operatorname{sgn}(x) = -1 \quad \text{and} \quad \lim_{x \to 0^+} \operatorname{sgn}(x) = 1.$$

Hence $\lim_{x \to 0} \operatorname{sgn}(x)$ does not exist. Like two-sided limits, one-sided limits are unique and have equivalent Heine definitions.

## Continuity of a Function at a Point

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4</span><span class="math-callout__name">(Continuity at a Point)</span></p>

Let $a \in M \subset \mathbb{R}$ and let $f\colon M \to \mathbb{R}$. The function $f$ is **continuous at the point $a$** if

$$\forall\, \varepsilon\; \exists\, \delta\colon\; f[U(a, \delta) \cap M] \subset U(f(a), \varepsilon).$$

Compared to the limit of $f$ at $a$, the element $L$ is replaced with the value $f(a)$, and $P(a, \delta)$ is replaced with the larger neighborhood $U(a, \delta)$.

</div>

In other words, $f\colon M \to \mathbb{R}$ is continuous at $a \in M$ iff

$$\forall\, \varepsilon\; \exists\, \delta\colon\; x \in M \;\land\; |x - a| < \delta \;\Rightarrow\; |f(x) - f(a)| < \varepsilon.$$

Else we say that $f$ is **discontinuous** at $a$. For example, $\operatorname{sgn}(x)$ is discontinuous at 0, but is continuous at every $x \neq 0$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5</span><span class="math-callout__name">(On Continuity at a Point)</span></p>

Suppose that $b \in M \subset \mathbb{R}$, that $b$ is a limit point of $M$ and that a function $f\colon M \to \mathbb{R}$ is given. The following three claims are mutually equivalent.

1. The function $f$ is continuous at the point $b$.
2. $\lim_{x \to b} f(x) = f(b)$.
3. For every sequence $(a_n) \subset M$ with $\lim a_n = b$ also $\lim f(a_n) = f(b)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**Implication $1 \Rightarrow 2$.** We assume that $f$ is continuous at $b$ according to Definition 4 and that an $\varepsilon$ is given. Thus there is a $\delta$ such that $f[U(b, \delta) \cap M] \subset U(f(b), \varepsilon)$. Thus also $f[P(b, \delta) \cap M] \subset U(f(b), \varepsilon)$ and, by the definition of limit of a function, $\lim_{x \to b} f(x) = f(b)$.

**Implication $2 \Rightarrow 3$.** We assume that $\lim_{x \to b} f(x) = f(b)$ and that a sequence $(a_n) \subset M$ with $\lim a_n = b$ is given, as well as an $\varepsilon$. By the definition of limit of a function there is a $\delta$ such that

$$f[P(b, \delta) \cap M] \subset U(f(b), \varepsilon). \qquad (*)$$

We take an $n_0$ such that $n \ge n_0 \Rightarrow a_n \in U(b, \delta)$. Hence $n \ge n_0 \Rightarrow f(a_n) \in U(f(b), \varepsilon)$: either $a_n \neq b$, and we can use inclusion $(*)$, or $a_n = b$ but then $f(a_n) = f(b) \in U(f(b), \varepsilon)$. Thus $\lim f(a_n) = f(b)$.

**Implication $3 \Rightarrow 1$**, i.e., $\lnot 1 \Rightarrow \lnot 3$. We assume that $f$ is not continuous at $b$ according to Definition 4. Thus there is an $\varepsilon$ such that for every $\delta$ there is an $a = a(\delta) \in U(b, \delta) \cap M$ with $f(a) \notin U(f(b), \varepsilon)$. We *choose* for every $n$ some such $a_n := a(1/n)$ and get the sequence $(a_n) \subset M$ such that $\lim a_n = b$ but $f(a_n) \notin U(f(b), \varepsilon)$ for every $n$ — $(f(a_n))$ does not have the limit $f(b)$. Therefore part 3 does not hold. $\square$

In the proof of the last implication we used again the so called **axiom of choice** of set theory.

</details>
</div>

We consider continuity of a function at a point that is not a limit point of the definition domain.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6</span><span class="math-callout__name">(Isolated Points)</span></p>

A point $b \in M \subset \mathbb{R}$ is an **isolated point** of $M$ if

$$\exists\, \varepsilon\colon\; U(b, \varepsilon) \cap M = \lbrace b \rbrace.$$

</div>

For $b \in M \subset \mathbb{R}$ we see at once that

$$b \text{ is not a limit point of } M \iff b \text{ is an isolated point of } M.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7</span><span class="math-callout__name">(Continuity at an Isolated Point)</span></p>

Let $b \in M \subset \mathbb{R}$, $b$ be an isolated point of $M$ and let $f\colon M \to \mathbb{R}$ be any function. Then $f$ is continuous at $b$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $b$, $M$ and $f$ be as stated. Then for some $\delta$, $U(b, \delta) \cap M = \lbrace b \rbrace$. For this $\delta$ the inclusion

$$f[U(b, \delta) \cap M] = \lbrace f(b) \rbrace \subset U(f(b), \varepsilon)$$

holds for every $\varepsilon$. Hence $f$ is continuous at $b$ according to Definition 4. $\square$

</details>
</div>

So, for example, every sequence $(a_n) \subset \mathbb{R}$ when viewed as a function $a$ from $\mathbb{N}$ to $\mathbb{R}$ is continuous at every point $n \in \mathbb{N} \subset \mathbb{R}$ of its definition domain $\mathbb{N}$.

## One-Sided Continuity

Let $a \in M \subset \mathbb{R}$ and $f\colon M \to \mathbb{R}$. The function $f$ is **left-continuous**, resp. **right-continuous**, at the point $a$ if

$$\forall\, \varepsilon\; \exists\, \delta\colon\; f[U^-(a, \delta) \cap M] \subset U(f(a), \varepsilon), \quad \text{resp.}\quad f[U^+(a, \delta) \cap M] \subset U(f(a), \varepsilon).$$

It is easy to see that

$$f \text{ is cont. at } a \iff f \text{ is left-cont. at } a \;\land\; f \text{ is right-cont. at } a.$$

## The Riemann Function

This function $r\colon \mathbb{R} \to \lbrace 0 \rbrace \cup \lbrace 1/n \mid n \in \mathbb{N} \rbrace$ is defined by

$$r(x) = \begin{cases} 0 & \dots\; x \text{ is an irrational number} \newline \frac{1}{n} & \dots\; x = \frac{m}{n} \in \mathbb{Q} \text{ and } \frac{m}{n} \text{ is in lowest terms.} \end{cases}$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8</span><span class="math-callout__name">(On the Riemann Function $r(x)$)</span></p>

The Riemann function is continuous at $x$ if and only if $x$ is irrational.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $x = \frac{m}{n} \in \mathbb{Q}$, where $\frac{m}{n}$ is in lowest terms, and let $\varepsilon \le \frac{1}{n}$. For every $\delta$ there is an irrational number $\alpha \in U(x, \delta)$. But $r(\alpha) = 0 \notin U(r(x), \varepsilon) = U(\frac{1}{n}, \varepsilon)$, and $r$ is not continuous at the point $x$.

Let $x \in \mathbb{R}$ be irrational and let an $\varepsilon \in (0, 1)$ be given. We define $\delta := \min(M)$ for the set

$$M := \lbrace |x - \tfrac{m}{n}| \mid \tfrac{m}{n} \in \mathbb{Q},\; \tfrac{m}{n} \in U(x, 1),\; 1/n \ge \varepsilon \rbrace.$$

This $\delta > 0$ exists because $M \neq \emptyset$ and $M$ is a finite set of positive numbers. Also $y \in U(x, \delta) \Rightarrow r(y) \in U(r(x), \varepsilon) = U(0, \varepsilon)$ because for every $y \in U(x, \delta)$ one has that $r(y) = 0$ or $r(y) = \frac{1}{n} < \varepsilon$. Therefore $r$ is continuous at the point $x$. $\square$

</details>
</div>

## Limits of Monotonous Functions

Monotonicity of functions is similar to monotonicity of sequences.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 9</span><span class="math-callout__name">(Monotonous Functions)</span></p>

Let $M \subset \mathbb{R}$ and $f\colon M \to \mathbb{R}$. The function $f$

1. is **non-decreasing** (on $M$) if for every $x, y \in M$ one has that $x \le y \Rightarrow f(x) \le f(y)$, and
2. is **non-increasing** (on $M$) if for every $x, y \in M$ one has that $x \le y \Rightarrow f(x) \ge f(y)$.

The function $f$ is **monotonous** (on $M$) if it is non-decreasing or non-increasing.

</div>

Recall when a set of real numbers is bounded from above (or from below) and when it is unbounded from above (or from below). We explain after the proof of the next theorem why it is stated only for one-sided limits and not for two-sided limits.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10</span><span class="math-callout__name">(Limits of Monotonous Functions)</span></p>

Let $M \subset \mathbb{R}$, $a \in \mathbb{R}$ be a left limit point of $M$ and let $f\colon M \to \mathbb{R}$ be a function that is non-decreasing on $P^-(a, \delta) \cap M$ for some $\delta$. Then the left-sided limit of the function $f$ at the point $a$ exists. With $N := f[P^-(a, \delta) \cap M] \subset \mathbb{R}$ we have that

$$\lim_{x \to a^-} f(x) = \begin{cases} +\infty & \dots\; N \text{ is unb. from above} \newline \sup(N) \in \mathbb{R} & \dots\; N \text{ is bounded from above.} \end{cases}$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Suppose that $N$ is unbounded from above and that an $\varepsilon$ is given. Thus there is an $x \in P^-(a, \delta) \cap M$ such that $f(x) > 1/\varepsilon$. Since $f$ is non-decreasing on $P^-(a, \delta) \cap M$, for $\theta := a - x$ it holds that $y \in P^-(a, \theta) \cap M \Rightarrow x < y < a \Rightarrow f(y) \ge f(x) > 1/\varepsilon$. Thus $f[P^-(a, \theta) \cap M] \subset U(+\infty, \varepsilon)$ and $\lim_{x \to a^-} f(x) = +\infty$.

Suppose that $N$ is bounded from above, $s := \sup(N)$ and that an $\varepsilon$ is given. By the definition of $s$ there is an $x \in P^-(a, \delta) \cap M$ such that $s - \varepsilon < f(x) \le s$. Since $f$ is non-decreasing on $P^-(a, \delta) \cap M$, for $\theta := a - x$ it holds that $y \in P^-(a, \theta) \cap M \Rightarrow x < y < a \Rightarrow s - \varepsilon < f(x) \le f(y) \le s$. Hence $f[P^-(a, \theta) \cap M] \subset U(s, \varepsilon)$ and $\lim_{x \to a^-} f(x) = s$. $\square$

</details>
</div>

There are several other obvious variants of the theorem: for locally non-increasing functions and/or infinite limit points and/or right-sided limits. Existence of two-sided limits can be proven by monotonicity by reducing them to one-sided limits.

But monotonicity by itself does not guarantee existence of two-sided limits: consider the function $\operatorname{sgn}(x)\colon \mathbb{R} \to \lbrace -1, 0, 1 \rbrace$ (recall that $\operatorname{sgn}(x) = -1$ for $x < 0$, $\operatorname{sgn}(0) = 0$ and $\operatorname{sgn}(x) = 1$ for $x > 0$). It is monotonous (non-decreasing) on the whole $\mathbb{R}$, but $\lim_{x \to 0} \operatorname{sgn}(x)$ does not exist, $\lim_{x \to 0^-} \operatorname{sgn}(x) = -1$ and $\lim_{x \to 0^+} \operatorname{sgn}(x) = 1$.

## Arithmetic of Limits of Functions

We state the next theorem for two-sided limits and prove it by means of Heine's definition of limits of functions. Fortunately now we need not estimate sums, products and ratios. Such estimates were dealt with in the proof of the theorem on arithmetic of limits of sequences.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 11</span><span class="math-callout__name">(Arithmetic of Limits of Functions)</span></p>

Let $M \subset \mathbb{R}$, $A, K, L \in \mathbb{R}^*$, $A$ be a limit point of $M$ and let the functions $f, g\colon M \to \mathbb{R}$ have limits $\lim_{x \to A} f(x) = K$ and $\lim_{x \to A} g(x) = L$. Then the following hold.

1. $\lim_{x \to A}(f(x) + g(x)) = K + L$ whenever the right-hand side is defined.
2. $\lim_{x \to A} f(x)g(x) = KL$ whenever the right-hand side is defined.
3. $\lim_{x \to A} f(x)/g(x) = K/L$ whenever the right-hand side is defined. Here if $g(x) = 0$, $f(x)/g(x) := 0$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

All proofs of 1–3 are similar and we therefore give in detail only the proof of 3. Let $(a_n) \subset M \setminus \lbrace A \rbrace$ be any sequence with $\lim a_n = A$. By Heine's definition of limits of functions (implication $\Rightarrow$), $\lim f(a_n) = K$ and $\lim g(a_n) = L$. We assume that the right-hand side is defined (so that $L \neq 0$ and $g(a_n) \neq 0$ for every $n \ge n_0$). By the theorem on arithmetic of limits of sequences,

$$\lim \frac{f(a_n)}{g(a_n)} = \frac{\lim f(a_n)}{\lim g(a_n)} = \frac{K}{L}.$$

Since this holds for every sequence $(f(a_n)/g(a_n))$ with $(a_n)$ as above, by Heine's definition of limits of functions (implication $\Leftarrow$) also $\lim_{x \to A} f(x)/g(x) = K/L$. $\square$

</details>
</div>

There are obvious versions of the previous theorem for one-sided limits.

## Limits of Functions and Order

We give functional versions of the theorem on limits and order, and of the theorem on two cops. Recall that for $M, N \subset \mathbb{R}$ the comparison $M < N$ means that for every $a \in M$ and $b \in N$ one has that $a < b$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 12</span><span class="math-callout__name">(Limits of Functions and Order)</span></p>

Let $A, K, L \in \mathbb{R}^*$, $A$ be a limit point of $M \subset \mathbb{R}$ and let the functions $f, g\colon M \to \mathbb{R}$ have limits $\lim_{x \to A} f(x) = K$ and $\lim_{x \to A} g(x) = L$. The following hold.

1. If $K < L$ then there is a $\delta$ such that $f[P(A, \delta) \cap M] < g[P(A, \delta) \cap M]$.
2. If for every $\delta$ there are $x, y \in P(A, \delta) \cap M$ with $f(x) \ge g(y)$, then $K \ge L$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**1.** Since $K < L$, there is an $\varepsilon$ such that $U(K, \varepsilon) < U(L, \varepsilon)$. Then by the assumption on limits of $f$ and $g$ there exists a $\delta$ such that $f[P(A, \delta) \cap M] \subset U(K, \varepsilon)$ and $g[P(A, \delta) \cap M] \subset U(L, \varepsilon)$. Hence

$$f[P(A, \delta) \cap M] < g[P(A, \delta) \cap M].$$

**2.** We already know from the proof of this for sequences that part 2 is a reformulation of part 1. If part 1 is the implication $\varphi \Rightarrow \psi$, then part 2 is $\lnot \psi \Rightarrow \lnot \varphi$. $\square$

</details>
</div>

Recall that for $a, b \in \mathbb{R}$ we denote by $I(a, b)$ the closed real interval with endpoints $a$ and $b$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 13</span><span class="math-callout__name">(Two Functional Cops)</span></p>

Let $A, L \in \mathbb{R}^*$, $A$ be a limit point of $M \subset \mathbb{R}$ and let functions $f, g, h\colon M \to \mathbb{R}$ be given such that $\lim_{x \to A} f(x) = \lim_{x \to A} h(x) = L$ and that there is a $\delta$ such that for any $x \in P(A, \delta) \cap M$, $g(x) \in I(f(x), h(x))$. Then also

$$\lim_{x \to A} g(x) = L.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $A$, $L$, $M$, $f$, $g$ and $h$ be as stated and let an $\varepsilon$ be given. Thus there exists a $\delta$ such that the sets $f[P(A, \delta) \cap M]$ and $h[P(A, \delta) \cap M]$ are contained in $U(L, \varepsilon)$. Therefore and due to the convexity of the neighborhood $U(L, \varepsilon)$, for every $x \in P(A, \delta) \cap M$ one has that $I(f(x), h(x)) \subset U(L, \varepsilon)$. By the assumption one has that $g[P(A, \delta) \cap M] \subset U(L, \varepsilon)$, hence $\lim_{x \to A} g(x) = L$. $\square$

</details>
</div>

## Limits of Composite Functions

Composition of functions has no analogy for sequences. Therefore the next limit theorem on this operation is more interesting than the previous four theorems. After its proof we explain why our formulation is better than some other formulations.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 14</span><span class="math-callout__name">(Limits of Composite Functions)</span></p>

Let $A, K, L \in \mathbb{R}^*$, $M, N \subset \mathbb{R}$, $A$ be a limit point of $M$ and $K$ a limit point of $N$, and let functions

$$g\colon M \to N \quad \text{and} \quad f\colon N \to \mathbb{R}$$

have limits $\lim_{x \to A} g(x) = K$ and $\lim_{x \to K} f(x) = L$. Then the composite function $f(g)\colon M \to \mathbb{R}$ has the limit

$$\lim_{x \to A} f(g)(x) = L$$

if and only if at least one the two conditions below holds.

1. If $K \in N$ (so that $K \in \mathbb{R}$) then $f(K) = L$ (so that $L \in \mathbb{R}$).
2. There is a $\delta$ such that $K \notin g[P(A, \delta) \cap M]$.

If neither 1 nor 2 holds then either $\lim_{x \to A} f(g)(x)$ does not exist or $\lim_{x \to A} f(g)(x) = f(K) \neq L$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let an $\varepsilon$ be given. By the assumption on limits of $f$ and $g$ there is a $\delta$ such that (i) $f[P(K, \delta) \cap N] \subset U(L, \varepsilon)$, and a $\theta$ such that (ii) $g[P(A, \theta) \cap M] \subset U(K, \delta)$.

**Condition 1 holds.** Then inclusion (i) strengthens to $f[U(K, \delta) \cap N] \subset U(L, \varepsilon)$. Therefore in

$$f(g)[P(A, \theta) \cap M] = f[g[P(A, \theta) \cap M]] \subset f[U(K, \delta) \cap N] \subset U(L, \varepsilon)$$

the second inclusion holds and $\lim_{x \to A} f(g)(x) = L$.

**Condition 2 holds.** We take the $\theta$ smaller than the $\delta$ in Condition 2 and strengthen inclusion (ii) to $g[P(A, \theta) \cap M] \subset P(K, \delta)$. Therefore in

$$f(g)[P(A, \theta) \cap M] = f[g[P(A, \theta) \cap M]] \subset f[P(K, \delta) \cap N] \subset U(L, \varepsilon)$$

the first inclusion holds and again $\lim_{x \to A} f(g)(x) = L$.

**Neither condition 1 nor 2 holds.** Then $K \in N$ but $f(K) \neq L$, and for every $n$ there exists an $a_n \in P(A, 1/n) \cap M$ such that $g(a_n) = K$. Then the sequence $(a_n) \subset M \setminus \lbrace A \rbrace$, has the limit $\lim a_n = A$ and

$$\lim f(g)(a_n) = \lim f(g(a_n)) = \lim f(K) = f(K) \neq L.$$

By Heine's definition of limits of functions, either $\lim_{x \to A} f(g)(x)$ does not exist or $\lim_{x \to A} f(g)(x) = f(K) \neq L$. $\square$

</details>
</div>

If $K \notin N$, for example when $K = \pm\infty$, then Condition 1 always holds. Elsewhere Condition 1 is not formulated as an implication as here, but only as the requirement that $f(K) = L$. By our extension of Condition 1 here we have obtained the underlined equivalence. Another advantage of our formulation is that we say what happens if neither of the two conditions holds.

## Asymptotic Symbols $O$, $o$ and $\sim$

These are the most frequently used symbols denoting asymptotic relations between functions. One also uses symbols $\Theta$, $\ll$, $\Omega$ and others.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 15</span><span class="math-callout__name">(Big $O$)</span></p>

Let $M \subset \mathbb{R}$, $f, g\colon M \to \mathbb{R}$ and $N \subset M$. If

$$\exists\, c > 0\; \forall\, x \in N\colon\; |f(x)| \le c \cdot |g(x)|,$$

we write $f(x) = O(g(x))$ ($x \in N$) and say that the function $f$ is **big $O$** of the function $g$ on the set $N$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Big $O$)</span></p>

1. Is $x^2 = O(x^3)$ ($x \in \mathbb{R}$)? No, there is a problem at 0.
2. Is $x^3 = O(x^2)$ ($x \in \mathbb{R}$)? No, there is a problem at infinities.
3. Is $x^3 = O(x^2)$ ($x \in (-20, 20)$)? Yes.
4. Is $\log x = O(x^{1/3})$ ($x \in (0, +\infty)$)? No, there is a problem at 0.
5. Is $\log x = O(x^{1/3})$ ($x \in (1, +\infty)$)? Yes.

</div>

The remaining two asymptotic symbols are defined by means of limits.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 16</span><span class="math-callout__name">(Little $o$ and $\sim$)</span></p>

Let $A \in \mathbb{R}^*$ be a limit point of $M \subset \mathbb{R}$, let $f, g\colon M \to \mathbb{R}$ and let $g \neq 0$ on $P(A, \delta) \cap M$ for some $\delta$.

1. If $\lim_{x \to A} f(x)/g(x) = 0$, we write $f(x) = o(g(x))$ ($x \to A$) and say that the function $f$ is **little $o$** of $g$ when $x$ goes to $A$.
2. If $\lim_{x \to A} f(x)/g(x) = 1$, we write $f(x) \sim g(x)$ ($x \to A$) and say that the function $f$ is **asymptotically equal** to $g$ when $x$ goes to $A$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Little $o$ and $\sim$)</span></p>

1. Is $x^2 = o(x^3)$ ($x \to +\infty$)? Yes.
2. Is $x^3 = o(x^2)$ ($x \to 0$)? Yes.
3. Is $x^2 = o(x^3)$ ($x \to 0$)? No.
4. Is $(x+1)^3 \sim x^3$ ($x \to 1$)? No, the ratio goes to 2.
5. Is $(x+1)^3 \sim x^3$ ($x \to +\infty$)? Yes.
6. Is $e^{-1/x^2} = o(x^{20})$ ($x \to 0$)? No, $e^{-1/x^2}$ goes to 0 faster than any $x^n$.

</div>

# Lecture 6 — Properties of Continuous Functions

## Heine's Definition of Continuity at a Point

From the last lecture we know that continuity of a function $f\colon M \to \mathbb{R}$ at a point $a \in M \subset \mathbb{R}$ means that

$$\forall\, \varepsilon\; \exists\, \delta\colon\; f[U(a, \delta) \cap M] \subset U(f(a), \varepsilon).$$

In this lecture we will refer frequently (9 times, to be precise) to the next result.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1</span><span class="math-callout__name">(Heine's Definition of Continuity)</span></p>

$f\colon M \to \mathbb{R}$ is continuous at a point $a \in M \subset \mathbb{R}$ if and only if

$$\forall\, (a_n) \subset M\colon\; \lim a_n = a \;\Rightarrow\; \lim f(a_n) = f(a).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We proved this equivalence as $1 \iff 3$ in Proposition 5 in Lecture 5 for limit points. If $a \in M$ is an isolated point of $M$ then $f$ is continuous at $a$ by Proposition 7 in Lecture 5. But then $\lim a_n = a$ means that $a_n = a$ for every $n \ge n_0$. Hence $f(a_n) = f(a)$ for every $n \ge n_0$ and $\lim f(a_n) = f(a)$. $\square$

</details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2</span><span class="math-callout__name">(Continuity on a Set)</span></p>

Let $M \subset \mathbb{R}$ and let $f\colon M \to \mathbb{R}$. The function $f$ is **continuous** (on $M$) if $f$ is continuous at every point of $M$.

</div>

## Dense Sets

We introduce the relation of density of a set in another set.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3</span><span class="math-callout__name">(Dense Sets)</span></p>

Let $N \subset M \subset \mathbb{R}$. We say that the set $N$ is **dense in the set $M$** if

$$\forall\, a \in M\; \forall\, \delta\colon\; U(a, \delta) \cap N \neq \emptyset.$$

</div>

Let $N \subset M \subset \mathbb{R}$. Clearly, $N$ is dense in $M$ iff for every point $a \in M$ there is a sequence $(b_n) \subset N$ such that $\lim b_n = a$. For example, the set of fractions $\mathbb{Q}$ is dense in $\mathbb{R}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4</span><span class="math-callout__name">(Density and Continuity)</span></p>

Suppose that $N \subset M \subset \mathbb{R}$, that $N$ is dense in $M$ and that $f, g\colon M \to \mathbb{R}$ are two continuous functions such that $\forall\, x \in N\colon f(x) = g(x)$. Then

$$f = g$$

— the functions $f$ and $g$ coincide.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $y \in M$ be any point and $(a_n) \subset N$ be a sequence with $\lim a_n = y$. Then

$$f(y) = f(\lim a_n) = \lim f(a_n) = \lim g(a_n) = g(\lim a_n) = g(y).$$

Here the 2nd and 4th equality follow from Proposition 1. The 3rd equality follows from the assumption that $f$ and $g$ are equal on $N$. Thus $f = g$ completely. $\square$

</details>
</div>

Recall that if $A \subset B$ and $C$ are sets and $f\colon B \to C$ is a function, its **restriction** to $A$ is the function $f \mid A\colon A \to C$ given by $\forall\, x \in A\colon (f \mid A)(x) := f(x)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5</span><span class="math-callout__name">(H. Blumberg, 1922)</span></p>

For any function $f\colon \mathbb{R} \to \mathbb{R}$ there is a set $M \subset \mathbb{R}$ dense in $\mathbb{R}$ and such that the restriction $f \mid M$ is a continuous function.

</div>

*Henry Blumberg (1886–1950)* was an American mathematician who was born in Lithuania.

## Counting Continuous Functions

For $M \subset \mathbb{R}$ we introduce the notation

$$C(M) := \lbrace f\colon M \to \mathbb{R} \mid f \text{ is continuous} \rbrace.$$

It is the set of all continuous real functions defined on the set $M$. The next theorem is a basic result in set theory.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6</span><span class="math-callout__name">(Cantor–Bernstein)</span></p>

If there exist injections $f\colon X \to Y$ and $g\colon Y \to X$ then there is a bijection $h\colon X \to Y$. The map $h$ can be chosen so that for every $x \in X$ one has that $h(x) = f(x)$ or $h(x) = g^{-1}(x)$.

</div>

How many continuous functions $f\colon \mathbb{R} \to \mathbb{R}$ are there? That many as the real numbers.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 7</span><span class="math-callout__name">(Counting Continuous Functions)</span></p>

There exists a bijection $h\colon \mathbb{R} \to C(\mathbb{R})$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By the previous theorem it suffices to find injections $f\colon \mathbb{R} \to C(\mathbb{R})$ and $g\colon C(\mathbb{R}) \to \mathbb{R}$. The former one is obvious,

$$f(a) := (b \mapsto a),$$

i.e., $f(a)$ is the constant function with the value $a$.

We describe the latter injection $g\colon C(\mathbb{R}) \to \mathbb{R}$. We view the numbers in $\mathbb{R}$ as infinite decimal expansions, for instance $-\pi = -3.141592\dots$ or $2022.00000\dots$. By Proposition 4 every function $j \in C(\mathbb{R})$ is completely determined by the countably many values $j(x)$, $x \in \mathbb{Q}$. Let $r\colon \mathbb{N} \to \mathbb{Q}$ and $s\colon \mathbb{N} \to \mathbb{N} \times \mathbb{N}$ be bijections, for example ($k, l, n \in \mathbb{N}$)

$$s(n) = s(2^{k-1} \cdot (2l - 1)) = (s_1(n),\, s_2(n)) := (k, l).$$

We encode the decimal digits $0, 1, \dots, 9$, the decimal point $.$ and the minus sign $-$ by two decimal digits:

$$c(0) := 00,\; c(1) := 01,\; \dots,\; c(9) := 09,\; c(.) := 10\; \text{and}\; c(-) := 11.$$

The map $g\colon C(\mathbb{R}) \to \mathbb{R}$ has at the function $j \in C(\mathbb{R})$ the value

$$g(j) := 0.a_1 a_2 a_3 \dots a_{2n-1} a_{2n} \dots =: \alpha.$$

The digits $a_n \in \lbrace 0, 1, \dots, 9 \rbrace$ are defined as follows. For $k, l \in \mathbb{N}$ we consider the decimal expansions

$$j(r(k)) =: b(1, k)\, b(2, k) \dots b(l, k) \dots$$

of the values $j(r(k))$ of the function $j$ on the fractions $r(k) \in \mathbb{Q}$, with symbols $b(l, k) \in \lbrace 0, 1, \dots, 9, ., - \rbrace$. Then we set

$$a_{2n-1}\, a_{2n} = c(b(l, k)) := c(b(s_1(n),\, s_2(n))).$$

A short meditation reveals that the map $g$ is injective: the single decimal expansion $\alpha$ stores all values of the function $j$ on all rational numbers. $\square$

</details>
</div>

## Attaining Intermediate Values by Continuous Functions

The image of the function $\operatorname{sgn}(x)$ is $\lbrace -1, 0, 1 \rbrace$, but nothing else between these three points. Images of intervals by continuous functions cannot look like this.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8</span><span class="math-callout__name">(On Intermediate Values)</span></p>

Let $a, b, c \in \mathbb{R}$, $a < b$, $f\colon [a, b] \to \mathbb{R}$ be a continuous function and let $f(a) < c < f(b)$ or $f(a) > c > f(b)$. Then

$$\exists\, d \in (a, b)\colon\; f(d) = c.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We suppose that $f(a) < c < f(b)$, the case that $f(a) > c > f(b)$ is treated similarly. Let

$$A := \lbrace x \in [a, b] \mid f(x) < c \rbrace \quad \text{and} \quad d := \sup(A) \in [a, b].$$

The number $d$ is correctly defined because the set $A$ is nonempty ($a \in A$) and bounded from above (for instance, $b$ is an upper bound). We show that both $f(d) < c$ and $f(d) > c$ lead to contradiction, so that $f(d) = c$.

The continuity of $f$ at $a$ and $b$ implies that $d \in (a, b)$.

Let $f(d) < c$. The continuity of $f$ at $d$ implies that there is a $\delta$ such that $x \in U(d, \delta) \cap [a, b] \Rightarrow f(x) < c$. But then $A$ contains numbers larger than $d$, in contradiction with the fact that $d$ is an upper bound of $A$.

Let $f(d) > c$. In the same vein, there is a $\delta$ such that $x \in U(d, \delta) \cap [a, b] \Rightarrow f(x) > c$. But then every $x \in [a, d)$ sufficiently close to $d$ lies outside of $A$, in contradiction with the fact that $d$ is the smallest upper bound of $A$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 9</span><span class="math-callout__name">(Continuous Image of an Interval)</span></p>

Let $I \subset \mathbb{R}$ be an interval (i.e., a convex set) and $f\colon I \to \mathbb{R}$ be a continuous function. Then

$$f[I] = \lbrace f(x) \mid x \in I \rbrace \subset \mathbb{R}$$

is an interval too.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Theorem 8 shows that the set $f[I]$ is convex. $\square$

</details>
</div>

You may wish to attempt the following corollary of the theorem on intermediate values as an exercise.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 10</span><span class="math-callout__name">(On Climbing)</span></p>

A climber starts climbing a mountain at midnight and reaches the summit exactly after 24 hours, again at midnight. Then the climber descends, again for exactly 24 hours, in the base camp. Prove that there is a time $t_0 \in [0, 24]$ when the climber is in both days in the same altitude.

</div>

We prove the next corollary. Recall that a function $f\colon M \to \mathbb{R}$ is **increasing**, resp. **decreasing** (on $M \subset \mathbb{R}$), if for every $x, y \in M$ one has that $x < y \Rightarrow f(x) < f(y)$, resp. $f(x) > f(y)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 11</span><span class="math-callout__name">(Continuity and Injectivity on an Interval)</span></p>

Suppose that $I \subset \mathbb{R}$ is an interval and that $f\colon I \to \mathbb{R}$ is a continuous injective function. Then $f$ is either increasing or decreasing.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

If $f$ neither increases nor decreases then there exist three numbers $a < b < c$ in $I$ such that $f(a) < f(b) > f(c)$ or $f(a) > f(b) < f(c)$. In the former case every $d$ satisfying $f(a), f(c) < d < f(b)$ is attained, by Theorem 8, as $d = f(x) = f(y)$ for some $x \in (a, b)$ and $y \in (b, c)$. This contradicts the injectivity of $f$. In the latter case we get a very similar contradiction. $\square$

</details>
</div>

## Continuous Functions on Compact Sets

Compact sets play in analysis and elsewhere (e.g., in optimization) an important role.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 12</span><span class="math-callout__name">(Compact Sets)</span></p>

A set $M \subset \mathbb{R}$ is **compact** if every sequence $(a_n) \subset M$ has a convergent subsequence $(a_{m_n})$ with $\lim a_{m_n} \in M$.

</div>

By the Bolzano–Weierstrass theorem and the theorem on limits of sequences and order we know that every interval $[a, b]$ is compact. We characterize compact sets later and now prove on them an important theorem.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 13</span><span class="math-callout__name">(The Min-Max Principle)</span></p>

Let $M \subset \mathbb{R}$ be a nonempty compact set and $f\colon M \to \mathbb{R}$ be a continuous function. Then there exist points $a, b \in M$ such that

$$\forall\, x \in M\colon\; f(a) \le f(x) \le f(b).$$

We say that $f$ attains at $a \in M$ its **minimum** (smallest value) $f(a)$ on $M$ and that $f$ attains at $b \in M$ its **maximum** (largest value) $f(b)$ on $M$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We only prove the existence of the maximum of $f$, the proof for the minimum is very similar. Clearly, $f[M] \neq \emptyset$ and we show that this set is bounded from above. Suppose not, then there is a sequence $(a_n) \subset M$ such that $\lim f(a_n) = +\infty$. By the compactness of $M$ the sequence $(a_n)$ has a convergent subsequence $(a_{m_n})$ with $a := \lim a_{m_n} \in M$. Then $\lim f(a_{m_n}) = +\infty$ too. But this contradicts the fact that by Proposition 1, $\lim f(a_{m_n}) = f(a)$. Thus we can define

$$s := \sup(f[M]) \in \mathbb{R}$$

and by the definition of supremum there is a sequence $(a_n) \subset M$ with $\lim f(a_n) = s$. Due to compactness of $M$ the sequence $(a_n)$ has a convergent subsequence $(a_{m_n})$ with $b := \lim a_{m_n} \in M$. By Proposition 1 one has that $\lim f(a_{m_n}) = f(b) = s$. Since $s = f(b)$ is an upper bound of $f[M]$, we have that $f(b) \ge f(x)$ for every $x \in M$. $\square$

</details>
</div>

For non-compact $M$ the theorem need not hold. For example, the function $f\colon [0, 1) \to \mathbb{R}$, $f(x) = \frac{1}{1-x}$, is continuous but not bounded from above and does not have maximum. The function $f\colon [0, 1) \to \mathbb{R}$, $f(x) = x$, is continuous and bounded from above but still does not have maximum.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 14</span><span class="math-callout__name">(Global and Local Extrema)</span></p>

Let $a \in M \subset \mathbb{R}$ and let $f\colon M \to \mathbb{R}$ be any function. The function $f$ has on $M$ a **global maximum**, resp. a **global minimum**, at $a$ if

$$\forall\, x \in M\colon\; f(x) \le f(a), \quad \text{resp.}\quad f(x) \ge f(a).$$

The function $f$ has on $M$ a **local maximum**, resp. a **local minimum**, at $a$ if

$$\exists\, \delta\; \forall\, x \in U(a, \delta) \cap M\colon\; f(x) \le f(a), \quad \text{resp.}\quad f(x) \ge f(a).$$

When strict inequalities ($<$, resp. $>$) hold for every $x \neq a$, we speak of a **strict** global maximum, etc.

</div>

## Compact Sets in $\mathbb{R}$

We know when a set $M \subset \mathbb{R}$ is **bounded**: $\exists\, c\; \forall\, a \in M\colon \|a\| < c$. It is **closed** if

$$\forall\, (a_n) \subset M\colon\; \lim a_n = a \;\Rightarrow\; a \in M.$$

It is **open** if

$$\forall\, a \in M\; \exists\, \delta\colon\; U(a, \delta) \subset M.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 15</span><span class="math-callout__name">(Closed Sets)</span></p>

A set $M \subset \mathbb{R}$ is closed if and only if the set $\mathbb{R} \setminus M$ is open.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

$\mathbb{R} \setminus M$ is not open iff there is a point $a \in \mathbb{R} \setminus M$ such that for every $\delta$, $U(a, \delta) \cap M \neq \emptyset$. Equivalently (choosing for every $n$ some $a_n \in U(a, 1/n) \cap M$), there is a point $a \in \mathbb{R} \setminus M$ and a sequence $(a_n) \subset M$ such that $\lim a_n = a$. Equivalently, $M$ is not closed. $\square$

</details>
</div>

Using the following structural description of open sets one can relatively easily imagine them. By *open intervals* we mean in it the intervals $(-\infty, a)$, $(a, +\infty)$ and $(a, b)$ for $a < b$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 16</span><span class="math-callout__name">(Structure of Open Sets)</span></p>

A set $M \subset \mathbb{R}$ is open if and only if there is a system of open intervals $\lbrace I_j \mid j \in X \rbrace$ such that the index set $X$ is at most countable, the intervals $I_j$ are mutually disjoint and

$$\bigcup_{j \in X} I_j = M.$$

</div>

Closed sets are complements of open sets and therefore they are unions of "gaps" between the above intervals $I_j$. If $\|X\| = n \in \mathbb{N}_0$, there are at most $n + 1$ gaps. What is hard to imagine is that for countable $X$ the set of gaps may be uncountable. This is the reason that it is harder to imagine closed sets.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 17</span><span class="math-callout__name">(Compact Sets)</span></p>

Let $M \subset \mathbb{R}$. Then $M$ is compact if and only if $M$ is closed and bounded.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $M \subset \mathbb{R}$ be closed and bounded and let $(a_n) \subset M$ be any sequence. Since $(a_n)$ is bounded, by the Bolzano–Weierstrass theorem it has a convergent subsequence $(a_{m_n})$ with $a := \lim a_{m_n} \in \mathbb{R}$. Since $M$ is closed, $a \in M$. Thus $M$ is compact.

Suppose that $M \subset \mathbb{R}$ is not bounded. We construct a sequence $(a_n) \subset M$ such that $\|a_m - a_n\| > 1$ for every two indices $m \neq n$. This property is inherited by every subsequence which therefore cannot be convergent and $M$ is not compact. The first term $a_1 \in M$ is taken arbitrarily. Suppose that $a_1, a_2, \dots, a_n$ have been defined such that $\|a_i - a_j\| > 1$ for every $i, j$ with $1 \le i < j \le n$. Since $M$ is not bounded, there is a point $a_{n+1} \in M$ such that $\|a_{n+1}\| > 1 + \max(\|a_1\|, \dots, \|a_n\|)$. Then $\|a_{n+1} - a_i\| > 1$ for every $i = 1, 2, \dots, n$. In this way we define the whole $(a_n)$.

Suppose that $M \subset \mathbb{R}$ is not closed. Then there is a convergent sequence $(a_n) \subset M$ such that $a := \lim a_{m_n} \in \mathbb{R} \setminus M$. Every subsequence has the same limit $a$, and so it does not have limit in $M$. Thus $M$ is not compact. $\square$

</details>
</div>

## Continuity and Various Operations

We present several operations which produce new continuous functions from old ones. Recall that for two functions $f, g\colon M \to \mathbb{R}$ their **sum**, **product** and **ratio** function is defined as ($x \in M$)

$$(f + g)(x) := f(x) + g(x), \quad (fg)(x) := f(x) \cdot g(x) \quad \text{and} \quad (f/g)(x) := f(x)/g(x),$$

respectively.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 18</span><span class="math-callout__name">(Arithmetic of Continuity)</span></p>

Let $M \subset \mathbb{R}$ and $f, g\colon M \to \mathbb{R}$ be continuous functions. Then the sum and product function

$$f + g,\; fg\colon M \to \mathbb{R}$$

are continuous. If $g \neq 0$ on $M$ then also the ratio function

$$f/g\colon M \to \mathbb{R}$$

is continuous.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

All three proofs are similar and we only prove the part with the ratio function. Let $a \in M$ be any point and $(a_n) \subset M$ be any sequence with $\lim a_n = a$. By Proposition 1 (implication $\Rightarrow$) one has that $\lim f(a_n) = f(a)$ and $\lim g(a_n) = g(a)$. By the theorem on arithmetic of limits of sequences,

$$\lim(f/g)(a_n) = \lim f(a_n)/g(a_n) = \lim f(a_n) / \lim g(a_n) = f(a)/g(a) = (f/g)(a).$$

By Proposition 1 (implication $\Leftarrow$), the function $f/g$ is continuous at the point $a$. $\square$

</details>
</div>

**Rational functions** $r(x)$ are ratios of two polynomials, i.e., functions of the form

$$r(x) := \frac{a_m x^m + \cdots + a_1 x + a_0}{b_n x^n + \cdots + b_1 x + b_0}\colon M \to \mathbb{R},$$

where $a_i, b_i \in \mathbb{R}$, $m, n \in \mathbb{N}_0$ and $a_m b_n \neq 0$; in the numerator we allow also the identically zero polynomial. The definition domain $M$ of this function is the set $M = \mathbb{R} \setminus \lbrace z_1, z_2, \dots, z_k \rbrace$, where $z_i \in \mathbb{R}$ are all real roots of the polynomial in the denominator ($k \in \mathbb{N}_0$ and $k \le n$).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 19</span><span class="math-callout__name">(Continuity of Rational Functions)</span></p>

Every rational function is continuous on its definition domain.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The identical function $f(x) = x$ and the constant functions $f(x) = c$, $c \in \mathbb{R}$, are continuous on $\mathbb{R}$. Starting with them and repeatedly applying the previous proposition we obtain that every rational function is continuous. $\square$

</details>
</div>

All earlier mentioned elementary functions $\exp(x)$, $\log x$, $\cos x$, $\sin x$, $a^x$ ($a \ge 0$), $\arccos x$, $\arcsin x$, $\tan x$, $\arctan x$, $\cot x$ and $\text{arccot}\, x$ are continuous on their definition domains.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 20</span><span class="math-callout__name">(Continuity and Composition)</span></p>

Let $M, N \subset \mathbb{R}$ and let $g\colon M \to N$ and $f\colon N \to \mathbb{R}$ be continuous functions. Then the composite function

$$f(g)\colon M \to \mathbb{R}$$

is continuous.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $a \in M$ be any point and $(a_n) \subset M$ be any sequence with $\lim a_n = a$. By Proposition 1 (implication $\Rightarrow$) one has that $\lim g(a_n) = g(a)$ and also that

$$\lim f(g)(a_n) = \lim f(g(a_n)) = f(g(a)) = f(g)(a).$$

By Proposition 1 (implication $\Leftarrow$), $f(g)$ is continuous at $a$. $\square$

</details>
</div>

We know that every injection $f\colon A \to B$ has the inverse function (or inverse) $f^{-1}\colon f[A] \to A$ that is given by

$$\forall\, y \in f[A]\; \forall\, x \in A\colon\; f^{-1}(y) = x \iff f(x) = y.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 21</span><span class="math-callout__name">(Continuity of Inverses)</span></p>

Let $M \subset \mathbb{R}$ and let $f\colon M \to \mathbb{R}$ be a continuous injective function. Then the inverse $f^{-1}\colon f[M] \to M$ is continuous if *(i)* $M$ is compact or *(ii)* $M$ is an interval.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**(i)** We assume that $M$ is compact, $b \in f[M]$ is any point and that $(b_n) \subset f[M]$ is any sequence with $\lim b_n = b$. We set $a := f^{-1}(b) \in M$ and $a_n := f^{-1}(b_n) \in M$. We show that $\lim a_n = a$, which by Proposition 1 proves the continuity of $f^{-1}$ at $b$. Let $(a_{m_n})$ be any subsequence of the sequence $(a_n) \subset M$ with $\lim a_{m_n} = L \in \mathbb{R}^*$. But $L \in M$ because $M$ is bounded and closed (by Theorem 17). By Proposition 1, $\lim f(a_{m_n}) = f(L) = b$ because $(f(a_{m_n}))$ is a subsequence of the sequence $(b_n)$. Due to the injectivity of $f$, $L = a$. Thus the sequence $(a_n)$ does not have two subsequences with different limits and by part 2 of Proposition 6 in Lecture 2 $(a_n)$ has a limit and we have just proven that this limit is $a$.

**(ii)** Let $M$ be an interval. By Corollary 11 the function $f$ increases or decreases. Suppose that $f$ is decreasing, the increasing case is similar. By Corollary 9 the image $f[M]$ is an interval. Let $b \in f[M]$ and let an $\varepsilon$ be given. We show that $f^{-1}$ is right-continuous at $b$. This is trivial when $b$ is the right endpoint of the interval $f[M]$. Suppose that $b$ is not the right endpoint of this interval. Since $f^{-1}$ is decreasing, $a := f^{-1}(b) \in M$ is not the left endpoint of the interval $M$ and we can assume that $\varepsilon$ is so small that $[a - \varepsilon, a] \subset M$. We set $\delta := f(a - \varepsilon) - f(a) = f(a - \varepsilon) - b$. Since $f^{-1}$ decreases, it maps $[b, b + \delta] \subset f[M]$ to $[a - \varepsilon, a] \subset M$. Hence

$$f^{-1}[U^+(b, \delta) \cap f[M]] \subset U(f^{-1}(b), \varepsilon) = U(a, \varepsilon)$$

and $f^{-1}$ is right-continuous at $b$. The left continuity is proven similarly and we see that $f^{-1}$ is continuous at $b$. $\square$

The theorem also holds for (iii) open $M$ and (iv) closed $M$ if $f$ increases or decreases, but we skip these proofs here. Part (ii) of the theorem implies that $\log x$ and inverse trigonometric functions are continuous.

</details>
</div>

# Lecture 7 — Derivatives of Functions

## Derivatives of Functions

Derivatives are another fundamental notion of mathematical analysis. We encounter them far beyond the borders of analysis, especially in physics, but also in economical, biological, sociological and other models. Often derivative at a point is considered only for interior points. But then one cannot compute the derivative at 0 of the function $d(x) := \frac{\sin(1/x)}{\sin(1/x)}$ for $x \neq 0$, $d(0) := 1$, with the definition domain $M = \mathbb{R} \setminus \lbrace 1/\pi n \mid n \in \mathbb{Z} \setminus \lbrace 0 \rbrace \rbrace \ni 0$ that contains no neighborhood $U(0, \delta)$. And there are problems with differentiating inverse functions. Thus we take a more general road.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1</span><span class="math-callout__name">(Derivatives of Functions)</span></p>

Let $a \in M$ be a limit point of the set $M \subset \mathbb{R}$ and $f = f(x)\colon M \to \mathbb{R}$ be a function. We set

$$f'(a) = \frac{df}{dx}(a) := \lim_{x \to a} \frac{f(x) - f(a)}{x - a} \stackrel{(*)}{=} \lim_{h \to 0} \frac{f(a + h) - f(a)}{h}$$

and say that the limit $f'(a) = \frac{df}{dx}(a) \in \mathbb{R}^*$ is the **derivative** of the function $f$ at the point $a$.

</div>

The equality $(*)$ follows by two applications of the theorem on limits of composite functions (Theorem 14 in Lecture 5) or by a direct argument. For the **finite derivative**, i.e., if $f'(a) \in \mathbb{R}$, we say that $f$ is **differentiable** at $a$. Then we have for $x \in M$ that

$$f(x) = \underbrace{f(a) + f'(a) \cdot (x - a)}_{\text{linear approximation of } f} + \underbrace{o((x - a))}_{\text{its error}} \quad (x \to a).$$

So near $a$ the function $f$ is closely approximated by the above linear function.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2</span><span class="math-callout__name">(One-Sided Derivatives)</span></p>

Suppose that $a \in M$ is a left, resp. right, limit point of $M \subset \mathbb{R}$, and that $f = f(x)\colon M \to \mathbb{R}$ is a function. We set

$$f'_-(a) := \lim_{x \to a^-} \frac{f(x) - f(a)}{x - a} \stackrel{(*)}{=} \lim_{h \to 0^-} \frac{f(a + h) - f(a)}{h},$$

resp.

$$f'_+(a) := \lim_{x \to a^+} \frac{f(x) - f(a)}{x - a} \stackrel{(*)}{=} \lim_{h \to 0^+} \frac{f(a + h) - f(a)}{h},$$

and say that $f'\_-(a) \in \mathbb{R}^\ast$, resp. $f'\_+(a) \in \mathbb{R}^\ast$, is the **left-sided**, resp. **right-sided**, **derivative** of $f$ at the point $a$.

</div>

Derivatives and one-sided derivatives relate as follows. If $f$ has the derivative $f'(a) \in \mathbb{R}^\ast$ then $f$ has at least one one-sided derivative and $f'\_-(a) = f'\_+(a) = f'(a)$ whenever these values are defined. If the one-sided derivatives coincide, $f'\_-(a) = f'\_+(a) = L \in \mathbb{R}^\ast$, then also $f'(a) = L$. If $f'\_-(a) \neq f'\_+(a)$ then $f'(a)$ does not exist.

## Derivatives and Extremes

Consider the function $f\colon [0, 1] \to \mathbb{R}$, $f(x) = x$, and the limit points 0 and 1 of its definition domain $[0, 1]$. Then $f'(0) = 1$ and $f'(1) = 1$. At the same time $f$ has at 0 a global minimum and at 1 a global maximum. But this means that the following theorem **does not hold**: *If a function has nonzero derivative at a limit point of the definition domain then it does not have local extreme at the point.*

There exist lecture notes that successfully "prove" this theorem. In order that we get it correctly below in Theorem 4, we introduce limit points of a special kind.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3</span><span class="math-callout__name">(Two-Sided Limit Points — TLP)</span></p>

A point $a \in M$ is a **two-sided limit point**, abbreviated **TLP**, of the set $M \subset \mathbb{R}$ if

$$\forall\, \delta\colon\; P^-(a, \delta) \cap M \neq \emptyset \neq P^+(a, \delta) \cap M.$$

</div>

So the point $a$ is flanked on both sides by other arbitrarily close points of the set $M$. Every TLP of $M$ is a limit point of $M$, but not the other way around.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4</span><span class="math-callout__name">(Necessary Condition for Extremes)</span></p>

We assume that $b \in M$ is a TLP of $M \subset \mathbb{R}$ and that $f\colon M \to \mathbb{R}$ is a function such that $f'(b) \in \mathbb{R}^*$ exists and is nonzero. Then

$$\forall\, \delta\; \exists\, c, d \in U(b, \delta) \cap M\colon\; f(c) < f(b) < f(d)$$

— the function $f$ has no local extreme at $b$, it has at $b$ neither local minimum nor local maximum.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $b$, $M$ and $f$ be as stated and let a $\delta$ be given. We assume that $f'(b) < 0$, the case with $f'(b) > 0$ is similar. We take $\varepsilon$ small enough so that $U(f'(b), \varepsilon) < \lbrace 0 \rbrace$ (i.e., $y \in U(f'(b), \varepsilon) \Rightarrow y < 0$). By Definition 1 there is a $\theta$ such that

$$x \in P(b, \theta) \cap M \;\Rightarrow\; \frac{f(x) - f(b)}{x - b} \in U(f'(b), \varepsilon).$$

Thus if $x \in P^-(b, \theta) \cap M$ then $f(x) > f(b)$, because $x - b < 0$ and the fraction is negative. And similarly if $x \in P^+(b, \theta) \cap M$ then $f(x) < f(b)$. We may assume that $\theta \le \delta$ and take any

$$c \in P^+(b, \theta) \cap M \quad \text{and} \quad d \in P^-(b, \theta) \cap M.$$

The elements $c$ and $d$ exist due to the fact that $b$ is a TLP of $M$. (Here the above mentioned lecture notes err if $b$ is not a TLP of $M$.) Hence $c, d \in U(b, \delta) \cap M$, $f(c) < f(b)$ and $f(d) > f(b)$. $\square$

</details>
</div>

In other words, a function may have local extremes only in the points that (i) are not TLPs of the definition domain or (ii) are such that the derivative does not exist in them or (iii) are such that the derivative vanishes (i.e., is 0) in them.

## Derivatives and Continuity

Differentiability of a function at a point is stronger property than its continuity at the point.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5</span><span class="math-callout__name">(Derivatives and Continuity)</span></p>

Let $b \in M$ be a limit point of $M \subset \mathbb{R}$ and $f\colon M \to \mathbb{R}$ be a function. If $f'(b) \in \mathbb{R}$ (i.e., the derivative is finite) then $f$ is continuous at $b$. The same holds for both one-sided derivatives and the corresponding one-sided continuity.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By the theorem on arithmetic of limits of functions,

$$\lim_{x \to b} f(x) = \lim_{x \to b} \left(f(b) + (x - b) \cdot \frac{f(x) - f(b)}{x - b}\right) = \lim_{x \to b} f(b) + \lim_{x \to b}(x - b) \cdot \lim_{x \to b} \frac{f(x) - f(b)}{x - b} = f(b) + 0 \cdot f'(b) = f(b)$$

and by Proposition 5 (Lecture 5) the function $f$ is continuous at $b$. The same computation works for each one-sided derivative, limit and continuity. $\square$

</details>
</div>

Clearly, $\operatorname{sgn}'(0) = \lim_{x \to 0} \frac{\operatorname{sgn}(x) - \operatorname{sgn}(0)}{x - 0} = \frac{1}{0^+},\, \frac{-1}{0^-} = +\infty$. Thus existence of an infinite derivative does not imply continuity at the point because $\operatorname{sgn}(x)$ is discontinuous at 0, it is even neither left-continuous nor right-continuous there.

In the second example we compute one-sided derivatives at 0 of $|x|$. They are not equal,

$$(|x|)'_-(0) = \lim_{x \to 0^-} \frac{-x - 0}{x - 0} = -1 \quad \text{and} \quad (|x|)'_+(0) = \lim_{x \to 0^+} \frac{x - 0}{x - 0} = 1,$$

and $(|x|)'(0)$ does not exist. But $|x|$ is continuous at 0. Thus, of course, continuity at a point does not in general imply existence of a derivative.

In the third example we compute the derivative of the square root function $\sqrt{x}\colon [0, +\infty) \to [0, +\infty)$. Let $a > 0$. Then

$$(\sqrt{x})'(a) = \lim_{x \to a} \frac{\sqrt{x} - \sqrt{a}}{x - a} = \lim_{x \to a} \frac{x - a}{(x - a)(\sqrt{x} + \sqrt{a})} = \lim_{x \to a} \frac{1}{\sqrt{x} + \sqrt{a}} = \frac{1}{2\sqrt{a}}.$$

At 0 one has that $(\sqrt{x})'(0) = \lim_{x \to 0} \frac{\sqrt{x} - \sqrt{0}}{x - 0} = \lim_{x \to 0} \frac{1}{\sqrt{x}} = \frac{1}{0} = +\infty$. But $\sqrt{x}$ is continuous at 0. Thus infinite derivative is compatible with continuity at the point.

## Derivatives of Constants and Powers

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6</span><span class="math-callout__name">(Derivatives of $c$ and $x^n$)</span></p>

The following formulas hold.

1. If for $c \in \mathbb{R}$ we denote by $f_c\colon \mathbb{R} \to \lbrace c \rbrace$ the constant function with the value $c$, then for every $a \in \mathbb{R}$ one has that $f'_c(a) = 0$.
2. For every $n \in \mathbb{N}$ and every $a \in \mathbb{R}$, $(x^n)'(a) = na^{n-1}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**1.** Let $a, c \in \mathbb{R}$. Then $f'_c(a) = \lim_{x \to a} \frac{c - c}{x - a} = \lim_{x \to a} 0 = 0$.

**2.** Let $n \in \mathbb{N}$ and $a \in \mathbb{R}$. Then

$$(x^n)'(a) = \lim_{x \to a} \frac{x^n - a^n}{x - a} = \lim_{x \to a} \frac{(x - a)(x^{n-1} + x^{n-2}a + \cdots + a^{n-1})}{x - a} = \lim_{x \to a}(x^{n-1} + x^{n-2}a + \cdots + a^{n-1}) = \underbrace{a^{n-1} + a^{n-1} + \cdots + a^{n-1}}_{n \text{ summands}} = na^{n-1}.$$

The penultimate equality holds due to the arithmetic of limits of functions. $\square$

</details>
</div>

## Geometry of Derivatives: Tangent Lines

For $M \subset \mathbb{R}$ and $f\colon M \to \mathbb{R}$, the **graph** of the function $f$ is the plane set

$$G_f := \lbrace(x, f(x)) \mid x \in M \rbrace \subset \mathbb{R}^2.$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 7</span><span class="math-callout__name">(Tangent — Standard Definition)</span></p>

Let $a \in M \subset \mathbb{R}$, where $a$ is a limit point of $M$, and $f\colon M \to \mathbb{R}$ be a function that is differentiable at $a$. The **tangent** (line) to the graph $G_f$ at the point $(a, f(a)) \in G_f$ is the line $\ell$ given by the equation

$$\ell\colon\; y = f'(a) \cdot (x - a) + f(a).$$

Thus it is the only line with slope $f'(a)$ going through the point $(a, f(a))$.

</div>

We now show how to define tangents **without derivatives**. Let $(a, b), (a', b') \in \mathbb{R}^2$ be two distinct points in the plane. We define the **line** going through them as the set

$$\kappa(a, b, a', b') := \lbrace(a, b) + t \cdot (a' - a, b' - b) \mid t \in \mathbb{R}\rbrace \subset \mathbb{R}^2.$$

For a line, in its representation $(\kappa)$ either always $a = a'$ or always $a \neq a'$. In the former case we speak of a **vertical** line and in the latter case of a **non-vertical** line. A **secant** of the graph $G_f$ is any line $\kappa(x, f(x), x', f(x'))$, $x, x' \in M$, $x \neq x'$. Every secant is non-vertical. For a distinguished point $(a, f(a)) \in G_f$, the **main secants** go through it and through another **secondary** point of $G_f$. Other secants of $G_f$ are **non-main**.

To any pair $(s, b) \in \mathbb{R}^2$ we can associate the set (which is in fact a line)

$$\ell(s, b) := \lbrace(x, sx + b) \mid x \in \mathbb{R}\rbrace \subset \mathbb{R}^2$$

with the **slope** $s$. Every set $\ell(s, b)$ is a non-vertical line, every non-vertical line is of the form $\ell(s, b)$, and for every non-vertical line $\kappa$ there is exactly one pair $(s, b) \in \mathbb{R}^2$ such that $\kappa = \ell(s, b)$. The slope $s$ of a non-vertical line $\kappa(a, b, a', b')$ is $s = \frac{b' - b}{a' - a}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 8</span><span class="math-callout__name">(Limits of Lines)</span></p>

If $\ell$ is a non-vertical line, $(\ell_n)$ is a sequence of non-vertical lines and their $(\ell)$ representations $\ell = \ell(s, b)$ and $\ell_n = \ell(s_n, b_n)$ satisfy that

$$\lim s_n = s \;\land\; \lim b_n = b,$$

we write that $\lim \ell_n = \ell$ and say that the lines $\ell_n$ have the **limit** $\ell$.

</div>

Limits of lines are unique because the $(\ell)$ representations are unique and so are the limits of real sequences.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 9</span><span class="math-callout__name">(Tangent — By Limits)</span></p>

Let $a \in M \subset \mathbb{R}$, where $a$ is a limit point of $M$, $f\colon M \to \mathbb{R}$ be a function and let $\ell$ be a non-vertical line. If for every sequence $(x_n) \subset M \setminus \lbrace a \rbrace$ with $\lim x_n = a$ we have in the sense of Definition 8 that

$$\lim \kappa(a, f(a), x_n, f(x_n)) = \ell,$$

we say that the line $\ell$ is the **tangent** (line) to the graph of $f$ at the point $(a, f(a)) \in G_f$.

</div>

Thus the tangent at $(a, f(a))$ is in this definition the limit of any sequence of main secants of the graph such that the secondary points go in limit to $(a, f(a))$. It is a (rigorous!) definition of tangents avoiding mentioning $f'(a)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10</span><span class="math-callout__name">(Equivalence of Definitions)</span></p>

Let $a \in M$, where $a$ is a limit point of $M \subset \mathbb{R}$, $f\colon M \to \mathbb{R}$ be a function and $\ell$ be a non-vertical line. The next two claims are equivalent.

1. The line $\ell$ is tangent to $G_f$ at $(a, f(a))$ by Definition 9.
2. The function $f$ has the derivative $f'(a) \in \mathbb{R}$ and $\ell = \ell(f'(a),\, f(a) - a \cdot f'(a))$, so that $\ell$ is tangent to $G_f$ at $(a, f(a))$ by Definition 7.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**Implication $1 \Rightarrow 2$.** We assume that $\ell$ is tangent to $G_f$ at $(a, f(a))$ by Definition 9. Let $(x_n) \subset M \setminus \lbrace a \rbrace$ be any sequence with $\lim x_n = a$, let $\kappa_n := \kappa(a, f(a), x_n, f(x_n))$ and let $s_n$ be the slope of the secant $\kappa_n$. By the assumption, $\lim \kappa_n = \ell$ and the formula for slopes gives that

$$\lim \frac{f(x_n) - f(a)}{x_n - a} = \lim s_n = s,$$

where $s$ is the slope of the line $\ell$. By Heine's definition of limits of functions and Definition 1, $f'(a) = s$. We proved above that the tangent $\ell$ goes through $(a, f(a))$, hence $\ell = \ell(s, f(a) - sa) = \ell(f'(a), f(a) - a \cdot f'(a))$.

**Implication $1 \Leftarrow 2$.** We assume that the derivative $f'(a) \in \mathbb{R}$ exists and that the line $\ell$ is given by the stated formula. Let $(x_n) \subset M \setminus \lbrace a \rbrace$ be any sequence with $\lim x_n = a$. By the assumption and Heine's definition of limits of functions,

$$\lim \underbrace{\frac{f(x_n) - f(a)}{x_n - a}}_{s_n} = f'(a).$$

Since the fraction $s_n$ is the slope of the main secant $\kappa_n := \kappa(a, f(a), x_n, f(x_n)) = \ell(s_n, f(a) - s_n a)$, these secants have the limit

$$\lim \kappa_n = \ell(f'(a), f(a) - f'(a) \cdot a) = \ell.$$

By Definition 9 the line $\ell$ is tangent to $G_f$ at $(a, f(a))$. $\square$

</details>
</div>

Below, in Theorem 11, we present the third definition of tangents that does not even mention the point $(a, f(a))$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 11</span><span class="math-callout__name">(Limits of Non-Main Secants)</span></p>

Let $a \in M$, $a$ be a TLP of $M \subset \mathbb{R}$, $f\colon M \setminus \lbrace a \rbrace \to \mathbb{R}$ be a function and $\ell$ be a non-vertical line. The following two claims are equivalent.

1. The function $f$ can be extended by the value $f(a)$ to $f\colon M \to \mathbb{R}$ so that $\ell$ is tangent to $G_f$ at $(a, f(a))$ by Definition 9.
2. For every two sequences $(x_n), (x'_n) \subset M \setminus \lbrace a \rbrace$ satisfying that $\lim x_n = \lim x'_n = a$ and $x_n < a < x'_n$ for every $n$, we have by Definition 8 the limit of lines $\lim \kappa(x_n, f(x_n), x'_n, f(x'_n)) = \ell$.

</div>

So, according to the second part, the tangent is the limit of all those sequences of main secants of the graph, in which the pairs of determining points go in limit to $(a, f_0(a))$ and the points in each pair are separated by $(a, f_0(a))$. Thus we give here a definition of tangent in the non-existent point $(a, f(a))$ of the graph $G_f$!

## Arithmetic of Derivatives

We describe relations between derivatives and arithmetic operations.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 12</span><span class="math-callout__name">(Linearity of Derivatives)</span></p>

Let $a \in M$, where $a$ is a limit point of $M \subset \mathbb{R}$, $f, g\colon M \to \mathbb{R}$ and $\alpha \in \mathbb{R} \setminus \lbrace 0 \rbrace$. Then the equality

$$(\alpha f(x))'(a) = \alpha f'(a)$$

holds whenever one side is defined, and the equality

$$(f(x) + g(x))'(a) = f'(a) + g'(a)$$

holds whenever the right-hand side is defined. The same formulas hold for one-sided derivatives.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By the arithmetic of limits of functions,

$$(\alpha f(x))'(a) = \lim_{x \to a} \frac{\alpha f(x) - \alpha f(a)}{x - a} = \alpha \lim_{x \to a} \frac{f(x) - f(a)}{x - a} = \alpha f'(a).$$

Let $h(x) := f(x) + g(x)$. Then by the same theorem also

$$h'(a) = \lim_{x \to a} \frac{h(x) - h(a)}{x - a} = \lim_{x \to a} \frac{f(x) - f(a)}{x - a} + \lim_{x \to a} \frac{g(x) - g(a)}{x - a} = f'(a) + g'(a)$$

whenever the last expression is defined in the arithmetic of $\mathbb{R}^*$. For one-sided derivatives both computations work without change. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 13</span><span class="math-callout__name">(Leibniz Formula)</span></p>

Let $a \in M$, where $a$ is a limit point of $M \subset \mathbb{R}$, and $f, g\colon M \to \mathbb{R}$ be functions. If $f$ or $g$ is continuous at $a$ then

$$(fg)'(a) = f'(a) \cdot g(a) + f(a) \cdot g'(a)$$

if the right-hand side is defined. The same formula holds for one-sided derivatives and one-sided continuity.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $g$ be continuous at $a$, the other case with $f$ is symmetric. By the assumption and by the arithmetic of limits of functions,

$$(fg)'(a) = \lim_{x \to a} \frac{f(x)g(x) - f(a)g(a)}{x - a} = \lim_{x \to a} \frac{(f(x) - f(a))g(x) + f(a)(g(x) - g(a))}{x - a}$$
$$= \lim_{x \to a} \frac{f(x) - f(a)}{x - a} \lim_{x \to a} g(x) + f(a) \lim_{x \to a} \frac{g(x) - g(a)}{x - a} = f'(a)g(a) + f(a)g'(a).$$

For one-sided derivatives this computation does not change. $\square$

</details>
</div>

The formula bears the name of the German philosopher, mathematician and polymath *Gottfried W. Leibniz (1646–1716)*. Together with I. Newton, Leibniz is considered to be the discoverer of the differential and integral calculus.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 14</span><span class="math-callout__name">(Derivatives of Ratios)</span></p>

Let $a \in M \subset \mathbb{R}$, where $a$ is a limit point of $M$, and $f, g\colon M \to \mathbb{R}$ be functions. If $g(a) \neq 0$ and $g$ is continuous at $a$ then

$$\left(\frac{f}{g}\right)'(a) = \frac{f'(a) \cdot g(a) - f(a) \cdot g'(a)}{g(a)^2},$$

if the right-hand side is defined. The same formula holds for one-sided derivatives and one-sided continuity.

</div>

## Derivatives of Composite and Inverse Functions

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 15</span><span class="math-callout__name">(Derivatives of Composite Functions)</span></p>

Let $a \in M$, where $a$ is a limit point of $M \subset \mathbb{R}$, $g\colon M \to N$ be continuous at $a$, with $g'(a) \in \mathbb{R}^*$ and such that $g(a) \in N$ is a limit point of $N \subset \mathbb{R}$, and let $f\colon N \to \mathbb{R}$ have the derivative $f'(g(a)) \in \mathbb{R}^*$. Then the composite function

$$f(g)\colon M \to \mathbb{R}$$

has the derivative

$$(f(g))'(a) = f'(g(a)) \cdot g'(a)$$

whenever this product is defined, i.e., is neither $0 \cdot (\pm\infty)$ nor $(\pm\infty) \cdot 0$.

</div>

We define that a function $f\colon M \to \mathbb{R}$ **increases**, resp. **decreases**, **at a point** $a \in M \subset \mathbb{R}$ if for some $\delta$ one has that

$$x \in P^-(a, \delta) \cap M,\; x' \in P^+(a, \delta) \cap M \;\Rightarrow\; f(x) < f(a) < f(x'),$$

resp. the opposite inequalities hold.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 16</span><span class="math-callout__name">(Derivatives of Inverse Functions)</span></p>

Let $a \in M$, $a$ be a limit point of $M \subset \mathbb{R}$, $f\colon M \to \mathbb{R}$ be an injective function with the derivative $f'(a) \in \mathbb{R}^*$ and let the inverse function $f^{-1}\colon f[M] \to M$ be continuous at $b := f(a)$. Then the following hold.

1. If $f'(a) \in \mathbb{R} \setminus \lbrace 0 \rbrace$ then $f^{-1}$ has the derivative

   $$(f^{-1})'(b) = \frac{1}{f'(a)} = \frac{1}{f'(f^{-1}(b))}.$$

2. If $f'(a) = 0$ and $f$ increases, resp. decreases, at the point $a$ then $f^{-1}$ has the derivative

   $$(f^{-1})'(b) = +\infty, \quad \text{resp.}\quad (f^{-1})'(b) = -\infty.$$

3. If $f'(a) = \pm\infty$ and $b$ is a limit point of $f[M]$ then $f^{-1}$ has the derivative $(f^{-1})'(b) = 0$.

</div>

## A Table of Derivatives of Elementary Functions

We present formulas for these derivatives.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 17</span><span class="math-callout__name">(A Table of Derivatives)</span></p>

We have the following derivatives.

1. On $\mathbb{R}$: $\exp(x)' = \exp(x)$, $(\sin x)' = \cos x$, $(\cos x)' = -\sin x$, $(\arctan x)' = 1/(1 + x^2)$, $(\text{arccot}\, x)' = -1/(1 + x^2)$, $(x^n)' = nx^{n-1}$ for $n \in \mathbb{N}$ and $c' = 0$ for every $c \in \mathbb{R}$.
2. On $\mathbb{R} \setminus \lbrace 0 \rbrace$: $(x^b)' = bx^{b-1}$ for every negative $b \in \mathbb{Z}$.
3. On $(0, +\infty)$: $(x^b)' = bx^{b-1}$ for every $b \in \mathbb{R} \setminus \mathbb{Z}$ and $(\log x)' = 1/x$.
4. On $\mathbb{R} \setminus \lbrace k\pi + \pi/2 \mid k \in \mathbb{Z} \rbrace$: $(\tan x)' = 1/(\cos x)^2$.
5. On $\mathbb{R} \setminus \lbrace k\pi \mid k \in \mathbb{Z} \rbrace$: $(\cot x)' = -1/(\sin x)^2$.
6. On $(-1, 1)$: $(\arcsin x)' = 1/\sqrt{1 - x^2}$ and $(\arccos x)' = -1/\sqrt{1 - x^2}$.

</div>

Sometimes it is incorrectly claimed that the derivative of a function has to be continuous. This probably arises by confusion with the correct proposition that a function is continuous at any point where it is differentiable (Proposition 5). A derivative *can* be discontinuous — we present an example below (Proposition 18).

# Lecture 8 — Mean Value Theorems and Their Corollaries

## Mean Value Theorems

We present three of them.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1</span><span class="math-callout__name">(Rolle's Theorem)</span></p>

Let $a < b$ be real numbers and $f\colon [a, b] \to \mathbb{R}$ with $f(a) = f(b)$ be a continuous function that has finite or infinite derivative at each point of the interval $(a, b)$. Then

$$\exists\, c \in (a, b)\colon\; f'(c) = 0.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

If $f$ is constant, that is $f(x) = f(a) = f(b)$ for every $x \in [a, b]$, then $f'(x) = 0$ for every $x \in (a, b)$. Let $f$ not be constant and $f(x) > f(a) = f(b)$ for some $x \in (a, b)$, the case with $f(x) < f(a) = f(b)$ is treated similarly. According to the principle of minimum and maximum (in Lecture 6), the function $f$ attains its greatest value in some $c \in [a, b]$. Clearly, $c \in (a, b)$. By the assumption about derivatives and Theorem 4 in Lecture 7, $f'(c) = 0$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2</span><span class="math-callout__name">(Lagrange's Theorem)</span></p>

Let $a < b$ be real numbers and $f\colon [a, b] \to \mathbb{R}$ be a continuous function that has finite or infinite derivative at each point of the interval $(a, b)$. Then

$$\exists\, c \in (a, b)\colon\; f'(c) = \frac{f(b) - f(a)}{b - a}.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Consider the function

$$g(x) := f(x) - (x - a) \cdot \frac{f(b) - f(a)}{b - a}\colon [a, b] \to \mathbb{R}.$$

It satisfies the assumptions of Rolle's theorem, especially $g(a) = g(b) = f(a)$, therefore

$$0 = g'(c) = f'(c) - (f(b) - f(a))/(b - a)$$

for some $c \in (a, b)$ and we are done. $\square$

</details>
</div>

Geometrically, this theorem says that under the given assumptions there is always a tangent to $G_f$ at some point $(c, f(c))$, $c \in (a, b)$, which is parallel to the secant $\kappa(a, f(a), b, f(b))$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3</span><span class="math-callout__name">(Cauchy's Mean Value Theorem)</span></p>

Let $a < b$ and $f, g\colon [a, b] \to \mathbb{R}$ with $g(b) \neq g(a)$ be continuous functions that have derivative at each point of the interval $(a, b)$. Derivatives of the function $f$ may be infinite, but derivatives of the function $g$ have to be finite. Then

$$\exists\, c \in (a, b)\colon\; f'(c) = \frac{f(b) - f(a)}{g(b) - g(a)} \cdot g'(c).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Consider the function

$$h(x) := f(x) - (g(x) - g(a)) \cdot \frac{f(b) - f(a)}{g(b) - g(a)}\colon [a, b] \to \mathbb{R}.$$

It satisfies the assumptions of Rolle's theorem, especially $h(a) = h(b) = f(a)$, therefore

$$0 = h'(c) = f'(c) - g'(c) \cdot (f(b) - f(a))/(g(b) - g(a))$$

for some $c \in (a, b)$ and we are done. $\square$

</details>
</div>

## Derivatives and Monotonicity of Functions

A non-negative (resp. non-positive) derivative means that the original function does not decrease (resp. does not increase). A positive (resp. negative) derivative means that the original function increases (resp. decreases). For any set $M \subset \mathbb{R}$ we denote by $M^0 := \lbrace a \in M \mid \exists\, \delta\colon U(a, \delta) \subset M \rbrace$ its **interior**. The interior of an interval $I$ is the open interval $I^0 \subset I$ obtained from $I$ by omitting the endpoints.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4</span><span class="math-callout__name">(Derivatives and Monotonicity 1)</span></p>

Let $I \subset \mathbb{R}$ be an interval and $f\colon I \to \mathbb{R}$ be a continuous function that has finite or infinite derivative at each point in the interior $I^0$ of $I$. Then the following hold.

1. $f' \ge 0$, resp. $f' \le 0$, on $I^0 \;\Rightarrow\; f$ is non-decreasing, resp. non-increasing, on $I$.
2. $f' > 0$, resp. $f' < 0$, on $I^0 \;\Rightarrow\; f$ is increasing, resp. decreasing, on $I$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $f' < 0$ on $I^0$ and $x < y$ be in $I$. By Theorem 2,

$$\frac{f(y) - f(x)}{y - x} = f'(z) < 0$$

for some $z \in (x, y) \subset I^0$. This inequality and $y - x > 0$ imply that $f(x) > f(y)$ — $f$ decreases on $I$. The other three cases in 1 and 2 are treated similarly. $\square$

</details>
</div>

The proof of the following proposition is similar to the proof of Theorem 4 in Lecture 7 and therefore we omit it.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5</span><span class="math-callout__name">(Derivative and Monotonicity 2)</span></p>

Let $a \in M \subset \mathbb{R}$, $f\colon M \to \mathbb{R}$ be a function and the one-sided derivatives below may be infinite. The following hold.

1. When $a$ is a left limit point of $M$ and $f'_-(a) < 0$, resp. $f'_-(a) > 0$, then there exists a $\delta$ such that $f[P^-(a, \delta) \cap M] > \lbrace f(a) \rbrace$, resp. $< \lbrace f(a) \rbrace$.
2. When $a$ is a right limit point of $M$ and $f'_+(a) < 0$, resp. $f'_+(a) > 0$, then there exists a $\delta$ such that $f[P^+(a, \delta) \cap M] < \lbrace f(a) \rbrace$, resp. $> \lbrace f(a) \rbrace$.

</div>

Last time we calculated that $(|x|)'_-(0) = -1$ and $(|x|)'_+(0) = 1$. Thus, according to the previous proposition, the function $|x|$ has a strict local minimum in 0. Of course, this is clear even without any theory.

## Extending Derivatives by Limits

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6</span><span class="math-callout__name">(Extending Derivatives)</span></p>

Let $a, b \in \mathbb{R}$ with $a < b$, $f\colon [a, b) \to \mathbb{R}$ be a continuous function that has finite derivative on the interval $(a, b)$ and let $\lim_{x \to a} f'(x) =: L \in \mathbb{R}^*$. Then

$$f'_+(a) = L.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $a$, $b$, $f$ and $L$ be as stated, and let an $\varepsilon$ be given. There exists a $\delta \le b - a$ such that $x \in P^+(a, \delta) \Rightarrow f'(x) \in U(L, \varepsilon)$. Let $x \in P^+(a, \delta)$ be arbitrary. According to Theorem 2, there exists a $y \in (a, x) \subset P^+(a, \delta)$ such that

$$\frac{f(x) - f(a)}{x - a} = f'(y) \in U(L, \varepsilon).$$

Thus $f'_+(a) = L$. $\square$

</details>
</div>

A similar proposition holds for left derivatives.

## L'Hospital's Rule

This is a method for calculating limits of ratios of functions $f(x)/g(x)$ leading to indeterminate expressions $0/0$ and $\pm\infty/\pm\infty$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 7</span><span class="math-callout__name">(L'Hospital's Rule)</span></p>

Let $A \in \mathbb{R}$. Let for some $\delta$ functions $f, g\colon P^+(A, \delta) \to \mathbb{R}$ have finite derivatives on $P^+(A, \delta)$, $g' \neq 0$ on $P^+(A, \delta)$, and let

1. $\lim_{x \to A} f(x) = \lim_{x \to A} g(x) = 0$ or
2. $\lim_{x \to A} g(x) = \pm\infty$.

Then

$$\lim_{x \to A} \frac{f(x)}{g(x)} = \lim_{x \to A} \frac{f'(x)}{g'(x)}$$

if the last limit exists. This theorem also holds for left neighborhoods $P^-(A, \delta)$, ordinary neighborhoods $P(A, \delta)$ and for $A = \pm\infty$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**Case 1.** Let $\lim_{x \to A} f(x) = \lim_{x \to A} g(x) = 0$, $\lim_{x \to A} \frac{f'(x)}{g'(x)} =: L \in \mathbb{R}^*$ and $A \in \mathbb{R}$. We define $f(A) = g(A) := 0$. $A$ is a limit point of the definition domain of the fraction $f(x)/g(x)$: it is not possible that $g = 0$ on some $P^+(A, \theta)$, for then also $g' = 0$ on $P^+(A, \theta)$. We set

$$P_0^+(A, \delta) := \lbrace x \in (A, A + \delta) \mid g(x) \neq 0 \rbrace.$$

By Theorem 3, there is a function $c\colon P_0^+(A, \delta) \to P^+(A, \delta)$ such that for every $x \in P_0^+(A, \delta)$,

$$c(x) \in (A, x) \quad \text{and} \quad \frac{f(x)}{g(x)} = \frac{f(x) - f(A)}{g(x) - g(A)} = \frac{f'(c(x))}{g'(c(x))}.$$

Clearly, $\lim_{x \to A} c(x) = A$. Since $A \notin P^+(A, \delta)$, condition 1 in the theorem on limits of composite functions is satisfied. According to this theorem, we get that

$$\lim_{x \to A} \frac{f(x)}{g(x)} = \lim_{x \to A} \frac{f'(c(x))}{g'(c(x))} = \lim_{y \to A} \frac{f'(y)}{g'(y)} = L.$$

The proof for functions defined on $P^-(A, \delta)$ is similar. We reduce $P(A, \delta)$ to two one-sided neighborhoods. Finally, let $A = +\infty$, the case with $A = -\infty$ is treated similarly. By substituting $x := 1/y$ and using the theorem on limits of composite functions we reduce it to the limit at 0 and the definition domain $P^+(0, \delta)$:

$$\lim_{x \to +\infty} \frac{f(x)}{g(x)} = \lim_{y \to 0} \frac{f(1/y)}{g(1/y)}$$

and

$$\lim_{x \to +\infty} \frac{f'(x)}{g'(x)} = \lim_{y \to 0} \frac{f'(1/y)}{g'(1/y)} = \lim_{y \to 0} \frac{f'(1/y) \cdot (-y^{-2})}{g'(1/y) \cdot (-y^{-2})} = \lim_{y \to 0} \frac{(f(1/y))'}{(g(1/y))'},$$

where the first equality holds due to the theorem on limits of composite functions and the last due to the formula for derivatives of composite functions.

**Case 2.** Let $\lim_{x \to A} g(x) = \pm\infty$ and $\lim_{x \to A} \frac{f'(x)}{g'(x)} =: L \in \mathbb{R}^*$. We will prove this case later using integrals. $\square$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(L'Hospital's Rule)</span></p>

$$\lim_{x \to 0} \sqrt{x} \log x = \lim_{x \to 0} \frac{(\log x)'}{(1/\sqrt{x})'} = \lim_{x \to 0} \frac{1/x}{(-1/2)x^{-3/2}} = -2 \lim_{x \to 0} x^{1/2} = 0,$$

and more generally $\lim_{x \to 0} x^c \log x = 0$ for every $c > 0$. Or

$$\lim_{x \to 0} \frac{x^2}{\cos x - 1} = \lim_{x \to 0} \frac{(x^2)'}{(\cos x - 1)'} = \lim_{x \to 0} \frac{2x}{-\sin x} = -2 \lim_{x \to 0} \frac{(x)'}{(\sin x)'} = -2 \lim_{x \to 0} \frac{1}{\cos x} = -2.$$

</div>

## Higher Order Derivatives

Definition domains of functions will now mostly be open sets. Each point of such a set is its TLP.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 8</span><span class="math-callout__name">(Higher Order Derivatives $f^{(n)}(x)$)</span></p>

Let $M \subset \mathbb{R}$ be a nonempty open set and $f = f(x)\colon M \to \mathbb{R}$ be a function. For $n \in \mathbb{N}_0 = \lbrace 0, 1, \dots \rbrace$ we define by induction a finite or infinite sequence of functions $f^{(n)}(x)\colon M \to \mathbb{R}$.

1. At the beginning we set $f^{(0)}(x) := f(x)$.
2. For $n > 0$, when the function $f^{(n-1)}(x)$ is defined and has finite derivative at each point $a \in M$, we define for each $a \in M$ the value of the $n$-th function as $f^{(n)}(a) := (f^{(n-1)}(x))'(a)$.

The function $f^{(n)}$ is called the **order $n$ derivative** of the function $f$ or the **$n$-th derivative** of $f$.

</div>

So the function $f^{(0)}$ is $f$ itself and $f^{(1)}$ is its derivative $f'$. If $f^{(n-1)}\colon M \to \mathbb{R}$ is defined and has derivative at a point $b \in M$, finite or infinite, we still write $f^{(n)}(b) := (f^{(n-1)}(x))'(b) \in \mathbb{R}^*$ and call it the $n$-th derivative of the function $f$ at the point $b$. The function $f^{(2)}$, the second derivative of $f$, is also denoted as $f''$. For example, for $M = \mathbb{R}$, $(x \sin x)'' = (\sin x + x \cos x)' = 2\cos x - x \sin x$. Second derivatives can be used to justify existence of extremes of functions.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 9</span><span class="math-callout__name">($f''$ and Extremes)</span></p>

Suppose that $a \in M$, that $M \subset \mathbb{R}$ is an open set, and that $f\colon M \to \mathbb{R}$ is a function with finite $f'\colon M \to \mathbb{R}$, $f'(a) = 0$ and $f''(a) \in \mathbb{R}^*$, possibly infinite. Then the following hold.

1. $f''(a) > 0 \;\Rightarrow\; f$ has at $a$ a strict local minimum.
2. $f''(a) < 0 \;\Rightarrow\; f$ has at $a$ a strict local maximum.

It is clear that the set $M$ can be taken in the form $U(a, \delta)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We prove only part 1, for part 2 the argument is similar. So let $M = U(a, \delta)$, on $U(a, \delta)$ there exists finite $f'$, $f'(a) = 0$ and $f''(a) > 0$. By Proposition 5 there exists a $\theta \le \delta$ such that $f' < f'(0) = 0$ on $P^-(a, \theta)$ and $f' > f'(0) = 0$ on $P^+(a, \theta)$. Let $x \in P^-(a, \theta)$ be arbitrary. By Theorem 2 there exists a $y \in (x, a) \subset P^-(a, \theta)$ such that

$$\frac{f(a) - f(x)}{a - x} = f'(y) < 0.$$

Because the denominator is positive, the numerator is negative and $f(a) < f(x)$. For $x \in P^+(a, \theta)$ one has that $f'(y) > 0$ and the denominator is negative, so the numerator is again negative and again $f(a) < f(x)$. Thus $f$ has at $a$ a strict local minimum. $\square$

</details>
</div>

The proposition says nothing about the case $f''(a) = 0$. This case can be partially resolved by generalizing the proposition to derivatives with orders $> 2$.

## Convexity and Concavity of Functions

Let $B := (c, d) \in \mathbb{R}^2$ be a point in the plane and $\ell$ be a non-vertical line, given by the equation $y = sx + b$. If the inequality $d \ge sc + b$, resp. $d > sc + b$, holds, we write $B \ge \ell$, resp. $B > \ell$, and say that $B$ **lies above** $\ell$, resp. that $B$ **lies strictly above** $\ell$. By reversing the inequalities we define that $B$ **lies below** $\ell$, resp. that $B$ **lies strictly below** $\ell$, symbolically $B \le \ell$, resp. $B < \ell$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 10</span><span class="math-callout__name">(Convex and Concave Functions)</span></p>

Let $f\colon I \to \mathbb{R}$ be a function defined on an interval $I \subset \mathbb{R}$. The function $f$ is **convex** (on $I$) if for every three numbers $a < b < c$ in $I$ the "inequality"

$$(b, f(b)) \le \kappa(a, f(a), c, f(c))$$

holds. If this "inequality" is strict, $f$ is **strictly convex** (on $I$). If the opposite "inequalities" hold, we call the function $f$ **concave**, resp. **strictly concave**, (on $I$).

</div>

Recall that $\kappa(a, f(a), c, f(c))$ is the secant of the graph $G_f$ going through the points $(a, f(a))$ and $(c, f(c))$. A typical example of a strictly convex function is $f(x) = x^2\colon \mathbb{R} \to \mathbb{R}$. The function $f(x) = -x^2\colon \mathbb{R} \to \mathbb{R}$ is then strictly concave. In general, $f\colon I \to \mathbb{R}$ is (strictly) convex $\iff$ $-f$ is (strictly) concave. (Strict) convexity, resp. (strict) concavity, is preserved when the function is restricted to a subinterval.

We present without proof the interesting fact that convexity and concavity imply continuity, in fact even one-sided differentiability.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 11</span><span class="math-callout__name">(Existence of One-Sided Derivatives)</span></p>

Every convex, resp. concave, function $f\colon I \to \mathbb{R}$ that is defined on an open interval $I \subset \mathbb{R}$ has finite one-sided derivatives

$$f'_-,\; f'_+\colon I \to \mathbb{R}.$$

They are non-decreasing, resp. non-increasing.

</div>

By Proposition 5 in Lecture 7, such function $f$ is left- and right-continuous at each point in $I$ and is therefore continuous on $I$. However, the (two-sided) derivative $f'$ may not exist at some points, as the convex function $|x|$ shows.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 12</span><span class="math-callout__name">($f''$, Convexity and Concavity)</span></p>

Let $I \subset \mathbb{R}$ be an interval and let $f\colon I \to \mathbb{R}$ be a continuous function that has at each point $b \in I^0$ possibly infinite second derivative $f''(b) \in \mathbb{R}^*$. Then the following hold.

1. $f'' \ge 0$, resp. $f'' \le 0$, on $I^0 \;\Rightarrow\; f$ is convex, resp. concave, on $I$.
2. $f'' > 0$, resp. $f'' < 0$, on $I^0 \;\Rightarrow\; f$ is strictly convex, resp. strictly concave, on $I$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

To prove this theorem, we need the following geometric lemma (left as an exercise). It says that if we go from left to right and append to a non-vertical straight segment $(a, a')(b, b')$ another non-vertical straight segment $(b, b')(c, c')$ with the same or greater slope, then the common point $(b, b')$ lies below the line going through the extreme points $(a, a')$ and $(c, c')$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 13</span><span class="math-callout__name">(On Slopes)</span></p>

Let $(a, a')$, $(b, b')$ and $(c, c')$ be in $\mathbb{R}^2$ and $a < b < c$. Then

$$\frac{b' - a'}{b - a} \le \frac{c' - b'}{c - b} \;\Rightarrow\; (b, b') \le \kappa(a, a', c, c').$$

Furthermore, strict inequality implies strict "inequality" and both of these implications hold with opposite inequalities and "inequalities".

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 12</summary>

The assumption on the existence of $f''$ means that there exists finite $f'\colon I^0 \to \mathbb{R}$. We still have to assume the continuity of $f$ so that it holds also in endpoints of $I$. Let $f'' \ge 0$ on $I^0$, the other three cases in 1 and 2 are treated similarly. Let $a < b < c$ be any three numbers in $I$. By Theorem 2 there exist a $y \in (a, b)$ and a $z \in (b, c)$ such that

$$s := \frac{f(b) - f(a)}{b - a} = f'(y) \quad \text{and} \quad t := \frac{f(c) - f(b)}{c - b} = f'(z).$$

By Theorem 4, $f'$ is non-decreasing on $I^0$ because $f''$ is non-negative. As $y < z$, the slope $s = f'(y)$ of the straight segment $(a, f(a))(b, f(b))$ is at most the slope $t = f'(z)$ of the straight segment $(b, f(b))(c, f(c))$. According to Lemma 13, the point $(b, f(b))$ lies below the line $\kappa(a, f(a), c, f(c))$. Thus the condition in Definition 10 holds and $f$ is convex on $I$. $\square$

</details>
</div>

## Inflection Points

They can be defined in various ways, but for us they are the points of the graph where it passes from one side of the tangent to the other.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 14</span><span class="math-callout__name">(Inflection Point)</span></p>

Let $a \in M \subset \mathbb{R}$, where $a$ is a TLP of the set $M$, $f\colon M \to \mathbb{R}$ and $\ell$ be tangent to $G_f$ at $(a, f(a))$. The point $(a, f(a))$ is called the **inflection point** of the graph of $f$, if there is a $\delta$ such that for every $x \in P^-(a, \delta) \cap M$ and every $x' \in P^+(a, \delta) \cap M$,

$$(x, f(x)) \le \ell \;\text{ and }\; (x', f(x')) \ge \ell,$$

or the reversed "inequalities" always hold.

</div>

For example, the point $(0, 0)$ is the inflection point of the graph of the function $f(x) = x^3\colon \mathbb{R} \to \mathbb{R}$, because in it $G_f$ goes from the lower to the upper side of the tangent $y = 0$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 15</span><span class="math-callout__name">(No Inflection)</span></p>

Let $f\colon U(a, \delta) \to \mathbb{R}$ and $\exists\, f''(a) \in \mathbb{R}^*$, but it is not zero. Then $(a, f(a))$ is not an inflection point of the graph of $f$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The assumption on $f''$ means that (after possibly decreasing $\delta$) there exists finite $f'\colon U(a, \delta) \to \mathbb{R}$. Let $f''(a) > 0$, the case with $f''(a) < 0$ is treated similarly. Let $\ell$ be tangent to $G_f$ at $(a, f(a))$, so that it has the slope $f'(a)$ and passes through the point $(a, f(a))$. By Proposition 5 there exists a $\theta \le \delta$ such that for every $x \in P^-(a, \theta)$ and every $x' \in P^+(a, \theta)$,

$$f'(x) < f'(a) \quad \text{and} \quad f'(x') > f'(a). \qquad (1)$$

Let $x \in P^-(a, \theta)$ and $x' \in P^+(a, \theta)$ be arbitrary and let $s$ and $t$ be the slopes of the secants $\kappa(x, f(x), a, f(a))$ and $\kappa(a, f(a), x', f(x'))$ of $G_f$, respectively. Due to the inequalities (1) and the mean value Theorem 2 we can easily see that $s < f'(a) < t$. Hence

$$(x, f(x)) > \ell \quad \text{and} \quad (x', f(x')) > \ell$$

and the condition in Definition 14 is not met. $\square$

</details>
</div>

We give without proof a sufficient condition for inflection.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 16</span><span class="math-callout__name">(Inflection Exists)</span></p>

Let $f\colon U(a, \delta) \to \mathbb{R}$, for every $b \in U(a, \delta)$ there exists finite $f''(b)$, $f''(a) = 0$ and $f'' \ge 0$ on $P^-(a, \delta)$ and $f'' \le 0$ on $P^+(a, \delta)$ or opposite inequalities hold. Then

$(a, f(a))$ is an inflection point of the graph of $f$.

</div>

## Asymptotes of Functions

An asymptote of a function is a line, possibly vertical, to which the graph of the function gets in infinity arbitrarily close.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 17</span><span class="math-callout__name">(Vertical Asymptotes)</span></p>

Let $M \subset \mathbb{R}$, $b \in \mathbb{R}$ be a left limit point of $M$ and $f\colon M \to \mathbb{R}$. If

$$\lim_{x \to b^-} f(x) = \pm\infty,$$

we call the line $x = b$ the **left vertical asymptote** of $f$. Right vertical asymptotes are defined similarly.

</div>

For example, the line $x = 0$ is both the left and right vertical asymptote of $f(x) = 1/x\colon \mathbb{R} \setminus \lbrace 0 \rbrace \to \mathbb{R}$. It is also the right vertical asymptote of $f(x) = \log x\colon (0, +\infty) \to \mathbb{R}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 18</span><span class="math-callout__name">(Asymptotes in Infinity)</span></p>

Let $M \subset \mathbb{R}$, $+\infty$ be a limit point of $M$, $s, b \in \mathbb{R}$ and $f\colon M \to \mathbb{R}$. If

$$\lim_{x \to +\infty}(f(x) - sx - b) = 0,$$

we call the line $y = sx + b$ the **asymptote of the function $f$ in $+\infty$**. Asymptotes in $-\infty$ are defined similarly.

</div>

Obviously, $y = sx + b$ is the asymptote of a function $f$ in $+\infty$ iff $\lim_{x \to +\infty} f(x)/x = s$ and $\lim_{x \to +\infty}(f(x) - sx) = b$. Similarly for asymptotes in $-\infty$. For example, $y = 0 = 0x + 0$ is the asymptote of the function $f(x) = 1/x$ both in $+\infty$ and in $-\infty$.

## Graphing Functions

To graph a function $f$ (i.e., to make a picture of $G_f$), usually given by a formula, we first determine its definition domain, the set $M \subset \mathbb{R}$ maximal to inclusion such that $f\colon M \to \mathbb{R}$. Almost always it is a union of at most countably many intervals. We determine whether $f$ is of a special form (even, odd, periodic, ...). We determine where $f$ is continuous and where $f'$ exists. We find one-sided limits at the points of discontinuity of $f$ and at the limit points of $M$ lying outside $M$. We calculate one-sided derivatives, then Proposition 6 helps. Using Theorem 4 we determine maximum intervals of monotonicity. We find local and global extremes. We determine intersections of $G_f$ with the coordinate axes and the image $f[M]$.

We determine where $f''$ exists and, using Theorem 12, determine maximum intervals of convexity and concavity. Using Proposition 15 and Theorem 16 we find inflection points of the graph. We determine asymptotes of the function $f$ and draw its graph by hand or computer.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1</span><span class="math-callout__name">(Graphing $\tan x$)</span></p>

Let $f(x) := \tan x = \frac{\sin x}{\cos x}$. The definition domain is $M = \bigcup_{n \in \mathbb{Z}} (\pi n - \pi/2,\, \pi n + \pi/2)$. It is a $\pi$-periodic function, $\sin(\pi + x) = -\sin x$ and $\cos(\pi + x) = -\cos x$. Due to the continuity of sine and cosine and due to the arithmetic of continuity, $f$ is continuous on $M$. For $b(n) := \pi n + \frac{\pi}{2}$, $n \in \mathbb{Z}$, one has the limits $\lim_{x \to b(n)^-} f(x) = +\infty$ and $\lim_{x \to b(n)^+} f(x) = -\infty$ — each line $x = b(n)$ is both the left and right vertical asymptote of $f$. There are no asymptotes in $-\infty$ and $+\infty$, nor the limits $\lim_{x \to \pm\infty} f(x)$ exist. Because $f'(x) = 1/\cos^2 x > 0$ on $M$, $f$ increases on each interval $(b(n) - \pi, b(n))$. Because of this and the periodicity, $f$ has no extremes. $G_f$ intersects the $y$-axis only in the origin $(0, 0)$ and the $x$-axis exactly in the points $(b(n) - \frac{\pi}{2}, 0) = (\pi n, 0)$, $n \in \mathbb{Z}$. By the above infinite limits and continuity of $f$ (attaining intermediate values) we see that $f[M] = f[(b(n) - \pi, b(n))] = \mathbb{R}$.

The second derivative is $f''(x) = \frac{2\sin x}{\cos^3 x}\colon M \to \mathbb{R}$. We have that $f''(x) = 0 \iff x = b(n) - \frac{\pi}{2}$, that $f'' < 0$ on $(b(n) - \pi, b(n) - \frac{\pi}{2})$ and that $f'' > 0$ on $(b(n) - \frac{\pi}{2}, b(n))$. Thus $f$ is strictly concave on $(b(n) - \pi, b(n) - \frac{\pi}{2}]$, strictly convex on $[b(n) - \frac{\pi}{2}, b(n))$ and the inflection points are exactly $(b(n) - \frac{\pi}{2}, 0) = (\pi n, 0)$, $n \in \mathbb{Z}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2</span><span class="math-callout__name">(Graphing $\arcsin(2x/(1+x^2))$)</span></p>

Let $f(x) := \arcsin\bigl(2x/(1+x^2)\bigr)$. The definition domain is $M = \mathbb{R}$ because the definition domain of arcsin is $[-1, 1]$ and $2|x| \le 1 + x^2$ for every $x \in \mathbb{R}$ ($x^2 \pm 2x + 1 = (x \pm 1)^2 \ge 0$). This function is odd, i.e., $f(-x) = -f(x)$, because $\sin x$, $\arcsin x$ and $\frac{2x}{1+x^2}$ are odd. According to the theorems on continuity of inverse functions, of rational functions and of composite functions, $f$ is continuous on $M$. Clearly, $\lim_{x \to -\infty} f(x) = \lim_{x \to +\infty} f(x) = \arcsin(0) = 0$, because $\frac{2x}{1+x^2} \to 0$ for $x \to \pm\infty$, and so $y = 0 = 0x + 0$ is the asymptote of $f$ both in $-\infty$ and $+\infty$. There are no vertical asymptotes.

The derivative on $\lbrace x \in \mathbb{R} \mid \frac{2x}{1+x^2} \neq \pm 1 \rbrace = \mathbb{R} \setminus \lbrace -1, 1 \rbrace$ is

$$f'(x) = \frac{2 \cdot \operatorname{sgn}(1 - x^2)}{1 + x^2}.$$

Obviously, $\lim_{x \to 1^\pm} f'(x) = \mp 1$. By Proposition 6 we have that $f'_\pm(1) = \mp 1$. Since $f$ is odd, $f'_\pm(-1) = \pm 1$. Because $f' < 0$ on $(-\infty, -1)$, $f' > 0$ on $(-1, 1)$ and $f' < 0$ on $(1, +\infty)$, by Proposition 4 the function $f$ decreases on $(-\infty, -1]$, increases on $[-1, 1]$ and decreases on $[1, +\infty)$. Also $f(x) < 0$ for $x < 0$ and $f(x) > 0$ for $x > 0$ (and $f(0) = 0$). According to these intervals of monotonicity and signs and according to the above zero limits, we see that $f$ has at $x = -1$ the strict global minimum with the value $f(-1) = -\pi/2$, that at $x = 1$ it has symmetrically ($f$ is odd) the strict global maximum with the value $f(1) = \pi/2$ and that $f$ has no other local extrema. It follows that $G_f$ intersects both coordinate axes only in $(0, 0)$ and that $f[M] = f[\mathbb{R}] = [-\pi/2, \pi/2]$.

The second derivative is on $\mathbb{R} \setminus \lbrace -1, 1 \rbrace$ equal to $f''(x) = \frac{-4x \cdot \operatorname{sgn}(1 - x^2)}{(1 + x^2)^2}$. Because $f'' < 0$ on $(-\infty, -1)$, $f'' > 0$ on $(-1, 0)$, $f'' < 0$ on $(0, 1)$, $f'' > 0$ on $(1, +\infty)$ and $f''(x) = 0 \iff x = 0$ (the second derivatives $f''(\pm 1)$ do not exist), by Theorem 12, Proposition 15 and Theorem 16, $f$ is strictly concave on $(-\infty, -1]$, strictly convex on $[-1, 0]$, strictly concave on $[0, 1]$, strictly convex on $[1, +\infty)$ and $(0, 0)$ is the only inflection point (at the points $(-1, f(-1))$ and $(1, f(1))$ tangents do not exist).

</div>

# Lecture 9 — Taylor Polynomials and Series. Primitives

## Taylor Polynomials

In Lecture 7, after defining derivative, we learned that differentiability of a function $f\colon M \to \mathbb{R}$ at a limit point $a \in M \subset \mathbb{R}$ of $M$ provides the linear approximation

$$f(x) = f(a) + f'(a) \cdot (x - a) + o(x - a) \quad (x \to a).$$

In the following theorem, which is also a definition, we use higher-order derivatives to strengthen the approximation by means of polynomials.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 1</span><span class="math-callout__name">(Taylor Polynomial)</span></p>

Let $n \in \mathbb{N}_0$ and let $f\colon U(b, \delta) \to \mathbb{R}$ be a function with finite $f^{(n)}(b) \in \mathbb{R}$. For $n = 0$, this means that $f$ is continuous at $b$. Then there is exactly one polynomial

$$p(x) := \sum_{j=0}^{n} a_j (x - b)^j, \quad a_j \in \mathbb{R}, \quad \text{s.t.} \quad \lim_{x \to b} \frac{f(x) - p(x)}{(x - b)^n} = 0.$$

Its coefficients are given by the formula $a_j = f^{(j)}(b)/j!$. We call it the **Taylor polynomial** of the function $f$ of order $n$ centered at $b$ and denote it as $T_n^{f,b}(x)$.

</div>

$T_n^{f,b}(x)$ equals

$$f(b) + f'(b) \cdot (x - b) + \frac{f''(b)}{2} \cdot (x - b)^2 + \cdots + \frac{f^{(n)}(b)}{n!} \cdot (x - b)^n$$

and the linear approximation above is $T_1^{f,a}(x)$. Also, $T_0^{f,b}(x) = f(b)$ and for every $n \in \mathbb{N}$ we have the identity

$$\bigl(T_n^{f,b}(x)\bigr)' = T_{n-1}^{f',b}(x).$$

To prove Theorem 1 we need the following lemma.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 2</span><span class="math-callout__name">(On the Zero Polynomial)</span></p>

For any numbers $b \in \mathbb{R}$ and $n \in \mathbb{N}_0$ and any polynomial $p(x) = \sum_{j=0}^{n} a_j x^n$ with $a_j \in \mathbb{R}$ one has the implication

$$\lim_{x \to b} \frac{p(x)}{(x - b)^n} = 0 \;\Rightarrow\; \forall\, j = 0, 1, \dots, n\colon\; a_j = 0.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Theorem 1</summary>

The assumption on $f^{(n)}(b)$ means that (after possibly decreasing $\delta$) for every $j = 0, 1, \dots, n-1$ there exists $f^{(j)}\colon U(b, \delta) \to \mathbb{R}$. First we prove that for $p(x) = T_n^{f,b}(x)$ the limit (0) holds. For $n = 0$ this follows from the continuity of $f$ at $b$. For $n = 1$ we have by the theorem on arithmetic of limits of functions that the limit

$$\lim_{x \to b} \frac{f(x) - \overbrace{(f(b) + f'(b) \cdot (x-b))}^{T_1^{f,b}(x)}}{x - b} = \lim_{x \to b} \frac{f(x) - f(b)}{x - b} - \lim_{x \to b} f'(b)$$

indeed equals $f'(b) - f'(b) = 0$. For $n \ge 2$ we get by l'Hospital's rule, the identity above and induction on $n$ that

$$\lim_{x \to b} \frac{f(x) - T_n^{f,b}(x)}{(x - b)^n} = \lim_{x \to b} \frac{(f(x) - T_n^{f,b}(x))'}{((x - b)^n)'} = \frac{1}{n} \lim_{x \to b} \frac{f'(x) - T_{n-1}^{f',b}(x)}{(x - b)^{n-1}} = \frac{1}{n} \cdot 0 = 0.$$

Let $p(x) = \sum_{j=0}^{n} b_j x^j$ with $b_j \in \mathbb{R}$ be any polynomial for which the limit (0) holds. Then

$$\lim_{x \to b} \frac{p(x) - T_n^{f,b}(x)}{(x - b)^n} = \lim_{x \to b} \frac{p(x) - f(x)}{(x - b)^n} + \lim_{x \to b} \frac{f(x) - T_n^{f,b}(x)}{(x - b)^n} = 0 + 0 = 0.$$

Thus, according to Lemma 2, $p(x) = T_n^{f,b}(x)$. $\square$

</details>
</div>

We state concisely the strengthened approximation.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 3</span><span class="math-callout__name">(Taylor Approximation)</span></p>

If $n \in \mathbb{N}_0$ and $f\colon U(b, \delta) \to \mathbb{R}$ is a function with finite $f^{(n)}(b) \in \mathbb{R}$ (i.e., $f$ is continuous at $b$ for $n = 0$), then for $x \in U(b, \delta)$ and $x \to b$,

$$f(x) = T_n^{f,b}(x) + o((x - b)^n) = \sum_{j=0}^{n} \frac{f^{(j)}(b)}{j!}(x - b)^j + \underbrace{o((x - b)^n)}_{e(x)}.$$

The notation $o(\dots)$ means that $\lim_{x \to b} e(x)/(x - b)^n = 0$.

</div>

## Taylor Polynomials of Elementary Functions

We present several Taylor polynomials centered at 0. We justify these formulas, calculate a few limits with them, and discuss when the extension of Taylor polynomials of $f$ to an infinite series converges to $f(x)$. In the following formulas $n \in \mathbb{N}_0$ is arbitrary.

1. $f(x) = \exp x$ has TP $T_n^{f,0}(x) = \sum_{j=0}^{n} x^j/j!$.
2. $f(x) = \sin x$ has TP $T_{2n+1}^{f,0}(x) = \sum_{j=0}^{n} (-1)^j x^{2j+1}/(2j+1)!$.
3. $f(x) = \cos x$ has TP $T_{2n}^{f,0}(x) = \sum_{j=0}^{n} (-1)^j x^{2j}/(2j)!$.
4. For $\forall\, a \in \mathbb{R}$, $f(x) = (1+x)^a$ has TP $T_n^{f,0}(x) = \sum_{j=0}^{n} \binom{a}{j} x^j$, where $\binom{a}{j} = a(a-1)(a-2)\cdots(a-j+1)/j!$ is the **generalized binomial coefficient** with $\binom{a}{0} := 1$.
5. $f(x) = \log(1+x)$ has TP $T_n^{f,0}(x) = \sum_{j=1}^{n} (-1)^{j+1} x^j/j$ for $n > 0$ and $T_0^{f,0}(x) = 0$.
6. $f(x) = \log\bigl(\frac{1}{1-x}\bigr)$ has TP $T_n^{f,0}(x) = \sum_{j=1}^{n} x^j/j$ for $n > 0$ and $T_0^{f,0}(x) = 0$.
7. $f(x) = \arctan x$ has TP $T_{2n+1}^{f,0}(x) = \sum_{j=0}^{n} (-1)^j x^{2j+1}/(2j+1)$.
8. $f(x) = \arcsin x$ has TP $T_{2n+1}^{f,0}(x) = \sum_{j=0}^{n} \binom{j-1/2}{j} x^{2j+1}/(2j+1)$.
9. $f(x) = \arccos x$ has TP $T_{2n+1}^{f,0}(x) = \pi/2 - \sum_{j=0}^{n} \binom{j-1/2}{j} x^{2j+1}/(2j+1)$.

The TP of $\arctan x$ is obtained from the TP of its derivative $\arctan'(x) = \frac{1}{1+x^2}$. We get this TP from (partial sums of) the geometric series $\frac{1}{1+x^2} = 1 - x^2 + x^4 - \dots$, $x \in (-1, 1)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4</span><span class="math-callout__name">(TP of $f$ from TP of $f'$)</span></p>

We suppose that $f\colon U(0, \delta) \to \mathbb{R}$ has finite $f'\colon U(0, \delta) \to \mathbb{R}$ and finite $f^{(n+1)}(0) \in \mathbb{R}$, $n \in \mathbb{N}_0$. Then for $x \to 0$,

$$f'(x) = \sum_{j=0}^{n} a_j x^j + o(x^n), \quad a_j \in \mathbb{R} \;\Rightarrow\; f(x) = f(0) + \sum_{j=0}^{n} \frac{a_j}{j+1} \cdot x^{j+1} + o(x^{n+1}).$$

</div>

### Computing Limits by Taylor Polynomials

We will use Corollary 3. Using $T_1^{f,0}$ in Formula 2, we immediately see that

$$\lim_{x \to 0} \frac{\sin x}{x} = \lim_{x \to 0} \frac{x + o(x)}{x} = \lim_{x \to 0} \frac{x}{x} + \lim_{x \to 0} \frac{o(x)}{x} = 1 + 0 = 1.$$

Or, using $T_2^{f,0}$ in Formula 3,

$$\lim_{x \to 0} \frac{x^4}{(\cos x - 1)^2} = \lim_{x \to 0} \frac{x^4}{(1 - x^2/2 + o(x^2) - 1)^2} = \lim_{x \to 0} \frac{x^4}{x^4/4 + o(x^4)} = \lim_{x \to 0} \frac{1}{1/4 + o(x^4)/x^4} = 4.$$

## Taylor Series

Taylor series of a function arises from its Taylor polynomials by extending them to infinity.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5</span><span class="math-callout__name">(Taylor Series)</span></p>

Let $f\colon U(a, \delta) \to \mathbb{R}$ have finite $f^{(n)}\colon U(a, \delta) \to \mathbb{R}$ for every $n \in \mathbb{N}_0$. If for every $x \in U(a, \delta)$,

$$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!} \cdot (x - a)^n,$$

we say that the function $f$ is on $U(a, \delta)$ the sum of its **Taylor series** $\sum_{n=0}^{\infty} f^{(n)}(a) \cdot (x - a)^n / n!$ centered at $a$.

</div>

Hence Taylor polynomials are partial sums of Taylor series. The following theorem shows when the situation of the previous definition occurs. For $n \in \mathbb{N}_0$ and a function $f\colon U(a, \delta) \to \mathbb{R}$ with finite $f^{(n+1)}\colon U(a, \delta) \to \mathbb{R}$, we define the **remainder of the Taylor polynomial** $T_n^{f,a}(x)$ as

$$R_n^{f,a}(x) := f(x) - T_n^{f,a}(x), \quad x \in U(a, \delta).$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6</span><span class="math-callout__name">(Remainders of Taylor Polynomials)</span></p>

Suppose that $n \in \mathbb{N}_0$ and that $f\colon U(a, \delta) \to \mathbb{R}$ with finite $f^{(n+1)}\colon U(a, \delta) \to \mathbb{R}$. Then the following hold.

1. *(Lagrange's remainder)* $\forall\, x \in P(a, \delta)$ $\exists\, c$ between $a$ and $x$ such that

   $$R_n^{f,a}(x) = \frac{f^{(n+1)}(c)}{(n+1)!} \cdot (x - a)^{n+1}.$$

2. *(Cauchy's remainder)* $\forall\, x \in P(a, \delta)$ $\exists\, c$ between $a$ and $x$ such that

   $$R_n^{f,a}(x) = \frac{f^{(n+1)}(c) \cdot (x - c)^n}{n!} \cdot (x - a).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We prove more generally that for any $g\colon U(a, \delta) \to \mathbb{R}$ with finite and nonzero $g'\colon U(a, \delta) \to \mathbb{R}$ and any $x \in P(a, \delta)$ there is a number $c$ between $a$ and $x$ such that

$$R_n^{f,a}(x) = \frac{1}{n!} \cdot \frac{g(x) - g(a)}{g'(c)} \cdot f^{(n+1)}(c) \cdot (x - c)^n. \qquad (\mathrm{R})$$

Then Lagrange's remainder arises for $g(t) := (x - t)^{n+1}$ and Cauchy's for $g(t) := t$.

Let $x \in P(a, \delta)$ and the function $g$ be as stated. Consider the auxiliary function

$$F(t) := f(x) - \sum_{i=0}^{n} \frac{f^{(i+1)}(t)}{i!} \cdot (x - t)^i.$$

We apply to $F$, $g$ and the interval $I$ with endpoints $a$ and $x$ Cauchy's mean value theorem. On this interval $F$ is continuous, $F(x) = 0$, $F(a) = f(x) - T_n^{f,a}(x)$, $g(a) \neq g(x)$ (due to Lagrange's mean value theorem) and on $I$,

$$F'(t) = -f'(t) - \sum_{i=1}^{n} \left(\frac{f^{(i+1)}(t)}{i!} \cdot (x - t)^i - \frac{f^{(i)}(t)}{i!} \cdot i(x - t)^{i-1}\right) = -\frac{f^{(n+1)}(t)}{n!} \cdot (x - t)^n.$$

By Cauchy's mean value theorem (equality $(*)$) there exists a number $c \in I^0$ such that

$$-\frac{f(x) - T_n^{f,a}(x)}{g(x) - g(a)} = \frac{F(x) - F(a)}{g(x) - g(a)} \stackrel{(*)}{=} \frac{F'(c)}{g'(c)} = -\frac{f^{(n+1)}(c) \cdot (x - c)^n}{n! \cdot g'(c)}.$$

Now the relation (R) follows by a simple rearrangement. $\square$

</details>
</div>

For all nine formulas for TP above we state for which $x \in \mathbb{R}$ they give Taylor series of $f$ centered at 0 and converging to $f(x)$. We omit the proofs, they follow easily from the previous theorem.

1. $\forall\, x \in \mathbb{R}$, $\mathrm{e}^x = \sum_{n \ge 0} x^n/n!$.
2. $\forall\, x \in \mathbb{R}$, $\sin x = \sum_{n \ge 0} (-1)^n x^{2n+1}/(2n+1)!$.
3. $\forall\, x \in \mathbb{R}$, $\cos x = \sum_{n \ge 0} (-1)^n x^{2n}/(2n)!$.
4. $\forall\, x \in (-1, 1)$ and $\forall\, a \in \mathbb{R}$, $(1+x)^a = \sum_{n \ge 0} \binom{a}{n} x^n$.
5. $\forall\, x \in (-1, 1)$, $\log(1+x) = \sum_{n \ge 1} (-1)^{n+1} x^n/n$.
6. $\forall\, x \in (-1, 1)$, $\log\bigl(\frac{1}{1-x}\bigr) = \sum_{n \ge 1} x^n/n$.
7. $\forall\, x \in (-1, 1)$, $\arctan x = \sum_{n \ge 0} (-1)^n x^{2n+1}/(2n+1)$.
8. $\forall\, x \in (-1, 1)$, $\arcsin x = \sum_{n \ge 0} \binom{n-1/2}{n} x^{2n+1}/(2n+1)$.
9. $\forall\, x \in (-1, 1)$, $\arccos x = \frac{\pi}{2} - \sum_{n \ge 0} \binom{n-1/2}{n} x^{2n+1}/(2n+1)$.

Some of these expansions hold in larger domains. Expansion 4 with $a \in \mathbb{N}_0$ holds $\forall\, x \in \mathbb{R}$, expansion 5 holds also for $x = -1$, expansion 6 also for $x = 1$ and expansions 7 also for $x = 1$ and expansions 8 and 9 also for $x = -1$.

Coefficients in Taylor series can often be interpreted combinatorially. We give without proof one example of many.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7</span><span class="math-callout__name">(Bell Numbers $B_n$)</span></p>

For any $x \in (-1, 1)$ it is true that

$$\mathrm{e}^{\mathrm{e}^x - 1} = \exp(\exp(x) - 1) = \sum_{n=0}^{\infty} \frac{B_n}{n!} x^n$$

where $B_n$ is the number of partitions of an $n$-element set.

</div>

For example, $B_3 = 5$ because of the five partitions $\lbrace\lbrace 1, 2, 3 \rbrace\rbrace$, $\lbrace\lbrace 1, 2 \rbrace, \lbrace 3 \rbrace\rbrace$, $\lbrace\lbrace 1, 3 \rbrace, \lbrace 2 \rbrace\rbrace$, $\lbrace\lbrace 1 \rbrace, \lbrace 2, 3 \rbrace\rbrace$ and $\lbrace\lbrace 1 \rbrace, \lbrace 2 \rbrace, \lbrace 3 \rbrace\rbrace$ of the set $\lbrace 1, 2, 3 \rbrace$.

## Primitive Functions

An interval $I \subset \mathbb{R}$ is **non-trivial** if $I \neq \emptyset, \lbrace a \rbrace$ for every $a \in \mathbb{R}$. Non-trivial are exactly those non-empty intervals, each point of which is their limit point.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 8</span><span class="math-callout__name">(Primitive Functions)</span></p>

For any functions $F, f\colon I \to \mathbb{R}$ defined on a non-trivial interval $I \subset \mathbb{R}$, we say that $F$ is a **primitive** (function) of $f$, and write $F = \int f$, if $F$ has finite derivative on $I$ and

$$\forall\, b \in I\colon\; F'(b) = f(b).$$

Sometimes $F$ is also called an **antiderivative** of $f$.

</div>

We emphasize that for every $b \in I$, including endpoints, here $F'(b)$ always means the ordinary, two-sided derivative. It follows from the earlier result on derivatives that every primitive function is continuous. For example, $ax^2/2 + bx + c$ is a primitive of the linear function $ax + b$ on any nontrivial interval, $\mathrm{e}^x$ is on $\mathbb{R}$ a primitive of itself, $c + \arcsin x$ is on $(-1, 1)$ an antiderivative of the function $1/\sqrt{1 - x^2}$ and $2x^{3/2}/3$ is a PF of $\sqrt{x}$ on $[0, +\infty)$.

Antiderivative of a given function is not determined uniquely, but every two of them differ only by a constant shift.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9</span><span class="math-callout__name">(Non-Uniqueness of PF)</span></p>

$F_1, F_2, f\colon I \to \mathbb{R}$ are functions defined on a nontrivial interval $I \subset \mathbb{R}$ and both $F_1$ and $F_2$ are primitives of $f$. Then there is a $c \in \mathbb{R}$ such that

$$F_1 - F_2 = c \;\text{ on } I.$$

Conversely, if $F$ is a primitive of $f$ then for every $c \in \mathbb{R}$ also $F + c$ is a primitive of $f$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $F_1$, $F_2$, $f$ and $I$ be as stated, and $a < b$ be any two numbers from $I$. By Lagrange's mean value theorem, used for the function $F_1 - F_2$ and the interval $[a, b]$, there exists a $c \in (a, b)$ such that

$$\frac{(F_1 - F_2)(b) - (F_1 - F_2)(a)}{b - a} = (F_1 - F_2)'(c) = F_1'(c) - F_2'(c) = f(c) - f(c) = 0.$$

So $F_1(b) - F_2(b) = F_1(a) - F_2(a)$ and $F_1(x) - F_2(x) = c$ for some constant $c$ and every $x \in I$.

The second claim is clear, $(F + c)' = F' + c' = f + 0 = f$. $\square$

</details>
</div>

In the rest of the lecture we prove that every continuous function has an antiderivative. We have to prepare for it some tools.

## Exchange of Limits and Derivatives

We prove a theorem describing situations when one can swap limit for $n \to \infty$ and differentiation, without changing the result.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 10</span><span class="math-callout__name">(Pointwise Convergence $f_n \to f$)</span></p>

$M \subset \mathbb{R}$ is a set and $f, f_n\colon M \to \mathbb{R}$ for $n \in \mathbb{N}$ are functions. When

$$\forall\, \varepsilon\; \forall\, x \in M\; \exists\, n_0\colon\; n \ge n_0 \;\Rightarrow\; |f_n(x) - f(x)| < \varepsilon,$$

we write $f_n \to f$ (on $M$) and say that the functions $f_n$ **converge on $M$ pointwisely** to $f$.

</div>

Thus for every $x \in M$, $\lim f_n(x) = f(x)$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 11</span><span class="math-callout__name">(Uniform Convergence $f_n \rightrightarrows f$)</span></p>

$M \subset \mathbb{R}$ is a set and $f, f_n\colon M \to \mathbb{R}$ for $n \in \mathbb{N}$ are functions. When

$$\forall\, \varepsilon\; \exists\, n_0\; \forall\, x \in M\colon\; n \ge n_0 \;\Rightarrow\; |f_n(x) - f(x)| < \varepsilon,$$

we write $f_n \rightrightarrows f$ (on $M$) and say that the functions $f_n$ **converge on $M$ uniformly** to $f$.

</div>

Now one requires more: single $n_0$ works for every $x \in M$. Clearly, $f_n \rightrightarrows f$ implies that $f_n \to f$, but the converse in general does not hold.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 12</span><span class="math-callout__name">(Swapping Limits — Moore–Osgood)</span></p>

Let $f_n, f\colon M \to \mathbb{R}$, where $n \in \mathbb{N}$ and $M \subset \mathbb{R}$, let $f_n \rightrightarrows f$ (on $M$), $A \in \mathbb{R}^*$ be a limit point of $M$ and let $\lim_{x \to A} f_n(x) =: a_n \in \mathbb{R}$ for every $n$. Then the following finite limits exist and are equal:

$$\lim a_n = \lim_{x \to A} f(x), \quad \text{i.e.,}\quad \lim_{n \to \infty} \lim_{x \to A} f_n(x) = \lim_{x \to A} \lim_{n \to \infty} f_n(x).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

From $f_n \rightrightarrows f$ (on $M$) it follows that $(f_n(x)) \subset \mathbb{R}$ is uniformly Cauchy for $x \in M$, that is, for every $\varepsilon$ there is an $n_0$ such that for every $x \in M$ and every $m, n \ge n_0$, $|f_m(x) - f_n(x)| < \varepsilon$. Then for every two fixed indices $m, n \ge n_0$ the limit transition $\lim_{x \to A}$ gives the inequality $|a_m - a_n| \le \varepsilon$. Thus $(a_n) \subset \mathbb{R}$ is a Cauchy sequence and has a finite limit $\lim a_n =: a \in \mathbb{R}$. The next estimate holds for every $n \in \mathbb{N}$ and every $x \in M$:

$$|f(x) - a| \le \underbrace{|f(x) - f_n(x)|}_{V_1} + \underbrace{|f_n(x) - a_n|}_{V_2} + \underbrace{|a_n - a|}_{V_3}.$$

Let an $\varepsilon$ be given. Because $\lim a_n = a$, there exists an $n_0$ such that $n \ge n_0 \Rightarrow V_3 < \varepsilon/3$. Because $f_n \rightrightarrows f$ (on $M$), there exists an $n_1$ such that $n \ge n_1 \Rightarrow V_1 < \varepsilon/3$ for every $x \in M$. Let $m \ge \max(n_0, n_1)$. Since $\lim_{x \to A} f_m(x) = a_m$, we can take a $\delta$ such that $V_2 < \varepsilon/3$ for $n := m$ and every $x \in P(A, \delta) \cap M$. Thus for $n := m$ and every $x \in P(A, \delta) \cap M$,

$$|f(x) - a| \le \varepsilon/3 + \varepsilon/3 + \varepsilon/3 = \varepsilon$$

and $\lim_{x \to A} f(x) = a = \lim a_n$. $\square$

</details>
</div>

Here is the theorem that swaps limits and derivatives.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 13</span><span class="math-callout__name">(Swapping $d/dx$ and $\lim_{n \to \infty}$)</span></p>

For $n \in \mathbb{N}$ let $f_n\colon I \to \mathbb{R}$ be functions defined on a nontrivial interval $I \subset \mathbb{R}$ and such that the following three conditions hold.

1. For every $n$ there exists $f'_n\colon I \to \mathbb{R}$.
2. $f'_n \rightrightarrows f$ (on $I$) for some function $f\colon I \to \mathbb{R}$.
3. There exists an $a \in I$ such that the sequence $(f_n(a)) \subset \mathbb{R}$ converges.

Then $f_n \to F$ (on $I$) for some function $F\colon I \to \mathbb{R}$, there exists $F'\colon I \to \mathbb{R}$ and

$$F' = f \;\text{ on } I, \quad \text{i.e.,}\quad \bigl(\lim_{n \to \infty} f_n\bigr)' = \lim_{n \to \infty} f'_n.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $f_n$, $I$, $f$ and $a$ be as stated, and let $b \in I$ be any point. First we prove that the sequence $(f_n(b)) \subset \mathbb{R}$ is Cauchy. For $b = a$ this is true by Condition 3, so we can assume that for example $a < b$, the case with $b < a$ is treated similarly. Let an $\varepsilon$ be given. It follows from Conditions 2 and 3 that the sequence of functions $(f'_n)$ is uniformly Cauchy on $I$ and that the sequence $(f_n(a))$ is Cauchy. So there exists an $n_0$ such that $m, n \ge n_0 \Rightarrow |f'_m(x) - f'_n(x)| < \varepsilon$ for every $x \in I$ and also $m, n \ge n_0 \Rightarrow |f_m(a) - f_n(a)| < \varepsilon$. We take two arbitrary indices $m, n \ge n_0$ and use Lagrange's mean value theorem for the function $f_m - f_n$ and the interval $[a, b]$. This gives for some number $c \in (a, b)$ the equality and estimate

$$\frac{(f_m - f_n)(b) - (f_m - f_n)(a)}{b - a} = (f_m - f_n)'(c)$$

and

$$|f_m(b) - f_n(b)| \le |b - a| \cdot |f'_m(c) - f'_n(c)| + |f_m(a) - f_n(a)| < (b - a)\varepsilon + \varepsilon = \varepsilon(b - a + 1),$$

respectively. So the sequence $(f_n(b))$ is Cauchy, therefore convergent, and for every $b \in I$ we can define $F(b) := \lim f_n(b) \in \mathbb{R}$. So we get the function $F\colon I \to \mathbb{R}$ such that $f_n \to F$ (on $I$).

We prove that $F' = f$ on $I$. We use the previous theorem and then verify that its assumptions are satisfied. For any $b \in I$ indeed

$$F'(b) = \lim_{x \to b} \frac{F(x) - F(b)}{x - b} = \lim_{x \to b} \lim_{n \to \infty} \frac{f_n(x) - f_n(b)}{x - b} \stackrel{\text{Thm 12}}{=} \lim_{n \to \infty} \lim_{x \to b} \frac{f_n(x) - f_n(b)}{x - b} = \lim_{n \to \infty} f'_n(b) = f(b).$$

We check that in this use of Theorem 12 its assumptions are satisfied. We use the theorem for the sequence of functions $g_n(x) := \frac{f_n(x) - f_n(b)}{x - b}\colon I \setminus \lbrace b \rbrace \to \mathbb{R}$. Of course, $\lim_{x \to b} g_n(x) = f'_n(b)$ for every $n$ and also $\lim f'_n(b) = f(b)$. It remains to check that $g_n \rightrightarrows g$ (on $I \setminus \lbrace b \rbrace$) for the function $g(x) := \frac{F(x) - F(b)}{x - b}$. For this we check that the sequence $(g_n(x))$ is uniformly Cauchy on $I \setminus \lbrace b \rbrace$. For every $m, n \in \mathbb{N}$ and every $x \in I \setminus \lbrace b \rbrace$ we have the identity

$$|g_m(x) - g_n(x)| = \frac{|(f_m(x) - f_n(x)) - (f_m(b) - f_n(b))|}{|x - b|} \stackrel{(*)}{=} \underbrace{|f'_m(c) - f'_n(c)|}_{V},$$

for a $c$ between $b$ and $x$. We get equality $(*)$ where Lagrange's mean value theorem is used for the function $f_m - f_n$ and the interval with endpoints $b$ and $x$. By Condition 2, for any given $\varepsilon$ there exists an $n_0$ such that for every $m, n \ge n_0$ and every $c \in I$ one has that $|V| < \varepsilon$. Thus the sequence $(g_n(x))$ is uniformly Cauchy on $I \setminus \lbrace b \rbrace$ and the proof is complete. $\square$

</details>
</div>

## Every Continuous Function Has a Primitive Function

To prove it we need one more tool.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 14</span><span class="math-callout__name">(Uniform Continuity)</span></p>

Let $M \subset \mathbb{R}$. The function $f\colon M \to \mathbb{R}$ is **uniformly continuous** (on $M$) if

$$\forall\, \varepsilon\; \exists\, \delta\colon\; a \in M \;\Rightarrow\; f[U(a, \delta) \cap M] \subset U(f(a), \varepsilon).$$

</div>

So one $\delta$ works for all points $a \in M$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 15</span><span class="math-callout__name">(Continuity and Compactness)</span></p>

Let $M \subset \mathbb{R}$ be a compact set. If a function $f\colon M \to \mathbb{R}$ is continuous then it is uniformly continuous.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We suppose that $M \subset \mathbb{R}$ is compact and that $f\colon M \to \mathbb{R}$ is not uniformly continuous. So there is an $\varepsilon > 0$ such that for every $n$ there are two points $a_n, b_n \in M$ such that $|a_n - b_n| < 1/n$ but $|f(a_n) - f(b_n)| \ge \varepsilon$. We use compactness of $M$ and select from $(a_n)$ and $(b_n)$ convergent subsequences with limits in $M$. For simplicity of notation we assume that both $(a_n)$ and $(b_n)$ already converge and have limits $\lim a_n =: a \in M$ and $\lim b_n =: b \in M$. From $|a_n - b_n| < 1/n$ it follows that $a = b$. But from $|f(a_n) - f(b_n)| \ge \varepsilon$ and the convergence of $(a_n)$ and $(b_n)$ to $a$ it follows that for every $\delta$,

$$f[U(a, \delta) \cap M] \not\subset U(f(a), \varepsilon/2).$$

Thus the function $f$ is not continuous at $a$ and $f$ is not continuous on $M$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 16</span><span class="math-callout__name">(Existence of Antiderivative)</span></p>

Suppose that $f\colon I \to \mathbb{R}$ is a continuous function defined on a nontrivial interval $I \subset \mathbb{R}$. Then $f$ has a primitive function $F\colon I \to \mathbb{R}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Brief proof</summary>

We first assume that $I$ is compact, $I = [a, b]$ with $a < b$. A function $g\colon I \to \mathbb{R}$ is a **broken line** if it is continuous and there exists a partition $a = a_0 < a_1 < \cdots < a_k = b$ of $I$ such that each restriction $g \mid [a_{i-1}, a_i]$ is linear, i.e., of the form $g(x) = c_i x + d_i$. By Theorem 15,

$$\forall\, n\; \exists\, \text{broken line}\; g_n\colon x \in I \;\Rightarrow\; |f(x) - g_n(x)| < 1/n.$$

Since $\int(cx + d) = cx^2/2 + dx + e$, according to Proposition 6 in Lecture 8 there exist $G_n\colon I \to \mathbb{R}$ such that $G_n = \int g_n$ and $G_n(a) = 0$. But then, since $g_n \rightrightarrows f$ (on $I$) and $G'_n = g_n$ on $I$, by Theorem 13 there exists an $F\colon I \to \mathbb{R}$ such that $G_n \to F$ (on $I$) and, especially, $F' = f$ on $I$, that is, $F = \int f$.

If the interval $I$ is not compact, we write it as a union of nested non-trivial compact intervals $I_n\colon I_1 \subset I_2 \subset \dots$ and $\bigcup_{n \ge 1} I_n = I$. On each $I_n$ we take an appropriate $F_n = \int f \mid I_n$ and then $F := \bigcup_{n \ge 1} F_n$ is a primitive function of $f$ on $I$. $\square$

</details>
</div>

In a simpler way we prove this theorem later again by the Riemann integral.

# Lecture 10 — Area Under $G_f$. The Newton Integral. Integration by Parts and by Substitution

## What Are Antiderivatives Good For?

For computing areas $A_f$ of domains $D_f$ under graphs $G_f$ of functions $f\colon I \to \mathbb{R}$ defined on nontrivial intervals $I \subset \mathbb{R}$. Recall that $G_f = \lbrace(x, f(x)) \mid x \in I \rbrace \subset \mathbb{R}^2$ and that $I(c, d) \subset \mathbb{R}$ denotes the closed interval with the endpoints $c, d \in \mathbb{R}$. We define the **domain under** $G_f$ as the plane set

$$D_f := \lbrace(x, y) \mid x \in I \;\land\; y \in I(0, f(x)) \rbrace \subset \mathbb{R}^2$$

(so $G_f \subset D_f$). But what exactly is the plane **area** $A_f \in \mathbb{R}$ of $D_f$? First, $A_f$ will be a *signed area*, the parts of $D_f$ below the $x$-axis will contribute to $A_f$ negatively and those above the $x$-axis positively. Second, $A_f$ has not yet been defined in our lectures and for us it does not yet exist as a rigorous mathematical object. We bring it in existence only by a precise definition.

We give two definitions of $A_f$ in Definition 5 and a third one in Definition 6.

## Riemann Sums and Telescoping PF Sums for $A_f$

We consider two setups, with functions $f\colon I \to \mathbb{R}$ where $I$ is an interval. The first one, in this passage, is of continuous functions $f\colon [a, b] \to \mathbb{R}$, for real numbers $a < b$.

We select a **partition** $P = (a_0, a_1, \dots, a_k)$ of $[a, b]$, $k \in \mathbb{N}$ and $a = a_0 < a_1 < \cdots < a_k = b$, and define the corresponding **Riemann sum** as

$$R(P, \bar{t}, f) := \sum_{i=1}^{k} (a_i - a_{i-1}) \cdot f(t_i),$$

where $\bar{t} = (t_1, \dots, t_k)$ with $t_i \in [a_{i-1}, a_i]$ are any $k$ **test points** of $P$. This definition applies to any function $f\colon [a, b] \to \mathbb{R}$, not only to continuous ones. Note that $R(P, \bar{t}, f)$ is the signed area of the **bar graph** $B_f \subset \mathbb{R}^2$ consisting of $k$ bars (rectangles),

$$B_f := \bigcup_{i=1}^{k} [a_{i-1}, a_i] \times I(0, f(t_i)).$$

Bars under the $x$-axis (i.e., with $f(t_i) < 0$) contribute negative areas. We define the **norm of $P$** as

$$\Delta(P) := \max(\lbrace a_i - a_{i-1} \mid i = 1, 2, \dots, k \rbrace).$$

The next proposition shows that all partitions with small norm and arbitrary test points yield similar Riemann sums.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1</span><span class="math-callout__name">(On Riemann Sums)</span></p>

Let $a, b \in \mathbb{R}$ with $a < b$ and $f\colon [a, b] \to \mathbb{R}$ be a continuous function. Then

$$\forall\, \varepsilon\; \exists\, \delta\colon\; \Delta(P), \Delta(Q) < \delta \;\Rightarrow\; |R(P, \bar{t}, f) - R(Q, \bar{u}, f)| < \varepsilon$$

for any partitions $P$ and $Q$ of $[a, b]$ with any test points $\bar{t}$ and $\bar{u}$, respectively.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $a$, $b$ and $f$ be as stated, and let an $\varepsilon$ be given. By Theorem 15 in Lecture 9 we know that $f$ is uniformly continuous and therefore there is a $\delta$ such that for any $c, d \in [a, b]$, $|c - d| < \delta \Rightarrow |f(c) - f(d)| < \varepsilon/2(b - a)$. Now suppose that $P = (a_0, a_1, \dots, a_k)$ is a partition of $[a, b]$ with test points $\bar{t}$, that $Q = (b_0, b_1, \dots, b_l)$ is a partition of $[a, b]$ with test points $\bar{u}$, and that both $\Delta(P), \Delta(Q) < \delta$. We assume additionally that $P \subset Q$, i.e., that $a_0 = b_{i_0} = a$, $a_1 = b_{i_1}$, ..., $a_k = b_{i_k} = b$ for some indices $i_0 = 0 < i_1 < \cdots < i_k = l$. Later we reduce general partitions $P$ and $Q$ to this case. We have that

$$|R(P, \bar{t}, f) - R(Q, \bar{u}, f)| = \left|\sum_{r=1}^{k} \sum_{j=i_{r-1}+1}^{i_r} (b_j - b_{j-1}) \cdot (f(t_r) - f(u_j))\right|$$

$$\stackrel{|t_r - u_j| < \delta \text{ and } \Delta\text{ ineq.}}{<} \sum_{r=1}^{k} \sum_{j=i_{r-1}+1}^{i_r} (b_j - b_{j-1}) \cdot \varepsilon/2(b - a) = (b - a) \cdot \varepsilon/2(b - a) = \varepsilon/2.$$

If $P$ and $Q$ are general partitions of $[a, b]$ with respective test points $\bar{t}$ and $\bar{u}$ and with $\Delta(P), \Delta(Q) < \delta$, we set $R := P \cup Q$ (then also $\Delta(R) < \delta$) and take arbitrary test points $\bar{v}$ of $R$. Since $P \subset R$ and $Q \subset R$, we get by the previous case that

$$|R(P, \bar{t}, f) - R(Q, \bar{u}, f)| \le |R(P, \bar{t}, f) - R(R, \bar{v}, f)| + |R(R, \bar{v}, f) - R(Q, \bar{u}, f)| < \varepsilon/2 + \varepsilon/2 = \varepsilon. \quad \square$$

</details>
</div>

Since for small $\Delta(P)$ the bar graph $B_f$ closely approximates the domain $D_f$, one can expect that $R(P, \bar{t}, f) \to A_f$ as $\Delta(P) \to 0$. We define this limit formally.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2</span><span class="math-callout__name">(Limits of Riemann Sums)</span></p>

Let $a, b, L \in \mathbb{R}$, $a < b$ and $f\colon [a, b] \to \mathbb{R}$ be a function, not necessarily continuous. If for any sequences $(P_n)$ of partitions $P_n$ of $[a, b]$ and $(\bar{t}(n))$ of tuples of test points of $P_n$ it is true that

$$\lim \Delta(P_n) = 0 \;\Rightarrow\; \lim R(P_n, \overline{t(n)}, f) = L,$$

we write $\lim_{\Delta(P) \to 0} R(P, \bar{t}, f) = L$ and say that the **Riemann sums of $f$ have the limit $L$**.

</div>

These limits are unique by definition and below we easily deduce from Proposition 1 that for continuous functions they always exist.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 3</span><span class="math-callout__name">(Limits of Riemann Sums Exist)</span></p>

For every continuous function $f\colon [a, b] \to \mathbb{R}$, $a, b \in \mathbb{R}$ with $a < b$, the (finite) limit

$$\lim_{\Delta(P) \to 0} R(P, \bar{t}, f) \in \mathbb{R}$$

exists.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $f$, $a$ and $b$ be as stated, and let $(P_n)$ be an arbitrary sequence of partitions of the interval $[a, b]$ with respective test points $\overline{t(n)}$ and such that $\lim \Delta(P_n) = 0$. By Proposition 1 the sequence $(R(P_n, \overline{t(n)}, f))$ is Cauchy and therefore it has a limit $L \in \mathbb{R}$. If $(Q_n)$ and $\overline{u(n)}$ is another sequence of partitions of $[a, b]$ with respective test points $\overline{u(n)}$ and with $\lim \Delta(Q_n) = 0$, then by Proposition 1,

$$\lim_{n \to \infty} \bigl(R(P_n, \overline{t(n)}, f) - R(Q_n, \overline{u(n)}, f)\bigr) = 0.$$

Therefore also $\lim R(Q_n, \overline{u(n)}, f) = L$. $\square$

</details>
</div>

However, in this lecture we are more interested in Newton's approach to the areas $A_f$. We express the summands $(a_i - a_{i-1}) \cdot f(t_i)$ in Riemann sums in terms of any PF $F$ of the continuous $f$ as follows; we know that $F$ exists by the last theorem in Lecture 9. Let $P = (a_0, a_1, \dots, a_k)$ be any partition of $[a, b]$. We use Lagrange's mean value theorem for $F$ and every interval $[a_{i-1}, a_i]$:

$$\frac{F(a_i) - F(a_{i-1})}{a_i - a_{i-1}} = F'(c_i) = f(c_i)$$

for some point $c_i \in (a_{i-1}, a_i)$. Thus

$$F(b) - F(a) = \sum_{i=1}^{k}(F(a_i) - F(a_{i-1})) = \sum_{i=1}^{k}(a_i - a_{i-1}) \cdot f(c_i) = R(P, \bar{c}, f),$$

with the test points $\bar{c} = (c_1, \dots, c_k)$ of $P$. In view of Proposition 1 we get the following equality.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 4</span><span class="math-callout__name">(Riemann = Newton)</span></p>

Let $a < b$ be real numbers, let $f\colon [a, b] \to \mathbb{R}$ be a continuous function and let $F\colon [a, b] \to \mathbb{R}$ be a primitive of $f$. Then

$$\lim_{\Delta(P) \to 0} R(P, \bar{t}, f) = F(b) - F(a).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $f$, $a$, $b$, $F$ be as stated, and let $(P_n)$ be any sequence of partitions of $[a, b]$ with test points $\overline{t(n)}$ and such that $\lim \Delta(P_n) = 0$. We know by the above argument that there exist test points $\overline{c(n)}$ of $P_n$ such that, for every $n$, $F(b) - F(a) = R(P_n, \overline{c(n)}, f)$. Hence, by the arithmetic of limits of sequences,

$$\lim R(P_n, \overline{t(n)}, f) = \lim\bigl(R(P_n, \overline{t(n)}, f) - R(P_n, \overline{c(n)}, f)\bigr) + \lim R(P_n, \overline{c(n)}, f) = 0 + F(b) - F(a) = F(b) - F(a).$$

Thus we get the stated limit. $\square$

</details>
</div>

Now we can give two definitions of the area $A_f$ of the domain $D_f$ under $G_f$ for any continuous function $f\colon [a, b] \to \mathbb{R}$. By the previous corollary they give for $A_f$ the same value.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5</span><span class="math-callout__name">(Area Under Graph)</span></p>

Let $f\colon [a, b] \to \mathbb{R}$, for real numbers $a < b$, be a continuous function and $D_f \subset \mathbb{R}^2$ be the domain under its graph $G_f$, as defined earlier. One can define the area $A_f \in \mathbb{R}$ of $D_f$ in two ways.

1. *(I. Newton)* Set $A_f := F(b) - F(a)$ for any antiderivative $F\colon [a, b] \to \mathbb{R}$ of $f$.
2. *(B. Riemann)* Set $A_f := \lim_{\Delta(P) \to 0} R(P, \bar{t}, f)$ (see Definition 2).

</div>

At first look these two definitions appear very differently, but we know from Corollary 4 that $A_f$ is the same in both. The former is considerably simpler than the latter, but the latter works in certain cases when the former does not work. Later we will see that the scopes of both definitions are in fact incomparable.

For example, if $f(x) = x^2\colon [-1, 1] \to \mathbb{R}$ then $F(x) = x^3/3$ is a primitive of $f$ on $[-1, 1]$. By Newton's definition the area of the domain $D_f = \lbrace(x, y) \mid -1 \le x \le 1 \;\land\; 0 \le y \le x^2 \rbrace$ equals $A_f = F(1) - F(-1) = 1/3 - (-1)^3/3 = 2/3$.

## The Newton Integral

Now we consider the second setup of functions $f\colon I \to \mathbb{R}$, namely of functions $f\colon (a, b) \to \mathbb{R}$, for real $a < b$, that have a primitive function $F$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6</span><span class="math-callout__name">(Newton Integral)</span></p>

Let $a, b \in \mathbb{R}$ with $a < b$ and $F, f\colon (a, b) \to \mathbb{R}$ be functions such that $F$ is a primitive of $f$. We define the **Newton integral** of $f$ over the interval $(a, b)$ as the difference

$$\text{(N)} \int_a^b f = F(b) - F(a) := \lim_{x \to b} F(x) - \lim_{x \to a} F(x),$$

if the last two limits exist and are finite. Then we define the **area** $A_f$ of the domain $D_f$ under $G_f$ as

$$A_f := \text{(N)} \int_a^b f.$$

</div>

By now it is clear that above we need not use one-sided limits. Since any two primitives $F_1$ and $F_2$ of $f$ differ only by a constant shift, $F_1 = F_2 + c$, the value of $\text{(N)} \int_a^b f$, if it exists, is independent of the choice of $F$.

The situation in Definition 6 is strictly more general than the former one with continuous $f\colon [a, b] \to \mathbb{R}$, because if a function $f\colon (a, b) \to \mathbb{R}$ has a primitive $F\colon (a, b) \to \mathbb{R}$ then $f$ need not be continuous. Even if $f$ is continuous and $F$ is extended to $F\colon [a, b] \to \mathbb{R}$ by limits at $a$ and $b$, then the derivatives $F'(a)$ and $F'(b)$ need not exist and $f$ cannot be extended to $a$ and $b$.

If for a function $f\colon (a, b) \to \mathbb{R}$, where $a < b$ are real numbers, the Newton integral $\text{(N)} \int_a^b f$ exists, we say that the function $f$ is **Newton-integrable** (on $(a, b)$) and write that $f \in \mathrm{N}(a, b)$. It is easy too see that if $f \in \mathrm{N}(a, b)$ then $f \in \mathrm{N}(c, d)$ for any numbers $c < d$ in the interval $(a, b)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7</span><span class="math-callout__name">(Monotonicity of the Newton Integral)</span></p>

If $f, g \in \mathrm{N}(a, b)$ and $f \le g$ on $(a, b)$ then

$$\text{(N)} \int_a^b f \le \text{(N)} \int_a^b g.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $F$ and $G$ be the respective primitives of $f$ and $g$ on $(a, b)$. We take any numbers $c < d$ in $(a, b)$ and use the Lagrange mean value theorem for the function $F - G$ and interval $[c, d]$. We get that for some point $e \in (c, d)$,

$$(F(d) - G(d)) - (F(c) - G(c)) = (F - G)'(e) \cdot (d - c) = (f(e) - g(e)) \cdot (d - c) \le 0.$$

Hence $F(d) - F(c) \le G(d) - G(c)$. This inequality is preserved under the limit transitions $c \to a$ and $d \to b$ and we get the stated inequality between both Newton integrals. $\square$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Newton Integrals)</span></p>

$$\text{(N)} \int_0^1 \sqrt{x} = \frac{2 \cdot 1^{3/2}}{3} - \frac{2 \cdot 0^{3/2}}{3} = \frac{2}{3}$$

but

$$\text{(N)} \int_0^1 \frac{1}{x} = \log 1 - \log 0 = 0 - (-\infty) = \;?$$

does not exist because the limit of the primitive $\log x$ at 0 is not finite.

</div>

## Proof of the Second Case of L'Hospital's Rule

As an application of the Newton integral we prove the remaining case of l'Hospital's rule for $\lim_{x \to A} g(x) = \pm\infty$ (Condition 2 in Theorem 7 in Lecture 8). We prove first an asymptotics for Newton integrals.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8</span><span class="math-callout__name">(Asymptotics of (N) $\int$)</span></p>

We assume that $f, g \in \mathrm{N}(a, b)$, $g > 0$ on $(a, b)$, that $f(x) = o(g(x))$ $(x \to a)$ and that $\lim_{x \to a} \text{(N)} \int_x^b g = +\infty$. Then

$$\text{(N)} \int_x^b f = o\!\left(\text{(N)} \int_x^b g\right) \quad (x \to a).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let an $\varepsilon$ be given. By the assumption of the first $o$ there exists a $\delta \le b - a$ such that $x \in (a, a + \delta) \Rightarrow |f(x)| < \frac{\varepsilon}{2} \cdot g(x)$. By the assumption of the limit $+\infty$ there exists a $\theta < \delta$ such that $x \in (a, a + \theta) \Rightarrow |\text{(N)} \int_{a+\delta}^b f| < \frac{\varepsilon}{2} \cdot \text{(N)} \int_x^b g$. Thus if $x \in (a, a + \theta)$ then

$$\left|\text{(N)} \int_x^b f\right| = \left|\text{(N)} \int_x^{a+\delta} f + \text{(N)} \int_{a+\delta}^b f\right| \le \left|\text{(N)} \int_x^{a+\delta} f\right| + \left|\text{(N)} \int_{a+\delta}^b f\right|$$
$$\stackrel{\text{both } \Rightarrow \text{s and Prop. 7}}{<} \frac{\varepsilon}{2} \cdot \text{(N)} \int_x^{b \text{ or } a+\delta} g + \frac{\varepsilon}{2} \cdot \text{(N)} \int_x^b g = \varepsilon \cdot \text{(N)} \int_x^b g. \quad \square$$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9</span><span class="math-callout__name">(L'Hospital's Rule, Condition 2)</span></p>

Let $A \in \mathbb{R}$. Let for a $\delta$ functions $f, g\colon P^+(A, \delta) \to \mathbb{R}$ have on $P^+(A, \delta)$ finite derivatives, $g' \neq 0$ on $P^+(A, \delta)$, and let $\lim_{x \to A} g(x) = \pm\infty$. Then

$$\lim_{x \to A} \frac{f(x)}{g(x)} = \lim_{x \to A} \frac{f'(x)}{g'(x)}$$

if the last limit exists. This theorem also holds for left neighborhoods $P^-(A, \delta)$, ordinary neighborhoods $P(A, \delta)$ and for $A = \pm\infty$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $A$, $\delta$, $f$ and $g$ be as stated and let $A \in \mathbb{R}$. We assume that $\lim_{x \to A} g(x) = +\infty$ and $g > 0$ on $(A, A + \delta)$, the case with limit $-\infty$ is treated similarly. Let $\lim_{x \to A} f'(x)/g'(x) =: L \in \mathbb{R}^*$. We assume first that $L = 0$, i.e., $f'(x) = o(g'(x))$ $(x \to A)$. We fix a $\theta < \delta$ and get by the previous theorem that

$$\text{(N)} \int_x^{\theta} f' = o\!\left(\text{(N)} \int_x^{\theta} g'\right) \quad (x \to A),$$

which gives that $f(x) = f(\theta) - o(1)(g(\theta) - g(x))$. Thus $f(x)/g(x) = f(\theta)/g(x) + o(1)(1 - g(\theta)/g(x)) = o(1) + o(1)(1 - o(1)) = o(1)$ and $\lim_{x \to A} f(x)/g(x) = 0 = L$.

Let $L \in \mathbb{R}$. But then with $h(x) := f(x) - Lg(x)$ we have that $\lim_{x \to A} h'(x)/g'(x) = 0$ and therefore, by the just proved case, $0 = \lim_{x \to A} h(x)/g(x) = \lim_{x \to A} f(x)/g(x) - L$ and $\lim_{x \to A} f(x)/g(x) = L$. If $L = +\infty$ then $\lim_{x \to A} g'(x)/f'(x) = 0^+$. Thus by the previous case $\lim_{x \to A} g(x)/f(x) = 0^+$ and we get that $\lim_{x \to A} f(x)/g(x) = +\infty$.

For the left deleted neighborhoods $P^-(A, \delta)$ and for two-sided neighborhoods $P(A, \delta)$ the proofs are similar, and for $A = \pm\infty$ we use the substitution $x := 1/y$ as in the $\frac{0}{0}$ case. $\square$

</details>
</div>

## The Stirling Formula

One can prove the Stirling asymptotic formula

$$1 \cdot 2 \cdot \ldots \cdot n = n! \sim \sqrt{2\pi n} \left(\frac{n}{\mathrm{e}}\right)^n \quad (n \to \infty)$$

by using only the Newton integral (but it is not too simple).

## The Darboux Property

A function $f\colon I \to \mathbb{R}$, defined on an interval $I \subset \mathbb{R}$, has the **Darboux property** (or is **Darboux**) if it attains every intermediate value: if $a < b$ are in $I$ and $c$ is such that $f(a) < c < f(b)$ or $f(a) > c > f(b)$ then $c = f(d)$ for some $d \in (a, b)$. We proved earlier (Theorem 8 in Lecture 6) that continuous functions are Darboux. Now we extend it to a larger class of functions.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10</span><span class="math-callout__name">(Derivatives Are Darboux)</span></p>

Any function $f\colon I \to \mathbb{R}$, defined on an interval $I \subset \mathbb{R}$, with a primitive function has the Darboux property.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We assume that $f\colon [a, b] \to \mathbb{R}$, where $a < b$ are real numbers, has a primitive function $F\colon [a, b] \to \mathbb{R}$ and that $f(a) < c < f(b)$, the case $f(a) > c > f(b)$ is treated similarly. We consider the function

$$G(x) := F(x) - cx\colon [a, b] \to \mathbb{R}.$$

It has on $[a, b]$ the finite derivative $G'(x) = F'(x) - c = f(x) - c$. In particular, $G$ is continuous. By the min-max principle (Theorem 13 in Lecture 6), $G$ attains at some $d \in [a, b]$ its minimum value. From

$$G'(a) = f(a) - c < 0 \quad \text{and} \quad G'(b) = f(b) - c > 0$$

it follows (by Proposition 5 in Lecture 8) that $d \in (a, b)$. By another earlier theorem (Theorem 4 in Lecture 7), $f(d) - c = G'(d) = 0$, so that $f(d) = c$. $\square$

</details>
</div>

Since every continuous function has a primitive function and since there exist non-continuous functions which have primitives, the previous class of functions with the Darboux property is strictly larger than the class of continuous functions. The theorem is usually used in reverse: if a function does not have the Darboux property then it has no primitive function. For example, the signum function $\operatorname{sgn}(x)$ is not Darboux on any nontrivial interval $I \ni 0$ and therefore it does not have primitive there.

## Linearity of Antiderivatives

Recall that the notation $F = \int f$ for two functions $F, f\colon I \to \mathbb{R}$ means that $F$ is a primitive of $f$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 11</span><span class="math-callout__name">(Linearity of $\int$)</span></p>

Suppose that $f, g\colon I \to \mathbb{R}$ are functions defined on a nontrivial interval $I \subset \mathbb{R}$ and that $a, b \in \mathbb{R}$. Then

$$\int(af + bg) = a \int f + b \int g,$$

meaning that if $F$, resp. $G$, is an antiderivative of $f$, resp. $g$, then $aF + bG$ is an antiderivative of $af + bg$.

</div>

## Integration by Parts

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 12</span><span class="math-callout__name">(Integration by Parts)</span></p>

Suppose that $I \subset \mathbb{R}$ is a nontrivial interval and that $f, g, F, G\colon I \to \mathbb{R}$ are functions such that $F$ is a primitive of $f$ and $G$ is a primitive of $g$. Then

$$\int fG = FG - \int Fg,$$

meaning that if $H$ is a primitive of $Fg$ then $FG - H$ is a primitive of $fG$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

This is an immediate consequence of the Leibniz formula and linearity of differentiation:

$$(FG - H)' = F'G + FG' - H' = fG + Fg - Fg = fG. \quad \square$$

</details>
</div>

One can write the integration by parts formula also as $\int F'G = FG - \int FG'$. Note how the prime moves from $F$ to $G$. For example,

$$\int \log x = \int x' \log x = x \log x - \int x(\log x)' = x \log x - \int \frac{x}{x} = x \log x - x.$$

Or $\int x \sin x = \int x(-\cos x)' = -x\cos x + \int x' \cos x = -x\cos x + \sin x.$

It is usually easy to check the result by taking the derivative.

## Integration by Substitution

This is another useful technique for computing primitives. The formula has two forms.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 13</span><span class="math-callout__name">(Integration by Substitution)</span></p>

If $I, J \subset \mathbb{R}$ are nontrivial intervals, $g\colon I \to J$, $f\colon J \to \mathbb{R}$ and $g$ has on $I$ finite $g'$, then the following hold.

1. If $F = \int f$ on $J$ then $F(g) = \int f(g) \cdot g'$ on $I$.
2. If $g$ is onto and $g' \neq 0$ on $I$ then one has the implication $G = \int f(g) \cdot g'$ on $I \;\Rightarrow\; G(g^{-1}) = \int f$ on $J$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**1.** The formula for derivatives of composite functions gives that $(F(g))' = F'(g) \cdot g' = f(g) \cdot g'$.

**2.** Since $g'$ is Darboux (Theorem 10), either $g' > 0$ or $g' < 0$ on $I$. Therefore $g$ either increases or decreases. Thus we have the continuous inverse $g^{-1}\colon J \to I$ because $g$ is continuous on an interval. The formulas for derivatives of composite functions and of inverse functions give that

$$(G(g^{-1}))' = G'(g^{-1}) \cdot (g^{-1})' = f(\underbrace{g(g^{-1})}_{=\mathrm{id}}) \cdot g'(g^{-1}) \cdot \frac{1}{g'(g^{-1})} = f. \quad \square$$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1</span><span class="math-callout__name">(Substitution: Linear Argument)</span></p>

If $F = \int f$ on $I$ and $a, b \in \mathbb{R}$ with $a \neq 0$ then the first formula gives that

$$\frac{F(ax + b)}{a} = \int f(ax + b) \quad \text{on } J := (I - b)/a.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2</span><span class="math-callout__name">(Substitution: $\int\sqrt{1-t^2}$)</span></p>

What is $\int f := \int \sqrt{1 - t^2}$ on $J = (-1, 1)$? We plug in for $t$ the function $g(x) := \sin x\colon I := (-\pi/2, \pi/2) \to J$. We get by integration by parts that

$$\int f(g) \cdot g' = \int \cos^2 x = \int (\sin x)' \cos x = \sin x \cdot \cos x - \int \sin x (\cos x)' = \sin x \cdot \cos x + \int (1 - \cos^2 x) = \sin x \cdot \cos x + x - \int \cos^2 x$$

and therefore $\int f(g) \cdot g' = \int \cos^2 x = \frac{\sin x \cdot \cos x + x}{2} =: G(x)$. Thus by the second formula and since $\cos x = \sqrt{1 - \sin^2 x}$ on $I$,

$$\int f = \int \sqrt{1 - t^2} = G(g^{-1}) = \frac{t\sqrt{1 - t^2} + \arcsin t}{2}.$$

</div>

# Lecture 11 — More on the Newton Integral. Computing Primitives of Rational Functions

## The General Newton Integral

We extend $\text{(N)} \int_a^b f$ to functions defined on any nonempty open interval $(A, B)$ with $A < B$ in $\mathbb{R}^*$. These are exactly the intervals $(-\infty, a)$, $(a, b)$, $(a, +\infty)$ and $(-\infty, +\infty) = \mathbb{R}$ with any real numbers $a < b$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1</span><span class="math-callout__name">(General Newton Integral)</span></p>

Let $A < B$ be in $\mathbb{R}^*$ and $F, f\colon (A, B) \to \mathbb{R}$ be functions such that $F$ is a primitive of $f$. We define the **Newton integral** of $f$ over the interval $(A, B)$ as the difference

$$\text{(N)} \int_A^B f = F(B) - F(A) := \lim_{x \to B} F(x) - \lim_{x \to A} F(x),$$

if the last two limits exist and are finite. Then we define the area $A_f$ of the domain $D_f$ under $G_f$ as $A_f := \text{(N)} \int_A^B f$.

</div>

Like for the earlier Newton integral over $(a, b)$, the value of the present integral does not depend on the choice of $F$ because any two primitives of $f$ differ by a constant shift. If $\text{(N)} \int_A^B f$ is defined we say that $f$ is **Newton-integrable over** $(A, B)$ and write that $f \in \mathrm{N}(A, B)$.

For instance, $\frac{1}{1+x^2} \in \mathrm{N}(0, +\infty)$ as

$$\text{(N)} \int_0^{+\infty} \frac{1}{1+x^2} = \overbrace{\arctan(+\infty)}^{\lim_{x \to +\infty} \arctan x} - \arctan(0) = \pi/2 - 0 = \pi/2.$$

For $F\colon (A, B) \to \mathbb{R}$ we introduce the notation

$$[F]_A^B := \lim_{x \to B} F(x) - \lim_{x \to A} F(x),$$

if both limits exist and are finite.

We extend $\text{(N)} \int_A^B f$ a little more by allowing $B \le A$. We set $\text{(N)} \int_A^A f := 0$ for any function $f$ and have that

$$\text{(N)} \int_B^A f = -\text{(N)} \int_A^B f$$

if $f \in \mathrm{N}(A, B)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2</span><span class="math-callout__name">(Additivity of Integral)</span></p>

If $A, B, C \in \mathbb{R}^*$ and $f \in \mathrm{N}(\min(A, B, C), \max(A, B, C))$ then

$$\text{(N)} \int_A^C f = \text{(N)} \int_A^B f + \text{(N)} \int_B^C f,$$

that is, $\text{(N)} \int_A^B f + \text{(N)} \int_B^C f + \text{(N)} \int_C^A f = 0$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3</span><span class="math-callout__name">(Linearity of Integral)</span></p>

If $A$ and $B$ are in $\mathbb{R}^*$, $a, b \in \mathbb{R}$ and $f, g \in \mathrm{N}(A, B)$ then

$$\text{(N)} \int_A^B (af + bg) = a \cdot \text{(N)} \int_A^B f + b \cdot \text{(N)} \int_A^B g.$$

</div>

## Integration by Parts for the General Newton Integral

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4</span><span class="math-callout__name">($\text{(N)} \int_A^B$ by Parts)</span></p>

Consider four functions $f, g, F, G\colon (A, B) \to \mathbb{R}$, where $A < B$ are in $\mathbb{R}^*$, such that $F$, resp. $G$, is a primitive of $f$, resp. $g$. Then the equality

$$\underbrace{\text{(N)} \int_A^B fG}_{T_1} = \underbrace{[FG]_A^B}_{T_2} - \underbrace{\text{(N)} \int_A^B Fg}_{T_3}$$

holds whenever two of the three terms $T_i$ are defined.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**Case 1.** Suppose that the first two terms $T_1, T_2 \in \mathbb{R}$ are defined. So $fG$ has on $(A, B)$ a primitive $H$ with $[H]_A^B = T_1$ and $[FG]_A^B = T_2$. Then $(FG - H)' = F'G + FG' - H' = fG + Fg - fG = Fg$ and the last equality is a rearrangement of the equality stated in the theorem: $[FG - H]_A^B = [FG]_A^B - [H]_A^B = T_2 - T_1$.

**Case 2.** Suppose that the first and third term $T_1 \in \mathbb{R}$ and $T_3 \in \mathbb{R}$ are defined. So $fG$, resp. $Fg$, has on $(A, B)$ a primitive $H_1$, resp. $H_2$, with $[H_1]_A^B = T_1$ and $[H_2]_A^B = T_3$. Then $(H_1 + H_2)' = fG + Fg = (FG)'$ on $(A, B)$. By an earlier result (Theorem 9 in Lecture 9) there is a constant $c$ such that $H_1 + H_2 + c = FG$ on $(A, B)$. Hence $[FG]_A^B = [H_1 + H_2 + c]_A^B = [H_1]_A^B + [H_2]_A^B = T_1 + T_3$, which is a rearrangement of the equality stated in the theorem.

**Case 3.** The case when $T_2, T_3 \in \mathbb{R}$ are defined is similar to Case 1 and is left to the reader as an exercise. $\square$

</details>
</div>

**For example,** we set $I_n := \text{(N)} \int_0^{+\infty} x^n \mathrm{e}^{-x}$, $n \in \mathbb{N}_0$. Then $I_0 = [-\mathrm{e}^{-x}]_0^{+\infty} = -\mathrm{e}^{-\infty} - (-\mathrm{e}^{-0}) = -0 - (-1) = 1$. For $n > 0$ we get by the last theorem and Propositions 2 and 3 that

$$I_n = \text{(N)} \int_0^{+\infty} x^n (-\mathrm{e}^{-x})' = [-x^n \mathrm{e}^{-x}]_0^{+\infty} + \text{(N)} \int_0^{+\infty} (x^n)' \mathrm{e}^{-x} = -0 + 0 + n \cdot \text{(N)} \int_0^{+\infty} x^{n-1} \mathrm{e}^{-x} = n \cdot I_{n-1}.$$

Therefore $I_n = n! = \prod_{j=1}^n j$ for every $n \in \mathbb{N}_0$. This representation of factorials by integrals can be used to prove the Stirling formula.

## Integration by Substitution for the General Newton Integral

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5</span><span class="math-callout__name">($\text{(N)} \int_A^B f$ by Substitution)</span></p>

If $A < B$ and $C < D$ are in $\mathbb{R}^*$, $g\colon (A, B) \to (C, D)$, $f\colon (C, D) \to \mathbb{R}$ and $g$ has on $(A, B)$ finite $g'$, then the following two claims are true.

1. Suppose that $f$ has on $(C, D)$ a primitive function $F$. Then the equality

   $$\text{(N)} \int_A^B f(g) \cdot g' = \text{(N)} \int_{g(A)}^{g(B)} f$$

   holds if the right-hand side is defined.

2. If $g$ is onto and $g' \neq 0$ on $(A, B)$ then the equality

   $$\text{(N)} \int_C^D f = \text{(N)} \int_{g^{-1}(C)}^{g^{-1}(D)} f(g) \cdot g'$$

   holds if the right-hand side is defined. Here $\lbrace g^{-1}(C), g^{-1}(D) \rbrace = \lbrace A, B \rbrace$ (in some order).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**1.** Let $A$, $B$, $C$, $D$, $g$, $f$ and $F$ be as stated and let the right-hand side be defined. This means that the limits $g(A) := \lim_{x \to A} g(x) \in \mathbb{R}^\ast$ and $g(B) := \lim_{x \to B} g(x) \in \mathbb{R}^\ast$ exist. It follows that $g(A)$ and $g(B)$ are limit points of $(C, D)$. It also means that the right-hand side has the value $\lim_{y \to g(B)} F(y) - \lim_{y \to g(A)} F(y)$, in particular the last two limits exist and are finite. We already know that $F(g)$ is on $(A, B)$ a primitive of $f(g) \cdot g'$. Thus

$$\text{(N)} \int_{g(A)}^{g(B)} f = \lim_{y \to g(B)} F(y) - \lim_{y \to g(A)} F(y) = \lim_{x \to B} F(g(x)) - \lim_{x \to A} F(g(x)) = \text{(N)} \int_A^B f(g) \cdot g'.$$

Here the first and third equality follow from the definition of the general Newton integral. The crucial middle equality follows by the theorem on limits of composite functions (Theorem 14 in Lecture 5) whose Condition 1 holds as the outer function $F$ is continuous.

**2.** Let $A$, $B$, $C$, $D$, $g$ and $f$ be as stated and let the right-hand side be defined. From the proof of part 2 of Theorem 13 in Lecture 10 we know that $g$ is an increasing or decreasing bijection, and therefore so is the inverse $g^{-1}\colon (C, D) \to (A, B)$ (which is also continuous). Thus the limits $g^{-1}(C) := \lim_{y \to C} g^{-1}(y) \in \mathbb{R}^\ast$ and $g^{-1}(D) := \lim_{y \to D} g^{-1}(y) \in \mathbb{R}^\ast$ exist and are equal $\lbrace A, B \rbrace$ (in some order). Since the right-hand side is defined, $f(g) \cdot g'$ has on $(A, B)$ a primitive function $G$ and the right-hand side has a finite value. We already know that $G(g^{-1})$ is on $(C, D)$ a primitive of $f$. Thus

$$\text{(N)} \int_{g^{-1}(C)}^{g^{-1}(D)} f(g) \cdot g' = \lim_{x \to g^{-1}(D)} G(x) - \lim_{x \to g^{-1}(C)} G(x) = \lim_{y \to D} G(g^{-1}(y)) - \lim_{y \to C} G(g^{-1}(y)) = \text{(N)} \int_C^D f.$$

The first and third equality follow from the definition of the general Newton integral and the second equality again follows by the theorem on limits of composite functions. $\square$

</details>
</div>

**For example,** last time we computed that $\int \sqrt{1 - t^2} = \frac{t\sqrt{1 - t^2} + \arcsin t}{2} =: F(t)$ on $(-1, 1)$. By Proposition 6 in Lecture 8, $F'(-1) = F'(1) = 0$, and therefore this relation holds even on $[-1, 1]$. Thus the area of $D_f$ is (as defined)

$$A_f = \text{(N)} \int_{-1}^1 \sqrt{1 - t^2} = \lim_{t \to 1} F(t) - \lim_{t \to -1} F(t) = (\arcsin 1)/2 - (\arcsin(-1))/2 = \pi/4 - (-\pi/4) = \pi/2.$$

This agrees with the double area $\pi$ of the unit disc $\lbrace(x, y) \in \mathbb{R}^2 \mid x^2 + y^2 \le 1 \rbrace$ because $D_f$ is its upper half.

## A Table of Antiderivatives of Some Elementary Functions

This table is obtained completely mechanically by inverting the rules for differentiation in the table of derivatives in Theorem 17 in Lecture 7.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6</span><span class="math-callout__name">(A Table of Primitives)</span></p>

The following formulas hold.

1. On $\mathbb{R}$: $\int \exp(x) = \exp(x)$, $\int \sin x = -\cos x$, $\int \cos x = \sin x$, $\int 1/(1+x^2) = \arctan x$ (and also $= -\text{arccot}\, x$) and $\int x^n = x^{n+1}/(n+1)$ for every $n \in \mathbb{N}\_0$.
2. Both on $(-\infty, 0)$ and on $(0, +\infty)$: $\int 1/x = \log(\|x\|)$ and $\int x^n = x^{n+1}/(n+1)$ for every $n \in \lbrace -2, -3, \dots \rbrace$.
3. On $(0, +\infty)$: $\int x^b = x^{b+1}/(b+1)$ for every $b \in \mathbb{R} \setminus \mathbb{Z}$ and $(\log x)' = 1/x$.
4. On every interval $(k\pi - \pi/2, k\pi + \pi/2)$ with $k \in \mathbb{Z}$: $\int 1/(\cos x)^2 = \tan x$.
5. On every interval $(k\pi, (k+1)\pi)$ with $k \in \mathbb{Z}$: $\int 1/(\sin x)^2 = -\cot x$.
6. On $(-1, 1)$: $\int 1/\sqrt{1 - x^2} = \arcsin x$ (and also $= -\arccos x$).

</div>

In connection with the first formula in 2 note that, formally and interestingly, both $(\log x)' = 1/x$ and $(\log(-x))' = (1/(-x)) \cdot (-x)' = 1/x$. This seemingly contradicts the basic result that primitives of the same function only differ by a constant shift. Resolution of this conundrum is simple: the functions $\log x$ and $\log(-x)$ have disjoint definition domains.

## Computing Primitives of Rational Functions

This is a large class of functions for which antiderivatives can be explicitly computed. Recall that a rational function $r = r(x)$ is a ratio of two polynomials:

$$r(x) = \frac{p(x)}{q(x)}\colon \underbrace{\mathbb{R} \setminus Z(r)}_{\mathrm{Def}(r)} \to \mathbb{R}.$$

Here $p(x), q(x) \in \mathbb{R}[x]$ are polynomials with real coefficients, $q(x)$ is not the zero polynomial and $Z(r) = \lbrace a \in \mathbb{R} \mid q(a) = 0 \rbrace$ is the zero set (the set of real roots) of the denominator $q(x)$. It is well known that $\|Z(r)\| \le \deg q$, the **degree** of the polynomial $q = q(x)$. An **irreducible trinomial** $a(x)$ is any real monic (= with the leading coefficient 1) quadratic polynomial

$$a(x) := x^2 + bx + c$$

such that $b^2 - 4c < 0$, i.e., $a(x)$ has no real root. Note that then $a(x) > 0$ for every $x \in \mathbb{R}$. For example, $x^2 + 2x + 2$ is an irreducible trinomial. In the rest of the lecture we prove, modulo the proof of Theorem 8 (the Fundamental Theorem of Algebra), the next theorem.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 7</span><span class="math-callout__name">($\int r(x)$)</span></p>

For any rational function $r = r(x)$ there exists a function $R(x)$ of the form

$$R(x) = r_0(x) + \sum_{i=1}^{k} s_i \cdot \log(|x - \alpha_i|) + \sum_{i=1}^{l} t_i \cdot \log(a_i(x)) + \sum_{i=1}^{m} u_i \cdot \arctan(b_i(x)),$$

where $r_0(x)$ is a rational function, $k, l, m \in \mathbb{N}_0$, empty sums are defined as $0$, $s_i, t_i, u_i \in \mathbb{R}$, $\alpha_i \in Z(r)$, the $a_i(x)$ are irreducible trinomials and the $b_i(x)$ are real non-constant linear polynomials, and such that

$$R(x) = \int r(x)$$

on any nontrivial interval $I \subset \mathrm{Def}(r)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Antiderivative of a Rational Function)</span></p>

By linearity of integration, by integration by substitution and by the above table of primitives,

$$\int r(x) := \int \left(\frac{1}{x^4} + \frac{1}{x-1} + \frac{2x+2}{x^2+2x+2} + \frac{1}{x^2+2x+2}\right) = -\frac{1}{3x^3} + \log(|x-1|) + \log(x^2+2x+2) + \arctan(x+1)$$

on any nontrivial interval $I \subset \mathbb{R} \setminus \lbrace 0, 1 \rbrace$.

</div>

## Partial Fractions

We will not prove the next theorem here.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8</span><span class="math-callout__name">(FTAlg)</span></p>

Every non-constant complex polynomial $p(x) \in \mathbb{C}[x]$ has at least one root, a number $\alpha \in \mathbb{C}$ such that $p(\alpha) = 0$.

</div>

From FTAlg we get irreducible decompositions in $\mathbb{R}[x]$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 9</span><span class="math-callout__name">(Decompositions of Real Polynomials)</span></p>

Every nonzero real polynomial $q(x)$ can be written as

$$q(x) = c \cdot \underbrace{\prod_{i=1}^{k}(x - \alpha_i)^{m_i}}_{\text{type 1 r. factors}} \cdot \underbrace{\prod_{i=1}^{l} a_i(x)^{n_i}}_{\text{type 2 r. factors}}$$

where $c \in \mathbb{R} \setminus \lbrace 0 \rbrace$ is its leading coefficient, $k, l \in \mathbb{N}_0$, empty products are defined as 1, $m_i, n_i \in \mathbb{N}$, the $\alpha_i \in \mathbb{R}$ are the all distinct real roots of $q(x)$, and the $a_i(x)$ are distinct irreducible trinomials.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

If $\alpha = a + bi \in \mathbb{C}$ is a root of $q(x)$ then also its conjugate $\overline{\alpha} = a - bi$ is a root because $q(x) \in \mathbb{R}[x]$. If $\alpha \in \mathbb{R}$, we divide $q(x)$ by $x - \alpha$ with remainder and get that $q(x) = (x - \alpha)q_1(x)$ for $q_1(x) \in \mathbb{R}[x]$. So we have split off one root factor $x - \alpha$ of type 1. If $\alpha \in \mathbb{C} \setminus \mathbb{R}$, i.e., if $b \neq 0$, then $a_\alpha(x) := (x - \alpha)(x - \overline{\alpha}) = x^2 - 2a \cdot x + (a^2 + b^2) \in \mathbb{R}[x]$ and is an irreducible trinomial: $(2a)^2 - 4(a^2 + b^2) = -4b^2 < 0$. We divide $q(x)$ by $a_\alpha(x)$ with remainder and get $q(x) = a_\alpha(x) s_1(x)$. Again, $s_1(x)$ is real and we have split off one root factor $a_\alpha(x)$ of type 2. We apply the same procedure to $q_1(x)$, resp. $s_1(x)$. Eventually, splitting off terminates at the constant polynomial $c$ and we get the stated decomposition. $\square$

</details>
</div>

We obtain partial fractions decompositions of rational functions by means of the next identity.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 10</span><span class="math-callout__name">(Bachet's Identity)</span></p>

Let $p(x)$ and $q(x)$ be two real polynomials with no common complex root, i.e., $p(z) = q(z) = 0$ for no $z \in \mathbb{C}$. Then there exist polynomials $r(x), s(x) \in \mathbb{R}[x]$ such that

$$r(x) \cdot p(x) + s(x) \cdot q(x) = 1.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

For the given polynomials $p(x)$ and $q(x)$ we consider the set of real polynomials $S = \lbrace r(x) \cdot p(x) + s(x) \cdot q(x) \mid r(x), s(x) \in \mathbb{R}[x] \rbrace$ and take nonzero $t(x) \in S$ with the minimum degree. We divide any $a(x) \in S$ by $t(x)$ with remainder: $a(x) = t(x) \cdot b(x) + c(x)$ where $b(x), c(x) \in \mathbb{R}[x]$ and $\deg c(x) < \deg t(x)$ or $c(x)$ is the zero polynomial. But $c(x) = a(x) - b(x) \cdot t(x) \in S$ (because $S$ is closed to subtraction and multiples). Thus $c(x)$ is the zero polynomial and $a(x) = b(x) \cdot t(x)$ — $t(x)$ divides any element of $S$. But $p(x), q(x) \in S$ and so $t(x)$ divides both of them. But these polynomials have no common complex root and therefore, by Theorem 8, $t(x)$ is a nonzero constant polynomial. We may assume that $t(x) = 1$ and get the stated identity. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 11</span><span class="math-callout__name">(Partial Fractions)</span></p>

Every rational function $r(x) = p(x)/q(x) \in \mathbb{R}(x)$, with $q(x)$ decomposed as in Corollary 9, expresses as

$$r(x) = s(x) + \sum_{i=1}^{k} \sum_{j=1}^{m_i} \frac{\beta_{i,j}}{(x - \alpha_i)^j} + \sum_{i=1}^{l} \sum_{j=1}^{n_i} \frac{\gamma_{i,j} x + \delta_{i,j}}{a_i(x)^j}$$

where $s(x) \in \mathbb{R}[x]$ is a polynomial, $k$, $l$, $m_i$, $n_i$, $\alpha_i$ and $a_i(x)$ are as in Corollary 9, and $\beta_{i,j}, \gamma_{i,j}, \delta_{i,j} \in \mathbb{R}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

After dividing Bachet's identity by $p(x)q(x)$ we have that for any $n$ real polynomials $q_1(x), \dots, q_n(x)$ such that no $q_i(x)$ and $q_j(x)$ with $i \neq j$ have a common complex root there exist $n$ real polynomials $s_1(x), \dots, s_n(x)$ such that

$$\frac{1}{q_1(x)q_2(x) \cdots q_n(x)} = \sum_{i=1}^{n} \frac{s_i(x)}{q_i(x)}.$$

Now let a rational function $r(x) = p(x)/q(x)$ be given and $q(x)$ be decomposed as in Corollary 9. We use the last displayed identity for $n := k + l$, $q_1(x) := (x - \alpha_1)^{m_1}$, ..., $q_k(x) := (x - \alpha_k)^{m_k}$, $q_{k+1}(x) := a_1(x)^{n_1}$, ..., $q_{k+l}(x) := a_l(x)^{n_l}$ and get real polynomials $b_1(x), \dots, b_k(x), c_1(x), \dots, c_l(x)$ such that

$$r(x) = \frac{p(x)}{q(x)} = \sum_{i=1}^{k} \frac{b_i(x)}{(x - \alpha_i)^{m_i}} + \sum_{i=1}^{l} \frac{c_i(x)}{a_i(x)^{n_i}}.$$

In each of the above $k + l$ fractions we divide numerator by denominator with remainder: $b_i(x) = (x - \alpha_i)^{m_i} \cdot s_i(x) + d_i(x)$ and $c_i(x) = a_i(x)^{n_i} \cdot s_{i+k}(x) + d_{k+i}(x)$ where $d_i(x)$ is either the zero polynomial or has degree less than that of the denominator (which is $m_i$ or $2n_i$). With $s(x) := \sum_{i=1}^{k+l} s_i(x) \in \mathbb{R}[x]$ we rewrite the last displayed equality as

$$r(x) = s(x) + \sum_{i=1}^{k} \frac{d_i(x)}{(x - \alpha_i)^{m_i}} + \sum_{i=1}^{l} \frac{d_{k+i}(x)}{a_i(x)^{n_i}}.$$

For each $i \in \lbrace 1, 2, \dots, k \rbrace$ we repeatedly divide $d_i(x)$ by $x - \alpha_i$ with remainder and express the $i$-th summand in the first sum in the above stated form. We do the same for each summand in the second sum. $\square$

</details>
</div>

## Proof of Theorem 7 on the Form of $\int r(x)$

We express the given rational function $r(x)$ as a sum of partial fractions as in Theorem 11:

$$r(x) = s(x) + \sum_{i=1}^{k} \sum_{j=1}^{m_i} \frac{\beta_{i,j}}{(x - \alpha_i)^j} + \sum_{i=1}^{l} \sum_{j=1}^{n_i} \frac{\gamma_{i,j} x + \delta_{i,j}}{a_i(x)^j}.$$

We use linearity of antiderivatives and integrate each summand in the expression separately. It is easy to integrate the first two terms: $\int s(x)$ is a polynomial (on any nontrivial real interval $I$), $\int \beta/(x - \alpha)^j = -\beta/(j-1)(x - \alpha)^{j-1}$ for any $j \ge 2$ and $\int \beta/(x - \alpha) = \beta \log(\|x - \alpha\|)$, where the last two antiderivatives hold on any nontrivial interval $I \subset \mathbb{R} \setminus \lbrace \alpha \rbrace$. Thus these contributions to $\int r(x)$ are of the first two types given in Theorem 7.

It remains to integrate the third term, which means to compute primitives of the form

$$\int \frac{\gamma x + \delta}{(x^2 + bx + c)^j}$$

where $j \in \mathbb{N}$ and $\gamma, \delta, b, c \in \mathbb{R}$ are such that $b^2 - 4c < 0$. With $d := \sqrt{c - b^2/4} > 0$ and $e := (\delta - \gamma b/2)/d^{2j-1}$ we write the last rational function as

$$\frac{\gamma x + \delta}{(x^2 + bx + c)^j} = \frac{\gamma}{2} \cdot \underbrace{\frac{2x + b}{(x^2 + bx + c)^j}}_{T := (\dots)'/(\dots)^j} + e \cdot \underbrace{\frac{1/d}{((x/d + b/2d)^2 + 1)^j}}_{U := (\dots)'/(((\dots))^2 + 1)^j}.$$

By the first integration by substitution formula, $\int T = 1/(j-1)(x^2 + bx + c)^{j-1}$ for $j \ge 2$ and $\int T = \log(x^2 + bx + c)$ for $j = 1$ (on any nontrivial real interval $I$). Thus we get contributions to $\int r(x)$ of the first and third type given in Theorem 7.

Finally, we compute $\int U$. By the first integration by substitution formula, $\int U = I_j(x/d + b/2d)$ (on any nontrivial real interval $I$) for

$$I_j = I_j(y) := \int \frac{1}{(y^2 + 1)^j}.$$

For $j \in \mathbb{N}$, integration by parts and differentiation of composite functions lead to the relation

$$I_j = \int y' \cdot \frac{1}{(y^2 + 1)^j} = \frac{y}{(y^2 + 1)^j} + 2j \int \frac{(y^2 + 1) - 1}{(y^2 + 1)^{j+1}} = \frac{y}{(y^2 + 1)^j} + 2j \cdot I_j - 2j \cdot I_{j+1}.$$

Hence we get the recurrence $I_1 = \arctan y$ (by the above table of primitives) and, for $j \in \mathbb{N}$,

$$I_{j+1} = \frac{y}{2j \cdot (y^2 + 1)^j} - (1 - 1/2j) \cdot I_j.$$

It follows from it that for every $j \in \mathbb{N}$, $I_j(y) = u(y) + r \cdot \arctan y$ where $u(y) \in \mathbb{Q}(y)$ is a rational function and $r \in \mathbb{Q}$. Since $\int U = I_j(x/d + b/2d)$, the last contribution to $\int r(x)$ is of the first and fourth type given in Theorem 7.

# Lecture 12 — The Riemann Integral

## The Riemann Integral After B. Riemann

We introduced Riemann sums in Lecture 10 and proved there in Corollary 3 that every continuous function $f\colon [a, b] \to \mathbb{R}$ is Riemann integrable. In this lecture we develop this theory in full. We consider functions of the type $f\colon [a, b] \to \mathbb{R}$, where $a < b$ are real numbers, partitions $P = (a_0, a_1, \dots, a_k)$ of $[a, b]$, where $k \in \mathbb{N}$ and $a = a_0 < a_1 < \cdots < a_k = b$, test points $\bar{t} = (t_1, \dots, t_k)$ of $P$, where $t_i \in [a_{i-1}, a_i]$, and Riemann sums

$$R(P, \bar{t}, f) = \sum_{i=1}^{k} (a_i - a_{i-1}) \cdot f(t_i).$$

We noted earlier that $R(P, \bar{t}, f)$ is the signed area of the bar graph $B_f = \bigcup_{i=1}^{k} [a_{i-1}, a_i] \times I(0, f(t_i))$ where $I(c, d)$ is the closed real interval with the endpoints $c$ and $d$. For small norm $\Delta(P) = \max_{1 \le i \le k}(a_i - a_{i-1})$ of $P$ the set $B_f$ closely approximates the domain $D_f$ under $G_f$ of $f$ and one uses limits of Riemann sums (Definition 2 of Lecture 10) to define the area $A_f$ of $D_f$. We repeat the definition here in another formulation and introduce by it the Riemann integral.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1</span><span class="math-callout__name">(Riemann Integral)</span></p>

We say that a function $f\colon [a, b] \to \mathbb{R}$ is **Riemann integrable**, and write that $f \in \mathrm{R}(a, b)$, if there exists a number $L \in \mathbb{R}$ such that for

$$\forall\, \varepsilon\; \exists\, \delta\colon \text{ any partition } P \text{ of } [a, b] \text{ and any test points } \bar{t} \text{ of } P,$$

$$\Delta(P) < \delta \;\Rightarrow\; |R(P, \bar{t}, f) - L| < \varepsilon.$$

Then we also write $\text{(R)} \int_a^b f = L$ or $\text{(R)} \int_a^b f(x)\,\mathrm{d}x = L$ and say the **(Riemann) integral** over $[a, b]$ of the function $f$ equals $L$.

</div>

For simplicity of notation we omit the qualification (R) when it is clear that the integral is Riemann one. The latter notation $\int_a^b f(x)\,\mathrm{d}x$, which is due to G. W. Leibniz, comes from Riemann sums: the sign of sum $\sum$ morphed in the integral sign $\int$ and $\mathrm{d}x$ denotes the common length $a_i - a_{i-1}$ of intervals in an equipartition $P$ of $[a, b]$. We extend the scope of the notation $\int_a^b f$ slightly by setting $\int_a^a f := 0$ for any $a \in \mathbb{R}$ and any function $f$, and $\int_b^a f := -\int_a^b f$ if $f \in \mathrm{R}(a, b)$. Since this definition is important, we state two other equivalent forms of it.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2</span><span class="math-callout__name">(Equivalent Definitions of R. Integrability)</span></p>

Let $f\colon [a, b] \to \mathbb{R}$ be any function. The next three claims are logically equivalent.

1. $f \in \mathrm{R}(a, b)$.
2. *(Cauchy's condition)* $\forall\, \varepsilon\; \exists\, \delta$ such that for any partitions $P$ and $Q$ of $[a, b]$ with respective test points $\bar{t}$ and $\bar{u}$, if $\Delta(P), \Delta(Q) < \delta$ then $\|R(P, \bar{t}, f) - R(Q, \bar{u}, f)\| < \varepsilon$.
3. *(Heine's definition)* For any sequence $(P_n)$ of partitions of $[a, b]$ with test points $\overline{t(n)}$, if $\lim \Delta(P_n) = 0$ then the sequence $(R(P_n, \overline{t(n)}, f))$ is convergent.

If 1 holds then every sequence of Riemann sums in 3 with norms going to 0 has the limit $\lim R(P_n, \overline{t(n)}, f) = \int_a^b f$.

</div>

Finitely many changes in functional values have no influence on the Riemann integral.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3</span><span class="math-callout__name">(Changing Values)</span></p>

We suppose that $f$ is in $\mathrm{R}(a, b)$ and that $g\colon [a, b] \to \mathbb{R}$ differs from $f$ in only finitely many values. Then $g \in \mathrm{R}(a, b)$ and $\int_a^b g = \int_a^b f$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $f \in \mathrm{R}(a, b)$. We suppose that $g$ differs from $f$ in $k$ points $c_1, \dots, c_k \in [a, b]$. Let $(P_n)$ be any sequence of partitions of $[a, b]$ with $\Delta(P_n) \to 0$ and let $\overline{t(n)}$ be test points of $P_n$. Then

$$R(P_n, \overline{t(n)}, g) = R(P_n, \overline{t(n)}, f) + O(k \cdot \Delta(P_n)).$$

The implicit constant in $O$ can be taken to be $\max_{1 \le i \le k} |g(c_i) - f(c_i)|$. Since $\lim \Delta(P_n) = 0$, also $\lim R(P_n, \overline{t(n)}, g) = \int_a^b f$. We are done by the previous proposition. $\square$

</details>
</div>

Of course, if $f, g\colon [a, b] \to \mathbb{R}$, $f$ is not Riemann integrable and $g$ differs from $f$ in only finitely many values then it is not Riemann integrable either (why?). This stability of $\text{(R)} \int_a^b f$ is in stark contrast with the fact that $\text{(N)} \int_a^b f$ can be destroyed by a single change in a functional value (if the Darboux property of $f$ is destroyed). Using the proposition we extend the definition of Riemann integral to any nontrivial bounded interval.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4</span><span class="math-callout__name">($\int_a^b f$ for $f$ Defined on $(a, b)$)</span></p>

Let $a < b$ be real numbers and $f\colon I \to \mathbb{R}$ for an interval of type $I = (a, b)$ or $I = (a, b]$ or $I = [a, b)$. We extend $f$ to $f_0\colon [a, b] \to \mathbb{R}$ by arbitrary values on $a$ and on $b$ and define

$$\int_a^b f := \int_a^b f_0,$$

if the right-hand side exists.

</div>

Like for the Newton integral, restriction preserves Riemann integrability.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5</span><span class="math-callout__name">(On Restrictions)</span></p>

If $a < b < c$ are real numbers and $f\colon [a, c] \to \mathbb{R}$ then

$$f \in \mathrm{R}(a, c) \;\iff\; f \in \mathrm{R}(a, b) \;\land\; f \in \mathrm{R}(b, c).$$

In the positive case, $\int_a^c f = \int_a^b f + \int_b^c f$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**Implication $\Rightarrow$.** Let $f \in \mathrm{R}(a, c)$ and let an $\varepsilon$ be given. We prove for the restriction of $f$ to $[a, b]$ Cauchy's condition of Proposition 2. Let $P_0$ and $Q_0$ be two partitions of $[a, b]$ with respective test points $\overline{t(0)}$ and $\overline{u(0)}$ and such that $\Delta(P_0), \Delta(Q_0) < \delta$, where $\delta$ guarantees satisfaction of Cauchy's condition for $\mathrm{R}(a, c)$ and $\varepsilon$. We extend $P_0$ and $Q_0$ to partitions $P$ and $Q$ of $[a, c]$ arbitrarily but so that $\Delta(P), \Delta(Q) < \delta$ and that the intervals of $P$ and $Q$ contained in $[b, c]$ are identical. We also extend $\overline{t(0)}$ and $\overline{u(0)}$ identically to test points $\bar{t}$ and $\bar{u}$ of, respectively, $P$ and $Q$. Then indeed $\|R(P_0, \overline{t(0)}, f) - R(Q_0, \overline{u(0)}, f)\| = \|R(P, \bar{t}, f) - R(Q, \bar{u}, f)\| < \varepsilon$. The proof of Cauchy's condition for the restriction $f$ to $[b, c]$ is similar. The identity $\int_a^c f = \int_a^b f + \int_b^c f$ follows by merging partitions of $[a, b]$ and $[b, c]$ in partitions of $[a, c]$ (with norms going to 0) and using the last claim in Proposition 2.

**Implication $\Leftarrow$.** Let $f \in \mathrm{R}(a, b) \cap \mathrm{R}(b, c)$. It follows that $f$ is bounded and we denote by $d > 0$ the bounding constant. Let $P$ be any partition of $[a, c]$ with test points $\bar{t}$. We split $P$ in the partitions $P_1$ and $P_2$ of, respectively, $[a, b]$ and $[b, c]$ and with respective test points $\overline{t(1)}$ and $\overline{t(2)}$ as follows. If $b \in P$, we do the splitting in the obvious way. If $b \notin P$, we obtain $P_1$ and $P_2$ by splitting the interval $[a_{i-1}, a_i]$ of $P$ such that $b \in (a_{i-1}, a_i)$ in the intervals $[a_{i-1}, b]$ and $[b, a_i]$, and get $\overline{t(1)}$ and $\overline{t(2)}$ by selecting two arbitrary test points in the two new intervals. Then $R(P, \bar{t}, f) = R(P_1, \overline{t(1)}, f) + R(P_2, \overline{t(2)}, f) + O(\Delta(P)d)$. Thus satisfaction of Cauchy's condition for $\mathrm{R}(a, b)$ and $\mathrm{R}(b, c)$ follows its satisfaction for $\mathrm{R}(a, c)$ by the same argument as for the opposite implication. The identity $\int_a^c f = \int_a^b f + \int_b^c f$ follows by the same argument as above. $\square$

</details>
</div>

We also state the analogous result for the Newton integral, including the opposite implication.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6</span><span class="math-callout__name">($\Leftarrow$ for the Newton $\int$)</span></p>

Let $A < C < B < D$ be in $\mathbb{R}^\ast$, $f\colon (A, D) \to \mathbb{R}$ and let $f \in \mathrm{N}(A, B) \cap \mathrm{N}(C, D)$. Then $f \in \mathrm{N}(A, D) \cap \mathrm{N}(C, B)$ and

$$\text{(N)} \int_A^D f = \text{(N)} \int_A^B f + \text{(N)} \int_C^D f - \text{(N)} \int_C^B f.$$

</div>

We give the fourth definition of the area under graph; see Lecture 10 for the definitions of $D_f$ and $G_f$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 7</span><span class="math-callout__name">(Again $A_f$)</span></p>

If $f \in \mathrm{R}(a, b)$ then we define the area $A_f$ of the domain $D_f$ under the graph $G_f$ of the function $f\colon [a, b] \to \mathbb{R}$ (or $f\colon [a, b) \to \mathbb{R}, \dots$) as

$$A_f := \int_a^b f(x)\,\mathrm{d}x.$$

</div>

## Existence and Non-Existence of the Riemann Integral

We begin with two non-existence results. Recall that for $M \subset \mathbb{R}$ a function $f\colon M \to \mathbb{R}$ is **bounded** if $\exists\, c\; \forall\, x \in M\colon \|f(x)\| < c$. Else $f$ is **unbounded**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8</span><span class="math-callout__name">(Unbounded Functions Are Bad)</span></p>

If the function $f\colon [a, b] \to \mathbb{R}$ is unbounded then $f \notin \mathrm{R}(a, b)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We suppose that $f\colon [a, b] \to \mathbb{R}$ is unbounded and show that for every $n$ there exists a partition $P$ of $[a, b]$ with test points $\bar{t}$ such that $\Delta(P) < 1/n$ and $|R(P, \bar{t}, f)| > n$. This refutes Cauchy's condition for the Riemann integrability of $f$.

It follows from the unboundedness of $f$ and from the compactness of $[a, b]$ that there is a convergent sequence $(b_n) \subset [a, b]$ with $\lim b_n = \alpha \in [a, b]$ and with $\lim \|f(b_n)\| = +\infty$. Let an $n \in \mathbb{N}$ be given. For $P$ we take any partition $P = (a_0, \dots, a_k)$ of $[a, b]$ with $\Delta(P) < 1/n$ and such that there is a *unique* index $j \in \lbrace 1, \dots, k \rbrace$ for which $\alpha \in [a_{j-1}, a_j]$. Then we select arbitrary test points $t_i \in [a_{i-1}, a_i]$ for all $i \neq j$ and consider the incomplete Riemann sum $s := \sum_{i=1, i \neq j}^{k} (a_i - a_{i-1})f(t_i)$. Now we can select the remaining test point $t_j \in [a_{j-1}, a_j]$ so that $\|(a_j - a_{j-1})f(t_j)\| > \|s\| + n$ (because $b_n \in [a_{j-1}, a_j]$ for every large enough $n$). We then define $\bar{t}$ as consisting of all these test points and get (by the triangle inequality $\|u + v\| \ge \|u\| - \|v\|$) that $\|R(P, \bar{t}, f)\| \ge \|(a_j - a_{j-1})f(t_j)\| - \|s\| > n$, as required. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 9</span><span class="math-callout__name">(So Are Too Discontinuous Functions)</span></p>

If the function $f\colon [a, b] \to \mathbb{R}$ is discontinuous at every point of some subinterval $[c, d] \subset [a, b]$ with $c < d$, then $f \notin \mathrm{R}(a, b)$.

</div>

For example, since Dirichlet's function $d\colon [0, 1] \to \lbrace 0, 1 \rbrace$, given by $d(x) = 0$ for rational $x$ and $d(x) = 1$ for irrational $x$, is discontinuous everywhere, it is not Riemann integrable. To prove Proposition 9 in its generality we need the following result of independent interest.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10</span><span class="math-callout__name">(Baire's Theorem)</span></p>

If $a < b$ are real numbers and $[a, b] = \bigcup_{n=1}^{\infty} M_n$ then some of the sets $M_n$ is not sparse.

</div>

Here a set $M \subset [a, b]$ is **sparse** (in $[a, b]$) if for every neighborhood $U(c, \varepsilon)$ with $c \in [a, b]$ there is a neighborhood $U(d, \delta) \subset U(c, \varepsilon) \cap [a, b]$ such that $U(d, \delta) \cap M = \emptyset$.

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We suppose that in the countable union $[a, b] = \bigcup_{n=1}^{\infty} M_n$ every set $M_n$ is sparse and deduce a contradiction. Since $M_1$ is sparse, there is a subinterval $[a_1, b_1] \subset [a, b]$ such that $a_1 < b_1$ and $[a_1, b_1] \cap M_1 = \emptyset$. Since $M_2$ is sparse, there is a subinterval $[a_2, b_2] \subset [a_1, b_1]$ such that $a_2 < b_2$ and $[a_2, b_2] \cap M_2 = \emptyset$. Continuing this way we obtain a sequence of nested intervals $[a, b] \supset [a_1, b_1] \supset [a_2, b_2] \supset \cdots \supset [a_n, b_n] \supset \dots$ such that for every $n$, $a_n < b_n$ and $[a_n, b_n] \cap M_n = \emptyset$. Let $\alpha := \lim a_n \in [a, b]$. This limit exists and lies in $[a, b]$ because the sequence $(a_n)$ is non-decreasing and is bounded from below by $a$ and from above by $b$. In fact, $a_n < b_m$ for every $n$ and every $m$, which implies that $\alpha \in [a_n, b_n]$ for every $n$. But this means that $\alpha \notin M_n$ for every $n$, which is a contradiction as $\alpha \in [a, b]$. $\square$

</details>
</div>

There is a powerful criterion — Lebesgue's theorem below — by which one usually easily determines if the given function is Riemann integrable or not. To state it we need two definitions. For any function $f\colon M \to \mathbb{R}$, $M \subset \mathbb{R}$, we define

$$\mathrm{DC}(f) := \lbrace x \in M \mid f \text{ is discontinuous at } x \rbrace.$$

We say that a set $M \subset \mathbb{R}$ **has measure** $0$ if for every $\varepsilon$ there exist intervals $[a_n, b_n]$, $n \in \mathbb{N}$ and $a_n < b_n$, such that

$$M \subset \bigcup_{n=1}^{\infty} [a_n, b_n] \quad \text{and} \quad \sum_{n=1}^{\infty} (b_n - a_n) < \varepsilon.$$

It is easy to see that every at most countable set has measure 0, that any countable union of measure 0 sets has measure 0, that any subset of a measure 0 set has measure 0 and that no nontrivial interval has measure 0.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 11</span><span class="math-callout__name">(Lebesgue's Theorem)</span></p>

For any $f\colon [a, b] \to \mathbb{R}$,

$$f \in \mathrm{R}(a, b) \;\iff\; f \text{ is bounded and } \mathrm{DC}(f) \text{ has measure } 0.$$

</div>

Lebesgue's theorem implies closedness of the class of Riemann integrable functions to several operations.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 12</span><span class="math-callout__name">(Nice Operations for $\mathrm{R}(a, b)$)</span></p>

The following implications hold.

1. $f, g \in \mathrm{R}(a, b) \Rightarrow cf + dg \in \mathrm{R}(a, b)$ for any $c, d \in \mathbb{R}$.
2. $f, g \in \mathrm{R}(a, b) \Rightarrow f \cdot g \in \mathrm{R}(a, b)$.
3. If $g\colon [a, b] \to M \subset \mathbb{R}$, $f\colon M \to \mathbb{R}$, $g \in \mathrm{R}(a, b)$ and $f$ is continuous and bounded, then $f(g) \in \mathrm{R}(a, b)$.
4. If $g\colon [c, d] \to [a, b]$, $f\colon [a, b] \to \mathbb{R}$, $g$ is continuous and $f \in \mathrm{R}(a, b)$, then $f(g) \in \mathrm{R}(c, d)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**1.** We suppose that $f, g\colon [a, b] \to \mathbb{R}$ are Riemann integrable. Hence $f$ and $g$ are bounded and so is $cf + dg$. Since $\mathrm{DC}(cf + dg) \subset \mathrm{DC}(f) \cup \mathrm{DC}(g)$ and the latter two sets have measure 0, so has the former set.

**2.** This proof is similar to the previous one, we only replace the operation of linear combination with multiplication.

**3.** Since $f$ is bounded, so is the composition $f(g)$. Since the inclusion $\mathrm{DC}(f(g)) \subset \mathrm{DC}(g)$ holds and the latter set has measure 0, so has the former set.

**4.** This proof is similar to the previous one, the only change is the inclusion $\mathrm{DC}(f(g)) \subset \mathrm{DC}(f)$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 13</span><span class="math-callout__name">(Continuous Functions Are R. Integrable)</span></p>

If $f\colon [a, b] \to \mathbb{R}$ is continuous then $f \in \mathrm{R}(a, b)$.

</div>

This follows immediately from Theorem 11 because any continuous function defined on a compact set is bounded and has $\mathrm{DC}(f) = \emptyset$. But we also proved it directly already in Corollary 3 of Lecture 10.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 14</span><span class="math-callout__name">(Monotone Functions Are R. Integrable)</span></p>

If $f\colon [a, b] \to \mathbb{R}$ is monotone then $f \in \mathrm{R}(a, b)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We assume that $f$ is non-decreasing, the case with non-increasing $f$ is similar. We first deduce this theorem from Lebesgue's, and then give a direct proof.

The function $f$ is bounded because $f(a) \le f(x) \le f(b)$ for every $x \in [a, b]$. We define an injection $\varphi\colon \mathrm{DC}(f) \to \mathbb{Q}$. This proves that $\mathrm{DC}(f)$ is at most countable, therefore has measure 0 and $f \in \mathrm{R}(a, b)$ by Lebesgue's theorem. If $p \in \mathrm{DC}(f)$ then by the monotonicity of $f$ both one-sided limits $l(p) := \lim_{x \to p^-} f(x)$ and $r(p) := \lim_{x \to p^+} f(x)$ exist, are finite, $l(p) \le f(p) \le r(p)$ and at least one of the two inequalities is strict. We define $\varphi(p)$ to be any fraction in $(l(p), r(p)) \cap \mathbb{Q}$. It is easy to see that $\varphi(p) < \varphi(q)$ for any $p < q$ in $\mathrm{DC}(f)$.

We prove directly that $f \in \mathrm{R}(a, b)$ by proving for $f$ Cauchy's condition of Proposition 2. Let $P = (a_0, \dots, a_k)$ and $Q = (b_0, \dots, b_l)$ be two partitions of $[a, b]$ with respective test points $\bar{t}$ and $\bar{u}$ and let an $\varepsilon$ be given. We set $\delta := +\infty$ for $f(a) = f(b)$ (when $f$ is a constant function) and else set $\delta := \varepsilon/2(f(b) - f(a))$. We assume additionally that $P \subset Q$, i.e., $a_0 = b_{i_0} = a$, $a_1 = b_{i_1}$, ..., $a_k = b_{i_k} = b$ for some indices $i_0 = 0 < i_1 < \cdots < i_k = l$. As earlier, we reduce general partitions $P$ and $Q$ to this case. Let $k = 1$. Then, since $f$ is non-decreasing on $[a, b]$, $R(P, \bar{t}, f) - R(Q, \bar{u}, f)$ is at least $(a_1 - a_0)f(a_0) - \sum_{i=1}^l (b_i - b_{i-1})f(b_l) = (b - a)(f(a) - f(b))$ and, similarly, at most $(b - a)(f(b) - f(a))$. So for $k = 1$, $|R(P, \bar{t}, f) - R(Q, \bar{u}, f)| \le (b - a) \cdot (f(b) - f(a))$. For general $k$ we use this bound for any partition $a_{r-1} = b_{i_{r-1}} < b_{i_{r-1}+1} < \cdots < b_{i_r} = a_r$ of the interval $[a_{r-1}, a_r]$, $r = 1, 2, \dots, k$, thus with $a$ replaced by $a_{r-1}$ and $b$ by $a_r$. If $\Delta(P) < \delta$ (hence $\Delta(Q) < \delta$ too) then by the triangle inequality,

$$|R(P, \bar{t}, f) - R(Q, \bar{u}, f)| \le \sum_{r=1}^{k} (a_r - a_{r-1}) \cdot (f(a_r) - f(a_{r-1})) \le \frac{\varepsilon}{2(f(b) - f(a))} \sum_{r=1}^{k} (f(a_r) - f(a_{r-1})) = \frac{\varepsilon}{2(f(b) - f(a))} \cdot (f(b) - f(a)) = \varepsilon/2.$$

If $P$ and $Q$ are general partitions of $[a, b]$ with respective test points $\bar{t}$ and $\bar{u}$ and with $\Delta(P), \Delta(Q) < \delta$, we set $R := P \cup Q$ (then also $\Delta(R) < \delta$) and take arbitrary test points $\bar{v}$ of $R$. Since $P \subset R$ and $Q \subset R$, we get by the previous case that $|R(P, \bar{t}, f) - R(Q, \bar{u}, f)| \le |R(P, \bar{t}, f) - R(R, \bar{v}, f)| + |R(R, \bar{v}, f) - R(Q, \bar{u}, f)| < \varepsilon/2 + \varepsilon/2 = \varepsilon$. $\square$

</details>
</div>

## Comparison of Riemann and Newton Integrals

We revisit the relation between the Riemann integral and primitive functions that we considered in Lecture 10. We proved there in Corollary 4 that for continuous $f$, $\text{(R)} \int_a^b f = \text{(N)} \int_a^b f$. Now we extend it to a more general situation. In the proof of the next theorem, which is known as the **Second Fundamental Theorem of Calculus**, we again rely on Lagrange's mean value theorem.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 15</span><span class="math-callout__name">(FTC 2)</span></p>

Let $f\colon (a, b) \to \mathbb{R}$, where $a < b$ are real numbers, have a primitive function $F\colon (a, b) \to \mathbb{R}$ and let $f \in \mathrm{R}(a, b)$ (see Definition 4). Then there exist finite limits $F(a) := \lim_{x \to a} F(x)$ and $F(b) := \lim_{x \to b} F(x)$ and

$$\text{(R)} \int_a^b f = F(b) - F(a) = \text{(N)} \int_a^b f.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We extend $f$ arbitrarily to $f\colon [a, b] \to \mathbb{R}$, assume that $f \in \mathrm{R}(a, b)$ and consider the primitive function $F$ of $f$ on $(a, b)$. We first prove that the limits $F(a) := \lim_{x \to a} F(x)$ and $F(b) := \lim_{x \to b} F(x)$ exist and are finite. For it we show that $F$ is uniformly continuous on $(a, b)$ (in fact, even Lipschitz continuous). Since $f$ is bounded by Proposition 8, we may take a bounding constant $C > 0$. Lagrange's mean value theorem implies that for any subinterval $[c, d] \subset (a, b)$ with $c < d$ there is a point $e \in (c, d)$ such that $F(d) - F(c) = f(e) \cdot (d - c)$. Thus $|F(d) - F(c)| = |f(e)| \cdot |d - c| < C|d - c|$ and $F$ is uniformly continuous on $(a, b)$.

Next we show that $F(b) - F(a) = \text{(R)} \int_a^b f$. Let an $\varepsilon$ be given. We may take such numbers $c < d$ in $(a, b)$ that $|F(a) - F(c)|, |F(b) - F(d)| < \varepsilon$, $C|a - c|, C|b - d| < \varepsilon$ and that there is a partition $P = (a_0, \dots, a_k)$ of $[a, b]$ such that $a_1 = c$, $a_{k-1} = d$ and that for any test points $\bar{t}$ of $P$, $|\int_a^b f - R(P, \bar{t}, f)| < \varepsilon$. From Lecture 10 we know that there exist test points $\bar{e}$ of the restriction of $P$ to $[c, d] = [a_1, a_{k-1}]$ such that $F(d) - F(c) = \sum_{i=2}^{k-1}(a_i - a_{i-1}) \cdot f(e_i)$. We define the test points $\bar{u}$ of $P$ as consisting of $\bar{e}$ and of two arbitrary test points $u_1$ and $u_k$ in the respective intervals $[a, a_1] = [a, c]$ and $[a_{k-1}, b] = [d, b]$. Then

$$\left|\text{(R)} \int_a^b f - (F(b) - F(a))\right| \le \left|\text{(R)} \int_a^b f - R(P, \bar{u}, f)\right| + \left|R(P, \bar{u}, f) - (F(d) - F(c))\right| + \left|(F(d) - F(c)) - (F(b) - F(a))\right|$$
$$\le \varepsilon + |(c - a) \cdot f(u_1) + (b - d) \cdot f(u_k)| + |F(d) - F(b)| + |F(a) - F(c)| < \varepsilon + 2\varepsilon + 2\varepsilon = 5\varepsilon.$$

But $\varepsilon > 0$ may be arbitrarily small, so $\text{(R)} \int_a^b f = F(b) - F(a)$. $\square$

</details>
</div>

The *First Fundamental Theorem of Calculus* is as follows. A function $f\colon M \to \mathbb{R}$, $M \subset \mathbb{R}$, is **Lipschitz continuous** if there is a constant $C > 0$ such that $\forall\, x, y \in M\colon \|f(x) - f(y)\| \le C\|x - y\|$. It is a property stronger than continuity or even than uniform continuity; every Lipschitz continuous function is uniformly continuous.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 16</span><span class="math-callout__name">(FTC 1)</span></p>

Let $f \in \mathrm{R}(a, b)$. Then $f \in \mathrm{R}(a, x)$ for every $x \in (a, b]$ and the function $F\colon [a, b] \to \mathbb{R}$, given by

$$F(x) := \int_a^x f,$$

is Lipschitz continuous. Moreover, it is such that $F'(x) = f(x)$ for every point $x \in [a, b]$ of continuity of $f$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

So let $f \in \mathrm{R}(a, b)$. By Proposition 5, $f$ is Riemann integrable on any subinterval $[a', b']$, $a' < b'$, of $[a, b]$. So $F$ is correctly defined and $F(a) = 0$. Since $f$ is bounded (by Proposition 8), we may take a bounding constant $c > 0$. We set $C := 1 + c$. Let $x < y$ be in $[a, b]$ and, by Definition 1, let $P$ be a partition of $[x, y]$ with test points $\bar{t}$ such that $|\int_x^y f - R(P, \bar{t}, f)| < y - x$. By Proposition 5 and the definition of $F$,

$$|F(y) - F(x)| = \left|\int_x^y f\right| \le y - x + |R(P, \bar{t}, f)| \le y - x + c(y - x) = C|y - x|$$

and $|F(y) - F(x)| \le C|y - x|$. Thus $F$ is Lipschitz continuous.

We prove the second part about the derivative of $F$. Let $x_0$ in $[a, b]$ be such that $f$ is continuous at $x_0$ and let an $\varepsilon$ be given. We take a $\delta$ such that $x \in U(x_0, \delta) \cap [a, b] \Rightarrow f(x) \in U(f(x_0), \varepsilon)$. Let $x \in P(x_0, \delta) \cap [a, b]$ be arbitrary, say $x > x_0$ (in the case that $x < x_0$ the argument is similar). Then by taking a partition $P$ of $[x_0, x]$ with test points $\bar{t}$ and such that $|\int_{x_0}^x f - R(P, \bar{t}, f)| < \varepsilon(x - x_0)$ we see that

$$\frac{F(x) - F(x_0)}{x - x_0} - f(x_0) = \frac{1}{x - x_0} \int_{x_0}^x f - f(x_0)$$

is less than $\frac{R(P, \bar{t}, f) + \varepsilon(x - x_0)}{x - x_0} - f(x_0) < \frac{(x - x_0)(f(x_0) + \varepsilon) + \varepsilon(x - x_0)}{x - x_0} - f(x_0) = 2\varepsilon$, and similarly it is also $> -2\varepsilon$. Thus $F'(x_0) = f(x_0)$. $\square$

</details>
</div>

As an immediate corollary of FTC 1 we obtain another proof of the last theorem of Lecture 9 that every continuous function has a primitive function.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 17</span><span class="math-callout__name">(Existence of Primitives)</span></p>

Any continuous function $f\colon [a, b] \to \mathbb{R}$ has a primitive function.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

If $f\colon [a, b] \to \mathbb{R}$ is continuous then $f \in \mathrm{R}(a, b)$ by Theorem 13. By the previous theorem, $\int_a^x f$ is a primitive of $f(x)$ on $[a, b]$. $\square$

</details>
</div>

It is easy to give examples of functions $f\colon (a, b) \to \mathbb{R}$ that are Riemann integrable but are not Newton integrable, and vice versa. For example, $\text{(R)} \int_{-1}^1 \operatorname{sgn} = \text{(R, N)} \int_{-1}^0 \operatorname{sgn} + \text{(R, N)} \int_0^1 \operatorname{sgn} = [-x]_{-1}^0 + [x]_0^1 = -1 + 1 = 0$, but $\text{(N)} \int_{-1}^1 \operatorname{sgn}$ is not defined because $\operatorname{sgn}(x)$ does not have a primitive function on $(-1, 1)$, it is not Darboux there. On the other hand, $\text{(N)} \int_0^1 1/\sqrt{x} = [2\sqrt{x}]_0^1 = 2$ but the integral $\text{(R)} \int_0^1 1/\sqrt{x}$ does not exist because the integrand is unbounded on the interval $(0, 1)$, see Proposition 8.

This discrepancy can be fixed by using more general primitives. One says that $F\colon I \to \mathbb{R}$ is a **generalized primitive function** of $f\colon I \to \mathbb{R}$, where $I$ is a nontrivial real interval, if $F$ is continuous and $F'(x) = f(x)$ holds for every $x \in I$, up to finitely many exceptions $x$. One then defines the **extended general Newton integral** of $f\colon (A, B) \to \mathbb{R}$ by setting $\text{(N}_\text{e}\text{)} \int_A^B f := [F]_A^B$ for any generalized primitive $F$ of $f$ on $(A, B)$. Now

$$\text{(N}_\text{e}\text{)} \int_{-1}^1 \operatorname{sgn}(x) = [\,|x|\,]_{-1}^1 = 1 - 1 = 0.$$

# Lecture 13 — The Riemann Integral and Its Upgrade the Henstock–Kurzweil Integral. Use of Integrals

## The Riemann Integral After J.-G. Darboux

We give another equivalent definition of the Riemann integral. For real numbers $a < b$ and for a partition $P = (a_0, a_1, \dots, a_k)$ of the interval $[a, b]$ we denote $I_i := [a_{i-1}, a_i]$ and $\|I_i\| := a_i - a_{i-1}$. For a function $f\colon [a, b] \to \mathbb{R}$, the sums

$$s(P, f) := \sum_{i=1}^{k} |I_i| \cdot \inf(f[I_i]) \quad \text{and} \quad S(P, f) := \sum_{i=1}^{k} |I_i| \cdot \sup(f[I_i]),$$

$s(P, f) \in \mathbb{R} \cup \lbrace -\infty \rbrace$ and $S(P, f) \in \mathbb{R} \cup \lbrace +\infty \rbrace$ (infima and suprema are taken in $(\mathbb{R}^*, <)$), are called the **lower** and the **upper sum** (for $P$ and $f$), respectively.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1</span><span class="math-callout__name">(Monotonicity of Lower and Upper Sums)</span></p>

Let $P \subset Q$ be partitions of the interval $[a, b]$ and let $f\colon [a, b] \to \mathbb{R}$. Then

$$s(P, f) \le s(Q, f) \quad \text{and} \quad S(P, f) \ge S(Q, f).$$

</div>

We prove equivalence of the fourth definition of the Riemann integral.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2</span><span class="math-callout__name">(4th Definition of the R. $\int$)</span></p>

Let $a < b$ be real numbers and $f\colon [a, b] \to \mathbb{R}$. Then

$$f \in \mathrm{R}(a, b) \;\iff\; \exists\, c\; \forall\, \varepsilon\; \exists\, P\; \forall\, \bar{t}\colon\; |c - R(P, \bar{t}, f)| < \varepsilon.$$

</div>

Let $a < b$ be real numbers and let $\mathcal{D} = \mathcal{D}(a, b)$ denote the set of all partitions of the interval $[a, b]$. Then

$$\underline{\int_a^b} f := \sup(\lbrace s(P, f) \mid P \in \mathcal{D} \rbrace) \in \mathbb{R}^*$$

and

$$\overline{\int_a^b} f := \inf(\lbrace S(P, f) \mid P \in \mathcal{D} \rbrace) \in \mathbb{R}^*$$

(infima and suprema again taken in $(\mathbb{R}^*, <)$) is the so-called **lower** and **upper integral** (of $f$ over $[a, b]$), respectively.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3</span><span class="math-callout__name">($\underline{\int} \le \overline{\int}$)</span></p>

Let $f\colon [a, b] \to \mathbb{R}$ be a function. Then for every two partitions $P, Q \in \mathcal{D}(a, b)$,

$$s(P, f) \le \underline{\int_a^b} f \le \overline{\int_a^b} f \le S(Q, f).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $f$ be as stated, and let $P$ and $Q$ be partitions of the interval $[a, b]$. We already know the trick with $R := P \cup Q$. Then $P, Q \subset R$ and, by Proposition 1,

$$s(P, f) \le s(R, f) \le S(R, f) \le S(Q, f) \quad \text{so}\quad s(P, f) \le S(Q, f).$$

We now use the fact that in any linear order $(X, \prec)$, for every two sets $A, B \subset X$ with $A \preceq B$ we have $\sup(A) \preceq \inf(B)$, if these elements exist. Every $a \in A$ is a lower bound of the set $B$, so $A \preceq \lbrace \inf(B) \rbrace$. Thus $\inf(B)$ is an upper bound of the set $A$ and $\sup(A) \preceq \inf(B)$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4</span><span class="math-callout__name">(Riemann = Darboux)</span></p>

A function $f$ from $[a, b]$ to $\mathbb{R}$ is

$$f \in \mathrm{R}(a, b) \;\iff\; \underline{\int_a^b} f = \overline{\int_a^b} f \in \mathbb{R}.$$

In the positive case, $\text{(R)} \int_a^b f = \underline{\int_a^b} f = \overline{\int_a^b} f$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**Implication $\Rightarrow$.** Let $f \in \mathrm{R}(a, b)$. Then $f$ is bounded and infima in $s(P, f)$ and suprema in $S(P, f)$ are finite. We can thus approximate them arbitrarily closely by functional values and get that for every $\varepsilon$ and every $P \in \mathcal{D}(a, b)$ there are test points $\bar{t}$ of $P$ such that $|s(P, f) - R(P, \bar{t}, f)| < \varepsilon$, and that for every $\varepsilon$ and every $P \in \mathcal{D}(a, b)$ there are test points $\bar{t}$ such that $|S(P, f) - R(P, \bar{t}, f)| < \varepsilon$. Hence, by Proposition 3 here and Definition 1 in the last lecture, the implication and the last part of the statement follow.

**Implication $\Leftarrow$.** Let $I := \underline{\int_a^b} f = \overline{\int_a^b} f \in \mathbb{R}$, so $f$ is bounded, and let an $\varepsilon$ be given. By this assumption and by Proposition 3 we take $P, Q \in \mathcal{D}(a, b)$ such that $s(P, f) \le I \le S(Q, f)$ and $0 \le S(Q, f) - s(P, f) < \varepsilon$. We put $R := P \cup Q$ and take arbitrary test points $\bar{v}$ of $R$. By Propositions 1 and 3, $s(P, f) \le s(R, f) \le I$, $R(R, \bar{t}, f) \le S(R, f) \le S(Q, f)$ and thus also $|R(R, \bar{t}, f) - I| < \varepsilon$ and $f \in \mathrm{R}(a, b)$ by Proposition 2. $\square$

</details>
</div>

## The Henstock–Kurzweil Integral — The Correct Definition of the Riemann Integral

Last time we saw that $\text{(N)} \int_0^1 1/\sqrt{x} = 2$, but that $\text{(R)} \int_0^1 1/\sqrt{x}$ does not exist, because the integrand is unbounded. The inability of the Riemann integral to integrate unbounded functions is its serious shortcoming. In 1957 the Czech mathematician *Jaroslav Kurzweil (1926–2022)* and a little later the English mathematician *Ralph Henstock (1923–2007)* modified the condition $\Delta(P) < \delta$ and improved the Riemann integral to be able to integrate unbounded functions.

Let $I \subset \mathbb{R}$ be an interval. We call each function $\delta_c\colon I \to (0, +\infty)$ a **gauge** (on $I$). A partition $P = (a_0, \dots, a_k)$ of the interval $[a, b]$ and its test points $\bar{t} = (t_1, \dots, t_k)$, $t_i \in [a_{i-1}, a_i]$, are $\delta_c$**-fine** if

$$\forall\, i = 1, 2, \dots, k\colon\; a_i - a_{i-1} < \delta_c(t_i).$$

For example, if $\Delta(P) < \delta$, then the partition $P$ together with any test points $\bar{t}$ are $\delta_c$-fine for the constant gauge $\delta_c = \delta$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5</span><span class="math-callout__name">(Cousin's Lemma)</span></p>

Let $a < b$ be in $\mathbb{R}$. For every gauge $\delta_c\colon [a, b] \to (0, +\infty)$ there exist $\delta_c$-fine partition $P \in \mathcal{D}(a, b)$ with test points $\bar{t}$. Even every finite system $[a_i, b_i]$, $i \in I$, of mutually disjoint subintervals $[a_i, b_i] \subset [a, b]$ with test points $t_i \in [a_i, b_i]$, for which $b_i - a_i < \delta_c(t_i)$ for $\forall\, i \in I$, can be completed to a $\delta_c$-fine partition of $[a, b]$ with test points $\bar{t}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (abridged)</summary>

The set $M := [a, b] \setminus \bigcup_{i \in I}(a_i, b_i)$ is compact and therefore we can select from its (open) cover $M \subset \bigcup_{x \in M} U(x, \delta_c(x)/2)$ a finite subcover $U(x_i, \delta_c(x_i)/2)$, $i = 1, 2, \dots, n$. We add to the intervals $[a_i, b_i]$, $i \in I$, suitable closed subintervals of the intervals $(x_i - \delta_c(x_i), x_i + \delta_c(x_i))$ (containing the corresponding point $x_i$) and obtain a partition of $[a, b]$. The obtained test points $\bar{t}$ are the $t_i$, $i \in I$, and $x_1, \dots, x_n$. The result is $\delta_c$-fine. $\square$

</details>
</div>

The definition of the Henstock–Kurzweil integral follows. The previous proposition shows that the implication in it can always be satisfied non-vacuously, by a valid assumption.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6</span><span class="math-callout__name">(Henstock–Kurzweil Integral)</span></p>

A function $f\colon [a, b] \to \mathbb{R}$ is **Henstock–Kurzweil integrable**, symbolically written $f \in \mathrm{HK}(a, b)$, if there is a number $L \in \mathbb{R}$ such that for $\forall\, \varepsilon$ $\exists\, \delta_c$, where $\delta_c$ is a gauge on $[a, b]$, such that for every partition $P$ of $[a, b]$ and test points $\bar{t}$ of $P$ it holds that

$$P \text{ and } \bar{t} \text{ are } \delta_c\text{-fine} \;\Rightarrow\; |R(P, \bar{t}, f) - L| < \varepsilon.$$

Then we also write $\text{(HK)} \int_a^b f = L$ or $\text{(HK)} \int_a^b f(x)\,\mathrm{d}x = L$ and say that the **Henstock–Kurzweil integral** of the function $f$ over the interval $[a, b]$ equals $L$.

</div>

It is clear from the definition that $\mathrm{R}(a, b) \subset \mathrm{HK}(a, b)$. The following theorem shows that the Henstock–Kurzweil integral is finally the right partner for the Newton integral.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 7</span><span class="math-callout__name">(HK. $\int$ and N. $\int$)</span></p>

Let $a < b$ be in $\mathbb{R}$, $F\colon [a, b] \to \mathbb{R}$ be a continuous function and let $F' = f$ on $(a, b)$ (the values $f(a)$ and $f(b)$ are arbitrary). Then $f \in \mathrm{HK}(a, b)$ and

$$\text{(HK)} \int_a^b f = F(b) - F(a) = \text{(N)} \int_a^b f.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $\varepsilon$ and $x \in (a, b)$ be given. Due to the equality $F'(x) = f(x)$ there is a value $\delta_c(x) > 0$ such that for every $y \in [a, b]$,

$$y \in U(x, \delta_c(x)) \;\Rightarrow\; |F(y) - F(x) - f(x)(y - x)| \le \varepsilon|y - x|. \quad (*)$$

Moreover, there exist values $\delta_c(a) > 0$ and $\delta_c(b) > 0$ such that $|f(a)\delta_c(a)|, |f(b)\delta_c(b))| < \varepsilon$ and that $|F(y) - F(a)|, |F(z) - F(b)| < \varepsilon$ for every $y \in [a, a + \delta_c(a))$ and every $z \in (b - \delta_c(b), b]$.

If the partition $P = (a_0, \dots, a_k) \in \mathcal{D}(a, b)$ with test points $\bar{t}$ are $\delta_c$-fine, then for every test point in an interval $[a_{i-1}, a_i]$, with $t_i \neq a, b$, one has that

$$|F(a_i) - F(a_{i-1}) - f(t_i)(a_i - a_{i-1})| \stackrel{\Delta\text{-ineq.}}{\le} |F(a_i) - F(t_i) - f(t_i)(a_i - t_i)| + |F(t_i) - F(a_{i-1}) - f(t_i)(t_i - a_{i-1})| \stackrel{(*)}{\le} \varepsilon|a_i - t_i| + \varepsilon|t_i - a_{i-1}| = \varepsilon(a_i - a_{i-1}).$$

If $t_i \in [a_{i-1}, a_i]$ and $t_i \in \lbrace a, b \rbrace$, then $|F(a_i) - F(a_{i-1}) - f(t_i)(a_i - a_{i-1})| < 2\varepsilon$ because $i = 1$ and $t_1 = a$ or $i = k$ and $t_k = b$. According to these two estimates,

$$|F(b) - F(a) - R(P, \bar{t}, f)| \stackrel{\Delta\text{-ineq.}}{\le} \sum_{i=1}^{k} |F(a_i) - F(a_{i-1}) - (a_i - a_{i-1})f(t_i)| < \varepsilon(b - a) + 4\varepsilon$$

so that $F(b) - F(a) = \text{(HK)} \int_a^b f$. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 8</span><span class="math-callout__name">(HK Integrates $1/\sqrt{x}$)</span></p>

Let $1/\sqrt{0} := 1$. Then $\text{(HK)} \int_0^1 1/\sqrt{x} = 2$.

</div>

## Integration by Parts and by Substitution for $\text{(R)} \int_a^b f$

We present the third version of these two integration formulae. The first one was for primitive functions, the second one for the Newton integral, and this one is for the Riemann integral. Substitution now turns out to be surprisingly non-trivial. In the following theorem, the values $f(a)$, $f(b)$, $g(a)$ and $g(b)$ are arbitrary.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9</span><span class="math-callout__name">(Integration by Parts for R. $\int$)</span></p>

Let $a < b$ be in $\mathbb{R}$, let the functions $F, G, f, g\colon (a, b) \to \mathbb{R}$ satisfy on $(a, b)$ that $F' = f$ and $G' = g$, and let $Fg, fG \in \mathrm{R}(a, b)$. Then the equality holds that

$$\int_a^b Fg = [FG]_a^b - \int_a^b fG.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By the linearity of the Riemann integral also $fG + Fg \in \mathrm{R}(a, b)$. From $(FG)' = fG + Fg$ on $(a, b)$ and from FTC 2 (Theorem 15 of Lecture 12) we have that

$$\text{(R)} \int_a^b fG + \text{(R)} \int_a^b Fg = \text{(R)} \int_a^b (fG + Fg) = \text{(N)} \int_a^b (fG + Fg) = [FG]_a^b,$$

which is a rearrangement of the stated equality. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 10</span><span class="math-callout__name">(R. $\int$ by Substitution)</span></p>

Let $G\colon [a, b] \to \mathbb{R}$ have on $[a, b]$ continuous derivative $G'$ and let $f\colon G[[a, b]] \to \mathbb{R}$ be continuous. Then the equality of Riemann integrals holds that

$$\int_{G(a)}^{G(b)} f = \int_a^b f(G) G'.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

For $x \in G[[a, b]]$ we consider the function $F(x) := \int_{G(a)}^x f$ (by the results of the last lecture it is well defined). According to FTC 1 (Theorem 16 of Lecture 12) and derivatives of composite functions, the function $F(G)$ is on $[a, b]$ a primitive function of $f(G)G'$. By FTC 2 (Theorem 15 of Lecture 12) and definition of $F$ ($F(G(a)) = 0$)

$$\int_a^b f(G) G' = [F(G)]_a^b = F(G(b)) - F(G(a)) = \int_{G(a)}^{G(b)} f. \quad \square$$

</details>
</div>

A theorem on substitution directly for the Riemann integral, with an equivalence for Riemann integrability, was proven by H. Kestelman only in 1961. We present here an improved version due to the Czech mathematicians D. Preiss and J. Uher in 1970.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 11</span><span class="math-callout__name">(D. Preiss and J. Uher, 1970)</span></p>

Let $g \in \mathrm{R}(a, b)$, for $x \in [a, b]$ let $G(x) := \int_a^x g$ and let $f\colon G[[a, b]] \to \mathbb{R}$ be bounded. Then $f$ is Riemann integrable on the interval $G[[a, b]]$ if and only if $f(G)g \in \mathrm{R}(a, b)$, and in the positive case the equality of Riemann integrals holds that

$$\int_{G(a)}^{G(b)} f = \int_a^b f(G)g.$$

</div>

## Use of Integrals in Formulas for Lengths, Areas and Volumes

We denote by the symbol $\|uv\|$ (always $\ge 0$) the length of the straight segment with endpoints $u, v \in \mathbb{R}^2$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 12</span><span class="math-callout__name">(Length of $G_f$)</span></p>

We say that $f\colon [a, b] \to \mathbb{R}$ has **rectifiable graph** if the supremum

$$\ell(f) := \sup\!\left(\left\lbrace \sum_{i=1}^{k} \bigl|(a_{i-1}, f(a_{i-1}))\,(a_i, f(a_i))\bigr| \;\middle|\; (a_0, \dots, a_k) \in \mathcal{D}(a, b) \right\rbrace\right)$$

is finite. The number $\ell(f)$ is then called the **length of the graph** of the function $f$.

</div>

This supremum is actually the supremum of lengths of broken lines inscribed in the graph $G_f$ of $f$. This formula can be extended to curves of the form $\varphi\colon [a, b] \to \mathbb{R}^n$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 13</span><span class="math-callout__name">(Length of $G_f$)</span></p>

Suppose that $f\colon [a, b] \to \mathbb{R}$ is a continuous function that has on $(a, b)$ finite derivative $f' \in \mathrm{R}(a, b)$. Then $f$ has a rectifiable graph with length

$$\ell(f) = \int_a^b \sqrt{1 + (f')^2}.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $g := \sqrt{1 + (f')^2}$. By the results of the last lecture the Riemann integral $\int_a^b g$ exists. The sum in Definition 12, which we denote as $K(P, f)$, does not decrease under subdivision of $P = (a_0, \dots, a_k)$, so for any sequence $(P_n) \subset \mathcal{D}(a, b)$ with $\lim \Delta(P_n) = 0$ one has that $\lim K(P_n, f) = \ell(f)$ or, for a non-rectifiable graph, this limit is always $+\infty$. But

$$K(P, f) = \sum_{i=1}^{k} (a_i - a_{i-1})\sqrt{1 + [(f(a_i) - f(a_{i-1}))/(a_i - a_{i-1})]^2}$$

and by the Lagrange mean value theorem, $\frac{f(a_i) - f(a_{i-1})}{a_i - a_{i-1}} = f'(t_i)$ for some $t_i \in (a_{i-1}, a_i)$. Let us denote these test points as $\bar{t}$. So for $(P_n)$ as above,

$$\int_a^b g = \lim_{n \to \infty} R(P_n, \overline{t(n)}, g) = \lim_{n \to \infty} K(P_n, f) = \ell(f). \quad \square$$

</details>
</div>

We did not define areas of planar regions independently of the integral, the same for the volume in $\mathbb{R}^3$, and so the following two formulas are — at least in our lectures — unlike the length of the graph only at the level of definitions.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 14</span><span class="math-callout__name">(Area Between Two Graphs)</span></p>

Let $f, g \in \mathrm{R}(a, b)$ and $f \le g$ on $[a, b]$. Then

$$\text{area}\bigl(\lbrace(x, y) \in \mathbb{R}^2 \mid x \in [a, b] \;\land\; f(x) \le y \le g(x) \rbrace\bigr) := \int_a^b (g - f).$$

</div>

For any non-negative function $f\colon [a, b] \to \mathbb{R}$ we define the **solid of revolution** (obtained by rotating $G_f$ around the axis $x$) as $V(a, b, f) := \lbrace(x, y, z) \in \mathbb{R}^3 \mid x \in [a, b] \;\land\; y^2 + z^2 \le f(x)^2 \rbrace$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 15</span><span class="math-callout__name">(Solid of Revolution)</span></p>

Let $f \in \mathrm{R}(a, b)$ be non-negative. Then

$$\text{volume}\bigl(V(a, b, f)\bigr) := \pi \int_a^b f^2.$$

</div>

Intuitively — or as a mnemonic — the Riemann integral $\int_a^b \pi \cdot f(x)^2\,\mathrm{d}x$ for the volume of the body $V(a, b, f)$ follows from the formula $\pi r^2$ for the area of the circle with radius $r > 0$. For $x$ running in $[a, b]$ the integral adds the volumes $\pi \cdot f(x)^2\,\mathrm{d}x$ of thin pancakes with radii $f(x)$ and thickness $\mathrm{d}x$.

## Estimates of Sums Using Integrals

They are useful, for example, in analytic number theory, where sums of the form $\sum_{n \in X} f(n)$, for sets $X \subset \mathbb{Z}$ and functions $f(x)$ given by analytic formulas, appear frequently.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 16</span><span class="math-callout__name">($\sum f(n)$ for Monotone $f$)</span></p>

Let $a < b$ be integers and $f\colon [a, b] \to \mathbb{R}$ be a monotone function. Then

$$\sum_{a < n \le b} f(n) = \text{(R)} \int_a^b f + \theta(f(b) - f(a)),$$

for some number $\theta \in [0, 1]$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The integral exists due to the monotonicity of the function $f$ (Theorem 14 in Lecture 12). We assume that $f$ is non-decreasing, the case with non-increasing $f$ is solved similarly. So now we prove the inequalities

$$0 \le \sum_{a < n \le b} f(n) - \int_a^b f \le f(b) - f(a).$$

For $b = a + 1$ the sum is $f(a+1)$ and because $f(a) \le f(x) \le f(a+1)$ for $x \in [a, a+1]$, by the monotonicity of $\text{(R)} \int$ one has that $f(a) \cdot 1 \le \int_a^{a+1} f \le f(a+1) \cdot 1$. Adding these simple inequalities with the limits $a = m, b = m+1$ for $m = a, a+1, \dots, b-1$ we get the general case. $\square$

</details>
</div>

For example, for the harmonic numbers $H_n := \sum_{i=1}^n 1/i$ we get the estimate that for $n \ge 3$,

$$H_n = 1 + \sum_{i=2}^n \frac{1}{i} = 1 + \int_1^n 1/x + \theta(1/n - 1) = [\log x]_1^n + \delta = \log n + \delta$$

where $1/n \le \delta \le 1$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 17</span><span class="math-callout__name">(Integral Criterion)</span></p>

Suppose that $m \in \mathbb{N}$ and that $f\colon [m, +\infty) \to \mathbb{R}$ is a non-negative and non-increasing function. Then the series

$$\sum_{n=m}^{\infty} f(n) \text{ converges} \;\iff\; \lim_{n \to \infty} \int_m^n f < +\infty.$$

</div>

For example, the series $\sum_{n=2}^{\infty} 1/n \log n$ diverges, i.e., has the sum $+\infty$, because $\lim_{n \to \infty} \int_2^n \frac{\mathrm{d}y}{y \log y} = \lim_{n \to \infty} [\log(\log y)]_2^n = +\infty$. Conversely, we prove convergence of the series $\sum_{n=2}^{\infty} 1/n(\log n)^c$ for every real $c > 1$ by the same method.

We present a variant of Proposition 16 for functions with integrable derivative; then a more accurate estimate of the sum in the form of an identity is obtained. Recall that $\lfloor a \rfloor$ is the lower integer part of $a \in \mathbb{R}$, the largest $m \in \mathbb{Z}$ with $m \le a$. We introduce the notation $\langle a \rangle := a - \lfloor a \rfloor - \frac{1}{2} \in [-\frac{1}{2}, \frac{1}{2})$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 18</span><span class="math-callout__name">($\sum f(n)$ for Differentiable $f$)</span></p>

Let $a < b$ be real numbers and let $f \in \mathrm{R}(a, b)$ have on $(a, b)$ the derivative $f' \in \mathrm{R}(a, b)$. Then the formula holds that

$$\sum_{a < n \le b} f(n) = \int_a^b f + \int_a^b \underbrace{\langle x \rangle f'(x)}_{T} - [\langle x \rangle f(x)]_a^b.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The formula is additive in intervals $[a, b)$, so it is enough to consider only the case that $m \le a < b \le m + 1$ for some $m \in \mathbb{Z}$. Integration by parts (Theorem 9) then gives that

$$T = \int_a^b (x - m - 1/2)f'(x) = [(x - m - 1/2)f(x)]_a^b - \int_a^b f.$$

We substitute this in the right-hand side of the formula and see that only $(\lfloor b \rfloor - m)f(b)$ remains of it. For $b < m + 1$ it is 0, which agrees with the left-hand side. For $b = m + 1$ it is $f(m+1)$, again in agreement with the left-hand side. $\square$

</details>
</div>

For harmonic numbers, the more accurate estimate $H_n = \sum_{i=1}^n 1/i = \log n + \gamma + O(1/n)$ $(n \in \mathbb{N})$, which we mentioned in part 1 of Theorem 4 of Lecture 4, is easily derived with this formula.

## Abel's Summation Formula

We conclude the lecture and the whole course with Abel's summation formula. For a sequence $(a_n) = (a_1, a_2, \dots) \subset \mathbb{R}$ and a number $x \in \mathbb{R}$ we define $A(x) := \sum_{n \le x} a_n$, with an empty sum defined as 0.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 19</span><span class="math-callout__name">(Abel's Summation)</span></p>

Let $(a_n) \subset \mathbb{R}$, $a < b$ be positive real numbers and $f\colon [a, b] \to \mathbb{R}$ be a function that has on $(a, b)$ derivative $f' \in \mathrm{R}(a, b)$. Then

$$\sum_{a < n \le b} a_n f(n) = [A(x) f(x)]_a^b - \underbrace{\int_a^b A(x) f'(x)}_{T}.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We use the trick from the previous proof. The formula is again additive in intervals $[a, b)$, so again it is enough to consider only the case that $m \le a < b \le m + 1$ for some $m \in \mathbb{N}_0$. FTC 2 (Theorem 15 in Lecture 12) then gives that

$$T = \int_a^b A(m) f'(x)\,\mathrm{d}x = A(m) [f(x)]_a^b.$$

We substitute it in the right-hand side of the formula and see that it turns in $(A(b) - A(m))f(b)$. For $b < m + 1$ it is 0, in agreement with the left-hand side. For $b = m + 1$ it is $a_{m+1} f(m+1)$, again in agreement with the left-hand side. $\square$

</details>
</div>
