---
layout: default
title: Set Theory
date: 2024-10-20
# excerpt: ...
# tags:
#   - sampling
#   - statistics
#   - algorithms
---


## Axiom of choice

The **axiom of choice (AC)** is a rule in set theory that says:

> If you have any collection of nonempty sets, then you can choose **one element from each set**, all at once.

$\textbf{Axiom (Axiom of choice):}$ For any family of nonempty sets $\lbrace A_i \rbrace_{i\in I}$, there exists a function $f$ (a **choice function**) with $f(i)\in A_i$ for every $i\in I$.

<figure>
  <img src="{{ '/assets/images/notes/set-theory/ac_figure.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>$(S_i)$ is an infinite indexed family of sets indexed over the real numbers $\mathbb{R}$; that is, there is a set $S_i$ for each real number $i$, with a small sample shown above. Each set contains at least one, and possibly infinitely many, elements. The axiom of choice allows us to select a single element from each set, forming a corresponding family of elements $(x_i)$ also indexed over the real numbers, with $x_i$ drawn from $S_i$. In general, the collections may be indexed over any set $I$, (called index set whose elements are used as indices for elements in a set) not just $\mathbb{R}$.</figcaption>
</figure>

If $I$ is finite, you can just pick elements one by one — no special axiom needed.
AC matters when the collection is **infinite**, especially when it’s so large or unstructured that there’s no explicit "rule" for choosing elements.

AC is equivalent to several extremely useful theorems (in standard set theory, ZF):

* **Zorn’s Lemma** (used everywhere in algebra/analysis)
  * e.g., every vector space has a basis
  * every ring has a maximal ideal
* **Well-ordering theorem**
  * every set can be well-ordered (even $\mathbb{R}$, though not constructively)

AC implies results that feel counterintuitive:
* **Banach–Tarski paradox**: a solid ball in 3D can be "cut" into finitely many pieces and reassembled into *two* balls of the same size (using very non-measurable sets).

So AC is widely accepted because it makes math work smoothly, but it’s not "constructive": it asserts existence of a selection without telling you how to build it.

<!-- <div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Multiple Blocks and Shared Memory)</span></p>
</div> -->

**Question**: *Axiom of choice is a foundation for what?*
**Short Answer**: *AC is not a foundation by itself, t’s an optional strengthening of the usual foundational framework for essentially all of modern pure math.*
**Long Answer**:
<div class="accordion">
  <details markdown="1">
    <summary>AC is not a foundation for itself, it is an optional strengthening</summary>

Good instinct: an “axiom” is only meaningful relative to a *theory*. The Axiom of Choice isn’t the foundation of a separate field by itself; it’s an **optional strengthening of the usual foundational framework for essentially all of modern pure math**.

### What framework is it an axiom for?

* The standard foundation is **ZF set theory** (Zermelo–Fraenkel axioms).
* When you **add** the Axiom of Choice (AC), you get **ZFC**.

So AC is “a fundament for” **ZFC**, which is the most commonly used axiomatic base for mathematics.

### What does adding AC buy you (what is built on it)?

Adding AC gives you a package of powerful existence principles that a lot of mainstream math quietly relies on—especially in “infinite” contexts:

#### 1) Linear algebra / functional analysis

* **Every vector space has a basis.**
  In particular, (\mathbb{R}) as a vector space over (\mathbb{Q}) has a (Hamel) basis — but you can’t explicitly construct it.
* Many results in functional analysis use AC-like principles (often via Zorn’s Lemma / Hahn–Banach).

#### 2) Algebra

Using Zorn’s Lemma (equivalent to AC in ZF), you get:

* **Every ring has a maximal ideal**
* Existence of **algebraic closures**
* Various “maximal/minimal” objects in module theory, field theory, etc.

#### 3) Topology

* **Tychonoff’s theorem**: arbitrary products of compact spaces are compact (this is actually equivalent to AC in a strong sense; weaker versions correspond to weaker choice principles).

#### 4) Order/structure theorems

* **Well-ordering theorem**: every set can be well-ordered (another equivalent form of AC).
* Existence of maximal chains, maximal elements, etc.

### How to think of it conceptually

AC is a foundation for **non-constructive existence** in infinite settings.

It says: even if you have infinitely many “independent” choices to make, you may assume there is a global function that makes them simultaneously—without providing an algorithm.

### Is it *needed* for “all math”?

No. A lot of mathematics can be done in **ZF** (no choice), and there are also intermediate systems:

* **DC (Dependent Choice):** enough for most classical analysis (e.g., many results about (\mathbb{R})), but weaker than full AC.
* **Countable Choice:** weaker still.

But if you open a typical graduate textbook in algebra/topology/analysis and it uses **Zorn’s Lemma**, it’s effectively working in **ZFC**.

If you tell me which area you care about (analysis vs algebra vs logic), I can point to 2–3 “signature theorems” in that area that are equivalent to, imply, or avoid AC.

  </details>
</div>