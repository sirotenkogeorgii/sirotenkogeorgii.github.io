---
layout: default
title: Pravděpodobnost a statistika 1
date: 2025-03-10
excerpt: Lecture notes for NMAI059 Pravděpodobnost a statistika 1 by Robert Šámal, covering probability spaces, conditional probability, discrete random variables, expectation, and variance.
tags:
  - probability
  - statistics
  - mathematics
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

# Pravděpodobnost a statistika 1

## 1 Probability -- Introduction

Before building probability from the ground up, here is a quick motivating application.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1</span><span class="math-callout__name">(Polynomial Identity Testing)</span></p>

Given two polynomials $f, g$ of degree $d$, we want to check whether they are identical as fast as possible. The polynomial $f$ is given by its coefficients, i.e. $f(x) = \sum_{i=0}^{d} a_i x^i$, and $g$ can be written as $g_1 \cdot g_2$ where we are given the coefficients of $g_1$ and $g_2$.

The naive approach -- comparing coefficients of $g$ with $a_i$ -- costs $O(d^2)$ (or $O(d \log d)$ via polynomial multiplication using DFT). Instead, we can verify in $O(d)$ time with a small probability of error.

Choose a parameter $k$ (a small natural number). Independently sample $x_1, \ldots, x_k$ uniformly at random from $S = \lbrace 1, 2, \ldots, 10d \rbrace$. For each $x_i$, check whether $f(x_i) = g_1(x_i) g_2(x_i)$. If any check fails, we know $f \neq g$. If all checks pass for $i = 1, \ldots, k$, we conclude $f = g$.

The error probability: all $x_i$ are roots of $f - g$, which has degree at most $d$ and thus at most $d$ roots. Since each $x_i$ is chosen from $10d$ values, the probability of a false positive for each $i$ is $\le d/(10d) = 1/10$, giving overall error probability $\le 1/10^k$.

This is the simplest case of the **Schwartz-Zippel algorithm**. The same principle underlies randomized primality testing.

</div>

### Probability -- Intuition and Definitions

Some phenomena we cannot or do not wish to describe causally: rolling a die obeys physical laws, but predicting the outcome would require knowing the exact throw mechanics. The number of emails in a day depends on many people's decisions. Program runtime on a real computer depends on caches, other processes, etc.

We set aside the philosophical question of true randomness vs. determinism and focus on what we need to describe probabilistic situations mathematically.

**Elementary events.** First we need a set $\Omega$ whose elements (called elementary events) correspond to individual outcomes of a random experiment. The word "experiment" is used broadly: it can mean a single number, a die roll, the entire execution of a program, or even the state of the universe.

For a single die roll we typically use $\Omega = [6] = \lbrace 1, 2, \ldots, 6 \rbrace$. For three die rolls, $\Omega = [6]^3$. For an infinite sequence of rolls, $\Omega = [6]^{\mathbb{N}}$. For counting emails in a day, $\Omega = \mathbb{N}_0$. For the duration of a run, $\Omega = \mathbb{R}$.

**Event space.** We choose an *event space* $\mathcal{F}$ as a subcollection of the power set $\mathscr{P}(\Omega)$. An event is a subset of $\Omega$ -- "something that happened" -- whose probability we want to measure. Often $\mathcal{F} = \mathscr{P}(\Omega)$, which is always possible when $\Omega$ is countable.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2</span><span class="math-callout__name">($\sigma$-algebra / Event Space)</span></p>

$\mathcal{F} \subseteq \mathscr{P}(\Omega)$ is an **event space** (also called a $\sigma$-algebra) if:

1. $\emptyset \in \mathcal{F}$ and $\Omega \in \mathcal{F}$.
2. $A \in \mathcal{F} \Rightarrow \Omega \setminus A \in \mathcal{F}$ (i.e. the complement $A^c \in \mathcal{F}$).
3. $A_1, A_2, \ldots \in \mathcal{F} \Rightarrow \bigcup_{i=1}^{\infty} A_i \in \mathcal{F}$.

</div>

From condition 3, closure under countable unions implies closure under finite unions (set $A_{k+1} = A_{k+2} = \cdots = \emptyset$). Combined with condition 2, we also get closure under intersections and complements.

The power set $\mathscr{P}(\Omega)$ satisfies these conditions. However, for $\Omega = \mathbb{R}$ or $\Omega = \lbrace 0, 1 \rbrace^{\mathbb{N}}$, we cannot always take $\mathcal{F} = \mathscr{P}(\Omega)$ and still define a probability on it.

### Axioms of Probability

For an event $A \in \mathcal{F}$, the number $P(A)$ represents a degree of belief that "event $A$ occurs", i.e. that a randomly chosen elementary event $\omega \in \Omega$ satisfies $\omega \in A$. For repeatable experiments, $P(A)$ can be interpreted as the long-run frequency of $A$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3</span><span class="math-callout__name">(Probability)</span></p>

$P : \mathcal{F} \to [0, 1]$ is called a **probability** (probability measure) if:

1. $P(\Omega) = 1$.
2. $P\!\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$ for any sequence of pairwise disjoint events $A_1, A_2, \ldots \in \mathcal{F}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Observations)</span></p>

- $P(\emptyset) = 0$: set $A_i = \emptyset$ for all $i$ in the countable additivity axiom.
- Axiom 2 also holds for finitely many disjoint sets: given disjoint $A_1, \ldots, A_n$, set $A_{n+1} = A_{n+2} = \cdots = \emptyset$ to get $P\!\left(\bigcup_{i=1}^{n} A_i\right) = \sum_{i=1}^{n} P(A_i)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4</span><span class="math-callout__name">(Probability Space)</span></p>

A **probability space** is a triple $(\Omega, \mathcal{F}, P)$ such that:

- $\Omega \neq \emptyset$ is an arbitrary set of elementary events,
- $\mathcal{F} \subseteq \mathscr{P}(\Omega)$ is an event space,
- $P$ is a probability on $\mathcal{F}$.

</div>

**Terminology:**
- "$A$ is a certain event" means $P(A) = 1$. We also say $A$ occurs *almost surely* (a.s.).
- "$A$ is an impossible event" means $P(A) = 0$.
- The **odds** of event $A$ are $O(A) = \frac{P(A)}{P(A^c)}$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5</span><span class="math-callout__name">(Basic Properties of Probability)</span></p>

In a probability space $(\Omega, \mathcal{F}, P)$, for any $A, B \in \mathcal{F}$:

1. $P(A) + P(A^c) = 1$ where $A^c = \Omega \setminus A$.
2. $A \subseteq B \Rightarrow P(B \setminus A) = P(B) - P(A) \Rightarrow P(A) \le P(B)$.
3. $P(A \cup B) = P(A) + P(B) - P(A \cap B)$.
4. $P(A_1 \cup A_2 \cup \ldots) \le \sum_i P(A_i)$ (subadditivity / Boole's inequality).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

1. $\Omega = A \cup A^c$ and $A, A^c$ are disjoint. So $P(\Omega) = P(A) + P(A^c)$. Since $P(\Omega) = 1$, we are done.

2. Write $B = A \cup (B \setminus A)$, a disjoint union, so $P(B) = P(A) + P(B \setminus A) \ge P(A)$.

3. Let $C_1 = A \setminus B$, $C_2 = A \cap B$, $C_3 = B \setminus A$. Then $A \cup B = C_1 \cup C_2 \cup C_3$ and $A = C_1 \cup C_2$, $B = C_2 \cup C_3$, with $C_1, C_2, C_3$ pairwise disjoint. So $P(A \cup B) = P(C_1) + P(C_2) + P(C_3)$, while $P(A) = P(C_1) + P(C_2)$ and $P(B) = P(C_2) + P(C_3)$. The result follows.

4. Use the "disjointification trick": set $B_1 = A_1$ and $B_i = A_i \setminus \bigcup_{j < i} A_j$ for $i > 1$. Then $B_i \subseteq A_i$ so $P(B_i) \le P(A_i)$, the $B_i$ are pairwise disjoint, and $\bigcup_{i=1}^{\infty} A_i = \bigcup_{i=1}^{\infty} B_i$. Hence $P\!\left(\bigcup_{i=1}^{\infty} A_i\right) = P\!\left(\bigcup_{i=1}^{\infty} B_i\right) = \sum_{i=1}^{\infty} P(B_i) \le \sum_{i=1}^{\infty} P(A_i)$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6</span><span class="math-callout__name">(Inclusion-Exclusion Principle)</span></p>

In a probability space $(\Omega, \mathcal{F}, P)$, for $A_1, \ldots, A_n \in \mathcal{F}$:

$$P(A_1 \cup \cdots \cup A_n) = \sum_{k=1}^{n} (-1)^{k-1} \sum_{I \in \binom{[n]}{k}} P\!\left(\bigcap_{i \in I} A_i\right).$$

</div>

### Examples of Probability Spaces

**Classical (uniform on a finite set).** $\Omega$ is any finite set, $\mathcal{F} = \mathscr{P}(\Omega)$, $P(A) = \lvert A \rvert / \lvert \Omega \rvert$. This is the classic formula: number of favorable outcomes divided by total outcomes. Imagine an urn with balls; $A$ is the set of red balls; we ask for the probability of drawing a red ball.

**Discrete.** Generalization of the above. $\Omega = \lbrace \omega_1, \omega_2, \ldots \rbrace$ is any countable set. Given $p_1, p_2, \ldots \in [0, 1]$ with $\sum_i p_i = 1$, define $P(A) = \sum_{i : \omega_i \in A} p_i$. Imagine an urn where each ball has a different probability of being drawn. We can write $p : \Omega \to \mathbb{R}$ with $p(\omega_i) = p_i$ and then $P(A) = \sum_{a \in A} p(a)$.

**Geometric.** For a "nice" $\Omega \subseteq \mathbb{R}^d$ with $d \ge 1$, let $V_d(A)$ denote the $d$-dimensional volume. Define $P(A) = V_d(A) / V_d(\Omega)$. Imagine a dartboard ($d = 2$): probability is proportional to area.

## 2 Conditional Probability

We have established a language for speaking about random events, but things become truly interesting when we start conditioning.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 7</span><span class="math-callout__name">(Conditional Probability)</span></p>

If $A, B \in \mathcal{F}$ and $P(B) > 0$, the **conditional probability** of $A$ given $B$ is

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8</span><span class="math-callout__name">(Conditional Probability is a Probability)</span></p>

In a probability space $(\Omega, \mathcal{F}, P)$, for a fixed $B \in \mathcal{F}$ with $P(B) > 0$, define $Q(A) := P(A \mid B)$ for all $A \in \mathcal{F}$. Then $(\Omega, \mathcal{F}, Q)$ is a probability space.

</div>

Conditioning on an event means working in a new probability space where we "rescale" all probabilities by dividing by $P(B)$, so that $B$ has probability 1.

### Chain Rule for Conditional Probability

Rewriting the definition gives $P(A \cap B) = P(B) P(A \mid B)$. The generalization to multiple events:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9</span><span class="math-callout__name">(Chain Rule)</span></p>

If $A_1, \ldots, A_n \in \mathcal{F}$ and $P(A_1 \cap \cdots \cap A_n) > 0$, then

$$P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1) \, P(A_2 \mid A_1) \, P(A_3 \mid A_1 \cap A_2) \cdots P\!\left(A_n \mid \bigcap_{i=1}^{n-1} A_i\right).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Since $P(A_1 \cap \cdots \cap A_n) > 0$, all partial intersections $P(A_1 \cap \cdots \cap A_k) > 0$ as well, so all conditional probabilities on the right-hand side are defined. The right-hand side telescopes:

$$P(A_1) \cdot \frac{P(A_1 \cap A_2)}{P(A_1)} \cdot \frac{P(A_1 \cap A_2 \cap A_3)}{P(A_1 \cap A_2)} \cdots \frac{P(A_1 \cap \cdots \cap A_n)}{P(A_1 \cap \cdots \cap A_{n-1})}$$

and most terms cancel, leaving $P(A_1 \cap \cdots \cap A_n)$.

</details>
</div>

Note that the left-hand side is symmetric in $A_1, \ldots, A_n$, so we get $n!$ different expressions on the right-hand side -- we can choose whichever ordering makes the computation easiest.

### Law of Total Probability

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 10</span><span class="math-callout__name">(Partition)</span></p>

A countable system of sets $B_1, B_2, \ldots \in \mathcal{F}$ is a **partition** of $\Omega$ if:

- $B_i \cap B_j = \emptyset$ for $i \neq j$, and
- $\bigcup_i B_i = \Omega$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 11</span><span class="math-callout__name">(Law of Total Probability)</span></p>

If $(\Omega, \mathcal{F}, P)$ is a probability space, $B_1, B_2, \ldots$ is a partition of $\Omega$, and $A \in \mathcal{F}$, then

$$P(A) = \sum_i P(B_i) \, P(A \mid B_i)$$

(terms with $P(B_i) = 0$ are treated as $0$).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Since $B_1, B_2, \ldots$ is a partition, we can write $A = (A \cap B_1) \cup (A \cap B_2) \cup \ldots$ as a disjoint union. By the definition of probability, $P(A) = \sum_i P(A \cap B_i)$. For terms with $P(B_i) > 0$, rewrite $P(A \cap B_i) = P(B_i) P(A \mid B_i)$. For $P(B_i) = 0$, we have $P(A \cap B_i) = 0$ as well, so treating the summand as $0$ is correct.

</details>
</div>

### Application -- Gambler's Ruin

Given $n \in \mathbb{N}$, a gambler starts with $a$ coins, $0 \le a \le n$. In each round, a fair coin flip determines whether the gambler wins or loses 1 coin. The game continues until the gambler has $0$ (ruin) or $n$ (win). Let $p(a)$ be the probability of winning.

Clearly $p(0) = 0$ and $p(n) = 1$. Let $V$ denote the event "gambler wins" and $PV$ the event "wins the first round". By the law of total probability:

$$P(V) = P(PV) P(V \mid PV) + P(PV^c) P(V \mid PV^c).$$

Note $P(V \mid PV) = p(a+1)$ and $P(V \mid PV^c) = p(a-1)$, so with $P(PV) = 1/2$:

$$p(a) = \tfrac{1}{2} p(a+1) + \tfrac{1}{2} p(a-1).$$

Rearranging: $p(a) - p(a-1) = p(a+1) - p(a)$. All consecutive differences are equal, so each equals $1/n$, giving $p(a) = a/n$.

### Bayes' Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 12</span><span class="math-callout__name">(Bayes' Theorem)</span></p>

If $B_1, B_2, \ldots$ is a partition of $\Omega$, $A \in \mathcal{F}$, $P(A) > 0$, and $P(B_j) > 0$, then

$$P(B_j \mid A) = \frac{P(B_j) \, P(A \mid B_j)}{\sum_i P(B_i) \, P(A \mid B_i)}.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By definition, $P(B_j \mid A) = \frac{P(B_j \cap A)}{P(A)} = \frac{P(B_j) P(A \mid B_j)}{P(A)}$. The denominator $P(A)$ is expanded using the law of total probability.

</details>
</div>

The intuition: imagine that the sets $B_i$ describe mutually exclusive hidden states that we cannot observe directly. We know their prior probabilities $P(B_i)$ and their likelihoods $P(A \mid B_i)$. After observing event $A$, Bayes' theorem tells us how to update our beliefs about which state is active.

### Independence of Events

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 13</span><span class="math-callout__name">(Independence of Two Events)</span></p>

Events $A, B \in \mathcal{F}$ are **independent** if $P(A \cap B) = P(A) P(B)$.

</div>

When $P(B) > 0$, independence is equivalent to $P(A \mid B) = P(A)$ -- knowing that $B$ occurred does not change the probability of $A$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 14</span><span class="math-callout__name">(Mutual Independence)</span></p>

Let $I$ be an arbitrary index set. Events $\lbrace A_i : i \in I \rbrace$ are **(mutually) independent** if for every finite subset $J \subseteq I$:

$$P\!\left(\bigcap_{j \in J} A_j\right) = \prod_{j \in J} P(A_j).$$

If this condition holds only for all two-element subsets $J$, the events are called **pairwise independent**.

</div>

## 3 Discrete Random Variables

Often we are interested in a number determined by the outcome of a random experiment: the distance from center when throwing at a target, the number of rolls until the first six, the number of comparisons in quicksort (depending on random pivot choices).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 15</span><span class="math-callout__name">(Discrete Random Variable)</span></p>

Given a probability space $(\Omega, \mathcal{F}, P)$, a function $X : \Omega \to \mathbb{R}$ is a **discrete random variable** if $\mathrm{Im}(X)$ (the range of $X$) is countable and for all $x \in \mathbb{R}$:

$$\lbrace \omega \in \Omega : X(\omega) = x \rbrace \in \mathcal{F}.$$

</div>

A discrete random variable induces a new probability space on its range: we forget the underlying elementary events and track only the values $X(\omega) \in \mathrm{Im}(X)$. The probability defined on this space is called the **distribution of $X$**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 16</span><span class="math-callout__name">(Probability Mass Function)</span></p>

The **probability mass function** (pmf) of a discrete random variable $X$ is the function $p_X : \mathbb{R} \to [0, 1]$ defined by

$$p_X(x) = P(\lbrace X = x \rbrace) = P(X = x).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 17</span></p>

$\sum_{x \in \mathrm{Im}(X)} p_X(x) = 1.$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 18</span></p>

For any countable set $S = \lbrace s_i : i \in I \rbrace$ and numbers $c_i \in [0, 1]$ with $\sum_{i \in I} c_i = 1$, there exists a probability space and a discrete random variable $X$ on it with $p_X(s_i) = c_i$ for all $i \in I$.

</div>

### Examples of Discrete Distributions

**Bernoulli / Alternative distribution.** $X \sim \mathrm{Bern}(p)$ for $p \in [0, 1]$. The pmf is $p_X(1) = p$, $p_X(0) = 1 - p$, and $p_X(x) = 0$ for $x \neq 0, 1$.

For any event $A \in \mathcal{F}$, the **indicator random variable** $I_A$ is defined by $I_A(\omega) = 1$ if $\omega \in A$ and $I_A(\omega) = 0$ otherwise. Then $I_A \sim \mathrm{Bern}(P(A))$.

**Geometric distribution.** $X \sim \mathrm{Geom}(p)$, where $p$ is the probability of success. $X$ represents the index of the first success in a sequence of independent trials. By independence:

$$p_X(k) = (1 - p)^{k-1} p \quad \text{for } k = 1, 2, \ldots$$

Verification: $\sum_{k=1}^{\infty} (1-p)^{k-1} p = \frac{(1-p)^0 p}{1 - (1-p)} = 1$.

**Binomial distribution.** $X \sim \mathrm{Bin}(n, p)$: the number of successes in $n$ independent trials, each with success probability $p \in [0, 1]$. By combinatorics:

$$p_X(k) = \binom{n}{k} p^k (1-p)^{n-k} \quad \text{for } k \in \lbrace 0, 1, \ldots, n \rbrace.$$

Verification via the binomial theorem: $\sum_{k=0}^{n} \binom{n}{k} p^k (1-p)^{n-k} = (p + (1-p))^n = 1$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 19</span><span class="math-callout__name">(Drawing with Replacement)</span></p>

An urn contains $N$ balls, $K$ of which are red. We draw a ball, check its color, return it, and repeat $n$ times. Let $X$ be the number of red balls drawn. Since each draw is independent with success probability $K/N$, we have $X \sim \mathrm{Bin}(n, K/N)$.

</div>

**Hypergeometric distribution.** $X \sim \mathrm{Hyper}(N, K, n)$: the number of red balls when drawing $n$ balls *without* replacement from an urn of $N$ balls, $K$ red.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 20</span><span class="math-callout__name">(Drawing without Replacement)</span></p>

With $N$ balls, $K$ red, drawing $n$ without replacement, the number of red balls $X \sim \mathrm{Hyper}(N, K, n)$ has pmf:

$$p_X(k) = \frac{\binom{K}{k} \binom{N-K}{n-k}}{\binom{N}{n}}, \quad 0 \le k \le n.$$

When $n \ll N$, the hypergeometric pmf is approximately the same as the binomial (whether or not we replace balls matters little if we draw few).

</div>

**Poisson distribution.** $X \sim \mathrm{Pois}(\lambda)$ for $\lambda > 0$. Models the count of events in a fixed interval (emails, server requests, etc.). For $k \in \mathbb{N}_0$:

$$p_X(k) = \frac{\lambda^k}{k!} e^{-\lambda}.$$

Verification: $\sum_{k=0}^{\infty} \frac{\lambda^k}{k!} e^{-\lambda} = e^{-\lambda} \cdot e^{\lambda} = 1$ by the Taylor expansion of $e^{\lambda}$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 21</span><span class="math-callout__name">(Poisson as a Limit of Binomial)</span></p>

$\mathrm{Pois}(\lambda)$ is the limit of $\mathrm{Bin}(n, \lambda/n)$ as $n \to \infty$. Precisely, if $X_n \sim \mathrm{Bin}(n, \lambda/n)$ and $X \sim \mathrm{Pois}(\lambda)$, then for each fixed $k$:

$$\lim_{n \to \infty} p_{X_n}(k) = p_X(k).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof sketch</summary>

$$p_{X_n}(k) = \binom{n}{k} \left(\frac{\lambda}{n}\right)^k \left(1 - \frac{\lambda}{n}\right)^{n-k} = \frac{\lambda^k}{k!} \cdot \frac{n(n-1)\cdots(n-k+1)}{n^k} \cdot \left(1 - \frac{\lambda}{n}\right)^n \cdot \left(1 - \frac{\lambda}{n}\right)^{-k}.$$

As $n \to \infty$: $(1 - \lambda/n)^n \to e^{-\lambda}$, the ratio $\frac{n(n-1)\cdots(n-k+1)}{n^k} \to 1$, and $(1 - \lambda/n)^{-k} \to 1$.

</details>
</div>

This means that for large $n$, a $\mathrm{Bin}(n, p)$ distribution with $np = \lambda$ is well approximated by $\mathrm{Pois}(\lambda)$. This approximation is often more practical than working with the binomial directly for large $n$ and small $p$.

**Poisson paradigm.** If $A_1, \ldots, A_n$ are nearly independent events with $P(A_i) = p_i$, $\sum_i p_i = \lambda$, $n$ large, and each $p_i$ small, then approximately $\sum_{i=1}^{n} I_{A_i} \sim \mathrm{Pois}(\lambda)$.

### 3.2 Expected Value

Imagine $X$ as the payout in one round of a game with $p_X(x_i) = p_i$ for $i = 1, \ldots, k$. Playing $n$ rounds independently, the value $x_i$ appears roughly $n_i \approx n p_i$ times, so the average payout per round is approximately $\frac{1}{n} \sum n_i x_i = \sum p_i x_i$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 23</span><span class="math-callout__name">(Expected Value)</span></p>

If $X$ is a discrete random variable, its **expected value** (expectation) is

$$\mathbb{E}(X) = \sum_{x \in \mathrm{Im}(X)} x \cdot P(X = x),$$

provided the sum converges (i.e. it makes sense).

</div>

If the sum contains terms of type $1 - 1 + 1 - 1 + \cdots$, the expectation is undefined (different groupings can yield different values).

The expectation can also be thought of as the center of mass: place mass $p_i$ at position $x_i$ on a rod, and the center of mass is at $\sum p_i x_i$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 24</span></p>

If $X$ is a discrete random variable on a discrete probability space $(\Omega, \mathcal{F}, P)$ and $\mathbb{E}(X)$ is defined, then

$$\mathbb{E}(X) = \sum_{\omega \in \Omega} X(\omega) P(\lbrace \omega \rbrace).$$

</div>

### Law of the Unconscious Statistician

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 25</span></p>

For a real function $g$ and a discrete random variable $X$, $Y = g(X)$ is also a discrete random variable.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 26</span><span class="math-callout__name">(Law of the Unconscious Statistician)</span></p>

If $X$ is a discrete random variable and $g$ is a real function, then

$$\mathbb{E}(g(X)) = \sum_{x \in \mathrm{Im}(X)} g(x) \, P(X = x),$$

provided the sum converges absolutely.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 27</span><span class="math-callout__name">(Properties of Expectation)</span></p>

Let $X, Y$ be discrete random variables and $a, b \in \mathbb{R}$.

1. If $P(X \ge 0) = 1$, then $\mathbb{E}(X) \ge 0$. Moreover, if $\mathbb{E}(X) = 0$, then $P(X = 0) = 1$.
2. If $\mathbb{E}(X) \ge 0$, then $P(X \ge 0) > 0$.
3. $\mathbb{E}(aX + b) = a \cdot \mathbb{E}(X) + b$.
4. $\mathbb{E}(X + Y) = \mathbb{E}(X) + \mathbb{E}(Y)$ (**linearity of expectation**).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**1.** By definition, $\mathbb{E}(X) = \sum_{x \in \mathrm{Im}(X)} x \, P(X = x)$. All terms are non-negative (if $x < 0$ then $P(X = x) = 0$), so the sum is non-negative. If the sum equals $0$ and all terms are non-negative, each must be zero, so $P(X = x) = 0$ for all $x > 0$.

**2.** Follows directly from 1: if $P(X \ge 0) = 0$, then all values would be negative, contradicting $\mathbb{E}(X) \ge 0$.

**3.** By Theorem 26: $\mathbb{E}(aX + b) = \sum_x (ax + b) P(X = x) = a \sum_x x \, P(X = x) + b \sum_x P(X = x) = a \, \mathbb{E}(X) + b$.

**4.** For a discrete probability space, using Remark 24: $\mathbb{E}(X + Y) = \sum_{\omega \in \Omega} (X(\omega) + Y(\omega)) P(\lbrace \omega \rbrace) = \sum_{\omega} X(\omega) P(\lbrace \omega \rbrace) + \sum_{\omega} Y(\omega) P(\lbrace \omega \rbrace) = \mathbb{E}(X) + \mathbb{E}(Y)$.

</details>
</div>

Properties 3 and 4 together give the **linearity of expectation**: the expectation of a sum of $n$ random variables equals the sum of their expectations.

### Conditional Expectation

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 29</span><span class="math-callout__name">(Conditional Expectation)</span></p>

If $X$ is a discrete random variable and $P(B) > 0$, the **conditional expectation** of $X$ given $B$ is

$$\mathbb{E}(X \mid B) = \sum_{x \in \mathrm{Im}(X)} x \cdot P(X = x \mid B),$$

provided the sum converges.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 30</span><span class="math-callout__name">(Law of Total Expectation)</span></p>

If $B_1, B_2, \ldots$ is a partition of $\Omega$ and $X$ is a discrete random variable, then

$$\mathbb{E}(X) = \sum_i P(B_i) \cdot \mathbb{E}(X \mid B_i),$$

whenever the sum converges (terms with $P(B_i) = 0$ are treated as $0$).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Starting from the definition of $\mathbb{E}(X)$ and applying the law of total probability:

$$\mathbb{E}(X) = \sum_{x \in \mathrm{Im}(X)} x \cdot P(X = x) = \sum_x x \cdot \sum_i P(B_i) P(X = x \mid B_i) = \sum_i P(B_i) \sum_x x \cdot P(X = x \mid B_i) = \sum_i P(B_i) \, \mathbb{E}(X \mid B_i).$$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 31</span><span class="math-callout__name">(Expected Number of Rolls until First Six)</span></p>

Let $X$ be the number of die rolls until the first six (inclusive). Let $PS$ denote the event that the first roll is a six. Using Theorem 30 with partition $\lbrace PS, PS^c \rbrace$:

- $\mathbb{E}(X \mid PS) = 1$ (the first roll is already a six).
- $\mathbb{E}(X \mid PS^c) = 1 + \mathbb{E}(X)$ (after a failed first roll, we are back to the same situation plus one roll).

So $\mathbb{E}(X) = \frac{1}{6} \cdot 1 + \frac{5}{6} \cdot (1 + \mathbb{E}(X))$, giving $(1 - \frac{5}{6}) \mathbb{E}(X) = 1$, hence $\mathbb{E}(X) = 6$.

Alternative: since $X \sim \mathrm{Geom}(1/6)$, this follows directly from $\mathbb{E}(\mathrm{Geom}(p)) = 1/p$.

</div>

### Alternative Formula for Expectation

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 32</span><span class="math-callout__name">(Tail-Sum Formula)</span></p>

If $X$ is a discrete random variable taking values in $\mathbb{N}_0 = \lbrace 0, 1, 2, \ldots \rbrace$, then

$$\mathbb{E}(X) = \sum_{n=0}^{\infty} P(X > n).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Write $X = \sum_{n=0}^{\infty} I_{X > n}$. If $X(\omega) = k$, then $I_{X > n}(\omega) = 1$ for exactly $n = 0, 1, \ldots, k-1$, i.e. for $k$ values. By linearity of expectation:

$$\mathbb{E}(X) = \sum_{n=0}^{\infty} \mathbb{E}(I_{X > n}) = \sum_{n=0}^{\infty} P(X > n).$$

Alternative direct proof by swapping the order of summation:

$$\mathbb{E}(X) = \sum_{k=0}^{\infty} k \cdot P(X = k) = \sum_{k=0}^{\infty} \sum_{0 \le n < k} P(X = k) = \sum_{n=0}^{\infty} \sum_{k > n} P(X = k) = \sum_{n=0}^{\infty} P(X > n).$$

</details>
</div>

### Variance

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 33</span><span class="math-callout__name">(Variance and Standard Deviation)</span></p>

The **variance** of a random variable $X$ is $\mathrm{var}(X) = \mathbb{E}((X - \mathbb{E}X)^2)$. The **standard deviation** is $\sigma_X = \sqrt{\mathrm{var}(X)}$.

</div>

By Theorem 27 (part 1) applied to $(X - \mathbb{E}(X))^2$, the variance is always $\ge 0$. It equals $0$ if and only if $P(X = \mathbb{E}(X)) = 1$ (the random variable is almost surely constant).

Variance measures how much $X$ fluctuates around its mean. The standard deviation has the same units as $X$ (if $X$ is in meters, $\sigma_X$ is in meters), while variance is in squared units, which is why formulas often use variance.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 34</span><span class="math-callout__name">(Computational Formula for Variance)</span></p>

For a discrete random variable $X$:

$$\mathrm{var}(X) = \mathbb{E}(X^2) - (\mathbb{E}(X))^2 = \mathbb{E}(X(X-1)) + \mathbb{E}(X) - (\mathbb{E}(X))^2.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $\mu = \mathbb{E}(X)$. Then $\mathrm{var}(X) = \mathbb{E}((X - \mu)^2) = \mathbb{E}(X^2 - 2\mu X + \mu^2) = \mathbb{E}(X^2) - 2\mu \mathbb{E}(X) + \mu^2 = \mathbb{E}(X^2) - (\mathbb{E}(X))^2$. The second form follows from $\mathbb{E}(X(X-1)) = \mathbb{E}(X^2) - \mathbb{E}(X)$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 35</span><span class="math-callout__name">(Variance under Affine Transformation)</span></p>

For real numbers $a, b$ and a discrete random variable $X$:

$$\mathrm{var}(aX + b) = a^2 \, \mathrm{var}(X).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $Y = aX + b$. Then $Y - \mathbb{E}(Y) = (aX + b) - (a\mathbb{E}(X) + b) = a(X - \mathbb{E}(X))$. So $\mathbb{E}((Y - \mathbb{E}(Y))^2) = \mathbb{E}(a^2 (X - \mathbb{E}(X))^2) = a^2 \mathbb{E}((X - \mathbb{E}(X))^2)$.

</details>
</div>

### 3.3 Properties of Specific Distributions

**Bernoulli.** For $X \sim \mathrm{Bern}(p)$: $\mathbb{E}(X) = p$, $\mathrm{var}(X) = \mathbb{E}(X^2) - (\mathbb{E}(X))^2 = p - p^2 = p(1-p)$. Note $X^2 = X$ since $X$ only takes values $0$ and $1$.

**Geometric.** For $X \sim \mathrm{Geom}(p)$: $\mathbb{E}(X) = 1/p$ and $\mathrm{var}(X) = (1-p)/p^2$.

The mean can be derived via the tail-sum formula (Theorem 32):

$$\mathbb{E}(X) = \sum_{n=0}^{\infty} P(X > n) = \sum_{n=0}^{\infty} (1-p)^n = \frac{1}{1-(1-p)} = \frac{1}{p}.$$

Or directly from the definition: $\mathbb{E}(X) = \sum_{n=1}^{\infty} n (1-p)^{n-1} p = \frac{1}{1-(1-p)} = \frac{1}{p}$.

**Binomial.** For $X \sim \mathrm{Bin}(n, p)$: write $X = \sum_{i=1}^n X_i$ where $X_i \sim \mathrm{Bern}(p)$ are the indicator variables for each trial. By linearity: $\mathbb{E}(X) = \sum_{i=1}^n \mathbb{E}(X_i) = np$. Later (using the variance of sums of independent variables): $\mathrm{var}(X) = np(1-p)$.

Alternatively, directly from the definition:

$$\mathbb{E}(X) = \sum_{k=0}^{n} k \binom{n}{k} p^k (1-p)^{n-k} = \sum_{k=1}^{n} n \binom{n-1}{k-1} p \cdot p^{k-1} (1-p)^{(n-1)-(k-1)} = np(p + (1-p))^{n-1} = np.$$

Similarly, $\mathbb{E}(X(X-1)) = n(n-1)p^2$, so by Theorem 34: $\mathrm{var}(X) = n(n-1)p^2 + np - n^2p^2 = np(1-p)$.

**Hypergeometric.** For $X \sim \mathrm{Hyper}(N, K, n)$: $\mathbb{E}(X) = n \frac{K}{N}$ and $\mathrm{var}(X) = n \frac{K}{N}\!\left(1 - \frac{K}{N}\right) \frac{N - n}{N - 1}$.

For $\mathbb{E}(X)$: write $X = \sum_{i=1}^n X_i$ where $X_i = 1$ if the $i$-th drawn ball is red. By symmetry (swapping the $i$-th and first ball doesn't change the process), $P(X_i = 1) = K/N$ for all $i$, so $\mathbb{E}(X) = nK/N$.

Alternatively, write $X = \sum_{j=1}^K Y_j$ where $Y_j = 1$ if the $j$-th red ball is among the $n$ drawn. Then $Y_j \sim \mathrm{Bin}(n/N)$ -- more precisely, $P(Y_j = 1) = \binom{N-1}{n-1}/\binom{N}{n} = n/N$. So $\mathbb{E}(X) = nK/N$.

**Poisson.** For $X \sim \mathrm{Pois}(\lambda)$: $\mathbb{E}(X) = \lambda$ and $\mathrm{var}(X) = \lambda$.

Direct computation:

$$\mathbb{E}(X) = \sum_{k=0}^{\infty} k \cdot e^{-\lambda} \frac{\lambda^k}{k!} = \sum_{k=1}^{\infty} k \cdot e^{-\lambda} \frac{\lambda^k}{k!} = \lambda \sum_{k=1}^{\infty} e^{-\lambda} \frac{\lambda^{k-1}}{(k-1)!} = \lambda \sum_{k=0}^{\infty} e^{-\lambda} \frac{\lambda^k}{k!} = \lambda \cdot 1 = \lambda.$$

Similarly, one can compute $\mathbb{E}(X(X-1)) = \lambda^2$ and use Theorem 34 to get $\mathrm{var}(X) = \lambda$.

## 4 Random Vectors

Let $X$ and $Y$ be discrete random variables on the same probability space $(\Omega, \mathcal{F}, P)$. We want to treat the pair $(X, Y)$ as a single object -- a **random vector**.

### Joint Distribution

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 36</span><span class="math-callout__name">(Joint PMF)</span></p>

For discrete random variables $X, Y$ on $(\Omega, \mathcal{F}, P)$, their **joint probability mass function** (joint pmf) $p_{X,Y} : \mathbb{R}^2 \to [0, 1]$ is defined by

$$p_{X,Y}(x, y) = P(\lbrace \omega \in \Omega : X(\omega) = x \;\&\; Y(\omega) = y \rbrace).$$

The pair $(X, Y)$ forms a random vector if all these probabilities are defined, i.e. for every $x, y \in \mathbb{R}$, $\lbrace X = x \;\&\; Y = y \rbrace \in \mathcal{F}$.

</div>

This generalizes to more than two variables: $p_{X_1, \ldots, X_n}(x_1, \ldots, x_n)$.

### Marginal Distributions

Given $p_{X,Y}$, how do we recover the distributions of the individual components $p_X$ and $p_Y$? And conversely, can we determine $p_{X,Y}$ from $p_X$ and $p_Y$?

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 37</span><span class="math-callout__name">(Marginal Distributions)</span></p>

Let $X, Y$ be discrete random variables. Then

$$p_X(x) = P(X = x) = \sum_{y \in \mathrm{Im}(Y)} P(X = x \;\&\; Y = y) = \sum_{y \in \mathrm{Im}(Y)} p_{X,Y}(x, y),$$

$$p_Y(y) = P(Y = y) = \sum_{x \in \mathrm{Im}(X)} P(X = x \;\&\; Y = y) = \sum_{x \in \mathrm{Im}(X)} p_{X,Y}(x, y).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The event $\lbrace X = x \rbrace$ is the disjoint union of events $\lbrace X = x \;\&\; Y = y \rbrace$ over all $y \in \mathrm{Im}(Y)$, so the formula on the first line follows directly from the definition of probability. The second line is analogous.

</details>
</div>

In general, one cannot reconstruct $p_{X,Y}$ from $p_X$ and $p_Y$ alone -- the joint distribution carries more information than the marginals.

### Independence of Random Variables

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 38</span><span class="math-callout__name">(Independence of Random Variables)</span></p>

Discrete random variables $X, Y$ are **independent** if for every $x, y \in \mathbb{R}$, the events $\lbrace X = x \rbrace$ and $\lbrace Y = y \rbrace$ are independent. Equivalently,

$$P(X = x, Y = y) = P(X = x) P(Y = y).$$

</div>

For independent random variables, the converse of Theorem 37 holds: from $p_X$ and $p_Y$ we can reconstruct $p_{X,Y}$.

### Functions of Random Vectors

Given $p_{X,Y}$ and a function $g : \mathbb{R}^2 \to \mathbb{R}$, we can determine the distribution of $Z = g(X, Y)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 39</span><span class="math-callout__name">(Distribution of a Function of a Random Vector)</span></p>

Let $(X, Y)$ be a random vector and $g : \mathbb{R}^2 \to \mathbb{R}$. Then $Z = g(X, Y)$ is a random variable on $(\Omega, \mathcal{F}, P)$ with

$$p_Z(z) = P(Z = z) = \sum_{\substack{x \in \mathrm{Im}(X),\, y \in \mathrm{Im}(Y) \\ g(x,y) = z}} P(X = x \;\&\; Y = y).$$

</div>

An important special case:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 40</span><span class="math-callout__name">(Convolution Formula)</span></p>

If $X, Y$ are discrete random variables, then $Z = X + Y$ has pmf

$$P(Z = z) = \sum_{x \in \mathrm{Im}(X)} P(X = x \;\&\; Y = z - x).$$

If moreover $X$ and $Y$ are independent, the pmf of $Z$ is the **convolution** of $p_X$ and $p_Y$:

$$P(Z = z) = \sum_{x \in \mathrm{Im}(X)} P(X = x) P(Y = z - x).$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 41</span><span class="math-callout__name">(Sum of Two Dice)</span></p>

Let $X, Y$ be independent uniform random variables on $\lbrace 1, 2, \ldots, 6 \rbrace$ (two independent die rolls). The distribution of $Z = X + Y$:

$$P(X + Y = k) = \sum_{x=1}^{6} P(X = x) P(Y = k - x).$$

Each term is $1/36$ or $0$ (the latter when $1 \le k - x \le 6$ fails). Counting valid terms:

| $k$ | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| $p_Z(k)$ | $\frac{1}{36}$ | $\frac{2}{36}$ | $\frac{3}{36}$ | $\frac{4}{36}$ | $\frac{5}{36}$ | $\frac{6}{36}$ | $\frac{5}{36}$ | $\frac{4}{36}$ | $\frac{3}{36}$ | $\frac{2}{36}$ | $\frac{1}{36}$ |

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 42</span><span class="math-callout__name">(Sum of Independent Binomials)</span></p>

Let $X \sim \mathrm{Bin}(m, p)$ and $Y \sim \mathrm{Bin}(n, p)$ be independent. Then $Z = X + Y \sim \mathrm{Bin}(m + n, p)$.

Using the convolution formula:

$$P(Z = z) = \sum_{k=0}^{m} \binom{m}{k} p^k (1-p)^{m-k} \binom{n}{z-k} p^{z-k} (1-p)^{n-(z-k)} = p^z (1-p)^{m+n-z} \sum_{k=0}^{m} \binom{m}{k} \binom{n}{z-k} = p^z (1-p)^{m+n-z} \binom{m+n}{z},$$

where the last step uses the Vandermonde identity.

</div>

### Expectation of Functions of Random Vectors

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 43</span><span class="math-callout__name">(LOTUS for Random Vectors)</span></p>

Using the notation of Theorem 39:

$$\mathbb{E}(g(X, Y)) = \sum_{x \in \mathrm{Im}(X)} \sum_{y \in \mathrm{Im}(Y)} g(x, y) \, P(X = x \;\&\; Y = y).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 44</span><span class="math-callout__name">(Linearity of Expectation -- General Form)</span></p>

For random variables $X, Y$ and $a, b \in \mathbb{R}$:

$$\mathbb{E}(aX + bY) = a\,\mathbb{E}(X) + b\,\mathbb{E}(Y).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 45</span><span class="math-callout__name">(Product of Independent Variables)</span></p>

For **independent** discrete random variables $X, Y$:

$$\mathbb{E}(XY) = \mathbb{E}(X) \, \mathbb{E}(Y).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Apply Theorem 43 with $g(x, y) = xy$:

$$\mathbb{E}(XY) = \sum_x \sum_y xy \, P(X = x \;\&\; Y = y) = \sum_x \sum_y xy \, P(X = x) P(Y = y) = \left(\sum_x x \, P(X = x)\right)\!\left(\sum_y y \, P(Y = y)\right) = \mathbb{E}(X) \cdot \mathbb{E}(Y).$$

</details>
</div>

### Covariance

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 46</span><span class="math-callout__name">(Covariance)</span></p>

For random variables $X, Y$, their **covariance** is

$$\mathrm{cov}(X, Y) = \mathbb{E}\!\big((X - \mathbb{E}X)(Y - \mathbb{E}Y)\big).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 47</span><span class="math-callout__name">(Computational Formula for Covariance)</span></p>

$$\mathrm{cov}(X, Y) = \mathbb{E}(XY) - \mathbb{E}(X)\,\mathbb{E}(Y).$$

</div>

Basic properties of covariance:
- $\mathrm{var}(X) = \mathrm{cov}(X, X)$ (directly from the definition).
- $\mathrm{cov}(X, aY + bZ + c) = a\,\mathrm{cov}(X, Y) + b\,\mathrm{cov}(X, Z)$ (linearity of expectation).
- $\mathrm{cov}(X, Y) = 0$ if $X, Y$ are independent (by Theorem 45 and 47).
- The converse does not hold: zero covariance does not imply independence.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 48</span><span class="math-callout__name">(Correlation)</span></p>

The **correlation** of random variables $X, Y$ is

$$\varrho(X, Y) = \frac{\mathrm{cov}(X, Y)}{\sqrt{\mathrm{var}(X)\,\mathrm{var}(Y)}}.$$

</div>

- Correlation is a normalized version of covariance: $-1 \le \varrho(X, Y) \le 1$ (by the Cauchy-Schwarz inequality).
- Correlation does not imply causation! Two variables may be correlated by chance or because they share a common cause.
- Conversely, zero correlation does not imply independence. (E.g., $X$ arbitrary, $Y = +X$ or $Y = -X$ each with probability $1/2$ -- same distribution, but $\mathrm{cov}(X, Y)$ can be zero.)

### Variance of a Sum

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 49</span><span class="math-callout__name">(Variance of a Sum)</span></p>

Let $X = \sum_{i=1}^{n} X_i$. Then

$$\mathrm{var}(X) = \sum_{i=1}^{n} \sum_{j=1}^{n} \mathrm{cov}(X_i, X_j) = \sum_{i=1}^{n} \mathrm{var}(X_i) + \sum_{i \neq j} \mathrm{cov}(X_i, X_j).$$

In particular, if $X_1, \ldots, X_n$ are independent:

$$\mathrm{var}(X) = \sum_{i=1}^{n} \mathrm{var}(X_i).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Compute $\mathbb{E}(X^2)$ and $(\mathbb{E}(X))^2$ using linearity and expanding the square of a sum:

$$\mathbb{E}(X^2) = \mathbb{E}\!\left(\left(\sum_{i=1}^n X_i\right)^2\right) = \sum_{i=1}^n \sum_{j=1}^n \mathbb{E}(X_i X_j),$$

$$(\mathbb{E}(X))^2 = \left(\sum_{i=1}^n \mathbb{E}(X_i)\right)^2 = \sum_{i=1}^n \sum_{j=1}^n \mathbb{E}(X_i)\,\mathbb{E}(X_j).$$

Subtracting and using Theorem 47 gives the first equality. The rest follows from covariance properties.

</details>
</div>

### Conditional Distribution

For discrete random variables $X, Y$ on $(\Omega, \mathcal{F}, P)$ and an event $A \in \mathcal{F}$, the conditional distribution is simply the familiar conditional probability applied to events of the form $\lbrace X = x \rbrace$:

- $p_{X \mid A}(x) := P(X = x \mid A)$
- $p_{X \mid Y}(x \mid y) := P(X = x \mid Y = y)$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 50</span></p>

$$p_{X \mid Y}(x \mid y) = \frac{p_{X,Y}(x, y)}{p_Y(y)} = \frac{p_{X,Y}(x, y)}{\sum_{x'} p_{X,Y}(x', y)}.$$

This follows directly from the definition of conditional probability and Theorem 37.

</div>

## 5 Continuous Random Variables

In this chapter we study random variables that need not be discrete. A typical example: a uniformly random number from an interval (programming languages simulate this for $(0, 1)$).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 51</span><span class="math-callout__name">(Random Variable -- General)</span></p>

A **random variable** on $(\Omega, \mathcal{F}, P)$ is a mapping $X : \Omega \to \mathbb{R}$ such that for every $x \in \mathbb{R}$:

$$\lbrace \omega \in \Omega : X(\omega) \le x \rbrace \in \mathcal{F}.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 52</span></p>

Every discrete random variable is a random variable in this general sense.

</div>

For continuous random variables, each individual value has probability zero, so the pmf is no longer useful. We need a new descriptor.

### Cumulative Distribution Function

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 53</span><span class="math-callout__name">(Cumulative Distribution Function)</span></p>

The **cumulative distribution function** (CDF) of a random variable $X$ is the function $F_X : \mathbb{R} \to [0, 1]$ defined by

$$F_X(x) := P(X \le x) = P(\lbrace \omega \in \Omega : X(\omega) \le x \rbrace).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 54</span><span class="math-callout__name">(Properties of the CDF)</span></p>

Let $X$ be a random variable. Then:

1. $F_X$ is non-decreasing.
2. $\lim_{x \to -\infty} F_X(x) = 0$.
3. $\lim_{x \to +\infty} F_X(x) = 1$.
4. $F_X$ is right-continuous.

</div>

### Continuous Random Variables and the PDF

We now focus on the most common type of non-discrete random variable: continuous ones.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 55</span><span class="math-callout__name">(Continuous Random Variable, PDF)</span></p>

A random variable $X$ is called **continuous** if there exists a non-negative real function $f_X$ such that

$$F_X(x) = P(X \le x) = \int_{-\infty}^{x} f_X(t)\,dt.$$

The function $f_X$ is called the **probability density function** (pdf) of $X$.

</div>

Intuitively, we can generate a continuous random variable by sampling a uniform random point under the graph of $f_X$ and taking its $x$-coordinate. The condition $\int_{-\infty}^{\infty} f_X = 1$ ensures that the total area is 1.

The density $f_X(x)$ is approximately $P(x - h < X < x + h) / (2h)$ for small $h$ -- the probability per unit length.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 56</span><span class="math-callout__name">(Properties of Continuous Random Variables)</span></p>

Let $X$ be a continuous random variable with density $f_X$. Then:

1. $P(X = x) = 0$ for every $x \in \mathbb{R}$.
2. $P(a \le X \le b) = \int_a^b f_X(t)\,dt$ for every $a < b$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

From the definition of the CDF: $P(a < X \le b) = P(X \le b) - P(X \le a) = F_X(b) - F_X(a) = \int_a^b f_X(t)\,dt$.

For property 1: set $b = x$ and $a = x - 1/n$, then as $n \to \infty$ the integral tends to zero (since $f_X$ is bounded in any finite interval). So $P(X = x) = 0$.

Property 2 follows because $P(X = a) = 0$, so $P(a \le X \le b) = P(a < X \le b)$.

</details>
</div>

### Expected Value of a Continuous Random Variable

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 57</span><span class="math-callout__name">(Expected Value -- Continuous Case)</span></p>

If $X$ is a continuous random variable with density $f_X$, its **expected value** is

$$\mathbb{E}(X) = \int_{-\infty}^{\infty} x \, f_X(x)\,dx,$$

provided the integral converges (i.e. it is not of the form $\infty - \infty$).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 58</span><span class="math-callout__name">(LOTUS -- Continuous Case)</span></p>

If $X$ is a continuous random variable with density $f_X$ and $g$ is a real function, then

$$\mathbb{E}(g(X)) = \int_{-\infty}^{\infty} g(x) \, f_X(x)\,dx,$$

provided the integral converges.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 59</span><span class="math-callout__name">(Linearity of Expectation -- General)</span></p>

For $X_1, \ldots, X_n$ (discrete or continuous) random variables:

$$\mathbb{E}(a_1 X_1 + \cdots + a_n X_n) = a_1 \mathbb{E}(X_1) + \cdots + a_n \mathbb{E}(X_n).$$

</div>

### Variance of a Continuous Random Variable

For a continuous random variable $X$ with $\mathbb{E}(X) = \mu$, the variance is defined identically as in the discrete case:

$$\mathrm{var}(X) := \mathbb{E}((X - \mu)^2) = \int_{-\infty}^{\infty} (x - \mu)^2 f_X(x)\,dx.$$

The computational formula $\mathrm{var}(X) = \mathbb{E}(X^2) - (\mathbb{E}(X))^2$ still holds (same proof via linearity).

## 6 Specific Continuous Distributions and Their Parameters

### 6.1 Uniform Distribution

$X \sim U(a, b)$: the density is $f_X(x) = \frac{1}{b-a}$ for $x \in [a, b]$ and $0$ otherwise. The CDF is $F_X(x) = \frac{x - a}{b - a}$ for $x \in [a, b]$, with $F_X(x) = 0$ for $x \le a$ and $F_X(x) = 1$ for $x \ge b$.

$$\mathbb{E}(X) = \int_a^b \frac{x}{b-a}\,dx = \frac{a + b}{2}, \qquad \mathbb{E}(X^2) = \int_a^b \frac{x^2}{b-a}\,dx = \frac{a^2 + ab + b^2}{3}.$$

$$\mathrm{var}(X) = \frac{(b-a)^2}{12}.$$

### 6.2 Exponential Distribution

$X \sim \mathrm{Exp}(\lambda)$ for $\lambda > 0$:

$$F_X(x) = \begin{cases} 0 & x \le 0, \\ 1 - e^{-\lambda x} & x \ge 0, \end{cases} \qquad f_X(x) = \begin{cases} 0 & x \le 0, \\ \lambda e^{-\lambda x} & x \ge 0. \end{cases}$$

Models waiting times: time until the next phone call, the next server request, radioactive decay, etc.

**Relation to $\mathrm{Geom}(p)$.** For $X \sim \mathrm{Exp}(\lambda)$ and $Y \sim \mathrm{Geom}(p)$:
- $P(X > n\delta) = e^{-\lambda n \delta}$ for any $\delta > 0$.
- $P(Y > n) = (1-p)^n$.

Choosing $p$ so that $1 - p = e^{-\lambda\delta}$ (i.e. $p \doteq \lambda\delta$), we get $P(X > n\delta) = P(Y > n)$. So the exponential distribution is the continuous limit of the geometric as $\delta \to 0$.

**Expected value and variance** (by integration by parts):

$$\mathbb{E}(X) = \int_0^{\infty} x \lambda e^{-\lambda x}\,dx = \frac{1}{\lambda}, \qquad \mathbb{E}(X^2) = \int_0^{\infty} x^2 \lambda e^{-\lambda x}\,dx = \frac{2}{\lambda^2}.$$

$$\mathrm{var}(X) = \frac{2}{\lambda^2} - \frac{1}{\lambda^2} = \frac{1}{\lambda^2}.$$

### 6.3 Normal Distribution

A random variable $X$ has the **standard normal distribution**, written $X \sim N(0, 1)$, if its density is

$$\varphi(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}.$$

The corresponding CDF is denoted $\Phi$. There is no closed-form primitive for $\varphi$; $\Phi$ must be computed numerically. (The constant $\sqrt{\pi}$ ensures $\int_{-\infty}^{\infty} \varphi = 1$.)

For $X \sim N(0, 1)$: $\mathbb{E}(X) = 0$ and $\mathrm{var}(X) = 1$.

**General normal distribution.** We write $X \sim N(\mu, \sigma^2)$ when $Z = \frac{X - \mu}{\sigma} \sim N(0, 1)$. Equivalently, $X = \mu + \sigma Z$ where $Z \sim N(0, 1)$. The density of $N(\mu, \sigma^2)$ is $\frac{1}{\sigma}\varphi\!\left(\frac{x - \mu}{\sigma}\right)$.

By Theorems 27 and 35: $\mathbb{E}(X) = \mu$ and $\mathrm{var}(X) = \sigma^2$.

**Stability under summation.** If $X_1, \ldots, X_k$ are independent with $X_i \sim N(\mu_i, \sigma_i^2)$, then

$$X_1 + \cdots + X_k \sim N(\mu, \sigma^2), \quad \text{where } \mu = \mu_1 + \cdots + \mu_k,\; \sigma^2 = \sigma_1^2 + \cdots + \sigma_k^2.$$

This is a remarkable and essentially unique property of the normal distribution.

**The 68-95-99.7 rule** ($3\sigma$ rule). For $X \sim N(\mu, \sigma^2)$:

$$P(\mu - \sigma < X < \mu + \sigma) \approx 0.68, \quad P(\mu - 2\sigma < X < \mu + 2\sigma) \approx 0.95, \quad P(\mu - 3\sigma < X < \mu + 3\sigma) \approx 0.997.$$

### 6.4 Cauchy Distribution

The Cauchy distribution has density $f(x) = \frac{1}{\pi(1 + x^2)}$ and CDF $F(x) = \frac{1}{\pi}\arctan x + \frac{1}{2}$.

This is mainly a cautionary example. Although its graph resembles the normal distribution (just "more spread out"), the Cauchy distribution has **no expected value**:

$$\int_0^{\infty} x \cdot f(x)\,dx = \left[\frac{1}{2\pi} \log(1 + x^2)\right]_0^{\infty} = \infty.$$

So the integral is of the form $\infty - \infty$. Empirically, if you generate many samples and compute the mean, it wanders far from zero (unlike $N(0,1)$ samples).

### 6.5 Other Distributions

A brief list of other important continuous distributions (some will appear later):

- **Gamma distribution** -- the distribution of a sum of independent exponential random variables. Models situations requiring "multiple waiting periods".
- **Beta distribution** with parameters $\alpha, \beta$: density proportional to $x^{\alpha - 1}(1-x)^{\beta - 1}$ for $x \in (0, 1)$.
- **Chi-squared distribution** with $k$ degrees of freedom ($\chi^2_k$): the distribution of $Z_1^2 + \cdots + Z_k^2$ where $Z_i \sim N(0, 1)$ are independent. A special case of the gamma distribution.
- **Student's $t$-distribution** -- appears in the statistical part of the course.

## 7 Quantile Function and Universality of the Uniform Distribution

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Quantile Function)</span></p>

For a random variable $X$, the **quantile function** $Q_X : [0, 1] \to \mathbb{R}$ is defined by

$$Q_X(p) := \min \lbrace x \in \mathbb{R} : p \le F_X(x) \rbrace.$$

</div>

- If $F_X$ is continuous (and strictly increasing), then $Q_X = F_X^{-1}$.
- In general: $Q_X(p) \le x \iff p \le F_X(x)$.
- The **median** is $m = Q_X(1/2)$: $P(X \le m) \ge 1/2$ and $P(X > m) \le 1/2$.
- The **first quartile** is $Q_X(1/4)$; the **tenth percentile** is $Q_X(0.1)$, etc.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 60</span><span class="math-callout__name">(CDF of a Continuous RV is Uniform)</span></p>

Let $X$ be a continuous random variable with strictly increasing CDF $F = F_X$. Then $F(X) \sim U(0, 1)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $Y = F(X)$. For $y \in (0, 1)$, choose $x$ such that $y = F(x)$ (possible since $F$ is continuous and strictly increasing). Then $P(Y \le y) = P(F(X) \le F(x)) = P(X \le x) = F(x) = y$. So $F_Y$ is the CDF of $U(0, 1)$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 61</span><span class="math-callout__name">(Inverse CDF Transform)</span></p>

Let $U \sim U(0, 1)$ and $F$ be a CDF-type function (non-decreasing, continuous, with limits $0$ and $1$). Let $Q$ be the corresponding quantile function. Then $Q(U)$ is a random variable with CDF $F$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $X = Q(U)$. Then $P(X \le x) = P(Q(U) \le x) = P(U \le F(x)) = F(x)$, using the property $Q(p) \le x \iff p \le F(x)$ and $F(x) \in [0, 1]$.

</details>
</div>

Theorem 61 is used in practice to **generate random variables with any desired distribution**: given access to $U \sim U(0, 1)$, compute $Q(U)$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 62</span><span class="math-callout__name">(Generating Exponential Samples)</span></p>

To generate $X \sim \mathrm{Exp}(\lambda)$ from $U \sim U(0, 1)$: the CDF is $F(x) = 1 - e^{-\lambda x}$, which is continuous and strictly increasing. The quantile function is $Q(p) = \frac{-\log(1-p)}{\lambda}$. By Theorem 61, $Q(U) = \frac{-\log(1-U)}{\lambda} \sim \mathrm{Exp}(\lambda)$.

</div>

## 8 Continuous Random Vectors

### Joint CDF

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 63</span><span class="math-callout__name">(Joint CDF)</span></p>

For random variables $X, Y$ on $(\Omega, \mathcal{F}, P)$, the **joint CDF** $F_{X,Y} : \mathbb{R}^2 \to [0, 1]$ is

$$F_{X,Y}(x, y) = P(\lbrace \omega \in \Omega : X(\omega) \le x \;\&\; Y(\omega) \le y \rbrace).$$

</div>

This generalizes to $n$ variables: $F_{X_1, \ldots, X_n}(x_1, \ldots, x_n) = P(X_1 \le x_1 \;\&\; \ldots \;\&\; X_n \le x_n)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 64</span><span class="math-callout__name">(Probability of a Rectangle)</span></p>

If $F = F_{X,Y}$, then

$$P(X \in (a, b] \;\&\; Y \in (c, d]) = F(b, d) - F(a, d) - F(b, c) + F(a, c).$$

</div>

### Joint Density

Often the joint CDF can be written as an integral of a non-negative function $f_{X,Y}$:

$$F_{X,Y}(x, y) = \int_{-\infty}^{x} \int_{-\infty}^{y} f_{X,Y}(s, t)\,dt\,ds.$$

The function $f_{X,Y}$ is called the **joint density**. (Note that $f_{X,Y} > 1$ is possible -- it is a density, not a probability.) We have $f_{X,Y}(x, y) = \frac{\partial^2 F_{X,Y}(x, y)}{\partial x \, \partial y}$ and

$$P((X, Y) \in S) = \iint_S f_{X,Y}(x, y)\,dx\,dy.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 65</span><span class="math-callout__name">(LOTUS for Continuous Random Vectors)</span></p>

$$\mathbb{E}(g(X, Y)) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} g(x, y)\,f_{X,Y}(x, y)\,dx\,dy.$$

</div>

From this, $\mathbb{E}(aX + bY + c) = a\,\mathbb{E}(X) + b\,\mathbb{E}(Y) + c$, which proves Theorem 59 for the continuous case.

### Independence of Continuous Random Variables

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 66</span><span class="math-callout__name">(Independence -- Continuous Case)</span></p>

Random variables $X, Y$ are **independent** if the events $\lbrace X \le x \rbrace$ and $\lbrace Y \le y \rbrace$ are independent for all $x, y \in \mathbb{R}$. Equivalently,

$$F_{X,Y}(x, y) = F_X(x) \, F_Y(y).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 67</span><span class="math-callout__name">(Independence via Densities)</span></p>

If $X, Y$ have joint density $f_{X,Y}$ and marginal densities $f_X, f_Y$, the following are equivalent:
- $X, Y$ are independent.
- $f_{X,Y}(x, y) = f_X(x)\,f_Y(y)$ for all $x, y$.

</div>

### Sum of Continuous Random Variables

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 68</span><span class="math-callout__name">(Convolution for Continuous RVs)</span></p>

If $X, Y$ are independent continuous random variables, then $Z = X + Y$ is also continuous with density given by the **convolution** of $f_X$ and $f_Y$:

$$f_Z(z) = \int_{-\infty}^{\infty} f_X(x)\,f_Y(z - x)\,dx.$$

</div>

### Marginal Density

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 70</span><span class="math-callout__name">(Marginal Density)</span></p>

$$f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x, y)\,dy, \qquad f_Y(y) = \int_{-\infty}^{\infty} f_{X,Y}(x, y)\,dx.$$

</div>

### Conditional Density

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 71</span><span class="math-callout__name">(Conditional Density)</span></p>

For continuous random variables $X, Y$, the **conditional density** of $X$ given $Y = y$ is

$$f_{X \mid Y}(x \mid y) = \frac{f_{X,Y}(x, y)}{f_Y(y)},$$

defined when $f_Y(y) > 0$.

</div>

### Multivariate Normal Distribution

The standard $n$-dimensional normal distribution has density

$$f(t_1, \ldots, t_n) = \varphi(t_1)\varphi(t_2)\cdots\varphi(t_n) = \frac{1}{(\sqrt{2\pi})^n} e^{-(t_1^2 + \cdots + t_n^2)/2}.$$

This is a **radially symmetric** function: $f(t) = (2\pi)^{-n/2} e^{-r^2/2}$ where $r^2 = t_1^2 + \cdots + t_n^2$, so $f$ depends only on the distance from the origin.

If $Z = (Z_1, \ldots, Z_n)$ has this density, then $Z_1, \ldots, Z_n$ are independent $N(0, 1)$ (by Theorem 67). Moreover, $Z / \lVert Z \rVert$ is uniformly distributed on the $n$-dimensional sphere. Any linear combination $\sum u_i Z_i$ (with $\sum u_i^2 = 1$) also has distribution $N(0, 1)$ -- this is the "stability under summation" property.

**General multivariate normal.** A random vector with density $c \cdot e^{-Q(t)}$, where $Q(t) = (t - \mu)^T M (t - \mu)$ is a quadratic form with $M$ positive semidefinite. The coordinates are generally **not** independent.

### Conditioning for Continuous RVs

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 72</span><span class="math-callout__name">(Conditional CDF and Density)</span></p>

For a random variable $X$ on $(\Omega, \mathcal{F}, P)$ and an event $B \in \mathcal{F}$ with $P(B) > 0$:

$$F_{X \mid B}(x) := P(X \le x \mid B).$$

The corresponding density $f_{X \mid B}$ satisfies: if $B = \lbrace X \in S \rbrace$ for some $S \subseteq \mathbb{R}$, then

$$f_{X \mid B}(x) = \begin{cases} \frac{f_X(x)}{P(X \in S)} & \text{if } x \in S, \\ 0 & \text{otherwise}. \end{cases}$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 73</span><span class="math-callout__name">(Decomposition of Density via Partition)</span></p>

Let $X$ be a continuous random variable and $B_1, B_2, \ldots$ a partition of $\Omega$. Then

$$F_X(x) = \sum_i P(B_i)\,F_{X \mid B_i}(x), \qquad f_X(x) = \sum_i P(B_i)\,f_{X \mid B_i}(x).$$

</div>

**Conditional density and expectation given $Y = y$:**
- $f_{X \mid Y}(x \mid y) := \frac{f_{X,Y}(x, y)}{f_Y(y)}$ is the density of $X$ given $Y = y$.
- $\mathbb{E}(X \mid Y = y) := \int_{-\infty}^{\infty} x \, f_{X \mid Y}(x \mid y)\,dx$.
- $\mathbb{E}(g(X) \mid Y = y) = \int_{-\infty}^{\infty} g(x) \, f_{X \mid Y}(x \mid y)\,dx$.
- Law of total expectation: $\mathbb{E}(X) = \int_{-\infty}^{\infty} \mathbb{E}(X \mid Y = y)\,f_Y(y)\,dy$.
- Equivalently: $\mathbb{E}(X) = \mathbb{E}(\mathbb{E}(X \mid Y))$.

## 9 Inequalities

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 74</span><span class="math-callout__name">(Markov's Inequality)</span></p>

Let $X$ be a random variable with $X \ge 0$ and let $a > 0$. Then

$$P(X \ge a) \le \frac{\mathbb{E}(X)}{a}.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By the law of total expectation with partition $\lbrace X \ge a \rbrace$ and $\lbrace X < a \rbrace$:

$$\mathbb{E}(X) = P(X \ge a)\,\mathbb{E}(X \mid X \ge a) + P(X < a)\,\mathbb{E}(X \mid X < a) \ge P(X \ge a) \cdot a + 0.$$

Dividing by $a$ gives the result.

</details>
</div>

**Remarks:**
- Setting $a = b \cdot \mathbb{E}(X)$: $P(X \ge b \cdot \mathbb{E}(X)) \le 1/b$.
- Markov's inequality is tight: consider $X = a$ with probability $1/a$ and $X = 0$ otherwise (so $\mathbb{E}(X) = 1$).

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 75</span><span class="math-callout__name">(Algorithm Runtime)</span></p>

A randomized algorithm has expected runtime $\mathbb{E}(X) = n^2$. By Markov's inequality (since runtime is non-negative):

$$P(X \ge c \cdot n^2) \le \frac{1}{c}.$$

For instance, the probability that the algorithm runs more than $10 n^2$ steps is at most $1/10$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 76</span><span class="math-callout__name">(Chebyshev's Inequality)</span></p>

Let $X$ have finite mean $\mu$ and variance $\sigma^2$. Then

$$P(\lvert X - \mu \rvert \ge t \cdot \sigma) \le \frac{1}{t^2}.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $Y = (X - \mu)^2$. Then $Y \ge 0$ and $\mathbb{E}(Y) = \sigma^2$. Apply Markov's inequality to $Y$ with $a = t^2 \sigma^2$:

$$P(\lvert X - \mu \rvert \ge t\sigma) = P(Y \ge t^2 \sigma^2) \le \frac{\sigma^2}{t^2 \sigma^2} = \frac{1}{t^2}.$$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 77</span><span class="math-callout__name">(Chernoff's Inequality)</span></p>

Let $X = \sum_{i=1}^{n} X_i$ where $X_i$ are independent random variables taking values $\pm 1$ with probability $1/2$ each. Then for $t > 0$:

$$P(X \le -t \cdot \sigma) = P(X \ge t \cdot \sigma) \le e^{-t^2/2},$$

where $\sigma = \sigma_X = \sqrt{n}$.

</div>

Chernoff's inequality gives an exponentially decaying tail bound, much stronger than Chebyshev. It applies when we have structural knowledge about $X$ (sum of independent bounded variables), giving better estimates than the general-purpose inequalities.

## 10 Limit Theorems

### 10.1 Laws of Large Numbers

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 78</span><span class="math-callout__name">(Strong Law of Large Numbers)</span></p>

Let $X_1, X_2, \ldots$ be identically distributed independent random variables with mean $\mu$ and variance $\sigma^2$. Define the **sample mean** $\bar{X}_n = (X_1 + \cdots + X_n)/n$. Then

$$\lim_{n \to \infty} \bar{X}_n = \mu \quad \text{almost surely (i.e. with probability 1).}$$

</div>

The sample mean converges to the true mean with probability 1. This is the theoretical justification for repeating measurements and taking averages.

We will not prove the strong law here but instead prove a weaker version:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 79</span><span class="math-callout__name">(Weak Law of Large Numbers)</span></p>

Let $X_1, X_2, \ldots$ be identically distributed independent random variables with mean $\mu$ and variance $\sigma^2$. Then for every $\varepsilon > 0$:

$$\lim_{n \to \infty} P(\lvert \bar{X}_n - \mu \rvert > \varepsilon) = 0.$$

We say $\bar{X}_n$ **converges in probability** to $\mu$ and write $\bar{X}_n \xrightarrow{P} \mu$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By Theorem 59, $\mathbb{E}(\bar{X}_n) = (\mathbb{E}(X_1) + \cdots + \mathbb{E}(X_n))/n = \mu$. Since the $X_i$ are independent, by Theorem 35: $\mathrm{var}(\bar{X}_n) = (\mathrm{var}(X_1) + \cdots + \mathrm{var}(X_n))/n^2 = \sigma^2/n$.

Applying Chebyshev's inequality with $a = \sqrt{n}\varepsilon/\sigma$:

$$P(\lvert \bar{X}_n - \mu \rvert > \varepsilon) \le \frac{\sigma^2}{n\varepsilon^2} \to 0 \text{ as } n \to \infty.$$

</details>
</div>

The proof also gives an explicit error bound: $P(\lvert \bar{X}_n - \mu \rvert > \varepsilon) \le \sigma^2/(n\varepsilon^2)$.

### 10.2 Central Limit Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 81</span><span class="math-callout__name">(Central Limit Theorem)</span></p>

Let $X_1, X_2, \ldots$ be identically distributed independent random variables with mean $\mu$ and variance $\sigma^2$. Define the standardized sum

$$Y_n = \frac{(X_1 + \cdots + X_n) - n\mu}{\sqrt{n} \cdot \sigma}.$$

Then $Y_n \xrightarrow{d} N(0, 1)$. That is, if $F_n$ is the CDF of $Y_n$:

$$\lim_{n \to \infty} F_n(x) = \Phi(x) \quad \text{for every } x \in \mathbb{R}.$$

</div>

We say $Y_n$ **converges in distribution** to $N(0, 1)$.

**Key observations:**
- By the same argument as in the WLLN proof, $\mathbb{E}(Y_n) = 0$ and $\mathrm{var}(Y_n) = 1$. But the CLT says much more: the variance does not collapse (unlike in the LLN where $\bar{X}_n \to \mu$), and the distribution approaches a specific shape.
- If all $X_i$ are normal $N(\mu_i, \sigma_i^2)$, then by the stability of normal distributions, $Y_n$ is exactly $N(0, 1)$. So no distribution other than normal can be a limit here.

**Special case (Bernoulli).** If $X_i \sim \mathrm{Bern}(p)$, then $\sum X_i \sim \mathrm{Bin}(n, p)$, and the CLT says that the rescaled binomial $(\mathrm{Bin}(n, p) - np)/\sqrt{np(1-p)}$ converges in distribution to $N(0, 1)$.

**De Moivre-Laplace theorem.** A more precise local version: for $k$ close to $np$,

$$\binom{n}{k} p^k (1-p)^{n-k} \approx \frac{1}{\sqrt{2\pi n p(1-p)}} \, e^{-\frac{(k - np)^2}{2np(1-p)}}.$$

More precisely, if $\lvert k - pn \rvert < c\sqrt{np(1-p)}$ for some fixed $c$, then the ratio of the two sides tends to $1$ as $n \to \infty$.

## 11 Statistics -- Introduction

So far we have studied models of random situations: given random variables, their distributions (joint pmf/pdf), we asked "what will happen" -- what is the probability of an event, the expected value, etc. Now we reverse the perspective: from observed data we want to deduce what model produced them, or what properties it has. The problems fall into several types:

- **Point estimates** -- we want to determine the value of some unknown parameter (often the mean of an unknown distribution).
- **Interval estimates** -- we want to find an interval that contains an unknown parameter with, say, 99% probability.
- **Hypothesis testing** -- we want to decide whether the observed data support our hypothesis or refute it.
- **Linear regression** -- what is the relationship between two measured quantities?

**Exploratory data analysis.** Before applying formal methods, one typically performs exploratory analysis: visualising the data (histograms, boxplots, violin plots, scatter plots, etc.), checking for missing or erroneous values, and looking for interesting patterns. The key caveat is distinguishing genuine hidden structure from random fluctuations -- with enough data one can find spurious correlations in anything.

Confirmatory analysis is the mathematically rigorous discipline that addresses this. Its foundation is the way we select objects from a large set (the population) to measure. The logical approach is to sample *uniformly at random without replacement*; for simplicity of analysis we instead sample *uniformly at random with replacement*.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Random Sample)</span></p>

A sequence of i.i.d. random variables $X_1, \ldots, X_n$ with a common distribution is called a **random sample** of size $n$. If all $X_i$ have CDF $F$, we write $X_1, \ldots, X_n \sim F$.

</div>

Note that in practice we do not observe the random variables themselves; we only obtain their concrete values for our measurements, called **realisations** (observations). We denote by $x_i$ the measured value of $X_i$, i.e. $x_i = X_i(\omega)$ for the elementary event $\omega \in \Omega$ that actually occurred.

**Parametric vs. nonparametric models.** When looking for a model that explains the data, we can either allow an arbitrary CDF of the unknown random variable (nonparametric), or restrict to a parametric family $\lbrace F_\vartheta : \vartheta \in \Theta \rbrace$ where $\vartheta$ is an unknown parameter and $\Theta$ is the parameter space. Examples:

- $\mathrm{Pois}(\lambda)$: parameter $\vartheta = \lambda$, $\Theta = \mathbb{R}^+$
- $U(a, b)$: parameter $\vartheta = (a, b)$, $\Theta = \mathbb{R}^2$
- $N(\mu, \sigma^2)$: parameter $\vartheta = (\mu, \sigma)$, $\Theta = \mathbb{R} \times \mathbb{R}^+$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Statistic)</span></p>

A **statistic** is any function $T = T(X_1, \ldots, X_n)$ of the random sample. It is itself a random variable -- one that derives its randomness solely from $X_1, \ldots, X_n$ (no additional randomness such as a coin toss is involved). The role of statisticians is to find suitable statistics that solve the problems mentioned above (e.g. point estimation of an unknown parameter).

</div>

## 12 Interval Estimates

Let $T_1 \le T_2$ be two statistics. We say they determine a **confidence interval** (CI) with confidence level $1 - \alpha$ ($1 - \alpha$ confidence interval) if

$$P(T_1 \le \vartheta \le T_2) \ge 1 - \alpha.$$

Sometimes we also consider one-sided confidence intervals (the case $T_1 = -\infty$ or $T_2 = +\infty$). We may also require the stronger property $P(\vartheta > T_2) = P(\vartheta < T_1) = \alpha/2$.

Note that $\vartheta$ is a parameter -- its value is unknown but fixed (e.g. the true number of left-handers in the country). The randomness is in the boundary points of the interval: $T_1, T_2$ are random variables. Which interval we get as our approximation of reality depends on the realisation of the random sample $X_1, \ldots, X_n$.

Our first statistical task is essentially an exercise in working with the normal distribution. Imagine measuring a physical quantity by repeated weighing on the same scale, where the measurement error is normally distributed with known standard deviation $\sigma$. We seek $\vartheta$, the unknown true weight. The measurements are realisations of random variables with distribution $N(\vartheta, \sigma^2)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 82</span><span class="math-callout__name">(Confidence Interval for Normal RV with Known Variance)</span></p>

Let $X_1, \ldots, X_n$ be a random sample from $N(\vartheta, \sigma^2)$, where $\sigma$ is known and $\vartheta$ is to be determined. Choose $\alpha \in (0, 1)$. Let $\Phi(z_{\alpha/2}) = 1 - \alpha/2$. Denote $S_n = (X_1 + \cdots + X_n)/n$ (the sample mean) and set

$$C_n := \left[ S_n - z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}},\; S_n + z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}} \right].$$

Then $P(C_n \ni \vartheta) = 1 - \alpha$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Define $Y_n = \frac{S_n - \vartheta}{\sigma / \sqrt{n}}$. Then $Y_n \sim N(0, 1)$: we compute $\mathbb{E}(Y_n) = 0$ and $\mathrm{var}(Y_n) = 1$ (as in the proof of Theorem 79). We also use the closure of the normal distribution under sums. It remains to note that

$$P(\vartheta > S_n + z_{\alpha/2}\sigma/\sqrt{n}) = P(Y_n < -z_{\alpha/2}) = \Phi(-z_{\alpha/2}) = \alpha/2.$$

</details>
</div>

Now we pass to the more general case: if we have enough summands (so that we can apply the CLT), we can construct a confidence interval in exactly the same way as for the normal distribution -- except it will not have exactly the required error probability, but only asymptotically.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 83</span><span class="math-callout__name">(Confidence Interval via CLT)</span></p>

Let $X_1, \ldots, X_n$ be a random sample from some distribution with mean $\vartheta$ and variance $\sigma^2$. Suppose $\sigma$ is known and we choose $\alpha \in (0, 1)$. Choose $z_{\alpha/2}$, $S_n$, and $C_n$ as in the previous theorem. Then

$$\lim_{n \to \infty} P(C_n \ni \vartheta) = 1 - \alpha.$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Define $Y_n$ as before and note that the CLT says $Y_n \xrightarrow{d} N(0, 1)$. Therefore

$$P(\vartheta > S_n + z_{\alpha/2}\sigma/\sqrt{n}) = P(Y_n < -z_{\alpha/2}) \xrightarrow{n \to \infty} \Phi(-z_{\alpha/2}) = \alpha/2.$$

</details>
</div>

We have dropped the requirement of normality but still assume the variance is known. To remove even this assumption we use Student's t-test, discussed below.

## 13 Point Estimates

Consider a random sample $X_1, \ldots, X_n \sim F_\vartheta$ and suppose we want to estimate some function $g(\vartheta)$ of the unknown parameter. For instance, for $N(\mu, \sigma^2)$ we might want to estimate the variance, so $\vartheta = (\mu, \sigma)$ and $g(\vartheta) = \sigma^2$. Even for a distribution with a single parameter (e.g. $\mathrm{Geom}(p)$), we might prefer to estimate the mean $1/p$ rather than $p$ itself.

We seek a statistic $\hat{\Theta}_n = \hat{\Theta}_n(X_1, \ldots, X_n)$ that "well approximates" the value $g(\vartheta)$. Since our estimate is a random variable, it cannot always equal the true value. So what can we realistically expect?

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 84</span><span class="math-callout__name">(Properties of Point Estimates)</span></p>

For a random sample $X_1, \ldots, X_n \sim F_\vartheta$ and a function $g$, we call the point estimate $\hat{\Theta}_n$:

- **unbiased** if $\mathbb{E}(\hat{\Theta}_n) = g(\vartheta)$,
- **asymptotically unbiased** if $\lim_{n \to \infty} \mathbb{E}(\hat{\Theta}_n) = g(\vartheta)$,
- **consistent** if $\hat{\Theta}_n \xrightarrow{P} g(\vartheta)$ (convergence in probability, as in Theorem 79).

We further define:

- **bias**: $\mathrm{bias}_\vartheta(\hat{\Theta}_n) = \mathbb{E}(\hat{\Theta}_n) - \vartheta$ (the notation indicates that the bias may depend on which statistic we chose and on the true parameter value $\vartheta$),
- **mean squared error** (MSE): $MSE_\vartheta(\hat{\Theta}_n) = \mathbb{E}((\hat{\Theta}_n - \vartheta)^2)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(On MSE and Bias)</span></p>

- An estimate can be unbiased yet useless: e.g. one that gives a value one million too high half the time and one million too low the other half.
- Consistency partially addresses this: it says that for growing $n$ the probability of large errors tends to zero.
- If we care about behaviour for small $n$ as well, we use the MSE. By Markov's inequality applied to $Y = (\hat{\Theta}_n - \vartheta)^2$, small MSE implies small probability of large deviation.
- To achieve small MSE, we need both small bias and small variance. Sometimes a (slightly) biased estimate with smaller variance gives a better MSE.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 85</span><span class="math-callout__name">(Bias-Variance Decomposition)</span></p>

$$MSE(\hat{\Theta}_n) = \mathrm{bias}_\vartheta(\hat{\Theta}_n)^2 + \mathrm{var}_\vartheta(\hat{\Theta}_n).$$

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

By Theorem 35, $\mathrm{var}(\hat{\Theta}_n) = \mathrm{var}(\hat{\Theta}_n - \vartheta)$. By Theorem 34 we can write this as

$$\mathbb{E}\!\left((\hat{\Theta}_n - \vartheta)^2\right) - \left(\mathbb{E}(\hat{\Theta}_n - \vartheta)\right)^2.$$

The first term is $MSE_\vartheta(\hat{\Theta}_n)$ and the second is $\mathrm{bias}_\vartheta^2$.

Alternative direct proof: let $\mu = \mathbb{E}(\hat{\Theta}_n)$. Then

$$MSE_\vartheta(\hat{\Theta}_n) = \mathbb{E}\!\left(((\hat{\Theta}_n - \mu) - (\vartheta - \mu))^2\right) = \mathbb{E}((\hat{\Theta}_n - \mu)^2) - 2\mathbb{E}((\hat{\Theta}_n - \mu)(\vartheta - \mu)) + \mathbb{E}((\vartheta - \mu)^2).$$

The first term is $\mathrm{var}(\hat{\Theta}_n)$, the last is $\mathrm{bias}^2$, and the middle term is zero because $\mathbb{E}(\hat{\Theta}_n - \mu) = 0$ and $\vartheta - \mu$ is a constant.

</details>
</div>

**Sample mean and sample variance.** Let us look at several concrete estimates and their properties.

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i \qquad \text{(sample mean)}$$

$$S_n^2 = \frac{1}{n}\sum_{i=1}^{n} (X_i - \bar{X}_n)^2 \qquad \text{(sample variance)}$$

$$\widehat{S}_n^2 = \frac{1}{n-1}\sum_{i=1}^{n} (X_i - \bar{X}_n)^2 \qquad \text{(corrected sample variance)}$$

The two variants of sample variance are related by $\widehat{S}_n^2 = \frac{n}{n-1} S_n^2$; the factor $\frac{n}{n-1}$ is called **Bessel's correction**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 86</span><span class="math-callout__name">(Properties of Sample Mean and Variance)</span></p>

1. $\bar{X}_n$ is a consistent unbiased estimate of $\mu$.
2. $S_n^2$ is a consistent asymptotically unbiased estimate of $\sigma^2$.
3. $\widehat{S}_n^2$ is a consistent unbiased estimate of $\sigma^2$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

1. The law of large numbers (Theorem 79) gives consistency. In its proof we computed $\mathbb{E}(\bar{X}_n) = \mu$, so the estimate is unbiased.

2. We omit the proof of consistency. But $\mathbb{E}(S_n^2)$: by a trick similar to the MSE computation -- add and subtract $\mu$ and use the fact that $\mathbb{E}(X_1) = \cdots = \mathbb{E}(X_n)$ -- one shows that $\mathbb{E}(S_n^2) = \frac{n-1}{n}\sigma^2$, which tends to $\sigma^2$.

3. Follows immediately from 2: $\mathbb{E}(\widehat{S}_n^2) = \frac{n}{n-1}\mathbb{E}(S_n^2) = \sigma^2$.

</details>
</div>

### 13.1 Method of Moments

Let us now consider a general method for constructing point estimates. Recall that the $r$-th moment of a random variable $X$ is $\mathbb{E}(X^r)$. We define two related notions:

- $m_r(\vartheta) := \mathbb{E}(X^r)$ for $X \sim F_\vartheta$ -- the $r$-th (theoretical) moment,
- $\widehat{m_r(\vartheta)} := \frac{1}{n}\sum_{i=1}^{n} X_i^r$ for a random sample $X_1, \ldots, X_n$ from $F_\vartheta$ -- the $r$-th **sample moment**.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 87</span></p>

$\widehat{m_r(\vartheta)}$ is an unbiased consistent estimate of $m_r(\vartheta)$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

Apply Theorem 86 to the random variables $X_1^r, \ldots, X_n^r$.

</details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Method of Moments)</span></p>

**Method of moments:** choose $\vartheta$ that solves the system of equations

$$m_r(\vartheta) = \widehat{m_r(\vartheta)}, \qquad r = 1, \ldots, k,$$

where $k$ is the number of real-valued components of $\vartheta$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 88</span><span class="math-callout__name">(Bernoulli -- Method of Moments)</span></p>

Continuing with our left-handers example: $X_i$ is 1 if person $i$ is left-handed, 0 otherwise. So $X_1, \ldots, X_n$ is a random sample from $\mathrm{Bern}(\vartheta)$, where $\vartheta$ is the (unknown) proportion of left-handers.

We have $m_1(\vartheta) = \vartheta$ and $\widehat{m_1(\vartheta)} = \bar{X}_n$. So the method of moments gives $\hat{\Theta}_n = \bar{X}_n$ -- the natural estimate.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 89</span><span class="math-callout__name">(Normal -- Method of Moments)</span></p>

Let $X_1, \ldots, X_n$ be a random sample from $N(\mu, \sigma^2)$ (e.g. measuring the height of randomly chosen people). The parameter is $\vartheta = (\mu, \sigma)$. Determine $\vartheta$ by the method of moments.

The first moment is $m_1(\vartheta) = \mu$. The second moment is $m_2(\vartheta) = \mathbb{E}(X^2) = \mathrm{var}(X) + \mathbb{E}(X)^2 = \sigma^2 + \mu^2$. Setting these equal to the sample moments:

$$\mu = m_1(\vartheta) = \widehat{m_1(\vartheta)} = \bar{X}_n,$$

$$\sigma^2 + \mu^2 = m_2(\vartheta) = \widehat{m_2(\vartheta)} = \frac{X_1^2 + \cdots + X_n^2}{n}.$$

Solving gives $\hat{\mu} = \bar{X}_n$ and $\hat{\sigma} = \sqrt{\frac{X_1^2 + \cdots + X_n^2}{n} - \bar{X}_n^2}$.

</div>

### 13.2 Maximum Likelihood (ML)

Now let us examine another method for constructing estimates. Sometimes both methods give the same result, sometimes they differ -- in which case we choose the one with better MSE.

Consider again a random sample $X = (X_1, \ldots, X_n)$ from a model with parameter $\vartheta$. Let $x = (x_1, \ldots, x_n)$ be a realisation. For discrete random variables the joint pmf is

$$p_X(x; \vartheta) = \prod_{i=1}^{n} p_{X_i}(x_i; \vartheta).$$

For continuous random variables, $x$ has the joint density

$$f_X(x; \vartheta) = \prod_{i=1}^{n} f_{X_i}(x_i; \vartheta).$$

We define the **likelihood** $L(x; \vartheta)$ as $p_X(x; \vartheta)$ or $f_X(x; \vartheta)$ depending on the type of the random variables.

Note the key difference from earlier: before, $\vartheta$ was fixed and we varied $x$; now $x$ is fixed (observed data) and we vary $\vartheta$, looking for the most plausible parameter value.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Maximum Likelihood Method)</span></p>

**Maximum likelihood (ML):** choose $\vartheta$ for which $L(x; \vartheta)$ is maximal.

</div>

Computationally it is often easier to maximise the **log-likelihood** $\ell(x; \vartheta) = \log L(x; \vartheta)$, which gives the same result by monotonicity of the logarithm.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 90</span><span class="math-callout__name">(Bernoulli -- Maximum Likelihood)</span></p>

Again counting left-handers: $X_i$ is 1 if the $i$-th person is left-handed, 0 otherwise. So $X_1, \ldots, X_n \sim \mathrm{Bern}(\vartheta)$. Determine $\vartheta$ by maximum likelihood.

Suppose exactly $k$ of the $n$ people are left-handed. Then

$$L(x; \vartheta) = \vartheta^k (1 - \vartheta)^{n-k}, \qquad \ell(x; \vartheta) = k \log \vartheta + (n-k)\log(1-\vartheta).$$

Setting $\ell'(x; \vartheta) = \frac{k}{\vartheta} - \frac{n-k}{1-\vartheta} = 0$ gives $\vartheta = k/n$. Once again we arrive at $\hat{\Theta}_n = \bar{X}_n$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 91</span><span class="math-callout__name">(Normal -- Maximum Likelihood)</span></p>

Let $X_1, \ldots, X_n$ be a random sample from $N(\mu, \sigma^2)$ (e.g. measuring height). The parameter is $\vartheta = (\mu, \sigma)$. Determine $\vartheta$ by maximum likelihood.

Since we have continuous random variables, we work with the density $f(x_i; \vartheta) = \frac{1}{\sigma\sqrt{2\pi}} e^{-(x_i - \mu)^2/(2\sigma^2)}$, so

$$\ell(x_i; \vartheta) = -\frac{(x_i - \mu)^2}{2\sigma^2} - \log\sigma - \log\sqrt{2\pi},$$

$$\ell(x; \vartheta) = -\sum_{i=1}^{n}\frac{(x_i - \mu)^2}{2\sigma^2} - n\log\sigma - n\log\sqrt{2\pi}.$$

Setting partial derivatives to zero:

$$\frac{\partial}{\partial \mu}\ell(x;\vartheta) = \sum_{i=1}^{n}\frac{2(x_i - \mu)}{2\sigma^2} = 0, \qquad \frac{\partial}{\partial \sigma}\ell(x;\vartheta) = \sum_{i=1}^{n}\frac{2(x_i - \mu)^2}{2\sigma^3} - \frac{n}{\sigma} = 0.$$

From the first equation we get $\hat{\mu} = \bar{X}_n$. From the second, $\hat{\sigma}^2 = S_n^2$.

</div>

### 13.3 Confidence Intervals for Normal RV with Unknown Variance -- Student's t-test

We now introduce **Student's $t$-distribution**. Let $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$ be i.i.d. Let $\bar{X}_n$ be their sample mean and $\widehat{S}_n^2$ their corrected sample variance. The **Student's $t$-distribution with $n - 1$ degrees of freedom** is the distribution of

$$\frac{\bar{X}_n - \mu}{\widehat{S}_n / \sqrt{n}}.$$

This distribution is the same regardless of $\mu$ and $\sigma$ -- it depends only on the number of summands. Its density has a shape similar to the standard normal but with heavier tails (it is an even function). The CDF is denoted $\Psi_{n-1}$. Values can be found in tables or computed via library functions (`pt(x,n-1)` in R, `scipy.stats.t.cdf(x,n-1)` in Python).

Note that if we replace $\widehat{S}_n$ by the true $\sigma$, we get $({\bar{X}_n - \mu})/({\sigma/\sqrt{n}})$, which has distribution $N(0,1)$. So Student's $t$-distribution is an approximation of $N(0,1)$: since $\widehat{S}_n^2$ is a consistent estimate of $\sigma^2$, the $t$-distribution with $n - 1$ degrees of freedom approaches $N(0, 1)$ as $n \to \infty$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 92</span><span class="math-callout__name">(Confidence Interval via Student's $t$-distribution)</span></p>

Let $X_1, \ldots, X_n$ be a random sample from $N(\mu, \sigma^2)$. The parameter is $\vartheta = (\mu, \sigma)$, but we want to determine only $\mu$. Choose $\alpha \in (0, 1)$. Let $\Psi_{n-1}(t_{\alpha/2}) = 1 - \alpha/2$. Define

$$C_n' := [\bar{X}_n - \delta,\; \bar{X}_n + \delta] \qquad \text{where } \delta = t_{\alpha/2}\frac{\widehat{S}_n}{\sqrt{n}}.$$

Then $P(C_n' \ni \mu) = 1 - \alpha$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

As with previous confidence intervals, note that

$$\bar{X}_n - \delta < \mu < \bar{X}_n + \delta \quad \Longleftrightarrow \quad -t_{\alpha/2} < \frac{\bar{X}_n - \mu}{\widehat{S}_n/\sqrt{n}} < t_{\alpha/2}.$$

Now use the definition of $t_{\alpha/2}$ and the symmetry of the $t$-distribution: $\Psi_{n-1}(-t_{\alpha/2}) = \alpha/2$.

</details>
</div>

## 14 Hypothesis Testing

In this chapter we introduce the basics of statistical hypothesis testing, which is a cornerstone of modern science.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 93</span><span class="math-callout__name">(Are Boys and Girls Born Equally Often?)</span></p>

If we have access to all birth records, we can count exactly. But the counts fluctuate randomly over time -- our task is to determine whether the data are explainable by random fluctuation, or whether one sex is born more often.

This was studied in 1710 by the Scottish polymath John Arbuthnot. He found 82 years out of 82 had more boys, and concluded boys are born more often. Under the assumption of equal probability, the chance of this outcome is about $1/2^{82}$, essentially zero. This is the approach we now call the **sign test**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 95</span><span class="math-callout__name">(Biased Coin?)</span></p>

Out of 1000 coin flips, heads came up 472 times. Is this (or rather, the coin that produced it) suspicious?

We set a threshold $x$: if the number of heads is less than $x$, we declare the coin biased. We choose $x$ to balance two possible errors: too small means we miss many biased coins; too close to 500 means we flag fair coins too often. The standard approach: first fix a **significance level** $\alpha$, then choose $x$ so that falsely flagging a fair coin has probability (approximately) $\alpha$.

</div>

The general procedure (following R. A. Fisher):

- We consider two hypotheses: $H_0$ and $H_1$.
- $H_0$ is the **null hypothesis** -- the default, conservative model (e.g. "the drug has no effect", "the coin is fair").
- $H_1$ is the **alternative hypothesis** -- the alternative model of interest.
- The output is either: **reject $H_0$** (the data tell us our description of the world is incorrect), or **do not reject $H_0$** (the data are consistent with $H_0$).

Since all observations are results of a random process, we can never be certain. We may commit two types of errors:

- **Type I error:** Falsely rejecting $H_0$ when it is true ("false alarm").
- **Type II error:** Failing to reject $H_0$ when it is false ("missed opportunity").

The procedure in detail:

1. Choose a suitable statistical model.
2. Choose the **significance level** $\alpha$. The test is constructed so that the probability of a Type I error (falsely rejecting $H_0$) is $\alpha$. Typically $\alpha = 0.05$.
3. Determine a **test statistic** $T = h(X_1, \ldots, X_n)$.
4. Determine the **critical region** (rejection region) $W$.
5. Measure the data $x_1, \ldots, x_n$, compute $t = h(x_1, \ldots, x_n)$.
6. **Decision rule:** reject $H_0$ if $t \in W$.
7. By construction, $\alpha = P(h(X) \in W; H_0)$.
8. Define $\beta = P(h(X) \notin W; H_1)$ as the probability of Type II error. The value $1 - \beta$ is called the **power** of the test.

**$p$-value.** Often we also compute the **$p$-value**: the smallest $\alpha$ for which we would reject $H_0$. Equivalently, it is the probability of observing data "at least as extreme" as what we actually measured, assuming $H_0$ is true.

Note that $P(h(X) \in W \mid H_0)$ uses a semicolon rather than a conditional bar: the world we live in (and which hypothesis describes it) is not treated as a random event. This changes in Bayesian statistics, which we do not cover here.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 96</span><span class="math-callout__name">(One-Sample Z-test)</span></p>

Is our drink poured to the correct measure? More precisely: is the mean volume what it says on the menu?

- $X_1, \ldots, X_n$ random sample from $N(\vartheta, \sigma^2)$.
- Assume $\sigma^2$ is known from experience.
- $H_0: \vartheta = \mu$ (e.g. $\mu = 0.5\,\ell$). $\quad H_1: \vartheta \neq \mu$.

We use the statistic $Z = \frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}}$. If $H_0$ is true, $Z \sim N(0,1)$. We choose the critical region $W = \lbrace x : \lvert x \rvert > z_{\alpha/2} \rbrace$ where $z_{\alpha/2} = \Phi^{-1}(1 - \alpha/2)$. This gives Type I error probability exactly $\alpha$. (The Type II error probability -- and hence the power -- depends on how close $\vartheta$ is to $\mu$.)

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 97</span><span class="math-callout__name">(Two-Sample Z-test)</span></p>

Are our drinks poured the same as our friend's? More precisely: is the mean the same for both of us?

- $X_1, \ldots, X_n$ random sample from $N(\vartheta_X, \sigma^2)$, $Y_1, \ldots, Y_m$ random sample from $N(\vartheta_Y, \sigma^2)$.
- Assume $\sigma^2$ is known.
- $H_0: \vartheta_X = \vartheta_Y$. $\quad H_1: \vartheta_X \neq \vartheta_Y$.

Let $S = \bar{X}_n - \bar{Y}_m$. If $H_0$ holds, $\mathbb{E}(S) = 0$ and $\mathrm{var}(S) = \sigma^2/n + \sigma^2/m$. So

$$Z = \frac{\bar{X}_n - \bar{Y}_m}{\sigma\sqrt{1/n + 1/m}} \sim N(0, 1)$$

under $H_0$, and we proceed as in the one-sample case.

**Without known variance (T-test).** If $\sigma^2$ is unknown, we replace it with the sample variance $\widehat{S}_n^2 = \frac{1}{n-1}\sum_{i=1}^{n}(X_i - \bar{X}_n)^2$ and use

$$T = \frac{\bar{X}_n - \mu}{\widehat{S}_n/\sqrt{n}},$$

which has Student's $t$-distribution with $n - 1$ degrees of freedom (under $H_0$). The critical region becomes $W = \lbrace x : \lvert x \rvert > t_{\alpha/2} \rbrace$ where $t_{\alpha/2} = \Psi_{n-1}^{-1}(1 - \alpha/2)$.

</div>

**$p$-hacking.** A tempting but illegitimate approach: measure data, look for something interesting, and present the correlation as proven. With enough data, spurious correlations appear by chance alone (cf. Ramsey's theorem). Worse: repeating tests until a desired result appears is equivalent to the "child's game" of rolling until you get a six.

The correct approach is **reproducibility**: after exploratory analysis, collect new independent data and analyse them confirmatively. One can also split data into a hypothesis-generating part and a confirmation part (cross validation). Meta-analyses provide community-level reproducibility checks.

### 14.1 Goodness-of-Fit Tests

So far we have mainly dealt with numerical data where computing the mean makes sense. Now we turn to **categorical data**, where values only have meaning as labels: eye colour, political preference, birthplace. The only thing we can examine is the count of each value's occurrence. Recall the multinomial distribution:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 98</span><span class="math-callout__name">(Multinomial Distribution)</span></p>

Given $p_1, \ldots, p_k \ge 0$ with $p_1 + \cdots + p_k = 1$. An experiment is repeated $n$ times; each time one of $k$ outcomes occurs, the $i$-th with probability $p_i$. Let $X_i$ count how many times outcome $i$ occurred. Then $(X_1, \ldots, X_k)$ has the **multinomial distribution** with parameters $n$ and $(p_1, \ldots, p_k)$.

- Trivial case: $X_i$ = number of rolls of a die that show face $i$.
- Important case: $X_i$ = count of the $i$-th letter, $i$-th word type, etc.
- $P(X_1 = x_1, \ldots, X_k = x_k) = \binom{n}{x_1, \ldots, x_k} p_1^{x_1} \cdots p_k^{x_k}$. $\quad (\ast)$

</div>

Now suppose the individual probabilities are unknown parameters $\vartheta = (\vartheta_1, \ldots, \vartheta_k) \in \Theta$ (where $\Theta$ is the set of all non-negative $k$-tuples summing to 1). We pick a favourite value $\vartheta^\ast \in \Theta$ and test the null hypothesis $H_0: \vartheta = \vartheta^\ast$. As in the previous chapter, we want a statistic $T$ and a critical region $W = (\gamma, \infty)$.

The general approach (useful beyond goodness-of-fit tests) is the **likelihood-ratio test**. From the chapter on point estimates, for each $\vartheta \in \Theta$ we compute the likelihood $L(x; \vartheta)$. We compared the ML estimate $\hat{\vartheta}$ with the hypothesised $\vartheta^\ast$:

$$G = 2\log\frac{L(x;\hat{\vartheta})}{L(x;\vartheta^\ast)}.$$

For the multinomial distribution with $L(x; \vartheta) = \prod_{i=1}^k \vartheta_i^{x_i}$ and ML estimate $\hat{\vartheta}_i = x_i/n$, substituting gives

$$G = 2\sum_{i=1}^{k} x_i \log\frac{x_i}{n\vartheta_i^\ast}.$$

Introducing notation $E_i = n\vartheta_i^\ast$ (expected count) and $O_i = x_i$ (observed count):

$$G = 2\sum_{i=1}^{k} O_i \log\frac{O_i}{E_i}.$$

By a Taylor approximation this is close to the **chi-squared statistic**:

$$\chi^2 = \sum_{i=1}^{k} \frac{(E_i - O_i)^2}{E_i}.$$

Both statistics measure the deviation of the data from the ideal $E_i = O_i$ (all tests had the same outcome).

**Determining the critical region.** We choose $\alpha$ and seek $\gamma$ such that $P(T > \gamma; H_0) = \alpha$ (for $T = G$ or $T = \chi^2$). Three options:

- For small $n$: enumerate all $k^n$ possible outcomes (or more efficiently all $\binom{n+k-1}{n}$ multinomial outcomes), compute $T$ for each, and pick $\gamma$ so that $T > \gamma$ in exactly $100\alpha\%$ of cases. This is the **exact test** but is often impractical.
- For larger $n$: simulate many random samples (each of $n$ ideal draws), compute $T$ each time, and take as $\gamma$ the worst $T$ from the best outcomes.
- For large $n$: use the **$\chi^2$-distribution** with $k - 1$ degrees of freedom. The $\chi^2$-distribution with $k - 1$ degrees of freedom is the distribution of $Q = Z_1^2 + \cdots + Z_{k-1}^2$ where $Z_i \sim N(0, 1)$ are i.i.d. It can be shown that for large $n$ the statistic $\chi^2$ (under $H_0$) approximately follows this distribution. We set $\gamma = F_Q^{-1}(1 - \alpha)$ and get $P(Q > \gamma) \doteq \alpha$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 99</span><span class="math-callout__name">(Is the Die Fair?)</span></p>

A die is rolled repeatedly. The individual faces come up with counts 92, 120, 88, 98, 95, 107. Is the die fair?

We have $n = 600$, $x = (92, 120, 88, 98, 95, 107)$. We test $H_0: \vartheta = \vartheta^\ast = (1/6, \ldots, 1/6)$. The ML estimate is $\hat{\vartheta} = (92/600, 120/600, 88/600, \ldots)$.

Is $\vartheta^\ast$ also sufficiently likely? We compute:

$$\chi^2 = \frac{(92-100)^2}{100} + \frac{(120-100)^2}{100} + \frac{(88-100)^2}{100} + \frac{(98-100)^2}{100} + \frac{(95-100)^2}{100} + \frac{(107-100)^2}{100} = 6.87.$$

Using the $\chi^2$-distribution with 5 degrees of freedom: `qchisq(0.95, 5)` $\doteq 11.1$. Since $6.87 < 11.1$, we cannot reject $H_0$. The $p$-value is `1 - pchisq(6.87, 5)` $\approx 0.23$ -- about 23% of die rolls are even more unbalanced than ours.

</div>

**Further extensions:**

- For examining the distribution of any continuous random variable $Y$, we can choose "bins" $B_1, \ldots, B_k$ (a partition of $\mathbb{R}$) and count how often $Y \in B_i$.
- An analogous test exists for testing independence of (discrete) random variables.

## 15 Linear Regression

*(Content to be added.)*

## 16 Nonparametric Statistics

### 16.1 Empirical Distribution Function

So far we have mostly discussed **parametric statistics** -- models described by a few parameters (e.g. mean and variance). This simplifies many things (we need less data), but it cannot help when the situation does not fit a simple parametric model.

If we cannot find parameters, what can we do? Every random variable is completely described by its CDF. We can try to approximate this CDF from data.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 100</span><span class="math-callout__name">(Empirical CDF)</span></p>

Let $X_1, \ldots, X_n \sim F$ be a random sample. The **empirical distribution function** (empirical CDF) is defined as

$$\widehat{F}_n(x) = \frac{\sum_{i=1}^{n} I(X_i \le x)}{n},$$

where $I(X_i \le x) = 1$ if $X_i \le x$ and $0$ otherwise.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 101</span><span class="math-callout__name">(Properties of the Empirical CDF)</span></p>

For fixed $x$:

- $\mathbb{E}(\widehat{F}_n(x)) = F(x)$,
- $\mathrm{var}(\widehat{F}_n(x)) = \frac{F(x)(1 - F(x))}{n}$,
- $\widehat{F}_n(x) \xrightarrow{P} F(x)$ (convergence in probability).

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof</summary>

By the weak law of large numbers.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Stronger Convergence Results)</span></p>

- The **Glivenko-Cantelli theorem** states that the convergence holds "for all $x$ simultaneously":

$$\lim_{n \to \infty} \sup_{x \in \mathbb{R}} \lvert F(x) - \widehat{F}_n(x) \rvert = 0 \qquad \text{almost surely.}$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 102</span><span class="math-callout__name">(Dvoretzky-Kiefer-Wolfowitz)</span></p>

Let $X_1, \ldots, X_n \sim F$ be i.i.d., $\widehat{F}_n$ their empirical CDF. Suppose $\mathbb{E}(X_i)$ is finite for each $i = 1, \ldots, n$. Choose $\alpha \in (0, 1)$ and set $\varepsilon = \sqrt{\frac{1}{2n}\log\frac{2}{\alpha}}$. Then

$$P\!\left(\widehat{F}_n(x) - \varepsilon \le F(x) \le \widehat{F}_n(x) + \varepsilon\right) \ge 1 - \alpha.$$

</div>

This gives an effective confidence band for the true CDF. A consequence is the **Kolmogorov-Smirnov test**.

### 16.2 Permutation Test

- We have two independent sets of random variables (random samples): $X_1, \ldots, X_n \sim F_X$ and $Y_1, \ldots, Y_m \sim F_Y$.
- We want to decide whether $H_0: F_X = F_Y$ or $H_1: F_X \neq F_Y$.
- Examples: program runtime before/after an optimisation, cholesterol level with/without a supplement, frequency of short words in texts by author X vs. Y.
- We know nothing about $F_X, F_Y$ (in particular, we do not assume normality).

Procedure:

- Choose a suitable statistic, e.g. $T(X_1, \ldots, X_n, Y_1, \ldots, Y_m) = \lvert \bar{X}_n - \bar{Y}_m \rvert$.
- Compute $t_{\mathrm{obs}} := T(X_1, \ldots, X_n, Y_1, \ldots, Y_m)$.
- Under $H_0$, "all permutations are the same": $X_i$ and $Y_j$ were generated from the same distribution.
- Randomly permute all $m + n$ values and for each permutation compute $T$, yielding $T_1, \ldots, T_{(m+n)!}$.
- The $p$-value is the probability that $T > t_{\mathrm{obs}}$:

$$p = \frac{1}{(m+n)!}\sum_j I(T_j > t_{\mathrm{obs}}).$$

- We reject $H_0$ if $p < \alpha$ (e.g. $\alpha = 0.05$).

**Practical improvement:** Enumerating all permutations may be too expensive. Instead, take $B$ random permutations and estimate the $p$-value as

$$\frac{1}{B}\sum_{j=1}^{B} I(T_j > t_{\mathrm{obs}}).$$

For sufficiently large $m, n$ the permutation test gives results similar to CLT-based tests, so it is most useful for moderate sample sizes.

## 17 Generating Random Variables

The basic method is **inverse sampling**, which uses Theorem 61 (the quantile function is the inverse of the CDF). For a continuous random variable, if $U \sim U(0, 1)$, then $X = F^{-1}(U)$ has CDF $F$.

This works for discrete random variables too: if the desired random variable takes values $x_1, x_2, \ldots$ with probabilities $p_1, p_2, \ldots$, we partition $[0, 1]$ into intervals of lengths $p_1, p_2, \ldots$ and if $U$ falls into the $i$-th interval, we output $x_i$.

**Rejection sampling.** When we cannot easily compute the quantile function but know the density, we use **rejection sampling**. We want to generate a random variable $X$ with density $f_X$, using an auxiliary random variable $Y$ with density $f_Y$ that we can generate, such that for some constant $c > 0$ and all $t$: $f_X(t) \le c \cdot f_Y(t)$.

The algorithm:

1. Generate $y$ from $Y$ (with density $f_Y$) and $u$ from $U \sim U(0, 1)$.
2. If $u \le \frac{f_X(y)}{c \cdot f_Y(y)}$, set $X := y$.
3. Otherwise reject $Y, U$ and go back to step 1.

Explanation: for a given $y$, the value $c \cdot u \cdot f_Y(y)$ is uniform on $[0, c \cdot f_Y(y)]$. If it is less than $f_X(y)$, the pair $(y, c \cdot u \cdot f_Y(y))$ is a random point under the graph of $f_X$, and so $y$ can serve as a realisation of $X$. If $c \cdot f_Y(y) > f_X(y)$, we reject and try again.

Other special techniques exist: e.g. a Gamma distribution is a sum of independent Exponentials; a Normal distribution can be conveniently generated using polar coordinates (Box-Muller transform).

## 18 Notation Summary

| Symbol | Meaning |
| --- | --- |
| $\lbrace X = x \rbrace = \lbrace \omega \in \Omega : X(\omega) = x \rbrace$ | event that $X$ equals $x$ |
| $p_X(x) = P(X = x)$ | probability mass function of $X$ |
| $p_{X,Y}(x, y) = P(X = x \mathbin{\&} Y = y)$ | joint pmf of $X, Y$ |
| $p_{X \mid Y}(x \mid y) = P(X = x \mid Y = y)$ | conditional pmf of $X$ given $Y$ |
| $F_X(x) = P(X \le x)$ | CDF of $X$ |
| $F_{X,Y}(x, y) = P(X \le x \mathbin{\&} Y \le y)$ | joint CDF of $X, Y$ |
| $f_X(x)$ | density of $X$ |
| $f_{X,Y}(x, y)$ | joint density of $X, Y$ |
| $f_{X \mid Y}(x \mid y) = f_{X,Y}(x, y)/f_Y(y)$ | conditional density of $X$ given $Y$ |
| $(a, b)$, $[a, b]$ | open, closed interval |
| $\langle a, b \rangle$ | inner product |

## 20 Bonuses

*(Not covered in lectures, included for the interested reader.)*

**Continuity of probability**

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 103</span><span class="math-callout__name">(Continuity of Probability)</span></p>

Let $A_1 \subseteq A_2 \subseteq A_3 \subseteq \cdots$ be events and $A = \bigcup_{i=1}^{\infty} A_i$. Then

$$P(A) = \lim_{i \to \infty} P(A_i).$$

</div>

**Borel-Cantelli lemma**

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 104</span><span class="math-callout__name">(Borel-Cantelli Lemma)</span></p>

Let events $A_1, A_2, \ldots$ satisfy $P(A_i) = p_i > 0$ for each $i$. Define "Nic" (none) as the event "none of $\lbrace A_i \rbrace$ occurred" and "Inf" as the event "infinitely many of $\lbrace A_i \rbrace$ occurred".

1. If $\sum_i p_i < \infty$, then $P(\mathrm{Inf}) = 0$.
2. If the events $A_1, A_2, \ldots$ are independent and $\sum_i p_i = \infty$, then $P(\mathrm{Nic}) = 0$ and $P(\mathrm{Inf}) = 1$.

</div>
