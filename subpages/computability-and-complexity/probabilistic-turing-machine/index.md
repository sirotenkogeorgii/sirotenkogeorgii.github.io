---
title: Probabilistic Turing Machine
layout: default
noindex: true
---

# (Alternative) Probabilistic Turing Machine

A **probabilistic Turing machine** is a non-deterministic Turing machine that chooses between the available transitions at each point according to some probability distribution (usually there are two possible next moves). As a consequence, a probabilistic Turing machine can (unlike a deterministic Turing machine) have stochastic results. Because of this randomness, running a PTM twice on the exact same input can produce different results. It might finish quickly one time and run forever the next, or it might say "Yes" in one run and "No" in another. PTM uses probability—like flipping a coin—to decide which action to take next.

In the case of equal probabilities for the transitions, probabilistic Turing machines can be defined as deterministic Turing machines having an additional "write" instruction where the value of the write is uniformly distributed in the Turing machine's alphabet (generally, an equal likelihood of writing a "1" or a "0" on to the tape). 

You can think of a PTM in two ways:

* **Random Writing:** In the case of equal probabilities for the transitions, probabilistic Turing machines can be defined as deterministic Turing machines but has a special instruction to write a random symbol (usually a 0 or 1 with equal chance) onto the tape.
* **Random Tape:** It is a deterministic Turing machine that has access to a second tape filled entirely with random bits, which it reads to make decisions.

A quantum computer (or quantum Turing machine) is another model of computation that is inherently probabilistic, because relies heavily on similar probabilistic principles.

### Formal Definition

Sure, here’s a rephrased version with the same meaning:

A probabilistic Turing machine is defined as a 7-tuple $M = (Q, \Sigma, \Gamma, q_0, A, \delta_1, \delta_2)$, where

* $Q$ is a finite set of states.
* $\Sigm$ is the input alphabet.
* $\Gamma$ is the tape alphabet and contains the blank symbol (#).
* $q_0 \in Q$ is the start state.
* $A \subseteq Q$ is the set of accepting (final) states.
* $\delta_1 : Q \times \Gamma \to Q \times \Gamma \times \{L, R\}$ is the first probabilistic transition function.
  Here, $L$ means “move the tape head one cell to the left” and $R$ means “move it one cell to the right.”
* $\delta_2 : Q \times \Gamma \to Q \times \Gamma \times \{L, R\}$ is the second probabilistic transition function.

At each computation step, the machine randomly chooses whether to apply $\delta_1$ or $\delta_2$. This choice is made independently at every step, much like flipping a fair coin each time to decide which transition function to use.

Because of this random choice, the machine can make mistakes: a string that should be accepted might sometimes be rejected, and a string that should be rejected might sometimes be accepted. To formalize this, we say that a language $L$ is recognized **with error probability** $\epsilon$ by a probabilistic Turing machine $M$ if:

1. For every string $w \in L$,
   
   $$\Pr[M \text{ accepts } w] \ge 1 - \epsilon$$.
   
2. For every string $w \notin L$,

   $$\Pr[M \text{ rejects } w] \ge 1 - \epsilon$$.

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/computability-and-complexity/PTM.png' | relative_url }}" alt="PTM" loading="lazy">
  </figure>
</div>


### Complexity Classes

*(see [notes]({{ '/notes/computability-and-complexity/' | relative_url }}))*

#### Open Questions

One of the main issues in complexity theory is whether randomness actually makes computation more powerful. In other words:

* Is there a problem that a probabilistic Turing machine can solve in polynomial time, but no deterministic Turing machine can solve in polynomial time?
* Or can every probabilistic Turing machine be efficiently simulated by a deterministic one, with only a polynomial increase in running time?

We know that $P \subseteq BPP$, because a deterministic Turing machine is simply a special case of a probabilistic one. What we *don’t* yet know is whether $BPP \subseteq P$; this is widely conjectured to be true, which would give $BPP = P$.

An analogous question can be asked for logarithmic space instead of polynomial time (does $L = BPL$?), and this equality is believed even more strongly.

At the same time, the advantages randomness provides in interactive proof systems, and the elegant randomized algorithms it yields for hard problems—such as polynomial-time primality testing and log-space algorithms for testing graph connectivity—indicate that randomness might genuinely increase computational power.


### Application

Computational complexity theory models **randomized algorithms** as **probabilistic Turing machines**. Both *Las Vegas* and *Monte Carlo* algorithms fit into this framework, and several associated complexity classes are defined.

The most basic randomized complexity class is **RP**. It consists of all decision problems for which there exists an efficient (polynomial-time) randomized algorithm (i.e., a probabilistic Turing machine) that:

* always correctly identifies NO-instances ($w \noin L$), and
* correctly identifies YES-instances ($w \in L$) with probability at least $\dfrac{1}{2}$.

The complementary class is **co-RP**, where YES-instances are always accepted correctly and NO-instances may be misclassified with probability at most $\dfrac{1}{2}$.

Problems that admit (possibly nonterminating) algorithms whose *expected* running time is polynomial and whose answers are always correct belong to the class **ZPP** (zero-error probabilistic polynomial time).

If we allow both YES- and NO-instances to be recognized with some bounded error, we obtain the class **BPP**. This class is often viewed as the randomized analogue of $P$; in other words, $BPP$ represents the class of efficiently solvable problems when randomized algorithms are allowed.
