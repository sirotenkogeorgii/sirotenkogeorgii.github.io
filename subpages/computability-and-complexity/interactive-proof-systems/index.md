---
title: Interactive proof systems
layout: default
noindex: true
---

# Interactive proof systems

> “What is intuitively required from a theorem-proving procedure? First, that it is possible to “prove” a true theorem. Second, that it is impossible to “prove” a false theorem. Third, that communicating the proof should be eﬃcient, in the following sense. It does not matter how long must the prover compute during the proving process, but it is essential that the computation required from the verifier is easy.” *Goldwasser, Micali, Rackoﬀ 1985*

## Background

The standard notion of a mathematical proof matches the certificate-style definition of NP. To prove that a statement is true, one presents a sequence of symbols—something that could be written in a book or on paper—and such a valid sequence exists only for true statements.

In practice, however, people often use more flexible ways to convince others that a statement is correct: they *interact*. The person checking the proof (the **verifier**) asks questions or requests clarifications, and the person giving the proof (the **prover**) responds, possibly through several rounds, until the verifier is satisfied.

It is natural to ask how powerful these interactive proofs are from a complexity-theoretic viewpoint. For instance, is it possible to prove that a given formula is *not* satisfiable? (Recall that this problem is coNP-complete and is not believed to admit polynomial-size certificates.)
Surprisingly, the answer is yes.

Interactive proofs turned out to be far more powerful than initially expected and have many important applications. Besides their conceptual significance, they led to key developments in cryptographic protocols, our understanding of approximation algorithms, program checking, and the complexity of well-known “elusive” problems (NP problems not known to be in P nor NP-complete), such as graph isomorphism and the approximate shortest lattice vector problem.

## Definition

In computational complexity theory, an interactive proof system is a theoretical model of computation where two parties exchange messages: a *prover* and a *verifier*. Through this back-and-forth communication, they try to determine whether a particular string belongs to a certain language. The prover is assumed to have unlimited computational power but might be untrustworthy, while the verifier is computationally limited but always honest. The interaction continues—messages going back and forth—until the verifier reaches a decision and is sufficiently "convinced" that the answer is correct.

Every interactive proof system must satisfy two key properties:

* **Completeness:** If the statement being checked is true, then an honest prover (one that follows the protocol correctly) can persuade the honest verifier of its truth.
* **Soundness:** If the statement is false, then no prover—whether or not it follows the protocol—can convince the honest verifier that it is true, except with a very small probability of error.

The exact power of such a system, and thus the complexity class of the languages it can decide, depends on the constraints placed on the verifier and on what capabilities it has. In particular, many interactive proof systems crucially rely on the verifier having access to randomness for making probabilistic choices.
