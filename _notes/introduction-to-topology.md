---
layout: default
title: Introduction To Topology
date: 2024-11-01
# excerpt: Connecting differential equations, stability analysis, and attractor theory to the training dynamics of modern machine learning models.
# tags:
#   - dynamical-systems
#   - machine-learning
#   - theory
---

# Introduction To Topology

## 1. Theory of Sets

### 1.1 Introduction

There are two popular definitions of natural numbers

#### 1.1.1 Peano axioms

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

#### 1.1.2 von Neumann naturals (Set-theoretic definition/construction)

This is a way to **build** the naturals inside set theory (ZF).

Define:
* $0 := \varnothing$
* $S(n) := n \cup {n}$
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

### 1.2 DeMorgan's Laws

$\textbf{Theorem (DeMorgan's Laws).}$ Let $A\subset S$, $B\subset S$. Then
$$C(A \cup B) = C(A) \cap C(B)$$
$$C(A \cap B) = C(A) \cup C(B)$$

$\textbf{Theorem (DeMorgan's Laws for indexed families).}$ Let $\lbrace A_{\alpha}\rbrace_{\alpha \in I}$ be an indexed family of subsets of a set $S$. Then
$$C(\bigcup_{\alpha\in I} A_{\alpha}) = \bigcap_{\alpha\in I} C(A_{\alpha})$$
$$C(\bigcap_{\alpha\in I} A_{\alpha}) = \bigcup_{\alpha\in I} C(A_{\alpha})$$