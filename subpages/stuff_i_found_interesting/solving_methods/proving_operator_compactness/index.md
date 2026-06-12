---
layout: default
title: Proving Compactness
date: 2026-04-20
# excerpt: "Lecture notes on PDEs in data science: gradient flows, energy landscapes, existence, uniqueness, and stability, with a view toward applications in physics and machine learning."
tags:
  - compactness
  - compact-operators
  - operator-theory
  - functional-analysis
  - banach-space
  - hilbert-space
  - infinite-dimensional-space
  - weak-convergence
  - arzela-ascoli-theorem
---

To prove that a bounded linear operator $T: X \to Y$ between Banach spaces is **compact**, you must show that it maps bounded subsets of $X$ to relatively compact subsets of $Y$ (subsets with a compact closure). Depending on the specific operator, you can use the following primary standard strategies.

## 1. The Finite Rank Approximation Method

This method directly builds on your previous query. If you can show that $T$ is the **limit of a sequence of finite-rank operators**, it is compact.

* **The Theorem:** A bounded linear operator on a separable Hilbert space (or a Banach space with the approximation property) is compact if and only if there exists a sequence of finite-rank operators $\lbrace T_n\rbrace$ such that $\lim_{n \to \infty} \Vert{}T - T_n\Vert{} = 0$.
* **How to use it:** Construct a sequence of simpler, finite-dimensional operators $T_n$ (e.g., by truncating an infinite basis expansion) and prove that the operator norm of the error $\Vert{}T - T_n\Vert{}$ vanishes as $n \to \infty$.


- [BIP: Sheet02, Exercise 2.1c](/subpages/books/numerical_methods_for_bip/problems/sheet02/)

## 2. The Sequential Compactness Definition

This is the most common foundational method derived from the definition of compactness in metric spaces.

* **The Test:** An operator $T$ is compact if, for every bounded sequence $\lbrace x_n\rbrace$ in $X$, the image sequence $\lbrace Tx_n\rbrace$ contains a **convergent subsequence** in $Y$.
* **How to use it:** Start with an arbitrary sequence $\lbrace x_n\rbrace$ satisfying $\Vert{}x_n\Vert{} \leq M$. Use properties of the space $X$ or weak convergence (e.g., Banach-Alaoglu theorem) to extract a subsequence $\lbrace x_{n_k}\rbrace$ such that $T x_{n_k}$ converges strongly in $Y$.

## 3. The Arzelà-Ascoli Theorem (For Function Spaces)

When your operator maps into a space of continuous functions (like $C(K)$), you can bypass standard sequence definitions by testing for equicontinuity.

* **The Test:** The image of the unit ball $T(B_1(0))$ must be **uniformly bounded** and **equicontinuous**.
* **How to use it:**
  * 1. Prove that for all $x$ with $\Vert{}x\Vert{} \leq 1$, the function $Tx$ is bounded by a constant independent of $x$.
  * 2. Prove that for every $\varepsilon > 0$, there exists a $\delta > 0$ such that $\vert{}(Tx)(t_1) - (Tx)(t_2)\vert{} < \varepsilon$ whenever $\vert{}t_1 - t_2\vert{} < \delta$, uniformly for all $x$ in the unit ball.

## 4. Operator Composition and Ideals

Compact operators form a **two-sided closed ideal** within the algebra of bounded operators. You can prove an operator is compact if it is built from other known compact operators.

* **Composition Rule:** If $A$ is a bounded operator and $B$ is a compact operator, then both $AB$ and $BA$ are **compact**.
* **Embedding Rules:** If your operator can be factored through a compact embedding (such as a Rellich-Kondrachov Sobolev embedding $W^{k,p} \hookrightarrow L^p$), the operator is automatically compact.

## Summary Checklist for Proofs

| **If your operator is...** | **Best Approach to Try First** |
|---|---|
| An **Integral Operator** (e.g., $Kx(t) = \int k(t,s)x(s)ds$) | Show the kernel is $L^2$ (Hilbert-Schmidt) or use **Arzelà-Ascoli**. |
| Defined via an **Infinite Matrix** or **Basis** | Truncate the matrix to form $T_n$ and use **Finite Rank Approximation**. |
| Acting on **Abstract Banach Spaces** | Take a bounded sequence and find a convergent subsequence via **Weak Convergence**. |
