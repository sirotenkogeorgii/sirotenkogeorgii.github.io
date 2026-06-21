---
title: Problems from the Numerical Methods for Bayesian Inverse Problems course. Sheet 02
layout: default
noindex: true
tags:
  - inverse-problems
  - bayesian-inference
  - numerical-methods
  - functional-analysis
  - fourier-analysis
  - compact-operator
  - exercises
---

**Table of Contents**
- TOC
{:toc}


## Exercise 2.1 — Convolution

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation and Convention</span><span class="math-callout__name">(Exercise 2.1)</span></p>

Let $e_n(x):=\exp(2\pi i n x)$ for $n\in\mathbb Z$. Then $(e_n)\_{n\in\mathbb Z}$ is an orthonormal basis of $L^2\_{\text{per}}([0,1])$, and $c_n(g)=\langle g, e_n\rangle_{L^2}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 2.1b</span><span class="math-callout__name">()</span></p>

Show that for $f\in X$ and all $n\in\mathbb{Z}$

$$c_n(T_K f) = c_n(K)c_n(f).$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 2.1c</span><span class="math-callout__name">()</span></p>

Show that $T_K$ is compact if and only if $(\lvert c_n(K) \rvert)\_{n\in\mathbb{Z}}$ is a null-sequence.

</div>

By part (2.1b), $T_K$ is diagonal in the Fourier basis: $T_K e_n = c_n(K)e_n$, with $n\in\mathbb Z$. Since $c_m(e_n)=\delta_{mn}$ then 

$$c_m(T_K e_n)=c_m(K)c_m(e_n)=c_n(K)\delta_{mn}.$$

We prove the standard diagonal compactness criterion.

**If $(\lvert c_n(K)\rvert)\_{n\in\mathbb Z}$ is a null-sequence.**

For $N\in\mathbb N$ we define 

$$T_K^{(N)}f := \sum_{\lvert n\rvert\leq N} c_n(K)c_n(f)e_n.$$

This is finite-rank and 

$$(T_K-T_K^{(N)})f = \sum_{\lvert n\rvert>N} c_n(K)c_n(f)e_n,$$

and Parseval gives

$$
\|(T_K-T_K^{(N)})f\|_{L^2}^2
=
\sum_{|n|>N}|c_n(K)|^2|c_n(f)|^2
\leq
\left(\sup_{|n|>N}|c_n(K)|^2\right)\|f\|_{L^2}^2.
$$

Hence 

$$\|T_K-T_K^{(N)}\|_{\mathcal{L}(X;X)} \leq \sup_{\lvert n\rvert>N}\lvert c_n(K)\rvert$$

goes to zero. Thus $T_K$ is the operator-norm limit of finite-rank operators, hence compact.

**Only if $T_K$ is compact.**

The sequence $(e_n)$ converges weakly to $0$ in $L^2$ as $\lvert n\rvert\to\infty$. A compact operator maps weakly convergent sequences to strongly convergent sequences. Therefore $\|T_K e_n\|_{L^2}\to 0$, but 

$$\|T_K e_n\|_{L^2} = \|c_n(K)e_n\|_{L^2} = \lvert c_n(K)\rvert,$$

thus $\lvert c_n(K)\rvert\to 0$ as $\lvert n\rvert\to\infty$.
