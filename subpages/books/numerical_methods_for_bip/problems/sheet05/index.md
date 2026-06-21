---
title: Problems from the Numerical Methods for Bayesian Inverse Problems course. Sheet 05
layout: default
noindex: true
tags:
  - inverse-problems
  - bayesian-inference
#   - numerical-methods
#   - functional-analysis
#   - karhunen-loeve-expansion
#   - hadamard-product
#   - maclaurin-series
#   - schur-product-theorem
#   - gaussian-random-field
#   - random-field
#   - covariance-function
#   - kernel-function]
#   - self-adjoint
#   - trace-class-operator
#   - eigenfunction
#   - mercer-theorem
#   - riemann-sum
  # - galerkin-method
  - change-of-variables
  - jeffreys-prior
  - diffeomorphism
  - multilevel-monte-carlo
  - monte-carlo
  - bias-variance-decomposition
---

**Table of Contents**
- TOC
{:toc}

## Exercise 5.1 — Jeffreys Prior

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 5.1</span><span class="math-callout__name"></span></p>

Suppose $X$ is a $\mathbb{R}^n$-valued RV and $g : \mathbb{R}^n \to \mathbb{R}^n$ is a diffeomorphism with nonnegative Jacobian determinant $\det Dg : \mathbb{R}^n \to (0,\infty)$. Then $Z := g(X)$ is a RV representing another parametrization of $X$. To obtain a prior for the reparametrized RV $Z$, we could proceed in two ways:

* **(i)** Given the prior $\pi_X(x) \propto \sqrt{\det(I_X(x))}$, the density of $Z$ is obtained through a change of variables, i.e., $\pi_Z(z) = \pi_X(g^{−1}(z)) \det(Dg^{−1}(z))$, or
* **(ii)** We may directly choose the Jeffreys prior $\pi_Z(z) = \sqrt{\det(I_Z(z))}$ obtained with the reparametrized likelihood $\pi_{Y\mid Z}(y\mid z) = \pi_{Y\mid X}(y\mid g^{−1}(z))$.

Show that both constructions lead to the same prior, so that the Jeffreys prior is invariant under reparametrizations.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Let $z=g(x)$, and write $h=g^{-1}$. The reparametrized likelihood is

$$\pi_{Y|Z}(y|z)=\pi_{Y|X}(y|h(z)).$$

The key point is that the Fisher information transforms like a quadratic form.

For the score, (**!!!KEY STEP!!!**)

$$\nabla_z \log \pi_{Y|Z}(y|z) = Dh(z)^\top \nabla_x \log \pi_{Y|X}(y|x), \qquad x=h(z),$$

where we used $\pi_{Y\mid Z}(y\mid z) = \pi_{Y\mid X}(y\mid h(z))$, since $Z:=g(X)$ then

$$\nabla_z \log \pi_{Y|Z}(y|z) = \nabla_z \log \pi_{Y|X}(y|h(z)) = Dh(z)^\top \nabla_x \log \pi_{Y|X}(y|x).$$

Therefore

$$I_Z(z) = \mathbb E\left[ \nabla_z \log \pi_{Y|Z}(Y|z) \nabla_z \log \pi_{Y|Z}(Y|z)^\top \right] = Dh(z)^\top I_X(h(z))Dh(z).$$

Taking determinants gives

$$\det I_Z(z) = \det(Dh(z))^2 \det I_X(h(z)).$$

Since $\det Dg>0$, also $\det Dh>0$. Hence

$$\sqrt{\det I_Z(z)} = \det(Dh(z))\sqrt{\det I_X(h(z))}.$$

Thus the Jeffreys prior in the $z$-coordinates is

$$\pi_Z(z) \propto \sqrt{\det I_Z(z)} = \sqrt{\det I_X(g^{-1}(z))}\det(Dg^{-1}(z)).$$

But this is exactly the density obtained by changing variables from the Jeffreys prior in the $x$-coordinates:

$$\pi_Z(z) = \pi_X(g^{-1}(z))\det(Dg^{-1}(z)).$$

So Jeffreys prior is invariant under smooth reparametrizations.

</details>
</div>

## Exercise 5.4 — Jeffreys Prior

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Intro</span><span class="math-callout__name"></span></p>

Let $D\subset\mathbb{R}^d$, $d= 1,2,3$, be bounded and convex. Consider the elliptic boundary value problem

where $f \in L^2(D)$ and $y \in Y \subseteq \mathbb{R}$ is a (random) parameter. Let $Y=[-1,1]$ (for example $y$ is uniformly distributed on $[-1,1]$) and, for $\varphi \in C^1(\bar D)$, let the parametrised coefficient be given by

$$\kappa(x,y)=\exp(y\varphi(x)).$$

For $h>0$ and $k=1,\ldots,N$, we denote with $u_h^{(k)} \in V_h$ finite element (FE) solutions for the coefficient $\kappa(\cdot,y_k)$, where we assume that $y_k:\Omega \to Y$ are independently drawn from some underlying probability space $(\Omega,\Sigma,\mathbb{P})$. Consider

$$\hat u_{h,N} = \frac{1}{N}\sum_{k=1}^N u_h^{(k)}, \qquad \hat S_{h,N}^2 = \frac{1}{N-1}\sum_{k=1}^N \left(u_h^{(k)}-\hat u_{h,N}\right)^2,$$

which are the Monte Carlo (MC) estimators for the mean and variance of the FE solution $u_h$ to (5.4.1), respectively.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 5.4a</span><span class="math-callout__name"></span></p>

Show that $\hat S_{h,N}^2$ is an unbiased estimator for the variance of the FE solution

$$\operatorname{Var}[u]_h=\operatorname{Var}[u_h].$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>
</details>
</div>

<!-- We interpret variance pointwise in $x$, i.e.

$$\operatorname{Var}[u_h](x) = \mathbb E\left[(u_h(x)-\mathbb E[u_h](x))^2\right].$$

### 5.4a

Fix $x\in D$, and write

$$U_k := u_h^{(k)}(x), \qquad \overline U_N := \frac1N\sum_{k=1}^N U_k.$$

Then

$$\widehat S_{h,N}^2(x) = \frac1{N-1}\sum_{k=1}^N (U_k-\overline U_N)^2.$$

Using

$$\sum_{k=1}^N (U_k-\overline U_N)^2 = \sum_{k=1}^N U_k^2 - N\overline U_N^2,$$

we get

$$\mathbb E[\widehat S_{h,N}^2(x)] = \frac1{N-1} \left( N\mathbb E[U_1^2] - N\mathbb E[\overline U_N^2] \right).$$

Since the $U_k$ are i.i.d.,

$$\mathbb E[\overline U_N^2] = \operatorname{Var}(\overline U_N)+(\mathbb E U_1)^2 = \frac1N \operatorname{Var}(U_1)+(\mathbb E U_1)^2.$$

Therefore

$$\mathbb E[\widehat S_{h,N}^2(x)] = \operatorname{Var}(U_1) = \operatorname{Var}[u_h](x).$$

Since this holds for every $x$,

$$\mathbb E[\widehat S_{h,N}^2] = \operatorname{Var}[u_h].$$

So $\widehat S_{h,N}^2$ is an unbiased estimator of the FE variance.

--- -->













































































<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 5.4b</span><span class="math-callout__name"></span></p>

The sample variance is **not** an unbiased estimate for the variance $\operatorname{Var}[u]$ of the true solution $u \in H_0^1(D)$ to the BVP (5.4.1). Assuming that

$$u,u_h,\nabla u,\nabla u_h \in L^4(\Omega;L^\infty(D)),$$

show that the bias satisfies

$$\left\| \mathbb{E}\left[ \operatorname{Var}[u]-\hat S_{h,N}^2 \right] \right\|_{H_0^1(D)} \le C h^\alpha$$

for some $\alpha \in \mathbb{R}$. You may assume that $\kappa$ and $f$ satisfy sufficient conditions such that the estimate

$$\|u-u_h\|_{L^p(\Omega;H_0^1(D))} \le c h \|f\|_{L^2(D)} \tag{5.4.2}$$

holds for some constant $c=c(p)>0$ and all $1 \le p < \infty$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>
</details>
</div>


## Exercise 5.5 — Multilevel Monte Carlo Bias and Variance

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Intro</span><span class="math-callout__name"></span></p>

Let $Q$ be a random variable and let $\lbrace Q_\ell\rbrace_{\ell\in\mathbb{N}}$ be a sequence of random variables. Consider the multilevel Monte Carlo (MLMC) estimator, which is defined as

$$\hat{Q}_L^{ML} = \sum_{\ell=1}^L \hat{Y}_\ell,$$

where $\lbrace \hat{Y}_\ell\rbrace_{\ell\in\mathbb{N}}$ is a sequence of independent Monte Carlo estimators given by

$$\hat{Y}_0 := \frac{1}{N_0} \sum_{k=0}^{N_0} Q_0^{(k)}, \quad\text{and}\quad \hat{Y}_\ell := \frac{1}{N_\ell} \sum_{k=0}^{N_\ell} (Q_{\ell}^{(k)} - Q_{\ell - 1}^{(k)})$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 5.5a</span><span class="math-callout__name"></span></p>

Show that $\hat{Q}^{ML}_L$ is an unbiased estimate of $\mathbb{E}[Q_L]$. You may use the fact that $\lbrace Q^{(k)}_\ell \rbrace^{N_\ell}_{k=1}$ are i.i.d. samples of $Q_\ell$ for all $\ell= 0,1,2,\dots,L$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

We compute the expectation:

$$\mathbb E[\widehat Y_0] = \mathbb E[Q_0],$$

and for $\ell\ge 1$,

$$\mathbb E[\widehat Y_\ell] = \mathbb E[Q_\ell-Q_{\ell-1}] = \mathbb E[Q_\ell]-\mathbb E[Q_{\ell-1}].$$

Therefore

$$
\begin{aligned}
\mathbb E[\widehat Q_L^{ML}]
&=
\mathbb E[Q_0]
+
\sum_{\ell=1}^L
\left(
\mathbb E[Q_\ell]-\mathbb E[Q_{\ell-1}]
\right) \
&=
\mathbb E[Q_L].
\end{aligned}
$$

So $\widehat Q_L^{ML}$ is an unbiased estimator of $\mathbb E[Q_L]$.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 5.5b</span><span class="math-callout__name"></span></p>

Show that the variance of the MLMC estimator is given by $\text{Var}[\hat{Q}^{ML}_L] = \sum_{\ell=0}^L \text{Var}[\hat{Y}_\ell]$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Since the estimators $\widehat Y_0,\dots,\widehat Y_L$ are independent,

$$\operatorname{Var}[\widehat Q_L^{ML}] = \operatorname{Var}\left[ \sum_{\ell=0}^L \widehat Y_\ell \right] = \sum_{\ell=0}^L \operatorname{Var}[\widehat Y_\ell].$$

Moreover,

$$\operatorname{Var}[\widehat Y_0] = \frac{\operatorname{Var}[Q_0]}{N_0},$$

and for $\ell\ge 1$,

$$\operatorname{Var}[\widehat Y_\ell] = \frac{\operatorname{Var}[Q_\ell-Q_{\ell-1}]}{N_\ell}.$$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 5.5c</span><span class="math-callout__name"></span></p>

Prove that the mean-squared error of the MLMC estimator can be decomposed as

$$\mathbb{E}\Bigl[ \Bigl| \hat{Q}_L^{ML} - \mathbb{E}[Q] \Bigr| \Bigr] = (\mathbb{E}[Q_L - Q])^2 + \frac{\text{Var}[Q_0]}{N_0} + \sum_{\ell=1}^L \frac{\text{Var}[Q_\ell - Q_{\ell - 1}]}{N_\ell}$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

We decompose the error into bias plus centered fluctuation:

$$\widehat Q_L^{ML}-\mathbb E[Q] = \left(\widehat Q_L^{ML}-\mathbb E[Q_L]\right) + \left(\mathbb E[Q_L]-\mathbb E[Q]\right).$$

The first term has expectation zero, so the cross term vanishes after taking expectation. Hence

$$\mathbb E\left[ \left|\widehat Q_L^{ML}-\mathbb E[Q]\right|^2 \right] = \operatorname{Var}[\widehat Q_L^{ML}] + \left(\mathbb E[Q_L-Q]\right)^2.$$

Using part 5.5b,

$$\operatorname{Var}[\widehat Q_L^{ML}] = \frac{\operatorname{Var}[Q_0]}{N_0} + \sum_{\ell=1}^L \frac{\operatorname{Var}[Q_\ell-Q_{\ell-1}]}{N_\ell}.$$

Therefore

$$\boxed{\mathbb E\left[ \left|\widehat Q_L^{ML}-\mathbb E[Q]\right|^2 \right] = \left(\mathbb E[Q_L-Q]\right)^2 + \frac{\operatorname{Var}[Q_0]}{N_0} + \sum_{\ell=1}^L \frac{\operatorname{Var}[Q_\ell-Q_{\ell-1}]}{N_\ell}}.$$

The first term is the discretization bias, while the remaining terms are the Monte Carlo variances on the different levels.

</details>
</div>