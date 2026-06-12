---
layout: default
title: "Hilbert–Schmidt operators"
tags:
  - operator-theory
  - compact-operators
  - compactness
  - hilbert-schmidt-integral-operator
  - hilbert-schmidt-operator
  - kernel-functions
---

# Hilbert–Schmidt integral operators: the honest infinite matrices

There is a slogan that, once internalized, makes a large slice of operator theory feel inevitable rather than miraculous:

> **A Hilbert–Schmidt integral operator is nothing but a matrix whose rows and columns are indexed by points, and whose Frobenius norm happens to be finite.**

Everything below is an elaboration of this slogan. The kernel $K(x,y)$ *is* the matrix; the variable $x$ is the row index, $y$ the column index, and the $L^2$ norm of the kernel is the Frobenius (Hilbert–Schmidt) norm. The familiar linear-algebra facts — finite Frobenius norm implies a clean singular value decomposition, Hermitian matrices are orthogonally diagonalizable, etc. — survive into infinite dimensions essentially verbatim, *provided* one keeps the Frobenius norm finite. That proviso is exactly the Hilbert–Schmidt condition. The operators for which it fails are the ones that misbehave.

## Conventions

Throughout, $(X,\mu)$ and $(Y,\nu)$ are $\sigma$-finite measure spaces, and all Hilbert spaces are separable and over $\mathbb{C}$. Inner products are **conjugate-linear in the second slot**: $\langle f,g\rangle = \int f\,\overline{g}$. Given a kernel $K\in L^2(X\times Y)$ we define the **integral operator**

$$
(T_K f)(x) \;=\; \int_Y K(x,y)\,f(y)\,d\nu(y), \qquad f\in L^2(Y).
$$

The single normalization worth remembering is the rank-one dictionary entry: if $K(x,y)=\phi(x)\,\overline{\psi(y)}$ then

$$
(T_K f)(x) = \phi(x)\int_Y \overline{\psi(y)}\,f(y)\,d\nu(y) = \langle f,\psi\rangle\,\phi(x),
$$

so $T_K$ is the rank-one operator $f\mapsto \langle f,\psi\rangle\,\phi$. This little computation is the Rosetta Stone: it tells us how to translate between kernels and operators, and the rest is bookkeeping with bases.

## 1. The fundamental isometry

**Key idea.** Pick orthonormal bases on each side. The kernel expands in the product basis, the operator expands in the same coefficients, and the two $\ell^2$ sums over those coefficients are literally identical. There is no inequality to be lost — the correspondence is an *isometry*, not merely a bound.

Let me first recall the intrinsic definition so we know what we are landing in. A bounded operator $T\colon H_1\to H_2$ is **Hilbert–Schmidt** if for some orthonormal basis $(e_j)$ of $H_1$,

$$
\|T\|_{\mathrm{HS}}^2 := \sum_j \|T e_j\|^2 < \infty.
$$

**This quantity is independent of the basis.** Indeed, picking an ONB $(f_i)$ of $H_2$ and applying Parseval twice,

$$
\sum_j \|Te_j\|^2 = \sum_{i,j}|\langle Te_j,f_i\rangle|^2 = \sum_{i,j}|\langle e_j, T^* f_i\rangle|^2 = \sum_i \|T^* f_i\|^2,
$$

and the right-hand side no longer mentions $(e_j)$; running the argument symmetrically shows independence on both sides. As a bonus we read off $\|T\|_{\mathrm{HS}}=\|T^*\|_{\mathrm{HS}}$. The Hilbert–Schmidt operators $\mathcal{B}_2(H_1,H_2)$ form a Hilbert space under $\langle S,T\rangle_{\mathrm{HS}} = \sum_j \langle Se_j, Te_j\rangle = \operatorname{tr}(T^*S)$.

Now the central theorem.

> **Theorem (kernels are Hilbert–Schmidt operators).** The map $K\mapsto T_K$ is an isometric isomorphism
> $$ L^2(X\times Y)\;\xrightarrow{\ \cong\ }\;\mathcal{B}_2\big(L^2(Y),\,L^2(X)\big), \qquad \|T_K\|_{\mathrm{HS}} = \|K\|_{L^2(X\times Y)}. $$
> In particular every $T_K$ is bounded, with $\|T_K\|_{\mathrm{op}}\le \|K\|_{L^2}$, and *every* Hilbert–Schmidt operator between these spaces arises from a unique kernel.

**Proof.** Fix ONBs $(\phi_i)$ of $L^2(X)$ and $(\psi_j)$ of $L^2(Y)$. The functions $(x,y)\mapsto \phi_i(x)\overline{\psi_j(y)}$ form an ONB of $L^2(X\times Y)$ — orthonormality is the factorization $\langle \phi_i\overline{\psi_j},\,\phi_k\overline{\psi_l}\rangle = \delta_{ik}\delta_{jl}$, and completeness is the standard fact that products of bases span the $L^2$ of a product space. Expand

$$
K = \sum_{i,j} c_{ij}\,\phi_i\,\overline{\psi_j}, \qquad c_{ij}=\langle K,\phi_i\overline{\psi_j}\rangle, \qquad \|K\|_{L^2}^2 = \sum_{i,j}|c_{ij}|^2.
$$

Apply $T_K$ to a basis vector and use the rank-one dictionary term by term:

$$
T_K\psi_k = \sum_{i,j} c_{ij}\,\langle \psi_k,\psi_j\rangle\,\phi_i = \sum_i c_{ik}\,\phi_i.
$$

Hence $\|T_K\psi_k\|^2=\sum_i |c_{ik}|^2$, and summing over $k$,

$$
\|T_K\|_{\mathrm{HS}}^2 = \sum_k \|T_K\psi_k\|^2 = \sum_{i,k}|c_{ik}|^2 = \|K\|_{L^2}^2.
$$

This is the isometry. The operator-norm bound follows because the operator norm is the supremum singular value, which never exceeds the $\ell^2$ norm of all singular values: $\|T_K\|_{\mathrm{op}}\le \|T_K\|_{\mathrm{HS}}$. Surjectivity: given any $T\in\mathcal{B}_2$, the doubly-indexed array $c_{ij}:=\langle T\psi_j,\phi_i\rangle$ is square-summable (that is the HS condition), so $K:=\sum c_{ij}\phi_i\overline{\psi_j}\in L^2(X\times Y)$ and reversing the computation gives $T_K=T$. Injectivity is the isometry. $\quad\blacksquare$

The matrix $(c_{ij})=(\langle T_K\psi_j,\phi_i\rangle)$ is, by construction, the matrix of $T_K$ in the chosen bases. So the theorem says precisely: **the kernel and the matrix are the same object, viewed in two coordinate systems**, and $L^2$-norm of one equals Frobenius-norm of the other.

**Sharpness of the operator-norm bound.** The inequality $\|T_K\|_{\mathrm{op}}\le\|K\|_{L^2}$ is an equality iff $T_K$ has at most one nonzero singular value, i.e. iff $K$ is (essentially) of rank one, $K(x,y)=\phi(x)\overline{\psi(y)}$. In all other cases the gap $\|K\|_{L^2}^2 - \|T_K\|_{\mathrm{op}}^2 = \sum_{n\ge 2}\sigma_n^2$ is strictly positive.

## 2. Compactness comes for free

**Key idea.** Truncating the kernel basis expansion gives finite-rank operators converging in operator norm; a norm-limit of finite-rank operators is compact.

Let $K_N = \sum_{i,j\le N} c_{ij}\,\phi_i\overline{\psi_j}$. Then $T_{K_N}$ has rank $\le N$, and by the isometry

$$
\|T_K - T_{K_N}\|_{\mathrm{op}} \le \|T_K-T_{K_N}\|_{\mathrm{HS}} = \|K-K_N\|_{L^2} \xrightarrow{N\to\infty} 0.
$$

Hence $T_K$ is an operator-norm limit of finite-rank operators, therefore **compact**. The Hilbert–Schmidt operators sit strictly between finite-rank and compact:

$$
\text{finite rank}\;\subsetneq\; \text{Hilbert–Schmidt }(\mathcal{S}_2)\;\subsetneq\; \text{compact}.
$$

The second inclusion is strict: an operator with singular values $\sigma_n = n^{-1/2}$ is compact (since $\sigma_n\to0$) but not Hilbert–Schmidt (since $\sum n^{-1}=\infty$). The HS condition is genuinely the requirement that the singular values lie in $\ell^2$, which is stronger than merely tending to $0$.

## 3. Adjoints, and when the kernel is symmetric

A direct application of Fubini to $\langle T_K f,g\rangle$ identifies the adjoint as another integral operator:

$$
T_K^* = T_{K^*}, \qquad K^*(x,y) = \overline{K(y,x)}.
$$

Thus $T_K$ is **self-adjoint** precisely when the kernel is *Hermitian*: $K(x,y)=\overline{K(y,x)}$ a.e. (with $X=Y$). And $T_K$ is **positive semidefinite**, $\langle T_K f,f\rangle\ge 0$, precisely when $K$ is a *positive-definite kernel* in the sense $\sum_{m,n}\overline{a_m}a_n K(x_m,x_n)\ge 0$ — the same condition that defines reproducing kernels in machine learning. Keep that coincidence in mind for §6.

## 4. The Hilbert–Schmidt spectral theorem

**Key idea.** A self-adjoint compact operator is, up to choice of orthonormal eigenbasis, a diagonal matrix with a decaying diagonal. Re-reading the diagonalization through the kernel dictionary gives an $L^2$-convergent series expansion of $K$ itself.

> **Theorem.** Let $K\in L^2(X\times X)$ be Hermitian. Then there is an orthonormal system $(e_n)$ in $L^2(X)$ and real eigenvalues $\lambda_n\to 0$ with
> $$ T_K f = \sum_n \lambda_n \langle f,e_n\rangle\, e_n, \qquad K(x,y) = \sum_n \lambda_n\, e_n(x)\,\overline{e_n(y)} \ \text{ in } L^2(X\times X), $$
> and the eigenvalues are **square-summable**: $\sum_n \lambda_n^2 = \|K\|_{L^2}^2 < \infty$.

The first two displays are just the abstract spectral theorem for self-adjoint compact operators, transported through the rank-one dictionary ($\lambda_n\,e_n\overline{e_n}$ is the kernel of the rank-one projection scaled by $\lambda_n$). The square-summability is the isometry of §1 applied to the diagonal form: in the eigenbasis the "matrix" is $\operatorname{diag}(\lambda_n)$, whose Frobenius norm is $(\sum\lambda_n^2)^{1/2}$. This is the cleanest possible illustration of the slogan — a self-adjoint Hilbert–Schmidt operator is an *infinite Hermitian matrix with an $\ell^2$ diagonal*.

**Without symmetry, use the SVD.** A general $K\in L^2(X\times Y)$ need not have any eigenvalues at all (see Volterra below). The correct universal statement is the **singular value decomposition**: there exist orthonormal systems $(u_n)\subset L^2(X)$, $(v_n)\subset L^2(Y)$ and singular values $\sigma_n\ge 0$ with

$$
K(x,y)=\sum_n \sigma_n\, u_n(x)\,\overline{v_n(y)}, \qquad \sum_n \sigma_n^2 = \|K\|_{L^2}^2.
$$

This is obtained by applying the symmetric theorem to the Hermitian operator $T_K^* T_K = T_{K^*\!*K}$ and taking $\sigma_n = \sqrt{\lambda_n(T_K^*T_K)}$. The Hilbert–Schmidt condition is exactly $\sigma\in\ell^2$; trace-class ($\mathcal{S}_1$) is $\sigma\in\ell^1$; bounded compact is $\sigma\in c_0$. These are the Schatten classes, the operator-theoretic avatars of the sequence spaces $\ell^p$.

## 5. Mercer's theorem: upgrading the convergence

The expansion $K=\sum_n\lambda_n e_n\overline{e_n}$ converges in $L^2$, which for many purposes is unsatisfying — it says nothing pointwise. Under stronger hypotheses the convergence upgrades dramatically.

> **Theorem (Mercer).** Let $X$ be a compact metric space with a finite Borel measure of full support, and let $K\in C(X\times X)$ be a *continuous, Hermitian, positive-semidefinite* kernel. Then the eigenfunctions $e_n$ may be taken continuous, and
> $$ K(x,y) = \sum_n \lambda_n\, e_n(x)\,\overline{e_n(y)} $$
> converges **absolutely and uniformly** on $X\times X$. Consequently $\sum_n \lambda_n = \int_X K(x,x)\,d\mu(x) = \operatorname{tr}(T_K) < \infty$, so $T_K$ is **trace-class**, not merely Hilbert–Schmidt.

The two takeaways: (i) positivity plus continuity buys you uniform convergence and a genuinely pointwise identity; (ii) the trace is computed by integrating the kernel along the diagonal — the continuous analogue of "trace = sum of diagonal entries." The leap from $\sum\lambda_n^2<\infty$ (HS, generic) to $\sum\lambda_n<\infty$ (trace-class, Mercer) is where positivity earns its keep, and it is exactly the regularity that makes covariance kernels and reproducing kernels so well-behaved.

## 6. Two worked examples that are secretly the same

### The Volterra operator: Hilbert–Schmidt, yet spectrally invisible

On $L^2(0,1)$ consider $(Vf)(x)=\int_0^x f(t)\,dt$, with kernel $K(x,t)=\mathbf 1_{\{t\le x\}}$. Then

$$
\|V\|_{\mathrm{HS}}^2 = \iint_{[0,1]^2}\mathbf 1_{\{t\le x\}}\,dt\,dx = \int_0^1 x\,dx = \tfrac12,
$$

so $V$ is Hilbert–Schmidt with $\|V\|_{\mathrm{HS}}=1/\sqrt2$. Yet $V$ has **no eigenvalues whatsoever**: its spectrum is $\{0\}$, and it is quasinilpotent ($\|V^n\|^{1/n}\to 0$). The kernel is not Hermitian, so §4's eigenexpansion simply does not apply — a vivid reminder that compactness alone gives you *singular* values, not eigenvalues.

To see the singular values, form $V^*V$. Since $(V^*g)(x)=\int_x^1 g(t)\,dt$, a short computation gives the kernel of $V^*V$ as

$$
(V^*V)(x,s) = \min(1-x,\,1-s).
$$

### Brownian motion's covariance: the eigenproblem solved

Now take the Hermitian kernel $K(s,t)=\min(s,t)$ on $[0,1]^2$ — the covariance function of standard Brownian motion, $\operatorname{Cov}(W_s,W_t)=\min(s,t)$. The eigenvalue equation $\int_0^1 \min(s,t)e(t)\,dt = \lambda e(s)$ reads

$$
\int_0^s t\,e(t)\,dt + s\int_s^1 e(t)\,dt = \lambda\, e(s).
$$

**Differentiate once** (the boundary terms cancel): $\int_s^1 e(t)\,dt = \lambda\, e'(s)$. **Differentiate again:** $-e(s)=\lambda\,e''(s)$. So $e$ solves the harmonic oscillator $e''+\lambda^{-1}e=0$, with boundary conditions read off from the integral equation at the endpoints: $e(0)=0$ (set $s=0$ in the original) and $e'(1)=0$ (set $s=1$ in the once-differentiated form). The eigenpairs are therefore

$$
\boxed{\;\lambda_n = \frac{1}{\big(n-\tfrac12\big)^2\pi^2}, \qquad e_n(s)=\sqrt2\,\sin\!\big((n-\tfrac12)\pi s\big), \qquad n=1,2,\dots\;}
$$

**Trace check (Mercer in action).** $\sum_n \lambda_n = \frac{1}{\pi^2}\sum_{n\ge1}\frac{1}{(n-1/2)^2} = \frac{1}{\pi^2}\cdot\frac{\pi^2}{2} = \frac12$, which matches $\int_0^1 K(s,s)\,ds=\int_0^1 s\,ds=\frac12$. The kernel is continuous and positive-definite, so Mercer guarantees this trace identity, and it holds on the nose.

**The punchline linking the two examples.** The Brownian eigenvalues are $\sigma_n^2$ for $\sigma_n = 1/((n-\tfrac12)\pi)$ — and these are *exactly the singular values of the Volterra operator*, because $\min(1-x,1-s)$ is just $\min(s,t)$ time-reversed and shares its spectrum. The Volterra problem and the Brownian covariance problem are one computation in two costumes. One checks $\sum_n\sigma_n^2 = \frac{1}{\pi^2}\cdot\frac{\pi^2}{2}=\frac12=\|V\|_{\mathrm{HS}}^2$, closing the loop with §1.

**The payoff: the Karhunen–Loève expansion.** Feeding the eigendata into the spectral expansion of the covariance gives the canonical series representation of Brownian motion,

$$
W_t = \sum_{n=1}^\infty \sqrt{\lambda_n}\,Z_n\,e_n(t) = \sqrt2\sum_{n=1}^\infty Z_n\,\frac{\sin\!\big((n-\tfrac12)\pi t\big)}{(n-\tfrac12)\pi}, \qquad Z_n\overset{\text{iid}}{\sim}\mathcal N(0,1),
$$

converging in $L^2(\Omega)$ uniformly in $t$. This is not a coincidence of Brownian motion: **the Karhunen–Loève expansion of any $L^2$ stochastic process is precisely the Mercer expansion of its covariance operator**, with the eigenvalues controlling how fast the random field can be truncated.

## 7. What is really going on, and where it leads

A few closing orientation remarks.

**The whole subject is finite linear algebra with a convergence hypothesis bolted on.** Frobenius norm $\rightsquigarrow$ $L^2$ kernel norm; spectral theorem for Hermitian matrices $\rightsquigarrow$ §4; SVD $\rightsquigarrow$ the singular expansion; trace $=$ sum of diagonal $\rightsquigarrow$ Mercer's diagonal integral. The single new phenomenon in infinite dimensions is that the spectrum must accumulate at $0$ — compactness — and the Hilbert–Schmidt condition pins down *how fast* ($\ell^2$).

**Covariance operators and inverse problems.** A Gaussian measure on a function space is determined by its mean and a covariance operator, which is the trace-class integral operator with kernel the covariance function; the Cameron–Martin space is its range with the inverse-square-root inner product. The eigendecay $\lambda_n$ is exactly the spectral information that governs almost-sure regularity of samples and the well-posedness of Bayesian inverse problems — the smoother the kernel, the faster the decay, the more effective the truncation. Mercer is the theorem that licenses representing the prior by finitely many coordinates.

**Green's functions.** Inverting an elliptic operator such as the Dirichlet Laplacian $-\Delta$ on a bounded domain produces a self-adjoint integral operator whose kernel is the Green's function; its eigenfunctions are the Laplacian eigenmodes and its eigenvalues the reciprocals $\lambda_n = 1/\mu_n$. The Hilbert–Schmidt condition $\sum \mu_n^{-2}<\infty$ holds precisely when the Weyl asymptotics $\mu_n \sim c\,n^{2/d}$ give a convergent sum, i.e. in dimension $d\le 3$ — a clean instance of geometry dictating membership in a Schatten class.

**Reproducing kernels.** A continuous positive-definite kernel is simultaneously a Mercer kernel and the reproducing kernel of an RKHS $\mathcal H_K$, with $\mathcal H_K=\{\sum a_n\sqrt{\lambda_n}\,e_n : \sum|a_n|^2<\infty\}$. This is the foundation of kernel methods and Gaussian-process regression: the same eigenexpansion that diagonalizes the integral operator also gives the feature map. The objects of this note are, from a slightly rotated angle, the central objects of kernel-based machine learning.

---

*The exercise worth doing after reading: take any covariance kernel you care about, solve its integral eigenproblem by reducing to an ODE as in §6, and verify the trace against the diagonal integral. If the trace check fails, the arithmetic is wrong — Mercer does not negotiate.*