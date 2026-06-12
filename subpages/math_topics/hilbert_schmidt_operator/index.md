---
layout: default
title: "Hilbert–Schmidt operators"
tags:
  - operator-theory
  - compact-operators
  - compactness
  - hilbert-schmidt-operator
  - kernel-functions
---

# Hilbert–Schmidt operators

## 0. Conventions

Throughout, $H$ is a **separable** complex Hilbert space with inner product $\langle\cdot,\cdot\rangle$, taken **linear in the first variable** and conjugate-linear in the second. We fix an orthonormal basis (ONB) $\lbrace e_n\rbrace_{n\in\mathbb N}$; all sums over $n$ run over this index set. Operators are bounded linear maps $T:H\to H$ unless stated otherwise, with operator norm $\|T\|_{\mathrm{op}}=\sup_{\|x\|=1}\|Tx\|$, and adjoint $T^*$ defined by $\langle Tx,y\rangle=\langle x,T^*y\rangle$.

For $x,y\in H$ I write $u_{x,y}$ for the **rank-one operator** $u_{x,y}(z)=\langle z,y\rangle\,x$. (One often denotes this $x\otimes\bar y$; the reason for the bar will become clear.) Its adjoint is $u_{x,y}^*=u_{y,x}$.

I assume the reader is comfortable with compact operators, the spectral theorem for compact self-adjoint operators, the polar decomposition $T=U|T|$ with $|T|=(T^*T)^{1/2}$, and **singular values**: when $T$ is compact, the $\sigma_k(T)$ are the eigenvalues of $|T|$ listed with multiplicity in decreasing order, $\sigma_1\ge\sigma_2\ge\cdots\to 0$.

---

## 1. The driving idea

A general bounded operator on an infinite-dimensional space is a wild object: its matrix $(\langle Te_m,e_n\rangle)_{n,m}$ can be essentially arbitrary subject only to defining a bounded map. The **Hilbert–Schmidt** class isolates the operators that are tame in the most naive possible sense — *the ones whose matrix entries are square-summable*, exactly as an $\ell^2$ sequence is a square-summable list of numbers.

This one condition turns out to be astonishingly robust. It is independent of the basis used to write the matrix; it makes the operators themselves into a Hilbert space; it forces compactness; and — the punchline — on $L^2$ it identifies these operators with the integral kernels that are square-integrable. The slogan to carry through the whole note:

> **A Hilbert–Schmidt operator is to a bounded operator what an $L^2$ kernel is to a general distributional kernel: it is the one whose "kernel" is square-integrable.**

Everything below is an elaboration of that slogan.

---

## 2. Definition and the one fact that makes it work

**Definition.** An operator $T:H\to H$ is **Hilbert–Schmidt** (HS) if

$$
\|T\|_{\mathrm{HS}}^2 \;:=\; \sum_{n}\|Te_n\|^2 \;<\;\infty.
$$

We write $\mathcal S_2(H)$, or simply $\mathcal S_2$, for the set of such operators.

The definition mentions a basis, so the very first thing to check is that the basis was a fiction.

**Key fact (basis independence, and $\|T\|_{\mathrm{HS}}=\|T^*\|_{\mathrm{HS}}$).** *For any two orthonormal bases $\lbrace e_n\rbrace $, $\lbrace f_m\rbrace $ of $H$,*

$$
\sum_n \|Te_n\|^2 \;=\; \sum_m \|T^* f_m\|^2 .
$$

*In particular the left side is independent of $\lbrace e_n\rbrace $, the quantity $\|T\|_{\mathrm{HS}}$ is well defined, and $\|T\|_{\mathrm{HS}}=\|T^*\|_{\mathrm{HS}}$.*

**The whole proof is Parseval applied twice, with a Tonelli swap in between.** Since the summands are nonnegative we may reorder freely:
$$
\sum_n \|Te_n\|^2
=\sum_n\sum_m |\langle Te_n,f_m\rangle|^2
=\sum_m\sum_n |\langle e_n,T^*f_m\rangle|^2
=\sum_m \|T^*f_m\|^2 .
$$
The first and last equalities are Parseval (expand $Te_n$ in $\lbrace f_m\rbrace $, expand $T^*f_m$ in $\lbrace e_n\rbrace $); the middle one is $\langle Te_n,f_m\rangle=\langle e_n,T^*f_m\rangle$ together with Tonelli. The left-hand side never saw $\lbrace f_m\rbrace $ and the right-hand side never saw $\lbrace e_n\rbrace $, so both equal a common number depending on $T$ alone. Taking $f=e$ gives $\|T\|_{\mathrm{HS}}=\|T^*\|_{\mathrm{HS}}$. $\qquad\blacksquare$

This is the engine. Almost every basic property below is a one-line corollary of it.

---

## 3. Three faces of one number

The single quantity $\|T\|_{\mathrm{HS}}^2$ can be computed three different ways, and the content of the basic theory is that they all agree.

**(I) Geometric / basis face.**
$$
\|T\|_{\mathrm{HS}}^2=\sum_n \|Te_n\|^2 .
$$

**(II) Matrix / kernel face.** Writing $T_{nm}=\langle Te_m,e_n\rangle$ for the matrix of $T$,
$$
\|T\|_{\mathrm{HS}}^2=\sum_{n,m}|T_{nm}|^2 .
$$
This is just Parseval inside (I). In finite dimensions this is the **Frobenius norm** $\sqrt{\sum_{i,j}|T_{ij}|^2}=\sqrt{\operatorname{tr}(T^*T)}$; the HS norm is precisely its infinite-dimensional incarnation.

**(III) Spectral / singular-value face.** If $T$ is compact with singular values $\lbrace \sigma_k\rbrace $,
$$
\|T\|_{\mathrm{HS}}^2=\sum_k \sigma_k^2 = \operatorname{tr}(T^*T).
$$
*Proof.* $T^*T$ is positive, self-adjoint, compact; diagonalize it in an ONB $\lbrace v_k\rbrace $ with $T^*Tv_k=\sigma_k^2 v_k$. Computing the HS norm in this basis,
$$
\|T\|_{\mathrm{HS}}^2=\sum_k\|Tv_k\|^2=\sum_k\langle T^*Tv_k,v_k\rangle=\sum_k\sigma_k^2 .\qquad\blacksquare
$$

Keep all three faces in view; different proofs want different faces.

---

## 4. Elementary properties

**(a) HS dominates operator norm: $\|T\|_{\mathrm{op}}\le \|T\|_{\mathrm{HS}}$.** For a unit vector $x$, complete $x=e_1$ to an ONB; then $\|Tx\|^2\le \sum_n\|Te_n\|^2=\|T\|_{\mathrm{HS}}^2$. In singular-value language this is just $\sigma_1^2\le\sum_k\sigma_k^2$.

> **Sharpness.** Equality $\|T\|_{\mathrm{op}}=\|T\|_{\mathrm{HS}}$ holds iff at most one singular value is nonzero, i.e. iff $T$ has rank $\le 1$.

**(b) $\mathcal S_2$ is a two-sided ideal.** If $T\in\mathcal S_2$ and $A,B$ are bounded, then $ATB\in\mathcal S_2$ with
$$
\|ATB\|_{\mathrm{HS}}\le \|A\|_{\mathrm{op}}\,\|T\|_{\mathrm{HS}}\,\|B\|_{\mathrm{op}} .
$$
Left multiplication is immediate: $\sum_n\|ATe_n\|^2\le\|A\|_{\mathrm{op}}^2\sum_n\|Te_n\|^2$, so $\|AT\|_{\mathrm{HS}}\le\|A\|_{\mathrm{op}}\|T\|_{\mathrm{HS}}$. Right multiplication is the same statement read through the adjoint: $\|TB\|_{\mathrm{HS}}=\|B^*T^*\|_{\mathrm{HS}}\le\|B^*\|_{\mathrm{op}}\|T^*\|_{\mathrm{HS}}=\|B\|_{\mathrm{op}}\|T\|_{\mathrm{HS}}$, using §2. Compose the two.

**(c) Every HS operator is compact.** This is where "tame matrix" becomes a structural fact. Let $P_N$ be the orthogonal projection onto $\operatorname{span}(e_1,\dots,e_N)$. Then $TP_N$ has finite rank, and
$$
\|T-TP_N\|_{\mathrm{op}}^2\le\|T-TP_N\|_{\mathrm{HS}}^2=\sum_{n>N}\|Te_n\|^2\xrightarrow[N\to\infty]{}0 ,
$$
the tail of a convergent series. Thus $T$ is an operator-norm limit of finite-rank operators, hence compact. (The same computation shows finite-rank operators are **dense in $\mathcal S_2$ in the HS norm**, a fact we use in §5.)

So the inclusions are strict and ordered:
$$
\text{finite rank}\;\subsetneq\;\mathcal S_2\;\subsetneq\;\text{compact}\;\subsetneq\;\text{bounded}.
$$
The middle inclusion is strict: a diagonal operator $\operatorname{diag}(\sigma_k)$ with $\sigma_k=1/k$ is compact but not HS ($\sum 1/k^2<\infty$ — wait, that one *is* HS); take instead $\sigma_k = 1/\sqrt{\log(k+2)}\to 0$, compact, with $\sum\sigma_k^2=\infty$, not HS.

---

## 5. $\mathcal S_2$ is itself a Hilbert space

The square-summability of the entries is begging to be the squared norm of a Hilbert space, and indeed it is.

**The Hilbert–Schmidt inner product.** For $S,T\in\mathcal S_2$ define

$$
\langle S,T\rangle_{\mathrm{HS}} \;=\; \sum_n \langle Se_n,Te_n\rangle \;=\; \operatorname{tr}(T^*S).
$$

The series converges absolutely (Cauchy–Schwarz in each slot, then Cauchy–Schwarz on the $\ell^2$ sequences $(\|Se_n\|),(\|Te_n\|)$), it is independent of the basis by polarization of §2, and it is a genuine inner product with $\langle T,T\rangle_{\mathrm{HS}}=\|T\|_{\mathrm{HS}}^2$. The middle expression is the **trace pairing**: $\operatorname{tr}(T^*S)=\sum_n\langle T^*Se_n,e_n\rangle=\sum_n\langle Se_n,Te_n\rangle$.

**Completeness.** *$(\mathcal S_2,\langle\cdot,\cdot\rangle_{\mathrm{HS}})$ is a Hilbert space.* Let $\lbrace T_k\rbrace $ be HS-Cauchy. By property (a) it is also $\mathrm{op}$-Cauchy, hence $T_k\to T$ in operator norm for some bounded $T$. Fix $\varepsilon>0$ and $K$ with $\|T_j-T_k\|_{\mathrm{HS}}<\varepsilon$ for $j,k\ge K$. For each finite $N$ and $k\ge K$,

$$
\sum_{n\le N}\|(T-T_k)e_n\|^2=\lim_{j\to\infty}\sum_{n\le N}\|(T_j-T_k)e_n\|^2\le \varepsilon^2 ,
$$

since $T_j\to T$ pointwise and the sum is finite. Letting $N\to\infty$ gives $\|T-T_k\|_{\mathrm{HS}}\le\varepsilon$; in particular $T-T_k\in\mathcal S_2$, so $T\in\mathcal S_2$ and $T_k\to T$ in HS norm. $\qquad\blacksquare$

**A clean orthonormal basis of $\mathcal S_2$, and the tensor picture.** The rank-one operators $u_{e_n,e_m}$ are orthonormal in $\mathcal S_2$:

$$
\langle u_{x_1,y_1},u_{x_2,y_2}\rangle_{\mathrm{HS}}=\langle x_1,x_2\rangle\,\overline{\langle y_1,y_2\rangle},
\qquad\text{so}\quad \langle u_{e_n,e_m},u_{e_{n'},e_{m'}}\rangle_{\mathrm{HS}}=\delta_{nn'}\delta_{mm'} .
$$

(The conjugate on the second factor is the reason $u_{x,y}$ deserves the name $x\otimes\bar y$.) They span the finite-rank operators, which are HS-dense by (c). Hence $\lbrace u_{e_n,e_m}\rbrace _{n,m}$ is an ONB of $\mathcal S_2$, and the map sending $u_{e_n,e_m}\mapsto e_n\otimes\overline{e_m}$ extends to a **unitary identification**

$$
\boxed{\;\mathcal S_2(H)\;\cong\;H\otimes\overline H\;}
$$

This is the structural reason HS operators are the "easy" operators: as a Hilbert space they are nothing more exotic than a tensor square of $H$.

---

## 6. The motivating theorem: integral operators with $L^2$ kernels

Now we cash out the slogan from §1. This is the example that gave the class its name — it is Erhard Schmidt's 1907 setting for the theory of integral equations.

Let $(X,\mu)$ be a $\sigma$-finite measure space, $H=L^2(X,\mu)$. Given a kernel $K\in L^2(X\times X,\,\mu\otimes\mu)$, define

$$
(T_K f)(x)=\int_X K(x,y)\,f(y)\,d\mu(y).
$$
  
**Theorem (Schmidt).** *$K\mapsto T_K$ is a unitary isomorphism from $L^2(X\times X)$ onto $\mathcal S_2(L^2(X))$. In particular $T_K$ is Hilbert–Schmidt iff $K\in L^2(X\times X)$, and then*

$$
\|T_K\|_{\mathrm{HS}}=\|K\|_{L^2(X\times X)} .
$$

*Conversely, every Hilbert–Schmidt operator on $L^2(X)$ is $T_K$ for a unique such $K$.*

**Key idea: the matrix entries of $T_K$ are exactly the Fourier coefficients of $K$.** Pick an ONB $\lbrace e_n\rbrace $ of $L^2(X)$. Then the products $\lbrace (x,y)\mapsto e_n(x)\overline{e_m(y)}\rbrace _{n,m}$ form an ONB of $L^2(X\times X)$ — this is the tensor decomposition $L^2(X\times X)\cong L^2(X)\otimes\overline{L^2(X)}$. Compute one matrix entry against this basis:

$$
\langle T_K e_m,e_n\rangle
=\int_X\!\!\int_X K(x,y)\,e_m(y)\,\overline{e_n(x)}\,d\mu(y)\,d\mu(x)
=\big\langle K,\; e_n\otimes\overline{e_m}\big\rangle_{L^2(X\times X)} .
$$

So the matrix of $T_K$ in the basis $\lbrace e_n\rbrace $ *is* the array of Fourier coefficients of $K$ in the product basis. Parseval on $L^2(X\times X)$ now reads

$$
\|K\|_{L^2}^2=\sum_{n,m}\big|\langle K,e_n\otimes\overline{e_m}\rangle\big|^2=\sum_{n,m}|\langle T_K e_m,e_n\rangle|^2=\|T_K\|_{\mathrm{HS}}^2 ,
$$

using face (II) of §3. This already gives the isometry and injectivity. For surjectivity, given $T\in\mathcal S_2$ define $K=\sum_{n,m}\langle Te_m,e_n\rangle\,e_n\otimes\overline{e_m}$; the coefficients are square-summable (face II), so the series converges in $L^2(X\times X)$, and $T_K=T$ by matching matrix entries. $\qquad\blacksquare$

The theorem is really the §5 tensor identification made concrete: $\mathcal S_2(L^2(X))\cong L^2(X)\otimes\overline{L^2(X)}\cong L^2(X\times X)$, and the last isomorphism is the elementary "a function of two variables is the tensor of functions of one variable" fact. **This is why HS operators feel so usable: their kernel lives in an ordinary $L^2$ space, where you may differentiate under the integral, mollify, truncate, and estimate with the full toolkit of measure theory.**

---

## 7. Where $\mathcal S_2$ sits: the Schatten hierarchy

Faces (II)–(III) say the HS class is "$\ell^2$ of the singular values." This places it inside a one-parameter family of ideals, the **Schatten classes**: for $0<p<\infty$,
$$
\mathcal S_p=\Big\lbrace T \text{ compact}:\ \|T\|_p:=\Big(\sum_k\sigma_k(T)^p\Big)^{1/p}<\infty\Big\rbrace ,
$$
with the limiting case $\mathcal S_\infty$ = compact operators and $\|T\|_\infty=\sigma_1=\|T\|_{\mathrm{op}}$. Since $\ell^p\subset\ell^q$ for $p\le q$, the classes nest:
$$
\mathcal S_1\subset \mathcal S_2\subset \cdots\subset \mathcal S_\infty=\lbrace \text{compact}\rbrace .
$$
The two endpoints flanking $\mathcal S_2$ are the most important.

- $\mathcal S_1$ is the **trace-class** operators: $\sum_k\sigma_k<\infty$. On these $\operatorname{tr}(T)=\sum_n\langle Te_n,e_n\rangle$ converges absolutely and is basis-independent.
- $\mathcal S_\infty$ is the compact operators.

**The structural punchline.** Among all the $\mathcal S_p$, *the value $p=2$ is the only one for which $\|\cdot\|_p$ comes from an inner product.* This is the operator-theoretic echo of the elementary fact that among $\ell^p$ (or $L^p$) spaces, only $\ell^2$ is a Hilbert space. The Schatten classes are the **noncommutative $\ell^p$ spaces**, singular values playing the role of the sequence of values, and $\mathcal S_2$ is their unique Hilbert member. Everything in §5 is the shadow of this principle.

**$\mathcal S_2\cdot\mathcal S_2=\mathcal S_1$.** The trace class and HS class are linked by a perfect factorization, exactly as $L^1=L^2\cdot L^2$ via Cauchy–Schwarz.

- *Product of two HS is trace class.* If $S,T\in\mathcal S_2$ then $ST\in\mathcal S_1$ and

$$
\|ST\|_1\le \|S\|_{\mathrm{HS}}\,\|T\|_{\mathrm{HS}} .
$$

This is the Cauchy–Schwarz inequality $|\langle A,B\rangle_{\mathrm{HS}}|\le \|A\|_{\mathrm{HS}}\|B\|_{\mathrm{HS}}$ in disguise, applied to $|\operatorname{tr}(ST)|=|\langle T^*, S\rangle_{\mathrm{HS}}|$, refined to the full $\mathcal S_1$-norm via polar decomposition.
- *Every trace-class operator factors as a product of two HS operators.* Write the polar decomposition $T=U|T|$ and split $|T|^{1/2}$: 
$$
T=\big(U|T|^{1/2}\big)\big(|T|^{1/2}\big).
$$
Both factors are HS because $|T|^{1/2}$ has singular values $\sigma_k^{1/2}$, and $\sum_k(\sigma_k^{1/2})^2=\sum_k\sigma_k=\|T\|_1<\infty$.

So trace class = (HS)$^2$, and the HS class is the natural "square root level" of trace class — a remark that recurs the moment Gaussian measures enter, as we now see.

---

## 8. What it's good for

The abstract picture is pretty, but the reason Hilbert–Schmidt operators are everywhere is that the HS *norm* keeps showing up as the right finiteness condition or the right statistic. Four vignettes.

### 8.1 Gaussian measures and the Karhunen–Loève expansion

This is the application that most often forces HS operators on a probabilist or a numerical analyst. Let $C:H\to H$ be positive, self-adjoint, and **trace class**; diagonalize $Ce_k=\lambda_k e_k$ with $\lambda_k\ge0$, $\sum_k\lambda_k<\infty$. There is a unique centered Gaussian measure $\mu=N(0,C)$ on $H$ with covariance $C$, i.e. $\int_H\langle x,h\rangle\langle x,k\rangle\,d\mu(x)=\langle Ch,k\rangle$. The trace-class condition is exactly what makes the measure live on $H$:
$$
\int_H \|x\|^2\,d\mu(x)=\operatorname{tr}(C)=\sum_k\lambda_k<\infty .
$$
A sample is built by the **Karhunen–Loève expansion**
$$
X=\sum_k \sqrt{\lambda_k}\,\xi_k\,e_k,\qquad \xi_k\ \text{i.i.d.}\ N(0,1),
$$
convergent in $L^2(\Omega;H)$ and distributed as $N(0,C)$. Now look at the operator doing the work: $C^{1/2}$ has singular values $\sqrt{\lambda_k}$, and
$$
\sum_k\big(\sqrt{\lambda_k}\big)^2=\sum_k\lambda_k=\operatorname{tr}(C)<\infty,
$$
so **$C^{1/2}$ is Hilbert–Schmidt** — precisely the $\mathcal S_1=(\mathcal S_2)^2$ factorization of §7, with $C=C^{1/2}\cdot C^{1/2}$. Conceptually, $X=C^{1/2}W$ where $W$ is white noise (the canonical cylindrical Gaussian with "covariance" the identity, which does *not* live on $H$). The white-noise measure fails to exist on $H$ precisely because $I$ is not trace class; an operator must be **Hilbert–Schmidt to convert white noise into an honest $H$-valued Gaussian field**. This is the operator-theoretic heart of Gaussian random fields, the construction of Brownian motion and the Brownian bridge, stochastic PDE, and Gaussian-prior Bayesian inverse problems.

### 8.2 Statistical independence: the HSIC

Take reproducing-kernel Hilbert spaces $\mathcal F,\mathcal G$ on spaces $\mathcal X,\mathcal Y$ and a joint distribution of $(X,Y)$. The **cross-covariance operator** $C_{XY}:\mathcal G\to\mathcal F$ encodes all kernelized correlations between features of $X$ and features of $Y$. Its squared Hilbert–Schmidt norm,

$$
\mathrm{HSIC}(X,Y)=\|C_{XY}\|_{\mathrm{HS}}^2 ,
$$

is the **Hilbert–Schmidt Independence Criterion**; for characteristic kernels it vanishes iff $X$ and $Y$ are independent. Here the HS norm of an operator has become a single scalar dependence statistic with a clean estimator — the abstract norm of §2 doing concrete statistical work.

### 8.3 Quantum mechanics and quantum information

States are density operators $\rho$ (positive, trace class, $\operatorname{tr}\rho=1$, hence $\rho\in\mathcal S_1$). The HS inner product $\langle A,B\rangle_{\mathrm{HS}}=\operatorname{tr}(A^*B)$ is the natural geometry on observables and states: the **purity** of a state is $\operatorname{tr}(\rho^2)=\|\rho\|_{\mathrm{HS}}^2$ (equal to $1$ for pure states, $<1$ for mixed ones), and the **Hilbert–Schmidt distance** $\|\rho-\sigma\|_{\mathrm{HS}}$ is a frequently used, easily computed metric on the state space.

### 8.4 Integral equations and spectral theory

Back to §6: because an HS operator on $L^2$ is compact, the Fredholm alternative applies to $f-\lambda T_K f=g$, and the spectral theorem decomposes self-adjoint $T_K$ as $\sum_k\mu_k\,u_{\phi_k,\phi_k}$ with $\sum_k\mu_k^2=\|K\|_{L^2}^2<\infty$. The eigenfunction expansions underlying Sturm–Liouville theory, Mercer's theorem, and the spectral analysis of Green's functions all live in this HS world, and the $L^2$-summability of eigenvalues is just face (III).

---

## 9. Closing remarks

**What is really going on.** A Hilbert–Schmidt operator is a square-summable matrix, full stop — and *every* good property follows from refusing to let that statement depend on coordinates. Basis-independence (§2) is the single nontrivial input; the Hilbert-space structure (§5), the tensor and kernel identifications (§5–6), and the place in the Schatten scale (§7) are all corollaries. The right mental model is the dictionary

$$
\begin{array}{ccc}
\text{numbers in } \ell^2 & \longleftrightarrow & \text{operators in } \mathcal S_2\\[2pt]
\text{functions in } L^2(X) & \longleftrightarrow & \text{kernels in } L^2(X\times X)\\[2pt]
\sum_n |a_n|^2 & \longleftrightarrow & \sum_{n,m}|T_{nm}|^2=\sum_k\sigma_k^2=\operatorname{tr}(T^*T).
\end{array}
$$

**Two generalizations worth knowing.**

1. *Beyond $p=2$.* The Schatten classes $\mathcal S_p$ are complete (Banach for $p\ge1$), with $(\mathcal S_p)^*\cong\mathcal S_{p'}$ for $1\le p<\infty$, $\tfrac1p+\tfrac1{p'}=1$, the trace pairing playing the role of integration. The duality $(\mathcal S_1)^*\cong\mathcal S_\infty$ (compact)$^{**}\cong\mathcal B(H)$ mirrors $(\ell^1)^*\cong\ell^\infty$. The whole edifice is the prototype of the **noncommutative $L^p$ spaces** attached to a von Neumann algebra with a trace, where the singular-value sum is replaced by integration against the trace.

2. *Beyond a single Hilbert space.* For $T:H_1\to H_2$ the same definition and theory hold verbatim, with $\mathcal S_2(H_1,H_2)\cong H_2\otimes\overline{H_1}$. Separability is not essential for the basic facts — the defining sum is a sum of nonnegative terms over an arbitrary index set, of which at most countably many are nonzero when it is finite, so a Hilbert–Schmidt operator automatically has separable range and one loses nothing by working separably.

**A parting slogan.** Bounded operators are the measurable kernels; Hilbert–Schmidt operators are the $L^2$ kernels; trace-class operators are the $L^1$ kernels obtained by multiplying two of the former. Once you see operator theory through this $L^p$-of-singular-values lens, the special role of $p=2$ — the only Hilbert level, the level at which Gaussian measures, RKHS dependence measures, and quantum purity all naturally live — stops being a coincidence and becomes the point.