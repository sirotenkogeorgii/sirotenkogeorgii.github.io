---
layout: default
title: "One idea, many faces: a tour of decompositions in functional analysis"
date: 2026-06-14
mathjax: true
tags: 
  - functional-analysis, 
  - operator-theory
  - spectral-theory, 
  - harmonic-analysis
---

> **Conventions.** Throughout, $H$ (and $H_1, H_2$) denote separable complex Hilbert
> spaces with inner product $\langle\cdot,\cdot\rangle$ conjugate-linear in the *first*
> slot. "Operator" means bounded linear operator unless stated otherwise; $\mathcal B(H)$
> is the algebra of these, $\mathcal K(H)$ the compact ones. An *orthonormal system* (ONS)
> is a family $\lbrace e_n\rbrace$ with $\langle e_m,e_n\rangle=\delta_{mn}$; it is an *orthonormal
> basis* (ONB) if its span is dense. $(X,\mathcal M,\mu)$ is a $\sigma$-finite measure
> space. I write $M_g$ for the operator "multiply by $g$" on some $L^2(\mu)$. I will be
> terse on routine verifications and spend the words on *why* things fit together.

When you first meet functional analysis you are handed a confusing bouquet of theorems
that all seem to be saying *write a complicated thing as a sum of simple things*: Fourier
series, the spectral theorem, finite-rank approximation, the approximation of measurable
functions by simple functions. You asked the right two questions: **are these the same
idea wearing different hats, and if so why are there so many hats?**

The short answer is yes, and the long answer is the subject of this note. I will argue
that almost every decomposition you will meet is an instance of a single move — *choose
atoms adapted to a structure you want to make transparent, then reconstruct by
superposition* — and that the apparent zoo is really a small number of templates applied
in different categories (measure, topology, geometry, representation theory). The
organizing sun around which the others orbit is the **spectral theorem in its
multiplication-operator form**, and the secret that ties even the measure-theoretic
construction to it is that **the spectral theorem is just Lebesgue integration with
operator-valued atoms.**

<figure>
  <img src="/assets/images/notes/math_topics/operator_decompositions/decompositions-map.svg" alt="A map of decompositions in functional analysis: the spectral theorem at the root, with branches to Fourier analysis, the SVD family, the non-diagonalizable fallbacks, and the Karhunen–Loève confluence."
       style="width:100%;max-width:760px;height:auto;display:block;margin:1.5em auto;">
  <figcaption style="text-align:center;font-size:0.9em;color:#666;">
    <strong>Figure 1.</strong> The map this note draws. One theorem at the root; Fourier
    analysis specializes it to a symmetry; the SVD family feeds it $A^\ast A$; the
    fallbacks on the right are what survives when diagonalization fails; and
    Karhunen–Loève is where several branches meet on a single covariance operator. The
    teal twin on the right is the same integration machinery with scalars in place of
    projections (§5).
  </figcaption>
</figure>

---

## 1. What a decomposition *is*

Strip away the specifics and every decomposition answers three questions.

**(i) What are the atoms?** A distinguished family of "simple" objects: indicator
functions, characters $e^{i\xi x}$, rank-one operators, orthogonal projections,
irreducible subspaces. "Simple" is never absolute — it is *simple relative to a structure
we want to exploit*.

**(ii) In what sense do the atoms reconstruct?** Exactly (a convergent series in some
topology), or only approximately (the atoms are merely *dense*, and we take limits). This
is the difference between *being a basis* and *generating a dense set*.

**(iii) What does the decomposition trivialize?** A good decomposition is never neutral:
it is chosen so that some operation — translation, a fixed self-adjoint operator,
integration, a group action — becomes diagonal, i.e. acts atom-by-atom.

The whole art is that **(iii) dictates (i)**. You do not pick atoms and hope; you decide
what you want to be diagonal, and the atoms are forced on you as its eigen-objects.

<figure>
  <img src="/assets/images/notes/math_topics/operator_decompositions/reconstruction-two-atom-systems.png" alt="A target function reconstructed by Fourier partial sums on the left and by indicator block averages on the right; both converge to the same curve."
       style="width:100%;max-width:860px;height:auto;display:block;margin:1.5em auto;">
  <figcaption style="text-align:center;font-size:0.9em;color:#666;">
    <strong>Figure 2.</strong> The same target function reconstructed by superposition in
    two atom systems. <strong>Left:</strong> partial sums of the global atoms $\sin(n\pi x)$
    — smooth and spread out. <strong>Right:</strong> block averages on the local indicators
    $\mathbf{1}_{A_k}$ — piecewise-constant. Both converge to the same object; the structure
    you privilege (smoothness vs. locality) is what forces the choice of atoms — point (iii)
    dictating point (i).
  </figcaption>
</figure>

---

## 2. The two great templates

Everything below is one of two moves, or a hybrid.

**Template A — diagonalize an operator (or a commuting family).**
Given an operator $A$ you care about, look for atoms that $A$ scales:
$Ae_n=\lambda_n e_n$. Then in the $\lbrace e_n\rbrace$ coordinates $A$ is just the diagonal matrix
$\mathrm{diag}(\lambda_n)$, and any vector decomposes as $x=\sum_n\langle e_n,x\rangle e_n$
with $A$ acting termwise. Spectral theorem, Fourier analysis, eigenfunction expansions,
SVD: all of this.

**Template B — build by atoms of a measure/order structure.**
Approximate a function by combinations of indicators, $f=\lim\sum_i a_i\mathbf 1_{A_i}$.
Here the atoms are not eigenvectors of any operator; they are the building blocks of
*measurability and integration* itself. Simple-function approximation, the
Radon–Nikodym/Lebesgue decomposition of measures, the Jordan decomposition of signed
measures.

These look unrelated — one is about operators, the other about the construction of $L^p$ —
until you notice (Section 5) that Template A is *literally Template B run with operators
in place of scalars*. That observation is the spine of this note.

<figure>
  <img src="/assets/images/notes/math_topics/operator_decompositions/two-templates.png" alt="Left: a unit circle mapped to an ellipse along the eigen-axes of a symmetric matrix. Right: a simple-function staircase approximating a curve."
       style="width:100%;max-width:860px;height:auto;display:block;margin:1.5em auto;">
  <figcaption style="text-align:center;font-size:0.9em;color:#666;">
    <strong>Figure 3.</strong> The two templates side by side. <strong>(A) Diagonalize:</strong>
    the unit circle maps to an ellipse whose axes are the eigenvectors of a self-adjoint $A$,
    scaled by the eigenvalues $\lambda_n$ — in eigen-coordinates $A$ is just
    $\mathrm{diag}(\lambda_n)$. <strong>(B) Build from measure atoms:</strong> a simple function
    $\sum_k a_k\mathbf{1}_{A_k}$ climbing toward a target. The atoms in (A) are eigenvectors,
    in (B) indicators; §5 shows these are the same construction.
  </figcaption>
</figure>

---

## 3. A field guide

Let me lay out the specimens before dissecting them. Read this as a map, not a list to
memorize.

**Decompositions of vectors / functions.**

- *Orthonormal basis expansion* $x=\sum_n\langle e_n,x\rangle e_n$. The primitive move in a
  Hilbert space. **Parseval** $\|x\|^2=\sum_n\lvert\langle e_n,x\rangle\rvert^2$ holds with *equality*
  iff $\lbrace e_n\rbrace$ is complete; **Bessel's inequality** $\le$ is what you get from a mere ONS.
- *Fourier series / transform / Plancherel.* Atoms are the characters $e^{in\theta}$ on the
  circle, $e^{i\xi x}$ on the line. Diagonalizes translation and convolution.
- *Simple-function approximation.* Atoms are indicators $\mathbf 1_A$; the workhorse behind
  the very definition of the integral and of $L^p$.
- *Wavelets, Littlewood–Paley, frames.* Atoms localized in *both* space and frequency;
  frames relax orthogonality and even linear independence (overcomplete reconstruction).
- *Schmidt decomposition* of a vector in a tensor product $H_1\otimes H_2$ — the SVD wearing
  the clothes of quantum entanglement.

**Decompositions of operators.**

- *Spectral theorem* for self-adjoint / normal / unitary $A$: $A=\int_{\sigma(A)}\lambda\,
  dE(\lambda)$, with $E$ a projection-valued measure. For compact self-adjoint $A$ this is
  the eigenexpansion $A=\sum_n\lambda_n\langle e_n,\cdot\rangle e_n$.
- *Singular value decomposition* of compact $A:H_1\to H_2$:
  $A=\sum_n\sigma_n\langle v_n,\cdot\rangle u_n$. The spectral theorem for operators that are
  neither square nor self-adjoint.
- *Polar decomposition* $A=U\lvert A\rvert$, $\lvert A\rvert=(A^\ast A)^{1/2}$, $U$ a partial isometry. The operator
  analogue of $z=e^{i\theta}\lvert z\rvert$ and the skeleton of the SVD.
- *Finite-rank approximation*: $\mathcal K(H)=\overline{\lbrace \text{finite rank}\rbrace}^{\,\|\cdot\|}$,
  realized concretely by truncating the SVD.
- *Schatten classes* (trace-class, Hilbert–Schmidt): refinements graded by the
  $\ell^p$-summability of the singular values $\lbrace \sigma_n\rbrace$.
- *Holomorphic / Riesz functional calculus*: spectral projections
  $P=\frac1{2\pi i}\oint(z-A)^{-1}\,dz$ isolating a piece of the spectrum, valid even for
  non-normal $A$.
- *Lebesgue decomposition of the spectrum* $H=H_{\mathrm{pp}}\oplus H_{\mathrm{ac}}\oplus
  H_{\mathrm{sc}}$ (pure point / absolutely continuous / singular continuous) — bound states
  versus scattering states in quantum mechanics.
- *Wold decomposition* of an isometry: shift $\oplus$ unitary.
- *Direct integral / von Neumann reduction* $H=\int^{\oplus}H_\xi\,d\mu(\xi)$ — the continuous
  cousin of "decompose into irreducibles."

**Decompositions of spaces and structures.**

- *Orthogonal decomposition* $H=M\oplus M^\perp$ (the projection theorem). The most-used fact
  in the subject.
- *Peter–Weyl* $L^2(G)=\widehat{\bigoplus}\_\pi(\dim\pi)\,V_\pi$ for compact $G$: decompose a
  representation into irreducibles. Fourier series is the abelian special case.
- *Hodge decomposition* $\Omega^k=\mathcal H^k\oplus d\,\Omega^{k-1}\oplus d^\ast\Omega^{k+1}$
  on a compact Riemannian manifold; *Helmholtz* (vector field $=$ gradient $+$ curl) is its
  baby version.
- *Radon–Nikodym / Lebesgue* decomposition of measures; *Jordan* $\mu=\mu^+-\mu^-$.

That is a lot. Now the unification.

---

## 4. The master theorem: every nice operator is multiplication by a coordinate

**Driving idea.** Diagonalizing a matrix means finding coordinates in which it is
$\mathrm{diag}(\lambda_i)$, i.e. *multiplication by the function $i\mapsto\lambda_i$ on the
finite set of indices*. The correct infinite-dimensional generalization is not "find
eigenvectors" — eigenvectors may fail to exist — but **find a measure space on which the
operator becomes multiplication by a function.**

> **Spectral theorem (multiplication form).** For every self-adjoint $A$ on a separable
> $H$ there is a $\sigma$-finite measure space $(X,\mu)$, a real-valued $g\in
> L^\infty_{\mathrm{loc}}(\mu)$, and a unitary $W:H\to L^2(X,\mu)$ such that
> 
> $$W A W^{-1} = M_g, \qquad (M_g\psi)(x)=g(x)\,\psi(x).$$
> 
> For normal $A$ the same holds with complex-valued $g$; for unitary $A$, with $\lvert g\rvert=1$.

Read that slowly: *up to a change of orthonormal coordinates, the only self-adjoint
operators in the world are "multiply by a real function."* This is the theorem all the
others are corollaries or special realizations of. The abstract reason is
**Gelfand–Naimark**: the commutative $C^\ast$-algebra generated by a normal $A$ is
isometrically $\ast$-isomorphic to $C(\sigma(A))$, so $A$ "is" the coordinate function
$\lambda\mapsto\lambda$ on its spectrum, and the measure $\mu$ records how $H$ is built
out of $\sigma(A)$.

Two corollaries deserve names because you keep meeting them.

**The equivalent projection-valued form.** Pulling the indicator functions
$\mathbf 1\_\Omega\in C(\sigma(A))$ back through $W$ gives orthogonal projections
$E(\Omega):=W^{-1}M\_{\mathbf 1\_\Omega}W$. The assignment $\Omega\mapsto E(\Omega)$ is a
*projection-valued measure*: $E(\sigma(A))=I$, and it is countably additive in the strong
topology. Then

$$A=\int_{\sigma(A)}\lambda\,dE(\lambda), \qquad f(A)=\int_{\sigma(A)} f(\lambda)\,dE(\lambda).$$

The second identity is the **Borel functional calculus** — you can apply any bounded Borel
$f$ to $A$, and it respects sums, products and adjoints. (This is where $e^{itA}$,
$\sqrt A$ for $A\ge0$, and spectral projections all come from.)

**Discrete spectrum collapses the integral to a sum.** If $A$ is *compact* and
self-adjoint, $\sigma(A)$ is a sequence $\lambda_n\to0$, the projection-valued measure is
atomic, $E=\sum_n P_n$ with $P_n$ the orthogonal projection onto $\ker(A-\lambda_n)$, and
the integral degenerates to the **eigenexpansion**

$$A=\sum_n\lambda_n\,\langle e_n,\cdot\rangle\,e_n, \qquad x=\sum_n\langle e_n,x\rangle e_n.$$

*Discrete spectrum gives sums; continuous spectrum gives integrals.* That single sentence
explains why some decompositions are series (Fourier series, eigenfunction expansions) and
others are integrals (Fourier transform, direct integrals): it is the
$\mathrm{pp}$-versus-$\mathrm{ac}$ distinction of Section 3 in action.

<figure>
  <img src="/assets/images/notes/math_topics/operator_decompositions/spectral-staircase.png" alt="Left: a step function jumping at eigenvalues. Right: a smooth increasing ramp. Both are the cumulative spectral measure."
       style="width:100%;max-width:860px;height:auto;display:block;margin:1.5em auto;">
  <figcaption style="text-align:center;font-size:0.9em;color:#666;">
    <strong>Figure 4.</strong> The resolution of identity
    $\lambda\mapsto E((-\infty,\lambda])$, scalarized to the increasing function
    $\langle x, E((-\infty,\lambda])\,x\rangle$. <strong>Left:</strong> a discrete spectrum
    makes it jump at each eigenvalue, so $A=\sum_n\lambda_n P_n$ is a <em>sum</em>.
    <strong>Right:</strong> a continuous spectrum makes it rise smoothly, so
    $A=\int\lambda\,dE$ is an <em>integral</em>. One picture for “discrete gives sums,
    continuous gives integrals.”
  </figcaption>
</figure>

---

## 5. The secret handshake: the spectral theorem *is* Lebesgue integration

Here is the observation that retired my own confusion as a student, and that directly
answers "is simple-function approximation related to the spectral theorem?"

Recall how the Lebesgue integral is built. You approximate a measurable $f$ by simple
functions, $f=\lim_i \sum_k a_k^{(i)}\mathbf 1\_{A_k^{(i)}}$, *define*
$\int \sum_k a_k\mathbf 1_{A_k}\,d\mu=\sum_k a_k\,\mu(A_k)$, and pass to the limit. The
atoms are indicators; the bookkeeping device is the scalar measure $\mu$.

Now run the *identical* construction with the projection-valued measure $E$ in place of
$\mu$. A "simple function" $\sum_k\lambda_k\mathbf 1_{\Omega_k}$ is sent to the operator
$\sum_k\lambda_k E(\Omega_k)$ — a finite combination of spectral projections, the operator
world's simple function. Refine the partition of $\sigma(A)$ and pass to the limit:

$$\int_{\sigma(A)}\lambda\,dE(\lambda) \;=\;\lim_{\text{mesh}\to0}\sum_k \lambda_k\,E(\Omega_k) \;=\;A.$$

This is *the same theorem of integration*, with scalars promoted to commuting projections.
Concretely, it reduces to ordinary scalar integration vector by vector: for each $x\in H$,

$$\mu_x(\Omega):=\langle x,E(\Omega)x\rangle=\|E(\Omega)x\|^2$$

is an honest finite measure on $\sigma(A)$, and

$$ \langle x, f(A)x\rangle=\int_{\sigma(A)} f(\lambda)\,d\mu_x(\lambda). $$

So your items "spectral theorem" and "simple-function approximation" are not cousins — they
are the **same construction at two levels of generality**. Indicators are the building
blocks of a measure; *spectral projections are the indicators of an operator.* Once you see
this, the list in Section 3 stops looking like a zoo and starts looking like one theorem
viewed from several angles.

<figure>
  <img src="/assets/images/notes/math_topics/operator_decompositions/secret-handshake.png" alt="Two nearly identical pictures: a function with its range partitioned into levels, captioned with the scalar Lebesgue integral on the left and the operator spectral integral on the right."
       style="width:100%;max-width:860px;height:auto;display:block;margin:1.5em auto;">
  <figcaption style="text-align:center;font-size:0.9em;color:#666;">
    <strong>Figure 5.</strong> The secret handshake. <strong>Left:</strong> the scalar
    Lebesgue integral partitions the <em>range</em> of $f$ into levels and weights each by
    the measure $\mu(A_k)$ of its preimage, $\int f\,d\mu=\sum_k a_k\,\mu(A_k)$.
    <strong>Right:</strong> the spectral integral partitions the <em>spectrum</em>
    identically and weights each band by a projection $E(\Omega_k)$,
    $A=\int\lambda\,dE=\sum_k\lambda_k E(\Omega_k)$. The same construction, with scalars
    promoted to projections.
  </figcaption>
</figure>

---

## 6. Fourier analysis: diagonalizing a symmetry, not an operator

Where do the characters $e^{i\xi x}$ come from? Not from one operator but from a whole
*group of symmetries*: translation. Let $T_a f(x)=f(x-a)$. These commute, and

$$ T_a\,e^{i\xi x}=e^{-i\xi a}\,e^{i\xi x},$$

so each character is a *joint eigenvector* of every translation at once. The Fourier
transform $\mathcal F$ is precisely the unitary that simultaneously diagonalizes the whole
translation group: it conjugates each $T_a$ to multiplication by $e^{-i\xi a}$, and the
self-adjoint generator $P=-i\,d/dx$ (momentum) to multiplication by $\xi$. **Plancherel**,
$\|f\|\_{L^2}=\|\hat f\|\_{L^2}$, is the statement that $\mathcal F$ is unitary — i.e. it is
the multiplication-operator form of the spectral theorem for $P$, whose spectrum is the
whole real line (purely absolutely continuous, hence an integral, not a sum).

<figure>
  <img src="/assets/images/notes/math_topics/operator_decompositions/fourier-translation-eigen.png" alt="Top: a bump and its translate, the shape moves. Bottom: a cosine wave and its translate, identical shape shifted only by a phase."
       style="width:100%;max-width:720px;height:auto;display:block;margin:1.5em auto;">
  <figcaption style="text-align:center;font-size:0.9em;color:#666;">
    <strong>Figure 6.</strong> Why the characters are the Fourier atoms. <strong>Top:</strong>
    a generic function and its translate $f(x-a)$ — the shape moves, so it is no eigenvector
    of translation. <strong>Bottom:</strong> a character $e^{i\xi x}$ and its translate differ
    only by a global phase, $T_a e^{i\xi x}=e^{-i\xi a}e^{i\xi x}$. Each character is a
    <em>joint eigenvector</em> of every translation at once, which is exactly why $\mathcal{F}$
    diagonalizes the whole translation group.
  </figcaption>
</figure>

Why characters specifically, and not some other atoms? Because they are the
*one-dimensional representations* of the group $(\mathbb R,+)$. This is the cleanest case
of the **Peter–Weyl / Gelfand** principle: to diagonalize a commutative algebra of
operators you decompose along its joint spectrum, and for a group algebra that joint
spectrum is the group of characters. Fourier series ($G=$ circle, characters $e^{in\theta}$,
discrete spectrum, hence a *sum*) and the Fourier transform ($G=\mathbb R$, continuous
spectrum, hence an *integral*) are two faces of one theorem; spherical harmonics are the
same story for the rotation group $SO(3)$, where the irreducibles are now finite-dimensional
and the "Fourier coefficients" become blocks. The general statement,

$$L^2(G)\;=\;\widehat{\bigoplus_{\pi\in\widehat G}}\;(\dim\pi)\,V_\pi \qquad(G\text{ compact}),$$

is Peter–Weyl, and it contains all classical Fourier analysis as the abelian special case.

So: **Fourier $\subset$ spectral theorem**, specialized to the translation symmetry. The
atoms are forced by the symmetry you insist on diagonalizing.

---

## 7. Between two spaces: SVD, polar, and finite rank

The spectral theorem wants $A$ to map $H$ to *itself* and to be normal. What if $A:H_1\to
H_2$ is an arbitrary compact operator — a rectangular, non-self-adjoint object, like a
forward map in an inverse problem? The fix is to manufacture a self-adjoint operator from
$A$.

**Driving idea.** $A^\ast A$ is self-adjoint, compact, and positive on $H_1$. Diagonalize
*it* and transport the result through $A$.

Apply Section 4 to $A^\ast A=\sum_n\sigma_n^2\langle v_n,\cdot\rangle v_n$ with
$\sigma_n\ge0$, $\sigma_n\downarrow0$, $\lbrace v_n\rbrace$ an ONS in $H_1$. Set $u_n:=\sigma_n^{-1}Av_n$
(for $\sigma_n>0$); a one-line check shows $\lbrace u_n\rbrace$ is an ONS in $H_2$, and

$$ A=\sum_n \sigma_n\,\langle v_n,\cdot\rangle\,u_n \qquad(\textbf{SVD}). $$

The **polar decomposition** $A=U|A|$ with $\lvert A\rvert=(A^\ast A)^{1/2}=\sum_n\sigma_n\langle v_n,
\cdot\rangle v_n$ and $U:v_n\mapsto u_n$ is just the SVD with the diagonal part packaged
separately — the exact analogue of $z=e^{i\theta}\,\lvert z\rvert$ for complex numbers.

<figure>
  <img src="/assets/images/notes/math_topics/operator_decompositions/svd-geometry-polar.png" alt="Left: a unit circle mapped to an ellipse with singular vectors and singular-value semi-axes. Right: the same map as a stretch followed by a rotation."
       style="width:100%;max-width:860px;height:auto;display:block;margin:1.5em auto;">
  <figcaption style="text-align:center;font-size:0.9em;color:#666;">
    <strong>Figure 7.</strong> SVD and polar decomposition. <strong>Left:</strong> a compact
    $A$ sends the unit sphere to an ellipse, mapping orthonormal inputs $v_n$ to $\sigma_n u_n$
    — the singular values are the semi-axes. <strong>Right:</strong> the same map factored as
    $A=U\lvert A\rvert$: the positive operator $\lvert A\rvert=(A^\ast A)^{1/2}$ stretches along
    the $v$-frame (dashed), then the partial isometry $U$ rotates onto the $u$-frame (solid).
    The operator analogue of $z=e^{i\theta}\lvert z\rvert$.
  </figcaption>
</figure>

**Finite-rank approximation is now a one-liner, with a sharp error.** Truncating,
$A_k:=\sum_{n\le k}\sigma_n\langle v_n,\cdot\rangle u_n$ has rank $\le k$, and

$$ \|A-A_k\| = \sigma_{k+1}, \qquad \min_{\operatorname{rank}B\le k}\|A-B\| = \sigma_{k+1}.$$

This is **Eckart–Young–Schmidt–Mirsky**: the truncated SVD is the *best* rank-$k$
approximation, in operator norm *and* in every Schatten norm, and the inequality is an
*equality* — there is nothing better. Letting $k\to\infty$ gives $\|A-A_k\|=\sigma_{k+1}\to0$,
which proves the structural fact that **compact operators are exactly the operator-norm
limits of finite-rank ones** in a Hilbert space. Your item 3 is therefore the
*quantitative* shadow of your item 2.

<figure>
  <img src="/assets/images/notes/math_topics/operator_decompositions/eckart-young-rank-k.png" alt="A synthetic image and its best rank-1, rank-5, rank-20 approximations, each labelled with operator-norm error equal to the next singular value, plus a singular-value decay curve."
       style="width:100%;max-width:880px;height:auto;display:block;margin:1.5em auto;">
  <figcaption style="text-align:center;font-size:0.9em;color:#666;">
    <strong>Figure 8.</strong> Eckart–Young–Schmidt–Mirsky in action: a synthetic pattern
    (smooth blobs and a sharp ring) and its best rank-$k$ approximations. The truncated SVD is
    the <em>optimal</em> rank-$k$ approximant, and the operator-norm error is exactly the next
    singular value, $\|A-A_k\|=\sigma_{k+1}$ (printed above each panel and marked on the decay
    curve). As $k$ grows the error tends to $0$ — which is why the compact operators are
    precisely the norm-limits of finite-rank ones.
  </figcaption>
</figure>

The same algebra, read in $H_1\otimes H_2\cong$ Hilbert–Schmidt operators, is the **Schmidt
decomposition** of an entangled state; the singular values are the entanglement spectrum.
And grading operators by how fast $\sigma_n\to0$ produces the **Schatten classes**: $A$ is
trace-class iff $\sum_n\sigma_n<\infty$, Hilbert–Schmidt iff $\sum_n\sigma_n^2<\infty$. These
are the operator analogues of $\ell^1\subset\ell^2\subset c_0$ — the same summability
hierarchy you know for sequences, applied to the SVD coefficients.

---

## 8. A worked unification you have already met: Karhunen–Loève

To see four of these decompositions collapse into one, take a centered Gaussian field
$X=(X_t)_{t\in[0,1]}$ with continuous covariance $K(s,t)=\mathbb E[X_sX_t]$. The covariance
operator 

$$(\mathcal C\varphi)(s)=\int_0^1 K(s,t)\varphi(t)\,dt$$

on $L^2[0,1]$ is **self-adjoint,
positive, and compact** (indeed trace-class, with $\operatorname{tr}\mathcal C=\int_0^1
K(t,t)\,dt$). So Section 4 applies verbatim:

$$ \mathcal C\varphi_n=\lambda_n\varphi_n,\qquad  K(s,t)\;\overset{\text{Mercer}}{=}\;\sum_n\lambda_n\,\varphi_n(s)\varphi_n(t),\qquad X_t=\sum_n\sqrt{\lambda_n}\,\xi_n\,\varphi_n(t),\ \ \xi_n\overset{\text{iid}}\sim N(0,1).$$

Mercer's theorem is the spectral theorem for $\mathcal C$ promoted to *uniform* convergence
of the kernel; the Karhunen–Loève expansion is the resulting eigenfunction decomposition of
the *field*; the eigenvalue decay $\lambda_n\downarrow0$ is the SVD spectrum that controls
how well a truncation approximates $X$ in $L^2$ (Eckart–Young again, now as the optimal
finite-dimensional reduction of a random function). When $X$ is stationary, translation
invariance forces $\varphi_n$ to be the *Fourier* atoms and KL collapses to the spectral
representation of stationary processes. One operator, and every theme of this note appears
at once.

<figure>
  <img src="/assets/images/notes/math_topics/operator_decompositions/karhunen-loeve.png" alt="Three panels: a Brownian covariance heatmap, its sine eigenfunctions with eigenvalue decay, and a sample path reconstructed from increasing numbers of modes."
       style="width:100%;max-width:900px;height:auto;display:block;margin:1.5em auto;">
  <figcaption style="text-align:center;font-size:0.9em;color:#666;">
    <strong>Figure 9.</strong> Karhunen–Loève for Brownian motion, where several themes meet on
    one covariance operator. <strong>(a)</strong> the kernel $K(s,t)=\min(s,t)$;
    <strong>(b)</strong> its eigenfunctions $\varphi_n$ (here sines) and the eigenvalue decay
    $\lambda_n\downarrow 0$; <strong>(c)</strong> one sample path reconstructed from the
    truncated expansion $X_t=\sum_n\sqrt{\lambda_n}\,\xi_n\varphi_n(t)$. More modes give a
    rougher, more faithful path, and the decay of $\lambda_n$ controls the truncation error —
    Eckart–Young once more.
  </figcaption>
</figure>

---

## 9. When diagonalization fails — and what replaces it

It is just as instructive to see the *boundaries* of Template A, because they explain why
some of the more exotic decompositions exist at all.

**Non-normal operators have no spectral theorem.** In finite dimensions you fall back on the
**Jordan decomposition** $A=D+N$ (diagonalizable plus nilpotent). In infinite dimensions even
that breaks: there need be no eigenvectors and no Jordan form. The replacement is the
**holomorphic functional calculus** and **Riesz projections**
$P_\Gamma=\frac1{2\pi i}\oint_\Gamma(z-A)^{-1}\,dz$, which still let you peel off the part of
the spectrum enclosed by a contour $\Gamma$ without needing $A$ to be normal. This is a
decomposition of the *space* into $A$-invariant pieces, bought with complex analysis instead
of orthogonality. (It is also why eigendecomposition of a non-normal matrix is numerically
treacherous while the SVD is rock-stable: the latter only ever diagonalizes the *self-adjoint*
$A^\ast A$.)

**Continuous spectrum needs finer bookkeeping.** Even for self-adjoint $A$, the spectral
measure can be non-atomic, so there are no eigenvectors to expand in — only the
projection-valued integral. The **Lebesgue decomposition** $H=H_{\mathrm{pp}}\oplus
H_{\mathrm{ac}}\oplus H_{\mathrm{sc}}$ sorts $H$ by the type of the scalar spectral measures
$\mu_x$, and in quantum mechanics it is exactly the split into bound states ($H_{\mathrm{pp}}$,
genuine eigenvectors) and scattering states ($H_{\mathrm{ac}}$). The exotic middle piece
$H_{\mathrm{sc}}$ is the kind of thing that only an honest operator-valued integral can even
*name*. The fully general "diagonalization in the presence of continuous spectrum" is the
**direct integral** $H=\int^\oplus H_\xi\,d\mu(\xi)$ — the continuous analogue of "decompose
into irreducibles," and the gateway to von Neumann algebra theory.

<figure>
  <img src="/assets/images/notes/math_topics/operator_decompositions/diagonalization-fails.png" alt="Left: eigenvalues of a perturbed Jordan block scattered onto a large circle, with an inset comparing eigenvalue and singular-value sensitivity. Right: three cumulative spectral measures — step, smooth, and Cantor staircase."
       style="width:100%;max-width:880px;height:auto;display:block;margin:1.5em auto;">
  <figcaption style="text-align:center;font-size:0.9em;color:#666;">
    <strong>Figure 10.</strong> The boundaries of diagonalization. <strong>Left:</strong> a
    non-normal Jordan block has all eigenvalues at $0$, yet a perturbation of size $10^{-3}$
    flings them onto a circle of radius $\approx 0.75$ while the singular values barely move
    (inset, log scale) — why the SVD is stable where eigendecomposition is treacherous.
    <strong>Right:</strong> the three spectral types as cumulative spectral measures — pure
    point (jumps, i.e. genuine eigenvectors), absolutely continuous (smooth), and singular
    continuous (the Cantor devil’s staircase) — the split
    $H=H_{\mathrm{pp}}\oplus H_{\mathrm{ac}}\oplus H_{\mathrm{sc}}$.
  </figcaption>
</figure>

---

## 10. So why *so many*? Six honest reasons

We can now answer the third question directly, and not just hand-wave "different tools for
different jobs."

**1. There is no universal basis, because operators do not commute.** A single ONB can
simultaneously diagonalize a *commuting* family and no more. Position and momentum, the
shift and its adjoint, the Laplacian and a generic potential — each demands its own atoms.
The multiplicity of decompositions is the shadow of noncommutativity.

**2. The atoms encode the structure you chose to trivialize.** Indicators for *measure*,
characters for *translation*, eigenvectors for *a fixed self-adjoint operator*, rank-one
operators for *low-complexity approximation*, irreducibles for *a group action*. Change the
structure you privilege and the optimal atoms change. There is no neutral decomposition.

**3. Localization is a genuine trade-off (the uncertainty principle).** Fourier atoms are
perfectly sharp in frequency and completely spread in space; indicators are the reverse;
$\|xf\|_2\,\|\xi\hat f\|_2\ge\tfrac1{4\pi}\|f\|_2^2$ forbids being sharp in both, with
*equality only for Gaussians*. Wavelets, Gabor frames and Littlewood–Paley pieces exist
precisely to buy intermediate trade-offs. A whole family of decompositions is parametrized
by *where on the space/frequency see-saw you want to sit.*

<figure>
  <img src="/assets/images/notes/math_topics/operator_decompositions/time-frequency-tiling.png" alt="Four tilings of the time-frequency plane with equal-area cells: tall thin (indicators), short wide (Fourier), uniform grid (Gabor), and dyadic multi-resolution (wavelets)."
       style="width:100%;max-width:640px;height:auto;display:block;margin:1.5em auto;">
  <figcaption style="text-align:center;font-size:0.9em;color:#666;">
    <strong>Figure 11.</strong> The uncertainty see-saw: each panel tiles the time–frequency
    plane with equal-area cells (the Heisenberg lower bound). Indicators are sharp in time and
    spread in frequency; Fourier atoms are the reverse; the Gabor/STFT grid fixes a uniform
    compromise; wavelets trade resolution dyadically — fine in time at high frequency, fine in
    frequency at low. A whole family of decompositions is parametrized by where on this see-saw
    you choose to sit.
  </figcaption>
</figure>

**4. Existence, computation and stability are different demands.** The spectral theorem is
an *existence* statement. Realizing it in practice needs *constructions* — Gram–Schmidt
($QR$), Krylov subspaces (Lanczos/Arnoldi), Cholesky — and *stability* analysis (the SVD is
chosen over eigendecomposition in numerics exactly for its robustness). The same underlying
factorization spawns several decompositions according to whether you want to *prove it
exists*, *compute it*, or *trust it under perturbation*.

**5. Exact reconstruction versus dense approximation are different contracts.** An ONB
reconstructs *exactly* (equality in Parseval); simple functions and finite-rank operators
only *approximate* (they are dense, not a basis). Frames sit in between: overcomplete, so
reconstruction is exact but not unique. These are genuinely different convergence promises,
and you reach for different ones depending on whether you need an identity or a limit.

**6. They live in different categories.** Some decompositions construct the *space*
(simple functions building $L^p$); some analyze *operators* on it (spectral theorem, SVD);
some respect *geometry* (Hodge, Helmholtz); some respect *symmetry* (Peter–Weyl). They are
not competitors — they sit at different layers of the same edifice. Simple functions must
exist before you can integrate; you must integrate before $L^2$ exists; $L^2$ must exist
before the spectral theorem has a stage to perform on.

---

## 11. What is really going on

If I had to compress the whole note into one sentence: **a decomposition is a coordinate
system, and the spectral theorem is the assertion that for any reasonable operator a
coordinate system exists in which it is multiplication by a function.** Everything else is
either (a) a *special realization* of that statement adapted to a symmetry (Fourier,
harmonics, Peter–Weyl), (b) a *bridge between two spaces* obtained by feeding $A^\ast A$ to
the master theorem (SVD, polar, finite-rank, Schmidt, Schatten), (c) the *measure-theoretic
substrate* of the very same integral, now with scalar atoms instead of operator atoms
(simple functions, Radon–Nikodym, Lebesgue), or (d) what you do *when diagonalization fails*
(Jordan, Riesz projections, direct integrals).

Three parting remarks, in the spirit of "where does this go next."

- **The unifying abstraction is Gelfand–Naimark.** A commutative $C^\ast$-algebra is
  $C_0$ of its spectrum; the Fourier transform is the Gelfand transform of the group algebra,
  and the spectral theorem is the Gelfand transform of the algebra generated by one normal
  operator. If you want one theorem from which Sections 4–6 fall out, that is it.
- **Noncommutative decomposition is the modern frontier.** When the operators genuinely do
  not commute, "decompose into irreducibles" becomes the *direct integral* decomposition of a
  von Neumann algebra into factors — the structure theory that the type $\mathrm{I/II/III}$
  classification organizes. Your $H_{\mathrm{pp}}/H_{\mathrm{ac}}/H_{\mathrm{sc}}$ split is the
  abelian, baby version of this.
- **Geometry and analysis meet in Hodge theory.** $\Omega^k=\mathcal H^k\oplus
  \operatorname{im}d\oplus\operatorname{im}d^\ast$ is, on the one hand, the spectral
  decomposition of the Hodge Laplacian (Template A, compact resolvent, hence a *sum*), and on
  the other hand a theorem that the harmonic atoms $\mathcal H^k$ compute de Rham cohomology
  $H^k_{\mathrm{dR}}$. The same diagonalization that an analyst reads as "expand in
  eigenforms" a geometer reads as "every class has a unique harmonic representative." That
  two such different readings of one decomposition are both true is, to me, the best evidence
  that we have been talking about a single idea all along.