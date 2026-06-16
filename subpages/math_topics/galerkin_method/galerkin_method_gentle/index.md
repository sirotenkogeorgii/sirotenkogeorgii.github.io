---
layout: post
title: "The Galerkin method, or: how to solve an infinite-dimensional equation with finitely many numbers"
date: 2026-06-14
mathjax: true
---

Many of the central equations of analysis ask us to find an *unknown function*. The Poisson equation $-\Delta u = f$ (with, say, $u$ vanishing on the boundary of a domain $\Omega$) is the prototype: we are handed a function $f$ and asked to produce a function $u$. The trouble is that a function is an infinite amount of data — to specify $u$ exactly we must give its value at uncountably many points — whereas any computation we can actually carry out manipulates only *finitely many numbers*. Every numerical method for such a problem must therefore confront the same basic question: **how do we replace the infinite-dimensional unknown by a finite-dimensional surrogate, in a way that is faithful to the original equation?**

There are two broad philosophies. The first is to *discretize the operator*: replace the derivatives in $-\Delta u = f$ by difference quotients on a grid, turning the differential equation into a system of algebraic equations relating the values $u(x_i)$ at grid points. This is the finite-difference idea. It is intuitive, but it ties the method tightly to the grid, becomes awkward on complicated domains, and its convergence theory has a slightly ad hoc flavour.

The Galerkin method takes the second route — *discretize the solution space*. Rather than tampering with the operator, we keep the equation morally intact and instead agree to look for the solution only within a chosen finite-dimensional space of candidate functions. The whole subtlety is then concentrated into a single, very clean idea, which is the subject of this post. The slogan, which we will spend the rest of the post unpacking and justifying, is:

> **The Galerkin solution is the projection of the true solution onto the subspace we are searching in.**

Once one sees the method as projection, everything else — existence, uniqueness, error estimates, the reduction to a linear system — falls out almost mechanically.

## The weak formulation, and why we want it

Before we can project, we must phrase the problem in a language where "projection" and "orthogonality" make sense, i.e. in terms of an inner product. This is the role of the *weak* (or *variational*) formulation, and it is worth seeing it arise concretely before we abstract it.

**The key move is to test the equation against arbitrary functions.** Take $-\Delta u = f$ on $\Omega$ with $u\mid\_{\partial\Omega} = 0$. Multiply both sides by a smooth test function $v$ that also vanishes on the boundary, integrate over $\Omega$, and integrate by parts. The boundary term dies (because $v = 0$ on $\partial\Omega$), and we are left with

$$ \int_\Omega \nabla u \cdot \nabla v \, dx \;=\; \int_\Omega f\, v \, dx. \tag{1} $$

This is to hold for *every* admissible test function $v$. Equation (1) is genuinely weaker than the original: it asks for one derivative of $u$ to exist and be square-integrable, not two. We have traded one derivative on $u$ for one derivative on $v$ — symmetrizing the demand for regularity, and thereby enlarging the space in which we are entitled to hunt for a solution. The natural home for both $u$ and $v$ turns out to be the Sobolev space $H^1_0(\Omega)$ of functions with one square-integrable derivative and vanishing boundary trace.

Let me now fix conventions for the abstract setting, which we will use throughout.

- We work over the real scalars, in a real Hilbert space $V$ with norm $\|\cdot\|$. (Think $V = H^1_0(\Omega)$.)
- $a(\cdot,\cdot)\colon V \times V \to \mathbb{R}$ is a **bilinear form** — the left-hand side of (1), $a(u,v) = \int_\Omega \nabla u\cdot\nabla v$.
- $\ell\colon V \to \mathbb{R}$ is a **continuous linear functional** — the right-hand side, $\ell(v) = \int_\Omega f v$.

The abstract problem is then:

$$ \text{find } u \in V \text{ such that } \quad a(u,v) = \ell(v) \quad \text{for all } v \in V. \tag{P} $$

We will assume two structural properties of $a$, which are exactly the hypotheses of the Lax–Milgram theorem:

- **Boundedness** (continuity): there is $M > 0$ with $\;\lvert a(u,v)\rvert \le M\,\|u\|\,\|v\|\;$ for all $u,v$.
- **Coercivity** (ellipticity): there is $\alpha > 0$ with $\;a(v,v) \ge \alpha\,\|v\|^2\;$ for all $v$.

For the Poisson problem these are precisely the Poincaré inequality (coercivity) and Cauchy–Schwarz (boundedness). The Lax–Milgram theorem then guarantees that (P) has a **unique** solution $u \in V$. We take this well-posedness of the *continuous* problem as given; the point of this post is what happens when we discretize.

A remark on the symmetric case, which is the cleanest and which we will return to repeatedly: if in addition $a(u,v) = a(v,u)$, then coercivity makes $a$ itself an **inner product** on $V$. We call it the *energy inner product*, write $\|v\|_a := \sqrt{a(v,v)}$ for the associated **energy norm**, and note that boundedness and coercivity say exactly that this energy norm is equivalent to the original norm: $\;\alpha\|v\|^2 \le \|v\|_a^2 \le M\|v\|^2$. This is the case in which "projection" is literal rather than figurative.

## The Galerkin recipe

We now do the only thing a computer can do: restrict to finitely many dimensions.

**Choose a finite-dimensional subspace** $V_h \subset V$, of dimension $N$, with a basis $\phi_1, \dots, \phi_N$. (The subscript $h$ is traditional and suggests a mesh spacing; the heuristic is that smaller $h$ means a richer space $V_h$, a finer resolution of $V$.) The defining act of the Galerkin method is to pose the *same variational equation*, but with both the solution and the test functions confined to $V_h$:

$$ \text{find } u_h \in V_h \text{ such that } \quad a(u_h, v_h) = \ell(v_h) \quad \text{for all } v_h \in V_h. \tag{$\text{P}_h$} $$

Notice what we have *not* done: we have not discretized the operator, introduced a grid, or approximated any derivatives. We have only narrowed the search. Notice also the symmetry of the restriction — we shrink the trial space (where $u_h$ lives) and the test space (which $v_h$ ranges over) *together*. That both are $V_h$ is the hallmark of the (standard, or *Bubnov*–) Galerkin method; allowing them to differ gives the Petrov–Galerkin variants discussed at the end.

**This is already a linear system in disguise.** Since $\ell$ and $a(\cdot, v)$ are linear, it suffices to impose ($\text{P}\_h$) against each basis function $\phi_i$ rather than against all $v_h$. Expanding the unknown in the basis as $u_h = \sum_{j=1}^N c_j \phi_j$ and substituting, the $i$-th equation reads

$$ \sum_{j=1}^N c_j\, a(\phi_j, \phi_i) \;=\; \ell(\phi_i), \qquad i = 1, \dots, N. $$

In matrix form this is

$$ K\mathbf{c} = \mathbf{b}, \qquad K_{ij} = a(\phi_j, \phi_i), \qquad b_i = \ell(\phi_i). $$

The matrix $K$ is the **stiffness matrix** and $\mathbf{b}$ the **load vector** — names inherited from the method's origins in structural mechanics. Solving (P) has been reduced to a finite linear-algebra problem we can hand to a computer.

**The discrete problem is automatically well-posed, for free.** We need not develop any new theory to know that $K\mathbf{c} = \mathbf{b}$ has a unique solution. Coercivity holds on all of $V$, hence in particular on the subspace $V_h$. So for any $\mathbf{c} \ne 0$, writing $w = \sum_j c_j \phi_j \ne 0$,

$$ \mathbf{c}^{\mathsf T} K \mathbf{c} = a(w, w) \ge \alpha \|w\|^2 > 0. $$

Thus $K$ is positive definite, hence invertible. The discrete solution inherits its existence and uniqueness directly from the structure of the continuous problem — no separate discrete machinery required. (In the symmetric case $K$ is moreover symmetric positive definite, the friendliest matrix there is.)

## Galerkin orthogonality: the engine of the method

Everything so far has been setup. Here is the one identity that makes the method *work*, and from which all error estimates descend.

The exact solution $u$ satisfies $a(u, v) = \ell(v)$ for **all** $v \in V$ — in particular for all $v_h \in V_h \subset V$. The discrete solution $u_h$ satisfies $a(u_h, v_h) = \ell(v_h)$ for all $v_h \in V_h$. Subtracting these two statements, the right-hand sides cancel exactly, and we obtain the **Galerkin orthogonality** relation:

$$ a(u - u_h,\, v_h) = 0 \qquad \text{for all } v_h \in V_h. \tag{G} $$

The error $u - u_h$ is *orthogonal*, in the form $a$, to the entire subspace $V_h$. This deserves to be read slowly, because it is the whole game. The method does not — cannot — make the residual vanish; a function drawn from an $N$-dimensional space has only $N$ degrees of freedom and generically cannot satisfy a PDE exactly. What it does instead is spend those $N$ degrees of freedom to annihilate the residual *in the $N$ directions we can see*, namely the directions of the test functions $\phi_1, \dots, \phi_N$. The error is pushed entirely into the part of $V$ orthogonal to $V_h$ — the part our finite resolution is blind to anyway.

**In the symmetric case this is literally an orthogonal projection.** When $a$ is the energy inner product, (G) says that $u_h$ is the foot of the perpendicular dropped from $u$ onto $V_h$. And the orthogonal projection is, by the projection theorem, the *closest* point: $u_h$ is the **best approximation** to $u$ from $V_h$ in the energy norm. Let me give the one-line proof, because it is just Pythagoras. For any competitor $v_h \in V_h$, decompose $u - v_h = (u - u_h) + (u_h - v_h)$. The two pieces are $a$-orthogonal — the first is orthogonal to all of $V_h$ by (G), the second lies in $V_h$ — so

$$ \|u - v_h\|_a^2 = \|u - u_h\|_a^2 + \|u_h - v_h\|_a^2 \;\ge\; \|u - u_h\|_a^2, $$

with equality if and only if $v_h = u_h$. Hence

$$ \|u - u_h\|_a = \min_{v_h \in V_h} \|u - v_h\|_a. \tag{2} $$

This is the conceptual summit of the subject. **The Galerkin method automatically finds the best approximation the subspace has to offer** (in the energy norm). One could not do better even with advance knowledge of the true solution $u$; the method extracts the optimal element of $V_h$ without ever seeing $u$.

<figure>
  <img src="{{ '/assets/images/notes/galerkin_method/galerkin_projection.png' | relative_url }}" alt="Two panels. Left, a three-dimensional scene: a horizontal plane labelled V-h, a point u floating above it joined to the origin, and its foot u-h directly below on the plane; the vertical segment from u-h up to u is the error, meeting the plane at a marked right angle, while a dotted line out to an arbitrary competitor point v-h in the plane is visibly longer. Right, the same situation drawn as a right triangle: the base from v-h to u-h lies in V-h, the vertical leg from u-h to u is the error perpendicular to it, the hypotenuse from v-h to u is any competitor, and a box states the Pythagorean identity." loading="lazy">
  <figcaption>The one picture the whole method reduces to. The true solution $u$ lives off the subspace $V_h$; the Galerkin solution $u_h$ is its <strong>orthogonal projection</strong> onto $V_h$, so the error $u - u_h$ is perpendicular to every direction we can resolve — this is Galerkin orthogonality (G). Because the projection is the foot of the perpendicular, Pythagoras gives $\|u - v_h\|_a^2 = \|u - u_h\|_a^2 + \|u_h - v_h\|_a^2$ for any competitor $v_h \in V_h$ (right), so $u_h$ is the <strong>closest</strong> point of $V_h$ to $u$ in the energy norm — the best approximation the subspace can offer, found without ever seeing $u$.</figcaption>
</figure>

## Céa's lemma: quasi-optimality in general

In the non-symmetric case $a$ is not an inner product, so $u_h$ need not be the exact best approximation. But it is *quasi-optimal*: best up to a constant that depends only on the form $a$, never on the subspace or on $u$. This is **Céa's lemma**:

$$ \|u - u_h\| \;\le\; \frac{M}{\alpha}\, \min_{v_h \in V_h} \|u - v_h\|. \tag{3} $$

The proof is a clean two-line application of coercivity, boundedness, and Galerkin orthogonality, and it is instructive to see exactly where each hypothesis enters. Fix any $v_h \in V_h$. Start from coercivity, insert and split using bilinearity:

$$ \alpha\,\|u - u_h\|^2 \;\le\; a(u - u_h,\, u - u_h) \;=\; \underbrace{a(u - u_h,\, u - v_h)}_{\text{bound this}} \;+\; \underbrace{a(u - u_h,\, v_h - u_h)}_{=\,0}. $$

The second term vanishes by Galerkin orthogonality (G), since $v_h - u_h \in V_h$ — *this is the only place (G) is used, and it is indispensable*. The first term is controlled by boundedness: it is $\le M\,\|u - u_h\|\,\|u - v_h\|$. Cancelling one factor of $\|u - u_h\|$ leaves $\|u - u_h\| \le (M/\alpha)\,\|u - v_h\|$, and taking the infimum over $v_h \in V_h$ gives (3).

In the symmetric case the same argument carried out in the energy norm sharpens the constant from $M/\alpha$ to $\sqrt{M/\alpha}$: starting from (2), $\;\alpha\|u-u_h\|^2 \le \|u-u_h\|\_a^2 \le \|u-v_h\|\_a^2 \le M\|u-v_h\|^2$, so $\|u - u_h\| \le \sqrt{M/\alpha}\,\min_{v_h}\|u - v_h\|$.

<figure>
  <img src="{{ '/assets/images/notes/galerkin_method/galerkin_cea_oblique.png' | relative_url }}" alt="Two panels, each showing a point u and a straight line labelled V-h. Left, the symmetric case: u is projected straight down onto the line at a right angle, the foot u-h being the exact closest point. Right, the non-symmetric case: the Galerkin condition projects u onto the line along a slanted, oblique direction, so u-h lands further along the line than the true closest point (marked 'best approx.'), yet only by a bounded amount." loading="lazy">
  <figcaption>What "quasi-optimal" means, geometrically. When $a$ is symmetric it is a genuine inner product and the Galerkin condition makes $u_h$ the <strong>orthogonal</strong> projection of $u$ onto $V_h$ — exactly the best approximation (left). When $a$ is non-symmetric the condition $a(u - u_h, v_h) = 0$ still projects $u$ onto $V_h$, but <strong>obliquely</strong>: $u_h$ overshoots the true closest point (right). Céa's lemma is precisely the statement that this overshoot is bounded once and for all — $\|u - u_h\| \le \frac{M}{\alpha}\min_{v_h}\|u - v_h\|$ — so the method is never worse than the best approximation times the fixed stability constant $M/\alpha$.</figcaption>
</figure>

**The significance of Céa's lemma is that it decouples the problem into two independent halves.** On one side sits the *stability* of the method — the constant $M/\alpha$ — which is a property of the operator $a$ alone, established once and for all by Lax–Milgram-type analysis, with no reference to any discretization. On the other side sits the *approximation power* of the subspace — the quantity $\min_{v_h \in V_h}\|u - v_h\|$ — which is a question of pure approximation theory: *how well can functions from $V_h$ approximate the target $u$?* This second question has nothing to do with the PDE; it is about the subspace and the regularity of $u$. So the entire error analysis of a Galerkin method reduces to: (i) bound $M/\alpha$ once, abstractly; (ii) estimate how well $V_h$ interpolates. For finite elements, step (ii) becomes a polynomial interpolation estimate of the form $\min_{v_h}\|u - v_h\| \lesssim h^k \|u\|_{H^{k+1}}$, and one reads off the convergence rate immediately. The hard analytic content has been quarantined into two clean, separately-solvable pieces.

## A worked example: the 1D Poisson problem

Let me make all of this concrete with the simplest nontrivial instance: $-u'' = f$ on the interval $(0,1)$ with $u(0) = u(1) = 0$. Here $V = H^1_0(0,1)$, and the weak form has

$$ a(u,v) = \int_0^1 u'(x)\,v'(x)\,dx, \qquad \ell(v) = \int_0^1 f(x)\,v(x)\,dx. $$

This $a$ is symmetric and coercive (Poincaré), so we are in the best case: $u_h$ will be the energy-best approximation.

**Choosing the subspace: hat functions.** Partition $[0,1]$ by nodes $x_i = ih$ for $i = 0, \dots, N+1$, with spacing $h = 1/(N+1)$. Let $V_h$ be the space of continuous, piecewise-linear functions on this mesh that vanish at the two endpoints; it has dimension $N$. The natural basis is the family of **hat functions** $\phi_i$ ($i = 1, \dots, N$), where $\phi_i$ equals $1$ at the node $x_i$, equals $0$ at every other node, and is linear in between. Each $\phi_i$ is a little tent supported only on the two intervals $[x_{i-1}, x_{i+1}]$ touching its node. These are the simplest finite elements.

**Assembling the stiffness matrix.** On its support, the derivative of $\phi_i$ is the constant $+1/h$ on $[x_{i-1}, x_i]$ and $-1/h$ on $[x_i, x_{i+1}]$, and zero elsewhere. Computing the entries $K_{ij} = \int_0^1 \phi_i' \phi_j'$:

- Diagonal: $\;a(\phi_i, \phi_i) = \int (\phi_i')^2 = (1/h^2)\cdot h + (1/h^2)\cdot h = 2/h.$
- Off-diagonal neighbours: on the single shared interval $[x_i, x_{i+1}]$ we have $\phi_i' = -1/h$ and $\phi_{i+1}' = +1/h$, so $\;a(\phi_i, \phi_{i+1}) = (-1/h^2)\cdot h = -1/h.$
- All other entries vanish, because non-adjacent hat functions have disjoint supports.

Hence the stiffness matrix is tridiagonal:

$$ K = \frac{1}{h}\begin{pmatrix} 2 & -1 & & \\ -1 & 2 & -1 & \\ & \ddots & \ddots & \ddots \\ & & -1 & 2 \end{pmatrix}. $$

**An illuminating coincidence.** This is, up to the scaling $1/h$, *exactly* the standard second-difference matrix that the finite-difference method produces for the same problem. In this simplest of cases the two philosophies land on the same left-hand side. But the agreement is not complete, and the differences are instructive. The load vector is $b_i = \int_0^1 f\,\phi_i$ — a *local weighted average* of $f$ against the hat function — rather than the pointwise sample $f(x_i)$ that finite differences would use. More importantly, the philosophies diverge entirely the moment one leaves this toy setting: on curved domains, with higher-order elements, or in the non-symmetric case, the Galerkin viewpoint generalizes cleanly while the finite-difference stencil does not.

**Where the sparsity comes from, and why it matters.** The matrix $K$ is tridiagonal — overwhelmingly zero — purely because each hat function overlaps only its immediate neighbours. This *locality of the basis* is the single most important practical feature of the finite element method: it is what makes the linear system sparse, and sparsity is what makes solving it feasible for the millions of unknowns arising in real engineering problems. Choosing a basis of locally-supported functions is not a minor implementation detail; it is the reason the whole enterprise is computationally viable.

<figure>
  <img src="{{ '/assets/images/notes/galerkin_method/galerkin_1d_poisson.png' | relative_url }}" alt="Three panels. Left, a row of overlapping triangular hat functions on a mesh of the unit interval, one tent peaked at each interior node. Middle, the exact solution sine of pi x drawn as a smooth arch, with two piecewise-linear Galerkin approximations overlaid — a coarse one with three nodes and a finer one with seven — both passing exactly through the curve at their nodes; a box notes that the nodal values are exact. Right, the stiffness matrix shown as a grid that is non-zero only on the main diagonal and the two adjacent diagonals, with entries two and minus one." loading="lazy">
  <figcaption>The recipe on the simplest real problem, $-u'' = f$ on $(0,1)$. The subspace $V_h$ is spanned by <strong>hat functions</strong>, one little tent per interior node (left). Solving $a(u_h, v_h) = \ell(v_h)$ then produces a piecewise-linear $u_h$ that converges to the exact $u = \sin\pi x$ as the mesh is refined (middle) — and in one dimension it is even <em>nodally exact</em>, agreeing with $u$ at every node. Because each hat overlaps only its two neighbours, the stiffness matrix $K_{ij} = a(\phi_j, \phi_i)$ is <strong>tridiagonal</strong> (right, shown scaled by $h$ to expose the clean $-1, 2, -1$ pattern) — the sparsity that makes the finite element method fast is a direct consequence of choosing a locally-supported basis.</figcaption>
</figure>

## The broader landscape

The recipe above admits many variations, each obtained by varying one ingredient — the choice of subspace, or the relationship between trial and test spaces.

**The Ritz viewpoint, and a note on history.** When $a$ is symmetric positive definite, the variational problem (P) is equivalent to *minimizing the energy functional*

$$ J(v) = \tfrac{1}{2}\,a(v,v) - \ell(v) $$

over $V$ — its unique minimizer is exactly the solution $u$ (set the first variation to zero to recover (P)). From this angle, the Galerkin method is nothing but minimizing $J$ over the subspace $V_h$ instead of over $V$; this is the **Rayleigh–Ritz method**, and it predates Galerkin's. Galerkin's contribution (1915) was precisely to liberate the method from the need for an energy functional: the orthogonality condition ($\text{P}_h$) makes sense, and Céa's lemma holds, even when $a$ is non-symmetric and *no* minimization principle exists. The weak formulation is more fundamental than the variational one.

<figure>
  <img src="{{ '/assets/images/notes/galerkin_method/galerkin_ritz_energy.png' | relative_url }}" alt="Two panels. Left, a three-dimensional energy bowl over a plane of coefficients, with a red line marking the subspace V-h cutting across it; the global minimum of the bowl is marked u and the lowest point along the red line is marked u-h. Right, the same energy as elliptical contour lines: the straight line V-h is tangent to one contour ellipse exactly at the point u-h, and a segment joins the global minimum u to u-h." loading="lazy">
  <figcaption>Why minimising energy is the same as projecting. For symmetric positive-definite $a$, the solution $u$ is the unique minimiser of the energy $J(v) = \frac{1}{2}a(v,v) - \ell(v)$ — the bottom of the bowl (left). The Rayleigh–Ritz method minimises that same $J$ but only over the subspace $V_h$, a line through coefficient space, and the constrained minimiser is $u_h$. Geometrically the line is <strong>tangent</strong> to a level set of $J$ at $u_h$ (right), which is exactly the condition that $u - u_h$ be energy-orthogonal to $V_h$. So Ritz minimisation and Galerkin projection land on the same point — but only Galerkin's orthogonality condition still makes sense once $a$ is non-symmetric and there is no energy left to minimise.</figcaption>
</figure>

**Petrov–Galerkin and the inf-sup condition.** If we let the test space $W_h$ differ from the trial space $V_h$ — seeking $u_h \in V_h$ with $a(u_h, w_h) = \ell(w_h)$ for all $w_h \in W_h$ — we obtain the *Petrov–Galerkin* method. Galerkin orthogonality becomes $a(u - u_h, w_h) = 0$ for $w_h \in W_h$, and the symmetry that gave us coercivity on the diagonal is gone. The right replacement for coercivity is the **inf-sup (Babuška–Brezzi) condition**, which guarantees stability and a Céa-type bound in this more general setting. This framework is essential for problems where the naive symmetric choice misbehaves — convection-dominated transport, or saddle-point problems like the Stokes equations.

**Spectral methods.** If instead of local hat functions we take $V_h$ to be the span of *globally smooth* functions — trigonometric polynomials, or Legendre/Chebyshev polynomials — we obtain *spectral* and *Fourier–Galerkin* methods. The basis is no longer local, so the stiffness matrix is generally dense; but if the basis functions are chosen as eigenfunctions of the underlying operator, $K$ becomes *diagonal* and the method is simply the truncation of an eigenfunction expansion. For smooth solutions the approximation error in (3) decays faster than any power of $h$ — *spectral accuracy*. This is the same circle of ideas as the Karhunen–Loève expansion of a Gaussian field: diagonalize the covariance/elliptic operator, and the Galerkin projection onto the top eigenmodes is the optimal finite-rank truncation.

<figure>
  <img src="{{ '/assets/images/notes/galerkin_method/galerkin_convergence_rates.png' | relative_url }}" alt="Two panels. Left, a smooth wiggly target function with two approximations at the same dimension N equals six: a spectral truncation that tracks it almost perfectly, and a piecewise-linear hat-function approximation that visibly misses the curve between nodes. Right, a semilog plot of approximation error against the subspace dimension N: the hat-function error decays algebraically as a gently bending line, while the spectral error decays exponentially as a straight line plunging far below it." loading="lazy">
  <figcaption>The other half of Céa's lemma: the approximation power of the subspace. The error is the stability constant $M/\alpha$ times $\min_{v_h}\|u - v_h\|$, and that minimum depends entirely on the basis. Local hat functions give <strong>algebraic</strong> convergence, $\min_{v_h}\|u - v_h\| \sim h^k$ (orange), while a globally smooth spectral basis gives <strong>spectral accuracy</strong> on a smooth target — error decaying faster than <em>every</em> power of $h$ (green; a straight line on this semilog axis means exponential decay). Same projection principle, same stability constant: the dramatic difference in convergence comes purely from how well the chosen $V_h$ can approximate.</figcaption>
</figure>

## What is really going on

I will close with a few remarks on the larger picture, in the spirit of identifying the one idea that, once grasped, makes the rest inevitable.

**Remark 1 (it is all projection).** The deepest way to read the Galerkin method is as the infinite-dimensional incarnation of a piece of linear algebra everyone already knows: *to solve a linear system approximately within a subspace, project onto that subspace*. Once phrased this way, a striking number of apparently unrelated constructions reveal themselves as Galerkin methods in disguise. Least-squares regression is the Galerkin projection of data onto a span of features. Truncating a Fourier series is Galerkin projection onto low frequencies. Even *conditional expectation* — $\mathbb{E}[X \mid \mathcal{G}]$ — is, by its defining property $\mathbb{E}[(X - \mathbb{E}[X\mid\mathcal G])\,Z] = 0$ for all $\mathcal G$-measurable $Z$, precisely the orthogonal projection of $X$ onto the $L^2$ subspace of $\mathcal G$-measurable functions; that defining property *is* Galerkin orthogonality (G), written probabilistically. The method is less a trick for PDEs than a recurring shape in mathematics.

**Remark 2 (it converts analysis into linear algebra, cheaply).** The Galerkin framework takes a *qualitative analytic* question — does this PDE have a solution, and how regular is it? — and converts it into a *quantitative computational* one — solve $K\mathbf{c} = \mathbf{b}$. The bridge between the two, Céa's lemma, is almost trivial once Galerkin orthogonality is in hand; it cost us two lines. The genuine difficulty has not vanished but has been *relocated and partitioned*: into Lax–Milgram (well-posedness of the continuous problem) on one side, and approximation theory for the subspace on the other. Good mathematical methods rarely make difficulty disappear; the best ones move it somewhere we already know how to handle it.

**Remark 3 (a connection forward).** The same projection philosophy reaches well beyond classical PDEs. Discretizing a Fokker–Planck equation — equivalently, the Wasserstein gradient flow of an entropy functional — by a Galerkin scheme is the bridge between the continuous dynamics of diffusion and their computable approximations, and the stability constant $M/\alpha$ that governs Céa's lemma is the same ratio that controls the conditioning of the resulting linear systems. Whenever you find yourself replacing a function by finitely many coefficients and asking only that some residual be orthogonal to what you can resolve, you are doing Galerkin's method, whatever name the textbook gives it.