---
layout: post
title: "The Galerkin Method: How a One-Line Orthogonality Tames Infinite Dimensions"
date: 2026-06-14
tags: [numerical-analysis, functional-analysis, PDE, finite-elements, optimal-transport]
mathjax: true
---

A great many of the equations one actually wants to solve — elliptic PDEs, the stationary states of a diffusion, the Euler–Lagrange conditions of a variational problem — live most naturally not as pointwise identities but as *weak* (variational) statements: find $u$ in some function space $V$ such that

$$
a(u,v) = \ell(v) \qquad \text{for all } v \in V. \tag{1}
$$

Here $a$ is a bilinear form encoding the operator and $\ell$ a linear functional encoding the data. The trouble is that $V$ is infinite-dimensional, and a computer can only ever store finitely many numbers. The Galerkin method is the simplest possible response to this difficulty, and — this is the part worth internalizing — one of the most robust ideas in all of numerical analysis: **don't change the equation, change the space**. Restrict (1) to a finite-dimensional subspace $V_h \subset V$, test only against that subspace, and solve the resulting finite linear system.

What makes the method more than a heuristic is a single observation, *Galerkin orthogonality*, which says that the error of the computed solution is "invisible" to the subspace you projected onto. Combined with coercivity, this one line upgrades the naive recipe into a *quasi-optimal* method: the Galerkin solution is, up to a fixed constant, as accurate as the very best approximation the subspace can offer. Everything else in the theory — finite elements, spectral methods, Petrov–Galerkin, even the conjugate gradient iteration — is an elaboration of this fact.

Throughout, $V$ is a real Hilbert space with norm $\|\cdot\|$ and dual $V^\ast$; $a:V\times V\to\mathbb R$ is bilinear; $\ell\in V^\ast$. I write $\langle\cdot,\cdot\rangle$ for the duality pairing when needed and reserve $(\cdot,\cdot)$ for the inner product. The subscript $h$ on $V_h$ is the traditional mesh-size label; nothing forces $V_h$ to come from a mesh.

## 1. The continuous problem and Lax–Milgram

The variational problem (1) is well-posed under two hypotheses on $a$, both of which we fix once and for all:

- **Boundedness:** there is $C>0$ with $\lvert a(u,v)\rvert\le C\|u\|\,\|v\|$ for all $u,v\in V$.
- **Coercivity:** there is $\alpha>0$ with $a(v,v)\ge \alpha\|v\|^2$ for all $v\in V$.

**Lax–Milgram.** *Under these two hypotheses, for every $\ell\in V^\ast$ there is a unique $u\in V$ solving (1), and it obeys the a priori bound $\|u\|\le \|\ell\|_{V^\ast}/\alpha$.*

I will take this as known — it is the workhorse you already met with the Dirichlet Laplacian. The single sentence to carry forward is that **coercivity is what converts the abstract equation into a genuinely solvable, stable problem**, and the stability margin is measured by the ratio $C/\alpha$, a kind of condition number of the bilinear form.

**The model problem.** Keep one example in mind throughout. On a bounded domain $\Omega\subset\mathbb R^d$ consider the Poisson equation $-\Delta u = f$ with homogeneous Dirichlet data. Multiplying by a test function and integrating by parts gives the weak form (1) with

$$
V = H^1_0(\Omega),\qquad a(u,v)=\int_\Omega \nabla u\cdot\nabla v\,dx,\qquad \ell(v)=\int_\Omega f v\,dx.
$$

Boundedness is Cauchy–Schwarz; coercivity is the Poincaré inequality (the gradient seminorm controls the full $H^1$ norm on $H^1_0$). Here $a$ is symmetric, a special structure we will exploit in §5.

<figure>
  <img src="{{ '/assets/images/notes/math_topics/galerkin_method/gal_lax_milgram_shell.png' | relative_url }}" alt="Two polar plots. In each, a blue closed curve traces a(v,v) as the unit vector v sweeps the circle, trapped between an inner dashed green circle of radius alpha and an outer dashed red circle of radius C, the annulus between them lightly shaded. Left, a well-conditioned form: the curve is nearly circular and the shell is thin (C over alpha equals 2.4). Right, an ill-conditioned form: the curve is a pronounced peanut and the shell is thick (C over alpha equals 9.6)." loading="lazy">
  <figcaption>What Lax–Milgram's two hypotheses buy, drawn on the unit sphere. Coercivity forces $a(v,v)\ge\alpha\|v\|^2$ — a floor (green) — and boundedness caps it at $a(v,v)\le C\|v\|^2$ — a ceiling (red); the form's Rayleigh quotient is trapped in the <strong>energy shell</strong> between them. Its thickness is the ratio $C/\alpha$, a condition number of the form: a well-conditioned form (left) hugs a circle, an ill-conditioned one (right) bulges. Crucially the shell is a pointwise inequality on all of $V$, so it is <strong>inherited by every subspace</strong> $V_h\subset V$ with the <em>same</em> constants — which is why §2's discrete problem is stable for free and the margin $C/\alpha$ never grows with $N$.</figcaption>
</figure>

## 2. The Galerkin recipe

Pick a finite-dimensional subspace $V_h \subset V$, $\dim V_h = N$. The **Galerkin approximation** $u_h\in V_h$ is defined by imposing (1) only against the subspace:

$$
a(u_h, v_h) = \ell(v_h) \qquad \text{for all } v_h \in V_h. \tag{2}
$$

That is the entire method. Two remarks make it concrete.

**The discrete problem is well-posed for free.** Boundedness and coercivity are *inherited* by any subspace, since they are pointwise inequalities that hold on all of $V$ and hence on $V_h\subset V$ with the *same* constants $C,\alpha$. So Lax–Milgram applies verbatim to (2): $u_h$ exists, is unique, and satisfies $\|u_h\|\le\|\ell\|_{V^\ast}/\alpha$. This robustness — stability constants independent of $N$ — is exactly what later guarantees convergence as we refine. It is also exactly what *fails* in the Petrov–Galerkin setting of §8, which is why that setting is harder.

**The discrete problem is linear algebra.** Choose a basis $\lbrace \phi_1,\dots,\phi_N\rbrace$ of $V_h$ and expand $u_h=\sum_{j} c_j\phi_j$. Testing (2) against each basis function $\phi_i$ turns it into the square linear system

$$
A c = b, \qquad A_{ij} = a(\phi_j,\phi_i),\qquad b_i=\ell(\phi_i). \tag{3}
$$

The matrix $A$ — the *stiffness matrix* in the PDE tradition — is invertible precisely because of coercivity: for any $c\neq 0$, writing $w=\sum_j c_j\phi_j\in V_h\setminus\lbrace 0\rbrace$,

$$
c^{\mathsf T} A c = \sum_{i,j} c_i\,a(\phi_j,\phi_i)\,c_j = a(w,w)\ge \alpha\|w\|^2 > 0,
$$

so $A$ is positive definite. If in addition $a$ is symmetric, then $A$ is symmetric positive definite, and one can solve (3) by Cholesky or, iteratively, by conjugate gradients (a point we return to). The art of the method is now entirely in *choosing the basis* so that (i) $A$ is cheap to assemble and well-conditioned, and (ii) $V_h$ approximates the true solution well. These two desiderata pull in opposite directions and organize the whole subject.

## 3. Galerkin orthogonality: the one line that matters

Here is the observation that elevates the recipe to a method. The true solution $u$ satisfies $a(u,v_h)=\ell(v_h)$ for *every* $v_h\in V$, in particular for $v_h\in V_h$. The discrete solution satisfies $a(u_h,v_h)=\ell(v_h)$ for $v_h\in V_h$. Subtracting,

$$
\boxed{\,a(u-u_h,\,v_h)=0\qquad\text{for all } v_h\in V_h.\,} \tag{4}
$$

This is **Galerkin orthogonality**. In words: *the error $u-u_h$ is $a$-orthogonal to the entire trial space.* The method has, without being told to, made the error as "uncorrelated" with $V_h$ as the form $a$ allows. Geometrically, when $a$ is an inner product this says $u_h$ is the orthogonal projection of $u$ onto $V_h$ — but (4) holds even when $a$ is merely coercive and nonsymmetric, where "orthogonality" is with respect to a possibly skew pairing. Everything below is a consequence of squeezing this identity between the two inequalities $C$ and $\alpha$.

## 4. Céa's lemma: quasi-optimality

**Key idea.** Coercivity lets us start from the error in the energy of $a$; Galerkin orthogonality lets us swap $u_h$ for *any* element of $V_h$ we like; boundedness then closes the estimate. The freedom to choose that element is what produces a best-approximation bound.

**Céa's Lemma.** *Under boundedness and coercivity, the Galerkin solution is quasi-optimal:*

$$
\|u-u_h\|\;\le\;\frac{C}{\alpha}\,\inf_{v_h\in V_h}\|u-v_h\|. \tag{5}
$$

*Proof.* Fix any $v_h\in V_h$. By coercivity and then by writing $u-u_h=(u-v_h)+(v_h-u_h)$ inside $a$,

$$
\alpha\|u-u_h\|^2 \le a(u-u_h,\,u-u_h) = a(u-u_h,\,u-v_h) + a(u-u_h,\,v_h-u_h).
$$

The second term vanishes: $v_h-u_h\in V_h$, so Galerkin orthogonality (4) kills it. Bounding the first term with boundedness,

$$
\alpha\|u-u_h\|^2 \le a(u-u_h,\,u-v_h)\le C\,\|u-u_h\|\,\|u-v_h\|.
$$

Divide by $\|u-u_h\|$ (trivial if it is zero) to get $\|u-u_h\|\le (C/\alpha)\|u-v_h\|$, and take the infimum over $v_h\in V_h$. $\qquad\blacksquare$

The conceptual payoff is large and worth stating plainly. **Céa's lemma decouples the analysis into stability and approximation.** The constant $C/\alpha$ is a property of the *form* alone — the numerical analyst pays for it once and it never grows with $N$. The infimum on the right is pure *approximation theory*: how well can the chosen subspace represent the (unknown but, with luck, regular) solution $u$? Convergence of the Galerkin method is therefore reduced to the question "is the family $\lbrace V_h\rbrace$ dense enough in $V$?" — to which the answer is supplied by interpolation estimates, with no further reference to the equation.

**Equality and sharpness.** Inequality (5) is generally not tight; the constant $C/\alpha$ is a worst case over the geometry that $a$ can impose between the error and the residual directions. In the symmetric case it can be improved to $\sqrt{C/\alpha}$ (next section), and that square-root constant *is* attained in the limiting geometry. When $a$ is symmetric *and* $V_h$ happens to be spanned by eigenvectors of the associated operator, even the infimum is achieved exactly by truncation, and the inequality becomes the Pythagorean identity below.

<figure>
  <img src="{{ '/assets/images/notes/math_topics/galerkin_method/gal_cea_worst_case.png' | relative_url }}" alt="Left, a vector diagram: the true solution u sits above a horizontal line V-h; its Euclidean-best foot Pu lies directly below, but the Galerkin solution u-h lands much further along the line, reached by an oblique slanted projection, so the dotted error from u-h to u is far longer than the vertical error from Pu to u. Right, a graph: the worst-case error ratio rises with the skewness s of the form, the measured blue points lying exactly on the red curve C-over-alpha equals square root of one plus s squared." loading="lazy">
  <figcaption>Céa's constant $C/\alpha$ is not merely an upper bound — it is <strong>attained</strong>. For the skew coercive family with matrix $A=\begin{pmatrix}1&s\\-s&1\end{pmatrix}$ (so $\alpha=1$, $C=\sqrt{1+s^2}$), testing against $V_h$ projects $u$ <strong>obliquely</strong>: the Galerkin solution $u_h$ overshoots the Euclidean-best approximation $Pu$ along the subspace (left). Sweeping the skewness $s$, the realised worst-case ratio $\|u-u_h\|/\inf_v\|u-v\|$ (blue) sits exactly on the bound $C/\alpha=\sqrt{1+s^2}$ (red): the more skew the form, the more the projection tilts and the error inflates by precisely the stability constant — the sharpness promised just above.</figcaption>
</figure>

## 5. The symmetric (Ritz) case: projection and energy minimization

Suppose now $a$ is symmetric in addition to bounded and coercive. Then $a$ *is* an inner product on $V$, and

$$
\|v\|_a := \sqrt{a(v,v)}
$$

is a norm — the **energy norm** — equivalent to $\|\cdot\|$ via $\alpha\|v\|^2\le\|v\|_a^2\le C\|v\|^2$. In this setting two extra structures appear, and they coincide.

**Galerkin = orthogonal projection.** Galerkin orthogonality (4) now literally says $u-u_h\perp_a V_h$. Hence $u_h$ is the $a$-orthogonal projection of $u$ onto $V_h$, and is the **best approximation in the energy norm**. The proof is one application of Pythagoras: for any $v_h\in V_h$, since $u-u_h\perp_a (u_h-v_h)$ and $u_h-v_h\in V_h$,

$$
\|u-v_h\|_a^2 = \|u-u_h\|_a^2 + \|u_h-v_h\|_a^2 \;\ge\; \|u-u_h\|_a^2,
$$

so $\|u-u_h\|\_a=\inf_{v_h\in V_h}\|u-v_h\|\_a$, with equality iff $v_h=u_h$.

**Galerkin = Ritz (energy minimization).** Define the energy functional $J(v)=\tfrac12 a(v,v)-\ell(v)$. Its unique minimizer over a subspace is characterized by the vanishing of its directional derivatives, which is exactly (2). So the Galerkin solution $u_h$ *minimizes the energy over $V_h$*:

$$
u_h = \operatorname*{arg\,min}_{v_h\in V_h} \Big(\tfrac12 a(v_h,v_h)-\ell(v_h)\Big).
$$

This is the **Rayleigh–Ritz method**, and the identification "test against the subspace = minimize over the subspace" is special to the symmetric case. It is the discrete shadow of the fact that symmetric elliptic problems are gradient flows / steepest-descent of a convex energy — the same structural fact that, in the Wasserstein metric, turns the heat equation into the gradient flow of entropy. (One can run the Galerkin philosophy over the JKO scheme too: discretize the manifold of measures by a finite-dimensional family and minimize the De Giorgi functional within it.)

**Improved constant.** Translating the energy-norm optimality back to $\|\cdot\|$ sharpens Céa. For any $v_h$,

$$
\alpha\|u-u_h\|^2 \le \|u-u_h\|_a^2 \le \|u-v_h\|_a^2 \le C\|u-v_h\|^2,
$$

hence

$$
\|u-u_h\|\le \sqrt{\tfrac{C}{\alpha}}\,\inf_{v_h\in V_h}\|u-v_h\|. \tag{6}
$$

The constant has dropped from $C/\alpha$ to its square root — a real gain when the form is poorly conditioned.

<figure>
  <img src="{{ '/assets/images/notes/math_topics/galerkin_method/gal_symmetric_sqrt_gain.png' | relative_url }}" alt="Two graphs against the form condition number C-over-alpha. Left, two rising curves: a straight red line y equals C-over-alpha and a lower blue square-root curve y equals square root of C-over-alpha, the widening gap between them shaded and labelled energy-norm optimality gain. Right, measured worst-case ratios: red squares for the nonsymmetric case lie on the straight line, blue dots for the symmetric case stay on or below the square-root curve." loading="lazy">
  <figcaption>The payoff of symmetry, quantified. When $a$ is symmetric it is a genuine inner product, Galerkin becomes the <strong>energy-orthogonal projection</strong>, and the quasi-optimality constant sharpens from $C/\alpha$ to $\sqrt{C/\alpha}$ (left) — a gain that widens as the form grows ill-conditioned. The right panel confirms it numerically: the oblique (nonsymmetric) projection of the previous figure rides the $C/\alpha$ line, while the symmetric projection's realised error stays at or below $\sqrt{C/\alpha}$. The square root is exactly the improvement extracted by passing through the energy norm in inequality (6).</figcaption>
</figure>

## 6. A concrete instance: linear finite elements in 1D

Abstractions earn their keep on an example. Take the model problem on $\Omega=(0,1)$: $-u^{''}=f$, $u(0)=u(1)=0$, so $a(u,v)=\int_0^1 u'v'\,dx$. Lay down a uniform mesh $x_i=ih$, $h=1/N$, $i=0,\dots,N$, and let $V_h\subset H^1\_0$ be the continuous, piecewise-linear functions vanishing at the endpoints. The natural basis is the **hat functions** $\phi_i$ ($i=1,\dots,N-1$), with $\phi_i(x_j)=\delta\_{ij}$, each supported on the two intervals adjoining $x_i$.

Because $\phi_i'$ is $\pm 1/h$ on those intervals and zero elsewhere, the stiffness entries are immediate: $a(\phi_i,\phi_i)=2/h$, $a(\phi_i,\phi_{i\pm1})=-1/h$, and $0$ otherwise. Thus

$$
A=\frac1h\begin{pmatrix}2&-1&&\\-1&2&-1&\\&\ddots&\ddots&\ddots\\&&-1&2\end{pmatrix},
$$

the celebrated tridiagonal matrix — *the same* (up to the $1/h$ scaling) one obtains from the second-order finite-difference Laplacian. Two lessons hide in this coincidence. First, **local basis functions produce sparse stiffness matrices**: $a(\phi_i,\phi_j)=0$ unless the supports overlap, which is the entire computational reason finite elements scale. Second, the finite element and finite difference methods, arrived at from opposite philosophies (weak projection vs. pointwise Taylor expansion), here land on the same algebra — a recurring theme.

The price of locality is conditioning: $\kappa(A)\sim h^{-2}$ as $h\to 0$, so the very refinement that improves the approximation infimum in (6) degrades the linear-algebra stability of (3). This tension between approximation accuracy and matrix conditioning is the perennial design constraint of the method.

**What Céa buys here.** Standard interpolation theory gives $\inf_{v_h}\|u-v_h\|\_{H^1}\le C\,h\,\lvert u\rvert\_{H^2}$ for $u\in H^2$. Feeding this into (6) yields the first-order energy-norm convergence $\|u-u_h\|\_{H^1}=O(h)$ — proved without ever touching the discrete equation again, exactly as advertised: stability $\times$ approximation.

<figure>
  <img src="{{ '/assets/images/notes/math_topics/galerkin_method/gal_fem_conditioning.png' | relative_url }}" alt="A log-log plot with two trends crossing. A descending blue line with circular markers, slope minus one, is the energy-norm error falling like h as the number of elements N grows; an ascending red line with square markers, slope plus two, is the stiffness condition number rising like h to the minus two. A small inset shows the stiffness matrix sparsity pattern: nonzeros only on the main diagonal and its two neighbours." loading="lazy">
  <figcaption>The central design tension of the finite element method, on the 1D model problem $-u''=f$. Refining the mesh (larger $N=1/h$) drives the energy-norm error down like $h$ (blue, slope $-1$) — exactly the $O(h)$ rate Céa plus interpolation predicts — but simultaneously inflates the stiffness matrix's condition number like $h^{-2}$ (red, slope $+2$), so the system $Ac=b$ grows harder to solve at the very rate the approximation improves. The inset is the source of the tradeoff in both directions: hat functions overlap only their neighbours, so $A$ is <strong>tridiagonal</strong> (cheap, sparse) — but that same locality is what makes $\kappa(A)$ blow up.</figcaption>
</figure>

## 7. Spectral Galerkin and the link to Karhunen–Loève

At the opposite extreme from local hat functions sit **global** bases. Suppose the symmetric form $a$ is associated to a self-adjoint operator with an orthonormal eigenbasis $\lbrace e_k\rbrace$, $a(e_k,e_j)=\lambda_k\delta_{kj}$ (for the 1D model problem these are the sines $\sqrt2\sin(k\pi x)$ with $\lambda_k=k^2\pi^2$). Choosing $V_h=\operatorname{span}\lbrace e_1,\dots,e_N\rbrace$ *diagonalizes* the stiffness matrix: $A=\operatorname{diag}(\lambda_1,\dots,\lambda_N)$, and the Galerkin solution is nothing but the **truncated eigenexpansion** of $u$,

$$
u_h=\sum_{k=1}^N \frac{\ell(e_k)}{\lambda_k}\,e_k,
$$

with each coefficient computed independently. Here the best-approximation infimum in (6) is *attained exactly* — Galerkin = truncation — and convergence is as fast as the eigencoefficients of $u$ decay: spectral (faster than any polynomial in $1/N$) when $u$ is smooth.

This is the same machinery as the **Karhunen–Loève expansion**: there one diagonalizes a covariance operator and truncates; here one diagonalizes the energy form and truncates. In both cases the eigenbasis is optimal precisely because it makes the relevant quadratic form diagonal, so that "project onto the first $N$ modes" and "keep the $N$ best modes" become the same instruction. The trade-off versus finite elements is the mirror image of §6: a perfectly conditioned, dense (non-sparse) system, global basis functions, and a heavy regularity demand on the geometry of $\Omega$ for the eigenbasis to be usable.

<figure>
  <img src="{{ '/assets/images/notes/math_topics/galerkin_method/gal_spectral_vs_fem.png' | relative_url }}" alt="Left, a semi-log plot of relative energy-norm error versus the number of modes or elements N: the purple spectral-Galerkin curve falls as a straight descending line (exponential decay) for a smooth target, reaching ten to the minus eight by N equals forty, while the blue finite-element curve flattens out, decaying only algebraically like one over N. Right, two sparsity patterns: the spectral stiffness matrix is purely diagonal, the finite-element one tridiagonal." loading="lazy">
  <figcaption>The mirror image of the previous figure. Choosing $V_h$ to be the span of the first $N$ eigenfunctions <strong>diagonalises</strong> the energy form, so Galerkin becomes truncation of the eigenexpansion and the best-approximation infimum is hit exactly. For a smooth target the eigencoefficients decay faster than any power, so the spectral error falls <strong>exponentially</strong> (purple, a straight line on this semi-log axis), leaving local FEM's algebraic $O(h)$ rate (blue) far behind. The price is on the right: the spectral system is perfectly conditioned but its global basis gives a <strong>dense</strong> (here diagonal) matrix, against FEM's sparse-but-ill-conditioned tridiagonal. This is the same diagonalise-and-truncate move as the <strong>Karhunen–Loève</strong> expansion of a covariance operator.</figcaption>
</figure>

## 8. Petrov–Galerkin and the inf–sup condition

Coercivity is a luxury. For many problems — convection-dominated transport, mixed formulations, saddle-point systems — the natural bilinear form is not coercive, and one is forced to **test against a different space than one trial-solves in**. This is the *Petrov–Galerkin* method: choose a trial space $V_h$ and a (possibly different) test space $W_h$ with $\dim V_h=\dim W_h$, and seek $u_h\in V_h$ with

$$
a(u_h,w_h)=\ell(w_h)\qquad\text{for all } w_h\in W_h.
$$

Now well-posedness is *not* automatic, because the comfortable inheritance of §2 breaks. The right replacement for coercivity is the **discrete inf–sup (Ladyzhenskaya–Babuška–Brezzi) condition**: there exists $\beta_h>0$ with

$$
\inf_{0\neq u_h\in V_h}\ \sup_{0\neq w_h\in W_h}\ \frac{a(u_h,w_h)}{\|u_h\|\,\|w_h\|}\ \ge\ \beta_h. \tag{7}
$$

The subtlety — and the source of much grief in the design of mixed methods — is that (7) is **not** inherited from the continuous inf–sup condition; a pair of spaces can each be perfectly reasonable yet violate (7) jointly, producing spurious modes. One must *design* compatible pairs (the discrete LBB condition is the precise statement of what "compatible" means).

When (7) does hold, Babuška's theorem gives quasi-optimality of exactly the Céa shape, with $\beta_h$ playing the role of $\alpha$:

$$
\|u-u_h\|\le \frac{C}{\beta_h}\,\inf_{v_h\in V_h}\|u-v_h\|. \tag{8}
$$

(The clean constant $C/\beta_h$ — rather than the older $1+C/\beta_h$ — is the sharp Xu–Zikatanov refinement; for symmetric coercive $a$ one has $\beta_h\ge\alpha$ and recovers Céa.) The moral is that **the analysis is unchanged in form; only stability is now something you must earn rather than inherit**. Galerkin orthogonality (4) survives intact — it never used coercivity — so it is again the load-bearing identity; what changes is how the orthogonality is converted into an estimate.

<figure>
  <img src="{{ '/assets/images/notes/math_topics/galerkin_method/gal_inf_sup.png' | relative_url }}" alt="Left, a vector diagram: a black trial direction phi, an orange vector M-phi (the form's image of phi), a green good test direction aligned with M-phi, and a red bad test direction perpendicular to M-phi. Right, a graph of the inf-sup constant beta-h, equal to the absolute cosine of the angle between the test direction and M-phi, against the test-space orientation: it peaks at one when the test aligns with M-phi and plunges to zero, into a shaded danger zone of spurious modes, when the test is perpendicular." loading="lazy">
  <figcaption>Why Petrov–Galerkin stability must be <em>earned</em>. Without coercivity, well-posedness rests on the discrete inf–sup constant $\beta_h=\inf_{u_h}\sup_{w_h} a(u_h,w_h)/(\|u_h\|\,\|w_h\|)$ — which, for one-dimensional trial and test lines, is exactly $\lvert\cos\angle(\psi,\,M\phi)\rvert$: the test space must <strong>see</strong> the form's image $M\phi$ of the trial direction. A well-aligned test space (green) keeps $\beta_h$ bounded below, and Babuška's theorem then delivers quasi-optimality $\|u-u_h\|\le (C/\beta_h)\inf_{v_h}\|u-v_h\|$; a test space nearly orthogonal to $M\phi$ (red) collapses $\beta_h\to 0$ and admits a <strong>spurious mode</strong>. Unlike coercivity, this is <em>not</em> inherited from the continuous problem — compatible trial/test pairs must be designed, which is the whole difficulty of mixed methods.</figcaption>
</figure>

## 9. What is really going on, and where it connects

**It is projection, made stable.** Strip away the analysis and the Galerkin method is the assertion that a good way to approximate the solution of an operator equation is to project the equation — not the solution — onto a subspace. Galerkin orthogonality says the residual is annihilated by the test space; coercivity (or inf–sup) says that controlling the residual controls the error. Quasi-optimality (5)/(6)/(8) is the precise sense in which "projecting the equation" and "approximating the solution" agree up to a stability constant. Methods that fail to be quasi-optimal — for which the constant blows up as $N\to\infty$ — are exactly those for which this agreement breaks, and the discrete inf–sup condition is the diagnostic.

**Krylov subspace solvers are Galerkin in disguise.** The same projection principle reappears one level down, in how one *solves* the linear system (3). The conjugate gradient method is precisely the Galerkin method applied to the sequence of Krylov subspaces $\mathcal K_m=\operatorname{span}\lbrace r_0,Ar_0,\dots,A^{m-1}r_0\rbrace$ with the energy inner product induced by the SPD matrix $A$: at each step it imposes Galerkin orthogonality of the residual against $\mathcal K_m$, which is why its iterates are energy-optimal over those subspaces. GMRES is the Petrov–Galerkin/minimal-residual variant for nonsymmetric $A$. So the method governs both the discretization of the PDE and the iterative inversion of the matrix it produces — Galerkin all the way down.

**It is the engine of model reduction.** Finally, the method is agnostic about where $V_h$ comes from. In reduced-order modeling one takes $V_h$ to be the span of a handful of *data-driven* basis vectors (POD/PCA modes of a snapshot set — again a Karhunen–Loève truncation) and runs Galerkin on those. The same quasi-optimality estimate (5) then certifies that the reduced model is as accurate as the reduced basis can possibly be. This is the bridge from classical numerical PDE to the spectral and low-rank methods that recur throughout data science: choose a good $N$-dimensional space, project the dynamics onto it, and let Céa's lemma promise you lost nothing essential beyond a stability constant.

<figure>
  <img src="{{ '/assets/images/notes/math_topics/galerkin_method/gal_krylov_galerkin.png' | relative_url }}" alt="Two semi-log convergence plots. Left, conjugate gradients on a symmetric positive-definite stiffness matrix: the green energy-norm error descends steadily beneath a dashed Chebyshev worst-case bound. Right, GMRES on a generic nonsymmetric system: the orange relative residual falls in a straight geometric line over about twenty-five iterations down to ten to the minus twelve." loading="lazy">
  <figcaption>The same projection principle, one level down — in how the linear system is <em>solved</em>. Conjugate gradients (left) is precisely Galerkin applied to the Krylov subspaces $\mathcal K_m=\operatorname{span}\lbrace r_0,Ar_0,\dots,A^{m-1}r_0\rbrace$ in the energy inner product of the SPD matrix $A$: each step imposes Galerkin orthogonality of the residual against $\mathcal K_m$, so the iterate is <strong>energy-optimal</strong> over that subspace — bounded by the classical Chebyshev/$\sqrt{\kappa}$ estimate (dashed). GMRES (right) is the <strong>Petrov–Galerkin</strong>, minimal-residual variant for nonsymmetric $A$. Discretising the PDE and inverting its matrix are thus the same idea twice over — Galerkin all the way down.</figcaption>
</figure>

---

*Two closing remarks.* (i) The method is sometimes presented as one instance of the *method of weighted residuals*: enforce that the residual $a(u_h,\cdot)-\ell$, viewed as a functional, is orthogonal to a chosen space of weights. Collocation (weights = Dirac masses) and least-squares (weights = the residual's own Riesz representative) are sibling choices; "Galerkin" is the choice weights $=$ trial space, and it is precisely this self-referential choice that makes the orthogonality (4) hold and quasi-optimality follow. (ii) The hypotheses split cleanly by role: boundedness and coercivity are about the *form*, the choice of $V_h$ is about *approximation*, and the inf–sup condition (when needed) is about the *compatibility of trial and test spaces*. Keeping these three concerns separate is the single most useful organizing principle when reading the finite-element literature.