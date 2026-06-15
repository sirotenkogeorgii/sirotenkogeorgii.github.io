---
layout: post
title: "The Weak Formulation: How to Solve a PDE by Lowering Your Standards"
date: 2026-06-14
categories: [analysis, pde, functional-analysis]
---

There is a recurring pattern in analysis: when a problem refuses to have a solution of the kind you want, you enlarge the universe of admissible solutions until one appears, and only afterwards do you investigate how good it really is. The weak (or *variational*) formulation of an elliptic PDE is the canonical instance of this move. We will see that it is built entirely out of a single, almost embarrassingly elementary operation — integration by parts — and that this one operation simultaneously (i) relaxes the regularity demanded of a solution, (ii) recasts the PDE as an equation between *bounded linear functionals* on a Hilbert space, where the full machinery of functional analysis becomes available, (iii) exposes the variational/energetic structure hiding inside the equation, and (iv) hands us, for free, the foundational object of the finite element method. My aim here is to make all four of these visible from the same vantage point.

## Conventions

Throughout, $\Omega \subset \mathbb{R}^d$ is a bounded open set; from the point at which boundary terms enter we additionally assume $\partial\Omega$ is Lipschitz, so that the trace and divergence theorems hold and an outward unit normal $n$ is defined $\mathcal{H}^{d-1}$-a.e. All function spaces are real. We write $(u,v) = \int_\Omega uv$ for the $L^2(\Omega)$ inner product and $\langle\cdot,\cdot\rangle$ for a duality pairing. I take the Laplacian with the *analyst's sign*, so that $-\Delta = -\sum_{i} \partial_{ii}$ is the positive operator: this is the sign that makes the energy below convex, and one should regard the minus sign as the price of having a sensible variational principle rather than as a nuisance. The space of test functions is $C_c^\infty(\Omega)$; the Sobolev space $H^1(\Omega) = \lbrace u \in L^2 : \nabla u \in L^2\rbrace$ carries the norm $\|u\|\_{H^1}^2 = \|u\|\_{L^2}^2 + \|\nabla u\|\_{L^2}^2$, and $H_0^1(\Omega)$ is the closure of $C_c^\infty(\Omega)$ in this norm — the functions that "vanish on the boundary" in the trace sense. Its dual is denoted $H^{-1}(\Omega)$.

## The one idea

Here is the whole story in one line. To probe the equation $-\Delta u = f$, multiply both sides by a test function $v$, integrate over $\Omega$, and move one derivative off $u$ and onto $v$ using Green's identity:

$$
\int_\Omega (-\Delta u)\, v \;=\; \int_\Omega \nabla u \cdot \nabla v \;-\; \int_{\partial\Omega} (\partial_n u)\, v \, dS.
$$

Everything downstream is bookkeeping on this identity. The left side wanted *two* derivatives of $u$; the right side is happy with *one* derivative of $u$ and one of $v$. That asymmetry — we paid for the privilege by demanding a derivative of $v$, but $v$ is a test function we get to choose, so we do not care — is the entire mechanism by which the weak formulation lowers the regularity bar. The boundary integral is the second protagonist: depending on how we choose the test space, it either disappears or *becomes* the boundary condition. Hold onto it.

## The model problem and its weak form

Take the homogeneous Dirichlet problem for the Poisson equation,

$$
-\Delta u = f \ \text{ in } \Omega, \qquad u = 0 \ \text{ on } \partial\Omega. \tag{$\ast$}
$$

**Choose the test space to kill the boundary term.** If we insist that test functions vanish on $\partial\Omega$ — i.e. take $v \in H_0^1(\Omega)$ — the boundary integral in Green's identity drops out, and a (sufficiently smooth) solution of $(\ast)$ satisfies

$$
\int_\Omega \nabla u \cdot \nabla v \;=\; \int_\Omega f\, v \qquad \text{for all } v \in H_0^1(\Omega). \tag{W}
$$

**Promote this consequence to a definition.** A function $u \in H_0^1(\Omega)$ is a *weak solution* of $(\ast)$ if it satisfies (W). Notice what has happened: the requirement $u \in C^2$ that we needed even to *write down* $-\Delta u$ has evaporated. To make sense of (W) we need only that $\nabla u$ be square-integrable, and the homogeneous Dirichlet condition is no longer a side constraint to be checked but is *baked into the space* $H_0^1$ in which we hunt for $u$.

It is worth packaging (W) abstractly, because the abstraction is what unlocks the existence theory. Define the bilinear form and the linear functional

$$
a(u,v) = \int_\Omega \nabla u \cdot \nabla v, \qquad \ell(v) = \int_\Omega f\, v,
$$

so that the weak problem reads: **find $u \in H_0^1$ such that $a(u,v) = \ell(v)$ for all $v \in H_0^1$.** Equivalently, the form $a$ induces a bounded operator $A : H_0^1 \to H^{-1}$ via $\langle Au, v\rangle = a(u,v)$, and we are asking to solve $Au = \ell$ in the dual space. The PDE has become a *linear equation in a Hilbert space*. This is the payoff of the whole construction, and the next section collects on it.

**Why $H_0^1$ and not just $C_c^\infty$?** One could phrase (W) using $v \in C_c^\infty(\Omega)$ and an unspecified $u$; this is the *distributional* formulation and it is correct but inert, because $C_c^\infty$ is not complete and supports no fixed-point or projection theorem. Completing in the $H^1$-norm to get $H_0^1$ costs nothing — the density of $C_c^\infty$ means (W) for all $v \in C_c^\infty$ is the same as (W) for all $v \in H_0^1$ — and buys us a *Hilbert space*, which is exactly the setting the existence theorems demand. The methodological lesson, which recurs everywhere in analysis, is that the natural space for a problem is whatever space makes the relevant operators bounded and the relevant sequences convergent; here that space is the *energy space* $H_0^1$, and the weak formulation is precisely the reformulation of the PDE that lives natively in it.

## Existence and uniqueness: Lax–Milgram

The abstract problem $a(u,v) = \ell(v)$ is solved, once and for all, by a theorem whose only ingredients are completeness and a coercivity estimate.

> **Lax–Milgram.** Let $H$ be a real Hilbert space, $a : H \times H \to \mathbb{R}$ a bilinear form that is *bounded*, $\lvert a(u,v)\rvert \le M\|u\|\,\|v\|$, and *coercive*, $a(u,u) \ge \alpha\|u\|^2$ with $\alpha > 0$, and let $\ell \in H^\ast$. Then there is a unique $u \in H$ with $a(u,v) = \ell(v)$ for all $v \in H$, and moreover $\|u\| \le \alpha^{-1}\|\ell\|_{H^\ast}$.

The key driving idea of the proof is that coercivity makes the operator $A$ from above *bounded below*, hence injective with closed range; boundedness-below of the adjoint then kills the orthogonal complement of the range, so $A$ is onto. (When $a$ is symmetric one need not even invoke this: coercivity makes $a$ itself an equivalent inner product on $H$, and $u$ is just the Riesz representative of $\ell$ in that inner product. The general case is a perturbation of this observation.) I will not rehearse the proof — it is standard — but I do want to verify its hypotheses for our PDE, because *that* verification is where all the real analysis lives.

**Boundedness of $a$ is Cauchy–Schwarz.** Immediately,

$$
|a(u,v)| = \Big| \int_\Omega \nabla u \cdot \nabla v\Big| \le \|\nabla u\|_{L^2}\|\nabla v\|_{L^2} \le \|u\|_{H^1}\|v\|_{H^1},
$$

so $M = 1$.

**Coercivity is the Poincaré inequality.** This is the one nontrivial estimate, and it is exactly the point where the boundedness of $\Omega$ and the homogeneous boundary condition do their work. Recall the Poincaré inequality: there is a constant $C_P = C_P(\Omega) > 0$ with

$$
\|u\|_{L^2} \le C_P \|\nabla u\|_{L^2} \qquad \text{for all } u \in H_0^1(\Omega). \tag{P}
$$

The mechanism behind (P) is that a function pinned to zero on the boundary cannot be large in the interior without having a correspondingly large gradient to climb up from the boundary value — boundedness of the domain caps how far it can climb. Granting (P),

$$
a(u,u) = \|\nabla u\|_{L^2}^2 = \tfrac12\|\nabla u\|_{L^2}^2 + \tfrac12\|\nabla u\|_{L^2}^2 \ge \tfrac12\|\nabla u\|_{L^2}^2 + \tfrac{1}{2C_P^2}\|u\|_{L^2}^2 \ge \alpha \|u\|_{H^1}^2,
$$

with $\alpha = \tfrac12\min(1, C_P^{-2})$; the cleaner bookkeeping $\|u\|_{H^1}^2 \le (1+C_P^2)\|\nabla u\|^2$ gives the sharper $\alpha = (1+C_P^2)^{-1}$. Either way, **coercivity holds and its constant is the inverse Poincaré constant.** Note that on the full space $H^1$ this fails — constants have zero gradient but nonzero norm — which is a first hint that the *choice of space encodes the boundary condition* and is not a mere technicality.

**Continuity of $\ell$ is duality.** If $f \in L^2(\Omega)$, then $\lvert\ell(v)\rvert = \lvert\int f v\rvert \le \|f\|\_{L^2}\|v\|\_{L^2} \le C_P\|f\|\_{L^2}\|v\|\_{H^1}$, so $\ell \in H^{-1}$. In fact the *natural* hypothesis is the weaker $f \in H^{-1}(\Omega)$, and (W) makes perfect sense then with $\ell(v) = \langle f, v\rangle$; this is one of the quiet generalizations the weak formulation buys us, since the classical equation has no meaning for such rough $f$.

All three hypotheses verified, **Lax–Milgram delivers a unique weak solution $u \in H_0^1$ of $(\ast)$, depending continuously on the data:** $\|u\|\_{H^1} \le \alpha^{-1}\|f\|\_{H^{-1}}$. This last estimate is *well-posedness in the sense of Hadamard* — existence, uniqueness, and stability — and it dropped out of the same coercivity that gave existence. That is not a coincidence; it is the reason coercive problems are the well-behaved ones.

## The hierarchy of solution concepts

It pays to be precise about how weak solutions sit relative to the classical ones, since the entire enterprise is a bet that we can pass between them.

**Classical $\Rightarrow$ weak.** Any $u \in C^2(\Omega) \cap C(\overline\Omega)$ solving $(\ast)$ pointwise is a weak solution: the derivation of (W) at the top *was* this implication.

**Weak + regularity $\Rightarrow$ classical.** Conversely, suppose a weak solution $u$ happens to lie in $C^2$. Take $v \in C_c^\infty(\Omega)$ in (W) and integrate by parts in the *reverse* direction — now legitimate, since $u$ is twice differentiable and $v$ has compact support so no boundary term appears:

$$
\int_\Omega f v = \int_\Omega \nabla u \cdot \nabla v = -\int_\Omega (\Delta u)\, v, \quad\text{hence}\quad \int_\Omega (-\Delta u - f)\, v = 0 \ \ \forall v \in C_c^\infty.
$$

By the *fundamental lemma of the calculus of variations* (if $g \in L^1_{\mathrm{loc}}$ and $\int g v = 0$ for all $v \in C_c^\infty$, then $g = 0$ a.e.), we recover $-\Delta u = f$ pointwise, and membership in $H_0^1 \cap C(\overline\Omega)$ recovers $u\mid\_{\partial\Omega} = 0$. So the weak solution is the classical one *whenever it is smooth enough to be classical*. The weak formulation has not changed the answer; it has only widened the search.

The genuinely interesting question is the missing arrow — when does a weak solution acquire the regularity needed to be classical? — and I return to it below under elliptic regularity.

## What the symmetry buys: Dirichlet's principle

Because the form $a(u,v) = \int \nabla u \cdot \nabla v$ is symmetric, the weak problem is the stationarity condition of an energy. Define

$$
J(v) = \tfrac12 a(v,v) - \ell(v) = \tfrac12 \int_\Omega |\nabla v|^2 - \int_\Omega f v \qquad (v \in H_0^1).
$$

**The weak solution minimizes $J$, and conversely.** For any $u, w \in H_0^1$ and $t \in \mathbb{R}$, expand using bilinearity and symmetry:

$$
J(u + t w) = J(u) + t\big(a(u,w) - \ell(w)\big) + \tfrac{t^2}{2}\,a(w,w).
$$

If $u$ solves (W) the linear term vanishes for every $w$, and since $a(w,w) = \|\nabla w\|^2 \ge 0$ the quadratic term is nonnegative, so $J(u+tw) \ge J(u)$: $u$ is the global minimizer (strict, by coercivity, so it is unique — recovering uniqueness a second way). Conversely, at a minimizer the first variation $\frac{d}{dt}\mid\_{t=0} J(u+tw) = a(u,w) - \ell(w)$ must vanish for all $w$, which is precisely (W). So **(W) is the Euler–Lagrange equation of $J$, written in its natural integrated form.** This equivalence — *Dirichlet's principle* — is more than an aesthetic remark. Historically Riemann tried to *define* the solution as the minimizer, and Weierstrass's objection (a bounded-below functional need not attain its infimum) is exactly the gap that coercivity-plus-completeness, i.e. the $H_0^1$ setting plus Lax–Milgram, finally closes. The variational viewpoint is also the conceptually primary one: many PDEs of mathematical physics *are* Euler–Lagrange equations of a physically meaningful energy, and their weak form is just the assertion that the energy's first variation vanishes against all admissible perturbations.

## How boundary conditions really enter: essential vs. natural

Return to the boundary integral we set aside. Its fate under different boundary conditions reveals one of the most useful organizing principles in the subject.

**Dirichlet conditions are *essential*: they live in the space.** For $(\ast)$ we forced $v \in H_0^1$ to annihilate the boundary term, and we sought $u \in H_0^1$ so that $u$ itself carries the condition $u\mid\_{\partial\Omega}=0$. The boundary condition never appears in the equation (W); it is enforced by the *choice of trial and test spaces*. For inhomogeneous Dirichlet data $u = g_D$ on $\partial\Omega$, one writes $u = \tilde u + G$ where $G$ is any $H^1$-extension of $g_D$, and solves (W) for $\tilde u \in H_0^1$ with the modified functional $\ell(v) - a(G,v)$; the shift is cosmetic.

**Neumann conditions are *natural*: they emerge from the form.** Now consider

$$
-\Delta u = f \ \text{ in } \Omega, \qquad \partial_n u = g \ \text{ on } \partial\Omega,
$$

and this time test against the *full* space $v \in H^1(\Omega)$, retaining the boundary term in Green's identity. Substituting the desired $\partial_n u = g$:

$$
\int_\Omega \nabla u \cdot \nabla v = \int_\Omega f v + \int_{\partial\Omega} g\, v \, dS \qquad \text{for all } v \in H^1(\Omega). \tag{N}
$$

The Neumann data did not enter through the space — it entered through the *linear functional* $\ell(v) = \int_\Omega f v + \int_{\partial\Omega} g v$. We never imposed $\partial_n u = g$ on the trial functions; it is *automatically satisfied by the minimizer*, as the reader can confirm by running the recovery argument with $v \in C^\infty(\overline\Omega)$ not compactly supported and reading off the boundary integral. Hence the slogan: **Dirichlet conditions constrain the space (essential); Neumann conditions modify the right-hand side and are honoured for free by the weak solution (natural).** Knowing which is which is precisely what tells you, in a finite element code, which conditions to impose on the trial functions and which to integrate into the load vector.

The Neumann problem also illustrates the *Fredholm* face of the theory. On $H^1$ the form is no longer coercive — constants satisfy $a(c,c)=0$ — so Lax–Milgram does not apply directly. The kernel is exactly the constants, and the range condition (testing (N) with $v \equiv 1$) forces the *compatibility condition*

$$
\int_\Omega f + \int_{\partial\Omega} g = 0.
$$

When this holds, one quotients out the constants — work on $\dot H^1 = \lbrace u \in H^1 : \int_\Omega u = 0\rbrace$, where the *Poincaré–Wirtinger* inequality restores coercivity — and obtains a solution unique up to an additive constant. This is the variational shadow of the fact that $-\Delta$ with Neumann conditions has a one-dimensional kernel and so is solvable iff the data is orthogonal to it.

## Beyond symmetry: the inf–sup condition

Coercivity is generous but not necessary. For *nonsymmetric* problems — say a convection–diffusion operator $-\Delta u + b\cdot\nabla u + cu = f$, whose form $a(u,v) = \int \nabla u\cdot\nabla v + \int (b\cdot\nabla u)v + \int c\,uv$ is not symmetric — coercivity may still hold if the lower-order terms are controlled, and Lax–Milgram applies verbatim. But the genuinely general statement, which also covers *saddle-point* problems (Stokes flow, mixed formulations, constrained minimization) where coercivity definitively fails, is the **Banach–Nečas–Babuška theorem**: with possibly distinct trial space $U$ and test space $V$ (both Banach, $V$ reflexive), the problem "find $u\in U$ with $a(u,v) = \ell(v)$ for all $v \in V$" is well-posed iff the *inf–sup* (or LBB) condition holds,

$$
\inf_{u \in U} \sup_{v \in V} \frac{a(u,v)}{\|u\|_U\,\|v\|_V} = \beta > 0,
$$

together with a nondegeneracy condition on $V$ ($\sup_u a(u,v) > 0$ for every $v \neq 0$). Coercivity is the special case $U = V$ with $v = u$ realizing the supremum, so $\beta \ge \alpha$. The point of decoupling trial and test spaces is that it is exactly what makes *mixed* weak formulations — where one solves for a field and a Lagrange multiplier (a pressure, a flux) simultaneously — tractable, and the inf–sup constant $\beta$ is the quantity that controls their stability both at the continuous level and, crucially, after discretization.

## From weak back to strong: elliptic regularity

We bought existence by lowering our standards. The remarkable fact is that the solution we obtain often *exceeds* the standard we settled for. This is the content of **elliptic regularity**: if $f \in L^2(\Omega)$ and $\partial\Omega$ is $C^{1,1}$ (or $\Omega$ is convex), the weak solution of $(\ast)$ in fact lies in $H^2(\Omega)$, with the *a priori* estimate

$$
\|u\|_{H^2(\Omega)} \le C\,\|f\|_{L^2(\Omega)}.
$$

The driving idea is that $-\Delta u = f$ *controls a particular combination of the second derivatives* (their sum), and the ellipticity of the operator — uniform invertibility of its symbol — upgrades this to control of *all* second derivatives; on the interior this is seen cleanly by testing with difference quotients $D_h^{-} D_h^{+} u$ and bounding them uniformly in $h$, then passing to the limit. **Bootstrapping** then iterates the gain: if $f \in H^k$ one obtains $u \in H^{k+2}$, and if $f$ is smooth so is $u$, whereupon (by the hierarchy above) the weak solution is a genuine classical solution. So the logical arc of the whole subject is: *relax to gain existence, then regularize to recover the strong solution you originally wanted.* The two halves are run by completely different machinery — functional analysis on one side, hard interior/boundary estimates on the other — and the weak formulation is the hinge between them.

A caveat worth internalizing, since it is where intuition often fails: the $H^2$ conclusion is *false on nonsmooth domains*. On a domain with a reentrant corner of interior angle $\omega > \pi$, the solution behaves like $r^{\pi/\omega}\sin(\tfrac{\pi}{\omega}\theta)$ near the corner, whose second derivatives fail to be square-integrable — the weak solution exists and is unique, but it is genuinely not $H^2$. The weak formulation, having never asked for two derivatives, is unbothered; it is the *upgrade* that is obstructed, and the obstruction is geometric.

## The numerical payoff: Galerkin and finite elements

Everything above was about the infinite-dimensional problem, but the same formulation discretizes with no conceptual change, which is why it underlies essentially all of computational PDE. **The Galerkin recipe is to replace $H_0^1$ by a finite-dimensional subspace $V_h \subset H_0^1$ and pose the identical equation there:** find $u_h \in V_h$ with

$$
a(u_h, v_h) = \ell(v_h) \qquad \text{for all } v_h \in V_h.
$$

Expanding $u_h = \sum_j U_j \phi_j$ in a basis $\lbrace \phi_i\rbrace$ of $V_h$ and testing against each $\phi_i$ turns this into the linear system $\mathbf{A}\,\mathbf{U} = \mathbf{F}$ with *stiffness matrix* $\mathbf{A}\_{ij} = a(\phi_j, \phi_i)$ and *load vector* $\mathbf{F}\_i = \ell(\phi_i)$. Coercivity of $a$ makes $\mathbf{A}$ positive definite (and, for symmetric $a$, symmetric), hence invertible — so the discrete problem inherits well-posedness from the continuous one automatically. **In the finite element method one takes $V_h$ to be continuous piecewise polynomials on a mesh,** whose local supports make $\mathbf{A}$ sparse; the essential/natural dichotomy from above tells the implementer to enforce Dirichlet data on the nodal values (constraining $V_h$) while assembling Neumann data into $\mathbf{F}$.

The quality of $u_h$ is governed by a single clean estimate, **Céa's lemma**, whose proof is two lines of *Galerkin orthogonality* $a(u - u_h, v_h) = 0$ (subtract the discrete equation from the continuous one tested on $v_h \in V_h$) combined with boundedness and coercivity:

$$
\|u - u_h\|_{H^1} \;\le\; \frac{M}{\alpha}\, \inf_{v_h \in V_h} \|u - v_h\|_{H^1}.
$$

That is, **the Galerkin solution is quasi-optimal: up to the constant $M/\alpha$ it is as good as the best approximation of $u$ available in $V_h$.** (For symmetric $a$ the bound sharpens to $\sqrt{M/\alpha}$ in the energy norm, where Galerkin orthogonality says $u_h$ is the *exact* energy-orthogonal projection of $u$ onto $V_h$ — the discrete solution is the literal best approximation in the energy inner product.) The entire convergence theory of finite elements is then reduced to a question of *approximation theory* — how well do piecewise polynomials approximate $u$? — which is answered by interpolation estimates in terms of the mesh size and the regularity of $u$. And so the loop closes once more: the regularity we extracted in the previous section is exactly the input that Céa's lemma converts into a convergence rate.

## Remarks

**What is really going on** is that the weak formulation moves the PDE into the function space where its operator is *natively bounded and invertible*. The classical operator $-\Delta : C^2 \to C^0$ is unbounded and its domain is not complete; the weak operator $-\Delta : H_0^1 \to H^{-1}$ is a bounded isomorphism, and Lax–Milgram is the statement that it is one. Choosing the energy space is not a trick to handle rough data — it is the recognition of the *correct* domain of the operator, the one on which the problem was well-posed all along.

**The conceptual endpoint is the distributional derivative.** A weak derivative is *defined* by the integration-by-parts identity $\int (\partial_i u)\varphi = -\int u\,\partial_i \varphi$; that is, weak derivatives are engineered precisely so that the one move underlying this entire post holds *by fiat*, for every locally integrable function. From that vantage the weak formulation is not a clever reformulation of a PDE but simply the PDE *read in the only sense in which its terms are universally defined*. Sobolev spaces are then the completions that make these definitions closed under limits.

**The same template recurs across the analyst's toolkit**, and is worth recognizing in its other guises. The minimizing-movements / JKO scheme realizes the Fokker–Planck equation (and hence the score-based diffusion PDEs) as a *weak/variational* time discretization — at each step one solves a variational problem whose Euler–Lagrange equation is one implicit Euler step of the gradient flow, the metric having been swapped from $L^2$ to Wasserstein. Indeed the entire theory of gradient flows in metric spaces is an extended meditation on what "the equation holds against all admissible variations" should mean when there is no linear structure to differentiate in. And in the Bayesian inverse-problems setting, the precision operator of a Gaussian field is exactly a coercive elliptic operator presented through its bilinear form — the Cameron–Martin inner product *is* the energy inner product $a(\cdot,\cdot)$, and Lax–Milgram is what guarantees the covariance operator inverting it exists and is bounded. Once you see the pattern — multiply by a test object, integrate, move the derivative, read off a bounded form on a complete space — you see it everywhere, and the elliptic Dirichlet problem is simply its cleanest specimen.