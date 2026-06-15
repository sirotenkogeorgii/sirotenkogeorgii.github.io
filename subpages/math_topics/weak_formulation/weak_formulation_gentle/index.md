---
layout: post
title: "Weak Formulations, or: How to Solve a Differential Equation Without Differentiating"
date: 2026-06-14
tags: [pde, functional-analysis, sobolev-spaces, calculus-of-variations, expository]
---

There is a small intellectual scandal at the heart of the theory of partial differential equations, and it is worth confronting it head-on before we do anything technical. Consider the most innocent equation imaginable, the Poisson equation

$$
-\Delta u = f \qquad \text{in } \Omega, \qquad u = 0 \text{ on } \partial\Omega,
$$

which models, among a thousand other things, the steady-state temperature in a region $\Omega$ heated by a source $f$, or the displacement of an elastic membrane pinned to a frame. To even *write down* the equation we have already demanded that $u$ be twice differentiable, since $\Delta u = \partial_{11} u + \cdots + \partial_{nn} u$ involves second derivatives. So a "solution" is, by the literal reading, a function $u \in C^2(\Omega)$ satisfying the equation at every point.

The scandal is that this literal reading is *both too strong and too weak at the same time*. It is too strong because nature routinely hands us sources $f$ — a point load on a beam, a discontinuous heat source, the kink in a plucked string — for which **no $C^2$ solution exists**, even though the physical system manifestly does something sensible. And it is too weak because $C^2$, as a space, is analytically miserable: it is not a Hilbert space, it has no usable inner product, compactness is hard to come by, and the whole arsenal of orthogonal projections and energy minimization that makes linear analysis powerful is simply unavailable.

<figure>
  <img src="{{ '/assets/images/notes/weak_formulation/scandal_point_load.png' | relative_url }}" alt="Two panels. Left: the solution of minus u double-prime equals a point load on the unit interval, a triangular tent pinned to zero at both ends with a sharp kink at the apex where the load sits; the kink is annotated 'no second derivative here'. Right: the slope of that tent, a step function that jumps down by one at the apex, with a note that the second derivative is therefore a spike, not a function, so no classical solution exists." loading="lazy">
  <figcaption>The opening scandal made concrete. A concentrated load $f=\delta_{x_0}$ on a string produces the obvious physical answer — a triangular tent pinned at the ends (left). Yet that tent has a <strong>kink</strong>: its slope $u'$ is a step that jumps down by $1$ (right), so the classical second derivative $u'' = \delta_{x_0}$ is a spike rather than a function and <strong>no $C^2$ solution exists</strong>. The shape is manifestly sensible and manifestly not classical — exactly the "too strong" half of the scandal. The tent is, however, a perfectly good <em>weak</em> solution, as we are about to make precise.</figcaption>
</figure>

The **weak formulation** is the resolution. Its governing idea can be stated in one sentence, and everything else in this post is commentary on it:

> *Stop asking what a function equals at each point, and start asking what it does when paired against test functions; this trades pointwise regularity — which is fragile — for Hilbert-space geometry, which is robust.*

Let me set conventions, and then we will earn that sentence.

**Conventions.** Throughout, $\Omega \subset \mathbb{R}^n$ is a bounded open set with a reasonably regular (say Lipschitz) boundary $\partial\Omega$. By $C_c^\infty(\Omega)$ I mean the smooth functions compactly supported strictly inside $\Omega$ — these are our *test functions*, and the key feature for us is that they and all their derivatives vanish on (a neighborhood of) the boundary. The pairing $\int_\Omega u\,v\,dx$ is the $L^2(\Omega)$ inner product, written $\langle u, v\rangle$. All functions are real-valued; the complex case is a routine modification.

## 1. The testing trick

Suppose, optimistically, that a classical solution $u \in C^2(\Omega)$ does exist. Take any test function $v \in C_c^\infty(\Omega)$, multiply the equation by it, and integrate over $\Omega$:

$$
-\int_\Omega (\Delta u)\, v \, dx = \int_\Omega f\, v\, dx.
$$

This is a true statement — nothing has happened yet. The whole game is what we do to the left-hand side. **Integrate by parts** (Green's first identity), which moves one derivative off of $u$ and onto $v$:

$$
-\int_\Omega (\Delta u)\, v\, dx
= \int_\Omega \nabla u \cdot \nabla v \, dx \;-\; \underbrace{\int_{\partial\Omega} (\partial_\nu u)\, v \, dS}_{=\,0}.
$$

The boundary integral, the term with the outward normal derivative $\partial_\nu u$, vanishes **because $v$ vanishes on $\partial\Omega$**. This is the first place we are paid for choosing test functions that die at the boundary: the boundary condition $u = 0$ and the test-function support condition $v\mid\_{\partial\Omega} = 0$ conspire to delete the term we cannot control.

What remains is the identity

$$
\boxed{\;\int_\Omega \nabla u \cdot \nabla v \, dx \;=\; \int_\Omega f\, v\, dx \qquad \text{for all } v \in C_c^\infty(\Omega).\;}
$$

Stare at this. The original equation asked for second derivatives of $u$; this identity asks only for **one** derivative of $u$ (inside $\nabla u$), and it is moreover *symmetric* between $u$ and $v$ — each appears under a single gradient. We have spent the painful second derivative and bought back a symmetric expression in first derivatives. That asymmetry — derivatives are expensive, and integration by parts lets us *redistribute* them — is the entire mechanism, and it will pay off twice more before we are done.

<figure>
  <img src="{{ '/assets/images/notes/weak_formulation/testing_trick.png' | relative_url }}" alt="Two panels. Left: a smooth bump test function on the interval from zero to one, strictly positive on a sub-interval in the middle and identically zero near both endpoints, its support shaded; a note says v and all its derivatives vanish near the boundary so the boundary integral is zero. Right: two integrands plotted against x — the curve minus Laplacian-u times v, with two derivatives on u, and the curve grad-u dot grad-v, with one derivative on each; a box reports that the two integrals are numerically equal at 3.329." loading="lazy">
  <figcaption>The testing trick in one picture. A test function $v \in C_c^\infty(\Omega)$ (left) is smooth, supported strictly inside $\Omega$, and flat-zero near the boundary together with all its derivatives — which is exactly why the boundary term $\int_{\partial\Omega}(\partial_\nu u)\,v\,dS$ dies. Multiplying $-\Delta u = f$ by $v$ and integrating by parts then trades the integrand $-(\Delta u)\,v$ (two derivatives piled on $u$) for the symmetric $\nabla u \cdot \nabla v$ (one derivative on each), and the two carry the <strong>same integral</strong> (right, both $=3.329$ here). The derivative was not destroyed — it was <em>redistributed</em> from $u$ onto $v$.</figcaption>
</figure>

**The conceptual inversion.** So far this is a *consequence* of being a classical solution. The decisive move is to turn it into a *definition*. We declare: $u$ is a **weak solution** if the boxed identity holds for all test functions $v$. A classical solution is automatically weak, by the computation above. But the converse can fail — and that failure is exactly the room we needed. A weak solution only has to have one derivative make sense, and only in an integrated, averaged sense, so functions that are far too rough to be classical solutions can still be weak ones.

## 2. What "one derivative in an averaged sense" means

To take the definition seriously we must say precisely what $\nabla u$ means when $u$ is not differentiable. Here the same integration-by-parts move, applied in reverse, *defines* differentiation.

Suppose $u \in C^1$. Then for any $\varphi \in C_c^\infty(\Omega)$, integration by parts gives $\int_\Omega u\, \partial_i \varphi\, dx = -\int_\Omega (\partial_i u)\, \varphi\, dx$, with no boundary term (again, $\varphi$ has compact support). The right-hand side only sees $\partial_i u$ through how it pairs against test functions. So we promote this identity to a definition for functions that may not be differentiable at all:

> A function $g \in L^1_{\mathrm{loc}}(\Omega)$ is the **weak derivative** $\partial_i u$ of $u$ if
> 
> $$\int_\Omega u\,\partial_i \varphi\, dx = -\int_\Omega g\,\varphi\, dx \qquad \text{for every } \varphi \in C_c^\infty(\Omega).$$

The defining property of the derivative is no longer a limit of difference quotients — a pointwise, fragile notion — but an integration-by-parts identity tested against all $\varphi$. The classic example: $u(x) = \lvert x\rvert$ on $(-1,1)$ has no classical derivative at $0$, but its weak derivative is the sign function $\mathrm{sgn}(x)$, which is exactly what the integration-by-parts identity produces. The single bad point is invisible to integration; it has measure zero and contributes nothing.

<figure>
  <img src="{{ '/assets/images/notes/weak_formulation/weak_derivative.png' | relative_url }}" alt="Two panels. Left: the V-shaped graph of u equals the absolute value of x on minus one to one, with a red dot at the kink at the origin labelled 'no classical derivative here', overlaid by three progressively sharper smooth curves that approach the V from above. Right: the derivatives of those smooth curves, S-shaped curves that steepen and converge to the step function sgn of x; a box notes the integration-by-parts pairing and that the value at zero has measure zero." loading="lazy">
  <figcaption>What "one derivative in an averaged sense" looks like. The kink of $u = |x|$ (left) has no classical derivative at $0$, but the smooth mollifications $u_\varepsilon = \sqrt{x^2+\varepsilon^2}$ close in on it from above, and their honest derivatives $u_\varepsilon' = x/\sqrt{x^2+\varepsilon^2}$ converge to the step $\mathrm{sgn}(x)$ (right). That step <em>is</em> the weak derivative of $|x|$: it is precisely what the integration-by-parts identity $\int u\,\varphi'\,dx = -\int \mathrm{sgn}(x)\,\varphi\,dx$ produces. The single bad point at $x=0$ has measure zero, so integration never sees it — whatever value we assign $\mathrm{sgn}(0)$ changes nothing.</figcaption>
</figure>

This lets us build the natural home for weak solutions, the **Sobolev space**

$$
H^1(\Omega) := \big\{ u \in L^2(\Omega) : \partial_i u \in L^2(\Omega) \text{ weakly, } i = 1,\dots,n \big\},
$$

equipped with the inner product $\langle u, v\rangle_{H^1} = \int_\Omega u v\, dx + \int_\Omega \nabla u \cdot \nabla v\, dx$. The crucial structural fact, which I will not prove but which is the technical engine behind everything, is that **$H^1(\Omega)$ is complete** — a Hilbert space. This is precisely the property $C^2$ lacked. Completeness is what lets us take limits of approximating sequences and know the limit still lives in the space, and it is the non-negotiable hypothesis of every existence theorem below.

**Encoding the boundary condition.** We still need to impose $u = 0$ on $\partial\Omega$, but a generic $H^1$ function need not even have well-defined boundary values pointwise. The trick is again to bypass pointwise notions: define

$$
H^1_0(\Omega) := \overline{C_c^\infty(\Omega)}^{\,\|\cdot\|_{H^1}},
$$

the closure of the test functions in the $H^1$ norm. A function in $H^1_0$ is, by construction, an $H^1$-limit of functions that genuinely vanish near the boundary, and this is the correct weak rendering of the homogeneous Dirichlet condition. Notice the test functions now play *both* roles: as the trial space (where $u$ lives, after closure) and as the test space (what we pair against). For the symmetric problems we are looking at, this is no coincidence, as we will see.

## 3. The abstract skeleton

Strip away the specifics and a clean structure emerges. Define a **bilinear form** and a **linear functional** on $H = H^1_0(\Omega)$:

$$
a(u,v) := \int_\Omega \nabla u \cdot \nabla v\, dx, \qquad \ell(v) := \int_\Omega f\, v\, dx.
$$

The weak formulation of the Poisson problem is then, with perfect economy:

> Find $u \in H$ such that $\quad a(u,v) = \ell(v) \quad$ for all $v \in H$.

This is no longer a statement about $\Omega$, gradients, or Laplacians — it is a statement about a bilinear form on an abstract Hilbert space. Enormously many different PDEs (variable-coefficient diffusion $-\nabla\cdot(A\nabla u) = f$, reaction-diffusion $-\Delta u + cu = f$, linear elasticity, and so on) collapse to *the same abstract problem* with a different $a$ and $\ell$. Solve the abstract problem once and you have solved them all. This is the payoff of the change in perspective: a single theorem will do the work of a library of ad hoc constructions.

## 4. The existence engine: Lax–Milgram

When does the abstract problem have a unique solution? The answer is one of the most useful theorems in applied analysis.

**Lax–Milgram theorem.** *Let $H$ be a Hilbert space and $a: H \times H \to \mathbb{R}$ a bilinear form which is*

- ***bounded** (continuous): $\lvert a(u,v)\rvert \le M\,\|u\|\,\|v\|$ for some $M < \infty$, and*
- ***coercive**: $a(u,u) \ge \alpha\,\|u\|^2$ for some $\alpha > 0$.*

*Then for every bounded linear functional $\ell \in H^\ast$ there is a unique $u \in H$ with $a(u,v) = \ell(v)$ for all $v \in H$, and moreover $\|u\| \le \|\ell\|_{H^\ast}/\alpha$.*

It is worth pausing on what each hypothesis *buys*. Boundedness says the form is continuous, so approximate solutions converge to genuine ones. Coercivity is the heart of the matter: it says the form "sees" the whole norm — energy cannot leak away into directions the form ignores — and it is exactly what rules out the degeneracy that would let solutions fail to exist or fail to be unique. The final inequality is a **stability estimate**: it says the solution depends continuously on the data, so the problem is *well-posed* in the sense of Hadamard. Existence, uniqueness, and stability all fall out of two inequalities.

<figure>
  <img src="{{ '/assets/images/notes/weak_formulation/coercive_bowl.png' | relative_url }}" alt="Two 3-D surface plots of the energy J over a plane of coefficients. Left, the coercive case: a clean elliptic paraboloid, a bowl with a single lowest point marked in red. Right, the non-coercive case: a parabolic trough that is completely flat along one horizontal direction, with a green line marking a whole valley of equally-minimal points." loading="lazy">
  <figcaption>Coercivity is what gives the energy a bottom. The quadratic energy $J(v) = \tfrac{1}{2}a(v,v) - \ell(v)$, drawn over a plane of coefficients. When $a$ is <strong>coercive</strong> ($a(u,u) \ge \alpha\|u\|^2$ with $\alpha > 0$) it is a strictly convex bowl with a single minimiser (left) — this is the geometric content of Lax–Milgram. When coercivity fails there is a <strong>flat direction</strong> in which the form sees nothing: the bowl degenerates into a trough (right) and a whole line of competitors ties for the minimum, so the solution is no longer unique. For Poisson on all of $H^1$ the constants are exactly such a flat direction ($\nabla(\text{const}) = 0$); the next figure shows how passing to $H^1_0$ removes it.</figcaption>
</figure>

**Verifying the hypotheses for Poisson.** Boundedness is Cauchy–Schwarz: $\lvert a(u,v)\rvert = \lvert\int \nabla u\cdot\nabla v\rvert \le \|\nabla u\|_{L^2}\|\nabla v\|\_{L^2} \le \|u\|\_{H^1}\|v\|\_{H^1}$. Coercivity, $a(u,u) = \int \lvert\nabla u\rvert^2 = \|\nabla u\|\_{L^2}^2 \gtrsim \|u\|\_{H^1}^2$, looks like it is *missing* the $\|u\|\_{L^2}^2$ part of the $H^1$ norm — and on all of $H^1$ it genuinely would fail (take $u$ constant: $\nabla u = 0$ but $u \ne 0$). It is rescued on $H^1_0$ by the

> **Poincaré inequality.** *For $\Omega$ bounded there is $C_\Omega$ with $\|u\|_{L^2(\Omega)} \le C_\Omega \|\nabla u\|_{L^2(\Omega)}$ for all $u \in H^1_0(\Omega)$.*

A function that is forced to vanish at the boundary cannot be large in the bulk without having a correspondingly large gradient to climb up from zero; that is the whole content of Poincaré. With it, $\|\nabla u\|\_{L^2}^2$ controls the full $H^1$ norm on $H^1_0$, coercivity holds, and **Lax–Milgram delivers a unique weak solution** to the Poisson problem for *every* $f \in L^2(\Omega)$ — including the rough sources for which no classical solution existed. This is precisely the room we promised ourselves in §1.

<figure>
  <img src="{{ '/assets/images/notes/weak_formulation/poincare.png' | relative_url }}" alt="Two panels. Left: four functions on the interval from zero to one, all pinned to zero at both endpoints — a sine arch, a parabola, a triangular tent, and a full sine wave — illustrating that to be tall in the middle a pinned function must have steep slopes. Right: a scatter plot of bulk size, the L2 norm of u, against gradient size, the L2 norm of grad u; every function lands on or below the straight Poincaré line of slope one over pi, the region above the line is shaded and labelled forbidden, and a star marks the sine arch, which achieves equality." loading="lazy">
  <figcaption>The Poincaré inequality, and why it rescues coercivity. Every $u \in H^1_0(0,1)$ is pinned to zero at both ends (left), so it cannot be large in the bulk without paying for it with a large gradient — to rise, it must climb. Plotting bulk size $\|u\|_{L^2}$ against gradient size $\|\nabla u\|_{L^2}$ (right), <strong>no admissible function can climb above the line</strong> $\|u\|_{L^2} = \tfrac{1}{\pi}\|\nabla u\|_{L^2}$ — that ceiling <em>is</em> Poincaré, with sharp constant $C_\Omega = 1/\pi$ on the unit interval, achieved by the first eigenfunction $\sin(\pi x)$ (the star). This is exactly the inequality that lets $\|\nabla u\|_{L^2}^2$ control the full $H^1$ norm on $H^1_0$, turning the degenerate trough of the previous figure back into a coercive bowl.</figcaption>
</figure>

## 5. The variational shadow: energy minimization

When $a$ is *symmetric* — as it is for Poisson, since $\nabla u \cdot \nabla v = \nabla v \cdot \nabla u$ — there is a second, equivalent face of the weak formulation that physicists discovered first and that explains *why* the whole framework is natural. Consider the **Dirichlet energy**

$$
J(v) := \tfrac{1}{2}\,a(v,v) - \ell(v) = \tfrac{1}{2}\int_\Omega |\nabla v|^2\, dx - \int_\Omega f\, v\, dx.
$$

**Claim:** $u$ solves the weak formulation if and only if $u$ minimizes $J$ over $H = H^1_0(\Omega)$.

The proof is a one-line computation that any calculus-of-variations veteran will recognize. For $u$ a minimizer and any direction $v$, the function $t \mapsto J(u + tv)$ has a minimum at $t = 0$, so its derivative vanishes there:

$$
0 = \frac{d}{dt}\Big|_{t=0} J(u + tv) = a(u,v) - \ell(v).
$$

That is the weak formulation, appearing as the **Euler–Lagrange equation** of the energy. Conversely, coercivity makes $J$ strictly convex and coercive (in the sense that $J(v) \to +\infty$ as $\|v\| \to \infty$), so it has a unique minimizer, recovering Lax–Milgram in the symmetric case directly from the geometry of convex minimization — indeed, from the Riesz representation theorem, of which symmetric Lax–Milgram is a mild generalization.

This dual picture is conceptually load-bearing. The physical principle "nature minimizes energy" and the analytical object "weak solution of a PDE" are *the same thing viewed from two sides*. The membrane settles into the shape that minimizes its stored elastic energy; that shape is, in the weak sense, a solution of $-\Delta u = f$. The weak formulation is what makes this folklore into a theorem, because minimization is a Hilbert-space operation and minimizers live happily in $H^1_0$ even when they are too rough to be classical solutions.

## 6. What we gained, and the payoff downstream

It is worth tallying the profit explicitly.

**We decoupled existence from regularity.** This is the single most important strategic consequence. In the classical approach you must produce a *smooth* solution in one stroke, which is brutally hard. The weak approach splits the labor in two: first prove a weak solution *exists*, cheaply, using only Hilbert-space geometry (Lax–Milgram); then, *separately*, study how smooth that solution is, using **elliptic regularity theory**. The regularity theorems say, roughly, that the weak solution is exactly as smooth as the data allows: if $f \in L^2$ and $\partial\Omega$ is smooth then $u \in H^2$; bootstrapping, if $f \in C^\infty$ then $u \in C^\infty$ and the weak solution *was a classical solution all along*. So we lose nothing in the smooth case — a classical solution is just a weak solution that happens to be regular — and we gain solutions in every case where regularity fails. The weak solution is the genuinely fundamental object; classical solutions are the lucky special cases.

**We made the problem computable.** Here is perhaps the most consequential practical payoff. The **Galerkin method** is nothing more than the weak formulation *restricted to a finite-dimensional subspace* $V_h \subset H$: find $u_h \in V_h$ with $a(u_h, v_h) = \ell(v_h)$ for all $v_h \in V_h$. Pick a basis $\lbrace \phi_j\rbrace$ of $V_h$, write $u_h = \sum_j c_j \phi_j$, and the abstract problem becomes the linear system $\sum_j a(\phi_j, \phi_i)\, c_j = \ell(\phi_i)$, i.e. a matrix equation $K\mathbf{c} = \mathbf{b}$ with stiffness matrix $K_{ij} = a(\phi_j,\phi_i)$. When the $\phi_j$ are chosen to be little tent functions supported on a mesh, this is the **finite element method**, the workhorse of computational engineering. Coercivity and boundedness even hand you a free error bound (Céa's lemma): the computed $u_h$ is quasi-optimal, no worse (up to the constant $M/\alpha$) than the *best possible* approximation of $u$ from $V_h$. The same two inequalities that gave existence give convergence of the numerical scheme.

<figure>
  <img src="{{ '/assets/images/notes/weak_formulation/galerkin_fem.png' | relative_url }}" alt="Three panels. Left: a set of overlapping triangular hat basis functions on a mesh of the unit interval, one tent peaked at each interior node. Middle: the exact parabolic solution of minus u double-prime equals one drawn as a smooth curve, with two piecewise-linear Galerkin approximations overlaid — a coarse one with two interior nodes and a finer one with six — visibly converging to it. Right: the stiffness matrix shown as a grid, non-zero only on the main diagonal and the two adjacent diagonals, that is, tridiagonal and sparse, with entries twenty and minus ten." loading="lazy">
  <figcaption>How the weak formulation becomes a computation. Restrict the weak problem to a finite-dimensional $V_h \subset H$ spanned by little <strong>hat functions</strong> on a mesh (left). Writing $u_h = \sum_j c_j \phi_j$ turns $a(u_h, v_h) = \ell(v_h)$ into the linear system $K\mathbf{c} = \mathbf{b}$, whose solution converges to the true $u$ as the mesh is refined (middle: $n=2$ then $n=6$ closing in on the exact parabola $u = \tfrac{1}{2}x(1-x)$ of $-u''=1$). Because each hat overlaps only its immediate neighbours, the stiffness matrix $K_{ij} = a(\phi_j, \phi_i)$ is <strong>sparse</strong> — here tridiagonal (right) — which is what makes the finite element method fast. And the same two inequalities (boundedness, coercivity) that bought existence also bound the error, via Céa's lemma.</figcaption>
</figure>

## Remarks

**1. The duality philosophy is everywhere.** The move that animates this whole subject — *identify an object by how it pairs against test functions* — is the founding idea of the theory of distributions (Schwartz). A distribution is, by definition, nothing but a rule for eating test functions, and from this vantage the Dirac delta, the weak derivative, and the weak solution are all instances of one idea: replace a fragile pointwise object with the robust functional it induces. Once you internalize "a function is what it does to test functions," weak formulations stop looking like a trick and start looking inevitable.

**2. The same skeleton stretches very far.** For *time-dependent* problems (the heat equation, the wave equation) one runs the same testing argument in space and handles time via a Gelfand triple $V \hookrightarrow H \hookrightarrow V^\ast$ and Galerkin-in-time, yielding weak solutions in spaces like $L^2(0,T; H^1_0)$. For *nonlinear* problems, coercivity is replaced by **monotonicity** of the operator (the Browder–Minty theory). For *saddle-point* problems like the Stokes equations of fluid flow, coercivity on the whole space fails and is replaced by the celebrated **inf–sup (Babuška–Brezzi) condition**, the structural heart of mixed finite element methods. In each case the philosophy is unchanged; only the inequality guaranteeing well-posedness is upgraded.

**3. A bridge to gradient flows.** The variational picture of §5 is the static tip of a much larger iceberg. A gradient flow $\partial_t u = -\nabla J(u)$ can be solved by the **minimizing-movements / JKO scheme**: discretize time and, at each step, minimize $J(v) + \tfrac{1}{2\tau}\,d(v, u^{\mathrm{old}})^2$ over $v$. Each step is a *variational* (hence weak) problem of exactly the type above, and the limit as the step $\tau \to 0$ recovers the flow. The Fokker–Planck equation realized as the gradient flow of entropy in Wasserstein space — the analytic backbone of diffusion models — is the most celebrated instance. Weak formulations are not merely a tool for elliptic boundary-value problems; they are the language in which "the steepest-descent direction of an energy" becomes a rigorous, solvable object. The membrane finding its rest shape and a diffusion model denoising an image are, structurally, the same story told at two scales.