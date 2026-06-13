---
layout: default
title: "The Laplacian, or the Geometry of Averaging"
tags:
  - pde
  - analysis
  - laplacian
  - harmonic-functions
  - diffusion
  - spectral-theory
  - random-walks
  - optimal-transport
---

From Brezis:
> So far the operators we have studied (compact, self-adjoint, etc.) have all been *bounded*. PDE theory, however, is dominated by **unbounded** operators — the Laplacian, the heat operator, transport operators, etc. — and these typically arise as generators of *time evolutions*. The central question is:

The Laplacian $\Delta = \sum_{i=1}^n \partial_{x_i}^2$ is the most ubiquitous differential operator in all of mathematics: it sits at the centre of potential theory, the heat and wave equations, spectral geometry, probability, and—as we will see at the end—the optimal-transport picture of diffusion. It is tempting to treat it as a piece of notation, a Swiss-army symbol that one simply learns to manipulate. I want instead to argue that almost everything the Laplacian does is downstream of a single geometric fact, and that once that fact is internalised the operator's many faces stop looking like coincidences.

That single fact is this: **the Laplacian measures the infinitesimal failure of a function to equal its own local average.** Diffusion smooths because heat flows from above-average to below-average regions; harmonic functions are rigid because they have no such failure to correct; the spectrum encodes geometry because averaging is sensitive to shape. Keep that sentence in mind throughout.

## Conventions and normalisations

A nuisance worth settling first, because half of all sign errors in the subject come from leaving it implicit.

I take $\Delta = \sum_i \partial_i^2$ as the basic object, so that on $\mathbb{R}^n$ it has the Fourier symbol $-\lvert\xi\rvert^2$ (negative). Analysts frequently prefer $-\Delta$, which is the *non-negative* operator (the one whose quadratic form is $\int \lvert\nabla u\rvert^2 \ge 0$); geometers building the Hodge Laplacian make the same positive choice. I will write $\Delta$ for the analyst-with-the-minus and flag $-\Delta$ explicitly whenever positivity matters. I use the Fourier transform $\hat u(\xi) = \int_{\mathbb{R}^n} u(x)\, e^{-i x\cdot\xi}\,dx$, so $\widehat{\partial_j u} = i\xi_j\,\hat u$. I write $\fint_E = \tfrac{1}{\lvert E\rvert}\int_E$ for an average, $B_r(x)$ for the open ball, $\partial B_r(x)$ for its sphere, and $\alpha(n) = \lvert B_1\rvert$, $\omega(n) = \lvert\partial B_1\rvert = n\alpha(n)$.

## 1. The one idea: deviation from the local average

Everything starts with a second-order Taylor expansion and a symmetry argument. Fix smooth $u$ and expand about $x$:

$$
u(y) = u(x) + \nabla u(x)\cdot(y-x) + \tfrac12 (y-x)^{\mathsf T}\,\nabla^2 u(x)\,(y-x) + O(|y-x|^3).
$$

**The linear term averages to zero; the quadratic term sees only the trace.** Integrate over a sphere $\partial B_r(x)$. The linear term is odd about $x$ and vanishes. In the quadratic term, $\fint_{\partial B_r}(y_i-x_i)(y_j-x_j)\,dS = 0$ for $i\ne j$ by reflection symmetry, while the diagonal entries are all equal and sum to $\fint_{\partial B_r}\lvert y-x\rvert^2 = r^2$, so each is $r^2/n$. Hence

$$
\fint_{\partial B_r(x)} u\, dS \;=\; u(x) + \frac{r^2}{2n}\,\Delta u(x) + O(r^4),
$$

and the same computation over the solid ball (where $\fint_{B_r}\lvert y-x\rvert^2 = \tfrac{n}{n+2}r^2$) gives the slightly different constant $\tfrac{r^2}{2(n+2)}\Delta u(x)$. Rearranging the spherical version isolates the operator:

$$
\boxed{\;\Delta u(x) = \lim_{r\to 0}\, \frac{2n}{r^2}\left(\fint_{\partial B_r(x)} u\, dS - u(x)\right).\;}
$$

This is the definition I want you to carry around. The Laplacian is, up to the dimensional factor $2n/r^2$, exactly *average-minus-value*. A point where $\Delta u > 0$ is a point lying below the surrounding average—a valley that diffusion will fill in. This single picture already predicts the maximum principle (an interior maximum cannot lie below its average, so $\Delta u \le 0$ there) and the direction of heat flow, before we have written down a single PDE.

<figure>
  <img src="{{ '/assets/images/notes/math_topics/laplacian/lap_average_minus_value.png' | relative_url }}" alt="Left: a sine curve with two marked points. At a concave hump the value (open dot) sits above the midpoint of the chord through its two neighbours (cross), so the average lies below the value and the Laplacian is negative; at a convex valley the value sits below that chord midpoint, so the average exceeds the value and the Laplacian is positive. Right: for a two-dimensional function, the spherical average minus the value plotted against radius, lying on top of the parabola r-squared over 2n times the Laplacian for small r." loading="lazy">
  <figcaption>The single fact the whole essay rests on. <strong>Left:</strong> at a concave hump the point overshoots the average of its neighbours ($\langle u\rangle < u$, so $\Delta u<0$); at a convex valley it undershoots ($\langle u\rangle > u$, so $\Delta u>0$). <strong>Right:</strong> sharpening this into the definition — the spherical average minus the value is $\frac{r^2}{2n}\Delta u(x)+O(r^4)$, so the numeric gap (purple) peels away from its leading parabola (orange) only at fourth order. The factor $\frac{2n}{r^2}$ in the boxed limit is exactly what undoes this $r^2$ and isolates $\Delta u(x)$.</figcaption>
</figure>

## 2. Why *this* operator: rigid motions force it

The averaging formula explains what $\Delta$ *does*; a symmetry argument explains why it is essentially the *only* operator that could do it. Ask: which linear differential operators $L$ on $\mathbb{R}^n$ commute with all rigid motions—translations and rotations?

**Translation-invariance makes $L$ a Fourier multiplier.** A continuous linear operator commuting with all translations acts as $\widehat{Lu}(\xi) = m(\xi)\,\hat u(\xi)$ for some symbol $m$; for a differential operator with constant coefficients $m$ is a polynomial.

**Rotation-invariance makes the symbol radial.** Commuting with $\mathrm{O}(n)$ forces $m(R\xi) = m(\xi)$, so $m(\xi) = \mu(\lvert\xi\rvert^2)$ for a polynomial $\mu$.

A radial polynomial symbol of degree at most $2$ is $m(\xi) = a - b\lvert\xi\rvert^2$. If we further insist (as any reasonable "averaging defect" should) that $L$ annihilates constants, then $m(0)=0$, so $a=0$; second order means $b\ne 0$. Thus $L = b\,\Delta$. **Up to a scalar, the Laplacian is the unique second-order constant-coefficient operator invariant under Euclidean motions and killing constants.** Its uniqueness is not an accident of notation—it is forced by isotropy of space. This is also the cleanest reason the operator appears identically across physics: any local, isotropic, linear second-order law *must* be built from $\Delta$.

The symbol identity worth memorising, the engine of all the Fourier-analytic estimates, is

$$
\widehat{\Delta u}(\xi) = -|\xi|^2\,\hat u(\xi).
$$

Plane waves $e^{i x\cdot\xi}$ are the eigenfunctions; $\Delta$ is diagonal in the frequency basis with eigenvalue $-\lvert\xi\rvert^2$. High frequencies are damped hardest—which is the whole story of why $e^{t\Delta}$ smooths.

<figure>
  <img src="{{ '/assets/images/notes/math_topics/laplacian/lap_radial_symbol.png' | relative_url }}" alt="Left: a three-dimensional downward-opening paraboloid, the graph of m of xi equals minus the squared length of xi over the frequency plane. Right: the same surface seen from above as filled contours, whose level sets are perfectly concentric circles." loading="lazy">
  <figcaption>Why $\Delta$ is forced, drawn as a symbol. Translation-invariance makes any admissible operator a Fourier multiplier $m(\xi)$; rotation-invariance makes that multiplier radial — its level sets are <em>circles</em> (right), so $m(\xi)=\mu(\lvert\xi\rvert^2)$. Demanding second order and that constants are annihilated leaves only $m(\xi)=-b\lvert\xi\rvert^2$, the downward paraboloid (left). Up to the scalar $b$ that is the Laplacian, $\widehat{\Delta u}(\xi)=-\lvert\xi\rvert^2\,\hat u(\xi)$ — with the high frequencies on the rim damped hardest.</figcaption>
</figure>

## 3. The variational face: gradient of the Dirichlet energy

There is a third, equally fundamental, characterisation: $-\Delta$ is the derivative of an energy. Consider the Dirichlet energy

$$
E(u) = \tfrac12 \int_\Omega |\nabla u|^2\,dx.
$$

Perturb $u \mapsto u + \varepsilon\varphi$ with $\varphi$ compactly supported and integrate by parts:

$$
\left.\frac{d}{d\varepsilon}\right|_{0} E(u+\varepsilon\varphi) = \int_\Omega \nabla u\cdot\nabla\varphi = -\int_\Omega (\Delta u)\,\varphi.
$$

So the first variation is $\langle -\Delta u, \varphi\rangle_{L^2}$, i.e. **$-\Delta u$ is the $L^2$-gradient of the Dirichlet energy.** Critical points—solutions of $-\Delta u = 0$—are the harmonic functions, and the energy descent direction $-\nabla E = \Delta u$ is exactly the right-hand side of the heat equation. The "averaging defect" and "energy gradient" pictures are two readings of the same operator: smoothing the function and decreasing its Dirichlet energy are the same act. (We will meet a far less obvious gradient-flow reading in §8, where the metric is changed from $L^2$ to Wasserstein.)

<figure>
  <img src="{{ '/assets/images/notes/math_topics/laplacian/lap_dirichlet_energy.png' | relative_url }}" alt="Left: several snapshots of a wrinkly profile on an interval with fixed endpoints, relaxing over time toward the straight line joining the endpoint values. Right: the Dirichlet energy plotted against time, falling steeply then levelling off at the minimal energy of the straight-line state." loading="lazy">
  <figcaption>The variational reading. Because $-\Delta u$ is the $L^2$-gradient of the Dirichlet energy $E(u)=\frac{1}{2}\int\lvert\nabla u\rvert^2$, the heat equation $\partial_t u=\Delta u$ is gradient <em>descent</em> on $E$. <strong>Left:</strong> any initial profile slides toward the harmonic (here straight-line) minimiser. <strong>Right:</strong> the energy falls monotonically, $\frac{d}{dt}E=-\int\lvert\Delta u\rvert^2\le 0$, with the wrinkliest (highest-frequency) modes — which carry the most energy — flattening first.</figcaption>
</figure>

## 4. Harmonic functions: rigidity from the mean value property

Call $u$ harmonic if $\Delta u = 0$ on a domain $\Omega$. The defining limit of §1 upgrades, for harmonic functions, to an *exact* identity at every radius:

$$
u(x) = \fint_{\partial B_r(x)} u\, dS = \fint_{B_r(x)} u\, dy \qquad \text{whenever } \overline{B_r(x)}\subset\Omega.
$$

This **mean value property** is equivalent to harmonicity, and from it the entire rigidity of harmonic functions follows almost mechanically.

<figure>
  <img src="{{ '/assets/images/notes/math_topics/laplacian/lap_harmonic_mvp.png' | relative_url }}" alt="Left: a heatmap of a harmonic function on a disk, the real part of z cubed, with its six extrema marked by circles that all lie on the boundary circle and none in the interior. Right: the circular average of a function about the centre plotted against radius; for the harmonic function it is a flat horizontal line equal to the centre value, while for a non-harmonic paraboloid it rises like r-squared." loading="lazy">
  <figcaption>Harmonicity as the absence of any averaging defect. <strong>Left:</strong> a harmonic function ($\Delta u=0$) on a disk has every interior value equal to its surrounding average, so no interior bump or dip can form — the extrema ($\circ$) are pushed entirely onto the boundary. That <em>is</em> the maximum principle. <strong>Right:</strong> the mean value property made quantitative — for a harmonic $u$ the circular average $\langle u\rangle_{\partial B_r}$ stays pinned to $u(0)$ at <em>every</em> radius (flat blue line), whereas a non-harmonic function drifts off like $\frac{r^2}{2n}\Delta u$ (red).</figcaption>
</figure>

**Maximum principle.** If $u$ attains an interior maximum, equality in the mean value property forces $u$ to equal its maximum on a whole ball, hence (by connectedness) on all of $\Omega$. So a non-constant harmonic function takes its extrema only on the boundary. This is the analyst's version of the physical truism that a body with no internal heat sources is hottest at its surface.

**Smoothness and analyticity.** Averaging is a convolution with a smooth kernel, so a merely continuous function satisfying the mean value property is automatically $C^\infty$, indeed real-analytic. Harmonicity has no "weak" solutions that aren't already classical.

**Liouville and gradient estimates.** Differentiating the mean value identity gives $\lvert\nabla u(x)\rvert \le \tfrac{C_n}{r}\sup_{B_r}\lvert u\rvert$; letting $r\to\infty$ shows a bounded harmonic function on all of $\mathbb{R}^n$ is constant. Rigidity again: with no boundary to push against and no averaging defect to exploit, there is simply nothing for a harmonic function to do.

## 5. The fundamental solution and Green's functions

To *solve* $-\Delta u = f$ rather than merely study its kernel, invert the symbol: $\hat u(\xi) = \lvert\xi\rvert^{-2}\hat f(\xi)$, so $u = \Phi \ast f$ where $\widehat{\Phi}(\xi) = \lvert\xi\rvert^{-2}$. Transforming back (the symbol is radial, so $\Phi$ is too) gives the **fundamental solution**, normalised by $-\Delta\Phi = \delta_0$:

$$
\Phi(x) = \begin{cases} -\dfrac{1}{2\pi}\log|x|, & n=2,\\[2mm] \dfrac{1}{n(n-2)\alpha(n)}\,|x|^{2-n}, & n\ge 3. \end{cases}
$$

The exponent $2-n$ is forced by scaling: $\Delta$ has dimensions of (length)$^{-2}$, $\delta_0$ has dimensions (length)$^{-n}$, so $\Phi \sim \lvert x\rvert^{2-n}$ on dimensional grounds alone—the logarithm in $n=2$ is the borderline case where the power law degenerates. Physically $\Phi$ is the Newtonian/Coulomb potential of a point charge, and $u = \Phi * f$ is the potential of the charge density $f$.

**Boundary value problems and the Green's function.** On a domain $\Omega$ the same logic produces the Green's function $G(x,y) = \Phi(x-y) - h^x(y)$, where the corrector $h^x$ is harmonic and matches $\Phi(x-\cdot)$ on $\partial\Omega$, so that $G(x,\cdot)=0$ there. The Dirichlet problem $-\Delta u = f$ in $\Omega$, $u = g$ on $\partial\Omega$, is then solved explicitly:

$$
u(x) = \int_\Omega G(x,y)\,f(y)\,dy - \int_{\partial\Omega} \frac{\partial G}{\partial\nu}(x,y)\,g(y)\,dS(y).
$$

On a half-space or ball one finds $G$ by reflection (the method of images), and differentiating the boundary term recovers the **Poisson kernel**, the precise weighting by which boundary data propagates inward. Existence of $G$ on general domains is exactly the solvability of the Dirichlet problem—the question Perron's method and, historically, Hilbert's Dirichlet-principle programme were built to settle.

<figure>
  <img src="{{ '/assets/images/notes/math_topics/laplacian/lap_fundamental_solution.png' | relative_url }}" alt="Left: radial profiles of the fundamental solution, a minus-log curve for dimension two and a one-over-distance curve for dimension three, both blowing up at the origin. Right: the Green's function of the half-plane built by the method of images, a positive source above the boundary line and a negative mirror image below it, the contour plot antisymmetric across the line so it vanishes exactly on the boundary." loading="lazy">
  <figcaption>Solving $-\Delta u=f$ by inverting the symbol. <strong>Left:</strong> the fundamental solution $\Phi$ with $-\Delta\Phi=\delta_0$ — the logarithmic potential $-\frac{1}{2\pi}\log\lvert x\rvert$ in $n=2$ and the Newtonian/Coulomb $\frac{1}{4\pi}\lvert x\rvert^{-1}$ in $n=3$, both carrying the dimensionally-forced singularity $\Phi\sim\lvert x\rvert^{2-n}$. <strong>Right:</strong> on a domain the Green's function subtracts a harmonic corrector $h^x$; on a half-space that corrector is a single negative <em>image</em> charge, whose antisymmetry forces $G=0$ along the boundary so $u$ can match its data there.</figcaption>
</figure>

## 6. The heat semigroup: smoothing and the probabilistic picture

Now let the operator act in time. The heat equation $\partial_t u = \Delta u$ is solved on $\mathbb{R}^n$ by convolution against the **heat kernel**

$$
p_t(x) = (4\pi t)^{-n/2}\, e^{-|x|^2/4t},
$$

and we write $u(t) = e^{t\Delta}u_0 = p_t \ast u_0$. Three features deserve emphasis.

**Instantaneous smoothing.** In Fourier variables $e^{t\Delta}$ is multiplication by $e^{-t\lvert\xi\rvert^2}$, a Gaussian damping that crushes high frequencies super-polynomially. Hence $u(t)$ is $C^\infty$—indeed analytic—for every $t>0$, no matter how rough $u_0$ is. This is the operator-theoretic shadow of §1: each instant of flow replaces a value by a (Gaussian-weighted) average of its neighbours, and averaging is smoothing.

**Mass conservation and irreversibility.** Since $\widehat{p_t}(0)=1$, total mass $\int u$ is preserved; since $e^{-t\lvert\xi\rvert^2}$ is never invertible-with-bounds backward in time, the flow has no continuous inverse for $t<0$. Heat forgets. The Dirichlet energy decays monotonically, $\tfrac{d}{dt}E(u) = -\int\lvert\Delta u\rvert^2 \le 0$, in line with the gradient-flow reading of §3.

**The probabilistic dictionary.** The kernel $p_t$ is the transition density of Brownian motion run at the right speed: if $(X_t)$ solves $dX_t = \sqrt{2}\,dB_t$, then $\mathbb{E}[u_0(X_t) \mid X_0 = x] = (e^{t\Delta}u_0)(x)$. In this language **$\Delta$ is the infinitesimal generator of Brownian motion**, harmonic functions are exactly those constant along the flow in expectation (martingales after composition), and the maximum principle becomes the statement that a diffusing particle's expected payoff is controlled by where it can exit. The probabilistic, analytic, and variational accounts of $\Delta$ are one object viewed through three lenses.

<figure>
  <img src="{{ '/assets/images/notes/math_topics/laplacian/lap_heat_semigroup.png' | relative_url }}" alt="Left: the heat kernel, a Gaussian bump, plotted at several times; it starts tall and narrow and spreads into a low broad curve while the area underneath stays one. Right: sixty simulated Brownian-motion paths fanning out from the origin inside a plus-or-minus square-root-of-2t envelope, with a marginal histogram of their endpoints on the right matching the Gaussian heat kernel at the final time." loading="lazy">
  <figcaption>The heat semigroup and its probabilistic shadow. <strong>Left:</strong> convolution with the heat kernel $p_t=(4\pi t)^{-n/2}e^{-\lvert x\rvert^2/4t}$ — in Fourier variables the multiplier $e^{-t\lvert\xi\rvert^2}$ crushes high frequencies super-polynomially, so $u(t)$ is instantly smooth while total mass is conserved. <strong>Right:</strong> the same kernel is the transition density of Brownian motion $dX_t=\sqrt{2}\,dB_t$ — paths spread as $\sqrt{2t}$ and their endpoints reproduce $p_T$ (histogram vs. curve). In this dictionary $\Delta$ <em>is</em> the generator of the diffusion.</figcaption>
</figure>

## 7. Spectral theory: hearing the shape of a drum

On a bounded domain $\Omega$ with, say, Dirichlet boundary conditions, $-\Delta$ becomes a self-adjoint operator with *compact* resolvent. The spectral theorem then gives a discrete spectrum

$$
0 < \lambda_1 \le \lambda_2 \le \lambda_3 \le \cdots \to \infty,
$$

with an $L^2(\Omega)$-orthonormal basis of eigenfunctions $-\Delta\phi_k = \lambda_k\phi_k$. Every solution of the heat or wave equation decomposes along this basis—$e^{t\Delta}$ acts as $e^{-\lambda_k t}$ on the $k$-th mode—so the $\lambda_k$ are the resonant frequencies of the domain: the overtones of a drum with boundary $\partial\Omega$.

**Weyl's law: the spectrum hears the volume.** The eigenvalue counting function $N(\lambda) = \#\lbrace k : \lambda_k \le \lambda\rbrace$ satisfies

$$
N(\lambda) \sim \frac{\alpha(n)}{(2\pi)^n}\,|\Omega|\;\lambda^{n/2}, \qquad \lambda\to\infty.
$$

One can read off the leading constant by a heuristic dear to physicists: each eigenmode occupies a cell of volume $(2\pi)^n$ in phase space, and the modes with $-\Delta \le \lambda$ fill the region $\lbrace(x,\xi): x\in\Omega,\ \lvert\xi\rvert^2 \le \lambda\rbrace$ of phase-space volume $\lvert\Omega\rvert\cdot\alpha(n)\lambda^{n/2}$. So the asymptotics of the spectrum recover the dimension $n$ and the volume $\lvert\Omega\rvert$. Refinements of the same expansion (via the small-$t$ asymptotics of $\operatorname{tr} e^{t\Delta}$) recover the surface area, the Euler characteristic, and more.

<figure>
  <img src="{{ '/assets/images/notes/math_topics/laplacian/lap_spectral_drum.png' | relative_url }}" alt="Top row: the first four Dirichlet vibration modes of a circular drum, shown as red-blue heatmaps with increasing numbers of radial and angular nodal lines and their eigenvalues 5.8, 14.7, 26.4 and 30.5. Bottom: the eigenvalue counting function as a blue staircase tracking the straight dashed Weyl line lambda over four." loading="lazy">
  <figcaption>The spectrum of a drum. <strong>Top:</strong> the lowest Dirichlet eigenmodes $-\Delta\phi_k=\lambda_k\phi_k$ of the unit disk (Bessel modes $J_m(\sqrt{\lambda_k}\,r)\cos m\theta$) — a higher $\lambda_k$ means more nodal lines and a higher overtone. <strong>Bottom:</strong> Weyl's law in action — the counting function $N(\lambda)=\#\lbrace k:\lambda_k\le\lambda\rbrace$ hugs the asymptote $\frac{\lvert\Omega\rvert}{4\pi}\lambda$, so the spectrum audibly encodes the area and dimension of the drum. It rides just below the line because of the negative boundary correction — and, as the next paragraph's Gordon–Webb–Wolpert drums show, the area is heard but the exact shape is not.</figcaption>
</figure>

**Kac's question.** Mark Kac asked in 1966 whether the *entire* spectrum determines $\Omega$ up to congruence—"can one hear the shape of a drum?" Weyl's law says you can hear the volume and dimension; later heat-trace terms add the perimeter and topology. But the full answer is *no*: Gordon, Webb, and Wolpert (1992) exhibited non-congruent planar domains with identical spectra. The map from shape to spectrum is informative but not injective—averaging, it turns out, blurs just enough.

## 8. The dynamical face: heat flow as a Wasserstein gradient flow

I have kept the most structurally surprising reading for last, because it reframes the heat equation in the language of optimal transport and is the gateway to the modern theory of diffusions, Fokker–Planck equations, and score-based generative models.

We saw in §3 that $\Delta u = -\nabla_{L^2} E(u)$: heat flow is gradient descent of the Dirichlet energy in the *flat* $L^2$ geometry. The 1998 insight of Jordan, Kinderlehrer, and Otto (the JKO scheme) is that the very same PDE is *also* gradient descent of a completely different functional in a completely different geometry. View a heat solution as a flow of probability densities $\rho(t,\cdot)$ and equip the space of densities with the **quadratic Wasserstein distance** $W_2$. Then

$$
\partial_t \rho = \Delta\rho \qquad \text{is the } W_2\text{-gradient flow of the Boltzmann entropy } \mathcal{H}(\rho) = \int \rho\log\rho.
$$

**Why the entropy, and why this metric.** The Wasserstein gradient of a functional $\mathcal{F}$ is computed by transporting mass: $\nabla_{W_2}\mathcal{F}(\rho) = -\nabla\!\cdot\!\big(\rho\,\nabla \tfrac{\delta\mathcal F}{\delta\rho}\big)$. For the entropy, $\tfrac{\delta\mathcal H}{\delta\rho} = \log\rho + 1$, so $\nabla\tfrac{\delta\mathcal H}{\delta\rho} = \nabla\rho/\rho$ and

$$
-\nabla_{W_2}\mathcal{H}(\rho) = \nabla\!\cdot\!\Big(\rho\cdot\frac{\nabla\rho}{\rho}\Big) = \nabla\!\cdot\!(\nabla\rho) = \Delta\rho.
$$

So the Laplacian re-emerges, now as the velocity field of steepest entropy *increase* measured in transport cost. Concretely, JKO realises the flow as a sequence of minimising-movement steps

$$
\rho^{k+1} = \arg\min_{\rho}\ \Big\{ \mathcal{H}(\rho) + \frac{1}{2\tau}\,W_2(\rho,\rho^k)^2 \Big\},
$$

whose limit as the step $\tau\to 0$ is exactly the heat equation. The diffusive term that smooths densities and the entropy that the universe loves to maximise are revealed to be the *same* tendency, read in the geometry of how cheaply mass can be rearranged.

<figure>
  <img src="{{ '/assets/images/notes/math_topics/laplacian/lap_wasserstein_flow.png' | relative_url }}" alt="Left: a probability density that starts as two sharp separated peaks and, under the heat equation, spreads and merges into a single broad bump while the total area stays one. Right: the Boltzmann entropy, the integral of rho log rho, plotted against time and decreasing monotonically, annotated with the JKO minimising-movement step." loading="lazy">
  <figcaption>The dynamical face: the same heat equation, read in a different geometry. Viewing a solution as a flow of probability densities, $\partial_t\rho=\Delta\rho$ is the gradient <em>descent</em> of the Boltzmann entropy $\mathcal{H}(\rho)=\int\rho\log\rho$ in the Wasserstein metric $W_2$, because $-\nabla_{W_2}\mathcal{H}(\rho)=\nabla\!\cdot\!(\nabla\rho)=\Delta\rho$. <strong>Left:</strong> two bumps diffuse and merge, mass conserved. <strong>Right:</strong> $\mathcal{H}$ falls monotonically — realised in the limit by the JKO minimising-movement steps that trade entropy against transport cost.</figcaption>
</figure>

**The bridge to SDEs and generative models.** Adding a potential $V$ promotes this to the Fokker–Planck equation $\partial_t\rho = \Delta\rho + \nabla\!\cdot\!(\rho\,\nabla V)$, the $W_2$-gradient flow of the free energy $\int\rho\log\rho + \int V\rho$, and the law of the diffusion $dX_t = -\nabla V(X_t)\,dt + \sqrt{2}\,dB_t$. The $\Delta\rho$ term is the footprint of the Brownian noise from §6; the drift is the potential. This is precisely the forward process underlying diffusion-based generative models, where the reverse-time SDE replaces $\nabla V$ by the learned score $\nabla\log\rho_t$—and where the Laplacian's role as generator of the noise is what makes the whole time-reversal calculus go through.

## Closing remarks: what is really going on, and where it generalises

**One operator, one idea.** Strip away the formalism and every section above is a retelling of §1. Harmonic rigidity is "no averaging defect to correct." Heat smoothing is "averaging, iterated." Self-adjointness and the real spectrum are "averaging is symmetric." Weyl's law is "averaging is sensitive to volume." Even the Wasserstein picture is averaging—now read as the entropy-increasing rearrangement of mass. The Laplacian is what isotropic, local averaging looks like to first non-trivial order, and its uniqueness (§2) is why this one operator keeps reappearing.

**Where it goes next.** On a Riemannian manifold the averaging idea survives verbatim with geodesic spheres, producing the **Laplace–Beltrami operator** 

$$\Delta_g = \tfrac{1}{\sqrt{|g|}}\partial_i(\sqrt{\lvert g\rvert}\,g^{ij}\partial_j)$$

Here the sphere-average expansion acquires a curvature correction—this is the **Bochner formula** 

$$\tfrac12\Delta\lvert\nabla u\rvert^2 = \lvert\nabla^2 u\rvert^2 + \langle\nabla u,\nabla\Delta u\rangle + \operatorname{Ric}(\nabla u,\nabla u)$$

—and that correction is the seed of the Bakry–Émery calculus and the Lott–Sturm–Villani synthetic definition of Ricci curvature *via* the convexity of entropy along Wasserstein geodesics. The §8 picture is not a curiosity; it is how curvature and diffusion are now unified.

Two further generalisations are worth a sentence each. The **$p$-Laplacian** $\Delta_p u = \nabla\!\cdot\!(\lvert\nabla u\rvert^{p-2}\nabla u)$ is the gradient of the $p$-Dirichlet energy and the natural nonlinear cousin (the limit $p\to\infty$ governs optimal Lipschitz extensions). The **fractional Laplacian** $(-\Delta)^s$, defined by the symbol $\lvert\xi\rvert^{2s}$, replaces local averaging by a nonlocal, heavy-tailed average and generates Lévy flights rather than Brownian motion—diffusion that occasionally jumps. In each case the recipe is unchanged: pick an energy or a notion of averaging, and the associated "Laplacian" is its gradient. The operator is less a fixed symbol than a principle.