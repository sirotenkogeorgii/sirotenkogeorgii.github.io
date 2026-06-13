---
layout: default
title: "The Fourier Transform: Diagonalizing Translation"
tags:
  - operator-theory
  - fourier-analysis
  - fourier-transform
  - change-of-basis
---

# The Fourier Transform: Diagonalizing Translation

There is a temptation, when one first meets the Fourier transform, to regard it as a clever integral formula — a machine that eats a function and excretes another function, with some miraculous properties that one memorizes (convolution becomes multiplication, derivatives become polynomials, and so on). This is a serviceable picture, but it inverts the logic. The integral formula is not the *reason* the Fourier transform is useful; it is a *consequence* of a single structural fact, and almost everything else follows from that fact by formal manipulation. The purpose of this note is to put that organizing idea first, derive the standard machinery as corollaries, and then say a few words about what is really going on underneath.

I will work on $\mathbb{R}^d$ throughout, and assume the reader is comfortable with measure theory, $L^p$ spaces, and the basic vocabulary of functional analysis. The emphasis is on ideas and on the *shape* of the proofs; routine estimates are indicated rather than belabored.

## Conventions and normalizations

The single most common source of confusion with the Fourier transform is not mathematical but bureaucratic: there are several incompatible conventions, differing in where one puts the factors of $2\pi$, and formulas that are clean in one convention acquire ugly constants in another. I fix the **analyst's convention** once and for all:

$$
\widehat{f}(\xi) \;=\; \int_{\mathbb{R}^d} f(x)\, e^{-2\pi i\, x\cdot\xi}\, dx, \qquad \xi \in \mathbb{R}^d.
$$

The virtue of placing the $2\pi$ inside the exponent is that the transform becomes a genuine isometry of $L^2$ with *no* prefactor, the inversion formula has *no* prefactor, and Poisson summation pairs the integer lattice with itself. The price is the recurring factors of $2\pi$ in the differentiation rule. Other common choices — $e^{-ix\cdot\xi}$ with a $(2\pi)^{-d/2}$ out front (the "symmetric" convention favored in PDE), or $(2\pi)^{-d}$ on the inverse only (the probabilist's characteristic-function convention) — are obtained from this one by rescaling $\xi$ and are entirely equivalent. The reader should simply *commit to one*; almost all sign errors and constant errors in practice are convention errors in disguise.

I write $\mathcal{F} f = \widehat{f}$ for the transform, $\mathcal{F}^{-1} g = \check{g}$ for its inverse (to be justified below), and $\tau_h f(x) = f(x - h)$ for translation by $h$.

## The one idea: characters diagonalize translation

Here is the fact from which everything flows.

**The Fourier transform is the change of basis that simultaneously diagonalizes every translation operator.**

Let me unpack this, because it is the whole story. Consider the family of translation operators $\lbrace \tau_h\rbrace\_{h \in \mathbb{R}^d}$ acting on functions. These commute with one another ($\tau_h \tau_{h'} = \tau_{h+h'}$), and they are the symmetries of $\mathbb{R}^d$ as a group. A great deal of analysis — convolution, differential operators with constant coefficients, the heat and wave and Schrödinger evolutions, stationary stochastic processes — is built entirely out of operators that *commute with all translations*. Call such an operator **translation-invariant**.

Now, a basic principle of linear algebra: a commuting family of operators can be simultaneously diagonalized, and once you are in the eigenbasis, *every* operator built from the family acts by a scalar on each eigenvector. So we ask: what are the simultaneous eigenfunctions of all the $\tau_h$?

We want $e_\xi$ with $\tau_h e_\xi = \lambda(h)\, e_\xi$ for every $h$, i.e. $e_\xi(x - h) = \lambda(h) e_\xi(x)$. Setting $x = h$ gives $e_\xi(0) = \lambda(x) e_\xi(x)$, so up to normalization $e_\xi(x) = \lambda(-x)$, and the relation $\tau_h\tau_{h'} = \tau_{h+h'}$ forces $\lambda$ to be a homomorphism into the multiplicative group. If we further demand boundedness (so that the eigenfunctions are not exponentially exploding), the only candidates are the **characters**

$$
e_\xi(x) = e^{2\pi i\, x\cdot\xi}, \qquad \tau_h e_\xi = e^{-2\pi i\, h\cdot\xi}\, e_\xi.
$$

These are the joint eigenfunctions, indexed by the frequency $\xi$, with eigenvalue $e^{-2\pi i h\cdot \xi}$ for $\tau_h$. The Fourier transform is nothing more than the operation of writing a function in terms of this eigenbasis: $\widehat{f}(\xi)$ is the "coefficient" of the character $e_\xi$ in $f$, and the inversion formula

$$
f(x) = \int_{\mathbb{R}^d} \widehat{f}(\xi)\, e^{2\pi i x\cdot\xi}\, d\xi
$$

is the assertion that $f$ is the superposition of its frequency components. The continuous index $\xi$ (rather than a discrete sum) appears because $\mathbb{R}^d$ is non-compact; on the circle one gets Fourier *series* instead, and on a finite group the *discrete* Fourier transform. (More on this unification at the end.)

<figure>
  <img src="{{ '/assets/images/notes/math_topics/fourier_transform/ft_characters.png' | relative_url }}" alt="Left: a Gaussian bump and its translate are two different functions, so a bump is not an eigenvector of translation. Right: the real part of a character and its translate are the same wave differing only by a phase, with the eigenvalue drawn as a unit-modulus phasor on the unit circle." loading="lazy">
  <figcaption>Why characters are special. A generic bump (left) is carried to a genuinely different function by $\tau_h$, so it is not an eigenvector. A character $e_\xi$ (right) is only multiplied by the unit-modulus scalar $e^{-2\pi i h\xi}$ — same wave, rotated phase — so it is a simultaneous eigenfunction of every translation. The Fourier transform is the change of basis into these joint eigenvectors.</figcaption>
</figure>

**Everything in the standard dictionary is the statement that some operation is diagonal in this basis.** Let me record the dictionary now and then come back to make the analytic framework rigorous.

| Operation on $f$ | Operation on $\widehat{f}$ | Why |
|---|---|---|
| translation $\tau_h f$ | modulation $e^{-2\pi i h\cdot\xi}\,\widehat{f}(\xi)$ | $\tau_h$ acts by the eigenvalue |
| modulation $e^{2\pi i \eta\cdot x} f$ | translation $\widehat{f}(\xi - \eta)$ | dual to the above |
| dilation $f(\lambda x)$ | $\lambda^{-d}\,\widehat{f}(\xi/\lambda)$ | change of variables |
| differentiation $\partial_{x_j} f$ | $2\pi i\,\xi_j\, \widehat{f}(\xi)$ | $\partial_{x_j}$ is the generator of translation |
| multiplication $x_j f$ | $\tfrac{i}{2\pi}\,\partial_{\xi_j}\widehat{f}(\xi)$ | dual to the above |
| convolution $f * g$ | $\widehat{f}\,\widehat{g}$ | convolution is "$\sum$ of translates" |
| multiplication $f g$ | $\widehat{f} * \widehat{g}$ | dual to the above |

The two entries that carry the real weight are **differentiation $\mapsto$ multiplication** and **convolution $\mapsto$ multiplication**, and they are the same fact wearing different clothes. A constant-coefficient differential operator $P(\partial) = \sum_\alpha c_\alpha \partial^\alpha$ is translation-invariant, hence diagonal in the character basis; its eigenvalue on $e_\xi$ is the **symbol** $P(2\pi i \xi)$, so $\widehat{P(\partial) f}(\xi) = P(2\pi i\xi)\,\widehat{f}(\xi)$. This single observation reduces every constant-coefficient linear PDE to an algebraic equation, frequency by frequency. The cost of solving the PDE has been entirely repackaged as the cost of computing a Fourier transform. That trade is the reason the subject exists.

<figure>
  <img src="{{ '/assets/images/notes/math_topics/fourier_transform/ft_dictionary.png' | relative_url }}" alt="Left: the magnitude of f-hat and of the transform of its derivative, which equals 2 pi absolute-xi times f-hat, so high frequencies are amplified and the value at zero is suppressed. Right: the transforms of two Gaussians and their pointwise product, which is the transform of their convolution." loading="lazy">
  <figcaption>The two weight-bearing entries, each an instance of "translation-invariant $\Rightarrow$ diagonal." Left: differentiation multiplies $\widehat{f}$ by $2\pi i\xi$, amplifying high frequencies and killing the mean at $\xi=0$. Right: convolution becomes the pointwise product $\widehat{f}\,\widehat{g}$, so convolving in space narrows the transform in frequency.</figcaption>
</figure>

## Four homes for the transform

The formula $\widehat{f}(\xi) = \int f(x) e^{-2\pi i x\cdot\xi}\,dx$ requires the integral to converge, and the various good properties we want — boundedness, inversion, the Plancherel isometry — hold on different function spaces. There is a small ladder of spaces, each the natural domain for a different part of the theory, and a recurring strategy: *prove the identity on a dense, well-behaved subspace where everything converges absolutely, then extend by continuity.*

<figure>
  <img src="{{ '/assets/images/notes/math_topics/fourier_transform/ft_space_ladder.png' | relative_url }}" alt="A nested diagram: the largest region is the tempered distributions S-prime, containing the Dirac delta, polynomials, and characters; inside it two overlapping regions L1 and L2; and at their intersection the small Schwartz core S. A legend lists what the Fourier transform does on each: homeomorphism on S-prime, unitary on L2, isomorphism on S, and bounded-into on L1." loading="lazy">
  <figcaption>The four homes, and the recurring strategy. The transform is pinned down on the small self-dual core $\mathcal{S}$, where every integral converges absolutely, then pushed outward by continuity and duality: unitary on $L^2$, a homeomorphism of the tempered distributions $\mathcal{S}'$, and merely bounded into $C_0$ on $L^1$ (not onto). Objects like $\delta$, the constant $1$, and polynomials live only in $\mathcal{S}'$.</figcaption>
</figure>

### 1. $L^1$: where the integral makes literal sense

If $f \in L^1(\mathbb{R}^d)$ the defining integral converges absolutely for every $\xi$, and we read off immediately that $\|\widehat{f}\|_{\infty} \le \|f\|_1$ and that $\widehat{f}$ is continuous (dominated convergence). The substantive fact at this level is:

**Riemann–Lebesgue.** If $f \in L^1$ then $\widehat{f} \in C_0(\mathbb{R}^d)$; that is, $\widehat{f}$ is continuous and $\widehat{f}(\xi) \to 0$ as $\lvert\xi\rvert \to \infty$.

*Key idea:* the result is obvious for a smooth bump (integrate by parts to gain decay), and these are dense in $L^1$, so the uniform bound $\|\widehat{f}\|_\infty \le \|f\|_1$ propagates the decay to all of $L^1$. The cleanest mechanical proof uses the translation identity: $e^{-2\pi i h\cdot\xi}\widehat f(\xi) = \widehat{\tau_h f}(\xi)$, and choosing $h = \xi/(2\lvert \xi\rvert^2)$ makes the exponential equal $-1$, so $2\widehat f(\xi) = \widehat{(f - \tau_h f)}(\xi)$; since $\|f - \tau_h f\|_1 \to 0$ as $h \to 0$ (continuity of translation in $L^1$), the right side is small for large $\lvert\xi\rvert$. $\square$

<figure>
  <img src="{{ '/assets/images/notes/math_topics/fourier_transform/ft_riemann_lebesgue.png' | relative_url }}" alt="Left: a box indicator, a continuous tent function, and a smooth Gaussian. Right: the magnitudes of their Fourier transforms on a logarithmic scale; the box decays like one over xi, the tent like one over xi squared, and the Gaussian decays exponentially. All tend to zero." loading="lazy">
  <figcaption>Riemann–Lebesgue, and the smoothness–decay dictionary in one picture. The rougher the input, the slower $\widehat{f}$ decays: the jump-discontinuous box gives $\lvert\widehat{f}\rvert\sim 1/\lvert\xi\rvert$, the continuous tent gives $\sim 1/\xi^2$, and the smooth Gaussian decays faster than any polynomial — but all tend to $0$.</figcaption>
</figure>

The map $\mathcal{F}: L^1 \to C_0$ is bounded and injective but, importantly, **not surjective**: its image is a dense proper subspace of $C_0$. So $L^1$ is *not* the natural home of the transform — there is no clean inversion within it, because $\widehat{f}$ need not be integrable. We need a space the transform maps *onto itself*.

### 2. Schwartz space: the natural home

That space is the **Schwartz class** $\mathcal{S}(\mathbb{R}^d)$ of smooth functions all of whose derivatives decay faster than any polynomial:

$$
\mathcal{S} = \Big\{ f \in C^\infty : \ \sup_x |x^\alpha \partial^\beta f(x)| < \infty \text{ for all multi-indices } \alpha, \beta \Big\}.
$$

The point of this definition is that it is *exactly* self-dual under the dictionary above. The two key entries say that multiplying by $x_j$ corresponds to differentiating $\widehat f$, and differentiating $f$ corresponds to multiplying $\widehat f$ by $\xi_j$. Decay of $f$ controls smoothness of $\widehat f$, and smoothness of $f$ controls decay of $\widehat f$. Schwartz space demands *both* decay and smoothness, to all orders, and is therefore preserved:

**$\mathcal{F}$ maps $\mathcal{S}$ continuously into $\mathcal{S}$.** Indeed $\xi^\alpha \partial_\xi^\beta \widehat f$ is, up to constants, the Fourier transform of $\partial^\alpha(x^\beta f)$, which is in $L^1$; so $\widehat f$ has all the required bounded derivatives. The transform is in fact a *topological isomorphism* of $\mathcal{S}$, once we establish inversion.

**Inversion on $\mathcal{S}$.** The clean way to prove the inversion formula avoids fighting with the non-absolutely-convergent integral $\int \widehat f(\xi) e^{2\pi i x\xi}\,d\xi$ directly. Two ingredients:

*(i) The multiplication formula.* For $f, g \in L^1$, Fubini gives the symmetric identity

$$
\int_{\mathbb{R}^d} \widehat{f}(\xi)\, g(\xi)\, d\xi \;=\; \int_{\mathbb{R}^d} f(x)\, \widehat{g}(x)\, dx,
$$

both sides being equal to the double integral $\iint f(x) g(\xi) e^{-2\pi i x\cdot\xi}\,dx\,d\xi$. This is the workhorse: it lets us move the transform off of $f$ and onto a function $g$ of our choosing.

*(ii) A Gaussian test function we can transform by hand.* Take $g(\xi) = e^{2\pi i x_0\cdot\xi}\, e^{-\pi \epsilon^2 \lvert \xi\rvert^2}$. Using the self-duality of the Gaussian (proved in the next section) together with the modulation and dilation rules, its inverse transform $\widehat{g}$ is an explicit Gaussian of width $\sim \epsilon$ centered at $x_0$ — that is, $\widehat{g}(x) = \epsilon^{-d} e^{-\pi \lvert x - x_0\rvert^2/\epsilon^2}$, a member of an **approximate identity** $\phi_\epsilon$ as $\epsilon \to 0$.

Feeding these into the multiplication formula:

$$
\int \widehat{f}(\xi)\, e^{2\pi i x_0\cdot\xi}\, e^{-\pi\epsilon^2|\xi|^2}\, d\xi \;=\; \int f(x)\, \phi_\epsilon(x - x_0)\, dx \;=\; (f * \phi_\epsilon)(x_0).
$$

Now let $\epsilon \to 0$. On the left, the Gaussian cutoff increases to $1$ and dominated convergence gives $\int \widehat f(\xi) e^{2\pi i x_0\cdot\xi}\,d\xi$ (this is legitimate because $\widehat f \in L^1$ for Schwartz $f$). On the right, convolution against an approximate identity recovers $f(x_0)$. Hence

$$
\boxed{\,f(x) = \int_{\mathbb{R}^d} \widehat{f}(\xi)\, e^{2\pi i x\cdot\xi}\, d\xi\,} \qquad (f \in \mathcal{S}).
$$

This is the precise sense in which the characters $e_\xi$ form a "continuous basis." Note also the elegant operator identity hiding here: applying $\mathcal F$ twice gives $\mathcal{F}^2 f(x) = f(-x)$ (the inverse transform is the forward transform composed with reflection), so $\mathcal{F}^4 = \mathrm{Id}$. The Fourier transform has order four. We will exploit this.

### 3. $L^2$: the Plancherel isometry

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Plancherel / Parseval)</span></p>

Schwartz functions are dense in $L^2$, and on them the transform is an **isometry**. For $f, g \in \mathcal{S}$,

$$
\int_{\mathbb{R}^d} f\, \overline{g}\;dx \;=\; \int_{\mathbb{R}^d} \widehat{f}\,\overline{\widehat{g}}\;d\xi, \qquad\text{in particular}\qquad \|\widehat{f}\|_{L^2} = \|f\|_{L^2}.
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Apply the multiplication formula with $g$ replaced by $\overline{\widehat{g}}$. One checks from the definition that $\widehat{\,\overline{\widehat{g}}\,} = \overline{g}$ (this is exactly the inversion formula together with conjugation symmetry $\overline{\widehat{g}(\xi)} = \widehat{\overline g}(-\xi)$), and the formula collapses to $\int \widehat f\, \overline{\widehat g} = \int f\,\overline g$. Setting $g = f$ gives the norm identity. $\square$

</details>
</div>

Because $\mathcal{F}$ is a linear isometry on the dense subspace $\mathcal{S} \subset L^2$, it extends uniquely to a **unitary operator on all of $L^2(\mathbb{R}^d)$**. This extension no longer agrees with the integral formula pointwise — for $f \in L^2 \setminus L^1$ the integral need not converge — but it is defined as the $L^2$-limit of $\widehat{f_n}$ for any Schwartz approximants $f_n \to f$, and one often realizes it concretely as $\widehat f(\xi) = \lim_{R\to\infty}\int_{\lvert x\rvert\le R} f(x) e^{-2\pi i x\xi}\,dx$ (limit in $L^2$). The upshot is the structurally cleanest statement in the whole subject:

**The Fourier transform is a unitary automorphism of $L^2(\mathbb{R}^d)$.**

Since it is unitary and satisfies $\mathcal{F}^4 = \mathrm{Id}$, its spectrum consists of the four fourth-roots of unity $\lbrace 1, i, -1, -i\rbrace$, and $L^2$ splits into the four corresponding eigenspaces. We will identify an explicit orthonormal eigenbasis (the Hermite functions) below. This is the spectral-theoretic heart of the matter: the Fourier transform is not just *a* unitary operator, it is one whose eigenstructure is completely understood.

### 4. Tempered distributions: where it all lives comfortably

Finally, duality extends the transform far beyond functions. The dual space $\mathcal{S}'$ of **tempered distributions** consists of continuous linear functionals on $\mathcal{S}$; it contains every $L^p$ function, every polynomial, the Dirac delta, and much else. We *define* $\widehat{u}$ for $u \in \mathcal{S}'$ by transposing the multiplication formula:

$$
\langle \widehat{u}, \varphi\rangle := \langle u, \widehat{\varphi}\rangle, \qquad \varphi \in \mathcal{S}.
$$

Because $\mathcal{F}: \mathcal{S} \to \mathcal{S}$ is a homeomorphism, this makes $\mathcal{F}: \mathcal{S}' \to \mathcal{S}'$ a homeomorphism too, consistent with all the previous definitions. Now $\widehat{\delta} = 1$ and $\widehat{1} = \delta$ (the constant function and the point mass are Fourier-dual — frequencies and positions at their most extreme), $\widehat{e^{2\pi i \eta\cdot x}} = \delta_\eta$, and every constant-coefficient PDE can be analyzed in this setting via *fundamental solutions* $P(\partial) E = \delta$, i.e. $\widehat{E} = 1/P(2\pi i \xi)$. This is the natural home of the transform for PDE and for the theory of pseudodifferential and Fourier integral operators.

## The Gaussian: fixed point, heat flow, and the eigenbasis

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">()</span></p>

Among all functions, one is singled out by the transform.

**The Gaussian $g(x) = e^{-\pi \lvert x\rvert^2}$ is a fixed point: $\widehat{g} = g$.**

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Proof (one dimension; the general case factorizes).* The function $g(x) = e^{-\pi x^2}$ solves the first-order ODE $g'(x) = -2\pi x\, g(x)$. Transform both sides using the dictionary: $\widehat{g'} = 2\pi i\xi\, \widehat g$ on the left, and $\widehat{x g} = \frac{i}{2\pi}\widehat g{}'$ so $\widehat{-2\pi x g} = -i\,\widehat g{}'$ on the right. The ODE becomes $2\pi i \xi\, \widehat g = -i\,\widehat g{}'$, i.e. $\widehat g{}'(\xi) = -2\pi\xi\,\widehat g(\xi)$ — *the same ODE*. Hence $\widehat g = c\,g$, and evaluating at $\xi = 0$ gives $c = \widehat g(0) = \int e^{-\pi x^2}dx = 1$. So $\widehat g = g$. $\square$

</details>
</div>

This is the reason the Gaussian is *the* universal smoothing kernel, the reason it is the steady profile of diffusion, and the reason it minimizes the uncertainty principle. Let me draw out the first two.

<figure>
  <img src="{{ '/assets/images/notes/math_topics/fourier_transform/ft_gaussian.png' | relative_url }}" alt="Left: the Gaussian e to the minus pi x squared and its Fourier transform plotted on top of each other, identical. Right: three Gaussians of different widths, solid, with their transforms dashed; a narrow Gaussian in x has a wide transform and vice versa, while the width-one Gaussian coincides with its own transform." loading="lazy">
  <figcaption>The Gaussian is the fixed point $\widehat{g}=g$ (left), the one profile identical in position and frequency. The dilation rule (right) shows the duality around it: narrowing $f$ in $x$ widens $\widehat{f}$ in $\xi$, with $s=1$ the self-dual balance point.</figcaption>
</figure>

**Heat flow is diagonal in frequency, and the Gaussian is what you see.** Consider the heat equation $\partial_t u = \Delta u$, $u(0,\cdot) = u_0$. Transforming in $x$ and using $\widehat{\Delta u} = -4\pi^2\lvert\xi\rvert^2\widehat u$ turns the PDE into the decoupled family of ODEs $\partial_t \widehat u(t,\xi) = -4\pi^2\lvert\xi\rvert^2\,\widehat u(t,\xi)$, one for each frequency, solved instantly by

$$
\widehat u(t,\xi) = e^{-4\pi^2 t|\xi|^2}\,\widehat{u_0}(\xi).
$$

Each frequency simply decays exponentially, the faster the higher the frequency — diffusion is a low-pass filter. Inverting the (Gaussian!) multiplier gives $u(t,\cdot) = u_0 * K_t$ with the **heat kernel**

$$
K_t(x) = (4\pi t)^{-d/2}\, e^{-|x|^2/(4t)}.
$$

The smoothing, the infinite propagation speed, the $t^{-d/2}$ decay rate — all of it is read off from the Gaussian multiplier with no further work. The same template handles the Schrödinger equation (multiplier $e^{-4\pi^2 i t\lvert\xi\rvert^2}$, a *unitary* phase rather than a decay, hence dispersion without smoothing) and the wave equation (multiplier $\cos(2\pi t\lvert\xi\rvert)$, supported on the light cone).

<figure>
  <img src="{{ '/assets/images/notes/math_topics/fourier_transform/ft_heat_flow.png' | relative_url }}" alt="Left: a rough initial profile made of six cosine modes that progressively smooths into a nearly flat curve as time increases. Right: the Gaussian multiplier e to the minus four pi squared t xi squared, which is flat at one for t equal zero and narrows as t grows, suppressing high frequencies." loading="lazy">
  <figcaption>Heat flow is a low-pass filter, diagonal in frequency. Each mode of the initial data is scaled by $e^{-4\pi^2 t\lvert\xi\rvert^2}$ (right), so high frequencies die first and the spatial profile smooths (left). The Schrödinger and wave equations replace this real-decaying multiplier by a unit-modulus phase.</figcaption>
</figure>

**Hermite functions diagonalize the transform.** Define the rescaled creation/annihilation operators $A = x + \frac{1}{2\pi}\partial_x$ and $A^\ast = x - \frac{1}{2\pi}\partial_x$ on $\mathbb{R}$. The Gaussian $h_0 = \sqrt2\, e^{-\pi x^2}$ (normalized in $L^2$) is annihilated by $A$, and the **Hermite functions** $h_n = c_n (A^\ast)^n h_0$ form an orthonormal basis of $L^2(\mathbb{R})$. A short computation using the dictionary shows that conjugating $\partial_x$ by $\mathcal F$ turns it into multiplication by $2\pi i \xi$ and vice versa, so $\mathcal F$ intertwines $A^\ast$ with $-i A^\ast$; consequently each Hermite function is an eigenfunction,

$$
\mathcal{F} h_n = (-i)^n\, h_n.
$$

This is the explicit spectral decomposition promised earlier: the eigenvalue $(-i)^n$ cycles through $\lbrace 1, -i, -1, i\rbrace$ as $n$ runs through residues mod $4$, recovering $\mathcal{F}^4 = \mathrm{Id}$, and the ground state $h_0$ (the Gaussian) is the fixed point we found by hand. The harmonic oscillator Hamiltonian $-\frac{1}{4\pi^2}\partial_x^2 + x^2$, whose eigenfunctions these are, commutes with $\mathcal F$ — which is the structural reason a single basis diagonalizes both.

<figure>
  <img src="{{ '/assets/images/notes/math_topics/fourier_transform/ft_hermite.png' | relative_url }}" alt="The first four Hermite functions plotted under a shared dotted Gaussian envelope: h0 has no zeros, h1 one, h2 two, h3 three. Each is labelled with its Fourier eigenvalue plus one, minus i, minus one, plus i, cycling with period four." loading="lazy">
  <figcaption>The explicit eigenbasis. Each Hermite function $h_n$ is an eigenfunction of $\mathcal{F}$ with eigenvalue $(-i)^n$, cycling through $\lbrace 1,-i,-1,i\rbrace$ as $n$ runs mod $4$ — the spectral decomposition behind $\mathcal{F}^4=\mathrm{Id}$. The ground state $h_0$ is the Gaussian fixed point; the dotted curve is the envelope $e^{-x^2/2}$.</figcaption>
</figure>

## The uncertainty principle

A function and its transform cannot both be sharply localized. This is not quantum mysticism; it is a theorem about the incompatibility of the position operator $x$ and the frequency operator $\frac{1}{2\pi i}\partial_x$, and the Gaussian saturates it.

**Heisenberg's inequality.** For $f \in \mathcal{S}(\mathbb{R})$ with $\|f\|_{L^2} = 1$,

$$
\left(\int_{\mathbb{R}} x^2\,|f(x)|^2\,dx\right)\left(\int_{\mathbb{R}} \xi^2\,|\widehat f(\xi)|^2\,d\xi\right) \;\ge\; \frac{1}{16\pi^2},
$$

with equality if and only if $f$ is a Gaussian $C e^{-a x^2}$, $a > 0$.

*Key idea:* integrate by parts to express the total mass as a position–momentum pairing, then apply Cauchy–Schwarz. Concretely, since $\frac{d}{dx}(x\lvert f\rvert^2) = \lvert f\rvert^2 + x(f'\bar f + f\bar f')$ and the boundary terms vanish by decay,

$$
1 = \int |f|^2\,dx = -\int x\,(f'\bar f + \overline{f'\bar f})\,dx = -2\,\mathrm{Re}\!\int x f'\,\bar f\,dx \le 2\,\|xf\|_{2}\,\|f'\|_{2}
$$

by Cauchy–Schwarz. Now Plancherel converts the derivative norm to a frequency moment: $\|f'\|_2 = \|\widehat{f'}\|_2 = \|2\pi i\xi\,\widehat f\|_2 = 2\pi\,\|\xi\widehat f\|_2$. Hence $1 \le 4\pi\,\|xf\|_2\,\|\xi\widehat f\|_2$, which is the claim after squaring. Equality requires equality in Cauchy–Schwarz, i.e. $f'(x) = -2a x\, f(x)$ for a constant $a$, whose only $L^2$ solutions are Gaussians. $\square$

Two remarks. First, the *content* of the proof is the commutator $[\,x,\,\tfrac{1}{2\pi i}\partial_x\,] = \tfrac{1}{2\pi i}\mathrm{Id} \ne 0$; the inequality is the analytic shadow of the operators failing to commute, and the same Cauchy–Schwarz argument with any pair of non-commuting self-adjoint operators is the general Robertson uncertainty relation. Second, the Gaussian appears as the extremizer for the same reason it appeared as the fixed point: it is the unique state in which position and frequency localization are perfectly balanced, the harmonic oscillator ground state.

<figure>
  <img src="{{ '/assets/images/notes/math_topics/fourier_transform/ft_uncertainty.png' | relative_url }}" alt="Left: a narrow Gaussian with a wide transform and a wide Gaussian with a narrow transform, illustrating the trade-off. Right: the hyperbola delta-x times delta-xi equal to one over four pi, with the region below it shaded as forbidden; several Gaussians sit exactly on the curve and a tent function sits above it." loading="lazy">
  <figcaption>The uncertainty trade-off. Squeezing $f$ in position spreads $\widehat{f}$ in frequency (left). The spread product obeys $\Delta x\,\Delta\xi\ge 1/4\pi$ (right): the shaded region is forbidden, Gaussians saturate the bound (dots on the curve), and every other state — such as the tent function — sits strictly above it.</figcaption>
</figure>

## A computation that ties the lattice to itself

One identity deserves singling out because it is the engine of analytic number theory, sampling theory, and the theory of theta functions.

**Poisson summation.** For $f \in \mathcal{S}(\mathbb{R}^d)$,

$$
\sum_{n \in \mathbb{Z}^d} f(n) \;=\; \sum_{k \in \mathbb{Z}^d} \widehat{f}(k).
$$

*Key idea:* periodize and expand in Fourier series. The function $F(x) = \sum_{n} f(x + n)$ is $\mathbb{Z}^d$-periodic and smooth, so it equals its Fourier series $F(x) = \sum_k c_k e^{2\pi i k\cdot x}$; computing the coefficient $c_k = \int_{[0,1]^d} F(x) e^{-2\pi i k\cdot x}\,dx$ and unfolding the periodization shows $c_k = \widehat f(k)$. Evaluating $F(0)$ both ways gives the identity. $\square$

The aesthetic payoff of the analyst's convention is on full display: the integer lattice is its *own* dual, with no stray constants. Applied to the Gaussian heat kernel this identity produces the **Jacobi theta transformation** $\theta(1/t) = \sqrt{t}\,\theta(t)$, which is the functional equation underlying the analytic continuation of the Riemann zeta function — a strikingly arithmetic consequence of the self-duality of the Gaussian. Applied to band-limited functions it yields the **Shannon sampling theorem**: a signal with no frequencies above $W$ is determined by its samples at spacing $1/(2W)$, because aliasing is precisely the overlapping of translated copies of $\widehat f$ in the Poisson sum.

<figure>
  <img src="{{ '/assets/images/notes/math_topics/fourier_transform/ft_poisson.png' | relative_url }}" alt="Left: a single bump and its integer translates summed into a periodic function. Right top: a band-limited spectrum sampled fast enough that its replicated copies stay separate. Right bottom: the same spectrum sampled too slowly, so the copies overlap, with the overlap shaded — aliasing." loading="lazy">
  <figcaption>Poisson summation, and its two faces. Periodizing $f$ by summing its integer translates (left) makes $\sum_n f(n)=\sum_k \widehat{f}(k)$. Dually, sampling replicates the spectrum: if the sampling rate exceeds $2W$ the copies stay separate and the signal is recoverable (Shannon); if not, they overlap and alias (right).</figcaption>
</figure>

## Where this shows up: a few bridges

Because translation-invariance is everywhere, so is the Fourier transform. A handful of connections worth keeping in view:

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Stationary processes and Karhunen–Loève)</span></p>

A wide-sense stationary process has a covariance $K(x,y) = k(x - y)$ depending only on the displacement — that is, the covariance operator is a *convolution*, hence translation-invariant, hence diagonalized by the characters. Its eigenfunctions are therefore the Fourier modes, and the eigenvalues are the values of $\widehat{k}$. This is the **Wiener–Khinchin theorem**: the power spectral density is the Fourier transform of the autocovariance, and the Karhunen–Loève expansion of a stationary process is (essentially) its Fourier series, with the spectral density playing the role of the eigenvalue sequence. The Fourier basis is thus not an arbitrary analytic choice but the *intrinsic* optimal basis for any stationarity-respecting problem.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Positive-definite kernels (Bochner))</span></p>

Closely related, and the reason Gaussians are the default covariance kernel in spatial statistics and machine learning, is **Bochner's theorem**: a continuous function $\phi:\mathbb{R}^d \to \mathbb{C}$ is *positive-definite* — meaning $\sum_{j,k} s_j \overline{s_k}\,\phi(x_j - x_k) \ge 0$ for all finite node sets and all coefficients — *if and only if* it is the Fourier transform of a finite non-negative measure $\nu$, i.e. $\phi(t) = \int_{\mathbb{R}^d} e^{2\pi i\langle\xi, t\rangle}\,d\nu(\xi)$. One direction is a one-line computation and is the half used in practice: substituting the integral representation and interchanging the (finite) sum with the integral,


$$
\sum_{j,k} s_j \overline{s_k}\,\phi(x_j - x_k) = \int_{\mathbb{R}^d}\Big|\sum_{j} s_j\, e^{2\pi i\langle\xi, x_j\rangle}\Big|^2 d\nu(\xi)\ \ge\ 0,
$$

since the integrand is a modulus squared against a positive measure. The harder converse (positive-definiteness *forces* the spectral measure to be positive) is the substantive content, and it is exactly the statement that the autocovariance of a stationary process has a non-negative power spectrum — Bochner and Wiener–Khinchin are the same theorem read in the two directions. The **Gaussian kernel** 

$$g(t) = e^{-\beta\|t\|^2}$$ 

is the cleanest instance: its transform is again a (positive) Gaussian, so the spectral measure $\nu$ is a genuine rescaled Gaussian density, which has *full support*. Full support upgrades the conclusion to *strict* positive-definiteness for distinct nodes — the integrand above is a trigonometric polynomial in $\xi$, and a non-trivial one cannot vanish $\nu$-almost everywhere when $\nu$ charges every open set, so the quadratic form is zero only for $s = 0$. This is precisely the structural fact that makes Gaussian-kernel interpolation matrices invertible. (As always, watch the convention: in the symmetric $e^{-i\langle\xi,t\rangle}$ normalization the same computation produces 

$$\widehat{g}(\xi) = \tfrac{\pi}{\beta}\,e^{-\|\xi\|^2/(4\beta)}$$ 

with its prefactor and width rescaling, rather than the bare fixed point $\widehat g = g$ of the analyst's convention — the positivity, and hence the conclusion, is convention-independent.)

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Lévy processes)</span></p>

The transition semigroup of a Lévy process commutes with translations, so it is a Fourier multiplier $e^{t\psi(\xi)}$; the symbol $\psi$ is exactly the **Lévy–Khintchine exponent**, decomposing into a drift (first-order), a Gaussian part ($-c\lvert\xi\rvert^2$, the diffusion), and a jump integral. The characteristic function of the process — its Fourier transform — *is* the natural object, and infinite divisibility becomes the statement that $\psi$ is well-defined.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Inverse problems)</span></p>

Deconvolution, deblurring, and many forward models are convolutions; in frequency they become *division* by the symbol $\widehat{g}$. The reason such problems are ill-posed is visible at a glance: a smoothing kernel has $\widehat g(\xi) \to 0$ at high frequency, so inverting it amplifies high-frequency noise without bound. Regularization (Tikhonov, spectral cutoffs) is precisely the prescription for how to truncate or temper the division $1/\widehat g$ in the unstable band — an analysis carried out entirely in the Fourier-diagonalized picture.

</div>

## What is really going on: harmonic analysis on groups

Step back from $\mathbb{R}^d$ and the whole structure clarifies. The only feature of $\mathbb{R}^d$ we used is that it is a **locally compact abelian (LCA) group**: it has an associative commutative addition, a translation-invariant (Haar) measure, and a topology compatible with both. For any LCA group $G$, the **dual group** $\widehat{G}$ is the set of continuous characters $\chi: G \to \mathbb{T}$ (homomorphisms to the unit circle), itself an LCA group, and the Fourier transform is the pairing

$$
\widehat{f}(\chi) = \int_G f(x)\,\overline{\chi(x)}\,dx, \qquad \chi \in \widehat{G}.
$$

The examples now line up as instances of one theorem:

- $G = \mathbb{R}^d$ has $\widehat{G} \cong \mathbb{R}^d$ (the characters are $x \mapsto e^{2\pi i x\cdot\xi}$): the Fourier *transform*.
- $G = \mathbb{T}^d$ (the torus) has $\widehat{G} \cong \mathbb{Z}^d$: Fourier *series*.
- $G = \mathbb{Z}^d$ has $\widehat{G} \cong \mathbb{T}^d$: the dual statement.
- $G = \mathbb{Z}/N$ has $\widehat{G} \cong \mathbb{Z}/N$: the *discrete* Fourier transform (and the FFT is the algorithm that exploits the factorization of $N$).

<figure>
  <img src="{{ '/assets/images/notes/math_topics/fourier_transform/ft_lca_groups.png' | relative_url }}" alt="A two-by-four grid of signal-and-transform pairs. Column R to R: a continuous aperiodic Gaussian maps to a continuous aperiodic Gaussian. Column T to Z: a continuous periodic signal maps to a discrete spectrum. Column Z to T: a discrete signal maps to a continuous periodic spectrum. Column Z mod N to Z mod N: a finite discrete signal maps to a finite discrete spectrum." loading="lazy">
  <figcaption>One theorem on four groups. The pairing $\widehat{f}(\chi)=\int_G f\,\overline{\chi}$ specializes to the transform on $\mathbb{R}$, Fourier series on the circle $\mathbb{T}$ (discrete spectrum), its dual on $\mathbb{Z}$ (periodic spectrum), and the DFT on $\mathbb{Z}/N$. Continuous $\leftrightarrow$ continuous, periodic $\leftrightarrow$ discrete, finite $\leftrightarrow$ finite — the self-duality $\widehat{\widehat{G}}\cong G$ row by row.</figcaption>
</figure>

**Pontryagin duality** is the statement that $\widehat{\widehat{G}} \cong G$ canonically — the source of every inversion formula above. Plancherel is the statement that $\mathcal F: L^2(G) \to L^2(\widehat{G})$ is unitary for suitably normalized Haar measures. And the convolution-to-multiplication law is the statement that $\mathcal{F}$ is an *algebra* homomorphism from the convolution algebra $L^1(G)$ to an algebra of functions on $\widehat G$ — in the language of operator algebras, the **Gelfand transform** of the commutative Banach algebra $L^1(G)$, whose maximal ideal space is precisely $\widehat G$. The Fourier transform, from this height, is the assertion that a commutative harmonic-analytic object is determined by its characters.

The non-abelian world is the natural sequel: when $G$ is a non-commutative group, characters no longer suffice and one must replace them with irreducible *representations* of varying dimension. The Fourier transform becomes the **Peter–Weyl decomposition** (for compact $G$) or the Plancherel theorem for the unitary dual, and "diagonalizing translation" generalizes to "decomposing the regular representation into irreducibles." But that, as they say, is another story.

### Two closing remarks

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(On what the transform "is.")</span></p>

If you remember one thing, let it be that the Fourier transform is a **change of basis to the joint eigenvectors of translation**, and that every clean formula in the subject is the statement that some translation-invariant operation acts diagonally in that basis. The integral $\int f e^{-2\pi i x\xi}$ is the bookkeeping; the eigenbasis is the idea.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(On the recurring Gaussian)</span></p>

It is not a coincidence that the Gaussian was the fixed point of $\mathcal F$, the kernel of the heat flow, the ground state of the harmonic oscillator, and the extremizer of the uncertainty principle. All four are facets of a single fact: the Gaussian is the unique function that "looks the same" in position and in frequency, the point of perfect balance for the position–frequency duality that the whole theory is organized around. When the Gaussian appears — and it appears everywhere — it is the Fourier transform telling you that you have found the symmetric point of some translation-invariant structure.

</div>
