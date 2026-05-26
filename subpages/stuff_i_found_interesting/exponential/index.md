---
layout: note
title: "Why the exponential function is everywhere"
date: 2024-10-20
math: true
---

# Why the exponential function is everywhere

> **Personal notes.** The recurring puzzle: the same function $\exp$ shows up in growth and decay, in Euler's identity, in linear ODEs, in the Gaussian, in softmax and the sigmoid — fields with little obvious kinship. Such coincidences usually signal a single structure being met through different doors. Here is the structure.

**Conventions.** $\exp$ denotes the function; $\lambda$ a scalar that may be real, complex, or an operator; $t$ the additive parameter we usually call time. *Continuity* (in fact mere measurability) is the standing regularity assumption throughout — it is the minimal hypothesis that removes the pathologies, and every genuine application supplies it for free.

**The thesis in one line.** The exponential is the unique continuous bridge between *additive* structure and *multiplicative* structure. Whenever a local, additive parameter — time, angle, translation, energy, log-probability — drives a global object by composition, normalization, or repeated action, the exponential is the map that performs the conversion. It is not a special function that happens to recur; it is the unique solution to a structural problem that recurs.

## The structural fact

Forget calculus for a moment. The defining property is the functional equation

$$
\exp(a+b) = \exp(a)\,\exp(b), \tag{1}
$$

which says exactly that $\exp$ is a homomorphism $(\mathbb{R},+)\to(\mathbb{R}\_{>0},\times)$: it converts addition into multiplication.

<figure>
  <img src="{{ '/assets/images/notes/stuff_i_found_interesting/exponential/additive_multiplicative_bridge.svg' | relative_url }}" alt="Diagram showing additive increments becoming multiplicative exponential factors" loading="lazy">
  <figcaption>The whole story in one picture: splitting an additive input into $a+b$ is the same as splitting the corresponding exponential effect into two multiplicative factors.</figcaption>
</figure>

The sharp statement is **Cauchy's theorem**:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Cauchy's theorem)</span></p>

The maps $f:\mathbb R\to\mathbb R_{>0}$ satisfying $f(x+y)=f(x)f(y)$, together with any standard regularity hypothesis — continuity, measurability, or boundedness on a set of positive measure — are exactly

$$
f(x) = \exp(\lambda x), \qquad \lambda = f'(0). \tag{2}
$$

</div>

<details class="proof" markdown="1">
<summary>Proof</summary>

In the homomorphism setting above take $f:\mathbb R\to\mathbb R_{>0}$. Then

$$
f(0)=f(0+0)=f(0)^2,
$$

so $f(0)=1$. Also

$$
f(x)=f(x/2+x/2)=f(x/2)^2>0.
$$

Thus $g(x)=\log f(x)$ is well-defined, and the multiplicative equation becomes
the additive Cauchy equation

$$
g(x+y)=g(x)+g(y).
$$

For integers $n$, $g(nx)=ng(x)$, and hence for rationals $q=m/n$,

$$
g(q)=qg(1).
$$

If $x\in\mathbb R$, choose rationals $q_k\to x$. Continuity gives

$$
g(x)=\lim_{k\to\infty}g(q_k)=\lim_{k\to\infty}q_k g(1)=xg(1).
$$

Writing $\lambda=g(1)$, we get

$$
f(x)=e^{g(x)}=e^{\lambda x}.
$$

Conversely, every function $x\mapsto e^{\lambda x}$ plainly satisfies
$f(x+y)=f(x)f(y)$. Differentiating at $0$ gives $f'(0)=\lambda$. The measurable
and locally bounded versions reduce to the same conclusion by the standard
regularity theorem for the additive Cauchy equation applied to $g=\log f$.

</details>

**The regularity is doing real work.** Drop it entirely and the axiom of choice manufactures monstrous non-measurable solutions of (1). So continuity is not a technicality glossing the statement; it is *precisely* the hypothesis that pins $\exp$ down as the unique additive-to-multiplicative map. Everything below is (2) read in a different category: the same equation, with a different target for multiplication.

## Four doors into the same room

**The dynamical law $f' = \lambda f$.** The most local thing one can say about a quantity is that its rate of change now is proportional to its value now — no memory, no preferred scale. Differentiating (1) at $b=0$ gives $f'=\lambda f$, so this law and the homomorphism property are the same statement. $\exp$ is the function compiled by the simplest feedback rule, which is why it is the leading-order model for decay, interest, cooling, chemical reaction rates, and unconstrained population.

$$
dx \approx \lambda x(t)\,dt,
\qquad
\frac{dx}{dt} = \lambda x,
\qquad
x(t) = x(0)e^{\lambda t}.
$$

So exponentials appear whenever the **rate of change is proportional to the current state**. The real invariant is the relative rate:

$$
\frac{x'(t)}{x(t)} = \lambda.
$$

Equivalently, $\log x(t)$ grows linearly:

$$
\frac{d}{dt}\log x(t)=\lambda
\quad\Longrightarrow\quad
\log x(t)-\log x(0)=\lambda t.
$$

This is the clean bridge. Additive increments in time add in log-space, and after exponentiating they become multiplicative factors:

$$
\frac{x(t+\Delta t)}{x(t)}=e^{\lambda\Delta t}.
$$

For two successive intervals, the correct semigroup statement is

$$
x(t_1+t_2)=x(0)e^{\lambda(t_1+t_2)}
          =\frac{x(t_1)x(t_2)}{x(0)}.
$$

After normalising by the initial value, $r(t)=x(t)/x(0)$, this becomes exactly

$$
r(t_1+t_2)=r(t_1)r(t_2).
$$

<figure>
  <img src="{{ '/assets/images/notes/stuff_i_found_interesting/exponential/exponential_growth_decay.svg' | relative_url }}" alt="Plot of exponential growth and decay with equal time steps marked as equal multiplicative factors" loading="lazy">
  <figcaption>Growth and decay are the same mechanism with different signs of $\lambda$. Equal time increments multiply the current value by the same factor; that is what a constant relative rate means.</figcaption>
</figure>

The exponential is not a technical accident. It is the canonical function that converts **additive infinitesimal change** into **multiplicative global change**.

**One-parameter groups, and Euler's identity as a corollary.** Whenever operations compose by *adding* a parameter, $T(s)\circ T(t)=T(s+t)$ — a flow, a semigroup — we again have a homomorphism out of $(\mathbb{R},+)$, and it is forced to be

$$
T(t)=\exp(tA), \qquad A=T'(0). \tag{3}
$$

Linear systems $\dot x=Ax$, the matrix exponential, Markov and heat semigroups are all (3) with different generators $A$. The derivative at the identity is the local rule; exponentiation is the act of integrating that local rule into a global flow. Restricting (3) to the imaginary axis sends $(\mathbb{R},+)$ onto the circle group: $t\mapsto e^{it}$ is the unique one-parameter family of rotations, *adding angles = multiplying unit complex numbers*. Thus $e^{i\pi}=-1$ records nothing more mystical than a half-turn; the generator $i$ is the infinitesimal $90^\circ$ rotation.

Euler’s formula $e^{it}=\cos t+i\sin t$ is not a coincidence. It says that exponential also converts additive time/angle into multiplicative rotations. Angles add: $\theta+\phi$. Rotations compose multiplicatively: $e^{i\theta}e^{i\phi}=e^{i(\theta+\phi)}$. So $e^{it}$ is the same additive-to-multiplicative bridge, but now the target is the unit circle: $(\mathbb R,+)\longrightarrow S^1$.

The differential equation viewpoint also says the same thing: $z'(t)=iz(t), \qquad z(0)=1$. The solution is $z(t)=e^{it}$. Multiplication by $i$ means “rotate by $90^\circ$.” Therefore the flow of the equation $z'=iz$ is uniform circular motion. Euler’s identity is the exponential function describing rotation.

<figure>
  <img src="{{ '/assets/images/notes/stuff_i_found_interesting/exponential/rotation_exponential.svg' | relative_url }}" alt="Unit circle diagram showing addition of angles as multiplication of complex exponentials" loading="lazy">
  <figcaption>The complex exponential is the same homomorphism, but now the multiplicative target is the unit circle. Euler's identity is the value at a half-turn.</figcaption>
</figure>

**Eigenfunctions and symmetry.** Exponentials are the eigenfunctions of $d/dx$, hence they simultaneously diagonalise every translation-invariant linear operator (every convolution). This is the real content of Fourier analysis: $e^{i\omega t}$ are the characters of the translation group, the coordinates in which "shift" becomes "multiply by a phase". Indeed, if $\tau_a$ shifts a signal by $a$, then

$$
(\tau_a e^{i\omega x})(x)=e^{i\omega(x+a)}
=e^{i\omega a}e^{i\omega x}.
$$

A translation, which is additive in $a$, acts on each exponential mode by multiplication by the scalar phase $e^{i\omega a}$. A system with a translation symmetry — in time or space — is therefore one whose dynamics go diagonal in the exponential basis. Same bridge: the symmetry is additive, its action is multiplicative.

**Entropy and information.** Probabilities of independent events multiply; we prefer additive bookkeeping, so we pass to logs, and information is $-\log p$. The function inverting $\log$ is $\exp$, and it reappears the moment we optimise. Maximising entropy $-\int p\log p$ under *linear* constraints (fixed mean, variance, expected features) is a short Lagrange-multiplier calculation. The Euler-Lagrange stationarity condition has the form

$$
-(1+\log p(x))-\alpha-\langle\theta,T(x)\rangle=0,
$$

so $\log p$ is affine in the constrained features. Exponentiating gives the exponential family:

$$
p(x)\;\propto\;\exp\!\big(\langle\theta,\,T(x)\rangle\big). \tag{4}
$$

From this single source: the **Gaussian** is the maximum-entropy law at fixed mean and variance (and, not coincidentally, the heat kernel and a Fourier fixed point — three doors, one room); **softmax** $p_i\propto e^{s_i}$ is the Gibbs distribution at fixed expected score; and the **sigmoid** $1/(1+e^{-x})$ is two-class softmax, the posterior of a model with linear log-odds — not an *ad hoc* squashing nonlinearity.

<figure>
  <img src="{{ '/assets/images/notes/stuff_i_found_interesting/exponential/logits_probabilities.svg' | relative_url }}" alt="Plot showing linear log odds becoming sigmoid probabilities through exponentiation and normalization" loading="lazy">
  <figcaption>In probabilistic models the additive object is usually a score, energy, or log-odds. Exponentiation makes it positive; normalization makes it a probability distribution.</figcaption>
</figure>

## The unification

These are not four reuses of one symbol but four instances of one phenomenon. In each case one identifies *which* addition and *which* multiplication:

| setting | additive side | multiplicative side |
|---|---|---|
| scalar exponential | input $a+b$ | factors $e^a e^b$ |
| dynamics / Lie theory | time, or a generator $A$ | states, group elements $\exp(tA)$ |
| symmetry / Fourier | translations, frequencies | phases $e^{i\omega t}$ |
| information | log-likelihood, energy, information | probabilities |

The fundamentalness one senses is just this: a large fraction of mathematics is the passage from a *local, linear, additive* description (a generator, a tangent space, a constraint, a log) to the *global, multiplicative* object it generates (a flow, a group, a distribution). The exponential is the universal map effecting that passage, universal because the additive-to-multiplicative homomorphism is unique under the mildest continuity.

The cleanest way to say it: $e$ is not special — *multiplicativity* is special, and $\exp$ is multiplicativity written in additive coordinates. The constant $e$ merely calibrates the rate of the calculus-compatible homomorphism; changing base only rescales $\lambda$.

## The core intuition

The general slogan is:

$$\boxed{\text{Exponential turns infinitesimal/additive structure into global/multiplicative structure.}}$$
 
Exponential functions are fundamental because they are the natural language of processes where:

$$\text{local change accumulates continuously}$$

and

$$\text{independent or successive effects compose multiplicatively}.$$

That is why the same function appears in growth, decay, rotations, differential equations, Gaussian densities, softmax, sigmoid, entropy, statistical mechanics, Fourier analysis, and Lie theory.

The exponential is not merely a function. It is the canonical passage from **linear infinitesimal structure** to **nonlinear global structure**.

## Remarks

1. The "why $\exp$ and nothing else" content lives entirely in (2): Cauchy's equation plus a whisper of regularity. Every other characterisation is (1) read in a suitable category.

2. The deepest generalisation is Lie-theoretic: $\exp:\mathfrak g\to G$ from a Lie algebra (additive tangent space) to its group (multiplicative). Growth, rotation, heat flow, and Markov evolution are all one-parameter subgroups $\exp(tA)$; the scalar exponential is this map for the $1$-dimensional algebra. *Continuous symmetry $\Leftrightarrow$ exponentiated generator* is the move that keeps returning.

3. The information thread rejoins the algebraic one through statistical mechanics. The partition function $Z(\theta)=\int\exp\langle\theta,T\rangle$ is a moment-generating function, $\log Z$ generates cumulants, and the Legendre duality between natural parameters $\theta$ and mean parameters is once more the additive$\leftrightarrow$multiplicative duality — now between log-probabilities and probabilities.

## But exponentials are not literally everywhere

They dominate when the underlying structure is linear, memoryless, multiplicative, or governed by constant relative rates. Other patterns produce other functions. A good diagnostic is the logarithmic derivative:

$$
\frac{d}{dt}\log x(t)=\frac{x'(t)}{x(t)}.
$$

If this is constant, $x$ is exponential. If it changes with $t$ or with $x$, the structural reason for $\exp$ has disappeared.

For example, $x'(t)=cx(t)$ gives exponentials, but

$$x'(t)=cx(t)^2$$

does not. It gives finite-time blow-up of rational type:

$$
x(t)=\frac{x(0)}{1-cx(0)t}.
$$

Similarly, scale invariance often gives power laws, not exponentials:

$$f(\lambda x)=\lambda^\alpha f(x) \quad\Rightarrow\quad f(x)=Cx^\alpha.$$

<figure>
  <img src="{{ '/assets/images/notes/stuff_i_found_interesting/exponential/not_everywhere_counterexamples.svg' | relative_url }}" alt="Plots of finite-time blow-up and a power law as non-exponential counterexamples" loading="lazy">
  <figcaption>Change the structural law and the canonical function changes. Nonlinear feedback gives rational blow-up; scale invariance gives power laws.</figcaption>
</figure>

So exponentials are not universal. They are universal for a very specific and very common structural reason: **additive parameters acting multiplicatively**.


## Addendum: does "homomorphic" already mean "exponential"?

*A natural worry about the thesis above. If all the structure lived in the homomorphism (1), then surely knowing two groups are homomorphic — one additive, one multiplicative — should already deliver the exponential. Does it? The honest answer sharpens the whole picture: the question is not well-posed until one names the category, and the answer flips between bare groups and topological groups. The labels "additive" and "multiplicative" are pure notation; a group is a group, and the word $\exp$ acquires meaning only once analytic structure is present.*

**As abstract groups: no, and not even close.** Take the very pair in question, $(\mathbb{R},+)$ and $(\mathbb{R}\_{>0},\times)$. Both are divisible torsion-free abelian groups, hence $\mathbb{Q}$-vector spaces of dimension $2^{\aleph_0}$, hence abstractly isomorphic. But the isomorphisms form an enormous set: choose a Hamel basis of each and match the bases however one pleases. These are exactly the non-measurable monsters that Cauchy's theorem (2) excludes. At the level of bare group structure the exponential is one drop in a $\mathbb{Q}$-linear sea of isomorphisms, distinguished by nothing.

**Worse — the isomorphism may not exist at all.** Restrict to the rationals. By unique factorisation, $(\mathbb{Q}\_{>0},\times)$ is the *free* abelian group on the primes,

$$
(\mathbb{Q}_{>0},\times)\;\cong\;\bigoplus_{p\ \text{prime}}\mathbb{Z}, \tag{5}
$$

of countably infinite rank and not divisible, whereas $(\mathbb{Q},+)$ *is* divisible (rank $1$ over $\mathbb{Q}$). The two are therefore **not isomorphic**, and indeed $\exp$ does not even map $\mathbb{Q}$ into $\mathbb{Q}\_{>0}$. So "additive group $\cong$ multiplicative group" is a genuine constraint that can fail outright; nothing forces the picture from group theory alone.

**As topological / Lie groups: now yes, and it is a theorem.** Impose continuity and rigidity returns. The clean statement is about one-parameter subgroups:

$$
\text{every continuous } \phi:(\mathbb{R},+)\to G \text{ has the form } \phi(t)=\exp_G(tX), \quad X\in\mathfrak g \text{ unique.} \tag{6}
$$

A continuous homomorphism out of $\mathbb{R}$ is *forced* to be an exponential — but it is the target group's own exponential map $\exp_G:\mathfrak g\to G$, and the sole remaining freedom is the generator $X$. For $G=(\mathbb{R}\_{>0},\times)$ this collapses to $\phi(t)=e^{\lambda t}$ with $\lambda=\phi'(0)$, i.e. the choice of base.

**Where the leftover freedom lives.** Even when the answer is "exponential," an isomorphism is never *canonical*: the isomorphisms $G\to H$ form a torsor under $\operatorname{Aut}(H)$. The continuous automorphisms of $(\mathbb{R}\_{>0},\times)$ are exactly $x\mapsto x^{c}$ with $c\neq 0$, so the family $\exp(\lambda x)$ is precisely the orbit of $\exp$ under continuous automorphisms — *which exponential* is *which base*. Drop continuity and $\operatorname{Aut}(\mathbb{R},+)$ swells to $GL$ of an infinite-dimensional $\mathbb{Q}$-vector space, and the orbit explodes back into the monster sea.

**The moral, sharply.** What makes $\exp$ the definite article is never the additive-versus-multiplicative *labels* — those are interchangeable bookkeeping — but the demand that the map respect the *analytic structure* of the real line. The exponential is the unique homomorphism compatible with that structure; strip the structure and one strips the uniqueness with it. This is exactly why the thesis leaned on continuity: it is not decoration but the entire reason "the exponential" is a definite article rather than an indefinite one. The algebraic isomorphisms see only the $\mathbb{Q}$-vector-space skeleton — its dimension — while the continuous ones see $\mathbb{R}$ as a one-dimensional Lie group, where the generator is the lone surviving coordinate. The bridge of the title is real, but it is a bridge between *structured* worlds; between bare groups there is either a chaos of bridges or none.



