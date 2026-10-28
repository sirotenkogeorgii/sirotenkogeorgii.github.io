---
layout: default
title: "From Finite to Infinite: Machines for Passing to the Limit"
tags:
  - mathematics
  - analysis
  - functional-analysis
  - measure-theory
  - probability
  - topology
---

# From Finite to Infinite: Machines for Passing to the Limit

## Conventions and the key idea

A property that holds at every finite stage survives in the limit exactly when it is witnessed *uniformly* in the stage and is *closed* in a topology in which the stages are *precompact*. Almost every "finite $\Rightarrow$ infinite" theorem in analysis, probability, logic and numerics is a variation on this one sentence.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Finite approximation, infinite object)</span></p>

A *finite approximation* is a family $(X\_N)$ indexed by a directed set: the dimension $N$ of a Galerkin space, a finite subset $F$ of an index set or of a set of axioms, a mesh size $h \to 0$, a time step $\tau \to 0$. The *infinite object* $X$ is what we want. Two questions are always lurking, and they are different: **existence** (is there an $X$ compatible with all the $X\_N$?) and **identification** (does $X$ inherit the property $P$ enjoyed by every $X\_N$, and is it unique?).

</div>

<div class="math-callout math-callout--info" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(The skeleton: four moves)</span></p>

Nearly every argument below has four moves:

1. *Approximate*: solve the finite problem.
2. *Estimate*: prove a bound on $X\_N$ that does not depend on $N$. This is where all the analysis lives.
3. *Extract*: use compactness (or completeness) to get a limit $X$ along a subsequence (or along the whole sequence).
4. *Identify*: show the defining property is closed under that mode of convergence.

When step 3 uses completeness instead of compactness, the limit is unique and step 4 is automatic; when it uses compactness, you get existence without uniqueness, and step 4 is usually the hard part. That dichotomy organises the post.

</div>

<figure>
  <img src="{{ '/assets/images/notes/math_topics/from-finite-to-infinite/ffi_skeleton.png' | relative_url }}" alt="Flow diagram of the four moves approximate, estimate, extract, identify, with a fork after extract: by completeness, giving a unique limit, or by compactness, giving a limit point along a subsequence." loading="lazy">
  <figcaption>The skeleton. Every machine in this note is a way of doing step 3, and the fork there decides everything downstream. Completeness (continuity, consistency, stability) hands over a unique limit and makes identification automatic. Compactness (compactness proper, transfer) hands over a limit point along a subsequence and leaves identification as the real work.</figcaption>
</figure>

<div class="math-callout math-callout--info" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Map of the machines)</span></p>

The machines, by what they consume and what they produce:

| Machine | Finite input | Infinite output | Unique? | The real price |
| --- | --- | --- | --- | --- |
| Monotonicity / uniform tails | estimates on partial sums | interchange of limit and infinite sum | yes | a bound uniform in the truncation |
| Continuity (density) | object on a dense subspace | its extension to the closure | yes | a uniform modulus of continuity |
| Consistency (projective limits) | compatible finite marginals | an object on the inverse limit | yes, once it exists | inner regularity / compact fibres |
| Compactness | approximants with uniform a priori bounds | a limit point | no (subsequences) | identifying the limit, nonlinear terms |
| Transfer (logic) | a statement true in all large finite structures | the statement in a limit structure | — | first-order expressibility |
| Stability (numerics, Γ-limits) | a consistent discrete scheme | convergence to the true solution, with rates | yes | a stability bound uniform in $h$ |

</div>

## 0. Interchanging limits: monotonicity, uniform tails, domination

The most elementary machines answer a narrow question: for a double sequence $a\_{n,k}$, when is $\lim\_n \sum\_k a\_{n,k} = \sum\_k \lim\_n a\_{n,k}$? Each partial sum $S\_{n,K} = \sum\_{k \le K} a\_{n,k}$ is a finite object, and the issue is whether the limits in $n$ and $K$ commute.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Monotone convergence and Fatou)</span></p>

For nonnegative terms the infinite sum is a supremum of finite ones, $\sum\_k a\_k = \sup\_K \sum\_{k \le K} a\_k$, so any bound $\sum\_{k \le K} a\_k \le M$ uniform in $K$ passes to the infinite sum for free. Two suprema commute (monotone convergence: if $a\_{n,k} \uparrow a\_k$ in $n$, then $\sum\_k a\_{n,k} \uparrow \sum\_k a\_k$), while a supremum against a liminf commutes in one direction only. That is Fatou:

$$
\sum_k \liminf_{n} a_{n,k} \le \liminf_{n} \sum_k a_{n,k}, \qquad a_{n,k} \ge 0.
$$

For each $K$, $\sum\_{k \le K} \liminf\_n a\_{n,k} \le \liminf\_n S\_{n,K} \le \liminf\_n \sum\_k a\_{n,k}$; take the supremum over $K$. Strict inequality happens: $a\_{n,k} = \delta\_{nk}$ gives $0 < 1$, with the mass escaping to $k = \infty$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Moore–Osgood: uniform tails)</span></p>

Equality needs that escape blocked. If $S\_{n,K} \to S\_K$ as $n \to \infty$ for each $K$, and $S\_{n,K} \to S\_n$ as $K \to \infty$ uniformly in $n$, then both iterated limits exist and agree: the $\varepsilon/3$ argument chooses $K$ first, simultaneously for all $n$, then $n$. In $\ell^p$, $1 \le p < \infty$, this is sharp: if $x^{(n)} \to x$ coordinatewise, then

$$
x^{(n)} \to x \text{ in } \ell^p \iff \lim_{K \to \infty} \sup_n \sum_{k > K} |x^{(n)}_k|^p = 0.
$$

For $\Leftarrow$: $x \in \ell^p$ by Fatou, and $\lVert x^{(n)} - x\rVert\_p^p$ splits into the first $K$ coordinates, which vanish as $n \to \infty$, and tails bounded by $2^{p-1}\sum\_{k>K}(\lvert x^{(n)}\_k\rvert^p + \lvert x\_k\rvert^p)$. For $\Rightarrow$: the tail of $x^{(n)}$ is within $\lVert x^{(n)} - x\rVert\_p$ of the tail of $x$; treat finitely many $n$ separately. The basis vectors $e\_n$ are the archetypal failure: coordinatewise null, tails never small.

</div>

<figure>
  <img src="{{ '/assets/images/notes/math_topics/from-finite-to-infinite/ffi_escape_vs_tails.png' | relative_url }}" alt="Left: an 8 by 8 grid with ones on the diagonal and zeros elsewhere, row sums equal to one, column limits equal to zero. Right: a semilog plot of the tail supremum against K, flat at one for the basis vectors and decaying geometrically for a dominated family." loading="lazy">
  <figcaption>Why interchanging limits needs a uniform tail. Left: $a_{n,k} = \delta_{nk}$ has row sums $1$ but every column tends to $0$, so $\sum_k \liminf_n a_{n,k} = 0 < 1 = \liminf_n \sum_k a_{n,k}$. Fatou is strict because the unit of mass runs off to $k = \infty$. Right: the Moore–Osgood quantity $\sup_n \sum_{k>K} \lvert x^{(n)}_k\rvert^2$ for the basis vectors $e_n$ (never small, however far out the tail starts) and for a dominated family (geometrically small, uniformly in $n$). Coordinatewise convergence upgrades to $\ell^2$ convergence exactly when this quantity tends to $0$.</figcaption>
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Domination and Vitali)</span></p>

A summable majorant $\lvert a\_{n,k}\rvert \le g\_k$, $g \in \ell^1$, gives $\sup\_n \sum\_{k>K} \lvert a\_{n,k}\rvert \le \sum\_{k>K} g\_k \to 0$. So dominated convergence is a sufficient condition for uniform tails, not a necessary one. The sharp version is Vitali's theorem: on a probability space, $f\_n \to f$ in $L^1$ iff $f\_n \to f$ in measure and $(f\_n)$ is uniformly integrable, $\lim\_{M \to \infty} \sup\_n \int\_{\lbrace\lvert f\_n\rvert > M\rbrace} \lvert f\_n\rvert = 0$. Uniform integrability is a uniform tail condition in the *range* of $f\_n$; on infinite measure spaces one adds tightness, a uniform tail condition in the *domain*.

</div>

## 1. Extension by continuity: density does the work

If you can define an object on a dense class of "finite" or "nice" inputs with a bound that does not see how nice they are, the object on all inputs is forced.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(BLT: extension by continuity)</span></p>

Let $D$ be a dense subspace of a normed space $X$, $Y$ a Banach space, and $T : D \to Y$ linear with $\lVert Tx\rVert \le C\lVert x\rVert$ on $D$. Then $T$ extends uniquely to a bounded operator on $X$ with the same norm. The proof is one line: if $x\_n \to x$ then $(Tx\_n)$ is Cauchy, so set $Tx = \lim Tx\_n$. The nonlinear version is the same sentence: a uniformly continuous map from a dense subset of a metric space into a complete one extends uniquely to the closure. All the content is in the estimate; the extension is free.

</div>

<figure>
  <img src="{{ '/assets/images/notes/math_topics/from-finite-to-infinite/ffi_blt.png' | relative_url }}" alt="Left: a dense grid of dyadic points in the unit square, a target point x off the grid, and three grid points converging to it. Right: the rotated and scaled image grid, the images of the three approximants, and dashed circles of radius C times the approximation error shrinking onto the point defined as T x." loading="lazy">
  <figcaption>Extension by continuity. Left: $T$ is only defined on the dense set $D$ (dyadic points); a general $x$ is reached by approximants $x_1, x_2, x_3 \in D$. Right: $T$ is a rotation and scaling with $\lVert Td\rVert \le C\lVert d\rVert$, so the images $Tx_n$ are Cauchy and sit inside shrinking balls of radius $C\lVert x_n - x\rVert$ around a single point. That point is <em>defined</em> to be $Tx$. All the content is in the estimate; the extension itself costs nothing.</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Banach–Steinhaus: convergence from a dense set)</span></p>

The same argument transports *convergence*. If $T\_n : X \to Y$ are bounded with $\sup\_n \lVert T\_n\rVert \le M$ and $T\_n d$ converges for every $d$ in a dense $D$, then $T\_n x$ converges for every $x$, since

$$
\|T_n x - T_m x\| \le 2M\|x - d\| + \|T_n d - T_m d\|.
$$

Banach–Steinhaus makes the hypothesis automatic once each orbit $(T\_n x)$ is bounded, and shows it is necessary for convergence everywhere. This is how one gets a continuous function whose Fourier series diverges at a point: the Lebesgue constants $\lVert D\_N\rVert\_{L^1}$ grow like $\log N$.

</div>

<figure>
  <img src="{{ '/assets/images/notes/math_topics/from-finite-to-infinite/ffi_dirichlet.png' | relative_url }}" alt="Left: the Dirichlet kernels for N equal to 4 and 16 on minus pi to pi, with a tall central peak and side lobes of constant height. Right: the Lebesgue constants against N on a logarithmic axis, following the line four over pi squared times log N plus a constant." loading="lazy">
  <figcaption>When the uniform bound fails. The partial sum $S_N f(0) = \frac{1}{2\pi}\int f(t)\,D_N(t)\,dt$ has operator norm $L_N = \frac{1}{2\pi}\lVert D_N\rVert_{L^1}$ on $C(\mathbb T)$. Left: the Dirichlet kernels for $N = 4$ and $N = 16$, whose side lobes keep their height. Right: $L_N$ grows like $\frac{4}{\pi^2}\ln N$. Convergence holds on the dense set of trigonometric polynomials, but with $\sup_N \lVert S_N\rVert = \infty$ the transport argument is unavailable, and Banach–Steinhaus turns this into a continuous function whose Fourier series diverges at $0$.</figcaption>
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Testing against a dense set of functionals)</span></p>

Dually, if $(x\_n)$ is bounded and $f(x\_n) \to f(x)$ for all $f$ in a norm-dense $D \subset X^\ast$, then $x\_n \rightharpoonup x$, because $\lvert f(x\_n - x)\rvert \le \lvert g(x\_n - x)\rvert + \lVert f - g\rVert \sup\_n \lVert x\_n - x\rVert$. For $1 < p < \infty$ the finitely supported sequences are dense in $\ell^q$, so bounded plus coordinatewise convergent implies weakly convergent in $\ell^p$. At $p = 1$ this fails: $(e\_n)$ is bounded and coordinatewise null, but pairs with $(1,1,1,\dots) \in \ell^\infty$ to give $1$.

</div>

<div class="math-callout math-callout--example" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Plancherel)</span></p>

The Fourier transform is defined by an honest integral on $L^1 \cap L^2$, where $\lVert\hat f\rVert\_2 = \lVert f\rVert\_2$. BLT extends it to a unitary on $L^2$. Note what the extension does *not* give: for a general $f \in L^2$, $\hat f$ is the $L^2$-limit of the truncated integrals over $\lvert x\rvert < R$, not the integral itself. This is the template: the infinite object is *defined* as the limit of its truncations, and the formula survives only in that limiting sense.

</div>

<div class="math-callout math-callout--example" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Itô integral)</span></p>

On simple predictable integrands $H = \sum\_i h\_i \mathbf 1\_{(t\_i, t\_{i+1}]}$ the stochastic integral is a finite sum, and the Itô isometry

$$
\mathbb{E}\Big|\int_0^T H\,dW\Big|^2 = \mathbb{E}\int_0^T |H_t|^2\,dt
$$

is a direct computation (cross terms vanish by the martingale property). BLT extends the integral to all predictable $H \in L^2(\Omega \times [0,T])$. The same story builds the Lebesgue integral from step functions (Daniell) and Sobolev traces from smooth functions (via the trace inequality $\lVert u\vert\_{\partial\Omega}\rVert \le C\lVert u\rVert\_{H^1}$).

</div>

<div class="math-callout math-callout--example" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Carathéodory)</span></p>

A premeasure on an algebra extends to a measure on the generated σ-algebra, uniquely if σ-finite (Dynkin's π–λ theorem). Here the role of the uniform bound is played by countable additivity on the algebra, i.e. continuity at $\varnothing$. This is the measure-theoretic BLT: one checks continuity on the finitely generated "cylinder" sets and the rest is bookkeeping.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Stein's maximal principle)</span></p>

Norm convergence passes from a dense class for free; almost-everywhere convergence does not. The fix is again a uniform estimate, this time on the maximal operator: if $T\_n f \to Tf$ a.e. for $f$ in a dense class and $T^\ast f = \sup\_n \lvert T\_n f\rvert$ satisfies a weak-type bound, then $T\_n f \to Tf$ a.e. for all $f$. Lebesgue differentiation (via Hardy–Littlewood) is the easy case; Carleson's theorem on a.e. convergence of truncated Fourier series of $L^2$ functions is the hard one, and its entire difficulty is the maximal inequality.

</div>

## 2. Extension by consistency: projective limits

When the finite data are *marginals* of the object you want, compatibility gives finite additivity for free; the genuine content is countable additivity, and it is always paid for with compactness.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Kolmogorov extension)</span></p>

Let $I$ be any index set and $(\mu\_F)$, $F \subset I$ finite, Borel probability measures on $\mathbb R^F$ that are consistent under coordinate projections. Then there is a unique probability $\mu$ on $\mathbb R^I$ with the product σ-algebra having these marginals. Consistency makes $\mu$ a well-defined, finitely additive set function on the algebra of cylinder sets. For countable additivity, suppose cylinders $C\_n \downarrow \varnothing$ but $\mu(C\_n) \ge \varepsilon$. Inner regularity of each $\mu\_F$ lets you shrink $C\_n$ to compact cylinders $K\_n \subset C\_n$, still decreasing, with $\mu(K\_n) \ge \varepsilon/2$. Each $K\_n$ is nonempty, and a decreasing sequence of nonempty compact cylinders has nonempty intersection (diagonal argument, or Tychonoff). Contradiction. Without topology the theorem is false; Ionescu-Tulcea rescues the purely measurable case only when the finite data come as a *sequential* family of kernels.

</div>

<figure>
  <img src="{{ '/assets/images/notes/math_topics/from-finite-to-infinite/ffi_nested.png' | relative_url }}" alt="Left: seven nested intervals shrinking onto a common point, labelled K 1 to K 7. Right: seven nested half-lines starting at 1, 2, up to 7 and running off to the right, whose intersection is empty." loading="lazy">
  <figcaption>Why compactness pays for countable additivity. Left: nested nonempty compact sets $K_n$ always have a common point (Cantor's intersection theorem); this is what the inner-regularity step in Kolmogorov's proof buys. Right: the nested nonempty sets $X_n = \{n, n+1, \dots\}$ have empty intersection. Each finite stage is satisfiable, but the thread escapes to infinity. The same picture is the Mittag-Leffler failure mode.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What the finite marginals cannot see)</span></p>

The product σ-algebra on $\mathbb R^{[0,1]}$ contains only events depending on countably many coordinates, so "the path is continuous" is not even an event. Finite-dimensional data fix the law but not path regularity. The upgrade is a multiscale estimate, Kolmogorov–Chentsov:

$$
\mathbb{E}|X_t - X_s|^{\alpha} \le C|t-s|^{1+\beta} \quad\Longrightarrow\quad X \text{ has a } \gamma\text{-Hölder modification for every } \gamma < \beta/\alpha.
$$

The proof bounds increments on dyadic grids level by level and sums a geometric series: a chaining argument, i.e. a uniform-in-$N$ estimate over the finite grids $2^{-N}\mathbb Z$.

</div>

<div class="math-callout math-callout--lemma" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(König)</span></p>

An infinite, finitely branching tree has an infinite branch. Equivalently, an inverse limit of nonempty *finite* sets is nonempty, and more generally an inverse limit of nonempty compact Hausdorff spaces is nonempty. This is the combinatorial engine behind:

- *De Bruijn–Erdős*: a graph is $k$-colourable iff every finite subgraph is.
- *Infinite Hall*: a family of finite sets has a system of distinct representatives iff every finite subfamily does (finiteness of each set is essential).
- *Wang tiles*: a finite tile set tiles the plane iff it tiles every finite square.

In each case the finite solutions form an inverse system of nonempty finite sets under restriction, and König supplies a compatible thread.

</div>

<figure>
  <img src="{{ '/assets/images/notes/math_topics/from-finite-to-infinite/ffi_konig.png' | relative_url }}" alt="A binary tree drawn to depth six. Grey subtrees die out after a few levels, blue nodes have infinite subtrees and continue below the bottom level, and one orange path from the root always steps into a blue child." loading="lazy">
  <figcaption>König's lemma. Grey subtrees die out after finitely many levels; blue nodes have infinitely many descendants. Because the tree is infinite and finitely branching, at least one child of every blue node is blue, so stepping into a blue child at each level (orange) never gets stuck. The nodes at level $n$ are the compatible finite solutions at stage $n$; the branch is the thread through the inverse limit.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Mittag-Leffler)</span></p>

Drop finiteness and inverse limits can be empty: $X\_n = \lbrace n, n+1, \dots\rbrace$ with inclusions has empty limit. For an inverse *sequence* of nonempty sets, surjective bonding maps suffice (dependent choice). The Mittag-Leffler condition (images stabilise) reduces to that case. For abelian groups, its failure is measured by the derived functor $\varprojlim^1$: for $\mathbb Z \xleftarrow{p} \mathbb Z \xleftarrow{p} \cdots$ we have $\varprojlim = 0$ but $\varprojlim^1 = \mathbb Z\_p/\mathbb Z \neq 0$, the algebraic shadow of a limit that "should" exist and doesn't.

</div>

## 3. Extension by compactness: existence from uniform bounds

Compactness converts "for every $N$ there is an approximate solution with a bound independent of $N$" into "there is a solution". It never gives uniqueness, and it makes you pay at the identification step.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Compactness theorem)</span></p>

A set $\Sigma$ of first-order sentences has a model iff every finite subset does. In the propositional case this is Tychonoff on $\lbrace0,1\rbrace^{\mathrm{Atoms}}$: the valuations satisfying each finite $\Delta \subset \Sigma$ form a closed nonempty set, these sets have the finite intersection property, so they meet. De Bruijn–Erdős falls out again (one atom per vertex–colour pair), as does: every partial order extends to a linear order (true for finite orders by induction).

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The diagonal argument)</span></p>

The countable prototype is that a bounded sequence in $\ell^p$ has a coordinatewise convergent subsequence. Extract for coordinate 1, refine for coordinate 2, and so on, then take the diagonal. This is sequential compactness of a countable product of compact intervals. It is also the core of Arzelà–Ascoli (converge on a countable dense set, then spread by equicontinuity) and of Banach–Alaoglu in separable spaces. It yields coordinatewise convergence only; upgrading that to norm convergence is exactly the uniform tail condition of §0.

</div>

<figure>
  <img src="{{ '/assets/images/notes/math_topics/from-finite-to-infinite/ffi_diagonal.png' | relative_url }}" alt="Six rows of dots. Row zero contains all indices from 1 to 30; each later row keeps a subset of the previous row. The j-th kept index of row j is circled in orange and the circled indices are joined by a dotted line moving right and down." loading="lazy">
  <figcaption>The diagonal argument. Row $0$ is the full index set. Row $j$ is a subsequence of row $j-1$ along which coordinate $j$ converges. The diagonal $n_j$ (orange), the $j$-th term of row $j$, lies in every row $i \le j$ from the $i$-th position on, so every coordinate converges along it. This is sequential compactness of a countable product of compact intervals, and it is the engine inside Arzelà–Ascoli and separable Banach–Alaoglu.</figcaption>
</figure>

<div class="math-callout math-callout--example" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Arzelà–Ascoli and Peano)</span></p>

For $x^{\prime} = f(t,x)$ with $f$ continuous and bounded by $M$, the Euler polygons $x\_h$ are $M$-Lipschitz, hence equicontinuous and bounded; Arzelà–Ascoli extracts a uniformly convergent subsequence, and since $f$ is uniformly continuous on compacts one passes to the limit in $x\_h(t) = x\_0 + \int\_0^t f(s, x\_h(\lfloor s\rfloor\_h))\,ds$. That is Peano's theorem. Uniqueness genuinely fails ($x^{\prime} = \sqrt{\lvert x\rvert}$, $x(0) = 0$): compactness found *a* limit point, and different subsequences may find different ones. Uniqueness needs a Lipschitz condition, which is a stability estimate (§5), not a compactness one.

</div>

<figure>
  <img src="{{ '/assets/images/notes/math_topics/from-finite-to-infinite/ffi_peano.png' | relative_url }}" alt="A fan of grey parabolic solution curves leaving the time axis at different times; Euler polygons started at h squared, for three step sizes, converging to the parabola t squared over four; and the zero solution in blue." loading="lazy">
  <figcaption>Peano without uniqueness. For $x' = \sqrt{\lvert x\rvert}$, $x(0) = 0$ there is a whole fan of solutions $x_c(t) = (t-c)_+^2/4$ (grey). Euler polygons started exactly at $0$ stay at $0$ (blue). Euler polygons started at $x_h(0) = h^2$, a perturbation that vanishes with $h$, converge to $t^2/4$ (orange). Both are consistent schemes, and compactness only promises that every limit point solves the equation. Which solution you reach depends on the subsequence, that is, on the approximation.</figcaption>
</figure>

<div class="math-callout math-callout--example" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Brouwer to Schauder)</span></p>

A continuous self-map $f$ of a compact convex $K$ in a normed space has a fixed point. Project onto the convex hull of a finite $\varepsilon$-net (Schauder projection $P\_\varepsilon$, with $\lVert P\_\varepsilon y - y\rVert < \varepsilon$), apply Brouwer to $P\_\varepsilon \circ f$ on a finite-dimensional simplex to get $x\_\varepsilon$, and extract a convergent subsequence by compactness of $K$. The $\varepsilon$-net is the truncation; compactness of $K$ is the uniform bound.

</div>

<div class="math-callout math-callout--example" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Galerkin with weak compactness)</span></p>

For $-\Delta u = f$ the Galerkin solutions $u\_N \in V\_N$ satisfy $\lVert\nabla u\_N\rVert\_2 \le C\lVert f\rVert$ by testing with $u\_N$, uniformly in $N$; Banach–Alaoglu gives a weak limit, and linear equations pass to weak limits. Nonlinear equations need more. For 3D Navier–Stokes, the energy identity bounds $u\_N$ in $L^\infty L^2 \cap L^2 H^1$, the equation bounds $\partial\_t u\_N$ in $L^{4/3} H^{-1}$, and Aubin–Lions upgrades this to strong $L^2\_{\mathrm{loc}}$ convergence, enough to pass to the limit in $(u\cdot\nabla)u$. The result is a Leray–Hopf weak solution, and the compactness method is honest about what it doesn't know: uniqueness in this class is open without forcing, and with a suitable forcing it fails (Albritton–Brué–Colombo, 2022).

</div>

<div class="math-callout math-callout--example" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Compact operators)</span></p>

In a Hilbert space, compact operators are exactly the norm-limits of finite-rank ones, and the spectral theorem for compact self-adjoint operators is the finite-dimensional one transported by compactness: maximise $\langle Tx, x\rangle$ on the unit sphere, extract a weakly convergent maximising sequence, use compactness of $T$ to make the maximum attained, and iterate on the orthogonal complement.

</div>

<div class="math-callout math-callout--example" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Prokhorov, Donsker, and the probabilistic Galerkin method)</span></p>

Tight families of probability measures on a Polish space are weakly precompact. Hence the slogan: *convergence of finite-dimensional distributions + tightness = weak convergence*. For the rescaled random walk $S\_{\lfloor nt\rfloor}/\sqrt n$, the CLT gives the finite-dimensional limits (a finite-dimensional fact), and a uniform bound on the modulus of continuity gives tightness in $C[0,1]$. The limit is Wiener measure: Donsker's invariance principle. Mean-field limits work the same way: tightness of the empirical measures of $N$ interacting particles, then identification of every limit point as a solution of the McKean–Vlasov martingale problem (Sznitman's propagation of chaos), then uniqueness of that problem to upgrade subsequential to full convergence.

</div>

<figure>
  <img src="{{ '/assets/images/notes/math_topics/from-finite-to-infinite/ffi_donsker.png' | relative_url }}" alt="Three panels showing the same plus-minus-one random walk rescaled to the unit interval with 20, 200 and 2000 steps, looking progressively more like a Brownian path." loading="lazy">
  <figcaption>Donsker's invariance principle. One sequence of $\pm 1$ steps, shown as $S_{\lfloor nt\rfloor}/\sqrt n$ for $n = 20, 200, 2000$. The central limit theorem fixes the law of every finite collection of values $(X_{t_1}, \dots, X_{t_k})$, a uniform bound on the modulus of continuity gives tightness in $C[0,1]$, and Prokhorov turns the two into weak convergence to Wiener measure.</figcaption>
</figure>

<div class="math-callout math-callout--example" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Minimizing movements (JKO))</span></p>

For a functional $F$ on $\mathcal P\_2(\mathbb R^d)$ and step $\tau > 0$, set

$$
\rho^\tau_{k+1} \in \operatorname*{argmin}_{\rho} \Big\{ F(\rho) + \frac{W_2^2(\rho,\rho^\tau_k)}{2\tau} \Big\}.
$$

Comparing with $\rho = \rho^\tau\_k$ and summing gives the discrete energy estimate $\sum\_k W\_2^2(\rho^\tau\_{k+1}, \rho^\tau\_k)/\tau \le 2\,(F(\rho\_0) - \inf F)$, whence by Cauchy–Schwarz $W\_2(\rho^\tau(t), \rho^\tau(s)) \le C(\lvert t-s\rvert + \tau)^{1/2}$: a Hölder bound uniform in $\tau$. A refined Arzelà–Ascoli theorem in the narrow topology (Ambrosio–Gigli–Savaré) extracts a limit curve, and the Euler–Lagrange equation of each step identifies it as a gradient flow; for $F$ = entropy plus potential energy one recovers Fokker–Planck (Jordan–Kinderlehrer–Otto, 1998). Time discretisation is the truncation, the energy is the uniform bound, and the limit is the continuous-time object.

</div>

## 4. Transfer: let logic take the limit

If a statement is first-order and holds in "most" finite structures, it holds in their ultraproduct; and many infinite structures *are* such ultraproducts.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Łoś)</span></p>

For structures $(M\_n)$ and a nonprincipal ultrafilter $\mathcal U$ on $\mathbb N$, a first-order sentence $\varphi$ holds in $\prod M\_n/\mathcal U$ iff $\lbrace n : M\_n \models \varphi\rbrace \in \mathcal U$. The ultraproduct is a limit in which every first-order property that holds eventually is preserved exactly, with no estimates at all. The price is moved into the language: only first-order statements transfer, so "finite", "countable" or "for all subsets" do not.

</div>

<div class="math-callout math-callout--example" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Ax–Grothendieck)</span></p>

An injective polynomial map $\mathbb C^n \to \mathbb C^n$ is surjective. For a finite set, injective $\Rightarrow$ surjective is the pigeonhole principle. The algebraic closure $\overline{\mathbb F}\_p$ is a union of finite fields, each mapped into itself by a polynomial with coefficients in it, so the statement holds over $\overline{\mathbb F}\_p$. For fixed $n$ and degree bound $d$ it is a single first-order sentence in the language of fields. A nonprincipal ultraproduct $\prod\_p \overline{\mathbb F}\_p/\mathcal U$ is algebraically closed of characteristic $0$ and cardinality continuum, hence isomorphic to $\mathbb C$ (Steinitz), and Łoś transfers the sentence. Pigeonhole, extrapolated to $\mathbb C$.

</div>

<div class="math-callout math-callout--example" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Nonstandard analysis)</span></p>

The same mechanism with a single ultrapower: a hyperfinite random walk with infinitesimal steps, pushed through the Loeb measure construction and the standard-part map, *is* Brownian motion (Anderson, 1976). The truncation is never removed; it is made infinitely fine.

</div>

<div class="math-callout math-callout--example" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Correspondence principles)</span></p>

Furstenberg's correspondence turns density statements about subsets of $[N]$ into recurrence statements for measure-preserving systems: take weak-* limits of empirical measures along the shift (compactness), then argue in the infinite system. Szemerédi's theorem becomes multiple recurrence. The converse direction, from infinitary back to finitary, is again compactness, now in the form of a contradiction argument: if the finitary statement failed uniformly, a limit of counterexamples would contradict the infinitary one.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Hard versus soft: metastability)</span></p>

Tao's metastability makes the price of such arguments visible. The infinite convergence principle (bounded monotone sequences converge) is equivalent to the finite one: for every $\varepsilon > 0$ and every $F : \mathbb N \to \mathbb N$ there is $M$ such that every monotone sequence in $[0,1]$ is $\varepsilon$-stable on some window $[N, F(N)]$ with $N \le M$. The bound $M$ depends on $F$ but not on the sequence. Every soft compactness argument has such a finitary shadow, with bounds that are uniform but typically ineffective.

</div>

<figure>
  <img src="{{ '/assets/images/notes/math_topics/from-finite-to-infinite/ffi_metastability.png' | relative_url }}" alt="Two increasing sequences of dots approaching 1, one fast and one slow, each with a shaded window from N to 2N marking where the sequence first moves by less than epsilon." loading="lazy">
  <figcaption>Metastability. Two monotone sequences with $\varepsilon = 0.05$ and $F(N) = 2N$; the shaded windows $[N, 2N]$ are the first on which each sequence moves by less than $\varepsilon$. The finitary statement is that such a window exists with $N \le M(\varepsilon, F)$ for a bound $M$ depending only on $\varepsilon$ and $F$: a monotone sequence in $[0,1]$ can move by $\varepsilon$ at most $1/\varepsilon$ times, so it cannot dodge every window $[N, F(N)]$ forever. The bound is uniform but, for general $F$, astronomically large.</figcaption>
</figure>

## 5. Stability: quantitative passage to the limit

When you already know the infinite object exists, the question becomes whether the truncations converge *to it*, and at what rate. The answer is always: consistency plus a stability bound uniform in the discretisation.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Lax equivalence)</span></p>

For a well-posed linear initial value problem and a consistent linear scheme $u\_{n+1} = S\_h u\_n$, convergence is equivalent to stability, $\sup\_{nh \le T} \lVert S\_h^n\rVert < \infty$. One direction is uniform boundedness (Banach–Steinhaus); the other is a telescoping identity,

$$
S_h^n u_0 - u(nh) = \sum_{k=0}^{n-1} S_h^{\,n-1-k}\big(S_h u(kh) - u((k+1)h)\big),
$$

which bounds the global error by the stability constant times $n$ local truncation errors, each $o(h)$.

</div>

<figure>
  <img src="{{ '/assets/images/notes/math_topics/from-finite-to-infinite/ffi_lax.png' | relative_url }}" alt="Left: the exact decaying exponential with three explicit Euler approximations: a stable one hugging the exact curve, a stable but oscillating one, and an unstable one whose oscillations grow. Right: a log-log plot of the global error against the step size with a slope-one reference line." loading="lazy">
  <figcaption>Lax equivalence. Left: explicit Euler for $u' = -\lambda u$ is consistent for every step size, but the amplification factor $1 - \lambda h$ must satisfy $\lvert 1-\lambda h\rvert \le 1$ for $\sup_n \lVert S_h^n\rVert$ to stay bounded. At $\lambda h = 2.1$ the local errors are amplified geometrically and the scheme diverges. Right: in the stable regime the telescoping identity bounds the global error by the stability constant times the sum of $n = 1/h$ local errors of size $O(h^2)$, which is the observed first-order rate.</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Céa and inf-sup)</span></p>

For a coercive, bounded bilinear form and a conforming Galerkin space $V\_h$, $\lVert u - u\_h\rVert \le (M/\alpha) \inf\_{v \in V\_h} \lVert u - v\rVert$: the Galerkin solution is quasi-optimal, so convergence reduces to approximability of $u$ by $V\_h$. For saddle-point problems (Stokes, mixed methods) coercivity is replaced by a discrete inf-sup constant, and the theorem needs it *uniform in $h$* (Babuška–Brezzi). When it degenerates you see exactly the failures one expects from a non-uniform estimate: spurious pressure modes and locking.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Γ-convergence)</span></p>

Functionals $F\_n$ Γ-converge to $F$ if (i) $F(x) \le \liminf F\_n(x\_n)$ whenever $x\_n \to x$, and (ii) every $x$ has a recovery sequence with $F\_n(x\_n) \to F(x)$. With equicoercivity (a compactness hypothesis, again), minima converge and cluster points of minimisers minimise $F$. This is the variational form of "consistency + stability": (ii) is consistency, (i) is the stability of the lower bound. Examples: Modica–Mortola (diffuse interfaces Γ-converge to perimeter), discrete-to-continuum limits of lattice energies, and, with an additional lower bound on slopes, convergence of gradient flows (Sandier–Serfaty).

</div>

<figure>
  <img src="{{ '/assets/images/notes/math_topics/from-finite-to-infinite/ffi_modica_mortola.png' | relative_url }}" alt="Left: hyperbolic tangent profiles with widths 0.4, 0.15 and 0.05 sharpening towards a step function. Right: the energy against epsilon on logarithmic axes: flat at the perimeter constant for the optimal profiles, growing like one over epsilon for a profile of fixed width." loading="lazy">
  <figcaption>Γ-convergence in the Modica–Mortola functional $F_\varepsilon(u) = \int \frac{\varepsilon}{2}\lvert u'\rvert^2 + \frac{W(u)}{\varepsilon}$ with $W(u) = (1-u^2)^2/4$. Left: the optimal profiles $\tanh(x/\sqrt2\varepsilon)$ sharpen to a step. Right: along this recovery sequence the energy converges to the perimeter constant $c_W = \int_{-1}^1 \sqrt{2W(u)}\,du = 2\sqrt2/3$ per interface, which is the Γ-limit (consistency). A profile whose width does not shrink with $\varepsilon$ has energy of order $1/\varepsilon$ and is not a recovery sequence. The liminf inequality is the stability of the lower bound; equicoercivity is what makes minimisers converge at all.</figcaption>
</figure>

<div class="math-callout math-callout--example" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Finite sections)</span></p>

To solve $Ax = b$ on $\ell^2(\mathbb N)$, solve $P\_N A P\_N x\_N = P\_N b$. This converges for every $b$ iff $A$ is invertible and the sections are stable, $\sup\_{N \ge N\_0} \lVert(P\_N A P\_N)^{-1}\rVert < \infty$. Invertibility alone is not enough: the unitary $A$ swapping $e\_1 \leftrightarrow e\_2$, $e\_3 \leftrightarrow e\_4, \dots$ has singular sections for every odd $N$. For Toeplitz operators with continuous symbol $a$, stability holds iff $T(a)$ is invertible (Gohberg–Feldman), and the truncations carry spectral information in the limit: Szegő's theorem gives $\frac1N \log\det T\_N(a) \to \frac1{2\pi}\int \log a$. For general self-adjoint operators, truncation eigenvalues can accumulate in gaps of the essential spectrum (*spectral pollution*), a reminder that spectra are not continuous under strong convergence.

</div>

<div class="math-callout math-callout--example" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Richardson extrapolation)</span></p>

If $A(h) = A + c\,h^p + O(h^q)$ with $q > p$, then

$$
\frac{2^p A(h/2) - A(h)}{2^p - 1} = A + O(h^q),
$$

and iterating gives Romberg integration and Bulirsch–Stoer. Richardson extrapolation is only as good as the asymptotic expansion behind it (Euler–Maclaurin for the trapezoidal rule); Aitken's $\Delta^2$, Shanks' transformation and Padé approximants play the same game for sequences and power series, and finite-size scaling does it for $N$-particle systems in statistical physics.

</div>

## 6. When the passage fails, and how to repair it

Each failure is the absence of one ingredient of the skeleton, and each repair restores that ingredient in a weaker form.

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The property is not closed)</span></p>

Universal properties ("no finite configuration of type X occurs") are closed in the product topology and pass to limits; existential ones with unbounded witnesses do not. Every finite linear order has a maximum, $\mathbb N$ does not. Compactness itself proves that finiteness is not first-order expressible. Rule of thumb: a $\Pi\_1$ property passes to the limit, a $\Sigma$ property needs a bound on its witnesses.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Nonlinearity does not commute with weak limits)</span></p>

$\sin(nx) \rightharpoonup 0$ but $\sin^2(nx) \rightharpoonup 1/2$. Weak compactness finds a limit, but the limit of the nonlinear term is not the nonlinear term of the limit. Repairs: strong compactness (Rellich–Kondrachov, Aubin–Lions), monotonicity (Minty's trick), convexity (weak lower semicontinuity), compensated compactness (the div–curl lemma), or keeping the oscillations as data (Young measures).

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Compactness is lost to a symmetry)</span></p>

In $\mathbb R^d$ bounded sequences in $H^1$ can escape by translation or concentrate by dilation. Lions' concentration–compactness classifies minimising sequences into compactness up to symmetries, vanishing, or dichotomy, and profile decompositions (Gérard, Bahouri–Gérard) write any bounded sequence as a sum of rescaled, translated profiles plus a remainder that is small in the critical norm. The Yamabe problem is the classic instance: minimising sequences may concentrate into a standard bubble, and existence follows once one shows the Yamabe constant lies strictly below the bubble's energy (Aubin, Schoen).

</div>

<figure>
  <img src="{{ '/assets/images/notes/math_topics/from-finite-to-infinite/ffi_lost_compactness.png' | relative_url }}" alt="Three panels: a rapidly oscillating sine and its square with their weak limits 0 and one half drawn as dashed lines; three Gaussian bumps concentrating into a spike at the origin; three Gaussian bumps translating to the right with an arrow pointing further right." loading="lazy">
  <figcaption>Three ways to lose compactness. Oscillation: $\sin(nx) \rightharpoonup 0$ while $\sin^2(nx) \rightharpoonup 1/2$, so nonlinear functions do not commute with weak limits (the previous remark). Concentration: the dilates $\sqrt\lambda\,\varphi(\lambda x)$ keep their $L^2$ norm but converge weakly to $0$. Escape: the translates $\varphi(x - n)$ do the same. Each is a bounded sequence with no strongly convergent subsequence, and each is the failure of a different ingredient: closedness of the nonlinear property, or compactness lost to the dilation and translation symmetries of $\mathbb R^d$.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The inverse limit is empty)</span></p>

Without finiteness or compactness of the stages, the thread may not exist (Mittag-Leffler, §2), and for groups the obstruction is $\varprojlim^1$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The passage is not effective)</span></p>

Weak König's lemma (every infinite binary tree has an infinite path) is not computably true: there are computable infinite binary trees with no computable path, e.g. the tree of separators of two computably inseparable c.e. sets. The finite stages are computable; the limit need not be. What survives is a basis theorem: every nonempty $\Pi^0\_1$ class has a low member (Jockusch–Soare). Likewise a Specker sequence is a computable, bounded, monotone sequence of rationals with uncomputable limit, which is why metastability (§4), not convergence, is the effective content of monotone convergence. Over ZF, the compactness theorem is equivalent to the Boolean prime ideal theorem; full Tychonoff is equivalent to the axiom of choice.

</div>

## Remarks

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(One theorem in disguise)</span></p>

Put the product topology on the space of candidate infinite objects. "Determined by finite information" means continuous; "a property checkable on finite pieces" means closed; "every finite piece can be satisfied" means the finite intersection property. König, Kolmogorov, the compactness theorem, Tychonoff and the weak-* arguments in PDE are all the same statement: *a closed condition satisfiable at every finite stage is satisfiable in the limit, provided the ambient space is compact.* Analysis differs from combinatorics only in that compactness must be earned by an a priori estimate rather than given by finiteness of the stages.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Soft versus quantitative)</span></p>

Continuity and stability give uniqueness and rates; compactness and transfer give existence without either. Each soft argument has a finitary counterpart with uniform but often ineffective bounds, and the reverse-mathematics and computability results of §6 measure exactly how ineffective. Numerical analysis can be read as the part of mathematics that refuses to stop at compactness: it insists on the stability estimate that makes the limit computable.

</div>
