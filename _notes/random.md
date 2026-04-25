---
layout: default
title: random
date: 2024-10-20
---

# Random Stuff

## Edmonds-Karp algorithm analysis

<div class="math-callout math-callout--remark">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(naming)</span></p>
  <p>The name "Ford–Fulkerson" is often also used for the Edmonds–Karp algorithm, which is a fully defined implementation of the Ford–Fulkerson method.</p>
</div>

<div class="math-callout math-callout--info">
  <p class="math-callout__title"><span class="math-callout__label">Recall</span></p>
  <p>Edmonds–Karp is an efficient implementation of the Ford–Fulkerson method that selects shortest augmenting paths in the residual graph. It assigns a weight of $1$ to every edge and runs BFS to find a breadth-first shortest path from $s$ to $t$ in $G_f$.</p>
</div>

### Complexity

In Ford–Fulkerson, we repeatedly add augmenting paths to the current flow. If at some point no more augmenting paths exist, the current flow is maximal, so the best that can be guaranteed is that the answer will be correct if the algorithm terminates. For irrational capacities the algorithm may not terminate, but with integer capacities it does, and its running time is $O(Ef_{\text{max}})$: each augmenting path length is bounded from above by $O(E)$, the number of flow augmentations is bounded from above by $f_{\text{max}}$.

The Edmonds–Karp variant guarantees termination and runs in $O(VE^2)$ time, independent of the maximum flow value.

### Analysis

<!-- $\textbf{Lemma A (about paths length)}:$ The length of the shortest path from the source to the target (sink) increases monotonically.

*Proof*: 

* Let $R_i$ be a residual graph after the $i$-th flow augmentation and the shortest path from $s$ to $t$ be $l$ in $R_i$.
* How does $R_i$ differ from $R_{i+1}$? We removed all edges edges that were saturated and my removing edges we cannot create new paths. However, we can add new edges to $R_{i+1}$ by increasing the flow in the opposite direction. Let's show that the newly created paths shortest paths from $s$ to $t$ in $R_{i+1}$ are not shorter.
* Sort vertices in a graph by their distance from the source. We $R_i$ increased the flow in the edges going by one layer ahead, therefore the only edges that are added to $R_{i+1}$ are those going by one layer back in the sorted graph. So, the length of the shortest path from $s$ to $t$ in $R_{i+1}$ is at least the same. -->

<div class="math-callout math-callout--theorem">
  <p class="math-callout__title"><span class="math-callout__label">Lemma A</span><span class="math-callout__name">(monotonically increasing distances from source)</span></p>
  <p>Distances $d_i(s,u)$ increase monotonically, where $s$ is a source, $u$ is any other vertex, and $i$ is the order iteration of augmentation.</p>
</div>

*Proof*: 
* Let $R_i$ be a residual graph after the $i$-th flow augmentation and $u$ be some vertex in $R_i$.
* How does the shortest path in $R_i$ differ from shortest path in $R_{i+1}$ for vertices $s$ and $u$? Stepping from $R_{i}$ to $R_{i+1}$ we removed all edges that were saturated and by removing edges we cannot create new shortest path from $s$ to $u$. However, we can add new edges to $R_{i+1}$ by increasing the flow in the opposite direction. Let's show that the newly created shortest paths from $s$ to $t$ in $R_{i+1}$ are not shorter than in $R_{i}$.
* Sort vertices in a graph by their distance from the source. We $R_i$ increased the flow in the edges going by one layer ahead, therefore the only edges that are added to $R_{i+1}$ are those going by one layer back in the sorted graph. So, the length of the shortest path from $s$ to $u$ in $R_{i+1}$ is at least the same.

$\textbf{Corollary from Lemma A}:$ The length of the shortest path from the source to the target (sink) increases monotonically.

<div class="math-callout math-callout--remark">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Dinic vs Edmonds–Karp)</span></p>
  <p>In the Dinic's algorithm we have a lemma similar to the corollary above, but there we look for a blocking flow, increasing flow in all shortest paths (not in only the shortest path like in Edmonds–Karp), making the shortest path from the source to the target increase strictly (by at least $1$).</p>
</div>

$\textbf{Lemma B (number of flow augmentations)}:$ The number of flow augmentations in the Edmonds-Karp algorithm is $O(VE)$.

*Proof*: The residual graph contains $O(E)$ edges. Consider the edge $(u,v)$. Simple observation is that in general case for capacities having any values, the number of flow augmentations is bounded from above by the number of saturations. We can estimate the number of saturations taking into account that at least one edge is saturated during the flow augmentation:
* Consider the case where the edge $(v,u)$ is used to push flow back (which is necessary to make $(u,v)$ available for saturation again). Let this occur at some intermediate iteration $k > i$. Since $(v,u)$ is on the shortest path at iteration $k$, we must have: $d_k(s, u) = d_k(s, v) + 1$
* From Lemma A, we know that distances are monotonically increasing, so $d_k(s, v) \ge d_i(s, v)$. We can substitute the distance relationship from the first iteration $i$ (where $d_i(s, v) = d_i(s, u) + 1$) into our equation for iteration $k$:
  
  $$d_k(s, u) = d_k(s, v) + 1 \ge d_i(s, v) + 1 = (d_i(s, u) + 1) + 1 = d_i(s, u) + 2$$

* This shows that each time the edge $(u,v)$ becomes saturated (after the first time), the shortest-path distance from the source to $u$ must have increased by at least $2$. The shortest path distance from $s$ to any node $u$ cannot exceed $\lvert V\rvert - 1$ (otherwise $u$ is unreachable). Therefore, a specific edge $(u,v)$ can become saturated at most $\frac{\lvert V\rvert}{2}$ times.

Since there are $O(E)$ edges in the residual graph, and each can be saturated $O(V)$ times, the total number of saturations (and thus the total number of flow augmentations) is bounded by $O(VE)$.

<figure>
  <img src="{{ 'assets/images/notes/random/Edmonds_Karp_proof1.jpeg' | relative_url }}" alt="a" loading="lazy">
</figure>

<div class="math-callout math-callout--theorem">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(running time)</span></p>
  <p>The Edmonds–Karp maximum-flow algorithm runs in $O(VE^2)$ time.</p>
</div>

*Proof.* BFS runs in $O(E)$ time, and there are $O(VE)$ flow augmentations. All other bookkeeping is $O(V)$ per flow augmentation.

## Value of Information

**Value of Information (VOI, or VoI)** is the *expected benefit* you get from acquiring additional information **before** making a decision.

If you must choose an action (e.g., treat vs. wait, ship vs. hold, invest vs. don’t), and outcomes are uncertain, then information (a test, a study, a sensor reading, a survey, running an experiment) can reduce uncertainty, which can lead you to pick a better action on average. So VOI answers: **“How much is it worth to learn X before deciding?”**

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Value of Information, VOI or VoI)</span></p>

  Let:

  * $a \in \mathcal{A}$ be an action
  * $\theta$ be the unknown “state of the world”
  * $U(a,\theta)$ be the utility (profit, health benefit, etc.)
  * current beliefs about $\theta$ are $p(\theta)$

  **Without extra information**, the best expected utility is

  $$
  \max_{a} \; \mathbb{E}_{\theta}[U(a,\theta)] .
  $$

  If you can observe some information $I$ (e.g., a test result) with distribution $p(I \mid \theta)$, then **with information** you choose an action *after seeing $I$*:
  
  $$
  \mathbb{E}_{I}\!\left[\max_{a} \; \mathbb{E}_{\theta \mid I}[U(a,\theta)]\right] .
  $$

  So the **Value of Information** is

  $$
  \text{VOI} = \mathbb{E}_{I}\!\left[\max_{a} \; \mathbb{E}_{\theta \mid I}[U(a,\theta)]\right] - \max_{a} \; \mathbb{E}_{\theta}[U(a,\theta)].
  $$

</div>

In words: **(best expected outcome with the info) − (best expected outcome without it)**.

#### Important notes

* $\text{VOI}$ is an *expected* value: information might not help in every case, but on average it can.
* You typically compare $\text{VOI}$ to the **cost of information** (money, time, risk).
  * If $\text{VOI} > \text{Cost}$, it’s worth getting the info.

#### Common $\text{VOI}$ variants

* **EVPI** (Expected Value of Perfect Information): value if you could learn the true state $\theta$ exactly. This is an *upper bound* on any realistic info.
* **EVSI** (Expected Value of Sample Information): value of a particular imperfect test/study/measurement.

#### Example

Suppose you must decide today whether to **launch** a feature.

* If demand is high: launch gives $+100$, don’t launch gives $0$
* If demand is low: launch gives $−50$, don’t launch gives $0$
* You believe high demand is $40%$ likely.

Without extra info:

* Expected value(launch) = $0.4 \cdot 100 + 0.6 \cdot (-50)=40-30=10$
* Expected value(don’t) = $0$. So you launch (value = $10$).

Now imagine you can run a market test that sometimes changes your belief, leading you to launch only when it looks promising. If that improves your expected value from $10$ to, say, $18$, then value of that information that this test gives is $8$: **VOI = 18 − 10 = 8**. If the test costs $5$, it’s worth it; if it costs $20$, it isn’t.

## The first fundamental theorem of calculus

## The second fundamental theorem of calculus

## Inverse function theorem

## Involution

$\textbf{Definition (Involution):}$ A function $f$ is an **involution** if

$$f(f(x)) = x \quad \text{for all } x \text{ in the domain.}$$

So $f$ is its own inverse: $f^{-1} = f$.

$\textbf{Properties:}$
1. Any involution is a **bijection**.
2. The composition $g \circ f$ of two involutions f and g is an involution $\iff$ they commpute ($g \circ f = f \circ g$)

<figure>
  <img src="{{ 'assets/images/notes/random/involution.png' | relative_url }}" alt="a" loading="lazy">
</figure>

**Examples**

*Real-valued functions*

* $f(x)= -x$ on $\mathbb{R}: (f(f(x)) = -(-x)=x)$.
* $f(x)= \frac{1}{x}$ on $\mathbb{R}\setminus \lbrace 0\rbrace: f(f(x))=\frac{1}{1/x}=x$.
* A swap (permutation) like $(1\ 3)(2\ 5)$: applying it twice restores every element.

*In group theory*

An element $g$ of a group is an **involution** if it has order $2$:

$$g^2 = e,$$

where $e$ is the identity.
Example: in the multiplicative group $\lbrace \pm 1\rbrace$, the element $-1$ is an involution because $(-1)^2=1$.

*In geometry / linear algebra*

A transformation is an involution if applying it twice is the identity transformation.

* Reflection across a line in the plane (or a plane in 3D).
* A matrix $A$ such that $A^2 = I$ (the identity matrix) is sometimes called an **involutory matrix**.

*In differential geometry (a bit different usage)*

An **involutive distribution** (or system) is one closed under the Lie bracket; this is the condition in the Frobenius theorem that ensures it comes from a foliation. Different technical meaning, but still “closed under the operation.”

## Orthogonal polynomials

## Legendre polynomials

## Integration by substitution, U-substitution, Reverse chain rule, Change of ariables

### Change of variables in the probability density function

## Pushforward measure

## Riemann integral

## Darboux integral

## Riemann–Stieltjes integration

The Riemann–Stieltjes integral generalises the Riemann integral by integrating $f$ against another **function** $g$ — the *integrator* — instead of against $\mathrm{d}x$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Riemann–Stieltjes integral)</span></p>

Let $f, g : [a, b] \to \mathbb{R}$. For a partition $P = (a = t_0 < t_1 < \cdots < t_n = b)$ with tags $\xi_i \in [t_{i-1}, t_i]$, form the **Riemann–Stieltjes sum**

$$
S(P, \xi, f, g) \;:=\; \sum_{i=1}^{n} f(\xi_i) \, \bigl[g(t_i) - g(t_{i-1})\bigr].
$$

If there exists $L \in \mathbb{R}$ such that for every $\varepsilon > 0$ there is $\delta > 0$ with

$$
\bigl|S(P, \xi, f, g) - L\bigr| < \varepsilon \qquad \text{whenever the mesh } \Delta(P) < \delta, \text{ for } \textbf{every} \text{ tag choice } \xi,
$$

then $f$ is **Riemann–Stieltjes integrable with respect to $g$** and we write $\int_a^b f \, \mathrm{d}g := L$. Setting $g(x) = x$ recovers the ordinary Riemann integral.

</div>

### Classical sufficient conditions for existence

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Propositions</span><span class="math-callout__name">(The cleanest classical statements on Riemann–Stieltjes integral)</span></p>

* If $f$ is **continuous** on $[a, b]$ and $g$ is of **bounded variation** on $[a, b]$, then $\int_a^b f \, \mathrm{d}g$ exists.
* More generally (Helly–Stieltjes): $\int_a^b f \, \mathrm{d}g$ exists whenever $f$ and $g$ have no common discontinuities and one of them is BV.

A function $g$ is of **bounded variation (BV)** on $[a, b]$ iff its total variation

$$
V_a^b(g) \;:=\; \sup_{P} \, \sum_{i=1}^{n} \bigl|g(t_i) - g(t_{i-1})\bigr|
$$

is finite (sup over all partitions $P$). Equivalently (Jordan decomposition), $g$ is a difference of two non-decreasing functions, which lets one identify $\mathrm{d}g$ with a signed Borel measure on $[a, b]$.

</div>

### Bounded variation is the load-bearing hypothesis

The whole machinery of Riemann–Stieltjes integration rests on the BV assumption. **Why?** Two consequences of BV are essential:

1. **Tag-independence of the limit.** If $g$ is BV and $f$ is continuous, then *every* tag choice $\xi$ produces sums that converge to the **same** limit. Were the limit to depend on $\xi$, the integral would not be well-defined. The cancellation that makes this work is exactly what fails when $g$ is rough.

2. **Existence of the signed measure $\mathrm{d}g$.** BV $\Leftrightarrow$ $g$ defines a signed Borel measure (via Jordan + Hahn decomposition). Without this, "$\mathrm{d}g$" is just notation with no underlying measure-theoretic object.

The figure below illustrates point (1) for the BV integrator $g(x) = x^2$ on $[0,1]$ with $f(x) = x$. The true value is $\int_0^1 x \, \mathrm{d}(x^2) = \int_0^1 x \cdot 2x \, \mathrm{d}x = 2/3$. All three tag rules — left, midpoint, right — converge to this single value as the mesh refines, regardless of the (sometimes large) initial bias for coarse $n$.

<figure>
  <img src="{{ '/assets/images/notes/random/rs_bv_tag_independence.png' | relative_url }}" alt="For BV integrator g(x)=x^2, left/mid/right Riemann-Stieltjes sums all converge to 2/3" loading="lazy">
</figure>

### Worked manifestations

* **Discrete sums.** If $g$ jumps by $\Delta g(c_k) := g(c_k^+) - g(c_k^-)$ at finitely many points $c_k$ and is constant elsewhere, then $\int f \, \mathrm{d}g = \sum_k f(c_k) \, \Delta g(c_k)$ — the Riemann–Stieltjes integral collapses to a finite weighted sum.
* **Lebesgue–Stieltjes.** When $g$ is BV (and right-continuous), $\mathrm{d}g$ extends to a signed Borel measure $\mu_g$, and $\int f \, \mathrm{d}g = \int f \, \mathrm{d}\mu_g$ — Riemann–Stieltjes *is* Lebesgue–Stieltjes integration on continuous integrands.
* **Integration by parts.** If $f, g$ are BV with no common discontinuities,
  
  $$
  \int_a^b f \, \mathrm{d}g + \int_a^b g \, \mathrm{d}f = f(b) g(b) - f(a) g(a) ,
  $$
  
  a clean generalisation of integration by parts that does *not* require differentiability.

## Why Riemann–Stieltjes integration fails for Brownian motion (and SDEs)

The Riemann–Stieltjes machinery presumes the integrator has bounded variation. Brownian motion violates this assumption catastrophically — and this failure is the *reason* stochastic calculus (Itô, Stratonovich) had to be invented.

### Two pathological properties of Brownian paths

For standard Brownian motion $W$ on $[0, T]$, the following hold $\mathbb{P}$-a.s.:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Fact</span><span class="math-callout__name">(pathwise variations of Brownian motion)</span></p>

* **Total variation is infinite** on every subinterval:
  
  $$
  V_a^b\bigl(W(\omega)\bigr) = +\infty \qquad \text{for every } 0 \leq a < b \leq T.
  $$

* **Quadratic variation equals time elapsed**: along any sequence of partitions whose mesh tends to zero,
  
  $$
  [W]_t := \lim_{|P| \to 0} \sum_{i} \bigl(W_{t_i} - W_{t_{i-1}}\bigr)^2 = t \qquad (\text{in } L^2 \text{ and } \mathbb{P}\text{-a.s. along refining partitions}).
  $$

</div>

These two facts are *connected*: any function $g$ of bounded variation has quadratic variation **zero**, since

$$
\sum_i \bigl(g(t_i) - g(t_{i-1})\bigr)^2 \;\leq\; \max_i \bigl|g(t_i) - g(t_{i-1})\bigr| \cdot V_a^b(g) \;\to\; 0
$$

as the mesh shrinks (the max factor vanishes by continuity, the sum is bounded by $V_a^b(g) < \infty$). Contrapositive: positive quadratic variation **forces** unbounded total variation. Brownian motion is the canonical example of this "rough" scaling — increments behave like $\sqrt{\mathrm{d}t}$ rather than $\mathrm{d}t$.

The two divergent behaviours are visible numerically. Below, $V_n = \sum_i \|\Delta_i W\|$ grows like $\sqrt{n}$ (left, log-log), while $Q_n = \sum_i (\Delta_i W)^2$ converges to $T = 1$ (right):

<figure>
  <img src="{{ '/assets/images/notes/random/rs_bm_variations.png' | relative_url }}" alt="Total variation of Brownian motion grows like sqrt(n); quadratic variation converges to T=1" loading="lazy">
</figure>

### Why the Riemann–Stieltjes limit ceases to exist

Riemann–Stieltjes integrability requires the sums

$$
S_n \;=\; \sum_{i=1}^{n} f\bigl(\xi_i^{(n)}\bigr) \, \bigl[W_{t_i^{(n)}}(\omega) - W_{t_{i-1}^{(n)}}(\omega)\bigr]
$$

to converge to **the same** limit no matter how the tags $\xi_i^{(n)} \in [t_{i-1}^{(n)}, t_i^{(n)}]$ are chosen. With $W$ as integrator this **flatly fails**: the limit *depends on the tag rule*. The reason is structural — $f(\xi_i) - f(t_{i-1})$ is correlated with $\Delta_i W$ in a way that does **not** vanish in the limit when $g$ has positive quadratic variation, while it *does* vanish when $g$ is BV.

### The canonical example: $\int_0^T W_t \, \mathrm{d}W_t$

Take the simplest non-trivial integrand, $f = W$. Different tag choices give *genuinely different* limits in $L^2$:

| Tag choice $\xi_i$ | $L^2$-limit of $\sum f(\xi_i) \, \Delta_i W$ | Calculus name |
|---|---|---|
| Left endpoint $\xi_i = t_{i-1}$ | $\frac{1}{2}(W_T^2 - T)$ | **Itô** |
| Midpoint $\xi_i = (t_{i-1} + t_i)/2$ | $\frac{1}{2} W_T^2$ | **Stratonovich** |
| Right endpoint $\xi_i = t_i$ | $\frac{1}{2}(W_T^2 + T)$ | "anti-Itô" |

The discrepancy between any two consecutive choices is exactly the **quadratic-variation term**:

$$
\sum_i \bigl(W_{t_i} - W_{t_{i-1}}\bigr)^2 \;\xrightarrow[n \to \infty]{L^2}\; T \;\neq\; 0 .
$$

For a BV integrator this term would vanish and all tag choices would agree — that is the very content of the Riemann–Stieltjes existence theorem. For Brownian motion it does not vanish, so the three sums separate and stay separated as $n \to \infty$. The figure shows this for a single sample path on $[0, 1]$: as $n$ grows, the three running sums settle onto three distinct horizontal asymptotes separated by exactly $T/2$.

<figure>
  <img src="{{ '/assets/images/notes/random/rs_bm_tag_dependence.png' | relative_url }}" alt="Three tag rules for the sum of W times dW converge to three distinct limits separated by T/2" loading="lazy">
</figure>

### The fix: stochastic calculus

Both standard resolutions *commit to a tag rule* and use probabilistic structure on the integrand and on the notion of convergence:

* **Itô integral.** Always evaluate the integrand at the **left endpoint** $\xi_i = t_{i-1}$, restrict to **adapted** integrands ($f_t$ depends only on the information up to time $t$), and use $L^2$-convergence based on **Itô's isometry**:
  
  $$
  \mathbb{E}\!\left[\,\Bigl(\int_0^T f_t \, \mathrm{d}W_t\Bigr)^{\!2}\,\right] \;=\; \mathbb{E}\!\left[\,\int_0^T f_t^2 \, \mathrm{d}t\,\right].
  $$
  
  The left-endpoint convention makes the resulting integral a **martingale** — the property that makes Itô calculus the default in mathematical finance and stochastic analysis.

* **Stratonovich integral.** Use the **midpoint** rule. The resulting integral obeys the *ordinary* chain rule (no Itô correction term), making it the natural choice in physics and on manifolds, but loses the martingale property.

The two are linked by the **Itô–Stratonovich correction**

$$
\int_0^T f_t \circ \mathrm{d}W_t \;=\; \int_0^T f_t \, \mathrm{d}W_t \;+\; \frac{1}{2} \int_0^T \frac{\partial f}{\partial W}(t, W_t) \, \mathrm{d}t,
$$

whose extra $\frac{1}{2} \mathrm{d}t$ term is yet another manifestation of the **quadratic variation** $[W]\_t = t$.

### Summary in one line

Riemann–Stieltjes integration depends on the integrator having **bounded variation** (equivalently, vanishing quadratic variation). Brownian paths are pathwise of unbounded variation and have positive quadratic variation $[W]_t = t$, so the Riemann–Stieltjes sums fail to have a tag-independent limit, and one must instead choose a tagging convention up front and develop a *probabilistic* integration theory (Itô, Stratonovich, …) around it.

## Signed Borel measure

A **signed Borel measure** generalises the notion of a measure by allowing negative values. It is what makes "$\mathrm{d}g$" rigorous when $g$ is a BV function (and is the object underlying the Riemann–Stieltjes / Lebesgue–Stieltjes integral above).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Signed measure)</span></p>

A function $\mu : \mathcal{B}(X) \to [-\infty, +\infty]$ on the Borel $\sigma$-algebra of a topological space $X$ is a **signed measure** iff

1. $\mu(\emptyset) = 0$.
2. $\mu$ takes at most one of the values $+\infty$ or $-\infty$ (so that no expression of the form $\infty - \infty$ ever arises).
3. **Countable additivity**: for every disjoint sequence $(A_n) \subset \mathcal{B}(X)$,

$$
\mu\!\left(\bigsqcup_{n=1}^{\infty} A_n\right) = \sum_{n=1}^{\infty} \mu(A_n),
$$

with the series converging unconditionally.

It is **finite** if $\|\mu(X)\| < \infty$ and **$\sigma$-finite** if $X$ is a countable union of sets of finite total variation (defined below).

</div>

### Comparison with positive measures

| Property | Positive measure | Signed measure |
|---|---|---|
| Range | $[0, \infty]$ | $[-\infty, +\infty]$ (one infinity allowed) |
| Monotone? | $A \subseteq B \Rightarrow \mu(A) \leq \mu(B)$ | **No** (a subset can have more negative mass) |
| Countably additive? | Yes | Yes |
| Continuity from above/below? | Yes | Yes (with finite-mass caveats) |

### Examples

* **Difference of two positive measures**: $\mu = \mu_1 - \mu_2$ where at least one is finite. The Jordan decomposition (below) says *every* signed measure has this form, and uniquely so under a mutual-singularity constraint.
* **$\mu_g$ from a BV function**: for $g : [a, b] \to \mathbb{R}$ of bounded variation (and right-continuous), the rule $\mu_g((s, t]) := g(t) - g(s)$ extends to a signed Borel measure on $[a, b]$. This is the measure that makes "$\mathrm{d}g$" rigorous in $\int f \, \mathrm{d}g$.
* **Density against a positive measure**: if $f \in L^1(\mu)$ for a positive measure $\mu$, then $\nu(A) := \int_A f \, \mathrm{d}\mu$ is a signed measure (positive iff $f \geq 0$ a.e.). Radon–Nikodym goes the other way.
* **Physical "charge"**: net electric charge in a region; positive and negative contributions can cancel.

### Total variation measure

To every signed measure $\mu$ one associates a positive measure $\|\mu\|$, the **total variation measure**:

$$
|\mu|(A) \;:=\; \sup \!\left\lbrace \sum_{n} |\mu(A_n)| \,:\, (A_n) \text{ a measurable partition of } A \right\rbrace .
$$

The **total variation norm** is $\|\mu\| := \|\mu\|(X)$. The space of finite signed Borel measures with this norm is a Banach space, and by the Riesz representation theorem it is the dual of $C_0(X)$ when $X$ is locally compact Hausdorff.

## Hahn and Jordan decompositions

Hahn and Jordan are two faces of the same fundamental theorem about signed measures: **every signed measure splits canonically into a positive and a negative part**. Hahn states it set-theoretically (split the *space*); Jordan states it measure-theoretically (split the *measure*).

### Hahn decomposition (set-theoretic)

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Hahn decomposition)</span></p>

Let $\mu$ be a signed measure on $(X, \mathcal{A})$. There exist measurable sets $P, N \subseteq X$ with $X = P \sqcup N$ such that:

* $P$ is **positive** for $\mu$: $\mu(E) \geq 0$ for every measurable $E \subseteq P$.
* $N$ is **negative** for $\mu$: $\mu(E) \leq 0$ for every measurable $E \subseteq N$.

The pair $(P, N)$ is unique up to *$\mu$-null* sets: any other decomposition $(P', N')$ satisfies $\|\mu\|(P \triangle P') = 0$.

</div>

**Intuition.** Cut $X$ into the regions where $\mu$ is positive and where it is negative. On $P$, $\mu$ behaves like a non-negative measure; on $N$, it behaves like the negative of a non-negative measure.

**Construction sketch.** Repeatedly extract subsets of maximal $\mu$-mass: find $P_1 \subseteq X$ with $\mu(P_1) \geq \tfrac{1}{2} \sup\lbrace \mu(E) : E \in \mathcal{A} \rbrace$, then $P_2 \subseteq X \setminus P_1$ likewise, etc. Set $P := \bigcup P_n$. Countable additivity of $\mu$ together with the "at most one infinity" rule force $P$ to be positive and its complement negative. Non-uniqueness on $\mu$-null sets is unavoidable, since adding a $\mu$-null set to $P$ does not change positivity.

For an absolutely continuous signed measure $\mathrm{d}\mu = f(x) \, \mathrm{d}x$ on $[0, 1]$, the Hahn decomposition is *visibly* the partition of $[0, 1]$ into the sign-sets of $f$:

$$
P = \lbrace x : f(x) \geq 0 \rbrace, \qquad N = \lbrace x : f(x) < 0 \rbrace.
$$

<figure>
  <img src="{{ '/assets/images/notes/random/hahn_decomposition.png' | relative_url }}" alt="Hahn decomposition of a signed density on [0,1]: the positive set P and negative set N partition the interval according to the sign of the density" loading="lazy">
</figure>

### Jordan decomposition (measure-theoretic)

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Jordan decomposition of a signed measure)</span></p>

Every signed measure $\mu$ admits a *unique* decomposition

$$
\mu \;=\; \mu^{+} \;-\; \mu^{-}
$$

into two **mutually singular positive measures** $\mu^{+}, \mu^{-} \geq 0$ — that is, there exists a measurable set on which $\mu^{+}$ vanishes and $\mu^{-}$ is concentrated, and vice versa. The total variation measure is

$$
|\mu| \;=\; \mu^{+} + \mu^{-}.
$$

</div>

**Construction from Hahn.** Let $(P, N)$ be a Hahn decomposition. Define

$$
\mu^{+}(A) \;:=\; \mu(A \cap P), \qquad \mu^{-}(A) \;:=\; -\mu(A \cap N).
$$

Then:

* $\mu^{+}, \mu^{-} \geq 0$ by the positivity / negativity of $P, N$.
* $\mu^{+} + \mu^{-} \,=\, \mu(\,\cdot\, \cap P) - \mu(\,\cdot\, \cap N) \,=\, \|\mu\|$ on $A = (A \cap P) \sqcup (A \cap N)$.
* $\mu^{+} - \mu^{-} \,=\, \mu(\,\cdot\, \cap P) + \mu(\,\cdot\, \cap N) \,=\, \mu$.
* $\mu^{+} \perp \mu^{-}$: $\mu^{+}$ lives on $P$, $\mu^{-}$ lives on $N$, and $P \cap N = \emptyset$.

So Hahn $\to$ Jordan is mechanical. The reverse direction is also straightforward: given $\mu = \mu^{+} - \mu^{-}$ with $\mu^{+} \perp \mu^{-}$, the supports of $\mu^{+}$ and $\mu^{-}$ recover $(P, N)$ up to a null set.

For the absolutely continuous case $\mathrm{d}\mu = f \, \mathrm{d}x$, the Jordan decomposition reads off as

$$
\mathrm{d}\mu^{+} = f_{+} \, \mathrm{d}x, \qquad \mathrm{d}\mu^{-} = f_{-} \, \mathrm{d}x, \qquad \mathrm{d}|\mu| = |f| \, \mathrm{d}x ,
$$

with $f_{+} := \max(f, 0)$ and $f_{-} := \max(-f, 0)$.

<figure>
  <img src="{{ '/assets/images/notes/random/jordan_measure.png' | relative_url }}" alt="Jordan decomposition: positive part f+ = max(f,0), negative part f- = max(-f,0), total variation density |f|" loading="lazy">
</figure>

Notice the **disjoint supports** in the figure: $\mu^{+}$ lives where $f > 0$ and $\mu^{-}$ lives where $f < 0$. This is exactly the mutual singularity $\mu^{+} \perp \mu^{-}$.

### Why "mutually singular" matters for uniqueness

The decomposition $\mu = \mu_1 - \mu_2$ as a difference of positive measures is **not unique** without an extra constraint: e.g., $\mu = (\mu_1 + \rho) - (\mu_2 + \rho)$ for any positive $\rho$. Mutual singularity ($\mu^{+} \perp \mu^{-}$) pins down the *minimal* such decomposition — equivalently, the one with smallest total mass

$$
\mu^{+}(X) + \mu^{-}(X) \;=\; |\mu|(X).
$$

So Jordan is uniqueness *up to mutually singular positive parts*.

### Jordan decomposition for BV functions

There is a closely related — but distinct — function-level result, also due to Jordan, that connects BV functions to signed Lebesgue–Stieltjes measures:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Jordan decomposition of a BV function)</span></p>

A function $g : [a, b] \to \mathbb{R}$ is of bounded variation iff it can be written as

$$
g \;=\; g^{\uparrow} - g^{\downarrow}
$$

for two non-decreasing functions $g^{\uparrow}, g^{\downarrow} : [a, b] \to \mathbb{R}$. The minimal such decomposition is given by

$$
g^{\uparrow}(x) := \tfrac{1}{2}\bigl(V_a^x(g) + g(x) - g(a)\bigr), \qquad g^{\downarrow}(x) := \tfrac{1}{2}\bigl(V_a^x(g) - g(x) + g(a)\bigr),
$$

where $V_a^x(g)$ is the total variation of $g$ on $[a, x]$.

</div>

This is the bridge between BV functions and signed measures: each non-decreasing $g^{\uparrow}$ defines a positive Lebesgue–Stieltjes measure $\mu_{g^{\uparrow}}$, and the *measure-level* Jordan decomposition of $\mu_g$ is

$$
\mu_g \;=\; \mu_{g^{\uparrow}} \;-\; \mu_{g^{\downarrow}}.
$$

So the BV hypothesis on $g$ is what makes "$\mathrm{d}g$" make sense as a signed Borel measure: BV $\Rightarrow$ Jordan-decomposable into monotone parts $\Rightarrow$ each part defines a positive Stieltjes measure $\Rightarrow$ their difference is a signed Borel measure.

The figure below illustrates the construction for $g(x) = \sin(2\pi x)$ on $[0, 1]$. The total variation function $V(x)$ is non-decreasing and reaches $V_0^1(g) = 4$. Both $g^{\uparrow}$ and $g^{\downarrow}$ are non-decreasing — $g^{\uparrow}$ rises on intervals where $g$ is increasing and is flat elsewhere, and conversely for $g^{\downarrow}$. The dotted reconstruction confirms $g^{\uparrow} - g^{\downarrow} + g(0) = g$:

<figure>
  <img src="{{ '/assets/images/notes/random/jordan_bv_function.png' | relative_url }}" alt="Jordan decomposition of g(x) = sin(2 pi x) into two non-decreasing functions g_up and g_down" loading="lazy">
</figure>

### One-line summary

A **signed Borel measure** is a Borel set function with all properties of a measure except non-negativity. **Hahn** and **Jordan** are two equivalent ways of saying *it is canonically a difference of two positive measures*: Hahn does it by partitioning the *space*, Jordan by decomposing the *measure*. The function-level Jordan decomposition (for BV functions) is the analogue that connects the two worlds and is exactly what makes $\int f \, \mathrm{d}g$ well-defined when $g$ is BV.

## Why Itô calculus needs measure theory

Every key object in Itô calculus is a measure-theoretic object that **cannot** be defined or analysed by pathwise / classical-analysis methods. The figure below pictures the dependency: each layer of stochastic calculus rests on the layer below, with the foundations themselves living in measure theory.

<figure>
  <img src="{{ '/assets/images/notes/random/ito_dependency_stack.png' | relative_url }}" alt="Stack diagram: Itô calculus on top, Brownian motion below, probabilistic structures below that, measure-theoretic foundations at the bottom" loading="lazy">
</figure>

### 1. Brownian motion itself is a measure-theoretic construction

Itô calculus is built *on top of* Brownian motion, but $W$ is not a function — it is a **family of random variables** $\lbrace W_t : t \geq 0 \rbrace$ on a probability space $(\Omega, \mathcal{A}, \mathbb{P})$. The standard properties already require measure theory:

* **Existence**: Kolmogorov's extension theorem builds $W$ from a consistent family of finite-dimensional Gaussian distributions on $\mathbb{R}^I$ — the construction lives entirely in measure theory (cylinder $\sigma$-algebras, projective limits).
* **Continuity of paths** is a $\mathbb{P}$-almost-sure statement, repaired by Kolmogorov's continuity theorem via *modification on a null set*. Both the statement and the proof are measure-theoretic.
* **Sample paths are nowhere differentiable, with infinite total variation $\mathbb{P}$-a.s.** — you cannot do calculus on individual paths in the classical sense; you can only do it under $\mathbb{P}$.

So the very ground beneath Itô calculus is a probability space, not a Euclidean space.

### 2. The Itô integral cannot be defined pathwise

This is the central reason. We saw earlier that for a Brownian integrator,

$$
\sum_i f(\xi_i) \, \bigl[W_{t_i} - W_{t_{i-1}}\bigr]
$$

has **no tag-independent limit** — Riemann–Stieltjes integration *flatly fails*. The Itô integral $\int_0^T f_t \, \mathrm{d}W_t$ exists only as a limit in a *probabilistic* sense:

$$
\int_0^T f_t \, \mathrm{d}W_t \;:=\; L^2\text{-}\!\lim_{n \to \infty} \sum_i f_{t_{i-1}^{(n)}} \bigl(W_{t_i^{(n)}} - W_{t_{i-1}^{(n)}}\bigr).
$$

The "$L^2$-lim" requires:

* The Hilbert space $L^2(\Omega, \mathcal{A}, \mathbb{P})$ — measure-theoretic by definition.
* **Itô's isometry** $\mathbb{E}\bigl[(\int f \, \mathrm{d}W)^2\bigr] = \mathbb{E}\bigl[\int f^2 \, \mathrm{d}t\bigr]$ — both sides are Lebesgue integrals (the right-hand side over the product space $\Omega \times [0, T]$ via Fubini).
* A **density argument**: the integral is first defined for simple integrands, then extended by completeness of $L^2$ to all suitable predictable $f$. The argument relies on $L^2$ being a *complete* metric space — measure-theoretic.

Without $L^2(\Omega)$ and the dominated/monotone convergence theorems, there is no Itô integral.

The figure below illustrates Itô's isometry as a Monte Carlo identity. Take the deterministic integrand $f(t) = \sin(2\pi t)$, so $\int_0^T f^2 \, \mathrm{d}t = T/2 = 0.5$. The left panel shows the running sample mean of $(\int f \, \mathrm{d}W^{(k)})^2$ over $K$ Brownian paths, converging to $0.5$ — exactly the prediction of Itô's isometry. The right panel shows that the distribution of the Itô integral $\int_0^T f \, \mathrm{d}W$ is *Gaussian* with variance $\int_0^T f^2 \, \mathrm{d}t$ — a fact provable only inside the $L^2$-construction.

<figure>
  <img src="{{ '/assets/images/notes/random/ito_isometry.png' | relative_url }}" alt="Left: Monte Carlo running mean of squared Itô integral converges to ∫ f^2 dt. Right: histogram of integral values matches a centered Gaussian with that variance." loading="lazy">
</figure>

### 3. Filtrations encode information flow — that is, $\sigma$-algebras

A central concept in stochastic calculus is the **filtration** $(\mathcal{F}\_t)\_{t \geq 0}$, an increasing family of sub-$\sigma$-algebras of $\mathcal{A}$ where $\mathcal{F}\_t$ models "information available up to time $t$". The integrand $f_t$ in $\int f_t \, \mathrm{d}W_t$ must be **adapted** (or **predictable**) — meaning $f_t$ is $\mathcal{F}\_t$-measurable for each $t$ — so that one is "not allowed to peek into the future" when forming the Riemann sum.

This non-anticipation requirement is **the** thing that makes the left-endpoint convention (Itô) physically meaningful and turns the integral into a martingale. It is *literally a measurability condition* — there is no way to express it without $\sigma$-algebras.

The figure below makes the filtration tangible. At time $t^\ast = 0.55$, the path of $W$ to the left of the dashed line is the information visible to $\mathcal{F}\_{t^\ast}$; the shaded right-hand region is the *future*, which an adapted integrand is forbidden to see. The green dot shows an Itô-admissible value $f_{t^\ast} = \max_{s \leq t^\ast} W_s$ (depends only on the past); the red dot shows a *forbidden* anticipating choice $f_{t^\ast} = \max_{s \geq t^\ast} W_s$ (depends on the future, hence is **not** $\mathcal{F}\_{t^\ast}$-measurable).

<figure>
  <img src="{{ '/assets/images/notes/random/ito_adapted_vs_anticipating.png' | relative_url }}" alt="Brownian path with a vertical filtration boundary at t*: green dot uses past only (adapted, Itô-admissible); red dot peeks into the future (anticipating, not Itô-admissible)." loading="lazy">
</figure>

Stopping times $\tau$ are equally measure-theoretic: the condition "$\tau$ is an $\mathcal{F}\_t$-stopping time" means $\lbrace \tau \leq t \rbrace \in \mathcal{F}\_t$. The optional sampling theorem, localisation, and the construction of *local* martingales all rest on this.

### 4. Conditional expectation — and hence martingales — are measure-theoretic

A martingale is defined by

$$
\mathbb{E}[M_t \mid \mathcal{F}_s] \;=\; M_s \qquad \text{for } s \leq t .
$$

The conditional expectation $\mathbb{E}[\,\cdot\, \mid \mathcal{F}\_s]$ is **defined via Radon–Nikodym**: it is the (a.s.-unique) $\mathcal{F}\_s$-measurable random variable whose integrals over $\mathcal{F}\_s$-events match those of the original variable. It does not exist in classical analysis. Yet the entire theory of Itô integrals, semimartingales, and SDE solutions is organised around martingale and supermartingale properties.

### 5. Almost-sure properties dominate the theory

Statements in Itô calculus are typically of the form:

* "$X$ has continuous paths $\mathbb{P}$-a.s." (existence of solutions).
* "Two solutions $X, Y$ are pathwise unique iff $\mathbb{P}(X_t = Y_t \text{ for all } t) = 1$."
* "The exceptional set on which Itô's formula fails has probability zero."

All of these are inseparable from the notion of a **null set** — a measure-theoretic notion. The associated equivalence classes (modifications, indistinguishable processes) live one level above functions and require the language built in the *blindness-to-null-sets* section.

### 6. Itô's isometry, Itô's formula, and SDE existence rely on Lebesgue integration

The proofs of the foundational identities of Itô calculus — Itô's isometry, the Burkholder–Davis–Gundy inequalities, Itô's formula, Doob's maximal inequality — all use:

* Dominated and monotone convergence,
* Fubini's theorem (to swap $\mathbb{E}$ with $\int_0^T \cdots \, \mathrm{d}t$),
* Properties of $L^p(\Omega)$ (Hölder, BDG, completeness),
* Conditional Jensen's inequality.

These are **the** core theorems of Lebesgue integration; without them the standard Picard-iteration existence proof for SDE solutions, and the $L^2$-construction of the Itô integral itself, do not work.

### 7. Change of measure (Girsanov) is *intrinsically* about measures

Girsanov's theorem says: under an equivalent measure $\mathbb{Q} \ll \mathbb{P}$ with Radon–Nikodym derivative

$$
\frac{\mathrm{d}\mathbb{Q}}{\mathrm{d}\mathbb{P}} \;=\; \exp\!\left(-\int_0^T \theta_t \, \mathrm{d}W_t \;-\; \tfrac{1}{2} \int_0^T \theta_t^2 \, \mathrm{d}t\right) ,
$$

the process $W^{\mathbb{Q}}\_t := W_t + \int_0^t \theta_s \, \mathrm{d}s$ is a Brownian motion. This is **the** workhorse of mathematical finance (risk-neutral pricing) and stochastic control — and the entire statement is a sentence about *signed* Radon–Nikodym derivatives between equivalent probability measures. There is no way to phrase it without measure theory.

### 8. Modes of convergence

Random sequences can converge in *several* inequivalent ways: a.s., in $L^p$, in probability, in distribution. Each is a measure-theoretic construct, with its own machinery (Borel–Cantelli, dominated convergence, Slutsky). Itô calculus uses all of them: SDE solutions converge $\mathbb{P}$-a.s., the Itô integral converges in $L^2$, weak convergence of measures appears in the convergence of approximation schemes (e.g. Donsker $\to$ Brownian motion).

Treating all these convergence modes as "the same kind of limit" — as classical analysis must — would silently conflate genuinely different statements.

### Summary table

| Object in Itô calculus | What measure-theoretic ingredient it requires |
|---|---|
| Brownian motion | Probability space, Kolmogorov extension/continuity, null sets |
| The Itô integral | $L^2(\Omega, \mathcal{A}, \mathbb{P})$, Itô isometry, Fubini |
| Adapted / predictable integrands | Filtrations $\mathcal{F}\_t$, measurability w.r.t. sub-$\sigma$-algebras |
| Stopping times, localisation | Measurable events $\lbrace \tau \leq t \rbrace \in \mathcal{F}\_t$ |
| Martingale property | Conditional expectation via Radon–Nikodym |
| SDE existence and uniqueness | Picard iteration in $L^2$, dominated convergence |
| Pathwise vs. distributional uniqueness | Modifications, indistinguishability — null-set arguments |
| Girsanov change of measure | Equivalent measures, Radon–Nikodym derivative |
| Convergence of approximation schemes | Weak convergence of measures, tightness |

### One-line summary

Itô calculus is the calculus of **integrals against rough random integrators on a probability space**, and *every* word in that phrase — "rough" (positive quadratic variation), "random" ($L^2(\Omega)$), "integrator" (signed/martingale measures), "probability space" — is a measure-theoretic object. Without measure theory there is no probability space, no $L^2$ to take limits in, no $\sigma$-algebras to express adaptedness, no conditional expectations for martingales, no null sets for almost-sure properties, and ultimately no Itô integral.

## Why integrate against $\mathrm{d}W_t$ rather than $\mathrm{d}t$?

A natural question is: if the integrand $f$ is allowed to depend on $\omega$, why is the stochastic integral $\int_0^T f_t \, \mathrm{d}W_t$ a *new* object — why can't we absorb the randomness of $\mathrm{d}W$ into $f$ and reduce everything to a pathwise Lebesgue / Riemann integral $\int_0^T g(t, \omega) \, \mathrm{d}t$?

The short answer: **you cannot**, because the formal "thing" you would need to absorb — namely $\dot{W}\_t$ — *does not exist as a function*. The stochasticity in $\mathrm{d}W_t$ lives at a finer scale than $\mathrm{d}t$ and cannot be reduced to it.

### 1. SDEs already separate $\mathrm{d}t$ and $\mathrm{d}W_t$ on purpose

The standard form of an SDE is

$$
\mathrm{d}X_t \;=\; \underbrace{a(t, X_t) \, \mathrm{d}t}_{\text{drift}} \;+\; \underbrace{b(t, X_t) \, \mathrm{d}W_t}_{\text{diffusion / noise}} ,
$$

interpreted as the integral identity

$$
X_t = X_0 + \int_0^t a(s, X_s) \, \mathrm{d}s + \int_0^t b(s, X_s) \, \mathrm{d}W_s .
$$

The two integrals are *genuinely different objects*:

* $\int_0^t a(s, X_s) \, \mathrm{d}s$ is an ordinary **Lebesgue integral** in $s$ for each fixed $\omega$ (a.s.). The integrand is a continuous random function; pathwise Riemann integration works fine.
* $\int_0^t b(s, X_s) \, \mathrm{d}W_s$ is a **stochastic integral**. The integrand may be the same kind of well-behaved object, but the *integrator* $W$ has unbounded total variation, so this integral is not pathwise Riemann–Stieltjes and lives in $L^2(\Omega)$, not in any function space of $s$.

These two terms encode fundamentally different content: the drift is *predictable* (no surprises from $\omega$), while the diffusion is *uncorrelated random kicks* whose typical size scales like $\sqrt{\mathrm{d}t}$, not $\mathrm{d}t$.

### 2. Why you can't absorb the noise into $f$ and integrate against $\mathrm{d}t$

The natural attempt is "if $W$ is the integrator, write $\mathrm{d}W_t = \dot{W}_t \, \mathrm{d}t$, absorb the new factor into $f$, and reduce to":

$$
\int_0^T f(t) \, \mathrm{d}W_t \;\;\stackrel{?}{=}\;\; \int_0^T f(t) \, \dot{W}_t \, \mathrm{d}t .
$$

This **does not work** because $\dot{W}_t$ does not exist as a random variable. Specifically:

* By Lévy's modulus, Brownian paths are nowhere differentiable $\mathbb{P}$-a.s.: for every $t$, the difference quotient $(W_{t+h} - W_t)/h$ has typical size $1/\sqrt{h} \to \infty$ as $h \downarrow 0$. There is no finite-valued $\dot{W}_t$ to plug in.
* Even formally, $\dot{W}_t$ would have variance $\delta(0) = \infty$ (the "white noise" formal calculation): it cannot be a random *number*, only a random *distribution* (in Schwartz's sense). One cannot multiply a function by a distribution and integrate the product like a Lebesgue integral.

So the operation "$f \cdot \mathrm{d}W \to f \dot{W} \cdot \mathrm{d}t$" requires dividing by $\mathrm{d}t$, but $\dot{W}_t$ is *infinitely large* with random sign — there is nothing finite to absorb.

The figure below shows the difference quotient $(W(t^\ast + h) - W(t^\ast))/h$ for shrinking $h$, sampled at three base times $t^\ast \in \lbrace 0.2, 0.5, 0.8 \rbrace$ on a single Brownian path. The magnitudes track the dashed reference $1/\sqrt{h}$ (which $\to \infty$) — they do *not* settle on a finite limit as $h \downarrow 0$:

<figure>
  <img src="{{ '/assets/images/notes/random/difference_quotient_blowup.png' | relative_url }}" alt="Difference quotient |W(t*+h) - W(t*)|/h diverges like 1/sqrt(h) at three base times t*=0.2, 0.5, 0.8" loading="lazy">
</figure>

### 3. The deeper reason: $\mathrm{d}W_t$ and $\mathrm{d}t$ are scaled differently

Brownian increments scale as

$$
\mathrm{d}W_t \;\sim\; \sqrt{\mathrm{d}t}, \qquad \text{not} \qquad \mathrm{d}W_t \;\sim\; \mathrm{d}t .
$$

This is the meaning of $[W]_t = t$ and is the entire reason quadratic-variation effects (the "Itô correction") appear. A $\mathrm{d}t$-integral is **first-order** in the partition mesh: replacing $\mathrm{d}t$ by $\sqrt{\mathrm{d}t}$ would require a totally different limit theory. Concretely, the two integrals collect mass at different orders:

$$
\sum_i f(t_i)\, \Delta_i t \;\;\xrightarrow{n \to \infty}\;\; \int_0^T f(t)\, \mathrm{d}t \quad (\text{Riemann; needs only continuity of } f) ,
$$

$$
\sum_i f(t_i)\, \Delta_i W \;\;\xrightarrow{n \to \infty}\;\; \int_0^T f(t)\, \mathrm{d}W_t \quad (\text{Itô; needs adaptedness, } L^2\text{-limit, isometry}) .
$$

Even when both limits exist, they live in different worlds: the first is a smooth $\omega$-by-$\omega$ Riemann integral; the second is a Hilbert-space limit of random variables with variance $\int_0^T f(t)^2 \, \mathrm{d}t$.

The figure makes the scaling difference visible. As the partition mesh $\Delta t$ shrinks, the deterministic increment $\Delta t$ goes to zero linearly (slope 1 in log-log), while the typical Brownian increment $\sqrt{\mathbb{E}[(\Delta W)^2]}$ goes to zero only at rate $\sqrt{\Delta t}$ (slope $1/2$):

<figure>
  <img src="{{ '/assets/images/notes/random/dW_vs_dt_scaling.png' | relative_url }}" alt="Log-log plot of typical increment magnitude vs partition mesh: dt (slope 1, deterministic) and sqrt(E[dW^2]) (slope 1/2, Brownian)" loading="lazy">
</figure>

### 4. The discrete picture: random walk → Brownian motion

In a Donsker-type discrete model, an SDE looks like

$$
X_{k+1} - X_k \;=\; a(k, X_k) \cdot \Delta t \;+\; b(k, X_k) \cdot \sqrt{\Delta t} \cdot Z_k, \qquad Z_k \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1) .
$$

The two terms scale differently in $\Delta t$:

* The drift contributes $\Delta t$ per step — order-1 in time.
* The noise contributes $\sqrt{\Delta t}$ per step — half-order in time.

In the continuum limit these become $a(t, X_t) \, \mathrm{d}t$ and $b(t, X_t) \, \mathrm{d}W_t$. The reason there are *two* integrators is that there are **two different scalings** in the discrete model, and you cannot collapse them into one without erasing one or the other.

If you tried to absorb the noise into the $\mathrm{d}t$-integral, you'd need

$$
b(k, X_k) \cdot \sqrt{\Delta t} \cdot Z_k \;\;\stackrel{?}{=}\;\; \tilde{g}(k, X_k, \omega) \cdot \Delta t,
$$

i.e. $\tilde{g} = b \cdot Z_k / \sqrt{\Delta t}$, which **diverges** as $\Delta t \to 0$. This is exactly the formal "$\dot{W}_t$" problem: nothing finite plays the role of the absorbed factor.

### 5. Concrete example: the Itô correction is the "missing" $\frac{1}{2} T$

Take $f = W$ on $[0, T]$. The classical-chain-rule guess (i.e. pretending $\dot{W}$ exists) would give

$$
\int_0^T W_t \, \dot{W}_t \, \mathrm{d}t \;\;\stackrel{?}{=}\;\; \tfrac{1}{2} W_T^2 \qquad (\text{by } \mathrm{d}(W^2/2) = W \dot{W}\, \mathrm{d}t) .
$$

But the Itô integral is

$$
\int_0^T W_t \, \mathrm{d}W_t \;=\; \tfrac{1}{2} W_T^2 \;-\; \tfrac{1}{2} T .
$$

The extra **$-\frac{1}{2}T$** is the **Itô correction** and comes directly from the quadratic variation $[W]_T = T$. It is *not* an artefact of how you set up the integral — it is a real, measurable quantity present in the limit. Pretending you could absorb $\mathrm{d}W$ into $\mathrm{d}t$ would *erase this term*, and the resulting "integral" would conflict with experiment (e.g. the wrong drift in geometric Brownian motion / Black–Scholes). The $-\frac{1}{2}T$ is the price you pay for the integrator being rough; no amount of absorbing the noise into $f$ makes it disappear.

The figure below confirms this Monte-Carlo. The left panel scatters $\bigl(\frac{1}{2}W_T^2,\, \int_0^T W \, \mathrm{d}W\bigr)$ over $K = 800$ Brownian paths with $T = 1$. The points lie on the line $y = x - T/2$ (red dashed), *not* on the would-be classical line $y = x$ (black dotted). The right panel is the histogram of the differences $\frac{1}{2} W_T^2 - \int_0^T W \, \mathrm{d}W$: tightly concentrated around the *deterministic* value $T/2 = 0.5$ (the small spread is finite-time discretisation error). The Itô correction is **constant across all sample paths** — a fingerprint that no $\mathrm{d}t$-integral can produce.

<figure>
  <img src="{{ '/assets/images/notes/random/ito_correction_vs_classical.png' | relative_url }}" alt="Left: scatter of (W_T^2/2, Itô integral of W dW) lying on y = x - T/2, not y = x. Right: histogram of the difference concentrated at T/2 = 0.5." loading="lazy">
</figure>

### 6. The white-noise perspective: it doesn't simplify anything

There *is* a framework — Hida / white-noise calculus — that takes the formal $\dot{W}_t$ seriously by treating it as a random Schwartz distribution. In that framework one can write

$$
\int_0^T f(t) \, \mathrm{d}W_t \;=\; \int_0^T f(t) \, \dot{W}_t \, \mathrm{d}t \qquad (\text{interpreted distributionally}) .
$$

But:

* $\dot{W}_t$ is *not* a function — it lives in a space of distributions $\mathcal{S}'(\mathbb{R})$ that is itself measure-theoretic (Bochner / Minlos theorem on cylindrical Gaussian measures).
* Multiplying $f \cdot \dot{W}$ requires Wick products and Hida–Sobolev spaces — strictly *more* machinery, not less.
* The Itô correction reappears in the form of Wick-vs-pointwise multiplication conventions.

So even when you take the white-noise perspective, you are not "removing" the stochastic integral — you are repackaging it in an equally measure-theoretic vocabulary.

### 7. Bottom line

| Question | Answer |
|---|---|
| Can we write $\int f(t) \, \mathrm{d}W_t = \int f(t) \cdot \dot{W}_t \, \mathrm{d}t$? | No — $\dot{W}_t$ does not exist as a random variable. |
| Can we absorb the stochasticity into $f$? | No — the noise scales like $\sqrt{\mathrm{d}t}$, but $\mathrm{d}t$-integration only sees order-$\mathrm{d}t$ contributions. |
| Why is $\mathrm{d}W_t$ a *necessary* integrator? | Because Brownian increments are uncorrelated, mean-zero, of typical size $\sqrt{\mathrm{d}t}$, contributing at quadratic-variation order to sums. This contribution is invisible to $\mathrm{d}t$-integration but real (e.g. the $-\frac{1}{2}T$ in $\int W \, \mathrm{d}W$). |
| Where does the "extra" content of $\mathrm{d}W_t$ live? | In the quadratic variation $[W]_t = t$ — a probabilistic invariant of the integrator, with no analogue for $\mathrm{d}t$. |

**Summary.** $\mathrm{d}t$ is "smooth time" and integrates first-order quantities; $\mathrm{d}W_t$ is "rough random time" and integrates *half-order* quantities. They are not interchangeable, and there is no nontrivial way to express one as the other. The whole reason stochastic calculus needs its own integral against $\mathrm{d}W_t$ — with its own convergence theory, isometry, and chain rule (Itô's formula) — is precisely that the $\mathrm{d}W_t$-integral captures content (quadratic variation, martingale property, Itô correction) that no $\mathrm{d}t$-integral with a clever integrand can reproduce.

## What integrating against $\mathrm{d}W_t$ accomplishes (the positive answer)

The previous section explained why you *cannot* reduce $\int f \, \mathrm{d}W$ to a $\mathrm{d}t$-integral. This section gives the complementary positive picture: *what* the stochastic integral $\int f \, \mathrm{d}W$ actually computes, *why* $W_t$ specifically is the right integrator, and *how* the same noise source produces genuinely different processes when filtered through different response functions.

### The right question

Forget for a moment whether $\dot{W}$ exists. Ask instead: *what kind of object* is $\int_0^T f_t \, \mathrm{d}W_t$, and *what is it computing*?

The answer is: **it accumulates time-varying responses to random shocks**.

* $W_t$ is a model for "cumulative random noise up to time $t$".
* The increment $\mathrm{d}W_t$ (more precisely $W_{t + \mathrm{d}t} - W_t$) is the **random shock** in the infinitesimal interval $[t, t + \mathrm{d}t]$.
* $f_t$ is the **sensitivity** / **exposure** / **response coefficient** to that shock at time $t$.
* The integral $\int_0^T f_t \, \mathrm{d}W_t$ is the total weighted accumulation of shocks: at each instant $t$, take the random shock $\mathrm{d}W_t$, scale it by $f_t$, and sum.

So $\mathrm{d}W_t$ plays the role of "**the random thing being weighted**", while $f_t$ plays the role of "**how strongly we respond to it**". You cannot reduce this to a $\mathrm{d}t$-integral because in a $\mathrm{d}t$-integral *there is no random thing being weighted* — $\mathrm{d}t$ is deterministic.

### Why $W_t$ specifically — out of all possible integrators

Brownian motion is *uniquely characterised* (Lévy's theorem) by four properties:

| Property of $W$ | What it gives the integrator |
|---|---|
| Independent increments | random shocks across disjoint intervals are uncorrelated — "memoryless noise" |
| Stationary increments | the noise distribution is the same at every time — "time-homogeneous" |
| Continuous paths | no jumps — appropriate for "many small kicks" rather than "rare large shocks" |
| $W_t \sim \mathcal{N}(0, t)$ | each increment is Gaussian — the natural CLT-limit of independent perturbations |

These are exactly the four properties one wants from a model of **continuous Gaussian noise integrated over time**. They are not arbitrary: they are forced by the natural physical / modelling assumption that the noise is the cumulative effect of many small independent disturbances (Donsker's theorem makes this precise — every diffusive scaling limit of bounded-variance random walks is Brownian motion).

So **we integrate against $\mathrm{d}W_t$ because $W_t$ is the universal model of continuous Gaussian noise**, and $\int f \, \mathrm{d}W$ is the natural object that accumulates a time-varying response to it.

### Other choices of integrator exist — and they have other roles

Integrating against $W$ is not the *only* option; it is the *canonical Gaussian-noise* option. Other integrators model other phenomena:

| Integrator $X$ | Models | Stochastic integral $\int f \, \mathrm{d}X$ represents |
|---|---|---|
| $\mathrm{d}t$ | smooth time | ordinary deterministic accumulation (drift) |
| $\mathrm{d}W_t$ | continuous Gaussian noise | cumulative response to Brownian shocks |
| $\mathrm{d}N_t$ (Poisson) | rare random jumps | cumulative jump-triggered effect |
| $\mathrm{d}\widetilde{N}_t$ (compensated Poisson) | jump noise without drift | jump martingale integral |
| $\mathrm{d}M_t$ (martingale) | general "fair game" noise | general Itô-type integral |
| $\mathrm{d}X_t$ (semimartingale) | drift + martingale | most general stochastic integral |

The reason **$W$** is the default in textbooks is that it is the "Gaussian, continuous" prototype — every other integrator that one builds (jump-diffusions, semimartingales, …) is constructed from $W$ and Poisson processes via the Lévy–Itô decomposition.

### Same noise, different responses → different processes

The picture below makes the "weighted accumulation of shocks" interpretation tangible. **Top panel:** a single Brownian sample path $W_t$ — the common noise source. **Middle panel:** five response functions $f$ — uniform exposure $f \equiv 1$, increasing $f(t) = t$, decreasing $f(t) = T - t$, early-only $f = \mathbf{1}_{[0, T/2]}$, oscillating $f(t) = \sin(2\pi t)$. **Bottom panel:** the running Itô integrals $M_t(f) := \int_0^t f(s) \, \mathrm{d}W_s$ on the *same* path.

<figure>
  <img src="{{ '/assets/images/notes/random/dW_response_functions.png' | relative_url }}" alt="Three-panel figure: Brownian path on top, five response functions f(t) in the middle, running Itô integrals at the bottom, demonstrating that different f's filter the same noise into different processes." loading="lazy">
</figure>

What to read off:

* **$f \equiv 1$** (blue): integral is exactly $W_t$ — the unweighted accumulator just *is* the noise process.
* **$f(t) = t$** (green): early shocks count for little, late shocks dominate. The integral grows mostly toward the right end of $[0, T]$.
* **$f(t) = T - t$** (orange): the opposite — early shocks dominate, late ones are damped. The integral makes its big moves early and then "settles".
* **$f(t) = \mathbf{1}_{[0, T/2]}$** (red): the integrator literally **freezes** after $t = T/2$, because $f$ is zero there — the shocks are still happening but they are not being recorded. This visually shows that $f$ is the *gate* through which noise enters the integral.
* **$f(t) = \sin(2\pi t)$** (purple): negative on $(1/2, 1)$, so the integral *unwinds* late shocks in the opposite direction.

Crucially, *all five integrals are driven by the same $W_t$*. The differences between them are entirely the work of $f$. There is no clever way to write any of them as $\int g(t, \omega) \, \mathrm{d}t$ for some function $g$ — the random "thing being weighted" lives in $\mathrm{d}W$ itself.

### Where $\mathrm{d}W_t$ shows up — concrete interpretations

In each application below, $\mathrm{d}W_t$ is the "**unit of randomness**" being aggregated and $f_t$ is the "**weighting**":

* **Mathematical finance.** $W_t$ models the random part of asset log-returns. If $f_t$ is the number of shares you hold at time $t$ and $S_t$ is the price (with $\mathrm{d}S_t / S_t = \mu \, \mathrm{d}t + \sigma \, \mathrm{d}W_t$), then
  
  $$
  \int_0^T f_t \, \mathrm{d}S_t = \int_0^T f_t \, \mu \, S_t \, \mathrm{d}t + \int_0^T f_t \, \sigma \, S_t \, \mathrm{d}W_t
  $$
  
  is your cumulative P&L. The $\mathrm{d}W$ part is the **random** P&L; you cannot model trading without it.
* **Physics (Langevin equation).** $m \dot{v} = -\gamma v + \sqrt{2 \gamma k_B T} \, \xi$ with $\xi = \dot{W}$ formally; the integrated form $\mathrm{d}v = -(\gamma / m) v \, \mathrm{d}t + (\sqrt{2 \gamma k_B T} / m) \, \mathrm{d}W_t$ writes the random thermal force *as an integrator* because the molecular kicks are too rough to be a function. $\int (\text{response}) \, \mathrm{d}W$ is then the cumulative momentum transfer.
* **Filtering / Kalman.** The "innovation process" of a Kalman filter is a Brownian motion under the optimal filter measure; the filter integrates the prediction error against this $\mathrm{d}W$ with the **Kalman gain** as $f_t$.
* **Stochastic control / RL.** Cost functionals of the form $\int_0^T \ell(t, X_t) \, \mathrm{d}t + \int_0^T \sigma(t, X_t) \, \mathrm{d}W_t$ have a deterministic accrual term and a **martingale noise term**; the latter is what stochastic Bellman equations have to handle.
* **Diffusion generative models.** The forward noising SDE $\mathrm{d}X_t = -\frac{1}{2} \beta_t X_t \, \mathrm{d}t + \sqrt{\beta_t} \, \mathrm{d}W_t$ literally *defines* the corruption process via integration against $\mathrm{d}W$; $f_t = \sqrt{\beta_t}$ is the **noise schedule**.

In every one of these, the $\mathrm{d}W$ term answers a specific modelling question: **"how do random shocks at each instant in time accumulate, weighted by how much I respond to them?"** That is what no $\mathrm{d}t$-integral can express, because $\mathrm{d}t$ has no random shock to weight.

### One-sentence answer

We integrate against $\mathrm{d}W_t$ because $W_t$ is the universal carrier of **continuous Gaussian noise**, and $\int_0^T f_t \, \mathrm{d}W_t$ is the natural operation that **accumulates a time-varying response** $f_t$ **to that noise** — a question that simply cannot be posed in the language of $\mathrm{d}t$-integration alone, since $\mathrm{d}t$ knows nothing about random shocks.

## Infinitesimal generator of a Markov / diffusion process

The **infinitesimal generator** $\mathscr{L}$ is the operator that plays the role of "derivative of the process at the present instant", just as the time derivative $\partial_t$ does for a deterministic flow. For a Markov process $(X_t)_{t \geq 0}$ with state space $\mathbb{R}^d$, $\mathscr{L}$ is the linear operator that, applied to a smooth test function $f$, returns the *expected instantaneous rate of change* of $f$ along the process:

$$
(\mathscr{L} f)(x) \;=\; \lim_{t \downarrow 0} \frac{\mathbb{E}\!\left[f(X_t) \mid X_0 = x\right] - f(x)}{t}.
$$

Everything else in the theory — Kolmogorov's PDEs, Itô's formula in expectation, Feynman–Kac, Dynkin's formula, even the score-based view of diffusion generative models — is a consequence of how this single operator looks for the process in question.

### From the Markov semigroup to its generator

The starting point is not the operator itself but the **Markov semigroup** of the process. For each $t \geq 0$ define a linear operator $P_t$ acting on bounded measurable functions by

$$
(P_t f)(x) \;:=\; \mathbb{E}\!\left[f(X_t) \mid X_0 = x\right] \;=\; \int f(y) \, p(t, x, \mathrm{d}y),
$$

i.e. *integrate $f$ against the transition kernel of the process started at $x$*. By the Markov property,

$$
P_{t + s} \;=\; P_t \, P_s \qquad \text{(semigroup property)},
$$

with $P_0 = \mathrm{Id}$. So $\lbrace P_t\rbrace\_{t \geq 0}$ is a one-parameter semigroup of operators — formally analogous to $e^{tA}$ for a matrix $A$.

Just as $A = \frac{\mathrm{d}}{\mathrm{d}t}\big\|_{t=0} e^{tA}$, the **generator** is the derivative of the semigroup at $t = 0$:

$$
\boxed{\;\mathscr{L} f \;:=\; \lim_{t \downarrow 0} \frac{P_t f - f}{t}\;}
$$

(on the domain of $f$'s where the limit exists in a suitable norm). Heuristically:

$$
P_t \;=\; e^{t \mathscr{L}}, \qquad \frac{\mathrm{d}}{\mathrm{d}t} P_t f \;=\; \mathscr{L} P_t f \;=\; P_t \mathscr{L} f.
$$

This last identity is the **Kolmogorov backward equation** in disguise — and it says the entire stochastic dynamics is recoverable from the single operator $\mathscr{L}$ by exponentiation.

### What $\mathscr{L}$ looks like for a diffusion (the Itô-formula derivation)

Suppose $X_t$ solves the SDE

$$
\mathrm{d} X_t \;=\; b(X_t) \, \mathrm{d} t \;+\; \sigma(X_t) \, \mathrm{d} W_t, \qquad X_0 = x,
$$

with drift $b: \mathbb{R}^d \to \mathbb{R}^d$ and diffusion coefficient $\sigma: \mathbb{R}^d \to \mathbb{R}^{d \times m}$ (so $W$ is $m$-dimensional Brownian motion). Apply Itô's formula to $f \in C^2(\mathbb{R}^d)$:

$$
\mathrm{d} f(X_t) \;=\; \nabla f(X_t)^\top \, b(X_t) \, \mathrm{d} t \;+\; \tfrac{1}{2} \, \mathrm{tr}\!\left(\sigma(X_t) \sigma(X_t)^\top \, \mathrm{Hess}\, f(X_t)\right) \mathrm{d} t \;+\; \nabla f(X_t)^\top \sigma(X_t) \, \mathrm{d} W_t.
$$

Integrate from $0$ to $t$ and take expectation: the $\mathrm{d}W$-term is a martingale and vanishes in expectation (started from $x$), leaving

$$
\mathbb{E}\!\left[f(X_t)\right] - f(x) \;=\; \mathbb{E} \int_0^t \!\Big(b \cdot \nabla f \;+\; \tfrac{1}{2} \mathrm{tr}(\sigma \sigma^\top \mathrm{Hess}\, f)\Big)(X_s) \, \mathrm{d} s.
$$

Divide by $t$ and let $t \downarrow 0$. The right-hand side tends to the integrand at $s = 0$ — and that integrand is, by definition, $(\mathscr{L} f)(x)$. We have *derived* the explicit formula for the generator of an Itô diffusion:

$$
\boxed{\;(\mathscr{L} f)(x) \;=\; \sum_{i=1}^d b_i(x) \, \partial_i f(x) \;+\; \tfrac{1}{2} \sum_{i,j=1}^d a_{ij}(x) \, \partial_i \partial_j f(x), \qquad a := \sigma \sigma^\top.\;}
$$

So **$\mathscr{L}$ is a second-order partial differential operator**: a first-order "drift" piece $b \cdot \nabla$ and a second-order "noise" piece $\frac{1}{2} a : \nabla^2$. Two facts to internalise:

* The *drift* $b$ shows up with $\nabla f$ — it transports the process deterministically.
* The *diffusion* $\sigma$ shows up only through $a = \sigma \sigma^\top$ and only with second derivatives — that is the **Itô correction** at the operator level. Stochasticity, at the level of $\mathscr{L}$, is *exactly* the second-order term.

The matrix $a(x) = \sigma(x) \sigma(x)^\top$ is symmetric positive semidefinite. If $a(x)$ is strictly positive definite for every $x$, $\mathscr{L}$ is an **(uniformly) elliptic operator**; this is what guarantees the noise is "non-degenerate in every direction" and yields smooth transition densities, strong existence/uniqueness, etc.

### A concrete example: the Ornstein–Uhlenbeck generator at work

Take the OU process $\mathrm{d} X_t = -\theta X_t \, \mathrm{d} t + \sigma \, \mathrm{d} W_t$ with $\theta = \sigma = 1$. The general formula above specialises to

$$
(\mathscr{L} f)(x) \;=\; -x \, f'(x) \;+\; \tfrac{1}{2} f''(x).
$$

The picture below shows the operator in action on three test functions $f$. Top row: the test function $f$ itself. Bottom row: $\mathscr{L} f$, the *expected instantaneous rate of change* of $f$ along the OU process at each starting point $x$.

<figure>
  <img src="{{ '/assets/images/notes/random/generator_action_on_test_functions.png' | relative_url }}" alt="2x3 grid showing three test functions f(x) — quadratic x^2, sin(πx), and Gaussian exp(-x²/2) — in the top row, with the action of the OU generator Lf = -x f'(x) + (1/2) f''(x) plotted in the bottom row." loading="lazy">
</figure>

Read the columns:

* **$f(x) = x^2$.** Then $\mathscr{L} f = 1 - 2 x^2$. At $x = 0$, $\mathscr{L} f = 1 > 0$: the process diffuses *outward* from the origin, so $\mathbb{E}[X_t^2 \mid X_0 = 0]$ is increasing. At $\|x\| > 1/\sqrt{2}$, $\mathscr{L} f < 0$: the mean-reversion $-\theta x$ is strong enough to drag $X_t^2$ down. The zero-crossings $x = \pm 1/\sqrt{2}$ identify the **stationary level** of $X^2$, i.e. the variance of the invariant Gaussian, namely $\mathrm{Var}\_\pi(X) = \sigma^2 / (2\theta) = 1/2$.
* **$f(x) = \sin(\pi x)$.** The diffusion term $\frac{1}{2} f''$ contributes $-\frac{\pi^2}{2} \sin(\pi x)$: noise *smooths* oscillating $f$ toward its mean, exactly because the second derivative of an oscillation is large and opposes the bumps. The drift adds a sign-twisting $-\pi x \cos(\pi x)$ term.
* **$f(x) = e^{-x^2/2}$.** The bell shape gets bent: noise *flattens* it (negative $\mathscr{L} f$ near $x = 0$), drift pulls mass toward $0$ (negative $\mathscr{L} f$ on the wings).

In every case, the *sign* of $(\mathscr{L} f)(x)$ tells you which way $\mathbb{E}\!\left[f(X_t) \mid X_0 = x\right]$ moves at the very first instant, and the *magnitude* tells you how fast.

### Verifying the limit definition numerically

The definition of $\mathscr{L}$ is a limit; let us watch that limit happen on the OU process. Take $f(x) = x^2$, for which we already computed $\mathscr{L} f(x) = 1 - 2 x^2$. Using the closed-form OU transition $X_t \mid X_0 = x_0 \sim \mathcal{N}\left(x_0 e^{-t}, \, \tfrac{1}{2}(1 - e^{-2t})\right)$, one computes

$$
\frac{(P_t f - f)(x_0)}{t} \;=\; \frac{(1 - e^{-2 t})\,(\tfrac{1}{2} - x_0^2)}{t} \;\xrightarrow[t \downarrow 0]{}\; 2 \cdot (\tfrac{1}{2} - x_0^2) \;=\; 1 - 2 x_0^2 \;=\; (\mathscr{L} f)(x_0). \checkmark
$$

The figure plots the left-hand side as a function of $t$ (log scale) at three starting points $x_0 \in \{-1, 0, 1.5\}$ and shows the corresponding theoretical limits $\mathscr{L} f(x_0) \in \{-1, 1, -3.5\}$ as dashed horizontal lines:

<figure>
  <img src="{{ '/assets/images/notes/random/generator_limit_verification.png' | relative_url }}" alt="Log-scale plot of (P_t f - f)/t as a function of t for the OU process with f(x)=x^2 at three starting points x_0=-1, 0, 1.5; each curve converges as t→0 to the dashed horizontal line giving the theoretical limit Lf(x_0)=1-2x_0^2." loading="lazy">
</figure>

The convergence is exactly what the definition says: the difference quotient $(P_t f - f)/t$ collapses onto $(\mathscr{L} f)(x_0)$ as $t \downarrow 0$, *for every starting point* $x_0$.

### A small zoo of generators

| Process | Drift $b$ / jump rate | Diffusion $\sigma$ | Generator $\mathscr{L}$ |
|---|---|---|---|
| Brownian motion $W_t$ in $\mathbb{R}^d$ | $0$ | $I$ | $\frac{1}{2} \Delta f$ |
| Drifted Brownian motion $W_t + \mu t$ | $\mu$ | $I$ | $\mu \cdot \nabla f + \frac{1}{2} \Delta f$ |
| Ornstein–Uhlenbeck $\mathrm{d} X = -\theta X \, \mathrm{d} t + \sigma \, \mathrm{d} W$ | $-\theta x$ | $\sigma$ | $-\theta x \cdot \nabla f + \frac{\sigma^2}{2} \Delta f$ |
| Geometric Brownian motion $\mathrm{d} S = \mu S \, \mathrm{d} t + \sigma S \, \mathrm{d} W$ | $\mu x$ | $\sigma x$ | $\mu x \, f'(x) + \frac{\sigma^2 x^2}{2} f''(x)$ |
| Compound Poisson with intensity $\lambda$, jump law $\nu$ | — | — | $\lambda \int \!\big(f(x + y) - f(x)\big) \, \nu(\mathrm{d}y)$ |
| Continuous-time Markov chain on a discrete state space | — | — | $(\mathscr{L} f)(i) = \sum_j Q_{ij} f(j)$,  $Q$ the rate matrix |

Two patterns to notice:

* **Diffusions** give *second-order differential* operators.
* **Jump processes** give *integral* operators (or, for discrete chains, *matrix* operators).

The general Lévy–Khintchine generator combines both: drift, second-order diffusion, and a Lévy jump integral.

### The two PDEs that fall out — backward and forward (Fokker–Planck)

Because $P_t = e^{t \mathscr{L}}$, two evolution equations are immediate.

**Kolmogorov backward equation.** Apply $\partial_t$ to $u(t, x) := (P_t f)(x) = \mathbb{E}\left[f(X_t) \mid X_0 = x\right]$:

$$
\partial_t u(t, x) \;=\; \mathscr{L}_x u(t, x), \qquad u(0, x) = f(x).
$$

This describes the evolution of *the conditional expectation as a function of the starting point* $x$ — hence "backward" (in the initial condition).

**Kolmogorov forward / Fokker–Planck equation.** Apply $\partial_t$ to the *density* $p(t, x)$ of $X_t$ (assuming it exists and starts from a known $p_0$). The duality $\langle P_t f, p_0 \rangle = \langle f, P_t^{\ast} p_0 \rangle$ gives that the density evolves under the *adjoint* $\mathscr{L}^{\ast}$:

$$
\partial_t p(t, x) \;=\; \mathscr{L}^{*} p(t, x), \qquad p(0, \cdot) = p_0.
$$

For a non-degenerate diffusion this is the explicit Fokker–Planck PDE

$$
\partial_t p \;=\; -\nabla \cdot (b \, p) \;+\; \tfrac{1}{2} \, \nabla^2 : (a \, p),
$$

i.e. drift becomes a transport term and diffusion becomes a divergence-of-flux term. The picture below shows this evolution for the OU process started at $x_0 = 2$, sampled and overlaid with the analytic transition density at four times. The dashed curve is the **invariant density** $\pi = \mathcal{N}(0, 1/2)$, which is precisely the stationary solution $\mathscr{L}^{*} \pi = 0$:

<figure>
  <img src="{{ '/assets/images/notes/random/generator_fokker_planck.png' | relative_url }}" alt="Four panels showing the OU transition density at times t=0.1, 0.5, 1.0, 3.0 starting from x_0=2: empirical histograms (60000 samples) overlaid with the analytic Gaussian transition and the dashed invariant density N(0, 1/2). The density starts as a sharp bump near x=2 and progressively converges to the invariant Gaussian." loading="lazy">
</figure>

What you see: the density starts as a sharp Gaussian bump near $x = 2$ (the deterministic starting point), then *both* drifts back toward $0$ (driven by $b(x) = -x$) *and* spreads out (driven by $\frac{1}{2} \partial_x^2$). At $t = 3$ it is essentially indistinguishable from the invariant $\pi$. The whole picture is the operator $\mathscr{L}^{*}$ at work in real time.

### Why $\mathscr{L}$ is *the* invariant of the process

Several different-looking objects all encode the same information about a Markov process; the generator is the lightest among them.

* **Transition kernel $p(t, x, \cdot)$.** Heaviest representation; one function for every $t > 0$.
* **Semigroup $P_t = e^{t \mathscr{L}}$.** One operator for every $t > 0$, but they are all generated by one of them.
* **Generator $\mathscr{L}$.** A *single* operator. Contains all transition kernels via exponentiation, all conditional expectations via the backward PDE, and all densities via the forward PDE.

This is why textbook taxonomies of stochastic processes (Brownian, OU, GBM, CIR, jump-diffusion, …) are organised by their generators: they are the most economical complete description.

### Where the generator shows up in the wild

Once you have $\mathscr{L}$, an enormous amount of probabilistic content follows mechanically.

* **Dynkin's formula.** For nice $f$ and stopping time $\tau$,
  
  $$
  \mathbb{E}\!\left[f(X_\tau) \mid X_0 = x\right] \;=\; f(x) \;+\; \mathbb{E}\!\left[\int_0^\tau (\mathscr{L} f)(X_s) \, \mathrm{d}s \,\Big|\, X_0 = x\right].
  $$
  
  This is the workhorse for computing exit-time distributions, ruin probabilities, hitting times, etc.
* **Feynman–Kac.** The PDE $\partial_t u + \mathscr{L} u - V u + h = 0$ with terminal condition $u(T, x) = g(x)$ has the probabilistic representation
  
  $$
  u(t, x) \;=\; \mathbb{E}\!\left[g(X_T) \, e^{-\int_t^T V(X_s)\,\mathrm{d}s} + \int_t^T h(X_s) \, e^{-\int_t^s V(X_r)\,\mathrm{d}r} \, \mathrm{d}s \,\Big|\, X_t = x\right].
  $$
  
  This is the bridge between linear parabolic PDEs and Markov processes; in finance it is what underlies risk-neutral pricing.
* **Martingale problem (Stroock–Varadhan).** A process $X_t$ is "the diffusion with generator $\mathscr{L}$" *iff* $f(X_t) - \int_0^t (\mathscr{L} f)(X_s) \, \mathrm{d}s$ is a martingale for every $f \in C_c^\infty$. This is an alternative *definition* of the process, often more flexible than the SDE itself (no need for a Brownian filtration up front).
* **Carré du champ and functional inequalities.** The bilinear form $\Gamma(f, g) := \frac{1}{2}\!\left(\mathscr{L}(fg) - f \mathscr{L} g - g \mathscr{L} f\right)$ — the *carré du champ* — measures "how much variance the noise injects into $f$" per unit time. For the OU generator, $\Gamma(f, f) = \frac{1}{2} (f')^2$. Spectral / functional inequalities (Poincaré, log-Sobolev) on $\mathscr{L}$ control mixing rates and concentration; this is the analytic backbone of modern probability.
* **Diffusion generative models.** The forward noising SDE $\mathrm{d} X_t = -\frac{1}{2} \beta_t X_t \, \mathrm{d}t + \sqrt{\beta_t} \, \mathrm{d} W_t$ has generator $\mathscr{L}_t f = -\frac{\beta_t}{2} x \cdot \nabla f + \frac{\beta_t}{2} \Delta f$ (a time-inhomogeneous OU); the reverse-time SDE used for sampling is built from the *adjoint* of this generator together with the score $\nabla \log p_t$. The whole "score-based" framework is, at the operator level, a statement about $\mathscr{L}$ and $\mathscr{L}^{*}$.

### One-sentence summary

The infinitesimal generator $\mathscr{L}$ is the **time-derivative at $t = 0$** of the Markov semigroup $P_t = e^{t \mathscr{L}}$; for an Itô diffusion it is the second-order elliptic operator $\mathscr{L} f = b \cdot \nabla f + \frac{1}{2} \mathrm{tr}(\sigma \sigma^\top \mathrm{Hess}\, f)$ (drift + Itô-corrected diffusion), and from this single operator one recovers the Kolmogorov backward and Fokker–Planck PDEs, expectations along the path (Dynkin), Feynman–Kac, the martingale-problem characterisation, and — ultimately — the score-based reverse-SDE machinery of diffusion generative models.

## Filtration

A **filtration** is the mathematical formalisation of "the flow of information through time" — an increasing family of σ-algebras, one for every time $t$, where $\mathcal{F}_t$ encodes everything that has been *observed / decided / become measurable* by time $t$. It is the missing first ingredient that gives meaning to the words "by time $t$" or "given the past" in every subsequent definition (adapted, martingale, stopping time, BM-with-respect-to-$\mathbb{F}$, Itô integral, …).

### Reading the notation

Textbooks compress a lot into a line like "let $\mathbb{F} := (\mathcal{F}\_t)\_{t \in I}$ be a filtration in $\mathcal{A}$." Decomposed:

| Symbol | What it is | Role |
|---|---|---|
| $\mathcal{A}$ | a fixed σ-algebra on the sample space $\Omega$ | the *ambient* universe of "all events that could ever be considered" — typically the σ-algebra carrying the probability measure $\mathbb{P}$ |
| $I$ | the **index set** (= "time set") | typically $I = \mathbb{N}_0$ (discrete time), $I = [0, T]$ (finite horizon), or $I = [0, \infty)$ (infinite horizon) |
| $\mathcal{F}_t$ | a σ-algebra on $\Omega$, one for each $t \in I$ | "what is observable / decidable by time $t$" |
| $\mathbb{F}$ | a *shorthand name* for the whole indexed family $(\mathcal{F}\_t)\_{t \in I}$ | one bold letter for "the entire filtration" — convenient when writing things like "$X$ is $\mathbb{F}$-adapted" |

The phrase "*in $\mathcal{A}$*" is the requirement

$$
\mathcal{F}_t \;\subseteq\; \mathcal{A} \qquad \text{for every } t \in I.
$$

That is: every "time-$t$ knowledge" σ-algebra lives inside the master σ-algebra $\mathcal{A}$. You are never observing more than the universe contains.

### The actual definition

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Filtration)</span></p>

Let $(\Omega, \mathcal{A})$ be a measurable space and $I \subseteq [0, \infty)$ a totally ordered index set. A **filtration in $\mathcal{A}$** indexed by $I$ is a family

$$
\mathbb{F} \;=\; (\mathcal{F}_t)_{t \in I}
$$

of sub-σ-algebras of $\mathcal{A}$ such that

$$
\mathcal{F}_s \;\subseteq\; \mathcal{F}_t \;\subseteq\; \mathcal{A} \qquad \text{whenever } s \leq t.
$$

The triple $(\Omega, \mathcal{A}, \mathbb{F})$ is a **filtered measurable space**; if a probability measure $\mathbb{P}$ is fixed on $(\Omega, \mathcal{A})$, the quadruple $(\Omega, \mathcal{A}, \mathbb{P}, \mathbb{F})$ is a **filtered probability space** — the standard setting for stochastic processes.

</div>

So the *only* axiom — beyond "each $\mathcal{F}_t$ is a sub-σ-algebra of the ambient $\mathcal{A}$" — is **monotonicity in $t$**:

$$
\boxed{\;s \leq t \;\Longrightarrow\; \mathcal{F}_s \;\subseteq\; \mathcal{F}_t.\;}
$$

That single inclusion encodes the asymmetry of time: information can only *grow*, never shrink. Once an event is "decided" (it happened or it didn't), it stays decided forever.

### Three equivalent ways to picture it

* **As an information ladder.** $\mathcal{F}_t$ is the collection of events whose occurrence has been *resolved* by time $t$. As $t$ increases, more and more events become resolved. The filtration is the entire "ladder" of resolution levels.
* **As partition refinement.** Each σ-algebra $\mathcal{F}_t$ corresponds (up to refinement, in nice cases) to a partition of $\Omega$ into "indistinguishable cells" — outcomes that look identical given the information at time $t$. Monotonicity says these partitions get *finer* as $t$ grows; cells split, never merge.
* **As an observer's diary.** Imagine an observer who writes down what they have seen so far. $\mathcal{F}_t$ is the σ-algebra generated by their diary entries up to time $t$. The diary is append-only — that is the monotonicity.

### Visualisation 1: filtration as partition refinement (3 fair coin flips)

The "partition refinement" picture is cleanest on a finite sample space. Take $\Omega = \{H, T\}^3$, the space of outcomes of three fair coin flips, with $\mathcal{A} = 2^\Omega$ as the ambient σ-algebra. The natural filtration of the flip sequence $\xi_1, \xi_2, \xi_3$ is

$$
\mathcal{F}_n \;:=\; \sigma(\xi_1, \dots, \xi_n) \qquad (n = 0, 1, 2, 3),
$$

with the convention $\mathcal{F}_0 = \{\emptyset, \Omega\}$ (no flips observed yet). At each $n$, $\mathcal{F}_n$ is generated by the partition of $\Omega$ into "outcomes that agree on the first $n$ flips":

<figure>
  <img src="{{ '/assets/images/notes/random/filtration_tree_coin_flips.png' | relative_url }}" alt="Four-column tree showing the natural filtration of 3 fair coin flips: at n=0 the entire sample space Ω={H,T}^3 is one cell; at n=1 it splits into 2 cells (H··· and T···); at n=2 it splits into 4 cells (HH···, HT···, TH···, TT···); at n=3 it splits into 8 singleton cells. Grey lines show parent-child refinement edges." loading="lazy">
</figure>

Reading the picture:

* **$n = 0$**: nothing has been observed; $\mathcal{F}_0$ is *trivial*, with one partition cell $\Omega$. The only $\mathcal{F}_0$-measurable random variables are constants — you cannot distinguish any two outcomes yet.
* **$n = 1$**: the first flip is observed; $\mathcal{F}_1$ has two cells, $H\cdots$ and $T\cdots$. A random variable is $\mathcal{F}_1$-measurable iff it is constant on each of these two cells (= a function of $\xi_1$ alone).
* **$n = 2$**: the first two flips are observed; $\mathcal{F}_2$ has four cells.
* **$n = 3$**: every flip is observed; $\mathcal{F}_3$ has eight *singleton* cells, $\mathcal{F}_3 = \mathcal{A} = 2^\Omega$. Now *every* random variable on $\Omega$ is $\mathcal{F}_3$-measurable; perfect information.

The grey lines between consecutive columns are the *refinement edges*: each cell at level $n$ splits into exactly two children at level $n+1$. This is precisely the content of $\mathcal{F}\_n \subseteq \mathcal{F}\_{n+1}$ at the level of partitions: the new σ-algebra is obtained by *splitting cells*, never by merging.

### Concrete examples

* **Natural filtration of a process $X$.** Given any process $X = (X_t)_{t \in I}$ on $(\Omega, \mathcal{A}, \mathbb{P})$, the *natural filtration* of $X$ is
  
  $$
  \mathcal{F}_t^X \;:=\; \sigma\!\left(X_s \,:\, s \leq t,\; s \in I\right)
  $$
  
  — the σ-algebra generated by all events of the form "$X_s \in B$" for $s \leq t$, $B$ Borel. This is the filtration of "what an observer who only sees $X$ knows by time $t$." Every process is automatically adapted to its own natural filtration.
* **Brownian filtration.** When $X = W$ is Brownian motion, $\mathbb{F}^W := (\mathcal{F}_t^W)$ is *the* canonical filtration on which BM and the Itô integral are constructed. Most textbook SDEs are formulated on $\mathbb{F}^W$ unless the model needs more.
* **Filtration with a random initial condition.** If you want $X_0 = \xi$ to be a *random* starting point (independent of $W$), you take
  
  $$
  \mathcal{F}_t \;:=\; \sigma\!\left(\xi,\; W_s : s \leq t\right),
  $$
  
  i.e. the natural filtration of $W$ *enlarged by* $\sigma(\xi)$. Then $X_0$ is $\mathcal{F}_0$-measurable — which the natural filtration of $W$ alone could not give, since $\mathcal{F}_0^W = \{\emptyset, \Omega\}$ knows nothing.
* **Joint filtration of two processes.** For SDE systems driven by two independent Brownian motions $W^1, W^2$, one works on
  
  $$
  \mathcal{F}_t \;:=\; \sigma\!\left(W^1_s, W^2_s : s \leq t\right).
  $$
  
  Both BMs remain $(\mathcal{F}_t)$-Brownian motions on this filtration (their increments are still independent of $\mathcal{F}_t$ because of the independence of $W^1, W^2$).
* **Discrete filtration of coin flips.** $\Omega = \{H, T\}^\infty$ with $\mathcal{F}_n := \sigma(\xi_1, \dots, \xi_n)$, where $\xi_k$ is the outcome of the $k$-th flip. $\mathcal{F}_n$ knows the first $n$ flips and nothing else (the picture above is the first three levels of this).
* **The trivial extreme: $\mathcal{F}_t = \{\emptyset, \Omega\}$ for all $t$.** No information ever; nothing is observable. Every random variable except constants fails to be adapted.
* **The other trivial extreme: $\mathcal{F}_t = \mathcal{A}$ for all $t$.** Total clairvoyance; every random variable is adapted at every time. The filtration "knows the whole future at $t = 0$" — stochastic calculus collapses (adaptedness becomes vacuous).

The interesting filtrations live strictly between these two extremes.

### Visualisation 2: filtration as cell structure on a path space

For a process taking finitely many values, the picture is even more direct. Below we plot all $2^3 = 8$ sample paths of a 3-step ±1 random walk $X_n = \xi_1 + \dots + \xi_n$ on the same axes, four times — once for each $n \in \{0, 1, 2, 3\}$. In each panel, the paths are *coloured by which cell of $\mathcal{F}_n$ they belong to* (= which prefix $\xi_1, \dots, \xi_n$ they share):

<figure>
  <img src="{{ '/assets/images/notes/random/filtration_random_walk_paths.png' | relative_url }}" alt="Four side-by-side panels showing the same eight ±1 random-walk paths, coloured by partition cell of F_n at n=0, 1, 2, 3. At n=0 all 8 paths share one colour (one cell); at n=1 the paths split into 2 colour groups (one cell per first-step direction); at n=2 into 4 groups; at n=3 each path is its own colour (8 singleton cells)." loading="lazy">
</figure>

Reading the panels:

* **$n = 0$ (1 cell).** No information yet; all 8 paths are visually indistinguishable to $\mathcal{F}_0$ — same colour.
* **$n = 1$ (2 cells).** Knowing $\xi_1$ splits the 8 paths into two groups of 4 (those starting up vs those starting down). Two colours appear.
* **$n = 2$ (4 cells).** Knowing $(\xi_1, \xi_2)$ splits the paths further into 4 groups of 2.
* **$n = 3$ (8 cells).** Knowing the entire trajectory $(\xi_1, \xi_2, \xi_3)$ resolves each path uniquely. Every path is alone in its own cell — perfect information for the natural filtration.

The dashed vertical line in each panel is the time $n$ at which the σ-algebra $\mathcal{F}_n$ is being read off. *Before* the line, paths in the same cell are *forced to coincide* (because they share the prefix). *After* the line, paths in the same cell may diverge — the filtration $\mathcal{F}_n$ has *no information* about what happens after $n$. That is exactly the visual content of "$\mathcal{F}_n$ is the σ-algebra of what is decided by time $n$".

The same picture applies in continuous time with $W$ in place of $X_n$: cells of $\mathcal{F}_t^W$ are paths that *agree on $[0, t]$* (and may differ arbitrarily after). Since BM paths almost never agree exactly, every cell is a singleton with probability one — but the conceptual picture is identical.

### What "adapted" and "previsible" mean given a filtration

Once you have $\mathbb{F}$, you can classify processes by *how* they relate to it.

* **Adapted.** A process $X = (X_t)\_{t \in I}$ is **$\mathbb{F}$-adapted** if $X_t$ is $\mathcal{F}\_t$-measurable for every $t \in I$. "$X_t$ is decided by time-$t$ information." This is the minimum requirement to even talk about $X$ in the filtered model.
* **Previsible (predictable).** Loosely: $X_t$ is $\mathcal{F}\_{t-}$-measurable, i.e. determined by *strictly earlier* information. In discrete time: $X_n$ is $\mathcal{F}\_{n-1}$-measurable. This is what the "non-anticipating" condition for Itô integrands really means.
* **Stopping time.** A random time $\tau: \Omega \to I \cup \{\infty\}$ such that $\{\tau \leq t\} \in \mathcal{F}\_t$ for every $t$ — "by time $t$ you know whether $\tau$ has fired." Defined entirely in terms of $\mathbb{F}$.

All three of these notions are *meaningless without a filtration*; they are statements about the relationship between a process and a flow of information.

### The "usual conditions" (a small but important refinement)

In continuous-time stochastic calculus, textbooks often demand that $\mathbb{F}$ satisfies the **usual conditions**:

* **Completeness.** Every subset of a $\mathbb{P}$-null set in $\mathcal{A}$ already lies in $\mathcal{F}\_0$. (Null sets are "decided" from the start.)
* **Right-continuity.** $\mathcal{F}\_t = \mathcal{F}\_{t+} := \bigcap_{s > t} \mathcal{F}\_s$ for every $t$. Information at $t$ equals "information observable just after $t$" — no hidden surprises that require an infinitesimal peek into the future.

These two technical conditions are what make stopping-time theory, optional projection, and the construction of càdlàg modifications of martingales work cleanly. Most "natural" filtrations don't satisfy them on the nose; one usually replaces $\mathbb{F}^W$ by its *augmented* / *right-continuous* version, sometimes denoted $\mathbb{F}^W_+$ or written without notation as a tacit convention.

### How this connects to the upcoming sections

The notion of filtration is the *exact* object underneath the concepts that follow.

* **"$X$ is a martingale w.r.t. $\mathbb{F}$"** = $\mathbb{F}$-adapted, integrable, with $\mathbb{E}[X_t \mid \mathcal{F}_s] = X_s$ for $s \leq t$.
* **"$W$ is a Brownian motion w.r.t. $(\mathcal{F}_t)$"** = the joint statement (i) $W_t$ is $\mathcal{F}_t$-measurable for every $t$ (= $W$ is $\mathbb{F}$-adapted) and (ii) future increments are $\mathcal{F}_t$-independent.
* **"$W_t$ is $\mathcal{F}_t$-measurable"** = adaptedness at the single time $t$, where $\mathcal{F}_t$ is the time-$t$ slice of $\mathbb{F}$.
* **"$(W_{t+s} - W_t)_{s}$ is independent of $\mathcal{F}_t$"** = increment-independence relative to the time-$t$ slice.

In every one of these, the **filtration is the missing first ingredient** — the object that gives meaning to "by time $t$" or "given the past". Defining it explicitly is the price of admission to stochastic calculus.

### One-sentence summary

A **filtration** $\mathbb{F} = (\mathcal{F}\_t)\_{t \in I}$ in $\mathcal{A}$ is an increasing family of sub-σ-algebras of the ambient σ-algebra $\mathcal{A}$ — one for each "time" $t \in I$ — that formalises the *append-only flow of information* through time, equivalent to a *partition of $\Omega$ that gets finer as $t$ grows*; this single object is the missing first ingredient that gives meaning to "what is known by time $t$" in every subsequent definition (adapted, martingale, stopping time, BM-with-respect-to-$\mathbb{F}$, Itô integral, …).

## Martingale property

A **martingale** is the mathematical formalisation of a *fair game* — a stochastic process whose future expected value, conditional on everything you know now, equals its present value. Nothing about the past or present gives you a statistical edge in predicting the next move. This single property is the backbone of stochastic calculus: it is what makes Itô integrals "noise" (mean zero), what makes the $\mathrm{d} W$-term vanish in expectation in Itô's formula, and what makes the operator $\mathscr{L}$ in the previous section the *unique* drift-killer for $f(X_t)$.

### Setup: filtration and adaptedness

You cannot define "what you know at time $t$" without naming the history of information up to $t$. That object is a **filtration** $(\mathcal{F}\_t)\_{t \geq 0}$: an increasing family of $\sigma$-algebras with

$$
\mathcal{F}_s \;\subseteq\; \mathcal{F}_t \qquad \text{whenever } s \leq t.
$$

Concretely, $\mathcal{F}\_t$ is the collection of events whose occurrence has been *decided* by time $t$ — every coin flip, every Brownian increment, every random thing that has already happened up to that instant.

A process $(X_t)$ is **adapted** to $(\mathcal{F}_t)$ if $X_t$ is $\mathcal{F}_t$-measurable for every $t$, i.e. observing the history up to time $t$ is enough to know the value of $X_t$. Without adaptedness the equation defining the martingale property does not even make sense.

### The definition

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Martingale)</span></p>

A real-valued process $(X_t)_{t \geq 0}$ adapted to a filtration $(\mathcal{F}_t)$ is a **martingale** (with respect to that filtration) if

1. **Integrability:** $\mathbb{E}[\lvert X_t \rvert] < \infty$ for every $t$.
2. **Adaptedness:** $X_t$ is $\mathcal{F}_t$-measurable for every $t$.
3. **Martingale property:** for every $s \leq t$,
   
   $$
   \mathbb{E}\!\left[X_t \mid \mathcal{F}_s\right] \;=\; X_s \quad \text{a.s.}
   $$

</div>

That single equation is *the* martingale property. Read it as: "given everything you know up to time $s$, the best forecast of $X_t$ for any later $t$ is just the current value $X_s$." There is no expected drift up or down — the process is, on average, going nowhere.

The discrete-time version is identical, with $n$ in place of $t$:  $\mathbb{E}[X_{n+1} \mid \mathcal{F}_n] = X_n$.

### Three siblings: martingale, sub-, super-

Same setup; replace the equality with an inequality and you get the two close relatives:

| Type | Defining condition | Interpretation |
|---|---|---|
| **Martingale** | $\mathbb{E}[X_t \mid \mathcal{F}_s] = X_s$ | fair game |
| **Submartingale** | $\mathbb{E}[X_t \mid \mathcal{F}_s] \geq X_s$ | favourable game (drifts upward in expectation) |
| **Supermartingale** | $\mathbb{E}[X_t \mid \mathcal{F}_s] \leq X_s$ | unfavourable game (drifts downward in expectation) |

The mnemonic that catches everyone the wrong way: a *super*martingale is the one that goes *down* in expectation. The "super" is from the casino's point of view, not the gambler's.

The picture below makes the three regimes visible by drawing many sample paths of a discrete random walk $X_n = \xi_1 + \dots + \xi_n$ with i.i.d. increments of mean $\mu \in \{+0.05, 0, -0.05\}$. The black curve in each panel is the empirical mean across $K = 5000$ paths; the orange dashed line is the theoretical mean $\mu \cdot n$.

<figure>
  <img src="{{ '/assets/images/notes/random/martingale_sub_super_comparison.png' | relative_url }}" alt="Three side-by-side panels showing 40 sample paths and the empirical mean (black) of random walks with i.i.d. increments of mean +0.05 (submartingale, drifting up), 0 (martingale, mean stays at 0), and -0.05 (supermartingale, drifting down). Orange dashed reference line gives the theoretical mean μ·n." loading="lazy">
</figure>

The middle panel is the diagnostic for "martingale": no matter how wild the individual paths look, *the average of many of them stays flat at $0$ for all $n$*. The left and right panels show what *non*-martingales look like — a clear linear drift in the empirical mean.

### What follows immediately

Take $s = 0$ in the defining identity and apply the tower property:

$$
\mathbb{E}[X_t] \;=\; \mathbb{E}\!\big[\mathbb{E}[X_t \mid \mathcal{F}_0]\big] \;=\; \mathbb{E}[X_0] \quad \text{for all } t.
$$

So **every martingale has constant expectation** — the unconditional mean is preserved through time. This is the most-used corollary in practice; it is what makes statements like $\mathbb{E}\left[\int_0^T f \, \mathrm{d}W_t\right] = 0$ collapse to a one-line proof.

A second consequence: for $s \leq t$, the *increment* $X_t - X_s$ is uncorrelated with anything in $\mathcal{F}_s$. Every $\mathcal{F}_s$-measurable bounded $Z$ satisfies

$$
\mathbb{E}\!\left[Z \, (X_t - X_s)\right] \;=\; 0,
$$

i.e. martingale increments are **orthogonal to the past** in $L^2$. This is the ingredient that powers Itô isometry.

The picture below visualises the "constant expectation" corollary on Brownian motion. Sixty light purple paths show the wild path-by-path behaviour; the red curve is the empirical mean across $K = 10000$ paths, and the dashed black line is the theoretical $\mathbb{E}[W_t] = 0$:

<figure>
  <img src="{{ '/assets/images/notes/random/martingale_brownian_mean.png' | relative_url }}" alt="60 sample paths of standard Brownian motion plotted lightly in purple over [0,1], with the empirical mean across 10000 paths shown in red, hugging the dashed black line E[W_t]=0 for all t — confirming that BM is a martingale." loading="lazy">
</figure>

The individual paths spread out (their *variance* grows linearly with $t$, since $\mathrm{Var}(W_t) = t$), but their *mean* is locked to $0$ for all $t$. That is the visible signature of the martingale property.

### Canonical examples

* **Brownian motion $W_t$.** With its own filtration, $\mathbb{E}[W_t \mid \mathcal{F}_s] = W_s + \mathbb{E}[W_t - W_s] = W_s$ by independent increments and zero mean. So $W$ is the prototypical continuous-time martingale.
* **$W_t^2 - t$.** Pure $W_t^2$ is *not* a martingale (its expectation grows as $t$, so it can't have constant mean). But $W_t^2 - t$ *is*: the deterministic $-t$ exactly cancels the variance growth. This is the simplest visible **Itô correction** at the level of martingales.
* **The Doléans exponential** $\mathcal{E}(W)_t := \exp\!\left(W_t - \tfrac{t}{2}\right)$. The defining example of a positive martingale; foundational for Girsanov / change of measure.
* **Itô integrals.** For $f \in L^2$ adapted, $M_t := \int_0^t f_s \, \mathrm{d}W_s$ is a martingale. This is the operator-level reason why the $\mathrm{d}W$-term *vanishes in expectation* in Itô's formula — exactly the step we used in the previous section to derive the generator.
* **Random walks.** $S_n = X_1 + \dots + X_n$ with i.i.d. mean-zero increments is a discrete-time martingale (positive mean → submartingale, negative mean → supermartingale).
* **Likelihood ratios.** If $P, Q$ are equivalent measures and $L_n := \frac{\mathrm{d}Q}{\mathrm{d}P}\big\|_{\mathcal{F}\_n}$, then $L_n$ is a $P$-martingale. This is the abstract engine behind Bayesian updating and Girsanov.

### The simplest visible Itô correction: $W_t^2$ vs $W_t^2 - t$

The bullet "$W_t^2 - t$ is a martingale" deserves a picture. Both panels below use the *same* $K = 20000$ Brownian sample paths.

<figure>
  <img src="{{ '/assets/images/notes/random/martingale_ito_correction.png' | relative_url }}" alt="Two-panel figure. Left: 50 sample paths of W_t^2 in purple, with the empirical mean across 20000 paths in red tracking the dashed line E[W_t^2]=t — clear linear drift, so W_t^2 is NOT a martingale. Right: 50 sample paths of W_t^2 - t in green with the empirical mean in red hugging the dashed line at 0 — the -t exactly kills the drift, so W_t^2 - t IS a martingale." loading="lazy">
</figure>

* **Left:** $W_t^2$. The empirical mean (red) tracks the theoretical curve $\mathbb{E}[W_t^2] = t$ (dashed black) almost exactly. Constant mean is *violated*: $\mathbb{E}[W_t^2]$ grows linearly in $t$. So $W_t^2$ is *not* a martingale — it is a sub-martingale.
* **Right:** $W_t^2 - t$. The empirical mean is now indistinguishable from zero. The deterministic compensator $-t$ has *exactly* removed the variance-driven drift, restoring the martingale property.

This is the conceptual content of the Itô correction in its cleanest form: the second-order operator $\frac{1}{2} f'' (W_t)$ in Itô's formula contributes a deterministic drift to $f(W_t)$, and subtracting its time-integral is what produces a martingale. For $f(x) = x^2$, $\frac{1}{2} f'' = 1$, so the compensator is $\int_0^t 1 \, \mathrm{d}s = t$ — exactly the term we subtracted.

### The "filtration matters" subtlety

The martingale property is *always* relative to a specific filtration. The same process $X$ can be a martingale w.r.t. one filtration $(\mathcal{F}_t)$ and *not* a martingale w.r.t. a larger one $(\mathcal{G}_t)$ that knows more — for instance one that already "peeks ahead" at later randomness. Whenever you write "$X$ is a martingale" without naming a filtration, the convention is the **natural filtration**

$$
\mathcal{F}_t^X \;:=\; \sigma\!\left(X_s \,:\, s \leq t\right),
$$

i.e. exactly the information generated by observing $X$ itself up to time $t$.

### Two big theorems that make martingales powerful

* **Optional stopping theorem.** For a martingale $X$ and a "nice" stopping time $\tau$ (e.g. bounded, or such that $X^\tau$ is uniformly integrable),
  
  $$
  \mathbb{E}[X_\tau] \;=\; \mathbb{E}[X_0].
  $$
  
  This is what lets you compute exit probabilities, gambler's-ruin probabilities, and expected hitting times — all standard barrier problems collapse to one line of algebra once an appropriate martingale is in hand.
* **Martingale convergence theorem (Doob).** Any $L^1$-bounded martingale converges almost surely to a finite limit $X_\infty$. Submartingales bounded above and supermartingales bounded below also converge a.s. This is the clean replacement for "monotone bounded sequences converge" in the random setting, and it is the structural reason behind a vast range of "averages stabilise" results in probability.

### Connection to the generator (the section just above)

This is the bridge to what you were just reading. The **martingale problem** of Stroock–Varadhan says: a process $(X_t)$ *is* the diffusion with generator $\mathscr{L}$ if and only if, for every $f \in C_c^\infty(\mathbb{R}^d)$,

$$
M_t^f \;:=\; f(X_t) \;-\; f(X_0) \;-\; \int_0^t (\mathscr{L} f)(X_s) \, \mathrm{d}s
$$

is a martingale with respect to the natural filtration of $X$.

Read this carefully. The first part $f(X_t) - f(X_0)$ is the change in $f$ along the path. The integral is the *expected* change you should subtract to make a fair game out of the result. So **the generator $\mathscr{L}$ is precisely the operator whose subtraction kills the drift in $f(X_t)$**, and the martingale property is the meta-axiom that defines what "kills the drift" means.

Two angles on this same equivalence:

* From SDE → generator: applying Itô's formula to $f(X_t)$ produces a $\mathrm{d}t$-integral with integrand $\mathscr{L} f(X_s)$ plus a $\mathrm{d}W$-integral; the latter is a martingale, so subtracting the former produces $M_t^f$.
* From generator → process: *defining* the process by demanding $M_t^f$ be a martingale for every test $f$ characterises the law of $X$ uniquely (under regularity). This is the flexible alternative to specifying an SDE — useful exactly when you do not have a Brownian filtration up front.

### One-sentence summary

A **martingale** is an integrable, adapted process whose conditional expectation $\mathbb{E}[X_t \mid \mathcal{F}_s] = X_s$ encodes a *fair game* — the future is, in expectation, the present — and this property is the abstract device through which stochastic calculus turns randomness into something quantifiable: it makes Itô integrals mean-zero, makes Itô's formula's $\mathrm{d}W$-term disappear in expectation, and (via Stroock–Varadhan) characterises a diffusion uniquely by which $f(X_t) - \int_0^t \mathscr{L} f(X_s) \, \mathrm{d}s$ are martingales.

## Brownian motion with respect to a filtration: adaptedness, independence, and change of measure

When SDE textbooks write phrases like "$W$ is a Brownian motion *with respect to* the filtration $(\mathcal{F}\_t)$", or "$W_t$ is $\mathcal{F}\_t$-measurable", or "the increments $W_{t+s} - W_t$ are independent of $\mathcal{F}\_t$", what looks like extra measure-theoretic decoration is actually the *real* definition of Brownian motion — the part that makes Itô integration work. This section unpacks each piece, then connects them to the genuinely different notion of "Brownian motion *with respect to a probability measure $\mathbb{P}$*" (Girsanov).

### Three different "with respect to"s

It is worth separating three objects that get conflated in casual reading:

| Object | Type | Role | "BM is …" with respect to it |
|---|---|---|---|
| $\mathbb{P}$ | a probability *measure* on $(\Omega, \mathcal{F})$ | "how likely is each event" | $W$ has a particular law; under another measure $\mathbb{Q}$ the same path is no longer BM (Girsanov) |
| $\mathcal{F}$ | a *σ-algebra* on $\Omega$ | "what events are observable / decidable" | $W_t$ is $\mathcal{F}$-measurable iff $\mathcal{F}$ resolves its value |
| $(\mathcal{F}\_t)\_{t \geq 0}$ | a *filtration* (increasing family of σ-algebras) | "what is observable *by time $t$*" | $W$ is an "$(\mathcal{F}_t)$-Brownian motion" iff it is adapted *and* its future increments are independent of $\mathcal{F}_t$ |

In what follows, "BM w.r.t. $\mathcal{F}_t$" means the *filtration* sense (the standard one in stochastic calculus); the *measure* sense is treated separately at the end.

### Brownian motion with respect to a filtration $(\mathcal{F}_t)$

The textbook definition of BM you may have seen specifies only the *law*: $W_0 = 0$, independent stationary Gaussian increments $W_t - W_s \sim \mathcal{N}(0, t-s)$, continuous paths a.s. But once you do stochastic calculus you also need to say *with respect to which information set* the process is unfolding. The richer notion is:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($(\mathcal{F}_t)$-Brownian motion)</span></p>

A process $W = (W_t)_{t \geq 0}$ on $(\Omega, \mathcal{F}, \mathbb{P})$ is a **Brownian motion with respect to the filtration $(\mathcal{F}_t)$** (an "$(\mathcal{F}_t)$-Brownian motion") if

1. $W_0 = 0$ a.s.;
2. $W$ is **adapted** to $(\mathcal{F}_t)$, i.e. $W_t$ is $\mathcal{F}_t$-measurable for every $t$;
3. $W$ has continuous paths a.s.;
4. for every $s, t \geq 0$, the increment $W_{t+s} - W_t$ is **independent of $\mathcal{F}_t$** and $\sim \mathcal{N}(0, s)$.

</div>

So "$W$ is a BM w.r.t. $(\mathcal{F}_t)$" is a *joint* statement about $W$ and the filtration. Two things must hold simultaneously:

* The filtration is **rich enough** to "see" $W$ — that is condition 2 (adaptedness).
* The filtration is **not too rich** — it must not contain any information about future increments of $W$. That is condition 4 (independence).

The natural choice is the *natural filtration* $\mathcal{F}_t^W := \sigma(W_s : s \leq t)$, with which any BM is automatically a $(\mathcal{F}_t^W)$-BM. But many models need a *bigger* filtration (e.g. one carrying additional independent randomness, or independent jump processes), and then both conditions become real constraints.

### What "$W_t$ is $\mathcal{F}$-measurable" means

For a real-valued random variable $X: \Omega \to \mathbb{R}$ on $(\Omega, \mathcal{F}_0, \mathbb{P})$, *$X$ is $\mathcal{F}$-measurable* (for some sub-σ-algebra $\mathcal{F} \subseteq \mathcal{F}_0$) means

$$
X^{-1}(B) \;\in\; \mathcal{F} \qquad \text{for every Borel set } B \subseteq \mathbb{R}.
$$

Every event of the form "$X$ falls in $B$" is one of the events that $\mathcal{F}$ already knows about. Equivalently, by the Doob–Dynkin lemma, there exists a deterministic Borel function $g$ such that $X = g(\text{the part of } \omega \text{ that } \mathcal{F} \text{ resolves})$ — so *$X$ is a deterministic function of the information in $\mathcal{F}$*. Three increasingly intuitive readings:

* **Operational:** "If you know which events in $\mathcal{F}$ occurred, you know the value of $X$."
* **Coarse-graining:** $\mathcal{F}$ is (up to refinement) a partition of $\Omega$; $X$ is $\mathcal{F}$-measurable iff $X$ is constant on each cell.
* **Information-theoretic:** $X$ contributes no information beyond what $\mathcal{F}$ already has, $\sigma(X) \subseteq \mathcal{F}$.

For BM, "$W_t$ is $\mathcal{F}_t$-measurable" specialises to: by time $t$ the value $W_t$ has been *observed* (or is at least determined by what has been observed). The picture below makes this tangible by progressively "unveiling" a sample path: at each time $t$, the *past* segment is solid (decided, $\mathcal{F}_t$-measurable) while the *future* segment is dotted in grey (not yet $\mathcal{F}_t$-measurable):

<figure>
  <img src="{{ '/assets/images/notes/random/filtration_adaptedness.png' | relative_url }}" alt="Four-panel figure showing the same Brownian sample path at four growing times t=0.15, 0.40, 0.65, 0.90. In each panel, the past portion of the path is drawn solid in purple (F_t-measurable: decided), the future portion is dotted in grey (not yet F_t-measurable), and the value W_t at the present moment is highlighted with a red dot — illustrating how the filtration progressively decides more and more of the path." loading="lazy">
</figure>

The σ-algebra $\mathcal{F}_t$ grows monotonically as $t$ grows; what was random at time $t_1$ becomes decided by time $t_2 > t_1$. If $\mathcal{F}_t$ is the natural filtration $\mathcal{F}_t^W$, adaptedness is automatic; if $\mathcal{F}_t$ is *bigger* than $\mathcal{F}_t^W$, adaptedness is *still* automatic — bigger σ-algebras only make it easier to be measurable. The non-trivial requirement on bigger filtrations is the next one.

### What "$(W_{t+s} - W_t)_{s \in I}$ is independent of $\mathcal{F}$" means

Two random objects are *independent* in the measure-theoretic sense if the σ-algebras they generate are independent. For the increment process

$$
\Delta W^{(t)} \;:=\; (W_{t+s} - W_t)_{s \in I},
$$

independence from a σ-algebra $\mathcal{F}$ means

$$
\mathbb{P}\!\left(\{\Delta W^{(t)} \in A\} \cap B\right) \;=\; \mathbb{P}\!\left(\Delta W^{(t)} \in A\right) \cdot \mathbb{P}(B)
\qquad \text{for all } A,\, B \in \mathcal{F},
$$

equivalently $\sigma(\Delta W^{(t)}) \perp \mathcal{F}$. Three increasingly geometric readings:

* **No information leakage.** Knowing that some event in $\mathcal{F}$ happened does *not* shift the conditional law of any future increment $W_{t+s} - W_t$. In symbols, $\mathbb{P}(\Delta W^{(t)} \in A \mid \mathcal{F}) = \mathbb{P}(\Delta W^{(t)} \in A)$ a.s.
* **Conditional expectations factor.** For any bounded measurable $\varphi$ of the future increments,
  
  $$
  \mathbb{E}\!\left[\varphi(\Delta W^{(t)}) \,\big|\, \mathcal{F}\right] \;=\; \mathbb{E}\!\left[\varphi(\Delta W^{(t)})\right] \quad \text{a.s.}
  $$
  
  Conditioning on $\mathcal{F}$ does nothing — the future increments are statistically detached from $\mathcal{F}$.
* **Strong Markov in distribution.** When $\mathcal{F} = \mathcal{F}_t$, this is the *Markov property in its strongest σ-algebraic form*: the future of $W$ after time $t$ is independent of the *entire past*, not just of the present location $W_t$.

The picture below visualises this on $K = 30000$ Brownian sample paths. We pick $t^* = 0.5$ and partition the paths into three groups by the *quartile* of $W_{t^\ast}$ they fell into (a coarse summary of their past behaviour). Then we plot the empirical density of the *future* increment $W_{t^\ast + s} - W_{t^\ast}$ ($s = 0.3$) for each group separately, all on the same axes:

<figure>
  <img src="{{ '/assets/images/notes/random/filtration_increment_independence.png' | relative_url }}" alt="Two-panel figure. Left: past Brownian paths up to t*=0.5, coloured by which quartile W_{t*} fell into (bottom 25% in blue, middle 50% in green, top 25% in red). Right: empirical densities of the future increment W_{t*+s} - W_{t*} (s=0.3) for each of the three past-quartile groups, all overlapping the dashed black theoretical N(0, 0.3) density — visually demonstrating that the conditional law of the future increment is the same regardless of past behaviour." loading="lazy">
</figure>

The three coloured histograms are *visually indistinguishable* and all match the theoretical $\mathcal{N}(0, s)$ density (dashed black). That is the empirical picture of independence: no matter how the past went, the future increment is the same Gaussian. By contrast, *position-dependence* would show as the three histograms being shifted relative to each other — which is exactly what would happen for a non-BM process whose increments depend on the past.

A subtle but important point: independence of $\Delta W^{(t)}$ from $\mathcal{F}_t$ is **strictly stronger** than independence from just $W_t$, or even from the natural past $\mathcal{F}_t^W$. If $\mathcal{F}_t$ is enlarged with extra randomness (e.g. a coin flip $C$ that is independent of $W$), independence from $\mathcal{F}_t = \sigma(\mathcal{F}_t^W, C)$ is the additional requirement that $C$ does not encode anything about future BM increments — automatic if $C$ was constructed independently of $W$, but not free in general.

### Why all three together — the "$(\mathcal{F}_t)$-BM" package

Adaptedness and increment-independence are the two halves of "$W$ unfolds in time *along with* the filtration $(\mathcal{F}_t)$":

* Adaptedness says **the past of $W$ is contained in the filtration**:  $\sigma(W_s : s \leq t) \subseteq \mathcal{F}_t$.
* Increment-independence says **the future increments of $W$ are independent of the filtration**:  $\sigma(W_{t+s} - W_t : s \geq 0) \perp \mathcal{F}_t$.

These two together are exactly what the constructions of stochastic calculus need:

* In the **Itô integral** $\int_0^t H_s \, \mathrm{d}W_s$, the integrand $H_s$ is required to be $\mathcal{F}\_s$-measurable (*non-anticipating*). This makes the increment $W_{s + \mathrm{d}s} - W_s$ — independent of $\mathcal{F}\_s$ — independent of $H_s$ as well, and that independence is what powers the Itô isometry $\mathbb{E}\big[(\int H \, \mathrm{d}W)^2\big] = \mathbb{E}\!\int H^2 \, \mathrm{d}s$ and the martingale property of the integral (recall the previous section).
* In **SDE solutions** $\mathrm{d} X_t = b(X_t) \, \mathrm{d}t + \sigma(X_t) \, \mathrm{d}W_t$, the noise driving $X$ must be independent of the part of the filtration the solution may use; without increment-independence one cannot even define the right-hand side cleanly.

Real models routinely *need* a filtration strictly larger than $\mathcal{F}_t^W$:

* **Coupled SDE systems** $\mathrm{d}X = \dots, \mathrm{d}Y = \dots$ where each equation is driven by its own BM but whose coefficients use both $X_t, Y_t$.
* **Filtering**: the filtration carries both the signal $X$ and the noisy observation $Y$.
* **Jump-diffusions**: the filtration carries both a BM and an independent Poisson process.
* **Random initial conditions**: $X_0$ is a $\mathcal{F}_0$-measurable random variable independent of $W$.

In every one of these, you genuinely need a filtration *strictly larger* than $\mathcal{F}_t^W$, and the requirement that $W$ remains a Brownian motion *with respect to that bigger filtration* is the non-trivial part — exactly the content of the increment-independence condition.

### And the genuinely measure-theoretic reading: BM with respect to *measure* $\mathbb{P}$

Up to here, "with respect to" referred to a *filtration*. The phrase "BM with respect to a probability measure $\mathbb{P}$" is a *different* concept and shows up in **Girsanov's theorem**. Two equivalent measures $\mathbb{P}, \mathbb{Q}$ on the same $(\Omega, \mathcal{F})$ can disagree on whether a given process is a BM:

* Under $\mathbb{P}$, the process $W_t$ is a standard BM (mean $0$, increments $\mathcal{N}(0, \mathrm{d}t)$).
* Define an equivalent measure $\mathbb{Q}$ by the **Radon–Nikodym density**
  
  $$
  \frac{\mathrm{d}\mathbb{Q}}{\mathrm{d}\mathbb{P}} \;=\; L_T \;:=\; \exp\!\left(\theta W_T - \tfrac{1}{2} \theta^2 T\right) \qquad \text{(Cameron–Martin–Girsanov density)}.
  $$
  
* Under $\mathbb{Q}$, the *new* process $\widetilde{W}_t := W_t - \theta t$ is a BM, while the *original* $W_t$ has acquired a drift $\theta t$ and is *no longer Brownian*.

So the statement "$W$ is a BM" is not absolute — it depends on which measure you compute expectations against. The picture below illustrates this with $K = 8000$ sample paths and $\theta = 1.5$. **Left:** the same $\omega$-paths plotted with line opacity proportional to their importance weight $L_T(\omega)$; the empirical $\mathbb{P}$-mean (green) hugs $0$ as expected, while the *importance-weighted* empirical $\mathbb{Q}$-mean (red) tracks the theoretical drift line $\theta t$ (black dashed). **Right:** empirical density of $W_T$ under $\mathbb{P}$ (centred at $0$) and under $\mathbb{Q}$ (centred at $\theta T$), both matching their theoretical Gaussians:

<figure>
  <img src="{{ '/assets/images/notes/random/filtration_girsanov.png' | relative_url }}" alt="Two-panel Girsanov visualisation with theta=1.5. Left: 100 Brownian sample paths from 8000 simulated, with line opacity proportional to the importance weight L_T = exp(θ W_T - θ²T/2); the empirical P-mean (green) sits at 0, the L_T-weighted empirical Q-mean (red) tracks the theoretical drift line θt (black dashed). Right: empirical density of W_T under P (green, centred at 0) and the L_T-weighted Q-density (red, centred at θT=1.5), both overlapping their theoretical N(0,T) and N(θT,T) densities respectively." loading="lazy">
</figure>

The intuition: under $\mathbb{Q}$, the paths with *higher* terminal value $W_T$ are *more probable* (because $L_T = e^{\theta W_T - \theta^2 T / 2}$ is increasing in $W_T$). When we resample / reweight by $L_T$, the typical path acquires an *upward bias*, and the operation is exactly equivalent to having added a deterministic drift $\theta t$. This is the abstract engine of:

* **Risk-neutral pricing** in mathematical finance (change to the measure under which discounted assets are martingales).
* **Maximum-likelihood / importance-sampling** changes of measure in statistics.
* **Bayesian posterior updates** in their measure-theoretic form (likelihood ratio = Radon–Nikodym density).

So if you ever read "$W$ is a BM under $\mathbb{P}$" in a context where another measure $\mathbb{Q}$ is in play, the statement is genuinely measure-dependent and not a casual phrasing.

### One-sentence summaries

* **"$W$ w.r.t. filtration $(\mathcal{F}_t)$"** = the joint specification "$W$ is observed by $\mathcal{F}_t$ (adapted) *and* has no information leakage into $\mathcal{F}_t$ from the future (independent increments)" — the standard hypothesis under which Itô calculus actually works.
* **"$W_t$ is $\mathcal{F}$-measurable"** = the value $W_t$ is a deterministic function of the information in $\mathcal{F}$; equivalently, every event $\{W_t \in B\}$ is decided by $\mathcal{F}$.
* **"$(W_{t+s} - W_t)_{s \in I}$ independent of $\mathcal{F}$"** = no event in $\mathcal{F}$ shifts the conditional law of any future Brownian increment; equivalently, $\mathbb{E}[\varphi(\text{future inc.}) \mid \mathcal{F}] = \mathbb{E}[\varphi(\text{future inc.})]$ for all bounded $\varphi$ — the strong "no anticipation" condition.
* **"$W$ w.r.t. measure $\mathbb{P}$"** = a *different* "with respect to": the property of being BM is measure-dependent, and Girsanov shows how multiplying by a Radon–Nikodym density turns BM under $\mathbb{P}$ into BM-with-drift under $\mathbb{Q}$.

## Lebesgue integral

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Lebesgue's Criterion for Riemann Integrability)</span></p>

For any function $f: [a,b] \to \mathbb{R}$ the following equivalence holds 

$$f \in R(a,b) \iff f \text{ is bounded and } DC(f) \text{ has measure zero},$$

where $DC(f) := \lbrace x \in M \mid f \text{ is discontinuous in } x  \rbrace$ for $f: M \to \mathbb{R}$.

</div>

*proof*: #TODO:

## Partition of an interval

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Partition of an interval)</span></p>

Let $a,b \in \mathbb{R}$ with $a < b$. A **partition** of the interval $[a,b]$ is a finite ordered set

$$P = (a_0,a_1,\dots,a_k)$$

such that

$$a = a_0 < a_1 < \cdots < a_k = b, \qquad k \in \mathbb{N}.$$

</div>

## Norm of partition

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Norm of a partition)</span></p>

The **norm** of a partition $P$ is defined by

$$\Delta(P) := \max_{i=1,\dots,k} (a_i - a_{i-1}).$$

</div>

## Choice of tags / sample points

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Choice of tags / sample points)</span></p>

Given a partition $P = (a_0,\dots,a_k)$, a **tag vector**

$$\bar t = (t_1,\dots,t_k)$$

is a choice of points such that

$$t_i \in [a_{i-1},a_i], \qquad i=1,\dots,k.$$

</div>

## Riemann sum

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Riemann sum)</span></p>

Let $f : [a,b] \to \mathbb{R}$ be an arbitrary function. For a partition $P$ of $[a,b]$ and a corresponding tag vector $\bar t$, the **Riemann sum** of $f$ with respect to $P$ and $\bar t$ is

$$R(P,\bar t,f) := \sum_{i=1}^{k} (a_i - a_{i-1}) f(t_i).$$

</div>

## Riemann Integral

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Riemann integral)</span></p>

Let $a<b$ and let $f:[a,b]\to\mathbb{R}$. We say that $f$ is **Riemann integrable** on $[a,b]$, and write $f \in \mathcal{R}(a,b),$ if there exists a real number $L\in\mathbb{R}$ such that $\forall \varepsilon>0\;\exists \delta>0$ s.t. for every partition $P$ of $[a,b]$ and every choice of tags $\bar t$ for $P$,

$$\Delta(P)<\delta \Longrightarrow \bigl|R(P,\bar t,f)-L\bigr|<\varepsilon.$$

In this case, we define

$$\int_a^b f(x)dx := L,$$

and also write

$$(R)\int_a^b f = L.$$

The number $L$ is called the **(Riemann) integral of $f$ over $[a,b]$**.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Unbounded functions are bad)</span></p>

Function $f: [a,b] \to \mathbb{R}$ is unbounded $\implies$ $f \notin \mathcal{R}(a,b)$

</div>

## Zvana 1

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Zvana 1)</span></p>

Let $f: [a,b] \to \mathbb{R}$ be in $\mathcal{R}(a,b)$. Then $f\in \mathcal{R}(a,x)$ for every $x \in (a,b]$ and the function $F: [a,b]\to\mathbb{R}$ is given as

$$F(x):=\int_a^x f$$

is Lipschitz continuous and $F'(x)=f(x)$ in every point  $x\in [a,b]$ of continuity $f$.

</div>

## Zvana 2

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Zvana 2)</span></p>

Let $f:(a,b)\to\mathbb{R}$ has primitive function $F:(a,b)\to\mathbb{R}$ and let $f\in \mathcal{R}(a,b)$. Then there exists finite limit $F(a):=\lim_{x\to a}F(x)$ and $F(b):=\lim_{x\to b}F(x)$ and

$$(R)\int_a^b f = F(b)-F(a) = (N)\int_a^b f$$

</div>

## Lebesgue–Stieltjes integral

## Poisson distribution

### Poisson Process

### Poisson Point Process

## Boltzmann Distribution

### Add Boltzmann Distribution

## Karger's algorithm

### Add Karger's algorithm

## Karger–Stein algorithm

### Add Karger–Stein algorithm

## Randomized Quicksort

### Randomized Quicksort

## Zero-knowledge protocols

### Zero-knowledge protocols

## Integration by substitution

## Measure theory is blind to sets of probability zero

The slogan *"measure theory is blind to null sets"* is precise: integrals, expectations, and probabilities against $\mathbb{P}$ cannot detect what happens on a set $N \in \mathcal{A}$ with $\mathbb{P}(N) = 0$. Below is what this actually means, why it holds, and why it is the reason almost-sure statements are the natural language for stochastic processes.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Principle</span><span class="math-callout__name">(Blindness to null sets)</span></p>

Let $(\Omega, \mathcal{A}, \mathbb{P})$ be a measure space and $N \in \mathcal{A}$ with $\mathbb{P}(N) = 0$. For any integrable random variables $X, Y$ with $X = Y$ on $\Omega \setminus N$,

$$
\mathbb{E}[X] \;=\; \int_\Omega X \, \mathrm{d}\mathbb{P} \;=\; \int_\Omega Y \, \mathrm{d}\mathbb{P} \;=\; \mathbb{E}[Y].
$$

In particular, probabilities are unchanged: $\mathbb{P}(A) = \mathbb{P}(A')$ whenever the symmetric difference $A \triangle A' \subseteq N$.

</div>

### Why this holds

For a non-negative simple function $\varphi = \sum_i c_i \mathbf{1}_{A_i}$,

$$
\int \varphi \, \mathrm{d}\mathbb{P} = \sum_i c_i \, \mathbb{P}(A_i),
$$

and $\mathbb{P}(A_i \cap N) = 0$ for every $i$, so any mass sitting inside $N$ contributes $c_i \cdot 0 = 0$. Extending to general integrable $X$ by monotone / dominated convergence preserves the property: modifying the integrand on a null set can neither change a simple-function approximation nor its limit.

### What "blindness" looks like concretely

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(finite-set modification)</span></p>

Take $f(x) = \sin(2\pi x)$ on $[0,1]$ and define $g$ by

$$
g(x) = \begin{cases} y_i, & x = x_i \in N := \lbrace x_1, \dots, x_4 \rbrace, \\ f(x), & x \notin N, \end{cases}
$$

for *any* values $y_1, \dots, y_4 \in \mathbb{R}$. The set $N$ has Lebesgue measure $0$, so

$$
\int_0^1 g(x) \, \mathrm{d}x \;=\; \int_0^1 f(x) \, \mathrm{d}x \;=\; 0,
$$

no matter how we choose the $y_i$. Visually the two functions disagree at four points, but the integral cannot tell them apart.

</div>

<figure>
  <img src="{{ '/assets/images/notes/random/null_set_finite_modification.png' | relative_url }}" alt="Two functions on [0,1] that agree off a four-point null set have the same integral" loading="lazy">
</figure>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Dirichlet function)</span></p>

The rationals $\mathbb{Q} \cap [0,1]$ are **dense** in $[0,1]$ yet have Lebesgue measure $0$. The Dirichlet function

$$
\mathbf{1}_\mathbb{Q}(x) = \begin{cases} 1, & x \in \mathbb{Q}, \\ 0, & x \notin \mathbb{Q}, \end{cases}
$$

equals the constant function $0$ *almost everywhere*, hence

$$
\int_0^1 \mathbf{1}_\mathbb{Q}(x) \, \mathrm{d}x \;=\; 0.
$$

(Lebesgue integrable, though not Riemann integrable: the set of discontinuities is $[0,1]$, which does *not* have measure zero.)

</div>

<figure>
  <img src="{{ '/assets/images/notes/random/null_set_dirichlet.png' | relative_url }}" alt="Dirichlet function plotted on a dense set of rationals; its Lebesgue integral equals that of the zero function" loading="lazy">
</figure>

### $L^p$ spaces are built on this blindness

The $L^p$ norm

$$
\|X\|_{L^p} \;=\; \bigl( \mathbb{E}[|X|^p] \bigr)^{1/p}
$$

cannot distinguish $X$ from $Y$ when $X = Y$ $\mathbb{P}$-a.s. If we treated such $X$ and $Y$ as different objects, then $\|X - Y\|\_{L^p} = 0$ would *not* imply $X = Y$, and $\|\cdot\|\_{L^p}$ would fail to be a norm. The fix: define $L^p(\Omega, \mathcal{A}, \mathbb{P})$ as the space of *equivalence classes* under the relation

$$
X \sim Y \;\iff\; X = Y \ \mathbb{P}\text{-a.s.}
$$

So $L^p$ lives *one level above* functions — its elements are equivalence classes of functions that are indistinguishable to the integral.

### Why this matters for stochastic processes

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Modification / version)</span></p>

Two stochastic processes $X = (X_t)\_{t \in I}$ and $Y = (Y_t)\_{t \in I}$ on the same space $(\Omega, \mathcal{A}, \mathbb{P})$ are called **modifications** (or *versions*) of each other iff

$$
\mathbb{P}(X_t = Y_t) = 1 \qquad \text{for every } t \in I.
$$

</div>

Modifications are **statistically indistinguishable**: every finite-dimensional distribution, every expectation, every integral against $\mathbb{P}$ agrees. So if one construction of Brownian motion happens to produce discontinuous paths on some null set $N$, one can *modify* the process on $N$ to repair continuity — and the modified process is still Brownian motion (it still satisfies the measure-theoretic axioms and is now also continuous). This is exactly the operation used in **Kolmogorov's continuity theorem**.

<figure>
  <img src="{{ '/assets/images/notes/random/null_set_brownian_modification.png' | relative_url }}" alt="Two modifications of the same Brownian motion: identical in distribution and in all integrals, yet only one has continuous paths" loading="lazy">
</figure>

### What blindness does *not* mean

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(pathwise vs. measure-theoretic properties)</span></p>

The blindness is about quantities defined as integrals against $\mathbb{P}$ — *not* about pathwise or topological properties of a single sample realisation.

* Two modifications can differ pathwise in drastic ways. One sample path can be continuous and another can have jumps — continuity is a *uniform* pathwise property, not a measure-theoretic one.
* The events $\lbrace \omega : t \mapsto X_t(\omega) \text{ is continuous on } I \rbrace$ and $\lbrace \omega : t \mapsto Y_t(\omega) \text{ is continuous on } I \rbrace$ may literally be different sets, even when $X, Y$ are modifications of each other.

That is precisely why (W4) in the definition of Brownian motion (continuity of paths) has to be stated as a separate axiom: it is a property of the whole random function that is *not* deducible from the finite-dimensional distributions alone, and it is exactly the kind of property that null-set modifications can create or destroy.

</div>

## Hölder continuity

Hölder continuity generalises Lipschitz continuity by allowing a *weaker* modulus of continuity than a linear one. It is the natural regularity class for sample paths of stochastic processes such as Brownian motion, and it is the class produced by **Kolmogorov's continuity theorem**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">($\gamma$-Hölder continuity)</span></p>

Let $I \subseteq \mathbb{R}$ be an interval and $\gamma \in (0, 1]$. A function $f : I \to \mathbb{R}$ is **$\gamma$-Hölder continuous** iff there exists a constant $K \ge 0$ such that

$$
|f(t) - f(s)| \le K \, |t - s|^{\gamma} \qquad \text{for all } s, t \in I.
$$

The smallest such $K$ is the **Hölder constant** of $f$ (sometimes written $\lVert f \rVert_{C^{0,\gamma}}$).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(why $\gamma \leq 1$)</span></p>

If $\gamma > 1$ and $f$ is $\gamma$-Hölder on a connected interval, then taking $s \to t$ in $\|f(t)-f(s)\|/\|t-s\| \le K \|t-s\|^{\gamma - 1}$ gives $f'(t) = 0$ everywhere, hence $f$ is constant. So the interesting range is $\gamma \in (0, 1]$.

</div>

### Hierarchy of regularity

Throughout this hierarchy we work on a **compact / bounded** index set $I \subseteq \mathbb{R}^d$ (typically $I = [0, T]$ for the sample-path / SDE setting); without compactness the leftmost inclusion only holds *locally* (a $C^1$ function on $\mathbb{R}$ such as $x \mapsto x^2$ has unbounded derivative and so is not globally Lipschitz). With this proviso, $\gamma$-Hölder continuity sits **between continuity and differentiability**:

$$
\underbrace{C^1(I)}_{\text{differentiable}} \;\subsetneq\; \underbrace{\text{Lipschitz}(I) \;=\; 1\text{-Hölder}(I)}_{\gamma = 1} \;\subsetneq\; \underbrace{\gamma\text{-Hölder}(I), \; 0 < \gamma < 1}_{\text{fractional smoothness}} \;\subsetneq\; \underbrace{C^0(I)}_{\text{continuous}}.
$$

Intuitively:

* $\gamma = 1$ (**Lipschitz**) means the difference quotients are bounded — no vertical tangents; differentiable almost everywhere (Rademacher's theorem).
* $\gamma < 1$ (**strictly fractional**) allows the slope to blow up as $s \to t$, but at a controlled rate: the difference quotient grows at most like $\|t-s\|^{\gamma - 1} \to \infty$. Example: $f(t) = \sqrt{t}$ is $\tfrac{1}{2}$-Hölder on $[0,1]$ but not Lipschitz at $0$.
* The **smaller** $\gamma$ is, the *weaker* the regularity — smaller exponents permit wilder local fluctuations.
* $\gamma$-Hölder for *any* $\gamma > 0$ implies uniform continuity (hence continuity).

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(which $C^1$ — pinning down the domain)</span></p>

The "$C^1$" appearing on the leftmost end of the chain is the **specific** space $C^1(I)$ on a *compact / bounded* domain $I$, not $C^1$ on an arbitrary domain. The distinction is genuine:

* On a **compact** domain $K$ (e.g. $K = [0, T]$), $f \in C^1(K)$ implies $f$ is **globally Lipschitz on $K$** with constant $\sup_K \|f'\|$, which is finite because the continuous function $f'$ attains its maximum on the compact $K$. The mean value theorem then gives $|f(x) - f(y)| \leq \sup_K |f'| \cdot |x - y|$ for all $x, y \in K$.
* On an **open or unbounded** domain $U$ (e.g. $U = \mathbb{R}$), $f \in C^1(U)$ implies $f$ is only **locally Lipschitz** (Lipschitz on every compact subset of $U$), not necessarily globally Lipschitz. The standard counterexample is $f(x) = x^2$ on $\mathbb{R}$: it is $C^\infty$, but
  
  $$
  \frac{|f(x) - f(y)|}{|x - y|} \;=\; |x + y| \;\xrightarrow{|x|, |y| \to \infty}\; \infty,
  $$
  
  so $f$ is *not* (globally) Lipschitz on $\mathbb{R}$.
* The clean global statement is **bounded derivative $\Leftrightarrow$ globally Lipschitz** (on a convex / path-connected domain). Continuity of $f'$ alone gives boundedness only on compacts; that is exactly why the inclusion $C^1 \subsetneq \text{Lipschitz}$ requires the domain to be compact.

If you want a globally-defined function space whose elements are *automatically* Lipschitz everywhere — without assuming the domain is compact — the right object is $C^1_b(U)$, the space of continuously differentiable functions on $U$ with **bounded** derivative; that one fits into the hierarchy on any $U$.

This subtlety is harmless for sample paths of stochastic processes on a bounded time interval $I = [0, T]$ — the implicit setting throughout this section — but worth flagging when transferring the hierarchy to function classes on $\mathbb{R}^d$.

</div>

### Geometric picture: the Hölder cusp

Fix $s \in I$. The $\gamma$-Hölder condition confines the graph of $f$ near $(s, f(s))$ to the "cusp" region

$$
\bigl\lbrace (t, y) \,:\, |y - f(s)| \le K \, |t - s|^{\gamma} \bigr\rbrace.
$$

For $\gamma = 1$ the cusp is a double cone (a linear envelope). For $\gamma = \tfrac{1}{2}$ it is the *parabolic* envelope $y \sim \sqrt{\|t-s\|}$ characteristic of Brownian scaling. For smaller $\gamma$ the envelope is *narrower* near $s$ but opens *faster* — permitting steeper short-time excursions.

<figure>
  <img src="{{ '/assets/images/notes/random/holder_cusps.png' | relative_url }}" alt="Hölder envelopes |t-s|^gamma for gamma = 1, 3/4, 1/2, 1/4 around s=0" loading="lazy">
</figure>

### Concrete examples

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Examples</span><span class="math-callout__name">(different Hölder exponents)</span></p>

* **$f(x) = x$** on $[0,1]$ is **1-Hölder** (Lipschitz) with constant $K = 1$: $\|f(x)-f(y)\| = \|x-y\|$.
* **$f(x) = \sqrt{x}$** on $[0,1]$ is **$\tfrac{1}{2}$-Hölder** but *not* Lipschitz: the derivative $1/(2\sqrt{x})$ blows up at $x = 0$, yet $\|\sqrt{x}-\sqrt{y}\| \le \|x-y\|^{1/2}$.
* **$f(x) = x^{1/3}$** on $[0,1]$ is **$\tfrac{1}{3}$-Hölder**: even wilder near $0$ (steeper cusp), but the $\tfrac{1}{3}$-power envelope still contains it.
* **$f(x) = \mathbf{1}\_{\lbrace x \geq 1/2 \rbrace}$** (step function) is *not* $\gamma$-Hölder for any $\gamma > 0$ on an interval containing $1/2$: the jump violates continuity, let alone Hölder continuity.

</div>

<figure>
  <img src="{{ '/assets/images/notes/random/holder_examples.png' | relative_url }}" alt="Three functions with Hölder exponents gamma = 1 (linear), gamma = 1/2 (square root), gamma = 1/3 (cube root)" loading="lazy">
</figure>

### Hölder-continuous *paths* of a stochastic process

For a stochastic process $X = (X_t)\_{t \in [0, T]}$, the statement "$X$ has $\gamma$-Hölder-continuous paths" is a **pathwise** assertion: for every (or $\mathbb{P}$-a.e.) $\omega \in \Omega$, the realisation $t \mapsto X_t(\omega)$ is $\gamma$-Hölder on $[0, T]$ in the ordinary analytic sense, with a constant $K_\gamma(\omega)$ that may depend on $\omega$:

$$
|X_t(\omega) - X_s(\omega)| \le K_\gamma(\omega) \, |t - s|^{\gamma} \qquad \text{for all } s, t \in [0, T].
$$

Two features matter:

* The inequality is **uniform in $s, t$**: once $\omega$ is fixed, *the same* constant works for every pair of times.
* The constant $K_\gamma : \Omega \to (0, \infty)$ is itself a **random variable** — different sample paths need different constants. The theorem does *not* claim a deterministic $K$ that works for every $\omega$ (for Brownian motion such a uniform bound fails: $\sup_\omega K_\gamma(\omega) = +\infty$).

### Kolmogorov's continuity theorem: moments $\Rightarrow$ Hölder paths

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Kolmogorov's continuity theorem)</span></p>

Let $\widetilde{X} = (\widetilde{X}_t)\_{t \in [0, T]}$ be a real-valued stochastic process such that

$$
\mathbb{E}\!\left[\,|\widetilde{X}_t - \widetilde{X}_s|^{\alpha}\,\right] \;\le\; C \, |t - s|^{\beta} \qquad \text{for all } s, t \in [0, T],
$$

for some $\alpha > 0$, $\beta > 1$, $C > 0$. Then there exists a modification $X$ of $\widetilde{X}$ whose paths are $\gamma$-Hölder continuous for every $\gamma \in \bigl(0, \tfrac{\beta - 1}{\alpha}\bigr)$, with a random Hölder constant $K_\gamma(\omega)$.

</div>

The theorem converts a **moment condition** on increments into **pathwise regularity** of a modification — which is exactly the kind of argument that uses the "measure-theoretic blindness" discussed above: one starts with *any* version of the process and surgically replaces it on a null set to gain continuity.

### Why Brownian motion is $\gamma$-Hölder for $\gamma < \tfrac{1}{2}$ but *not* for $\gamma \geq \tfrac{1}{2}$

For standard Brownian motion $W$, the increment $W_t - W_s \sim \mathcal{N}(0, t-s)$ has Gaussian moments

$$
\mathbb{E}\!\left[\,|W_t - W_s|^{2p}\,\right] \;=\; C_p \, |t - s|^{p} \qquad (p \in \mathbb{N}),
$$

so Kolmogorov's theorem applies with $\alpha = 2p$, $\beta = p$, giving Hölder exponents up to

$$
\gamma \;<\; \frac{\beta - 1}{\alpha} \;=\; \frac{p - 1}{2p} \;\;\xrightarrow[p \to \infty]{}\;\; \frac{1}{2}.
$$

Hence Brownian paths are $\gamma$-Hölder for **every** $\gamma < \tfrac{1}{2}$. The critical exponent $\gamma = \tfrac{1}{2}$ is **not attained**: by **Lévy's modulus of continuity**, the true pathwise modulus of Brownian motion is

$$
\limsup_{h \downarrow 0} \; \sup_{0 \le s \le T - h} \; \frac{|W_{s+h} - W_s|}{\sqrt{2 h \log(1/h)}} \;=\; 1 \qquad \mathbb{P}\text{-a.s.},
$$

which is **slightly coarser** than $\sqrt{h}$ — so $\|W_t - W_s\| \le K \, \|t-s\|^{1/2}$ cannot hold with a finite $K$ on any interval. Consequently Brownian paths are **nowhere differentiable** $\mathbb{P}$-a.s., consistent with $\gamma < 1$ being strict.

<figure>
  <img src="{{ '/assets/images/notes/random/holder_brownian_envelope.png' | relative_url }}" alt="Sample Brownian path inside Hölder envelopes K t^gamma for gamma = 0.25, 0.4, 0.49; tight 1/2-envelope illustrating Lévy's modulus" loading="lazy">
</figure>

The figure shows a single sample path $W_t(\omega)$ on $[0,1]$ together with the **tightest-possible** envelope $K_\gamma(\omega) \, t^{\gamma}$ for $\gamma \in \lbrace 0.25,\, 0.40,\, 0.49 \rbrace$. All three envelopes comfortably contain the path. The dotted envelope at $\gamma = 1/2$ has to be tightened to the supremum of $\|W_t\|/\sqrt{t}$ over the sample — and Lévy's modulus says this supremum fails to be a.s. finite as the time resolution is refined, which is why $\gamma = 1/2$ is *not* an admissible Hölder exponent for Brownian paths.

## Measurable function

## Limsup and Liminf

**limsup** and **liminf** are two ways to talk about the “eventual upper” and “eventual lower” behavior of a sequence (or function), even when the ordinary limit doesn’t exist.

For a sequence $(a_n)$

**Step 1**: look at the “tail supremum/infimum”

For each $n$, consider the tail $\lbrace a_n, a_{n+1}, a_{n+2}, \dots\rbrace$.
* $s_n = \sup \lbrace a_k : k \ge n \rbrace$ = the largest value you can get from the tail (or the least upper bound).
* $i_n = \inf \lbrace a_k : k \ge n \rbrace$ = the smallest value you can get from the tail (or the greatest lower bound).

These tails shrink as $n$ increases, so:
* $s_n$ is a **non-increasing** sequence,
* $i_n$ is a **non-decreasing** sequence.

**Step 2**: take limits of those

$$\limsup_{n\to\infty} a_n := \lim_{n\to\infty} s_n$$
$$\liminf_{n\to\infty} a_n := \lim_{n\to\infty} i_n$$

These limits always exist in the extended real numbers $(-\infty, +\infty]$.

**Intuition**

Equivalently:
* $\limsup$ is the **largest subsequential limit**.
* $\liminf$ is the **smallest subsequential limit**.

<figure>
  <img src="{{ 'assets/images/notes/random/limsup_liminf.png' | relative_url }}" alt="a" loading="lazy">
</figure>

## Cantor’s diagonal argument (classic version)

Cantor’s original diagonal argument was presented for **real numbers**.

Suppose (for contradiction) that you can list *all* real numbers in the interval $(0,1)$ in a sequence:

$$r_0,\ r_1,\ r_2,\ \dots$$

Write each $r_n$ in its decimal expansion:

$$
\begin{aligned}
r_0 &= 0.d_{00}d_{01}d_{02}d_{03}\dots\\
r_1 &= 0.d_{10}d_{11}d_{12}d_{13}\dots\\
r_2 &= 0.d_{20}d_{21}d_{22}d_{23}\dots\\
&\ \vdots
\end{aligned}
$$

Now **construct a new real number** $s\in(0,1)$ by changing the diagonal digits:

$$s = 0.c_0c_1c_2c_3\dots$$

where, for example,

$$
c_n =
\begin{cases}
1 & \text{if } d_{nn}\neq 1\\
2 & \text{if } d_{nn}=1
\end{cases}
$$

(Any rule that guarantees $c_n \neq d_{nn}$ works.)

Then $s$ differs from $r_n$ in the $n$-th decimal place, so $s \neq r_n$ for every $n$. That means $s$ is **not on the list**, contradicting the assumption that the list contained all reals in $(0,1)$.

So $(0,1)$ (hence $\mathbb R$) is **uncountable**. $\square$

## The set of all partial functions is uncountable

*(The set of **partial computable** functions is **countable**)*

**Step 1: Reduce to total functions**

Every **total** function $f:\mathbb N\to\mathbb N$ is also a **partial** function $\mathbb N\rightharpoonup\mathbb N$ (it’s just defined everywhere). So if we show the set $\mathbb N^\mathbb N$ of total functions is uncountable, then the larger set of partial functions is uncountable too.

**Step 2: Diagonal argument for $\mathbb N^\mathbb N$**

Assume for contradiction that all total functions $\mathbb N\to\mathbb N$ can be listed:

$$f_0, f_1, f_2, \dots$$

Now build a new function $g:\mathbb N\to\mathbb N$ by

$$g(n) = f_n(n) + 1.$$

This $g$ is a perfectly valid total function.

But $g$ cannot equal any $f_k$ in the list:

* Compare $g$ to $f_k$ at input $k$:
  
  $$g(k) = f_k(k)+1 \neq f_k(k).$$
  
  So $g \neq f_k$ for every $k$, meaning $g$ is not on the supposed complete list. Contradiction.

Therefore, the set of all total functions $\mathbb N\to\mathbb N$ is **uncountable**.

**Step 3: Conclude for partial functions**

Since

$$\mathbb N^\mathbb N \subseteq \lbrace\mathbb N \rightharpoonup \mathbb N\rbrace,$$

and a set that contains an uncountable subset must be uncountable, the collection of all partial functions $\mathbb N\rightharpoonup\mathbb N$ is **uncountable**.

That’s the core reason: you can always “diagonalize” to produce a function not captured by any countable list.

## The Recursion Theorem



*Proof:*

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Kleene Recursion / Fixed-Point Theorem)</span></p>

For every **computable** function $g:\mathbb N\to\mathbb N$, there exists an index $e_0$ such that

$$\varphi_{g(e_0)} = \varphi_{e_0}.$$

Such an $e_0$ is called a *fixed point* of $g$ (note: typically $g(e_0)\neq e_0$; only the computed functions are equal).

</div>

*proof:*

Let $(\varphi_e)\_{e\in\mathbb N}$ be the standard numbering of the partial computable functions.

**Step 1:** A “compile the result of running program $e$ on input $i$” operator

We use a standard uniformity fact (it’s essentially the s-m-n theorem / effective programming):

> There exists a total computable function $h:\mathbb N^2\to\mathbb N$ such that for all $e,i,x$,
>
> $$
> \varphi_{h(e,i)}(x)\ \simeq
> \begin{cases}
> \varphi_t(x), & \text{if }\varphi_e(i)\downarrow=t,\\
> \uparrow, & \text{if }\varphi_e(i)\uparrow.
> \end{cases}
> $$
> 
> In words: $h(e,i)$ is an index of a machine that first tries to compute $\varphi_e(i)$.

* If that halts with output $t$, it then behaves like $\varphi_t$.
* If that never halts, it diverges everywhere.

(You can think of $h$ as: “run $e$ on $i$, interpret the output as code $t$, then run code $t$ on the actual input $x$”.)

**Step 2:** Diagonalize $h$

Define a total computable function

$$d(e) \ :=\ h(e,e).$$

So $\varphi_{d(e)}$ is “run $\varphi_e(e)$; if it outputs $t$, then become $\varphi_t$”.

**Step 3:** Let $e_1$ be an index for $g\circ d$

Since $g$ and $d$ are total computable, their composition $g\circ d$ is total computable. Therefore it has some index $e_1$ with

$$\varphi_{e_1} = g\circ d.$$

This means: for every input $y$,

$$\varphi_{e_1}(y) = g(d(y)).$$


In particular, plugging in $y=e_1$,

$$\varphi_{e_1}(e_1) = g(d(e_1)).$$

**Step 4:** Define the candidate fixed point and verify it

Let

$$e_0 := d(e_1).$$

We will show $\varphi_{e_0} = \varphi_{g(e_0)}$.

By definition of $e_0$, we have $\varphi_{e_0} = \varphi_{d(e_1)}$.
But by the definition of $d$ and then $h$,

$$\varphi_{d(e_1)} = \varphi_{h(e_1,e_1)}.$$

And by the defining property of $h$,

$$
\varphi_{h(e_1,e_1)} = \varphi_{\varphi_{e_1}(e_1)}
\quad\text{(if }\varphi_{e_1}(e_1)\downarrow\text{ otherwise both sides are everywhere undefined).}
$$

Now substitute the earlier computation $\varphi_{e_1}(e_1)=g(d(e_1))=g(e_0)$. This gives

$$\varphi_{d(e_1)} = \varphi_{g(e_0)}.$$

Since $e_0=d(e_1)$, the left side is $\varphi_{e_0}$. Hence

$$\varphi_{e_0} = \varphi_{g(e_0)},$$

which is exactly the fixed-point property. $\square$

### How to construct the function $h$?

Given the standard enumeration $M_0,M_1,\dots$, of all Turing Machines, it sufficies to choose $h(e,i)$ such that $M_{h(e,i)}$ on input $x$ first simulates $M_e$ on input $i$, and if this computation terminates with output $t$, simulates $M_t$ on input $x$. This involves hardwaring $e$ and $i$ into $M(e,i)$.

Hardwiring $e$ and $i$ into $M_{h(e,i)}$ just means:

> Build a *new* Turing machine whose program text contains the *constants* $e$ and $i$, so the machine doesn’t need to read them as input—they are baked into its code.

So $M_{h(e,i)}$ is a machine **parameterized** by $e$ and $i$, but the parameters are stored inside the machine description.

**Concretely, what does $M_{h(e,i)}$ do?**

For each fixed pair $(e,i)$, define a machine $N_{e,i}$ that on input $x$:

1. Simulate $M_e$ on input $i$.
2. If that simulation halts with output $t$, then simulate $M_t$ on input $x$ and output whatever it outputs. We can simulate $M_t$ effectively (jump to it), because in standard enumeration of Turing Machines, the function $i \mapsto M_i$ is effective (computable) mapping.
3. If $M_e(i)$ never halts, then loop forever.

Here, $e$ and $i$ are not inputs to $N_{e,i}$. They are constants inside its finite control.

### What this proof is *really* doing (intuition in one paragraph)

You want a program $P$ that, when you apply $g$ to its *code*, produces a program with the **same behavior** as $P$. The trick is to build a “template” $d(e)$ that can *look at what program $e$ does on its own code $e$* and then “turn into” whatever program that output names. Then you choose $e_1$ so that on input $e_1$ it outputs $g(d(e_1))$. Feeding this self-input through $d$ forces the behavior of $d(e_1)$ to match the behavior of $\varphi_{g(d(e_1))}$. Setting $e_0=d(e_1)$ gives the fixed point.

<figure>
  <img src="{{ '/assets/images/notes/computability-and-complexity/proof_of_recursion_theorem.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Proof of the Recursion Theorem</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Visual Proof)</span></p>

Let $e_1$ and $e_2$ be indices corresponding to the functions $g\circ d$ and $d$, that is 

$$g\circ d = \phi_{e_1}\qquad d = \phi_{e_2}$$

and note that both functions are total. In the figure above, columns correspond to different indices $e$, keeping the argument $i$ fixed, and rows correspond to different arguments $i$, keeping the index $e$ fixed.

Applying the maping $g$ to the indicces of the form $d(e)$ can be interpreted as transforming column $e_2$ into column $e_1$ in the sense that for all rows $i=0,1,\dots$ the partial function $\phi_{d(i)}$ in column $e_2$ is transformed into the partial function $\phi_{g(d(i))}$ in column $e_1$. In particular, the partial function $\phi_{d(e_1)}$ is transformed into the partial function $\phi_{g(d(e_1))}$, which appears in row $e_1$ and column $e_1$, that is, on the diagonal, hence is equal to $\phi_{\phi_{e_1}(e_1)} = \phi_{d(e_1)}$. Consequently, the index $e_0=d(e_1)$ is a fixed point of $g$.

</div>

**Do we create a 2D grid of partial computable function using $\phi_{h(e,i)}$?**
<div class="accordion">
  <details markdown="1">
    <summary>Do we create a 2D grid of partial computable function using $\phi_{h(e,i)}$?</summary>

Yes — that’s a good mental model.

* $h(e,i)$ is a **computable pairing/compilation function** that takes two numbers $(e,i)$ and returns a single index.
* So $\varphi_{h(e,i)}$ is a family of partial computable functions indexed by a **2D parameter** $(e,i)$.

So you can picture a “grid”:

$$
\begin{matrix}
\varphi_{h(0,0)} & \varphi_{h(0,1)} & \varphi_{h(0,2)} & \cdots\\
\varphi_{h(1,0)} & \varphi_{h(1,1)} & \varphi_{h(1,2)} & \cdots\\
\varphi_{h(2,0)} & \varphi_{h(2,1)} & \varphi_{h(2,2)} & \cdots\\
\vdots & \vdots & \vdots & \ddots
\end{matrix}
$$

But it’s important what this grid *means* here.

### What each cell $\varphi_{h(e,i)}$ does

Each cell is the “conditional wrapper” around the computation $\varphi_e(i)$:

$$
\varphi_{h(e,i)}(x)\simeq
\begin{cases}
\varphi_t(x) & \text{if }\varphi_e(i)\downarrow=t,\\
\uparrow & \text{otherwise.}
\end{cases}
$$

So:

* If program $e$ halts on input $i$ and outputs a code $t$, then the whole cell becomes “run program $t$ on input $x$”.
* If $e(i)$ doesn’t halt, the whole function in that cell is undefined everywhere.

### Why this “grid” matters for the recursion theorem

The diagonal $d(e)=h(e,e)$ is literally “take the diagonal of that grid”:

$$\varphi_{d(e)} = \varphi_{h(e,e)}.$$

So $\varphi_{d(e)}$ is: “run $e$ on its own index $e$; if it outputs $t$, act like $\varphi_t$.”

### One more nuance

This grid is not a *new* enumeration independent of $(\varphi_n)$. It’s a **re-indexing** / **effective transformation** inside the same universe of partial computable functions: each $(e,i)$ gets compiled to a single standard index $h(e,i)$.

  </details>
</div>

**You said: "It’s a re-indexing / effective transformation inside the same universe of partial computable functions." But will every function of the original standard numbering appear in this 2D grid?**
<div class="accordion">
  <details markdown="1">
    <summary>You said: "It’s a re-indexing / effective transformation inside the same universe of partial computable functions." But will every function of the original standard numbering appear in this 2D grid?</summary>

Not necessarily — and in fact, **in this particular construction it definitely won’t**.

### What the grid contains

The grid is the family

$$\lbrace\varphi_{h(e,i)} : e,i\in\mathbb N\rbrace ,$$

where

$$
\varphi_{h(e,i)}(x)\simeq
\begin{cases}
\varphi_t(x) & \text{if }\varphi_e(i)\downarrow=t,\\
\uparrow & \text{otherwise.}
\end{cases}
$$

So every function in the grid is either
1. **everywhere undefined** $(\phi_\uparrow)$, or
2. **exactly equal to** some $\varphi_t$ *for a $t$ that happens to be output by some computation $\varphi_e(i)$*.

Equivalently, the set of functions you see in the grid is

$$\lbrace\phi_\uparrow\rbrace\ \cup\ \lbrace\varphi_t : t\in \mathrm{Range}(\alpha)\rbrace,$$

where $\alpha(e,i)=\varphi_e(i)$ is the (partial) universal function. In words: you only get those $\varphi_t$ whose **index $t$** is produced as an output of *some* program on *some* input.

### Why this does **not** have to be all partial computable functions

To get every $\varphi_t$ in the grid, you’d need:

> For every $t$, there exist $e,i$ such that $\varphi_e(i)\downarrow=t$.

That’s saying the universal partial computable function $\alpha(e,i)=\varphi_e(i)$ is **surjective onto $\mathbb N$** (hits every natural number as an output somewhere). There’s no reason a priori that it must be.

And indeed, it’s easy to see it can fail: there are perfectly valid standard numberings where **no program ever outputs, say, 17** (or any infinite set of numbers), e.g. if you arranged your enumeration so every machine’s outputs are always even, or always $0$, etc. (You can still enumerate all partial computable *functions* while restricting the *codes* that appear as outputs of programs; outputs are just data, not indices.)

So: **this grid is not meant to be an enumeration of all partial computable functions**.

### But don’t we “need” every $\varphi_t$ to appear?

No. For the recursion theorem proof, you only need one fact:

* for each $e,i$, the function $\varphi_{h(e,i)}$ is computable/definable as above, and
* $h$ is total computable (so you can diagonalize $d(e)=h(e,e)$).

The proof never uses “the grid contains all functions”.

### A related fact that *is* always true (and might be what you expected)

Although the specific grid above might not contain every $\varphi_t$, there **does** exist a computable re-indexing $s(e,i)$ (from the s-m-n theorem) such that:

$$\varphi_{s(e,i)}(x)\simeq \varphi_e(\langle i,x\rangle).$$

That kind of grid *does* represent “all computable 2-argument behaviors” sliced into 1-argument functions. But that’s a different construction than your $h$, which is a “run-and-then-jump-to-index” gadget.

If you tell me whether you’re trying to understand the recursion theorem proof specifically, or you’re trying to understand “grids of partial computable functions” more generally, I can show the cleanest grid that’s actually surjective (or explain why we usually don’t care about surjectivity here).

  </details>
</div>

### Why is Kleene's recursion theorem important?

Kleene's Recursion Theorem is one of the most surprising and profound results in computability theory. At its simplest, it proves that **programs can have "self-awareness."**

The theorem essentially states that for any computable transformation of programs, there exists a program that is a "fixed point" for that transformation. In plain English: **You can always write a program that knows its own source code.**

Here is why it is so important:

#### 1. It proves the existence of "Quines"

A **Quine** is a program that takes no input and produces a copy of its own source code as its only output.

* Before Kleene, it wasn't obvious that this was possible without an infinite regress (a program containing a string of itself, which contains a string of itself...).
* The Recursion Theorem provides a mathematical guarantee that such "self-printing" programs exist in every universal programming language.

#### 2. It allows for Self-Referential Algorithms

In standard programming, we use recursion by calling a function by name: `factorial(n-1)`. But what if you couldn't use names? Or what if you wanted a program to modify its own logic based on its own complexity?
The theorem allows us to define functions like:

$$f(e,x) = \dots \text{ something involving } e \dots$$

Where $e$ is the "ID" (source code) of the program $f$ itself. This makes it possible to write programs that say: **"If my own code is shorter than 100 lines, do X; otherwise, do Y."**

#### 3. It provides a tool for Undecidability Proofs

The Recursion Theorem is a "heavy hitter" for proving that certain things are impossible to compute.

**Example: The "Minimal Program" Problem**
Suppose you wanted to write a program  that decides if a given program $M$ is the shortest possible program for a specific task.
1. Using the Recursion Theorem, we can build a "Rebel" program $R$.
2. $R$ looks at its own source code.
3. $R$ then uses $M$ to find a program $P$ that is much longer than $R$ but does the same thing.
4. $R$ then executes $P$. This creates a logical contradiction that proves $M$ (the minimality tester) cannot exist.

#### 4. The Biological Parallel: Self-Replication

The theorem is the mathematical foundation for **self-replication**.
* In biology, DNA contains the "code" to build a cell, but the cell is what reads the DNA to build another cell.
* Kleene’s theorem proves that this "mechanical" self-replication is a fundamental property of any powerful enough information system. It is why computer viruses (which must copy themselves) and cellular automata (like Conway's Game of Life) are mathematically possible.

#### Summary of Importance

| Area | Impact |
| --- | --- |
| **Philosophy** | Shows that "self-reference" doesn't necessarily lead to paradox; it can be stable. |
| **Programming** | Guarantees that programs can access their own source code/ID without infinite loops. |
| **Logic** | Provides a shortcut for proving that certain properties of programs are uncomputable. |
| **AI/Artificial Life** | Establishes the formal requirements for a machine to build a copy of itself. |

#### The "Fixed Point" Intuition

Think of it like this: if you have a map of a park and you drop that map anywhere on the ground inside that park, there is always at least one point on the map that is directly above the actual spot in the park it represents. Kleene's theorem is the "software version" of that geographical fact.


## Legendre Transformation

## Rényi entropy

#TODO:FINISH!

**Rényi entropy** is a one-parameter family of entropy measures that generalizes various notions of entropy, including Hartley entropy, Shannon entropy, collision entropy, and min-entropy by letting you “tune” how strongly you care about high-probability vs. low-probability outcomes. In the context of **fractal dimension** estimation, the Rényi entropy forms the basis of the concept of generalized dimensions

### Definition (discrete distribution)

For a discrete random variable $X$ with probabilities $p_1,\dots,p_n$, the Rényi entropy of order $\alpha$ (with $\alpha \ge 0,\ \alpha \ne 1$) is

$$H_\alpha(X) = \frac{1}{1-\alpha}\log \left(\sum_{i=1}^n p_i^\alpha\right).$$

* The log base sets the units: base $2$ → **bits**, base $e$ → **nats**.

### Key special cases

Rényi entropy smoothly connects several common entropy notions:

* **$\alpha \to 1$** gives **Shannon entropy**:
  
  $$\lim_{\alpha\to 1} H_\alpha(X) = -\sum_i p_i \log p_i.$$

* **$\alpha = 0$** gives **Hartley entropy** (log of support size):
  
  $$H_0(X) = \log\lvert\lbrace i: p_i>0\rbrace\rvert.$$

* **$\alpha = 2$** gives **collision entropy**:
  
  $$H_2(X) = -\log\left(\sum_i p_i^2\right),$$
  
  related to the probability that two independent samples from $X$ are equal.

* **$\alpha \to \infty$** gives **min-entropy**:
  
  $$H_\infty(X) = -\log\left(\max_i p_i\right),$$
  
  which focuses entirely on the most likely outcome.

**Derivation of Shannon entropy:**
<div class="accordion">
  <details markdown="1">
    <summary>Derivation of Shannon entropy</summary>

Sure. Start from Rényi entropy (use natural log for clarity):

$$H_\alpha(p)=\frac{1}{1-\alpha}\ln\left(\sum_{i} p_i^\alpha\right),\qquad \alpha\neq 1.$$

Let

$$S(\alpha)=\sum_i p_i^\alpha.$$

Then

$$H_\alpha(p)=\frac{\ln S(\alpha)}{1-\alpha}.$$

### Step 1: Recognize the $0/0$ form

At $\alpha=1$,

$$
S(1)=\sum_i p_i = 1 \quad\Rightarrow\quad \ln S(1)=\ln 1=0,
$$

and $1-\alpha \to 0$. So the limit $\alpha\to 1$ is $0/0$, and we can use L’Hôpital’s rule.

### Step 2: Apply L’Hôpital’s rule

$$
\lim_{\alpha\to 1} H_\alpha(p)
=\lim_{\alpha\to 1}\frac{\ln S(\alpha)}{1-\alpha}
=\lim_{\alpha\to 1}\frac{\frac{d}{d\alpha}\ln S(\alpha)}{\frac{d}{d\alpha}(1-\alpha)}.
$$

Compute derivatives:

* Denominator:
  
  $$\frac{d}{d\alpha}(1-\alpha)=-1.$$
  
* Numerator:
  
  $$\frac{d}{d\alpha}\ln S(\alpha)=\frac{S'(\alpha)}{S(\alpha)}.$$
  
  And
  
  $$S'(\alpha)=\frac{d}{d\alpha}\sum_i p_i^\alpha=\sum_i \frac{d}{d\alpha}p_i^\alpha=\sum_i p_i^\alpha \ln p_i.$$

So

$$\frac{d}{d\alpha}\ln S(\alpha)=\frac{\sum_i p_i^\alpha \ln p_i}{\sum_i p_i^\alpha}.$$

### Step 3: Evaluate at $\alpha=1$

$$
\lim_{\alpha\to 1} H_\alpha(p)
=\lim_{\alpha\to 1}\frac{\frac{\sum_i p_i^\alpha \ln p_i}{\sum_i p_i^\alpha}}{-1}
= - \frac{\sum_i p_i^1 \ln p_i}{\sum_i p_i^1}.
$$

But $\sum_i p_i^1=\sum_i p_i=1$, so

$$\lim_{\alpha\to 1} H_\alpha(p)= -\sum_i p_i \ln p_i,$$

which is exactly **Shannon entropy** (in nats).

### If you want log base 2 (bits)

If Rényi uses $\log_2$ instead of $\ln$, then

$$\lim_{\alpha\to 1} H_\alpha(p)= -\sum_i p_i \log_2 p_i.$$

(Equivalently, $-\sum_i p_i \ln p_i$ divided by $\ln 2$.)

*Note:* Terms with $p_i=0$ are handled by continuity: $p\ln p \to 0$ as $p\to 0^+$.

  </details>
</div>

### Intuition: what $\alpha$ controls

* Larger $\alpha$ puts **more weight on high-probability outcomes** (more “peak-sensitive”).
* Smaller $\alpha$ puts **more weight on the tail / rare outcomes**.

So Rényi entropy is useful when you want an entropy notion that’s more conservative (large $\alpha$, e.g. security/guessing) or more tail-aware (small $\alpha$, e.g. diversity measures).

### Properties (high-level)

* **Additivity for independent variables**: if $X$ and $Y$ are independent, then
  
  $$H_\alpha(X,Y) = H_\alpha(X) + H_\alpha(Y).$$
  
* For a fixed distribution, $H_\alpha(X)$ is **non-increasing in $\alpha$** (as $\alpha$ increases, entropy typically decreases).

### Continuous version (note)

For a density $f(x)$, a common “differential Rényi entropy” is

$$h_\alpha(X)=\frac{1}{1-\alpha}\log\left(\int f(x)^\alpha,dx\right),$$

but (like differential Shannon entropy) it depends on the coordinate scaling, so it’s often used via *differences* or related divergences.

If you want, I can show how $H_\alpha$ behaves on a simple distribution (e.g., Bernoulli or a “one big probability + small tail” example) to make the $\alpha$-tuning really concrete.

<figure>
  <img src="{{ '/assets/images/notes/random/renyi_entropy.png' | relative_url }}" alt="a" loading="lazy">
  <figcaption>Rényi entropy of a random variable with two possible outcomes against $p_1$, where $P = (p_1, 1 − p_1)$. Shown are $Η_0$, $Η_1$, $Η_2$ and $Η_∞$, with the unit on the vertical axis being the shannon.</figcaption>
</figure>

## Transfer Entropy

**Transfer entropy** is a *nonparametric* measure that quantifies how much information flows in a *directed* (time-asymmetric) way between two random processes. In particular, the transfer entropy from a process $X$ to a process $Y$ captures how much better we can predict the *next* value of $Y$ when we also know the *past* of $X$, beyond what we already learn from the *past* of $Y$ itself.

Let $\lbrace X_t\rbrace$ and $\lbrace Y_t\rbrace$ be two stochastic processes (with $t \in \mathbb{N}$). If information is measured using **Shannon entropy**, the transfer entropy from $X$ to $Y$ with history length $L$ can be expressed as:

$$T_{X\to Y} = H\left(Y_t \mid Y_{t-1:t-L}\right)-H\left(Y_t \mid Y_{t-1:t-L},X_{t-1:t-L}\right),$$

where $H(\cdot)$ denotes Shannon entropy. Variants of this definition can also be built using alternative entropy notions, such as **Rényi entropy**.

An equivalent viewpoint is that transfer entropy is a **conditional mutual information**, where the conditioning includes the past of the affected variable:

$$T_{X\to Y} = I\left(Y_t;,X_{t-1:t-L}\mid Y_{t-1:t-L}\right).$$

For **vector autoregressive (VAR)** processes, transfer entropy aligns with **Granger causality**. This makes it particularly appealing in settings where Granger-style modeling assumptions may be inappropriate—for example, when working with **nonlinear** signals.

## Connection Between Transfer Entropy and Granger Causality

They’re closely related because both ask the same question:

> “Does the past of $X$ help predict $Y$ beyond what $Y$’s own past already explains?”

<figure>
  <img src="{{ '/assets/images/notes/random/transfer_entropy.jpeg' | relative_url }}" alt="a" loading="lazy">
  <!-- <figcaption>Phase portrait of the degenerate (defective) repeated-eigenvalue system</figcaption> -->
</figure>

### Granger causality (GC)

In its standard form (linear VAR models), $X$ *Granger-causes* $Y$ if adding lags of $X$ to a predictive model for $Y_t$ significantly improves prediction (reduces prediction error variance).

### Transfer entropy (TE)

Transfer entropy is an information-theoretic version of that idea:

$$T_{X\to Y} = I\left(Y_t;X_{t-1:t-L}\mid Y_{t-1:t-L}\right),$$

i.e., the extra information $X$’s past provides about $Y_t$, conditioned on $Y$’s past.

### The key connection

For **Gaussian** processes (and with the usual VAR/linear assumptions), **transfer entropy is equivalent to Granger causality up to a constant factor**:

* If variables are (jointly) Gaussian, conditional mutual information reduces to a log-ratio of conditional variances/determinants.
* Granger causality in a VAR is also a log-ratio of prediction error variances (restricted model without $X$ lags vs. full model with them).

Concretely, in the **scalar Gaussian** case:

$$T_{X\to Y} = \tfrac{1}{2}\ln\left(\frac{\mathrm{Var}(\varepsilon_{\text{restricted}})}{\mathrm{Var}(\varepsilon_{\text{full}})}\right),$$

and the right-hand side is exactly the standard (log-variance) GC measure (sometimes defined without the $1/2$, depending on convention).

For the **multivariate Gaussian** case, the same relationship holds with covariance determinants:

$$T_{X\to Y} = \tfrac{1}{2},\ln\left(\frac{\det \Sigma_{\text{restricted}}}{\det \Sigma_{\text{full}}}\right).$$

So:

* **TE = 0** ⇔ **GC = 0** (under Gaussian/VAR conditions)
* **TE and GC rank influences the same way** (monotonic relationship)

### Practical takeaway

* **GC** is typically easier and more statistically mature for **linear-Gaussian** settings.
* **TE** is more general (can capture **nonlinear / non-Gaussian** dependencies), but estimating it well usually requires more data and careful estimators.

For a **linear VAR**, transfer entropy and Granger causality are essentially the *same test* expressed in two languages.

### Setup (VAR with lags)

Let $Z_t = [Y_t^\top, X_t^\top]^\top$. A VAR($L$) for $Y_t$ can be written as:

* **Restricted model** (no $X$ lags):
  
  $$Y_t = \sum_{k=1}^L A_k Y_{t-k} + \varepsilon^{(r)}_t$$
  
* **Full model** (includes $X$ lags):
  
  $$Y_t = \sum_{k=1}^L A_k Y_{t-k} + \sum_{k=1}^L B_k X_{t-k} + \varepsilon^{(f)}_t$$
  

Granger causality $X \to Y$ means the full model predicts better, i.e. the prediction error covariance shrinks.

### Granger causality measure

For multivariate $Y$, the standard (Geweke) GC magnitude is

$$\mathcal{F}_{X\to Y}=\ln\left(\frac{\det \Sigma_r}{\det \Sigma_f}\right),$$

where $\Sigma_r = \mathrm{Cov}(\varepsilon^{(r)}_t)$ and $\Sigma_f = \mathrm{Cov}(\varepsilon^{(f)}_t)$.

For scalar $Y$, $\det$ just becomes variance:

$$\mathcal{F}_{X\to Y}=\ln\left(\frac{\sigma_r^2}{\sigma_f^2}\right).$$

### Transfer entropy for linear VAR (Gaussian innovations)

If the VAR innovations are Gaussian (which is the usual assumption behind the likelihood-based VAR estimation), then transfer entropy is:

$$T_{X\to Y}=\tfrac{1}{2}\ln\left(\frac{\det \Sigma_r}{\det \Sigma_f}\right).$$

So the relationship is simply:

$$\boxed{\mathcal{F}_{X\to Y}=2T_{X\to Y}}$$

(up to log base conventions; using $\log_2$ gives bits instead of nats).

### Intuition

* TE is conditional mutual information: “how much uncertainty about $Y_t$ is removed by adding $X$’s past, given $Y$’s past”.
* In a linear Gaussian VAR, uncertainty = prediction error covariance.
* So TE becomes the log ratio of restricted vs full residual covariance—exactly what GC measures.

### Practical implication

If you’re already fitting a VAR and doing GC, you **don’t gain anything new** by computing TE—TE is just a rescaled GC number in this setting. You *would* gain something if you move beyond linear/Gaussian assumptions.


## Fractal Dimension

## Markov Random Field

A **Markov Random Field (MRF)** is a probabilistic model for a collection of random variables that have a **graph structure** and satisfy a **local “Markov” property**: each variable is conditionally independent of all *non-neighbors* given its *neighbors*. In other words, a random field is said to be a Markov random field if it satisfies Markov properties.

When the joint probability density of the random variables is strictly positive, it is also referred to as a **Gibbs random field**, because, according to the **Hammersley–Clifford theorem**, it can then be represented by a **Gibbs measure** for an appropriate (locally defined) energy function. The prototypical Markov random field is the **Ising model**; indeed, the Markov random field was introduced as the general setting for the Ising model.

### Definition (graphical view)

Let $G=(V,E)$ be an **undirected** graph. Each node $i\in V$ corresponds to a random variable $X_i$. MRF property (one common form):

$$X_i \perp\!\!\!\!\perp X_{V\setminus({i}\cup N(i))}\ \mid\ X_{N(i)}$$

where $N(i)$ is the set of neighbors of node $i$.

### How probabilities are represented (energy / factors)

MRFs are typically written using **clique potentials** (functions over sets of nodes) and normalized into a probability distribution. The key theorem is the **Hammersley–Clifford theorem**, which (under a say “positive distribution” condition) says the joint distribution factorizes over cliques:

$$p(x) = \frac{1}{Z}\prod_{C \in \mathcal{C}} \psi_C(x_C)$$

* $\psi_C(\cdot)$: **potential function** over a clique $C$ (not necessarily a probability)
* $Z$: **partition function** (normalization constant)

Often it’s convenient to use an **energy form**:

$$p(x) = \frac{1}{Z}\exp(-E(x))$$

### Pairwise MRF (very common)

Most practical MRFs use only node and edge terms:

$$p(x)=\frac{1}{Z}\prod_{i\in V}\psi_i(x_i)\prod_{(i,j)\in E}\psi_{ij}(x_i,x_j)$$

This is common in vision (smoothing labels), spatial statistics, etc.

### What it’s used for

* **Image denoising / segmentation**: neighboring pixels tend to have similar labels → enforce smoothness.
* **Spatial data**: nearby locations are correlated (disease mapping, geostatistics).
* **Structured prediction** in ML: assigning labels with dependencies.

### Relationship to other models

* **MRF vs Bayesian network**: MRF uses an **undirected** graph; Bayesian networks use **directed** edges and factorization by parents.
* **MRF vs Conditional Random Field (CRF)**: an MRF models a **joint** distribution $p(X)$; a CRF models a **conditional** distribution $p(Y\mid X)$ (useful when you have inputs/features $X$ and want structured outputs $Y$).

### Inference tasks (what you typically do with an MRF)

* **Marginals**: $p(x_i)$, $p(x_i,x_j)$
* **MAP estimate**: $\arg\max_x p(x)$ (most likely configuration)

Exact inference can be hard on loopy graphs, so people use **belief propagation**, **MCMC**, **variational methods**, or **graph cuts** (for some energy forms).

Perfect context: **Boltzmann Machines (BM), RBMs, and Hopfield nets are all special cases / close cousins of Markov Random Fields**, usually written in the **energy-based** form.

### MRFs as energy-based models (the common language here)

An undirected graphical model (MRF) can be written as:

$$p(x)=\frac{1}{Z}\exp(-E(x))$$

* $x$ is a configuration of all variables (e.g., all neuron states).
* $E(x)$ is an **energy**: lower energy ⇒ higher probability.
* $Z=\sum_x \exp(-E(x))$ is the **partition function** (normalizer).

This is exactly the statistical mechanics / Boltzmann distribution view that BMs use.

### Boltzmann Machine = pairwise binary MRF

A (general) Boltzmann Machine typically has **binary units** $x_i \in \lbrace 0,1\rbrace$ (sometimes $\lbrace -1,+1\rbrace$) on an **undirected graph** (often fully connected), with energy:

$$E(x) = -\sum_i b_i x_i - \sum_{i<j} w_{ij} x_i x_j = -(\sum_i b_i x_i + \sum_{i<j} w_{ij} x_i x_j)$$

That’s a **pairwise MRF**:
* node potentials from biases $b_i$
* edge potentials from weights $w_{ij}$

So a BM **is** an MRF with pairwise interactions (an Ising model / binary pairwise MRF).

The “Markov” part shows up in the conditional:

$$p(x_i=1 \mid x_{-i}) = \sigma\Big(b_i + \sum_{j \in N(i)} w_{ij} x_j\Big)$$

where $N(i)$ are neighbors in the graph. (If it’s fully connected, “neighbors” is everyone else.)

### Restricted Boltzmann Machine = bipartite MRF

An RBM splits units into **visible** $v$ and **hidden** $h$, with edges only across the split (no $v-v$ or $h-h$):

$$E(v,h) = -\sum_i b_i v_i - \sum_j c_j h_j - \sum_{i,j} w_{ij} v_i h_j$$

Graphically: a **bipartite undirected graph** → still an MRF.

The key win: conditional independence becomes trivial:

* Hidden units independent given visibles:
  
  $$p(h \mid v)=\prod_j p(h_j \mid v)$$
  
* Visible units independent given hiddens:
  
  $$p(v \mid h)=\prod_i p(v_i \mid h)$$
  
That’s why Gibbs sampling is fast in RBMs (alternate sampling whole layers).

Also, the model over visibles alone is:

$$p(v)=\sum_h p(v,h)$$

which is still an MRF-ish induced distribution, but the nice factorization happens in the **joint**.

### Hopfield network = deterministic (or zero-temperature) cousin of BM

Classic Hopfield nets are usually:
* binary states $s_i \in \lbrace -1,+1\rbrace$
* symmetric weights $w_{ij}=w_{ji}$, no self connections $w_{ii}=0$
* asynchronous updates that *decrease an energy function*

Energy:

$$E(s) = -\frac{1}{2}\sum_{i\neq j} w_{ij} s_i s_j - \sum_i \theta_i s_i$$

Compare that with the BM energy: it’s the same *form*.

Relationship:
* A **Boltzmann Machine** is like a Hopfield net with **stochastic** updates (finite temperature).
* A **Hopfield net** is like the **$T\to 0$** (zero temperature) limit: it does greedy descent to a local minimum → “associative memory”.

So: Hopfield = (often fully-connected) pairwise MRF **used for optimization** rather than probabilistic modeling.

### “MRF” vs “Boltzmann distribution” vs “Gibbs distribution”

In this context, these words almost collapse into each other:
* **MRF**: the *graph + conditional independence structure* (undirected).
* **Gibbs/Boltzmann distribution**: the *energy-based parameterization* $p \propto e^{-E}$.
* For positive distributions, “MRF factorization over cliques” $\iff$ “Gibbs energy over cliques”.

### Why RBMs matter: structure makes inference cheap-ish

General BMs (general MRFs with loops / full connectivity) make inference and learning hard because:
* $Z$ is expensive
* sampling mixes slowly

RBMs restrict the graph so:
* conditionals factorize
* block Gibbs sampling is easy
  but $Z$ is still hard, so you use approximations like Contrastive Divergence.

Here’s the clean mapping between the two common binary conventions, and how Hopfield / Ising / Boltzmann Machine energies line up.

### Two encodings

* **Spin / Ising / Hopfield convention:**
  
  $$s_i \in \lbrace -1,+1\rbrace$$

* **Neuron / RBM convention:**
  
  $$x_i \in \lbrace 0,1\rbrace$$
  
They’re related by:

$$
s_i = 2x_i - 1
\qquad\Longleftrightarrow\qquad
x_i = \frac{s_i+1}{2}
$$

### Start from a 0/1 Boltzmann Machine energy

A standard BM (pairwise, binary) energy is:

$$E(x)= -\sum_i b_i x_i -\sum_{i<j} w_{ij} x_i x_j$$

Substitute $x_i=(s_i+1)/2$.

#### Expand the pair term

$$x_i x_j = \frac{(s_i+1)(s_j+1)}{4} = \frac{s_i s_j + s_i + s_j + 1}{4}$$

So

$$
-\sum_{i<j} w_{ij} x_i x_j
= -\sum_{i<j}\frac{w_{ij}}{4}s_i s_j
-\sum_{i<j}\frac{w_{ij}}{4}(s_i+s_j)
-\sum_{i<j}\frac{w_{ij}}{4}
$$

#### Expand the bias term

$$-\sum_i b_i x_i = -\sum_i \frac{b_i}{2}s_i -\sum_i \frac{b_i}{2}$$

Now group things into:

* quadratic in spins: $s_i s_j$
* linear in spins: $s_i$
* constant (can be dropped for argmin/MAP, but matters for $Z$)

### Resulting spin-form (Ising/Hopfield-style) energy

You get:

$$E(s)= -\sum_{i<j} J_{ij} s_i s_j -\sum_i h_i s_i + \text{const}$$

with the parameter mapping:

#### Couplings

$$J_{ij} = \frac{w_{ij}}{4}$$

#### Fields (effective biases)

Each edge contributes linear terms to both endpoints. The total field at node $i$ becomes:

$$h_i = \frac{b_i}{2} + \frac{1}{4}\sum_{j\neq i} w_{ij}$$

(assuming (w_{ij}) is symmetric and you’re summing over undirected edges)

Constants:

$$\text{const} = -\sum_i \frac{b_i}{2} - \sum_{i<j}\frac{w_{ij}}{4}$$

Often ignored unless you need exact probabilities.

### Where the Hopfield “$\frac{1}{2}$” comes from

Hopfield energy is often written as:

$$E_{\text{Hop}}(s)= -\frac{1}{2}\sum_{i\neq j} W_{ij} s_i s_j -\sum_i \theta_i s_i$$

But note:

$$\sum_{i\neq j} = 2\sum_{i<j}$$

So the Hopfield form is just avoiding double counting. If you rewrite Hopfield using $i<j$, it’s:

$$E_{\text{Hop}}(s)= -\sum_{i<j} W_{ij} s_i s_j -\sum_i \theta_i s_i$$

So it matches the Ising/BM spin form with:

$$J_{ij} \leftrightarrow W_{ij},\quad h_i \leftrightarrow \theta_i$$

### Probabilistic vs deterministic update (BM vs Hopfield)

Once you’re in spin form $E(s)$:

* **Boltzmann Machine / Ising MRF**:
  
  $$p(s) \propto \exp(-E(s)/T)$$
  
  (Temperature $T$ often absorbed into parameters.)

* **Hopfield** is essentially **$T\to 0$**:
  updates greedily decrease energy → converge to a local minimum (an attractor).

### RBM note (same mapping, just bipartite)

RBM energy in $0/1$:

$$E(v,h) = -b^\top v - c^\top h - v^\top W h$$

You can map $v,h$ to $\pm 1$ spins the same way; you’ll get couplings scaled by $1/4$ and extra field terms coming from row/column sums of $W$.

## Independent Component Analysis

### Introduction

Many statistical models are **generative**: they specify a complete probability distribution over all variables involved. A large and important subclass introduces **latent (hidden) variables** to explain how the observed data are produced.

Typical examples include **mixture models** (where the latent variable indicates the unknown mixture component for each data point), **hidden Markov models**, and **factor analysis**.

Latent variables are often given a **simple** prior distribution, frequently one that factorizes into independent parts. In that sense, fitting a latent-variable model can be viewed as representing the data in terms of “independent components.” The **independent component analysis (ICA)** algorithm can be seen as one of the simplest continuous latent-variable models of this kind.

### Generative model for independent component analysis

Assume we observe $N$ data points $D=\lbrace x^{(n)}\rbrace_{n=1}^N$, where each $x$ is a $J$-dimensional vector. ICA posits that each observation is formed as a **linear mixture** of $I$ underlying source signals $s$:

$$x = Gs \qquad (34.1)$$

where the mixing matrix $G$ is **unknown**.

The most basic setup assumes the number of sources equals the number of observed dimensions, i.e. $I=J$. The goal is then to recover the sources $s$ (typically only **up to** scaling and permutation ambiguities), or equivalently to estimate an inverse (or “unmixing”) transform for $G$ from the samples $\lbrace x^{(n)}\rbrace$.

A key assumption is that the latent components are **independent**, with factorized prior

$$P(s\mid \mathcal H)=\prod_i p_i(s_i),$$

where $\mathcal H$ denotes the modeling assumptions, including the chosen marginal densities $p_i$.

Given $G$ and $\mathcal H$ (note that the probability of the latent vector depends only on the prior distribution given by $\mathcal H$ and is indenendent on the encoding matrix $G$), the joint probability of observations and sources factorizes over samples:

$$P(\lbrace x^{(n)}, s^{(n)}\rbrace_{n=1}^N \mid G,\mathcal H) = \prod_{n=1}^N \Big[P(x^{(n)}\mid s^{(n)},G,\mathcal H)\underbrace{P(s^{(n)}\mid \mathcal H)}_{=P(s^{(n)}\mid \mathcal G,H)} \Big] \qquad \text{34.2}$$

In the **noise-free** ICA model, $x$ is assumed to be generated exactly from $Gs$. In the noise-free model the prediction becomes deterministic and we use Dirac delta function. This makes the conditional distribution a set of delta constraints:

$$= \prod_{n=1}^N \left[ \underbrace{\left(\prod_j \delta\left(x^{(n)}_j - \sum_i G_{ji}s^{(n)}_i\right)\right)}_{P(x^{(n)}\mid s^{(n)},G,\mathcal H)}\underbrace{\left(\prod_i p_i\left(s^{(n)}_i\right)\right)}_{P(s^{(n)}\mid \mathcal H)}\right] \qquad \text{34.3}$$

The assumption of *no observation noise* is not typical in many latent-variable modeling contexts (since real data are rarely noiseless), but it simplifies the inference problem substantially.

### The likelihood function

To estimate the mixing matrix $G$ from the dataset $D$, we focus on the **likelihood**

$$P(D\mid G,\mathcal H)=\prod_n P(x^{(n)}\mid G,\mathcal H). \qquad\text{34.4}$$

So the full likelihood is a product over datapoints, and each factor $P(x^{(n)}\mid G,\mathcal H)$ is obtained by **integrating out** the latent sources $s^{(n)}$.

When carrying out the marginalization, we use the delta-function identity

$$\int \delta(\alpha x-y)f(s)ds=\frac{1}{\lvert \alpha\rvert}f(x/\alpha),$$

and we also adopt a summation convention so expressions like $G_{ji}s_i^{(n)}$ mean $\sum_i G_{ji}s_i^{(n)}$.

For one datapoint, the marginal likelihood is

$$P(x^{(n)}\mid G,\mathcal H)=\int d^I s^{(n)}P(x^{(n)}\mid s^{(n)},G,\mathcal H)P(s^{(n)}\mid\mathcal H), \qquad\text{34.5}$$

and under the noiseless ICA model this becomes

$$=\int d^I s^{(n)}\left[\prod_j \delta\left(x_j^{(n)}-G_{ji}s_i^{(n)}\right)\right] \left[\prod_i p_i\left(s_i^{(n)}\right)\right]. \qquad\text{34.6}$$

Evaluating the integral yields

$$P(x^{(n)}\mid G,\mathcal H)=\frac{1}{\lvert\det G\rvert}\prod_i p_i\left((G^{-1})_{ij}x_j^{(n)}\right). \qquad\text{34.7}$$

Taking logs,

$$\ln P(x^{(n)}\mid G,\mathcal H)= -\ln\lvert\det G\rvert+\sum_i \ln p_i\left((G^{-1})_{ij}x_j^{(n)}\right). \qquad\text{34.8}$$

It is convenient to work with the **unmixing matrix** $W = G^{-1}$. Then the single-sample log-likelihood can be written as

$$\ln P(x^{(n)}\mid G,\mathcal H)= \ln\lvert\det W\rvert+\sum_i \ln p_i\left(W_{ij}x_j^{(n)}\right). \text{34.9}$$

From here on, we assume $\det W>0$, so we can drop the absolute value.

### Gradients and useful identities

To build a maximum-likelihood learning rule, we differentiate the log-likelihood. The following matrix derivatives are used:

$$\frac{\partial}{\partial G_{ji}}\ln\det G = (G^{-1})_{ij}=W_{ij}, \text{34.10}$$

$$\frac{\partial}{\partial G_{ji}}(G^{-1})_{lm}=-(G^{-1})_{lj}(G^{-1})_{im}=-W_{lj}W_{im}, \text{34.11}$$

$$\frac{\partial f}{\partial W_{ij}} = -G_{jm}\left(\frac{\partial f}{\partial G_{lm}}\right)G_{li}. \text{34.12}$$

Define the current estimated sources (sometimes called “activations”)

$$a_i \equiv W_{ij}x_j,$$

and introduce

$$\phi_i(a_i)\equiv \frac{d}{da_i}\ln p_i(a_i). \text{34.13}$$

With these definitions, the gradient of the single-sample log-likelihood w.r.t. $G_{ji}$ becomes

$$\frac{\partial}{\partial G_{ji}}\ln P(x^{(n)}\mid G,\mathcal H)= -W_{ij}-a_i z_l W_{lj}, \text{34.14}$$

where $z_i=\phi_i(a_i)$. Equivalently, differentiating with respect to $W_{ij}$ gives

$$\frac{\partial}{\partial W_{ij}}\ln P(x^{(n)}\mid G,\mathcal H)= G_{ji}+x_j z_i. \text{34.15}$$

If we update (W) by moving **up** the gradient, we obtain the learning rule

$$\Delta W \propto (W^{\top})^{-1}+ z x^{\top}. \text{34.16}$$

This yields the standard ICA-style iterative procedure.

### Algorithm summar

For each datapoint $x$:

1. **Linear step:** compute
   
   $$a = Wx.$$
   
2. **Nonlinear step:** apply a componentwise nonlinearity
   
   $$z_i=\phi_i(a_i),$$
   
   where a common choice is $\phi_i(a_i)=-\tanh(a_i)$.
3. **Weight update:**
   
   $$\Delta W \propto (W^\top)^{-1}+ z x^\top.$$

Intuitively, $z_i=\phi_i(a_i)$ indicates how each component $a_i$ should be nudged to increase the probability of the observed data under the assumed source model.

### Choosing $\phi$

The function $\phi$ effectively encodes the assumed **prior** over the latent sources.

* If we choose a **linear** nonlinearity, $\phi_i(a_i)=-\kappa a_i$, then (through the definition in (34.13)) we are implicitly assuming a **Gaussian** distribution for the sources. But a Gaussian prior is rotationally symmetric, so there is no preferred orientation in latent space—meaning the method cannot identify the true mixing matrix $G$ (or the original sources) from the data. In other words, Gaussian sources are not “separable” by ICA.

* Therefore, ICA relies on the sources being **non-Gaussian** (often with heavier tails than a Gaussian). A widely used choice is
  
  $$\phi_i(a_i) = -\tanh(a_i). \text{34.17}$$
  
  This corresponds to assuming a source density of the form
  
  $$p_i(s_i)\propto \frac{1}{\cosh(s_i)} \propto \frac{1}{e^{s_i}+e^{-s_i}}. \text{34.18}$$
  
  This distribution has **heavier tails** than a Gaussian, which is consistent with many real-world source signals.


## Fisher Information

https://en.wikipedia.org/wiki/Informant_(statistics)
https://en.wikipedia.org/wiki/Observed_information
https://en.wikipedia.org/wiki/Jeffreys_prior
https://en.wikipedia.org/wiki/Fisher_information#Matrix_form
https://en.wikipedia.org/wiki/Fisher_information_metric
https://en.wikipedia.org/wiki/Kullback–Leibler_divergence#Fisher_information_metric

https://www.youtube.com/watch?v=pneluWj-U-o
https://www.youtube.com/watch?v=82molmnRCg0
https://www.youtube.com/watch?v=f_wU0LeNUvE
https://www.youtube.com/watch?v=kmAc1nDizu0
https://www.youtube.com/watch?v=jlh21U2texo
https://www.youtube.com/watch?v=q-Bbxvxpqvw
https://www.youtube.com/watch?v=pneluWj-U-o

https://math.stackexchange.com/questions/3959787/confusion-about-the-definition-of-the-fisher-information-for-discrete-random-var
https://math.stackexchange.com/questions/381364/meaning-of-fishers-information?rq=1
https://math.stackexchange.com/questions/4503324/intuitive-understanding-of-the-fisher-information
https://math.stackexchange.com/questions/265917/intuitive-explanation-of-a-definition-of-the-fisher-information
https://math.stackexchange.com/questions/1314090/intuition-behind-fisher-information-and-expected-value?noredirect=1&lq=1
https://math.stackexchange.com/questions/1523416/what-are-differences-and-relationship-between-shannon-entropy-and-fisher-informa?noredirect=1&lq=1
https://math.stackexchange.com/questions/1892907/what-are-the-measurement-units-of-fisher-information-dimensional-analysis?rq=1 
https://math.stackexchange.com/questions/1858607/what-is-the-difference-between-observed-information-and-fisher-information?noredirect=1&lq=1 
https://math.stackexchange.com/questions/2479473/can-fisher-information-be-zero?rq=1 
https://math.stackexchange.com/questions/1891564/how-many-types-of-mathematical-information-are-there?rq=1 
https://math.stackexchange.com/questions/5069004/intuition-behind-matrix-form-of-fisher-information?rq=1 
https://math.stackexchange.com/questions/5093002/proof-that-the-maximum-likelihood-estimator-attains-the-cramer-rao-bound?rq=1 
https://math.stackexchange.com/questions/3740898/qualitative-understanding-of-fisher-information?rq=1 
https://math.stackexchange.com/questions/3404537/unachievable-cramer-rao-lower-bound?rq=1 
https://math.stackexchange.com/questions/2027945/logliklihood-function-and-fisher-information-martix?rq=1 
https://math.stackexchange.com/questions/4600425/why-is-the-fisher-information-important
https://arxiv.org/pdf/1705.01064
https://arxiv.org/pdf/2208.00549


## Bounded Variation Functions

### Bounded variation in 1D (one variable)

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bounded variation in 1D)</span></p>

Let $f:[a,b]\to\mathbb{R}$. Its **total variation** on $[a,b]$ is

$$V_a^b(f) = \sup_{P}\sum_{i=1}^{n}\lvert f(x_i)-f(x_{i-1})\rvert$$

where the supremum is over all partitions $P:\ a=x_0<x_1<\dots<x_n=b$.

* $f$ has **bounded variation (BV)** on $[a,b]$ if $V_a^b(f)<\infty$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition)</span></p>

You walk along the graph and add up *all* up-and-down changes (absolute increments). BV means the total “amount of oscillation” is finite.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Bounded variation in 1D)</span></p>

* Every **monotone** function is BV, with $V_a^b(f)=\lvert f(b)-f(a)\rvert$.
* $f$ is BV **iff** it can be written as a difference of two increasing functions (Jordan decomposition).
* If $f$ is differentiable and $f'$ is integrable, then
  
  $$V_a^b(f)\le \int_a^b \lvert f'(x)\rvert dx$$
  
  and equality holds when $f$ is absolutely continuous.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Bounded variation in 1D)</span></p>

* $f(x)=\sin x$ on $[0,2\pi]$ is BV (finite total variation).
* A classic non-example: $f(x)=\sin(1/x)$ on $(0,1]$ has *infinite* variation near $0$.

</div>

### Bounded variation in multiple variables

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bounded variation in ND)</span></p>

Let $\Omega\subset\mathbb{R}^n$ be open and $u\in L^1(\Omega)$. We say $u\in BV(\Omega)$ if its **distributional gradient** $Du$ is a **finite vector-valued Radon measure**. The **total variation** of $u$ on $\Omega$ is

$$\lvert Du\rvert(\Omega) < \infty$$

A very usable equivalent definition is the dual (supremum) formula:

$$\lvert Du\rvert(\Omega) =\sup\lbrace\int_{\Omega} u\mathrm{ div }\varphi dx : \varphi\in C_c^1(\Omega;\mathbb{R}^n), \| \varphi\|_\infty\le 1\rbrace$$

* If $u$ is smooth, then $Du=\nabla u dx$ and
  
  $$\lvert Du\rvert(\Omega)=\int_{\Omega}\lvert\nabla u(x)\rvert dx$$

  so BV is the natural extension of “$\nabla u\in L^1$” to non-smooth functions.

This notion reduces to the 1D definition when $n=1$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition)</span></p>

BV functions may have jumps (like edges in images), but the total “edge strength + slope” is finite.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(BV in $\mathbb{R}^n$)</span></p>

* $f(x)=\sin x$ on $[0,2\pi]$ is BV (finite total variation).
* A classic non-example: $f(x)=\sin(1/x)$ on $(0,1]$ has *infinite* variation near $0$.

The indicator $u=\mathbf{1}_E$ of a set $E$ is in $BV(\Omega)$ exactly when $E$ has **finite perimeter** in $\Omega$. Then $\lvert Du\rvert(\Omega)$ corresponds to the surface area of the boundary (in a precise sense).

</div>

## Shannon Entropy Rate

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Shannon Entropy Rate)</span></p>

Shannon entropy rate is the **average information per symbol** generated by a stochastic process.

Let $(X_1,X_2,\dots)$ be a stationary process. Define block entropy

$$H(X_1,\dots,X_n)$$

The **entropy rate** is

$$h = \lim_{n\to\infty} \frac{1}{n}, H(X_1,\dots,X_n)$$

when this limit exists (it does for many common processes; for stationary processes one often uses $\lim$ or $\liminf$).

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition)</span></p>

$H(X_1,\dots,X_n)$ is the average number of bits you need to encode the first $n$ symbols optimally; dividing by $n$ gives “bits per symbol” in the long run.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(i.i.d. case (simplest))</span></p>

If you flip a biased coin with $\Pr(1)=p$ independently each time, then the entropy **per bit** is

$$H(p) = -p\log_2 p - (1-p)\log_2(1-p)$$

This is already a “rate” because each bit is independent and identically distributed.

* If $p=\tfrac12$, $H(p)=1$ bit per flip (max uncertainty).
* If $p$ is near 0 or 1, $H(p)$ is near 0 bits per flip (very predictable).

**In a general process bits may be dependent.**

</div>

## Cylinder Sets

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Cylinder Sets)</span></p>

We work with infinite binary sequences $X = x_0x_1x_2\ldots \in 2^\omega$.

A **finite** binary string is $\sigma \in 2^{<\omega}$ (like $\sigma = 10110$).

The **basic cylinder** generated by $\sigma$ is the set

$$[\sigma] = \lbrace X \in 2^\omega : \sigma \prec X\rbrace$$

meaning: all infinite sequences $X$ whose first $\lvert\sigma\rvert$ bits are exactly $\sigma$.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(All infinite sequences starting with 101)</span></p>

If $\sigma=101$, then

$$[101] = \lbrace 101000\ldots, 101011\ldots, 101111\ldots, \dots\rbrace$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition)</span></p>

Why “cylinder”? Because if you picture $2^\omega$ as an infinite product space $\lbrace 0,1\rbrace^\mathbb{N}$, fixing the first $k$ coordinates and letting the rest vary gives a product set shaped like a “cylinder” in product topology.

**Measure fact (fair coin):** under the fair-coin/Lebesgue measure $\mu$,

$$\mu([\sigma]) = 2^{-\lvert\sigma\rvert}$$

since you need $\lvert\sigma\rvert$ specific coin flips.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Cylinders are interval-like blocks, not sets of isolated numbers)</span></p>

Cylinders are not sets of isolated numbers (points). They are **interval-like blocks** (clopen sets in the product topology).

If you map $X\in 2^\omega$ to a real in $[0,1]$ via binary expansion, then:

* $[\sigma]$ corresponds to a **dyadic interval** of length $2^{-\lvert\sigma\rvert}$:
  * e.g. $[0]$ = $[0, 1/2)$,
  * $[1]$ = $[1/2, 1)$,
  * $[01]$ = $[1/4, 1/2)$,
  * $[101]$ = $[5/8, 6/8)$.

These intervals are equal-length for fixed $\lvert\sigma\rvert$, but they’re **adjacent intervals that tile $[0,1)$**, not “points with equal gaps.”

</div>

## Cantor Space

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Cantor Space)</span></p>

**Cantor space** is the set

$$2^\omega=\lbrace 0,1\rbrace^{\mathbb{N}}$$

the set of all **infinite binary sequences** $x_0x_1x_2\ldots$, equipped with a natural topology (and often a natural probability measure).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(The topology of Cantor Space (how "closeness" works))</span></p>

Two sequences are “close” if they agree on a long initial prefix.

A basic open set is a **cylinder**

$$[\sigma]=\lbrace X\in 2^\omega:\sigma\prec X\rbrace$$

i.e. all sequences starting with a finite string $\sigma$.

Equivalently, you can use a metric:

$$
d(X,Y)=
\begin{cases}
0 & X=Y,\\
2^{-n} & \text{where } n \text{ is the first index with } X(n)\neq Y(n)
\end{cases}
$$

So if the first disagreement happens late, distance is tiny.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Properties</span><span class="math-callout__name">(Cantor Space)</span></p>

With this topology, Cantor space is:
* **compact**
* **totally disconnected** (it’s made of clopen “blocks” like cylinders)
* **perfect** (no isolated points)
* **uncountable**

A topological space is a Cantor space if it is **homeomorphic** to the Cantor set. 

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Connection to the classic Cantor set)</span></p>

Cantor space is homeomorphic (topologically “the same”) as the middle-thirds Cantor set $C\subset[0,1]$.

One explicit link: map a binary sequence $X$ to a ternary expansion using only digits $0$ and $2$:

$$X \mapsto \sum_{n=1}^{\infty} \frac{2X_{n-1}}{3^n}$$

This lands exactly in the middle-thirds Cantor set. (Up to the usual base-expansion ambiguities on a countable set.)

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Use in Martin-Löf randomness)</span></p>

* Strings $\sigma$ define cylinders $[\sigma]$, which are **simple, computable** building blocks.
* The fair-coin measure $\mu$ is natural: $\mu([\sigma])=2^{-\lvert\sigma\rvert}$.
* Martin-Löf tests are built as effective unions of cylinders.

</div>

## Effectively Open Sets

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Effectively Open Sets)</span></p>

A set $U\subseteq 2^\omega$ is **effectively open** (a.k.a. a $\Sigma^0_1$ class) if there is a computably enumerable set $S\subseteq 2^{<\omega}$ such that

$$U=\bigcup_{\sigma\in S}[\sigma]$$

</div>

## Martin-Löf Test

#TODO: what about other measures?
#TODO: what is special about this test?
#TODO: why should the measure of the sequence of subsets decrease by at least two times
#TODO: what is the connection between Kolmogorov complexity and shannon entropy rate?
#TODO: is ML random real incompressable? why? how is it connected to the shannon entropy rate?

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Martin-Löf test)</span></p>

A **Martin-Löf test** (for randomness w.r.t. the fair-coin / Lebesgue measure) is an *effective* way to specify a sequence of “statistical tests” whose rejection regions have rapidly vanishing measure.

A **Martin-Löf test** is a sequence $(U_n)_{n\in\mathbb{N}}$ of subsets of $2^\omega$ such that:

1. **Uniform effective openness:** each $U_n$ is effectively open, and this is *uniform* in $n$.
   Formally, there exists a c.e. set $W \subseteq \mathbb{N}\times 2^{<\omega}$ with
   
   $$U_n=\bigcup \lbrace[\sigma] : (n,\sigma)\in W\rbrace \quad\text{for every }n$$
   
   (Equivalently: there is a computable procedure that, given $n$, enumerates the strings $\sigma$ whose cylinders make up $U_n$.)

2. **Lebesgue Measure bound:** for every $n$,
   
   $$\mu(U_n)\le 2^{-n}$$

The associated **Martin-Löf null set** (the set of sequences that fail the test) is

$$\bigcap_{n\in\mathbb{N}} U_n$$

A sequence $X\in 2^\omega$ **fails** the test if $X\in\bigcap_n U_n$, and **passes** it otherwise (i.e., $X\notin\bigcap_n U_n$).

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Common but optional normalization)</span></p>

Many texts also assume $U_{n+1}\subseteq U_n$ (nestedness). Any Martin-Löf test can be converted into a nested one by replacing $U_n$ with $\bigcap_{k\le n} U_k$, preserving the measure bound up to a constant-factor adjustment.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Choice of opens sets in calculus)</span></p>

The choice of open sets in calculus is **not** arbitrary. Open sets are the natural domains for most of the *local* ideas in calculus.

The key intuition is this:

Calculus studies what happens to a function **near a point**.
To talk about “near,” you usually want to be able to move a little in every direction while staying inside the domain. That is exactly what openness gives you.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why open sets matter)</span></p>

A set $U$ is open if every point $x \in U$ has a small ball around it still contained in $U$.

So if $f : U \to \mathbb{R}$ and $U$ is open, then for every $x \in U$, there are nearby points of the domain all around $x$. This is perfect for:
* limits
* derivatives
* gradients
* Jacobians
* Taylor expansions
* local linear approximations
* inverse / implicit function theorems

All of these are **local** notions. When doing **differential calculus**, we usually restrict attention to open sets because that is the most natural setting for local analysis.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Derivative in one variable)</span></p>

To define

$$f'(a)=\lim_{h\to 0}\frac{f(a+h)-f(a)}{h}$$

you need $a+h$ to be in the domain for small positive and negative $h$.

That automatically happens if the domain contains an open interval around $a$.

If the domain is not open, say $f:[0,\infty)\to\mathbb R$, then at $a=0$ you cannot approach from both sides. You can still define a **one-sided derivative**, but it is less symmetric and many standard theorems need adjustment.

So openness is not required for *every possible definition*, but it is the clean setting where the standard theory works smoothly.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why textbooks often assume “open domain”)</span></p>

Because many theorems become clean and elegant:

* “Let $f:U\to\mathbb R$ be differentiable, where $U\subset \mathbb R^n$ is open…”
* then every point of $U$ is an interior point
* local arguments work without constantly saying “for interior points only”
* no annoying boundary exceptions
* partial derivatives and directional derivatives are naturally defined nearby

So openness is often a **technical assumption that matches the geometry of calculus**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Another way to think about it)</span></p>

Open sets are the sets with **no boundary points included as mandatory stopping points**.

Boundary points cause trouble because near them, part of the nearby space lies outside the domain.

Calculus likes freedom to zoom in around a point without hitting the edge immediately.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Summary)</span></p>

Open sets are special in calculus because:

1. **Calculus is local** — it studies behavior near a point.
2. **Open sets provide neighborhoods** around every point.
3. This lets us approach points from all nearby directions.
4. Standard definitions and theorems become cleaner and more symmetric.
5. If the domain is not open, calculus still works, but usually with extra boundary cases or modified notions.

**Open sets are the natural domains for local differential analysis.**

> Calculus works fine on closed sets, but the **standard two-sided local definitions and theorems** are really about **interior points**.
> On a closed set, boundary points are the problem.

</div>

## Why are Definite and Semi-definite Matrices Important?

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Analogy of positive / non-negative numbers)</span></p>

Because a PSD operator is the mathematical form of “**cannot produce negative energy / curvature / variance**.” That simple sign condition ends up being exactly what many theories need.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(It makes quadratic forms behave like a notion of size)</span></p>

For a general operator $A$, the quantity

$\langle x, Ax\rangle$

can be positive for some directions and negative for others, so it is hard to interpret.

If $A$ is PSD, then

$$\langle x, Ax\rangle \ge 0$$

for every $x$. That means this quadratic form behaves like a generalized squared length, energy, cost, or variance.

That is incredibly useful, because many objects in mathematics and applications are naturally quadratic:
* energy in physics
* squared error in optimization
* variance/covariance in probability
* curvature in second-order analysis
* kernels and Gram matrices in ML
* diffusion/elliptic operators in PDEs

So PSD is often the exact condition that says: “this quantity is physically or mathematically meaningful.”

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(It prevents pathological behavior)</span></p>

If the operator were not PSD, then there would be some direction $x$ with

$$\langle x, Ax\rangle < 0$$

That often means something breaks:

* an “energy” could become negative
* a “variance” could be negative, which is impossible
* an optimization objective could curve downward in a direction you did not want
* a kernel matrix could stop representing valid similarities
* a numerical method could become unstable

So PSD is a natural safety condition.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(It appears in physics as “energy cannot be negative”)</span></p>

Many operators represent energy, stiffness, dissipation, or mass. PSD then means:

* stored energy is nonnegative
* dissipation is nonnegative
* the system is stable rather than self-amplifying in an impossible way

That is why PSD shows up in:

* mechanics
* elasticity
* quantum theory
* control
* PDEs

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The deepest intuition)</span></p>

You keep seeing PSD assumptions because they encode the idea that a linear transformation has **no negative self-interaction**.

When you apply the operator to a vector and compare it with the same vector, the result is never negative. That is exactly the structure needed whenever the operator represents:

* energy
* curvature
* variance
* similarity
* squared magnitude
* diffusion
* stability

So PSD is not an arbitrary technical assumption. It is often the minimal assumption that makes the object represent something real and well-behaved.

</div>

## Spectral Decay

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Spectral Decay)</span></p>

**Spectral decay** means how fast the **singular values** or **eigenvalues** of a matrix get smaller as you move down the spectrum.

If a matrix $A$ has singular values

$$\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r \ge 0$$

then spectral decay describes the behavior of the sequence

$$\sigma_1,\sigma_2,\sigma_3,\dots$$

Similarly, for a PSD matrix, people often talk about its eigenvalues

$$\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_n \ge 0$$

and spectral decay means how fast these $\lambda_i$ decrease.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition)</span></p>

Spectral decay tells you how quickly the “energy” or “information” of a matrix becomes concentrated in the top eigenvalues or singular values. It tells you how much “importance” each direction has.

* If the values drop **very quickly**, then only a few directions matter.
* If they drop **slowly**, then many directions matter.

So spectral decay tells you whether the matrix is effectively:

* **low-rank-ish** or compressible
* or spread across many significant directions

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Slow and fast spectral decay)</span></p>

Suppose the singular values are

$$10, 5, 1, 0.1, 0.01$$

This is **fast spectral decay**: after a few components, everything is tiny.

But if they are

$$10, 9, 8, 7, 6$$

this is **slow spectral decay**: many directions still carry substantial weight.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Common types of decay)</span></p>

People sometimes describe decay as:

* **exponential decay**:
  
  $$\sigma_i \sim c e^{-ai}$$
  
  very fast

* **polynomial decay**:
  
  $$\sigma_i \sim i^{-p}$$
  
  slower

* **flat / weak decay**:
  singular values stay large for a long time

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Applications)</span></p>

Spectral decay is important because it controls how well a matrix can be approximated by a low-rank matrix.

If the singular values decay fast, then truncating after the first $k$ values gives a good approximation:

$$A \approx A_k$$

That matters in:

* PCA
* SVD compression
* kernel methods
* numerical linear algebra
* inverse problems
* machine learning

</div>

## When diagonal of the matrix could be approximated by eigenvalues

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(In general, a diagonal is not approximable by eigenvalues)</span></p>

Usually, **you cannot** approximate a diagonal entry $a_{ii}$ by a “corresponding” eigenvalue $\lambda_i$ without extra assumptions.

The key issue is:

* **Diagonal entries depend on the basis**
* **Eigenvalues do not**

So in a generic basis, there is no natural reason why $a_{ii}$ should be close to any particular eigenvalue.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(When it holds)</span></p>

You get

$$a_{ii}=\lambda$$

exactly when the basis vector $e_i$ is an eigenvector of $A$.
That happens, for example, if the matrix is diagonal in the chosen basis.

So exact equality means: the chosen coordinates are already aligned with the eigenvectors.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(The main approximate case)</span></p>

You can approximate $a_{ii}$ by an eigenvalue when the matrix is **almost diagonal in the given basis**.

Write

$$A = D + E$$

where $D=\operatorname{diag}(a_{11},\dots,a_{nn})$ and $E$ contains the off-diagonal part.

If the off-diagonal entries are small, then the eigenvalues of $A$ are close to the diagonal entries of $D$, at least under suitable separation assumptions.

So the rough rule is:

> **small off-diagonal terms + well-separated diagonal entries**
> $\Rightarrow$
> each eigenvalue is close to one diagonal entry.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

For a any matrix, the diagonal entry is

$$a_{ii} = e_i^* A e_i,$$

which is the **Rayleigh quotient** of the coordinate vector $e_i$ if the matrix $A$ is symmetric/Hermitian.

If $e_i$ is close to an eigenvector $v_k$, then

$$e_i^* A e_i \approx \lambda_k$$

So $a_{ii}$ is close to an eigenvalue when the standard basis vector $e_i$ is close to an eigenvector.

That is the real geometric condition.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Very useful interpretation for symmetric/Hermitian matrices)</span></p>

If $A$ is symmetric/Hermitian with eigenpairs $(\lambda_k,v_k)$, then

$$a_{ii} = e_i^* A e_i = \sum_{k=1}^n \lvert \langle e_i,v_k\rangle \rvert^2 \lambda_k$$

So $a_{ii}$ is a **weighted average of the eigenvalues**.

Therefore:

* if one weight $\lvert\langle e_i,v_k\rangle\rvert^2$ is close to $1$, then
  
  $$a_{ii} \approx \lambda_k;$$
  
* if the mass is spread over many eigenvectors, then $a_{ii}$ is just an average, not close to any one eigenvalue in general.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Gershgorin discs)</span></p>

A very common sufficient condition is based on the row sums of off-diagonal entries.

Define

$$r_i = \sum_{j\ne i} \lvert a_{ij}\rvert$$

Then every eigenvalue lies in the union of Gershgorin discs, and for a symmetric/Hermitian matrix this means intervals

$$[a_{ii}-r_i, a_{ii}+r_i]$$

So if $r_i$ is small, there is an eigenvalue near $a_{ii}$.
If these intervals are also well separated, then you can often identify a specific eigenvalue with that diagonal entry.

So a practical criterion is:

$$
\sum_{j\ne i}\lvert a_{ij}\rvert \ll 1
\quad\text{and the diagonal entries are separated.}
$$

Then $a_{ii}$ is a good approximation to a nearby eigenvalue.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Gershgorin discs)</span></p>

Every eigenvalue $\lambda$ of a matrix $A\in \mathcal{C}^{n\times n}$ lies in a circle centered to $a_{ii}$ and a radius $r_i=\sum_{i\neq j} \lvert a_{ij} \rvert$ for some $i \in \lbrace 1, \dots, n \rbrace$.

</div>

<figure>
  <img src="{{ '/assets/images/notes/linear-algebra/gershgorin_discs.png' | relative_url }}" alt="GPU global memory" loading="lazy">
  <figcaption>Gershgorin discs</figcaption>
</figure>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $\lambda$ be an eigenvalue of the matrix $A$ and $x$ is its corresponding eigenvector. Let $\rvert x_i\lvert = \max_{k} \lvert x_k \rvert$ be the largest by magnitude the entry of the eigenvector $x$. Then

$$\sum_j a_{ij}x_j = \lambda x_i$$

$$ \lambda = \sum_j a_{ij}\frac{x_j}{x_i} = a_{ii} + \sum_{j\neq i} a_{ij}\frac{x_j}{x_i}$$

$$ \lvert \lambda - a_{ii}\rvert = \lvert \sum_{j\neq i} a_{ij}\frac{x_j}{x_i}\rvert \leq \sum_{j\neq i} \lvert a_{ij}\rvert \lvert \frac{x_j}{x_i}\rvert \leq \sum_{j\neq i} \lvert a_{ij}\rvert$$

$\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Perturbation viewpoint)</span></p>

If

$$A = D + E$$

with $D$ diagonal and $E$ small in norm, then for Hermitian matrices the eigenvalues of $A$ are close to those of $D$, i.e. close to the diagonal entries.

A standard bound is:

$$\lvert\lambda_k(A)-\lambda_k(D)\rvert \le \lvert E\rvert_2$$

after appropriate ordering.

Since the eigenvalues of $D$ are just its diagonal entries, this says:

> when the off-diagonal perturbation is small in operator norm, eigenvalues are close to diagonal entries.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Important warning)</span></p>

Even if the matrix is symmetric, it is **not** true in general that

$$a_{ii} \approx \lambda_i$$

Example:

$$
A=\begin{pmatrix}
1 & 1\\
1 & 1
\end{pmatrix}
$$

The diagonal entries are both $1$, but the eigenvalues are $2$ and $0$.
So neither diagonal entry equals a “corresponding” eigenvalue in any obvious way.

This matrix is not close to diagonal in the standard basis, and the coordinate vectors are not eigenvectors.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Summary)</span></p>

You can approximate a diagonal element $a_{ii}$ by an eigenvalue when:

1. the matrix is **diagonal or nearly diagonal** in the chosen basis,
2. the basis vector $e_i$ is **close to an eigenvector**,
3. the off-diagonal couplings are **small**,
4. and ideally the relevant eigenvalue is **well separated** from the others.

For symmetric/Hermitian matrices, the most useful mental picture is:

$$a_{ii} = \text{weighted average of eigenvalues}$$

So $a_{ii}$ is close to one eigenvalue only when one eigenvector dominates that coordinate direction.

If you want, I can show you:

* a **2x2 example** where this approximation works,
* the **precise perturbation formula**,
* or how this connects to **Gershgorin circles** and **Rayleigh quotients**.

</div>

## Eigenvalue does not depend on the basis

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Eigenvalues belong to the operator, not to the coordinates)</span></p>

The **eigenvalue of a linear operator does not depend on the basis**, including the “magnitude” of the basis vectors.

The clean way to see it is this:

If $T:V\to V$, an eigenvalue is defined by

$$T v = \lambda v \quad \text{for some } v\neq 0$$

This statement makes no reference to a basis at all.

A basis is only something you choose afterward to write $T$ as a matrix.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Changing basis changes the matrix, but not the eigenvalues)</span></p>

Suppose $A$ is the matrix of $T$ in one basis, and $A'$ is the matrix in another basis. Then

$$A' = S^{-1} A S$$

for some invertible matrix $S$.

Matrices related this way are called **similar**. Similar matrices always have the same eigenvalues, because

$$\det(A' - \lambda I) = \det(S^{-1}(A-\lambda I)S) = \det(A-\lambda I)$$

So the characteristic polynomial stays the same.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(But diagonal entries can change)</span></p>

This is what I meant by “diagonal entries depend on the basis.”

Example:

$$
A=
\begin{pmatrix}
2 & 1\\
1 & 2
\end{pmatrix}
$$

in the standard basis has diagonal entries $2,2$.

Its eigenvalues are $3$ and $1$.
If you switch to the eigenbasis, the matrix becomes

$$
A'=
\begin{pmatrix}
3 & 0\\
0 & 1
\end{pmatrix},
$$

whose diagonal entries are now $3,1$.

So:

* same operator
* same eigenvalues
* different diagonal entries

That is the sense in which diagonal entries are basis-dependent.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What about changing only the magnitudes of basis vectors?)</span></p>

Rescaling is just a **special kind** of basis change.

If you only rescale basis vectors, then the change-of-basis matrix is diagonal:

$$S = \operatorname{diag}(c_1,\dots,c_n)$$

Then

$$A' = S^{-1} A S$$

In this special case, the diagonal entries actually stay the same:

$$a'_{ii} = a_{ii}$$

So pure rescaling of the basis vectors does **not** change the diagonal entries either. It only changes some off-diagonal entries.

So the full picture is:

* **general basis change**: diagonal entries may change
* **pure rescaling of basis vectors**: diagonal entries stay the same
* **any basis change at all**: eigenvalues stay the same

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(The key distinction)</span></p>

There are two different things:

**Changing the basis**

You are only changing coordinates.
The operator itself stays the same.
So eigenvalues stay the same.

**Scaling the operator**

For example replacing $A$ by $2A$.
Now you changed the actual map.
Then the eigenvalues do change: they get multiplied by $2$.

So “basis scaling” and “matrix scaling” are completely different operations.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Eigenvalues are intrinsic to the linear operator)</span></p>

Eigenvalues are **intrinsic** to the linear operator, while diagonal entries are usually just **coordinate artifacts**.

</div>


## Analytic Function

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Analytic function)</span></p>

A function $f:U\to\mathcal{R}$ is **analytic** at the point $p\in U$ if there exists an open neighborhood $V$ of $p$ (inside $U$) and a power series $\sum_n a_n z^n$ such that 

$$f(x)=\sum_n a_n (z-p)^n$$

for any $x\in V$. As usual, the whole function is analytic if it is analytic at each point.

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Taylor series)</span></p>

For smooth $f$, the power series $\sum_n \frac{f^{(n)}(p)}{n!}z^n$ is called **Taylor series** of $f$ at $p$.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Analytic functions)</span></p>

* $e^x$, $\sin x$, $\cos x$, and polynomials are analytic.
* $\frac{1}{1-x}$ is analytic for $\lvert x\rvert <1$, because
  
  $$\frac{1}{1-x}=\sum_{n=0}^\infty x^n$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Analytic is stronger than smooth)</span></p>

A useful contrast:

* **Smooth** means infinitely differentiable.
* **Analytic** is stronger: the function must equal its Taylor series near each point.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Smooth, but not analytic functions)</span></p>

So a function can be smooth but **not** analytic. A classic example is

$$
f(x)=
\begin{cases}
e^{-1/x^2}, & x\neq 0 \\
0, & x=0
\end{cases}
$$

This function is infinitely differentiable at $0$, but its Taylor series at $0$ is just $0$, which does not equal the function for $x\neq 0$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(All your usual closure properties for analytic functions)</span></p>

The sums, products, compositions, nonzero quotients of analytic functions are analytic.

</div>

## Divergence

### What is Divergence?

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Divergence)</span></p>

**Divergence** measures how much a vector field behaves like a **source** or a **sink** at a point.

If $F(x) = (F_1(x), \dots, F_d(x))$, then

$$\operatorname{div} F(x) = \sum_{i=1}^d \frac{\partial F_i}{\partial x_i}(x)$$

So it adds up how strongly each component changes in its own coordinate direction.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Intuition)</span></p>

Think of $F(x)$ as a velocity field of a fluid.

* **Positive divergence** at a point: fluid is, locally, flowing outward from that point. It looks like a source.
* **Negative divergence**: fluid is flowing inward. It looks like a sink.
* **Zero divergence**: no net creation or removal of fluid locally. What flows in roughly equals what flows out.

A very important phrase is:

> divergence = **net outward flow per unit volume**

So it is not about whether the field is large. A vector field can have large magnitude but zero divergence.

For example, a constant flow to the right has zero divergence: fluid moves, but it is not being created or destroyed anywhere.

**The deepest intuition:**

Divergence answers:

**Is stuff being locally created, destroyed, or conserved here?**

Depending on the context, “stuff” could mean:

* fluid
* electric flux
* probability mass
* heat
* particles
* volume under a dynamical flow

So divergence is a local measure of **expansion/compression** or **source/sink behavior**.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Constant field)</span></p>

In 2D, if $F(x,y) = (P(x,y), Q(x,y))$, then

$$\operatorname{div} F = \frac{\partial P}{\partial x} + \frac{\partial Q}{\partial y}$$

* $\partial P/\partial x$: does the horizontal flow get stronger as you move in the horizontal direction?
* $\partial Q/\partial y$: does the vertical flow get stronger as you move in the vertical direction?

If both happen, more flow leaves a tiny region than enters, so divergence is positive.

$$F(x,y) = (1,0)$$

Then

$$\operatorname{div} F = 0 + 0 = 0$$

**Interpretation:** everything moves right, but there is no local expansion or compression.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Expanding field)</span></p>

$$F(x,y) = (x,y)$$

Then

$$\operatorname{div} F = 1 + 1 = 2$$

**Interpretation:** vectors point away from the origin and get larger outward, so the field acts like a source.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Contracting field)</span></p>

$$F(x,y) = (-x,-y)$$

Then

$$\operatorname{div} F = -1 + (-1) = -2$$

**Interpretation:** everything points inward, like a sink.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Rotational field)</span></p>

$$F(x,y) = (-y,x)$$

Then

$$\operatorname{div} F = 0 + 0 = 0$$

**Interpretation:** pure rotation. Things swirl around, but there is no net local outflow.

This example is useful because it shows divergence is **not** about spinning. Spinning is related to **curl**, not divergence.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Applications: Where it is used)</span></p>

**Fluid mechanics**

Divergence tells whether fluid is locally expanding or compressing.

For incompressible flow,

$$\operatorname{div} u = 0$$

This means fluid density is preserved locally.

**Electromagnetism**

Gauss’s law relates divergence of the electric field to charge density:

$$\nabla \cdot E = \frac{\rho}{\varepsilon_0}$$

So charges act like sources or sinks of electric field.

**PDEs and conservation laws**

Divergence appears in equations expressing conservation of mass, momentum, heat, probability, and so on.

A typical conservation law looks like

$$\frac{\partial \rho}{\partial t} + \nabla \cdot J = 0$$

where $J$ is a flux. This says: change in density = minus net outflow.

**Differential geometry / vector calculus**

It is a basic operator alongside:
* gradient
* divergence
* curl
* Laplacian

**Probability and generative modeling**

In continuous normalizing flows and diffusion-related topics, divergence appears because it measures how a flow changes local volume. That is why it enters log-density evolution formulas.

For example, along a flow $\dot{x} = f(x,t)$, the density changes according to divergence of $f$. Roughly:

* positive divergence expands volume, decreasing density
* negative divergence contracts volume, increasing density

</div>

### How is Divergence related to Laplacian

#TODO:

### Why is computing Divergence expensive using autodiff?

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Divergence is expensive with autodiff)</span></p>

Divergence is the **trace of the Jacobian**.

If you have a vector field $f(x) \in \mathbb{R}^d$, then

$$\operatorname{div} f(x) = \sum_{i=1}^d \frac{\partial f_i(x)}{\partial x_i}$$

So even though the final result is just **one scalar**, it depends on **many partial derivatives**.

The expensive part in autodiff is this:

* To get all diagonal terms $\partial f_i / \partial x_i$, you effectively need access to the Jacobian information.
* Standard autodiff is very good at:

  * gradients of a scalar output w.r.t. many inputs, or
  * Jacobian-vector products / vector-Jacobian products.
* But divergence needs **many entries of the Jacobian**, not just one gradient.

In practice, that means one of two costly options:

1. **Form the whole Jacobian**

   * This is usually very expensive in memory and time.
   * A $d \times d$ Jacobian has $d^2$ entries.

2. **Compute the diagonal terms one by one**

   * Often requires about $d$ separate autodiff calls.
   * So cost scales roughly linearly with dimension, on top of the cost of evaluating the network.

That is why in high dimensions, exact divergence becomes expensive.

A useful intuition:

* Computing $f(x)$: one forward pass.
* Computing divergence exactly: often like doing **many backward/forward passes**, one per dimension or equivalent work.

This is also why methods like the **Hutchinson trace estimator** are popular: they estimate the divergence (trace) without explicitly computing the full Jacobian diagonal, making it much cheaper.

</div>

### When and Why is Hutchinson estimator cheaper?

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(When and Why is Hutchinson estimator cheaper?)</span></p>

**Why is Hutchinson estimator cheaper?**

Because Hutchinson is usually **not** replacing “compute the trace of a matrix you already have explicitly.” It replaces “compute the trace of a matrix that is too expensive to form or whose diagonal is expensive to get.”

The key identity is

$$\operatorname{tr}(A)=\mathbb{E}[z^\top A z]$$

for a random vector $z$ with suitable properties, so you estimate

$$\operatorname{tr}(A)\approx \frac1m \sum_{i=1}^m z_i^\top A z_i$$

The speedup comes from the fact that this only needs **matrix-vector products**, not the full matrix or all diagonal entries.

**If the matrix is explicitly available**

If you literally have a dense matrix $A \in \mathbb{R}^{n\times n}$ in memory, then yes:

* exact trace = sum of diagonal entries = $O(n)$
* one Hutchinson sample = one matvec $Az$ = $O(n^2)$ for dense $A$

So in that case, Hutchinson is obviously worse.

**When Hutchinson helps**

It helps when $A$ is available only **implicitly**, for example as:

* a Hessian
* a Jacobian-derived operator
* an inverse like $A^{-1}$
* $\log A$, $f(A)$, or other matrix function
* a huge sparse / structured linear operator

Then:

* getting all diagonal entries may require $n$ expensive solves or $n$ passes
* forming the full matrix may cost $O(n^2)$ memory or worse
* but computing $Az$ for a vector $z$ may be relatively cheap

So instead of paying for all $n$ columns / diagonal entries, you pay for only $m$ operator applications, where often $m \ll n$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why “trace = sum of diagonal” is misleading here)</span></p>

The formula itself is cheap **only if the diagonal is already accessible**.

But in many applications, to get $A_{ii}$ you do not have direct access to the entries of $A$. You only have access to the action $v \mapsto Av$. In that model:

* exact trace may require recovering all diagonal entries
* Hutchinson estimates the sum of diagonals without recovering them individually

That is the whole point.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Cost comparison and Tradeoff)</span></p>

**Cost comparison**

If $A$ is dense and explicit:

* exact trace: $O(n)$
* Hutchinson: $O(mn^2)$

No advantage.

If $A$ is implicit and matvec cost is $T$:

* exact trace may be $O(nT)$ or impossible without forming $A$
* Hutchinson is $O(mT)$

Advantage when $m \ll n$.

**Tradeoff**

Hutchinson gives:

* **lower cost**
* **lower memory**
* but only an **approximation**

You accept variance in exchange for avoiding a much more expensive exact computation.

So your intuition is correct: if you can read off the diagonal cheaply, do that. Hutchinson is useful precisely when you **cannot**.

</div>

### Time complexity of exact trace estimation

Let $T$ be the cost of applying the matrix/operator $A$ to **one vector**

So if $A$ is only available as an operator, not as explicit entries, then getting the **exact** trace often means recovering all diagonal entries, and that takes about **$n$** operator-level computations.

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Trace is easy only for an explicit matrix)</span></p>

If you have the full matrix $A\in\mathbb R^{n\times n}$ stored, then

$$\operatorname{tr}(A)=\sum_{i=1}^n A_{ii},$$

so you just read the diagonal. That is $O(n)$.

But that is **not** the setting people mean when they write exact trace is $O(nT)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Implicit matrix access: Operator-access setting)</span></p>

Suppose you do **not** have entries of $A$. You only know how to compute

$$v \mapsto Av$$

and one such application costs $T$.

To compute the exact trace, you need all diagonal entries $A_{11},\dots,A_{nn}$.

How do you get $A_{ii}$ from operator access?

Use the standard basis vector $e_i$. Then

$$Ae_i$$

is exactly the $i$-th column of $A$. Its $i$-th component is

$$e_i^\top A e_i = A_{ii}$$

So one way to get the exact trace is:

1. for each $i=1,\dots,n$, apply $A$ to $e_i$
2. extract the $i$-th component
3. sum them up

That is $n$ operator applications, each costing $T$, hence

$$O(nT).$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Important nuance)</span></p>

$O(nT)$ is not a universal law. It means:

* **if** the only access you have to $A$ is via operator applications,
* **and** one application costs $T$,
* **then** a straightforward exact trace computation needs $n$ such applications.

If the matrix is explicit, trace is just $O(n)$, not $O(nT)$.

So the phrase “exact trace is $O(nT)$” really belongs to the **implicit-operator setting**.

</div>


### How can we compute $Az$ without computing having $z$?

Because in many problems you do **not** have the entries of $A$, but you do have a procedure that computes the action of $A$ on a vector.

That is what "implicit matrix" means.

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Explicit matrix vs. Implicit linear operator)</span></p>

**1. Explicit matrix**

You store all entries of $A$. Then computing $Az$ is the usual multiplication.

**2. Implicit linear operator**

You do **not** store $A$, but you know a routine

$$z \mapsto Az$$

without ever materializing the full matrix.

This happens a lot.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Hessian)</span></p>

Let $A = \nabla^2 f(x)$, the Hessian.

You usually do **not** form the full Hessian matrix, because it is huge. But with autodiff you can compute a Hessian-vector product

$$\nabla^2 f(x), z$$

directly.

So you can evaluate $Az$ even though you never built $A$.

This is much cheaper than computing every Hessian entry.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Jacobian)</span></p>

Suppose $A = J(x)$, a Jacobian. You may not form all partial derivatives, but autodiff can compute:

* $Jz$ by forward-mode AD
* $J^\top z$ by reverse-mode AD

again without storing the whole Jacobian.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Matrix function)</span></p>

If $A = \log(B)$ or $A = f(B)$, you may approximate $Az = f(B)z$ using Krylov methods, Chebyshev methods, Lanczos, etc., without ever forming $f(B)$.

</div>

### Classic example: Hessian-vector products in deep learning / optimization

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Large Hessian you do not want to compute)</span></p>

Let

$$A = \nabla^2 L(\theta),$$

the Hessian of a loss $L(\theta)$ with respect to parameters $\theta$.

If a model has millions of parameters, then $A$ is a huge $n\times n$ matrix. You would never form it explicitly.

But you can still compute

$$Az = \nabla^2 L(\theta), z$$

for a given vector $z$.

If $n=10^6$, then the Hessian would have $10^{12}$ entries, which is impossible to store in practice.

But one Hessian-vector product can often be computed with a cost comparable to a small number of gradient evaluations.

So here you truly compute $Az$ without ever having the full $A$.

</div>


### How to avoid computing the whole Hessian for matvec

Define

$$g(\theta)=\nabla L(\theta).$$

Then consider the scalar

$$\phi(\theta)=g(\theta)^\top z$$

Now differentiate again:

$$
\nabla_\theta \phi(\theta) = \nabla_\theta \big(g(\theta)^\top z\big)
\nabla^2 L(\theta), z
$$

So instead of building the whole Hessian, you:

1. compute the gradient $g(\theta)$
2. form the scalar $g(\theta)^\top z$
3. differentiate that scalar w.r.t. $\theta$

and the result is exactly $Hz$.

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What is this used for?)</span></p>

This shows up in:

* Newton-type optimization methods
* curvature estimation
* trace estimation with Hutchinson
* Laplace approximation
* influence functions
* second-order analysis of neural networks

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Inverse os sparse matrix)</span></p>

Let

$$A = B^{-1}$$

for some large sparse matrix $B$.

You do not form $A$ explicitly, because inverting a huge matrix is expensive and destroys sparsity.

But to compute

$$Az = B^{-1}z,$$

you just solve the linear system

$$Bx = z$$

Then $x = Az$.

</div>

## On Covariance Matrix of Gaussian I

For a 2D Gaussian with mean $0$, rotation and elongation are described by the **covariance matrix**

### The natural variances are along the ellipse’s principal axes

If you diagonalize (\Sigma),
[
\Sigma = Q \Lambda Q^\top,
\qquad
\Lambda=
\begin{pmatrix}
\lambda_1 & 0\
0 & \lambda_2
\end{pmatrix},
]
then:

* the columns of (Q) give the two principal directions
* (\lambda_1,\lambda_2) are the variances in those directions

So the elongated rotated Gaussian has:

* one **large eigenvalue** = large variance along the long axis
* one **small eigenvalue** = small variance along the short axis

### Is variance the radius of the ellipse?

Not exactly.

The ellipse is usually drawn as a **constant-density** or **constant Mahalanobis distance** contour:
[
x^\top \Sigma^{-1} x = c.
]

Along the principal axes, the semi-axis lengths of this ellipse are
[
\sqrt{c,\lambda_1}, \qquad \sqrt{c,\lambda_2}.
]

So:

* **variance** = (\lambda_i)
* **standard deviation** = (\sqrt{\lambda_i})
* **ellipse radius / semi-axis length** at contour level (c) = (\sqrt{c,\lambda_i})

Thus the “radius” is proportional to the **standard deviation**, not the variance.

### Example

Suppose
[
\Sigma = Q
\begin{pmatrix}
9 & 0\
0 & 1
\end{pmatrix}
Q^\top
]
for some rotation matrix (Q).

Then the Gaussian is rotated, centered at (0), and:

* variance along one principal axis = (9)
* variance along the other = (1)
* standard deviations are (3) and (1)

So the cloud is 3 times wider in one principal direction than in the other.

If you draw the contour
[
x^\top \Sigma^{-1}x = 1,
]
the ellipse semi-axes are (3) and (1).

### Important distinction

If the Gaussian is rotated, then the diagonal entries of (\Sigma) are usually **not** the principal variances. They are only variances along the coordinate axes.

The actual elongation is captured by the eigenvalues of (\Sigma).

A good summary is:

* mean = center of the Gaussian
* eigenvectors of covariance = orientation
* eigenvalues of covariance = variances along the principal axes
* square roots of eigenvalues = ellipse radii up to a scaling constant

### Example

<figure>
  <img src="{{ '/assets/images/notes/random/2D_Gaussian_Ellipse.png' | relative_url }}" alt="a" loading="lazy">
  <!-- <figcaption>The dynamical system is linear, but the trajectory is not necessarily linear</figcaption> -->
</figure>

Yes — here is a sketch:

[rotated_gaussian_sketch.png](sandbox:/mnt/data/rotated_gaussian_sketch.png)

How to read it:

* The center is the mean, here (\mu=(0,0)).
* The ellipse is a contour of equal density, for example
  [
  x^\top \Sigma^{-1}x = 1.
  ]
* The long and short tilted directions are the eigenvector directions of (\Sigma).
* Their lengths on this contour are
  [
  \sqrt{\lambda_1},\quad \sqrt{\lambda_2},
  ]
  where (\lambda_1,\lambda_2) are the eigenvalues of (\Sigma).

So in the picture:

* variance along the long principal axis is (\lambda_1 = 9)
* variance along the short principal axis is (\lambda_2 = 1)
* the semi-axis lengths of this particular ellipse are (3) and (1)

So the “radius of the ellipse” corresponds to **standard deviation**, not variance.

### in \Sigma = Q\Lambda Q^\top, the columns of Q are in which basis?

They are usually written in the **original coordinate basis** you are using for (x), for example the standard basis (e_1,e_2).

So in
[
\Sigma = Q \Lambda Q^\top,
]
the columns of (Q) are the eigenvectors of (\Sigma), expressed in your current coordinates.

More concretely, in 2D:

* suppose you describe points as (x=(x_1,x_2)) in the standard basis
* then each column (q_1,q_2) of (Q) is a vector in that same (x_1,x_2) coordinate system
* these vectors give the directions of the principal axes of the Gaussian ellipse

If

$$
Q=\begin{pmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \cos\theta
\end{pmatrix},
$$

then:

* first column:
  [
  q_1=\begin{pmatrix}\cos\theta\ \sin\theta\end{pmatrix}
  ]
  is the first principal direction, written in the standard basis
* second column:
  [
  q_2=\begin{pmatrix}-\cos? \text{ no, } -\sin\theta\ \cos\theta\end{pmatrix}
  ]
  sorry, correctly:
  [
  q_2=\begin{pmatrix}-\sin\theta\ \cos\theta\end{pmatrix}
  ]
  is the second principal direction, also in the standard basis

So the answer is:

* **the columns of (Q) are the new principal-axis basis vectors**
* but they are **represented in the old/original basis**

This is the standard linear algebra convention.

A useful interpretation:

* $\Lambda$ describes the variances in the principal-axis coordinate system
* $Q$ tells you how that principal-axis system sits inside the original coordinates

Equivalently, if $z$ are coordinates in the principal basis and $x$ are coordinates in the original basis, then

$$x = Q z.$$

So:

* columns of $Q$ = principal basis vectors
* written in the original basis

For the Gaussian, if

$$z \sim \mathcal N(0,\Lambda),$$

and

$$x=Qz,$$

then

$$
x \sim \mathcal N(0,Q\Lambda Q^\top).
$$

That is exactly why the rotated covariance has that form.

If you want, I can also explain the difference between “basis vectors as geometric vectors” and “coordinates of a vector in a basis,” since that is usually the point that causes confusion here.

### Do roots of the Lambda values correspond to the radius?

Yes — for the usual covariance ellipse, the **square roots of the eigenvalues** are the semi-axis lengths.

If

$$
\Sigma = Q\Lambda Q^\top,\qquad
\Lambda=\operatorname{diag}(\lambda_1,\lambda_2),
$$

then the contour

$$x^\top \Sigma^{-1} x = 1$$

is an ellipse whose principal semi-axes have lengths

$$\sqrt{\lambda_1},\qquad \sqrt{\lambda_2}.$$

So:

* (\lambda_i) = variance along principal direction (q_i)
* (\sqrt{\lambda_i}) = standard deviation along that direction
* on the contour (x^\top\Sigma^{-1}x=1), that is exactly the “radius” in that principal direction

More generally, for the contour

$$x^\top \Sigma^{-1}x = c,$$

the semi-axis lengths are

$$\sqrt{c,\lambda_1},\qquad \sqrt{c,\lambda_2}$$

So the root of the eigenvalue is the radius only for the (c=1) ellipse; otherwise it is scaled by (\sqrt c).

A compact way to see it:
set $x = Qy$. Then

$$
x^\top \Sigma^{-1}x
= y^\top \Lambda^{-1} y
= \frac{y_1^2}{\lambda_1} + \frac{y_2^2}{\lambda_2}.
$$

For level $1$, this is

$$
\frac{y_1^2}{\lambda_1}+\frac{y_2^2}{\lambda_2}=1,
$$

which is the standard ellipse equation with semi-axes $\sqrt{\lambda_1},\sqrt{\lambda_2}$.

So yes: **roots of the $\Lambda$ values correspond to the ellipse radii along the principal axes**.

### What is the connection between the elongation of the ellipse of 2D gaussian and the regularity of the covariance matrix? what about the conditional number of the covariance matrix?

Yes — the elongation of a 2D Gaussian ellipse is **directly tied** to the eigenvalues of the covariance matrix, and the **condition number** is exactly a measure of that elongation.

Let

$$
\Sigma = Q \Lambda Q^\top,
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\lambda_2),
\qquad
\lambda_1 \ge \lambda_2 \ge 0.
$$

Then the Gaussian’s principal axes are given by the columns of (Q), and the covariance ellipse

$$x^\top \Sigma^{-1} x = 1$$

has semi-axis lengths

$$\sqrt{\lambda_1},\qquad \sqrt{\lambda_2}.$$

So the shape is controlled by the eigenvalues.

---

## Elongation

The ellipse is elongated when one eigenvalue is much bigger than the other.

Its aspect ratio is
[
\frac{\text{long radius}}{\text{short radius}}
==============================================

# \frac{\sqrt{\lambda_1}}{\sqrt{\lambda_2}}

\sqrt{\frac{\lambda_1}{\lambda_2}}.
]

So:

* if (\lambda_1=\lambda_2), the contour is a circle
* if (\lambda_1 \gg \lambda_2), the contour is a very stretched ellipse
* if (\lambda_2=0), it collapses to a line

---

## Condition number

For a symmetric positive definite covariance matrix, the 2-norm condition number is

$$\kappa(\Sigma)=\frac{\lambda_{\max}}{\lambda_{\min}} =\frac{\lambda_1}{\lambda_2}$$

Therefore,

$$\text{aspect ratio of ellipse} = \sqrt{\kappa(\Sigma)}$$

This is the cleanest connection.

So:

* **small condition number** $\approx 1$ means nearly round ellipse
* **large condition number** means highly elongated ellipse
* **infinite condition number** means singular covariance

---

## What “regularity” means here

There are two common meanings people use.

### 1. Positive definite / nonsingular

A covariance matrix is:

* **positive definite** if $\lambda_1>0$ and $\lambda_2>0$
* **singular** if one eigenvalue is zero

Geometrically:

* positive definite $\Rightarrow$ genuine 2D ellipse
* singular $\Rightarrow$ ellipse collapses into a lower-dimensional object

So if the ellipse gets more and more elongated because $\lambda_2\to 0$, the covariance becomes closer and closer to singular.

### 2. Well-conditioned

A covariance matrix is “regular” in the numerical sense if it is not close to singular, meaning its condition number is not too large.

Geometrically:

* well-conditioned covariance = not too elongated
* ill-conditioned covariance = very thin ellipse

---

## Intuition

Variance in direction $u$ is

$$u^\top \Sigma u$$

So if one direction has huge variance and an orthogonal one has tiny variance, the Gaussian spreads a lot in one direction and barely at all in the other. That is exactly what a long thin ellipse is.

The matrix becomes hard to invert stably when one variance is tiny, because inversion turns

$$\lambda_i \mapsto \frac{1}{\lambda_i}$$

So a very small $\lambda_2$ becomes a very large $1/\lambda_2$, which is why near-singular covariance matrices are numerically unstable.

---

## Example

If

$$
\Lambda=
\begin{pmatrix}
9 & 0\
0 & 1
\end{pmatrix},
$$

then:

* radii are $3$ and $1$
* aspect ratio is $3$
* condition number is $9$

Indeed,

$$\sqrt{\kappa(\Sigma)}=\sqrt{9}=3.$$

If instead

$$
\Lambda=
\begin{pmatrix}
100 & 0\\
0 & 0.01
\end{pmatrix},
$$

then:

* radii are $10$ and $0.1$
* aspect ratio is $100$
* condition number is $10000$

That is a very elongated, nearly degenerate ellipse.

---

## Bottom line

The elongation of the 2D Gaussian ellipse is controlled by the ratio of covariance eigenvalues:

$$\text{elongation} = \sqrt{\frac{\lambda_{\max}}{\lambda_{\min}}}.$$

And since

$$\kappa(\Sigma)=\frac{\lambda_{\max}}{\lambda_{\min}},$$

the condition number is the **square of the ellipse aspect ratio**.

So a large condition number means the covariance is close to singular and the Gaussian is highly elongated.

I can also show this from the equation $x^\top \Sigma^{-1}x=1$ step by step if you want.

## Diagonalization vs. Spectral Decomposition vs. Eigendecomposition

They are related, but not identical.

For a square matrix $A$:

### Diagonalization

This means writing

$$A = P D P^{-1}$$

where $D$ is diagonal.

What it says:

* (A) has a basis of eigenvectors.
* The columns of (P) are those eigenvectors.
* The diagonal entries of (D) are the corresponding eigenvalues.

A matrix is diagonalizable **iff** it has enough linearly independent eigenvectors.

---

### Eigendecomposition

This is usually just the same idea, written in eigenvalue language:
[
A = V \Lambda V^{-1}
]
where

* (V) contains eigenvectors,
* (\Lambda) is diagonal with eigenvalues.

So in most linear algebra contexts:

**eigendecomposition = diagonalization using eigenvectors**

It is basically the same concept, just a more descriptive name.

---

### Spectral decomposition

This is a more specific and more structured form, usually used when (A) is a matrix/operator covered by the spectral theorem, especially:

* real symmetric matrices,
* complex Hermitian matrices,
* more generally normal matrices.

Then one can write
[
A = Q \Lambda Q^*
]
with (Q) unitary/orthogonal, so the eigenvectors are orthonormal.

For a real symmetric or complex Hermitian matrix, spectral decomposition is often written as
[
A = \sum_{i=1}^n \lambda_i u_i u_i^*
]
or, grouping repeated eigenvalues,
[
A = \sum_{\lambda} \lambda P_\lambda
]
where (P_\lambda) is the orthogonal projector onto the eigenspace for (\lambda).

What is special here:

* eigenvectors can be chosen **orthonormally**,
* the decomposition uses **orthogonal projections**,
* it usually relies on the **spectral theorem**.

---

## Main difference in one sentence

* **Diagonalization**: “Can I turn this matrix into a diagonal one by a change of basis?”
* **Eigendecomposition**: “Write the diagonalization explicitly in terms of eigenvalues and eigenvectors.”
* **Spectral decomposition**: “A special orthonormal/projector-based eigendecomposition, usually for symmetric/Hermitian/normal matrices.”

---

## Relationship

Every spectral decomposition is an eigendecomposition, and every eigendecomposition is a diagonalization.

But not every diagonalization is called spectral decomposition.

So:
[
\text{spectral decomposition} \subset \text{eigendecomposition} \approx \text{diagonalization}
]

---

## Example 1: diagonalizable but not spectral in the symmetric sense

[
A=\begin{pmatrix}1&1\0&2\end{pmatrix}
]
This matrix has two distinct eigenvalues, so it is diagonalizable:
[
A = P D P^{-1}
]
But its eigenvectors are not orthogonal, and (A) is not symmetric/Hermitian, so this is not usually called a spectral decomposition in the projector/orthonormal sense.

---

## Example 2: spectral decomposition

[
A=\begin{pmatrix}2&1\1&2\end{pmatrix}
]
This is real symmetric, so it has an orthonormal eigenbasis. Hence
[
A = Q \Lambda Q^T
]
and also
[
A = \lambda_1 u_1 u_1^T + \lambda_2 u_2 u_2^T
]
This is a spectral decomposition.

---

## Important caveat

Some books use the terms a little loosely:

* some use **spectral decomposition** to mean any eigendecomposition,
* others reserve it strictly for the spectral-theorem case.

So the safest interpretation is:

* **diagonalization / eigendecomposition**: general diagonalizable matrix,
* **spectral decomposition**: usually the orthonormal decomposition from the spectral theorem.

If you want, I can also give you a small table comparing the three notions by assumptions, formula, and when they exist.

---

### Diagonalization == Eigendecomposition?

Yes — **conceptually, diagonalization and eigendecomposition are the same thing**.

A matrix $A$ is diagonalizable **iff** there exists a basis of eigenvectors.
So if

$$A = P D P^{-1}$$

then the columns of $P$ must be eigenvectors of $A$.

Why? Because from

$$AP = PD$$

if $p_i$ is the $i$-th column of $P$ and $d_i$ is the corresponding diagonal entry of $D$, then

$$A p_i = d_i p_i$$

So each column of $P$ is an eigenvector.

So in that strict mathematical sense:

* **diagonalization** = existence of a basis of eigenvectors,
* **eigendecomposition** = writing that diagonalization explicitly with eigenvectors.

They are not merely similar; they are essentially the same statement.

### Can you “diagonalize without using eigenvectors”?

There are two senses here.

#### 1. In theory: no

If you truly diagonalize a matrix by similarity, eigenvectors are unavoidable. Even if you do not mention them, they are built into the change-of-basis matrix $P$.

#### 2. In computation or derivation: sometimes yes, implicitly

You might reach a diagonal form without first solving “find all eigenvectors” in the usual classroom way. For example, you could:

* use invariant subspaces,
* use minimal polynomial information,
* use spectral theorem arguments,
* use Schur decomposition first and then conclude the matrix is diagonal.

But if you end up with

$$A=PDP^{-1}$$

then $P$ still consists of eigenvectors, whether you explicitly computed them that way or not.

So you can avoid **talking about** eigenvectors first, but you cannot avoid them **existing inside the result**.

### Important contrast: row reduction is not diagonalization

Sometimes people think:
“Can I just use row operations to make a matrix diagonal?”

Usually no, because row operations produce a row-equivalent matrix, not a **similar** matrix. Diagonalization is about similarity:

$$B = P^{-1} A P$$

Row reduction does not preserve eigenvalues in the needed way.

### Spectral decomposition is slightly more specific

For symmetric/Hermitian matrices, the change-of-basis matrix can be chosen orthogonal/unitary, so

$$A = Q \Lambda Q^\ast$$

That is still an eigendecomposition, but with the extra property that the eigenvectors are orthonormal.

So your intuition is right:

* **Diagonalization and eigendecomposition are truly identical in substance.**
* **Spectral decomposition is a special nicer version** when the eigenvectors can be chosen orthonormally.

A very compact summary is:

$$A \text{ is diagonalizable } \iff A \text{ has a basis of eigenvectors.}$$

So there is no diagonalization without eigenvectors in the actual mathematical sense.

---

### Only symmetric matrices can have orthogonal eigenbasis? 

Not in full generality.

The clean answer is:

* **Over $\mathbb R$:** a **real matrix has an orthonormal eigenbasis over $\mathbb R$** **iff** it is **symmetric**.
* **Over $\mathbb C$:** the right class is larger: a matrix has an **orthonormal eigenbasis** **iff** it is **normal** $(A^\ast A=AA^\ast)$. Hermitian matrices are only one special case.

### Real case

If a real matrix $A$ has an orthonormal eigenbasis, then

$$A = Q D Q^\top$$

with $Q$ orthogonal and $D$ diagonal. Then

$$A^\top = (QDQ^\top)^\top = QD^\top Q^\top = QDQ^\top = A$$

since $D$ is diagonal. So $A$ must be symmetric.

And conversely, every real symmetric matrix has an orthonormal eigenbasis.

So for **real matrices with real eigenvectors**, the answer is:

**yes, exactly symmetric matrices.**

---

### Complex case

For complex matrices, “having an orthonormal eigenbasis” does **not** force the matrix to be Hermitian/symmetric. It only forces it to be **normal**.

Example:

$$A=\begin{pmatrix}1&0 \\ 0&i \end{pmatrix}$$

Its standard basis vectors are orthonormal eigenvectors, but $A$ is **not Hermitian** because $A^\ast \neq A$.

So:

* **Hermitian** $\Rightarrow$ orthonormal eigenbasis
* **Normal** $\Rightarrow$ orthonormal eigenbasis
* but **not only Hermitian/symmetric**

---

### Important wording point

There are two slightly different phrases:

* **orthogonal eigenvectors**: just some eigenvectors happen to be orthogonal
* **orthogonal eigenbasis**: a full basis of eigenvectors that are mutually orthogonal

A non-symmetric matrix might have **some** orthogonal eigenvectors, but having a **full orthogonal/orthonormal eigenbasis** is much stronger.

### Best summary

* For **real matrices**:
  
  $$\text{orthonormal eigenbasis} \iff \text{symmetric}$$
  
* For **complex matrices**:
  
  $$\text{orthonormal eigenbasis} \iff \text{normal}$$
  

If you want, I can also explain why **distinct eigenvalues of a symmetric matrix automatically give orthogonal eigenvectors**.

## Covariance of Sum and Average

For the sample average

$$\bar X=\frac1n\sum_{i=1}^n X_i,$$

the covariance shrinks by a factor of $n$ when the $X_i$ are i.i.d. with covariance $\Sigma$.

So:

$$\boxed{\operatorname{Cov}(\bar X)=\frac{\Sigma}{n}}.$$

## Why?

Use the rule

$$\operatorname{Cov}(aY)=a^2\operatorname{Cov}(Y)$$

for scalars $a$, and

$$\operatorname{Cov}!\left(\sum_i Y_i\right) = \sum_i \operatorname{Cov}(Y_i) + \sum_{i\neq j}\operatorname{Cov}(Y_i,Y_j)$$


Now if $X_1,\dots,X_n$ are independent, then for $i\neq j$,

$$\operatorname{Cov}(X_i,X_j)=0$$

Hence
[
\operatorname{Cov}(\bar X)
==========================

# \operatorname{Cov}!\left(\frac1n\sum_{i=1}^n X_i\right)

# \frac1{n^2}\operatorname{Cov}!\left(\sum_{i=1}^n X_i\right)

\frac1{n^2}\sum_{i=1}^n \operatorname{Cov}(X_i).
]

Since each $X_i$ has covariance $\Sigma$,
[
\operatorname{Cov}(\bar X)
==========================

# \frac1{n^2}\cdot n\Sigma

\frac{\Sigma}{n}.
]

## Scalar case

If $X_i$ are one-dimensional with variance $\sigma^2$, then this becomes

$$\operatorname{Var}(\bar X)=\frac{\sigma^2}{n}.$$

## In your earlier derivation

That is why
[
\mathbb E!\left[n(\bar X-\mu)(\bar X-\mu)^T\right]
==================================================

# n,\operatorname{Cov}(\bar X)

# n\cdot \frac{\Sigma}{n}

\Sigma.
]

If you want, I can also show how this changes when the $X_i$ are **not independent**.

## Qualitative Change in Dynamical Systems

In dynamical systems, a **qualitative change** means the system’s **overall behavior pattern** changes, not just the exact numerical values.

For example, a small parameter change might cause the system to go from:

* settling to a stable equilibrium,
* to oscillating in a periodic cycle,
* to having multiple equilibria,
* or to becoming unstable.

That is a qualitative change because the **type of motion** changes.

A standard example is a **bifurcation**: as a parameter passes a critical value, the structure of the system changes in a meaningful way. For instance, an equilibrium can switch from stable to unstable, or new equilibria can appear.

What **“qualitative”** means here:

It refers to properties like:

* how many equilibria there are,
* whether they are stable or unstable,
* whether trajectories converge, diverge, or oscillate,
* whether there are periodic orbits,
* how phase portraits are organized.

So “qualitative” means **structural / behavioral / geometric**, rather than **exactly quantitative**.

A good contrast is:

* **Quantitative change**: an equilibrium moves from $x=1.2$ to $x=1.3$.
* **Qualitative change**: the equilibrium changes from stable to unstable, or disappears entirely.

So in this context, “qualitative” stands for:

**the kind of behavior the system has**, rather than the precise numbers describing it.

Example:
Consider

$$\dot x = r - x^2$$


* If (r>0), there are two equilibria.
* If (r=0), there is one degenerate equilibrium.
* If (r<0), there are no real equilibria.

This is a qualitative change as (r) crosses (0), because the phase portrait changes in structure.

If you want, I can also explain this in a more intuitive way with phase portraits and bifurcations.

## Why are bifurcations usually considered to be harmful in application of dynamical systems?

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Are bifurcations harmful?)</span></p>

They are not inherently harmful. They are often *treated* as harmful in applications because a bifurcation means a **small parameter change can cause a big qualitative change in behavior**.

In pure math, that is interesting. In engineering or applied settings, it is often a problem because people usually want the system to behave **smoothly, predictably, and robustly**.

The main reasons are:

1. **Loss of stability**
   A stable equilibrium or periodic motion can disappear or become unstable at a bifurcation. Then the system may jump to oscillations, multiple steady states, or runaway behavior.

2. **High sensitivity near the bifurcation point**
   Close to a bifurcation, tiny changes in parameters, noise, delays, or modeling error can change the outcome a lot. That makes design and control harder.

3. **Sudden regime changes**
   The system can switch from “works fine” to “works badly” without much warning. In applications, abrupt transitions are often undesirable.

4. **Reduced predictability**
   After a bifurcation, there may be several possible long-term behaviors, or even chaotic motion. Then forecasting and control become much more difficult.

5. **Safety and performance concerns**
   In real systems, a bifurcation may correspond to flutter in aircraft, voltage collapse in power systems, unwanted oscillations in control loops, population collapse in ecology, or arrhythmias in physiology.

So the issue is less “bifurcations are bad” and more:

**applications often value robustness, while bifurcations mark loss of robustness.**

That said, bifurcations are sometimes useful or even intentionally exploited. For example, switching devices, pattern formation, lasers, neural models, and some sensing mechanisms rely on operating near a bifurcation because the system becomes very responsive.

But sometimes they are useful:
* Saddle-node behavior can be used for switching and memory.
* Hopf bifurcations can be used to model or generate rhythms, clocks, lasers, and neural oscillators.

A good one-line summary is:

**Bifurcations are usually considered harmful when the goal is reliable operation, because they are thresholds where qualitative behavior can change dramatically under small perturbations.**

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span>(Saddle-node bifurcation application)</p>

A simple model is

$$\dot x = \mu - x^2$$

where $\mu$ is a parameter.

What happens:

* If $\mu > 0$, there are two equilibria:
  
  $$x = \pm \sqrt{\mu}$$
  
  The positive one is stable, the negative one is unstable.
* If $\mu = 0$, the two equilibria merge at $x=0$.
* If $\mu < 0$, there is **no equilibrium at all**.

Why this is harmful in applications:

Imagine $x$ is the operating state of a device and $\mu$ is a control parameter.

* For $\mu>0$, the system has a stable resting state, so it behaves normally.
* As $\mu$ decreases toward $0$, that stable state moves and becomes weaker.
* Once $\mu$ crosses $0$, the stable equilibrium **disappears**.

So the system can no longer “settle down.” It may drift away, hit physical limits, or jump to some other regime.

**Application intuition:**
This is the mathematical picture behind things like voltage collapse, mechanical snap-through, or sudden failure of an operating point. A tiny parameter change can destroy the state you were relying on.

The harmful part is not the bifurcation itself, but that **the desired stable state vanishes**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span>(Hopf bifurcation application)</p>

A standard normal form is

$$
\dot x = \mu x - y - x(x^2+y^2), \qquad
\dot y = x + \mu y - y(x^2+y^2)
$$

What happens:

* If $\mu < 0$, the equilibrium at $(0,0)$ is stable.
* At $\mu = 0$, stability changes.
* If $\mu > 0$, the equilibrium becomes unstable and a **stable limit cycle** appears.

So the system changes from “settling to a steady state” to “sustained oscillation.”

Why this is harmful in applications:

Suppose $(0,0)$ is the desired operating condition.

* Before the bifurcation, disturbances die out.
* After the bifurcation, disturbances do not die out; instead the system keeps oscillating.

**Application intuition:**
That can mean unwanted vibration in a machine, chatter in a control loop, oscillating chemical concentrations in a reactor, or rhythmic instability in physiology.

So the issue is that the system goes from:

* **stable steady operation**
  to
* **self-excited oscillation**

Even if the oscillation is bounded, it may still be bad because it reduces accuracy, increases wear, creates noise, or threatens safety.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The contrast)</span></p>

**For a saddle-node bifurcation:**
* the problem is often **loss of equilibrium**

**For a Hopf bifurcation:**
* the problem is often **loss of steady behavior and onset of oscillations**

</div>

## Chaos Needs At Least 3D for Continuous-time Autonomous ODEs

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Chaos is impossible in 2D)</span></p>

In 2D continuous-time autonomous systems, the long-term behavior is too restricted.

The key point is this:

For a trajectory that stays in a bounded region of the plane, the **Poincaré–Bendixson theorem** says its long-term limit set can only be very simple:

* an equilibrium point,
* a closed orbit,
* or a set involving equilibria and trajectories connecting them.

What it **cannot** be is a complicated aperiodic invariant set of the kind needed for chaos.

Chaos needs a trajectory to do something like this forever:

* remain bounded,
* never settle to an equilibrium,
* never settle to a periodic orbit,
* and still keep wandering in a complicated sensitive way.

But Poincaré–Bendixson says that in the plane, once you rule out equilibria in the limit region, the trajectory must approach a **periodic orbit**. So there is no room for a strange attractor.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Chaos is impossible in 1D)</span></p>

**In 1D it is even more impossible**

For a 1D autonomous ODE

$$\dot x = f(x)$$

solutions cannot cross each other. The motion is just left, right, or toward an equilibrium. There are not even periodic orbits.

So:

* **1D:** no chaos
* **2D:** no chaos because of Poincaré–Bendixson
* **3D and higher:** chaos becomes possible

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why this blocks chaos in 2D)</span></p>

A chaotic attractor is not:

* a single fixed point,
* and not a single periodic orbit.

It is a much more complicated **invariant set** with recurrent but nonperiodic motion.

The theorem says that kind of set cannot arise as the omega-limit set of a bounded trajectory in a smooth 2D autonomous ODE. So genuine chaos is impossible there.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why 3D changes things)</span></p>

In 3 dimensions, trajectories can wind around each other without being forced into a fixed point or closed orbit. That extra degree of freedom allows complicated invariant sets such as strange attractors.

That is why systems like the Lorenz system can be chaotic, while planar autonomous systems cannot.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(One important caveat)</span></p>

This statement is about **continuous-time autonomous ODEs**.

It does **not** say:

* 2D discrete maps cannot be chaotic — they can
* forced 2D systems cannot be chaotic — periodic forcing effectively adds another variable, making the autonomous version 3D

So the clean statement is:

> Smooth autonomous continuous-time systems need at least 3 dimensions for chaos, because in 2 dimensions the Poincaré–Bendixson theorem forces bounded long-term behavior to be too simple.

If you want, I can also explain this using the idea of an **omega-limit set**, which is the most precise way to see it.

</div>

## Strange Attractor vs. Chaotic Attractor

* https://en.wikipedia.org/wiki/Strange_nonchaotic_attractor
* https://www.quantamagazine.org/variable-stars-have-strange-nonchaotic-attractors-20150310/
* KIC 5520878

They are related, but not exactly the same.

A **chaotic attractor** is an attractor whose dynamics are **chaotic**, meaning roughly:

* trajectories stay in a bounded attracting set,
* they are aperiodic,
* and they show sensitive dependence on initial conditions.

A **strange attractor** is an attractor with a **complicated geometric structure**, typically fractal or non-smooth.

So the distinction is:

* **chaotic** describes the **time behavior**
* **strange** describes the **shape/geometry**

Often the same attractor is both, which is why the terms are sometimes used almost interchangeably.

### Typical situation

For famous examples like the Lorenz attractor:

* it is **chaotic** because motion on it is sensitive and nonperiodic
* it is **strange** because its set has a fractal-type geometry

So it is both a **chaotic attractor** and a **strange attractor**.

### Can one exist without the other?

In principle, yes, at least conceptually:

* an attractor could be **chaotic** without emphasizing fractal geometry
* an attractor could be called **strange** because of complicated geometry, even if the chaos aspect is not the point

But in many dynamical systems courses, “strange attractor” usually means an attractor that is also chaotic.

### Good short definitions

You can remember it like this:

> A **chaotic attractor** is defined by chaotic dynamics.
> A **strange attractor** is defined by strange geometry.

Or even shorter:

> **chaotic = behavior**,
> **strange = shape**.

### One subtlety

Different books are not always perfectly strict with the terminology. Some use “strange attractor” almost as a synonym for “chaotic attractor,” especially in physics-style texts.

So if your course is informal, the instructor may treat them as basically the same thing. But mathematically, the clean distinction is:

* **chaotic attractor** → focus on dynamics
* **strange attractor** → focus on geometry

If you want, I can also give you a tiny table comparing:
fixed point attractor, limit cycle attractor, torus attractor, and strange/chaotic attractor.

## Ghost Attractor

The **ghost region** is the place where the two fixed points **used to be** just before they collided and disappeared.

For this system

$$\dot x=\mu-x^2,\qquad \dot y=-y,$$

when $\mu>0$, there are two equilibria at $x=\pm \sqrt{\mu}$, $y=0$.

When $\mu=0$, they collide.

When $\mu<0$, there are **no fixed points anymore**. But if (\mu<0) is only slightly negative, the vector field is still very small near the old collision point. In particular,

$$\dot x=\mu-x^2=-(\lvert \mu\rvert+x^2),$$

so near $x=0$,

$$\dot x \approx -\lvert \mu\rvert$$

which is small if $\lvert\mu\rvert$ is small. That means trajectories move **very slowly** through that neighborhood.

So the "ghost" is:

* **not** a real fixed point,
* but a **bottleneck / slow region** in the flow,
* caused by the lingering influence of the saddle and node that have just annihilated each other.

In the phase portrait for $\mu<0$, trajectories get pulled toward $y=0$ because $\dot y=-y$, then they drift slowly through the dotted region before finally moving away. It looks as though an equilibrium is still “almost there,” hence the name **ghost**.

A good intuition is:

* before bifurcation: two equilibria actually trap/organize trajectories,
* after bifurcation: the equilibria are gone, but the flow still remembers them for a while.

That is why the time spent in the ghost region becomes very large near the bifurcation.

## Hysteresis

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Hysteresis)</span></p>

**Hysteresis** is the dependence of the state of a system on its history. System’s output depends not only on its current input, but also on its past.

So the system has a kind of **memory**.

</div>

<div class="gd-grid">
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/Irreversible_Hysteresis.png' | relative_url }}" alt="Gradient descent stuck in local minima" loading="lazy">
    <figcaption>Irreversible hysteresis graph</figcaption>
  </figure>
  <figure>
    <img src="{{ '/assets/images/notes/model-based-time-series-analysis/Reversible_Hysteresis.png' | relative_url }}" alt="Gradient descent slow convergence" loading="lazy">
    <figcaption>Reversible hysteresis graph</figcaption>
  </figure>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(History and memory)</span></p>

In systems with bistability, the same input level can correspond to two distinct stable states (e.g., "low output" and "high output"). The actual state of the system depends on its history –whether the input level was increasing (forward trajectory) or decreasing (backward trajectory). Thus, it is difficult to determine which state a cell is in if given only a bistability curve. The cell's ability to "remember" its prior state ensures stability and prevents it from switching states unnecessarily due to minor fluctuations in input.

This memory is often maintained through molecular feedback loops, such as positive feedback in signaling pathways

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(History and memory)</span></p>

This memory is often maintained through molecular feedback loops, such as positive feedback in signaling pathways, or the persistence of regulatory molecules like proteins or phosphorylated components. For example, the refractory period in action potentials is primarily controlled by history. Absolute refraction period prevents a volted-gated sodium channel from activating or refiring after it has just fired. This is because following the absolute refractory period, the neuron is less excitable due to hyperpolarization caused by potassium efflux. This molecular inhibitory feedback creates a memory for the neuron or cell, so that the neuron does not fire too soon. As time passes, the neuron or cell will slowly lose the memory of having fired and will begin to fire again. Thus, memory is time-dependent, which is important in maintaining homeostasis and regulating many different biological processes.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Hysteretic Current-Voltage Curve)</span></p>

Suppose $\alpha$ is small and $I$ is initially below the homoclinic bifurcation (thick line in Figure). Then the junction will be operating at the stable fixed point, corresponding to the zero-voltage state. As $I$ is increased, nothing changes until $I$ exceeds $1$. Then the stable fixed point disappears in a saddle-node bifurcation, and the junction jumps into a nonzero voltage state (the limit cycle).

If $I$ is brought back down, the limit cycle persists below $I=1$ but its frequency tends to zero continuously as $I_c$ is approached. Specifically, the frequency tends to zero like $[\ln( I I_c )]^{–1}$, just as expected from the scaling law. The voltage also returns to zero continuously as $I\to I_c$.

<figure>
  <img src="{{ '/assets/images/notes/model-based-time-series-analysis/HystereticCurrentVoltageCurve.png' | relative_url }}" alt="Gradient descent slow convergence" loading="lazy">
</figure>

In practice, the voltage appears to jump discontinuously back to zero, but that is to be expected because $[\ln( I- I_c )]^{–1}$ has **infinite derivatives of all orders at $I_c$!** The steepness of the curve makes it impossible to resolve the
continuous return to zero.

</div>

Here is a clean summary you can put into your notes.

## Degenerate equilibrium — summary

Consider a dynamical system

[
\dot{x} = f(x), \qquad x \in \mathbb{R}^n,
]

and let (x_*) be an equilibrium, so

[
f(x_*) = 0.
]

### 1. Linearization

Near (x_*), the vector field is approximated by

[
f(x_*+\delta) = f(x_*) + Df(x_*)\delta + o(|\delta|).
]

Since (f(x_*)=0) at an equilibrium, the local dynamics is approximated by

[
\dot{\delta} = Df(x_*)\delta.
]

So the Jacobian (Df(x_*)) gives the first-order local behavior near the equilibrium.

---

### 2. What “degenerate” means

An equilibrium is called **degenerate** if the Jacobian at that point is singular:

[
\det Df(x_*) = 0.
]

Equivalently: **(0) is an eigenvalue** of the linearization.

This means that in at least one direction, the linear term gives no information about attraction or repulsion.

---

### 3. Intuition

Degeneracy means the equilibrium is **flat to first order**.

It does **not** mean there is no motion in that direction in the true nonlinear system.
It only means there is no motion visible in the **linear approximation** in that direction.

So higher-order terms become important.

A good slogan is:

> Zero eigenvalue means “flat to first order,” not “frozen.”

---

### 4. Why linearization may fail

In nondegenerate cases, the linearization often captures the local phase portrait well.

In degenerate cases, it may be inconclusive, because the missing first-order effect can be replaced by quadratic or higher-order terms.

Example:

[
\dot{x} = x^2.
]

At (x=0),

[
f(0)=0, \qquad f'(0)=0,
]

so (x=0) is degenerate.

But the linearization

[
\dot{x}=0
]

does not describe the true behavior:

* if (x>0), then (\dot{x}>0),
* if (x<0), then (\dot{x}>0).

So the nonlinear term determines the dynamics.

---

### 5. Degenerate does not always mean manifold attractor

A zero eigenvalue can happen for different reasons.

#### Case A: line/manifold of equilibria

Example:

[
\dot{x}=0, \qquad \dot{y}=-y.
]

Every point ((x,0)) is an equilibrium, so the (x)-axis is a manifold of equilibria.
Here the zero eigenvalue corresponds to genuine neutral motion along that equilibrium manifold.

#### Case B: isolated equilibrium with zero eigenvalue

Example:

[
\dot{x}=x^2, \qquad \dot{y}=-y.
]

At ((0,0)), one eigenvalue is zero, but there is no line of equilibria.
The only equilibrium is ((0,0)), and the motion in the (x)-direction is determined by the nonlinear term (x^2).

So:

> Degeneracy does not automatically mean line attractor or manifold attractor.

---

### 6. Relation to non-hyperbolicity

An equilibrium is **hyperbolic** if no eigenvalue of $Df(x^\ast)$ has real part zero.

It is **non-hyperbolic** if at least one eigenvalue has real part zero.

A degenerate equilibrium is a special case of non-hyperbolicity where $0$ itself is an eigenvalue.

So:

$$\text{degenerate} \implies \text{non-hyperbolic}.$$

But not conversely.

Example of non-hyperbolic but not degenerate:

$$A=\begin{pmatrix}0 & -1\\ 1 & 0\end{pmatrix}$$

has eigenvalues $\pm i$, so it is non-hyperbolic, but

$$\det A = 1 \neq 0,$$

so it is not degenerate.

---

### 7. In 2D Hamiltonian systems

For a planar Hamiltonian system

[
\dot{x} = H_y, \qquad \dot{y} = -H_x,
]

equilibria occur at critical points of (H):

[
\nabla H(x_*,y_*)=0.
]

The equilibrium is **nondegenerate** if the Hessian of (H) is invertible:

[
\det \nabla^2 H(x_*,y_*) \neq 0.
]

It is **degenerate** if

[
\det \nabla^2 H(x_*,y_*) = 0.
]

In the nondegenerate case:

* saddle point of (H) (\Rightarrow) saddle equilibrium,
* local minimum or maximum of (H) (\Rightarrow) center.

A local maximum also gives a center because Hamiltonian trajectories follow **level sets of (H)**, not gradient descent.

---

## Short final definition

A **degenerate equilibrium** is an equilibrium where the Jacobian has a zero eigenvalue, so the linearization loses information in at least one direction and higher-order nonlinear terms are needed to determine the true local dynamics.

If you want, I can turn this into a shorter “exam-style” version too.



## Computational Learning Theory

* https://en.wikipedia.org/wiki/Computational_learning_theory
* https://ai.stackexchange.com/questions/20355/what-are-some-resources-on-computational-learning-theory
* https://stats.stackexchange.com/questions/63077/statistical-learning-theory-vs-computational-learning-theory
* https://arxiv.org/pdf/2011.04483
* https://www.amazon.de/-/en/Understanding-Machine-Learning-Theory-Algorithms/dp/1107512824?utm_source=chatgpt.com
* https://www.amazon.de/-/en/Introduction-Computational-Learning-Theory-Press/dp/0262111934?utm_source=chatgpt.com
* https://www.amazon.de/-/en/Deep-Learning-Foundations-Taeho-Jo/dp/3031328787?utm_source=chatgpt.com
* https://www.amazon.de/-/en/Computational-Learning-Theory-Natural-Systems/dp/0262581264?utm_source=chatgpt.com
* https://www.amazon.de/-/en/Understanding-Machine-Learning-Theory-Algorithms/dp/1107057132?utm_source=chatgpt.com
* https://www.amazon.de/-/en/Mehryar-Mohri-ebook/dp/B08BT4MJ29?utm_source=chatgpt.com
* https://www.amazon.de/-/en/Principles-Deep-Learning-Theory-Understanding/dp/1316519333?utm_source=chatgpt.com
* https://www.jeffreyheinz.net/classes/22F/materials/Valiant2014-Chaps1-2.pdf?utm_source=chatgpt.com
* https://www.jeffreyheinz.net/classes/18S/materials/Kearns-Vairani-1994-Introduction-to-Computational-Learning-Thoery-Ch01.pdf?utm_source=chatgpt.com
* https://www.cs.cmu.edu/afs/cs/academic/class/10601-f10/lecture/lec10.pdf?utm_source=chatgpt.com
* https://www.cs.cmu.edu/~avrim/ML14/?utm_source=chatgpt.com
* https://www.cs.cmu.edu/~avrim/courses.html?utm_source=chatgpt.com
* https://openeclass.panteion.gr/modules/document/file.php/PMS152/LEARNING/Anthony%20Martin%2C%20Bartlett%20Peter%20L.%20%282009%29%20--%20Neural%20Network%20Learning_%20Theoretical%20Foundations.pdf?utm_source=chatgpt.com
* https://dblp.uni-trier.de/rec/books/daglib/0041035.html?utm_source=chatgpt.com
* https://api.pageplace.de/preview/DT0400.9780511822902_A23680035/preview-9780511822902_A23680035.pdf?utm_source=chatgpt.com
* https://deeplearningtheory.com/?utm_source=chatgpt.com
* https://tor-lattimore.com/downloads/book/book.pdf?utm_source=chatgpt.com
* https://www.cs.huji.ac.il/~shais/papers/OLsurvey.pdf?utm_source=chatgpt.com
* https://cesa-bianchi.di.unimi.it/predbook/?utm_source=chatgpt.com
* https://home.mathematik.uni-freiburg.de/maxwell/coursenotes-mlandml-website.pdf?utm_source=chatgpt.com
* https://home.ttic.edu/~avrim/MLT22/?utm_source=chatgpt.com
* https://www.cs.ox.ac.uk/people/varun.kanade/teaching/CLT-MT2024/lectures/CLT%20%28updated%29.pdf?utm_source=chatgpt.com
* https://people.mpi-inf.mpg.de/~mehlhorn/SeminarEvolvability/ValiantLearnable.pdf?utm_source=chatgpt.com
* https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf?utm_source=chatgpt.com
* https://cs.nyu.edu/~mohri/mlbook/?utm_source=chatgpt.com
* https://rl.uni-freiburg.de/teaching/ws20/deeplearning?utm_source=chatgpt.com
* https://www.cambridge.org/eg/universitypress/subjects/computer-science/pattern-recognition-and-machine-learning/understanding-machine-learning-theory-algorithms?format=HB&isbn=9781107057135&utm_source=chatgpt.com
* https://www.cambridge.org/de/universitypress/subjects/physics/statistical-physics/principles-deep-learning-theory-effective-theory-approach-understanding-neural-networks?format=HB&isbn=9781316519332
* https://www.cambridge.org/core/books/abs/understanding-machine-learning/online-learning/A43CEEF3F7F18953B592C983087C8A36?utm_source=chatgpt.com
* https://www.cambridge.org/core/books/prediction-learning-and-games/A05C9F6ABC752FAB8954C885D0065C8F?utm_source=chatgpt.com
* https://assets.cambridge.org/97811084/86828/frontmatter/9781108486828_frontmatter.pdf?utm_source=chatgpt.com
* https://cdn.aaai.org/AAAI/1990/AAAI90-163.pdf?utm_source=chatgpt.com
* https://dokumen.pub/mehryar-mohri-afshin-rostamizadeh-and-ameet-talwalkar-foundations-of-machine-learning-second-edition-the-mit-press-cambridge-ma-2018-504-pp-cdn-9653-hardback-isbn-9780262039406-2nbsped-9780262039406.html?utm_source=chatgpt.com
* https://dokumen.pub/theoretical-machine-learning-lecture-notes-princeton-cos511.html?utm_source=chatgpt.com
* https://dokumen.pub/understanding-machine-learning-from-theory-to-algorithms-1nbsped-9781107057135.html?utm_source=chatgpt.com
* https://dokumen.pub/neural-network-learning-theoretical-foundations.html?utm_source=chatgpt.com
* https://dokumen.pub/prediction-learning-and-games-9780521841085-9780511189951-0511189958-0521841089-9780511546921-0511546920.html?utm_source=chatgpt.com
* https://dokumen.pub/understanding-machine-learning-9781107057135-1107057132-w-5259393.html?utm_source=chatgpt.com
* https://dlib.hust.edu.vn/server/api/core/bitstreams/3293054a-2c5f-4c97-bb14-b7c891e16c90/content
* https://svivek.com/teaching/lectures/slides/colt/agnostic-learning.pdf?utm_source=chatgpt.com
* https://ece.iisc.ac.in/~aditya/E1245_Online_Prediction_Learning_F2018/lattimore-szepesvari18bandit-algorithms.pdf?utm_source=chatgpt.com
* https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/?utm_source=chatgpt.com
* https://www.cs.huji.ac.il/~shais/Advanced2011/Online.pdf?utm_source=chatgpt.com
* https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/copy.html?utm_source=chatgpt.com
* https://people.seas.harvard.edu/~valiant/PAC-%20Summary.pdf?utm_source=chatgpt.com
* https://api.pageplace.de/preview/DT0400.9780511189951_A23689971/preview-9780511189951_A23689971.pdf?utm_source=chatgpt.com
* https://api.pageplace.de/preview/DT0400.9781108687492_A49236430/preview-9781108687492_A49236430.pdf?utm_source=chatgpt.com
* https://github.com/yanshengjia/ml-road/blob/master/resources/Foundations%20of%20Machine%20Learning%20%282nd%20Edition%29.pdf?utm_source=chatgpt.com
* https://kyl.neocities.org/books/%5BTEC%20ROB%5D%20the%20principles%20of%20deep%20learning%20theory.pdf?utm_source=chatgpt.com
* https://www.stat.berkeley.edu/~bartlett/nnl/index.html?utm_source=chatgpt.com
* https://people.eecs.berkeley.edu/~nika/pubs/nhaghtal_cs_2018.pdf?utm_source=chatgpt.com
* https://simons.berkeley.edu/sites/default/files/docs/5318/slides.pdf?utm_source=chatgpt.com
* https://www.zvab.com/products/isbn/9780262111935?ref_=pd_offer-1-d_0&utm_source=chatgpt.com
* https://virtualmmx.ddns.net/gbooks/BanditAlgorithms.pdf?utm_source=chatgpt.com
* https://virtualmmx.ddns.net/gbooks/UnderstandingMachineLearning.pdf?utm_source=chatgpt.com
* https://d1b10bmlvqabco.cloudfront.net/attach/jl2b00mpen3au/jl2b0jnvzgn3hf/jntb4o0banvp/CMSC422PACandVC.pdf?utm_source=chatgpt.com
* https://www.ekzhang.com/assets/pdf/CS_228_Notes.pdf?utm_source=chatgpt.com
* https://books.google.de/books?hl=de&id=ttJkAwAAQBAJ&printsec=frontcover&source=gbs_atb&utm_source=chatgpt.com#v=onepage&q&f=false
* https://books.google.de/books?id=QVA8ugEACAAJ&utm_source=chatgpt.com
* https://arxiv.org/abs/2106.10165?utm_source=chatgpt.com
* https://arxiv.org/pdf/2511.08791
* https://www.semanticscholar.org/paper/Online-Learning-and-Online-Convex-Optimization-Shalev-Shwartz/bcce96a2a074448953fc61a29a84afbdfc8db55a?utm_source=chatgpt.com
* https://cseweb.ucsd.edu/classes/fa12/cse291-b/vcnotes.pdf?utm_source=chatgpt.com
* https://www.springerprofessional.de/en/deep-learning-foundations/25846382?utm_source=chatgpt.com
* https://books.google.at/books?hl=de&id=vCA01wY6iywC&printsec=copyright&utm_source=chatgpt.com#v=onepage&q&f=false
* https://www.scribd.com/document/907458671/Foundations-of-Machine-Learning-2nd-Edition-Mehryar?utm_source=chatgpt.com
* https://www.scribd.com/document/718886563/Bandit-Algorithms-Tor-Lattimore-Csaba-Szepesvari-Z-Library?utm_source=chatgpt.com
* https://www.scribd.com/document/461103850/lattimore-szepesvari18bandit-algorithms-pdf?utm_source=chatgpt.com
* https://mitpress.mit.edu/9780262039406/foundations-of-machine-learning/?utm_source=chatgpt.com
* https://direct.mit.edu/books/monograph/2604/chapter-abstract/70331/Index?redirectedFrom=fulltext
* https://www.cs.cornell.edu/jeh/book.pdf?utm_source=chatgpt.com
* https://pages.cs.wisc.edu/~dpage/cs760/colt.pdf?utm_source=chatgpt.com
* https://dl.acm.org/doi/10.5555/2371238?utm_source=chatgpt.com
* https://dl.acm.org/doi/10.5555/200548?utm_source=chatgpt.com
* https://dl.acm.org/doi/book/10.5555/1137817?utm_source=chatgpt.com
* https://www.emerald.com/ftmal/article-abstract/4/2/107/1332161/Online-Learning-and-Online-Convex-Optimization?redirectedFrom=fulltext&utm_source=chatgpt.com
* https://homepages.cwi.nl/~wmkoolen/MLT_2025/?utm_source=chatgpt.com
* https://www.medimops.de/deep-learning-foundations-paperback-M03031328817.html?utm_source=chatgpt.com
* https://www.ebay.co.uk/p/86720448
* https://www.hlevkin.com/hlevkin/45MachineDeepLearning/ML/Foundations_of_Machine_Learning.pdf?utm_source=chatgpt.com
* https://freecomputerbooks.com/Neural-Network-Learning-Theoretical-Foundations.html?utm_source=chatgpt.com
* https://www.d.umn.edu/~rmaclin/cs8751/fall2005/Notes/L17_Computational_Learning_Theory.pdf?utm_source=chatgpt.com
* https://www.denizalpgoktas.com/pubs/notes/Online_Learning_and_Online_Convex_Optimization.pdf?utm_source=chatgpt.com
* https://constantinides.net/2024/07/16/notes-on-computational-learning-theory/?utm_source=chatgpt.com
* https://www.tandfonline.com/doi/full/10.1080/14697688.2015.1080489?utm_source=chatgpt.com
* https://www.mimuw.edu.pl/~fopss18/PDF/Worrell-lecture-notes.pdf?utm_source=chatgpt.com
* https://home.ttic.edu/~avrim/MLT22/05-04-Hardness.pdf?utm_source=chatgpt.com
* https://www.di.ens.fr/appstat/fall-2018/notes/cours_online_learning.pdf?utm_source=chatgpt.com
* https://bibbase.org/network/publication/lattimore-szepesvri-banditalgorithms-2020?utm_source=chatgpt.com
* https://www.cs.ox.ac.uk/people/varun.kanade/teaching/CLT-MT2018/lectures/lecture03.pdf?utm_source=chatgpt.com
* https://www.schaumburg-buch.de/shop/item/9789811682339/foundations-of-deep-learning-von-fengxiang-tao-he-e-book-pdf?utm_source=chatgpt.com
* https://openurl.ebsco.com/EPDB%3Asrh%3A14%3A54742190/detailv2?id=ebsco%3Adoi%3A10.1561%2F2200000018&sid=ebsco%3Aplink%3Acrawler&utm_source=chatgpt.com&crl=f&link_origin=none
* https://www.perlego.com/book/4222798/bandit-algorithms-pdf?utm_source=chatgpt.com
* https://www.exlibris.ch/de/buecher-buch/englische-ebooks/fengxiang-he/foundations-of-deep-learning/id/9789811682339/?srsltid=AfmBOoq-bFjdXIgI2Tk6dQrG5GhcnMHnEwJQbgun9eiFMghEt1sUEo-1&utm_source=chatgpt.com
* https://www.cs.ubc.ca/~nickhar/W14/?utm_source=chatgpt.com
* https://nakulgopalan.github.io/cs4641/course/17-computational-learning-theory-annotated.pdf?utm_source=chatgpt.com
* https://www.overdrive.com/media/103351/prediction-learning-and-games?utm_source=chatgpt.com
* https://www.ou.edu/content/dam/coe/cs/course-syllabi/fall-2025/CS_4713-5713_Diochnos_Fa25.pdf?utm_source=chatgpt.com
* https://www.betterworldbooks.com/product/detail/understanding-machine-learning-from-theory-to-algorithms-9781107057135
* https://www.bibsonomy.org/pow-challenge/?return=https%3A%2F%2Fwww.bibsonomy.org%2Fbibtex%2Fe5571f0a85ff77ab750e5eecc6c45106%3Futm_source%3Dchatgpt.com
* https://www.bibsonomy.org/pow-challenge/?return=https%3A%2F%2Fwww.bibsonomy.org%2Fbibtex%2F22aaa93082f1d8d95409e4a2383447295%2Fsb3000%3Futm_source%3Dchatgpt.com
* https://scispace.com/papers/an-introduction-to-computational-learning-theory-59rvusb1xt?utm_source=chatgpt.com
* https://www.jennwv.com/courses/F11.html?utm_source=chatgpt.com
* https://portal.mardi4nfdi.de/wiki/Publication%3A3166527?utm_source=chatgpt.com
* https://www.mdpi.com/2227-7390/13/3/451?utm_source=chatgpt.com
* https://cs.uwaterloo.ca/~klarson/teaching/W15-486/lectures/22Colt.pdf?utm_source=chatgpt.com
* https://cdn.bookey.app/files/pdf/book/en/probably-approximately-correct.pdf?utm_source=chatgpt.com
* https://archive.org/details/neuralnetworklea0000anth?utm_source=chatgpt.com
* https://books.google.de/books/about/Probably_Approximately_Correct.html?id=NqIWBQAAQBAJ&redir_esc=y
* https://books.google.de/books?id=SiSwzgEACAAJ&printsec=frontcover&redir_esc=y#v=onepage&q&f=false
* https://chatgpt.com/share/69a15b52-5108-8011-945a-1a510e913ecb





































## Inconsistency

https://share.google/aimode/wtrHG1fk9Ac8kNTqv
https://share.google/aimode/JCBRv9GCiTnyBebfQ
https://share.google/aimode/d6k7KB7zcbETX09Oj

https://stats.stackexchange.com/questions/236774/inconsistency-in-unit-on-gradient-descent-equation
https://www.reddit.com/r/learnmachinelearning/comments/1cc0u9a/adam_dimensional_analysis_seems_inconsistent/
http://www.machinedlearnings.com/2011/06/dimensional-analysis-and-gradient.html
https://www.quora.com/What-is-the-intuition-behind-updating-the-weights-in-a-gradient-descent-algorithm-by-the-gradient-of-the-error-Why-not-update-by-some-other-quantity
https://timvieira.github.io/blog/post/2016/05/27/dimensional-analysis-of-gradient-ascent/
https://math.stackexchange.com/questions/223252/why-is-gradient-the-direction-of-steepest-ascent
https://www.reddit.com/r/MachineLearning/comments/9sfv8x/d_a_note_on_why_gradient_descent_is_even_needed/
https://www.reddit.com/r/MachineLearning/comments/48u7aw/is_there_any_good_theory_on_why_gradient_descent/

## Chain rule in general settings (Fréchet, Hilbert, manifolds)

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Starting thinking about the derivative as a linear map)</span></p>

The 1D chain rule, the multivariable chain rule, and the various chain rules one meets in PDE and geometry are all the same statement, written for different spaces. The unifying picture stops thinking of "the derivative of $f$ at $x$" as a number (or a row of numbers), and starts thinking of it as a **linear map** between the right vector spaces. Once this is in place, the chain rule has a one-line statement that does not change as the spaces become richer.

</div>

### Why a derivative is "really" a linear map

In one variable we conflate two ideas because they coincide:

* The number $f'(x) \in \mathbb R$ — "the slope."
* The linear map $h \mapsto f'(x)\,h$ from $\mathbb R$ to $\mathbb R$ — "the best linear approximation of the increment $f(x+h)-f(x)$."

The first is a *representation* of the second by a single number, available because $\mathrm{Hom}(\mathbb R,\mathbb R)\cong\mathbb R$. In higher (or infinite) dimensions, no single number can capture what a derivative does — but the linear-map picture continues to make sense, and is in fact the *intrinsic* object.

### Fréchet differentiability

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Fréchet derivative)</span></p>

Let $V,W$ be normed vector spaces, $U\subseteq V$ open, and $f:U\to W$. We say $f$ is **Fréchet-differentiable at $x\in U$** if there exists a *bounded linear map* $Df(x):V\to W$ such that

$$
\lim_{h\to 0}\frac{\bigl\|f(x+h)-f(x)-Df(x)\,h\bigr\|_W}{\|h\|_V} = 0.
$$

The map $Df(x)$ is called the **(Fréchet) differential** of $f$ at $x$. Equivalent names you will encounter: $df(x)$, $f'(x)$, the *total derivative*.

</div>

The displayed limit just says: $f(x+h)$ is well-approximated by the affine map $h\mapsto f(x)+Df(x)\,h$, with error of higher order than $\|h\|$. The "boundedness" of $Df(x)$ — i.e. continuity as a linear map — is automatic when $V$ is finite-dimensional, but a real condition in infinite dimensions.

### Why this looks like a Taylor expansion

The numerator $f(x+h)-f(x)-Df(x)\,h$ is *exactly* the first-order Taylor remainder. The resemblance is not a coincidence: the Fréchet definition is the first-order Taylor expansion taken as the **definition** of differentiability.

It pays to separate two distinct statements:

* **First-order Taylor.** $f(x+h)=f(x)+L\,h+o(\|h\|)$, with $L$ a linear map.
* **Higher-order Taylor's theorem.** $f(x+h)=f(x)+Df(x)h+\tfrac12 D^2 f(x)(h,h)+\dots+\tfrac{1}{n!}D^nf(x)(h,\dots,h)+R_n(h)$, with $R_n(h)/\|h\|^n\to 0$ (or with a Lagrange/integral remainder if you want a closed form).

The relationship between the Fréchet definition and Taylor expansion is then:

$$
\boxed{\;\text{Fréchet differentiability at }x\;\Longleftrightarrow\;\text{first-order Taylor expansion exists at }x.\;}
$$

This is a **definition** — there is nothing to prove. One side is just a rewording of the other.

**Higher-order Taylor's theorem** is the *theorem* one deduces from this by *iterating* differentiability: assume $C^k$ regularity, differentiate the Fréchet definition $k$ times, and the higher-order expansion with controlled remainder falls out by induction on the order. So:

1. **Definition:** differentiability $=$ first-order Taylor.
2. **Theorem:** $C^k$ regularity $\Rightarrow$ $k$-th order Taylor with controlled remainder.

Higher-order Taylor is downstream of the first-order definition, not the other way around.

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Historical: how this view crystallized)</span></p>

* **Newton/Leibniz (17th c.).** $f'(x)$ as a ratio of "infinitesimals" — not rigorous by modern standards.
* **Cauchy/Weierstrass (19th c.).** $f'(x)=\lim_{h\to 0}\frac{f(x+h)-f(x)}{h}$. The **limit-quotient definition** taught in calculus courses.
* **Carathéodory (early 20th c.).** Reformulated 1D differentiability as: there exists a number $L$ with $f(x+h)=f(x)+Lh+o(h)$. The **first-order-Taylor-as-definition** view — equivalent to Cauchy in 1D.
* **Fréchet (1911).** Generalized the Carathéodory form to normed/Banach spaces. The reason is forced by the spaces themselves: in 1D you can divide by $h$, so the limit-quotient form makes sense; in higher (or infinite) dimensions you *cannot divide by a vector*, so the limit-quotient form is unavailable. The "$L$ plus $o(\|h\|)$" form, being division-free, generalizes — and the Fréchet derivative is born.

In 1D the two formulations are equivalent twins; in higher dimensions only the Carathéodory/Fréchet form survives.

</div>

### Two operationalizations of one idea

The "first-order Taylor" *form* of differentiability — "linear approximation plus higher-order remainder" — is the **definitional** content. The familiar limit-quotient form $\lim\frac{f(x+h)-f(x)}{h}$ in 1D is one *operationalization* of that idea, available because we can divide. The Fréchet definition is *another* operationalization, available even when we can't divide.

```
                 first-order Taylor
       (the IDEA: f(x+h) ≈ f(x) + linear·h + o(||h||))
                /                       \
               /                         \
   limit-quotient form           Fréchet/Carathéodory form
   (only works in 1D,            (works in any normed
    because you divide            space — division-free)
    by h)
```

In modern multivariable and functional analysis one skips the limit-quotient road and goes directly to Fréchet, because that is the form that scales. In 1D textbooks the limit-quotient form is taught first because it is computationally friendly, but conceptually it is one of two equivalent renderings of "first-order Taylor."

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Two practical consequences)</span></p>

* **Multivariable case.** Existence of all partial derivatives at a point $x\in\mathbb R^n$ is *not* enough for Fréchet differentiability — there are classical counterexamples where every directional derivative at $x$ exists but no global linear approximation does. This is precisely the failure of Carathéodory's first-order Taylor expansion, even though every "1D slice" is differentiable.

* **Higher-order chain rule.** The one-line chain rule $D(g\circ f)(x)=Dg(f(x))\circ Df(x)$ is what one obtains by composing two **first-order** Taylor expansions. Higher-order versions (e.g., the Faà di Bruno formula) come from composing higher-order Taylor expansions and reading off the polynomial coefficients — same engine, applied at each order.

</div>

### One-line summary

> The Fréchet definition is "first-order Taylor expansion exists, with a linear leading term." Higher-order Taylor's theorem is the result one proves from this by iterating differentiability. The resemblance is the *content*, not a coincidence.

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Notation: $d$ vs $D$ vs $\nabla$ vs $J$ — what each means</summary>

A natural reader question at this point: why is the same object sometimes written $dE$, sometimes $DE$, sometimes $\nabla E$, sometimes $J_f$? Are these different things, or different conventions for the same thing?

**Short answer.** $dE$ and $DE$ denote *the same mathematical object* — a linear map. The choice between them is conventional/stylistic, not about whether the object is a vector, a matrix, or a general operator. The notations $\nabla E$ and $J_f$ refer to *different* objects: a *vector* (the Riesz representative of $dE$) and a *matrix* (a coordinate representation of $Df$), respectively.

A common misconception worth fixing first: **the differential $dE$ is not a vector.** It is a *linear functional* — an element of the dual space. What you may be picturing as "the vector" is the gradient $\nabla E$, which is a different object that happens to have the same numerical components in $\mathbb R^N$.

#### The categorical picture

For a smooth $E:\mathbb R^N\to\mathbb R$, three things at the point $x$ deserve separate names:

| object | notation | type | "shape" in coordinates |
|---|---|---|---|
| **differential** | $dE(x)=DE(x)$ | linear functional $\mathbb R^N\to\mathbb R$ | row (1×N) |
| **gradient** | $\nabla E(x)$ | vector in $\mathbb R^N$ (Riesz representative of $dE(x)$) | column (N×1) |
| **partials** | $\partial_i E(x)$ | numbers (components in a chosen basis) | scalars |

For a vector-valued map $f:\mathbb R^n\to\mathbb R^m$:

| object | notation | type | shape |
|---|---|---|---|
| **differential** | $df(x)=Df(x)$ | linear map $\mathbb R^n\to\mathbb R^m$ | m×n |
| **Jacobian matrix** | $J_f(x)$ | the matrix *representation* of $Df(x)$ in standard bases | m×n |

A few things to absorb from these tables:

* **The matrix is never the differential itself.** The matrix is one *representation* of the differential, available once you pick a basis. The differential is the abstract linear map.
* **$dE$ is dual ("row"); $\nabla E$ is primal ("column").** They have the same components numerically but live in different spaces. They get identified only because of the inner product on $\mathbb R^N$ via the Riesz representation theorem.
* **The same object has many names.** $dE(x)$, $DE(x)$, $E'(x)$, the *total derivative*, the *Fréchet derivative* — same thing. Likewise $df$, $Df$, $f'$, the *Jacobian* (as a linear map, not as the matrix), the *pushforward*.

#### What the $d$ vs $D$ convention actually tracks

The hypothesis "$d$ for scalar output, $D$ for vector/operator output" is a *soft pattern* observable in many books, but it is not a rule:

* **Differential-geometry / forms tradition.** Lowercase $d$ is preferred and carries algebraic baggage: $df$ is a **1-form**, $d$ is the **exterior derivative** with $d^2=0$, and the basis $\lbrace dx^1,\dots,dx^n\rbrace$ uses the same letter. This is the source of the lowercase. Most natural when the codomain is $\mathbb R$, but not restricted to it.
* **Analysis / functional-analysis tradition.** Uppercase $D$ is preferred, with no geometric baggage; it just denotes "the linear-map-valued first derivative." Codomain-agnostic.
* **Multivariable calculus textbooks.** Often use $Df$ for vector-valued $f$ and $df$ for scalar-valued $f$, which is probably the pattern most students notice. But Spivak (*Calculus on Manifolds*) uses $Df$ for everything, while Lee (*Smooth Manifolds*) uses $df$ for everything.

So the choice is **stylistic and field-dependent**, not dimensional. The same author may write $dE$ for a scalar-valued energy on one page and $df$ for a vector-valued map on the next.

#### In the general setting (Hilbert/Banach spaces, manifolds, operators)

There is **no "always use $D$" rule**. Both $d$ and $D$ remain in use:

* On a smooth manifold $M$ with $E:M\to\mathbb R$, the differential is $dE_x:T_xM\to\mathbb R$ — a covector. Most differential-geometry books write $dE$.
* For $f:M\to N$ between manifolds, the differential is $df_x:T_xM\to T_{f(x)}N$, also called the *pushforward* and sometimes written $f_{*,x}$ or $Tf$. Many analysis-flavored geometry books prefer $Df_x$.
* On a Hilbert space $\mathcal H$ with $E:\mathcal H\to\mathbb R$, the differential is a bounded linear functional $dE(u)\in\mathcal H^*$. Both $dE(u)$ and $DE(u)$ are common; the Riesz representative is the gradient $\nabla_{\mathcal H}E(u)\in\mathcal H$.
* For $f:\mathcal X\to\mathcal Y$ between Banach spaces (so the derivative is a bounded linear *operator*), most functional-analysis textbooks write $Df(u)\in\mathcal L(\mathcal X,\mathcal Y)$, but you'll also see $df(u)$.

#### The dictionary that carries through every setting

* **Differential / total derivative.** A linear map between the appropriate tangent / ambient spaces. Notations: $df$, $Df$, sometimes $f_{*}$, $Tf$, $f'$.
* **Gradient.** The Riesz / metric-dual representative of the differential. Requires an inner product (or a Riemannian metric on a manifold). Notation: $\nabla f$, $\mathrm{grad}\,f$, or $\nabla_{\mathcal H}f$ to specify the metric.
* **Partial derivatives / coordinate components.** $\partial_i f$, $\frac{\partial f}{\partial x^i}$. These appear once a basis is fixed.
* **Jacobian matrix.** The matrix representation of $Df$ in chosen bases. Notation: $J_f$ — and this *is* genuinely a matrix, distinct from the linear map it represents.

#### Why gradient-flow theory uses $dE$

The choice of writing $dE(x(t)).\dot x(t)$ rather than $DE(x(t)).\dot x(t)$ has nothing to do with the differential being "a vector":

* Gradient-flow theory eventually passes to non-Euclidean settings (Hilbert spaces, Wasserstein space) where $\nabla E$ is metric-dependent and $\partial E/\partial x$ is meaningless. The differential $dE$ as a linear functional is the intrinsic object that survives.
* Lowercase $d$ matches the differential-geometry / Otto-calculus tradition that the surrounding literature uses, so the notation is forward-compatible with later chapters.

One could equally well write $DE(x)\cdot\dot x$, $E'(x)\cdot\dot x$, or $\langle\nabla E(x),\dot x\rangle$ — all four denote the same number. The first three are the "differential applied to a velocity" picture; the fourth is "gradient inner-producted with velocity," which sneaks in an inner product.

#### Practical takeaways

* When you see $df$ or $Df$, mentally translate to "the linear-map first derivative." Don't read anything into the case of the letter.
* When you see $\nabla f$, mentally flag "this requires an inner product; the choice of inner product matters."
* When you see $J_f$, that's specifically the matrix.
* The dot in $dE(x).\dot x$ (or $Df(x).\dot x$, etc.) is **always** "linear map evaluated on a vector," regardless of whether the case is upper or lower.

</details>
</div>

### Coordinate renderings

In familiar settings, $Df(x)$ is represented by a familiar object:

| $V$ | $W$ | What $Df(x)$ "is" (as an object you compute with) |
|---|---|---|
| $\mathbb R$ | $\mathbb R$ | the number $f'(x)$, acting by $h\mapsto f'(x)h$ |
| $\mathbb R^n$ | $\mathbb R^m$ | the $m\times n$ Jacobian matrix |
| $\mathbb R^n$ | $\mathbb R$ | the $1\times n$ row of partials = the differential $dE(x)$ |
| $\mathbb R$ | $\mathbb R^n$ | the column $\dot c(t)\in\mathbb R^n$ (velocity), acting by $\tau\mapsto\tau\,\dot c(t)$ |
| Hilbert $\mathcal H$ | $\mathbb R$ | a bounded linear functional $\mathcal H\to\mathbb R$, *represented by* a gradient via Riesz |

In every case $Df(x)$ is a linear map. The matrix / number / vector is one *representation* of that map; it is never the map itself.

### The chain rule, one line

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Chain rule, abstract form)</span></p>

Let $V\xrightarrow{f}W\xrightarrow{g}Z$ be maps between normed spaces, $f$ Fréchet-differentiable at $x$, $g$ Fréchet-differentiable at $f(x)$. Then $g\circ f$ is Fréchet-differentiable at $x$, and

$$
D(g\circ f)(x) \;=\; Dg\bigl(f(x)\bigr)\;\circ\;Df(x).
$$

The $\circ$ on the right is **composition of linear maps**.

</div>

The intuition is the same one that makes the 1D rule plausible:

$$
g(f(x+h))\approx g\bigl(f(x)+Df(x)h\bigr)\approx g(f(x))+Dg(f(x))\bigl(Df(x)h\bigr),
$$

and the composition of the two linearizations reads off the slope of $g\circ f$.

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Decoding the two $\approx$'s</summary>

The two $\approx$'s above are *the same operation* — the first-order Taylor approximation — applied to **two different functions** at **two different base points**.

**The pattern.** For any differentiable $\varphi$ and any base point $y$ with small perturbation $k$,

$$
\varphi(y+k) \;\approx\; \varphi(y)+D\varphi(y)\,k. \qquad(\star)
$$

This is the Fréchet definition rebranded as "approximation"; the error is $o(\|k\|)$ as $k\to 0$.

**Both $\approx$'s are instances of $(\star)$:**

| step | function $\varphi$ | base point $y$ | perturbation $k$ | $(\star)$ becomes |
|---|---|---|---|---|
| **1st $\approx$** | $f$ | $x$ | $h$ | $f(x+h)\approx f(x)+Df(x)\,h$ |
| **2nd $\approx$** | $g$ | $f(x)$ | $Df(x)\,h$ | $g\bigl(f(x)+Df(x)h\bigr)\approx g(f(x))+Dg(f(x))\bigl(Df(x)h\bigr)$ |

**Naming the pieces.** Setting $y:=f(x)$ and $k:=Df(x)\,h$, the second step reads

$$
g(\underbrace{f(x+h)}_{\approx\,y+k})\;\approx\;g(y+k)\;\stackrel{(\star)\text{ for }g}{\approx}\;g(y)+Dg(y)\,k,
$$

and substituting $y=f(x)$, $k=Df(x)\,h$ recovers $g(f(x))+Dg(f(x))\bigl(Df(x)h\bigr)$.

**Why $(\star)$ is justified for $g$ at $f(x)$.** $(\star)$ requires the perturbation $k$ to be small. Here $k=Df(x)\,h$, and:

* $Df(x)$ is a *bounded* linear map (part of Fréchet differentiability), so there is a constant $C$ with $\|Df(x)\,h\|\le C\,\|h\|$;
* hence $\|k\|\le C\,\|h\|\to 0$ as $h\to 0$.

So whenever $h$ is small enough that the Taylor expansion of $f$ at $x$ is valid, $k=Df(x)\,h$ is small enough that the Taylor expansion of $g$ at $f(x)$ is valid too. Both approximations have $o(\|h\|)$ error, and composing them keeps that order — which is the rigorous content behind the chain rule.

**Reading the line LHS-to-RHS.**

$$
\underbrace{g(f(x+h))}_{\text{LHS}}
\;\stackrel{\text{Taylor of }f}{\approx}\;
g\bigl(\underbrace{f(x)+Df(x)h}_{=\,y+k}\bigr)
\;\stackrel{\text{Taylor of }g}{\approx}\;
\underbrace{g(f(x))+Dg(f(x))\bigl(Df(x)h\bigr)}_{\text{RHS}}.
$$

Comparing LHS and RHS,

$$
g(f(x+h))\;\approx\;g(f(x))+\bigl[Dg(f(x))\circ Df(x)\bigr]h,
$$

which is exactly $(\star)$ applied to the composition $g\circ f$ — i.e. it reads off $D(g\circ f)(x)=Dg(f(x))\circ Df(x)$, the chain rule.

So the derivation is **two applications of the same Taylor approximation, glued together**: first push $h$ inside (linearize $f$), then pull the linearized increment $Df(x)\,h$ out through $g$ (linearize $g$). The chain rule is what falls out when you compose those two linearizations.

</details>
</div>

In coordinates on $\mathbb R^n,\mathbb R^m,\mathbb R^p$, "composition of linear maps" is matrix multiplication: the Jacobian of $g\circ f$ is the product $J_g(f(x))\cdot J_f(x)$. The familiar one-variable rule is the $1\times 1$ case.

### Worked case: a curve into a Euclidean landscape

This is the situation that produces the notation $dE(x(t)).\dot x(t)$ in gradient-flow theory. Compose

$$
\mathbb R\;\xrightarrow{\;x\;}\;\mathbb R^N\;\xrightarrow{\;E\;}\;\mathbb R,\qquad f:=E\circ x.
$$

Each piece has a derivative which is a linear map:

* $Dx(t):\mathbb R\to\mathbb R^N$, the linear map $\tau\mapsto\tau\,\dot x(t)$. So $Dx(t)$ "is" the velocity vector $\dot x(t)$, viewed as a linear map $\mathbb R\to\mathbb R^N$.
* $DE(x(t)):\mathbb R^N\to\mathbb R$, the linear functional $v\mapsto DE(x(t))\,v$. So $DE(x(t))$ "is" the differential $dE(x(t))$.

The chain rule says $D(E\circ x)(t)=DE(x(t))\circ Dx(t)$, a linear map $\mathbb R\to\mathbb R$. Apply both sides to the unit tangent vector $1\in\mathbb R$:

$$
\underbrace{D(E\circ x)(t)\cdot 1}_{=\,\frac{d}{dt}E(x(t))}
\;=\;DE(x(t))\bigl(\underbrace{Dx(t)\cdot 1}_{=\,\dot x(t)}\bigr)
\;=\;DE(x(t)).\dot x(t).
$$

The expression $DE(x(t)).\dot x(t)$ is "the differential $DE(x(t))$ *applied to* the vector $\dot x(t)$." The dot is **evaluation of one linear map on a vector**, not multiplication and not division. The next equality $\;=\langle\nabla E(x(t)),\dot x(t)\rangle$ is the Riesz identification turning the linear functional $dE(x(t))$ into the gradient vector $\nabla E(x(t))$ — a coordinate-dependent picture.

### Chain rule in Hilbert spaces

Let $\mathcal H$ be a (real) Hilbert space with inner product $\langle\cdot,\cdot\rangle_{\mathcal H}$. Suppose $E:\mathcal H\to\mathbb R$ is Fréchet-differentiable at $u\in\mathcal H$. The differential $dE(u)$ is then a *bounded linear functional* on $\mathcal H$:

$$
dE(u):\mathcal H\to\mathbb R,\qquad v\mapsto dE(u).v.
$$

By the **Riesz representation theorem**, every bounded linear functional on $\mathcal H$ is represented by a unique vector. Concretely, there exists a unique $\nabla_{\mathcal H} E(u)\in\mathcal H$ such that

$$
dE(u).v \;=\; \langle\nabla_{\mathcal H} E(u),\,v\rangle_{\mathcal H}\qquad\text{for all } v\in\mathcal H.
$$

This $\nabla_{\mathcal H} E(u)$ is called the **gradient** of $E$ at $u$ *with respect to the inner product $\langle\cdot,\cdot\rangle_{\mathcal H}$*. The differential is intrinsic; the gradient is metric-dependent.

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Chain rule for a curve in a Hilbert space)</span></p>

Let $u:I\to\mathcal H$ be Fréchet-differentiable on an interval $I\subseteq\mathbb R$ (so $\partial_t u(t)\in\mathcal H$ for each $t$), and let $E:\mathcal H\to\mathbb R$ be Fréchet-differentiable. Then $t\mapsto E(u(t))$ is differentiable on $I$, and

$$
\frac{d}{dt}E(u(t)) \;=\; dE(u(t)).\,\partial_t u(t)
\;=\; \langle\nabla_{\mathcal H} E(u(t)),\,\partial_t u(t)\rangle_{\mathcal H}.
$$

</div>

Read the chain: the first equality is the abstract chain rule (composition of two Fréchet derivatives, applied to the unit tangent in $\mathbb R$); the second equality is Riesz, identifying the linear functional $dE$ with a vector $\nabla_{\mathcal H} E$ via the inner product.

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Why $\partial_t u(t)$ instead of $Du(t)$?</summary>

A natural reader question at this point: in the curve case earlier we wrote things like $Du(t)$ for the Fréchet derivative of a curve, but here in the chain rule we suddenly write $\partial_t u(t)$. Why?

#### Short answer

$\partial_t u(t)$ is the **vector** in $\mathcal H$ — not the linear map. It is what you actually plug into the linear functional $dE(u(t))$. The object $Du(t)$, in strict Fréchet terms, is a *linear map* $\mathbb R\to\mathcal H$, not a vector. The two are equivalent under the canonical isomorphism $\mathrm{Hom}(\mathbb R,\mathcal H)\cong\mathcal H$, but writing $\partial_t u(t)$ explicitly picks out the vector and matches PDE conventions.

#### The three things, three notations

For a curve $u:I\to\mathcal H$ on an interval $I\subseteq\mathbb R$:

| object | type | how the others are derived from it |
|---|---|---|
| $Du(t)$ | linear map $\mathbb R\to\mathcal H$ (Fréchet derivative) | — |
| $u'(t)$ | vector in $\mathcal H$ (the velocity) | $u'(t):=Du(t)\cdot 1$ |
| $\partial_t u(t)$ | vector in $\mathcal H$, identical to $u'(t)$ | PDE name for the same thing |

So $u'(t)$ and $\partial_t u(t)$ are the same object — a vector in $\mathcal H$. The notation $Du(t)$ is the *abstract* linear-map version; $\partial_t u(t)$ is the *concrete* vector version.

#### Why $\partial_t$ specifically — disambiguation in PDE-land

The reason you almost always see $\partial_t$ rather than $Du$ in this context is that in PDE settings $u$ is virtually never just "a curve in $\mathcal H$" — it is a function of *several* variables, and you need the notation to say *which* variable you are differentiating with respect to.

Concretely, an evolution equation like the heat equation has $u:[0,T]\times\Omega\to\mathbb R$, equivalent to a curve $u:[0,T]\to\mathcal H$ with $\mathcal H=L^2(\Omega)$ via $u(t)\equiv u(t,\cdot)$. Two natural derivatives are in play:

* $\partial_t u$ — the time derivative (in $\mathcal H$), which is what the chain rule needs.
* $\partial_{x_i} u,\ \nabla u,\ \Delta u$ — spatial derivatives (in $\Omega$), which appear on the other side of the PDE.

If $Du$ were used for the time derivative, there would be immediate trouble: the same PDE has spatial $D$'s on the right-hand side. PDE notation reserves $\partial_t$ specifically for "differentiate the curve in $\mathcal H$ with respect to the curve parameter," and reserves $\partial_x,\nabla,\Delta$ for "differentiate the values of $u$ with respect to spatial coordinates."

#### The other reason — already-overloaded $D$

Even setting aside spatial derivatives, $D$ is **already in use** in the chain rule for the Fréchet derivative of the *energy*:

$$
dE(u):\mathcal H\to\mathbb R\text{ — the differential of the energy } E:\mathcal H\to\mathbb R.
$$

Writing $Du$ for the time derivative of the curve creates a notation collision: $Du$ would be a derivative with respect to time, $dE$ a derivative with respect to the curve's value. Same letter, two different "directions" of differentiation. PDE notation breaks the collision by reserving $\partial_t$ for the time slot.

#### What the chain rule actually does, fully decoded

Start from the abstract chain rule for $E\circ u:I\to\mathbb R$:

$$
D(E\circ u)(t)\;=\;dE(u(t))\;\circ\;Du(t).
$$

Both sides are linear maps $\mathbb R\to\mathbb R$. To extract the *number* $\frac{d}{dt}E(u(t))$, apply both sides to the unit tangent $1\in\mathbb R$:

$$
\underbrace{D(E\circ u)(t)\cdot 1}_{=\,\frac{d}{dt}E(u(t))}
\;=\;dE(u(t))\Bigl(\underbrace{Du(t)\cdot 1}_{=\,u'(t)\,=\,\partial_t u(t)}\Bigr)
\;=\;dE(u(t)).\partial_t u(t).
$$

Read the second equality slowly: $dE(u(t))$ is a *linear functional on $\mathcal H$*, so it eats *vectors in $\mathcal H$*. The thing being fed in is $Du(t)\cdot 1$ — the vector that $Du(t)$ produces when the unit tangent is plugged in. That vector is $\partial_t u(t)$. So writing $\partial_t u(t)$ in the chain rule is exactly the right move: it is the object whose *type* matches the input of $dE$.

If $Du(t)$ were written in that slot, it would technically be wrong — $Du(t)$ is a linear map, and $dE(u(t))$ does not eat linear maps; it eats elements of $\mathcal H$. The chain rule

$$
\frac{d}{dt}E(u(t))\;=\;dE(u(t)).\partial_t u(t)
$$

silently performs the identification "linear map $\mathbb R\to\mathcal H$ ↔ its value at $1$" by writing $\partial_t u(t)$ in the slot where the vector belongs.

#### Quick notation dictionary in the PDE / curve-in-$\mathcal H$ setting

* $\partial_t u$ — the time derivative of the curve, a vector in $\mathcal H$. Same as $u'(t)$ or $\dot u(t)$.
* $\partial_{x_i} u,\ \nabla u,\ \Delta u$ — spatial derivatives of $u$ regarded as a function of $x$.
* $du$ — almost never used in this slot, because $d$ is reserved for differentials of *functionals* and for forms.
* $Du$ — used when one wants to emphasize the abstract Fréchet picture (often in functional-analysis textbooks), with the understanding that $Du(t)\cdot 1=\partial_t u(t)$.
* $dE(u)$ — the Fréchet differential of the energy $E$ at the point $u\in\mathcal H$; a linear functional on $\mathcal H$.
* $\nabla_{\mathcal H}E(u)$ — the Riesz representative of $dE(u)$; a vector in $\mathcal H$. Metric-dependent.

#### One-line summary

> $\partial_t u(t)$ is the *vector* in $\mathcal H$ — the object that goes into linear functionals like $dE(u(t))$. $Du(t)$ is the abstract linear map $\mathbb R\to\mathcal H$ that *produces* this vector when applied to $1$. PDE notation prefers $\partial_t$ because it (a) explicitly names the variable of differentiation, distinguishing it from spatial derivatives, and (b) avoids notation collision with the Fréchet derivative $dE$ of the energy.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The gradient depends on the inner product)</span></p>

Because the gradient is *defined* by the Riesz identity $\langle\nabla_{\mathcal H} E(u),v\rangle_{\mathcal H}=dE(u).v$, switching the inner product changes the gradient. A different inner product $\langle\cdot,\cdot\rangle_{\mathcal H'}$ on the same space yields a *different* gradient $\nabla_{\mathcal H'} E$, and a different gradient flow

$$
\partial_t u = -\nabla_{\mathcal H'} E(u).
$$

This is the entire reason that Otto / Wasserstein gradient flows look different from $L^2$-gradient flows on the same energy: same $E$, different inner product, different evolution. The differential $dE$ does not depend on which inner product you choose.

</div>

### Worked example: the heat equation as an $L^2$-gradient flow

Let $\Omega\subset\mathbb R^d$ be bounded with smooth boundary, $\mathcal H=L^2(\Omega)$, and consider the **Dirichlet energy**

$$
E[u]=\frac{1}{2}\int_\Omega |\nabla u|^2\,dx
$$

restricted to (say) $H^1_0(\Omega)$. Compute the differential by varying $u\rightsquigarrow u+\varepsilon\eta$ with $\eta\in H^1_0(\Omega)$:

$$
\frac{d}{d\varepsilon}\bigg|_0 E[u+\varepsilon\eta]
= \int_\Omega\nabla u\cdot\nabla\eta\,dx
\;\stackrel{\text{int. by parts}}{=}\;
-\int_\Omega(\Delta u)\,\eta\,dx.
$$

So $dE[u].\eta = -\int_\Omega(\Delta u)\eta\,dx$. By Riesz in $L^2$,

$$
dE[u].\eta = \langle\nabla_{L^2}E[u],\,\eta\rangle_{L^2} = \int_\Omega (\nabla_{L^2}E[u])\,\eta\,dx,
$$

so $\nabla_{L^2}E[u]=-\Delta u$. The $L^2$-gradient flow is therefore

$$
\partial_t u = -\nabla_{L^2}E[u] = \Delta u,
$$

i.e. the **heat equation**. Its stationary solutions ($\Delta u=0$) are exactly the critical points of $E$, the Euler–Lagrange solutions — same picture as in finite dimensions, with the inner-product structure spelled out.

### Why this picture pays off

This abstract viewpoint is what allows the same chain rule to operate across:

* **Hilbert spaces.** As above; gradient $=$ Riesz representative of the differential.
* **Riemannian manifolds.** $M$ smooth, $E:M\to\mathbb R$, $\gamma:I\to M$. Then $\dot\gamma(t)\in T_{\gamma(t)}M$, $dE(\gamma(t)):T_{\gamma(t)}M\to\mathbb R$, and the chain rule

  $$
  \frac{d}{dt}E(\gamma(t)) = dE(\gamma(t)).\dot\gamma(t)
  $$

  is *the very definition* of the action of cotangent on tangent. There is no $\nabla E$ until you choose a Riemannian metric.
* **Wasserstein / metric measure spaces.** In Otto calculus, "tangent vectors" at a probability measure are velocity fields modulo gradients, with no canonical inner product until one is chosen; the differential of a functional remains a linear map on tangents, the chain rule still reads $\frac{d}{dt}\mathcal F(\rho_t)=d\mathcal F(\rho_t).\partial_t\rho_t$, and the gradient is the metric-dependent representative.

In all of these the chain rule is the same one-line composition statement; only the spaces $V,W,Z$ and the available identifications change.

### Sanity-check exercise

Take $E(x_1,x_2)=\tfrac12(x_1^2+3x_2^2)$ and $x(t)=(e^{-t},e^{-3t})$ (the closed-form gradient-flow trajectory from $(1,1)$ for this energy).

  $$\dot x(t)=(-e^{-t},-3e^{-3t})$$

* $dE(x(t))$ is the row 
  
  $$(x_1(t),3x_2(t))=(e^{-t},3e^{-3t})$$
  
* Apply: 
  
  $$dE(x(t)).\dot x(t)=e^{-t}\cdot(-e^{-t})+3e^{-3t}\cdot(-3e^{-3t})=-e^{-2t}-9e^{-6t}$$

* Compare to 
  
  $$\frac{d}{dt}E(x(t))=\frac{d}{dt}\bigl[\tfrac12 e^{-2t}+\tfrac32 e^{-6t}\bigr]=-e^{-2t}-9e^{-6t}. ✓$$

The two computations agree because the chain rule is true. The first line is the *conceptual* statement (apply a linear map to a vector); the second is the *coordinate calculation* (sum of products of partials and components). Once the spaces are not $\mathbb R^n$, only the first line still makes sense — that is why the abstract picture is the durable one.

## Implicit (backward) Euler vs explicit (forward) Euler

Forward Euler (a.k.a. *explicit Euler*) is the discretisation that every numerical analyst meets first; *implicit Euler* (a.k.a. *backward Euler*) is its much-more-stable partner. The two differ by **a single character** in their update rule, but the consequences are large — and in the gradient-flow / non-smooth-ODE setting, the implicit scheme is not a numerical curiosity but the **analytical engine** that proves existence of solutions when the right-hand side fails to be Lipschitz.

### The two schemes side by side

For an ODE $\dot x = f(x)$ with step size $h>0$:

| scheme | update rule | RHS evaluated at |
|---|---|---|
| **Explicit (forward) Euler** | $x_{k+1}=x_k+h\,f(x_k)$ | the **old** point $x_k$ |
| **Implicit (backward) Euler** | $x_{k+1}=x_k+h\,f(x_{k+1})$ | the **new** point $x_{k+1}$ |

Forward Euler is an *explicit formula* in $x_k$ — just plug in. Backward Euler is an **equation**

$$
x_{k+1}-h\,f(x_{k+1}) = x_k,
$$

generally nonlinear, that must be solved at every time step (typically by Newton, fixed-point iteration, or — in the gradient-flow case — by minimization). That extra work is the cost; what one buys with it is **stability**.

### Geometric picture: tangent at start vs tangent at end

* **Forward Euler** uses the *initial* tangent of the step: stand at $x_k$, look at the slope $f(x_k)$ there, walk in that direction for time $h$.
* **Backward Euler** uses the *final* tangent: choose $x_{k+1}$ so that the line through it with slope $f(x_{k+1})$ passes back through $x_k$ at time $t_k$.

For the dissipative ODE $\dot x=-x$ with $x_k=1$ and a coarse $h=0.6$, the two schemes overshoot in **opposite directions**: forward Euler lands too low (it commits to the steepest initial slope and overruns), backward Euler lands too high (it uses a less-steep terminal slope). The exact value $e^{-h}$ sits in between.

<figure>
  <img src="{{ '/assets/images/notes/random/euler_geometric.png' | relative_url }}" alt="Geometric picture of one step of forward and backward Euler on the linear decay ODE: forward Euler follows the tangent at the start and undershoots, backward Euler uses the tangent at the end and overshoots, the exact solution sits between them" loading="lazy">
  <figcaption>One step on $\dot{x}=-x$ from $x_k=1$ with $h=0.6$. Forward Euler (red) extrapolates the tangent at $x_k$ and lands at $x_{k+1}^{\rm FE}=0.4$. Backward Euler (blue) draws the line of slope $f(x_{k+1})$ through the new point that reaches back to $x_k$ at time $t_k$, landing at $x_{k+1}^{\rm BE}=1/(1+h)\approx 0.625$. The true value $x_k e^{-h}\approx 0.549$ (green diamond) sits in between.</figcaption>
</figure>

### Why bother — stability

The classical test problem is the linear decay $\dot x=\lambda x$ with $\lambda<0$. Exact solution: $x(t)=x_0 e^{\lambda t}$, smoothly decaying.

* **Forward Euler:** $x_{k+1}=(1+h\lambda)x_k$, so $x_k=(1+h\lambda)^k x_0$. Stable iff $\|1+h\lambda\|<1$, i.e. iff $h<2/\|\lambda\|$. For *stiff* problems (large $\|\lambda\|$), this forces a microscopic step.
* **Backward Euler:** $x_{k+1}=x_k/(1-h\lambda)$, so $x_k=\bigl(1/(1-h\lambda)\bigr)^k x_0$. Stable for **every** $h>0$; the discrete dynamics inherit the dissipativity of the continuous ones, regardless of step size.

Backward Euler is **A-stable** (unconditionally stable); forward Euler is only conditionally stable. In stiff regimes, this difference is decisive — and it is *the* reason implicit methods dominate parabolic-PDE numerics.

The next figure shows the three regimes for $\lambda=-50$: well inside the forward-Euler stability disk (smooth on both sides), just inside (forward Euler oscillates but stays bounded), outside (forward Euler blows up exponentially while backward Euler keeps decaying smoothly).

<figure>
  <img src="{{ '/assets/images/notes/random/euler_stability_decay.png' | relative_url }}" alt="Three panels showing forward and backward Euler on the linear decay equation x-dot equals minus 50 x for three step sizes; forward Euler is stable only when h is below 2 over absolute lambda" loading="lazy">
  <figcaption>$\dot x=\lambda x,\ \lambda=-50$: as $h$ crosses $2/|\lambda|=0.04$, forward Euler (red) goes from smooth decay to bounded oscillation to outright divergence; backward Euler (blue) stays close to the exact solution at every step size.</figcaption>
</figure>

### Stability regions in the complex $h\lambda$ plane

Plotting the stability conditions in the complex plane $z=h\lambda$ exposes the structural difference. Forward Euler is stable only on a **disk** of radius $1$ centred at $-1$ — a tiny corner of the complex plane that excludes most of the left half-plane. Backward Euler is stable on the **complement of a disk** of radius $1$ centred at $+1$, which covers the *entire* left half-plane (where the exact solution itself decays) and much else besides.

<figure>
  <img src="{{ '/assets/images/notes/random/euler_stability_regions.png' | relative_url }}" alt="Linear stability regions in the complex plane: forward Euler is stable only inside a small disk centered at minus one, while backward Euler is stable everywhere outside a disk centered at plus one, including the entire left half-plane" loading="lazy">
  <figcaption>Stability regions in $z=h\lambda$. Left: forward Euler is stable on the red disk $|1+h\lambda|\le 1$. Right: backward Euler is stable on the blue region $|1-h\lambda|\ge 1$. The grey strip is the left half-plane $\mathrm{Re}(h\lambda)<0$ where the *exact* solution decays — backward Euler covers this entire strip, forward Euler only a fraction.</figcaption>
</figure>

The takeaway: backward Euler's stable region **strictly contains** the half-plane on which the exact ODE is dissipative. Forward Euler's region is a small disk that misses most of it.

<figure>
  <div class="ee-viz">
    <style>
      .ee-viz {
        --color-text-primary: #1a1a1a;
        --color-text-secondary: #4a4a4a;
        --color-text-tertiary: #777777;
        --color-background-primary: #ffffff;
        --color-background-secondary: #f3f4f8;
        --color-border-primary: #aaaaaa;
        --color-border-secondary: #c8c8c8;
        --color-border-tertiary: #dddddd;
        --border-radius-md: 6px;
        --font-sans: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      }
      .ee-viz select { font-family: var(--font-sans); font-size: 13px; padding: 6px 8px; border: 0.5px solid var(--color-border-tertiary); border-radius: var(--border-radius-md); background: var(--color-background-primary); color: var(--color-text-primary); }
    </style>

    <div style="padding: 1rem 0;">

      <div style="margin-bottom: 1.25rem;">
        <label for="ee-ode-select" style="display: block; font-size: 13px; color: var(--color-text-secondary); margin-bottom: 6px;">Initial value problem</label>
        <select id="ee-ode-select" style="width: 100%;">
          <option value="0">dy/dt = y,&nbsp;&nbsp;y(0) = 1&nbsp;&nbsp;(exponential growth)</option>
          <option value="1">dy/dt = -y,&nbsp;&nbsp;y(0) = 1&nbsp;&nbsp;(exponential decay)</option>
          <option value="2" selected>dy/dt = -10y,&nbsp;&nbsp;y(0) = 1&nbsp;&nbsp;(stiff — implicit handles any h)</option>
          <option value="3">dy/dt = -2ty,&nbsp;&nbsp;y(0) = 1&nbsp;&nbsp;(Gaussian)</option>
          <option value="4">dy/dt = t - y,&nbsp;&nbsp;y(0) = 1</option>
        </select>
      </div>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 1rem;">
        <div>
          <label style="display: flex; justify-content: space-between; font-size: 13px; color: var(--color-text-secondary); margin-bottom: 6px;">
            <span>Step size&nbsp;&nbsp;<em>h</em></span>
            <span id="ee-h-out" style="color: var(--color-text-primary); font-weight: 500; font-variant-numeric: tabular-nums;">0.30</span>
          </label>
          <input type="range" id="ee-h-slider" min="0.02" max="1" step="0.02" value="0.3" style="width: 100%;">
        </div>
        <div>
          <label style="display: flex; justify-content: space-between; font-size: 13px; color: var(--color-text-secondary); margin-bottom: 6px;">
            <span>Final time&nbsp;&nbsp;<em>t<sub>max</sub></em></span>
            <span id="ee-t-out" style="color: var(--color-text-primary); font-weight: 500; font-variant-numeric: tabular-nums;">2.0</span>
          </label>
          <input type="range" id="ee-t-slider" min="0.5" max="6" step="0.5" value="2" style="width: 100%;">
        </div>
      </div>

      <div style="margin-bottom: 1.25rem;">
        <label style="display: inline-flex; align-items: center; gap: 8px; font-size: 13px; color: var(--color-text-secondary); cursor: pointer;">
          <input type="checkbox" id="ee-compare-toggle" checked style="width: 16px; height: 16px;">
          <span>Show explicit Euler for comparison</span>
        </label>
      </div>

      <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 1.25rem;">
        <div style="background: var(--color-background-secondary); border-radius: var(--border-radius-md); padding: 12px 16px;">
          <div style="font-size: 12px; color: var(--color-text-secondary);">Steps</div>
          <div id="ee-stat-steps" style="font-size: 22px; font-weight: 500; font-variant-numeric: tabular-nums;">—</div>
        </div>
        <div style="background: var(--color-background-secondary); border-radius: var(--border-radius-md); padding: 12px 16px;">
          <div style="font-size: 12px; color: var(--color-text-secondary);">Max |error| (implicit)</div>
          <div id="ee-stat-maxerr" style="font-size: 22px; font-weight: 500; font-variant-numeric: tabular-nums;">—</div>
        </div>
        <div style="background: var(--color-background-secondary); border-radius: var(--border-radius-md); padding: 12px 16px;">
          <div style="font-size: 12px; color: var(--color-text-secondary);">y(t<sub>max</sub>) implicit</div>
          <div id="ee-stat-final" style="font-size: 22px; font-weight: 500; font-variant-numeric: tabular-nums;">—</div>
        </div>
      </div>

      <div style="display: flex; gap: 18px; margin-bottom: 8px; font-size: 13px; color: var(--color-text-secondary); flex-wrap: wrap;">
        <span style="display: inline-flex; align-items: center; gap: 8px;">
          <svg width="28" height="10" viewBox="0 0 28 10"><line x1="0" y1="5" x2="28" y2="5" stroke="#888780" stroke-width="2"/></svg>
          Exact
        </span>
        <span style="display: inline-flex; align-items: center; gap: 8px;">
          <svg width="28" height="10" viewBox="0 0 28 10"><line x1="0" y1="5" x2="28" y2="5" stroke="#D85A30" stroke-width="2" stroke-dasharray="5,3"/><circle cx="14" cy="5" r="2.5" fill="#D85A30"/></svg>
          Implicit Euler
        </span>
        <span id="ee-explicit-legend" style="display: inline-flex; align-items: center; gap: 8px;">
          <svg width="28" height="10" viewBox="0 0 28 10"><line x1="0" y1="5" x2="28" y2="5" stroke="#378ADD" stroke-width="2" stroke-dasharray="2,3"/><polygon points="11,9 17,9 14,2" fill="#378ADD"/></svg>
          Explicit Euler
        </span>
      </div>

      <div style="position: relative; width: 100%; height: 320px;">
        <canvas id="ee-euler-chart" role="img" aria-label="Plot comparing the exact ODE solution to its implicit Euler approximation, optionally overlaid with the explicit Euler method.">Plot comparing exact, implicit Euler, and optionally explicit Euler approximations.</canvas>
      </div>
    </div>
  </div>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
  <script>
  (function () {
    if (typeof Chart === 'undefined') { return; }
    const ODES = [
      { f: (t, y) => y,          exact: t => Math.exp(t),                y0: 1,
        implicit: (tn, yn, h) => { const d = 1 - h; return Math.abs(d) < 1e-9 ? NaN : yn / d; } },
      { f: (t, y) => -y,         exact: t => Math.exp(-t),               y0: 1,
        implicit: (tn, yn, h) => yn / (1 + h) },
      { f: (t, y) => -10 * y,    exact: t => Math.exp(-10 * t),          y0: 1,
        implicit: (tn, yn, h) => yn / (1 + 10 * h) },
      { f: (t, y) => -2 * t * y, exact: t => Math.exp(-t * t),           y0: 1,
        implicit: (tn, yn, h) => yn / (1 + 2 * h * (tn + h)) },
      { f: (t, y) => t - y,      exact: t => t - 1 + 2 * Math.exp(-t),   y0: 1,
        implicit: (tn, yn, h) => (yn + h * (tn + h)) / (1 + h) },
    ];

    const isDark = matchMedia('(prefers-color-scheme: dark)').matches;
    const grid = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
    const tick = isDark ? 'rgba(255,255,255,0.65)' : 'rgba(0,0,0,0.65)';

    const ctx = document.getElementById('ee-euler-chart').getContext('2d');
    const chart = new Chart(ctx, {
      type: 'line',
      data: { datasets: [
        { label: 'Exact',    data: [], borderColor: '#888780', borderWidth: 2, pointRadius: 0, tension: 0 },
        { label: 'Implicit', data: [], borderColor: '#D85A30', backgroundColor: '#D85A30', borderWidth: 2, pointRadius: 4, pointHoverRadius: 6, borderDash: [5, 3], pointStyle: 'circle' },
        { label: 'Explicit', data: [], borderColor: '#378ADD', backgroundColor: '#378ADD', borderWidth: 2, pointRadius: 4, pointHoverRadius: 6, borderDash: [2, 3], pointStyle: 'triangle' },
      ]},
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        spanGaps: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            mode: 'nearest', intersect: false,
            callbacks: {
              title: (items) => 't = ' + (+items[0].parsed.x).toFixed(3),
              label: (item) => item.dataset.label + ': ' + (Number.isFinite(item.parsed.y) ? (+item.parsed.y).toFixed(4) : '—'),
            }
          }
        },
        scales: {
          x: { type: 'linear', title: { display: true, text: 't', color: tick }, grid: { color: grid }, ticks: { color: tick } },
          y: { title: { display: true, text: 'y', color: tick }, grid: { color: grid }, ticks: { color: tick } },
        }
      }
    });

    function fmt(v) {
      if (Number.isNaN(v)) return 'NaN';
      if (!Number.isFinite(v)) return v > 0 ? '+∞' : '-∞';
      const a = Math.abs(v);
      if (a !== 0 && (a < 0.01 || a >= 1000)) return v.toExponential(2);
      return v.toFixed(3);
    }

    function compute() {
      const i = +document.getElementById('ee-ode-select').value;
      const h = +document.getElementById('ee-h-slider').value;
      const tmax = +document.getElementById('ee-t-slider').value;
      const showExplicit = document.getElementById('ee-compare-toggle').checked;
      const ode = ODES[i];

      const exactPts = [];
      const N = 240;
      for (let k = 0; k <= N; k++) {
        const t = (tmax * k) / N;
        exactPts.push({ x: t, y: ode.exact(t) });
      }

      const exactMax = exactPts.reduce((m, p) => Math.max(m, Math.abs(p.y)), 0);
      const threshold = Math.max(exactMax * 15, 8);

      const impPts = [{ x: 0, y: ode.y0 }];
      let t = 0, y = ode.y0;
      let maxErr = 0;
      let lastFinite = y;
      let guard = 0;
      while (t < tmax - 1e-9 && guard++ < 10000) {
        const step = Math.min(h, tmax - t);
        y = ode.implicit(t, y, step);
        t = t + step;
        if (!Number.isFinite(y) || Math.abs(y) > threshold) {
          impPts.push({ x: t, y: NaN });
          break;
        }
        impPts.push({ x: t, y: y });
        lastFinite = y;
        const err = Math.abs(y - ode.exact(t));
        if (err > maxErr) maxErr = err;
      }

      const expPts = [];
      if (showExplicit) {
        let te = 0, ye = ode.y0;
        expPts.push({ x: te, y: ye });
        let g2 = 0;
        while (te < tmax - 1e-9 && g2++ < 10000) {
          const step = Math.min(h, tmax - te);
          ye = ye + step * ode.f(te, ye);
          te = te + step;
          if (!Number.isFinite(ye) || Math.abs(ye) > threshold) {
            expPts.push({ x: te, y: NaN });
            break;
          }
          expPts.push({ x: te, y: ye });
        }
      }

      chart.data.datasets[0].data = exactPts;
      chart.data.datasets[1].data = impPts;
      chart.data.datasets[1].pointRadius = impPts.length > 60 ? 2 : (impPts.length > 30 ? 3 : 4);
      chart.data.datasets[2].data = expPts;
      chart.data.datasets[2].hidden = !showExplicit;
      chart.data.datasets[2].pointRadius = expPts.length > 60 ? 2 : (expPts.length > 30 ? 3 : 4);
      chart.update('none');

      document.getElementById('ee-h-out').textContent = h.toFixed(2);
      document.getElementById('ee-t-out').textContent = tmax.toFixed(1);
      document.getElementById('ee-stat-steps').textContent = String(impPts.length - 1);
      document.getElementById('ee-stat-maxerr').textContent = fmt(maxErr);
      document.getElementById('ee-stat-final').textContent = fmt(lastFinite);

      document.getElementById('ee-explicit-legend').style.display = showExplicit ? 'inline-flex' : 'none';
    }

    document.getElementById('ee-ode-select').addEventListener('change', compute);
    document.getElementById('ee-h-slider').addEventListener('input', compute);
    document.getElementById('ee-t-slider').addEventListener('input', compute);
    document.getElementById('ee-compare-toggle').addEventListener('change', compute);
    compute();
  })();
  </script>
  <figcaption>Interactive ODE explorer: pick an initial-value problem, adjust the step size $h$ and the final time $t_{\max}$, and toggle the explicit-Euler overlay. Try the stiff problem $\dot y=-10y$ at large $h$: implicit Euler stays close to the exact solution, explicit Euler oscillates and blows up. Stats show step count, the maximum absolute error of implicit Euler against the exact solution, and the final value $y(t_{\max})$.</figcaption>
</figure>

### Implicit Euler step = proximal step (the gradient-flow connection)

Apply backward Euler to a gradient flow $\dot x=-\nabla E(x)$:

$$
\frac{x_{k+1}-x_k}{h}=-\nabla E(x_{k+1})\quad\Longleftrightarrow\quad x_{k+1}+h\,\nabla E(x_{k+1})=x_k.\qquad(\heartsuit)
$$

The magic identity: $(\heartsuit)$ is exactly the first-order optimality condition of the optimization problem

$$
\boxed{\quad x_{k+1}\;\in\;\arg\min_{y\in\mathbb R^N}\;\Bigl\lbrace\,E(y)+\frac{1}{2h}\,\|y-x_k\|^2\,\Bigr\rbrace.\quad}
$$

Indeed, differentiating the bracket in $y$ yields $\nabla E(y)+\frac1h(y-x_k)$, and zeroing this at $y=x_{k+1}$ recovers $(\heartsuit)$.

So **one implicit Euler step = one minimization step** of "energy plus quadratic penalty for moving." The map $x_k\mapsto x_{k+1}$ is called the **proximal map** of $E$, often denoted $\mathrm{prox}_{hE}(x_k)$. The discrete trajectory $x_0,x_1,x_2,\dots$ is the **minimizing-movement scheme** in finite dimensions, and the **JKO scheme** (after Jordan–Kinderlehrer–Otto) on Wasserstein space.

The reason this matters for non-smooth gradient-flow theory:

* The minimization problem makes sense **even when $\nabla E$ does not exist or is not Lipschitz**, provided $E$ is convex, proper, lsc — the quadratic penalty makes the bracketed functional strictly convex and coercive, so $x_{k+1}$ exists and is unique.
* In the non-smooth case the optimality condition becomes the *differential inclusion*

  $$
  \frac{x_{k+1}-x_k}{h}\;\in\;-\partial E(x_{k+1}),
  $$

  i.e. the time-discretisation of the subgradient flow.
* As $h\downarrow 0$, the piecewise-constant or piecewise-linear interpolations of the iterates converge (under mild hypotheses) to a solution of the continuous gradient flow — *even if classical ODE theory has no purchase*. The implicit Euler scheme is thereby the **analytical** tool for proving existence of gradient flows of irregular energies.

### Concrete non-smooth case: $E(x)=|x|$ and soft-thresholding

This is the canonical non-smooth example: $\nabla E$ does not exist at $0$, so forward Euler cannot even be defined there. The proximal step is solvable in closed form:

$$
\mathrm{prox}_{h|\cdot|}(x)=\arg\min_y\Bigl\lbrace |y|+\tfrac{1}{2h}(y-x)^2\Bigr\rbrace=\mathrm{sign}(x)\cdot\max(|x|-h,\,0).
$$

This is the **soft-thresholding operator** — a workhorse of compressed sensing, LASSO, and many proximal-gradient algorithms. Iterated from $x_0$ with step $h$, it descends linearly at speed $1$ toward $0$ and then *stops*:

$$
2\;\mapsto\;1.6\;\mapsto\;1.2\;\mapsto\;0.8\;\mapsto\;0.4\;\mapsto\;0\;\mapsto\;0\;\mapsto\;\cdots
$$

(starting from $x_0=2$ with $h=0.4$). This matches the exact subgradient flow $x(t)=\max(2-t,0)$ on the nose, hitting the kink and parking there.

<figure>
  <img src="{{ '/assets/images/notes/random/euler_softthreshold.png' | relative_url }}" alt="Two panels: left shows the soft-thresholding map as the piecewise-linear function with a flat zero region near the origin; right shows iterates of implicit Euler from x_0 equals 2 descending linearly to zero and remaining at zero, matching the exact subgradient flow" loading="lazy">
  <figcaption>Implicit Euler for $E(x)=|x|$. Left: one step is the soft-thresholding operator $x\mapsto\mathrm{sign}(x)\max(|x|-h,0)$, which collapses the band $|x|\le h$ to zero. Right: iterates from $x_0=2$ with $h=0.4$ exactly track the subgradient flow $\max(2-t,0)$, descend at unit speed, and stop dead at the kink. Forward Euler cannot be applied here at all.</figcaption>
</figure>

### Comparison summary

| property | forward Euler | backward Euler |
|---|---|---|
| update | explicit formula | requires solving an equation |
| cost per step | cheap | expensive (one nonlinear solve) |
| stability for stiff problems | requires tiny $h$ | unconditionally stable (A-stable) |
| order of accuracy | first-order | first-order |
| works when $\nabla E$ Lipschitz | yes | yes |
| works when $\nabla E$ non-Lipschitz / non-existent | no | yes (proximal step) |
| variational interpretation | none | minimizing-movement / JKO |
| analytical role | numerical only | proves existence in non-smooth gradient-flow theory |

### One-line summary

> Forward Euler $x_{k+1}=x_k+h\,f(x_k)$ uses the slope at the *start* of the step; backward Euler $x_{k+1}=x_k+h\,f(x_{k+1})$ uses the slope at the *end*, requiring an equation solve. The reward is **unconditional stability** (A-stability) — and in the gradient-flow case, the backward-Euler step coincides with the **proximal step** $x_{k+1}=\arg\min_y\{E(y)+\frac1{2h}\|y-x_k\|^2\}$, which is well-defined for convex $E$ regardless of smoothness, and is the analytical engine that proves existence of gradient flows when classical ODE theory cannot.
