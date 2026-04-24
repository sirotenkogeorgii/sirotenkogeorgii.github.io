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

### Riemann–Stieltjes integral

## Lebesgue integral

$\textbf{Theorem (Lebesgue's Criterion for Riemann Integrability):}$ For any function $f: [a,b] \to \mathbb{R}$ the following equivalence holds 

$$f \in R(a,b) \iff f \text{ is bounded and } DC(f) \text{ has measure zero},$$

where $DC(f) := \lbrace x \in M \mid f \text{ is discontinuous in } x  \rbrace$ for $f: M \to \mathbb{R}$.

*proof*: #TODO:

## Partition of an interval

$\textbf{Definition (Partition of an interval):}$ Let $a,b \in \mathbb{R}$ with $a < b$. A **partition** of the interval $[a,b]$ is a finite ordered set

$$P = (a_0,a_1,\dots,a_k)$$

such that

$$a = a_0 < a_1 < \cdots < a_k = b, \qquad k \in \mathbb{N}.$$

## Norm of partition

$\textbf{Definition (Norm of a partition):}$ The **norm** of a partition $P$ is defined by

$$\Delta(P) := \max_{i=1,\dots,k} (a_i - a_{i-1}).$$

## Choice of tags / sample points

$\textbf{Definition (Choice of tags / sample points):}$ Given a partition $P = (a_0,\dots,a_k)$, a **tag vector**

$$\bar t = (t_1,\dots,t_k)$$

is a choice of points such that

$$t_i \in [a_{i-1},a_i], \qquad i=1,\dots,k.$$

## Riemann sum

$\textbf{Definition (Riemann sum):}$ Let $f : [a,b] \to \mathbb{R}$ be an arbitrary function. For a partition $P$ of $[a,b]$ and a corresponding tag vector $\bar t$, the **Riemann sum** of $f$ with respect to $P$ and $\bar t$ is

$$R(P,\bar t,f) := \sum_{i=1}^{k} (a_i - a_{i-1}) f(t_i).$$

## Riemann Integral

$\textbf{Definition (Riemann integral):}$ Let $a<b$ and let $f:[a,b]\to\mathbb{R}$. We say that $f$ is **Riemann integrable** on $[a,b]$, and write $f \in \mathcal{R}(a,b),$ if there exists a real number $L\in\mathbb{R}$ such that $\forall \varepsilon>0\;\exists \delta>0$ s.t. for every partition $P$ of $[a,b]$ and every choice of tags $\bar t$ for $P$,


$$\Delta(P)<\delta \Longrightarrow \bigl|R(P,\bar t,f)-L\bigr|<\varepsilon.$$


In this case, we define

$$\int_a^b f(x)dx := L,$$

and also write

$$(R)\int_a^b f = L.$$

The number $L$ is called the **(Riemann) integral of $f$ over $[a,b]$**.


## Proposition (unbounded functions are bad): 
Function $f: [a,b] \to \mathbb{R}$ is unbounded $\implies$ $f \notin \mathcal{R}(a,b)$

## Zvana 1

Let $f: [a,b] \to \mathbb{R}$ be in $\mathcal{R}(a,b)$. Then $f\in \mathcal{R}(a,x)$ for every $x \in (a,b]$ and the function $F: [a,b]\to\mathbb{R}$ is given as

$$F(x):=\int_a^x f$$

is Lipschitz continuous and $F'(x)=f(x)$ in every point  $x\in [a,b]$ of continuity $f$.

## Zvana 2

Let $f:(a,b)\to\mathbb{R}$ has primitive function $F:(a,b)\to\mathbb{R}$ and let $f\in \mathcal{R}(a,b)$. Then there exists finite limit $F(a):=\lim_{x\to a}F(x)$ and $F(b):=\lim_{x\to b}F(x)$ and

$$(R)\int_a^b f = F(b)-F(a) = (N)\int_a^b f$$

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

<div class="math-callout math-callout--info" markdown="1">
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

<div class="math-callout math-callout--info" markdown="1">
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

cannot distinguish $X$ from $Y$ when $X = Y$ $\mathbb{P}$-a.s. If we treated such $X$ and $Y$ as different objects, then $\|X - Y\|_{L^p} = 0$ would *not* imply $X = Y$, and $\|\cdot\|_{L^p}$ would fail to be a norm. The fix: define $L^p(\Omega, \mathcal{A}, \mathbb{P})$ as the space of *equivalence classes* under the relation

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
