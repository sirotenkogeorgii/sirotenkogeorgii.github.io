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

### Where the Hopfield “$\frac{1}{2$” comes from

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

RBM energy in 0/1:

$$E(v,h) = -b^\top v - c^\top h - v^\top W h$$

You can map $v,h$ to $\pm 1$ spins the same way; you’ll get couplings scaled by $1/4$ and extra field terms coming from row/column sums of $W$.