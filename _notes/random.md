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

Given the standard enumeration $M_0,M_1,\dots$, of all Turing Machines, it sufficies to choose $h(e,i)$ such that $M_{h(e,i)}$ on input $x$ first simulates M_e on input $i$, and if this computation terminates with output $t$, simulates $M_t$ on input $x$. This involves hardwaring $e$ and $i$ into $M(e,i)$.

Hardwiring $e$ and $i$ into $M_{h(e,i)}$ just means:

> Build a *new* Turing machine whose program text contains the *constants* $e$ and $i$, so the machine doesn’t need to read them as input—they are baked into its code.

So $M_{h(e,i)}$ is a machine **parameterized** by $e$ and $i$, but the parameters are stored inside the machine description.

**Concretely, what does $M_{h(e,i)}$ do?**

For each fixed pair $(e,i)$, define a machine $N_{e,i}$ that on input $x$:

1. Simulate $M_e$ on input $i$.
2. If that simulation halts with output $t$, then simulate $M_t$ on input $x$ and output whatever it outputs. We can simulate $M_t$ effectively (jump to it), because in standard enumeration of Turing Machines, the function $i \mapto M_i$ is effective (computable) mapping.
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