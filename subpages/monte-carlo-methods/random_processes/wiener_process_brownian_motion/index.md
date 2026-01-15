---
title: Wiener Process and Brownian Motion
layout: default
noindex: true
---

# Wiener Process and Brownian Motion

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Wiener process)</span></p>

A **Wiener process with a parameter $\sigma > 0$** is a stochastic process $W = \lbrace W_t, t \ge 0\rbrace$ defined by these properties:

1. **Independent increments:**
   For any $t_1 < t_2 \le t_3 < t_4$, the increments

   $$W_{t_4} - W_{t_3} \quad \text{and} \quad W_{t_2} - W_{t_1}$$

   are independent. Equivalently, for $t > s$, the increment $W_t - W_s$ does not depend on the past trajectory $\lbrace W_u, 0 \le u \le s\rbrace$. The inequalities $t_1 < t_2 \le t_3 < t_4$ mean that those two intervals ($t_1,t_2$) and ($t_3,t_4$) are not overlapping, because independence of increments is defined on non-overlapping intervals.

2. **Gaussian stationary increments:**
   For all $t \ge s \ge 0$,
   
   $$W_t - W_s \sim \mathcal{N}(0, \sigma^2\lvert t-s\rvert)$$

   Variance of the difference increases linearly. 
   
3. **Continuous paths:**
   The sample paths of $\lbrace W_t\rbrace$ are continuous, and $W_0 = 0$ almost everywhere.

</div>

> **Convention**: A *Wiener process with a parameter $\sigma > 0$* is often called just a *Wiener process*.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The Wiener process is fundamental in probability theory and serves as the building block for many other stochastic processes. It can be interpreted as the continuous-time version of a random walk. Two example paths over ([0,1]) are shown in Figure 4.9.

</div>


<div class="gd-grid">
  <figure>
    <img src="{{ 'assets/images/notes/monte-carlo-methods/wiener_process_on_interval.png' | relative_url }}" alt="a" loading="lazy">
    <figcaption>Two example paths over the interval [0,1]</figcaption>
  </figure>
  <figure>
    <img src="{{ 'assets/images/notes/monte-carlo-methods/wiener_process_with_different_sigma.png' | relative_url }}" alt="b" loading="lazy">
    <figcaption>Wiener process with different sigmas</figcaption>
  </figure>
  <figure>
    <img src="{{ 'assets/images/notes/monte-carlo-methods/wiener_process_3d.png' | relative_url }}" alt="c" loading="lazy">
    <figcaption>3D Wiener process</figcaption>
  </figure>
</div>

### Wiener Process belongs to a class of Gaussian Processes

Let us find a pdf of $n$-dimensional Wiener process. For this purpose we create a vector $Y=(W(t_1), \dots, W(t_n))$ composed from the Wiener process sections at time points $t_1 < \dots < t_n$. Recall that the **time sections of Wiener process are dependent** but the **increments on non-overlapping intervals are independent** . We also create a vector $Z = (W(t_1),W(t_2)-W(t_1),\dots,W(t_n)-W(t_{n-1}))$.

Because this vector $Z$ is composed from the independent normal random variables, the pdf of the multivariate random variable $Z$ is the following:

$$f_Z(z)=\prod_{k=1}^n \dfrac{1}{\sqrt{2\pi\sigma^2(t_k-t_{k-1})}}\exp \left(-\frac{z_k^2}{2\sigma^2(t_k-t_{k-1})} \right), \quad z=(z_1,\dots,z_n),$$

where $t_0 = 0$. We also notice that $Y=AZ$, where $A\in \lbrace 0,1 \rbrace^n$ is an lower triangular matrix full of ones. Using the change of variables we get the connection between the pdfs of $Y$ and $Z$:

$$f_Y(y) = \frac{1}{\lvert \text{det }A \rvert}f_Z(z=A^{-1}y), \quad y=(y_1,\dots,y_n)$$
$$f_Y(y) = \prod_{k=1}^n \dfrac{1}{\sqrt{2\pi\sigma^2(t_k-t_{k-1})}}\exp \left(-\dfrac{(y_k-y_{k-1})^2}{2\sigma^2(t_k-t_{k-1})} \right), \quad t_0=0,y_0=0.$$

$$
A^{-1}=\begin{pmatrix}
1 & 0 & 0 & \cdots & 0\\
-1& 1 & 0 & \cdots & 0\\
0 & -1& 1 & \cdots & 0\\
\vdots & \ddots & \ddots & \ddots & \vdots\\
0 & \cdots & 0 & -1 & 1
\end{pmatrix}.
$$

Recall that linear transformation converts normal random vectors into normal random vectors again (preserves normality) and $\text{det}(A)=\text{det}(A^\top)$. We just showed that any vector composed from the time sections of the Wiener process is normally distributed. In addition, the following holds:
* $\mathbb{E}[Y]=\mathbb{E}[AZ] = A\mathbb{E}[Z] = 0$
* $\text{Cov}(Y)=\text{Cov}(AZ) = A\text{Cov}(Z)A^\top = \sigma^2\|\min(t_i,t_j) \|^n_{i,j=1}$.

The matrix $\text{Cov}(Y)=\sigma^2\|\min(t_i,t_j) \|^n_{i,j=1}$ means $\text{Cov}(Y)_{ij} = \sigma^2\min(t_i,t_j)$. 

*proof:*

$$\text{Cov}(Y)_{ij} = \text{Cov}(Y_i,Y_j) = \text{Cov}(W(t_i),W(t_j)) = (A\text{Cov}(Z)A^\top)_{ij}= \sigma^2\sum_{k=1}^{\min(i,j)}\Delta t_k = \sigma^2\min(t_i,t_j)$$

$$\Delta t_0 = t_0$$

$$\Delta t_1 = t_1 - t_0$$

$$\Delta t_2 = t_2 - t_1$$

$$\dots$$

$$\implies \sum_{k=1}^n \Delta t_n = t_n$$

$$\square$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Gaussian Process)</span></p>

A stochastic process $\lbrace X(t),t\ge0 \rbrace$ is Gaussian, if for any $n\ge 1$ and time point $0 \le t_1 < \dots < t_n$ vector $(X(t_1),\dots,X(t_n))$ is a normal random vector.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Wiener Process)</span></p>

A stochastic process $\lbrace W(t),t\ge 0 \rbrace$ is a Wiener process *if and only if*
* **(a)** $W(t)$ is a Gaussian Process.
* **(b)** $\mathbb{E}[W(t)]=0$ for any $t\ge 0$.
* **(c)** $\text{Cov}(W(t),W(s))=R_W(t,s) = \sigma^2\min(t,s)$ for any $t,s\ge 0$.

</div>

*proof:*

* ($\Rightarrow$) Let a stochastic process $\lbrace W(t),t\ge 0 \rbrace$ be a Wiener process. Then we have already shown above tha the process is Gaussian (a). We have shown that $\mathbb{E}[Y]=0$, meaning that taking $n=1$ gives us $\mathbb{E}[W(t)]=0$ for any $t\ge 0$ (b). Also we have shown that at any time points $t_1 < \dots < t_n$ the covariance between time sections $t_i$ and $t_j$ is $\min(t,s)$ and in the case $t=s$ we have $\text{Cov}(W(t),W(t))=\text{Var}(W(t))=\text{Var}(W(t)-W(0))=\sigma^2t (c)$.
* ($\Leftarrow$) If the process is Gaussian and $\mathbb{E}[X(t)]=0$ for any time section $t\ge 0$, then $X(t),X(s)~\mathcal{N}(0, \sigma^2)$ and their difference follows Gaussian distribution as well and $\mathbb{E}[X(s)-X(t)] = 0$. $\text{Var}(X(s)-X(t))=\mathbb{E}[(X(s)-X(t))^2] = \mathbb{E}[X(s)^2] - 2\mathbb{E}[X(s)X(t)] + \mathbb{E}[X(t)^2] = \text{Var}(X(s)) + \text{Var}(X(t)) - 2\text{Cov}(X(t),X(s)) = \text{Cov}(X(t),X(t)) + \text{Cov}(X(s),X(s)) - 2\text{Cov}(X(t),X(s)) = \sigma^2t + \sigma^2s - 2\sigma^2\min(t,s) = \sigma^2\lvert t-s \rvert$ assuming $s\le t \implies (X(t)-X(s))~\mathcal{N}(0, \sigma^2\lvert t-s \rvert)$. To show independence of increments, we consider two random inrements $X(t_1) - X(t_2)$ and $X(3) - X(4)$ for $t_1 < t_2 \le t_3 < t_4$. Consider their covariance $\text{Cov}(X(t_1) - X(t_2), X(t_3) - X(t_4)) = \mathcal{E}[(X(t_1) - X(t_2))(X(t_3) - X(t_4))] = \mathcal{E}[X(t_1)X(t_3)] - \mathcal{E}[X(t_1)X(t_4)] - \mathcal{E}[X(t_2)X(t_3)] + \mathcal{E}[X(t_2)X(t_4)] = \text{Cov}(X(t_1),X(t_3)) - \text{Cov}(X(t_1),X(t_4)) - \text{Cov}(X(t_2),X(t_3)) + \text{Cov}(X(t_2),X(t_4)) = \sigma^2(\min(t_1,t_3) - \min(t_1,t_4) - \min(t_2,t_3) + \min(t_2,t_4)) = \sigma^2(t_1 - t_1 - t_2 + t_2) = 0$. For a jointly Gaussian vector of increments $(X(t_1) - X(t_2), X(t_3) - X(t_4))$, **pairwise uncorrelated $\implies$ independent**, meaning that vector of increments is independent. It is trivial to extend the bivariate vector of increments to arbitrary multivariate vector of increments showing pairwise uncorrelatation $\implies$ independence.

#### Wiener Process as a Random Walk

It is important to note that Wiener process is possible to view and even define as a random walk. 

> **Question**: Is  Wiener process a continuous-time random walk?
> **Answer**: A Wiener process (Brownian motion) is the continuous-time, continuous-state limit of a discrete random walk, acting as a fundamental model for random movement with independent, normally distributed steps and continuous paths, while a continuous random walk often refers to a generalized random walk with random step sizes and/or random waiting times (CTRW), but the true idealization connecting to the Wiener process is via Donsker's theorem, showing a scaled discrete random walk converges to Brownian motion, making the Wiener process an idealized model for physical phenomena like diffusion. 

**Difference between Wiener process and continuous-time random walk:**
<div class="accordion">
  <details markdown="1">
    <summary>Answer</summary>

Not in the usual (technical) sense.

### Continuous-time random walk (CTRW)

A **CTRW** typically means:

* the process **jumps** at random times (often given by i.i.d. waiting times),
* between jumps it stays **constant**,
* jump sizes are often i.i.d.

So a CTRW path is **piecewise constant with jumps** (càdlàg), not continuous.

### Wiener process (Brownian motion)

A **Wiener process**:

* has **continuous sample paths** (no jumps),
* has independent, stationary **Gaussian** increments.

So Brownian motion is **continuous-time**, but it is **not a jump walk** like a CTRW.

### Relationship

Brownian motion is the **diffusion limit** of random walks / CTRWs under appropriate scaling:

* speed up time,
* shrink step sizes,
* (and in CTRW cases, choose waiting times so the number of jumps grows like (t/\Delta)).

Then the rescaled jump process can converge (weakly) to Brownian motion.

So: **Brownian motion is a continuous-time limit of random walks**, but it’s **not itself** a continuous-time random walk (in the jump-process sense).

  </details>
</div>

Assume we have a stock whose price changes by jumps at times $1/N,2/N,3/N,\dots$ with $N\ge 1$. For a fixed $t>0$, define the price (or accumulated return) at time $t$ by

$$X_N(t)=\sum_{i=1}^{\lfloor Nt\rfloor}\Delta X_i,$$

where the increments are

$$
\Delta X_i=
\begin{cases}
\dfrac{\sigma}{\sqrt{N}}, & \text{with probability } \tfrac12, \\
-\dfrac{\sigma}{\sqrt{N}}, & \text{with probability } \tfrac12,
\end{cases}
$$

and $\lbrace\Delta X_i\rbrace$ are independent and identically distributed. Then $\mathbb E[\Delta X_i]=0$ and $\mathrm{Var}(\Delta X_i)=\sigma^2/N$. By the central limit theorem, for $t>0$,

$$
\frac{X_N(t)}{\left(\sigma/\sqrt{N}\right)\sqrt{\lfloor Nt\rfloor}}
\ \xRightarrow[N\to\infty]{d}\
Z,\qquad Z\sim \mathcal N(0,1).
$$

Since $\lfloor Nt\rfloor/N\to t$ as $N\to\infty$, it follows that

$$X_N(t)\ \xRightarrow[N\to\infty]{d}\ X(t),\qquad X(t)\sim \mathcal N(0,\sigma^2 t),\ \ t>0.$$

<figure>
   <img src="{{ 'assets/images/notes/monte-carlo-methods/sketch_random_walk_wiener_process.jpeg' | relative_url }}" alt="a" loading="lazy">
</figure>

In fact, the limiting object $X(t)$ is not just a collection of Gaussian random variables indexed by $t$: the limit process is a Wiener process $W(t)$ with diffusion parameter $\sigma$. Proving this requires the notion of **weak convergence of stochastic processes** (equivalently, weak convergence of the associated measures). A detailed treatment, including a proof that the above discrete-time process converges to Brownian motion, can be found in standard monographs on weak convergence.

In functional analysis, **weak convergence** of an abstract sequence $x_n$ to an element $x$ in a linear topological space $E$ means that the numerical sequence $f(x_n)$ converges to $f(x)$ for every continuous linear functional $f$ from the dual space $E^*$.

In probability theory, when we speak about weak convergence of random variables $\xi_n$ to $\xi$, for simplicity this is often expressed as **pointwise convergence of distribution functions**:

$$F_n(x)\to F(x)$$

at every point $x$ where $F$ is continuous. More fundamentally, however, weak convergence $\xi_n \Rightarrow \xi$ is understood as **weak convergence of the corresponding probability measures** $P_n \Rightarrow P$, where $P_n$ and $P$ are the measures induced by the distribution functions $F_n$ and $F$. These measures can be viewed as elements of a suitable linear topological space of measures.

Without going into the details of how that space is constructed, note that weak convergence of measures can be defined in several equivalent ways. For example, if $P_n$ and $P$ are probability measures on a measurable (typically metric) space $(S,\mathcal F)$, then $P_n \Rightarrow P$ can be characterized by the convergence of expectations

$$\mathbb E_{P_n}[f] \to \mathbb E_{P}[f], \qquad n\to\infty,$$

for every bounded continuous function $f$ on $S$. Here $\mathbb E_{P_n}$ and $\mathbb E_P$ denote expectations taken with respect to $P_n$ and $P$, respectively.

**Intuition for weak convergence to the probability measure:**
* You define a **sequence (family of processes)** $\lbrace X^{(n)}\rbrace_{n\ge1}$, where $n$ controls something like “more steps”, “finer time grid”, or a rescaling, in general it controls something where sort of the approximation/rescaling. Examples: process parametrised with $n$, where length of the partial sums $S_n$ in the case of CLT or finer time grid for convergence to Wiener process of discrete-time random walk:


  * **CLT (random variables):** $X^{(n)} := \dfrac{S_n - n\mu}{\sigma\sqrt{n}}$ with $S_n=\sum_{k=1}^n \xi_k$. Here $n$ is the number of summands.

  * **Random walk $\to$ Wiener process (processes):** define a path-valued object, e.g.
    
    $$X^{(n)}(t) := \frac{1}{\sqrt{n}}S_{\lfloor nt\rfloor},\quad t\in[0,1],$$

    where increasing $n$ means a **finer time grid** and the right **diffusive scaling**. Then $X^{(n)}\Rightarrow W$ (weakly on a path space).

  Small nuance: “length of the process” isn’t quite the point; it’s more that $n$ changes the **scale/resolution** (number of steps and normalization) so that a nontrivial limit exists.

* As $n\to\infty$, you look at the **distribution (law)** of $X^{(n)}$ as a random element of a path space (e.g. $C[0,T]$ or $D[0,T]$).
* If $X^{(n)} \Rightarrow X$, that means the laws converge to a **limit law**—analogous to the CLT where the law of the shifted/rescaled sample mean converges to $\mathcal N(0,1)$.

Example (process-level CLT / Donsker):

$$X^{(n)}(t)=\frac{1}{\sqrt{n}}\sum_{k=1}^{\lfloor nt\rfloor}\xi_k,\quad t\in[0,1],$$

then

$$X^{(n)} \Rightarrow W$$

where $W$ is Brownian motion. This is exactly the “CLT, but for whole paths”.

One nuance: for processes you usually need (i) convergence of finite-dimensional distributions and (ii) tightness/regularity to conclude full process convergence. But conceptually, it’s the same “rescaled objects $\to$ limiting distribution” idea as the CLT.


**Is not weak convergence of a stochastic process similar to ergodicity?**
<div class="accordion">
  <details markdown="1">
    <summary>Answer</summary>

No—**weak convergence** and **ergodicity** are different concepts.

* **Weak convergence of a stochastic process** means the *laws* (distributions) of processes $X^{(n)}$ converge to the law of a limit process $X$. Formally, $X^{(n)} \Rightarrow X$ as random elements in a function space (e.g. $C[0,T]$ or $D[0,T]$), meaning expectations of bounded continuous functionals converge:
  
  $$\mathbb E[F(X^{(n)})]\to \mathbb E[F(X)].$$

The convergence is **not about pairing up sample paths across runs**. It’s about the **distributions (laws)** of the processes.

Think of each process $X^{(n)}$ as a random object taking values in a path space (e.g. $C[0,T]$ or $D[0,T]$). “$X^{(n)}$ converges to $X$” means:

* for any “nice” functional $F$ that looks at a whole path (e.g. $F(\omega)=\max_{t\le T}\omega(t)$ or $F(\omega)=\omega(t_1)\omega(t_2)$),
  
  $$\mathbb E[F(X^{(n)})] \to \mathbb E[F(X)].$$
  
  That’s weak convergence on path space.

Equivalently, it means **finite-dimensional distributions** converge:

$$(X^{(n)}_{t_1},\dots,X^{(n)}_{t_k}) \Rightarrow (X_{t_1},\dots,X_{t_k})$$

for all choices of times $t_1,\dots,t_k$, plus a tightness/regularity condition to upgrade this to process-level convergence.

Independence between *different runs* is irrelevant: you could simulate $X^{(1)}, X^{(2)}, \dots$ independently and still say their **laws** approach a limit law.

The “sequence” is not a sequence of *independent runs*. It’s a sequence of **random objects indexed by $n$**, and the convergence is about their **distributions**.

### A concrete sequence you can point to

Take i.i.d. $\xi_1,\xi_2,\dots$ with $\mathbb E[\xi_i]=0$, $\mathrm{Var}(\xi_i)=1$. Define

$$X_n :=\frac{1}{\sqrt{n}}\sum_{i=1}^n \xi_i.$$

That’s a genuine sequence $X_1,X_2,X_3,\dots$ of random variables (all defined using the *same* underlying $\xi_i$’s). They are **not independent across $n$**—they’re built from the same data.

The CLT says:

$$X_n \Rightarrow Z,\quad Z\sim \mathcal N(0,1),$$

meaning the **law** of $X_n$ approaches the law of $Z$.

So the converging sequence is:

$$\mathcal L(X_n)\ \to\ \mathcal L(Z).$$

### Same idea for processes

Define a random function (a path) by

$$X^{(n)}(t) := \frac{1}{\sqrt{n}}\sum_{i=1}^{\lfloor nt\rfloor}\xi_i,\quad t\in[0,1].$$

Then $X^{(n)}$ is a random element of a path space (like $D[0,1]$). Donsker’s theorem says

$$X^{(n)} \Rightarrow W$$

in distribution on that path space.

### Why “independent runs” aren’t needed

When people *simulate*, they might generate independent sample paths to **estimate** probabilities/expectations. But the *mathematical convergence* $X^{(n)}\Rightarrow X$ is a statement about laws, and you can (and often do) define all $X^{(n)}$ on one probability space using the same underlying randomness, like above.

If you tell me the exact approximation scheme you’re using (random walk with step $\Delta=1/n$, Euler–Maruyama, etc.), I can write down the explicit $X^{(n)}$ sequence in your notation.


* **Ergodicity** is a property of a *single* (usually stationary) process: time averages along almost every sample path equal ensemble averages. For example, for suitable $f$,
  
  $$\frac{1}{T}\int_0^T f(X_t),dt \xrightarrow[T\to\infty]{} \mathbb E[f(X_0)] \quad \text{(a.s. or in probability)}.$$
  
They answer different questions: weak convergence is about **limits of distributions across $n$**; ergodicity is about **long-time behavior within one process**.

  </details>
</div>


#### Law of the iterated logarithm

$\textbf{Theorem (Law of the iterated logarithm (Kolmogorov’s LIL)):}$ Let $X_1,X_2,\dots$ be i.i.d. random variables with $\mathbb{E}[X_i]=0$ and $\text{Var}(X_i)=\simga^2\in (0,\intfy)$ and define the patial sums $S_n = \sum_{i=1}^n Y_i$. Then

$$\limsup_{n\to \infty} \dfrac{\lvert S_n \rvert}{\sigma\sqrt{2n\log\log n}} = 1 \quad \text{a.s}$$
$$\liminf_{n\to \infty} \dfrac{\lvert S_n \rvert}{\sigma\sqrt{2n\log\log n}} = -1 \quad \text{a.s}$$

So "eventually", the random walk keeps hitting values as large as about $\sigma\sqrt{2n\log\log n}$ and as small as its negative, infinitely often — but it doesn’t exceed this envelope by more than a *negligible factor*, almost surely.

The **law of the iterated logarithm** describes the magnitude of the fluctuations of a *random walk*, the **largest typical fluctuations** of a random walk (or sum of i.i.d. random variables) on an almost-sure scale—sitting exactly between what the Law of Large Numbers and the Central Limit Theorem tell you.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The "iterated logarithm" is the $\log\log n$ term: it grows extremely slowly, but it’s the precise correction needed to describe the almost-sure extremal oscillations.

</div>

**Intuition: where it fits vs LLN and CLT**
* **LLN**: $S_n/n \to 0$ almost surely (for mean 0). This says the average stabilizes, but doesn’t quantify fine fluctuations.
* **CLT**: $S_n/(\sigma\sqrt{n}) \Rightarrow \mathcal{N}(0,1)$. This describes distribution at a *fixed* $n$-scale, not the pathwise extremes over all large $n$.
* **LIL**: gives the *sharp almost-sure growth rate* of the maxima/minima of $S_n$. It says the path oscillates inside an envelope of size $\sqrt{n\log\log n}$ with exact constants.

A realization of a Wiener process could be viewed as a sum of independent increments with a fixed variance if you take a fix time step. We can apply LIL on Wiener process: 

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">((LIL for Wiener Process))</span></p>

Let $W_t$ be a Wiener process and fix $\Delta>0$. Define increments

$$X_k := W_{k\Delta}-W_{(k-1)\Delta}.$$

Then:

* $X_k$ are **independent** and **identically distributed** (stationary independent increments),
* $X_k \sim \mathcal N(0,\Delta)$, so $\mathbb E[X_k]=0$ and $\mathrm{Var}(X_k)=\Delta$.

And $W_{n\Delta}=\sum_{k=1}^n X_k.$ So Kolmogorov’s LIL for i.i.d. sums gives, almost surely,

$$
\limsup_{n\to\infty}\frac{W_{n\Delta}}{\sigma\sqrt{\Delta},\sqrt{2n\log\log n}}=1,
\qquad
\liminf_{n\to\infty}\frac{W_{n\Delta}}{\sigma\sqrt{\Delta},\sqrt{2n\log\log n}}=-1.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

The law of the iterated logarithm (LIL) is an **asymptotic result**, so it’s fundamentally about what happens as $n\to\infty$ (or $t\to\infty$ for Brownian motion).

</div>

<div class="gd-grid">
  <figure>
    <img src="{{ 'assets/images/notes/monte-carlo-methods/LIL_wiki.png' | relative_url }}" alt="GP1" loading="lazy">
  </figure>
  <figure>
    <img src="{{ 'assets/images/notes/monte-carlo-methods/LIL_mathoverflow.png' | relative_url }}" alt="GP3" loading="lazy">
  </figure>
</div>

#### Bachelier Theorem

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Bachelier)</span></p>

For any $T$, $x\ge 0$:

$$\mathbb{P}\left(\max_{t\in [0,T]} W(t)\ge x\right) = 2\mathbb{P}(W(T)\ge x) = \mathbb{P}(\lvert W(T) \rvert \ge x)$$

</div>

<figure>
   <img src="{{ 'assets/images/notes/monte-carlo-methods/wiener_process_area.png' | relative_url }}" alt="a" loading="lazy">
   <figcaption>Realizations (trajectories) of Wiener process with a probability 99.7% is in area between $y=\pm 3\sqrt{t}$</figcaption>
</figure>


### Rest

A Wiener process could be viewed as a homogeneous Gaussian process $X(t)$ with independent increments. Also Wiener process serves as one of the models of **Brownian motion**.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name"></span></p>

A **homogeneous Gaussian process** means a stationary (translation-invariant) Gaussian process — i.e., shifting the input space doesn’t change its probabilistic behavior. "Stationary" means the probabilistic behavior of a process doesn’t change when you shift time/space.

</div>

More details on homogeneous GP:

<div class="accordion">
  <details markdown="1">
    <summary>homogeneous Gaussian process</summary>

A **homogeneous Gaussian process** usually means a **stationary (translation-invariant) Gaussian process** — i.e., shifting the input space doesn’t change its probabilistic behavior.

Formally, a (real-valued) Gaussian process $\lbrace X(t): t \in \mathbb{R}^d\rbrace$ is **homogeneous** if:

1. **Constant mean**
   
   $$\mathbb{E}[X(t)] = m \quad \text{for all } t$$

2. **Covariance depends only on displacement**
   
   $$\mathrm{Cov}(X(s), X(t)) = C(t-s)$$
   
   So the covariance is a function of the **difference** $h=t-s$, not of the absolute locations.

Because it’s Gaussian, specifying the mean and covariance fully determines all finite-dimensional distributions.

  </details>
</div>


The key consequences for $W = \lbrace W_t, t \ge 0\rbrace$ include:

1. **Gaussian process characterization:**
   $W$ is a Gaussian process with
   
   $$\mathbb{E}[W_t] = 0, \qquad \text{Cov}(W_s, W_t) = \min(s,t)$$
   
   It is the unique Gaussian process with continuous paths having these mean and covariance properties.

2. **Markov property:**
   $W$ is a time-homogeneous strong Markov process. In particular, for any finite stopping time $\tau$ and any $t \ge 0$,
   
   $$(W_{\tau+t}\mid W_u, u \le \tau) \sim (W_{\tau+t}\mid W_\tau)$$
   
    That line is basically saying: once you know the value at time $\tau$, the whole past before $\tau$ gives you no extra information about the future after $\tau$.

The simulation method below relies on the **Markov** and **Gaussian** properties of the Wiener process.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Simulating a Wiener Process)</span></p>

1. Choose a set of distinct time points
   
   $$0 = t_0 < t_1 < t_2 < \cdots < t_n$$
   
   at which you want to simulate the process.

2. Generate independent standard normal variables
   
   $$Z_1,\ldots,Z_n \stackrel{iid}{\sim} \mathcal{N}(0,1),$$
   
   and compute
   
   $$W_{t_k} = \sum_{i=1}^k \sqrt{t_i - t_{i-1}}, Z_i, \quad k = 1,\ldots,n$$

</div>

This method is **exact** in the sense that the values $\lbrace W_{t_k}\rbrace$ are sampled from the correct distributions. However, it only produces a **discrete set of points** from the underlying continuous-time process.

To build a continuous approximation of the path, one can use **linear interpolation** between successive simulated points. Specifically, on each interval $[t_{k-1}, t_k]$, approximate $\lbrace W_s\rbrace$ by

$$\widehat{W}_s = \frac{W_{t_k}(s - t_{k-1}) + W_{t_{k-1}}(t_k - s)} {t_k - t_{k-1}}, \qquad s \in [t_{k-1}, t_k].$$

The path can also be refined adaptively using a **Brownian bridge**.

A process $\lbrace B_t, t \ge 0\rbrace$ defined by

$$B_t = \mu t + \sigma W_t, \qquad t \ge 0,$$

where $\lbrace W_t\rbrace$ is a Wiener process, is called a **Brownian motion** with **drift** $\mu$ and **diffusion coefficient** $\sigma^2$.
When $\mu = 0$ and $\sigma^2 = 1$, this reduces to **standard Brownian motion** (i.e., the Wiener process).

The simulation of Brownian motion at times $t_1,\ldots,t_n$ follows directly from this definition.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Simulating Brownian Motion)</span></p>

1. First generate $W_{t_1},\ldots,W_{t_n}$ from a Wiener process.

2. Then set
   
   $$B_{t_i} = \mu t_i + \sigma W_{t_i}, \quad i = 1,\ldots,n.$$

</div>

If $\lbrace W_{t,i}, t \ge 0\rbrace$ for $i = 1,\ldots,n$ are independent Wiener processes and

$$W_t = (W_{t,1},\ldots,W_{t,n}),$$

then $\lbrace W_t, t \ge 0\rbrace$ is called an **$n$-dimensional Wiener process**.

