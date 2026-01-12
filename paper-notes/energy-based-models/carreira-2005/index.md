---
title: On Contrastive Divergence Learning
layout: default
noindex: true
---

# On Contrastive Divergence Learning

[link](https://www.cs.toronto.edu/~fritz/absps/cdmiguel.pdf)

### 0) What this paper is about (one-paragraph core)

Maximum-likelihood (ML) learning in energy-based Markov random fields is hard because the gradient needs expectations under the model distribution, which typically requires long MCMC runs to equilibrium. Hinton’s contrastive divergence (CD) idea runs the Markov chain only a few steps starting from the data; it often works well empirically. This paper studies CD’s **fixed points** and **bias**: 
* (i) CD is **biased in general** (its fixed points usually differ from ML’s), but 
* (ii) in empirical tests on small Boltzmann machines where exact computations are possible, the bias is **typically very small**, and 
* (iii) a practical strategy is **CD for most training + a short exact-ML phase** (“CD-ML”) to remove the residual bias at low extra cost. 

---

## 1) Setup and background

### 1.1 Energy-based model and ML gradient

They consider distributions over discrete vectors $x$ (discrete “without loss of generality”):

$$p(x;W)=\frac{1}{Z(W)} e^{-E(x;W)},\qquad Z(W)=\sum_x e^{-E(x;W)}.$$ 

Given iid data $X=\lbrace x_n\rbrace_{n=1}^N$, average log-likelihood:

$$L(W;X)=\frac1N\sum_{n=1}^N \log p(x_n;W) =\langle \log p(x;W)\rangle_{0} =-\langle E(x;W)\rangle_{0}-\log Z(W),$$

where $\langle\cdot\rangle_0$ means expectation under the empirical data distribution $p_0(x)=\frac1N\sum_n \delta(x-x_n)$. 

Gradient:

$$\frac{\partial L}{\partial W} = -\left\langle \frac{\partial E}{\partial W}\right\rangle_{0} + \left\langle \frac{\partial E}{\partial W}\right\rangle_{\infty},$$

where $\langle\cdot\rangle_{\infty}$ is expectation under the model distribution $p_\infty(x;W)=p(x;W)$. The hard part is $\langle\cdot\rangle_\infty$ (depends on $Z$); MCMC can estimate it but is slow and high-variance. 

### 1.2 Contrastive divergence objective and intuition

ML minimizes $KL(p_0\parallel p_\infty)$. CD learning approximately follows the gradient of:

$$CD_n = KL(p_0\|p_\infty) - KL(p_n\|p_\infty),$$

where $p_n$ is the distribution after running the Markov chain **for $n$ steps**, initialized at the data distribution $p_0$. In practice $n$ is small (often $1$), reducing compute and gradient variance. 

The paper’s goal: characterize **$a$** whether CD converges / what its fixed points are relative to ML, and **$b$** how large the bias is in practice. 

## 2) Models studied and the exact ML vs exact $\text{CD}_n$ rules

They focus on **Boltzmann machines** with binary units $\in\lbrace 0,1\rbrace$, no biases (for simplicity). Two cases: fully visible BM (VBM) and restricted BM (RBM). 

### 2.1 Fully Visible Boltzmann Machine VBM($v$)

* $v$ visible units $x=(x_1,\dots,x_v)^\top$, no hidden units.
* Symmetric weight matrix $W=(w_{ij})$.
* Energy:
  
  $$E(x;W)=-\frac12 x^T W x.$$
  
* Since $\partial E/\partial w_{ij}=-x_i x_j$:

**Exact ML update:**

$$w_{ij}^{(\tau+1)}=w_{ij}^{(\tau)}+\eta\left(\langle x_i x_j\rangle_0-\langle x_i x_j\rangle_\infty\right).$$

**Exact CD(_n) update:**

$$w_{ij}^{(\tau+1)}=w_{ij}^{(\tau)}+\eta\left(\langle x_i x_j\rangle_0-\langle x_i x_j\rangle_n\right),$$

where $p_n = T^n p_0$ for transition matrix $T$ (Gibbs sampler in their experiments). 

Key property: for VBMs the log-likelihood has a **unique optimum** (negative definite Hessian), so “the ML solution” is unambiguous. 

### 2.2 Restricted Boltzmann Machine RBM($(v,h)$)

* $v$ visible $x$, $h$ hidden $y$, bipartite connections only.
* Weight matrix $W=(w_{ij})$ size $h\times v$.
* Energy:
  
  $$E(x,y;W)=-y^\top W x.$$
  
* Gibbs sampling decomposes into two half steps (hidden given visible, then visible given hidden). They write $T = T_x T_y$. 
* Since $\partial E/\partial w_{ij}=-y_i x_j$:

**Exact ML update:**

$$w_{ij}^{(\tau+1)}=w_{ij}^{(\tau)}+\eta\Big(\big\langle\langle y_i x_j\rangle_{p(y\mid x;W)}\big\rangle_{0}-\langle y_i x_j\rangle_\infty\Big).$$

**Exact $\text{CD}_n$ update:**

$$w_{ij}^{(\tau+1)}=w_{ij}^{(\tau)}+\eta\Big(\big\langle\langle y_i x_j\rangle_{p(y\mid x;W)}\big\rangle_{0}-\langle y_i x_j\rangle_n\Big).$$

RBMs can have **multiple local optima**, complicating comparison. 

### 2.3 “Exact” in this paper

To isolate CD vs ML (not sampling noise), they use **small models** where they can compute:
* the **exact** model distribution $p_\infty$,
* and the **exact** distribution after $n$ MCMC steps $p_n$,
  at each learning iteration. ML means $n\to\infty$; $\text{CD}_n$ means finite $n$ but computed exactly. 

## 3) Theoretical core: analysis of fixed points and why CD is biased

### 3.1 Geometry viewpoint

For $v$ binary units, any distribution $p$ is a point in the simplex $\Delta_{2^v}\subset \mathbb{R}^{2^v}$. A BM family defines a lower-dimensional **model manifold** $M$ inside the simplex (parameterized by weights $W$). Learning traces trajectories within $M$. 

A Gibbs sampler transition operator is a stochastic matrix $T$ with stationary distribution $p_\infty$ (the model distribution). For any initial distribution $p$, the chain produces $p_n = T^n p$, and $T^\infty p = p_\infty$ (for finite weights). 

### 3.2 Fixed point conditions

Let $g=\partial E/\partial W$ (vector of energy derivatives).

* **ML fixed points** (for gradient ascent):
  
  $$\langle g\rangle_0 = \langle g\rangle_\infty.$$
  
* **$\text{CD}_n$ fixed points**:
  
  $$\langle g\rangle_0 = \langle g\rangle_n.$$

Main claim: **in general these sets differ**. More explicitly, they show existence of:
* some $p_0$ where ML is fixed but $CD_1$ is not, and
* some $p_0$ where $CD_1$ is fixed but ML is not. 

### 3.3 Linear-algebra framework (their key derivation)

Define a matrix $G$ collecting derivatives of the energy wrt each parameter for each state:

$$G_{i x} = -\frac{\partial E}{\partial w_i}(x;W),$$

treating $W$ as a vector with $\|W\|$ elements and states $x\in\lbrace 0,\dots,2^v-1\rbrace$. 

Define the “moments” (expected sufficient statistics) under distribution $p$:

$$s = \left\langle -\frac{\partial E}{\partial W}\right\rangle_p = Gp.$$

So moments are **linear** in the distribution $p$. 

Fix the weights $W$, hence fix the model distribution $p_\infty$ and its moments $s_\infty = G p_\infty$.

Now define two sets of data distributions (dependent on that fixed $W$):

* ML-fixed-point data distributions:
  
  $$P_0=\lbrace p_0:; Gp_0=s_\infty,; \mathbf{1}^T p_0=1,; p_0\ge 0\rbrace.$$
  

* $\text{CD}_1$-fixed-point data distributions:
  
  $$P_1=\lbrace p_0:; G T p_0=s*\infty,; \mathbf{1}^T p_0=1,; p_0\ge 0\rbrace.$$
  
Both contain $p_\infty$ (because $p_\infty = T p_\infty$). But generically $P_0 \neq P_1$; their intersection is lower-dimensional (measure zero within the simplex), implying **CD bias is the rule, not the exception**. 

### 3.4 Concrete example: VBM(2) (Figure 1)

For VBM with $v=2$ (4 states), the model manifold is a 1D curve (a line segment) inside the 3D simplex (tetrahedron). Figure 1 (page 3) illustrates:
* the simplex,
* the VBM manifold (red vertical segment),
* and that ML corresponds to orthogonal projection onto the manifold.

They compute the “no-bias” set $P_0\cap P_1$ (within the simplex) and find it is the union of three planes:

* $p_{11}=0$,
* $p_{11}=1/4$,
* $3p_{01}+p_{11}=1$,
  which indeed has measure zero in the simplex. So for almost every data distribution $p_0$, $\text{CD}_1$ is biased relative to ML. 

### 3.5 Important nuance: reachable distributions

If the data distribution is actually representable by the model (i.e., $p_0\in M$ and equals some $p_\infty$), it is invariant under $T$, so it is a fixed point for both ML and CD. But real data are usually not exactly reachable by small models. 

### 3.6 What they do *not* prove

They do **not** prove CD convergence in general (even in the “exact” case), though empirically it appears to converge in their experiments. They point to stochastic approximation tools as likely relevant for the noisy case. 

## 4) Empirical study 1: Fully visible BMs (VBM) — “How big is the bias?”

### 4.1 Experimental protocol (important details)

To remove sampling noise:

* use small VBMs where exact $p_\infty$ and exact $p_1$ are computable,
* define ML as exact $n\to\infty$, $\text{CD}_n$ as exact $n$-step chain,
* choose **$n=1$** with **fixed variable update ordering** for Gibbs sampling because it “should produce the greatest bias” (since CD $\to$ ML as $n\to\infty$). 

Optimization settings (both ML and CD):
* same initial weights,
* constant learning rate $\eta=0.9$,
* max 10,000 iterations (rarely reached),
* stop when gradient norm criterion $\|e\| < 10^{-7}$, with
  $e=\langle x_i x_j\rangle_0-\langle x_i x_j\rangle_\infty$ for ML and
  $e=\langle x_i x_j\rangle_0-\langle x_i x_j\rangle_1$ for $\text{CD}_1$. 

They sample many data distributions uniformly in the simplex; for $v=2$, they use $10^4$ distributions and start learning from $W=0$. 

### 4.2 Observations: uniqueness of CD convergence (VBM)

For VBM($v$), they argue:

* ML optimum is unique (known property),
* empirically CD also seems to converge to a unique point for $v=3\ldots10$ (tested from many initial weights),
* for $v=2$ they can prove uniqueness. 

### 4.3 Results (Figures 2–5, pages 5–6)

**Performance (KL to data):**
Histograms of $\text{KL}(p_0\parallel p_{ML})$ and $KL(p_0\parallel p_{CD})$ are very similar; CD is “very close to ML on average” (Figure 2). 

**Bias measure:**
They use symmetrized KL divergence between the ML and CD learned model distributions $p_{ML}$ vs $p_{CD}$. The histogram (Figure 3) shows the bias is “almost always very small,” typically **$<5%$ of the ML KL error** for the same distribution (they note this explicitly). 

**Where bias is larger:**
Bias increases for data distributions near the **simplex boundary/corners**—intuitively, sparse or peaky distributions that the small model fits poorly. They note the distributions with highest ML KL error tend also to have highest CD bias. 

**Geometry explanation:**
Figure 4 plots KL error vs Euclidean distance from the simplex center (uniform distribution). Near the center, both methods have low error; near boundaries, variability increases. They connect the “branches” in the plot to directions aligned with the model manifold vs away from it (using the simplex geometry in Figure 1). 

**Learning curves:**
Figure 5 shows ML and CD decrease similarly and converge at similar rates (in iterations). CD ends slightly worse; in one example CD’s KL increases slightly at the end, suggesting it approached the ML optimum but then moved away. 

**Key takeaway from VBM experiments:** $\text{CD}_1$ is biased in theory, but the **magnitude** is usually small; and ML iterations are much more expensive in practical MCMC settings, so CD’s speed/variance advantages can dominate. 

## 5) Empirical study 2: RBMs — multiple optima and “pairing” phenomenon

### 5.1 Why RBMs are harder to analyze

RBMs have more representational power but introduce **multiple local optima** for both ML and CD, so you can’t characterize bias by a single convergence point across many datasets. 

### 5.2 Their method to map optima and convergence relations

For a fixed data distribution:

1. Generate 60 random initial weight vectors.
2. Run ML and CD from each start to get discovered optima.
3. Also run ML starting from CD optima and CD starting from ML optima, iterating until no new optima appear.
4. Build a convergence graph where an arrow ($A\to B$) indicates “running the other algorithm from optimum $A$ converges to optimum $B$.” 

They need a criterion to decide if two solutions are “the same”; they use **symmetrized KL distance threshold 0.01**, with **$10^5$** parameter updates, and also use a **Good–Turing** estimator to approximate how many optima were missed. 

### 5.3 Representative RBM experiment (Figure 6, pages 6–7)

Configuration:
* $v=6$, $h=4$.
* Data distribution derived from 4 binary vectors, plus smoothing: add count 0.1 to every possible vector then renormalize (so it’s near simplex boundary). 
* 60 initializations: 20 from $N(0,0.1)$, 20 from $N(0,1)$, 20 from $N(0,10)$.
* Found 27 ML optima and 28 CD optima; estimated missing about 3 and 4 respectively (Good–Turing). 

Visualization:
* 2D embedding via **SNE** (stochastic neighbor embedding), perplexity 3, to preserve local distances better than PCA. 

Robust phenomenon:
* ML and CD optima often come in **pairs** that **converge to each other** under the other method.
* CD’s optimum in a pair has **greater or equal KL error** than the associated ML optimum, but the difference is **small**.
* Variation due to **initialization choice** affects KL more than the CD-vs-ML bias (Panels B–C). 

Interpretation: Even with $n=1$, CD tends to land near an ML optimum in practice, consistent with the “small bias” story from VBMs. 

## 6) Practical strategy: CD to initialize ML (CD-ML)

### 6.1 Motivation

Since CD typically gets close but may retain small bias, one can:
* increase $n$ over training (more costly), or
* do a simple two-phase procedure: **run CD to near convergence, then run ML briefly** to “clean up.” They call this **CD-ML**. 

### 6.2 “More realistic” data distribution (Figure 7, pages 7–8)

They use a distribution from real image statistics:
* Take all $3\times 3$ patches from **11,000** USPS $16\times16$ digit images.
* Threshold grayscale (256 levels) at 150 to get 9-bit binary vectors ($v=9$).
* $p_0$ is normalized counts over **2,156,000** patches, lying on simplex boundary. 

Experiment:
* 60 initializations: $20$ $N(0,1/3)$, $20$ $N(0,1)$, $20$ $N(0,3)$.
* For each: compare
  * ML for $10^4$ iterations, vs
  * CD for $10^4$ iterations + short ML run.
* Tested $h=1,\dots,8$. They find a unique ML optimum and several CD optima. If CD is followed by **1,000 ML iterations**, all CD optima converge to the ML optimum. 

Cost model + outcome:

* Assume 1 ML iteration costs $20\times$ a CD iteration (their “reasonable estimate” for that RBM size).
* Plot KL vs CPU cost: CD-ML reaches **about the same final KL as full ML**, but at a **small fraction of the compute**; the curve drops sharply right when switching from CD to ML, implying only a few expensive ML steps are needed. 

## 7) Final conclusions and how to cite this paper’s contributions

### 7.1 Main conclusions (what you should take away)

1. **Negative theoretical result:** CD’s fixed points generally differ from ML’s fixed points → CD is **biased** as an ML estimator for random fields. 
2. **Positive empirical result:** For Gibbs sampling (the practical default), the bias is **usually very small**; CD often converges **very near** an ML optimum in their exact-computation studies. 
3. **Practical recommendation:** Use **fast CD** to get close, then **short ML** to remove residual bias (“CD-ML”), saving substantial compute compared to full ML. 

### 7.2 Why theory is hard (their explanation)

The distribution $p_1$ (or $p_n$) is a “moving target” that depends on $W$ and on the sampling scheme (e.g., Gibbs); this complicates deriving an explicit objective or convergence proof for CD. 

### 7.3 Relation to prior work (as positioned here)

They mention:

* MacKay (2001): examples of CD bias but with unusual sampling operators.
* Williams & Agakov (2002): CD can be unbiased for 2D Gaussian BMs and reduces variance.
* Yuille (2004): gives a condition for CD to be unbiased, hard to apply.
  They highlight convergence of exact CD as an open problem; noisy-case analysis likely via stochastic approximation. 

## 8) “If you need to reuse these notes in your own research writing”

**When citing this paper, the key claims you can safely attribute to it:**

* $\text{CD}_n$ can be seen as approximately following the gradient of $\text{KL}(p_0\parallel p_\infty)-\text{KL}(p_n\parallel p_\infty)$. 
* Fixed points of CD and ML generally differ (bias) via their (G) vs (GT) hyperplane characterization (the $P_0$ vs $P_1$ sets). 
* In exact small-model experiments, CD bias is typically small; largest near simplex boundaries (sparse/peaky distributions). 
* In RBMs, optima often come in ML–CD pairs; initialization affects KL more than CD bias. 
* CD-ML (CD then brief ML) can reach ML-level KL at far lower compute under reasonable cost assumptions. 
