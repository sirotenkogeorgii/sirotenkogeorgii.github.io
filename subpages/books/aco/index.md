---
layout: default
title: Algorithms for Combinatorial Optimization
date: 2025-03-10
---

# Algorithms for Combinatorial Optimization

## Chapter 1: What is a Combinatorial Problem?

### What is a Combinatorial Problem?

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.1</span><span class="math-callout__name">(Shortest path in a graph)</span></p>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.2</span><span class="math-callout__name">((Cyclic) traveling salesman)</span></p>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.3</span><span class="math-callout__name">(Pseudo-boolean optimization)</span></p>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.4</span><span class="math-callout__name">(Constraint optimization problem)</span></p>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.5</span><span class="math-callout__name">(Prime factorization)</span></p>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.6</span><span class="math-callout__name">((Non-binary) knapsack problem)</span></p>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1.7</span><span class="math-callout__name">(Combinatorial Problem)</span></p>

A problem of the form

$$\min_{\mathbf{x} \in S \subset \mathbb{Z}^n} f(\mathbf{x})$$

we will call *combinatorial*.

</div>

### Labeling Problem

There are many problems in computer science which can be formulated as a *labeling problem*. Examples can be found in bio-informatics, communication theory, statistical physics, computer vision, signal processing, information retrieval and machine learning.

Labeling problems as a modeling tool naturally appear when

- the target object (the object we model) consists of *many small parts*,
- each part must be labeled by a label from a *finite set*, and
- parts (and, therefore, their labels) are *mutually dependent*.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.10</span><span class="math-callout__name">(Image segmentation)</span></p>

Image segmentation is a classical image analysis task: Each pixel of an input image must be assigned a label of an object visible in the image. For instance, for street scenes the labels could belong to the set $\lbrace\texttt{pedestrian}, \texttt{car}, \texttt{tree}, \texttt{building}\rbrace$.

The target object is an image, i.e. a two-dimensional array of pixels. Each pixel constitutes an elementary part of the image and must be labeled with a label from a finite set. The simplest assumption about image segments, i.e. groups of pixels having the same label, is the so called "compactness assumption". It states that it is more probable that neighboring pixels are labeled with the same label than with different ones.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.11</span><span class="math-callout__name">(Depth reconstruction)</span></p>

Depth reconstruction is another important image analysis problem. In the classical setting there are (at least) two images taken from different viewpoints. The task is to match pixels from these two images to each other. Assuming the positions of cameras and their focal lengths are known, this allows us to estimate depth of the scene.

The target object is a two-dimensional pixel array, where each pixel must be labeled with a label from a finite set. Each label represents depth information of the associated pixel in an image, i.e. how far the depicted observation is placed from the camera. The set of labels is chosen as natural numbers in a given interval, for instance, $\lbrace 0, 1, \ldots, 255 \rbrace$.

Assuming that the observed surface is smooth, one would expect the difference $\lvert s - s' \rvert$ between labels $s$ and $s'$ in neighboring pixels to be small.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.12</span><span class="math-callout__name">(Cell tracking problem in bio-imaging)</span></p>

Given is a sequence of images that show the development of a living organism from an early embryo consisting of only a few cells to a fully grown animal. During this sequence, the images show moving and splitting cells.

Under the assumption that the image is already *pre-segmented*, i.e. the cells were already found in each image, the task at hand is to track each individual cell and its descendants from the first to the last frame.

The cells are the elementary parts of the considered object. Each cell in a given image frame is labeled with pointers to one or two cells in the next frame. One pointer means the cell only moved, and two pointers correspond to a cell division. The simplest tracking model forbids two different cells to have the same descendants. This rule defines dependencies between object parts.

</div>

#### Basic Definitions

**Graph.** Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ be an undirected graph consisting of a finite set of *nodes* $\mathcal{V}$ and a set of edges $\mathcal{E} \subseteq \binom{\mathcal{V}}{2}$. The set $\mathcal{E}$ will also be called a *neighborhood structure* of $\mathcal{V}$. We typically use lower case letters $u$ and $v$ for nodes and write $uv$ to denote an edge $\lbrace u, v \rbrace \in \mathcal{E}$ connecting $u$ and $v$. Since the graph is undirected, $uv$ and $vu$ denote the same edge. The notation $\mathcal{N}(u)$ will be used for the set of nodes $\lbrace v \mid uv \in \mathcal{E} \rbrace$ connected to the node $u$.

The graph $\mathcal{G}$ is considered as a model of the considered target object, where the nodes represent the elementary object parts and edges stand for mutually dependencies between them.

**Labels and unary costs.** A finite *set of labels* $\mathcal{Y}_u$ is associated with each node $u \in \mathcal{V}$. Our preference for each label is expressed by the *unary cost function* $\theta_u \colon \mathcal{Y}_u \to \mathbb{R}$, which is defined for each node $u \in \mathcal{V}$. The value $\theta_u(s)$ determines the *cost*, which we pay for assigning label $s \in \mathcal{Y}_u$ to the node $u$. Sometimes we will use very high costs to implicitly forbid certain labels or label pairs. The notation $\infty$ will be used to denote such high costs.

Unary costs are usually defined by what is known from observation. They are often called the "data term" to emphasize that they depend on the input data or observation.

**Dependence and pairwise costs.** Dependencies between labels assigned to different graph nodes are modeled with *pairwise cost functions* $\theta_{uv} \colon \mathcal{Y}_u \times \mathcal{Y}_v \to \mathbb{R}$, which are defined for each edge $uv \in \mathcal{E}$ of the graph.

A simple (although not always the best) way to model the compactness assumption in Example 1.10 is to assign

$$\theta_{uv}(s, t) = \begin{cases} 0, & s = t \\ \alpha, & s \neq t \end{cases}$$

for any pair of labels $(s, t) \in \mathcal{Y}_u \times \mathcal{Y}_v$ with some $\alpha > 0$. A simple way to model a smooth surface in depth reconstruction in Example 1.11 is to assign

$$\theta_{uv}(s, t) = \lvert s - t \rvert \,,$$

to penalize large differences between depth in the neighboring nodes.

In the cell tracking example the pairwise costs should forbid the same labels to be assigned to neighboring nodes when no cell division happens:

$$\theta_{uv}(s, t) = \begin{cases} 0, & s \neq t \\ \infty, & s = t \,. \end{cases}$$

These examples show that pairwise costs often incorporate the prior information about a considered object, therefore, they are often collectively referred to as the *prior*.

Costs and cost functions are also called *potentials* and *potential functions*. We prefer the term *cost* since it is more widely used in general optimization literature.

Since unary and pairwise costs are functions of discrete variables, they can be seen as vectors. Therefore we can treat the unary cost function $\theta_u$ as a *unary cost vector* $(\theta_u(s), \ s \in \mathcal{Y}_u)$. Similarly, each pairwise cost function can be considered as a *pairwise cost vector* $\theta_{uv} = (\theta_{uv}(s,t), \ (s,t) \in \mathcal{Y}_u \times \mathcal{Y}_v)$. All unary vectors stacked together form the vector of all unary costs $\theta_{\mathcal{V}} = (\theta_u, \ u \in \mathcal{V})$. The vector $\theta_{\mathcal{E}}$ of all pairwise costs is defined similarly as $(\theta_{uv}, \ uv \in \mathcal{E})$. Stacking together the latter two results in a long *cost vector* $\theta = (\theta_{\mathcal{V}}, \theta_{\mathcal{E}})$ with dimension $\mathcal{I} := \sum_{u \in \mathcal{V}} \lvert \mathcal{Y}_u \rvert + \sum_{uv \in \mathcal{E}} \lvert \mathcal{Y}_{uv} \rvert$.

**Labeling.** We will often use the notation $\mathcal{Y}_{\mathcal{A}}$ for all possible label assignments to a subset of nodes $\mathcal{A} \subseteq \mathcal{V}$. Formally, $\mathcal{Y}_{\mathcal{A}}$ stands for the Cartesian product $\prod_{u \in \mathcal{A}} \mathcal{Y}_u$. In particular, $\mathcal{Y}_{uv}$ denotes $\mathcal{Y}_u \times \mathcal{Y}_v$ and is the set of all possible pairs of labels in nodes $u$ and $v$. A vector $y \in \mathcal{Y}_{\mathcal{V}}$ of labels assigned to *all* nodes of the graph is called *labeling*. We will refer to coordinates of this vector with the node index, i.e. $y_u$ stands for the label assigned to the node $u$. One may also speak about *partial labelings*, if only a subset $\mathcal{A}$ of the nodes is labeled.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1.13</span><span class="math-callout__name">(Labeling problem)</span></p>

The triple $(\mathcal{G}, \mathcal{Y}_{\mathcal{V}}, \theta)$ consisting of a graph $\mathcal{G}$, discrete space of all labelings $\mathcal{Y}_{\mathcal{V}}$ and a corresponding cost vector $\theta$, is called a *labeling problem*.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1.14</span><span class="math-callout__name">(Energy minimization problem (Labeling problem))</span></p>

The problem

$$y^* = \arg\min_{y \in \mathcal{Y}_{\mathcal{V}}} \left[ E(y; \theta) := \sum_{u \in \mathcal{V}} \theta_u(y_u) + \sum_{uv \in \mathcal{E}} \theta_{uv}(y_u, y_v) \right]$$

of finding a labeling $y^*$ with minimal total cost will be called *energy minimization* or *maximum a posteriori (MAP) inference* problem for the labeling problem $(\mathcal{G}, \mathcal{Y}_{\mathcal{V}}, \theta)$.

</div>

For the sake of notation we will sometimes use the short form

$$y^* = \arg\min_{y \in \mathcal{Y}_{\mathcal{V}}} \left[ E(y; \theta) := \sum_{w \in \mathcal{V} \cup \mathcal{E}} \theta_w(y_w) \right]$$

with $y_w$ being equal to $y_u$, if $w$ corresponds to a node, i.e. $w = u \in \mathcal{V}$, and $y_{uv}$, if $w$ corresponds to an edge, i.e. $w = uv \in \mathcal{E}$.

Problems equivalent or very closely related to energy minimization have also other names depending on the corresponding community they are studied in: *maximum likelihood explanation (MLE) inference* (machine learning, natural language processing community), *weighted/valued/partial constraint satisfaction problem* (constraint satisfaction community).

### Dynamic Programming

#### Case Study: Labeling Problem

Let the graph $\mathcal{G}$ be chain-structured. In this case the set of nodes can be totally ordered, i.e. $\mathcal{V} = \lbrace 1, \ldots, n \rbrace$. The set of edges contains pairs of neighboring nodes in the chain: $\mathcal{E} = \lbrace (i, i+1) \colon i = 1, \ldots, n-1 \rbrace$.

Consider a chain-structured labeling problem. An optimal labeling starts in the first (left-most) node and ends in the last (right-most) one. To select its last label optimally assume we have computed the function $F_n \colon \mathcal{Y}_n \to \mathbb{R}$ such that $F_n(s) + \theta_n(s)$ denotes the cost of the best (minimal-cost) labeling with the very last label being $s$. In other words, the value $F_n(s)$ is the cost of the labeling without the unary cost $\theta_n(s)$. Then the minimization $y_n = \arg\min_{s \in \mathcal{Y}_n}(F_n(s) + \theta_n(s))$, which can be performed by enumeration, allows us to determine the last label of an optimal labeling $y$.

To find other labels the same considerations can be applied given that the functions $F_i \colon \mathcal{Y}_i \to \mathbb{R}$ are computed such that $F_i(s)$ equals the cost of the best partial labeling in the nodes $1, \ldots, i$ with the label $s$ assigned to the $i$-th node.

Functions $F_i$ can be recursively computed as

$$F_i(s) := \min_{t \in \mathcal{Y}_{i-1}} \left( F_{i-1}(t) + \theta_{i-1}(t) + \theta_{i-1,i}(t, s) \right) \,,$$

with $F_1(s) = 0$ for all $s \in \mathcal{Y}_1$.

These considerations can be derived from the following recursive representation of the energy of a chain-structured labeling problem:

$$\min_{y \in \mathcal{Y}_{\mathcal{V}}} E(y; \theta) = \min_{y_1, \ldots, y_n} \left( \sum_{i=1}^{n-1} \left( \theta_i(y_i) + \theta_{i,i+1}(y_i, y_{i+1}) \right) + \theta_n(y_n) \right)$$

where functions $F_i \colon \mathcal{Y}_i \to \mathbb{R}$ for $i \in \mathcal{V}$ are introduced "on the fly":

$$= \min_{y_2, \ldots, y_n} \left( \underbrace{\min_{y_1 \in \mathcal{Y}_1} \left( \theta_1(y_1) + \theta_{1,2}(y_1, y_2) \right)}_{F_2(y_2)} + \sum_{i=2}^{n-1} \left( \theta_i(y_i) + \theta_{i,i+1}(y_i, y_{i+1}) \right) + \theta_n(y_n) \right)$$

$$= \min_{y_3, \ldots, y_n} \left( \underbrace{\min_{y_2 \in \mathcal{Y}_2} \left( F_2(y_2) + \theta_2(y_2) + \theta_{2,3}(y_2, y_3) \right)}_{F_3(y_3)} + \sum_{i=3}^{n-1} \left( \theta_i(y_i) + \theta_{i,i+1}(y_i, y_{i+1}) \right) + \theta_n(y_n) \right)$$

$$\cdots = \min_{y_n \in \mathcal{Y}_n} \left( F_n(y_n) + \theta_n(y_n) \right) \,.$$

These transformations confirm that the energy of the optimal labeling can be computed as $\min_{y \in \mathcal{Y}_{\mathcal{V}}} E(y) = \min_{s \in \mathcal{Y}_n}(F_n(s) + \theta_n(s))$ and values $F_i(s)$ can be computed recursively.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm 1</span><span class="math-callout__name">(Dynamic programming for chain-structured labeling problems)</span></p>

**Given:** $(\mathcal{G}, \mathcal{Y}_{\mathcal{V}}, \theta)$ — a chain-structured labeling problem

1. $F_1(s) := 0$, for all $s \in \mathcal{Y}_1$
2. **for** $i = 2$ **to** $n$ **do**
    - $F_i(s) := \min_{t \in \mathcal{Y}_{i-1}} \left( F_{i-1}(t) + \theta_{i-1}(t) + \theta_{i-1,i}(t, s) \right)$, $\forall s \in \mathcal{Y}_i$
    - $r_i(s) := \arg\min_{t \in \mathcal{Y}_{i-1}} \left( F_{i-1}(t) + \theta_{i-1}(t) + \theta_{i-1,i}(t, s) \right)$, $\forall s \in \mathcal{Y}_i$
3. **end for**
4. $E^* = \min_{s \in \mathcal{Y}_n} (F_n(s) + \theta_n(s))$
5. **return** $E^*$, $\lbrace r_i(s) \colon i = 2, \ldots, n, \ s \in \mathcal{Y}_i \rbrace$

</div>

Values $F_i(s)$ are often called *forward min-marginals* or *forward messages*, since they represent an optimal labeling from the first node to the label $s$ in the node $i$, and all unary/pairwise costs with indexes smaller than $i$ are marginalized out.

Values $r_i(s)$ are pointers needed to recover the optimal labeling, when its energy is computed, as given by Algorithm 2.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm 2</span><span class="math-callout__name">(Reconstructing an optimal labeling)</span></p>

**Given:** $r_i(s) \colon i = 2, \ldots, n, \ s \in \mathcal{Y}_i$

1. $y_n = \arg\min_{s \in \mathcal{Y}_n} (F_n(s) + \theta_n(s))$
2. **for** $i = n$ **to** $2$ **do**
    - $y_{i-1} = r_i(y_i)$
3. **end for**
4. **return** $y$

</div>

**Computation complexity.** The complexity of each minimization in line 4 of Algorithm 1 is $O(\lvert \mathcal{Y}_{i-1} \rvert)$. One has to perform $\lvert \mathcal{Y}_i \rvert$ such minimizations on each iteration. Therefore, complexity of the algorithm reads

$$O\!\left(\sum_{i=2}^{n} \lvert \mathcal{Y}_{i-1} \rvert \lvert \mathcal{Y}_i \rvert\right),$$

which results in $O(L^2 \lvert \mathcal{V} \rvert) = O(L^2 \lvert \mathcal{E} \rvert)$ if we assume that all nodes have an equal number $L$ of labels. This is exactly the memory, which is required to define a chain-structured labeling problem: $\lvert \mathcal{V} \rvert$ unary cost vectors of the size $L$ each plus $\lvert \mathcal{E} \rvert - 1$ pairwise costs of the size $L^2$ each. In other words, dynamic programming for chain-structured labeling problems has linear complexity with respect to the model size.

The functions $F_i$ are known as *Bellman* functions, in honor of the author of dynamic programming. Besides the original name, Algorithm 1 is often referred to as the *Viterbi algorithm*.

## Chapter 2: (Integer) Linear Programs

We start this chapter with a general notion of a (constrained) optimization problem and its relaxations. Further, we will focus on a specific, polyhedral constraints, which will lead us naturally to the linear programs. Finally, we will consider linear programs with integer constraints, known as integer linear programs.

### Optimization Problems

In the optimization literature the notation

$$\min_{x \in X} f(x)$$

is adopted for a problem of *minimizing* a function $f \colon X \to \mathbb{R}$ on a set $X \subseteq \mathbb{R}^n$. More precisely, it means that *the optimal value* of $f$ defined as $f^* = \inf_{x \in X} f(x)$ must be found.

- If $f^* = -\infty$, that is, there exists a sequence $x^t \in X$ such that $f(x^t) \xrightarrow{t \to \infty} -\infty$, the problem is called *unbounded*.
- When $X$ is an empty set, the problem is called *infeasible*, and the notation $f^* = \infty$ is adopted.
- The fact that the problem is neither infeasible nor unbounded, in other words, *feasible* and *bounded*, is denoted as $-\infty < f^* < \infty$.

The set $X$ is called the *feasible set* and $x \in X$ is a *feasible point*. A vector $x^* \in X$ such that $f(x^*) = f^*$ is called an *(optimal) solution* or *optimal point* of the problem. In general, an optimal solution is not unique, or may not exist, even if $-\infty < f^* < \infty$.

A feasible point $x$ such that $f(x) \approx f^*$ (denoted as $f(x) \approx f^*$) is often referred to as an *approximate solution* of the minimization problem.

The problem of the form $\min_{x \in X} f(x)$ is referred to as a *constraint minimization* problem. The *constrained maximization* problem $\max_{x \in X} f(x)$ is defined by substituting min with max, inf with sup, and $-\infty$ with $\infty$ and the other way around. Minimization and maximization problems together are also called *optimization* problems.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.1</span><span class="math-callout__name">(Optimal values exist but optimal points do not)</span></p>

The optimal values of the problems

$$\min_{x \geq 1} e^{\frac{1}{x}} \,; \qquad \min_{x > 1} x$$

are finite and equal to 1, but their optimal points do not exist. For the first problem, $x \geq 1$ is feasible and $e^{\frac{1}{x+1}} < e^{\frac{1}{x}}$, so $x$ is never optimal. For the second, $1 + \frac{x-1}{2}$ is feasible and strictly smaller than $x$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.2</span><span class="math-callout__name">(Infeasible problems)</span></p>

The following optimization problems are infeasible, as their respective feasible sets are empty:

$$\min_{\substack{x \geq 1 \\ x \leq 0}} f(x); \qquad \max_{\substack{x \in \{0,1\} \\ x \geq 2}} f(x); \qquad \min_{\substack{x \in \{0,1\} \\ x \in [0.2, 0.9]}} f(x) \,.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.3</span><span class="math-callout__name">(Unbounded problems)</span></p>

The following optimization problems are unbounded:

$$\min_{x \leq 0} x + 1; \qquad \min_{x \geq 0} 2 - x; \qquad \max_{x \leq 0} x^2 \,.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.4</span><span class="math-callout__name">(Multiple optimal solutions)</span></p>

The following optimization problems have multiple optimal solutions:

$$\min_{x \in [-1,1]} 1 - x^2; \qquad \min_{x \in \{0,1\}} \lvert x - 0.5 \rvert; \qquad \max_{x \in [0,1]} 1 \,.$$

</div>

In the rest of the monograph we will deal mostly with *linear objectives*, i.e. $f(x) = \langle c, x \rangle$ for some vector $c \in \mathbb{R}^n$. The feasible set $X$ will usually be either *finite* or *polyhedral*.

#### Relaxations of Optimization Problems

Important optimization problems are often computationally intractable. Therefore, it is natural to consider tractable approximations of these problems to obtain an approximate solution of the problem or at least a bound on its optimal value.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.5</span><span class="math-callout__name">(Relaxation)</span></p>

The optimization problem

$$\min_{x \in X'} g(x)$$

is called a *relaxation* of the problem $\min_{x \in X} f(x)$ if $X' \supseteq X$ and $g(x) \leq f(x)$ for any $x \in X$. The solution of the relaxed problem is often referred to as a *relaxed solution*.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.6</span></p>

The problems $\min_{x \geq 0} f(x)$ and $\min_{x \in [0,1]^n} f(x)$ are relaxations of the problem $\min_{x \in \{0,1\}^n} f(x)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.7</span></p>

The problem $\min_{x \in [0,1]} x^2$ is a relaxation of the problem $\min_{x \in [0,1]} \lvert x \rvert$. The advantage of this relaxation is the differentiability of $x^2$, which may often lead to faster optimization algorithms.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.8</span></p>

The problems $\min_{x \in X \subset \mathbb{R}^n} \langle c, x \rangle$ and $\min_{x \colon Ax = b} \langle c, x \rangle$ are relaxations of $\min_{\substack{x \in X \subset \mathbb{R}^n \\ Ax = b}} \langle c, x \rangle$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.9</span><span class="math-callout__name">(Relaxations provide lower bounds)</span></p>

Any relaxation constitutes a lower bound, i.e.

$$\min_{x \in X'} g(x) \leq \min_{x \in X} f(x),$$

if $X \subseteq X'$ and $g(x) \leq f(x)$ for $x \in X$. Moreover, if $x' \in X$ and $f(x') = g(x')$ holds for a relaxed solution $x' \in \arg\min_{x \in X'} g(x)$, then $x'$ is an optimal solution of the non-relaxed problem $\min_{x \in X} f(x)$ as well. In this case, one says that the lower bound provided by the relaxation is *tight*.

</div>

Relaxations, which provide tight lower bounds for *all* instances of optimization problems from a certain class, are called *tight relaxations*.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.10</span></p>

Problem $\min_{x \in [0,1]} cx$ is a tight relaxation for $\min_{x \in \{0,1\}} cx$ for any constant $c$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.11</span><span class="math-callout__name">(Tighter relaxation)</span></p>

The relaxation $\min_{x \in X'} g(x)$ is called *tighter* than the relaxation $\min_{x \in \hat{X}'} \hat{g}(x)$, if it provides better lower bound, i.e. $\min_{x \in X'} g(x) \geq \min_{x \in \hat{X}'} \hat{g}(x)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.12</span><span class="math-callout__name">(Equivalent relaxations)</span></p>

Two relaxations are called *equivalent*, if their bounds coincide.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.13</span><span class="math-callout__name">(Uniqueness of tight relaxation solution)</span></p>

Let $\min_{x \in X} f(x)$ be an optimization problem and $\min_{x \in X'} g(x)$ be its relaxation. If $x' \in X$ is the unique solution of the relaxed problem and $g(x') = f(x')$, then it is also the unique solution of the non-relaxed one.

*Proof.* Since $x'$ is the unique relaxed solution, it holds that $f(x') = g(x') < g(x)$ for $x \in X' \setminus \lbrace x' \rbrace$. Therefore, $f(x') < g(x) \leq f(x)$ for all $x \in (X \setminus \lbrace x' \rbrace) \subseteq (X' \setminus \lbrace x' \rbrace)$, which implies that $x'$ is the unique solution of the non-relaxed problem. $\square$

</div>

### Linear and Affine Spaces

Consider the vector space $\mathbb{R}^n$. We call vectors $\mathbf{x}^i \in \mathbb{R}^n$, $i \in [k]$ for some $k$, *linearly independent*, if

$$\sum_{i=1}^{k} a_i \mathbf{x}^i = 0$$

implies $a_i = 0$ for all $i \in [k]$.

A *linear subspace* $S$ in $\mathbb{R}^n$ is a subset of $\mathbb{R}^n$ closed under vector addition and multiplication.

A simple example if a linear subspace is a *linear hyperplane*, defined as the set of vectors $\mathbf{x}$ that satisfy $\langle \mathbf{a}, \mathbf{x} \rangle = 0$ for some $\mathbf{a} \in \mathbb{R}^n$, $\mathbf{a} \neq \mathbf{0}$.

It is known, that in general, a linear subspace $S$ can be represented as an intersection of an arbitrary number $m$ of linear hyperplanes:

$$S := \lbrace \mathbf{x} \in \mathbb{R}^d \colon A\mathbf{x} = \mathbf{b} \rbrace \,.$$

Here $A$ is a $n \times m$ matrix. Every linear subspace has a *dimension*, defined as the maximum number of linearly independent vectors in it. Equivalently, $\dim(S) = d - \operatorname{rank}(A)$.

An *affine subspace* $S'$ of $\mathbb{R}^n$ is a linear subspace $S$ translated by a vector $\mathbf{u}$: $S := \lbrace \mathbf{x} + \mathbf{u} \colon \mathbf{x} \in S \rbrace$. The *dimension* of $S'$ is that of $S$. Equivalently, an affine subspace is the set of points satisfying a set of (inhomogeneous) equations

$$S' := \lbrace \mathbf{x} \in \mathbb{R}^n \colon A\mathbf{x} = \mathbf{b} \rbrace$$

for some matrix $A$.

The dimension of *any subset* of $\mathbb{R}^n$ is the smallest dimension of any affine subspace which contains it.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.14</span></p>

The dimension of the non-empty set $P := \lbrace \mathbf{x} \in \mathbb{R}^n \colon A\mathbf{x} = \mathbf{b}, \ \mathbf{x} \geq \mathbf{0} \rbrace \neq \emptyset$ is $\dim P = n - \operatorname{rank} A$, i.e., the same as the affine subspace $\lbrace \mathbf{x} \in \mathbb{R}^n \colon A\mathbf{x} = \mathbf{b} \rbrace$.

</div>

The special case of an affine subspace is an *affine hyperplane*, defined as the set $\lbrace \mathbf{x} \in \mathbb{R}^n \colon \langle \mathbf{a}, \mathbf{x} \rangle = b \rbrace$ for some $\mathbf{a} \in \mathbb{R}^n$, $\mathbf{a} \neq \mathbf{0}$. The dimension of any affine hyperplane is $n - 1$. The set $\lbrace \mathbf{x} \in \mathbb{R}^n \mid \langle \mathbf{a}, \mathbf{x} \rangle \leq b \rbrace$ is called an *affine half-space*.

### Linear Constraints and Polyhedra

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.15</span><span class="math-callout__name">(Convex polyhedron)</span></p>

A *convex polyhedron* $P$ in $\mathbb{R}^n$ is a set which can be represented by a finite set of linear inequalities, i.e. $P = \lbrace x \in \mathbb{R}^n \colon Ax \leq b \rbrace$ for a matrix $A \in \mathbb{R}^{m \times n}$ and a vector $b \in \mathbb{R}^m$. A bounded convex polyhedron is called a *convex polytope*.

</div>

Since in the following all considered polyhedra will be convex, we will omit this word when speaking about them.

As any subset of $\mathbb{R}^n$, polyhedra have their dimension. Polyhedra of dimension $n$ are called *full-dimensional*.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.16</span><span class="math-callout__name">($n$-dimensional cube)</span></p>

The set $[0,1]^n$, called an *$n$-dimensional cube*, is a polytope, since it can be represented as

$$\lbrace x \in \mathbb{R}^n \mid -x_i \leq 0, \ x_i \leq 1, \ i = 1 \ldots, n \rbrace$$

and it is bounded. The $n$-dimensional cube is full-dimensional. Constraints $x \in [0,1]^n$ constitute a special case of *box constraints*.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.17</span></p>

The set of the type $\lbrace x \in \mathbb{R}^n \colon Ax = b, \ Bx \leq d \rbrace$, where $A$ and $B$ are arbitrary matrices of a suitable dimension, is a polyhedron in $\mathbb{R}^n$.

*Proof.* The equality constraints $Ax = b$ can be equivalently represented as inequalities: $\lbrace Ax \leq b, \ -Ax \leq -b \rbrace$. $\square$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.18</span></p>

A hyperplane $\langle \mathbf{a}, \mathbf{x} \rangle = b$ is a polyhedron, and so is an intersection of multiple hyperplanes $A\mathbf{x} = \mathbf{b}$. Indeed, as an intersection of hyperplanes $A\mathbf{x} = \mathbf{b}$ defines an affine subspace, which is unbounded unless it is empty or consists of a single point.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.19</span></p>

The intersection of two polyhedra is a polyhedron itself. Moreover, if one of the polyhedra is a polytope, the intersection is a polytope as well.

*Proof.* An intersection of two polyhedra given by $Ax \leq b$ and $Bx \leq d$, respectively, is the set $\lbrace x \mid Ax \leq b, \ Bx \leq d \rbrace$ constrained by both these sets of inequalities. The second statement is trivial as the intersection is a subset of the polytope at hand. $\square$

</div>

#### Convex Hulls and Vertices of Polyhedra

There is an important one-to-one relation between a polytope and the set of its vertices, which plays an important role in the analysis of (integer) linear programs.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.20</span><span class="math-callout__name">(Vertex of a polyhedron)</span></p>

A vector $x' \in \mathbb{R}^n$ is called a *vertex* of a polyhedron $P$ if there exists a vector $c \in \mathbb{R}^n$ such that the linear function $\langle c, x \rangle$ attains a unique maximum at $x'$ on $P$.

In other words, $x'$ is a vertex, if it is the unique solution of $\max_{x \in P} \langle c, x \rangle$.

</div>

We will use the notation $\operatorname{vrtx}(P)$ for the set of vertices of a polyhedron $P$.

Note that since $\max_{x \in P} \langle c, x \rangle = -\min_{x \in P}(-\langle c, x \rangle)$, the maximization problem $\max_{x \in P} \langle c, x \rangle$ can be equivalently exchanged with $\min_{x \in P} \langle c, x \rangle$ in Definition 2.20.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.21</span></p>

Let us show that $\lbrace 0,1 \rbrace^n \subseteq \operatorname{vrtx}([0,1]^n)$. Indeed, consider $z \in \lbrace 0,1 \rbrace^n$ and a linear function $\langle c, x \rangle$ with $c_i = 2z_i - 1$. Then

$$\langle c, x \rangle = 2\sum_{i=1}^{n} z_i x_i - \sum_{i=1}^{n} x_i = 2\!\!\sum_{\substack{i=1,\ldots,n:\\z_i=1}} x_i - \sum_{\substack{i=1,\ldots,n:\\z_i=0}} x_i - \sum_{\substack{i=1,\ldots,n:\\z_i=1}} x_i \,.$$

Therefore, $\max_{x \in [0,1]^n} \langle c, x \rangle = \sum_{i=1}^{n} z_i$, and the maximum is attained in the point $z$ only. Therefore, according to Definition 2.20, $z$ is a vertex of $[0,1]^n$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.22</span><span class="math-callout__name">($n$-dimensional simplex)</span></p>

The polytope $\Delta^n := \lbrace x \in \mathbb{R}^n_+ \colon \sum_{i=1}^{n} x_i = 1 \rbrace$ is called the *$n$-dimensional (probability) simplex*. For finite $\mathcal{X}$ we also use the notation $\Delta^{\mathcal{X}}$ for vectors from $\Delta^{\lvert \mathcal{X} \rvert}$ whose coordinates are indexed by elements of the set $\mathcal{X}$. Note that the dimension of the $n$-dimensional simplex is actually $n-1$, as follows from Example 2.14.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.23</span></p>

Let us show that vectors $x^i \in \mathbb{R}^n$, $i = 1, \ldots, n$ such that $x^i_j = \llbracket i = j \rrbracket$ are vertices of the simplex $\Delta^n$. Indeed, for $c = x^i$ the maximum $\max_{x \in \Delta^n} \langle c, x \rangle$ is attained in a single point $x^i$. Therefore, $x^i \in \operatorname{vrtx}(\Delta^n)$.

Note also that $x^i \in \mathbb{R}^n$, $i = 1, \ldots, n$, are the only binary (with coordinates 0 and 1) vectors in $\Delta^n$. In other words, $\lbrace x^i \in \mathbb{R}^n \mid i = 1, \ldots, n \rbrace = \Delta^n \cap \lbrace 0,1 \rbrace^n$. This implies that $\Delta^n \cap \lbrace 0,1 \rbrace^n \subseteq \operatorname{vrtx}(\Delta^n)$. Moreover, later we will show that $\Delta^n \cap \lbrace 0,1 \rbrace^n = \operatorname{vrtx}(\Delta^n)$.

</div>

**Faces of the polyhedron.** Face of the polyhedron $P \in \mathbb{R}^n$ is the set that can be expressed as $\arg\min_{x \in P} \langle c, x \rangle$ for some $c \in \mathbb{R}^n$. The most important types of faces are:

- *vertex* — dimension 0 (point)
- *edge* — dimension 1 (line) and
- *facet* — dimension $\dim P - 1$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.24</span><span class="math-callout__name">(Convex combination)</span></p>

For any finite number of points $x^i \in \mathbb{R}^N$, $i = 1, \ldots, n$ and any $p \in \Delta^n$ the point $\sum_{i=1}^{n} p_i x^i$ is called a *convex combination* of $x^i$, $i = 1, \ldots, n$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.25</span><span class="math-callout__name">(Convex set, two-point definition)</span></p>

A set $X \subseteq \mathbb{R}^N$ is called *convex*, if for any $\alpha \in [0,1]$ and any two points $x, z \in X$ their convex combination $\alpha x + (1 - \alpha) z$ also belongs to $X$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.26</span><span class="math-callout__name">(Convex set, general definition)</span></p>

A set $X \subseteq \mathbb{R}^N$ is called *convex*, if for any finite subset $x^i \in X$, $i = 1, \ldots, n$ of its points and any $p \in \Delta^n$ it holds that $(\sum_{i=1}^{n} p_i x^i) \in X$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 2.27</span></p>

Let $X$ be representable as the intersection of an arbitrary number of convex sets. Then $X$ is convex as well.

*Proof.* Let $x$ and $z$ belong to the intersection. Then $x$ and $z$ belong to each of the sets. Consider now one of these sets. Then, for any $\alpha \in [0,1]$ the point $\hat{x} = \alpha x + (1 - \alpha)z$ belongs to this set as well due to its convexity. Since this consideration holds for any set involved in the intersection, $\hat{x}$ belongs to the intersection as well. $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.28</span><span class="math-callout__name">(Polyhedra are convex sets)</span></p>

*Proof.* It is necessary to prove that the set $X = \lbrace x \colon Ax \leq b \rbrace$ is convex, where $A$ is a matrix and $b$ a vector of suitable dimensions. Let $p \in \Delta^n$. The proof follows from

$$A \sum_{i=1}^{n} p_i x^i = \sum_{i=1}^{n} p_i A x^i \leq \sum_{i=1}^{n} p_i b = b \sum_{i=1}^{n} p_i = b \,,$$

where the inequality holds since $p_i \geq 0$, and the last equality holds due to $\sum_{i=1}^{n} p_i = 1$. $\square$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.29</span><span class="math-callout__name">(Convex hull)</span></p>

For $X \subset \mathbb{R}^n$ the set $\lbrace s \in \mathbb{R}^n \mid \exists N \colon s = \sum_{i=1}^{N} p_i x^i, \ x^i \in X, \ p \in \Delta^N \rbrace$ of points representable as a convex combination of a finite number of points of $X$ is called the *convex hull* of $X$ and will be denoted as $\operatorname{conv}(X)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.32</span></p>

Let $Z \subseteq \mathbb{R}^n$ and $X \subseteq Z$. Then $\operatorname{conv}(X) \subseteq \operatorname{conv}(Z)$.

*Proof.* Any $x \in \operatorname{conv}(X)$ is representable as $\sum_{i=1}^{N} p_i x^i$ with some natural $N$, $p \in \Delta^N$ and $x^i \in X$. Since $x^i \in Z$, the point $x = \sum_{i=1}^{N} p_i x^i$ also belongs to $\operatorname{conv}(Z)$. $\square$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.33</span></p>

Let us show that simplex $\Delta^n$ is a convex hull of vectors $\lbrace x^i \in \mathbb{R}^n \mid i = 1, \ldots, n, \ x^i_j = \llbracket i = j \rrbracket \rbrace = \Delta^n \cap \lbrace 0,1 \rbrace^n$. Indeed, any vector $p \in \Delta^n$ is representable as $\sum_{i=1}^{n} p_i x^i$. The other way around, all vectors of the form $\sum_{i=1}^{n} p_i x^i$ belong to $\Delta^n$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 2.34</span><span class="math-callout__name">(Convex hull of a Cartesian product)</span></p>

Let $X = \prod_{i=1}^{n} X^i$ be the Cartesian product of $X^i \subseteq \mathbb{R}^{d_i}$, where $d_i$ is the dimension of the subspace. Then $\operatorname{conv}(X) = \prod_{i=1}^{n} \operatorname{conv}(X^i)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.35</span></p>

Due to Lemma 2.34 the equality $\operatorname{conv}(\lbrace 0,1 \rbrace^n) = [0,1]^n$ follows directly from the fact that $\operatorname{conv}(\lbrace 0,1 \rbrace) = [0,1]$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 2.36</span></p>

Let $X$ be a convex set. Then the set $Z := \lbrace z \mid z = Ax + b, \ x \in X \rbrace$ is convex as well.

*Proof.* Let $z, z' \in Z$. They are representable as $z = Ax + b$ and $z' = Ax' + b$ for some $x, x' \in X$. Consider $p \in (0,1)$ and $pz + (1-p)z' = p(Ax+b) + (1-p)(Ax'+b) = A(px + (1-p)x') + b$. Since $X$ is convex, $px + (1-p)x' \in X$ and, therefore, $pz + (1-p)z' \in Z$. $\square$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 2.37</span><span class="math-callout__name">(Convex hull of a linear mapping image)</span></p>

$\operatorname{conv}\lbrace Ax + b \mid x \in X \rbrace = \lbrace Ax + b \mid x \in \operatorname{conv}(X) \rbrace$, or in other words, convex hull commutes with linear mapping.

</div>

From Definitions 2.25 and 2.29 it follows directly that the convex hull is a convex set. The inverse holds trivially, since each convex set can be seen as its own convex hull. A non-trivial result, stated in the theorem below, describes an important relation between *finite* sets and their convex hulls:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.38</span><span class="math-callout__name">(Minkowski (1896))</span></p>

The set $P \subset \mathbb{R}^n$ is a polytope if and only if it is representable as the convex hull of a finite set of points.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2.39</span></p>

A polytope is the convex hull of its vertices.

</div>

Whereas Corollary 2.39 claims that any polytope is uniquely defined by a finite set (of its vertices) by its convex hull, the following theorem describes vertices of a convex hull of a finite set of points:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.40</span></p>

Let $X \subset \mathbb{R}^n$ be a finite set and $z$ be a vertex of $\operatorname{conv}(X)$. Then $z \in X$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 2.41</span></p>

Let $a \in \mathbb{R}^n$ and $p^* = \arg\min_{p \in \Delta^n} \sum_{i=1}^{n} p_i a_i$. Then $p^*_j = 0$ for all $j$ such that $a_j > \min_{i=1,\ldots,n} \lbrace a_i \rbrace$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.42</span></p>

According to Example 2.16, $[0,1]^n$ is a polytope. Let us show that the set of its vertices is precisely the set $\lbrace 0,1 \rbrace^n$.

Since $\operatorname{conv}(\lbrace 0,1 \rbrace^n) = [0,1]^n$ (see Example 2.35), Theorem 2.40 implies $\operatorname{vrtx}([0,1]^n) \subseteq \lbrace 0,1 \rbrace^n$. The reverse inclusion $\lbrace 0,1 \rbrace^n \subseteq \operatorname{vrtx}([0,1]^n)$ is proven in Example 2.21.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.43</span></p>

$\Delta^n \cap \lbrace 0,1 \rbrace^n = \operatorname{vrtx}(\Delta^n)$. In other words, the binary vectors of a simplex are all its vertices.

Since $\Delta^n = \operatorname{conv}(\Delta^n \cap \lbrace 0,1 \rbrace^n)$ (see Example 2.33), Theorem 2.40 implies $\operatorname{vrtx}(\Delta^n) \subseteq \Delta^n \cap \lbrace 0,1 \rbrace^n$. The inverse inclusion $\Delta^n \cap \lbrace 0,1 \rbrace^n \subseteq \operatorname{vrtx}(\Delta^n)$ is proven in Example 2.23.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.44</span></p>

Let $X$ be a polytope and $Z$ be a polyhedron in $\mathbb{R}^n$, and $x \in X \cap Z$. If additionally $x$ is a vertex of $X$, then $x$ is a vertex of $X \cap Z$.

*Proof.* Since $x$ is a vertex of $X$, there exists $c \in \mathbb{R}^n$ such that $x$ is the unique solution of $\min_{x' \in X} \langle c, x' \rangle$. Since $x \in Z$, it holds also that it is a unique solution of $\min_{x' \in X \cap Z} \langle c, x' \rangle$ (see Proposition 2.13), and, therefore, $x$ is a vertex of $X \cap Z$. $\square$

</div>

### Linear Programs

Linear programs (LP) are the most important components of integer linear programs — a unified representation for combinatorial problems. We will see below that although linear programs constitute a continuous, non-discrete object, the set of points, which is sufficient to check to obtain their exact solution, is finite. Moreover, it can be shown that this set grows exponentially with the dimension of the problem in general. In this sense linear programs can be seen as combinatorial problems themselves. The main difference, however, is in their solvability: There exist polynomial algorithms for any linear program, whereas the class of combinatorial problems contains $\mathcal{NP}$-hard problems.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.45</span><span class="math-callout__name">(Linear program)</span></p>

Let $P$ be a polyhedron in $\mathbb{R}^n$ defined by a set of linear constraints. Optimization problems of the form

$$\min_{x \in P} \langle c, x \rangle \qquad \text{and} \qquad \max_{x \in P} \langle c, x \rangle$$

are called *linear programs (LP)*.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 2.46</span></p>

For any $a \in \mathbb{R}^n$ it holds that

$$\min_{i=1,\ldots,n} \lbrace a_i \rbrace = \min_{p \in \Delta^n} \sum_{i=1}^{n} p_i a_i = \min_{\mu \in \operatorname{conv}\lbrace a_i, \ i=1,\ldots,n \rbrace} \mu \,.$$

*Proof.* The first equality directly follows from Lemma 2.41 and the second equality holds per definition of the convex hull. $\square$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2.47</span></p>

For any $c \in \mathbb{R}^n$ and any finite $X \subset \mathbb{R}^n$ it follows

$$\min_{x \in X} \langle c, x \rangle = \min_{x \in \operatorname{conv}(X)} \langle c, x \rangle \,.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2.48</span><span class="math-callout__name">(Combinatorial property of linear programs)</span></p>

For any $c \in \mathbb{R}^n$ and any polytope $P$ it holds that

$$\min_{x \in P} \langle c, x \rangle = \min_{x \in \operatorname{vrtx}(P)} \langle c, x \rangle \,.$$

*Proof.* The statement follows directly from $P = \operatorname{conv}(\operatorname{vrtx}(P))$ (Corollary 2.39) and Corollary 2.47. $\square$

</div>

Corollary 2.48 claims that to solve a linear program over a polytope it is sufficient to evaluate the objective on the finite set of vertices of the polytope. This does not mean that the minimum is attained in a non-vertex point of the polytope. However, it guarantees that there will always be a vertex corresponding to the minimal value of the objective. Solutions which correspond to vertices of a polytope $P$ are called *basic solutions* and can be found with the famous simplex algorithm.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 2.49</span></p>

For any polytope $P$ it holds

$$\arg\min_{\mathbf{x} \in P} \langle \mathbf{c}, \mathbf{x} \rangle = \operatorname{conv}\!\left( \arg\min_{\mathbf{x} \in \operatorname{vrtx}(P)} \langle \mathbf{c}, \mathbf{x} \rangle \right).$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2.50</span></p>

A face of a polytope is a polytope itself.

*Proof.* By definition, a face is the set representable as $\arg\min_{\mathbf{x} \in P} \langle \mathbf{c}, \mathbf{x} \rangle$ for some $\mathbf{c}$. Due to Lemma 2.49 it is representable as a convex hull of a finite set of points, therefore, due to Theorem 2.38 it is a polytope. $\square$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2.51</span></p>

Vertices of a polytope face are vertices of the polytope itself.

*Proof.* Follows from the proof of Lemma 2.49, as the set $\hat{P} \subseteq \operatorname{vrtx}(P)$ is the subset of vertices of the face that is obtained by minimizing the cost vector $\mathbf{c}$ over $P$. $\square$

</div>

### Integer Linear Programs

The integer linear program (ILP) is a unified format to represent combinatorial problems. The main advantage of formulating combinatorial problems as ILPs, is the unified geometrical (polyhedral) interpretation common to all integer linear programs. This has allowed for significant progress in creating standardized methods to solve them.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.52</span><span class="math-callout__name">(Integer linear program)</span></p>

A linear program with additional constraints allowing all the variables to take only values 0 or 1, e.g.

$$\min_{x \in P \cap \{0,1\}^n} \langle c, x \rangle$$

is called a *0/1 integer linear program* (further we omit mentioning '0/1' for brevity). Here $P$ is a polyhedron in $\mathbb{R}^n$. Constraints of the form $x \in \lbrace 0,1 \rbrace^n$ are called *integrality constraints*.

</div>

The class of integer linear programs is $\mathcal{NP}$-hard, as is shown by the following two examples:

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.53</span><span class="math-callout__name">(Binary knapsack problem)</span></p>

Let $\lbrace 1, \ldots, n \rbrace$ be indexes of $n$ items, each of which has its volume $a_i$ and its cost $c_i$. Therefore, we assume two vectors $\mathbf{a}, \mathbf{c} \in \mathbb{R}^n$ to be given. The task is to select a subset of items to maximize their total cost, whereas their total volume should not exceed a given (knapsack) volume $b$. This can be equivalently formulated as the integer linear program

$$\max_{\mathbf{x} \in \{0,1\}^n} \langle \mathbf{c}, \mathbf{x} \rangle$$

$$\text{s.t. } \langle \mathbf{a}, \mathbf{x} \rangle \leq b \,.$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.54</span><span class="math-callout__name">(Max-weight independent set problem)</span></p>

Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ be an undirected graph with a *node set* $\mathcal{V}$ and an *edge set* $\mathcal{E} \subseteq \binom{\mathcal{V}}{2}$. An *independent* or *stable set* is the subset of mutually non-adjacent vertices $\mathcal{V}' \subseteq \mathcal{V}$, e.g. for any $i, j \in \mathcal{V}'$ it holds $\lbrace i, j \rbrace \notin \mathcal{E}$. Given the costs $c_i$, $i \in \mathcal{V}$, the *maximum weight independent set (MWIS)* problem consists in finding an independent set with the maximum total cost.

Introduce a binary variable $x_i \in \lbrace 0,1 \rbrace$ for each node $i \in \mathcal{V}$. Its value 1 denotes that the respective node belongs to the maximum-weight independent set. This leads to an ILP formulation:

$$\max_{\mathbf{x} \in \{0,1\}^{\mathcal{V}}} \langle \mathbf{c}, \mathbf{x} \rangle$$

$$\text{s.t. } x_i + x_j \leq 1, \ \lbrace i, j \rbrace \in \mathcal{E} \,.$$

</div>

**Universality of an ILP representation.** Any combinatorial optimization problem $\min_{\mathbf{x} \in S \subset \mathbb{Z}^n} f(\mathbf{x})$ can be formulated as an integer programming problem if $S$ is finite:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.56</span></p>

Any combinatorial optimization problem $\min_{\mathbf{x} \in S \subset \mathbb{Z}^n} f(\mathbf{x})$ can be formulated as an integer programming problem if $S$ is finite.

</div>

**ILP as LP.** Somewhat surprisingly, any ILP can be represented as a linear program by a specially constructed polytope. Note that the feasible set of the problem $\min_{x \in P \cap \{0,1\}^n} \langle c, x \rangle$ is finite, therefore one can use Corollary 2.47, which implies

$$\min_{x \in P \cap \{0,1\}^n} \langle c, x \rangle = \min_{x \in \operatorname{conv}(P \cap \{0,1\}^n)} \langle c, x \rangle \,,$$

and, since according to Theorem 2.38, the convex hull of a finite set is a polytope, the right-hand side constitutes a linear program. This fact does not mean polynomial solvability of the right-hand-side and, therefore, of the ILP. This is because the polytope $\operatorname{conv}(P \cap \lbrace 0,1 \rbrace^n)$ may require exponentially many linear (in)equalities to be specified explicitly.

We will call the polytope $\operatorname{conv}(P \cap \lbrace 0,1 \rbrace^n)$ *the integer hull* of $P$, although this name has a broader meaning as the convex set of all integer points (not only those having coordinates 0 and 1) in $P$.

**Vertices of the integer hull.**

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 2.57</span></p>

Let $X \subseteq \lbrace 0,1 \rbrace^n$. For any $x \in \operatorname{conv}(X)$ it holds $0 \leq x_i \leq 1$, that is, $\operatorname{conv}(X) \subseteq [0,1]^n$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2.58</span></p>

For any $x \in \operatorname{conv}(P \cap \lbrace 0,1 \rbrace^n)$ it holds that $0 \leq x_i \leq 1$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.59</span></p>

For $X \subseteq \lbrace 0,1 \rbrace^n$ it holds that $\operatorname{vrtx}(\operatorname{conv}(X)) = X$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2.60</span></p>

$\operatorname{vrtx}(\operatorname{conv}(P \cap \lbrace 0,1 \rbrace^n)) = P \cap \lbrace 0,1 \rbrace^n$. In other words, each feasible point of a 0/1 ILP is a vertex of its integer hull.

</div>

### Linear Program Relaxation

The simplest approximation of an $\mathcal{NP}$-hard integer linear program by a polynomially solvable linear problem can be constructed by omitting integrality constraints. This approximation, known as *linear program relaxation*, or simply *LP relaxation*, is a very important and powerful tool for obtaining approximate and exact solutions of integer linear programs.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.61</span><span class="math-callout__name">(LP relaxation)</span></p>

The linear program

$$\min_{x \in P \cap [0,1]^n} \langle c, x \rangle$$

is called *linear programming (LP) relaxation* of the integer linear program $\min_{x \in P \cap \{0,1\}^n} \langle c, x \rangle$.

</div>

Note that

- the LP relaxation is constructed by substituting the integrality constraints $x_i \in \lbrace 0,1 \rbrace$ with *interval* or *box* constraints $x_i \in [0,1]$. Since $x_i \in [0,1]$ is equivalent to $0 \leq x_i \leq 1$, these constraints define a polyhedron. As the intersection of two polyhedra is a polyhedron (see Proposition 2.19), the problem is a linear program;
- the LP relaxation is a relaxation in terms of Definition 2.5, since its feasible set is a superset of the one for the non-relaxed problem and their objectives coincide.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.62</span><span class="math-callout__name">(Properties of LP relaxations)</span></p>

Let $f^*$ and $\hat{f}$ be optimal values of the ILP problem and its LP relaxation respectively. Let also $\hat{x}$ be a solution of the LP relaxation. Then the following assertions hold:

1. $\hat{f} \leq f^*$.
2. If $\hat{x} \in \lbrace 0,1 \rbrace^n$, then $\hat{x}$ is a solution of the non-relaxed ILP problem.
3. $P \cap [0,1]^n \supseteq \operatorname{conv}(P \cap \lbrace 0,1 \rbrace^n)$, that is, the feasible set of the LP relaxation is a superset of the integer hull.

</div>

#### Vertices of the Feasible Set of the LP Relaxation

While statement (2) of Proposition 2.62 claims that integrality of a relaxed solution implies its optimality for the non-relaxed ILP problem, it does not say, whether there has to be a solution to the LP relaxation which is integral.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.63</span></p>

For any polyhedron $P$ it holds that $P \cap \lbrace 0,1 \rbrace^n \subseteq \operatorname{vrtx}(P \cap [0,1]^n)$.

In other words, all feasible vectors of an ILP are vertices of the feasible set of its LP relaxation and, therefore, can be reached as a relaxed solution.

*Proof.* Consider the intersection of the polyhedron $P$ and the polytope $[0,1]^n$. According to Proposition 2.44, vertices of $[0,1]^n$ belonging to $P$ are also vertices of $P \cap [0,1]^n$. As shown in Example 2.42, any vertex of the $n$-dimensional cube $[0,1]^n$ is a vector from the set $\lbrace 0,1 \rbrace^n$ and vice versa. Therefore, any point from $P \cap \lbrace 0,1 \rbrace^n$ is a vertex of $P \cap [0,1]^n$. $\square$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2.64</span></p>

$\operatorname{vrtx}(\operatorname{conv}(P \cap \lbrace 0,1 \rbrace^n)) \subseteq \operatorname{vrtx}(P \cap [0,1]^n)$.

*Proof.* The statement follows from $\operatorname{vrtx}(\operatorname{conv}\lbrace P \cap \lbrace 0,1 \rbrace^n \rbrace) = P \cap \lbrace 0,1 \rbrace^n$ (Corollary 2.60), and the second holds due to Proposition 2.63. $\square$

</div>

The last proposition of the chapter states that there is no other binary vector in the feasible set of LP relaxation than those, which are feasible for the non-relaxed integer linear program:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.65</span></p>

For any polyhedron $P \subset \mathbb{R}^n$ it holds that $\operatorname{vrtx}(P \cap [0,1]^n) \cap \lbrace 0,1 \rbrace^n = P \cap \lbrace 0,1 \rbrace^n$.

*Proof.* Let $x \in P \cap \lbrace 0,1 \rbrace^n$. Trivially $x \in \lbrace 0,1 \rbrace^n$. According to Proposition 2.63, $x \in \operatorname{vrtx}(P \cap [0,1]^n)$. Therefore, $P \cap \lbrace 0,1 \rbrace^n \subseteq \operatorname{vrtx}(P \cap [0,1]^n) \cap \lbrace 0,1 \rbrace^n$.

The other way around, let $x \in \operatorname{vrtx}(P \cap [0,1]^n) \cap \lbrace 0,1 \rbrace^n$. Trivially, $x \in \operatorname{vrtx}(P \cap [0,1]^n) \subseteq P \cap [0,1]^n$, it holds $x \in P$. Therefore, $x \in P \cap \lbrace 0,1 \rbrace^n$. $\square$

</div>

## Chapter 3: Linearization: From Quadratic to Linear Integer Objective

In this chapter we consider linearly constrained integer programs with quadratic objectives such as *labeling* and *quadratic assignment*. The main goal is to turn them into *linear* integer programs in the most efficient way, where efficiency means the relation between the number of (additional) constraints and tightness of the resulting LP relaxation.

The problems will deal with the form

$$\min_{\mathbf{x} \in P \cap \lbrace 0,1 \rbrace^n} \sum_{i=1}^{n} c_i x_i + \sum_{\lbrace i,j \rbrace \in \mathcal{E}} c_{\lbrace ij \rbrace} x_i x_j \,,$$

where the polytope $P$ is defined by a set of linear constraints, $\mathcal{E} \subseteq \binom{[n]}{2}$ and the brackets "$\lbrace \cdot \rbrace$" in $y_{\lbrace ij \rbrace}$ denote that $\lbrace ij \rbrace$ and $\lbrace ji \rbrace$ are the same. Respectively $c_{\lbrace ij \rbrace}$ and $c_{\lbrace ji \rbrace}$ are the same as well.

A general approach to transformation of this problem to ILP format consists of two steps:

- Introduce a new variable $y_{\lbrace ij \rbrace}$ for each $\lbrace i, j \rbrace \in \mathcal{E}$.
- Add constraints $y_{\lbrace ij \rbrace} = x_i x_j$, $\lbrace i, j \rbrace \in \mathcal{E}$ to the feasible set.

This turns the problem into

$$\min_{\substack{\mathbf{x} \in P \cap \lbrace 0,1 \rbrace^n,\, \mathbf{y} \in \lbrace 0,1 \rbrace^{\mathcal{E}} \\ y_{\lbrace ij \rbrace} = x_i x_j,\; ij \in \mathcal{E}}} \sum_{i=1}^{n} c_i x_i + \sum_{\lbrace i,j \rbrace \in \mathcal{E}} c_{\lbrace ij \rbrace} y_{\lbrace ij \rbrace} \,.$$

The above process of variable substitution $y_{\lbrace ij \rbrace} = x_i x_j$ is often referred to as *variable lifting* in the literature. Respectively, variables $y_{\lbrace ij \rbrace}$ are called *lifted* and the resulting integer program a *lifted* problem.

Although the objective is now linear, the constraints are not anymore. So the main question we deal with in this chapter is how to linearize the constraints $y_{\lbrace ij \rbrace} = x_i x_j$ given that $\mathbf{x}$ and $\mathbf{y}$ are binary vectors.

### 3.1 Fortet's Linearization

The first linearization method proposed by Fortet is the substitution of $y_{\lbrace ij \rbrace} = x_i x_j$ by

$$y_{\lbrace ij \rbrace} \leq x_i \,,\quad y_{\lbrace ij \rbrace} \leq x_j \,,\quad y_{\lbrace ij \rbrace} \geq x_i + x_j - 1 \,,$$

for all $\lbrace i, j \rbrace \in \mathcal{E}$.

It is easy to see that for any binary $\mathbf{x}$ and $\mathbf{y}$ these constraints are equivalent to $y_{\lbrace ij \rbrace} = x_i x_j$.

This adds $3\lvert \mathcal{E} \rvert$ additional non-trivial constraints (in addition to the integrality constraints $\mathbf{y} \in \lbrace 0,1 \rbrace^{\mathcal{E}}$ and/or the non-negativity constraints $\mathbf{y} \geq 0$ important for the LP relaxation).

Being very simple this linearization method is known for its main weakness: The resulting LP relaxation is typically not particularly tight. Therefore, several other linearizations have been proposed. We consider the one known to be tighter than most of others in the following section.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.1</span></p>

Consider the problem

$$\min_{\mathbf{x} \in \lbrace 0,1 \rbrace^2} c_1 x_1 + c_2 x_2 + c_{12} x_1 x_2$$

$$\text{s.t. } x_1 + x_2 = 1 \,.$$

Due to the simplex constraint the quadratic term is always zero for any integer solution.

Let us now consider linearization according to Fortet's constraints:

$$\min_{\substack{\mathbf{x} \in \lbrace 0,1 \rbrace^2 \\ y \in \lbrace 0,1 \rbrace}} c_1 x_1 + c_2 x_2 + c_{12} y$$

$$\text{s.t. } x_1 + x_2 = 1 \,,\quad y \leq x_1 \,,\quad y \leq x_2 \,,\quad y \geq x_1 + x_2 - 1 \,.$$

Consider now the LP relaxation:

$$\min_{\substack{\mathbf{x} \in [0,1]^2 \\ y \in [0,1]}} c_1 x_1 + c_2 x_2 + c_{12} y$$

$$\text{s.t. } x_1 + x_2 = 1 \,,\quad y \leq x_1 \,,\quad y \leq x_2 \,,\quad y \geq 0 \,,$$

where we also simplified the last constraint given the first one. Since $y \in [0,1]$ anyway, we can ignore this constraint line further.

Now, the value of $y$ is constrained by the values of $x_1$ and $x_2$ and given the first constraint, $y \leq 1/2$ and $y$ attains its maximum value if $x_1 = x_2 = 1/2$. This implies, that if $c_{12}$ is negative and large compared to $c_1$ and $c_2$, the solution of the LP relaxation will be $(1/2, 1/2, 1/2)$.

</div>

### 3.2 Sherali-Adams Linearization

Is there a generic mechanism of building valid constraints for $y = x_i x_j$ given the constraints on $\mathbf{x}$?

The answer is given by Sherali and Adams. Consider the following general idea: If $\langle \mathbf{a}, \mathbf{x} \rangle \leq b$ is a valid constraint, then $x_j(\langle \mathbf{a}, \mathbf{x} \rangle \leq b)$ is a valid constraint also, since $x_j \in \lbrace 0, 1 \rbrace \geq 0$. Therefore,

$$x_j(\langle \mathbf{a}, \mathbf{x} \rangle \leq b) \;\Leftrightarrow\; \sum_{i=1}^{n} a_i \underbrace{x_i x_j}_{y_{\lbrace ij \rbrace}} \leq b x_j$$

is a valid constraint either. Since $(1 - x_j) \in \lbrace 0, 1 \rbrace > 0$ as well, feasibility holds also for

$$(1 - x_j)(\langle \mathbf{a}, \mathbf{x} \rangle \leq b) \;\Leftrightarrow\; \langle \mathbf{a}, \mathbf{x} \rangle - \sum_{i=1}^{n} a_i \underbrace{x_i x_j}_{y_{\lbrace ij \rbrace}} \leq b - b x_j \,.$$

Note that both of these are valid linear constraints that include the new lifted variable $\mathbf{y}$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.2</span></p>

Consider constraints $x_i \geq 0$, $x_i \leq 1$, $i \in [n]$. After multiplication of them to $x_j$ one obtains

$$x_i x_j \geq 0 \,,\quad x_i x_j \leq x_j \,,\quad i \in [n]$$

and after multiplication to $(1 - x_j)$

$$x_i - x_i x_j \geq 0 \,,\quad x_i - x_i x_j \leq 1 - x_j \,,\quad i \in [n] \,.$$

Easy to see, that after substitution $y_{\lbrace ij \rbrace} = x_i x_j$ these result in the Fortet's linearization constraints plus the trivial constraint $y_{\lbrace ij \rbrace} \geq 0$.

In other words, Sherali-Adams linearization contains Fortet's linearization "automatically", since w.l.o.g. one can assume that the box constraints $x \in [0, 1]$ are fulfilled for all integer variables.

</div>

And what about equality constraints $\langle \mathbf{a}, \mathbf{x} \rangle = b$? Multiplication of this constraint by $x_j$ and $(1 - x_j)$ is equivalent:

$$(1 - x_j)(\langle \mathbf{a}, \mathbf{x} \rangle = b) \;\Leftrightarrow\; -\sum_{i=1}^{n} a_i \underbrace{x_i x_j}_{y_{\lbrace ij \rbrace}} = b - b x_j \,.$$

However, since $\langle \mathbf{a}, \mathbf{x} \rangle = b$, it simplifies to

$$x_j(\langle \mathbf{a}, \mathbf{x} \rangle = b) \;\Leftrightarrow\; \sum_{i=1}^{n} a_i \underbrace{x_i x_j}_{y_{\lbrace ij \rbrace}} = b x_j \,.$$

Therefore, in the equality case there it is sufficient to multiply by $x_j$ for all $j$.

#### 3.2.1 General Sherali-Adams Linearization Scheme

This section summarizes the observations above. Consider the following general zero-one quadratic programming problem with linear constraints:

$$\min_{\mathbf{x} \in P \cap \lbrace 0,1 \rbrace^n} \sum_{i=1}^{n} c_i x_i + \sum_{ij \in \mathcal{E}} c_{ij} x_i x_j \,,$$

where $\mathcal{E} \subseteq \lbrace (i,j) \colon 1 < i < j < n \rbrace$ and $P := P_1 \cup P_2 \cup P_3$ is given by:

$$P_1 = \lbrace x \in \mathbb{R}^n \colon \sum_{i=1}^{n} a_{ki} x_i = b_k \text{ for } k \in [K] \rbrace \,,$$

$$P_2 = \lbrace x \in \mathbb{R}^n \colon \sum_{i=1}^{n} G_{li} x_i \geq g_l \text{ for } l \in [L] \rbrace \,,$$

$$P_3 = \lbrace x \in \mathbb{R}^n \colon 0 \leq x_i \leq 1 \text{ for } i \in [n] \rbrace \,.$$

Perform now the following set of operations:

1. Form $nK$ constraints $x_j \bigl(\sum_{i=1}^{n} a_{ki} x_i = b_k\bigr)$, $k \in [K]$, $j \in [n]$;
2. Form $nK$ constraints $x_j \bigl(\sum_{i=1}^{n} G_{li} x_i \geq g_l\bigr)$, $l \in [L]$, $j \in [n]$;
3. Form $nK$ constraints $(1 - x_j)\bigl(\sum_{i=1}^{n} G_{li} x_i \geq g_l\bigr)$, $l \in [L]$, $j \in [n]$;
4. Form $3n(n-1)/2$ constraints:
   - $(1 - x_j)(x_i \geq 0)$, $i \in [n]$, $j \in [m] \setminus \lbrace i \rbrace$,
   - $(1 - x_j)(x_i \leq 1)$, $i \in [n-1]$, $j = i+1, \ldots, n$.

Then substitute $y_{\lbrace ij \rbrace} := x_i x_j$ for $i < j$ and $x_i^2$ by $x_i$ as $x^2 = x$ for any $x \in \lbrace 0, 1 \rbrace$.

This results in:

$$\min_{\mathbf{x} \in \lbrace 0,1 \rbrace^n} \sum_{i=1}^{n} c_i x_i + \sum_{ij \in \mathcal{E}} c_{ij} x_i x_j$$

$$\text{s.t. } (a_{kj} - b_k) x_j + \sum_{i \in [n] \setminus \lbrace j \rbrace} a_{ki} y_{\lbrace ij \rbrace} = 0 \,,\quad \forall (j, k) \in [n] \times [K]$$

$$\sum_{i=1}^{n} G_{li} x_i - g_l \geq \sum_{i \in [n] \setminus \lbrace j \rbrace} G_{li} y_{\lbrace ij \rbrace} + (G_{lj} - g_l) x_j \geq 0 \,,\quad \forall (j, l) \in [n] \times [L]$$

$$y_{\lbrace ij \rbrace} \geq 0 \,,\quad \forall \lbrace i, j \rbrace \in \tbinom{[n]}{2}$$

$$x_i - y_{\lbrace ij \rbrace} \geq 0 \,,\quad \forall \lbrace i, j \rbrace \in \tbinom{[n]}{2}$$

$$y_{\lbrace ij \rbrace} - x_i - x_j \geq -1 \,,\quad \forall \lbrace i, j \rbrace \in \tbinom{[n]}{2}$$

$$\mathbf{x} \in P_1 \,,\quad \mathbf{x} \in P_3$$

Note that:

- The initial constraint $\mathbf{x} \in P_2$ is not included explicitly as it is implied by the linearized constraints;
- If the constraints of $P_1$ and $P_2$ together imply those in $P_3$, the Fortet's constraints can be excluded as they are implied by the Sherali-Adams constraints.
- Other simplifications are often possible. E.g. if $P_1$ contains a *simplex* (or *uniqueness* or *set partitioning*) constraint $x_1 + \cdots + x_m = 1$ then $y_{\lbrace ij \rbrace} = 0$ for all $i, j \in [m]$, $i \neq j$, see Example 3.3 below.
- Contrary to the Fortet linearization that introduces new variables $y_{\lbrace ij \rbrace}$ only for $\lbrace i, j \rbrace \in \mathcal{E}$ the Sherali-Adams method introduces them for all $\binom{[n]}{2}$. In case of sparse quadratic terms, i.e. $\lvert \mathcal{E} \rvert \ll \lvert \binom{[n]}{2} \rvert$ this significantly increases the size of the initial problem. In some cases the unnecessary variables and constraints can be ignored without influencing the resulting LP relaxation. In general, one may consider less tight relaxation by defining only a subset of quadratic variables $y_{\lbrace ij \rbrace}$ and the respective constraints.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.3</span></p>

Now let us get back to Example 3.1. Multiplication of the simplex constraint $x_1 + x_2 = 1$ by $x_1$ results in $x_1^2 + x_1 x_2 = x_1 \Rightarrow x_1 + y = x_1 \Rightarrow y = 0$. The same result is obtained with multiplication by $x_2$.

Hence the result of linearization reads now

$$\min_{\mathbf{x} \in \lbrace 0,1 \rbrace^2} c_1 x_1 + c_2 x_2 + c_{12} y$$

$$\text{s.t. } x_1 + x_2 = 1 \,,\quad y = 0 \,,$$

or, after getting rid of $y$:

$$\min_{\mathbf{x} \in \lbrace 0,1 \rbrace^2} c_1 x_1 + c_2 x_2$$

$$\text{s.t. } x_1 + x_2 = 1 \,.$$

Note that LP relaxation of this problem is tight, contrary to the relaxation with the Fortet's linearization.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.4</span><span class="math-callout__name">(Uniqueness/simplex constraints)</span></p>

Let $\mathbf{x} \in \mathbb{R}^n$. Consider the *uniqueness* constraint

$$\sum_{i \in I \subseteq [n]} x_i = 1 \,.$$

After its multiplication by $x_j$, $j \in I$, we obtain

$$x_j^2 + \sum_{i \in I \setminus \lbrace j \rbrace} x_i x_j = x_j$$

$$\sum_{i \in I \setminus \lbrace j \rbrace} x_i x_j = 0$$

$$0 \leq y_{\lbrace ij \rbrace} := x_i x_j \;\Rightarrow\; y_{\lbrace ij \rbrace} = 0 \,.$$

If we multiply the constraint by $x_j$, $j \notin I$ this results in

$$\sum_{i \in I} y_{\lbrace ij \rbrace} = x_j \,.$$

The latter constraint is often called *marginalization* or *coupling constraint*.

</div>

### 3.3 Case Study: Labeling Problem

Consider the labeling problem as defined in (1.5):

$$\min_{\mathbf{y} \in \mathcal{Y}_\mathcal{V}} \sum_{u \in \mathcal{V}} \theta_u(y_u) + \sum_{uv \in \mathcal{E}} \theta_{uv}(y_u, y_v) \,.$$

Let $\mathcal{I}_1 := \cup_{u \in \mathcal{V}} \mathcal{Y}_u$ be the set of *all labels in all nodes* of the labeling problem. Introduce a binary vector

$$\mathbb{R}^{\mathcal{I}_1} \ni \boldsymbol{\mu} = (\mu_u(s) \in \lbrace 0, 1 \rbrace \colon u \in \mathcal{V}, s \in \mathcal{Y}_u)$$

whose coordinates correspond to labels in different nodes. As usual, the value 1 will denote selection of the respective label. With this notation the labeling problem can be rewritten as a *quadratic integer problem*

$$\min_{\boldsymbol{\mu} \in \lbrace 0,1 \rbrace_{\geq 0}^{\mathcal{I}_1 \cup \mathcal{I}_2}} \sum_{u \in \mathcal{V}} \sum_{s \in \mathcal{Y}_u} \theta_u(s) \mu_u(s) + \sum_{uv \in \mathcal{E}} \sum_{sl \in \mathcal{Y}_{uv}} \theta_{uv}(s,l) \mu_u(s) \mu_v(l)$$

$$\text{s.t. } \sum_{s \in \mathcal{Y}_u} \mu_u(s) = 1 \,,\; u \in \mathcal{V} \,.$$

Let us now linearize it with the Sherali-Adams method.

- First of all note that the *uniqueness* (known also as *simplex*) *constraints* together with non-negativity of $\boldsymbol{\mu}$ imply $\mu \in [0,1]^{\mathcal{I}_1}$. Therefore, we can omit the Fortet's constraints as they will be implied by the other constraints of the linearized problem.
- Since the only non-trivial constraints are *uniqueness constraints*, according to Example 3.4:
  - $\mu_u(s)\mu_v(l) = 0$ for any $s \neq l$ for all $u \in \mathcal{V}$.
  - By introducing $\mu_{uv}(s,l) := \mu_u(s)\mu_v(l)$ the uniqueness constraints are turned into the marginalization constraint

$$\sum_{s \in \mathcal{Y}_u} \mu_{uv}(s, l) = \mu_v(l) \,,\; u \in \mathcal{V}$$

after multiplication by $\mu_v(l)$, $v \in \mathcal{V}$, $v \neq u$, $l \in \mathcal{Y}_s$.

- Note, however, that the only non-zero cost in the original problem are given for $\mu_{uv}(\cdot)$ for $uv \in \mathcal{E}$. Therefore, the values of $\mu_{uv} := (\mu_{uv}(s,l) \colon s \in \mathcal{Y}_u, \; l \in \mathcal{Y}_v)$ for $uv \notin \mathcal{E}$ do not influence the objective value. But do they existence additionally constrain other variables be it binary or relaxed ones? The following statement gives the negative answer to this question:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 3.5</span></p>

The sets

$$P_x = \left\lbrace \mathbf{x} \in \mathbb{R}_{\geq 0}^{I \cup J} \colon \sum_{i \in I} x_i = 1 \,,\; \sum_{j \in J} x_j = 1 \right\rbrace$$

and

$$P_y = \left\lbrace \mathbf{x} \in P_x \mid \exists \mathbf{y} \in \mathbb{R}_{\geq 0}^{I \times J} \colon \sum_{i \in I} y_{ij} = x_j \,,\; \forall j \in J \,,\;\; \sum_{j \in J} y_{ij} = x_i \,,\; \forall i \in I \right\rbrace$$

are equal, i.e. $P_x = P_y$. Moreover, when $\mathbf{x} \in P_x \cap \lbrace 0,1 \rbrace^{I \cup J}$, then the condition for $\mathbf{y}$ can be strengthened by assuming $\mathbf{y} \in \lbrace 0,1 \rbrace^{I \times J}$.

*Proof.* It is sufficient to prove that for any $\mathbf{x} \in P_x$ there exists $\mathbf{y} \in \mathbb{R}_{\geq 0}^{I \times J}$ such that the additional two constrains in the definition of $P_y$ are fulfilled. Indeed, consider $y_{ij} = x_i x_j$. Then

$$\sum_{i \in I} y_{ij} = \sum_{i \in I} x_i x_j = x_j \underbrace{\sum_{i \in I} x_i}_{\mathbf{x} \in P_x \Rightarrow \sum_{i \in I} x_i = 1} = x_j \,,$$

$$\sum_{j \in J} y_{ij} = \sum_{j \in J} x_i x_j = x_i \underbrace{\sum_{j \in J} x_j}_{\mathbf{x} \in P_x \Rightarrow \sum_{j \in J} x_j = 1} = x_i \,.$$

The additional constraint $\mathbf{x} \in \lbrace 0,1 \rbrace^{I \cup J}$ implies $y_{ij} = x_i x_j \in \lbrace 0,1 \rbrace$. $\square$

</div>

Putting all together for the labeling problem, we obtain the following linear constraints for the linearized (lifted) labeling problem:

$$\mathcal{L} := \begin{cases} \sum_{l \in \mathcal{Y}_v} \mu_{uv}(s, l) = \mu_u(s) \,,\quad u \in \mathcal{V},\; v \in \mathcal{N}(u),\; s \in \mathcal{Y}_u & \text{(a)} \\ \sum_{l \in \mathcal{Y}_v} \mu_v(l) = 1 \,,\quad v \in \mathcal{V} & \text{(b)} \\ \boldsymbol{\mu} \geq 0 \,, & \text{(c)} \end{cases}$$

The set $\mathcal{L}$ is known as *local (marginal) polytope*.

The resulting (lifted) linearized labeling problem takes the form

$$\min_{\boldsymbol{\mu} \in \lbrace 0,1 \rbrace^{\mathcal{I}_1 \cup \mathcal{I}_2} \cap \mathcal{L}} \underbrace{\sum_{u \in \mathcal{V}} \sum_{s \in \mathcal{Y}_u} \theta_u(s) \mu_u(s) + \sum_{uv \in \mathcal{E}} \sum_{sl \in \mathcal{Y}_{uv}} \theta_{uv}(s,l) \mu_{uv}(s,l)}_{\langle \boldsymbol{\theta}, \boldsymbol{\mu} \rangle} \,.$$

Here the set $\mathcal{I}_2 = \sum_{uv \in \mathcal{E}} \lvert \mathcal{Y}_{uv} \rvert$ stands for the total number of the lifted, or *pairwise* variables. In the following we will denote $\mathcal{I} = \mathcal{I}_1 \cup \mathcal{I}_2$. Note that $\mathcal{L} \cap [0,1]^{\mathcal{I}} = \mathcal{L}$, so the LP relaxation of the linearized problem reads

$$\min_{\boldsymbol{\mu} \in \mathcal{L}} \langle \boldsymbol{\theta}, \boldsymbol{\mu} \rangle \,.$$

Naturally, this relaxation referred to as the *local polytope relaxation* of the labeling problem.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.6</span><span class="math-callout__name">(How large-scale is the ILP representation of the labeling problem?)</span></p>

Consider the depth reconstruction Example 1.11 from Chapter 1. Each element of the nodes set $\mathcal{V}$ corresponds to a pixel in an input image. Let the latter be as small as $100 \times 100 = 10^4$. Let also each node be associated with 10 labels only. It implies there are in total $10^{10^4}$ labelings.

To estimate the number of variables in the ILP representation we count for $10 \times 10^4 = 10^5$ total number of labels (corresponding to the variables $\mu_u(s)$).

The number of edges in the grid graph is $\approx 2 \times 10^4$ and each edge is associated with $10^2$ label pairs (and the corresponding variables $\mu_{uv}(s,l)$). Therefore, the total number of label pairs is $\approx 2 \times 10^4 \times 10^2 = 2 \times 10^6$.

This results in $O(10^6)$ variables in the ILP representation.

As it follows from the definition of the local polytope $\mathcal{L}$, the number of equality constraints grows linearly with the total number of labels and therefore is of order $O(10^5)$. However, since there is the non-negativity constraint $\mu_w(s) \geq 0$ for each variable, the total number of constraints has the same order $O(10^6)$ as the number of variables.

The millions of constraints and variables turn the labeling problem to the class of *large-scale optimization* problems. Large size of the problem significantly limits the set of methods, which can be applied to tackle it.

</div>

#### 3.3.1 ILP Properties of the Labeling Problem

Due to the ILP representation of the labeling problem, we can directly make conclusions about its own properties and the properties of its natural LP relaxation:

- The integer hull of the set of feasible binary vectors $\mathcal{L} \cap \lbrace 0,1 \rbrace^{\mathcal{I}}$ is $\mathcal{M} := \operatorname{conv}(\mathcal{L} \cap \lbrace 0,1 \rbrace^{\mathcal{I}})$ and is called *marginal polytope*. As it follows from Corollary 2.60 each vertex of the marginal polytope corresponds to a labeling, and the other way around, each labeling corresponds to a vertex of $\mathcal{M}$.
- The *local polytope relaxation* (as any other relaxation) constitutes a lower bound, i.e. $E(\mathbf{y}; \boldsymbol{\theta}) \geq \min_{\boldsymbol{\mu} \in \mathcal{L}} \langle \boldsymbol{\theta}, \boldsymbol{\mu} \rangle$ for any (in particular the optimal) labeling $\mathbf{y}$. Cases when this bound is tight, i.e. $\min_{\mathbf{y} \in \mathcal{Y}_\mathcal{V}} E(\mathbf{y}; \boldsymbol{\theta}) = \min_{\boldsymbol{\mu} \in \mathcal{L}} \langle \boldsymbol{\theta}, \boldsymbol{\mu} \rangle$, are called *LP-tight*.
- The marginal polytope is a subset of the local polytope, i.e. $\mathcal{M} \subseteq \mathcal{L}$. According to Corollary 2.64, moreover, $\operatorname{vrtx}(\mathcal{M}) \subseteq \operatorname{vrtx}(\mathcal{L})$ holds, that is, all vertices of the marginal polytope are also vertices of the local polytope.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 3.7</span></p>

Consider Fortet's linearization of the labeling problem. Show explicitly that the respective LP relaxation is less tight than the local polytope relaxation. To this end show that the respective feasible set contains local polytope as its subset.

</div>

### 3.4 Bibliography and Further Reading

A detailed description of all facets of the reformulation-linearization technique by Sherali and Adams can be found in their excellently written book.

## Chapter 4: Convex Functions and Their Basic Properties

This section introduces the notion of a convex optimization problem. This type of problems plays a central role in optimization as a whole, since there are good reasons to treat these problems as efficiently solvable. Most of the methods we consider below are based on approximating a difficult non-convex problem by a much easier convex one and then solving the latter. One such approximation technique known as *Lagrange duality* and its application to integer linear programs are considered in the next chapter.

### 4.1 Convex Optimization Problems

#### 4.1.1 Extended Value Functions

So far, when speaking about the optimization problem

$$\min_{x \in X \subseteq \mathbb{R}^n} f(x) \,,$$

we assumed that the value $f(x)$ is finite for all $x \in X$. But this may not always hold. Consider for example, that $f(x)$ is defined implicitly, as the result of another optimization, e.g. $f(x) = \max_{z \in Z} g(x, z)$ with $g \colon X \times Z \to \mathbb{R}$. The maximization problem may turn out to be unbounded for some $x$. Loosely speaking, this would mean that $f(x) = \infty$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.1</span><span class="math-callout__name">(Implicitly defined extended value function)</span></p>

Consider the following extended value function, defined implicitly:

$$f(x) = \max_{z \in \mathbb{R}} (x_1 z + x_2) \,.$$

It is easy to see that $f(x) = \begin{cases} x_2, & x_1 = 0 \\ \infty, & x_1 \neq 0 \end{cases}$. Therefore, $f$ is extended value, although $x_1 z + x_2$ is defined for all $z$, $x$ and $z$ is an unconstrained variable.

</div>

Let us now take a closer look at the minimization problem $\min_{x \in X} f(x)$ when $f$ is extended value:

- Let now $X'$ be the maximal subset of $X$ such that the value $f(x)$ is finite for all $x \in X'$. Then the optimal value of the problem is finite as well and can be attained in some $x \in X'$ only.
- If $f(x)$ is unbounded from above for all $x \in X$, this implies that its minimal value is unbounded as well. In other words, $\min_{x \in X} f(x) = \infty$. This is the same as in the situation, where the problem is infeasible, since there is no element $x \in X$ such that $f(x)$ is finite.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.2</span><span class="math-callout__name">(Extended value function)</span></p>

A mapping of the form $f \colon X \to \mathbb{R} \cup \lbrace \infty, -\infty \rbrace$ is called an *extended value function*. The set $\lbrace x \in X \mid -\infty < f(x) < \infty \rbrace$ is called *domain* of $f$ and is denoted by $\operatorname{dom} f$.

</div>

Let $f$ be an extended value function. The constrained minimization problem $\min_{x \in X} f(x)$ is defined similarly as before with the following modifications:

- The feasible set is $\lbrace x \in X \mid f(x) < \infty \rbrace$.
- The problem is called unbounded if $\inf_{x \in X} f(x) = -\infty$. In particular, it is sufficient that there exists an $x \in X$ such that $f(x) = -\infty$ for the problem to be unbounded.

The maximization of extended value functions is treated similarly by substituting min with max, inf with sup, and $-\infty$ with $\infty$ and the other way around.

Most importantly, with the extended-valued functions one can "integrate" constraints into the objective function. Let $X \subset \mathbb{R}^n$ and let $\iota_X \colon \mathbb{R}^n \to \mathbb{R} \cup \lbrace \infty \rbrace$ be the extended value function

$$\iota_X(x) = \begin{cases} 0, & x \in X \\ \infty, & x \notin X \end{cases} \,.$$

Unless $f$ is equal to $-\infty$ for some $x \in X$, the optimization problem $\min_{x \in X} f(x)$ can be equivalently written as $\min_{x \in \mathbb{R}^n} (f(x) + \iota_X(x))$. Similarly, $\max_{x \in X} f(x) = \max_{x \in \mathbb{R}^n} (f(x) - \iota_X(x))$.

In the following, when speaking about a function, we will assume an *extended value* function, if the opposite is not stated.

#### 4.1.2 Convex Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.3</span><span class="math-callout__name">(Epigraph)</span></p>

An *epigraph* of a function $f$ is defined as the set $\operatorname{epi} f := \lbrace (x, t) \colon x \in \operatorname{dom} f \cup \lbrace x \colon f(x) = -\infty \rbrace,\; t \geq f(x) \rbrace$.

</div>

Loosely speaking, epigraph is set of points "above" the plot of a function.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.4</span></p>

The epigraph of a linear function $f \colon \mathbb{R}^n \to \mathbb{R}$ is a half-space in $\mathbb{R}^{n+1}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.5</span></p>

For any two functions $f$ and $g$ it holds that $\operatorname{epi}(\max\lbrace f, g \rbrace) = \operatorname{epi}(f) \cap \operatorname{epi}(g)$.

*Proof.* The proof follows directly from the definition of the epigraph of a function: $t \geq f(x)$ and $t \geq g(x)$ together imply $t \geq \max\lbrace f(x), g(x) \rbrace$, and the other way around. $\square$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.6</span><span class="math-callout__name">(Convex function)</span></p>

A function $f \colon \mathbb{R}^n \to \mathbb{R} \cup \lbrace \infty \rbrace$ is called *convex* if $\operatorname{epi} f$ is a convex set.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.7</span><span class="math-callout__name">(Concave function)</span></p>

A function $f \colon \mathbb{R}^n \to \mathbb{R} \cup \lbrace -\infty \rbrace$ is called *concave* if $(-f)$ is convex.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 4.8</span></p>

Prove that a concave and convex function $f$ is *linear*, that is, it is representable as $\langle c, x \rangle + b$ for some $c \in \mathbb{R}^n$ and $b \in \mathbb{R}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 4.9</span></p>

Prove that for any convex function $f$ the set $X = \lbrace x \colon f(x) \leq 0 \rbrace$ is convex.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.10</span></p>

A function $f \colon \mathbb{R}^n \to \mathbb{R} \cup \lbrace \infty \rbrace$ is convex if and only if the inequality

$$f(px + (1-p)z) \leq p f(x) + (1-p) f(z)$$

holds for any $x, z \in \mathbb{R}^n$ and $p \in (0, 1)$. Here we assume that $p \cdot \infty = \infty$ for any positive $p$, $\infty + x = \infty$, $-\infty + x = -\infty$ for any $x \in \mathbb{R}$ and $-(-\infty) = \infty$.

*Proof.* In case either $f(x)$ or $f(z)$ is infinite, the inequality trivially holds. Otherwise, $(x, f(x)) \in \operatorname{epi} f$ as well as $(z, f(z)) \in \operatorname{epi} f$. Since $\operatorname{epi} f$ is a convex set, if and only if it contains also $(px + (1-p)z, pf(x) + (1-p)f(z))$. By the definition of $\operatorname{epi} f$ this is equivalent to $f(px + (1-p)z) \leq pf(x) + (1-p)f(z)$. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.11</span></p>

The natural addition and multiplication rules for infinite values defined in Proposition 4.10 are considered as defined further in this monograph. Additionally $-\infty \cdot \infty = -\infty$ and the result of $\infty - \infty$ is undefined.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 4.12</span></p>

Let $f \colon \mathbb{R}^n \to \mathbb{R} \cup \lbrace \infty, -\infty \rbrace$ be an extended value function such that $\operatorname{dom} f \neq \emptyset$, and there is $x \in \mathbb{R}^n$ such that $f(x) = -\infty$. Show that $f$ is not convex.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 4.13</span></p>

Show that a function $f \colon \mathbb{R}^n \to \mathbb{R} \cup \lbrace \infty \rbrace$ is convex if and only if the inequality

$$f\!\left(\sum_{i=1}^{N} p_i x^i\right) \leq \sum_{i=1}^{N} p_i f(x^i)$$

holds for any natural $N$, any $N$-tuple $x^i \in \mathbb{R}^n$, $i = 1, \ldots, N$, and any $p \in \Delta^N$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.14</span></p>

Let $f$ and $g$ be convex functions and $\alpha$ and $\beta$ be non-negative numbers. Then $\alpha f + \beta g$ is convex as well.

*Proof.* Let $p \in [0,1]$. Then convexity of $f$ and $g$ implies the following sequence of inequalities, which proves the statement of the proposition:

$$(\alpha f + \beta g)(px + (1-p)z) = \alpha f(px + (1-p)z) + \beta g(px + (1-p)z)$$

$$\leq \alpha(pf(x) + (1-p)f(z)) + \beta(pg(x) + \beta(1-p)g(z))$$

$$= p(\alpha f + \beta g)(x) + (1-p)(\alpha f + \beta g)(z) \,. \quad\square$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.15</span><span class="math-callout__name">(Non-decreasing function)</span></p>

A function $h \colon \mathbb{R}^k \to \mathbb{R}$ is called *non-decreasing*, if $h(x_1, \ldots, x_k) \geq h(z_1, \ldots, z_k)$ as soon as $x_i \geq z_i$ for all $i = 1, \ldots, k$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.16</span></p>

Let $f(x) = h(f_1(x), \ldots, f_k(x))$ with $f_i \colon \mathbb{R}^n \to \mathbb{R}$ be convex functions, and $h \colon \mathbb{R}^k \to \mathbb{R}$ be convex and non-decreasing. Then $f$ is convex.

*Proof.* The following inequality holds for any $p \in [0, 1]$ and arbitrary $x$ and $z$ due to convexity of $f_i$ and the non-decreasing property of $h$:

$$h(f_1(px + (1-p)z), \ldots, f_k(px + (1-p)z))$$

$$\leq h(pf_1(x) + (1-p)f_1(z), \ldots, pf_k(x) + (1-p)f_k(z)) \,.$$

Further, due to convexity of $h$, the right-hand-side does not exceed

$$ph(f_1(x), \ldots, f_k(x)) + (1-p)h(f_1(z), \ldots, f_k(z)) \,,$$

that finalizes the proof. $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.17</span></p>

Let $f$ be a convex function and $X \subset \mathbb{R}^n$ be a convex set. Then the function $f + \iota_X \colon \mathbb{R}^n \to \mathbb{R}$ is convex, where $\iota_X$ is defined as in (4.3).

*Proof.* Consider the convex function $\hat{\iota}_X(x) := \begin{cases} -\infty, & x \in X \\ \infty, & x \notin X \end{cases}$. It is straightforward to see that $f + \iota_X = \max\lbrace f, \hat{\iota}_X \rbrace$. Due to Proposition 4.5, it holds that $\operatorname{epi}(f + \iota_X) = \operatorname{epi}(\max\lbrace f, \hat{\iota}_X \rbrace) = \operatorname{epi}(f) \cap \operatorname{epi}(\hat{\iota}_X)$ is convex as it is the intersection of convex sets (see Lemma 2.27). $\square$

Note that a similar claim holds also for a *concave* function $f$ and a *convex* set $X$. In this case the function $f - \iota_X$ is concave.

</div>

#### 4.1.3 Local and Global Minima

Let the set $B_\epsilon(x) = \lbrace x' \in \mathbb{R}^n \mid \lVert x - x' \rVert \leq \epsilon \rbrace$ denote the ball of radius $\epsilon \geq 0$ around $x \in \mathbb{R}^n$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.18</span><span class="math-callout__name">(Local and global minima)</span></p>

Let $f \colon X \to \mathbb{R} \cup \lbrace \infty, -\infty \rbrace$ be an extended value function. A point $x$ such that $f(x) < \infty$ is called a *local minimum* of $f$ if there exists $\epsilon > 0$ such that $x \in \arg\min_{x' \in B_\epsilon(x)} (f(x') + \iota_X(x'))$. The point $x$ is called a *global minimum* of $f$, if $x \in \arg\min_{x' \in X} f(x')$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.19</span></p>

Any local minimum of a convex function is also its global minimum.

*Proof.* Let $x$ be a local, but not a global minimum on $X \subseteq \mathbb{R}^n$. Therefore, there exists $z \in X$ such that $f(z) < f(x)$. Since $f$ is convex it holds that $f(px + (1-p)z) \leq pf(x) + (1-p)f(z) < f(x)$ for any $p \in (0, 1)$. For any $\epsilon > 0$ there is $p > 0$ such that $(px + (1-p)z) \in B_\epsilon$, therefore, $f(px + (1-p)z) < f(x)$ implies that $x$ is not a local minimum. $\square$

</div>

Due to Proposition 4.19, it makes sense to speak about *minima* of a convex function, without subdividing them into local and global ones.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.20</span></p>

The set of minima of a convex function is convex.

*Proof.* Let $x$ and $z$ be two minima and, therefore, $f(x) = f(z)$. Then for all $p \in (0,1)$ the inequality $f(px + (1-p)z) \leq pf(x) + (1-p)f(z) = pf(x) + (1-p)f(x) = f(x)$ implies $px + (1-p)z$ is a minimum as well. $\square$

</div>

Note that the set of maxima of a concave function is *convex* as well.

The following important proposition essentially states that a maximum of convex functions is a convex function itself. This observation is crucial for properties of Lagrange relaxations, which will be considered later in this chapter.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.21</span></p>

Let $Z$ be any set and let $f \colon \mathbb{R}^n \times Z \to \mathbb{R}$ be convex w.r.t. $x \in \mathbb{R}^n$ for each fixed $z \in Z$. Then the function $g(x) = \sup_{z \in Z} f(x, z)$ is convex.

*Proof.* For each fixed $z$ the epigraph of $f(\cdot, z)$ is a convex set. The epigraph of the function $g$ is the intersection of the epigraphs of the functions $f(\cdot, z) \colon \mathbb{R}^n \to \mathbb{R}$ for all $z \in Z$. Therefore, according to Lemma 2.27 $\operatorname{epi}(g)$ is a convex set as an intersection of convex sets. So by Definition 4.6 $g$ is convex. $\square$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 4.22</span></p>

The function $g(x) = \inf_{z \in Z} f(x, z)$ is concave if $f \colon \mathbb{R}^n \times Z \to \mathbb{R}$ is concave w.r.t. $x$ for each fixed $z \in Z$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.23</span><span class="math-callout__name">(Convex optimization problem)</span></p>

Let $f$ be a convex function and $X$ be a convex set. The problems of the form $\min_{x \in X} f(x)$ and $\max_{x \in X} (-f(x))$ are called *convex optimization problems*. In other words, convex optimization is a minimization of a convex or maximization of a concave function on a convex set.

</div>

According to Proposition 4.17, for any convex $f$ and $X$ the problem $\min_{x \in \mathbb{R}^n} (f(x) + \iota_X(x))$ is convex and equivalent to $\min_{x \in X} f(x)$. Therefore, we conclude that the set of optimal solutions of this problem is convex.

Linear programs, as well as the optimization problems from Examples 2.1 and 2.3 are convex.

## Chapter 5: Lagrange Duality

In Chapter 2 we defined a general notion of relaxation and considered an important example thereof, the linear programming relaxations. The latter are constructed by substituting the integrality constraints with box constraints.

Here we will consider another powerful method for constructing relaxations, where some constraints are substituted with a specially constructed function, which is then added to the original objective.

Although these two types of relaxations seem to have nothing in common at the first glance, they are in fact often equivalent. At the end of the chapter we will consider this point and formulate sufficient conditions for this equivalence.

### 5.1 Dual Problem

Let us consider an optimization problem of the form

$$\min_{\mathbf{x} \in P} f(\mathbf{x})$$

$$\text{s.t. } A\mathbf{x} = \mathbf{b} \,,$$

where $P \subseteq \mathbb{R}^n$. We will assume that omitting the equality constraints $A\mathbf{x} = \mathbf{b}$ would make the optimization problem easy (or at least *easier*) to solve. Therefore, we write these constraints separately instead of including them into the set $P$. Below we consider a powerful technique, which allows us to partially get rid of these constraints to simplify optimization.

Consider the following function of a vector $\boldsymbol{\lambda} \in \mathbb{R}^n$

$$\min_{\mathbf{x} \in P} \bigl[f(\mathbf{x}) + \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle\bigr] \,,$$

which we will call the *Lagrange dual function*. The expression $L(\mathbf{x}, \boldsymbol{\lambda}) := f(\mathbf{x}) + \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle$ is referred to as the *Lagrangean*. One also says that the constraint $A\mathbf{x} = \mathbf{b}$ is *dualized* or *relaxed*. Variables $\boldsymbol{\lambda}$ are called *dual variables*. The dimensionality of $\boldsymbol{\lambda}$ equals the number of rows of the matrix $A$. In other words, each elementary constraint gets a corresponding dual variable.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.1</span></p>

Consider

$$\min_{\mathbf{x} \in [0,1]^2} \langle \mathbf{c}, \mathbf{x} \rangle$$

$$\text{s.t. } x_1 = x_2 \,.$$

Its Lagrange dual function reads

$$\min_{\mathbf{x} \in [0,1]^2} \langle \mathbf{c}, \mathbf{x} \rangle + \lambda(x_1 - x_2)$$

with $\lambda$ being a real number.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.2</span></p>

Consider

$$\min_{\mathbf{x} \in P \cap \lbrace 0,1 \rbrace^3} 3x_1 + 5x_2 + 7x_3$$

$$\text{s.t. } x_1 + 2x_2 = 0 \quad \mid \lambda_1$$

$$x_1 + x_2 + x_3 = 1 \quad \mid \lambda_2$$

Lagrange dual is equal to

$$\min_{\mathbf{x} \in P \cap \lbrace 0,1 \rbrace^3} 3x_1 + 5x_2 + 7x_3 + \lambda_1(x_1 + 2x_2) + \lambda_2(x_1 + x_2 + x_3 - 1)$$

</div>

For any $\boldsymbol{\lambda}$ the Lagrange dual is a relaxation of the primal problem, because:

- The feasible set of the dual is a superset of the one for the primal.
- The objective values of both problems are equal for any feasible point of the primal, since in this case $A\mathbf{x} = \mathbf{b}$ holds, and, therefore, $\langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle = 0$.

These kinds of relaxations are called *Lagrange* relaxations.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5.3</span></p>

For every $\boldsymbol{\lambda} \in \mathbb{R}$ the optimal value of the Lagrange dual does not exceed the optimal value of the primal, and they coincide if the constraint $A\mathbf{x}' = \mathbf{b}$ holds in an optimum $\mathbf{x}'$ of the dual.

*Proof.* The first claim follows directly from Proposition 2.9. The second one follows from Proposition 2.9 and the fact that the objective functions of the primal and the dual coincide if $A\mathbf{x}' = \mathbf{b}$. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.4</span></p>

Note that the problem

$$g(\boldsymbol{\lambda}) = \min_{\mathbf{x} \in P} f(\mathbf{x}) + \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle$$

is a Lagrange dual of

$$\min_{\mathbf{x} \in P} f(\mathbf{x})$$

$$\text{s.t. } A\mathbf{x} \leq \mathbf{b}$$

for any $\boldsymbol{\lambda} \geq 0$.

</div>

In general, Lagrange relaxation is a way to construct a relaxation by dualizing any equality or inequality constraints, not necessarily linear ones. In these cases the constraint is defined as $h(x) = b$ (or $h(x) \leq b$), and is substituted by the term $\lambda(h(x) - b)$ (with $\lambda \geq 0$), which is then added to the objective. In this monograph we make use of the dualization of linear constraints only, therefore, we restrict our attention to this very important special case.

Since it is important to have the tightest possible lower bound, one considers the *Lagrange dual problem*

$$\max_{\boldsymbol{\lambda}} \min_{\mathbf{x} \in P} f(\mathbf{x}) + \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle \,,$$

which amounts to maximizing the dual w.r.t. $\boldsymbol{\lambda}$. One speaks about *full Lagrange dual* if all constraints are dualized and not only a subset of them.

The most important property of the Lagrange dual problem is its concavity w.r.t. $\boldsymbol{\lambda}$, which does not depend on the fact whether the primal problem is convex or not:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5.5</span></p>

The dual function is concave w.r.t. $\boldsymbol{\lambda}$.

*Proof.* The proof follows from Corollary 4.22 and the fact that the objective of the dual is linear w.r.t. $\boldsymbol{\lambda}$, and, therefore, concave. $\square$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.6</span><span class="math-callout__name">(Piecewise linear Lagrange dual)</span></p>

Consider the Lagrange dual as in the primal problem:

$$\min_{\mathbf{x} \in P} \langle \mathbf{c}, \mathbf{x} \rangle + \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle$$

It is a concave piecewise linear function, when $P$ is

- a finite set, since the expression satisfies Definition 4.30 in this case.
- a polytope. In this case

$$\min_{\mathbf{x} \in P} \langle \mathbf{c}, \mathbf{x} \rangle + \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle \overset{\text{Cor. 2.48}}{=} \min_{\mathbf{x} \in \operatorname{vrtx}(P)} \left\langle \mathbf{c} + A^\top \boldsymbol{\lambda}, \mathbf{x} \right\rangle - \langle \boldsymbol{\lambda}, \mathbf{b} \rangle \,,$$

and the right-hand-side satisfies Definition 4.30.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.7</span></p>

In a more general case

$$\min_{\mathbf{x} \in P} f(\mathbf{x})$$

$$\text{s.t. } A\mathbf{x} = \mathbf{b}$$

$$B\mathbf{x} \leq \mathbf{d}$$

the Lagrange dual problem takes the form

$$\max_{\substack{\boldsymbol{\lambda} \\ \boldsymbol{\mu} \geq 0}} \min_{\mathbf{x} \in P} f(\mathbf{x}) + \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle + \langle \boldsymbol{\mu}, B\mathbf{x} - \mathbf{d} \rangle \,.$$

All key properties like concavity and piecewise linearity of the Lagrange dual objective hold also in this more general case.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.8</span></p>

For a maximization problem

$$\max_{\mathbf{x} \in P} f(\mathbf{x})$$

$$\text{s.t. } A\mathbf{x} = \mathbf{b}$$

$$B\mathbf{x} \leq \mathbf{d}$$

the Lagrange dual reads

$$\min_{\substack{\boldsymbol{\lambda} \\ \boldsymbol{\mu} \geq 0}} \max_{\mathbf{x} \in P} f(\mathbf{x}) + \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle - \langle \boldsymbol{\mu}, B\mathbf{x} - \mathbf{d} \rangle \,.$$

The respective Lagrange dual objective is convex. It is also piecewise linear in the same cases as its minimization counterpart.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5.9</span></p>

Let $P$ be a polytope and the set $P \cap \lbrace \mathbf{x} \colon A\mathbf{x} = \mathbf{b} \rbrace$ be non-empty. Then there exist optimal primal and dual solutions. Moreover, the *strong duality* holds, i.e.

$$\min_{\substack{\mathbf{x} \in P \\ A\mathbf{x} = \mathbf{b}}} \langle \mathbf{c}, \mathbf{x} \rangle = \max_{\boldsymbol{\lambda}} \min_{\mathbf{x} \in P} \langle \mathbf{c}, \mathbf{x} \rangle + \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle \,.$$

*Proof.* Let us first show that there exist optimal primal and dual solutions. From $P$ being a polytope it follows that the primal feasible set $P \cap \lbrace \mathbf{x} \colon A\mathbf{x} = \mathbf{b} \rbrace$ is a polytope as well (see Proposition 2.19). Therefore, the optimal primal value $p^*$ is finite and attained in one of its vertices, according to Corollary 2.48.

Let $d^*$ denote the optimal dual value. The following sequence of inequalities, which uses Proposition 5.3,

$$\infty > p^* \geq d^* = \max_{\boldsymbol{\lambda}} \min_{\mathbf{x} \in P} \langle \mathbf{c}, \mathbf{x} \rangle + \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle \overset{\boldsymbol{\lambda}=0}{\geq} \min_{\mathbf{x} \in P} \langle \mathbf{c}, \mathbf{x} \rangle > -\infty$$

implies that the dual optimal value $d^*$ is also finite.

Consider the dual objective

$$g(\boldsymbol{\lambda}) := \min_{\mathbf{x} \in P} \langle \mathbf{c}, \mathbf{x} \rangle + \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle \overset{\text{Cor. 2.48}}{=} \min_{\mathbf{x} \in \operatorname{vrtx}(P)} \langle \mathbf{c}, \mathbf{x} \rangle + \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle \,,$$

and the set $\hat{P}(\boldsymbol{\lambda}) := \arg\min_{\mathbf{x} \in \operatorname{vrtx}(P)} \langle \mathbf{c}, \mathbf{x} \rangle + \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle$. Since $P$ is non-empty, $\hat{P}(\boldsymbol{\lambda})$ is so as well. The maximal dual value $d^*$ is attained in all vectors $\boldsymbol{\lambda}$ satisfying

$$d^* = \langle \mathbf{c}, \mathbf{x}' \rangle + \langle \boldsymbol{\lambda}, A\mathbf{x}' - \mathbf{b} \rangle \,,$$

where $\mathbf{x}'$ is some vector from $\hat{P}(\boldsymbol{\lambda})$. The set of $\boldsymbol{\lambda}$ satisfying this is non-empty as soon as $A\mathbf{x}' - \mathbf{b} = \mathbf{0}$ implies $d^* = \langle \mathbf{c}, \mathbf{x}' \rangle$. This implication is given by the following sequence of equalities:

$$d^* = \max_{\boldsymbol{\lambda}} \min_{\mathbf{x} \in \operatorname{vrtx}(P)} \langle \mathbf{c}, \mathbf{x} \rangle + \langle \boldsymbol{\lambda}, A\mathbf{x}' - \mathbf{b} \rangle$$

$$\overset{A\mathbf{x}'=\mathbf{b}}{=} \max_{\boldsymbol{\lambda}} \langle \mathbf{c}, \mathbf{x}' \rangle = \langle \mathbf{c}, \mathbf{x}' \rangle \,.$$

Therefore, the dual solution exists. Let $\boldsymbol{\lambda}^*$ be such a solution.

According to Lemma 4.32,

$$\partial g(\boldsymbol{\lambda}^*) = \operatorname{conv}\lbrace A\mathbf{x}^* - \mathbf{b} \colon \mathbf{x}^* \in \hat{P}(\boldsymbol{\lambda}^*) \rbrace \overset{\text{Lem. 2.37}}{=} \lbrace A\mathbf{x}^* - \mathbf{b} \colon \mathbf{x}^* \in \operatorname{conv}(\hat{P}(\boldsymbol{\lambda}^*)) \rbrace \,.$$

Since $\boldsymbol{\lambda}^*$ is a dual optimal solution, Theorem 4.35 implies that $\partial g(\boldsymbol{\lambda}^*) \ni \mathbf{0}$. In other words, there exists $\mathbf{x}^* \in \operatorname{conv}(\hat{P}(\boldsymbol{\lambda}^*))$ such that

$$A\mathbf{x}^* - \mathbf{b} = \mathbf{0} \,.$$

Observe that $\mathbf{x}^* \in P$, since $\hat{P}(\boldsymbol{\lambda}^*) \subseteq \operatorname{vrtx}(P)$ and, therefore, $\operatorname{conv}(\hat{P}(\boldsymbol{\lambda}^*)) \subseteq \operatorname{conv}(\operatorname{vrtx}(P)) = P$. This yields

$$d^* = \langle \mathbf{c}, \mathbf{x}^* \rangle + \langle \boldsymbol{\lambda}^*, A\mathbf{x}^* - \mathbf{b} \rangle \overset{A\mathbf{x}^*-\mathbf{b}=\mathbf{0}}{=} \langle \mathbf{c}, \mathbf{x}^* \rangle \overset{\mathbf{x}^* \in P,\; A\mathbf{x}^*=\mathbf{b}}{\geq} \min_{\substack{\mathbf{x} \in P \\ A\mathbf{x}=\mathbf{b}}} \langle \mathbf{c}, \mathbf{x} \rangle = p^* \,.$$

Together with $p^* \geq d^*$ this implies $p^* = d^*$. $\square$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.10</span></p>

Consider Example 5.1, the primal

$$\min_{\mathbf{x} \in [0,1]^2} \langle \mathbf{c}, \mathbf{x} \rangle$$

$$\text{s.t. } x_1 = x_2 \,,$$

and the dual problem:

$$\max_\lambda \min_{\mathbf{x} \in [0,1]^2} \bigl[\langle \mathbf{c}, \mathbf{x} \rangle + \lambda(x_1 - x_2)\bigr]$$

$$= \max_\lambda \min_{\mathbf{x} \in [0,1]^2} \bigl[(c_1 + \lambda)x_1 + (c_2 - \lambda)x_2\bigr] \,.$$

The primal problem can be equivalently rewritten as

$$\min_{x_1 \in [0,1]} (c_1 + c_2)x_1 = \begin{cases} 0, & (c_1 + c_2) \geq 0 \\ (c_1 + c_2), & (c_1 + c_2) < 0 \,. \end{cases}$$

Consider the dual. It can be equivalently rewritten as

$$\max_\lambda \bigl(\min\lbrace 0, (c_1 + \lambda) \rbrace + \min\lbrace 0, (c_2 - \lambda) \rbrace\bigr) \,.$$

The necessary and sufficient optimality condition is the existence of the zero subgradient of the function. According to Lemma 4.32, this condition holds when either

$$c_1 + \lambda \geq 0 \text{ and } c_2 - \lambda \geq 0 \,,$$

or

$$c_1 + \lambda < 0 \text{ and } c_2 - \lambda < 0 \,.$$

One pair of these inequalities is always attained when $c_1 + \lambda = c_2 - \lambda$, therefore, $\lambda = \frac{c_2 - c_1}{2}$. Substituting $\lambda$ with this value turns the first condition into $c_1 + c_2 \geq 0$. Similarly the second condition becomes $c_1 + c_2 < 0$.

Summing up,

$$\max_\lambda \bigl(\min\lbrace 0, (c_1 + \lambda) \rbrace + \min\lbrace 0, (c_2 - \lambda) \rbrace\bigr) = \begin{cases} 0, & (c_1 + c_2) \geq 0 \\ (c_1 + c_2), & (c_1 + c_2) < 0 \,, \end{cases}$$

which coincides with the primal optimal value.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.11</span></p>

Let $P$ be a polytope and the set $P \cap \lbrace \mathbf{x} \colon A\mathbf{x} = \mathbf{b},\; B\mathbf{x} \leq \mathbf{d} \rbrace$ be non-empty. Then there exist optimal primal and dual solutions. Moreover, the *strong duality* holds, i.e.

$$\min_{\substack{\mathbf{x} \in P \\ A\mathbf{x}=\mathbf{b},\; B\mathbf{x} \leq \mathbf{d}}} \langle \mathbf{c}, \mathbf{x} \rangle = \max_{\substack{\boldsymbol{\lambda} \\ \boldsymbol{\mu} \geq 0}} \min_{\mathbf{x} \in P} \langle \mathbf{c}, \mathbf{x} \rangle + \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle + \langle \boldsymbol{\mu}, B\mathbf{x} - \mathbf{d} \rangle \,.$$

</div>

We will use this theorem without a proof.

### 5.2 Lagrange Relaxation of Integer Linear Programs

Since we are interested in solving integer linear programs, let us consider the Lagrange relaxation of this kind of problem. In what follows we will deal with integer linear programs of the form

$$\min_{\mathbf{x} \in P \cap \lbrace 0,1 \rbrace^n} \langle \mathbf{c}, \mathbf{x} \rangle$$

$$\text{s.t. } A\mathbf{x} = \mathbf{b} \,,$$

$$B\mathbf{x} \leq \mathbf{d} \,,$$

where $P$ is a polytope. The assumption that $P$ is bounded (i.e. it is a polytope and not a general polyhedron) can be made w.l.o.g., since the problem can be turned into an equivalent one with a bounded $P$ by adding the constraints $\mathbf{x} \in [0, 1]^n$.

As before, we assume that without constraint $A\mathbf{x} = \mathbf{b}$ and $B\mathbf{x} \leq \mathbf{d}$ the problem becomes easy or at least easier to solve. Therefore, these constraints are separated out from the polytope $P$. Dualizing these constraints we obtain the Lagrange dual problem:

$$\max_{\substack{\boldsymbol{\lambda} \\ \boldsymbol{\mu} \geq 0}} \min_{\mathbf{x} \in P \cap \lbrace 0,1 \rbrace^n} \langle \mathbf{c}, \mathbf{x} \rangle + \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle + \langle \boldsymbol{\mu}, B\mathbf{x} - \mathbf{d} \rangle \,.$$

Note that objective of the dual problem is a concave piecewise linear, therefore, non-differentiable, function.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.12</span></p>

Consider the ILP and assume $P = \mathbb{R}^n$. Then the dual objective can be rewritten as

$$g(\boldsymbol{\lambda}, \boldsymbol{\mu}) = \min_{\mathbf{x} \in \lbrace 0,1 \rbrace^n} \langle \mathbf{c}, \mathbf{x} \rangle + \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle + \langle \boldsymbol{\mu}, B\mathbf{x} - \mathbf{d} \rangle$$

$$= -\langle \boldsymbol{\lambda}, \mathbf{b} \rangle - \langle \boldsymbol{\mu}, \mathbf{d} \rangle + \min_{\mathbf{x} \in \lbrace 0,1 \rbrace^n} \left\langle \mathbf{c} + A^\top \boldsymbol{\lambda} + B^\top \boldsymbol{\mu}, \mathbf{x} \right\rangle$$

$$= -\langle \boldsymbol{\lambda}, \mathbf{b} \rangle - \langle \boldsymbol{\mu}, \mathbf{d} \rangle + \left\langle \mathbf{c} + A^\top \boldsymbol{\lambda} + B^\top \boldsymbol{\mu}, \mathbf{x}^* \right\rangle \,,$$

with

$$x_i^* = \begin{cases} 0, & (\mathbf{c} + A^\top \boldsymbol{\lambda} + B^\top \boldsymbol{\mu})_i > 0 \\ 1, & (\mathbf{c} + A^\top \boldsymbol{\lambda} + B^\top \boldsymbol{\mu})_i < 0 \\ \in \lbrace 0, 1 \rbrace, & \text{otherwise} \,. \end{cases}$$

In this important special case the computation of the dual function can be done independently for each coordinate $i \in [n]$ and is, therefore, particularly easy.

</div>

**Use Case: Binary Knapsack Problem.** Consider the binary knapsack problem from Example 2.53:

$$\max_{\mathbf{x} \in \lbrace 0,1 \rbrace^n} \langle \mathbf{c}, \mathbf{x} \rangle$$

$$\text{s.t. } \langle \mathbf{w}, \mathbf{x} \rangle \leq b \,.$$

By dualizing its inequality constraint we obtain the Lagrange dual problem

$$\min_{\mu \geq 0} \max_{\mathbf{x} \in \lbrace 0,1 \rbrace^n} \langle \mathbf{c}, \mathbf{x} \rangle - \mu(\langle \mathbf{w}, \mathbf{x} \rangle - b)$$

$$= \min_{\mu \geq 0} \mu b + \max_{\mathbf{x} \in \lbrace 0,1 \rbrace^n} \langle \mathbf{c} - \mu \mathbf{w}, \mathbf{x} \rangle$$

$$= \min_{\mu \geq 0} \mu b + \langle \mathbf{c} - \mu \mathbf{w}, \mathbf{x}^*(\mu) \rangle \,,$$

where

$$x_i^*(\mu)_i = \begin{cases} 1, & c_i - \mu w_i > 0 \\ 0, & c_i - \mu w_i \leq 0 \,. \end{cases}$$

Note the "weight-informed cost" $\mathbf{c} - \mu \mathbf{w}$ plays the decisive role in computation of the dual. We will see that similar phenomena appear in, essentially, any Lagrange dual with linear objective and constraints and treat it in a general case in Section 5.2.2.

Assume now w.l.o.g. that $c_i > 0$ and $b \geq w_i > 0$, moreover, $\frac{c_1}{w_1} \geq \frac{c_2}{w_2} \geq \cdots \geq \frac{c_n}{w_n}$. According to the dual, for $\frac{c_k}{w_k} \geq \mu \geq \frac{c_{k+1}}{w_{k+1}}$ the dual objective takes the form

$$g(\mu) = \mu b + \sum_{i=1}^{k} (c_i - \mu w_i) = \mu(b - \sum_{i=1}^{k} w_i) + \sum_{i=1}^{k} c_i \,.$$

According to Corollary 5.17 $\mu \geq 0$ is optimal, if $\mu(b - \sum_{i=1}^{k-1} w_i - w_k x_k^*) = 0$. In turn, $\mu \neq 0$ implies $b - \sum_{i=1}^{k-1} w_i - w_k x_k^* = 0$. This allows to compute $k$ as the largest one for which $\sum_{i=1}^{k-1} w_i \leq b$ and $x_k^* = (b - \sum_{i=1}^{k-1} w_i) / w_k$.

#### 5.2.1 Primal of the Relaxed Problem for ILPs

The Lagrange dual can be treated as a problem of selecting the best relaxation from a given set of relaxations. This class is parametrized with the dual vector $(\boldsymbol{\lambda})$. The solution of the dual problem is a value of this parameter. To obtain the corresponding relaxed solution one has to solve the related relaxed problem $\min_{\mathbf{x} \in X} f(\mathbf{x}) + \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle$ with fixed $\boldsymbol{\lambda}$. In this problem, however, both, the objective and the initial feasible sets are changed compared to the primal problem. This makes analysis of the relaxed solution much more complicated.

This differs from the case of the LP relaxation for integer linear programs, where only the feasible set is increased, but the objective itself remains unchanged. Interestingly, for integer linear programs, one can define a relaxation, which is equivalent to the Lagrange relaxation, by changing only the feasible set.

To do so, let us consider the Lagrange dual and exchange the feasible set $\lbrace 0, 1 \rbrace^n$ with its convex hull $[0, 1]^n$, corresponding to $\operatorname{conv}(P \cap \lbrace 0,1 \rbrace^n)$:

$$\min_{\mathbf{x} \in \operatorname{conv}(P \cap \lbrace 0,1 \rbrace^n)} \langle \mathbf{c}, \mathbf{x} \rangle$$

$$\text{s.t. } A\mathbf{x} = \mathbf{b} \,,\quad B\mathbf{x} \leq \mathbf{d}$$

Assuming that the ILP is feasible implies that this problem is so, too. Since the feasible set of this problem is a polytope, Theorem 5.11 implies that its Lagrange dual is tight:

$$\max_{\substack{\boldsymbol{\lambda} \\ \boldsymbol{\mu} \geq 0}} \min_{\mathbf{x} \in \operatorname{conv}(P \cap \lbrace 0,1 \rbrace^n)} \langle \mathbf{c}, \mathbf{x} \rangle + \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle + \langle \boldsymbol{\mu}, B\mathbf{x} - \mathbf{d} \rangle$$

Comparing it to the ILP Lagrange dual we conclude

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5.13</span></p>

$$\max_{\substack{\boldsymbol{\lambda} \\ \boldsymbol{\mu} \geq 0}} \min_{\mathbf{x} \in P \cap \lbrace 0,1 \rbrace^n} \langle \mathbf{c}, \mathbf{x} \rangle + \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle + \langle \boldsymbol{\mu}, B\mathbf{x} - \mathbf{d} \rangle = \min_{\substack{\mathbf{x} \in \operatorname{conv}(P \cap \lbrace 0,1 \rbrace^n) \\ A\mathbf{x}=\mathbf{b},\; B\mathbf{x} \leq \mathbf{d}}} \langle \mathbf{c}, \mathbf{x} \rangle$$

In other words, the Lagrange dual is tight for the *primal relaxed* problem. Therefore, we will call the latter the *primal relaxed* problem for the Lagrange relaxation.

</div>

Note that the primal relaxed problem is also a convex relaxation of the ILP problem, since $\operatorname{conv}(P \cap \lbrace 0,1 \rbrace^n) \supseteq P \cap \lbrace 0,1 \rbrace^n$ by definition of the convex hull. Contrary to the Lagrange relaxation, the primal relaxed problem modifies only the feasible set of the non-relaxed problem, and keeps its objective unchanged. Note also that the primal relaxed problem is a linear program, since its feasible set is a polytope. It differs from the LP relaxation of the ILP

$$\min_{\substack{\mathbf{x} \in P \cap [0,1]^n \\ A\mathbf{x}=\mathbf{b},\; B\mathbf{x} \leq \mathbf{d}}} \langle \mathbf{c}, \mathbf{x} \rangle$$

by the feasible set only. The following statement compares these two linear programs and establishes conditions under which they coincide.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 5.14</span></p>

Let $\operatorname{vrtx}(P \cap [0,1]^n) \subseteq \lbrace 0,1 \rbrace^n$, that is, all vertices of the polytope $P \cap [0,1]^n$ are defined by binary vectors. Then the primal relaxed problem is equal to the LP relaxation. Otherwise, the relaxation is tighter.

*Proof.* According to Proposition 2.63, it holds that $\operatorname{vrtx}(P \cap [0,1]^n) \supseteq P \cap \lbrace 0,1 \rbrace^n$. According to Proposition 2.65, it holds $\operatorname{vrtx}(P \cap [0,1]^n) \cap \lbrace 0,1 \rbrace^n = P \cap \lbrace 0,1 \rbrace^n$. Therefore, $\operatorname{vrtx}(P \cap [0,1]^n) \subseteq \lbrace 0,1 \rbrace^n$ implies $\operatorname{vrtx}(P \cap [0,1]^n) = P \cap \lbrace 0,1 \rbrace^n$. From Corollary 2.39 it follows that $\operatorname{conv}(P \cap \lbrace 0,1 \rbrace^n) = P \cap [0,1]^n$, and, therefore, the considered Lagrange and LP relaxations are equivalent.

In general, however, it holds that $\operatorname{conv}(P \cap \lbrace 0,1 \rbrace^n) \subseteq P \cap [0,1]^n$ (see Proposition 2.62), which implies that the Lagrange relaxation is tighter than the LP one, as their objective functions coincide. $\square$

Since the primal relaxed and dual problems provide the same bound, one also says that *the Lagrange relaxation is in general tighter than the LP one*.

</div>

#### 5.2.2 Reparametrization

For (integer) linear programs the Lagrange relaxation has a simple and intuitive interpretation, expressed in terms of *reparametrization* or *equivalent transformation* of the cost vector. The optimization of the dual problem can be viewed as finding such an equivalent transformation, such that the dualized constraints are fulfilled for the (relaxed) primal problem.

We will first assume that the dualized constraints have the form $A\mathbf{x} = \mathbf{0}$. Later we consider the differences in the general case $A\mathbf{x} = \mathbf{b}$.

Let us consider the Lagrange dual problem. For $\mathbf{b} = \mathbf{0}$ its objective is

$$\langle \mathbf{c}, \mathbf{x} \rangle + \langle \boldsymbol{\lambda}, A\mathbf{x} \rangle = \langle \mathbf{c}, \mathbf{x} \rangle + \left\langle A^\top \boldsymbol{\lambda}, \mathbf{x} \right\rangle = \left\langle \mathbf{c} + A^\top \boldsymbol{\lambda}, \mathbf{x} \right\rangle \,.$$

Note that for any feasible $\mathbf{x}$, it holds that $A\mathbf{x} = \mathbf{0}$ and therefore

$$\langle \mathbf{c}, \mathbf{x} \rangle = \left\langle \mathbf{c} + A^\top \boldsymbol{\lambda}, \mathbf{x} \right\rangle \,.$$

It implies that the problem

$$\min_{\mathbf{x} \in P \cap \lbrace 0,1 \rbrace^n} \left\langle \mathbf{c} + A^\top \boldsymbol{\lambda}, \mathbf{x} \right\rangle$$

$$\text{s.t. } A\mathbf{x} = \mathbf{0}$$

is *equivalent* to the original ILP (with $\mathbf{b} = \mathbf{0}$), i.e. for any $\boldsymbol{\lambda}$ and any feasible $\mathbf{x}$ the objectives of both problems have equal values. Clearly, the solutions of both problems are therefore also equal. The transformation $\mathbf{c} \to \mathbf{c} + A^\top \boldsymbol{\lambda}$ is therefore called *an equivalent transformation* or a *reparametrization* of the problem. The dual problem

$$\max_{\boldsymbol{\lambda}} \left[D(\boldsymbol{\lambda}) := \min_{\mathbf{x} \in P \cap \lbrace 0,1 \rbrace^n} \left\langle \mathbf{c} + A^\top \boldsymbol{\lambda}, \mathbf{x} \right\rangle\right]$$

therefore consists in finding an *optimal reparametrization* of the problem.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.15</span></p>

In Example 5.10 the reparametrized cost vector has coordinates $c_1 + \lambda$ and $c_2 - \lambda$.

</div>

Note that the equality also implies that the *relaxed* primal problem, defined by the right-hand-side of Proposition 5.13, is equivalent to its reparametrization for any $\boldsymbol{\lambda}$:

$$\min_{\substack{\operatorname{conv}(\mathbf{x} \in P \cap \lbrace 0,1 \rbrace^n) \\ A\mathbf{x}=\mathbf{0}}} \langle \mathbf{c}, \mathbf{x} \rangle = \min_{\substack{\operatorname{conv}(\mathbf{x} \in P \cap \lbrace 0,1 \rbrace^n) \\ A\mathbf{x}=\mathbf{0}}} \left\langle \mathbf{c} + A^\top \boldsymbol{\lambda}, \mathbf{x} \right\rangle \,.$$

**General case $A\mathbf{x} = \mathbf{b}$.** In case $\mathbf{b} \neq \mathbf{0}$

$$\langle \mathbf{c}, \mathbf{x} \rangle = \left\langle \mathbf{c} + A^\top \boldsymbol{\lambda}, \mathbf{x} \right\rangle - \langle \mathbf{b}, \boldsymbol{\lambda} \rangle \,.$$

Therefore, the total reparametrization shifts the total cost of each feasible solution by $\langle \mathbf{b}, \boldsymbol{\lambda} \rangle$ contrary to the case $\mathbf{b} = \mathbf{0}$, where the total cost remains the same. The shift, however, preserves the order of solution costs and does not influence their optimality.

The primal problem equivalent to the ILP turns into

$$-\langle \mathbf{b}, \boldsymbol{\lambda} \rangle + \min_{\substack{\mathbf{x} \in P \cap \lbrace 0,1 \rbrace^n \\ \text{s.t. } A\mathbf{x} = \mathbf{b}}} \left\langle \mathbf{c} + A^\top \boldsymbol{\lambda}, \mathbf{x} \right\rangle$$

and the Lagrange dual reads

$$\max_{\boldsymbol{\lambda}} \left[D(\boldsymbol{\lambda}) := -\langle \mathbf{b}, \boldsymbol{\lambda} \rangle + \min_{\mathbf{x} \in P \cap \lbrace 0,1 \rbrace^n} \left\langle \mathbf{c} + A^\top \boldsymbol{\lambda}, \mathbf{x} \right\rangle\right] \,.$$

Most properties of the dual objective $D(\boldsymbol{\lambda})$ remain the same for any vector $\mathbf{b}$. In particular, the key complexity of its evaluation remains $\min_{\mathbf{x} \in P \cap \lbrace 0,1 \rbrace^n} \langle \mathbf{c} + A^\top \boldsymbol{\lambda}, \mathbf{x} \rangle$, which is independent of $\mathbf{b}$.

Apart from theoretical, the reparametrized costs have a powerful applied usage: When used with nearly optimal dual vector $\boldsymbol{\lambda}$ they may significantly improve results of multiple primal heuristics, especially those based in the greedy algorithms, see Section 8.2. In Chapter 8 we will see how primal heuristics can profit from being applied to reparametrized costs instead of original ones.

**Slack variables and Lagrange dual.** Instead of dualizing inequality one may turn it into equality with a *slack* variable and dualize the equality afterwards. Consider the problem with inequality $A\mathbf{x} \leq \mathbf{b}$ and the respective non-negative slack vector $\mathbf{s}$:

$$\min_{\substack{\mathbf{x} \in P \\ \mathbf{s} \geq 0}} f(\mathbf{x})$$

$$\text{s.t. } A\mathbf{x} + \mathbf{s} = \mathbf{b} \,.$$

The dual problem turns into

$$\max_{\boldsymbol{\lambda}} \min_{\substack{\mathbf{x} \in P \\ \mathbf{s} \geq 0}} f(\mathbf{x}) + \langle \boldsymbol{\lambda}, A\mathbf{x} + \mathbf{s} - \mathbf{b} \rangle$$

If $\lambda_i < 0$ for any $i$, the minimization subproblem is unbounded, as selecting arbitrary large value of $s_i$ makes the values of the dual objective arbitrary small. From the other side, for $\boldsymbol{\lambda} \geq 0$ the assignment $\mathbf{s} = 0$ is optimal. Therefore, the dual can be equivalently rewritten as

$$\max_{\boldsymbol{\lambda} \geq 0} \min_{\mathbf{x} \in P} f(\mathbf{x}) + \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle \,,$$

thus it coincides with the dual obtained by dualizing the initial inequality.

In general, usage of the slack variables for linear $f$ allows to profit from the reparametrization, as in this case the equality constraint is dualized. However, this profit comes with a hitch: Usage of slack variables changes the problem structure and the primal heuristics addressing $\min_{\mathbf{x} \in P} \langle \mathbf{c}, \mathbf{x} \rangle$ may not be applicable to $\min_{\substack{\mathbf{x} \in P \\ \mathbf{s} \geq 0}} \langle \mathbf{c}, \mathbf{x} \rangle$.

One may even obtain tighter relaxation with slack variables if the latter can be bounded from above. For example, if all elements of the matrix $A$ are non-negative and the set $P$ is a subset of $\lbrace 0, 1 \rbrace^n$ like in the max-weight independent set problem, see Example 2.54. In this case it holds $\mathbf{b} \geq \mathbf{s} \geq 0$ and, therefore, the minimization subproblem in the dual is not unbounded anymore for $\lambda_i < 0$.

#### 5.2.3 Primal-Dual Optimality Conditions

Consider now a general *not necessarily convex* problem

$$\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x})$$

$$\text{s.t. } h_i(\mathbf{x}) = 0 \,,\; i \in [m_1]$$

$$q_i(\mathbf{x}) \leq 0 \,,\; i \in [m_2] \,.$$

Its Lagrangean reads:

$$L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\mu}) = f(\mathbf{x}) + \sum_{i=1}^{m_1} \lambda_i h_i(\mathbf{x}) + \sum_{i=1}^{m_2} \mu_i q_i(\mathbf{x}) = f(\mathbf{x}) + \langle \boldsymbol{\lambda}, \mathbf{h}(\mathbf{x}) \rangle + \langle \boldsymbol{\mu}, \mathbf{q}(\mathbf{x}) \rangle \,,$$

where $\mathbf{h}$ and $\mathbf{q}$ are the respective vector-functions.

The corresponding Lagrange dual problem is

$$\max_{\substack{\boldsymbol{\lambda} \\ \boldsymbol{\mu} \geq 0}} \left[g(\boldsymbol{\lambda}, \boldsymbol{\mu}) := \min_{\mathbf{x} \in \mathbb{R}^n} L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\mu})\right] \,.$$

**Certificate of optimality and stopping criteria.** Any dual feasible solution $(\boldsymbol{\lambda}, \boldsymbol{\mu})$, $\boldsymbol{\mu} \geq 0$, establishes a lower bound for the optimal value $p^*$ of the primal problem: $p^* \geq g(\boldsymbol{\lambda}, \boldsymbol{\mu})$. One says that $(\boldsymbol{\lambda}, \boldsymbol{\mu})$ provides a *proof* or *certificate* that $p^* \geq g(\boldsymbol{\lambda}, \boldsymbol{\mu})$. Should $p^* = g(\boldsymbol{\lambda}, \boldsymbol{\mu})$, one speaks about *optimality certificate*. Dual suboptimal points allow to estimate how good is the primal solution $\mathbf{x}$ without knowing $p^*$. For example, $\mathbf{x}$ is $\epsilon$-close to the optimum, if $f(\mathbf{x}) - g(\boldsymbol{\lambda}, \boldsymbol{\mu}) \leq \epsilon$ and, therefore:

$$f(\mathbf{x}) - p^* \leq f(\mathbf{x}) - g(\boldsymbol{\lambda}, \boldsymbol{\mu}) \leq \epsilon \,.$$

This condition also proves that the dual solution $(\boldsymbol{\lambda}, \boldsymbol{\mu})$ is $\epsilon$-close to the (dual) optimum, as

$$d^* - g(\boldsymbol{\lambda}, \boldsymbol{\mu}) \leq f(\mathbf{x}) - g(\boldsymbol{\lambda}, \boldsymbol{\mu}) \leq \epsilon \,.$$

This and similar conditions can be used as a stopping criteria for optimization.

**Complementary slackness.** Let $\mathbf{x}^*$ and $(\boldsymbol{\lambda}^*, \boldsymbol{\mu}^*)$ be primal and dual optima of the problem and primal and dual optimal values are equal. Then

$$f(\mathbf{x}^*) = g(\boldsymbol{\lambda}^*, \boldsymbol{\mu}^*)$$

$$= \min_{x \in \mathbb{R}^n} \bigl(f(\mathbf{x}) + \langle \boldsymbol{\lambda}^*, \mathbf{h}(\mathbf{x}) \rangle + \langle \boldsymbol{\mu}^*, \mathbf{q}(\mathbf{x}) \rangle\bigr)$$

$$\leq f(\mathbf{x}^*) + \langle \boldsymbol{\lambda}^*, \mathbf{h}(\mathbf{x}^*) \rangle + \langle \boldsymbol{\mu}^*, \mathbf{q}(\mathbf{x}^*) \rangle$$

$$\overset{h_i(\mathbf{x}^*)=0,\; q_i(\mathbf{x}^*) \leq 0,\; \mu^*_i \geq 0}{\leq} f(\mathbf{x}^*)$$

The last inequality follows from feasibility of primal and dual solutions. We conclude that both inequalities hold with equality and, therefore:

- $\mathbf{x}^*$ is a minimizer of $L(\mathbf{x}, \boldsymbol{\lambda}^*, \boldsymbol{\mu}^*)$ over $\mathbf{x}$;
- $\langle \boldsymbol{\mu}^*, \mathbf{q}(\mathbf{x}^*) \rangle = 0$, and since $\mu^*_i \geq 0$ and $q_i(\mathbf{x}^*) \leq 0$ it holds

$$\mu^*_i \cdot q_i(\mathbf{x}^*) = 0 \,,\; i \in [m_2]$$

It implies that either the respective inequality holds as equality or the corresponding dual variable is zero:

$$\mu^*_i > 0 \Rightarrow q_i(\mathbf{x}^*) = 0 \,,$$

$$q_i(\mathbf{x}^*) < 0 \Rightarrow \mu^*_i = 0 \,.$$

**Karush-Kuhn-Tucker (KKT) optimality conditions.** Now we assume that $f$, $h_i$, $i \in [m_1]$ and $q_i$, $i \in [m_2]$ and, therefore, $L$ are differentiable.

**Non-convex case.** Let as above $\mathbf{x}^*$ and $(\boldsymbol{\lambda}^*, \boldsymbol{\mu}^*)$ be primal and dual optima and primal and dual optimal values are equal. Since $\mathbf{x}^*$ minimizes $L(x, \boldsymbol{\lambda}^*, \boldsymbol{\mu}^*)$ over $\mathbf{x}$, the gradient of $L$ must vanish:

$$\nabla f(\mathbf{x}^*) + \sum_{i=1}^{m_1} \lambda_i \nabla h_i(\mathbf{x}^*) + \sum_{i=1}^{m_1} \mu_i \nabla q_i(\mathbf{x}^*) = 0 \,.$$

Thus, we have

$$h_i(\mathbf{x}^*) = 0 \,,\; i \in [m_1]$$

$$q_i(\mathbf{x}^*) \leq 0 \,,\; i \in [m_2]$$

$$\mu^*_i \geq 0 \,,\; i \in [m_2]$$

$$\mu^*_i q_i(\mathbf{x}^*) = 0 \,,\; i \in [m_2]$$

$$\nabla f(\mathbf{x}^*) + \sum_{i=1}^{m_1} \lambda_i \nabla h_i(\mathbf{x}^*) + \sum_{i=1}^{m_1} \mu_i \nabla q_i(\mathbf{x}^*) = 0 \,,$$

which are called *Karush-Kuhn-Tucker (KKT)* conditions.

To summarize, for *any* optimization problem with differentiable objective and constraint functions for which strong duality holds, any pair of primal and dual optimal points must satisfy the KKT conditions. In other words, KKT conditions are *necessary* in this case.

**Convex case.** When the problem is convex, the KKT conditions are not only necessary, but also sufficient for primal-dual optimality *and* strong duality.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 5.16</span></p>

Let the problem be convex, that is $h_i$ is affine and $f$, $q_i$ are convex, see Exercise 4.9. Let also $\hat{\mathbf{x}}$, $\hat{\boldsymbol{\lambda}}$, $\hat{\boldsymbol{\mu}}$ be any points that satisfy the KKT conditions in place of $\mathbf{x}^*$, $\boldsymbol{\lambda}^*$, $\boldsymbol{\mu}^*$.

Then $\hat{\mathbf{x}}$, $(\hat{\boldsymbol{\lambda}}, \hat{\boldsymbol{\mu}})$ are primal and dual optima, with zero duality gap.

*Proof.* Denote

$$L(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\mu}) = \iota_{\operatorname{conv}(P \cap \lbrace 0,1 \rbrace^n)}(\mathbf{x}) + \langle \mathbf{c}, \mathbf{x} \rangle + \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle + \langle \boldsymbol{\mu}, B\mathbf{x} - \mathbf{d} \rangle \,,$$

where $\iota$ is defined as in (4.3). In other words, we consider the convex case with $f(\mathbf{x}) = \iota_{\operatorname{conv}(P \cap \lbrace 0,1 \rbrace^n)}(\mathbf{x}) + \langle \mathbf{c}, \mathbf{x} \rangle$.

**Sufficiency:** Condition (5.93) implies that $\mathbf{x}^*$ minimizes $L(\mathbf{x}, \boldsymbol{\lambda}^*, \boldsymbol{\mu}^*)$ over $\mathbf{x}$ and applying the same transformation as in the complementary slackness proof, we obtain that $\mathbf{x}^*$ and $(\boldsymbol{\lambda}^*, \boldsymbol{\mu}^*)$ are primal and dual optimal solutions.

**Necessity:** Theorem 5.11 implies that there exists an optimal primal solution $\mathbf{x}^*$ of the relaxed primal problem such that the strong duality holds. Then it follows that $\mathbf{x}^*$ is a minimizer of $L(\mathbf{x}, \boldsymbol{\lambda}^*, \boldsymbol{\mu}^*)$ and, therefore, of (5.93), over $\mathbf{x}$. Optimality of $\mathbf{x}^*$ implies its feasibility for the primal problem. Additionally, the complementary slackness holds. $\square$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 5.17</span><span class="math-callout__name">(Dual optimality condition)</span></p>

Vectors $\boldsymbol{\lambda}^*$ and $\boldsymbol{\mu}^* \geq 0$ are an optimum of the dual problem if and only if there exists

$$\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \operatorname{conv}(P \cap \lbrace 0,1 \rbrace^n)} \left\langle \mathbf{c} + A^\top \boldsymbol{\lambda}^* + B^\top \boldsymbol{\mu}^*, \mathbf{x} \right\rangle$$

such that $A\mathbf{x}^* = \mathbf{b}$, $B\mathbf{x}^* \leq \mathbf{d}$ and $\langle \boldsymbol{\mu}^*, B\mathbf{x}^* - \mathbf{d} \rangle = 0$. Such an $\mathbf{x}^*$ is a solution of the relaxed primal problem.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.18</span></p>

If $\mathbf{x}^*$ in Corollary 5.17 is binary, i.e. $\mathbf{x}^* \in \lbrace 0, 1 \rbrace^n$, then $\mathbf{x}^*$ is a solution of the non-relaxed ILP problem. This directly follows from the fact that the primal relaxed problem is a relaxation of the ILP.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.19</span><span class="math-callout__name">(Binary knapsack problem)</span></p>

Let us get back to the Lagrange dual of the binary knapsack problem and exchange the feasible set $\lbrace 0, 1 \rbrace^n$ with its convex hull $[0, 1]^n$, corresponding to $\operatorname{conv}(P \cap \lbrace 0,1 \rbrace^n)$:

$$\min_{\mu \geq 0} \max_{\mathbf{x} \in [0,1]^n} \langle \mathbf{c}, \mathbf{x} \rangle - \mu(\langle \mathbf{w}, \mathbf{x} \rangle - b) = \min_{\mu \geq 0} \mu b + \langle \mathbf{c} - \mu \mathbf{w}, \mathbf{x}^*(\mu) \rangle \,,$$

where

$$x^*(\mu)_i = \begin{cases} 1, & c_i - \mu w_i > 0 \\ 0, & c_i - \mu w_i < 0 \\ [0, 1], & c_i - \mu w_i = 0 \,. \end{cases}$$

Note that now we separately treat the case $c_i - \mu w_i = 0$, as the relaxed primal variable $x^*(\mu)_i$ may take arbitrary value between zero and one in this case. Considering the same assumptions about $c_i$ and $w_i$ as before, we now rewrite it for $\mu = \frac{c_k}{w_k}$ as

$$g(\mu) = \mu b + \sum_{i=1}^{k-1} (c_i - \mu w_i) + (c_k - \mu w_k) x^*_k$$

$$= \mu(b - \sum_{i=1}^{k-1} w_i - w_k x^*_k) + \sum_{i=1}^{k-1} c_i + c_k x^*_k \,.$$

According to Corollary 5.17 $\mu \geq 0$ is optimal, if $\mu(b - \sum_{i=1}^{k-1} w_i - w_k x_k^*) = 0$. In turn, $\mu \neq 0$ implies $b - \sum_{i=1}^{k-1} w_i - w_k x_k^* = 0$. This allows to compute $k$ as the largest one for which $\sum_{i=1}^{k-1} w_i \leq b$ and $x_k^* = (b - \sum_{i=1}^{k-1} w_i) / w_k$.

</div>

### 5.3 Explicit Dual Constraints, Linear Program Duality

Consider the Lagrange dual. According to Definition 4.2

$$\operatorname{dom} g = \lbrace (\boldsymbol{\lambda}, \boldsymbol{\mu}) \colon g(\boldsymbol{\lambda}, \boldsymbol{\mu}) > -\infty \rbrace \,.$$

Since the dual $g$ is to be maximized afterwards, points $(\boldsymbol{\lambda}, \boldsymbol{\mu})$ outside $\operatorname{dom} g$ can be explicitly excluded from optimization, as we have already done it when considering dualization of inequalities via usage of slack variables. Below we will treat another important case of a full Lagrange dual for linear programs. The word *full* points out that *all* constraints of the problem are dualized.

Consider

$$\min_{\substack{\mathbf{x} \in \mathbb{R}^n \\ A\mathbf{x}=\mathbf{b} \\ B\mathbf{x} \leq \mathbf{d}}} \langle \mathbf{c}, \mathbf{x} \rangle \geq \max_{\substack{\boldsymbol{\lambda} \\ \boldsymbol{\mu} \geq 0}} \min_{\mathbf{x} \in \mathbb{R}^n} \langle \mathbf{c}, \mathbf{x} \rangle + \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle + \langle \boldsymbol{\mu}, B\mathbf{x} - \mathbf{d} \rangle$$

$$= \max_{\substack{\boldsymbol{\lambda} \\ \boldsymbol{\mu} \geq 0}} -\langle \boldsymbol{\lambda}, \mathbf{b} \rangle - \langle \boldsymbol{\mu}, \mathbf{d} \rangle + \min_{\mathbf{x} \in \mathbb{R}^n} \left\langle \mathbf{c} + A^\top \boldsymbol{\lambda} + B^\top \boldsymbol{\mu}, \mathbf{x} \right\rangle$$

The minimization subproblem above is bounded only if $\mathbf{c} + A^\top \boldsymbol{\lambda} + B^\top \boldsymbol{\mu} = 0$, hence we can add this condition as an explicit constraint:

$$\max_{\boldsymbol{\lambda}, \boldsymbol{\mu}} -\langle \boldsymbol{\lambda}, \mathbf{b} \rangle - \langle \boldsymbol{\mu}, \mathbf{d} \rangle$$

$$\text{s.t. } \mathbf{c} + A^\top \boldsymbol{\lambda} + B^\top \boldsymbol{\mu} = 0$$

$$\boldsymbol{\mu} \geq 0$$

The term $(-\langle \boldsymbol{\lambda}, \mathbf{b} \rangle - \langle \boldsymbol{\mu}, \mathbf{d} \rangle)$ is referred, therefore, to as *dual LP objective* and the constraints as *constraints of the dual LP*. Note that the full dual to a linear program is a linear program itself.

**LP in standard form.** As a special case consider now the LP in the *standard* form:

$$\min_{\mathbf{x} \in \mathbb{R}^n} \langle \mathbf{c}, \mathbf{x} \rangle$$

$$\text{s.t. } A\mathbf{x} = \mathbf{b}$$

$$\mathbf{x} \geq 0$$

According to the above, the respective *dual LP* takes the form

$$\max_{\boldsymbol{\lambda}, \boldsymbol{\mu}} -\langle \boldsymbol{\lambda}, \mathbf{b} \rangle$$

$$\text{s.t. } \mathbf{c} + A^\top \boldsymbol{\lambda} - \boldsymbol{\mu} = 0$$

$$\boldsymbol{\mu} \geq 0$$

We can additionally eliminate $\boldsymbol{\mu}$ and obtain

$$\max_{\boldsymbol{\lambda}} -\langle \boldsymbol{\lambda}, \mathbf{b} \rangle$$

$$\text{s.t. } \mathbf{c} + A^\top \boldsymbol{\lambda} \geq 0 \,,$$

which is an LP in the *inequality* form.

**LP in the inequality form.** Similarly, consider the linear program in the inequality form:

$$\min_{\mathbf{x} \in \mathbb{R}^n} \langle \mathbf{c}, \mathbf{x} \rangle$$

$$\text{s.t. } B\mathbf{x} \leq \mathbf{d}$$

Comparing to the dual constraints above we conclude that the dual takes the form

$$\max_{\boldsymbol{\mu}} -\langle \boldsymbol{\mu}, \mathbf{d} \rangle$$

$$\text{s.t. } B^\top \boldsymbol{\mu} = -\mathbf{c}$$

$$\boldsymbol{\mu} \geq 0 \,,$$

which, in turn, is an LP in the standard form.

Note the symmetry between forms of the respective primal and dual linear programs: Dual of an LP in the standard form is an LP in the inequality form and vice versa.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 5.20</span></p>

Compute the dual LP for the standard form and the inequality form. Show that they are equal to the primal LP in the respective other form, which means that *the dual of the dual LP is the primal LP*.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.21</span></p>

Consider

$$\min_\mathbf{x} 3x_1 + 5x_2 + 7x_3$$

$$\text{s.t. } x_1 + 2x_2 = 0 \quad \mid \lambda_1$$

$$x_1 + x_2 + x_3 = 1 \quad \mid \lambda_2$$

$$x_1 \geq 0 \quad \mid \mu_1$$

$$x_2 \geq 0 \quad \mid \mu_2$$

$$x_3 \geq 0 \quad \mid \mu_3$$

The dual problem is equal to:

$$\max_{\substack{\boldsymbol{\lambda} \\ \boldsymbol{\mu} \geq 0}} \min_\mathbf{x} \bigl[3x_1 + 5x_2 + 7x_3 + \lambda_1(x_1 + 2x_2) + \lambda_2(x_1 + x_2 + x_3 - 1) - \mu_1 x_1 - \mu_2 x_2 - \mu_3 x_3\bigr]$$

$$= \max_{\boldsymbol{\mu} \geq 0} \bigl[-\lambda_2 + \min_\mathbf{x} x_1(3 + \lambda_1 + \lambda_2 - \mu_1) + x_2(5 + 2\lambda_1 + \lambda_2 - \mu_2) + x_3(7 + \lambda_2 - \mu_3)\bigr]$$

$$= \max_{\boldsymbol{\mu} \geq 0} -\lambda_2$$

$$\text{s.t. } 3 + \lambda_1 + \lambda_2 - \mu_1 = 0$$

$$5 + 2\lambda_1 + \lambda_2 - \mu_2 = 0$$

$$7 + \lambda_2 - \mu_3 = 0$$

$$= \max_\boldsymbol{\lambda} -\lambda_2$$

$$\text{s.t. } 3 + \lambda_1 + \lambda_2 \geq 0$$

$$5 + 2\lambda_1 + \lambda_2 \geq 0$$

$$7 + \lambda_2 \geq 0$$

Consider now the dual LP in the inequality form and compute the full dual to it:

$$\max_\boldsymbol{\lambda} -\lambda_2$$

$$\text{s.t. } 3 + \lambda_1 + \lambda_2 \geq 0 \quad \mid x_1$$

$$5 + 2\lambda_1 + \lambda_2 \geq 0 \quad \mid x_2$$

$$7 + \lambda_2 \geq 0 \quad \mid x_3$$

$$\leq \min_{\mathbf{x} \geq 0} \max_\boldsymbol{\lambda} \bigl[-\lambda_2 + x_1(3 + \lambda_1 + \lambda_2) + x_2(5 + 2\lambda_1 + \lambda_2) + x_3(7 + \lambda_2)\bigr]$$

$$= \min_{\mathbf{x} \geq 0} \bigl[3x_1 + 5x_2 + 7x_3 + \max_\boldsymbol{\lambda} \lambda_1(x_1 + 2x_2) + \lambda_2(x_1 + x_2 + x_3 - 1)\bigr]$$

$$= \min_{\mathbf{x} \geq 0} 3x_1 + 5x_2 + 7x_3$$

$$\text{s.t. } x_1 + 2x_2 = 0$$

$$x_1 + x_2 + x_3 = 1 \,,$$

which is the initial problem.

</div>

### 5.4 Bibliography and Further Reading

For an extended introduction to the duality theory we recommend the excellent textbook on convex optimization [12]. It also details the notion of the extended value functions and their relation to convex analysis and Lagrange duality. More general and deep analysis of the mathematical phenomenon of duality is presented in [13].

### 5.5 Case Study: Lagrange Dual of the Labeling Problem

Below we derive the Lagrangean dual of the local polytope linearization of the labeling problem (3.44). This dual is obtained by relaxing the coupling constraints of the local polytope. Based on the theoretical background provided in Chapter 5 we will analyze properties of the dual.

The second important topic we deal with in this chapter is the optimality conditions for the considered dual problem. We will give exact (necessary and sufficient) as well as approximate (only necessary) optimality conditions. Whereas checking the first ones can be as computationally expensive as optimizing the dual itself, the latter can be verified with a simple and efficient algorithm.

In the last subsection we return to the acyclic labeling problems, and show that the approximate optimality conditions are exact for them.

#### 5.5.1 Reparametrization and Lagrange Dual

Recall the local polytope linearization (3.44) of the labeling problem

$$\min_{\boldsymbol{\mu} \in \mathcal{L} \cap \lbrace 0,1 \rbrace^{\mathcal{I}}} \langle \boldsymbol{\theta}, \boldsymbol{\mu} \rangle$$

with the local polytope $\mathcal{L}$ defined by (3.43). Note that $\mathcal{L}$ can be rewritten as follows:

$$\mathcal{L} = \begin{cases} \boldsymbol{\mu}_u \in \Delta^{\mathcal{Y}_u}, & \forall u \in \mathcal{V} \\ \boldsymbol{\mu}_{uv} \in \Delta^{\mathcal{Y}_{uv}}, & \forall uv \in \mathcal{E} \\ \sum_{t \in \mathcal{Y}_v} \mu_{uv}(s,t) = \mu_u(s), & \forall u \in \mathcal{V},\; v \in \mathcal{N}(u),\; s \in \mathcal{Y}_u. \end{cases}$$

As in Section 3.3, the notation $\boldsymbol{\mu}_u$ stands for the vector $(\mu_u(s) \colon s \in \mathcal{Y}_u)$, which encodes the selected label in the node $u$, and $\boldsymbol{\mu}_{uv} = (\mu_{uv}(s,t) \colon (s,t) \in \mathcal{Y}_{uv})$ encodes the selected label pair on the edge $uv$.

Let us construct the Lagrange dual to (5.114) by relaxing the coupling constraints (the last line in (5.115)). This is done similarly to the general scheme provided in Section 5.1.

Consider the linear term of the objective, which corresponds to dualizing the coupling constraints. In other words, we will specify the term $\langle \boldsymbol{\lambda}, A\mathbf{x} \rangle$ corresponding to the constraints $A\mathbf{x} = 0$, when the role of the latter is played by the coupling constraints. Since for each node $u \in \mathcal{V}$ there is one constraint for each of its labels $s \in \mathcal{Y}_u$ and each neighboring node $v \in \mathcal{N}(u)$, the role of the dual vector $\boldsymbol{\lambda}$ is played by the vector $\boldsymbol{\phi} \in \mathbb{R}^{\mathcal{J}}$ with coordinates $\phi_{u,v}(s)$, where $\mathcal{J} := \lbrace (u, v, s) \mid u \in \mathcal{V},\; v \in \mathcal{N}(u),\; s \in \mathcal{Y}_u \rbrace$. The comma between $u$ and $v$ in the lower index underlines that $\phi_{u,v}(s)$ and $\phi_{v,u}(s)$ are two different values, contrary to $\mu_{uv}(s,t)$ and $\mu_{vu}(t,s)$, where $u$ and $v$ are not separated by the comma.

Given the above notation, the considered linear term corresponding to the coupling constraints reads:

$$\sum_{u \in \mathcal{V}} \sum_{v \in \mathcal{N}(u)} \sum_{s \in \mathcal{Y}_u} \phi_{u,v}(s) \left( \sum_{t \in \mathcal{Y}_v} \mu_{uv}(s,t) - \mu_u(s) \right).$$

By adding it to the objective

$$\langle \boldsymbol{\theta}, \boldsymbol{\mu} \rangle = \sum_{u \in \mathcal{V}} \sum_{s \in \mathcal{Y}_u} \theta_u(s) \mu_u(s) + \sum_{uv \in \mathcal{E}} \sum_{(s,t) \in \mathcal{Y}_{uv}} \theta_{uv}(s,t) \mu_{uv}(s,t)$$

and regrouping terms one obtains

$$\sum_{u \in \mathcal{V}} \sum_{s \in \mathcal{Y}_u} \mu_u(s) \left( \theta_u(s) - \sum_{v \in \mathcal{N}(u)} \phi_{u,v}(s) \right) + \sum_{uv \in \mathcal{E}} \sum_{(s,t) \in \mathcal{Y}_{uv}} \mu_{uv}(s,t) \left( \theta_{uv}(s,t) + \phi_{u,v}(s) + \phi_{v,u}(t) \right) = \langle \boldsymbol{\theta}^\phi, \boldsymbol{\mu} \rangle \,,$$

where we introduced the *reparametrized costs* $\boldsymbol{\theta}^\phi$ defined as

$$\theta_u^\phi(s) := \theta_u(s) - \sum_{v \in \mathcal{N}(u)} \phi_{u,v}(s), \quad v \in \mathcal{V},\; s \in \mathcal{Y}_u \,,$$

$$\theta_{uv}^\phi(s,t) := \theta_{uv}(s,t) + \phi_{u,v}(s) + \phi_{v,u}(t), \quad uv \in \mathcal{E},\; (s,t) \in \mathcal{Y}_{uv} \,.$$

Note that the reparametrized Lagrangean $\langle \boldsymbol{\theta}^\phi, \boldsymbol{\mu} \rangle$ is an instance of the reparametrization introduced in Section 5.2.2 for general (integer) linear programs, see Expression (5.60). In other words, the Lagrangian dual constructed by relaxing the coupling constraints consists in finding an optimal reparametrization:

$$\max_{\boldsymbol{\phi} \in \mathbb{R}^{\mathcal{J}}} \underbrace{\min_{\substack{\boldsymbol{\mu} \in \lbrace 0,1 \rbrace^{\mathcal{I}} \\ \boldsymbol{\mu}_u \in \Delta^{\mathcal{Y}_u}, u \in \mathcal{V} \\ \boldsymbol{\mu}_{uv} \in \Delta^{\mathcal{Y}_{uv}}, uv \in \mathcal{E}}}}_{\text{dual function } \mathcal{D}(\boldsymbol{\phi})} \langle \boldsymbol{\theta}^\phi, \boldsymbol{\mu} \rangle \,.$$

Note that the constraints on $\boldsymbol{\mu}$ decouple into the coordinates indexed by $u$ and $uv$:

$$\min_{\substack{\boldsymbol{\mu} \in \lbrace 0,1 \rbrace^{\mathcal{I}} \\ \boldsymbol{\mu}_u \in \Delta^{\mathcal{Y}_u}, u \in \mathcal{V} \\ \boldsymbol{\mu}_{uv} \in \Delta^{\mathcal{Y}_{uv}}, uv \in \mathcal{E}}} \left[ \langle \boldsymbol{\theta}^\phi, \boldsymbol{\mu} \rangle = \sum_{u \in \mathcal{V}} \langle \boldsymbol{\theta}_u^\phi, \boldsymbol{\mu}_u \rangle + \sum_{uv \in \mathcal{E}} \langle \boldsymbol{\theta}_{uv}^\phi, \boldsymbol{\mu}_{uv} \rangle \right]$$

$$= \sum_{u \in \mathcal{V}} \min_{s \in \mathcal{Y}_u} \theta_u^\phi(s) + \sum_{uv \in \mathcal{E}} \min_{(s,t) \in \mathcal{Y}_{uv}} \theta_{uv}^\phi(s,t) \,.$$

Therefore, the dual problem (5.120) takes the form

$$\max_{\boldsymbol{\phi} \in \mathbb{R}^{\mathcal{J}}} \mathcal{D}(\boldsymbol{\phi}) := \max_{\boldsymbol{\phi} \in \mathbb{R}^{\mathcal{J}}} \left( \sum_{u \in \mathcal{V}} \min_{s \in \mathcal{Y}_u} \theta_u^\phi(s) + \sum_{uv \in \mathcal{E}} \min_{(s,t) \in \mathcal{Y}_{uv}} \theta_{uv}^\phi(s,t) \right).$$

Note that to compute the value of the dual objective in (5.122) for a fixed $\boldsymbol{\phi}$ one has to select *independently* in each node and each edge an optimal label and an optimal label pair, respectively. Such labels and label pairs will be called *locally optimal* in the following. Note also that these locally optimal labels and label pairs need not be consistent. In other words, if the label $s$ is selected in node $u$ and the label pair $(t, t')$ in edge $uv$, this does not generally imply that $s = t$. However, as we will show later in this chapter, some kind of relaxed consistency (known as *arc-consistency*) is enforced in the optimum of the dual function.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.22</span><span class="math-callout__name">(How large-scale is the Lagrangean dual of the labeling problem?)</span></p>

Consider now the Lagrangean dual (5.122). The number of variables is equal to $\lvert \mathcal{J} \rvert$, that grows linearly with the number of labels and the number of edges in a graph. This is in contrast to the primal problem (5.114), where the number of variables grows quadratically with the number of labels.

Consider Example 3.6, where the size of the ILP representation of the labeling problem was estimated for the depth reconstruction problem from Example 1.11. In that case the dual problem has $O(10^5)$ variables, which is an order of magnitude less than the size $\mathcal{O}(10^6)$ of the primal problem.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5.23</span></p>

For the dual problem (5.122) it holds that:

1. $\langle \boldsymbol{\theta}^\phi, \boldsymbol{\mu} \rangle = \langle \boldsymbol{\theta}, \boldsymbol{\mu} \rangle$ holds for any $\boldsymbol{\phi} \in \mathbb{R}^{\mathcal{J}}$ and $\boldsymbol{\mu} \in \mathcal{L}$ (in particular for $\boldsymbol{\mu} \in \mathcal{L} \cap \lbrace 0,1 \rbrace^{\mathcal{I}}$ and $\boldsymbol{\mu} \in \mathcal{M}$). This implies that for any labeling $\mathbf{y} \in \mathcal{Y}_\mathcal{V}$ it holds that $E(\mathbf{y}; \boldsymbol{\theta}) = E(\mathbf{y}; \boldsymbol{\theta}^\phi)$ with $E$ being the energy of $\mathbf{y}$ as defined in (1.5).
2. $\mathcal{D}(\boldsymbol{\phi})$ is a lower bound for optimal energy, i.e. $\mathcal{D}(\boldsymbol{\phi}) \leq \langle \boldsymbol{\theta}, \boldsymbol{\mu} \rangle$ for any $\boldsymbol{\mu} \in \mathcal{L} \cap \lbrace 0,1 \rbrace^{\mathcal{I}}$ and $\boldsymbol{\phi} \in \mathbb{R}^{\mathcal{J}}$.
3. $\mathcal{D}$ is concave piecewise linear, and, therefore, a non-differentiable function.
4. The primal relaxed problem corresponding to (5.122) is the local polytope relaxation, i.e. $\max_{\boldsymbol{\phi} \in \mathbb{R}^{\mathcal{J}}} \mathcal{D}(\boldsymbol{\phi}) = \min_{\boldsymbol{\mu} \in \mathcal{L}} \langle \boldsymbol{\theta}, \boldsymbol{\mu} \rangle$.
5. The optimality condition for the dual $\mathcal{D}(\boldsymbol{\phi})$ reads:

$$\boldsymbol{\phi} \in \arg \max_{\boldsymbol{\phi}' \in \mathbb{R}^{\mathcal{J}}} \mathcal{D}(\boldsymbol{\phi}')$$

if and only if the polytope

$$\mathcal{L}(\boldsymbol{\phi}) := \left\lbrace \boldsymbol{\mu} \in \mathcal{L} \colon \mu_w(s) = 0 \text{ if } \theta_w^\phi(s) > \min_{s' \in \mathcal{Y}_w} \theta_w^\phi(s'),\; w \in \mathcal{V} \cup \mathcal{E},\; s \in \mathcal{Y}_w \right\rbrace$$

is non-empty. In other words, there must be a relaxed labeling $\boldsymbol{\mu} \in \mathcal{L}$ with non-zero coordinates assigned only to locally optimal labels and label pairs. Such a relaxed labeling is also a solution to the local polytope relaxation $\min_{\boldsymbol{\mu} \in \mathcal{L}} \langle \boldsymbol{\theta}, \boldsymbol{\mu} \rangle$.

6. **Tightness of the Lagrange dual:** Let the set $\mathcal{L}(\boldsymbol{\phi})$ defined by (5.123) contain an integer labeling, in other words, there is $\mathbf{y} \in \mathcal{Y}_\mathcal{V}$ such that $\boldsymbol{\delta}(\mathbf{y}) \in \mathcal{L}(\boldsymbol{\phi})$. Then $\mathbf{y}$ is the solution of the (non-relaxed) energy minimization problem (1.5). In this case $\mathcal{D}(\boldsymbol{\phi}) = E(\mathbf{y}; \boldsymbol{\theta})$. We will call this case *LP-tight*, since the Lagrange dual is equivalent to the local polytope relaxation. This statement holds also in the opposite direction, that is, $\mathcal{D}(\boldsymbol{\phi}) = E(\mathbf{y}; \boldsymbol{\theta})$ if and only if there is $\mathbf{y} \in \mathcal{Y}_\mathcal{V}$ such that $\boldsymbol{\delta}(\mathbf{y}) \in \mathcal{L}(\boldsymbol{\phi})$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Proposition 5.23</summary>

1. Equality follows from the fact that for any $\boldsymbol{\mu} \in \mathcal{L}$ the coupling constraints hold, and, therefore, $\langle \boldsymbol{\theta}^\phi, \boldsymbol{\mu} \rangle$ is a reparametrization of $\langle \boldsymbol{\theta}, \boldsymbol{\mu} \rangle$, as introduced in Section 5.2.2.

2. This follows from Proposition 5.3, since $\mathcal{D}(\boldsymbol{\phi})$ is a Lagrangian relaxation of the labeling problem (5.114).

3. The dual is concave piecewise linear as any Lagrange dual of an integer linear program is. This also can be seen directly from (5.122): The reparametrized costs $\theta^\phi$ are linear functions of $\boldsymbol{\phi}$ according to (5.119), and the dual objective has also the representation (5.120) as a minimum over linear functions of $\boldsymbol{\phi}$.

4. It is sufficient to derive the dual from the local polytope relaxation similar to (5.121) and apply the strong duality Theorem 5.11.

5. The proof follows from Corollary 5.17. The condition (5.93) translates into

$$\boldsymbol{\mu} \in \arg \min_{\substack{\boldsymbol{\mu}_u' \in \Delta^{\mathcal{Y}_u}, u \in \mathcal{V} \\ \boldsymbol{\mu}_{uv}' \in \Delta^{\mathcal{Y}_{uv}}, uv \in \mathcal{E}}} \langle \boldsymbol{\theta}^\phi, \boldsymbol{\mu}' \rangle \,.$$

Let $\boldsymbol{\phi} \in \mathbb{R}^{\mathcal{J}}$ be a dual optimum. Then according to Corollary 5.17 there exists $\boldsymbol{\mu}$ satisfying both (5.125) and the coupling constraints. It implies $\boldsymbol{\mu} \in \mathcal{L}$. To show that $\boldsymbol{\mu} \in \mathcal{L}(\boldsymbol{\phi})$ it remains to prove that $\mu_w(s) = 0$ if $s \notin \arg \min_{s \in \mathcal{Y}_w} \theta_w^\phi(s)$.

6. From Item 5 it follows that $\boldsymbol{\delta}(\mathbf{y})$ is a minimizer of the local polytope relaxation. Since $\boldsymbol{\delta}(\mathbf{y}) \in \lbrace 0,1 \rbrace^{\mathcal{I}}$ it also minimizes the non-relaxed problem (5.114) and, therefore, $\mathbf{y}$ is an optimal labeling. The equality $\mathcal{D}(\boldsymbol{\phi}) = E(\mathbf{y}; \boldsymbol{\theta}) \equiv \langle \boldsymbol{\theta}, \boldsymbol{\delta}(\mathbf{y}) \rangle$ follows from (??), which holds for any $\boldsymbol{\mu} \in \mathcal{L}(\boldsymbol{\phi})$ and, therefore, for $\boldsymbol{\delta}(\mathbf{y})$.

To prove the statement in the opposite direction assume that $\mathcal{L}(\boldsymbol{\phi}) \neq \emptyset$, but that it does not contain any integer labeling, i.e. $\boldsymbol{\delta}(\mathbf{y}) \notin \mathcal{L}(\boldsymbol{\phi})$ for any $\mathbf{y} \in \mathcal{Y}$. Due to Item 4 a vector $\boldsymbol{\mu} \in \mathcal{L}$ is an optimum of the local polytope relaxation if and only if $\langle \boldsymbol{\theta}^\phi, \boldsymbol{\mu} \rangle = \mathcal{D}(\boldsymbol{\phi})$. In turn, according to Item 5, the equality $\langle \boldsymbol{\theta}^\phi, \boldsymbol{\mu} \rangle = \mathcal{D}(\boldsymbol{\phi})$ holds only for $\boldsymbol{\mu} \in \mathcal{L}(\boldsymbol{\phi})$. Since $\boldsymbol{\delta}(\mathbf{y}) \notin \mathcal{L}(\boldsymbol{\phi})$, the vector $\boldsymbol{\delta}(\mathbf{y})$ is not a solution of the relaxed problem. Therefore,

$$\mathcal{D}(\boldsymbol{\phi}) = \langle \boldsymbol{\theta}^\phi, \boldsymbol{\mu} \rangle < \langle \boldsymbol{\theta}^\phi, \boldsymbol{\delta}(\mathbf{y}) \rangle = E(\mathbf{y}; \boldsymbol{\theta}) \,,$$

which is a contradiction. $\square$

</details>
</div>

#### 5.5.2 Necessary Optimality Condition: Arc-Consistency and Node-Edge Agreement

As follows from Proposition 5.23(5), to determine whether a given dual vector $\boldsymbol{\phi}$ is optimal, it suffices to check whether the polytope $\mathcal{L}(\boldsymbol{\phi})$ defined in (5.123) is non-empty. Such kinds of problems, where it is necessary to verify whether a set is empty, are called *feasibility problems*. In general, feasibility problems for linear programming are as difficult as linear programs themselves.

In practice, a simpler optimality condition is often desirable, which could be checked often enough, for example on each iteration of some iterative algorithm. In this section we formulate such a condition. The price for its simplicity is that it is only necessary for dual optimality and no longer sufficient. Informally, instead of looking for an integer or relaxed labeling consisting of locally optimal labels and label pairs, which would require a *global* reasoning, one may check whether the locally optimal labels and label pairs are *locally* consistent. Below, we give a formal definition of this simpler condition and an algorithm for its verification.

In the following, we will use two functions, which turn real-valued vectors from the primal space $\mathbb{R}^{\mathcal{I}}$ into binary ones:

- For any $\mu \in \mathbb{R}$ let $\operatorname{nz}[\mu] = \llbracket \mu \neq 0 \rrbracket$ be the indicator function of $\mu$ being **non-zero**. When applied to a vector $\boldsymbol{\mu} \in \mathbb{R}^n$, it acts coordinate-wise, i.e. $\operatorname{nz}[\boldsymbol{\mu}]_i = \operatorname{nz}[\mu_i]$.

- For $\boldsymbol{\theta} \in \mathbb{R}^{\mathcal{I}}$ let $\operatorname{mi}[\boldsymbol{\theta}]$ be defined such that locally minimal labels and label pairs (w.r.t. $\theta$) obtain the value 1 and others zero, i.e. $\operatorname{mi}[\boldsymbol{\theta}]_w(x_w) := \llbracket \theta_w(x_w) = \min_{x_w \in \mathcal{Y}_w} \theta_w(x_w) \rrbracket$ for $w \in \mathcal{V} \cup \mathcal{E}$. Here $\operatorname{mi}$ stands for $\min$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.24</span></p>

$\operatorname{nz}[\boldsymbol{\delta}(\mathbf{y})] = \boldsymbol{\delta}(\mathbf{y})$;

$\operatorname{nz}[(0, 0.2, 0.8, 0)] = (0, 1, 1, 0)$;

$\operatorname{mi}[(0, -2, -1, -2, 3)] = (0, 1, 0, 1, 0)$;

$\operatorname{mi}[(0, -2), (1, 1, 2, 0), (7, -1)] = (0, 1), (0, 0, 0, 1), (0, 1)$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.25</span><span class="math-callout__name">(Arc-Consistency)</span></p>

A binary vector $\boldsymbol{\xi} \in \lbrace 0, 1 \rbrace^{\mathcal{I}}$ is called *arc-consistent* if

1. $\xi_{uv}(s, t) = 1$ implies $\xi_u(s) = \xi_v(t) = 1$ for all $uv \in \mathcal{E}$, $(s, t) \in \mathcal{Y}_{uv}$, and
2. $\xi_u(s) = 1$ implies that for any $v \in \mathcal{N}(u)$ there exists $t \in \mathcal{Y}_v$ such that $\xi_{uv}(s, t) = 1$.

The set of arc-consistent vectors from $\lbrace 0, 1 \rbrace^{\mathcal{I}}$ will be denoted as $\mathcal{AC}^{\mathcal{I}}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.26</span><span class="math-callout__name">(Strict Arc-Consistency)</span></p>

$\boldsymbol{\xi} \in \lbrace 0, 1 \rbrace^{\mathcal{I}}$ is *strictly arc-consistent* if it is arc-consistent and $\sum_{s \in \mathcal{Y}_u} \xi_u(s) = 1$ for all $u \in \mathcal{V}$ as well as $\sum_{(s,t) \in \mathcal{Y}_{uv}} \xi_{uv}(s,t) = 1$ for all $uv \in \mathcal{E}$.

</div>

In other words, strict arc-consistency means that (i) for a single label in each node and a single label pair in each edge the corresponding coordinate of $\boldsymbol{\xi}$ is equal to 1 and (ii) these labels and label pairs are consistent with each other.

**Node-edge agreement as necessary condition for dual optimality.** The statements of the following proposition follow directly from the definition of the local polytope $\mathcal{L}$. They show that arc consistency is an intrinsic property of all vectors in $\mathcal{L}$, and that strict arc-consistency is equivalent to the notion of integer labeling.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5.27</span></p>

1. $\boldsymbol{\mu} \in \mathcal{L}$ implies that $\operatorname{nz}[\boldsymbol{\mu}]$ is arc-consistent.
2. For any $\mathbf{y} \in \mathcal{Y}_\mathcal{V}$, $\boldsymbol{\delta}(\mathbf{y})$ is strictly arc-consistent.
3. For any $\boldsymbol{\theta} \in \mathbb{R}^{\mathcal{I}}$ the following two statements are equivalent:
   - a) $\operatorname{mi}[\boldsymbol{\theta}]$ is strictly arc-consistent;
   - b) $\operatorname{mi}[\boldsymbol{\theta}] \in \mathcal{L} \cap \lbrace 0, 1 \rbrace^{\mathcal{I}}$, i.e. $\operatorname{mi}[\boldsymbol{\theta}]$ is an integer labeling.

</div>

Proposition 5.27(1) defines a *necessary* condition for the dual optimum given in Proposition 5.23(5). Coordinates of $\boldsymbol{\mu} \in \mathcal{L}(\boldsymbol{\phi})$ may be greater than 0 only for locally optimal labels and label pairs. Therefore, for a dual vector $\boldsymbol{\phi}$ to be optimal it is *necessary* that there exists an arc-consistent subset of locally optimal coordinates (corresponding to labels and label pairs) of $\boldsymbol{\theta}^\phi$:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.28</span><span class="math-callout__name">(Node-Edge Agreement)</span></p>

One says that nodes and edges *agree* (or there is a *node-edge agreement*) for a cost vector $\boldsymbol{\theta} \in \mathbb{R}^{\mathcal{I}}$, if for each node (edge) $w \in \mathcal{V} \cup \mathcal{E}$ there is a non-empty subset of locally optimal labels (label pairs) $\mathbb{S}_w \subseteq \arg \min_{s \in \mathcal{Y}_w} \theta_w(s)$ such that the vector $\boldsymbol{\xi}$ with coordinates $\xi_w(s) = \llbracket s \in \mathbb{S}_w \rrbracket$ is arc-consistent.

</div>

Verification of the node-edge agreement is an efficiently solvable problem. The respective *relaxation labeling* algorithm is described, e.g., in [1, Sec. 6.2.4].

#### 5.5.3 Dual Optimality for Acyclic Labeling Problems

By construction, node-edge agreement is only necessary, but not sufficient dual optimality condition in general. However, there are a few cases, where this condition is sufficient as well. This holds in particular for binary labeling problems, i.e. when only two labels are associated with each node (see Chapter 9 for details), or when the labeling problems are acyclic. The latter case is considered below.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5.29</span></p>

Let $(\mathcal{G}, \mathcal{Y}_\mathcal{V}, \boldsymbol{\theta})$ be a labeling problem with the graph $\mathcal{G}$ being acyclic. Then the node-edge agreement implies $\boldsymbol{\phi}$ is the dual optimum. Moreover, there is an optimal integer labeling $\mathbf{y} \in \mathcal{Y}_\mathcal{V}$ such that $\boldsymbol{\delta}(\mathbf{y}) \leq \boldsymbol{\xi}$, where $\boldsymbol{\xi}$ is the vector defining the node-edge agreement.

Furthermore, for any $(w, s) \in (\mathcal{V} \cup \mathcal{E}) \times \mathcal{Y}_w$ such that $\boldsymbol{\xi} \neq \mathbf{0}$ there is an optimal integer labeling $\mathbf{y}^* \in \mathcal{Y}_\mathcal{V}$ such that $y_w^* = s$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Proposition 5.29</summary>

Without loss of generality we assume that $\mathcal{G}$ is connected, i.e. is a tree. Otherwise the proof can be done for each connected component separately.

Let $\boldsymbol{\xi}$ be the arc-consistent subset of labels and label pairs that defines the node-edge agreement. The proof will be done by constructing a labeling $\mathbf{y}$ such that $\boldsymbol{\delta}(\mathbf{y}) \leq \boldsymbol{\xi}$.

We will iteratively build a *connected* subgraph $(\mathcal{V}', \mathcal{E}')$ and assign labels to the nodes of $\mathcal{V}'$ such that if label pair $(y_u, y_v)$ is assigned to the edge $(u, v) \in \mathcal{E}'$ it holds that $\xi_u(y_u) = 1$, $\xi_v(y_v) = 1$ and $\xi_{uv}(y_u, y_v) = 1$. Our procedure finalizes when $\mathcal{V}' = \mathcal{V}$ and $\mathcal{E}' = \mathcal{E}$.

The sets $\mathcal{V}'$ and $\mathcal{E}'$ are empty at the beginning of the procedure. Let $u \in \mathcal{V}$ be any vertex. Consider a label $s \in \mathcal{Y}_u$ for which $\xi_u(s) = 1$. Assign $y_u := s$ and $\mathcal{V}' := \mathcal{V}' \cup \lbrace u \rbrace$.

On each iteration a node $v \in \mathcal{V} \setminus \mathcal{V}'$ is considered, which is incident to some node of $\mathcal{V}'$. Note that any node $v \in \mathcal{V} \setminus \mathcal{V}'$ is incident to *at most* one node in $\mathcal{V}'$. Indeed, if $u'$ and $u''$ would be two nodes of $\mathcal{V}'$ incident to $v$, there would be a path $p$ between them in the graph $(\mathcal{V}', \mathcal{E}')$, since the latter is connected. Therefore, there would be a cycle $u', v, u'', p, u'$, which would mean a contradiction, as the initial graph is acyclic.

Let therefore $v \in \mathcal{V} \setminus \mathcal{V}'$ be any node connected to some node $u \in \mathcal{V}'$. By definition of node-edge agreement there is a $t \in \mathcal{Y}_v$ such that $\xi_{uv}(y_u, t) = \xi_v(t) = 1$. Assign $y_v := t$ and $\mathcal{V}' := \mathcal{V}' \cup \lbrace v \rbrace$, $\mathcal{E}' := \mathcal{E}' \cup \lbrace uv \rbrace$.

Repeat the process until $\mathcal{V}' = \mathcal{V}$ and therefore $\mathcal{E} = \mathcal{E}'$.

By construction it holds that $\boldsymbol{\delta}(\mathbf{y}) \leq \boldsymbol{\xi}$ for the labeling $\mathbf{y}$, which finalizes the proof of the first two statements.

For the last statement, the labeling $\mathbf{y}^*$ can be constructed with $(u, s)$ being selected at the first step of the procedure. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 5.30</span></p>

Let $(\mathcal{G}, \mathcal{Y}_\mathcal{V}, \boldsymbol{\theta})$ be a labeling problem with the graph $\mathcal{G}$ being acyclic. Let $\mathcal{L}$ and $\mathcal{M}$ be the corresponding local and marginal polytopes. Then $\mathcal{L} = \mathcal{M}$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Corollary 5.30</summary>

Since $\mathcal{M} \subseteq \mathcal{L}$ it suffices to show that $\mathcal{L} \subseteq \mathcal{M}$. Consider a maximizer $\boldsymbol{\phi}$ of the Lagrange dual $\mathcal{D}$ defined by (5.122) for an acyclic problem with the cost vector $\boldsymbol{\theta}$. Since node-edge agreement is necessary for optimality, Proposition 5.29 implies that $\mathcal{L}(\boldsymbol{\phi})$ contains an integer labeling. In its turns, it implies (see statement 5 of Proposition 5.23) that this labeling is a solution to the local polytope relaxation of the labeling problem.

In other words, Proposition 5.29 implies that for any cost vector $\boldsymbol{\theta}$ there is always an integer solution of the local polytope relaxation of the labeling problem for acyclic graphs. It implies that only for vectors $\boldsymbol{\delta}(\mathbf{y})$, $\mathbf{y} \in \mathcal{Y}_\mathcal{V}$, corresponding to integer labelings, a cost vector $\boldsymbol{\theta}$ may exist such that $\boldsymbol{\delta}(\mathbf{y})$ is the unique solution of the relaxed problem $\min_{\boldsymbol{\mu} \in \mathcal{L}} \langle \boldsymbol{\theta}, \boldsymbol{\mu} \rangle$.

Due to Definition 2.20 this implies $\operatorname{vrtx}(\mathcal{L}) \subseteq \lbrace \boldsymbol{\delta}(\mathbf{y}) \mid \mathbf{y} \in \mathcal{Y}_\mathcal{V} \rbrace$. Therefore,

$$\mathcal{L} \subseteq \operatorname{conv}\lbrace \boldsymbol{\delta}(\mathbf{y}) \mid \mathbf{y} \in \mathcal{Y}_\mathcal{V} \rbrace = \mathcal{M} \subseteq \mathcal{L} \,,$$

which finalizes the proof. $\square$

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 5.31</span></p>

Let $(\mathcal{G}, \mathcal{Y}_\mathcal{V}, \boldsymbol{\theta})$ be a labeling problem with the graph $\mathcal{G}$ being acyclic. Let also $\boldsymbol{\mu}^* \in \mathcal{L}$ be a solution of the local polytope relaxation of the labeling problem. Then from $\mu_u^*(s) > 0$ for some $u \in \mathcal{V}$, $s \in \mathcal{Y}_u$ it follows that there is an optimal integer labeling $\mathbf{y}^*$ such that $y_u = s$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Corollary 5.31</summary>

The labeling $\mathbf{y}^*$ can be constructed as in the proof of Proposition 5.29 with $(u, s)$ being selected as the first node and label. $\square$

</details>
</div>

### 5.6 Bibliography and Further Reading

The local polytope relaxation for labeling problems, its dual and its analysis were first provided by Schlesinger (see [14], [15]). An algorithm equivalent to the relaxation labeling was independently proposed [16], Schlesinger (see Shlezinger [14]) and revisited by [17].

The comprehensive overview [18] is a standard reference for the relation of the primal and dual formulations of the energy minimization problem. It includes definitions of arc-consistency, the relaxation labeling algorithm and a lot of related notions and facts.

How to estimate a primal relaxed solution during dual optimization of the labeling problem is described in [19].

We refer to [20] and references therein for a definition of constraint satisfaction problems in general. The description of its solvable subclasses was first given in [21], see also [22]. An analysis of solvability of constraint satisfaction problems, where $\min/\max$ operations instead of $\vee/\wedge$ is given in [23]. A simple generalization of the relaxation labeling algorithm for such problems can be found in [24].

## Chapter 6: Basic Scalable Convex Optimization Techniques

The goal of this chapter is to give a brief overview of the simplest optimization methods applicable to large-scale convex programs, such as the Lagrange dual of the MAP-inference problem. To keep our exposition short, we restrict our overview to the three most basic techniques: gradient descent, the subgradient method and block-coordinate descent. The latter two will be used for the labeling problem, whereas the first one plays the role of a baseline for comparisons. These methods are applicable to smooth and non-smooth problems with different convergence guarantees. Since the convergence proofs and the derivation of the convergence rates are often quite involved and the techniques used in the proofs do not play any significant role for further chapters of the monograph, we mostly omit the proofs in this chapter and refer to the corresponding literature instead.

Throughout the chapter we will use the notation $\lVert \cdot \rVert$ for the Euclidean norm in $\mathbb{R}^n$.

### 6.1 Gradient Descent

We start the chapter with differentiable functions and the most basic algorithm for their optimization. Although for MAP-inference we will deal mostly with non-smooth optimization, we provide the basic results about gradient descent as a baseline for comparison to other methods.

**Lipschitz-continuity.** We will be interested in differentiable functions with continuous gradient. Its "degree of continuity" is determined by the following definition:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6.1</span><span class="math-callout__name">(Lipschitz-Continuous)</span></p>

A mapping $f \colon \mathbb{R}^n \to \mathbb{R}^m$ is called *Lipschitz-continuous* on $X \subseteq \mathbb{R}^n$ if there is a constant $L \geq 0$ such that for any $x, z \in X$ it holds that

$$\lVert f(x) - f(z) \rVert \leq L \lVert x - z \rVert \,.$$

The smallest value $L$ satisfying the condition above is called the *Lipschitz constant* for $f$ on $X$.

</div>

Informally speaking, the Lipschitz constant is an upper bound for the speed of change of the value of the mapping $f$. The following statement considers the limit case of this inequality and allows us to estimate the value of the Lipschitz constant in simple cases:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.2</span></p>

If $f \colon \mathbb{R}^n \to \mathbb{R}^m$ is differentiable, then

$$L = \sup_{x \in X} \lVert \nabla f(x) \rVert_2 \,,$$

where $\lVert \cdot \rVert_2$ is a spectral norm, i.e. the largest eigenvalue of the matrix $\nabla f$. In a special case, when $m = 1$, it coincides with the Euclidean norm.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.3</span></p>

Proposition 6.2 implies that:

- $f(x) = x$ is Lipschitz-continuous on $\mathbb{R}$;
- $f(x) = x^n$, $n = 2, 3, \ldots$ is Lipschitz-continuous on any bounded subset of $\mathbb{R}$, e.g. on $[0, 1]$, but not on $\mathbb{R}$ itself.

</div>

We will use the notation $C_L^{1,1}(X)$ for functions $f \colon X \to \mathbb{R}$, which are differentiable on $X$ and whose gradient is Lipschitz-continuous on $X$ with the Lipschitz constant $L$. In general $C_L^{k,m}(X)$ is a standard notation for the set of $k$-times differentiable functions such that their $m$th derivative is Lipschitz-continuous on $X$ with the constant $L$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.4</span></p>

$C_L^{1,1}(\mathbb{R})$ includes such functions as $x$ and $x^2$. The functions $x^n$ for $n = 3, 4, \ldots$ belong to $C_L^{1,1}(X)$ for any bounded $X \subset \mathbb{R}$, but not for $X = \mathbb{R}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.5</span></p>

Let $f \colon \mathbb{R}^n \to \mathbb{R}$ be convex and differentiable. Then $\nabla f(x) = \bar{0}$ is the necessary and sufficient condition for $x \in \mathbb{R}^n$ to be a minimum of $f$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 6.5</summary>

This follows directly from the similar property of the subgradient expressed by Theorem 4.35 together with Proposition 4.27. $\square$

</details>
</div>

**Gradient descent algorithm.** Let $f \colon \mathbb{R}^n \to \mathbb{R}$ and $x^0 \in \mathbb{R}^n$ be an initial point. The iterative process of the form

$$x^{t+1} = x^t - \alpha^t \nabla f(x^t)$$

for $t = 0, \ldots, \infty$ and some $\alpha^t > 0$ is called *gradient descent*. The value $\alpha^t$ is called the *step-size* of the algorithm.

Let $x^* \in \arg \min_{x \in \operatorname{dom} f} f(x)$ and $f^* = f(x^*)$. The following statement specifies convergence properties of the gradient descent:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.6</span></p>

Let $f \in C_L^{1,1}(\mathbb{R}^n)$ be convex. Then with $\alpha^t = \frac{1}{L}$ for the sequence $x^t$ defined by (6.2) it holds that:

$$\lVert x^{t+1} - x^* \rVert^2 \leq \lVert x^t - x^* \rVert^2 - \frac{1}{L^2} \lVert \nabla f(x^t) \rVert^2 \,,$$

$$f(x^{t+1}) \leq f(x^t) - \frac{1}{2L} \lVert \nabla f(x^t) \rVert^2 \,,$$

$$f(x^t) - f^* \leq \frac{2L \lVert x^0 - x^* \rVert}{t + 4} \,.$$

</div>

Let us consider the statement of the theorem in detail. First of all, it considers a *constant* step size $\alpha := \alpha^t = \frac{1}{L}$, which is inverse-proportional to the speed of change of the gradient $\nabla f$ expressed by its Lipschitz constant. In other words, the faster the gradient changes, the smaller the step-size must be selected. Therefore, the Lipschitz constant defines the vicinity in which the considered function has "approximately the same" gradient. The larger the constant the smaller is the vicinity and, hence, the smaller the step-size.

Expression (6.3) states that each step of the algorithm produces a solution estimate $x^{t+1}$ which is closer to the optimum $x^*$ than the previous estimate $x^t$.

Expression (6.4) shows that the algorithm is strictly monotonous, i.e. unless $\nabla f = \bar{0}$, the objective value strictly decreases on each iteration.

Expression (6.5) provides the *convergence rate* of the gradient descent algorithm. The accuracy $\epsilon = f(x^t) - f^*$ is attained after at most $O\!\left(\frac{L}{\epsilon}\right)$ iterations. This complexity is often formulated in a "dual" fashion, when one says that the algorithm requires $O\!\left(\frac{L}{\epsilon}\right)$ iterations to attain a given accuracy $\epsilon$.

**Step-size selection for gradient descent.** One may object that a single Lipschitz constant $L$ can be a too rough estimation for a behavior of a function on its whole domain and, therefore, the constant step-size policy $\alpha = \frac{1}{L}$ may not work well in practice. This is indeed often the case. Consider the function $x^3$ on the interval $[0, b]$. The Lipschitz constant of its derivative $3x^2$ on this interval is bounded by the value $6b$ according to Proposition 6.2. If $b = 1$ this results in $L = 6$ and if $b = 100$ Proposition 6.2 gives $L = 600$. The step-size $\frac{1}{600}$ would lead to a much slower convergence in the vicinity of $x = 1$ than the step-size $\frac{1}{6}$. In other words, the step-size and the convergence speed depend on the domain on which the estimation of the Lipschitz constant was done. Therefore, a variable step-size quite often works better in practice.

There are three typical ways to define $\alpha^t$ in (6.2):

- $\alpha^t = \frac{1}{L}$ — constant step size. Requires knowing the Lipschitz constant or a good estimate of it. May be very inefficient if $\lVert \nabla f \rVert$ significantly varies over the domain of $f$.
- $\alpha^t = \arg \min_{\alpha \in \mathbb{R}_+} f(x^t - \alpha \nabla f(x^t))$, which is the step-size that minimizes $f$ in the negative gradient direction. For efficiency of the gradient descent algorithm as a whole, a closed form solution for $\alpha^t$ is typically required, which is often not available.
- $\alpha^t$ is selected to satisfy

$$f(x^{t+1}) \leq f(x^t) - \frac{\alpha^t}{2} \lVert \nabla f(x^t) \rVert^2 \,.$$

Here $\frac{1}{\alpha^t}$ plays the role of a local estimate for $L$. To this end $\alpha^t$ is searched in the form $\alpha^t = \frac{2\alpha^{t-1}}{2^n}$ for $n = 0, 1, \ldots, \infty$ until (6.6) is satisfied. In other words, one tries to double the step size, and divides the step-size by two until the condition is not satisfied anymore. The initial value $\alpha^0$ can be selected arbitrary, as soon as more than just few iterations of the algorithm are used. See the *Goldstein-Armijo* rule in [10], [12] for a substantiation.

Though the practical behavior can differ for the above three cases, their worst-case analysis is similar and for any of these strategies it results in the convergence rate given by Theorem 6.6.

### 6.2 Sub-Gradient Method

However, we can not use the gradient descent for our dual MAP-inference problem (5.122) directly, because it is not differentiable.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6.7</span><span class="math-callout__name">(Subgradient Method)</span></p>

Let $x^0 \in \mathbb{R}^n$ be a starting point and $f \colon \mathbb{R}^n \to \mathbb{R}$ be a convex function. The iterative process of the form

$$x^{t+1} = x^t - \alpha^t g(x^t) \,,$$

for $t = 0, \ldots, \infty$, some $\alpha^t > 0$ and $g(x^t) \in \partial f(x^t)$, is called a *subgradient method*.

</div>

The update rule (6.7) is basically the same as (6.2). However, when the function $f$ is non-smooth, the step-size $\alpha^t$ must be selected differently than in the smooth case:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.8</span></p>

Let $f$ be convex and Lipschitz continuous in a ball $B_R(x^*) = \lbrace x \in \mathbb{R}^n \colon \lVert x^* - x \rVert \leq R \rbrace$ with Lipschitz constant $M$. Let $x^0 \in B_R(x^*)$ be the initial point and let the step-size $\alpha^t$ satisfy

$$\alpha^t > 0, \quad \alpha^t \xrightarrow{t \to \infty} 0, \quad \text{and} \quad \sum_{t=1}^{\infty} \alpha^t = \infty \,.$$

Then the following holds:

- $f(x^t) - f^* \xrightarrow{t \to \infty} 0$;
- $0 < \alpha^t < 2\frac{f(x^t) - f^*}{\lVert g^t \rVert^2}$ implies $\lVert x^{t+1} - x^* \rVert \leq \lVert x^t - x^* \rVert$;
- In particular, $\alpha^t = \frac{R}{\lVert g^t \rVert \sqrt{t+1}}$ for all $t = 1, 2, \ldots$ implies

$$f(x^t) - f^* \leq \frac{MR}{\sqrt{t+1}} \,.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 6.9</span><span class="math-callout__name">(Practical step-size rules)</span></p>

Since neither the constant $R$ nor the optimal value $f^*$ are typically known, a variety of different practical step-size rules are used. One of them [26, Sec. 6.3.1] is the approximation of those for (6.9):

$$\alpha^t = \frac{\beta^t (f(x^t) - \hat{f}^t)}{\lVert g^t \rVert^2} \,,$$

where $\hat{f}^t$ is the approximation of the optimal value and $0 < \beta^t < 2$.

In particular [26, Sec. 6.3.1] suggests two common ways to choose $\beta^t$ and $\hat{f}^t$:

- The first option is assign $\hat{f}^t$ to use the best known lower bound for $f^*$. The latter can often be computed from the dual information. For example, if the primal problem is ILP, the subgradient is used to optimize its convex Lagrange dual, the best lower bound for $f^*$ can be either the best known value of the initial ILP objective, or, if available, the best known value of the primal of the Lagrange relaxation. The value of $\beta^t$ is selected then as, e.g.,

$$\beta^t = \frac{1 + m}{t + m} \,,$$

where $m$ is some fixed integer preventing too rapid decrease of the step-size on the first iterations and division by 0 for $t = 0$. Note that $\beta^t$ selected in this way satisfies the conditions for the diminishing step-size rule (6.8).

- The second option is to set $\beta^t = 1$ and

$$\hat{f}^t := m^t / (1 + \gamma^t)$$

with $m^t$ being the best attained value so far, i.e., $m^t := \min_{0 \leq i \leq t} f(x^i)$. Here $\gamma^t > 0$ and its value is increased by a certain factor (e.g., 1.05) if the previous iteration improves the best attained value so far, and decreased by some other factor otherwise. The method assumes $m^t > 0$.

- The third option is the modification of the second one, where the updates are performed according to

$$\hat{f}^t := \max\lbrace \tilde{f}^t, m^t / (1 + \gamma^t) \rbrace$$

with $\tilde{f}^t$ being the best available lower bound for $f^*$.

Note that the formulas have to be respectively changed if the subgradient is used to maximize a concave function. This case is covered in [26, Sec. 6.3.1].

</div>

Note the following important properties of the sub-gradient method stated by Theorem 6.8:

- For a non-differentiable function $f$ the convergence is guaranteed only if the step-size $\alpha^t$ satisfies the *diminishing step-size rule* (6.8). This is due to the fact that the subgradient need not vanish in the vicinity of the minimum, contrary to the gradient, see Figure 6.2 for illustration. Since neither $f(x^t) - f^*$ nor $R$ is typically known, the claims of Theorem 6.8 can be simplified as *there exists* $\alpha^t$ such that (6.9) holds, and *there exist* $\lbrace \alpha^t \rbrace$ such that (6.10) holds. Step-sizes which decrease as $O(\frac{1}{t})$ or $O(\frac{1}{\sqrt{t}})$ are typical choices that satisfy condition (6.8).

- The update rule (6.7) does not guarantee the monotonic improvement of the function value if $f$ is non-differentiable. In other words, it may happen that $f(x^{t+1}) \geq f(x^t)$, even for an arbitrary small $\alpha^t$, see Figure 6.3 for illustration. It implies that minimization of $f$ in the direction of a negative subgradient does not make sense, contrary to the case when $f$ is smooth. On the other hand, the distance to the optimum $x^*$ never grows during iterations, according to (6.9).

- The convergence rate defined by (6.10) can be written as $O(\frac{1}{\sqrt{t}})$, or, alternatively, as $O(\frac{1}{\epsilon^2})$. This is significantly slower than the convergence rate $O(\frac{L}{\epsilon})$ in the smooth case. For example, to attain the precision $\epsilon = 0.1$ the subgradient algorithm must perform 100 times more iterations than to obtain the precision $\epsilon = 1$. Note that the gradient descent would require only 10 times more iterations given that the function to be optimized is smooth.

| | Smooth $f$ | Non-smooth $f$ |
| --- | --- | --- |
| Update rule | $x^{t+1} = x^t - \alpha^t \nabla f(x^t)$ | $x^{t+1} = x^t - \alpha^t g^t$, $g^t \in \partial f(x^t)$ |
| Step-size $\alpha_t > 0$ | $\alpha^t = \frac{1}{L}$ | $\alpha^t \xrightarrow{t \to \infty} 0$, $\sum_{t=1}^{\infty} \alpha^t = \infty$ |
| Distance to optimum | $\lVert x^{t+1} - x^* \rVert^2 \leq \lVert x^t - x^* \rVert^2 - \frac{1}{L^2} \lVert \nabla f(x^t) \rVert^2$ | $\lVert x^{t+1} - x^* \rVert \leq \lVert x^t - x^* \rVert$ |
| Monotonicity | $f(x^{t+1}) \leq f(x^t) - \frac{1}{2L} \lVert \nabla f(x^t) \rVert^2$ | — |
| Convergence rate | $f(x^t) - f^* \leq \frac{2L \lVert x^0 - x^* \rVert}{t+4}$ | $f^t - f^* \leq \frac{MR}{\sqrt{t+1}}$ |

### 6.3 Coordinate Descent

Coordinate or block-coordinate descent is the last optimization method in our short overview. Like the subgradient method it is well-defined for both smooth and non-smooth functions, although the corresponding convergence guarantees differ significantly.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6.10</span><span class="math-callout__name">(Cyclic Coordinate Descent)</span></p>

For a function $f \colon \prod_{j=1}^{n} \mathbb{R}^{n_j} \to \mathbb{R}$ the iterative process

$$x_i^{t+1} = \arg \min_{\xi \in \mathbb{R}^{n_i}} f(x_1^{t+1}, \ldots, x_{i-1}^{t+1}, \xi, x_{i+1}^t, \ldots, x_n^t)$$

$$i = (i+1) \mod n$$

$$t = 1, 2, \ldots$$

is called *cyclic* (or *Gauss-Seidel*) *coordinate descent*. Here $i \mod n$ denotes the remainder from the integer division of $i$ to $n$.

</div>

The method is also known as *alternating minimization*, since one alternates the optimization w.r.t. the different variables. When the dimensionality $n_j$ of individual variables is greater than one, one often speaks also about *block*-coordinate descent.

Note that by definition the coordinate descent is monotonous, i.e. it holds that

$$\underbrace{f(x_1^{t+1}, \ldots, x_{i-1}^{t+1}, x_i^t, x_{i+1}^t, \ldots, x_n^t)}_{f_{i-1}^t} \geq \underbrace{f(x_1^{t+1}, \ldots, x_{i-1}^{t+1}, x_i^{t+1}, x_{i+1}^t, \ldots, x_n^t)}_{f_{i-1}^{t+1}} \,,$$

and therefore, the sequence $f_i^t$ is monotonously non-increasing w.r.t. $k = n \cdot t + i \to \infty$. Assuming that the minimal value of $f$ exists, i.e. $f^* > -\infty$, this implies convergence of the sequence $f_i^t$. To analyze this convergence and the convergence of the argument sequences $x_i^t$ for each $i$ as $t \to \infty$, we will require the following simple lemma:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 6.11</span></p>

Let $f \colon \mathbb{R}^n \times \mathbb{R}^m \to \mathbb{R}$ be a convex function of two (vector-)variables. Then for any fixed $z' \in \mathbb{R}^m$ the function $f_{z'}(x) := f(x, z')$ is convex w.r.t. $x$ and its optimum is attained in $x \in \mathbb{R}^n$ if and only if $\bar{0} \in \partial f_{z'}(x)$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Lemma 6.11</summary>

The epigraph of $f_{z'}$ is an intersection of the epigraph of $f$ and the linear subspace $z = z'$. Therefore, it is convex as an intersection of two convex sets. This proves convexity of $f_{z'}$. The claim about optimality of $x$ follows from Theorem 4.35. $\square$

</details>
</div>

Let us now consider a function $f \colon \prod_{i=1}^{n} \mathbb{R}^{n_j} \to \mathbb{R}$ of $n$ variables. Let $f_i(\xi; x') := f(x_1', \ldots, x_{i-1}', \xi, x_{i+1}', \ldots, x_n')$ be the function of the $i$-th variable given all others are fixed to the corresponding coordinates of $x'$. Assume $x'$ is a fixed point of the algorithm (6.15), i.e. $x_i' \in \arg \min_{\xi \in \mathbb{R}^{n_i}} f_i(\xi; x')$. Due to Lemma 6.11 it holds that

$$\bar{0} \in \partial f_i(x_i'; x') \quad \text{for all } i = 1, \ldots, n \,.$$

If $f$ is convex differentiable, then condition (6.19) is equivalent to $\nabla f(x') = \bar{0}$ and, therefore, a fixed point is an optimum.

However, if $f$ is non-differentiable, this implication does not work, as shown in Figure 6.3(c). This is due to the non-uniqueness of the subgradient, which causes this behavior of the fixed point for non-differentiable functions. In other words, there may exist a zero subgradient for each coordinate, but not for all of them together.

Moreover, in general, the algorithm (6.15) does not even guarantee convergence to such a point where the conditions (6.19) are fulfilled. This negative statement holds even if $f$ is differentiable, as shown by [28]. Indeed, to guarantee convergence, an additional *uniqueness* property is required:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.12</span></p>

Let $f$ be continuously differentiable on $\mathbb{R}^n$. Furthermore, let for each $i$ and $x \in \mathbb{R}^n$ the minimum in (6.15) be *uniquely* attained. Then for every limit point $x^*$ of the sequence $x^t$, $t = 1, 2, \ldots$ defined by (6.15) it holds that $\nabla f(x^*) = 0$.

</div>

If $f$ is convex differentiable the optimality of $x$ follows from the condition $\nabla f(x) = \bar{0}$. Theorem 6.12 does not claim existence nor uniqueness of the limit points. It only proves their properties if they exist. In particular, if the set $F(x^0) := \lbrace x \colon f(x) \leq f(x^0) \rbrace$ is bounded, then $x^t \in F(x^0)$ due to the monotonicity of the coordinate descent, i.e. $f(x^t) \leq f(x^0)$. Hence $\lbrace x^t \rbrace$ is bounded and has limit points, which all correspond to the optimum of $f$ according to Theorem 6.12.

**Convergence rate.** A little is known about the convergence rate of cyclic coordinate descent (6.15) for general convex differentiable functions. Recently, the convergence rate $O(\frac{1}{t})$ was proven by [29] for the case $n = 2$, i.e. when only two blocks of variables are considered in (6.15).

### 6.4 Bibliography and Further Reading

In this chapter we have reviewed classical results which can be found in the text-books [10], [25], [26].

The subgradient method as well as a number of its variants and applications to the non-smooth minimization have been proposed by N. Z. Shor in 1960's, see [30] and references therein.

Although coordinate descent belongs to one of the earliest algorithms, the results about its convergence and the corresponding convergence rates in general are still missing. Earlier works on this topic mostly considered specializations to different function subclasses e.g. [31]–[34]. The most general convergence results for variants of the coordinate descent were recently given in [29].

A very recent series of publications **Dlask and Werner** considers theoretical properties and application of block-coordinate ascent to (relaxations of) the combinatorial problems.

### 6.5 Case Study: Labeling Problem

In this section we apply the optimization methods from the above sections to the labeling problem.

First of all, we consider one of the simplest optimization methods for this problem, known as *Iterated Conditional Modes (ICM)*. This algorithm is a direct implementation of the coordinate descent technique to the primal discrete objective of the labeling problem.

Afterwards, we consider more powerful techniques addressing the Lagrangean relaxation and the respective dual problem. As we learned in Section 5.5, the latter can be seen as an unconstrained concave non-smooth maximization problem. Therefore, the subgradient method is applicable directly and we provide the corresponding analysis.

However, the most efficient existing methods for the Lagrangean dual are based on the block-coordinate ascent technique. Although these methods do not attain the dual optimum in general, they are often able to attain practically good approximate solutions in a moderate number of iterations.

#### 6.5.1 Primal Coordinate Descent

**Iterated Conditional Modes.** Iterated Conditional Modes (ICM) is one of the first and simplest algorithms for energy minimization. As we show below, it is an implementation of the coordinate descent (6.15) for the primal non-relaxed energy minimization objective $E(\mathbf{y}; \boldsymbol{\theta})$.

The algorithm starts with some labeling $\mathbf{y} \in \mathcal{Y}_\mathcal{V}$ and iteratively tries to improve it. In its elementary step it minimizes the objective w.r.t. the label of a node $u \in \mathcal{V}$, whereas the labels of other nodes are kept fixed.

Let the notation

$$l(\mathbf{y}, u, s) := y' \in \mathcal{Y}_\mathcal{V} \quad \text{with} \quad y_v' = \begin{cases} y_v, & v \neq u \\ s, & v = u \end{cases}$$

define a labeling where all coordinates but the one corresponding to node $u$ coincide with the labeling $\mathbf{y}$ and the coordinate $u$ is assigned the label $s$.

Then the elementary step of the algorithm can be written as

$$y_u := \arg \min_{s \in \mathcal{Y}_u} E(l(\mathbf{y}, u, s); \boldsymbol{\theta}) \,.$$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm 3</span><span class="math-callout__name">(ICM: Coordinate Descent for $E(y; \theta)$)</span></p>

1. **Init:** $\mathbf{y} \in \mathcal{Y}_\mathcal{V}$
2. **repeat**
3. &emsp; **for** $u \in \mathcal{V}$ **do**
4. &emsp;&emsp; $y_u := \arg \min_{s \in \mathcal{Y}_u} E(l(\mathbf{y}, u, s); \boldsymbol{\theta})$
5. &emsp; **end for**
6. **until** the labeling $\mathbf{y}$ does not change anymore

</div>

Algorithm 3 applies this rule to each node sequentially and iterates this procedure until the labeling $\mathbf{y}$ does not change anymore.

As any coordinate descent, Algorithm 3 guarantees that the objective value of $E$ is monotonically nonincreasing. However, since $E$ as a function of $\mathbf{y}$ is neither convex nor differentiable (moreover, it is defined on a discrete set $\mathcal{Y}_\mathcal{V}$ only), Algorithm 3 does not guarantee attainment of the optimal value of $E$ and its output significantly depends on the initial labeling. In practice Algorithm 3 typically returns labelings with significantly higher energies than many other techniques.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.13</span></p>

Consider a simple two-node model with two labels in each node. Suppose that the unary costs are all zero and the pairwise costs are $\theta_{uv}(0, 0) = 0$, $\theta_{uv}(1, 1) = 1$, $\theta_{uv}(0, 1) = \theta_{uv}(1, 0) = 2$. If the initial labeling is $(1, 1)$, the ICM updates cannot improve the labeling, since they can switch only to the labelings $(1, 0)$ or $(0, 1)$, each with total cost 2, while the labeling $(1, 1)$ has total cost 1. However, the optimum is attained in labeling $(0, 0)$, which has total cost 0.

</div>

#### 6.5.2 Block-ICM Algorithm

The idea of the ICM algorithm goes beyond Algorithm 3. In general, one has to fix labels in a subset of nodes and optimize with respect to the labels in the remaining nodes. Should this optimization be tractable then the whole algorithm is as well.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6.14</span><span class="math-callout__name">(Induced Subgraph)</span></p>

A subgraph $(\mathcal{V}', \mathcal{E}')$ of a graph $(\mathcal{V}, \mathcal{E})$ is called *induced* by its set of nodes $\mathcal{V}'$, if $\mathcal{E}' = \lbrace uv \in \mathcal{E} \colon u, v \in \mathcal{V}' \rbrace$, i.e. its set of edges contains exactly those edges from $\mathcal{E}$, that connect nodes from $\mathcal{V}'$.

</div>

Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ be a graph defining a graphical model and $\mathcal{V}^f \subset \mathcal{V}$ be a set of nodes with fixed labels. The corresponding partial labeling will be denoted as $y^f \in \mathcal{Y}_{\mathcal{V}^f}$.

Introducing $\mathcal{V}' = \mathcal{V} \setminus \mathcal{V}^f$ as complement of $\mathcal{V}^f$ we can generalize the notation defined in (6.20) such that

$$l(\mathbf{y}^f, \mathcal{V}^f, \mathbf{y}') := \tilde{\mathbf{y}} \in \mathcal{Y}_\mathcal{V} \quad \text{with} \quad \tilde{y}_u = \begin{cases} y_u^f, & u \in \mathcal{V}^f \\ y_u', & u \in \mathcal{V}' \end{cases}$$

defines a labeling, where nodes from the set $\mathcal{V}^f$ are labeled with $\mathbf{y}^f$, otherwise with $\mathbf{y}'$.

Let $(\mathcal{V}', \mathcal{E}')$ be the subgraph of $\mathcal{G}$ induced by $\mathcal{V}'$. It turns out that if it is acyclic, one can optimize w.r.t. the labeling on $\mathcal{V}'$,

$$\mathbf{y} := \arg \min_{\mathbf{y}' \in \mathcal{Y}_{\mathcal{V}'}} E(l(\mathbf{y}^f, \mathcal{V}^f, \mathbf{y}'); \boldsymbol{\theta}) \,,$$

using dynamic programming. Let $(\mathcal{V}^f, \mathcal{E}^f)$ be the subgraph of $\mathcal{G}$ induced by $\mathcal{V}^f$ and let $\mathcal{E}'^f = \lbrace uv \in \mathcal{E} \colon u \in \mathcal{V}', v \in \mathcal{V}^f \rbrace$ be the set of edges of $\mathcal{G}$ connecting the nodes from $\mathcal{V}'$ and $\mathcal{V}^f$. Consider the energy $E(l(\mathbf{y}^f, \mathcal{V}^f, \mathbf{y}'); \boldsymbol{\theta})$, defined by (6.22):

$$E(l(\mathbf{y}^f, \mathcal{V}^f, \mathbf{y}'); \boldsymbol{\theta})$$

$$= \sum_{u \in \mathcal{V}'} \left( \theta_u(y_u') + \sum_{v \in \mathcal{N}(u) \cap \mathcal{V}^f} \theta_{uv}(y_u', y_v^f) \right) + \sum_{uv \in \mathcal{E}'} \theta_{uv}(y_u', y_v') + \sum_{u \in \mathcal{V}^f} \theta_u(y_u^f) + \sum_{uv \in \mathcal{E}^f} \theta_{uv}(y_u^f, y_v^f) \,.$$

Recall that $\mathbf{y}^f$ is fixed. Therefore, the last expression represents the energy of a problem on the graph $(\mathcal{V}', \mathcal{E}')$ with modified unary costs, plus a constant represented by the last two terms of the expression. It implies that minimization of $E(l(\mathbf{y}^f, \mathcal{V}^f, \mathbf{y}'); \boldsymbol{\theta})$ w.r.t. $\mathbf{y}'$ can be done efficiently by dynamic programming if the graph $(\mathcal{V}', \mathcal{E}')$ is acyclic.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm 4</span><span class="math-callout__name">(Block-ICM: Block-Coordinate Descent for $E(\mathbf{y}; \boldsymbol{\theta})$)</span></p>

1. **Init:** $\mathbf{y} \in \mathcal{Y}_\mathcal{V}$
2. **repeat**
3. &emsp; Generate $\mathcal{V}' \subseteq \mathcal{V}$ such that $(\mathcal{V}', \mathcal{E}')$ induced by $\mathcal{V}'$ is acyclic
4. &emsp; Define $\mathcal{V}^f := \mathcal{V} \setminus \mathcal{V}'$ and $\mathbf{y}^f := \mathbf{y}\rvert_{\mathcal{V}^f}$
5. &emsp; Compute $\mathbf{y}^* := \arg \min_{\mathbf{y}' \in \mathcal{Y}_{\mathcal{V}'}} E(l(\mathbf{y}^f, \mathcal{V}^f, \mathbf{y}'); \boldsymbol{\theta})$
6. &emsp; Re-assign the values of $\mathbf{y}$ on the coordinates of $\mathcal{V}'$: $\mathbf{y}\rvert_{\mathcal{V}'} := \mathbf{y}^*$
7. **until** some stopping condition holds

</div>

Algorithm 4 summarizes these observations. At each iteration it generates a subset $\mathcal{V}'$ of nodes over which it minimizes the corresponding auxiliary energy (6.24) with dynamic programming, until the current labeling does not change for a certain number of iterations or the total iteration limit is reached.

The way the sets $\mathcal{V}'$ are generated can be either quite generic or depend on a particular graph structure. For grid-graphs a simple and well-parallelizable approach is to consider four graphs $(\mathcal{V}', \mathcal{E}')$ in a cyclic order: columns with even indexes, columns with odd indexes, rows with even indexes and rows with odd indexes. Note that minimization in line 5 of Algorithm 4 can be done for all columns (rows) of each of these subgraphs in parallel.

#### 6.5.3 Dual Sub-Gradient Method

Now we turn to the maximization of the Lagrange dual. Recall, it has the form of an unconstrained piecewise linear concave function

$$\mathcal{D}(\boldsymbol{\phi}) = \sum_{u \in \mathcal{V}} \min_{s \in \mathcal{Y}_u} \theta_u^\phi(s) + \sum_{uv \in \mathcal{E}} \min_{(s,t) \in \mathcal{Y}_{uv}} \theta_{uv}^\phi(s,t) \,,$$

with the reparametrized costs $\boldsymbol{\theta}^\phi$ defined by (5.119).

The subgradient method (6.7) is the simplest way to optimize concave non-smooth functions in general. To be able to use it we have to compute a subgradient of $\mathcal{D}$.

Consider its single coordinate $\frac{\partial \mathcal{D}}{\partial \phi_{u,v}(s)}$ and the function

$$\mathcal{D}_{u,v}(\phi_{u,v}) := \min_{s' \in \mathcal{Y}_u} \theta_u^\phi(s') + \min_{(s', t') \in \mathcal{Y}_{uv}} \theta_{uv}^\phi(s', t') \,,$$

containing exactly those two terms of $\mathcal{D}$, which depend on $\phi_{u,v}$. Other terms do not contribute to $\frac{\partial \mathcal{D}}{\partial \phi_{u,v}(s)}$, due to the linear properties of the subgradient (Proposition 4.28) and due to the fact that the subgradient of a constant function is identical to zero (since any constant function is differentiable and has gradient zero, which is also its subgradient, see Proposition 4.27). In other words, $\frac{\partial \mathcal{D}}{\partial \phi_{u,v}(s)} = \frac{\partial \mathcal{D}_{u,v}}{\partial \phi_{u,v}(s)}$.

Recall also how $\boldsymbol{\theta}^\phi$ depends on $\boldsymbol{\phi}$:

$$\theta_u^\phi(s) := \theta_u(s) - \sum_{v \in \mathcal{N}(u)} \phi_{u,v}(s), \quad u \in \mathcal{V},\; s \in \mathcal{Y}_u \,,$$

$$\theta_{uv}^\phi(s,t) := \theta_{uv}(s,t) + \phi_{u,v}(s) + \phi_{v,u}(t), \quad uv \in \mathcal{E},\; (s,t) \in \mathcal{Y}_{uv} \,.$$

Due to the linearity of the subgradient we can compute it for each term of $\mathcal{D}_{u,v}$ separately. Subgradients of the first term $\min_{s' \in \mathcal{Y}_u} \theta_u^\phi(s')$ are by virtue of Lemma 4.32 equal to the convex hull of the vectors $\frac{\partial \theta_u^\phi(s')}{\partial \phi_{u,v}(s)}$ for all minimizers $s'$ of (6.28). Since the numerator is a linear and, therefore, differentiable function of $\boldsymbol{\phi}$, we can apply the standard differentiation rules, yielding $\frac{\partial \theta_u^\phi(s')}{\partial \phi_{u,v}(s)} = -\llbracket s = s' \rrbracket$.

Similarly, subgradients of the second term in (6.26) can be computed as $\frac{\partial \theta_{uv}^\phi(s'', t'')}{\partial \phi_{u,v}(s)} = \llbracket s = s'' \rrbracket$ for each minimizer $(s'', t'')$ of the second term. Note that if $s' = s'' = s$ the subgradients of both terms cancel out.

We now combine our observations to construct a subgradient of $\mathcal{D}$. Let

$$y_u' \in \arg \min_{s \in \mathcal{Y}_u} \theta_u^\phi(s)$$

and

$$(y_u'', y_v'') \in \arg \min_{(s,t) \in \mathcal{Y}_{uv}} \theta_{uv}^\phi(s,t)$$

be defined for all $u \in \mathcal{V}$ and all $uv \in \mathcal{E}$. The vector $\frac{\partial \mathcal{D}}{\partial \boldsymbol{\phi}}$ with coordinates

$$\frac{\partial \mathcal{D}}{\partial \phi_{u,v}(s)} = \begin{cases} 0, & s \neq y_u' \text{ and } s \neq y_u'' \\ 0, & s = y_u' \text{ and } s = y_u'' \\ -1, & s = y_u' \text{ and } s \neq y_u'' \\ 1, & s \neq y_u' \text{ and } s = y_u'' \end{cases}$$

is a subgradient of $\mathcal{D}$.

Note that (6.31) defines only one possible subgradient. It is unique only if $y_u'$ and $(y_u'', y_v'')$ are *unique* solutions of the respective local minimization problems (6.29) and (6.30). Multiple solutions imply multiple subgradients of the form (6.31) and the convex combinations thereof.

However, any of these subgradients can be used in the subgradient method (6.7), which is now defined up to the step-size $\alpha^t$:

$$\boldsymbol{\phi}^{t+1} = \boldsymbol{\phi}^t + \alpha^t \frac{\partial \mathcal{D}}{\partial \boldsymbol{\phi}} \,.$$

We recommend to use the practical step-size rules, as described in Remark 6.9.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 6.15</span></p>

Note that subgradients computed according to (6.31) are quite sparse. For a given node $u$ and edge $uv$ there are $\lvert \mathcal{Y}_u \rvert$ entries of the subgradient, corresponding to the $\lvert \mathcal{Y}_u \rvert$ dual variables $\phi_{u,v}(s)$, $s \in \mathcal{Y}_u$. According to (6.31) at most two coordinates of the subgradient have non-zero entries. The rest $\lvert \mathcal{Y}_u \rvert - 2$ are zero. This, in turn, implies that only a small number of coordinates of the dual vector $\boldsymbol{\phi}$ are changed on each iteration of the subgradient method. This is one of the reasons, why this algorithm converges quite slowly in practice, especially for problems with a large number of labels $\mathcal{Y}_u$.

</div>

**Sufficient dual optimality condition.** Note that Theorem 4.35 provides a necessary and sufficient optimality condition for the Lagrange dual $\mathcal{D}$. It is the existence of a zero element in $\partial \mathcal{D}(\boldsymbol{\phi})$. However, checking this condition is in general computationally costly, as it would require to check whether the convex polyhedral set $\partial \mathcal{D}$ is non-empty. This is in general as difficult as solving a linear program of the size comparable to the size of the relaxed labeling problem. Therefore, a simplified *sufficient* optimality condition is typically used, where a zero subgradient is searched only in the form given by (6.31). To this end it must be verified, whether $y_u' = y_u''$ for all $u \in \mathcal{V}$, or, in other words, if there is an integer labeling consisting of locally optimal labels and label pairs. Indeed, Item 5 of Proposition 5.23 also implies sufficiency of this condition.

#### 6.5.4 Min-Sum Diffusion as Block-Coordinate Ascent

Till the end of this section we will concentrate on a block-coordinate ascent method for the Lagrange dual (6.25). The one we deal with is one of the first such methods proposed for the labeling problem, *min-sum diffusion*. Although it is not the most efficient algorithm of the considered type, its simplicity allows us to illustrate the properties typical for most block-coordinate ascent algorithms for the Lagrange dual of the labeling problem. An additional advantage of min-sum diffusion as a starting point of our consideration is that it can be turned into the state-of-the-art block-coordinate ascent method by a small modification, see [1], [35], [36].

**Min-sum diffusion.** To construct a block-coordinate ascent algorithm one has to first partition the coordinates into blocks. To get an efficient algorithm, the minimization with respect to each block should be efficient. Ideally, the minimum should have a closed form or be attained by a finite-step algorithm with linear complexity.

One such possibility is to assume that one block consists of variables $(\phi_{u,v}(s) \colon v \in \mathcal{N}(u), s \in \mathcal{Y}_u)$ "attached" to one node of the graph. For a given edge $uv$ and label $s \in \mathcal{Y}_u$ we will refer to the set of label pairs $\lbrace (s, l),\; l \in \mathcal{Y}_v \rbrace$ as a *pencil*. Each pencil is defined by the triple $(s, u, v)$, where $u \in \mathcal{V}$, $v \in \mathcal{N}(u)$ and $s \in \mathcal{Y}_u$. Pencils $(s, u, v)$, $v \in \mathcal{N}(u)$, will be called *associated* with the label $s$ in node $u$.

**The elementary update step** of the diffusion algorithm consists of the following two operations,

$$\forall v \in \mathcal{N}(u)\; \forall s \in \mathcal{Y}_u \quad \phi_{u,v}^{t+1}(s) := \phi_{u,v}^t(s) - \min_{l \in \mathcal{Y}_v} \theta_{uv}^{\phi^t}(s, l) \,,$$

$$\forall v \in \mathcal{N}(u)\; \forall s \in \mathcal{Y}_u \quad \phi_{u,v}^{t+2}(s) := \phi_{u,v}^{t+1}(s) + \frac{\theta_u^{\phi^{t+1}}(s)}{\lvert \mathcal{N}(u) \rvert} \,,$$

and is illustrated in Figure 6.8. Loosely speaking, executing one elementary update step for a node $u$ redistributes the minimum costs evenly between all pencils of a given label while moving all costs away from the node itself.

For fixed $s$ and $v$ the first operation (6.33) computes the minimal cost in the pencil $(s, u, v)$, subtracts it from the costs of all label pairs of the pencil and adds it to the cost of the label $s$. As a result, the minimal cost in each pencil associated with the label $s$ becomes zero.

The second operation then redistributes the reparametrized unary costs $\theta_u^\phi(s)$ now corresponding to the label $s$ equally between all pencils $(s, u, v)$, $v \in \mathcal{N}(u)$. As a result, the unary cost of the label $s$ becomes zero and the minimal pairwise costs in all pencils associated with $s$ become equal to each other.

In other words, after operations (6.33) and (6.34) have been executed for node $u \in \mathcal{V}$ the following holds:

$$\theta_u^\phi(s) = 0, \quad \forall s \in \mathcal{Y}_u \,,$$

$$\min_{t \in \mathcal{Y}_v} \theta_{uv}^\phi(s,t) = \min_{t \in \mathcal{Y}_{v'}} \theta_{uv'}^\phi(s,t), \quad \forall v, v' \in \mathcal{N}(u),\; s \in \mathcal{Y}_u \,.$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.16</span></p>

Operations (6.33)-(6.34) maximize the Lagrange dual $\mathcal{D}$ w.r.t. the block of variables $(\phi_{u,v}(s) \colon v \in \mathcal{N}(u), s \in \mathcal{Y}_u)$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Proposition 6.16</summary>

By virtue of Lemma 6.11 it is sufficient to prove the existence of a zero subgradient in a subdifferential of $\mathcal{D}$ when the latter is treated as a function of only the considered block of variables.

We will search for a zero subgradient in the form (6.31). Since only the costs in the node $u$ and its incident edges $uv$, $v \in \mathcal{N}(u)$, are dependent on the considered variable block, it is sufficient to show that

$$\exists y_u \in \mathcal{Y}_u\; \forall v \in \mathcal{N}(u)\; \exists y_v \in \mathcal{Y}_v \colon$$

$$y_u \in \arg \min_{s \in \mathcal{Y}_u} \theta_u^\phi(s) \text{ and } (y_u, y_v) \in \arg \min_{(s,l) \in \mathcal{Y}_{uv}} \theta_{uv}^\phi(s, l) \,.$$

Note that $y_u \in \arg \min_{s \in \mathcal{Y}_u} \theta_u^\phi(s)$ holds for any $y_u \in \mathcal{Y}_u$ since all labels in node $u$ have the same cost after the elementary diffusion step due to (6.35), namely cost 0. Let us, therefore, assign $y_u$ such that $(y_u, y_{v'})$ is a minimizer of $\theta_{uv'}^\phi$ for some $v' \in \mathcal{N}(u)$. Condition (6.36) implies that it then also minimizes $\theta_{uv}^\phi$ for all $v \in \mathcal{N}(u)$. This finalizes the proof. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 6.17</span></p>

If the block-optimality condition (6.37) holds, the elementary step (6.33)-(6.34) of the diffusion algorithm applied to node $u$ does not change the dual value, since the dual vector is already optimal with respect to the considered block of variables.

</div>

**The min-sum diffusion algorithm** consists in a cyclic repetition of the operations (6.33)-(6.34) for all $u \in \mathcal{V}$, which corresponds to the cyclic block-coordinate ascent method (6.15), up to substitution of the convex function $f$ by the concave function $\mathcal{D}$ and minimization w.r.t. each coordinate block by maximization.

Since $\mathcal{D}$ is neither differentiable nor does it have a unique optimum, the min-sum diffusion is not guaranteed to optimize the dual $\mathcal{D}(\boldsymbol{\phi})$. Its convergence properties are analyzed below.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.18</span></p>

Let $\boldsymbol{\theta}^\phi \in \mathbb{R}^{\mathcal{I}}$ satisfy node-edge agreement. Then $F(\boldsymbol{\theta}^\phi)$ does so as well. Moreover, in this case the transformation $F$ does not change the dual value, i.e. $\mathcal{D}(\boldsymbol{\phi}') = \mathcal{D}(\boldsymbol{\phi})$, where $\boldsymbol{\theta}^{\phi'} = F(\boldsymbol{\theta}^\phi)$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 6.18</summary>

For the first part we refer to [1, Ch.8].

To prove the second one, note that according to the definition, node-edge agreement implies the block-optimality condition (6.37). Therefore, as noted in Remark 6.17, $\mathcal{D}(\phi') = \mathcal{D}(\phi)$. $\square$

</details>
</div>

An inverse statement falls even stronger:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.19</span></p>

Let $\boldsymbol{\theta}^\phi$ be a fix-point of the diffusion algorithm. Then $\boldsymbol{\theta}^\phi$ in cycle consistent.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Theorem 6.19</summary>

Consider a binary vector $\boldsymbol{\xi} \in \mathbb{R}^{\mathcal{I}}$ such that $\xi_{uv}(s, t) = \xi_u(s) = \xi_v(t) := 1$ if $(s, t)$ is locally optimal label pair for the edge $uv \in \mathcal{E}$. Assign other coordinates of $\boldsymbol{\xi}$ to zero. The first condition of Definition 5.25 holds by construction. The second condition holds due to (6.36) similarly as in proof of Proposition 6.16. $\square$

</details>
</div>

Theorem 6.18 guarantees that as soon as at some iteration of the min-sun diffusion algorithm the node-edge agreement holds, it holds for further iterations as well. In general, it is known that the max-sum diffusion converges to the node-edge agreement [37], [38].

**Rounding for min-sum diffusion.** To obtain an integer labeling one has to apply some rounding method. Unfortunately, the naïve dual rounding, consisting in selection of a locally optimal label in each node, leads to very poor results when used with the min-sum diffusion, since it takes into account only the reparametrized unary costs and those are all equal to zero in the diffusion algorithm. One way to deal with that is to remember the locally optimal label after each "collection step" (6.33) of the diffusion algorithm, when the pairwise costs are shifted to the unary costs.

Another one, which typically gives similar result when the algorithm converged, is to "equally" split the pairwise costs between corresponding unary factors, i.e. to perform

$$\forall v \in \mathcal{N}(u)\; \forall s \in \mathcal{Y}_u \quad \phi_{u,v}(s) := \phi_{u,v}(s) - \frac{1}{2} \min_{l \in \mathcal{Y}_v} \theta_{uv}^\phi(s, l) \,,$$

for all nodes $u \in \mathcal{V}$. Afterwards the naïve dual rounding can be applied.

### 6.6 Bibliography and Further Reading

The ICM algorithm [39] is among the first methods addressing the labeling problem. There are several publications suggesting different variants of the block-ICM method with tree-structured subproblems. Among the recent ones are [40] and [41], [42], where its parallelization properties were used to achieve faster convergence.

Another efficient block-ICM scheme for optimizing over small, but arbitrary structured subproblems is proposed in [43] and known as *Lazy Flipper*.

The dual subgradient algorithm for the labeling problem in the described form was proposed in [44]. Independently, the subgradient method was evaluated for a slightly different (although equivalent) dual problem in [45]–[47].

Although the min-sum diffusion algorithm is known at least since the 70's, its detailed convergence analysis was given only recently in [37] and later in. The very recent work [38] has improved the analysis and provided a convergence rate of the algorithm. The anisotropic diffusion was introduced in the work [35], along with a unified analysis and evaluation of different dual coordinate ascent algorithms including the CMP (Convex Message Passing) algorithm [48] and MPLP (Message Passing for Linear Programming) [49]. The latter was recently significantly improved in [50].

A min-sum diffusion-like algorithm for general convex piecewise linear functions was suggested in [51]. A taxonomy of dual block-coordinate ascent methods for the labeling problem has been proposed in [52]. A unified analysis of the convergence properties of block-coordinate descent algorithms for non-smooth convex functions was recently given in [53].

There is an alternative method that guarantees the monotonic improvement of the dual objective as well as the convergence to the node-edge agreement. Its representative is the *Augmenting DAG (directed acyclic graph)* algorithm, proposed by [15] and recently reviewed in [18]. A generalization of this algorithm to higher order models was suggested in [54]. This algorithm can be seen as an instance of the combinatorial primal-dual method [55]. The latter covers in particular such famous techniques as the algorithm of Ford-Fulkerson for max-flow and the Hungarian method for the linear assignment problem.

### 6.7 Smoothing Technique

We need differentiation rule of $\max_{i \in I} f_i(x)$ for any $I$, not necessarily finite.

#### 6.7.1 Entropy-Based Smoothing

Consider an LP relaxation of some ILP problem

$$\min_{\substack{\mathbf{x} \in [0,1]^n \\ A\mathbf{x} = \mathbf{b}}} \langle \mathbf{c}, \mathbf{x} \rangle \,.$$

To keep our further exposition simple, we will need the following

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Assumption 6.20</span></p>

Conditions $A\mathbf{x} = \mathbf{b}$ and $\mathbf{x} \geq 0$ imply $\mathbf{x} \in [0, 1]^n$.

</div>

Assumption 6.20 holds, for instance, for the local polytope relaxation of the labeling problem (??) and for the slack-based LP relaxation of the max-weight independent set problem (see Theoretical Exercise Sheet 2).

Consider the primal-dual pair

$$\min_{\substack{\mathbf{x} \in [0,1]^n \\ A\mathbf{x} = \mathbf{b}}} \langle \mathbf{c}, \mathbf{x} \rangle = \min_{\substack{\mathbf{x} \geq 0 \\ A\mathbf{x} = \mathbf{b}}} \langle \mathbf{c}, \mathbf{x} \rangle = \max_{\boldsymbol{\lambda}} \min_{\mathbf{x} \geq 0} \langle \mathbf{c}, \mathbf{x} \rangle + \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle$$

$$= \max_{\boldsymbol{\lambda}} \underbrace{-\langle \boldsymbol{\lambda}, \mathbf{b} \rangle + \min_{\mathbf{x} \geq 0} \langle \overbrace{\mathbf{c} + A^\top \boldsymbol{\lambda}}^{\mathbf{c}^{\boldsymbol{\lambda}}}, \mathbf{x} \rangle}_{g(\boldsymbol{\lambda})} \,.$$

As shown in (??) the dual objective $g(\boldsymbol{\lambda})$ is convex, but non-differentiable. This significantly restricts the set of methods that can be used for its optimization. In this section we consider one technique to approximate $g(\boldsymbol{\lambda})$ with a parametrized set of convex smooth functions. Under proper selection of the *smoothing parameter* defining the accuracy of approximation one can attain a required optimization accuracy for $g(\boldsymbol{\lambda})$ by optimizing its smooth approximation.

To this end consider $H(\mathbf{x}) := -(\sum_{i=1}^{n} x_i \log x_i - x_i)$ — the entropy function "shifted" by the vector $\mathbf{x}$. This function is concave, hence the function

$$g_T(\boldsymbol{\lambda}) = \min_{\substack{\mathbf{x} \geq 0 \\ A\mathbf{x} = \mathbf{b}}} \langle \mathbf{c}, \mathbf{x} \rangle + \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle - TH(\mathbf{x})$$

$$= -\langle \boldsymbol{\lambda}, \mathbf{b} \rangle + \min_{\mathbf{x} \geq 0} \langle \mathbf{c}^{\boldsymbol{\lambda}}, \mathbf{x} \rangle - TH(\mathbf{x})$$

$$= -\langle \boldsymbol{\lambda}, \mathbf{b} \rangle + \sum_{i=1}^{n} \underbrace{\min_{x_i \geq 0} c_i^{\boldsymbol{\lambda}} x_i + T(x_i \log x_i - x_i)}_{l_i(x_i)}$$

is convex. Consider the unconstrained optimality condition for the function $l_i$:

$$0 = \frac{\partial l}{\partial x_i} = c_i^{\boldsymbol{\lambda}} + T(\log x_i + x_i \frac{1}{x_i} - 1) = c_i^{\boldsymbol{\lambda}} + T \log x_i \,.$$

This implies $x_i[\boldsymbol{\lambda}] = e^{-c_i^{\boldsymbol{\lambda}}/T}$ and since $x_i[\boldsymbol{\lambda}] \geq 0$, it holds $x_i[\boldsymbol{\lambda}] = \arg \min_{x_i \geq 0} l_i(x_i)$.

Consider now

$$g_T(\boldsymbol{\lambda}) = -\langle \boldsymbol{\lambda}, \mathbf{b} \rangle + \langle \mathbf{c}^{\boldsymbol{\lambda}}, \mathbf{x}[\boldsymbol{\lambda}] \rangle - TH(\mathbf{x}[\boldsymbol{\lambda}])$$

$$= -\langle \boldsymbol{\lambda}, \mathbf{b} \rangle + \sum_{i=1}^{n} c_i^{\boldsymbol{\lambda}} e^{-c_i^{\boldsymbol{\lambda}}/T} + T\left(e^{-c_i^{\boldsymbol{\lambda}}/T}(-c_i^{\boldsymbol{\lambda}}/T) - e^{-c_i^{\boldsymbol{\lambda}}/T}\right)$$

$$= -\langle \boldsymbol{\lambda}, \mathbf{b} \rangle - T \sum_{i=1}^{n} e^{-c_i^{\boldsymbol{\lambda}}/T} \,.$$

Hence the function $g_T(\boldsymbol{\lambda})$ is concave and infinitely continuously differentiable.

**Strong duality** holds not only for bounded linear programs, but for a much broader class. Since it plays an important role for the smoothing approach, we give the following theorem here.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.21</span></p>

$$\min_{\substack{\mathbf{x} \geq 0 \\ A\mathbf{x} = \mathbf{b}}} \langle \mathbf{c}, \mathbf{x} \rangle - TH(\mathbf{x}) = \max_{\boldsymbol{\lambda}} \min_{\mathbf{x} \geq 0} \langle \mathbf{c}, \mathbf{x} \rangle + \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle - TH(\mathbf{x})$$

</div>

We refer to e.g., [12] for a proof.

Note that given the strong duality, KKT conditions (??) together with Assumption 6.20 imply that for an optimal dual solution $\boldsymbol{\lambda}^*$ and the optimal primal solution $\mathbf{x}^* = e^{\mathbf{c}^{\boldsymbol{\lambda}^*}/T} \geq 0$ it holds $A\mathbf{x}^* = \mathbf{b}$ and, therefore, $\mathbf{x}^* \in [0, 1]$.

**Smooth approximation bounds.** Note that for any $x \in [0, 1]$ it holds $-1 \leq x \log x - x \leq 0$ and, therefore, $0 \leq -H(\mathbf{x}) \leq n$ for $\mathbf{x} \in [0, 1]^n$. In turn, due to Assumption 6.20 it implies

$$\left( \min_{\substack{\mathbf{x} \geq 0 \\ A\mathbf{x} = \mathbf{b}}} \langle \mathbf{c}, \mathbf{x} \rangle \right) - Tn \leq \min_{\substack{\mathbf{x} \geq 0 \\ A\mathbf{x} = \mathbf{b}}} \langle \mathbf{c}, \mathbf{x} \rangle - TH(\mathbf{x}) \leq \min_{\substack{\mathbf{x} \geq 0 \\ A\mathbf{x} = \mathbf{b}}} \langle \mathbf{c}, \mathbf{x} \rangle \,.$$

The same in terms of the dual function:

$$g(\boldsymbol{\lambda}^*) - Tn \leq g_T(\hat{\boldsymbol{\lambda}}^*) \leq g(\boldsymbol{\lambda}^*) \,,$$

where $\boldsymbol{\lambda}^*$ and $\hat{\boldsymbol{\lambda}}^*$ are maximizers of the original (??) and smoothed (??) dual problems.

Moreover, even stronger conditions hold, if the used optimization algorithm maintains $\mathbf{x}[\boldsymbol{\lambda}] \in [0, 1]^n$ on all its iterations. In this case

$$g(\boldsymbol{\lambda}) - Tn \leq g_T(\boldsymbol{\lambda}) \leq g(\boldsymbol{\lambda}) \,.$$

All these conditions substantiate the intuitive assumption that lower values of the *smoothing parameter* $T$ (sometimes called *temperature*) lead to a better approximation of the original function.

#### 6.7.2 Smoothed Dual Coordinate Ascent

Consider a coordinate ascent optimization of the smoothed dual $g_T(\boldsymbol{\lambda})$. Let at iteration $t + 1$ one fixes all dual variables $\lambda_{j'}$, $j' \neq j$ and optimizes on $\lambda_j$ only. Assume that

$$\lambda_{j'}^{t+1} = \begin{cases} \lambda_{j'}^t, & j' \neq j \\ \lambda_j^t + \nu, & j' = j \end{cases}$$

Then we can derive the following update rule for $\mathbf{x}[\boldsymbol{\lambda}^t]$:

$$x_i[\boldsymbol{\lambda}^{t+1}] = e^{-c_i^{\boldsymbol{\lambda}^{t+1}}/T} = e^{-[\mathbf{c} + A^\top \boldsymbol{\lambda}^{t+1}]_i/T} = e^{-[c_i + \sum_{j'=1}^{m} a_{j'i} \lambda_{j'}^{t+1}]/T} = x_i[\boldsymbol{\lambda}^t] e^{-a_{ji}\nu/T} \,.$$

The value of $\nu$ can be found from the optimality condition for $\lambda_j$:

$$\frac{\partial g_T}{\partial \lambda_j} \stackrel{(6.41)}{=} [A\mathbf{x}[\boldsymbol{\lambda}^{t+1}] - \mathbf{b}]_j = 0 \quad \Rightarrow \quad \sum_{i=1}^{n} a_{ji} x_i[\boldsymbol{\lambda}^{t+1}] = b_j$$

By plugging in (6.49) we obtain equation

$$\sum_{i=1}^{n} a_{ji} x_i[\boldsymbol{\lambda}^t] e^{-a_{ji}\nu/T} = b_j$$

that must be solved w.r.t. $\nu$. A unique solution to this equation together with continuous differentiability of $g_T$ would imply that the conditions of Theorem 6.12 hold and any limit point $\boldsymbol{\lambda}^*$ of the sequence $\boldsymbol{\lambda}^t$ is the optimal dual solution. The strong duality in turn implies that $\mathbf{x}[\boldsymbol{\lambda}^*]$ is the optimal primal relaxed solution.

Note that due to one-to-one correspondence between $\boldsymbol{\lambda}$ and $\mathbf{x}[\boldsymbol{\lambda}]$ the algorithm can explicitly update either the former or the latter.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.22</span><span class="math-callout__name">(Smooth coordinate minimization for uniqueness constraints)</span></p>

Let $[A\mathbf{x}[\boldsymbol{\lambda}^{t+1}]]_j = b_j$ has a form of a uniqueness constraint

$$\sum_{i \in K} x_i = 1 \,.$$

Then (6.53) turns into

$$e^{-\nu/T} \sum_{i \in K} x_i[\boldsymbol{\lambda}^t] = 1 \,,$$

which implies

$$e^{-\nu/T} = \frac{1}{\sum_{i \in K} x_i[\boldsymbol{\lambda}^t]}$$

and

$$\nu = -T \log \frac{1}{\sum_{i \in K} x_i[\boldsymbol{\lambda}^t]} = T \log \sum_{i \in K} x_i[\boldsymbol{\lambda}^t] = T \log \sum_{i \in K} e^{-c_i^{\boldsymbol{\lambda}^t}/T}$$

$$= c_{i^*}^{\boldsymbol{\lambda}^t} + T \log \sum_{i \in K} e^{-(c_i^{\boldsymbol{\lambda}^t} - c_{i^*}^{\boldsymbol{\lambda}^t})/T} \,,$$

where $i^* \in \arg \min_{i \in [n]} c_i^{\boldsymbol{\lambda}^t}$. This leads us to the mathematically equivalent primal and dual update rules

$$\lambda_{j'}^{t+1} = \begin{cases} \lambda_{j'}^t, & j' \neq j \\ \lambda_j^t + c_{i^*}^{\boldsymbol{\lambda}^t} + T \log \sum_{i \in K} e^{-(c_i^{\boldsymbol{\lambda}^t} - c_{i^*}^{\boldsymbol{\lambda}^t})/T}, & j' = j \end{cases}$$

$$x_i^{t+1} = \begin{cases} x_i^t, & i \notin K \\ \frac{1}{\sum_{i \in K} x_i^t}, & i \in K \,. \end{cases}$$

</div>

Both rules have their advantages and disadvantages: The primal update rule (6.57) is significantly ($5 - 10$ times) faster than the dual rule (6.56) because the latter contains an extensive usage of the (slow) exponentiation operations.

On the negative side, the primal rule may result in division by 0, usually for low $T$, whereas the dual rule is numerically stable for any $T$ and $\boldsymbol{\lambda}$ due to the fact that $-(c_i^{\boldsymbol{\lambda}^t} - c_{i^*}^{\boldsymbol{\lambda}^t})/T < 0$, hence $e^{-(c_i^{\boldsymbol{\lambda}^t} - c_{i^*}^{\boldsymbol{\lambda}^t})/T} \in [0, 1]$ and $\sum_{i \in K} e^{-(c_i^{\boldsymbol{\lambda}^t} - c_{i^*}^{\boldsymbol{\lambda}^t})/T} \geq 1$ since $e^{-(c_{i^*}^{\boldsymbol{\lambda}^t} - c_{i^*}^{\boldsymbol{\lambda}^t})/T} = 1$.

#### 6.7.3 Use Case: Linear Assignment Problem

Consider the linear assignment problem

$$\min_{\mathbf{x} \in \lbrace 0, 1 \rbrace^{n \times n}} \sum_{i=1}^{n} \sum_{j=1}^{n} c_{ij} x_{ij}$$

$$\text{s.t.} \quad \sum_{j=1}^{n} x_{ij} = 1,\; \forall j \in [n]$$

$$\sum_{i=1}^{n} x_{ij} = 1,\; \forall i \in [n]$$

Its natural relaxation satisfies Assumption 6.20 and is known to be LP-tight, hence we address its entropy-smoothed approximation:

$$\min_{\mathbf{x} \geq 0} \sum_{i=1}^{n} \sum_{j=1}^{n} c_{ij} x_{ij} + T \sum_{i=1}^{n} \sum_{j=1}^{n} (x_{ij} \log x_{ij} - x_{ij})$$

$$\text{s.t.} \quad \sum_{j=1}^{n} x_{ij} = 1,\; \forall j \in [n]$$

$$\sum_{i=1}^{n} x_{ij} = 1,\; \forall i \in [n]$$

Using the primal update rule (6.49) from Example 6.22 we obtain Algorithm 5, which is a famous Sinkhorn algorithm widely used to approximately solve the linear assignment problem in neural networks.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm 5</span><span class="math-callout__name">(Sinkhorn Algorithm for Linear Assignment)</span></p>

1. Initialization: $x_{ij} = \exp(-c_{ij}/T)$, $i, j \in [n]$
2. **while** stopping criterion not fulfilled **do**
3. &emsp; Normalize all rows:
4. &emsp; **for** $i = 1$ **to** $n$ **do**
5. &emsp;&emsp; $s := \sum_{j=1}^{n} x_{ij}$
6. &emsp;&emsp; $x_{ij} := \frac{x_{ij}}{s}$, $\forall j \in [n]$
7. &emsp; **end for**
8. &emsp; Normalize all columns:
9. &emsp; **for** $j = 1$ **to** $n$ **do**
10. &emsp;&emsp; $s := \sum_{i=1}^{n} x_{ij}$
11. &emsp;&emsp; $x_{ij} := \frac{x_{ij}}{s}$, $\forall i \in [n]$
12. &emsp; **end for**
13. **end while**
14. **return** $\mathbf{x}$

</div>

#### 6.7.4 Use Case: Max-Weight Independent Set Problem

## Chapter 7: Overview: How Off-the-Shelf ILP Solvers Work

### 7.1 Systematic Search: Branch-and-Bound

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 7.1</span><span class="math-callout__name">(Binary knapsack problem)</span></p>

Consider the binary knapsack problem and the respective branching tree in Figure 7.1. A general algorithm for a minimization (maximization) branch-and-bound approach looks as follows:

1. Solve the **basis LP relaxation**; Initialize a feasible integer solution $S = \infty(-\infty)$
2. **Branch:**
   - a) **Select a** non-terminated **leaf**
   - b) ...and **branch** on a fractional variable.
   - c) Solve the respective LP relaxation.
3. **Evaluate** the results:
   - a) Is the obtained lower (upper) **bound** higher (lower) than $S$? If yes — **terminate** the branch (no branching anymore on this leaf) and go to Item 2. This includes also the case when the respective problem is infeasible and the bound is equal to $\infty$ ($-\infty$).
   - b) Is solution **integral**? Yes — update $S$ and terminate the branch.
   - c) Go to Item 2 unless all branches are terminated.
4. **Output** $S$.

</div>

Remarks to existing implementations of the branch-and-bound methods:

- Points in which different branching methods may differ are:
  - Selection of the (fractional) variable to branch on. E.g., for the variable selection one may "select one with its value close to 0.5 that has a large coefficient in the objective function".
  - Selection of the leaf for branching. The depth-first and breadth-first are two of many possible strategies.

  For the practically used heuristics for both items above and their empirical evaluation see, e.g., [56].

- In general the branching tree grows exponentially large and, therefore, good bounds and primal heuristics are vitally important for the practical efficiency of the method.

- As follows from Item 3a, better primal solutions may improve performance of the algorithm since they allow to terminate non-productive branches earlier. Typically primal heuristics are used to obtain better primal solutions as fast as possible, see e.g., [56] and Chapter 8.

- The algorithm for solving the underlying LP relaxation must allow efficient fixing of variables and "warm-start" from the relaxation of the upper level, see Section 7.4.

### 7.2 Cutting Plane Method: Tightening the Bound

An orthogonal way to address ILP problems is to tighten the feasible set and, as a result, the relaxation it defines. Let $P$ be the integer hull of the ILP of interest.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 7.2</span><span class="math-callout__name">(Separation Problem)</span></p>

Let $P$ be a convex polytope in $\mathbb{R}^n$ and $\mathbf{y} \in \mathbb{R}^n$. The *separation problem* consists in answering the question whether $\mathbf{y} \in P$. Should it be that $\mathbf{y} \notin P$ a *separation inequality* $\langle \mathbf{a}, \mathbf{x} \rangle \leq b$ has to be found such that $\langle \mathbf{a}, \mathbf{y} \rangle > b$ and at the same time $\langle \mathbf{a}, \mathbf{x} \rangle \leq b$ for all $\mathbf{x} \in P$.

</div>

The role of the convex polytope in Definition 7.2 is usually played by the integer hull of the problem, the point $\mathbf{y}$ is a solution of the current LP relaxation, the separation inequality is the cutting plane that tightens the relaxation. Intuitively "the best tightening" is achieved if the resulting inequality is *facet-defining*, as these "can not be pushed further" into the feasible set without cutting off at least one feasible integer point.

The following *cutting plane* algorithm can be used to address the general-form ILP $\min_{\mathbf{x} \in P^0 \cap \lbrace 0,1 \rbrace^n} \langle \mathbf{c}, \mathbf{x} \rangle$:

1. Set $t = 0$.
2. **Solve** the relaxed problem $\mathbf{y} = \arg \min_{\mathbf{x} \in P^t} \langle \mathbf{c}, \mathbf{x} \rangle$.
3. If $\mathbf{y} \notin \text{conv}(P^0 \cap \lbrace 0,1 \rbrace^n)$
   - a) Find a **separation inequality** $p^t := (\langle \mathbf{a}^t, \mathbf{x} \rangle \leq b^t)$,
   - b) Add it to the current feasible set: $P^{t+1} := P^t \cup \lbrace p^t \rbrace$.
   - c) Set $t := t + 1$ and go to Item 2.
4. **Return** $\mathbf{y}$.

Checking $\mathbf{y} \notin \text{conv}(P^0 \cap \lbrace 0,1 \rbrace^n)$ in Item 3 can be difficult in practice, so one substitutes it with $\mathbf{y} \notin \lbrace 0,1 \rbrace^n$ by requiring $\mathbf{y} \in \text{vrtx}(P)$ in Item 2.

**Theoretical power of the cutting plane method** is well-described by the following theorem:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 7.3</span></p>

Let $P = \lbrace \mathbf{x} \in \mathbb{R}^n \colon A\mathbf{x} \leq \mathbf{b} \rbrace$ be a polyhedron such that the length of encoding of each inequality in $A\mathbf{x} \leq \mathbf{b}$ is at most $l$. Then the optimization problem $\min_{\mathbf{x} \in P} \langle \mathbf{c}, \mathbf{x} \rangle$ can be solved in polynomial time (in $l$ and coding length of $c$) if and only if the separation problem with respect to $P$ can be solved in polynomial time (in $l$ and coding length of $\mathbf{y}$) for any $\mathbf{y} \in \mathbb{R}^n$.

</div>

In other words, given that a polynomial separation oracle exists, $\min_{\mathbf{x} \in P} \langle \mathbf{c}, \mathbf{x} \rangle$ is solvable in polynomial time even if the number of inequalities defining $P$ is exponential in $n$. Important is, however, that each inequality contains only polynomial in $n$.

On the other side, Theorem 7.3 implies that NP-hard ILP problems cannot be solved with the cutting plane method alone in polynomial time. Although there is a general polynomial method, known as Gomory-cuts **Gomory-cuts**, for generating separation cuts for any ILP, the number of cuts required to solve an ILP problem grows exponentially with $n$ in general. In practice it means that the found cuts lead to smaller and smaller improvements, as the algorithm progresses.

A much more practically efficient way to explore the cutting plane idea for ILPs is the branch-and-cut method described below.

#### 7.2.1 Branch-and-Cut Method

The branch-and-cut method assumes that there is a (usually exponentially large) set of potential additional inequalities (cutting planes) that can be efficiently computed to separate any relaxed solution. Should the relaxed solution satisfy all these inequalities or their computation becomes too expensive with respect to the attained improvement of the LP bound, one resorts to branching:

1. Solve the **basis LP relaxation**; Initialize a feasible integer solution, let $S = \infty(-\infty)$ be its total cost.
2. **Branch:**
   - a) **Select a** non-terminated **leaf**
   - b) ...and **branch** on a fractional variable.
   - c) Solve the respective LP relaxation **without additional separation inequalities**.
3. **Evaluate** the results:
   - a) Is the obtained lower (upper) **bound** higher (lower) than $S$? If yes — **terminate** the branch (no branching anymore on this leaf) and go to Item 2. This includes also the case when the respective problem is infeasible and the bound is equal to $\infty$ ($-\infty$).
   - b) Is solution **integral**? Yes — update $S$ and terminate the branch.
   - c) Can an additional **separation inequality** be found? If yes — add it to the feasible set and go to Item 3a. Otherwise:
   - d) Go to Item 2 unless all branches are terminated.
4. **Output** $S$.

As can be seen, the algorithm differs from the original branch-and-bound only by Item 3c, which iteratively tightens the underlying relaxation as along as it can be done with a given class of considered separation inequalities. Although a separation inequality always exists (unless the found solution is integral), finding them may get computationally expensive or the resulting tightening of the feasible set only gets very small. This is essentially the result of Theorem 7.3. Therefore, one restricts the separation routine to the specific class(es) of inequalities that can be efficiently checked. When none of the inequalities is violated, one resorts to branching. Other, usually heuristic, strategies to keep the balance between branching and cutting are used as well. For example, one may restrict the number of cutting planes that can be added to the initial LP in each leaf. To speed-up search for the violated inequalities one often stores those that have been found so far in a separate table and checks them first in the separation step.

#### 7.2.2 Two "Off-the-Shelf" Constraint Classes for $0/1$ ILPs

A number of separation inequality classes are used by standard ILP solvers. For instance, according to [57] the SCIP open source mixed integer programming solver "features separators for SCIP features separators for knapsack cover cuts [59], complemented mixed integer rounding cuts [60], Gomory mixed integer cuts [61], strong Chvátal-Gomory cuts [62], flow cover cuts [63], implied bound cuts [64], and clique cuts [64], [65]".

We review here two classes that address $0/1$ integer programs: Knapsack cover cuts [59] and clique cuts [64], [65].

**Knapsack cover cuts.** In the practice of off-the-shelf solvers knapsack cuts are generated to tighten each single (binary) *knapsack inequality* $\langle \mathbf{a}, \mathbf{x} \rangle \leq b$ where $\mathbf{a} \geq 0$, $b \geq 0$ and $\mathbf{x} \in \lbrace 0, 1 \rbrace^N$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 7.4</span><span class="math-callout__name">(Cover)</span></p>

Subset of variable indices $C \subset N$ is called *cover*, if $\sum_{i \in C} a_i > b$. The cover is *minimal*, if $\sum_{i \in C \setminus \lbrace k \rbrace} a_i \leq b$ for all $k \in C$.

</div>

Observe that the *cover inequality* defined as

$$\sum_{i \in C} x_i \leq \lvert C \rvert - 1 \,,$$

is valid for the knapsack polytope $P^K := \text{conv}(X^K)$ with $X^K := \lbrace \mathbf{x} \in \lbrace 0,1 \rbrace^N \colon \langle \mathbf{a}, \mathbf{x} \rangle \leq b \rbrace$. It is known that if $C$ is minimal cover, this constraint is facet-defining for $P^{K'} := \text{conv}(X^K \cap \lbrace \mathbf{x} \in \lbrace 0,1 \rbrace^N \colon x_i = 0\; \forall i \in N \setminus C \rbrace)$.

To separate on cover inequalities rewrite (7.1) as

$$\sum_{i \in C} (1 - x_i) \geq 1 \,.$$

Then an LP solution $x^*$ satisfies all the cover inequalities with respect to the given knapsack if and only if

$$\sum_{i \in C} (1 - x_i^*) \geq 1$$

for any cover $C \subset N$. This can be equivalently rewritten as

$$\min_{\mathbf{z} \in \lbrace 0,1 \rbrace^N} \left\lbrace \sum_{i \in N} (1 - x_i^*) z_i \colon \text{s.t.} \sum_{i \in N} a_i z_i > b \right\rbrace \geq 1 \,.$$

Here one minimizes over all possible covers $C$ defined by the binary vectors $\mathbf{z}$. Note that (7.4) is a knapsack problem itself and is usually solved exactly by dynamic programming (for integer $\mathbf{a}$, $b$ and small $N$) or approximately by a greedy method (for large $N$), see Section 8.2.

In practice one uses tighter *lifted* cover inequalities that may correspond to the facets of $P^K$ and not only of $P^{K'}$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 7.5</span><span class="math-callout__name">(Minimization binary knapsack problem)</span></p>

The minimization binary knapsack problem

$$\min_{\mathbf{x} \in \lbrace 0,1 \rbrace^n} \langle \mathbf{c}, \mathbf{x} \rangle$$

$$\text{s.t.} \quad \langle \mathbf{a}, \mathbf{x} \rangle \geq b$$

can be formulated as the (maximization) binary knapsack problem

$$\max_{\mathbf{x} \in \lbrace 0,1 \rbrace^n} \langle \mathbf{c}, \mathbf{x} \rangle$$

$$\text{s.t.} \quad \langle \mathbf{a}, \mathbf{x} \rangle \leq \sum_{i=1}^{n} a_i - b$$

by maximizing the profit of the items *not contained* in the knapsack under condition that their total weight is at most $\sum_{i=1}^{n} a_i - b$. Consequently, if $\mathbf{x}^*$ is the solution of the minimization knapsack, then $\mathbf{1} - \mathbf{x}^*$ is the solution of the maximization knapsack.

</div>

**Clique cuts.** We say that two variables $x_i$ and $x_j$ are *conflicting*, if $x_i + x_j \leq 1$. Conflicting variables can be found during presolving with the *probing* procedure, see Section 7.3.2. All pairs of the conflicting variables together are stored usually in a form of a *conflict graph* $(\mathcal{V}, \mathcal{E})$. Convex hull of all binary vectors satisfying all conflicts form the *stable set polytope*: $\text{conv}\lbrace \mathbf{x} \in [0, 1]^\mathcal{V} \colon x_i + x_j \leq 1,\; \lbrace i, j \rbrace \in \mathcal{E} \rbrace$. The most practical class of facet-defining constraints for this polytope are *clique inequalities*

$$\sum_{i \in C} x_i \leq 1 \,,$$

where $C$ is a maximal clique in $(\mathcal{V}, \mathcal{E})$.

For the separation of violated clique inequalities one uses the LP values of the variables as weights for the nodes $\mathcal{V}$ and searches for the maximum-weight clique or at least a clique with the weight greater than 1. Since the *maximum weighted clique* problem is NP-hard, one has to resort to heuristics in order to efficiently separate the clique cuts. Note that one cannot just add all maximal cliques as inequalities to the input problem, as the number of maximal cliques grows exponentially with the graph size.

### 7.3 (I)LP Presolve: Domain Propagation and Probing

Domain propagation and probing are used as a preprocessing, before branching as well as on different levels of the branching tree.

#### 7.3.1 Domain Propagation

Consider the convex polyhedron $P := \lbrace \mathbf{x} \in \mathbb{R}^m \colon A\mathbf{x} \leq \mathbf{b} \rbrace$. The goal of *domain propagation* or *bound strengthening* is to find $\mathbf{b}' \leq \mathbf{b}$ such that $P = \lbrace \mathbf{x} \in \mathbb{R}^m \colon A\mathbf{x} \leq \mathbf{b}' \rbrace$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 7.6</span><span class="math-callout__name">(Variable's upper bound)</span></p>

Consider

$$P := \begin{cases} 2x_1 + x_2 \leq 3 \\ 0 \leq x_1 \leq 3 \\ 0 \leq x_2 \end{cases}$$

Observe that to fulfill the first inequality $x_1$ should not exceed 1.5. Hence, one can rewrite $P$ as

$$P = \begin{cases} 2x_1 + x_2 \leq 3 \\ 0 \leq x_1 \leq 1.5 \\ 0 \leq x_2 \end{cases}$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 7.7</span><span class="math-callout__name">(Constraint's upper bound)</span></p>

Consider

$$P := \begin{cases} x_1 + x_2 \leq 10 \\ 0 \leq x_1 \leq 2 \\ 0 \leq x_2 \leq 1 \end{cases}$$

Observe that due to bounds on $x_1$ and $x_2$ the first inequality can be tightened and one can rewrite $P$ as

$$P = \begin{cases} x_1 + x_2 \leq 3 \\ 0 \leq x_1 \leq 2 \\ 0 \leq x_2 \leq 1 \end{cases}$$

Moreover, the first constraint can be completely eliminated as it does not pose any additional restrictions:

$$P = \begin{cases} 0 \leq x_1 \leq 2 \\ 0 \leq x_2 \leq 1 \end{cases}$$

</div>

The goal of domain propagation is

- Fixing some variables: if their upper and lower bounds coincide. In case of an integer variable $l \leq x \leq u$, $x \in \mathbb{Z}$, it is sufficient that $u - l \leq 1$ to conclude $x = \lfloor u \rfloor$ if $\lfloor u \rfloor = \lceil l \rceil$ or the problem is infeasible if $\lfloor u \rfloor < \lceil l \rceil$.
- Detect infeasibility if for $l \leq x \leq u$ it turns that $u \leq l$ after domain propagation.
- Eliminating constraints as in Example 7.7.

Domain propagation procedures can be performed in all nodes of the branch-and-bound tree.

Below we consider the formalization of the domain propagation. Let $\mathbf{x} \in \mathbb{R}^n$ and $A := (\mathbf{a_j})_{j \in [m]}$. Consider the constraint set

$$A\mathbf{x} \leq \mathbf{b}$$

$$\mathbf{l} \leq \mathbf{x} \leq \mathbf{u}$$

Denote $A^j \mathbf{x} \leq \mathbf{b}^j$ the set of constraints obtained from the initial one by deleting the $j$-th row $\langle \mathbf{a_j}, \mathbf{x} \rangle \leq b_j$. For the sake of notation we will assume that

$$\langle \mathbf{a_j}, \mathbf{x} \rangle = \sum_{i \in C^+} a_{ji} x_i - \sum_{i \in C^-} a_{ji} x_i \,,$$

where $C^+$ and $C^-$ denote variables with positive and negative coefficients respectively (variables with zero coefficients can be ignored) and all $a_{ji} > 0$ therefore.

**Identification of infeasibility.** Consider the following LP:

$$z = \min \sum_{i \in C^+} a_{ji} x_i - \sum_{i \in C^-} a_{ji} x_i$$

$$\text{s.t.} \quad A^j \mathbf{x} \leq \mathbf{b}^j$$

$$\mathbf{l} \leq \mathbf{x} \leq \mathbf{u}$$

Observe that if $z > b_j$, the feasible region is empty. However, the above problem is nearly as difficult as the initial LP, hence for the sake of efficiency we consider its relaxation that gives us a lower bound $z_\text{LB}$ for $z$: $z \geq z_\text{LB} > b_j$. The simplest (and the weakest) lower bound is obtained by completely discarding the system $A^j \mathbf{x} \leq \mathbf{b}^j$. In that case we can conclude that the system is infeasible if

$$z_\text{LB} = \sum_{i \in C^+} a_{ji} l_i - \sum_{i \in C^-} a_{ji} u_i > b_j$$

**Identification of redundancy.** Consider the following LP:

$$z = \max \sum_{i \in C^+} a_{ji} x_i - \sum_{i \in C^-} a_{ji} x_i$$

$$\text{s.t.} \quad A^j \mathbf{x} \leq \mathbf{b}^j$$

$$\mathbf{l} \leq \mathbf{x} \leq \mathbf{u}$$

If $z \leq b_j$, the inequality $\langle \mathbf{a_j}, \mathbf{x} \rangle \leq b_j$ is redundant. Similarly to the previous case, by discarding $A^j \mathbf{x} \leq \mathbf{b}^j$, we obtain an upper bound $z_\text{UB}$ and the sufficient condition $z \leq z_\text{UB} \leq b_j$. In that case we can conclude that the inequality $\langle \mathbf{a_j}, \mathbf{x} \rangle \leq b_j$ is redundant if

$$z_\text{UB} = \sum_{i \in C^+} a_{ji} u_i - \sum_{i \in C^-} a_{ji} l_i \leq b_j \,.$$

**Improving an upper bound.** Consider the variable $x_k$ and the following LP:

$$z = \min \sum_{i \in C^+ \setminus \lbrace k \rbrace} a_{ji} x_i - \sum_{i \in C^-} a_{ji} x_i$$

$$\text{s.t.} \quad A^j \mathbf{x} \leq \mathbf{b}^j$$

$$\mathbf{l} \leq \mathbf{x} \leq \mathbf{u}$$

Observe that $z_k + a_{jk} x_k \leq b_j$ and, therefore, $x_k \leq (b_j - z_k)/a_{jk}$. Hence the upper bound $u_k$ can be improved by setting to the latter value if $(b_j - z_k)/a_{jk} < u_k$. Again, by discarding $A^j \mathbf{x} \leq \mathbf{b}^j$ we can conclude that $u_k$ can be improved if

$$\left( b_j - \sum_{i \in C^+ \setminus \lbrace k \rbrace} a_{ji} l_i + \sum_{i \in C^-} a_{ji} u_i \right) / a_{jk} < u_k \,.$$

**Improving a lower bound.** Consider the variable $x_k$ and the following LP:

$$z = \min \sum_{i \in C^+} a_{ji} x_i - \sum_{i \in C^- \setminus \lbrace k \rbrace} a_{ji} x_i$$

$$\text{s.t.} \quad A^j \mathbf{x} \leq \mathbf{b}^j$$

$$\mathbf{l} \leq \mathbf{x} \leq \mathbf{u}$$

Observe that $z_k - a_{jk} x_k \leq b_j$ and, therefore, $x_k \geq (z_k - b_j)/a_{jk}$. Hence the lower bound $l_k$ can be improved by setting to the latter value if $(z_k - b_j)/a_{jk} > l_k$. Again, by discarding $A^j \mathbf{x} \leq \mathbf{b}^j$ we can conclude that $l_k$ can be improved if

$$\left( \sum_{i \in C^+} a_{ji} l_i - \sum_{i \in C^- \setminus \lbrace k \rbrace} a_{ji} u_i - b_j \right) / a_{jk} > l_k \,.$$

**Iterative domain propagation.** Domain propagation can be run iteratively over all constraints until no significant change is happening. The following example shows that the iterative improvement can be infinite:

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 7.8</span></p>

For $0 < a < 1$ consider the constraints

$$x_1 - ax_2 \leq 0$$

$$ax_1 - x_2 \geq 0$$

$$0 \leq x_1, x_2 \leq 1$$

Improving the bounds of $x_1$ using the first constraint yields $x_1 \leq ax_2 \leq a$. Exploiting this new upper bound on $x_1$, the second constraint implies $x_2 \leq ax_1 \leq a^2$. If we iterate and process each constraint $t$ times, we reach $x_1 \leq a^{2t-1}$ and $x_2 \leq a^{2t}$. Thus we find an infinite sequence of bound reductions.

</div>

Therefore one must use an improvement threshold to stop domain propagation.

#### 7.3.2 Probing

Probing is the technique mostly useful with binary integer variables. Probing means temporarily setting a variable $x_k \in \lbrace 0, 1 \rbrace$ to 0 (or 1) and then performing the domain propagation. Possible outcomes of such operation are:

- *The problem becomes infeasible*: We can fix the value of $x_k$ to 0 (1 resp.). For example, $5x_1 + 3x_2 \geq 4$ becomes infeasible if $x_1$ is set to 0. We conclude that $x_1 = 1$.
- *Some constraint becomes redundant*: It can be tightened by *coefficient reduction* or *improvement*. For example, $2x_1 + x_2 \geq 1$ becomes strictly redundant when $x_1$ is set to 1. Therefore, the constraint can be tightened to $x_1 + x_2 \geq 1$. Note that $(0.5, 0)$ is no longer feasible to the tightened constraint, whereas all feasible integer solutions are preserved.
- *Some other binary variable $x_j \in \lbrace 0, 1 \rbrace$ gets fixed to one of its values.* An additional linear constraint can be added that tightens the relaxation. In particular, the conflict graph, see Section 7.2.2, can be constructed. As example consider $5x_1 + 4x_2 \leq 8$. If $x_1$ is set to 1, $x_2$ must be set to 0. This implies $x_1 + x_2 \leq 1$. Adding this inequality tightens the LP relaxation, as $(1, 0.75)$ is feasible for the original inequality, but is infeasible for the implication inequality.

Consider now the system

$$A\mathbf{x} \leq \mathbf{b}$$

$$\mathbf{x} \in \lbrace 0, 1 \rbrace^n$$

i.e., comparing to the problem treated in Section 7.3.1, the variable $\mathbf{x}$ is now binary integer instead of being continuous. Otherwise we preserve notation introduced in Section 7.3.1.

**Fixing variables.**

1. Consider $x_k$, $k \in C^+$, $j \in [m]$, and the following ILP:

$$z_k = \min \sum_{i \in C^+} a_{ji} x_i - \sum_{i \in C^-} a_{ji} x_i$$

$$\text{s.t.} \quad A^j \mathbf{x} \leq \mathbf{b}^j$$

$$\mathbf{x} \in \lbrace 0,1 \rbrace^n$$

$$x_k = 1$$

If $z_k > b_j$, the set described by constraints (7.37) and additionally $x_k = 1$ is empty. It implies that $x_k \neq 1$ in any feasible solution of (7.37) and can be set to 0. When we discard the system $A^j \mathbf{x} \leq \mathbf{b}^j$, we can fix the variable $x_k$ to 0 if

$$z_k = \left( a_{jk} - \sum_{i \in C^-} a_{ji} \right) > b_j \,.$$

2. Consider $x_k$, $k \in C^-$, $j \in [m]$, and the following ILP:

$$z_k = \min \sum_{i \in C^+} a_{ji} x_i - \sum_{i \in C^-} a_{ji} x_i$$

$$\text{s.t.} \quad A^j \mathbf{x} \leq \mathbf{b}^j$$

$$\mathbf{x} \in \lbrace 0,1 \rbrace^n$$

$$x_k = 0$$

If $z_k > b_j$, the set described by constraints (7.37) and additionally $x_k = 0$ is empty. It implies that $x_k \neq 0$ in any feasible solution of (7.37) and can be set to 1. When we discard the system $A^j \mathbf{x} \leq \mathbf{b}^j$, we can fix the variable $x_k$ to 1 if

$$z_k = \left( -\sum_{i \in C^- \setminus \lbrace k \rbrace} a_{ji} \right) > b_j \,.$$

**Improving coefficients.**

1. Consider $x_k$, $k \in C^+$, $j \in [m]$, and the following ILP:

$$z_k = \max \sum_{i \in C^+} a_{ji} x_i - \sum_{i \in C^-} a_{ji} x_i$$

$$\text{s.t.} \quad A^j \mathbf{x} \leq \mathbf{b}^j$$

$$\mathbf{x} \in \lbrace 0,1 \rbrace^n$$

$$x_k = 0$$

If $z_k \leq b_j$, then the equality $\langle \mathbf{a_j}, \mathbf{x} \rangle \leq b_j$ is redundant under condition $x_k = 0$. This implies:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7.9</span></p>

In the above setting the set of feasible solutions of (7.37) does not change if we re-assign $a_{jk} := a_{jk} - \delta$ and $b_j := b_j - \delta$ with $\delta = b_j - z_k \geq 0$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Proposition 7.9</summary>

Indeed, if $x_k = 0$ the value of $a_{jk}$ does not play any role and the value of $b_j$ is essentially set to $z_k$.

To prove the implication for $x_k = 1$ consider

$$\sum_{i \in C^+} a_{ji} x_i - \sum_{i \in C^-} a_{ji} x_i \leq b_j$$

$$a_{jk} x_k + \sum_{i \in C^+ \setminus \lbrace k \rbrace} a_{ji} x_i - \sum_{i \in C^-} a_{ji} x_i \leq b_j$$

$$a_{jk} x_k - \delta x_k + \sum_{i \in C^+ \setminus \lbrace k \rbrace} a_{ji} x_i - \sum_{i \in C^-} a_{ji} x_i \leq b_j - \delta x_k$$

$$(a_{jk} - \delta) x_k + \sum_{i \in C^+ \setminus \lbrace k \rbrace} a_{ji} x_i - \sum_{i \in C^-} a_{ji} x_i \leq b_j - \delta \,. \quad \square$$

</details>
</div>

In total, when we discard the system $A^j \mathbf{x} \leq \mathbf{b}^j$, we can re-assign $a_{jk}$ and $b_j$ if

$$\sum_{i \in C^+ \setminus \lbrace k \rbrace} a_{ji} < b_j \,.$$

2. Consider $x_k$, $k \in C^-$, $j \in [m]$, and the following ILP:

$$z_k = \max \sum_{i \in C^+} a_{ji} x_i - \sum_{i \in C^-} a_{ji} x_i$$

$$\text{s.t.} \quad A^j \mathbf{x} \leq \mathbf{b}^j$$

$$\mathbf{x} \in \lbrace 0,1 \rbrace^n$$

$$x_k = 1$$

If $z_k \leq b_j$, then the equality $\langle \mathbf{a_j}, \mathbf{x} \rangle \leq b_j$ is redundant under condition $x_k = 1$. Consequently, given $x_k = 1$ we re-assign $a_{jk} := a_{jk} + (b_j - z_k)$ without changing the set of feasible solutions. Furthermore, also when $x = 0$ we can re-assign $a_{jk} := a_{jk} + (b_j - z_k)$ without changing the set of feasible solutions.

In total, we can re-assign $a_{jk} := a_{jk} + (b_j - z_k)$ without changing the set of feasible solutions. When we discard the system $A^j \mathbf{x} \leq \mathbf{b}^j$, we can increase $a_{jk}$ if

$$\sum_{i \in C^+} a_{ji} - a_{jk} < b_j \,.$$

**Logical implications.** As noted above, probing in combination with domain propagation can be used to efficiently derive logical implications between variables.

Consider $x_{k_1}$, $x_{k_2}$, $k_1, k_2 \in [n]$, and the following ILP:

$$z = \min \sum_{i \in C^+} a_{ji} x_i - \sum_{i \in C^-} a_{ji} x_i$$

$$\text{s.t.} \quad A^j \mathbf{x} \leq \mathbf{b}^j$$

$$\mathbf{x} \in \lbrace 0,1 \rbrace^n$$

$$x_{k_1} = 1$$

$$x_{k_2} = 1$$

If $z > b_j$ then the feasible set of (7.37) under condition of $x_{k_1} = x_{k_2} = 1$ is empty. In that case we can imply

$$x_{k_1} = 1 \quad \Rightarrow \quad x_{k_2} = 0 \,,$$

$$x_{k_2} = 1 \quad \Rightarrow \quad x_{k_1} = 0 \,,$$

which can be expressed as $x_{k_1} + x_{k_2} \leq 1$ for binary variables. Similarly the following logical implications can be identified:

$$x_{k_1} = 0 \quad \Rightarrow \quad x_{k_2} = 0 \,,$$

$$x_{k_1} = 0 \quad \Rightarrow \quad x_{k_1} = 1 \,,$$

$$x_{k_1} = 1 \quad \Rightarrow \quad x_{k_1} = 1 \,,$$

$$x_{k_2} = 0 \quad \Rightarrow \quad x_{k_1} = 0 \,,$$

$$x_{k_2} = 0 \quad \Rightarrow \quad x_{k_1} = 1 \,,$$

$$x_{k_2} = 1 \quad \Rightarrow \quad x_{k_1} = 1$$

The identification of logical implications proceeds in two phases. Suppose, w.l.o.g., that we want to identify logical implications associated with $x_{k_1} = 1$. In the first phase we reduce the initial system $A\mathbf{x} \leq \mathbf{b}$ by substituting $x_1 = 1$ throughout. In the second phase, each of the inequalities of the reduced system is analyzed in turn, using the above probing and domain propagation techniques to modify bounds and to fix variables, to establish whether any variable can be fixed. If so, logical implications have been identified.

Logical functions can be used to strengthen various functions embedded in an ILP solver, such as knapsack cover and clique constraints-based separation (Section 7.2.2) and primal heuristic computation.

### 7.4 Solving LP Relaxation

#### 7.4.1 Forms of Linear Programming Problem

Linear programs can be represented in different forms, suitable to different algorithms. Below we review these forms and provide transformations between them.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 7.10</span><span class="math-callout__name">(Forms of the Linear Program)</span></p>

Let $\mathbf{x} \in \mathbb{R}^n$. We distinguish the following three forms of the linear programs:

**General:**

$$\min \langle \mathbf{c}, \mathbf{x} \rangle$$

$$A\mathbf{x} \leq \mathbf{b}$$

$$A'\mathbf{x} = \mathbf{b}'$$

$$x_i \geq 0,\; i \in I \subseteq [n]$$

**Canonical:**

$$\min \langle \mathbf{c}, \mathbf{x} \rangle$$

$$A\mathbf{x} \leq \mathbf{b}$$

$$\mathbf{x} \geq \mathbf{0}$$

**Standard:**

$$\min \langle \mathbf{c}, \mathbf{x} \rangle$$

$$A\mathbf{x} = \mathbf{b}$$

$$\mathbf{x} \geq \mathbf{0}$$

</div>

These forms are equivalent, as any LP can be represented in any of them. The general form is essentially the most general. The following transformations are used to switch between different forms:

1. **Sign flipping:** $A\mathbf{x} \geq \mathbf{b} \Leftrightarrow (-A\mathbf{x} \leq -\mathbf{b})$
2. **Equality-to-inequality:** $A\mathbf{x} = \mathbf{b} \Leftrightarrow \lbrace A\mathbf{x} \leq \mathbf{b},\; A\mathbf{x} \geq \mathbf{b} \rbrace$
3. **Inequality-to-equality:** $a_j x_i \leq b_j \Rightarrow \lbrace a_j x_i + s = b_j,\; s \geq 0 \rbrace$, where the newly introduced *slack* variable $s$ is assigned zero cost.
4. **Unconstrained-to-constrained variables:** unconstrained $x_i$ is substituted with $x_i^+ - x_i^-$ and constrains $\lbrace x_i^+ \geq 0, x_i^- \geq 0 \rbrace$.

Below we list the transformations required to switch between these forms of linear programs taking into account that canonical and standard forms are special cases of the general one:

- **general $\to$ canonical**: 2, 1, 4;
- **general $\to$ standard**: 3, 4;
- **canonical $\to$ standard**: 3;
- **standard $\to$ canonical**: 2, 1.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 7.11</span></p>

The names *canonical* and *standard* do not seem to be consistent through the literature. E.g. in [6] the inverse inequality $\geq$ is used for the definition of the canonical form and in [3] the canonical form corresponds to the form of equations corresponding to the iterations of the simplex method as shown in (7.88).

</div>

#### 7.4.2 Simplex Method

Simplex method is one of the standard methods used by off-the-shelf LP solvers. Below we describe its idea and the basic simplified algorithm on an example.

**The idea of the simplex method** is the following procedure:

- Start with a vertex of the feasible set — the underlying polyhedron (in our case always a polytope)
- On each iteration move to the neighboring vertex with a lower objective value. In degenerate cases one stays in the same vertex and the same objective value. Special procedures are used to avoid cycling if the objective value does not drop over some iterations.

The above monotonicity and a finite number of vertices of a polytope guarantee that the algorithm attains a fixed point in a finite number of iterations. This number of iterations need not be polynomial: There are specially constructed examples that require an exponential number of iterations. However, in most practical applications, the number of simplex iterations grows proportionally to the number of variables.

The following theorem states that the fixed point is an optimum:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 7.12</span></p>

Let $P$ be a polytope, $\mathbf{c} \in \mathbb{R}^n$ be a cost vector. Let $\mathbf{x} \in \text{vrtx}(P)$ and $\mathcal{N}_P(x)$ be the set of its neighboring vertices in $P$. If $\langle \mathbf{c}, \mathbf{x} \rangle \leq \langle \mathbf{c}, \mathbf{z} \rangle$ for all $\mathbf{z} \in \mathcal{N}_P(x)$ then $\mathbf{x} \in \arg \min_{\mathbf{x}' \in P} \langle \mathbf{c}, \mathbf{x}' \rangle$.

</div>

The proof is based on the following

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 7.13</span></p>

Let the conditions of Theorem 7.12 hold. Then $\mathbf{x}$ is the local minimum of $\langle \mathbf{c}, \mathbf{x} \rangle$, i.e., there exists $\epsilon > 0$ such that $\mathbf{x} \in \arg \min_{\mathbf{x}' \in B_\epsilon(\mathbf{x}) \cap P} \langle \mathbf{c}, \mathbf{x}' \rangle$.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Lemma 7.13</summary>

Induction on the dimension $d$ of the polytope. For $d = 2$: By definition of the polytope's face (see Chapter 2) an optimum of a LP can be attained only on its face. For $d = 2$ the only faces are vertices and edges. To show local optimality it is sufficient to show optimality w.r.t. points on the edges adjacent to $\mathbf{x}$ that belong to the $\epsilon$-ball around $\mathbf{x}$, $B_\epsilon(\mathbf{x})$. It holds due to linearity of the objective (it is also linear along each edge) and the fact that all neighboring vertices has no better objective value than $\mathbf{x}$.

Assume it holds for the dimension $d$. Consider now $d+1$. A polytope with dimension $d+1$ has faces of dimension of at most $d$. Each of them is a polytope itself (see Corollary 2.50). In each of these polytopes $\mathbf{x}$ is a local optimum by assumption of the induction. Therefore, it is a local optimum for the intersection of their union with $B_\epsilon(\mathbf{x})$. $\square$

</details>
</div>

The proof of Theorem 7.12 follows directly from convexity of the problem $\min_{\mathbf{x}' \in P} \langle \mathbf{c}, \mathbf{x}' \rangle$, which implies that each local minimum is a global one, see Proposition 4.19.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 7.14</span><span class="math-callout__name">(Simplex Method)</span></p>

Consider the linear program in Figure 7.3:

$$\min_{\mathbf{x} \geq 0} -x_1 - 14x_2 - 6x_3$$

$$x_1 + x_2 + x_3 \leq 4$$

$$x_1 \leq 2$$

$$x_3 \leq 3$$

$$3x_2 + x_3 \leq 6$$

**Transformation into standard form:** The simplex algorithm requires the LP to be in a standard form. Therefore, we convert inequalities to equalities by introducing slack variables

$$x_1 + x_2 + x_3 + s_1 = 4$$

$$x_1 + s_2 = 2$$

$$x_3 + s_3 = 3$$

$$3x_2 + x_3 + s_4 = 6$$

**Simplex Tableau: Treat objective as equality.** Rewrite the problem as the following system of linear equations with the objective $z$ being one of the variables:

$$x_1 + x_2 + x_3 + s_1 = 4$$

$$x_1 + s_2 = 2$$

$$x_3 + s_3 = 3$$

$$3x_2 + x_3 + s_4 = 6$$

$$-z - x_1 - 14x_2 - 6x_3 = 0$$

In a compact form this takes the form of *simplex tableau*:

| | $x_1$ | $x_2$ | $x_3$ | $s_1$ | $s_2$ | $s_3$ | $s_4$ | RHS |
|---|---|---|---|---|---|---|---|---|
| $s_1$ | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 4 |
| $s_2$ | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 2 |
| $s_3$ | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 3 |
| $s_4$ | 0 | 3 | 1 | 0 | 0 | 0 | 1 | 6 |
| $-z$ | $-1$ | $-14$ | $-6$ | 0 | 0 | 0 | 0 | 0 |

Note that we already have a feasible solution to the problem: $x_i = 0$ for $i = 1, 2, 3$, $s_1 = 4$, $s_2 = 2$, $s_3 = 3$, $s_4 = 6$. It corresponds to the objective value $z = 0$.

</div>

**Basis solution.** The simplex algorithm performs linear operations (i.e., multiplication of constraints by a constant and adding one constraint to another) with the constraints defining simplex tableau such that:

- The number of constraints remains the same.
- These constraints define a linear system *equivalent* to the initial one.
- On each iteration of the simplex method the simplex tableau has the form:

| | $x_1$ | $x_2$ | $x_3$ | $x_4 \ldots x_n$ | RHS |
|---|---|---|---|---|---|
| $x_1$ | 1 | 0 | 0 | $\ldots$ | $v_1$ |
| $x_2$ | 0 | 1 | 0 | $\ldots$ | $v_2$ |
| $x_3$ | 0 | 0 | 1 | $\ldots$ | $v_3$ |
| $-z$ | $\ldots$ | $\ldots$ | $\ldots$ | $\ldots$ | $O$ |

W.l.o.g. we shifted the variables with $0/1$ coefficients (here $x_1$, $x_2$ and $x_3$) to the left side of the tableau. The variables $x_1, x_2, x_3$ take the *non-negative* values $v_1, v_2, v_3$. In this case the vector $(v_1, v_2, v_3, 0, \ldots, 0)$ is the feasible solution corresponding to the objective value $-O$. This solution is called a *basic (feasible) solution* or simply *basis*. Respectively, variables $x_1$, $x_2$, $x_3$ defining this solution are referred to as *basic variables*. The costs in the objective value row except for the RHS, are called *reduced costs*. Essentially in the course of the algorithm we switch between different basic solutions, where $m$ (number of constraints) variables get a (potentially) non-zero value and others — zero. The following proposition states the most important property of the basic solutions:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7.15</span></p>

Let $P = \lbrace A\mathbf{x} = \mathbf{b},\; \mathbf{x} \geq 0 \rbrace$ define a convex polytope. Then each basic feasible solution is a vertex of this polytope.

</div>

<div class="accordion" markdown="1">
<details>
<summary>Proof of Proposition 7.15</summary>

Indeed, let $\mathbf{x} = (v_1, \ldots, v_m, 0, \ldots, 0)$ be a basic feasible solution. Consider the cost vector $\mathbf{c}$ of the form

$$c_i = \begin{cases} 0, & i \in [m] \,, \\ 1, & i > m \,. \end{cases}$$

The objective value corresponding to $\mathbf{x}$ is equal to 0, which implies its minimality, as $\mathbf{c} \geq 0$ and $\mathbf{x} \geq 0$. It remains to show the uniqueness of the solution.

Consider now the equivalent representation of the system $A\mathbf{x} = \mathbf{b}$ that has a form (7.88) w.r.t. $\mathbf{x}$. Let $\mathbf{y} \geq 0$ be another solution, i.e., $\langle \mathbf{c}, \mathbf{y} \rangle = 0$. Then $y_i = 0$ for $i > m$. Let $\mathbf{z}^m$ be the first $m$ coordinates of any vector $\mathbf{z}$ and $I^m$ be the $m \times m$ identity matrix. Feasibility of $\mathbf{y}$ implies $A\mathbf{y} = \mathbf{b}$, and, therefore, $I^m \mathbf{y}^m = \mathbf{v}^m$, in the equivalent matrix representation (7.88). This implies $y_i = v_i$ for $i \in [m]$. Therefore, $\mathbf{y} = \mathbf{x}$. $\square$

</details>
</div>

The inverse implication that each vertex is a basic solution, holds as well:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 7.16</span></p>

Let $A$ have a full rank. Then each vertex of a non-empty convex polytope $P = \lbrace A\mathbf{x} = \mathbf{b},\; \mathbf{x} \geq 0 \rbrace$ is representable as a basic feasible solution in the respective simplex tableau.

</div>

Returning to the simplex tableau of Example 7.14, the last line contains coefficients of the respective variables in the objective function. Negative values of some of these coefficients imply that increasing the value of the respective variable (and possibly decreasing the value of another one to keep feasibility of the solution) will decrease the objective value. In our case such candidate variables are $x_1, x_2, x_3$.

**Identify the variable entering the basis.** Any of the variables with the negative coefficient in the objective row can be potentially added to the basis. There are several practical rules of selecting one of these variables, although there are no theoretical results claiming that one rule is actually better than another one. We will follow the rule of selecting the variable with the smallest respective coefficient in the objective row. The smallest value in the objective row is $-14$, so $x_2$ is the variable that enter the basis solution. As $x_2$ grows, the objective value takes the value $z = -0 - 14x_2$. Therefore, it is reasonable to set $x_2$ as large as possible to maximally decrease $z$. However, $x_2$ usually cannot be increased indefinitely, as we should guarantee the fulfillment of the constraints and non-negativity of all variables.

The basic variables $s_1, \ldots, s_4$ depend on $x_2$ as follows (given that $x_1 = x_3 = 0$):

$$s_1 = 4 - x_2 \quad : \quad s_1 \geq 0 \;\Rightarrow\; x_2 \leq 4/1 = 4$$

$$s_2 = 2 \quad : \quad s_2 \geq 0 \;\Rightarrow\; x_2 \leq 2/0 = \infty$$

$$s_3 = 3 \quad : \quad s_3 \geq 0 \;\Rightarrow\; x_2 \leq 3/0 = \infty$$

$$s_4 = 6 - 3x_2 \quad : \quad s_4 \geq 0 \;\Rightarrow\; x_2 \leq 6/3 = 2$$

We see that if $x_2$ increases beyond $6/3 = 2$ then $s_4$ becomes negative and if $x_2$ increases beyond $4/1 = 4$ then also $s_2$ becomes negative. To keep all variables non-negative we are allowed to set $x_2$ to the maximum value of 2. This also implies that $s_4$ gets the value 0 and leaves the basis.

**The value of variable $x_i$ entering the basis** is in general defined as the minimal non-negative ratio $\text{RHS}_j / a_{ji}$ for all current basis variables $x_j$. Negative and infinite ratios are ignored as they do not restrict the value of $x_i$. However, should this ratio be negative or infinite for all basis variables, the initial problem is unbounded and the algorithm can be stopped.

**Pivot on $x_2$ in the fourth ($s_4$) row.** Now our goal to bring the tableau to the form (7.88), with $x_2$ being the new basis variable in place of $s_4$. First we normalize the (fourth) pivot row by dividing its elements by 3 to obtain the coefficient 1 in front of $x_2$:

| | $x_1$ | $x_2$ | $x_3$ | $s_1$ | $s_2$ | $s_3$ | $s_4$ | RHS |
|---|---|---|---|---|---|---|---|---|
| $s_1$ | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 4 |
| $s_2$ | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 2 |
| $s_3$ | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 3 |
| $x_2$ | 0 | 1 | $\frac{1}{3}$ | 0 | 0 | 0 | $\frac{1}{3}$ | 2 |
| $-z$ | $-1$ | $-14$ | $-6$ | 0 | 0 | 0 | 0 | 0 |

Afterwards, we eliminate $x_2$ from all other rows including the objective function row by subtracting it from them with the suitable multipliers (1, 0, 0 and $-14$ for rows 1, 2, 3, 5 from top to bottom):

| | $x_1$ | $x_2$ | $x_3$ | $s_1$ | $s_2$ | $s_3$ | $s_4$ | RHS |
|---|---|---|---|---|---|---|---|---|
| $s_1$ | 1 | 0 | $\frac{2}{3}$ | 1 | 0 | 0 | $-\frac{1}{3}$ | 2 |
| $s_2$ | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 2 |
| $s_3$ | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 3 |
| $x_2$ | 0 | 1 | $\frac{1}{3}$ | 0 | 0 | 0 | $\frac{1}{3}$ | 2 |
| $-z$ | $-1$ | 0 | $-\frac{4}{3}$ | 0 | 0 | 0 | $\frac{14}{3}$ | 28 |

Pivoting is exchanging variables in the basis. One can show that changing the basis means moving to one of the adjacent vertices of the polytope in a non-degenerate case.

**Identify the variable entering the basis:** The smallest negative value in the objective row is $-\frac{4}{3}$, so $x_3$ is the new entering variable.

**Identify the variable leaving the basis (minimum ratio test):**

$$s_1 = 2 - \frac{2}{3} x_3 \quad : \quad s_1 \geq 0 \;\Rightarrow\; x_3 \leq 2/(2/3) = 3$$

$$s_2 = 2 \quad : \quad s_2 \geq 0 \;\Rightarrow\; x_3 \leq 2/0 = \infty$$

$$s_3 = 3 - x_3 \quad : \quad s_3 \geq 0 \;\Rightarrow\; x_3 \leq 3/1 = 3$$

$$x_2 = 2 - \frac{1}{3} x_3 \quad : \quad x_2 \geq 0 \;\Rightarrow\; x_3 \leq 2/(1/3) = 6$$

The minimum ratio is 3, so $s_1$ or $s_3$ can leave the basis. Let us select $s_3$.

**Pivot on $x_3$ in the third ($s_3$) row.** Normalize the (third) pivot row: Nothing to be done, the row is normalized.

| | $x_1$ | $x_2$ | $x_3$ | $s_1$ | $s_2$ | $s_3$ | $s_4$ | RHS |
|---|---|---|---|---|---|---|---|---|
| $s_1$ | 1 | 0 | $\frac{2}{3}$ | 1 | 0 | 0 | $-\frac{1}{3}$ | 2 |
| $s_2$ | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 2 |
| $x_3$ | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 3 |
| $x_2$ | 0 | 1 | $\frac{1}{3}$ | 0 | 0 | 0 | $\frac{1}{3}$ | 2 |
| $-z$ | $-1$ | 0 | $-\frac{4}{3}$ | 0 | 0 | 0 | $\frac{14}{3}$ | 28 |

Eliminate $x_3$ from the other rows by subtracting it with multipliers $2/3$, $0$, $1/3$ and $-4/3$ respectively:

| | $x_1$ | $x_2$ | $x_3$ | $s_1$ | $s_2$ | $s_3$ | $s_4$ | RHS |
|---|---|---|---|---|---|---|---|---|
| $s_1$ | 1 | 0 | 0 | 1 | 0 | $-\frac{2}{3}$ | $-\frac{1}{3}$ | 0 |
| $s_2$ | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 2 |
| $x_3$ | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 3 |
| $x_2$ | 0 | 1 | 0 | 0 | 0 | $-\frac{1}{3}$ | $\frac{1}{3}$ | 1 |
| $-z$ | $-1$ | 0 | 0 | 0 | 0 | $\frac{4}{3}$ | $\frac{14}{3}$ | 32 |

**Identify the variable entering the basis:** The only negative value in the objective row is $-1$, so $x_1$ is the new entering variable.

**Identify the variable leaving the basis (minimum ratio test):**

$$s_1 = 0 - x_1 \quad : \quad s_1 \geq 0 \;\Rightarrow\; x_1 \leq 0/1 = 0$$

$$s_2 = 2 - x_1 \quad : \quad s_2 \geq 0 \;\Rightarrow\; x_1 \leq 2/1 = 1$$

$$x_3 = 3 \quad : \quad x_3 \geq 0 \;\Rightarrow\; x_1 \leq 3/0 = \infty$$

$$x_2 = 1 \quad : \quad x_2 \geq 0 \;\Rightarrow\; x_1 \leq 1/0 = \infty$$

The minimum ratio is 0, so $s_1$ leaves the basis. Since the ratio is 0, the respective vertex is degenerate.

**Pivot on $x_1$ in the first ($s_1$) row.** Normalize the (first) pivot row: Nothing to be done, the row is normalized.

| | $x_1$ | $x_2$ | $x_3$ | $s_1$ | $s_2$ | $s_3$ | $s_4$ | RHS |
|---|---|---|---|---|---|---|---|---|
| $x_1$ | 1 | 0 | 0 | 1 | 0 | $-\frac{2}{3}$ | $-\frac{1}{3}$ | 0 |
| $s_2$ | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 2 |
| $x_3$ | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 3 |
| $x_2$ | 0 | 1 | 0 | 0 | 0 | $-\frac{1}{3}$ | $\frac{1}{3}$ | 1 |
| $-z$ | $-1$ | 0 | 0 | 0 | 0 | $\frac{4}{3}$ | $\frac{14}{3}$ | 32 |

Eliminate $x_1$ from the other rows by subtracting it with multipliers 1, 0, 0 and $-1$ respectively:

| | $x_1$ | $x_2$ | $x_3$ | $s_1$ | $s_2$ | $s_3$ | $s_4$ | RHS |
|---|---|---|---|---|---|---|---|---|
| $x_1$ | 1 | 0 | 0 | 1 | 0 | $-\frac{2}{3}$ | $-\frac{1}{3}$ | 0 |
| $s_2$ | 0 | 0 | 0 | $-1$ | 1 | $\frac{2}{3}$ | $\frac{1}{3}$ | 2 |
| $x_3$ | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 3 |
| $x_2$ | 0 | 1 | 0 | 0 | 0 | $-\frac{1}{3}$ | $\frac{1}{3}$ | 1 |
| $-z$ | 0 | 0 | 0 | 1 | 0 | $\frac{2}{3}$ | $\frac{13}{3}$ | 32 |

All reduced costs are non-negative, so we stop.

**Stopping condition.** Indeed, considering any positive value of the non-basic variables cannot decrease the objective, if all reduced costs are non-negative. Therefore, the optimum is attained. A formal proof can be found in [3], [6].

**Solution:** At this point, all coefficients in the objective function row are non-negative, indicating an optimal solution. The minimum value of the objective function $z$ is $-32$, occurring at $(x_1, x_2, x_3) = (0, 1, 3)$. One can see in Figure 7.3 that the vertex $(0, 1, 3)$ is indeed degenerate.

**Degeneracy of basic solutions.** Should the decisive ratio $\text{RHS}_j / a_{ji}$ be equal to 0, the variable $x_i$ enters the basis with the zero value and one speaks about *degenerate* solution. This is precisely the case when no improvement in the objective value is attained after the basis change. It happens when the same vertex of the polytope can be defined by multiple basis solutions (but the same full solution, as in this case we just select a subset of variables with 0 value to belong to the basis). Geometrically it means a degeneracy of the polytope vertex, when it is build by an intersection of more than minimally necessary number of facets. For example, for a 3-dimensional polytope this implies more than 3 hyper-planes, see, e.g., the vertex $(2, 2, 0)$ in Figure 7.3. Special rules to avoid infinite looping are proposed for handling degenerate cases.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 7.17</span></p>

Prove that if there is no degeneracy, $c_i < 0$ and $a_{ji} > 0$ for at least one $j$ from the basis, the objective value strictly decreases.

</div>

**Computational complexity of the simplex method.**

- The naïve simplex method requires $O(nm)$ operations per iteration for the $A \in \mathbb{R}^{n \times m}$.
- Assuming the average $O(n)$ iterations to attain the optimum results in $O(n^2 m)$ time complexity.
- Most combinatorial problems are very sparse — each constraint contains $\ll n$ non-zero coefficients. To make use of this a *revised simplex method* [3, Sec.3.5.3] is used, its complexity is $O(rnm + m^2)$, where $r$ is fraction of the non-zero coefficients. For the problems with $m \ll n$ this gives nearly linear (that considers only non-zero coefficients in $A$) complexity of each iteration.
- Still, even for this method the number of iterations required to attain the optimum or at least get close to it grows proportionally to $n$.
- In total, simplex method usually requires around $O(n^2)$ operations, which can be prohibiting to large-scale problems.

**Reduced costs as reparametrization.** Consider a linear program in the standard form (7.84). Reduced costs produced by the simplex method correspond to the reparametrized costs $\mathbf{c}^\lambda = \mathbf{c} + A^\top \boldsymbol{\lambda}$ of the following Lagrange dual

$$\max_{\boldsymbol{\lambda}} -\langle \boldsymbol{\lambda}, \mathbf{b} \rangle + \min_{\mathbf{x} \geq 0} \left\langle \mathbf{c} + A^\top \boldsymbol{\lambda}, \mathbf{x} \right\rangle \,.$$

The $j$-th coordinate $\lambda_j$ of the dual vector is the negated sum of multipliers applied to the $j$-th constraint before subtracting it from the cost line of the simplex tableau. Note that before reaching the optimum, the vector $\mathbf{c}^\lambda$ has negative coordinates. It renders the inner minimization is unbounded, so the respective lower bound is trivial (equal to $-\infty$).

**Advantages of the simplex method.** At the same time simplex method has nearly unique advantages that make it an ultimate tool for branching, cutting, *pricing* and learning:

**Algorithmic advantages.** The following operations are very cheap, efficient or can be done "on the fly":

- The reduced costs is a special case of reparametrized costs as they can be represented as $\mathbf{c} - A^\top \boldsymbol{\lambda}$, where $\boldsymbol{\lambda}$ accumulates multipliers used to turn the tableau to the canonical form (7.88) on each pivoting iteration. So one can explicitly track the respective dual variables.
- "Warm" start: Small changes of costs, e.g., during learning, are likely to change the optimal solution only slightly, so it will be reachable within several iterations from the current one. The respective new reduced costs can be computed from the tracked dual variables.
- Removing columns (variables) is required, e.g., by fixing variable value during branching. The column with a non-basis variable can directly be removed from the simplex tableau and the RHS corrected respectively.
- Adding columns corresponds to adding variables and is an important for *branch-and-price* algorithms that are beyond of scope of this course. Adding a column corresponding to a non-basis variable into the simplex tableau is straightforward.
- Adding constraints is required for cutting plane and branch-and-cut algorithms. By adding a new inequality constraint one automatically adds the respective slack variable to the basis. Transformation of the added inequality to the canonical form of the simplex tableau (7.88) requires $O(m)$ operations.

## Chapter 8: Primal Heuristics

### Memetic Algorithms

Primal heuristics are algorithms operating in the domain of the primal ILP problem $\min_{\mathbf{x} \in P \cap \lbrace 0,1 \rbrace^n} \langle \mathbf{c}, \mathbf{x} \rangle$, that have a weak theoretical substantiation and do not guarantee to attain the optimum. One of the most powerful and general classes of primal heuristics is known as *memetic algorithms*. They contain three types of subroutines:

1. **Generation** of *feasible* solutions, sometimes also referred to as *solution proposals*.
2. **Local search** to improve the available feasible solutions.
3. **Recombination** (also known as *crossover/fusion/merging*) of the feasible solutions. Here one attempts to build a better feasible solution on a basis of two or more proposals that are *fused/merged* together.

In the most general scenario one *generates* a pool of feasible solutions that are improved with *local search*. One selects a pair of the existing solutions, *recombines* them and adds them back to the pool. The simplest approach used in practice is an iterative recombination of the current solution with the newly generated (and improved by local search) solution proposal. The result of the recombination is considered as the new current solution.

**Proposal generation** should satisfy two criteria:

- **Quality** means that the proposals should have good (low for minimization problems) objective value to produce possibly good final solutions.
- **Diversity** is required to efficiently search through the solution space.

Often *randomized greedy* algorithms are used to generate proposals. The "greedy" is responsible for the quality and "randomized" for the diversity of the proposals.

**Local search** allows to improve quality of the proposals. The special case of the local search is the proposal *mutation*, when a subset of variables is fixed and one searches for the new feasible solution by optimizing over the rest of them. One searches for a better solution in the given *neighborhood* of a current one and, if the better solution is found, it is considered to be the current one and the process repeats until no better solution is found. In the case of mutation the neighborhood is defined by all possible values of the free, non-fixed variables.

The neighborhood that guarantees attaining an optimal solution in the course of the local search is called *exact*. Although the necessary and sufficient conditions for an exact neighborhood are known, it is often NP-hard to optimize through it.

**Recombination** is most often termed as *crossover* in the literature, but this term is reserved to the recovering of a basic solution from the interior point one (see Chapter 7). Here we restrict ourselves to the *optimized recombination*. Its idea is to fix the variables that are common to the two solutions that are merged, and optimize *w.r.t.* the rest. In other words, the optimized recombination is a special case of mutation, when the set of variables to be fixed is defined by two (or more) solution proposals.

### Greedy Algorithms

#### Greedy Algorithms for Binary ILPs

Consider the ILP

$$\min_{\substack{\mathbf{x} \in P \cap \lbrace 0,1 \rbrace^n \\ A\mathbf{x} \leqq \mathbf{b}}} \langle \mathbf{c}, \mathbf{x} \rangle \,.$$

We assume that the set $P \cap \lbrace 0,1 \rbrace^n$ can be represented as a Cartesian product of $k$ subspaces, $P \cap \lbrace 0,1 \rbrace^n = \hat{P}_1 \times \hat{P}_2 \times \cdots \times \hat{P}_k$, where each $\hat{P}_l$ has a form of $P_l \cap \lbrace 0,1 \rbrace^{n_l}$, $l \in [k]$ and covers a subset of the coordinates, e.g., $n_1 + n_2 + \cdots + n_k = n$. The respective variables and cost vectors will be denoted as $\mathbf{x}^l \in P_l \cap \lbrace 0,1 \rbrace^{n_l}$ and $\mathbf{c}^l \in \mathbb{R}^{n_l}$, $l \in [k]$.

The *greedy* algorithm sequentially, for $l = 1 \ldots k$, solves

$$\mathbf{x}^l \in \operatorname*{arg\,min}_{\substack{\mathbf{x} \in P_l \cap \lbrace 0,1 \rbrace^{n_l} \\ A^l \mathbf{x} \leqq \mathbf{b}^l}} \left\langle \mathbf{c}^l, \mathbf{x} \right\rangle \,.$$

Here $A^l \mathbf{x} \leqq \mathbf{b}^l$, $\mathbf{x} \in \lbrace 0,1 \rbrace^{n_l}$, $l \in [k]$, be the set of constraints derived from $A\mathbf{x} \leqq \mathbf{b}$ by fixing already computed values of $\mathbf{x}^{l-}$ and taking into account feasibility of the *yet-not-processed-variables* $\mathbf{x}^{l+}$.

The greedy algorithm is *well-defined*, if the resulting solution $\mathbf{x}^* := (\mathbf{x}^1, \ldots, \mathbf{x}^k)$ satisfies the initial constraints $A\mathbf{x}^* \leqq \mathbf{b}$ for any input cost vector $\mathbf{c}$.

In general, the algorithm has no optimality guarantees. The special type of problems, known as *weighted matroids*, can be solved with the algorithm exactly, though.

**Case study: Labeling problem**

Let $(\mathcal{G}, \mathcal{Y}_\mathcal{V}, \boldsymbol{\theta})$ be a labeling problem defined on a graph $\mathcal{G}$ with label sets $\mathcal{Y}_u$, $u \in \mathcal{V}$, and the cost vector $\boldsymbol{\theta} \in \mathbb{R}^{\mathcal{I}}$. W.l.o.g., we assume $\mathcal{G}$ to be connected. Let $u, v$: $uv \in \mathcal{E}$ be two neighboring nodes. The greedy algorithm starts with finding an optimal labeling in the induced subgraph defined by these nodes:

$$(y_u, y_v) := \operatorname*{arg\,min}_{s \in \mathcal{Y}_u, t \in \mathcal{Y}_v} \left(\theta_u(s) + \theta_v(t) + \theta_{uv}(s, t)\right) \,.$$

The subgraph $\mathcal{G}' := (\lbrace u, v \rbrace, uv)$ is called *labelled* as all of its nodes have got labels already. On each iteration the algorithm increases the size of the labelled subgraph by a single node and by all edges of $\mathcal{G}$ that connect it to the subgraph labelled on the previous iteration.

To do so, the algorithm finds a node $u$ that is connected to the already labelled subgraph $\mathcal{G}' = (\mathcal{V}', \mathcal{E}')$ and labels it by optimizing the sum of pairwise costs assigned to the edges between $u$ and $\mathcal{G}'$ as well as the unary cost of $u$:

$$y_u := \operatorname*{arg\,min}_{s \in \mathcal{Y}_u} \left(\theta_u(s) + \sum_{v \in \mathcal{N}(u) \cap \mathcal{V}'} \theta_{uv}(s, y_v)\right) \,.$$

The labelled subgraph is updated: $\mathcal{V}' := \mathcal{V}' \cup \lbrace u \rbrace$, $\mathcal{E}' := \mathcal{E}' \cup (\mathcal{N}(u) \cap \mathcal{V}')$. The iterations proceed until $\mathcal{G}'$ becomes equal to $\mathcal{G}$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 8.1</span></p>

Selecting the two-node subgraph for initialization is necessary to make the greedy algorithm robust *w.r.t.* possible problem reparametrizations.

If, in contrast, one would start with a single node labeling, this would result in a poor solution, e.g., in the case when all unary costs are equal to zero. To achieve this it is sufficient to perform the distribution operation (6.34) of the min-sum diffusion algorithm described in Section 6.5.4.

Optimization of a pairwise costs $\boldsymbol{\theta}_{uv}$ of a single edge $uv$ is not sufficient for a robust initialization either. Such reparametrization can be obtained by executing the collection step (6.34) of the min-sum diffusion algorithm for each node.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 8.2</span></p>

The order of processing nodes by the greedy algorithm can be arbitrary, as long as the connectivity condition $v \in \mathcal{N}(u) \cap \mathcal{V}'$ is satisfied. Different initial subgraphs and different processing order may lead to different labelings as a result.

</div>

**Case study: Max-weight independent set problem**

Consider the max-weight independent set problem as in Example 2.54 and assume all costs to be strictly positive, as nodes with non-positive costs can always be removed from the set without changing the optimal value.

The classical greedy algorithm iteratively

1. Finds a node $i$ with the highest *degree-normalized* cost $\frac{c_i}{\lvert \mathcal{N}(i) \rvert}$ in the *current* graph. Here $\mathcal{N}(i)$ is the set of neighboring nodes to $i$ in the current graph and $\lvert \mathcal{N}(i) \rvert$ is its degree. The best node $i \in \operatorname{arg\,max}_{j \in \mathcal{V}} \frac{c_j}{\lvert \mathcal{N}(j) \rvert}$ is labelled as a part of the resulting independent set.
2. Removes the node $i$ from the graph together will all its neighbors.

The iterations proceed until the resulting graph is empty. The selected nodes form an independent set by construction.

The degree-normalization takes into account that nodes with many neighbors are less preferable, as they block all their neighbors from being part of the required independent set.

This algorithm fits into the general scheme (8.2) as follows:

- The number of subspaces $k$ is equal to the number of nodes in the initial graph.
- These subspaces are of two types: (i) Those corresponding to the selected nodes with the highest degree-normalized cost and (ii) their neighbors. The respective subproblems are optimized in the same order.

**Case study: Binary knapsack problem**

Consider the binary knapsack problem as in Example 2.53:

$$\max_{\mathbf{x} \in \lbrace 0,1 \rbrace^n} \langle \mathbf{c}, \mathbf{x} \rangle$$

$$\text{s.t.} \quad \langle \mathbf{a}, \mathbf{x} \rangle \leq b \,.$$

W.l.o.g., we assume $c_i > 0$, $a_i > 0$ and $a_i \leq b$.

Define *efficiency* of the item $i$ as the ratio of the item cost to its volume: $\frac{c_i}{a_i}$.

The *greedy-knapsack* algorithm starts with the empty knapsack and adds items in the decreasing order of their efficiency if it is not prohibited by the volume constraint. A small modification of the algorithm computes a solution with a total cost at least half as large as the optimal solution value. The algorithm's complexity is determined by the complexity of sorting the $n$ items according to their efficiency and in the most general setting is $O(n \log n)$.

In contrast, a *naïve-greedy-knapsack* algorithm uses costs instead of efficiencies and has no approximation guarantees.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 8.3</span></p>

A variant of the *greedy-knapsack* algorithm that stops when it fails to add an item for the first time is called the *greedy-split-knapsack* and can be computed in linear time as it does not require sorting of all elements. This algorithm has an intimate relation to the solution of the Lagrange relaxation of the knapsack problem.

Recall Example 5.19: The solution of the Lagrange relaxation reads $x_i = 1$ for $i \in [s-1]$, $x_s = (b - \sum_{i=1}^{s-1} a_i)/a_s$ and $x_t = 0$ for $t > k$, where $s$ is the maximal value such that $\sum_{i=1}^{s-1} a_i < b$.

A simple "rounding" of this solution to an integer one consists in setting $x_k$ to zero if it is not equal to one. Precisely this solution is returned also by the greedy-split-knapsack algorithm.

</div>

#### Leveraging Lagrange Relaxation

In this section we restrict ourselves to the dualization of the equality constraints. Consider the Lagrange dual of (8.1):

$$\max_{\boldsymbol{\lambda}} -\langle \boldsymbol{\lambda}, \mathbf{b} \rangle + \min_{\mathbf{x} \in P \cap \lbrace 0,1 \rbrace^n} \left\langle \mathbf{c} + A^\top \boldsymbol{\lambda}, \mathbf{x} \right\rangle \,.$$

The dual optimality condition (Corollary 5.17) implies that if there exists

$$\mathbf{x}^* \in \operatorname*{arg\,min}_{\mathbf{x} \in P \cap \lbrace 0,1 \rbrace^n} \left\langle \mathbf{c} + A^\top \boldsymbol{\lambda}, \mathbf{x} \right\rangle$$

such that $A\mathbf{x}^* = \mathbf{b}$, then $\mathbf{x}^*$ is the exact solution of the ILP (8.1). Hence, the greedy algorithm (8.2), when applied to the reparametrized costs $\mathbf{c}^\lambda = \mathbf{c} + A^\top \boldsymbol{\lambda}$ can be interpreted as an attempt to find a solution $\mathbf{x}^*$ that satisfies the above condition. According to the dual optimality condition (Corollary 5.17) this greedy algorithm finds an optimal ILP solution if the solutions $\mathbf{x}^l$ to all greedy subproblems also satisfy the unconstrained versions. We will refer to the class of algorithms above as *reparametrized greedy algorithms*.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.4</span><span class="math-callout__name">(Acyclic labeling problem)</span></p>

Let $\mathcal{G}$ be acyclic and the reparametrized costs $\boldsymbol{\theta}^\phi$ be the fix-point of the diffusion algorithm. Proposition 5.29 together with Theorem 6.19 imply that the dual vector $\boldsymbol{\phi}$ is optimal. All reparametrized unary costs are zero. W.l.o.g. we will assume that also all locally optimal label pairs have zero reparametrized pairwise costs and otherwise all costs are non-negative.

Apply now the greedy algorithm (8.3)-(8.4) to the reparametrized costs $\boldsymbol{\theta}^\phi$. Below we show that this algorithm finds an optimal integer solution. It is sufficient to show that the solution corresponds to the zero objective value.

Indeed, at the first step (8.3) the algorithm labels the connected subgraph $\mathcal{G}' = (\mathcal{V}', \mathcal{E}') = (\lbrace u, v \rbrace, uv)$ for arbitrary $uv \in \mathcal{E}$. The optimal labeling $(y_u, y_v)$ of this subgraph has the total cost zero. Since the graph is acyclic, the set $\mathcal{N}(u) \cap \mathcal{V}'$ contains a single element $v$, hence (8.4) turns into

$$y_u := \operatorname*{arg\,min}_{s \in \mathcal{Y}_u} \left(\theta_u^\phi(s) + \theta_{uv}^\phi(s, y_v)\right) \,.$$

It holds $\theta_u^\phi(s) = 0$ for any $s \in \mathcal{Y}_u$ and $\theta_{uv}^\phi(s, y_v) \geq 0$ by construction. Therefore, it is sufficient to prove that there is $s \in \mathcal{Y}_u$ such that $\theta_{uv}^\phi(s, y_v) = 0$. The subgraph $\mathcal{G}'$ is connected, therefore there is $v' \in \mathcal{V}'$ such that $uv' \in \mathcal{E}'$. By assumption $\theta_{vv'}^\phi(y_v, y_{v'}) = 0$ and $(y_v, y_{v'})$ is locally optimal for the edge $vv'$, i.e.,

$$\min_{t \in \mathcal{Y}_{v'}} \theta_{vv'}^\phi(y_v, t) = 0 \,.$$

For the "redistribution" operation of the min-sum diffusion algorithm (6.34), as well as for its fix point, it holds $\min_{t \in \mathcal{Y}_{v'}} \theta_{vv'}^\phi(y_v, t) = \min_{s \in \mathcal{Y}_u} \theta_{uv}^\phi(u, y_v)$ for any two neighbors $v'$ and $u$ of $v$, see (6.36). Taking into account (8.20), we obtain $\min_{s \in \mathcal{Y}_u} \theta_{uv}^\phi(u, y_v) = 0$, which finalizes the proof.

Note that the greedy method (8.3)-(8.4) applied to the non-reparametrized costs does not have any optimality guarantee.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.5</span><span class="math-callout__name">(Max-weight independent set problem, slack-based clique ILP formulation)</span></p>

Consider the clique-based ILP representation of the max-weight independent set problem:

$$\max_{\mathbf{x} \in \lbrace 0,1 \rbrace^{n+m}} \langle \mathbf{c}, \mathbf{x} \rangle$$

$$\text{s.t.} \quad \sum_{i \in \bar{K}_j} x_i = 1 ,\; j \in [m] \,.$$

The dual problem reads

$$\min_{\boldsymbol{\lambda}} \left[\sum_{j=1}^{m} \lambda_j + \max_{\mathbf{x} \in \lbrace 0,1 \rbrace^{n+m}} \langle \mathbf{c}^\lambda, \mathbf{x} \rangle \right]$$

We split the problem into $k = m$ subproblems, one for each clique $\bar{K}_j$, $j \in [m]$. Here $\hat{K}_j$ contains all elements of $\bar{K}_j$ but those that have been fixed in the previous iterations of the algorithm. The reparametrized greedy iteration subproblem (8.17) takes the form

$$(x_i^*)_{i \in \hat{K}_j} \in \operatorname*{arg\,max}_{\mathbf{x} \in \lbrace 0,1 \rbrace^{\hat{K}_j}} \sum_{i \in \hat{K}_j} c_i^\lambda x_i$$

$$\text{s.t.} \quad \sum_{i \in \hat{K}_j} x_i = 1$$

Here we optimize over the non-fixed variables from $\hat{K}_j$, but take into account all (including fixed) variables in the whole clique $\bar{K}_j$ by considering the corresponding clique constraint.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm 6</span><span class="math-callout__name">(Reparametrized greedy algorithm for the MWIS problem)</span></p>

1. **Input:** $\mathbf{c}^\lambda \in \mathbb{R}^{n+m}$ — reduced costs
2. Initialize $\mathbf{x} := -\mathbf{1} \in \mathbb{R}^{n+m}$ as undefined
3. Iterate over all cliques:
4. **for** $j \in [m]$ **do**
5. &emsp; Skip cliques with all fixed nodes:
6. &emsp; **if** $\lbrace i \in \bar{K}_j \mid x_i = 1 \rbrace = \emptyset$ **then**
7. &emsp;&emsp; Find an assignable and locally optimal node
8. &emsp;&emsp; $i^* := \operatorname*{arg\,max}_{i \in \bar{K}_j \text{ s.t. } x_i = -1} c_i^\lambda$
9. &emsp;&emsp; $x_{i^*} := 1$ and set it to one
10. &emsp;&emsp; and all its graph neighbors to zero:
11. &emsp;&emsp; **for** $i$: $\lbrace i^*, i \rbrace \in \mathcal{E}$ **do**
12. &emsp;&emsp;&emsp; $x_i := 0$
13. &emsp;&emsp; **end for**
14. &emsp; **end if**
15. **end for**
16. Return $\mathbf{x}$

Note that Algorithm 6 is well-defined and finds an optimal solution, if $\lvert \lbrace i \in \bar{K}_j : c_i^\lambda \geq 0 \rbrace \rvert = 1$ as this solution satisfies the optimality criterion (8.18).

</div>

**Identification of an integer solution is NP-complete**

Assume the Lagrange relaxation (8.14) is tight and there is an integer $\mathbf{x}^*$ that satisfies conditions (8.15)-(8.16). It does not mean that finding this $\mathbf{x}^*$ is an easy problem. On the contrary, this problem is NP-complete in general:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 8.6</span><span class="math-callout__name">(Subset Sum Problem)</span></p>

The decision problem: *Is there* $\mathbf{x} \in \lbrace 0, 1 \rbrace^n$ *such that* $\sum_{i=1}^{n} a_i x_i = b$ is called the *subset sum problem*.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8.7</span></p>

The subset sum problem is NP-complete.

</div>

The subset-problem reduces to the problem of identification of an integer relaxed solution with $\mathbf{c}^\lambda = 0$, $P = \mathbb{R}^n$ and $A\mathbf{x} = \mathbf{b}$ being equal to $\sum_{i=1}^{n} a_i x_i = b$.

The identification problem is in NP class, as for any $\mathbf{x}' \in \lbrace 0,1 \rbrace^n$ it can be verified in polynomial time that $A\mathbf{x}' = \mathbf{b}$ and

$$\left\langle \mathbf{c} + A^\top \boldsymbol{\lambda}, \mathbf{x} \right\rangle = \min_{\mathbf{x} \in P \cap \lbrace 0,1 \rbrace^n} \left\langle \mathbf{c} + A^\top \boldsymbol{\lambda}, \mathbf{x} \right\rangle \,,$$

since the right-hand-side is assumed to be known as the relaxed optimal value.

Due to NP-completeness, the reparametrized greedy algorithm can be seen as a computationally cheap method to find $\mathbf{x}^*$, however, without guarantee in general. To improve performance, its *randomized* versions are used, that allow to efficiently sample the search space around the optimum. We consider these algorithms in Section 8.2.3 below.

**Non-optimal dual and non-tight relaxations**

Apart from being comparably computationally cheap, the practical advantage of the reparametrized greedy algorithm is its applicability in a significantly broader setting than in the optimum of the tight Lagrange relaxation. Indeed, it remains applicable even when

- the Lagrange relaxation (8.14) is not tight, or
- the attained dual solution $\boldsymbol{\lambda}$ is non-optimal.

The latter is especially typical for large-scale problems, when one is limited in usage to the (non-polynomially convergent) first-order dual optimization methods such as sub-gradient one. These are rarely able to attain the optimum with numerical precision. A relative approximation accuracy of 0.1% of the optimal dual value is often the best one can hope for with the first-order techniques in large-scale practical applications.

A (well-defined) reparametrized greedy algorithm typically finds progressively better integer solutions as the dual variables get closer to the relaxed optimum. This effect gets even more pronouncing, when the Lagrange relaxation is nearly tight and the relaxed solutions have *mostly* integer coordinates.

**ILPs with inequalities for reparametrized greedy algorithms**

The optimality condition for dual variables corresponding to inequalities $A\mathbf{x} \leq \mathbf{b}$ differs from the one for equalities (see Corollary 5.17). Namely $\boldsymbol{\lambda}$ is optimal if there is a solution $\mathbf{x}^*$ to (8.15) such that in addition to the feasibility $A\mathbf{x}^* \leq \mathbf{b}$ the complementary slackness condition

$$\langle \boldsymbol{\lambda}, A\mathbf{x}^* - \mathbf{b} \rangle = 0$$

holds. This implies one would search for a vector

$$\mathbf{x}^* \in \operatorname*{arg\,min}_{\substack{\mathbf{x} \in P \cap \lbrace 0,1 \rbrace^n \\ A\mathbf{x} \leq \mathbf{b},\; \langle \boldsymbol{\lambda}, A\mathbf{x} - \mathbf{b} \rangle = 0}} \left\langle \mathbf{c}^\lambda, \mathbf{x} \right\rangle \,,$$

however, the feasible set of this problem may be empty, if the relaxation has no integer solution.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.8</span><span class="math-callout__name">(Binary knapsack problem)</span></p>

The condition (8.28) takes the form

$$\mathbf{x}^* \in \operatorname*{arg\,min}_{\substack{\mathbf{x} \in \lbrace 0,1 \rbrace^n \\ A\mathbf{x} = \mathbf{b}}} \left\langle \mathbf{c}^\lambda, \mathbf{x} \right\rangle$$

Assume, as above, that the variables are enumerated in the decreasing order of their efficiencies $\frac{c_i}{a_i}$. Recall Example 5.19: An optimal dual value $\mu$ is equal to $\frac{c_s}{a_s}$, where $s$ is the maximal value such that $\sum_{i=1}^{s-1} a_i < b$.

If $\sum_{i=1}^{s} a_i = b$, then the solution $\mathbf{x}^*$ of (8.29) has coordinates $x_i^* = \llbracket i \leq s \rrbracket$ and this is the exact solution of the problem (and the Lagrange relaxation).

In general, however, the feasible set of (8.29) is empty.

</div>

#### Randomized Greedy Algorithms

As mentioned in Chapter 8, *randomized* greedy subroutines are a common tool for proposal generation in memetic algorithms. They can be used with original as well as with the reparametrized costs.

**Order randomization.** Randomization may lead to the optimal solution (that satisfies (8.15) and (8.16)), if, e.g., the processing of the (randomized) greedy algorithm iterations (8.2) (resp. (8.17)) is done in the order $\pi(l)$ instead of $l$ with $\pi$ being a *suitable* randomized permutation.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.9</span><span class="math-callout__name">(Labeling problem)</span></p>

Consider the greedy algorithm described in Section 8.2.1 and applied to the initial or reparametrized costs. One can randomly select the initial pair of neighboring nodes $uv \in \mathcal{E}$, see (8.3), and on each iteration *randomly* select a neighboring node (node $u$ in (8.4)).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.10</span><span class="math-callout__name">(Max-weight independent set problem)</span></p>

Consider the greedy algorithm described in Example 8.5. The order of processing of the cliques can be randomly selected.

</div>

**Cost randomization.** Another randomization method consists in randomizing the costs. There are several modifications of the technique:

- Instead of selecting the minimal (maximal) in (8.2) or (8.17), one selects elements randomly *w.r.t.* the distribution of their costs. Possible options $\propto \exp(c/T)$ or uniform selection from $\alpha$ best options.
- Adding noise to the costs during generation. This can be in particular useful, if the existing greedy algorithm does not allow to other sorts of randomization.
- Adding noise may not be required, if the dual is being optimized with the subgradient method. In this case reparametrizations computed on different iterations are diverse enough and "naturally randomized" due to the suboptimal length of the subgradient steps.

Costs randomization allows to explore a larger part of the solution space, although it may require longer to obtain good solutions. Usually it is used in combination with the order randomization and shows best results when the Lagrange relaxation (8.14) is far from being tight and/or the dual solution $\boldsymbol{\lambda}$ is highly non-optimal.

### Local Search

#### Main Definitions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 8.11</span><span class="math-callout__name">(Neighborhood)</span></p>

A mapping $\mathcal{N} \colon X \to 2^X$ from a set $X$ to its powerset is called *neighborhood*.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 8.12</span><span class="math-callout__name">(Local Search)</span></p>

Algorithm 7 is called *local search* for the lowest value of the function $f \colon X \to \mathbb{R}$ with respect to the neighborhood $\mathcal{N}$. Note that the special case of Algorithm 7, with $x^* = \operatorname{arg\,min}_{x \in \mathcal{N}(x^t)} f(x)$, is covered by this definition as well.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm 7</span><span class="math-callout__name">(General local search algorithm)</span></p>

1. **Input:** Initial solution $x^0 \in X$
2. $t := 0$
3. **while** $\exists x^* \in \mathcal{N}(x^t)$: $f(x^*) < f(x^t)$ **do**
4. &emsp; $t := t + 1$
5. &emsp; $x^t := x^*$
6. **end while**
7. Return $x^t$.

</div>

Such algorithms as (block)-ICM (Section 6.5.1) and simplex method (Section 7.4.2) satisfy Definition 8.12. For continuous $X$, the coordinate-descent (Section 6.3) and gradient descent (Section 6.1) can be seen as local search.

#### Mutation

Mutation is the type of local search problems, when a subset of coordinates of a feasible solution $\mathbf{x}$ is *fixed* and the problem is optimized with respect to the rest of *free* coordinates. In case of general ILP solvers the resulting problem is obtained with domain propagation. Usually the mutation problem belongs to the same problem class as the initial one, but has a much smaller size. Hence in general one has four options how to address it, with

- The total *enumeration* of possible solutions;
- *Branching* (branch-and-bound/cut) as a more scalable way to obtain an exact solution;
- *Heuristic* algorithms (such as greedy ones) to obtain a (hopefully) good feasible solution;
- *Specialized* exact or approximate algorithms, if the resulting problem belongs to an easier solvable subclass of the more general initial problem.

Whereas mutation problems with a small set of feasible solutions are usually solved with simple and efficient enumeration algorithms, the subproblems with larger feasible set often allow to attain better objective values. The sweet spot are the mutation subproblems with exponentially large feasible set that build a special, efficiently solvable special case of the initial problem.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.13</span><span class="math-callout__name">(Labeling problem)</span></p>

The ICM algorithm, see Section 6.5.1, is a classical mutation-based method. In the case of the single-node updates, as in Algorithm 3, these are usually computed with the total enumeration of all labels in a given node. In the case of the block-ICM, see Algorithm 4, the resulting subproblem belongs to the efficiently solvable subclass of the acyclic labeling problems and can be solved with the specialized, dynamic programming algorithm.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.14</span><span class="math-callout__name">(Binary knapsack)</span></p>

Assume $\mathbf{x} \in \lbrace 0,1 \rbrace^n$ to be a feasible solution of the problem (8.11) and let all coordinates but $x_i$, $i \in [k]$, be fixed. Then the mutation problem takes the form

$$\max_{\mathbf{x} \in \lbrace 0,1 \rbrace^k} \sum_{i=1}^{k} c_i x_i$$

$$\text{s.t.} \quad \sum_{i=1}^{k} a_i x_i \leq \hat{b} := b - \sum_{j=k+1}^{n} a_j x_j \,,$$

which is a binary knapsack problem with $k$ variables and smaller maximal volume parameter $\hat{b}$. Hence it can be either addressed with the dynamic programming (if $\hat{b}$ is small enough), or one of the greedy algorithms (see Section 8.2.1) can be used to find an approximate solution.

</div>

**Mutation for General ILPs.** Mutation is also used by off-the-shelf ILP solvers as one of the primal heuristics. In the simplest implementation one fixes a certain amount, (80-90%) of randomly selected variables and solves the problem with respect to the remaining ones. However, fixing of one variable may lead to the implicit fixing of another one, see Example 8.16. In turn, this may result in a very small feasible sets of the mutation problem leading to its overall inefficiency. The problem can be mitigated by interchangeable fixing of the variables and domain propagation until a certain number of *actually* free variables is attained.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.16</span><span class="math-callout__name">(Implicit variable fixing)</span></p>

Consider the local polytope relaxation of the labeling problem (3.45). Let $uv \in \mathcal{E}$ and the node variables are fixed, e.g., $\mu_u(s) = \llbracket s = y_u \rrbracket$ and $\mu_v(l) = \llbracket l = y_v \rrbracket$. This implies that all pairwise variables in the edge $uv$ are implicitly fixed as well to $\mu_{uv}(s, l) = \llbracket s = y_u \rrbracket \cdot \llbracket l = y_v \rrbracket$ that significantly decreases the size of the mutation problem.

</div>

#### Exact Neighborhood of Binary ILPs

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 8.17</span><span class="math-callout__name">(Exact Neighborhood)</span></p>

Let $X$ be finite. A neighborhood $\mathcal{N}$ is called *exact*, if Algorithm 7 returns an element of $\operatorname{arg\,min}_{x \in X} f(x)$.

</div>

In this section we formulate necessary and sufficient conditions for a local search neighborhood to be *exact*. Consider the ILP:

$$\min_{x \in P \cap \lbrace 0,1 \rbrace^n} \langle \mathbf{c}, \mathbf{x} \rangle \,.$$

Recall (see Section 2.5) that the set $M = \text{conv}(P \cap \lbrace 0,1 \rbrace^n)$ is called its *integer hull*.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 8.18</span><span class="math-callout__name">(Adjacent Vertices)</span></p>

Vertices of a polytope are called *adjacent*, if they are connected by an edge. For a given $\mathbf{x} \in \text{vrtx}(M) = P \cap \lbrace 0,1 \rbrace^n$ let $\text{Ad}(\mathbf{x}) \subseteq \text{vrtx}(M)$ be the set of adjacent vertices of the polytope $M$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 8.19</span></p>

A neighborhood $\mathcal{N} \colon \text{vrtx}(M) \to 2^{\text{vrtx}(M)}$ is exact for any cost vector $\mathbf{c}$ if and only if $\mathcal{N}(\mathbf{x}) \supseteq \text{Ad}(\mathbf{x})$ for any $\mathbf{x} \in \text{vrtx}(M)$.

</div>

To prove Theorem 8.19 we will require the following

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 8.20</span></p>

Let $P$ be a polytope and $\text{vrtx}(P) \in \lbrace 0,1 \rbrace^n$. Let also $\hat{\mathbf{x}}, \hat{\mathbf{y}} \in \text{vrtx}(P)$. Then $\hat{\mathbf{y}} \in \text{Ad}(\hat{\mathbf{x}})$ if and only if there exists $\mathbf{c} \in \mathbb{R}^n$ such that $\hat{\mathbf{x}}$ is a unique optimal solution and $\hat{\mathbf{y}}$ is the unique second best solution of $\min_{\mathbf{x} \in P} \langle \mathbf{c}, \mathbf{x} \rangle$.

</div>

*Proof.* **Necessity:** Assume $\hat{\mathbf{y}} \in \text{Ad}(\hat{\mathbf{x}})$, i.e., $\hat{\mathbf{x}}$ and $\hat{\mathbf{y}}$ belong to some edge of the polytope. By the definitions of the edge, there is a cost vector $\mathbf{c}$ such the edge is the set $\operatorname{arg\,min}_{\mathbf{x} \in P} \langle \mathbf{c}, \mathbf{x} \rangle$. In particular, $\langle \mathbf{c}, \hat{\mathbf{x}} \rangle = \langle \mathbf{c}, \hat{\mathbf{y}} \rangle = c^0 := \min_{\mathbf{x} \in P} \langle \mathbf{c}, \mathbf{x} \rangle$. Below we will slightly modify $\mathbf{c}$ to obtain $\mathbf{c}'$ such that the statement of the lemma holds.

Define $Q := \min_{\mathbf{x} \in \text{vrtx}(P) \setminus \lbrace \hat{\mathbf{x}}, \hat{\mathbf{y}} \rbrace} [\langle \mathbf{c}, \mathbf{x} \rangle - c^0] > 0$ as the minimal difference in the objective value between the edge $\hat{\mathbf{x}} - \hat{\mathbf{y}}$ and the rest of vertices of the polytope $P$.

Since $\hat{\mathbf{x}} \neq \hat{\mathbf{y}}$ there is $j \in [n]$ such that $\hat{y}_j \neq \hat{x}_j$ and only two cases are considered below are possible.

**Case 1:** $\hat{y}_j = 1$, $\hat{x}_j = 0$. Consider vector $\mathbf{c}'$ such that $c'_j = c_j + Q/2$ and $c'_i = c_i$ for $i \in [n] \setminus \lbrace j \rbrace$. Then $\hat{\mathbf{x}}$ is the unique best and $\hat{\mathbf{y}}$ is the unique second best solution.

**Case 2:** $\hat{y}_j = 0$, $\hat{x}_j = 1$. Consider vector $\mathbf{c}'$ such that $c'_j = c_j - Q/2$ and $c'_i = c_i$ for $i \in [n] \setminus \lbrace j \rbrace$. Then $\hat{\mathbf{x}}$ is the unique best and $\hat{\mathbf{y}}$ is the unique second best solution.

Thus, the necessity is proven.

**Sufficiency:** Assume $\hat{\mathbf{x}}$ is the unique best and $\hat{\mathbf{y}}$ is the second best solution for some cost vector $\mathbf{c}$ and $\hat{\mathbf{x}} \notin \text{Ad}(\hat{\mathbf{y}})$. Then, according to Lemma 7.13, $\hat{\mathbf{y}}$ is the local optimum of the convex optimization problem $\min_{\mathbf{x} \in P} \langle \mathbf{c}, \mathbf{x} \rangle$. However, its unique global optimum is $\hat{\mathbf{x}}$, which is a contradiction, as for convex problems all local optima are global ones (consider the convex function $\langle \mathbf{c}, \mathbf{x} \rangle + \iota_P(x)$ and apply Proposition 4.19). $\square$

Now we are ready to prove Theorem 8.19:

*Proof.* **Sufficiency:** For the sake of notation denote $X := P \cap \lbrace 0,1 \rbrace^n = \text{vrtx}(M)$. The neighborhood $\mathcal{N}$ is exact for any cost vector $\mathbf{c}$, by the simplex algorithm applied to

$$\min_{\substack{y \\ p_\mathbf{x}: \mathbf{x} \in X}} \langle \mathbf{c}, \mathbf{y} \rangle$$

$$\text{s.t.} \quad y = \sum_{\mathbf{x} \in X} p_\mathbf{x} \mathbf{x}$$

$$\sum_{\mathbf{x} \in X} p_\mathbf{x} = 1$$

$$p_\mathbf{x} \geq 0, \; \mathbf{x} \in X$$

$$\mathbf{y} \geq 0$$

Here we used the representation (2.35) of the polytope $M$ and added the constraint $\mathbf{y} \geq 0$ as all $\mathbf{x}$ are binary integer vectors and, therefore, non-negative. Not that the used representation in an LP in the standard form, so the simplex algorithm is directly applicable.

**Necessity:** Assume $\hat{\mathbf{x}} \in \text{Ad}(\hat{\mathbf{y}})$ and $\mathcal{N}(\hat{\mathbf{y}}) \ni \hat{\mathbf{x}}$. Then, according to Lemma 8.20 applied to the polytope $M$, there is such a cost vector $\mathbf{c}$ that $\hat{\mathbf{x}}$ is the unique best solution and $\hat{\mathbf{y}}$ is the unique second best one. The respective local search, when started in this point, as $\langle \mathbf{c}, \hat{\mathbf{y}} \rangle < \langle \mathbf{c}, \mathbf{x} \rangle$ for any $\mathbf{x} \in \mathcal{N}(\hat{\mathbf{y}})$. $\square$

Unfortunately, Theorem 8.19 has mainly a theoretical value, as even answering the question whether two solutions correspond to adjacent vertices of the integer hull is NP-hard in general.

### Optimal Recombination

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 8.21</span><span class="math-callout__name">(Binary Labeling Problem)</span></p>

A labeling problem is called *binary*, if each of its nodes contains at most two labels. Without loss of generality one can assume these labels to be *zero* and *one*.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.22</span><span class="math-callout__name">(Labeling problem)</span></p>

Let $(\mathcal{G}, \boldsymbol{\theta}, \mathcal{Y}_\mathcal{V})$ be a multilabel labeling problem and let $\mathbf{y}^1, \mathbf{y}^2 \in \mathcal{Y}_\mathcal{V}$ be two labelings. The optimal recombination problem consists in finding a labeling $\mathbf{z} \in \mathcal{Y}_\mathcal{V}$ with the minimal total cost such that $z_u \in \lbrace y_u^1, y_u^2 \rbrace$ for all $u \in \mathcal{V}$.

This problem can be posed as a binary labeling problem $(\mathcal{G}', \boldsymbol{\theta}', \mathcal{Y}'_{v'})$ with $\mathcal{G}' = \mathcal{G}$, $\mathcal{Y}'_u = \lbrace y_u^1, y_u^2 \rbrace$ and $\theta'_u(y_u^i) = \theta_u(y_u^i)$, $\theta'_{uv}(y_u^i, y_u^j) = \theta_{uv}(y_u^i, y_u^j)$, where $i, j \in \lbrace 1, 2 \rbrace$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.23</span><span class="math-callout__name">(Max-weight independent set problem)</span></p>

Let the graph $(\mathcal{V}, \mathcal{E})$ and costs $\mathbf{c} \in \mathbb{R}^\mathcal{V}$ define a max-weight independent set problem. And let $\mathbf{x}^1, \mathbf{x}^2 \in \lbrace 0,1 \rbrace^\mathcal{V}$ be two independent sets defined by their indicator vectors. The respective optimal recombination problem consist in finding an independent set $\mathbf{z} \in \lbrace 0,1 \rbrace^\mathcal{V}$ in $\mathcal{G}$ such that for all $i \in [\lvert \mathcal{V} \rvert]$ such that $x_i^1 = x_i^2$ it also holds $z_i = x_i^1$.

This problem can be posed as a binary labeling problem $((\mathcal{V}', \mathcal{E}'), \boldsymbol{\theta}', \mathcal{Y}'_{v'})$ with $\mathcal{V}' = \lbrace i \in \mathcal{V} : x_i^1 \neq x_i^2 \rbrace$, $\mathcal{E}' = \lbrace ij \in \mathcal{E} : i, j \in \mathcal{V}' \rbrace$, $\mathcal{Y}'_i = \lbrace x_i^1, x_i^2 \rbrace$, $\boldsymbol{\theta}'_u = 0$ and

$$\theta'_{ij}(x_i^1, x_j^2) = \begin{cases} -\infty, & x_i^1 = x_j^2 = 1 \\ 0, & \text{otherwise} \,. \end{cases}$$

The $-\infty$ value in the pairwise costs assures that the optimization result is an independent set, e.g., there is no edge between selected nodes in $\mathcal{E}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.24</span><span class="math-callout__name">(Binary knapsack)</span></p>

Optimal recombination is NOT binary labeling.

</div>

## Chapter 9: Binary Labeling Problems

### Submodular Functions

Let $X$ be a finite set and $2^X$ be its powerset. Mappings of the form $f \colon 2^X \to \mathbb{R}$ are called *set-functions*. Finite sets can be represented by binary indicator vectors with coordinates indexed by elements of $X$. A binary vector $\xi \in \lbrace 0,1 \rbrace^X$ corresponds to the set $A \subset X$ if for all $x \in X$ it holds that

$$\xi_x = \begin{cases} 0, & x \notin A \,, \\ 1, & x \in A \,. \end{cases}$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 9.1</span></p>

Let $(\mathcal{G}, \mathcal{Y}_\mathcal{V}, \theta)$ be a binary graphical model, i.e. $\mathcal{Y}_\mathcal{V} = \lbrace 0,1 \rbrace^\mathcal{V}$. Then the energy function

$$E(y) = \sum_{u \in \mathcal{V}} \theta_u(y_u) + \sum_{uv \in \mathcal{E}} \theta_{uv}(y_u, y_v)$$

can be seen as a set-function w.r.t. $y$, since each labeling $y \in \mathcal{Y}_\mathcal{V}$ is a binary vector.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 9.2</span><span class="math-callout__name">(Submodular Set-Function)</span></p>

A set-function $f \colon 2^X \to \mathbb{R}$ is called *submodular* if

$$f(A) + f(B) \geq f(A \cap B) + f(A \cup B) \quad \forall A, B \in 2^X \,.$$

</div>

Expressing this definition in terms of the corresponding graphical model energy, we obtain

$$E(y) + E(y') \geq E(y \wedge y') + E(y \vee y') \quad \forall y, y' \in \mathcal{Y}_\mathcal{V}$$

with logical *and* $\wedge$ and *or* $\vee$ operations defined as in Chapter ??.

Similarly to convex/concave functions, a function $f$ is called *supermodular*, if $(-f)$ is submodular.

The *submodular minimization problem* consists in computing

$$A^* = \operatorname*{arg\,min}_{A \in 2^X} f(A) \,,$$

where the set-function $f$ is submodular. This problem is known to be polynomially solvable (assuming that the value of $f(A)$ can be computed in polynomial time for any $A \in X$), although the order of the polynomial is quite high in general. However, different subclasses of submodular functions have lower computational complexity. One of such sub-classes, namely, submodular pairwise energy minimization problems, we will consider below.

#### Submodularity on a Lattice

Submodularity can also be defined for multilabel graphical model energies. To do so, we assume that the set of labels in each node is totally ordered, e.g. the relations $\leq$ and $\geq$ are naturally defined for any two labels $s, l \in \mathcal{Y}_u$, $u \in \mathcal{V}$. The set of labelings $\mathcal{Y}_\mathcal{V}$ in this case is partially ordered, where the operation $\geq$ ($\leq$) is defined point-wise, which is $y \leq y'$ for $y, y' \in \mathcal{Y}_\mathcal{V}$, if $y_u \leq y'_u$ for all $u \in \mathcal{V}$. For any two labelings $y$ and $y'$ their node-wise maximum $y \vee y'$ and minimum $y \wedge y'$ are always well-defined. Moreover, the set of labelings $\mathcal{Y}_\mathcal{V}$ is a *lattice*, since it has its *supremum* and *infimum*, i.e. the *highest* labeling $\hat{y}$ such that $\hat{y} \geq y$ for all $y \in \mathcal{Y}_\mathcal{V}$ and the *lowest* one $\check{y}$ such that $\check{y} \leq y$ for all $y \in \mathcal{Y}_\mathcal{V}$.

In general, a non-empty partially ordered set $A$ equipped with operations $\vee$ and $\wedge$ is called a *lattice* if $x \wedge z$ and $x \vee z$ are defined for all $x, z \in A$.

A classical example of a lattice is the *set lattice*, i.e. a family of subsets of a given set with the *subset* relation $\subseteq$ in place of $\leq$ and closed w.r.t. the operations $\cap$ and $\cup$ standing for $\wedge$ and $\vee$. A lattice is called *finite*, if the underlying partially ordered set is finite. Moreover, it is known that any finite lattice is isomorphic to some finite set lattice. This substantiates the generalized definition of submodularity:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 9.3</span><span class="math-callout__name">(Submodularity on a Lattice)</span></p>

Let $(X, \wedge, \vee)$ be a finite lattice. A function $f \colon X \to \mathbb{R}$ is called *submodular*, if

$$f(x) + f(z) \geq f(x \wedge z) + f(x \vee z)$$

holds for any $x, z \in X$.

</div>

For a function to be supermodular, the opposite inequality with the relation $\leq$ has to hold.

Translated to the energy function of a graphical model with a lattice structure submodularity requires that

$$E(y) + E(y') \geq E(y \wedge y') + E(y \vee y')$$

for any two labelings $y, y' \in \mathcal{Y}_\mathcal{V}$.

Although this is a valid definition, it cannot be checked explicitly in practice, as it would require evaluating the relation (9.7) for all possible pairs of labelings. Fortunately, there is no need to do so, due to the following remarkable fact:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.4</span></p>

Let $(\mathcal{G}, \mathcal{Y}_\mathcal{V}, \theta)$ be a graphical model and $E$ be its energy. Let also label sets corresponding to the nodes of the graphical model be totally ordered and operations $\vee$ and $\wedge$ be defined as above. Then $E$ is submodular if and only if all its pairwise cost functions $\theta_{uv} \colon \mathcal{Y}_u \times \mathcal{Y}_v \to \mathbb{R}$, $uv \in \mathcal{E}$, are submodular.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 9.5</span></p>

Submodular functions on a lattice $(W, \min, \max)$ with $W = \lbrace 1, \ldots, n_1 \rbrace \times \cdots \times \lbrace 1, \ldots, n_d \rbrace$ and min and max applied coordinate-wise are equivalent to *Monge arrays*. The latter have a number of applications in optimization.

</div>

#### General Properties of Submodular Functions

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 9.6</span><span class="math-callout__name">(Modular Function)</span></p>

Let $(X, \wedge, \vee)$ be a finite lattice. A function $f \colon X \to \mathbb{R}$ is called *modular*, if it is both sub- and supermodular, i.e.:

$$f(x) + f(x') = f(x \vee x') + f(x \wedge x') \,.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 9.7</span></p>

Let $X$ be a totally ordered set. Then any function $f \colon X \to \mathbb{R}$ defined on this set is modular.

</div>

*Proof.* Consider any $x, x' \in X$. W.l.o.g. assume $x \leq x'$. Then $x \wedge x' = x$ and $x \vee x' = x'$. Therefore $f(x \wedge x') + f(x \vee x') = f(x) + f(x')$, which finalizes the proof. $\square$

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 9.8</span></p>

Let $f$ and $g$ be (sub/super)modular and $\alpha, \beta \geq 0$. Then $\alpha f + \beta g$ is (sub/super)modular.

</div>

The proof follows trivially from the definition of (sub/super)modularity.

In particular, Lemma 9.8 implies, that adding a modular function to a submodular one results in a submodular function, as any modular function is submodular.

### Submodular Pairwise Energies

#### Submodular Functions of Two Variables

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 9.9</span><span class="math-callout__name">(Binary problem)</span></p>

In the binary case, when $\mathcal{Y}_u = \lbrace 0, 1 \rbrace$ for all $u \in \mathcal{V}$, the submodularity condition simplifies to

$$\theta_{uv}(0, 1) + \theta_{uv}(1, 0) \geq \theta_{uv}(0, 0) + \theta_{uv}(1, 1) \,.$$

It is also easy to see that in this case each factor is either submodular or supermodular.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 9.10</span><span class="math-callout__name">(Ising model)</span></p>

Let the energy function be binary and $\theta_{uv}(s, t) = \lambda_{uv} \llbracket s \neq t \rrbracket$ for some constant $\lambda_{uv}$. If $\lambda_{uv} > 0$, the corresponding cost function is submodular, for $\lambda_{uv} < 0$ it is supermodular. It can be shown, see Exercise 9.19, that using reparametrization any pairwise binary energy function can be transformed into the Ising model.

</div>

### Binary Submodular Problems as Min-Cut

We start this section with a definition of the min-$st$-cut problem, which is one of the combinatorial problems having efficient (small) polynomial time algorithms. Later on in this section we will show how submodular MAP-inference problems can be reduced to the min-$st$-cut. Moreover, not only (multilabel) submodular problems are reducible to min-$st$-cut, but also the local polytope relaxation of binary problems allows for such a reduction.

#### Min-$st$-Cut Problem

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 9.12</span><span class="math-callout__name">(Min-$st$-Cut)</span></p>

Let $\mathcal{G}' = (\mathcal{V}', \mathcal{E}')$ be a directed graph, $c \colon \mathcal{E}' \to \mathbb{R}$ be the *weight function of the edges* and $s, t \in \mathcal{V}'$ be two different nodes, which will be called *source* and *target* respectively. An *$st$-cut* $C = (S, T)$ is a partition of $\mathcal{V}'$ into two sets $S$ and $T$ such that

$$S \cap T = \emptyset$$

$$S \cup T = \mathcal{V}'$$

$$s \in S, \; t \in T \,.$$

</div>

The weight of the cut $C$ is the total weight of edges leading from $S$ to $T$:

$$c(S, T) := \sum_{\substack{u \in S, v \in T \\ (u,v) \in \mathcal{E}'}} c_{u,v} \,.$$

The *min-$st$-cut* (or shortly *min-cut*) problem consists in finding a cut with minimal total weight.

As many combinatorial problems, min-cut can be formulated as an integer linear program, see Example 9.13. In general it is NP-hard, just like other ILPs. However, if all weights $c_{u,v}$ are non-negative, its linear programming relaxation is tight and therefore the problem is polynomially solvable. The corresponding linear programming relaxation has a special structure, which allows us to find its minimum with polynomial finite-step algorithms with the worst-case complexity $O(\lvert \mathcal{V}' \rvert^3)$. Typically, the full Lagrange dual of the problem is solved instead of the primal problem. It constitutes the *max-flow* problem, for which a number of efficient polynomial finite-step solvers exist.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 9.13</span><span class="math-callout__name">(ILP representation of the min-$st$-cut and its LP relaxation)</span></p>

The min-$st$-cut problem with non-negative edge weights has the following simple ILP representation:

$$\min_{x \in \lbrace 0,1 \rbrace^{\mathcal{V}' \cup \mathcal{E}'}} \sum_{(u,v) \in \mathcal{E}'} c_{u,v} x_{u,v}$$

$$\text{s.t.} \quad x_u - x_v + x_{u,v} \geq 0 \quad \forall (u,v) \in \mathcal{E}'$$

$$x_s = 0, \; x_t = 1 \,.$$

The coordinates of the binary vector $x$ are indexed by vertices and edges of the graph. The constraint (9.15) essentially states that if $u \in S$ and $v \in T$, the edge connecting them is in the cut. The non-negativity of the weights assures that for $u \in T$ and $v \in S$ the edge variable $x_{u,v}$ is set to zero.

The LP relaxation of (9.14)-(9.16) is obtained as usual, by convexifying the integrality constraints.

</div>

#### Oriented Forms

Although the graphical form of combinatorial problems is of great importance for the development of the field, their algebraic forms have the advantage of a more compact problem representation and help to give unambiguous proofs. Along with the ILP formulation for the min-cut problem, we will consider another algebraic form, the *quadratic pseudo-boolean minimization* problem.

For a binary variable $x \in \lbrace 0, 1 \rbrace$, let $\bar{x}$ denote its negation, that is, $\bar{x} = 1 - x$. Functions $f \colon \lbrace 0,1 \rbrace^n \to \mathbb{R}$ are called *pseudo-boolean*. Note that the class of pseudo-boolean functions coincides with the class of set-functions as defined in §9.1. An important example of pseudo-boolean functions $f(x)$, $x \in \lbrace 0,1 \rbrace^n$, are those representable as a *quadratic* polynomial of $x_i$ and $\bar{x}_i$. A subclass of such functions, tightly related to the min-cut problem, is defined as follows:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 9.14</span><span class="math-callout__name">(Quadratic Pseudo-Boolean Function)</span></p>

Let $\nu$, $\alpha_i$, $\beta_i$ and $\gamma_{i,j}$, $i, j = 1, \ldots, n$ be arbitrary real numbers. For $x \in \lbrace 0,1 \rbrace^n$ a quadratic pseudo-boolean function of the type

$$f(x) = \nu + \sum_{i=1}^{n} \alpha_i x_i + \sum_{i=1}^{n} \beta_i \bar{x}_i + \sum_{i=1}^{n} \sum_{j=1}^{n} \gamma_{ij} \bar{x}_i x_j$$

will be called an *oriented form*.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 9.15</span></p>

The following quadratic pseudo-boolean functions are oriented forms: $-5 + 7x_3\bar{x}_1 + 12x_3 + \bar{x}_1$; $\quad 2\bar{x}_1 x_2 + 3\bar{x}_2 x_1 - 6x_3\bar{x}_4$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 9.16</span></p>

The quadratic pseudo-boolean functions below are *not* oriented forms: $-5 + 7x_3 x_1 + 12x_3 + \bar{x}_1$; $\quad 2x_1 x_2 + 3\bar{x}_2 x_1 - 6x_3\bar{x}_4$.

</div>

#### Min-Cut as Oriented Form

Oriented forms are well suited for representing graph partitions like the one found in the min-cut problem. Let the coordinates of a binary vector $x \in \lbrace 0,1 \rbrace^{\mathcal{V}'}$ be the indicator variables, denoting the subset each vertex belongs to. In other words, if $u \in S$, $x_u = 0$, and, on the other hand, $x_u = 1$ is equivalent to $u \in T$. In this way, the vector $x$ represents a partition of the graph $\mathcal{G}'$.

Let the monomial $c_{u,v}\bar{x}_u x_v$ correspond to the directed edge $(u, v)$ with the weight $c_{u,v}$. Then the total weight of the cut (9.13) corresponding to the vector $x$ is equal to

$$\sum_{(u,v) \in \mathcal{E}'} c_{u,v} \bar{x}_u x_v \,.$$

Note that since $s \in S$ and $t \in T$ it follows that $x_s = 0$ and $x_t = 1$. Therefore, $c_{s,v}\bar{x}_s x_v = c_{s,v} x_v$ and $c_{u,t}\bar{x}_u x_t = c_{u,t}\bar{x}_u$ and $c_{s,t}\bar{x}_s x_t = c_{s,t}$. It also implies that (directed) edges $(u, s)$ and $(t, u)$, $u \in \mathcal{V}$ can be ignored, since they do not belong to any cut and the corresponding monomials vanish.

In conclusion, each min-cut problem can be equivalently represented as minimization of the oriented form

$$c_{s,t} + \sum_{v \colon\, (s,v) \in \mathcal{E}'} c_{s,v} x_v + \sum_{u \colon\, (u,t) \in \mathcal{E}'} c_{u,t} \bar{x}_u + \sum_{\substack{u,v \in \mathcal{E}' \\ u \neq s,\; v \neq t}} c_{u,v} \bar{x}_u x_v$$

with the constant $c_{s,t}$ being zero if the edge $(s, t)$ does not exist. Conversely, minimization of an oriented form can be treated as a min-cut problem. Table 9.1 summarizes the mapping between the min-cut problem and minimization of the oriented forms.

| min-cut | oriented form | $E(y; \theta)$ |
| --- | --- | --- |
| $\mathcal{V}'$ | $x \in \lbrace 0,1 \rbrace^{\mathcal{V}'}$ | $\mathcal{V} = \mathcal{V}' \setminus \lbrace s, t \rbrace$ |
| $u \in S$ | $x_u = 0$ | $y_u = 0$ |
| $u \in T$ | $x_u = 1$ | $y_u = 1$ |
| $c_{u,v}$, $(u,v) \in \mathcal{E}'$ | $c_{u,v}\bar{x}_u x_v$ | $\theta_{uv}(0, 1) = c_{u,v}$ |
| $c_{s,v}$, $(s,v) \in \mathcal{E}'$ | $c_{s,v} x_v$ | $\theta_v(1) = c_{s,v}$ |
| $c_{u,t}$, $(u,t) \in \mathcal{E}'$ | $c_{u,t}\bar{x}_u$ | $\theta_u(0) = c_{u,t}$ |
| $c_{s,t}$, $(s,t) \in \mathcal{E}'$ | $c_{s,t}$ | constant |

In particular, it means that a min-cut problem with non-negative weights corresponds to an oriented form with non-negative coefficients and the other way around.

#### Binary Energy as Oriented Form

To show a close relation between binary MAP-inference and min-cut problems, we show that the former can be equivalently seen as minimization of an oriented form. We will start with an inverse transformation by showing that an oriented form can be seen as a binary graphical model energy up to a constant term.

For each monomial $c_{i,j}\bar{x}_i x_j$ add an edge between nodes $i$ and $j$ to the graph $\mathcal{G}$, should it not exist yet. Set all costs $\theta$ initially to zero. For each monomial $\alpha_i x_i$ assign $\theta_i(1) := \alpha_i$, for $\beta_i \bar{x}_i$ assign $\theta_i(0) := \beta_i$ and for $c_{i,j}\bar{x}_i x_j$ assign $\theta_{ij}(0, 1) := c_{i,j}$.

To specify the inverse transformation of a graphical model to an oriented form note that only pairwise factors having the structure of (9.20), that is every entry is zero except one non-zero value on the diagonal, can be given by an oriented form according to the above construction. In the following we will adopt the name *diagonal form* for such factors. It turns out that any pairwise cost function can be transformed into the diagonal form by reparametrization, given by the formulas below:

$$\phi_{u,v}(0) = \theta_{uv}(1, 0) - \theta_{uv}(0, 0)$$

$$\phi_{v,u}(0) = -\theta_{uv}(1, 0)$$

$$\phi_{v,u}(1) = -\theta_{uv}(1, 1)$$

Note that after the transformation (9.21)-(9.23) the only non-zero pairwise cost value is equal to

$$\theta_{uv}^\phi(0, 1) = \theta_{uv}(0, 1) + \theta_{uv}(1, 0) - \theta_{uv}(1, 1) - \theta_{uv}(0, 0) \,,$$

which is non-negative for submodular pairwise cost functions. To guarantee that all monomials in the oriented form are non-negative, one has to guarantee that the unary costs are non-negative as well. This can easily be achieved by subtracting the minimal value from the unary costs:

$$\forall s \in \mathcal{Y}_u: \quad \theta_u^\phi(s) := \theta_u^\phi(s) - \min_{l \in \mathcal{Y}_u} \theta_u^\phi(l) \,.$$

Since each labeling contains either the label 0 or 1 in each node, such an operation only subtracts a constant value from the energies of all labelings and, therefore, does not affect the optimum of the MAP-inference problem.

#### Equivalence: Binary Energy Minimization $=$ Min-Cut

Summarizing, transformations (9.21)-(9.23) performed for each edge $uv \in \mathcal{E}$ and followed by transformation (9.25) applied to each node $u \in \mathcal{V}$ translate the cost-vector into a form where it can be represented as an oriented form. Moreover, this oriented form has only non-negative coefficients, if the MAP-inference problem is submodular.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 9.17</span></p>

The binary MAP-inference problem reduces to the min-$st$-cut problem and the other way around. Should the costs of the MAP-inference problem be submodular, then it can be reduced to the min-$st$-cut problem with non-negative edge weights. Inversely, such min-cut problems are equivalently representable as binary submodular MAP-inference problems.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 9.18</span></p>

The transformation (9.21)-(9.23) is asymmetric, since $\theta_{uv}(0, 1) \neq 0$ and $\theta_{uv}(1, 0) = 0$. A symmetric transformation can also be found, e.g. such that $\theta_{uv}(0, 1) = \theta_{uv}(1, 0) \neq 0$. Although it can be useful in some cases, the corresponding min-$st$-cut graph would contain two directed edges $(u, v)$ and $(v, u)$ for such a symmetric representation, whereas the diagonal representation (9.20) requires only a single edge. The latter results in a smaller min-$st$-cut graph and faster optimization.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 9.19</span></p>

Show that using reparametrization any pairwise binary energy can be transformed into the form of the Ising model (see Example 9.10). Moreover, submodular pairwise costs are transformed into submodular ones with $\lambda_{uv} \geq 0$, and supermodular into supermodular ones with $\lambda_{uv} \leq 0$.

</div>

### Half-Integrality of Local Polytope

We start with the most important property of the local polytope of binary MAP-inference problems (we will refer to it as *binary local polytope*, although it is not a commonly used term in the literature). This is the characterization of its vertices, which all turn out to have *half-integer* coordinates, i.e. for any vertex $\mu$ of the binary local polytope it holds that $\mu \in \lbrace 0, 1, \frac{1}{2} \rbrace^{\mathcal{I}}$. Recall that $\mathcal{I}$ indexes all labels in all nodes as well as all label pairs in all edges.

For the following statement recall the definition of the set $\mathcal{L}(\phi)$ from (5.123):

$$\mathcal{L}(\phi) = \lbrace \mu \in \mathcal{L} : \mu_w(s) = 0 \text{ if } \theta_w^\phi(s) > \min_{s' \in \mathcal{Y}_w} \theta_w^\phi(s'),\; w \in \mathcal{V} \cup \mathcal{E},\; s \in \mathcal{Y}_w \rbrace \,,$$

which is the set of all vectors in the local polytope such that their non-zero coordinates correspond only to locally minimal entries of the cost vector.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.20</span></p>

Let $(\mathcal{G}, \mathcal{Y}_\mathcal{V}, \theta)$ be a binary graphical model. Let $\phi \in \mathbb{R}^{\mathcal{J}}$ be a reparametrization and $\theta^\phi$ be the corresponding reparametrized costs such that the corresponding arc-consistency closure is non-empty, i.e. $\text{cl}(\text{mi}[\theta^\phi]) \neq \bar{0}$. Then $\mathcal{L}(\phi) \cap \lbrace 0, \frac{1}{2}, 1 \rbrace^{\mathcal{I}} \neq \emptyset$.

In other words, there is a half-integral point in the local polytope, such that its non-zero coordinates correspond to locally minimal labels and label pairs of the cost vector $\theta^\phi$.

</div>

Note that due to Proposition 5.23(5), Theorem 9.20 implies that a non-empty arc-consistency closure is sufficient for dual optimality in case of binary problems.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 9.21</span></p>

All vertices of the binary local polytope have half-integral coordinates.

</div>

*Proof.* Let $\theta$ be a cost vector such that the relaxed problem $\min_{\mu \in \mathcal{L}} \langle \theta, \mu \rangle$ has a unique solution. Let also $\phi$ be an optimal reparametrization, which implies that $\text{cl}(\text{mi}[\theta^\phi]) \neq \bar{0}$. According to Theorem 9.20 the relaxed solution is half-integral. Together with Definition 2.20 this proves the statement. $\square$

### LP Relaxation as Min-Cut

The goal of this section is to reduce the LP relaxation of the binary MAP-inference problem to the min-$st$-cut problem. Such a transformation would imply existence of a polynomial, practically efficient algorithm for the LP relaxation.

**Separability of the local polytope.** As noted in §5.5.1, the local polytope for arbitrary, not only binary problems can be written in the following compact form, see (5.115):

$$\mathcal{L} = \begin{cases} \mu_u \in \Delta^{\mathcal{Y}_u}, & \forall u \in \mathcal{V} \\ \mu_{uv} \in \Delta^{\mathcal{Y}_{uv}}, & \forall uv \in \mathcal{E} \\ \sum_{t \in \mathcal{Y}_v} \mu_{uv}(s, t) = \mu_u(s), & \forall u \in \mathcal{V},\; v \in \mathcal{N}(u),\; s \in \mathcal{Y}_u \,. \end{cases}$$

Let us fix the "unary" vectors $\mu_u$ for all nodes $u \in \mathcal{V}$. Let $\mathcal{L}_{uv}(\mu_u, \mu_v)$ be the set of all "pairwise" vectors $\mu_{uv}$, such that the whole vector $\mu$ belongs to $\mathcal{L}$, i.e.:

$$\mathcal{L}_{uv}(\mu_u, \mu_v) = \left\lbrace \mu_{uv} \in \Delta^{\mathcal{Y}_{uv}} : \sum_{t \in \mathcal{Y}_v} \mu_{uv}(s, t) = \mu_u(s) \quad \forall s \in \mathcal{Y}_u \right\rbrace$$

This separable structure implies that given the "unary" vectors $\mu_u$, $u \in \mathcal{V}$, optimization with respect to the "pairwise" ones decomposes into $\lvert \mathcal{E} \rvert$ small independent optimization problems, one for each pairwise factor:

$$\min_{\mu \in \mathcal{L}} \langle \theta, \mu \rangle = \min_{\substack{\mu_u \in \Delta^{\mathcal{Y}_u}, \\ u \in \mathcal{V}}} \sum_{u \in \mathcal{V}} \langle \theta_u, \mu_u \rangle + \sum_{uv \in \mathcal{E}} \min_{\mu_{uv} \in \mathcal{L}_{uv}(\mu_u, \mu_v)} \langle \theta_{uv}, \mu_{uv} \rangle \,.$$

**Binary energy as oriented form with twice the number of variables.** In general, we will assume that the binary energy is transformed into a quadratic pseudo-boolean function with *non-negative* coefficients. By adding an additional binary variable $z$ such that $y = \bar{z}$ we rewrite each term of the pseudo-boolean function as a symmetric oriented form containing both $y$ and $z$ vectors. As noted above, we assume all coefficients $c_u$, $c_v$ and $c_{u,v}$ to be non-negative.

Let $E$ be an energy and $g(y, z)$ be the oriented form, constructed according to (9.30)-(9.33). By construction, $E(y) = g(y, \bar{y})$, therefore it holds that:

$$\min_{y,z \in \lbrace 0,1 \rbrace^\mathcal{V}} g(y, z) \leq \min_{\substack{y,z \in \lbrace 0,1 \rbrace^\mathcal{V} \\ z = \bar{y}}} g(y, z) = \min_{y \in \lbrace 0,1 \rbrace^\mathcal{V}} E(y) \,.$$

The following theorem relates the first term in (9.34) to the LP relaxation of the energy minimization:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.22</span></p>

Let $(\mathcal{G}, \mathcal{Y}_\mathcal{V}, \theta)$ be a binary MAP-inference problem and its submodular and supermodular pairwise terms be represented in the "diagonal" and "parallel" forms respectively. Let $g$ be constructed according to (9.30)-(9.33). Then it holds that

$$\min_{y,z \in \lbrace 0,1 \rbrace^\mathcal{V}} g(y, z) = \min_{\mu \in \mathcal{L}} \langle \theta, \mu \rangle \,.$$

Moreover, a minimizer $\mu'$ of the right-hand-side can be constructed from a minimizer $(y', z')$ of the left-hand-side as follows:

$$\mu'_u(1) = \begin{cases} y'_u, & y'_u = \bar{z}'_u \\ \frac{1}{2}, & y'_u \neq \bar{z}'_u \,, \end{cases}$$

$$\mu'_u(0) = 1 - \mu'_u(1)$$

$$\mu'_{uv} = \operatorname*{arg\,min}_{\mu_{uv} \in \mathcal{L}_{uv}(\mu'_u, \mu'_v) \cap \lbrace 0, \frac{1}{2}, 1 \rbrace^{\mathcal{Y}_{uv}}} \langle \theta_{uv}, \mu_{uv} \rangle \,,$$

where $\mathcal{L}_{uv}$ is defined by (9.28).

</div>

### Persistency of the Binary Local Polytope

In this section we get back to the question of how integer coordinates of a relaxed solution are related to an exact non-relaxed solution. For the binary local polytope this property indeed holds, as we show below.

First, we will give the definition of the considered property and a sufficient condition for its existence for general graphical models. Afterwards we concentrate on the binary MAP-inference and show that the sufficient condition always holds in this case.

**Persistency definition and criterion for general graphical models**

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 9.23</span><span class="math-callout__name">(Persistency of an Integer Coordinate)</span></p>

Let

$$\mu' \in \operatorname*{arg\,min}_{\mu \in \mathcal{L}} \langle \theta, \mu \rangle$$

be a relaxed solution of the MAP-inference problem for a graphical model $(\mathcal{G}, \mathcal{Y}_\mathcal{V}, \theta)$. For any $u \in \mathcal{V}$ its coordinate $\mu'_u(s)$ is called *(weakly) persistent* or *(weakly) partially optimal*, if (i) $\mu'_u(s) \in \lbrace 0, 1 \rbrace$ and (ii) there exists an exact integer solution

$$\mu^* \in \operatorname*{arg\,min}_{\mu \in \mathcal{L} \cap \lbrace 0,1 \rbrace^{\mathcal{I}}} \langle \theta, \mu \rangle$$

such that $\mu^*_u(s) = \mu'_u(s)$. It is *strongly persistent* if this property holds for *all* exact integer solutions $\mu^*$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 9.24</span><span class="math-callout__name">(Persistency of a Partial Labeling)</span></p>

A partial labeling $y' \in \mathcal{Y}_\mathcal{A}$ on a subset $\mathcal{A} \subseteq \mathcal{V}$ is *(weakly) partially optimal* or *(weakly) persistent* if $y' = y^*\vert_\mathcal{A}$ for some $y^* \in \operatorname{arg\,min}_{y \in \mathcal{Y}_\mathcal{V}} \langle \theta, \delta(y) \rangle$.

</div>

The following proposition formulates a sufficient persistency condition:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 9.25</span><span class="math-callout__name">(Persistency criterion)</span></p>

A partial labeling $y' \in \mathcal{Y}_\mathcal{A}$ is persistent, if for all $\tilde{y} \in \mathcal{Y}_{\mathcal{V} \setminus \mathcal{A}}$ the following holds

$$y' \in \operatorname*{arg\,min}_{y \in \mathcal{Y}_\mathcal{A}} E((y, \tilde{y})) \,,$$

where $(\cdot, \cdot)$ stands for the labeling concatenation, i.e.

$$(y, \tilde{y}) = \begin{cases} y_u, & u \in \mathcal{A} \,, \\ \tilde{y}_u, & \mathcal{V} \setminus \mathcal{A} \,. \end{cases}$$

</div>

*Proof.* Consider the equation

$$\min_{y \in \mathcal{Y}_\mathcal{V}} E(y) = \min_{\tilde{y} \in \mathcal{Y}_{\mathcal{V} \setminus \mathcal{A}}} \min_{y \in \mathcal{Y}_\mathcal{A}} E((y, \tilde{y})) \,.$$

Let $\tilde{y} \in \mathcal{Y}_{\mathcal{V} \setminus \mathcal{A}}$ be such that it leads to a minimal value on the right hand side of (9.43). Then $\tilde{y}$ is part of an optimal solution. By the assumption (9.41), $y'$ is an optimal solution to the inner minimization problem of (9.43), hence $(y', \tilde{y})$ minimizes $E$. $\square$

#### Persistency of the Binary Local Polytope

Let $\mathcal{I}$ be the set of indexes associated with possible labels and label pairs of a binary graphical model. For an arbitrary vector $\mu \in \mathbb{R}^{\mathcal{I}}$ let $\text{Int}(\mu) = \lbrace u \in \mathcal{V} : \mu_u \in \lbrace 0, 1 \rbrace^2 \rbrace$ be the set of nodes corresponding to integer coordinates of $\mu$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 9.26</span></p>

Let $\mu' \in \operatorname{arg\,min}_{\mu \in \mathcal{L}} \langle \theta, \mu \rangle$ be a relaxed solution of a binary MAP-inference problem. Then $\mu'\vert_{\text{Int}(\mu')}$ is persistent.

</div>

*Proof.* The proof is illustrated in Figure 9.8. Let $\mathcal{A} := \text{Int}(\mu')$. Let also $y' \in \mathcal{X}_\mathcal{A}$ be selected such that $\delta(y') = \mu'\vert_\mathcal{A}$. The criterion of Proposition 9.25 is independent of a reparametrization of $\theta$. W.l.o.g. we will assume the latter to be (dual) optimal and such that for all $w \in \mathcal{V} \cup \mathcal{E}$ it holds that

$$\min_{s \in \mathcal{Y}_w} \theta_w(s) = 0 \,.$$

Let us consider the persistency criterion (9.41). For a fixed $\tilde{y} \in \mathcal{Y}_{\mathcal{V} \setminus \mathcal{A}}$ it holds that

$$\operatorname*{arg\,min}_{y \in \mathcal{Y}_\mathcal{A}} E((y, \tilde{y})) = \operatorname*{arg\,min}_{y \in \mathcal{Y}_\mathcal{A}} \left(\underbrace{E_\mathcal{A}(y) + \sum_{\substack{uv \in \mathcal{E}, \\ u \in \mathcal{A}, v \in \mathcal{V} \setminus \mathcal{A}}} \theta_{uv}(y_u, \tilde{y}_v)}_{\geq 0}\right) \,,$$

where the energy $E_\mathcal{A}$ is restricted to the induced subgraph of $\mathcal{A}$ and the inequality holds due to (9.44).

Since $\mu'$ is primal relaxed optimal and $\theta$ is dual optimal, it holds that $E_\mathcal{A}(y') = 0$ due to the strong duality, see Proposition 5.23(4). Moreover, since the solution $\mu'$ outside $\mathcal{A}$ is non-integer, it holds that $\mu'_u(s) > 0$ for all $s \in \mathcal{Y}_u$, $u \in \mathcal{V} \setminus \mathcal{A}$. Therefore, due to the local polytope constraints, $\mu_{uv}(y'_u, y_v) > 0$ for all $u \in \mathcal{A}$, $v \in \mathcal{V} \setminus \mathcal{A}$ with $uv \in \mathcal{E}$ and $y_v \in \mathcal{Y}_v$. Statement 5 of Proposition 5.23 implies $\theta_{uv}(y'_u, \tilde{y}_v) = 0$, in particular $\theta_{uv}(y'_u, \tilde{y}_v) = 0$. Hence, we obtain

$$\left(E_\mathcal{A}(y') + \sum_{\substack{uv \in \mathcal{E}_{\partial \mathcal{A}}, \\ u \in \mathcal{A}}} \theta_{uv}(y'_u, \tilde{y}_v)\right) = 0 \,.$$

Comparing to (9.45) we obtain $y' \in \operatorname{arg\,min}_{y \in \mathcal{Y}_\mathcal{A}} E((y, \tilde{y}))$, which finalizes the proof. $\square$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 9.27</span></p>

An important property of $y'$ follows from the fact that it minimizes (9.45). It reads

$$E((y, \tilde{y})) \geq E((y', \tilde{y})), \quad \forall y \in \mathcal{Y}_{\text{Int}(\mu)} \text{ and } \tilde{y} \in \mathcal{Y}_{\mathcal{V} \setminus \text{Int}(\mu)} \,,$$

where $\mu$ is the relaxed solution. This property constitutes a basis for a powerful class of min-cut-based primal heuristics applicable to general graphical models, which we consider in §9.7.2.

</div>

#### Fusion Moves

The property (9.47) allows algorithms like $\alpha$-expansion (or $\alpha\beta$-swap) to work with arbitrary pairwise potentials. Consider the $\alpha$-expansion Algorithm. Let $\mathcal{L}(y^t, \alpha)$ be the local polytope of the auxiliary binary problem in line ?? of the algorithm, such that each node $u$ contains only two labels, $\alpha$ and $y^t_u$. Let

$$\mu^* \in \operatorname*{arg\,min}_{\mu \in \mathcal{L}(y^t, \alpha)} \langle \theta, \mu \rangle$$

be a solution of such a relaxed binary problem. Construct the new labeling as

$$\forall u \in \mathcal{V} \quad y_u^{t+1} := \begin{cases} \alpha, & \mu^*_u(\alpha) = 1 \,, \\ y_u^t, & \text{otherwise} \,. \end{cases}$$

The property (9.47) implies that $E(y^{t+1}) \leq E(y^t)$, which guarantees monotonicity of the algorithm. Since the new current solution $y^{t+1}$ is "fused" from the old one $y^t$ and the *proposal* consisting of a constant labeling $\alpha$, the modified algorithm is usually referred to as *fusion moves*.

Note several important differences between Algorithm ?? and fusion moves:

- Fusion moves do not require any metric properties of the initial multi-label problem to be applicable, since the key computational step — the relaxed binary problem (9.48) is efficiently solvable for any costs.
- Due to the above property there is no need to restrict the proposal labeling to a constant one. The best results are often obtained when application-specific knowledge is used for proposal generation. For example, for stereo reconstruction of smooth surfaces the labelings corresponding to such surfaces can be used as proposals.
- Similar to $\alpha\beta$-swap, the fusion moves algorithm does not have any optimality guarantees in general, which conforms to the fact that such guarantees can not be obtained by polynomial algorithms for general graphical models. In particular, there is no guarantee that the set of nodes with integer coordinates $\text{Int}(\mu^*)$ is non-empty for the solution of any of the auxiliary problems (9.48).
