---
layout: default
title: Computational Optimal Transport
date: 2024-11-01
# excerpt: Connecting differential equations, stability analysis, and attractor theory to the training dynamics of modern machine learning models.
# tags:
#   - dynamical-systems
#   - machine-learning
#   - theory
---

# Computational Optimal Transport

## Chapter 2: Theoretical Foundations

This chapter describes the basics of optimal transport, introducing first the related notions of optimal matchings and couplings between probability vectors $(\mathbf{a}, \mathbf{b})$, generalizing gradually this computation to transport between discrete measures $(\alpha, \beta)$, to cover lastly the general setting of arbitrary measures.

### Histograms and Measures

We will use interchangeably the terms histogram and probability vector for any element $\mathbf{a} \in \Sigma_n$ that belongs to the probability simplex

$$
\Sigma_n \stackrel{\text{def}}{=} \left\lbrace \mathbf{a} \in \mathbb{R}_+^n \;:\; \sum_{i=1}^n \mathbf{a}_i = 1 \right\rbrace.
$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.1</span><span class="math-callout__name">(Discrete Measures)</span></p>

A discrete measure with weights $\mathbf{a}$ and locations $x_1, \ldots, x_n \in \mathcal{X}$ reads

$$
\alpha = \sum_{i=1}^n \mathbf{a}_i \delta_{x_i},
$$

where $\delta_x$ is the Dirac at position $x$, intuitively a unit of mass which is infinitely concentrated at location $x$. Such a measure describes a probability measure if, additionally, $\mathbf{a} \in \Sigma_n$ and more generally a positive measure if all the elements of vector $\mathbf{a}$ are nonnegative. To avoid degeneracy issues where locations with no mass are accounted for, we will assume when considering discrete measures that all the elements of $\mathbf{a}$ are positive.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.2</span><span class="math-callout__name">(General Measures)</span></p>

A convenient feature of OT is that it can deal with measures that are either or both discrete and continuous within the same framework. To do so, one relies on the set of Radon measures $\mathcal{M}(\mathcal{X})$ on the space $\mathcal{X}$. The formal definition of that set requires that $\mathcal{X}$ is equipped with a distance, usually denoted $d$, because one can access a measure only by "testing" (integrating) it against continuous functions, denoted $f \in \mathcal{C}(\mathcal{X})$.

Integration of $f \in \mathcal{C}(\mathcal{X})$ against a discrete measure $\alpha$ computes a sum

$$
\int_{\mathcal{X}} f(x) \mathrm{d}\alpha(x) = \sum_{i=1}^n \mathbf{a}_i f(x_i).
$$

More general measures, for instance on $\mathcal{X} = \mathbb{R}^d$, can have a density $\mathrm{d}\alpha(x) = \rho_\alpha(x)\mathrm{d}x$ w.r.t. the Lebesgue measure, often denoted $\rho_\alpha = \frac{\mathrm{d}\alpha}{\mathrm{d}x}$, which means that

$$
\forall\, h \in \mathcal{C}(\mathbb{R}^d), \quad \int_{\mathbb{R}^d} h(x)\mathrm{d}\alpha(x) = \int_{\mathbb{R}^d} h(x)\rho_\alpha(x)\mathrm{d}x.
$$

We denote $\mathcal{M}\_+(\mathcal{X})$ the set of all positive measures on $\mathcal{X}$. The set of probability measures is denoted $\mathcal{M}\_+^1(\mathcal{X})$, which means that any $\alpha \in \mathcal{M}\_+^1(\mathcal{X})$ is positive, and that $\alpha(\mathcal{X}) = \int\_\mathcal{X} \mathrm{d}\alpha = 1$.

</div>

### Assignment and Monge Problem

Given a cost matrix $(\mathbf{C}_{i,j})\_{i \in [\![n]\!], j \in [\![m]\!]}$, assuming $n = m$, the optimal assignment problem seeks for a bijection $\sigma$ in the set $\mathrm{Perm}(n)$ of permutations of $n$ elements solving

$$
\min_{\sigma \in \mathrm{Perm}(n)} \frac{1}{n} \sum_{i=1}^n \mathbf{C}_{i,\sigma(i)}.
$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.3</span><span class="math-callout__name">(Uniqueness)</span></p>

Note that the optimal assignment problem may have several optimal solutions. Suppose, for instance, that $n = m = 2$ and that the matrix $\mathbf{C}$ is the pairwise distance matrix between the four corners of a 2-D square of side length 1. In that case only two assignments exist, and they are both optimal.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.4</span><span class="math-callout__name">(Monge Problem Between Discrete Measures)</span></p>

For discrete measures

$$
\alpha = \sum_{i=1}^n \mathbf{a}_i \delta_{x_i} \quad \text{and} \quad \beta = \sum_{j=1}^m \mathbf{b}_j \delta_{y_j},
$$

the Monge problem seeks a map $T : \lbrace x_1, \ldots, x_n \rbrace \to \lbrace y_1, \ldots, y_m \rbrace$ that must verify

$$
\forall\, j \in [\![m]\!], \quad \mathbf{b}_j = \sum_{i : T(x_i) = y_j} \mathbf{a}_i,
$$

which we write in compact form as $T_\sharp \alpha = \beta$. This map should minimize the transportation cost, parameterized by a function $c(x, y)$ defined for points $(x, y) \in \mathcal{X} \times \mathcal{Y}$,

$$
\min_T \left\lbrace \sum_i c(x_i, T(x_i)) \;:\; T_\sharp \alpha = \beta \right\rbrace.
$$

When $n \neq m$, note that, optimality aside, Monge maps may not even exist between a discrete measure to another.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.5</span><span class="math-callout__name">(Push-forward Operator)</span></p>

For a continuous map $T : \mathcal{X} \to \mathcal{Y}$, we define its corresponding push-forward operator $T_\sharp : \mathcal{M}(\mathcal{X}) \to \mathcal{M}(\mathcal{Y})$. For discrete measures, the push-forward operation consists simply in moving the positions of all the points in the support of the measure

$$T_\sharp \alpha \stackrel{\text{def}}{=} \sum_i \mathbf{a}_i \delta_{T(x_i)}$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.1</span><span class="math-callout__name">(Push-forward)</span></p>

For $T : \mathcal{X} \to \mathcal{Y}$, the push-forward measure $\beta = T_\sharp \alpha \in \mathcal{M}(\mathcal{Y})$ of some $\alpha \in \mathcal{M}(\mathcal{X})$ satisfies

$$\forall\, h \in \mathcal{C}(\mathcal{Y}), \quad \int_\mathcal{Y} h(y)\mathrm{d}\beta(y) = \int_\mathcal{X} h(T(x))\mathrm{d}\alpha(x)$$

Equivalently, for any measurable set $B \subset \mathcal{Y}$, one has

$$\beta(B) = \alpha(\lbrace x \in \mathcal{X} \;:\; T(x) \in B \rbrace) = \alpha(T^{-1}(B))$$

Note that $T_\sharp$ preserves positivity and total mass, so that if $\alpha \in \mathcal{M}\_+^1(\mathcal{X})$ then $T_\sharp \alpha \in \mathcal{M}\_+^1(\mathcal{Y})$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.6</span><span class="math-callout__name">(Push-forward for Multivariate Densities)</span></p>

Explicitly doing the change of variables for measures with densities $(\rho_\alpha, \rho_\beta)$ on $\mathbb{R}^d$ (assuming $T$ is smooth and bijective) shows that a push-forward acts on densities as

$$
\rho_\alpha(x) = |\det(T'(x))| \rho_\beta(T(x)),
$$

where $T'(x) \in \mathbb{R}^{d \times d}$ is the Jacobian matrix of $T$. This implies

$$
|\det(T'(x))| = \frac{\rho_\alpha(x)}{\rho_\beta(T(x))}.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.7</span><span class="math-callout__name">(Monge Problem Between Arbitrary Measures)</span></p>

The Monge problem can be extended to the case where two arbitrary probability measures $(\alpha, \beta)$, supported on two spaces $(\mathcal{X}, \mathcal{Y})$ can be linked through a map $T : \mathcal{X} \to \mathcal{Y}$ that minimizes

$$
\min_T \left\lbrace \int_\mathcal{X} c(x, T(x))\mathrm{d}\alpha(x) \;:\; T_\sharp \alpha = \beta \right\rbrace.
$$

The constraint $T_\sharp \alpha = \beta$ means that $T$ pushes forward the mass of $\alpha$ to $\beta$, using the push-forward operator defined in Definition 2.1.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.8</span><span class="math-callout__name">(Push-forward vs. Pull-back)</span></p>

The push-forward $T_\sharp$ of measures should not be confused with the pull-back of functions $T^\sharp : \mathcal{C}(\mathcal{Y}) \to \mathcal{C}(\mathcal{X})$ which corresponds to "warping" between functions, defined as $T^\sharp g = g \circ T$. Push-forward and pull-back are actually adjoint to one another, in the sense that

$$
\forall\, (\alpha, g) \in \mathcal{M}(\mathcal{X}) \times \mathcal{C}(\mathcal{Y}), \quad \int_\mathcal{Y} g\,\mathrm{d}(T_\sharp \alpha) = \int_\mathcal{X} (T^\sharp g)\mathrm{d}\alpha.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.9</span><span class="math-callout__name">(Measures and Random Variables)</span></p>

Radon measures can also be viewed as representing the distributions of random variables. A random variable $X$ on $\mathcal{X}$ is actually a map $X : \Omega \to \mathcal{X}$ from some abstract (often unspecified) probability space $(\Omega, \mathbb{P})$, and its distribution $\alpha$ is the Radon measure $\alpha \in \mathcal{M}\_+^1(\mathcal{X})$ such that $\mathbb{P}(X \in A) = \alpha(A) = \int_A \mathrm{d}\alpha(x)$. Equivalently, it is the push-forward of $\mathbb{P}$ by $X$, $\alpha = X_\sharp \mathbb{P}$. Applying another push-forward $\beta = T_\sharp \alpha$ for $T : \mathcal{X} \to \mathcal{Y}$, is equivalent to defining another random variable $Y = T(X)$, so that $\beta$ is the distribution of $Y$. Drawing a random sample $y$ from $Y$ is thus simply achieved by computing $y = T(x)$, where $x$ is drawn from $X$.

</div>

### Kantorovich Relaxation

The assignment problem, and its generalization found in the Monge problem, is not always relevant to studying discrete measures. Indeed, because the assignment problem is formulated as a permutation problem, it can only be used to compare *uniform* histograms of the *same* size. Additionally, the assignment problem is combinatorial, and the feasible set for the Monge problem, despite being continuously parameterized, is *nonconvex*. Both are therefore difficult to solve when approached in their original formulation.

The key idea of Kantorovich [1942] is to relax the deterministic nature of transportation, namely the fact that a source point $x_i$ can only be assigned to another point or location $y_{\sigma_i}$ or $T(x_i)$ only. Kantorovich proposes instead that the mass at any point $x_i$ be potentially dispatched across several locations. This flexibility is encoded using, in place of a permutation $\sigma$ or a map $T$, a coupling matrix $\mathbf{P} \in \mathbb{R}\_+^{n \times m}$, where $\mathbf{P}_{i,j}$ describes the amount of mass flowing from bin $i$ toward bin $j$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Admissible Couplings)</span></p>

Admissible couplings admit a far simpler characterization than Monge maps,

$$
\mathbf{U}(\mathbf{a}, \mathbf{b}) \stackrel{\text{def}}{=} \left\lbrace \mathbf{P} \in \mathbb{R}_+^{n \times m} \;:\; \mathbf{P}\mathbb{1}_m = \mathbf{a} \quad \text{and} \quad \mathbf{P}^\mathrm{T}\mathbb{1}_n = \mathbf{b} \right\rbrace,
$$

where we used the following matrix-vector notation:

$$
\mathbf{P}\mathbb{1}_m = \left(\sum_j \mathbf{P}_{i,j}\right)_i \in \mathbb{R}^n \quad \text{and} \quad \mathbf{P}^\mathrm{T}\mathbb{1}_n = \left(\sum_i \mathbf{P}_{i,j}\right)_j \in \mathbb{R}^m.
$$

The set of matrices $\mathbf{U}(\mathbf{a}, \mathbf{b})$ is bounded and defined by $n + m$ equality constraints, and therefore is a convex polytope. Additionally, Kantorovich's relaxed formulation is always symmetric, in the sense that a coupling $\mathbf{P}$ is in $\mathbf{U}(\mathbf{a}, \mathbf{b})$ if and only if $\mathbf{P}^\mathrm{T}$ is in $\mathbf{U}(\mathbf{b}, \mathbf{a})$.

</div>

Kantorovich's optimal transport problem now reads

$$
\mathrm{L}_\mathbf{C}(\mathbf{a}, \mathbf{b}) \stackrel{\text{def}}{=} \min_{\mathbf{P} \in \mathbf{U}(\mathbf{a}, \mathbf{b})} \langle \mathbf{C}, \mathbf{P} \rangle \stackrel{\text{def}}{=} \sum_{i,j} \mathbf{C}_{i,j} \mathbf{P}_{i,j}.
$$

This is a linear program, and as is usually the case with such programs, its optimal solutions are not necessarily unique.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.10</span><span class="math-callout__name">(Mines and Factories)</span></p>

The Kantorovich problem finds a very natural illustration in the following resource allocation problem. Suppose that an operator runs $n$ warehouses and $m$ factories. Each warehouse is indexed with an integer $i$ and contains $\mathbf{a}_i$ units of raw material. These raw materials must all be moved to the factories, with a prescribed quantity $\mathbf{b}_j$ needed at factory $j$. To transfer resources from a warehouse $i$ to a factory $j$, the operator can use a transportation company that will charge $\mathbf{C}_{i,j}$ to move a single unit of resource from location $i$ to location $j$. The operator chooses to solve the linear program to obtain a transportation plan $\mathbf{P}^\star$ that quantifies for each pair $i, j$ the amount of goods $\mathbf{P}_{i,j}$ that must be transported from warehouse $i$ to factory $j$. The operator pays on aggregate a total of $\langle \mathbf{P}^\star, \mathbf{C} \rangle$ to the transportation company.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.11</span><span class="math-callout__name">(Kantorovich Problem Between Discrete Measures)</span></p>

For discrete measures $\alpha, \beta$ of the form $\alpha = \sum_i \mathbf{a}_i \delta_{x_i}$, $\beta = \sum_j \mathbf{b}_j \delta_{y_j}$, we store in the matrix $\mathbf{C}$ all pairwise costs between points in the supports of $\alpha, \beta$, namely $\mathbf{C}_{i,j} \stackrel{\text{def}}{=} c(x_i, y_j)$, to define

$$
\mathcal{L}_c(\alpha, \beta) \stackrel{\text{def}}{=} \mathrm{L}_\mathbf{C}(\mathbf{a}, \mathbf{b}).
$$

Therefore, the Kantorovich formulation of optimal transport between discrete measures is the same as the problem between their associated probability weight vectors $\mathbf{a}, \mathbf{b}$ except that the cost matrix $\mathbf{C}$ depends on the support of $\alpha$ and $\beta$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.13</span><span class="math-callout__name">(Kantorovich Problem Between Arbitrary Measures)</span></p>

Definition of $\mathcal{L}_c$ is extended to arbitrary measures by considering couplings $\pi \in \mathcal{M}_+^1(\mathcal{X} \times \mathcal{Y})$ which are joint distributions over the product space. The mass conservation constraint is rewritten as a marginal constraint on joint probability distributions

$$
\mathcal{U}(\alpha, \beta) \stackrel{\text{def}}{=} \left\lbrace \pi \in \mathcal{M}_+^1(\mathcal{X} \times \mathcal{Y}) \;:\; P_{\mathcal{X}\sharp}\pi = \alpha \quad \text{and} \quad P_{\mathcal{Y}\sharp}\pi = \beta \right\rbrace.
$$

Here $P_{\mathcal{X}\sharp}$ and $P_{\mathcal{Y}\sharp}$ are the push-forwards of the projections $P_\mathcal{X}(x,y) = x$ and $P_\mathcal{Y}(x,y) = y$. The Kantorovich problem is then generalized as

$$
\mathcal{L}_c(\alpha, \beta) \stackrel{\text{def}}{=} \min_{\pi \in \mathcal{U}(\alpha,\beta)} \int_{\mathcal{X} \times \mathcal{Y}} c(x,y)\mathrm{d}\pi(x,y).
$$

This is an infinite-dimensional linear program over a space of measures. If $(\mathcal{X}, \mathcal{Y})$ are compact spaces and $c$ is continuous, then it is easy to show that it always has solutions.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.14</span><span class="math-callout__name">(Probabilistic Interpretation)</span></p>

Kantorovich's problem can be reinterpreted through the prism of random variables. Indeed, the problem is equivalent to

$$
\mathcal{L}_c(\alpha, \beta) = \min_{(X,Y)} \left\lbrace \mathbb{E}_{(X,Y)}(c(X,Y)) \;:\; X \sim \alpha, Y \sim \beta \right\rbrace,
$$

where $(X, Y)$ is a couple of random variables over $\mathcal{X} \times \mathcal{Y}$ and $X \sim \alpha$ (resp., $Y \sim \beta$) means that the law of $X$ (resp., $Y$), represented as a measure, must be $\alpha$ (resp., $\beta$). The law of the couple $(X, Y)$ is then $\pi \in \mathcal{U}(\alpha, \beta)$ over the product space $\mathcal{X} \times \mathcal{Y}$.

</div>

**Permutation matrices as couplings.** For a permutation $\sigma \in \mathrm{Perm}(n)$, we write $\mathbf{P}_\sigma$ for the corresponding permutation matrix,

$$
\forall\, (i,j) \in [\![n]\!]^2, \quad (\mathbf{P}_\sigma)_{i,j} = \begin{cases} 1/n & \text{if } j = \sigma_i, \\ 0 & \text{otherwise.} \end{cases}
$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.1</span><span class="math-callout__name">(Kantorovich for Matching)</span></p>

If $m = n$ and $\mathbf{a} = \mathbf{b} = \mathbb{1}_n / n$, then there exists an optimal solution for the Kantorovich problem $\mathbf{P}_{\sigma^\star}$, which is a permutation matrix associated to an optimal permutation $\sigma^\star \in \mathrm{Perm}(n)$ for the assignment problem.

*Proof.* Birkhoff's theorem states that the set of extremal points of $\mathbf{U}(\mathbb{1}_n/n, \mathbb{1}_n/n)$ is equal to the set of permutation matrices. A fundamental theorem of linear programming states that the minimum of a linear objective in a nonempty polyhedron, if finite, is reached at an extremal point of the polyhedron. $\square$

</div>

### Metric Properties of Optimal Transport

An important feature of OT is that it defines a distance between histograms and probability measures as soon as the cost matrix satisfies certain suitable properties. OT can be understood as a canonical way to lift a ground distance between points to a distance between histogram or measures.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.2</span><span class="math-callout__name">(Wasserstein Distance on the Simplex)</span></p>

We suppose $n = m$ and that for some $p \ge 1$, $\mathbf{C} = \mathbf{D}^p = (\mathbf{D}_{i,j}^p)_{i,j} \in \mathbb{R}^{n \times n}$, where $\mathbf{D} \in \mathbb{R}_+^{n \times n}$ is a distance on $[\![n]\!]$, i.e.

1. $\mathbf{D} \in \mathbb{R}_+^{n \times n}$ is symmetric;
2. $\mathbf{D}_{i,j} = 0$ if and only if $i = j$;
3. $\forall\, (i,j,k) \in [\![n]\!]^3$, $\mathbf{D}_{i,k} \le \mathbf{D}_{i,j} + \mathbf{D}_{j,k}$.

Then

$$
\mathrm{W}_p(\mathbf{a}, \mathbf{b}) \stackrel{\text{def}}{=} \mathrm{L}_{\mathbf{D}^p}(\mathbf{a}, \mathbf{b})^{1/p}
$$

defines the $p$-Wasserstein distance on $\Sigma_n$, i.e. $\mathrm{W}_p$ is symmetric, positive, $\mathrm{W}_p(\mathbf{a}, \mathbf{b}) = 0$ if and only if $\mathbf{a} = \mathbf{b}$, and it satisfies the triangle inequality

$$
\forall\, \mathbf{a}, \mathbf{b}, \mathbf{c} \in \Sigma_n, \quad \mathrm{W}_p(\mathbf{a}, \mathbf{c}) \le \mathrm{W}_p(\mathbf{a}, \mathbf{b}) + \mathrm{W}_p(\mathbf{b}, \mathbf{c}).
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.15</span><span class="math-callout__name">(The Cases $0 < p \le 1$)</span></p>

Note that if $0 < p \le 1$, then $\mathbf{D}^p$ is itself distance. This implies that while for $p \ge 1$, $\mathrm{W}_p(\mathbf{a}, \mathbf{b})$ is a distance, in the case $p \le 1$, it is actually $\mathrm{W}_p(\mathbf{a}, \mathbf{b})^p$ which defines a distance on the simplex.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.3</span><span class="math-callout__name">(Wasserstein Distance Between Measures)</span></p>

We assume $\mathcal{X} = \mathcal{Y}$ and that for some $p \ge 1$, $c(x,y) = d(x,y)^p$, where $d$ is a distance on $\mathcal{X}$, i.e.

1. $d(x,y) = d(y,x) \ge 0$;
2. $d(x,y) = 0$ if and only if $x = y$;
3. $\forall\, (x,y,z) \in \mathcal{X}^3$, $d(x,z) \le d(x,y) + d(y,z)$.

Then the $p$-Wasserstein distance on $\mathcal{X}$,

$$
\mathcal{W}_p(\alpha, \beta) \stackrel{\text{def}}{=} \mathcal{L}_{d^p}(\alpha, \beta)^{1/p}
$$

is indeed a distance, namely $\mathcal{W}_p$ is symmetric, nonnegative, $\mathcal{W}_p(\alpha, \beta) = 0$ if and only if $\alpha = \beta$, and it satisfies the triangle inequality

$$
\forall\, (\alpha, \beta, \gamma) \in \mathcal{M}_+^1(\mathcal{X})^3, \quad \mathcal{W}_p(\alpha, \gamma) \le \mathcal{W}_p(\alpha, \beta) + \mathcal{W}_p(\beta, \gamma).
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.18</span><span class="math-callout__name">(Geometric Intuition and Weak Convergence)</span></p>

The Wasserstein distance $\mathcal{W}_p$ has many important properties, the most important being that it is a weak distance, i.e. it allows one to compare singular distributions (for instance, discrete ones) whose supports do not overlap and to quantify the spatial shift between the supports of two distributions. In particular, "classical" distances (or divergences) are not even defined between discrete distributions (the $L^2$ norm can only be applied to continuous measures with a density with respect to a base measure, and the discrete $\ell^2$ norm requires that positions $(x_i, y_j)$ take values in a predetermined discrete set to work properly). In sharp contrast, for any $p > 0$, $\mathcal{W}_p^p(\delta_x, \delta_y) = d(x,y)$. This shows that $\mathcal{W}_p(\delta_x, \delta_y) \to 0$ if $x \to y$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.2</span><span class="math-callout__name">(Weak Convergence)</span></p>

On a compact domain $\mathcal{X}$, $(\alpha_k)_k$ converges weakly to $\alpha$ in $\mathcal{M}\_+^1(\mathcal{X})$ (denoted $\alpha_k \rightharpoonup \alpha$) if and only if for any continuous function 

$$g \in \mathcal{C}(\mathcal{X})$, $\int_\mathcal{X} g\,\mathrm{d}\alpha_k \to \int_\mathcal{X} g\,\mathrm{d}\alpha$$ 

This notion of weak convergence corresponds to the convergence in the law of random vectors.

This convergence can be shown to be equivalent to $\mathcal{W}_p(\alpha_k, \alpha) \to 0$ (together with a convergence of the moments up to order $p$ for unbounded metric spaces).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.19</span><span class="math-callout__name">(Translations)</span></p>

A nice feature of the Wasserstein distance over a Euclidean space $\mathcal{X} = \mathbb{R}^d$ for the ground cost $c(x,y) = \|x - y\|^2$ is that one can factor out translations; indeed, denoting $T_\tau : x \mapsto x - \tau$ the translation operator, one has

$$
\mathcal{W}_2(T_{\tau\sharp}\alpha, T_{\tau'\sharp}\beta)^2 = \mathcal{W}_2(\alpha, \beta)^2 - 2\langle \tau - \tau', \mathbf{m}_\alpha - \mathbf{m}_\beta \rangle + \|\tau - \tau'\|^2,
$$

where 

$$\mathbf{m}_\alpha \stackrel{\text{def}}{=} \int_\mathcal{X} x\,\mathrm{d}\alpha(x) \in \mathbb{R}^d$$

is the mean of $\alpha$. In particular, this implies the nice decomposition of the distance as

$$
\mathcal{W}_2(\alpha, \beta)^2 = \mathcal{W}_2(\tilde{\alpha}, \tilde{\beta})^2 + \|\mathbf{m}_\alpha - \mathbf{m}_\beta\|^2,
$$

where $(\tilde{\alpha}, \tilde{\beta})$ are the "centered" zero mean measures $\tilde{\alpha} = T_{\mathbf{m}_\alpha \sharp}\alpha$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.20</span><span class="math-callout__name">(The Case $p = +\infty$)</span></p>

Informally, the limit of $\mathcal{W}_p^p$ as $p \to +\infty$ is

$$
\mathcal{W}_\infty(\alpha, \beta) \stackrel{\text{def}}{=} \min_{\pi \in \mathcal{U}(\alpha,\beta)} \sup_{(x,y) \in \mathrm{Supp}(\pi)} d(x,y),
$$

where the sup should be understood as the essential supremum according to the measure $\pi$ on $\mathcal{X}^2$. In contrast to the cases $p < +\infty$, this is a nonconvex optimization problem, which is difficult to solve numerically and to study theoretically. The $\mathcal{W}_\infty$ distance is related to the Hausdorff distance between the supports of $(\alpha, \beta)$.

</div>

### Dual Problem

The Kantorovich problem is a constrained convex minimization problem, and as such, it can be naturally paired with a so-called dual problem, which is a constrained concave maximization problem.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.4</span><span class="math-callout__name">(Kantorovich Duality)</span></p>

The Kantorovich problem admits the dual

$$
\mathrm{L}_\mathbf{C}(\mathbf{a}, \mathbf{b}) = \max_{(\mathbf{f}, \mathbf{g}) \in \mathbf{R}(\mathbf{C})} \langle \mathbf{f}, \mathbf{a} \rangle + \langle \mathbf{g}, \mathbf{b} \rangle,
$$

where the set of admissible dual variables is

$$
\mathbf{R}(\mathbf{C}) \stackrel{\text{def}}{=} \left\lbrace (\mathbf{f}, \mathbf{g}) \in \mathbb{R}^n \times \mathbb{R}^m \;:\; \forall\, (i,j) \in [\![n]\!] \times [\![m]\!],\; \mathbf{f} \oplus \mathbf{g} \le \mathbf{C} \right\rbrace.
$$

Such dual variables are often referred to as "Kantorovich potentials."

*Proof.* This result is a direct consequence of strong duality for linear programs. The Lagrangian associated to the Kantorovich problem reads

$$
\min_{\mathbf{P} \ge 0} \max_{(\mathbf{f}, \mathbf{g}) \in \mathbb{R}^n \times \mathbb{R}^m} \langle \mathbf{C}, \mathbf{P} \rangle + \langle \mathbf{a} - \mathbf{P}\mathbb{1}_m, \mathbf{f} \rangle + \langle \mathbf{b} - \mathbf{P}^\mathrm{T}\mathbb{1}_n, \mathbf{g} \rangle.
$$

We exchange the min and the max, which is always possible for linear programs (in finite dimension), to obtain the constraint $\mathbf{C} - \mathbf{f}\mathbb{1}_m^\mathrm{T} - \mathbb{1}_n\mathbf{g}^\mathrm{T} = \mathbf{C} - \mathbf{f} \oplus \mathbf{g} \ge 0$. $\square$

The primal-dual optimality relation allows us to locate the support of the optimal transport plan:

$$
\lbrace (i,j) \in [\![n]\!] \times [\![m]\!] : \mathbf{P}_{i,j} > 0 \rbrace \subset \left\lbrace (i,j) \in [\![n]\!] \times [\![m]\!] \;:\; \mathbf{f}_i + \mathbf{g}_j = \mathbf{C}_{i,j} \right\rbrace.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.21</span><span class="math-callout__name">(Intuitive Interpretation of the Dual)</span></p>

Following the interpretation given to the Kantorovich problem in Remark 2.10, consider an operator wishing to move resources from warehouses to factories at least cost. Suppose he outsources this task to a vendor. The vendor splits the logistic task into collecting (price $\mathbf{f}_i$ at warehouse $i$) and delivering (price $\mathbf{g}_j$ at factory $j$), for a total of $\langle \mathbf{f}, \mathbf{a} \rangle + \langle \mathbf{g}, \mathbf{b} \rangle$.

The operator checks that for all pairs $i, j$ the prices verify $\mathbf{f}\_i + \mathbf{g}\_j \le \mathbf{C}\_{i,j}$. If so, *any* attempt at doing the job by himself would necessarily be more expensive than the vendor's price, since for any transport plan $\mathbf{P}$ (including the optimal one $\mathbf{P}^\star$), the marginal constraints imply

$$
\sum_{i,j} \mathbf{P}_{i,j} \mathbf{C}_{i,j} \ge \sum_{i,j} \mathbf{P}_{i,j}(\mathbf{f}_i + \mathbf{g}_j) = \langle \mathbf{f}, \mathbf{a} \rangle + \langle \mathbf{g}, \mathbf{b} \rangle.
$$

The vendor, wishing to charge as much as possible, maximizes $\langle \mathbf{f}, \mathbf{a} \rangle + \langle \mathbf{g}, \mathbf{b} \rangle$ subject to $\mathbf{f} \oplus \mathbf{g} \le \mathbf{C}$, which results in Problem (2.20). The best price obtained by the vendor is in fact exactly equal to the best possible cost the operator would obtain by computing $\mathrm{L}_\mathbf{C}(\mathbf{a}, \mathbf{b})$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.22</span><span class="math-callout__name">(Dual Problem Between Arbitrary Measures)</span></p>

To extend this primal-dual construction to arbitrary measures, it is important to realize that measures are naturally paired in duality with continuous functions (a measure can be accessed only through integration against continuous functions).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 2.5</span><span class="math-callout__name">(Kantorovich Duality for Measures)</span></p>

One has

$$
\mathcal{L}_c(\alpha, \beta) = \sup_{(f,g) \in \mathcal{R}(c)} \int_\mathcal{X} f(x)\mathrm{d}\alpha(x) + \int_\mathcal{Y} g(y)\mathrm{d}\beta(y),
$$

where the set of admissible dual potentials is

$$
\mathcal{R}(c) \stackrel{\text{def}}{=} \left\lbrace (f, g) \in \mathcal{C}(\mathcal{X}) \times \mathcal{C}(\mathcal{Y}) \;:\; \forall(x,y),\; f(x) + g(y) \le c(x,y) \right\rbrace.
$$

Here, $(f, g)$ is a pair of continuous functions and are also called "Kantorovich potentials." The primal-dual optimality conditions allow us to track the support of the optimal plan:

$$
\mathrm{Supp}(\pi) \subset \lbrace (x,y) \in \mathcal{X} \times \mathcal{Y} \;:\; f(x) + g(y) = c(x,y) \rbrace.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.23</span><span class="math-callout__name">(Unconstrained Dual)</span></p>

In the case $\int_\mathcal{X} \mathrm{d}\alpha = \int_\mathcal{Y} \mathrm{d}\beta = 1$, the constrained dual problem can be replaced by an unconstrained one,

$$
\mathcal{L}_c(\alpha, \beta) = \sup_{(f,g) \in \mathcal{C}(\mathcal{X}) \times \mathcal{C}(\mathcal{Y})} \int_\mathcal{X} f\,\mathrm{d}\alpha + \int_\mathcal{Y} g\,\mathrm{d}\beta + \min_{\mathcal{X} \otimes \mathcal{Y}} (c - f \oplus g),
$$

where we denoted $(f \oplus g)(x,y) = f(x) + g(y)$. Here the minimum should be considered as the essential supremum associated to the measure $\alpha \otimes \beta$, i.e., it does not change if $f$ or $g$ is modified on sets of zero measure for $\alpha$ and $\beta$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.24</span><span class="math-callout__name">(Monge-Kantorovich Equivalence-Brenier Theorem)</span></p>

The following theorem is often attributed to Brenier [1991] and ensures that in $\mathbb{R}^d$ for $p = 2$, if at least one of the two input measures has a density, and for measures with second order moments, then the Kantorovich and Monge problems are equivalent.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.1</span><span class="math-callout__name">(Brenier)</span></p>

In the case $\mathcal{X} = \mathcal{Y} = \mathbb{R}^d$ and $c(x,y) = \|x - y\|^2$, if at least one of the two input measures (denoted $\alpha$) has a density $\rho_\alpha$ with respect to the Lebesgue measure, then the optimal $\pi$ in the Kantorovich formulation is unique and is supported on the graph $(x, T(x))$ of a "Monge map" $T : \mathbb{R}^d \to \mathbb{R}^d$. This means that $\pi = (\mathrm{Id}, T)_\sharp \alpha$, i.e.

$$
\forall\, h \in \mathcal{C}(\mathcal{X} \times \mathcal{Y}), \quad \int_{\mathcal{X} \times \mathcal{Y}} h(x,y)\mathrm{d}\pi(x,y) = \int_\mathcal{X} h(x, T(x))\mathrm{d}\alpha(x).
$$

Furthermore, this map $T$ is uniquely defined as the gradient of a convex function $\varphi$, $T(x) = \nabla\varphi(x)$, where $\varphi$ is the unique (up to an additive constant) convex function such that $(\nabla\varphi)_\sharp \alpha = \beta$. This convex function is related to the dual potential $f$ solving the dual problem as $\varphi(x) = \frac{\|x\|^2}{2} - f(x)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.25</span><span class="math-callout__name">(Monge--Ampère Equation)</span></p>

For measures with densities, using the change of variables formula, one obtains that $\varphi$ is the unique (up to the addition of a constant) convex function which solves the following Monge--Ampère-type equation:

$$
\det(\partial^2 \varphi(x)) \rho_\beta(\nabla\varphi(x)) = \rho_\alpha(x)
$$

where $\partial^2 \varphi(x) \in \mathbb{R}^{d \times d}$ is the Hessian of $\varphi$. The Monge--Ampère operator $\det(\partial^2 \varphi(x))$ can be understood as a nonlinear degenerate Laplacian. In the limit of small displacements, $\varphi = \mathrm{Id} + \varepsilon\psi$, one indeed recovers the Laplacian $\Delta$ as a linearization since for smooth maps

$$
\det(\partial^2 \varphi(x)) = 1 + \varepsilon \Delta \psi(x) + o(\varepsilon).
$$

</div>

### Special Cases

In general, computing OT distances is numerically involved. We first review special favorable cases where the resolution of the OT problem is relatively easy.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.26</span><span class="math-callout__name">(Binary Cost Matrix and 1-norm)</span></p>

One can easily check that when the cost matrix $\mathbf{C}$ is 0 on the diagonal and 1 elsewhere, namely, when $\mathbf{C} = \mathbb{1}_{n \times n} - \mathbb{I}_n$, the 1-Wasserstein distance between $\mathbf{a}$ and $\mathbf{b}$ is equal to the 1-norm of their difference, $\mathrm{L}_\mathbf{C}(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_1$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.27</span><span class="math-callout__name">(Kronecker Cost Function and Total Variation)</span></p>

One can also easily check that this result extends to arbitrary measures in the case where $c(x,y)$ is 0 if $x = y$ and 1 when $x \neq y$. The OT distance between two discrete measures $\alpha$ and $\beta$ is equal to their total variation distance.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.28</span><span class="math-callout__name">(1-D Case---Empirical Measures)</span></p>

Here $\mathcal{X} = \mathbb{R}$. Assuming $\alpha = \frac{1}{n}\sum_{i=1}^n \delta_{x_i}$ and $\beta = \frac{1}{n}\sum_{j=1}^n \delta_{y_j}$, and assuming (without loss of generality) that the points are ordered, i.e. $x_1 \le x_2 \le \cdots \le x_n$ and $y_1 \le y_2 \le \cdots \le y_n$, then one has the simple formula

$$
\mathcal{W}_p(\alpha, \beta)^p = \frac{1}{n} \sum_{i=1}^n |x_i - y_i|^p,
$$

i.e. locally (if one assumes distinct points), $\mathcal{W}_p(\alpha, \beta)$ is the $\ell^p$ norm between two vectors of ordered values of $\alpha$ and $\beta$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.30</span><span class="math-callout__name">(1-D Case---Generic Case)</span></p>

For a measure $\alpha$ on $\mathbb{R}$, we introduce the cumulative distribution function from $\mathbb{R}$ to $[0, 1]$ defined as

$$
\forall\, x \in \mathbb{R}, \quad \mathcal{C}_\alpha(x) \stackrel{\text{def}}{=} \int_{-\infty}^x \mathrm{d}\alpha,
$$

and its pseudoinverse $\mathcal{C}_\alpha^{-1} : [0, 1] \to \mathbb{R} \cup \lbrace -\infty \rbrace$

$$
\forall\, r \in [0,1], \quad \mathcal{C}_\alpha^{-1}(r) = \min_x \left\lbrace x \in \mathbb{R} \cup \lbrace -\infty \rbrace \;:\; \mathcal{C}_\alpha(x) \ge r \right\rbrace.
$$

That function is also called the generalized quantile function of $\alpha$. For any $p \ge 1$, one has

$$
\mathcal{W}_p(\alpha, \beta)^p = \left\|\mathcal{C}_\alpha^{-1} - \mathcal{C}_\beta^{-1}\right\|_{L^p([0,1])}^p = \int_0^1 |\mathcal{C}_\alpha^{-1}(r) - \mathcal{C}_\beta^{-1}(r)|^p \mathrm{d}r.
$$

This means that through the map $\alpha \mapsto \mathcal{C}_\alpha^{-1}$, the Wasserstein distance is isometric to a linear space equipped with the $L^p$ norm. For $p = 1$, one even has the simpler formula

$$
\mathcal{W}_1(\alpha, \beta) = \|\mathcal{C}_\alpha - \mathcal{C}_\beta\|_{L^1(\mathbb{R})} = \int_\mathbb{R} |\mathcal{C}_\alpha(x) - \mathcal{C}_\beta(x)|\mathrm{d}x.
$$

An optimal Monge map $T$ such that $T_\sharp \alpha = \beta$ is then defined by

$$
T = \mathcal{C}_\beta^{-1} \circ \mathcal{C}_\alpha.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.31</span><span class="math-callout__name">(Distance Between Gaussians)</span></p>

If $\alpha = \mathcal{N}(\mathbf{m}_\alpha, \boldsymbol{\Sigma}_\alpha)$ and $\beta = \mathcal{N}(\mathbf{m}_\beta, \boldsymbol{\Sigma}_\beta)$ are two Gaussians in $\mathbb{R}^d$, then one can show that the optimal transport map is

$$
T : x \mapsto \mathbf{m}_\beta + A(x - \mathbf{m}_\alpha),
$$

where $A = \boldsymbol{\Sigma}_\alpha^{-1/2}\left(\boldsymbol{\Sigma}_\alpha^{1/2}\boldsymbol{\Sigma}_\beta\boldsymbol{\Sigma}_\alpha^{1/2}\right)^{1/2}\boldsymbol{\Sigma}_\alpha^{-1/2} = A^\mathrm{T}$. The transport cost of that map is

$$
\mathcal{W}_2^2(\alpha, \beta) = \|\mathbf{m}_\alpha - \mathbf{m}_\beta\|^2 + \mathcal{B}(\boldsymbol{\Sigma}_\alpha, \boldsymbol{\Sigma}_\beta)^2,
$$

where $\mathcal{B}$ is the so-called Bures metric between positive definite matrices,

$$
\mathcal{B}(\boldsymbol{\Sigma}_\alpha, \boldsymbol{\Sigma}_\beta)^2 \stackrel{\text{def}}{=} \mathrm{tr}\left(\boldsymbol{\Sigma}_\alpha + \boldsymbol{\Sigma}_\beta - 2(\boldsymbol{\Sigma}_\alpha^{1/2}\boldsymbol{\Sigma}_\beta\boldsymbol{\Sigma}_\alpha^{1/2})^{1/2}\right).
$$

One can show that $\mathcal{B}$ is a distance on covariance matrices and that $\mathcal{B}^2$ is convex with respect to both its arguments. In the case where $\boldsymbol{\Sigma}_\alpha = \mathrm{diag}(r_i)_i$ and $\boldsymbol{\Sigma}_\beta = \mathrm{diag}(s_i)_i$ are diagonal, the Bures metric is the Hellinger distance

$$
\mathcal{B}(\boldsymbol{\Sigma}_\alpha, \boldsymbol{\Sigma}_\beta) = \|\sqrt{r} - \sqrt{s}\|_2.
$$

For 1-D Gaussians, $\mathcal{W}_2$ is thus the Euclidean distance on the 2-D plane plotting the mean and the standard deviation of a Gaussian $(\mathbf{m}, \sqrt{\Sigma})$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.32</span><span class="math-callout__name">(Distance Between Elliptically Contoured Distributions)</span></p>

Gelbrich provides a more general result than that provided in Remark 2.31: the Bures metric between Gaussians extends more generally to *elliptically contoured distributions*. In a nutshell, one can first show that for two measures with given mean and covariance matrices, the distance between the two Gaussians with these respective parameters is a lower bound of the Wasserstein distance between the two measures. Additionally, the closed form for the Wasserstein distance extends to families of elliptically contoured densities: if two densities $\rho_\alpha$ and $\rho_\beta$ belong to such a family, namely when $\rho_\alpha$ and $\rho_\beta$ can be written for any point $x$ using a mean, positive definite parameter, and the same nonnegative valued function $h$ such that $\int_{\mathbb{R}^d} h(\langle x, x \rangle)\mathrm{d}x = 1$, then their optimal transport map is also the linear map and their Wasserstein distance is also given by the expression involving the Bures metric, with a slightly different scaling that depends only on the generator function $h$.

</div>

## Chapter 3: Algorithmic Foundations

This chapter describes the most common algorithmic tools from combinatorial optimization and linear programming that can be used to solve the discrete formulation of optimal transport, as described in the primal problem (2.11) or alternatively its dual (2.20).

### The Kantorovich Linear Programs

We have already introduced the primal OT problem:

$$
\mathrm{L}_\mathbf{C}(\mathbf{a}, \mathbf{b}) = \min_{\mathbf{P} \in \mathbf{U}(\mathbf{a}, \mathbf{b})} \sum_{i \in [\![n]\!], j \in [\![m]\!]} \mathbf{C}_{i,j} \mathbf{P}_{i,j}.
$$

To make the link with the linear programming literature, one can cast the equation above as a linear program in *standard* form. Let $\mathbb{I}_n$ stand for the identity matrix of size $n$ and let $\otimes$ be Kronecker's product. The $(n+m) \times nm$ matrix

$$
\mathbf{A} = \begin{bmatrix} \mathbb{1}_n^\mathrm{T} \otimes \mathbb{I}_m \\ \mathbb{I}_n \otimes \mathbb{1}_m^\mathrm{T} \end{bmatrix} \in \mathbb{R}^{(n+m) \times nm}
$$

can be used to encode the row-sum and column-sum constraints. Simply cast a matrix $\mathbf{P} \in \mathbb{R}^{n \times m}$ as a vector $\mathbf{p} \in \mathbb{R}^{nm}$ (enumerated columnwise) to obtain the equivalence:

$$
\mathbf{P} \in \mathbb{R}^{n \times m} \in \mathbf{U}(\mathbf{a}, \mathbf{b}) \Leftrightarrow \mathbf{p} \in \mathbb{R}_+^{nm},\; \mathbf{A}\mathbf{p} = \begin{bmatrix} \mathbf{a} \\ \mathbf{b} \end{bmatrix}.
$$

Therefore we can write the original optimal transport problem as

$$
\mathrm{L}_\mathbf{C}(\mathbf{a}, \mathbf{b}) = \min_{\substack{\mathbf{p} \in \mathbb{R}_+^{nm} \\ \mathbf{A}\mathbf{p} = [\mathbf{a};\, \mathbf{b}]}} \mathbf{c}^\mathrm{T}\mathbf{p},
$$

where the $nm$-dimensional vector $\mathbf{c}$ is equal to the stacked columns contained in the cost matrix $\mathbf{C}$. The dual problem corresponding to this is

$$
\mathrm{L}_\mathbf{C}(\mathbf{a}, \mathbf{b}) = \max_{\substack{\mathbf{h} \in \mathbb{R}^{n+m} \\ A^\mathrm{T}\mathbf{h} \le \mathbf{c}}} \begin{bmatrix} \mathbf{a} \\ \mathbf{b} \end{bmatrix}^\mathrm{T} \mathbf{h}.
$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 3.1</span><span class="math-callout__name">(Redundant Constraints)</span></p>

Note that one of the $n + m$ constraints described above is redundant, or that, in other words, the line vectors of matrix $A$ are not linearly independent. Indeed, summing all $n$ first lines and the subsequent $m$ lines results in the same vector. One can show that removing a line in $A$ and the corresponding entry in $[\mathbf{a};\, \mathbf{b}]$ yields a properly defined linear system. For simplicity, we retain in what follows a redundant formulation, keeping in mind that degeneracy will pop up in some of our computations.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 3.2</span><span class="math-callout__name">(Derivation of Duality)</span></p>

We provide a simple derivation of the duality result above. Write $\mathbf{q} = [\mathbf{a};\, \mathbf{b}]$. Consider a relaxed primal problem of the optimal transport problem, where the constraint $\mathbf{A}\mathbf{p} = \mathbf{q}$ is no longer necessarily enforced but bears instead a cost $\mathbf{h}^\mathrm{T}(\mathbf{A}\mathbf{p} - \mathbf{q})$ parameterized by an arbitrary vector of costs $\mathbf{h} \in \mathbb{R}^{n+m}$:

$$
\mathrm{H}(\mathbf{h}) \stackrel{\text{def}}{=} \min_{\mathbf{p} \in \mathbb{R}_+^{nm}} \mathbf{c}^\mathrm{T}\mathbf{p} - \mathbf{h}^\mathrm{T}(\mathbf{A}\mathbf{p} - \mathbf{q}).
$$

Since $\mathrm{H}(\mathbf{h}) \le \mathrm{L}_\mathbf{C}(\mathbf{a}, \mathbf{b})$ for any cost vector $\mathbf{h}$, the goal of duality theory is to compute the best lower bound $\underline{z}$ by maximizing $\mathrm{H}$ over *any* cost vector $\mathbf{h}$:

$$
\underline{z} = \max_{\mathbf{h}} \left( \mathrm{H}(\mathbf{h}) = \max_{\mathbf{h}}\, \mathbf{h}^\mathrm{T}\mathbf{q} + \min_{\mathbf{p} \in \mathbb{R}_+^{nm}} (\mathbf{c} - A^\mathrm{T}\mathbf{h})^\mathrm{T}\mathbf{p} \right).
$$

The minimization on $\mathbf{p}$ is $-\infty$ if any coordinate of $\mathbf{c}^\mathrm{T} - A^\mathrm{T}\mathbf{h}$ is negative. Therefore the best possible lower bound becomes

$$
\underline{z} = \max_{\substack{\mathbf{h} \in \mathbb{R}^{n+m} \\ A^\mathrm{T}\mathbf{h} \le \mathbf{c}}} \mathbf{h}^\mathrm{T}\mathbf{q}.
$$

We have therefore proved a weak duality result, namely that $\underline{z} \le \bar{z}$.

</div>

### C-Transforms

We present an important property of the dual optimal transport problem which takes a more important meaning when used for the semidiscrete optimal transport problem. This section builds upon the original formulation that splits dual variables according to row and column sum constraints:

$$
\mathrm{L}_\mathbf{C}(\mathbf{a}, \mathbf{b}) = \max_{(\mathbf{f}, \mathbf{g}) \in \mathbf{R}(\mathbf{C})} \langle \mathbf{f}, \mathbf{a} \rangle + \langle \mathbf{g}, \mathbf{b} \rangle.
$$

Consider any dual feasible pair $(\mathbf{f}, \mathbf{g})$. If we "freeze" the value of $\mathbf{f}$, the best vector solution for $\mathbf{g}$ is the $\mathbf{C}$-transform of $\mathbf{f}$, denoted $\mathbf{f}^\mathbf{C} \in \mathbb{R}^m$ and defined as

$$
(\mathbf{f}^\mathbf{C})_j = \min_{i \in [\![n]\!]} \mathbf{C}_{ij} - \mathbf{f}_i.
$$

Similarly, if we "freeze" the values of $\mathbf{g}$ and consider instead the $\bar{\mathbf{C}}$-transform of $\mathbf{g}$, namely vector $\mathbf{g}^{\bar{\mathbf{C}}} \in \mathbb{R}^n$ defined as

$$
(\mathbf{g}^{\bar{\mathbf{C}}})_i = \min_{j \in [\![m]\!]} \mathbf{C}_{ij} - \mathbf{g}_j.
$$

This allows us to reformulate the dual problem as a piecewise affine concave maximization problem expressed in a single variable $\mathbf{f}$ as

$$
\mathrm{L}_\mathbf{C}(\mathbf{a}, \mathbf{b}) = \max_{\mathbf{f} \in \mathbb{R}^n} \langle \mathbf{f}, \mathbf{a} \rangle + \langle \mathbf{f}^\mathbf{C}, \mathbf{b} \rangle.
$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.1</span><span class="math-callout__name">(Properties of C-Transforms)</span></p>

The following identities, in which the inequality sign between vectors should be understood elementwise, hold:

1. $\mathbf{f} \le \mathbf{f}' \Rightarrow \mathbf{f}^\mathbf{C} \ge \mathbf{f}'{}^\mathbf{C}$,
2. $\mathbf{f}^{\mathbf{C}\bar{\mathbf{C}}} \ge \mathbf{f}$, $\;\mathbf{g}^{\bar{\mathbf{C}}\mathbf{C}} \ge \mathbf{g}$,
3. $\mathbf{f}^{\mathbf{C}\bar{\mathbf{C}}\mathbf{C}} = \mathbf{f}^\mathbf{C}$.

*Proof.* The first inequality follows from the definition of $\mathbf{C}$-transforms. For (ii), expanding the definition of $\mathbf{f}^{\mathbf{C}\bar{\mathbf{C}}}$ we have

$$
\left(\mathbf{f}^{\mathbf{C}\bar{\mathbf{C}}}\right)_i = \min_{j \in [\![m]\!]} \mathbf{C}_{ij} - \mathbf{f}_j^\mathbf{C} = \min_{j \in [\![m]\!]} \mathbf{C}_{ij} - \min_{i' \in [\![n]\!]} \mathbf{C}_{i'j} - \mathbf{f}_{i'}.
$$

Since $-\min_{i' \in [\![n]\!]} \mathbf{C}_{i'j} - \mathbf{f}_{i'} \ge -(\mathbf{C}_{ij} - \mathbf{f}_i)$, we recover $\left(\mathbf{f}^{\mathbf{C}\bar{\mathbf{C}}}\right)_i \ge \mathbf{f}_i$.

For (iii), set $\mathbf{g} = \mathbf{f}^\mathbf{C}$. Then $\mathbf{g}^{\bar{\mathbf{C}}} = \mathbf{f}^{\mathbf{C}\bar{\mathbf{C}}} \ge \mathbf{f}$. Therefore, using result (i) we have $\mathbf{f}^{\mathbf{C}\bar{\mathbf{C}}\mathbf{C}} \le \mathbf{f}^\mathbf{C}$. Result (ii) yields $\mathbf{f}^{\mathbf{C}\bar{\mathbf{C}}\mathbf{C}} \ge \mathbf{f}^\mathbf{C}$, proving the equality. $\square$

</div>

### Complementary Slackness

Primal and dual problems can be solved independently to obtain optimal primal $\mathbf{P}^\star$ and dual $(\mathbf{f}^\star, \mathbf{g}^\star)$ solutions.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.1</span><span class="math-callout__name">(Complementary Variables)</span></p>

A matrix $\mathbf{P} \in \mathbb{R}^{n \times m}$ and a pair of vectors $(\mathbf{f}, \mathbf{g})$ are complementary w.r.t. $\mathbf{C}$ if for all pairs of indices $(i, j)$ such that $\mathbf{P}_{i,j} > 0$ one also has $\mathbf{C}_{i,j} = \mathbf{f}_i + \mathbf{g}_j$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.2</span><span class="math-callout__name">(Complementary Slackness)</span></p>

Let $\mathbf{P}^\star$ and $\mathbf{f}^\star, \mathbf{g}^\star$ be optimal solutions for the primal and dual problems, respectively. Then, for any pair $(i,j) \in [\![n]\!] \times [\![m]\!]$, $\mathbf{P}_{i,j}^\star(\mathbf{C}_{i,j} - \mathbf{f}_i^\star + \mathbf{g}_j^\star) = 0$ holds. In other words, if $\mathbf{P}_{i,j}^\star > 0$, then necessarily $\mathbf{f}_i^\star + \mathbf{g}_j^\star = \mathbf{C}_{i,j}$; if $\mathbf{f}_i^\star + \mathbf{g}_j^\star < \mathbf{C}_{i,j}$ then necessarily $\mathbf{P}_{i,j}^\star = 0$.

*Proof.* We have by strong duality that $\langle \mathbf{P}^\star, \mathbf{C} \rangle = \langle \mathbf{f}^\star, \mathbf{a} \rangle + \langle \mathbf{g}^\star, \mathbf{b} \rangle$. Recall that $\mathbf{P}^\star \mathbb{1}_m = \mathbf{a}$ and $\mathbf{P}^{\star\mathrm{T}}\mathbb{1}_n = \mathbf{b}$; therefore

$$
\langle \mathbf{f}^\star, \mathbf{a} \rangle + \langle \mathbf{g}^\star, \mathbf{b} \rangle = \langle \mathbf{f}^\star, \mathbf{P}^\star \mathbb{1}_m \rangle + \langle \mathbf{g}^\star, \mathbf{P}^{\star\mathrm{T}}\mathbb{1}_n \rangle = \langle \mathbf{f}^\star \mathbb{1}_m^\mathrm{T}, \mathbf{P}^\star \rangle + \langle \mathbb{1}_n \mathbf{g}^{\star\mathrm{T}}, \mathbf{P}^\star \rangle,
$$

which results in $\langle \mathbf{P}^\star, \mathbf{C} - \mathbf{f}^\star \oplus \mathbf{g}^\star \rangle = 0$. Because $(\mathbf{f}^\star, \mathbf{g}^\star)$ belongs to the polyhedron of dual constraints, each entry of the matrix $\mathbf{C} - \mathbf{f}^\star \oplus \mathbf{g}^\star$ is necessarily nonnegative. Therefore, since all the entries of $\mathbf{P}$ are nonnegative, the constraint that the dot-product above is equal to 0 enforces the result. $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.3</span><span class="math-callout__name">(Optimality from Complementary Slackness)</span></p>

If $\mathbf{P}$ and $(\mathbf{f}, \mathbf{g})$ are complementary and feasible solutions for the primal and dual problems, respectively, then $\mathbf{P}$ and $(\mathbf{f}, \mathbf{g})$ are both primal and dual optimal.

*Proof.* By weak duality, we have that

$$
\mathrm{L}_\mathbf{C}(\mathbf{a}, \mathbf{b}) \le \langle \mathbf{P}, \mathbf{C} \rangle = \langle \mathbf{P},\, \mathbf{f} \oplus \mathbf{g} \rangle = \langle \mathbf{a}, \mathbf{f} \rangle + \langle \mathbf{b}, \mathbf{g} \rangle \le \mathrm{L}_\mathbf{C}(\mathbf{a}, \mathbf{b})
$$

and therefore $\mathbf{P}$ and $(\mathbf{f}, \mathbf{g})$ are respectively primal and dual optimal. $\square$

</div>

### Vertices of the Transportation Polytope

Recall that a vertex or an extremal point of a convex set is formally a point $\mathbf{x}$ in that set such that, if there exist $\mathbf{y}$ and $\mathbf{z}$ in that set with $\mathbf{x} = (\mathbf{y} + \mathbf{z})/2$, then necessarily $\mathbf{x} = \mathbf{y} = \mathbf{z}$. A linear program with a nonempty and bounded feasible set attains its minimum at a vertex of the feasible set. Since the feasible set $\mathbf{U}(\mathbf{a}, \mathbf{b})$ of the primal optimal transport problem is bounded, one can restrict the search for an optimal $\mathbf{P}$ to the set of extreme points of the polytope $\mathbf{U}(\mathbf{a}, \mathbf{b})$.

#### Tree Structure of the Support of All Vertices of $\mathbf{U}(\mathbf{a}, \mathbf{b})$

Let $V = (1, 2, \ldots, n)$ and $V' = (1', 2', \ldots, m')$ be two sets of nodes. Consider their union $V \cup V'$, with $n + m$ nodes, and the set $\mathcal{E}$ of all $nm$ directed edges $\lbrace(i, j'), i \in [\![n]\!], j \in [\![m]\!]\rbrace$ between them. To each edge $(i, j')$ we associate the corresponding cost value $\mathbf{C}_{ij}$. The complete bipartite graph $\mathcal{G}$ between $V$ and $V'$ is $(V \cup V', E)$. A transport plan is a flow on that graph satisfying source ($\mathbf{a}_i$ flowing out of each node $i$) and sink ($\mathbf{b}_j$ flowing into each node $j'$) constraints.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.4</span><span class="math-callout__name">(Extremal Solutions)</span></p>

Let $\mathbf{P}$ be an extremal point of the polytope $\mathbf{U}(\mathbf{a}, \mathbf{b})$. Let $S(\mathbf{P}) \subset \mathcal{E}$ be the subset of edges $\lbrace (i, j'), i \in [\![n]\!], j \in [\![m]\!] \rbrace$ such that $\mathbf{P}_{ij} > 0$. Then the graph $G(\mathbf{P}) \stackrel{\text{def}}{=} (V \cup V', S(\mathbf{P}))$ has no cycles. In particular, $\mathbf{P}$ cannot have more than $n + m - 1$ nonzero entries.

*Proof.* We proceed by contradiction. Suppose that $\mathbf{P}$ is an extremal point and that its support $S(\mathbf{P})$ contains a cycle. Then we can construct two feasible matrices $\mathbf{Q}$ and $\mathbf{R}$ such that $\mathbf{P} = (\mathbf{Q} + \mathbf{R})/2$ by considering a directed cycle and adding/subtracting a perturbation $\varepsilon < \min_{(i,j') \in S(\mathbf{P})} \mathbf{P}_{ij}$ along the cycle. Since $\mathbf{Q}, \mathbf{R} \neq \mathbf{P}$, this contradicts the fact that $\mathbf{P}$ is an extremal point. Since a graph with $k$ nodes and no cycles cannot have more than $k - 1$ edges, $\mathbf{P}$ cannot have more than $n + m - 1$ nonzero entries. $\square$

</div>

#### The North-West Corner Rule

The north-west (NW) corner rule is a heuristic that produces a vertex of the polytope $\mathbf{U}(\mathbf{a}, \mathbf{b})$ in up to $n + m$ operations. This heuristic can play a role in initializing any algorithm working on the primal, such as the network simplex.

The rule starts by giving the highest possible value to $\mathbf{P}_{1,1}$ by setting it to $\min(\mathbf{a}_1, \mathbf{b}_1)$. At each step, the entry $\mathbf{P}_{i,j}$ is chosen to saturate either the row constraint at $i$, the column constraint at $j$, or both if possible. The indices $i, j$ are then updated as follows: $i$ is incremented in the first case, $j$ is in the second, and both $i$ and $j$ are in the third case. The rule proceeds until $\mathbf{P}_{n,m}$ has received a value. We write $\mathbf{NW}(\mathbf{a}, \mathbf{b})$ for the unique plan that can be obtained through this heuristic.

### A Heuristic Description of the Network Simplex

Consider a feasible matrix $\mathbf{P}$ whose graph $G(\mathbf{P}) = (V \cup V', S(\mathbf{P}))$ has no cycles. $\mathbf{P}$ has therefore no more than $n + m - 1$ nonzero entries and is a vertex of $\mathbf{U}(\mathbf{a}, \mathbf{b})$ by Proposition 3.4. Following Proposition 3.3, it is therefore sufficient to obtain a dual solution $(\mathbf{f}, \mathbf{g})$ which is feasible (i.e. $\mathbf{C} - \mathbf{f} \oplus \mathbf{g}$ has nonnegative entries) and complementary to $\mathbf{P}$ (pairs of indices $(i, j')$ in $S(\mathbf{P})$ are such that $\mathbf{C}_{i,j} = \mathbf{f}_i + \mathbf{g}_j$), to prove that $\mathbf{P}$ is optimal. The network simplex relies on two simple principles: to each feasible primal solution $\mathbf{P}$ one can associate a complementary pair $(\mathbf{f}, \mathbf{g})$. If that pair is feasible, then we have reached optimality. If not, one can consider a modification of $\mathbf{P}$ that remains feasible and whose complementary pair $(\mathbf{f}, \mathbf{g})$ is modified so that it becomes closer to feasibility.

#### Obtaining a Dual Pair Complementary to $\mathbf{P}$

The simplex proceeds by associating first to any extremal solution $\mathbf{P}$ a pair of $(\mathbf{f}, \mathbf{g})$ complementary dual variables. This is simply carried out by finding two vectors $\mathbf{f}$ and $\mathbf{g}$ such that for any $(i, j')$ in $S(\mathbf{P})$, $\mathbf{f}_i + \mathbf{g}_j$ is equal to $\mathbf{C}_{i,j}$. Since $\mathbf{P}$ is extremal, $G(\mathbf{P})$ has no cycles, so $G(\mathbf{P})$ is either a tree or a forest (a union of trees). Since $s \le n + m - 1 < n + m$, the linear system is always undetermined. We can choose arbitrarily a root node in each tree and assign the value 0 to its corresponding dual variable. From there, we can traverse the tree using a breadth-first or depth-first search to obtain a sequence of simple variable assignments that determines the values of all other dual variables in that tree.

#### Network Simplex Update

The dual pair $(\mathbf{f}, \mathbf{g})$ obtained previously might be feasible, in the sense that for all $i, j$ we have $\mathbf{f}_i + \mathbf{g}_j \le \mathbf{C}_{i,j}$, in which case we have reached the optimum by Proposition 3.3. When that is not the case, namely when there exists $i, j$ such that $\mathbf{f}_i + \mathbf{g}_j > \mathbf{C}_{i,j}$, the network simplex algorithm kicks in. We first initialize a graph $G$ to be equal to $G(\mathbf{P})$ and add the violating edge $(i, j')$ to $G$. Two cases can arise:

- (a) $G$ is still a forest, which can happen if $(i, j')$ links two existing subtrees. The approach outlined above can be used on graph $G$ to recover a new complementary dual vector $(\mathbf{f}, \mathbf{g})$.
- (b) $G$ now has a cycle. In that case, we need to remove an edge in $G$ to ensure that $G$ is still a forest, yet also modify $\mathbf{P}$ so that $\mathbf{P}$ is feasible and $G(\mathbf{P})$ remains included in $G$. We increase the flow of all "positive" edges and decrease that of "negative" edges in the cycle by the largest possible amount $\theta$, controlled by the smallest flow negatively impacted by the cycle. We then close the update by removing the edge achieving the minimum from $G$.

#### Improvement of the Primal Solution

One can show that the manipulation above can only improve the cost of $\mathbf{P}$. When a cycle is created and $\mathbf{P}$ is updated to $\tilde{\mathbf{P}}$, the following equality holds:

$$
\langle \tilde{\mathbf{P}}, \mathbf{C} \rangle - \langle \mathbf{P}, \mathbf{C} \rangle = \theta \left(\sum_{k=1}^l \mathbf{C}_{i_k, j_k} - \sum_{k=1}^l \mathbf{C}_{i_{k+1}, j_k}\right) = \theta\left(\mathbf{C}_{i,j} - (\mathbf{f}_i + \mathbf{g}_j)\right).
$$

That term is, by definition, negative, since $i, j$ were chosen because $\mathbf{C}_{i,j} < \mathbf{f}_i + \mathbf{g}_j$. Therefore, if $\theta > 0$, we have that $\langle \tilde{\mathbf{P}}, \mathbf{C} \rangle < \langle \mathbf{P}, \mathbf{C} \rangle$.

The network simplex algorithm can therefore be summarized as follows: Initialize the algorithm with an extremal solution $\mathbf{P}$, given for instance by the NW corner rule. Initialize the graph $G$ with $G(\mathbf{P})$. Compute a pair of dual variables $(\mathbf{f}, \mathbf{g})$ that are complementary to $\mathbf{P}$ using the tree structure(s) in $G$. (i) Look for a violating pair of indices to the constraint $\mathbf{C} - \mathbf{f} \oplus \mathbf{g} \ge 0$; if none, $\mathbf{P}$ is optimal and stop. If there is a violating pair $(i, j')$, (ii) add the edge $(i, j')$ to $G$. If $G$ still has no cycles, update $(\mathbf{f}, \mathbf{g})$ accordingly; if there is a cycle, direct it making sure $(i, j')$ is labeled as positive, and remove a negative edge in that cycle with the smallest flow value, updating $\mathbf{P}, G$; then build a complementary pair $\mathbf{f}, \mathbf{g}$ accordingly; return to (i).

Orlin [1997] was the first to prove the polynomial time complexity of the network simplex. Tarjan [1997] provided shortly after an improved bound in $O\left((n + m)nm\log(n + m)\log((n + m)\|\mathbf{C}\|_\infty)\right)$.

### Dual Ascent Methods

Dual ascent methods precede the network simplex by a few decades. The Hungarian algorithm is the best known algorithm in that family, and it can work only in the particular case when $\mathbf{a}$ and $\mathbf{b}$ are equal and are both uniform, namely $\mathbf{a} = \mathbf{b} = \mathbb{1}_n/n$. By contrast to the network simplex, presented above in the primal, dual ascent methods maintain at each iteration dual feasible solutions whose objective is progressively improved by adding a sparse vector to $\mathbf{f}$ and $\mathbf{g}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.2</span><span class="math-callout__name">(Indicator Vectors)</span></p>

For $S \subset [\![n]\!]$, $S' \subset [\![m]\!]' \stackrel{\text{def}}{=} \lbrace 1', \ldots, m' \rbrace$ we write $\mathbb{1}_S$ for the vector in $\mathbb{R}^n$ of zeros except for ones at the indices enumerated in $S$, and likewise for the vector $\mathbb{1}_{S'}$ in $\mathbb{R}^m$ with indices in $S'$.

We say that $(i, j')$ is a *balanced* pair (or edge) if $\mathbf{f}_i + \mathbf{g}_j = \mathbf{C}_{ij}$ and *inactive* otherwise, namely if $\mathbf{f}_i + \mathbf{g}_j < \mathbf{C}_{ij}$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.5</span><span class="math-callout__name">(Dual Feasibility of Perturbation)</span></p>

$(\tilde{\mathbf{f}}, \tilde{\mathbf{g}}) \stackrel{\text{def}}{=} (\mathbf{f}, \mathbf{g}) + \varepsilon(\mathbb{1}_S, -\mathbb{1}_{S'})$ is dual feasible for a small enough $\varepsilon > 0$ if for all $i \in S$, the fact that $(i, j')$ is balanced implies that $j' \in S'$.

*Proof.* For any $i \in S$, consider the set $\mathcal{I}_i$ of all $j' \in [\![m]\!]'$ such that $(i, j')$ is inactive, namely such that $\mathbf{f}_i + \mathbf{g}_j < \mathbf{C}_{ij}$. Define $\varepsilon_i \stackrel{\text{def}}{=} \min_{j \in I_i} \mathbf{C}_{i,j} - \mathbf{f}_i - \mathbf{g}_j$, the smallest margin by which $\mathbf{f}_i$ can be increased without violating the constraints corresponding to $j' \in \mathcal{I}_i$. Consider now the set $\mathcal{B}_i$ of balanced edges associated with $i$. Note that $\mathcal{B}_i = [\![m]\!]' \setminus \mathcal{I}_i$. The assumption above is that $j' \in \mathcal{B}_i \Rightarrow j' \in S'$. Therefore, for $j' \in \mathcal{B}_i$, $\tilde{\mathbf{f}}_i + \tilde{\mathbf{g}}_j = \mathbf{f}_i + \mathbf{g}_j = \mathbf{C}_{i,j}$. Choosing now an increase $\varepsilon$ smaller than the smallest possible allowed, namely $\min_{i \in S} \varepsilon_i$, we recover that $(\tilde{\mathbf{f}}, \tilde{\mathbf{g}})$ is dual feasible. $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.6</span><span class="math-callout__name">(Dual Ascent Direction)</span></p>

Either $(\mathbf{f}, \mathbf{g})$ is optimal for the dual problem or there exists $S \subset [\![n]\!]$, $S' \subset [\![m]\!]'$ such that $(\tilde{\mathbf{f}}, \tilde{\mathbf{g}}) \stackrel{\text{def}}{=} (\mathbf{f}, \mathbf{g}) + \varepsilon(\mathbb{1}_S, -\mathbb{1}_{S'})$ is feasible for a small enough $\varepsilon > 0$ and has a strictly better objective.

*Proof.* We consider first a complementary primal variable $\mathbf{P}$ to $(\mathbf{f}, \mathbf{g})$. Let $\mathcal{B}$ be the set of balanced edges, and form the bipartite graph whose vertices $\lbrace 1, \ldots, n, 1', \ldots, m' \rbrace$ are linked with edges in $\mathcal{B}$ only, complemented by a source node $s$ connected to all nodes $i \in [\![n]\!]$ with *capacitated* edges with respective capacities $\mathbf{a}_i$, and a terminal node $t$ also connected to all nodes $j' \in [\![m]\!]'$ with edges of respective capacities $\mathbf{b}_j$. The Ford--Fulkerson algorithm can be used to compute a maximal flow $\mathbf{F}$ on that network. If the throughput of that flow $\mathbf{F}$ is equal to 1, then a feasible primal solution $\mathbf{P}$, complementary to $\mathbf{f}, \mathbf{g}$ can be extracted from $\mathbf{F}$, resulting in the optimality of $(\mathbf{f}, \mathbf{g})$ and $\mathbf{P}$ by Proposition 3.3. If the throughput is strictly smaller than 1, the labeling algorithm identifies sets $S, S'$ such that $\mathbb{1}_S^\mathrm{T}\mathbf{a} - \mathbb{1}_{S'}^\mathrm{T}\mathbf{b} > 0$, ensuring that $(\tilde{\mathbf{f}}, \tilde{\mathbf{g}})$ has a better objective than $(\mathbf{f}, \mathbf{g})$. $\square$

</div>

### Auction Algorithm

The auction algorithm was originally proposed by Bertsekas [1981] and later refined. The algorithm can be adapted for arbitrary marginals, but we present it here in its formulation to solve optimal assignment problems.

**Complementary slackness.** In the optimal assignment problem, the primal-dual conditions become easier to formulate, because any extremal solution $\mathbf{P}$ is necessarily a permutation matrix $\mathbf{P}_\sigma$. Given primal $\mathbf{P}_{\sigma^\star}$ and dual $\mathbf{f}^\star, \mathbf{g}^\star$ optimal solutions we necessarily have that $\mathbf{f}_i^\star + \mathbf{g}_{\sigma_i^\star}^\star = \mathbf{C}_{i, \sigma_i^\star}$. By the principle of $\mathbf{C}$-transforms, one can choose $\mathbf{f}^\star$ to be equal to $\mathbf{g}^{\bar{\mathbf{C}}}$. We therefore have that

$$
\mathbf{C}_{i, \sigma_i^\star} - \mathbf{g}_{\sigma_i}^\star = \min_j \mathbf{C}_{i,j} - \mathbf{g}_j^\star.
$$

Conversely, it is easy to show that if there exists a vector $\mathbf{g}$ and a permutation $\sigma$ such that $\mathbf{C}_{i,\sigma_i} - \mathbf{g}_{\sigma_i} = \min_j \mathbf{C}_{i,j} - \mathbf{g}_j$ holds, then they are both optimal, in the sense that $\sigma$ is an optimal assignment and $\mathbf{g}^{\bar{\mathbf{C}}}, \mathbf{g}$ is an optimal dual pair.

**Partial assignments and $\varepsilon$-complementary slackness.** The goal of the auction algorithm is to modify iteratively a triplet $S, \xi, \mathbf{g}$, where $S$ is a subset of $[\![n]\!]$, $\xi$ a partial assignment vector, namely an injective map from $S$ to $[\![n]\!]$, and $\mathbf{g}$ a dual vector. The algorithm converges toward a solution satisfying an *approximate* complementary slackness property, while $S$ grows to cover $[\![n]\!]$ as $\xi$ describes a permutation. The algorithm works by maintaining three properties after each iteration:

- (a) $\forall\, i \in S$, $\quad \mathbf{C}_{i, \xi_i} - \mathbf{g}_{\xi_i} \le \varepsilon + \min_j \mathbf{C}_{i,j} - \mathbf{g}_j$ ($\varepsilon$-CS).
- (b) The size of $S$ can only increase at each iteration.
- (c) There exists an index $i$ such that $\mathbf{g}_i$ decreases by at least $\varepsilon$.

**Auction algorithm updates.** Given a point $j$ the auction algorithm uses not only the optimum appearing in the usual $\mathbf{C}$-transform but also a second best,

$$
j_i^1 \in \operatorname{argmin}_j \mathbf{C}_{i,j} - \mathbf{g}_j, \quad j_i^2 \in \operatorname{argmin}_{j \neq j_i^1} \mathbf{C}_{i,j} - \mathbf{g}_j,
$$

to define the following updates on $\mathbf{g}$ for an index $i \notin S$, as well as on $S$ and $\xi$:

1. **update $\mathbf{g}$**: Remove to the $j_i^1$-th entry of $\mathbf{g}$ the sum of $\varepsilon$ and the difference between the second lowest and lowest adjusted cost:

$$
\mathbf{g}_{j_i^1} \leftarrow \mathbf{g}_{j_i^1} - \left(\underbrace{(\mathbf{C}_{i,j_i^2} - \mathbf{g}_{j_i^2}) - (\mathbf{C}_{i,j_i^1} - \mathbf{g}_{j_i^1})}_{\ge \varepsilon > 0} + \varepsilon\right) = \mathbf{C}_{i,j_i^1} - (\mathbf{C}_{i,j_i^2} - \mathbf{g}_{j_i^2}) - \varepsilon.
$$

2. **update $S$ and $\xi$**: If there exists an index $i' \in S$ such that $\xi_{i'} = j_i^1$, remove it by updating $S \leftarrow S \setminus \lbrace i' \rbrace$. Set $\xi_i = j_i^1$ and add $i$ to $S$, $S \leftarrow S \cup \lbrace i \rbrace$.

The algorithm starts from an empty set of assigned points $S = \emptyset$ with no assignment and empty partial assignment vector $\xi$, and $\mathbf{g} = \mathbf{0}_n$, terminates when $S = [\![n]\!]$, and loops through both steps above until it terminates.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.7</span><span class="math-callout__name">($\varepsilon$-Complementary Slackness Maintained)</span></p>

The auction algorithm maintains $\varepsilon$-complementary slackness at each iteration.

*Proof.* Let $\mathbf{g}, \xi, S$ be the three variables at the beginning of a given iteration. We therefore assume that for any $i' \in S$ the relationship $\mathbf{C}_{i, \xi_{i'}} - \mathbf{g}_{\xi_{i'}} \le \varepsilon + \min_j \mathbf{C}_{i',j} - \mathbf{g}_j$ holds. Consider now the particular $i \notin S$ considered in an iteration. The update $\mathbf{g}^{\mathrm{n}}_{j_i^1} = \mathbf{C}_{i,j_i^1} - (\mathbf{C}_{i,j_i^2} - \mathbf{g}_{j_i^2}) - \varepsilon$ yields $\mathbf{C}_{i,j_i^1} - \mathbf{g}^{\mathrm{n}}_{j_i^1} = \varepsilon + (\mathbf{C}_{i,j_i^2} - \mathbf{g}_{j_i^2}) = \varepsilon + \min_{j \neq j_i^1}(\mathbf{C}_{i,j} - \mathbf{g}_j)$. Since $-\mathbf{g} \le -\mathbf{g}^{\mathrm{n}}$ this implies $\mathbf{C}_{i,j_i^1} - \mathbf{g}^{\mathrm{n}}_{j_i^1} \le \varepsilon + \min_{j \neq j_i^1}(\mathbf{C}_{i,j} - \mathbf{g}^{\mathrm{n}}_j)$ which gives the $\varepsilon$-complementary slackness property for index $i$. For other indices $i' \neq i$, since $\mathbf{g}^{\mathrm{n}} \le \mathbf{g}$ the sequence of inequalities still holds. $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.8</span><span class="math-callout__name">(Termination of the Auction Algorithm)</span></p>

The number of steps of the auction algorithm is at most $N = n\|\mathbf{C}\|_\infty / \varepsilon$.

*Proof.* Suppose that the algorithm has not stopped after $T > N$ steps. Then there exists an index $j$ which is not in the image of $\xi$, namely whose price coordinate $\mathbf{g}_j$ has never been updated and is still $\mathbf{g}_j = 0$. In that case, there cannot exist an index $j'$ such that $\mathbf{g}_{j'}$ was updated $n$ times with $n > \|\mathbf{C}\|_\infty / \varepsilon$. Indeed, if that were the case then for any index $i$, $\mathbf{g}_{j'} \le -n\varepsilon < -\|\mathbf{C}\|_\infty \le -\mathbf{C}_{i,j} = \mathbf{g}_j - \mathbf{C}_{i,j}$, which would result in $\mathbf{C}_{i,j'} - \mathbf{g}_{j'} > \mathbf{C}_{i,j} + (\mathbf{C}_{i,j} - \mathbf{g}_j)$, contradicting $\varepsilon$-CS. Therefore, the total number of iterations $T$ cannot be larger than $n\|\mathbf{C}\|_\infty / \varepsilon = N$. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 3.3</span><span class="math-callout__name">(Complexity)</span></p>

Note that this result yields a naive number of operations of $N^3\|\mathbf{C}\|\_\infty / \varepsilon$ for the algorithm to terminate. That complexity can be reduced to $N^3 \log \|\mathbf{C}\|\_\infty$ when using a clever method known as $\varepsilon$-scaling, designed to decrease the value of $\varepsilon$ with each iteration.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.9</span><span class="math-callout__name">(Suboptimality of the Auction Algorithm)</span></p>

The auction algorithm finds an assignment whose cost is $n\varepsilon$ suboptimal.

*Proof.* Let $\sigma, \mathbf{g}^\star$ be the primal and dual optimal solutions of the assignment problem of matrix $\mathbf{C}$, with optimum $t^\star = \sum_i \mathbf{C}_{i,\sigma_i} = \sum_i \min_j \mathbf{C}_{i,j} - \mathbf{g}_j^\star + \sum_j \mathbf{g}_j^\star$. Let $\xi, \mathbf{g}$ be the solutions output by the auction algorithm upon termination. The $\varepsilon$-CS conditions yield that for any $i \in S$, $\min_j \mathbf{C}_{i,j} - \mathbf{g}_j \ge \mathbf{C}_{i,\xi_i} - \mathbf{g}_{\xi_i} - \varepsilon$. Therefore by simple suboptimality of $\mathbf{g}$ we first have

$$
t^\star \ge \sum_i \left(\min_j \mathbf{C}_{i,j} - \mathbf{g}_j\right) + \sum_j \mathbf{g}_j \ge \sum_i -\varepsilon + \left(\mathbf{C}_{i,\xi_i} - \mathbf{g}_{\xi_i}\right) + \sum_j \mathbf{g}_j = -n\varepsilon + \sum_i \mathbf{C}_{i,\xi_j} \ge -n\varepsilon + t^\star,
$$

where the last inequality follows by the suboptimality of $\xi$ as a permutation. $\square$

</div>

## Chapter 4: Entropic Regularization of Optimal Transport

This chapter introduces a family of numerical schemes to approximate solutions to the Kantorovich formulation of optimal transport and its many generalizations. It operates by adding an entropic regularization penalty to the original problem. This regularization has several important advantages: the minimization of the regularized problem can be solved using a simple alternate minimization scheme (Sinkhorn's algorithm); that scheme translates into iterations that are simple matrix-vector products, making them particularly suited to execution on GPUs; the resulting approximate distance is smooth with respect to input histogram weights and positions of the Diracs and can be differentiated using automatic differentiation.

### Entropic Regularization

The discrete entropy of a coupling matrix is defined as

$$
\mathbf{H}(\mathbf{P}) \stackrel{\text{def}}{=} -\sum_{i,j} \mathbf{P}_{i,j}(\log(\mathbf{P}_{i,j}) - 1),
$$

with an analogous definition for vectors, with the convention that $\mathbf{H}(\mathbf{a}) = -\infty$ if one of the entries $\mathbf{a}_j$ is 0 or negative. The function $\mathbf{H}$ is 1-strongly concave, because its Hessian is $\partial^2 \mathbf{H}(P) = -\mathrm{diag}(1/\mathbf{P}\_{i,j})$ and $\mathbf{P}\_{i,j} \le 1$. The idea of the entropic regularization of optimal transport is to use $-\mathbf{H}$ as a regularizing function to obtain approximate solutions to the original transport problem:

$$
\mathrm{L}_\mathbf{C}^\varepsilon(\mathbf{a}, \mathbf{b}) \stackrel{\text{def}}{=} \min_{\mathbf{P} \in \mathbf{U}(\mathbf{a}, \mathbf{b})} \langle \mathbf{P}, \mathbf{C} \rangle - \varepsilon \mathbf{H}(\mathbf{P}).
$$

Since the objective is an $\varepsilon$-strongly convex function, this problem has a unique optimal solution. The idea to regularize the optimal transport problem by an entropic term can be traced back to modeling ideas in transportation theory [Wilson, 1969]. To mitigate sparsity, researchers in transportation proposed a model, called the "gravity model" [Erlander, 1980], that is able to form a more "blurred" prediction of traffic given marginals and transportation costs.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.1</span><span class="math-callout__name">(Convergence with $\varepsilon$)</span></p>

The unique solution $\mathbf{P}_\varepsilon$ of the regularized problem converges to the optimal solution with maximal entropy within the set of all optimal solutions of the Kantorovich problem, namely

$$
\mathbf{P}_\varepsilon \xrightarrow{\varepsilon \to 0} \operatorname{argmin}_{\mathbf{P}} \left\lbrace -\mathbf{H}(\mathbf{P}) \;:\; \mathbf{P} \in \mathbf{U}(\mathbf{a}, \mathbf{b}),\, \langle \mathbf{P}, \mathbf{C} \rangle = \mathrm{L}_\mathbf{C}(\mathbf{a}, \mathbf{b}) \right\rbrace,
$$

so that in particular $\mathrm{L}\_\mathbf{C}^\varepsilon(\mathbf{a}, \mathbf{b}) \xrightarrow{\varepsilon \to 0} \mathrm{L}\_\mathbf{C}(\mathbf{a}, \mathbf{b})$.

One also has

$$
\mathbf{P}_\varepsilon \xrightarrow{\varepsilon \to \infty} \mathbf{a} \otimes \mathbf{b} = \mathbf{a}\mathbf{b}^\mathrm{T} = (\mathbf{a}_i \mathbf{b}_j)_{i,j}.
$$

</div>

*Proof.* We consider a sequence $(\varepsilon_\ell)\_\ell$ such that $\varepsilon\_\ell \to 0$ and $\varepsilon\_\ell > 0$. We denote $\mathbf{P}\_\ell$ the solution for $\varepsilon = \varepsilon\_\ell$. Since $\mathbf{U}(\mathbf{a}, \mathbf{b})$ is bounded, we can extract a subsequence such that $\mathbf{P}\_\ell \to \mathbf{P}^\star$. Since $\mathbf{U}(\mathbf{a}, \mathbf{b})$ is closed, $\mathbf{P}^\star \in \mathbf{U}(\mathbf{a}, \mathbf{b})$. By optimality of $\mathbf{P}$ and $\mathbf{P}_\ell$ for their respective optimization problems, one has

$$
0 \le \langle \mathbf{C}, \mathbf{P}_\ell \rangle - \langle \mathbf{C}, \mathbf{P} \rangle \le \varepsilon_\ell (\mathbf{H}(\mathbf{P}_\ell) - \mathbf{H}(\mathbf{P})).
$$

Since $\mathbf{H}$ is continuous, taking the limit shows that $\langle \mathbf{C}, \mathbf{P}^\star \rangle = \langle \mathbf{C}, \mathbf{P} \rangle$ so that $\mathbf{P}^\star$ is a feasible point. Furthermore, dividing by $\varepsilon_\ell$ and taking the limit shows that $\mathbf{H}(\mathbf{P}) \le \mathbf{H}(\mathbf{P}^\star)$. Since the solution $\mathbf{P}_0^\star$ to this program is unique by strict convexity of $-\mathbf{H}$, one has $\mathbf{P}^\star = \mathbf{P}_0^\star$. In the limit $\varepsilon \to +\infty$, a similar proof shows that one should rather consider $\min_{\mathbf{P} \in \mathbf{U}(\mathbf{a}, \mathbf{b})} -\mathbf{H}(\mathbf{P})$, the solution of which is $\mathbf{a} \otimes \mathbf{b}$. $\square$

Formula (4.3) states that for a small regularization $\varepsilon$, the solution converges to the maximum entropy optimal transport coupling. In sharp contrast, (4.4) shows that for a large regularization, the solution converges to the coupling with maximal entropy between two prescribed marginals $\mathbf{a}, \mathbf{b}$, namely the joint probability between two independent random variables distributed following $\mathbf{a}, \mathbf{b}$. A key insight is that, as $\varepsilon$ increases, the optimal coupling becomes less and less sparse, which in turn has the effect of both accelerating computational algorithms and leading to faster statistical convergence.

Defining the Kullback--Leibler divergence between couplings as

$$
\mathrm{KL}(\mathbf{P}|\mathbf{K}) \stackrel{\text{def}}{=} \sum_{i,j} \mathbf{P}_{i,j} \log\left(\frac{\mathbf{P}_{i,j}}{\mathbf{K}_{i,j}}\right) - \mathbf{P}_{i,j} + \mathbf{K}_{i,j},
$$

the unique solution $\mathbf{P}_\varepsilon$ is a projection onto $\mathbf{U}(\mathbf{a}, \mathbf{b})$ of the Gibbs kernel associated to the cost matrix $\mathbf{C}$ as

$$
\mathbf{K}_{i,j} \stackrel{\text{def}}{=} e^{-\frac{\mathbf{C}_{i,j}}{\varepsilon}}.
$$

Indeed one has that

$$
\mathbf{P}_\varepsilon = \mathrm{Proj}_{\mathbf{U}(\mathbf{a}, \mathbf{b})}^{\mathrm{KL}}(\mathbf{K}) \stackrel{\text{def}}{=} \operatorname{argmin}_{\mathbf{P} \in \mathbf{U}(\mathbf{a}, \mathbf{b})} \mathrm{KL}(\mathbf{P}|\mathbf{K}).
$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.1</span><span class="math-callout__name">(Entropic Regularization Between Discrete Measures)</span></p>

For discrete measures of the form $\alpha = \sum_i \mathbf{a}\_i \delta\_{x_i}$, $\beta = \sum_j \mathbf{b}\_j \delta\_{y_j}$, the definition of regularized transport extends naturally to

$$
\mathcal{L}_c^\varepsilon(\alpha, \beta) \stackrel{\text{def}}{=} \mathrm{L}_\mathbf{C}^\varepsilon(\mathbf{a}, \mathbf{b}),
$$

with cost $\mathbf{C}_{i,j} = c(x_i, y_j)$, to emphasize the dependency with respect to the positions $(x_i, y_j)$ supporting the input measures.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.2</span><span class="math-callout__name">(General Formulation)</span></p>

One can consider arbitrary measures by replacing the discrete entropy by the relative entropy with respect to the product measure $\mathrm{d}\alpha \otimes \mathrm{d}\beta(x, y) \stackrel{\text{def}}{=} \mathrm{d}\alpha(x)\mathrm{d}\beta(y)$, and propose a regularized counterpart using

$$
\mathcal{L}_c^\varepsilon(\alpha, \beta) \stackrel{\text{def}}{=} \min_{\pi \in \mathcal{U}(\alpha, \beta)} \int_{\mathcal{X} \times \mathcal{Y}} c(x,y)\mathrm{d}\pi(x,y) + \varepsilon\, \mathrm{KL}(\pi | \alpha \otimes \beta),
$$

where the relative entropy is a generalization of the discrete Kullback--Leibler divergence

$$
\mathrm{KL}(\pi|\xi) \stackrel{\text{def}}{=} \int_{\mathcal{X} \times \mathcal{Y}} \log\left(\frac{\mathrm{d}\pi}{\mathrm{d}\xi}(x,y)\right)\mathrm{d}\pi(x,y) + \int_{\mathcal{X} \times \mathcal{Y}} (\mathrm{d}\xi(x,y) - \mathrm{d}\pi(x,y)),
$$

and by convention $\mathrm{KL}(\pi|\xi) = +\infty$ if $\pi$ does not have a density $\frac{\mathrm{d}\pi}{\mathrm{d}\xi}$ with respect to $\xi$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.2</span><span class="math-callout__name">(Reference Measure Independence)</span></p>

For any $\pi \in \mathcal{U}(\alpha, \beta)$, and for any $(\alpha', \beta')$ having the same 0 measure sets as $(\alpha, \beta)$ (so that they have both densities with respect to one another) one has

$$
\mathrm{KL}(\pi|\alpha \otimes \beta) = \mathrm{KL}(\pi|\alpha' \otimes \beta') - \mathrm{KL}(\alpha \otimes \beta|\alpha' \otimes \beta').
$$

This proposition shows that choosing $\mathrm{KL}(\cdot|\alpha' \otimes \beta')$ in place of $\mathrm{KL}(\cdot|\alpha \otimes \beta)$ results in the same solution.

</div>

The regularized problem can be refactored as a projection problem

$$
\min_{\pi \in \mathcal{U}(\alpha, \beta)} \mathrm{KL}(\pi|\mathcal{K}),
$$

where $\mathcal{K}$ is the Gibbs distribution $\mathrm{d}\mathcal{K}(x,y) \stackrel{\text{def}}{=} e^{-\frac{c(x,y)}{\varepsilon}} \mathrm{d}\alpha(x)\mathrm{d}\beta(y)$. This problem is often referred to as the "static Schrödinger problem." As $\varepsilon \to 0$, the unique solution converges to the maximum entropy solution to the OT problem.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.3</span><span class="math-callout__name">(Mutual Entropy)</span></p>

One can rephrase the regularized problem using random variables

$$
\mathcal{L}_c^\varepsilon(\alpha, \beta) = \min_{(X,Y)} \left\lbrace \mathbb{E}_{(X,Y)}(c(X,Y)) + \varepsilon\, \mathrm{I}(X, Y) \;:\; X \sim \alpha,\, Y \sim \beta \right\rbrace,
$$

where, denoting $\pi$ the distribution of $(X, Y)$, $\mathrm{I}(X, Y) \stackrel{\text{def}}{=} \mathrm{KL}(\pi|\alpha \otimes \beta)$ is the so-called mutual information between the two random variables. One has $\mathrm{I}(X, Y) \ge 0$ and $\mathrm{I}(X, Y) = 0$ if and only if the two random variables are independent.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.4</span><span class="math-callout__name">(Independence and Couplings)</span></p>

A coupling $\pi \in \mathcal{U}(\alpha, \beta)$ describes the distribution of a couple of random variables $(X, Y)$ defined on $(\mathcal{X}, \mathcal{Y})$, where $X$ (resp., $Y$) has law $\alpha$ (resp., $\beta$). Proposition 4.1 carries over for generic (nonnecessary discrete) measures, so that the solution $\pi_\varepsilon$ converges to the tensor product coupling $\alpha \otimes \beta$ as $\varepsilon \to +\infty$. This coupling $\alpha \otimes \beta$ corresponds to the random variables $(X, Y)$ being independent. In contrast, as $\varepsilon \to 0$, $\pi_\varepsilon$ converges to a solution $\pi_0$ of the OT problem. On $\mathcal{X} = \mathcal{Y} = \mathbb{R}^d$, if $\alpha$ and $\beta$ have densities with respect to the Lebesgue measure, then $\pi_0$ is unique and supported on the graph of a bijective Monge map $T : \mathbb{R}^d \to \mathbb{R}^d$. In the simple 1-D case $d = 1$, a convenient way to visualize the dependency structure between $X$ and $Y$ is to use the copula $\xi_\pi$ associated to the joint distribution $\pi$.

</div>

### Sinkhorn's Algorithm and Its Convergence

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.3</span><span class="math-callout__name">(Form of the Optimal Solution)</span></p>

The solution to the regularized problem is unique and has the form

$$
\forall\, (i,j) \in [\![n]\!] \times [\![m]\!], \quad \mathbf{P}_{i,j} = \mathbf{u}_i \mathbf{K}_{i,j} \mathbf{v}_j,
$$

for two (unknown) scaling variables $(\mathbf{u}, \mathbf{v}) \in \mathbb{R}_+^n \times \mathbb{R}_+^m$.

*Proof.* Introducing two dual variables $\mathbf{f} \in \mathbb{R}^n, \mathbf{g} \in \mathbb{R}^m$ for each marginal constraint, the Lagrangian reads

$$
\mathcal{E}(\mathbf{P}, \mathbf{f}, \mathbf{g}) = \langle \mathbf{P}, \mathbf{C} \rangle - \varepsilon \mathbf{H}(\mathbf{P}) - \langle \mathbf{f}, \mathbf{P}\mathbb{1}_m - \mathbf{a} \rangle - \langle \mathbf{g}, \mathbf{P}^\mathrm{T}\mathbb{1}_n - \mathbf{b} \rangle.
$$

First order conditions then yield

$$
\frac{\partial \mathcal{E}(\mathbf{P}, \mathbf{f}, \mathbf{g})}{\partial \mathbf{P}_{i,j}} = \mathbf{C}_{i,j} + \varepsilon \log(\mathbf{P}_{i,j}) - \mathbf{f}_i - \mathbf{g}_j = 0,
$$

which results in the expression $\mathbf{P}_{i,j} = e^{\mathbf{f}_i / \varepsilon} e^{-\mathbf{C}_{i,j}/\varepsilon} e^{\mathbf{g}_j / \varepsilon}$, which can be rewritten using nonnegative vectors $\mathbf{u}$ and $\mathbf{v}$. $\square$

</div>

**Regularized OT as matrix scaling.** The factorization of the optimal solution can be conveniently rewritten in matrix form as $\mathbf{P} = \mathrm{diag}(\mathbf{u})\mathbf{K}\,\mathrm{diag}(\mathbf{v})$. The variables $(\mathbf{u}, \mathbf{v})$ must therefore satisfy the following nonlinear equations which correspond to the mass conservation constraints inherent to $\mathbf{U}(\mathbf{a}, \mathbf{b})$:

$$
\mathrm{diag}(\mathbf{u})\mathbf{K}\,\mathrm{diag}(\mathbf{v})\mathbb{1}_m = \mathbf{a}, \quad \mathrm{diag}(\mathbf{v})\mathbf{K}^\top\mathrm{diag}(\mathbf{u})\mathbb{1}_n = \mathbf{b}.
$$

These two equations can be further simplified as

$$
\mathbf{u} \odot (\mathbf{K}\mathbf{v}) = \mathbf{a} \quad \text{and} \quad \mathbf{v} \odot (\mathbf{K}^\mathrm{T}\mathbf{u}) = \mathbf{b},
$$

where $\odot$ corresponds to entrywise multiplication of vectors. This problem is known in the numerical analysis community as the matrix scaling problem. An intuitive way to handle these equations is to solve them iteratively, by modifying first $\mathbf{u}$ so that it satisfies the left-hand side and then $\mathbf{v}$ to satisfy its right-hand side. These two updates define Sinkhorn's algorithm,

$$
\mathbf{u}^{(\ell+1)} \stackrel{\text{def}}{=} \frac{\mathbf{a}}{\mathbf{K}\mathbf{v}^{(\ell)}} \quad \text{and} \quad \mathbf{v}^{(\ell+1)} \stackrel{\text{def}}{=} \frac{\mathbf{b}}{\mathbf{K}^\mathrm{T}\mathbf{u}^{(\ell+1)}},
$$

initialized with an arbitrary positive vector $\mathbf{v}^{(0)} = \mathbb{1}_m$. The division operator used above between two vectors is to be understood entrywise. These iterations converge and all result in the same optimal coupling $\mathrm{diag}(\mathbf{u})\mathbf{K}\,\mathrm{diag}(\mathbf{v})$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.5</span><span class="math-callout__name">(Historical Perspective)</span></p>

The iterations first appeared in [Yule, 1912, Kruithof, 1937]. They were later known as the iterative proportional fitting procedure (IPFP) [Deming and Stephan, 1940] and RAS [Bacharach, 1965] methods. The proof of their convergence is attributed to Sinkhorn [1964], hence the name of the algorithm. This regularization was later extended in infinite dimensions by Ruschendorf [1995]. It was rebranded as "softassign" by Kosowsky and Yuille [1994] in the assignment case, and used to solve matching problems in economics more recently by Galichon and Salanié [2009]. This regularization has received renewed attention in data sciences following [Cuturi, 2013].

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.6</span><span class="math-callout__name">(Overall Complexity)</span></p>

By doing a careful convergence analysis, Altschuler et al. [2017] showed that by setting $\varepsilon = \frac{4\log(n)}{\tau}$, $O(\|\mathbf{C}\|_\infty^3 \log(n)\tau^{-3})$ Sinkhorn iterations (with an additional rounding step to compute a valid coupling $\hat{\mathbf{P}} \in \mathbf{U}(\mathbf{a}, \mathbf{b})$) are enough to ensure that $\langle \hat{\mathbf{P}}, \mathbf{C} \rangle \le \mathrm{L}_\mathbf{C}(\mathbf{a}, \mathbf{b}) + \tau$. This implies that Sinkhorn computes a $\tau$-approximate solution of the unregularized OT problem in $O(n^2 \log(n)\tau^{-3})$ operations.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.7</span><span class="math-callout__name">(Numerical Stability)</span></p>

The convergence of Sinkhorn's algorithm deteriorates as $\varepsilon \to 0$. In numerical practice, however, that slowdown is rarely observed in practice for a simpler reason: Sinkhorn's algorithm will often fail to terminate as soon as some of the elements of the kernel $\mathbf{K}$ become too negligible to be stored in memory as positive numbers, and become instead null. This can then result in a matrix product $\mathbf{K}\mathbf{v}$ or $\mathbf{K}^\mathrm{T}\mathbf{u}$ with ever smaller entries that become null and result in a division by 0 in the Sinkhorn update. Such issues can be partly resolved by carrying out computations on the multipliers $\mathbf{u}$ and $\mathbf{v}$ in the log domain.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.8</span><span class="math-callout__name">(Relation with Iterative Projections)</span></p>

Denoting

$$
\mathcal{C}_\mathbf{a}^1 \stackrel{\text{def}}{=} \lbrace \mathbf{P} \;:\; \mathbf{P}\mathbb{1}_m = \mathbf{a} \rbrace \quad \text{and} \quad \mathcal{C}_\mathbf{b}^2 \stackrel{\text{def}}{=} \left\lbrace \mathbf{P} \;:\; \mathbf{P}^\mathrm{T}\mathbb{1}_m = \mathbf{b} \right\rbrace
$$

the rows and columns constraints, one has $\mathbf{U}(\mathbf{a}, \mathbf{b}) = \mathcal{C}_\mathbf{a}^1 \cap \mathcal{C}_\mathbf{b}^2$. One can use Bregman iterative projections [Bregman, 1967],

$$
\mathbf{P}^{(\ell+1)} \stackrel{\text{def}}{=} \mathrm{Proj}_{\mathcal{C}_\mathbf{a}^1}^{\mathrm{KL}}(\mathbf{P}^{(\ell)}) \quad \text{and} \quad \mathbf{P}^{(\ell+2)} \stackrel{\text{def}}{=} \mathrm{Proj}_{\mathcal{C}_\mathbf{b}^2}^{\mathrm{KL}}(\mathbf{P}^{(\ell+1)}).
$$

Since the sets $\mathcal{C}_\mathbf{a}^1$ and $\mathcal{C}_\mathbf{b}^2$ are affine, these iterations are known to converge to the solution and are equivalent to Sinkhorn iterations.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.9</span><span class="math-callout__name">(Proximal Point Algorithm)</span></p>

In order to approximate a solution of the unregularized ($\varepsilon = 0$) problem, it is possible to use iteratively the Sinkhorn algorithm, using the so-called proximal point algorithm for the KL metric. Denoting $F(\mathbf{P}) \stackrel{\text{def}}{=} \langle \mathbf{P}, \pi \rangle + \iota_{\mathbf{U}(\mathbf{a}, \mathbf{b})}(\mathbf{P})$ the unregularized objective function, the proximal point iterations for the $\mathrm{KL}$ divergence compute iteratively

$$
\mathbf{P}^{(\ell+1)} \stackrel{\text{def}}{=} \mathrm{Prox}_{\frac{1}{\varepsilon}F}^{\mathrm{KL}}(\mathbf{P}^{(\ell)}) \stackrel{\text{def}}{=} \operatorname{argmin}_{\mathbf{P} \in \mathbb{R}_+^{n \times m}} \mathrm{KL}(\mathbf{P}|\mathbf{P}^{(\ell)}) + \frac{1}{\varepsilon} F(\mathbf{P}),
$$

starting from an arbitrary $\mathbf{P}^{(0)}$. The proximal point iterates apply therefore iteratively Sinkhorn's algorithm with a decaying regularization parameter $\varepsilon / \ell$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.10</span><span class="math-callout__name">(Other Regularizations)</span></p>

It is possible to replace the entropic term $-\mathbf{H}(\mathbf{P})$ by any strictly convex penalty $R(\mathbf{P})$. A typical example is the squared $\ell^2$ norm

$$
R(\mathbf{P}) = \sum_{i,j} \mathbf{P}_{i,j}^2 + \iota_{\mathbb{R}_+}(\mathbf{P}_{i,j}).
$$

Note, however, that if the penalty function is defined even when entries of $\mathbf{P}$ are nonpositive, then one must add back a nonnegativity constraint $\mathbf{P} \ge 0$. The main advantage of the quadratic regularization over entropy is that it produces sparse approximation of the optimal coupling, yet this comes at the expense of a slower algorithm that cannot be parallelized as efficiently as Sinkhorn.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.11</span><span class="math-callout__name">(Barycentric Projection)</span></p>

Consider the setting where we use entropic regularization to approximate OT between discrete measures. The Kantorovich formulation and its entropic regularization both yield a coupling $\mathbf{P} \in \mathbf{U}(\mathbf{a}, \mathbf{b})$. In order to define a transportation map $T : \mathcal{X} \to \mathcal{Y}$, in the case where $\mathcal{Y} = \mathbb{R}^d$, one can define the so-called barycentric projection map

$$
T : x_i \in \mathcal{X} \longmapsto \frac{1}{\mathbf{a}_i} \sum_j \mathbf{P}_{i,j} y_j \in \mathcal{Y},
$$

where the input measures are discrete. Note that this map is only defined for points $(x_i)_i$ in the support of $\alpha$. In the case where $T$ is a permutation matrix, then $T$ is equal to a Monge map, and as $\varepsilon \to 0$, the barycentric projection progressively converges to that map if it is unique.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.12</span><span class="math-callout__name">(Hilbert Metric)</span></p>

The global convergence analysis of Sinkhorn is greatly simplified using the Hilbert projective metric on $\mathbb{R}_{+,*}^n$ (positive vectors), defined as

$$
\forall\, (\mathbf{u}, \mathbf{u}') \in (\mathbb{R}_{+,*}^n)^2, \quad d_\mathcal{H}(\mathbf{u}, \mathbf{u}') \stackrel{\text{def}}{=} \log \max_{i,j} \frac{\mathbf{u}_i \mathbf{u}_j'}{\mathbf{u}_j \mathbf{u}_i'}.
$$

It can be shown to be a distance on the projective cone $\mathbb{R}_{+,*}^n / \sim$, where $\mathbf{u} \sim \mathbf{u}'$ means that $\exists r > 0, \mathbf{u} = r\mathbf{u}'$. By a logarithmic change of variables, the Hilbert metric on the rays of the positive cone is isometric to the variation seminorm

$$
d_\mathcal{H}(\mathbf{u}, \mathbf{u}') = \|\log(\mathbf{u}) - \log(\mathbf{u}')\|_\mathrm{var},
$$

where $\|\mathbf{f}\|_\mathrm{var} \stackrel{\text{def}}{=} (\max_i \mathbf{f}_i) - (\min_i \mathbf{f}_i)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.1</span><span class="math-callout__name">(Birkhoff Contraction)</span></p>

Let $\mathbf{K} \in \mathbb{R}_{+,*}^{n \times m}$; then for $(\mathbf{v}, \mathbf{v}') \in (\mathbb{R}_{+,*}^m)^2$

$$
d_\mathcal{H}(\mathbf{K}\mathbf{v}, \mathbf{K}\mathbf{v}') \le \lambda(\mathbf{K}) d_\mathcal{H}(\mathbf{v}, \mathbf{v}'), \quad \text{where} \quad \begin{cases} \lambda(\mathbf{K}) \stackrel{\text{def}}{=} \dfrac{\sqrt{\eta(\mathbf{K})} - 1}{\sqrt{\eta(\mathbf{K})} + 1} < 1, \\[6pt] \eta(\mathbf{K}) \stackrel{\text{def}}{=} \max_{i,j,k,\ell} \dfrac{\mathbf{K}_{i,k}\mathbf{K}_{j,\ell}}{\mathbf{K}_{j,k}\mathbf{K}_{i,\ell}}. \end{cases}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.13</span><span class="math-callout__name">(Perron--Frobenius)</span></p>

A typical application of Theorem 4.1 is to provide a quantitative proof of the Perron--Frobenius theorem. A matrix $\mathbf{K} \in \mathbb{R}_+^{n \times n}$ with $\mathbf{K}^\top\mathbb{1}_n = \mathbb{1}_n$ maps $\Sigma_n$ into $\Sigma_n$. If furthermore $\mathbf{K} > 0$, then according to Theorem 4.1, it is strictly contractant for the metric $d_\mathcal{H}$, hence there exists a unique invariant probability distribution $p^\star \in \Sigma_n$ with $\mathbf{K}p^\star = p^\star$. Furthermore, for any $p_0 \in \Sigma_n$, $d_\mathcal{H}(\mathbf{K}^\ell p_0, p^\star) \le \lambda(\mathbf{K})^\ell d_\mathcal{H}(p_0, p^\star)$, i.e. one has linear convergence of the iterates of the matrix toward $p^\star$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.14</span><span class="math-callout__name">(Global Convergence)</span></p>

The following theorem, proved by [Franklin and Lorenz, 1989], makes use of Theorem 4.1 to show the linear convergence of Sinkhorn's iterations.

**Theorem 4.2.** One has $(\mathbf{u}^{(\ell)}, \mathbf{v}^{(\ell)}) \to (\mathbf{u}^\star, \mathbf{v}^\star)$ and

$$
d_\mathcal{H}(\mathbf{u}^{(\ell)}, \mathbf{u}^\star) = O(\lambda(\mathbf{K})^{2\ell}), \quad d_\mathcal{H}(\mathbf{v}^{(\ell)}, \mathbf{v}^\star) = O(\lambda(\mathbf{K})^{2\ell}).
$$

One also has

$$
d_\mathcal{H}(\mathbf{u}^{(\ell)}, \mathbf{u}^\star) \le \frac{d_\mathcal{H}(\mathbf{P}^{(\ell)}\mathbb{1}_m, \mathbf{a})}{1 - \lambda(\mathbf{K})^2}, \quad d_\mathcal{H}(\mathbf{v}^{(\ell)}, \mathbf{v}^\star) \le \frac{d_\mathcal{H}(\mathbf{P}^{(\ell),\top}\mathbb{1}_n, \mathbf{b})}{1 - \lambda(\mathbf{K})^2},
$$

where $\mathbf{P}^{(\ell)} \stackrel{\text{def}}{=} \mathrm{diag}(\mathbf{u}^{(\ell)})\mathbf{K}\,\mathrm{diag}(\mathbf{v}^{(\ell)})$. Last, one has

$$
\|\log(\mathbf{P}^{(\ell)}) - \log(\mathbf{P}^\star)\|_\infty \le d_\mathcal{H}(\mathbf{u}^{(\ell)}, \mathbf{u}^\star) + d_\mathcal{H}(\mathbf{v}^{(\ell)}, \mathbf{v}^\star),
$$

where $\mathbf{P}^\star$ is the unique solution of the regularized problem.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.15</span><span class="math-callout__name">(Local Convergence)</span></p>

The global linear rate is often quite pessimistic, typically in $\mathcal{X} = \mathcal{Y} = \mathbb{R}^d$ for cases where there exists a Monge map when $\varepsilon = 0$. The global rate is in contrast rather sharp for more difficult situations where the cost matrix $\mathbf{C}$ is close to being random, and in these cases, the rate scales exponentially bad with $\varepsilon$, $1 - \lambda(\mathbf{K}) \sim e^{-1/\varepsilon}$. One can write a Sinkhorn update as iterations of a fixed-point map $\mathbf{f}^{(\ell+1)} = \Phi(\mathbf{f}^{(\ell)})$, where

$$
\Phi \stackrel{\text{def}}{=} \Phi_2 \odot \Phi_1, \quad \text{where} \quad \begin{cases} \Phi_1(\mathbf{f}) = \varepsilon \log \mathbf{K}^\mathrm{T}(e^{\mathbf{f}/\varepsilon}) - \log(\mathbf{b}), \\ \Phi_2(\mathbf{g}) = \varepsilon \log \mathbf{K}(e^{\mathbf{g}/\varepsilon}) - \log(\mathbf{a}). \end{cases}
$$

For optimal $(\mathbf{f}, \mathbf{g})$, the Jacobian has the form $\partial\Phi(\mathbf{f}) = \mathrm{diag}(\mathbf{a})^{-1} \odot \mathbf{P} \odot \mathrm{diag}(\mathbf{b})^{-1} \odot \mathbf{P}^\mathrm{T}$, which is a positive matrix with a single dominant eigenvector $\mathbb{1}_n$ with associated eigenvalue 1. Since $\mathbf{f}$ is defined up to an additive constant, it is actually the second eigenvalue $1 - \kappa < 1$ which governs the local linear rate, and this shows that for $\ell$ large enough, $\|\mathbf{f}^{(\ell)} - \mathbf{f}\| = O((1 - \kappa)^\ell)$. Numerically, in "simple cases" this rate scales like $\kappa \sim \varepsilon$.

</div>

### Speeding Up Sinkhorn's Iterations

The main computational bottleneck of Sinkhorn's iterations is the vector-matrix multiplication against kernels $\mathbf{K}$ and $\mathbf{K}^\top$, with complexity $O(nm)$ if implemented naively. We now detail several important cases where the complexity can be improved significantly.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.16</span><span class="math-callout__name">(Parallel and GPU Friendly Computation)</span></p>

The simplicity of Sinkhorn's algorithm yields an extremely efficient approach to compute simultaneously several regularized Wasserstein distances between pairs of histograms. Let $N$ be an integer, $\mathbf{a}_1, \ldots, \mathbf{a}_N$ be histograms in $\Sigma_n$, and $\mathbf{b}_1, \ldots, \mathbf{b}_N$ be histograms in $\Sigma_m$. Writing $\mathbf{A} = [\mathbf{a}_1, \ldots, \mathbf{a}_N]$ and $\mathbf{B} = [\mathbf{b}_1, \ldots, \mathbf{b}_N]$ for the $n \times N$ and $m \times N$ matrices storing all $N$ histograms, one can notice that all Sinkhorn iterations for all these $N$ pairs can be carried out in parallel, by setting

$$
\mathbf{U}^{(\ell+1)} \stackrel{\text{def}}{=} \frac{\mathbf{A}}{\mathbf{K}\mathbf{V}^{(\ell)}} \quad \text{and} \quad \mathbf{V}^{(\ell+1)} \stackrel{\text{def}}{=} \frac{\mathbf{B}}{\mathbf{K}^\mathrm{T}\mathbf{U}^{(\ell+1)}}.
$$

Note that the basic Sinkhorn iterations are intrinsically GPU friendly, since they only consist in matrix-vector products. The matrix-matrix operations present even better opportunities for parallelism.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.17</span><span class="math-callout__name">(Speed-up for Separable Kernels)</span></p>

An important particular case for which the complexity of each Sinkhorn iteration can be significantly reduced occurs when each index $i$ and $j$ considered in the cost matrix can be described as a $d$-uple taken in the cartesian product of $d$ finite sets $[\![n_1]\!], \ldots, [\![n_d]\!]$. If the cost $\mathbf{C}_{ij}$ is additive along these sub-indices, i.e. $\mathbf{C}_{ij} = \sum_{k=1}^d \mathbf{C}^k_{i_k, j_k}$, then the kernel has a separable multiplicative structure,

$$
\mathbf{K}_{i,j} = \prod_{k=1}^d \mathbf{K}^k_{i_k, j_k}.
$$

Such separability allows for very fast evaluation of $\mathbf{K}\mathbf{u}$. Instead of instantiating $\mathbf{K}$ as a matrix of size $n \times n$ (which would have prohibitive size since $n = \prod_k n_k$ is usually exponential in $d$), one can instead recover $\mathbf{K}\mathbf{u}$ by simply applying $\mathbf{K}^k$ along each "slice" of $\mathbf{u}$. If $n = m$, the complexity reduces to $O(n^{1+1/d})$ in place of $O(n^2)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.18</span><span class="math-callout__name">(Approximated Convolutions)</span></p>

The main computational bottleneck of Sinkhorn's iterations lies in the multiplication of a vector by $\mathbf{K}$ or by its adjoint. Besides using separability, it is also possible to exploit other special structures in the kernel. The simplest case is for translation invariant kernels $\mathbf{K}_{i,j} = k_{i-j}$, which is typically the case when discretizing the measure on a fixed uniform grid. Then $\mathbf{K}\mathbf{v} = k \star \mathbf{v}$ is a convolution, and there are several algorithms to approximate the convolution in nearly linear time. The most usual one is by Fourier transform $\mathcal{F}$, because $\mathcal{F}(k \star \mathbf{v}) = \mathcal{F}(k) \odot \mathcal{F}(\mathbf{v})$. Another popular way to speed up computation is by approximating the convolution using a succession of autoregressive filters.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.19</span><span class="math-callout__name">(Geodesic in Heat Approximation)</span></p>

For nonplanar domains, the kernel $\mathbf{K}$ is not a convolution, but in the case where the cost is $\mathbf{C}_{i,j} = d_\mathcal{M}(x_i, y_j)^p$ where $d_\mathcal{M}$ is a geodesic distance on a surface $\mathcal{M}$ (or a more general manifold), it is also possible to perform fast approximations of the application of $\mathbf{K} = e^{-\frac{d_\mathcal{M}^p}{\varepsilon}}$ to a vector. Varadhan's formulas assert that this kernel is close to the Laplacian kernel (for $p = 1$) and the heat kernel (for $p = 2$). The first formula of Varadhan states

$$
-\frac{\sqrt{t}}{2}\log(\mathcal{P}_t(x,y)) = d_\mathcal{M}(x,y) + o(t),
$$

where $\Delta_\mathcal{M}$ is the Laplace--Beltrami operator associated to the manifold $\mathcal{M}$ and $\mathcal{P}_t \stackrel{\text{def}}{=} (\mathrm{Id} - t\Delta_\mathcal{M})^{-1}$. These formulas can be used to approximate efficiently the multiplication by the Gibbs kernel $\mathbf{K}_{i,j} = e^{-\frac{d(x_i, y_j)^p}{\varepsilon}}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.20</span><span class="math-callout__name">(Extrapolation Acceleration)</span></p>

Since the Sinkhorn algorithm is a fixed-point algorithm, one can use standard linear or even nonlinear extrapolation schemes to enhance the conditioning of the fixed-point mapping near the solution, and improve the linear convergence rate. This is similar to the successive overrelaxation method, so that the local linear rate of convergence is improved from $O((1 - \kappa)^\ell)$ to $O((1 - \sqrt{\kappa})^\ell)$ for some $\kappa > 0$.

</div>

### Stability and Log-Domain Computations

As briefly mentioned in Remark 4.7, the Sinkhorn algorithm suffers from numerical overflow when the regularization parameter $\varepsilon$ is small compared to the entries of the cost matrix $\mathbf{C}$. This concern can be alleviated to some extent by carrying out computations in the log domain. The relevance of this approach is made more clear by considering the dual problem, in which these log-domain computations arise naturally.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.4</span><span class="math-callout__name">(Entropic Dual)</span></p>

One has

$$
\mathrm{L}_\mathbf{C}^\varepsilon(\mathbf{a}, \mathbf{b}) = \max_{\mathbf{f} \in \mathbb{R}^n, \mathbf{g} \in \mathbb{R}^m} \langle \mathbf{f}, \mathbf{a} \rangle + \langle \mathbf{g}, \mathbf{b} \rangle - \varepsilon \langle e^{\mathbf{f}/\varepsilon}, \mathbf{K} e^{\mathbf{g}/\varepsilon} \rangle.
$$

The optimal $(\mathbf{f}, \mathbf{g})$ are linked to scalings $(\mathbf{u}, \mathbf{v})$ through

$$
(\mathbf{u}, \mathbf{v}) = (e^{\mathbf{f}/\varepsilon}, e^{\mathbf{g}/\varepsilon}).
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.21</span><span class="math-callout__name">(Sinkhorn as Block Coordinate Ascent on the Dual)</span></p>

A simple approach to solving the unconstrained maximization problem is to use an exact block coordinate ascent strategy, namely to update alternatively $\mathbf{f}$ and $\mathbf{g}$ to cancel the respective gradients in these variables of the objective. The gradients are

$$
\nabla|_\mathbf{f} Q(\mathbf{f}, \mathbf{g}) = \mathbf{a} - e^{\mathbf{f}/\varepsilon} \odot \left(\mathbf{K} e^{\mathbf{g}/\varepsilon}\right),
$$

$$
\nabla|_\mathbf{g} Q(\mathbf{f}, \mathbf{g}) = \mathbf{b} - e^{\mathbf{g}/\varepsilon} \odot \left(\mathbf{K}^\mathrm{T} e^{\mathbf{f}/\varepsilon}\right).
$$

Block coordinate ascent can therefore be implemented in closed form by applying successively the following updates, starting from any arbitrary $\mathbf{g}^{(0)}$, for $\ell \ge 0$:

$$
\mathbf{f}^{(\ell+1)} = \varepsilon \log \mathbf{a} - \varepsilon \log\left(\mathbf{K} e^{\mathbf{g}^{(\ell)}/\varepsilon}\right),
$$

$$
\mathbf{g}^{(\ell+1)} = \varepsilon \log \mathbf{b} - \varepsilon \log\left(\mathbf{K}^\mathrm{T} e^{\mathbf{f}^{(\ell+1)}/\varepsilon}\right).
$$

Such iterations are mathematically equivalent to the Sinkhorn iterations when considering the primal-dual relations $(\mathbf{f}^{(\ell)}, \mathbf{g}^{(\ell)}) = \varepsilon(\log(\mathbf{u}^{(\ell)}), \log(\mathbf{v}^{(\ell)}))$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.22</span><span class="math-callout__name">(Soft-min Rewriting)</span></p>

The log-domain Sinkhorn iterates can be given an alternative interpretation using the soft-minimum. Given a vector $\mathbf{z}$ of real numbers we write $\min_\varepsilon \mathbf{z}$ for the *soft-minimum* of its coordinates, namely

$$
\min_\varepsilon \mathbf{z} = -\varepsilon \log \sum_j e^{-\mathbf{z}_i / \varepsilon}.
$$

Note that $\min_\varepsilon(\mathbf{z})$ converges to $\min \mathbf{z}$ for any vector $\mathbf{z}$ as $\varepsilon \to 0$. Indeed, $\min_\varepsilon$ can be interpreted as a differentiable approximation of the min function. Using this notation, Sinkhorn's iterates read

$$
(\mathbf{f}^{(\ell+1)})_i = \min_\varepsilon\, (\mathbf{C}_{ij} - \mathbf{g}_j^{(\ell)})_j + \varepsilon \log \mathbf{a}_i,
$$

$$
(\mathbf{g}^{(\ell+1)})_j = \min_\varepsilon\, (\mathbf{C}_{ij} - \mathbf{f}_i^{(\ell)})_i + \varepsilon \log \mathbf{b}_j.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.23</span><span class="math-callout__name">(Log-domain Sinkhorn)</span></p>

While mathematically equivalent to the Sinkhorn updates, the log-domain iterations suggest using the *log-sum-exp* stabilization trick to avoid underflow for small values of $\varepsilon$. Writing $\underline{z} = \min \mathbf{z}$, that trick suggests evaluating $\min_\varepsilon \mathbf{z}$ as

$$
\min_\varepsilon \mathbf{z} = \underline{z} - \varepsilon \log \sum_i e^{-(\mathbf{z}_i - \underline{z})/\varepsilon}.
$$

Instead of substracting $\underline{z}$ to stabilize the log-domain iterations, one can actually substract the previously computed scalings. This leads to the stabilized iteration

$$
\mathbf{f}^{(\ell+1)} = \mathrm{Min}_\varepsilon^\mathrm{row}\left(\mathbf{S}(\mathbf{f}^{(\ell)}, \mathbf{g}^{(\ell)})\right) + \mathbf{f}^{(\ell)} + \varepsilon \log(\mathbf{a}),
$$

$$
\mathbf{g}^{(\ell+1)} = \mathrm{Min}_\varepsilon^\mathrm{col}\left(\mathbf{S}(\mathbf{f}^{(\ell+1)}, \mathbf{g}^{(\ell)})\right) + \mathbf{g}^{(\ell)} + \varepsilon \log(\mathbf{b}),
$$

where $\mathbf{S}(\mathbf{f}, \mathbf{g}) = \left(\mathbf{C}_{i,j} - \mathbf{f}_i - \mathbf{g}_j\right)_{i,j}$. In contrast to the original iterations, these log-domain iterations are stable for arbitrary $\varepsilon > 0$, because the quantity $\mathbf{S}(\mathbf{f}, \mathbf{g})$ stays bounded during the iterations. The downside is that it requires $nm$ computations of exp at each step. Computing a $\mathrm{Min}_\varepsilon^\mathrm{row}$ or $\mathrm{Min}_\varepsilon^\mathrm{col}$ is typically substantially slower than matrix multiplications. There is therefore no efficient way to parallelize the application of Sinkhorn maps for several marginals simultaneously.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.24</span><span class="math-callout__name">(Dual for Generic Measures)</span></p>

For generic and not necessarily discrete input measures $(\alpha, \beta)$, the dual problem reads

$$
\sup_{(f,g) \in \mathcal{C}(\mathcal{X}) \times \mathcal{C}(\mathcal{Y})} \int_\mathcal{X} f\,\mathrm{d}\alpha + \int_\mathcal{Y} g\,\mathrm{d}\beta - \varepsilon \int_{\mathcal{X} \times \mathcal{Y}} e^{\frac{-c(x,y) + f(x) + g(y)}{\varepsilon}} \mathrm{d}\alpha(x)\mathrm{d}\beta(y).
$$

This corresponds to a smoothing of the constraint $\mathcal{R}(c)$ appearing in the original problem, which is retrieved in the limit $\varepsilon \to 0$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.25</span><span class="math-callout__name">(Unconstrained Entropic Dual)</span></p>

As in Remark 2.23, in the case $\int_\mathcal{X} \mathrm{d}\alpha = \int_\mathcal{Y} \mathrm{d}\beta = 1$, one can consider an alternative dual formulation

$$
\sup_{(f,g) \in \mathcal{C}(\mathcal{X}) \times \mathcal{C}(\mathcal{Y})} \int_\mathcal{X} f\,\mathrm{d}\alpha + \int_\mathcal{Y} g\,\mathrm{d}\beta + \min_\varepsilon (c - f \oplus g),
$$

which achieves the same optimal value. A chief advantage of this alternative formulation is that it is better conditioned, in the sense that the Hessian of the functional is uniformly bounded by $\varepsilon$.

</div>

### Regularized Approximations of the Optimal Transport Cost

The entropic dual is a smooth unconstrained concave maximization problem, which approximates the original Kantorovich dual, as detailed in the following propositions.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.5</span><span class="math-callout__name">(Dual Feasibility)</span></p>

Any pair of optimal solutions $(\mathbf{f}^\star, \mathbf{g}^\star)$ to the entropic dual are such that $(\mathbf{f}^\star, \mathbf{g}^\star) \in \mathbf{R}(\mathbf{C})$, the set of feasible Kantorovich potentials. As a consequence, we have that for any $\varepsilon$,

$$
\langle \mathbf{f}^\star, \mathbf{a} \rangle + \langle \mathbf{g}^\star, \mathbf{b} \rangle \le \mathrm{L}_\mathbf{C}(\mathbf{a}, \mathbf{b}).
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.6</span><span class="math-callout__name">(Convexity and Gradient)</span></p>

$\mathrm{L}_\mathbf{C}^\varepsilon(\mathbf{a}, \mathbf{b})$ is a jointly convex function of $\mathbf{a}$ and $\mathbf{b}$ for $\varepsilon \ge 0$. When $\varepsilon > 0$, its gradient is equal to

$$
\nabla \mathrm{L}_\mathbf{C}^\varepsilon(\mathbf{a}, \mathbf{b}) = \begin{bmatrix} \mathbf{f}^\star \\ \mathbf{g}^\star \end{bmatrix},
$$

where $\mathbf{f}^\star$ and $\mathbf{g}^\star$ are the optimal solutions of the entropic dual chosen so that their coordinates sum to 0.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.1</span><span class="math-callout__name">(Sinkhorn Divergences)</span></p>

Let $\mathbf{f}^\star$ and $\mathbf{g}^\star$ be optimal solutions to the entropic dual and $\mathbf{P}^\star$ be the solution to the regularized problem. The Wasserstein distance is approximated using the following primal and dual Sinkhorn divergences:

$$
\mathfrak{P}_\mathbf{C}^\varepsilon(\mathbf{a}, \mathbf{b}) \stackrel{\text{def}}{=} \langle \mathbf{C}, \mathbf{P}^\star \rangle = \langle e^{\frac{\mathbf{f}^\star}{\varepsilon}},\, (\mathbf{K} \odot \mathbf{C}) e^{\frac{\mathbf{g}^\star}{\varepsilon}} \rangle,
$$

$$
\mathfrak{D}_\mathbf{C}^\varepsilon(\mathbf{a}, \mathbf{b}) \stackrel{\text{def}}{=} \langle \mathbf{f}^\star, \mathbf{a} \rangle + \langle \mathbf{g}^\star, \mathbf{b} \rangle,
$$

where $\odot$ stands for the elementwise product of matrices.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.7</span><span class="math-callout__name">(Sinkhorn Divergence Bounds)</span></p>

The following relationship holds:

$$
\mathfrak{D}_\mathbf{C}^\varepsilon(\mathbf{a}, \mathbf{b}) \le \mathrm{L}_\mathbf{C}^\varepsilon(\mathbf{a}, \mathbf{b}) \le \mathfrak{P}_\mathbf{C}^\varepsilon(\mathbf{a}, \mathbf{b}).
$$

Furthermore

$$
\mathfrak{P}_\mathbf{C}^\varepsilon(\mathbf{a}, \mathbf{b}) - \mathfrak{D}_\mathbf{C}^\varepsilon(\mathbf{a}, \mathbf{b}) = \varepsilon(\mathbf{H}(\mathbf{P}^\star) + 1).
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.8</span><span class="math-callout__name">(Finite Sinkhorn Divergences)</span></p>

When a predetermined number of $L$ iterations is set and used to evaluate $\mathfrak{D}_\mathbf{C}^\varepsilon$ using iterates $\mathbf{f}^{(L)}$ and $\mathbf{g}^{(L)}$ instead of optimal solutions $\mathbf{f}^\star$ and $\mathbf{g}^\star$, one recovers a lower bound. Defining the finite step approximation

$$
\mathfrak{D}_\mathbf{C}^{(L)}(\mathbf{a}, \mathbf{b}) \stackrel{\text{def}}{=} \langle \mathbf{f}^{(L)}, \mathbf{a} \rangle + \langle \mathbf{g}^{(L)}, \mathbf{b} \rangle,
$$

this "algorithmic" Sinkhorn functional lower bounds the regularized cost function as soon as $L \ge 1$:

$$
\mathfrak{D}_\mathbf{C}^{(L)}(\mathbf{a}, \mathbf{b}) \le \mathrm{L}_\mathbf{C}^\varepsilon(\mathbf{a}, \mathbf{b}).
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.26</span><span class="math-callout__name">(Primal Infeasibility of the Sinkhorn Iterates)</span></p>

Note that the primal iterates are not primal feasible, since, by definition, these iterates are designed to satisfy upon convergence marginal constraints. Therefore, it is not valid to consider $\langle \mathbf{C}, \mathbf{P}^{(2L+1)} \rangle$ as an approximation of $\mathrm{L}_\mathbf{C}(\mathbf{a}, \mathbf{b})$ since $\mathbf{P}^{(2L+1)}$ is not feasible. Using the rounding scheme of Altschuler et al. [2017], one can, however, yield an upper bound on $\mathrm{L}_\mathbf{C}^\varepsilon(\mathbf{a}, \mathbf{b})$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.27</span><span class="math-callout__name">(Nonconvexity of Finite Dual Sinkhorn Divergence)</span></p>

Unlike the regularized expression $\mathrm{L}_\mathbf{C}^\varepsilon$, the finite Sinkhorn divergence $\mathfrak{D}_\mathbf{C}^{(L)}(\mathbf{a}, \mathbf{b})$ is *not*, in general, a convex function of its arguments (this can be easily checked numerically). $\mathfrak{D}_\mathbf{C}^{(L)}(\mathbf{a}, \mathbf{b})$ is, however, a differentiable function which can be differentiated using automatic differentiation techniques with respect to any of its arguments, notably $\mathbf{C}, \mathbf{a}$, or $\mathbf{b}$.

</div>

### Generalized Sinkhorn

The regularized OT problem is a special case of a structured convex optimization problem of the form

$$
\min_\mathbf{P} \sum_{i,j} \mathbf{C}_{i,j}\mathbf{P}_{i,j} - \varepsilon \mathbf{H}(\mathbf{P}) + F(\mathbf{P}\mathbb{1}_m) + G(\mathbf{P}^\mathrm{T}\mathbb{1}_n).
$$

Indeed, defining $F = \iota_{\lbrace\mathbf{a}\rbrace}$ and $G = \iota_{\lbrace\mathbf{b}\rbrace}$, where the indicator function of a closed convex set $\mathcal{C}$ is

$$
\iota_\mathcal{C}(x) = \begin{cases} 0 & \text{if } x \in \mathcal{C}, \\ +\infty & \text{otherwise}, \end{cases}
$$

one retrieves the hard marginal constraints defining $\mathcal{U}(\mathbf{a}, \mathbf{b})$. Sinkhorn iterations can hence be extended to this more general problem, and they read

$$
\mathbf{u} \leftarrow \frac{\mathrm{Prox}_F^{\mathrm{KL}}(\mathbf{K}\mathbf{v})}{\mathbf{K}\mathbf{v}} \quad \text{and} \quad \mathbf{v} \leftarrow \frac{\mathrm{Prox}_G^{\mathrm{KL}}(\mathbf{K}^\mathrm{T}\mathbf{u})}{\mathbf{K}^\mathrm{T}\mathbf{u}},
$$

where the proximal operator for the $\mathrm{KL}$ divergence is

$$
\forall\, \mathbf{u} \in \mathbb{R}_+^N, \quad \mathrm{Prox}_F^{\mathrm{KL}}(\mathbf{u}) = \operatorname{argmin}_{\mathbf{u}' \in \mathbb{R}_+^N} \mathrm{KL}(\mathbf{u}'|\mathbf{u}) + F(\mathbf{u}').
$$

Iterations are thus interesting in the cases where $\mathrm{Prox}_F^{\mathrm{KL}}$ and $\mathrm{Prox}_G^{\mathrm{KL}}$ can be computed in closed form or very efficiently. This is in particular the case for separable functions of the form $F(\mathbf{u}) = \sum_i F_i(\mathbf{u}_i)$ since in this case $\mathrm{Prox}_F^{\mathrm{KL}}(\mathbf{u}) = \left(\mathrm{Prox}_{F_i}^{\mathrm{KL}}(\mathbf{u}_i)\right)_i$.

This algorithm can be used to approximate the solution to various generalizations of OT, and in particular unbalanced OT problems and gradient flow problems.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.28</span><span class="math-callout__name">(Duality and Legendre Transform)</span></p>

The dual problem to the generalized Sinkhorn reads

$$
\max_{\mathbf{f}, \mathbf{g}} -F^*(\mathbf{f}) - G^*(\mathbf{g}) - \varepsilon \sum_{i,j} e^{\frac{\mathbf{f}_i + \mathbf{g}_j - \mathbf{C}_{i,j}}{\varepsilon}},
$$

so that $(\mathbf{u}, \mathbf{v}) = (e^{\mathbf{f}/\varepsilon}, e^{\mathbf{g}/\varepsilon})$ are the associated scalings. Here, $F^*$ and $G^*$ are the Fenchel--Legendre conjugates, which are convex functions defined as

$$
\forall\, \mathbf{f} \in \mathbb{R}^n, \quad F^*(\mathbf{f}) \stackrel{\text{def}}{=} \max_{\mathbf{a} \in \mathbb{R}^n} \langle \mathbf{f}, \mathbf{a} \rangle - F(\mathbf{a}).
$$

The generalized Sinkhorn iterates are a special case of Dykstra's algorithm (extended to Bregman divergence) and is an alternate maximization scheme on the dual problem.

</div>

## Chapter 5: Semidiscrete Optimal Transport

This chapter studies methods to tackle the optimal transport problem when one of the two input measures is discrete (a sum of Dirac masses) and the other one is arbitrary, including notably the case where it has a density with respect to the Lebesgue measure. When the ambient space has low dimension, this problem has a strong geometrical flavor because one can show that the optimal transport from a continuous density toward a discrete one is a piecewise constant map, where the preimage of each point in the support of the discrete measure is a union of disjoint cells. When the cost is the squared Euclidean distance, these cells correspond to the so-called Laguerre cells, which are Voronoi cells offset by a constant. This connection allows us to borrow tools from computational geometry to obtain fast computational schemes. In high dimensions, the semidiscrete formulation can also be interpreted as a stochastic programming problem, which can also benefit from a bit of regularization.

### $c$-Transform and $\bar{c}$-Transform

Recall that the dual OT problem reads

$$
\sup_{(f,g)} \mathcal{E}(f,g) \stackrel{\text{def}}{=} \int_\mathcal{X} f(x)\mathrm{d}\alpha(x) + \int_\mathcal{Y} g(y)\mathrm{d}\beta(y) + \iota_{\mathcal{R}(c)}(f,g),
$$

where we used the indicator function notation. Keeping either dual potential $f$ or $g$ fixed and optimizing w.r.t. $g$ or $f$, respectively, leads to closed form solutions that provide the definition of the $c$-transform:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.1</span><span class="math-callout__name">($c$-Transform and $\bar{c}$-Transform)</span></p>

$$
\forall\, y \in \mathcal{Y}, \quad f^c(y) \stackrel{\text{def}}{=} \inf_{x \in \mathcal{X}}\, c(x,y) - f(x),
$$

$$
\forall\, x \in \mathcal{X}, \quad g^{\bar{c}}(x) \stackrel{\text{def}}{=} \inf_{y \in \mathcal{Y}}\, c(x,y) - g(y),
$$

where we denoted $\bar{c}(y, x) \stackrel{\text{def}}{=} c(x, y)$. Indeed, one can check that

$$
f^c \in \operatorname{argmax}_g \mathcal{E}(f,g) \quad \text{and} \quad g^{\bar{c}} \in \operatorname{argmax}_f \mathcal{E}(f,g).
$$

</div>

Note that these partial minimizations define maximizers on the support of respectively $\alpha$ and $\beta$, while the definitions actually define functions on the whole spaces $\mathcal{X}$ and $\mathcal{Y}$. This is thus a way to extend in a canonical way solutions of the dual on the whole spaces. When $\mathcal{X} = \mathbb{R}^d$ and $c(x,y) = \|x - y\|_q^p = (\sum_{i=1}^d |x_i - y_i|)^{p/2}$, then the $c$-transform $f^c$ is the so-called inf-convolution between $-f$ and $\|\cdot\|^p$. The definition of $f^c$ is also often referred to as a "Hopf--Lax formula."

The map $(f,g) \in \mathcal{C}(\mathcal{X}) \times \mathcal{C}(\mathcal{Y}) \mapsto (g^{\bar{c}}, f^c) \in \mathcal{C}(\mathcal{X}) \times \mathcal{C}(\mathcal{Y})$ replaces dual potentials by "better" ones (improving the dual objective $\mathcal{E}$). Functions that can be written in the form $f^c$ and $g^{\bar{c}}$ are called $c$-concave and $\bar{c}$-concave functions. In the special case $c(x,y) = \langle x, y \rangle$ in $\mathcal{X} = \mathcal{Y} = \mathbb{R}^d$, this definition coincides with the usual notion of concave functions. Extending naturally Proposition 3.1 to a continuous case, one has the property that

$$
f^{c\bar{c}c} = f^c \quad \text{and} \quad g^{\bar{c}c\bar{c}} = g^{\bar{c}},
$$

where we denoted $f^{c\bar{c}} = (f^c)^{\bar{c}}$. This invariance property shows that one can "improve" only once the dual potential this way. Alternatively, this means that alternate maximization does not converge (it immediately enters a cycle), which is classical for functionals involving a nonsmooth (a constraint) coupling of the optimized variables. This is in sharp contrast with entropic regularization of OT as shown in Chapter 4. In this case, because of the regularization, the dual objective is smooth, and alternate maximization corresponds to Sinkhorn iterations. These iterates, written over the dual variables, define entropically smoothed versions of the $c$-transform, where min operations are replaced by a "soft-min."

Using the $c$-transforms, one can reformulate the dual as an unconstrained convex program over a single potential,

$$
\mathcal{L}_c(\alpha, \beta) = \sup_{f \in \mathcal{C}(\mathcal{X})} \int_\mathcal{X} f(x)\mathrm{d}\alpha(x) + \int_\mathcal{Y} f^c(y)\mathrm{d}\beta(y)
$$

$$
= \sup_{g \in \mathcal{C}(\mathcal{Y})} \int_\mathcal{X} g^{\bar{c}}(x)\mathrm{d}\alpha(x) + \int_\mathcal{Y} g(y)\mathrm{d}\beta(y).
$$

### Semidiscrete Formulation

A case of particular interest is when $\beta = \sum_j \mathbf{b}_j \delta_{y_j}$ is discrete (of course the same construction applies if $\alpha$ is discrete by exchanging the role of $\alpha, \beta$). One can adapt the definition of the $\bar{c}$ transform to this setting by restricting the minimization to the support $(y_j)_j$ of $\beta$,

$$
\forall\, \mathbf{g} \in \mathbb{R}^m,\; \forall\, x \in \mathcal{X}, \quad \mathbf{g}^{\bar{c}}(x) \stackrel{\text{def}}{=} \min_{j \in [\![m]\!]} c(x, y_j) - \mathbf{g}_j.
$$

This transform maps a vector $\mathbf{g}$ to a continuous function $\mathbf{g}^{\bar{c}} \in \mathcal{C}(\mathcal{X})$.

Crucially, using the discrete $\bar{c}$-transform in the semidiscrete problem yields a finite-dimensional optimization,

$$
\mathcal{L}_c(\alpha, \beta) = \max_{\mathbf{g} \in \mathbb{R}^m} \mathcal{E}(\mathbf{g}) \stackrel{\text{def}}{=} \int_\mathcal{X} \mathbf{g}^{\bar{c}}(x)\mathrm{d}\alpha(x) + \sum_j \mathbf{g}_j \mathbf{b}_j.
$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.2</span><span class="math-callout__name">(Laguerre Cells)</span></p>

The Laguerre cells associated to the dual weights $\mathbf{g}$ are

$$
\mathbb{L}_j(\mathbf{g}) \stackrel{\text{def}}{=} \left\lbrace x \in \mathcal{X} \;:\; \forall\, j' \neq j,\; c(x, y_j) - \mathbf{g}_j \le c(x, y_{j'}) - \mathbf{g}_{j'} \right\rbrace.
$$

They induce a disjoint decomposition of $\mathcal{X} = \bigcup_j \mathbb{L}_j(\mathbf{g})$. When $\mathbf{g}$ is constant, the Laguerre cells decomposition corresponds to the Voronoi diagram partition of the space.

</div>

This allows one to conveniently rewrite the minimized energy as

$$
\mathcal{E}(\mathbf{g}) = \sum_{j=1}^m \int_{\mathbb{L}_j(\mathbf{g})} \left(c(x, y_j) - \mathbf{g}_j\right) \mathrm{d}\alpha(x) + \langle \mathbf{g}, \mathbf{b} \rangle.
$$

The gradient of this function can be computed as follows:

$$
\forall\, j \in [\![m]\!], \quad \nabla \mathcal{E}(\mathbf{g})_j = -\int_{\mathbb{L}_j(\mathbf{g})} \mathrm{d}\alpha(x) + \mathbf{b}_j.
$$

Once the optimal $\mathbf{g}$ is computed, the optimal transport map $T$ from $\alpha$ to $\beta$ is mapping any $x \in \mathbb{L}_j(\mathbf{g})$ toward $y_j$, so it is piecewise constant.

In the special case $c(x,y) = \|x - y\|^2$, the decomposition in Laguerre cells is also known as a "power diagram." The cells are polyhedral and can be computed efficiently using computational geometry algorithms. The most widely used algorithm relies on the fact that the power diagram of points in $\mathbb{R}^d$ is equal to the projection on $\mathbb{R}^d$ of the convex hull of the set of points $((y_j, \|y_j\|^2 - \mathbf{g}_j))_{j=1}^m \subset \mathbb{R}^{d+1}$.

### Entropic Semidiscrete Formulation

The dual of the entropic regularized problem between arbitrary measures is a smooth unconstrained optimization problem:

$$
\mathcal{L}_c^\varepsilon(\alpha, \beta) = \sup_{(f,g) \in \mathcal{C}(\mathcal{X}) \times \mathcal{C}(\mathcal{Y})} \int_\mathcal{X} f\,\mathrm{d}\alpha + \int_\mathcal{Y} g\,\mathrm{d}\beta - \varepsilon \int_{\mathcal{X} \times \mathcal{Y}} e^{\frac{-c + f \oplus g}{\varepsilon}} \mathrm{d}\alpha\,\mathrm{d}\beta,
$$

where we denoted $(f \oplus g)(x,y) \stackrel{\text{def}}{=} f(x) + g(y)$.

Similarly to the unregularized problem, one can minimize explicitly with respect to either $f$ or $g$, which yields a smoothed $c$-transform

$$
\forall\, y \in \mathcal{Y}, \quad f^{c,\varepsilon}(y) \stackrel{\text{def}}{=} -\varepsilon \log\left(\int_\mathcal{X} e^{\frac{-c(x,y) + f(x)}{\varepsilon}} \mathrm{d}\alpha(x)\right),
$$

$$
\forall\, x \in \mathcal{X}, \quad g^{\bar{c},\varepsilon}(x) \stackrel{\text{def}}{=} -\varepsilon \log\left(\int_\mathcal{Y} e^{\frac{-c(x,y) + g(y)}{\varepsilon}} \mathrm{d}\beta(y)\right).
$$

Note that the rewriting of Sinkhorn using the soft-min operator $\min_\varepsilon$ corresponds to the alternate computation of entropic smoothed $c$-transforms.

In the case of a discrete measure $\beta = \sum_{j=1}^m \mathbf{b}_j \delta_{y_j}$, the problem simplifies to a finite-dimensional problem expressed as a function of the discrete dual potential $\mathbf{g} \in \mathbb{R}^m$,

$$
\forall\, x \in \mathcal{X}, \quad \mathbf{g}^{\bar{c},\varepsilon}(x) \stackrel{\text{def}}{=} -\varepsilon \log\left(\sum_{j=1}^m e^{\frac{-c(x,y_j) + \mathbf{g}_j}{\varepsilon}} \mathbf{b}_j\right).
$$

Instead of maximizing the general dual, one can thus solve the following finite-dimensional optimization problem:

$$
\max_{\mathbf{g} \in \mathbb{R}^m} \mathcal{E}^\varepsilon(\mathbf{g}) \stackrel{\text{def}}{=} \int_\mathcal{X} \mathbf{g}^{\bar{c},\varepsilon}(x)\mathrm{d}\alpha(x) + \langle \mathbf{g}, \mathbf{b} \rangle.
$$

Note that this optimization problem is still valid even in the unregularized case $\varepsilon = 0$ and in this case $\mathbf{g}^{\bar{c},\varepsilon=0} = \mathbf{g}^{\bar{c}}$ is the $\bar{c}$-transform so that the problem is in fact the semidiscrete formulation. The gradient of this functional reads

$$
\forall\, j \in [\![m]\!], \quad \nabla \mathcal{E}^\varepsilon(\mathbf{g})_j = -\int_\mathcal{X} \chi_j^\varepsilon(x)\mathrm{d}\alpha(x) + \mathbf{b}_j,
$$

where $\chi_j^\varepsilon$ is a smoothed version of the indicator $\chi_j^0$ of the Laguerre cell $\mathbb{L}_j(\mathbf{g})$,

$$
\chi_j^\varepsilon(x) = \frac{e^{\frac{-c(x,y_j) + \mathbf{g}_j}{\varepsilon}}}{\sum_\ell e^{\frac{-c(x,y_\ell) + \mathbf{g}_\ell}{\varepsilon}}}.
$$

Note that the family of functions $(\chi_j^\varepsilon)_j$ is a partition of unity, i.e. $\sum_j \chi_j^\varepsilon = 1$ and $\chi_j^\varepsilon \ge 0$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.1</span><span class="math-callout__name">(Second Order Methods and Connection with Logistic Regression)</span></p>

A crucial aspect of the smoothed semidiscrete formulation is that it corresponds to the minimization of a smooth function. Indeed, as shown in [Genevay et al., 2016], the Hessian of $\mathcal{E}^\varepsilon$ is upper bounded by $1/\varepsilon$, so that $\nabla \mathcal{E}^\varepsilon$ is $\frac{1}{\varepsilon}$-Lipschitz continuous. In fact, that problem is very closely related to a multiclass logistic regression problem and enjoys the same favorable properties (generalizations of self-concordance). In particular, the Newton method converges quadratically, and one can use in practice quasi-Newton techniques, such as L-BFGS. The use of second order schemes (Newton or L-BFGS) is also advocated in the unregularized case $\varepsilon = 0$. In [Kitagawa et al., 2016, Theo. 5.1], the Hessian of $\mathcal{E}^0(\mathbf{g})$ is shown to be uniformly bounded as long as the volume of the Laguerre cells is bounded by below and $\alpha$ has a continuous density.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.2</span><span class="math-callout__name">(Legendre Transforms of OT Cost Functions)</span></p>

As stated in Proposition 4.6, $\mathrm{L}_\mathbf{C}^\varepsilon(\mathbf{a}, \mathbf{b})$ is a convex function of $(\mathbf{a}, \mathbf{b})$ (which is also true in the unregularized case $\varepsilon = 0$). It is thus possible to compute its Legendre--Fenchel transform. Denoting $F_\mathbf{a}(\mathbf{b}) = \mathrm{L}_\mathbf{C}^\varepsilon(\mathbf{a}, \mathbf{b})$, one has, for a fixed $\mathbf{a}$:

$$
F_\mathbf{a}^*(\mathbf{g}) = -\varepsilon H(\mathbf{a}) + \sum_i \mathbf{a}_i \mathbf{g}^{\bar{c},\varepsilon}(x_i).
$$

Here $\mathbf{g}^{\bar{c},\varepsilon}$ is the entropic-smoothed $c$-transform. In the unregularized case $\varepsilon = 0$, and for generic measures, Carlier et al. [2015] show, denoting $\mathcal{F}_\alpha(\beta) \stackrel{\text{def}}{=} \mathcal{L}_c(\alpha, \beta)$,

$$
\forall\, g \in \mathcal{C}(\mathcal{Y}), \quad \mathcal{F}_\alpha^*(g) = \int_\mathcal{X} g^{\bar{c}}(x)\mathrm{d}\alpha(x),
$$

where the $\bar{c}$-transform $g^{\bar{c}} \in \mathcal{C}(\mathcal{X})$ of $g$ is defined in §5.1. Note that here, since $\mathcal{M}(\mathcal{X})$ is in duality with $\mathcal{C}(\mathcal{X})$, the Legendre transform is a function of continuous functions. For the discrete case, denoting $G(\mathbf{a}, \mathbf{b}) \stackrel{\text{def}}{=} \mathrm{L}_\mathbf{C}^\varepsilon(\mathbf{a}, \mathbf{b})$, one can derive the Legendre transform for both arguments,

$$
\forall\, (\mathbf{f}, \mathbf{g}) \in \mathbb{R}^n \times \mathbb{R}^m, \quad G^*(\mathbf{f}, \mathbf{g}) = -\varepsilon \log \sum_{i,j} e^{\frac{-\mathbf{C}_{i,j} + \mathbf{f}_i + \mathbf{g}_j}{\varepsilon}},
$$

which can be seen as a smoothed version of the Legendre transform of $\mathcal{G}(\alpha, \beta) \stackrel{\text{def}}{=} \mathcal{L}_c(\alpha, \beta)$,

$$
\forall\, (f,g) \in \mathcal{C}(\mathcal{X}) \times \mathcal{C}(\mathcal{Y}), \quad \mathcal{G}^*(f,g) = \inf_{(x,y) \in \mathcal{X} \times \mathcal{Y}} c(x,y) - f(x) - g(y).
$$

</div>

### Stochastic Optimization Methods

The semidiscrete formulation and its smoothed version are appealing because the energies to be minimized are written as an expectation with respect to the probability distribution $\alpha$,

$$
\mathcal{E}^\varepsilon(\mathbf{g}) = \int_\mathcal{X} E^\varepsilon(\mathbf{g}, x)\mathrm{d}\alpha(x) = \mathbb{E}_X(E^\varepsilon(\mathbf{g}, X)),
$$

where $E^\varepsilon(\mathbf{g}, x) \stackrel{\text{def}}{=} \mathbf{g}^{\bar{c},\varepsilon}(x) - \langle \mathbf{g}, \mathbf{b} \rangle$, and $X$ denotes a random vector distributed on $\mathcal{X}$ according to $\alpha$. Note that the gradient of each of the involved functional reads

$$
\nabla_\mathbf{g} E^\varepsilon(x, \mathbf{g}) = (\chi_j^\varepsilon(x) - \mathbf{b}_j)_{j=1}^m \in \mathbb{R}^m.
$$

One can thus use stochastic optimization methods to perform the maximization, as proposed in Genevay et al. [2016]. This allows us to obtain provably convergent algorithms without the need to resort to an arbitrary discretization of $\alpha$ using sums of Diracs or using quadrature formula for the integrals. The measure $\alpha$ is used as a black box from which one can draw independent samples, which is a natural computational setup for many high-dimensional applications in statistics and machine learning.

**Stochastic gradient descent.** Initializing $\mathbf{g}^{(0)} = \mathbb{0}_P$, the stochastic gradient descent algorithm (SGD; used here as a maximization method) draws at step $\ell$ a point $x_\ell \in \mathcal{X}$ according to distribution $\alpha$ (independently from all past and future samples $(x_\ell)_\ell$) to form the update

$$
\mathbf{g}^{(\ell+1)} \stackrel{\text{def}}{=} \mathbf{g}^{(\ell)} + \tau_\ell \nabla_\mathbf{g} E^\varepsilon(\mathbf{g}^{(\ell)}, x_\ell).
$$

The step size $\tau_\ell$ should decay fast enough to zero in order to ensure that the "noise" created by using $\nabla_\mathbf{g} E^\varepsilon(x_\ell, \mathbf{g})$ as a proxy for the true gradient $\nabla \mathcal{E}^\varepsilon(\mathbf{g})$ is canceled in the limit. A typical choice of schedule is

$$
\tau_\ell \stackrel{\text{def}}{=} \frac{\tau_0}{1 + \ell / \ell_0},
$$

where $\ell_0$ indicates roughly the number of iterations serving as a warmup phase. One can prove the convergence result

$$
\mathcal{E}^\varepsilon(\mathbf{g}^\star) - \mathbb{E}(\mathcal{E}^\varepsilon(\mathbf{g}^{(\ell)})) = O\left(\frac{1}{\sqrt{\ell}}\right),
$$

where $\mathbf{g}^\star$ is a solution and where $\mathbb{E}$ indicates an expectation with respect to the i.i.d. sampling of $(x_\ell)_\ell$ performed at each iteration.

**Stochastic gradient descent with averaging.** SGD is slow because of the fast decay of the stepsize $\tau_\ell$ toward zero. To improve the convergence speed, it is possible to average the past iterates, which is equivalent to running a "classical" SGD on auxiliary variables $(\tilde{\mathbf{g}}^{(\ell)})_\ell$

$$
\tilde{\mathbf{g}}^{(\ell+1)} \stackrel{\text{def}}{=} \tilde{\mathbf{g}}^{(\ell)} + \tau_\ell \nabla_\mathbf{g} E^\varepsilon(\tilde{\mathbf{g}}^{(\ell)}, x_\ell),
$$

where $x_\ell$ is drawn according to $\alpha$ (and all the $(x_\ell)_\ell$ are independent) and output as the estimated weight vector the average

$$
\mathbf{g}^{(\ell)} \stackrel{\text{def}}{=} \frac{1}{\ell} \sum_{k=1}^\ell \tilde{\mathbf{g}}^{(k)}.
$$

This defines the stochastic gradient descent with averaging (SGA) algorithm. One can avoid explicitly storing all the iterates by simply updating a running average as follows:

$$
\mathbf{g}^{(\ell+1)} = \frac{1}{\ell+1}\tilde{\mathbf{g}}^{(\ell+1)} + \frac{\ell}{\ell+1}\mathbf{g}^{(\ell)}.
$$

In this case, a typical choice of decay is rather of the form

$$
\tau_\ell \stackrel{\text{def}}{=} \frac{\tau_0}{1 + \sqrt{\ell / \ell_0}}.
$$

Notice that the step size now goes much slower to 0 than for SGD, at rate $\ell^{-1/2}$. Bach [2014] proves that SGA leads to a faster convergence (the constants involved are smaller) than SGD, since in contrast to SGD, SGA is adaptive to the local strong convexity (or concavity for maximization problems) of the functional.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.3</span><span class="math-callout__name">(Continuous-Continuous Problems)</span></p>

When neither $\alpha$ nor $\beta$ is a discrete measure, one cannot resort to semidiscrete strategies involving finite-dimensional dual variables. The only option is to use stochastic optimization methods on the dual problem, as proposed in [Genevay et al., 2016]. A suitable regularization of that problem is crucial, for instance by setting an entropic regularization strength $\varepsilon > 0$, to obtain an unconstrained problem that can be solved by stochastic descent schemes. A possible approach to revisit the infinite-dimensional optimization problem over a space of continuous functions is to restrict it to a much smaller subset, such as that spanned by multilayer neural networks [Seguy et al., 2018]. This approach leads to nonconvex finite-dimensional optimization problems with no approximation guarantees, but this can provide an effective way to compute a proxy for the Wasserstein distance in high-dimensional scenarios. Another solution is to use nonparametric families, which is equivalent to considering some sort of progressive refinement, as that proposed by Genevay et al. [2016] using reproducing kernel Hilbert spaces, whose dimension is proportional to the number of iterations of the SGD algorithm.

</div>

## Chapter 6: $\mathcal{W}_1$ Optimal Transport

This chapter focuses on optimal transport problems in which the ground cost is equal to a distance. Historically, this corresponds to the original problem posed by Monge in 1781; this setting was also that chosen in early applications of optimal transport in computer vision under the name of "earth mover's distances".

Unlike the case where the ground cost is a *squared* Hilbertian distance (studied in particular in Chapter 7), transport problems where the cost is a metric are more difficult to analyze theoretically. In contrast to Remark 2.24 that states the uniqueness of a transport map or coupling between two absolutely continuous measures when using a squared metric, the optimal Kantorovich coupling is in general not unique when the cost is the ground distance itself. Hence, in this regime it is often impossible to recover a uniquely defined Monge map, making this class of problems ill-suited for interpolation of measures.

Although more difficult to analyze in theory, optimal transport with a linear ground distance is usually more robust to outliers and noise than a quadratic cost. Furthermore, a cost that is a metric results in an elegant dual reformulation involving local flow, divergence constraints, or Lipschitzness of the dual potential, suggesting cheaper numerical algorithms that align with *minimum-cost flow* methods over networks in graph theory. This setting is also popular because the associated OT distances define a norm that can compare arbitrary distributions, even if they are not positive; this property is shared by a larger class of so-called *dual norms*.

### $\mathcal{W}_1$ on Metric Spaces

Here we assume that $d$ is a distance on $\mathcal{X} = \mathcal{Y}$, and we solve the OT problem with the ground cost $c(x, y) = d(x, y)$. We denote the Lipschitz constant of a function $f \in \mathcal{C}(\mathcal{X})$ as

$$
\mathrm{Lip}(f) \stackrel{\text{def}}{=} \sup \left\lbrace \frac{|f(x) - f(y)|}{d(x, y)} \;:\; (x, y) \in \mathcal{X}^2,\, x \neq y \right\rbrace.
$$

We define Lipschitz functions to be those functions $f$ satisfying $\mathrm{Lip}(f) < +\infty$; they form a convex subset of $\mathcal{C}(\mathcal{X})$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.1</span><span class="math-callout__name">(Lipschitz $c$-Transforms)</span></p>

Suppose $\mathcal{X} = \mathcal{Y}$ and $c(x, y) = d(x, y)$. Then, there exists $g$ such that $f = g^c$ if and only $\mathrm{Lip}(f) \le 1$. Furthermore, if $\mathrm{Lip}(f) \le 1$, then $f^c = -f$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Proposition 6.1</summary>

First, suppose $f = g^c$. Then, for $x, y \in \mathcal{X}$,

$$
|f(x) - f(y)| = \left|\inf_{z \in \mathcal{X}} d(x, z) - g(z) \;-\; \inf_{z \in \mathcal{X}} d(y, z) - g(z)\right| \le \sup_{z \in \mathcal{X}} |d(x, z) - d(y, z)| \le d(x, y).
$$

The first equality follows from the definition of $g^c$, the next inequality from the identity $|\inf f - \inf g| \le \sup |f - g|$, and the last from the triangle inequality. This shows that $\mathrm{Lip}(f) \le 1$.

Now, suppose $\mathrm{Lip}(f) \le 1$, and define $g \stackrel{\text{def}}{=} -f$. By the Lipschitz property, for all $x, y \in \mathcal{X}$, $f(y) - d(x, y) \le f(x) \le f(y) + d(x, y)$. Applying these inequalities,

$$
g^c(y) = \inf_{x \in \mathcal{X}} [d(x, y) + f(x)] \ge \inf_{x \in \mathcal{X}} [d(x, y) + f(y) - d(x, y)] = f(y),
$$

$$
g^c(y) = \inf_{x \in \mathcal{X}} [d(x, y) + f(x)] \le \inf_{x \in \mathcal{X}} [d(x, y) + f(y) + d(x, y)] = f(y).
$$

Hence, $f = g^c$ with $g = -f$. Using the same inequalities shows

$$
f^c(y) = \inf_{x \in \mathcal{X}} [d(x, y) - f(x)] \ge \inf_{x \in \mathcal{X}} [d(x, y) - f(y) - d(x, y)] = -f(y),
$$

$$
f^c(y) = \inf_{x \in \mathcal{X}} [d(x, y) - f(x)] \le \inf_{x \in \mathcal{X}} [d(x, y) - f(y) + d(x, y)] = -f(y).
$$

This shows $f^c = -f$.

</details>
</div>

Starting from the single potential formulation (5.4), one can iterate the construction and replace the couple $(g, g^c)$ by $(g^c, (g^c)^c)$. The last proposition shows that one can thus use $(g^c, -g^c)$, which in turn is equivalent to any pair $(f, -f)$ such that $\mathrm{Lip}(f) \le 1$. This leads to the following alternative expression for the $\mathcal{W}_1$ distance:

$$
\mathcal{W}_1(\alpha, \beta) = \max_f \left\lbrace \int_\mathcal{X} f(x)(\mathrm{d}\alpha(x) - \mathrm{d}\beta(x)) \;:\; \mathrm{Lip}(f) \le 1 \right\rbrace. \tag{6.1}
$$

This expression shows that $\mathcal{W}_1$ is actually a norm, *i.e.* $\mathcal{W}_1(\alpha, \beta) = \|\alpha - \beta\|\_{\mathcal{W}_1}$, and that it is still valid for any measures (not necessary positive) as long as $\int_\mathcal{X} \alpha = \int_\mathcal{X} \beta$. This norm is often called the Kantorovich and Rubinstein norm.

For discrete measures of the form (2.1), writing $\alpha - \beta = \sum_k \mathbf{m}_k \delta_{z_k}$ with $z_k \in \mathcal{X}$ and $\sum_k \mathbf{m}_k = 0$, the optimization (6.1) can be rewritten as

$$
\mathcal{W}_1(\alpha, \beta) = \max_{(\mathbf{f}_k)_k} \left\lbrace \sum_k \mathbf{f}_k \mathbf{m}_k \;:\; \forall\,(k, \ell),\, |\mathbf{f}_k - \mathbf{f}_\ell| \le d(z_k, z_\ell) \right\rbrace, \tag{6.2}
$$

which is a finite-dimensional convex program with quadratic-cone constraints. It can be solved using interior point methods or, as we detail next for a similar problem, using proximal methods.

When using $d(x, y) = |x - y|$ with $\mathcal{X} = \mathbb{R}$, we can reduce the number of constraints by ordering the $z_k$'s via $z_1 \le z_2 \le \ldots$ In this case, we only have to solve

$$
\mathcal{W}_1(\alpha, \beta) = \max_{(\mathbf{f}_k)_k} \left\lbrace \sum_k \mathbf{f}_k \mathbf{m}_k \;:\; \forall\,k,\, |\mathbf{f}_{k+1} - \mathbf{f}_k| \le z_{k+1} - z_k \right\rbrace,
$$

which is a linear program. Note that furthermore, in this 1-D case, a closed form expression for $\mathcal{W}_1$ using cumulative functions is given in (2.37).

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 6.1</span><span class="math-callout__name">($\mathcal{W}_p$ with $0 < p \le 1$)</span></p>

If $0 < p \le 1$, then $\tilde{d}(x, y) \stackrel{\text{def}}{=} d(x, y)^p$ satisfies the triangular inequality, and hence $\tilde{d}$ is itself a distance. One can thus apply the results and algorithms detailed above for $\mathcal{W}_1$ to compute $\mathcal{W}_p$ by simply using $\tilde{d}$ in place of $d$. This is equivalent to stating that $\mathcal{W}_p$ is the dual of $p$-Hölder functions $\lbrace f : \mathrm{Lip}_p(f) \le 1 \rbrace$, where

$$
\mathrm{Lip}_p(f) \stackrel{\text{def}}{=} \sup \left\lbrace \frac{|f(x) - f(y)|}{d(x, y)^p} \;:\; (x, y) \in \mathcal{X}^2,\, x \neq y \right\rbrace.
$$

</div>

### $\mathcal{W}_1$ on Euclidean Spaces

In the special case of Euclidean spaces $\mathcal{X} = \mathcal{Y} = \mathbb{R}^d$, using $c(x, y) = \|x - y\|$, the global Lipschitz constraint appearing in (6.1) can be made local as a uniform bound on the gradient of $f$,

$$
\mathcal{W}_1(\alpha, \beta) = \max_f \left\lbrace \int_{\mathbb{R}^d} f(x)(\mathrm{d}\alpha(x) - \mathrm{d}\beta(x)) \;:\; \|\nabla f\|_\infty \le 1 \right\rbrace. \tag{6.3}
$$

Here the constraint $\|\nabla f\|\_\infty \le 1$ signifies that the norm of the gradient of $f$ at any point $x$ is upper bounded by 1, $\|\nabla f(x)\|\_2 \le 1$ for any $x$.

Considering the dual problem to (6.3), one obtains an optimization problem under fixed divergence constraint

$$
\mathcal{W}_1(\alpha, \beta) = \min_s \left\lbrace \int_{\mathbb{R}^d} \|s(x)\|_2 \,\mathrm{d}x \;:\; \mathrm{div}(s) = \alpha - \beta \right\rbrace, \tag{6.4}
$$

which is often called the Beckmann formulation. Here the vectorial function $s(x) \in \mathbb{R}^2$ can be interpreted as a flow field, describing locally the movement of mass. Outside the support of the two input measures, $\mathrm{div}(s) = 0$, which is the conservation of mass constraint. Once properly discretized using finite elements, Problems (6.3) and (6.4) become nonsmooth convex optimization problems. It is possible to use an off-the-shelf interior points quadratic-cone optimization solver, but as advocated in §7.3, large-scale problems require the use of simpler but more adapted first order methods. One can thus use, for instance, Douglas–Rachford (DR) iterations (7.14) or the related alternating direction method of multipliers method.

### $\mathcal{W}_1$ on a Graph

The previous formulations (6.3) and (6.4) of $\mathcal{W}_1$ can be generalized to the setting where $\mathcal{X}$ is a geodesic space, *i.e.* $c(x, y) = d(x, y)$ where $d$ is a geodesic distance. When $\mathcal{X} = [\![1, n]\!]$ is a discrete set, equipped with undirected edges $(i, j) \in \mathcal{E} \subset \mathcal{X}^2$ labeled with a weight (length) $\mathbf{w}\_{i,j}$, we recover the important case where $\mathcal{X}$ is a graph equipped with the geodesic distance (or shortest path metric):

$$
\mathbf{D}_{i,j} \stackrel{\text{def}}{=} \min_{K \ge 0,\,(i_k)_k : i \to j} \left\lbrace \sum_{k=1}^{K-1} \mathbf{w}_{i_k, i_{k+1}} \;:\; \forall\,k \in [\![1, K-1]\!],\, (i_k, i_{k+1}) \in \mathcal{E} \right\rbrace,
$$

where $i \to j$ indicates that $i_1 = i$ and $i_K = j$, namely that the path starts at $i$ and ends at $j$.

We consider two vectors $(\mathbf{a}, \mathbf{b}) \in (\mathbb{R}^n)^2$ defining (signed) discrete measures on the graph $\mathcal{X}$ such that $\sum_i \mathbf{a}\_i = \sum_i \mathbf{b}\_i$ (these weights do not need to be positive). The goal is now to compute $\mathrm{W}_1(\mathbf{a}, \mathbf{b})$, as introduced in (2.17) for $p = 1$, when the ground metric is the graph geodesic distance. This computation should be carried out without going as far as having to compute a "full" coupling $\mathbf{P}$ of size $n \times n$, to rely instead on local operators thanks to the underlying connectivity of the graph. These operators are discrete formulations for the gradient and divergence differential operators.

A discrete dual Kantorovich potential $\mathbf{f} \in \mathbb{R}^n$ is a vector indexed by all vertices of the graph. The gradient operator $\nabla : \mathbb{R}^n \to \mathbb{R}^\mathcal{E}$ is defined as

$$
\forall\,(i, j) \in \mathcal{E}, \quad (\nabla \mathbf{f})_{i,j} \stackrel{\text{def}}{=} \mathbf{f}_i - \mathbf{f}_j.
$$

A flow $\mathbf{s} = (\mathbf{s}\_{i,j})\_{i,j}$ is defined on edges, and the divergence operator $\mathrm{div} : \mathbb{R}^\mathcal{E} \to \mathbb{R}^n$, which is the adjoint of the gradient $\nabla$, maps flows to vectors defined on vertices and is defined as

$$
\forall\,i \in [\![1, n]\!], \quad \mathrm{div}(\mathbf{s})_i \stackrel{\text{def}}{=} \sum_{j : (i, j) \in \mathcal{E}} (\mathbf{s}_{i,j} - \mathbf{s}_{j,i}) \in \mathbb{R}^n.
$$

Problem (6.3) becomes, in the graph setting,

$$
\mathrm{W}_1(\mathbf{a}, \mathbf{b}) = \max_{\mathbf{f} \in \mathbb{R}^n} \left\lbrace \sum_{i=1}^n \mathbf{f}_i (\mathbf{a}_i - \mathbf{b}_i) \;:\; \forall\,(i, j) \in \mathcal{E},\, |(\nabla \mathbf{f})_{i,j}| \le \mathbf{w}_{i,j} \right\rbrace. \tag{6.5}
$$

The associated dual problem, which is analogous to Formula (6.4), is then

$$
\mathrm{W}_1(\mathbf{a}, \mathbf{b}) = \min_{\mathbf{s} \in \mathbb{R}_+^\mathcal{E}} \left\lbrace \sum_{(i,j) \in \mathcal{E}} \mathbf{w}_{i,j} \mathbf{s}_{i,j} \;:\; \mathrm{div}(\mathbf{s}) = \mathbf{a} - \mathbf{b} \right\rbrace. \text{(6.6)}
$$

This is a linear program and more precisely an instance of min-cost flow problems. Highly efficient dedicated simplex solvers have been devised to solve it. Formulation (6.6) is the so-called Beckmann formulation and has been used and extended to define and study traffic congestion models.

## Chapter 7: Dynamic Formulations

This chapter presents the geodesic (also called dynamic) point of view of optimal transport when the cost is a squared geodesic distance. This describes the optimal transport between two measures as a curve in the space of measures minimizing a total length. The dynamic point of view offers an alternative and intuitive interpretation of optimal transport, which not only allows us to draw links with fluid dynamics but also results in an efficient numerical tool to compute OT in small dimensions when interpolating between two densities. The drawback of that approach is that it cannot scale to large-scale sparse measures and works only in low dimensions on regular domains (because one needs to grid the space) with a squared geodesic cost.

In this chapter, we use the notation $(\alpha_0, \alpha_1)$ in place of $(\alpha, \beta)$ in agreement with the idea that we start at time $t = 0$ from one measure to reach another one at time $t = 1$.

### Continuous Formulation

In the case $\mathcal{X} = \mathcal{Y} = \mathbb{R}^d$, and $c(x,y) = \|x - y\|^2$, the optimal transport distance $\mathcal{W}_2^2(\alpha, \beta) = \mathcal{L}_c(\alpha, \beta)$ as defined in (2.15) can be computed by looking for a minimal length path $(\alpha_t)_{t=0}^1$ between these two measures. This path is described by advecting the measure using a vector field $v_t$ defined at each instant. The vector field $v_t$ and the path $\alpha_t$ must satisfy the conservation of mass formula, resulting in

$$
\frac{\partial \alpha_t}{\partial t} + \mathrm{div}(\alpha_t v_t) = 0 \quad \text{and} \quad \alpha_{t=0} = \alpha_0,\, \alpha_{t=1} = \alpha_1, \tag{7.1}
$$

where the equation above should be understood in the sense of distributions on $\mathbb{R}^d$. The infinitesimal length of such a vector field is measured using the $L^2$ norm associated to the measure $\alpha_t$, that is defined as

$$
\|v_t\|_{L^2(\alpha_t)} = \left( \int_{\mathbb{R}^d} \|v_t(x)\|^2 \,\mathrm{d}\alpha_t(x) \right)^{1/2}.
$$

This definition leads to the following minimal-path reformulation of $\mathcal{W}_2$, originally introduced by Benamou and Brenier [2000]:

$$
\mathcal{W}_2^2(\alpha_0, \alpha_1) = \min_{(\alpha_t, v_t)_t \,\text{sat.}\,(7.1)} \int_0^1 \int_{\mathbb{R}^d} \|v_t(x)\|^2 \,\mathrm{d}\alpha_t(x) \mathrm{d}t, \tag{7.2}
$$

where $\alpha_t$ is a scalar-valued measure and $v_t$ a vector-valued measure.

The formulation (7.2) is a nonconvex formulation in the variables $(\alpha_t, v_t)_t$ because of the constraint (7.1) involving the product $\alpha_t v_t$. Introducing a vector-valued measure (often called the "momentum")

$$
J_t \stackrel{\text{def}}{=} \alpha_t v_t,
$$

Benamou and Brenier showed in their landmark paper [2000] that it is instead convex in the variable $(\alpha_t, J_t)_t$ when writing

$$
\mathcal{W}_2^2(\alpha_0, \alpha_1) = \min_{(\alpha_t, J_t)_t \in \mathcal{C}(\alpha_0, \alpha_1)} \int_0^1 \int_{\mathbb{R}^d} \theta(\alpha_t(x), J_t(x)) \,\mathrm{d}x \mathrm{d}t, \tag{7.3}
$$

where we define the set of constraints as

$$
\mathcal{C}(\alpha_0, \alpha_1) \stackrel{\text{def}}{=} \left\lbrace (\alpha_t, J_t) \;:\; \frac{\partial \alpha_t}{\partial t} + \mathrm{div}(J_t) = 0,\, \alpha_{t=0} = \alpha_0,\, \alpha_{t=1} = \alpha_1 \right\rbrace, \tag{7.4}
$$

and where $\theta :\to \mathbb{R}^+ \cup \lbrace +\infty \rbrace$ is the following lower semicontinuous convex function

$$
\forall\,(a, b) \in \mathbb{R}_+ \times \mathbb{R}^d, \quad \theta(a, b) = \begin{cases} \frac{\|b\|^2}{a} & \text{if } a > 0, \\ 0 & \text{if } (a, b) = 0, \\ +\infty & \text{otherwise.} \end{cases} \tag{7.5}
$$

This definition might seem complicated, but it is crucial to impose that the momentum $J_t(x)$ should vanish when $\alpha_t(x) = 0$. Note also that (7.3) is written in an informal way as if the measures $(\alpha_t, J_t)$ were density functions, but this is acceptable because $\theta$ is a 1-homogeneous function (and hence defined even if the measures do not have a density with respect to Lebesgue measure) and can thus be extended in an unambiguous way from density to functions.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 7.1</span><span class="math-callout__name">(Links with McCann's Interpolation)</span></p>

In the case (see Equation (2.28)) where there exists an optimal Monge map $T : \mathbb{R}^d \to \mathbb{R}^d$ with $T_\sharp \alpha_0 = \alpha_1$, then $\alpha_t$ is equal to McCann's interpolation

$$
\alpha_t = ((1 - t)\mathrm{Id} + tT)_\sharp \alpha_0. \tag{7.6}
$$

In the 1-D case, using Remark 2.30, this interpolation can be computed thanks to the relation

$$
\mathcal{C}_{\alpha_t}^{-1} = (1 - t)\mathcal{C}_{\alpha_0}^{-1} + t\mathcal{C}_{\alpha_1}^{-1}. \tag{7.7}
$$

In the case that there is "only" an optimal coupling $\pi$ that is not necessarily supported on a Monge map, one can compute this interpolant as

$$
\alpha_t = P_{t\sharp}\pi \quad \text{where} \quad P_t : (x, y) \in \mathbb{R}^d \times \mathbb{R}^d \mapsto (1 - t)x + ty. \tag{7.8}
$$

For instance, in the discrete setup (2.3), denoting $\mathbf{P}$ a solution to (2.11), an interpolation is defined as

$$
\alpha_t = \sum_{i,j} \mathbf{P}_{i,j} \delta_{(1-t)x_i + ty_j}. \tag{7.9}
$$

Such an interpolation is typically supported on $n + m - 1$ points, which is the maximum number of nonzero elements of $\mathbf{P}$. This construction can be generalized to geodesic spaces $\mathcal{X}$ by replacing $P_t$ by the interpolation along geodesic paths. McCann's interpolation finds many applications, for instance, color, shape, and illumination interpolations in computer graphics.

</div>

### Discretization on Uniform Staggered Grids

For simplicity, we describe the numerical scheme in dimension $d = 2$; the extension to higher dimensions is straightforward. We follow the discretization method introduced by Papadakis et al. [2014], which is inspired by staggered grid techniques which are commonly used in fluid dynamics. We discretize time as $t_k = k/T \in [0, 1]$ and assume the space is uniformly discretized at points $x_i = (i_1/n_1, i_2/n_2) \in X = [0, 1]^2$. We use a staggered grid representation, so that $\alpha_t$ is represented using $\mathbf{a} \in \mathbb{R}^{(T+1) \times n_1 \times n_2}$ associated to half grid points in time, whereas $J$ is represented using $\mathbf{J} = (\mathbf{J}_1, \mathbf{J}_2)$, where $\mathbf{J}_1 \in \mathbb{R}^{T \times (n_1+1) \times n_2}$ and $\mathbf{J}_2 \in \mathbb{R}^{T \times n_1 \times (n_2+1)}$ are stored at half grid points in each space direction. Using this representation, for $(k, i_1, i_2) \in [\![1, T]\!] \times [\![1, n_1]\!] \times [\![1, n_2]\!]$, the time derivative is computed as

$$
(\partial_t \mathbf{a})_{k,i} \stackrel{\text{def}}{=} \mathbf{a}_{k+1,i} - \mathbf{a}_{k,i}
$$

and spatial divergence as

$$
\mathrm{div}(\mathbf{J})_{k,i} \stackrel{\text{def}}{=} \mathbf{J}^1_{k,i_1+1,i_2} - \mathbf{J}^1_{k,i_1,i_2} + \mathbf{J}^2_{k,i_1,i_2+1} - \mathbf{J}^2_{k,i_1,i_2}, \tag{7.10}
$$

which are both defined at grid points, thus forming arrays of $\mathbb{R}^{T \times n_1 \times n_2}$.

In order to evaluate the functional to be optimized, one needs interpolation operators from midgrid points to grid points. The simplest choice is to use a linear operator $\mathcal{I}(r, s) = \frac{r+s}{2}$, which is the one we consider next. The discrete counterpart to (7.3) reads

$$
\min_{(\mathbf{a}, \mathbf{J}) \in \mathbf{C}(\mathbf{a}_0, \mathbf{a}_1)} \Theta(\mathcal{I}_a(\mathbf{a}), \mathcal{I}_J(\mathbf{J})), \tag{7.11}
$$

where $\Theta(\tilde{\mathbf{a}}, \tilde{\mathbf{J}}) \stackrel{\text{def}}{=} \sum_{k=1}^T \sum_{i_1=1}^{n_1} \sum_{i_2=1}^{n_2} \theta(\tilde{\mathbf{a}}_{k,i}, \tilde{\mathbf{J}}_{k,i})$ and where the constraint now reads

$$
\mathbf{C}(\mathbf{a}_0, \mathbf{a}_1) \stackrel{\text{def}}{=} \lbrace (\mathbf{a}, \mathbf{J}) \;:\; \partial_t \mathbf{a} + \mathrm{div}(\mathbf{J}) = 0,\, (\mathbf{a}_{0,\cdot},\, \mathbf{a}_{T,\cdot}) = (\mathbf{a}_0, \mathbf{a}_1) \rbrace.
$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 7.2</span><span class="math-callout__name">(Dynamic Formulation on Graphs)</span></p>

In the case where $\mathcal{X}$ is a graph and $c(x, y) = d_\mathcal{X}(x, y)^2$ is the squared geodesic distance, it is possible to derive faithful discretization methods that use a discrete divergence associated to the graph structure in place of the uniform grid discretization (7.10). In order to ensure that the heat equation has a gradient flow structure (see §9.3 for more details about gradient flows) for the corresponding dynamic Wasserstein distance, Maas [2011] and later Mielke [2013] proposed to use a logarithmic mean $\mathcal{I}(r, s)$ (see also [Solomon et al., 2016b, Chow et al., 2012, 2017b,a]).

</div>

### Proximal Solvers

The discretized dynamic OT problem (7.11) is challenging to solve because it requires us to minimize a nonsmooth optimization problem under affine constraints. Indeed, the function $\theta$ is convex but nonsmooth for measures with vanishing mass $\mathbf{a}_{k,i}$. When interpolating between two compactly supported input measures $(\mathbf{a}_0, \mathbf{a}_1)$, one typically expects the mass of the interpolated measures $(\mathbf{a}_k)_{k=1}^T$ to vanish as well, and the difficult part of the optimization process is indeed to track this evolution of the support. In particular, it is not possible to use standard smooth optimization techniques.

There are several ways to recast (7.11) into a quadratic-cone program, either by considering the dual problem or simply by replacing the functional $\theta(\mathbf{a}_{k,i}, \mathbf{J}_{k,i})$ by a linear function under constraints. With the introduction of an extra variable $\tilde{\mathbf{z}}$, it is thus possible to solve the discretized problem using standard interior point solvers for quadratic-cone programs. These solvers have fast convergence rates and are thus capable of computing a solution with high precision. Unfortunately, each iteration is costly and requires the resolution of a linear system of dimension that scales with the number of discretization points. They are thus not applicable for large-scale multidimensional problems encountered in imaging applications.

An alternative to these high-precision solvers are low-precision first order methods, which are well suited for nonsmooth but highly structured problems such as (7.11). The DR algorithm [Lions and Mercier, 1979] is specifically tailored to solve nonsmooth structured problems of the form

$$
\min_{x \in \mathcal{H}} F(x) + G(x), \tag{7.12}
$$

where $\mathcal{H}$ is some Euclidean space, and where $F, G : \mathcal{H} \to \mathbb{R} \cup \lbrace +\infty \rbrace$ are two closed convex functions, for which one can "easily" (*e.g.* in closed form or using a rapidly converging scheme) compute the so-called proximal operator

$$
\forall\,x \in \mathcal{H}, \quad \mathrm{Prox}_{\tau F}(x) \stackrel{\text{def}}{=} \operatorname*{argmin}_{x' \in \mathcal{H}} \frac{1}{2}\|x - x'\|^2 + \tau F(x) \tag{7.13}
$$

for a parameter $\tau > 0$. The iterations of the DR algorithm define a sequence $(x^{(\ell)}, w^{(\ell)}) \in \mathcal{H}^2$ using an initialization $(x^{(0)}, w^{(0)}) \in \mathcal{H}^2$ and

$$
w^{(\ell+1)} \stackrel{\text{def}}{=} w^{(\ell)} + \alpha(\mathrm{Prox}_{\gamma F}(2x^{(\ell)} - w^{(\ell)}) - x^{(\ell)}), \tag{7.14}
$$

$$
x^{(\ell+1)} \stackrel{\text{def}}{=} \mathrm{Prox}_{\gamma G}(w^{(\ell+1)}).
$$

If $0 < \alpha < 2$ and $\gamma > 0$, one can show that $x^{(\ell)} \to z^\star$, where $z^\star$ is a solution of (7.12). This algorithm is closely related to another popular method, the alternating direction method of multipliers (ADMM).

There are many ways to recast Problem (7.11) in the form (7.12). A simple way to achieve this is by setting $x = (\mathbf{a}, \mathbf{J}, \tilde{\mathbf{a}}, \tilde{\mathbf{J}})$ and letting

$$
F(x) \stackrel{\text{def}}{=} \Theta(\tilde{\mathbf{a}}, \tilde{\mathbf{J}}) + \iota_{\mathbf{C}(\mathbf{a}_0, \mathbf{a}_1)}(\mathbf{a}, \mathbf{J}) \quad \text{and} \quad G(x) = \iota_\mathcal{D}(\mathbf{a}, \mathbf{J}, \tilde{\mathbf{a}}, \tilde{\mathbf{J}}),
$$

where $\mathcal{D} \stackrel{\text{def}}{=} \lbrace (\mathbf{a}, \mathbf{J}, \tilde{\mathbf{a}}, \tilde{\mathbf{J}}) : \tilde{\mathbf{a}} = \mathcal{I}_a(\mathbf{a}),\, \tilde{\mathbf{J}} = \mathcal{I}_J(\mathbf{J}) \rbrace$. The proximal operator $\mathrm{Prox}_{\tau\Theta}$ is computed by solving a cubic polynomial equation at each grid position. The orthogonal projection on the affine constraint $\mathbf{C}(\mathbf{a}_0, \mathbf{a}_1)$ involves the resolution of a Poisson equation, which can be achieved in $O(N \log(N))$ operations using the fast Fourier transform, where $N = Tn_1 n_2$ is the number of grid points. Lastly, the proximal operator $\mathrm{Prox}_{\tau G}$ is a linear projector, which requires the inversion of a small linear system.

### Dynamical Unbalanced OT

In order to be able to match input measures with different mass $\alpha_0(\mathcal{X}) \neq \alpha_1(\mathcal{X})$ (the so-called "unbalanced" settings, the terminology introduced by Benamou [2003]), and also to cope with local mass variation, several normalizations or relaxations have been proposed, in particular by relaxing the fixed marginal constraint; see §10.2. A general methodology consists in introducing a source term $s_t(x)$ in the continuity equation (7.4). We thus consider

$$
\bar{\mathcal{C}}(\alpha_0, \alpha_1) \stackrel{\text{def}}{=} \left\lbrace (\alpha_t, J_t, s_t) \;:\; \frac{\partial \alpha_t}{\partial t} + \mathrm{div}(J_t) = s_t,\, \alpha_{t=0} = \alpha_0,\, \alpha_{t=1} = \alpha_1 \right\rbrace.
$$

The crucial question is how to measure the cost associated to this source term and introduce it in the original dynamic formulation (7.3). In order to avoid having to "teleport" mass (mass which travels at infinite speed and suddenly grows in a region where there was no mass before), the associated cost should be infinite. It turns out that this can be achieved in a simple convex way, by also allowing $s_t$ to be an arbitrary measure (*e.g.* using a 1-homogeneous cost) by penalizing $s_t$ in the same way as the momentum $J_t$,

$$
\mathrm{WFR}^2(\alpha_0, \alpha_1) = \min_{(\alpha_t, J_t, s_t)_t \in \bar{\mathcal{C}}(\alpha_0, \alpha_1)} \Theta(\alpha, J, s), \tag{7.15}
$$

where $\Theta(\alpha, J, s) \stackrel{\text{def}}{=} \int_0^1 \int_{\mathbb{R}^d} \left(\theta(\alpha_t(x), J_t(x)) + \tau \theta(\alpha_t(x), s_t(x))\right) \mathrm{d}x \mathrm{d}t$, where $\theta$ is the convex 1-homogeneous function introduced in (7.5), and $\tau$ is a weight controlling the trade-off between mass transportation and mass creation/destruction. This "dynamic" formulation has a "static" counterpart; see Remark 10.5. The convex optimization problem (7.15) can be solved using methods similar to those detailed in §7.3.

As $\tau \to 0$, and if $\alpha_0(\mathcal{X}) = \alpha_1(\mathcal{X})$, then one retrieves the classical OT problem, $\mathrm{WFR}(\alpha_0, \alpha_1) \to \mathcal{W}(\alpha_0, \alpha_1)$. In contrast, as $\tau \to +\infty$, this distance approaches the Hellinger metric over densities

$$
\frac{1}{\tau}\mathrm{WFR}(\alpha_0, \alpha_1)^2 \xrightarrow{\tau \to +\infty} \int_\mathcal{X} \left|\sqrt{\rho_{\alpha_0}(x)} - \sqrt{\rho_{\alpha_1}(x)}\right|^2 \mathrm{d}x = \int_\mathcal{X} \left|1 - \sqrt{\frac{\mathrm{d}\alpha_1}{\mathrm{d}\alpha_0}}(x)\right|^2 \mathrm{d}\alpha_0(x).
$$

### More General Mobility Functionals

It is possible to generalize the dynamic formulation (7.3) by considering other "mobility functions" $\theta$ in place of the one defined in (7.5). A possible choice for this mobility functional is proposed in Dolbeault et al. [2009],

$$
\forall\,(a, b) \in \mathbb{R}_+ \times \mathbb{R}^d, \quad \theta(a, b) = a^{s-p}\|b\|^p, \tag{7.16}
$$

where the parameter should satisfy $p \ge 1$ and $s \in [1, p]$ in order for $\theta$ to be convex. Note that this definition should be handled with care in the case $1 < s \le p$ because $\theta$ does not have a linear growth at infinity, so that solutions to (7.3) must be constrained to have a density with respect to the Lebesgue measure.

The case $s = 1$ corresponds to the classical OT problem and the optimal value of (7.3) defines $\mathcal{W}_p(\alpha, \beta)$. In this case, $\theta$ is 1-homogeneous, so that solutions to (7.3) can be arbitrary measures. The case $(s = 1, p = 2)$ is the initial setup considered in (7.3) to define $\mathcal{W}_2$.

The limiting case $s = p$ is also interesting, because it corresponds to a dual Sobolev norm $W^{-1,p}$ and the value of (7.3) is then equal to

$$
\|\alpha - \beta\|_{W^{-1,p}(\mathbb{R}^d)}^p = \min_f \left\lbrace \int_{\mathbb{R}^d} f\,\mathrm{d}(\alpha - \beta) \;:\; \int_{\mathbb{R}^d} \|\nabla f(x)\|^q \,\mathrm{d}x \le 1 \right\rbrace
$$

for $1/q + 1/p = 1$. In the limit $(p = s, q) \to (1, \infty)$, one recovers the $\mathcal{W}_1$ norm. The case $s = p = 2$ corresponds to the Sobolev $H^{-1}(\mathbb{R}^d)$ Hilbert norm defined in (8.15).

### Dynamic Formulation over the Paths Space

There is a natural dynamical formulation of both classical and entropic regularized (see §4) formulations of OT, which is based on studying abstract optimization problems on the space $\bar{\mathcal{X}}$ of all possible paths $\gamma : [0, 1] \to \mathcal{X}$ (*i.e.* curves) on the space $\mathcal{X}$. For simplicity, we assume $\mathcal{X} = \mathbb{R}^d$, but this extends to more general spaces such as geodesic spaces and graphs. Informally, the dynamic of "particles" between two input measures $\alpha_0, \alpha_1$ at times $t = 0, 1$ is described by a probability distribution $\tilde{\pi} \in \mathcal{M}_+^1(\bar{\mathcal{X}})$. Such a distribution should satisfy that the distributions of starting and end points must match $(\alpha_0, \alpha_1)$, which is formally written using push-forward as

$$
\bar{\mathcal{U}}(\alpha_0, \alpha_1) \stackrel{\text{def}}{=} \left\lbrace \tilde{\pi} \in \mathcal{M}_+^1(\bar{\mathcal{X}}) \;:\; \bar{P}_{0\sharp}\tilde{\pi} = \alpha_0,\, \bar{P}_{1\sharp}\tilde{\pi} = \alpha_1 \right\rbrace,
$$

where, for any path $\gamma \in \bar{\mathcal{X}}$, $P_0(\gamma) = \gamma(0)$, $P_1(\gamma) = \gamma(1)$.

**OT over the space of paths.** The dynamical version of classical OT (2.15), formulated over the space of paths, then reads

$$
\mathcal{W}_2(\alpha_0, \alpha_1)^2 = \min_{\tilde{\pi} \in \bar{\mathcal{U}}(\alpha_0, \alpha_1)} \int_{\bar{\mathcal{X}}} \mathcal{L}(\gamma)^2 \,\mathrm{d}\tilde{\pi}(\gamma), \tag{7.17}
$$

where $\mathcal{L}(\gamma) = \int_0^1 |\gamma'(s)|^2 \mathrm{d}s$ is the kinetic energy of a path $s \in [0, 1] \mapsto \gamma(s) \in \mathcal{X}$. The connection between optimal couplings $\pi^\star$ and $\tilde{\pi}^\star$ solving respectively (7.17) and (2.15) is that $\tilde{\pi}^\star$ only gives mass to geodesics joining pairs of points in proportion prescribed by $\pi^\star$. In the particular case of discrete measures, this means that

$$
\pi^\star = \sum_{i,j} \mathbf{P}_{i,j} \delta_{(x_i, y_j)} \quad \text{and} \quad \tilde{\pi}^\star = \sum_{i,j} \mathbf{P}_{i,j} \delta_{\gamma_{x_i, y_j}},
$$

where $\gamma_{x_i, y_j}$ is the geodesic between $x_i$ and $y_j$. Furthermore, the measures defined by the distribution of the curve points $\gamma(t)$ at time $t$, where $\gamma$ is drawn following $\tilde{\pi}^\star$, *i.e.*

$$
t \in [0, 1] \mapsto \alpha_t \stackrel{\text{def}}{=} P_{t\sharp}\tilde{\pi}^\star \quad \text{where} \quad P_t(\gamma) = \gamma(t) \in \mathcal{X}, \tag{7.18}
$$

is a solution to the dynamical formulation (7.3), *i.e.* it is the displacement interpolation. In the discrete case, one recovers (7.9).

**Entropic OT over the space of paths.** We now turn to the re-interpretation of entropic OT, defined in Chapter 4, using the space of paths. Similarly to (4.11), this is defined using a Kullback–Leibler projection, but this time of a reference measure over the space of paths $\bar{\mathcal{K}}$ which is the distribution of a reversible Brownian motion (Wiener process), which has a uniform distribution at the initial and final times

$$
\min_{\tilde{\pi} \in \bar{\mathcal{U}}(\alpha_0, \alpha_1)} \mathrm{KL}(\tilde{\pi}|\bar{\mathcal{K}}). \tag{7.19}
$$

One can show that the (unique) solution $\tilde{\pi}^\star_\varepsilon$ to (7.19) converges to a solution of (7.17) as $\varepsilon \to 0$. Furthermore, this solution is linked to the solution of the static entropic OT problem (4.9) using Brownian bridge $\bar{\gamma}^\varepsilon_{x,y} \in \bar{\mathcal{X}}$ (which are similar to fuzzy geodesic and converge to $\delta_{\gamma_{x,y}}$ as $\varepsilon \to 0$). In the discrete setting, this means that

$$
\pi^\star_\varepsilon = \sum_{i,j} \mathbf{P}^\star_{\varepsilon,i,j} \delta_{(x_i, y_j)} \quad \text{and} \quad \tilde{\pi}^\star_\varepsilon = \sum_{i,j} \mathbf{P}^\star_{\varepsilon,i,j} \bar{\gamma}^\varepsilon_{x_i, y_j}, \tag{7.20}
$$

where $\mathbf{P}^\star_{\varepsilon,i,j}$ can be computed using Sinkhorn's algorithm. Similarly to (7.18), one then can define an entropic interpolation as

$$
\alpha_{\varepsilon,t} \stackrel{\text{def}}{=} P_{t\sharp}\tilde{\pi}^\star_\varepsilon.
$$

Since the law $P_{t\sharp}\bar{\gamma}^\varepsilon_{x,y}$ of the position at time $t$ along a Brownian bridge is a Gaussian $\mathcal{G}_{t(1-t)\varepsilon^2}(\cdot - \gamma_{x,y}(t))$ of variance $t(1-t)\varepsilon^2$ centered at $\gamma_{x,y}(t)$, one can deduce that $\alpha_{\varepsilon,t}$ is a Gaussian blurring of a set of traveling Diracs

$$
\alpha_{\varepsilon,t} = \sum_{i,j} \mathbf{P}^\star_{\varepsilon,i,j} \mathcal{G}_{t(1-t)\varepsilon^2}(\cdot - \gamma_{x_i, y_j}(t)).
$$

Another way to describe this entropic interpolation $(\alpha_t)_t$ is using a regularization of the Benamou–Brenier dynamic formulation (7.2), namely

$$
\min_{(\alpha_t, v_t)_t \,\text{sat.}\,(7.1)} \int_0^1 \int_{\mathbb{R}^d} \left(\|v_t(x)\|^2 + \frac{\varepsilon}{4}\|\nabla \log(\alpha_t)(x)\|^2\right) \mathrm{d}\alpha_t(x) \mathrm{d}t. \tag{7.21}
$$

## Chapter 8: Statistical Divergences

We study in this chapter the statistical properties of the Wasserstein distance. More specifically, we compare it to other major distances and divergences routinely used in data sciences. We quantify how one can approximate the distance between two probability distributions when having only access to samples from said distributions. To introduce these subjects, §8.1 and §8.2 review respectively divergences and integral probability metrics between probability distributions. A divergence $D$ typically satisfies $D(\alpha, \beta) \ge 0$ and $D(\alpha, \beta) = 0$ if and only if $\alpha = \beta$, but it does not need to be symmetric or satisfy the triangular inequality. An integral probability metric for measures is a dual norm defined using a prescribed family of test functions. These quantities are sound alternatives to Wasserstein distances and are routinely used as loss functions to tackle inference problems, as will be covered in §9.

### $\varphi$-Divergences

Before detailing in the following section "weak" norms, whose construction shares similarities with $\mathcal{W}_1$, let us detail a generic construction of so-called divergences between measures, which can then be used as loss functions when estimating probability distributions. Such divergences compare two input measures by comparing their mass *pointwise*, without introducing any notion of mass transportation. Divergences are functionals which, by looking at the pointwise ratio between two measures, give a sense of how close they are. They have nice analytical and computational properties and build upon *entropy functions*.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 8.1</span><span class="math-callout__name">(Entropy Function)</span></p>

A function $\varphi : \mathbb{R} \to \mathbb{R} \cup \lbrace \infty \rbrace$ is an entropy function if it is lower semicontinuous, convex, $\mathrm{dom}\,\varphi \subset [0, \infty[$, and satisfies the following feasibility condition: $\mathrm{dom}\,\varphi \cap\, ]0, \infty[\, \neq \emptyset$. The speed of growth of $\varphi$ at $\infty$ is described by

$$
\varphi'_\infty = \lim_{x \to +\infty} \varphi(x)/x \in \mathbb{R} \cup \lbrace \infty \rbrace.
$$

If $\varphi'_\infty = \infty$, then $\varphi$ grows faster than any linear function and $\varphi$ is said *superlinear*.

</div>

Any entropy function $\varphi$ induces a $\varphi$-divergence (also known as Csiszár divergence or $f$-divergence) as follows.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 8.2</span><span class="math-callout__name">($\varphi$-Divergences)</span></p>

Let $\varphi$ be an entropy function. For $\alpha, \beta \in \mathcal{M}(\mathcal{X})$, let $\frac{\mathrm{d}\alpha}{\mathrm{d}\beta}\beta + \alpha^\perp$ be the Lebesgue decomposition of $\alpha$ with respect to $\beta$. The divergence $\mathcal{D}_\varphi$ is defined by

$$
\mathcal{D}_\varphi(\alpha|\beta) \stackrel{\text{def}}{=} \int_\mathcal{X} \varphi\left(\frac{\mathrm{d}\alpha}{\mathrm{d}\beta}\right)\mathrm{d}\beta + \varphi'_\infty \alpha^\perp(\mathcal{X}) \tag{8.1}
$$

if $\alpha, \beta$ are nonnegative and $\infty$ otherwise.

</div>

The additional term $\varphi'_\infty \alpha^\perp(\mathcal{X})$ in (8.1) is important to ensure that $\mathcal{D}_\varphi$ defines a continuous functional (for the weak topology of measures) even if $\varphi$ has a linear growth at infinity. If $\varphi$ has a superlinear growth, *e.g.* the usual entropy (8.4), then $\varphi'_\infty = +\infty$ so that $\mathcal{D}_\varphi(\alpha|\beta) = +\infty$ if $\alpha$ does not have a density with respect to $\beta$.

In the discrete setting, assuming $\alpha = \sum_i \mathbf{a}\_i \delta\_{x_i}$ and $\beta = \sum_i \mathbf{b}\_i \delta\_{x_i}$ are supported on the same set of $n$ points $(x_i)\_{i=1}^n \subset \mathcal{X}$, (8.1) defines a divergence on $\Sigma_n$

$$
\mathbf{D}_\varphi(\mathbf{a}|\mathbf{b}) = \sum_{i \in \mathrm{Supp}(\mathbf{b})} \varphi\left(\frac{\mathbf{a}_i}{\mathbf{b}_i}\right)\mathbf{b}_i + \varphi'_\infty \sum_{i \notin \mathrm{Supp}(\mathbf{b})} \mathbf{a}_i. \tag{8.3}
$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8.1</span><span class="math-callout__name">(Properties of $\varphi$-Divergences)</span></p>

If $\varphi$ is an entropy function, then $\mathcal{D}_\varphi$ is jointly 1-homogeneous, convex and weakly* lower semicontinuous in $(\alpha, \beta)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 8.1</span><span class="math-callout__name">(Dual Expression)</span></p>

A $\varphi$-divergence can be expressed using the Legendre transform

$$
\varphi^*(s) \stackrel{\text{def}}{=} \sup_{t \in \mathbb{R}}\, st - \varphi(t)
$$

of $\varphi$ (see also (4.54)) as

$$
\mathcal{D}_\varphi(\alpha|\beta) = \sup_{f : \mathcal{X} \to \mathbb{R}} \int_\mathcal{X} f(x)\mathrm{d}\alpha(x) - \int_\mathcal{X} \varphi^*(f(x))\mathrm{d}\beta(x).
$$

</div>

We now review a few popular instances of this framework.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.1</span><span class="math-callout__name">(Kullback–Leibler Divergence)</span></p>

The Kullback–Leibler divergence $\mathrm{KL} \stackrel{\text{def}}{=} \mathcal{D}_{\varphi_\mathrm{KL}}$, also known as the relative entropy, is the divergence associated to the Shannon–Boltzmann entropy function $\varphi_\mathrm{KL}$, given by

$$
\varphi_\mathrm{KL}(s) = \begin{cases} s\log(s) - s + 1 & \text{for } s > 0, \\ 1 & \text{for } s = 0, \\ +\infty & \text{otherwise.} \end{cases} \tag{8.4}
$$

For univariate Gaussians $\alpha = \mathcal{N}(m_\alpha, \sigma_\alpha^2)$ and $\beta = \mathcal{N}(m_\beta, \sigma_\beta^2)$, one has

$$
\mathrm{KL}(\alpha|\beta) = \frac{1}{2}\left(\frac{\sigma_\alpha^2}{\sigma_\beta^2} + \log\left(\frac{\sigma_\beta^2}{\sigma_\alpha^2}\right) + \frac{|m_\alpha - m_\beta|}{\sigma_\beta^2} - 1\right). \tag{8.6}
$$

The divergence between $\alpha$ and $\beta$ diverges to infinity as $\sigma_\beta$ diminishes to 0 and $\beta$ becomes a Dirac mass. Singular Gaussians are infinitely far from all other Gaussians in the KL geometry.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 8.1b</span><span class="math-callout__name">(Bregman Divergence)</span></p>

The discrete KL divergence, $\mathbf{KL} \stackrel{\text{def}}{=} \mathbf{D}_{\varphi_\mathrm{KL}}$, has the unique property of being both a $\varphi$-divergence and a Bregman divergence. For discrete vectors in $\mathbb{R}^n$, a Bregman divergence associated to a smooth strictly convex function $\psi : \mathbb{R}^n \to \mathbb{R}$ is defined as

$$
\mathbf{B}_\psi(\mathbf{a}|\mathbf{b}) \stackrel{\text{def}}{=} \psi(\mathbf{a}) - \psi(\mathbf{b}) - \langle \nabla\psi(\mathbf{b}),\, \mathbf{a} - \mathbf{b} \rangle, \tag{8.5}
$$

where $\langle \cdot, \cdot \rangle$ is the canonical inner product on $\mathbb{R}^n$. Note that $\mathbf{B}_\psi(\mathbf{a}|\mathbf{b})$ is a convex function of $\mathbf{a}$ and a linear function of $\psi$. Similarly to $\varphi$-divergence, a Bregman divergence satisfies $\mathbf{B}_\psi(\mathbf{a}|\mathbf{b}) \ge 0$ and $\mathbf{B}_\psi(\mathbf{a}|\mathbf{b}) = 0$ if and only if $\mathbf{a} = \mathbf{b}$. The KL divergence is the Bregman divergence for minus the entropy $\psi = -\mathbf{H}$ defined in (4.1), *i.e.* $\mathbf{KL} = \mathbf{B}_{-\mathbf{H}}$. A Bregman divergence is locally a squared Euclidean distance since

$$
\mathbf{B}_\psi(\mathbf{a} + \varepsilon | \mathbf{a} + \eta) = \langle \partial^2\psi(\mathbf{a})(\varepsilon - \eta),\, \varepsilon - \eta \rangle + o(\|\varepsilon - \eta\|^2).
$$

These properties make Bregman divergences suitable to replace Euclidean distances in first order optimization methods. The best known example is mirror gradient descent, which is an explicit descent step of the form (9.32). Bregman divergences are also important in convex optimization and can be used, for instance, to derive Sinkhorn iterations and study its convergence in finite dimension; see Remark 4.8.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 8.2</span><span class="math-callout__name">(Hyperbolic Geometry of KL)</span></p>

It is interesting to contrast the geometry of the Kullback–Leibler divergence to that defined by quadratic optimal transport when comparing Gaussians. The infinitesimal geometry of KL, obtained by performing a Taylor expansion at order 2, is

$$
\mathrm{KL}(\mathcal{N}(m + \delta_m, (\sigma + \delta_\sigma)^2)|\mathcal{N}(m, \sigma^2)) = \frac{1}{\sigma^2}\left(\frac{1}{2}\delta_m^2 + \delta_\sigma^2\right) + o(\delta_m^2, \delta_\sigma^2).
$$

This local Riemannian metric, the so-called Fisher metric, expressed over $(m/\sqrt{2}, \sigma) \in \mathbb{R} \times \mathbb{R}_{+,*}$, matches exactly that of the hyperbolic Poincaré half plane. Geodesics over this space are half circles centered along the $\sigma = 0$ line and have an exponential speed; they only reach the limit $\sigma = 0$ after an infinite time.

The KL hyperbolic geometry over the space of Gaussian parameters $(m, \sigma)$ should be contrasted with the Euclidean geometry associated to OT, since in the univariate case

$$
\mathcal{W}_2^2(\alpha, \beta) = |m_\alpha - m_\beta|^2 + |\sigma_\alpha - \sigma_\beta|^2. \tag{8.7}
$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.2</span><span class="math-callout__name">(Total Variation)</span></p>

The total variation distance $\mathrm{TV} \stackrel{\text{def}}{=} \mathcal{D}_{\varphi_\mathrm{TV}}$ is the divergence associated to

$$
\varphi_\mathrm{TV}(s) = \begin{cases} |s - 1| & \text{for } s \ge 0, \\ +\infty & \text{otherwise.} \end{cases} \tag{8.8}
$$

It actually defines a norm on the full space of measure $\mathcal{M}(\mathcal{X})$ where

$$
\mathrm{TV}(\alpha|\beta) = \|\alpha - \beta\|_\mathrm{TV}, \quad \text{where} \quad \|\alpha\|_\mathrm{TV} = |\alpha|(\mathcal{X}) = \int_\mathcal{X} \mathrm{d}|\alpha|(x). \tag{8.9}
$$

If $\alpha$ has a density $\rho_\alpha$ on $\mathcal{X} = \mathbb{R}^d$, then the TV norm is the $L^1$ norm on functions, $\|\alpha\|\_\mathrm{TV} = \int\_\mathcal{X} |\rho\_\alpha(x)|\mathrm{d}x = \|\rho\_\alpha\|\_{L^1}$. If $\alpha$ is discrete as in (8.2), then the TV norm is the $\ell^1$ norm of vectors in $\mathbb{R}^n$, $\|\alpha\|\_\mathrm{TV} = \sum\_i |\mathbf{a}\_i| = \|\mathbf{a}\|\_{\ell^1}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 8.2b</span><span class="math-callout__name">(Strong vs. Weak Topology)</span></p>

The total variation norm (8.9) defines the so-called "strong" topology on the space of measure. On a compact domain $\mathcal{X}$ of radius $R$, one has

$$
\mathcal{W}_1(\alpha, \beta) \le R\,\|\alpha - \beta\|_\mathrm{TV}
$$

so that this strong notion of convergence implies the weak convergence metrized by Wasserstein distances. The converse is, however, not true, since $\delta_x$ does not converge strongly to $\delta_y$ if $x \to y$ (note that $\|\delta_x - \delta_y\|\_\mathrm{TV} = 2$ if $x \neq y$). A chief advantage is that $\mathcal{M}\_+^1(\mathcal{X})$ (once again on a compact ground space $\mathcal{X}$) is compact for the weak topology, so that from any sequence of probability measures $(\alpha_k)\_k$, one can always extract a converging subsequence.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.3</span><span class="math-callout__name">(Hellinger)</span></p>

The Hellinger distance $\mathfrak{h} \stackrel{\text{def}}{=} \mathcal{D}_{\varphi_H}^{1/2}$ is the square root of the divergence associated to

$$
\varphi_H(s) = \begin{cases} |\sqrt{s} - 1|^2 & \text{for } s \ge 0, \\ +\infty & \text{otherwise.} \end{cases}
$$

As its name suggests, $\mathfrak{h}$ is a distance on $\mathcal{M}\_+(\mathcal{X})$, which metrizes the strong topology as $\|\cdot\|\_\mathrm{TV}$. If $(\alpha, \beta)$ have densities $(\rho_\alpha, \rho_\beta)$ on $\mathcal{X} = \mathbb{R}^d$, then $\mathfrak{h}(\alpha, \beta) = \|\sqrt{\rho_\alpha} - \sqrt{\rho_\beta}\|\_{L^2}$. If $(\alpha, \beta)$ are discrete as in (8.2), then $\mathfrak{h}(\alpha, \beta) = \|\sqrt{\mathbf{a}} - \sqrt{\mathbf{b}}\|$. Considering $\varphi\_{L^p}(s) = |s^{1/p} - 1|^p$ generalizes the Hellinger ($p = 2$) and total variation ($p = 1$) distances and $\mathcal{D}\_{\varphi\_{L^p}}^{1/p}$ is a distance which metrizes the strong convergence for $0 < p < +\infty$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.4</span><span class="math-callout__name">(Jensen–Shannon Distance)</span></p>

The KL divergence is not symmetric and, while being a Bregman divergence (which are locally quadratic norms), it is not the square of a distance. On the other hand, the Jensen–Shannon distance $\mathrm{JS}(\alpha, \beta)$, defined as

$$
\mathrm{JS}(\alpha, \beta)^2 \stackrel{\text{def}}{=} \frac{1}{2}\left(\mathrm{KL}(\alpha|\xi) + \mathrm{KL}(\beta|\xi)\right) \quad \text{where} \quad \xi = \frac{\alpha + \beta}{2},
$$

is a distance. $\mathrm{JS}^2$ can be shown to be a $\varphi$-divergence for $\varphi(s) = t\log(t) - (t+1)\log(t+1)$. In sharp contrast with KL, $\mathrm{JS}(\alpha, \beta)$ is always bounded; more precisely, it satisfies $0 \le \mathrm{JS}(\alpha, \beta)^2 \le \ln(2)$. Similarly to the TV norm and the Hellinger distance, it metrizes the strong convergence.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.5</span><span class="math-callout__name">($\chi^2$-Divergence)</span></p>

The $\chi^2$-divergence $\chi^2 \stackrel{\text{def}}{=} \mathcal{D}_{\varphi_{\chi^2}}$ is the divergence associated to

$$
\varphi_{\chi^2}(s) = \begin{cases} |s - 1|^2 & \text{for } s \ge 0, \\ +\infty & \text{otherwise.} \end{cases}
$$

If $(\alpha, \beta)$ are discrete as in (8.2) and have the same support, then

$$
\chi^2(\alpha|\beta) = \sum_i \frac{(\mathbf{a}_i - \mathbf{b}_i)^2}{\mathbf{b}_i}.
$$

</div>

### Integral Probability Metrics

Formulation (6.3) is a special case of a dual norm. A dual norm is a convenient way to design "weak" norms that can deal with arbitrary measures. For a symmetric convex set $B$ of measurable functions, one defines

$$
\|\alpha\|_B \stackrel{\text{def}}{=} \max_f \left\lbrace \int_\mathcal{X} f(x)\mathrm{d}\alpha(x) \;:\; f \in B \right\rbrace. \tag{8.10}
$$

These dual norms are often called "integral probability metrics."

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.6</span><span class="math-callout__name">(Total Variation as Dual Norm)</span></p>

The total variation norm (Example 8.2) is a dual norm associated to the whole space of continuous functions

$$
B = \lbrace f \in \mathcal{C}(\mathcal{X}) \;:\; \|f\|_\infty \le 1 \rbrace.
$$

The total variation distance is the only nontrivial divergence that is also a dual norm.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 8.3</span><span class="math-callout__name">(Metrizing the Weak Convergence)</span></p>

By using smaller "balls" $B$, which typically only contain continuous (and sometimes regular) functions, one defines weaker dual norms. In order for $\|\cdot\|\_B$ to metrize the weak convergence (see Definition 2.2), it is sufficient for the space spanned by $B$ to be dense in the set of continuous functions for the sup-norm $\|\cdot\|\_\infty$ (i.e. for the topology of uniform convergence).

</div>

#### $\mathcal{W}_1$ and Flat Norm

If the set $B$ is bounded, then $\|\cdot\|\_B$ is a norm on the whole space $\mathcal{M}(\mathcal{X})$ of measures. This is not the case of $\mathcal{W}\_1$, which is only defined for $\alpha$ such that $\int\_\mathcal{X} \mathrm{d}\alpha = 0$ (otherwise $\|\alpha\|\_B = +\infty$). This can be alleviated by imposing a bound on the value of the potential $f$, in order to define for instance the flat norm.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.7</span><span class="math-callout__name">($\mathcal{W}_1$ Norm)</span></p>

$\mathcal{W}_1$ as defined in (6.3), is a special case of dual norm (8.10), using

$$
B = \lbrace f \;:\; \mathrm{Lip}(f) \le 1 \rbrace
$$

the set of 1-Lipschitz functions.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.8</span><span class="math-callout__name">(Flat Norm and Dudley Metric)</span></p>

The flat norm is defined using

$$
B = \lbrace f \;:\; \|\nabla f\|_\infty \le 1 \quad \text{and} \quad \|f\|_\infty \le 1 \rbrace. \tag{8.11}
$$

It metrizes the weak convergence on the whole space $\mathcal{M}(\mathcal{X})$. Formula (6.2) is extended to compute the flat norm by adding the constraint $|\mathbf{f}\_k| \le 1$. The flat norm is sometimes called the "Kantorovich–Rubinstein" norm. The flat norm is similar to the Dudley metric, which uses

$$
B = \lbrace f \;:\; \|\nabla f\|_\infty + \|f\|_\infty \le 1 \rbrace.
$$

</div>

#### Dual RKHS Norms and Maximum Mean Discrepancies

It is also possible to define "Euclidean" norms (built using quadratic functionals) on measures using the machinery of kernel methods and more specifically reproducing kernel Hilbert spaces (RKHS).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 8.3</span><span class="math-callout__name">(Positive/Negative Definite Kernels)</span></p>

A symmetric function $k$ (resp., $\varphi$) defined on a set $\mathcal{X} \times \mathcal{X}$ is said to be positive (resp., negative) definite if for any $n \ge 0$, family $x_1, \ldots, x_n \in \mathcal{Z}$, and vector $r \in \mathbb{R}^n$ the following inequality holds:

$$
\sum_{i,j=1}^n r_i r_j k(x_i, x_j) \ge 0, \quad \left(\text{resp.} \quad \sum_{i,j=1}^n r_i r_j \varphi(x_i, x_j) \le 0\right). \tag{8.12}
$$

The kernel is said to be conditionally positive if positivity only holds in (8.12) for zero mean vectors $r$ (i.e. such that $\langle r, \mathbb{1}_n \rangle = 0$).

</div>

If $k$ is conditionally positive, one defines the following norm:

$$
\|\alpha\|_k^2 \stackrel{\text{def}}{=} \int_{\mathcal{X} \times \mathcal{X}} k(x, y)\mathrm{d}\alpha(x)\mathrm{d}\alpha(y). \tag{8.13}
$$

These norms are often referred to as "maximum mean discrepancy" (MMD) and have also been called "kernel norms" in shape analysis. This expression (8.13) can be rephrased, introducing two independent random vectors $(X, X')$ on $\mathcal{X}$ distributed with law $\alpha$, as

$$
\|\alpha\|_k^2 = \mathbb{E}_{X, X'}(k(X, X')).
$$

One can show that $\|\cdot\|\_k^2$ is the dual norm in the sense of (8.10) associated to the unit ball $B$ of the RKHS associated to $k$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 8.4</span><span class="math-callout__name">(Universal Kernels)</span></p>

According to Remark 8.3, the MMD norm $\|\cdot\|\_k$ metrizes the weak convergence if the span of the dual ball $B$ is dense in the space of continuous functions $\mathcal{C}(\mathcal{X})$. This means that finite sums of the form $\sum\_{i=1}^n a_i k(x_i, \cdot)$ (for arbitrary choice of $n$ and points $(x_i)\_i$) are dense in $\mathcal{C}(\mathcal{X})$ for the uniform norm $\|\cdot\|\_\infty$. For translation-invariant kernels over $\mathcal{X} = \mathbb{R}^d$, $k(x, y) = k_0(x - y)$, this is equivalent to having a nonvanishing Fourier transform, $\hat{k}\_0(\omega) > 0$.

</div>

In particular, when $\alpha = \sum\_{i=1}^n \mathbf{a}\_i \delta\_{x\_i}$ and $\beta = \sum\_{i=1}^n \mathbf{b}\_i \delta\_{x\_i}$ are supported on the same set of points, $\|\alpha - \beta\|\_k^2 = \langle \mathbf{k}(\mathbf{a} - \mathbf{b}),\, \mathbf{a} - \mathbf{b} \rangle$, so that $\|\cdot\|\_k$ is a Euclidean norm (proper if $\mathbf{k}$ is positive definite, degenerate otherwise if $\mathbf{k}$ is semidefinite) on the simplex $\Sigma_n$. To compute the discrepancy between two discrete measures of the form (2.3), one can use

$$
\|\alpha - \beta\|_k^2 = \sum_{i,i'} \mathbf{a}_i \mathbf{a}_{i'} k(x_i, x_{i'}) + \sum_{j,j'} \mathbf{b}_j \mathbf{b}_{j'} k(y_j, y_{j'}) - 2\sum_{i,j} \mathbf{a}_i \mathbf{b}_j k(x_i, y_j). \tag{8.14}
$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.9</span><span class="math-callout__name">(Gaussian RKHS)</span></p>

One of the most popular kernels is the Gaussian one $k(x, y) = e^{-\frac{\|x-y\|^2}{2\sigma^2}}$, which is a positive universal kernel on $\mathcal{X} = \mathbb{R}^d$. An attractive feature of the Gaussian kernel is that it is separable as a product of 1-D kernels, which facilitates computations when working on regular grids. However, an important issue that arises when using the Gaussian kernel is that one needs to select the bandwidth parameter $\sigma$. This bandwidth should match the "typical scale" between observations in the measures to be compared. If the measures have multiscale features, a Gaussian kernel is thus not well adapted, and one should consider a "scale-free" kernel as we detail next.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.10</span><span class="math-callout__name">($H^{-1}(\mathbb{R}^d)$ Norm)</span></p>

Another important dual norm is $H^{-1}(\mathbb{R}^d)$, the dual (over distributions) of the Sobolev space $H^1(\mathbb{R}^d)$ of functions having derivatives in $L^2(\mathbb{R}^d)$. It is defined using the primal RKHS norm $\|\nabla f\|\_{L^2(\mathbb{R}^d)}^2$. It is not defined for singular measures (*e.g.* Diracs) unless $d = 1$ because functions in the Sobolev space $H^1(\mathbb{R}^d)$ are in general not continuous. This $H^{-1}$ norm (defined on the space of zero mean measures with densities) can also be formulated in divergence form,

$$
\|\alpha - \beta\|_{H^{-1}(\mathbb{R}^d)}^2 = \min_s \left\lbrace \int_{\mathbb{R}^d} \|s(x)\|_2^2 \,\mathrm{d}x \;:\; \mathrm{div}(s) = \alpha - \beta \right\rbrace, \tag{8.15}
$$

which should be contrasted with (6.4), where an $L^1$ norm of the vector field $s$ was used in place of the $L^2$ norm used here. The "weighted" version of this Sobolev dual norm,

$$
\|\rho\|_{H^{-1}(\alpha)}^2 = \min_{\mathrm{div}(s) = \rho} \int_{\mathbb{R}^d} \|s(x)\|_2^2 \,\mathrm{d}\alpha(x),
$$

can be interpreted as the natural "linearization" of the Wasserstein $\mathcal{W}_2$ norm, in the sense that the Benamou–Brenier dynamic formulation can be interpreted infinitesimally as

$$
\mathcal{W}_2(\alpha, \alpha + \varepsilon\rho) = \varepsilon\,\|\rho\|_{H^{-1}(\alpha)} + o(\varepsilon). \tag{8.16}
$$

The functionals $\mathcal{W}\_2(\alpha, \beta)$ and $\|\alpha - \beta\|\_{H^{-1}(\alpha)}$ can be shown to be equivalent. The issue is that $\|\alpha - \beta\|\_{H^{-1}(\alpha)}$ is not a norm (because of the weighting by $\alpha$), and one cannot in general replace it by $\|\alpha - \beta\|\_{H^{-1}(\mathbb{R}^d)}$ unless $(\alpha, \beta)$ have densities. In this case, if $\alpha$ and $\beta$ have densities on the same support bounded from below by $a > 0$ and from above by $b < +\infty$, then

$$
b^{-1/2}\,\|\alpha - \beta\|_{H^{-1}(\mathbb{R}^d)} \le \mathcal{W}_2(\alpha, \beta) \le a^{-1/2}\,\|\alpha - \beta\|_{H^{-1}(\mathbb{R}^d)}. \tag{8.17}
$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.11</span><span class="math-callout__name">(Negative Sobolev Spaces)</span></p>

One can generalize this construction by considering the Sobolev space $H^{-r}(\mathbb{R}^d)$ of arbitrary negative index, which is the dual of the functional Sobolev space $H^r(\mathbb{R}^d)$ of functions having $r$ derivatives (in the sense of distributions) in $L^2(\mathbb{R}^d)$. In order to metrize the weak convergence, one needs functions in $H^r(\mathbb{R}^d)$ to be continuous, which is the case when $r > d/2$. For arbitrary $\alpha$ (not necessarily integers), these spaces are defined using the Fourier transform, and for a measure $\alpha$ with Fourier transform $\hat{\alpha}(\omega)$ (written here as a density with respect to the Lebesgue measure $\mathrm{d}\omega$)

$$
\|\alpha\|_{H^{-r}(\mathbb{R}^d)}^2 \stackrel{\text{def}}{=} \int_{\mathbb{R}^d} \|\omega\|^{-2r} |\hat{\alpha}(\omega)|^2 \mathrm{d}\omega.
$$

This corresponds to a dual RKHS norm with a convolutive kernel $k(x, y) = k_0(x - y)$ with $\hat{k}\_0(\omega) = \pm \|\omega\|^{-2r}$. Taking the inverse Fourier transform, one sees that (up to constant) one has

$$
\forall\,x \in \mathbb{R}^d, \quad k_0(x) = \begin{cases} \frac{1}{\|x\|^{d-2r}} & \text{if } r < d/2, \\ -\|x\|^{2r-d} & \text{if } r > d/2. \end{cases} \tag{8.18}
$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 8.12</span><span class="math-callout__name">(Energy Distance)</span></p>

The energy distance (or Cramér distance when $d = 1$) associated to a distance $d$ is defined as

$$
\|\alpha - \beta\|_{\mathrm{ED}(\mathcal{X}, d^p)} \stackrel{\text{def}}{=} \|\alpha - \beta\|_{k_\mathrm{ED}} \quad \text{where} \quad k_\mathrm{ED}(x, y) = -d(x, y)^p \tag{8.19}
$$

for $0 < p < 2$. It is a valid MMD norm over measures if $d$ is negative definite (see Definition 8.3), a typical example being the Euclidean distance $d(x, y) = \|x - y\|$. For $\mathcal{X} = \mathbb{R}^d$, $d(x, y) = \|x - y\|$, using (8.18), one sees that the energy distance is a Sobolev norm

$$
\|\cdot\|_{\mathrm{ED}(\mathbb{R}^d, \|\cdot\|^p)} = \|\cdot\|_{H^{-\frac{d+p}{2}}(\mathbb{R}^d)}.
$$

A chief advantage of the energy distance over more usual kernels such as the Gaussian (Example 8.9) is that it is scale-free and does not depend on a bandwidth parameter $\sigma$. When denoting $f_s(x) = sx$ the dilation by a factor $s > 0$,

$$
\|f_{s\sharp}(\alpha - \beta)\|_{\mathrm{ED}(\mathbb{R}^d, \|\cdot\|^p)} = s^{p/2}\,\|\alpha - \beta\|_{\mathrm{ED}(\mathbb{R}^d, \|\cdot\|^p)},
$$

while the Wasserstein distance exhibits a perfect linear scaling,

$$
\mathcal{W}_p(f_{s\sharp}\alpha, f_{s\sharp}\beta) = s\,\mathcal{W}_p(\alpha, \beta).
$$

Note, however, that for the energy distance, the parameter $p$ must satisfy $0 < p < 2$, and that for $p = 2$, it degenerates to the distance between the means, so it is not a norm anymore.

</div>

### Wasserstein Spaces Are Not Hilbertian

Some of the special cases of the Wasserstein geometry outlined earlier in §2.6 have highlighted the fact that the optimal transport distance can sometimes be computed in closed form. They also illustrate that in such cases the optimal transport distance is a *Hilbertian* metric between probability measures, in the sense that there exists a map $\phi$ from the space of input measures onto a Hilbert space, as defined below.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 8.4</span><span class="math-callout__name">(Hilbertian Distance)</span></p>

A distance $d$ defined on a set $\mathcal{Z} \times \mathcal{Z}$ is said to be Hilbertian if there exists a Hilbert space $\mathcal{H}$ and a mapping $\phi : \mathcal{Z} \to \mathcal{H}$ such that for any pair $z, z'$ in $\mathcal{Z}$ we have that $d(z, z') = \|\phi(z) - \phi(z')\|\_\mathcal{H}$.

</div>

Hilbertian distances have many favorable properties when used in a data analysis context. First, they can be easily cast as radial basis function kernels: for any Hilbertian distance $d$, it is indeed known that $e^{-d^p/t}$ is a positive definite kernel for any value $0 \le p \le 2$ and any positive scalar $t$. Points living in a Hilbertian space can also be efficiently embedded in lower dimensions with low distortion factors.

Because Hilbertian distances have such properties, one might hope that the Wasserstein distance remains Hilbertian in more general settings. This can be disproved using the following equivalence.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8.1b</span><span class="math-callout__name">(Hilbertian Characterization)</span></p>

A distance $d$ is Hilbertian if and only if $d^2$ is negative definite.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Proposition 8.1b</summary>

If a distance is Hilbertian, then $d^2$ is trivially negative definite. Indeed, given $n$ points in $\mathcal{Z}$, the sum $\sum r_i r_j d^2(z_i, z_j)$ can be rewritten as $\sum r_i r_j \|\phi(z_i) - \phi(z_j)\|\_\mathcal{H}^2$ which can be expanded, taking advantage of the fact that $\sum r_i = 0$ to $-2\sum r_i r_j \langle \phi(z_i), \phi(z_j) \rangle\_\mathcal{H}$ which is negative by definition of a Hilbert dot product. If, on the contrary, $d^2$ is negative definite, then the fact that $d$ is Hilbertian proceeds from a key result by Schoenberg [1938].

</details>
</div>

It is therefore sufficient to show that the squared Wasserstein distance is not negative definite to show that it is not Hilbertian, as stated in the following proposition.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8.2</span><span class="math-callout__name">(Wasserstein is Not Hilbertian)</span></p>

If $\mathcal{X} = \mathbb{R}^d$ with $d \ge 2$ and the ground cost is set to $d(x, y) = \|x - y\|\_2$, then the $p$-Wasserstein distance is not Hilbertian for $p = 1, 2$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of Proposition 8.2</summary>

It suffices to prove the result for $d = 2$ since any counterexample in that dimension suffices to obtain a counterexample in any higher dimension. We provide a nonrandom counterexample which works using measures supported on four vectors $x^1, x^2, x^3, x^4 \in \mathbb{R}^2$ defined as follows: $x^1 = [0, 0]$, $x^2 = [1, 0]$, $x^3 = [0, 1]$, $x^4 = [1, 1]$. We now consider all points on the regular grid on the simplex of four dimensions, with increments of $1/4$. There are $35 = \binom{4+4-1}{4}$ such points in the simplex. Each probability vector $\mathbf{a}^i$ on that grid is such that for $j \le 4$, we have that $\mathbf{a}\_j^i$ is in the set $\lbrace 0, \frac{1}{4}, \frac{1}{2}, \frac{3}{4}, 1 \rbrace$ and such that $\sum\_{j=1}^4 \mathbf{a}\_j^i = 1$. For a given $p$, the $35 \times 35$ pairwise Wasserstein distance matrix $\mathbf{D}\_p$ between these histograms can be computed. $\mathbf{D}\_p$ is not negative definite if and only if its elementwise square $\mathbf{D}\_p^2$ is such that $\mathbf{J}\mathbf{D}\_p^2\mathbf{J}$ has positive eigenvalues, where $\mathbf{J}$ is the centering matrix $\mathbf{J} = \mathbb{I}\_n - \frac{1}{n}\mathbb{1}\_{n,n}$, which is the case as illustrated in Figure 8.6.

</details>
</div>

#### Embeddings and Distortion

An important body of work quantifies the hardness of approximating Wasserstein distances using Hilbertian embeddings. It has been shown that embedding measures in $\ell_2$ spaces incurs necessarily an important distortion as soon as $\mathcal{X} = \mathbb{R}^d$ with $d \ge 2$.

It is possible to embed quasi-isometrically $p$-Wasserstein spaces for $0 < p \le 1$ in $\ell_1$, but the equivalence constant between the distances grows fast with the dimension $d$. A closely related technique consists in using the characterization of $\mathcal{W}\_1$ as the dual of Lipschitz functions $f$ (see §6.2) and approximating the Lipschitz constraint $\|\nabla f\|\_1 \le 1$ by a weighted $\ell_1$ ball over the wavelets coefficients. This also provides a quasi-isometric embedding in $\ell_1$ and comes with the advantage that this embedding can be computed approximately in linear time when the input measures are discretized on uniform grids.

#### Negative/Positive Definite Variants of Optimal Transport

The *sliced* approximation to Wasserstein distances, essentially a sum of 1-D directional transportation distance computed on random push-forwards of measures projected on lines, is negative definite as the sum of negative definite functions. This result can be used to define a positive definite kernel. Another way to recover a positive definite kernel is to cast the optimal transport problem as a soft-min problem (over all possible transportation tables) rather than a minimum, as proposed by Kosowsky and Yuille [1994] to introduce entropic regularization. That soft-min defines a term whose neg-exponential (also known as a generating function) is positive definite.

### Empirical Estimators for OT, MMD and $\varphi$-Divergences

In an applied setting, given two input measures $(\alpha, \beta) \in \mathcal{M}\_+^1(\mathcal{X})^2$, an important statistical problem is to approximate the (usually unknown) divergence $D(\alpha, \beta)$ using only samples $(x_i)\_{i=1}^n$ from $\alpha$ and $(y_j)\_{j=1}^m$ from $\beta$. These samples are assumed to be independently identically distributed from their respective distributions.

#### Empirical Estimators for OT and MMD

For both Wasserstein distances $\mathcal{W}\_p$ (see 2.18) and MMD norms (see §8.2), a straightforward estimator of the unknown distance between distributions is to compute it directly between the empirical measures, hoping ideally that one can control the rate of convergence of the latter to the former,

$$
D(\alpha, \beta) \approx D(\hat{\alpha}_n, \hat{\beta}_m) \quad \text{where} \quad \begin{cases} \hat{\alpha}_n \stackrel{\text{def}}{=} \frac{1}{n}\sum_i \delta_{x_i}, \\ \hat{\beta}_m \stackrel{\text{def}}{=} \frac{1}{m}\sum_j \delta_{y_j}. \end{cases}
$$

Note that for $D(\alpha, \beta) = \|\cdot\|\_\mathrm{TV}$, since the TV norm does not metrize the weak convergence, $\|\hat{\alpha}\_n - \hat{\beta}\_n\|\_\mathrm{TV}$ is not a consistent estimator, namely it does not converge toward $\|\alpha - \beta\|\_\mathrm{TV}$. Indeed, with probability 1, $\|\hat{\alpha}\_n - \hat{\beta}\_n\|\_\mathrm{TV} = 2$ since the support of the two discrete measures does not overlap. Similar issues arise with other $\varphi$-divergences, which cannot be estimated using divergences between empirical distributions.

**Rates for OT.** For $\mathcal{X} = \mathbb{R}^d$ and measure supported on bounded domain, it is shown by Dudley [1969] that for $d > 2$, and $1 \le p < +\infty$,

$$
\mathbb{E}(|\mathcal{W}_p(\hat{\alpha}_n, \hat{\beta}_n) - \mathcal{W}_p(\alpha, \beta)|) = O(n^{-\frac{1}{d}}),
$$

where the expectation $\mathbb{E}$ is taken with respect to the random samples $(x_i, y_i)\_i$. This rate is tight in $\mathbb{R}^d$ if one of the two measures has a density with respect to the Lebesgue measure. This rate can be refined when the measures are supported on low-dimensional subdomains: Weed and Bach [2017] show that, indeed, the rate depends on the intrinsic dimensionality of the support.

**Rates for MMD.** For weak norms $\|\cdot\|\_k^2$ which are dual of RKHS norms (also called MMD), as defined in (8.13), and contrary to Wasserstein distances, the sample complexity does not depend on the ambient dimension

$$
\mathbb{E}(|\|\hat{\alpha}_n - \hat{\beta}_n\|_k - \|\alpha - \beta\|_k|) = O(n^{-\frac{1}{2}}).
$$

In order to define an unbiased estimator, and thus to be able to use, for instance, SGD when minimizing such losses, one should rather use the unbiased estimator

$$
\mathrm{MMD}_k(\hat{\alpha}_n, \hat{\beta}_n)^2 \stackrel{\text{def}}{=} \frac{1}{n(n-1)}\sum_{i,i'} k(x_i, x_{i'}) + \frac{1}{n(n-1)}\sum_{j,j'} k(y_j, y_{j'}) - 2\frac{1}{n^2}\sum_{i,j} k(x_i, y_j),
$$

which should be compared to (8.14). It satisfies $\mathbb{E}(\mathrm{MMD}\_k(\hat{\alpha}\_n, \hat{\beta}\_n)^2) = \|\alpha - \beta\|\_k^2$.

#### Empirical Estimators for $\varphi$-Divergences

It is not possible to approximate $\mathcal{D}\_\varphi(\alpha|\beta)$, as defined in (8.2), from discrete samples using $\mathcal{D}\_\varphi(\hat{\alpha}\_n|\hat{\beta}\_n)$. Indeed, this quantity is either $+\infty$ (for instance, for the KL divergence) or is not converging to $\mathcal{D}\_\varphi(\alpha|\beta)$ as $n \to +\infty$ (for instance, for the TV norm). Instead, it is required to use a density estimator to somehow smooth the discrete empirical measures and replace them by densities. In a Euclidean space $\mathcal{X} = \mathbb{R}^d$, introducing $h_\sigma = h(\cdot/\sigma)$ with a smooth windowing function and a bandwidth $\sigma > 0$, a density estimator for $\alpha$ is defined using a convolution against this kernel,

$$
\hat{\alpha}_n \star h_\sigma = \frac{1}{n}\sum_i h_\sigma(\cdot - x_i). \tag{8.20}
$$

It is also possible to devise nonparametric estimators, bypassing the choice of a fixed bandwidth $\sigma$ to select instead a number $k$ of nearest neighbors. Denoting $\Delta_k(x)$ the distance between $x \in \mathbb{R}^d$ and its $k$th nearest neighbor among the $(x_i)\_{i=1}^n$, a density estimator is defined as

$$
\rho_{\hat{\alpha}_n}^k(x) \stackrel{\text{def}}{=} \frac{k/n}{|B_d|\Delta_k(x)^r}, \tag{8.21}
$$

where $|B_d|$ is the volume of the unit ball in $\mathbb{R}^d$. Instead of somehow "counting" the number of samples falling in an area of width $\sigma$ in (8.20), this formula (8.21) estimates the radius required to encapsulate $k$ samples.

### Entropic Regularization: Between OT and MMD

Following Proposition 4.7, we recall that the Sinkhorn divergence is defined as

$$
\mathfrak{W}_\mathbf{C}^\varepsilon(\mathbf{a}, \mathbf{b}) \stackrel{\text{def}}{=} \langle \mathbf{P}^\star,\, \mathbf{C} \rangle = \langle e^{\frac{\mathbf{f}^\star}{\varepsilon}},\, (\mathbf{K} \odot \mathbf{C})e^{\frac{\mathbf{g}^\star}{\varepsilon}} \rangle,
$$

where $\mathbf{P}^\star$ is the solution of (4.2) while $(\mathbf{f}^\star, \mathbf{g}^\star)$ are solutions of (4.30). Assuming $\mathbf{C}\_{i,j} = d(x_i, x_j)^p$ for some distance $d$ on $\mathcal{X}$, for two discrete probability distributions of the form (2.3), this defines a regularized Wasserstein cost

$$
\mathcal{W}_{p,\varepsilon}(\alpha, \beta)^p \stackrel{\text{def}}{=} \mathfrak{W}_\mathbf{C}^\varepsilon(\mathbf{a}, \mathbf{b}).
$$

This definition is generalized to any input distribution (not necessarily discrete) as

$$
\mathcal{W}_{p,\varepsilon}(\alpha, \beta)^p \stackrel{\text{def}}{=} \int_{\mathcal{X} \times \mathcal{X}} d(x, y)^p \mathrm{d}\pi^\star(x, y),
$$

where $\pi^\star$ is the solution of (4.9). In order to cancel the bias introduced by the regularization (in particular, $\mathcal{W}\_{p,\varepsilon}(\alpha, \alpha) \neq 0$), we introduce a corrected regularized divergence

$$
\tilde{\mathcal{W}}_{p,\varepsilon}(\alpha, \beta)^p \stackrel{\text{def}}{=} 2\,\mathcal{W}_{p,\varepsilon}(\alpha, \beta)^p - \mathcal{W}_{p,\varepsilon}(\alpha, \alpha)^p - \mathcal{W}_{p,\varepsilon}(\beta, \beta)^p.
$$

It is proved in [Feydy et al., 2019] that if $e^{-c/\varepsilon}$ is a positive kernel, then a related corrected divergence (obtained by using $\mathrm{L}\_\mathbf{C}^\varepsilon$ in place of $\mathfrak{W}\_\mathbf{C}^\varepsilon$) is positive.

The following proposition, whose proof can be found in [Ramdas et al., 2017], shows that this regularized divergence interpolates between the Wasserstein distance and the energy distance defined in Example 8.12.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 8.3</span><span class="math-callout__name">(Interpolation Between OT and MMD)</span></p>

One has

$$
\tilde{\mathcal{W}}_{p,\varepsilon}(\alpha, \beta) \xrightarrow{\varepsilon \to 0} 2\,\mathcal{W}_p(\alpha, \beta) \quad \text{and} \quad \tilde{\mathcal{W}}_{p,\varepsilon}(\alpha, \beta)^p \xrightarrow{\varepsilon \to +\infty} \|\alpha - \beta\|_{\mathrm{ED}(\mathcal{X}, d)}^2,
$$

where $\|\cdot\|\_{\mathrm{ED}(\mathcal{X}, d)}$ is defined in (8.19).

</div>

Figure 8.9 shows numerically the impact of $\varepsilon$ on the sample complexity rates. It is proved in Genevay et al. [2019], in the case of $c(x, y) = \|x - y\|^2$ on $\mathcal{X} = \mathbb{R}^d$, that these rates interpolate between the ones of OT and MMD.

## Chapter 9: Variational Wasserstein Problems

In data analysis, common divergences between probability measures (e.g. Euclidean, total variation, Hellinger, Kullback--Leibler) are often used to measure a fitting error or a loss in parameter estimation problems. The optimal transport geometry has a unique ability, not shared with other information divergences, to leverage physical ideas (mass displacement) and geometry (a ground cost between observations or bins) to compare measures. The main technical challenge lies in approximating and differentiating efficiently the Wasserstein distance.

### 9.1 Differentiating the Wasserstein Loss

In statistics, text processing or imaging, one must usually compare a probability distribution $\beta$ arising from measurements to a model, namely a parameterized family of distributions $\lbrace \alpha_\theta, \theta \in \Theta \rbrace$, where $\Theta$ is a subset of a Euclidean space. The computation of a suitable parameter $\theta$ is obtained by minimizing directly

$$
\min_{\theta \in \Theta} \mathcal{E}(\theta) \stackrel{\text{def}}{=} \mathcal{L}_c(\alpha_\theta, \beta).
$$

**Convexity.** The Wasserstein distance between two histograms or two densities is convex with respect to its two inputs. Therefore, when $\theta$ is itself a histogram, namely $\Theta = \Sigma_n$ and $\alpha_\theta = \theta$, or more generally when $\theta$ describes $K$ weights in the simplex, $\Theta = \Sigma_K$, and $\alpha_\theta = \sum_{i=1}^K \theta_i \alpha_i$ is a convex combination of known atoms $\alpha_1, \ldots, \alpha_K$ in $\Sigma_N$, Problem (9.1) remains convex. However, for more general parameterizations $\theta \mapsto \alpha_\theta$, Problem (9.1) is in general not convex.

#### 9.1.1 Eulerian Discretization

A first way to discretize the problem is to suppose that both distributions $\beta = \sum_{j=1}^m \mathbf{b}_j \delta_{y_j}$ and $\alpha_\theta = \sum_{i=1}^n \mathbf{a}(\theta)_i \delta_{x_i}$ are discrete distributions defined on fixed locations $(x_i)_i$ and $(y_j)_j$. The parameterized measure $\alpha_\theta$ is in that case entirely represented through the weight vector $\mathbf{a} : \theta \mapsto \mathbf{a}(\theta) \in \Sigma_n$, which in practice might be very sparse if the grid is large. In its original form, the objective of Problem (9.1) is not differentiable. In order to obtain a smooth minimization problem, we use the entropic regularized OT and approximate (9.1) using

$$
\min_{\theta \in \Theta} \mathcal{E}_E(\theta) \stackrel{\text{def}}{=} \mathrm{L}_\mathbf{C}^\varepsilon(\mathbf{a}(\theta), \mathbf{b}) \quad \text{where} \quad \mathbf{C}_{i,j} \stackrel{\text{def}}{=} c(x_i, y_j).
$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 9.1</span><span class="math-callout__name">(Derivative with Respect to Histograms)</span></p>

For $\varepsilon > 0$, $(\mathbf{a}, \mathbf{b}) \mapsto \mathrm{L}_\mathbf{C}^\varepsilon(\mathbf{a}, \mathbf{b})$ is convex and differentiable. Its gradient reads

$$
\nabla \mathrm{L}_\mathbf{C}^\varepsilon(\mathbf{a}, \mathbf{b}) = (\mathbf{f}, \mathbf{g}),
$$

where $(\mathbf{f}, \mathbf{g})$ is the unique solution to (4.30), centered such that $\sum_i \mathbf{f}_i = \sum_j \mathbf{g}_j = 0$. For $\varepsilon = 0$, this formula defines the elements of the sub-differential of $\mathrm{L}_\mathbf{C}^\varepsilon$, and the function is differentiable if they are unique.

</div>

The zero mean condition on $(\mathbf{f}, \mathbf{g})$ is important when using gradient descent to guarantee conservation of mass. Using the chain rule, one thus obtains that $\mathcal{E}_E$ is smooth and that its gradient is

$$
\nabla \mathcal{E}_E(\theta) = [\partial \mathbf{a}(\theta)]^\top(\mathbf{f}),
$$

where $\partial \mathbf{a}(\theta) \in \mathbb{R}^{n \times \dim(\Theta)}$ is the Jacobian (differential) of the map $\mathbf{a}(\theta)$, and where $\mathbf{f} \in \mathbb{R}^n$ is the dual potential vector associated to the dual entropic OT (4.30) between $\mathbf{a}(\theta)$ and $\mathbf{b}$ for the cost matrix $\mathbf{C}$ (which is fixed in a Eulerian setting, and in particular independent of $\theta$).

#### 9.1.2 Lagrangian Discretization

A different approach consists in using instead fixed (typically uniform) weights and approximating an input measure $\alpha$ as an empirical measure $\alpha_\theta = \frac{1}{n} \sum_i \delta_{x(\theta)_i}$ for a point-cloud parameterization map $x : \theta \mapsto x(\theta) = (x(\theta)_i)_{i=1}^n \in \mathcal{X}^n$, where we assume here that $\mathcal{X}$ is Euclidean. Problem (9.1) is thus approximated as

$$
\min_\theta \mathcal{E}_L(\theta) \stackrel{\text{def}}{=} \mathrm{L}_{\mathbf{C}(x(\theta))}^\varepsilon(\mathbb{1}_n / n, \mathbf{b}) \quad \text{where} \quad \mathbf{C}(x)_{i,j} \stackrel{\text{def}}{=} c(x(\theta)_i, y_j).
$$

Note that here the cost matrix $\mathbf{C}(x(\theta))$ now depends on $\theta$ since the support of $\alpha_\theta$ changes with $\theta$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 9.2</span><span class="math-callout__name">(Derivative with Respect to the Cost)</span></p>

For fixed input histograms $(\mathbf{a}, \mathbf{b})$, for $\varepsilon > 0$, the mapping $\mathbf{C} \mapsto \mathcal{R}(\mathbf{C}) \stackrel{\text{def}}{=} \mathrm{L}_\mathbf{C}^\varepsilon(\mathbf{a}, \mathbf{b})$ is concave and smooth, and

$$
\nabla \mathcal{R}(\mathbf{C}) = \mathbf{P},
$$

where $\mathbf{P}$ is the unique optimal solution of (4.2). For $\varepsilon = 0$, this formula defines the set of upper gradients.

</div>

Assuming $(\mathcal{X}, \mathcal{Y})$ are convex subsets of $\mathbb{R}^d$, for discrete measures $(\alpha, \beta)$ of the form (2.3), one obtains using the chain rule that $x = (x_i)_{i=1}^n \in \mathcal{X}^n \mapsto \mathcal{F}(x) \stackrel{\text{def}}{=} \mathrm{L}_{\mathbf{C}(x)}(\mathbb{1}_n / n, \mathbf{b})$ is smooth and that

$$
\nabla \mathcal{F}(x) = \left( \sum_{j=1}^m \mathbf{P}_{i,j} \nabla_1 c(x_i, y_j) \right)_{i=1}^n \in \mathcal{X}^n,
$$

where $\nabla_1 c$ is the gradient with respect to the first variable. For instance, for $\mathcal{X} = \mathcal{Y} = \mathbb{R}^d$, for $c(s, t) = \|s - t\|^2$, one has

$$
\nabla \mathcal{F}(x) = 2 \left( \mathbf{a}_i x_i - \sum_{j=1}^m \mathbf{P}_{i,j} y_j \right)_{i=1}^n,
$$

where $\mathbf{a}_i = 1/n$ here. Note that, up to a constant, this gradient is $\mathrm{Id} - T$, where $T$ is the barycentric projection defined in (4.19). Using the chain rule, one thus obtains that the Lagrangian discretized problem (9.4) is smooth and its gradient is

$$
\nabla \mathcal{E}_L(\theta) = [\partial x(\theta)]^\top (\nabla \mathcal{F}(x(\theta))),
$$

where $\partial x(\theta) \in \mathbb{R}^{\dim(\Theta) \times (nd)}$ is the Jacobian of the map $x(\theta)$ and where $\nabla \mathcal{F}$ is implemented as in (9.6) or (9.7) using for $\mathbf{P}$ the optimal coupling matrix between $\alpha_\theta$ and $\beta$.

#### 9.1.3 Automatic Differentiation

The difficulty when applying formulas (9.3) and (9.8) is that one needs to compute the exact optimal solutions $\mathbf{f}$ or $\mathbf{P}$ for these formulas to be valid, which can only be achieved with acceptable precision using a very large number of Sinkhorn iterates. In challenging situations, the computational budget to compute a single Wasserstein distance is usually limited, therefore allowing only for a few Sinkhorn iterations. In that case, it is usually better to differentiate directly the output of Sinkhorn's algorithm, using reverse mode automatic differentiation. This corresponds to using the "algorithmic" Sinkhorn divergences as introduced in (4.48), rather than the quantity $\mathrm{L}_\mathbf{C}^\varepsilon$ in (4.2), and differentiating it directly as a composition of simple maps using the inputs, either the histogram in the Eulerian case or the cost matrix in the Lagrangian cases.

The cost for computing the gradient of functionals involving Sinkhorn divergences is the same as that of computation of the functional itself. The only downside is that reverse mode automatic differentiation is memory intensive (the memory grows proportionally with the number of iterations). There exist, however, subsampling strategies that mitigate this problem.

### 9.2 Wasserstein Barycenters, Clustering and Dictionary Learning

A basic problem in unsupervised learning is to compute the "mean" or "barycenter" of several data points. A classical way to define such a weighted mean of points $(x_s)_{s=1}^S \in \mathcal{X}^S$ living in a metric space $(\mathcal{X}, d)$ is by solving a variational problem

$$
\min_{x \in \mathcal{X}} \sum_{s=1}^S \lambda_s d(x, x_s)^p
$$

for a given family of weights $(\lambda_s)_s \in \Sigma_S$, where $p$ is often set to $p = 2$. When $\mathcal{X} = \mathbb{R}^d$ and $d(x, y) = \|x - y\|_2$, this leads to the usual definition of the linear average $x = \sum_s \lambda_s x_s$ for $p = 2$ and the more evolved median point when $p = 1$. This process is often referred to as the "Frechet" or "Karcher" mean. For a generic distance $d$, Problem (9.9) is usually a difficult nonconvex optimization problem. Fortunately, in the case of optimal transport distances, the problem can be formulated as a convex program for which existence can be proved and efficient numerical schemes exist.

**Frechet means over the Wasserstein space.** Given input histograms $\lbrace \mathbf{b}_s \rbrace_{s=1}^S$, where $\mathbf{b}_s \in \Sigma_{n_s}$, and weights $\lambda \in \Sigma_S$, a Wasserstein barycenter is computed by minimizing

$$
\min_{\mathbf{a} \in \Sigma_n} \sum_{s=1}^S \lambda_s \mathrm{L}_{\mathbf{C}_s}(\mathbf{a}, \mathbf{b}_s),
$$

where the cost matrices $\mathbf{C}_s \in \mathbb{R}^{n \times n_s}$ need to be specified. A typical setup is "Eulerian," so that all the barycenters are defined on the same grid, $n_s = n$, $\mathbf{C}_s = \mathbf{C} = \mathbf{D}^p$ is set to be a distance matrix, to solve

$$
\min_{\mathbf{a} \in \Sigma_n} \sum_{s=1}^S \lambda_s \mathrm{W}_p^p(\mathbf{a}, \mathbf{b}_s).
$$

The barycenter problem for histograms (9.10) is in fact a linear program, since one can look for the $S$ couplings $(\mathbf{P}_s)_s$ between each input and the barycenter itself, which by construction must be constrained to share the same row marginal,

$$
\min_{\mathbf{a} \in \Sigma_n, (\mathbf{P}_s \in \mathbb{R}^{n \times n_s})_s} \left\lbrace \sum_{s=1}^S \lambda_s \langle \mathbf{P}_s, \mathbf{C}_s \rangle \;:\; \forall\, s,\; \mathbf{P}_s^\top \mathbb{1}_{n_s} = \mathbf{a},\; \mathbf{P}_s^\top \mathbb{1}_n = \mathbf{b}_s \right\rbrace.
$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 9.1</span><span class="math-callout__name">(Barycenter of Arbitrary Measures)</span></p>

Given a set of input measures $(\beta_s)_s$ defined on some space $\mathcal{X}$, the barycenter problem becomes

$$
\min_{\alpha \in \mathcal{M}_+^1(\mathcal{X})} \sum_{s=1}^S \lambda_s \mathcal{L}_c(\alpha, \beta_s).
$$

In the case where $\mathcal{X} = \mathbb{R}^d$ and $c(x, y) = \|x - y\|^2$, Agueh and Carlier [2011] show that if one of the input measures has a density, then this barycenter is unique.

</div>

Problem (9.11) can be viewed as a generalization of computing barycenters of points $(x_s)_{s=1}^S \in \mathcal{X}^S$ to arbitrary measures. Indeed, if $\beta_s = \delta_{x_s}$ is a single Dirac mass, then a solution to (9.11) is $\delta_{x^\star}$, where $x^\star$ is a Frechet mean solving (9.9). Note that for $c(x, y) = \|x - y\|^2$, the mean of the barycenter $\alpha^\star$ is necessarily the barycenter of the mean, i.e.

$$
\int_\mathcal{X} x \mathrm{d}\alpha^\star(x) = \sum_s \lambda_s \int_\mathcal{X} x \mathrm{d}\alpha_s(x),
$$

and the support of $\alpha^\star$ is located in the convex hull of the supports of the $(\alpha_s)_s$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 9.2</span><span class="math-callout__name">($k$-Means as a Wasserstein Variational Problem)</span></p>

When the family of input measures $(\beta_s)_s$ is limited to but one measure $\beta$, this measure is supported on a discrete finite subset of $\mathcal{X} = \mathbb{R}^d$, and the cost is the squared Euclidean distance, then one can show that the barycenter problem

$$
\min_{\alpha \in \mathcal{M}_+^1(\mathcal{X})} \mathcal{L}_c(\alpha, \beta),
$$

where $\alpha$ is constrained to be a discrete measure with a finite support of size up to $k$, is equivalent to the usual $k$-means problem taking $\beta$. Indeed, one can easily show that the centroids output by the $k$-means problem correspond to the support of the solution $\alpha$ and that its weights correspond to the fraction of points in $\beta$ assigned to each centroid. Approximating $\mathcal{L}_c$ using entropic regularization results in smoothed out assignments that appear in soft-clustering variants of $k$-means, such as mixtures of Gaussians.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 9.3</span><span class="math-callout__name">(Distribution of Distributions and Consistency)</span></p>

It is possible to generalize (9.11) to a possibly infinite collection of measures. This problem is described by considering a probability distribution $M$ over the space $\mathcal{M}_+^1(\mathcal{X})$ of probability distributions, i.e. $M \in \mathcal{M}_+^1(\mathcal{M}_+^1(\mathcal{X}))$. A barycenter is then a solution of

$$
\min_{\alpha \in \mathcal{M}_+^1(\mathcal{X})} \mathbb{E}_M(\mathcal{L}_c(\alpha, \beta)) = \int_{\mathcal{M}_+^1(\mathcal{X})} \mathcal{L}_c(\alpha, \beta) \mathrm{d}M(\beta),
$$

where $\beta$ is a random measure distributed according to $M$. Drawing uniformly at random a finite number $S$ of input measures $(\beta_s)_{s=1}^S$ according to $M$, one can then define $\hat{\beta}_S$ as being a solution of (9.11) for uniform weights $\lambda_s = 1/S$ (note that here $\hat{\beta}_S$ is itself a random measure). The convergence (in expectation or with high probability) of $\mathcal{L}_c(\hat{\beta}_S, \alpha)$ to zero (where $\alpha$ is the unique solution to (9.13)) corresponds to the consistency of the barycenters. This can be interpreted as a law of large numbers over the Wasserstein space.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 9.4</span><span class="math-callout__name">(Fixed-Point Map)</span></p>

When dealing with the Euclidean space $\mathcal{X} = \mathbb{R}^d$ with ground cost $c(x, y) = \|x - y\|^2$, it is possible to study the barycenter problem using transportation maps. Indeed, if $\alpha$ has a density, according to Remark 2.24, one can define optimal transportation maps $T_s$ between $\alpha$ and $\alpha_s$, in particular such that $T_{s,\sharp} \alpha = \alpha_s$. The average map

$$
T^{(\alpha)} \stackrel{\text{def}}{=} \sum_{s=1}^S \lambda_s T_s
$$

(the notation above makes explicit the dependence of this map on $\alpha$) is itself an optimal map between $\alpha$ and $T_\sharp^{(\alpha)} \alpha$ (a positive combination of optimal maps is equal by Brenier's theorem, Remark 2.24, to the sum of gradients of convex functions, equal to the gradient of a sum of convex functions, and therefore optimal by Brenier's theorem again). First order optimality conditions of the barycenter problem (9.13) actually read $T^{(\alpha^\star)} = \mathrm{Id}_{\mathbb{R}^d}$ (the identity map) at the optimal measure $\alpha^\star$ (the barycenter), and the barycenter $\alpha^\star$ is the unique (under regularity conditions) solution to the fixed-point equation

$$
G(\alpha) = \alpha \quad \text{where} \quad G(\alpha) \stackrel{\text{def}}{=} T_\sharp^{(\alpha)} \alpha,
$$

It has been shown that $\alpha \mapsto G(\alpha)$ strictly decreases the objective function of (9.13) if $\alpha$ is not the barycenter and that the fixed-point iterations $\alpha^{(\ell+1)} \stackrel{\text{def}}{=} G(\alpha^{(\ell)})$ converge to the barycenter $\alpha^\star$. This fixed point algorithm can be used in cases where the optimal transportation maps are known in closed form (e.g. for Gaussians).

</div>

**Special cases.** In general, solving (9.10) or (9.11) is not straightforward, but there exist some special cases for which solutions are explicit or simple.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 9.5</span><span class="math-callout__name">(Barycenter of Gaussians)</span></p>

It is shown in Agueh and Carlier [2011] that the barycenter of Gaussians distributions $\alpha_s = \mathcal{N}(\mathbf{m}_s, \boldsymbol{\Sigma}_s)$, for the squared Euclidean cost $c(x, y) = \|x - y\|^2$, is itself a Gaussian $\mathcal{N}(\mathbf{m}^\star, \boldsymbol{\Sigma}^\star)$. Making use of (2.41), one sees that the barycenter mean is the mean of the inputs

$$
\mathbf{m}^\star = \sum_s \lambda_s \mathbf{m}_s
$$

while the covariance minimizes

$$
\min_{\boldsymbol{\Sigma}} \sum_s \lambda_s \mathcal{B}(\boldsymbol{\Sigma}, \boldsymbol{\Sigma}_s)^2,
$$

where $\mathcal{B}$ is the Bure metric (2.42). The first order optimality condition of this convex problem shows that $\boldsymbol{\Sigma}^\star$ is the unique positive definite fixed point of the map

$$
\boldsymbol{\Sigma}^\star = \Psi(\boldsymbol{\Sigma}^\star) \quad \text{where} \quad \Psi(\boldsymbol{\Sigma}) \stackrel{\text{def}}{=} \sum_s \lambda_s (\boldsymbol{\Sigma}^{\frac{1}{2}} \boldsymbol{\Sigma}_s \boldsymbol{\Sigma}^{\frac{1}{2}})^{\frac{1}{2}},
$$

where $\boldsymbol{\Sigma}^{\frac{1}{2}}$ is the square root of positive semidefinite matrices. While $\Psi$ is not strictly contracting, iterating this fixed-point map, i.e. defining $\boldsymbol{\Sigma}^{(\ell+1)} \stackrel{\text{def}}{=} \Psi(\boldsymbol{\Sigma}^{(\ell)})$ converges in practice to the solution $\boldsymbol{\Sigma}^\star$. Alvarez-Esteban et al. [2016] have also proposed to use an alternative map

$$
\bar{\Psi}(\boldsymbol{\Sigma}) \stackrel{\text{def}}{=} \boldsymbol{\Sigma}^{-\frac{1}{2}} \left( \sum_s \lambda_s (\boldsymbol{\Sigma}^{\frac{1}{2}} \boldsymbol{\Sigma}_s \boldsymbol{\Sigma}^{\frac{1}{2}})^{\frac{1}{2}} \right)^2 \boldsymbol{\Sigma}^{-\frac{1}{2}}
$$

for which the iterations $\boldsymbol{\Sigma}^{(\ell+1)} \stackrel{\text{def}}{=} \bar{\Psi}(\boldsymbol{\Sigma}^{(\ell)})$ converge. This is because the fixed-point map $G$ defined in (9.14) preserves Gaussian distributions, and in fact,

$$
G(\mathcal{N}(\mathbf{m}, \boldsymbol{\Sigma})) = \mathcal{N}(\mathbf{m}^\star, \bar{\Psi}(\boldsymbol{\Sigma})).
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 9.6</span><span class="math-callout__name">(1-D Cases)</span></p>

For 1-D distributions, the $W_p$ barycenter can be computed almost in closed form using the fact that the transport is the monotone rearrangement, as detailed in Remark 2.30. The simplest case is for empirical measures with $n$ points, i.e. $\beta_s = \frac{1}{n} \sum_{i=1}^n \delta_{y_{s,i}}$, where the points are assumed to be sorted $y_{s,1} \leq y_{s,2} \leq \ldots$. Using (2.33) the barycenter $\alpha_\lambda$ is also an empirical measure on $n$ points

$$
\alpha_\lambda = \frac{1}{n} \sum_{i=1}^n \delta_{x_{\lambda,i}} \quad \text{where} \quad x_{\lambda,i} = A_\lambda(x_{s,i})_s,
$$

where $A_\lambda$ is the barycentric map

$$
A_\lambda(x_s)_s \stackrel{\text{def}}{=} \operatorname*{argmin}_{x \in \mathbb{R}} \sum_{s=1}^S \lambda_s |x - x_s|^p.
$$

For instance, for $p = 2$, one has $x_{\lambda,i} = \sum_{s=1}^S \lambda_s x_{s,i}$. In the general case, one needs to use the cumulative functions as defined in (2.34), and using (2.36), one has

$$
\forall\, r \in [0, 1], \quad \mathcal{C}_{\alpha_\lambda}^{-1}(r) = A_\lambda(\mathcal{C}_{\alpha_s}^{-1}(r))_{s=1}^S,
$$

which can be used, for instance, to compute barycenters between discrete measures supported on less than $n$ points in $O(n \log(n))$ operations, using a simple sorting procedure.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 9.7</span><span class="math-callout__name">(Simple Cases)</span></p>

Denoting by $T_{r,u} : x \mapsto rx + u$ a scaling and translation, and assuming that $\alpha_s = T_{r_s, u_s, \sharp} \alpha_0$ is obtained by scaling and translating an initial template measure, then a barycenter $\alpha_\lambda$ is also obtained using scaling and translation

$$
\alpha_\lambda = T_{r^\star, u^\star, \sharp} \alpha_0 \quad \text{where} \quad \begin{cases} r^\star = \left( \sum_s \lambda_s / r_s \right)^{-1}, \\ u^\star = \sum_s \lambda_s u_s. \end{cases}
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 9.8</span><span class="math-callout__name">(Case $S = 2$)</span></p>

In the case where $\mathcal{X} = \mathbb{R}^d$ and $c(x, y) = \|x - y\|^2$ (this can be extended more generally to geodesic spaces), the barycenter between $S = 2$ measures $(\alpha_0, \alpha_1)$ is the McCann interpolant as already introduced in (7.6). Denoting $T_\sharp \alpha_0 = \alpha_1$ the Monge map, one has that the barycenter $\alpha_\lambda$ reads $\alpha_\lambda = (\lambda_1 \mathrm{Id} + \lambda_2 T)_\sharp \alpha_0$.

</div>

**Entropic approximation of barycenters.** One can use entropic smoothing and approximate the solution of (9.10) using

$$
\min_{\mathbf{a} \in \Sigma_n} \sum_{s=1}^S \lambda_s \mathrm{L}_{\mathbf{C}_s}^\varepsilon(\mathbf{a}, \mathbf{b}_s)
$$

for some $\varepsilon > 0$. This is a smooth convex minimization problem. A simpler yet very effective approach is to rewrite (9.15) as a (weighted) KL projection problem

$$
\min_{(\mathbf{P}_s)_s} \left\lbrace \sum_s \lambda_s \varepsilon \mathrm{KL}(\mathbf{P}_s | \mathbf{K}_s) \;:\; \forall\, s,\; \mathbf{P}_s^\top \mathbb{1}_m = \mathbf{b}_s,\; \mathbf{P}_1 \mathbb{1}_1 = \cdots = \mathbf{P}_S \mathbb{1}_S \right\rbrace
$$

where we denoted $\mathbf{K}_s \stackrel{\text{def}}{=} e^{-\mathbf{C}_s / \varepsilon}$. Here, the barycenter $\mathbf{a}$ is implicitly encoded in the row marginals of all the couplings $\mathbf{P}_s \in \mathbb{R}^{n \times n_s}$ as $\mathbf{a} = \mathbf{P}_1 \mathbb{1}_1 = \cdots = \mathbf{P}_S \mathbb{1}_S$. One can generalize Sinkhorn to this problem, which also corresponds to iterative projections. The optimal couplings $(\mathbf{P}_s)_s$ solving (9.16) are computed in scaling form as

$$
\mathbf{P}_s = \mathrm{diag}(\mathbf{u}_s) \mathbf{K} \mathrm{diag}(\mathbf{v}_s),
$$

and the scalings are sequentially updated as

$$
\forall\, s \in [\![1, S]\!], \quad \mathbf{v}_s^{(\ell+1)} \stackrel{\text{def}}{=} \frac{\mathbf{b}_s}{\mathbf{K}_s^\top \mathbf{u}_s^{(\ell)}},
$$

$$
\forall\, s \in [\![1, S]\!], \quad \mathbf{u}_s^{(\ell+1)} \stackrel{\text{def}}{=} \frac{\mathbf{a}^{(\ell+1)}}{\mathbf{K}_s \mathbf{v}_s^{(\ell+1)}},
$$

$$
\text{where} \quad \mathbf{a}^{(\ell+1)} \stackrel{\text{def}}{=} \prod_s (\mathbf{K}_s \mathbf{v}_s^{(\ell+1)})^{\lambda_s}.
$$

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 9.1</span><span class="math-callout__name">(Dual Problem for Barycenters)</span></p>

The optimal $(\mathbf{u}_s, \mathbf{v}_s)$ appearing in (9.17) can be written as $(\mathbf{u}_s, \mathbf{v}_s) = (e^{\mathbf{f}_s / \varepsilon}, e^{\mathbf{g}_s / \varepsilon})$, where $(\mathbf{f}_s, \mathbf{g}_s)_s$ are the solutions of the following program (whose value matches the one of (9.15)):

$$
\max_{(\mathbf{f}_s, \mathbf{g}_s)_s} \left\lbrace \sum_s \lambda_s \left( \langle \mathbf{g}_s, \mathbf{b}_s \rangle - \varepsilon \langle \mathbf{K}_s e^{\mathbf{g}_s / \varepsilon}, e^{\mathbf{f}_s / \varepsilon} \rangle \right) \;:\; \sum_s \lambda_s \mathbf{f}_s = 0 \right\rbrace.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 9.9</span><span class="math-callout__name">(Wasserstein Propagation)</span></p>

It is possible to generalize the barycenter problem (9.10), where one looks for distributions $(\mathbf{b}_u)_{u \in U}$ at some given set $U$ of nodes in a graph $\mathcal{G}$ given a set of fixed input distributions $(\mathbf{b}_v)_{v \in V}$ on the complementary set $V$ of the nodes. The unknown are determined by minimizing the overall transportation distance between all pairs of nodes $(r, s) \in \mathcal{G}$ forming edges in the graph

$$
\min_{(\mathbf{b}_u \in \Sigma_{n_u})_{u \in U}} \sum_{(r,s) \in \mathcal{G}} \mathrm{L}_{\mathbf{C}_{r,s}}(\mathbf{b}_r, \mathbf{b}_s),
$$

where the cost matrices $\mathbf{C}_{r,s} \in \mathbb{R}^{n_r \times n_s}$ need to be specified by the user. The barycenter problem (9.10) is a special case of this problem where the considered graph $\mathcal{G}$ is "star shaped," where $U$ is a single vertex connected to all the other vertices $V$.

</div>

### 9.3 Gradient Flows

Given a smooth function $\mathbf{a} \mapsto F(\mathbf{a})$, one can use the standard gradient descent

$$
\mathbf{a}^{(\ell+1)} \stackrel{\text{def}}{=} \mathbf{a}^{(\ell)} - \tau \nabla F(\mathbf{a}^{(\ell)}),
$$

where $\tau$ is a small enough step size. This corresponds to a so-called "explicit" minimization scheme and only applies for smooth functions $F$. For nonsmooth functions, one can use instead an "implicit" scheme, which is also called the proximal-point algorithm

$$
\mathbf{a}^{(\ell+1)} \stackrel{\text{def}}{=} \mathrm{Prox}_{\tau F}^{\|\cdot\|}(\mathbf{a}^{(\ell)}) \stackrel{\text{def}}{=} \operatorname*{argmin}_{\mathbf{a}} \frac{1}{2} \left\| \mathbf{a} - \mathbf{a}^{(\ell)} \right\|^2 + \tau F(\mathbf{a}).
$$

Note that this corresponds to the Euclidean proximal operator, already encountered in (7.13). The update (9.24) can be understood as iterating the explicit operator $\mathrm{Id} - \tau \nabla F$, while (9.25) makes use of the implicit operator $(\mathrm{Id} + \tau \nabla F)^{-1}$. For convex $F$, iterations (9.25) always converge, for any value of $\tau > 0$.

If the function $F$ is defined on the simplex of histograms $\Sigma_n$, then it makes sense to use an optimal transport metric in place of the $\ell^2$ norm $\|\cdot\|$ in (9.25), in order to solve

$$
\mathbf{a}^{(\ell+1)} \stackrel{\text{def}}{=} \operatorname*{argmin}_{\mathbf{a}} \mathrm{W}_p(\mathbf{a}, \mathbf{a}^{(\ell)})^p + \tau F(\mathbf{a}).
$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 9.10</span><span class="math-callout__name">(Wasserstein Gradient Flows)</span></p>

Equation (9.26) can be generalized to arbitrary measures by defining the iteration

$$
\alpha^{(\ell+1)} \stackrel{\text{def}}{=} \operatorname*{argmin}_\alpha \mathcal{W}_p(\alpha, \alpha^{(\ell)})^p + \tau F(\alpha)
$$

for some function $F$ defined on $\mathcal{M}_+^1(\mathcal{X})$. This implicit time stepping is a useful tool to construct continuous flows, by formally taking the limit $\tau \to 0$ and introducing the time $t = \tau \ell$, so that $\alpha^{(\ell)}$ is intended to approximate a continuous flow $t \in \mathbb{R}_+ \mapsto \alpha_t$. For the special case $p = 2$ and $\mathcal{X} = \mathbb{R}^d$, a formal calculus shows that $\alpha_t$ is expected to solve a PDE of the form

$$
\frac{\partial \alpha_t}{\partial t} = \mathrm{div}(\alpha_t \nabla(F'(\alpha_t))),
$$

where $F'(\alpha)$ denotes the derivative of the function $F$ in the sense that it is a continuous function $F'(\alpha) \in \mathcal{C}(\mathcal{X})$ such that

$$
F(\alpha + \varepsilon \xi) = F(\alpha) + \varepsilon \int_\mathcal{X} F'(\alpha) \mathrm{d}\xi(x) + o(\varepsilon).
$$

A typical example is when using $F = -H$, where $H(\alpha) = \mathrm{KL}(\alpha | \mathcal{L}_{\mathbb{R}^d})$ is the relative entropy with respect to the Lebesgue measure $\mathcal{L}_{\mathbb{R}^d}$ on $\mathcal{X} = \mathbb{R}^d$

$$
H(\alpha) = -\int_{\mathbb{R}^d} \rho_\alpha(x)(\log(\rho_\alpha(x)) - 1) \mathrm{d}x
$$

(setting $H(\alpha) = -\infty$ when $\alpha$ does not have a density), then (9.28) shows that the gradient flow of this neg-entropy is the linear heat diffusion

$$
\frac{\partial \alpha_t}{\partial t} = \Delta \alpha_t,
$$

where $\Delta$ is the spatial Laplacian. The heat diffusion can therefore be interpreted either as the "classical" Euclidian flow (performing "vertical" movements with respect to mass amplitudes) of the Dirichlet energy $\int_{\mathbb{R}^d} \|\nabla \rho_\alpha(x)\|^2 \mathrm{d}x$ or, alternatively, as the entropy for the optimal transport flow (performing a "horizontal" movement with respect to mass positions). Interest in Wasserstein gradient flows was sparked by the seminal paper of Jordan, Kinderlehrer and Otto [1998], and these evolutions are often called "JKO flows" following their work. JKO flows can be used to study in particular nonlinear evolution equations such as the porous medium equation, total variation flows, quantum drifts, or heat evolutions on manifolds.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 9.11</span><span class="math-callout__name">(Gradient Flows in Metric Spaces)</span></p>

The implicit stepping (9.27) is a special case of a more general formalism to define gradient flows over metric spaces $(\mathcal{X}, d)$. For some function $F(x)$ defined for $x \in \mathcal{X}$, the implicit discrete minimization step is then defined as

$$
x^{(\ell+1)} \in \operatorname*{argmin}_{x \in \mathcal{X}} d(x^{(\ell)}, x)^2 + \tau F(x).
$$

The JKO step (9.27) corresponds to the use of the Wasserstein distance on the space of probability distributions. In some cases, one can show that (9.31) admits a continuous flow limit $x_t$ as $\tau \to 0$ and $k\tau = t$. In the case that $\mathcal{X}$ also has a Euclidean structure, an explicit stepping is defined by linearizing $F$

$$
x^{(\ell+1)} = \operatorname*{argmin}_{x \in \mathcal{X}} d(x^{(\ell)}, x)^2 + \tau \langle \nabla F(x^{(\ell)}),\, x \rangle.
$$

In sharp contrast to the implicit formula (9.31) it is usually straightforward to compute but can be unstable. The implicit step is always stable, is also defined for nonsmooth $F$, but is usually not accessible in closed form.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 9.12</span><span class="math-callout__name">(Lagrangian Discretization Using Particles Systems)</span></p>

The finite-dimensional problem in (9.26) can be interpreted as the Eulerian discretization of a flow over the space of measures (9.27). An alternative way to discretize the problem, using the so-called Lagrangian method using particles systems, is to parameterize instead the solution as a (discrete) empirical measure moving with time, where the locations of that measure (and not its weights) become the variables of interest. In practice, one can consider a dynamic point cloud of particles $\alpha_t = \frac{1}{n} \sum_{i=1}^n \delta_{x_i(t)}$ indexed with time. The initial problem (9.26) is then replaced by a set of $n$ coupled ODE prescribing the dynamic of the points $X(t) = (x_i(t))_i \in \mathcal{X}^n$. If the energy $F$ is finite for discrete measures, then one can simply define $\mathcal{F}(X) = F(\frac{1}{n} \sum_{i=1}^n \delta_{x_i})$. Typical examples are linear functions $F(\alpha) = \int_\mathcal{X} V(x) \mathrm{d}\alpha(x)$ and quadratic interactions $F(\alpha) = \int_{\mathcal{X}^2} W(x, y) \mathrm{d}\alpha(x) \mathrm{d}\alpha(y)$, in which case one can use respectively

$$
\mathcal{F}(X) = \frac{1}{n} \sum_i V(x_i) \quad \text{and} \quad \mathcal{F}(X) = \frac{1}{n^2} \sum_{i,j} W(x_i, x_j).
$$

For functions such as generalized entropy, which are only finite for measures having densities, one should apply a density estimator to convert the point cloud into a density, which allows us to also define function $\mathcal{F}(x)$ consistent with $F$ as $n \to +\infty$. A typical example is for the entropy $F(\alpha) = H(\alpha)$ defined in (9.29), for which a consistent estimator (up to a constant term) can be obtained by summing the logarithms of the distances to nearest neighbors

$$
\mathcal{F}(X) = \frac{1}{n} \sum_i \log(d_X(x_i)) \quad \text{where} \quad d_X(x) = \min_{x' \in X, x' \neq x} \|x - x'\|.
$$

For small enough step sizes $\tau$, assuming $\mathcal{X} = \mathbb{R}^d$, the Wasserstein distance $\mathcal{W}_2$ matches the Euclidean distance on the points, i.e. if $|t - t'|$ is small enough, $\mathcal{W}_2(\alpha_t, \alpha_{t'}) = \|X(t) - X(t')\|$. The gradient flow is thus equivalent to the Euclidean flow on positions $X'(t) = -\nabla \mathcal{F}(X(t))$, which is discretized for times $t_k = \tau k$ similarly to (9.24) using explicit Euler steps

$$
X^{(\ell+1)} \stackrel{\text{def}}{=} X^{(\ell)} - \tau \nabla \mathcal{F}(X^{(\ell)}).
$$

In the simplest case of a linear function $F(\alpha) = \int_\mathcal{X} V(x) \mathrm{d}\alpha(x)$, the flow operates independently over each particle for the function $V$, $x_i'(t) = -\nabla V(x_i(t))$ (and is an advection PDE of the density along the integral curves of the flow).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 9.13</span><span class="math-callout__name">(Geodesic Convexity)</span></p>

An important concept related to gradient flows is the convexity of the functional $F$ with respect to the Wasserstein-2 geometry, i.e. the convexity of $F$ along Wasserstein geodesics (i.e. displacement interpolations as shown in Remark 7.1). The Wasserstein gradient flow (with a continuous time) for geodesically convex functionals converges to a global minimizer.

</div>

## Chapter 10: Extensions of Optimal Transport

This chapter details several variational problems that are related to (and share the same structure of) the Kantorovich formulation of optimal transport. The goal is to extend optimal transport to more general settings: several input histograms and measures, unnormalized ones, more general classes of measures, and optimal transport between measures that focuses on local regularities (points nearby in the source measure should be mapped onto points nearby in the target measure) rather than a total transport cost, including cases where these two measures live in different metric spaces.

### 10.1 Multimarginal Problems

Instead of coupling two input histograms using the Kantorovich formulation (2.11), one can couple $S$ histograms $(\mathbf{a}_s)_{s=1}^S$, where $\mathbf{a}_s \in \Sigma_{n_s}$, by solving the following multimarginal problem:

$$
\min_{\mathbf{P} \in \mathbf{U}(\mathbf{a}_s)_s} \langle \mathbf{C}, \mathbf{P} \rangle \stackrel{\text{def}}{=} \sum_s \sum_{i_s=1}^{n_s} \mathbf{C}_{i_1, \ldots, i_S} \mathbf{P}_{i_1, \ldots, i_S},
$$

where the set of valid couplings is

$$
\mathbf{U}(\mathbf{a}_s)_s = \left\lbrace \mathbf{P} \in \mathbb{R}^{n_1 \times \ldots \times n_S} \;:\; \forall\, s, \forall\, i_s,\; \sum_{\ell \neq s} \sum_{i_\ell = 1}^{n_\ell} \mathbf{P}_{i_1, \ldots, i_S} = \mathbf{a}_{s, i_s} \right\rbrace.
$$

The entropic regularization scheme (4.2) naturally extends to this setting

$$
\min_{\mathbf{P} \in \mathbf{U}(\mathbf{a}_s)_s} \langle \mathbf{P}, \mathbf{C} \rangle - \varepsilon \mathbf{H}(\mathbf{P}),
$$

and one can then apply Sinkhorn's algorithm to compute the optimal $\mathbf{P}$ in scaling form, where each entry indexed by a multi-index vector $i = (i_1, \ldots, i_S)$

$$
\mathbf{P}_i = \mathbf{K}_i \prod_{s=1}^S \mathbf{u}_{s, i_s} \quad \text{where} \quad \mathbf{K} \stackrel{\text{def}}{=} e^{-\frac{\mathbf{C}}{\varepsilon}},
$$

where $\mathbf{u}_s \in \mathbb{R}_+^{n_s}$ are (unknown) scaling vectors, which are iteratively updated, by cycling repeatedly through $s = 1, \ldots, S$,

$$
\mathbf{u}_{s, i_s} \leftarrow \frac{\mathbf{a}_{s, i_s}}{\sum_{\ell \neq s} \sum_{i_\ell = 1}^{n_\ell} \mathbf{K}_i \prod_{r \neq s} \mathbf{u}_{\ell, i_r}}.
$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 10.1</span><span class="math-callout__name">(General Measures)</span></p>

The discrete multimarginal problem (10.1) is generalized to measures $(\alpha_s)_s$ on spaces $(\mathcal{X}_1, \ldots, \mathcal{X}_S)$ by computing a coupling measure

$$
\min_{\pi \in \mathcal{U}(\alpha_s)_s} \int_{\mathcal{X}_1 \times \ldots \times \mathcal{X}_S} c(x_1, \ldots, x_S) \mathrm{d}\pi(x_1, \ldots, x_S),
$$

where the set of couplings is

$$
\mathcal{U}(\alpha_s)_s \stackrel{\text{def}}{=} \left\lbrace \pi \in \mathcal{M}_+^1(\mathcal{X}_1 \times \ldots \times \mathcal{X}_S) \;:\; \forall\, s = 1, \ldots, S,\; P_{s,\sharp} \pi = \alpha_s \right\rbrace
$$

where $P_s : \mathcal{X}_1 \times \ldots \times \mathcal{X}_S \to \mathcal{X}_s$ is the projection on the $s$th component, $P_s(x_1, \ldots, x_S) = x_s$. A typical application of multimarginal OT is to compute approximation of solutions to quantum chemistry problems, and in particular, in density functional theory. This problem is obtained when considering the singular Coulomb interaction cost

$$
c(x_1, \ldots, x_S) = \sum_{i \neq j} \frac{1}{\|x_i - x_j\|}.
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 10.2</span><span class="math-callout__name">(Multimarginal Formulation of the Barycenter)</span></p>

It is possible to recast the linear program optimization (9.11) as an optimization over a single coupling over $\mathcal{X}^{S+1}$ where the last marginal is the barycenter and the other ones are the input measures $(\alpha_s)_{s=1}^S$

$$
\min_{\bar{\pi} \in \mathcal{M}_+^1(X^{S+1})} \int_{\mathcal{X}^{S+1}} \sum_{s=1}^S \lambda_s c(x, x_s) \mathrm{d}\bar{\pi}(x_1, \ldots, x_s, x)
$$

$$
\text{subject to} \quad \forall\, s = 1, \ldots, S, \quad P_{s,\sharp} \bar{\pi} = \alpha_s.
$$

By explicitly minimizing in (10.4) with respect to the last marginal (associated to $x \in \mathcal{X}$), one obtains that solutions $\alpha$ of the barycenter problem (9.11) can be computed as $\alpha = A_{\lambda, \sharp} \pi$, where $A_\lambda$ is the "barycentric map" defined as

$$
A_\lambda : (x_1, \ldots, x_S) \in \mathcal{X}^S \mapsto \operatorname*{argmin}_{x \in \mathcal{X}} \sum_s \lambda_s c(x, x_s)
$$

(assuming this map is single-valued), where $\pi$ is any solution of the multimarginal problem (10.3) with cost

$$
c(x_1, \ldots, x_S) = \sum_\ell \lambda_\ell c(x_\ell, A_\lambda(x_1, \ldots, x_S)).
$$

For instance, for $c(x, y) = \|x - y\|^2$, one has, removing the constant squared terms,

$$
c(x_1, \ldots, x_S) = -\sum_{r \leq s} \lambda_r \lambda_s \langle x_r,\, x_s \rangle,
$$

which is a problem studied in Gangbo and Swiech [1998]. If all the input measures are discrete $\beta_s = \sum_{i_s = 1}^{n_s} \mathbf{a}_{s, i_s} \delta_{x_{s, i_s}}$, then the barycenter $\alpha$ is also discrete and is obtained using the formula

$$
\alpha = \sum_{(i_1, \ldots, i_S)} \mathbf{P}_{(i_1, \ldots, i_S)} \delta_{A_\lambda(x_{i_1}, \ldots, x_{i_S})},
$$

where $\mathbf{P}$ is an optimal solution of (10.1) with cost matrix $\mathbf{C}_{i_1, \ldots, i_S} = c(x_{i_1}, \ldots, x_{i_S})$. Since $\mathbf{P}$ is a nonnegative tensor of $\prod_s n_s$ dimensions obtained as the solution of a linear program with $\sum_s n_s - S + 1$ equality constraints, an optimal solution $\mathbf{P}$ with up to $\sum_s n_s - S + 1$ nonzero values can be obtained. A barycenter $\alpha$ with a support of up to $\sum_s n_s - S + 1$ points can therefore be obtained.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 10.3</span><span class="math-callout__name">(Relaxation of Euler Equations)</span></p>

A convex relaxation of Euler equations of incompressible fluid dynamics has been proposed by Brenier (1990, 1993, 1999, 2008) and Ambrosio and Figalli [2009]. Similarly to the setting exposed in $\S$7.6, it corresponds to the problem of finding a probability distribution $\bar{\pi} \in \mathcal{M}_+^1(\bar{\mathcal{X}})$ over the set $\bar{\mathcal{X}}$ of all paths $\gamma : [0, 1] \to \mathcal{X}$, which describes the movement of particles in the fluid. This is a relaxed version of the initial partial differential equation model because, as in the Kantorovich formulation of OT, mass can be split. The evolution with time does not necessarily define a diffeomorphism of the underlying space $\mathcal{X}$. The dynamic of the fluid is obtained by minimizing as in (7.17) the energy $\int_0^1 \|\gamma'(t)\|^2 \mathrm{d}t$ of each path. The difference with OT over the space of paths is the additional incompressibility of the fluid. This incompressibility is taken care of by imposing that the density of particles should be uniform at any time $t \in [0, 1]$ (and not just imposed at initial and final times $t \in \lbrace 0, 1 \rbrace$ as in classical OT). Assuming $\mathcal{X}$ is compact and denoting $\rho_\mathcal{X}$ the uniform distribution on $\mathcal{X}$, this reads $\bar{P}_{t,\sharp} \bar{\pi} = \rho_\mathcal{X}$ where $\bar{P}_t : \gamma \in \bar{\mathcal{X}} \to \gamma(t) \in \mathcal{X}$. One can discretize this problem by replacing a continuous path $(\gamma(t))_{t \in [0,1]}$ by a sequence of $S$ points $(x_{i_1}, x_{i_2}, \ldots, x_{i_S})$ on a grid $(x_k)_{k=1}^n \subset \mathcal{X}$, and $\Pi$ is represented by an $S$-way coupling $\mathbf{P} \in \mathbb{R}^{n^S} \in \mathcal{U}(\mathbf{a}_s)_s$, where the marginals are uniform $\mathbf{a}_s = n^{-1} \mathbb{1}_n$. The cost of the corresponding multimarginal problem is then

$$
\mathbf{C}_{i_1, \ldots, i_S} = \sum_{s=1}^{S-1} \|x_{i_s} - x_{i_{s+1}}\|^2 + R \|x_{\sigma(i_1)} - x_{i_S}\|^2.
$$

Here $R$ is a large enough penalization constant, which is here to enforce the movement of particles between initial and final times, which is prescribed by a permutation $\sigma : [\![n]\!] \to [\![n]\!]$. This resulting multimarginal problem is implemented efficiently in conjunction with Sinkhorn iterations (10.2) using the special structure of the cost. Indeed, in place of the $O(n^S)$ cost required to compute the denominator appearing in (10.2), one can decompose it as a succession of $S$ matrix-vector multiplications, hence with a low cost $Sn^2$.

</div>

### 10.2 Unbalanced Optimal Transport

A major bottleneck of optimal transport in its usual form is that it requires the two input measures $(\alpha, \beta)$ to have the same total mass. While many workarounds have been proposed (including renormalizing the input measure, or using dual norms such as detailed in $\S$ 8.2), it is only recently that satisfying unifying theories have been developed.

Following Liero et al. [2018], to account for arbitrary positive histograms $(\mathbf{a}, \mathbf{b}) \in \mathbb{R}_+^n \times \mathbb{R}_+^m$, the initial Kantorovich formulation (2.11) is "relaxed" by only penalizing marginal deviation using some divergence $\mathbf{D}_\varphi$, defined in (8.3). This equivalently corresponds to minimizing an OT distance between approximate measures

$$
\mathrm{L}_\mathbf{C}^\tau(\mathbf{a}, \mathbf{b}) = \min_{\tilde{\mathbf{a}}, \tilde{\mathbf{b}}} \mathrm{L}_\mathbf{C}(\mathbf{a}, \mathbf{b}) + \tau_1 \mathbf{D}_\varphi(\mathbf{a}, \tilde{\mathbf{a}}) + \tau_2 \mathbf{D}_\varphi(\mathbf{b}, \tilde{\mathbf{b}})
$$

$$
= \min_{\mathbf{P} \in \mathbb{R}_+^{n \times m}} \langle \mathbf{C}, \mathbf{P} \rangle + \tau_1 \mathbf{D}_\varphi(\mathbf{P} \mathbb{1}_m | \mathbf{a}) + \tau_2 \mathbf{D}_\varphi(\mathbf{P}^\top \mathbb{1}_m | \mathbf{b}),
$$

where $(\tau_1, \tau_2)$ controls how much mass variations are penalized as opposed to transportation of the mass. In the limit $\tau_1 = \tau_2 \to +\infty$, assuming $\sum_i \mathbf{a}_i = \sum_j \mathbf{b}_j$ (the "balanced" case), one recovers the original optimal transport formulation with hard marginal constraint (2.11).

This formalism recovers many different previous works, for instance introducing for $\mathbf{D}_\varphi$ an $\ell^2$ norm or an $\ell^1$ norm as in partial transport. A case of particular importance is when using $\mathbf{D}_\varphi = \mathbf{KL}$ the Kullback--Leibler divergence. For this cost, in the limit $\tau = \tau_1 = \tau_2 \to 0$, one obtains the so-called squared Hellinger distance

$$
\mathrm{L}_\mathbf{C}^\tau(\mathbf{a}, \mathbf{b}) \xrightarrow{\tau \to 0} \mathfrak{h}^2(\mathbf{a}, \mathbf{b}) = \sum_i (\sqrt{\mathbf{a}_i} - \sqrt{\mathbf{b}_i})^2.
$$

Sinkhorn's iterations (4.15) can be adapted to this problem by making use of the generalized algorithm detailed in $\S$4.6. The solution has the form (4.12) and the scalings are updated as

$$
\mathbf{u} \leftarrow \left( \frac{\mathbf{a}}{\mathbf{K}\mathbf{v}} \right)^{\frac{\tau_1}{\tau_1 + \varepsilon}} \quad \text{and} \quad \mathbf{v} \leftarrow \left( \frac{\mathbf{b}}{\mathbf{K}^\top \mathbf{u}} \right)^{\frac{\tau_2}{\tau_2 + \varepsilon}}.
$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 10.4</span><span class="math-callout__name">(Generic Measure)</span></p>

For $(\alpha, \beta)$ two arbitrary measures, the unbalanced version (also called "log-entropic") of (2.15) reads

$$
\mathcal{L}_c^\tau(\alpha, \beta) \stackrel{\text{def}}{=} \min_{\pi \in \mathcal{M}_+(\mathcal{X} \times \mathcal{Y})} \int_{\mathcal{X} \times \mathcal{Y}} c(x, y) \mathrm{d}\pi(x, y) + \tau \mathcal{D}_\varphi(P_{1,\sharp} \pi | \alpha) + \tau \mathcal{D}_\varphi(P_{2,\sharp} \pi | \beta),
$$

where divergences $\mathcal{D}_\varphi$ between measures are defined in (8.1). In the special case $c(x, y) = \|x - y\|^2$, $\mathcal{D}_\varphi = \mathrm{KL}$, $\mathcal{L}_c^\tau(\alpha, \beta)^{1/2}$ is the Gaussian--Hellinger distance, and it is shown to be a distance on $\mathcal{M}_+^1(\mathbb{R}^d)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 10.5</span><span class="math-callout__name">(Wasserstein--Fisher--Rao)</span></p>

For the particular choice of cost

$$
c(x, y) = -\log \cos(\min(d(x, y) / \kappa,\, \pi/2)),
$$

where $\kappa$ is some cutoff distance, and using $\mathcal{D}_\varphi = \mathrm{KL}$, then

$$
\mathrm{WFR}(\alpha, \beta) \stackrel{\text{def}}{=} \mathcal{L}_c^\tau(\alpha, \beta)^{\frac{1}{2}}
$$

is the so-called Wasserstein--Fisher--Rao or Hellinger--Kantorovich distance. In the special case $\mathcal{X} = \mathbb{R}^d$, this static (Kantorovich-like) formulation matches its dynamical counterparts (7.15). This dynamical formulation is detailed in $\S$7.4.

</div>

The barycenter problem (9.11) can be generalized to handle an unbalanced setting by replacing $\mathcal{L}_c$ with $\mathcal{L}_c^\tau$. Using finite values for $\tau$ (recall that OT is equivalent to $\tau = \infty$) is thus important to prevent irregular interpolations that arise because of mass splitting, which happens because of a "hard" mass conservation constraint. In practice, unbalanced OT techniques seem to outperform classical OT for applications (such as in imaging or machine learning) where the input data is noisy or not perfectly known.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 10.6</span><span class="math-callout__name">(Connection with Dual Norms)</span></p>

A particularly simple setup to account for mass variation is to use dual norms, as detailed in $\S$8.2. By choosing a compact set $B \subset \mathcal{C}(\mathcal{X})$ one obtains a norm defined on the whole space $\mathcal{M}(\mathcal{X})$ (in particular, the measures do not need to be positive). A particular instance of this setting is the flat norm (8.11), which is recovered as a special instance of unbalanced transport, when using $\mathcal{D}_\varphi(\alpha | \alpha') = \|\alpha - \alpha'\|_{\mathrm{TV}}$ to be the total variation norm (8.9).

</div>

### 10.3 Problems with Extra Constraints on the Couplings

Many other OT-like problems have been proposed in the literature. They typically correspond to adding extra constraints $\mathcal{C}$ on the set of feasible couplings appearing in the original OT problem (2.15)

$$
\min_{\pi \in \mathcal{U}(\alpha, \beta)} \left\lbrace \int_{\mathcal{X} \times \mathcal{Y}} c(x, y) \mathrm{d}\pi(x, y) \;:\; \pi \in \mathcal{C} \right\rbrace.
$$

The optimal transport with **capacity constraint** corresponds to imposing that the density $\rho_\pi$ (for instance, with respect to the Lebesgue measure) is upper bounded

$$
\mathcal{C} = \lbrace \pi \;:\; \rho_\pi \leq \kappa \rbrace
$$

for some $\kappa > 0$. This constraint rules out singular couplings localized on Monge maps.

The **martingale transport problem**, which finds many applications in finance, imposes the so-called martingale constraint on the conditional mean of the coupling, when $\mathcal{X} = \mathcal{Y} = \mathbb{R}^d$:

$$
\mathcal{C} = \left\lbrace \pi \;:\; \forall\, x \in \mathbb{R}^d,\; \int_{\mathbb{R}^d} y \frac{\mathrm{d}\pi(x, y)}{\mathrm{d}\alpha(x) \mathrm{d}\beta(y)} \mathrm{d}\beta(y) = x \right\rbrace.
$$

This constraint imposes that the barycentric projection map (4.20) of any admissible coupling must be equal to the identity. For arbitrary $(\alpha, \beta)$, this set $\mathcal{C}$ is typically empty, but necessary and sufficient conditions exist ($\alpha$ and $\beta$ should be in "convex order") to ensure $\mathcal{C} \neq \emptyset$ so that $(\alpha, \beta)$ satisfy a martingale constraint. Using an entropic penalization as in (4.9), one can solve approximately (10.10) using the Dykstra algorithm, which is a generalization of Sinkhorn's algorithm shown in $\S$4.2.

### 10.4 Sliced Wasserstein Distance and Barycenters

One can define a distance between two measures $(\alpha, \beta)$ defined on $\mathbb{R}^d$ by aggregating 1-D Wasserstein distances between their projections onto all directions of the sphere. This defines

$$
\mathrm{SW}(\alpha, \beta)^2 \stackrel{\text{def}}{=} \int_{\mathbf{S}^d} \mathcal{W}_2(P_{\theta, \sharp} \alpha, P_{\theta, \sharp} \beta)^2 \mathrm{d}\theta,
$$

where $\mathbf{S}^d = \lbrace \theta \in \mathbb{R}^d : \|\theta\| = 1 \rbrace$ is the $d$-dimensional sphere, and $P_\theta : x \in \mathbb{R}^d \to \mathbb{R}$ is the projection. It is related to the problem of Radon inversion over measure spaces.

**Lagrangian discretization and stochastic gradient descent.** The advantage of this functional is that 1-D Wasserstein distances are simple to compute, as detailed in $\S$2.6. In the specific case where $m = n$ and

$$
\alpha = \frac{1}{n} \sum_{i=1}^n \delta_{x_i} \quad \text{and} \quad \beta = \frac{1}{n} \sum_{i=1}^m \delta_{y_i},
$$

this is achieved by simply sorting points

$$
\mathrm{SW}(\alpha, \beta)^2 = \int_{\mathbf{S}^d} \left( \sum_{i=1}^n |\langle x_{\sigma_\theta(i)} - y_{\kappa_\theta(i)},\, \theta \rangle|^2 \right) \mathrm{d}\theta,
$$

where $\sigma_\theta, \kappa_\theta \in \mathrm{Perm}(n)$ are the permutation ordering in increasing order, respectively, $(\langle x_i, \theta \rangle)_i$ and $(\langle y_i, \theta \rangle)_i$.

Fixing the vector $y$, the function $\mathcal{E}_\beta(x) \stackrel{\text{def}}{=} \mathrm{SW}(\alpha, \beta)^2$ is smooth, and one can use this function to define a mapping by gradient descent

$$
x \leftarrow x - \tau \nabla \mathcal{E}_\beta(x) \quad \text{where}
$$

$$
\nabla \mathcal{E}_\beta(x)_i = 2 \int_{\mathbf{S}^d} \left( \langle x_i - y_{\kappa_\theta \circ \sigma_\theta^{-1}(i)},\, \theta \rangle \right) \theta\, \mathrm{d}\theta
$$

using a small enough step size $\tau > 0$. To make the method tractable, one can use a stochastic gradient descent (SGD), replacing this integral with a discrete sum against randomly drawn directions $\theta \in \mathbf{S}^d$. The flow (10.15) can be understood as (Lagrangian implementation of) a Wasserstein gradient flow (in the sense of $\S$9.3) of the function $\alpha \mapsto \mathrm{SW}(\alpha, \beta)^2$. Numerically, one finds that this flow has no local minimizer and that it thus converges to $\alpha = \beta$. At convergence, it defines a matching (similar to a Monge map) between the two distributions.

It is simple to extend this Lagrangian scheme to compute approximate "sliced" barycenters of measures, by mimicking the Frechet definition of Wasserstein barycenters (9.11) and minimizing

$$
\min_{\alpha \in \mathcal{M}_+^1(\mathcal{X})} \sum_{s=1}^S \lambda_s \mathrm{SW}(\alpha, \beta_s)^2,
$$

given a set $(\beta_s)_{s=1}^S$ of fixed input measures. Using a Lagrangian discretization of the form (10.14) for both $\alpha$ and the $(\beta_s)_s$, one can perform the nonconvex minimization over the position $x = (x_i)_i$

$$
\min_x \mathcal{E}(x) \stackrel{\text{def}}{=} \sum_s \lambda_s \mathcal{E}_{\beta_s}(x), \quad \text{and} \quad \nabla \mathcal{E}(x) = \sum_s \lambda_s \nabla \mathcal{E}_{\beta_s}(x),
$$

by gradient descent using formula (10.15) to compute $\nabla \mathcal{E}_{\beta_s}(x)$ (coupled with a random sampling of the direction $\theta$).

**Eulerian discretization and Radon transform.** A related way to compute an approximated sliced barycenter, without resorting to an iterative minimization scheme, is to use the fact that (10.13) computes a distance between the Radon transforms $\mathcal{R}(\alpha)$ and $\mathcal{R}(\beta)$ where

$$
\mathcal{R}(\alpha) \stackrel{\text{def}}{=} (P_{\theta, \sharp} \alpha)_{\theta \in \mathbf{S}^d}.
$$

A crucial point is that the Radon transform is invertible and that its inverse can be computed using a filtered backprojection formula. Given a collection of measures $\rho = (\rho_\theta)_{\theta \in \mathbf{S}^d}$, one defines the filtered backprojection operator as

$$
\mathcal{R}^+(\rho) = C_d \Delta^{\frac{d-1}{2}} \mathcal{B}(\rho),
$$

where $\xi = \mathcal{B}(\rho) \in \mathcal{M}(\mathbb{R}^d)$ is the measure defined through the relation

$$
\forall\, g \in \mathcal{C}(\mathbb{R}^d), \quad \int_{\mathbb{R}^d} g(x) \mathrm{d}\xi(x) = \int_{\mathbf{S}^d} \int_{\mathbb{R}^{d-1}} \int_\mathbb{R} g(r\theta + U_\theta z) \mathrm{d}\rho_\theta(r) \mathrm{d}z \mathrm{d}\theta,
$$

where $U_\theta$ is any orthogonal basis of $\theta^\perp$, and where $C_d \in \mathbb{R}$ is a normalizing constant which depends on the dimension. Here $\Delta^{\frac{d-1}{2}}$ is a fractional Laplacian, which is the high-pass filter defined over the Fourier domain as $\hat{\Delta}^{\frac{d-1}{2}}(\omega) = \|\omega\|^{d-1}$. One then has the left-inverse relation $\mathcal{R}^+ \circ \mathcal{R} = \mathbb{I}_{\mathcal{M}(\mathbb{R}^d)}$, so that $\mathcal{R}^+$ is a valid reconstruction formula.

In order to compute barycenters of input densities, it makes sense to replace formula (9.11) by its equivalent using Radon transform, and thus consider independently for each $\theta$ the 1-D barycenter problem

$$
\rho_\theta^\star \in \operatorname*{argmin}_{(\rho_\theta \in \mathcal{M}_+^1(\mathbb{R}))} \sum_{s=1}^S \lambda_s \mathcal{W}_2(\rho_\theta, P_{\theta, \sharp} \beta_s)^2.
$$

Each 1-D barycenter problem is easily computed using the monotone rearrangement as detailed in Remark 9.6. The Radon approximation $\alpha_R \stackrel{\text{def}}{=} \mathcal{R}^+(\rho^\star)$ of a sliced barycenter solving (9.11) is then obtained by the inverse Radon transform $\mathcal{R}^+$. Note that in general, $\alpha_R$ is not a solution to (9.11) because the Radon transform is not surjective, so that $\rho^\star$, which is obtained as a barycenter of the Radon transforms $\mathcal{R}(\beta_s)$, does not necessarily belong to the range of $\mathcal{R}$. But numerically it seems in practice to be almost the case.

**Sliced Wasserstein kernels.** Beside its computational simplicity, another advantage of the sliced Wasserstein distance is that it is isometric to a Euclidean distance (it is thus a "Hilbertian" metric), as detailed in Remark 2.30, and in particular formula (2.36). This should be contrasted with the Wasserstein distance $\mathcal{W}_2$ on $\mathbb{R}^d$, which is not Hilbertian in dimension $d \geq 2$. It is thus possible to use this sliced distance to equip the space of distributions $\mathcal{M}_+^1(\mathbb{R}^d)$ with a reproducing kernel Hilbert space structure (as detailed in $\S$8.3). One can, for instance, use the exponential and energy distance kernels

$$
k(\alpha, \beta) = e^{-\frac{\mathrm{SW}(\alpha, \beta)^p}{2\sigma^p}} \quad \text{and} \quad k(\alpha, \beta) = -\mathrm{SW}(\alpha, \beta)^p
$$

for $1 \leq p \leq 2$ for the exponential kernels and $0 < p < 2$ for the energy distance kernels. For any collection $(\alpha_i)_i$ of input measures, the matrix $(k(\alpha_i, \alpha_j))_{i,j}$ is symmetric positive semidefinite. It is possible to use these kernels to perform a variety of machine learning tasks using the "kernel trick," for instance, in regression, classification (SVM and logistic), clustering (K-means) and dimensionality reduction (PCA).

### 10.5 Transporting Vectors and Matrices

Real-valued measures $\alpha \in \mathcal{M}(\mathcal{X})$ are easily generalized to vector-valued measures $\alpha \in \mathcal{M}(\mathcal{X}; \mathbb{V})$, where $\mathbb{V}$ is some vector space. For notational simplicity, we assume $\mathbb{V}$ is Euclidean and equipped with some inner product $\langle \cdot, \cdot \rangle$ (typically $\mathbb{V} = \mathbb{R}^d$ and the inner product is the canonical one). Thanks to this inner product, vector-valued measures are identified with the dual of continuous functions $g : \mathcal{X} \to \mathbb{V}$, i.e. for any such $g$, one defines its integration against the measure as

$$
\int_\mathcal{X} g(x) \mathrm{d}\alpha(x) \in \mathbb{R},
$$

which is a linear operation on $g$ and $\alpha$. A discrete measure has the form $\alpha = \sum_i \mathbf{a}_i \delta_{x_i}$ where $(x_i, a_i) \in \mathcal{X} \times \mathbb{V}$ and the integration formula (10.21) simply reads

$$
\int_\mathcal{X} g(x) \mathrm{d}\alpha(x) = \sum_i \langle \mathbf{a}_i,\, g(x_i) \rangle \in \mathbb{R}.
$$

Equivalently, if $\mathbb{V} = \mathbb{R}^d$, then such an $\alpha$ can be viewed as a collection $(\alpha_s)_{s=1}^d$ of $d$ "classical" real-valued measures (its coordinates), writing

$$
\int_\mathcal{X} g(x) \mathrm{d}\alpha(x) = \sum_{s=1}^d \int_\mathcal{X} g_s(x) \mathrm{d}\alpha_s(x),
$$

where $g(x) = (g_s(x))_{s=1}^d$ are the coordinates of $g$ in the canonical basis.

**Dual norms.** It is nontrivial, and in fact in general impossible, to extend OT distances to such a general setting. Even coping with real-valued measures taking both positive and negative values is difficult. The only simple option is to consider dual norms, as defined in $\S$8.2. Indeed, formula (6.3) readily extends to $\mathcal{M}(\mathcal{X}; \mathbb{V})$ by considering $B$ to be a subset of $\mathcal{C}(\mathcal{X}; \mathbb{V})$. So in particular, $\mathcal{W}_1$, the flat norm and MMD norms can be computed for vector-valued measures.

**OT over cone-valued measures.** It is possible to define more advanced OT distances when $\alpha$ is restricted to be in a subset $\mathcal{M}(\mathcal{X}; \mathcal{V}) \subset \mathcal{M}(\mathcal{X}; \mathbb{V})$. The set $\mathcal{V}$ should be a positively 1-homogeneous convex cone of $\mathbb{V}$

$$
\mathcal{V} \stackrel{\text{def}}{=} \left\lbrace \lambda u \;:\; \lambda \in \mathbb{R}^+, u \in \mathcal{V}_0 \right\rbrace
$$

where $\mathcal{V}_0$ is a compact convex set. A typical example is the set of positive measures where $\mathcal{V} = \mathbb{R}_+^d$. Another important example is the set of positive symmetric matrices $\mathcal{V} = \mathcal{S}_+^d \subset \mathbb{R}^{d \times d}$.

**OT over positive matrices.** A related but quite different setting is to replace discrete measures, i.e. histograms $\mathbf{a} \in \Sigma_n$, by positive matrices with unit trace $A \in \mathcal{S}_n^+$ such that $\mathrm{tr}(A) = 1$. The rationale is that the eigenvalues $\lambda(A) \in \Sigma_n$ of $A$ play the role of a histogram, but one also has to take care of the rotations of the eigenvectors, so that this problem is more complicated.

One can extend several divergences introduced in $\S$8.1 to this setting. For instance, the Bures metric (2.42) is a generalization of the Hellinger distance (defined in Remark 8.3), since they are equal on positive diagonal matrices. One can also extend the Kullback--Leibler divergence (4.6) (see also Remark 8.1), which is generalized to positive matrices as

$$
\mathbf{KL}(A | B) \stackrel{\text{def}}{=} \mathrm{tr}\left( P \log(P) - P \log(Q) - P + Q \right)
$$

where $\log(\cdot)$ is the matrix logarithm. This matrix $\mathbf{KL}$ is convex with both of its arguments.

It is possible to solve convex dynamic formulations to define OT distances between such matrices. There also exists an equivalent of Sinkhorn's algorithm, which is due to Gurvits [2004] and has been extensively studied. It is known to converge only in some cases but seems empirically to always work.

### 10.6 Gromov--Wasserstein Distances

For some applications such as shape matching, an important weakness of optimal transport distances lies in the fact that they are not invariant to important families of invariances, such as rescaling, translation or rotations. Although some nonconvex variants of OT to handle such global transformations have been proposed, these methods require specifying first a subset of invariances, possibly between different metric spaces, to be relevant. We describe in this section a more general and very natural extension of OT that can deal with measures defined on different spaces without requiring the definition of a family of invariances.

#### 10.6.1 Hausdorff Distance

The Hausdorff distance between two sets $A, B \subset \mathcal{Z}$ for some metric $d_\mathcal{Z}$ is

$$
\mathcal{H}_\mathcal{Z}(A, B) \stackrel{\text{def}}{=} \max \left( \sup_{a \in A} \inf_{b \in B} d_\mathcal{Z}(a, b),\, \sup_{b \in B} \inf_{a \in A} d_\mathcal{Z}(a, b) \right),
$$

see Figure 10.5. This defines a distance between compact sets $\mathcal{K}(\mathcal{Z})$ of $\mathcal{Z}$, and if $\mathcal{Z}$ is compact, then $(\mathcal{K}(\mathcal{Z}), \mathcal{H}_\mathcal{Z})$ is itself compact.

This distance between sets $(A, B)$ can be defined similarly to the Wasserstein distance between measures (which should be somehow understood as "weighted" sets). One replaces the measures couplings (2.14) by sets couplings

$$
\mathcal{R}(A, B) \stackrel{\text{def}}{=} \left\lbrace R \in \mathcal{X} \times \mathcal{Y} \;:\; \begin{array}{l} \forall\, a \in A, \exists\, b \in B, (a, b) \in R \\ \forall\, b \in B, \exists\, a \in A, (a, b) \in R \end{array} \right\rbrace.
$$

One should replace integration (since one does not have access to measures) by maximization, and one has

$$
\mathcal{H}_\mathcal{Z}(A, B) = \inf_{R \in \mathcal{R}(A, B)} \sup_{(a, b) \in R} d(a, b).
$$

Note that the support of a measure coupling $\pi \in \mathcal{U}(\alpha, \beta)$ is a set coupling between the supports, i.e. $\mathrm{Supp}(\pi) \in \mathcal{R}(\mathrm{Supp}(\alpha), \mathrm{Supp}(\beta))$. The Hausdorff distance is thus connected to the $\infty$-Wasserstein distance (see Remark 2.20) and one has $\mathcal{H}(A, B) \leq \mathcal{W}_\infty(\alpha, \beta)$ for any measure $(\alpha, \beta)$ whose supports are $(A, B)$.

#### 10.6.2 Gromov--Hausdorff Distance

The Gromov--Hausdorff (GH) distance is a way to measure the distance between two metric spaces $(\mathcal{X}, d_\mathcal{X}), (\mathcal{Y}, d_\mathcal{Y})$ by quantifying how far they are from being isometric to each other. It is defined as the minimum Hausdorff distance between every possible isometric embedding of the two spaces in a third one,

$$
\mathcal{GH}(d_\mathcal{X}, d_\mathcal{Y}) \stackrel{\text{def}}{=} \inf_{\mathcal{Z}, f, g} \left\lbrace \mathcal{H}_\mathcal{Z}(f(\mathcal{X}), g(\mathcal{Y})) \;:\; \begin{array}{l} f : \mathcal{X} \xrightarrow{\mathrm{isom}} \mathcal{Z} \\ g : \mathcal{Y} \xrightarrow{\mathrm{isom}} \mathcal{Z} \end{array} \right\rbrace.
$$

Here, the constraint is that $f$ must be an isometric embedding, meaning that $d_\mathcal{Z}(f(x), f(x')) = d_\mathcal{X}(x, x')$ for any $(x, x') \in \mathcal{X}^2$ (similarly for $g$). One can show that $\mathcal{GH}$ defines a distance between compact metric spaces up to isometries, so that in particular $\mathcal{GH}(d_\mathcal{X}, d_\mathcal{Y}) = 0$ if and only if there exists an isometry $h : \mathcal{X} \to \mathcal{Y}$, i.e. $h$ is bijective and $d_\mathcal{Y}(h(x), h(x')) = d_\mathcal{X}(x, x')$ for any $(x, x') \in \mathcal{X}^2$.

Similarly to (10.23), it is possible to rewrite equivalently the GH distance using couplings as follows:

$$
\mathcal{GH}(d_\mathcal{X}, d_\mathcal{Y}) = \frac{1}{2} \inf_{R \in \mathcal{R}(\mathcal{X}, \mathcal{Y})} \sup_{((x, y), (x', y')) \in R^2} |d_\mathcal{X}(x, x') - d_\mathcal{X}(y, y')|.
$$

For discrete spaces $\mathcal{X} = (x_i)_{i=1}^n, \mathcal{Y} = (y_j)_{j=1}^m$ represented using a distance matrix $\mathbf{D} = (d_\mathcal{X}(x_i, x_{i'}))_{i,i'} \in \mathbb{R}^{n \times n}$, $\mathbf{D}' = (d_\mathcal{Y}(y_j, y_{j'}))_{j,j'} \in \mathbb{R}^{m \times m}$, one can rewrite this optimization using binary matrices $\mathbf{R} \in \lbrace 0, 1 \rbrace^{n \times m}$ indicating the support of the set couplings $R$ as follows:

$$
\mathrm{GH}(\mathbf{D}, \mathbf{D}') = \frac{1}{2} \inf_{\mathbf{R}\mathbb{1} > 0, \mathbf{R}^\top \mathbb{1} > 0} \max_{(i, i', j, j')} \mathbf{R}_{i,j} \mathbf{R}_{j,j'} |\mathbf{D}_{i,i'} - \mathbf{D}'_{j,j'}|.
$$

The initial motivation of the GH distance is to define and study limits of metric spaces.

#### 10.6.3 Gromov--Wasserstein Distance

Optimal transport needs a ground cost $\mathbf{C}$ to compare histograms $(\mathbf{a}, \mathbf{b})$ and thus cannot be used if the bins of those histograms are not defined on the same underlying space, or if one cannot preregister these spaces to define a ground cost between any pair of bins in the first and second histograms, respectively. To address this limitation, one can instead only assume a weaker assumption, namely that two matrices $\mathbf{D} \in \mathbb{R}^{n \times n}$ and $\mathbf{D}' \in \mathbb{R}^{m \times m}$ quantify similarity relationships between the points on which the histograms are defined. A typical scenario is when these matrices are (power of) distance matrices. The GW problem reads

$$
\mathrm{GW}((\mathbf{a}, \mathbf{D}), (\mathbf{b}, \mathbf{D}'))^2 \stackrel{\text{def}}{=} \min_{\mathbf{P} \in \mathbf{U}(\mathbf{a}, \mathbf{b})} \mathcal{E}_{\mathbf{D}, \mathbf{D}'}(\mathbf{P})
$$

where $\quad \mathcal{E}_{\mathbf{D}, \mathbf{D}'}(\mathbf{P}) \stackrel{\text{def}}{=} \sum_{i, j, i', j'} |\mathbf{D}_{i,i'} - \mathbf{D}'_{j,j'}|^2 \mathbf{P}_{i,j} \mathbf{P}_{i',j'},$

see Figure 10.8. This problem is similar to the GH problem (10.24) when replacing maximization by a sum and set couplings by measure couplings. This is a nonconvex problem, which can be recast as a quadratic assignment problem and is in full generality NP-hard to solve for arbitrary inputs. It is in fact equivalent to a graph matching problem.

One can show that GW satisfies the triangular inequality, and in fact it defines a distance between metric spaces equipped with a probability distribution, here assumed to be discrete in definition (10.25), up to isometries preserving the measures.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 10.7</span><span class="math-callout__name">(Gromov--Wasserstein Distance)</span></p>

The general setting corresponds to computing couplings between metric measure spaces $(\mathcal{X}, d_\mathcal{X}, \alpha_\mathcal{X})$ and $(\mathcal{Y}, d_\mathcal{Y}, \alpha_\mathcal{Y})$, where $(d_\mathcal{X}, d_\mathcal{Y})$ are distances, while $\alpha_\mathcal{X}$ and $\alpha_\mathcal{Y}$ are measures on their respective spaces. One defines

$$
\mathcal{GW}((\alpha_\mathcal{X}, d_\mathcal{X}), (\alpha_\mathcal{Y}, d_\mathcal{Y}))^2 \stackrel{\text{def}}{=} \min_{\pi \in \mathcal{U}(\alpha_\mathcal{X}, \alpha_\mathcal{Y})} \int_{\mathcal{X}^2 \times \mathcal{Y}^2} |d_\mathcal{X}(x, x') - d_\mathcal{Y}(y, y')|^2 \mathrm{d}\pi(x, y) \mathrm{d}\pi(x', y').
$$

$\mathcal{GW}$ defines a distance between metric measure spaces up to isometries, where one says that $(\mathcal{X}, \alpha_\mathcal{X}, d_\mathcal{X})$ and $(\mathcal{Y}, \alpha_\mathcal{Y}, d_\mathcal{Y})$ are isometric if there exists a bijection $\varphi : \mathcal{X} \to \mathcal{Y}$ such that $\varphi_\sharp \alpha_\mathcal{X} = \alpha_\mathcal{Y}$ and $d_\mathcal{Y}(\varphi(x), \varphi(x')) = d_\mathcal{X}(x, x')$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 10.8</span><span class="math-callout__name">(Gromov--Wasserstein Geodesics)</span></p>

The space of metric spaces (up to isometries) endowed with this $\mathcal{GW}$ distance (10.26) has a geodesic structure. Sturm [2012] shows that the geodesic between $(\mathcal{X}_0, d_{\mathcal{X}_0}, \alpha_0)$ and $(\mathcal{X}_1, d_{\mathcal{X}_1}, \alpha_1)$ can be chosen to be $t \in [0, 1] \mapsto (\mathcal{X}_0 \times \mathcal{X}_1, d_t, \pi^\star)$, where $\pi^\star$ is a solution of (10.26) and for all $((x_0, x_1), (x_0', x_1')) \in (\mathcal{X}_0 \times \mathcal{X}_1)^2$,

$$
d_t((x_0, x_1), (x_0', x_1')) \stackrel{\text{def}}{=} (1 - t) d_{\mathcal{X}_0}(x_0, x_0') + t d_{\mathcal{X}_1}(x_1, x_1').
$$

This formula allows one to define and analyze gradient flows which minimize functionals involving metric spaces; see Sturm [2012]. It is, however, difficult to handle numerically, because it involves computations over the product space $\mathcal{X}_0 \times \mathcal{X}_1$.

</div>

#### 10.6.4 Entropic Regularization

To approximate the computation of GW, and to help convergence of minimization schemes to better minima, one can consider the entropic regularized variant

$$
\min_{\mathbf{P} \in \mathbf{U}(\mathbf{a}, \mathbf{b})} \mathcal{E}_{\mathbf{D}, \mathbf{D}'}(\mathbf{P}) - \varepsilon \mathbf{H}(\mathbf{P}).
$$

One can use iteratively Sinkhorn's algorithm to progressively compute a stationary point of (10.27). Indeed, successive linearizations of the objective function lead to consider the succession of updates

$$
\mathbf{P}^{(\ell+1)} \stackrel{\text{def}}{=} \min_{\mathbf{P} \in \mathbf{U}(\mathbf{a}, \mathbf{b})} \langle \mathbf{P}, \mathbf{C}^{(\ell)} \rangle - \varepsilon H(\mathbf{P}) \quad \text{where}
$$

$$
\mathbf{C}^{(\ell)} \stackrel{\text{def}}{=} \nabla \mathcal{E}_{\mathbf{D}, \mathbf{D}'}(\mathbf{P}^{(\ell)}) = -\mathbf{D} \mathbf{P}^{(\ell)} \mathbf{D}',
$$

which can be interpreted as a mirror-descent scheme. Each update can thus be solved using Sinkhorn iterations (4.15) with cost $\mathbf{C}^{(\ell)}$.
