---
layout: default
title: Basic Category Theory
date: 2024-11-01
# excerpt: Connecting differential equations, stability analysis, and attractor theory to the training dynamics of modern machine learning models.
# tags:
#   - dynamical-systems
#   - machine-learning
#   - theory
---

# Basic Category Theory

## Introduction

Category theory takes a bird's-eye view of mathematics. From high in the sky, details become invisible, but we can spot patterns that were impossible to detect from ground level. The most important concept in this book is that of **universal property**. The further you go in mathematics, especially pure mathematics, the more universal properties you will meet.

### Universal Properties

Phrases of the form "there exists a unique such-and-such satisfying some condition" are common in category theory. To prove the existence part, we show there is at least one. To prove the uniqueness part, we show there is at most one; in other words, any two such-and-suches satisfying the condition are equal.

Properties such as these are called "universal" because they state how the object being described relates to the *entire* universe in which it lives.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 0.1</span><span class="math-callout__name">(Terminal set)</span></p>

Let $1$ denote a set with one element. Then $1$ has the following property: for all sets $X$, there exists a unique map from $X$ to $1$.

Indeed, there *exists* a map $X \to 1$ (send every element to the single element of $1$), and it is *unique* (any map $X \to 1$ must send each element of $X$ to the single element of $1$).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 0.2</span><span class="math-callout__name">(Initial ring)</span></p>

The ring $\mathbb{Z}$ has the following property: for all rings $R$, there exists a unique homomorphism $\mathbb{Z} \to R$.

**Existence:** Define $\phi \colon \mathbb{Z} \to R$ by

$$
\phi(n) = \begin{cases} \underbrace{1 + \cdots + 1}_{n} & \text{if } n > 0, \\ 0 & \text{if } n = 0, \\ -\phi(-n) & \text{if } n < 0. \end{cases}
$$

**Uniqueness:** Any ring homomorphism $\psi \colon \mathbb{Z} \to R$ must satisfy $\psi(1) = 1$, and since homomorphisms preserve addition, $\psi(n) = \underbrace{\psi(1) + \cdots + \psi(1)}\_{n} = \underbrace{1 + \cdots + 1}\_{n} = \phi(n)$ for all $n > 0$. Similarly for $n \le 0$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 0.3</span><span class="math-callout__name">(Uniqueness of initial rings)</span></p>

*Let $A$ be a ring with the following property: for all rings $R$, there exists a unique homomorphism $A \to R$. Then $A \cong \mathbb{Z}$.*

**Proof.** Call a ring with this property "initial". Since $A$ is initial, there is a unique homomorphism $\phi \colon A \to \mathbb{Z}$. Since $\mathbb{Z}$ is initial, there is a unique homomorphism $\phi' \colon \mathbb{Z} \to A$. Now $\phi' \circ \phi \colon A \to A$ is a homomorphism, but so is $1_A$; by the uniqueness part of initiality of $A$, we have $\phi' \circ \phi = 1_A$. Similarly, $\phi \circ \phi' = 1_{\mathbb{Z}}$. So $\phi$ and $\phi'$ are mutually inverse, giving $A \cong \mathbb{Z}$. $\square$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 0.4</span><span class="math-callout__name">(Basis of a vector space)</span></p>

Let $V$ be a vector space with a basis $(v_s)_{s \in S}$. Define a function $i \colon S \to V$ by $i(s) = v_s$. Then $V$ together with $i$ has the following universal property: for all vector spaces $W$ and all functions $f \colon S \to W$, there exists a unique linear map $\bar{f} \colon V \to W$ such that $\bar{f} \circ i = f$.

In other words, the function $\lbrace \text{linear maps } V \to W \rbrace \to \lbrace \text{functions } S \to W \rbrace$ given by $\bar{f} \mapsto \bar{f} \circ i$ is bijective.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 0.5</span><span class="math-callout__name">(Discrete topological space)</span></p>

Given a set $S$, we can build a topological space $D(S)$ by equipping $S$ with the **discrete topology**: all subsets are open. With this topology, *any* map from $S$ to a space $X$ is continuous.

Define $i \colon S \to D(S)$ by $i(s) = s$. Then $D(S)$ together with $i$ has the universal property: for all topological spaces $X$ and all functions $f \colon S \to X$, there exists a unique continuous map $\bar{f} \colon D(S) \to X$ such that $\bar{f} \circ i = f$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 0.6</span><span class="math-callout__name">(Tensor product)</span></p>

Given vector spaces $U$, $V$ and $W$, a **bilinear map** $f \colon U \times V \to W$ is a function that is linear in each variable:

$$
f(u, v_1 + \lambda v_2) = f(u, v_1) + \lambda f(u, v_2), \quad f(u_1 + \lambda u_2, v) = f(u_1, v) + \lambda f(u_2, v).
$$

There exist a vector space $T$ and a bilinear map $b \colon U \times V \to T$ with the following universal property: for all vector spaces $W$ and all bilinear maps $f \colon U \times V \to W$, there exists a unique linear map $\bar{f} \colon T \to W$ such that $\bar{f} \circ b = f$.

The vector space $T$ is called the **tensor product** of $U$ and $V$, written $U \otimes V$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 0.7</span><span class="math-callout__name">(Uniqueness of tensor products)</span></p>

*Let $U$ and $V$ be vector spaces. Suppose that $b \colon U \times V \to T$ and $b' \colon U \times V \to T'$ are both universal bilinear maps out of $U \times V$. Then $T \cong T'$. More precisely, there exists a unique isomorphism $j \colon T \to T'$ such that $j \circ b = b'$.*

**Proof.** Using the universality of $b$, take the bilinear map $b'$ to obtain a unique linear map $j \colon T \to T'$ satisfying $j \circ b = b'$. Similarly, using universality of $b'$, obtain $j' \colon T' \to T$ satisfying $j' \circ b' = b$.

Now $j' \circ j \colon T \to T$ satisfies $(j' \circ j) \circ b = b$; but the identity $1_T$ also satisfies $1_T \circ b = b$. By the uniqueness part of the universal property of $b$, we have $j' \circ j = 1_T$. Similarly, $j \circ j' = 1_{T'}$. So $j$ is an isomorphism. $\square$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 0.8</span><span class="math-callout__name">(Kernel of a group homomorphism)</span></p>

Let $\theta \colon G \to H$ be a homomorphism of groups. Associated with $\theta$ is the diagram

$$
\ker(\theta) \xhookrightarrow{\iota} G \underset{\varepsilon}{\overset{\theta}{\rightrightarrows}} H,
$$

where $\iota$ is the inclusion of $\ker(\theta)$ into $G$ and $\varepsilon$ is the trivial homomorphism ($\varepsilon(g) = 1$ for all $g \in G$). The map $\iota$ satisfies $\theta \circ \iota = \varepsilon \circ \iota$, and is universal as such.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 0.9</span><span class="math-callout__name">(Pushout in topological spaces)</span></p>

Take a topological space covered by two open subsets: $X = U \cup V$. The diagram of inclusion maps

$$
U \cap V \xhookrightarrow{i} U, \quad U \cap V \xhookrightarrow{j} V, \quad U \xhookrightarrow{i'} X, \quad V \xhookrightarrow{j'} X
$$

has a universal property: given $Y$, $f$ and $g$ such that $f \circ i = g \circ j$, there is exactly one continuous map $h \colon X \to Y$ such that $h \circ j' = f$ and $h \circ i' = g$.

Under favourable conditions, the induced diagram of fundamental groups satisfies the same property in the world of groups and group homomorphisms. This is **van Kampen's theorem**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Approaches to universal properties)</span></p>

As the book progresses, we will develop different ways of talking about universal properties. Once we have set up the basic vocabulary of categories and functors, we will study **adjoint functors**, then **representable functors**, then **limits**. Each provides an approach to universal properties, placing the idea in a different light:

* Examples 0.4 and 0.5 are most readily described via adjoint functors.
* Example 0.6 via representable functors.
* Examples 0.1, 0.2, 0.8 and 0.9 in terms of limits.

</div>

---

## Chapter 1: Categories, Functors and Natural Transformations

A category is a system of related objects. The objects do not live in isolation: there is some notion of map between objects, binding them together. Categories are *themselves* mathematical objects, so there is a notion of "map between categories" called **functors**. Maps between functors are called **natural transformations**.

### 1.1 Categories

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1.1.1</span><span class="math-callout__name">(Category)</span></p>

A **category** $\mathscr{A}$ consists of:

* a collection $\text{ob}(\mathscr{A})$ of **objects**;
* for each $A, B \in \text{ob}(\mathscr{A})$, a collection $\mathscr{A}(A, B)$ of **maps** (or **arrows** or **morphisms**) from $A$ to $B$;
* for each $A, B, C \in \text{ob}(\mathscr{A})$, a function

$$
\mathscr{A}(B, C) \times \mathscr{A}(A, B) \to \mathscr{A}(A, C), \quad (g, f) \mapsto g \circ f,
$$

  called **composition**;
* for each $A \in \text{ob}(\mathscr{A})$, an element $1_A$ of $\mathscr{A}(A, A)$, called the **identity** on $A$,

satisfying the following axioms:

* **associativity**: for each $f \in \mathscr{A}(A, B)$, $g \in \mathscr{A}(B, C)$ and $h \in \mathscr{A}(C, D)$, we have $(h \circ g) \circ f = h \circ (g \circ f)$;
* **identity laws**: for each $f \in \mathscr{A}(A, B)$, we have $f \circ 1_A = f = 1_B \circ f$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 1.1.2</span><span class="math-callout__name">(Notation and conventions)</span></p>

**(a)** We often write $A \in \mathscr{A}$ to mean $A \in \text{ob}(\mathscr{A})$; $f \colon A \to B$ or $A \xrightarrow{f} B$ to mean $f \in \mathscr{A}(A, B)$; and $gf$ to mean $g \circ f$. People also write $\mathscr{A}(A, B)$ as $\text{Hom}\_{\mathscr{A}}(A, B)$ or $\text{Hom}(A, B)$.

**(b)** The definition ensures that from each string $A_0 \xrightarrow{f_1} A_1 \xrightarrow{f_2} \cdots \xrightarrow{f_n} A_n$ of maps, it is possible to construct exactly one map $A_0 \to A_n$ (namely $f_n f_{n-1} \cdots f_1$). An identity map can be thought of as the composite of zero maps.

**(c)** We often speak of **commutative diagrams**: a diagram is said to commute if whenever there are two paths from an object $X$ to an object $Y$, the map obtained by composing along one path is equal to the map obtained by composing along the other.

**(d)** The word "collection" means roughly "set", though it is better to interpret it as "class". We return to this in Chapter 3.

**(e)** If $f \in \mathscr{A}(A, B)$, we call $A$ the **domain** and $B$ the **codomain** of $f$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Examples 1.1.3</span><span class="math-callout__name">(Categories of mathematical structures)</span></p>

**(a)** The category **Set**: objects are sets, maps are functions, composition and identities are the usual ones.

**(b)** The category **Grp**: objects are groups, maps are group homomorphisms.

**(c)** The category **Ring**: objects are rings, maps are ring homomorphisms.

**(d)** For each field $k$, the category $\mathbf{Vect}_k$: objects are vector spaces over $k$, maps are linear maps.

**(e)** The category **Top**: objects are topological spaces, maps are continuous maps.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1.1.4</span><span class="math-callout__name">(Isomorphism)</span></p>

A map $f \colon A \to B$ in a category $\mathscr{A}$ is an **isomorphism** if there exists a map $g \colon B \to A$ in $\mathscr{A}$ such that $gf = 1_A$ and $fg = 1_B$.

We call $g$ the **inverse** of $f$ and write $g = f^{-1}$. If there exists an isomorphism from $A$ to $B$, we say $A$ and $B$ are **isomorphic** and write $A \cong B$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.1.5</span><span class="math-callout__name">(Isomorphisms in Set)</span></p>

The isomorphisms in **Set** are exactly the bijections. This amounts to the assertion that a function has a two-sided inverse if and only if it is injective and surjective.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.1.7</span><span class="math-callout__name">(Isomorphisms in Top)</span></p>

The isomorphisms in **Top** are exactly the homeomorphisms. A bijective map in **Top** is *not* necessarily an isomorphism: for example, the map $[0, 1) \to \lbrace z \in \mathbb{C} \mid \lvert z \rvert = 1 \rbrace$ given by $t \mapsto e^{2\pi i t}$ is a continuous bijection but not a homeomorphism.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Objects vs. sets)</span></p>

In general, the objects of a category are *not* "sets equipped with extra stuff". Thus, in a general category, it does not make sense to talk about the "elements" of an object. Similarly, the maps need not be mappings or functions in the usual sense:

*The objects of a category need not be remotely like sets.*

*The maps in a category need not be remotely like functions.*

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Examples 1.1.8</span><span class="math-callout__name">(Categories as mathematical structures)</span></p>

**(a)** A category can be specified by saying directly what its objects, maps, composition and identities are. For example, there is a category $\mathbf{0}$ with no objects or maps at all. There is a category $\mathbf{1}$ with one object and only the identity map. There is a category with two objects and one non-identity map from the first to the second, drawn as $\bullet \to \bullet$.

**(b)** Some categories contain no maps apart from identities. These are called **discrete** categories. A discrete category amounts to just a class of objects.

**(c)** A group is essentially the same thing as a category that has only one object and in which all the maps are isomorphisms. A category $\mathscr{A}$ with a single object $A$ has a set $\mathscr{A}(A, A)$ with an associative composition and a two-sided unit $1_A$. If every map is an isomorphism, then every element has an inverse, making $\mathscr{A}(A, A)$ a group. The category of groups is equivalent to the category of (small) one-object categories in which every map is an isomorphism.

**(d)** A **monoid** is a set equipped with an associative binary operation and a two-sided unit element. A category with one object is essentially the same thing as a monoid.

**(e)** A **preorder** is a reflexive transitive binary relation. A **preordered set** $(S, \le)$ can be regarded as a category $\mathscr{A}$ in which, for each $A, B \in \mathscr{A}$, there is at most one map from $A$ to $B$. We write $A \le B$ to mean that a map $A \to B$ exists. An **order** (or **partially ordered set** / **poset**) is a preorder satisfying: if $A \le B$ and $B \le A$ then $A = B$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Construction 1.1.9</span><span class="math-callout__name">(Opposite category)</span></p>

Every category $\mathscr{A}$ has an **opposite** or **dual** category $\mathscr{A}^{\text{op}}$, defined by reversing the arrows. Formally, $\text{ob}(\mathscr{A}^{\text{op}}) = \text{ob}(\mathscr{A})$ and $\mathscr{A}^{\text{op}}(B, A) = \mathscr{A}(A, B)$ for all objects $A$ and $B$. Identities in $\mathscr{A}^{\text{op}}$ are the same as in $\mathscr{A}$. Composition in $\mathscr{A}^{\text{op}}$ is the same as in $\mathscr{A}$, but with the arguments reversed.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 1.1.10</span><span class="math-callout__name">(Principle of duality)</span></p>

The **principle of duality** is fundamental to category theory. Informally, it states that every categorical definition, theorem and proof has a **dual**, obtained by reversing all the arrows. Given any theorem, reversing the arrows throughout its statement and proof produces a dual theorem.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Construction 1.1.11</span><span class="math-callout__name">(Product category)</span></p>

Given categories $\mathscr{A}$ and $\mathscr{B}$, there is a **product category** $\mathscr{A} \times \mathscr{B}$, in which

$$
\text{ob}(\mathscr{A} \times \mathscr{B}) = \text{ob}(\mathscr{A}) \times \text{ob}(\mathscr{B}), \quad (\mathscr{A} \times \mathscr{B})((A, B), (A', B')) = \mathscr{A}(A, A') \times \mathscr{B}(B, B').
$$

An object is a pair $(A, B)$ with $A \in \mathscr{A}$ and $B \in \mathscr{B}$. A map $(A, B) \to (A', B')$ is a pair $(f, g)$ where $f \colon A \to A'$ in $\mathscr{A}$ and $g \colon B \to B'$ in $\mathscr{B}$.

</div>

### 1.2 Functors

One of the lessons of category theory is that whenever we meet a new type of mathematical object, we should always ask whether there is a sensible notion of "map" between such objects. A map between categories is called a functor.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1.2.1</span><span class="math-callout__name">(Functor)</span></p>

Let $\mathscr{A}$ and $\mathscr{B}$ be categories. A **functor** $F \colon \mathscr{A} \to \mathscr{B}$ consists of:

* a function $\text{ob}(\mathscr{A}) \to \text{ob}(\mathscr{B})$, written $A \mapsto F(A)$;
* for each $A, A' \in \mathscr{A}$, a function $\mathscr{A}(A, A') \to \mathscr{B}(F(A), F(A'))$, written $f \mapsto F(f)$,

satisfying the following axioms:

* $F(f' \circ f) = F(f') \circ F(f)$ whenever $A \xrightarrow{f} A' \xrightarrow{f'} A''$ in $\mathscr{A}$;
* $F(1_A) = 1_{F(A)}$ whenever $A \in \mathscr{A}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 1.2.2</span><span class="math-callout__name">(Functors preserve composition)</span></p>

**(a)** The definition ensures that from each string $A_0 \xrightarrow{f_1} \cdots \xrightarrow{f_n} A_n$ of maps in $\mathscr{A}$, it is possible to construct exactly one map $F(A_0) \to F(A_n)$ in $\mathscr{B}$.

**(b)** Structures and structure-preserving maps form a category. In particular, there is a category **CAT** whose objects are categories and whose maps are functors. Functors can be composed: given $\mathscr{A} \xrightarrow{F} \mathscr{B} \xrightarrow{G} \mathscr{C}$, there arises $\mathscr{A} \xrightarrow{G \circ F} \mathscr{C}$. For every category $\mathscr{A}$, there is an identity functor $1_{\mathscr{A}} \colon \mathscr{A} \to \mathscr{A}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Examples 1.2.3</span><span class="math-callout__name">(Forgetful functors)</span></p>

Perhaps the easiest examples of functors are the **forgetful functors** (an informal term):

**(a)** $U \colon \mathbf{Grp} \to \mathbf{Set}$: if $G$ is a group then $U(G)$ is the underlying set; if $f \colon G \to H$ is a homomorphism then $U(f)$ is the function $f$ itself.

**(b)** Similarly, there are forgetful functors $\mathbf{Ring} \to \mathbf{Set}$ and $\mathbf{Vect}_k \to \mathbf{Set}$.

**(c)** Forgetful functors need not forget *all* the structure. There is a functor $\mathbf{Ring} \to \mathbf{Ab}$ that forgets the multiplicative structure, remembering just the underlying additive group. There is a functor $U \colon \mathbf{Ring} \to \mathbf{Mon}$ that forgets the additive structure, remembering just the underlying multiplicative monoid.

**(d)** There is an inclusion functor $U \colon \mathbf{Ab} \to \mathbf{Grp}$ defined by $U(A) = A$ and $U(f) = f$. It forgets that abelian groups are abelian.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Examples 1.2.4</span><span class="math-callout__name">(Free functors)</span></p>

**Free functors** are in some sense dual to forgetful functors:

**(a)** Given any set $S$, the **free group** $F(S)$ on $S$ is a group containing $S$ as a subset and with no further properties than those forced by the definition of group. Elements of $F(S)$ are formal expressions or **words** such as $x^{-4}yx^2zy^{-3}$ (where $x, y, z \in S$). This gives a functor $F \colon \mathbf{Set} \to \mathbf{Grp}$.

**(b)** The free commutative ring $F(S)$ on a set $S$ gives a functor $F \colon \mathbf{Set} \to \mathbf{CRing}$. In fact, $F(S)$ is the ring of polynomials over $\mathbb{Z}$ in commuting variables $x_s$ ($s \in S$).

**(c)** The free vector space on a set $S$ over a field $k$ gives a functor $F \colon \mathbf{Set} \to \mathbf{Vect}_k$. The space $F(S)$ is the set of all formal $k$-linear combinations $\sum_{s \in S} \lambda_s s$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Examples 1.2.5</span><span class="math-callout__name">(Functors in algebraic topology)</span></p>

**(a)** Let $\mathbf{Top}_*$ be the category of topological spaces equipped with a basepoint. There is a functor $\pi_1 \colon \mathbf{Top}_* \to \mathbf{Grp}$ assigning to each space $X$ with basepoint $x$ the fundamental group $\pi_1(X, x)$, and to each basepoint-preserving continuous map $f \colon (X, x) \to (Y, y)$ the homomorphism $f_* \colon \pi_1(X, x) \to \pi_1(Y, y)$.

**(b)** For each $n \in \mathbb{N}$, there is a functor $H_n \colon \mathbf{Top} \to \mathbf{Ab}$ assigning to a space its $n$th homology group.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.2.6</span><span class="math-callout__name">(Functors from polynomial equations)</span></p>

Any system of polynomial equations such as $2x^2 + y^2 - 3z^2 = 1$ and $x^3 + x = y^2$ gives rise to a functor $\mathbf{CRing} \to \mathbf{Set}$. For each commutative ring $A$, let $F(A)$ be the set of triples $(x, y, z) \in A \times A \times A$ satisfying the equations. A ring homomorphism $f \colon A \to B$ induces $F(f) \colon F(A) \to F(B)$. In algebraic geometry, a **scheme** is a functor $\mathbf{CRing} \to \mathbf{Set}$ with certain properties.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.2.7</span><span class="math-callout__name">(Functors between one-object categories)</span></p>

Let $G$ and $H$ be monoids (or groups), regarded as one-object categories $\mathscr{G}$ and $\mathscr{H}$. A functor $F \colon \mathscr{G} \to \mathscr{H}$ must send the unique object of $\mathscr{G}$ to the unique object of $\mathscr{H}$, so it is determined by its effect on maps. The functor amounts to a function $F \colon G \to H$ such that $F(g'g) = F(g')F(g)$ for all $g', g \in G$ and $F(1) = 1$; that is, a homomorphism $G \to H$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.2.8</span><span class="math-callout__name">(Group actions as functors)</span></p>

Let $G$ be a monoid, regarded as a one-object category $\mathscr{G}$. A functor $F \colon \mathscr{G} \to \mathbf{Set}$ consists of a set $S$ (the value of $F$ at the unique object) together with, for each $g \in G$, a function $F(g) \colon S \to S$, satisfying $(F(g))(s) = g \cdot s$. This amounts to a set $S$ together with a left action of $G$: a **left $G$-set**.

Similarly, a functor $\mathscr{G} \to \mathbf{Vect}_k$ is exactly a $k$-linear representation of $G$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.2.9</span><span class="math-callout__name">(Order-preserving maps)</span></p>

When $A$ and $B$ are (pre)ordered sets, a functor between the corresponding categories is exactly an **order-preserving map**, that is, a function $f \colon A \to B$ such that $a \le a' \implies f(a) \le f(a')$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1.2.10</span><span class="math-callout__name">(Contravariant functor)</span></p>

Let $\mathscr{A}$ and $\mathscr{B}$ be categories. A **contravariant functor** from $\mathscr{A}$ to $\mathscr{B}$ is a functor $\mathscr{A}^{\text{op}} \to \mathscr{B}$.

An ordinary functor $\mathscr{A} \to \mathscr{B}$ is sometimes called a **covariant functor**, for emphasis.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.2.11</span><span class="math-callout__name">(Continuous functions as a contravariant functor)</span></p>

Given a topological space $X$, let $C(X)$ be the ring of continuous real-valued functions on $X$. A continuous map $f \colon X \to Y$ induces a ring homomorphism $C(f) \colon C(Y) \to C(X)$ defined by $(C(f))(q) = q \circ f$. Note that $C(f)$ goes in the *opposite* direction from $f$, making $C$ a contravariant functor from **Top** to **Ring**.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.2.12</span><span class="math-callout__name">(Hom functor and dual spaces)</span></p>

Let $k$ be a field. For any two vector spaces $V$ and $W$ over $k$, there is a vector space $\mathbf{Hom}(V, W) = \lbrace \text{linear maps } V \to W \rbrace$. Fixing $W$, any linear map $f \colon V \to V'$ induces a linear map $f^* \colon \mathbf{Hom}(V', W) \to \mathbf{Hom}(V, W)$ defined by $f^*(q) = q \circ f$. This defines a functor $\mathbf{Hom}(-, W) \colon \mathbf{Vect}_k^{\text{op}} \to \mathbf{Vect}_k$.

An important special case is where $W = k$: the vector space $\mathbf{Hom}(V, k)$ is called the **dual** of $V$, written $V^*$. This gives a contravariant functor $(\ )^* = \mathbf{Hom}(-, k) \colon \mathbf{Vect}_k^{\text{op}} \to \mathbf{Vect}_k$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1.2.15</span><span class="math-callout__name">(Presheaf)</span></p>

Let $\mathscr{A}$ be a category. A **presheaf** on $\mathscr{A}$ is a functor $\mathscr{A}^{\text{op}} \to \mathbf{Set}$.

The name comes from topology: let $X$ be a topological space and $\mathscr{O}(X)$ the poset of open subsets of $X$, viewed as a category. A **presheaf** on $X$ is a presheaf on $\mathscr{O}(X)$. For example, there is a presheaf $F$ defined by $F(U) = \lbrace \text{continuous functions } U \to \mathbb{R} \rbrace$ with restriction maps $F(U') \to F(U)$ whenever $U \subseteq U'$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1.2.16</span><span class="math-callout__name">(Faithful and full functors)</span></p>

A functor $F \colon \mathscr{A} \to \mathscr{B}$ is **faithful** (respectively, **full**) if for each $A, A' \in \mathscr{A}$, the function

$$
\mathscr{A}(A, A') \to \mathscr{B}(F(A), F(A')), \quad f \mapsto F(f)
$$

is injective (respectively, surjective).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Warning 1.2.17</span><span class="math-callout__name"></span></p>

Faithfulness does *not* say that if $f_1$ and $f_2$ are distinct maps in $\mathscr{A}$ then $F(f_1) \ne F(f_2)$. Note the roles of $A$ and $A'$ in the definition: faithfulness requires injectivity of $\mathscr{A}(A, A') \to \mathscr{B}(F(A), F(A'))$ for each *fixed* pair $(A, A')$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1.2.18</span><span class="math-callout__name">(Subcategory)</span></p>

Let $\mathscr{A}$ be a category. A **subcategory** $\mathscr{S}$ of $\mathscr{A}$ consists of a subclass $\text{ob}(\mathscr{S})$ of $\text{ob}(\mathscr{A})$ together with, for each $S, S' \in \text{ob}(\mathscr{S})$, a subclass $\mathscr{S}(S, S')$ of $\mathscr{A}(S, S')$, such that $\mathscr{S}$ is closed under composition and identities. It is a **full subcategory** if $\mathscr{S}(S, S') = \mathscr{A}(S, S')$ for all $S, S' \in \text{ob}(\mathscr{S})$.

A full subcategory can be specified simply by saying what its objects are. For example, **Ab** is the full subcategory of **Grp** consisting of the groups that are abelian.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Warning 1.2.19</span><span class="math-callout__name">(Image of a functor)</span></p>

The image of a functor need not be a subcategory. For example, consider a functor $F$ from the category $(A \xrightarrow{f} B \quad B' \xrightarrow{g} C)$ to a category with three objects $X$, $Y$, $Z$ and maps $p$, $q$, $qp$, where $F(A) = X$, $F(B) = F(B') = Y$, $F(C) = Z$, $F(f) = p$, and $F(g) = q$. Then $p$ and $q$ are in the image of $F$, but $qp$ is not.

</div>

### 1.3 Natural Transformations

We now know about categories and functors. Perhaps surprisingly, there is a further notion of "map between functors" called natural transformations. This notion only applies when the functors have the same domain and codomain:

$$
\mathscr{A} \underset{G}{\overset{F}{\rightrightarrows}} \mathscr{B}.
$$

To motivate the definition, consider the discrete category $\mathscr{A}$ whose objects are $0, 1, 2, \ldots$ A functor $F$ from $\mathscr{A}$ to $\mathscr{B}$ is simply a sequence $(F_0, F_1, F_2, \ldots)$ of objects of $\mathscr{B}$. Given another functor $G = (G_0, G_1, G_2, \ldots)$, it is natural to define a "map" from $F$ to $G$ as a sequence of maps $(\alpha_0, \alpha_1, \alpha_2, \ldots)$ where $\alpha_i \colon F_i \to G_i$ in $\mathscr{B}$. In the general case, where $\mathscr{A}$ may have nontrivial maps, we demand compatibility between the maps in $\mathscr{A}$ and the maps $\alpha_A$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1.3.1</span><span class="math-callout__name">(Natural transformation)</span></p>

Let $\mathscr{A}$ and $\mathscr{B}$ be categories and let $F, G \colon \mathscr{A} \to \mathscr{B}$ be functors. A **natural transformation** $\alpha \colon F \to G$ is a family $\left( F(A) \xrightarrow{\alpha_A} G(A) \right)_{A \in \mathscr{A}}$ of maps in $\mathscr{B}$ such that for every map $A \xrightarrow{f} A'$ in $\mathscr{A}$, the square

$$
\begin{array}{ccc}
F(A) & \xrightarrow{F(f)} & F(A') \\
\downarrow{\alpha_A} & & \downarrow{\alpha_{A'}} \\
G(A) & \xrightarrow{G(f)} & G(A')
\end{array}
$$

commutes, i.e. $\alpha_{A'} \circ F(f) = G(f) \circ \alpha_A$. The maps $\alpha_A$ are called the **components** of $\alpha$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 1.3.2</span><span class="math-callout__name">(Naturality squares)</span></p>

**(a)** The definition is set up so that from each map $A \xrightarrow{f} A'$ in $\mathscr{A}$, it is possible to construct exactly one map $F(A) \to G(A')$ in $\mathscr{B}$. When $f = 1_A$, this map is $\alpha_A$. For a general $f$, it is the diagonal of the naturality square, and "exactly one" implies that the square commutes.

**(b)** We write $\mathscr{A} \overset{F}{\underset{G}{\Downarrow \alpha}} \mathscr{B}$ to mean that $\alpha$ is a natural transformation from $F$ to $G$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.3.3</span><span class="math-callout__name">(Discrete categories)</span></p>

Let $\mathscr{A}$ be a discrete category, and let $F, G \colon \mathscr{A} \to \mathscr{B}$ be functors. Then $F$ and $G$ are just families $(F(A))\_{A \in \mathscr{A}}$ and $(G(A))\_{A \in \mathscr{A}}$ of objects of $\mathscr{B}$. A natural transformation $\alpha \colon F \to G$ is just a family $\left( F(A) \xrightarrow{\alpha_A} G(A) \right)\_{A \in \mathscr{A}}$ of maps in $\mathscr{B}$. The naturality axiom holds automatically because the only maps in $\mathscr{A}$ are identities.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.3.4</span><span class="math-callout__name">(G-equivariant maps)</span></p>

Recall from Examples 1.1.8 that a group (or monoid) $G$ can be regarded as a one-object category, and from Example 1.2.8 that a functor from $G$ to **Set** is a left $G$-set. Take two $G$-sets $S$ and $T$, regarded as functors $G \to \mathbf{Set}$. A natural transformation $\alpha \colon S \to T$ consists of a single map $\alpha \colon S \to T$ (since $G$ has just one object) satisfying the naturality condition: $\alpha(g \cdot s) = g \cdot \alpha(s)$ for all $s \in S$ and $g \in G$. This is a map of $G$-sets, sometimes called a **$G$-equivariant** map.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.3.5</span><span class="math-callout__name">(Determinant as a natural transformation)</span></p>

Fix a natural number $n$. For any commutative ring $R$, the $n \times n$ matrices with entries in $R$ form a monoid $M_n(R)$ under multiplication. Any ring homomorphism $R \to S$ induces a monoid homomorphism $M_n(R) \to M_n(S)$. This defines a functor $M_n \colon \mathbf{CRing} \to \mathbf{Mon}$.

Also, the elements of any ring $R$ form a monoid $U(R)$ under multiplication, giving another functor $U \colon \mathbf{CRing} \to \mathbf{Mon}$.

For each $R$, the determinant $\det_R \colon M_n(R) \to U(R)$ is a monoid homomorphism (since $\det_R(XY) = \det_R(X)\det_R(Y)$ and $\det_R(I) = 1$). The naturality squares commute because determinant is defined in the same way for all rings. This gives a natural transformation

$$
\mathbf{CRing} \overset{M_n}{\underset{U}{\Downarrow \det}} \mathbf{Mon}.
$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Construction 1.3.6</span><span class="math-callout__name">(Composition of natural transformations and functor categories)</span></p>

Given natural transformations

$$
\mathscr{A} \overset{F}{\underset{G}{\Downarrow \alpha}} \mathscr{B} \quad \text{and} \quad \mathscr{A} \overset{G}{\underset{H}{\Downarrow \beta}} \mathscr{B},
$$

there is a composite natural transformation $\beta \circ \alpha \colon F \to H$ defined by $(\beta \circ \alpha)_A = \beta_A \circ \alpha_A$ for all $A \in \mathscr{A}$. There is also an identity natural transformation $1_F \colon F \to F$ on any functor $F$, defined by $(1_F)_A = 1_{F(A)}$.

So for any two categories $\mathscr{A}$ and $\mathscr{B}$, there is a category whose objects are the functors from $\mathscr{A}$ to $\mathscr{B}$ and whose maps are the natural transformations between them. This is called the **functor category** from $\mathscr{A}$ to $\mathscr{B}$, and written as $[\mathscr{A}, \mathscr{B}]$ or $\mathscr{B}^{\mathscr{A}}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.3.7</span><span class="math-callout__name">(Functor category from a discrete category)</span></p>

Let $\mathbf{2}$ be the discrete category with two objects. A functor from $\mathbf{2}$ to a category $\mathscr{B}$ is a pair of objects of $\mathscr{B}$, and a natural transformation is a pair of maps. The functor category $[\mathbf{2}, \mathscr{B}]$ is therefore isomorphic to the product category $\mathscr{B} \times \mathscr{B}$ (Construction 1.1.11). This fits well with the alternative notation $\mathscr{B}^2$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.3.8</span><span class="math-callout__name">(Functor categories of G-sets)</span></p>

Let $G$ be a monoid. Then $[G, \mathbf{Set}]$ is the category of left $G$-sets, and $[G^{\text{op}}, \mathbf{Set}]$ is the category of right $G$-sets (Example 1.2.14).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.3.9</span><span class="math-callout__name">(Ordered sets as functor categories)</span></p>

Take ordered sets $A$ and $B$, viewed as categories (Example 1.1.8(e)). Given order-preserving maps $f, g \colon A \to B$ viewed as functors (Example 1.2.9), there is at most one natural transformation from $f$ to $g$, and there is one if and only if $f(a) \le g(a)$ for all $a \in A$. (The naturality axiom holds automatically in an ordered set because all diagrams commute.)

So $[A, B]$ is an ordered set too; its elements are the order-preserving maps from $A$ to $B$, and $f \le g$ if and only if $f(a) \le g(a)$ for all $a \in A$.

</div>

#### Natural Isomorphism

Everyday phrases such as "*the* cyclic group of order 6" reflect the fact that given two isomorphic objects of a category, we usually neither know nor care whether they are actually equal. In particular, given two functors $F, G \colon \mathscr{A} \to \mathscr{B}$, we usually do not care whether they are literally equal. What really matters is whether they are naturally isomorphic.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1.3.10</span><span class="math-callout__name">(Natural isomorphism)</span></p>

Let $\mathscr{A}$ and $\mathscr{B}$ be categories. A **natural isomorphism** between functors from $\mathscr{A}$ to $\mathscr{B}$ is an isomorphism in $[\mathscr{A}, \mathscr{B}]$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 1.3.11</span><span class="math-callout__name">(Componentwise characterization)</span></p>

*Let $\alpha \colon F \to G$ be a natural transformation between functors $\mathscr{A} \to \mathscr{B}$. Then $\alpha$ is a natural isomorphism if and only if $\alpha_A \colon F(A) \to G(A)$ is an isomorphism for all $A \in \mathscr{A}$.*

**Proof.** Exercise 1.3.26. $\square$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1.3.12</span><span class="math-callout__name">(Naturally isomorphic in $A$)</span></p>

Given functors $\mathscr{A} \overset{F}{\underset{G}{\rightrightarrows}} \mathscr{B}$, we say that $F(A) \cong G(A)$ **naturally in** $A$ if $F$ and $G$ are naturally isomorphic.

If $F \cong G$ then certainly $F(A) \cong G(A)$ for each individual $A$, but the point is that we can choose isomorphisms $\alpha_A \colon F(A) \to G(A)$ in such a way that the naturality axiom is satisfied. We write $F \cong G$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.3.13</span><span class="math-callout__name">(Natural isomorphism for discrete categories)</span></p>

Let $F, G \colon \mathscr{A} \to \mathscr{B}$ be functors from a discrete category $\mathscr{A}$ to a category $\mathscr{B}$. Then $F \cong G$ if and only if $F(A) \cong G(A)$ for all $A \in \mathscr{A}$.

But this is only true because $\mathscr{A}$ is discrete. In general, it is emphatically false: there are many examples of categories and functors $\mathscr{A} \overset{F}{\underset{G}{\rightrightarrows}} \mathscr{B}$ such that $F(A) \cong G(A)$ for all $A \in \mathscr{A}$, but not *naturally* in $A$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.3.14</span><span class="math-callout__name">(Double dual as a natural isomorphism)</span></p>

Let **FDVect** be the category of finite-dimensional vector spaces over some field $k$. The dual vector space construction defines a contravariant functor from **FDVect** to itself (Example 1.2.12), and the double dual construction therefore defines a covariant functor $(\ )^{**}$ from **FDVect** to itself.

For each $V \in \mathbf{FDVect}$, there is a canonical isomorphism $\alpha_V \colon V \to V^{**}$ given by "evaluation at $v$": for $v \in V$, the element $\alpha_V(v) \in V^{**}$ maps $\phi \in V^*$ to $\phi(v) \in k$. This defines a natural transformation $\alpha \colon 1\_{\mathbf{FDVect}} \to (\ )^{**}$. By Lemma 1.3.11, $\alpha$ is a natural isomorphism: $V \cong V^{**}$ naturally in $V$.

This makes precise the intuition that the isomorphism between a finite-dimensional vector space and its double dual is **canonical** (no arbitrary choices needed), while the isomorphism $V \cong V^*$ requires an arbitrary choice of basis and is *not* natural.

</div>

#### Equivalence of Categories

The concept of natural isomorphism leads unavoidably to another central concept: equivalence of categories. Two objects of a category can be equal, not equal but isomorphic, or not even isomorphic. Isomorphism is usually the right notion of "sameness" for objects. Applied to functor categories, this means:

* the right notion of sameness of two elements of a set is equality;
* the right notion of sameness of two objects of a category is isomorphism;
* the right notion of sameness of two functors $\mathscr{A} \rightrightarrows \mathscr{B}$ is natural isomorphism.

But what is the right notion of sameness of two *categories*? Isomorphism ($\mathscr{A} \cong \mathscr{B}$, meaning there are functors $F$, $G$ with $G \circ F = 1_{\mathscr{A}}$ and $F \circ G = 1_{\mathscr{B}}$) is unreasonably strict, since equality of functors is too strict.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1.3.15</span><span class="math-callout__name">(Equivalence of categories)</span></p>

An **equivalence** between categories $\mathscr{A}$ and $\mathscr{B}$ consists of a pair of functors

$$
\mathscr{A} \underset{G}{\overset{F}{\rightleftarrows}} \mathscr{B}
$$

together with natural isomorphisms

$$
\eta \colon 1_{\mathscr{A}} \to G \circ F, \qquad \varepsilon \colon F \circ G \to 1_{\mathscr{B}}.
$$

If there exists an equivalence between $\mathscr{A}$ and $\mathscr{B}$, we say that $\mathscr{A}$ and $\mathscr{B}$ are **equivalent**, and write $\mathscr{A} \simeq \mathscr{B}$. We also say that the functors $F$ and $G$ are **equivalences**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Warning 1.3.16</span><span class="math-callout__name">(Notation: $\cong$ vs. $\simeq$)</span></p>

The symbol $\cong$ is used for isomorphism of objects of a category (and in particular for isomorphism of categories as objects of **CAT**). The symbol $\simeq$ is used for equivalence of categories. This is the convention used in this book and by most category theorists.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 1.3.17</span><span class="math-callout__name">(Essentially surjective on objects)</span></p>

A functor $F \colon \mathscr{A} \to \mathscr{B}$ is **essentially surjective on objects** if for all $B \in \mathscr{B}$, there exists $A \in \mathscr{A}$ such that $F(A) \cong B$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 1.3.18</span><span class="math-callout__name">(Characterization of equivalences)</span></p>

*A functor is an equivalence if and only if it is full, faithful and essentially surjective on objects.*

**Proof.** Exercise 1.3.32. $\square$

This result is analogous to the theorem that every bijective group homomorphism is an isomorphism, or that a natural transformation whose components are all isomorphisms is itself an isomorphism (Lemma 1.3.11). It allows us to show that a functor $F$ is an equivalence without directly constructing an "inverse" $G$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 1.3.19</span><span class="math-callout__name">(Full and faithful functors and subcategories)</span></p>

*Let $F \colon \mathscr{C} \to \mathscr{D}$ be a full and faithful functor. Then $\mathscr{C}$ is equivalent to the full subcategory $\mathscr{C}'$ of $\mathscr{D}$ whose objects are those of the form $F(C)$ for some $C \in \mathscr{C}$.*

**Proof.** The functor $F' \colon \mathscr{C} \to \mathscr{C}'$ defined by $F'(C) = F(C)$ is full and faithful (since $F$ is) and essentially surjective on objects (by definition of $\mathscr{C}'$). $\square$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.3.20</span><span class="math-callout__name">(Equivalent subcategories)</span></p>

Let $\mathscr{A}$ be any category, and let $\mathscr{B}$ be any full subcategory containing at least one object from each isomorphism class of $\mathscr{A}$. Then the inclusion functor $\mathscr{B} \hookrightarrow \mathscr{A}$ is faithful (like any inclusion of subcategories), full, and essentially surjective on objects. Hence $\mathscr{B} \simeq \mathscr{A}$.

For example, let **FinSet** be the category of finite sets and let $\mathscr{B}$ be the full subcategory with objects $\mathbf{0}, \mathbf{1}, \ldots$ (one chosen set of each cardinality $n$). Then $\mathscr{B} \simeq \mathbf{FinSet}$, even though $\mathscr{B}$ is much smaller.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.3.21</span><span class="math-callout__name">(Monoids as one-object categories)</span></p>

In Example 1.1.8(d), we saw that monoids are essentially the same thing as one-object categories. Let $\mathscr{C}$ be the full subcategory of **CAT** whose objects are the one-object categories. Let **Mon** be the category of monoids. Then $\mathscr{C} \simeq \mathbf{Mon}$: there is a canonical functor $F \colon \mathscr{C} \to \mathbf{Mon}$ sending a one-object category to the monoid of maps from the single object to itself. This functor is full and faithful (by Example 1.2.7) and essentially surjective on objects.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.3.22</span><span class="math-callout__name">(Duality)</span></p>

An equivalence of the form $\mathscr{A}^{\text{op}} \simeq \mathscr{B}$ is sometimes called a **duality** between $\mathscr{A}$ and $\mathscr{B}$. One says that $\mathscr{A}$ is **dual** to $\mathscr{B}$. Famous examples include:

* **Stone duality:** the category of Boolean algebras is dual to the category of totally disconnected compact Hausdorff spaces.
* **Gelfand--Naimark duality:** the category of commutative unital $C^*$-algebras is dual to the category of compact Hausdorff spaces.
* **Affine algebraic geometry:** the category of affine varieties over an algebraically closed field $k$ is dual to the category of finitely generated $k$-algebras with no nontrivial nilpotents.
* **Pontryagin duality:** the category of locally compact abelian topological groups is dual to itself.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 1.3.23</span><span class="math-callout__name">(Groups as a category of nonempty sets)</span></p>

Let $\mathscr{A}$ be the category whose objects are groups and whose maps are *all* functions between them (not necessarily homomorphisms). Let $\mathbf{Set}_{\ne \emptyset}$ be the category of nonempty sets. The forgetful functor $U \colon \mathscr{A} \to \mathbf{Set}\_{\ne \emptyset}$ is full and faithful. Since every nonempty set can be given at least one group structure, $U$ is essentially surjective on objects. Hence $U$ is an equivalence: $\mathscr{A}$, although defined in terms of groups, is really just the category of nonempty sets.

</div>

#### Vertical and Horizontal Composition

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 1.3.24</span><span class="math-callout__name">(Review of structures)</span></p>

So far we have defined:

* categories (Section 1.1);
* functors between categories (Section 1.2);
* natural transformations between functors (Section 1.3);
* composition of functors and the identity functor (Remark 1.2.2(b));
* **vertical composition** of natural transformations and the identity natural transformation (Construction 1.3.6).

There is also **horizontal composition**, which takes natural transformations

$$
\mathscr{A} \overset{F}{\underset{G}{\Downarrow \alpha}} \mathscr{A}' \overset{F'}{\underset{G'}{\Downarrow \alpha'}} \mathscr{A}''
$$

and produces a natural transformation $\alpha' * \alpha \colon F' \circ F \to G' \circ G$. The component of $\alpha' * \alpha$ at $A \in \mathscr{A}$ is defined to be the diagonal of the naturality square:

$$
(\alpha' * \alpha)_A = \alpha'\_{G(A)} \circ F'(\alpha_A) = G'(\alpha_A) \circ \alpha'\_{F(A)}.
$$

The special cases where either $\alpha$ or $\alpha'$ is an identity have their own notation: given a functor $F \colon \mathscr{A} \to \mathscr{A}'$ and a natural transformation $\alpha' \colon F' \to G'$ between functors $\mathscr{A}' \to \mathscr{A}''$, we write $\alpha' F$ for $\alpha' * 1_F$, so that $(\alpha' F)_A = \alpha'\_{F(A)}$. Dually, given $\alpha \colon F \to G$ and $F' \colon \mathscr{A}' \to \mathscr{A}''$, we write $F' \alpha$ for $1\_{F'} * \alpha$, so that $(F' \alpha)_A = F'(\alpha_A)$.

Vertical and horizontal composition obey the **interchange law**:

$$
(\beta' \circ \alpha') * (\beta \circ \alpha) = (\beta' * \beta) \circ (\alpha' * \alpha) \colon F' \circ F \to H' \circ H.
$$

All of this enables us to construct, for any categories $\mathscr{A}$, $\mathscr{A}'$ and $\mathscr{A}''$, a functor

$$
[\mathscr{A}', \mathscr{A}''] \times [\mathscr{A}, \mathscr{A}'] \to [\mathscr{A}, \mathscr{A}''],
$$

given on objects by $(F', F) \mapsto F' \circ F$ and on maps by $(\alpha', \alpha) \mapsto \alpha' * \alpha$.

The diagrams above contain not only objects and arrows $\to$, but also double arrows $\Rightarrow$ sweeping out 2-dimensional regions. This is **2-category theory**. There is a 2-category of categories, functors and natural transformations. If we take this seriously, we are led to 3-categories, and eventually $\infty$-categories. But in this book, we climb no higher than the first rung or two of this infinite ladder.

</div>

## Chapter 2: Adjoints

The slogan of Saunders Mac Lane's book *Categories for the Working Mathematician* is: *Adjoint functors arise everywhere.* We will approach the theory of adjoints from three different directions, each carrying its own intuition, and then prove that the three approaches are equivalent.

### 2.1 Definition and Examples

Consider a pair of functors in opposite directions, $F \colon \mathscr{A} \to \mathscr{B}$ and $G \colon \mathscr{B} \to \mathscr{A}$. Roughly speaking, $F$ is said to be left adjoint to $G$ if, whenever $A \in \mathscr{A}$ and $B \in \mathscr{B}$, maps $F(A) \to B$ are essentially the same thing as maps $A \to G(B)$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.1.1</span><span class="math-callout__name">(Left and Right Adjoint)</span></p>

Let $\mathscr{A} \overset{F}{\underset{G}{\rightleftarrows}} \mathscr{B}$ be categories and functors. We say that $F$ is **left adjoint** to $G$, and $G$ is **right adjoint** to $F$, and write $F \dashv G$, if

$$
\mathscr{B}(F(A), B) \cong \mathscr{A}(A, G(B))
$$

naturally in $A \in \mathscr{A}$ and $B \in \mathscr{B}$. The meaning of "naturally" is defined below. An **adjunction** between $F$ and $G$ is a choice of natural isomorphism as above.

</div>

"Naturally in $A \in \mathscr{A}$ and $B \in \mathscr{B}$" means that there is a specified bijection for each $A \in \mathscr{A}$ and $B \in \mathscr{B}$, and that it satisfies a naturality axiom. Given objects $A \in \mathscr{A}$ and $B \in \mathscr{B}$, the correspondence between maps $F(A) \to B$ and $A \to G(B)$ is denoted by a horizontal bar, in both directions:

$$
\left(F(A) \xrightarrow{g} B\right) \mapsto \left(A \xrightarrow{\bar{g}} G(B)\right), \qquad \left(F(A) \xrightarrow{\bar{f}} B\right) \leftarrow \left(A \xrightarrow{f} G(B)\right).
$$

So $\bar{\bar{f}} = f$ and $\bar{\bar{g}} = g$. We call $\bar{f}$ the **transpose** of $f$, and similarly for $g$. The naturality axiom has two parts:

$$
\overline{q \circ g} = G(q) \circ \bar{g} \qquad \text{for all } g \text{ and } q,
$$

$$
\overline{f \circ p} = \bar{f} \circ F(p) \qquad \text{for all } p \text{ and } f.
$$

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remarks 2.1.2</span></p>

**(a)** The naturality axiom might seem ad hoc, but we will see in Chapter 4 that it simply says that two particular functors are naturally isomorphic. In this section, we ignore the naturality axiom altogether, trusting that it embodies our usual intuitive idea of naturality: something defined without making any arbitrary choices.

**(b)** The naturality axiom implies that from each array of maps

$$
A_0 \to \cdots \to A_n, \quad F(A_n) \to B_0, \quad B_0 \to \cdots \to B_m,
$$

it is possible to construct exactly one map $A_0 \to G(B_m)$.

**(c)** Not only do adjoint functors arise everywhere; better, whenever you see a pair of functors $\mathscr{A} \rightleftarrows \mathscr{B}$, there is an excellent chance that they are adjoint (one way round or the other).

**(d)** A given functor $G$ may or may not have a left adjoint, but if it does, it is unique up to isomorphism, so we may speak of "*the* left adjoint of $G$." The same goes for right adjoints. We prove this later (Example 4.3.13).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Examples 2.1.3</span><span class="math-callout__name">(Algebra: free $\dashv$ forgetful)</span></p>

Forgetful functors between categories of algebraic structures usually have left adjoints. For instance:

**(a)** Let $k$ be a field. There is an adjunction $F \dashv U$ between $\mathbf{Set}$ and $\mathbf{Vect}_k$, where $U$ is the forgetful functor and $F$ is the free functor. Adjointness says that given a set $S$ and a vector space $V$, a linear map $F(S) \to V$ is essentially the same thing as a function $S \to U(V)$.

Given a linear map $g \colon F(S) \to V$, we may define a map of sets $\bar{g} \colon S \to U(V)$ by $\bar{g}(s) = g(s)$ for all $s \in S$. In the other direction, given $f \colon S \to U(V)$, we may define a linear map $\bar{f} \colon F(S) \to V$ by $\bar{f}(\sum_{s \in S} \lambda_s s) = \sum_{s \in S} \lambda_s f(s)$. These two functions "bar" are mutually inverse, giving a canonical bijection $\mathbf{Vect}_k(F(S), V) \cong \mathbf{Set}(S, U(V))$.

**(b)** In the same way, there is an adjunction $F \dashv U$ between $\mathbf{Set}$ and $\mathbf{Grp}$, where $F$ and $U$ are the free and forgetful functors.

**(c)** There is an adjunction $F \dashv U$ between $\mathbf{Grp}$ and $\mathbf{Ab}$, where $U$ is the inclusion functor. If $G$ is a group then $F(G)$ is the **abelianization** $G_{\text{ab}}$ of $G$. This is an abelian quotient group of $G$, with the property that every map from $G$ to an abelian group factorizes uniquely through $G_{\text{ab}}$. The bijection is $\mathbf{Ab}(G_{\text{ab}}, A) \cong \mathbf{Grp}(G, U(A))$.

**(d)** There are adjunctions $F \dashv U \dashv R$ between the categories of groups and monoids. The middle functor $U$ is inclusion. The left adjoint $F$ is obtained from a monoid $M$ by throwing in an inverse to every element. The right adjoint $R(M)$ is the submonoid of $M$ consisting of all the invertible elements. The category $\mathbf{Grp}$ is both a **reflective** and a **coreflective** subcategory of $\mathbf{Mon}$.

**(e)** Let $\mathbf{Field}$ be the category of fields, with ring homomorphisms as the maps. The forgetful functor $\mathbf{Field} \to \mathbf{Set}$ does *not* have a left adjoint. The theory of fields is unlike the theories of groups, rings, and so on, because the operation $x \mapsto x^{-1}$ is not defined for *all* $x$ (only for $x \ne 0$).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.1.4</span><span class="math-callout__name">(Algebraic theories)</span></p>

At several points in this book, we make contact with the idea of an **algebraic theory**. You already know several examples: the theory of groups is an algebraic theory, as are the theory of rings, the theory of vector spaces over $\mathbb{R}$, the theory of monoids, and (rather trivially) the theory of sets.

An algebraic theory consists of two things: first, a collection of operations, each with a specified arity (number of inputs), and second, a collection of equations. For example, the theory of groups has one operation of arity 2, one of arity 1, and one of arity 0. An **algebra** or **model** for an algebraic theory consists of a set $X$ together with a specified map $X^n \to X$ for each operation of arity $n$, such that the equations hold everywhere. The main property is that the operations are defined everywhere on the set, and the equations hold everywhere too. This is why the theories of groups, rings, and so on, are algebraic theories, but the theory of fields is not.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.1.5</span><span class="math-callout__name">(Topological spaces)</span></p>

There are adjunctions $D \dashv U \dashv I$ between $\mathbf{Set}$ and $\mathbf{Top}$, where $U$ sends a space to its set of points, $D$ equips a set with the discrete topology, and $I$ equips a set with the indiscrete topology.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.1.6</span><span class="math-callout__name">(Products and exponentials in Set)</span></p>

Given sets $A$ and $B$, we can form their cartesian product $A \times B$. We can also form the set $B^A$ of functions from $A$ to $B$. Fix a set $B$. Taking the product with $B$ defines a functor $- \times B \colon \mathbf{Set} \to \mathbf{Set}$, and there is also a functor $(-)^B \colon \mathbf{Set} \to \mathbf{Set}$ defined by $C \mapsto C^B$. There is a canonical bijection

$$
\mathbf{Set}(A \times B, C) \cong \mathbf{Set}(A, C^B)
$$

for any sets $A$ and $C$. It is defined by simply changing the punctuation: given a map $g \colon A \times B \to C$, define $\bar{g} \colon A \to C^B$ by $(\bar{g}(a))(b) = g(a,b)$, and in the other direction, given $f \colon A \to C^B$, define $\bar{f} \colon A \times B \to C$ by $\bar{f}(a,b) = (f(a))(b)$. This gives an adjunction $- \times B \dashv (-)^B$ for every set $B$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.1.7</span><span class="math-callout__name">(Initial and Terminal Objects)</span></p>

Let $\mathscr{A}$ be a category. An object $I \in \mathscr{A}$ is **initial** if for every $A \in \mathscr{A}$, there is exactly one map $I \to A$. An object $T \in \mathscr{A}$ is **terminal** if for every $A \in \mathscr{A}$, there is exactly one map $A \to T$.

For example, the empty set is initial in $\mathbf{Set}$, the trivial group is initial in $\mathbf{Grp}$, and $\mathbb{Z}$ is initial in $\mathbf{Ring}$. The one-element set is terminal in $\mathbf{Set}$, the trivial group is terminal (as well as initial) in $\mathbf{Grp}$, and the trivial (one-element) ring is terminal in $\mathbf{Ring}$. The terminal object of $\mathbf{CAT}$ is the category $\mathbf{1}$ containing just one object and one map.

A category need not have an initial object, but if it does have one, it is unique up to isomorphism. Indeed, it is unique up to *unique* isomorphism.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 2.1.8</span><span class="math-callout__name">(Uniqueness of initial objects)</span></p>

*Let $I$ and $I'$ be initial objects of a category. Then there is a unique isomorphism $I \to I'$. In particular, $I \cong I'$.*

**Proof.** Since $I$ is initial, there is a unique map $f \colon I \to I'$. Since $I'$ is initial, there is a unique map $f' \colon I' \to I$. Now $f' \circ f$ and $1_I$ are both maps $I \to I$, and $I$ is initial, so $f' \circ f = 1_I$. Similarly, $f \circ f' = 1_{I'}$. Hence $f$ is an isomorphism, as required. $\square$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.1.9</span><span class="math-callout__name">(Initial and terminal objects as adjoints)</span></p>

Initial and terminal objects can be described as adjoints. Let $\mathscr{A}$ be a category. There is precisely one functor $\mathscr{A} \to \mathbf{1}$. Also, a functor $\mathbf{1} \to \mathscr{A}$ is essentially just an object of $\mathscr{A}$ (namely, the object to which the unique object of $\mathbf{1}$ is mapped). Viewing functors $\mathbf{1} \to \mathscr{A}$ as objects of $\mathscr{A}$, a left adjoint to $\mathscr{A} \to \mathbf{1}$ is exactly an initial object of $\mathscr{A}$.

Similarly, a right adjoint to the unique functor $\mathscr{A} \to \mathbf{1}$ is exactly a terminal object of $\mathscr{A}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.1.10</span><span class="math-callout__name">(Duality of initial and terminal)</span></p>

The concept of terminal object is dual to the concept of initial object. More generally, the concepts of left and right adjoint are dual to one another. Since any two initial objects of a category are uniquely isomorphic, the principle of duality implies that the same is true of terminal objects.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.1.11</span><span class="math-callout__name">(Composition of adjunctions)</span></p>

Adjunctions can be composed. Take adjunctions

$$
\mathscr{A} \overset{F}{\underset{G}{\rightleftarrows}} \mathscr{A}' \overset{F'}{\underset{G'}{\rightleftarrows}} \mathscr{A}''
$$

(thus $F \dashv G$ and $F' \dashv G'$). Then we obtain an adjunction $F' \circ F \dashv G \circ G'$, since for $A \in \mathscr{A}$ and $A'' \in \mathscr{A}''$,

$$
\mathscr{A}''(F'(F(A)), A'') \cong \mathscr{A}'(F(A), G'(A'')) \cong \mathscr{A}(A, G(G'(A'')))
$$

naturally in $A$ and $A''$.

</div>

### 2.2 Adjunctions via Units and Counits

In the previous section, we met the definition of adjunction. In this section, we meet a second way of rephrasing the definition that is most useful for theoretical purposes.

To start building the theory of adjoint functors, we have to take seriously the naturality requirement. Intuitively, naturality says that as $A$ varies in $\mathscr{A}$ and $B$ varies in $\mathscr{B}$, the isomorphism between $\mathscr{B}(F(A), B)$ and $\mathscr{A}(A, G(B))$ varies in a way that is compatible with all the structure already in place.

For each $A \in \mathscr{A}$, we have a map

$$
\left(A \xrightarrow{\eta_A} GF(A)\right) = \overline{\left(F(A) \xrightarrow{1} F(A)\right)}.
$$

Dually, for each $B \in \mathscr{B}$, we have a map

$$
\left(FG(B) \xrightarrow{\varepsilon_B} B\right) = \overline{\left(G(B) \xrightarrow{1} G(B)\right)}.
$$

These define natural transformations

$$
\eta \colon 1_{\mathscr{A}} \to G \circ F, \qquad \varepsilon \colon F \circ G \to 1_{\mathscr{B}},
$$

called the **unit** and **counit** of the adjunction, respectively.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.2.1</span><span class="math-callout__name">(Unit and counit for $\mathbf{Vect}_k \rightleftarrows \mathbf{Set}$)</span></p>

Take the usual adjunction $F \dashv U$ between $\mathbf{Set}$ and $\mathbf{Vect}_k$. Its unit $\eta \colon 1_{\mathbf{Set}} \to U \circ F$ has components

$$
\eta_S \colon S \to UF(S) = \left\lbrace \text{formal } k\text{-linear sums } \textstyle\sum_{s \in S} \lambda_s s \right\rbrace, \qquad s \mapsto s.
$$

The component of the counit $\varepsilon$ at a vector space $V$ is the linear map

$$
\varepsilon_V \colon FU(V) \to V
$$

that sends a *formal* linear sum $\sum_{v \in V} \lambda_v v$ to its *actual* value in $V$. The vector space $FU(V)$ is enormous: for instance, if $k = \mathbb{R}$ and $V$ is $\mathbb{R}^2$, then $FU(V)$ is a vector space with one basis element for every element of $\mathbb{R}^2$, hence uncountably infinite-dimensional.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 2.2.2</span><span class="math-callout__name">(Triangle identities)</span></p>

*Given an adjunction $F \dashv G$ with unit $\eta$ and counit $\varepsilon$, the triangles*

$$
\varepsilon F \circ F\eta = 1_F, \qquad G\varepsilon \circ \eta G = 1_G
$$

*commute.*

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.2.3</span><span class="math-callout__name">(Triangle identities as commutative diagrams)</span></p>

The triangle identities are commutative diagrams in the functor categories $[\mathscr{A}, \mathscr{B}]$ and $[\mathscr{B}, \mathscr{A}]$, respectively. An equivalent statement is that the triangles

$$
F(A) \xrightarrow{F(\eta_A)} FGF(A) \xrightarrow{\varepsilon_{F(A)}} F(A) = 1_{F(A)}
$$

$$
G(B) \xrightarrow{\eta_{G(B)}} GFG(B) \xrightarrow{G(\varepsilon_B)} G(B) = 1_{G(B)}
$$

commute for all $A \in \mathscr{A}$ and $B \in \mathscr{B}$.

</div>

Amazingly, the unit and counit determine the whole adjunction, even though they appear to know only the transposes *of identities*. This is the main content of the following pair of results.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 2.2.4</span><span class="math-callout__name">(Transpose formulas)</span></p>

*Let $\mathscr{A} \overset{F}{\underset{G}{\rightleftarrows}} \mathscr{B}$ be an adjunction, with unit $\eta$ and counit $\varepsilon$. Then*

$$
\bar{g} = G(g) \circ \eta_A
$$

*for any $g \colon F(A) \to B$, and*

$$
\bar{f} = \varepsilon_B \circ F(f)
$$

*for any $f \colon A \to G(B)$.*

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.2.5</span><span class="math-callout__name">(Equivalence of adjunction definitions)</span></p>

*Take categories and functors $\mathscr{A} \overset{F}{\underset{G}{\rightleftarrows}} \mathscr{B}$. There is a one-to-one correspondence between:*

* *(a) adjunctions between $F$ and $G$ (with $F$ on the left and $G$ on the right);*
* *(b) pairs $\left(1_{\mathscr{A}} \xrightarrow{\eta} GF,\ FG \xrightarrow{\varepsilon} 1_{\mathscr{B}}\right)$ of natural transformations satisfying the triangle identities.*

**Proof.** We have shown that every adjunction between $F$ and $G$ gives rise to a pair $(\eta, \varepsilon)$ satisfying the triangle identities. We now show that this process is bijective.

So, take a pair $(\eta, \varepsilon)$ of natural transformations satisfying the triangle identities. For each $A$ and $B$, define functions

$$
\mathscr{B}(F(A), B) \rightleftarrows \mathscr{A}(A, G(B)),
$$

both denoted by a bar, as follows. Given $g \in \mathscr{B}(F(A), B)$, put $\bar{g} = G(g) \circ \eta_A \in \mathscr{A}(A, G(B))$. Similarly, in the opposite direction, put $\bar{f} = \varepsilon_B \circ F(f)$.

These two functions are mutually inverse. Indeed, given $g \colon F(A) \to B$, we have a commutative diagram $\varepsilon_B \circ FG(g) \circ F(\eta_A) = \varepsilon_B \circ F(\bar{g}) = \bar{\bar{g}}$. The composite by one route is $\varepsilon_B \circ FG(g) \circ F(\eta_A) = g$ (by the triangle identities and naturality), so $\bar{\bar{g}} = g$. Dually, $\bar{\bar{f}} = f$ for any map $f \colon A \to G(B)$ in $\mathscr{A}$. $\square$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 2.2.6</span></p>

*Take categories and functors $\mathscr{A} \overset{F}{\underset{G}{\rightleftarrows}} \mathscr{B}$. Then $F \dashv G$ if and only if there exist natural transformations $1 \xrightarrow{\eta} GF$ and $FG \xrightarrow{\varepsilon} 1$ satisfying the triangle identities.* $\square$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.2.7</span><span class="math-callout__name">(Adjunctions between ordered sets)</span></p>

An adjunction between ordered sets consists of order-preserving maps $A \overset{f}{\underset{g}{\rightleftarrows}} B$ such that

$$
\forall a \in A,\ \forall b \in B, \qquad f(a) \le b \iff a \le g(b).
$$

This is because both sides of the isomorphism in the definition of adjunction are sets with at most one element, so they are isomorphic if and only if they are both empty or both nonempty. The naturality requirements hold automatically, since in an ordered set, any two maps with the same domain and codomain are equal.

The unit is the statement that $a \le gf(a)$ for all $a \in A$, and the counit is the statement that $fg(b) \le b$ for all $b \in B$. The triangle identities say nothing, since they assert the equality of two maps in an ordered set with the same domain and codomain.

For instance, let $X$ be a topological space. Take the set $\mathscr{C}(X)$ of closed subsets of $X$ and the set $\mathscr{P}(X)$ of all subsets of $X$, both ordered by $\subseteq$. There are order-preserving maps $\mathscr{P}(X) \overset{\mathrm{Cl}}{\underset{i}{\rightleftarrows}} \mathscr{C}(X)$, where $i$ is the inclusion map and $\mathrm{Cl}$ is closure. This is an adjunction, with $\mathrm{Cl}$ left adjoint to $i$, as witnessed by the fact that $\mathrm{Cl}(A) \subseteq B \iff A \subseteq B$ for all $A \subseteq X$ and closed $B \subseteq X$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.2.8</span></p>

Theorem 2.2.5 states that an adjunction may be regarded as a quadruple $(F, G, \eta, \varepsilon)$ of functors and natural transformations satisfying the triangle identities. An equivalence $(F, G, \eta, \varepsilon)$ of categories (as in Definition 1.3.15) is not necessarily an adjunction. It *is* true that $F$ is left adjoint to $G$, but $\eta$ and $\varepsilon$ are not necessarily the unit and counit (because there is no reason why they should satisfy the triangle identities).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 2.2.9</span><span class="math-callout__name">(String diagrams)</span></p>

There is a way of drawing natural transformations that makes the triangle identities intuitively plausible. Given a natural transformation $\alpha \colon F_4 F_3 F_2 F_1 \to G_2 G_1$, instead of drawing the usual commutative diagram, we can also draw $\alpha$ as a **string diagram**. In this notation, vertical composition of natural transformations corresponds to joining string diagrams together vertically, and horizontal composition corresponds to putting them side by side. The identity on a functor $F$ is drawn as a simple string.

The unit and counit of an adjunction are drawn as cups and caps. The triangle identities then become the topologically plausible equations that a string with a kink can be pulled straight.

</div>

### 2.3 Adjunctions via Initial Objects

We now come to the third formulation of adjointness, which is the one you will probably see most often in everyday mathematics. Consider once more the adjunction $F \dashv U$ between $\mathbf{Set}$ and $\mathbf{Vect}_k$. Let $S$ be a set. The universal property of $F(S)$, the vector space whose basis is $S$, is most commonly stated like this:

> given a vector space $V$, any function $f \colon S \to V$ extends uniquely to a linear map $\bar{f} \colon F(S) \to V$.

In precise language, the statement reads: for any $V \in \mathbf{Vect}_k$ and $f \in \mathbf{Set}(S, U(V))$, there is a unique $\bar{f} \in \mathbf{Vect}_k(F(S), V)$ such that the diagram involving the unit $\eta_S \colon S \to UF(S)$ commutes. In this section, we show that this statement is equivalent to the statement that $F$ is left adjoint to $U$ with unit $\eta$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 2.3.1</span><span class="math-callout__name">(Comma Category)</span></p>

Given categories and functors

$$
\mathscr{A} \xrightarrow{P} \mathscr{C} \xleftarrow{Q} \mathscr{B},
$$

the **comma category** $(P \Rightarrow Q)$ (often written as $(P \downarrow Q)$) is the category defined as follows:

* objects are triples $(A, h, B)$ with $A \in \mathscr{A}$, $B \in \mathscr{B}$, and $h \colon P(A) \to Q(B)$ in $\mathscr{C}$;
* maps $(A, h, B) \to (A', h', B')$ are pairs $(f \colon A \to A',\ g \colon B \to B')$ of maps such that the square

$$
Q(g) \circ h = h' \circ P(f)
$$

commutes.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.3.3</span><span class="math-callout__name">(Slice and coslice categories)</span></p>

Let $\mathscr{A}$ be a category and $A \in \mathscr{A}$. The **slice category** of $\mathscr{A}$ over $A$, denoted by $\mathscr{A}/A$, is the category whose objects are maps into $A$ and whose maps are commutative triangles. More precisely, an object is a pair $(X, h)$ with $X \in \mathscr{A}$ and $h \colon X \to A$, and a map $(X, h) \to (X', h')$ is a map $f \colon X \to X'$ in $\mathscr{A}$ making the triangle $h = h' \circ f$ commute.

Slice categories are a special case of comma categories: $\mathscr{A}/A \cong (1_{\mathscr{A}} \Rightarrow A)$.

Dually (reversing all the arrows), there is a **coslice category** $A/\mathscr{A} \cong (A \Rightarrow 1_{\mathscr{A}})$, whose objects are the maps out of $A$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 2.3.4</span><span class="math-callout__name">(Comma category $(A \Rightarrow G)$)</span></p>

Let $G \colon \mathscr{B} \to \mathscr{A}$ be a functor and let $A \in \mathscr{A}$. We can form the comma category $(A \Rightarrow G)$. Its objects are pairs $(B \in \mathscr{B},\ f \colon A \to G(B))$. A map $(B, f) \to (B', f')$ in $(A \Rightarrow G)$ is a map $q \colon B \to B'$ in $\mathscr{B}$ making the triangle $G(q) \circ f = f'$ commute.

Speaking casually, we say that $f \colon A \to G(B)$ is an object of $(A \Rightarrow G)$, when what we should really say is that the pair $(B, f)$ is an object of $(A \Rightarrow G)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 2.3.5</span><span class="math-callout__name">(Unit map is initial)</span></p>

*Take an adjunction $\mathscr{A} \overset{F}{\underset{G}{\rightleftarrows}} \mathscr{B}$ and an object $A \in \mathscr{A}$. Then the unit map $\eta_A \colon A \to GF(A)$ is an initial object of $(A \Rightarrow G)$.*

**Proof.** Let $(B, f \colon A \to G(B))$ be an object of $(A \Rightarrow G)$. We have to show that there is exactly one map from $(F(A), \eta_A)$ to $(B, f)$.

A map $(F(A), \eta_A) \to (B, f)$ in $(A \Rightarrow G)$ is a map $q \colon F(A) \to B$ in $\mathscr{B}$ such that $G(q) \circ \eta_A = f$. By Lemma 2.2.4, $G(q) \circ \eta_A = \bar{q}$. So the condition is that $\bar{q} = f$, that is, $q = \bar{f}$. Hence there is exactly one such map. $\square$

</div>

Lemma 2.3.5 tells us that being left adjoint to $G$ with unit $\eta$ implies that $\eta_A$ is initial in $(A \Rightarrow G)$ for each $A$. The converse also holds, and this leads to the third characterization of adjointness.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.3.6</span><span class="math-callout__name">(Adjunctions via initial objects)</span></p>

*Let $G \colon \mathscr{B} \to \mathscr{A}$ be a functor. Then $G$ has a left adjoint if and only if for each $A \in \mathscr{A}$, the comma category $(A \Rightarrow G)$ has an initial object.*

**Proof.** "$\Rightarrow$": This is Lemma 2.3.5.

"$\Leftarrow$": Suppose that for each $A \in \mathscr{A}$, the comma category $(A \Rightarrow G)$ has an initial object. Choose one, say $(F(A), \eta_A \colon A \to GF(A))$. This defines a function $F$ on objects of $\mathscr{A}$ and a family of maps $\eta_A$. We need to extend $F$ to a functor and show that $F \dashv G$ with unit $\eta$.

Given a map $p \colon A' \to A$ in $\mathscr{A}$, the composite $\eta_A \circ p \colon A' \to GF(A)$ is an object of $(A' \Rightarrow G)$. Since $\eta_{A'}$ is initial, there is a unique map $F(A') \to F(A)$ in $\mathscr{B}$ making the appropriate triangle commute. Define $F(p)$ to be this map. One checks that this makes $F$ a functor and $\eta$ a natural transformation, and by Theorem 2.2.5, $F \dashv G$. $\square$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 2.3.8</span><span class="math-callout__name">(Three characterizations of adjointness)</span></p>

*Let $\mathscr{A} \overset{F}{\underset{G}{\rightleftarrows}} \mathscr{B}$ be functors. The following are equivalent:*

1. *$F \dashv G$ (i.e., $\mathscr{B}(F(A), B) \cong \mathscr{A}(A, G(B))$ naturally in $A$ and $B$);*
2. *there exist natural transformations $\eta \colon 1_{\mathscr{A}} \to GF$ and $\varepsilon \colon FG \to 1_{\mathscr{B}}$ satisfying the triangle identities;*
3. *for each $A \in \mathscr{A}$, the object $\eta_A \colon A \to GF(A)$ is initial in $(A \Rightarrow G)$.* $\square$

</div>

## Chapter 3: Interlude on Sets

Sets and functions are ubiquitous in mathematics. Category theory is often used to shed light on common constructions in mathematics. If we hope to do this in an advanced context, we must begin by settling the basic notions of set and function.

### 3.1 Constructions with Sets

We have made no definition of "set," nor of "function." Nevertheless, guided by our intuition, we can list some properties that we expect the world of sets and functions to have. Intuitively, a set is a bag of featureless points, and a function $A \to B$ is an assignment of a point in $B$ to each point in $A$. Composition of functions is associative and there is an identity function on every set. Hence:

> *Sets and functions form a category, denoted by $\mathbf{Set}$.*

Let us list some of the special features of the category of sets.

#### The Empty Set

There is a set $\emptyset$ with no elements. For any set $B$, there is exactly one function from $\emptyset$ to $B$ (specified by "doing nothing"), and there cannot be two different ways to do nothing. Hence $\emptyset$ is an initial object of $\mathbf{Set}$.

On the other hand, for a nonempty set $A$, there are *no* functions $A \to \emptyset$, because there is nowhere for elements of $A$ to go.

#### The One-Element Set

There is a set $1$ with exactly one element. For any set $A$, there is exactly one function from $A$ to $1$, since every element of $A$ must be mapped to the unique element of $1$. That is: $1$ is a terminal object of $\mathbf{Set}$.

A function *from* $1$ *to* a set $B$ is just a choice of an element of $B$. In short, the functions $1 \to B$ are the elements of $B$. Hence:

> *The concept of element is a special case of the concept of function.*

#### Products

Any two sets $A$ and $B$ have a product $A \times B$, whose elements are the ordered pairs $(a, b)$ with $a \in A$ and $b \in B$. More generally, take any set $I$ and any family $(A_i)_{i \in I}$ of sets. There is a product set $\prod_{i \in I} A_i$, whose elements are families $(a_i)_{i \in I}$ with $a_i \in A_i$ for each $i$.

#### Sums

Any two sets $A$ and $B$ have a **sum** $A + B$, obtained by putting all the points into one big bag. If $A$ and $B$ are finite sets with $m$ and $n$ elements respectively, then $A + B$ always has $m + n$ elements. There are inclusion functions $A \xrightarrow{i} A + B \xleftarrow{j} B$.

Sum is sometimes called **disjoint union** and written as $\coprod$. It is not to be confused with ordinary union $\cup$: we can take the sum of *any* two sets $A$ and $B$, whereas $A \cup B$ only makes sense when $A$ and $B$ come as subsets of some larger set. More generally, any family $(A_i)_{i \in I}$ of sets has a sum $\sum_{i \in I} A_i$.

#### Sets of Functions

For any two sets $A$ and $B$, we can form the set $A^B$ of functions from $B$ to $A$. This is a special case of the product construction: $A^B$ is the product $\prod_{b \in B} A$ of the constant family $(A)_{b \in B}$. Indeed, an element of $\prod_{b \in B} A$ is a family $(a_b)_{b \in B}$ consisting of one element $a_b \in A$ for each $b \in B$; in other words, it is a function $B \to A$.

#### Arithmetic of Sets

We are using notation reminiscent of arithmetic: $A \times B$, $A + B$, and $A^B$. There is good reason for this: if $A$ is a finite set with $m$ elements and $B$ a finite set with $n$ elements, then $A \times B$ has $m \times n$ elements, $A + B$ has $m + n$ elements, and $A^B$ has $m^n$ elements. All the usual laws of arithmetic have their counterparts for sets:

$$
A \times (B + C) \cong (A \times B) + (A \times C), \qquad A^{B+C} \cong A^B \times A^C, \qquad (A^B)^C \cong A^{B \times C},
$$

and so on, where $\cong$ is isomorphism in $\mathbf{Set}$. These isomorphisms hold for all sets, not just finite ones.

#### The Two-Element Set

Let $2$ be the set $1 + 1$ (a set with two elements, written as $\mathtt{true}$ and $\mathtt{false}$). Given a subset $S \subseteq A$, we obtain the **characteristic function** $\chi_S \colon A \to 2$ defined by

$$
\chi_S(a) = \begin{cases} \mathtt{true} & \text{if } a \in S, \\ \mathtt{false} & \text{if } a \notin S. \end{cases}
$$

Conversely, given a function $f \colon A \to 2$, we obtain a subset $f^{-1}\lbrace\mathtt{true}\rbrace = \lbrace a \in A \mid f(a) = \mathtt{true}\rbrace$ of $A$. These two processes are mutually inverse. Hence:

> *Subsets of $A$ correspond one-to-one with functions $A \to 2$.*

When we think of $2^A$ as the set of all subsets of $A$, we call it the **power set** $\mathscr{P}(A)$.

#### Equalizers

Given sets and functions $A \overset{f}{\underset{g}{\rightrightarrows}} B$, there is a set

$$
\lbrace a \in A \mid f(a) = g(a) \rbrace.
$$

This set is called the **equalizer** of $f$ and $g$, since it is the part of $A$ on which the two functions are equal.

#### Quotients

Let $A$ be a set and $\sim$ an equivalence relation on $A$. There is a set $A/{\sim}$, the **quotient of $A$ by $\sim$**, whose elements are the equivalence classes. There is also a canonical map $p \colon A \to A/{\sim}$, sending an element to its equivalence class. It is surjective, and has the property that $p(a) = p(a') \iff a \sim a'$. In fact, it has a universal property: any function $f \colon A \to B$ such that $a \sim a' \implies f(a) = f(a')$ factorizes uniquely through $p$.

Thus, for any set $B$, the functions $A/{\sim} \to B$ correspond one-to-one with the functions $f \colon A \to B$ satisfying $a \sim a' \implies f(a) = f(a')$. This fact is at the heart of the famous isomorphism theorems of algebra.

#### Natural Numbers

A function with domain $\mathbb{N}$ is usually called a **sequence**. A crucial property of $\mathbb{N}$ is that some sequences can be defined recursively: given a set $X$, an element $a \in X$, and a function $r \colon X \to X$, there is a unique sequence $(x_n)_{n=0}^{\infty}$ of elements of $X$ such that

$$
x_0 = a, \qquad x_{n+1} = r(x_n) \text{ for all } n \in \mathbb{N}.
$$

This is a *universal* property of $\mathbb{N}$.

#### Choice

Let $f \colon A \to B$ be a map in a category $\mathscr{A}$. A **section** (or **right inverse**) of $f$ is a map $i \colon B \to A$ in $\mathscr{A}$ such that $f \circ i = 1_B$. In the category of sets, any map with a section is certainly surjective. The converse statement is called the **axiom of choice**:

> *Every surjection has a section.*

It is called "choice" because specifying a section of $f \colon A \to B$ amounts to choosing, for each $b \in B$, an element of the nonempty set $\lbrace a \in A \mid f(a) = b \rbrace$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Foundations)</span></p>

The properties listed above are not theorems, since we do not have rigorous definitions of set and function. Instead of *defining* a set to be a such-and-such and a function to be a such-and-such else, we list some *properties* that we assume sets and functions to have. In other words, we never attempt to say what sets and functions *are*; we just say what you can *do* with them. What we are doing is laying *foundations*: the foundation consists of the basic concepts (set and function), which are not built on anything else, but are assumed to satisfy a stated list of properties. On top of the foundations are built further definitions and theorems, and so on, towering upwards.

</div>

### 3.2 Small and Large Categories

We have now made some assumptions about the nature of sets. One consequence of those assumptions is that in many of the categories we have met, the collection of all objects is too large to form a set. In fact, even the collection of *isomorphism classes* of objects is often too large to form a set.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Cardinality)</span></p>

Given sets $A$ and $B$, write $\lvert A \rvert \le \lvert B \rvert$ (or $\lvert B \rvert \ge \lvert A \rvert$) if there exists an injection $A \to B$. We give no meaning to the expression "$\lvert A \rvert$" in isolation. Since identity maps are injective and the composite of two injections is an injection, we have $\lvert A \rvert \le \lvert A \rvert$ for all sets $A$, and $\lvert A \rvert \le \lvert B \rvert \le \lvert C \rvert \implies \lvert A \rvert \le \lvert C \rvert$.

Also, if $A \cong B$ then $\lvert A \rvert \le \lvert B \rvert \le \lvert A \rvert$. We write $\lvert A \rvert = \lvert B \rvert$ and say that $A$ and $B$ **have the same cardinality** if $A \cong B$, or equivalently if $\lvert A \rvert \le \lvert B \rvert \le \lvert A \rvert$. We write $\lvert A \rvert < \lvert B \rvert$ if $\lvert A \rvert \le \lvert B \rvert$ and $\lvert A \rvert \ne \lvert B \rvert$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.2.1</span><span class="math-callout__name">(Cantor--Bernstein)</span></p>

*Let $A$ and $B$ be sets. If $\lvert A \rvert \le \lvert B \rvert \le \lvert A \rvert$ then $A \cong B$.* $\square$

</div>

These observations tell us that $\le$ is a preorder on the collection of all sets. It is not a genuine order, since $\lvert A \rvert \le \lvert B \rvert \le \lvert A \rvert$ only implies $A \cong B$, not $A = B$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 3.2.2</span><span class="math-callout__name">(Cantor)</span></p>

*Let $A$ be a set. Then $\lvert A \rvert < \lvert \mathscr{P}(A) \rvert$.*

The lemma is easy for finite sets, since if $A$ has $n$ elements then $\mathscr{P}(A)$ has $2^n$ elements, and $n < 2^n$. $\square$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 3.2.3</span></p>

*For every set $A$, there is a set $B$ such that $\lvert A \rvert < \lvert B \rvert$.* $\square$

In other words, there is no biggest set.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.2.4</span></p>

*Let $I$ be a set, and let $(A_i)_{i \in I}$ be a family of sets. Then there exists a set not isomorphic to any of the sets $A_i$.*

**Proof.** Put $A = \mathscr{P}\!\left(\sum_{i \in I} A_i\right)$. For each $j \in I$, we have the inclusion function $A_j \to \sum_{i \in I} A_i$, so by Theorem 3.2.2, $\lvert A_j \rvert \le \left\lvert \sum_{i \in I} A_i \right\rvert < \lvert A \rvert$. Hence $A_j \not\cong A$. $\square$

</div>

We use the word **class** informally to mean any collection of mathematical objects. All sets are classes, but some classes (such as the class of all sets) are too big to be sets. A class will be called **small** if it is a set, and **large** otherwise. For example, Proposition 3.2.4 states that the class of isomorphism classes of sets is large. The crucial point is:

> *Any individual set is small, but the class of sets is large.*

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Small, large, and locally small categories)</span></p>

A category $\mathscr{A}$ is **small** if the class or collection of all maps in $\mathscr{A}$ is small, and **large** otherwise. If $\mathscr{A}$ is small then the class of objects of $\mathscr{A}$ is small too, since objects correspond one-to-one with identity maps.

A category $\mathscr{A}$ is **locally small** if for each $A, B \in \mathscr{A}$, the class $\mathscr{A}(A, B)$ is small. (So, small implies locally small.) Many authors take local smallness to be part of the definition of category. The class $\mathscr{A}(A, B)$ is often called the **hom-set** from $A$ to $B$, although strictly speaking, we should only call it this when $\mathscr{A}$ is locally small.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.2.5</span></p>

$\mathbf{Set}$ is locally small, because for any two sets $A$ and $B$, the functions from $A$ to $B$ form a set.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.2.6</span></p>

$\mathbf{Vect}_k$, $\mathbf{Grp}$, $\mathbf{Ab}$, $\mathbf{Ring}$ and $\mathbf{Top}$ are all locally small. For example, given rings $A$ and $B$, a homomorphism from $A$ to $B$ is a function from $A$ to $B$ with certain properties, and the collection of all functions from $A$ to $B$ is small, so the collection of homomorphisms from $A$ to $B$ is certainly small.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Essentially small)</span></p>

A category is **essentially small** if it is equivalent to some small category. For example, the category of finite sets is essentially small, since by Example 1.3.20, it is equivalent to the small category $\mathscr{B}$ whose objects are the natural numbers $\mathbf{0}, \mathbf{1}, \ldots$ (one chosen set of each cardinality).

If two categories $\mathscr{A}$ and $\mathscr{B}$ are equivalent, the class of isomorphism classes of objects of $\mathscr{A}$ is in bijection with that of $\mathscr{B}$. In a small category, the class of objects is small, so the class of isomorphism classes of objects is certainly small. Hence in an essentially small category, the class of isomorphism classes of objects is small.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 3.2.8</span></p>

*$\mathbf{Set}$ is not essentially small.*

**Proof.** Proposition 3.2.4 states that the class of isomorphism classes of sets is large. The result follows. $\square$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.2.9</span></p>

For any field $k$, the category $\mathbf{Vect}_k$ of vector spaces over $k$ is not essentially small. It is enough to prove that the class of isomorphism classes of vector spaces is large. For any set $I$ and family $(V_i)_{i \in I}$ of vector spaces, the set $S = \mathscr{P}\!\left(\sum_{i \in I} U(V_i)\right)$ has the property that $\lvert U(V_i) \rvert < \lvert S \rvert$ for all $i \in I$. The free vector space $F(S)$ on $S$ contains a copy of $S$ as a basis, so $\lvert S \rvert \le \lvert UF(S) \rvert$. Hence $\lvert U(V_i) \rvert < \lvert UF(S) \rvert$ for all $i$, and so $F(S) \not\cong V_i$ for all $i$, as required. Similarly, none of the categories $\mathbf{Grp}$, $\mathbf{Ab}$, $\mathbf{Ring}$ and $\mathbf{Top}$ is essentially small.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 3.2.10</span><span class="math-callout__name">(Cat)</span></p>

We denote by $\mathbf{Cat}$ the category of small categories and functors between them.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 3.2.11</span></p>

Monoids are by definition *sets* equipped with certain structure, so the one-object categories that they correspond to are small. Let $\mathscr{M}$ be the full subcategory of $\mathbf{Cat}$ consisting of the one-object categories. Then there is an equivalence of categories $\mathbf{Mon} \simeq \mathscr{M}$. This is proved by the argument in Example 1.3.21, noting that because each object of $\mathscr{M}$ is a *small* one-object category, the collection of maps from the single object to itself really is a set.

</div>

### 3.3 Historical Remarks

The set theory we have been developing is rather different from what many mathematicians think of as set theory (ZFC). Here is a brief comparison.

#### Types

The concept of **type** is fundamental. The square root of 2 is a real number, $\mathbb{Q}$ is a field, $S_3$ is a group, $\log$ is a function from $(0, \infty)$ to $\mathbb{R}$, and $\frac{d}{dx}$ is an operation that takes a function $\mathbb{R} \to \mathbb{R}$ and produces another such function. We all have an inbuilt sense of type, and it would not usually occur to us to ask whether two things of different type were equal.

#### Membership-Based Set Theory (ZFC)

Those who came after Cantor sought to compile a definitive axiomatization of set theory. The list they arrived at, known as ZFC (Zermelo--Fraenkel with Choice), takes sets and *membership* as basic concepts. Several features of this approach are problematic from a modern standpoint:

* In ZFC, *everything* is a set. A function is a set, the number $\sqrt{2}$ is a set, $\log$ is a set, and so on. This loses the fundamental notion of type.
* The elements of sets are always sets too. But in ordinary mathematics, real numbers themselves are not regarded as sets.
* Membership is a *global* relation: for *any* two sets $A$ and $B$, it makes sense to ask whether $A \in B$. This permits nonsensical questions such as "is $\mathbb{Q} \in \sqrt{2}$?"

#### Categorical Set Theory

Taking sets and *functions* (rather than sets and membership) as the basic concepts leads to a theory containing all of the meaningful results of Cantor and others, but with none of the aspects of ZFC that seem so remote from the rest of mathematics. In particular, the function-based approach respects the fundamental notion of type.

Objects are understood through their place in the ambient category. We get inside an object by probing it with maps to or from other objects. For example, an element of a set $A$ is a map $1 \to A$, and a subset of $A$ is a map $A \to 2$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Categorical axioms for set theory)</span></p>

One popular choice of categorical axioms for set theory can be summarized informally as follows:

1. Composition of functions is associative and has identities.
2. There is a terminal set.
3. There is a set with no elements.
4. A function is determined by its effect on elements.
5. Given sets $A$ and $B$, one can form their product $A \times B$.
6. Given sets $A$ and $B$, one can form the set of functions from $A$ to $B$.
7. Given $f \colon A \to B$ and $b \in B$, one can form the inverse image $f^{-1}\lbrace b \rbrace$.
8. The subsets of a set $A$ correspond to the functions from $A$ to $\lbrace 0, 1 \rbrace$.
9. The natural numbers form a set.
10. Every surjection has a section.

All ten together can be expressed in categorical jargon as "sets and functions form a well-pointed topos with natural numbers object and choice." But in order to state the axioms, it is not *necessary* to appeal to any general notion of category. They can be expressed directly in terms of sets and functions.

</div>

## Chapter 4: Representables

A category is a world of objects, all looking at one another. Each sees the world from a different viewpoint. This chapter explores the theme of how each object sees and is seen by the category in which it lives. We are naturally led to the notion of representable functor, which (after adjunctions) provides our second approach to the idea of universal property.

Consider, for instance, the category of topological spaces. A map from $1$ to a space $X$ is essentially a point of $X$, so $1$ "sees points." A map from $\mathbb{R}$ to $X$ could reasonably be called a curve in $X$, so $\mathbb{R}$ "sees curves." In the category of groups, a map from $\mathbb{Z}$ to a group $G$ amounts to an element of $G$, so $\mathbb{Z}$ "sees elements." In the category of fields, a map $K \to L$ is a way of realizing $L$ as an extension of $K$, so each field sees the extensions of itself. In the ordered set $(\mathbb{R}, \le)$, the object $0$ sees whether a number is nonnegative.

### 4.1 Definitions and Examples

Fix an object $A$ of a category $\mathscr{A}$. We will consider the totality of maps out of $A$. To each $B \in \mathscr{A}$, there is assigned the set (or class) $\mathscr{A}(A, B)$ of maps from $A$ to $B$. This assignation is functorial in $B$: any map $B \to B'$ induces a function $\mathscr{A}(A, B) \to \mathscr{A}(A, B')$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.1.1</span><span class="math-callout__name">(Covariant representable functor)</span></p>

Let $\mathscr{A}$ be a locally small category and $A \in \mathscr{A}$. We define a functor

$$
H^A = \mathscr{A}(A, -) \colon \mathscr{A} \to \mathbf{Set}
$$

as follows:

* for objects $B \in \mathscr{A}$, put $H^A(B) = \mathscr{A}(A, B)$;
* for maps $B \xrightarrow{g} B'$ in $\mathscr{A}$, define

$$
H^A(g) = \mathscr{A}(A, g) \colon \mathscr{A}(A, B) \to \mathscr{A}(A, B')
$$

by $p \mapsto g \circ p$ for all $p \colon A \to B$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remarks 4.1.2</span></p>

**(a)** Recall that "locally small" means that each class $\mathscr{A}(A, B)$ is in fact a set. This hypothesis is clearly necessary in order for the definition to make sense.

**(b)** Sometimes $H^A(g)$ is written as $g \circ -$ or $g_\*$. All three forms, as well as $\mathscr{A}(A, g)$, are in use.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.1.3</span><span class="math-callout__name">(Representable functor)</span></p>

Let $\mathscr{A}$ be a locally small category. A functor $X \colon \mathscr{A} \to \mathbf{Set}$ is **representable** if $X \cong H^A$ for some $A \in \mathscr{A}$. A **representation** of $X$ is a choice of an object $A \in \mathscr{A}$ and an isomorphism between $H^A$ and $X$.

Representable functors are sometimes just called "representables." Only set-valued functors (that is, functors with codomain $\mathbf{Set}$) can be representable.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.1.4</span><span class="math-callout__name">(Identity functor on $\mathbf{Set}$)</span></p>

Consider $H^1 \colon \mathbf{Set} \to \mathbf{Set}$, where $1$ is the one-element set. Since a map from $1$ to a set $B$ amounts to an element of $B$, we have $H^1(B) \cong B$ for each $B \in \mathbf{Set}$. It is easily verified that this isomorphism is natural in $B$, so $H^1$ is isomorphic to the identity functor $1_{\mathbf{Set}}$. Hence $1_{\mathbf{Set}}$ is representable.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.1.5</span><span class="math-callout__name">(Forgetful functors)</span></p>

All of the "seeing" functors in the introduction to this chapter are representable. The forgetful functor $\mathbf{Top} \to \mathbf{Set}$ is isomorphic to $H^1 = \mathbf{Top}(1, -)$, and the forgetful functor $\mathbf{Grp} \to \mathbf{Set}$ is isomorphic to $\mathbf{Grp}(\mathbb{Z}, -)$. For each prime $p$, there is a functor $U_p \colon \mathbf{Grp} \to \mathbf{Set}$ defined on objects by

$$
U_p(G) = \lbrace \text{elements of } G \text{ of order } 1 \text{ or } p \rbrace,
$$

and $U_p \cong \mathbf{Grp}(\mathbb{Z}/p\mathbb{Z}, -)$. Hence $U_p$ is representable.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.1.6</span><span class="math-callout__name">(Objects functor on $\mathbf{Cat}$)</span></p>

There is a functor $\mathrm{ob} \colon \mathbf{Cat} \to \mathbf{Set}$ sending a small category to its set of objects. It is representable. Indeed, consider the terminal category $\mathbf{1}$ (with one object and only the identity map). A functor from $\mathbf{1}$ to a category $\mathscr{B}$ simply picks out an object of $\mathscr{B}$. Thus, $H^{\mathbf{1}}(\mathscr{B}) \cong \mathrm{ob}\,\mathscr{B}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.1.7</span><span class="math-callout__name">(Monoid as one-object category)</span></p>

Let $M$ be a monoid, regarded as a one-object category. A set-valued functor on $M$ is just an $M$-set. Since the category $M$ has only one object, there is only one representable functor on it (up to isomorphism). As an $M$-set, the unique representable is the so-called **left regular representation** of $M$, that is, the underlying set of $M$ acted on by multiplication on the left.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.1.8</span><span class="math-callout__name">(Fundamental group)</span></p>

Let $\mathbf{Toph}_\*$ be the category whose objects are topological spaces equipped with a basepoint and whose arrows are homotopy classes of basepoint-preserving continuous maps. Let $S^1 \in \mathbf{Toph}_\*$ be the circle. Then for any object $X \in \mathbf{Toph}_\*$, the maps $S^1 \to X$ in $\mathbf{Toph}_\*$ are the elements of the fundamental group $\pi_1(X)$. Formally, the composite functor

$$
\mathbf{Toph}\_\* \xrightarrow{\pi_1} \mathbf{Grp} \xrightarrow{U} \mathbf{Set}
$$

is isomorphic to $\mathbf{Toph}\_\*(S^1, -)$. In particular, it is representable.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.1.9</span><span class="math-callout__name">(Tensor product and bilinear maps)</span></p>

Fix a field $k$ and vector spaces $U$ and $V$ over $k$. There is a functor

$$
\mathbf{Bilin}(U, V; -) \colon \mathbf{Vect}_k \to \mathbf{Set}
$$

whose value $\mathbf{Bilin}(U, V; W)$ at $W \in \mathbf{Vect}_k$ is the set of bilinear maps $U \times V \to W$. It can be shown that this functor is representable; in other words, there is a space $T$ with the property that

$$
\mathbf{Bilin}(U, V; W) \cong \mathbf{Vect}_k(T, W)
$$

naturally in $W$. This $T$ is the tensor product $U \otimes V$, which we met just after the proof of Lemma 0.7.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 4.1.10</span><span class="math-callout__name">(Adjunctions and representability)</span></p>

*Let $\mathscr{A} \overset{F}{\underset{G}{\rightleftarrows}} \mathscr{B}$ be an adjunction between locally small categories, and let $A \in \mathscr{A}$. Then the functor*

$$
\mathscr{A}(A, G(-)) \colon \mathscr{B} \to \mathbf{Set}
$$

*(that is, the composite $\mathscr{B} \xrightarrow{G} \mathscr{A} \xrightarrow{H^A} \mathbf{Set}$) is representable.*

**Proof.** We have $\mathscr{A}(A, G(B)) \cong \mathscr{B}(F(A), B)$ for each $B \in \mathscr{B}$. If we can show that this isomorphism is natural in $B$, then we will have proved that $\mathscr{A}(A, G(-))$ is isomorphic to $H^{F(A)}$ and is therefore representable. The naturality follows from the naturality condition in the definition of adjunction. $\square$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 4.1.11</span><span class="math-callout__name">(Left adjoints give representability)</span></p>

*Any set-valued functor with a left adjoint is representable.*

**Proof.** Let $G \colon \mathscr{A} \to \mathbf{Set}$ be a functor with a left adjoint $F$. Write $1$ for the one-point set. Then

$$
G(A) \cong \mathbf{Set}(1, G(A))
$$

naturally in $A \in \mathscr{A}$ (by Example 4.1.4), that is, $G \cong \mathbf{Set}(1, G(-))$. So by Lemma 4.1.10, $G$ is representable; indeed, $G \cong H^{F(1)}$. $\square$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.1.12</span><span class="math-callout__name">(Forgetful functor on $\mathbf{Top}$)</span></p>

Several of the examples of representables mentioned above arise as in Proposition 4.1.11. For instance, $U \colon \mathbf{Top} \to \mathbf{Set}$ has a left adjoint $D$ (Example 2.1.5), and $D(1) \cong 1$, so we recover the result that $U \cong H^1$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.1.13</span><span class="math-callout__name">(Forgetful functor on $\mathbf{Vect}_k$)</span></p>

The forgetful functor $U \colon \mathbf{Vect}_k \to \mathbf{Set}$ is representable, since it has a left adjoint. If $F$ denotes the left adjoint then $F(1)$ is the $1$-dimensional vector space $k$, so $U \cong H^k$. This is also easy to see directly: a map from $k$ to a vector space $V$ is uniquely determined by the image of $1$, which can be any element of $V$; hence $\mathbf{Vect}_k(k, V) \cong U(V)$ naturally in $V$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.1.14</span><span class="math-callout__name">(Forgetful functor on $\mathbf{CRing}$)</span></p>

Forgetful functors between categories of algebraic structures usually have left adjoints. Take the category $\mathbf{CRing}$ of commutative rings and the forgetful functor $U \colon \mathbf{CRing} \to \mathbf{Set}$. This general principle suggests that $U$ has a left adjoint, and Proposition 4.1.11 then tells us that $U$ is representable.

Given a set $S$, let $\mathbb{Z}[S]$ be the ring of polynomials over $\mathbb{Z}$ in commuting variables $x_s$ ($s \in S$). Then $S \mapsto \mathbb{Z}[S]$ defines a functor $\mathbf{Set} \to \mathbf{CRing}$, and this is left adjoint to $U$. Hence $U \cong H^{\mathbb{Z}[x]}$. Again, this can be verified directly: for any ring $R$, the maps $\mathbb{Z}[x] \to R$ correspond one-to-one with the elements of $R$.

</div>

We have defined, for each object $A$ of our category $\mathscr{A}$, a functor $H^A \in [\mathscr{A}, \mathbf{Set}]$. This describes how $A$ sees the world. As $A$ varies, the view varies. On the other hand, it is always the same world being seen, so the different views from different objects are somehow related.

Precisely, a map $A' \xrightarrow{f} A$ induces a natural transformation $H^A \Rightarrow H^{A'}$, whose $B$-component (for $B \in \mathscr{A}$) is the function

$$
H^A(B) = \mathscr{A}(A, B) \to H^{A'}(B) = \mathscr{A}(A', B), \qquad p \mapsto p \circ f.
$$

Note the reversal of direction! Each functor $H^A$ is covariant, but they come together to form a *contravariant* functor, as in the following definition.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.1.15</span><span class="math-callout__name">(Contravariant Yoneda functor $H^\bullet$)</span></p>

Let $\mathscr{A}$ be a locally small category. The functor

$$
H^\bullet \colon \mathscr{A}^{\mathrm{op}} \to [\mathscr{A}, \mathbf{Set}]
$$

is defined on objects $A$ by $H^\bullet(A) = H^A$ and on maps $f$ by $H^\bullet(f) = H^f$.

The symbol $\bullet$ is another type of blank, like $-$.

</div>

All of the definitions presented so far in this chapter can be dualized. At the formal level, this is trivial: reverse all the arrows, so that every $\mathscr{A}$ becomes an $\mathscr{A}^{\mathrm{op}}$ and vice versa. But in our usual examples, the flavour is different. We are no longer asking what objects *see*, but how they are *seen*.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.1.16</span><span class="math-callout__name">(Contravariant representable functor)</span></p>

Let $\mathscr{A}$ be a locally small category and $A \in \mathscr{A}$. We define a functor

$$
H_A = \mathscr{A}(-, A) \colon \mathscr{A}^{\mathrm{op}} \to \mathbf{Set}
$$

as follows:

* for objects $B \in \mathscr{A}$, put $H_A(B) = \mathscr{A}(B, A)$;
* for maps $B' \xrightarrow{g} B$ in $\mathscr{A}$, define

$$
H_A(g) = \mathscr{A}(g, A) = g^\* = - \circ g \colon \mathscr{A}(B, A) \to \mathscr{A}(B', A)
$$

by $p \mapsto p \circ g$ for all $p \colon B \to A$.

</div>

If you know about dual vector spaces, this construction will seem familiar. In particular, a map $B' \to B$ induces a map in the opposite direction, $H_A(B) \to H_A(B')$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.1.17</span><span class="math-callout__name">(Representable contravariant functor)</span></p>

Let $\mathscr{A}$ be a locally small category. A functor $X \colon \mathscr{A}^{\mathrm{op}} \to \mathbf{Set}$ is **representable** if $X \cong H_A$ for some $A \in \mathscr{A}$. A **representation** is a choice of an object $A \in \mathscr{A}$ and an isomorphism between $H_A$ and $X$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.1.18</span><span class="math-callout__name">(Power set functor)</span></p>

There is a functor $\mathscr{P} \colon \mathbf{Set}^{\mathrm{op}} \to \mathbf{Set}$ sending each set $B$ to its power set $\mathscr{P}(B)$, and defined on maps $g \colon B' \to B$ by $(\mathscr{P}(g))(U) = g^{-1}U$ for all $U \in \mathscr{P}(B)$. As we saw in Section 3.1, a subset amounts to a map into the two-point set $2$. Precisely put, $\mathscr{P} \cong H_2$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.1.19</span><span class="math-callout__name">(Open subsets functor)</span></p>

Similarly, there is a functor $\mathscr{O} \colon \mathbf{Top}^{\mathrm{op}} \to \mathbf{Set}$ defined on objects $B$ by taking $\mathscr{O}(B)$ to be the set of open subsets of $B$. If $S$ denotes the two-point topological space in which exactly one of the two singleton subsets is open, then continuous maps from a space $B$ into $S$ correspond naturally to open subsets of $B$. Hence $\mathscr{O} \cong H_S$, and $\mathscr{O}$ is representable.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.1.20</span><span class="math-callout__name">(Ring of continuous functions)</span></p>

In Example 1.2.11, we defined a functor $C \colon \mathbf{Top}^{\mathrm{op}} \to \mathbf{Ring}$, assigning to each space the ring of continuous real-valued functions on it. The composite functor

$$
\mathbf{Top}^{\mathrm{op}} \xrightarrow{C} \mathbf{Ring} \xrightarrow{U} \mathbf{Set}
$$

is representable, since by definition, $U(C(X)) = \mathbf{Top}(X, \mathbb{R})$ for topological spaces $X$.

</div>

Previously, we assembled the covariant representables $(H^A)\_{A \in \mathscr{A}}$ into one big functor $H^\bullet$. We now do the same for the contravariant representables $(H_A)\_{A \in \mathscr{A}}$.

Any map $A \xrightarrow{f} A'$ in $\mathscr{A}$ induces a natural transformation $H_A \Rightarrow H_{A'}$, whose component at an object $B \in \mathscr{A}$ is

$$
H_A(B) = \mathscr{A}(B, A) \to H_{A'}(B) = \mathscr{A}(B, A'), \qquad p \mapsto f \circ p.
$$

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.1.21</span><span class="math-callout__name">(Yoneda embedding)</span></p>

Let $\mathscr{A}$ be a locally small category. The **Yoneda embedding** of $\mathscr{A}$ is the functor

$$
H\_\bullet \colon \mathscr{A} \to [\mathscr{A}^{\mathrm{op}}, \mathbf{Set}]
$$

defined on objects $A$ by $H\_\bullet(A) = H_A$ and on maps $f$ by $H\_\bullet(f) = H_f$.

</div>

Here is a summary of the definitions so far:

* For each $A \in \mathscr{A}$, we have a functor $\mathscr{A} \xrightarrow{H^A} \mathbf{Set}$.
* Putting them all together gives a functor $\mathscr{A}^{\mathrm{op}} \xrightarrow{H^\bullet} [\mathscr{A}, \mathbf{Set}]$.
* For each $A \in \mathscr{A}$, we have a functor $\mathscr{A}^{\mathrm{op}} \xrightarrow{H_A} \mathbf{Set}$.
* Putting them all together gives a functor $\mathscr{A} \xrightarrow{H\_\bullet} [\mathscr{A}^{\mathrm{op}}, \mathbf{Set}]$.

The second pair of functors is the dual of the first. In the theory of representable functors, it does not make much difference whether we work with the first or the second pair. We choose to work with the second pair, the $H_A$s and $H\_\bullet$. In a sense to be explained, $H\_\bullet$ "embeds" $\mathscr{A}$ into $[\mathscr{A}^{\mathrm{op}}, \mathbf{Set}]$. This can be useful, because the category $[\mathscr{A}^{\mathrm{op}}, \mathbf{Set}]$ has some good properties that $\mathscr{A}$ might not have.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.1.22</span><span class="math-callout__name">(Hom functor)</span></p>

Let $\mathscr{A}$ be a locally small category. The functor

$$
\mathrm{Hom}\_{\mathscr{A}} \colon \mathscr{A}^{\mathrm{op}} \times \mathscr{A} \to \mathbf{Set}
$$

is defined by

$$
(A, B) \mapsto \mathscr{A}(A, B), \qquad (f, g) \mapsto g \circ - \circ f.
$$

In other words, $\mathrm{Hom}\_{\mathscr{A}}(A, B) = \mathscr{A}(A, B)$ and $(\mathrm{Hom}\_{\mathscr{A}}(f, g))(p) = g \circ p \circ f$ whenever $A' \xrightarrow{f} A \xrightarrow{p} B \xrightarrow{g} B'$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remarks 4.1.23</span></p>

**(a)** The existence of the functor $\mathrm{Hom}\_{\mathscr{A}}$ is something like the fact that for a metric space $(X, d)$, the metric is itself a continuous map $d \colon X \times X \to \mathbb{R}$.

**(b)** $\mathrm{Hom}\_{\mathscr{A}}$ is the functor $\mathscr{A}^{\mathrm{op}} \times \mathscr{A} \to \mathbf{Set}$ corresponding to the families of functors $(H^A)\_{A \in \mathscr{A}}$ and $(H_B)\_{B \in \mathscr{A}}$.

**(c)** For any category $\mathscr{B}$, there is an adjunction $(- \times \mathscr{B}) \dashv [\mathscr{B}, -]$ of functors $\mathbf{CAT} \to \mathbf{CAT}$; in other words, there is a canonical bijection $\mathbf{CAT}(\mathscr{A} \times \mathscr{B}, \mathscr{C}) \cong \mathbf{CAT}(\mathscr{A}, [\mathscr{B}, \mathscr{C}])$. Under this bijection, the functors $\mathrm{Hom}\_{\mathscr{A}}$ and $H^\bullet$ (or $H\_\bullet$) correspond to one another. Thus, $\mathrm{Hom}\_{\mathscr{A}}$ carries the same information as $H^\bullet$ (or $H\_\bullet$), presented slightly differently.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 4.1.24</span><span class="math-callout__name">(Naturality in the definition of adjunction)</span></p>

We can now explain the naturality in the definition of adjunction (Definition 2.1.1). Take categories and functors $\mathscr{A} \overset{F}{\underset{G}{\rightleftarrows}} \mathscr{B}$. They give rise to functors

$$
\mathscr{A}^{\mathrm{op}} \times \mathscr{B} \xrightarrow{1 \times G} \mathscr{A}^{\mathrm{op}} \times \mathscr{A} \xrightarrow{\mathrm{Hom}\_{\mathscr{A}}} \mathbf{Set}
$$

and

$$
\mathscr{A}^{\mathrm{op}} \times \mathscr{B} \xrightarrow{F^{\mathrm{op}} \times 1} \mathscr{B}^{\mathrm{op}} \times \mathscr{B} \xrightarrow{\mathrm{Hom}\_{\mathscr{B}}} \mathbf{Set}.
$$

The composite going down sends $(A, B)$ to $\mathscr{B}(F(A), B)$; it can be written as $\mathscr{B}(F(-), -)$. The composite going right sends $(A, B)$ to $\mathscr{A}(A, G(B))$; it can be written as $\mathscr{A}(-, G(-))$. These two functors $\mathscr{A}^{\mathrm{op}} \times \mathscr{B} \to \mathbf{Set}$ are naturally isomorphic if and only if $F$ and $G$ are adjoint. This justifies the naturality requirements (2.2) and (2.3) in the definition of adjunction.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 4.1.25</span><span class="math-callout__name">(Generalized element)</span></p>

Let $A$ be an object of a category. A **generalized element** of $A$ is a map with codomain $A$. A map $S \to A$ is a generalized element of $A$ of **shape** $S$.

"Generalized element" is nothing more than a synonym of "map," but sometimes it is useful to think of maps as generalized elements. For example, when $A$ is a set, a generalized element of $A$ of shape $1$ is an ordinary element, and a generalized element of shape $\mathbb{N}$ is a sequence in $A$. In the category of topological spaces, the generalized elements of shape $1$ are the points, and the generalized elements of shape $S^1$ (the circle) are, by definition, loops. In categories of geometric objects, we might equally well say "figures of shape $S$."

For an object $S$ of a category $\mathscr{A}$, the functor $H^S \colon \mathscr{A} \to \mathbf{Set}$ sends an object to its set of generalized elements of shape $S$. The functoriality tells us that any map $A \to B$ in $\mathscr{A}$ transforms $S$-elements of $A$ into $S$-elements of $B$.

</div>

### 4.2 The Yoneda Lemma

What do representables see?

Recall from Definition 1.2.15 that functors $\mathscr{A}^{\mathrm{op}} \to \mathbf{Set}$ are sometimes called "presheaves" on $\mathscr{A}$. So for each $A \in \mathscr{A}$ we have a representable presheaf $H_A$, and we are asking how the rest of the presheaf category $[\mathscr{A}^{\mathrm{op}}, \mathbf{Set}]$ looks from the viewpoint of $H_A$. In other words, if $X$ is another presheaf, what are the maps $H_A \to X$?

We start by fixing a locally small category $\mathscr{A}$. We then take an object $A \in \mathscr{A}$ and a functor $X \colon \mathscr{A}^{\mathrm{op}} \to \mathbf{Set}$. The object $A$ gives rise to another functor $H_A = \mathscr{A}(-, A) \colon \mathscr{A}^{\mathrm{op}} \to \mathbf{Set}$. The question is: what are the maps $H_A \to X$? Since $H_A$ and $X$ are both objects of the presheaf category $[\mathscr{A}^{\mathrm{op}}, \mathbf{Set}]$, the "maps" concerned are maps in $[\mathscr{A}^{\mathrm{op}}, \mathbf{Set}]$, i.e., natural transformations. The set of such natural transformations is called $[\mathscr{A}^{\mathrm{op}}, \mathbf{Set}](H_A, X)$.

There is an informal principle of general category theory that allows us to guess the answer. Given as input an object $A \in \mathscr{A}$ and a presheaf $X$ on $\mathscr{A}$, we can construct a set, namely $[\mathscr{A}^{\mathrm{op}}, \mathbf{Set}](H_A, X)$. Are there any other ways to construct a set from the same input data $(A, X)$? Yes: simply take the set $X(A)$! The informal principle suggests that these two sets are the same:

$$
[\mathscr{A}^{\mathrm{op}}, \mathbf{Set}](H_A, X) \cong X(A).
$$

This turns out to be true; and that is the Yoneda lemma.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 4.2.1</span><span class="math-callout__name">(Yoneda lemma)</span></p>

*Let $\mathscr{A}$ be a locally small category. Then*

$$
[\mathscr{A}^{\mathrm{op}}, \mathbf{Set}](H_A, X) \cong X(A)
$$

*naturally in $A \in \mathscr{A}$ and $X \in [\mathscr{A}^{\mathrm{op}}, \mathbf{Set}]$.*

</div>

Informally, the Yoneda lemma says that for any $A \in \mathscr{A}$ and presheaf $X$ on $\mathscr{A}$:

> *A natural transformation $H_A \to X$ is an element of $X(A)$.*

The word "naturally" means that each side of the isomorphism is functorial in both $A$ and $X$, and that the isomorphisms can be chosen in a way compatible with these induced maps. Precisely, the Yoneda lemma states that the composite functor

$$
\mathscr{A}^{\mathrm{op}} \times [\mathscr{A}^{\mathrm{op}}, \mathbf{Set}] \xrightarrow{H\_\bullet^{\mathrm{op}} \times 1} [\mathscr{A}^{\mathrm{op}}, \mathbf{Set}]^{\mathrm{op}} \times [\mathscr{A}^{\mathrm{op}}, \mathbf{Set}] \xrightarrow{\mathrm{Hom}} \mathbf{Set}
$$

is naturally isomorphic to the evaluation functor $(A, X) \mapsto X(A)$.

**Proof of the Yoneda lemma.** We have to define, for each $A$ and $X$, a bijection between the sets $[\mathscr{A}^{\mathrm{op}}, \mathbf{Set}](H_A, X)$ and $X(A)$. We then have to show that our bijection is natural in $A$ and $X$.

First, fix $A \in \mathscr{A}$ and $X \in [\mathscr{A}^{\mathrm{op}}, \mathbf{Set}]$. We define functions

$$
[\mathscr{A}^{\mathrm{op}}, \mathbf{Set}](H_A, X) \underset{(\tilde{\ })}{\overset{(\hat{\ })}{\rightleftarrows}} X(A)
$$

and show that they are mutually inverse. So we have to do four things: define $(\hat{\ })$, define $(\tilde{\ })$, show that $\widehat{\tilde{\ }}$ is the identity, and show that $\widetilde{\hat{\ }}$ is the identity.

* Given $\alpha \colon H_A \to X$, define $\hat{\alpha} \in X(A)$ by $\hat{\alpha} = \alpha_A(1_A)$.
* Let $x \in X(A)$. We have to define a natural transformation $\tilde{x} \colon H_A \to X$. That is, we have to define for each $B \in \mathscr{A}$ a function $\tilde{x}\_B \colon H_A(B) = \mathscr{A}(B, A) \to X(B)$ and show that the family $\tilde{x} = (\tilde{x}\_B)\_{B \in \mathscr{A}}$ satisfies naturality. Given $B \in \mathscr{A}$ and $f \in \mathscr{A}(B, A)$, define $\tilde{x}\_B(f) = (X(f))(x) \in X(B)$. The naturality of $\tilde{x}$ follows from the functoriality of $X$.
* Given $x \in X(A)$, we have to show that $\hat{\tilde{x}} = x$, and indeed, $\hat{\tilde{x}} = \tilde{x}\_A(1_A) = (X(1_A))(x) = 1_{X(A)}(x) = x$.
* Given $\alpha \colon H_A \to X$, we have to show that $\widetilde{\hat{\alpha}} = \alpha$. This reduces to showing that $(X(f))(\alpha_A(1_A)) = \alpha_B(f)$ for all $B \in \mathscr{A}$ and $f \colon B \to A$ in $\mathscr{A}$. By naturality of $\alpha$, the square with $H_A(f) = - \circ f$ on top and $X(f)$ on the bottom commutes, which when taken at $1_A \in \mathscr{A}(A, A)$ gives exactly the desired equation.

It is worth pausing to consider the significance of the fact that $\widetilde{\hat{\alpha}} = \alpha$. Since $\hat{\alpha}$ is the value of $\alpha$ at $1_A$, this implies:

> *A natural transformation $H_A \to X$ is determined by its value at $1_A$.*

This establishes the bijection for each $A$ and $X \in [\mathscr{A}^{\mathrm{op}}, \mathbf{Set}]$. We now show that the bijection is natural in $A$ and $X$.

**Naturality in $A$:** For each $X \in [\mathscr{A}^{\mathrm{op}}, \mathbf{Set}]$ and $B \xrightarrow{f} A$ in $\mathscr{A}$, the square

$$
[\mathscr{A}^{\mathrm{op}}, \mathbf{Set}](H_A, X) \xrightarrow{- \circ H_f} [\mathscr{A}^{\mathrm{op}}, \mathbf{Set}](H_B, X)
$$

$$
\downarrow (\hat{\ }) \qquad\qquad\qquad\qquad \downarrow (\hat{\ })
$$

$$
X(A) \xrightarrow{X(f)} X(B)
$$

commutes. Indeed, for $\alpha \colon H_A \to X$, we have $(\alpha \circ H_f)\_B(1_B) = \alpha_B((H_f)\_B(1_B)) = \alpha_B(f \circ 1_B) = \alpha_B(f)$, which equals $(X(f))(\alpha_A(1_A))$ by naturality of $\alpha$.

**Naturality in $X$:** For each $A \in \mathscr{A}$ and natural transformation $\theta \colon X \to X'$, the square

$$
[\mathscr{A}^{\mathrm{op}}, \mathbf{Set}](H_A, X) \xrightarrow{\theta \circ -} [\mathscr{A}^{\mathrm{op}}, \mathbf{Set}](H_A, X')
$$

$$
\downarrow (\hat{\ }) \qquad\qquad\qquad\qquad \downarrow (\hat{\ })
$$

$$
X(A) \xrightarrow{\theta_A} X'(A)
$$

commutes. Indeed, for $\alpha \colon H_A \to X$, we have $(\theta \circ \alpha)\_A(1_A) = \theta_A(\alpha_A(1_A))$. This completes the proof. $\square$

### 4.3 Consequences of the Yoneda Lemma

The Yoneda lemma is fundamental in category theory. Here we look at three important consequences.

#### A Representation is a Universal Element

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 4.3.2</span><span class="math-callout__name">(Representation as universal element — contravariant)</span></p>

*Let $\mathscr{A}$ be a locally small category and $X \colon \mathscr{A}^{\mathrm{op}} \to \mathbf{Set}$. Then a representation of $X$ consists of an object $A \in \mathscr{A}$ together with an element $u \in X(A)$ such that:*

*for each $B \in \mathscr{A}$ and $x \in X(B)$, there is a unique map $\bar{x} \colon B \to A$ such that $(X\bar{x})(u) = x$.*

</div>

Pairs $(B, x)$ with $B \in \mathscr{A}$ and $x \in X(B)$ are sometimes called **elements** of the presheaf $X$. An element $u$ satisfying the condition above is sometimes called a **universal** element of $X$. So, Corollary 4.3.2 says that a representation of a presheaf $X$ amounts to a universal element of $X$.

**Proof.** By the Yoneda lemma, we have only to show that for $A \in \mathscr{A}$ and $u \in X(A)$, the natural transformation $\tilde{u} \colon H_A \to X$ is an isomorphism if and only if the condition above holds. Now, $\tilde{u}$ is an isomorphism if and only if for all $B \in \mathscr{A}$, the function $\tilde{u}\_B \colon H_A(B) = \mathscr{A}(B, A) \to X(B)$ is a bijection, if and only if for all $B \in \mathscr{A}$ and $x \in X(B)$, there is a unique $\bar{x} \in \mathscr{A}(B, A)$ such that $\tilde{u}\_B(\bar{x}) = x$. But $\tilde{u}\_B(\bar{x}) = (X\bar{x})(u)$, so this is exactly the stated condition. $\square$

Our examples will use the dual form, for covariant set-valued functors:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 4.3.3</span><span class="math-callout__name">(Representation as universal element — covariant)</span></p>

*Let $\mathscr{A}$ be a locally small category and $X \colon \mathscr{A} \to \mathbf{Set}$. Then a representation of $X$ consists of an object $A \in \mathscr{A}$ together with an element $u \in X(A)$ such that:*

*for each $B \in \mathscr{A}$ and $x \in X(B)$, there is a unique map $\bar{x} \colon A \to B$ such that $(X\bar{x})(u) = x$.*

**Proof.** Follows immediately by duality. $\square$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.3.4</span><span class="math-callout__name">(Free vector space)</span></p>

Fix a set $S$ and consider the functor $X = \mathbf{Set}(S, U(-)) \colon \mathbf{Vect}_k \to \mathbf{Set}$, where $V \mapsto \mathbf{Set}(S, U(V))$. Here are two familiar (and true!) statements about $X$:

**(a)** there exist a vector space $F(S)$ and an isomorphism $\mathbf{Vect}_k(F(S), V) \cong \mathbf{Set}(S, U(V))$ natural in $V \in \mathbf{Vect}_k$ (Example 2.1.3(a));

**(b)** there exist a vector space $F(S)$ and a function $u \colon S \to U(F(S))$ such that: for each vector space $V$ and function $f \colon S \to U(V)$, there is a unique linear map $\bar{f} \colon F(S) \to V$ such that $U(\bar{f}) \circ u = f$.

Each of these two statements says that $X$ is representable. Statement (a) says that there is an isomorphism $X(V) \cong \mathbf{Vect}(F(S), V)$ natural in $V$, that is, an isomorphism $X \cong H^{F(S)}$. Statement (b) says that $u \in X(F(S))$ satisfies the condition of Corollary 4.3.3. Corollary 4.3.3 tells us that this is an illusion: all natural isomorphisms arise in this way. It is the word "natural" in (a) that hides the explicit detail.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.3.5</span><span class="math-callout__name">(Adjunctions and universal elements)</span></p>

The same can be said for any other adjunction $\mathscr{A} \overset{F}{\underset{G}{\rightleftarrows}} \mathscr{B}$. Fix $A \in \mathscr{A}$ and put $X = \mathscr{A}(A, G(-)) \colon \mathscr{B} \to \mathbf{Set}$. Then $X$ is representable, and this can be expressed in either of the following ways:

**(a)** $\mathscr{A}(A, G(B)) \cong \mathscr{B}(F(A), B)$ naturally in $B$; in other words, $X \cong H^{F(A)}$ (as in Lemma 4.1.10);

**(b)** the unit map $\eta_A \colon A \to G(F(A))$ is an initial object of the comma category $(A \Rightarrow G)$; that is, $\eta_A \in X(F(A))$ satisfies condition (4.7).

This observation can be developed into an alternative proof of Theorem 2.3.6, the reformulation of adjointness in terms of initial objects.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 4.3.6</span><span class="math-callout__name">(Universal element of the forgetful functor on $\mathbf{Grp}$)</span></p>

For any group $G$ and element $x \in G$, there is a unique homomorphism $\phi \colon \mathbb{Z} \to G$ such that $\phi(1) = x$. This means that $1 \in U(\mathbb{Z})$ is a universal element of the forgetful functor $U \colon \mathbf{Grp} \to \mathbf{Set}$; in other words, condition (4.7) holds when $\mathscr{A} = \mathbf{Grp}$, $X = U$, $A = \mathbb{Z}$ and $u = 1$. So $1 \in U(\mathbb{Z})$ gives a representation $H^{\mathbb{Z}} \xrightarrow{\sim} U$ of $U$.

On the other hand, the same is true with $-1$ in place of $1$. The isomorphisms $H^{\mathbb{Z}} \xrightarrow{\sim} U$ coming from $1$ and $-1$ are not equal, because Corollary 4.3.3 provides a *one-to-one* correspondence between universal elements and representations.

</div>

#### The Yoneda Embedding

Here is a second corollary of the Yoneda lemma.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 4.3.7</span><span class="math-callout__name">(Yoneda embedding is full and faithful)</span></p>

*For any locally small category $\mathscr{A}$, the Yoneda embedding*

$$
H\_\bullet \colon \mathscr{A} \to [\mathscr{A}^{\mathrm{op}}, \mathbf{Set}]
$$

*is full and faithful.*

</div>

Informally, this says that for $A, A' \in \mathscr{A}$, a map $H_A \to H_{A'}$ of presheaves is the same thing as a map $A \to A'$ in $\mathscr{A}$.

**Proof.** We have to show that for each $A, A' \in \mathscr{A}$, the function

$$
\mathscr{A}(A, A') \to [\mathscr{A}^{\mathrm{op}}, \mathbf{Set}](H_A, H_{A'}), \qquad f \mapsto H_f
$$

is bijective. By the Yoneda lemma (taking $X$ to be $H_{A'}$), the function

$$
(\hat{\ }) \colon H_{A'}(A) \to [\mathscr{A}^{\mathrm{op}}, \mathbf{Set}](H_A, H_{A'})
$$

is bijective. So it is enough to prove that these two functions are equal. Thus, given $f \colon A \to A'$, we have to prove that $\tilde{f} = H_f$, or equivalently, $\widehat{H_f} = f$. And indeed,

$$
\widehat{H_f} = (H_f)\_A(1_A) = f \circ 1_A = f,
$$

as required. $\square$

In mathematics at large, the word "embedding" is used (sometimes informally) to mean a map $A \to B$ that makes $A$ isomorphic to its image in $B$. A full and faithful functor $\mathscr{A} \to \mathscr{B}$ can reasonably be called an embedding, as it makes $\mathscr{A}$ equivalent to a full subcategory of $\mathscr{B}$.

In the case at hand, the Yoneda embedding $H\_\bullet \colon \mathscr{A} \to [\mathscr{A}^{\mathrm{op}}, \mathbf{Set}]$ embeds $\mathscr{A}$ into its own presheaf category. So, $\mathscr{A}$ is equivalent to the full subcategory of $[\mathscr{A}^{\mathrm{op}}, \mathbf{Set}]$ whose objects are the representables.

## Chapter 5: Limits

Limits, and the dual concept, colimits, provide our third approach to the idea of universal property. Adjointness is about the relationships *between* categories. Representability is a property of *set-valued* functors. Limits are about what goes on *inside* a category.

The concept of limit unifies many familiar constructions in mathematics. Whenever you meet a method for taking some objects and maps in a category and constructing a new object out of them, there is a good chance that you are looking at either a limit or a colimit. For instance, in group theory, we can take a homomorphism between two groups and form its kernel, which is a new group. This construction is an example of a limit in the category of groups. Or, we might take two natural numbers and form their lowest common multiple. This is an example of a colimit in the poset of natural numbers, ordered by divisibility.

### 5.1 Limits: Definition and Examples

The definition of limit is very general. We build up to it by first examining some particularly useful types of limit: products, equalizers, and pullbacks.

#### Products

Let $X$ and $Y$ be sets. The familiar cartesian product $X \times Y$ is characterized by the property that an element of $X \times Y$ is an element of $X$ together with an element of $Y$. Since elements are just maps from $1$, this says that a map $1 \to X \times Y$ amounts to a map $1 \to X$ together with a map $1 \to Y$. A little thought reveals that the same is true when $1$ is replaced by any set $A$ whatsoever. The bijection between maps $A \to X \times Y$ and pairs of maps $(A \to X,\, A \to Y)$ is given by composing with the projection maps $X \xleftarrow{p_1} X \times Y \xrightarrow{p_2} Y$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.1.1</span><span class="math-callout__name">(Product)</span></p>

Let $\mathscr{A}$ be a category and $X, Y \in \mathscr{A}$. A **product** of $X$ and $Y$ consists of an object $P$ and maps $P \xrightarrow{p_1} X$, $P \xrightarrow{p_2} Y$ with the property that for all objects $A$ and maps $f_1 \colon A \to X$, $f_2 \colon A \to Y$ in $\mathscr{A}$, there exists a unique map $\bar{f} \colon A \to P$ such that $p_1 \circ \bar{f} = f_1$ and $p_2 \circ \bar{f} = f_2$. The maps $p_1$ and $p_2$ are called the **projections**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remarks 5.1.2</span></p>

**(a)** Products do not always exist. But when objects $X$ and $Y$ of a category do have a product, it is unique up to isomorphism (as in Lemma 2.1.8, or by Corollary 6.1.2). This justifies talking about *the* product of $X$ and $Y$.

**(b)** Strictly speaking, the product consists of the object $P$ *together with* the projections $p_1$ and $p_2$. But informally, we often refer to $P$ alone as the product of $X$ and $Y$. We write $P$ as $X \times Y$.

In general, in any category, the map $\bar{f}$ is usually written as $(f_1, f_2)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.1.3</span><span class="math-callout__name">(Products in $\mathbf{Set}$)</span></p>

Any two sets $X$ and $Y$ have a product in $\mathbf{Set}$. It is the usual cartesian product $X \times Y$, equipped with the usual projection maps $p_1$ and $p_2$. To verify, define $\bar{f} \colon A \to X \times Y$ by $\bar{f}(a) = (f_1(a), f_2(a))$. Then $p_i \circ \bar{f} = f_i$ for $i = 1, 2$. Moreover, this is the *only* map making the diagram commute: if $\hat{f}$ also makes it commute, then $\hat{f}(a) = (p_1(\hat{f}(a)), p_2(\hat{f}(a))) = (f_1(a), f_2(a)) = \bar{f}(a)$ for all $a \in A$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.1.4</span><span class="math-callout__name">(Products in $\mathbf{Top}$)</span></p>

In the category of topological spaces, any two objects $X$ and $Y$ have a product. It is the set $X \times Y$ equipped with the product topology and the standard projection maps. The product topology is deliberately designed so that a function $A \to X \times Y$, $t \mapsto (x(t), y(t))$, is continuous if and only if it is continuous in each coordinate ($t \mapsto x(t)$ and $t \mapsto y(t)$ are both continuous). The product topology is the smallest topology on $X \times Y$ for which the projections are continuous.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.1.5</span><span class="math-callout__name">(Products in $\mathbf{Vect}_k$)</span></p>

Let $X$ and $Y$ be vector spaces. We can form their direct sum, $X \oplus Y$, with linear projection maps $X \xleftarrow{p_1} X \oplus Y \xrightarrow{p_2} Y$. It can be shown that $X \oplus Y$, together with $p_1$ and $p_2$, is the product of $X$ and $Y$ in the category of vector spaces.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Examples 5.1.6</span><span class="math-callout__name">(Products in ordered sets)</span></p>

**(a)** Let $x, y \in \mathbb{R}$. Their minimum $\min\lbrace x, y\rbrace$ satisfies $\min\lbrace x, y\rbrace \le x$ and $\min\lbrace x, y\rbrace \le y$, and has the further property that whenever $a \in \mathbb{R}$ with $a \le x$ and $a \le y$, we have $a \le \min\lbrace x, y\rbrace$. This means exactly that when the poset $(\mathbb{R}, \le)$ is viewed as a category, the product of $x$ and $y$ is $\min\lbrace x, y\rbrace$.

**(b)** Fix a set $S$. Let $X, Y \in \mathscr{P}(S)$. Then $X \cap Y$ satisfies $X \cap Y \subseteq X$ and $X \cap Y \subseteq Y$, and has the further property that whenever $A \in \mathscr{P}(S)$ with $A \subseteq X$ and $A \subseteq Y$, we have $A \subseteq X \cap Y$. This means that $X \cap Y$ is the product of $X$ and $Y$ in the poset $(\mathscr{P}(S), \subseteq)$ regarded as a category.

**(c)** Let $x, y \in \mathbb{N}$. Their greatest common divisor $\gcd(x, y)$ satisfies $\gcd(x, y) \mid x$ and $\gcd(x, y) \mid y$, and has the further property that whenever $a \in \mathbb{N}$ with $a \mid x$ and $a \mid y$, we have $a \mid \gcd(x, y)$. This means that $\gcd(x, y)$ is the product of $x$ and $y$ in the poset $(\mathbb{N}, \mid)$ regarded as a category.

Generally, let $(A, \le)$ be a poset and $x, y \in A$. A **lower bound** for $x$ and $y$ is an element $a \in A$ such that $a \le x$ and $a \le y$. A **greatest lower bound** or **meet** of $x$ and $y$ is a lower bound $z$ with the further property that whenever $a$ is a lower bound for $x$ and $y$, we have $a \le z$. When a poset is regarded as a category, meets are exactly products. The meet of $x$ and $y$ is usually written as $x \wedge y$ rather than $x \times y$. Thus:

$$
x \wedge y = \min\lbrace x, y\rbrace, \qquad X \wedge Y = X \cap Y, \qquad x \wedge y = \gcd(x, y).
$$

</div>

We have been discussing products $X \times Y$ of *two* objects, so-called **binary products**. But there is no reason to stick to two. The definition changes in the most obvious way:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.1.7</span><span class="math-callout__name">(Arbitrary product)</span></p>

Let $\mathscr{A}$ be a category, $I$ a set, and $(X_i)\_{i \in I}$ a family of objects of $\mathscr{A}$. A **product** of $(X_i)\_{i \in I}$ consists of an object $P$ and a family of maps $(P \xrightarrow{p_i} X_i)\_{i \in I}$ with the property that for all objects $A$ and families of maps $(A \xrightarrow{f_i} X_i)\_{i \in I}$, there exists a unique map $\bar{f} \colon A \to P$ such that $p_i \circ \bar{f} = f_i$ for all $i \in I$.

When the product $P$ exists, we write $P$ as $\prod_{i \in I} X_i$. We call the maps $f_i$ the **components** of the map $(f_i)\_{i \in I}$. Taking $I$ to be a two-element set, we recover the special case of binary products.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.1.8</span><span class="math-callout__name">(Products in ordered sets — general)</span></p>

In ordered sets, the extension from binary to arbitrary products works in the obvious way: given an ordered set $(A, \le)$, a **lower bound** for a family $(x_i)\_{i \in I}$ of elements is an element $a \in A$ such that $a \le x_i$ for all $i$, and a **greatest lower bound** or **meet** of the family is a lower bound greater than any other, written as $\bigwedge_{i \in I} x_i$. These are the products in $(A, \le)$.

For example, in $\mathbb{R}$ with its usual ordering, the meet of a family $(x_i)\_{i \in I}$ is $\inf\lbrace x_i \mid i \in I\rbrace$ (and one exists if and only if the other does).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.1.9</span><span class="math-callout__name">(Empty product is terminal object)</span></p>

What happens to the definition of product when the indexing set $I$ is empty? When $I$ is empty, there is exactly one family $(X_i)\_{i \in \emptyset}$, the **empty family**, and exactly one family of maps $(A \xrightarrow{f_i} X_i)\_{i \in \emptyset}$ for any given object $A$. A product of the empty family therefore consists of an object $P$ of $\mathscr{A}$ such that for each object $A$ of $\mathscr{A}$, there exists a unique map $\bar{f} \colon A \to P$. In other words, a product of the empty family is exactly a terminal object.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.1.10</span><span class="math-callout__name">(Powers)</span></p>

Take an object $X$ of a category $\mathscr{A}$, and a set $I$. There is a constant family $(X)\_{i \in I}$. Its product $\prod_{i \in I} X$, if it exists, is written as $X^I$ and called a **power** of $X$. When $X$ is a set, $X^I$ is the set of functions from $I$ to $X$, also written as $\mathbf{Set}(I, X)$.

</div>

#### Equalizers

To define our second type of limit, we need a preliminary piece of terminology: a **fork** in a category consists of objects and maps $A \xrightarrow{f} X \overset{s}{\underset{t}{\rightrightarrows}} Y$ such that $s \circ f = t \circ f$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.1.11</span><span class="math-callout__name">(Equalizer)</span></p>

Let $\mathscr{A}$ be a category and let $X \overset{s}{\underset{t}{\rightrightarrows}} Y$ be objects and maps in $\mathscr{A}$. An **equalizer** of $s$ and $t$ is an object $E$ together with a map $E \xrightarrow{i} X$ such that $E \xrightarrow{i} X \overset{s}{\underset{t}{\rightrightarrows}} Y$ is a fork, and with the property that for any fork $A \xrightarrow{f} X \overset{s}{\underset{t}{\rightrightarrows}} Y$, there exists a unique map $\bar{f} \colon A \to E$ such that $i \circ \bar{f} = f$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.1.12</span><span class="math-callout__name">(Equalizers in $\mathbf{Set}$)</span></p>

We have already met equalizers in $\mathbf{Set}$ (Section 3.1). Given functions $X \overset{s}{\underset{t}{\rightrightarrows}} Y$, write $E = \lbrace x \in X \mid s(x) = t(x)\rbrace$ and $i \colon E \to X$ for the inclusion. Then $si = ti$, so we have a fork, and one can check that it is universal among all forks on $s$ and $t$.

An equalizer describes the set of solutions of a single equation, but by combining equalizers with products, we can also describe the solution-set of any system of simultaneous equations. Take a set $\Lambda$ and a family $(X \overset{s_\lambda}{\underset{t_\lambda}{\rightrightarrows}} Y_\lambda)\_{\lambda \in \Lambda}$ of pairs of maps in $\mathbf{Set}$. Then the solution-set $\lbrace x \in X \mid s_\lambda(x) = t_\lambda(x) \text{ for all } \lambda \in \Lambda\rbrace$ is the equalizer of the functions $(s_\lambda)\_{\lambda \in \Lambda}, (t_\lambda)\_{\lambda \in \Lambda} \colon X \rightrightarrows \prod_{\lambda \in \Lambda} Y_\lambda$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.1.13</span><span class="math-callout__name">(Equalizers in $\mathbf{Top}$)</span></p>

Take continuous maps $X \overset{s}{\underset{t}{\rightrightarrows}} Y$ between topological spaces. We can form their equalizer $E$ in the category of sets, with inclusion map $i \colon E \to X$. Since $E$ is a subset of the space $X$, it acquires the subspace topology from $X$, and $i$ is then continuous. This space $E$, together with $i$, is the equalizer of $s$ and $t$ in $\mathbf{Top}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.1.14</span><span class="math-callout__name">(Kernels as equalizers)</span></p>

Let $\theta \colon G \to H$ be a homomorphism of groups. As in Example 0.8, the homomorphism $\theta$ gives rise to a fork $\ker \theta \xhookrightarrow{\iota} G \overset{\theta}{\underset{\varepsilon}{\rightrightarrows}} H$, where $\iota$ is the inclusion and $\varepsilon$ is the trivial homomorphism. This is an equalizer in $\mathbf{Grp}$. Thus, kernels are a special case of equalizers.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.1.15</span><span class="math-callout__name">(Equalizers in $\mathbf{Vect}_k$)</span></p>

Let $V \overset{s}{\underset{t}{\rightrightarrows}} W$ be linear maps between vector spaces. There is a linear map $t - s \colon V \to W$, and the equalizer of $s$ and $t$ in the category of vector spaces is the space $\ker(t - s)$ together with the inclusion map $\ker(t - s) \hookrightarrow V$.

</div>

#### Pullbacks

We explore one more type of limit before formulating the general definition.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.1.16</span><span class="math-callout__name">(Pullback)</span></p>

Let $\mathscr{A}$ be a category, and take objects and maps $X \xrightarrow{s} Z \xleftarrow{t} Y$ in $\mathscr{A}$. A **pullback** of this diagram is an object $P \in \mathscr{A}$ together with maps $p_1 \colon P \to X$ and $p_2 \colon P \to Y$ such that $s \circ p_1 = t \circ p_2$, and with the property that for any commutative square $s \circ f_1 = t \circ f_2$ (with $f_1 \colon A \to X$, $f_2 \colon A \to Y$), there is a unique map $\bar{f} \colon A \to P$ such that $p_1 \bar{f} = f_1$ and $p_2 \bar{f} = f_2$.

We call this a **pullback square**. Another name for pullback is **fibred product**. When $Z$ is a terminal object (and $s$ and $t$ are the only maps they can possibly be), a pullback of the diagram is simply a product of $X$ and $Y$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Examples 5.1.17</span><span class="math-callout__name">(Pullbacks in $\mathbf{Set}$)</span></p>

The pullback of a diagram $X \xrightarrow{s} Z \xleftarrow{t} Y$ in $\mathbf{Set}$ is

$$
P = \lbrace (x, y) \in X \times Y \mid s(x) = t(y)\rbrace
$$

with projections $p_1(x, y) = x$ and $p_2(x, y) = y$.

**(a)** The formation of inverse images is an instance of pullbacks. Given a function $f \colon X \to Y$ and a subset $Y' \subseteq Y$, we obtain the inverse image $f^{-1}Y' = \lbrace x \in X \mid f(x) \in Y'\rbrace \subseteq X$ and a new function $f' \colon f^{-1}Y' \to Y'$, $x \mapsto f(x)$. With the inclusion functions $j \colon Y' \hookrightarrow Y$ and $i \colon f^{-1}Y' \hookrightarrow X$, this gives a pullback square. People sometimes say that $f^{-1}Y'$ is obtained by "pulling $Y'$ back" along $f$: hence the name.

**(b)** Intersection of subsets provides another example of pullbacks. Let $X$ and $Y$ be subsets of a set $Z$. Then the square with $X \cap Y$ at the top-left, $Y$ at the top-right, $X$ at the bottom-left, and $Z$ at the bottom-right (all arrows being inclusions of subsets) is a pullback square in $\mathbf{Set}$. This is a special case of (a), since $X \cap Y$ is the inverse image of $Y \subseteq Z$ under the inclusion map $X \hookrightarrow Z$.

</div>

#### The Definition of Limit

We have now looked at three constructions: products, equalizers and pullbacks. They clearly have something in common. Each starts with some objects and (in the case of equalizers and pullbacks) some maps between them. In each, we aim to construct a new object together with some maps from it to the original objects, with a universal property.

The starting data in each construction can be seen as a functor from a small category $\mathbf{I}$ (called the **shape**) to $\mathscr{A}$. For products, $\mathbf{I} = \mathbf{T}$ is the discrete two-object category. For equalizers, $\mathbf{I} = \mathbf{E}$ is the category with two objects and two parallel non-identity arrows. For pullbacks, $\mathbf{I} = \mathbf{P}$ is the category with three objects forming a "corner" shape.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.1.18</span><span class="math-callout__name">(Diagram)</span></p>

Let $\mathscr{A}$ be a category and $\mathbf{I}$ a small category. A functor $\mathbf{I} \to \mathscr{A}$ is called a **diagram** in $\mathscr{A}$ of **shape** $\mathbf{I}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.1.19</span><span class="math-callout__name">(Cone and limit)</span></p>

Let $\mathscr{A}$ be a category, $\mathbf{I}$ a small category, and $D \colon \mathbf{I} \to \mathscr{A}$ a diagram in $\mathscr{A}$.

**(a)** A **cone** on $D$ is an object $A \in \mathscr{A}$ (the **vertex** of the cone) together with a family $(A \xrightarrow{f_I} D(I))\_{I \in \mathbf{I}}$ of maps in $\mathscr{A}$ such that for all maps $I \xrightarrow{u} J$ in $\mathbf{I}$, the triangle $D(u) \circ f_I = f_J$ commutes. (Here and later, we abbreviate $D(u)$ as $Du$.)

**(b)** A **limit** of $D$ is a cone $(L \xrightarrow{p_I} D(I))\_{I \in \mathbf{I}}$ with the property that for any cone $(A \xrightarrow{f_I} D(I))\_{I \in \mathbf{I}}$ on $D$, there exists a unique map $\bar{f} \colon A \to L$ such that $p_I \circ \bar{f} = f_I$ for all $I \in \mathbf{I}$. The maps $p_I$ are called the **projections** of the limit.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remarks 5.1.20</span></p>

**(a)** Loosely, the universal property says that for any $A \in \mathscr{A}$, maps $A \to L$ correspond one-to-one with cones on $D$ with vertex $A$. In Section 6.1, we will use this thought to rephrase the definition of limit in terms of representability. From this it will follow that limits are unique up to canonical isomorphism, when they exist (Corollary 6.1.2). Alternatively, uniqueness can be proved by the usual kind of direct argument, as in Lemma 2.1.8.

**(b)** If $(L \xrightarrow{p_I} D(I))\_{I \in \mathbf{I}}$ is a limit of $D$, we sometimes abuse language slightly by referring to $L$ (rather than the whole cone) as the limit of $D$. For emphasis, we sometimes call $(L \xrightarrow{p_I} D(I))\_{I \in \mathbf{I}}$ a **limit cone**. We write $L = \varprojlim D$. Remark (a) can then be stated as: *a map into $\varprojlim D$ is a cone on $D$*.

**(c)** By assuming from the outset that the shape category $\mathbf{I}$ is small, we are restricting ourselves to what are officially called **small limits**. We will seldom be interested in any other kind.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Examples 5.1.21</span><span class="math-callout__name">(Limit shapes)</span></p>

Let $\mathscr{A}$ be any category. Recall the categories $\mathbf{T}$, $\mathbf{E}$ and $\mathbf{P}$.

**(a)** A diagram $D$ of shape $\mathbf{T}$ in $\mathscr{A}$ is a pair $(X, Y)$ of objects of $\mathscr{A}$. A cone on $D$ is an object $A$ together with maps $f_1 \colon A \to X$ and $f_2 \colon A \to Y$ (as in Definition 5.1.1), and a limit of $D$ is a product of $X$ and $Y$. More generally, let $I$ be a set and write $\mathbf{I}$ for the discrete category on $I$. A functor $D \colon \mathbf{I} \to \mathscr{A}$ is an $I$-indexed family $(X_i)\_{i \in I}$ of objects of $\mathscr{A}$, and a limit of $D$ is exactly a product of the family $(X_i)\_{i \in I}$. In particular, a limit of the unique functor $\emptyset \to \mathscr{A}$ is a terminal object of $\mathscr{A}$.

**(b)** A diagram $D$ of shape $\mathbf{E}$ in $\mathscr{A}$ is a parallel pair $X \overset{s}{\underset{t}{\rightrightarrows}} Y$ of maps in $\mathscr{A}$. A cone on $D$ consists of an object $A$ and a map $f \colon A \to X$ such that $s \circ f = t \circ f$ (i.e., a fork). A limit of $D$ is a universal fork on $s$ and $t$, that is, an equalizer of $s$ and $t$.

**(c)** A diagram $D$ of shape $\mathbf{P}$ in $\mathscr{A}$ consists of objects and maps $X \xrightarrow{s} Z \xleftarrow{t} Y$. A cone on $D$ is a commutative square. A limit of $D$ is a pullback.

**(d)** Let $\mathbf{I} = (\mathbb{N}, \le)^{\mathrm{op}}$. A diagram $D \colon \mathbf{I} \to \mathscr{A}$ consists of objects and maps $\cdots \xrightarrow{s_3} X_2 \xrightarrow{s_2} X_1 \xrightarrow{s_1} X_0$. For example, a set $X_0$ and a chain of subsets $\cdots \subseteq X_2 \subseteq X_1 \subseteq X_0$ gives such a diagram in $\mathbf{Set}$, whose limit is $\bigcap_{i \in \mathbb{N}} X_i$. In this and similar contexts, limits are sometimes referred to as **inverse limits**.

</div>

In general, the limit of a diagram $D$ is the terminal object in the category of cones on $D$, and is therefore an extremal example of a cone on $D$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.1.22</span><span class="math-callout__name">(Limits in $\mathbf{Set}$ — explicit formula)</span></p>

Let $D \colon \mathbf{I} \to \mathbf{Set}$, and let us ask what $\varprojlim D$ would have to be if it existed. We would have

$$
\varprojlim D \cong \mathbf{Set}\!\left(1, \varprojlim D\right) \cong \lbrace\text{cones on } D \text{ with vertex } 1\rbrace \cong \left\lbrace (x_I)\_{I \in \mathbf{I}} \;\middle|\; x_I \in D(I) \text{ for all } I \in \mathbf{I} \text{ and } (Du)(x_J) = x_K \text{ for all } J \xrightarrow{u} K \text{ in } \mathbf{I}\right\rbrace.
$$

In fact, this set really *is* the limit of $D$ in $\mathbf{Set}$, with projections $p_J \colon \varprojlim D \to D(J)$ given by $p_J((x_I)\_{I \in \mathbf{I}}) = x_J$. So in $\mathbf{Set}$, all limits exist.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.1.23</span><span class="math-callout__name">(Limits in algebraic categories)</span></p>

The same formula gives limits in categories of algebras such as $\mathbf{Grp}$, $\mathbf{Ring}$, $\mathbf{Vect}_k$, etc. Of course, we also have to say what the group/ring/... structure on the set is, but this works in the most straightforward way imaginable. For instance, in $\mathbf{Vect}_k$, if $(x_I)\_{I \in \mathbf{I}}, (y_I)\_{I \in \mathbf{I}} \in \varprojlim D$ then $(x_I)\_{I \in \mathbf{I}} + (y_I)\_{I \in \mathbf{I}} = (x_I + y_I)\_{I \in \mathbf{I}}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.1.24</span><span class="math-callout__name">(Limits in $\mathbf{Top}$)</span></p>

The same formula also gives limits in $\mathbf{Top}$. The topology on the set given by the formula is the smallest for which the projection maps are continuous.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.1.25</span><span class="math-callout__name">(Completeness)</span></p>

**(a)** Let $\mathbf{I}$ be a small category. A category $\mathscr{A}$ **has limits of shape $\mathbf{I}$** if for every diagram $D$ of shape $\mathbf{I}$ in $\mathscr{A}$, a limit of $D$ exists.

**(b)** A category **has all limits** (or properly, **has small limits**) if it has limits of shape $\mathbf{I}$ for all small categories $\mathbf{I}$.

Thus, $\mathbf{Set}$, $\mathbf{Top}$, $\mathbf{Grp}$, $\mathbf{Ring}$, $\mathbf{Vect}_k$, ... all have all limits. A **finite limit** is a limit of shape $\mathbf{I}$ for some finite category $\mathbf{I}$. For instance, binary products, terminal objects, equalizers and pullbacks are all finite limits.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 5.1.26</span><span class="math-callout__name">(Building limits from basic ones)</span></p>

*Let $\mathscr{A}$ be a category.*

*(a) If $\mathscr{A}$ has all products and equalizers then $\mathscr{A}$ has all limits.*

*(b) If $\mathscr{A}$ has binary products, a terminal object and equalizers then $\mathscr{A}$ has finite limits.*

</div>

The idea behind the proof is as follows. The explicit formula for limits in $\mathbf{Set}$ describes $\varprojlim D$ as the subset of the product $\prod_{I \in \mathbf{I}} D(I)$ consisting of those elements for which certain equations hold. But the set of solutions to a system of simultaneous equations can be described via products and equalizers (as in Example 5.1.12). And in fact, this same description is valid in any category.

More precisely, for a diagram $D \colon \mathbf{I} \to \mathbf{Set}$, $\varprojlim D$ is the equalizer of

$$
\prod_{I \in \mathbf{I}} D(I) \overset{s}{\underset{t}{\rightrightarrows}} \prod_{J \xrightarrow{u} K \text{ in } \mathbf{I}} D(K)
$$

where the components of $s$ and $t$ are defined, for each map $J \xrightarrow{u} K$ in $\mathbf{I}$, by $s_u((x_I)\_{I \in \mathbf{I}}) = (Du)(x_J)$ and $t_u((x_I)\_{I \in \mathbf{I}}) = x_K$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.1.27</span><span class="math-callout__name">(Compact Hausdorff spaces)</span></p>

Let $\mathbf{CptHff}$ denote the category of compact Hausdorff spaces and continuous maps. Given continuous maps $s$ and $t$ from a topological space $X$ to a Hausdorff space $Y$, the subset $\lbrace x \in X \mid s(x) = t(x)\rbrace$ of $X$ is closed. From this it follows that $\mathbf{CptHff}$ has equalizers. Also, Tychonoff's theorem states that any product (in $\mathbf{Top}$) of compact spaces is compact, and any product of Hausdorff spaces is Hausdorff. From this it follows that $\mathbf{CptHff}$ has all products. Hence by Proposition 5.1.26(a), $\mathbf{CptHff}$ has all limits.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.1.28</span><span class="math-callout__name">(Finite limits in $\mathbf{Vect}_k$)</span></p>

Recall from Example 5.1.15 that kernels provide equalizers in $\mathbf{Vect}_k$. By Proposition 5.1.26(b), finite limits in $\mathbf{Vect}_k$ can always be expressed in terms of $\oplus$ (binary direct sum), $\lbrace 0\rbrace$, and kernels. The same is true in $\mathbf{Ab}$.

</div>

#### Monics

For functions between sets, injectivity is an important concept. For maps in an arbitrary category, injectivity does not make sense, but there is a concept that plays a similar role.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.1.29</span><span class="math-callout__name">(Monic)</span></p>

Let $\mathscr{A}$ be a category. A map $X \xrightarrow{f} Y$ in $\mathscr{A}$ is **monic** (or a **monomorphism**) if for all objects $A$ and maps $A \overset{x}{\underset{x'}{\rightrightarrows}} X$,

$$
f \circ x = f \circ x' \implies x = x'.
$$

This can be rephrased suggestively in terms of generalized elements: $f$ is monic if for all generalized elements $x$ and $x'$ of $X$ (of the same shape), $fx = fx' \implies x = x'$. Being monic is, therefore, the generalized-element analogue of injectivity.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.1.30</span><span class="math-callout__name">(Monics in $\mathbf{Set}$)</span></p>

In $\mathbf{Set}$, a map is monic if and only if it is injective. Indeed, if $f$ is injective then certainly $f$ is monic, and for the converse, take $A = 1$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.1.31</span><span class="math-callout__name">(Monics in algebraic categories)</span></p>

In categories of algebras such as $\mathbf{Grp}$, $\mathbf{Vect}_k$, $\mathbf{Ring}$, etc., it is also true that the monic maps are exactly the injections. It is easy to show that injections are monic. For the converse, take $A = F(1)$ where $F$ is the free functor (Examples 2.1.3).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.1.32</span><span class="math-callout__name">(Monics and pullbacks)</span></p>

*A map $X \xrightarrow{f} Y$ is monic if and only if the square*

$$
\begin{array}{ccc}
X & \xrightarrow{1} & X \\
\downarrow 1 & & \downarrow f \\
X & \xrightarrow{f} & Y
\end{array}
$$

*is a pullback.* $\square$

</div>

The significance of this lemma is that whenever we prove a result about limits, a result about monics will follow. For example, we will soon show that the forgetful functors from $\mathbf{Grp}$, $\mathbf{Vect}_k$, etc., to $\mathbf{Set}$ preserve limits (in a sense to be defined), from which it will follow immediately that they also preserve monics. This in turn gives an alternative proof that monics in these categories are injective.

### 5.2 Colimits: Definition and Examples

We have seen that examples of limits occur throughout mathematics. It therefore makes sense to examine the dual concept, colimit, and ask whether it is similarly ubiquitous.

By dualizing, we can write down the definition of colimit immediately. We then specialize to sums, coequalizers and pushouts, the duals of products, equalizers and pullbacks.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.2.1</span><span class="math-callout__name">(Cocone and colimit)</span></p>

Let $\mathscr{A}$ be a category and $\mathbf{I}$ a small category. Let $D \colon \mathbf{I} \to \mathscr{A}$ be a diagram in $\mathscr{A}$, and write $D^{\mathrm{op}}$ for the corresponding functor $\mathbf{I}^{\mathrm{op}} \to \mathscr{A}^{\mathrm{op}}$. A **cocone** on $D$ is a cone on $D^{\mathrm{op}}$, and a **colimit** of $D$ is a limit of $D^{\mathrm{op}}$.

Explicitly, a cocone on $D$ is an object $A \in \mathscr{A}$ (the **vertex**) together with a family $(D(I) \xrightarrow{f_I} A)\_{I \in \mathbf{I}}$ of maps in $\mathscr{A}$ such that for all maps $I \xrightarrow{u} J$ in $\mathbf{I}$, the diagram $f_I = f_J \circ Du$ commutes. A colimit of $D$ is a cocone $(D(I) \xrightarrow{p_I} C)\_{I \in \mathbf{I}}$ with the property that for any cocone $(D(I) \xrightarrow{f_I} A)\_{I \in \mathbf{I}}$ on $D$, there is a unique map $\bar{f} \colon C \to A$ such that $\bar{f} \circ p_I = f_I$ for all $I \in \mathbf{I}$.

We write (the vertex of) the colimit as $\varinjlim D$, and call the maps $p_I$ **coprojections**.

</div>

#### Sums

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.2.2</span><span class="math-callout__name">(Sum / coproduct)</span></p>

A **sum** or **coproduct** is a colimit over a discrete category. (That is, it is a colimit of shape $\mathbf{I}$ for some discrete category $\mathbf{I}$.)

Let $(X_i)\_{i \in I}$ be a family of objects of a category. Their sum (if it exists) is written as $\sum_{i \in I} X_i$ or $\coprod_{i \in I} X_i$. When $I$ is a finite set $\lbrace 1, \ldots, n\rbrace$, we write $\sum_{i \in I} X_i$ as $X_1 + \cdots + X_n$, or as $0$ if $n = 0$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.2.3</span><span class="math-callout__name">(Empty sum is initial object)</span></p>

By the dual of Example 5.1.9, a sum of the empty family is exactly an initial object.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.2.4</span><span class="math-callout__name">(Sums in $\mathbf{Set}$)</span></p>

Sums in $\mathbf{Set}$ were described in Section 3.1. Take two sets $X_1$ and $X_2$. Form their sum $X_1 + X_2$, and consider the inclusions $X_1 \xrightarrow{p_1} X_1 + X_2 \xleftarrow{p_2} X_2$. This is a colimit cocone. To prove this: for any diagram $X_1 \xrightarrow{f_1} A \xleftarrow{f_2} X_2$ of sets and functions, there is a unique function $\bar{f} \colon X_1 + X_2 \to A$ making the diagram commute. This is because $p_1$ and $p_2$ are injections whose images partition $X_1 + X_2$, so every element of $X_1 + X_2$ is *either* equal to $p_1(x_1)$ for some unique $x_1 \in X_1$, *or* equal to $p_2(x_2)$ for some unique $x_2 \in X_2$, but not both. We define $\bar{f}(x) = f_1(x_1)$ in the first case and $\bar{f}(x) = f_2(x_2)$ in the second.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.2.5</span><span class="math-callout__name">(Sums in $\mathbf{Vect}_k$)</span></p>

Let $X_1$ and $X_2$ be vector spaces. There are linear maps $X_1 \xrightarrow{i_1} X_1 \oplus X_2 \xleftarrow{i_2} X_2$ defined by $i_1(x_1) = (x_1, 0)$ and $i_2(x_2) = (0, x_2)$, and it can be checked that this is a colimit cocone in $\mathbf{Vect}_k$. Hence binary direct sums are sums in the categorical sense. This is remarkable, since we saw in Example 5.1.5 that $X_1 \oplus X_2$ is also the *product* of $X_1$ and $X_2$! Contrast this with the category of sets (or almost any other category), where sums and products are very different.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.2.6</span><span class="math-callout__name">(Sums in ordered sets)</span></p>

Let $(A, \le)$ be an ordered set. **Upper bounds** and **least upper bounds** (or **joins**) in $A$ are defined by dualizing the definitions in Example 5.1.6, and, dually, they are sums in the corresponding category. The join of a family $(x_i)\_{i \in I}$ is written as $\bigvee_{i \in I} x_i$. In the binary case, the join of $x_1$ and $x_2$ is written as $x_1 \vee x_2$. A join of the empty family (where $I = \emptyset$) is an initial object of the category $A$, as in Example 5.2.3. Equivalently, it is a **least element** of $A$: an element $0 \in A$ such that $0 \le a$ for all $a \in A$.

For instance, in $(\mathbb{R}, \le)$, join is supremum and there is no least element. In a power set $(\mathscr{P}(S), \subseteq)$, join is union and the least element is $\emptyset$. In $(\mathbb{N}, \mid)$, join is lowest common multiple and the least element is $1$ (since $1$ divides everything). But also, everything divides $0$, so $0$ is greatest!

</div>

#### Coequalizers

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.2.7</span><span class="math-callout__name">(Coequalizer)</span></p>

A **coequalizer** is a colimit of shape $\mathbf{E}$.

In other words, given a diagram $X \overset{s}{\underset{t}{\rightrightarrows}} Y$, a coequalizer of $s$ and $t$ is a map $Y \xrightarrow{p} C$ satisfying $p \circ s = p \circ t$ and universal with this property.

</div>

Coequalizers are something like quotients. Given any equivalence relation $\sim$ on a set $A$, we can form the set $A/{\sim}$ of equivalence classes and the quotient map $p \colon A \to A/{\sim}$. For any set $B$, the maps $A/{\sim} \to B$ correspond one-to-one with the maps $f \colon A \to B$ such that $a \sim a' \implies f(a) = f(a')$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.2.9</span><span class="math-callout__name">(Coequalizers in $\mathbf{Set}$)</span></p>

Take sets and functions $X \overset{s}{\underset{t}{\rightrightarrows}} Y$. To find the coequalizer of $s$ and $t$, we must construct in some canonical way a set $C$ and a function $p \colon Y \to C$ such that $p(s(x)) = p(t(x))$ for all $x \in X$. So, let $\sim$ be the equivalence relation on $Y$ generated by $s(x) \sim t(x)$ for all $x \in X$, and take the quotient map $p \colon Y \to Y/{\sim}$. This is indeed the coequalizer of $s$ and $t$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.2.10</span><span class="math-callout__name">(Coequalizers in $\mathbf{Ab}$)</span></p>

For each pair of homomorphisms $A \overset{s}{\underset{t}{\rightrightarrows}} B$ in $\mathbf{Ab}$, there is a homomorphism $t - s \colon A \to B$, which gives rise to a subgroup $\mathrm{im}(t - s)$ of $B$. The coequalizer of $s$ and $t$ is the canonical homomorphism $B \to B/\mathrm{im}(t - s)$. (Compare Example 5.1.15.)

</div>

#### Pushouts

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.2.11</span><span class="math-callout__name">(Pushout)</span></p>

A **pushout** is a colimit of shape $\mathbf{P}^{\mathrm{op}}$.

In other words, the pushout of a diagram $X \xrightarrow{s} Y$, $X \xrightarrow{t} Z$ is (if it exists) a commutative square that is universal as such. A pushout in a category $\mathscr{A}$ is a pullback in $\mathscr{A}^{\mathrm{op}}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.2.12</span><span class="math-callout__name">(Pushouts in $\mathbf{Set}$)</span></p>

Take a diagram $X \xrightarrow{s} Y$, $X \xrightarrow{t} Z$ in $\mathbf{Set}$. Its pushout $P$ is $(Y + Z)/{\sim}$, where $\sim$ is the equivalence relation on $Y + Z$ generated by $s(x) \sim t(x)$ for all $x \in X$. The coprojection $Y \to P$ sends $y \in Y$ to its equivalence class in $P$, and similarly for the coprojection $Z \to P$.

For example, let $Y$ and $Z$ be subsets of some set $A$. Then the square with $Y \cap Z$ at the top-left, $Y$ at the top-right, $Z$ at the bottom-left, and $Y \cup Z$ at the bottom-right (all arrows being inclusions) is a pushout square in $\mathbf{Set}$. (It is also a pullback square! This coincidence is a special property of the category of sets.)

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.2.13</span><span class="math-callout__name">(Pushouts and initial objects)</span></p>

If $\mathscr{A}$ is a category with an initial object $0$, and if $Y, Z \in \mathscr{A}$, then a pushout of the unique diagram $0 \to Y$, $0 \to Z$ is exactly a sum of $Y$ and $Z$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.2.14</span><span class="math-callout__name">(Van Kampen theorem)</span></p>

The van Kampen theorem says that given a pushout square in $\mathbf{Top}$ satisfying certain further hypotheses, the square in $\mathbf{Grp}$ obtained by taking fundamental groups throughout is also a pushout.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.2.15</span><span class="math-callout__name">(Direct limits)</span></p>

A diagram $D \colon (\mathbb{N}, \le) \to \mathscr{A}$ consists of objects and maps

$$
X_0 \xrightarrow{s_1} X_1 \xrightarrow{s_2} X_2 \xrightarrow{s_3} \cdots
$$

in $\mathscr{A}$. Colimits of such diagrams are traditionally called **direct limits**. Although the old terms "inverse limit" and "direct limit" are made redundant by the general categorical terms "limit" and "colimit" respectively, it is worth being aware of them.

</div>

#### General Formula for Colimits in $\mathbf{Set}$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.2.16</span><span class="math-callout__name">(Colimits in $\mathbf{Set}$)</span></p>

The colimit of a diagram $D \colon \mathbf{I} \to \mathbf{Set}$ is given by

$$
\varinjlim D = \left(\sum_{I \in \mathbf{I}} D(I)\right)\!/{\sim}
$$

where $\sim$ is the equivalence relation on $\sum D(I)$ generated by

$$
x \sim (Du)(x)
$$

for all $I \xrightarrow{u} J$ in $\mathbf{I}$ and $x \in D(I)$. For any set $A$, the maps $\left(\sum D(I)\right)/{\sim} \;\to A$ correspond bijectively with the maps $f \colon \sum D(I) \to A$ such that $f(x) = f((Du)(x))$ for all $u$ and $x$; but these are exactly the cocones on $D$ with vertex $A$.

</div>

There is a duality between the formulas for limits in $\mathbf{Set}$ (Example 5.1.22) and colimits in $\mathbf{Set}$. Whereas the limit is constructed as a *subset* of a *product*, the colimit is a *quotient* of a *sum*.

#### Epics

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.2.17</span><span class="math-callout__name">(Epic / epimorphism)</span></p>

Let $\mathscr{A}$ be a category. A map $X \xrightarrow{f} Y$ in $\mathscr{A}$ is **epic** (or an **epimorphism**) if for all objects $Z$ and maps $Y \overset{g}{\underset{g'}{\rightrightarrows}} Z$,

$$
g \circ f = g' \circ f \implies g = g'.
$$

This is the formal dual of the definition of monic. (In other words, an epic in $\mathscr{A}$ is a monic in $\mathscr{A}^{\mathrm{op}}$.) It is in some sense the categorical version of surjectivity. But whereas the definition of monic closely resembles the definition of injective, the definition of epic does not look much like the definition of surjective.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.2.18</span><span class="math-callout__name">(Epics in $\mathbf{Set}$)</span></p>

In $\mathbf{Set}$, a map is epic if and only if it is surjective. If $f$ is surjective then certainly $f$ is epic. To see the converse, take $Z$ to be a two-element set $\lbrace\mathtt{true}, \mathtt{false}\rbrace$, take $g$ to be the characteristic function of the image of $f$, and take $g'$ to be the function with constant value $\mathtt{true}$.

Any isomorphism in any category is both monic and epic. In $\mathbf{Set}$, the converse also holds, since any injective surjective function is invertible (Example 1.1.5).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.2.19</span><span class="math-callout__name">(Epics in categories of algebras)</span></p>

In categories of algebras, any surjective map is certainly epic. In some such categories, including $\mathbf{Ab}$, $\mathbf{Vect}_k$ and $\mathbf{Grp}$, the converse also holds. However, there are other categories of algebras where it fails. For instance, in $\mathbf{Ring}$, the inclusion $\mathbb{Z} \hookrightarrow \mathbb{Q}$ is epic but not surjective (Exercise 5.2.23). This is also an example of a map that is monic and epic but not an isomorphism.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.2.20</span><span class="math-callout__name">(Epics in Hausdorff spaces)</span></p>

In the category of Hausdorff topological spaces and continuous maps, any map with dense image is epic.

</div>

There is a dual of Lemma 5.1.32, saying that a map is epic if and only if a certain square is a pushout.

### 5.3 Interactions between Functors and Limits

We saw in Example 5.1.23 that limits in categories such as $\mathbf{Grp}$, $\mathbf{Ring}$ and $\mathbf{Vect}_k$ can be computed by first taking the limit in the category of sets, then equipping the result with a suitable algebraic structure. On the other hand, colimits in these categories are unlike colimits in $\mathbf{Set}$. For example, the underlying set of the initial object of $\mathbf{Grp}$ (which has one element) is not the initial object of $\mathbf{Set}$ (which has no elements), and the underlying set of the direct sum $X \oplus Y$ of two vector spaces is not the sum of the underlying sets of $X$ and $Y$. So, these forgetful functors interact well with limits and badly with colimits.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.3.1</span><span class="math-callout__name">(Preservation, reflection and creation of limits)</span></p>

**(a)** A functor $F \colon \mathscr{A} \to \mathscr{B}$ **preserves limits of shape $\mathbf{I}$** if for all diagrams $D \colon \mathbf{I} \to \mathscr{A}$ and all cones $(A \xrightarrow{p_I} D(I))\_{I \in \mathbf{I}}$ on $D$,

$$
(A \xrightarrow{p_I} D(I))_{I \in \mathbf{I}} \text{ is a limit cone on } D \text{ in } \mathscr{A}
$$

$$
\implies (F(A) \xrightarrow{Fp_I} FD(I))_{I \in \mathbf{I}} \text{ is a limit cone on } F \circ D \text{ in } \mathscr{B}.
$$

**(b)** A functor $F \colon \mathscr{A} \to \mathscr{B}$ **preserves limits** if it preserves limits of shape $\mathbf{I}$ for all small categories $\mathbf{I}$.

**(c)** **Reflection** of limits is defined as in (a), but with $\impliedby$ in place of $\implies$.

Of course, the same terminology applies to colimits.

</div>

An equivalent formulation: a functor $F \colon \mathscr{A} \to \mathscr{B}$ preserves limits if and only if whenever $D \colon \mathbf{I} \to \mathscr{A}$ is a diagram that has a limit, the composite $F \circ D \colon \mathbf{I} \to \mathscr{B}$ also has a limit, and the canonical map

$$
F\!\left(\varprojlim_\mathbf{I} D\right) \to \varprojlim_\mathbf{I}(F \circ D)
$$

is an isomorphism. Here the "canonical map" has $I$-component $F(p_I)$, where $p_I$ is the $I$-th projection of the limit cone on $D$. In particular, if $F$ preserves limits then

$$
F\!\left(\varprojlim_\mathbf{I} D\right) \cong \varprojlim_\mathbf{I}(F \circ D)
$$

whenever $D$ is a diagram with a limit. Preservation of limits says more than this: the left- and right-hand sides are required to be not just isomorphic, but isomorphic *in a particular way*.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.3.2</span><span class="math-callout__name">(Forgetful functor $U \colon \mathbf{Top} \to \mathbf{Set}$)</span></p>

The forgetful functor $U \colon \mathbf{Top} \to \mathbf{Set}$ preserves both limits and colimits. (As we will see, this follows from the fact that $U$ has adjoints on both sides.) It does not reflect all limits or all colimits. For instance, choose any non-discrete spaces $X$ and $Y$, and let $Z$ be the set $U(X) \times U(Y)$ equipped with the discrete topology. Then $X \leftarrow Z \to Y$ is a cone in $\mathbf{Top}$ whose image in $\mathbf{Set}$ is a product cone $U(X) \leftarrow U(X) \times U(Y) \to U(Y)$, but it is not a product cone in $\mathbf{Top}$, since the discrete topology on $U(X) \times U(Y)$ is strictly larger than the product topology.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.3.3</span><span class="math-callout__name">(Forgetful functors seldom preserve colimits)</span></p>

The forgetful functor $\mathbf{Grp} \to \mathbf{Set}$ does not preserve initial objects and the forgetful functor $\mathbf{Vect}_k \to \mathbf{Set}$ does not preserve binary sums. Forgetful functors out of categories of algebras very seldom preserve all colimits.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 5.3.4</span><span class="math-callout__name">(Forgetful functors on algebras preserve limits)</span></p>

We saw that (in the examples mentioned) forgetful functors on categories of algebras do preserve limits. In fact, something stronger is true. Let us examine the case of binary products in $\mathbf{Grp}$, although all of the following can be said for any limits in any of the categories $\mathbf{Grp}$, $\mathbf{Ab}$, $\mathbf{Vect}_k$, $\mathbf{Ring}$, etc.

Take groups $X_1$ and $X_2$. We can form the product set $U(X_1) \times U(X_2)$, which comes equipped with projections $U(X_1) \xleftarrow{p_1} U(X_1) \times U(X_2) \xrightarrow{p_2} U(X_2)$. There is exactly one group structure on the set $U(X_1) \times U(X_2)$ with the property that $p_1$ and $p_2$ are homomorphisms: the componentwise operations $(x_1, x_2) \cdot (x'_1, x'_2) = (x_1 x'_1, x_2 x'_2)$, $(x_1, x_2)^{-1} = (x_1^{-1}, x_2^{-1})$, and identity $(1, 1)$. Write $L$ for the set $U(X_1) \times U(X_2)$ equipped with this group structure. Then $X_1 \xleftarrow{p_1} L \xrightarrow{p_2} X_2$ is a product cone in $\mathbf{Grp}$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 5.3.5</span><span class="math-callout__name">(Creation of limits)</span></p>

A functor $F \colon \mathscr{A} \to \mathscr{B}$ **creates limits (of shape $\mathbf{I}$)** if whenever $D \colon \mathbf{I} \to \mathscr{A}$ is a diagram in $\mathscr{A}$,

- for any limit cone $(B \xrightarrow{q_I} FD(I))\_{I \in \mathbf{I}}$ on the diagram $F \circ D$, there is a unique cone $(A \xrightarrow{p_I} D(I))\_{I \in \mathbf{I}}$ on $D$ such that $F(A) = B$ and $F(p_I) = q_I$ for all $I \in \mathbf{I}$;
- this cone $(A \xrightarrow{p_I} D(I))\_{I \in \mathbf{I}}$ is a limit cone on $D$.

The forgetful functors from $\mathbf{Grp}$, $\mathbf{Ring}$, $\ldots$ to $\mathbf{Set}$ all create limits (Exercise 5.3.11).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 5.3.6</span><span class="math-callout__name">(Creation implies preservation)</span></p>

*Let $F \colon \mathscr{A} \to \mathscr{B}$ be a functor and $\mathbf{I}$ a small category. Suppose that $\mathscr{B}$ has, and $F$ creates, limits of shape $\mathbf{I}$. Then $\mathscr{A}$ has, and $F$ preserves, limits of shape $\mathbf{I}$.*

**Proof.** Exercise 5.3.12. $\square$

</div>

Since $\mathbf{Set}$ has all limits, it follows that all our categories of algebras have all limits, and that the forgetful functors preserve them.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 5.3.7</span><span class="math-callout__name">(Strictness of creation)</span></p>

There is something suspicious about Definition 5.3.5. It refers to *equality* of objects of a category, a relation that, as we saw on page 31, is usually too strict to be appropriate. It is almost always better to replace equality by isomorphism. If we replace equality by isomorphism throughout the definition of "creates limits", we obtain a more healthy and inclusive notion. In fact, what we are calling creation of limits should really be called *strict* creation of limits, with "creation of limits" reserved for the more inclusive notion. That is how "creates" is used in most of the literature.

</div>

## Chapter 6: Adjoints, Representables and Limits

We have approached the idea of universal property from three different angles, producing three different formalisms: adjointness, representability, and limits. In this final chapter, we work out the connections between them.

In principle, anything that can be described in one of the three formalisms can also be described in the others. The situation is similar to that of cartesian and polar coordinates: anything that can be done in polar coordinates can in principle be done in cartesian coordinates, and vice versa, but some things are more gracefully done in one system than the other.

Key highlights:

- Limits and colimits in functor categories work in the simplest possible way.
- The embedding of a category $\mathbf{A}$ into its presheaf category $[\mathbf{A}^{\mathrm{op}}, \mathbf{Set}]$ preserves limits (but not colimits).
- The representables are the prime numbers of presheaves: every presheaf can be expressed canonically as a colimit of representables.
- A functor with a left adjoint preserves limits. Under suitable hypotheses, the converse holds too.
- Categories of presheaves $[\mathbf{A}^{\mathrm{op}}, \mathbf{Set}]$ behave very much like the category of sets.

### 6.1 Limits in Terms of Representables and Adjoints

There is more than one way to present the definition of limit. In Chapter 5, we used an explicit form that is particularly convenient for examples. But for developing the *theory* of limits and colimits, a rephrased form is useful. We rephrase it in two different ways: once in terms of representability, and once in terms of adjoints.

#### The Diagonal Functor and Cones

Given categories $\mathbf{I}$ and $\mathscr{A}$ and an object $A \in \mathscr{A}$, there is a functor $\Delta A \colon \mathbf{I} \to \mathscr{A}$ with constant value $A$ on objects and $1_A$ on maps. This defines, for each $\mathbf{I}$ and $\mathscr{A}$, the **diagonal functor**

$$
\Delta \colon \mathscr{A} \to [\mathbf{I}, \mathscr{A}].
$$

Given a diagram $D \colon \mathbf{I} \to \mathscr{A}$ and an object $A \in \mathscr{A}$, a cone on $D$ with vertex $A$ is simply a natural transformation $\Delta A \Rightarrow D$. Writing $\mathrm{Cone}(A, D)$ for the set of cones on $D$ with vertex $A$, we therefore have

$$
\mathrm{Cone}(A, D) = [\mathbf{I}, \mathscr{A}](\Delta A, D).
$$

Thus, $\mathrm{Cone}(A, D)$ is functorial in $A$ (contravariantly) and $D$ (covariantly).

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.1.1</span><span class="math-callout__name">(Limits as representations)</span></p>

*Let $\mathbf{I}$ be a small category, $\mathscr{A}$ a category, and $D \colon \mathbf{I} \to \mathscr{A}$ a diagram. Then there is a one-to-one correspondence between limit cones on $D$ and representations of the functor*

$$
\mathrm{Cone}(-, D) \colon \mathscr{A}^{\mathrm{op}} \to \mathbf{Set},
$$

*with the representing objects of $\mathrm{Cone}(-, D)$ being the limit objects (that is, the vertices of the limit cones) of $D$.*

*Briefly put: a limit of $D$ is a representation of $[\mathbf{I}, \mathscr{A}](\Delta -, D)$.*

**Proof.** By Corollary 4.3.2, a representation of $\mathrm{Cone}(-, D)$ consists of a cone on $D$ with a certain universal property. This is exactly the universal property in the definition of limit cone. $\square$

</div>

The proposition implies that if $D$ has a limit then

$$
\mathrm{Cone}(A, D) \cong \mathscr{A}\!\left(A, \varprojlim_\mathbf{I} D\right)
$$

naturally in $A$. The correspondence is given from left to right by $(f_I)\_{I \in \mathbf{I}} \mapsto \bar{f}$, and from right to left by $g \mapsto (p_I \circ g)\_{I \in \mathbf{I}}$, where $p_I \colon \varprojlim D \to D(I)$ are the projections.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 6.1.2</span><span class="math-callout__name">(Uniqueness of limits)</span></p>

*Limits are unique up to isomorphism.* $\square$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 6.1.3</span><span class="math-callout__name">(Functoriality of limits)</span></p>

*Let $\mathbf{I}$ be a small category and $\alpha \colon D \Rightarrow D'$ a natural transformation between diagrams $D, D' \colon \mathbf{I} \to \mathscr{A}$. Let $({\varprojlim} D \xrightarrow{p_I} D(I))\_{I \in \mathbf{I}}$ and $({\varprojlim} D' \xrightarrow{p'_I} D'(I))\_{I \in \mathbf{I}}$ be limit cones. Then:*

*(a) there is a unique map $\varprojlim \alpha \colon \varprojlim D \to \varprojlim D'$ such that for all $I \in \mathbf{I}$, the square*

$$
\begin{array}{ccc}
\varprojlim D & \xrightarrow{p_I} & D(I) \\
{\scriptstyle \varprojlim \alpha}\downarrow & & \downarrow{\scriptstyle \alpha_I} \\
\varprojlim D' & \xrightarrow{p'_I} & D'(I)
\end{array}
$$

*commutes;*

*(b) given cones $(A \xrightarrow{f_I} D(I))\_{I \in \mathbf{I}}$ and $(A' \xrightarrow{f'_I} D'(I))\_{I \in \mathbf{I}}$ and a map $s \colon A \to A'$ such that $\alpha_I \circ f_I = f'_I \circ s$ commutes for all $I \in \mathbf{I}$, then the square*

$$
\begin{array}{ccc}
A & \xrightarrow{\bar{f}} & \varprojlim D \\
{\scriptstyle s}\downarrow & & \downarrow{\scriptstyle \varprojlim \alpha} \\
A' & \xrightarrow{\bar{f}'} & \varprojlim D'
\end{array}
$$

*also commutes.*

**Proof.** Part (a) follows immediately from the fact that $(\varprojlim D \xrightarrow{\alpha_I p_I} D'(I))\_{I \in \mathbf{I}}$ is a cone on $D'$. For (b), note that for each $I \in \mathbf{I}$, $p'_I \circ (\varprojlim \alpha) \circ \bar{f} = \alpha_I \circ p_I \circ \bar{f} = \alpha_I \circ f_I = f'_I \circ s = p'_I \circ \bar{f}' \circ s$. So by Exercise 5.1.36(a), $(\varprojlim \alpha) \circ \bar{f} = \bar{f}' \circ s$. $\square$

</div>

#### Limits as Adjoints

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.1.4</span><span class="math-callout__name">(Limits as right adjoints)</span></p>

*Let $\mathbf{I}$ be a small category and $\mathscr{A}$ a category with all limits of shape $\mathbf{I}$. Then $\varprojlim$ defines a functor $[\mathbf{I}, \mathscr{A}] \to \mathscr{A}$, and this functor is right adjoint to the diagonal functor.*

**Proof.** Choose for each $D \in [\mathbf{I}, \mathscr{A}]$ a limit cone on $D$, and call its vertex $\varprojlim D$. For each map $\alpha \colon D \to D'$ in $[\mathbf{I}, \mathscr{A}]$, we have a canonical map $\varprojlim \alpha \colon \varprojlim D \to \varprojlim D'$, defined as in Lemma 6.1.3(a). This makes $\varprojlim$ into a functor. Proposition 6.1.1 implies that

$$
[\mathbf{I}, \mathscr{A}](\Delta A, D) = \mathrm{Cone}(A, D) \cong \mathscr{A}\!\left(A, \varprojlim_\mathbf{I} D\right)
$$

naturally in $A$, and taking $s = 1_A$ in Lemma 6.1.3(b) tells us that the isomorphism is also natural in $D$. $\square$

</div>

To define the functor $\varprojlim$, we had to *choose* for each $D$ a limit cone on $D$. This is a non-canonical choice. Nevertheless, different choices only affect the functor $\varprojlim$ up to natural isomorphism, by uniqueness of adjoints.

### 6.2 Limits and Colimits of Presheaves

What do limits and colimits look like in functor categories $[\mathscr{A}, \mathscr{B}]$? In particular, what do they look like in presheaf categories $[\mathscr{A}^{\mathrm{op}}, \mathbf{Set}]$? More particularly still, what about limits and colimits of representables? Are they, too, representable?

#### Representables Preserve Limits

By definition of product, a map $A \to X \times Y$ amounts to a pair of maps $(A \to X, A \to Y)$. So there is a bijection

$$
\mathscr{A}(A, X \times Y) \cong \mathscr{A}(A, X) \times \mathscr{A}(A, Y)
$$

natural in $A, X, Y \in \mathscr{A}$. Similarly, by definition of equalizer, maps $A \to \mathrm{Eq}(X \overset{s}{\underset{t}{\rightrightarrows}} Y)$ correspond one-to-one with maps $f \colon A \to X$ such that $s \circ f = t \circ f$. Writing $s_* = \mathscr{A}(A, s)$ and $t_* = \mathscr{A}(A, t)$, this gives

$$
\mathscr{A}\!\left(A, \mathrm{Eq}\!\left(X \overset{s}{\underset{t}{\rightrightarrows}} Y\right)\right) \cong \mathrm{Eq}\!\left(\mathscr{A}(A, X) \overset{s_*}{\underset{t_*}{\rightrightarrows}} \mathscr{A}(A, Y)\right).
$$

These isomorphisms suggest that more generally, we might have

$$
\mathscr{A}\!\left(A, \varprojlim_\mathbf{I} D\right) \cong \varprojlim_\mathbf{I} \mathscr{A}(A, D)
$$

naturally in $A \in \mathscr{A}$ and $D \in [\mathbf{I}, \mathscr{A}]$. Here $\mathscr{A}(A, D)$ is the functor $\mathbf{I} \to \mathbf{Set}$ given by $I \mapsto \mathscr{A}(A, D(I))$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma 6.2.1</span><span class="math-callout__name">(Cones as limits)</span></p>

*Let $\mathbf{I}$ be a small category, $\mathscr{A}$ a locally small category, $D \colon \mathbf{I} \to \mathscr{A}$ a diagram, and $A \in \mathscr{A}$. Then*

$$
\mathrm{Cone}(A, D) \cong \varprojlim_\mathbf{I} \mathscr{A}(A, D)
$$

*naturally in $A$ and $D$.*

**Proof.** Like all functors from a small category into $\mathbf{Set}$, the functor $\mathscr{A}(A, D)$ does have a limit, given by the explicit formula (5.16). According to this formula, $\varprojlim \mathscr{A}(A, D)$ is the set of all families $(f_I)\_{I \in \mathbf{I}}$ such that $f_I \in \mathscr{A}(A, D(I))$ for all $I \in \mathbf{I}$ and $(Du) \circ f_I = f_J$ for all $I \xrightarrow{u} J$ in $\mathbf{I}$. But an element of $\varprojlim \mathscr{A}(A, D)$ is nothing but a cone on $D$ with vertex $A$. $\square$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.2.2</span><span class="math-callout__name">(Representables preserve limits)</span></p>

*Let $\mathscr{A}$ be a locally small category and $A \in \mathscr{A}$. Then $\mathscr{A}(A, -) \colon \mathscr{A} \to \mathbf{Set}$ preserves limits.*

**Proof.** Let $\mathbf{I}$ be a small category and let $D \colon \mathbf{I} \to \mathscr{A}$ be a diagram that has a limit. Then

$$
\mathscr{A}\!\left(A, \varprojlim_\mathbf{I} D\right) \cong \mathrm{Cone}(A, D) \cong \varprojlim_\mathbf{I} \mathscr{A}(A, D)
$$

naturally in $A$. Here the first isomorphism is Proposition 6.1.1 and the second is Lemma 6.2.1. $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 6.2.3</span><span class="math-callout__name">(Dual formulation)</span></p>

Proposition 6.2.2 tells us that

$$
\mathscr{A}\!\left(A, \varprojlim_\mathbf{I} D\right) \cong \varprojlim_\mathbf{I} \mathscr{A}(A, D).
$$

To dualize, we replace $\mathscr{A}$ by $\mathscr{A}^{\mathrm{op}}$. Thus, $\mathscr{A}(-, A) \colon \mathscr{A}^{\mathrm{op}} \to \mathbf{Set}$ preserves limits. A limit in $\mathscr{A}^{\mathrm{op}}$ is a colimit in $\mathscr{A}$, so $\mathscr{A}(-, A)$ transforms colimits in $\mathscr{A}$ into limits in $\mathbf{Set}$:

$$
\mathscr{A}\!\left(\varinjlim_\mathbf{I} D, A\right) \cong \varprojlim_{\mathbf{I}^{\mathrm{op}}} \mathscr{A}(D, A).
$$

The right-hand side is a *limit*, not a colimit! Even though (6.8) and (6.9) are dual statements, there are, in total, more limits than colimits involved. Somehow, limits have the upper hand.

For example, $\mathscr{A}(X + Y, A) \cong \mathscr{A}(X, A) \times \mathscr{A}(Y, A)$.

</div>

#### Limits in Functor Categories

Earlier, we learned that it is sometimes useful to view functors as objects in their own right. We now begin an analysis of limits and colimits in functor categories $[\mathbf{A}, \mathscr{S}]$. Here $\mathbf{A}$ is small and $\mathscr{S}$ is locally small; the most important cases for us will be $\mathscr{S} = \mathbf{Set}$ and $\mathscr{S} = \mathbf{Set}^{\mathrm{op}}$.

Limits and colimits in $[\mathbf{A}, \mathscr{S}]$ work in the simplest way imaginable. For instance, if $\mathscr{S}$ has binary products then so does $[\mathbf{A}, \mathscr{S}]$, and the product of two functors $X, Y \colon \mathbf{A} \to \mathscr{S}$ is the functor $X \times Y \colon \mathbf{A} \to \mathscr{S}$ given by

$$
(X \times Y)(A) = X(A) \times Y(A)
$$

for all $A \in \mathbf{A}$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation 6.2.4</span><span class="math-callout__name">(Evaluation functor)</span></p>

Let $\mathbf{A}$ and $\mathscr{S}$ be categories. For each $A \in \mathbf{A}$, there is a functor

$$
\mathrm{ev}_A \colon [\mathbf{A}, \mathscr{S}] \to \mathscr{S}, \qquad X \mapsto X(A),
$$

called **evaluation** at $A$. Given a diagram $D \colon \mathbf{I} \to [\mathbf{A}, \mathscr{S}]$, we have for each $A \in \mathbf{A}$ a functor

$$
\mathrm{ev}_A \circ D \colon \mathbf{I} \to \mathscr{S}, \qquad I \mapsto D(I)(A).
$$

We write $\mathrm{ev}_A \circ D$ as $D(-)(A)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.2.5</span><span class="math-callout__name">(Limits in functor categories)</span></p>

*Let $\mathbf{A}$ and $\mathbf{I}$ be small categories and $\mathscr{S}$ a locally small category. Let $D \colon \mathbf{I} \to [\mathbf{A}, \mathscr{S}]$ be a diagram, and suppose that for each $A \in \mathbf{A}$, the diagram $D(-)(A) \colon \mathbf{I} \to \mathscr{S}$ has a limit. Then there is a cone on $D$ whose image under $\mathrm{ev}_A$ is a limit cone on $D(-)(A)$ for each $A \in \mathbf{A}$. Moreover, any such cone on $D$ is a limit cone.*

</div>

Theorem 6.2.5 is often expressed as a slogan: *Limits in a functor category are computed pointwise.* The "points" are the objects of $\mathbf{A}$. Of course, the dual states that colimits in a functor category are also computed pointwise.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 6.2.6</span><span class="math-callout__name">(Functor categories inherit limits)</span></p>

*Let $\mathbf{I}$ and $\mathbf{A}$ be small categories, and $\mathscr{S}$ a locally small category. If $\mathscr{S}$ has all limits (respectively, colimits) of shape $\mathbf{I}$ then so does $[\mathbf{A}, \mathscr{S}]$, and for each $A \in \mathbf{A}$, the evaluation functor $\mathrm{ev}_A \colon [\mathbf{A}, \mathscr{S}] \to \mathscr{S}$ preserves them.* $\square$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Warning 6.2.7</span><span class="math-callout__name">(Non-pointwise limits)</span></p>

If $\mathscr{S}$ does *not* have all limits of shape $\mathbf{I}$ then $[\mathbf{A}, \mathscr{S}]$ may contain limits of shape $\mathbf{I}$ that are not computed pointwise, that is, are not preserved by all the evaluation functors.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.2.8</span><span class="math-callout__name">(Limits commute with limits)</span></p>

*Let $\mathbf{I}$ and $\mathbf{J}$ be small categories. Let $\mathscr{S}$ be a locally small category with limits of shape $\mathbf{I}$ and of shape $\mathbf{J}$. Then for all $D \colon \mathbf{I} \times \mathbf{J} \to \mathscr{S}$, we have*

$$
\varprojlim_\mathbf{J}\, \varprojlim_\mathbf{I}\, D^\bullet \cong \varprojlim_{\mathbf{I} \times \mathbf{J}} D \cong \varprojlim_\mathbf{I}\, \varprojlim_\mathbf{J}\, D_\bullet,
$$

*and all these limits exist. In particular, $\mathscr{S}$ has limits of shape $\mathbf{I} \times \mathbf{J}$.*

</div>

This is sometimes half-jokingly called Fubini's theorem, as it is something like changing the order of integration in a double integral. The analogy is more appealing with *co*limits, since, like integrals, colimits can be thought of as a context-sensitive version of sums.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.2.9</span><span class="math-callout__name">(Binary products commute)</span></p>

When $\mathbf{I} = \mathbf{J}$ is the discrete category with two objects, Proposition 6.2.8 says that binary products commute with binary products: if $\mathscr{S}$ has binary products and $S_{11}, S_{12}, S_{21}, S_{22} \in \mathscr{S}$ then

$$
(S_{11} \times S_{21}) \times (S_{12} \times S_{22}) \cong \prod_{i,j \in \lbrace 1,2\rbrace} S_{ij} \cong (S_{11} \times S_{12}) \times (S_{21} \times S_{22}).
$$

More generally, there are canonical isomorphisms $S \times T \cong T \times S$ and $(S \times T) \times U \cong S \times (T \times U)$ in any category with binary products. If there is also a terminal object $1$, then $S \times 1 \cong S \cong 1 \times S$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Warning 6.2.10</span><span class="math-callout__name">(Limits do not commute with colimits)</span></p>

The dual of Proposition 6.2.8 states that colimits commute with colimits. For instance,

$$
(S_{11} + S_{21}) + (S_{12} + S_{22}) \cong (S_{11} + S_{12}) + (S_{21} + S_{22}).
$$

But limits do *not* in general commute with colimits. For instance, in general,

$$
(S_{11} + S_{21}) \times (S_{12} + S_{22}) \not\cong (S_{11} \times S_{12}) + (S_{21} \times S_{22}).
$$

A counterexample is given by taking $\mathscr{S} = \mathbf{Set}$ and each $S_{ij}$ to be a one-element set. Then the left-hand side has $(1 + 1) \times (1 + 1) = 4$ elements, whereas the right-hand side has $(1 \times 1) + (1 \times 1) = 2$ elements.

</div>

#### The Yoneda Embedding and Limits

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 6.2.11</span><span class="math-callout__name">(Presheaf categories have all limits and colimits)</span></p>

*Let $\mathbf{A}$ be a small category. Then $[\mathbf{A}^{\mathrm{op}}, \mathbf{Set}]$ has all limits and colimits, and for each $A \in \mathbf{A}$, the evaluation functor $\mathrm{ev}_A \colon [\mathbf{A}^{\mathrm{op}}, \mathbf{Set}] \to \mathbf{Set}$ preserves them.* $\square$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary 6.2.12</span><span class="math-callout__name">(Yoneda embedding preserves limits)</span></p>

*The Yoneda embedding $H_\bullet \colon \mathbf{A} \to [\mathbf{A}^{\mathrm{op}}, \mathbf{Set}]$ preserves limits, for any small category $\mathbf{A}$.*

**Proof.** Let $D \colon \mathbf{I} \to \mathbf{A}$ be a diagram in $\mathbf{A}$, and let $(\varprojlim D \xrightarrow{p_I} D(I))\_{I \in \mathbf{I}}$ be a limit cone. For each $A \in \mathbf{A}$, the composite $\mathbf{A} \xrightarrow{H_\bullet} [\mathbf{A}^{\mathrm{op}}, \mathbf{Set}] \xrightarrow{\mathrm{ev}_A} \mathbf{Set}$ is $H^A$, which preserves limits (Proposition 6.2.2). So for each $A \in \mathbf{A}$,

$$
\left(\mathrm{ev}_A\, H_\bullet\!\left(\varprojlim_\mathbf{I} D\right) \xrightarrow{\mathrm{ev}_A\, H_\bullet(p_I)} \mathrm{ev}_A\, H_\bullet(D(I))\right)_{I \in \mathbf{I}}
$$

is a limit cone. But then, by the "moreover" part of Theorem 6.2.5 applied to the diagram $H_\bullet \circ D$ in $[\mathbf{A}^{\mathrm{op}}, \mathbf{Set}]$, the cone $(H_\bullet(\varprojlim D) \xrightarrow{H_\bullet(p_I)} H_\bullet(D(I)))\_{I \in \mathbf{I}}$ is also a limit, as required. $\square$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.2.13</span><span class="math-callout__name">(Yoneda preserves products)</span></p>

Let $\mathbf{A}$ be a category with binary products. Corollary 6.2.12 implies that for all $X, Y \in \mathbf{A}$,

$$
H_{X \times Y} \cong H_X \times H_Y
$$

in $[\mathbf{A}^{\mathrm{op}}, \mathbf{Set}]$. When evaluated at a particular object $A$, this says $\mathbf{A}(A, X \times Y) \cong \mathbf{A}(A, X) \times \mathbf{A}(A, Y)$.

If $\mathbf{A}$ has all limits, taking limits does not help us escape from $\mathbf{A}$ into the rest of $[\mathbf{A}^{\mathrm{op}}, \mathbf{Set}]$: any limit of representable presheaves is again representable.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Warning 6.2.14</span><span class="math-callout__name">(Yoneda does not preserve colimits)</span></p>

The Yoneda embedding does *not* preserve colimits. For example, if $\mathbf{A}$ has an initial object $0$ then $H_0$ is not initial, since $H_0(0) = \mathbf{A}(0, 0)$ is a one-element set, whereas the initial object of $[\mathbf{A}^{\mathrm{op}}, \mathbf{Set}]$ is the presheaf with constant value $\emptyset$.

</div>

#### Every Presheaf is a Colimit of Representables

We now know that the Yoneda embedding preserves limits but not colimits. In fact, the situation for colimits is at the opposite extreme from the situation for limits: by taking colimits of representable presheaves, we can obtain any presheaf we like! The representables are the building blocks of presheaves.

Every positive integer can be expressed as a product of primes in an essentially unique way. Somewhat similarly, every presheaf can be expressed as a colimit of representables in a canonical (though not unique) way.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.2.15</span><span class="math-callout__name">(Presheaves on a discrete category)</span></p>

Let $\mathbf{A}$ be the discrete category with two objects, $K$ and $L$. A presheaf $X$ on $\mathbf{A}$ is just a pair $(X(K), X(L))$ of sets, and $[\mathbf{A}^{\mathrm{op}}, \mathbf{Set}] \cong \mathbf{Set} \times \mathbf{Set}$. There are two representables, $H_K \cong (1, \emptyset)$ and $H_L \cong (\emptyset, 1)$. Every object of $\mathbf{Set} \times \mathbf{Set}$ is a sum of copies of $(1, \emptyset)$ and $(\emptyset, 1)$. For instance, if $X(K)$ has three elements and $X(L)$ has two elements, then $X \cong H_K + H_K + H_K + H_L + H_L$, exhibiting $X$ as a sum of representables.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6.2.16</span><span class="math-callout__name">(Category of elements)</span></p>

Let $\mathbf{A}$ be a category and $X$ a presheaf on $\mathbf{A}$. The **category of elements** $\mathbf{E}(X)$ of $X$ is the category in which:

- objects are pairs $(A, x)$ with $A \in \mathbf{A}$ and $x \in X(A)$;
- maps $(A', x') \to (A, x)$ are maps $f \colon A' \to A$ in $\mathbf{A}$ such that $(Xf)(x) = x'$.

There is a projection functor $P \colon \mathbf{E}(X) \to \mathbf{A}$ defined by $P(A, x) = A$ and $P(f) = f$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.2.17</span><span class="math-callout__name">(Density theorem)</span></p>

*Let $\mathbf{A}$ be a small category and $X$ a presheaf on $\mathbf{A}$. Then $X$ is the colimit of the diagram*

$$
\mathbf{E}(X) \xrightarrow{P} \mathbf{A} \xrightarrow{H_\bullet} [\mathbf{A}^{\mathrm{op}}, \mathbf{Set}]
$$

*in $[\mathbf{A}^{\mathrm{op}}, \mathbf{Set}]$; that is, $X \cong \varinjlim_{\mathbf{E}(X)} (H_\bullet \circ P)$.*

</div>

The density theorem states that every presheaf is a colimit of representables in a canonical way. It is secretly dual to the Yoneda lemma.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remarks 6.2.19</span><span class="math-callout__name">(Category of elements and density)</span></p>

**(a)** The term "category of elements" is compatible with the generalized element terminology introduced in Definition 4.1.25. The Yoneda lemma implies that for a presheaf $X$, the generalized elements of $X$ of representable shape correspond to pairs $(A, x)$ with $A \in \mathbf{A}$ and $x \in X(A)$. In other words, they are the objects of the category of elements.

**(b)** In topology, a subspace $A$ of a space $B$ is called dense if every point in $B$ can be obtained as a limit of points in $A$. This provides some explanation for the name of Theorem 6.2.17: the category $\mathbf{A}$ is "dense" in $[\mathbf{A}^{\mathrm{op}}, \mathbf{Set}]$ because every object of $[\mathbf{A}^{\mathrm{op}}, \mathbf{Set}]$ can be obtained as a colimit of objects of $\mathbf{A}$.

</div>

### 6.3 Interactions between Adjoint Functors and Limits

We saw in Proposition 4.1.11 that any set-valued functor with a left adjoint is representable, and in Proposition 6.2.2 that any representable preserves limits. Hence, any set-valued functor with a left adjoint preserves limits. In fact, this conclusion holds not only for set-valued functors, but in complete generality.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.3.1</span><span class="math-callout__name">(Adjoints preserve limits and colimits)</span></p>

*Let $\mathscr{A} \overset{F}{\underset{G}{\rightleftarrows}} \mathscr{B}$ be an adjunction ($F \dashv G$). Then $F$ preserves colimits and $G$ preserves limits.*

**Proof.** By duality, it is enough to prove that $G$ preserves limits. Let $D \colon \mathbf{I} \to \mathscr{B}$ be a diagram for which a limit exists. Then

$$
\mathscr{A}\!\left(A, G\!\left(\varprojlim_\mathbf{I} D\right)\right) \cong \mathscr{B}\!\left(F(A), \varprojlim_\mathbf{I} D\right) \cong \varprojlim_\mathbf{I} \mathscr{B}(F(A), D) \cong \varprojlim_\mathbf{I} \mathscr{A}(A, G \circ D) \cong \mathrm{Cone}(A, G \circ D)
$$

naturally in $A \in \mathscr{A}$. Here the first isomorphism is by adjointness, the second is because representables preserve limits, the third is by adjointness again, and the fourth is by Lemma 6.2.1. So $G(\varprojlim D)$ represents $\mathrm{Cone}(-, G \circ D)$; that is, it is a limit of $G \circ D$. $\square$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.3.2</span><span class="math-callout__name">(Forgetful functors from algebras)</span></p>

Forgetful functors from categories of algebras to $\mathbf{Set}$ have left adjoints, but hardly ever right adjoints. Correspondingly, they preserve all limits, but rarely all colimits.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.3.3</span><span class="math-callout__name">(Arithmetic of sets)</span></p>

Every set $B$ gives rise to an adjunction $(- \times B) \dashv (-)^B$ of functors from $\mathbf{Set}$ to $\mathbf{Set}$ (Example 2.1.6). So $- \times B$ preserves colimits and $(-)^B$ preserves limits. In particular, $- \times B$ preserves finite sums and $(-)^B$ preserves finite products, giving isomorphisms

$$
0 \times B \cong 0, \qquad (A_1 + A_2) \times B \cong (A_1 \times B) + (A_2 \times B),
$$

$$
1^B \cong 1, \qquad (A_1 \times A_2)^B \cong A_1^B \times A_2^B.
$$

These are the analogues of standard rules of arithmetic.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.3.4</span><span class="math-callout__name">(Limits commute with limits, again)</span></p>

Given a category $\mathscr{A}$ with all limits of shape $\mathbf{I}$, we have the adjunction $\Delta \dashv \varprojlim$ (Proposition 6.1.4). Hence $\varprojlim_\mathbf{I}$ preserves limits, or equivalently, limits of shape $\mathbf{I}$ commute with (all) limits. This gives another proof that limits commute with limits (Proposition 6.2.8).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.3.5</span><span class="math-callout__name">(Proving non-existence of adjoints)</span></p>

Theorem 6.3.1 is often used to prove that a functor does *not* have an adjoint. For instance, it was claimed that the forgetful functor $U \colon \mathbf{Field} \to \mathbf{Set}$ does not have a left adjoint. We can now prove this. If $U$ had a left adjoint $F \colon \mathbf{Set} \to \mathbf{Field}$, then $F$ would preserve colimits, and in particular, initial objects. Hence $F(\emptyset)$ would be an initial object of $\mathbf{Field}$. But $\mathbf{Field}$ has no initial object, since there are no maps between fields of different characteristic.

</div>

#### Adjoint Functor Theorems

Every functor with a left adjoint preserves limits, but limit-preservation alone does not guarantee the existence of a left adjoint. For example, the unique functor $\mathscr{B} \to \mathbf{1}$ always preserves limits, but by Example 2.1.9, it only has a left adjoint if $\mathscr{B}$ has an initial object.

On the other hand, if we have a limit-preserving functor $G \colon \mathscr{B} \to \mathscr{A}$ *and* $\mathscr{B}$ has all limits, then there is an excellent chance that $G$ has a left adjoint. The condition of having all limits is important enough to have its own word:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6.3.6</span><span class="math-callout__name">(Complete category)</span></p>

A category is **complete** (or properly, **small complete**) if it has all limits.

</div>

There are various results called adjoint functor theorems, all of the following form:

*Let $\mathscr{A}$ be a category, $\mathscr{B}$ a complete category, and $G \colon \mathscr{B} \to \mathscr{A}$ a functor. Suppose that $\mathscr{A}$, $\mathscr{B}$ and $G$ satisfy certain further conditions. Then*

$$
G \text{ has a left adjoint} \iff G \text{ preserves limits}.
$$

The forwards implication is immediate from Theorem 6.3.1. It is the backwards implication that concerns us here.

Typically, the "further conditions" involve the distinction between small and large collections. But there is a special case in which these complications disappear: when the categories $\mathscr{A}$ and $\mathscr{B}$ are ordered sets.

As we saw in Section 5.1, limits in ordered sets are meets. More precisely, if $D \colon \mathbf{I} \to \mathbf{B}$ is a diagram in an ordered set $\mathbf{B}$, then

$$
\varprojlim D = \bigwedge_{I \in \mathbf{I}} D(I),
$$

with one side defined if and only if the other is. So an ordered set is complete if and only if every subset has a meet. Similarly, a map $G \colon \mathbf{B} \to \mathbf{A}$ of ordered sets preserves limits if and only if

$$
G\!\Bigl(\bigwedge_{i \in I} B_i\Bigr) = \bigwedge_{i \in I} G(B_i)
$$

whenever $(B_i)_{i \in I}$ is a family of elements of $\mathbf{B}$ for which a meet exists.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition 6.3.7</span><span class="math-callout__name">(Adjoint functor theorem for ordered sets)</span></p>

*Let $\mathbf{A}$ be an ordered set, $\mathbf{B}$ a complete ordered set, and $G \colon \mathbf{B} \to \mathbf{A}$ an order-preserving map. Then*

$$
G \text{ has a left adjoint} \iff G \text{ preserves meets}.
$$

**Proof.** Suppose that $G$ preserves meets. By Corollary 2.3.7, it is enough to show that for each $A \in \mathbf{A}$, the comma category $(A \Rightarrow G)$ has an initial object. Let $A \in \mathbf{A}$. Then $(A \Rightarrow G)$ is an ordered set, namely $\lbrace B \in \mathbf{B} \mid A \le G(B) \rbrace$ with the order inherited from $\mathbf{B}$. We have to show that $(A \Rightarrow G)$ has a least element.

Since $\mathbf{B}$ is complete, the meet $\bigwedge_{B \in \mathbf{B},\, A \le G(B)} B$ exists in $\mathbf{B}$. This is the meet of all the elements of $(A \Rightarrow G)$, so it suffices to show that the meet is itself an element of $(A \Rightarrow G)$. And indeed, since $G$ preserves meets, we have

$$
G\!\Bigl(\bigwedge_{B \in \mathbf{B},\, A \le G(B)} B\Bigr) = \bigwedge_{B \in \mathbf{B},\, A \le G(B)} G(B) \ge A,
$$

as required. $\square$

</div>

In Proposition 6.3.7, the left adjoint $F$ is given by

$$
F(A) = \bigwedge_{B \in \mathbf{B},\, A \le G(B)} B.
$$

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.3.8</span><span class="math-callout__name">(Complete ordered sets have least elements)</span></p>

Consider Proposition 6.3.7 in the case $\mathbf{A} = \mathbf{1}$. The unique functor $G \colon \mathbf{B} \to \mathbf{1}$ automatically preserves meets, and a left adjoint to $G$ is an initial object of $\mathbf{B}$. So the proposition states that a complete ordered set has a least element. This is not quite trivial, since completeness means the existence of all meets, whereas a least element is an empty *join*.

By the formula above, the least element of $\mathbf{B}$ is $\bigwedge_{B \in \mathbf{B}} B$. Thus, a least element is not only a colimit of the functor $\emptyset \to \mathbf{B}$; it is also a limit of the identity functor $\mathbf{B} \to \mathbf{B}$.

The synonym "least upper bound" for "join" suggests a theorem: that a poset with all meets also has all joins. Indeed, given a poset $\mathbf{B}$ with all meets, the join of a subset of $\mathbf{B}$ is simply the meet of its upper bounds: quite literally, its least upper bound.

</div>

Let us now attempt to extend Proposition 6.3.7 from ordered sets to categories, starting with a limit-preserving functor $G$ from a complete category $\mathscr{B}$ to a category $\mathscr{A}$. In the case of ordered sets, we had for each $A \in \mathscr{A}$ an inclusion map $P_A \colon (A \Rightarrow G) \hookrightarrow \mathbf{B}$, and we showed that the left adjoint $F$ was given by

$$
F(A) = \varprojlim_{(A \Rightarrow G)} P_A.
$$

In the general case, the analogue of the inclusion functor is the projection functor

$$
P_A \colon (A \Rightarrow G) \to \mathscr{B}, \qquad \bigl(B,\, A \xrightarrow{f} G(B)\bigr) \mapsto B.
$$

The case of ordered sets suggests that the limit $\varprojlim_{(A \Rightarrow G)} P_A$ might define a left adjoint $F$ to $G$. And indeed, it can be shown that if this limit in $\mathscr{B}$ exists and is preserved by $G$, then it really does give a left adjoint (Theorem X.1.2 of Mac Lane (1971)).

However, if $\mathscr{B}$ is a large category then $(A \Rightarrow G)$ might also be large, so the limit defining the left adjoint is not guaranteed to be small. Hence there is no guarantee that this limit exists in $\mathscr{B}$, nor that it is preserved by $G$. The situation therefore becomes more complicated. Each of the best-known adjoint functor theorems imposes further conditions implying that the large limit can be replaced by a small limit in some clever way.

The two most famous adjoint functor theorems are the "general" and the "special".

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6.3.9</span><span class="math-callout__name">(Weakly initial set)</span></p>

Let $\mathscr{C}$ be a category. A **weakly initial set** in $\mathscr{C}$ is a set $\mathbf{S}$ of objects with the property that for each $C \in \mathscr{C}$, there exist an element $S \in \mathbf{S}$ and a map $S \to C$.

Note that $\mathbf{S}$ must be a set, that is, small. So the existence of a weakly initial set is some kind of size restriction, comparable to finiteness conditions in algebra.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.3.10</span><span class="math-callout__name">(General adjoint functor theorem)</span></p>

*Let $\mathscr{A}$ be a category, $\mathscr{B}$ a complete category, and $G \colon \mathscr{B} \to \mathscr{A}$ a functor. Suppose that $\mathscr{B}$ is locally small and that for each $A \in \mathscr{A}$, the category $(A \Rightarrow G)$ has a weakly initial set. Then*

$$
G \text{ has a left adjoint} \iff G \text{ preserves limits}.
$$

*Proof.* See the appendix. $\square$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.3.11</span><span class="math-callout__name">(GAFT for forgetful functors to Set)</span></p>

The general adjoint functor theorem (GAFT) implies that for any category $\mathscr{B}$ of algebras ($\mathbf{Grp}$, $\mathbf{Vect}_k$, $\ldots$), the forgetful functor $U \colon \mathscr{B} \to \mathbf{Set}$ has a left adjoint. Indeed, $\mathscr{B}$ has all limits, $U$ preserves them, and $\mathscr{B}$ is locally small. To apply GAFT, we just have to check that for each $A \in \mathbf{Set}$, the comma category $(A \Rightarrow U)$ has a weakly initial set. This requires a little cardinal arithmetic.

So GAFT tells us that, for instance, the free group functor exists. GAFT avoids the trickiness of explicitly constructing the free group on a generating set $A$. The price to be paid is that GAFT does not give us an explicit description of free groups (or left adjoints more generally).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.3.12</span><span class="math-callout__name">(GAFT for forgetful functors between algebras)</span></p>

More generally, GAFT guarantees that forgetful functors between categories of algebras, such as

$$
\mathbf{Ab} \to \mathbf{Grp}, \quad \mathbf{Grp} \to \mathbf{Mon}, \quad \mathbf{Ring} \to \mathbf{Mon}, \quad \mathbf{Vect}_{\mathbb{C}} \to \mathbf{Vect}_{\mathbb{R}},
$$

have left adjoints. This is "more generally" because $\mathbf{Set}$ can be seen as a degenerate example of a category of algebras: a set is a set equipped with no operations satisfying no equations.

</div>

The special adjoint functor theorem (SAFT) operates under much tighter hypotheses than GAFT, and is much less widely applicable. Its main advantage is that it removes the condition on weakly initial sets. Indeed, it removes *all* further conditions on the functor $G$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.3.13</span><span class="math-callout__name">(Special adjoint functor theorem)</span></p>

*Let $\mathscr{A}$ be a category, $\mathscr{B}$ a complete category, and $G \colon \mathscr{B} \to \mathscr{A}$ a functor. Suppose that $\mathscr{A}$ and $\mathscr{B}$ are locally small, and that $\mathscr{B}$ satisfies certain further conditions. Then*

$$
G \text{ has a left adjoint} \iff G \text{ preserves limits}.
$$

A precise statement and proof can be found in Section V.8 of Mac Lane (1971).

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.3.14</span><span class="math-callout__name">(Stone--Cech compactification)</span></p>

Here is the classic application of SAFT. Let $\mathbf{CptHff}$ be the category of compact Hausdorff spaces, and $U \colon \mathbf{CptHff} \to \mathbf{Top}$ the forgetful functor. SAFT tells us that $U$ has a left adjoint $F$, turning any space into a compact Hausdorff space in a canonical way.

Given a space $X$, the resulting compact Hausdorff space $F(X)$ is called its **Stone--Cech compactification**. Provided that $X$ satisfies some mild separation conditions, the unit of the adjunction at $X$ is an embedding, so that $UF(X)$ contains $X$ as a subspace.

Another advantage of SAFT is that one can extract from its proof a fairly explicit formula for the left adjoint. In this case, $F(X)$ is the closure of the image of the canonical map

$$
X \to [0, 1]^{\mathbf{Top}(X, [0, 1])},
$$

where the codomain is a power of $[0, 1]$ in $\mathbf{Top}$.

</div>

#### Cartesian Closed Categories

We have seen that for every set $B$, there is an adjunction $(- \times B) \dashv (-)^B$ (Example 2.1.6), and that for every category $\mathscr{B}$, there is an adjunction $(- \times \mathscr{B}) \dashv [\mathscr{B}, -]$ (Remark 4.1.23(c)).

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition 6.3.15</span><span class="math-callout__name">(Cartesian closed category)</span></p>

A category $\mathscr{A}$ is **cartesian closed** if it has finite products and for each $B \in \mathscr{A}$, the functor $- \times B \colon \mathscr{A} \to \mathscr{A}$ has a right adjoint.

We write the right adjoint as $(-)^B$, and for $C \in \mathscr{A}$, call $C^B$ an **exponential**. We may think of $C^B$ as the space of maps from $B$ to $C$. Adjointness says that for all $A, B, C \in \mathscr{A}$,

$$
\mathscr{A}(A \times B, C) \cong \mathscr{A}(A, C^B)
$$

naturally in $A$ and $C$. In fact, the isomorphism is natural in $B$ too; that comes for free.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.3.16</span><span class="math-callout__name">(Set is cartesian closed)</span></p>

$\mathbf{Set}$ is cartesian closed; $C^B$ is the function set $\mathbf{Set}(B, C)$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.3.17</span><span class="math-callout__name">(CAT is cartesian closed)</span></p>

$\mathbf{CAT}$ is cartesian closed; $\mathscr{C}^{\mathscr{B}}$ is the functor category $[\mathscr{B}, \mathscr{C}]$.

</div>

In any cartesian closed category with finite sums, the isomorphisms of Example 6.3.3 hold, for the same reasons as stated there. The objects of a cartesian closed category therefore possess an arithmetic like that of the natural numbers. These isomorphisms also provide a way of proving that a category is *not* cartesian closed.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example 6.3.18</span><span class="math-callout__name">(Vect is not cartesian closed)</span></p>

$\mathbf{Vect}_k$ is not cartesian closed, for any field $k$. It does have finite products, as we saw in Example 5.1.5: binary product is direct sum $\oplus$, and the terminal object is the trivial vector space $\lbrace 0 \rbrace$, which is also initial. But if $\mathbf{Vect}_k$ were cartesian closed then the arithmetic equations would hold, so that $\lbrace 0 \rbrace \oplus B \cong \lbrace 0 \rbrace$ for all vector spaces $B$. This is plainly false.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark 6.3.19</span><span class="math-callout__name">(Monoidal closed categories)</span></p>

For any vector spaces $V$ and $W$, the set $\mathbf{Vect}_k(V, W)$ of linear maps can itself be given the structure of a vector space, as in Example 1.2.12. Let us call this vector space $[V, W]$.

Given that exponentials are supposed to be "spaces of maps", you might expect $\mathbf{Vect}_k$ to be cartesian closed, with $[-, -]$ as its exponential. We have just seen that this cannot be so. But as it turns out, the linear maps $U \to [V, W]$ correspond to the *bilinear* maps $U \times V \to W$, or equivalently the linear maps $U \otimes V \to W$. In the jargon, $\mathbf{Vect}_k$ is an example of a "monoidal closed category": like cartesian closed categories, but with the cartesian (categorical) product replaced by the tensor product of vector spaces.

</div>

For any set $I$, the product category $\mathbf{Set}^I$ is cartesian closed, just because $\mathbf{Set}$ is. Put another way, $[\mathbf{A}^{\mathrm{op}}, \mathbf{Set}]$ is cartesian closed whenever $\mathbf{A}$ is discrete. We now show that, in fact, $[\mathbf{A}^{\mathrm{op}}, \mathbf{Set}]$ is cartesian closed for any small category $\mathbf{A}$ whatsoever.

Write $\hat{\mathbf{A}} = [\mathbf{A}^{\mathrm{op}}, \mathbf{Set}]$. If $\hat{\mathbf{A}}$ *is* cartesian closed, what must exponentials in $\hat{\mathbf{A}}$ be? In other words, given presheaves $Y$ and $Z$, what must $Z^Y$ be in order that

$$
\hat{\mathbf{A}}(X, Z^Y) \cong \hat{\mathbf{A}}(X \times Y, Z)
$$

for all presheaves $X$? If this is true for all presheaves $X$, then in particular it is true when $X$ is representable, so

$$
Z^Y(A) \cong \hat{\mathbf{A}}(H_A, Z^Y) \cong \hat{\mathbf{A}}(H_A \times Y, Z)
$$

for all $A \in \mathbf{A}$, the first step by Yoneda. This tells us what $Z^Y$ must be. Notice that $Z^Y(A)$ is not simply $Z(A)^{Y(A)}$: exponentials in a presheaf category are *not* generally computed pointwise.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem 6.3.20</span><span class="math-callout__name">(Presheaf categories are cartesian closed)</span></p>

*For any small category $\mathbf{A}$, the presheaf category $\hat{\mathbf{A}}$ is cartesian closed.*

**Proof.** We know that $\hat{\mathbf{A}}$ has all limits, and in particular, finite products. It remains to show that $\hat{\mathbf{A}}$ has exponentials. Fix $Y \in \hat{\mathbf{A}}$.

First we prove that $- \times Y \colon \hat{\mathbf{A}} \to \hat{\mathbf{A}}$ preserves colimits. (Eventually we will prove that $- \times Y$ has a right adjoint, from which preservation of colimits follows, but our proof that it has a right adjoint will *use* preservation of colimits.) Indeed, since products and colimits in $\hat{\mathbf{A}}$ are computed pointwise, it is enough to prove that for any set $S$, the functor $- \times S \colon \mathbf{Set} \to \mathbf{Set}$ preserves colimits, and this follows from the fact that $\mathbf{Set}$ is cartesian closed.

For each presheaf $Z$ on $\mathbf{A}$, let $Z^Y$ be the presheaf defined by

$$
Z^Y(A) = \hat{\mathbf{A}}(H_A \times Y, Z)
$$

for all $A \in \mathbf{A}$. This defines a functor $(-)^Y \colon \hat{\mathbf{A}} \to \hat{\mathbf{A}}$.

We claim that $(- \times Y) \dashv (-)^Y$. Let $X, Z \in \hat{\mathbf{A}}$. Write $P \colon \mathbb{E}(X) \to \mathbf{A}$ for the projection and $H_P = H_\bullet \circ P$. Then

$$
\hat{\mathbf{A}}(X, Z^Y) \cong \hat{\mathbf{A}}\!\Bigl(\varinjlim_{\mathbb{E}(X)} H_P,\, Z^Y\Bigr) \cong \varprojlim_{\mathbb{E}(X)^{\mathrm{op}}} \hat{\mathbf{A}}(H_P, Z^Y) \cong \varprojlim_{\mathbb{E}(X)^{\mathrm{op}}} Z^Y(P)
$$

$$
\cong \varprojlim_{\mathbb{E}(X)^{\mathrm{op}}} \hat{\mathbf{A}}(H_P \times Y, Z) \cong \hat{\mathbf{A}}\!\Bigl(\varinjlim_{\mathbb{E}(X)} (H_P \times Y),\, Z\Bigr) \cong \hat{\mathbf{A}}\!\Bigl(\Bigl(\varinjlim_{\mathbb{E}(X)} H_P\Bigr) \times Y,\, Z\Bigr) \cong \hat{\mathbf{A}}(X \times Y, Z)
$$

naturally in $X$ and $Z$. The key steps use: Theorem 6.2.17 (expressing $X$ as a colimit of representables), the fact that representables preserve limits, Yoneda, the definition of $Z^Y$, and the fact that $- \times Y$ preserves colimits. $\square$

</div>

This result can be seen as a step along the road to topos theory. A **topos** is a category with certain special properties. Topos theory unifies, in an extraordinary way, important aspects of logic and geometry.

For instance, a topos can be regarded as a "universe of sets": $\mathbf{Set}$ is the most basic example of a topos, and every topos shares enough features with $\mathbf{Set}$ that one can reason with its objects as if they were sets of some exotic kind. On the other hand, a topos can be regarded as a generalized topological space: every space gives rise to a topos (namely, the category of sheaves on it), and topological properties of the space can be reinterpreted in a useful way as categorical properties of its associated topos.

By definition, a topos is a cartesian closed category with finite limits and with one further property: the existence of a so-called **subobject classifier**. For example, the two-element set $2$ is the subobject classifier of $\mathbf{Set}$, which means, informally, that subsets of a set $A$ correspond one-to-one with maps $A \to 2$.

### Appendix: Proof of the General Adjoint Functor Theorem

Here we prove the general adjoint functor theorem. The left-to-right implication follows immediately from Theorem 6.3.1; it is the right-to-left implication that we have to prove.

The heart of the proof is the case $\mathscr{A} = \mathbf{1}$, where GAFT asserts that a complete locally small category with a weakly initial set has an initial object. We prove this first.

The proof of this special case is illuminated by considering the even more special case where $\mathscr{A} = \mathbf{1}$ and the category $\mathscr{B}$ is a poset $\mathbf{B}$. We saw in Example 6.3.8 that the initial object (least element) of a complete poset $\mathbf{B}$ can be constructed as the meet of all its elements. Otherwise put, it is the limit of the identity functor $1_\mathbf{B} \colon \mathbf{B} \to \mathbf{B}$.

One might try to extend this to arbitrary categories $\mathscr{B}$ by proving that the limit of the identity functor $1_\mathscr{B} \colon \mathscr{B} \to \mathscr{B}$ is (if it exists) an initial object. This is indeed true. However, it is unhelpful: for if $\mathscr{B}$ is large then the limit of $1_\mathscr{B}$ is a large limit, but we are only given that $\mathscr{B}$ has small limits.

The clever idea behind GAFT is that to construct the least element of a complete poset, it is not necessary to take the meet of *all* the elements. More economically, we could just take the meet of the elements of some weakly initial subset.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma A.1</span><span class="math-callout__name">(Weakly initial sets give initial objects)</span></p>

*Let $\mathscr{C}$ be a complete locally small category with a weakly initial set. Then $\mathscr{C}$ has an initial object.*

**Proof.** Let $\mathbf{S}$ be a weakly initial set in $\mathscr{C}$. Regard $\mathbf{S}$ as a full subcategory of $\mathscr{C}$; then $\mathbf{S}$ is small, since $\mathscr{C}$ is locally small. We may therefore take a limit cone

$$
\bigl(0 \xrightarrow{p_S} S\bigr)_{S \in \mathbf{S}}
$$

of the inclusion $\mathbf{S} \hookrightarrow \mathscr{C}$. We prove that $0$ is initial.

Let $C \in \mathscr{C}$. We have to show that there is exactly one map $0 \to C$. Certainly there is at least one, since we may choose some $S \in \mathbf{S}$ and map $j \colon S \to C$, and we then have the composite $jp_S \colon 0 \to C$. To prove uniqueness, let $f, g \colon 0 \to C$. Form the equalizer

$$
E \xrightarrow{i} 0 \rightrightarrows C.
$$

Since $\mathbf{S}$ is weakly initial, we may choose $S \in \mathbf{S}$ and $h \colon S \to E$. We then have maps

$$
0 \xrightarrow{p_S} S \xrightarrow{h} E \xrightarrow{i} 0
$$

with the property that for all $S' \in \mathbf{S}$, $p_{S'} \cdot (ihp_S) = (p_{S'} \cdot ih) p_S = p_{S'} = p_{S'} \cdot 1_0$. But the cone $(p_S)$ is a limit cone, so $ihp_S = 1_0$. Hence

$$
f = fihp_S = gihp_S = g,
$$

as required. $\square$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma A.2</span><span class="math-callout__name">(Projection functor creates limits)</span></p>

*Let $\mathscr{A}$ and $\mathscr{B}$ be categories. Let $G \colon \mathscr{B} \to \mathscr{A}$ be a functor that preserves limits. Then the projection functor $P_A \colon (A \Rightarrow G) \to \mathscr{B}$ creates limits, for each $A \in \mathscr{A}$. In particular, if $\mathscr{B}$ is complete then so is each comma category $(A \Rightarrow G)$.*

**Proof.** The first statement is Exercise A.5(b), and the second follows from Lemma 5.3.6. $\square$

</div>

We now prove GAFT. By Corollary 2.3.7, it is enough to show that $(A \Rightarrow G)$ has an initial object for each $A \in \mathscr{A}$. Let $A \in \mathscr{A}$. By Lemma A.2, $(A \Rightarrow G)$ is complete, and by hypothesis, it has a weakly initial set. It is also locally small, since $\mathscr{B}$ is. Hence by Lemma A.1, it has an initial object, as required. $\square$

