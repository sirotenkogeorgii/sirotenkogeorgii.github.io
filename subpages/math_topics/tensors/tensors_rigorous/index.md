---
layout: post
title: "What is a tensor, really? One idea behind a confusing word"
date: 2026-06-21
tags: [linear-algebra, multilinear-algebra, tensor-products, category-theory, functional-analysis]
mathjax: true
---

> *Conventions.* Throughout, $\mathbb{F}$ is a field — concretely $\mathbb{R}$ or $\mathbb{C}$ — and "vector space" means a vector space over $\mathbb{F}$. Unless I say otherwise everything is finite-dimensional, so I can be cavalier about duals and completions; I will flag carefully the places where infinite dimensions force a genuine change. I write $V^\ast = \operatorname{Hom}(V,\mathbb{F})$ for the dual, $\{e_i\}$ for a basis of $V$ with dual basis $\{e^i\}$ characterised by $e^i(e_j)=\delta^i_j$, and I use the Einstein summation convention (a repeated upper/lower index pair is summed) wherever it lightens notation. The symbol $\otimes$ is the tensor product, defined below; $\hat\otimes$ is its completion in the topological setting.

## The problem: too many tensors

If you learn the word *tensor* from three different sources you will come away with three apparently unrelated definitions, and a vague suspicion that everyone is bluffing.

- A **machine-learning** course tells you a tensor is a multidimensional array: a vector is a $1$-tensor, a matrix is a $2$-tensor, and a stack of $k$ indices $T_{i_1\cdots i_k}$ is a $k$-tensor. PyTorch agrees.
- A **physics** course tells you a tensor is *an object that transforms like a tensor* — a quantity with upper and lower indices obeying a specific change-of-coordinates law, the law being the actual content of the definition.
- An **algebra** course tells you a tensor is an element of an abstract space $V\otimes W$ built by a quotient construction, characterised by a universal property and otherwise resolutely coordinate-free.

These look like three different subjects. They are not. **There is exactly one idea here, and the three definitions are the same object viewed in three lights: the abstract space, its coordinates, and the rule its coordinates must obey to be basis-independent.** The purpose of this post is to make that single idea visible and then watch the zoo of "tensors" fall out of it as special cases.

Let me state the idea before doing any work, because everything else is commentary on it.

> **The driving idea.** The tensor product is the machine that converts *multilinearity* into *linearity*. Bilinear maps are awkward because their domain is a *product* $V\times W$, not a vector space we can do linear algebra on. The tensor product manufactures a single new space $V\otimes W$ together with a universal bilinear map $V\times W \to V\otimes W$, in such a way that every bilinear map out of $V\times W$ becomes an ordinary *linear* map out of $V\otimes W$. Everything called a "tensor" is an element of such a space, and the various definitions differ only in whether you describe that element abstractly, by its coordinates, or by how its coordinates respond to a change of basis.

## Why bilinear maps want a home

Start with the irritation that motivates the whole construction. A map $B\colon V\times W \to U$ is **bilinear** if it is linear in each slot separately:
$$
B(\alpha v + \alpha' v', w) = \alpha B(v,w) + \alpha' B(v',w), \qquad B(v, \beta w + \beta' w') = \beta B(v,w)+\beta' B(v,w').
$$
The determinant in its rows, the dot product, matrix multiplication, the evaluation pairing $V^\ast\times V\to\mathbb{F}$, the multiplication map of an algebra — all bilinear. These are everywhere, and yet $B$ is *not* a linear map: its domain $V\times W$ is a vector space (the direct sum $V\oplus W$, really), but $B$ is not linear on it, since $B(v+v', w+w') \ne B(v,w)+B(v',w')$ in general. Bilinear is genuinely a different and clumsier category of object than linear.

The wish, then, is concrete: **find a single vector space through which all bilinear maps factor linearly.** If we had a space $T$ and a fixed bilinear map $\otimes\colon V\times W\to T$, $(v,w)\mapsto v\otimes w$, so universal that every bilinear $B\colon V\times W\to U$ equalled $\tilde B\circ\otimes$ for a *unique linear* $\tilde B\colon T\to U$, then the entire study of bilinear maps $V\times W\to U$ would collapse into the study of linear maps $T\to U$. We could stop thinking about bilinearity altogether and just do linear algebra on $T$. That $T$ is the tensor product $V\otimes W$.

## Building the space, then forgetting how

There is a construction, and there is a universal property. The construction reassures you the object exists; the universal property is what you actually use. After this section I will never open up the construction again, and neither should you.

**Step 1 — make everything formally independent.** Let $F$ be the free vector space on the *set* $V\times W$: formal finite linear combinations $\sum_k c_k\,(v_k,w_k)$ with the pairs $(v,w)$ as a basis. This space is gigantic and stupid — it knows nothing about the linear structures of $V$ and $W$; the pairs $(2v,w)$ and $(v,2w)$ and $2(v,w)$ are three unrelated basis vectors.

**Step 2 — impose exactly the bilinearity relations.** Let $R\subseteq F$ be the subspace spanned by all elements of the form
$$
(\alpha v+\alpha'v',\,w) - \alpha(v,w) - \alpha'(v',w), \qquad
(v,\,\beta w+\beta'w') - \beta(v,w) - \beta'(v,w'),
$$
and define
$$
V\otimes W := F/R.
$$
Write $v\otimes w$ for the image of the pair $(v,w)$. By construction $(v,w)\mapsto v\otimes w$ is bilinear — we quotiented by precisely the failures of bilinearity and by nothing else.

**Step 3 — read off the universal property.** Given any bilinear $B\colon V\times W\to U$, the universal property of the *free* space extends $B$ to a linear map $F\to U$; bilinearity of $B$ says exactly that this map kills $R$; so it descends to a unique linear $\tilde B\colon V\otimes W\to U$ with $\tilde B(v\otimes w)=B(v,w)$. That is the whole point, so let me box it.

> **Universal property of the tensor product.** There is a bilinear map $\otimes\colon V\times W\to V\otimes W$ such that for every vector space $U$ and every bilinear map $B\colon V\times W\to U$ there exists a *unique* linear map $\tilde B\colon V\otimes W\to U$ with $B = \tilde B\circ\otimes$. Equivalently, the natural map
> $$
> \operatorname{Hom}(V\otimes W,\;U)\;\xrightarrow{\;\sim\;}\;\operatorname{Bil}(V\times W,\;U), \qquad \tilde B\mapsto \tilde B\circ\otimes,
> $$
> is an isomorphism.

**The construction is now disposable.** Any two spaces satisfying this property are canonically isomorphic — feed each universal map into the other's universal property and the two comparison maps are forced to be mutually inverse. So "the" tensor product is well-defined up to unique isomorphism, and we never again care that it was built as a quotient of an absurd free space. *This is the recurring moral of universal-property definitions: the construction proves existence and then gets out of the way.*

**Coordinates and dimension.** Take bases $\{e_i\}_{i=1}^m$ of $V$ and $\{f_j\}_{j=1}^n$ of $W$. Then $\{e_i\otimes f_j\}_{i,j}$ is a basis of $V\otimes W$, so
$$
\dim(V\otimes W) = (\dim V)(\dim W).
$$
*Why a basis:* spanning is bilinearity ($v\otimes w = v^i w^j\, e_i\otimes f_j$ after expanding both arguments); independence follows because the coordinate functionals $e^i\otimes f^j$, defined on simple tensors by $(e^i\otimes f^j)(v\otimes w)=e^i(v)f^j(w)$ and extended via the universal property, separate the $e_i\otimes f_j$. Note the contrast with the *direct sum*, where $\dim(V\oplus W)=\dim V+\dim W$: the tensor product **multiplies** dimensions while the direct sum adds them. This is the first hint that $\otimes$ deserves to be read as a product.

## Simple tensors are the exception, not the rule

Elements of the form $v\otimes w$ are called **simple** (or **pure**, or **decomposable**) tensors. They span $V\otimes W$, but they are a thin, curved subset of it — almost no tensor is simple.

To see this, fix bases and identify $V\otimes W$ with $m\times n$ matrices via $e_i\otimes f_j \mapsto E_{ij}$ (the matrix with a $1$ in slot $(i,j)$). Under this identification the simple tensor $v\otimes w$ becomes the **outer product**
$$
v\otimes w \;\longleftrightarrow\; v\,w^{\!\top} = \big(v^i w^j\big)_{ij},
$$
a rank-one matrix. A general tensor is a finite sum $\sum_k v_k\otimes w_k$, corresponding to a general matrix. So:

> **Tensor rank.** The rank of $T\in V\otimes W$ is the least $r$ such that $T=\sum_{k=1}^r v_k\otimes w_k$. Under the identification above this is exactly the rank of the corresponding matrix. Simple tensors are precisely the rank-$\le 1$ tensors, and they form the (Segre) cone of rank-one matrices — a measure-zero, non-linear subset for $m,n\ge 2$.

This already dissolves one beginner's confusion: "is a tensor just $v\otimes w$?" No. The simple tensors are the *generators*; the space is their linear span, and a generic element needs several of them. Be warned that for $\ge 3$ factors tensor rank is genuinely subtle — it is not lower-semicontinuous, the rank can jump down in limits ("border rank"), and deciding it is NP-hard. The clean matrix picture is special to two factors.

## Reconciliation I: the multidimensional array

Now the machine-learning definition. Once bases are fixed, a tensor $T\in V\otimes W$ is determined by its coordinates $T^{ij}=(e^i\otimes f^j)(T)$, and we can write the array $(T^{ij})$. More factors give more indices: an element of $V_1\otimes\cdots\otimes V_k$ has coordinates $T^{i_1\cdots i_k}$, a $k$-dimensional grid of numbers. That *is* the array picture, and it is correct — as a description **relative to a chosen basis**.

The thing the array picture leaves out is the only thing that matters conceptually: **the array depends on the basis, the tensor does not.** A list of numbers $T^{ij}$ is not yet a tensor any more than the column $(1,0,0)^\top$ "is" a vector — it is the *coordinate representation* of a vector, and a different basis gives a different column for the same vector. The tensor is the basis-independent element; the array is its shadow. To recover the tensor from arrays you must also carry the rule for how the shadow changes under a change of basis — which is exactly the physicist's definition, to which we now turn.

## Duality: tensors as maps

Before the transformation law, one more structural fact unlocks half the examples. In finite dimensions there is a canonical isomorphism
$$
V^\ast\otimes W \;\xrightarrow{\;\sim\;}\; \operatorname{Hom}(V,W), \qquad \varphi\otimes w \;\longmapsto\; \big(v\mapsto \varphi(v)\,w\big),
$$
extended linearly. (It is well-defined by the universal property, since $(\varphi,w)\mapsto(v\mapsto\varphi(v)w)$ is bilinear; it is an isomorphism because both sides have dimension $\dim V\cdot\dim W$ and it is injective on a basis.) Specialising:

- $V^\ast\otimes V^\ast \cong \operatorname{Bil}(V\times V,\mathbb{F})$: a $\binom{0}{2}$-tensor *is* a bilinear form. An inner product, or a metric $g_{ij}$, lives here.
- $V^\ast\otimes V \cong \operatorname{Hom}(V,V) = \operatorname{End}(V)$: a $\binom{1}{1}$-tensor *is* a linear operator. The identity operator corresponds to the canonical element $\sum_i e_i\otimes e^i$, basis-independently.
- $(V\otimes W)^\ast \cong V^\ast\otimes W^\ast$ canonically, which is why "the dual of a tensor product is the tensor product of duals."

So the slogan **"a tensor is a multilinear map"** is just the universal property read backwards: a tensor in $(V^\ast)^{\otimes p}\otimes V^{\otimes q}$ *is* a multilinear map eating $p$ vectors and $q$ covectors and returning a scalar. The array and the multilinear-map are the same object; one is the map's matrix of values on basis arguments.

> *Infinite-dimensional caveat.* The map $V^\ast\otimes W\to\operatorname{Hom}(V,W)$ lands only in the **finite-rank** operators when $V$ is infinite-dimensional; you reach all bounded operators (or Hilbert–Schmidt, or trace-class) only after a topological completion. This is the first place the finite-dimensional fairy tale breaks, and I return to it at the end.

## Mixed tensors, contraction, and the $(p,q)$ zoo

Iterating $\otimes$ (it is associative up to canonical isomorphism, and $\mathbb{F}\otimes V\cong V$ acts as a unit) gives the **tensor algebra** built from a single space $V$:
$$
T^p_q(V) \;:=\; \underbrace{V\otimes\cdots\otimes V}_{p}\;\otimes\;\underbrace{V^\ast\otimes\cdots\otimes V^\ast}_{q},
$$
the space of **type $(p,q)$ tensors**: $p$ *contravariant* slots and $q$ *covariant* slots. Coordinates carry $p$ upper and $q$ lower indices, $T^{i_1\cdots i_p}{}_{j_1\cdots j_q}$. This is the central object of differential geometry and physics, and it specialises to everything above: $(1,0)$ vectors, $(0,1)$ covectors, $(0,2)$ bilinear forms, $(1,1)$ operators.

The operation that makes this algebra come alive is **contraction**: pair one upper index against one lower index and sum, e.g.
$$
T^{i}{}_{j} \;\longmapsto\; \sum_i T^{i}{}_{i}.
$$
Abstractly this is induced by the evaluation pairing $V\otimes V^\ast\to\mathbb{F}$, $v\otimes\varphi\mapsto\varphi(v)$, applied in the chosen pair of slots. The trace of an operator is the contraction of its $(1,1)$ tensor; the divergence, the metric trace, and tensor "dot products" are all contractions. Contraction lowers type by $(1,1)$ and — crucially — **its output is basis-independent**, which we verify in the next section as the cleanest possible illustration of the transformation law.

## Reconciliation II: "transforms like a tensor"

Here is the physicist's definition, and here is why it is not a definition by ritual but a precise way of pinning down a basis-independent object using only its coordinates.

**Set up the change of basis.** Let $\tilde e_j = A^i{}_j\, e_i$ be a new basis ($A\in GL(V)$ invertible). The dual basis transforms by the inverse, $\tilde e^i = (A^{-1})^i{}_j\, e^j$. Now ask how the *coordinates* of a fixed object change.

**Contravariant slots pick up $A^{-1}$.** A vector $v = v^i e_i = \tilde v^j \tilde e_j$ is fixed; substituting $\tilde e_j = A^i{}_j e_i$ and matching coefficients gives
$$
\tilde v^j = (A^{-1})^j{}_i\, v^i.
$$
The components transform *oppositely* to the basis — hence "contra-variant", upper index.

**Covariant slots pick up $A$.** Dually, a covector $\omega=\omega_i e^i$ fixed gives
$$
\tilde\omega_j = A^i{}_j\,\omega_i,
$$
the components transforming *with* the basis — "co-variant", lower index.

**A general $(p,q)$ tensor multiplies these rules slot by slot.** Because the coordinates of $T\in T^p_q(V)$ are obtained by feeding basis vectors and covectors into the corresponding multilinear map, each upper index contributes a factor $A^{-1}$ and each lower index a factor $A$:
$$
\boxed{\;\tilde T^{i_1\cdots i_p}{}_{j_1\cdots j_q}
= (A^{-1})^{i_1}{}_{k_1}\cdots (A^{-1})^{i_p}{}_{k_p}\;
A^{l_1}{}_{j_1}\cdots A^{l_q}{}_{j_q}\;
T^{k_1\cdots k_p}{}_{l_1\cdots l_q}.\;}
$$

> **What the law really says.** "$X$ transforms like a $(p,q)$ tensor" $\iff$ "the arrays $X$ in each basis are the coordinate representations of a single basis-independent element of $T^p_q(V)$." The transformation law is not an extra axiom bolted onto an array; it is the *compatibility condition* that certifies an array as the shadow of a genuine geometric object. Conversely, if a quantity is built coordinate-by-coordinate but violates this law (the Christoffel symbols $\Gamma^k_{ij}$ are the standard cautionary example — they have indices but pick up an inhomogeneous correction term), then it is *not* a tensor, and that failure is meaningful.

**Contraction, rechecked.** With the law in hand the basis-independence of contraction is one line: in a $(1,1)$ contraction the $A^{-1}$ from the upper index meets the $A$ from the lower index and they annihilate,
$$
\tilde T^{i}{}_{i} = (A^{-1})^i{}_k A^l{}_i\, T^k{}_l = \delta^l_k\, T^k{}_l = T^k{}_k,
$$
so the trace is a genuine scalar. Lowering an upper index against a covariant one is *the* reason indices come in two flavours: it is precisely the matched pairs that contract to invariants. This is the algebra behind the slogan that *physical laws should be tensor equations* — an equation $S=T$ between tensors of the same type holds in one basis iff it holds in all, so writing your law tensorially guarantees it is coordinate-independent, which is exactly what a law of nature ought to be.

## Why the law is automatic: $\otimes$ is a functor

It is worth seeing that none of the transformation law is an accident; it is forced by a structural fact. The tensor product is a **functor**: a pair of linear maps $S\colon V\to V'$, $R\colon W\to W'$ induces a linear map
$$
S\otimes R\colon V\otimes W\to V'\otimes W', \qquad (S\otimes R)(v\otimes w) = Sv\otimes Rw,
$$
respecting composition and identities. A change of basis is just an invertible $A\colon V\to V$, and its induced action on $V^{\otimes p}\otimes(V^\ast)^{\otimes q}$ is $A^{\otimes p}\otimes (A^{-\top})^{\otimes q}$ — the inverse-transpose appearing on the dual factors because dualising is a *contravariant* functor, $(A^{-1})^\top$ being the action on $V^\ast$. Spell that action out in coordinates and you recover the boxed law verbatim. So the upper/lower index bookkeeping is nothing but the functoriality of $\otimes$ together with the contravariance of $(-)^\ast$. Physicists discovered the law empirically; it is a theorem of category theory.

## Where the word was born: tensor fields

The name comes from elasticity — Voigt's *Tensor* measured tension, a direction-dependent stress — and the modern home of the word is differential geometry. On a smooth manifold $M$ each point $p$ carries a tangent space $T_pM$ and its dual $T_p^\ast M$; a **tensor field** of type $(p,q)$ is a smooth assignment
$$
p \;\longmapsto\; T(p)\in \big(T_pM\big)^{\otimes p}\otimes\big(T_p^\ast M\big)^{\otimes q}.
$$
In a coordinate chart you get exactly the indexed arrays $T^{i_1\cdots i_p}{}_{j_1\cdots j_q}(x)$ obeying the transformation law with $A = $ the Jacobian of the change of chart. The metric tensor $g_{ij}$ (a $(0,2)$ field), the Riemann curvature $R^i{}_{jkl}$, the stress–energy tensor — these are the load-bearing examples, and the entire apparatus of general relativity is the assertion that the laws of physics are equalities of tensor fields, hence independent of the observer's coordinates. This is the setting that fixes the cultural meaning of "tensor" for most scientists, and it is, again, just $T^p_q$ of a vector space, varying smoothly from point to point.

## The dictionary

Collecting the reconciliations, here is the same object under its various names.

| Community | "A tensor is…" | What it really is |
|---|---|---|
| ML / numerics | a $k$-dimensional array $T_{i_1\cdots i_k}$ | coordinates of an element of $V_1\otimes\cdots\otimes V_k$ in fixed bases |
| Physics / geometry | an indexed object obeying the $(p,q)$ transformation law | a basis-independent element of $T^p_q(V)$, certified by the law |
| Algebra | an element of $V\otimes W$ (universal property) | the thing itself, coordinate-free |
| Multilinear algebra | a multilinear map | $(V^\ast)^{\otimes p}\otimes V^{\otimes q}$ via the duality $V^\ast\otimes W\cong\operatorname{Hom}(V,W)$ |
| Differential geometry | a tensor field | a smooth section $p\mapsto T(p)\in T^p_q(T_pM)$ |

The reason there are "so many tensors" is sociological, not mathematical: each community fixes the part of the structure it cares about (the numbers, the invariance, the abstraction, the maps) and names *that* the tensor. The full object carries all four aspects at once.

## The universal property travels: beyond vector spaces

The real payoff of having defined $\otimes$ by a universal property — converting multilinear into linear — is that the *same* recipe works whenever "bilinear" makes sense, and it gives the right notion every time.

- **Modules over a ring.** $M\otimes_R N$ linearises $R$-bilinear maps. Here dimension counting fails and genuinely new phenomena appear — tensoring is only right-exact, its failure measured by the $\operatorname{Tor}$ functors, and $\mathbb{Z}/2\otimes_{\mathbb{Z}}\mathbb{Z}/3 = 0$ shows simple tensors can collapse. The vector-space case is the deceptively easy one.
- **Algebras.** If $A,B$ are algebras, $A\otimes B$ is again an algebra under $(a\otimes b)(a'\otimes b') = aa'\otimes bb'$, the universal recipient of a pair of commuting homomorphisms. This is how you build $M_m\otimes M_n\cong M_{mn}$ and, in quantum information, the state space of a composite system.
- **Representations.** Tensoring representations of a group multiplies characters; decomposing $V\otimes W$ back into irreducibles is the Clebsch–Gordan problem, the algebra behind addition of angular momentum.
- **Hilbert spaces.** Here the algebraic tensor product is only a dense subspace; one completes it in the inner product $\langle v\otimes w, v'\otimes w'\rangle = \langle v,v'\rangle\langle w,w'\rangle$ to get $H_1\hat\otimes H_2$. The example to keep in mind: the **Hilbert–Schmidt operators** $\operatorname{HS}(H_1,H_2)$ are exactly $H_2\hat\otimes \overline{H_1}$, the completed tensor product, with the algebraic $H_2\otimes\overline{H_1}$ recovering the finite-rank operators — the honest infinite-dimensional version of the finite-dimensional iso $V^\ast\otimes W\cong\operatorname{Hom}(V,W)$ flagged earlier. (The conjugate $\overline{H_1}$ appears because the Riesz identification $H_1^\ast\cong\overline{H_1}$ is antilinear.) Trace-class operators correspond to the projective tensor norm, Hilbert–Schmidt to the Hilbert norm, compact to the injective norm — the choice of cross-norm is the new freedom that finite dimensions hid.
- **A machine-learning aside.** A feature map $\phi$ into a Hilbert space turns a kernel into an inner product $k(x,y)=\langle\phi(x),\phi(y)\rangle$, and the rank-one tensors $\phi(x)\otimes\phi(x)$ are exactly the building blocks whose averages give covariance operators — the same outer-product-as-simple-tensor picture, now in a (possibly infinite-dimensional) RKHS. The "tensors" of a deep network are arrays in fixed bases; their genuinely tensorial content is whatever survives a change of those bases, which for most layers is nothing — another reminder that *having indices is not the same as being a tensor*.

## What's really going on

Two closing remarks on the structure underneath everything above.

**Remark 1 (tensor product as multiplication).** The category $\mathbf{Vect}_{\mathbb F}$ has two ways to combine spaces: the direct sum $\oplus$, which adds dimensions, and the tensor product $\otimes$, which multiplies them, with $\mathbb{F}$ as multiplicative unit and the distributive law $U\otimes(V\oplus W)\cong (U\otimes V)\oplus(U\otimes W)$ holding canonically. So $(\mathbf{Vect},\oplus,\otimes)$ behaves like a rig (a ring without subtraction), and $\otimes$ is, quite literally, the multiplication of vector spaces. The "dimensions multiply" fact you proved by counting basis elements is the shadow of this. Categorically, $\otimes$ makes $\mathbf{Vect}$ a *symmetric monoidal category*, and that single fact organises everything from the trace to quantum entanglement.

**Remark 2 (the one identity to remember).** Of all the formulas, the load-bearing one is the **tensor–hom adjunction**:
$$
\operatorname{Hom}(U\otimes V,\;W)\;\cong\;\operatorname{Hom}\big(U,\;\operatorname{Hom}(V,W)\big).
$$
Read left to right it says a linear map out of $U\otimes V$ is the same as a bilinear map on $U\times V$ — our original universal property. Read as an adjunction it says $-\otimes V$ is left adjoint to $\operatorname{Hom}(V,-)$, which is the abstract reason tensoring is right-exact, the reason currying works, and the reason $\otimes$ and internal hom always come as a pair. If you remember one sentence from this post, remember that **the tensor product exists in order to represent multilinear maps as linear ones**, and that this single mandate forces the construction, the dimension count, the transformation law, the contractions, and the entire vocabulary that made "tensor" look like five different words. It was only ever one.