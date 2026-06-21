---
layout: default
title: "What is a tensor? The many faces of one idea"
date: 2026-06-21
tags: [linear-algebra, multilinear-algebra, functional-analysis, differential-geometry, machine-learning]
---

> **Conventions.** All vector spaces are over a fixed field $k$ (read $\mathbb{R}$ or $\mathbb{C}$ if you like); they are finite-dimensional unless I say otherwise, and I will be explicit at the one point where infinite dimensions change the story qualitatively. I write $V^\ast$ for the dual space of linear functionals $V \to k$, I use the Einstein summation convention (a repeated index, once up and once down, is silently summed) from the point where it earns its keep, and I reserve the word *map* for functions that respect whatever structure is in play. Throughout, "$\cong$" means "canonically isomorphic," and I will be fussy about the word *canonical* because that fussiness is the whole subject.

If you have spent any time near mathematics, physics, or machine learning, you have met the word **tensor** wearing wildly different costumes, and nobody warned you they were the same person.

- A machine-learning engineer says a tensor is *an $n$-dimensional array of numbers*: a scalar is a rank-$0$ tensor, a vector a rank-$1$ tensor, a matrix a rank-$2$ tensor, and `np.zeros((3,4,5))` a rank-$3$ tensor.
- A physicist says a tensor is *a thing that transforms correctly under a change of coordinates*, mutters about upper and lower indices, and writes $T^{i}{}\_{jk}$ as though that settled the matter.
- An algebraist says a tensor is *an element of a tensor product space* $V \otimes W$, defined by a universal property, and is visibly uninterested in your coordinates.
- A differential geometer says the metric, the curvature, and the stress–energy are tensors — meaning *tensor fields*, one tensor glued smoothly to each point of a manifold.
- A functional analyst says, ominously, that there is *more than one* tensor product of two Banach spaces, and that telling them apart was good enough for a thesis by Grothendieck.

These are not five unrelated abuses of a popular word. They are five projections of a single object onto five different screens. The goal of this post is to build that object once, carefully, and then watch each costume fall out of it. The punchline, which I will state now and earn later, is:

> **The tensor product $V \otimes W$ is the universal home for bilinear maps: it is the unique vector space through which every bilinear map out of $V \times W$ factors as a single *linear* map. Tensors are what you get when you insist on trading multilinearity for linearity.**

Everything else — the array of components, the transformation law, the multilinear functional — is bookkeeping for that one trade.

---

## 1. The real problem is multilinearity, not "high dimensions"

Linear algebra is, almost by definition, the study of *linear* maps $T : V \to W$, the ones satisfying $T(\lambda u + v) = \lambda T(u) + T(v)$. This theory is spectacularly complete: a linear map between finite-dimensional spaces is captured entirely by a matrix, and everything we love (rank, eigenvalues, decompositions) lives downstream of that.

But the moment you do anything *interesting* with more than one vector at a time, you leave the linear world:

- the dot product $\langle u, v\rangle$ is linear in $u$ for fixed $v$, and linear in $v$ for fixed $u$, but it is **not** linear in the pair $(u,v)$ jointly (doubling both quadruples the output);
- matrix multiplication $(A,B) \mapsto AB$ is bilinear;
- the determinant $\det(v_1, \dots, v_n)$, read as a function of its $n$ column vectors, is linear in each column separately;
- area, volume, the cross product, moments of inertia, the Riemann curvature — all are *multilinear*, linear in each slot when the others are frozen.

Call a map $b : V \times W \to U$ **bilinear** if it is linear in each argument separately. The trouble is that bilinear maps are second-class citizens of linear algebra: $V \times W$ is itself a vector space, but $b$ is emphatically *not* a linear map out of it (a linear map $V\times W \to U$ would have to satisfy $b(v_1+v_2, w_1+w_2) = b(v_1,w_1)+b(v_2,w_2)$, which is false — bilinearity gives cross terms). So all the heavy machinery built for linear maps simply does not apply to $b$.

**The driving question.** Can we manufacture a *new* space, built out of $V$ and $W$, so that bilinear maps out of $V \times W$ become honest *linear* maps out of the new space? If so, we get to reuse the entire theory of linear maps to study multilinear phenomena. That new space is the tensor product, and the rest is detail.

---

## 2. First face: components, or "a tensor is a box of numbers"

Before the abstraction, the cheap and concrete picture — the one machine learning runs on.

Fix bases $e_1, \dots, e_m$ of $V$ and $f_1, \dots, f_n$ of $W$. A bilinear form $b : V \times W \to k$ is completely determined by what it does to basis pairs, because bilinearity lets us expand:
$$
b\Big(\textstyle\sum_i x^i e_i,\ \sum_j y^j f_j\Big) \;=\; \sum_{i,j} x^i y^j \, b(e_i, f_j) \;=\; \sum_{i,j} B_{ij}\, x^i y^j,
\qquad B_{ij} := b(e_i, f_j).
$$
So the entire bilinear form collapses to an $m \times n$ array of numbers $B_{ij}$. Push this further: a *trilinear* map needs a three-index array $T_{ijk}$, and a $p$-fold multilinear map needs a $p$-dimensional array. **There is your "tensor = $n$-dimensional array."** The number of indices is what ML calls the tensor's *rank* (a usage I will object to in §9), and each index ranges over the dimension of one factor.

This picture is correct, useful, and a trap. It is a trap for the same reason that confusing a linear map with its matrix is a trap: **the array is a representation in a chosen basis, not the object itself.** Change the basis and every number in the box changes, even though the underlying multilinear map has not moved. A serious notion of "tensor" must say which arrays represent *the same* object — and that requirement is exactly the physicist's transformation law.

---

## 3. Second face: the transformation law, or "transforms correctly"

Let us see how the box of numbers reacts to a change of basis, because that reaction *is* the physicist's definition.

Suppose we switch to a new basis $e'_i = A^j{}\_i\, e_j$ (the new basis vectors expressed in the old ones; $A$ is the invertible change-of-basis matrix). A vector $v$ has the same identity in both bases, $v = v^i e_i = v'^i e'_i$, and chasing the substitution gives

$$
v'^i = (A^{-1})^i{}_j\, v^j.
$$

The components of a *vector* transform with $A^{-1}$ — **oppositely** to the basis. We call this **contravariant** and brand it with an **upper** index $v^i$.

Now dual vectors. The dual basis $\varepsilon^i$ is fixed by $\varepsilon^i(e_j) = \delta^i_j$, and a covector's components are $\omega_i = \omega(e_i)$. A quick check shows

$$
\omega'_i = A^j{}_i\, \omega_j,
$$

i.e. covector components transform with $A$ itself, **the same way** as the basis. We call this **covariant** and brand it with a **lower** index $\omega_i$.

A general tensor is something with several slots, some behaving like vectors and some like covectors, so its components carry a pattern of upper and lower indices, each upper one dragging an $A^{-1}$ and each lower one an $A$ under a change of basis:

$$
T'^{i_1 \cdots i_p}{}_{j_1 \cdots j_q}
= (A^{-1})^{i_1}{}_{a_1}\cdots (A^{-1})^{i_p}{}_{a_p}\;
A^{b_1}{}_{j_1}\cdots A^{b_q}{}_{j_q}\;
T^{a_1 \cdots a_p}{}_{b_1 \cdots b_q}.
$$

This is the law that the slogan *"a tensor is whatever transforms like a tensor"* is pointing at. It is not a mystical incantation; it is the precise statement that **the array is the shadow of a basis-independent object, and these are the shadow's marching orders when the light source moves.** The "type" $(p,q)$ — $p$ contravariant slots, $q$ covariant slots — is the only invariant data here. But notice the conceptual cost: this definition describes the object purely through how its *coordinates* wobble. It never says what the object *is*. For that we need the third face.

---

## 4. Third face: the tensor product and its universal property

Here is the actual construction, and it is worth slowing down because it answers the §1 driving question on the nose.

**The universal property (the definition that matters).**
A **tensor product** of $V$ and $W$ is a vector space $V \otimes W$ together with a bilinear map

$$
\otimes : V \times W \to V \otimes W, \qquad (v, w) \mapsto v \otimes w,
$$

that is *universal* among bilinear maps: for **every** vector space $U$ and **every** bilinear map $b : V \times W \to U$, there exists a **unique** *linear* map $\tilde b : V \otimes W \to U$ making the triangle commute,

$$
b(v, w) = \tilde b(v \otimes w) \quad \text{for all } v, w.
$$

Read that slowly, because it is the whole subject in one breath. It says: bilinear maps out of $V \times W$ are *the same data* as linear maps out of $V \otimes W$. The space $V \otimes W$ is precisely engineered to absorb the bilinearity, so that downstream you only ever see a linear map. **We traded multilinearity for linearity, exactly as promised.** Anything satisfying this property is unique up to a unique isomorphism — so it is legitimate to speak of *the* tensor product, and the universal property, not any particular construction, is its true identity card.

**A construction (to prove it exists).**
Take the free vector space $F$ on the *set* $V \times W$ — formal finite linear combinations of symbols $(v, w)$, with no relations at all, so $(v_1 + v_2, w)$ and $(v_1, w) + (v_2, w)$ are stubbornly different elements. Now quotient by the subspace $R$ generated by all the relations bilinearity *should* satisfy:

$$
(v_1 + v_2, w) - (v_1, w) - (v_2, w), \quad
(v, w_1 + w_2) - (v, w_1) - (v, w_2), \quad
(\lambda v, w) - \lambda(v, w), \quad
(v, \lambda w) - \lambda(v, w).
$$

Define $V \otimes W := F / R$ and $v \otimes w := [(v,w)]$. By construction $\otimes$ is bilinear, and one checks the universal property falls out: a bilinear $b$ defines a linear map on $F$ that kills $R$, hence descends to $V\otimes W$, uniquely. Existence done.

**Two facts you must internalize.**

1. *A basis, hence the array, reappears.* If $\lbrace e_i\rbrace$ is a basis of $V$ and $\lbrace f_j\rbrace$ a basis of $W$, then $\lbrace e_i \otimes f_j\rbrace$ is a basis of $V \otimes W$. Consequently

$$
\dim(V \otimes W) = \dim V \cdot \dim W,
$$

and a general element is $T = \sum_{i,j} T^{ij}\, e_i \otimes f_j$. **The numbers $T^{ij}$ are exactly the box of §2,** now understood as components with respect to a *specific basis of the tensor product*, and the transformation law of §3 is forced on them by how $e_i \otimes f_j$ changes when $e_i, f_j$ do. The three faces have started to merge.

2. *Most tensors are not simple.* Elements of the form $v \otimes w$ are called **simple** (or decomposable, or rank-one) tensors. They do **not** exhaust $V \otimes W$ — they form a thin nonlinear cone inside it, and the generic element is a genuine *sum* $\sum_k v_k \otimes w_k$ that cannot be collapsed to a single $v \otimes w$. This is the single most common misconception about tensor products. (Sharp statement: $v\otimes w = 0$ iff $v = 0$ or $w = 0$; and $v_1 \otimes w_1 = v_2 \otimes w_2 \neq 0$ iff $v_2 = \lambda v_1$ and $w_2 = \lambda^{-1} w_1$ for some scalar $\lambda$. So even the simple tensors have just a one-parameter redundancy, and everything else is honest sums.)

---

## 5. Reconciliation: the faces were one face

With the construction in hand, the menagerie collapses. The key is a single family of canonical isomorphisms.

**Tensors are multilinear maps.** The universal property, run in reverse and applied to functionals, gives the workhorse identifications (finite dimensions):

$$
V^\ast \otimes W^\ast \;\cong\; \mathrm{Bil}(V \times W,\, k), \qquad
\underbrace{V^\ast \otimes \cdots \otimes V^\ast}_{p} \;\cong\; \{p\text{-linear forms on } V\}.
$$

So "element of a tensor product" and "multilinear map" are literally the same thing — the simple tensor $\phi \otimes \psi$ *is* the bilinear form $(v,w) \mapsto \phi(v)\psi(w)$. The physicist's tensor and the algebraist's tensor were never different.

**Matrices are tensors — and which kind.** The single most clarifying isomorphism in the subject is

$$
\operatorname{Hom}(V, W) \;\cong\; V^\ast \otimes W,
$$

where the simple tensor $\phi \otimes w$ acts as the rank-one map $v \mapsto \phi(v)\,w$ (an outer product, in coordinates). A linear map is thus a type-$(1,1)$ tensor: one covariant slot to eat the input vector, one contravariant slot to produce the output. This is why a matrix carries one upper and one lower index, $A^i{}_j$, and why matrix entries transform the way they do — it was never a convention, it was the type.

**The general type-$(p,q)$ tensor** is then exactly an element of

$$
T^{(p,q)}(V) \;=\; \underbrace{V \otimes \cdots \otimes V}_{p} \otimes \underbrace{V^\ast \otimes \cdots \otimes V^\ast}_{q},
$$

with $\dim = (\dim V)^{p+q}$, and §3's transformation law is now a *theorem* about how the basis $e_{i_1}\otimes\cdots\otimes \varepsilon^{j_q}$ responds to a change of basis, not an axiom imposed from outside.

**Contraction is just the evaluation pairing.** The natural map $V \otimes V^\ast \to k$, $v \otimes \phi \mapsto \phi(v)$, is the trace; applied to one upper and one lower slot of a bigger tensor it is **contraction**, $T^{i}{}\_{i}$ (summed). The trace of a matrix is the contraction of its $(1,1)$-tensor. And — flagging this for later — the operation an ML library calls `einsum` *is* tensor contraction: you name the indices, repeat the ones you want summed, and the Einstein convention does the rest.

**Symmetry splits the box.** Inside $V \otimes V$ live the symmetric part $\mathrm{Sym}^2 V$ and the antisymmetric part $\Lambda^2 V$, and more generally the symmetric powers (think: polynomials, symmetric bilinear forms, the metric) and the exterior powers $\Lambda^k V$ (think: oriented $k$-volumes, differential forms, and the determinant, which is the single basis element of the top power $\Lambda^n V \cong k$). The whole apparatus of differential forms is the antisymmetric corner of the tensor world.

So: array, transformation law, multilinear map, abstract element — **one object, four descriptions.** Data, marching orders, function, construction.

---

## 6. Fourth face: tensor fields, where the geometry lives

Now glue. A smooth manifold $M$ has, at each point $x$, a tangent space $T_x M$ — a genuine vector space, but a *different* one at each point, with no canonical way to identify $T_x M$ with $T_y M$. A **tensor field** of type $(p,q)$ assigns to each $x$ a tensor in $T^{(p,q)}(T_x M)$, varying smoothly with $x$.

- The **metric** $g$ is a smoothly varying symmetric $(0,2)$-tensor: at each point it eats two tangent vectors and returns a number (their inner product), letting you measure lengths and angles.
- The **stress–energy** and **Riemann curvature** tensors are higher-type fields of exactly this kind.

And now the physicist's transformation law of §3 returns with a vengeance, *because it has to*: moving between two coordinate charts on $M$ is precisely a change of basis on each $T_x M$, with the change-of-basis matrix being the Jacobian $\partial x'^i / \partial x^j$ of the transition map. Covariant and contravariant indices transform with the Jacobian and its inverse, point by point. The reason physics is *obsessed* with the transformation law is that on a manifold there is no global basis to hide in, so the only way to certify that a coordinate expression denotes a real geometric object — and not a coordinate artifact — is to check it transforms correctly. **The transformation law is not the definition of a tensor; it is the field-theoretic *verification* that you are looking at one.**

So the geometer's "tensor" is just §4's tensor, evaluated in the tangent space, repeated smoothly across the manifold. Same object, now a section of a bundle.

---

## 7. Fifth face: infinitely many tensor products (the analyst's warning)

Everything above used finite dimension exactly once but crucially: that a tensor product has a basis $\lbrace e_i \otimes f_j\rbrace$ and the right dimension. In infinite dimensions the *algebraic* tensor product $V \otimes W$ (finite sums of simple tensors) still exists and still has the universal property — but for Banach or Hilbert spaces it is **not complete**, and completing it forces a genuine choice, because there is no single canonical norm on $V \otimes W$. This is where "there are many tensors" becomes literally true in a new way.

Two extreme cross-norms bracket all the reasonable ones. For $x \in V \otimes W$:

$$
\|x\|_\pi \;=\; \inf\Big\{ \textstyle\sum_k \|a_k\|\,\|b_k\| \;:\; x = \sum_k a_k \otimes b_k \Big\}
\qquad \text{(projective, the largest),}
$$

$$
\|x\|_\varepsilon \;=\; \sup\Big\{ \big| \textstyle\sum_k \phi(a_k)\,\psi(b_k) \big| \;:\; \|\phi\|_{V^\ast},\,\|\psi\|_{W^\ast} \le 1 \Big\}
\qquad \text{(injective, the smallest).}
$$

Completing in these norms yields genuinely different Banach spaces $V \widehat{\otimes}_\pi W$ and $V \widehat{\otimes}_\varepsilon W$, and the gap between them is the entire content of Grothendieck's *Résumé*. Concretely and beautifully, for the duality with operators,

$$
V^\ast \,\widehat{\otimes}_\pi\, W \;\cong\; \{\text{nuclear operators } V \to W\}, \qquad
V^\ast \,\widehat{\otimes}_\varepsilon\, W \;\cong\; \overline{\{\text{finite-rank operators}\}} = \{\text{approximable operators}\},
$$

so the *same* algebraic tensors, completed differently, become trace-class operators on the one hand and (norm-limits of finite rank) compact operators on the other.

The Hilbert space case is the one you will meet most and is the most pleasant: there is a canonical inner product on $H_1 \otimes H_2$ with $\langle a_1 \otimes b_1, a_2 \otimes b_2\rangle = \langle a_1, a_2\rangle \langle b_1, b_2\rangle$, and its completion $H_1 \widehat{\otimes} H_2$ is *the* Hilbert space tensor product. This is exactly the structure behind two facts you already use:

$$
L^2(\mu) \,\widehat{\otimes}\, L^2(\nu) \;\cong\; L^2(\mu \times \nu), \qquad
\overline{H} \,\widehat{\otimes}\, H \;\cong\; \{\text{Hilbert–Schmidt operators on } H\}.
$$

The first says separable-variable functions span the joint $L^2$; the second realizes Hilbert–Schmidt operators as honest square-summable tensors, with the HS norm the $\ell^2$ cross-norm sitting strictly between $\|\cdot\|\_\varepsilon$ and $\|\cdot\|\_\pi$. (Sharp remark: $\|\cdot\|\_\pi = \|\cdot\|\_\varepsilon$ on $V \otimes W$ for *all* $W$ iff $V$ is finite-dimensional. The proliferation of tensor norms is precisely an infinite-dimensional phenomenon; in finite dimensions the §4 object is unique and all this collapses.)

---

## 8. Sixth face: what machine learning actually means

Now we can be honest, and charitable, about the data-science usage.

When PyTorch or NumPy calls an array a "tensor," it means **face one** of §2: a multidimensional array, a data container. A mathematician winces because the array is presented in a *fixed* basis (the standard one) with *no* transformation law attached — there is no claim that anything transforms correctly, and indices are not typed as up or down. By the strict geometric definition, a raw NumPy array is a tensor only in the trivial sense that any box of numbers represents *some* tensor once you fix a basis and a type. So the wince is fair: the ML "tensor" keeps the data and discards the structure that made it a tensor.

But the deeper structure is not absent in ML — it is just unlabelled, and recognizing it pays off:

- **A bilinear layer** $\;y = x_1^\top W x_2\;$ is literally a type-$(0,2)$ tensor $W$ contracted against two input vectors. The attention score $q^\top k$ before softmax is a bilinear form; with a learned $q^\top W k$ it is a general one.
- **`einsum` is contraction.** Every `einsum` string is an Einstein-summation spec: shared indices are contracted (summed), free indices survive. Once you see `'ij,jk->ik'` as "contract the middle slot," matrix multiplication, batched attention, and convolutions all become the same operation in different index patterns.
- **Tensor decompositions are the real subject.** CP / CANDECOMP–PARAFAC writes a tensor as a minimal sum of simple tensors $\sum_{k=1}^r a_k \otimes b_k \otimes c_k$; Tucker writes it as a small core tensor contracted with factor matrices. These are the higher-order analogues of the SVD, and they are exactly §4's "general tensors are sums of simple ones" turned into an algorithm.

The honest summary: ML's *noun* "tensor" is face one, but ML's *verbs* — contraction, products, decomposition, the entire index gymnastics of a transformer — are shadows of the genuine structure of §4–§5. Learning the structure lets you read the verbs as mathematics rather than as array-indexing folklore.

---

## 9. A bonus confusion worth defusing: the two "ranks"

While we are clearing the air: the word **rank** is overloaded across these communities, and it bites people.

- ML's "rank-$3$ tensor" means the *number of indices* — better called the **order** or **degree** of the tensor. This is just "how many slots."
- The mathematician's **tensor rank** is the §4–§8 notion: the *minimal number of simple tensors* in a decomposition $\sum_{k=1}^r v_k \otimes w_k \otimes \cdots$. For order-$2$ tensors this is exactly matrix rank, computed cleanly by the SVD.

These are unrelated, and the second is genuinely harder than your matrix intuition suggests: for order $\ge 3$, tensor rank can *exceed* every dimension, can differ over $\mathbb{R}$ versus $\mathbb{C}$, the set of low-rank tensors is **not** closed (so "best rank-$r$ approximation" may fail to exist), and computing the rank is **NP-hard** in general — a sharp contrast with the tame, SVD-soluble matrix case. The jump from order $2$ to order $3$ is the jump from linear algebra to a genuinely wild theory.

---

## 10. What's really going on

If you remember one sentence, remember the universal property: **$V \otimes W$ is the receiver that turns every bilinear map into a single linear map, and a tensor is an element of such a receiver.** From that one idea:

- the **array** is the component representation in a basis $\lbrace e_i \otimes f_j\rbrace$ (§2, §4);
- the **transformation law** is how those components must move so the underlying element stays put — a theorem, not an axiom (§3, §5);
- the **multilinear map** picture is the dual reading $V^\ast \otimes W^\ast \cong \mathrm{Bil}(V\times W)$, with $\operatorname{Hom}(V,W)\cong V^\ast\otimes W$ as its most useful special case (§5);
- the **tensor field** is this object placed in each tangent space and varied smoothly, where the transformation law re-emerges as the consistency check across charts (§6);
- the **many analytic tensor products** are what happens when dimension goes to infinity and the single finite-dimensional object splinters into a family indexed by your choice of cross-norm (§7);
- the **ML tensor** is face one with its structure left implicit, and reattaching the structure is what turns `einsum` and CP-decomposition back into mathematics (§8–§9).

Two parting remarks on where this generalizes. First, the universal-property style of definition is not a flourish — it is the template for *every* "free" construction in algebra (free groups, free modules, polynomial rings, group algebras all factor a hard map through an easy linear one in the same way), and tensor products are simply its multilinear instance. Second, the slogan worth carrying past this post is that **tensors are how mathematics linearizes the nonlinear**: whenever something is multilinear — a product, a curvature, a moment, an interaction term — you can launder it through a tensor product and hand it to the full, finished theory of linear maps. That laundering, performed once and reused everywhere, is why a single word ended up wearing so many costumes.