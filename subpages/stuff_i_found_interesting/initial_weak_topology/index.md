## Setup and convention

# Initial Topology


<div class="math-callout math-callout--info" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Motivation</span><span class="math-callout__name">(The driving idea)</span></p>

Throughout, $X$ is a bare set, $(Y_i)\_{i\in I}$ a family of topological spaces, and $f_i \colon X \to Y_i$ a family of maps. The question "what topology should $X$ carry?" has a canonical answer whenever the *only* structure you care about on $X$ is the one transmitted to it through the $f_i$. The **initial topology** is that answer.

You are pulled in two directions. Making the $f_i$ continuous wants *many* open sets (you must throw in every $f_i^{-1}(U)$). Continuity of maps *into* $X$ wants *few* open sets (fewer opens = easier to be continuous). The initial topology is the equilibrium: the **coarsest topology making all the $f_i$ continuous.** Everything below is a consequence of taking "coarsest" seriously.

</div>

"Open sets" here always refers to open sets *in $X$* — the topology we are trying to choose. The two pulls are constraints on that topology.

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why making the $f_i$ continuous wants *many* open sets in $X$.)</span></p>

By definition, $f_i \colon X \to Y_i$ is continuous iff $f_i^{-1}(U) \subseteq X$ is open in $X$ for **every** open $U \subseteq Y_i$. So each open set in the *target* $Y_i$ *forces* a corresponding subset of $X$ to be open. The more maps $f_i$ you bolt on — and the more opens each $Y_i$ has — the more subsets of $X$ you are forced to include in $\tau_X$.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">()</span></p>

Let $X = \lbrace a, b\rbrace$ and $f \colon X \to \mathbb{R}$ with $f(a) = 0$, $f(b) = 1$. Then

$$
f^{-1}\bigl((-\tfrac12, \tfrac12)\bigr) = \lbrace a\rbrace, \qquad f^{-1}\bigl((\tfrac12, \tfrac32)\bigr) = \lbrace b\rbrace,
$$

so making $f$ continuous *forces* both $\lbrace a\rbrace$ and $\lbrace b\rbrace$ to be open in $X$ — i.e. forces $\tau_X$ to be the full discrete topology $\mathcal{P}(X)$. The indiscrete topology $\lbrace\emptyset, X\rbrace$ would not suffice; it is "too small."

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why continuity of maps *into* $X$ wants *few* open sets in $X$.)</span></p>

A map $g \colon Z \to X$ is continuous iff $g^{-1}(V) \subseteq Z$ is open in $Z$ for **every** open $V \subseteq X$. So each open set in $X$ is *one more preimage* $g$ must keep open. The more opens $X$ has, the more conditions $g$ must satisfy — and the easier it is for some $g$ to fail.

</div>

<div class="math-callout math-callout--question" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">()</span></p>

If $X = \mathbb{R}$ carries the **discrete** topology, then the identity map $\mathrm{id} \colon (\mathbb{R}, \tau\_{\text{usual}}) \to (\mathbb{R}, \tau\_{\text{discrete}})$ is *not* continuous: $\lbrace 0\rbrace$ is open in the target, but $\mathrm{id}^{-1}(\lbrace 0\rbrace) = \lbrace 0\rbrace$ is not open in the standard topology on $\mathbb{R}$. If instead $X$ carries the **indiscrete** topology, *every* map into $X$ is automatically continuous, because the only opens are $\emptyset$ and $X$, whose preimages are trivially open.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The equilibrium)</span></p>

**The equilibrium.** The initial topology resolves the tension by taking the *smallest* (= coarsest) topology that still meets the first constraint: it throws in every $f_i^{-1}(U)$ — nothing more, nothing less. The first constraint is honored *exactly*; the second is honored *as much as the first allows*.

</div>

## Definition and existence

For each $i$, continuity of $f_i$ forces every $f_i^{-1}(U)$ ($U\subseteq Y_i$ open) to be open in $X$. So consider the subbasis

$$
\mathcal{S} = \lbrace\, f_i^{-1}(U) : i\in I,\ U\subseteq Y_i \text{ open}\,\rbrace
$$

and let $\tau$ be the topology it generates (arbitrary unions of finite intersections of members of $\mathcal{S}$). This $\tau$ is well-defined and is exactly the initial topology: an arbitrary intersection of topologies is a topology, the discrete topology makes everything continuous, so the family of topologies rendering all $f_i$ continuous is nonempty and closed under intersection — its intersection is the coarsest one, and unwinding what "$f_i$ continuous" demands recovers precisely the topology generated by $\mathcal S$.

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Subbasis and topology generated by it)</span></p>

A **subbasis** for a topology on $X$ is *any* collection $\mathcal{S} \subseteq \mathcal{P}(X)$ of subsets of $X$ — no closure conditions are required. The **topology generated by $\mathcal{S}$** is then the family

$$
\tau(\mathcal{S}) = \Bigl\lbrace\, \bigcup\_{\lambda \in \Lambda} (S_{\lambda,1} \cap \dots \cap S_{\lambda,n_\lambda}) \;:\; S_{\lambda,k} \in \mathcal{S} \,\Bigr\rbrace,
$$

i.e. arbitrary unions of *finite* intersections of elements of $\mathcal{S}$ (together with $\emptyset$ and $X$ by convention). Equivalently, $\tau(\mathcal{S})$ is the smallest topology on $X$ containing $\mathcal{S}$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Basis vs. subbasis)</span></p>

A **basis** $\mathcal{B}$ is a collection with an extra closure property: the finite intersection of two basis elements must itself be a *union* of basis elements. The topology is then just $\lbrace$ arbitrary unions of $\mathcal{B}$-elements $\rbrace$ — no intersections needed at the end.

For a **subbasis** no such closure is assumed; you *manufacture* a basis from $\mathcal{S}$ by closing it under finite intersections, then close *that* under arbitrary unions to get the topology. So a subbasis is "even less than a basis": you can throw in any generators you want, and the closure procedure produces a legitimate topology.

This is why the construction above works: $\mathcal{S} = \lbrace f_i^{-1}(U) \rbrace$ has *no* reason to be closed under finite intersection in general (e.g. $f_i^{-1}(U) \cap f_j^{-1}(V)$ for $i \neq j$ is typically not of the form $f_k^{-1}(W)$ for any single $k$), so calling it a *sub*basis — not a basis — is the honest description.

</div>

## The real content: the universal property

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What "universal property" means, and why this one is "initial")</span></p>

**Universal.** A *universal property* characterizes an object by a statement that quantifies over **every** other object in the ambient category. Here the property is:

> $\tau$ is the unique topology on $X$ such that **for every** topological space $Z$ and **for every** map $g \colon Z \to X$, $g$ is continuous iff each $f_i \circ g$ is continuous.

The "for every $Z$" is the universal quantifier — that is the precise sense in which the property is *universal*. The merit is **characterization without construction**: any topology on $X$ satisfying the property must equal $\tau$. So $\tau$ is fixed by how it *interacts with the outside world* (maps in from arbitrary $Z$), not by the messy "unions of finite intersections of pullbacks" recipe.

**Initial.** The word "initial" is borrowed from category theory.

* In a category $\mathcal{C}$, an object $I$ is **initial** if for every object $X \in \mathcal{C}$ there is *exactly one* morphism $I \to X$. The defining feature: arrows go *out* of $I$.
* Dually, a **terminal** (or **final**) object is one with exactly one morphism *into* it from every object.

Now consider the category whose objects are topologies on the bare set $X$ that make all the $f_i$ continuous, with refinement as morphisms. The *coarsest* such topology is initial in this category: every other admissible topology sits "above" it, and the unique arrow goes outward. Equivalently — and this is the picture that matches the slogan — $\tau$ is initial because *the universal property tests it via arrows going **out** through the $f_i$* (or equivalently, via arrows coming **in** through $g \colon Z \to X$, which we then *compose* with the outgoing $f_i$).

**The duality, in one sentence.** Initial topology $=$ "as **coarse** as possible, tested by maps **into** $X$"; **final topology** $=$ "as **fine** as possible, tested by maps **out of** $X$." Subspace and product live on the initial side; quotient and disjoint union on the final side.

</div>

The reason the initial topology matters is not the construction but this characterization.

**Mapping-in property.** For any topological space $Z$ and any map $g\colon Z\to X$,

$$
g \text{ is continuous} \iff f_i\circ g \text{ is continuous for every } i\in I.
$$

The forward direction is trivial. For the reverse, it suffices to check preimages of subbasic opens, and

$$
g^{-1}\!\big(f_i^{-1}(U)\big) = (f_i\circ g)^{-1}(U),
$$

which is open by hypothesis. This property *characterizes* $\tau$ uniquely among all topologies on $X$: it is the topology for which **continuity into $X$ is tested one coordinate at a time.** That is the whole point — you trade a single hard continuity question for a family of easy ones.

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">("One coordinate at a time", and the trade)</span></p>

**The slogan.** "Continuity into $X$ is tested one coordinate at a time" means: instead of staring at the (potentially mysterious) topology on $X$ and verifying that $g^{-1}(U)$ is open for *every* open $U \subseteq X$, you only need to check that each *coordinate* $f_i \circ g \colon Z \to Y_i$ is continuous as a map into the *simpler* space $Y_i$. The $f_i$ play the role of "coordinates" or "test maps"; the universal property says these tests are *jointly sufficient*.

**The trade.** "A single hard question for a family of easy ones" unpacks as:

* the *hard* question is **"is $g \colon Z \to X$ continuous?"** — hard because the topology on $X$ may have no usable concrete description (e.g. the weak topology on a Banach space has *no* metric, the product topology on $\mathbb{R}^\mathbb{R}$ has uncountably many "coordinate constraints" and no basis you can write down compactly);
* the *easy* questions are **"is $f_i \circ g \colon Z \to Y_i$ continuous?"** for each $i$ — easy because each $Y_i$ is, by assumption, a space you already understand (often $\mathbb{R}$ or $\mathbb{C}$).

The universal property turns the *one* opaque check on $X$ into the *family* of transparent checks on the $Y_i$, **without losing information** — the two are logically equivalent. That equivalence is what the construction of $\tau$ buys you, and it is the entire reason to bother with the initial topology in the first place.

**Concrete example (finite-dimensional).** Take $X = \mathbb{R}^n$ with the product topology and $f_i = \pi_i \colon \mathbb{R}^n \to \mathbb{R}$ the $i$-th coordinate projection. A map $g \colon Z \to \mathbb{R}^n$, $g(z) = (g_1(z), \dots, g_n(z))$, is continuous iff each component $g_i = \pi_i \circ g \colon Z \to \mathbb{R}$ is continuous. This is the elementary **"componentwise continuity"** theorem — and it is *exactly* the initial-topology universal property for $\mathbb{R}^n$.

**Concrete example (functional analysis).** A map $g \colon Z \to (E, \text{weak})$ into a Banach space equipped with its weak topology is continuous iff $\varphi \circ g \colon Z \to \mathbb{K}$ is continuous for *every* $\varphi \in E^\ast$. The first question — "$g$ continuous into the weak topology" — is opaque because the weak topology has no metric. The second is a family of completely standard scalar-valued continuity checks. That swap is what the initial-topology technology buys you in functional analysis.

</div>

**Net/convergence form.** Dually, $x_\alpha \to x$ in the initial topology $\iff f_i(x_\alpha)\to f_i(x)$ for all $i$. Convergence is "coordinatewise against the test maps." Keep this picture in mind for the examples.

## Why we need it — the motivation made concrete

The abstraction earns its keep because the topologies you already use *are* initial topologies, and recognizing them as such immediately gives you their good behavior for free.

**Subspace topology.** $A\subseteq X$ with the single [inclusion](https://en.wikipedia.org/wiki/Inclusion_map) $\iota\colon A\hookrightarrow X$. The initial topology is the subspace topology, and the universal property is the familiar "a map into $A$ is continuous iff it is continuous as a map into $X$."

**Product topology.** $\prod_i Y_i$ with the projections $\pi_i$. The initial topology is the product topology — and this is the honest reason the product topology, not the box topology, is the "correct" one. The box topology makes each $\pi_i$ continuous too, but it is strictly finer, hence fails the coarseness that yields the universal property: with the box topology a map into the product need *not* be continuous merely because all its components are. The product topology is engineered precisely so that "continuous into the product" $=$ "continuous in each coordinate." (This is also the structural origin of Tychonoff's theorem behaving well.)

**Weak topology on a normed space $E$.** Take $I = E^\ast$ (the continuous dual) and the maps to be the functionals $\varphi\colon E\to \mathbb{K}$ themselves. The initial topology is the weak topology $\sigma(E,E^\ast)$ — by construction the coarsest topology keeping every continuous functional continuous. The convergence form says exactly: $x_\alpha \rightharpoonup x$ iff $\varphi(x_\alpha)\to\varphi(x)$ for all $\varphi\in E^\ast$, which is the *definition* of weak convergence you already know. The universal property is why weak continuity is so often checkable.

**Weak-\* topology on a dual $E^\ast$.** Now put $I = E$ and use the evaluation maps $\widehat{x}\colon E^\ast\to\mathbb{K}$, $\varphi\mapsto \varphi(x)$. The initial topology is $\sigma(E^\ast,E)$. This is the topology of pointwise convergence on $E^\ast$, and its coarseness is what makes the unit ball compact (Banach–Alaoglu) — compactness is *easy* in a coarse topology, so designing the weak-\* topology to be as coarse as possible while still separating points is exactly the right move.

In each case the *same* two-line argument (subbasis = pullbacks; continuity-in iff continuity-coordinatewise) does all the work, which is the economy the concept buys you.

## Remarks

1. **The categorical reading.** The initial topology is a limit. Subspace = equalizer-flavored, product = product, the general construction = the limit of the diagram of forgetful constraints. The mapping-in universal property is just the limit's universal property, which is why "initial" topologies are tested by maps *in*. The mirror notion is the **final topology** (the *finest* topology making a family $g_i\colon Y_i\to X$ continuous): it is a colimit, characterized by maps *out* ($h\colon X\to Z$ continuous iff each $h\circ g_i$ is), and it specializes to the quotient topology, the disjoint-union/coproduct topology, and direct limits. Subspace/product live on the initial side; quotient/coproduct on the final side. The duality is exact.

2. **Coarseness is a feature, not a defect.** A recurring theme in functional analysis — Banach–Alaoglu, the Krein–Milman setting, weak compactness via Eberlein–Šmulian — is that you *want* the weakest topology that still does a required job (here: keep certain maps continuous, separate points), because compactness, lower semicontinuity, and existence-of-extremizer arguments all get cheaper as the topology coarsens. The initial topology is the formal device that hands you "as coarse as possible, subject to constraints." That is the motivation in one sentence.

# Weak Topology

## Setup and convention

$E$ is a normed space (everything below works verbatim for a Hausdorff locally convex space) over $\mathbb{K}\in\lbrace \mathbb R,\mathbb C\rbrace$, and $E^\ast$ is its **continuous dual**, the space of continuous linear functionals $\varphi\colon E\to\mathbb K$. The norm topology on $E$ — call it the *strong* topology — is the one you start with. The weak topology is a deliberately *coarser* competitor placed alongside it.

**The driving idea.** In infinite dimensions the norm topology is too fine to be useful for existence arguments: by Riesz's theorem the closed unit ball is norm-compact *iff* $\dim E<\infty$. Compactness is the engine of "a minimizing sequence has a convergent subsequence," so losing it is fatal. The cure is to coarsen the topology — fewer open sets means more compact sets — but you cannot coarsen recklessly, or you lose the ability to see linear structure. The weak topology is the principled coarsening: **the coarsest topology on $E$ that still keeps every continuous functional continuous.** That single sentence is simultaneously the definition, the motivation, and the link to the previous discussion.

## Definition

The **weak topology** $\sigma(E,E^\ast)$ is the initial topology induced on $E$ by the family $E^\ast=\lbrace\varphi\colon E\to\mathbb K\rbrace$. Concretely it has subbasis $\lbrace\varphi^{-1}(U):\varphi\in E^\ast,\ U\subseteq\mathbb K\text{ open}\rbrace$, and because the maps are linear into a TVS it is exactly the locally convex topology generated by the seminorms

$$
p_\varphi(x)=|\varphi(x)|,\qquad \varphi\in E^*.
$$

A neighborhood base at $0$ consists of the convex sets $\lbracex:\|\varphi_1(x)\|<\varepsilon,\dots,\|\varphi_n(x)\|<\varepsilon\rbrace$ for finite $\lbrace\varphi_i\rbrace\subset E^\ast$ and $\varepsilon>0$. The convergence form is the one you already use as the *definition* of weak convergence:

$$
x_\alpha\rightharpoonup x \iff \varphi(x_\alpha)\to\varphi(x)\ \text{ for all }\varphi\in E^*.
$$

## Why we need it — motivation made concrete

**Recovering compactness.** This is the headline. Coarsening to $\sigma(E,E^\ast)$ buys back compactness lost in infinite dimensions: in a *reflexive* space the closed unit ball is weakly compact (Kakutani), and for duals the unit ball is weak-\* compact unconditionally (Banach–Alaoglu). Compactness is cheap in a coarse topology, so the whole design is "make it as coarse as possible while still separating points."

**The direct method.** Pair compactness with the fact that the norm — and, by Mazur's theorem, any convex norm-closed set — is *weakly* lower semicontinuous. Then a coercive convex functional on a reflexive space attains its minimum: extract a bounded minimizing sequence, pass to a weakly convergent subnet by compactness, and lower semicontinuity finishes. This is the backbone of the calculus of variations, PDE existence theory, and optimization. The weak topology exists so this argument runs.

**The natural notion of limit.** In many settings weak convergence, not norm convergence, is the right object — oscillations that don't decay in norm still converge weakly (e.g. $\sin(nx)\rightharpoonup 0$ in $L^2$), which is precisely the phenomenon you want to track.

## How it relates to the initial topology

There is nothing to prove here, only to observe: **the weak topology *is* an initial topology**, the special case where the family of maps is the continuous dual $E^\ast$ and every target is the scalar field $\mathbb K$. Every general fact specializes for free:

- the *mapping-in property* — $g\colon Z\to(E,\text{weak})$ is continuous iff $\varphi\circ g$ is continuous for all $\varphi\in E^\ast$;
- the *convergence form* above is the coordinatewise-convergence statement of the initial topology, read against the test maps $\varphi$;
- the subbasis is the standard "pullbacks of opens" subbasis.

Its sibling, the **weak-\* topology** $\sigma(E^\ast,E)$ on the dual, is the *same* construction with the roles swapped: the initial topology on $E^\ast$ induced by the evaluation maps $\widehat x\colon\varphi\mapsto\varphi(x)$, $x\in E$. Both are initial topologies; they differ only in which family of functionals you feed the machine.

## Similarities and differences

**Similarities** are exactly the inherited structure: same defining property (coarsest making the family continuous), same universal property (test continuity one functional at a time), same subbasis, same convergence description. If you understand initial topologies, you understand the *formal* behavior of the weak topology at no extra cost.

The **differences** are where the weak topology earns its own name. They all stem from the family being not an arbitrary set of maps but the *linear dual* of a *vector space*:

1. **Linear structure is automatic.** Because the maps are linear and the targets are TVSs, the weak topology is forced to be a *locally convex vector topology* — generated by seminorms, with convex basic neighborhoods. A generic initial topology has none of this: arbitrary targets and arbitrary maps give no seminorms, no convexity, no homogeneity. The weak topology lives in the category of locally convex spaces, not merely in $\mathbf{Top}$.

2. **Separation is a theorem, not a hypothesis.** A general initial topology is Hausdorff iff the family separates points — a near-tautology you simply check. For the weak topology, the statement "$E^\ast$ separates the points of $E$" is *Hahn–Banach*. So the Hausdorffness of $\sigma(E,E^\ast)$ rests on a genuine analytic theorem rather than a definitional checkbox.

3. **The family is canonical, not chosen.** With an initial topology you supply whatever maps you please, and different choices give different topologies — the family is extra data. The weak topology's family is determined by $E$ itself (the *entire* continuous dual), so it is an intrinsic invariant of the normed space, not an additional input.

4. **There is a pre-existing topology to compete with.** A bare set carrying an initial topology has no rival structure; there is nothing to compare against. But $E$ already has its norm topology, and the entire point is the comparison: $\sigma(E,E^\ast)\subseteq\text{norm topology}$ always, with **equality iff $\dim E<\infty$**. The "weak vs. strong" drama — weakly closed $\Rightarrow$ norm-closed but not conversely (the unit sphere is weakly dense in the ball), convergence weaker than norm convergence — exists only because the construction is deliberately coarsening a topology that is already there.

5. **Failure of metrizability / sequences are not enough.** On an infinite-dimensional $E$ the weak topology is never metrizable and not first-countable, so you must reason with nets, and weak sequential closure can differ from weak closure. This is a consequence of $E^\ast$ being "large" — an uncountable separating family — rather than of the initial-topology construction itself (a *countable* initial family, e.g. a countable product of metric spaces, stays metrizable). It is the reason Eberlein–Šmulian (which rescues sequences for *compactness* specifically) is a nontrivial theorem rather than an observation.

## Remarks

1. **Where this really lives.** The honest general setting is a *dual pair* $\langle E,F\rangle$ of vector spaces with a separating bilinear form; $\sigma(E,F)$ is then the weak topology of the pairing, and weak/weak-\* are the two ends of the same construction. The Mackey–Arens theorem says all locally convex topologies yielding the *same* dual $F$ sit between $\sigma(E,F)$ (coarsest) and the Mackey topology $\tau(E,F)$ (finest). In this light the weak topology is the *bottom* element of a whole lattice of "polar topologies" compatible with the duality — the maximally coarse choice, which is exactly the property the initial-topology construction guarantees.

2. **The one-sentence summary.** A weak topology is what you get when you point the initial-topology machine at a space's own dual; everything distinctive about it — local convexity, Hahn–Banach separation, the coarseness that resurrects compactness — comes not from the *construction* (which it shares with subspaces and products) but from the *linear input* you feed it.

# Appendix 

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Topology)</span></p>

A **topology** on a set $X$ is a collection $\tau \subseteq \mathcal{P}(X)$ of subsets of $X$ — called **open sets** — satisfying:

1. $\emptyset \in \tau$ and $X \in \tau$.
2. **Closed under arbitrary unions:** if $(U_i)\_{i\in I} \subseteq \tau$, then $\bigcup\_{i\in I} U_i \in \tau$.
3. **Closed under finite intersections:** if $U, V \in \tau$, then $U \cap V \in \tau$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Topological space)</span></p>

A **topological space** is a pair $(X, \tau)$ where $X$ is a set and $\tau$ is a topology on $X$. The elements of $X$ are called **points**, the elements of $\tau$ the **open sets** of the space.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Topology vs. topological space)</span></p>

The distinction is the same as between a *group operation* $\cdot$ and a *group* $(G, \cdot)$:

* a **topology** is the *structure* — a collection of subsets satisfying the three axioms;
* a **topological space** is the *set together with that structure* — the pair $(X, \tau)$.

The same underlying set $X$ can carry many different topologies. Two extreme examples on the same $X$:

* the **discrete topology** $\tau = \mathcal{P}(X)$ — *every* subset is open;
* the **indiscrete (trivial) topology** $\tau = \lbrace\emptyset, X\rbrace$ — only the empty set and $X$ are open.

Each choice produces a *different* topological space, with different continuous maps, different convergent sequences, and different compact sets. So when one writes "the space $X$" by abuse of language, the topology is always implicit in the background — continuity, convergence, and compactness are all properties of the pair $(X, \tau)$, not of the bare set $X$.

</div>