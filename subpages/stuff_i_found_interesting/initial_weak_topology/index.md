# On initial topologies, weak topologies and compactness tension

## Setup and convention

<div class="math-callout math-callout--info" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Motivation</span><span class="math-callout__name">(Setup)</span></p>

Suppose one is given a bare set $X$, a family $(Y_i)\_{i\in I}$ of [topological spaces](https://en.wikipedia.org/wiki/Topological_space), and a family of maps $f_i\colon X\to Y_i$. There is no topology on $X$ yet; the question is what topology it ought to carry, given that the *only* structure we care to see on $X$ is the structure transmitted through the $f_i$. The [**initial topology**](https://en.wikipedia.org/wiki/Initial_topology) is the canonical answer, and the point of these notes is that almost everything about it follows from taking one word — *coarsest* — seriously.

</div>

<div class="math-callout math-callout--info" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Convention</span><span class="math-callout__name">(Open sets)</span></p>

Throughout, "open" without qualification means open in whichever $Y_i$ is under discussion; a topology on $X$ is identified with its collection of open sets, and topologies are compared by inclusion (coarser $=$ fewer open sets). No separation axioms are assumed anywhere.

</div>

## Driving Idea

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Initial topology)</span></p>

There is a tension between two demands. Making each $f_i$ continuous wants *many* open sets on $X$: continuity of $f_i$ forces every preimage $f_i^{-1}(U)$ to be open, so the more maps and the finer the $Y_i$, the more opens we are obligated to include. On the other hand, continuity of a map *into* $X$ wants *few* open sets: with fewer opens on $X$ there are fewer preimages to check, so it is easier for an incoming map to be continuous. The initial topology is the equilibrium point of these two pressures —
 
**the coarsest topology on $X$ making every $f_i$ continuous.**
 
Everything below is a consequence of that one phrase.

</div>

TODO: add visualization of initial topology

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

## Construction and existence
 
The set of topologies on $X$ rendering all $f_i$ continuous is nonempty (the discrete topology works) and closed under arbitrary intersection (an intersection of topologies is a topology, and a map continuous for each member is continuous for the intersection). Hence the intersection of the entire family is itself such a topology, and it is by construction the coarsest one. This is the initial topology $\tau$; it **exists** and is **unique**.

<div class="math-callout math-callout--theorem" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Initial topology construction)</span></p>

Unwinding what the construction demands gives a concrete description. Continuity of $f_i$ forces each $f_i^{-1}(U)$ into $\tau$, and conversely the topology generated by exactly these sets already makes every $f_i$ continuous, so $\tau$ is generated by the subbasis
 
$$
\mathcal S \;=\; \lbrace \, f_i^{-1}(U) : i\in I,\ U\subseteq Y_i\ \text{open} \,\rbrace.
$$
 
Open sets of $X$ are thus the arbitrary unions of finite intersections of pullbacks. One rarely needs more than this description, but one almost never needs even this much, because of the next point.

</div>

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Subbasis and topology generated by it)</span></p>

A **subbasis** for a topology on $X$ is *any* collection $\mathcal{S} \subseteq \mathcal{P}(X)$ of subsets of $X$ — no closure conditions are required. The **topology generated by $\mathcal{S}$** is then the family

$$
\tau(\mathcal{S}) = \Bigl\lbrace\, \bigcup_{\lambda \in \Lambda} (S_{\lambda,1} \cap \dots \cap S_{\lambda,n_\lambda}) \;:\; S_{\lambda,k} \in \mathcal{S} \,\Bigr\rbrace,
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

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Property</span><span class="math-callout__name">(Mapping-in property)</span></p>

he reason to care about $\tau$ is not the subbasis but the following characterization, which holds for the initial topology and for no other topology on $X$.
 
**Mapping-in property.** For every topological space $Z$ and every map $g\colon Z\to X$,

$$
g\ \text{is continuous} \iff f_i\circ g\ \text{is continuous for every } i\in I.
$$
 
The forward implication is immediate (a composite of continuous maps is continuous). For the converse it suffices to test preimages of subbasic sets, and

$$
g^{-1}\!\big(f_i^{-1}(U)\big) = (f_i\circ g)^{-1}(U),
$$

which is open precisely because each $f_i\circ g$ is continuous. The content of the property is the trade it offers: **a single, possibly hard, continuity question about $g$ is replaced by a family of easy ones**, one per index. There is, the property asserts, nothing more to the continuity of a map into $X$ than the continuity of each of its readings $f_i\circ g$.

</div>

TODO: add visualization of the mapping-in property

<div class="math-callout math-callout--proposition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Net/convergence form)</span></p>

**Net/convergence form.** Dually, $x_\alpha \to x$ in the initial topology $\iff f_i(x_\alpha)\to f_i(x)$ for all $i$. Convergence in the initial topology is convergence "coordinatewise against the test maps."

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The trade)</span></p>

Instead of staring at the (potentially mysterious) topology on $X$ and verifying that $g^{-1}(U)$ is open for every open $U\subseteq X$, you only need to check that each $f_i\circ g: Z\to Y_i$ is continuous as a map into the simpler space $Y_i$.

**The trade.** "A single hard question for a family of easy ones" unpacks as:

* the *hard* question is **"is $g \colon Z \to X$ continuous?"** — hard because the topology on $X$ may have no usable concrete description (e.g. the weak topology on a Banach space has *no* metric, the product topology on $\mathbb{R}^\mathbb{R}$ has uncountably many "coordinate constraints" and no basis you can write down compactly);
* the *easy* questions are **"is $f_i \circ g \colon Z \to Y_i$ continuous?"** for each $i$ — easy because each $Y_i$ is, by assumption, a space you already understand (often $\mathbb{R}$ or $\mathbb{C}$).

The universal property turns the *one* opaque check on $X$ into the *family* of transparent checks on the $Y_i$, **without losing information** — the two are logically equivalent. That equivalence is what the construction of $\tau$ buys you, and it is the entire reason to bother with the initial topology in the first place.

</div>

<div class="math-callout math-callout--remark" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why maps *in*, and not out.)</span></p>

**Why maps *in*, and not out.** This often puzzles on first contact, so it is worth saying plainly. The $f_i$ point *out* of $X$, so the only composition they permit is *post*-composition with a map $g\colon Z\to X$ entering $X$; a map $h\colon X\to W$ leaving $X$ cannot be composed with $f_i$ at all, both starting at $X$. The data is therefore mute about maps out and maximally informative about maps in — and the choice of *coarsest* is exactly what makes "maps in" the favored direction, since coarsening $X$ eases incoming continuity and obstructs outgoing continuity. (Choosing *finest* for a family pointing the other way gives the **final topology**, characterized dually by maps out: quotient and coproduct topologies live there.)

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>In the initial topology, we checking over the family of functions f_i and their topological spaces Y_i are easier than checking for only one topological space X?</summary>

**The premise to correct first.** The question contains a hidden miscount. It looks like we are comparing *one* check (the map $g\colon Z\to X$) against *many* checks (the maps $f_i\circ g$), and surely one is fewer than many. But "check $g\colon Z\to X$ for continuity" is not one check — it is the assertion that $g^{-1}(W)$ is open for *every* open $W\subseteq X$, and the open sets of $X$ are an enormous, abstractly-generated collection. The family formulation does not add work; it *reorganizes the same work* into pieces that land in spaces you actually understand. Let me make both halves of that precise.

**The topology on $X$ is defined, not given — so the family is the only handle you have.** This is the crux. In the initial-topology setup $X$ began as a bare set; its topology is *manufactured* as the coarsest one making the $f_i$ continuous, i.e. generated by the subbasis $\lbrace f_i^{-1}(U)\rbrace$. A general open set of $X$ is therefore

$$
W=\bigcup_{\alpha}\ \bigcap_{k=1}^{n_\alpha} f_{i_{\alpha k}}^{-1}(U_{\alpha k}),
$$

an arbitrary union of finite intersections of pullbacks. There is no *independent* description of "the open sets of $X$" to test against — they have no existence prior to the family. So checking $g$ directly is not checking against some clean given object; it is quantifying over this entire generated lattice. The $Y_i$, by contrast, are honest pre-existing spaces with topologies you know concretely. The asymmetry is **epistemic**: $X$ is the unknown defined *through* the $f_i$, while the $Y_i$ are the known.

**Why the family check is literally the same check, collapsed to a subbasis.** Continuity need only be verified on a subbasis, because preimage commutes with the two generating operations. Concretely, using $g^{-1}\big(f_i^{-1}(U)\big)=(f_i\circ g)^{-1}(U)$ and the fact that $g^{-1}$ distributes over unions and intersections,

$$
g^{-1}(W)=\bigcup_\alpha\ \bigcap_{k}\ g^{-1}\big(f_{i_{\alpha k}}^{-1}(U_{\alpha k})\big)=\bigcup_\alpha\ \bigcap_k\ (f_{i_{\alpha k}}\circ g)^{-1}(U_{\alpha k}).
$$

If each $f_i\circ g$ is continuous, every $(f_i\circ g)^{-1}(U)$ is open in $Z$; finite intersections and arbitrary unions of opens are open; hence $g^{-1}(W)$ is open for *every* $W$. So verifying the family is not a different, larger task — it is exactly the subbasis test for the single map $g$, and the subbasis of $X$ *is* indexed by the family. The many-versus-one framing dissolves: the subbasis of $X$ has one generator per pair $(i,U)$, so "test $g$ on the subbasis" and "test each $f_i\circ g$" are the same sentence.

**Why each piece is then genuinely easier.** Three things have happened in the reorganization:

1. *You never enumerate the opens of $X$.* The hard, entangled part of $X$ — how pullbacks from different indices intersect to form its opens — is bypassed entirely; you only ever look at preimages under the individual, known $f_i$. The joint structure handles itself via the displayed distribution law.

2. *The targets are concrete and usually simpler.* Each $f_i\circ g$ is a map into a space $Y_i$ you understand and have tools for — often far simpler than $X$ itself. You have traded one map into an abstract codomain for several maps into familiar ones.

3. *The subproblems are decoupled.* Each index is a separate, self-contained question with no cross-talk. You handle one coordinate at a time and never confront $X$ as a single combined object.

**The payoff made vivid (weak topology).** Take $X=E$ with the weak topology, the family being all $\varphi\in E^*$ into the scalar field. The universal property says $g\colon Z\to(E,\text{weak})$ is continuous iff every $\varphi\circ g\colon Z\to\mathbb K$ is. The left side asks about continuity into a possibly monstrous infinite-dimensional space whose weak-open sets you could never list; the right side asks only that a family of *ordinary numerical functions* be continuous — a question you attack with $\varepsilon$–$\delta$, explicit formulas, all of elementary analysis. That is the whole trade in one line: continuity into the unknowable space is reduced to continuity of scalar functions in your computational comfort zone.

**Remark — what is really going on.** The reduction is not "many is cheaper than one"; it is "*known* is cheaper than *defined*." Because $X$'s topology is constructed from the family, the family is not an alternative route to $X$ — it is the *only* route, and the universal property is just the statement that this route is faithful: nothing about continuity into $X$ is invisible to the test maps. The same principle, read in the mirror, is why a *final* topology (quotient, coproduct) reverses the convenience — there $X$'s opens are defined by which preimages *out* are open, so maps *out* of $X$ become the cheap ones to check, and maps in the expensive ones.

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>So, X is essentially bypassed and the problem reduced on the verifying of open sets of Y_i instead of open sets of X (verifying I mean verifying that their preimages are in opens sets of Z)?</summary>

**The core is correct.** You never touch the open sets of $X$. The whole task becomes: for each index $i$, take the open sets of $Y_i$, pull them back to $Z$, and verify those preimages are open in $Z$. $X$ is bypassed as an object to be enumerated.

**Refinement 1 — pin down which map does the pulling back.** The map whose continuity you're checking is $g\colon Z\to X$, and you test it through the composite $f_i\circ g\colon Z\to Y_i$. So for an open $U\subseteq Y_i$ the relevant preimage is

$$
(f_i\circ g)^{-1}(U)\ \subseteq Z,
$$

and you check *that* is open in $Z$. The two-step path $Z \xrightarrow{g} X \xrightarrow{f_i} Y_i$ collapses into the single arrow $Z\to Y_i$; the middle stop at $X$ never has to be inspected. (Equivalently $(f_i\circ g)^{-1}(U)=g^{-1}\big(f_i^{-1}(U)\big)$, which is the same set seen as "$g$-preimage of a subbasic open of $X$" — but you compute it without ever listing $X$'s opens.)

**Refinement 2 — you don't even need all opens of $Y_i$, just a subbasis of $Y_i$.** Continuity is preserved under unions and intersections, so it suffices to check the preimages of a generating family for $Y_i$. So the task shrinks twice: from "all opens of $X$" down to "all opens of $Y_i$," and then down again to "a subbasis of each $Y_i$." For the weak topology, $Y_i=\mathbb K$ and a subbasis is just open intervals/disks — so you end up checking continuity of scalar functions against elementary open sets, which is as concrete as it gets.

**One caveat on the word "bypassed."** This holds in the context you're in — checking continuity of a map *into* $X$. The construction of $X$'s topology *itself* is still defined by the family (the subbasis $\lbrace f_i^{-1}(U)\rbrace$ literally *is* $X$'s topology). So $X$ isn't bypassed in the sense of "irrelevant"; it's bypassed in the sense of "never separately enumerated." Its opens are exactly the pullbacks, so working with $Y_i$-opens *is* working with $X$-opens, just expressed in the coordinates where you can actually compute.

So your sentence is right: the problem is relocated from the open sets of $X$ (abstract, generated, unlistable) to the open sets — really a subbasis — of the known spaces $Y_i$, pulled back into $Z$ and tested there.

</details>
</div>

## Why we need it: the examples are already initial topologies

The abstraction earns its keep because the topologies you already use *are* initial topologies, and recognizing them as such immediately gives you their good behavior for free.

**Subspace topology.** $A\subseteq X$ with the single [inclusion](https://en.wikipedia.org/wiki/Inclusion_map) $\iota\colon A\hookrightarrow X$. The initial topology is the [subspace topology](https://en.wikipedia.org/wiki/Subspace_topology), and the universal property is the familiar "a map into $A$ is continuous iff it is continuous as a map into $X$."

**Product topology.** $\prod_i Y_i$ with the projections $\pi_i$. The initial topology is the product topology — and this is the honest reason the product topology, not the box topology, is the "correct" one. The box topology makes each $\pi_i$ continuous too, but it is strictly finer, hence fails the coarseness that yields the universal property: with the box topology a map into the product need *not* be continuous merely because all its components are. The product topology is engineered precisely so that "continuous into the product" $=$ "continuous in each coordinate." (This is also the structural origin of Tychonoff's theorem behaving well.)

**Weak topology on a normed space $E$.** Take $I = E^\ast$ (the continuous dual) and the maps to be the functionals $\varphi\colon E\to \mathbb{K}$ themselves. The initial topology is the weak topology $\sigma(E,E^\ast)$ — by construction the coarsest topology keeping every continuous functional continuous. The convergence form says exactly: $x_\alpha \rightharpoonup x$ iff $\varphi(x_\alpha)\to\varphi(x)$ for all $\varphi\in E^\ast$, which is the *definition* of weak convergence you already know. The universal property is why weak continuity is so often checkable.

**Weak-\* topology on a dual $E^\ast$.** Now put $I = E$ and use the evaluation maps $\widehat{x}\colon E^\ast\to\mathbb{K}$, $\varphi\mapsto \varphi(x)$. The initial topology is $\sigma(E^\ast,E)$. This is the topology of pointwise convergence on $E^\ast$, and its coarseness is what makes the unit ball compact (Banach–Alaoglu) — compactness is *easy* in a coarse topology, so designing the weak-\* topology to be as coarse as possible while still separating points is exactly the right move.

In each case the *same* two-line argument (subbasis = pullbacks; continuity-in iff continuity-coordinatewise) does all the work, which is the economy the concept buys you.

## Remarks

1. **The categorical reading.** The initial topology is a limit. Subspace = equalizer-flavored, product = product, the general construction = the limit of the diagram of forgetful constraints. The mapping-in universal property is just the limit's universal property, which is why "initial" topologies are tested by maps *in*. The mirror notion is the **final topology** (the *finest* topology making a family $g_i\colon Y_i\to X$ continuous): it is a colimit, characterized by maps *out* ($h\colon X\to Z$ continuous iff each $h\circ g_i$ is), and it specializes to the quotient topology, the disjoint-union/coproduct topology, and direct limits. Subspace/product live on the initial side; quotient/coproduct on the final side. The duality is exact.

2. **Coarseness is a feature, not a defect.** A recurring theme in functional analysis — Banach–Alaoglu, the Krein–Milman setting, weak compactness via [Eberlein–Šmulian](https://en.wikipedia.org/wiki/Eberlein–Šmulian_theorem) — is that you *want* the weakest topology that still does a required job (here: keep certain maps continuous, separate points), because compactness, lower semicontinuity, and existence-of-extremizer arguments all get cheaper as the topology coarsens. The initial topology is the formal device that hands you "as coarse as possible, subject to constraints." That is the motivation in one sentence.

# Weak Topology

## Setup and convention

<div class="math-callout math-callout--info" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Motivation</span><span class="math-callout__name">(Setup)</span></p>

Let $E$ be a normed space over $\mathbb K\in\lbrace\mathbb R,\mathbb C\rbrace$, with continuous dual $E^\ast$. It already carries a topology — the one induced by the norm, which I will call *strong*. The weak topology is a deliberately *coarser* topology placed alongside the strong one, and the cleanest way to understand it is in one sentence: it is what the initial-topology construction produces when the family of test maps is the space's own dual. Everything distinctive about it comes not from that construction — which it shares with subspaces and products — but from the *linear input* one feeds in.

</div>
 
<div class="math-callout math-callout--info" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Convention</span><span class="math-callout__name">(Open sets)</span></p>

$E$ is infinite-dimensional unless stated otherwise (in finite dimensions there is nothing to say: weak and strong coincide). "Strong" $=$ norm topology; $\sigma(E,E^\ast)$ denotes the weak topology and $\sigma(E^\ast,E)$ the weak-$\ast$ topology. Topologies are compared by inclusion, coarser meaning fewer open sets. $B_E=\lbrace x:\|x\|\le 1\rbrace$.

</div>

<div class="math-callout math-callout--info" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Idea</span><span class="math-callout__name">(The driving idea)</span></p>

In infinite dimensions the norm topology is too fine to be useful for existence arguments: by Riesz's theorem the closed unit ball is norm-compact *iff* $\dim E<\infty$. Compactness is the engine of "a [minimizing sequence](https://en.wikipedia.org/wiki/Direct_method_in_the_calculus_of_variations) has a convergent subsequence," so losing it is fatal. The cure is to coarsen the topology — fewer open sets means more compact sets — but you cannot coarsen recklessly, or you lose the ability to see linear structure. The weak topology is the principled coarsening: **the coarsest topology on $E$ that still keeps every continuous functional continuous.** That single sentence is simultaneously the definition, the motivation, and the link to the previous discussion.

</div>

## Definition

<div class="math-callout math-callout--definition" markdown="1">
<p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Weak topology, Weak convergence)</span></p>

The **weak topology** $\sigma(E,E^\ast)$ is the **initial topology induced on $E$ by the family $E^\ast$**, i.e. by all $\varphi\colon E\to\mathbb K$ that are norm-continuous. It therefore has subbasis $\lbrace\varphi^{-1}(U):\varphi\in E^\ast,\ U\subseteq\mathbb K\ \text{open}\rbrace$. Because the test maps are *linear* into a topological vector space, this is not merely a topology but a locally convex vector topology, generated by the seminorms

$$p_\varphi(x)=|\varphi(x)|,\qquad \varphi\in E^*,$$

with a base of neighborhoods of $0$ given by the convex sets $\lbrace x:\|\varphi_1(x)\|<\varepsilon,\dots,\|\varphi_n(x)\|<\varepsilon\rbrace$ for finite $\lbrace\varphi_k\rbrace\subset E^\ast$. The convergence form is the statement one usually takes as the *definition* of **weak convergence**:

$$
x_\alpha\rightharpoonup x \iff \varphi(x_\alpha)\to\varphi(x)\ \text{ for every } \varphi\in E^*,
$$

which is exactly the initial topology's "convergence coordinatewise against the test maps," read against the functionals.

</div>

## Why we need it — motivation made concrete

**Recovering compactness.** This is the headline. Coarsening to $\sigma(E,E^\ast)$ buys back compactness lost in infinite dimensions: in a [reflexive space](https://en.wikipedia.org/wiki/Reflexive_space) the closed unit ball is weakly compact (Kakutani), and for duals the unit ball is weak-\* compact unconditionally ([Banach–Alaoglu theorem](https://en.wikipedia.org/wiki/Banach–Alaoglu_theorem)). Compactness is cheap in a coarse topology, so the whole design is "make it as coarse as possible while still separating points." The runaway almost-orthogonal sequence that broke norm-compactness is reined back in: in a Hilbert space an orthonormal sequence has $e_n\rightharpoonup 0$, since $\langle e_n,y\rangle\to0$ for each fixed $y$, even though $\|e_n-e_m\|=\sqrt2$. Coarsening has blurred the infinitely many "independent directions" until the sequence converges.

**The direct method.** Pair compactness with the fact that the norm — and, by Mazur's theorem, any convex norm-closed set — is *weakly* lower semicontinuous. Then a coercive convex functional on a reflexive space attains its minimum: extract a bounded [minimizing sequence]((https://en.wikipedia.org/wiki/Direct_method_in_the_calculus_of_variations)), pass to a weakly convergent subnet by compactness, and lower semicontinuity finishes. This is the backbone of the calculus of variations, PDE existence theory, and optimization. The weak topology exists so this argument runs.

**The natural notion of limit.** In many settings weak convergence, not norm convergence, is the right object — oscillations that don't decay in norm still converge weakly (e.g. $\sin(nx)\rightharpoonup 0$ in $L^2$), which is precisely the phenomenon you want to track.

## How it relates to the initial topology

There is nothing to prove here, only to observe: **the weak topology *is* an initial topology**, the special case where the family of maps is the continuous dual $E^\ast$ and every target is the scalar field $\mathbb K$. Every general fact specializes for free:

- the **mapping-in property** — $g\colon Z\to(E,\text{weak})$ is continuous iff $\varphi\circ g$ is continuous for all $\varphi\in E^\ast$;
- the **convergence form** above is the coordinatewise-convergence statement of the initial topology, read against the test maps $\varphi$;
- the subbasis is the standard "pullbacks of opens" subbasis.

Its sibling, the **weak-\* topology** $\sigma(E^\ast,E)$ on the dual, is the **same** construction with the roles swapped: the initial topology on $E^\ast$ induced by the evaluation maps $\widehat x\colon\varphi\mapsto\varphi(x)$, $x\in E$. Both are initial topologies; they differ only in which family of functionals you feed the machine.

**Weak vs. strong.** Because $\sigma(E,E^\ast)$ is the **coarsest** topology making the (norm-continuous) functionals continuous, it is contained in the norm topology: $\sigma(E,E^\ast)\subseteq\text{strong}$, with **equality if and only if $\dim E<\infty$.** Hence every weakly open (closed) set is strongly open (closed), but not conversely. The standard cautionary example: the unit sphere is weakly dense in the closed ball, so norm-closedness is strictly stronger than weak-closedness. The reconciliation for **convex** sets is Mazur's theorem — a convex set is weakly closed iff it is norm-closed — which is what makes the weak topology usable in convex analysis.

## Similarities and differences

**Similarities** are exactly the inherited structure: same defining property (coarsest making the family continuous), same universal property (test continuity one functional at a time), same subbasis, same convergence description. If you understand initial topologies, you understand the *formal* behavior of the weak topology at no extra cost.

The **differences** are where the weak topology earns its own name. They all stem from the family being not an arbitrary set of maps but the *linear dual* of a *vector space*:

1. **Linear structure is automatic.** Because the maps are linear and the targets are TVSs, the weak topology is forced to be a *locally convex vector topology* — generated by seminorms, with convex basic neighborhoods. A generic initial topology has none of this: arbitrary targets and arbitrary maps give no seminorms, no convexity, no homogeneity. The weak topology lives in the category of locally convex spaces, not merely in $\mathbf{Top}$.

2. **Separation is a theorem, not a hypothesis.** A general initial topology is Hausdorff iff the family separates points — a near-tautology you simply check. For the weak topology, the statement "$E^\ast$ separates the points of $E$" is *Hahn–Banach*. So the Hausdorffness of $\sigma(E,E^\ast)$ rests on a genuine analytic theorem rather than a definitional checkbox.

3. **The family is canonical, not chosen.** With an initial topology you supply whatever maps you please, and different choices give different topologies — the family is extra data. The weak topology's family is determined by $E$ itself (the *entire* continuous dual), so it is an intrinsic invariant of the normed space, not an additional input.

4. **There is a pre-existing topology to compete with.** A bare set carrying an initial topology has no rival structure; there is nothing to compare against. But $E$ already has its norm topology, and the entire point is the comparison: $\sigma(E,E^\ast)\subseteq\text{norm topology}$ always, with **equality iff $\dim E<\infty$**. The "weak vs. strong" drama — weakly closed $\Rightarrow$ norm-closed but not conversely (the unit sphere is weakly dense in the ball), convergence weaker than norm convergence — exists only because the construction is deliberately coarsening a topology that is already there.

5. **Failure of metrizability / sequences are not enough.** On an infinite-dimensional $E$ the weak topology is never metrizable and not first-countable, so you must reason with nets, and weak sequential closure can differ from weak closure. This is a consequence of $E^\ast$ being "large" — an uncountable separating family — rather than of the initial-topology construction itself (a *countable* initial family, e.g. a countable product of metric spaces, stays metrizable). It is the reason Eberlein–Šmulian (which rescues sequences for *compactness* specifically) is a nontrivial theorem rather than an observation.

## Remarks

1. **Where this really lives.** The honest general setting is a *dual pair* $\langle E,F\rangle$ of vector spaces with a separating bilinear form; $\sigma(E,F)$ is then the weak topology of the pairing, and weak/weak-\* are the two ends of the same construction. The [Mackey–Arens theorem](https://en.wikipedia.org/wiki/Mackey–Arens_theorem) says all locally convex topologies yielding the *same* dual $F$ sit between $\sigma(E,F)$ (coarsest) and the Mackey topology $\tau(E,F)$ (finest). In this light the weak topology is the *bottom* element of a whole lattice of "polar topologies" compatible with the duality — the maximally coarse choice, which is exactly the property the initial-topology construction guarantees.

2. **The one-sentence summary.** A weak topology is what you get when you point the initial-topology machine at a space's own dual; everything distinctive about it — local convexity, Hahn–Banach separation, the coarseness that resurrects compactness — comes not from the *construction* (which it shares with subspaces and products) but from the *linear input* you feed it.

# Compactness Tenstion in Infinte Dimensions

## Setup and convention

$E$ is an infinite-dimensional normed space, $B_E=\lbrace x:\|x\|\le 1\rbrace$ its closed unit ball. "Compact" means every open cover has a finite subcover; in a metric space this coincides with sequential compactness and with **complete + totally bounded**. The norm topology is *strong/fine*; the weak topology $\sigma(E,E^\ast)$ is *coarse*. The question has two halves — *why does compactness fail*, and *is there a tension* — and the answer to the second is yes, sharply, and it is the same tension as before with the roles relabelled. Let me take them in order.

**The driving idea.** Compactness is a **finiteness property**: it says the space is, in the covering sense, indistinguishable from a finite set. The unit ball in infinite dimensions is genuinely infinite — it contains infinitely many mutually separated points — so no finite cover can reach them all. You can only restore compactness by *blurring your vision* until those infinitely many points stop looking separated, and blurring means deleting open sets, i.e. coarsening. That single move — spend open sets to buy compactness — is the whole story, and it is what couples this discussion to the initial-topology one: in both, the **resource being traded is the supply of open sets.**

## Why infinite dimensions kills it

**Riesz's theorem.** $B_E$ is norm-compact iff $\dim E<\infty$. The mechanism is Riesz's "almost-orthogonal" lemma: in any infinite-dimensional normed space one can produce a sequence $(x_n)$ on the unit sphere with

$$\|x_n-x_m\|\ge \tfrac12\qquad(n\ne m).$$

Such a sequence has no Cauchy subsequence, hence no convergent subsequence; the ball is not sequentially compact, hence (it is metric) not compact. The cleanest model: in a Hilbert space take the orthonormal basis $(e_n)$, for which $\|e_n-e_m\|=\sqrt2$. These vectors march off to "infinitely many independent directions" and never settle.

**The metric diagnosis: total boundedness fails.** Since $E$ Banach is complete, completeness is *not* the obstruction — compactness $=$ complete $+$ totally bounded means the *only* thing that can go wrong is total boundedness, and it does. Covering $B_E$ by balls of radius $\varepsilon<\tfrac12$ would force each $x_n$ into a distinct ball, so you need infinitely many. The ball has **infinite metric entropy**: infinitely much independent information packed inside. *That* is the precise sense in which compactness is "expensive" here — it is not that the property is hard to define, it is that the object you want to apply it to is, measured in $\varepsilon$-balls, infinite. Finite dimensions are exactly the regime where bounded sets are totally bounded (Heine–Borel), so there the norm topology already sits at the sweet spot and nothing is owed.

## Is there again a tension between two desires?

Yes — and it is the cleanest possible echo of the initial-topology equilibrium. There the pull was **"make the $f_i$ continuous"** (wants *many* opens) against **"coarsest / easy maps-in"** (wants *few* opens). Here:

$$\underbrace{\textbf{compactness}}_{\text{wants FEW opens}} \quad\text{vs.}\quad \underbrace{\textbf{separation (Hausdorff)}}_{\text{wants MANY opens}}.$$

Compactness gets *easier* as you coarsen (fewer open covers to defeat). Telling points apart gets *easier* as you refine (more open sets to separate them with). They pull in opposite directions on the same lattice of topologies, and there is an exact equilibrium statement.

**The rigidity theorem — the equilibrium made precise.** Recall that a continuous bijection from a compact space to a Hausdorff space is automatically a homeomorphism. Feed it the identity map between two comparable topologies and you get: *if $(X,\tau)$ is compact Hausdorff, then every strictly coarser topology is no longer Hausdorff, and every strictly finer topology is no longer compact.* A compact Hausdorff topology is simultaneously **minimal among Hausdorff topologies and maximal among compact ones** — it sits on a knife's edge. Coarsen and you lose the ability to separate points; refine and you lose compactness. This is the same "equilibrium of two opposing monotone desiderata" that defined the initial topology, only now the two desiderata are compactness and separation rather than continuity-of-$f_i$ and ease-of-maps-in.

## The infinite-dimensional drama as a position on this axis

With that axis in hand, the whole weak-topology story is a single sentence: **the norm topology sits too far toward the fine end.** It has Hausdorff separation to spare — and pays for that surplus with a ball that is not compact. The remedy is to slide back toward the coarse end until the ball becomes compact, *without sliding so far that you stop separating points*. That landing spot is the weak topology: coarse enough to compactify the ball (in reflexive spaces; unconditionally for duals via Banach–Alaoglu), yet still Hausdorff because $E^\ast$ separates points — and that Hausdorffness is exactly the residue of "don't coarsen *past* separation," secured by Hahn–Banach.

**What recovers the runaway sequence.** The almost-orthogonal sequence that broke norm-compactness is *reined back in* by coarsening: $e_n\rightharpoonup 0$ weakly in a Hilbert space (since $\langle e_n,y\rangle\to0$ for each fixed $y$), even though $\|e_n-e_m\|=\sqrt2$. In the weak topology the infinitely many "independent directions" cease to be neighborhoods-apart, so the sequence converges and the ball can be compact. That is the blurring made literal.

**But the trade is not free — and here is the second price.** Coarsening to recover compactness costs you a *fineness*-flavored convenience: metrizability. On an infinite-dimensional $E$ the weak topology is never first-countable, hence never metrizable; you must reason with nets, and weak closure can exceed weak sequential closure. So the deepest version of the tension in infinite dimensions is

$$
\textbf{compactness of the ball}\ \text{(coarse)}\quad\text{vs.}\quad \textbf{metrizability / sequential adequacy}\ \text{(fine)},
$$

and Riesz says you cannot have both *at full strength*: the metrizable (norm) topology has a non-compact ball, the compact (weak) topology is non-metrizable. You buy compactness in the currency of resolution. **Sharpness / partial reconciliation:** the conflict relaxes on the ball alone for separable spaces — $(B_E,\text{weak})$ is metrizable iff $E^*$ is separable, and $(B_{E^\ast},w^\ast)$ is metrizable iff $E$ is separable. So on a separable predual the weak-\* ball is *both* compact (Alaoglu) *and* metrizable: the best of both worlds, but only locally (on the ball) and only under separability. The global topology stays non-metrizable.

## Remarks

1. **The unifying ledger.** Across this whole conversation the single conserved quantity has been *open sets*. The initial topology spends the minimum opens needed to make a family continuous (coarsest), optimizing maps in. Compactness is bought by spending opens (coarsening). Separation and metrizability are bought by hoarding opens (refining). Every "tension" you have noticed is two clients bidding for this one resource in opposite directions, and every named topology — initial, weak, weak-\*, even compact–Hausdorff — is the market-clearing price for some pair of demands. Recognizing the resource is recognizing why these subjects rhyme.

2. **Why Tychonoff is the hero of Alaoglu.** Compactness fails for $B_E$ in the metric world because total boundedness fails, and you cannot fix total boundedness. So Banach–Alaoglu abandons the metric route entirely: embed $B_{E^\ast}\hookrightarrow\prod_{x\in B_E}\lbrace\|\lambda\|\le1\rbrace$ by $\varphi\mapsto(\varphi(x))\_x$, observe the image is closed, and invoke Tychonoff (an arbitrary product of compacts is compact). The weak-\* topology is *defined* to be the subspace topology of this product — i.e. it is the initial topology for the evaluation maps — precisely so that compactness arrives for free from the product. This is the payoff of the earlier observation that the weak-\* topology is an initial topology: its coarseness is not a defect to be tolerated but the exact engineering that lets compactness through the door.


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