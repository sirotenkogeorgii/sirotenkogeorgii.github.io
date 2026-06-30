* [Math's Strangest Set](https://www.youtube.com/watch?v=hs3eDa3_DzU)

---
title: "A set you cannot measure: the Vitali construction"
date: 2026-06-30
tags: [measure-theory, real-analysis, axiom-of-choice]
---

# A set you cannot measure: the Vitali construction

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Conventions and Normalisations</span><span class="math-callout__name"></span></p>

**Conventions and normalisations.** Throughout, $m$ denotes Lebesgue measure on the
real line, and $m^\ast$ its outer measure. We will repeatedly use the two structural
properties that make Lebesgue measure what it is: it is *translation invariant*,
$m(E+x)=m(E)$, and *countably additive*, $m\big(\bigsqcup_n E_n\big)=\sum_n m(E_n)$ on
disjoint measurable families. It is convenient to do all of the work on the circle

$$
\mathbb{T} \;=\; \mathbb{R}/\mathbb{Z},
$$

which we identify as a set with the half-open interval $[0,1)$ and equip with the group
operation $\oplus$ of *addition modulo $1$*. Lebesgue measure descends to a measure on
$\mathbb{T}$ — it is simply TODO: the Haar probability measure of TODO: the compact group $\mathbb{T}$,
and as such it is invariant under every rotation $x\mapsto x\oplus t$. (Concretely:
$E\oplus t$ is the disjoint union of $(E+t)\cap[0,1)$ and $(E+t-1)\cap[0,1)$, two ordinary
translates, so rotation invariance on $\mathbb{T}$ is just translation invariance on
$\mathbb{R}$ together with additivity.) We write $\mathbb{Q}\_1 := \mathbb{Q}\cap[0,1)$;
this is a *countably infinite subgroup* of $(\mathbb{T},\oplus)$.

</div>

We will prove the following, which is the real content of the construction.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Vitali, 1905)</span></p>

There is no measure $\mu$ defined on the full power set $2^{\mathbb{T}}$ that is simultaneously countably additive, rotation invariant, and normalised so that $\mu(\mathbb{T})=1$. Equivalently: there exists a subset of $[0,1)$ that is not Lebesgue measurable.

</div>

So the comfortable dream of assigning a consistent "length" to *every* subset of the
line is impossible. TODO: Lebesgue measure is forced to live on a proper sub-$\sigma$-algebra,
and the Carathéodory machinery that singles out the measurable sets is not a technical
nicety we could dispense with on a better day — there is genuinely something out there
it must exclude.

## The key idea

Before any construction, here is the whole argument in one breath. The rationals
$\mathbb{Q}\_1$ form a *countable, dense* subgroup of the circle. Pick one representative
from each TODO: coset of this subgroup — TODO: a *transversal* $V$. The cosets are exactly the
rational translates of one another, so the countably many rotated copies
$\lbrace V\oplus q : q\in\mathbb{Q}\_1\rbrace$ are pairwise disjoint and together tile the whole
circle. Now run the two defining properties of measure against each other: rotation
invariance says all the copies have the *same* measure $c$, while countable additivity
says these equal measures must sum to $m(\mathbb{T})=1$. A constant summed over a
countably infinite index set is either $0$ or $+\infty$, never $1$. The contradiction is
the theorem. Everything below is bookkeeping around this collision.

The one genuinely non-constructive ingredient is the choice of transversal; we will see
at the end that it cannot be avoided.

## Step 1: An equivalence relation from the rationals

Define a relation on $\mathbb{T}$ by

$$
x \sim y \quad\Longleftrightarrow\quad x \ominus y \in \mathbb{Q}_1,
$$

i.e. $x$ and $y$ differ by a rational. **This is an equivalence relation.** Reflexivity
is $x\ominus x = 0\in\mathbb{Q}\_1$; symmetry uses that $\mathbb{Q}\_1$ is closed under
inversion in $\mathbb{T}$; transitivity uses that it is closed under $\oplus$. The
equivalence classes are precisely the cosets

$$
[x] \;=\; x\oplus \mathbb{Q}_1 \;=\;\lbrace \,x\oplus q : q\in\mathbb{Q}_1\,\rbrace.
$$

Two features of these classes matter. Each is *countable*, being the image of the
countable set $\mathbb{Q}\_1$. And each is *dense* in $\mathbb{T}$, since the rationals are
dense. So the partition of $[0,1)$ into classes is a partition into countably-infinite,
densely-interleaved pieces — already a sign that no single class can be "small in a
length sense" without all of them being so.

## Step 2: A transversal, via the axiom of choice

Each equivalence class is a nonempty subset of $\mathbb{T}$. Invoking the **axiom of
choice**, let $V\subseteq\mathbb{T}$ be a set containing *exactly one* element from each
class. Such a $V$ is a *Vitali set*. It is the simultaneous embodiment of "one point per
orbit" of the $\mathbb{Q}\_1$-action, and it is the only place the construction reaches
outside of explicit, definable mathematics. There is nothing canonical about $V$:
different choice functions give wildly different sets, and not one of them admits a
formula.

The defining property we will use is precisely the *one representative per class*
condition, which we restate operationally:

$$
v,v'\in V \ \text{ and }\ v\ominus v'\in\mathbb{Q}_1 \;\Longrightarrow\; v=v'. \tag{$\dagger$}
$$

## Step 3: The rational translates partition the circle

For each $q\in\mathbb{Q}\_1$ form the rotated copy

$$
V_q \;:=\; V\oplus q \;=\;\lbrace \,v\oplus q : v\in V\,\rbrace.
$$

**The copies are pairwise disjoint.** Suppose $x\in V_q\cap V_{q'}$. Then
$x=v\oplus q=v'\oplus q'$ for some $v,v'\in V$, whence
$v\ominus v' = q'\ominus q\in\mathbb{Q}\_1$. By $(\dagger)$ this forces $v=v'$, and then
$q=q'$. So distinct rationals give disjoint copies.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(A word on "distinct rationals)</span></p>

It is worth being precise about what the index set is, since this is a natural place to worry. The union below ranges over $\mathbb{Q}\_1=\mathbb{Q}\cap[0,1)$, a set of rational *numbers* — not a set of fractions as written. The expressions $\tfrac{n}{m}$ and $\tfrac{an}{am}$ are not two indices that happen to collide; they *are* the same element of $\mathbb{Q}\_1$, and the set notation has already deduplicated every redundant representation for free. So $V_{1/2}=V_{2/4}$ is not a breach of disjointness — it is one term written twice, counted once. Disjointness is only ever asserted between distinct rational numbers, which is exactly what the argument above delivers. In fact one can say slightly more: distinct rationals give not merely disjoint but genuinely *different* copies. If $V_q=V_{q'}$ with $r:=q'\ominus q\neq 0$, then $V=V\oplus r$, so $v\oplus r\in V$ for every $v\in V$; by $(\dagger)$ this forces $r=0$. Thus a Vitali set has *no* nonzero rational period — every rational rotation carries it to a fresh copy, and no accidental overlap can occur. For the counting argument in Step 4 the only thing that matters is that $\mathbb{Q}\_1$ is countably infinite *as a set of numbers*; the (also countable, but irrelevant) abundance of fraction representations plays no role.

</div>

**The copies cover the circle.** Let $x\in\mathbb{T}$ be arbitrary. Its class $[x]$
contains the unique representative $v\in V$, so $x\sim v$, i.e. $x\ominus v = q$ for some
$q\in\mathbb{Q}\_1$. Rearranging, $x = v\oplus q\in V_q$. Hence every point lies in some
copy.

Combining the two halves, we have an honest partition of the circle into countably many
rotated copies of a single set:

$$
\mathbb{T} \;=\; \bigsqcup_{q\in\mathbb{Q}_1} V_q. \tag{$\ast$}
$$

This is the entire geometric content of the construction. One set, rotated along a
countable dense subgroup, tiles the circle with no overlaps and no gaps.

## Step 4: The contradiction

Suppose, for contradiction, that $V$ is measurable, with $m(V)=c$ for some
$c\in[0,1]$. Rotation invariance gives $m(V_q)=m(V)=c$ for *every* $q$. Apply countable
additivity to the partition $(\ast)$, indexed by the countably infinite set
$\mathbb{Q}\_1$:

$$
1 \;=\; m(\mathbb{T}) \;=\; m\!\left(\bigsqcup_{q\in\mathbb{Q}_1} V_q\right)
\;=\; \sum_{q\in\mathbb{Q}_1} m(V_q) \;=\; \sum_{q\in\mathbb{Q}_1} c.
$$

Now examine the right-hand side, a sum of the constant $c$ over a countably infinite
index set. If $c=0$, the sum is $0$. If $c>0$, the sum is $+\infty$. In neither case is it
equal to $1$. This contradiction shows that **$V$ is not Lebesgue measurable**, and with
it that no measure on $2^{\mathbb{T}}$ can satisfy all three of countable additivity,
rotation invariance, and normalisation. $\blacksquare$

Notice exactly *where* the argument bites. Measurability of $V$ was used only to invoke
countable additivity; rotation invariance and the partition did the rest. The
incompatibility is among the *three* properties jointly — drop any one and the obstruction
evaporates. That observation is the doorway to everything in the next section.

**The interval version, for readers wary of $\mathbb{T}$.** If one prefers to avoid the
modular arithmetic, take $V\subseteq[0,1]$ as before but enumerate the rationals of
$[-1,1]$ as $q_0,q_1,\dots$ and use *ordinary* translation $V_k = V+q_k$. The same two
arguments show the $V_k$ are disjoint, while now
$[0,1]\subseteq\bigcup_k V_k\subseteq[-1,2]$. If $m(V)=c$, monotonicity and countable
additivity squeeze $1\le \sum_k c\le 3$, and a constant sum lying strictly between two
finite positive bounds is again impossible. The circle merely launders the bookkeeping:
it turns the sandwich $1\le\sum c\le 3$ into the cleaner equality $\sum c=1$.

## What is really going on

**The axiom of choice is not a stylistic flourish — it is necessary.** One might hope to
replace the choice function by something explicit. One cannot. Solovay (1970)
constructed a model of $\mathsf{ZF}$ together with the axiom of dependent choice in which
*every* subset of $\mathbb{R}$ is Lebesgue measurable. So the existence of a
non-measurable set is independent of the constructive part of set theory; some genuinely
non-constructive choice principle is required to produce a Vitali set. This is why no one
has ever "written one down": there is a theorem saying you can't.

**Finite additivity survives; only countable additivity dies.** The proof spent its
contradiction entirely on summing over a *countably infinite* family. If we ask only for
*finite* additivity, the obstruction disappears: Banach showed that there exist
finitely-additive, translation-invariant extensions of Lebesgue measure to *all* subsets
of $\mathbb{R}$ (and of $\mathbb{R}^2$). The Vitali set is invisible to such a measure
because the partition $(\ast)$ is infinite. The structural reason behind this dividing
line is *amenability*: the groups $\mathbb{Z}$, $\mathbb{R}$, $\mathbb{R}^2$ are amenable,
which is exactly what guarantees an invariant finitely-additive mean. The phenomenon is
genuinely about the interaction between the *size of the group* and the *strength of
additivity demanded*.

**Why three dimensions is worse.** From $\mathbb{R}^3$ onward even finite additivity under
*isometries* fails, and spectacularly: the Banach–Tarski paradox decomposes a solid ball
into finitely many pieces and reassembles them, by rigid motions, into two balls of the
original size. The mechanism is the same Vitali-style "tile a space by translates of a
transversal," but now the rotation group $SO(3)$ contains a free subgroup on two
generators — it is *non-amenable* — and a free group cannot carry an invariant mean. The
Vitali set is the one-dimensional, countably-additive shadow of this much larger
collapse; Banach–Tarski is what happens when the same idea is run on a non-amenable
group.

**A second, geometric proof via difference sets.** Here is an entirely different route to
non-measurability that illuminates *which* feature of $V$ is fatal. Steinhaus's theorem
says: if $A\subseteq\mathbb{R}$ is measurable with $m(A)>0$, then the difference set
$A-A=\lbrace a-a':a,a'\in A\rbrace$ contains an open interval about the origin. (One proof: by the
Lebesgue density theorem $A$ has a point of density $1$; near such a point $A$ and a small
translate of $A$ must overlap, producing the interval.) Now look at a Vitali set. By
$(\dagger)$, any two *distinct* points of $V$ differ by an irrational, so $V-V$ contains no
nonzero rational at all — and therefore contains no interval about $0$, since every such
interval is riddled with rationals. By the contrapositive of Steinhaus, $V$ cannot be
measurable with positive measure. But $V$ also cannot have measure zero: the copies
$(\ast)$ tile the circle, so $V$ has positive outer measure and a null set could not do
that. Either way $V$ is non-measurable. This proof never sums an infinite series; it
locates the pathology in the *arithmetic* of $V$ — its avoidance of rational differences —
rather than in a counting argument.

**The outer measure does not break, it only weakens.** Outer measure $m^\ast$ remains
perfectly well defined on $2^{\mathbb{T}}$, translation invariant, and *countably
subadditive*. What the Vitali set demonstrates is that the inequality
$m^\ast(\mathbb{T})\le\sum_q m^\ast(V_q)$ cannot be upgraded to equality on arbitrary
disjoint families: subadditivity is the most one can ask of a set function defined
everywhere. The Carathéodory criterion is precisely the device that carves out the
sub-$\sigma$-algebra on which subadditivity becomes additivity — and the Vitali set is the
canonical witness that this carving genuinely throws something away. A set $E$ is
Carathéodory measurable when it splits every test set $T$ additively,
$m^\ast(T)=m^\ast(T\cap E)+m^\ast(T\setminus E)$; a Vitali set is exactly a set that fails
this split, and the construction above is, in the end, a recipe for manufacturing such a
failure out of nothing but the group structure of the rationals.

**The pathology is sizelessness, not smallness.** It is natural to picture $V$ as built
from infinitely many vanishingly small pieces that accumulate to the visible mass of the
circle — the way countless invisible atoms make up a visible solid. This picture is
seductive and wrong, and seeing why sharpens the whole phenomenon. The copies $V_q$ are
not small; they are not large; they have *no size at all*, and sizelessness is a different
category from tininess. Suppose each piece did have a size, and suppose it were as small as
a size can be, namely $0$; then countable additivity would give $1=\sum_q 0 = 0$, an
immediate contradiction. So the resolution is emphatically not "each piece is
infinitesimal and infinitely many of them sum to $1$" — *any* common value, zero or
positive, already breaks the equal-size-plus-additivity constraint. The size of $V$ is not
small but *trapped*: its inner measure is $0$ (it contains no positive-measure measurable
set, by the difference-set argument above) while its outer measure is strictly positive
(the tiling $(\ast)$ forces it), and a genuine measure would have to be one number sitting
in both places at once. This is also why the atomic analogy points at the well-behaved
case: atoms have definite tiny sizes, and summing them into bulk matter is countable
additivity *working*. The same is true of the truly counterintuitive facts of real
analysis that the picture seems to describe — the rationals in $[0,1]$ covered by open
intervals of total length $\varepsilon$, a fat Cantor set that is nowhere dense yet of
positive measure, the Cantor set itself uncountable yet null. Each is "small in one sense,
large in another," and each is *fully measurable*. None is a Vitali phenomenon.
Non-measurability is a *symmetry obstruction*, not a smallness phenomenon: invariance wants
the copies equal, additivity wants their total finite, and over a countably infinite family
the two demands over-determine the size into having no admissible value — just as a uniform
distribution on the whole line has no admissible mean, not a tiny one.
 
**Could a better definition simply absorb it?** It is tempting to read all this as a
defect in the *definition* of measure — surely a cleverer notion of size would handle the
edge case. It would not, and the reason is that Vitali's result is best understood as a
*no-go theorem*, a kind of conservation law: one cannot have all four of {defined on every
subset, countably additive, translation invariant, total mass $1$} at once. "Handling the
Vitali set" therefore never means escaping the obstruction; it means choosing which of the
four to surrender, and each choice is a genuine, studied theory. Surrender *countable
additivity* and Banach's finitely-additive invariant extensions assign $V$ an honest
number — at the cost of every convergence theorem that makes integration worth doing.
Surrender *translation invariance* and the existence of a countably-additive measure on
all subsets becomes the classical measure problem, undecidable in $\mathsf{ZFC}$ and, by
Ulam, equivalent to a real-valued measurable cardinal — a large-cardinal axiom, and the
resulting measure non-canonical and (by Vitali) necessarily non-invariant. Surrender
*defined on every set* — the mainstream choice — and one keeps both additivity and
invariance, conceding only totality, which in practice costs nothing: every explicitly
constructible set is measurable. Surrender the *axiom of choice* and the set ceases to
exist at all: in Solovay's model of $\mathsf{ZF}+\mathsf{DC}$ (and under the Axiom of
Determinacy) every subset of $\mathbb{R}$ is Lebesgue measurable, bought with an
inaccessible cardinal and the loss of Hahn–Banach in full generality. The same axiom
furnishes both the pathology and much of the analysis we prize.
 
The upshot reframes the original worry. The Vitali set is not weird in itself, nor is the
definition too weak; weirdness here is *relational*. The set is unmeasurable only with
respect to the demand for countably-additive invariance, and perfectly well-behaved with
respect to a Banach measure — much as "the average position of a uniform distribution on
the whole line" is not a strange number but an undefined one, undefined for a reason one
can name. Standard analysis restricted the domain rather than weaken additivity or drop
invariance not from a failure of imagination but as a cost–benefit verdict: it preserves
exactly the two properties integration needs and forfeits only the ability to measure sets
no one can point to.

## Two closing remarks

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(On minimality of hypotheses)</span></p>

The proof used translation invariance only through the single countable subgroup $\mathbb{Q}\_1$, and used the circle only to make that subgroup act cocompactly with a tidy total mass. The same template — *a countable dense subgroup, a transversal, an invariant countably additive probability measure* — produces a non-measurable set in any compact group, and indeed underlies the general slogan that invariant measures and choice transversals are incompatible whenever the orbit space is "too small to see."

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(On what the example is really teaching)</span></p>

The Vitali set is often filed under "pathologies of the axiom of choice," but the sharper lesson is about *additivity at infinity*. Length behaves itself under finite cutting and translating; it is the passage to *countably many* pieces, combined with the demand that translation change nothing, that forces a choice. Measure theory's central compromise — restrict to a $\sigma$-algebra and you may keep all three of your desiderata — is the constructive response to exactly this example.

</div>
