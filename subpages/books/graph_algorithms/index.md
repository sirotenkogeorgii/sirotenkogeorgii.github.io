---
layout: default
title: Graph Algorithms
date: 2026-07-07
excerpt: Lecture notes on graph algorithms covering network flows, the Ford–Fulkerson method, minimum cuts, and applications such as bipartite matching.
tags:
  - graphs
  - algorithms
  - combinatorial-optimization
---

<style>
  .accordion summary {
    font-weight: 600;
    color: var(--accent-strong, #2c3e94);
    background-color: var(--accent-soft, #f5f6ff);
    padding: 0.35rem 0.6rem;
    border-left: 3px solid var(--accent-strong, #2c3e94);
    border-radius: 0.25rem;
  }
</style>

**Table of Contents**
- TOC
{:toc}

# Graph Algorithms

# Lecture 1: Network Flows

## Notation

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Graphs)</span></p>

Throughout these notes we fix the following conventions:

* $G$ — the current graph; $V$ its set of vertices (always $V \neq \emptyset$), $E$ its set of edges.
* $uv \in E$ denotes an edge, i.e. $\lbrace u, v \rbrace$ in the undirected case and the ordered pair $(u, v)$ in the directed case.
* $n := \lvert V(G)\rvert$ and $m := \lvert E(G)\rvert$. For another graph $H$ we write $V(H)$, $E(H)$, $n(H)$, $m(H)$; for a sequence of graphs $G\_1, G\_2, \dots$ we write $V\_1, V\_2, \dots$ and $n\_1, n\_2, \dots$
* We may always assume WLOG that the graph has no isolated vertices; then $m$ is $\Omega(n)$, so bounds of the form $O(n + m)$ simplify to $O(m)$.

</div>

## Networks and Flows

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Network)</span></p>

A **network** consists of:

* a directed graph $G = (V, E)$,
* two distinguished vertices $s, t \in V$ with $s \neq t$ — the **source** and the **target**,
* a function $c \colon E \to [0, +\infty)$ assigning **capacities** to edges.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 560 230" width="100%" style="max-width: 560px; height: auto;" role="img" aria-labelledby="gaf1-title">
  <title id="gaf1-title">A network with source, target and capacities</title>
  <defs>
    <marker id="gaf1-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
    </marker>
  </defs>
  <line x1="79" y1="103" x2="199" y2="58" stroke="#333" stroke-width="1.6" marker-end="url(#gaf1-arr)"/>
  <line x1="79" y1="117" x2="199" y2="162" stroke="#333" stroke-width="1.6" marker-end="url(#gaf1-arr)"/>
  <line x1="220" y1="72" x2="220" y2="146" stroke="#333" stroke-width="1.6" marker-end="url(#gaf1-arr)"/>
  <line x1="239" y1="55" x2="436" y2="104" stroke="#333" stroke-width="1.6" marker-end="url(#gaf1-arr)"/>
  <line x1="239" y1="165" x2="436" y2="116" stroke="#333" stroke-width="1.6" marker-end="url(#gaf1-arr)"/>
  <text x="128" y="66" font-family="serif" font-size="12" font-style="italic" fill="#a86f00">c = 3</text>
  <text x="112" y="152" font-family="serif" font-size="12" font-style="italic" fill="#a86f00">c = 2</text>
  <text x="230" y="113" font-family="serif" font-size="12" font-style="italic" fill="#a86f00">c = 1</text>
  <text x="336" y="62" font-family="serif" font-size="12" font-style="italic" fill="#a86f00">c = 2</text>
  <text x="336" y="164" font-family="serif" font-size="12" font-style="italic" fill="#a86f00">c = 3</text>
  <circle cx="60" cy="110" r="18" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="60" y="115" text-anchor="middle" font-family="serif" font-size="15" font-style="italic" fill="#0f9b6c">s</text>
  <text x="60" y="148" text-anchor="middle" font-family="serif" font-size="10" fill="#666">source</text>
  <circle cx="220" cy="50" r="18" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="220" y="55" text-anchor="middle" font-family="serif" font-size="15" font-style="italic" fill="#1565c0">a</text>
  <circle cx="220" cy="170" r="18" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="220" y="175" text-anchor="middle" font-family="serif" font-size="15" font-style="italic" fill="#1565c0">b</text>
  <circle cx="460" cy="110" r="18" fill="#fef2f2" stroke="#b91c1c" stroke-width="2"/>
  <text x="460" y="115" text-anchor="middle" font-family="serif" font-size="15" font-style="italic" fill="#b91c1c">t</text>
  <text x="460" y="148" text-anchor="middle" font-family="serif" font-size="10" fill="#666">target</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
A network: a directed graph $G = (V, E)$ with a source $s$, a target $t$, and capacities $c$ on the edges.
</figcaption>
</figure>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Inflow, Outflow, Excess)</span></p>

For a function $f \colon E \to [0, +\infty)$ and a vertex $v \in V$ we define the **inflow**, the **outflow**, and the **excess** at $v$:

$$
f^{+}(v) := \sum_{uv \in E} f(uv),
\qquad
f^{-}(v) := \sum_{vu \in E} f(vu),
\qquad
f^{\Delta}(v) := f^{+}(v) - f^{-}(v).
$$

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 520 210" width="100%" style="max-width: 480px; height: auto;" role="img" aria-labelledby="gaf2-title">
  <title id="gaf2-title">Inflow, outflow and excess at a vertex</title>
  <defs>
    <marker id="gaf2-arrg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#0f9b6c"/>
    </marker>
    <marker id="gaf2-arrb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#1565c0"/>
    </marker>
  </defs>
  <line x1="95" y1="50" x2="236" y2="92" stroke="#0f9b6c" stroke-width="1.8" marker-end="url(#gaf2-arrg)"/>
  <line x1="85" y1="105" x2="233" y2="105" stroke="#0f9b6c" stroke-width="1.8" marker-end="url(#gaf2-arrg)"/>
  <line x1="95" y1="160" x2="236" y2="118" stroke="#0f9b6c" stroke-width="1.8" marker-end="url(#gaf2-arrg)"/>
  <line x1="284" y1="94" x2="425" y2="52" stroke="#1565c0" stroke-width="1.8" marker-end="url(#gaf2-arrb)"/>
  <line x1="284" y1="116" x2="425" y2="158" stroke="#1565c0" stroke-width="1.8" marker-end="url(#gaf2-arrb)"/>
  <circle cx="260" cy="105" r="20" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <text x="260" y="110" text-anchor="middle" font-family="serif" font-size="15" font-style="italic" fill="#1565c0">v</text>
  <text x="130" y="28" text-anchor="middle" font-family="serif" font-size="12" fill="#0f9b6c">inflow f<tspan dy="-5" font-size="9">+</tspan><tspan dy="5">(v)</tspan></text>
  <text x="392" y="28" text-anchor="middle" font-family="serif" font-size="12" fill="#1565c0">outflow f<tspan dy="-5" font-size="9">−</tspan><tspan dy="5">(v)</tspan></text>
  <text x="260" y="196" text-anchor="middle" font-family="serif" font-size="12" fill="#333">excess f<tspan dy="-5" font-size="9">Δ</tspan><tspan dy="5">(v) = f</tspan><tspan dy="-5" font-size="9">+</tspan><tspan dy="5">(v) − f</tspan><tspan dy="-5" font-size="9">−</tspan><tspan dy="5">(v)</tspan></text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Inflow, outflow and excess at a vertex. For a flow, Kirchhoff's law forces $f^{\Delta}(v) = 0$ at every inner vertex $v \neq s, t$: whatever arrives must leave.
</figcaption>
</figure>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Flow, Flow Size)</span></p>

A **flow** is a function $f \colon E \to [0, +\infty)$ such that:

1. $f \le c$ (pointwise on edges — the flow respects the capacities),
2. $f^{\Delta}(v) = 0$ for every $v \neq s, t$ — **Kirchhoff's law**: whatever flows into an inner vertex flows out again.

The **size** of a flow is the excess at the target,

$$
|f| := f^{\Delta}(t).
$$

**Goal:** find a flow of maximum size.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Measuring the Size at the Source)</span></p>

Equivalently $\lvert f\rvert = -f^{\Delta}(s)$: every unit of flow contributes $+1$ to the excess of its head and $-1$ to the excess of its tail, so summing over all vertices everything cancels,

$$
\sum_{v \in V} f^{\Delta}(v) = 0.
$$

By Kirchhoff's law all terms with $v \neq s, t$ vanish, leaving $f^{\Delta}(s) + f^{\Delta}(t) = 0$.

</div>

## Cuts

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Edges Between Sets, Cut)</span></p>

For disjoint subsets $A, B \subseteq V$ we write

$$
E(A, B) := \lbrace ab \in E \mid a \in A \ \& \ b \in B \rbrace.
$$

A **cut** (more precisely an elementary $st$-cut) is a set of edges of the form $E(A, \overline{A})$ for any $A \subseteq V$ such that $s \in A$ and $t \notin A$, where $\overline{A} := V \setminus A$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Flow Across a Cut, Capacity of a Cut)</span></p>

For a flow $f$ and disjoint $A, B \subseteq V$ we set

$$
f(A, B) := \sum_{e \in E(A, B)} f(e),
\qquad
f^{\Delta}(A, \overline{A}) := f(A, \overline{A}) - f(\overline{A}, A),
$$

the latter being the **net flow** from $A$ to $\overline{A}$. Note that this is consistent with the excess of a single vertex: $f^{\Delta}(v) = f^{\Delta}\bigl(\overline{\lbrace v \rbrace}, \lbrace v \rbrace\bigr)$.

The **capacity of a cut** is the *directed* quantity

$$
c(A, \overline{A}) := \sum_{e \in E(A, \overline{A})} c(e),
$$

i.e. only edges leaving $A$ count; edges returning from $\overline{A}$ to $A$ do not.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 560 250" width="100%" style="max-width: 560px; height: auto;" role="img" aria-labelledby="gaf3-title">
  <title id="gaf3-title">An st-cut and the edges crossing it</title>
  <defs>
    <marker id="gaf3-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
    </marker>
    <marker id="gaf3-arrg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#888"/>
    </marker>
  </defs>
  <ellipse cx="150" cy="130" rx="105" ry="85" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <ellipse cx="410" cy="130" rx="105" ry="85" fill="#fce4ec" stroke="#c2185b" stroke-width="1.8"/>
  <text x="120" y="62" text-anchor="middle" font-family="serif" font-size="15" font-style="italic" fill="#1565c0">A</text>
  <text x="443" y="62" text-anchor="middle" font-family="serif" font-size="15" font-style="italic" fill="#c2185b">A</text>
  <line x1="437" y1="48" x2="450" y2="48" stroke="#c2185b" stroke-width="1.4"/>
  <circle cx="105" cy="130" r="15" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="105" y="135" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#0f9b6c">s</text>
  <circle cx="455" cy="130" r="15" fill="#fef2f2" stroke="#b91c1c" stroke-width="2"/>
  <text x="455" y="135" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#b91c1c">t</text>
  <line x1="243" y1="90" x2="318" y2="90" stroke="#333" stroke-width="1.8" marker-end="url(#gaf3-arr)"/>
  <line x1="252" y1="130" x2="309" y2="130" stroke="#333" stroke-width="1.8" marker-end="url(#gaf3-arr)"/>
  <line x1="243" y1="170" x2="318" y2="170" stroke="#333" stroke-width="1.8" marker-end="url(#gaf3-arr)"/>
  <line x1="325" y1="205" x2="237" y2="205" stroke="#888" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#gaf3-arrg)"/>
  <text x="281" y="26" text-anchor="middle" font-family="serif" font-size="11" fill="#333">cut edges — counted in the capacity</text>
  <text x="281" y="232" text-anchor="middle" font-family="serif" font-size="11" fill="#888">opposite direction — not counted</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
A cut $E(A, \overline{A})$ with $s \in A$ and $t \notin A$. The capacity $c(A, \overline{A})$ is a directed notion: it sums only over the edges leaving $A$ (solid), not over those returning (dashed) — yet the net flow through the cut always equals $|f|$.
</figcaption>
</figure>

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Flow Across a Cut)</span></p>

For every cut $E(A, \overline{A})$:

$$
f^{\Delta}(A, \overline{A}) = |f|.
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Sum the excesses over the far side of the cut:

$$
\sum_{v \in \overline{A}} f^{\Delta}(v) = f^{\Delta}(t) = |f|,
$$

because by Kirchhoff's law every vertex of $\overline{A}$ other than $t$ has zero excess (and $s \notin \overline{A}$).

On the other hand, in the same sum every edge with both endpoints inside $\overline{A}$ contributes $+f(e)$ once (at its head) and $-f(e)$ once (at its tail), so it cancels; only edges crossing the cut survive:

$$
\sum_{v \in \overline{A}} f^{\Delta}(v) = f(A, \overline{A}) - f(\overline{A}, A) = f^{\Delta}(A, \overline{A}).
$$

Combining the two evaluations gives $f^{\Delta}(A, \overline{A}) = \lvert f\rvert$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Flows vs. Cuts)</span></p>

The size of every flow is at most the capacity of every cut:

$$
|f| = f^{\Delta}(A, \overline{A}) = f(A, \overline{A}) - f(\overline{A}, A) \le c(A, \overline{A}).
$$

Indeed, $f(A, \overline{A}) \le c(A, \overline{A})$ since $f \le c$, and $f(\overline{A}, A) \ge 0$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Self-Certification)</span></p>

If $\lvert f\rvert = c(A, \overline{A})$ for some flow $f$ and some cut $E(A, \overline{A})$, then:

1. $f$ is a **maximum** flow (no flow can exceed the capacity of this cut),
2. $E(A, \overline{A})$ is a **minimum** cut (no cut can have capacity below the size of this flow).

So a flow and a cut of equal value certify each other's optimality — an algorithm that outputs such a pair is **self-certifying**.

</div>

## Augmenting Paths and the Ford–Fulkerson Algorithm

**Naive idea.** Find an $st$-path $e\_1, \dots, e\_k$ such that $f(e\_i) < c(e\_i)$ for all $i$. Then we can *push* $\varepsilon := \min\_i \bigl(c(e\_i) - f(e\_i)\bigr) > 0$ along the path — increase the flow by $\varepsilon$ on every edge of the path — which increases $\lvert f\rvert$ by $\varepsilon$: Kirchhoff's law is preserved at the inner vertices of the path since both their inflow and outflow grow by $\varepsilon$.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(The Naive Idea Is Broken)</span></p>

Pushing only along edges with spare capacity can get stuck at a non-maximum flow. In the network below (all capacities $1$), a greedy first augmentation along $s \to u \to v \to t$ saturates the diagonal edge $uv$; afterwards no $st$-path with spare capacity remains, and the naive rule terminates with $\lvert f\rvert = 1$ — although the maximum flow has size $2$. The larger flow exists, it just requires *rerouting* the unit currently using the diagonal, and the naive rule offers no way to undo a bad routing decision.

The fix: allow a path to also traverse an edge **backwards**, cancelling flow that was already sent through it.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 1020 260" width="100%" style="max-width: 820px; height: auto;" role="img" aria-labelledby="gaf4-title">
  <title id="gaf4-title">The naive augmenting rule gets stuck; a backward edge repairs it</title>
  <defs>
    <marker id="gaf4-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
    </marker>
    <marker id="gaf4-arrr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#b91c1c"/>
    </marker>
    <marker id="gaf4-arrg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#0f9b6c"/>
    </marker>
  </defs>
  <g>
    <text x="230" y="22" text-anchor="middle" font-family="serif" font-size="12" fill="#333">Greedy push along s → u → v → t, then stuck</text>
    <line x1="86" y1="122" x2="213" y2="63" stroke="#b91c1c" stroke-width="2" marker-end="url(#gaf4-arrr)"/>
    <line x1="86" y1="138" x2="213" y2="197" stroke="#333" stroke-width="1.6" marker-end="url(#gaf4-arr)"/>
    <line x1="230" y1="73" x2="230" y2="187" stroke="#b91c1c" stroke-width="2" marker-end="url(#gaf4-arrr)"/>
    <line x1="245" y1="62" x2="374" y2="122" stroke="#333" stroke-width="1.6" marker-end="url(#gaf4-arr)"/>
    <line x1="245" y1="198" x2="374" y2="138" stroke="#b91c1c" stroke-width="2" marker-end="url(#gaf4-arrr)"/>
    <text x="132" y="80" text-anchor="middle" font-family="serif" font-size="12" fill="#b91c1c">1/1</text>
    <text x="130" y="188" text-anchor="middle" font-family="serif" font-size="12" fill="#333">0/1</text>
    <text x="212" y="134" text-anchor="end" font-family="serif" font-size="12" fill="#b91c1c">1/1</text>
    <text x="330" y="78" text-anchor="middle" font-family="serif" font-size="12" fill="#333">0/1</text>
    <text x="330" y="190" text-anchor="middle" font-family="serif" font-size="12" fill="#b91c1c">1/1</text>
    <circle cx="70" cy="130" r="16" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
    <text x="70" y="135" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#0f9b6c">s</text>
    <circle cx="230" cy="55" r="16" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
    <text x="230" y="60" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#1565c0">u</text>
    <circle cx="230" cy="205" r="16" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
    <text x="230" y="210" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#1565c0">v</text>
    <circle cx="390" cy="130" r="16" fill="#fef2f2" stroke="#b91c1c" stroke-width="2"/>
    <text x="390" y="135" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#b91c1c">t</text>
    <text x="230" y="248" text-anchor="middle" font-family="serif" font-size="11" fill="#888">saturated edges in red: |f| = 1</text>
  </g>
  <g>
    <text x="790" y="22" text-anchor="middle" font-family="serif" font-size="12" fill="#333">Augmenting path uses the diagonal backwards</text>
    <line x1="646" y1="122" x2="773" y2="63" stroke="#b91c1c" stroke-width="2" marker-end="url(#gaf4-arrr)"/>
    <line x1="646" y1="138" x2="773" y2="197" stroke="#333" stroke-width="1.6" marker-end="url(#gaf4-arr)"/>
    <line x1="790" y1="73" x2="790" y2="187" stroke="#b91c1c" stroke-width="2" marker-end="url(#gaf4-arrr)"/>
    <line x1="805" y1="62" x2="934" y2="122" stroke="#333" stroke-width="1.6" marker-end="url(#gaf4-arr)"/>
    <line x1="805" y1="198" x2="934" y2="138" stroke="#b91c1c" stroke-width="2" marker-end="url(#gaf4-arrr)"/>
    <line x1="638" y1="150" x2="762" y2="208" stroke="#0f9b6c" stroke-width="1.8" stroke-dasharray="5 4" marker-end="url(#gaf4-arrg)"/>
    <line x1="803" y1="184" x2="803" y2="76" stroke="#0f9b6c" stroke-width="1.8" stroke-dasharray="5 4" marker-end="url(#gaf4-arrg)"/>
    <line x1="800" y1="50" x2="928" y2="110" stroke="#0f9b6c" stroke-width="1.8" stroke-dasharray="5 4" marker-end="url(#gaf4-arrg)"/>
    <text x="688" y="196" text-anchor="middle" font-family="serif" font-size="12" fill="#0f9b6c">+1</text>
    <text x="820" y="134" text-anchor="start" font-family="serif" font-size="12" fill="#0f9b6c">−1</text>
    <text x="872" y="66" text-anchor="middle" font-family="serif" font-size="12" fill="#0f9b6c">+1</text>
    <circle cx="630" cy="130" r="16" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
    <text x="630" y="135" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#0f9b6c">s</text>
    <circle cx="790" cy="55" r="16" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
    <text x="790" y="60" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#1565c0">u</text>
    <circle cx="790" cy="205" r="16" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
    <text x="790" y="210" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#1565c0">v</text>
    <circle cx="950" cy="130" r="16" fill="#fef2f2" stroke="#b91c1c" stroke-width="2"/>
    <text x="950" y="135" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#b91c1c">t</text>
    <text x="790" y="248" text-anchor="middle" font-family="serif" font-size="11" fill="#888">push 1 along s → v → u → t: |f| = 2</text>
  </g>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Left: the greedy rule saturates the diagonal and stalls at $|f| = 1$. Right: the augmenting path $s \to v \to u \to t$ traverses the saturated diagonal *backwards* ($r(vu) = f(uv) = 1 > 0$), cancelling its unit of flow and reaching the maximum $|f| = 2$.
</figcaption>
</figure>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Residual Capacity)</span></p>

The **residual capacity** of a pair $uv \in E$ is

$$
r(uv) := \underbrace{c(uv) - f(uv)}_{\text{increase } u \to v} + \underbrace{f(vu)}_{\text{decrease } v \to u},
$$

i.e. how much more can effectively be sent from $u$ to $v$ — either by using spare capacity of the edge $uv$, or by cancelling flow on the opposite edge $vu$.

An $st$-path $e\_1, \dots, e\_k$ with $r(e\_i) > 0$ for all $i$ is called an **augmenting path**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Ford–Fulkerson)</span></p>

Start with the zero flow and **repeat**:

1. Find an $st$-path $e\_1, \dots, e\_k$ such that $r(e\_i) > 0$ for all $i$ (e.g. by BFS over the edges of positive residual capacity, in time $O(m)$). If no such path exists, stop.
2. Set $\varepsilon := \min\_i r(e\_i)$; note $\varepsilon > 0$.
3. **Push** $\varepsilon$ through the path: on each $e\_i = uv$, first cancel flow on the opposite edge $vu$ (decrease $f(vu)$), and send the rest as an increase of $f(uv)$.

Each iteration increases $\lvert f\rvert$ by $\varepsilon > 0$.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 150" width="100%" style="max-width: 620px; height: auto;" role="img" aria-labelledby="gaf5-title">
  <title id="gaf5-title">Pushing epsilon through an augmenting path</title>
  <defs>
    <marker id="gaf5-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
    </marker>
    <marker id="gaf5-arrg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#0f9b6c"/>
    </marker>
  </defs>
  <line x1="90" y1="34" x2="550" y2="34" stroke="#0f9b6c" stroke-width="1.6" stroke-dasharray="6 4" marker-end="url(#gaf5-arrg)"/>
  <text x="320" y="22" text-anchor="middle" font-family="serif" font-size="11" fill="#0f9b6c">augmenting path (direction of traversal)</text>
  <line x1="76" y1="80" x2="212" y2="80" stroke="#333" stroke-width="1.8" marker-end="url(#gaf5-arr)"/>
  <line x1="384" y1="80" x2="248" y2="80" stroke="#333" stroke-width="1.8" marker-end="url(#gaf5-arr)"/>
  <line x1="416" y1="80" x2="552" y2="80" stroke="#333" stroke-width="1.8" marker-end="url(#gaf5-arr)"/>
  <text x="144" y="66" text-anchor="middle" font-family="serif" font-size="13" fill="#0f9b6c">+ε</text>
  <text x="316" y="66" text-anchor="middle" font-family="serif" font-size="13" fill="#0f9b6c">−ε</text>
  <text x="484" y="66" text-anchor="middle" font-family="serif" font-size="13" fill="#0f9b6c">+ε</text>
  <text x="316" y="103" text-anchor="middle" font-family="serif" font-size="10" fill="#888">(edge oriented against the path:</text>
  <text x="316" y="115" text-anchor="middle" font-family="serif" font-size="10" fill="#888">cancel flow instead of adding)</text>
  <circle cx="60" cy="80" r="16" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="60" y="85" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#0f9b6c">s</text>
  <circle cx="230" cy="80" r="16" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="230" y="85" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#1565c0">a</text>
  <circle cx="400" cy="80" r="16" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="400" y="85" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#1565c0">b</text>
  <circle cx="570" cy="80" r="16" fill="#fef2f2" stroke="#b91c1c" stroke-width="2"/>
  <text x="570" y="85" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#b91c1c">t</text>
  <text x="320" y="142" text-anchor="middle" font-family="serif" font-size="11" fill="#666">r(e) &gt; 0 on every step;  ε = min r(e)</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Pushing $\varepsilon$ through an augmenting path: edges traversed forwards gain $\varepsilon$; edges oriented against the path (here $ba$) have their flow cancelled by $\varepsilon$ instead. Kirchhoff's law survives at every inner vertex.
</figcaption>
</figure>

Two questions remain:

1. Does the Ford–Fulkerson algorithm stop?
2. Does it produce a maximum flow?

The second question has a positive answer whenever the algorithm stops, and the argument yields the central theorem of this chapter. (A trivial upper bound is always available: $\lvert f\rvert \le c^{+}(s)$, the total capacity of edges leaving the source.)

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Max-Flow Min-Cut)</span></p>

For every maximum flow $f$ there is a cut $E(A, \overline{A})$ such that

$$
|f| = c(A, \overline{A}).
$$

In particular, the maximum flow size equals the minimum cut capacity, and Ford–Fulkerson — when it stops — stops with a maximum flow and a certifying minimum cut.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Consider the moment the algorithm stops with flow $f$, i.e. no augmenting path exists, and set

$$
A := \lbrace v \in V \mid \exists \text{ an augmenting } sv\text{-path} \rbrace .
$$

Then $s \in A$ (the empty path works) and $t \notin A$ (otherwise the algorithm would not have stopped), so $E(A, \overline{A})$ is a cut.

Every pair $uv$ with $u \in A$, $v \in \overline{A}$ must have $r(uv) = 0$ — otherwise the augmenting path to $u$ would extend to $v$ and $v$ would lie in $A$. By the definition of residual capacity this means both:

* $f(uv) = c(uv)$ for every edge $uv \in E(A, \overline{A})$ — all cut edges are saturated, and
* $f(vu) = 0$ for every edge $vu \in E(\overline{A}, A)$ — nothing flows back.

Therefore, using the lemma on flow across a cut,

$$
|f| = f^{\Delta}(A, \overline{A}) = f(A, \overline{A}) - f(\overline{A}, A) = c(A, \overline{A}) - 0 = c(A, \overline{A}).
$$

By the self-certification observation, $f$ is maximum and $E(A, \overline{A})$ is minimum. Conversely, since *any* maximum flow admits no augmenting path (otherwise it could be enlarged), the same construction produces a matching cut for it.

</details>
</div>

## Termination and Integrality

Does Ford–Fulkerson always terminate? **Generally: no!** With arbitrary real capacities the sequence of augmentations may run forever (and even converge to a non-maximum value). However:

1. **Integer capacities $\Rightarrow$ yes.** Every $\varepsilon$ is a positive integer, so each iteration increases $\lvert f\rvert$ by at least $1$, and the flow size is bounded.
2. **Rational capacities $\Rightarrow$ yes.** This reduces to the integer case by the following scaling invariance.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Invariance to Scaling)</span></p>

If we multiply all capacities by a constant $\alpha > 0$, all quantities in the algorithm (flows, residual capacities, all intermediate results) also multiply by $\alpha$, and the algorithm makes exactly the same decisions. For rational capacities, choose $\alpha$ as the common denominator to obtain an equivalent integer-capacity network.

</div>

Since with integer capacities the algorithm only ever adds and subtracts integers, the flow it computes is itself integer, which gives:

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Integrality)</span></p>

In a network with integer capacities, at least one maximum flow is integer (i.e. $f(e) \in \mathbb{Z}$ for every edge $e$) — namely the one found by the Ford–Fulkerson algorithm.

</div>

Ford–Fulkerson on integer capacities can still be **slow** — the number of iterations is bounded only by $\lvert f\_{\max}\rvert$, which can be huge. But it is fast for *small* integer capacities:

1. If $c \in \lbrace 0, 1 \rbrace$: then $\lvert f\rvert \le n$, so there are at most $n$ iterations and the algorithm runs in time $O(nm)$.
2. If $c \in \lbrace 0, \dots, L \rbrace$: then $\lvert f\rvert \le n \cdot L$ and the algorithm runs in time $O(Lnm)$.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Better Augmenting-Path Algorithms)</span></p>

* Ford–Fulkerson with the **shortest** augmenting path (found by BFS) is the **Edmonds–Karp algorithm**; it runs in time $O(m^2 n)$ — even for arbitrary real capacities.
* Next up: **Dinitz's algorithm**, running in time $O(n^2 m)$.

</div>

## Application: Bipartite Matching

Given a bipartite graph with parts $L$ and $R$, we look for a **matching of maximum size** (a maximum set of edges, no two sharing a vertex). Build a network:

* orient all edges from $L$ to $R$,
* add a source $s$ with an edge $s\ell$ for every $\ell \in L$, and a target $t$ with an edge $rt$ for every $r \in R$,
* set all capacities to $c = 1$,

and find a maximum **integer** flow.

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 270" width="100%" style="max-width: 600px; height: auto;" role="img" aria-labelledby="gaf6-title">
  <title id="gaf6-title">The matching network of a bipartite graph</title>
  <defs>
    <marker id="gaf6-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#888"/>
    </marker>
    <marker id="gaf6-arrg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#0f9b6c"/>
    </marker>
  </defs>
  <text x="230" y="26" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#1565c0">L</text>
  <text x="410" y="26" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#c2185b">R</text>
  <line x1="65" y1="130" x2="212" y2="66" stroke="#888" stroke-width="1.4" marker-end="url(#gaf6-arr)"/>
  <line x1="66" y1="140" x2="212" y2="140" stroke="#888" stroke-width="1.4" marker-end="url(#gaf6-arr)"/>
  <line x1="65" y1="150" x2="212" y2="214" stroke="#888" stroke-width="1.4" marker-end="url(#gaf6-arr)"/>
  <line x1="428" y1="66" x2="575" y2="130" stroke="#888" stroke-width="1.4" marker-end="url(#gaf6-arr)"/>
  <line x1="428" y1="140" x2="574" y2="140" stroke="#888" stroke-width="1.4" marker-end="url(#gaf6-arr)"/>
  <line x1="428" y1="214" x2="575" y2="150" stroke="#888" stroke-width="1.4" marker-end="url(#gaf6-arr)"/>
  <line x1="246" y1="60" x2="392" y2="60" stroke="#888" stroke-width="1.4" marker-end="url(#gaf6-arr)"/>
  <line x1="245" y1="149" x2="393" y2="211" stroke="#888" stroke-width="1.4" marker-end="url(#gaf6-arr)"/>
  <line x1="245" y1="69" x2="393" y2="131" stroke="#0f9b6c" stroke-width="3" marker-end="url(#gaf6-arrg)"/>
  <line x1="245" y1="131" x2="393" y2="69" stroke="#0f9b6c" stroke-width="3" marker-end="url(#gaf6-arrg)"/>
  <line x1="246" y1="220" x2="392" y2="220" stroke="#0f9b6c" stroke-width="3" marker-end="url(#gaf6-arrg)"/>
  <circle cx="50" cy="140" r="16" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="50" y="145" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#0f9b6c">s</text>
  <circle cx="230" cy="60" r="15" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="230" cy="140" r="15" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="230" cy="220" r="15" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="410" cy="60" r="15" fill="#fce4ec" stroke="#c2185b" stroke-width="1.6"/>
  <circle cx="410" cy="140" r="15" fill="#fce4ec" stroke="#c2185b" stroke-width="1.6"/>
  <circle cx="410" cy="220" r="15" fill="#fce4ec" stroke="#c2185b" stroke-width="1.6"/>
  <circle cx="590" cy="140" r="16" fill="#fef2f2" stroke="#b91c1c" stroke-width="2"/>
  <text x="590" y="145" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#b91c1c">t</text>
  <text x="320" y="258" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#a86f00">all capacities = 1</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The matching network: $s$ feeds every vertex of $L$, every vertex of $R$ feeds $t$, all capacities are $1$. An integer flow of size $3$ (bold green middle edges) is exactly a matching of size $3$.
</figcaption>
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Integer Flows $\cong$ Matchings)</span></p>

Integer flows in this network correspond exactly to matchings: with unit capacities each $\ell \in L$ receives at most one unit of flow and each $r \in R$ sends out at most one unit, so the middle edges carrying flow $1$ form a matching of size $\lvert f\rvert$ — and conversely, every matching yields such a flow. Hence a maximum integer flow (which exists by the integrality theorem) gives a matching of maximum size.

</div>

Since all capacities are $0/1$, Ford–Fulkerson finds this flow in time $O(nm)$. Next week we will see how to do it in time $O(\sqrt{n} \cdot m)$.
