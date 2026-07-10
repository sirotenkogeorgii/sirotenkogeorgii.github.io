---
layout: default
title: Graph Algorithms
date: 2026-07-10
excerpt: Lecture notes on graph algorithms covering network flows, the Ford–Fulkerson method, Dinitz's algorithm, capacity scaling, Menger's theorems, the Karger–Stein randomized min-cut, shortest paths with the Bellman–Ford–Moore and Dijkstra algorithms (heaps, multi-level buckets, potentials, A-star), all-pairs shortest paths with Floyd–Warshall, the walk algebra, fast matrix multiplication and Seidel's algorithm, minimum spanning trees (cut lemma, red–blue meta-algorithm, Jarník, Borůvka, Kruskal, contractive Borůvka on minor-closed classes, Fredman–Tarjan), dynamic connectivity with Euler-tour trees and the Holm–de Lichtenberg–Thorup structure, and applications such as bipartite matching.
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

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 560 230" width="100%" style="max-width: 520px; height: auto;" role="img" aria-labelledby="gaf7-title">
  <title id="gaf7-title">Each edge cancels in the sum of all excesses</title>
  <defs>
    <marker id="gaf7-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
    </marker>
    <marker id="gaf7-arrf" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#888"/>
    </marker>
  </defs>
  <ellipse cx="280" cy="120" rx="245" ry="95" fill="none" stroke="#888" stroke-width="1.4" stroke-dasharray="7 5"/>
  <text x="82" y="58" font-family="serif" font-size="14" font-style="italic" fill="#888">V</text>
  <line x1="141" y1="86" x2="176" y2="126" stroke="#888" stroke-width="1.2" marker-end="url(#gaf7-arrf)"/>
  <line x1="203" y1="150" x2="288" y2="172" stroke="#888" stroke-width="1.2" marker-end="url(#gaf7-arrf)"/>
  <line x1="404" y1="106" x2="435" y2="130" stroke="#888" stroke-width="1.2" marker-end="url(#gaf7-arrf)"/>
  <line x1="205" y1="132" x2="373" y2="99" stroke="#333" stroke-width="2.2" marker-end="url(#gaf7-arr)"/>
  <text x="289" y="102" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#a86f00">f(e)</text>
  <circle cx="132" cy="76" r="8" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.4"/>
  <circle cx="300" cy="175" r="8" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.4"/>
  <circle cx="445" cy="138" r="8" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.4"/>
  <circle cx="190" cy="135" r="14" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="190" y="140" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#1565c0">u</text>
  <circle cx="390" cy="96" r="14" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="390" y="101" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#1565c0">v</text>
  <text x="172" y="170" text-anchor="middle" font-family="serif" font-size="13" fill="#b91c1c">−f(e) at u</text>
  <text x="424" y="70" text-anchor="middle" font-family="serif" font-size="13" fill="#0f9b6c">+f(e) at v</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Why $\sum_{v \in V} f^{\Delta}(v) = 0$: each edge $e = uv$ appears exactly twice in the sum — as $+f(e)$ in the excess of its head $v$ and as $-f(e)$ in the excess of its tail $u$ — so everything cancels.
</figcaption>
</figure>

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

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 560 270" width="100%" style="max-width: 560px; height: auto;" role="img" aria-labelledby="gaf8-title">
  <title id="gaf8-title">The certifying cut when Ford–Fulkerson stops</title>
  <defs>
    <marker id="gaf8-arrr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#b91c1c"/>
    </marker>
    <marker id="gaf8-arrf" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#888"/>
    </marker>
    <marker id="gaf8-arrg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#0f9b6c"/>
    </marker>
  </defs>
  <text x="280" y="22" text-anchor="middle" font-family="serif" font-size="11" fill="#333">when the algorithm stops: r(uv) = 0 for every u ∈ A, v ∉ A</text>
  <ellipse cx="150" cy="145" rx="110" ry="90" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <ellipse cx="410" cy="145" rx="110" ry="90" fill="#fce4ec" stroke="#c2185b" stroke-width="1.8"/>
  <text x="115" y="75" text-anchor="middle" font-family="serif" font-size="15" font-style="italic" fill="#1565c0">A</text>
  <text x="447" y="75" text-anchor="middle" font-family="serif" font-size="15" font-style="italic" fill="#c2185b">A</text>
  <line x1="441" y1="61" x2="454" y2="61" stroke="#c2185b" stroke-width="1.4"/>
  <line x1="88" y1="138" x2="146" y2="102" stroke="#0f9b6c" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#gaf8-arrg)"/>
  <line x1="88" y1="152" x2="161" y2="181" stroke="#0f9b6c" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#gaf8-arrg)"/>
  <circle cx="75" cy="145" r="13" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="75" y="150" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#0f9b6c">s</text>
  <circle cx="158" cy="95" r="9" fill="#fff" stroke="#1565c0" stroke-width="1.4"/>
  <circle cx="173" cy="186" r="9" fill="#fff" stroke="#1565c0" stroke-width="1.4"/>
  <circle cx="460" cy="145" r="13" fill="#fef2f2" stroke="#b91c1c" stroke-width="2"/>
  <text x="460" y="150" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#b91c1c">t</text>
  <line x1="248" y1="110" x2="315" y2="110" stroke="#b91c1c" stroke-width="2" marker-end="url(#gaf8-arrr)"/>
  <line x1="253" y1="150" x2="309" y2="150" stroke="#b91c1c" stroke-width="2" marker-end="url(#gaf8-arrr)"/>
  <text x="283" y="98" text-anchor="middle" font-family="serif" font-size="11" fill="#b91c1c">f = c</text>
  <text x="283" y="140" text-anchor="middle" font-family="serif" font-size="11" fill="#b91c1c">f = c</text>
  <line x1="352" y1="222" x2="212" y2="222" stroke="#888" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#gaf8-arrf)"/>
  <text x="283" y="240" text-anchor="middle" font-family="serif" font-size="11" fill="#888">f = 0</text>
  <text x="150" y="258" text-anchor="middle" font-family="serif" font-size="10" fill="#666">reachable by augmenting paths</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The certifying cut at termination: $A$ collects $s$ and everything reachable by augmenting paths (dashed green). Every edge leaving $A$ is saturated ($f = c$), every edge entering $A$ is empty ($f = 0$) — otherwise $A$ would grow — so $f^{\Delta}(A, \overline{A}) = c(A, \overline{A})$.
</figcaption>
</figure>

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

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 210" width="100%" style="max-width: 600px; height: auto;" role="img" aria-labelledby="gaf9-title">
  <title id="gaf9-title">Unit capacities force at most one unit through each vertex</title>
  <defs>
    <marker id="gaf9-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
    </marker>
    <marker id="gaf9-arrf" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#888"/>
    </marker>
  </defs>
  <g>
    <line x1="61" y1="100" x2="145" y2="100" stroke="#333" stroke-width="2" marker-end="url(#gaf9-arr)"/>
    <text x="103" y="88" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#a86f00">c = 1</text>
    <line x1="178" y1="93" x2="272" y2="52" stroke="#888" stroke-width="1.4" marker-end="url(#gaf9-arrf)"/>
    <line x1="181" y1="100" x2="269" y2="100" stroke="#888" stroke-width="1.4" marker-end="url(#gaf9-arrf)"/>
    <line x1="178" y1="107" x2="272" y2="148" stroke="#888" stroke-width="1.4" marker-end="url(#gaf9-arrf)"/>
    <circle cx="45" cy="100" r="14" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
    <text x="45" y="105" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#0f9b6c">s</text>
    <circle cx="165" cy="100" r="14" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
    <text x="165" y="105" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#1565c0">ℓ</text>
    <circle cx="283" cy="48" r="8" fill="#fce4ec" stroke="#c2185b" stroke-width="1.4"/>
    <circle cx="283" cy="100" r="8" fill="#fce4ec" stroke="#c2185b" stroke-width="1.4"/>
    <circle cx="283" cy="152" r="8" fill="#fce4ec" stroke="#c2185b" stroke-width="1.4"/>
    <text x="165" y="190" text-anchor="middle" font-family="serif" font-size="11" fill="#666">at most 1 unit can enter ℓ</text>
  </g>
  <g>
    <line x1="366" y1="52" x2="460" y2="93" stroke="#888" stroke-width="1.4" marker-end="url(#gaf9-arrf)"/>
    <line x1="369" y1="100" x2="457" y2="100" stroke="#888" stroke-width="1.4" marker-end="url(#gaf9-arrf)"/>
    <line x1="366" y1="148" x2="460" y2="107" stroke="#888" stroke-width="1.4" marker-end="url(#gaf9-arrf)"/>
    <line x1="491" y1="100" x2="575" y2="100" stroke="#333" stroke-width="2" marker-end="url(#gaf9-arr)"/>
    <text x="533" y="88" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#a86f00">c = 1</text>
    <circle cx="355" cy="48" r="8" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.4"/>
    <circle cx="355" cy="100" r="8" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.4"/>
    <circle cx="355" cy="152" r="8" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.4"/>
    <circle cx="475" cy="100" r="14" fill="#fce4ec" stroke="#c2185b" stroke-width="1.6"/>
    <text x="475" y="105" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#c2185b">r</text>
    <circle cx="591" cy="100" r="14" fill="#fef2f2" stroke="#b91c1c" stroke-width="2"/>
    <text x="591" y="105" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#b91c1c">t</text>
    <text x="475" y="190" text-anchor="middle" font-family="serif" font-size="11" fill="#666">at most 1 unit can leave r</text>
  </g>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Why unit capacities encode matchings: the single edge $s\ell$ of capacity $1$ lets at most one unit pass through each $\ell \in L$, and the single edge $rt$ lets at most one unit pass through each $r \in R$ — so no vertex can be matched twice.
</figcaption>
</figure>

Since all capacities are $0/1$, Ford–Fulkerson finds this flow in time $O(nm)$. Next week we will see how to do it in time $O(\sqrt{n} \cdot m)$.

# Lecture 2: Dinitz's Algorithm

Edmonds–Karp improves Ford–Fulkerson by always augmenting along a *shortest* augmenting path. Dinitz's algorithm pushes the idea further: instead of finding shortest paths one at a time, each round computes **all** shortest augmenting paths at once — a *blocking flow* in a *layered* version of the residual network — and the shortest-path distance from $s$ to $t$ then provably grows from round to round. This yields the $O(n^2 m)$ bound promised last time.

## Net Flows and the Residual Network

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Convention</span><span class="math-callout__name">(Symmetric Networks)</span></p>

From now on we assume WLOG that the edges of the network come in opposite pairs: whenever $uv \in E$, also $vu \in E$ (add the missing opposite edges with capacity $0$). This costs nothing and lets us treat quantities like the residual capacity as functions on ordered pairs of adjacent vertices.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Net Flow)</span></p>

For a flow $f$ the **net flow** of the pair $uv$ is

$$
f^{*}(uv) := f(uv) - f(vu).
$$

It satisfies:

1. **antisymmetry**: $f^\ast(uv) = -f^\ast(vu)$,
2. **capacity bounds**: $-c(vu) \le f^\ast(uv) \le c(uv)$,
3. **Kirchhoff's law in net form**: the excess can be computed as
   $$
   f^{\Delta}(v) = \sum_{uv \in E} f^{*}(uv),
   $$
   so $f^{\Delta}(v) = 0$ for all $v \neq s, t$, and $\lvert f\rvert = f^{\Delta}(t)$.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 560 180" width="100%" style="max-width: 480px; height: auto;" role="img" aria-labelledby="gaf10-title">
  <title id="gaf10-title">Net flow of an opposite pair of edges</title>
  <defs>
    <marker id="gaf10-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
    </marker>
  </defs>
  <path d="M 186,79 C 240,52 320,52 374,79" fill="none" stroke="#333" stroke-width="1.8" marker-end="url(#gaf10-arr)"/>
  <path d="M 374,101 C 320,128 240,128 186,101" fill="none" stroke="#333" stroke-width="1.8" marker-end="url(#gaf10-arr)"/>
  <text x="280" y="42" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#333">f(uv) = 3,  c(uv) = 4</text>
  <text x="280" y="152" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#333">f(vu) = 1,  c(vu) = 2</text>
  <text x="280" y="95" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#a86f00">f*(uv) = 3 − 1 = 2</text>
  <circle cx="170" cy="90" r="16" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="170" y="95" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#1565c0">u</text>
  <circle cx="390" cy="90" r="16" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="390" y="95" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#1565c0">v</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
An opposite pair with its net flow: $f^\ast(uv) = f(uv) - f(vu) = 2 = -f^\ast(vu)$. The residual capacities become $r(uv) = c(uv) - f^\ast(uv) = 2$ and $r(vu) = c(vu) + f^\ast(uv) = 4$.
</figcaption>
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Flows $\leftrightarrow$ Net Flows)</span></p>

Conversely, every function satisfying 1–3 is the net flow of some flow: given such an $f^\ast$, consider each opposite pair and WLOG $f^\ast(uv) > 0$; setting $f(uv) := f^\ast(uv)$ and $f(vu) := 0$ yields a flow with the prescribed net flow. So flows and net flows are two views of the same object, and we may switch freely between them.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Residual Network)</span></p>

The **residual network** of the network $G$ with a flow $f$ is

$$
R := (V, E, s, t, r),
$$

i.e. the same graph with the capacities replaced by the residual capacities of Lecture 1, which in terms of the net flow read

$$
r(uv) = c(uv) - f(uv) + f(vu) = c(uv) - f^{*}(uv).
$$

</div>

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Improving a Flow Using a Flow)</span></p>

Let $f$ be a flow in $G$ and let $g$ be a flow in the residual network $R$. Then there is a flow $f'$ in $G$ with

$$
|f'| = |f| + |g|.
$$

Conversely, for any two flows $f, h$ in $G$ there is a flow $g$ in $R$ with $\lvert g\rvert = \lvert h\rvert - \lvert f\rvert$ — in particular, taking $h$ maximum: the residual network always carries a flow of size $\lvert f\_{\max}\rvert - \lvert f\rvert$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Work with net flows and set $F := f^\ast + g^\ast$; we check that $F$ satisfies properties 1–3.

Antisymmetry is preserved under addition. For the capacity bound, $g$ respects the residual capacities, so

$$
F(uv) = f^{*}(uv) + g^{*}(uv) \le f^{*}(uv) + g(uv) \le f^{*}(uv) + r(uv) = c(uv),
$$

and the lower bound $F(uv) \ge -c(vu)$ follows by applying the same estimate to $vu$ and using antisymmetry. Finally, excesses simply add,

$$
F^{\Delta}(v) = f^{\Delta}(v) + g^{\Delta}(v),
$$

which vanishes for $v \neq s, t$ and gives $F^{\Delta}(t) = \lvert f\rvert + \lvert g\rvert$. By the observation above, $F$ is the net flow of a flow $f'$ in $G$ with $\lvert f'\rvert = \lvert f\rvert + \lvert g\rvert$.

For the converse, set $F := h^\ast - f^\ast$. It is antisymmetric, its excess vanishes off $s, t$ with $F^{\Delta}(t) = \lvert h\rvert - \lvert f\rvert$, and

$$
F(uv) = h^{*}(uv) - f^{*}(uv) \le c(uv) - f^{*}(uv) = r(uv),
$$

so $F$ is (the net flow of) a flow in $R$ of size $\lvert h\rvert - \lvert f\rvert$.

</details>
</div>

## Blocking Flows and the Layered Network

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Blocking Flow)</span></p>

A flow $g$ in a network is **blocking** if every $st$-path contains at least one **saturated** edge, i.e. an edge with $g(e) = c(e)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Blocking $\neq$ Maximum)</span></p>

Every maximum flow is blocking, but not conversely: the flow at which the naive algorithm of Lecture 1 got stuck is exactly a blocking flow that is not maximum. A blocking flow only rules out augmenting along *forward* edges — this is precisely what the naive greedy rule produces, and it is all Dinitz's algorithm will need from one round.

</div>

**Layering.** Let $R$ be the residual network, with the zero-residual edges already discarded. Run BFS from $s$: this partitions the reachable vertices into **layers** $L\_0 = \lbrace s \rbrace, L\_1, L\_2, \dots$ by their distance from $s$, and computes $\ell := $ the distance from $s$ to $t$. Now delete everything that cannot lie on a shortest $st$-path: layers behind $t$, edges that do not advance exactly one layer, and then repeatedly **dead ends** — vertices other than $t$ with no outgoing edge (and vertices other than $s$ with no incoming edge), together with their remaining edges. What survives is the **layered network**: every remaining vertex and edge lies on a shortest $st$-path. The BFS and the cleanup both cost $O(m)$.

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 265" width="100%" style="max-width: 620px; height: auto;" role="img" aria-labelledby="gaf11-title">
  <title id="gaf11-title">The layered network with a dead end removed by the cleanup</title>
  <defs>
    <marker id="gaf11-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
    </marker>
    <marker id="gaf11-arrf" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#888"/>
    </marker>
  </defs>
  <ellipse cx="180" cy="130" rx="32" ry="78" fill="none" stroke="#888" stroke-width="1.2" stroke-dasharray="5 4"/>
  <ellipse cx="300" cy="130" rx="32" ry="78" fill="none" stroke="#888" stroke-width="1.2" stroke-dasharray="5 4"/>
  <ellipse cx="420" cy="130" rx="32" ry="78" fill="none" stroke="#888" stroke-width="1.2" stroke-dasharray="5 4"/>
  <line x1="69" y1="124" x2="171" y2="84" stroke="#333" stroke-width="1.5" marker-end="url(#gaf11-arr)"/>
  <line x1="71" y1="130" x2="171" y2="130" stroke="#333" stroke-width="1.5" marker-end="url(#gaf11-arr)"/>
  <line x1="69" y1="136" x2="171" y2="176" stroke="#333" stroke-width="1.5" marker-end="url(#gaf11-arr)"/>
  <line x1="188" y1="77" x2="291" y2="58" stroke="#888" stroke-width="1.4" stroke-dasharray="4 3" marker-end="url(#gaf11-arrf)"/>
  <line x1="187" y1="84" x2="292" y2="112" stroke="#333" stroke-width="1.5" marker-end="url(#gaf11-arr)"/>
  <line x1="187" y1="133" x2="292" y2="172" stroke="#333" stroke-width="1.5" marker-end="url(#gaf11-arr)"/>
  <line x1="187" y1="180" x2="292" y2="177" stroke="#333" stroke-width="1.5" marker-end="url(#gaf11-arr)"/>
  <line x1="307" y1="113" x2="412" y2="92" stroke="#333" stroke-width="1.5" marker-end="url(#gaf11-arr)"/>
  <line x1="307" y1="117" x2="412" y2="148" stroke="#333" stroke-width="1.5" marker-end="url(#gaf11-arr)"/>
  <line x1="307" y1="173" x2="412" y2="152" stroke="#333" stroke-width="1.5" marker-end="url(#gaf11-arr)"/>
  <line x1="427" y1="92" x2="542" y2="124" stroke="#333" stroke-width="1.5" marker-end="url(#gaf11-arr)"/>
  <line x1="427" y1="148" x2="542" y2="136" stroke="#333" stroke-width="1.5" marker-end="url(#gaf11-arr)"/>
  <circle cx="180" cy="80" r="7" fill="#fff" stroke="#1565c0" stroke-width="1.4"/>
  <circle cx="180" cy="130" r="7" fill="#fff" stroke="#1565c0" stroke-width="1.4"/>
  <circle cx="180" cy="180" r="7" fill="#fff" stroke="#1565c0" stroke-width="1.4"/>
  <circle cx="300" cy="55" r="7" fill="#fff" stroke="#b91c1c" stroke-width="1.4"/>
  <line x1="291" y1="46" x2="309" y2="64" stroke="#b91c1c" stroke-width="2"/>
  <line x1="291" y1="64" x2="309" y2="46" stroke="#b91c1c" stroke-width="2"/>
  <circle cx="300" cy="115" r="7" fill="#fff" stroke="#1565c0" stroke-width="1.4"/>
  <circle cx="300" cy="175" r="7" fill="#fff" stroke="#1565c0" stroke-width="1.4"/>
  <circle cx="420" cy="90" r="7" fill="#fff" stroke="#1565c0" stroke-width="1.4"/>
  <circle cx="420" cy="150" r="7" fill="#fff" stroke="#1565c0" stroke-width="1.4"/>
  <circle cx="55" cy="130" r="15" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="55" y="135" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#0f9b6c">s</text>
  <circle cx="557" cy="130" r="15" fill="#fef2f2" stroke="#b91c1c" stroke-width="2"/>
  <text x="557" y="135" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#b91c1c">t</text>
  <text x="300" y="26" text-anchor="middle" font-family="serif" font-size="11" fill="#b91c1c">dead end — removed by the cleanup</text>
  <text x="55" y="168" text-anchor="middle" font-family="serif" font-size="12" fill="#666">L₀</text>
  <text x="180" y="232" text-anchor="middle" font-family="serif" font-size="12" fill="#666">L₁</text>
  <text x="300" y="232" text-anchor="middle" font-family="serif" font-size="12" fill="#666">L₂</text>
  <text x="420" y="232" text-anchor="middle" font-family="serif" font-size="12" fill="#666">L₃</text>
  <text x="557" y="168" text-anchor="middle" font-family="serif" font-size="12" fill="#666">L₄</text>
  <text x="320" y="258" text-anchor="middle" font-family="serif" font-size="11" fill="#666">every edge advances exactly one layer; ℓ = 4</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The layered network: BFS layers $L_0, \dots, L_\ell$ of the residual network, keeping only edges that advance one layer. The crossed-out vertex has no outgoing edge — a dead end; the cleanup deletes it together with its incoming (dashed) edge, and repeats until every vertex and edge lies on a shortest $st$-path.
</figcaption>
</figure>


## Dinitz's Algorithm

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Dinitz)</span></p>

0. $f \leftarrow$ everywhere-zero flow.
1. **Repeat:**
2. &nbsp;&nbsp;&nbsp;Construct the residual network $R$. — $O(m)$
3. &nbsp;&nbsp;&nbsp;Delete all edges $e \in R$ with $r(e) = 0$. — $O(m)$
4. &nbsp;&nbsp;&nbsp;$\ell \leftarrow$ distance from $s$ to $t$ in $R$. — $O(m)$
5. &nbsp;&nbsp;&nbsp;If $\ell = +\infty$, stop. — $O(1)$
6. &nbsp;&nbsp;&nbsp;Clean $R$ up into the layered network. — $O(m)$
7. &nbsp;&nbsp;&nbsp;$g \leftarrow$ a blocking flow in the layered network. — $O(nm)$
8. &nbsp;&nbsp;&nbsp;Improve $f$ using $g$ (via the improving lemma). — $O(m)$

One iteration of the loop is called a **phase**; a phase costs $O(nm)$, dominated by the blocking-flow computation.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Greedy Blocking Flow)</span></p>

In the layered network (capacities $r$):

1. $g \leftarrow$ everywhere-zero.
2. **While** there is an $st$-path $P$ (walk forward from $s$, in $O(n)$):
3. &nbsp;&nbsp;&nbsp;$\varepsilon \leftarrow \min\_{e \in P} \bigl(r(e) - g(e)\bigr)$.
4. &nbsp;&nbsp;&nbsp;For all $e \in P$: $g(e) \mathrel{+}= \varepsilon$.
5. &nbsp;&nbsp;&nbsp;For all $e \in P$: if $g(e) = r(e)$, delete $e$; clean up any dead ends this creates.
6. All the cleanup together costs $O(m)$.

Each pass of the while loop saturates (and deletes) at least one edge — the minimizer — and edges never return, so there are at most $m$ passes of $O(n)$ each: the blocking flow is computed in time $O(nm)$. The forward walk can hit a dead end only when a deletion just created one; removing such edges is charged to the global $O(m)$ cleanup budget.

</div>

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Progress)</span></p>

Between two consecutive phases, $\ell$ increases by at least $1$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Fix the BFS levels from the start of the phase: $\mathrm{level}(v) :=$ distance from $s$ to $v$ in $R$. Every edge $uv$ of $R$ satisfies $\mathrm{level}(v) \le \mathrm{level}(u) + 1$ — BFS cannot skip a layer. The phase changes the residual network in two ways: it removes the edges saturated by the blocking flow, and it creates reversals of edges that carried flow — and every created edge goes exactly one layer *back*.

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 225" width="100%" style="max-width: 620px; height: auto;" role="img" aria-labelledby="gaf12-title">
  <title id="gaf12-title">Phase i+1 seen in the layers of phase i</title>
  <defs>
    <marker id="gaf12-arrb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#1565c0"/>
    </marker>
    <marker id="gaf12-arrr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#b91c1c"/>
    </marker>
  </defs>
  <text x="320" y="30" text-anchor="middle" font-family="serif" font-size="11" fill="#b91c1c">new residual edges point one layer back</text>
  <ellipse cx="120" cy="130" rx="26" ry="60" fill="none" stroke="#888" stroke-width="1.2" stroke-dasharray="5 4"/>
  <ellipse cx="240" cy="130" rx="26" ry="60" fill="none" stroke="#888" stroke-width="1.2" stroke-dasharray="5 4"/>
  <ellipse cx="360" cy="130" rx="26" ry="60" fill="none" stroke="#888" stroke-width="1.2" stroke-dasharray="5 4"/>
  <ellipse cx="480" cy="130" rx="26" ry="60" fill="none" stroke="#888" stroke-width="1.2" stroke-dasharray="5 4"/>
  <line x1="59" y1="130" x2="91" y2="130" stroke="#1565c0" stroke-width="2.4" marker-end="url(#gaf12-arrb)"/>
  <line x1="149" y1="130" x2="211" y2="130" stroke="#1565c0" stroke-width="2.4" marker-end="url(#gaf12-arrb)"/>
  <line x1="269" y1="130" x2="331" y2="130" stroke="#1565c0" stroke-width="2.4" marker-end="url(#gaf12-arrb)"/>
  <line x1="389" y1="130" x2="451" y2="130" stroke="#1565c0" stroke-width="2.4" marker-end="url(#gaf12-arrb)"/>
  <line x1="509" y1="130" x2="571" y2="130" stroke="#1565c0" stroke-width="2.4" marker-end="url(#gaf12-arrb)"/>
  <path d="M 335,102 C 315,60 285,60 265,98" fill="none" stroke="#b91c1c" stroke-width="1.8" stroke-dasharray="5 4" marker-end="url(#gaf12-arrr)"/>
  <path d="M 455,102 C 435,60 405,60 385,98" fill="none" stroke="#b91c1c" stroke-width="1.8" stroke-dasharray="5 4" marker-end="url(#gaf12-arrr)"/>
  <circle cx="45" cy="130" r="14" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="45" y="135" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#0f9b6c">s</text>
  <circle cx="586" cy="130" r="14" fill="#fef2f2" stroke="#b91c1c" stroke-width="2"/>
  <text x="586" y="135" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#b91c1c">t</text>
  <text x="320" y="215" text-anchor="middle" font-family="serif" font-size="11" fill="#666">forward edges advance one layer; every new s–t path has length ≥ ℓ + 2</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Phase $i+1$ seen in the layers of phase $i$: the blocking flow saturates every all-forward path (blue), and the only edges it creates point one layer back (dashed red) — so any path that uses them must waste at least two steps.
</figcaption>
</figure>

Now take any $st$-path in the new residual network and read the levels along it: every step raises the level by at most $1$, steps along created edges lower it by $1$, and the path must climb from level $0$ to $\mathrm{level}(t) = \ell$. If the path had length $\ell$, every step would have to raise the level by exactly $1$; in particular it uses no created edge, so all its edges already existed in the old $R$ — making it a shortest $st$-path of the old $R$, contained in the cleaned layered network. But the blocking flow saturated at least one edge on every such path, and that edge is gone from the new residual network — a contradiction.

Hence every $st$-path of the new residual network is longer than $\ell$ (paths through a created edge even have length $\ge \ell + 2$), so $\ell$ increases by at least $1$ per phase.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Dinitz)</span></p>

Dinitz's algorithm finds a maximum flow in time $O(n^2 m)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The distance $\ell$ is the length of a path, so $1 \le \ell \le n - 1$; since it strictly increases between phases, there are at most $n$ phases, each costing $O(nm)$. When the algorithm stops, $t$ is unreachable in the residual network, i.e. there is no augmenting path — so $f$ is maximum by the max-flow min-cut argument of Lecture 1.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Sparse vs. Dense)</span></p>

For a sparse graph ($m \sim n$) the bound reads $O(n^3)$; for a dense one ($m \sim n^2$) it becomes $O(n^4)$. The improvements below attack both regimes.

</div>

## Special Networks

On networks with small integer capacities, Dinitz's algorithm is much faster than $O(n^2 m)$ — both the blocking flows and the number of phases become cheap.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Special Networks)</span></p>

1. If $c \equiv 1$: a blocking flow costs only $O(m)$ — each augmentation saturates and deletes *every* edge of its path, so the total length of all paths is at most $m$. With at most $n$ phases: $O(nm)$.
2. If $c \equiv 1$: there are only $O(\sqrt{m})$ phases, so Dinitz runs in time $O(m^{3/2})$.
3. If $c \equiv 1$ and $\min(\deg^{\mathrm{in}}(v), \deg^{\mathrm{out}}(v)) \le 1$ for every $v \neq s, t$ (*unit networks*): only $O(\sqrt{n})$ phases, so $O(\sqrt{n} \cdot m)$.
4. If $c \equiv 1$ and the graph is simple: only $O(n^{2/3})$ phases, so $O(n^{2/3} \cdot m)$. These bounds are tight.
5. If $c$ is integer: all the blocking flows together cost $O(n \cdot \lvert f\_{\max}\rvert)$ apart from cleanup — every pass of the greedy algorithm pushes at least one unit — plus $O(m)$ per phase, giving $O(n \lvert f\_{\max}\rvert + nm)$ in total.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of the phase bounds (2)–(4)</summary>

All three bounds follow from one stopping argument. Run Dinitz for $k$ phases and look at the current flow $f$ and residual network $R$; by the progress lemma the layering of $R$ now has $\ell \ge k + 1$, so the layers $L\_0 = \lbrace s\rbrace, L\_1, \dots$ up to the layer of $t$ number more than $k$. By the improving lemma (converse direction), $R$ carries a flow $g$ with $\lvert g\rvert = \lvert f\_{\max}\rvert - \lvert f\rvert$; and since with integer capacities every remaining phase increases $\lvert f\rvert$ by at least $1$,

$$
\#\text{remaining phases} \;\le\; |f_{\max}| - |f| \;=\; |g| \;\le\; r(C)
$$

for *every* cut $C$ in $R$, by the flows-vs.-cuts corollary of Lecture 1. It remains to find a small cut among the between-layer cuts $E(L\_i, L\_{i+1})$ — no residual edge can jump a layer forward, so each of these separates $s$ from $t$.

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 260" width="100%" style="max-width: 620px; height: auto;" role="img" aria-labelledby="gaf13-title">
  <title id="gaf13-title">After k phases: many layers, so some between-layer cut is small</title>
  <defs>
    <marker id="gaf13-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
    </marker>
  </defs>
  <text x="320" y="24" text-anchor="middle" font-family="serif" font-size="11" fill="#333">after k phases: ℓ ≥ k + 1 layers</text>
  <ellipse cx="110" cy="130" rx="22" ry="68" fill="none" stroke="#888" stroke-width="1.2" stroke-dasharray="5 4"/>
  <ellipse cx="200" cy="130" rx="22" ry="68" fill="none" stroke="#888" stroke-width="1.2" stroke-dasharray="5 4"/>
  <ellipse cx="300" cy="130" rx="22" ry="68" fill="none" stroke="#888" stroke-width="1.2" stroke-dasharray="5 4"/>
  <ellipse cx="400" cy="130" rx="22" ry="68" fill="none" stroke="#888" stroke-width="1.2" stroke-dasharray="5 4"/>
  <ellipse cx="490" cy="130" rx="22" ry="68" fill="none" stroke="#888" stroke-width="1.2" stroke-dasharray="5 4"/>
  <circle cx="296" cy="100" r="3.5" fill="#666"/>
  <circle cx="302" cy="130" r="3.5" fill="#666"/>
  <circle cx="297" cy="160" r="3.5" fill="#666"/>
  <circle cx="398" cy="95" r="3.5" fill="#666"/>
  <circle cx="403" cy="132" r="3.5" fill="#666"/>
  <circle cx="398" cy="165" r="3.5" fill="#666"/>
  <line x1="306" y1="101" x2="390" y2="96" stroke="#333" stroke-width="1.4" marker-end="url(#gaf13-arr)"/>
  <line x1="308" y1="131" x2="395" y2="132" stroke="#333" stroke-width="1.4" marker-end="url(#gaf13-arr)"/>
  <line x1="303" y1="161" x2="390" y2="164" stroke="#333" stroke-width="1.4" marker-end="url(#gaf13-arr)"/>
  <path d="M 350,52 L 358,78 L 344,104 L 358,130 L 344,156 L 358,182 L 350,208" fill="none" stroke="#a86f00" stroke-width="2.5"/>
  <text x="366" y="52" font-family="serif" font-size="14" font-style="italic" fill="#a86f00">C</text>
  <circle cx="40" cy="130" r="14" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="40" y="135" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#0f9b6c">s</text>
  <circle cx="580" cy="130" r="14" fill="#fef2f2" stroke="#b91c1c" stroke-width="2"/>
  <text x="580" y="135" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#b91c1c">t</text>
  <text x="300" y="226" text-anchor="middle" font-family="serif" font-size="12" fill="#666">Lᵢ</text>
  <text x="400" y="226" text-anchor="middle" font-family="serif" font-size="12" fill="#666">Lᵢ₊₁</text>
  <line x1="88" y1="243" x2="512" y2="243" stroke="#888" stroke-width="1"/>
  <line x1="88" y1="238" x2="88" y2="248" stroke="#888" stroke-width="1"/>
  <line x1="512" y1="238" x2="512" y2="248" stroke="#888" stroke-width="1"/>
  <text x="300" y="258" text-anchor="middle" font-family="serif" font-size="11" fill="#666">≥ k layers between s and t</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Stopping after $k$ phases: the residual network has at least $k+1$ layers, every residual edge advances at most one layer, and by pigeonhole two consecutive layers $L_i, L_{i+1}$ are small — so the between-layer cut $C$ (orange) has small residual capacity, bounding the number of remaining phases.
</figcaption>
</figure>


**(4) Simple graphs.** Set $p\_i := \lvert L\_i\rvert + \lvert L\_{i+1}\rvert$. Each vertex is counted at most twice, so $\sum\_i p\_i \le 2n$, and by pigeonhole some $i \le k$ has $p\_i \le 2n/k$. In a simple graph there is at most one edge per ordered pair, hence with unit capacities

$$
r\bigl(E(L_i, L_{i+1})\bigr) \;\le\; |L_i| \cdot |L_{i+1}| \;\le\; \Bigl(\frac{p_i}{2}\Bigr)^2 \le \Bigl(\frac{n}{k}\Bigr)^2 = \frac{n^2}{k^2}.
$$

So the total number of phases is at most $k + n^2/k^2$; choosing $k$ with $k = n^2/k^2$, i.e. $k = n^{2/3}$, gives $O(n^{2/3})$ phases.

**(2) Arbitrary multigraphs.** The edge sets $E(L\_i, L\_{i+1})$ are pairwise disjoint, so some $i \le k$ has at most $m/k$ edges between $L\_i$ and $L\_{i+1}$; with unit capacities $r(C) \le m/k$, and the phase count is at most $k + m/k = O(\sqrt{m})$ for $k = \sqrt{m}$.

**(3) Unit networks.** Some layer $L\_i$ with $0 < i \le k$ has $\lvert L\_i\rvert \le n/k$. Since every inner vertex has in-degree or out-degree at most $1$ and $c \equiv 1$, at most one unit of any flow in $R$ can pass through each vertex of $L\_i$, and every unit must pass through $L\_i$; hence $\lvert g\rvert \le n/k$, and the phase count is at most $k + n/k = O(\sqrt{n})$ for $k = \sqrt{n}$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Bipartite Matching, Fast)</span></p>

The matching network of Lecture 1 has unit capacities and every inner vertex has in-degree or out-degree exactly $1$, so it is a unit network: Dinitz's algorithm finds a maximum bipartite matching in time $O(\sqrt{n} \cdot m)$ — the bound promised last week.

</div>

## Improvements

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Faster Blocking Flows and Beyond)</span></p>

* **Three Indians method** (Malhotra, Kumar, Maheshwari): a better blocking-flow algorithm running in $O(n^2)$, which gives Dinitz in $O(n^3)$ — an improvement for dense graphs.
* **Sleator–Tarjan link-cut trees** (covered in Data Structures 2): a blocking flow in $O(m \log n)$, giving Dinitz in $O(nm \log n)$.
* The state of the art for max flow along these lines: **Orlin's algorithm**, running in $O(nm)$.

</div>

# Lecture 3: Capacity Scaling, Cuts and Connectivity

Dinitz's bound $O(n \lvert f\_{\max}\rvert + nm)$ from the special-networks theorem is excellent when the maximum flow is small — but with large integer capacities $\lvert f\_{\max}\rvert$ can be huge. *Capacity scaling* fixes this by feeding the capacities to Dinitz one binary digit at a time. The rest of the lecture harvests what max-flow theory says about *cuts*: Menger's theorems on edge- and vertex-connectivity, and a completely different, randomized approach to minimum cuts by edge contraction.

## Capacity Scaling

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Capacity Scaling)</span></p>

Let $G$ be a network with integer capacities $c \in \lbrace 0, \dots, C \rbrace$ and let $\ell := \lfloor \log C \rfloor + 1$ be the number of bits. Define networks $G\_0, G\_1, \dots, G\_\ell$, all on the same graph, where $G\_i$ carries the **topmost $i$ bits** of every capacity:

$$
c_i(e) := \bigl\lfloor c(e) / 2^{\ell - i} \bigr\rfloor,
\qquad\text{so}\qquad
c_{i+1}(e) =
\begin{cases}
2 c_i(e), \\
2 c_i(e) + 1,
\end{cases}
$$

depending on the next bit; $G\_0$ has all capacities zero and $G\_\ell = G$. Compute maximum flows $f\_i$ of the $G\_i$ in order:

1. $f\_0 \leftarrow 0$.
2. For $i = 0, \dots, \ell - 1$: note that $2 f\_i$ is a flow in $G\_{i+1}$ — capacities at least doubled — and improve it to a maximum flow $f\_{i+1}$ of $G\_{i+1}$ by running Dinitz's algorithm on the residual network. — $O(nm)$ per level, by the lemma below.
3. Return $f\_\ell$.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 560 165" width="100%" style="max-width: 500px; height: auto;" role="img" aria-labelledby="gaf14-title">
  <title id="gaf14-title">Capacity scaling reveals the capacities bit by bit</title>
  <defs>
    <marker id="gaf14-arrg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#0f9b6c"/>
    </marker>
  </defs>
  <text x="244" y="28" text-anchor="middle" font-family="serif" font-size="11" fill="#666">binary expansion of c(e) — ℓ bits</text>
  <rect x="100" y="42" width="36" height="30" fill="#fff7e0" stroke="#a86f00" stroke-width="1.6"/>
  <rect x="136" y="42" width="36" height="30" fill="#fff7e0" stroke="#a86f00" stroke-width="1.6"/>
  <rect x="172" y="42" width="36" height="30" fill="#fff7e0" stroke="#a86f00" stroke-width="1.6"/>
  <rect x="208" y="42" width="36" height="30" fill="#fff" stroke="#888" stroke-width="1.2"/>
  <rect x="244" y="42" width="36" height="30" fill="#fff" stroke="#888" stroke-width="1.2"/>
  <rect x="280" y="42" width="36" height="30" fill="#fff" stroke="#888" stroke-width="1.2"/>
  <rect x="316" y="42" width="36" height="30" fill="#fff" stroke="#888" stroke-width="1.2"/>
  <rect x="352" y="42" width="36" height="30" fill="#fff" stroke="#888" stroke-width="1.2"/>
  <text x="118" y="62" text-anchor="middle" font-family="serif" font-size="13" fill="#333">1</text>
  <text x="154" y="62" text-anchor="middle" font-family="serif" font-size="13" fill="#333">0</text>
  <text x="190" y="62" text-anchor="middle" font-family="serif" font-size="13" fill="#333">1</text>
  <text x="226" y="62" text-anchor="middle" font-family="serif" font-size="13" fill="#333">1</text>
  <text x="262" y="62" text-anchor="middle" font-family="serif" font-size="13" fill="#333">0</text>
  <text x="298" y="62" text-anchor="middle" font-family="serif" font-size="13" fill="#333">1</text>
  <text x="334" y="62" text-anchor="middle" font-family="serif" font-size="13" fill="#333">0</text>
  <text x="370" y="62" text-anchor="middle" font-family="serif" font-size="13" fill="#333">1</text>
  <line x1="100" y1="84" x2="208" y2="84" stroke="#a86f00" stroke-width="1.4"/>
  <line x1="100" y1="79" x2="100" y2="89" stroke="#a86f00" stroke-width="1.4"/>
  <line x1="208" y1="79" x2="208" y2="89" stroke="#a86f00" stroke-width="1.4"/>
  <text x="154" y="105" text-anchor="middle" font-family="serif" font-size="11.5" fill="#a86f00">cᵢ(e) — topmost i bits</text>
  <line x1="226" y1="135" x2="226" y2="80" stroke="#0f9b6c" stroke-width="1.6" marker-end="url(#gaf14-arrg)"/>
  <text x="320" y="150" text-anchor="middle" font-family="serif" font-size="11" fill="#0f9b6c">cᵢ₊₁ appends the next bit: 2cᵢ(e) or 2cᵢ(e) + 1</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Scaling reveals every capacity bit by bit: $c_i(e) = \lfloor c(e)/2^{\ell-i} \rfloor$ keeps the topmost $i$ bits, and moving to $G_{i+1}$ doubles it and appends the next bit.
</figcaption>
</figure>

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(One Scaling Step Is Cheap)</span></p>

$$
|f_{i+1}| - 2\,|f_i| \;\le\; m .
$$

Consequently the improvement in step 2 pushes at most $m$ additional units of flow, and Dinitz's algorithm started from $2f\_i$ finishes in time $O(nm)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Since $f\_i$ is maximum in $G\_i$, the max-flow min-cut theorem provides a cut $R$ with $\lvert f\_i\rvert = c\_i(R)$. Use the *same* cut in $G\_{i+1}$: every edge capacity at most doubles and gains $1$, and the cut has at most $m$ edges, so

$$
|f_{i+1}| \;\le\; c_{i+1}(R) \;\le\; 2\,c_i(R) + m \;=\; 2\,|f_i| + m .
$$

The improvement therefore closes a gap of $\Delta f \le m$ units. As in the special-networks theorem (case 5), every pass of the greedy blocking-flow algorithm pushes at least one unit, so all the passes together cost $O(n \cdot \Delta f) = O(nm)$, plus $O(m)$ bookkeeping per phase over at most $n$ phases — in total $O(nm)$ per level. (In general: Dinitz started from a flow whose value is $\Delta f$ below the maximum runs in $O(nm + n\,\Delta f)$.)

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Capacity Scaling)</span></p>

A maximum flow in a network with integer capacities $c \le C$ can be computed in time

$$
O(nm \cdot \log C)
$$

— $\ell = O(\log C)$ levels at $O(nm)$ each.

</div>

## Cuts and Edge Connectivity

So far a "cut" always meant an edge set of the form $E(A, \overline{A})$. For connectivity questions a more liberal notion is convenient:

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Cuts, Edge Connectivity)</span></p>

In a directed or undirected graph $G$ with $s, t \in V$, $s \neq t$:

* $C \subseteq E$ is an **$st$-cut** $\equiv$ $G - C$ contains no $st$-path;
* $C \subseteq E$ is a **cut** $\equiv$ there exist $s \neq t$ such that $C$ is an $st$-cut;
* an undirected $G$ is **$k$-edge-connected** $\equiv$ every cut has size at least $k$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Elementary Cuts, Small Cuts)</span></p>

An **elementary** cut is one of the form $C = E(A, B)$ for a partition $V = A \mathbin{\dot\cup} B$ with $s \in A$, $t \in B$ — the "flow cuts" of Lecture 1. Every *minimum* cut is elementary (removing a minimal cut splits the graph into two sides), but a general $st$-cut need not be. Small cases: cuts of size $1$ are exactly the **bridges**; $1$-edge-connected $\Leftrightarrow$ connected; $2$-edge-connected $\Leftrightarrow$ connected and bridgeless.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Menger — Edge Version)</span></p>

The maximum number of pairwise edge-disjoint $st$-paths equals the size of a minimum $st$-cut.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (algorithmic, via flows)</summary>

"$\le$": every $st$-path must use at least one edge of every $st$-cut, and edge-disjoint paths use distinct ones — so there cannot be more paths than cut edges.

"$\ge$": turn $G$ into a network with $c \equiv 1$ (an undirected edge becomes the pair of opposite directed edges) and compute an *integer* maximum flow $f$ with Dinitz — a unit-capacity simple network, so this costs only $O(n^{2/3} \cdot m)$. By max-flow min-cut, $\lvert f\rvert$ equals the capacity of a minimum elementary cut, which is a minimum $st$-cut here since every edge has capacity $1$. It remains to turn $f$ into $\lvert f\rvert$ edge-disjoint paths — see the decomposition below.

</details>
</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 250" width="100%" style="max-width: 600px; height: auto;" role="img" aria-labelledby="gaf15-title">
  <title id="gaf15-title">Menger: edge-disjoint paths versus a minimum st-cut</title>
  <defs>
    <marker id="gaf15-arrg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#0f9b6c"/>
    </marker>
    <marker id="gaf15-arrb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#1565c0"/>
    </marker>
    <marker id="gaf15-arrp" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#7b1fa2"/>
    </marker>
  </defs>
  <ellipse cx="310" cy="130" rx="285" ry="102" fill="none" stroke="#888" stroke-width="1.2" stroke-dasharray="6 5"/>
  <line x1="84" y1="123" x2="172" y2="79" stroke="#0f9b6c" stroke-width="2" marker-end="url(#gaf15-arrg)"/>
  <line x1="188" y1="75" x2="422" y2="75" stroke="#0f9b6c" stroke-width="2" marker-end="url(#gaf15-arrg)"/>
  <line x1="438" y1="79" x2="528" y2="123" stroke="#0f9b6c" stroke-width="2" marker-end="url(#gaf15-arrg)"/>
  <line x1="86" y1="130" x2="172" y2="130" stroke="#1565c0" stroke-width="2" marker-end="url(#gaf15-arrb)"/>
  <line x1="186" y1="130" x2="422" y2="130" stroke="#1565c0" stroke-width="2" marker-end="url(#gaf15-arrb)"/>
  <line x1="436" y1="130" x2="526" y2="130" stroke="#1565c0" stroke-width="2" marker-end="url(#gaf15-arrb)"/>
  <line x1="84" y1="137" x2="172" y2="181" stroke="#7b1fa2" stroke-width="2" marker-end="url(#gaf15-arrp)"/>
  <line x1="188" y1="185" x2="422" y2="185" stroke="#7b1fa2" stroke-width="2" marker-end="url(#gaf15-arrp)"/>
  <line x1="438" y1="181" x2="528" y2="137" stroke="#7b1fa2" stroke-width="2" marker-end="url(#gaf15-arrp)"/>
  <path d="M 310,32 L 319,58 L 301,84 L 319,110 L 301,136 L 319,162 L 301,188 L 319,214 L 310,232" fill="none" stroke="#b91c1c" stroke-width="2.2"/>
  <text x="334" y="30" font-family="serif" font-size="12" fill="#b91c1c">min st-cut, |C| = 3</text>
  <circle cx="180" cy="75" r="6" fill="#fff" stroke="#0f9b6c" stroke-width="1.6"/>
  <circle cx="430" cy="75" r="6" fill="#fff" stroke="#0f9b6c" stroke-width="1.6"/>
  <circle cx="180" cy="130" r="6" fill="#fff" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="430" cy="130" r="6" fill="#fff" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="180" cy="185" r="6" fill="#fff" stroke="#7b1fa2" stroke-width="1.6"/>
  <circle cx="430" cy="185" r="6" fill="#fff" stroke="#7b1fa2" stroke-width="1.6"/>
  <circle cx="70" cy="130" r="15" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="70" y="135" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#0f9b6c">s</text>
  <circle cx="542" cy="130" r="15" fill="#fef2f2" stroke="#b91c1c" stroke-width="2"/>
  <text x="542" y="135" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#b91c1c">t</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Menger's theorem: a system of three edge-disjoint $st$-paths certifies that every $st$-cut has size $\ge 3$, and the wavy minimum cut with $|C| = 3$ certifies that no fourth disjoint path exists.
</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Path Decomposition of an Integer Unit Flow)</span></p>

Given an integer flow $f$ in a unit-capacity network, walk greedily from $s$ along edges carrying flow:

* if the walk reaches $t$, we found an $st$-path — remove it from $f$, obtaining a flow $f'$ with $\lvert f'\rvert = \lvert f\rvert - 1$;
* if the walk revisits a vertex, we found a cycle — remove it, obtaining a flow $f'$ with $\lvert f'\rvert = \lvert f\rvert$;

and repeat. Every edge of the support is visited and removed once, so the whole decomposition runs in $O(m)$ and produces a system of $\lvert f\rvert$ edge-disjoint $st$-paths (plus some discarded cycles — a *circulation*).

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Global Edge Connectivity)</span></p>

For an undirected graph $G$, the largest $k$ such that $G$ is $k$-edge-connected equals the size of the smallest cut in $G$, and by Menger's theorem also the largest $k$ such that every pair $s, t$ is joined by $k$ edge-disjoint paths.

Algorithmically, the smallest cut can be found by min-$st$-cut computations:

1. Try all pairs $s, t$: $O(n^2)$ flow computations, $O(n^{2 + 2/3} \cdot m)$ in total.
2. Better — **fix $s$, try all $t$**: a minimum cut disconnects $s$ from *some* vertex $t$, so $n - 1$ computations suffice: $O(n^{1 + 2/3} \cdot m)$.

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Faster Global Min-Cut)</span></p>

Without flows one can do better: **Nagamochi–Ibaraki** find the smallest cut deterministically in $O(nm)$, and **Karger–Stein** do it with randomization even faster — the contraction idea behind their algorithm closes this lecture.

</div>

## Vertex Connectivity

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Separators, Vertex Connectivity)</span></p>

For a graph $G$ and vertices $s, t$:

* $U \subseteq V$ is an **$st$-separator** $\equiv$ $G - U$ contains no $st$-path and $s, t \notin U$;
* $U \subseteq V$ is a **separator** $\equiv$ there exist $s, t$ such that $U$ is an $st$-separator;
* an undirected $G$ is **$k$-(vertex-)connected** $\equiv$ every separator has size at least $k$, and $n > k$.

Note that an $st$-separator can exist only when $st \notin E$. Small cases: $1$-connected $\Leftrightarrow$ connected; $2$-connected $\Leftrightarrow$ connected with no articulation points.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Menger — Vertex Version)</span></p>

For $st \notin E$, the minimum size of an $st$-separator equals the maximum number of *internally vertex-disjoint* $st$-paths. (Analogous Mengerian theorems hold in all four combinations directed/undirected $\times$ edge/vertex.)

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Vertex Splitting)</span></p>

To find a minimum $st$-separator with flows, split every vertex $v$ into $v^{\mathrm{in}} \to v^{\mathrm{out}}$ joined by an edge of capacity $1$, and replace each edge $uv$ by $u^{\mathrm{out}} \to v^{\mathrm{in}}$ (and $v^{\mathrm{out}} \to u^{\mathrm{in}}$ if undirected). Flow units now correspond to internally vertex-disjoint paths, saturated split-edges to separator vertices. The result is a unit network, so Dinitz runs in $O(\sqrt{n} \cdot m)$.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 190" width="100%" style="max-width: 600px; height: auto;" role="img" aria-labelledby="gaf16-title">
  <title id="gaf16-title">Vertex splitting for vertex connectivity</title>
  <defs>
    <marker id="gaf16-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
    </marker>
    <marker id="gaf16-arrf" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#888"/>
    </marker>
  </defs>
  <line x1="45" y1="55" x2="126" y2="88" stroke="#333" stroke-width="1.5" marker-end="url(#gaf16-arr)"/>
  <line x1="45" y1="140" x2="126" y2="104" stroke="#333" stroke-width="1.5" marker-end="url(#gaf16-arr)"/>
  <line x1="155" y1="88" x2="236" y2="55" stroke="#333" stroke-width="1.5" marker-end="url(#gaf16-arr)"/>
  <line x1="155" y1="104" x2="236" y2="140" stroke="#333" stroke-width="1.5" marker-end="url(#gaf16-arr)"/>
  <circle cx="140" cy="96" r="16" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="140" y="101" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#1565c0">v</text>
  <line x1="278" y1="96" x2="330" y2="96" stroke="#888" stroke-width="1.6" stroke-dasharray="6 4" marker-end="url(#gaf16-arrf)"/>
  <text x="304" y="82" text-anchor="middle" font-family="serif" font-size="11" fill="#666">split</text>
  <line x1="360" y1="55" x2="406" y2="88" stroke="#333" stroke-width="1.5" marker-end="url(#gaf16-arr)"/>
  <line x1="360" y1="140" x2="406" y2="104" stroke="#333" stroke-width="1.5" marker-end="url(#gaf16-arr)"/>
  <line x1="434" y1="96" x2="506" y2="96" stroke="#333" stroke-width="2.2" marker-end="url(#gaf16-arr)"/>
  <text x="470" y="80" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#a86f00">c = 1</text>
  <line x1="534" y1="88" x2="586" y2="55" stroke="#333" stroke-width="1.5" marker-end="url(#gaf16-arr)"/>
  <line x1="534" y1="104" x2="586" y2="140" stroke="#333" stroke-width="1.5" marker-end="url(#gaf16-arr)"/>
  <circle cx="420" cy="96" r="14" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="420" y="101" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">v<tspan dy="-4" font-size="8">in</tspan></text>
  <circle cx="520" cy="96" r="14" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="520" y="101" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">v<tspan dy="-4" font-size="8">out</tspan></text>
  <text x="320" y="178" text-anchor="middle" font-family="serif" font-size="11" fill="#666">a separator vertex becomes a saturated unit edge</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Vertex splitting: $v$ becomes $v^{\mathrm{in}} \to v^{\mathrm{out}}$ with capacity $1$; incoming edges attach to $v^{\mathrm{in}}$, outgoing ones leave $v^{\mathrm{out}}$. Flow units then correspond to internally vertex-disjoint paths and cut split-edges to separator vertices.
</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Global Vertex Connectivity)</span></p>

To find a minimum separator of $G$:

1. Try all pairs $s, t$: $O(n^2 \cdot \sqrt{n}\, m) = O(n^{2.5} \cdot m)$.
2. Fix $s$, try all $t$: **broken!** Unlike the edge case, $s$ may belong to *every* minimum separator, and then no $t$ works with this $s$.
3. The fix: try $s\_1, s\_2, \dots$, each against all $t$, keeping the best separator found; **stop as soon as the best separator is smaller than the number of sources tried**. If the true minimum separator $U$ has $\lvert U\rvert = k$, then among any $k + 1$ tried sources one avoids $U$ and finds it — so the loop stops after at most $k + 1$ sources, in time $O(k \cdot n \cdot \sqrt{n}\, m) = O(k \cdot n^{1.5} \cdot m)$.

</div>

## Randomized Minimum Cut: Contraction

A completely different attack on the global minimum cut of an undirected **multigraph** — no flows at all, just random edge contractions.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Multigraph Contraction)</span></p>

$G/e$ arises from $G$ by merging the endpoints of $e$ into a single vertex, **keeping** parallel edges and **discarding** the loops this creates. Every edge of $G/e$ corresponds to an edge of $G$, and under this correspondence every cut of $G/e$ is a cut of $G$ of the same size — so

$$
\mathrm{mincut}(G) \;\le\; \mathrm{mincut}(G/e),
$$

with equality as long as $e$ does not lie in some fixed minimum cut: contracting can only *lose* those minimum cuts through $e$.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 210" width="100%" style="max-width: 580px; height: auto;" role="img" aria-labelledby="gaf17-title">
  <title id="gaf17-title">Multigraph contraction keeps parallel edges and discards loops</title>
  <defs>
    <marker id="gaf17-arrf" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#888"/>
    </marker>
  </defs>
  <path d="M 154,64 C 180,48 210,48 236,64" fill="none" stroke="#1565c0" stroke-width="2"/>
  <path d="M 154,78 C 180,94 210,94 236,78" fill="none" stroke="#1565c0" stroke-width="2"/>
  <line x1="148" y1="84" x2="184" y2="142" stroke="#333" stroke-width="1.6"/>
  <line x1="242" y1="84" x2="206" y2="142" stroke="#333" stroke-width="1.6"/>
  <circle cx="140" cy="70" r="14" fill="#fff" stroke="#1565c0" stroke-width="1.6"/>
  <text x="140" y="75" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#1565c0">u</text>
  <circle cx="250" cy="70" r="14" fill="#fff" stroke="#1565c0" stroke-width="1.6"/>
  <text x="250" y="75" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#1565c0">v</text>
  <circle cx="195" cy="155" r="14" fill="#fff" stroke="#1565c0" stroke-width="1.6"/>
  <text x="195" y="160" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#1565c0">w</text>
  <text x="195" y="35" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#1565c0">e (parallel pair)</text>
  <line x1="300" y1="105" x2="352" y2="105" stroke="#888" stroke-width="1.6" stroke-dasharray="6 4" marker-end="url(#gaf17-arrf)"/>
  <text x="326" y="90" text-anchor="middle" font-family="serif" font-size="11" fill="#666">G/e</text>
  <path d="M 452,88 C 436,110 436,130 452,148" fill="none" stroke="#333" stroke-width="1.6"/>
  <path d="M 488,88 C 504,110 504,130 488,148" fill="none" stroke="#333" stroke-width="1.6"/>
  <path d="M 484,52 C 500,30 530,38 526,62 C 523,78 500,80 486,70" fill="none" stroke="#888" stroke-width="1.6" stroke-dasharray="4 3"/>
  <line x1="516" y1="38" x2="534" y2="58" stroke="#b91c1c" stroke-width="2"/>
  <line x1="516" y1="58" x2="534" y2="38" stroke="#b91c1c" stroke-width="2"/>
  <circle cx="470" cy="72" r="17" fill="#fff" stroke="#1565c0" stroke-width="1.8"/>
  <text x="470" y="77" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">uv</text>
  <circle cx="470" cy="163" r="14" fill="#fff" stroke="#1565c0" stroke-width="1.6"/>
  <text x="470" y="168" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#1565c0">w</text>
  <text x="573" y="40" text-anchor="start" font-family="serif" font-size="11" fill="#888">loops</text>
  <text x="573" y="54" text-anchor="start" font-family="serif" font-size="11" fill="#888">removed</text>
  <text x="470" y="200" text-anchor="middle" font-family="serif" font-size="11" fill="#666">parallel edges kept</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Contracting $e = uv$: the endpoints merge into one vertex, the former $uw$- and $vw$-edges survive as parallel edges, and the parallel copies of $e$ itself would become loops — they are discarded. Cuts of $G/e$ correspond to cuts of $G$ of the same size, so a minimum cut $C$ is lost only if the picked edge lies in $C$.
</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Contract)</span></p>

$\mathrm{Contract}(G\_0, \ell)$:

1. $G \leftarrow G\_0$.
2. **While** $n > \ell$:
3. &nbsp;&nbsp;&nbsp;Pick $e \in E$ uniformly at random.
4. &nbsp;&nbsp;&nbsp;$G \leftarrow G/e$, remove loops.
5. Return $G$.

For $\ell = 2$ the two remaining vertices define a single cut of $G\_0$ — the algorithm's candidate minimum cut.

</div>

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Survival Probability)</span></p>

Fix a specific minimum cut $C$ of $G\_0$ and let $k := \lvert C\rvert$. Then

$$
\Pr\bigl[\,C \text{ survives } \mathrm{Contract}(G_0, \ell)\,\bigr] \;\ge\; \frac{\ell\,(\ell-1)}{n\,(n-1)} .
$$

In particular, for $\ell = 2$ a single run returns the cut $C$ itself with probability at least $2/(n(n-1))$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $G\_i$ be the graph before the $i$-th contraction; it has $n\_i = n - i + 1$ vertices. Assume $C$ survived to $G\_i$. Then every vertex $v$ of $G\_i$ has degree at least $k$: its incident edges correspond to a cut of $G\_0$, which has size at least $k$. Hence

$$
m_i \;\ge\; \frac{k\, n_i}{2},
\qquad
\Pr[\,\text{we select } e \in C\,] \;=\; \frac{k}{m_i} \;\le\; \frac{k}{k\, n_i / 2} \;=\; \frac{2}{n_i} ,
$$

and $C$ survives the $i$-th step with probability at least $1 - \frac{2}{n - i + 1} = \frac{n - i - 1}{n - i + 1}$. Multiplying over all $n - \ell$ steps, the product telescopes:

$$
\Pr[\,C \text{ survives all steps}\,]
\;\ge\;
\prod_{i=1}^{n-\ell} \frac{n-i-1}{n-i+1}
\;=\;
\frac{n-2}{n} \cdot \frac{n-3}{n-1} \cdot \frac{n-4}{n-2} \cdots \frac{\ell}{\ell+2} \cdot \frac{\ell-1}{\ell+1}
\;=\;
\frac{\ell\,(\ell-1)}{n\,(n-1)} \;\sim\; \frac{\ell^2}{n^2}.
$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Implementation in $O(n^2)$)</span></p>

Represent the multigraph by its **adjacency matrix** $A\_{ij} = $ number of parallel edges between $i$ and $j$, together with the vector of degrees. Then:

1. picking $e$ uniformly at random takes $O(n)$: choose the first endpoint $u$ with probability proportional to $\deg(u)$ using the degree vector, then the second endpoint $v$ with probability proportional to the row entries $A\_{uv}$;
2. contracting $e = uv$ takes $O(n)$: add row and column $v$ to row and column $u$, zero out $v$, set $A\_{uu} := 0$ (loop removal), and update the degrees.

One step costs $O(n)$, so $\mathrm{Contract}(G\_0, \ell)$ runs in $O((n - \ell) \cdot n) \subseteq O(n^2)$.

</div>

# Lecture 4: Karger–Stein and Shortest Paths

Last time we saw that $\mathrm{Contract}(G, 2)$ returns a fixed minimum cut with probability at least $2/(n(n-1))$, in time $O(n^2)$. Now we turn this into an algorithm that is actually *good* — the Karger–Stein algorithm — and then open a new chapter: shortest paths.

## The Karger–Stein Algorithm

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Silly Attempt)</span></p>

Run $\mathrm{Contract}(G, 2)$ independently $k$ times and return the smallest cut found. Each run succeeds with probability at least $2/(n(n-1)) \ge c/n^2$, so using $1 - x \le e^{-x}$:

$$
\Pr[\,\text{minimum is missed}\,] \;\le\; \Bigl(1 - \frac{c}{n^2}\Bigr)^k \;\le\; e^{-ck/n^2}.
$$

With $k \approx n^2$ the failure probability drops to a constant; with $k \approx n^2 \log n$ it is $e^{-c' \log n} = 1/\mathrm{poly}(n)$ — success *with high probability* — but the running time is $O(n^2 \cdot n^2 \log n) = O(n^4 \log n)$. Too slow.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Early Contractions Are Safe, Late Ones Are Risky)</span></p>

Look at the per-step survival probabilities of a minimum cut $C$: the **first** contraction preserves $C$ with probability at least $\frac{n-2}{n} = 1 - \frac{2}{n}$, while the **final** step (from $3$ vertices down to $2$) only guarantees $\frac{\ell - 1}{\ell + 1} = \frac{1}{3}$. Almost all the risk sits at the end — so contract only *partially* and put the repetition there. Stopping at

$$
\ell \;=\; \Bigl\lceil \frac{n}{\sqrt{2}} + 1 \Bigr\rceil
\qquad\text{gives}\qquad
\Pr[\,C \text{ survives}\,] \;\ge\; \frac{\ell(\ell-1)}{n(n-1)}
\;\ge\; \frac{\bigl(\tfrac{n}{\sqrt 2}+1\bigr)\tfrac{n}{\sqrt 2}}{n(n-1)}
\;=\; \frac{n + \sqrt{2}}{2(n-1)} \;\ge\; \frac{n}{2n} \;=\; \frac12 .
$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Karger–Stein)</span></p>

$\mathrm{MinCut}(G)$:

1. If $n \le 7$: use brute force.
2. $\ell \leftarrow \lceil n/\sqrt{2} + 1 \rceil$.
3. $C\_1 \leftarrow \mathrm{MinCut}(\mathrm{Contract}(G, \ell))$.
4. $C\_2 \leftarrow \mathrm{MinCut}(\mathrm{Contract}(G, \ell))$.
5. Return the smaller of $C\_1, C\_2$.

Each level halves the *risk* by trying twice, while the subproblem size shrinks by a factor $\sqrt{2}$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Karger–Stein — Running Time)</span></p>

$\mathrm{MinCut}(G)$ runs in time $O(n^2 \log n)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The recursion satisfies $T(n) = 2\,T\bigl(n/\sqrt{2} + O(1)\bigr) + O(n^2)$, the $O(n^2)$ being the two Contract calls. At depth $i$ of the recursion tree there are $2^i$ subproblems of size $n/(\sqrt{2})^i$, each costing $O\bigl(n^2/2^i\bigr)$ — so every level of the tree costs $O(n^2)$ in total. The size drops below the brute-force threshold after $\log\_{\sqrt 2} n = 2\log\_2 n = O(\log n)$ levels, hence $T(n) = O(n^2 \log n)$.

</details>
</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 270" width="100%" style="max-width: 580px; height: auto;" role="img" aria-labelledby="gaf18-title">
  <title id="gaf18-title">The Karger–Stein recursion tree</title>
  <defs>
    <marker id="gaf18-arrp" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#7b1fa2"/>
    </marker>
  </defs>
  <line x1="310" y1="45" x2="90" y2="210" stroke="#0f9b6c" stroke-width="2"/>
  <line x1="310" y1="45" x2="530" y2="210" stroke="#0f9b6c" stroke-width="2"/>
  <line x1="90" y1="210" x2="530" y2="210" stroke="#0f9b6c" stroke-width="2"/>
  <line x1="240" y1="98" x2="380" y2="98" stroke="#888" stroke-width="1.2" stroke-dasharray="5 4"/>
  <line x1="175" y1="146" x2="445" y2="146" stroke="#888" stroke-width="1.2" stroke-dasharray="5 4"/>
  <circle cx="310" cy="45" r="4" fill="#0f9b6c"/>
  <text x="310" y="30" text-anchor="middle" font-family="serif" font-size="12" fill="#333">G — n vertices</text>
  <text x="392" y="102" text-anchor="start" font-family="serif" font-size="11" fill="#333">depth i: 2<tspan dy="-4" font-size="8">i</tspan><tspan dy="4"> problems of size n/(√2)</tspan><tspan dy="-4" font-size="8">i</tspan></text>
  <text x="457" y="150" text-anchor="start" font-family="serif" font-size="11" fill="#a86f00">O(n²) per level</text>
  <line x1="60" y1="50" x2="60" y2="205" stroke="#7b1fa2" stroke-width="1.6" marker-start="url(#gaf18-arrp)" marker-end="url(#gaf18-arrp)"/>
  <text x="50" y="122" text-anchor="end" font-family="serif" font-size="11" fill="#7b1fa2">O(log n)</text>
  <text x="50" y="136" text-anchor="end" font-family="serif" font-size="11" fill="#7b1fa2">levels</text>
  <text x="310" y="232" text-anchor="middle" font-family="serif" font-size="11" fill="#666">n ≤ 7 → brute force</text>
  <text x="310" y="258" text-anchor="middle" font-family="serif" font-size="12" fill="#0f9b6c">total time: O(n² log n)</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The recursion tree of Karger–Stein: branching doubles the number of subproblems while the size shrinks by $\sqrt{2}$, so every level costs $O(n^2)$ and there are $O(\log n)$ of them.
</figcaption>
</figure>

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Success Probability)</span></p>

$\mathrm{MinCut}(G)$ finds a minimum cut with probability $\Omega(1 / \log n)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $P\_i := \Pr[\text{the minimum cut is found at recursion height } i\,]$; at the brute-force leaves $P\_0 = 1$. One branch succeeds if the fixed minimum cut survives its Contract call (probability $\ge \frac12$) *and* the recursive call finds it; the two branches are independent, so

$$
P_i \;\ge\; 1 - \Bigl(1 - \tfrac12 P_{i-1}\Bigr)^{2}.
$$

Study the recurrence $g\_0 = 1$, $g\_i = 1 - \bigl(1 - \tfrac12 g\_{i-1}\bigr)^2 = g\_{i-1} - g\_{i-1}^2/4$; by induction $P\_i \ge g\_i$. Substitute $z\_i := 4/g\_i - 1$, i.e. $g\_i = 4/(z\_i + 1)$, with $z\_0 = 3$:

$$
\frac{1}{z_i + 1} = \frac{1}{z_{i-1}+1} - \frac{1}{(z_{i-1}+1)^2}
= \frac{z_{i-1}}{(z_{i-1}+1)^2}
\quad\Longrightarrow\quad
z_i = z_{i-1} + 1 + \frac{1}{z_{i-1}} .
$$

Since always $z\_i \ge 1$, each step adds between $1$ and $2$, so $z\_i \le 3 + 2i$ and

$$
g_i \;=\; \frac{4}{z_i + 1} \;\ge\; \frac{4}{4 + 2i}.
$$

The recursion has depth $D = O(\log n)$, hence the overall success probability is at least $g\_D = \Omega(1/\log n)$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Iterating Karger–Stein)</span></p>

Run $\mathrm{MinCut}$ independently $k$ times and keep the best cut: this takes time $O(n^2 \log n \cdot k)$ and fails with probability

$$
\Pr[\,\text{fail}\,] \;\le\; \Bigl(1 - \frac{c}{\log n}\Bigr)^{k} \;\approx\; e^{-ck/\log n}:
$$

* $k \sim \log n$: constant failure probability;
* $k \sim \log^2 n$: failure $1/\mathrm{poly}(n)$ — a minimum cut **with high probability** in time $O(n^2 \log^3 n)$;
* $k \sim n \log n$: failure $e^{-\Omega(n)}$.

</div>

## Shortest Paths

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Lengths and Distances)</span></p>

Let $G$ be a directed graph with **edge lengths** $\ell \colon E \to \mathbb{R}$. The **distance** $d \colon V^2 \to \mathbb{R} \cup \lbrace \pm\infty \rbrace$ is

$$
d(u, v) := \min \lbrace\, \ell(P) \mid P \text{ is a } uv\text{-path} \,\rbrace,
$$

and $d(u,v) := +\infty$ if no $uv$-path exists.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Walks vs. Paths)</span></p>

If $G$ has **no negative cycles**, then at least one shortest $uv$-*walk* is a *path*: cutting a repeated vertex's cycle out of a walk removes a subwalk of nonnegative length, so it never makes the walk longer. Hence minimizing over walks or over paths gives the same distances, and we may switch freely.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Negative Cycles Break Everything)</span></p>

In the graph below, walks may traverse the negative cycle arbitrarily often, so the walk-distance from $u$ to $v$ is $-\infty$. Restricting to simple paths does not save the day either: there $d(u,v) = 2$, $d(u,t) = -1$ and $d(t,v) = 1$, so

$$
d(u,t) + d(t,v) \;=\; 0 \;<\; 2 \;=\; d(u,v)
$$

— the **triangle inequality fails**. Worse, computing shortest *simple* paths in graphs with negative cycles is **NP-hard** (it contains the Hamiltonian path problem).

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 560 200" width="100%" style="max-width: 500px; height: auto;" role="img" aria-labelledby="gaf19-title">
  <title id="gaf19-title">A negative cycle breaks distances</title>
  <defs>
    <marker id="gaf19-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
    </marker>
    <marker id="gaf19-arrr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#b91c1c"/>
    </marker>
  </defs>
  <text x="330" y="20" text-anchor="middle" font-family="serif" font-size="11" fill="#b91c1c">negative cycle: −2 + 0 &lt; 0</text>
  <line x1="86" y1="130" x2="212" y2="130" stroke="#333" stroke-width="1.8" marker-end="url(#gaf19-arr)"/>
  <text x="150" y="118" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#a86f00">1</text>
  <line x1="246" y1="130" x2="484" y2="130" stroke="#333" stroke-width="1.8" marker-end="url(#gaf19-arr)"/>
  <text x="365" y="118" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#a86f00">1</text>
  <line x1="240" y1="115" x2="316" y2="62" stroke="#b91c1c" stroke-width="1.8" marker-end="url(#gaf19-arrr)"/>
  <text x="262" y="76" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#b91c1c">−2</text>
  <path d="M 340,60 C 380,88 330,118 248,124" fill="none" stroke="#b91c1c" stroke-width="1.8" marker-end="url(#gaf19-arrr)"/>
  <text x="356" y="100" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#b91c1c">0</text>
  <circle cx="70" cy="130" r="15" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="70" y="135" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#0f9b6c">u</text>
  <circle cx="230" cy="130" r="15" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="230" y="135" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#1565c0">a</text>
  <circle cx="330" cy="52" r="15" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="330" y="57" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#1565c0">t</text>
  <circle cx="500" cy="130" r="15" fill="#fef2f2" stroke="#b91c1c" stroke-width="2"/>
  <text x="500" y="135" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#b91c1c">v</text>
  <text x="280" y="185" text-anchor="middle" font-family="serif" font-size="11" fill="#666">simple paths: d(u,v) = 2,  d(u,t) = −1,  d(t,v) = 1;  walks: d(u,v) = −∞</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The only simple $uv$-path is $u, a, v$ of length $2$, yet $d(u,t) + d(t,v) = -1 + 1 = 0$ — any $uv$-path through $t$ would have to visit $a$ twice. Walks may loop through the negative cycle $a \to t \to a$, driving the walk-distance to $-\infty$.
</figcaption>
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Triangle Inequality)</span></p>

If $G$ has no negative cycles, then for all $u, t, v$:

$$
d(u, v) \;\le\; d(u, t) + d(t, v),
$$

since concatenating a shortest $ut$-walk with a shortest $tv$-walk gives *some* $uv$-walk.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Prefix Property)</span></p>

A prefix of a shortest path is again a shortest path: if a $uv$-path $P$ passes through $t$ and its $ut$-prefix could be shortened, the whole of $P$ could be shortened too (no negative cycles needed here beyond replacing the prefix by a shorter *walk* and simplifying under the no-negative-cycles assumption).

</div>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Kinds of Shortest-Path Problems)</span></p>

* **P2PSP** (point-to-point): given $u, v$, find $d(u,v)$ and a shortest path.
* **SSSP** (single source): given $u$, find $d(u, v)$ for all $v$ — the answer fits into a **shortest-path tree**, $O(n)$ space.
* **APSP** (all pairs): all $u, v$ — the distance matrix and a collection of shortest-path trees for all sources, $O(n^2)$ space.

No general algorithm solves P2PSP faster than SSSP, so the single-source problem is the central one.

</div>

## Shortest-Path Trees

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Shortest-Path Tree)</span></p>

A **shortest-path tree** from $u \in V$ is an oriented tree $T\_u$ rooted at $u$ with edges pointing outwards such that for every vertex $v$ of $T\_u$, the unique path $u \to v$ in $T\_u$ is a shortest $uv$-path in $G$.

</div>

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Existence)</span></p>

If $G$ has no negative cycles, a shortest-path tree from $u$ exists, spanning all vertices reachable from $u$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Build $T\_u$ iteratively. Start with the single vertex $u$. While some reachable $v$ is not covered, take a shortest $uv$-path $P$ and follow it backwards to the last vertex $x$ already in the tree; graft the $x \to v$ suffix of $P$ onto the tree, *but replace its prefix by the tree path $u \to x$*. By the prefix property the $ux$-prefix of $P$ is a shortest $ux$-path, and the tree path to $x$ has the same length — so every new vertex receives a shortest path. Repeat until all reachable vertices are covered.

</details>
</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 230" width="100%" style="max-width: 580px; height: auto;" role="img" aria-labelledby="gaf20-title">
  <title id="gaf20-title">A shortest-path tree with one root-to-vertex path highlighted</title>
  <defs>
    <marker id="gaf20-arrb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#1565c0"/>
    </marker>
    <marker id="gaf20-arrr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#b91c1c"/>
    </marker>
  </defs>
  <text x="86" y="34" font-family="serif" font-size="13" font-style="italic" fill="#1565c0">T<tspan dy="4" font-size="9">u</tspan></text>
  <line x1="74" y1="108" x2="163" y2="66" stroke="#b91c1c" stroke-width="2.4" marker-end="url(#gaf20-arrr)"/>
  <line x1="74" y1="122" x2="163" y2="166" stroke="#1565c0" stroke-width="1.8" marker-end="url(#gaf20-arrb)"/>
  <line x1="188" y1="53" x2="307" y2="40" stroke="#1565c0" stroke-width="1.8" marker-end="url(#gaf20-arrb)"/>
  <line x1="189" y1="66" x2="307" y2="92" stroke="#b91c1c" stroke-width="2.4" marker-end="url(#gaf20-arrr)"/>
  <line x1="190" y1="172" x2="307" y2="172" stroke="#1565c0" stroke-width="1.8" marker-end="url(#gaf20-arrb)"/>
  <line x1="333" y1="38" x2="452" y2="30" stroke="#1565c0" stroke-width="1.8" marker-end="url(#gaf20-arrb)"/>
  <line x1="334" y1="98" x2="529" y2="112" stroke="#b91c1c" stroke-width="2.4" marker-end="url(#gaf20-arrr)"/>
  <line x1="333" y1="178" x2="452" y2="188" stroke="#1565c0" stroke-width="1.8" marker-end="url(#gaf20-arrb)"/>
  <circle cx="60" cy="115" r="16" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="60" y="120" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#0f9b6c">u</text>
  <circle cx="176" cy="60" r="13" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <circle cx="176" cy="172" r="13" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <circle cx="320" cy="38" r="13" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <circle cx="320" cy="96" r="13" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <circle cx="320" cy="172" r="13" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <circle cx="465" cy="29" r="13" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <circle cx="465" cy="189" r="13" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <circle cx="545" cy="113" r="15" fill="#fef2f2" stroke="#b91c1c" stroke-width="2"/>
  <text x="545" y="118" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#b91c1c">v</text>
  <text x="310" y="222" text-anchor="middle" font-family="serif" font-size="11" fill="#666">the tree path u → v (red) is a shortest uv-path in G</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
A shortest-path tree $T_u$: one oriented tree, $O(n)$ space, whose root-to-vertex paths are simultaneously shortest paths for *all* reachable vertices — here the path to $v$ is highlighted.
</figcaption>
</figure>

## The Relaxation Scheme

A common skeleton for SSSP algorithms; fix the source $u$.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Relaxation)</span></p>

We maintain a **value** $h(v)$ for every vertex — always $+\infty$ or the length of some $uv$-walk — together with a **state**:

* **unseen** — not reached yet, $h(v) = +\infty$;
* **open** — reached, still needs relaxing;
* **closed** — reached, no relaxing needed for now.

The operation $\mathrm{relax}(v)$ pushes $v$'s value along all outgoing edges: for each $vw \in E$,

$$
h(w) \;\leftarrow\; \min\bigl(h(w),\; h(v) + \ell(vw)\bigr).
$$

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 190" width="100%" style="max-width: 560px; height: auto;" role="img" aria-labelledby="gaf21-title">
  <title id="gaf21-title">Relaxing a vertex pushes its value along outgoing edges</title>
  <defs>
    <marker id="gaf21-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
    </marker>
    <marker id="gaf21-arrr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#b91c1c"/>
    </marker>
  </defs>
  <path d="M 66,100 C 120,55 190,145 268,102" fill="none" stroke="#333" stroke-width="1.6" stroke-dasharray="6 4" marker-end="url(#gaf21-arr)"/>
  <text x="165" y="48" text-anchor="middle" font-family="serif" font-size="11" fill="#666">a u–v walk of length h(v)</text>
  <line x1="298" y1="92" x2="452" y2="42" stroke="#b91c1c" stroke-width="2" marker-end="url(#gaf21-arrr)"/>
  <line x1="300" y1="100" x2="450" y2="100" stroke="#b91c1c" stroke-width="2" marker-end="url(#gaf21-arrr)"/>
  <line x1="298" y1="108" x2="452" y2="158" stroke="#b91c1c" stroke-width="2" marker-end="url(#gaf21-arrr)"/>
  <text x="380" y="76" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#b91c1c">relax(v)</text>
  <circle cx="50" cy="100" r="15" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="50" y="105" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#0f9b6c">u</text>
  <circle cx="284" cy="100" r="16" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <text x="284" y="105" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#1565c0">v</text>
  <circle cx="466" cy="38" r="12" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <circle cx="466" cy="100" r="12" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <circle cx="466" cy="162" r="12" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <text x="520" y="16" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#333">w — reopened if h(w) drops</text>
  <text x="290" y="182" text-anchor="middle" font-family="serif" font-size="11" fill="#666">states: unseen → open ⇄ closed</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Relaxing $v$: its value $h(v)$ — the length of some known $uv$-walk — is offered to every successor $w$ as $h(v) + \ell(vw)$. Any $w$ whose value drops is (re)opened and will be relaxed again.
</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Relaxation Scheme for SSSP)</span></p>

1. $h(\ast) \leftarrow +\infty$, $h(u) \leftarrow 0$.
2. $\mathrm{state}(\ast) \leftarrow$ unseen, $\mathrm{state}(u) \leftarrow$ open.
3. **While** there is a vertex $v$ with $\mathrm{state}(v) =$ open:
4. &nbsp;&nbsp;&nbsp;$\mathrm{state}(v) \leftarrow$ closed.
5. &nbsp;&nbsp;&nbsp;$\mathrm{relax}(v)$.
6. &nbsp;&nbsp;&nbsp;If $h(w)$ got changed: $\mathrm{state}(w) \leftarrow$ open, $\mathrm{pred}(w) \leftarrow v$.

The scheme leaves open *which* open vertex to pick in step 3 — different choices will give the concrete algorithms.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Properties of the Relaxation Scheme)</span></p>

Write NNC for "$G$ has no negative cycles". Then:

1. $h(v)$ never increases.
2. A finite $h(v)$ is always the length of some $uv$-walk; under NNC, $h(v) \ge d(u, v)$.
3. NNC $\Rightarrow$ a finite $h(v)$ is even the length of some $uv$-*path*.
4. NNC $\Rightarrow$ the algorithm always stops.
5. After stopping: $\mathrm{state}(v) =$ closed $\iff$ $v$ is reachable from $u$.
6. After stopping: $v$ is reachable $\iff$ $h(v)$ is finite.
7. After stopping: $h(v) = d(u, v)$ for all $v$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**1.** $h(w)$ is only ever replaced by a minimum involving its old value.

**2.** Induction over updates: when $h(w)$ is set to $h(v) + \ell(vw)$, a witnessing $uv$-walk for $h(v)$ extends by the edge $vw$. Under NNC every $uv$-walk has length at least $d(u,v)$, so $h(v) \ge d(u,v)$.

**3.** Unwind the $\mathrm{pred}$ pointers through the moments of their updates: $h(w) = h(v) + \ell(vw)$ held at update time, and recursing on $v$'s value *at that moment* spells out a $uw$-walk whose length telescopes to exactly the current $h(w)$. If this walk repeated a vertex $x$, the enclosed cycle would have length equal to the *drop* of $h(x)$ between the two visits — negative, contradicting NNC. So the walk is a path.

**4.** By 3, every $h$-value that ever occurs is the length of one of the finitely many paths from $u$; every update strictly decreases some $h(w)$ within this finite set, so there are finitely many updates, and every loop iteration either closes a vertex forever or follows an update.

**5 & 6.** Values propagate exactly along edges from $u$, so reached $\iff$ reachable $\iff$ finite $h$ at stop; every reached vertex is opened and later closed.

**7.** Suppose after stopping some vertex is **bad**, i.e. $h(v) > d(u, v)$, and among bad vertices choose $v$ whose shortest $uv$-path $P$ has the fewest edges. Let $tv$ be the last edge of $P$. Then $t$ is not bad, so $h(t) = d(u, t)$. When $h(t)$ last dropped to this value, $t$ became open, hence was later closed again — and relaxing it enforced

$$
h(v) \;\le\; h(t) + \ell(tv) \;=\; d(u,t) + \ell(tv) \;=\; d(u,v)
$$

(the last equality by the prefix property). So $v$ is not bad — a contradiction.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Predecessors Form a Shortest-Path Tree)</span></p>

After the scheme stops, the recorded pointers $\mathrm{pred}(v)$ span exactly the reachable vertices, and by property 7 together with the prefix property the pred-edges form a shortest-path tree from $u$ — the SSSP answer in $O(n)$ space.

</div>

# Lecture 5: Bellman–Ford–Moore and Dijkstra's Algorithm

The relaxation scheme from last lecture deliberately left one thing open: *which* open vertex to pick in step 3. This lecture fills that slot twice. Keeping the open vertices in a **queue** gives the Bellman–Ford–Moore algorithm, which runs in $O(nm)$ and tolerates negative edge lengths; always picking the open vertex of **minimum value** gives Dijkstra's algorithm, whose running time then becomes a question about data structures — heaps for general lengths, buckets for small integers.

## The Bellman–Ford–Moore Algorithm

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Bellman–Ford–Moore, BFM)</span></p>

Run the relaxation scheme, keeping the open vertices in a **queue**:

* in step 3, always select the vertex at the *head* of the queue;
* when a vertex becomes open, append it to the *tail* — unless it is open already, in which case it keeps its position and only its value drops.

</div>

For the whole analysis we assume — as last lecture — that $G$ has **no negative cycles** (NNC), so all seven properties of the relaxation scheme are available.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Phases)</span></p>

Split the run of BFM into **phases** by imaginary separators in the queue: **phase $0$** closes just the source $u$; **phase $i+1$** closes exactly the vertices that were opened during phase $i$. Since a vertex occupies at most one queue slot at a time, every phase closes at most $n$ vertices.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 180" width="100%" style="max-width: 580px; height: auto;" role="img" aria-labelledby="gaf22-title">
  <title id="gaf22-title">The queue of open vertices split into phases by imaginary separators</title>
  <rect x="30" y="70" width="560" height="44" rx="10" fill="#fff" stroke="#333" stroke-width="1.8"/>
  <circle cx="54" cy="92" r="8" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="54" y="96" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#0f9b6c">u</text>
  <line x1="78" y1="62" x2="78" y2="122" stroke="#b91c1c" stroke-width="1.6"/>
  <circle cx="100" cy="92" r="5" fill="#1565c0"/>
  <circle cx="124" cy="92" r="5" fill="#1565c0"/>
  <circle cx="148" cy="92" r="5" fill="#1565c0"/>
  <circle cx="172" cy="92" r="5" fill="#1565c0"/>
  <circle cx="196" cy="92" r="5" fill="#1565c0"/>
  <line x1="220" y1="62" x2="220" y2="122" stroke="#b91c1c" stroke-width="1.6"/>
  <circle cx="244" cy="92" r="5" fill="#1565c0"/>
  <circle cx="268" cy="92" r="5" fill="#1565c0"/>
  <circle cx="292" cy="92" r="5" fill="#1565c0"/>
  <line x1="316" y1="62" x2="316" y2="122" stroke="#b91c1c" stroke-width="1.6"/>
  <text x="370" y="97" text-anchor="middle" font-family="serif" font-size="15" fill="#333">⋯</text>
  <text x="54" y="140" text-anchor="middle" font-family="serif" font-size="11" fill="#0f9b6c">phase 0</text>
  <text x="148" y="52" text-anchor="middle" font-family="serif" font-size="11" fill="#333">phase 1 — ≤ n vertices</text>
  <text x="268" y="140" text-anchor="middle" font-family="serif" font-size="11" fill="#333">phase 2</text>
  <text x="310" y="168" text-anchor="middle" font-family="serif" font-size="11" fill="#666">phase i+1 closes exactly the vertices that phase i opened</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The FIFO queue of open vertices with imaginary separators (red): phase $0$ closes just the source $u$, phase $1$ the vertices opened while relaxing $u$, and so on. Each phase closes at most $n$ vertices.
</figcaption>
</figure>

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Phase Invariant)</span></p>

At the end of phase $i$, every vertex $v$ satisfies

$$
h(v) \;\le\; \ell(w) \qquad \text{for every } uv\text{-walk } w \text{ with at most } i \text{ edges}.
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By induction on $i$.

**$i = 0$:** the only walk with $0$ edges is the empty walk at $u$, and $h(u) = 0$ is exactly its length.

**$i - 1 \to i$:** consider a vertex $v$ at the end of phase $i$ and a $uv$-walk $w$ with at most $i$ edges. If $w$ has at most $i - 1$ edges, the induction hypothesis already gave $h(v) \le \ell(w)$ at the end of phase $i-1$, and values never increase. Otherwise $w$ has exactly $i$ edges: let $pv$ be its last edge and $w'$ its $up$-prefix, so that $w'$ has $i - 1$ edges and $\ell(w) = \ell(w') + \ell(pv)$.

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 210" width="100%" style="max-width: 560px; height: auto;" role="img" aria-labelledby="gaf23-title">
  <title id="gaf23-title">A uv-walk with i edges decomposed into a prefix and its last edge</title>
  <defs>
    <marker id="gaf23-arrg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#0f9b6c"/>
    </marker>
    <marker id="gaf23-arrr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#b91c1c"/>
    </marker>
  </defs>
  <path d="M 60,44 L 60,36 L 560,36 L 560,44" fill="none" stroke="#333" stroke-width="1.2"/>
  <text x="310" y="28" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#333">w — i edges</text>
  <path d="M 60,78 L 60,70 L 420,70 L 420,78" fill="none" stroke="#1565c0" stroke-width="1.2"/>
  <text x="240" y="62" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">w′ — i−1 edges</text>
  <line x1="76" y1="146" x2="136" y2="128" stroke="#0f9b6c" stroke-width="1.8" marker-end="url(#gaf23-arrg)"/>
  <line x1="164" y1="126" x2="226" y2="150" stroke="#0f9b6c" stroke-width="1.8" marker-end="url(#gaf23-arrg)"/>
  <line x1="254" y1="150" x2="316" y2="128" stroke="#0f9b6c" stroke-width="1.8" marker-end="url(#gaf23-arrg)"/>
  <line x1="344" y1="128" x2="402" y2="146" stroke="#0f9b6c" stroke-width="1.8" marker-end="url(#gaf23-arrg)"/>
  <line x1="438" y1="150" x2="542" y2="150" stroke="#b91c1c" stroke-width="2.2" marker-end="url(#gaf23-arrr)"/>
  <text x="490" y="136" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#b91c1c">ℓ(pv)</text>
  <circle cx="60" cy="150" r="15" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="60" y="155" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#0f9b6c">u</text>
  <circle cx="150" cy="124" r="7" fill="#fff" stroke="#0f9b6c" stroke-width="1.5"/>
  <circle cx="240" cy="152" r="7" fill="#fff" stroke="#0f9b6c" stroke-width="1.5"/>
  <circle cx="330" cy="126" r="7" fill="#fff" stroke="#0f9b6c" stroke-width="1.5"/>
  <circle cx="420" cy="150" r="15" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <text x="420" y="155" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#1565c0">p</text>
  <circle cx="560" cy="150" r="15" fill="#fef2f2" stroke="#b91c1c" stroke-width="2"/>
  <text x="560" y="155" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#b91c1c">v</text>
  <text x="310" y="198" text-anchor="middle" font-family="serif" font-size="11" fill="#666">ℓ(w) = ℓ(w′) + ℓ(pv)</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
A $uv$-walk $w$ with exactly $i$ edges splits into its $up$-prefix $w'$ with $i-1$ edges and the last edge $pv$.
</figcaption>
</figure>

By the induction hypothesis, $h(p) \le \ell(w')$ at the end of phase $i-1$. Look at the moment when $h(p)$ last dropped to (or below) this value: it happened in phase at most $i - 1$, and at that moment $p$ was opened (or was open already). An open vertex from phase at most $i-1$ is closed — and relaxed — in phase at most $i$, and that relaxation enforces

$$
h(v) \;\le\; h(p) + \ell(pv) \;\le\; \ell(w') + \ell(pv) \;=\; \ell(w).
$$

Values only drop afterwards, so the bound still holds at the end of phase $i$.

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 170" width="100%" style="max-width: 580px; height: auto;" role="img" aria-labelledby="gaf24-title">
  <title id="gaf24-title">Timing: p is opened in phase at most i minus one and relaxed in phase at most i</title>
  <defs>
    <marker id="gaf24-arrb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#1565c0"/>
    </marker>
    <marker id="gaf24-arrr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#b91c1c"/>
    </marker>
  </defs>
  <text x="325" y="16" text-anchor="middle" font-family="serif" font-size="11" fill="#1565c0">h(p) reached its final value in phase ≤ i−1 ⇒ p was opened</text>
  <line x1="325" y1="22" x2="325" y2="48" stroke="#1565c0" stroke-width="1.4" marker-end="url(#gaf24-arrb)"/>
  <rect x="30" y="52" width="560" height="40" rx="10" fill="#fff" stroke="#333" stroke-width="1.8"/>
  <line x1="120" y1="46" x2="120" y2="98" stroke="#b91c1c" stroke-width="1.6"/>
  <line x1="240" y1="46" x2="240" y2="98" stroke="#b91c1c" stroke-width="1.6"/>
  <line x1="410" y1="46" x2="410" y2="98" stroke="#b91c1c" stroke-width="1.6"/>
  <text x="75" y="112" text-anchor="middle" font-family="serif" font-size="11" fill="#333">phase 0</text>
  <text x="180" y="77" text-anchor="middle" font-family="serif" font-size="14" fill="#333">⋯</text>
  <text x="325" y="112" text-anchor="middle" font-family="serif" font-size="11" fill="#333">phase i−1</text>
  <text x="500" y="112" text-anchor="middle" font-family="serif" font-size="11" fill="#333">phase i</text>
  <circle cx="325" cy="72" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="2"/>
  <text x="341" y="76" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">p</text>
  <line x1="548" y1="136" x2="534" y2="100" stroke="#b91c1c" stroke-width="1.4" marker-end="url(#gaf24-arrr)"/>
  <text x="420" y="156" text-anchor="middle" font-family="serif" font-size="11" fill="#b91c1c">in phase ≤ i, p is closed and relaxed: h(v) ≤ h(p) + ℓ(pv)</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The timing argument: $p$ enters the queue no later than during phase $i-1$, so it is closed and relaxed no later than during phase $i$ — and relaxing $p$ pushes the correct bound onto $v$.
</figcaption>
</figure>

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Complexity of BFM)</span></p>

On a graph with no negative cycles, BFM runs in time $O(nm)$, stopping after at most $n + 1$ phases (phases $0, \dots, n$).

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Under NNC, deleting a cycle from a walk removes a subwalk of non-negative length, so every shortest walk can be replaced by an equally short *path* — one with at most $n - 1$ edges. The phase-invariant lemma with $i = n - 1$ therefore gives $h(v) \le d(u, v)$ for every $v$ at the end of phase $n-1$; combined with property 2 of the relaxation scheme ($h(v) \ge d(u,v)$ under NNC), in fact $h(v) = d(u, v)$ everywhere.

No value can ever decrease again, so the relaxations of phase $n$ change nothing and open no new vertices: the queue runs empty by the end of phase $n$ and the algorithm stops.

As for the running time: one phase closes at most $n$ vertices, each at most once, and closing $v$ costs $O(\deg v)$ for its relaxation — so a phase costs $O\bigl(\sum\_v \deg v\bigr) = O(m)$, and $O(n)$ phases cost $O(nm)$ in total.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Other Ways to Order the Open Vertices)</span></p>

* **Stack** (Pape's algorithm): LIFO instead of FIFO. Fast on some practical instances, but **exponential** in the worst case.
* **Round robin:** sweep over $v\_1, \dots, v\_n$ in a fixed order, relaxing whichever are open, and repeat. One sweep performs at least the work of one BFM phase, so at most $n + 1$ sweeps suffice — again $O(nm)$. This is the classical textbook formulation of Bellman–Ford.

</div>

## Dijkstra's Algorithm

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Dijkstra)</span></p>

Run the relaxation scheme, in step 3 always selecting an open vertex $v$ with $h(v)$ **minimal**.

*Assumption:* all edge lengths are **non-negative** — a stronger requirement than NNC, which then holds automatically.

</div>

Intuitively, Dijkstra's algorithm behaves like a weighted BFS: the closed vertices form a region that grows outwards from $u$ in layers of increasing distance.

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 560 250" width="100%" style="max-width: 460px; height: auto;" role="img" aria-labelledby="gaf25-title">
  <title id="gaf25-title">Dijkstra grows the closed region in layers around the source, like BFS</title>
  <path d="M 168,118 C 160,60 212,10 282,12 C 356,14 398,62 394,122 C 390,180 340,226 270,224 C 202,222 176,172 168,118 Z" fill="none" stroke="#1565c0" stroke-width="1.8"/>
  <path d="M 202,118 C 198,74 234,42 282,44 C 334,46 364,80 360,122 C 356,166 318,196 272,194 C 226,192 206,158 202,118 Z" fill="none" stroke="#1565c0" stroke-width="1.8"/>
  <path d="M 238,118 C 236,90 258,74 282,76 C 310,78 326,94 324,120 C 322,146 302,162 276,160 C 252,158 240,142 238,118 Z" fill="none" stroke="#1565c0" stroke-width="1.8"/>
  <circle cx="281" cy="118" r="13" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="281" y="123" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#0f9b6c">u</text>
  <text x="470" y="40" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">BFS layers</text>
  <text x="281" y="244" text-anchor="middle" font-family="serif" font-size="11" fill="#666">closed vertices grow outwards in layers of increasing d(u,·)</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Like BFS: Dijkstra's algorithm closes vertices in "layers" around the source $u$ — first the nearby ones, then ever more distant ones.
</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Dijkstra Closes in Distance Order)</span></p>

On graphs with no negative edges, Dijkstra's algorithm closes each vertex **at most once**, and vertices are closed in the order of **increasing** $d(u, \cdot)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We maintain the invariant that at the moment a vertex $v$ is selected as current,

$$
h(x) \;\le\; h(v) \;\le\; h(w)
\qquad \text{for every closed } x \text{ and every open } w,
$$

and that values of closed vertices never change (unseen vertices sit at $+\infty$ and are trivial). The right-hand inequality is just the minimality of the choice of $v$; the rest goes by induction over the selections.

When the current $v$ is relaxed, every value it offers is $h(v) + \ell(vw) \ge h(v)$, because $\ell(vw) \ge 0$. Consequently:

* **no closed vertex is reopened** — a closed $x$ has $h(x) \le h(v)$ by the invariant, so the offer cannot decrease its value;
* **all open values stay $\ge h(v)$** — old open values were $\ge h(v)$ already, and new ones are offers $\ge h(v)$.

The next current vertex is therefore selected with value $\ge h(v) \ge$ every closed value, so the invariant persists — and the values at the moments of closing form a non-decreasing sequence.

Each vertex is thus closed at most once, with a value that never changes afterwards. After stopping, $h = d(u, \cdot)$ by last lecture's theorem, so each vertex was closed with value exactly $d(u, v)$ — in the order of increasing distance.

</details>
</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 260" width="100%" style="max-width: 600px; height: auto;" role="img" aria-labelledby="gaf26-title">
  <title id="gaf26-title">The Dijkstra invariant: closed values below the current value below open values</title>
  <defs>
    <marker id="gaf26-arrr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#b91c1c"/>
    </marker>
  </defs>
  <path d="M 40,120 C 38,80 72,52 118,54 C 168,56 198,84 196,122 C 194,160 162,188 116,186 C 70,184 42,158 40,120 Z" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <circle cx="95" cy="112" r="12" fill="#fff" stroke="#0f9b6c" stroke-width="1.8"/>
  <text x="95" y="117" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#0f9b6c">u</text>
  <text x="118" y="162" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#0f9b6c">closed</text>
  <circle cx="285" cy="120" r="16" fill="#fef2f2" stroke="#b91c1c" stroke-width="2"/>
  <text x="285" y="125" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#b91c1c">v</text>
  <text x="285" y="90" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#b91c1c">current — relaxing</text>
  <path d="M 360,120 C 358,80 394,52 440,54 C 488,56 520,86 518,124 C 516,162 482,190 438,188 C 394,186 362,158 360,120 Z" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <text x="440" y="74" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">open</text>
  <circle cx="430" cy="100" r="6" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <circle cx="460" cy="135" r="6" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <circle cx="420" cy="165" r="6" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <line x1="300" y1="112" x2="418" y2="101" stroke="#b91c1c" stroke-width="1.8" marker-end="url(#gaf26-arrr)"/>
  <line x1="301" y1="123" x2="448" y2="134" stroke="#b91c1c" stroke-width="1.8" marker-end="url(#gaf26-arrr)"/>
  <line x1="298" y1="131" x2="409" y2="160" stroke="#b91c1c" stroke-width="1.8" marker-end="url(#gaf26-arrr)"/>
  <ellipse cx="585" cy="120" rx="42" ry="58" fill="none" stroke="#666" stroke-width="1.5" stroke-dasharray="6 4"/>
  <circle cx="575" cy="102" r="5" fill="#666"/>
  <circle cx="595" cy="140" r="5" fill="#666"/>
  <text x="585" y="46" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#666">unseen</text>
  <text x="320" y="240" text-anchor="middle" font-family="serif" font-size="11" fill="#666">invariant: h(closed) ≤ h(v) ≤ h(open) — offers h(v) + ℓ(vw) ≥ h(v) preserve it</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The invariant behind Dijkstra's algorithm: values of closed vertices $\le$ value of the current vertex $v$ $\le$ values of open vertices. With $\ell \ge 0$, relaxing $v$ only makes offers of at least $h(v)$, so no closed vertex is ever reopened.
</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Number of Relaxations)</span></p>

Dijkstra's algorithm performs at most $n$ relax operations — each vertex is relaxed only at the moment it is closed, which happens at most once. All relaxations together traverse each edge at most once, i.e. $O(m)$ edge work in total; the rest of the running time is spent *finding the minima*.

</div>

## The Time Complexity of Dijkstra's Algorithm

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Trivial Implementation)</span></p>

Store the values $h(v)$ in a plain array. One step finds the open minimum by a linear scan in $O(n)$, and there are at most $n$ steps — $O(n^2)$ for all scans; all relaxations together process $O(m) = O(n^2)$ edges. Total: $O(n^2)$.

</div>

The scans are the bottleneck, which suggests the **idea**: keep the open vertices in a data structure — generically a *"heap"* — supporting three operations:

* **Insert** in time $T\_I$ — a vertex becomes open,
* **ExtractMin** in time $T\_X$ — find and remove the open vertex of minimum value,
* **Decrease** in time $T\_D$ — an open vertex's value drops.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Heap-Based Dijkstra)</span></p>

With such a structure, Dijkstra's algorithm runs in time

$$
O(n \cdot T_I \;+\; n \cdot T_X \;+\; m \cdot T_D):
$$

every vertex is inserted and extracted at most once (a closed vertex is never reopened), and every edge causes at most one Decrease — when its tail is relaxed.

</div>

Plugging in concrete data structures:

| data structure | $T\_I$ | $T\_X$ | $T\_D$ | Dijkstra total |
|---|---|---|---|---|
| array | $O(1)$ | $O(n)$ | $O(1)$ | $O(n^2)$ |
| binary heap | $O(\log n)$ | $O(\log n)$ | $O(\log n)$ | $O(m \log n)$ |
| $k$-ary heap | $O(\log\_k n)$ | $O(k \log\_k n)$ | $O(\log\_k n)$ | $O(m \log\_{m/n} n)$ for $k = \max(m/n, 2)$ |
| Fibonacci heap | $O(1)$ | $O(\log n)$ | $O(1)$ | $O(m + n \log n)$ |

(The Fibonacci-heap bounds are amortized, which is all we need — the operations are summed over the whole run anyway.)

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 260" width="100%" style="max-width: 560px; height: auto;" role="img" aria-labelledby="gaf27-title">
  <title id="gaf27-title">A k-ary heap: few levels, but ExtractMin compares k children per level</title>
  <path d="M 330,32 L 130,215 L 530,215 Z" fill="none" stroke="#333" stroke-width="1.6"/>
  <path d="M 96,36 L 88,36 L 88,215 L 96,215" fill="none" stroke="#333" stroke-width="1.2"/>
  <text x="44" y="120" text-anchor="middle" font-family="serif" font-size="12" fill="#333">log<tspan dy="4" font-size="9">k</tspan><tspan dy="-4"> n levels</tspan></text>
  <line x1="330" y1="43" x2="244" y2="96" stroke="#0f9b6c" stroke-width="1.5"/>
  <line x1="330" y1="43" x2="302" y2="96" stroke="#0f9b6c" stroke-width="1.5"/>
  <line x1="330" y1="43" x2="360" y2="96" stroke="#0f9b6c" stroke-width="1.5"/>
  <line x1="330" y1="43" x2="418" y2="96" stroke="#0f9b6c" stroke-width="1.5"/>
  <circle cx="330" cy="36" r="7" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <circle cx="240" cy="102" r="6" fill="#fff" stroke="#0f9b6c" stroke-width="1.5"/>
  <circle cx="300" cy="102" r="6" fill="#fff" stroke="#0f9b6c" stroke-width="1.5"/>
  <circle cx="360" cy="102" r="6" fill="#fff" stroke="#0f9b6c" stroke-width="1.5"/>
  <circle cx="420" cy="102" r="6" fill="#fff" stroke="#0f9b6c" stroke-width="1.5"/>
  <line x1="298" y1="108" x2="234" y2="164" stroke="#0f9b6c" stroke-width="1.5"/>
  <line x1="299" y1="108" x2="277" y2="164" stroke="#0f9b6c" stroke-width="1.5"/>
  <line x1="302" y1="108" x2="320" y2="164" stroke="#0f9b6c" stroke-width="1.5"/>
  <line x1="304" y1="108" x2="363" y2="164" stroke="#0f9b6c" stroke-width="1.5"/>
  <circle cx="232" cy="170" r="6" fill="#fff" stroke="#0f9b6c" stroke-width="1.5"/>
  <circle cx="277" cy="170" r="6" fill="#fff" stroke="#0f9b6c" stroke-width="1.5"/>
  <circle cx="322" cy="170" r="6" fill="#fff" stroke="#0f9b6c" stroke-width="1.5"/>
  <circle cx="365" cy="170" r="6" fill="#fff" stroke="#0f9b6c" stroke-width="1.5"/>
  <text x="440" y="175" text-anchor="middle" font-family="serif" font-size="13" fill="#333">⋯</text>
  <circle cx="330" cy="36" r="12" fill="none" stroke="#c2185b" stroke-width="1.6"/>
  <circle cx="300" cy="102" r="11" fill="none" stroke="#c2185b" stroke-width="1.6"/>
  <circle cx="322" cy="170" r="11" fill="none" stroke="#c2185b" stroke-width="1.6"/>
  <text x="330" y="243" text-anchor="middle" font-family="serif" font-size="11" fill="#c2185b">ExtractMin sifts down: at each level, swap with the smallest of the k children</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
A $k$-ary heap has only $\log\_k n$ levels, so Insert and Decrease (which bubble *up*, one comparison per level) speed up — but ExtractMin sifts *down* and must find the smallest of $k$ children at every level. Each vertex keeps a pointer to its heap position so Decrease can find it.
</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Tuning the Arity $k$)</span></p>

The $k$-ary heap's total is

$$
\underbrace{m \cdot \frac{\log n}{\log k}}_{\text{decreases with } k}
\;+\;
\underbrace{n \cdot k \cdot \frac{\log n}{\log k}}_{\text{increases with } k},
$$

so the best $k$ balances the two terms: $m = n k$, i.e. $k = \max(m/n,\, 2)$, giving $O(m \log\_{m/n} n)$. Two regimes:

* **dense** graphs, $m \sim n^2$: then $\log(m/n) = \Theta(\log n)$ and the bound is $O(m) = O(n^2)$ — matching the trivial implementation;
* **sparse** graphs, $m \sim n$: then $k$ is constant and the bound is $O(m \log n)$ — matching the binary heap.

In between, the $k$-ary heap strictly beats both.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Fibonacci Heaps)</span></p>

Fibonacci heaps achieve amortized $O(1)$ Insert and Decrease with $O(\log n)$ ExtractMin, so Dijkstra runs in $O(m + n \log n)$. For **real** edge lengths this is optimal among comparison-based implementations, as the sorting reduction below shows.

</div>

## Integer Lengths: Buckets

Suppose now that all edge lengths are small integers: $\ell(e) \in \lbrace 0, \dots, L \rbrace$ for every edge $e$. Then the heap can be replaced by direct indexing.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Dijkstra with Buckets)</span></p>

Every finite value is the length of some path — at most $n - 1$ edges, each of length at most $L$ — so $h(v) \in \lbrace 0, \dots, nL \rbrace$. Keep an array of $nL + 1$ **buckets**, where bucket $i$ is a doubly linked list of the open vertices with $h(v) = i$, plus a pointer to the leftmost non-empty bucket:

* **Insert** and **Decrease** in $O(1)$ — link the vertex into its bucket, resp. move it between buckets;
* **ExtractMin**: advance the pointer to the next non-empty bucket and pop a vertex from it. Values at closing never decrease, so the pointer only ever moves **right** — all scans together cost $O(nL)$.

Total running time: $O(m + nL)$.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 230" width="100%" style="max-width: 580px; height: auto;" role="img" aria-labelledby="gaf28-title">
  <title id="gaf28-title">Bucket array indexed by value, with a pointer to the leftmost non-empty bucket</title>
  <defs>
    <marker id="gaf28-arrr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#b91c1c"/>
    </marker>
  </defs>
  <rect x="30" y="60" width="560" height="36" fill="#fff" stroke="#333" stroke-width="1.8"/>
  <line x1="70" y1="60" x2="70" y2="96" stroke="#333" stroke-width="1"/>
  <line x1="110" y1="60" x2="110" y2="96" stroke="#333" stroke-width="1"/>
  <line x1="150" y1="60" x2="150" y2="96" stroke="#333" stroke-width="1"/>
  <line x1="190" y1="60" x2="190" y2="96" stroke="#333" stroke-width="1"/>
  <line x1="230" y1="60" x2="230" y2="96" stroke="#333" stroke-width="1"/>
  <line x1="270" y1="60" x2="270" y2="96" stroke="#333" stroke-width="1"/>
  <line x1="310" y1="60" x2="310" y2="96" stroke="#333" stroke-width="1"/>
  <line x1="350" y1="60" x2="350" y2="96" stroke="#333" stroke-width="1"/>
  <line x1="390" y1="60" x2="390" y2="96" stroke="#333" stroke-width="1"/>
  <line x1="430" y1="60" x2="430" y2="96" stroke="#333" stroke-width="1"/>
  <line x1="470" y1="60" x2="470" y2="96" stroke="#333" stroke-width="1"/>
  <line x1="510" y1="60" x2="510" y2="96" stroke="#333" stroke-width="1"/>
  <line x1="550" y1="60" x2="550" y2="96" stroke="#333" stroke-width="1"/>
  <line x1="34" y1="94" x2="60" y2="62" stroke="#0f9b6c" stroke-width="1.2"/>
  <line x1="47" y1="94" x2="73" y2="62" stroke="#0f9b6c" stroke-width="1.2"/>
  <line x1="60" y1="94" x2="86" y2="62" stroke="#0f9b6c" stroke-width="1.2"/>
  <line x1="73" y1="94" x2="99" y2="62" stroke="#0f9b6c" stroke-width="1.2"/>
  <line x1="86" y1="94" x2="112" y2="62" stroke="#0f9b6c" stroke-width="1.2"/>
  <line x1="99" y1="94" x2="125" y2="62" stroke="#0f9b6c" stroke-width="1.2"/>
  <line x1="112" y1="94" x2="138" y2="62" stroke="#0f9b6c" stroke-width="1.2"/>
  <line x1="125" y1="94" x2="151" y2="62" stroke="#0f9b6c" stroke-width="1.2"/>
  <line x1="138" y1="94" x2="164" y2="62" stroke="#0f9b6c" stroke-width="1.2"/>
  <line x1="151" y1="94" x2="177" y2="62" stroke="#0f9b6c" stroke-width="1.2"/>
  <line x1="164" y1="94" x2="188" y2="64" stroke="#0f9b6c" stroke-width="1.2"/>
  <text x="110" y="46" text-anchor="middle" font-family="serif" font-size="11" fill="#0f9b6c">already scanned — stays empty</text>
  <circle cx="210" cy="78" r="7" fill="#b91c1c"/>
  <line x1="260" y1="140" x2="220" y2="102" stroke="#b91c1c" stroke-width="1.6" marker-end="url(#gaf28-arrr)"/>
  <text x="350" y="152" text-anchor="middle" font-family="serif" font-size="11" fill="#b91c1c">leftmost non-empty bucket i</text>
  <line x1="210" y1="96" x2="210" y2="114" stroke="#1565c0" stroke-width="1.4"/>
  <circle cx="210" cy="120" r="6" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <line x1="210" y1="126" x2="210" y2="138" stroke="#1565c0" stroke-width="1.4"/>
  <circle cx="210" cy="144" r="6" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <line x1="210" y1="150" x2="210" y2="162" stroke="#1565c0" stroke-width="1.4"/>
  <circle cx="210" cy="168" r="6" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <text x="118" y="148" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#1565c0">all v with h(v) = i</text>
  <text x="50" y="112" text-anchor="middle" font-family="serif" font-size="11" fill="#333">0</text>
  <text x="570" y="112" text-anchor="middle" font-family="serif" font-size="11" fill="#333">nL</text>
  <text x="310" y="215" text-anchor="middle" font-family="serif" font-size="11" fill="#666">Insert and Decrease move vertices between buckets in O(1); the scan pointer only moves right</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The bucket array indexed by $h$-values $0, \dots, nL$: bucket $i$ holds the list of open vertices with $h(v) = i$. Since closing values never decrease, everything left of the pointer stays empty forever — the total scanning cost is $O(nL)$.
</figcaption>
</figure>

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Value Window)</span></p>

Let $c$ be the value of the most recently closed vertex. Then at every moment, all finite values of open vertices lie in the window $[c,\, c + L]$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Every open vertex $w$ received its current value by the relaxation of some already closed vertex $x$: $h(w) = h(x) + \ell(xw)$. Closing values are non-decreasing, so $h(x) \le c$, and hence $h(w) \le c + L$. The lower bound $h(w) \ge c$ is the Dijkstra invariant.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Saving Memory)</span></p>

At most $L + 1$ *consecutive* buckets can be non-empty at any moment, so it suffices to index the array **modulo $L + 1$**: the space drops from $O(nL + n)$ to $O(L + n)$, while the running time stays $O(m + nL)$.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 210" width="100%" style="max-width: 580px; height: auto;" role="img" aria-labelledby="gaf29-title">
  <title id="gaf29-title">Only L plus one consecutive buckets can be non-empty: index the array modulo L plus one</title>
  <defs>
    <marker id="gaf29-arro" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#a86f00"/>
    </marker>
    <marker id="gaf29-arrb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#1565c0"/>
    </marker>
  </defs>
  <rect x="190" y="58" width="200" height="36" fill="#e3f2fd" stroke="none"/>
  <rect x="30" y="58" width="560" height="36" fill="none" stroke="#333" stroke-width="1.8"/>
  <line x1="70" y1="58" x2="70" y2="94" stroke="#333" stroke-width="1"/>
  <line x1="110" y1="58" x2="110" y2="94" stroke="#333" stroke-width="1"/>
  <line x1="150" y1="58" x2="150" y2="94" stroke="#333" stroke-width="1"/>
  <line x1="190" y1="58" x2="190" y2="94" stroke="#333" stroke-width="1"/>
  <line x1="230" y1="58" x2="230" y2="94" stroke="#333" stroke-width="1"/>
  <line x1="270" y1="58" x2="270" y2="94" stroke="#333" stroke-width="1"/>
  <line x1="310" y1="58" x2="310" y2="94" stroke="#333" stroke-width="1"/>
  <line x1="350" y1="58" x2="350" y2="94" stroke="#333" stroke-width="1"/>
  <line x1="390" y1="58" x2="390" y2="94" stroke="#333" stroke-width="1"/>
  <line x1="430" y1="58" x2="430" y2="94" stroke="#333" stroke-width="1"/>
  <line x1="470" y1="58" x2="470" y2="94" stroke="#333" stroke-width="1"/>
  <line x1="510" y1="58" x2="510" y2="94" stroke="#333" stroke-width="1"/>
  <line x1="550" y1="58" x2="550" y2="94" stroke="#333" stroke-width="1"/>
  <path d="M 190,50 L 190,42 L 390,42 L 390,50" fill="none" stroke="#1565c0" stroke-width="1.2"/>
  <text x="290" y="34" text-anchor="middle" font-family="serif" font-size="11" fill="#1565c0">possibly non-empty — L+1 buckets</text>
  <circle cx="210" cy="76" r="6" fill="#1565c0"/>
  <line x1="210" y1="118" x2="210" y2="100" stroke="#1565c0" stroke-width="1.4" marker-end="url(#gaf29-arrb)"/>
  <text x="210" y="132" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#1565c0">current c</text>
  <text x="130" y="80" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#666">empty</text>
  <text x="490" y="80" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#666">empty</text>
  <path d="M 570,104 C 570,158 50,158 50,104" fill="none" stroke="#a86f00" stroke-width="1.6" stroke-dasharray="6 4" marker-end="url(#gaf29-arro)"/>
  <text x="310" y="180" text-anchor="middle" font-family="serif" font-size="11" fill="#a86f00">indices taken mod (L+1) — the window wraps around</text>
  <text x="310" y="202" text-anchor="middle" font-family="serif" font-size="11" fill="#666">h(open) ≤ c + L  ⇒  space O(L + n)</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
By the value-window lemma, all open vertices live in the $L+1$ buckets $c, c+1, \dots, c+L$ — everything else is empty. Indexing modulo $L+1$ lets the window wrap around a small circular array.
</figcaption>
</figure>

## Dijkstra as a Sorting Algorithm

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Sorting Reduces to Dijkstra)</span></p>

To sort non-negative reals $x\_1, \dots, x\_n$, build a **star**: a center $u$ and leaves $v\_1, \dots, v\_n$ with edges $uv\_i$ of length $\ell(uv\_i) = x\_i$. Then $d(u, v\_i) = x\_i$, so Dijkstra's algorithm closes the leaves in the order of increasing $x\_i$ — it sorts the numbers.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 560 250" width="100%" style="max-width: 480px; height: auto;" role="img" aria-labelledby="gaf30-title">
  <title id="gaf30-title">Sorting as Dijkstra on a star graph</title>
  <defs>
    <marker id="gaf30-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
    </marker>
  </defs>
  <line x1="280" y1="108" x2="280" y2="52" stroke="#333" stroke-width="1.6" marker-end="url(#gaf30-arr)"/>
  <text x="294" y="82" font-family="serif" font-size="13" font-style="italic" fill="#a86f00">x₁</text>
  <line x1="294" y1="117" x2="412" y2="70" stroke="#333" stroke-width="1.6" marker-end="url(#gaf30-arr)"/>
  <text x="358" y="80" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#a86f00">x₂</text>
  <line x1="295" y1="128" x2="445" y2="149" stroke="#333" stroke-width="1.6" marker-end="url(#gaf30-arr)"/>
  <text x="372" y="126" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#a86f00">x₃</text>
  <line x1="290" y1="138" x2="362" y2="200" stroke="#333" stroke-width="1.6" marker-end="url(#gaf30-arr)"/>
  <text x="340" y="180" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#a86f00">x₄</text>
  <line x1="266" y1="118" x2="128" y2="96" stroke="#333" stroke-width="1.6" marker-end="url(#gaf30-arr)"/>
  <text x="195" y="94" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#a86f00">xₙ</text>
  <text x="200" y="192" text-anchor="middle" font-family="serif" font-size="15" fill="#333">⋯</text>
  <circle cx="280" cy="125" r="15" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="280" y="130" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#0f9b6c">u</text>
  <circle cx="280" cy="34" r="14" fill="#fff" stroke="#1565c0" stroke-width="1.6"/>
  <text x="280" y="39" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">v₁</text>
  <circle cx="426" cy="64" r="14" fill="#fff" stroke="#1565c0" stroke-width="1.6"/>
  <text x="426" y="69" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">v₂</text>
  <circle cx="462" cy="152" r="14" fill="#fff" stroke="#1565c0" stroke-width="1.6"/>
  <text x="462" y="157" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">v₃</text>
  <circle cx="372" cy="212" r="14" fill="#fff" stroke="#1565c0" stroke-width="1.6"/>
  <text x="372" y="217" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">v₄</text>
  <circle cx="112" cy="93" r="14" fill="#fff" stroke="#1565c0" stroke-width="1.6"/>
  <text x="112" y="98" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">vₙ</text>
  <text x="280" y="244" text-anchor="middle" font-family="serif" font-size="11" fill="#666">Dijkstra closes the leaves in order of increasing xᵢ — i.e. it sorts x₁, …, xₙ</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The star graph with $\ell(uv_i) = x_i$: running Dijkstra from $u$ closes the leaves by increasing distance $d(u, v_i) = x_i$, i.e. outputs the numbers in sorted order.
</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Lower Bound for Real Lengths)</span></p>

Sorting $n$ reals requires $\Omega(n \log n)$ comparisons, so any comparison-based implementation of Dijkstra's algorithm must spend $\Omega(n \log n)$ time — already on stars with $m = n$ edges. The Fibonacci-heap bound $O(m + n \log n)$ is therefore **optimal for real edge lengths**.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Bound Is Comparison-Model Only)</span></p>

The lower bound only binds algorithms that access the lengths through comparisons. For instance, $n$ independent random numbers uniform in $[0,1)$ can be sorted in expected $O(n)$ time by bucketing — the very same trick that let the bucket implementation beat the bound for integer lengths.

</div>

# Lecture 6: Faster Buckets and Potentials

The bucket implementation from last lecture runs Dijkstra's algorithm in $O(m + nL)$ — fine for small $L$, but *linear* in the range of lengths. This lecture pushes the dependence on $L$ down to $\log L$, then to $\log L / \log\log L$ with **multi-level buckets**; extends buckets to non-integer lengths via **Dinitz's trick**; and finally removes the non-negativity assumption altogether using **potentials**.

## Trees over Buckets

Recall the setup: integer lengths $\ell(e) \in \lbrace 0, \dots, L \rbrace$, values $h(v) \in \lbrace 0, \dots, nL \rbrace$, an array of buckets, and the value window of width $L + 1$ that made the modulo trick work. The expensive part was ExtractMin's linear scanning; the first idea is to search for the leftmost non-empty bucket with a tree.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(A Tree over the Buckets)</span></p>

Build a complete binary tree whose leaves are the buckets $0, \dots, nL$; every inner node stores a single bit — *"is there a non-empty bucket in my subtree?"* Insert and Decrease update the bits along one leaf-to-root path; ExtractMin descends from the root, always into the leftmost child whose bit is set. Every operation costs

$$
O(\log (nL)) \;=\; O(\log n + \log L).
$$

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 250" width="100%" style="max-width: 560px; height: auto;" role="img" aria-labelledby="gaf31-title">
  <title id="gaf31-title">A binary tree over the bucket array finds the leftmost non-empty bucket</title>
  <text x="310" y="16" text-anchor="middle" font-family="serif" font-size="11" fill="#333">each node remembers: is there a non-empty bucket below me?</text>
  <line x1="310" y1="42" x2="196" y2="82" stroke="#0f9b6c" stroke-width="2.2"/>
  <line x1="310" y1="42" x2="424" y2="82" stroke="#333" stroke-width="1.4"/>
  <line x1="190" y1="92" x2="134" y2="130" stroke="#333" stroke-width="1.4"/>
  <line x1="190" y1="92" x2="246" y2="130" stroke="#0f9b6c" stroke-width="2.2"/>
  <line x1="430" y1="92" x2="374" y2="130" stroke="#333" stroke-width="1.4"/>
  <line x1="430" y1="92" x2="486" y2="130" stroke="#333" stroke-width="1.4"/>
  <line x1="124" y1="144" x2="105" y2="182" stroke="#666" stroke-width="1" stroke-dasharray="4 3"/>
  <line x1="136" y1="144" x2="155" y2="182" stroke="#666" stroke-width="1" stroke-dasharray="4 3"/>
  <line x1="250" y1="142" x2="235" y2="182" stroke="#0f9b6c" stroke-width="2.2"/>
  <line x1="256" y1="144" x2="275" y2="182" stroke="#666" stroke-width="1" stroke-dasharray="4 3"/>
  <line x1="364" y1="144" x2="345" y2="182" stroke="#666" stroke-width="1" stroke-dasharray="4 3"/>
  <line x1="376" y1="144" x2="395" y2="182" stroke="#666" stroke-width="1" stroke-dasharray="4 3"/>
  <line x1="484" y1="144" x2="465" y2="182" stroke="#666" stroke-width="1" stroke-dasharray="4 3"/>
  <line x1="496" y1="144" x2="515" y2="182" stroke="#666" stroke-width="1" stroke-dasharray="4 3"/>
  <circle cx="310" cy="36" r="8" fill="#fff" stroke="#0f9b6c" stroke-width="2"/>
  <circle cx="190" cy="86" r="7" fill="#fff" stroke="#0f9b6c" stroke-width="2"/>
  <circle cx="430" cy="86" r="7" fill="#fff" stroke="#333" stroke-width="1.5"/>
  <circle cx="130" cy="137" r="7" fill="#fff" stroke="#333" stroke-width="1.5"/>
  <circle cx="250" cy="137" r="7" fill="#fff" stroke="#0f9b6c" stroke-width="2"/>
  <circle cx="370" cy="137" r="7" fill="#fff" stroke="#333" stroke-width="1.5"/>
  <circle cx="490" cy="137" r="7" fill="#fff" stroke="#333" stroke-width="1.5"/>
  <rect x="220" y="186" width="30" height="30" fill="#ecfdf5" stroke="none"/>
  <rect x="70" y="186" width="480" height="30" fill="none" stroke="#333" stroke-width="1.8"/>
  <line x1="100" y1="186" x2="100" y2="216" stroke="#333" stroke-width="1"/>
  <line x1="130" y1="186" x2="130" y2="216" stroke="#333" stroke-width="1"/>
  <line x1="160" y1="186" x2="160" y2="216" stroke="#333" stroke-width="1"/>
  <line x1="190" y1="186" x2="190" y2="216" stroke="#333" stroke-width="1"/>
  <line x1="220" y1="186" x2="220" y2="216" stroke="#333" stroke-width="1"/>
  <line x1="250" y1="186" x2="250" y2="216" stroke="#333" stroke-width="1"/>
  <line x1="280" y1="186" x2="280" y2="216" stroke="#333" stroke-width="1"/>
  <line x1="310" y1="186" x2="310" y2="216" stroke="#333" stroke-width="1"/>
  <line x1="340" y1="186" x2="340" y2="216" stroke="#333" stroke-width="1"/>
  <line x1="370" y1="186" x2="370" y2="216" stroke="#333" stroke-width="1"/>
  <line x1="400" y1="186" x2="400" y2="216" stroke="#333" stroke-width="1"/>
  <line x1="430" y1="186" x2="430" y2="216" stroke="#333" stroke-width="1"/>
  <line x1="460" y1="186" x2="460" y2="216" stroke="#333" stroke-width="1"/>
  <line x1="490" y1="186" x2="490" y2="216" stroke="#333" stroke-width="1"/>
  <line x1="520" y1="186" x2="520" y2="216" stroke="#333" stroke-width="1"/>
  <circle cx="235" cy="201" r="5" fill="#0f9b6c"/>
  <text x="85" y="238" text-anchor="middle" font-family="serif" font-size="11" fill="#333">0</text>
  <text x="535" y="238" text-anchor="middle" font-family="serif" font-size="11" fill="#333">nL</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
A complete binary tree over the buckets $0, \dots, nL$. ExtractMin walks from the root towards the leftmost set bit (green path); Insert and Decrease refresh one leaf-to-root path — all in $O(\log n + \log L)$.
</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Reduction to Keys $\lbrace 0, \dots, L \rbrace$)</span></p>

The $\log n$ part is wasteful: by the value-window lemma, all live values fit into a window of width $L + 1$. Chop the array $0, \dots, nL$ into **blocks** of $L + 1$ consecutive buckets; the window meets at most **two** blocks, so at any moment at most two blocks are non-empty. Maintaining which two blocks are live costs $O(1)$ per operation, so *any* data structure handling keys $\lbrace 0, \dots, L \rbrace$ — one instance per live block — yields a structure for keys $\lbrace 0, \dots, nL \rbrace$ with constant overhead.

With a binary tree per block, every operation costs $O(\log L)$ and Dijkstra runs in $O(m \log L)$.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 220" width="100%" style="max-width: 580px; height: auto;" role="img" aria-labelledby="gaf32-title">
  <title id="gaf32-title">The bucket array chopped into blocks; at most two blocks are non-empty</title>
  <text x="310" y="26" text-anchor="middle" font-family="serif" font-size="11" fill="#333">the value window spans at most 2 blocks — everything else is empty</text>
  <path d="M 240,70 L 175,136 L 305,136 Z" fill="none" stroke="#0f9b6c" stroke-width="1.8"/>
  <path d="M 380,70 L 315,136 L 445,136 Z" fill="none" stroke="#0f9b6c" stroke-width="1.8"/>
  <rect x="30" y="140" width="560" height="30" fill="#fff" stroke="#333" stroke-width="1.8"/>
  <line x1="65" y1="140" x2="65" y2="170" stroke="#333" stroke-width="1"/>
  <line x1="100" y1="140" x2="100" y2="170" stroke="#333" stroke-width="1"/>
  <line x1="135" y1="140" x2="135" y2="170" stroke="#333" stroke-width="1"/>
  <line x1="205" y1="140" x2="205" y2="170" stroke="#333" stroke-width="1"/>
  <line x1="240" y1="140" x2="240" y2="170" stroke="#333" stroke-width="1"/>
  <line x1="275" y1="140" x2="275" y2="170" stroke="#333" stroke-width="1"/>
  <line x1="345" y1="140" x2="345" y2="170" stroke="#333" stroke-width="1"/>
  <line x1="380" y1="140" x2="380" y2="170" stroke="#333" stroke-width="1"/>
  <line x1="415" y1="140" x2="415" y2="170" stroke="#333" stroke-width="1"/>
  <line x1="485" y1="140" x2="485" y2="170" stroke="#333" stroke-width="1"/>
  <line x1="520" y1="140" x2="520" y2="170" stroke="#333" stroke-width="1"/>
  <line x1="555" y1="140" x2="555" y2="170" stroke="#333" stroke-width="1"/>
  <line x1="170" y1="134" x2="170" y2="176" stroke="#333" stroke-width="2.6"/>
  <line x1="310" y1="134" x2="310" y2="176" stroke="#333" stroke-width="2.6"/>
  <line x1="450" y1="134" x2="450" y2="176" stroke="#333" stroke-width="2.6"/>
  <circle cx="222" cy="155" r="5" fill="#0f9b6c"/>
  <circle cx="292" cy="155" r="5" fill="#0f9b6c"/>
  <circle cx="330" cy="155" r="5" fill="#0f9b6c"/>
  <circle cx="398" cy="155" r="5" fill="#0f9b6c"/>
  <text x="100" y="130" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#666">empty</text>
  <text x="520" y="130" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#666">empty</text>
  <path d="M 172,180 L 172,188 L 308,188 L 308,180" fill="none" stroke="#666" stroke-width="1.2"/>
  <text x="240" y="206" text-anchor="middle" font-family="serif" font-size="11" fill="#666">block = L+1 buckets, with its own little tree</text>
  <text x="45" y="190" text-anchor="middle" font-family="serif" font-size="11" fill="#333">0</text>
  <text x="573" y="190" text-anchor="middle" font-family="serif" font-size="11" fill="#333">nL</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Blocks of $L+1$ buckets: the value window touches at most two of them, so it suffices to run a data structure for keys $\lbrace 0, \dots, L \rbrace$ on the two live blocks — a general reduction from range $nL$ to range $L$ with $O(1)$ overhead.
</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(van Emde Boas Trees)</span></p>

The reduction accepts any structure for keys $\lbrace 0, \dots, L \rbrace$. A **van Emde Boas tree** (covered in the Data Structures course) performs all three operations in $O(\log\log L)$ per operation, so Dijkstra runs in $O(m \log\log L)$.

</div>

## Multi-Level Buckets

A different route beats the tree with plain arrays. Write the keys in base $B$ and bucket them digit by digit.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Multi-Level Buckets)</span></p>

Fix a **base** $B$ (a power of two) and let

$$
d \;:=\; \lfloor \log_B L \rfloor + 1 \;\in\; \Theta\!\left(\frac{\log L}{\log B}\right)
$$

be the number of base-$B$ **digits** needed for the keys. The structure consists of:

* a **stack of at most $d$ levels**, stored in an array — the top level corresponds to the most significant digit, each level below it to the next digit down;
* each level is an **array of $B$ buckets**, indexed by the possible values of its digit; each bucket is a doubly linked list of items;
* $\mu$ — the value last returned by ExtractMin;
* at every level, the **active bucket** is the one indexed by $\mu$'s digit at that level.

An item $x$ lives at the level of the most significant digit in which $x$ differs from $\mu$ (or at the bottommost existing level, if they agree there and above). Since every live key is $\ge \mu$, at that digit $x$ exceeds $\mu$ — so $x$ sits strictly to the *right* of the active bucket.

Two invariants: **(1)** at every level, all buckets preceding the active one are empty; **(2)** items never move *upwards*.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 560 130" width="100%" style="max-width: 440px; height: auto;" role="img" aria-labelledby="gaf33-title">
  <title id="gaf33-title">A key written as d digits in base B</title>
  <text x="95" y="68" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#333">x in base B:</text>
  <rect x="180" y="44" width="45" height="36" fill="#fff" stroke="#333" stroke-width="1.5"/>
  <rect x="225" y="44" width="45" height="36" fill="#fff" stroke="#333" stroke-width="1.5"/>
  <rect x="270" y="44" width="45" height="36" fill="#fff" stroke="#333" stroke-width="1.5"/>
  <rect x="315" y="44" width="45" height="36" fill="#fff" stroke="#333" stroke-width="1.5"/>
  <rect x="360" y="44" width="45" height="36" fill="#fff" stroke="#333" stroke-width="1.5"/>
  <text x="202" y="36" text-anchor="middle" font-family="serif" font-size="11" fill="#666">4</text>
  <text x="247" y="36" text-anchor="middle" font-family="serif" font-size="11" fill="#666">3</text>
  <text x="292" y="36" text-anchor="middle" font-family="serif" font-size="11" fill="#666">2</text>
  <text x="337" y="36" text-anchor="middle" font-family="serif" font-size="11" fill="#666">1</text>
  <text x="382" y="36" text-anchor="middle" font-family="serif" font-size="11" fill="#666">0</text>
  <text x="202" y="68" text-anchor="middle" font-family="serif" font-size="15" fill="#1565c0">3</text>
  <text x="247" y="68" text-anchor="middle" font-family="serif" font-size="15" fill="#1565c0">0</text>
  <text x="292" y="68" text-anchor="middle" font-family="serif" font-size="15" fill="#1565c0">2</text>
  <text x="337" y="68" text-anchor="middle" font-family="serif" font-size="15" fill="#1565c0">1</text>
  <text x="382" y="68" text-anchor="middle" font-family="serif" font-size="15" fill="#1565c0">2</text>
  <path d="M 180,92 L 180,100 L 405,100 L 405,92" fill="none" stroke="#333" stroke-width="1.2"/>
  <text x="292" y="120" text-anchor="middle" font-family="serif" font-size="12" fill="#333">d = ⌊log<tspan dy="4" font-size="9">B</tspan><tspan dy="-4"> L⌋ + 1 digits</tspan></text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Keys as $d$-digit numbers in base $B$ — digit position $d-1$ (leftmost) is the most significant.
</figcaption>
</figure>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 300" width="100%" style="max-width: 600px; height: auto;" role="img" aria-labelledby="gaf34-title">
  <title id="gaf34-title">The multi-level bucket structure: a stack of levels of B buckets each</title>
  <defs>
    <marker id="gaf34-arrg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#0f9b6c"/>
    </marker>
  </defs>
  <rect x="150" y="40" width="400" height="30" fill="#fff" stroke="#333" stroke-width="1.6"/>
  <line x1="200" y1="40" x2="200" y2="70" stroke="#333" stroke-width="1"/>
  <line x1="250" y1="40" x2="250" y2="70" stroke="#333" stroke-width="1"/>
  <line x1="300" y1="40" x2="300" y2="70" stroke="#333" stroke-width="1"/>
  <line x1="350" y1="40" x2="350" y2="70" stroke="#333" stroke-width="1"/>
  <line x1="400" y1="40" x2="400" y2="70" stroke="#333" stroke-width="1"/>
  <line x1="450" y1="40" x2="450" y2="70" stroke="#333" stroke-width="1"/>
  <line x1="500" y1="40" x2="500" y2="70" stroke="#333" stroke-width="1"/>
  <line x1="158" y1="66" x2="182" y2="44" stroke="#666" stroke-width="1"/>
  <line x1="170" y1="66" x2="194" y2="44" stroke="#666" stroke-width="1"/>
  <ellipse cx="225" cy="55" rx="26" ry="18" fill="none" stroke="#b91c1c" stroke-width="1.8"/>
  <circle cx="280" cy="55" r="5" fill="#0f9b6c"/>
  <circle cx="330" cy="55" r="5" fill="#0f9b6c"/>
  <circle cx="430" cy="55" r="5" fill="#0f9b6c"/>
  <text x="140" y="59" text-anchor="end" font-family="serif" font-size="11" fill="#333">level d−1 — leftmost digit</text>
  <rect x="150" y="110" width="400" height="30" fill="#fff" stroke="#333" stroke-width="1.6"/>
  <line x1="200" y1="110" x2="200" y2="140" stroke="#333" stroke-width="1"/>
  <line x1="250" y1="110" x2="250" y2="140" stroke="#333" stroke-width="1"/>
  <line x1="300" y1="110" x2="300" y2="140" stroke="#333" stroke-width="1"/>
  <line x1="350" y1="110" x2="350" y2="140" stroke="#333" stroke-width="1"/>
  <line x1="400" y1="110" x2="400" y2="140" stroke="#333" stroke-width="1"/>
  <line x1="450" y1="110" x2="450" y2="140" stroke="#333" stroke-width="1"/>
  <line x1="500" y1="110" x2="500" y2="140" stroke="#333" stroke-width="1"/>
  <line x1="158" y1="136" x2="182" y2="114" stroke="#666" stroke-width="1"/>
  <line x1="170" y1="136" x2="194" y2="114" stroke="#666" stroke-width="1"/>
  <line x1="208" y1="136" x2="232" y2="114" stroke="#666" stroke-width="1"/>
  <line x1="220" y1="136" x2="244" y2="114" stroke="#666" stroke-width="1"/>
  <line x1="258" y1="136" x2="282" y2="114" stroke="#666" stroke-width="1"/>
  <line x1="270" y1="136" x2="294" y2="114" stroke="#666" stroke-width="1"/>
  <ellipse cx="325" cy="125" rx="26" ry="18" fill="none" stroke="#b91c1c" stroke-width="1.8"/>
  <circle cx="380" cy="125" r="5" fill="#0f9b6c"/>
  <circle cx="480" cy="125" r="5" fill="#0f9b6c"/>
  <text x="140" y="129" text-anchor="end" font-family="serif" font-size="11" fill="#333">level d−2 — next digit</text>
  <text x="350" y="172" text-anchor="middle" font-family="serif" font-size="15" fill="#333">⋮</text>
  <rect x="150" y="200" width="400" height="30" fill="#fff" stroke="#333" stroke-width="1.6"/>
  <line x1="200" y1="200" x2="200" y2="230" stroke="#333" stroke-width="1"/>
  <line x1="250" y1="200" x2="250" y2="230" stroke="#333" stroke-width="1"/>
  <line x1="300" y1="200" x2="300" y2="230" stroke="#333" stroke-width="1"/>
  <line x1="350" y1="200" x2="350" y2="230" stroke="#333" stroke-width="1"/>
  <line x1="400" y1="200" x2="400" y2="230" stroke="#333" stroke-width="1"/>
  <line x1="450" y1="200" x2="450" y2="230" stroke="#333" stroke-width="1"/>
  <line x1="500" y1="200" x2="500" y2="230" stroke="#333" stroke-width="1"/>
  <ellipse cx="175" cy="215" rx="26" ry="18" fill="none" stroke="#b91c1c" stroke-width="1.8"/>
  <circle cx="225" cy="215" r="5" fill="#0f9b6c"/>
  <circle cx="275" cy="215" r="5" fill="#0f9b6c"/>
  <circle cx="325" cy="215" r="5" fill="#0f9b6c"/>
  <line x1="233" y1="215" x2="266" y2="215" stroke="#0f9b6c" stroke-width="1.4" marker-end="url(#gaf34-arrg)"/>
  <line x1="283" y1="215" x2="316" y2="215" stroke="#0f9b6c" stroke-width="1.4" marker-end="url(#gaf34-arrg)"/>
  <text x="140" y="219" text-anchor="end" font-family="serif" font-size="11" fill="#333">bottommost level</text>
  <path d="M 560,40 L 568,40 L 568,230 L 560,230" fill="none" stroke="#a86f00" stroke-width="1.4"/>
  <text x="600" y="139" text-anchor="middle" font-family="serif" font-size="11" fill="#a86f00">≤ d levels</text>
  <text x="350" y="262" text-anchor="middle" font-family="serif" font-size="11" fill="#b91c1c">red ring = active bucket, selected by μ's digit — buckets to its left are empty</text>
  <text x="350" y="284" text-anchor="middle" font-family="serif" font-size="11" fill="#666">an item sits at the topmost level where its digit differs from μ's</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The stack of levels, each an array of $B$ buckets — one array per digit position. The digits of $\mu$ mark the active buckets; every item lives at the level of its most significant disagreement with $\mu$, always to the right of the active bucket.
</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Operations on Multi-Level Buckets)</span></p>

* **Insert($x$):** compute

  $$
  i \;:=\; \left\lfloor \frac{\mathrm{MSB}(x \oplus \mu)}{\log B} \right\rfloor
  $$

  — the position of the most significant digit where $x$ and $\mu$ differ ($\mathrm{MSB}$ of the XOR is a single machine-word operation, and dividing by $\log B$ turns a bit position into a digit position). Put $x$ into the bucket given by its digit at level $\max(i,\ \text{bottommost existing level})$. Time $O(1)$.

* **Decrease:** unlink the item from its bucket and re-Insert it with the new value. Time $O(1)$. (A smaller key agrees with $\mu$ at least as far down as before, so the item moves down or stays — never up.)

* **ExtractMin:**
  1. Look for a non-empty bucket at the bottommost level, scanning from the active bucket to the right; whenever a level is exhausted, discard it and continue in the level above.
  2. If the bucket found lives at level $0$, all its items agree with $\mu$ in every digit above and share the digit at level $0$ — they have *equal* keys. Remove any of them.
  3. Otherwise scan the bucket's items, find and remove the minimum — it becomes the new $\mu$ — and, if the bucket is still non-empty, create a new level below the current one and distribute the remaining items into it by their next digit.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Amortized Complexity of Multi-Level Buckets)</span></p>

Multi-level buckets support Insert in amortized $O(B + d)$, Decrease in $O(1)$, and ExtractMin in amortized $O(1)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We account for everything ExtractMin does:

* **Scanning buckets and removing levels — $O(B)$ per level.** Each ExtractMin creates at most one level, and every level is scanned left-to-right exactly once over its lifetime before being discarded (the active bucket only moves right). Charge the $O(B)$ creation plus the $O(B)$ scan to the Insert of the item whose extraction created the level — every item triggers at most one ExtractMin, so each Insert is charged $O(B)$.

* **Distributing items — $O(1)$ per item moved.** By invariant 2, an item only ever moves downwards, so it is distributed at most $d$ times during its lifetime. Charge $O(d)$ to its Insert.

* **Finding the minimum — $O(1)$ per item scanned.** Every item touched by the min-scan is either extracted or distributed downwards right after, so these touches are covered by the previous two budgets.

What remains of ExtractMin — and all of Decrease — is $O(1)$ actual work.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Dijkstra with Multi-Level Buckets)</span></p>

Plugging $T\_I = O(B + d)$, $T\_X = O(1)$, $T\_D = O(1)$ into the heap formula gives

$$
O\!\left(m + n \left(B + \frac{\log L}{\log B}\right)\right).
$$

Choosing $B := \log L / \log\log L$ balances the two terms: $\log B = \log\log L - \log\log\log L = \Theta(\log\log L)$, so both $B$ and $d$ are $\Theta(\log L / \log\log L)$, and Dijkstra runs in

$$
O\!\left(m + n \cdot \frac{\log L}{\log\log L}\right).
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(HOT Queues)</span></p>

Even better: the **HOT queue** of Goldberg et al. — a *Heap On Top* of buckets — brings the amortized cost per Insert down to $O(\sqrt{\log L})$, giving Dijkstra in $O(m + n\sqrt{\log L})$.

</div>

## Non-Integer Lengths: Dinitz's Trick

Buckets seem to need integers — but a lower bound on the edge lengths is enough.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Dinitz's Trick)</span></p>

Suppose $\ell(e) \ge \delta$ for every edge $e$ and some $\delta > 0$. Then **every open vertex $v$ with**

$$
h(v) \;<\; \min_{u \text{ open}} h(u) + \delta
$$

**can be closed immediately.** Consequently we may bucket the vertices by $\lfloor h(v) / \delta \rfloor$: the map is monotone, and the leftmost non-empty bucket may be emptied wholesale. This is exactly Dijkstra with an integer bucket structure for

$$
L \;=\; \frac{\max_e \ell(e)}{\min_e \ell(e)}
$$

— only the *ratio* of the lengths matters.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Any future improvement of $h(v)$ would arrive by relaxing some vertex $t$ closed later; at that moment $h(t) \ge \min\_{u \text{ open}} h(u)$ (values at closing never decrease below the current open minimum), so the offer is

$$
h(t) + \ell(tv) \;\ge\; \min_{u \text{ open}} h(u) + \delta \;>\; h(v)
$$

— no improvement. So $h(v)$ is already final and $v$ may be closed. For the buckets: all vertices in the leftmost non-empty bucket of width $\delta$ lie within $\delta$ of the open minimum (which sits in the same bucket), so all of them are closable — in any order.

</details>
</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 190" width="100%" style="max-width: 580px; height: auto;" role="img" aria-labelledby="gaf35-title">
  <title id="gaf35-title">All open vertices within delta of the open minimum can be closed at once</title>
  <defs>
    <marker id="gaf35-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
    </marker>
    <marker id="gaf35-arrr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#b91c1c"/>
    </marker>
  </defs>
  <rect x="300" y="94" width="130" height="32" fill="#fff3e0" stroke="none"/>
  <line x1="50" y1="110" x2="580" y2="110" stroke="#333" stroke-width="1.6" marker-end="url(#gaf35-arr)"/>
  <circle cx="90" cy="110" r="6" fill="#0f9b6c"/>
  <circle cx="130" cy="110" r="6" fill="#0f9b6c"/>
  <circle cx="165" cy="110" r="6" fill="#0f9b6c"/>
  <text x="125" y="148" text-anchor="middle" font-family="serif" font-size="11" fill="#0f9b6c">already closed</text>
  <circle cx="300" cy="110" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <circle cx="300" cy="110" r="13" fill="none" stroke="#b91c1c" stroke-width="1.8"/>
  <line x1="300" y1="66" x2="300" y2="92" stroke="#b91c1c" stroke-width="1.4" marker-end="url(#gaf35-arrr)"/>
  <text x="300" y="58" text-anchor="middle" font-family="serif" font-size="11" fill="#b91c1c">min over open</text>
  <line x1="430" y1="100" x2="430" y2="120" stroke="#a86f00" stroke-width="1.6"/>
  <text x="435" y="140" text-anchor="middle" font-family="serif" font-size="11" fill="#a86f00">min + δ</text>
  <circle cx="350" cy="110" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="395" cy="110" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="490" cy="110" r="6" fill="#fff" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="540" cy="110" r="6" fill="#fff" stroke="#1565c0" stroke-width="1.6"/>
  <text x="516" y="82" text-anchor="middle" font-family="serif" font-size="11" fill="#1565c0">still open</text>
  <text x="360" y="163" text-anchor="middle" font-family="serif" font-size="11" fill="#a86f00">all closable at once</text>
  <text x="310" y="185" text-anchor="middle" font-family="serif" font-size="11" fill="#666">buckets of width δ: the leftmost non-empty bucket can be emptied wholesale</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Values on the number line: with every edge of length at least $\delta$, no future offer can undercut $\min + \delta$ — so every open vertex in the shaded band is already final and can be closed in any order.
</figcaption>
</figure>

## Negative Lengths: Potentials

Everything so far assumed $\ell \ge 0$. What if some lengths are negative — while the graph still has no negative cycles?

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(First Attempt: Shift All Lengths — FAIL)</span></p>

Add a constant $\Delta$ to every edge length until nothing is negative, then run Dijkstra? **This fails:** a path with $k$ edges grows by $k\Delta$, so paths with different numbers of edges shift by *different* amounts — the shortest path can change.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 250" width="100%" style="max-width: 560px; height: auto;" role="img" aria-labelledby="gaf36-title">
  <title id="gaf36-title">Adding a constant to every edge length changes which path is shortest</title>
  <defs>
    <marker id="gaf36-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
    </marker>
  </defs>
  <path d="M 84,110 C 200,40 420,40 536,110" fill="none" stroke="#333" stroke-width="1.8" marker-end="url(#gaf36-arr)"/>
  <text x="310" y="56" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#a86f00">2</text>
  <line x1="82" y1="131" x2="216" y2="172" stroke="#333" stroke-width="1.8" marker-end="url(#gaf36-arr)"/>
  <text x="140" y="164" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#a86f00">−1</text>
  <line x1="243" y1="180" x2="376" y2="180" stroke="#333" stroke-width="1.8" marker-end="url(#gaf36-arr)"/>
  <text x="310" y="168" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#a86f00">1</text>
  <line x1="403" y1="172" x2="537" y2="131" stroke="#333" stroke-width="1.8" marker-end="url(#gaf36-arr)"/>
  <text x="480" y="164" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#a86f00">1</text>
  <circle cx="70" cy="120" r="15" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="70" y="125" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#0f9b6c">s</text>
  <circle cx="230" cy="180" r="13" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="230" y="185" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">a</text>
  <circle cx="390" cy="180" r="13" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="390" y="185" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">b</text>
  <circle cx="550" cy="120" r="15" fill="#fef2f2" stroke="#b91c1c" stroke-width="2"/>
  <text x="550" y="125" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#b91c1c">t</text>
  <text x="310" y="219" text-anchor="middle" font-family="serif" font-size="11" fill="#0f9b6c">original: lower path −1+1+1 = 1 beats the direct edge 2</text>
  <text x="310" y="240" text-anchor="middle" font-family="serif" font-size="11" fill="#b91c1c">after +Δ = 1 on every edge: lower path 4, direct edge 3 — the order flips</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Shifting all lengths by $\Delta$ penalizes the three-edge path three times but the direct edge only once — the shortest $st$-path changes. A per-edge shift cannot work; the fix has to be per-*vertex*.
</figcaption>
</figure>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Potentials and Reduced Lengths)</span></p>

A **potential** is any function $\varphi \colon V \to \mathbb{R}$. The **reduced edge lengths** with respect to $\varphi$ are

$$
\ell_\varphi(uv) \;:=\; \ell(uv) + \varphi(u) - \varphi(v),
$$

and $\varphi$ is **feasible** if $\ell\_\varphi(e) \ge 0$ for every edge $e$. With a feasible potential the reduced graph has no negative edges — Dijkstra applies.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Reduced Lengths Preserve Shortest Paths)</span></p>

For any $uv$-path $P = v\_1 v\_2 \dots v\_k$ (with $v\_1 = u$, $v\_k = v$):

$$
\ell_\varphi(P) \;=\; \sum_{i} \bigl( \ell(v_i v_{i+1}) + \varphi(v_i) - \varphi(v_{i+1}) \bigr)
\;=\; \ell(P) + \varphi(u) - \varphi(v)
$$

— the potentials of all inner vertices telescope away. Every $uv$-path shifts by the *same* amount $\varphi(u) - \varphi(v)$, so shortest paths are preserved (and $d\_\varphi(u,v) = d(u,v) + \varphi(u) - \varphi(v)$ recovers the true distances). We can therefore find shortest paths in the reduced graph instead.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 190" width="100%" style="max-width: 600px; height: auto;" role="img" aria-labelledby="gaf37-title">
  <title id="gaf37-title">Potentials of inner vertices telescope along a path</title>
  <defs>
    <marker id="gaf37-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
    </marker>
  </defs>
  <line x1="97" y1="90" x2="226" y2="90" stroke="#333" stroke-width="1.8" marker-end="url(#gaf37-arr)"/>
  <line x1="254" y1="90" x2="386" y2="90" stroke="#333" stroke-width="1.8" marker-end="url(#gaf37-arr)"/>
  <line x1="414" y1="90" x2="543" y2="90" stroke="#333" stroke-width="1.8" marker-end="url(#gaf37-arr)"/>
  <text x="160" y="72" text-anchor="middle" font-family="serif" font-size="11" fill="#a86f00">+φ(u) −φ(v₂)</text>
  <text x="320" y="72" text-anchor="middle" font-family="serif" font-size="11" fill="#a86f00">+φ(v₂) −φ(v₃)</text>
  <text x="480" y="72" text-anchor="middle" font-family="serif" font-size="11" fill="#a86f00">+φ(v₃) −φ(v)</text>
  <circle cx="80" cy="90" r="15" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="80" y="95" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#0f9b6c">u</text>
  <circle cx="240" cy="90" r="13" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="240" y="95" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">v₂</text>
  <circle cx="400" cy="90" r="13" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="400" y="95" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">v₃</text>
  <circle cx="560" cy="90" r="15" fill="#fef2f2" stroke="#b91c1c" stroke-width="2"/>
  <text x="560" y="95" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#b91c1c">v</text>
  <text x="80" y="130" text-anchor="middle" font-family="serif" font-size="11" fill="#0f9b6c">+φ(u) survives</text>
  <text x="240" y="130" text-anchor="middle" font-family="serif" font-size="11" fill="#666">−φ(v₂) + φ(v₂) = 0</text>
  <text x="400" y="130" text-anchor="middle" font-family="serif" font-size="11" fill="#666">−φ(v₃) + φ(v₃) = 0</text>
  <text x="560" y="130" text-anchor="middle" font-family="serif" font-size="11" fill="#b91c1c">−φ(v) survives</text>
  <text x="320" y="170" text-anchor="middle" font-family="serif" font-size="12" fill="#333">ℓ<tspan dy="4" font-size="9">φ</tspan><tspan dy="-4">(P) = ℓ(P) + φ(u) − φ(v)</tspan></text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Along any $uv$-path the inner potentials cancel in pairs — only $\varphi(u)$ and $-\varphi(v)$ survive. All $uv$-paths shift by the same constant, so their relative order is untouched.
</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Feasible Potentials Exist)</span></p>

In every graph with no negative cycles, a feasible potential can be found in time $O(nm)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Extend $G$ to $G'$ by adding a **new vertex $s$** and edges $sv$ of length $0$ for every $v \in V$. The new vertex has no incoming edges, so no new cycles arise — $G'$ still has no negative cycles — and every vertex is reachable from $s$. Run BFM from $s$ in time $O(nm)$ and set

$$
\varphi(v) \;:=\; d(s, v) \qquad \text{(finite for every } v\text{)}.
$$

Why is $\varphi$ feasible? For every edge $uv$:

$$
\ell_\varphi(uv) \;=\; \ell(uv) + d(s,u) - d(s,v) \;\ge\; 0
\quad\iff\quad
d(s,v) \;\le\; d(s,u) + \ell(uv),
$$

which is exactly the triangle inequality for distances — valid since $G'$ has no negative cycles.

</details>
</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 240" width="100%" style="max-width: 560px; height: auto;" role="img" aria-labelledby="gaf38-title">
  <title id="gaf38-title">The construction: a new source s joined to every vertex by a zero-length edge</title>
  <defs>
    <marker id="gaf38-arrg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#0f9b6c"/>
    </marker>
  </defs>
  <text x="70" y="40" font-family="serif" font-size="13" font-style="italic" fill="#333">G′ = G + s</text>
  <path d="M 230,120 C 228,60 300,24 400,26 C 505,28 572,68 570,122 C 568,176 500,216 396,214 C 296,212 232,178 230,120 Z" fill="none" stroke="#1565c0" stroke-width="1.8"/>
  <text x="537" y="52" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#1565c0">G</text>
  <line x1="95" y1="114" x2="288" y2="84" stroke="#0f9b6c" stroke-width="1.6" marker-end="url(#gaf38-arrg)"/>
  <line x1="95" y1="122" x2="348" y2="142" stroke="#0f9b6c" stroke-width="1.6" marker-end="url(#gaf38-arrg)"/>
  <line x1="94" y1="112" x2="428" y2="82" stroke="#0f9b6c" stroke-width="1.6" marker-end="url(#gaf38-arrg)"/>
  <line x1="94" y1="127" x2="468" y2="148" stroke="#0f9b6c" stroke-width="1.6" marker-end="url(#gaf38-arrg)"/>
  <line x1="94" y1="117" x2="508" y2="110" stroke="#0f9b6c" stroke-width="1.6" marker-end="url(#gaf38-arrg)"/>
  <circle cx="300" cy="80" r="6" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <circle cx="360" cy="145" r="6" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <circle cx="440" cy="80" r="6" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <circle cx="480" cy="150" r="6" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <circle cx="520" cy="110" r="6" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <circle cx="80" cy="120" r="15" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="80" y="125" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#0f9b6c">s</text>
  <text x="175" y="70" text-anchor="middle" font-family="serif" font-size="11" fill="#a86f00">all new edges have length 0</text>
  <text x="320" y="232" text-anchor="middle" font-family="serif" font-size="11" fill="#666">φ(v) := d(s,v), computed by one run of BFM</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Adding $s$ with zero-length edges to all of $V$ makes every vertex reachable without creating cycles; the BFM distances $d(s, \cdot)$ are then a feasible potential by the triangle inequality.
</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Where Potentials Lead)</span></p>

Two directions open up from here:

* **APSP:** one BFM run yields a feasible $\varphi$, after which *all-pairs* shortest paths cost $n$ runs of Dijkstra on the reduced graph — translating back via $d(u,v) = d\_\varphi(u,v) - \varphi(u) + \varphi(v)$. This beats $n$ runs of BFM whenever Dijkstra's implementation beats $O(nm)$.
* **A$^\ast$:** for point-to-point queries (P2PSP), one can plug in a *heuristic* potential — a lower bound on the remaining distance to the target — without any precomputation; Dijkstra on those reduced lengths is exactly the A$^\ast$ search algorithm.

</div>

# Lecture 7: Goal-Directed Search and All-Pairs Shortest Paths

Both directions promised at the end of last lecture get delivered now: first the **point-to-point** tricks — stopping early, searching from both ends, and the A$^\ast$ algorithm together with a precise theorem saying that A$^\ast$ *is* Dijkstra with a potential — and then a new chapter on **all-pairs** shortest paths, starring the Floyd–Warshall algorithm and its far-reaching algebraic generalization.

## Point-to-Point Shortest Paths

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Stopping Early)</span></p>

For a P2PSP query $s \to t$, run Dijkstra from $s$ and **stop as soon as $t$ is closed** — correct because vertices are closed in the order of increasing distance. Still, in a road-network-like graph the search explores the whole ball of radius $d = d(s,t)$ around $s$: roughly $d^2$ vertices.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Bidirectional Dijkstra)</span></p>

Run two searches at once — from $s$ along the edges and from $t$ *against* them — and stop when the two frontiers meet (the exact stopping rule needs a little care). Each ball only has to reach radius $\approx d/2$, so in the grid picture the work drops to about $2 \cdot (d/2)^2 = d^2/2$ — half of the unidirectional search.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 260" width="100%" style="max-width: 600px; height: auto;" role="img" aria-labelledby="gaf39-title">
  <title id="gaf39-title">One ball of radius d versus two balls of radius d halves</title>
  <text x="150" y="24" text-anchor="middle" font-family="serif" font-size="12" fill="#333">Dijkstra, stop at t</text>
  <circle cx="150" cy="130" r="100" fill="none" stroke="#1565c0" stroke-width="1.8"/>
  <circle cx="150" cy="130" r="68" fill="none" stroke="#1565c0" stroke-width="1.2" stroke-dasharray="5 4"/>
  <circle cx="150" cy="130" r="36" fill="none" stroke="#1565c0" stroke-width="1.2" stroke-dasharray="5 4"/>
  <circle cx="150" cy="130" r="10" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="150" y="134" text-anchor="middle" font-family="serif" font-size="10" font-style="italic" fill="#0f9b6c">s</text>
  <circle cx="247" cy="130" r="9" fill="#fef2f2" stroke="#b91c1c" stroke-width="2"/>
  <text x="247" y="134" text-anchor="middle" font-family="serif" font-size="10" font-style="italic" fill="#b91c1c">t</text>
  <text x="150" y="252" text-anchor="middle" font-family="serif" font-size="11" fill="#666">≈ d² vertices</text>
  <text x="495" y="24" text-anchor="middle" font-family="serif" font-size="12" fill="#333">bidirectional</text>
  <circle cx="425" cy="130" r="60" fill="none" stroke="#1565c0" stroke-width="1.8"/>
  <circle cx="425" cy="130" r="32" fill="none" stroke="#1565c0" stroke-width="1.2" stroke-dasharray="5 4"/>
  <circle cx="565" cy="130" r="60" fill="none" stroke="#7b1fa2" stroke-width="1.8"/>
  <circle cx="565" cy="130" r="32" fill="none" stroke="#7b1fa2" stroke-width="1.2" stroke-dasharray="5 4"/>
  <circle cx="425" cy="130" r="10" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="425" y="134" text-anchor="middle" font-family="serif" font-size="10" font-style="italic" fill="#0f9b6c">s</text>
  <circle cx="565" cy="130" r="9" fill="#fef2f2" stroke="#b91c1c" stroke-width="2"/>
  <text x="565" y="134" text-anchor="middle" font-family="serif" font-size="10" font-style="italic" fill="#b91c1c">t</text>
  <circle cx="495" cy="130" r="5" fill="#a86f00"/>
  <text x="495" y="112" text-anchor="middle" font-family="serif" font-size="10" fill="#a86f00">meet</text>
  <text x="495" y="252" text-anchor="middle" font-family="serif" font-size="11" fill="#666">≈ 2·(d/2)² = d²/2 vertices</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
In grid-like graphs a Dijkstra ball of radius $d$ holds $\approx d^2$ vertices; two balls of radius $d/2$ growing towards each other hold only half as many.
</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(A$^\ast$)</span></p>

Pick a **heuristic function** $\psi \colon V \to \mathbb{R}$ — necessarily *problem-specific* — that lower-bounds the remaining distance to the target:

$$
\psi(v) \;\le\; d(v, t) \qquad \text{for all } v.
$$

Run Dijkstra from $s$, but close the open vertex minimizing $h(v) + \psi(v)$ — the estimated length of an $st$-path *through* $v$ — and stop when $t$ closes. A typical choice in road networks: $\psi(v) :=$ the Euclidean distance from $v$ to $t$.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 240" width="100%" style="max-width: 560px; height: auto;" role="img" aria-labelledby="gaf40-title">
  <title id="gaf40-title">A star explores an elongated region towards the target instead of a full ball</title>
  <text x="310" y="28" text-anchor="middle" font-family="serif" font-size="11" fill="#0f9b6c">A* with a distance-to-t heuristic: a region stretched towards t</text>
  <circle cx="100" cy="130" r="88" fill="none" stroke="#666" stroke-width="1.4" stroke-dasharray="6 4"/>
  <ellipse cx="310" cy="130" rx="230" ry="52" fill="none" stroke="#0f9b6c" stroke-width="1.8"/>
  <line x1="150" y1="100" x2="120" y2="130" stroke="#0f9b6c" stroke-width="1"/>
  <line x1="205" y1="95" x2="170" y2="130" stroke="#0f9b6c" stroke-width="1"/>
  <line x1="260" y1="92" x2="225" y2="127" stroke="#0f9b6c" stroke-width="1"/>
  <line x1="315" y1="92" x2="280" y2="127" stroke="#0f9b6c" stroke-width="1"/>
  <line x1="370" y1="92" x2="335" y2="127" stroke="#0f9b6c" stroke-width="1"/>
  <line x1="425" y1="95" x2="390" y2="130" stroke="#0f9b6c" stroke-width="1"/>
  <line x1="478" y1="100" x2="448" y2="130" stroke="#0f9b6c" stroke-width="1"/>
  <line x1="112" y1="118" x2="497" y2="118" stroke="#a86f00" stroke-width="1.2" stroke-dasharray="2 4"/>
  <circle cx="100" cy="130" r="11" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="100" y="134" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#0f9b6c">s</text>
  <circle cx="520" cy="130" r="11" fill="#fef2f2" stroke="#b91c1c" stroke-width="2"/>
  <text x="520" y="134" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#b91c1c">t</text>
  <text x="100" y="236" text-anchor="middle" font-family="serif" font-size="11" fill="#666">plain Dijkstra ball</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The bias $h(v) + \psi(v)$ makes vertices "in the direction of $t$" look cheaper, so A$^\ast$ closes an elongated corridor (green) instead of the full Dijkstra ball (dashed).
</figcaption>
</figure>

## A$^\ast$ Is Dijkstra with a Potential

Written with reduced lengths, closing by $h(v) + \psi(v)$ smells of potentials — and indeed A$^\ast$ with heuristic $\psi$ is nothing but Dijkstra on the graph reduced by the potential $-\psi$. For that graph to have non-negative lengths we *want*

$$
\ell_{-\psi}(uv) \;=\; \ell(uv) - \psi(u) + \psi(v) \;\ge\; 0
\qquad \text{for every edge } uv.
$$

The ideal heuristic $\psi(v) = d(v,t)$ satisfies this — it is exactly the triangle inequality $d(u,t) \le \ell(uv) + d(v,t)$ — and so does the Euclidean heuristic whenever edge lengths dominate straight-line distances.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(A$^\ast$ $=$ Dijkstra on $G\_{-\psi}$)</span></p>

Run A$^\ast$ on $G$ from $s$ with values $h^\ast$, and Dijkstra on $G\_{-\psi}$ from $s$ with values $h$, breaking ties in the same way. Then in every step of the two runs,

$$
h(v) \;=\; h^\ast(v) - \psi(s) + \psi(v) \qquad \text{for all } v,
$$

and both algorithms select the same vertices in the same order.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Induction over the steps; A$^\ast$ minimizes $h^\ast(v) + \psi(v)$, Dijkstra minimizes $h(v)$.

**At the beginning:** $h^\ast(s) = 0$ and $h(s) = 0 - \psi(s) + \psi(s) = 0$; all other values are $+\infty$ in both runs. ✓

**Selection:** if the value identity holds, then

$$
h(v) \;=\; \bigl(h^\ast(v) + \psi(v)\bigr) - \psi(s)
$$

differs from A$^\ast$'s selection key only by the constant $\psi(s)$ — so the two algorithms pick the same open vertex (ties broken identically).

**Update:** relaxing the edge $uv$, A$^\ast$ offers $h^\ast(u) + \ell(uv)$, while Dijkstra offers

$$
h(u) + \ell_{-\psi}(uv)
= \bigl(h^\ast(u) - \psi(s) + \psi(u)\bigr) + \bigl(\ell(uv) - \psi(u) + \psi(v)\bigr)
= \bigl(h^\ast(u) + \ell(uv)\bigr) - \psi(s) + \psi(v),
$$

so the identity survives every update. ∎

</details>
</div>

In particular A$^\ast$ with a feasible $\psi$ inherits all of Dijkstra's guarantees — correctness, each vertex closed once — while exploring the goal-directed region.

## All-Pairs Shortest Paths

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Two Matrix Problems)</span></p>

* **APSP:** given the $n \times n$ matrix $L$ of edge lengths ($\ell(ij)$, or $+\infty$ for non-edges), compute the matrix $D$ of distances $d(i,j)$.
* **Transitive closure** (directed reachability): given the adjacency matrix $A$, compute $R = A^\ast$ with $R\_{ij} = 1$ iff there is a path $i \to j$, else $0$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Repeated SSSP)</span></p>

* Transitive closure: $n$ runs of BFS — $O(nm)$.
* APSP with non-negative lengths: $n$ runs of Dijkstra — $O(n(m + n \log n)) = O(nm + n^2 \log n)$ with Fibonacci heaps. (Negative lengths: first one BFM run for a feasible potential — Johnson's scheme from last lecture.)

This is the method of choice for **sparse** graphs; for dense ones ($m \sim n^2$) it degrades to $O(n^3)$ — which the next algorithm achieves directly, with negative edges allowed and a tiny constant.

</div>

## The Floyd–Warshall Algorithm

Number the vertices $V = \lbrace 1, \dots, n \rbrace$ and assume, as always, no negative cycles.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Restricted Distances)</span></p>

For $k \in \lbrace 0, \dots, n \rbrace$ let

$$
D^k_{ij} \;:=\; \min \bigl\lbrace \ell(w) : w \text{ is an } ij\text{-walk with all internal vertices in } \lbrace 1, \dots, k \rbrace \bigr\rbrace.
$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Endpoints of the Scale)</span></p>

$D^0$ is the input: $D^0\_{ij} = \ell(ij)$ (or $+\infty$ for non-edges) and $D^0\_{ii} = 0$ — the matrix $L$ with zeros on the diagonal. And $D^n = D$, the full distance matrix.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(One Step of Floyd–Warshall)</span></p>

For every $k \ge 1$:

$$
D^k_{ij} \;=\; \min\bigl( D^{k-1}_{ij},\;\; D^{k-1}_{ik} + D^{k-1}_{kj} \bigr),
$$

so $D^{k-1} \to D^k$ costs $O(1)$ per entry, $O(n^2)$ per step — and $D^0 \to D^1 \to \dots \to D^n$ solves APSP in $O(n^3)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

"$\le$": both candidates describe $ij$-walks with internal vertices in $\lbrace 1, \dots, k \rbrace$ — the first avoids $k$, the second passes through it.

"$\ge$": take an optimal $ij$-walk $w$ with internals in $\lbrace 1, \dots, k \rbrace$. If it avoids $k$, its length is at least $D^{k-1}\_{ij}$. If it visits $k$ several times, cut out the cycles between consecutive visits — they have non-negative length (no negative cycles) — so WLOG $w$ visits $k$ once and splits there into an $ik$-walk and a $kj$-walk, both with internals in $\lbrace 1, \dots, k-1 \rbrace$: length at least $D^{k-1}\_{ik} + D^{k-1}\_{kj}$. ∎

</details>
</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 250" width="100%" style="max-width: 580px; height: auto;" role="img" aria-labelledby="gaf41-title">
  <title id="gaf41-title">An ij-walk either avoids vertex k or splits at k into two walks</title>
  <defs>
    <marker id="gaf41-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
    </marker>
    <marker id="gaf41-arrg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#0f9b6c"/>
    </marker>
  </defs>
  <ellipse cx="310" cy="135" rx="195" ry="78" fill="none" stroke="#1565c0" stroke-width="1.6" stroke-dasharray="7 5"/>
  <text x="310" y="196" text-anchor="middle" font-family="serif" font-size="11" fill="#1565c0">internal vertices ∈ {1, …, k−1}</text>
  <path d="M 105,140 C 180,185 440,185 515,140" fill="none" stroke="#333" stroke-width="1.8" stroke-dasharray="6 4" marker-end="url(#gaf41-arr)"/>
  <text x="310" y="152" text-anchor="middle" font-family="serif" font-size="12" fill="#333">Dᵏ⁻¹ᵢⱼ — avoids k</text>
  <path d="M 103,128 C 160,92 235,72 294,70" fill="none" stroke="#0f9b6c" stroke-width="1.8" marker-end="url(#gaf41-arrg)"/>
  <path d="M 326,70 C 385,72 460,92 517,128" fill="none" stroke="#0f9b6c" stroke-width="1.8" marker-end="url(#gaf41-arrg)"/>
  <text x="185" y="66" text-anchor="middle" font-family="serif" font-size="12" fill="#0f9b6c">Dᵏ⁻¹ᵢₖ</text>
  <text x="435" y="66" text-anchor="middle" font-family="serif" font-size="12" fill="#0f9b6c">Dᵏ⁻¹ₖⱼ</text>
  <circle cx="310" cy="70" r="14" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <text x="310" y="75" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#1565c0">k</text>
  <circle cx="90" cy="135" r="15" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="90" y="140" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#0f9b6c">i</text>
  <circle cx="530" cy="135" r="15" fill="#fef2f2" stroke="#b91c1c" stroke-width="2"/>
  <text x="530" y="140" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#b91c1c">j</text>
  <text x="310" y="240" text-anchor="middle" font-family="serif" font-size="11" fill="#666">an optimal walk through k splits at k (visits beyond the first are cut — no negative cycles)</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The Floyd–Warshall step: the best $ij$-walk with internals in $\lbrace 1,\dots,k \rbrace$ either avoids $k$ entirely or splits at $k$ into two walks with internals in $\lbrace 1,\dots,k-1 \rbrace$.
</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(In Place: $\Theta(n^2)$ Memory)</span></p>

There is no need to keep two matrices. During pass $k$, row $k$ and column $k$ do not change: the update would set $D\_{ik} \leftarrow \min(D\_{ik},\, D\_{ik} + D\_{kk})$, and $D\_{kk} = 0$ (no negative cycles). So $D^k\_{ik} = D^{k-1}\_{ik}$ and $D^k\_{kj} = D^{k-1}\_{kj}$ — the values the update reads are the same in the old and new matrix, and everything can be done in place:

1. **For** $k = 1$ **to** $n$:
2. &nbsp;&nbsp;&nbsp;**for** $i = 1$ **to** $n$, **for** $j = 1$ **to** $n$:
3. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$D\_{ij} \leftarrow \min(D\_{ij},\; D\_{ik} + D\_{kj})$.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 250" width="100%" style="max-width: 500px; height: auto;" role="img" aria-labelledby="gaf42-title">
  <title id="gaf42-title">Row k and column k of the matrix stay fixed during pass k</title>
  <rect x="220" y="40" width="180" height="180" fill="#fff" stroke="#333" stroke-width="1.8"/>
  <rect x="220" y="118" width="180" height="24" fill="#fff3e0" stroke="none"/>
  <rect x="298" y="40" width="24" height="180" fill="#fff3e0" stroke="none"/>
  <rect x="220" y="40" width="180" height="180" fill="none" stroke="#333" stroke-width="1.8"/>
  <text x="435" y="134" font-family="serif" font-size="11" fill="#a86f00">row k — unchanged</text>
  <text x="310" y="28" text-anchor="middle" font-family="serif" font-size="11" fill="#a86f00">column k — unchanged</text>
  <circle cx="250" cy="80" r="5" fill="#b91c1c"/>
  <text x="250" y="68" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#b91c1c">Dᵢⱼ</text>
  <circle cx="310" cy="80" r="4" fill="#a86f00"/>
  <circle cx="250" cy="130" r="4" fill="#a86f00"/>
  <line x1="256" y1="80" x2="302" y2="80" stroke="#666" stroke-width="1" stroke-dasharray="3 3"/>
  <line x1="250" y1="86" x2="250" y2="124" stroke="#666" stroke-width="1" stroke-dasharray="3 3"/>
  <text x="205" y="244" font-family="serif" font-size="11" fill="#666">the update reads Dᵢₖ and Dₖⱼ — both live in the fixed row and column</text>
  <text x="410" y="230" font-family="serif" font-size="13" font-style="italic" fill="#333">D</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
In pass $k$ every update reads only entries of row $k$ and column $k$ (amber), which pass $k$ itself never modifies — old and new values coincide, so one matrix suffices.
</figcaption>
</figure>

## Walk Algebra

Floyd–Warshall secretly computes much more than distances. To see it, replace numbers by *sets of walks*.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Bundles and Their Algebra)</span></p>

A **$uv$-bundle** is a set of $uv$-walks. The **elementary bundles**:

* $\emptyset$ — the empty bundle;
* $e\_{uv} := \lbrace uv \rbrace$ for an edge $uv \in E$ — a $uv$-bundle containing the single one-edge walk;
* $\varepsilon\_u$ — the single-vertex walk at $u$, a $uu$-bundle.

The **operations**:

* $x \cup y$ — union of two $uv$-bundles (a $uv$-bundle);
* $x \cdot y$ — for a $uv$-bundle $x$ and a $vw$-bundle $y$: the set of all concatenations of a walk from $x$ with a walk from $y$ (a $uw$-bundle);
* $x^\ast := \varepsilon\_u \cup x \cup x{\cdot}x \cup x{\cdot}x{\cdot}x \cup \dots$ — **iteration** of a $uu$-bundle.

The whole setup mirrors *regular expressions* — with vertices for states and walks for words.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 560 180" width="100%" style="max-width: 480px; height: auto;" role="img" aria-labelledby="gaf43-title">
  <title id="gaf43-title">Concatenating a uv-bundle with a vw-bundle gives a uw-bundle</title>
  <defs>
    <marker id="gaf43-arrg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#0f9b6c"/>
    </marker>
    <marker id="gaf43-arrb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#1565c0"/>
    </marker>
  </defs>
  <path d="M 115,80 C 160,50 235,50 268,80" fill="none" stroke="#0f9b6c" stroke-width="1.8" marker-end="url(#gaf43-arrg)"/>
  <path d="M 115,100 C 160,130 235,130 268,100" fill="none" stroke="#0f9b6c" stroke-width="1.8" marker-end="url(#gaf43-arrg)"/>
  <path d="M 295,80 C 340,50 415,50 448,80" fill="none" stroke="#1565c0" stroke-width="1.8" marker-end="url(#gaf43-arrb)"/>
  <path d="M 295,100 C 340,130 415,130 448,100" fill="none" stroke="#1565c0" stroke-width="1.8" marker-end="url(#gaf43-arrb)"/>
  <text x="190" y="38" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#0f9b6c">x — uv-bundle</text>
  <text x="370" y="38" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">y — vw-bundle</text>
  <circle cx="100" cy="90" r="14" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="100" y="95" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#0f9b6c">u</text>
  <circle cx="281" cy="90" r="13" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="281" y="95" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">v</text>
  <circle cx="462" cy="90" r="14" fill="#fef2f2" stroke="#b91c1c" stroke-width="2"/>
  <text x="462" y="95" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#b91c1c">w</text>
  <text x="280" y="166" text-anchor="middle" font-family="serif" font-size="11" fill="#666">x·y = all concatenations, one walk from x followed by one from y — a uw-bundle</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The concatenation $x \cdot y$ pairs every walk of $x$ with every walk of $y$ — bundles compose like regular expressions.
</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Generalized Floyd–Warshall)</span></p>

Let $W^k\_{ij}$ be the bundle of *all* $ij$-walks with internal vertices in $\lbrace 1, \dots, k \rbrace$. Initialize

$$
W^0_{ij} = \begin{cases} e_{ij} & i \ne j,\ ij \in E \\ \varepsilon_i \cup e_{ii} & i = j \text{ (with } e_{ii} \text{ only if the loop exists)} \\ \emptyset & \text{otherwise,} \end{cases}
$$

and perform the step $W^{k-1} \to W^k$ via

$$
W^k_{ij} \;=\; W^{k-1}_{ij} \;\cup\; W^{k-1}_{ik} \cdot \bigl(W^{k-1}_{kk}\bigr)^{\!*} \cdot\, W^{k-1}_{kj}.
$$

The output is $W^n$ — all walks, sorted by endpoints. Note the $(W^{k-1}\_{kk})^\ast$ factor: unlike the numeric version, the bundle version keeps walks that revisit $k$, looping any number of times.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Representation: Walk Expressions)</span></p>

The bundles themselves are exponentially large (often infinite) — so we never store them as sets. Instead, each $W^k\_{ij}$ is a **walk expression** over $(\emptyset,\, e\_{uv},\, \varepsilon\_u,\, \cup,\, \cdot,\, {}^\ast)$, and all the expressions together form a **DAG with $O(n^3)$ nodes**: each of the $n$ steps adds $O(n^2)$ new operation nodes whose children are (shared) older subexpressions.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 560 230" width="100%" style="max-width: 440px; height: auto;" role="img" aria-labelledby="gaf44-title">
  <title id="gaf44-title">Walk expressions form a DAG with shared subexpressions</title>
  <text x="280" y="20" text-anchor="middle" font-family="serif" font-size="11" fill="#666">(x ∪ y)·z and z* share the subexpression z</text>
  <line x1="296" y1="62" x2="243" y2="103" stroke="#333" stroke-width="1.4"/>
  <line x1="322" y1="60" x2="416" y2="103" stroke="#333" stroke-width="1.4"/>
  <line x1="222" y1="130" x2="188" y2="167" stroke="#333" stroke-width="1.4"/>
  <line x1="242" y1="131" x2="272" y2="167" stroke="#333" stroke-width="1.4"/>
  <line x1="420" y1="132" x2="368" y2="169" stroke="#333" stroke-width="1.4"/>
  <line x1="313" y1="63" x2="352" y2="166" stroke="#333" stroke-width="1.4"/>
  <circle cx="310" cy="50" r="16" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <text x="310" y="58" text-anchor="middle" font-family="serif" font-size="22" fill="#1565c0">·</text>
  <circle cx="232" cy="116" r="16" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <text x="232" y="121" text-anchor="middle" font-family="serif" font-size="13" fill="#1565c0">∪</text>
  <circle cx="428" cy="116" r="16" fill="#f3e5f5" stroke="#7b1fa2" stroke-width="1.8"/>
  <text x="428" y="122" text-anchor="middle" font-family="serif" font-size="14" fill="#7b1fa2">✱</text>
  <circle cx="180" cy="182" r="15" fill="#fff" stroke="#0f9b6c" stroke-width="1.8"/>
  <text x="180" y="187" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#0f9b6c">x</text>
  <circle cx="280" cy="182" r="15" fill="#fff" stroke="#0f9b6c" stroke-width="1.8"/>
  <text x="280" y="187" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#0f9b6c">y</text>
  <circle cx="360" cy="182" r="15" fill="#fff" stroke="#0f9b6c" stroke-width="1.8"/>
  <text x="360" y="187" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#0f9b6c">z</text>
  <text x="280" y="222" text-anchor="middle" font-family="serif" font-size="11" fill="#666">operation nodes point at shared older subexpressions — O(n³) nodes in total</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Expressions are kept as a DAG, not as trees: subexpressions built in earlier Floyd–Warshall steps are referenced, never copied, keeping the total size at $O(n^3)$.
</figcaption>
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Evaluating by a Homomorphism)</span></p>

To extract an actual quantity, map the walk algebra homomorphically to values: choose an interpretation of $\bigl(\emptyset,\ e\_{uv},\ \varepsilon\_u,\ \cup,\ \cdot,\ {}^\ast\bigr)$ and evaluate the expression DAG bottom-up — $O(1)$ per node, $O(n^3)$ total. **Distances** arise from $f(x) := \min\_{p \in x} \ell(p)$:

$$
f(\emptyset) = +\infty, \qquad f(e_{uv}) = \ell(uv), \qquad f(\varepsilon_u) = 0,
$$

$$
f(x \cup y) = \min(f(x), f(y)), \qquad f(x \cdot y) = f(x) + f(y), \qquad
f(x^*) = \begin{cases} 0 & f(x) \ge 0 \\ -\infty & f(x) < 0. \end{cases}
$$

This reproduces Floyd–Warshall — and the $\ast$-rule even handles negative cycles gracefully, reporting $-\infty$ distances instead of requiring the NNC assumption.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Widest Paths)</span></p>

Give each edge a **width** and look for the $uv$-walk maximizing its narrowest edge: $f(\text{walk}) := \min\_{e \in \text{walk}} \mathrm{width}(e)$ and $f(\text{bundle}) := \max\_{\text{walk} \in \text{bundle}} f(\text{walk})$. The same skeleton applies with

$$
f(\emptyset) = -\infty, \qquad f(e_{uv}) = \mathrm{width}(uv), \qquad f(\varepsilon_u) = +\infty,
$$

$$
f(x \cup y) = \max(f(x), f(y)), \qquad f(x \cdot y) = \min(f(x), f(y)), \qquad f(x^*) = +\infty,
$$

and generalized Floyd–Warshall computes all-pairs widest paths in $O(n^3)$ time.

</div>

# Lecture 8: All-Pairs Shortest Paths via Matrix Multiplication

Floyd–Warshall solves both matrix problems from last lecture in $O(n^3)$. To go below the cube, we bring in the heavy machinery: **fast matrix multiplication**. The plan — count walks with matrix powers, generalize the product to other algebras, kill the log factor for reachability by divide and conquer, and finish with **Seidel's algorithm** for undirected unit-length APSP.

## Fast Matrix Multiplication

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(The Exponent $\omega$)</span></p>

Multiplying two $n \times n$ matrices over a ring costs:

* $O(n^3)$ — by definition;
* $O(n^{2.808})$ — Strassen 1969;
* $O(n^{2.376})$ — Coppersmith &amp; Winograd 1990;
* $O(n^{2.373})$ — Williams 2012;
* conjecturally $O(n^{2+\varepsilon})$ for every $\varepsilon > 0$; only restricted lower bounds around $\Omega(n^2 \log n)$ are known.

We write $O(n^\omega)$ for the cost of one multiplication and use it as a black box.

</div>

## Walks and Matrix Powers

Let $A$ be the adjacency matrix and — in this lecture — let $E$ denote the *identity* matrix (the edge set never appears as a matrix, so no confusion threatens).

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Powers Count Walks)</span></p>

$A^k\_{ij}$ = the number of $ij$-walks with **exactly $k$ edges**.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Induction on $k$, with $A^1 = A$ counting one-edge walks. For the step, let $M = A^{k-1}$ count walks with $k-1$ edges; then

$$
(M \cdot A)_{ij} \;=\; \sum_t M_{it} \, A_{tj} \;=\; \sum_{t:\ tj \in E(G)} M_{it}
$$

— every walk with $k$ edges is a walk with $k-1$ edges ending at some $t$, extended by a last edge $tj$. ∎

</details>
</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 220" width="100%" style="max-width: 560px; height: auto;" role="img" aria-labelledby="gaf45-title">
  <title id="gaf45-title">A walk with k edges is a walk with k minus one edges plus a last edge</title>
  <defs>
    <marker id="gaf45-arrg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#0f9b6c"/>
    </marker>
    <marker id="gaf45-arrr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#b91c1c"/>
    </marker>
  </defs>
  <path d="M 86,102 C 150,60 280,42 336,56" fill="none" stroke="#0f9b6c" stroke-width="1.8" marker-end="url(#gaf45-arrg)"/>
  <path d="M 88,110 C 170,100 270,104 334,108" fill="none" stroke="#0f9b6c" stroke-width="1.8" marker-end="url(#gaf45-arrg)"/>
  <path d="M 86,118 C 150,160 280,178 336,164" fill="none" stroke="#0f9b6c" stroke-width="1.8" marker-end="url(#gaf45-arrg)"/>
  <text x="205" y="36" text-anchor="middle" font-family="serif" font-size="11" fill="#0f9b6c">walks with k−1 edges — counted by Aᵏ⁻¹ᵢₜ</text>
  <line x1="366" y1="58" x2="524" y2="102" stroke="#b91c1c" stroke-width="1.8" marker-end="url(#gaf45-arrr)"/>
  <line x1="366" y1="110" x2="522" y2="110" stroke="#b91c1c" stroke-width="1.8" marker-end="url(#gaf45-arrr)"/>
  <line x1="366" y1="162" x2="524" y2="118" stroke="#b91c1c" stroke-width="1.8" marker-end="url(#gaf45-arrr)"/>
  <text x="452" y="86" text-anchor="middle" font-family="serif" font-size="11" fill="#b91c1c">last edge tj ∈ E(G)</text>
  <circle cx="70" cy="110" r="15" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="70" y="115" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#0f9b6c">i</text>
  <circle cx="352" cy="56" r="12" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="352" cy="110" r="12" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="352" y="115" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">t</text>
  <circle cx="352" cy="164" r="12" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="540" cy="110" r="15" fill="#fef2f2" stroke="#b91c1c" stroke-width="2"/>
  <text x="540" y="115" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#b91c1c">j</text>
  <text x="310" y="208" text-anchor="middle" font-family="serif" font-size="11" fill="#666">(Aᵏ)ᵢⱼ = Σₜ (Aᵏ⁻¹)ᵢₜ · Aₜⱼ — sum over the possible last edges</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The matrix product sums over the second-to-last vertex $t$: walks of $k-1$ edges into $t$, times the last edge $tj$ — exactly the induction step.
</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Reachability from Powers)</span></p>

$(A + E)^k\_{ij} > 0$ iff there is an $ij$-walk with **at most** $k$ edges — the identity summand lets a walk "stay put". In particular

$$
(A + E)^n_{ij} \;>\; 0 \quad\iff\quad A^\ast_{ij} = 1.
$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Transitive Closure by Repeated Squaring)</span></p>

Square $A + E$ repeatedly, $k := \lceil \log n \rceil$ times, so that $2^k \ge n$ — and **replace all non-zero entries by ones after every multiplication**: the factors stay $0/1$, every product has entries at most $n$, and no giant numbers arise. That is $O(\log n)$ multiplications at $O(n^\omega)$ each:

$$
\text{transitive closure in } O(n^\omega \log n).
$$

(The same trick computes any power: $A^{2k} = (A^k)^2$ and $A^{2k+1} = (A^k)^2 \cdot A$, i.e. $O(\log k)$ multiplications by binary exponentiation.)

</div>

## Products over Other Algebras

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(The $(\oplus, \otimes)$-Product)</span></p>

For matrices over a set $X$ with operations $\oplus, \otimes$, define

$$
(A \cdot B)_{ij} \;:=\; \bigoplus_k \; A_{ik} \otimes B_{kj}.
$$

Standard matrix multiplication is the $(+, \cdot)$-product.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Bundle Matrices)</span></p>

Take matrices of **bundles** with the $(\cup, \cdot)$-product — union as addition, concatenation as multiplication. With $A\_{ij} = e\_{ij}$ for edges (and $\emptyset$ otherwise), the power $A^t\_{ij}$ is the bundle of all $ij$-walks with exactly $t$ edges; adding $\varepsilon\_i$ on the diagonal turns "exactly" into "at most". Applying a walk-algebra homomorphism $f$ entrywise turns $(\cup, \cdot)$-products into $(\oplus, \otimes)$-products of values — every numeric product below is a shadow of the bundle one.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Boolean Products)</span></p>

The $(\vee, \wedge)$-product on $0/1$ matrices gives $(A \vee E)^n = A^\ast$ — reachability in $O(\log n)$ Boolean products. A Boolean product reduces to one standard integer product (multiply the $0/1$ matrices over $\mathbb{Z}$, clamp non-zeros to $1$), so each costs $O(n^\omega)$ — recovering the $O(n^\omega \log n)$ bound.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Distance Products)</span></p>

The $(\min, +)$-product on the length matrix $L$ (with zero diagonal — the matrix $D^0$ of last lecture) gives $L^n = D$: APSP in $O(\log n)$ **distance products**. But $(\min, +)$ has no subtraction — it is not a ring — so Strassen-style tricks do not apply: one distance product costs $O(n^3)$ by definition, $O(n^3 / \log n)$ by Chan (2008), with faster algorithms only for small integer lengths.

</div>

## Divide and Conquer: Reachability in $O(n^\omega)$

The log factor in $O(n^\omega \log n)$ can be removed.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Reachability by Block Divide and Conquer)</span></p>

Split $V$ into halves $X$ and $Y$ and write the (reflexive) adjacency matrix in blocks:

$$
A \;=\; \begin{pmatrix} P & Q \\ R & S \end{pmatrix}
\qquad\leadsto\qquad
A^\ast \;=\; \begin{pmatrix} I & J \\ K & L \end{pmatrix},
$$

where, computing with $(\vee, \wedge)$-products,

$$
I = (P \vee Q\,S^\ast R)^\ast, \qquad
J = I\,Q\,S^\ast, \qquad
K = S^\ast R\,I, \qquad
L = S^\ast \vee\; S^\ast R\,I\,Q\,S^\ast.
$$

This uses **two recursive $\ast$-computations of size $n/2$** ($S^\ast$ and the outer star for $I$), plus $O(1)$ Boolean products and $O(1)$ cheap $O(n^2)$ matrix operations.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (why the formulas hold)</summary>

Read each block as a set of walks:

* $I$ (walks $X \to X$): such a walk alternates steps inside $X$ ($P$) with *excursions* into $Y$ — jump over ($Q$), wander inside $Y$ ($S^\ast$), come back ($R$). One building block is $P \vee Q S^\ast R$; arbitrary walks iterate it: $(P \vee Q S^\ast R)^\ast$.
* $J$ (walks $X \to Y$): first return to $X$ for the last time on the $X$-side ($I$), hop over ($Q$), finish inside $Y$ ($S^\ast$).
* $K$ (walks $Y \to X$): mirror image: $S^\ast R\, I$.
* $L$ (walks $Y \to Y$): either never leave $Y$ ($S^\ast$), or leave, travel via $X$, and come back: $S^\ast R\, I\, Q\, S^\ast$. ∎

</details>
</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 260" width="100%" style="max-width: 600px; height: auto;" role="img" aria-labelledby="gaf46-title">
  <title id="gaf46-title">The vertex set split into halves X and Y with the four blocks of the adjacency matrix</title>
  <defs>
    <marker id="gaf46-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#a86f00"/>
    </marker>
  </defs>
  <ellipse cx="130" cy="140" rx="72" ry="58" fill="none" stroke="#1565c0" stroke-width="1.8"/>
  <text x="130" y="146" text-anchor="middle" font-family="serif" font-size="15" font-style="italic" fill="#1565c0">X</text>
  <ellipse cx="330" cy="140" rx="72" ry="58" fill="none" stroke="#0f9b6c" stroke-width="1.8"/>
  <text x="330" y="146" text-anchor="middle" font-family="serif" font-size="15" font-style="italic" fill="#0f9b6c">Y</text>
  <path d="M 100,86 C 70,40 160,36 148,80" fill="none" stroke="#a86f00" stroke-width="1.6" marker-end="url(#gaf46-arr)"/>
  <text x="112" y="34" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#a86f00">P</text>
  <path d="M 310,86 C 282,40 372,36 358,80" fill="none" stroke="#a86f00" stroke-width="1.6" marker-end="url(#gaf46-arr)"/>
  <text x="324" y="34" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#a86f00">S</text>
  <path d="M 198,116 C 230,102 240,102 260,114" fill="none" stroke="#a86f00" stroke-width="1.6" marker-end="url(#gaf46-arr)"/>
  <text x="230" y="94" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#a86f00">Q</text>
  <path d="M 262,170 C 240,184 222,184 200,172" fill="none" stroke="#a86f00" stroke-width="1.6" marker-end="url(#gaf46-arr)"/>
  <text x="230" y="204" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#a86f00">R</text>
  <rect x="452" y="36" width="120" height="80" fill="#fff" stroke="#333" stroke-width="1.6"/>
  <line x1="512" y1="36" x2="512" y2="116" stroke="#333" stroke-width="1"/>
  <line x1="452" y1="76" x2="572" y2="76" stroke="#333" stroke-width="1"/>
  <text x="482" y="62" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#333">P</text>
  <text x="542" y="62" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#333">Q</text>
  <text x="482" y="102" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#333">R</text>
  <text x="542" y="102" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#333">S</text>
  <text x="440" y="62" text-anchor="end" font-family="serif" font-size="12" font-style="italic" fill="#333">A =</text>
  <line x1="512" y1="130" x2="512" y2="152" stroke="#666" stroke-width="1.4" marker-end="url(#gaf46-arr)"/>
  <text x="530" y="146" font-family="serif" font-size="12" fill="#666">✱</text>
  <rect x="452" y="160" width="120" height="80" fill="#fff" stroke="#333" stroke-width="1.6"/>
  <line x1="512" y1="160" x2="512" y2="240" stroke="#333" stroke-width="1"/>
  <line x1="452" y1="200" x2="572" y2="200" stroke="#333" stroke-width="1"/>
  <text x="482" y="186" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#333">I</text>
  <text x="542" y="186" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#333">J</text>
  <text x="482" y="226" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#333">K</text>
  <text x="542" y="226" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#333">L</text>
  <text x="440" y="206" text-anchor="end" font-family="serif" font-size="12" font-style="italic" fill="#333">A✱ =</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Splitting $V$ into halves turns $A$ into four blocks: $P, S$ inside the halves, $Q, R$ across. The closure $A^\ast$ has blocks expressible with just two half-size stars and a handful of products.
</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Cost of the Recursion)</span></p>

Let $\mu(n)$ be the cost of one $n \times n$ Boolean product. The running time obeys

$$
T(n) \;=\; 2\,T(n/2) + \Theta(\mu(n)),
$$

and whenever $\mu(n) = \Omega(n^2)$ this solves to $T(n) = \Theta(\mu(n))$. With $\mu(n) = O(n^\omega)$: **transitive closure in $O(n^\omega)$** — no log factor.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Down one level of the recursion tree the number of subproblems doubles, but the size halves — and since $\mu$ grows at least quadratically, $\mu(n/2) \le \mu(n)/4$. So the total work per level drops by a factor of at least $2$, and the geometric sum is dominated by the root: $T(n) \le \mu(n) \cdot (1 + \tfrac12 + \tfrac14 + \dots) = O(\mu(n))$. ∎

</details>
</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 240" width="100%" style="max-width: 540px; height: auto;" role="img" aria-labelledby="gaf47-title">
  <title id="gaf47-title">Recursion tree whose per-level work shrinks geometrically</title>
  <line x1="250" y1="52" x2="162" y2="112" stroke="#333" stroke-width="1.4"/>
  <line x1="270" y1="52" x2="358" y2="112" stroke="#333" stroke-width="1.4"/>
  <line x1="152" y1="132" x2="112" y2="182" stroke="#333" stroke-width="1.4"/>
  <line x1="168" y1="132" x2="208" y2="182" stroke="#333" stroke-width="1.4"/>
  <line x1="352" y1="132" x2="312" y2="182" stroke="#333" stroke-width="1.4"/>
  <line x1="368" y1="132" x2="408" y2="182" stroke="#333" stroke-width="1.4"/>
  <circle cx="260" cy="42" r="17" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <text x="260" y="47" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">n</text>
  <circle cx="160" cy="122" r="15" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <text x="160" y="127" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#1565c0">n/2</text>
  <circle cx="360" cy="122" r="15" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <text x="360" y="127" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#1565c0">n/2</text>
  <circle cx="110" cy="192" r="13" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <text x="110" y="196" text-anchor="middle" font-family="serif" font-size="10" font-style="italic" fill="#1565c0">n/4</text>
  <circle cx="210" cy="192" r="13" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <text x="210" y="196" text-anchor="middle" font-family="serif" font-size="10" font-style="italic" fill="#1565c0">n/4</text>
  <circle cx="310" cy="192" r="13" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <text x="310" y="196" text-anchor="middle" font-family="serif" font-size="10" font-style="italic" fill="#1565c0">n/4</text>
  <circle cx="410" cy="192" r="13" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <text x="410" y="196" text-anchor="middle" font-family="serif" font-size="10" font-style="italic" fill="#1565c0">n/4</text>
  <text x="455" y="47" font-family="serif" font-size="11" fill="#0f9b6c">work μ(n)</text>
  <text x="455" y="127" font-family="serif" font-size="11" fill="#0f9b6c">2·μ(n/2) ≤ μ(n)/2</text>
  <text x="455" y="196" font-family="serif" font-size="11" fill="#0f9b6c">4·μ(n/4) ≤ μ(n)/4</text>
  <text x="310" y="232" text-anchor="middle" font-family="serif" font-size="11" fill="#666">problems ×2, size /2, product cost /4 ⇒ level work halves ⇒ T(n) = Θ(μ(n))</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Each level doubles the subproblem count but quarters the product cost ($\mu \in \Omega(n^2)$), so the root's $\mu(n)$ dominates the whole tree.
</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(The Same Scheme for Distances)</span></p>

The block identities hold just as well for distances with $(\min, +)$-products (stars now meaning distance closures), so the divide and conquer also gives APSP in $\Theta$(one distance product)$\; = O(n^3 / \log n)$ time via Chan's product — the best general bound known, still stuck around the cube.

</div>

## Seidel's Algorithm

For **undirected graphs with unit lengths** (WLOG connected), APSP can ride on fast matrix multiplication.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Graph Squaring)</span></p>

$G^2 := (V, E^2)$ where $ij \in E^2$ iff $i \ne j$ and $G$ has an $ij$-walk with at most $2$ edges — neighbors and vertices with a common neighbor. Its adjacency matrix is one Boolean squaring away: $A(G^2)$ is $(A \vee E)^2$ with the diagonal cleared, computable in $O(n^\omega)$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Distances Halve)</span></p>

Writing $d$ for distances in $G$ and $d'$ for distances in $G^2$ (from a fixed source $u$):

$$
d'(v) \;=\; \lceil d(v) / 2 \rceil
$$

— pairing up the edges of a shortest $G$-walk gives a $G^2$-walk half as long (rounded up), and conversely every $G^2$-edge expands to at most two $G$-edges. Consequently $d(v) \in \lbrace 2d'(v) - 1,\; 2d'(v) \rbrace$: after solving $G^2$, only the **parity** of $d(v)$ is missing. Also, the diameter halves — after $\lceil \log n \rceil$ squarings the graph is complete and $D$ is trivial.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 240" width="100%" style="max-width: 540px; height: auto;" role="img" aria-labelledby="gaf49-title">
  <title id="gaf49-title">Seidel's recursion: square the graph, solve, fix parities</title>
  <defs>
    <marker id="gaf49b-arrg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#0f9b6c"/>
    </marker>
    <marker id="gaf49b-arrb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#1565c0"/>
    </marker>
    <marker id="gaf49b-arrr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#b91c1c"/>
    </marker>
  </defs>
  <rect x="70" y="40" width="130" height="52" rx="9" fill="#fff" stroke="#333" stroke-width="1.6"/>
  <text x="135" y="71" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#333">G, A</text>
  <rect x="420" y="40" width="130" height="52" rx="9" fill="#fff" stroke="#333" stroke-width="1.6"/>
  <text x="485" y="71" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#333">G², A′</text>
  <rect x="70" y="160" width="130" height="52" rx="9" fill="#fff" stroke="#333" stroke-width="1.6"/>
  <text x="135" y="191" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#333">D</text>
  <rect x="420" y="160" width="130" height="52" rx="9" fill="#fff" stroke="#333" stroke-width="1.6"/>
  <text x="485" y="191" text-anchor="middle" font-family="serif" font-size="14" font-style="italic" fill="#333">D′</text>
  <line x1="204" y1="66" x2="414" y2="66" stroke="#0f9b6c" stroke-width="1.8" marker-end="url(#gaf49b-arrg)"/>
  <text x="310" y="54" text-anchor="middle" font-family="serif" font-size="11" fill="#0f9b6c">squaring — one Boolean product</text>
  <line x1="485" y1="96" x2="485" y2="154" stroke="#1565c0" stroke-width="1.8" marker-end="url(#gaf49b-arrb)"/>
  <text x="497" y="129" font-family="serif" font-size="11" fill="#1565c0">recurse</text>
  <line x1="414" y1="186" x2="204" y2="186" stroke="#b91c1c" stroke-width="1.8" marker-end="url(#gaf49b-arrr)"/>
  <text x="310" y="174" text-anchor="middle" font-family="serif" font-size="11" fill="#b91c1c">parity fix — one integer product</text>
  <text x="310" y="232" text-anchor="middle" font-family="serif" font-size="11" fill="#666">O(log n) levels: after ⌈log n⌉ squarings the graph is complete and D is trivial</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
One level of Seidel's algorithm: square the graph (Boolean product), solve APSP there recursively, then recover the true distances with a single integer product.
</figcaption>
</figure>

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Parity from the Neighbors)</span></p>

Fix the source $u$ and a vertex $v \neq u$. Then:

* if $d(v)$ is **even**, every neighbor $w$ of $v$ has $d'(w) \ge d'(v)$;
* if $d(v)$ is **odd**, every neighbor has $d'(w) \le d'(v)$, and at least one has $d'(w) = d'(v) - 1$.

Hence:

$$
d(v) \text{ is even} \quad\iff\quad \sum_{w \,\sim\, v} d'(w) \;\ge\; d'(v) \cdot \deg(v).
$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

In an undirected unit-length graph, distances of adjacent vertices differ by at most one: $d(w) \in \lbrace d(v)-1,\, d(v),\, d(v)+1 \rbrace$ for $w \sim v$. Now take ceilings:

* $d(v) = 2a$ even: neighbors have $d \in \lbrace 2a-1, 2a, 2a+1 \rbrace$, so $d' = \lceil d/2 \rceil \in \lbrace a, a, a+1 \rbrace$ — all $\ge a = d'(v)$.
* $d(v) = 2a - 1$ odd: neighbors have $d \in \lbrace 2a-2, 2a-1, 2a \rbrace$, so $d' \in \lbrace a-1, a, a \rbrace$ — all $\le a = d'(v)$; and the predecessor of $v$ on a shortest $uv$-path has $d = 2a-2$, i.e. $d' = a - 1$.

In the even case the neighbor sum is $\ge d'(v)\deg(v)$; in the odd case it is $\le d'(v)\deg(v) - 1$. The two cases are separated exactly by the stated inequality. ∎

</details>
</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 270" width="100%" style="max-width: 600px; height: auto;" role="img" aria-labelledby="gaf48-title">
  <title id="gaf48-title">Neighbors of v sit at distance d minus one, d, or d plus one from u</title>
  <defs>
    <marker id="gaf48-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
    </marker>
  </defs>
  <path d="M 88,122 C 180,74 330,74 428,118" fill="none" stroke="#333" stroke-width="1.6" stroke-dasharray="7 5" marker-end="url(#gaf48-arr)"/>
  <text x="258" y="70" text-anchor="middle" font-family="serif" font-size="11" fill="#333">shortest walk, d(v) edges</text>
  <line x1="463" y1="122" x2="536" y2="74" stroke="#333" stroke-width="1.5"/>
  <line x1="466" y1="130" x2="534" y2="130" stroke="#333" stroke-width="1.5"/>
  <line x1="463" y1="138" x2="536" y2="186" stroke="#333" stroke-width="1.5"/>
  <circle cx="70" cy="130" r="15" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="70" y="135" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#0f9b6c">u</text>
  <circle cx="450" cy="130" r="14" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <text x="450" y="135" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#1565c0">v</text>
  <circle cx="548" cy="68" r="11" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <circle cx="548" cy="130" r="11" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <circle cx="548" cy="192" r="11" fill="#fff" stroke="#1565c0" stroke-width="1.5"/>
  <text x="572" y="60" font-family="serif" font-size="11" fill="#666">d(v) − 1</text>
  <text x="572" y="134" font-family="serif" font-size="11" fill="#666">d(v)</text>
  <text x="572" y="204" font-family="serif" font-size="11" fill="#666">d(v) + 1</text>
  <text x="500" y="34" text-anchor="middle" font-family="serif" font-size="11" fill="#1565c0">neighbors of v</text>
  <text x="320" y="234" text-anchor="middle" font-family="serif" font-size="11" fill="#0f9b6c">d(v) even ⇒ every neighbor has d′ ≥ d′(v)</text>
  <text x="320" y="258" text-anchor="middle" font-family="serif" font-size="11" fill="#c2185b">d(v) odd ⇒ all neighbors have d′ ≤ d′(v), the predecessor even d′(v) − 1</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Unit lengths force neighbors of $v$ into distances $d(v)-1$, $d(v)$, $d(v)+1$; taking ceilings $\lceil d/2 \rceil$ pushes their $d'$-values strictly to opposite sides depending on the parity of $d(v)$.
</figcaption>
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(All Neighbor Sums in One Product)</span></p>

The sums the lemma needs — for *all* pairs $(u, v)$ at once — sit in one standard integer product:

$$
(D' \cdot A)_{uv} \;=\; \sum_k D'_{uk} \, A_{kv} \;=\; \sum_{k \,\sim\, v} d'(k),
$$

to be compared entrywise with $D'\_{uv} \cdot \deg(v)$. So the parity fix costs one matrix multiplication plus $O(n^2)$ postprocessing.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Seidel's Algorithm)</span></p>

APSP in undirected unit-length graphs is solvable in $O(n^\omega \log n)$ time: the recursion has $O(\log n)$ levels — the diameter halves each time — and every level costs two matrix multiplications (one Boolean squaring, one integer product for the parities).

</div>

# Lecture 9: Minimum Spanning Trees

After two chapters of shortest paths, a new classic begins: connect everything as cheaply as possible. The chapter opens with the structural theory — light edges, the cut lemma, uniqueness — then a meta-algorithm that colors edges **blue** (in) and **red** (out), of which all three classical algorithms (Jarník, Borůvka, Kruskal) are special cases.

## The Problem

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Minimum Spanning Tree)</span></p>

Let $G$ be a connected undirected graph with **weights** $w \colon E \to \mathbb{R}$, WLOG **injective** (how to justify the WLOG — see the tie-breaking remark below). A **minimum spanning tree** (MST) is a spanning tree $T$ minimizing $w(T) := \sum\_{e \in T} w(e)$.

Notation: $T[x,y]$ is the unique path in $T$ between $x$ and $y$, and $T[e] := T[x,y]$ for an edge $e = xy$.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 240" width="100%" style="max-width: 560px; height: auto;" role="img" aria-labelledby="gaf50-title">
  <title id="gaf50-title">A weighted graph with a spanning tree highlighted</title>
  <line x1="100" y1="80" x2="200" y2="50" stroke="#1565c0" stroke-width="2.6"/>
  <line x1="200" y1="50" x2="320" y2="60" stroke="#1565c0" stroke-width="2.6"/>
  <line x1="320" y1="60" x2="450" y2="70" stroke="#1565c0" stroke-width="2.6"/>
  <line x1="450" y1="70" x2="540" y2="120" stroke="#1565c0" stroke-width="2.6"/>
  <line x1="320" y1="60" x2="300" y2="190" stroke="#1565c0" stroke-width="2.6"/>
  <line x1="300" y1="190" x2="150" y2="170" stroke="#1565c0" stroke-width="2.6"/>
  <line x1="300" y1="190" x2="430" y2="170" stroke="#1565c0" stroke-width="2.6"/>
  <line x1="100" y1="80" x2="150" y2="170" stroke="#666" stroke-width="1.3" stroke-dasharray="5 4"/>
  <line x1="200" y1="50" x2="300" y2="190" stroke="#666" stroke-width="1.3" stroke-dasharray="5 4"/>
  <line x1="450" y1="70" x2="430" y2="170" stroke="#666" stroke-width="1.3" stroke-dasharray="5 4"/>
  <line x1="540" y1="120" x2="430" y2="170" stroke="#666" stroke-width="1.3" stroke-dasharray="5 4"/>
  <circle cx="100" cy="80" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <circle cx="200" cy="50" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <circle cx="320" cy="60" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <circle cx="450" cy="70" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <circle cx="540" cy="120" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <circle cx="430" cy="170" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <circle cx="300" cy="190" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <circle cx="150" cy="170" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <text x="310" y="228" text-anchor="middle" font-family="serif" font-size="11" fill="#666">a spanning tree (blue) — the MST minimizes the total weight of the picked edges</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
A weighted graph and one of its spanning trees (blue); dashed edges are unused. Among all spanning trees, the MST has minimum total weight.
</figcaption>
</figure>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(T-Light Edge)</span></p>

An edge $e \in E$ is **$T$-light** if some edge of the tree path connecting its endpoints is heavier:

$$
\exists f \in T[e] \colon\quad w(f) \;>\; w(e).
$$

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 250" width="100%" style="max-width: 560px; height: auto;" role="img" aria-labelledby="gaf51-title">
  <title id="gaf51-title">A T-light edge: a chord lighter than some edge on its tree path</title>
  <line x1="90" y1="160" x2="180" y2="90" stroke="#1565c0" stroke-width="2.6"/>
  <line x1="180" y1="90" x2="300" y2="70" stroke="#b91c1c" stroke-width="3"/>
  <line x1="300" y1="70" x2="420" y2="90" stroke="#1565c0" stroke-width="2.6"/>
  <line x1="420" y1="90" x2="530" y2="160" stroke="#1565c0" stroke-width="2.6"/>
  <path d="M 100,172 C 220,232 400,232 520,172" fill="none" stroke="#0f9b6c" stroke-width="2.4"/>
  <text x="240" y="64" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#b91c1c">f — heavier</text>
  <text x="310" y="240" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#0f9b6c">e — w(e) &lt; w(f)</text>
  <text x="300" y="34" text-anchor="middle" font-family="serif" font-size="11" fill="#1565c0">tree path T[e] = T[x,y]</text>
  <circle cx="90" cy="160" r="12" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <text x="90" y="165" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">x</text>
  <circle cx="180" cy="90" r="7" fill="#fff" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="300" cy="70" r="7" fill="#fff" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="420" cy="90" r="7" fill="#fff" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="530" cy="160" r="12" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <text x="530" y="165" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">y</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The edge $e = xy$ is $T$-light: on the tree path $T[e]$ sits a strictly heavier edge $f$.
</figcaption>
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Exchange Step)</span></p>

If some edge $e$ is $T$-light, then $T$ is **not** minimum: for the heavier $f \in T[e]$, the swap $T' := T - f + e$ is again a spanning tree ($T[e]$ closes a cycle with $e$, and removing any cycle edge reconnects it), and $w(T') < w(T)$.

</div>

## The Cut Lemma

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Cut Lemma, a.k.a. Blue Lemma)</span></p>

Let $C = E(X, Y)$ be an **elementary cut** — the set of edges with one endpoint in $X$ and the other in $Y$, for a bipartition $V = X \mathbin{\dot\cup} Y$ — let $e \in C$ be the **lightest** edge of the cut, and let $T$ be an arbitrary MST. Then $e \in T$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Assume the contrary, $e \notin T$. The tree path $T[e]$ connects the two endpoints of $e$, which lie on opposite sides of the cut — so the path crosses the cut at some edge $f \in T[e] \cap C$. Then $T' := T - f + e$ is another spanning tree, and $w(f) > w(e)$ because $e$ is the lightest edge of the cut and weights are injective — hence $w(T') < w(T)$, contradicting minimality. ∎

</details>
</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 260" width="100%" style="max-width: 600px; height: auto;" role="img" aria-labelledby="gaf52-title">
  <title id="gaf52-title">The tree path of e must cross the cut at some other edge f</title>
  <ellipse cx="160" cy="135" rx="110" ry="88" fill="none" stroke="#1565c0" stroke-width="1.8"/>
  <text x="80" y="70" font-family="serif" font-size="14" font-style="italic" fill="#1565c0">X</text>
  <ellipse cx="480" cy="135" rx="110" ry="88" fill="none" stroke="#0f9b6c" stroke-width="1.8"/>
  <text x="548" y="70" font-family="serif" font-size="14" font-style="italic" fill="#0f9b6c">Y</text>
  <line x1="252" y1="90" x2="390" y2="90" stroke="#b91c1c" stroke-width="2.6"/>
  <text x="321" y="78" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#b91c1c">f ∈ T[e] ∩ C</text>
  <line x1="256" y1="150" x2="386" y2="150" stroke="#0f9b6c" stroke-width="2.6"/>
  <text x="321" y="140" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#0f9b6c">e — lightest in C</text>
  <line x1="250" y1="200" x2="392" y2="200" stroke="#666" stroke-width="1.3" stroke-dasharray="5 4"/>
  <path d="M 244,150 C 200,140 210,104 240,92" fill="none" stroke="#1565c0" stroke-width="1.6" stroke-dasharray="6 4"/>
  <path d="M 398,92 C 430,104 438,138 398,148" fill="none" stroke="#1565c0" stroke-width="1.6" stroke-dasharray="6 4"/>
  <circle cx="244" cy="150" r="9" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <circle cx="398" cy="150" r="9" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="1.8"/>
  <circle cx="246" cy="90" r="7" fill="#fff" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="396" cy="90" r="7" fill="#fff" stroke="#0f9b6c" stroke-width="1.6"/>
  <text x="320" y="246" text-anchor="middle" font-family="serif" font-size="11" fill="#666">the dashed tree path T[e] leaves X — it must use some cut edge f; swap f for e</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The cut lemma: the endpoints of $e$ lie on opposite shores, so $T[e]$ (dashed) crosses the cut at some $f$ — and $T - f + e$ is a strictly lighter spanning tree.
</figcaption>
</figure>

## Jarník's Algorithm

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Jarník 1930, often called Prim's or "Dijkstra-style")</span></p>

1. $T \leftarrow \lbrace v\_0 \rbrace$ for an arbitrary vertex $v\_0$.
2. **While** $\lvert T \rvert < n$:
3. &nbsp;&nbsp;&nbsp;select the lightest edge $uv \in E$ with $u \in T$, $v \notin T$;
4. &nbsp;&nbsp;&nbsp;$T \leftarrow T \cup \lbrace uv \rbrace$.

If $G$ is connected, some edge always leaves the current tree, so the algorithm finds a spanning tree.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 250" width="100%" style="max-width: 540px; height: auto;" role="img" aria-labelledby="gaf53-title">
  <title id="gaf53-title">Jarnik grows one tree and always adds the lightest leaving edge</title>
  <path d="M 70,130 C 60,50 200,16 340,24 C 480,32 570,80 566,140 C 562,204 460,236 310,232 C 160,228 80,200 70,130 Z" fill="none" stroke="#333" stroke-width="1.5"/>
  <line x1="170" y1="130" x2="120" y2="92" stroke="#1565c0" stroke-width="2.4"/>
  <line x1="170" y1="130" x2="222" y2="82" stroke="#1565c0" stroke-width="2.4"/>
  <line x1="170" y1="130" x2="142" y2="182" stroke="#1565c0" stroke-width="2.4"/>
  <line x1="170" y1="130" x2="232" y2="164" stroke="#1565c0" stroke-width="2.4"/>
  <line x1="230" y1="78" x2="330" y2="66" stroke="#b91c1c" stroke-width="2.4"/>
  <text x="286" y="52" text-anchor="middle" font-family="serif" font-size="11" fill="#b91c1c">lightest edge leaving T</text>
  <circle cx="170" cy="130" r="12" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="170" y="135" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#0f9b6c">v₀</text>
  <circle cx="120" cy="92" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="222" cy="82" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="142" cy="182" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="232" cy="164" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="338" cy="65" r="6" fill="#fff" stroke="#333" stroke-width="1.4"/>
  <circle cx="420" cy="110" r="5" fill="#666"/>
  <circle cx="470" cy="170" r="5" fill="#666"/>
  <circle cx="360" cy="190" r="5" fill="#666"/>
  <circle cx="490" cy="80" r="5" fill="#666"/>
  <text x="310" y="246" text-anchor="middle" font-family="serif" font-size="11" fill="#666">one growing blue tree; each step takes the lightest edge across its boundary</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Jarník's algorithm grows a single tree from $v_0$, always along the lightest edge crossing the cut between the tree and the rest.
</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Jarník Finds the MST)</span></p>

Jarník's algorithm outputs a minimum spanning tree.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Every selected edge is the lightest edge of the elementary cut between the current tree and the remaining vertices — by the cut lemma it belongs to **every** MST. So the output $T$ satisfies $T \subseteq$ every MST; and since all spanning trees have exactly $n - 1$ edges, $T$ *equals* every MST. ∎

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Uniqueness)</span></p>

With injective weights the minimum spanning tree is **unique** — the proof above shows every MST equals Jarník's output.

</div>

## Characterizing the MST

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Light-Edge Characterization)</span></p>

$T$ is the minimum spanning tree $\iff$ there are no $T$-light edges.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

"$\Rightarrow$": a $T$-light edge would allow the exchange step — $T$ would not be minimum.

"$\Leftarrow$": suppose $T$ has no $T$-light edges; we claim Jarník's algorithm (which outputs the MST) outputs exactly $T$. If not, stop it at the **first** moment it selects an edge $e = uv \notin T$; up to that moment the current tree is contained in $T$. The path $T[e]$ starts at $u$ inside the current tree and ends at $v$ outside, so some edge $f \in T[e]$ also leaves the current tree. But then $f$ was a candidate in the same step, and the algorithm chose $e$ as the lightest: $w(f) > w(e)$. That makes $e$ a $T$-light edge — contradiction. ∎

</details>
</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 250" width="100%" style="max-width: 600px; height: auto;" role="img" aria-labelledby="gaf54-title">
  <title id="gaf54-title">If Jarnik ever left T, the selected edge would be T-light</title>
  <path d="M 80,130 C 74,66 150,36 230,40 C 306,44 344,86 340,132 C 336,180 270,212 190,208 C 116,204 86,180 80,130 Z" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="1.8"/>
  <text x="150" y="70" font-family="serif" font-size="11" fill="#0f9b6c">current tree ⊆ T</text>
  <line x1="230" y1="120" x2="470" y2="96" stroke="#b91c1c" stroke-width="2.6"/>
  <text x="352" y="86" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#b91c1c">e — selected, e ∉ T</text>
  <line x1="240" y1="170" x2="462" y2="182" stroke="#1565c0" stroke-width="2.6"/>
  <text x="352" y="206" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">f ∈ T[e], also leaves the tree</text>
  <path d="M 232,128 C 220,150 226,162 236,168" fill="none" stroke="#1565c0" stroke-width="1.5" stroke-dasharray="5 4"/>
  <path d="M 468,178 C 486,160 490,120 478,102" fill="none" stroke="#1565c0" stroke-width="1.5" stroke-dasharray="5 4"/>
  <circle cx="230" cy="120" r="9" fill="#fff" stroke="#0f9b6c" stroke-width="1.8"/>
  <text x="216" y="110" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#0f9b6c">u</text>
  <circle cx="478" cy="96" r="9" fill="#fff" stroke="#b91c1c" stroke-width="1.8"/>
  <text x="496" y="90" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#b91c1c">v</text>
  <circle cx="240" cy="170" r="7" fill="#fff" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="466" cy="182" r="7" fill="#fff" stroke="#1565c0" stroke-width="1.6"/>
  <text x="320" y="240" text-anchor="middle" font-family="serif" font-size="11" fill="#666">f competed with e and lost: w(f) &gt; w(e) ⇒ e is T-light ↯</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The moment Jarník would first deviate from $T$: the tree path $T[e]$ (dashed) also crosses the boundary, at some $f$ that lost the "lightest" contest to $e$ — so $e$ would be $T$-light.
</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Only the Order of Weights Matters)</span></p>

Everything above — the cut lemma, Jarník's algorithm, the characterization — uses weights **only through comparisons**. The actual values are irrelevant; all we need is an *edge-comparison oracle* answering "$w(e) < w(f)$?" in constant time.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Non-Injective Weights)</span></p>

With ties, several MSTs may exist, and "$\le$" on edges is not a linear order. The cure: break ties arbitrarily but consistently — e.g. compare pairs $(w(e), \mathrm{id}(e))$ lexicographically. This restores injectivity, justifying the WLOG at the start.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 560 250" width="100%" style="max-width: 380px; height: auto;" role="img" aria-labelledby="gaf55-title">
  <title id="gaf55-title">A five-cycle with all weights one has five different minimum spanning trees</title>
  <line x1="280" y1="40" x2="366" y2="103" stroke="#333" stroke-width="2"/>
  <line x1="366" y1="103" x2="333" y2="205" stroke="#333" stroke-width="2"/>
  <line x1="333" y1="205" x2="227" y2="205" stroke="#333" stroke-width="2"/>
  <line x1="227" y1="205" x2="194" y2="103" stroke="#333" stroke-width="2"/>
  <line x1="194" y1="103" x2="280" y2="40" stroke="#333" stroke-width="2"/>
  <text x="334" y="62" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#a86f00">1</text>
  <text x="366" y="162" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#a86f00">1</text>
  <text x="280" y="224" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#a86f00">1</text>
  <text x="196" y="162" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#a86f00">1</text>
  <text x="226" y="62" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#a86f00">1</text>
  <circle cx="280" cy="40" r="8" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <circle cx="366" cy="103" r="8" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <circle cx="333" cy="205" r="8" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <circle cx="227" cy="205" r="8" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <circle cx="194" cy="103" r="8" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The cycle $C_5$ with unit weights: dropping any one edge gives an MST — five of them. Lexicographic tie-breaking by $(w(e), \mathrm{id}(e))$ singles one out.
</figcaption>
</figure>

## The Red–Blue Meta-Algorithm

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Red–Blue Meta-Algorithm)</span></p>

1. All edges start uncolored.
2. Repeat, in any order, as long as some rule applies:
   * **Blue rule:** find an edge that is lightest in some elementary cut and not yet blue — color it **blue**.
   * **Red rule:** find an edge that is heaviest on some cycle and not yet red — color it **red**.

</div>

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Blue Lemma)</span></p>

Every blue edge belongs to the MST — this is exactly the cut lemma.

</div>

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Red Lemma)</span></p>

If $e$ is the heaviest edge of some cycle $C$, then $e$ does **not** belong to the MST.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By contradiction: let $e \in T$ where $T$ is the MST. Removing $e$ splits $T$ into two components $T\_1, T\_2$. The rest of the cycle, $C - e$, connects the endpoints of $e$, so it contains some edge $f$ with one endpoint in $T\_1$ and the other in $T\_2$. Then $T' := T - e + f$ is again a spanning tree, and $w(f) < w(e)$ since $e$ is the heaviest edge of $C$ — so $w(T') < w(T)$, a contradiction. ∎

</details>
</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 260" width="100%" style="max-width: 580px; height: auto;" role="img" aria-labelledby="gaf56-title">
  <title id="gaf56-title">The heaviest edge of a cycle can be swapped for a lighter cycle edge re-crossing the split</title>
  <line x1="200" y1="60" x2="330" y2="40" stroke="#333" stroke-width="2"/>
  <line x1="330" y1="40" x2="440" y2="80" stroke="#b91c1c" stroke-width="3"/>
  <line x1="440" y1="80" x2="430" y2="170" stroke="#333" stroke-width="2"/>
  <line x1="430" y1="170" x2="300" y2="210" stroke="#0f9b6c" stroke-width="3"/>
  <line x1="300" y1="210" x2="180" y2="160" stroke="#333" stroke-width="2"/>
  <line x1="180" y1="160" x2="200" y2="60" stroke="#333" stroke-width="2"/>
  <text x="404" y="46" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#b91c1c">e — heaviest on C</text>
  <text x="330" y="232" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#0f9b6c">f — re-crosses, w(f) &lt; w(e)</text>
  <ellipse cx="222" cy="128" rx="105" ry="105" fill="none" stroke="#0f9b6c" stroke-width="1.5" stroke-dasharray="7 5"/>
  <text x="130" y="52" font-family="serif" font-size="12" font-style="italic" fill="#0f9b6c">T₁</text>
  <ellipse cx="452" cy="128" rx="70" ry="80" fill="none" stroke="#0f9b6c" stroke-width="1.5" stroke-dasharray="7 5"/>
  <text x="520" y="62" font-family="serif" font-size="12" font-style="italic" fill="#0f9b6c">T₂</text>
  <circle cx="200" cy="60" r="7" fill="#fff" stroke="#333" stroke-width="1.6"/>
  <circle cx="330" cy="40" r="7" fill="#fff" stroke="#333" stroke-width="1.6"/>
  <circle cx="440" cy="80" r="7" fill="#fff" stroke="#333" stroke-width="1.6"/>
  <circle cx="430" cy="170" r="7" fill="#fff" stroke="#333" stroke-width="1.6"/>
  <circle cx="300" cy="210" r="7" fill="#fff" stroke="#333" stroke-width="1.6"/>
  <circle cx="180" cy="160" r="7" fill="#fff" stroke="#333" stroke-width="1.6"/>
  <text x="320" y="250" text-anchor="middle" font-family="serif" font-size="11" fill="#666">T − e falls apart into T₁, T₂; the cycle C − e re-connects them through some cheaper f</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The red lemma's exchange: deleting the heaviest cycle edge $e$ splits the MST into $T_1, T_2$ (dashed), and the remainder of the cycle re-crosses the split at a strictly lighter $f$.
</figcaption>
</figure>

Blue edges lie in the MST, red edges outside it — so **no edge is ever re-colored**, and the algorithm stops after at most $m$ coloring steps.

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(Rainbow Lemma)</span></p>

If some edge is still uncolored, one of the two rules applies. Consequently, when the algorithm stops, **all** edges are colored — and the blue ones are exactly the MST.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $e = uv$ be uncolored and let $B$ be the set of vertices reachable from $u$ via blue edges.

* **If $v \in B$:** the blue $uv$-path together with $e$ forms a cycle. Its heaviest edge cannot be blue — blue edges lie in the MST (blue lemma), while the heaviest edge of a cycle does not (red lemma). Since all other edges of the cycle are blue, the heaviest is $e$: the **red rule** colors $e$.
* **If $v \notin B$:** consider the cut $E(B, V \setminus B)$. It is non-empty ($e$ crosses it) and contains no blue edge — a blue edge across it would extend $B$. Its lightest edge lies in the MST (cut lemma), so it is not red either: the **blue rule** colors an uncolored edge.

Either way the algorithm continues. At termination every edge is colored; blue $\subseteq$ MST and red $\cap$ MST $= \emptyset$ then force blue $=$ MST. ∎

</details>
</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 260" width="100%" style="max-width: 580px; height: auto;" role="img" aria-labelledby="gaf57-title">
  <title id="gaf57-title">The blue-reachable set B around u and the uncolored edge e to v</title>
  <path d="M 90,130 C 84,60 170,28 260,32 C 350,36 400,80 396,134 C 392,192 320,224 230,220 C 140,216 96,190 90,130 Z" fill="none" stroke="#1565c0" stroke-width="1.8"/>
  <text x="120" y="56" font-family="serif" font-size="14" font-style="italic" fill="#1565c0">B</text>
  <line x1="240" y1="126" x2="180" y2="86" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="240" y1="126" x2="310" y2="80" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="240" y1="126" x2="176" y2="170" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="310" y1="80" x2="344" y2="140" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="176" y1="170" x2="252" y2="188" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="128" y1="70" x2="96" y2="42" stroke="#0f9b6c" stroke-width="1.4" stroke-dasharray="4 4"/>
  <line x1="240" y1="34" x2="238" y2="8" stroke="#0f9b6c" stroke-width="1.4" stroke-dasharray="4 4"/>
  <line x1="356" y1="62" x2="384" y2="38" stroke="#0f9b6c" stroke-width="1.4" stroke-dasharray="4 4"/>
  <line x1="150" y1="210" x2="122" y2="238" stroke="#0f9b6c" stroke-width="1.4" stroke-dasharray="4 4"/>
  <line x1="300" y1="216" x2="316" y2="244" stroke="#0f9b6c" stroke-width="1.4" stroke-dasharray="4 4"/>
  <line x1="252" y1="126" x2="540" y2="126" stroke="#333" stroke-width="2.4"/>
  <text x="420" y="114" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#333">e — uncolored</text>
  <circle cx="240" cy="126" r="10" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="240" y="131" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#0f9b6c">u</text>
  <circle cx="180" cy="86" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="310" cy="80" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="176" cy="170" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="344" cy="140" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="252" cy="188" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="552" cy="126" r="10" fill="#fff" stroke="#333" stroke-width="1.8"/>
  <text x="552" y="131" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#333">v</text>
  <text x="320" y="256" text-anchor="middle" font-family="serif" font-size="11" fill="#666">v ∈ B ⇒ red rule on e; v ∉ B ⇒ blue rule on the cut E(B, V∖B) — dashes are uncolored edges</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The rainbow lemma: $B$ is everything blue-reachable from $u$ (blue tree inside, dashed uncolored edges sticking out). Depending on whether $v$ lands inside or outside $B$, the red or the blue rule fires.
</figcaption>
</figure>

## The Classical Algorithms as Instances

**① Jarník** grows one blue tree: every selection is the blue rule on the cut around the current tree. Basic implementation $O(nm)$; with a heap over the outside vertices (as in Dijkstra) $O(m \log n)$.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(② Borůvka 1926)</span></p>

Maintain a blue forest, initially $n$ isolated vertices. In every step, **every tree** of the forest selects the lightest edge incident to it, and all selected edges are colored blue at once — each selection is the blue rule on the cut around that tree (injective weights guarantee the simultaneous additions close no cycle).

The number of trees at least **halves** in every step, so there are at most $\log n$ steps; a step costs $O(m)$ — total $O(m \log n)$.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 250" width="100%" style="max-width: 580px; height: auto;" role="img" aria-labelledby="gaf58-title">
  <title id="gaf58-title">Boruvka: every tree of the blue forest selects its lightest incident edge</title>
  <defs>
    <marker id="gaf58-arrr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#b91c1c"/>
    </marker>
  </defs>
  <path d="M 60,125 C 54,52 170,18 320,22 C 470,26 586,64 582,130 C 578,196 460,230 310,226 C 160,222 66,194 60,125 Z" fill="none" stroke="#333" stroke-width="1.5"/>
  <line x1="130" y1="80" x2="180" y2="60" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="130" y1="80" x2="150" y2="120" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="380" y1="70" x2="440" y2="60" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="180" y1="170" x2="230" y2="190" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="440" y1="160" x2="490" y2="180" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="440" y1="160" x2="480" y2="120" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="190" y1="63" x2="368" y2="69" stroke="#b91c1c" stroke-width="1.8" marker-end="url(#gaf58-arrr)"/>
  <line x1="160" y1="124" x2="176" y2="160" stroke="#b91c1c" stroke-width="1.8" marker-end="url(#gaf58-arrr)"/>
  <line x1="240" y1="188" x2="428" y2="164" stroke="#b91c1c" stroke-width="1.8" marker-end="url(#gaf58-arrr)"/>
  <line x1="448" y1="64" x2="476" y2="110" stroke="#b91c1c" stroke-width="1.8" marker-end="url(#gaf58-arrr)"/>
  <circle cx="130" cy="80" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="180" cy="60" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="150" cy="120" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="380" cy="70" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="440" cy="60" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="180" cy="170" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="230" cy="190" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="440" cy="160" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="490" cy="180" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="480" cy="120" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="320" y="240" text-anchor="middle" font-family="serif" font-size="11" fill="#666">every blue tree shoots its lightest incident edge (red arrows) — the forest at least halves</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
One Borůvka round: each tree of the blue forest picks the lightest edge leaving it; all picks are added simultaneously, merging trees in groups of two or more.
</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(③ Kruskal)</span></p>

Sort the edges by weight, $w(e\_1) < w(e\_2) < \dots < w(e\_m)$, and start with $T \leftarrow \emptyset$. For $i = 1, \dots, m$: **if** $T + e\_i$ is acyclic — i.e. the endpoints of $e\_i$ lie in different components of $T$ — then $T \leftarrow T + e\_i$.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 240" width="100%" style="max-width: 560px; height: auto;" role="img" aria-labelledby="gaf59-title">
  <title id="gaf59-title">Kruskal tests the next-lightest edge against the current blue forest</title>
  <path d="M 60,120 C 56,54 170,22 320,26 C 470,30 584,66 580,126 C 576,190 460,222 310,218 C 160,214 64,184 60,120 Z" fill="none" stroke="#333" stroke-width="1.5"/>
  <line x1="140" y1="90" x2="200" y2="70" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="140" y1="90" x2="170" y2="130" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="400" y1="80" x2="460" y2="100" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="270" y1="170" x2="330" y2="185" stroke="#1565c0" stroke-width="2.2"/>
  <polyline points="176,126 220,140 210,155 254,168" fill="none" stroke="#b91c1c" stroke-width="2.4"/>
  <text x="196" y="168" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#b91c1c">eᵢ</text>
  <circle cx="140" cy="90" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="200" cy="70" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="170" cy="130" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="400" cy="80" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="460" cy="100" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="270" cy="170" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="330" cy="185" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="320" y="232" text-anchor="middle" font-family="serif" font-size="11" fill="#666">edges arrive in increasing weight; eᵢ is accepted iff its endpoints lie in different blue components</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Kruskal's scan: the current $T$ is a blue forest; the next-lightest edge $e_i$ is kept exactly when it joins two different components.
</figcaption>
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Kruskal Is Red–Blue)</span></p>

* **Adding an edge:** when $e$ joins two components, it is the *first* — hence lightest — edge of the cut between the component $T\_1$ of one endpoint and the rest ever considered: all other cut edges come later in the sorted order, i.e. are heavier. That is the blue rule.
* **Dropping an edge:** when $e$ closes a cycle inside a component, $e$ arrives *after* every edge of the tree path between its endpoints — so $e$ is the heaviest edge of that cycle. That is the red rule.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 240" width="100%" style="max-width: 580px; height: auto;" role="img" aria-labelledby="gaf60-title">
  <title id="gaf60-title">An accepted Kruskal edge is the lightest edge of the cut around its component</title>
  <ellipse cx="170" cy="120" rx="105" ry="75" fill="none" stroke="#0f9b6c" stroke-width="1.8"/>
  <text x="100" y="62" font-family="serif" font-size="13" font-style="italic" fill="#0f9b6c">T₁</text>
  <ellipse cx="490" cy="120" rx="90" ry="70" fill="none" stroke="#1565c0" stroke-width="1.8"/>
  <text x="548" y="62" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">T₂ — comp. of T</text>
  <line x1="262" y1="120" x2="412" y2="120" stroke="#1565c0" stroke-width="2.8"/>
  <text x="337" y="108" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#1565c0">e</text>
  <line x1="252" y1="76" x2="416" y2="70" stroke="#666" stroke-width="1.2" stroke-dasharray="4 4"/>
  <line x1="254" y1="166" x2="414" y2="172" stroke="#666" stroke-width="1.2" stroke-dasharray="4 4"/>
  <circle cx="262" cy="120" r="8" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="1.8"/>
  <circle cx="412" cy="120" r="8" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.8"/>
  <circle cx="248" cy="76" r="5" fill="#fff" stroke="#666" stroke-width="1.3"/>
  <circle cx="420" cy="70" r="5" fill="#fff" stroke="#666" stroke-width="1.3"/>
  <circle cx="250" cy="166" r="5" fill="#fff" stroke="#666" stroke-width="1.3"/>
  <circle cx="418" cy="172" r="5" fill="#fff" stroke="#666" stroke-width="1.3"/>
  <text x="320" y="222" text-anchor="middle" font-family="serif" font-size="11" fill="#666">e is the first cut edge the sorted scan meets ⇒ the lightest of the cut (dashed rivals are heavier)</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Accepting $e$: among all edges of the cut $E(T_1, V \setminus T_1)$, the sorted order delivered $e$ first — the lightest — so accepting it is an application of the blue rule.
</figcaption>
</figure>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 250" width="100%" style="max-width: 500px; height: auto;" role="img" aria-labelledby="gaf61-title">
  <title id="gaf61-title">A rejected Kruskal edge is the heaviest edge of the cycle it closes</title>
  <path d="M 130,140 C 124,66 230,34 330,38 C 430,42 500,86 496,146 C 492,200 410,226 310,222 C 210,218 136,196 130,140 Z" fill="none" stroke="#0f9b6c" stroke-width="1.8"/>
  <text x="160" y="60" font-family="serif" font-size="13" font-style="italic" fill="#0f9b6c">T₁</text>
  <line x1="220" y1="110" x2="300" y2="80" stroke="#1565c0" stroke-width="2.4"/>
  <line x1="300" y1="80" x2="390" y2="100" stroke="#1565c0" stroke-width="2.4"/>
  <line x1="390" y1="100" x2="420" y2="160" stroke="#1565c0" stroke-width="2.4"/>
  <path d="M 228,122 C 280,190 360,196 412,170" fill="none" stroke="#b91c1c" stroke-width="2.6"/>
  <text x="316" y="206" text-anchor="middle" font-family="serif" font-size="13" font-style="italic" fill="#b91c1c">e</text>
  <circle cx="220" cy="110" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="300" cy="80" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="390" cy="100" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="420" cy="160" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="310" y="244" text-anchor="middle" font-family="serif" font-size="11" fill="#666">every blue edge of the closed cycle was considered — and accepted — before e ⇒ e is heaviest</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Rejecting $e$: the tree path inside $T_1$ consists of earlier (lighter) edges, so $e$ is the heaviest edge of the cycle it would close — the red rule.
</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Complexity of Kruskal — Enter Union–Find)</span></p>

Sorting costs $O(m \log n)$. The main loop asks $m$ times "are the endpoints in the same component?" and merges components at most $n - 1$ times. Recomputing components naively gives $O(mn)$ overall. Better: a **Union–Find** data structure maintaining connected components under edge insertions, with operations

* $\mathrm{Find}(x, y)$ — are $x$ and $y$ in the same component?
* $\mathrm{Union}(x, y)$ — add the edge $xy$, merging two components.

How fast Union–Find can be made is a story of its own.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 240" width="100%" style="max-width: 580px; height: auto;" role="img" aria-labelledby="gaf62-title">
  <title id="gaf62-title">Union-Find maintains the connected components under edge insertions</title>
  <ellipse cx="130" cy="110" rx="85" ry="62" fill="none" stroke="#333" stroke-width="1.5"/>
  <ellipse cx="350" cy="110" rx="80" ry="60" fill="none" stroke="#333" stroke-width="1.5"/>
  <ellipse cx="545" cy="110" rx="70" ry="58" fill="none" stroke="#333" stroke-width="1.5"/>
  <circle cx="100" cy="90" r="5" fill="#666"/>
  <circle cx="150" cy="130" r="5" fill="#666"/>
  <circle cx="160" cy="85" r="5" fill="#666"/>
  <circle cx="330" cy="95" r="5" fill="#666"/>
  <circle cx="390" cy="130" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="390" y="152" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#1565c0">x</text>
  <circle cx="520" cy="130" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="520" y="152" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#1565c0">y</text>
  <circle cx="560" cy="85" r="5" fill="#666"/>
  <line x1="396" y1="130" x2="514" y2="130" stroke="#1565c0" stroke-width="2.4"/>
  <text x="455" y="180" text-anchor="middle" font-family="serif" font-size="11" fill="#1565c0">Union(x, y)</text>
  <text x="320" y="206" text-anchor="middle" font-family="serif" font-size="11" fill="#666">Find(x, y): are x and y in the same component?&nbsp;&nbsp;&nbsp;Union(x, y): insert the edge, merging two blobs</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The Union–Find interface: components of the growing forest as disjoint sets, queried by $\mathrm{Find}$ and merged by $\mathrm{Union}$.
</figcaption>
</figure>

# Lecture 10: Faster Minimum Spanning Trees

All three classical algorithms hover around $O(m \log n)$. This lecture breaks that barrier three times: **contractive Borůvka** gets linear time on planar graphs — and, after a detour into **graph minors**, on every non-trivial minor-closed class; Jarník with a **Fibonacci heap** reaches $O(m + n \log n)$; and the **Fredman–Tarjan** algorithm combines both ideas into $O(m \log^\ast n)$ for all graphs.

## Contractive Borůvka

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Contractive Borůvka)</span></p>

Borůvka from last lecture, with each round's blue forest immediately **contracted**:

1. $T \leftarrow \emptyset$.
2. **While** $n > 1$:
3. &nbsp;&nbsp;&nbsp;every vertex selects the lightest incident edge; call the selection $L$ — $O(m)$;
4. &nbsp;&nbsp;&nbsp;$T \leftarrow T \cup L$ — $O(n)$;
5. &nbsp;&nbsp;&nbsp;contract all edges of $L$ — $O(m)$;
6. &nbsp;&nbsp;&nbsp;filter out loops and parallel edges, keeping the lightest edge of every parallel bundle — $O(m)$ by bucket sorting;
7. Return $T$.

Every edge carries its original **id**, so $T$ is collected in terms of the input graph's edges even as the working graph shrinks.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 258" width="100%" style="max-width: 580px; height: auto;" role="img" aria-labelledby="gaf63-title">
  <title id="gaf63-title">Each blue tree of the round is contracted to a single vertex</title>
  <defs>
    <marker id="gaf63-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
    </marker>
  </defs>
  <ellipse cx="140" cy="90" rx="72" ry="52" fill="none" stroke="#c2185b" stroke-width="1.5" stroke-dasharray="6 4"/>
  <ellipse cx="190" cy="185" rx="70" ry="40" fill="none" stroke="#c2185b" stroke-width="1.5" stroke-dasharray="6 4"/>
  <line x1="110" y1="80" x2="160" y2="60" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="110" y1="80" x2="140" y2="120" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="150" y1="180" x2="210" y2="165" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="210" y1="165" x2="240" y2="200" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="140" y1="120" x2="150" y2="180" stroke="#666" stroke-width="1.2" stroke-dasharray="4 4"/>
  <line x1="160" y1="60" x2="270" y2="120" stroke="#666" stroke-width="1.2" stroke-dasharray="4 4"/>
  <line x1="240" y1="200" x2="270" y2="120" stroke="#666" stroke-width="1.2" stroke-dasharray="4 4"/>
  <circle cx="110" cy="80" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="160" cy="60" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="140" cy="120" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="150" cy="180" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="210" cy="165" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="240" cy="200" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="270" cy="120" r="6" fill="#fff" stroke="#333" stroke-width="1.4"/>
  <line x1="320" y1="125" x2="368" y2="125" stroke="#333" stroke-width="1.8" marker-end="url(#gaf63-arr)"/>
  <circle cx="440" cy="85" r="17" fill="#e3f2fd" stroke="#1565c0" stroke-width="2.2"/>
  <circle cx="480" cy="185" r="17" fill="#e3f2fd" stroke="#1565c0" stroke-width="2.2"/>
  <circle cx="580" cy="125" r="8" fill="#fff" stroke="#333" stroke-width="1.4"/>
  <line x1="448" y1="101" x2="472" y2="169" stroke="#666" stroke-width="1.4"/>
  <line x1="456" y1="92" x2="572" y2="121" stroke="#666" stroke-width="1.4"/>
  <line x1="496" y1="177" x2="573" y2="131" stroke="#666" stroke-width="1.4"/>
  <text x="320" y="248" text-anchor="middle" font-family="serif" font-size="11" fill="#666">every tree of the round (dashed) becomes one vertex; surviving edges connect the new vertices</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
One contraction round: each blue tree collapses into a single vertex of the next graph; edges between different trees survive.
</figcaption>
</figure>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 220" width="100%" style="max-width: 580px; height: auto;" role="img" aria-labelledby="gaf64-title">
  <title id="gaf64-title">Edges keep their original identifiers through contractions</title>
  <defs>
    <marker id="gaf64-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
    </marker>
  </defs>
  <ellipse cx="150" cy="110" rx="80" ry="60" fill="none" stroke="#c2185b" stroke-width="1.5" stroke-dasharray="6 4"/>
  <line x1="120" y1="100" x2="170" y2="70" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="120" y1="100" x2="160" y2="145" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="170" y1="70" x2="295" y2="62" stroke="#a86f00" stroke-width="2"/>
  <text x="235" y="50" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#a86f00">id 17</text>
  <circle cx="120" cy="100" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="170" cy="70" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="176" y="92" font-family="serif" font-size="11" font-style="italic" fill="#1565c0">u</text>
  <circle cx="160" cy="145" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="305" cy="62" r="9" fill="#fff" stroke="#333" stroke-width="1.6"/>
  <text x="305" y="42" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#333">v</text>
  <line x1="360" y1="110" x2="404" y2="110" stroke="#333" stroke-width="1.8" marker-end="url(#gaf64-arr)"/>
  <circle cx="470" cy="120" r="18" fill="#e3f2fd" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="486" y1="112" x2="585" y2="75" stroke="#a86f00" stroke-width="2"/>
  <text x="545" y="76" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#a86f00">id 17</text>
  <circle cx="592" cy="72" r="9" fill="#fff" stroke="#333" stroke-width="1.6"/>
  <text x="592" y="52" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#333">v</text>
  <text x="320" y="204" text-anchor="middle" font-family="serif" font-size="11" fill="#666">after contracting the cluster, edge 17 now leaves the merged vertex — but still means the original uv</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Every edge keeps its original id through all contractions, so the tree $T$ is always reported as a set of edges of the input graph.
</figcaption>
</figure>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 230" width="100%" style="max-width: 580px; height: auto;" role="img" aria-labelledby="gaf65-title">
  <title id="gaf65-title">Contraction creates loops and parallel edges, which are filtered away</title>
  <defs>
    <marker id="gaf65-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
    </marker>
  </defs>
  <line x1="110" y1="70" x2="110" y2="170" stroke="#1565c0" stroke-width="3"/>
  <text x="92" y="124" text-anchor="middle" font-family="serif" font-size="11" fill="#1565c0">contract</text>
  <line x1="110" y1="70" x2="250" y2="120" stroke="#333" stroke-width="1.8"/>
  <line x1="110" y1="170" x2="250" y2="120" stroke="#333" stroke-width="1.8"/>
  <circle cx="110" cy="70" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="94" y="62" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#1565c0">a</text>
  <circle cx="110" cy="170" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="94" y="182" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#1565c0">b</text>
  <circle cx="250" cy="120" r="7" fill="#fff" stroke="#333" stroke-width="1.5"/>
  <text x="268" y="112" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#333">c</text>
  <line x1="310" y1="120" x2="354" y2="120" stroke="#333" stroke-width="1.8" marker-end="url(#gaf65-arr)"/>
  <path d="M 405,105 C 392,72 436,66 434,96" fill="none" stroke="#b91c1c" stroke-width="1.8" stroke-dasharray="5 4"/>
  <text x="420" y="52" text-anchor="middle" font-family="serif" font-size="11" fill="#b91c1c">loop — drop</text>
  <path d="M 434,113 C 470,86 540,86 574,113" fill="none" stroke="#0f9b6c" stroke-width="2.2"/>
  <text x="504" y="84" text-anchor="middle" font-family="serif" font-size="11" fill="#0f9b6c">keep the lightest</text>
  <path d="M 434,127 C 470,154 540,154 574,127" fill="none" stroke="#b91c1c" stroke-width="1.8" stroke-dasharray="5 4"/>
  <text x="504" y="170" text-anchor="middle" font-family="serif" font-size="11" fill="#b91c1c">parallel — drop</text>
  <circle cx="420" cy="120" r="13" fill="#e3f2fd" stroke="#1565c0" stroke-width="2"/>
  <text x="420" y="125" text-anchor="middle" font-family="serif" font-size="10" font-style="italic" fill="#1565c0">ab</text>
  <circle cx="586" cy="120" r="9" fill="#fff" stroke="#333" stroke-width="1.5"/>
  <text x="604" y="112" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#333">c</text>
  <text x="320" y="218" text-anchor="middle" font-family="serif" font-size="11" fill="#666">contracting ab: the former edges ac, bc become a parallel pair; a contracted parallel of ab becomes a loop</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Filtering after a contraction: loops disappear, and of every parallel bundle only the lightest edge can ever enter the MST — the rest is deleted (red rule, in fact).
</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Filtering in Linear Time)</span></p>

Keep vertices numbered $1, \dots, n\_i$ in each phase and normalize every edge $\lbrace u, v \rbrace$ to the triple $(\min(u,v),\ \max(u,v),\ \mathrm{id})$. Two passes of **bucket sort** with $n\_i$ buckets — by the second coordinate, then by the first — arrange the edges lexicographically in $O(\text{passes} \cdot (\text{items} + \text{buckets})) = O(m\_i + n\_i)$. Parallel edges are now adjacent: keep the lightest of each run, drop loops.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 220" width="100%" style="max-width: 560px; height: auto;" role="img" aria-labelledby="gaf66-title">
  <title id="gaf66-title">Radix sorting the normalized edge triples into n buckets</title>
  <defs>
    <marker id="gaf66-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#666"/>
    </marker>
  </defs>
  <text x="170" y="38" text-anchor="middle" font-family="serif" font-size="12" fill="#333">(2, 5, id₇)</text>
  <text x="320" y="38" text-anchor="middle" font-family="serif" font-size="12" fill="#333">(1, 4, id₃)</text>
  <text x="470" y="38" text-anchor="middle" font-family="serif" font-size="12" fill="#333">(2, 5, id₉)</text>
  <line x1="180" y1="48" x2="212" y2="86" stroke="#666" stroke-width="1.2" marker-end="url(#gaf66-arr)"/>
  <line x1="310" y1="48" x2="168" y2="86" stroke="#666" stroke-width="1.2" marker-end="url(#gaf66-arr)"/>
  <line x1="462" y1="48" x2="224" y2="88" stroke="#666" stroke-width="1.2" marker-end="url(#gaf66-arr)"/>
  <rect x="100" y="92" width="440" height="36" fill="#fff" stroke="#333" stroke-width="1.6"/>
  <line x1="155" y1="92" x2="155" y2="128" stroke="#333" stroke-width="1"/>
  <line x1="210" y1="92" x2="210" y2="128" stroke="#333" stroke-width="1"/>
  <line x1="265" y1="92" x2="265" y2="128" stroke="#333" stroke-width="1"/>
  <line x1="320" y1="92" x2="320" y2="128" stroke="#333" stroke-width="1"/>
  <line x1="375" y1="92" x2="375" y2="128" stroke="#333" stroke-width="1"/>
  <line x1="430" y1="92" x2="430" y2="128" stroke="#333" stroke-width="1"/>
  <line x1="485" y1="92" x2="485" y2="128" stroke="#333" stroke-width="1"/>
  <text x="320" y="152" text-anchor="middle" font-family="serif" font-size="11" fill="#666">n buckets</text>
  <text x="320" y="182" text-anchor="middle" font-family="serif" font-size="11" fill="#666">two passes — second coordinate, then first: lexicographic order, duplicates adjacent</text>
  <text x="320" y="204" text-anchor="middle" font-family="serif" font-size="11" fill="#666">O(passes · (items + buckets)) = O(m + n)</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Two bucket-sort passes over the normalized triples group parallel edges next to each other — no comparison sorting needed.
</figcaption>
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Phase Shrinkage)</span></p>

Let $G\_i$ be the graph at the *end* of phase $i$ (so $G\_0$ is the input), with $n\_i$ vertices and $m\_i$ edges. Then $m\_i \le m$, and every contracted tree swallows at least two vertices, so

$$
n_{i+1} \;\le\; n_i / 2 \;\le\; n / 2^{\,i+1}
$$

— at most $\log n$ phases, each costing $O(m\_i + n\_i)$.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(General Bounds)</span></p>

Contractive Borůvka runs in time

$$
O(m \log n) \qquad\text{and also}\qquad O\Bigl(\textstyle\sum_i m_i\Bigr) \;\le\; O\Bigl(\textstyle\sum_i \frac{n^2}{4^i}\Bigr) \;=\; O(n^2),
$$

the second bound because filtering keeps every $G\_i$ simple, so $m\_i \le n\_i^2 \le (n/2^i)^2$. On dense graphs the $O(n^2)$ bound wins.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Planar Graphs: MST in Linear Time)</span></p>

On a planar graph, contractive Borůvka runs in $O(n)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Contractions and deletions preserve planarity, so every $G\_i$ is planar — and simple, thanks to the filtering. A simple planar graph satisfies $m\_i \le 3 n\_i$ (Euler), hence

$$
\sum_i m_i \;\le\; 3 \sum_i n_i \;\le\; 3n \sum_i 2^{-i} \;\in\; O(n). \qquad ∎
$$

</details>
</div>

## A Detour: Graph Minors

The planar argument used only two facts: planarity survives contractions, and planar graphs have few edges. Both generalize.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Minor)</span></p>

$H \preceq G$ ($H$ is a **minor** of $G$) $\equiv$ $H$ can be obtained from $G$ by a sequence of vertex deletions, edge deletions, and edge contractions. For example, the triangle is a minor of $C\_5$.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 560 230" width="100%" style="max-width: 440px; height: auto;" role="img" aria-labelledby="gaf67-title">
  <title id="gaf67-title">The triangle is a minor of the five-cycle: contract two edges</title>
  <line x1="140" y1="60" x2="90" y2="170" stroke="#333" stroke-width="2"/>
  <line x1="90" y1="170" x2="190" y2="170" stroke="#333" stroke-width="2"/>
  <line x1="190" y1="170" x2="140" y2="60" stroke="#333" stroke-width="2"/>
  <circle cx="140" cy="60" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="90" cy="170" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="190" cy="170" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="280" y="130" text-anchor="middle" font-family="serif" font-size="24" fill="#333">≼</text>
  <line x1="430" y1="45" x2="354" y2="100" stroke="#b91c1c" stroke-width="2.2" stroke-dasharray="6 4"/>
  <line x1="354" y1="100" x2="383" y2="190" stroke="#b91c1c" stroke-width="2.2" stroke-dasharray="6 4"/>
  <line x1="383" y1="190" x2="477" y2="190" stroke="#333" stroke-width="2"/>
  <line x1="477" y1="190" x2="506" y2="100" stroke="#333" stroke-width="2"/>
  <line x1="506" y1="100" x2="430" y2="45" stroke="#333" stroke-width="2"/>
  <text x="352" y="66" text-anchor="middle" font-family="serif" font-size="11" fill="#b91c1c">contract these</text>
  <circle cx="430" cy="45" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="354" cy="100" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="383" cy="190" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="477" cy="190" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="506" cy="100" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Contracting the two dashed edges of $C_5$ leaves a triangle: $K_3 \preceq C_5$.
</figcaption>
</figure>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Minor-Closed Class, Edge Density)</span></p>

A class $\mathcal{C}$ of graphs is **minor-closed** $\equiv$ $\forall G \in \mathcal{C},\ \forall H \preceq G:\ H \in \mathcal{C}$; it is **non-trivial** if it is neither empty nor all graphs. Examples: the two trivial classes; planar graphs; graphs drawable on a fixed surface; graphs of bounded tree-width; forests.

The **edge density** of a graph and of a class:

$$
\rho(G) \;:=\; \frac{m(G)}{n(G)}, \qquad \rho(\mathcal{C}) \;:=\; \sup_{G \in \mathcal{C}} \rho(G)
\qquad\text{(for planar graphs: } \rho = 3\text{)}.
$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Density of Minor-Closed Classes)</span></p>

For every non-trivial minor-closed class $\mathcal{C}$ of graphs, $\rho(\mathcal{C})$ is **finite**.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Contractive Borůvka on Minor-Closed Classes)</span></p>

Let $G$ belong to a non-trivial minor-closed class $\mathcal{C}$. Every $G\_i$ is a minor of $G$, hence $G\_i \in \mathcal{C}$ and $m\_i \le \rho(\mathcal{C}) \cdot n\_i$ — so

$$
\sum_i m_i \;\le\; \rho(\mathcal{C}) \cdot n \sum_i 2^{-i} \;\in\; O(\rho(\mathcal{C}) \cdot n),
$$

linear time for every fixed class $\mathcal{C}$.

</div>

Proving the density theorem needs a language for minor-closed classes.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Forbidden Minors)</span></p>

For a class $\mathcal{H}$ of graphs, $\mathrm{Forb}(\mathcal{H})$ := the class of all graphs having **no member of $\mathcal{H}$ as a minor**. Examples: $\mathrm{Forb}(K\_2)$ = edgeless graphs, $\mathrm{Forb}(K\_3)$ = forests, $\mathrm{Forb}(K\_5, K\_{3,3})$ = planar graphs (the minor version of Kuratowski's theorem).

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Forb Is Minor-Closed)</span></p>

$\mathrm{Forb}(\mathcal{H})$ is always minor-closed: if $G' \preceq G$ and some $H \in \mathcal{H}$ satisfied $H \preceq G'$, then $H \preceq G$ too — minors of minors are minors.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Minor-Closed $=$ Forb)</span></p>

Every minor-closed class $\mathcal{C}$ equals $\mathrm{Forb}(\mathcal{H})$ for some $\mathcal{H}$ — e.g. $\mathcal{H} := \overline{\mathcal{C}}$, the complement class: a graph lies outside $\mathcal{C}$ iff one of its minors (itself!) lies outside $\mathcal{C}$. Far deeper is the **Robertson–Seymour graph minor theorem**: a *finite* $\mathcal{H}$ always suffices.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Mader; see Diestel, Graph Theory)</span></p>

For every $k$ there is an $h(k)$ such that every graph with average degree at least $h(k)$ contains a **subdivision of $K\_k$** — Mader's proof gives $h(k) \approx 2^{k^2}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof of the density theorem</summary>

Density is monotone: $\mathcal{C}' \subseteq \mathcal{C}$ implies $\rho(\mathcal{C}') \le \rho(\mathcal{C})$. Write the non-trivial $\mathcal{C}$ as $\mathrm{Forb}(\mathcal{H})$ with $\mathcal{H} \neq \emptyset$, pick any $H \in \mathcal{H}$ and set $k := n(H)$. Since $H \preceq K\_k$, avoiding $H$ is implied by avoiding... rather: containing $K\_k$ implies containing $H$, so

$$
\mathcal{C} \;=\; \mathrm{Forb}(\mathcal{H}) \;\subseteq\; \mathrm{Forb}(H) \;\subseteq\; \mathrm{Forb}(K_k),
\qquad
\rho(\mathcal{C}) \;\le\; \rho(\mathrm{Forb}(K_k)).
$$

Now Mader: a graph $G$ with $\rho(G) \ge h(k)/2$ has average degree $\ge h(k)$, so it contains a subdivision of $K\_k$ — and a subdivision is a minor, so $K\_k \preceq G$ and $G \notin \mathrm{Forb}(K\_k)$. Hence $\rho(\mathrm{Forb}(K\_k)) \le h(k)/2 < \infty$. ∎

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Kostochka–Thomason)</span></p>

The optimal threshold is much smaller than Mader's: already $\rho(G) \in \Omega(k \sqrt{\log k})$ forces $K\_k \preceq G$.

</div>

## Jarník with a Fibonacci Heap

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Jarník, Improved "Dijkstra-Style" Implementation)</span></p>

Maintain, for every **neighbor** $v$ of the current tree, its **value** — the weight of the lightest edge joining $v$ to the tree — and keep the neighbors in a heap keyed by value. In every step:

1. ExtractMin the neighbor $v$ with the smallest value, giving the edge $uv$ ($u \in T$, $v \notin T$) — $n\times$;
2. add $uv$ to $T$;
3. update values: Insert the new neighbors of $v$ — $n\times$ in total — and Decrease the values of existing neighbors whose lightest edge changed — $m\times$ in total.

With a Fibonacci heap: $O(m + n \log n)$ — already **linear** whenever $\rho = m/n \ge \log n$.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 260" width="100%" style="max-width: 600px; height: auto;" role="img" aria-labelledby="gaf68-title">
  <title id="gaf68-title">Each neighbor of the current tree carries the weight of its lightest edge into the tree</title>
  <line x1="150" y1="120" x2="110" y2="80" stroke="#1565c0" stroke-width="2.4"/>
  <line x1="150" y1="120" x2="200" y2="80" stroke="#1565c0" stroke-width="2.4"/>
  <line x1="150" y1="120" x2="170" y2="180" stroke="#1565c0" stroke-width="2.4"/>
  <line x1="170" y1="180" x2="230" y2="145" stroke="#1565c0" stroke-width="2.4"/>
  <line x1="200" y1="80" x2="352" y2="72" stroke="#c2185b" stroke-width="2.4"/>
  <line x1="230" y1="145" x2="352" y2="78" stroke="#666" stroke-width="1.2" stroke-dasharray="4 4"/>
  <line x1="230" y1="145" x2="382" y2="150" stroke="#c2185b" stroke-width="2.4"/>
  <line x1="200" y1="80" x2="380" y2="144" stroke="#666" stroke-width="1.2" stroke-dasharray="4 4"/>
  <line x1="170" y1="180" x2="322" y2="216" stroke="#c2185b" stroke-width="2.4"/>
  <line x1="230" y1="145" x2="320" y2="212" stroke="#666" stroke-width="1.2" stroke-dasharray="4 4"/>
  <circle cx="150" cy="120" r="8" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <circle cx="110" cy="80" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="200" cy="80" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="170" cy="180" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="230" cy="145" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="366" cy="72" r="8" fill="#fff" stroke="#c2185b" stroke-width="1.8"/>
  <circle cx="396" cy="150" r="8" fill="#fff" stroke="#c2185b" stroke-width="1.8"/>
  <circle cx="336" cy="220" r="8" fill="#fff" stroke="#c2185b" stroke-width="1.8"/>
  <text x="505" y="70" text-anchor="middle" font-family="serif" font-size="11" fill="#c2185b">value of a neighbor =</text>
  <text x="505" y="88" text-anchor="middle" font-family="serif" font-size="11" fill="#c2185b">weight of its lightest edge into T</text>
  <text x="505" y="150" text-anchor="middle" font-family="serif" font-size="11" fill="#666">gray rivals are heavier —</text>
  <text x="505" y="168" text-anchor="middle" font-family="serif" font-size="11" fill="#666">they surface later via Decrease</text>
  <text x="320" y="250" text-anchor="middle" font-family="serif" font-size="11" fill="#666">neighbors (magenta rings) live in a Fibonacci heap keyed by their values</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The improved Jarník bookkeeping: heap entries are the *neighbors* of the tree, each carrying its lightest connecting edge (magenta) — $n$ ExtractMins and Inserts, $m$ Decreases.
</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(A Toy Combination: $O(m \log\log n)$)</span></p>

The guiding idea for what follows: **identify some $F \subseteq$ MST, add $F$ to $T$, contract $F$, repeat on the smaller graph.** Toy instance: run $\log\log n$ contractive-Borůvka phases — time $O(m \log\log n)$ — leaving $n' \le n / 2^{\log\log n} = n / \log n$ vertices and $m' \le m$ edges; then finish with Fibonacci-heap Jarník in

$$
O(m' + n' \log n') \;\le\; O\Bigl(m + \frac{n}{\log n} \cdot \log n\Bigr) \;=\; O(m).
$$

Total: $O(m \log\log n)$ — already beating $O(m \log n)$.

</div>

## The Fredman–Tarjan Algorithm

Fredman and Tarjan (1987) push the combination much further: run Jarník *many times in parallel* with a **bounded heap**, and contract the resulting forest.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Fredman–Tarjan)</span></p>

Let $m\_0$ be the number of edges of the *input* graph and $G\_i$ the graph at the **start** of phase $i$ (with $n\_i$ vertices). Phase $i$ uses the heap-size limit

$$
t_i \;:=\; 2^{\lceil 2 m_0 / n_i \rceil}.
$$

Within the phase, repeatedly pick a vertex not yet in any tree and grow a Jarník tree from it (Fibonacci heap over the neighbors), **stopping the growth** as soon as:

* the heap becomes empty (the tree swallowed its whole component), or
* the heap size exceeds $t\_i$, or
* the tree touches a previously grown tree — merge with it.

When every vertex is in a tree, add the forest $F$ to $T$, contract every tree of $F$, filter loops and parallel edges as in contractive Borůvka, and start the next phase.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 240" width="100%" style="max-width: 580px; height: auto;" role="img" aria-labelledby="gaf69-title">
  <title id="gaf69-title">A Fredman-Tarjan phase grows a forest of bounded trees one after another</title>
  <path d="M 60,120 C 56,52 170,20 320,24 C 470,28 584,64 580,124 C 576,190 460,222 310,218 C 160,214 64,186 60,120 Z" fill="none" stroke="#333" stroke-width="1.5"/>
  <line x1="150" y1="90" x2="205" y2="70" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="150" y1="90" x2="185" y2="130" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="150" y1="90" x2="105" y2="120" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="330" y1="150" x2="385" y2="165" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="385" y1="165" x2="430" y2="130" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="460" y1="70" x2="515" y2="85" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="460" y1="70" x2="500" y2="40" stroke="#1565c0" stroke-width="2.2"/>
  <circle cx="150" cy="90" r="8" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <circle cx="205" cy="70" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="185" cy="130" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="105" cy="120" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="330" cy="150" r="8" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <circle cx="385" cy="165" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="430" cy="130" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="460" cy="70" r="8" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <circle cx="515" cy="85" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="500" cy="40" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="255" cy="185" r="4" fill="#666"/>
  <circle cx="545" cy="160" r="4" fill="#666"/>
  <text x="320" y="232" text-anchor="middle" font-family="serif" font-size="11" fill="#666">the phase's forest F — trees started at fresh vertices (green), each stopped by one of the three rules</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
One phase grows a forest of blue trees, one after another; at the end the whole forest joins $T$ and is contracted away.
</figcaption>
</figure>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 250" width="100%" style="max-width: 580px; height: auto;" role="img" aria-labelledby="gaf70-title">
  <title id="gaf70-title">A stopped tree is incident to at least t edges, or ran into an existing tree</title>
  <line x1="280" y1="130" x2="240" y2="90" stroke="#1565c0" stroke-width="2.4"/>
  <line x1="280" y1="130" x2="320" y2="90" stroke="#1565c0" stroke-width="2.4"/>
  <line x1="280" y1="130" x2="250" y2="170" stroke="#1565c0" stroke-width="2.4"/>
  <line x1="280" y1="130" x2="310" y2="170" stroke="#1565c0" stroke-width="2.4"/>
  <line x1="240" y1="90" x2="150" y2="62" stroke="#b91c1c" stroke-width="1.8"/>
  <line x1="240" y1="90" x2="222" y2="34" stroke="#b91c1c" stroke-width="1.8"/>
  <line x1="320" y1="90" x2="362" y2="40" stroke="#b91c1c" stroke-width="1.8"/>
  <line x1="320" y1="90" x2="420" y2="98" stroke="#b91c1c" stroke-width="1.8"/>
  <line x1="250" y1="170" x2="176" y2="212" stroke="#b91c1c" stroke-width="1.8"/>
  <line x1="310" y1="170" x2="360" y2="216" stroke="#b91c1c" stroke-width="1.8"/>
  <line x1="310" y1="170" x2="446" y2="152" stroke="#b91c1c" stroke-width="1.8"/>
  <circle cx="280" cy="130" r="8" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <circle cx="240" cy="90" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="320" cy="90" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="250" cy="170" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="310" cy="170" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="290" y="20" text-anchor="middle" font-family="serif" font-size="11" fill="#b91c1c">heap overflow: ≥ tᵢ incident edges when growth stops</text>
  <ellipse cx="530" cy="150" rx="62" ry="46" fill="none" stroke="#0f9b6c" stroke-width="1.6"/>
  <line x1="492" y1="122" x2="548" y2="170" stroke="#0f9b6c" stroke-width="1.4"/>
  <line x1="510" y1="115" x2="566" y2="163" stroke="#0f9b6c" stroke-width="1.4"/>
  <line x1="530" y1="110" x2="578" y2="152" stroke="#0f9b6c" stroke-width="1.4"/>
  <text x="530" y="222" text-anchor="middle" font-family="serif" font-size="11" fill="#0f9b6c">or: ran into an existing tree</text>
  <text x="280" y="244" text-anchor="middle" font-family="serif" font-size="11" fill="#666">in a non-final phase, every final tree of F ends up incident to at least tᵢ edges</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Why trees are "big": growth stops with an overflowing heap — at least $t_i$ edges touching the tree — or by fusing into an earlier tree, which itself stopped for the same reason.
</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Fredman–Tarjan Complexity)</span></p>

The Fredman–Tarjan algorithm computes the MST in time $O(m \log^\ast n)$, where $\log^\ast n$ is the iterated logarithm — the number of times $\log$ must be applied to $n$ to fall below a constant.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**① One phase costs $O(m\_0)$.** Every edge of $G\_i$ is inserted or decreased $O(1)$ times (amortized $O(1)$ in a Fibonacci heap), and there are at most $n\_i$ ExtractMins, each $O(\log t\_i)$ since heaps never exceed $t\_i$:

$$
O(m_i + n_i \log t_i) \;=\; O\!\left(m_i + n_i \cdot \frac{2m_0}{n_i}\right) \;=\; O(m_0).
$$

**② In a non-final phase, every tree of $F$ is incident to at least $t\_i$ edges.** Consider the *earliest-started* growth inside a final tree $K$: it cannot have stopped by joining an existing tree (there was none in $K$ before it), and if its heap emptied, its tree swallowed a whole component of the (connected) graph — making the phase final. So it stopped with heap size over $t\_i$, and each heap entry witnesses a distinct edge incident to $K$.

**③ Trees are few.** Every edge is incident to at most two trees, so

$$
n_{i+1} \;=\; \#\text{trees of } F \;\le\; \frac{2 m_0}{t_i}.
$$

**④ The limit explodes.** By ③,

$$
t_{i+1} \;=\; 2^{\lceil 2m_0 / n_{i+1} \rceil} \;\ge\; 2^{\,2m_0 \,/\, (2m_0/t_i)} \;=\; 2^{\,t_i},
$$

so $t\_1, t\_2, t\_3, \dots$ grows as a tower of twos of height $i$. As soon as $t\_i \ge n$, the heap limit can never bind, the first tree swallows all of $G\_i$, and the phase is final — that happens after at most $\log^\ast n$ phases, and with ① the theorem follows. ∎

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(Denser Graphs Finish Faster)</span></p>

The tower does not start at $2$ but at $t\_1 = 2^{\lceil 2m/n \rceil}$: the exact number of phases is $\beta(m, n) := \min \lbrace i : \log^{(i)} n \le m/n \rbrace$, giving time $O(m \, \beta(m,n))$. In particular, if $\rho = m/n \ge \log^{(k)} n$ (the $k$-times iterated logarithm), only $O(k)$ phases run: time $O(km)$.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Toward Linear Time)</span></p>

**Open problem:** is there a deterministic comparison-based MST algorithm in $O(m)$ for all graphs? What *is* known: $O(m)$ **expected** time by a randomized algorithm (Karger–Klein–Tarjan), and $O(m)$ worst-case for **integer** weights on the word RAM (Fredman–Willard) — optimal in both settings.

</div>

# Lecture 11: Dynamic Connectivity

Kruskal's algorithm met Union–Find: components under edge *insertions*. This lecture tackles the full problem — **Union–Find with deletions**.

## The Problem

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Info</span><span class="math-callout__name">(Dynamic Connectivity)</span></p>

Maintain a graph under **insertions and deletions** of vertices and edges, answering **connectivity queries** "are $u$ and $v$ in the same component?". Insertions alone are Union–Find; deletions are the hard part — a deleted edge may or may not disconnect anything, depending on the rest of the graph. **Goal:** polylogarithmic ($\approx O(\log n)$) amortized time per operation.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 220" width="100%" style="max-width: 540px; height: auto;" role="img" aria-labelledby="gaf72-title">
  <title id="gaf72-title">Edges come and go; queries ask whether two vertices are connected</title>
  <line x1="120" y1="80" x2="180" y2="130" stroke="#1565c0" stroke-width="2"/>
  <line x1="180" y1="130" x2="130" y2="180" stroke="#1565c0" stroke-width="2"/>
  <line x1="180" y1="130" x2="250" y2="90" stroke="#1565c0" stroke-width="2"/>
  <line x1="420" y1="70" x2="470" y2="130" stroke="#1565c0" stroke-width="2"/>
  <line x1="470" y1="130" x2="420" y2="180" stroke="#1565c0" stroke-width="2"/>
  <line x1="470" y1="130" x2="540" y2="100" stroke="#1565c0" stroke-width="2"/>
  <line x1="250" y1="90" x2="420" y2="70" stroke="#b91c1c" stroke-width="2" stroke-dasharray="7 5"/>
  <text x="335" y="62" text-anchor="middle" font-family="serif" font-size="11" fill="#b91c1c">inserted, later deleted</text>
  <circle cx="120" cy="80" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="180" cy="130" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="130" cy="180" r="10" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="112" y="196" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#0f9b6c">u</text>
  <circle cx="250" cy="90" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="420" cy="70" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="470" cy="130" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="420" cy="180" r="10" fill="#fef2f2" stroke="#b91c1c" stroke-width="2"/>
  <text x="440" y="196" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#b91c1c">v</text>
  <circle cx="540" cy="100" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="310" y="212" text-anchor="middle" font-family="serif" font-size="11" fill="#666">query: connected(u, v)? — the answer flips as edges appear and disappear</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Dynamic connectivity: edges are inserted and deleted online, and queries ask whether two vertices currently lie in the same component.
</figcaption>
</figure>

## Euler-Tour Trees: Connectivity for Forests

First the case where the graph is always a **forest**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(ET Sequence)</span></p>

Root the tree and run DFS, **recording every vertex each time it is visited**. The resulting *Euler-tour sequence* has length $2m + 1 = 2n - 1 \in \Theta(n)$.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 310" width="100%" style="max-width: 480px; height: auto;" role="img" aria-labelledby="gaf73-title">
  <title id="gaf73-title">A rooted tree and its Euler-tour sequence recorded by DFS</title>
  <line x1="310" y1="48" x2="200" y2="110" stroke="#1565c0" stroke-width="2"/>
  <line x1="310" y1="48" x2="310" y2="110" stroke="#1565c0" stroke-width="2"/>
  <line x1="310" y1="48" x2="420" y2="110" stroke="#1565c0" stroke-width="2"/>
  <line x1="310" y1="118" x2="262" y2="180" stroke="#1565c0" stroke-width="2"/>
  <line x1="310" y1="118" x2="360" y2="180" stroke="#1565c0" stroke-width="2"/>
  <line x1="262" y1="188" x2="262" y2="240" stroke="#1565c0" stroke-width="2"/>
  <circle cx="310" cy="40" r="12" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="310" y="45" text-anchor="middle" font-family="serif" font-size="12" fill="#0f9b6c">1</text>
  <circle cx="200" cy="118" r="11" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="200" y="123" text-anchor="middle" font-family="serif" font-size="12" fill="#1565c0">2</text>
  <circle cx="310" cy="118" r="11" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="310" y="123" text-anchor="middle" font-family="serif" font-size="12" fill="#1565c0">3</text>
  <circle cx="420" cy="118" r="11" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="420" y="123" text-anchor="middle" font-family="serif" font-size="12" fill="#1565c0">4</text>
  <circle cx="262" cy="188" r="11" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="262" y="193" text-anchor="middle" font-family="serif" font-size="12" fill="#1565c0">5</text>
  <circle cx="360" cy="188" r="11" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="360" y="193" text-anchor="middle" font-family="serif" font-size="12" fill="#1565c0">6</text>
  <circle cx="262" cy="250" r="11" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="262" y="255" text-anchor="middle" font-family="serif" font-size="12" fill="#1565c0">7</text>
  <text x="310" y="292" text-anchor="middle" font-family="serif" font-size="14" fill="#333">1 2 1 3 5 7 5 3 6 3 1 4 1</text>
  <text x="310" y="308" text-anchor="middle" font-family="serif" font-size="11" fill="#666">length 2n − 1</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
DFS from the root writes down every visit: the ET sequence of this tree is $1\,2\,1\,3\,5\,7\,5\,3\,6\,3\,1\,4\,1$.
</figcaption>
</figure>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 560 190" width="100%" style="max-width: 440px; height: auto;" role="img" aria-labelledby="gaf74-title">
  <title id="gaf74-title">Doubling the edges of a tree makes it Eulerian; the ET sequence is that tour</title>
  <defs>
    <marker id="gaf74-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#0f9b6c"/>
    </marker>
  </defs>
  <path d="M 138,102 C 180,72 240,60 268,74" fill="none" stroke="#0f9b6c" stroke-width="1.8" marker-end="url(#gaf74-arr)"/>
  <path d="M 266,92 C 236,104 186,116 144,116" fill="none" stroke="#0f9b6c" stroke-width="1.8" marker-end="url(#gaf74-arr)"/>
  <path d="M 300,74 C 330,58 392,66 424,92" fill="none" stroke="#0f9b6c" stroke-width="1.8" marker-end="url(#gaf74-arr)"/>
  <path d="M 420,108 C 388,120 328,110 298,94" fill="none" stroke="#0f9b6c" stroke-width="1.8" marker-end="url(#gaf74-arr)"/>
  <circle cx="128" cy="110" r="10" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="283" cy="84" r="10" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="435" cy="100" r="10" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="280" y="170" text-anchor="middle" font-family="serif" font-size="11" fill="#666">every edge doubled — the tree becomes Eulerian, and the ET sequence is its Euler tour</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Equivalently: double every edge and walk the Euler tour — hence the name.
</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Tree Operations as Sequence Surgery)</span></p>

Write the ET sequence schematically with capital letters for subsequences.

1. **Cut** the edge $xy$ ($x$ the parent of $y$): the sequence has the shape $uAx\,yBy\,xCu$ — delete the segment $yBy$ and one copy of $x$:

   $$
   uAxyByxCu \;\longrightarrow\; uAxCu \;+\; yBy.
   $$

2. **Re-root** at $v$: rotate the sequence, $uAvBu \longrightarrow vBuAv$ (drop the leading $u$, close the tour at $v$).
3. **Link** two trees by a new edge joining their roots $u$ and $v$: $uAu,\ vBv \longrightarrow uAu\,vBv\,u$.

Each operation is $O(1)$ **cuttings and pastings of sequences**, plus adding or removing a few elements.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 280" width="100%" style="max-width: 480px; height: auto;" role="img" aria-labelledby="gaf75-title">
  <title id="gaf75-title">Cutting the edge from x to its child y removes the segment between the two occurrences of y</title>
  <path d="M 210,58 C 200,80 222,96 210,118" fill="none" stroke="#1565c0" stroke-width="2"/>
  <line x1="210" y1="126" x2="210" y2="168" stroke="#1565c0" stroke-width="2"/>
  <line x1="210" y1="182" x2="210" y2="218" stroke="#b91c1c" stroke-width="2.6"/>
  <text x="248" y="204" text-anchor="middle" font-family="serif" font-size="11" fill="#b91c1c">cut xy</text>
  <path d="M 210,232 L 168,268 L 252,268 Z" fill="none" stroke="#1565c0" stroke-width="1.8"/>
  <text x="210" y="262" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#1565c0">B</text>
  <circle cx="210" cy="50" r="10" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="232" y="54" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#0f9b6c">u</text>
  <circle cx="210" cy="175" r="8" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="232" y="172" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">x</text>
  <circle cx="210" cy="225" r="8" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="234" y="230" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">y</text>
  <text x="440" y="120" text-anchor="middle" font-family="serif" font-size="13" fill="#333">u A x <tspan fill="#b91c1c">y B y</tspan> x C u</text>
  <text x="440" y="152" text-anchor="middle" font-family="serif" font-size="13" fill="#333">↓</text>
  <text x="440" y="184" text-anchor="middle" font-family="serif" font-size="13" fill="#333">u A x C u&nbsp;&nbsp;+&nbsp;&nbsp;<tspan fill="#b91c1c">y B y</tspan></text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The cut: $y$'s subtree occupies the contiguous segment $yBy$ between the two occurrences of $y$ — snip it out (and one spare $x$), and both trees have valid ET sequences.
</figcaption>
</figure>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 210" width="100%" style="max-width: 480px; height: auto;" role="img" aria-labelledby="gaf76-title">
  <title id="gaf76-title">Re-rooting rotates the ET sequence</title>
  <defs>
    <marker id="gaf76-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#333"/>
    </marker>
  </defs>
  <path d="M 160,62 C 150,88 172,104 160,132" fill="none" stroke="#1565c0" stroke-width="2"/>
  <circle cx="160" cy="54" r="10" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="184" y="58" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#0f9b6c">u</text>
  <circle cx="160" cy="140" r="10" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="184" y="145" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">v</text>
  <line x1="250" y1="100" x2="308" y2="100" stroke="#333" stroke-width="1.6" marker-end="url(#gaf76-arr)"/>
  <path d="M 420,62 C 410,88 432,104 420,132" fill="none" stroke="#1565c0" stroke-width="2"/>
  <circle cx="420" cy="54" r="10" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="444" y="58" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#0f9b6c">v</text>
  <circle cx="420" cy="140" r="10" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="444" y="145" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">u</text>
  <text x="310" y="192" text-anchor="middle" font-family="serif" font-size="13" fill="#333">uAvBu → vBuAv</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Re-rooting: the tour is cyclic up to the doubled endpoint, so moving the root to $v$ is one rotation of the sequence.
</figcaption>
</figure>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 620 220" width="100%" style="max-width: 480px; height: auto;" role="img" aria-labelledby="gaf77-title">
  <title id="gaf77-title">Linking two trees at their roots concatenates the ET sequences</title>
  <path d="M 180,70 L 120,160 L 240,160 Z" fill="none" stroke="#1565c0" stroke-width="1.8"/>
  <text x="180" y="140" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">A</text>
  <path d="M 440,70 L 380,160 L 500,160 Z" fill="none" stroke="#1565c0" stroke-width="1.8"/>
  <text x="440" y="140" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#1565c0">B</text>
  <line x1="192" y1="58" x2="428" y2="58" stroke="#0f9b6c" stroke-width="2.2" stroke-dasharray="7 5"/>
  <text x="310" y="44" text-anchor="middle" font-family="serif" font-size="11" fill="#0f9b6c">new edge uv</text>
  <circle cx="180" cy="62" r="10" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="158" y="56" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#0f9b6c">u</text>
  <circle cx="440" cy="62" r="10" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="464" y="56" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#0f9b6c">v</text>
  <text x="310" y="204" text-anchor="middle" font-family="serif" font-size="13" fill="#333">uAu, vBv → uAu vBv u</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Linking at the roots: concatenate the two tours and return to $u$ — combined with re-rooting, trees can be linked at arbitrary vertices.
</figcaption>
</figure>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(ET-Tree)</span></p>

Store the ET sequence in an **$(a,b)$-tree** with constant $a, b$: the **external nodes** are the elements of the sequence (the vertex occurrences), and the keys in **internal nodes** correspond to the gaps between neighboring occurrences — essentially to edges. All sequence operations — Insert, Delete, Cut (split), Join — run in $O(\log n)$.

Bookkeeping on the side: every vertex designates one **active occurrence** and keeps a pointer to its external node; every edge keeps pointers to its two occurrences (internal keys), paired with each other.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 250" width="100%" style="max-width: 580px; height: auto;" role="img" aria-labelledby="gaf78-title">
  <title id="gaf78-title">The ET sequence stored in the external nodes of an (a,b)-tree</title>
  <line x1="320" y1="58" x2="180" y2="112" stroke="#333" stroke-width="1.4"/>
  <line x1="320" y1="58" x2="320" y2="112" stroke="#333" stroke-width="1.4"/>
  <line x1="320" y1="58" x2="460" y2="112" stroke="#333" stroke-width="1.4"/>
  <line x1="180" y1="128" x2="120" y2="176" stroke="#333" stroke-width="1.2"/>
  <line x1="180" y1="128" x2="180" y2="176" stroke="#333" stroke-width="1.2"/>
  <line x1="180" y1="128" x2="240" y2="176" stroke="#333" stroke-width="1.2"/>
  <line x1="320" y1="128" x2="290" y2="176" stroke="#333" stroke-width="1.2"/>
  <line x1="320" y1="128" x2="350" y2="176" stroke="#333" stroke-width="1.2"/>
  <line x1="460" y1="128" x2="410" y2="176" stroke="#333" stroke-width="1.2"/>
  <line x1="460" y1="128" x2="470" y2="176" stroke="#333" stroke-width="1.2"/>
  <line x1="460" y1="128" x2="530" y2="176" stroke="#333" stroke-width="1.2"/>
  <rect x="296" y="42" width="48" height="20" rx="4" fill="#fff" stroke="#333" stroke-width="1.5"/>
  <rect x="156" y="112" width="48" height="20" rx="4" fill="#fff" stroke="#333" stroke-width="1.5"/>
  <rect x="296" y="112" width="48" height="20" rx="4" fill="#fff" stroke="#333" stroke-width="1.5"/>
  <rect x="436" y="112" width="48" height="20" rx="4" fill="#fff" stroke="#333" stroke-width="1.5"/>
  <circle cx="120" cy="188" r="11" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.5"/>
  <text x="120" y="193" text-anchor="middle" font-family="serif" font-size="11" fill="#1565c0">1</text>
  <circle cx="180" cy="188" r="11" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.5"/>
  <text x="180" y="193" text-anchor="middle" font-family="serif" font-size="11" fill="#1565c0">2</text>
  <circle cx="240" cy="188" r="11" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.5"/>
  <text x="240" y="193" text-anchor="middle" font-family="serif" font-size="11" fill="#1565c0">1</text>
  <circle cx="290" cy="188" r="11" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.5"/>
  <text x="290" y="193" text-anchor="middle" font-family="serif" font-size="11" fill="#1565c0">3</text>
  <circle cx="350" cy="188" r="11" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.5"/>
  <text x="350" y="193" text-anchor="middle" font-family="serif" font-size="11" fill="#1565c0">5</text>
  <circle cx="410" cy="188" r="11" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.5"/>
  <text x="410" y="193" text-anchor="middle" font-family="serif" font-size="11" fill="#1565c0">7</text>
  <circle cx="470" cy="188" r="11" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.5"/>
  <text x="470" y="193" text-anchor="middle" font-family="serif" font-size="11" fill="#1565c0">5</text>
  <circle cx="530" cy="188" r="11" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.5"/>
  <text x="530" y="193" text-anchor="middle" font-family="serif" font-size="11" fill="#1565c0">3</text>
  <line x1="265" y1="164" x2="265" y2="212" stroke="#b91c1c" stroke-width="1.8" stroke-dasharray="5 4"/>
  <text x="265" y="230" text-anchor="middle" font-family="serif" font-size="11" fill="#b91c1c">Cut = split here, in O(log n)</text>
  <text x="320" y="248" text-anchor="middle" font-family="serif" font-size="11" fill="#666">external nodes = the sequence; internal keys sit in the gaps ≈ edges</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The ET-tree: an $(a,b)$-tree whose leaves spell the ET sequence. Sequence cut-and-paste becomes $(a,b)$-tree split and join — $O(\log n)$ each.
</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(ET-Trees Handle Forests)</span></p>

ET-trees support **Cut, Link, Re-root, and FindRoot** in $O(\log n)$ time each — and FindRoot (walk to the leftmost external node) answers connectivity for forests:

$$
\text{connected}(u, v) \quad\iff\quad \mathrm{FindRoot}(u) = \mathrm{FindRoot}(v).
$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Beware)</span></p>

Two different trees coexist here: the **original tree** of the forest and the **ET-tree** (the $(a,b)$-tree over its tour). Their shapes are unrelated — the ET-tree is balanced no matter how path-like the original tree is.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Augmenting ET-Trees)</span></p>

ET-trees happily carry extra information attached to vertices. Example — **coloring vertices black/white**: store the color in the active occurrence (an external node) and let every internal node count the black external nodes in its subtree. Then in $O(\log n)$ we can change a vertex's color or count the black vertices of a tree, and enumerate them at $O(\log n)$ per vertex reported.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 560 200" width="100%" style="max-width: 400px; height: auto;" role="img" aria-labelledby="gaf79-title">
  <title id="gaf79-title">A tree with black and white vertices, counted in the ET-tree</title>
  <path d="M 130,110 C 126,52 220,24 330,30 C 430,36 470,80 466,118 C 462,164 380,184 290,180 C 200,176 134,158 130,110 Z" fill="none" stroke="#1565c0" stroke-width="1.6"/>
  <line x1="220" y1="90" x2="290" y2="120" stroke="#0f9b6c" stroke-width="1.8"/>
  <line x1="290" y1="120" x2="360" y2="80" stroke="#0f9b6c" stroke-width="1.8"/>
  <line x1="290" y1="120" x2="330" y2="155" stroke="#0f9b6c" stroke-width="1.8"/>
  <circle cx="220" cy="90" r="8" fill="#333"/>
  <circle cx="290" cy="120" r="8" fill="#fff" stroke="#333" stroke-width="1.6"/>
  <circle cx="360" cy="80" r="8" fill="#333"/>
  <circle cx="330" cy="155" r="8" fill="#fff" stroke="#333" stroke-width="1.6"/>
  <text x="298" y="196" text-anchor="middle" font-family="serif" font-size="11" fill="#666">colors live in active occurrences; internal nodes count black descendants</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The black/white augmentation — recolor, count, and enumerate marked vertices per tree in logarithmic time. Exactly this machinery powers the general structure below.
</figcaption>
</figure>

## General Graphs: Holm–de Lichtenberg–Thorup

For general graphs (Holm, de Lichtenberg, Thorup 2001) we maintain a **spanning forest** $F$ in ET-trees. Inserting an edge is easy: it either links two trees of $F$ or is stored as a **non-tree edge** — non-tree edges hang off the external nodes of their endpoints, so inserting, deleting, and enumerating them costs $O(\log n)$ each. Deleting a *tree* edge is the hard case: the structure must find a **replacement edge** reconnecting the two halves, or certify that none exists.

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 250" width="100%" style="max-width: 580px; height: auto;" role="img" aria-labelledby="gaf80-title">
  <title id="gaf80-title">Deleting a spanning-forest edge triggers a search for a replacement among non-tree edges</title>
  <line x1="130" y1="90" x2="200" y2="140" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="200" y1="140" x2="150" y2="190" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="200" y1="140" x2="290" y2="100" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="290" y1="100" x2="390" y2="120" stroke="#b91c1c" stroke-width="2.6"/>
  <line x1="358" y1="94" x2="322" y2="126" stroke="#b91c1c" stroke-width="2"/>
  <line x1="358" y1="126" x2="322" y2="94" stroke="#b91c1c" stroke-width="2"/>
  <line x1="390" y1="120" x2="470" y2="80" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="390" y1="120" x2="460" y2="180" stroke="#1565c0" stroke-width="2.2"/>
  <path d="M 150,190 C 260,240 380,230 460,185 " fill="none" stroke="#0f9b6c" stroke-width="2" stroke-dasharray="7 5"/>
  <text x="310" y="248" text-anchor="middle" font-family="serif" font-size="11" fill="#0f9b6c">non-tree edge — a replacement?</text>
  <path d="M 130,80 C 170,50 260,54 290,88" fill="none" stroke="#666" stroke-width="1.3" stroke-dasharray="4 4"/>
  <circle cx="130" cy="90" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="200" cy="140" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="150" cy="190" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="290" cy="100" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="390" cy="120" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="470" cy="80" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="460" cy="180" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="340" y="66" text-anchor="middle" font-family="serif" font-size="11" fill="#b91c1c">tree edge deleted</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The spanning forest (blue) answers queries; when one of its edges dies, some non-tree edge (dashed) crossing the split — if any — must be promoted.
</figcaption>
</figure>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Edge Levels)</span></p>

Every edge carries a **level** $\ell(e) \in \lbrace 0, \dots, L \rbrace$ with $L := \lfloor \log n \rfloor$, and

$$
F_i \;:=\; \text{the subforest of } F \text{ with edges of level} \ge i,
\qquad
F = F_0 \supseteq F_1 \supseteq \dots \supseteq F_L,
$$

each $F\_i$ kept in its own ET-trees (level-$i$ non-tree edges stored in the ET-trees of $F\_i$). Two **invariants**:

* **I1:** $F$ is a *maximum* spanning forest with respect to levels — equivalently, the endpoints of every non-tree edge of level $\ell$ are connected in $F\_\ell$;
* **I2:** every component of $F\_i$ has at most $\lfloor n / 2^i \rfloor$ vertices (this is why $L = \lfloor \log n \rfloor$ levels suffice).

Initially there are no edges, so both hold; **new edges get level $0$**, which preserves both.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 560 210" width="100%" style="max-width: 440px; height: auto;" role="img" aria-labelledby="gaf81-title">
  <title id="gaf81-title">Invariant one: endpoints of a level-l non-tree edge are connected at level l</title>
  <line x1="120" y1="150" x2="180" y2="80" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="180" y1="80" x2="290" y2="60" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="290" y1="60" x2="390" y2="90" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="390" y1="90" x2="440" y2="150" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="120" y1="158" x2="434" y2="158" stroke="#0f9b6c" stroke-width="2" stroke-dasharray="7 5"/>
  <text x="278" y="146" text-anchor="middle" font-family="serif" font-size="11" fill="#0f9b6c">non-tree edge uv, level ℓ</text>
  <text x="278" y="40" text-anchor="middle" font-family="serif" font-size="11" fill="#1565c0">a path in Fℓ — tree edges of level ≥ ℓ</text>
  <circle cx="120" cy="152" r="10" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="100" y="146" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#0f9b6c">u</text>
  <circle cx="180" cy="80" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="290" cy="60" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="390" cy="90" r="7" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="440" cy="152" r="10" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="2"/>
  <text x="462" y="146" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#0f9b6c">v</text>
  <text x="280" y="198" text-anchor="middle" font-family="serif" font-size="11" fill="#666">I1: the higher an edge's level, the deeper the forest that already connects its endpoints</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
Invariant I1 pictured: a level-$\ell$ non-tree edge is always "shadowed" by a tree path of level $\ge \ell$.
</figcaption>
</figure>

## Deleting a Tree Edge

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Observation</span><span class="math-callout__name">(Where Replacements Hide)</span></p>

Let the deleted tree edge $e = uv$ have level $\ell$; it disappears from $F\_0, \dots, F\_\ell$ (higher $F\_i$ never contained it). Any replacement edge must cross between the two halves — but by I1, a non-tree edge of level $\ell' > \ell$ has both endpoints connected in $F\_{\ell'} \subseteq F\_{\ell+1}$, which survived the deletion intact: both its endpoints are on the *same* side. So **replacement candidates have level $\le \ell$** — the search proceeds at level $\ell$, then $\ell - 1$, down to $0$.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 560 190" width="100%" style="max-width: 440px; height: auto;" role="img" aria-labelledby="gaf82-title">
  <title id="gaf82-title">Deleting e splits its level-l component into the trees T1 and T2</title>
  <ellipse cx="160" cy="100" rx="90" ry="62" fill="none" stroke="#1565c0" stroke-width="1.8"/>
  <text x="110" y="56" font-family="serif" font-size="13" font-style="italic" fill="#1565c0">T₁</text>
  <ellipse cx="420" cy="100" rx="90" ry="62" fill="none" stroke="#1565c0" stroke-width="1.8"/>
  <text x="474" y="56" font-family="serif" font-size="13" font-style="italic" fill="#1565c0">T₂</text>
  <line x1="252" y1="100" x2="328" y2="100" stroke="#b91c1c" stroke-width="2.6"/>
  <line x1="278" y1="86" x2="302" y2="114" stroke="#b91c1c" stroke-width="2"/>
  <line x1="278" y1="114" x2="302" y2="86" stroke="#b91c1c" stroke-width="2"/>
  <text x="290" y="72" text-anchor="middle" font-family="serif" font-size="12" font-style="italic" fill="#b91c1c">e — level ℓ</text>
  <circle cx="252" cy="100" r="8" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="1.8"/>
  <text x="240" y="122" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#0f9b6c">u</text>
  <circle cx="328" cy="100" r="8" fill="#ecfdf5" stroke="#0f9b6c" stroke-width="1.8"/>
  <text x="342" y="122" text-anchor="middle" font-family="serif" font-size="11" font-style="italic" fill="#0f9b6c">v</text>
  <text x="280" y="182" text-anchor="middle" font-family="serif" font-size="11" fill="#666">T₁, T₂ — the trees of Fℓ around u and v after the deletion; WLOG |T₁| ≤ |T₂|</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
The stage at level $\ell$: the deleted edge leaves two trees of $F_\ell$, and the search works on the smaller one.
</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Algorithm</span><span class="math-callout__name">(Delete a Tree Edge $e = uv$ of Level $\ell$)</span></p>

For $j = \ell$ down to $0$ — the **search at level $j$**:

1. Let $T\_1, T\_2$ be the trees of $F\_j$ containing $u$ and $v$; WLOG $\lvert T\_1 \rvert \le \lvert T\_2 \rvert$ (sizes from the ET-trees, $O(\log n)$).
2. **Raise all level-$j$ tree edges inside $T\_1$ to level $j+1$** ($O(\log n)$ per raised edge).
3. Enumerate the level-$j$ non-tree edges incident to $T\_1$, one at a time:
   * both endpoints in $T\_1$ — raise the edge's level to $j + 1$ and keep looking;
   * the other endpoint in $T\_2$ — **success**: insert it as a tree edge of $F\_j, \dots, F\_0$ (and remove it from the non-tree structures), $O(\log n)$ per level; stop everything.

If even level $0$ produces nothing, $e$ was a **bridge** — the component genuinely splits.

</div>

<figure style="margin: 1.5em auto; text-align: center;">
<svg viewBox="0 0 640 270" width="100%" style="max-width: 600px; height: auto;" role="img" aria-labelledby="gaf83-title">
  <title id="gaf83-title">The search at level l: raise T1's edges, then test its non-tree edges one by one</title>
  <ellipse cx="180" cy="130" rx="105" ry="75" fill="none" stroke="#1565c0" stroke-width="1.8"/>
  <text x="112" y="72" font-family="serif" font-size="13" font-style="italic" fill="#1565c0">T₁</text>
  <ellipse cx="480" cy="130" rx="105" ry="75" fill="none" stroke="#1565c0" stroke-width="1.8"/>
  <text x="548" y="72" font-family="serif" font-size="13" font-style="italic" fill="#1565c0">T₂</text>
  <line x1="140" y1="110" x2="200" y2="90" stroke="#0f9b6c" stroke-width="2.4"/>
  <line x1="140" y1="110" x2="170" y2="160" stroke="#0f9b6c" stroke-width="2.4"/>
  <text x="118" y="222" text-anchor="middle" font-family="serif" font-size="10" fill="#0f9b6c">tree edges → level ℓ+1</text>
  <path d="M 205,95 C 250,60 240,130 214,142" fill="none" stroke="#a86f00" stroke-width="1.8" stroke-dasharray="6 4"/>
  <text x="302" y="48" text-anchor="middle" font-family="serif" font-size="10" fill="#a86f00">stays in T₁ → level ℓ+1</text>
  <path d="M 175,165 C 290,220 400,200 452,165" fill="none" stroke="#c2185b" stroke-width="2" stroke-dasharray="7 5"/>
  <text x="320" y="236" text-anchor="middle" font-family="serif" font-size="11" fill="#c2185b">reaches T₂ — success: the replacement, promoted to a tree edge</text>
  <line x1="440" y1="110" x2="500" y2="95" stroke="#1565c0" stroke-width="2.2"/>
  <line x1="500" y1="95" x2="520" y2="150" stroke="#1565c0" stroke-width="2.2"/>
  <circle cx="140" cy="110" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="200" cy="90" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="170" cy="160" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="214" cy="142" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="440" cy="110" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="500" cy="95" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <circle cx="520" cy="150" r="6" fill="#e3f2fd" stroke="#1565c0" stroke-width="1.6"/>
  <text x="320" y="262" text-anchor="middle" font-family="serif" font-size="11" fill="#666">every failed candidate pays with a level increase — that is the whole amortization</text>
</svg>
<figcaption markdown="1" style="font-style: italic; font-size: 0.9em; margin-top: 0.4em; color: #555;">
One level of the search: $T_1$'s level-$\ell$ tree edges are pushed up, then its level-$\ell$ non-tree edges are tested — failures get pushed up too, the first success reconnects the forest.
</figcaption>
</figure>

<div class="math-callout math-callout--lemma" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Lemma</span><span class="math-callout__name">(The Invariants Survive)</span></p>

The level increases in the search are legal. **I2:** before the deletion, $T\_1 \cup \lbrace e \rbrace \cup T\_2$ was one component of $F\_j$, of size $\le n/2^j$; since $\lvert T\_1 \rvert \le \lvert T\_2 \rvert$, we get $\lvert T\_1 \rvert \le n/2^{j+1}$ — so all of $T\_1$ may live at level $j+1$. **I1:** raising the level of tree edges never hurts (they are in $F$), and a non-tree edge with both endpoints in $T\_1$ has them connected by $T\_1$'s tree path — which was just raised to level $j+1$ as well.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(Holm–de Lichtenberg–Thorup 2001)</span></p>

Fully dynamic graph connectivity with **amortized $O(\log^2 n)$ per edge insertion or deletion** and $O(\log n)$ per connectivity query.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (amortization)</summary>

Levels only ever increase, and by I2 they cannot exceed $L = \lfloor \log n \rfloor$; one increase costs $O(\log n)$ in ET-tree updates. So charge every inserted edge a deposit of $O(\log^2 n)$ for its lifetime of level increases.

Deleting a tree edge then costs, beyond the deposits: $O(\log n)$ per level for the $T\_1/T\_2$ bookkeeping — $O(\log^2 n)$ over all levels; every enumerated non-tree candidate either *fails* (and pays with its own level increase, already deposited) or *succeeds* (once, ending the search, $O(\log^2 n)$ for the promotion). Total: amortized $O(\log^2 n)$ per update. Queries are two FindRoots in the ET-trees of $F\_0$: $O(\log n)$. ∎

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What the ET-Trees Must Support)</span></p>

The search leans on three augmentations, all $O(\log n)$ per operation or per edge enumerated:

1. **counting external nodes** — the component sizes for the $\lvert T\_1 \rvert \le \lvert T\_2 \rvert$ test;
2. **finding level-$j$ tree edges** — black/white-style marks on the internal keys of $F\_j$'s ET-trees;
3. **finding level-$j$ non-tree edges** — each ET-tree of $F\_j$ remembers exactly the level-$j$ non-tree edges hanging off its vertices.

</div>
