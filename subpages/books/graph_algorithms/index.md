---
layout: default
title: Graph Algorithms
date: 2026-07-07
excerpt: Lecture notes on graph algorithms covering network flows, the Ford–Fulkerson method, Dinitz's algorithm, capacity scaling, Menger's theorems, the Karger–Stein randomized min-cut, shortest paths, and applications such as bipartite matching.
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
