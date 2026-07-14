---
title: Algorithmic Randomness and Computable Analysis
layout: default
noindex: true
tags:
  - computer-science
  - computability
  - algorithmic-randomness
  - computable-analysis
  - kolmogorov-complexity
  - measure-theory
  - information-content-measure
  - martin-löf-test
  - cantor-space
  - turing-machine
  - prefix-free-turing-machine
  - compressibility
  - minimum-description-length
  - halting-problem
  - chaitin-constant
  - solovay-reducibility
---

# Algorithmic Randomness and Computable Analysis

*Lecture notes by Ivan Titov, Summer 2026.*

**Table of Contents**
- TOC
{:toc}

## Notation

Before we can talk about randomness in any precise sense, we need a vocabulary for the objects on which the theory operates: finite binary words, infinite binary sequences, and the basic open sets of Cantor space that allow us to attach Lebesgue measure to subsets of these sequences.

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Words and Sequences)</span></p>

A **(binary) word** $w \in \lbrace 0, 1 \rbrace^{\ast}$ is a finite sequence of bits. The length of a word $w$ is denoted $l(w)$. The binary representation of a natural number $n$ is the binary word $\mathrm{bin}(n)$; its length satisfies 

$$l(\mathrm{bin}(n)) = \lfloor \log_2 n \rfloor + 1 = \log(n) + O(1)$$

For an infinite sequence $A = a_0 a_1 a_2 \dots$, the **prefix** of $A$ of length $n$ is

$$A \upharpoonright n := a_0 \dots a_{n-1}.$$

We use 

$$w_0 := \lambda,\ w_1 := 0,\ w_2 := 1,\ w_3 := 00,\ w_4 := 01,\ w_5 := 10,\ w_6 := 11,\ w_7 := 000,\ \dots$$

to denote, by default, the **length-lexicographical enumeration** of all binary words.

</div>

<figure class="math-figure">
  <svg viewBox="0 0 560 280" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:560px" aria-label="Length-lexicographic enumeration as a binary tree">
    <g font-family="serif" font-size="13" fill="#1f2430">
      <!-- Levels guide -->
      <g stroke="#cbd2e0" stroke-dasharray="2 4" stroke-width="0.5">
        <line x1="20" y1="40"  x2="540" y2="40"  />
        <line x1="20" y1="100" x2="540" y2="100" />
        <line x1="20" y1="170" x2="540" y2="170" />
        <line x1="20" y1="240" x2="540" y2="240" />
      </g>
      <text x="540" y="44"  text-anchor="end" font-size="10" fill="#5b6270">length 0</text>
      <text x="540" y="104" text-anchor="end" font-size="10" fill="#5b6270">length 1</text>
      <text x="540" y="174" text-anchor="end" font-size="10" fill="#5b6270">length 2</text>
      <text x="540" y="244" text-anchor="end" font-size="10" fill="#5b6270">length 3</text>

      <!-- Tree edges -->
      <g stroke="#5b6270" stroke-width="1" fill="none">
        <line x1="280" y1="48"  x2="170" y2="92"  />
        <line x1="280" y1="48"  x2="390" y2="92"  />
        <line x1="170" y1="108" x2="100" y2="162" />
        <line x1="170" y1="108" x2="240" y2="162" />
        <line x1="390" y1="108" x2="320" y2="162" />
        <line x1="390" y1="108" x2="460" y2="162" />
        <line x1="100" y1="178" x2="60"  y2="232" />
        <line x1="100" y1="178" x2="140" y2="232" />
        <line x1="240" y1="178" x2="210" y2="232" />
        <line x1="240" y1="178" x2="270" y2="232" />
        <line x1="320" y1="178" x2="290" y2="232" />
        <line x1="320" y1="178" x2="350" y2="232" />
        <line x1="460" y1="178" x2="420" y2="232" />
        <line x1="460" y1="178" x2="500" y2="232" />
      </g>

      <!-- Root -->
      <g>
        <circle cx="280" cy="40" r="14" fill="#fff7e0" stroke="#a86f00" />
        <text x="280" y="44" text-anchor="middle" font-style="italic">λ</text>
        <text x="280" y="22" text-anchor="middle" font-size="11" fill="#a86f00">w₀</text>
      </g>

      <!-- Length 1 -->
      <g>
        <circle cx="170" cy="100" r="14" fill="#fff7e0" stroke="#a86f00" />
        <text x="170" y="104" text-anchor="middle">0</text>
        <text x="170" y="82" text-anchor="middle" font-size="11" fill="#a86f00">w₁</text>
        <circle cx="390" cy="100" r="14" fill="#fff7e0" stroke="#a86f00" />
        <text x="390" y="104" text-anchor="middle">1</text>
        <text x="390" y="82" text-anchor="middle" font-size="11" fill="#a86f00">w₂</text>
      </g>

      <!-- Length 2 -->
      <g>
        <circle cx="100" cy="170" r="14" fill="#fff7e0" stroke="#a86f00" />
        <text x="100" y="174" text-anchor="middle">00</text>
        <text x="100" y="152" text-anchor="middle" font-size="11" fill="#a86f00">w₃</text>
        <circle cx="240" cy="170" r="14" fill="#fff7e0" stroke="#a86f00" />
        <text x="240" y="174" text-anchor="middle">01</text>
        <text x="240" y="152" text-anchor="middle" font-size="11" fill="#a86f00">w₄</text>
        <circle cx="320" cy="170" r="14" fill="#fff7e0" stroke="#a86f00" />
        <text x="320" y="174" text-anchor="middle">10</text>
        <text x="320" y="152" text-anchor="middle" font-size="11" fill="#a86f00">w₅</text>
        <circle cx="460" cy="170" r="14" fill="#fff7e0" stroke="#a86f00" />
        <text x="460" y="174" text-anchor="middle">11</text>
        <text x="460" y="152" text-anchor="middle" font-size="11" fill="#a86f00">w₆</text>
      </g>

      <!-- Length 3 (just first label and ellipsis) -->
      <g>
        <circle cx="60"  cy="240" r="13" fill="#fff7e0" stroke="#a86f00" />
        <text x="60"  y="244" text-anchor="middle" font-size="11">000</text>
        <text x="60"  y="222" text-anchor="middle" font-size="11" fill="#a86f00">w₇</text>
        <circle cx="140" cy="240" r="13" fill="#fff7e0" stroke="#a86f00" />
        <text x="140" y="244" text-anchor="middle" font-size="11">001</text>
        <circle cx="210" cy="240" r="13" fill="#fff7e0" stroke="#a86f00" />
        <text x="210" y="244" text-anchor="middle" font-size="11">010</text>
        <circle cx="270" cy="240" r="13" fill="#fff7e0" stroke="#a86f00" />
        <text x="270" y="244" text-anchor="middle" font-size="11">011</text>
        <circle cx="290" cy="240" r="13" fill="#fff7e0" stroke="#a86f00" />
        <text x="290" y="244" text-anchor="middle" font-size="11">100</text>
        <circle cx="350" cy="240" r="13" fill="#fff7e0" stroke="#a86f00" />
        <text x="350" y="244" text-anchor="middle" font-size="11">101</text>
        <circle cx="420" cy="240" r="13" fill="#fff7e0" stroke="#a86f00" />
        <text x="420" y="244" text-anchor="middle" font-size="11">110</text>
        <circle cx="500" cy="240" r="13" fill="#fff7e0" stroke="#a86f00" />
        <text x="500" y="244" text-anchor="middle" font-size="11">111</text>
      </g>
    </g>
  </svg>
  <figcaption>The length-lexicographic enumeration $w_0, w_1, w_2, \dots$ of all binary words. Words are first ordered by length, and within a fixed length they follow the lexicographic order $0 < 1$. There are $2^k$ words of length $k$, and the first index of length-$k$ words is $2^k - 1$.</figcaption>
</figure>

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Notation</span><span class="math-callout__name">(Cantor Space, Basic Open Sets, Lebesgue Measure)</span></p>

The **Cantor space** is the set $\lbrace 0, 1 \rbrace^{\omega}$ of infinite binary sequences. Its **basic open sets** are

$$ [\![\sigma]\!] = \lbrace \sigma X : X \in \lbrace 0, 1 \rbrace^{\omega} \rbrace, \qquad \sigma \in \lbrace 0, 1 \rbrace^{\ast}.$$

These correspond to the dyadic intervals $[0.\sigma,\ 0.\sigma + 2^{-l(\sigma)}]$ of length $2^{-l(\sigma)}$ in $\mathbb{R} \big\vert _{[0, 1]}$. The **Lebesgue measure** of a basic open set is

$$\lambda([\![\sigma]\!]) = 2^{-l(\sigma)}.$$

In Cantor space, **open sets** are (finite or countably infinite) unions of pairwise disjoint basic open sets $[[\sigma_i]]$:

$$S = ([\![\sigma_0]\!], [\![\sigma_1]\!], \dots), \qquad \lambda(S) = \sum_i \lambda([\![\sigma_i]\!]) = \sum_i 2^{-l(\sigma_i)}.$$

</div>

<figure class="math-figure">
  <svg viewBox="0 0 620 270" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:620px" aria-label="Basic open sets of Cantor space as dyadic intervals on [0,1]">
    <g font-family="serif" font-size="12" fill="#1f2430">
      <!-- Layer guide: full interval -->
      <g>
        <rect x="40" y="30" width="540" height="28" fill="rgba(168,111,0,0.10)" stroke="#a86f00" stroke-width="1.5" />
        <text x="310" y="48" text-anchor="middle" font-style="italic" font-size="13">⟦λ⟧ = [0, 1],   λ(⟦λ⟧) = 1</text>
      </g>

      <!-- Length 1 split -->
      <g>
        <rect x="40"  y="70" width="270" height="28" fill="rgba(60,120,40,0.10)" stroke="#3d7a26" stroke-width="1.5" />
        <text x="175" y="88" text-anchor="middle">⟦0⟧ = [0, 1/2],   measure 1/2</text>
        <rect x="310" y="70" width="270" height="28" fill="rgba(60,120,40,0.10)" stroke="#3d7a26" stroke-width="1.5" />
        <text x="445" y="88" text-anchor="middle">⟦1⟧ = [1/2, 1],   measure 1/2</text>
      </g>

      <!-- Length 2 split -->
      <g>
        <rect x="40"  y="110" width="135" height="28" fill="rgba(44,73,148,0.10)" stroke="#2c4994" stroke-width="1.5" />
        <text x="107" y="128" text-anchor="middle" font-size="11">⟦00⟧, m=1/4</text>
        <rect x="175" y="110" width="135" height="28" fill="rgba(44,73,148,0.10)" stroke="#2c4994" stroke-width="1.5" />
        <text x="242" y="128" text-anchor="middle" font-size="11">⟦01⟧, m=1/4</text>
        <rect x="310" y="110" width="135" height="28" fill="rgba(44,73,148,0.10)" stroke="#2c4994" stroke-width="1.5" />
        <text x="377" y="128" text-anchor="middle" font-size="11">⟦10⟧, m=1/4</text>
        <rect x="445" y="110" width="135" height="28" fill="rgba(44,73,148,0.10)" stroke="#2c4994" stroke-width="1.5" />
        <text x="512" y="128" text-anchor="middle" font-size="11">⟦11⟧, m=1/4</text>
      </g>

      <!-- Length 3 split -->
      <g stroke="#d65336" stroke-width="1.5" fill="rgba(214,83,54,0.10)">
        <rect x="40"  y="150" width="67.5" height="22" />
        <rect x="107.5" y="150" width="67.5" height="22" />
        <rect x="175" y="150" width="67.5" height="22" />
        <rect x="242.5" y="150" width="67.5" height="22" />
        <rect x="310" y="150" width="67.5" height="22" />
        <rect x="377.5" y="150" width="67.5" height="22" />
        <rect x="445" y="150" width="67.5" height="22" />
        <rect x="512.5" y="150" width="67.5" height="22" />
      </g>
      <g font-size="10" text-anchor="middle">
        <text x="73"  y="165">⟦000⟧</text>
        <text x="141" y="165">⟦001⟧</text>
        <text x="208" y="165">⟦010⟧</text>
        <text x="276" y="165">⟦011⟧</text>
        <text x="343" y="165">⟦100⟧</text>
        <text x="411" y="165">⟦101⟧</text>
        <text x="478" y="165">⟦110⟧</text>
        <text x="546" y="165">⟦111⟧</text>
      </g>

      <!-- Number line at bottom -->
      <g>
        <line x1="40" y1="200" x2="580" y2="200" stroke="#444" stroke-width="1.2" />
        <g stroke="#444" stroke-width="1">
          <line x1="40"  y1="196" x2="40"  y2="208" />
          <line x1="175" y1="196" x2="175" y2="208" />
          <line x1="310" y1="196" x2="310" y2="208" />
          <line x1="445" y1="196" x2="445" y2="208" />
          <line x1="580" y1="196" x2="580" y2="208" />
        </g>
        <g font-size="11" fill="#444" text-anchor="middle">
          <text x="40"  y="222">0</text>
          <text x="175" y="222">1/4</text>
          <text x="310" y="222">1/2</text>
          <text x="445" y="222">3/4</text>
          <text x="580" y="222">1</text>
        </g>
      </g>

      <!-- Annotation -->
      <text x="310" y="252" text-anchor="middle" font-size="11" fill="#5b6270" font-style="italic">
        Each refinement halves the measure: l(σ) = k ⟹ λ(⟦σ⟧) = 2⁻ᵏ.
      </text>
    </g>
  </svg>
  <figcaption>Basic open sets of Cantor space identified with dyadic intervals of $[0, 1]$. A finite word $\sigma$ of length $k$ corresponds to the set $[\![\sigma]\!]$ of all infinite binary sequences extending $\sigma$, which in turn corresponds to the dyadic interval $[0.\sigma,\ 0.\sigma + 2^{-k}]$ of length $2^{-k}$. Each level halves the measure of its parent.</figcaption>
</figure>

## Introduction

### Three Intuitions of Nonrandomness

Before formalizing anything, it is useful to look at concrete sequences that *feel* nonrandom and ask: what, exactly, is wrong with them? Three intuitions emerge — compressibility, predictability, and untypicality — and each will lead to a different formal definition of randomness in the rest of the course.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Three intuitively nonrandom finite sequences of length 2000)</span></p>

* $X := (01)^{1000} = \underbrace{0101 \dots 01}_{2000 \text{ digits}}.$
* $Y := x_1 x_2 \dots x_{2000}$ such that $b_1 + b_2 + \dots + b_{2000} \le 200$ (i.e., at most 200 ones).
* $Z := z_1 0\, z_2 0 \dots z_{1000} 0$ where $z_1, \dots, z_{1000}$ are "random" bits.

</div>

<figure class="math-figure">
  <svg viewBox="0 0 620 260" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:620px" aria-label="Bit patterns of the three intuitively nonrandom sequences">
    <defs>
      <pattern id="hatch01" width="2" height="22" patternUnits="userSpaceOnUse">
        <rect width="1" height="22" fill="#eef0f4" />
        <rect x="1" width="1" height="22" fill="#2c4994" />
      </pattern>
    </defs>
    <g font-family="serif" font-size="12" fill="#1f2430">
      <!-- X: alternating 01 -->
      <text x="20" y="34" font-weight="600">X</text>
      <text x="20" y="50" font-size="10" fill="#5b6270">(01)¹⁰⁰⁰</text>
      <rect x="80" y="22" width="500" height="22" fill="url(#hatch01)" stroke="#5b6270" stroke-width="0.5" />
      <text x="586" y="38" font-size="10" fill="#5b6270">2000 bits</text>

      <!-- Y: sparse ones (≤200 out of 2000) -->
      <text x="20" y="94" font-weight="600">Y</text>
      <text x="20" y="110" font-size="10" fill="#5b6270">≤ 200 ones</text>
      <rect x="80" y="82" width="500" height="22" fill="#eef0f4" stroke="#5b6270" stroke-width="0.5" />
      <!-- A few 1s scattered -->
      <g fill="#d65336">
        <rect x="92"  y="82" width="2" height="22" />
        <rect x="118" y="82" width="2" height="22" />
        <rect x="151" y="82" width="2" height="22" />
        <rect x="174" y="82" width="2" height="22" />
        <rect x="207" y="82" width="2" height="22" />
        <rect x="241" y="82" width="2" height="22" />
        <rect x="280" y="82" width="2" height="22" />
        <rect x="305" y="82" width="2" height="22" />
        <rect x="342" y="82" width="2" height="22" />
        <rect x="379" y="82" width="2" height="22" />
        <rect x="412" y="82" width="2" height="22" />
        <rect x="438" y="82" width="2" height="22" />
        <rect x="471" y="82" width="2" height="22" />
        <rect x="498" y="82" width="2" height="22" />
        <rect x="524" y="82" width="2" height="22" />
        <rect x="554" y="82" width="2" height="22" />
      </g>
      <text x="586" y="98" font-size="10" fill="#5b6270">2000 bits</text>

      <!-- Z: random in odd, zero in even -->
      <text x="20" y="154" font-weight="600">Z</text>
      <text x="20" y="170" font-size="10" fill="#5b6270">z₁0 z₂0 …</text>
      <rect x="80" y="142" width="500" height="22" fill="#eef0f4" stroke="#5b6270" stroke-width="0.5" />
      <!-- Random in odd positions: random distribution among the odd-position columns -->
      <g fill="#3d7a26">
        <rect x="82"  y="142" width="2" height="22" />
        <rect x="86"  y="142" width="2" height="22" />
        <rect x="94"  y="142" width="2" height="22" />
        <rect x="102" y="142" width="2" height="22" />
        <rect x="106" y="142" width="2" height="22" />
        <rect x="118" y="142" width="2" height="22" />
        <rect x="126" y="142" width="2" height="22" />
        <rect x="138" y="142" width="2" height="22" />
        <rect x="146" y="142" width="2" height="22" />
        <rect x="158" y="142" width="2" height="22" />
        <rect x="166" y="142" width="2" height="22" />
        <rect x="174" y="142" width="2" height="22" />
        <rect x="186" y="142" width="2" height="22" />
        <rect x="190" y="142" width="2" height="22" />
        <rect x="206" y="142" width="2" height="22" />
        <rect x="218" y="142" width="2" height="22" />
        <rect x="226" y="142" width="2" height="22" />
        <rect x="238" y="142" width="2" height="22" />
        <rect x="246" y="142" width="2" height="22" />
        <rect x="262" y="142" width="2" height="22" />
        <rect x="270" y="142" width="2" height="22" />
        <rect x="282" y="142" width="2" height="22" />
        <rect x="294" y="142" width="2" height="22" />
        <rect x="302" y="142" width="2" height="22" />
        <rect x="318" y="142" width="2" height="22" />
        <rect x="326" y="142" width="2" height="22" />
        <rect x="338" y="142" width="2" height="22" />
        <rect x="346" y="142" width="2" height="22" />
        <rect x="354" y="142" width="2" height="22" />
        <rect x="370" y="142" width="2" height="22" />
        <rect x="378" y="142" width="2" height="22" />
        <rect x="386" y="142" width="2" height="22" />
        <rect x="394" y="142" width="2" height="22" />
        <rect x="406" y="142" width="2" height="22" />
        <rect x="418" y="142" width="2" height="22" />
        <rect x="430" y="142" width="2" height="22" />
        <rect x="442" y="142" width="2" height="22" />
        <rect x="450" y="142" width="2" height="22" />
        <rect x="462" y="142" width="2" height="22" />
        <rect x="478" y="142" width="2" height="22" />
        <rect x="486" y="142" width="2" height="22" />
        <rect x="498" y="142" width="2" height="22" />
        <rect x="510" y="142" width="2" height="22" />
        <rect x="518" y="142" width="2" height="22" />
        <rect x="534" y="142" width="2" height="22" />
        <rect x="546" y="142" width="2" height="22" />
        <rect x="558" y="142" width="2" height="22" />
        <rect x="574" y="142" width="2" height="22" />
      </g>
      <text x="586" y="158" font-size="10" fill="#5b6270">2000 bits</text>

      <!-- Legend / interpretation -->
      <g transform="translate(80,200)">
        <rect x="0"   y="0" width="14" height="14" fill="#eef0f4" stroke="#5b6270" stroke-width="0.5" />
        <text x="20" y="11" font-size="11">bit 0</text>
        <rect x="80"  y="0" width="14" height="14" fill="#2c4994" />
        <text x="100" y="11" font-size="11">predictable 1</text>
        <rect x="200" y="0" width="14" height="14" fill="#d65336" />
        <text x="220" y="11" font-size="11">rare 1 (sparse)</text>
        <rect x="340" y="0" width="14" height="14" fill="#3d7a26" />
        <text x="360" y="11" font-size="11">random 1 (every other position)</text>
      </g>

      <text x="310" y="246" text-anchor="middle" font-size="11" fill="#5b6270" font-style="italic">
        Each strip shows a 2000-bit sequence as a horizontal pixel row.
      </text>
    </g>
  </svg>
  <figcaption>Three intuitively nonrandom 2000-bit sequences. $X$ has a perfectly periodic pattern of alternating bits. $Y$ has a heavily skewed bit-density (most positions are zero, only $\le 200$ are one). $Z$ is random on the odd positions but deterministically zero on the even ones — half of the information is "wasted".</figcaption>
</figure>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Three intuitively nonrandom infinite binary sequences)</span></p>

* $\tilde{X} := (01)^{\omega} = 010101 \dots$
* $\tilde{Y} := x_1 x_2 \dots$ such that $b_1 + b_2 + \dots + b_n \le \dfrac{n}{5}$ for all $n \ge 5$.
* $\tilde{Z} := z_1 0\, z_2 0\, z_3 0 \dots$ where each $z_i$ is "random".

</div>

### 1.1 Nonrandomness as Compressibility

**Intuition.** A finite sequence of bits is *nonrandom* if it possesses an essentially shorter description, i.e., it can be restored from a shorter source sequence. Formalizing "description" via Turing machines leads to **Kolmogorov complexity**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Kolmogorov complexity with respect to $M$)</span></p>

Let $M$ be a Turing machine that takes binary words as input and produces binary words as output. For a word $\sigma$, the **Kolmogorov complexity of $\sigma$ with respect to $M$** is the length of the shortest input $\tau$ on which $M$ outputs $\sigma$:

$$C_M(\sigma) = \min \lbrace l(\tau) : M(\tau) = \sigma \rbrace.$$

If no such $\tau$ exists, $C_M(\sigma)$ may remain undefined.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Universal Turing machine)</span></p>

A Turing machine $U$ is called **universal** if for every Turing machine $M$ there exists a binary word $\rho_M$, called the **code of $M$**, such that

$$U(\rho_M \tau) = M(\tau) \qquad \text{for every word } \tau.$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Kolmogorov complexity)</span></p>

The **Kolmogorov complexity** of a word $\sigma$ is its complexity with respect to a fixed universal Turing machine $U$:

$$C(\sigma) = C_U(\sigma) = \min \lbrace l(\rho_M \tau) : U(\rho_M \tau) = \sigma \rbrace = \min \lbrace l(\rho_M) + l(\tau) : M(\tau) = \sigma \rbrace.$$

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Kolmogorov complexity assigns a natural number)</span></p>

**Kolmogorov complexity** is a function that takes a finite binary string as its input and returns a **natural number** as its output:

$$C:\lbrace 0,1\rbrace^\ast \to \mathbb{N}$$

</div>

The choice of universal machine matters only up to a constant: passing through $U$ adds at most the cost of the encoding $\rho_M$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(Universality up to a constant, (1))</span></p>

For every Turing machine $M$ one can compute a constant $c_M = l(\rho_M)$ such that

$$C(\sigma) \le C_M(\sigma) + c_M \qquad \text{for every word } \sigma. \tag{1}$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Compressing $X$ and $Z$)</span></p>

* **Sequence $X = (01)^{1000}$.** Intuitive description: *"the string $01$ written 1000 times in a row"* — about 30 symbols. Formally, take the Turing machine $M$ defined by:

   * **Input:** $n$ (as a binary word).
   * **For $i$ from $1$ to $n$:** print "$01$".

   Then $M(\mathrm{bin}(1000)) = M(1111101000) = X$, so

   $$C_M(X) \le l(1111101000) = 10, \qquad \text{hence} \qquad C(X) \le 10 + c_M.$$

* **Sequence $Z = z_1 0\, z_2 0 \dots z_{1000} 0$.** Intuitive description: list $b_1, \dots, b_{1000}$ at the odd positions and zeros in between (about 1000 symbols, since the $b_i$ have to be written explicitly). Formally, take the Turing machine $L$:

   * **Input:** $\sigma = s_1 \dots s_n$.
   * **For $i$ from $1$ to $n$:** print "$s_i 0$".

   Then $L(z_1 z_2 \dots z_{1000}) = Z$, so

   $$C_L(Z) \le l(z_1 \dots z_{1000}) = 1000, \qquad \text{hence} \qquad C(Z) \le 1000 + c_L.$$

* **Sequence $Y$.** A short description of $Y$ is its index in a fixed computable enumeration of all binary words of length 2000 with at most 200 ones (see Section 1.3 — this enumeration is much smaller than $2^{2000}$). TODO: is not the complexity of such a word 200 + c_L, because 200 indices with ones, the rest are zeros hardcoded into a turing machine for this class of sequences Y?

</div>

TODO: is there a tension between having a description of turing machine for a specific class of sequences and more general turing machine that gets the description of what to do.

#### From finite strings to infinite sequences

**Intuition.** An infinite string is *nonrandom* if its arbitrarily long initial segments have essentially shorter descriptions.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Compressing prefixes of $\tilde X$ and $\tilde Z$)</span></p>

For every prefix $\tilde X \upharpoonright 2n = (01)^n$, the same machine $M$ as above gives $M(\mathrm{bin}(n)) = \tilde X \upharpoonright 2n$. Since $l(\mathrm{bin}(n)) = \log(n) + O(1)$,

$$C(\tilde X \upharpoonright 2n) \le \log(n) + c_M,$$

with $c_M$ independent of $n$.

For every prefix $\tilde Z \upharpoonright 2n = b_1 0 b_2 0 \dots b_n 0$, the machine $L$ gives $L(b_1 \dots b_n) = \tilde Z \upharpoonright 2n$, so

$$C(\tilde Z \upharpoonright 2n) \le n + c_L,$$

again with $c_L$ independent of $n$.

</div>

How incompressible can an infinite sequence be? Surprisingly, *no* infinite sequence has all of its prefixes incompressible — there is always some recurrent slack.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(1.3 — Martin-Löf, 1966)</span></p>

For every infinite binary sequence $A$ and every constant $c$, there exist infinitely many $n$ such that

$$C(A \upharpoonright n) < n - c.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

We enumerate the set $S = (\sigma_0, \sigma_1, \dots)$ in stages: at stage $n$, enumerate into $S$ all words of length $n$ that begin with $w_n$ (recall $(w_0, w_1, w_2, w_3, \dots) = (\lambda, 0, 1, 00, \dots)$ is the length-lexicographic enumeration of all binary words). The first few stages produce

$$
\underbrace{\lambda}_{\text{stage 0}},\quad
\underbrace{0}_{\text{stage 1}},\quad
\underbrace{10, 11}_{\text{stage 2}},\quad
\underbrace{000, 001}_{\text{stage 3}},\quad
\underbrace{0100, 0101, 0110, 0111}_{\text{stage 4}}, \dots
$$

Here $\sigma_n$ denotes a **single word**: the $n$-th word in the flattened enumeration $S$. A stage may enumerate several words, so the stage number and the final enumeration index are different roles. Equivalently, one could write "stage $s$" above and reserve $n$ for the index in $\sigma_n$.

For the machine $M$ that, given $\mathrm{bin}(n)$, returns $\sigma_n$, we have

$$C_M(\sigma_n) \le l(\mathrm{bin}(n)) \le \log(n) + 1, \qquad \text{hence} \qquad C(\sigma_n) \le \log(n) + c_M.$$

It remains to observe that every infinite sequence $A$ contains infinitely many prefixes in the enumeration $S$. For each $k$, let $s$ be the length-lexicographic index of $A \upharpoonright k$, so that $w_s = A \upharpoonright k$. Since a word of length $k$ has index at least $2^k - 1 \ge k$, we have $s \ge k$, and therefore $A \upharpoonright s$ begins with $w_s = A \upharpoonright k$ — that is, $A \upharpoonright s \in S$. As $k \to \infty$ these indices $s \to \infty$, giving infinitely many prefixes of $A$ in $S$.

Finally, $l(\sigma_n) - \log(n) \to \infty$ as $n \to \infty$: stage $s$ enumerates $2^{\,s - l(w_s)} \approx 2^s / s$ words of length $s$, so the cumulative count $N(s)$ up to stage $s$ satisfies $\log(N(s)) = s - \log(s) + O(1)$; hence even for the largest index in stage $s$, $l(\sigma_n) - \log(n) \ge \log(s) - O(1) \to \infty$. Combining this with $C(\sigma_n) \le \log(n) + c_M$, the slack $l(\sigma_n) - C(\sigma_n)$ becomes arbitrarily large along the prefixes of $A$.

</details>
</div>

TODO: grasp it intuitively, rn looks pretty technical

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(1.4)</span></p>

Using the construction in the previous proof, show that for every infinite binary sequence $X$, there exist infinitely many $n$ such that

$$C(X \upharpoonright n) \le n - \log(n) + O(1).$$

</div>

### 1.2 Nonrandomness as Predictability

**Intuition.** A sequence of bits is *nonrandom* if it is possible to predict some of its bits with sufficiently high probability — equivalently, if a gambler can beat the sequence by betting on its bits.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Predicting $X$, $Y$, $Z$ and their infinite analogues)</span></p>

* For $X$: $X(2i) = 0$ and $X(2i+1) = 1$ for every $i$, so *every* bit of $X$ is predictable. The same holds for $\tilde X$. In particular, betting one dollar that $\tilde X(2i+1) = 0$ (or $= 1$ depending on indexing) yields an unbounded increase of capital.
* For $Z$: $Z(2i+1) = 0$ for every $i$, so every second bit of $Z$ is predictable. The same holds for $\tilde Z$.
* For $Y$: for every $n$ from 0 to 1999 we can bet one dollar that the bit at position $n$ is 0 and earn approximately a $1800$ dollars profit (since at most 200 of the 2000 bits are 1). Similarly, for $\tilde Y$ we can secure an unbounded increase of capital without knowing exactly which positions hold the rare ones.

</div>

### 1.3 Nonrandomness as Untypicality

**Intuition.** A finite sequence with some *rare* property — one that is easy to describe and easy to check — is nonrandom.

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Untypical properties of $X$, $Y$, $Z$)</span></p>

* $X = (01)^{1000}$ is one of a kind.
* $Y$ has at most 200 ones. Among $2^{2000}$ binary words of length 2000, only

   $$\binom{2000}{\le 200} \approx 1.1 \cdot \binom{2000}{200} \approx 2^{938}$$

   have this property — a vanishingly small fraction.
* Every bit of $Z$ at an even position equals zero. Only $2^{1000}$ of $2^{2000}$ binary words of length 2000 share this property.

</div>

#### From finite strings to infinite sequences

**Intuition.** An infinite sequence is *nonrandom* if we can construct a family of sets of arbitrarily small measure ("layers of untypicality") all containing the sequence. Formalizing "we can construct" via uniform enumeration leads to **Martin-Löf tests**.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(1.5 — Martin-Löf test)</span></p>

A **Martin-Löf test** $\mathcal{M} = (M_0, M_1, \dots)$ is a uniformly enumerable family of open sets $M_i = ([[\sigma_0^i]], [[\sigma_1^i]], \dots)$ called **layers**, such that for every $i$ the layer $M_i$ has Lebesgue measure at most $2^{-i}$:

$$\mu(M_i) = \sum_j \lambda([\![\sigma_j^i]\!]) = \sum_j 2^{-l(\sigma_j^i)} \le 2^{-i}. \tag{2}$$

An infinite sequence $A$ is called **Martin-Löf nonrandom** iff there exists a Martin-Löf test $(M_1, M_2, \dots)$ such that $A$ is contained in *every* layer $M_i$ — in this case we say $\mathcal{M}$ **covers** $A$. Otherwise, $A$ is **Martin-Löf random**.

</div>

TODO: what is the intiutive connection between randomness and Martin-Lof?

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(1.6 — Martin-Löf tests covering $\tilde X$ and $\tilde Z$)</span></p>

A Martin-Löf test $\mathcal{M}^{\tilde X}$ covering $\tilde X$ is

$$M_1 = ([\![01]\!]),\quad M_2 = ([\![0101]\!]),\quad M_3 = ([\![010101]\!]),\ \dots$$

(each layer consists of a single basic open set), since $\tilde X \in M_i$ and $\mu(M_i) = 2^{-2i}$ for every $i$.

A Martin-Löf test $\mathcal{M}^{\tilde Z}$ covering $\tilde Z$ is

$$M_1 = ([\![00]\!], [\![10]\!]),$$

$$M_2 = ([\![0000]\!], [\![0010]\!], [\![1000]\!], [\![1010]\!]),$$

$$M_3 = ([\![000000]\!], [\![000010]\!], [\![001000]\!], [\![001010]\!], [\![100000]\!], [\![100010]\!], [\![101000]\!], [\![101010]\!]),\ \dots$$

or, more formally,

$$M_i = \lbrace [\![a_1 0\, a_2 0 \dots a_i 0]\!] : a_1 a_2 \dots a_i \in \lbrace 0, 1 \rbrace^{i} \rbrace \qquad \text{for every natural } i.$$

For every $i$ one checks that $\tilde Z \in M_i$ and that

$$\lambda(M_i) = \sum_{[\![\sigma]\!] \in M_i} 2^{-l(\sigma)} = 2^i \cdot 2^{-2i} = 2^{-i}.$$

</div>

<figure class="math-figure">
  <svg viewBox="0 0 620 320" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:620px" aria-label="Martin-Löf test layers covering the alternating sequence">
    <g font-family="serif" font-size="12" fill="#1f2430">
      <text x="20" y="40" font-weight="600">M₁ = ⟦01⟧</text>
      <rect x="40" y="50" width="540" height="20" fill="none" stroke="#cbd2e0" />
      <rect x="175" y="50" width="135" height="20" fill="rgba(168,111,0,0.18)" stroke="#a86f00" stroke-width="1.2" />
      <text x="380" y="65" font-size="11" fill="#5b6270">measure 1/4 ≤ 2⁻¹</text>

      <text x="20" y="100" font-weight="600">M₂ = ⟦0101⟧</text>
      <rect x="40" y="110" width="540" height="20" fill="none" stroke="#cbd2e0" />
      <rect x="40"  y="110" width="540" height="20" fill="rgba(168,111,0,0.04)" />
      <rect x="208.75" y="110" width="33.75" height="20" fill="rgba(168,111,0,0.30)" stroke="#a86f00" stroke-width="1.2" />
      <text x="380" y="125" font-size="11" fill="#5b6270">measure 1/16 ≤ 2⁻²</text>

      <text x="20" y="160" font-weight="600">M₃ = ⟦010101⟧</text>
      <rect x="40" y="170" width="540" height="20" fill="none" stroke="#cbd2e0" />
      <rect x="40"  y="170" width="540" height="20" fill="rgba(168,111,0,0.04)" />
      <rect x="217.19" y="170" width="8.44" height="20" fill="rgba(168,111,0,0.45)" stroke="#a86f00" stroke-width="1.2" />
      <text x="380" y="185" font-size="11" fill="#5b6270">measure 1/64 ≤ 2⁻³</text>

      <g>
        <line x1="220" y1="50" x2="220" y2="220" stroke="#d65336" stroke-width="1.2" stroke-dasharray="3 3" />
        <circle cx="220" cy="220" r="3.5" fill="#d65336" />
        <text x="220" y="240" font-size="11" fill="#d65336" text-anchor="middle">X̃ = 0.010101… = 1/3</text>
      </g>

      <line x1="40" y1="270" x2="580" y2="270" stroke="#444" stroke-width="1.2" />
      <g stroke="#444" stroke-width="1">
        <line x1="40"  y1="266" x2="40"  y2="278" />
        <line x1="175" y1="266" x2="175" y2="278" />
        <line x1="310" y1="266" x2="310" y2="278" />
        <line x1="445" y1="266" x2="445" y2="278" />
        <line x1="580" y1="266" x2="580" y2="278" />
      </g>
      <g font-size="11" fill="#444" text-anchor="middle">
        <text x="40"  y="294">0</text>
        <text x="175" y="294">1/4</text>
        <text x="310" y="294">1/2</text>
        <text x="445" y="294">3/4</text>
        <text x="580" y="294">1</text>
      </g>

      <text x="310" y="316" text-anchor="middle" font-size="11" fill="#5b6270" font-style="italic">
        Each layer is a single dyadic interval shrinking around the same limit point.
      </text>
    </g>
  </svg>
  <figcaption>Three layers $M_1 \supset M_2 \supset M_3$ of the Martin-Löf test covering $\tilde X = (01)^{\omega}$. Each layer halves twice when the prefix is extended by two bits, so its measure is $2^{-2i} \le 2^{-i}$ — exactly meeting the Martin-Löf condition. All layers contain the point $\tilde X$, identified with the binary expansion $0.010101\ldots = 1/3$.</figcaption>
</figure>

<figure class="math-figure">
  <svg viewBox="0 0 640 380" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:640px" aria-label="Martin-Löf test layers covering sequences with every second bit equal to zero">
    <g font-family="serif" font-size="12" fill="#1f2430">
      <!-- Pattern strip -->
      <text x="30" y="28" font-weight="600">Pattern for Z̃</text>
      <g text-anchor="middle" font-weight="600">
        <rect x="128" y="14" width="42" height="26" fill="#edf4ff" stroke="#2c4994" />
        <text x="149" y="32" fill="#2c4994">b₁</text>
        <rect x="170" y="14" width="42" height="26" fill="#fff7e0" stroke="#a86f00" />
        <text x="191" y="32" fill="#a86f00">0</text>
        <rect x="212" y="14" width="42" height="26" fill="#edf4ff" stroke="#2c4994" />
        <text x="233" y="32" fill="#2c4994">b₂</text>
        <rect x="254" y="14" width="42" height="26" fill="#fff7e0" stroke="#a86f00" />
        <text x="275" y="32" fill="#a86f00">0</text>
        <rect x="296" y="14" width="42" height="26" fill="#edf4ff" stroke="#2c4994" />
        <text x="317" y="32" fill="#2c4994">b₃</text>
        <rect x="338" y="14" width="42" height="26" fill="#fff7e0" stroke="#a86f00" />
        <text x="359" y="32" fill="#a86f00">0</text>
        <text x="398" y="32" fill="#5b6270">...</text>
      </g>
      <text x="470" y="31" font-size="11" fill="#5b6270">blue bits are free; zeros are forced</text>

      <!-- M1 -->
      <text x="20" y="82" font-weight="600">M₁ = ⟦00⟧ ∪ ⟦10⟧</text>
      <rect x="50" y="92" width="540" height="20" fill="none" stroke="#cbd2e0" />
      <rect x="50" y="92" width="135" height="20" fill="rgba(61,122,38,0.18)" stroke="#3d7a26" stroke-width="1.1" />
      <rect x="320" y="92" width="135" height="20" fill="rgba(61,122,38,0.18)" stroke="#3d7a26" stroke-width="1.1" />
      <rect x="320" y="92" width="135" height="20" fill="rgba(214,83,54,0.20)" stroke="#d65336" stroke-width="1.2" />
      <text x="410" y="86" font-size="11" fill="#5b6270">2 intervals · 2⁻² = 2⁻¹</text>

      <!-- M2 -->
      <text x="20" y="145" font-weight="600">M₂ = ⟦0000⟧ ∪ ⟦0010⟧ ∪ ⟦1000⟧ ∪ ⟦1010⟧</text>
      <rect x="50" y="155" width="540" height="20" fill="none" stroke="#cbd2e0" />
      <g fill="rgba(61,122,38,0.22)" stroke="#3d7a26" stroke-width="1.1">
        <rect x="50" y="155" width="33.75" height="20" />
        <rect x="117.5" y="155" width="33.75" height="20" />
        <rect x="320" y="155" width="33.75" height="20" />
        <rect x="387.5" y="155" width="33.75" height="20" />
      </g>
      <rect x="320" y="155" width="33.75" height="20" fill="rgba(214,83,54,0.28)" stroke="#d65336" stroke-width="1.2" />
      <text x="410" y="149" font-size="11" fill="#5b6270">4 intervals · 2⁻⁴ = 2⁻²</text>

      <!-- M3 -->
      <text x="20" y="208" font-weight="600">M₃: all prefixes a₁0a₂0a₃0</text>
      <rect x="50" y="218" width="540" height="20" fill="none" stroke="#cbd2e0" />
      <g fill="rgba(61,122,38,0.26)" stroke="#3d7a26" stroke-width="1">
        <rect x="50" y="218" width="8.44" height="20" />
        <rect x="66.88" y="218" width="8.44" height="20" />
        <rect x="117.5" y="218" width="8.44" height="20" />
        <rect x="134.38" y="218" width="8.44" height="20" />
        <rect x="320" y="218" width="8.44" height="20" />
        <rect x="336.88" y="218" width="8.44" height="20" />
        <rect x="387.5" y="218" width="8.44" height="20" />
        <rect x="404.38" y="218" width="8.44" height="20" />
      </g>
      <rect x="336.88" y="218" width="8.44" height="20" fill="rgba(214,83,54,0.38)" stroke="#d65336" stroke-width="1.2" />
      <text x="410" y="212" font-size="11" fill="#5b6270">8 intervals · 2⁻⁶ = 2⁻³</text>

      <!-- Sample covered point -->
      <g>
        <line x1="341" y1="92" x2="341" y2="270" stroke="#d65336" stroke-width="1.2" stroke-dasharray="3 3" />
        <circle cx="341" cy="270" r="3.5" fill="#d65336" />
        <text x="341" y="290" font-size="11" fill="#d65336" text-anchor="middle">one Z̃ with prefix 100010...</text>
      </g>

      <!-- Number line -->
      <line x1="50" y1="320" x2="590" y2="320" stroke="#444" stroke-width="1.2" />
      <g stroke="#444" stroke-width="1">
        <line x1="50"  y1="316" x2="50"  y2="328" />
        <line x1="185" y1="316" x2="185" y2="328" />
        <line x1="320" y1="316" x2="320" y2="328" />
        <line x1="455" y1="316" x2="455" y2="328" />
        <line x1="590" y1="316" x2="590" y2="328" />
      </g>
      <g font-size="11" fill="#444" text-anchor="middle">
        <text x="50"  y="344">0</text>
        <text x="185" y="344">1/4</text>
        <text x="320" y="344">1/2</text>
        <text x="455" y="344">3/4</text>
        <text x="590" y="344">1</text>
      </g>

      <text x="320" y="372" text-anchor="middle" font-size="11" fill="#5b6270" font-style="italic">
        Each layer keeps all choices of the free bits, but forces one more even-position zero.
      </text>
    </g>
  </svg>
  <figcaption>Three layers of the Martin-Löf test covering every sequence of the form $\tilde Z = b_1 0 b_2 0 b_3 0 \dots$. Layer $M_i$ is the union of $2^i$ cylinders, one for each possible choice of the free bits $b_1,\dots,b_i$; each cylinder has measure $2^{-2i}$, so the whole layer has measure $2^i \cdot 2^{-2i} = 2^{-i}$. The red marker shows one possible $\tilde Z$ path through the nested unions.</figcaption>
</figure>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(1.6)</span></p>

Construct a Martin-Löf test $\mathcal{M}$ that covers every sequence $X$ which does *not* contain the word $111$ as a substring.

</div>

## Kolmogorov complexity

In this section we examine the Kolmogorov complexity as a function on binary words, starting from the trivial upper bound and a few straightforward properties.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.1 — Trivial upper bound)</span></p>

There exists a constant $c$ such that the Kolmogorov complexity of every binary word $w$ is no greater than $l(w) + c$:

$$\exists c \ \forall w \in \lbrace 0, 1 \rbrace^{\ast}: \quad C(w) \le l(w) + c.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The identity Turing machine $M$ on $\lbrace 0, 1 \rbrace^{\ast}$ satisfies $M(w) = w$, hence $C_M(w) = l(w)$. By (1), there exists a constant $c_M$ with $C(w) \le C_M(w) + c_M = l(w) + c_M$ for every $w$.

</details>
</div>

<figure class="math-figure">
  <svg viewBox="0 0 660 280" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:660px" aria-label="The trivial upper bound as a literal description plus fixed overhead">
    <defs>
      <marker id="arrow-trivial-bound" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
        <path d="M0,0 L8,4 L0,8 Z" fill="#5b6270" />
      </marker>
    </defs>
    <g font-family="serif" font-size="13" fill="#1f2430">
      <!-- Literal word -->
      <text x="330" y="32" text-anchor="middle" font-size="15" font-weight="600">Describe w by printing w itself</text>
      <g>
        <rect x="80" y="70" width="110" height="44" rx="4" fill="#fff7e0" stroke="#a86f00" stroke-width="1.5" />
        <text x="135" y="88" text-anchor="middle" font-size="11" fill="#a86f00">fixed code</text>
        <text x="135" y="105" text-anchor="middle" font-weight="600">print input</text>

        <rect x="190" y="70" width="260" height="44" rx="4" fill="#edf4ff" stroke="#2c4994" stroke-width="1.5" />
        <text x="320" y="88" text-anchor="middle" font-size="11" fill="#2c4994">literal payload</text>
        <text x="320" y="105" text-anchor="middle" font-weight="600">w = 101101001...</text>

        <line x1="470" y1="92" x2="540" y2="92" stroke="#5b6270" stroke-width="1.4" marker-end="url(#arrow-trivial-bound)" />
        <rect x="545" y="70" width="55" height="44" rx="4" fill="#f2f7ed" stroke="#3d7a26" stroke-width="1.5" />
        <text x="572" y="97" text-anchor="middle" font-weight="600">w</text>
      </g>

      <!-- Length accounting -->
      <g>
        <line x1="80" y1="150" x2="190" y2="150" stroke="#a86f00" stroke-width="3" />
        <line x1="190" y1="150" x2="450" y2="150" stroke="#2c4994" stroke-width="3" />
        <g stroke="#5b6270" stroke-width="1">
          <line x1="80" y1="142" x2="80" y2="158" />
          <line x1="190" y1="142" x2="190" y2="158" />
          <line x1="450" y1="142" x2="450" y2="158" />
        </g>
        <text x="135" y="178" text-anchor="middle" fill="#a86f00">constant c</text>
        <text x="320" y="178" text-anchor="middle" fill="#2c4994">l(w) bits</text>
        <text x="265" y="216" text-anchor="middle" font-size="16">
          program length = c + l(w)
        </text>
      </g>

      <!-- Complexity inequality -->
      <g>
        <rect x="160" y="232" width="340" height="30" rx="4" fill="#f8f9fb" stroke="#cbd2e0" />
        <text x="330" y="252" text-anchor="middle" font-size="14">C(w) &lt;= l(w) + c for every word w</text>
      </g>
    </g>
  </svg>
  <figcaption>Proposition 2.1 says that a word is never harder to describe than by giving the word literally. The fixed part tells the universal machine to copy/print the remaining input; the variable part is just $w$ itself. The fixed part contributes only a constant $c$, so the total description length is $l(w) + c$.</figcaption>
</figure>

The next theorem expresses that no computable process can blow up Kolmogorov complexity by more than an additive constant — the constant depends on the *process* but not on the *input*.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(2.2 — Computable processes do not increase complexity)</span></p>

For every computable function $f :\subseteq \lbrace 0, 1 \rbrace^{\ast} \to \lbrace 0, 1 \rbrace^{\ast}$, there exists a constant $c_f$ such that

$$C(f(w)) < C(w) + c_f \qquad \text{for every } w \in \mathrm{dom}(f).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $M_f$ be a Turing machine computing $f$, and consider the machine $M$ that, on input $\sigma$, computes $f(U(\sigma))$. For every $w \in \mathrm{dom}(f)$, let $\sigma_w$ be an optimal encoding of $w$, i.e., $U(\sigma_w) = w$ and $l(\sigma_w) = C(w)$. Then

$$M(\sigma_w) = f(U(\sigma_w)) = f(w),$$

so $C_M(f(w)) \le l(\sigma_w) = C(w)$. By (1), there exists a constant $c_M$ with

$$C(f(w)) \le C_M(f(w)) + c_M \le C(w) + c_M,$$

which concludes the proof.

</details>
</div>

<figure class="math-figure">
  <svg viewBox="0 0 680 330" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:680px" aria-label="A computable process reuses a shortest description and adds only fixed overhead">
    <defs>
      <marker id="arrow-computable-process" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
        <path d="M0,0 L8,4 L0,8 Z" fill="#5b6270" />
      </marker>
    </defs>
    <g font-family="serif" font-size="13" fill="#1f2430">
      <text x="340" y="30" text-anchor="middle" font-size="15" font-weight="600">The shortest code for w can be reused to describe f(w)</text>

      <!-- Original decoding path -->
      <g>
        <rect x="36" y="70" width="120" height="46" rx="4" fill="#edf4ff" stroke="#2c4994" stroke-width="1.5" />
        <text x="96" y="88" text-anchor="middle" font-size="11" fill="#2c4994">shortest code</text>
        <text x="96" y="106" text-anchor="middle" font-weight="600">sigma_w</text>

        <line x1="166" y1="93" x2="218" y2="93" stroke="#5b6270" stroke-width="1.4" marker-end="url(#arrow-computable-process)" />

        <rect x="226" y="70" width="86" height="46" rx="4" fill="#fff7e0" stroke="#a86f00" stroke-width="1.5" />
        <text x="269" y="88" text-anchor="middle" font-size="11" fill="#a86f00">universal</text>
        <text x="269" y="106" text-anchor="middle" font-weight="600">U</text>

        <line x1="322" y1="93" x2="374" y2="93" stroke="#5b6270" stroke-width="1.4" marker-end="url(#arrow-computable-process)" />

        <rect x="382" y="70" width="76" height="46" rx="4" fill="#f2f7ed" stroke="#3d7a26" stroke-width="1.5" />
        <text x="420" y="99" text-anchor="middle" font-weight="600">w</text>

        <line x1="468" y1="93" x2="520" y2="93" stroke="#5b6270" stroke-width="1.4" marker-end="url(#arrow-computable-process)" />

        <rect x="528" y="70" width="86" height="46" rx="4" fill="#fff7e0" stroke="#a86f00" stroke-width="1.5" />
        <text x="571" y="88" text-anchor="middle" font-size="11" fill="#a86f00">computable</text>
        <text x="571" y="106" text-anchor="middle" font-weight="600">f</text>

        <line x1="571" y1="126" x2="571" y2="162" stroke="#5b6270" stroke-width="1.4" marker-end="url(#arrow-computable-process)" />
        <rect x="525" y="170" width="92" height="46" rx="4" fill="#f2f7ed" stroke="#3d7a26" stroke-width="1.5" />
        <text x="571" y="199" text-anchor="middle" font-weight="600">f(w)</text>
      </g>

      <!-- Composed machine shortcut -->
      <g>
        <path d="M96 132 C130 190, 225 238, 340 238 C455 238, 520 218, 571 218" fill="none" stroke="#2c4994" stroke-width="1.3" stroke-dasharray="4 4" marker-end="url(#arrow-computable-process)" />
        <rect x="230" y="192" width="220" height="54" rx="4" fill="#f8f9fb" stroke="#cbd2e0" />
        <text x="340" y="212" text-anchor="middle" font-size="11" fill="#5b6270">compose once</text>
        <text x="340" y="232" text-anchor="middle" font-weight="600">M = f after U</text>
      </g>

      <!-- Length accounting -->
      <g>
        <line x1="80" y1="278" x2="245" y2="278" stroke="#2c4994" stroke-width="3" />
        <line x1="245" y1="278" x2="330" y2="278" stroke="#a86f00" stroke-width="3" />
        <g stroke="#5b6270" stroke-width="1">
          <line x1="80" y1="270" x2="80" y2="286" />
          <line x1="245" y1="270" x2="245" y2="286" />
          <line x1="330" y1="270" x2="330" y2="286" />
        </g>
        <text x="162" y="305" text-anchor="middle" fill="#2c4994">C(w) bits</text>
        <text x="287" y="305" text-anchor="middle" fill="#a86f00">c_f</text>
        <text x="505" y="292" text-anchor="middle" font-size="15">C(f(w)) &lt; C(w) + c_f</text>
      </g>
    </g>
  </svg>
  <figcaption>Theorem 2.2 packages the computable post-processing step into a single machine $M = f \circ U$. Once $M$ is fixed, the same optimal code $\sigma_w$ that describes $w$ also makes $M$ output $f(w)$. The universal machine only pays a fixed simulation cost $c_f$ for this whole composed process, independent of the particular input $w$.</figcaption>
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.3 — Counting bound)</span></p>

For every $n$, there exist at most $2^n - 1$ words of Kolmogorov complexity less than $n$:

$$\forall n: \quad \#\lbrace w \in \lbrace 0, 1 \rbrace^{\ast} : C(w) < n \rbrace < 2^n.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

For every $w$ with $C(w) = k$, there exists at least one optimal encoding $\sigma$ with $U(\sigma) = w$ and $l(\sigma) = k$. There are exactly $2^n - 1$ words of length less than $n$, and each can be the optimal encoding of at most one word $w$. The bound follows.

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(2.3.1 — Layered counting on words of length $n$)</span></p>

Among $2^n$ words of length $n$, there are

* at most $2^n - 1$ words of Kolmogorov complexity $\le n - 1$;
* at most $2^{n-1} - 1$ words of Kolmogorov complexity $\le n - 2$;
* $\vdots$
* at most $2^2 - 1$ words of Kolmogorov complexity $\le 1$;
* at most $2^1 - 1$ words of Kolmogorov complexity $\le 0$.

</div>

<figure class="math-figure">
  <svg viewBox="0 0 620 360" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:620px" aria-label="Counting bound: how many words of length n have a given Kolmogorov complexity">
    <g font-family="serif" font-size="12" fill="#1f2430">
      <!-- Title -->
      <text x="310" y="20" text-anchor="middle" font-weight="600">Words of length n = 8 distributed by Kolmogorov complexity</text>

      <!-- Axes -->
      <line x1="60" y1="290" x2="580" y2="290" stroke="#444" stroke-width="1.2" />
      <line x1="60" y1="290" x2="60"  y2="50"  stroke="#444" stroke-width="1.2" />
      <polygon points="580,290 572,286 572,294" fill="#444" />
      <polygon points="60,50 56,58 64,58" fill="#444" />

      <!-- Y-axis label -->
      <text x="20" y="170" text-anchor="middle" font-size="11" fill="#444" transform="rotate(-90,20,170)">
        # words with C(w) ≤ k
      </text>
      <text x="320" y="320" text-anchor="middle" font-size="11" fill="#444">complexity threshold k</text>

      <!-- Y ticks for log-scale shown linearly via heights ∝ 2^k -->
      <!-- Total bar: height 240 ↔ 2^8 = 256 words; bar at k corresponds to 2^k -->
      <!-- bar height for k = scaling * (2^k - 1), with cap at 240 -->
      <!-- Use explicit values: k from 0..8 at x-positions evenly spaced -->

      <!-- Bars for k = 0..8 -->
      <!-- Heights computed from bound: at most 2^k - 1 words have C ≤ k -1 (cumulative) -->
      <!-- Actually the corollary bounds: ≤ 2^(n) - 1 of complexity ≤ n-1, ≤ 2^(n-1) - 1 of complexity ≤ n-2, etc. -->
      <!-- Here we plot the actual at-most count for "complexity ≤ k": which equals 2^(k+1) - 1 from the corollary, scaled to 256 = 2^8 -->
      <!-- Cumulative count C ≤ k upper bound: ≤ 2^(k+1) - 1 -->
      <!-- Scale: bar_height = 240 * (2^(k+1) - 1) / 255 -->

      <!-- k=0: 1 word ⇒ height ≈ 240 * 1/255 ≈ 0.94 -->
      <!-- k=1: 3 words ⇒ ≈ 2.82 -->
      <!-- k=2: 7 ⇒ 6.58 -->
      <!-- k=3: 15 ⇒ 14.1 -->
      <!-- k=4: 31 ⇒ 29.2 -->
      <!-- k=5: 63 ⇒ 59.3 -->
      <!-- k=6: 127 ⇒ 119.5 -->
      <!-- k=7: 255 ⇒ 240 -->

      <g stroke="#a86f00" fill="rgba(168,111,0,0.30)" stroke-width="1.2">
        <rect x="80"  y="289" width="40" height="1"   />
        <rect x="135" y="287" width="40" height="3"   />
        <rect x="190" y="283" width="40" height="7"   />
        <rect x="245" y="276" width="40" height="14"  />
        <rect x="300" y="261" width="40" height="29"  />
        <rect x="355" y="231" width="40" height="59"  />
        <rect x="410" y="170" width="40" height="120" />
        <rect x="465" y="50"  width="40" height="240" />
      </g>
      <!-- Label total 2^8 line -->
      <line x1="60" y1="50" x2="580" y2="50" stroke="#d65336" stroke-width="1" stroke-dasharray="3 3" />
      <text x="575" y="46" text-anchor="end" font-size="11" fill="#d65336">total 2⁸ = 256 words of length 8</text>

      <!-- X tick labels -->
      <g font-size="11" text-anchor="middle">
        <text x="100" y="306">≤ 0</text>
        <text x="155" y="306">≤ 1</text>
        <text x="210" y="306">≤ 2</text>
        <text x="265" y="306">≤ 3</text>
        <text x="320" y="306">≤ 4</text>
        <text x="375" y="306">≤ 5</text>
        <text x="430" y="306">≤ 6</text>
        <text x="485" y="306">≤ 7</text>
      </g>

      <!-- Bound annotations on top of bars -->
      <g font-size="10" fill="#5b6270" text-anchor="middle">
        <text x="100" y="284">≤1</text>
        <text x="155" y="282">≤3</text>
        <text x="210" y="278">≤7</text>
        <text x="265" y="271">≤15</text>
        <text x="320" y="256">≤31</text>
        <text x="375" y="226">≤63</text>
        <text x="430" y="165">≤127</text>
        <text x="485" y="45" >≤255</text>
      </g>

      <text x="310" y="350" text-anchor="middle" font-size="11" fill="#5b6270" font-style="italic">
        Each step doubles the count: the "low complexity" stratum is exponentially small in n.
      </text>
    </g>
  </svg>
  <figcaption>Counting bound for $n = 8$: of the $2^n = 256$ binary words of length 8, at most $2^{k+1} - 1$ have Kolmogorov complexity $\le k$ (Proposition 2.3). The bars grow geometrically, so even at $k = n - 1 = 7$ the upper bound $2^n - 1 = 255$ leaves at least one word of complexity $\ge n$ — incompressible words exist at every length.</figcaption>
</figure>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Average Kolmogorov Complexity)</span></p>

The function $g:\mathbb{N}\to\mathbb{N}$ is called **average Kolmogorov complexity** of a word of length $n$:

$$g(n):= 2^{-n}\sum_{l(w)=n} C(w).$$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.4 — Average complexity is asymptotically the length)</span></p>

Let $g(n)$ be the average Kolmogorov complexity of a word of length $n$. Then

$$\frac{g(n)}{n} \xrightarrow[n \to \infty]{} 1.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**Upper bound**

To establish the upper bound we consider the sum of complexities:

$$\sum_{l(w)=n} C(w) \le 2^n (l(w) + c) = 2^n (n + c),$$

were we used $C(w)\le l(w) + c$ and that totally the sum has $2^n$ terms, because there are only $2^n$ words of the length $n$.

Deviding by $2^n$ in $g(n)$ gives

$$g(n) = \frac{\sum_{l(w)=n} C(w)}{2^n} \le \frac{2^n(n+c)}{2^n} = n+c$$

Therefore, 

$$\frac{g(n)}{n} \le 1 + \frac{c}{n} \xrightarrow[n \to \infty]{} 1$$

**Lower bound**

For the lower bound, distribute the $2^n$ words of length $n$ across complexity strata using Corollary 2.3.1:

$$
\begin{aligned}
&\frac{2^{n-1}(n - 1) + 2^{n-2}(n - 2) + \dots + 2^0 (n - n)}{2^n \cdot n} \\
&= 
\frac{2^{n-1} \cdot n + 2^{n-2} \cdot n + \dots + 2^0 \cdot n}{2^n \cdot n}
- \frac{2^{n-1} \cdot 1 + 2^{n-2} \cdot 2 + \dots + 2^0 \cdot n}{2^n \cdot n} \\
&=
1 - 2^{-n} - \frac{\sum_{i=1}^{n} 2^{-i} \cdot i}{n} \\
&\ge 
1 - 2^{-n} - \frac{\sum_{i=1}^{\infty} 2^{-i} \cdot i}{n} \xrightarrow[n \to \infty]{} 1.
\end{aligned}
$$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(2.5 — Sharper two-sided bound)</span></p>

Show that there exist constants $c_m$ and $c_M$ such that

$$n - c_m \le g(n) \le n + c_M \qquad \text{for all } n.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

**RHS**

From the first part of the proof of the proposition 2.4, $\frac{g(n)}{n} \le 1 + \frac{c}{n}$. Therefore $g(n) \le n + c$.

**LHS**

From the second part of the proof of the proposition 2.4, 

$$\frac{g(n)}{n} \geq 1 - 2^{-n} - \frac{\sum_{i=1}^{n} 2^{-i} \cdot i}{n}.$$

Therefore,

$$g(n) \geq n - (\frac{n}{2^n} + \sum_{i=1}^{n} 2^{-i} \cdot i) = n - (2 - \frac{2}{2^n}) \geq n - 2 > n - (2 + \epsilon).$$

Setting $c_m := 2 + \epsilon$, where $\epsilon > 0$, we obtain the requires inequality.

</details>
</div>

<figure class="math-figure">
  <svg viewBox="0 0 660 360" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:660px" aria-label="Two-sided constant bound for average Kolmogorov complexity">
    <defs>
      <marker id="arrow-average-bound" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
        <path d="M0,0 L8,4 L0,8 Z" fill="#5b6270" />
      </marker>
    </defs>
    <g font-family="serif" font-size="12" fill="#1f2430">
      <!-- Axes -->
      <line x1="70" y1="290" x2="600" y2="290" stroke="#5b6270" stroke-width="1.3" marker-end="url(#arrow-average-bound)" />
      <line x1="70" y1="290" x2="70" y2="35" stroke="#5b6270" stroke-width="1.3" marker-end="url(#arrow-average-bound)" />
      <text x="605" y="307" text-anchor="end" fill="#5b6270">n</text>
      <text x="55" y="42" text-anchor="middle" fill="#5b6270">complexity</text>

      <!-- Vertical guide lines -->
      <g stroke="#cbd2e0" stroke-width="0.8">
        <line x1="165" y1="290" x2="165" y2="35" />
        <line x1="260" y1="290" x2="260" y2="35" />
        <line x1="355" y1="290" x2="355" y2="35" />
        <line x1="450" y1="290" x2="450" y2="35" />
        <line x1="545" y1="290" x2="545" y2="35" />
      </g>

      <!-- Constant-width band around y = n -->
      <polygon points="95,220 565,45 565,122 95,290" fill="rgba(44,73,148,0.10)" stroke="none" />
      <line x1="95" y1="220" x2="565" y2="45" stroke="#2c4994" stroke-width="1.6" stroke-dasharray="6 4" />
      <line x1="95" y1="255" x2="565" y2="82" stroke="#1f2430" stroke-width="1.5" />
      <line x1="95" y1="290" x2="565" y2="122" stroke="#3d7a26" stroke-width="1.6" stroke-dasharray="6 4" />

      <!-- Schematic average complexity curve trapped inside the band -->
      <path d="M95 268 C135 250, 165 248, 205 232 S285 204, 325 190 S405 157, 445 144 S520 109, 565 95"
            fill="none" stroke="#d65336" stroke-width="2.4" />

      <!-- Labels -->
      <text x="442" y="48" fill="#2c4994" font-weight="600">n + c_M</text>
      <text x="423" y="82" fill="#1f2430" font-weight="600">n</text>
      <text x="434" y="128" fill="#3d7a26" font-weight="600">n - c_m</text>
      <text x="235" y="217" fill="#d65336" font-weight="600">g(n)</text>

      <!-- Constant offsets at a fixed n -->
      <g stroke="#a86f00" stroke-width="1.1" fill="none">
        <line x1="585" y1="45" x2="585" y2="82" />
        <line x1="578" y1="45" x2="592" y2="45" />
        <line x1="578" y1="82" x2="592" y2="82" />
        <line x1="585" y1="82" x2="585" y2="122" />
        <line x1="578" y1="122" x2="592" y2="122" />
      </g>
      <text x="603" y="66" fill="#a86f00">c_M</text>
      <text x="603" y="105" fill="#a86f00">c_m</text>

      <!-- Tick labels -->
      <g font-size="11" fill="#5b6270" text-anchor="middle">
        <text x="70" y="307">0</text>
        <text x="165" y="307">n_1</text>
        <text x="260" y="307">n_2</text>
        <text x="355" y="307">n_3</text>
        <text x="450" y="307">n_4</text>
        <text x="545" y="307">n_5</text>
      </g>

      <text x="330" y="335" text-anchor="middle" font-size="11" fill="#5b6270" font-style="italic">
        The vertical width of the band is constant, so dividing by n squeezes g(n)/n toward 1.
      </text>
    </g>
  </svg>
  <figcaption>A schematic plot of the sharper two-sided bound. The line $y = n$ is surrounded by a constant-width band with upper edge $n + c_M$ and lower edge $n - c_m$. The average complexity $g(n)$ may wiggle inside this band, but its distance from $n$ is uniformly bounded for all $n$, which is exactly why $g(n)/n \to 1$.</figcaption>
</figure>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Axiomatical characterization of Kolmogorov complexity)</span></p>

A sequence of sets $(M_0, M_1, \dots)$ is called **uniformly computably enumerable** (or, shortly, uniformly c.e.) if the set of pairs $\lbrace (x, i) : x \in M_i \rbrace$ is computably enumerable.

We consider the sequence of sets of binary words $(S_0, S_1,\dots)$ defined by

$$S_n := \lbrace w \in\lbrace0, 1\rbrace^\ast : C(w) < n\rbrace \quad \forall n.$$

By Proposition 2.3, it holds $\lvert S_n\rvert < 2^n$ for all $n$.
* **(a)** Show that the sequence of sets $(S_0, S_1, \dots)$ is uniformly c.e.
* **(b)** Show that, for every uniformly c.e. sequence of sets of binary words $(V_0, V_1,\dots)$ such that $\lvert V_n\rvert < 2^n$ for all $n$, there exists a constant $c$ such that

  $$C(w) < n + c \text{ for every } w \in V_n \quad \forall n.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

* **(a)** For a fixed $n$, we generate $2^n - 1$ programs $w$ of the size less then $n$. Then for each generated program $w$ using interleaving execution of UTMs, if the UTM halts with the output $x$, then we print the output $(x, n)$. Doing it in interleaving manner with all $n$, each word in each $S_n$ will be eventually printed as a corresponding pair.
* **(b)** Let there be some procedure $P$ that enumerates the pairs $\lbrace (x,i): x\in M\rbrace$. Because the procedure is deterministic, for each pair within a shared $i$ group, we can keep a number in which the word $x$ appears within this group. The index of this word has length $\log 2^n = n$. We consider that the index $n$ is know in advance, i.e. $V_n$ is fixed, we are preconditioned on it.

The clean fix in (b) (and the whole reason the bound has no $\log n$ overhead) is to exploit that a *plain* machine reads its entire input: let the program be the $n$-bit index itself, and have $M$ set $n:=l(p)$ from the input length, then enumerate $V_n​$ to the $p$-th element. Then $C_M(w)\le n$ for every $w\in V_n$, and universality (eq. (1)) gives $C(w)\le C_M(w)+c_M\le n+c_M<n+c$ with $c:=c_M+1$. As written, you're missing (i) the machine that reads $n$ off the length, and (ii) the universality step that actually produces the constant $c$ — that constant is the simulation cost $c_M=l(\rho_M)$, and your write-up never says where $c$ comes from. Also worth a half-sentence: $\lvert V_n\rvert<2^n$ means $\le 2^n-1$ elements, which comfortably inject into the $2^n$ strings of length $n$.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Rate of convergence of Kolmogorov complexity to infinity)</span></p>

We consider the function $B : \mathbb N\to\mathbb N$ that returns the largest word which has the given Kolmogorov complexity:

$$B(n) := \max\lbrace m \in \mathbb N : C(m) \leq n\rbrace.$$

* **(a)** Show that $B$ is total.
* **(b)** Show that the function $B$ **grows faster** than every partially computable function, i.e., every partially computable function $f:\mathbb N\to\mathbb N$ satisfies 
  
  $$f(n) \leq B(n) \text{ for all but finitely many } n \in \text{dom}(f).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

* **(a)** 
  
The correct proof is purely existential:

$$B(n)=\max\lbrace m\in\mathbb N:C(m)\le n\rbrace.$$

Now

$$C(m)\le n$$

means that there exists a program $p$ with $\lvert p\rvert\le n$ and $U(p)=m$. There are only finitely many such programs $p$, namely at most

$$2^{n+1}-1.$$

Therefore the set of possible outputs

$$\lbrace U(p): |p|\le n,\ U(p)\downarrow\rbrace$$

is finite. Hence its maximum exists, assuming the set is nonempty. If there may be small $n$ for which it is empty, define instead. This is the same distinction with the busy-beaver function: it is well-defined by a finite maximum, but impossible to compute in general.

$$B(n):=\max\bigl(\lbrace m\in\mathbb N:C(m)\le n\rbrace\cup{0}\bigr).$$

So: **totality comes from finiteness, not from computability.**

It is absolutely possible that the universal TM does not halt on some codes. This is exactly why $B(n)$ exists mathematically, but is not computable. 

* **(b)** By contradiction, there are infinitely $n\in\text{dom}(f)$ s.t. $f(n)>B(n)$. Then Kolmogorov complexity of the word $f(n)$ is 

  $$C(f(n)) \le C(n) + c_f \leq \log n + c + c_f = \log n + \tilde{c} < n + \tilde{c}.$$

  For finitely many words we can have $\log n + \tilde{c} \geq n$, depending on the constant $\tilde{c}$, but for infinitely many words it holds that $\log n < n$, giving for them $C(f(n)) < n$. Since the complexity of the word $f(n)$ is less then $n$, it should be in the set $\lbrace m \in \mathbb N : C(m) \leq n\rbrace$ of the definition of the function $B$ for the input $n$, thus it is not possible that $f(n) > B(n)$ for infinitely many $n \in \text{dom}(f)$.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Why we cannot stop here)</span></p>

The picture so far is twofold:

* **Computable processes are gentle.** Although long words obtained from short words via a computable function have low Kolmogorov complexity (Theorem 2.2), this is the *exception*, not the rule.
* **Most words are incompressible.** Proposition 2.4 says the average complexity of words of length $n$ is asymptotically $n$. Hence, picking a word of length $n$ uniformly at random, its complexity will be close to $n$ with very high probability.

But this raises an effective question: *can we check incompressibility for a fixed word in any algorithmic way?* The next paragraph shows that the answer is essentially **no** — Kolmogorov complexity is only one-sided computable.

</div>

### 2.1 Kolmogorov Complexity as an Upper-Semicomputable Function

Within this paragraph we deal with functions mapping words to natural numbers. Since every natural number $n$ admits a computable bijection with a binary word (e.g., $\mathrm{bin}(n)$ or the $n$-th word $w_n$ in the length-lexicographic ordering), and vice versa, all notions defined here transfer to functions from naturals to naturals or from binary words to binary words.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.6 — Computability via the graph)</span></p>

A function $f$ is computable if and only if its graph

$$G_f := \lbrace (w, f(w)) : w \in \lbrace 0, 1 \rbrace^{\ast} \rbrace$$

is computably enumerable.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(2.7 — Lower- and upper-semicomputable functions)</span></p>

A function $f$ is **lower-semicomputable** if the set

$$H_f := \lbrace (w, y) : f(w) > y \rbrace$$

(called **hypograph** of $f$) is computably enumerable.

A function $f$ is **upper-semicomputable** if the set

$$E_f := \lbrace (w, y) : f(w) < y \rbrace$$

(called **epigraph** of $f$) is computably enumerable.

</div>

Computable functions are clearly both upper- and lower-semicomputable. Conversely, given both kinds of enumerations one can compute $f(w)$ by enumerating $H_f$ and $E_f$ in parallel until some $y$ satisfies $(w, y - 1) \in H_f$ and $(w, y + 1) \in E_f$ — which forces $y - 1 < f(w) < y + 1$ — and returning $y$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.8)</span></p>

A function is computable if and only if it is both lower- and upper-semicomputable.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.9 — Approximation characterization)</span></p>

A function $f$ is upper-semicomputable if and only if there exists a computable two-argument function $F : \lbrace 0, 1 \rbrace^{\ast} \times \mathbb{N} \to \mathbb{N}$ such that for every $w \in \lbrace 0, 1 \rbrace^{\ast}$:

* $F(w, 0) \ge F(w, 1) \ge \dots$, and
* $\displaystyle \lim_{n \to \infty} F(w, n) = f(w).$

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.10 — Kolmogorov complexity is upper-semicomputable)</span></p>

The Kolmogorov complexity is an upper-semicomputable function.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Apply Proposition 2.9 to

$$F(w, t) := \min \lbrace l(\sigma) : U(\sigma)[t] \!\downarrow\, = w \rbrace,$$

where $U(\sigma)[t]$ denotes the result of running $U$ on $\sigma$ for at most $t$ steps. As $t$ increases, more inputs $\sigma$ have $U(\sigma)[t]$ converging to $w$, so $F(w, t)$ is monotonically nonincreasing in $t$ and converges to $C(w)$.

</details>
</div>

<figure class="math-figure">
  <svg viewBox="0 0 620 300" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:620px" aria-label="Approximating an upper-semicomputable function from above">
    <g font-family="serif" font-size="12" fill="#1f2430">
      <!-- Title -->
      <text x="310" y="20" text-anchor="middle" font-weight="600">Approximating C(w) from above by F(w, t)</text>

      <!-- Axes -->
      <line x1="60"  y1="240" x2="580" y2="240" stroke="#444" stroke-width="1.2" />
      <line x1="60"  y1="240" x2="60"  y2="40"  stroke="#444" stroke-width="1.2" />
      <polygon points="580,240 572,236 572,244" fill="#444" />
      <polygon points="60,40 56,48 64,48" fill="#444" />
      <text x="585" y="244" font-size="11" fill="#444">t (steps)</text>
      <text x="60"  y="32"  font-size="11" fill="#444" text-anchor="middle">F(w, t)</text>

      <!-- True value C(w) -->
      <line x1="60" y1="200" x2="580" y2="200" stroke="#d65336" stroke-width="1.2" stroke-dasharray="4 3" />
      <text x="575" y="195" text-anchor="end" font-size="11" fill="#d65336">C(w) (true Kolmogorov complexity, unreachable in finite time)</text>

      <!-- Decreasing staircase: each step is a horizontal segment, then a vertical drop -->
      <g stroke="#2c4994" stroke-width="2" fill="none">
        <polyline points="
          80,70   140,70
          140,90  220,90
          220,110 300,110
          300,140 380,140
          380,165 460,165
          460,180 520,180
          520,190 575,190
        " />
        <!-- Tick marks where the function is sampled -->
        <g fill="#2c4994" stroke="none">
          <circle cx="80"  cy="70"  r="3" />
          <circle cx="140" cy="90"  r="3" />
          <circle cx="220" cy="110" r="3" />
          <circle cx="300" cy="140" r="3" />
          <circle cx="380" cy="165" r="3" />
          <circle cx="460" cy="180" r="3" />
          <circle cx="520" cy="190" r="3" />
        </g>
      </g>

      <!-- X tick labels -->
      <g font-size="11" fill="#444" text-anchor="middle">
        <text x="80"  y="258">t₀</text>
        <text x="140" y="258">t₁</text>
        <text x="220" y="258">t₂</text>
        <text x="300" y="258">t₃</text>
        <text x="380" y="258">t₄</text>
        <text x="460" y="258">t₅</text>
        <text x="520" y="258">t₆</text>
        <text x="575" y="258">∞</text>
      </g>

      <!-- Annotations -->
      <text x="105" y="64" font-size="11" fill="#2c4994">F(w, t₀) = trivial bound</text>
      <text x="540" y="208" font-size="11" fill="#5b6270" text-anchor="end" font-style="italic">F(w, t) ↓ C(w) as t → ∞</text>

      <!-- Vertical guide for one of the drops -->
      <g stroke="#cbd2e0" stroke-dasharray="2 3" stroke-width="0.5">
        <line x1="220" y1="240" x2="220" y2="40" />
      </g>

      <text x="310" y="288" text-anchor="middle" font-size="11" fill="#5b6270" font-style="italic">
        Every time the universal machine halts on a shorter input, F drops by at least one.
      </text>
    </g>
  </svg>
  <figcaption>Approximation characterization (Proposition 2.9) applied to $C$. Running $U$ for more and more steps $t$ enumerates new short inputs $\sigma$ with $U(\sigma) = w$, so $F(w, t)$ is a *decreasing staircase* converging to $C(w)$. The function can only be approximated *from above*: at any finite $t$ we only know an upper bound, never a tight value — this is the source of the asymmetry in Theorem 2.11.</figcaption>
</figure>

The next theorem makes the asymmetry between the upper and the (missing) lower side of $C$ painfully explicit: there is no computable, unbounded *lower bound* on Kolmogorov complexity at all.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(2.11 — No computable unbounded lower bound)</span></p>

There exists no computable function $f :\subseteq \lbrace 0, 1 \rbrace^{\ast} \to \mathbb{N}$ such that

* $f$ is unbounded (for every $n$ there exists $w$ with $f(w) \downarrow\, > n$);
* $C(w) > f(w)$ for every $w \in \mathrm{dom}(f)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Suppose such $f$ exists. Construct a Turing machine $M$ that, on input $\mathrm{bin}(n)$, runs $f(\lambda), f(0), f(1), f(00), \dots$ in parallel and returns the first word $\sigma$ for which $f(\sigma) \!\downarrow\, > n$. Since $f$ is unbounded, $M$ is computable and total. By the assumed bound on $f$,

$$C(M(\mathrm{bin}(n))) \ge f(M(\mathrm{bin}(n))) > n. \tag{3}$$

On the other hand, since $M$ is computable and total, Theorem 2.2 yields a constant $c_M$ with

$$C(M(\mathrm{bin}(n))) \le C(\mathrm{bin}(n)) + c_M \le \log(n) + c_M. \tag{4}$$

Combining (3) and (4), every $n$ would satisfy

$$n < C(M(\mathrm{bin}(n))) \le \log(n) + c_M,$$

which is false for all sufficiently large $n$ (e.g., $n > 2^{c_M + 1}$).

</details>
</div>

Applying the previous theorem with $f(w) := C(w) - 1$ — which would be computable and unbounded if $C$ were — yields:

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(2.11.1 — Noncomputability of $C$)</span></p>

Kolmogorov complexity is **not** computable.

</div>

### 2.2 Properties of Kolmogorov Complexity

We now collect a few stability properties of $C$ under small textual edits, and contrast them with the dramatic *instability* one sometimes encounters: a single-bit edit can, in the worst case, raise complexity arbitrarily.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.12 — Stability under last-bit edits)</span></p>

There exists a constant $c$ such that, for every binary word, *adding*, *changing*, or *deleting* the last bit changes its Kolmogorov complexity by at most $\pm c$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.13 — Worst-case instability under one-bit edits)</span></p>

For every $n$, there exists a word $w$ and:

(i) a word $w^{+}$ obtained from $w$ by **inserting** one bit at some position with $C(w^{+}) > C(w) + n$;

(ii) a word $w^{c}$ obtained from $w$ by **changing** one bit at some position with $C(w^{c}) > C(w) + n$;

(iii) a word $w^{-}$ obtained from $w$ by **deleting** one bit at some position with $C(w^{-}) > C(w) + n$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $M$ be the Turing machine that on input $\mathrm{bin}(n)$ returns $(01)^{2^n}$. Set $w_n := (01)^{2^n}$. Then 

$$C_M(w_n) \le l(\mathrm{bin}(n)) \le \log(n)$$

hence 

$$C(w_n) \le \log(n) + c_M$$

For (i), define

$$W_n^{+} := \lbrace (01)^k 1 (01)^{2^n - k} : 0 \le k \le 2^n \rbrace,$$

a set of $2^n + 1$ distinct words, each obtained from $w_n$ by inserting one bit. By Proposition 2.3, at most $2^n - 1$ words have complexity $< n$, so there exists $w_n^{+} \in W_n^{+}$ with $C(w_n^{+}) \ge n$. Hence

$$C(w_n^{+}) - C(w_n) \ge n - (\log(n) + c_M) \xrightarrow[n \to \infty]{} \infty,$$

establishing (i). The proofs of (ii) and (iii) are analogous, using

$$W_n^{c} = \lbrace (01)^{k-1} 11 (01)^{2^n - k} : 1 \le k \le 2^n \rbrace, \qquad W_n^{-} = \lbrace (01)^{k-1} 1 (01)^{2^n - k} : 1 \le k \le 2^n \rbrace,$$

respectively.

</details>
</div>

<figure class="math-figure">
  <svg viewBox="0 0 620 320" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:620px" aria-label="One-bit-insertion family W_n^+ around the periodic word w_n">
    <g font-family="serif" font-size="12" fill="#1f2430">
      <text x="310" y="20" text-anchor="middle" font-weight="600">Insertion family W<tspan font-size="9" dy="-4">+</tspan><tspan dy="4">ₙ</tspan>: 2ⁿ + 1 candidates from a single base word</text>

      <!-- Base word w_n = (01)^{2^n}, here shown for small case (01)^4 -->
      <text x="20" y="56" font-weight="600">wₙ = (01)²ⁿ</text>
      <g font-family="monospace" font-size="14">
        <rect x="160" y="40" width="320" height="22" fill="rgba(60,120,40,0.10)" stroke="#3d7a26" stroke-width="1" />
        <text x="320" y="56" text-anchor="middle">0 1 0 1 0 1 0 1 ⋯ 0 1</text>
      </g>
      <text x="490" y="56" font-size="11" fill="#5b6270">  l(wₙ) = 2 · 2ⁿ</text>

      <!-- Annotated sample insertions: row by row, varying position k -->
      <g font-family="monospace" font-size="13">
        <!-- k = 1: insert "1" at the beginning ⇒ 1·(01)^{2^n} -->
        <text x="20" y="100">k = 1</text>
        <rect x="100" y="86" width="14" height="20" fill="#fff7e0" stroke="#a86f00" stroke-width="1.2" />
        <text x="107" y="101" text-anchor="middle" font-weight="700" fill="#a86f00">1</text>
        <text x="120" y="100">0 1 0 1 0 1 0 1 ⋯ 0 1</text>

        <!-- k = 2 -->
        <text x="20" y="130">k = 2</text>
        <text x="100" y="130" font-weight="700">0 1</text>
        <rect x="124" y="116" width="14" height="20" fill="#fff7e0" stroke="#a86f00" stroke-width="1.2" />
        <text x="131" y="131" text-anchor="middle" font-weight="700" fill="#a86f00">1</text>
        <text x="144" y="130">0 1 0 1 0 1 ⋯ 0 1</text>

        <!-- k = 3 -->
        <text x="20" y="160">k = 3</text>
        <text x="100" y="160" font-weight="700">0 1 0 1</text>
        <rect x="148" y="146" width="14" height="20" fill="#fff7e0" stroke="#a86f00" stroke-width="1.2" />
        <text x="155" y="161" text-anchor="middle" font-weight="700" fill="#a86f00">1</text>
        <text x="168" y="160">0 1 0 1 ⋯ 0 1</text>

        <!-- ... -->
        <text x="20" y="186">⋮</text>
        <text x="160" y="186">⋮</text>

        <!-- k = 2^n + 1: insert at the very end -->
        <text x="20" y="218">k = 2ⁿ + 1</text>
        <text x="100" y="218" font-weight="700">0 1 0 1 0 1 ⋯ 0 1</text>
        <rect x="304" y="204" width="14" height="20" fill="#fff7e0" stroke="#a86f00" stroke-width="1.2" />
        <text x="311" y="219" text-anchor="middle" font-weight="700" fill="#a86f00">1</text>
      </g>

      <!-- Key inequality -->
      <g>
        <line x1="40" y1="246" x2="580" y2="246" stroke="#cbd2e0" stroke-width="0.7" />
      </g>
      <text x="310" y="268" text-anchor="middle" font-size="13" fill="#1f2430">
        |Wₙ⁺| = 2ⁿ + 1   versus   |{w : C(w) &lt; n}| ≤ 2ⁿ − 1
      </text>
      <text x="310" y="292" text-anchor="middle" font-size="11" fill="#5b6270" font-style="italic">
        Pigeonhole: at least one wₙ⁺ ∈ Wₙ⁺ has complexity ≥ n, while C(wₙ) ≤ log(n) + cM.
      </text>
    </g>
  </svg>
  <figcaption>The construction behind Proposition 2.13(i): from the highly compressible base word $w_n = (01)^{2^n}$ (complexity $\le \log(n) + c_M$), inserting a single "1" at any of the $2^n + 1$ positions produces a family $W_n^{+}$. Since at most $2^n - 1$ words have Kolmogorov complexity $< n$, *some* member of $W_n^{+}$ has complexity $\ge n$ — a single inserted bit can therefore raise complexity by at least $n - \log(n) - c_M \to \infty$.</figcaption>
</figure>

The previous examples show that plain Kolmogorov complexity is stable under some very local edits, but can also be violently unstable under one-bit changes in the middle of a word. The next question is different: how much information is needed to describe a *concatenation* $xy$ if we already know how to describe $x$ and $y$ separately?

The main obstruction is parsing. If a program contains first a code for $x$ and then a code for $y$, the decoding machine must know where the first code ends. This forces an extra self-delimiting description of the length of the first code.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.14 — Upper bound for concatenation)</span></p>

There exists a constant $c$ such that, for all binary words $x$ and $y$,

$$C(xy) \le C(x) + 2 \log C(x) + C(y) + c.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $\tau_x$ and $\tau_y$ be optimal plain codes for $x$ and $y$, so

$$U(\tau_x) = x, \qquad l(\tau_x) = C(x), \qquad U(\tau_y) = y, \qquad l(\tau_y) = C(y).$$

Construct a Turing machine $M$ that expects inputs of the form

$$\mathrm{bin}(l(\tau_x))\,01\,\tau_x\tau_y.$$

The separator $01$ makes the binary description of $l(\tau_x)$ parseable. Once $M$ has recovered $l(\tau_x)$, it splits the remaining word into the first $l(\tau_x)$ bits and the rest:

$$\tau = \tau_1\tau_2, \qquad l(\tau_1) = l(\tau_x).$$

It then returns $U(\tau_1)U(\tau_2)$ whenever both computations halt. For the specific input $\mathrm{bin}(l(\tau_x))01\tau_x\tau_y$, this gives $xy$. Hence

$$
\begin{aligned}
C(xy)
&\le C_M(xy) + c_M \\
&\le l(\mathrm{bin}(l(\tau_x))) + 2 + l(\tau_x) + l(\tau_y) + c_M \\
&\le 2\log C(x) + C(x) + C(y) + c
\end{aligned}
$$

after absorbing fixed additive constants into $c$.

</details>
</div>

<figure class="math-figure">
  <svg viewBox="0 0 660 300" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:660px" aria-label="Plain concatenation requires an encoded boundary">
    <g font-family="serif" font-size="13" fill="#1f2430">
      <text x="330" y="24" text-anchor="middle" font-weight="600">Plain concatenation: the boundary must be encoded</text>

      <g>
        <rect x="35" y="58" width="160" height="42" fill="rgba(168,111,0,0.12)" stroke="#a86f00" stroke-width="1.5" />
        <text x="115" y="83" text-anchor="middle">bin(l(τₓ))</text>
        <text x="115" y="118" text-anchor="middle" font-size="11" fill="#5b6270">about log C(x) bits</text>
      </g>

      <g>
        <rect x="195" y="58" width="48" height="42" fill="rgba(239,68,68,0.14)" stroke="#b91c1c" stroke-width="1.5" />
        <text x="219" y="83" text-anchor="middle" font-family="monospace">01</text>
        <text x="219" y="118" text-anchor="middle" font-size="11" fill="#5b6270">separator</text>
      </g>

      <g>
        <rect x="243" y="58" width="170" height="42" fill="rgba(29,78,216,0.10)" stroke="#1d4ed8" stroke-width="1.5" />
        <text x="328" y="83" text-anchor="middle">τₓ</text>
        <text x="328" y="118" text-anchor="middle" font-size="11" fill="#5b6270">C(x) bits</text>
      </g>

      <g>
        <rect x="413" y="58" width="210" height="42" fill="rgba(18,177,124,0.10)" stroke="#0f9b6c" stroke-width="1.5" />
        <text x="518" y="83" text-anchor="middle">τᵧ</text>
        <text x="518" y="118" text-anchor="middle" font-size="11" fill="#5b6270">C(y) bits</text>
      </g>

      <g stroke="#5b6270" stroke-width="1.2" fill="none">
        <path d="M 328 105 C 328 140, 250 148, 250 176" />
        <path d="M 518 105 C 518 140, 410 148, 410 176" />
      </g>
      <polygon points="250,176 245,166 255,166" fill="#5b6270" />
      <polygon points="410,176 405,166 415,166" fill="#5b6270" />

      <g>
        <rect x="175" y="178" width="150" height="38" fill="rgba(29,78,216,0.10)" stroke="#1d4ed8" stroke-width="1.5" />
        <text x="250" y="202" text-anchor="middle">U(τₓ) = x</text>
        <rect x="335" y="178" width="150" height="38" fill="rgba(18,177,124,0.10)" stroke="#0f9b6c" stroke-width="1.5" />
        <text x="410" y="202" text-anchor="middle">U(τᵧ) = y</text>
      </g>

      <g stroke="#5b6270" stroke-width="1.2" fill="none">
        <path d="M 250 218 C 250 244, 298 248, 330 258" />
        <path d="M 410 218 C 410 244, 362 248, 330 258" />
      </g>
      <polygon points="330,258 322,250 334,247" fill="#5b6270" />

      <rect x="252" y="258" width="156" height="30" rx="2" fill="rgba(94,96,96,0.08)" stroke="#1f2430" stroke-width="1.2" />
      <text x="330" y="278" text-anchor="middle" font-weight="600">output xy</text>
    </g>
  </svg>
  <figcaption>Why Proposition 2.14 has an overhead term: a plain code $\tau_x\tau_y$ is not automatically parseable. The machine first reads a self-delimiting description of $l(\tau_x)$, then uses that length to split the rest into the code for $x$ and the code for $y$.</figcaption>
</figure>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(2.15 — Sharper iterated-log upper bound)</span></p>

For every fixed $n$, prove that there exists a constant $c_n$ such that, for all binary words $x$ and $y$,

$$
C(xy)
\le
C(x)
+ \log C(x)
+ \log\log C(x)
+ \cdots
+ \underbrace{\log \cdots \log}_{n-1} C(x)
+ 2\underbrace{\log \cdots \log}_{n} C(x)
+ C(y)
+ c_n.
$$

The idea is to encode the length of the first code more efficiently by repeatedly encoding lengths of lengths.

</div>

The preceding upper bound is not merely an artifact of a clumsy construction. Some overhead for parsing the first description is unavoidable.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.16 — Additive parsing overhead cannot be eliminated)</span></p>

For every $n$, there exist binary words $x$ and $y$ such that

$$C(xy) > C(x) + \log C(x) + C(y) + n.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof sketch</summary>

The proof reuses the computable sequence $S$ from the proof of Theorem 1.3. The machine $M_S$ enumerates special words $\sigma \in S$ which are relatively compressible compared with their length. In particular, for all such prefixes,

$$C(\sigma) \le l(\sigma) - \log l(\sigma) + O(1). \tag{5}$$

Choose a long word $w$ with high plain complexity, say $C(w) \ge l(w)$, which exists by the counting bound. Let $x$ be a long prefix of $w$ that belongs to $S$, and write $w = xy$. Then

$$C(y) \le l(y) + O(1),$$

and by (5),

$$C(x) + C(y) \le l(x) - \log l(x) + l(y) + O(1). \tag{6}$$

Since $xy = w$,

$$
\begin{aligned}
C(xy)
&= C(w) \\
&\ge l(w) \\
&= l(x) + l(y) \\
&= \bigl(l(x) - \log l(x)\bigr) + l(y) + \log l(x) \\
&\ge C(x) + C(y) + \log l(x) + O(1),
\end{aligned}
$$

where the last step uses (6). The intended conclusion is that the extra logarithmic overhead cannot be removed uniformly.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(A typo to check in Proposition 2.16)</span></p>

The PDF's final displayed asymptotic line in the proof of Proposition 2.16 reads like

$$\log l(x) - \log\bigl(l(x) - \log l(x)\bigr) \to \infty,$$

but this expression actually tends to $0$. The surrounding argument clearly aims to show that a logarithmic additive term is necessary, so this proof should be checked against the lecturer's intended version before relying on the final inequality as written.

</div>

### 2.3 Prefix-Free Turing Machines

Plain Kolmogorov complexity has an awkward concatenation behavior because plain programs are not self-delimiting. Prefix-free complexity fixes this by allowing only domains in which no valid program is a proper prefix of another valid program.

<figure class="math-figure">
  <svg viewBox="0 0 660 300" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:660px" aria-label="Prefix-free words as leaves of a binary tree">
    <g font-family="serif" font-size="12" fill="#1f2430">
      <text x="330" y="24" text-anchor="middle" font-weight="600">Prefix-free domains are antichains in the binary tree</text>

      <g stroke="#5b6270" stroke-width="1" fill="none">
        <line x1="330" y1="48" x2="210" y2="92" />
        <line x1="330" y1="48" x2="450" y2="92" />
        <line x1="210" y1="108" x2="135" y2="152" />
        <line x1="210" y1="108" x2="285" y2="152" />
        <line x1="450" y1="108" x2="375" y2="152" />
        <line x1="450" y1="108" x2="525" y2="152" />
        <line x1="135" y1="168" x2="95" y2="212" />
        <line x1="135" y1="168" x2="175" y2="212" />
        <line x1="285" y1="168" x2="245" y2="212" />
        <line x1="285" y1="168" x2="325" y2="212" />
      </g>

      <g>
        <circle cx="330" cy="42" r="15" fill="#fff7e0" stroke="#a86f00" />
        <text x="330" y="47" text-anchor="middle">λ</text>
        <circle cx="210" cy="100" r="15" fill="#fff7e0" stroke="#a86f00" />
        <text x="210" y="105" text-anchor="middle">0</text>
        <circle cx="450" cy="100" r="15" fill="#fff7e0" stroke="#a86f00" />
        <text x="450" y="105" text-anchor="middle">1</text>

        <circle cx="135" cy="160" r="16" fill="rgba(29,78,216,0.12)" stroke="#1d4ed8" stroke-width="2" />
        <text x="135" y="165" text-anchor="middle">00</text>
        <circle cx="285" cy="160" r="16" fill="#fff7e0" stroke="#a86f00" />
        <text x="285" y="165" text-anchor="middle">01</text>
        <circle cx="375" cy="160" r="16" fill="rgba(18,177,124,0.12)" stroke="#0f9b6c" stroke-width="2" />
        <text x="375" y="165" text-anchor="middle">10</text>
        <circle cx="525" cy="160" r="16" fill="rgba(29,78,216,0.12)" stroke="#1d4ed8" stroke-width="2" />
        <text x="525" y="165" text-anchor="middle">11</text>

        <circle cx="95" cy="220" r="17" fill="rgba(239,68,68,0.13)" stroke="#b91c1c" stroke-width="2" stroke-dasharray="4 3" />
        <text x="95" y="225" text-anchor="middle">000</text>
        <circle cx="175" cy="220" r="17" fill="rgba(239,68,68,0.13)" stroke="#b91c1c" stroke-width="2" stroke-dasharray="4 3" />
        <text x="175" y="225" text-anchor="middle">001</text>
        <circle cx="245" cy="220" r="17" fill="rgba(18,177,124,0.12)" stroke="#0f9b6c" stroke-width="2" />
        <text x="245" y="225" text-anchor="middle">010</text>
        <circle cx="325" cy="220" r="17" fill="#fff7e0" stroke="#a86f00" />
        <text x="325" y="225" text-anchor="middle">011</text>
      </g>

      <text x="330" y="265" text-anchor="middle" font-size="11" fill="#5b6270" font-style="italic">
        Valid programs may be leaves like 00, 10, 11, 010; descendants of a valid program, such as 000 and 001 below 00, are forbidden.
      </text>
    </g>
  </svg>
  <figcaption>A prefix-free domain is an antichain in the full binary tree: once a word is accepted as a program, no extension of it can also be accepted. This is the structural reason prefix-free programs are self-delimiting.</figcaption>
</figure>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(2.17 — Prefix and comparability)</span></p>

A word $x$ is a **prefix** of a word $w$, written $x \sqsubseteq w$, if there exists a word $y$ such that

$$w = xy.$$

It is a **proper prefix**, written $x \sqsubset w$, if $x \sqsubseteq w$ and $x \ne w$.

We also write

$$w \sim v \quad :\Longleftrightarrow \quad w \sqsubseteq v \text{ or } v \sqsubseteq w.$$

Thus $w \sim v$ means that the two words are prefix-comparable.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.18 — Basic prefix properties)</span></p>

The prefix relation $\sqsubseteq$ is reflexive and transitive.

The comparability relation $\sim$ is reflexive and transitive, and satisfies the cancellation property

$$vw \sim v'w' \Longrightarrow v \sim v'$$

for all words $v, v', w, w' \in \lbrace 0,1\rbrace^{\ast}$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Concatenation of prefix-free sets is a prefix-free set)</span></p>

Let $V_1,\dots,V_n$ be prefix-free sets. Show that the set

$$V := \lbrace v_1, \dots, v_n : v_i \in V_i \quad \forall i\in[n] \rbrace$$

is prefix-free.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Redefine the suggested definition:

$$V^n := \lbrace v_1, \dots, v_n : v_i \in V_i for all i\in[n] \rbrace$$

We will prove it by induction. 

**Base case:** $V^1 = V_1$ is a prefix-free set.

**Induction step:** Let $V^m$ is a prefix-free set, which is a concatentation of $m$ prefix-freee sets and we are appended a prefix-free set $V_{m+1}$. By contradiction, there are two **different** words $w_1,w_2\in V^{m+1}$ that wlog $w_1 \sqsubset w_2$. Consider their splits $w_1=s^1_1 s^1_2$ and $w_2=s^2_1 s^2_2$, where $s^2_1, s^2_1$ come from the set $V^m$, and $s^1_2, s^2_2$ are newly appended parts. Then the key observation is that is $w_1 \sim w_2$, then $s^1_1 \sim s^2_1$, because if none of them is a prefix of another, then appending new characters cannot resolve this incompetability of prefixes. Thus, $w_1 \sim w_2 \implies s^1_1 \sim s^2_1$, which can only be consistent with the assumption in the case of $s^1_1 = s^2_1$. Then either $s^1_1 = s^2_1$ (contradicts the way we chose the words $w_1\neq w_2$) or $s^1_1, s^2_1$ are not prefix-free, which is a contradiction, because $V_{m+1}$ is a prefix-free set.

</details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(2.19 — Prefix-free machines and prefix-free complexity)</span></p>

A Turing machine $M$ is **prefix-free** if its domain is prefix-free:

$$x,w \in \mathrm{dom}(M),\ x \sqsubset w \quad \text{never occurs}.$$

A Turing machine $\widetilde U$ is a **universal prefix-free machine** if it is prefix-free and, for every prefix-free machine $M$, there is a code $i_M$ such that

$$\widetilde U(i_M, w) = M(w)$$

for every binary word $w$.

After fixing a universal prefix-free machine $\widetilde U$, the **prefix-free Kolmogorov complexity** of $w$ is

$$K(w) := \min \lbrace l(\sigma) : \widetilde U(\sigma) = w \rbrace.$$

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(2.20 — Existence of a universal prefix-free machine)</span></p>

There exists a universal prefix-free Turing machine $\widetilde U$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof idea</summary>

The difficulty is that we cannot effectively decide whether an arbitrary machine $M_i$ is prefix-free. So instead of testing $M_i$ directly, we effectively transform each machine into a prefix-free submachine.

Given an index $i$, build a machine $M_{T(i)}$ by running

$$M_i(\lambda),\ M_i(0),\ M_i(1),\ M_i(00),\dots$$

in parallel. Whenever $M_i(w)$ halts at stage $s$, enumerate $(w, M_i(w))$ into the graph of $M_{T(i)}$ only if no distinct comparable input $v \sim w$ has already halted at an earlier stage.

This construction has two key properties:

* $M_{T(i)}$ is always prefix-free, because the first comparable conflict blocks the later one.
* If $M_i$ was already prefix-free, then no conflict ever appears, so $M_{T(i)} = M_i$.

Now define

$$\widetilde U(i,w) := M_{T(i)}(w).$$

If $M_i$ is prefix-free, then $\widetilde U(i,w) = M_i(w)$, so $\widetilde U$ simulates every prefix-free machine. Prefix-freeness of $\widetilde U$ follows from the cancellation property in Proposition 2.18 together with the prefix-freeness of each $M_{T(i)}$ and the self-delimiting choice of machine codes $i$.

</details>
</div>

The central technical advantage of prefix-free complexity is that valid programs can be concatenated without separately encoding the boundary between them. The first program ends exactly when the first prefix-free computation accepts.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(2.21 — Concatenation for prefix-free complexity)</span></p>

There exists a constant $c$ such that, for all binary words $v$ and $w$,

$$K(vw) \le K(v) + K(w) + c.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Construct a prefix-free machine $M$ which, on input $\sigma$, tries all splits

$$\sigma = \sigma_1\sigma_2$$

and runs $\widetilde U(\sigma_1)$ and $\widetilde U(\sigma_2)$ in parallel. If it finds a split for which both computations halt, say

$$\widetilde U(\sigma_1) = v, \qquad \widetilde U(\sigma_2) = w,$$

then $M(\sigma)$ returns $vw$.

Because $\widetilde U$ is prefix-free, at most one prefix $\sigma_1$ of a fixed input $\sigma$ can be a valid $\widetilde U$-program. This makes the split unambiguous.

The machine $M$ is prefix-free. Indeed, if $M(\sigma)$ halts using the split $\sigma = \sigma_1\sigma_2$, then for a proper extension $\sigma\tau$ the only possible first component is still $\sigma_1$. The second component would have to be $\sigma_2\tau$, but $\widetilde U(\sigma_2)$ already halts, contradicting prefix-freeness of $\widetilde U$.

Let $\sigma_v$ and $\sigma_w$ be optimal prefix-free codes for $v$ and $w$. Then

$$M(\sigma_v\sigma_w) = vw,$$

so

$$C_M(vw) \le l(\sigma_v\sigma_w) = K(v) + K(w).$$

Since $M$ is prefix-free, it is simulated by $\widetilde U$ with only a constant overhead:

$$K(vw) \le C_M(vw) + c_M \le K(v) + K(w) + c_M.$$

</details>
</div>

<figure class="math-figure">
  <svg viewBox="0 0 660 300" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:660px" aria-label="Prefix-free concatenation has an automatic boundary">
    <g font-family="serif" font-size="13" fill="#1f2430">
      <text x="330" y="24" text-anchor="middle" font-weight="600">Prefix-free concatenation: the first halt marks the boundary</text>

      <rect x="80" y="64" width="245" height="42" fill="rgba(29,78,216,0.10)" stroke="#1d4ed8" stroke-width="1.5" />
      <text x="202" y="89" text-anchor="middle">σᵥ</text>
      <text x="202" y="123" text-anchor="middle" font-size="11" fill="#5b6270">prefix-free code for v</text>

      <rect x="325" y="64" width="255" height="42" fill="rgba(18,177,124,0.10)" stroke="#0f9b6c" stroke-width="1.5" />
      <text x="452" y="89" text-anchor="middle">σ_w</text>
      <text x="452" y="123" text-anchor="middle" font-size="11" fill="#5b6270">remaining code for w</text>

      <line x1="325" y1="54" x2="325" y2="134" stroke="#b91c1c" stroke-width="2" stroke-dasharray="4 4" />
      <text x="325" y="150" text-anchor="middle" fill="#7f1d1d" font-size="12">unique split</text>

      <g stroke="#5b6270" stroke-width="1.2" fill="none">
        <path d="M 202 108 C 202 152, 240 164, 280 178" />
        <path d="M 452 108 C 452 152, 420 164, 380 178" />
      </g>
      <polygon points="280,178 270,172 281,168" fill="#5b6270" />
      <polygon points="380,178 389,172 378,168" fill="#5b6270" />

      <rect x="205" y="180" width="150" height="38" fill="rgba(29,78,216,0.10)" stroke="#1d4ed8" stroke-width="1.5" />
      <text x="280" y="204" text-anchor="middle">Ũ(σᵥ) = v</text>
      <rect x="365" y="180" width="150" height="38" fill="rgba(18,177,124,0.10)" stroke="#0f9b6c" stroke-width="1.5" />
      <text x="440" y="204" text-anchor="middle">Ũ(σ_w) = w</text>

      <g stroke="#5b6270" stroke-width="1.2" fill="none">
        <path d="M 280 220 C 280 245, 315 250, 350 258" />
        <path d="M 440 220 C 440 245, 395 250, 350 258" />
      </g>
      <polygon points="350,258 342,250 354,247" fill="#5b6270" />

      <rect x="272" y="258" width="156" height="30" rx="2" fill="rgba(94,96,96,0.08)" stroke="#1f2430" stroke-width="1.2" />
      <text x="350" y="278" text-anchor="middle" font-weight="600">output vw</text>

      <text x="330" y="282" text-anchor="middle" font-size="0">.</text>
    </g>
  </svg>
  <figcaption>Theorem 2.21 removes the plain-complexity parsing overhead. Since $\widetilde U$ is prefix-free, there is at most one prefix of the input on which $\widetilde U$ halts; that halting prefix is exactly the first code, so the rest is the second code.</figcaption>
</figure>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Connections between $C$ and $K$)</span></p>

Show the following upper bounds for the prefix-free Kolmogorov complexity:

1. $K(w) \leq l(w) + K(l(w)) + O(1) \qquad \forall w$

   *(Hint: construct a prefix-free Turing machine $M$ that, for the input $\sigma_w\text{bin}(w)$, where $\sigma_w$ is an optimal prefix-free code of $w$, returns $w$)*

2. $K(w) \leq C(w) + K(C(w)) + O(1) \qquad \forall w$

   *(Hint: modify the machine $M$ constructed above by computing some data from their optimal (non-prefix-free) codes)*

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

1. $K(w) \leq l(w) + K(l(w)) + O(1) \qquad \forall w$
  
   * **Construction:** Let $M$ be a TM that takes the input $\text{input} = \sigma_w \text{bin}(w)$. Then it tries all splits of the input $\text{input} = \tilde{\sigma}\tilde{w}$ and running a prefix-free UTM $\tilde{U}(\tilde{\sigma})$ in a dovetailing manner. When the prefix-free UTM halts with $\tilde{U}(\tilde{\sigma}) = \tilde{l}(w)$, we interpret it as a length of the word $w$. Then $M$ prints $\tilde{w}$ if the length of $\tilde{w}$ the same as $\tilde{l}(w)$, othwerwise enter infinity loop to not halt.
   * **Justification of construction:** The key observation that makes this simple approach work is that the split is unique, i.e. only one prefix of $\text{input}$ is a valid $\tilde{U}$ program, otherwise $\tilde{U}$ is not prefix-free. 
   * **Veirfication of complexity:** For the machine $M$ to induce an upper bound of $K$, we have to show that $M$ is prefix-free. $M$ is prefix-free, because spliting is deterministic, prefix $\tilde{\sigma}$ of the input is unqiue, prefix also determines the number of next chars to be present, so it cannot be a prefix of something that is bigger.
     
     $$
     \begin{\aligned}
     K(w) 
     &\leq C_M(w) + O(1) \\
     &\leq l(\sigma_w \text{bin}(w)) + O(1) \\
     &= \underbrace{l(\sigma_w)}_{K(l(w))} + l(\text{bin}(w)) + O(1) = K(l(w)) + l(\text{bin}(w)) + O(1)$$
     \end{\aligned}

2. $K(w) \leq C(w) + K(C(w)) + O(1) \qquad \forall w$

   Similarly to the previous approach, but now we encode the shortest description (or the shortest program) that produces the word $w$. The length of the shortest program that produces $w$ is $C(w)$, which is a plan Kolmogorov complexity. The justification and verification steps are very similar to the previous construction.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Comparison of plain and prefix-free KC)</span></p>

Both inequalities say the same thing: the *only* reason $K$ can exceed $C$ (or exceed $l$) is the cost of announcing where the payload ends, and that cost is itself just the prefix-free complexity of a single number. Iterating the idea on the header gives the standard refinements, e.g. 

$$K(w) \le l(w) + K(l(w)) + O(1) \le l(w) + \log l(w) + 2\log\log l(w) + O(1),$$

matching the logarithmic overhead that Proposition 2.27 shows is unavoidable. Part 2 also yields $K \le C + K(C) + O(1)$ as the prefix-free counterpart of the plain coding theorem: $K$ and $C$ agree up to a term that only logs the description length.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(2.22 — Computable processes do not increase $K$)</span></p>

For every computable function $f :\subseteq \lbrace 0,1\rbrace^{\ast} \to \lbrace 0,1\rbrace^{\ast}$, there exists a constant $c_f$ such that

$$K(f(w)) < K(w) + c_f$$

for every $w \in \mathrm{dom}(f)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $M$ be the machine that computes $f(\widetilde U(\sigma))$ from input $\sigma$. If $w \in \mathrm{dom}(f)$ and $\sigma_w$ is an optimal prefix-free code for $w$, then

$$M(\sigma_w) = f(\widetilde U(\sigma_w)) = f(w), \qquad l(\sigma_w) = K(w).$$

Moreover, $\mathrm{dom}(M) \subseteq \mathrm{dom}(\widetilde U)$, so $M$ is prefix-free. Therefore $\widetilde U$ simulates $M$ with constant overhead:

$$K(f(w)) \le C_M(f(w)) + c_M \le K(w) + c_M.$$

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.23 — Upper-semicomputability of $K$)</span></p>

The prefix-free Kolmogorov complexity $K$ is upper-semicomputable.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof idea</summary>

Run the universal prefix-free machine $\widetilde U$ for longer and longer finite times. Define

$$F(w,t) := \min \lbrace l(\sigma) : \widetilde U(\sigma)[t] \downarrow = w \rbrace.$$

As $t$ increases, the value $F(w,t)$ can only decrease, and it converges to $K(w)$. This is exactly the approximation characterization of upper-semicomputability from Proposition 2.9.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(2.24 — No computable unbounded lower bound for $K$)</span></p>

There exists no computable function $f :\subseteq \lbrace 0,1\rbrace^{\ast} \to \mathbb{N}$ such that:

* $f$ is unbounded;
* $K(w) > f(w)$ for every $w \in \mathrm{dom}(f)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Suppose such an $f$ exists. Construct a prefix-free machine $M$ with inputs of the form

$$0,\ 10,\ 110,\ 1110,\dots, 1^n0,\dots$$

On input $1^n0$, the machine runs $f(\lambda), f(0), f(1), f(00), \dots$ in parallel and returns the first word $\sigma$ with

$$f(\sigma) \downarrow > 2n.$$

The input set $\lbrace 1^n0 : n \in \mathbb{N} \rbrace$ is prefix-free, so $M$ is prefix-free. Since $f$ is unbounded, $M$ is total on this input set. By the assumed lower bound,

$$K(M(1^n0)) > 2n. \tag{11}$$

But $M$ is prefix-free and computable, so it is simulated by $\widetilde U$:

$$K(M(1^n0)) \le C_M(M(1^n0)) + c_M \le l(1^n0) + c_M = n + 1 + c_M. \tag{12}$$

For large $n$, (11) and (12) contradict each other.

</details>
</div>

Prefix-free complexity obeys a strong summability principle. The reason is geometric: a prefix-free set of words corresponds to a disjoint family of basic open cylinders in Cantor space.

<figure class="math-figure">
  <svg viewBox="0 0 660 290" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:660px" aria-label="Kraft inequality as disjoint cylinders">
    <g font-family="serif" font-size="12" fill="#1f2430">
      <text x="330" y="24" text-anchor="middle" font-weight="600">Kraft inequality: prefix-free cylinders fit inside measure 1</text>

      <rect x="40" y="62" width="580" height="34" fill="rgba(94,96,96,0.08)" stroke="#1f2430" stroke-width="1.2" />
      <text x="330" y="84" text-anchor="middle">Cantor space, total measure 1</text>

      <rect x="40" y="126" width="290" height="42" fill="rgba(29,78,216,0.12)" stroke="#1d4ed8" stroke-width="1.5" />
      <text x="185" y="151" text-anchor="middle">[[0]], measure 2⁻¹</text>

      <rect x="330" y="126" width="145" height="42" fill="rgba(18,177,124,0.12)" stroke="#0f9b6c" stroke-width="1.5" />
      <text x="402" y="151" text-anchor="middle">[[10]], 2⁻²</text>

      <rect x="475" y="126" width="72.5" height="42" fill="rgba(168,111,0,0.14)" stroke="#a86f00" stroke-width="1.5" />
      <text x="511" y="151" text-anchor="middle">[[110]]</text>
      <text x="511" y="166" text-anchor="middle" font-size="10">2⁻³</text>

      <rect x="547.5" y="126" width="36.25" height="42" fill="rgba(239,68,68,0.14)" stroke="#b91c1c" stroke-width="1.5" />
      <text x="566" y="151" text-anchor="middle" font-size="10">1110</text>
      <text x="566" y="166" text-anchor="middle" font-size="9">2⁻⁴</text>

      <rect x="583.75" y="126" width="36.25" height="42" fill="rgba(94,96,96,0.06)" stroke="#cbd2e0" stroke-width="1" />
      <text x="602" y="151" text-anchor="middle" font-size="10">free</text>

      <g stroke="#5b6270" stroke-width="1">
        <line x1="40" y1="116" x2="40" y2="178" />
        <line x1="330" y1="116" x2="330" y2="178" />
        <line x1="475" y1="116" x2="475" y2="178" />
        <line x1="547.5" y1="116" x2="547.5" y2="178" />
        <line x1="583.75" y1="116" x2="583.75" y2="178" />
        <line x1="620" y1="116" x2="620" y2="178" />
      </g>

      <text x="330" y="218" text-anchor="middle" font-size="13">
        2⁻¹ + 2⁻² + 2⁻³ + 2⁻⁴ + ··· ≤ 1
      </text>
      <text x="330" y="248" text-anchor="middle" font-size="11" fill="#5b6270" font-style="italic">
        Prefix-freeness means these cylinders are disjoint; a non-prefix-free domain would create nested, overlapping intervals.
      </text>
    </g>
  </svg>
  <figcaption>Proposition 2.25 as a measure statement. Every program $\tau$ occupies the cylinder $[\![\tau]\!]$ of measure $2^{-l(\tau)}$. Prefix-freeness makes these cylinders pairwise disjoint, so their total measure is at most $1$.</figcaption>
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.25 — Kraft inequality for prefix-free domains)</span></p>

For every prefix-free machine $M$,

$$\sum_{\tau \in \mathrm{dom}(M)} 2^{-l(\tau)} \le 1.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (skeleton)</summary>

**Induced cylinders are pairwise disjoint:** Each $\tau \in \mathrm{dom}(M)$ defines a cylinder and the key observation is that those cylinders are pairwise disjoint since the Turing machine $M$ is prefix-free.

**Sum of Lebesgue measures of pairwise disjoint cylinders:** The total Lebesgue measure of the set $S=([[\tau_0]], [[\tau_1]], \dots)$ is

$$\lambda(S) = \sum_i \lambda([[\tau_i]]) = \sum_i 2^{-l(\tau_i)} \leq 1$$

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof (full)</summary>

This is Proposition 2.25; it is what makes Chaitin's $\Omega=\sum_{M(\sigma)\downarrow}2^{-l(\sigma)}$ a genuine probability in $[0,1]$.
 
**Driving idea.** Each program $\sigma$ "claims" the dyadic block of width $2^{-l(\sigma)}$ consisting of the infinite sequences extending it. Prefix-freeness is exactly the statement that no claim sits inside another, so the blocks are disjoint and their total width cannot exceed $1$. I give the elementary tree-counting version first (no measure theory), then the one-line measure restatement.
 
Let $D:=\operatorname{dom}(M)=\lbrace\sigma:M(\sigma)\downarrow\rbrace$; by hypothesis $D$ is prefix-free.
 
**Elementary bound on finite subsets.** Let $F\subseteq D$ be finite and put $L:=\max_{\sigma\in F}l(\sigma)$. To each $\sigma\in F$ associate its set of length-$L$ extensions,

$$E(\sigma):=\lbrace\sigma\rho:\rho\in\lbrace 0,1\rbrace^{\,L-l(\sigma)}\rbrace,\qquad \sharp E(\sigma)=2^{\,L-l(\sigma)}.$$

**The sets $E(\sigma)$ are pairwise disjoint.** A string $u$ of length $L$ has, for each $k\le L$, exactly one prefix of length $k$; so if $u\in E(\sigma)\cap E(\tau)$ then both $\sigma$ and $\tau$ are prefixes of $u$, hence (being prefixes of a common word) prefix-comparable. Prefix-freeness of $D$ then forces $\sigma=\tau$. Since all $E(\sigma)$ live inside the $2^{L}$ strings of length $L$,

$$\sum_{\sigma\in F}2^{\,L-l(\sigma)}=\sum_{\sigma\in F}\sharp E(\sigma)=\sharp\!\!\bigcup_{\sigma\in F}E(\sigma)\;\le\;2^{L}.$$

Dividing by $2^L$ gives $\sum_{\sigma\in F}2^{-l(\sigma)}\le 1$.
 
**Passing to the full sum.** All terms $2^{-l(\sigma)}$ are non-negative, so the (possibly infinite) series equals the supremum of its finite partial sums:

$$\sum_{\sigma\in D}2^{-l(\sigma)}=\sup_{\substack{F\subseteq D\\ F\text{ finite}}}\ \sum_{\sigma\in F}2^{-l(\sigma)}\;\le\;1. \qquad\blacksquare$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Kraft inequality for prefix-free domains: what is really going on)</span></p>

Identify each $\sigma$ with the basic open cylinder $[[\sigma]]=\lbrace\sigma X:X\in\lbrace 0,1\rbrace^\omega\rbrace$ in Cantor space, of Lebesgue measure $\lambda([[\sigma]])=2^{-l(\sigma)}$. The disjointness argument above is precisely the statement that the cylinders $\lbrace[[\sigma]]:\sigma\in D\rbrace$ are pairwise disjoint: a common point $X\in[[\sigma]]\cap[[\tau]]$ would make $\sigma,\tau$ comparable prefixes of $X$, forcing $\sigma=\tau$. Countable additivity and monotonicity of $\lambda$ then give the inequality in one line,

$$\sum_{\sigma\in D}2^{-l(\sigma)}=\lambda\!\Big(\bigsqcup_{\sigma\in D}[\![\sigma]\!]\Big)\le\lambda(\lbrace 0,1\rbrace^\omega)=1 ,$$

which is the geometric content of Kraft's inequality. Equality holds iff the prefix-free set is *complete* — its cylinders cover Cantor space up to a null set, equivalently the program tree has no infinite "gaps". For a halting machine this typically fails, and the slack $1-\sum 2^{-l(\sigma)}$ is the probability that a random infinite input never triggers a halt.

</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(2.26 — Summability of prefix-free complexity)</span></p>

The prefix-free complexity satisfies

$$\sum_{\sigma \in \lbrace 0,1\rbrace^{\ast}} 2^{-K(\sigma)} \le 1.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

For every word $\sigma$, choose an optimal prefix-free code $\tau_\sigma$ with

$$\widetilde U(\tau_\sigma) = \sigma, \qquad l(\tau_\sigma) = K(\sigma).$$

The set $\lbrace \tau_\sigma : \sigma \in \lbrace 0,1\rbrace^{\ast} \rbrace$ is a subset of $\mathrm{dom}(\widetilde U)$, hence prefix-free. Therefore, by Kraft's inequality,

$$
\sum_{\sigma \in \lbrace 0,1\rbrace^{\ast}} 2^{-K(\sigma)}
=
\sum_{\sigma \in \lbrace 0,1\rbrace^{\ast}} 2^{-l(\tau_\sigma)}
\le
\sum_{\tau \in \mathrm{dom}(\widetilde U)} 2^{-l(\tau)}
\le 1.
$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(2.25 vs 2.26)</span></p>

* In 2.25 we consider the **inputs** of the prefix-free TM.
* In 2.26 we consider the **outputs** of the prefix-free TM.
  * outputs are redicuble to inputs, so we use the result from 2.25 to prove 2.26.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.27 — Prefix-free complexity can exceed length by logarithmic overhead)</span></p>

For every fixed constant $c$, there exists a word $w$ such that

$$K(w) > l(w) + \log l(w) + c.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Suppose, toward a contradiction, that there is a constant $d$ such that

$$K(w) \le l(w) + \log l(w) + d$$

for every nonempty word $w$. Then

$$
\begin{aligned}
1
&\ge \sum_{w \in \lbrace 0,1\rbrace^{\ast}} 2^{-K(w)} \\
&\ge \sum_{w \in \lbrace 0,1\rbrace^{\ast}} 2^{-l(w)-\log l(w)-d} \\
&= 2^{-d} \sum_{n \ge 1} \sum_{l(w)=n} 2^{-n}\frac{1}{n} \\
&= 2^{-d} \sum_{n \ge 1} 2^n \cdot 2^{-n}\frac{1}{n} \\
&= 2^{-d} \sum_{n \ge 1} \frac{1}{n}
= \infty,
\end{aligned}
$$

contradicting Theorem 2.26. Hence no fixed $d$ can bound all $K(w)$ by $l(w)+\log l(w)+d$.

</details>
</div>

The next result is the constructive converse to Kraft's inequality: if a computably enumerable list of requested code lengths satisfies the Kraft sum bound, then one can effectively assign prefix-free codewords of exactly those lengths.

<figure class="math-figure">
  <svg viewBox="0 0 720 520" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:720px" aria-label="Kraft-Chaitin theorem as allocation of requested code lengths into prefix-free codewords">
    <defs>
      <marker id="arrow-kc-flow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
        <path d="M 0 0 L 8 4 L 0 8 Z" fill="#5b6270" />
      </marker>
    </defs>
    <g font-family="serif" font-size="12" fill="#1f2430">
      <text x="360" y="24" text-anchor="middle" font-weight="600">Kraft-Chaitin theorem: length requests become prefix-free descriptions</text>

      <!-- Request list -->
      <rect x="35" y="54" width="190" height="210" rx="6" fill="rgba(29,78,216,0.06)" stroke="#1d4ed8" stroke-width="1.3" />
      <text x="130" y="80" text-anchor="middle" font-weight="600">c.e. request stream</text>
      <text x="130" y="100" text-anchor="middle" font-size="11" fill="#5b6270">each row asks for length dᵢ</text>
      <g>
        <rect x="58" y="118" width="144" height="22" rx="3" fill="#ffffff" stroke="#cbd2e0" />
        <text x="130" y="134" text-anchor="middle">(d₀=2, τ₀)</text>
        <rect x="58" y="145" width="144" height="22" rx="3" fill="#ffffff" stroke="#cbd2e0" />
        <text x="130" y="161" text-anchor="middle">(d₁=3, τ₁)</text>
        <rect x="58" y="172" width="144" height="22" rx="3" fill="#ffffff" stroke="#cbd2e0" />
        <text x="130" y="188" text-anchor="middle">(d₂=4, τ₂)</text>
        <rect x="58" y="199" width="144" height="22" rx="3" fill="#ffffff" stroke="#cbd2e0" />
        <text x="130" y="215" text-anchor="middle">(d₃=4, τ₃)</text>
      </g>
      <text x="130" y="244" text-anchor="middle" font-size="12">Σ 2^-dᵢ ≤ 1</text>
      <text x="130" y="256" text-anchor="middle" font-size="10" fill="#5b6270">1/4 + 1/8 + 1/16 + 1/16 = 1/2</text>

      <!-- Arrow to interval -->
      <path d="M 230 145 L 288 145" stroke="#5b6270" stroke-width="1.4" marker-end="url(#arrow-kc-flow)" />
      <text x="259" y="128" text-anchor="middle" font-size="11" fill="#5b6270">allocate</text>

      <!-- Dyadic interval allocation -->
      <rect x="300" y="54" width="385" height="160" rx="6" fill="rgba(168,111,0,0.07)" stroke="#a86f00" stroke-width="1.3" />
      <text x="492" y="80" text-anchor="middle" font-weight="600">dyadic space [0,1]</text>
      <text x="492" y="100" text-anchor="middle" font-size="11" fill="#5b6270">requested lengths are interval widths 2^-dᵢ</text>
      <rect x="330" y="122" width="320" height="34" fill="rgba(94,96,96,0.08)" stroke="#1f2430" stroke-width="1" />
      <rect x="330" y="122" width="80" height="34" fill="rgba(29,78,216,0.14)" stroke="#1d4ed8" stroke-width="1.4" />
      <text x="370" y="144" text-anchor="middle">00</text>
      <rect x="410" y="122" width="40" height="34" fill="rgba(214,83,54,0.14)" stroke="#d65336" stroke-width="1.4" />
      <text x="430" y="144" text-anchor="middle">010</text>
      <rect x="450" y="122" width="20" height="34" fill="rgba(18,177,124,0.16)" stroke="#0f9b6c" stroke-width="1.4" />
      <text x="460" y="144" text-anchor="middle" font-size="10">0110</text>
      <rect x="470" y="122" width="20" height="34" fill="rgba(168,111,0,0.18)" stroke="#a86f00" stroke-width="1.4" />
      <text x="480" y="144" text-anchor="middle" font-size="10">0111</text>
      <text x="570" y="144" text-anchor="middle" fill="#5b6270">free remainder</text>
      <line x1="330" y1="176" x2="650" y2="176" stroke="#444" stroke-width="1.1" />
      <g stroke="#444">
        <line x1="330" y1="170" x2="330" y2="182" />
        <line x1="410" y1="170" x2="410" y2="182" />
        <line x1="450" y1="170" x2="450" y2="182" />
        <line x1="490" y1="170" x2="490" y2="182" />
        <line x1="650" y1="170" x2="650" y2="182" />
      </g>
      <g text-anchor="middle" font-size="10" fill="#444">
        <text x="330" y="196">0</text>
        <text x="410" y="196">1/4</text>
        <text x="450" y="196">3/8</text>
        <text x="490" y="196">1/2</text>
        <text x="650" y="196">1</text>
      </g>

      <!-- Arrow down -->
      <path d="M 492 218 L 492 263" stroke="#5b6270" stroke-width="1.4" marker-end="url(#arrow-kc-flow)" />
      <text x="545" y="244" text-anchor="middle" font-size="11" fill="#5b6270">read intervals as words</text>

      <!-- Binary tree -->
      <rect x="300" y="275" width="385" height="180" rx="6" fill="rgba(18,177,124,0.06)" stroke="#0f9b6c" stroke-width="1.3" />
      <text x="492" y="301" text-anchor="middle" font-weight="600">prefix-free code tree</text>
      <g stroke="#5b6270" stroke-width="1.1" fill="none">
        <line x1="492" y1="324" x2="410" y2="360" />
        <line x1="492" y1="324" x2="574" y2="360" />
        <line x1="410" y1="360" x2="365" y2="398" />
        <line x1="410" y1="360" x2="455" y2="398" />
        <line x1="455" y1="398" x2="430" y2="430" />
        <line x1="455" y1="398" x2="480" y2="430" />
        <line x1="480" y1="430" x2="465" y2="456" />
        <line x1="480" y1="430" x2="500" y2="456" />
      </g>
      <g text-anchor="middle">
        <circle cx="492" cy="320" r="11" fill="#fff7e0" stroke="#a86f00" />
        <text x="492" y="324">λ</text>
        <circle cx="410" cy="360" r="11" fill="#fff7e0" stroke="#a86f00" />
        <text x="410" y="364">0</text>
        <circle cx="574" cy="360" r="11" fill="#fff7e0" stroke="#a86f00" />
        <text x="574" y="364">1</text>
        <circle cx="365" cy="398" r="14" fill="rgba(29,78,216,0.14)" stroke="#1d4ed8" />
        <text x="365" y="402">00</text>
        <circle cx="455" cy="398" r="11" fill="#fff7e0" stroke="#a86f00" />
        <text x="455" y="402">01</text>
        <circle cx="430" cy="430" r="15" fill="rgba(214,83,54,0.14)" stroke="#d65336" />
        <text x="430" y="434">010</text>
        <circle cx="480" cy="430" r="13" fill="#fff7e0" stroke="#a86f00" />
        <text x="480" y="434">011</text>
        <circle cx="465" cy="456" r="16" fill="rgba(18,177,124,0.16)" stroke="#0f9b6c" />
        <text x="465" y="460" font-size="10">0110</text>
        <circle cx="500" cy="456" r="16" fill="rgba(168,111,0,0.18)" stroke="#a86f00" />
        <text x="500" y="460" font-size="10">0111</text>
      </g>
      <text x="590" y="403" font-size="11" fill="#5b6270">selected nodes are leaves</text>
      <text x="590" y="421" font-size="11" fill="#5b6270">so no code is a prefix</text>
      <text x="590" y="439" font-size="11" fill="#5b6270">of another code</text>

      <!-- Machine map -->
      <rect x="35" y="305" width="190" height="145" rx="6" fill="rgba(214,83,54,0.06)" stroke="#d65336" stroke-width="1.3" />
      <text x="130" y="331" text-anchor="middle" font-weight="600">prefix-free machine M</text>
      <g font-size="12">
        <text x="65" y="358">M(00) = τ₀</text>
        <text x="65" y="382">M(010) = τ₁</text>
        <text x="65" y="406">M(0110) = τ₂</text>
        <text x="65" y="430">M(0111) = τ₃</text>
      </g>
      <path d="M 230 365 L 288 365" stroke="#5b6270" stroke-width="1.4" marker-end="url(#arrow-kc-flow)" />
      <text x="259" y="348" text-anchor="middle" font-size="11" fill="#5b6270">domain</text>

      <text x="360" y="492" text-anchor="middle" font-size="11" fill="#5b6270" font-style="italic">
        The theorem says this allocation can be done effectively for every computably enumerable request list with Kraft sum at most 1.
      </text>
    </g>
  </svg>
  <figcaption>The Kraft-Chaitin theorem is the constructive converse to Kraft's inequality. A request $(d_i,\tau_i)$ asks for a code cylinder of measure $2^{-d_i}$; the Kraft bound says all requested cylinders fit inside Cantor space. The construction places them as disjoint dyadic blocks, reads their addresses as codewords $\sigma_i$, and defines a prefix-free machine by $M(\sigma_i)=\tau_i$.</figcaption>
</figure>

<figure class="math-figure">
  <svg viewBox="0 0 660 330" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:660px" aria-label="Kraft-Chaitin allocation by splitting free placeholders">
    <g font-family="serif" font-size="12" fill="#1f2430">
      <text x="330" y="24" text-anchor="middle" font-weight="600">Kraft-Chaitin allocation: requests consume free dyadic space</text>

      <text x="45" y="58" font-weight="600">Before request dₛ = 4</text>
      <rect x="45" y="72" width="560" height="34" fill="rgba(94,96,96,0.07)" stroke="#1f2430" stroke-width="1" />
      <rect x="45" y="72" width="140" height="34" fill="rgba(29,78,216,0.12)" stroke="#1d4ed8" stroke-width="1.3" />
      <text x="115" y="94" text-anchor="middle">used</text>
      <rect x="185" y="72" width="140" height="34" fill="rgba(18,177,124,0.12)" stroke="#0f9b6c" stroke-width="1.5" />
      <text x="255" y="94" text-anchor="middle">placeholder v₂</text>
      <rect x="325" y="72" width="70" height="34" fill="rgba(168,111,0,0.14)" stroke="#a86f00" stroke-width="1.5" />
      <text x="360" y="94" text-anchor="middle">v₃</text>
      <rect x="395" y="72" width="35" height="34" fill="rgba(168,111,0,0.10)" stroke="#a86f00" stroke-width="1.2" />
      <text x="412" y="94" text-anchor="middle">v₄</text>
      <text x="500" y="94" text-anchor="middle" fill="#5b6270">free tail</text>

      <g stroke="#5b6270" stroke-width="1.2" fill="none">
        <path d="M 255 112 C 255 142, 225 150, 210 172" />
        <path d="M 255 112 C 255 142, 285 150, 300 172" />
      </g>
      <polygon points="210,172 207,162 217,166" fill="#5b6270" />
      <polygon points="300,172 292,166 302,162" fill="#5b6270" />
      <text x="255" y="150" text-anchor="middle" font-size="11" fill="#5b6270">split v₂ into descendants</text>

      <text x="45" y="202" font-weight="600">After assigning σₛ of length 4</text>
      <rect x="45" y="216" width="560" height="34" fill="rgba(94,96,96,0.07)" stroke="#1f2430" stroke-width="1" />
      <rect x="45" y="216" width="140" height="34" fill="rgba(29,78,216,0.12)" stroke="#1d4ed8" stroke-width="1.3" />
      <text x="115" y="238" text-anchor="middle">used</text>
      <rect x="185" y="216" width="35" height="34" fill="rgba(239,68,68,0.14)" stroke="#b91c1c" stroke-width="1.7" />
      <text x="202" y="238" text-anchor="middle">σₛ</text>
      <rect x="220" y="216" width="35" height="34" fill="rgba(18,177,124,0.12)" stroke="#0f9b6c" stroke-width="1.5" />
      <text x="237" y="238" text-anchor="middle">v₄</text>
      <rect x="255" y="216" width="70" height="34" fill="rgba(18,177,124,0.12)" stroke="#0f9b6c" stroke-width="1.5" />
      <text x="290" y="238" text-anchor="middle">v₃</text>
      <rect x="325" y="216" width="70" height="34" fill="rgba(168,111,0,0.14)" stroke="#a86f00" stroke-width="1.5" />
      <text x="360" y="238" text-anchor="middle">v₃</text>
      <rect x="395" y="216" width="35" height="34" fill="rgba(168,111,0,0.10)" stroke="#a86f00" stroke-width="1.2" />
      <text x="412" y="238" text-anchor="middle">v₄</text>
      <text x="500" y="238" text-anchor="middle" fill="#5b6270">free tail</text>

      <text x="330" y="292" text-anchor="middle" font-size="11" fill="#5b6270" font-style="italic">
        A larger free placeholder can be subdivided until one descendant has exactly the requested length.
      </text>
    </g>
  </svg>
  <figcaption>The local step in the Kraft-Chaitin construction. The remainder $R_s$ is represented by available placeholders. When a request of length $d_s$ arrives, the algorithm either uses an existing length-$d_s$ placeholder or splits a larger free placeholder into descendants, assigning one codeword and keeping the rest for later requests.</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(2.28 — Kraft-Chaitin Theorem)</span></p>

Let

$$(d_0,\tau_0), (d_1,\tau_1), \dots$$

be a computably enumerable sequence of pairs with $d_i \in \mathbb{N}$ and $\tau_i \in \lbrace 0,1\rbrace^{\ast}$ such that

$$\sum_{i \in \mathbb{N}} 2^{-d_i} \le 1. \tag{13}$$

Then there exists a computable prefix-free machine $M$ and codewords $\sigma_i$ such that

$$l(\sigma_i) = d_i, \qquad M(\sigma_i) = \tau_i$$

for all $i$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof idea</summary>

Think of each requested codeword of length $d_i$ as asking for a dyadic interval of measure $2^{-d_i}$. Condition (13) says that the total requested measure fits inside the unit interval. The proof gives an effective way to place these intervals so that none is nested inside another.

At stage $s$, after assigning $\sigma_0,\dots,\sigma_{s-1}$, keep track of the remaining free measure

$$R_s = 1 - \sum_{i<s} 2^{-d_i}.$$

The binary expansion of $R_s$ records which placeholder intervals are still available. For every position $n$ where the $n$-th bit of $R_s$ is $1$, maintain a placeholder word $v_n^{(s)}$ of length $n$. The invariant is:

* $l(v_n^{(s)}) = n$ whenever $v_n^{(s)}$ is defined;
* the already assigned codes together with all placeholders form a prefix-free set.

For the next request $(d_s,\tau_s)$, there are two cases.

If the bit $R_s(d_s)$ is $1$, use the existing placeholder:

$$\sigma_s := v_{d_s}^{(s)}, \qquad M(\sigma_s) := \tau_s.$$

This subtracts $2^{-d_s}$ from the remaining measure and preserves the invariant.

If $R_s(d_s)=0$, take the largest $d' < d_s$ with $R_s(d')=1$. Such a $d'$ must exist, otherwise the Kraft bound would already be violated. Split the larger placeholder $v_{d'}^{(s)}$ into smaller descendants, choose one descendant of length $d_s$ as $\sigma_s$, and keep the other descendants as new placeholders:

$$\sigma_s := v_{d'}^{(s)}0^{d_s-d'}, \qquad M(\sigma_s) := \tau_s.$$

The remaining descendants

$$v_{d'}^{(s)}0,\ v_{d'}^{(s)}00,\ \dots,\ v_{d'}^{(s)}0^{d_s-d'}$$

are assigned to the positions whose bits in the new remainder become $1$. Each update preserves prefix-freeness because all new words live below the single old placeholder that was removed.

Proceeding effectively through the computably enumerable request list defines the desired computable prefix-free machine.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Functional upper bound for $K$)</span></p>

Let $f:\lbrace 0,1 \rbrace^\ast \to \mathbb{N}$ be a computable function that fulfills

$$\sum_{w \in \lbrace 0,1 \rbrace^\ast} 2^{-f(w)} \le \infty$$

Show that there exists a constant $c$ such that 

$$K(w) \leq f(w) + c \quad \forall w \in \lbrace 0,1 \rbrace^\ast$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**Normalize the finite sum:** Let the constant $\tilde{c}$ be such that $\sum_{w \in \lbrace 0,1 \rbrace^\ast} 2^{-f(w) - \tilde{c}} \leq 1$. Define $d(w) = f(w) + \tilde{c}$. Note that $d(w)$ is a computable function.

**Applying Kraft-Chaitin Thm:** For each pair $(d(w), w)$ we have a code $\sigma$ such that $l(\sigma) = d(w)$ and a prefix-free TM $M(\sigma) = w$. Then

$$K(w) \leq l(\sigma) + c_M = d(w) + c_M = f(w) + \underbrace{\tilde{c} + c_M}_{:= c} = f(w) + c,$$

since $M$ is prefix-free, $\widetilde U$ simulates it with constant overhead $c_M$, giving $K(w) \le C_M(w) + c_M \le l(\sigma) + c_M$.

*Remark:* $K(w) \le C_M(w) + c_M$ is valid, because $M$ is prefix-free and C_M(w) is *some* prefix-free code for $w$ (I believe it would be better to say $K_M$).

</details>
</div>

Returning to the Kraft–Chaitin theorem, we record a name and a complexity bound for the sequences satisfying its hypothesis. From now on we call any computably enumerable sequence of pairs $(d\_i,\tau\_i)$ obeying $\sum\_i 2^{-d\_i}\le 1$ a **KC set**.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(2.28.1 — KC sets bound $K$)</span></p>

Let $(d\_0,\tau\_0),(d\_1,\tau\_1),\dots$ be a KC set (a sequence fulfilling the requirements of Theorem 2.28). Then

$$K(\tau_i) \le d_i + O(1) \qquad \text{for all } i.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

For the prefix-free Turing machine $M$ constructed in the proof of Theorem 2.28 we have $l(\sigma\_i)=d\_i$ and $M(\sigma\_i)=\tau\_i$. Since $\widetilde U$ simulates $M$ with constant overhead,

$$K(\tau_i) \le C_M(\tau_i) + O(1) \le d_i + O(1).$$

</details>
</div>

We close the discussion of prefix-free complexity with an upper bound on $K(\sigma)$ in terms of the length $l(\sigma)$ together with the cost of describing that length.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(2.29 — Chaitin)</span></p>

$$K(\sigma) \le l(\sigma) + K(l(\sigma)) + O(1).$$

Here $K(l(\sigma))$ abbreviates $K(\mathrm{bin}(l(\sigma)))$, the prefix-free complexity of the binary encoding of the length.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Deferred to Sheet 2, Exercise 4.1.

</details>
</div>

## Information Content Measures

<div class="math-callout math-callout--info" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Motivation</span><span class="math-callout__name">(Prefix-free complexity is the "smallest" function behaving as complexity measure)</span></p>

Theorem 3.6 below will show that prefix-free complexity is, up to an additive constant, the **smallest** member of a whole class of functions that behave like a complexity measure. We first isolate the two defining properties of that class.

TODO: what does the smallest mean in this context? The most loose in definition?

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(3.1 — Information content measure)</span></p>

An **information content measure** is a partial function $F :\subseteq \lbrace 0,1\rbrace^{\ast} \to \mathbb{N}$ that satisfies

$$\sum_{\sigma \in \operatorname{dom}(F)} 2^{-F(\sigma)} \le 1 \tag{19}$$

and

$$\text{the set } \lbrace (\sigma, k) : F(\sigma) \le k\rbrace \text{ is c.e.} \tag{20}$$

We call the set in (20) the **epigraph** of $F$; condition (19) is the Kraft-type summability bound.

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Information content measure is not necessarily about prefix-free TMs)</span></p>

The definition of **information content measure** says nothing about the prefix-free TMs.

</div>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/algorithmic_randomness_computable_analysis/arca_epigraph_kraft.png' | relative_url }}" alt="Left: lattice plot of the c.e. epigraph of a partial function F defined on the words 0, 10, 001 with values 2, 4, 5; each domain word carries a shaded column of pairs (sigma, k) with k at least F(sigma), the least point of each column ringed as the value F(sigma), and stage annotations show the pair (001,7) being enumerated before the better pair (001,5). Right: a horizontal bar from 0 to 1 in which the masses 2^-2, 2^-4, 2^-5 of the three domain words fill 11/32 of the unit Kraft budget, the rest unused." loading="lazy">
  <figcaption>The definition has a computability side and a measure side, here for a partial $F$ with $\mathrm{dom}(F) = \lbrace 0, 10, 001\rbrace$ and $F(0)=2$, $F(10)=4$, $F(001)=5$. Left: the epigraph is the set of pairs $(\sigma, k)$ lying on or above the graph of $F$ — a shaded column over each domain word whose least element is the value $F(\sigma)$, and an empty column over every word outside the domain. Being c.e., these pairs are discovered one at a time, so a better certificate like $(001, 5)$ may appear after $(001, 7)$; this is exactly upper-semicomputability of $F$. Right: each domain word spends mass $2^{-F(\sigma)}$, and the Kraft bound asks the total to fit inside measure $1$, just as prefix-free code cylinders do. Note $\mathrm{dom}(F)$ is not even prefix-free ($0 \prec 001$) — the definition never asks for that.</figcaption>
</figure>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(Three information content measures)</span></p>

* The function $f :\subseteq \lbrace 0,1\rbrace^{\ast} \to \mathbb{N}$ defined by

  $$f(0^n 1) = n+1 \qquad \text{for all } n \ge 0$$

  is partially computable, so its epigraph is computably enumerable and (20) holds. Moreover

  $$\sum_{\sigma \in \operatorname{dom}(f)} 2^{-f(\sigma)} = \sum_{n \in \mathbb{N}} 2^{-l(0^n 1)} = \sum_{n \in \mathbb{N}} 2^{-(n+1)} = 1,$$

  so (19) holds as well.

* The prefix-free Kolmogorov complexity $K$ is an information content measure: it satisfies (19) by Theorem 2.26 and (20) by Proposition 2.23.

* The length functions on the domain of prefix-free TMs are information content measures.

</div>

<figure class="math-figure">
  <svg viewBox="0 0 650 310" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:650px" aria-label="The example f(0^n1)=n+1 as a complete Kraft allocation">
    <g font-family="serif" font-size="12" fill="#1f2430">
      <text x="325" y="24" text-anchor="middle" font-weight="600">Example f(0^n1) = n+1: a complete geometric budget</text>

      <rect x="55" y="58" width="540" height="42" fill="rgba(94,96,96,0.08)" stroke="#1f2430" stroke-width="1" />
      <rect x="325" y="58" width="270" height="42" fill="rgba(214,83,54,0.13)" stroke="#d65336" stroke-width="1.5" />
      <text x="460" y="83" text-anchor="middle">[[1]], f=1, mass 1/2</text>
      <rect x="190" y="58" width="135" height="42" fill="rgba(29,78,216,0.12)" stroke="#1d4ed8" stroke-width="1.5" />
      <text x="257" y="83" text-anchor="middle">[[01]], 1/4</text>
      <rect x="122.5" y="58" width="67.5" height="42" fill="rgba(18,177,124,0.13)" stroke="#0f9b6c" stroke-width="1.5" />
      <text x="156" y="83" text-anchor="middle">[[001]], 1/8</text>
      <rect x="88.75" y="58" width="33.75" height="42" fill="rgba(168,111,0,0.14)" stroke="#a86f00" stroke-width="1.5" />
      <text x="105" y="83" text-anchor="middle" font-size="10">1/16</text>
      <rect x="55" y="58" width="33.75" height="42" fill="rgba(168,111,0,0.08)" stroke="#a86f00" stroke-width="1.2" />
      <text x="72" y="83" text-anchor="middle" font-size="10">...</text>

      <line x1="55" y1="130" x2="595" y2="130" stroke="#444" stroke-width="1.1" />
      <g stroke="#444">
        <line x1="55" y1="124" x2="55" y2="136" />
        <line x1="325" y1="124" x2="325" y2="136" />
        <line x1="595" y1="124" x2="595" y2="136" />
      </g>
      <text x="55" y="152" text-anchor="middle" font-size="11">0</text>
      <text x="325" y="152" text-anchor="middle" font-size="11">1/2</text>
      <text x="595" y="152" text-anchor="middle" font-size="11">1</text>

      <g stroke="#5b6270" stroke-width="1.2" fill="none">
        <line x1="325" y1="195" x2="245" y2="235" />
        <line x1="325" y1="195" x2="405" y2="235" />
        <line x1="245" y1="235" x2="205" y2="273" />
        <line x1="245" y1="235" x2="285" y2="273" />
      </g>
      <g>
        <circle cx="325" cy="190" r="14" fill="#fff7e0" stroke="#a86f00" />
        <text x="325" y="194" text-anchor="middle">λ</text>
        <circle cx="245" cy="235" r="14" fill="#fff7e0" stroke="#a86f00" />
        <text x="245" y="239" text-anchor="middle">0</text>
        <circle cx="405" cy="235" r="14" fill="rgba(214,83,54,0.13)" stroke="#d65336" />
        <text x="405" y="239" text-anchor="middle">1</text>
        <circle cx="205" cy="273" r="14" fill="#fff7e0" stroke="#a86f00" />
        <text x="205" y="277" text-anchor="middle">00</text>
        <circle cx="285" cy="273" r="14" fill="rgba(29,78,216,0.12)" stroke="#1d4ed8" />
        <text x="285" y="277" text-anchor="middle">01</text>
      </g>
      <text x="445" y="239" font-size="11" fill="#5b6270">leaves 1, 01, 001, ...</text>
      <text x="445" y="257" font-size="11" fill="#5b6270">form a prefix-free set</text>
      <text x="445" y="275" font-size="11" fill="#5b6270">their cylinder masses add to 1</text>
    </g>
  </svg>
  <figcaption>The domain $\lbrace 0^n1 : n \ge 0\rbrace$ is prefix-free. Assigning $f(0^n1)=n+1=l(0^n1)$ makes the Kraft sum exactly the geometric series $1/2+1/4+1/8+\cdots=1$.</figcaption>
</figure>

The next theorem is the engine of the whole section: the class of information content measures can itself be effectively listed.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(3.2 — Enumeration of all information content measures)</span></p>

There exists a computable enumeration $F\_0, F\_1, \dots$ of all information content measures.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $E\_0, E\_1, \dots$, where $E\_i := \big((\sigma^i\_0,k\_0),(\sigma^i\_1,k\_1),\dots\big)$, be a computable enumeration of all c.e. epigraphs of the corresponding upper-semicomputable partial functions $\widetilde F\_0, \widetilde F\_1, \dots$ from $\lbrace 0,1\rbrace^{\ast}$ to $\mathbb{N}$. Each $\widetilde F\_i$ automatically satisfies (20), but not necessarily (19); the construction below "trims" each enumeration so that the Kraft bound can never be violated.

We build a machine $M$ that, given an index $i$, returns a (possibly finite) sequence of pairs $(\sigma\_0,k\_0),(\sigma\_1,k\_1),\dots$ in $\lbrace 0,1\rbrace^{\ast} \times \mathbb{N}$. For every $n$, the pair $(\sigma\_n,k\_n)$ is defined and equal to $(\sigma^i\_n,k^i\_n)$ if and only if the previous pair $(\sigma\_{n-1},k\_{n-1})$ is already defined (this requirement is dropped for $n=0$) **and** copying it keeps inequality (19) intact.

Formally, write

$$E_i^n := \big((\sigma^i_0,k_0),(\sigma^i_1,k_1),\dots,(\sigma^i_n,k_n)\big)$$

for the list of all pairs enumerated into $E\_i$ by step $n$, and consider the subset

$$\lbrace (\sigma^n_{i_0},k_{i_0}),(\sigma^n_{i_1},k_{i_1}),\dots,(\sigma^n_{i_m},k_{i_m})\rbrace,$$

where $\sigma^n\_{i\_0},\dots,\sigma^n\_{i\_m}$ are the **distinct** words appearing so far and $k\_{i\_0},\dots,k\_{i\_m}$ are their respective **minimal** weights in $E\_i^n$. If

$$\sum_{j=0}^{m} 2^{-k_{i_j}} \le 1, \tag{21}$$

we set $(\sigma\_n,k\_n) = (\sigma^i\_n,k^i\_n)$; otherwise $(\sigma\_n,k\_n)$ stays undefined.

For every $i$ this yields a c.e. epigraph $E\_{M(i)}$, and its associated upper-semicomputable function $\widetilde F\_{M(i)}$ fulfills (19) because (21) holds at every enumeration step. Finally, if the original $\widetilde F\_i$ already satisfies (19), then (21) never fails, every construction step terminates, and so $E\_{M(i)} = E\_i$ and $\widetilde F\_{M(i)} = \widetilde F\_i$. Hence $\big(\widetilde F\_{M(0)}, \widetilde F\_{M(1)}, \dots\big)$ is a well-defined enumeration of exactly the information content measures.

</details>
</div>

<figure class="math-figure">
  <svg viewBox="0 0 700 360" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:700px" aria-label="Enumeration of all information content measures by trimming c.e. epigraphs">
    <defs>
      <marker id="arrow-enum-filter" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
        <path d="M 0 0 L 8 4 L 0 8 Z" fill="#5b6270" />
      </marker>
    </defs>
    <g font-family="serif" font-size="12" fill="#1f2430">
      <text x="350" y="24" text-anchor="middle" font-weight="600">Enumerating information content measures by a Kraft filter</text>

      <!-- Raw streams -->
      <rect x="30" y="55" width="190" height="250" rx="6" fill="rgba(29,78,216,0.06)" stroke="#1d4ed8" stroke-width="1.2" />
      <text x="125" y="80" text-anchor="middle" font-weight="600">all c.e. epigraphs E_i</text>
      <g>
        <rect x="55" y="108" width="140" height="28" rx="4" fill="#ffffff" stroke="#cbd2e0" />
        <text x="125" y="127" text-anchor="middle">(00, 2)</text>
        <rect x="55" y="146" width="140" height="28" rx="4" fill="#ffffff" stroke="#cbd2e0" />
        <text x="125" y="165" text-anchor="middle">(11, 3)</text>
        <rect x="55" y="184" width="140" height="28" rx="4" fill="#ffffff" stroke="#cbd2e0" />
        <text x="125" y="203" text-anchor="middle">(10, 1)</text>
        <rect x="55" y="222" width="140" height="28" rx="4" fill="#ffffff" stroke="#cbd2e0" />
        <text x="125" y="241" text-anchor="middle">(01, 2)</text>
      </g>
      <text x="125" y="278" text-anchor="middle" font-size="11" fill="#5b6270">some streams overspend</text>

      <!-- Filter -->
      <path d="M 224 180 L 295 180" stroke="#5b6270" stroke-width="1.4" marker-end="url(#arrow-enum-filter)" />
      <rect x="300" y="80" width="150" height="200" rx="6" fill="rgba(168,111,0,0.08)" stroke="#a86f00" stroke-width="1.3" />
      <text x="375" y="106" text-anchor="middle" font-weight="600">M(i)</text>
      <text x="375" y="126" text-anchor="middle" font-size="11" fill="#5b6270">copy pair only if</text>
      <text x="375" y="147" text-anchor="middle">current Kraft sum ≤ 1</text>
      <rect x="325" y="178" width="100" height="20" fill="rgba(94,96,96,0.08)" stroke="#1f2430" />
      <rect x="325" y="178" width="78" height="20" fill="rgba(18,177,124,0.15)" stroke="#0f9b6c" />
      <text x="375" y="213" text-anchor="middle" font-size="11" fill="#0f6b4a">accept</text>
      <text x="375" y="242" text-anchor="middle" font-size="11" fill="#b91c1c">reject if next block crosses 1</text>

      <!-- Trimmed streams -->
      <path d="M 455 180 L 526 180" stroke="#5b6270" stroke-width="1.4" marker-end="url(#arrow-enum-filter)" />
      <rect x="530" y="55" width="140" height="250" rx="6" fill="rgba(18,177,124,0.07)" stroke="#0f9b6c" stroke-width="1.3" />
      <text x="600" y="80" text-anchor="middle" font-weight="600">F_M(i)</text>
      <g>
        <rect x="552" y="108" width="96" height="28" rx="4" fill="#ffffff" stroke="#cbd2e0" />
        <text x="600" y="127" text-anchor="middle">(00, 2)</text>
        <rect x="552" y="146" width="96" height="28" rx="4" fill="#ffffff" stroke="#cbd2e0" />
        <text x="600" y="165" text-anchor="middle">(11, 3)</text>
        <rect x="552" y="184" width="96" height="28" rx="4" fill="rgba(214,83,54,0.08)" stroke="#d65336" stroke-dasharray="4 3" />
        <text x="600" y="203" text-anchor="middle" fill="#b91c1c">blocked</text>
        <rect x="552" y="222" width="96" height="28" rx="4" fill="#ffffff" stroke="#cbd2e0" />
        <text x="600" y="241" text-anchor="middle">(01, 2)</text>
      </g>
      <text x="600" y="278" text-anchor="middle" font-size="11" fill="#5b6270">always satisfies (19)</text>
    </g>
  </svg>
  <figcaption>Theorem 3.2 starts from an effective enumeration of all c.e. epigraphs. The operator $M$ filters each stream by copying a new pair only when the current minimal values still satisfy the Kraft bound. If the original stream already came from an information content measure, no pair is lost; otherwise the filter trims the overspending parts.</figcaption>
</figure>

In what follows we fix such a computable enumeration $F\_0, F\_1, \dots$ of all information content measures. Out of this list we manufacture a single canonical measure that dominates the whole class.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(3.3 — Minimal information content measure)</span></p>

The function $\widetilde F :\subseteq \lbrace 0,1\rbrace^{\ast} \to \mathbb{N}$ defined by

$$\widetilde{F}(w) := \min_{i\,:\,w \in \operatorname{dom}(F_i)} \big(F_i(w) + i + 1\big)$$

is called the **minimal information content measure**. The penalty $i+1$ charges for the index needed to name the $i$-th measure.

</div>

TODO: what is the point of penalty in Minimal information content measure?

<figure class="math-figure">
  <svg viewBox="0 0 680 360" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:680px" aria-label="The minimal information content measure as the lower envelope of penalized measures">
    <g font-family="serif" font-size="12" fill="#1f2430">
      <text x="340" y="24" text-anchor="middle" font-weight="600">The minimal measure is a lower envelope after paying an index penalty</text>

      <line x1="80" y1="300" x2="610" y2="300" stroke="#444" stroke-width="1.2" />
      <line x1="80" y1="60" x2="80" y2="300" stroke="#444" stroke-width="1.2" />
      <text x="345" y="334" text-anchor="middle">words w_0, w_1, w_2, ...</text>
      <text x="28" y="180" text-anchor="middle" transform="rotate(-90 28 180)">value</text>

      <g stroke="#e1e5ee" stroke-width="0.8">
        <line x1="80" y1="250" x2="610" y2="250" />
        <line x1="80" y1="200" x2="610" y2="200" />
        <line x1="80" y1="150" x2="610" y2="150" />
        <line x1="80" y1="100" x2="610" y2="100" />
      </g>

      <g fill="#5b6270" font-size="11" text-anchor="middle">
        <text x="120" y="320">w0</text>
        <text x="205" y="320">w1</text>
        <text x="290" y="320">w2</text>
        <text x="375" y="320">w3</text>
        <text x="460" y="320">w4</text>
        <text x="545" y="320">w5</text>
      </g>

      <!-- Penalized measures -->
      <polyline points="120,230 205,140 290,180 375,112 460,210 545,120" fill="none" stroke="#1d4ed8" stroke-width="2" />
      <polyline points="120,160 205,215 290,125 375,190 460,105 545,205" fill="none" stroke="#d65336" stroke-width="2" />
      <polyline points="120,198 205,175 290,235 375,150 460,165 545,95" fill="none" stroke="#0f9b6c" stroke-width="2" />

      <!-- Lower envelope -->
      <polyline points="120,230 205,215 290,235 375,190 460,210 545,205" fill="none" stroke="#1f2430" stroke-width="3.2" />

      <g>
        <circle cx="120" cy="230" r="4" fill="#1f2430" />
        <circle cx="205" cy="215" r="4" fill="#1f2430" />
        <circle cx="290" cy="235" r="4" fill="#1f2430" />
        <circle cx="375" cy="190" r="4" fill="#1f2430" />
        <circle cx="460" cy="210" r="4" fill="#1f2430" />
        <circle cx="545" cy="205" r="4" fill="#1f2430" />
      </g>

      <g font-size="11">
        <line x1="470" y1="228" x2="500" y2="228" stroke="#1d4ed8" stroke-width="2" />
        <text x="508" y="232">F0(w)+1</text>
        <line x1="470" y1="250" x2="500" y2="250" stroke="#d65336" stroke-width="2" />
        <text x="508" y="254">F1(w)+2</text>
        <line x1="470" y1="272" x2="500" y2="272" stroke="#0f9b6c" stroke-width="2" />
        <text x="508" y="276">F2(w)+3</text>
        <line x1="470" y1="294" x2="500" y2="294" stroke="#1f2430" stroke-width="3.2" />
        <text x="508" y="298">tilde F(w)</text>
      </g>

      <text x="340" y="54" text-anchor="middle" font-size="11" fill="#5b6270">
        For each word, choose the cheapest measure after paying the cost of naming it.
      </text>
    </g>
  </svg>
  <figcaption>The formula $\widetilde F(w)=\min_i(F_i(w)+i+1)$ forms a lower envelope of all information content measures after adding the index penalty. The penalty makes the mixture summable: the $i$-th measure receives only a geometrically small share of the global Kraft budget.</figcaption>
</figure>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise 3.4</span><span class="math-callout__name">(Minimal i.c.m. is a valid i.c.m.)</span></p>

Show that $\widetilde F$ is itself an information content measure.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**Upper bound**

For any word $w\in\text{dom}(F)$ holds

$$2^{-\widetilde{F}(w)} = \max_{i\in\mathbb{N}} (2^{-(F_i(w) + i + 1)}) \leq \sum_{i=0}^\infty 2^{-(F_i(w) + i + 1)}$$

Summing over all words $w\in\operatorname{dom}(\widetilde{F})$, we get 

$$
\begin{aligned}
\sum_{w\in\operatorname{dom}(\widetilde{F})} 2^{-\widetilde{F}(w)} 
&= \sum_{w\in\operatorname{dom}(\widetilde{F})} \max_{i\in\mathbb{N}} (2^{-(F_i(w) + i + 1)}) \\
&\leq 
\sum_{w\in\operatorname{dom}(\widetilde{F})} \sum_{i=0}^\infty 2^{-(F_i(w) + i + 1)} \\
&= 
\sum_{i=0}^\infty \sum_{w\in\operatorname{dom}(\widetilde{F})} 2^{-(F_i(w) + i + 1)} \\
&=
\sum_{i=0}^\infty 2^{-(i+1)} \sum_{w\in\operatorname{dom}(\widetilde{F})} 2^{-F_i(w)} \\
&=
\sum_{i=0}^\infty 2^{-(i+1)} \sum_{w\in\operatorname{dom}(F_i)} 2^{-F_i(w)} \\
&\leq 
\sum_{i=0}^\infty 2^{-(i+1)} \cdot 1 \\
&=
1,
\end{aligned}
$$

**C.E of epigraph**

* To rigorously prove the epigraph $\lbrace (w, k) : \widetilde{F}(w) \le k \rbrace$is c.e., we dovetail the computation of $F_i(w)$ for all $i \in \mathbb{N}$. 
* Because $\widetilde{F}(w)$ is defined as a minimum, we only need one machine to halt and satisfy $F_i(w) + i + 1 \le k$. 
* The exact moment any dovetailed computation outputs a value that satisfies this bound, we enumerate the pair $(w, k)$ into the set. 
* Since a **positive confirmation will always be found in finite time if it exists**, the epigraph is computably enumerable.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Maximal i.c.m. is not valid i.c.m.: Divergence to infinity)</span></p>

You are given a computable enumeration of all information content measures, $F_0,F_1,\dots$. There are infinitely many valid ICMs in this enumeration that will halt and be defined for a given word $w$. Because of the $i+1$ penalty term, the value of $F_i(w)+i+1$ will grow without bound as $i\to\infty$.  

Therefore, taking the maximum over all $i$ results in infinity. By definition, an information content measure must be a function mapping strictly to $\mathbb{N}$. It cannot evaluate to infinity.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(3.5 — $\widetilde F$ is total)</span></p>

$\widetilde F$ is total.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

For every word $w$ there is an index $i\_w$ of an **atomic** information content measure defined by

$$F_{i_w}(w) = 0 \quad \text{and} \quad F_{i_w}(v)\uparrow \text{ for all } v \ne w.$$

Then the set $\lbrace i \in \mathbb{N} : w \in \operatorname{dom}(F\_i)\rbrace$ is non-empty (it contains $i\_w$), so the minimum defining $\widetilde F(w)$ is taken over a non-empty set and $\widetilde F(w)$ is defined.

</details>
</div>

From now on we denote by $\widetilde F$ the minimal information content measure. The point of the construction is that it recaptures $K$ up to a constant.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(3.6 — $\widetilde F$ coincides with $K$)</span></p>

For every word $w$ it holds that

$$K(w) = \widetilde F(w) \pm O(1).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**Upper bound on $\widetilde F$.** Since $K$ is an information content measure and $F\_0, F\_1, \dots$ lists all of them, there is an index $k$ with $F\_k = K$. By the definition of $\widetilde F$,

$$\widetilde F(w) \le F_k(w) + k + 1 = K(w) + k + 1,$$

so $\widetilde F(w) \le K(w) + O(1)$.

**Lower bound on $\widetilde F$ (equivalently, upper bound on $K$).** We claim that the set

$$\lbrace (k+1,\, w) : \widetilde F(w) \le k\rbrace$$

is a KC set, i.e. a list of length-requests $k+1$ for the target words $w$.

* It is c.e. by condition (20) applied to $\widetilde F$.
* Using $\sum_{k \ge \widetilde F(w)} 2^{-(k+1)} = 2^{-\widetilde F(w)}$, its Kraft sum is

  $$\sum_{(k+1,w)} 2^{-(k+1)} = \sum_{w \in \lbrace 0,1\rbrace^{\ast}} \sum_{k \ge \widetilde F(w)} 2^{-(k+1)} = \sum_{w \in \lbrace 0,1\rbrace^{\ast}} 2^{-\widetilde F(w)} \le 1,$$

  where the last inequality is condition (19) applied to $\widetilde F$.

Since $\widetilde F$ is total, for every $w$ the pair $(\widetilde F(w)+1,\, w)$ lies in this KC set. By Corollary 2.28.1 the requested word $w$ then satisfies

$$K(w) \le \big(\widetilde F(w)+1\big) + O(1) = \widetilde F(w) + O(1).$$

Combining the two bounds gives $K(w) = \widetilde F(w) \pm O(1)$.

</details>
</div>

<figure class="math-figure">
  <svg viewBox="0 0 680 320" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:680px" aria-label="Why the minimal information content measure coincides with prefix-free complexity">
    <defs>
      <marker id="arrow-k-eq-f" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
        <path d="M 0 0 L 8 4 L 0 8 Z" fill="#5b6270" />
      </marker>
    </defs>
    <g font-family="serif" font-size="12" fill="#1f2430">
      <text x="340" y="24" text-anchor="middle" font-weight="600">Why the minimal measure recovers prefix-free complexity</text>

      <rect x="55" y="70" width="240" height="82" rx="6" fill="rgba(29,78,216,0.06)" stroke="#1d4ed8" stroke-width="1.3" />
      <text x="175" y="96" text-anchor="middle" font-weight="600">K is in the list</text>
      <text x="175" y="119" text-anchor="middle">F_k = K</text>
      <text x="175" y="139" text-anchor="middle" font-size="11" fill="#5b6270">so the minimum can choose it</text>

      <path d="M 298 111 L 378 111" stroke="#5b6270" stroke-width="1.4" marker-end="url(#arrow-k-eq-f)" />
      <text x="338" y="94" text-anchor="middle" font-size="11" fill="#5b6270">upper bound</text>

      <rect x="385" y="70" width="240" height="82" rx="6" fill="rgba(18,177,124,0.07)" stroke="#0f9b6c" stroke-width="1.3" />
      <text x="505" y="96" text-anchor="middle" font-weight="600">tilde F(w) ≤ K(w)+O(1)</text>
      <text x="505" y="123" text-anchor="middle" font-size="11" fill="#5b6270">minimality gives domination</text>

      <rect x="55" y="190" width="240" height="82" rx="6" fill="rgba(168,111,0,0.08)" stroke="#a86f00" stroke-width="1.3" />
      <text x="175" y="216" text-anchor="middle" font-weight="600">tilde F is a KC request list</text>
      <text x="175" y="239" text-anchor="middle">(tilde F(w)+1, w)</text>
      <text x="175" y="259" text-anchor="middle" font-size="11" fill="#5b6270">Kraft sum still ≤ 1</text>

      <path d="M 298 231 L 378 231" stroke="#5b6270" stroke-width="1.4" marker-end="url(#arrow-k-eq-f)" />
      <text x="338" y="214" text-anchor="middle" font-size="11" fill="#5b6270">Kraft-Chaitin</text>

      <rect x="385" y="190" width="240" height="82" rx="6" fill="rgba(214,83,54,0.07)" stroke="#d65336" stroke-width="1.3" />
      <text x="505" y="216" text-anchor="middle" font-weight="600">K(w) ≤ tilde F(w)+O(1)</text>
      <text x="505" y="243" text-anchor="middle" font-size="11" fill="#5b6270">requests become prefix-free descriptions</text>

      <text x="340" y="301" text-anchor="middle" font-size="13">
        Together: K(w) = tilde F(w) +/- O(1)
      </text>
    </g>
  </svg>
  <figcaption>The proof of Theorem 3.6 has two directions. Since $K$ itself is one of the information content measures, $\widetilde F$ cannot be much larger than $K$. Conversely, the epigraph of $\widetilde F$ generates a KC set of length requests, so Kraft-Chaitin turns those requests into prefix-free programs and gives $K(w) \le \widetilde F(w)+O(1)$.</figcaption>
</figure>

A second consequence of the same circle of ideas counts how many words of a given length can be highly compressible.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(3.7 — Chaitin's counting theorem)</span></p>

There is a constant $c$ such that, for all natural numbers $n$ and $r$,

$$\#\lbrace \sigma \in \lbrace 0,1\rbrace^n : K(\sigma) \le n + K(\mathrm{bin}(n)) - r\rbrace \le 2^{\,n-r+c}. \tag{22}$$

In particular, the maximal prefix-free complexity among words of a fixed length $n$ equals $n + K(\mathrm{bin}(n))$ up to an additive constant:

$$\max\lbrace K(\sigma) : l(\sigma) = n\rbrace = n + K(\mathrm{bin}(n)) \pm O(1). \tag{23}$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

1. Show that the function F: \subseteq \lbrace 0,1 \rbrace^\ast \to \mathbb{N} defined by
   
   $$F(bin(n)) := -\log \sum_{\sigma\in\lbrace 0,1\rbrace^n} 2^{-K(\sigma)}$$

   is an information content measure.

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**UPPER BOUND**

Then for a single $bin(n) \in \operatorname{dom}(F)$ we get:

$$2^{-F(bin(n))} = 2^{\log \sum_{\sigma\in\lbrace 0,1\rbrace^n} 2^{-K(\sigma)}} = \sum_{\sigma\in\lbrace 0,1\rbrace^n} 2^{-K(\sigma)}$$

Over all $bin(n) \in \operatorname{dom}(F)$:

$$\sum_{bin(n)\in\operatorname{dom}(F)} 2^{-F(bin(n))} = \sum_{bin(n)\in\operatorname{dom}(F)}\ \sum_{\sigma\in\lbrace 0,1\rbrace^n} 2^{-K(\sigma)} \leq \sum_{\sigma\in\lbrace 0,1\rbrace^\ast} 2^{-K(\sigma)} \leq 1$$

**C.E. EPIGRAPH**

Prefix-free Kolmogorov complexity $K$ is upper semi-computable. For the given pair $(bin(n), k)$ we do dovetailing over all binary words $\sigma\in\lbrace 0,1\rbrace^n$ and for each word $\sigma$ we again in dovetailing manner enumerably upper compute $K(\sigma)$. Since $\log \sum_{\sigma\in\lbrace 0,1\rbrace^n} 2^{-K(\sigma)}$ only decreases, the moment when $F(bin(n)) > k$ would be confirmed in finite number of steps if it's true.

</details>
</div>
   
1. Show that there exists a constant $\varepsilon > 0$ such that
   
   $$2^{K(bin(n))} \geq 2^{-F(bin(n))-c}$$

   *(hint: $K$ is a minimal information content measure)*

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Since $K$ is the minimal among infomration content measures, and $F$ is an in information content measure, there is a constant $c$ such that

$$K(bin(n)) \leq F(bin(n)) + c$$

</details>
</div>

1. For all natural numbers $n, r$, we define the set of all words of length $n$ with prefix-free complexity below $n + K(\operatorname{bin}(n)) - r$:

   $$S_{n,r} = \lbrace\sigma \in 2^n : K(\sigma) \le n + K(\operatorname{bin}(n)) - r\rbrace.$$

   Show that there exists a constant $c$ such that

   $$\sharp S_{n,r} \le 2^{n-r+c}$$

   for all $n, r$.

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Recall

$$S_{n,r} = \lbrace \sigma\in{0,1}^n: K(\sigma)\le n+K(\operatorname{bin}(n))-r\rbrace.$$

If $\sigma\in S_{n,r}$, then

$$2^{-K(\sigma)} \ge 2^{-n-K(\operatorname{bin}(n))+r}.$$

Therefore

$$
\begin{aligned}
\sharp S_{n,r}\cdot 2^{-n-K(\operatorname{bin}(n))+r}
&\le
\sum_{\sigma\in S_{n,r}}2^{-K(\sigma)}
\\
&\le
\sum_{\sigma\in{0,1}^n}2^{-K(\sigma)}
\
&=
W_n
\\
&\le
2^c2^{-K(\operatorname{bin}(n))}.
\end{aligned}
$$

Canceling $2^{-K(\operatorname{bin}(n))}$, we obtain

$$#S_{n,r}\le 2^{n-r+c}.$$

Thus there is a constant $c$ such that

$$\boxed{\sharp S_{n,r}\le 2^{n-r+c}}$$

for all $n,r$.

</details>
</div>

1. Show that there exist two constants $c_1$ and $c_2$ such that

   $$n - K(\operatorname{bin}(n)) - c_1 \le \max \lbrace K(\sigma) : l(\sigma) = n \rbrace \le n + K(\operatorname{bin}(n)) + c_2.$$

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

The sheet’s displayed lower bound says

$$n-K(\operatorname{bin}(n))-c_1 \le \max_{l(\sigma)=n}K(\sigma),$$

but the theorem stated immediately before says the intended bound is

$$\boxed{n+K(\operatorname{bin}(n))-O(1)\le\max_{l(\sigma)=n}K(\sigma).}$$

We prove the stronger, intended form.

**Upper bound.**

By the known upper bound,

$$K(\sigma) \le l(\sigma)+K(\operatorname{bin}(l(\sigma)))+O(1).$$

For (l(\sigma)=n),

$$K(\sigma)\le n+K(\operatorname{bin}(n))+c_2.$$

Hence

$$\max_{l(\sigma)=n}K(\sigma)\le n+K(\operatorname{bin}(n))+c_2.$$

**Lower bound.**

Use the counting estimate with $r=c+1$. Then

$$\sharp S_{n,c+1} \le 2^{n-(c+1)+c} = 2^{n-1}.$$

So fewer than all $2^n$ strings of length $n$ lie in $S_{n,c+1}$. Hence there exists some $\sigma\in{0,1}^n$ with

$$\sigma\notin S_{n,c+1}.$$

Therefore

$$K(\sigma) > n+K(\operatorname{bin}(n))-(c+1).$$

Thus

$$\max_{l(\sigma)=n}K(\sigma) \ge n+K(\operatorname{bin}(n))-c_1$$

for a suitable constant $c_1$.

Combining both inequalities,

$$\boxed{\max_{l(\sigma)=n}K(\sigma) = n+K(\operatorname{bin}(n))\pm O(1).}$$

</details>
</div>

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Characterization of computable upper bounds for $K$)</span></p>

Let $f : \mathbb{N} \to \mathbb{N}$ be a totally computable function.

Show that $f$ satisfies the inequality

$$\sum_{n \in \mathbb{N}} 2^{-f(n)} < \infty$$

if and only if there exists a constant $c$ such that

$$K(\operatorname{bin}(n)) \leq f(n) + c$$

holds for all $n$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

$$(\Rightarrow)$$

Assume

$$\sum_n2^{-f(n)}<\infty.$$

Choose $d\in\mathbb N$ such that

$$2^{-d}\sum_n2^{-f(n)}\le 1.$$

Define a partial function $F$ on binary words by

$$F(\operatorname{bin}(n)):=f(n)+d.$$

Since $f$ is computable, $F$ has c.e. epigraph. Also,

$$\sum_{n}2^{-F(\operatorname{bin}(n))} = \sum_n2^{-f(n)-d} = 2^{-d}\sum_n2^{-f(n)} \le 1.$$

So $F$ is an information content measure.

By minimality of $K$, there is a constant $c'$ such that

$$K(\operatorname{bin}(n)) \le F(\operatorname{bin}(n))+c' = f(n)+d+c'.$$

Thus

$$K(\operatorname{bin}(n))\le f(n)+c$$

for $c=d+c'$.

---

$$(\Leftarrow)$$

Assume

$$K(\operatorname{bin}(n))\le f(n)+c$$

for all $n$. Then

$$2^{-f(n)} \le 2^c,2^{-K(\operatorname{bin}(n))}.$$

Therefore

$$\sum_n2^{-f(n)} \le 2^c\sum_n2^{-K(\operatorname{bin}(n))} \le 2^c\sum_{\sigma\in{0,1}^\ast}2^{-K(\sigma)} \le 2^c.$$

Hence

$$\sum_n2^{-f(n)}<\infty.$$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Test-to-icm-translation is not straightforward)</span></p>

Recall that a Martin-Löf test $S_0, S_1, \dots$ is called a **Schnorr test** if it satisfies

$$\mu(S_i) = 2^{-i}$$

for all $i$, that is, with “$=$” instead of “$\leq$”.

Let $(S_0, S_1, \dots)$, where

$$S_i = \lbrace \sigma^i_0, \sigma^i_1, \dots\rbrace$$

for all $i$, be a uniformly c.e. sequence of prefix-free sets of words such that the corresponding uniformly c.e. sequence of open sets $\widetilde{S}\_0, \widetilde{S}\_1, \dots$, where

$$\widetilde{S}_i = \lbrace [\sigma^i_0], [\sigma^i_1], \dots\rbrace$$

for all $i$, is a Schnorr test.

Show that the partial function

$$F : \subseteq \lbrace 0,1\rbrace^* \to \mathbb{N}$$

defined by

$$\sum_{\sigma \in S_n} 2^{-l(\sigma)+n}$$

is not an information content measure.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

The definition of $F$ in the exercise is malformed: after “defined by” it only shows the expression

$$\sum_{\sigma\in S_n}2^{-l(\sigma)+n}.$$

This expression is not itself a definition of a function. The intended “naive translation” is almost certainly

$$F(\sigma)=l(\sigma)-n \qquad\text{for }\sigma\in S_n.$$

We show why this naive construction is not an information content measure.

Since $S_n$ is prefix-free and the associated open set has measure

$$\mu(\widetilde S_n)=2^{-n},$$

we have

$$\sum_{\sigma\in S_n}2^{-l(\sigma)} = 2^{-n}.$$

Now the Kraft contribution of the $n$-th layer under the naive definition $F(\sigma)=l(\sigma)-n$ is

$$\sum_{\sigma\in S_n}2^{-F(\sigma)} = \sum_{\sigma\in S_n}2^{-(l(\sigma)-n)} = 2^n\sum_{\sigma\in S_n}2^{-l(\sigma)} = 2^n\cdot 2^{-n} = 1.$$

So each layer alone spends a full Kraft budget $1$. Summing over infinitely many layers gives

$$\sum_n\sum_{\sigma\in S_n}2^{-(l(\sigma)-n)} = \sum_n1 = \infty.$$

Thus the direct test-to-icm translation fails the Kraft inequality. This is exactly why the construction in the lecture notes uses extra slack, for example by passing to layers $i^2$ and defining $F(\sigma)=l(\sigma)-i$, because then the total Kraft cost becomes

$$\sum_i 2^{i-i^2}<\infty.$$

So the naive $F(\sigma)=l(\sigma)-n$ is not an information content measure.

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(3.7.1 — Almost every word is incompressible)</span></p>

$$\frac{\#\lbrace w \in \lbrace 0,1\rbrace^n : K(w) > n\rbrace}{2^n} \xrightarrow[\;n \to \infty\;]{} 1.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Apply (22) with $r = K(\mathrm{bin}(n))$, so that the threshold becomes $n + K(\mathrm{bin}(n)) - r = n$ and

$$\#\lbrace w \in \lbrace 0,1\rbrace^n : K(w) \le n\rbrace \le 2^{\,n - K(\mathrm{bin}(n)) + c}.$$

Therefore

$$\frac{\#\lbrace w : K(w) > n\rbrace}{2^n} = 1 - \frac{\sharp \lbrace w : K(w) \le n\rbrace}{2^n} \ge 1 - \frac{2^{\,n - K(\mathrm{bin}(n)) + c}}{2^n} = 1 - 2^{-K(\mathrm{bin}(n)) + c} \xrightarrow[\;n \to \infty\;]{} 1,$$

because only finitely many words have complexity below any fixed bound, so the distinct encodings $\mathrm{bin}(n)$ force $K(\mathrm{bin}(n)) \to \infty$ and hence $2^{-K(\mathrm{bin}(n)) + c} \to 0$.

</details>
</div>

This last observation motivates formalizing the unpredictability of an infinite binary sequence in terms of **prefix-free incompressibility** of each of its initial segments.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(3.8 — 1-randomness)</span></p>

A sequence $X \in \lbrace 0,1\rbrace^{\mathbb{N}}$ is **1-random** if there exists a constant $c$ such that

$$K(X \upharpoonright n) \ge n - c \qquad \text{for all } n,$$

and **1-nonrandom** otherwise.

</div>

## Incompressibility and Typicality

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Typicality implies incompressibility)</span></p>

In this exercise, we prove in three steps that all Martin-Löf random sequences are prefix-free incompressible.

1. For a prefix-free Turing machine $M$, we define the sequence $L:= (L_0,L_1,\dots)$ of sets of words $L_i$ defined by
   
   $$L_i := \lbrace \sigma: C_M(\sigma) \leq l(\sigma) - i \rbrace \qquad \forall i\in\mathbb{N}$$

   Show that $\lambda(L_i) \le \lambda(\text{dom}(M))$ for every $i$.

2. Show that the sequence $\tilde{L} = (\tilde{L}\_0, \tilde{L}\_1, \dots)$ of open sets

   $$\tilde{L}_i := \lbrace X\in\lbrace 0,1\rbrace^{\mathbb{N}} : \exists n (K(X \upharpoonright n) \le n - i) \rbrace$$

   is a well-defined Martin-Löf test.

3. Show that for every Martin-Löf random sequence $A\in \lbrace 0,1\rbrace^{\mathbb{N}}$ there exists a constant $c$ such that

   $$K(A \upharpoonright n) \geq n - c \qquad \text{for all } n$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof I</summary>

* If $L_i$ is empty, then trivially $\lambda(L_i) = 0 \le \lambda(\text{dom}(M))$.
* If $L_i$ is non-empty, consider the output word $\sigma \in L_i$. Because $\sigma$ is in $L_i$, there exists an optimal code $\tau_\sigma$ of $\sigma$, then

  $$l(\tau_\sigma) = C_M(\sigma) \leq l(\sigma) - i$$

  Then 
  
  $$2^{-\tau_\sigma} \leq 2^{-(\sigma + i)} = 2^{-i}2^{-\sigma}$$

  Summing over the whole $L_i$ and taking into account that the mapping $\sigma\mapsto\tau_\sigma$ is injective, we obtain

  $$\sigma_{\sigma\in L_i} 2^{-l(\sigma)} \leq 2^{i}\sum_{\tau_\sigma:\sigma\in L_i} 2^{-\tau_\sigma} \leq 2^{i}\sum_{\tau\in \text{dom}(M)} 2^{-\tau}$$

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof II</summary>

Let (U) be the fixed universal prefix-free machine defining prefix-free Kolmogorov complexity

$$K(\sigma)=C_U(\sigma).$$

The notes define $K$-incompressibility by bounds of the form

$$K(X\upharpoonright n)\ge n-c$$

for all $n$. 

Define

$$\widetilde L_i := \lbrace X\in{0,1}^{\mathbb N}:\exists n; K(X\upharpoonright n)<n-i\rbrace.$$

Equivalently,

$$\widetilde L_i = \bigcup_{\sigma:\ K(\sigma)<l(\sigma)-i}[![\sigma]!].$$

So each $\widetilde L_i$ is open, because it is a union of basic cylinders.

**Uniform enumerability.**

Since $K$ is upper semicomputable, we can enumerate all pairs $(\sigma,k)$ such that $K(\sigma)\le k$. Hence we can uniformly enumerate all words satisfying

$$K(\sigma)<l(\sigma)-i.$$

Thus the sequence $(\widetilde L_i)\_i$ is uniformly effectively open.

**Measure bound.**

Let

$$P_i:=\lbrace\sigma:K(\sigma)<l(\sigma)-i\rbrace.$$

Because $K(\sigma)$ and $l(\sigma)-i$ are integers,

$$K(\sigma)<l(\sigma)-i \quad\Longleftrightarrow\quad K(\sigma)\le l(\sigma)-(i+1).$$

Therefore $P_i$ is exactly the set of words compressed by at least $i+1$ bits with respect to the universal prefix-free machine $U$. Applying part 1 with $M=U$,

$$\sum_{\sigma\in P_i}2^{-l(\sigma)} \le 2^{-(i+1)}\lambda(\operatorname{dom}(U)).$$

Since $U$ is prefix-free,

$$\lambda(\operatorname{dom}(U))\le 1.$$

Therefore

$$\lambda(\widetilde L_i) \le \sum_{\sigma\in P_i}2^{-l(\sigma)} \le 2^{-(i+1)} \le 2^{-i}.$$

Hence

$$\boxed{\lambda(\widetilde L_i)\le 2^{-i}.}$$

Thus $(\widetilde L_i)\_i$ is a uniformly enumerable sequence of open sets with the required measure bounds, so it is a Martin-Löf test. This matches the lecture-note form of a Martin-Löf test: uniformly enumerable open layers with $\lambda(M_i)\le 2^{-i}$. 

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof III</summary>

Since every ML random real $A\in\lbrace 0,1\rbrace^{\mathbb{N}}$ escapes all layers of the ML test, for the test above we have that there exists a constant $c$ such that

$$K(X \upharpoonright n) \geq n - c \qquad \forall n$$

</details>
</div>

The previous exercise *(Typicality implies incompressibility)* foreshadowed that Martin-Löf randomness and prefix-free incompressibility describe the same sequences. We now state this equivalence as a theorem and prove one of its two implications.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(4.1 — Incompressibility equals typicality)</span></p>

A sequence is $1$-random $\iff$ it is Martin-Löf random.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof ($\Longleftarrow$: Martin-Löf random $\Rightarrow$ 1-random)</summary>

We prove the contrapositive: every $1$-nonrandom sequence fails some Martin-Löf test, hence is Martin-Löf nonrandom.

**A uniformly c.e. family of words.** For each $i$ define

$$L_i := \lbrace \sigma : K(\sigma) \le l(\sigma) - i\rbrace = (\sigma^i_0, \sigma^i_1, \dots).$$

Since $K$ is upper-semicomputable (Proposition 2.23), the sequence $(L\_1, L\_2, \dots)$ is uniformly c.e., and by construction $K(\sigma^i\_n) \le l(\sigma^i\_n) - i$ for all $n$.

**Bounding the measure of each layer.** For every $\sigma^i\_n$ fix a shortest prefix-free code $\tau^i\_n$ with $\widetilde U(\tau^i\_n) = \sigma^i\_n$ and $l(\tau^i\_n) = K(\sigma^i\_n)$. From the definition of $L\_i$ we get $l(\tau^i\_n) \le l(\sigma^i\_n) - i$ for all $n$, so the total weight of the cylinders in $L\_i$ obeys

$$\sum_{n \in \mathbb{N}} 2^{-l(\sigma^i_n)} \le \sum_{n \in \mathbb{N}} 2^{-\left(l(\tau^i_n) + i\right)} = 2^{-i} \sum_{n \in \mathbb{N}} 2^{-l(\tau^i_n)} \le 2^{-i} \sum_{\tau \in \operatorname{dom}(\widetilde U)} 2^{-l(\tau)} \le 2^{-i},$$

where the last inequality is Proposition 2.25, since $\widetilde U$ is prefix-free.

**The induced Martin-Löf test.** Hence the uniformly c.e. sequence of open sets $\widetilde L\_1, \widetilde L\_2, \dots$, where

$$\widetilde L_i = \big([\![\sigma^i_0]\!], [\![\sigma^i_1]\!], \dots\big), \qquad \lambda(\widetilde L_i) \le 2^{-i},$$

is a well-defined Martin-Löf test.

**Nonrandom sequences fail it.** Let $X$ be $1$-nonrandom. Then for every fixed $i$ there is an $n\_i$ with $K(X \upharpoonright n\_i) \le n\_i - i$, so $X \upharpoonright n\_i \in L\_i$ and therefore $X \in \widetilde L\_i$. As $i$ was arbitrary, $X$ lies in every layer and thus fails the Martin-Löf test $\widetilde L\_1, \widetilde L\_2, \dots$, i.e. $X$ is Martin-Löf nonrandom.

**($\Longrightarrow$)** The converse direction comes next.

</details>
</div>

It remains to prove the converse implication, namely that Martin-Löf nonrandomness already forces $1$-nonrandomness. The idea is to turn a Martin-Löf test that captures $X$ into an *information content measure* that compresses the initial segments of $X$ arbitrarily far below their length, using the minimality of $K$ to transfer that compression to $K$ itself.

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof ($\Longrightarrow$: 1-random $\Rightarrow$ Martin-Löf random)</summary>

We prove the contrapositive. Let $X$ be a Martin-Löf nonrandom sequence, and let $(\widetilde L\_1, \widetilde L\_2, \dots)$ be a Martin-Löf test with $X \in \widetilde L\_i$ for all $i$. We may take the words $L\_i = (\sigma^i\_0, \sigma^i\_1, \dots)$ that generate the open set $\widetilde L\_i$ to be **prefix-free** and **pairwise disjoint**: whenever a word $\sigma^i\_k$ has already been enumerated into some $L\_j$ with $j < i$, we split it into its two one-bit extensions $\sigma^i\_k 0$ and $\sigma^i\_k 1$, which keeps the generated open set unchanged.

**An information content measure built from the test.** Define a partial function $F :\subseteq \lbrace 0,1\rbrace^{\ast} \to \mathbb{N}$ by

$$F(\sigma) := l(\sigma) - i \qquad \text{for every } \sigma \in L_{i^2},\ i \ge 2.$$

Then $F$ is a well-defined information content measure. Condition (20) (its lower graph is c.e.) holds by construction, since the family $(L\_{i^2})\_{i \ge 2}$ is uniformly c.e. Condition (19) (the Kraft inequality) holds because, using that the words of each $L\_{i^2}$ are prefix-free and pairwise disjoint,

$$\sum_{\sigma \in \operatorname{dom}(F)} 2^{-F(\sigma)} = \sum_{i=2}^{\infty} \sum_{\sigma \in L_{i^2}} 2^{-(l(\sigma)-i)} = \sum_{i=2}^{\infty} 2^{\,i} \underbrace{\sum_{\sigma \in L_{i^2}} 2^{-l(\sigma)}}_{=\ \lambda(\widetilde L_{i^2})\ \le\ 2^{-i^2}} \le \sum_{i=2}^{\infty} 2^{\,i-i^2} < 1.$$

**Transferring the compression to $K$.** Recall that $K$ is a *minimal* information content measure, so by Corollary 3.6.1 there is a constant $c$ with

$$K(\sigma) \le F(\sigma) + c \qquad \text{for every } \sigma \in \operatorname{dom}(F). \tag{24}$$

**Conclusion.** Fix any $i$. Since $X \in \widetilde L\_{i^2}$, some word $\sigma \in L\_{i^2}$ is a prefix of $X$, say $\sigma = X \upharpoonright n$, so that $l(\sigma) = n$. Applying (24) to this $\sigma$ gives

$$K(X \upharpoonright n) \le F(\sigma) + c = \big(l(\sigma) - i\big) + c = n - i + c.$$

As $i$ was arbitrary, the deficit $K(X \upharpoonright n) - n$ can be pushed below any constant, so no $c'$ satisfies $K(X \upharpoonright n) \ge n - c'$ for all $n$. Hence $X$ is $1$-nonrandom, and the contrapositive — and with it the theorem — is proved. $\square$

</details>
</div>

This closes the loop opened by the exercise above: **prefix-free incompressibility ($1$-randomness) and Martin-Löf typicality single out exactly the same sequences.** We will lean on this equivalence repeatedly, switching freely between the compression viewpoint and the measure-theoretic viewpoint of randomness.

## Computable Approximations of Real Numbers

As we will see later, a "natural" example of a random sequence is the binary representation of the limit of the *slowest* nondecreasing computable Cauchy sequence. Before we can make that statement precise, we need a vocabulary for the limits of computable Cauchy sequences and for their convergence speed. We therefore begin by transferring the notion of computability from words and sequences to the real line.

### Computability on the Reals

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(5.1 — Computable real)</span></p>

A real $\alpha$ is **computable** if $\alpha = 0.A$ for a computable binary sequence $A$ (recall that $0.A$ denotes the real with binary expansion $0.a\_0 a\_1 a\_2 \dots$).

</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Dyadic rationals)</span></p>

In what follows we identify the dyadic rationals with their *finite* binary representations. This is without loss of generality: for every word $w \in \lbrace 0,1\rbrace^{\ast}$, both tails $w\,1000\dots$ and $w\,0111\dots$ name the same real and are computable, so a dyadic rational always has a computable expansion.

</div>

To compare a real with the rationals effectively, we track it through the rationals lying below it.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(5.2 — Left cut)</span></p>

For a real $\alpha$, the **left cut** of $\alpha$ is the set of all rationals smaller than $\alpha$:

$$L(\alpha) := \lbrace q \in \mathbb{Q} : q < \alpha \rbrace.$$

</div>

Computability and the Cauchy condition transfer verbatim to sequences of rationals.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(5.3 — Computable sequence of rationals)</span></p>

A sequence of rationals $(q\_n)\_{n\in\mathbb{N}}$ is **computable** if there exists a Turing machine that, on input $i$, returns $q\_i$.

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(5.4 — Cauchy sequence)</span></p>

A sequence of rationals $(a\_n)\_{n\in\mathbb{N}}$ is a **Cauchy sequence** if for every $\epsilon > 0$ there exists $N \in \mathbb{N}$ such that $\lvert a\_n - a\_m\rvert < \epsilon$ for all $m, n \ge N$.

</div>

Combining the two notions yields the two central approximability classes of this section.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(5.5 — Computably approximable, left-c.e.)</span></p>

* A real $\alpha$ is **computably approximable** (c.a.) if there exists a computable Cauchy sequence $(a\_n)\_{n\in\mathbb{N}}$ converging to $\alpha$. 
* A real $\alpha$ is **left computably enumerable** (left-c.e.) if there exists a *nondecreasing* computable Cauchy sequence $(a\_n)\_{n\in\mathbb{N}}$ converging to $\alpha$.

</div>

The bare definition of "converging" says nothing about *how fast*; the next proposition shows that for computable reals one may always demand a known, geometric rate. Geometrically this rate is a nested family of dyadic intervals shrinking onto $\alpha$.

<figure class="math-figure">
  <svg viewBox="0 0 620 250" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:620px" aria-label="Effective approximation of a real by nested dyadic intervals">
    <g font-family="serif" font-size="12" fill="#1f2430">
      <text x="310" y="22" text-anchor="middle" font-weight="600">An effective approximation: α ∈ [aₙ, aₙ + 2⁻ⁿ]</text>

      <!-- n = 1 -->
      <rect x="320" y="44" width="260" height="22" fill="rgba(29,78,216,0.08)" stroke="#1d4ed8" stroke-width="1.3" />
      <text x="450" y="59" text-anchor="middle" font-size="11">n = 1,  width 2⁻¹</text>
      <!-- n = 2 -->
      <rect x="350" y="74" width="130" height="22" fill="rgba(60,120,40,0.10)" stroke="#3d7a26" stroke-width="1.3" />
      <text x="415" y="89" text-anchor="middle" font-size="11">n = 2,  2⁻²</text>
      <!-- n = 3 -->
      <rect x="370" y="104" width="65" height="22" fill="rgba(168,111,0,0.10)" stroke="#a86f00" stroke-width="1.3" />
      <text x="402" y="119" text-anchor="middle" font-size="10">n = 3</text>

      <!-- left endpoints aₙ guide lines down to the axis -->
      <g stroke="#5b6270" stroke-dasharray="2 3" stroke-width="0.8">
        <line x1="320" y1="66"  x2="320" y2="180" />
        <line x1="350" y1="96"  x2="350" y2="180" />
        <line x1="370" y1="126" x2="370" y2="180" />
      </g>

      <!-- number line -->
      <line x1="60" y1="180" x2="580" y2="180" stroke="#444" stroke-width="1.2" />
      <g stroke="#444" stroke-width="1">
        <line x1="60"  y1="176" x2="60"  y2="184" />
        <line x1="580" y1="176" x2="580" y2="184" />
      </g>
      <g font-size="11" fill="#444" text-anchor="middle">
        <text x="60"  y="198">0</text>
        <text x="580" y="198">1</text>
      </g>

      <!-- aₙ ticks -->
      <g font-size="10" fill="#5b6270" text-anchor="middle">
        <text x="320" y="198">a₁</text>
        <text x="350" y="212">a₂</text>
        <text x="370" y="198">a₃</text>
      </g>

      <!-- α marker -->
      <line x1="398" y1="40" x2="398" y2="184" stroke="#d65336" stroke-width="1.6" />
      <circle cx="398" cy="180" r="3.5" fill="#d65336" />
      <text x="398" y="228" text-anchor="middle" font-size="12" fill="#d65336" font-weight="600">α</text>

      <text x="310" y="246" text-anchor="middle" font-size="11" fill="#5b6270" font-style="italic">
        Each step halves the bracket around α; the left endpoints aₙ = α↾n climb toward α from below.
      </text>
    </g>
  </svg>
  <figcaption>An effective approximation of a computable real $\alpha$. The dyadic rationals $a_n = \alpha \upharpoonright n$ approach $\alpha$ from below, while the nested intervals $[a_n,\ a_n + 2^{-n}]$ trap $\alpha$ with error below $2^{-n}$ — exactly condition (25) and the containment (27) used in the proof of Proposition 5.6.</figcaption>
</figure>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(5.6 — Characterizations of computable reals)</span></p>

For a real $\alpha$, the following are equivalent:

- **(i)** $\alpha$ is computable;
- **(ii)** $L(\alpha)$ is computable;
- **(iii)** there exists a computable Cauchy sequence $(a\_n)\_{n\in\mathbb{N}}$ that converges *effectively*, i.e.

$$\lvert \alpha - a_n\rvert < 2^{-n} \qquad \text{for every } n. \tag{25}$$

A sequence satisfying (25) is called an **effective approximation** of $\alpha$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**(i) $\Longrightarrow$ (ii).** Let $\alpha = 0.A$ be computable and let $M$ be a Turing machine that, given $n$, outputs the bit $A(n)$ (the $n$-th position of $A$). We build a machine $N$ that, on input a rational $q$, decides whether $q \in L(\alpha)$. It enumerates the dyadic rationals

$$a_n := \alpha \upharpoonright n = 0.A(0)A(1)\cdots A(n-1)$$

until it meets an index $N$ for which

$$q < a_N \qquad \text{or} \qquad q \ge a_N + 2^{-N}. \tag{26}$$

If the left alternative of (26) holds, then $q < a\_N \le \alpha$, so $q \in L(\alpha)$. If the right alternative holds, then — using that

$$\alpha \in \big[\,\alpha \upharpoonright n,\ \alpha \upharpoonright n + 2^{-n}\,\big] \qquad \text{for all } n \tag{27}$$

— we get $q \ge a\_N + 2^{-N} \ge \alpha$, so $q \notin L(\alpha)$.

For every $q$ the search halts, because (26) is eventually satisfied:

- if $q < \alpha$, then since $a\_n \to \alpha > q$ the left alternative of (26) is met;
- if $q > \alpha$, then since $a\_n + 2^{-n} \to \alpha < q$ the right alternative of (26) is met.

**(i) $\Longrightarrow$ (iii).** If $\alpha = 0.A$ is computable, then by (27) the sequence $a\_n := \alpha \upharpoonright n$ already satisfies $\lvert \alpha - a\_n\rvert < 2^{-n}$, so it is an effective approximation of $\alpha$.

**(ii) $\Longrightarrow$ (i).** If $L(\alpha)$ is computable for $\alpha = 0.A$, then for every $n$ we can compute the prefix $\alpha \upharpoonright n$ — and hence its last bit $A(n-1)$ — by

$$\alpha \upharpoonright n = \max\lbrace 0.\sigma : l(\sigma) = n \text{ and } 0.\sigma \in L(\alpha)\rbrace.$$

**(iii) $\Longrightarrow$ (i).** Suppose $(a_n)$ is a computable rational sequence with

$$|\alpha-a_n|<2^{-n}.$$

If $\alpha$ is dyadic, it has a finite computable binary expansion, so we are done. Otherwise, to compute the first $k$ bits of $\alpha$, wait until for some $n$ the interval

$$(a_n-2^{-n},a_n+2^{-n})$$

is contained in one dyadic interval

$$\left[\frac{j}{2^k},\frac{j+1}{2^k}\right)$$

of length $2^{-k}$. This eventually happens because $\alpha$ is not on a dyadic boundary of level $k$. Then the binary expansion of $j$ of length $k$ gives $\alpha\upharpoonright k$. Hence the binary expansion of $\alpha$ is computable.

</details>
</div>

Relativizing the equivalence (i) $\Leftrightarrow$ (ii) to the halting problem $\emptyset'$ gives a one-jump analogue for free.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Corollary</span><span class="math-callout__name">(5.6.1 — Relativization to $\emptyset'$)</span></p>

For a real $\alpha = 0.A$, the following are equivalent:

- **(i)** $A$ is $\emptyset'$-computable;
- **(ii)** $L(\alpha)$ is $\emptyset'$-computable.

</div>

The relativized characterization locates the computably approximable reals exactly within the arithmetical hierarchy.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(5.7 — c.a. reals are exactly $\Delta^0_2$)</span></p>

A real $\alpha = 0.A$ is computably approximable if and only if $A \in \Delta^0\_2$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**($\Longleftarrow$).** If $A \in \Delta^0\_2$, then by Post's theorem (the limit lemma) there is a computable function $g : \mathbb{N} \times \mathbb{N} \to \lbrace 0,1\rbrace$ with

$$A(n) = \lim_{k\to\infty} g(k,n) \qquad \text{for all } n.$$

Setting $a\_n := 0.A\_n$, where $A\_n$ is the length-$n$ word

$$A_n := g(n,0)\,g(n,1)\cdots g(n,n-1)$$

(the stage-$n$ guesses of the first $n$ bits), yields a computable approximation $(a\_n)\_{n\in\mathbb{N}}$ of $\alpha$.

**($\Longrightarrow$).** Conversely, suppose $\alpha = 0.A = \lim\_{n\to\infty} a\_n$ for a computable approximation $(a\_n)\_{n\in\mathbb{N}}$. Then $L(\alpha)$ lies in both halves of the second level of the hierarchy:

- $L(\alpha) \in \Sigma^0\_2$, since

  $$q \in L(\alpha) \iff \exists n\, \forall m \ge n\,(q < a_m);$$

- $L(\alpha) \in \Pi^0\_2$, since

  $$q \in L(\alpha) \iff \forall n\, \exists m \ge n\,(q < a_m).$$

Hence $L(\alpha) \in \Sigma^0\_2 \cap \Pi^0\_2 = \Delta^0\_2$, i.e. $L(\alpha)$ is $\emptyset'$-computable. By Corollary 5.6.1, $\alpha$ is then $\emptyset'$-computable, which is to say $A \in \Delta^0\_2$. $\square$

</details>
</div>

### Halting Probability and the Class of Left-c.e. Reals

Left-c.e. reals admit a strikingly concrete description: each of them is the halting probability of some prefix-free machine, and conversely. We first attach a real to every prefix-free machine.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(5.8 — Halting probability)</span></p>

Let $\widetilde M$ be a prefix-free Turing machine. The real

$$\alpha_{\widetilde M} := \sum_{\sigma \in \operatorname{dom}(\widetilde M)} 2^{-l(\sigma)}$$

is called the **halting probability** of $\widetilde M$. It is exactly the Lebesgue measure $\lambda(\operatorname{dom}(\widetilde M))$ of the machine's domain, so the prefix-free Kraft inequality guarantees $\alpha\_{\widetilde M} \le 1$.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(5.9 — Characterizations of left-c.e. reals)</span></p>

For a real $\alpha$, the following are equivalent:

- **(i)** $\alpha$ is left-c.e.;
- **(ii)** $L(\alpha)$ is c.e.;
- **(iii)** there exists a prefix-free machine $\widetilde M$ such that $\alpha$ is the halting probability of $\widetilde M$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Fix a computable enumeration $(q\_n)\_{n\in\mathbb{N}}$ of all rationals.

**(i) $\Longrightarrow$ (ii).** If $(a\_n)$ is a left-c.e. approximation of $\alpha$, then $L(\alpha)$ is the domain of the partial computable function

$$f(q) := \min\lbrace n : a_n > q\rbrace,$$

and is therefore c.e.

**(ii) $\Longrightarrow$ (i).** Given an enumeration $q\_0, q\_1, \dots$ of $L(\alpha)$, compute the index sequence

$$i_0 := 0, \qquad i_n := \min\lbrace m > i_{n-1} : q_m > q_{i_{n-1}}\rbrace.$$

Then $q\_{i\_0}, q\_{i\_1}, \dots$ is nondecreasing and converges to $\alpha$, hence is a left-c.e. approximation.

**(iii) $\Longrightarrow$ (i).** Given a prefix-free machine $\widetilde M$ with $\alpha = \sum\_{q \in \operatorname{dom}(\widetilde M)} 2^{-l(q)}$, the partial sums

$$a_n := \sum_{\substack{q \,\in\, \operatorname{dom}(\widetilde M[n]) \\ l(q) < n}} 2^{-l(q)},$$

where $\widetilde M[n]$ is $\widetilde M$ run for $n$ steps, form a nondecreasing computable sequence converging to $\alpha$, i.e. a left-c.e. approximation.

**(i) $\Longrightarrow$ (iii).** 

Let $\alpha$ be left-c.e. Choose a nondecreasing computable rational approximation $a_s\nearrow\alpha$. Then the left cut $L(\alpha)$ is c.e., so the dyadic rationals below $\alpha$ can be enumerated. Let $b_s$ be the maximum of the first $s$ enumerated dyadic rationals and $0$. Then

$$0=b_0\le b_1\le b_2\le\cdots\nearrow\alpha,$$

and each increment $\delta_s=b_{s+1}-b_s$ is dyadic. Write each positive dyadic $\delta_s$ as a finite sum

$$\delta_s=\sum_i 2^{-d_{s,i}}.$$

Enumerate the length requests $(d_{s,i},\lambda)$. Their Kraft sum is

$$\sum_{s,i}2^{-d_{s,i}} =\sum_s(b_{s+1}-b_s) =\alpha\le 1.$$

By the Kraft-Chaitin theorem, there is a computable prefix-free machine $\widetilde M$ with codewords of exactly these lengths. Therefore

$$\sum_{\sigma\in\operatorname{dom}(\widetilde M)}2^{-l(\sigma)} =\alpha.$$

So every left-c.e. real in $[0,1]$ is a halting probability.

</details>
</div>

The three notions of approximability introduced here nest, each layer being pinned down by its own list of equivalent characterizations.

<figure class="math-figure">
  <svg viewBox="0 0 640 320" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:640px" aria-label="Hierarchy of computable, left-c.e., and computably approximable reals">
    <g font-family="serif" font-size="12" fill="#1f2430">
      <!-- outer: computably approximable = Delta^0_2 -->
      <rect x="30" y="36" width="580" height="258" rx="10" fill="rgba(214,83,54,0.06)" stroke="#d65336" stroke-width="1.4" />
      <text x="320" y="60" text-anchor="middle" font-weight="600" fill="#d65336">Computably approximable (c.a.)</text>
      <text x="320" y="78" text-anchor="middle" font-size="11" fill="#5b6270">α = lim of a computable Cauchy sequence  ⟺  A ∈ Δ⁰₂   (Thm 5.7)</text>

      <!-- middle: left-c.e. -->
      <rect x="70" y="96" width="500" height="160" rx="10" fill="rgba(168,111,0,0.08)" stroke="#a86f00" stroke-width="1.4" />
      <text x="320" y="120" text-anchor="middle" font-weight="600" fill="#a86f00">Left-c.e.</text>
      <text x="320" y="138" text-anchor="middle" font-size="11" fill="#5b6270">nondecreasing computable approx.  ⟺  L(α) c.e.  ⟺  α is a halting probability   (Prop 5.9)</text>

      <!-- inner: computable -->
      <rect x="120" y="156" width="400" height="80" rx="10" fill="rgba(29,78,216,0.08)" stroke="#1d4ed8" stroke-width="1.4" />
      <text x="320" y="184" text-anchor="middle" font-weight="600" fill="#1d4ed8">Computable</text>
      <text x="320" y="204" text-anchor="middle" font-size="11" fill="#5b6270">L(α) computable  ⟺  |α − aₙ| &lt; 2⁻ⁿ   (Prop 5.6,  Def 5.1)</text>

      <!-- Omega marker in the left-c.e. ring -->
      <text x="320" y="248" text-anchor="middle" font-size="10" fill="#a86f00" font-style="italic">Ω (a halting probability): left-c.e. but not computable</text>
    </g>
  </svg>
  <figcaption>The three classes of reals studied in this section, ordered by the strictness of their approximability: <strong>computable</strong> $\subseteq$ <strong>left-c.e.</strong> $\subseteq$ <strong>computably approximable</strong> $\left(= \Delta^0_2\right)$. Each layer is characterized by Propositions 5.6 and 5.9 and Theorem 5.7. The halting probability $\alpha_{\widetilde M}$ of a universal prefix-free machine is the canonical example separating left-c.e. from computable.</figcaption>
</figure>

In particular, the halting probability of a *universal* prefix-free machine — Chaitin's $\Omega$ — is left-c.e. but, as we will see, also $1$-random. It is precisely the promised "natural" example of a random sequence: the binary expansion of the limit of the slowest nondecreasing computable Cauchy sequence.

We single this real out, together with its canonical approximation.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(5.10 — Chaitin's $\Omega$)</span></p>

The halting probability of a *universal* prefix-free Turing machine $\widetilde U$,

$$\Omega := \sum_{\sigma \in \operatorname{dom}(\widetilde U)} 2^{-l(\sigma)},$$

is called **Chaitin's $\Omega$**. Running $\widetilde U$ for $s$ steps and collecting the codewords that have already halted gives the nondecreasing computable approximation $w\_0, w\_1, \dots$ of $\Omega$,

$$w_s := \sum_{\substack{\sigma \,\in\, \operatorname{dom}(\widetilde U[s]) \\ l(\sigma) \le s}} 2^{-l(\sigma)}.$$

Since $(w\_s)$ is a *monotone nondecreasing computable* approximation, the existence of $(w\_s)$ already shows that $\Omega$ is a **left-c.e. real**. It is, however, *noncomputable* — this is the content of the next proposition.

</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(5.11 — $\Omega$ and the halting problem)</span></p>

The halting problem is Turing equivalent to $\Omega$:

$$\emptyset' \equiv_T \Omega.$$

In particular $\Omega$ is noncomputable, so it is a left-c.e. real that is not computable.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

This is Sheet 5, Exercise 1, both directions of which are spelled out in the exercises on Chaitin's $\Omega$ below.

For $\Omega \le\_T \emptyset'$: the approximation $(w\_s)$ is computable, and $\emptyset'$ can decide, for each $n$, the (c.e.) question of whether the first $n$ bits of $w\_s$ have stabilized, so $\emptyset'$ computes $\Omega \upharpoonright n$ for every $n$.

For $\emptyset' \le\_T \Omega$: this is the dovetailing argument recorded in the remark below — knowing $\Omega \upharpoonright N$ lets one run all programs until enough halting weight accumulates to match these $N$ bits, after which no further program of length $\le N$ can halt, deciding the halting problem for all such programs.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Relationship to the halting problem)</span></p>

Knowing the first $N$ bits of $\Omega$ allows one to decide the halting problem for every $\widetilde U$-program of length at most $N$. Indeed, dovetail all computations of $\widetilde U$, and let

$$\Omega_s=\sum_{\sigma\in \operatorname{dom}(\widetilde U[s])}2^{-l(\sigma)}$$

be the halting probability already observed by stage $s$. Since the first $N$ bits of $\Omega$ determine an interval of length $2^{-N}$ containing $\Omega$, we can wait until the remaining gap between $\Omega$ and $\Omega_s$ is smaller than $2^{-N}$. At that point, no unseen program of length at most $N$ can still halt, because such a program would contribute at least $2^{-N}$ to $\Omega$. Therefore, if a given program $p$ of length at most $N$ has not halted by then, it never halts.

Because many outstanding problems in **number theory**, such as **Goldbach's conjecture**, are equivalent to solving the halting problem for special programs (which would basically search for counter-examples and halt if one is found), knowing enough bits of Chaitin's constant would also imply knowing the answer to these problems. But as the halting problem is not generally solvable, calculating any but the first few bits of Chaitin's constant is not possible for a universal language. This reduces hard problems to impossible ones, much like trying to build an **oracle machine for the halting problem** would be.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Solovay domination property of $\Omega$)</span></p>

Let $\alpha$ be the halting probability of a prefix-free Turing machine $\tilde{M}$, and let $\Omega$ be the halting probability of a universal Turing machine $\tilde{U}$. Let further $a_0,a_1,\dots$ be a fixed left-c.e. approximation of $\alpha$. 

Show that there exists a left-c.e. approximation $w_0,w_1,\dots$ of $\Omega$ and a constant $c$ such that 

$$a_{n+1}−a_n \leq c(w_{n+1}−w_n) \qquad \text{ for all } n.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

This is the Solovay domination property of $\Omega$.

Let $\alpha$ be the halting probability of $\widetilde M$, and let $\widetilde U$ be universal. By universality, there is a fixed code $i_{\widetilde M}$ such that

$$\widetilde U(i_{\widetilde M},\sigma)=\widetilde M(\sigma).$$

Let $d=l(i_{\widetilde M})$. The part of $\Omega$ coming from this simulation is

$$\sum_{\sigma\in\operatorname{dom}(\widetilde M)} 2^{-d-l(\sigma)} = 2^{-d}\alpha.$$

Write the remaining contribution to $\Omega$ as $\beta$. Since it is also left-c.e., choose a left-c.e. approximation $b_n\nearrow\beta$. Define

$$w_n:=2^{-d}a_n+b_n.$$

Then $(w_n)$ is a nondecreasing computable approximation of

$$2^{-d}\alpha+\beta=\Omega.$$

Moreover,

$$w_{n+1}-w_n = 2^{-d}(a_{n+1}-a_n)+(b_{n+1}-b_n) \ge 2^{-d}(a_{n+1}-a_n).$$

Therefore

$$a_{n+1}-a_n \le 2^d(w_{n+1}-w_n).$$

So the required constant is $c=2^d$.

</details>
</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(Turing-reducibility)</span></p>

A set $A$ is **Turing-reducible** to $B$, written $A \le_T B$, if there exists an oracle Turing machine $M$ such that $A=M(B)$, i.e.

$$A(x)=M(x,B)\quad\text{for all }x\in\mathbb{N}$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(The halting problem is Turing reducible to Chaitin’s $\Omega$)</span></p>

Show that

$$\emptyset' \leq_T \Omega,$$

i.e. that the halting problem is Turing reducible to Chaitin’s Omega

$$\Omega = \sum_{\sigma \in \operatorname{dom} \widetilde U} 2^{-l(\sigma)}.$$

*Hint:* for a given $e$, build a prefix-free machine $\widetilde M$ such that

$$\widetilde M(0^e1)\downarrow \quad \Longleftrightarrow \quad \Phi_e(e)\downarrow.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

> Consider we have an access to the oracle that tells us the first $n$ bits of the Chaitin’s Omega for any given $n\in\mathbb{N}$. Knowing the first $n$ bits of the Chaitin’s Omega, we run all programs of the size up to $n$ bits. Because the first $n$ bits of the Chaitin’s Omega are determined by the sums $2^{-l(\sigma)}$ of all programs that halt and have length $l(\sigma) \leq n$. So, once the sum is is equal to the first $n$ bits of the Chaitin’s Omega, then no other programs of the length up to $n$ will halt, otherwise they will alter the sum and it won't be equal to the first $n$ bits of the Chaitin’s Omega.

The previous proof has the **right core idea**, but as written it is **not quite valid yet**.

The main idea is correct:

$$\text{enough bits of }\Omega \quad\Longrightarrow\quad \text{decide halting of all short }\widetilde U\text{-programs}.$$

But two points need fixing.

**1. The first $n$ bits are not determined only by programs of length $\le n$.**

Longer programs also contribute to $\Omega$. Their total contribution may affect the first $n$ bits through carries. What is true is slightly different:

If we know $\Omega$ to error $<2^{-n}$, then after enumerating enough halting programs, once the remaining possible measure is $<2^{-n}$, no unseen program of length $\le n$ can still halt, because such a program would contribute at least $2^{-n}$.

So replace “once the sum is equal to the first $n$ bits” by a **gap argument**.

**We still need to connect this to $\emptyset'$.**

I use the sheet’s three exercise statements and the lecture notes’ conventions for prefix-free machines, Chaitin’s $\Omega$, left-c.e. approximations, and Solovay reducibility.  

**Key idea.**
The first $N$ bits of $\Omega$ decide halting of all $\widetilde U$-programs of length $\le N$. So we only need to encode the question “does $\Phi_e(e)$ halt?” as halting of one computable $\widetilde U$-program.

Let $\widetilde M$ be the prefix-free machine defined by

$$\widetilde M(0^e1)\downarrow \quad\Longleftrightarrow\quad \Phi_e(e)\downarrow.$$

The set

$$\lbrace 0^e1:e\in\mathbb N\rbrace$$

is prefix-free: if $e<f$, then $0^e1$ and $0^f1$ first differ at the $(e+1)$-st symbol, where the first has $1$ and the second has $0$. Hence $\widetilde M$ is a prefix-free machine.

By universality of $\widetilde U$, there is a fixed code $\rho$ such that

$$\widetilde U(\rho,0^e1)\downarrow \quad\Longleftrightarrow\quad \widetilde M(0^e1)\downarrow \quad\Longleftrightarrow\quad \Phi_e(e)\downarrow.$$

So, given $e$, compute the corresponding $\widetilde U$-program

$$p_e:=\rho,0^e1.$$

It remains to explain how $\Omega$ decides whether $p_e\in\operatorname{dom}\widetilde U$.

Let $N=l(p_e)$. From the oracle $\Omega$, get a rational upper approximation $u$ with

$$\Omega < u < \Omega+2^{-N}.$$

Now dovetail all computations of $\widetilde U$, and let

$$\omega_s = \sum_{\sigma\in\operatorname{dom}(\widetilde U[s])}2^{-l(\sigma)}$$

be the finite halting weight seen by stage $s$. Wait until

$$u-\omega_s<2^{-N}.$$

This stage must eventually appear. If $p_e$ has halted by then, answer yes. If it has not halted, then it can never halt: otherwise its future contribution would be at least

$$2^{-l(p_e)}=2^{-N},$$

contradicting the fact that the remaining unseen halting weight is already $<2^{-N}$.

Thus $\Omega$ decides whether $\Phi_e(e)\downarrow$. Therefore

$$\emptyset'\le_T\Omega.$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What is $\Phi$?)</span></p>

Here $\Phi$ is standard computability-theory notation for an **effective enumeration of partial computable functions**.

More precisely,

$$\Phi_0,\Phi_1,\Phi_2,\dots$$

is a fixed listing of all partial computable functions $\mathbb N \rightharpoonup \mathbb N$, or equivalently all Turing machines/programs under some Gödel numbering.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(A watered-down Chaitin’s Omega)</span></p>

Recall the statement of the **Kleene Second Recursion Theorem for prefix-free machines**:

Let

$$h : \subseteq \lbrace 0,1\rbrace^\ast \times \mathbb N \to \lbrace 0,1\rbrace^\ast$$

be a partial computable function such that, for each $i$, the function $h(\cdot,i)$ is prefix-free. Then, from an index for $h$, we can compute an index $e$ such that

$$\widetilde M_e(w) = h(w,e)$$

for all $w$.

For $r \in (0,1)$, define the **watered-down halting probability**

$$\Omega^r = \sum_{\sigma \in \operatorname{dom} \widetilde U} 2^{-l(\sigma)/r}.$$

Show that

$$K(\Omega^{1/2}\!\upharpoonright n) < \frac n2 + O(1).$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Four equivalent characterizations of Solovay reducibility for left-c.e. reals)</span></p>

Let $\alpha$ and $\beta$ be left-c.e. reals. Show that the following four conditions are equivalent.

* **(i)** There are a constant $c$ and a partial computable function $f : \subseteq \mathbb Q \to \mathbb Q$ such that, for every $q < \beta$ fulfills $f(q)\downarrow < \alpha$ and $\alpha - f(q) < c(\beta - q)$.

* **(ii)** There exists a constant $c$ such that, for every pair of left-c.e. approximations $(a_n)\_{n \in \mathbb N}$ and $(b_n)\_{n \in \mathbb N}$ of $\alpha$ and $\beta$, respectively, there exists a computable function $g : \mathbb N \to \mathbb N$ such that $\alpha - a_{g(n)} < c(\beta - b_n)$ for all $n$.

* **(iii)** For every left-c.e. approximation $(b_n)$, there exist a constant $d$ and a left-c.e. approximation $a_0 < a_1 < \cdots \to \alpha$ such that $a_s - a_{s-1} \leq d(b_s - b_{s-1})$ for all $s$.

* **(iv)** There exist two left-c.e. approximations $(a_n)\_{n \in \mathbb N}$ and $(b_n)\_{n \in \mathbb N}$ of $\alpha$ and $\beta$, respectively, and a constant $c$ such that $\alpha - a_n < c(\beta - b_n)$ for all $n$.

</div>


<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

$\boxed{(i) \implies (ii)}$

> * For the increasing sequence of $q$ rationals, approximating $\beta$ from the left, we set $b_n := q$ and preserve the same constant $c$. 
> * The function $g(n)$ is a function that based on the rational $q$ computes the value $f(q)$ and given the sequence $a_m$ waits for the index m such that $a_m$ (is on the left from $\alpha$) approach sufficiently close to $\alpha$ such that $\alpha - a_m < c(\beta - b_n)$ and $g(n)$ outputs $m$.

Assessment:

You cannot define $g(n)$ by waiting until

$$\alpha-a_m<c(\beta-b_n),$$

because $\alpha,\beta$ are not computable quantities available to the algorithm.

Instead, use the computable condition

$$a_m>f(b_n).$$

Since $f(b_n)<\alpha$ and $a_m\nearrow\alpha$, such an $m$ eventually appears. Then

$$\alpha-a_m<\alpha-f(b_n)<c(\beta-b_n).$$

So define

$$g(n)=\min\lbrace m:a_m>f(b_n)\rbrace.$$

That fixes the proof.

$\boxed{(i) \implies (iii)}$

Assume (i), witnessed by $f$ and $c$. Fix an arbitrary left-c.e. approximation $(b_s)$ of $\beta$. Choose some $d>c$, say $d=2c$.

Let

$$t_s:=f(b_s).$$

This is defined for every $s$, since $b_s<\beta$. We construct a left-c.e. approximation $(a_s)$ of $\alpha$ with controlled increments.

Set

$$a_0:=t_0.$$

Given $a_{s-1}$, define

$$a_s := \min\Bigl( \max(a_{s-1},t_s), a_{s-1}+d(b_s-b_{s-1}) \Bigr).$$

Then $a_s\ge a_{s-1}$, $a_s<\alpha$, and

$$a_s-a_{s-1}\le d(b_s-b_{s-1}).$$

It remains to show $a_s\to\alpha$. We prove the invariant

$$\alpha-a_s<d(\beta-b_s).$$

For $s=0$,

$$\alpha-a_0 = \alpha-f(b_0) < c(\beta-b_0) < d(\beta-b_0).$$

Assume the invariant at stage $s-1$.

If $a_s=\max(a_{s-1},t_s)$, then $a_s\ge t_s$, hence

$$\alpha-a_s \le \alpha-t_s = \alpha-f(b_s) < c(\beta-b_s) < d(\beta-b_s).$$

If instead

$$a_s=a_{s-1}+d(b_s-b_{s-1}),$$

then

$$\alpha-a_s = \alpha-a_{s-1}-d(b_s-b_{s-1}) < d(\beta-b_{s-1})-d(b_s-b_{s-1}) = d(\beta-b_s).$$

Thus the invariant holds for all $s$. Since $b_s\to\beta$, we get

$$0\le \alpha-a_s<d(\beta-b_s)\to0.$$

Therefore $a_s\to\alpha$, and the increment bound gives (iii).

$\boxed{(ii) \implies (iv)}$

**Note:** It seems to be trivial to just set the sequence $(a_n)\_{n\in\mathbb{N}}$ to $(a_g(n))\_{n\in\mathbb{N}}$ from (ii), but the catch is that the sequence $(a_g(n))\_{n\in\mathbb{N}}$ does not have to left approximation, specifically an increasing sequence.

To fix this, we reorder the sequence (or taking the running maximum) $(a_g(n))\_{n\in\mathbb{N}}$ to make it $(a_g(n))\_{n\in\mathbb{N}}$ required for (iv). To do so, for the current $b_n$ we set the current $a_n$ as the maximum among $a_g(m)$ assigned for all previous previous $b_m < b_n$ and including myself $a_g(n)$. Doing so we obtain the monotone sequence approximating $\alpha$ from the left:

$$a'_n = \max_{m \leq n} a_g(m)$$

Then

$$\alpha - a'_n \leq \alpha - a_g(n) \leq c(\beta - b_n)$$

$\boxed{(iv) \implies (i)}$

For the given rational $q$ we start enumerating the pairs (a_n, b_n) from both approximating sequences from (iv). Once b_n \geq q, we set f(q) = a_n.

\alpha - f(q) = \alpha - a_n < c(\beta - b_n) < c(\beta - q)

**Note that made me struggle:** I thought that the approximating sequence from (iv) could be non-rational and we would have to find the rationals in the pairs of intervals. But in (iv), $(a_n)$ is a left-c.e. approximation of $\alpha$. By definition, that means $a_n\in\mathbb Q$ for every $n$, and $a_n\nearrow\alpha$.

$\boxed{(iii) \implies (iv)}$

> * The sequence b_n has a limit \beta implying the consecutive b_n - b_{n-1} shrinks to zero. 
> * The consequtive distances a_n - a_{n-1} is bounded by b_n - b_{n-1} up to multiplicative constant, so it shirnks to and thus has a limit. 
> * \alpha = \sum_{i}^\infty a_i - a_{i-1}
> * \beta = \sum_{i}^\infty b_i - b_{i-1}
> * \alpha - d\beta = \sum_{i}^\infty (a_i - db_i) - (a_{i-1} - db_{i-1})
> * a_n = \sum_{i}^n a_i - a_{i-1}
> * b_n = \sum_{i}^n b_i - b_{i-1}
> * a_n - db_n = \sum_{i}^n (a_i - db_i) - (a_{i-1} - db_{i-1})
> * From (iii) we know a_s - db_s \leq a_{s-1} - db_{s-1}, thus (a_i - db_i) - (a_{i-1} - db_{i-1}) \leq 0
> * Then \alpha - d\beta = \sum_{i}^\infty (a_i - db_i) - (a_{i-1} - db_{i-1}) \leq \sum_{i}^n (a_i - db_i) - (a_{i-1} - db_{i-1}) = a_n - db_n
> * Hence \alpha - a_n \leq d(\beta - b_n)

Main issues:

1. You do **not** need to prove that $(a_n)$ has a limit. In (iii), $(a_n)$ is already given as a left-c.e. approximation of $\alpha$, so $a_n\nearrow \alpha$.

2. The identities

   $$\alpha=\sum_i^\infty (a_i-a_{i-1}),\qquad \beta=\sum_i^\infty (b_i-b_{i-1})$$

   are only true if $a_0=b_0=0$. Otherwise you need initial terms.

3. The clean proof should use **tails**, not full sums.

**So the main idea in your proof is right:** compare accumulated increments. But the sharp way is to compare the **remaining increments after stage (n)**:

$$\text{tail of }a \le d\cdot \text{tail of }b.$$

That gives exactly

$$\alpha-a_n\le d(\beta-b_n).$$

**Full proof:**

Assume (iii). Thus for every left-c.e. approximation $(b_n)$ of $\beta$, there exist a constant $d$ and a left-c.e. approximation

$$a_0<a_1<\cdots\to\alpha$$

such that

$$a_s-a_{s-1}\le d(b_s-b_{s-1})$$

for all (s\ge 1).

Fix such approximations ((a_n)) and ((b_n)). Since (a_n\nearrow\alpha), we have

$$\alpha-a_n = \sum_{s=n+1}^{\infty}(a_s-a_{s-1}).$$

Using the increment bound from (iii),

$$\alpha-a_n = \sum_{s=n+1}^{\infty}(a_s-a_{s-1}) \le d\sum_{s=n+1}^{\infty}(b_s-b_{s-1}).$$

Since (b_n\nearrow\beta), the last sum telescopes to

$$\sum_{s=n+1}^{\infty}(b_s-b_{s-1}) = \beta-b_n.$$

Therefore

$$\alpha-a_n\le d(\beta-b_n).$$

Hence $(a_n)$, $(b_n)$, and $d$ satisfy condition (iv).

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Every computable real is K-trivial; every K-trivial real is c.a.(1))</span></p>

A set of words $T \subseteq \lbrace 0,1\rbrace^\ast$ is called a **binary tree** if, for every $w \in T$, all $v \sqsubseteq w$ fulfill $v \in T$.

An infinite sequence $A$ is a **path** in a binary tree $T$ if $A \upharpoonright n \in T$ for every $n$.

A path $A$ in a binary tree $T$ is called **isolated** if, for some $n$, there exists no other path $B \neq A$ in $T$ such that

$$A \upharpoonright n = B \upharpoonright n.$$

A real $\alpha$ is called **K-trivial** if there exists a constant $c$ such that

$$K(\alpha \upharpoonright n) \leq K(\operatorname{bin}(n)) + c$$

for all $n$.

---

Show that, for every computable real $\alpha$, we have

$$K(\alpha \upharpoonright n) = K(\operatorname{bin}(n)) \pm O(1).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

> Solution:
> We know that a real \alpha is computable iff there is a computable sequence (a_i)\_{i\in\mathbb{N}} such that \lvert \alpha-a_i\rvert < 2^{-n} for all n\in\mathbb{N}.
> Give the number n\in\mathbb{N} and *given* (or computable in constant time by the *given* Turing machine) the approximating computable sequence (a_i)\_{i\in\mathbb{N}}, satisfying the property above, we obtain a_n, which has the same prefix as the real \alpha. So, K-complexity of the prefix alpha is reduced to the K-complexity of the number or index n\in\mathbb{N}. 
> **Note aside:** I guess that we assume that the approximation is given explicitly or implicitly via given Turing machine. Turing machine is given, because I suspect we would have to add the complexity of the Turing machine that computes the approximating sequence.


TODO: decide, why it is false.

Let $\alpha$ be computable. From $n$ one can compute $\alpha\upharpoonright n$, so

$$K(\alpha\upharpoonright n)\le K(\operatorname{bin}(n))+O(1).$$

Conversely, from $\alpha\upharpoonright n$ one can compute its length $n$, hence $\operatorname{bin}(n)$. Therefore

$$K(\operatorname{bin}(n))\le K(\alpha\upharpoonright n)+O(1).$$

Thus

$$K(\alpha\upharpoonright n)=K(\operatorname{bin}(n))\pm O(1).$$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Another interpretation of isolated paths)</span></p>

An **isolated path** is an infinite path that has **some finite prefix not shared by any other infinite path**.

$$[A\upharpoonright n] \cap [T]= \lbrace A\rbrace,$$

where $[T]$ means the set of infinite paths through $T$.

</div>

<figure class="math-figure">
  <img src="{{ '/assets/images/notes/books/algorithmic_randomness_computable_analysis/arca_isolated_paths.png' | relative_url }}" alt="Two binary trees drawn upwards with the level on the vertical axis. Left: an isolated path A separates from the other paths B1, B2 at the isolating level n = 3; the ringed node is the isolating prefix rho, the shaded cone above it contains only A, and a dashed finite dead end above level 3 is marked as allowed. Right: a non-isolated path A = 000... with branches B_k = 0^k 1 000... peeling off at every level; the ringed branch points accumulate along A, and an arrow marks that B_5 still agrees with A up to level 5." loading="lazy">
  <figcaption>Left: an isolated path. After the isolating level $n = 3$ no other path shares the prefix $\rho = A \upharpoonright 3$, so $A$ has the cylinder of $\rho$ entirely to itself; the finite dead end above $n$ is allowed, because isolation only counts infinite paths. Right: a non-isolated path. In the tree with paths $A = 000\cdots$ and $B_k = 0^k 1\, 000\cdots$, every level $n$ is still shared: $B_n \neq A$ but $B_n \upharpoonright n = A \upharpoonright n$, so branch points accumulate along $A$ and no level ever isolates it. Nodes are drawn at the dyadic position of the word they code, which is why each path visibly converges to the real $0.A$ it represents. This dichotomy drives Exercises (4) and (5) below: a K-trivial sequence is an isolated path of $T_c$, and an isolated path of a computable tree is computable.</figcaption>
</figure>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Approximation does not automatically give the exact binary prefix.)</span></p>

Let \alpha be a computable real. For the approximating sequence $(a_n)\_{n\in\mathbb{N}}$ from the condition

$$\lvert\alpha−a_n\rvert<2^{−n}$$
 
it is not always safe to say that $a_n$ has the same first $n$ bits as $\alpha$. If $\alpha$ is very close to a dyadic boundary, a tiny error can change the $n$-th binary digit.

In this course, the primary definition of real computability is **by binary prefixes**, not by arbitrary rational approximations:

$$\alpha \text{ is computable iff } \alpha=0.A$$

for some **computable binary sequence** $A$. So from input $n$, a fixed machine can compute the bits $A(0),\dots,A(n-1)$, hence compute

$$\alpha\upharpoonright n.$$

This is exactly the clean version you want for the exercise. 

The notes also give an equivalent Cauchy-style characterization: $\alpha$ has a computable rational sequence $(a_n)$ with

$$|\alpha-a_n|<2^{-n}.$$

But for your current proof, the prefix definition is safer, because it avoids the dyadic-boundary issue. 

So your first line should be adjusted to:

$$\text{Since }\alpha\text{ is computable, there is a computable binary sequence }A \text{ such that }\alpha=0.A.$$

Then from $\operatorname{bin}(n)$, a fixed machine computes $A\upharpoonright n=\alpha\upharpoonright n$.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Every computable real is K-trivial; every K-trivial real is c.a.(2))</span></p>

Show that, for every $c$, the set

$$T_c := \lbrace \sigma : \forall \tau \sqsubseteq \sigma \, (K(\tau) \leq K(\operatorname{bin}(\ell(\tau))) + c)\rbrace$$

is a binary tree, which is computable in $\emptyset'$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

If $\sigma\in T_c$ and $\rho\sqsubseteq\sigma$, then every $\tau\sqsubseteq\rho$ is also a prefix of $\sigma$, so $\rho\in T_c$. Hence $T_c$ is a binary tree.

Moreover $K$ is computable in $\emptyset'$: with the halting oracle, one can decide which programs halt and search for the shortest one producing a given word. Therefore the finite condition

$$\forall \tau\sqsubseteq\sigma, K(\tau)\le K(\operatorname{bin}(l(\tau)))+c$$

is decidable in $\emptyset'$. Thus $T_c$ is $\emptyset'$-computable.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Prefix-free Kolmogorov complexity is computable in $\emptyset'$)</span></p>

$K$ is computable in $\emptyset'$: with the halting oracle, one can decide which programs halt and search for the shortest one producing a given word. Therefore the finite condition

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Every computable real is K-trivial; every K-trivial real is c.a.(3))</span></p>

Show that there exists an upper bound $N$ such that

$$\sharp \lbrace\sigma : \ell(\sigma) = n \text{ and } \sigma \in T_c\rbrace \leq N$$

for all $n$.

*(hint: use Chaitin’s Counting theorem)*

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Let $l(\sigma)=n$ and $\sigma\in T_c$. Then

$$K(\sigma)\le K(\operatorname{bin}(n))+c.$$

Chaitin’s counting theorem says that for some constant $d$,

$$\sharp \lbrace \sigma\in \lbrace 0,1\rbrace^n: K(\sigma)\le n+K(\operatorname{bin}(n))-r\rbrace \le 2^{n-r+d}.$$

Put $r=n-c$. Then

$$\sharp\lbrace\sigma:l(\sigma)=n,\ \sigma\in T_c\rbrace \le 2^{c+d}$$

for all sufficiently large $n$. Enlarging the constant to cover the finitely many small $n$, we get a uniform bound $N$.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Using a variable parameter to obtain a uniform bound)</span></p>

In the proof of the level-size bound for $T_c$, we use Chaitin's counting theorem in a uniform form:

$$\exists d\ \forall n\ \forall r:\quad \sharp \lbrace \sigma\in 2^n : K(\sigma)\le n+K(\operatorname{bin}(n))-r\rbrace \le 2^{n-r+d}.$$

Since the estimate holds for every pair $(n,r)$, we are allowed to choose $r$ depending on $n$. For strings $\sigma\in T_c$ of length $n$, we have

$$K(\sigma)\le K(\operatorname{bin}(n))+c = n+K(\operatorname{bin}(n))-(n-c).$$

Thus we set

$$r(n):=n-c.$$

Although $r$ varies with $n$, the former constant $r$ becomes a variable, the resulting upper bound becomes

$$2^{n-r(n)+d} = 2^{n-(n-c)+d} = 2^{c+d},$$

which is independent of $n$. This is the key point: the variable choice of $r$ is used precisely to cancel the $n$-dependence in the counting theorem. Hence one obtains a single uniform bound on the number of nodes of $T_c$ at each level.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Every computable real is K-trivial; every K-trivial real is c.a.(4))</span></p>

Show that an infinite binary sequence $A$ that satisfies

$$K(A \upharpoonright n) \leq K(\operatorname{bin}(n)) + c$$

for all $n$ is an isolated path in $T_c$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Assume $A$ is **not** isolated. Then for every prefix length $n$, there is another infinite path $B\neq A$ with

$$B\upharpoonright n=A\upharpoonright n.$$

Since $B\neq A$, it must eventually split away from $A$. So beyond every $n$, there is some splitting node on the path $A$.

**If $A$ has more than $N$ splitting points along it, can you choose one side-branch from each split and follow it to one common high level?**

Each chosen side-branch gives a distinct node at that common high level. Together with the prefix of $A$ itself, this would produce more than $N$ nodes on one level of $T_c$, contradicting Exercise 3.3.

The key phrase to make rigorous is: side-branches created at different split points remain distinct at every sufficiently high common level. It is guaranteed by the construction of the binary tree on words based on prefixes: two different branches never merge. 

More regorous way:

Assume $A$ is not isolated. Then along $A$ there are infinitely many splitting points. Choose $N$ distinct splitting points

$$s_1<s_2<\dots<s_N.$$

At each split $s_j$, choose an infinite path $B_j$ that follows the side-branch different from $A$. Then choose a level

$$L>\max_j s_j.$$

Since each $B_j$ is infinite, each prefix $B_j\upharpoonright L$ lies in $T_c$. Also $A\upharpoonright L\in T_c$. These $N+1$ nodes are distinct, because branches in a prefix tree never merge.

Thus level $L$ has at least $N+1$ nodes, contradicting the level bound from Exercise 3.3. So $A$ must be isolated.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Every computable real is K-trivial; every K-trivial real is c.a.(5))</span></p>

Show that every isolated path in a computable tree is computable.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Let $A$ be an isolated path in a computable tree $T$. Let $\rho\sqsubset A$ be an isolating prefix. We may hard-code $\rho$.

To compute $A\upharpoonright n$, search for a level $m\ge n$ such that all nodes $\tau\in T$ of length $m$ extending $\rho$ have the same prefix of length $n$. Such an $m$ must exist; otherwise, by König’s lemma, there would be another infinite path extending $\rho$, contradicting isolation. Since $T$ is computable, the finite search at each level is effective. Output the common length-$n$ prefix. Thus $A$ is computable. $\squared$

> In the solution for the exercise 3.5, there is the following phrase "we hardcode the prefix the makes the path isolated". Is it possible? It works for a single isolated pass, but I do not see why is succulent to systematically compute all isolated paths?

My objection is exactly the right distinction.

The phrase “hardcode the isolating prefix” is legitimate because Exercise 3.5 is **non-uniform**:

$$\text{For every isolated path }A\text{ in a computable tree }T,\ A\text{ is computable.}$$

This means: for each particular isolated path $A$, there exists a Turing machine computing $A$. The machine is allowed to depend on $A$, so it may contain the isolating prefix $\rho$ as finite built-in information.

It does **not** mean:

> Given $T$, systematically find all isolated paths.

That stronger uniform task is not what is being proved.

How the hardcoded prefix is used:

Let $\rho\sqsubset A$ isolate $A$. To compute $A\upharpoonright m$, search levels $L=m,m+1,m+2,\dots$ of the computable tree $T$. Look at all nodes $\sigma\in T$ of length $L$ extending $\rho$.

Because $\rho$ isolates $A$, all wrong side-branches above $\rho$ are finite. So eventually, for large enough $L$, every surviving node extending $\rho$ has the same prefix of length $m$, namely $A\upharpoonright m$.

Then output that common prefix.

So the algorithm is:

$$m \mapsto \text{search until all length-}L\text{ extensions of }\rho \text{ agree on their first }m\text{ bits.}$$

This search halts because $\rho$ isolates $A$.

So, corrected understanding should be:

> We do not compute the isolating prefix from the tree. We fix one isolated path $A$, use its isolating prefix as finite advice, and then compute the rest effectively from the computable tree.

That is enough for Exercise 3.5. The sheet only asks to show that every isolated path in a computable tree is computable. 

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Hardcoded finite advice for infinite properties)</span></p>

Some infinite properties can be reduced, non-uniformly, to finite information. For example, an isolated path $A$ through a computable tree has a finite prefix $\rho\sqsubset A$ such that no other infinite path extends $\rho$. Once this prefix is known, it can be hardcoded into a machine.

The machine does not need to know the whole path $A$. Given a requested length $m$, it searches sufficiently long levels of the computable tree above $\rho$ until all surviving extensions agree on their first $m$ bits. Since $\rho$ isolates $A$, this search eventually stabilizes and outputs $A\upharpoonright m$.

This is a **non-uniform** use of finite advice. It proves that each isolated path is computable, but it does not necessarily give a uniform procedure which, from the tree alone, finds the isolating prefix or lists all isolated paths. The finite prefix can be used effectively once supplied; finding or verifying that it is isolating may still involve an infinite property.

</div>


<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Every computable real is K-trivial; every K-trivial real is c.a.(6))</span></p>

Show that every K-trivial real is computably approximable.

*(hint: Relativize the latter step. Recall that the binary representation of every c.a. real lies in $\Delta^0_2$)*

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Let $\alpha=0.A$ be K-trivial. Then for some $c$,

$$K(A\upharpoonright n)\le K(\operatorname{bin}(n))+c$$

for all $n$. By Exercise 3.4, $A$ is an isolated path in $T_c$. By Exercise 3.2, $T_c$ is $\emptyset'$-computable. Relativizing Exercise 3.5, every isolated path in an $\emptyset'$-computable tree is $\emptyset'$-computable. Hence

$$A\le_T\emptyset',$$

so $A\in\Delta^0_2$. By the course theorem that c.a. reals are exactly the reals with $\Delta^0_2$ binary expansion, $\alpha$ is computably approximable.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Finite evidence versus infinite properties)</span></p>

A recurring pattern in computability theory and algorithmic randomness is the distinction between **finite verifiable evidence** and genuinely **infinite properties**.

If a property has a finite positive witness, then it is often computably enumerable: one can search for the witness and eventually find it if it exists. For example, $K(\sigma)\le k$ is c.e., because it is witnessed by a program of length at most $k$ that halts and outputs $\sigma$. By contrast, $K(\sigma)>k$ requires knowing that no such program ever appears, which is not directly verifiable in finite time.

Similarly, for a computable tree $T$, membership $\sigma\in T$ is a finite property, but the assertion that $\sigma$ has an infinite continuation is not generally decidable. Thus, when proving that an isolated path is computable, we may hardcode an isolating prefix for a particular path. This is a non-uniform argument: it proves that each isolated path is computable, but it does not give a procedure that finds all isolated paths from the tree alone.

In short, finite witnesses often give c.e. information, while absence, uniqueness, or infinite continuation usually requires extra structure, an oracle, compactness, or non-uniform finite advice.

</div>

### Basics of Computability Theory and Prefix-Free Machines

The randomness proof for $\Omega$ rests on a small toolkit of classical computability results — the $s$-$m$-$n$ theorem, the fixed-point theorem, and two forms of Kleene's recursion theorem. We collect them here, the last one in the prefix-free form that the proof actually uses.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(5.12 — $s$-$m$-$n$ theorem)</span></p>

Let $g(\cdot,\cdot)$ be a partially computable two-argument function on words. Then there exists a *totally* computable function $s(\cdot)$ such that

$$M_{s(x)}(y) = g(x,y) \qquad \text{for all } x, y.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $M$ be a Turing machine computing $g$, and fix a word $x$. Build a machine $N$ that, on input $y$, first writes the pair $(x, y)$ onto the input tape of $M$ and then simulates $M$ on it. An index for $N$ can be computed from $x$; call it $s(x)$. Then $s$ is total computable and $M\_{s(x)}(y) = M(x,y) = g(x,y)$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(5.13 — Fixed-point theorem)</span></p>

Let $f$ be a totally computable function. Then there exists an index $e$ such that

$$M_e(x) = M_{f(e)}(x) \qquad \text{for all } x.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By the $s$-$m$-$n$ theorem, fix a total computable function $d$ such that

$$M_{d(e)}(k) := M_{M_e(e)}(k) \quad \text{whenever } M_e(e)\downarrow$$

(and $M\_{d(e)}(k)\uparrow$ in case $M\_e(e)\uparrow$). Choose $i$ with $M\_i = f \circ d$ and put $n = d(i)$. Since $M\_i$ is total, $M\_i(i) = (f\circ d)(i) = f(n)$, so

$$M_n = M_{d(i)} = M_{M_i(i)} = M_{f(n)}.$$

Thus $n$ is a fixed point of $f$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(5.14 — Kleene's second recursion theorem)</span></p>

Let $g(\cdot,\cdot)$ be a partially computable two-argument function of words. Then there exists an index $e$ such that

$$M_e(y) = g(e,y) \qquad \text{for all } y.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

By the $s$-$m$-$n$ theorem, fix a total computable function $s(\cdot)$ with $g(x,y) = M\_{s(x)}(y)$ for all $x, y$. Applying the fixed-point theorem to $s$ yields an index $e$ with $M\_e(y) = M\_{s(e)}(y) = g(e,y)$ for every $y$.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(5.15 — Recursion theorem for prefix-free machines)</span></p>

Let

$$h : \subseteq \lbrace 0,1\rbrace^\ast \times \mathbb{N} \to \lbrace 0,1\rbrace^\ast$$

be a partial computable function such that, for each $e$, the section $h(\cdot, e)$ is prefix-free. Then there exists an index $e$ such that

$$\widetilde M_e = h(\cdot, e).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Apply the recursion theorem (Theorem 5.14) to $h$, read as $g(e, y) := h(y, e)$: there is an index $e$ with $M\_e(\cdot) = h(\cdot, e)$. Since $h(\cdot, e)$ is prefix-free by hypothesis, the machine $M\_e$ is prefix-free; so, by the construction in the proof of Theorem 2.20, its prefix-free counterpart $\widetilde M\_e$ coincides with $M\_e$. Hence $\widetilde M\_e = h(\cdot, e)$.

</details>
</div>

## Randomness of Reals

In algorithmic randomness a real in $[0,1]$ is studied through its binary expansion. For a non-dyadic real this expansion is unique. For a dyadic rational there are two binary names, such as

$$0.1000\dots = 0.0111\dots,$$

and we use the terminating expansion by convention. This convention does not affect randomness: dyadic rationals have computable binary expansions, hence are never Martin-Löf random.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(6.1 — Martin-Löf random real)</span></p>

A real $\alpha \in [0,1]$ is **Martin-Löf random** if its chosen binary expansion $A$ is a Martin-Löf random sequence. Equivalently, by the Levin-Schnorr characterization,

$$\exists c\ \forall n\qquad K(\alpha \upharpoonright n) \ge n - c,$$

where $\alpha \upharpoonright n$ denotes the first $n$ bits of the binary expansion of $\alpha$.

</div>

The geometric picture is the same as for Cantor space: a finite prefix $\sigma$ names the dyadic interval of all reals whose binary expansion begins with $\sigma$. Randomness of a real says that no effective sequence of very small open covers can keep trapping the point.

<figure class="math-figure">
  <svg viewBox="0 0 660 330" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:660px" aria-label="Binary prefixes of a real as nested dyadic intervals">
    <g font-family="serif" font-size="12" fill="#1f2430">
      <text x="330" y="25" text-anchor="middle" font-weight="600">A binary expansion is a nested address on [0, 1]</text>

      <!-- Number line -->
      <line x1="60" y1="255" x2="600" y2="255" stroke="#444" stroke-width="1.2" />
      <g stroke="#444" stroke-width="1">
        <line x1="60" y1="249" x2="60" y2="263" />
        <line x1="195" y1="249" x2="195" y2="263" />
        <line x1="330" y1="249" x2="330" y2="263" />
        <line x1="465" y1="249" x2="465" y2="263" />
        <line x1="600" y1="249" x2="600" y2="263" />
      </g>
      <g font-size="11" fill="#444" text-anchor="middle">
        <text x="60" y="280">0</text>
        <text x="195" y="280">1/4</text>
        <text x="330" y="280">1/2</text>
        <text x="465" y="280">3/4</text>
        <text x="600" y="280">1</text>
      </g>

      <!-- Nested prefix intervals for A starts 101 -->
      <rect x="330" y="70" width="270" height="28" fill="rgba(29,78,216,0.08)" stroke="#1d4ed8" stroke-width="1.5" />
      <text x="465" y="89" text-anchor="middle">prefix 1: interval [1/2, 1], length 2^-1</text>

      <rect x="330" y="115" width="135" height="28" fill="rgba(60,120,40,0.10)" stroke="#3d7a26" stroke-width="1.5" />
      <text x="397.5" y="134" text-anchor="middle">prefix 10: length 2^-2</text>

      <rect x="397.5" y="160" width="67.5" height="28" fill="rgba(168,111,0,0.10)" stroke="#a86f00" stroke-width="1.5" />
      <text x="431.25" y="179" text-anchor="middle" font-size="11">prefix 101</text>

      <rect x="414.375" y="202" width="16.875" height="22" fill="rgba(214,83,54,0.10)" stroke="#d65336" stroke-width="1.4" />
      <text x="422.8" y="218" text-anchor="middle" font-size="10">1011...</text>

      <!-- Projection lines -->
      <g stroke="#5b6270" stroke-dasharray="2 3" stroke-width="0.8">
        <line x1="330" y1="98" x2="330" y2="255" />
        <line x1="600" y1="98" x2="600" y2="255" />
        <line x1="397.5" y1="188" x2="397.5" y2="255" />
        <line x1="465" y1="188" x2="465" y2="255" />
      </g>

      <!-- alpha marker -->
      <line x1="423" y1="54" x2="423" y2="263" stroke="#d65336" stroke-width="1.7" />
      <circle cx="423" cy="255" r="4" fill="#d65336" />
      <text x="423" y="300" text-anchor="middle" fill="#d65336" font-weight="600">alpha = 0.1011...</text>

      <!-- Dyadic ambiguity callout -->
      <g>
        <rect x="62" y="58" width="220" height="78" rx="6" fill="rgba(91,98,112,0.06)" stroke="#cbd2e0" />
        <text x="172" y="82" text-anchor="middle" font-weight="600" fill="#5b6270">dyadic boundary</text>
        <text x="172" y="103" text-anchor="middle" font-size="11">0.1000... = 0.0111...</text>
        <text x="172" y="122" text-anchor="middle" font-size="11">both names are computable</text>
      </g>
      <line x1="330" y1="137" x2="330" y2="255" stroke="#5b6270" stroke-dasharray="5 4" stroke-width="1.2" />
    </g>
  </svg>
  <figcaption>A real $\alpha = 0.A$ is located by nested dyadic intervals determined by the prefixes $A \upharpoonright n$. The cylinder $[\![\sigma]\!]$ in Cantor space becomes the interval of reals whose binary expansion begins with $\sigma$. Dyadic endpoints have two names, but these exceptional points are computable and therefore irrelevant to Martin-Löf randomness.</figcaption>
</figure>

The same definition can be stated directly on the real line. A **Martin-Löf test for reals** is a uniformly c.e. sequence of open sets $(U_i)_{i\ge 1}$ in $[0,1]$, each presented as a c.e. union of dyadic intervals, such that

$$\lambda(U_i) \le 2^{-i}.$$

A real $\alpha$ is Martin-Löf random iff

$$\alpha \notin \bigcap_i U_i$$

for every such test. Under the identification $[[\sigma]] \leftrightarrow [0.\sigma,\ 0.\sigma + 2^{-l(\sigma)}]$, this is exactly the Cantor-space definition transported to the unit interval.

<figure class="math-figure">
  <svg viewBox="0 0 680 360" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:680px" aria-label="Martin-Lof tests on reals as shrinking open covers">
    <g font-family="serif" font-size="12" fill="#1f2430">
      <text x="340" y="26" text-anchor="middle" font-weight="600">A real fails randomness if an effective null test traps it forever</text>

      <!-- Layer labels -->
      <text x="42" y="82" text-anchor="end" font-weight="600" fill="#1d4ed8">U1</text>
      <text x="42" y="152" text-anchor="end" font-weight="600" fill="#3d7a26">U2</text>
      <text x="42" y="222" text-anchor="end" font-weight="600" fill="#a86f00">U3</text>

      <!-- Axes for each layer -->
      <g stroke="#444" stroke-width="1">
        <line x1="60" y1="82" x2="620" y2="82" />
        <line x1="60" y1="152" x2="620" y2="152" />
        <line x1="60" y1="222" x2="620" y2="222" />
      </g>

      <!-- U1 intervals total schematic <= 1/2 -->
      <g fill="rgba(29,78,216,0.12)" stroke="#1d4ed8" stroke-width="1.4">
        <rect x="105" y="66" width="80" height="32" />
        <rect x="315" y="66" width="190" height="32" />
      </g>
      <text x="530" y="86" font-size="11" fill="#1d4ed8">measure at most 1/2</text>

      <!-- U2 intervals total <= 1/4 -->
      <g fill="rgba(60,120,40,0.12)" stroke="#3d7a26" stroke-width="1.4">
        <rect x="133" y="136" width="44" height="32" />
        <rect x="370" y="136" width="90" height="32" />
      </g>
      <text x="530" y="156" font-size="11" fill="#3d7a26">measure at most 1/4</text>

      <!-- U3 intervals total <= 1/8 -->
      <g fill="rgba(168,111,0,0.13)" stroke="#a86f00" stroke-width="1.4">
        <rect x="148" y="206" width="20" height="32" />
        <rect x="404" y="206" width="42" height="32" />
      </g>
      <text x="530" y="226" font-size="11" fill="#a86f00">measure at most 1/8</text>

      <!-- Trapped alpha marker -->
      <line x1="420" y1="48" x2="420" y2="260" stroke="#d65336" stroke-width="1.8" />
      <circle cx="420" cy="82" r="4" fill="#d65336" />
      <circle cx="420" cy="152" r="4" fill="#d65336" />
      <circle cx="420" cy="222" r="4" fill="#d65336" />
      <text x="420" y="284" text-anchor="middle" fill="#d65336" font-weight="600">nonrandom alpha</text>
      <text x="420" y="304" text-anchor="middle" fill="#5b6270" font-size="11">alpha lies in every U_i</text>

      <!-- Escaping beta marker -->
      <line x1="250" y1="48" x2="250" y2="260" stroke="#5b6270" stroke-width="1.3" stroke-dasharray="4 4" />
      <text x="250" y="284" text-anchor="middle" fill="#5b6270" font-weight="600">random candidate beta</text>
      <text x="250" y="304" text-anchor="middle" fill="#5b6270" font-size="11">escapes this test</text>

      <!-- Bottom scale -->
      <line x1="60" y1="330" x2="620" y2="330" stroke="#444" stroke-width="1.2" />
      <g stroke="#444" stroke-width="1">
        <line x1="60" y1="324" x2="60" y2="338" />
        <line x1="340" y1="324" x2="340" y2="338" />
        <line x1="620" y1="324" x2="620" y2="338" />
      </g>
      <g text-anchor="middle" font-size="11" fill="#444">
        <text x="60" y="352">0</text>
        <text x="340" y="352">1/2</text>
        <text x="620" y="352">1</text>
      </g>
    </g>
  </svg>
  <figcaption>A Martin-Löf test on $[0,1]$ is an effective sequence of open covers whose measures shrink like $2^{-i}$. A nonrandom real is one that remains inside every layer of at least one such test. A random real need not avoid every interval in every layer, but for each test it must miss at least one layer.</figcaption>
</figure>

This point of view also explains why computable reals cannot be random. If a computable procedure prints the first $n$ bits of $\alpha$, then the single interval determined by that prefix has measure $2^{-n}$. Taking one such interval at each level gives an effective Martin-Löf test that covers $\alpha$.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Computable reals are null-test visible)</span></p>

If $\alpha$ is computable, then the sequence

$$U_i := [\![\alpha \upharpoonright i]\!]$$

is a Martin-Löf test, since $\lambda(U_i)=2^{-i}$ and the intervals are uniformly computable. Moreover $\alpha \in U_i$ for every $i$. Hence no computable real is Martin-Löf random.

</div>

With the toolkit above in hand, we can finally exhibit the promised *natural* example of a random real — Chaitin's $\Omega$. Its approximation is left-c.e., but the increments encode halting information so densely that no effective null test can predict all of its bits.

<figure class="math-figure">
  <svg viewBox="0 0 680 360" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:680px" aria-label="Chaitin Omega as a monotone halting probability approximation">
    <g font-family="serif" font-size="12" fill="#1f2430">
      <text x="340" y="26" text-anchor="middle" font-weight="600">Omega is approached from below by accumulating halting weight</text>

      <!-- Staircase axes -->
      <line x1="70" y1="285" x2="620" y2="285" stroke="#444" stroke-width="1.2" />
      <line x1="70" y1="60" x2="70" y2="285" stroke="#444" stroke-width="1.2" />
      <text x="620" y="306" text-anchor="end" font-size="11" fill="#444">stage s</text>
      <text x="50" y="64" text-anchor="end" font-size="11" fill="#444">value</text>

      <!-- Omega line -->
      <line x1="70" y1="90" x2="620" y2="90" stroke="#d65336" stroke-width="1.7" />
      <text x="626" y="94" font-size="12" fill="#d65336" font-weight="600">Omega</text>

      <!-- Staircase w_s -->
      <polyline points="70,270 130,270 130,240 205,240 205,220 285,220 285,180 360,180 360,158 450,158 450,122 535,122 535,108 620,108" fill="none" stroke="#1d4ed8" stroke-width="2.4" />
      <g fill="#1d4ed8">
        <circle cx="130" cy="240" r="3.5" />
        <circle cx="205" cy="220" r="3.5" />
        <circle cx="285" cy="180" r="3.5" />
        <circle cx="360" cy="158" r="3.5" />
        <circle cx="450" cy="122" r="3.5" />
        <circle cx="535" cy="108" r="3.5" />
      </g>
      <text x="330" y="320" text-anchor="middle" fill="#1d4ed8" font-weight="600">w_s = sum of weights of programs seen halting by stage s</text>

      <!-- Gap annotation -->
      <line x1="560" y1="90" x2="560" y2="108" stroke="#a86f00" stroke-width="1.6" />
      <line x1="552" y1="90" x2="568" y2="90" stroke="#a86f00" stroke-width="1.6" />
      <line x1="552" y1="108" x2="568" y2="108" stroke="#a86f00" stroke-width="1.6" />
      <text x="548" y="104" text-anchor="end" fill="#a86f00" font-size="11">unknown remaining weight</text>

      <!-- Halting programs boxes -->
      <g>
        <rect x="90" y="48" width="116" height="34" rx="5" fill="rgba(29,78,216,0.07)" stroke="#1d4ed8" />
        <text x="148" y="69" text-anchor="middle" font-size="11">program halts</text>
        <line x1="148" y1="82" x2="130" y2="236" stroke="#1d4ed8" stroke-dasharray="3 3" />
      </g>
      <g>
        <rect x="230" y="48" width="116" height="34" rx="5" fill="rgba(29,78,216,0.07)" stroke="#1d4ed8" />
        <text x="288" y="69" text-anchor="middle" font-size="11">more weight</text>
        <line x1="288" y1="82" x2="285" y2="176" stroke="#1d4ed8" stroke-dasharray="3 3" />
      </g>
      <g>
        <rect x="400" y="48" width="126" height="34" rx="5" fill="rgba(29,78,216,0.07)" stroke="#1d4ed8" />
        <text x="463" y="69" text-anchor="middle" font-size="11">late short program?</text>
        <line x1="463" y1="82" x2="450" y2="118" stroke="#1d4ed8" stroke-dasharray="3 3" />
      </g>

      <!-- Stable bits bar -->
      <g>
        <rect x="96" y="200" width="62" height="18" fill="rgba(91,98,112,0.08)" stroke="#cbd2e0" />
        <text x="127" y="214" text-anchor="middle" font-size="10">few bits stable</text>
        <rect x="390" y="200" width="150" height="18" fill="rgba(60,120,40,0.10)" stroke="#3d7a26" />
        <text x="465" y="214" text-anchor="middle" font-size="10">longer prefix stabilized</text>
      </g>
    </g>
  </svg>
  <figcaption>The canonical approximation $w_s \nearrow \Omega$ adds the weights $2^{-l(\sigma)}$ of programs that have halted by stage $s$. Knowing a long prefix of $\Omega$ bounds the remaining unseen halting weight, which decides halting for all sufficiently short programs. Conversely, the proof below shows that if those prefixes were compressible, one could force a late halting contribution large enough to contradict the alleged stability of the prefix.</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(6.2 — Chaitin's $\Omega$ is Martin-Löf random)</span></p>

Chaitin's $\Omega$ is Martin-Löf random.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

For every constant $c$ we define a partial computable function $f^c$ from words to words, enumerating its graph stage by stage.

**Stage $s$.** If there exist a number $n \le s$ and a word $\tau$ such that

$$K(\tau) < l(\tau) - n \quad\text{and}\quad \widetilde U(\tau)[s]\downarrow = w_s \upharpoonright n,$$

then choose a word $\sigma$ *not* in the range of $\widetilde U[s]$ and set $f^c(\tau) := \sigma$.

(Formally, $f^c$ is computed by enumerating its domain and, whenever $\tau$ appears at some enumeration step, returning the value assigned to it.)

Each $f^c$ is prefix-free by construction, and the family is uniform in $c$, so the recursion theorem for prefix-free machines (Theorem 5.15) applies to $(f^0, f^1, \dots)$: there is an index $e$ with

$$\widetilde M_e(\tau) = f^e(\tau) \qquad \text{for all } \tau.$$

We claim that $K(\Omega \upharpoonright n) \ge n - e$ for every $n$. Suppose not, for some $n$. Then $\Omega \upharpoonright n$ has a short code $\tau$, i.e.

$$l(\tau) < n - e \quad\text{and}\quad \widetilde U(\tau)\downarrow = \Omega \upharpoonright n.$$

Since $\Omega$ is noncomputable — in particular not a dyadic rational — there is a stage $s$ with

$$\widetilde U(\tau)[s]\downarrow = w_s \upharpoonright n = \Omega \upharpoonright n. \tag{$\ast$}$$

At that stage the defining condition of $f^e$ is satisfied by this $n$ and $\tau$, so $f^e(\tau) = \sigma$ is defined with $\sigma \notin \operatorname{range}(\widetilde U[s])$. Because $\widetilde M\_e$ is simulated by $\widetilde U$ with coding constant $e$, there is a word $\nu$ with

$$l(\nu) < l(\tau) + e \quad\text{and}\quad \widetilde U(\nu) = \widetilde M_e(\tau) = \sigma.$$

As $\sigma \notin \operatorname{range}(\widetilde U[s])$, we have $\nu \notin \operatorname{dom}(\widetilde U[s])$, hence $\nu \in \operatorname{dom}(\widetilde U) \setminus \operatorname{dom}(\widetilde U[s])$. Therefore

$$\Omega - w_s = \sum_{\rho \,\in\, \operatorname{dom}(\widetilde U)\,\setminus\,\operatorname{dom}(\widetilde U[s])} 2^{-l(\rho)} \ge 2^{-l(\nu)},$$

so $\Omega$ and $w\_s$ must already disagree within their first $l(\nu)$ bits: $\Omega \upharpoonright l(\nu) \ne w\_s \upharpoonright l(\nu)$. But

$$l(\nu) < l(\tau) + e < (n - e) + e = n,$$

so $(\ast)$ forces $\Omega \upharpoonright l(\nu) = w\_s \upharpoonright l(\nu)$ — a contradiction.

Hence $K(\Omega \upharpoonright n) \ge n - e$ for all $n$; $\Omega$ is incompressible up to an additive constant, and therefore Martin-Löf random.

</details>
</div>

## Solovay Reducibility

The fact that $\Omega$ is random is not an isolated accident: any left-c.e. real whose approximation converges "at least as fast" as that of $\Omega$ inherits its randomness. The right notion of "at least as fast" is *Solovay reducibility*.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(7.1 — Solovay reducibility)</span></p>

A real $\alpha$ is **Solovay reducible** to a real $\beta$, written $\alpha \le\_S \beta$, if there exist a constant $c > 0$ and a partial computable function $g : \subseteq \mathbb{Q} \to \mathbb{Q}$ such that

$$0 < \alpha - g(q)\downarrow < c(\beta - q) \qquad \text{for every } q < \beta.$$

</div>

The function $g$ turns every rational lower bound $q < \beta$ into a rational lower bound $g(q) < \alpha$. The inequality says more than preservation of lower bounds: the remaining gap below $\alpha$ is controlled by the remaining gap below $\beta$, up to the fixed multiplier $c$.

<figure class="math-figure">
  <svg viewBox="0 0 700 360" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:700px" aria-label="Solovay reducibility as a computable map between lower cuts with controlled gaps">
    <g font-family="serif" font-size="12" fill="#1f2430">
      <text x="350" y="28" text-anchor="middle" font-weight="600">Solovay reducibility: map lower bounds for beta to lower bounds for alpha</text>

      <!-- beta axis -->
      <line x1="75" y1="112" x2="630" y2="112" stroke="#444" stroke-width="1.2" />
      <text x="75" y="95" text-anchor="middle" font-weight="600" fill="#3d7a26">0</text>
      <text x="630" y="95" text-anchor="middle" font-weight="600" fill="#3d7a26">1</text>
      <text x="42" y="116" text-anchor="end" font-weight="600" fill="#3d7a26">beta-line</text>

      <!-- beta q and beta -->
      <line x1="392" y1="88" x2="392" y2="136" stroke="#3d7a26" stroke-width="1.6" />
      <circle cx="392" cy="112" r="4" fill="#3d7a26" />
      <text x="392" y="75" text-anchor="middle" fill="#3d7a26" font-weight="600">q</text>
      <line x1="540" y1="76" x2="540" y2="148" stroke="#d65336" stroke-width="1.8" />
      <circle cx="540" cy="112" r="4" fill="#d65336" />
      <text x="540" y="75" text-anchor="middle" fill="#d65336" font-weight="600">beta</text>

      <!-- beta gap -->
      <line x1="392" y1="142" x2="540" y2="142" stroke="#a86f00" stroke-width="1.5" />
      <line x1="392" y1="136" x2="392" y2="148" stroke="#a86f00" stroke-width="1.5" />
      <line x1="540" y1="136" x2="540" y2="148" stroke="#a86f00" stroke-width="1.5" />
      <text x="466" y="163" text-anchor="middle" fill="#a86f00">beta - q</text>

      <!-- alpha axis -->
      <line x1="75" y1="258" x2="630" y2="258" stroke="#444" stroke-width="1.2" />
      <text x="75" y="241" text-anchor="middle" font-weight="600" fill="#1d4ed8">0</text>
      <text x="630" y="241" text-anchor="middle" font-weight="600" fill="#1d4ed8">1</text>
      <text x="42" y="262" text-anchor="end" font-weight="600" fill="#1d4ed8">alpha-line</text>

      <!-- alpha g(q) and alpha -->
      <line x1="292" y1="234" x2="292" y2="282" stroke="#1d4ed8" stroke-width="1.6" />
      <circle cx="292" cy="258" r="4" fill="#1d4ed8" />
      <text x="292" y="314" text-anchor="middle" fill="#1d4ed8" font-weight="600">g(q)</text>
      <line x1="408" y1="222" x2="408" y2="294" stroke="#d65336" stroke-width="1.8" />
      <circle cx="408" cy="258" r="4" fill="#d65336" />
      <text x="408" y="314" text-anchor="middle" fill="#d65336" font-weight="600">alpha</text>

      <!-- alpha gap -->
      <line x1="292" y1="216" x2="408" y2="216" stroke="#a86f00" stroke-width="1.5" />
      <line x1="292" y1="210" x2="292" y2="222" stroke="#a86f00" stroke-width="1.5" />
      <line x1="408" y1="210" x2="408" y2="222" stroke="#a86f00" stroke-width="1.5" />
      <text x="350" y="204" text-anchor="middle" fill="#a86f00">alpha - g(q)</text>

      <!-- computable map arrow -->
      <path d="M392 130 C390 175 335 200 298 236" fill="none" stroke="#5b6270" stroke-width="1.6" marker-end="url(#arrow-solovay-map)" />
      <text x="325" y="178" text-anchor="middle" fill="#5b6270" font-weight="600">partial computable g</text>

      <!-- inequality callout -->
      <g>
        <rect x="440" y="190" width="210" height="72" rx="6" fill="rgba(168,111,0,0.08)" stroke="#a86f00" />
        <text x="545" y="216" text-anchor="middle" font-weight="600" fill="#a86f00">controlled error</text>
        <text x="545" y="238" text-anchor="middle" font-size="12">alpha - g(q) &lt; c(beta - q)</text>
      </g>

      <!-- lower cut regions -->
      <rect x="75" y="104" width="317" height="16" fill="rgba(60,120,40,0.10)" />
      <rect x="75" y="250" width="217" height="16" fill="rgba(29,78,216,0.10)" />
      <text x="205" y="92" text-anchor="middle" fill="#3d7a26" font-size="11">known lower cut below beta</text>
      <text x="175" y="242" text-anchor="middle" fill="#1d4ed8" font-size="11">produced lower cut below alpha</text>

      <defs>
        <marker id="arrow-solovay-map" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
          <path d="M0,0 L10,4 L0,8 Z" fill="#5b6270" />
        </marker>
      </defs>
    </g>
  </svg>
  <figcaption>Solovay reducibility compares how much uncertainty remains after taking a rational lower approximation. Given any $q < \beta$, the computable map $g$ produces $g(q) < \alpha$, and the remaining $\alpha$-gap is at most a fixed multiple of the remaining $\beta$-gap. Thus better lower bounds for $\beta$ can be translated effectively into comparably good lower bounds for $\alpha$.</figcaption>
</figure>

Since $g$ maps the left cut of $\beta$ into the left cut of $\alpha$, it is natural to restrict this reducibility to the class of left-c.e. reals. There it turns into a statement about *speed of convergence*: informally, $\alpha \le\_S \beta$ holds iff some left-c.e. approximation of $\alpha$ converges no slower than a given left-c.e. approximation of $\beta$.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(7.2 — Speed-of-convergence characterizations)</span></p>

Let $\alpha$ and $\beta$ be left-c.e. reals. The following are equivalent.

- **(i)** $\alpha \le\_S \beta$.
- **(ii)** There is a constant $c$ such that, for every pair of left-c.e. approximations $(a\_n)\_{n\in\mathbb{N}}$ and $(b\_n)\_{n\in\mathbb{N}}$ of $\alpha$ and $\beta$, there is a computable function $g : \mathbb{N} \to \mathbb{N}$ with $\alpha - a\_{g(n)} < c(\beta - b\_n)$ for all $n$.
- **(iii)** For every left-c.e. approximation $(b\_n)$ of $\beta$ there are a constant $d$ and a left-c.e. approximation $a\_0 < a\_1 < \cdots \to \alpha$ such that $a\_s - a\_{s-1} \le d(b\_s - b\_{s-1})$ for all $s$.
- **(iv)** There exist left-c.e. approximations $(a\_n)\_{n\in\mathbb{N}}$ and $(b\_n)\_{n\in\mathbb{N}}$ of $\alpha$ and $\beta$ and a constant $c$ such that $\alpha - a\_n < c(\beta - b\_n)$ for all $n$.

</div>

<figure class="math-figure">
  <svg viewBox="0 0 720 520" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:720px" aria-label="Four speed of convergence views equivalent to Solovay reducibility">
    <g font-family="serif" font-size="12" fill="#1f2430">
      <text x="360" y="28" text-anchor="middle" font-weight="600">Four equivalent ways to see the same speed comparison</text>

      <!-- Panel frames -->
      <g fill="none" stroke="#cbd2e0" stroke-width="1.1">
        <rect x="32" y="52" width="316" height="195" rx="6" />
        <rect x="372" y="52" width="316" height="195" rx="6" />
        <rect x="32" y="277" width="316" height="195" rx="6" />
        <rect x="372" y="277" width="316" height="195" rx="6" />
      </g>

      <!-- Panel i -->
      <text x="48" y="78" font-weight="600" fill="#5b6270">(i) cut map</text>
      <line x1="72" y1="115" x2="312" y2="115" stroke="#444" />
      <line x1="72" y1="188" x2="312" y2="188" stroke="#444" />
      <line x1="220" y1="98" x2="220" y2="132" stroke="#3d7a26" stroke-width="1.5" />
      <text x="220" y="92" text-anchor="middle" fill="#3d7a26">q</text>
      <line x1="286" y1="91" x2="286" y2="139" stroke="#d65336" stroke-width="1.7" />
      <text x="286" y="86" text-anchor="middle" fill="#d65336">beta</text>
      <line x1="138" y1="171" x2="138" y2="205" stroke="#1d4ed8" stroke-width="1.5" />
      <text x="138" y="222" text-anchor="middle" fill="#1d4ed8">g(q)</text>
      <line x1="190" y1="164" x2="190" y2="212" stroke="#d65336" stroke-width="1.7" />
      <text x="190" y="222" text-anchor="middle" fill="#d65336">alpha</text>
      <path d="M220 128 C205 150 170 160 142 174" fill="none" stroke="#5b6270" stroke-width="1.4" marker-end="url(#arrow-speed)" />
      <text x="206" y="158" text-anchor="middle" fill="#5b6270" font-size="11">g</text>
      <text x="194" y="236" text-anchor="middle" font-size="11" fill="#a86f00">tail(alpha) &lt; c tail(beta)</text>

      <!-- Panel ii -->
      <text x="388" y="78" font-weight="600" fill="#5b6270">(ii) arbitrary approximations</text>
      <line x1="412" y1="116" x2="652" y2="116" stroke="#444" />
      <line x1="412" y1="188" x2="652" y2="188" stroke="#444" />
      <polyline points="412,130 460,130 460,122 510,122 510,117 560,117 560,114 652,114" fill="none" stroke="#3d7a26" stroke-width="2" />
      <polyline points="412,205 445,205 445,198 505,198 505,192 580,192 580,189 652,189" fill="none" stroke="#1d4ed8" stroke-width="2" />
      <line x1="560" y1="96" x2="560" y2="130" stroke="#3d7a26" stroke-dasharray="3 3" />
      <line x1="580" y1="170" x2="580" y2="206" stroke="#1d4ed8" stroke-dasharray="3 3" />
      <line x1="650" y1="94" x2="650" y2="134" stroke="#d65336" />
      <line x1="650" y1="168" x2="650" y2="208" stroke="#d65336" />
      <text x="650" y="91" text-anchor="middle" fill="#d65336">beta</text>
      <text x="650" y="223" text-anchor="middle" fill="#d65336">alpha</text>
      <text x="540" y="152" text-anchor="middle" font-size="11" fill="#5b6270">choose later a_{g(n)}</text>
      <path d="M560 132 C562 150 574 158 580 170" fill="none" stroke="#5b6270" stroke-width="1.2" marker-end="url(#arrow-speed)" />
      <text x="532" y="236" text-anchor="middle" font-size="11" fill="#a86f00">alpha - a_{g(n)} &lt; c(beta - b_n)</text>

      <!-- Panel iii -->
      <text x="48" y="303" font-weight="600" fill="#5b6270">(iii) increment domination</text>
      <line x1="72" y1="430" x2="312" y2="430" stroke="#444" />
      <polyline points="72,428 110,428 110,410 148,410 148,383 190,383 190,369 234,369 234,350 312,350" fill="none" stroke="#3d7a26" stroke-width="2.1" />
      <polyline points="72,455 110,455 110,447 148,447 148,435 190,435 190,429 234,429 234,421 312,421" fill="none" stroke="#1d4ed8" stroke-width="2.1" />
      <g stroke="#3d7a26" stroke-width="1.3">
        <line x1="148" y1="410" x2="148" y2="383" />
        <line x1="190" y1="383" x2="190" y2="369" />
      </g>
      <g stroke="#1d4ed8" stroke-width="1.3">
        <line x1="148" y1="447" x2="148" y2="435" />
        <line x1="190" y1="435" x2="190" y2="429" />
      </g>
      <text x="196" y="334" text-anchor="middle" fill="#3d7a26">beta increments</text>
      <text x="195" y="486" text-anchor="middle" fill="#1d4ed8">alpha increments</text>
      <text x="196" y="318" text-anchor="middle" font-size="11" fill="#a86f00">Delta a_s &le; d Delta b_s</text>

      <!-- Panel iv -->
      <text x="388" y="303" font-weight="600" fill="#5b6270">(iv) one synchronized witness</text>
      <line x1="412" y1="380" x2="652" y2="380" stroke="#444" />
      <line x1="412" y1="435" x2="652" y2="435" stroke="#444" />
      <polyline points="412,395 452,395 452,389 498,389 498,384 545,384 545,381 652,381" fill="none" stroke="#3d7a26" stroke-width="2" />
      <polyline points="412,452 452,452 452,446 498,446 498,440 545,440 545,436 652,436" fill="none" stroke="#1d4ed8" stroke-width="2" />
      <line x1="545" y1="382" x2="652" y2="382" stroke="#a86f00" stroke-width="4" opacity="0.35" />
      <line x1="545" y1="437" x2="652" y2="437" stroke="#a86f00" stroke-width="4" opacity="0.35" />
      <line x1="652" y1="360" x2="652" y2="455" stroke="#d65336" stroke-width="1.5" />
      <text x="652" y="474" text-anchor="middle" fill="#d65336">limits</text>
      <text x="532" y="337" text-anchor="middle" fill="#5b6270" font-size="11">same index n, selected approximations</text>
      <text x="532" y="492" text-anchor="middle" fill="#a86f00" font-size="11">alpha - a_n &lt; c(beta - b_n)</text>

      <defs>
        <marker id="arrow-speed" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
          <path d="M0,0 L10,4 L0,8 Z" fill="#5b6270" />
        </marker>
      </defs>
    </g>
  </svg>
  <figcaption>The four clauses in Proposition 7.2 are different coordinate systems for the same comparison. Clause (i) talks about a computable map on lower cuts. Clause (ii) says that, no matter which left-c.e. approximations are chosen, a computable schedule can wait long enough on the $\alpha$ side to make its remaining tail comparable to the $\beta$ tail. Clause (iii) compares the sizes of the increments themselves. Clause (iv) packages the comparison into one pair of synchronized approximations.</figcaption>
</figure>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Sheet 5, Exercise 3 — the four implications are carried out in the exercise on the equivalent characterizations of Solovay reducibility for left-c.e. reals above.

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(7.3 — Solovay reducibility transfers randomness upward)</span></p>

If $\alpha$ is a Martin-Löf random real and $\alpha \le\_S \beta$, then $\beta$ is Martin-Löf random as well.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $\alpha$ be Martin-Löf random with $\alpha \le\_S \beta$ via a function $g$ and a constant which, without loss of generality, we take to be a power of two, $c = 2^N$:

$$0 < \alpha - g(q) < 2^N(\beta - q) \qquad \text{for every } q < \beta. \tag{$\dagger$}$$

Assume toward a contradiction that $\beta$ is **not** Martin-Löf random, and fix a Martin-Löf test $L = (L\_1, L\_2, \dots)$ that covers $\beta$, so $\beta \in L\_i$ for every $i$. We construct a Martin-Löf test $M = (M\_{N+1}, M\_{N+2}, \dots)$ that covers $\alpha$, contradicting its randomness.

**Construction.** For every basic interval $[l, r] := [[\sigma]]$ enumerated into $L\_i$ with $i \ge N+1$, try to compute $g(l)$. If $g(l)\downarrow$, enumerate the interval

$$[\,g(l),\; g(l) + 2^N(r - l)\,]$$

into $M\_i$ (computably realized as a single dyadic node $[[\tau]]$ or a finite disjoint union $[[\tau\_1]] \cup [[\tau\_2]] \cup \cdots$).

**$M$ is a Martin-Löf test.** Each $M\_i$ is uniformly c.e. open, and for every $i$

$$\lambda(M_i) = \sum_{I \in M_i} \lambda(I) \;\le\; \sum_{\substack{[l,r]\in L_i \\ g(l)\downarrow}} \lambda\big([\,g(l),\, g(l) + 2^N(r-l)\,]\big) \;\le\; \sum_{[l,r]\in L_i} 2^N \lambda([l,r]) \;=\; 2^N \lambda(L_i) \;\le\; 2^{N-i}.$$

Thus the $i$-th layer has measure at most $2^{N-i} = 2^{-(i-N)}$, so $M$ (indexed from $i = N+1$) is a legitimate Martin-Löf test.

**$M$ covers $\alpha$.** Since $\beta \in L\_i$ for all $i$, fix for each $i \ge N+1$ an interval $[l, r] \in L\_i$ with $\beta \in [l, r]$. As $\beta$ is irrational, $l < \beta < r$; taking $q = l < \beta$ in $(\dagger)$ gives $g(l)\downarrow$ and

$$g(l) < \alpha < g(l) + 2^N(\beta - l) < g(l) + 2^N(r - l).$$

Hence the interval $[\,g(l),\, g(l) + 2^N(r-l)\,]$ enumerated into $M\_i$ contains $\alpha$. So $\alpha \in M\_i$ for every $i \ge N+1$, i.e. $M$ covers $\alpha$.

Then $\alpha$ is not Martin-Löf random — a contradiction. Therefore $\beta$ is Martin-Löf random. $\square$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Properties of the Solovay reducibility on $\mathbb{R}$ (1))</span></p>

Recall the definition of Solovay reducibility on $\mathbb{R}$: a real $\alpha$ is Solovay reducible to a real $\beta$, written $\alpha \leq_S \beta$, if there exists a constant $c$ and a partially computable function

$$g : \subseteq \mathbb{Q} \to \mathbb{Q}$$

such that

$$0 < \alpha - g(q) \downarrow < c(\beta - q)$$

for every rational $q < \beta$.

---

Show that $\leq_S$ is a reflexive and transitive relation of $\mathbb{R}$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

**Key idea.** Solovay reducibility is just a computable way of converting lower bounds for $\beta$ into comparably good lower bounds for $\alpha$.

For reflexivity, take

$$g(q)=q,\qquad c=2.$$

If $q<\alpha$, then

$$0<\alpha-g(q)=\alpha-q<2(\alpha-q),$$

so $\alpha\le_S\alpha$.

For transitivity, suppose

$$\alpha\le_S\beta \quad\text{via }f,c,$$

and

$$\beta\le_S\gamma \quad\text{via }h,d.$$

For every rational $q<\gamma$,

$$h(q)<\beta \quad\text{and}\quad \beta-h(q)<d(\gamma-q).$$

Therefore $f(h(q))\downarrow$, $f(h(q))<\alpha$, and

$$\alpha-f(h(q)) <c(\beta-h(q)) <cd(\gamma-q).$$

Thus $\alpha\le_S\gamma$, witnessed by $f\circ h$ and $cd$.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(What is Solovay reducibility roughly about)</span></p>

$$\boxed{\text{Solovay reducibility is just a computable way of converting lower bounds for $\beta$ into comparably good lower bounds for $\alpha$.}}$$

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Properties of the Solovay reducibility on $\mathbb{R}$ (2))</span></p>

Show that the left-c.e. reals are closed downward relative to $\leq_S$, i.e., if $\beta$ is a left-c.e. real and $\alpha \leq_S \beta$, then $\alpha$ is left-c.e. as well.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Assume $\beta$ is left-c.e. and $\alpha\le_S\beta$, witnessed by $g,c$.

Since $\beta$ is left-c.e., its left cut

$$\lbrace q\in\mathbb Q:q<\beta\rbrace$$

is c.e. Enumerate all such $q$, run $g(q)$, and enumerate the outputs $g(q)$.

For every $q<\beta$,

$$g(q)<\alpha,$$

so all enumerated values are rational lower bounds for $\alpha$. Moreover, if $q\nearrow\beta$, then

$$0<\alpha-g(q)<c(\beta-q)\to 0.$$

Hence the enumerated lower bounds are cofinal below $\alpha$. Taking the running maximum gives a computable nondecreasing rational sequence converging to $\alpha$. Therefore $\alpha$ is left-c.e.

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Properties of the Solovay reducibility on $\mathbb{R}$ (3))</span></p>

Show that $\alpha \leq_S \beta$ implies that

$$K(\alpha \upharpoonright n) \leq K(\beta \upharpoonright n) - O(1).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

We show the usual form

$$K(\alpha\upharpoonright n)\le K(\beta\upharpoonright n)+O(1),$$

which is what the notation $K(\beta\upharpoonright n)-O(1)$ means here.

Let $\alpha\le_S\beta$ via $g,c$. Choose a fixed integer $d$ so large that

$$c2^{1-d}<1.$$

Given $\beta\upharpoonright(n+d)$, compute a rational $q<\beta$ with

$$\beta-q<2^{1-(n+d)}.$$

Run $g(q)$. Then

$$0<\alpha-g(q)<c2^{1-(n+d)}<2^{-n}.$$

So $\alpha$ lies in an interval of length $<2^{-n}$ starting at the rational $g(q)$. This interval intersects only boundedly many dyadic intervals of length $2^{-n}$. Hence, from $\beta\upharpoonright(n+d)$ and one of finitely many extra indices, we can recover $\alpha\upharpoonright n$.

Thus

$$K(\alpha\upharpoonright n) \le K(\beta\upharpoonright(n+d))+O(1).$$

Since $d$ is fixed,

$$K(\beta\upharpoonright(n+d)) \le K(\beta\upharpoonright n)+O(1),$$

by appending the next $d$ bits as constant-size extra information. Hence

$$K(\alpha\upharpoonright n) \le K(\beta\upharpoonright n)+O(1).$$

</details>
</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>We do we actually read more than $n$ bits from prefix? Is it legal?</summary>

You are right that we are reading more than $n$ bits of $\beta$. But the point is that $d$ is a **fixed constant**, depending only on the Solovay constant $c$, not on $n$. So we first get

$$K(\alpha\upharpoonright n) \le K(\beta\upharpoonright(n+d))+O(1).$$

Then, because $d$ is fixed,

$$K(\beta\upharpoonright(n+d)) \le K(\beta\upharpoonright n)+O(1).$$

Indeed, a program for $\beta\upharpoonright n$, plus the next $d$ bits written literally, gives $\beta\upharpoonright(n+d)$. Since $d$ is constant, those extra $d$ bits cost only $O(1)$.

So the proof is not claiming that exactly $n$ bits of $\beta$ determine $n$ bits of $\alpha$. It claims:

$$n+O(1)\text{ bits of }\beta \quad\text{determine}\quad n\text{ bits of }\alpha.$$

In Kolmogorov complexity, a fixed finite loss in precision is invisible because it is absorbed into the $O(1)$ term.

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Switching between Kolmogorov complexity and approximation)</span></p>

The Solovay reduction is not directly a statement about descriptions; it is a statement about **precision transfer**. Namely, if $q<\beta$ is very close to $\beta$, then $g(q)<\alpha$ is comparably close to $\alpha$:

$$\alpha-g(q) < c(\beta-q).$$

Kolmogorov complexity enters because a finite prefix $\beta\upharpoonright n$ is itself a finite description of a rational approximation to $\beta$ with precision about $2^{-n}$. Therefore, a shortest program for $\beta\upharpoonright n$, together with the fixed Solovay reduction procedure, gives a program for $\alpha\upharpoonright n$, up to only constant overhead.

So the inequality

$$K(\alpha\upharpoonright n)\le K(\beta\upharpoonright n)+O(1)$$

should be read as:

$$\text{anything that describes }\beta\text{ to }n\text{ bits also describes }\alpha\text{ to }n\text{ bits,}$$

because the Solovay reduction converts $n$-bit accuracy for $\beta$ into $n$-bit accuracy for $\alpha$, after paying only a fixed constant in precision and code length.

</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Solovay reducibility and computable reals (1))</span></p>

Show that the computable reals are closed downwards in $\mathbb{R}$ relative to $\leq_S$, i.e., if $\alpha \leq_S \beta$ and $\beta$ is computable, then $\alpha$ is computable.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

We use the following characterization of $\beta$ computability:

>there exists a computable Cauchy sequence $(b\_n)\_{n\in\mathbb{N}}$ that converges *effectively*, i.e.
>
> $$\lvert \beta - b_n\rvert < 2^{-n} \qquad \text{for every } n.$$

Let $b_n$ be a computable sequence such that $\lvert \beta - b_n \rvert \le 2^{-n}$. We shift each $b_n$ by $2^{-n}$ to the left, getting $b_n' := b_n - 2^{-n}$. Then $b_n'$ approximates $\beta$ from the left. Then

$$0 \le \alpha - g(b_n') = \lvert \alpha - g(b_n') \rvert \le c(\beta - b_n') \le c2\cdot 2^{-n} = 2^{-n+1+\log_2(c)} \leq 2^{-n+k},$$

where $k:= \left\lceil 1+\log_2(c) \right\rceil$. Then we reindex the sequence.

$$a_n:=g!\left(b'_{n+k}\right).$$

Then

$$|\alpha-a_n| <2^{-(n+k)+k} =2^{-n}.$$

Merely restricting to indices $n\ge k$ without reindexing would still leave the bound $2^{-n+k}$.

Also, $a_n$ is computable: each $b'\_{n+k}<\beta$, so the partial computable function $g$ is guaranteed to halt on every such input.

</details>
</div>


<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Solovay reducibility and computable reals (2))</span></p>

Show that computable reals form the least Solovay degree on the set of left-c.e. reals, i.e., if $\alpha$ is computable and $\beta$ is left-c.e., then $\alpha \leq_S \beta$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

Let $(b_s)$ be a computable increasing rational sequence with $b_s\to\beta$, and let $(a_n)$ be a computable rational approximation to $\alpha$ satisfying

$$|\alpha-a_n|<2^{-n}.$$

For (q<\beta), search for the first $s$ with $b_s>q$. Then choose $n$ such that

$$2^{1-n}<b_s-q,$$

and define

$$g(q):=a_n-2^{-n}.$$

Since $q<\beta$, such an $s$ exists, so $g(q)\downarrow$. Moreover,

$$0<\alpha-g(q) =\alpha-a_n+2^{-n} <2^{1-n} <b_s-q \le \beta-q.$$

Thus $g$ witnesses

$$\alpha\le_S\beta$$

with constant $c=1$. 

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Solovay reducibility and computable reals (3))</span></p>

Show that there exists a right-c.e. real $\beta$ such that

$$\alpha \not\leq_S \beta$$

for any left-c.e. real $\alpha$, including the computable ones.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>

TODO: have no idea

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Strong Kolmogorov reducibility implies Solovay reducibility on left-c.e. reals)</span></p>

Let $\alpha$ and $\beta$ be two left-c.e. reals such that

$$K(\beta \upharpoonright n) - K(\alpha \upharpoonright n) \to_{n \to \infty} \infty.$$

Show that

$$\alpha \leq_S \beta.$$

*Hint: fix two left-c.e. approximations $(a_i)\_{i \in \mathbb{N}}$ and $(b_i)\_{i \in \mathbb{N}}$ of $\alpha$ and $\beta$, respectively. Then, given an optimal prefix-free code $\tau$ of $\alpha \upharpoonright n$, we can compute the first index $s$ such that*

$$a_s \upharpoonright n = \alpha \upharpoonright n.$$

*What do we know about $b_s$?*

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>


</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Kučera–Slaman theorem (1))</span></p>

Recall that the left derivation on a function $f$ in a point $x$ is defined by

$$f^{(l)}(x) := \lim_{y \nearrow x} \frac{g(x)-g(y)}{x-y}.$$

Let $g$ be a monotone nondecreasing piecewise linear function from $[0,\beta]$ to $[0,\alpha]$, where $\alpha,\beta \in (0,1)$.

Show that

$$\lambda\lbrace x \in [0,\beta) : g^{(l)}(x) > 2^k\rbrace < 2^{-k}$$

for every natural number $k$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>


</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Kučera–Slaman theorem (2))</span></p>

Let $(a_n)\_{n \in \mathbb{N}}$ and $(b_n)\_{n \in \mathbb{N}}$, where $a_0 = b_0 = 0$, be two left-c.e. approximations of $\alpha \in (0,1)$ and $\beta \in (0,1)$, respectively.

Show that the family of c.e. open sets

$$L = (L_1, L_2, \dots)$$

defined by

$$x \in L_i \iff \exists m,n \left( b_n < x \text{ and } \frac{a_m - a_n}{x - b_n} > 2^k \right)$$

is a Martin-Löf test.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>


</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Kučera–Slaman theorem (3))</span></p>

Let $\alpha$ be a left-c.e. real and $\beta$ be Martin-Löf random left-c.e. real. Show that

$$\alpha \leq_S \beta.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>


</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(rK-reducibility (1))</span></p>

A real $\alpha$ is called **rK-reducible** to a real $\beta$, written

$$\alpha \leq_{rK} \beta,$$

if there are a constant $k$ and a computable function

$$f : \subseteq \lbrace 0,1\rbrace^\ast \times \mathbb{N} \to \lbrace 0,1\rbrace^\ast$$

such that

$$(\forall n \in \mathbb{N})(\exists j \leq k) \big(f(\beta \upharpoonright n, j) \downarrow = \alpha \upharpoonright n\big).$$

Show that 

$$\alpha \leq_S \beta \implies \alpha \leq_{rK} \beta.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>


</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(rK-reducibility (2))</span></p>

Show that $\alpha \leq_{rK} \beta$ implies both $\alpha \leq_K \beta$ and $\alpha \leq_T \beta$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>


</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Totality and translation of Schnorr tests)</span></p>

Recall that a Schnorr test

$$L = (L_1, L_2, \dots)$$

is a Martin-Löf test that satisfies

$$\lambda(L_i) = 2^{-i}$$

for every $i$

A real $\alpha$ is called **Schnorr nonrandom** if there exists a Schnorr test

$$L = (L_1, L_2, \dots)$$

such that

$$\alpha \in L_i$$

for all $i$, and **Schnorr random** otherwise.

Show that, if $\alpha$ is a Schnorr random real that satisfies

$$\alpha \leq_S \beta$$

via a total function $f$, then $\beta$ is Schnorr random as well.

*Hint: by contradiction. Use the totality of $f$ to transform a Schnorr test that fails on $\beta$ into a Schnorr test that fails on $\alpha$.*

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>


</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Two ways to obtain the same complexity asymptote (1))</span></p>

Recall that, for a real $r \in (0,1)$, the watered-down halting probability $\Omega^r$ is defined by

$$\Omega^r := \sum_{\sigma \in \operatorname{dom}\widetilde U} 2^{-\ell(\sigma)/r}.$$

For two reals

$$\alpha = 0.a_0a_1\dots$$

and

$$\beta = 0.b_0b_1\dots,$$

we define their join $\alpha \oplus \beta$ by

$$\alpha \oplus \beta := 0.a_0b_0a_1b_1\dots$$

Show that $\Omega \oplus 0$ is a left-c.e. real.  
Here, the real $0$ is supposed to be binary represented as

$$0.000\dots$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>


</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Two ways to obtain the same complexity asymptote (2))</span></p>

Show that

$$\frac{K((\Omega \oplus 0) \upharpoonright n)}{n} \xrightarrow[n \to \infty]{} \frac{1}{2}$$

and

$$\frac{K(\Omega^{0.5} \upharpoonright n)}{n} \xrightarrow[n \to \infty]{} \frac{1}{2}.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>


</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(Two ways to obtain the same complexity asymptote (3))</span></p>

Show that

$$\Omega^{0.5} \leq_S \Omega \oplus 0.$$

*Hint: use the characterization (iii) of $\leq_S$ on left-c.e. reals from Sheet 5, Exercise 3.*

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>


</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(A universal Solovay test)</span></p>

Recall that a **Solovay test**

$$S = (I_0, I_1, \dots)$$

is a computable sequence of intervals with rational endpoints such that

$$\sum_{i \in \mathbb{N}} \lambda(I_i) < \infty.$$

A sequence $A$ fails the Solovay test $S$ if the real $0.A$ is contained in infinitely many $I_i$.

Show that there exists a Solovay test $S$ such that all Martin-Löf nonrandom sequences fail $S$.

*Hint: transform a universal Martin-Löf test into a Solovay test.*


</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>


</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(No universal Schnorr test: a formal proof (1))</span></p>

Recall that a **Schnorr test**

$$L = (L_1, L_2, \dots)$$

is a Martin-Löf test that satisfies

$$\lambda(L_i) = 2^{-i}$$

for every $i$.

An infinite binary sequence $\alpha$ is called **Schnorr nonrandom** if there exists a Schnorr test

$$L = (L_1, L_2, \dots)$$

such that $\alpha$ fails $L$, i.e.

$$\alpha \in L_i$$

for all $i$, and **Schnorr random** otherwise.

A Schnorr test $L$ is called **universal** if every Schnorr nonrandom sequence $A$ fails $L$.

Show that, if $U$ is a c.e. open set on $\lbrace 0,1\rbrace^{\mathbb{N}}$ such that $A \in U$, then there exists $n$ such that

$$A \upharpoonright n \subseteq U.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>


</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(No universal Schnorr test: a formal proof (2))</span></p>

Let $U$ be a c.e. open set such that

$$\lambda(U) = c,$$

where $c$ is a computable real.

Show that there exists a Turing machine $M$ such that, for every $\varepsilon > 0$, computes a bit $b \in \lbrace 0,1\rbrace$ such that

$$\lambda(U \cap [[b]]) \leq (1+\varepsilon)\frac{c}{2}.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>


</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(No universal Schnorr test: a formal proof (3))</span></p>

Let $U$ be a c.e. open set such that

$$\lambda(U) = \frac{1}{2}.$$

Compute an infinite sequence $A$ such that

$$\lambda(A \upharpoonright n) \leq (1+3^{-2^0})(1+3^{-2^1}) \dots (1+3^{-2^n}) \cdot \frac{1}{2}$$

for all $n$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>


</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(No universal Schnorr test: a formal proof (4))</span></p>

Let

$$L = (L_1, L_2, \dots)$$

be a Schnorr test. Show that there exists a computable infinite sequence that fails $L$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>


</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(No universal Schnorr test: a formal proof (5))</span></p>

Show that there exists no universal Schnorr test.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>


</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(The $S2a$ reducibility (1))</span></p>

For two c.a. reals $\alpha$ and $\beta$, we say that $\alpha$ is **$S2a$-reducible** to a real $\beta$, written

$$\alpha \leq^{2a}_S \beta,$$

if there exist a constant $c$ and two computable approximations

$$(a_n)_{n \in \mathbb{N}} \qquad\text{and}\qquad (b_n)_{n \in \mathbb{N}}$$

such that

$$|\alpha - a_n| < c\left(|\beta - b_n| + 2^{-n}\right)$$

for all $n$.

Show that $\leq^{2a}\_S$ is transitive.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>


</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(The $S2a$ reducibility (2))</span></p>

Show that, on the set of left-c.e. reals, $\leq^{2a}\_S$ is equivalent to $\leq_S$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>


</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(The $S2a$ reducibility (3))</span></p>

Show that $\leq^{2a}\_S$ implies $\leq_K$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>


</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(D.c.e. reals (1))</span></p>

A real $\gamma$ is called **d.c.e.** — the letter “d” is for “difference”, the rest is a secret... — if there exist two left-c.e. reals $\alpha$ and $\beta$ such that

$$\gamma = \alpha - \beta.$$

Recall that a real $\alpha$ is **right-c.e.** if $-\alpha$ is left-c.e.

Show that a real $\gamma$ is d.c.e. iff there exists a computable approximation

$$(q_n)_{n \in \mathbb{N}}$$

of $\gamma$ such that

$$\sum_{i \in \mathbb{N}} |q_{i+1} - q_i| < \infty.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>


</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(D.c.e. reals (2))</span></p>


Show that, if a d.c.e. real $\gamma$ is neither left-c.e. not right-c.e., then it is Martin-Löf nonrandom.

*Hint: construct a Solovay test that $\gamma$ fails.*

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>


</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(D.c.e. reals (3))</span></p>

Show that the set of d.c.e. reals is closed downwards relative to $\leq^{2a}\_S$ in the set of computably approximable reals.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Solution</summary>


</details>
</div>

We now return to the main line of the section: the structure that Solovay reducibility imposes on the class of left-c.e. reals.

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Randomness-test translation)</span></p>

The proof of Proposition 7.3 is a *randomness-test translation*: a Martin-Löf test that $\beta$ fails is rewritten, interval by interval, through the Solovay reduction into a Martin-Löf test that $\alpha$ fails, with the Solovay constant $c = 2^N$ paying for the blow-up in measure. This translation mechanism is the reason why some authors (see e.g. Downey and Hirschfeldt, *Algorithmic Randomness and Complexity*, Springer, 2010) regard Solovay reducibility and its variants as a standard tool for measuring the *relative randomness* of reals.

</div>

Solovay reducibility is a reflexive and transitive relation (Sheet 6, Exercise 1(i) — carried out in the exercise *Properties of the Solovay reducibility on $\mathbb{R}$ (1)* above), so it induces a degree structure: call two reals *Solovay equivalent* if each is Solovay reducible to the other, and call the resulting equivalence classes **Solovay degrees**, partially ordered by $\le\_S$. Moreover, the set of left-c.e. reals is closed downwards relative to $\le\_S$ (Sheet 6, Exercise 1(ii) — the exercise *Properties of the Solovay reducibility on $\mathbb{R}$ (2)* above), so the $\le\_S$-degree structure on the left-c.e. reals is embedded in the $\le\_S$-degree structure on $\mathbb{R}$.

In the remainder of the section we investigate the $\le\_S$-degree structure of the left-c.e. reals. The first observation is that Chaitin's $\Omega$ sits on top of it.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(7.4 — $\Omega$ is Solovay-above every left-c.e. real)</span></p>

Every left-c.e. real is Solovay reducible to Chaitin's $\Omega$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Let $\alpha$ be a left-c.e. real. By Proposition 5.9(iii), there exists a prefix-free machine $\widetilde M$ whose halting probability is $\alpha$:

$$\alpha = \sum_{\sigma \in \operatorname{dom}(\widetilde M)} 2^{-l(\sigma)}.$$

Since $\widetilde M$ is prefix-free, the universal prefix-free machine simulates it in an *additively optimal* way: by the construction of Theorem 2.20, there is a self-delimiting machine code $i$ such that $\widetilde U(i\sigma) = \widetilde M(\sigma)$ for every word $\sigma$, and the input $i\sigma$ has length $e + l(\sigma)$, where $e := l(i)$ is a constant depending only on $\widetilde M$.

**Two synchronized approximations.** Write $\widetilde U[s]$ for the finite part of $\operatorname{dom}(\widetilde U)$ enumerated after $s$ computation steps, and define, for all $s$,

$$a_s := \sum_{\sigma \,:\, i\sigma \in \widetilde U[s]} 2^{-l(\sigma)}, \qquad\qquad w_s := \sum_{\tau \in \widetilde U[s]} 2^{-l(\tau)}.$$

Both sequences are computable and nondecreasing; $(a\_s)\_{s \in \mathbb{N}}$ is a left-c.e. approximation of $\alpha$, and $(w\_s)\_{s \in \mathbb{N}}$ is the canonical left-c.e. approximation of $\Omega$.

**Comparing the tails.** For every stage $s$,

$$\alpha - a_s \;=\; \sum_{\substack{\sigma \,:\, i\sigma \in \operatorname{dom}(\widetilde U) \setminus \widetilde U[s]}} 2^{-l(\sigma)} \;=\; 2^{e} \sum_{\substack{\sigma \,:\, i\sigma \in \operatorname{dom}(\widetilde U) \setminus \widetilde U[s]}} 2^{-l(i\sigma)} \;\le\; 2^{e} \sum_{\tau \in \operatorname{dom}(\widetilde U) \setminus \widetilde U[s]} 2^{-l(\tau)} \;=\; 2^{e}\,(\Omega - w_s),$$

where the inequality holds because every word $i\sigma$ counted in the middle sum is one of the words $\tau$ counted on the right. Since $\operatorname{dom}(\widetilde U)$ is infinite, we have $\Omega - w\_s > 0$ for every $s$, hence

$$\alpha - a_s < 2^{e+1}(\Omega - w_s) \qquad \text{for all } s.$$

By the speed-of-convergence characterization (iv) of Proposition 7.2, this synchronized pair of approximations witnesses $\alpha \le\_S \Omega$ with the constant $2^{e+1}$. $\square$

</details>
</div>

<figure class="math-figure">
  <svg viewBox="0 0 700 330" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:700px" aria-label="The halting-probability weights of alpha embedded into Omega, with the tails compared">
    <g font-family="serif" font-size="12" fill="#1f2430">
      <text x="350" y="24" text-anchor="middle" font-weight="600">The halting probability α, embedded program by program into Ω</text>

      <!-- legend -->
      <rect x="140" y="38" width="12" height="12" fill="#1d4ed8" />
      <text x="158" y="48" font-size="11" fill="#5b6270">programs iσ — copies of M̃'s programs</text>
      <rect x="410" y="38" width="12" height="12" fill="#5b6270" />
      <text x="428" y="48" font-size="11" fill="#5b6270">other Ũ-programs</text>

      <!-- Omega bar -->
      <text x="76" y="104" text-anchor="end" font-size="14" font-weight="600" fill="#d65336">Ω</text>
      <rect x="90" y="86" width="48" height="28" fill="#1d4ed8" />
      <rect x="138" y="86" width="70" height="28" fill="#5b6270" />
      <rect x="208" y="86" width="30" height="28" fill="#1d4ed8" />
      <rect x="238" y="86" width="55" height="28" fill="#5b6270" />
      <rect x="293" y="86" width="26" height="28" fill="#edf4ff" stroke="#1d4ed8" stroke-width="1" />
      <rect x="319" y="86" width="88" height="28" fill="#eef0f4" stroke="#cbd2e0" stroke-width="1" />
      <rect x="407" y="86" width="14" height="28" fill="#edf4ff" stroke="#1d4ed8" stroke-width="1" />
      <rect x="421" y="86" width="219" height="28" fill="#eef0f4" stroke="#cbd2e0" stroke-width="1" />
      <line x1="293" y1="78" x2="293" y2="122" stroke="#444" stroke-dasharray="4 3" />
      <text x="191" y="78" text-anchor="middle" font-size="11" fill="#5b6270">wₛ (enumerated by stage s)</text>
      <line x1="293" y1="118" x2="640" y2="118" stroke="#a86f00" stroke-width="4" opacity="0.35" />
      <text x="466" y="134" text-anchor="middle" font-size="11" fill="#a86f00">Ω − wₛ</text>

      <!-- connectors between corresponding blue segments -->
      <g stroke="#cbd2e0" stroke-width="1.1">
        <line x1="114" y1="114" x2="137" y2="210" />
        <line x1="223" y1="114" x2="216" y2="210" />
        <line x1="306" y1="114" x2="273" y2="210" />
        <line x1="414" y1="114" x2="313" y2="210" />
      </g>
      <text x="173" y="172" text-anchor="middle" font-size="11" fill="#a86f00">weight ×2ᵉ</text>

      <!-- alpha bar -->
      <text x="76" y="228" text-anchor="end" font-size="14" font-weight="600" fill="#d65336">α</text>
      <rect x="90" y="210" width="94" height="28" fill="#1d4ed8" />
      <rect x="188" y="210" width="56" height="28" fill="#1d4ed8" />
      <rect x="248" y="210" width="50" height="28" fill="#edf4ff" stroke="#1d4ed8" stroke-width="1" />
      <rect x="300" y="210" width="26" height="28" fill="#edf4ff" stroke="#1d4ed8" stroke-width="1" />
      <line x1="246" y1="202" x2="246" y2="246" stroke="#444" stroke-dasharray="4 3" />
      <text x="345" y="228" font-size="11" fill="#5b6270">α = 2ᵉ · (total blue weight in Ω)</text>
      <line x1="248" y1="242" x2="326" y2="242" stroke="#a86f00" stroke-width="4" opacity="0.35" />
      <text x="167" y="258" text-anchor="middle" font-size="11" fill="#5b6270">aₛ (enumerated by stage s)</text>
      <text x="287" y="258" text-anchor="middle" font-size="11" fill="#a86f00">α − aₛ</text>

      <text x="350" y="300" text-anchor="middle" font-size="12.5" font-weight="600" fill="#a86f00">α − aₛ = 2ᵉ · (blue tail of Ω) ≤ 2ᵉ (Ω − wₛ)</text>
    </g>
  </svg>
  <figcaption>Proposition 7.4 as a weight transfer. Every program $\sigma$ of the prefix-free machine $\widetilde M$ with $\alpha = \sum_\sigma 2^{-l(\sigma)}$ reappears inside $\operatorname{dom}(\widetilde U)$ as $i\sigma$, carrying $2^{-e}$ times its weight (blue); the universal machine also has programs of its own (gray). The $\alpha$-weight still missing at stage $s$ consists exactly of the blue $\Omega$-weight not yet enumerated, so $\alpha - a_s \le 2^e(\Omega - w_s)$: the tail of $\alpha$ is dominated by a constant multiple of the tail of $\Omega$, which is precisely a Solovay reduction.</figcaption>
</figure>

For a reflexive and transitive relation $\le$ on a set $X$, the **greatest $\le$-degree** on $X$, if it exists, consists of all elements $x \in X$ such that $y \le x$ for every $y \in X$. We now show that the greatest $\le\_S$-degree on the set of left-c.e. reals exists and contains exactly the Martin-Löf random left-c.e. reals.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(7.5 — Random left-c.e. reals are Solovay-complete)</span></p>

For every two left-c.e. reals $\alpha$ and $\beta$ where $\beta$ is Martin-Löf random, it holds that $\alpha \le\_S \beta$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

Sheet 6, Exercise 4 — the argument is developed step by step in the exercises *Kučera–Slaman theorem (1)–(3)* above.

</details>
</div>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(7.6 — Kučera–Slaman theorem, 2001)</span></p>

On the set of left-c.e. reals, the Martin-Löf random left-c.e. reals form the greatest Solovay degree.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

A Martin-Löf random left-c.e. real exists: Chaitin's $\Omega$ is left-c.e. by Proposition 5.9 and Martin-Löf random by Theorem 6.2.

By Proposition 7.5, every left-c.e. real is Solovay reducible to every Martin-Löf random left-c.e. real. Hence all Martin-Löf random left-c.e. reals lie in the greatest $\le\_S$-degree of the left-c.e. reals — which, in particular, therefore exists.

Conversely, let $\gamma$ be a left-c.e. real lying in the greatest $\le\_S$-degree. Then in particular $\Omega \le\_S \gamma$, and since $\Omega$ is Martin-Löf random, Proposition 7.3 shows that $\gamma$ is Martin-Löf random as well.

Thus the greatest Solovay degree on the left-c.e. reals consists exactly of the Martin-Löf random left-c.e. reals. $\square$

</details>
</div>

<div class="math-callout math-callout--remark" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Remark</span><span class="math-callout__name">(Randomness as an order-theoretic property)</span></p>

On the left-c.e. reals, the Kučera–Slaman theorem turns Martin-Löf randomness into a purely order-theoretic property: a left-c.e. real is Martin-Löf random iff it lies $\le\_S$-above all left-c.e. reals, i.e. iff its lower cut is as hard to approximate as a left-c.e. lower cut can possibly be. In particular, all Martin-Löf random left-c.e. reals are Solovay equivalent to $\Omega$: from the point of view of $\le\_S$, there is only one "$\Omega$-like" degree.

</div>

<figure class="math-figure">
  <svg viewBox="0 0 700 400" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:700px" aria-label="The Solovay degree structure of the left-c.e. reals with computable reals at the bottom and random reals at the top">
    <g font-family="serif" font-size="12" fill="#1f2430">
      <text x="350" y="24" text-anchor="middle" font-weight="600">Solovay degrees of the left-c.e. reals</text>
      <text x="350" y="42" text-anchor="middle" font-size="11" fill="#5b6270">every left-c.e. real is ≤ₛ Ω (Prop 7.4); random left-c.e. reals are ≤ₛ-above all of them (Prop 7.5)</text>

      <!-- greatest degree -->
      <rect x="200" y="56" width="300" height="66" rx="8" fill="#fef2f2" stroke="#c62828" stroke-width="1.3" />
      <text x="350" y="78" text-anchor="middle" font-weight="600" fill="#c62828">greatest Solovay degree (Thm 7.6)</text>
      <text x="350" y="96" text-anchor="middle" font-size="11.5">Martin-Löf random left-c.e. reals</text>
      <text x="350" y="112" text-anchor="middle" font-size="11" fill="#5b6270">Ω and everything Solovay-equivalent to it</text>

      <!-- intermediate degrees -->
      <rect x="235" y="176" width="230" height="54" rx="8" fill="#edf4ff" stroke="#1d4ed8" stroke-width="1.2" stroke-dasharray="5 4" />
      <text x="350" y="197" text-anchor="middle" font-weight="600" fill="#1d4ed8">intermediate degrees</text>
      <text x="350" y="214" text-anchor="middle" font-size="11" fill="#5b6270">noncomputable, nonrandom</text>

      <!-- least degree -->
      <rect x="200" y="286" width="300" height="56" rx="8" fill="#ecfdf5" stroke="#3d7a26" stroke-width="1.3" />
      <text x="350" y="308" text-anchor="middle" font-weight="600" fill="#3d7a26">least Solovay degree</text>
      <text x="350" y="326" text-anchor="middle" font-size="11.5">computable reals</text>

      <!-- order arrows -->
      <line x1="350" y1="286" x2="350" y2="236" stroke="#444" stroke-width="1.4" marker-end="url(#arrow-degs)" />
      <line x1="350" y1="176" x2="350" y2="128" stroke="#444" stroke-width="1.4" marker-end="url(#arrow-degs)" />
      <text x="362" y="262" font-size="12" fill="#444">≤ₛ</text>
      <text x="362" y="154" font-size="12" fill="#444">≤ₛ</text>

      <!-- side annotations -->
      <line x1="560" y1="330" x2="560" y2="92" stroke="#a86f00" stroke-width="1.4" marker-end="url(#arrow-degs-amber)" />
      <text x="578" y="190" font-size="11" fill="#a86f00">randomness</text>
      <text x="578" y="206" font-size="11" fill="#a86f00">travels upward</text>
      <text x="578" y="222" font-size="11" fill="#a86f00">(Prop 7.3)</text>

      <line x1="140" y1="92" x2="140" y2="330" stroke="#5b6270" stroke-width="1.4" marker-end="url(#arrow-degs-gray)" />
      <text x="122" y="190" text-anchor="end" font-size="11" fill="#5b6270">left-c.e. reals are</text>
      <text x="122" y="206" text-anchor="end" font-size="11" fill="#5b6270">closed downwards</text>
      <text x="122" y="222" text-anchor="end" font-size="11" fill="#5b6270">(Sheet 6, Ex. 1)</text>

      <defs>
        <marker id="arrow-degs" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
          <path d="M0,0 L10,4 L0,8 Z" fill="#444" />
        </marker>
        <marker id="arrow-degs-amber" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
          <path d="M0,0 L10,4 L0,8 Z" fill="#a86f00" />
        </marker>
        <marker id="arrow-degs-gray" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
          <path d="M0,0 L10,4 L0,8 Z" fill="#5b6270" />
        </marker>
      </defs>
    </g>
  </svg>
  <figcaption>The Solovay degrees of the left-c.e. reals. The computable reals form the least degree, and by the Kučera–Slaman theorem the Martin-Löf random left-c.e. reals form the greatest one — the degree of $\Omega$. Proposition 7.3 gives the vertical direction its meaning: Martin-Löf randomness can only spread upward along $\le_S$, so the random reals sit exactly at the top of the order.</figcaption>
</figure>

## Relative Randomness and Relative Complexity

This section explains some close connections between Solovay reducibility and the relative complexity of reals. Solovay reducibility compares two reals by how well rational lower bounds for one can be converted into rational lower bounds for the other. Kolmogorov complexity suggests a different, prefix-by-prefix comparison: $\alpha$ should count as "no more random" than $\beta$ if the initial segments of $\alpha$ are no harder to describe than those of $\beta$. The next two definitions make the second idea precise, and the theorem that follows shows how closely the two comparisons are aligned.

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(8.1 — Kolmogorov reducibility, strong Kolmogorov reducibility)</span></p>

A real $\alpha$ is **Kolmogorov reducible** to a real $\beta$, written $\alpha \le\_K \beta$, if

$$K(\alpha \upharpoonright n) \le K(\beta \upharpoonright n) + O(1).$$

A real $\alpha$ is **strongly Kolmogorov reducible** to a real $\beta$, written $\alpha \ll\_K \beta$, if

$$K(\beta \upharpoonright n) - K(\alpha \upharpoonright n) \xrightarrow[n \to \infty]{} \infty.$$

</div>

<div class="math-callout math-callout--definition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Definition</span><span class="math-callout__name">(8.2 — Strong Solovay reducibility)</span></p>

A real $\alpha$ is **strongly Solovay reducible** to a real $\beta$, written $\alpha \ll\_S \beta$, if $\alpha \le\_S \beta$ via a function $g$ that fulfills

$$\lim_{q \nearrow \beta} \frac{\alpha - g(q)}{\beta - q} = 0.$$

</div>

Both definitions strengthen a bounded comparison into a vanishing one. Ordinary Solovay reducibility requires the gap ratio $(\alpha - g(q))/(\beta - q)$ to stay bounded by the constant $c$ from Definition 7.1; strong Solovay reducibility requires that it vanish as the lower bounds $q$ approach $\beta$. Similarly, $\alpha \ll\_K \beta$ strengthens $\alpha \le\_K \beta$: not only do the prefixes of $\alpha$ stay describable within a constant of those of $\beta$, but the complexity gap between them grows without bound.

<figure class="math-figure">
  <svg viewBox="0 0 700 300" xmlns="http://www.w3.org/2000/svg" width="100%" style="max-width:700px" aria-label="Bounded gap ratio for Solovay reducibility versus vanishing gap ratio for strong Solovay reducibility">
    <g font-family="serif" font-size="12" fill="#1f2430">
      <text x="350" y="24" text-anchor="middle" font-weight="600">The gap ratio (α − g(q)) / (β − q) as q approaches β</text>

      <!-- panel frames -->
      <g fill="none" stroke="#cbd2e0" stroke-width="1.1">
        <rect x="45" y="44" width="285" height="200" rx="6" />
        <rect x="375" y="44" width="285" height="200" rx="6" />
      </g>

      <!-- left panel: bounded ratio -->
      <text x="187" y="62" text-anchor="middle" font-weight="600" font-size="12" fill="#5b6270">α ≤ₛ β: ratio bounded by c</text>
      <text x="76" y="78" font-size="10.5" fill="#5b6270">gap ratio</text>
      <line x1="70" y1="80" x2="70" y2="228" stroke="#444" />
      <line x1="70" y1="228" x2="318" y2="228" stroke="#444" marker-end="url(#arrow-ratio)" />
      <line x1="305" y1="223" x2="305" y2="233" stroke="#d65336" stroke-width="1.5" />
      <text x="305" y="248" text-anchor="middle" font-size="11" fill="#d65336">β</text>
      <text x="200" y="246" text-anchor="middle" font-size="11" fill="#5b6270">q ↗</text>
      <line x1="70" y1="110" x2="318" y2="110" stroke="#a86f00" stroke-dasharray="5 4" />
      <text x="62" y="114" text-anchor="end" font-size="11" fill="#a86f00">c</text>
      <path d="M72 200 C95 150 115 215 140 180 C160 152 175 212 195 190 C215 165 235 205 255 145 C270 128 285 200 302 170" fill="none" stroke="#1d4ed8" stroke-width="2" />

      <!-- right panel: vanishing ratio -->
      <text x="517" y="62" text-anchor="middle" font-weight="600" font-size="12" fill="#5b6270">α ≪ₛ β: ratio vanishes</text>
      <text x="406" y="78" font-size="10.5" fill="#5b6270">gap ratio</text>
      <line x1="400" y1="80" x2="400" y2="228" stroke="#444" />
      <line x1="400" y1="228" x2="648" y2="228" stroke="#444" marker-end="url(#arrow-ratio)" />
      <line x1="635" y1="223" x2="635" y2="233" stroke="#d65336" stroke-width="1.5" />
      <text x="635" y="248" text-anchor="middle" font-size="11" fill="#d65336">β</text>
      <text x="530" y="246" text-anchor="middle" font-size="11" fill="#5b6270">q ↗</text>
      <path d="M402 120 C425 85 445 175 470 140 C495 108 510 195 535 185 C560 172 585 215 610 212 C622 210 630 220 632 222" fill="none" stroke="#1d4ed8" stroke-width="2" />
      <text x="612" y="190" text-anchor="middle" font-size="11" fill="#a86f00">→ 0</text>

      <defs>
        <marker id="arrow-ratio" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
          <path d="M0,0 L10,4 L0,8 Z" fill="#444" />
        </marker>
      </defs>
    </g>
  </svg>
  <figcaption>Ordinary versus strong Solovay reducibility. In both cases the computable map $g$ turns rational lower bounds $q < \beta$ into rational lower bounds $g(q) < \alpha$; the difference lies in the behaviour of the gap ratio $(\alpha - g(q))/(\beta - q)$. For $\alpha \le_S \beta$ (Definition 7.1) the ratio merely stays below a fixed constant $c$, however good the lower bound $q$ becomes; for $\alpha \ll_S \beta$ (Definition 8.2) it must vanish as $q \nearrow \beta$.</figcaption>
</figure>

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(8.3 — Solovay reducibility sits between the complexity reducibilities)</span></p>

For every two reals $\alpha$ and $\beta$, the following implications hold:

$$\alpha \ll_S \beta \;\Longrightarrow\; \alpha \le_S \beta \;\Longrightarrow\; \alpha \le_K \beta.$$

On the set of left-c.e. reals, the following implication also holds:

$$\alpha \ll_K \beta \;\Longrightarrow\; \alpha \ll_S \beta.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

**$\ll\_S$ implies $\le\_S$.** Immediate from Definition 8.2: a function $g$ witnessing $\alpha \ll\_S \beta$ witnesses $\alpha \le\_S \beta$ in particular.

**$\le\_S$ implies $\le\_K$.** First we dispose of a degenerate case. If $\beta$ is dyadic, then $\beta$ is computable, hence so is $\alpha$ (see the exercise *Solovay reducibility and computable reals (1)* above); then $K(\mathrm{bin}(n)) \le K(\beta \upharpoonright n) + O(1)$ (the length $n$ is computable from the word $\beta \upharpoonright n$) and $K(\alpha \upharpoonright n) \le K(\mathrm{bin}(n)) + O(1)$ (the prefix of a computable real is computable from its length), so $\alpha \le\_K \beta$ holds outright. Hence assume that $\beta$ is not dyadic, so that $\beta \upharpoonright n < \beta$, i.e. $\beta \upharpoonright n \in L(\beta)$, for every $n$.

Fix a natural number $N$ and a function $g$ on the rationals such that $\alpha \le\_S \beta$ via $g$ with the constant $c := 2^N$:

$$0 < \alpha - g(q) < 2^N(\beta - q) \qquad \text{for every } q < \beta.$$

We define a Turing machine $\widetilde M$ that, given an input of the form $b\tau$ with a single advice bit $b \in \lbrace 0,1 \rbrace$, computes $w := \widetilde U(\tau)$ and $n := l(w)$, evaluates $g(w)$ (identifying the word $w$ with the dyadic rational $0.w$, as usual), and returns the word of length $n - N$ that represents the rational

$$\big(\lfloor g(w) \cdot 2^{\,n-N} \rfloor + b\big) \cdot 2^{-(n-N)},$$

that is, the truncation of $g(w)$ to $n - N$ binary places, shifted up by the advice bit. The machine $\widetilde M$ is prefix-free: if two inputs $b\tau$ and $b'\tau'$ in its domain were comparable, then $\tau$ and $\tau'$ would be comparable words in $\operatorname{dom}(\widetilde U)$, hence equal.

Now fix $n$ and let $\tau$ be an optimal code of $\beta \upharpoonright n$, i.e. $\widetilde U(\tau) = \beta \upharpoonright n$ and $l(\tau) = K(\beta \upharpoonright n)$. Since $\beta \upharpoonright n \in L(\beta)$, the value $g(\beta \upharpoonright n)$ is defined, and taking $q = \beta \upharpoonright n$ in the displayed inequality, together with $\beta - \beta \upharpoonright n \le 2^{-n}$, gives

$$0 < \alpha - g(\beta \upharpoonright n) < 2^N(\beta - \beta \upharpoonright n) \le 2^N 2^{-n} = 2^{-(n-N)}.$$

Writing $m := n - N$ and $k := \lfloor g(\beta \upharpoonright n) \cdot 2^m \rfloor$, the real $\alpha$ therefore lies in the interval $(k 2^{-m}, (k+2) 2^{-m})$, so its prefix satisfies

$$\alpha \upharpoonright m \in \lbrace k \cdot 2^{-m},\; (k+1) \cdot 2^{-m} \rbrace,$$

and one advice bit $b$ decides which of the two candidates is correct. Hence $\widetilde M(b\tau) = \alpha \upharpoonright (n-N)$ for the right choice of $b$. Since $\widetilde M$ is prefix-free, $\widetilde U$ simulates it with constant overhead, so there exists a constant $C\_1$ such that

$$K(\alpha \upharpoonright (n-N)) \le C_{\widetilde M}(\alpha \upharpoonright (n-N)) + c_{\widetilde M} \le \big(l(\tau) + 1\big) + c_{\widetilde M} = K(\beta \upharpoonright n) + C_1 \qquad \text{for all } n, \tag{30}$$

where $c\_{\widetilde M}$ is the simulation constant of $\widetilde M$ and $C\_1 := c\_{\widetilde M} + 1$.

Finally, $\alpha \upharpoonright n$ can be recovered from $\alpha \upharpoonright (n-N)$ together with the $N$ missing bits: a prefix-free machine first reads $N$ designated bits and then runs $\widetilde U$ on the rest — this is the prefix-free analogue of the last-bit stability from Proposition 2.12, iterated $N$ times. Since $N$ is a fixed constant, there exists a constant $C\_2$ such that

$$K(\alpha \upharpoonright n) \le K(\alpha \upharpoonright (n-N)) + C_2 \qquad \text{for all } n. \tag{31}$$

Together, (30) and (31) give $K(\alpha \upharpoonright n) \le K(\beta \upharpoonright n) + C\_1 + C\_2$ for all $n$, i.e. $\alpha \le\_K \beta$.

**$\ll\_K$ implies $\ll\_S$ on the left-c.e. reals.** Let $\alpha$ and $\beta$ be left-c.e. reals that fulfill $K(\beta \upharpoonright n) - K(\alpha \upharpoonright n) \to\_{n \to \infty} \infty$, and fix two left-c.e. approximations $(a\_n)\_{n \in \mathbb{N}}$ and $(b\_n)\_{n \in \mathbb{N}}$ of $\alpha$ and $\beta$, respectively. Replacing $a\_s$ by $a\_s - 2^{-s}$ and $b\_s$ by $b\_s - 2^{-s}$ if necessary, we may assume that both approximations are strictly increasing, so in particular $a\_s < \alpha$ and $b\_s < \beta$ for every $s$. To avoid trivial representation issues we also assume that $\alpha$ is not dyadic; for dyadic $\alpha$ the same argument runs with the non-terminating binary expansion of $\alpha$, whose prefixes have the same complexity as those of the terminating one up to an additive constant.

We define a Turing machine $\widetilde M$ that, given a word $\sigma$, computes $n := l(\widetilde U(\sigma))$, then searches for the first index $s\_n$ (if it exists) such that

$$a_{s_n} \upharpoonright n = \widetilde U(\sigma),$$

and returns $b\_{s\_n} \upharpoonright n$. Then $\widetilde M$ is prefix-free since $\operatorname{dom}(\widetilde M) \subseteq \operatorname{dom}(\widetilde U)$, so $\widetilde U$ simulates it with constant overhead: there exists a constant $C$ such that

$$K(w) \le C_{\widetilde M}(w) + C \qquad \text{for every word } w.$$

For each $n$, let $\tau\_n$ be an optimal code of $\alpha \upharpoonright n$; that is, $\widetilde U(\tau\_n) = \alpha \upharpoonright n$ and $l(\tau\_n) = K(\alpha \upharpoonright n)$. Since $\alpha$ is not dyadic, we have $\alpha \upharpoonright n < \alpha < \alpha \upharpoonright n + 2^{-n}$, and since $a\_s \nearrow \alpha$, the index $s\_n$ exists; note also that $s\_n \le s\_{n+1}$ (agreement on $n+1$ bits implies agreement on $n$ bits) and $s\_n \to\_{n \to \infty} \infty$ (no single $a\_s < \alpha$ agrees with $\alpha$ on all prefixes). By construction, $\widetilde M(\tau\_n) \downarrow = b\_{s\_n} \upharpoonright n$, so

$$K(b_{s_n} \upharpoonright n) \le C_{\widetilde M}(b_{s_n} \upharpoonright n) + C \le l(\tau_n) + C = K(\alpha \upharpoonright n) + C.$$

Since $K(\beta \upharpoonright n) - K(\alpha \upharpoonright n) \to\_{n \to \infty} \infty$, the latter inequality implies that

$$K(\beta \upharpoonright n) - K(b_{s_n} \upharpoonright n) \xrightarrow[n \to \infty]{} \infty,$$

which in turn implies that

$$2^n(\beta - b_{s_n}) \xrightarrow[n \to \infty]{} \infty. \tag{32}$$

Indeed, suppose that $2^n(\beta - b\_{s\_n}) \le k$ held for some fixed $k$ and infinitely many $n$. For each such $n$, the words $\beta \upharpoonright n$ and $b\_{s\_n} \upharpoonright n$, read as multiples of $2^{-n}$, differ by an integer $d$ with $0 \le d \le k+1$; a machine that reads a self-delimiting code for $d$ followed by an optimal code for $b\_{s\_n} \upharpoonright n$ computes $\beta \upharpoonright n$, so $K(\beta \upharpoonright n) \le K(b\_{s\_n} \upharpoonright n) + 2 l(\mathrm{bin}(k+1)) + O(1)$ along that infinite set — contradicting the divergence just established.

**Constructing the witness $g$.** It remains to define a function $g$ witnessing $\alpha \ll\_S \beta$: given a rational $q$, the function $g$ searches for the first index $m$ such that $b\_m > q$ (which exists iff $q < \beta$) and returns $g(q) := a\_m$. Then, for every $q < \beta$, we obtain $g(q) \downarrow = a\_m < \alpha$, hence $\alpha - g(q) > 0$.

Now fix $q < \beta$ close enough to $\beta$ that the corresponding first index $m$ with $b\_m > q$ satisfies $m \ge s\_0$ (only the limit $q \nearrow \beta$ matters, so this is no restriction), and (noncomputably) fix the index $n$ such that $s\_n \le m < s\_{n+1}$. Since both approximations are increasing:

- $g(q) = a\_m \ge a\_{s\_n}$, and $a\_{s\_n} \upharpoonright n = \alpha \upharpoonright n$ gives $a\_{s\_n} \ge \alpha \upharpoonright n > \alpha - 2^{-n}$, hence $\alpha - g(q) \le \alpha - a\_{s\_n} \le 2^{-n}$;
- $q < b\_m \le b\_{s\_{n+1}}$, hence $\beta - q > \beta - b\_{s\_{n+1}}$.

Combining the two estimates,

$$\frac{\alpha - g(q)}{\beta - q} \;\le\; \frac{\alpha - a_{s_n}}{\beta - b_{s_{n+1}}} \;\le\; \frac{2^{-n}}{\beta - b_{s_{n+1}}} \;=\; \frac{2}{2^{\,n+1}(\beta - b_{s_{n+1}})} \;\xrightarrow[n \to \infty]{}\; 0,$$

where the convergence holds by (32). As $q \nearrow \beta$, the corresponding index $m$ tends to infinity, hence so does $n$; therefore the gap ratio tends to $0$. Finally, the ratio also stays bounded for $q$ bounded away from $\beta$ (for $q \le \beta - \delta$ we have $(\alpha - g(q))/(\beta - q) \le (\alpha - a\_0)/\delta$), so $g$ witnesses $\alpha \le\_S \beta$ and, with the vanishing limit, $\alpha \ll\_S \beta$. $\square$

</details>
</div>

A weaker version of the last implication — that $\alpha \ll\_K \beta$ implies plain $\alpha \le\_S \beta$ on the left-c.e. reals — was the content of the exercise *Strong Kolmogorov reducibility implies Solovay reducibility on left-c.e. reals* above; the proof just given refines that construction so as to control the gap ratio in the limit. Taken together, Theorem 8.3 yields the following chain on the left-c.e. reals:

$$\alpha \ll_K \beta \;\Longrightarrow\; \alpha \ll_S \beta \;\Longrightarrow\; \alpha \le_S \beta \;\Longrightarrow\; \alpha \le_K \beta.$$

