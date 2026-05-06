---
title: Algorithmic Randomness and Computable Analysis
layout: default
noindex: true
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

## Week 1: Introduction

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

* **Sequence $Y$.** A short description of $Y$ is its index in a fixed computable enumeration of all binary words of length 2000 with at most 200 ones (see Section 1.3 — this enumeration is much smaller than $2^{2000}$).

</div>

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

*Proof.* We enumerate the set $S = (\sigma_0, \sigma_1, \dots)$ in stages: at stage $n$, enumerate into $S$ all words of length $n$ that begin with $w_n$ (recall $(w_0, w_1, w_2, w_3, \dots) = (\lambda, 0, 1, 00, \dots)$ is the length-lexicographic enumeration of all binary words). The first few stages produce

$$
\underbrace{\lambda}_{\text{stage 0}},\quad
\underbrace{0}_{\text{stage 1}},\quad
\underbrace{10, 11}_{\text{stage 2}},\quad
\underbrace{000, 001}_{\text{stage 3}},\quad
\underbrace{0100, 0101, 0110, 0111}_{\text{stage 4}}, \dots
$$

For the machine $M$ that, given $\mathrm{bin}(n)$, returns $\sigma_n$, we have

$$C_M(\sigma_n) \le l(\mathrm{bin}(n)) \le \log(n) + 1, \qquad \text{hence} \qquad C(\sigma_n) \le \log(n) + c_M.$$

It remains to observe that every infinite sequence $A$ contains infinitely many prefixes in the enumeration $S$, and that $l(\sigma_n) - \log(n) \to \infty$ as $n \to \infty$, so the slack $l(\sigma_n) - C(\sigma_n)$ becomes arbitrarily large.

</details>
</div>

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

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Example</span><span class="math-callout__name">(1.6 — Martin-Löf tests covering $\tilde X$ and $\tilde Z$)</span></p>

A Martin-Löf test $\mathcal{M}^{\tilde X}$ covering $\tilde X$ is

$$M_1 = ([\![01]\!]),\quad M_2 = ([\![0101]\!]),\quad M_3 = ([\![010101]\!]),\ \dots$$

(each layer consists of a single basic open set), since $\tilde X \in M_i$ and $\mu(M_i) = 2^{-2i}$ for every $i$.

A Martin-Löf test $\mathcal{M}^{\tilde Z}$ covering $\tilde Z$ is

$$M_1 = ([\![00]\!], [\![10]\!]),$$

$$M_2 = ([\![0000]\!], [\![0100]\!], [\![0010]\!], [\![0110]\!]),$$

$$M_3 = ([\![000000]\!], [\![000100]\!], [\![001000]\!], [\![001100]\!], [\![010000]\!], [\![010100]\!], [\![011000]\!], [\![011100]\!]),\ \dots$$

or, more formally,

$$M_i = \lbrace [\![0 a_1\, 0 a_2 \dots 0 a_i]\!] : a_1 a_2 \dots a_i \in \lbrace 0, 1 \rbrace^{i} \rbrace \qquad \text{for every natural } i.$$

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

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(1.6)</span></p>

Construct a Martin-Löf test $\mathcal{M}$ that covers every sequence $X$ which does *not* contain the word $111$ as a substring.

</div>

## Week 2-3: Kolmogorov complexity

In this section we examine the Kolmogorov complexity as a function on binary words, starting from the trivial upper bound and a few straightforward properties.

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.1 — Trivial upper bound)</span></p>

There exists a constant $c$ such that the Kolmogorov complexity of every binary word $w$ is no greater than $l(w) + c$:

$$\forall w \in \lbrace 0, 1 \rbrace^{\ast}: \quad C(w) \le l(w) + c.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Proof.* The identity Turing machine $M$ on $\lbrace 0, 1 \rbrace^{\ast}$ satisfies $M(w) = w$, hence $C_M(w) = l(w)$. By (1), there exists a constant $c_M$ with $C(w) \le C_M(w) + c_M = l(w) + c_M$ for every $w$.

</details>
</div>

The next theorem expresses that no computable process can blow up Kolmogorov complexity by more than an additive constant — the constant depends on the *process* but not on the *input*.

<div class="math-callout math-callout--theorem" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Theorem</span><span class="math-callout__name">(2.2 — Computable processes do not increase complexity)</span></p>

For every computable function $f :\subseteq \lbrace 0, 1 \rbrace^{\ast} \to \lbrace 0, 1 \rbrace^{\ast}$, there exists a constant $c_f$ such that

$$C(f(w)) < C(w) + c_f \qquad \text{for every } w \in \mathrm{dom}(f).$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Proof.* Let $M_f$ be a Turing machine computing $f$, and consider the machine $M$ that, on input $\sigma$, computes $f(U(\sigma))$. For every $w \in \mathrm{dom}(f)$, let $\sigma_w$ be an optimal encoding of $w$, i.e., $U(\sigma_w) = w$ and $l(\sigma_w) = C(w)$. Then

$$M(\sigma_w) = f(U(\sigma_w)) = f(w),$$

so $C_M(f(w)) \le l(\sigma_w) = C(w)$. By (1), there exists a constant $c_M$ with

$$C(f(w)) \le C_M(f(w)) + c_M \le C(w) + c_M,$$

which concludes the proof.

</details>
</div>

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.3 — Counting bound)</span></p>

For every $n$, there exist at most $2^n - 1$ words of Kolmogorov complexity less than $n$:

$$\forall n: \quad \#\lbrace w \in \lbrace 0, 1 \rbrace^{\ast} : C(w) < n \rbrace < 2^n.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Proof.* For every $w$ with $C(w) = k$, there exists at least one optimal encoding $\sigma$ with $U(\sigma) = w$ and $l(\sigma) = k$. There are exactly $2^n - 1$ words of length less than $n$, and each can be the optimal encoding of at most one word $w$. The bound follows.

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

<div class="math-callout math-callout--proposition" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Proposition</span><span class="math-callout__name">(2.4 — Average complexity is asymptotically the length)</span></p>

Let $g(n)$ be the average Kolmogorov complexity of a word of length $n$. Then

$$\frac{g(n)}{n} \xrightarrow[n \to \infty]{} 1.$$

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Proof.* By Proposition 2.1, there is a constant $c$ such that $C(w) < l(w) + c = n + c$ for every word $w$ of length $n$, giving the upper bound. For the lower bound, distribute the $2^n$ words of length $n$ across complexity strata using Corollary 2.3.1:

$$
\frac{2^{n-1}(n - 1) + 2^{n-2}(n - 2) + \dots + 2^0 (n - n)}{2^n \cdot n}
=
\frac{2^{n-1} \cdot n + 2^{n-2} \cdot n + \dots + 2^0 \cdot n}{2^n \cdot n}
- \frac{2^{n-1} \cdot 1 + 2^{n-2} \cdot 2 + \dots + 2^0 \cdot n}{2^n \cdot n}
$$

$$
\ge 1 - 2^{-n} - \frac{\sum_{i=1}^{\infty} 2^{-i} \cdot i}{n} \xrightarrow[n \to \infty]{} 1.
$$

</details>
</div>

<div class="math-callout math-callout--question" markdown="1">
  <p class="math-callout__title"><span class="math-callout__label">Exercise</span><span class="math-callout__name">(2.5 — Sharper two-sided bound)</span></p>

Show that there exist constants $c_m$ and $c_M$ such that

$$n - c_m \le g(n) \le n + c_M \qquad \text{for all } n.$$

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

$$H_f := \lbrace (w, y) : f(w) < y \rbrace$$

(called **hypograph** of $f$) is computably enumerable.

A function $f$ is **upper-semicomputable** if the set

$$E_f := \lbrace (w, y) : f(w) > y \rbrace$$

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

*Proof.* Apply Proposition 2.9 to

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

* $f$ is unbounded (for every $n$ there exists $w$ with $f(w) \!\downarrow\, > n$);
* $C(w) > f(w)$ for every $w \in \mathrm{dom}(f)$.

</div>

<div class="accordion" markdown="1">
<details markdown="1">
<summary>Proof</summary>

*Proof.* Suppose such $f$ exists. Construct a Turing machine $M$ that, on input $\mathrm{bin}(n)$, runs $f(\lambda), f(0), f(1), f(00), \dots$ in parallel and returns the first word $\sigma$ for which $f(\sigma) \!\downarrow\, > n$. Since $f$ is unbounded, $M$ is computable and total. By the assumed bound on $f$,

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

*Proof.* Let $M$ be the Turing machine that on input $\mathrm{bin}(n)$ returns $(01)^{2^n}$. Set $w_n := (01)^{2^n}$. Then $C_M(w_n) \le l(\mathrm{bin}(n)) \le \log(n)$, hence $C(w_n) \le \log(n) + c_M$.

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
