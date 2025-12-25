---
title: Description Numbers
layout: default
noindex: true
---

# Description Numbers

## Definition

**Description numbers** are integers used in Turing machine theory. They’re essentially the same idea as **Gödel numbers** — and some texts even use that name for them. Once you fix a particular **universal Turing machine** and an encoding scheme, every Turing machine can be represented as a string and then mapped to a natural number. That number is the machine’s **description number**. Description numbers are central to Turing’s original proof that the **Halting Problem** is undecidable, and they’re a convenient tool for treating Turing machines as data when proving results about their behavior.

## Example

Suppose we have a Turing machine $M$ with states $q_1,\dots,q_R$, tape symbols $s_1,\dots,s_m$ (and blank $s_0$), and transition rules that, given a current state and scanned symbol, specify what to write, how to move the head (left/right/stay), and what the next state is. In Turing’s original universal-machine setup, such a machine can be written down as a single string by encoding its components like this:

1. **States.** The state $q_i$ is written as the letter **D** followed by **A** repeated $i$ times (a unary code).
2. **Tape symbols.** The symbol $s_j$ is written as **D** followed by **C** repeated $j$ times (so the blank $s_0$ is just **D**).
3. **Transitions.** Each transition is recorded by concatenating:
   
   $$\text{(current state)(input symbol)(symbol to write)(move direction)(next state)},$$

   where the move direction is one of **L**, **R**, or **N** (left, right, or no move), and states/symbols use the encodings above.

The input to the universal machine is then just a list of these transition-strings separated by semicolons. So the “program alphabet” consists of the seven symbols $\lbrace D,A,C,L,R,N,';'\rbrace$.

**Example.** Consider a tiny machine that writes $1,0,1,0,\dots$ forever on a blank tape:
1. In state $q_1$, on blank: write $1$, move right, go to $q_2$.
2. In state $q_2$, on blank: write $0$, move right, go to $q_1$.

Let blank be $s_0$, $0$ be $s_1$, and $1$ be $s_2$. Then the encoding becomes:

$$\texttt{DADDCCRDAA;DAADDCRDA;}$$

Finally, you can turn this string into a **single natural number** by replacing the seven symbols with digits, for instance:

$$A\mapsto 1,\ C\mapsto 2,\ D\mapsto 3,\ L\mapsto 4,\ R\mapsto 5,\ N\mapsto 6,\ ;\mapsto 7.$$

Under this replacement, the machine above corresponds to the description number

$$313322531173113325317.$$

There’s an analogous coding scheme for any chosen universal machine. In practice you rarely need to compute the number explicitly — the key point is that every Turing machine can be assigned some natural number code, even though many numbers won’t decode to any valid machine.

## Application to undecidability proofs

Description numbers show up all over undecidability arguments, including the standard proof that the **Halting Problem** cannot be decided. The basic reason is that once every Turing machine can be coded by a natural number, the collection of all Turing machines becomes **countable** (denumerable). But the collection of all partial functions $\mathbb{N}\rightharpoonup\mathbb{N}$ is **uncountable**, so there must be many functions that no Turing machine can compute.

Using an idea reminiscent of **Cantor’s diagonal method**, one can even point to a specific uncomputable task—namely, the halting problem. Let $U(e,x)$ denote the behavior of a universal Turing machine on description number $e$ and input $x$; if $e$ is not a valid code for a machine, define $U(e,x)=0$. Now assume, for contradiction, that there is an algorithm solving the halting problem: a machine $\mathrm{TEST}(e)$ that outputs $1$ if the machine coded by $e$ halts on **every** input, and outputs $0$ if there exists at least one input on which it runs forever.

From $U$ and $\mathrm{TEST}$, define a new machine $\delta(k)$ by

* if $\mathrm{TEST}(k)=1$, output $U(k,k)+1$;
* if $\mathrm{TEST}(k)=0$, output $0$.

This definition makes $\delta$ total (defined for every input), hence total recursive. Therefore $\delta$ itself has a description number; call it $e$. Feeding $e$ back into the universal machine, we get $\delta(k)=U(e,k)$, and in particular $\delta(e)=U(e,e)$. But if $\mathrm{TEST}(e)=1$, then by the way $\delta$ was defined we also have $\delta(e)=U(e,e)+1$, a contradiction. So such a $\mathrm{TEST}$ machine cannot exist, and the halting problem is undecidable.